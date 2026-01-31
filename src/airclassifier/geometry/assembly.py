"""
Cyclone assembly and system assembly modules.

This module provides:
- CycloneAssembly: Single cyclone with all components
- ClassificationSystemAssembly: Complete protein separation system with all Phase 1 components

Combined components include cyclones, zigzag classifiers, venturi eductors, and bag filters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import warp as wp

from .components import (
    CycloneBody, CycloneBodyParams,
    TangentialInlet, InletParams,
    VortexFinder, VortexFinderParams,
    DustOutlet, DustOutletParams,
    Overflow, OverflowParams,
)
from ..utils.constants import PI


@dataclass
class CycloneGeometryParams:
    """
    Complete cyclone geometry parameters.

    Standard cyclone design ratios (relative to cylinder diameter D):
    - Inlet width: 0.2-0.25 D
    - Inlet height: 0.5 D
    - Vortex finder diameter: 0.4-0.5 D
    - Vortex finder insertion: 0.5-0.6 D
    - Cylinder height: 1.5-2.0 D
    - Cone height: 2.5-3.0 D
    - Dust outlet diameter: 0.25-0.375 D
    """

    # Main dimensions
    cylinder_diameter: float    # [m] Main body diameter (D)
    cylinder_height: float      # [m] Height of cylindrical section
    cone_height: float          # [m] Height of conical section
    cone_tip_diameter: float    # [m] Diameter at bottom of cone

    # Inlet
    inlet_width: float          # [m] Tangential inlet width
    inlet_height: float         # [m] Tangential inlet height

    # Vortex finder
    vortex_finder_diameter: float   # [m] Gas outlet diameter
    vortex_finder_length: float     # [m] Insertion depth

    # Optional parameters with defaults
    inlet_length: float = 0.2   # [m] Inlet duct length

    # Dust outlet
    dust_outlet_diameter: float = None  # [m] If None, equals cone_tip_diameter
    dust_outlet_length: float = 0.1     # [m] Dust outlet pipe length

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 48

    def __post_init__(self):
        if self.dust_outlet_diameter is None:
            self.dust_outlet_diameter = self.cone_tip_diameter

    @classmethod
    def from_diameter(cls, D: float, **kwargs) -> "CycloneGeometryParams":
        """
        Create standard cyclone parameters based on cylinder diameter.

        Uses typical design ratios for high-efficiency cyclones.

        Args:
            D: Cylinder diameter [m]
            **kwargs: Override any default parameters

        Returns:
            CycloneGeometryParams instance
        """
        defaults = {
            "cylinder_diameter": D,
            "cylinder_height": 1.5 * D,
            "cone_height": 2.5 * D,
            "cone_tip_diameter": 0.375 * D,
            "inlet_width": 0.25 * D,
            "inlet_height": 0.5 * D,
            "inlet_length": 0.3 * D,
            "vortex_finder_diameter": 0.5 * D,
            "vortex_finder_length": 0.5 * D,
            "dust_outlet_diameter": 0.375 * D,
            "dust_outlet_length": 0.2 * D,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @property
    def total_height(self) -> float:
        """Total height from top to cone tip."""
        return self.cylinder_height + self.cone_height

    @property
    def aspect_ratio(self) -> float:
        """Height to diameter ratio."""
        return self.total_height / self.cylinder_diameter


class CycloneAssembly:
    """
    Complete cyclone air classifier assembly.

    Combines all components:
    - Cyclone body (cylinder + cone)
    - Tangential inlet
    - Vortex finder
    - Dust outlet
    - Overflow region

    Provides:
    - Unified mesh generation
    - Spatial queries (point inside/outside, closest point)
    - Boundary condition information
    """

    def __init__(self, params: CycloneGeometryParams, device: str = "cuda"):
        """
        Initialize complete cyclone assembly.

        Args:
            params: CycloneGeometryParams defining the cyclone
            device: Warp device ("cuda" or "cpu")
        """
        self.params = params
        self.device = device

        # Create all components
        self._create_components()

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._warp_mesh = None
        self._mesh_built = False

    def _create_components(self):
        """Create all cyclone components."""
        p = self.params

        # Cyclone body (cylinder + cone)
        self.body = CycloneBody(CycloneBodyParams(
            cylinder_diameter=p.cylinder_diameter,
            cylinder_height=p.cylinder_height,
            cone_height=p.cone_height,
            cone_tip_diameter=p.cone_tip_diameter,
            center=p.center,
            resolution_radial=p.resolution,
            resolution_axial_cylinder=p.resolution // 3,
            resolution_axial_cone=p.resolution // 2
        ))

        # Tangential inlet
        self.inlet = TangentialInlet(InletParams(
            width=p.inlet_width,
            height=p.inlet_height,
            length=p.inlet_length,
            cyclone_diameter=p.cylinder_diameter,
            inlet_top_offset=0.05,  # Slight offset from top
            cyclone_center=p.center,
            angular_position=0.0  # On +X side
        ))

        # Vortex finder
        self.vortex_finder = VortexFinder(VortexFinderParams(
            diameter=p.vortex_finder_diameter,
            length=p.vortex_finder_length,
            cyclone_center=p.center,
            protrusion_above=0.05,
            resolution_radial=p.resolution // 2,
            resolution_axial=p.resolution // 4
        ))

        # Dust outlet
        cone_bottom_y = p.center[1] - p.cylinder_height - p.cone_height
        self.dust_outlet = DustOutlet(DustOutletParams(
            diameter=p.dust_outlet_diameter,
            length=p.dust_outlet_length,
            cone_bottom_center=(p.center[0], cone_bottom_y, p.center[2]),
            resolution_radial=p.resolution // 2,
            resolution_axial=p.resolution // 6
        ))

        # Overflow region
        vf_top_y = p.center[1] + 0.05  # Matches vortex finder protrusion
        self.overflow = Overflow(OverflowParams(
            vortex_finder_diameter=p.vortex_finder_diameter,
            vortex_finder_top_y=vf_top_y,
            cyclone_center=p.center
        ))

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Body mesh
        body_verts, body_idx, _ = self.body.generate_mesh()
        all_vertices.append(body_verts)
        all_indices.append(body_idx + vertex_offset)
        vertex_offset += len(body_verts)

        # Vortex finder mesh
        vf_verts, vf_idx, _ = self.vortex_finder.generate_mesh()
        all_vertices.append(vf_verts)
        all_indices.append(vf_idx + vertex_offset)
        vertex_offset += len(vf_verts)

        # Dust outlet mesh
        do_verts, do_idx, _ = self.dust_outlet.generate_mesh()
        all_vertices.append(do_verts)
        all_indices.append(do_idx + vertex_offset)
        vertex_offset += len(do_verts)

        # Inlet mesh (optional - mainly for visualization)
        inlet_verts, inlet_idx, _ = self.inlet.generate_mesh()
        all_vertices.append(inlet_verts)
        all_indices.append(inlet_idx + vertex_offset)

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_warp_mesh(self) -> wp.Mesh:
        """
        Get Warp mesh for spatial queries.

        Returns:
            wp.Mesh object
        """
        if not self._mesh_built:
            self.build_mesh()

        if self._warp_mesh is None:
            points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
            indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)
            self._warp_mesh = wp.Mesh(points=points, indices=indices)

        return self._warp_mesh

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the cyclone.

        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        p = self.params

        # Calculate bounds
        r = p.cylinder_diameter / 2.0
        min_corner = np.array([
            p.center[0] - r - p.inlet_length,
            p.center[1] - p.cylinder_height - p.cone_height - p.dust_outlet_length,
            p.center[2] - r
        ])
        max_corner = np.array([
            p.center[0] + r + p.inlet_length,
            p.center[1] + 0.1,  # Vortex finder protrusion
            p.center[2] + r
        ])

        return min_corner, max_corner

    def get_inlet_conditions(self) -> Dict[str, Any]:
        """
        Get inlet boundary condition parameters.

        Returns:
            Dictionary with inlet parameters
        """
        return {
            "position": self.inlet.entry_point,
            "direction": self.inlet.get_inlet_velocity_direction(),
            "width": self.params.inlet_width,
            "height": self.params.inlet_height,
            "area": self.params.inlet_width * self.params.inlet_height,
        }

    def get_outlet_conditions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get outlet boundary condition parameters.

        Returns:
            Dictionary with overflow and underflow parameters
        """
        return {
            "overflow": self.overflow.get_exit_boundary_condition(),
            "underflow": {
                "center": self.dust_outlet.params.cone_bottom_center,
                "radius": self.dust_outlet.params.inner_radius,
                "plane": self.dust_outlet.get_outlet_plane(),
            }
        }

    def classify_position(self, point: np.ndarray) -> str:
        """
        Classify which region of the cyclone a point is in.

        Args:
            point: 3D position to classify

        Returns:
            Region name: "outside", "cylinder", "cone", "vortex_finder",
                        "dust_outlet", "overflow"
        """
        p = self.params

        # Check vortex finder first
        if self.vortex_finder.is_inside_tube(point):
            if point[1] > self.overflow.params.vortex_finder_top_y:
                return "overflow"
            return "vortex_finder"

        # Check dust outlet
        if self.dust_outlet.is_particle_collected(point):
            return "dust_outlet"

        # Check main body
        section, radius = self.body.get_position_at_height(point[1])

        # Radial distance
        dx = point[0] - p.center[0]
        dz = point[2] - p.center[2]
        r = np.sqrt(dx * dx + dz * dz)

        if r > radius:
            return "outside"

        return section

    def print_summary(self):
        """Print summary of cyclone geometry."""
        p = self.params
        print("=" * 50)
        print("Cyclone Air Classifier Geometry Summary")
        print("=" * 50)
        print(f"Cylinder diameter:     {p.cylinder_diameter * 1000:.1f} mm")
        print(f"Cylinder height:       {p.cylinder_height * 1000:.1f} mm")
        print(f"Cone height:           {p.cone_height * 1000:.1f} mm")
        print(f"Cone tip diameter:     {p.cone_tip_diameter * 1000:.1f} mm")
        print(f"Total height:          {p.total_height * 1000:.1f} mm")
        print(f"Aspect ratio:          {p.aspect_ratio:.2f}")
        print("-" * 50)
        print(f"Inlet width:           {p.inlet_width * 1000:.1f} mm")
        print(f"Inlet height:          {p.inlet_height * 1000:.1f} mm")
        print(f"Inlet area:            {p.inlet_width * p.inlet_height * 1e6:.1f} mm²")
        print("-" * 50)
        print(f"Vortex finder dia:     {p.vortex_finder_diameter * 1000:.1f} mm")
        print(f"Vortex finder depth:   {p.vortex_finder_length * 1000:.1f} mm")
        print("-" * 50)
        print(f"Dust outlet diameter:  {p.dust_outlet_diameter * 1000:.1f} mm")
        print("=" * 50)


def create_standard_cyclone(
    diameter: float,
    device: str = "cuda"
) -> CycloneAssembly:
    """
    Create a standard high-efficiency cyclone with typical proportions.

    Args:
        diameter: Main cylinder diameter [m]
        device: Warp device

    Returns:
        CycloneAssembly instance
    """
    params = CycloneGeometryParams.from_diameter(diameter)
    return CycloneAssembly(params, device=device)


# =============================================================================
# CLASSIFICATION SYSTEM ASSEMBLY (Phase 1 Components)
# =============================================================================

@dataclass
class ClassificationSystemParams:
    """
    Parameters for complete classification system.

    Combines all Phase 1 components into a protein separation system.
    """

    # Zigzag classifier parameters
    zigzag_channel_width: float = 0.15      # [m]
    zigzag_num_stages: int = 5
    zigzag_channel_depth: float = 0.30      # [m]

    # Venturi eductor parameters
    venturi_inlet_diameter: float = 0.10    # [m]
    venturi_throat_ratio: float = 0.5

    # Multi-cyclone parameters
    primary_cyclone_diameter: float = 0.40   # [m]
    secondary_cyclone_diameter: float = 0.25 # [m]
    tertiary_cyclone_diameter: float = 0.15  # [m]

    # Bag filter parameters
    bag_filter_flow_rate: float = 1.0       # [m3/s]
    bag_filter_air_to_cloth: float = 2.0    # [m3/min/m2]

    # Layout parameters
    component_spacing: float = 0.5          # [m] Spacing between components
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 24


class ClassificationSystemAssembly:
    """
    Complete protein separation/classification system assembly.

    Combines all Phase 1 components:
    - Venturi Eductor: Particle entrainment
    - Zigzag Classifier: Primary separation
    - Multi-Cyclone System: Staged collection
    - Bag Filter: Fine particle collection

    Process flow:
    Feed -> Venturi -> Zigzag -> Cyclones -> Bag Filter -> Clean Air

    Coordinate system:
    - Origin at center of system
    - X-axis: Main flow direction
    - Y-axis: Vertical (up)
    - Z-axis: Depth
    """

    def __init__(self, params: ClassificationSystemParams = None, device: str = "cpu"):
        """
        Initialize classification system assembly.

        Args:
            params: ClassificationSystemParams (uses defaults if None)
            device: Warp device for mesh operations
        """
        self.params = params or ClassificationSystemParams()
        self.device = device

        # Create components
        self._create_components()

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._mesh_built = False

    def _create_components(self):
        """Create all system components with proper positioning."""
        # Lazy imports to avoid circular dependency
        from .components import (
            create_standard_zigzag_classifier,
            create_standard_venturi_eductor,
            MultiCycloneSystem, MultiCycloneParams, CycloneStageParams,
            create_standard_bag_filter,
        )

        p = self.params
        spacing = p.component_spacing

        # Track X position along flow path
        x_pos = p.center[0]

        # 1. Venturi Eductor (first in flow path)
        self.venturi = create_standard_venturi_eductor(
            inlet_diameter=p.venturi_inlet_diameter,
            throat_ratio=p.venturi_throat_ratio
        )
        # Position venturi at start
        self._venturi_position = (x_pos, p.center[1], p.center[2])
        x_pos += self.venturi.params.total_length + spacing

        # 2. Zigzag Classifier
        self.zigzag = create_standard_zigzag_classifier(
            channel_width=p.zigzag_channel_width,
            num_stages=p.zigzag_num_stages,
            channel_depth=p.zigzag_channel_depth
        )
        self._zigzag_position = (x_pos, p.center[1], p.center[2])
        x_pos += self.zigzag.params.total_width + spacing

        # 3. Multi-Cyclone System
        cyclone_stages = [
            CycloneStageParams(
                name="primary",
                diameter=p.primary_cyclone_diameter,
                design_d50=40e-6,
            ),
            CycloneStageParams(
                name="secondary",
                diameter=p.secondary_cyclone_diameter,
                design_d50=20e-6,
            ),
            CycloneStageParams(
                name="tertiary",
                diameter=p.tertiary_cyclone_diameter,
                design_d50=10e-6,
            ),
        ]
        cyclone_params = MultiCycloneParams(
            stages=cyclone_stages,
            arrangement="series",
            spacing=spacing / 2,
            center=(x_pos, p.center[1], p.center[2]),
            resolution=p.resolution,
        )
        self.multi_cyclone = MultiCycloneSystem(cyclone_params)
        self._cyclone_position = (x_pos, p.center[1], p.center[2])

        # Calculate cyclone extent
        min_b, max_b = self.multi_cyclone.get_system_bounds()
        cyclone_width = max_b[0] - min_b[0]
        x_pos += cyclone_width + spacing

        # 4. Bag Filter
        self.bag_filter = create_standard_bag_filter(
            flow_rate_m3s=p.bag_filter_flow_rate,
            air_to_cloth=p.bag_filter_air_to_cloth
        )
        self._bag_filter_position = (x_pos, p.center[1], p.center[2])

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Helper to add component mesh with position offset
        def add_component_mesh(component, position):
            nonlocal vertex_offset
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add each component
        add_component_mesh(self.venturi, self._venturi_position)
        add_component_mesh(self.zigzag, self._zigzag_position)

        # Multi-cyclone has its own positioning
        mc_verts, mc_idx, _ = self.multi_cyclone.generate_mesh()
        all_vertices.append(mc_verts)
        all_indices.append(mc_idx + vertex_offset)
        vertex_offset += len(mc_verts)

        add_component_mesh(self.bag_filter, self._bag_filter_position)

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the entire system.

        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        if not self._mesh_built:
            self.build_mesh()

        min_corner = self._combined_vertices.min(axis=0)
        max_corner = self._combined_vertices.max(axis=0)

        return min_corner, max_corner

    def get_system_extent(self) -> np.ndarray:
        """
        Get system extent (dimensions) in each axis.

        Returns:
            Array of [width, height, depth]
        """
        min_c, max_c = self.get_bounds()
        return max_c - min_c

    def get_component(self, name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            name: Component name ('venturi', 'zigzag', 'multi_cyclone', 'bag_filter')

        Returns:
            Component instance
        """
        components = {
            'venturi': self.venturi,
            'zigzag': self.zigzag,
            'multi_cyclone': self.multi_cyclone,
            'bag_filter': self.bag_filter,
        }
        if name not in components:
            raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
        return components[name]

    def get_component_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get positions of all components.

        Returns:
            Dictionary of component names to positions
        """
        return {
            'venturi': self._venturi_position,
            'zigzag': self._zigzag_position,
            'multi_cyclone': self._cyclone_position,
            'bag_filter': self._bag_filter_position,
        }

    def to_warp_mesh(self) -> wp.Mesh:
        """
        Create a Warp mesh from the system geometry.

        Returns:
            wp.Mesh object
        """
        if not self._mesh_built:
            self.build_mesh()

        points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
        indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)

        return wp.Mesh(points=points, indices=indices)

    def print_summary(self):
        """Print summary of the classification system."""
        p = self.params

        print("=" * 60)
        print("Classification System Assembly Summary")
        print("=" * 60)

        print("\n1. VENTURI EDUCTOR")
        print(f"   Inlet diameter:  {p.venturi_inlet_diameter * 1000:.0f} mm")
        print(f"   Throat ratio:    {p.venturi_throat_ratio:.2f}")
        print(f"   Total length:    {self.venturi.params.total_length * 1000:.0f} mm")

        print("\n2. ZIGZAG CLASSIFIER")
        print(f"   Channel width:   {p.zigzag_channel_width * 1000:.0f} mm")
        print(f"   Number of stages: {p.zigzag_num_stages}")
        print(f"   Total height:    {self.zigzag.params.total_height * 1000:.0f} mm")

        print("\n3. MULTI-CYCLONE SYSTEM")
        for info in self.multi_cyclone.get_stage_info():
            print(f"   {info['name'].title():12s} D={info['diameter']:.0f}mm, d50={info['design_d50']:.0f}um")

        print("\n4. BAG FILTER")
        print(f"   Number of bags:  {self.bag_filter.params.num_bags}")
        print(f"   Filter area:     {self.bag_filter.params.total_filter_area:.1f} m2")
        print(f"   A/C ratio:       {self.bag_filter.params.get_air_to_cloth(p.bag_filter_flow_rate):.2f} m3/min/m2")

        print("-" * 60)
        extent = self.get_system_extent()
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")

        if self._mesh_built:
            n_verts = len(self._combined_vertices)
            n_tris = len(self._combined_indices) // 3
            print(f"Total mesh:    {n_verts} vertices, {n_tris} triangles")
        print("=" * 60)

    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_vertices

    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_indices


def create_standard_classification_system(device: str = "cpu") -> ClassificationSystemAssembly:
    """
    Create a standard classification system with default parameters.

    Args:
        device: Warp device

    Returns:
        ClassificationSystemAssembly instance
    """
    return ClassificationSystemAssembly(device=device)


def create_protein_separation_system(
    throughput_kg_h: float = 100,
    device: str = "cpu"
) -> ClassificationSystemAssembly:
    """
    Create a protein separation system sized for given throughput.

    Args:
        throughput_kg_h: Design throughput [kg/h]
        device: Warp device

    Returns:
        ClassificationSystemAssembly configured for protein separation
    """
    # Scale parameters based on throughput
    # Typical air/solids ratio: 2-3 m3/kg
    # Assume bulk density ~500 kg/m3 for legume flour
    air_flow_m3s = throughput_kg_h * 2.5 / 3600  # m3/s

    # Scale component sizes
    scale = (throughput_kg_h / 100) ** 0.5  # Square root scaling

    params = ClassificationSystemParams(
        zigzag_channel_width=0.15 * scale,
        zigzag_num_stages=5,
        zigzag_channel_depth=0.30 * scale,
        venturi_inlet_diameter=0.10 * scale,
        venturi_throat_ratio=0.5,
        primary_cyclone_diameter=0.40 * scale,
        secondary_cyclone_diameter=0.25 * scale,
        tertiary_cyclone_diameter=0.15 * scale,
        bag_filter_flow_rate=max(air_flow_m3s, 0.5),
        bag_filter_air_to_cloth=2.0,
    )

    return ClassificationSystemAssembly(params, device=device)
