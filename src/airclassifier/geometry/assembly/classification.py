"""
Classification system assembly module.

Provides complete classification system assembly combining Phase 1 components:
- Venturi Eductor
- Zigzag Classifier
- Multi-Cyclone System
- Bag Filter
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import warp as wp


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
        from ..components import (
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
