"""
Cyclone assembly - combines all components into a complete cyclone.

This module provides the CycloneAssembly class which creates and manages
all cyclone components and provides unified access to geometry, meshes,
and spatial queries.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
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
