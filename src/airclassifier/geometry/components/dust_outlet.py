"""
Dust outlet (underflow) component for cyclone air classifier.

The dust outlet is located at the bottom of the cone section where
coarse/heavy particles are collected after being separated from the
gas stream.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import Tube, TubeParams
from ...utils.constants import PI


@dataclass
class DustOutletParams:
    """Parameters for the dust outlet."""

    # Outlet dimensions
    diameter: float         # [m] Inner diameter of outlet
    length: float           # [m] Length of outlet pipe
    wall_thickness: float = 0.005  # [m] Wall thickness

    # Position (connected to bottom of cone)
    cone_bottom_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Optional: apex cone at transition
    include_apex_cone: bool = False
    apex_cone_height: float = 0.05  # [m] Height of transition cone

    # Flange options
    flanged: bool = True
    flange_width: float = 0.02      # [m] Width beyond pipe diameter
    flange_thickness: float = 0.008  # [m] Thickness of flange

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 8

    @property
    def inner_radius(self) -> float:
        """Inner radius of outlet."""
        return self.diameter / 2.0

    @property
    def outer_radius(self) -> float:
        """Outer radius including wall."""
        return self.inner_radius + self.wall_thickness

    @property
    def cross_sectional_area(self) -> float:
        """Flow cross-sectional area."""
        return PI * self.inner_radius ** 2


class DustOutlet:
    """
    Dust outlet (underflow) for collecting separated coarse particles.

    The dust outlet is positioned at the apex of the cone section.
    Particles that reach this region have been successfully separated
    from the gas stream by centrifugal action.
    """

    def __init__(self, params: DustOutletParams):
        """
        Initialize dust outlet.

        Args:
            params: DustOutletParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Create tube component extending downward from cone bottom
        self._tube = Tube(TubeParams(
            outer_radius=params.outer_radius,
            inner_radius=params.inner_radius,
            length=params.length,
            center=params.cone_bottom_center,
            axis="y",
            direction=-1,  # Extends downward
            resolution_radial=params.resolution_radial,
            resolution_axial=params.resolution_axial
        ))

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the dust outlet.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        verts, idx, norms = self._tube.generate_mesh()
        
        # Add flange at outlet end if enabled
        p = self.params
        if p.flanged:
            flange_verts, flange_idx, flange_norms = self._generate_outlet_flange()
            # Offset indices
            flange_idx = flange_idx + len(verts)
            # Combine
            verts = np.vstack([verts, flange_verts])
            idx = np.concatenate([idx, flange_idx])
            norms = np.vstack([norms, flange_norms])
        
        self._vertices = verts.astype(np.float32)
        self._indices = idx.astype(np.int32)
        self._normals = norms.astype(np.float32)
        return self._vertices, self._indices, self._normals
    
    def _generate_outlet_flange(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate flange ring at the outlet (bottom) end."""
        p = self.params
        n_seg = p.resolution_radial
        
        # Flange position: at outlet end (below cone bottom by length)
        outlet_y = p.cone_bottom_center[1] - p.length
        cx, cz = p.cone_bottom_center[0], p.cone_bottom_center[2]
        
        inner_r = p.outer_radius  # Inner edge of flange = outer edge of pipe
        outer_r = inner_r + p.flange_width
        ft = p.flange_thickness
        
        vertices = []
        indices = []
        normals = []
        
        # Flange extends below the outlet
        y_top = outlet_y
        y_bottom = outlet_y - ft
        
        # Outer cylinder of flange
        base_idx = 0
        for i in range(n_seg):
            theta = 2 * PI * i / n_seg
            x = cx + outer_r * np.cos(theta)
            z = cz + outer_r * np.sin(theta)
            
            # Top vertex
            vertices.append([x, y_top, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
            # Bottom vertex
            vertices.append([x, y_bottom, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
        
        # Triangles for outer cylinder
        for i in range(n_seg):
            i0 = base_idx + i * 2
            i1 = base_idx + i * 2 + 1
            i2 = base_idx + ((i + 1) % n_seg) * 2
            i3 = base_idx + ((i + 1) % n_seg) * 2 + 1
            indices.extend([i0, i1, i2])
            indices.extend([i2, i1, i3])
        
        # Top annular face (facing up)
        top_base = len(vertices)
        for i in range(n_seg):
            theta = 2 * PI * i / n_seg
            # Inner ring
            x_in = cx + inner_r * np.cos(theta)
            z_in = cz + inner_r * np.sin(theta)
            vertices.append([x_in, y_top, z_in])
            normals.append([0, 1, 0])
        for i in range(n_seg):
            theta = 2 * PI * i / n_seg
            # Outer ring
            x_out = cx + outer_r * np.cos(theta)
            z_out = cz + outer_r * np.sin(theta)
            vertices.append([x_out, y_top, z_out])
            normals.append([0, 1, 0])
        
        for i in range(n_seg):
            i0 = top_base + i
            i1 = top_base + (i + 1) % n_seg
            i2 = top_base + n_seg + i
            i3 = top_base + n_seg + (i + 1) % n_seg
            indices.extend([i0, i2, i1])
            indices.extend([i1, i2, i3])
        
        # Bottom annular face (facing down)
        bot_base = len(vertices)
        for i in range(n_seg):
            theta = 2 * PI * i / n_seg
            x_in = cx + inner_r * np.cos(theta)
            z_in = cz + inner_r * np.sin(theta)
            vertices.append([x_in, y_bottom, z_in])
            normals.append([0, -1, 0])
        for i in range(n_seg):
            theta = 2 * PI * i / n_seg
            x_out = cx + outer_r * np.cos(theta)
            z_out = cz + outer_r * np.sin(theta)
            vertices.append([x_out, y_bottom, z_out])
            normals.append([0, -1, 0])
        
        for i in range(n_seg):
            i0 = bot_base + i
            i1 = bot_base + (i + 1) % n_seg
            i2 = bot_base + n_seg + i
            i3 = bot_base + n_seg + (i + 1) % n_seg
            indices.extend([i0, i1, i2])
            indices.extend([i1, i3, i2])
        
        return np.array(vertices), np.array(indices), np.array(normals)

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the dust outlet geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

    def is_particle_collected(self, point: np.ndarray) -> bool:
        """
        Check if a particle at the given position should be considered collected.

        Args:
            point: 3D position of particle

        Returns:
            True if particle has entered the dust outlet region
        """
        p = self.params

        # Check radial distance from outlet axis
        dx = point[0] - p.cone_bottom_center[0]
        dz = point[2] - p.cone_bottom_center[2]
        r = np.sqrt(dx * dx + dz * dz)

        # Must be within outlet radius
        if r > p.inner_radius:
            return False

        # Must be at or below the cone bottom
        return point[1] <= p.cone_bottom_center[1]

    def get_outlet_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the plane defining the outlet (bottom of dust outlet pipe).

        Returns:
            Tuple of (center_point, normal_vector)
        """
        p = self.params
        center = np.array([
            p.cone_bottom_center[0],
            p.cone_bottom_center[1] - p.length,
            p.cone_bottom_center[2]
        ])
        normal = np.array([0.0, -1.0, 0.0])  # Pointing down (outward)

        return center, normal

    def get_collection_plane_y(self) -> float:
        """
        Get Y-coordinate of the collection plane.

        Particles below this Y are considered collected.
        """
        return self.params.cone_bottom_center[1]

    @property
    def tube(self) -> Tube:
        """Get the tube component."""
        return self._tube

    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices

    @property
    def indices(self) -> np.ndarray:
        """Get mesh triangle indices."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices


# =============================================================================
# WARP KERNEL FOR PARTICLE COLLECTION CHECK
# =============================================================================

@wp.func
def is_in_dust_outlet(
    p: wp.vec3,
    outlet_center: wp.vec3,
    outlet_radius: float
) -> bool:
    """
    Check if a particle position is in the dust outlet region.

    Args:
        p: Particle position
        outlet_center: Center of dust outlet (at cone bottom)
        outlet_radius: Inner radius of dust outlet

    Returns:
        True if particle is in the collection region
    """
    # Check if below outlet plane
    if p[1] > outlet_center[1]:
        return False

    # Check radial distance
    dx = p[0] - outlet_center[0]
    dz = p[2] - outlet_center[2]
    r_sq = dx * dx + dz * dz

    return r_sq <= outlet_radius * outlet_radius
