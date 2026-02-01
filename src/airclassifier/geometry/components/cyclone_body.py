"""
Cyclone body component - cylindrical section + conical section.

The cyclone body consists of:
1. Upper cylindrical section (barrel)
2. Lower conical section (cone)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import Cylinder, CylinderParams, Cone, ConeParams
from ...utils.constants import PI


@dataclass
class CycloneBodyParams:
    """Parameters for the cyclone body (cylinder + cone)."""

    # Cylindrical section
    cylinder_diameter: float    # [m] Diameter of cylindrical section
    cylinder_height: float      # [m] Height of cylindrical section

    # Conical section
    cone_height: float          # [m] Height of conical section
    cone_tip_diameter: float    # [m] Diameter at bottom of cone

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Top center of cylinder

    # Mesh resolution
    resolution_radial: int = 48
    resolution_axial_cylinder: int = 16
    resolution_axial_cone: int = 24

    @property
    def cylinder_radius(self) -> float:
        """Radius of cylindrical section."""
        return self.cylinder_diameter / 2.0

    @property
    def cone_top_radius(self) -> float:
        """Top radius of cone (equals cylinder radius)."""
        return self.cylinder_diameter / 2.0

    @property
    def cone_bottom_radius(self) -> float:
        """Bottom radius of cone."""
        return self.cone_tip_diameter / 2.0

    @property
    def total_height(self) -> float:
        """Total height of cyclone body."""
        return self.cylinder_height + self.cone_height

    @property
    def cone_half_angle(self) -> float:
        """Half-angle of the cone in radians."""
        dr = self.cone_top_radius - self.cone_bottom_radius
        return np.arctan2(dr, self.cone_height)

    @property
    def volume(self) -> float:
        """Total internal volume of cyclone body."""
        # Cylinder volume
        v_cyl = PI * self.cylinder_radius ** 2 * self.cylinder_height

        # Cone (frustum) volume
        r1, r2, h = self.cone_top_radius, self.cone_bottom_radius, self.cone_height
        v_cone = (PI * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

        return v_cyl + v_cone


class CycloneBody:
    """
    Cyclone body consisting of cylindrical barrel and conical section.

    Coordinate system:
    - Origin at top center of cylinder
    - Y-axis pointing downward (into the cyclone)
    - X-Z plane is horizontal
    """

    def __init__(self, params: CycloneBodyParams):
        """
        Initialize cyclone body.

        Args:
            params: CycloneBodyParams defining the geometry
        """
        self.params = params

        # Create cylinder component
        # The Cylinder primitive extends UPWARD from its center (base),
        # but we want the cylinder TOP at params.center (top of cyclone).
        # So the cylinder base must be at center[1] - cylinder_height.
        cylinder_base_center = (
            params.center[0],
            params.center[1] - params.cylinder_height,
            params.center[2]
        )
        self._cylinder = Cylinder(CylinderParams(
            radius=params.cylinder_radius,
            height=params.cylinder_height,
            center=cylinder_base_center,
            axis="y",
            resolution_radial=params.resolution_radial,
            resolution_axial=params.resolution_axial_cylinder
        ))

        # Create cone component (positioned below cylinder)
        # The Cone primitive has its TOP at center and extends downward.
        # Cone top connects to cylinder bottom at center[1] - cylinder_height.
        cone_center = (
            params.center[0],
            params.center[1] - params.cylinder_height,
            params.center[2]
        )
        self._cone = Cone(ConeParams(
            top_radius=params.cone_top_radius,
            bottom_radius=params.cone_bottom_radius,
            height=params.cone_height,
            center=cone_center,
            axis="y",
            resolution_radial=params.resolution_radial,
            resolution_axial=params.resolution_axial_cone
        ))

        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self, include_caps: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate combined mesh for the cyclone body.

        Args:
            include_caps: Whether to include end caps (usually False for internal flow)

        Returns:
            Tuple of (vertices, indices, normals)
        """
        # Generate individual meshes
        cyl_verts, cyl_idx, cyl_norms = self._cylinder.generate_mesh()
        cone_verts, cone_idx, cone_norms = self._cone.generate_mesh()

        # Combine meshes
        # Offset cone indices by number of cylinder vertices
        cone_idx_offset = cone_idx + len(cyl_verts)

        self._vertices = np.vstack([cyl_verts, cone_verts])
        self._indices = np.concatenate([cyl_idx, cone_idx_offset])
        self._normals = np.vstack([cyl_norms, cone_norms])

        return self._vertices, self._indices, self._normals

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """
        Create a Warp mesh from the cyclone body geometry.

        Args:
            device: Device to create mesh on

        Returns:
            wp.Mesh object
        """
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

    def get_position_at_height(self, y: float) -> Tuple[str, float]:
        """
        Determine which section a given Y-coordinate is in and return the local radius.

        Args:
            y: Y-coordinate (relative to top of cylinder)

        Returns:
            Tuple of (section_name, radius_at_height)
        """
        p = self.params
        y_rel = p.center[1] - y  # Distance from top

        if y_rel < 0:
            return ("above", p.cylinder_radius)
        elif y_rel <= p.cylinder_height:
            return ("cylinder", p.cylinder_radius)
        elif y_rel <= p.total_height:
            # In cone section - interpolate radius
            cone_y = y_rel - p.cylinder_height
            t = cone_y / p.cone_height
            r = p.cone_top_radius * (1.0 - t) + p.cone_bottom_radius * t
            return ("cone", r)
        else:
            return ("below", p.cone_bottom_radius)

    @property
    def cylinder(self) -> Cylinder:
        """Get the cylinder component."""
        return self._cylinder

    @property
    def cone(self) -> Cone:
        """Get the cone component."""
        return self._cone

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
# WARP SDF FUNCTION FOR CYCLONE BODY
# =============================================================================

@wp.func
def cyclone_body_sdf(
    p: wp.vec3,
    center: wp.vec3,
    cylinder_radius: float,
    cylinder_height: float,
    cone_height: float,
    cone_bottom_radius: float
) -> float:
    """
    Signed distance function for cyclone body (cylinder + cone).

    Args:
        p: Query point
        center: Top center of cylinder
        cylinder_radius: Radius of cylindrical section
        cylinder_height: Height of cylindrical section
        cone_height: Height of conical section
        cone_bottom_radius: Radius at bottom of cone

    Returns:
        Signed distance (negative inside, positive outside)
    """
    # Local coordinates (y positive going down into cyclone)
    local_x = p[0] - center[0]
    local_y = center[1] - p[1]  # Flip so positive is downward
    local_z = p[2] - center[2]

    # Radial distance from axis
    r = wp.sqrt(local_x * local_x + local_z * local_z)

    # Determine which section we're in
    total_height = cylinder_height + cone_height

    if local_y < 0.0:
        # Above cylinder
        if r <= cylinder_radius:
            return -local_y  # Distance to top cap
        else:
            # Outside and above
            return wp.sqrt((r - cylinder_radius) ** 2.0 + local_y ** 2.0)

    elif local_y <= cylinder_height:
        # In cylinder section
        return r - cylinder_radius

    elif local_y <= total_height:
        # In cone section
        cone_y = local_y - cylinder_height
        t = cone_y / cone_height
        r_at_height = cylinder_radius * (1.0 - t) + cone_bottom_radius * t

        # Distance to slant surface
        dr = cylinder_radius - cone_bottom_radius
        slant = wp.sqrt(cone_height * cone_height + dr * dr)
        dist_to_slant = (r - r_at_height) * cone_height / slant

        return dist_to_slant

    else:
        # Below cone
        if r <= cone_bottom_radius:
            return local_y - total_height  # Distance to bottom cap
        else:
            # Outside and below
            return wp.sqrt((r - cone_bottom_radius) ** 2.0 + (local_y - total_height) ** 2.0)
