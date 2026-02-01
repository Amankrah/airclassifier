"""
Tangential inlet component for cyclone air classifier.

The inlet introduces air and particles tangentially into the cyclone,
creating the swirling motion essential for particle separation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import warp as wp

from ..primitives import RectangularDuct, RectangularDuctParams
from ...utils.constants import PI


@dataclass
class InletParams:
    """Parameters for the tangential inlet."""

    # Inlet dimensions
    width: float            # [m] Width of inlet (tangential direction)
    height: float           # [m] Height of inlet (axial direction)
    length: float           # [m] Length of inlet duct

    # Position relative to cyclone
    cyclone_diameter: float # [m] Diameter of cyclone cylinder
    inlet_top_offset: float # [m] Distance from top of cyclone to top of inlet

    # Cyclone center position
    cyclone_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Entry angle (0 = purely tangential, positive = downward spiral)
    entry_angle: float = 0.0  # [radians]

    # Angular position of inlet (0 = +X direction)
    angular_position: float = 0.0  # [radians]

    @property
    def cross_sectional_area(self) -> float:
        """Inlet cross-sectional area."""
        return self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for flow calculations."""
        return 4.0 * self.cross_sectional_area / (2.0 * (self.width + self.height))

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (height/width)."""
        return self.height / self.width

    @property
    def area(self) -> float:
        """Inlet area (alias for cross_sectional_area)."""
        return self.cross_sectional_area


class TangentialInlet:
    """
    Tangential inlet for introducing air and particles into the cyclone.

    The inlet is positioned tangentially to the cyclone body, typically
    at the top of the cylindrical section.
    """

    def __init__(self, params: InletParams):
        """
        Initialize tangential inlet.

        Args:
            params: InletParams defining the inlet geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Calculate inlet position and orientation
        self._calculate_position()

    def _calculate_position(self):
        """Calculate the inlet position and direction vectors."""
        p = self.params
        r = p.cyclone_diameter / 2.0

        # Angular position
        theta = p.angular_position

        # Inlet center position (on cyclone surface)
        self.surface_point = np.array([
            p.cyclone_center[0] + r * np.cos(theta),
            p.cyclone_center[1] - p.inlet_top_offset - p.height / 2.0,
            p.cyclone_center[2] + r * np.sin(theta)
        ])

        # Tangent direction (perpendicular to radius, in XZ plane)
        # Positive tangent = counterclockwise when viewed from above
        self.tangent = np.array([-np.sin(theta), 0.0, np.cos(theta)])

        # Radial direction (outward from cyclone axis)
        self.radial = np.array([np.cos(theta), 0.0, np.sin(theta)])

        # Inlet direction (pointing inward, with possible downward angle)
        inlet_horizontal = -self.radial
        inlet_vertical = np.array([0.0, -np.sin(p.entry_angle), 0.0])
        self.inlet_direction = inlet_horizontal * np.cos(p.entry_angle) + inlet_vertical
        self.inlet_direction = self.inlet_direction / np.linalg.norm(self.inlet_direction)

        # Inlet start point (outer end of inlet duct)
        self.inlet_start = self.surface_point - self.inlet_direction * p.length

    def generate_mesh(self, num_length_segments: int = 8, 
                       num_curve_segments: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the inlet duct with curved saddle-cut end.
        
        The inner end of the duct curves to match the cylinder surface,
        like a real welded pipe connection (fish-mouth or saddle cut).

        Args:
            num_length_segments: Segments along duct length
            num_curve_segments: Segments for the curved saddle profile
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        all_vertices = []
        all_indices = []
        all_normals = []

        # Get orthonormal basis for the duct cross-section
        forward = self.inlet_direction
        up = np.array([0.0, 1.0, 0.0])

        # Handle case where inlet direction is nearly vertical
        if abs(np.dot(forward, up)) > 0.9:
            up = np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Store for later use
        self._right = right
        self._up = up

        # Half dimensions
        hw = p.width / 2.0
        hh = p.height / 2.0
        
        # Cyclone radius
        r_cyl = p.cyclone_diameter / 2.0
        
        # Cyclone center (just X and Z, Y handled separately)
        cx = p.cyclone_center[0]
        cz = p.cyclone_center[2]
        
        # Generate cross-sections along the duct length
        # Each cross-section has vertices around the perimeter
        n_perimeter = 2 * (num_curve_segments + 1) + 2  # Top + bottom curved edges + sides
        
        def get_saddle_point(local_w: float, local_h: float, t: float) -> np.ndarray:
            """
            Get a point on the duct, with saddle curve at the cylinder end.
            
            Args:
                local_w: Position across width (-hw to +hw)
                local_h: Position across height (-hh to +hh) 
                t: Position along length (0 = outer, 1 = cylinder surface)
            """
            # Base position along duct (linear interpolation)
            base_pos = self.inlet_start + t * (self.surface_point - self.inlet_start)
            
            # Add the width/height offset
            pos = base_pos + right * local_w + up * local_h
            
            # At t=1 (cylinder surface), curve the position to match cylinder
            if t > 0.5:
                # Blend factor for the saddle curve (0 at t=0.5, 1 at t=1)
                blend = (t - 0.5) * 2.0
                
                # Calculate how much to push the point to match cylinder surface
                # The point needs to be at radius r_cyl from the cylinder axis
                # Project point onto XZ plane relative to cyclone center
                px = pos[0] - cx
                pz = pos[2] - cz
                current_r = np.sqrt(px * px + pz * pz)
                
                if current_r > 0.01:  # Avoid division by zero
                    # Target radius on cylinder surface
                    target_r = r_cyl
                    
                    # How much we need to move radially
                    radial_adjustment = (target_r - current_r) * blend
                    
                    # Unit radial direction
                    radial_dir = np.array([px / current_r, 0.0, pz / current_r])
                    
                    # Apply adjustment
                    pos = pos + radial_dir * radial_adjustment
            
            return pos
        
        def get_cross_section_vertices(t: float) -> list:
            """Get vertices for a cross-section at position t along duct."""
            verts = []
            
            # Top edge (curved at cylinder end)
            for i in range(num_curve_segments + 1):
                w = -hw + (2 * hw) * i / num_curve_segments
                verts.append(get_saddle_point(w, hh, t))
            
            # Right edge (straight)
            verts.append(get_saddle_point(hw, -hh, t))
            
            # Bottom edge (curved at cylinder end, reversed direction)
            for i in range(num_curve_segments + 1):
                w = hw - (2 * hw) * i / num_curve_segments
                verts.append(get_saddle_point(w, -hh, t))
            
            # Left edge (straight)  
            verts.append(get_saddle_point(-hw, hh, t))
            
            return verts
        
        # Generate vertices for each cross-section along the length
        cross_sections = []
        for i in range(num_length_segments + 1):
            t = i / num_length_segments
            cs_verts = get_cross_section_vertices(t)
            cross_sections.append(cs_verts)
            all_vertices.extend(cs_verts)
        
        # Calculate normals for each vertex (pointing outward)
        for i in range(num_length_segments + 1):
            t = i / num_length_segments
            for j in range(len(cross_sections[0])):
                # Approximate normal by looking at adjacent vertices
                v = cross_sections[i][j]
                
                # Determine if this is top, bottom, left or right edge
                # and set normal accordingly
                if j <= num_curve_segments:  # Top edge
                    normal = up.tolist()
                elif j == num_curve_segments + 1:  # Right edge
                    normal = right.tolist()
                elif j <= 2 * num_curve_segments + 2:  # Bottom edge
                    normal = (-up).tolist()
                else:  # Left edge
                    normal = (-right).tolist()
                
                all_normals.append(normal)
        
        n_verts_per_section = len(cross_sections[0])
        
        # Generate triangles connecting adjacent cross-sections (side walls)
        for i in range(num_length_segments):
            base_curr = i * n_verts_per_section
            base_next = (i + 1) * n_verts_per_section
            
            for j in range(n_verts_per_section):
                j_next = (j + 1) % n_verts_per_section
                
                i0 = base_curr + j
                i1 = base_curr + j_next
                i2 = base_next + j
                i3 = base_next + j_next
                
                # Two triangles per quad
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
        
        # Add inlet face (outer end - flat rectangular)
        inlet_base = len(all_vertices)
        inlet_verts = cross_sections[0]
        
        # Add inlet face vertices with proper normals
        for v in inlet_verts:
            all_vertices.append(v)
            all_normals.append((-forward).tolist())  # Facing outward
        
        # Triangulate inlet face (fan from center)
        inlet_center = np.mean(inlet_verts, axis=0)
        all_vertices.append(inlet_center.tolist())
        all_normals.append((-forward).tolist())
        center_idx = len(all_vertices) - 1
        
        for j in range(n_verts_per_section):
            j_next = (j + 1) % n_verts_per_section
            all_indices.extend([center_idx, inlet_base + j, inlet_base + j_next])
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def get_inlet_velocity_direction(self) -> np.ndarray:
        """
        Get the direction of inlet velocity.

        For tangential inlet, this is primarily in the tangent direction
        with possible axial component.

        Returns:
            Unit vector in the direction of inlet flow
        """
        p = self.params

        # Combine tangential and possible axial component
        v = self.tangent.copy()
        if abs(p.entry_angle) > 1e-6:
            v[1] = -np.sin(p.entry_angle)
            v = v / np.linalg.norm(v)

        return v

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the inlet geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

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

    @property
    def entry_point(self) -> np.ndarray:
        """Get the point where flow enters the cyclone."""
        return self.surface_point.copy()

    @property
    def outer_point(self) -> np.ndarray:
        """Get the outer opening of the inlet."""
        return self.inlet_start.copy()
