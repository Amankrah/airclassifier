"""
Feed hopper component for flour/powder storage and discharge.

The feed hopper provides controlled storage and gravity discharge
of flour into the classification system. Design follows mass flow
principles for consistent discharge without ratholing.

Principle:
- Cylindrical section provides main storage volume
- Conical section ensures mass flow discharge
- Cone angle must exceed material's angle of repose + 10-15 deg
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class FeedHopperParams:
    """Parameters for feed hopper/silo."""

    # Geometry
    top_diameter: float          # [m] Top opening diameter
    bottom_diameter: float       # [m] Bottom discharge diameter
    cylindrical_height: float    # [m] Height of cylindrical section
    conical_height: float        # [m] Height of conical discharge section

    # Optional lid/cover
    has_lid: bool = True         # Include top cover
    lid_height: float = 0.05     # [m] Lid thickness

    # Design parameters
    wall_thickness: float = 0.003  # [m] Wall thickness (3mm default)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bottom center of discharge

    # Mesh resolution
    resolution_radial: int = 32
    resolution_axial: int = 24

    @property
    def top_radius(self) -> float:
        """Top radius."""
        return self.top_diameter / 2

    @property
    def bottom_radius(self) -> float:
        """Bottom discharge radius."""
        return self.bottom_diameter / 2

    @property
    def total_height(self) -> float:
        """Total hopper height."""
        h = self.conical_height + self.cylindrical_height
        if self.has_lid:
            h += self.lid_height
        return h

    @property
    def cone_half_angle(self) -> float:
        """Half-angle of the conical section in radians."""
        dr = self.top_radius - self.bottom_radius
        return np.arctan2(dr, self.conical_height)

    @property
    def cone_half_angle_degrees(self) -> float:
        """Half-angle in degrees (for checking mass flow criteria)."""
        return np.degrees(self.cone_half_angle)

    @property
    def cylindrical_volume(self) -> float:
        """Volume of cylindrical section [m^3]."""
        return PI * self.top_radius ** 2 * self.cylindrical_height

    @property
    def conical_volume(self) -> float:
        """Volume of conical section [m^3]."""
        r1, r2 = self.top_radius, self.bottom_radius
        h = self.conical_height
        return (PI * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

    @property
    def total_volume(self) -> float:
        """Total internal volume [m^3]."""
        return self.cylindrical_volume + self.conical_volume

    def capacity_kg(self, bulk_density: float = 500.0) -> float:
        """
        Calculate capacity in kg for given bulk density.

        Args:
            bulk_density: Material bulk density [kg/m^3] (default 500 for flour)

        Returns:
            Capacity in kg
        """
        return self.total_volume * bulk_density


class FeedHopper:
    """
    Feed hopper for powder/flour storage.

    Components:
    - Cylindrical storage section
    - Conical discharge section (mass flow design)
    - Optional lid/cover

    Coordinate system:
    - Origin at center of bottom discharge opening
    - Y-axis pointing upward
    """

    def __init__(self, params: FeedHopperParams):
        """
        Initialize feed hopper.

        Args:
            params: FeedHopperParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None
        
        # Cached separated meshes for animation
        self._body_vertices = None
        self._body_indices = None
        self._body_normals = None
        self._lid_vertices = None
        self._lid_indices = None
        self._lid_normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the feed hopper.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        n_radial = p.resolution_radial
        n_axial = p.resolution_axial

        # Generate conical section
        y_cone_bottom = p.center[1]
        y_cone_top = p.center[1] + p.conical_height

        n_cone = n_axial // 2
        for i in range(n_cone + 1):
            t = i / n_cone
            y = y_cone_bottom + t * p.conical_height
            r = p.bottom_radius + (p.top_radius - p.bottom_radius) * t

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)

                vertices.append([x, y, z])

                # Normal for conical surface (angled outward)
                dr = p.top_radius - p.bottom_radius
                slant = np.sqrt(p.conical_height ** 2 + dr ** 2)
                n_y = dr / slant
                n_r = p.conical_height / slant
                normals.append([n_r * np.cos(theta), n_y, n_r * np.sin(theta)])

        # Generate triangles for cone
        for i in range(n_cone):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = i * n_radial + j
                v1 = i * n_radial + j_next
                v2 = (i + 1) * n_radial + j_next
                v3 = (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Generate cylindrical section
        y_cyl_bottom = y_cone_top
        y_cyl_top = y_cyl_bottom + p.cylindrical_height

        cyl_start_idx = len(vertices)
        n_cyl = n_axial // 2
        for i in range(n_cyl + 1):
            t = i / n_cyl
            y = y_cyl_bottom + t * p.cylindrical_height
            r = p.top_radius

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)

                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

        # Generate triangles for cylinder
        for i in range(n_cyl):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = cyl_start_idx + i * n_radial + j
                v1 = cyl_start_idx + i * n_radial + j_next
                v2 = cyl_start_idx + (i + 1) * n_radial + j_next
                v3 = cyl_start_idx + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Generate lid if requested
        if p.has_lid:
            self._add_lid(vertices, indices, normals, y_cyl_top)
            # Add hinge brackets on hopper rim that connect to lid hinges
            self._add_hopper_hinge_brackets(vertices, indices, normals, y_cyl_top)

        # Generate bottom discharge ring
        self._add_discharge_ring(vertices, indices, normals, y_cone_bottom)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _add_lid(self, vertices: List, indices: List, normals: List, y_top: float):
        """
        Add industrial lid/cover to hopper with handle.
        
        Real hopper lid design (like a pot lid):
        - INNER SKIRT that goes DOWN INSIDE the hopper opening (seals against inner wall)
        - OUTER FLANGE that sits ON TOP of the hopper rim  
        - Domed surface for strength
        - Central handle (T-bar) for lifting
        - Hinge mounts on one side
        """
        p = self.params
        n_radial = p.resolution_radial
        
        # Lid dimensions - proper overlap design
        outer_flange_radius = p.top_radius * 1.08  # Outer flange extends beyond hopper
        inner_skirt_radius = p.top_radius * 0.97   # Inner skirt fits INSIDE hopper
        skirt_depth = p.lid_height * 1.5           # How far skirt goes down inside
        lid_thickness = p.lid_height * 0.4         # Thickness of lid plate
        dome_height = p.lid_height * 0.3           # Dome height
        
        y_rim = y_top                              # Hopper rim level
        y_lid_bottom = y_rim                       # Lid bottom sits on rim
        y_lid_top = y_rim + lid_thickness          # Lid top surface
        y_skirt_bottom = y_rim - skirt_depth       # Skirt goes down inside
        y_dome_peak = y_lid_top + dome_height
        
        # === 1. INNER SKIRT (goes down INSIDE hopper) ===
        skirt_start = len(vertices)
        
        # Skirt bottom ring (inside hopper)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + inner_skirt_radius * np.cos(theta)
            z = p.center[2] + inner_skirt_radius * np.sin(theta)
            vertices.append([x, y_skirt_bottom, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Skirt top ring (at rim level)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + inner_skirt_radius * np.cos(theta)
            z = p.center[2] + inner_skirt_radius * np.sin(theta)
            vertices.append([x, y_lid_bottom, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Inner skirt cylinder triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = skirt_start + j
            v1 = skirt_start + j_next
            v2 = skirt_start + n_radial + j_next
            v3 = skirt_start + n_radial + j
            indices.extend([v0, v2, v1])  # Inward facing
            indices.extend([v0, v3, v2])
        
        # Skirt bottom cap (closes the skirt)
        skirt_cap_center = len(vertices)
        vertices.append([p.center[0], y_skirt_bottom, p.center[2]])
        normals.append([0.0, -1.0, 0.0])
        
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([skirt_cap_center, skirt_start + j_next, skirt_start + j])
        
        # === 2. OUTER FLANGE (sits ON TOP of hopper rim) ===
        flange_start = len(vertices)
        
        # Flange bottom surface (sits on rim)
        # Inner edge at skirt radius
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + inner_skirt_radius * np.cos(theta)
            z = p.center[2] + inner_skirt_radius * np.sin(theta)
            vertices.append([x, y_lid_bottom, z])
            normals.append([0.0, -1.0, 0.0])
        
        # Outer edge at flange radius
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + outer_flange_radius * np.cos(theta)
            z = p.center[2] + outer_flange_radius * np.sin(theta)
            vertices.append([x, y_lid_bottom, z])
            normals.append([0.0, -1.0, 0.0])
        
        # Flange bottom surface triangles (annular ring)
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v_inner = flange_start + j
            v_inner_next = flange_start + j_next
            v_outer = flange_start + n_radial + j
            v_outer_next = flange_start + n_radial + j_next
            indices.extend([v_inner, v_outer, v_inner_next])
            indices.extend([v_inner_next, v_outer, v_outer_next])
        
        # Outer flange vertical edge
        flange_edge_start = len(vertices)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + outer_flange_radius * np.cos(theta)
            z = p.center[2] + outer_flange_radius * np.sin(theta)
            vertices.append([x, y_lid_bottom, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
            vertices.append([x, y_lid_top, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Flange edge triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = flange_edge_start + j * 2
            v1 = flange_edge_start + j * 2 + 1
            v2 = flange_edge_start + j_next * 2 + 1
            v3 = flange_edge_start + j_next * 2
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # === 3. DOMED LID SURFACE ===
        n_dome_rings = 4
        dome_ring_start = len(vertices)
        
        for ring in range(n_dome_rings + 1):
            t = ring / n_dome_rings
            r = outer_flange_radius * (1 - t)  # Radius decreases toward center
            # Dome height follows smooth curve
            y = y_lid_top + dome_height * np.sqrt(max(0, 1 - (1-t)**2))
            
            if ring == n_dome_rings:
                # Center point
                vertices.append([p.center[0], y_dome_peak, p.center[2]])
                normals.append([0.0, 1.0, 0.0])
            else:
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + r * np.cos(theta)
                    z = p.center[2] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    # Normal points outward and up on dome
                    nx = np.cos(theta) * (1-t) * 0.3
                    nz = np.sin(theta) * (1-t) * 0.3
                    ny = np.sqrt(max(0.01, 1 - nx**2 - nz**2))
                    normals.append([nx, ny, nz])
        
        # Dome triangles between rings
        for ring in range(n_dome_rings - 1):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = dome_ring_start + ring * n_radial + j
                v1 = dome_ring_start + ring * n_radial + j_next
                v2 = dome_ring_start + (ring + 1) * n_radial + j_next
                v3 = dome_ring_start + (ring + 1) * n_radial + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
        
        # Triangles to center
        center_idx = dome_ring_start + (n_dome_rings - 1) * n_radial + n_radial
        last_ring_start = dome_ring_start + (n_dome_rings - 1) * n_radial
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            indices.extend([last_ring_start + j, last_ring_start + j_next, center_idx])
        
        # === 4. HANDLE (T-bar style) ===
        self._add_handle(vertices, indices, normals, y_dome_peak)
        
        # === 5. HINGE MOUNTS (two on one side of lid) ===
        self._add_hinge_mounts(vertices, indices, normals, y_lid_bottom, outer_flange_radius)
    
    def _add_handle(self, vertices: List, indices: List, normals: List, y_base: float):
        """Add T-bar handle on top of lid for lifting/opening."""
        p = self.params
        
        # Handle dimensions
        bar_radius = 0.008  # 8mm diameter bar
        vertical_height = p.top_diameter * 0.08  # Vertical post height
        horizontal_width = p.top_diameter * 0.15  # Width of T-bar
        
        n_seg = 8  # Segments for round bar
        
        # === Vertical post ===
        post_start = len(vertices)
        y_post_top = y_base + vertical_height
        
        # Post bottom ring
        for j in range(n_seg):
            theta = (j / n_seg) * TWO_PI
            x = p.center[0] + bar_radius * np.cos(theta)
            z = p.center[2] + bar_radius * np.sin(theta)
            vertices.append([x, y_base, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Post top ring
        for j in range(n_seg):
            theta = (j / n_seg) * TWO_PI
            x = p.center[0] + bar_radius * np.cos(theta)
            z = p.center[2] + bar_radius * np.sin(theta)
            vertices.append([x, y_post_top, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Post triangles
        for j in range(n_seg):
            j_next = (j + 1) % n_seg
            v0 = post_start + j
            v1 = post_start + j_next
            v2 = post_start + n_seg + j_next
            v3 = post_start + n_seg + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # === Horizontal T-bar ===
        tbar_start = len(vertices)
        half_width = horizontal_width / 2
        
        # T-bar runs along X axis
        # Left end
        for j in range(n_seg):
            theta = (j / n_seg) * TWO_PI
            y = y_post_top + bar_radius * np.cos(theta)
            z = p.center[2] + bar_radius * np.sin(theta)
            vertices.append([p.center[0] - half_width, y, z])
            normals.append([0.0, np.cos(theta), np.sin(theta)])
        
        # Right end
        for j in range(n_seg):
            theta = (j / n_seg) * TWO_PI
            y = y_post_top + bar_radius * np.cos(theta)
            z = p.center[2] + bar_radius * np.sin(theta)
            vertices.append([p.center[0] + half_width, y, z])
            normals.append([0.0, np.cos(theta), np.sin(theta)])
        
        # T-bar triangles
        for j in range(n_seg):
            j_next = (j + 1) % n_seg
            v0 = tbar_start + j
            v1 = tbar_start + j_next
            v2 = tbar_start + n_seg + j_next
            v3 = tbar_start + n_seg + j
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])
        
        # End caps
        left_center_idx = len(vertices)
        vertices.append([p.center[0] - half_width, y_post_top, p.center[2]])
        normals.append([-1.0, 0.0, 0.0])
        
        right_center_idx = len(vertices)
        vertices.append([p.center[0] + half_width, y_post_top, p.center[2]])
        normals.append([1.0, 0.0, 0.0])
        
        # Left cap triangles
        for j in range(n_seg):
            j_next = (j + 1) % n_seg
            indices.extend([left_center_idx, tbar_start + j_next, tbar_start + j])
        
        # Right cap triangles
        for j in range(n_seg):
            j_next = (j + 1) % n_seg
            indices.extend([right_center_idx, tbar_start + n_seg + j, tbar_start + n_seg + j_next])
    
    def _add_hinge_mounts(self, vertices: List, indices: List, normals: List, 
                          y_base: float, lid_radius: float):
        """
        Add hinge knuckles on the lid that will connect to hopper brackets.
        
        These are the "female" part of the hinge on the lid flange edge.
        """
        p = self.params
        
        # Hinge knuckle dimensions
        knuckle_outer_radius = 0.012  # Outer radius of knuckle cylinder
        knuckle_inner_radius = 0.005  # Hole for pin
        knuckle_width = 0.015         # Width along Z
        
        lid_thickness = p.lid_height * 0.4
        n_seg = 12
        
        # Position two hinges on the -X side
        hinge_positions = [
            -p.top_radius * 0.4,
            p.top_radius * 0.4,
        ]
        
        for z_pos in hinge_positions:
            knuckle_start = len(vertices)
            
            # Knuckle center position - at edge of lid, halfway up the lid thickness
            x_center = p.center[0] - lid_radius - knuckle_outer_radius * 0.5
            y_center = y_base + lid_thickness * 0.5
            
            # Generate knuckle cylinder (hollow tube shape)
            # Front ring (closer to hopper center)
            for j in range(n_seg):
                theta = (j / n_seg) * TWO_PI
                x = x_center + knuckle_outer_radius * np.cos(theta)
                y = y_center + knuckle_outer_radius * np.sin(theta)
                vertices.append([x, y, z_pos - knuckle_width/2])
                normals.append([np.cos(theta), np.sin(theta), 0.0])
            
            # Back ring
            for j in range(n_seg):
                theta = (j / n_seg) * TWO_PI
                x = x_center + knuckle_outer_radius * np.cos(theta)
                y = y_center + knuckle_outer_radius * np.sin(theta)
                vertices.append([x, y, z_pos + knuckle_width/2])
                normals.append([np.cos(theta), np.sin(theta), 0.0])
            
            # Knuckle cylinder surface
            for j in range(n_seg):
                j_next = (j + 1) % n_seg
                v0 = knuckle_start + j
                v1 = knuckle_start + j_next
                v2 = knuckle_start + n_seg + j_next
                v3 = knuckle_start + n_seg + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
            
            # End caps (with center hole - simplified as solid for now)
            cap_front = len(vertices)
            vertices.append([x_center, y_center, z_pos - knuckle_width/2])
            normals.append([0.0, 0.0, -1.0])
            
            cap_back = len(vertices)
            vertices.append([x_center, y_center, z_pos + knuckle_width/2])
            normals.append([0.0, 0.0, 1.0])
            
            for j in range(n_seg):
                j_next = (j + 1) % n_seg
                indices.extend([cap_front, knuckle_start + j_next, knuckle_start + j])
                indices.extend([cap_back, knuckle_start + n_seg + j, knuckle_start + n_seg + j_next])
            
            # Connection arm from lid flange to knuckle
            arm_start = len(vertices)
            arm_width = 0.008
            arm_height = lid_thickness * 0.8
            
            # 4 corners at lid edge
            vertices.append([p.center[0] - lid_radius, y_base, z_pos - arm_width])
            vertices.append([p.center[0] - lid_radius, y_base, z_pos + arm_width])
            vertices.append([p.center[0] - lid_radius, y_base + arm_height, z_pos - arm_width])
            vertices.append([p.center[0] - lid_radius, y_base + arm_height, z_pos + arm_width])
            # 4 corners at knuckle
            vertices.append([x_center + knuckle_outer_radius, y_center - arm_height/2, z_pos - arm_width])
            vertices.append([x_center + knuckle_outer_radius, y_center - arm_height/2, z_pos + arm_width])
            vertices.append([x_center + knuckle_outer_radius, y_center + arm_height/2, z_pos - arm_width])
            vertices.append([x_center + knuckle_outer_radius, y_center + arm_height/2, z_pos + arm_width])
            
            for _ in range(8):
                normals.append([0.0, 0.0, 1.0])  # Simplified normals
            
            # Arm faces
            indices.extend([arm_start, arm_start+4, arm_start+6])
            indices.extend([arm_start, arm_start+6, arm_start+2])
            indices.extend([arm_start+1, arm_start+3, arm_start+7])
            indices.extend([arm_start+1, arm_start+7, arm_start+5])
            indices.extend([arm_start+2, arm_start+6, arm_start+7])
            indices.extend([arm_start+2, arm_start+7, arm_start+3])
            indices.extend([arm_start, arm_start+1, arm_start+5])
            indices.extend([arm_start, arm_start+5, arm_start+4])

    def _add_hopper_hinge_brackets(self, vertices: List, indices: List, normals: List, 
                                    y_rim: float):
        """
        Add hinge brackets on hopper body that connect to lid hinge knuckles.
        
        These brackets are welded to the hopper and hold the hinge pin.
        """
        p = self.params
        outer_flange_radius = p.top_radius * 1.08  # Must match lid
        
        # Bracket dimensions
        bracket_thickness = 0.008
        bracket_height = 0.04
        bracket_depth = 0.025  # How far out from hopper
        
        # Pin dimensions
        pin_radius = 0.004
        pin_length = 0.035
        
        n_seg = 8
        
        # Position two hinges on the -X side
        hinge_positions = [
            -p.top_radius * 0.4,
            p.top_radius * 0.4,
        ]
        
        lid_thickness = p.lid_height * 0.4
        knuckle_outer_radius = 0.012
        x_knuckle = p.center[0] - outer_flange_radius - knuckle_outer_radius * 0.5
        y_knuckle = y_rim + lid_thickness * 0.5
        
        for z_pos in hinge_positions:
            # === BRACKET (L-shaped, attached to hopper wall) ===
            bracket_start = len(vertices)
            
            # Bracket attaches to hopper at rim level
            x_hopper = p.center[0] - p.top_radius
            x_outer = x_knuckle - knuckle_outer_radius - 0.002  # Just past knuckle
            
            # Vertical part of bracket (attached to hopper)
            y_bracket_top = y_knuckle + knuckle_outer_radius + 0.005
            y_bracket_bottom = y_rim - bracket_height * 0.5
            
            # Back plate (against hopper)
            vertices.append([x_hopper, y_bracket_bottom, z_pos - bracket_thickness])
            vertices.append([x_hopper, y_bracket_bottom, z_pos + bracket_thickness])
            vertices.append([x_hopper, y_bracket_top, z_pos - bracket_thickness])
            vertices.append([x_hopper, y_bracket_top, z_pos + bracket_thickness])
            
            # Front plate (extending out)
            vertices.append([x_outer, y_bracket_bottom, z_pos - bracket_thickness])
            vertices.append([x_outer, y_bracket_bottom, z_pos + bracket_thickness])
            vertices.append([x_outer, y_bracket_top, z_pos - bracket_thickness])
            vertices.append([x_outer, y_bracket_top, z_pos + bracket_thickness])
            
            for _ in range(8):
                normals.append([-1.0, 0.0, 0.0])
            
            # Bracket faces
            # Top
            indices.extend([bracket_start+2, bracket_start+3, bracket_start+7])
            indices.extend([bracket_start+2, bracket_start+7, bracket_start+6])
            # Bottom
            indices.extend([bracket_start, bracket_start+4, bracket_start+5])
            indices.extend([bracket_start, bracket_start+5, bracket_start+1])
            # Back (at hopper)
            indices.extend([bracket_start, bracket_start+2, bracket_start+3])
            indices.extend([bracket_start, bracket_start+3, bracket_start+1])
            # Front
            indices.extend([bracket_start+4, bracket_start+6, bracket_start+7])
            indices.extend([bracket_start+4, bracket_start+7, bracket_start+5])
            # Sides
            indices.extend([bracket_start, bracket_start+4, bracket_start+6])
            indices.extend([bracket_start, bracket_start+6, bracket_start+2])
            indices.extend([bracket_start+1, bracket_start+3, bracket_start+7])
            indices.extend([bracket_start+1, bracket_start+7, bracket_start+5])
            
            # === HINGE PIN (passes through both bracket and lid knuckle) ===
            pin_start = len(vertices)
            pin_x = x_knuckle
            pin_y = y_knuckle
            
            z_pin_start = z_pos - pin_length / 2
            z_pin_end = z_pos + pin_length / 2
            
            # Pin cylinder
            for j in range(n_seg):
                theta = (j / n_seg) * TWO_PI
                x = pin_x + pin_radius * np.cos(theta)
                y = pin_y + pin_radius * np.sin(theta)
                vertices.append([x, y, z_pin_start])
                normals.append([np.cos(theta), np.sin(theta), 0.0])
            
            for j in range(n_seg):
                theta = (j / n_seg) * TWO_PI
                x = pin_x + pin_radius * np.cos(theta)
                y = pin_y + pin_radius * np.sin(theta)
                vertices.append([x, y, z_pin_end])
                normals.append([np.cos(theta), np.sin(theta), 0.0])
            
            for j in range(n_seg):
                j_next = (j + 1) % n_seg
                v0 = pin_start + j
                v1 = pin_start + j_next
                v2 = pin_start + n_seg + j_next
                v3 = pin_start + n_seg + j
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
            
            # Pin caps
            left_cap = len(vertices)
            vertices.append([pin_x, pin_y, z_pin_start])
            normals.append([0.0, 0.0, -1.0])
            
            right_cap = len(vertices)
            vertices.append([pin_x, pin_y, z_pin_end])
            normals.append([0.0, 0.0, 1.0])
            
            for j in range(n_seg):
                j_next = (j + 1) % n_seg
                indices.extend([left_cap, pin_start + j_next, pin_start + j])
                indices.extend([right_cap, pin_start + n_seg + j, pin_start + n_seg + j_next])

    def _add_discharge_ring(self, vertices: List, indices: List, normals: List, y_bottom: float):
        """Add discharge ring at bottom."""
        p = self.params
        n_radial = p.resolution_radial // 2

        ring_start_idx = len(vertices)
        ring_height = p.bottom_diameter * 0.2

        # Inner and outer rings
        for i in range(2):
            y = y_bottom - i * ring_height
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + p.bottom_radius * np.cos(theta)
                z = p.center[2] + p.bottom_radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), 0.0, np.sin(theta)])

        # Triangles for discharge ring
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = ring_start_idx + j
            v1 = ring_start_idx + j_next
            v2 = ring_start_idx + n_radial + j_next
            v3 = ring_start_idx + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def is_mass_flow_design(self, material_angle_of_repose: float = 35.0) -> bool:
        """
        Check if hopper meets mass flow criteria.

        Args:
            material_angle_of_repose: Material's angle of repose in degrees

        Returns:
            True if design should achieve mass flow
        """
        # Mass flow requires cone angle > angle of repose + 10-15 deg
        required_angle = material_angle_of_repose + 12
        return self.params.cone_half_angle_degrees > required_angle

    def get_discharge_center(self) -> Tuple[float, float, float]:
        """Get center position of discharge opening."""
        p = self.params
        return (p.center[0], p.center[1], p.center[2])

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the hopper geometry."""
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
    def normals(self) -> np.ndarray:
        """Get vertex normals."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals

    # =========================================================================
    # ANIMATION SUPPORT - Separate body and lid meshes
    # =========================================================================
    
    def get_body_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for hopper body WITHOUT lid (for animation).
        
        Returns:
            Tuple of (vertices, indices, normals) for body only
        """
        if self._body_vertices is None:
            self._generate_separated_meshes()
        return self._body_vertices, self._body_indices, self._body_normals
    
    def get_lid_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for lid with handle (for animation).
        
        The lid mesh includes:
        - Inner skirt
        - Outer flange
        - Domed surface
        - T-bar handle
        - Hinge knuckles
        
        Returns:
            Tuple of (vertices, indices, normals) for lid only
        """
        if self._lid_vertices is None:
            self._generate_separated_meshes()
        return self._lid_vertices, self._lid_indices, self._lid_normals
    
    def get_lid_hinge_position(self) -> Tuple[float, float, float]:
        """
        Get the hinge axis position for lid rotation.
        
        Returns:
            (x, y, z) position of hinge axis
        """
        p = self.params
        outer_flange_radius = p.top_radius * 1.08
        y_rim = p.center[1] + p.conical_height + p.cylindrical_height
        lid_thickness = p.lid_height * 0.4
        
        # Hinge is on -X side of lid, at the flange edge
        hinge_x = p.center[0] - outer_flange_radius
        hinge_y = y_rim + lid_thickness * 0.5
        hinge_z = p.center[2]
        
        return (hinge_x, hinge_y, hinge_z)
    
    def _generate_separated_meshes(self):
        """Generate separate meshes for body and lid (for animation)."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial
        
        # ===== BODY MESH (no lid) =====
        body_verts = []
        body_indices = []
        body_normals = []
        
        y_cone_bottom = p.center[1]
        y_cone_top = p.center[1] + p.conical_height
        y_cyl_top = y_cone_top + p.cylindrical_height
        
        # Generate conical section
        cone_start = len(body_verts)
        for i in range(n_axial + 1):
            t = i / n_axial
            y = y_cone_bottom + t * p.conical_height
            r = p.bottom_radius + t * (p.top_radius - p.bottom_radius)
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)
                body_verts.append([x, y, z])
                
                cone_angle = np.arctan2(p.top_radius - p.bottom_radius, p.conical_height)
                nx = np.cos(theta) * np.cos(cone_angle)
                ny = np.sin(cone_angle)
                nz = np.sin(theta) * np.cos(cone_angle)
                body_normals.append([nx, ny, nz])
        
        for i in range(n_axial):
            for j in range(n_radial):
                v0 = cone_start + i * n_radial + j
                v1 = cone_start + i * n_radial + (j + 1) % n_radial
                v2 = cone_start + (i + 1) * n_radial + j
                v3 = cone_start + (i + 1) * n_radial + (j + 1) % n_radial
                body_indices.extend([v0, v1, v2])
                body_indices.extend([v1, v3, v2])
        
        # Generate cylindrical section
        cyl_start = len(body_verts)
        for i in range(n_axial + 1):
            t = i / n_axial
            y = y_cone_top + t * p.cylindrical_height
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + p.top_radius * np.cos(theta)
                z = p.center[2] + p.top_radius * np.sin(theta)
                body_verts.append([x, y, z])
                body_normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        for i in range(n_axial):
            for j in range(n_radial):
                v0 = cyl_start + i * n_radial + j
                v1 = cyl_start + i * n_radial + (j + 1) % n_radial
                v2 = cyl_start + (i + 1) * n_radial + j
                v3 = cyl_start + (i + 1) * n_radial + (j + 1) % n_radial
                body_indices.extend([v0, v1, v2])
                body_indices.extend([v1, v3, v2])
        
        # Add hinge brackets on hopper (these stay with body)
        self._add_hopper_hinge_brackets(body_verts, body_indices, body_normals, y_cyl_top)
        
        # Generate bottom discharge ring
        self._add_discharge_ring(body_verts, body_indices, body_normals, y_cone_bottom)
        
        self._body_vertices = np.array(body_verts, dtype=np.float32)
        self._body_indices = np.array(body_indices, dtype=np.int32)
        self._body_normals = np.array(body_normals, dtype=np.float32)
        
        # ===== LID MESH (with handle) =====
        if p.has_lid:
            lid_verts = []
            lid_indices = []
            lid_normals = []
            
            self._add_lid(lid_verts, lid_indices, lid_normals, y_cyl_top)
            
            self._lid_vertices = np.array(lid_verts, dtype=np.float32)
            self._lid_indices = np.array(lid_indices, dtype=np.int32)
            self._lid_normals = np.array(lid_normals, dtype=np.float32)
        else:
            # No lid
            self._lid_vertices = np.array([], dtype=np.float32).reshape(0, 3)
            self._lid_indices = np.array([], dtype=np.int32)
            self._lid_normals = np.array([], dtype=np.float32).reshape(0, 3)

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for this component.
        
        The port positions represent the ACTUAL CONNECTION SURFACES where
        components physically meet, not just the centerlines.
        
        Returns:
            Dictionary of port name to ConnectionPort:
            - 'inlet': Top opening for material loading (gravity/pneumatic)
            - 'discharge': Bottom outlet to airlock/feeder
        """
        p = self.params
        
        # Inlet port at top of hopper (if no lid, material entry point)
        inlet_y = p.total_height
        
        # Discharge port: accounts for discharge ring extending below Y=0
        # The discharge ring extends down by bottom_diameter * 0.2
        ring_height = p.bottom_diameter * 0.2
        discharge_y = -ring_height  # Bottom surface of discharge ring
        
        return {
            'inlet': ConnectionPort(
                position=(0.0, inlet_y, 0.0),
                direction=(0.0, 1.0, 0.0),  # Points up (material enters from above)
                diameter=p.top_diameter if not p.has_lid else p.top_diameter * 0.5,
                port_type=PortType.GRAVITY,
                name="hopper_inlet",
                compatible_types=[PortType.GRAVITY, PortType.CIRCULAR],
            ),
            'discharge': ConnectionPort(
                position=(0.0, discharge_y, 0.0),
                direction=(0.0, -1.0, 0.0),  # Points down (gravity discharge)
                diameter=p.bottom_diameter,
                port_type=PortType.FLANGED,  # Has discharge ring/flange
                name="hopper_discharge",
                flange_diameter=p.bottom_diameter * 1.2,
                compatible_types=[PortType.GRAVITY, PortType.CIRCULAR, PortType.FLANGED],
            ),
        }


def create_standard_feed_hopper(
    capacity_kg: float = 500,
    bulk_density: float = 500,
    discharge_diameter: float = 0.15
) -> FeedHopper:
    """
    Create a standard feed hopper sized for given capacity.

    Args:
        capacity_kg: Design capacity [kg]
        bulk_density: Material bulk density [kg/m^3]
        discharge_diameter: Discharge opening diameter [m]

    Returns:
        FeedHopper instance
    """
    # Calculate required volume
    required_volume = capacity_kg / bulk_density

    # Standard proportions
    # Assume cylinder height = 1.5 * diameter
    # Cone angle = 45 degrees for mass flow with most powders

    # Estimate top diameter from volume
    # V_total = V_cyl + V_cone
    # With aspect ratio ~ 2:1 (height:diameter)
    # Approximate: V ~ 0.8 * pi * r^2 * h with h = 2 * r
    # V ~ 1.6 * pi * r^3

    r_estimate = (required_volume / (1.6 * PI)) ** (1/3)
    top_diameter = 2 * r_estimate

    # Ensure minimum size
    top_diameter = max(top_diameter, discharge_diameter * 3)

    params = FeedHopperParams(
        top_diameter=top_diameter,
        bottom_diameter=discharge_diameter,
        cylindrical_height=top_diameter * 0.75,
        conical_height=top_diameter * 0.6,
        has_lid=True,
    )

    return FeedHopper(params)
