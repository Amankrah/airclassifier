"""
Expanding transition with integrated dropout hopper.

This component replaces the plain round-to-rect transition between venturi
and zigzag. It morphs from circular to rectangular cross-section using
superellipse interpolation, and includes a conical hopper on the underside
where heavy particles settle when duct expansion causes velocity to drop.

Physical principle: At the expansion, air velocity decreases (continuity).
Particles with terminal velocity > local air velocity cannot be entrained
upward and fall into the hopper below.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None

from ..connection_ports import ConnectionPort, PortType


@dataclass
class ExpandingTransitionParams:
    """
    Parameters for expanding transition with dropout hopper.

    Attributes:
        inlet_diameter: Round inlet diameter [m] (connects to venturi outlet duct)
        outlet_width: Rectangular outlet width [m] (zigzag channel width)
        outlet_depth: Rectangular outlet depth [m] (zigzag channel depth)
        transition_length: Vertical length of expansion section [m]
        wall_thickness: Wall thickness [m]
        hopper_height: Height of hopper cone extending downward [m]
        hopper_outlet_diameter: Diameter of hopper discharge opening [m]
        center: Center position of transition inlet (x, y, z) [m]
        direction: Flow direction vector (dx, dy, dz)
        flanged: Whether to add flanges at connections
        flange_width: Width of flange beyond pipe [m]
        flange_thickness: Thickness of flange [m]
        resolution: Number of circumferential segments
    """
    # Expansion section (round -> rect)
    inlet_diameter: float
    outlet_width: float
    outlet_depth: float
    transition_length: float
    wall_thickness: float = 0.002

    # Dropout hopper
    hopper_height: float = 0.15
    hopper_outlet_diameter: float = 0.04

    # Position / orientation
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Flanges
    flanged: bool = True
    flange_width: float = 0.02
    flange_thickness: float = 0.008

    # Resolution
    resolution: int = 24

    @property
    def direction_normalized(self) -> Tuple[float, float, float]:
        d = np.array(self.direction)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return tuple(d)

    @property
    def length(self) -> float:
        """Alias for transition_length (compatibility with print_summary)."""
        return self.transition_length

    @property
    def hopper_half_angle_deg(self) -> float:
        """Derived hopper half-angle [deg] from geometry."""
        r_top = self.inlet_diameter / 2
        r_bot = self.hopper_outlet_diameter / 2
        if self.hopper_height > 0:
            return np.degrees(np.arctan((r_top - r_bot) / self.hopper_height))
        return 0.0

    @property
    def inlet_dimensions(self) -> Tuple[float, ...]:
        """Compatibility with TransitionParams: (diameter,) for round inlet."""
        return (self.inlet_diameter,)

    @property
    def outlet_dimensions(self) -> Tuple[float, ...]:
        """Compatibility with TransitionParams: (width, height) for rect outlet."""
        return (self.outlet_width, self.outlet_depth)


class ExpandingTransitionWithDropout:
    """
    Round-to-rectangular expanding transition with integrated dropout hopper.

    The expansion body morphs from circular to rectangular cross-section
    using superellipse interpolation. A conical hopper hangs below the
    inlet opening to collect heavy particles that cannot be carried upward.

    Ports:
        inlet: Round, at bottom of expansion body (connects to round duct)
        outlet: Rectangular, at top of expansion body (connects to zigzag)
        dropout: Round, at hopper discharge (gravity collection)
    """

    def __init__(self, params: ExpandingTransitionParams):
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices

    @property
    def indices(self) -> np.ndarray:
        if self._indices is None:
            self.generate_mesh()
        return self._indices

    @property
    def normals(self) -> np.ndarray:
        if self._normals is None:
            self.generate_mesh()
        return self._normals

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        p = self.params
        cx, cy, cz = p.center
        dx, dy, dz = p.direction_normalized

        return {
            'inlet': ConnectionPort(
                position=(cx, cy, cz),
                direction=(-dx, -dy, -dz),
                diameter=p.inlet_diameter,
                port_type=PortType.CIRCULAR,
                name="inlet",
            ),
            'outlet': ConnectionPort(
                position=(
                    cx + p.transition_length * dx,
                    cy + p.transition_length * dy,
                    cz + p.transition_length * dz,
                ),
                direction=(dx, dy, dz),
                width=p.outlet_width,
                height=p.outlet_depth,
                port_type=PortType.RECTANGULAR,
                name="outlet",
            ),
            'dropout': ConnectionPort(
                position=(
                    cx - p.hopper_height * dx,
                    cy - p.hopper_height * dy,
                    cz - p.hopper_height * dz,
                ),
                direction=(-dx, -dy, -dz),
                diameter=p.hopper_outlet_diameter,
                port_type=PortType.CIRCULAR,
                name="dropout",
            ),
        }

    # ------------------------------------------------------------------
    # Coordinate system helpers (same pattern as Transition component)
    # ------------------------------------------------------------------

    def _get_coordinate_system(self):
        """Get orthonormal basis (direction, perp1, perp2) from direction vector."""
        direction = np.array(self.params.direction_normalized)

        if abs(direction[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])

        perp1 = np.cross(direction, up)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        return direction, perp1, perp2

    def _transform_point(self, local_x, local_y, z_offset, direction, perp1, perp2):
        """Transform local (x, y, z_along_axis) to world coordinates."""
        cx, cy, cz = self.params.center
        return [
            cx + local_x * perp1[0] + local_y * perp2[0] + z_offset * direction[0],
            cy + local_x * perp1[1] + local_y * perp2[1] + z_offset * direction[1],
            cz + local_x * perp1[2] + local_y * perp2[2] + z_offset * direction[2],
        ]

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def generate_mesh(self):
        """Generate the complete mesh: expansion body + dropout hopper."""
        p = self.params
        n_seg = p.resolution
        n_div = max(8, n_seg // 3)
        direction, perp1, perp2 = self._get_coordinate_system()

        all_v: list = []
        all_i: list = []
        all_n: list = []

        # Part 1: Round-to-rect expansion body (z=0 to z=transition_length)
        self._generate_expansion_body(all_v, all_i, all_n,
                                      n_seg, n_div, direction, perp1, perp2)

        # Part 2: Conical hopper (z=0 downward to z=-hopper_height)
        self._generate_hopper(all_v, all_i, all_n,
                              n_seg, n_div, direction, perp1, perp2)

        self._vertices = np.array(all_v, dtype=np.float32)
        self._indices = np.array(all_i, dtype=np.int32)
        self._normals = np.array(all_n, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_expansion_body(self, vertices, indices, normals,
                                 n_seg, n_div, direction, perp1, perp2):
        """Generate round-to-rect expansion mesh (superellipse interpolation)."""
        p = self.params
        r_in = p.inlet_diameter / 2
        w_out = p.outlet_width / 2
        h_out = p.outlet_depth / 2
        t = p.wall_thickness

        # --- Outer surface ---
        for layer in range(n_div + 1):
            t_param = layer / n_div
            z_pos = t_param * p.transition_length

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg

                # Circle point at inlet
                circ_x = r_in * np.cos(theta)
                circ_y = r_in * np.sin(theta)

                # Superellipse point at outlet (n=2 circle -> n=20 near-rect)
                n_exp = 2 + 18 * t_param
                rect_x = w_out * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2 / n_exp)
                rect_y = h_out * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2 / n_exp)

                # Interpolate
                local_x = circ_x + t_param * (rect_x - circ_x)
                local_y = circ_y + t_param * (rect_y - circ_y)

                # Offset outward by wall thickness
                scale = 1 + t / max(abs(local_x), abs(local_y), r_in)
                outer_x = local_x * scale
                outer_y = local_y * scale

                pt = self._transform_point(outer_x, outer_y, z_pos,
                                           direction, perp1, perp2)
                vertices.append(pt)

                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                normals.append([nx, ny, nz])

            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % n_seg
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % n_seg
                    indices.extend([i0, i2, i1])
                    indices.extend([i1, i2, i3])

        # --- Inner surface ---
        for layer in range(n_div + 1):
            t_param = layer / n_div
            z_pos = t_param * p.transition_length

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg

                circ_x = r_in * np.cos(theta)
                circ_y = r_in * np.sin(theta)

                n_exp = 2 + 18 * t_param
                rect_x = w_out * np.sign(np.cos(theta)) * abs(np.cos(theta)) ** (2 / n_exp)
                rect_y = h_out * np.sign(np.sin(theta)) * abs(np.sin(theta)) ** (2 / n_exp)

                local_x = circ_x + t_param * (rect_x - circ_x)
                local_y = circ_y + t_param * (rect_y - circ_y)

                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2)
                vertices.append(pt)

                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                normals.append([-nx, -ny, -nz])

            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % n_seg
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % n_seg
                    indices.extend([i0, i1, i2])
                    indices.extend([i1, i3, i2])

        # --- Flanges ---
        if p.flanged:
            # Inlet flange (round) at z=0
            self._generate_flange_ring(
                vertices, indices, normals,
                inner_radius=r_in + t,
                z_offset=0.0,
                is_inlet=True,
                direction=direction, perp1=perp1, perp2=perp2,
                num_segments=n_seg,
            )
            # Outlet flange (rectangular) at z=transition_length
            self._generate_rectangular_flange(
                vertices, indices, normals,
                inner_width=w_out + t,
                inner_height=h_out + t,
                z_offset=p.transition_length,
                is_inlet=False,
                direction=direction, perp1=perp1, perp2=perp2,
            )

    def _generate_hopper(self, vertices, indices, normals,
                         n_seg, n_div, direction, perp1, perp2):
        """Generate conical hopper mesh extending downward from the inlet."""
        p = self.params
        r_top = p.inlet_diameter / 2
        r_bot = p.hopper_outlet_diameter / 2
        t = p.wall_thickness

        n_hopper_div = max(6, n_div // 2)

        # --- Outer surface (frustum z=0 to z=-hopper_height) ---
        for layer in range(n_hopper_div + 1):
            t_param = layer / n_hopper_div
            z_pos = -t_param * p.hopper_height
            radius = r_top + t_param * (r_bot - r_top)

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = (radius + t) * np.cos(theta)
                local_y = (radius + t) * np.sin(theta)

                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2)
                vertices.append(pt)

                # Cone normal with slope component
                dr = r_top - r_bot  # positive: tapers inward going down
                slope = dr / p.hopper_height if p.hopper_height > 0 else 0
                factor = 1.0 / np.sqrt(1 + slope ** 2)
                radial = factor
                ax = slope * factor

                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]

                normals.append([
                    radial * nx - ax * direction[0],
                    radial * ny - ax * direction[1],
                    radial * nz - ax * direction[2],
                ])

            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % n_seg
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % n_seg
                    indices.extend([i0, i2, i1])
                    indices.extend([i1, i2, i3])

        # --- Inner surface ---
        for layer in range(n_hopper_div + 1):
            t_param = layer / n_hopper_div
            z_pos = -t_param * p.hopper_height
            radius = r_top + t_param * (r_bot - r_top)

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)

                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2)
                vertices.append(pt)

                dr = r_top - r_bot
                slope = dr / p.hopper_height if p.hopper_height > 0 else 0
                factor = 1.0 / np.sqrt(1 + slope ** 2)
                radial = factor
                ax = slope * factor

                nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]

                normals.append([
                    -(radial * nx - ax * direction[0]),
                    -(radial * ny - ax * direction[1]),
                    -(radial * nz - ax * direction[2]),
                ])

            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0 = prev_base + i
                    i1 = prev_base + (i + 1) % n_seg
                    i2 = base_idx + i
                    i3 = base_idx + (i + 1) % n_seg
                    indices.extend([i0, i1, i2])
                    indices.extend([i1, i3, i2])

        # --- Bottom annular cap ---
        base_idx = len(vertices)
        z_bot = -p.hopper_height
        cap_normal = [-d for d in direction]

        for radius in [r_bot, r_bot + t]:
            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)
                pt = self._transform_point(local_x, local_y, z_bot,
                                           direction, perp1, perp2)
                vertices.append(pt)
                normals.append(cap_normal)

        for i in range(n_seg):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % n_seg
            i2 = base_idx + n_seg + i
            i3 = base_idx + n_seg + (i + 1) % n_seg
            indices.extend([i0, i2, i1])
            indices.extend([i1, i2, i3])

        # --- Hopper discharge flange ---
        if p.flanged:
            self._generate_flange_ring(
                vertices, indices, normals,
                inner_radius=r_bot + t,
                z_offset=-p.hopper_height,
                is_inlet=False,
                direction=direction, perp1=perp1, perp2=perp2,
                num_segments=n_seg,
            )

    # ------------------------------------------------------------------
    # Flange generation (follows Transition pattern exactly)
    # ------------------------------------------------------------------

    def _generate_flange_ring(self, vertices, indices, normals,
                              inner_radius, z_offset, is_inlet,
                              direction, perp1, perp2, num_segments):
        """Generate a circular flange ring at the specified axial position."""
        p = self.params
        if not p.flanged:
            return

        fw = p.flange_width
        ft = p.flange_thickness
        flange_r = inner_radius + fw

        if is_inlet:
            z_back = z_offset - ft
            z_front = z_offset
        else:
            z_back = z_offset
            z_front = z_offset + ft

        back_normal = [-d for d in direction]
        front_normal = list(direction)

        # Outer cylinder of flange
        base_idx = len(vertices)
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            local_x = flange_r * np.cos(theta)
            local_y = flange_r * np.sin(theta)

            pt_back = self._transform_point(local_x, local_y, z_back,
                                            direction, perp1, perp2)
            pt_front = self._transform_point(local_x, local_y, z_front,
                                             direction, perp1, perp2)

            radial_n = [
                np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0],
                np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1],
                np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2],
            ]

            vertices.append(pt_back)
            normals.append(radial_n)
            vertices.append(pt_front)
            normals.append(radial_n)

        for i in range(num_segments):
            i0 = base_idx + i * 2
            i1 = base_idx + i * 2 + 1
            i2 = base_idx + ((i + 1) % num_segments) * 2
            i3 = base_idx + ((i + 1) % num_segments) * 2 + 1
            indices.extend([i0, i2, i1])
            indices.extend([i1, i2, i3])

        # Annular faces (back and front)
        for face_z, face_normal in [(z_back, back_normal), (z_front, front_normal)]:
            face_base = len(vertices)

            # Inner ring
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = inner_radius * np.cos(theta)
                local_y = inner_radius * np.sin(theta)
                pt = self._transform_point(local_x, local_y, face_z,
                                           direction, perp1, perp2)
                vertices.append(pt)
                normals.append(face_normal)

            # Outer ring
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = flange_r * np.cos(theta)
                local_y = flange_r * np.sin(theta)
                pt = self._transform_point(local_x, local_y, face_z,
                                           direction, perp1, perp2)
                vertices.append(pt)
                normals.append(face_normal)

            for i in range(num_segments):
                i0 = face_base + i
                i1 = face_base + (i + 1) % num_segments
                i2 = face_base + num_segments + i
                i3 = face_base + num_segments + (i + 1) % num_segments

                if np.allclose(face_z, z_front):
                    indices.extend([i0, i1, i2])
                    indices.extend([i1, i3, i2])
                else:
                    indices.extend([i0, i2, i1])
                    indices.extend([i1, i2, i3])

    def _generate_rectangular_flange(self, vertices, indices, normals,
                                     inner_width, inner_height,
                                     z_offset, is_inlet,
                                     direction, perp1, perp2):
        """Generate a rectangular flange at the specified axial position."""
        p = self.params
        if not p.flanged:
            return

        fw = p.flange_width
        ft = p.flange_thickness

        outer_w = inner_width + fw
        outer_h = inner_height + fw

        if is_inlet:
            z_back = z_offset - ft
            z_front = z_offset
        else:
            z_back = z_offset
            z_front = z_offset + ft

        back_normal = [-d for d in direction]
        front_normal = list(direction)

        inner_corners = [
            (inner_width, inner_height),
            (-inner_width, inner_height),
            (-inner_width, -inner_height),
            (inner_width, -inner_height),
        ]
        outer_corners = [
            (outer_w, outer_h),
            (-outer_w, outer_h),
            (-outer_w, -outer_h),
            (outer_w, -outer_h),
        ]
        edge_normals = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # Outer edge faces (4 sides)
        for i in range(4):
            c1 = outer_corners[i]
            c2 = outer_corners[(i + 1) % 4]
            enx, eny = edge_normals[i]

            norm = [
                enx * perp1[0] + eny * perp2[0],
                enx * perp1[1] + eny * perp2[1],
                enx * perp1[2] + eny * perp2[2],
            ]

            side_base = len(vertices)
            pt1 = self._transform_point(c1[0], c1[1], z_back, direction, perp1, perp2)
            pt2 = self._transform_point(c2[0], c2[1], z_back, direction, perp1, perp2)
            pt3 = self._transform_point(c1[0], c1[1], z_front, direction, perp1, perp2)
            pt4 = self._transform_point(c2[0], c2[1], z_front, direction, perp1, perp2)

            vertices.extend([pt1, pt2, pt3, pt4])
            normals.extend([norm, norm, norm, norm])

            indices.extend([side_base, side_base + 1, side_base + 2])
            indices.extend([side_base + 1, side_base + 3, side_base + 2])

        # Annular faces (back and front)
        for face_z, face_normal in [(z_back, back_normal), (z_front, front_normal)]:
            face_base = len(vertices)

            for c in inner_corners:
                pt = self._transform_point(c[0], c[1], face_z, direction, perp1, perp2)
                vertices.append(pt)
                normals.append(face_normal)

            for c in outer_corners:
                pt = self._transform_point(c[0], c[1], face_z, direction, perp1, perp2)
                vertices.append(pt)
                normals.append(face_normal)

            for i in range(4):
                i0 = face_base + i
                i1 = face_base + (i + 1) % 4
                i2 = face_base + 4 + i
                i3 = face_base + 4 + (i + 1) % 4

                if np.allclose(face_z, z_front):
                    indices.extend([i0, i1, i2])
                    indices.extend([i1, i3, i2])
                else:
                    indices.extend([i0, i2, i1])
                    indices.extend([i1, i2, i3])

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet (expansion body bottom)."""
        return self.params.center

    def get_outlet_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet (expansion body top)."""
        cx, cy, cz = self.params.center
        dx, dy, dz = self.params.direction_normalized
        L = self.params.transition_length
        return (cx + L * dx, cy + L * dy, cz + L * dz)

    def get_hopper_discharge_position(self) -> Tuple[float, float, float]:
        """Get center position of hopper discharge opening."""
        cx, cy, cz = self.params.center
        dx, dy, dz = self.params.direction_normalized
        H = self.params.hopper_height
        return (cx - H * dx, cy - H * dy, cz - H * dz)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box (min, max) corners."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)

    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object from the geometry."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required for mesh creation")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32),
        )


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------

def create_standard_expanding_transition(
    inlet_diameter: float = 0.072,
    outlet_width: float = 0.120,
    outlet_depth: float = 0.200,
    transition_length: float = None,
    hopper_height: float = 0.15,
    hopper_outlet_diameter: float = 0.04,
    **kwargs,
) -> ExpandingTransitionWithDropout:
    """
    Create a standard expanding transition with dropout hopper.

    Default dimensions match typical venturi-to-zigzag connection:
    72mm round inlet -> 120x200mm rectangular outlet.

    Args:
        inlet_diameter: Round inlet diameter [m]
        outlet_width: Rectangular outlet width [m]
        outlet_depth: Rectangular outlet depth [m]
        transition_length: Expansion length [m] (auto-calculated if None)
        hopper_height: Hopper cone height [m]
        hopper_outlet_diameter: Hopper discharge diameter [m]
        **kwargs: Additional ExpandingTransitionParams fields

    Returns:
        ExpandingTransitionWithDropout instance
    """
    if transition_length is None:
        max_dim = max(outlet_width, outlet_depth)
        transition_length = (max_dim - inlet_diameter) / (2 * np.tan(np.radians(12)))
        transition_length = max(transition_length, 0.1)

    params = ExpandingTransitionParams(
        inlet_diameter=inlet_diameter,
        outlet_width=outlet_width,
        outlet_depth=outlet_depth,
        transition_length=transition_length,
        hopper_height=hopper_height,
        hopper_outlet_diameter=hopper_outlet_diameter,
        **kwargs,
    )
    return ExpandingTransitionWithDropout(params)
