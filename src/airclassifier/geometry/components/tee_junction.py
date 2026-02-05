"""
Pipe tee junction (T-junction) for flow split or merge.

A main pipe with a branch pipe departing at 90 degrees. Used for:
- Splitting flow before the venturi (bypass tee)
- Merging bypass flow back into the fines path (merge tee)

The tee has three ports: inlet, outlet (main through-flow), and branch.
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
class TeeJunctionParams:
    """
    Parameters for pipe tee junction.

    Attributes:
        main_diameter: Main (through) pipe diameter [m]
        branch_diameter: Branch pipe diameter [m]
        main_length: Length of main pipe body [m]
        branch_stub_length: Length of branch stub [m]
        wall_thickness: Wall thickness [m]
        center: Center position (x, y, z) [m] — at junction center
        main_direction: Main flow direction vector
        branch_direction: Branch departure direction vector (should be perpendicular)
        flanged: Whether to add flanges
        flange_width: Width of flange beyond pipe [m]
        flange_thickness: Thickness of flange [m]
        resolution: Number of circumferential segments
    """
    main_diameter: float
    branch_diameter: float
    main_length: float = 0.10
    branch_stub_length: float = 0.05
    wall_thickness: float = 0.002
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    main_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    branch_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    flanged: bool = True
    flange_width: float = 0.02
    flange_thickness: float = 0.008
    resolution: int = 24

    @property
    def main_direction_normalized(self) -> np.ndarray:
        d = np.array(self.main_direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return d

    @property
    def branch_direction_normalized(self) -> np.ndarray:
        d = np.array(self.branch_direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return d


class TeeJunction:
    """
    Pipe tee junction with main through-pipe and perpendicular branch.

    The main pipe runs along main_direction, centered at the origin.
    The branch pipe departs perpendicular from the junction center.

    Ports:
        inlet: Main pipe inlet (upstream)
        outlet: Main pipe outlet (downstream, through)
        branch: Branch pipe outlet
    """

    def __init__(self, params: TeeJunctionParams):
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
        md = p.main_direction_normalized
        bd = p.branch_direction_normalized
        half_L = p.main_length / 2

        return {
            'inlet': ConnectionPort(
                position=(
                    cx - half_L * md[0],
                    cy - half_L * md[1],
                    cz - half_L * md[2],
                ),
                direction=(-md[0], -md[1], -md[2]),
                diameter=p.main_diameter,
                port_type=PortType.CIRCULAR,
                name="inlet",
            ),
            'outlet': ConnectionPort(
                position=(
                    cx + half_L * md[0],
                    cy + half_L * md[1],
                    cz + half_L * md[2],
                ),
                direction=(md[0], md[1], md[2]),
                diameter=p.main_diameter,
                port_type=PortType.CIRCULAR,
                name="outlet",
            ),
            'branch': ConnectionPort(
                position=(
                    cx + p.branch_stub_length * bd[0],
                    cy + p.branch_stub_length * bd[1],
                    cz + p.branch_stub_length * bd[2],
                ),
                direction=(bd[0], bd[1], bd[2]),
                diameter=p.branch_diameter,
                port_type=PortType.CIRCULAR,
                name="branch",
            ),
        }

    # ------------------------------------------------------------------
    # Coordinate system helpers
    # ------------------------------------------------------------------

    def _get_main_basis(self):
        """Get orthonormal basis for the main pipe axis."""
        md = self.params.main_direction_normalized
        bd = self.params.branch_direction_normalized
        # perp1 = branch direction, perp2 = cross(main, branch)
        perp1 = bd.copy()
        perp2 = np.cross(md, perp1)
        norm = np.linalg.norm(perp2)
        if norm > 0:
            perp2 = perp2 / norm
        return md, perp1, perp2

    def _get_branch_basis(self):
        """Get orthonormal basis for the branch pipe axis."""
        md = self.params.main_direction_normalized
        bd = self.params.branch_direction_normalized
        # For the branch: direction=bd, perp1=md, perp2=cross(bd, md)
        perp1 = md.copy()
        perp2 = np.cross(bd, perp1)
        norm = np.linalg.norm(perp2)
        if norm > 0:
            perp2 = perp2 / norm
        return bd, perp1, perp2

    def _transform_point(self, local_x, local_y, z_offset,
                         direction, perp1, perp2, origin=None):
        """Transform local coordinates to world coordinates."""
        if origin is None:
            cx, cy, cz = self.params.center
        else:
            cx, cy, cz = origin
        return [
            cx + local_x * perp1[0] + local_y * perp2[0] + z_offset * direction[0],
            cy + local_x * perp1[1] + local_y * perp2[1] + z_offset * direction[1],
            cz + local_x * perp1[2] + local_y * perp2[2] + z_offset * direction[2],
        ]

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def generate_mesh(self):
        """Generate tee junction mesh: main pipe + branch stub."""
        p = self.params
        n_seg = p.resolution

        all_v: list = []
        all_i: list = []
        all_n: list = []

        md, m_perp1, m_perp2 = self._get_main_basis()
        bd, b_perp1, b_perp2 = self._get_branch_basis()

        # Main pipe: cylinder from -L/2 to +L/2 along main_direction
        self._generate_cylinder(
            all_v, all_i, all_n,
            radius=p.main_diameter / 2,
            length=p.main_length,
            direction=md, perp1=m_perp1, perp2=m_perp2,
            origin=np.array(self.params.center) - (p.main_length / 2) * md,
            n_seg=n_seg,
            cap_start=True, cap_end=True,
        )

        # Branch stub: cylinder from junction center outward along branch_direction
        self._generate_cylinder(
            all_v, all_i, all_n,
            radius=p.branch_diameter / 2,
            length=p.branch_stub_length,
            direction=bd, perp1=b_perp1, perp2=b_perp2,
            origin=np.array(self.params.center),
            n_seg=n_seg,
            cap_start=False, cap_end=True,
        )

        # Flanges at all three openings
        if p.flanged:
            half_L = p.main_length / 2
            inlet_origin = np.array(self.params.center) - half_L * md

            # Inlet flange
            self._generate_flange_ring(
                all_v, all_i, all_n,
                inner_radius=p.main_diameter / 2 + p.wall_thickness,
                z_offset=0.0,
                is_inlet=True,
                direction=md, perp1=m_perp1, perp2=m_perp2,
                origin=inlet_origin,
                num_segments=n_seg,
            )
            # Outlet flange
            self._generate_flange_ring(
                all_v, all_i, all_n,
                inner_radius=p.main_diameter / 2 + p.wall_thickness,
                z_offset=p.main_length,
                is_inlet=False,
                direction=md, perp1=m_perp1, perp2=m_perp2,
                origin=inlet_origin,
                num_segments=n_seg,
            )
            # Branch flange
            self._generate_flange_ring(
                all_v, all_i, all_n,
                inner_radius=p.branch_diameter / 2 + p.wall_thickness,
                z_offset=p.branch_stub_length,
                is_inlet=False,
                direction=bd, perp1=b_perp1, perp2=b_perp2,
                origin=np.array(self.params.center),
                num_segments=n_seg,
            )

        self._vertices = np.array(all_v, dtype=np.float32)
        self._indices = np.array(all_i, dtype=np.int32)
        self._normals = np.array(all_n, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_cylinder(self, vertices, indices, normals,
                           radius, length, direction, perp1, perp2,
                           origin, n_seg, cap_start=True, cap_end=True):
        """Generate a cylindrical pipe section (outer + inner walls)."""
        t = self.params.wall_thickness
        n_div = 2  # Simple cylinder, 2 divisions along length

        # --- Outer surface ---
        for layer in range(n_div + 1):
            t_param = layer / n_div
            z_pos = t_param * length

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = (radius + t) * np.cos(theta)
                local_y = (radius + t) * np.sin(theta)

                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2,
                                           origin=tuple(origin))
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
            z_pos = t_param * length

            base_idx = len(vertices)

            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)

                pt = self._transform_point(local_x, local_y, z_pos,
                                           direction, perp1, perp2,
                                           origin=tuple(origin))
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

        # --- End caps (annular rings) ---
        if cap_start:
            self._add_annular_cap(
                vertices, indices, normals,
                r_inner=radius, r_outer=radius + t,
                z_off=0.0, normal_positive=False,
                direction=direction, perp1=perp1, perp2=perp2,
                origin=origin, n_seg=n_seg,
            )
        if cap_end:
            self._add_annular_cap(
                vertices, indices, normals,
                r_inner=radius, r_outer=radius + t,
                z_off=length, normal_positive=True,
                direction=direction, perp1=perp1, perp2=perp2,
                origin=origin, n_seg=n_seg,
            )

    def _add_annular_cap(self, vertices, indices, normals,
                         r_inner, r_outer, z_off, normal_positive,
                         direction, perp1, perp2, origin, n_seg):
        """Add an annular end cap."""
        base_idx = len(vertices)
        norm = list(direction) if normal_positive else [-d for d in direction]

        for radius in [r_inner, r_outer]:
            for i in range(n_seg):
                theta = 2 * np.pi * i / n_seg
                local_x = radius * np.cos(theta)
                local_y = radius * np.sin(theta)
                pt = self._transform_point(local_x, local_y, z_off,
                                           direction, perp1, perp2,
                                           origin=tuple(origin))
                vertices.append(pt)
                normals.append(norm)

        for i in range(n_seg):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % n_seg
            i2 = base_idx + n_seg + i
            i3 = base_idx + n_seg + (i + 1) % n_seg

            if normal_positive:
                indices.extend([i0, i1, i2])
                indices.extend([i1, i3, i2])
            else:
                indices.extend([i0, i2, i1])
                indices.extend([i1, i2, i3])

    def _generate_flange_ring(self, vertices, indices, normals,
                              inner_radius, z_offset, is_inlet,
                              direction, perp1, perp2, origin,
                              num_segments):
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
                                            direction, perp1, perp2,
                                            origin=tuple(origin))
            pt_front = self._transform_point(local_x, local_y, z_front,
                                             direction, perp1, perp2,
                                             origin=tuple(origin))

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
                                           direction, perp1, perp2,
                                           origin=tuple(origin))
                vertices.append(pt)
                normals.append(face_normal)

            # Outer ring
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                local_x = flange_r * np.cos(theta)
                local_y = flange_r * np.sin(theta)
                pt = self._transform_point(local_x, local_y, face_z,
                                           direction, perp1, perp2,
                                           origin=tuple(origin))
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

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

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
