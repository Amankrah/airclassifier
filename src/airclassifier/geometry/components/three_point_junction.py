"""
Three-point pipe junction for wheel-only classification.

Geometry:
- 90° elbow from +Y (air, vertical from above) curving DOWN to +X (wheel, horizontal).
  Bottom closed (no open port at -Y).
- Solids inlet: circular cut on the upper part of the elbow with 15° stub welded.

Air enters from above, curves downward through the elbow, exits horizontally to the wheel.
Feed solids enter via the angled stub on the outer surface of the elbow.

Used when use_preclassification=False to merge air and solids before the wheel.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from ..connection_ports import ConnectionPort, PortType


@dataclass
class ThreePointJunctionParams:
    """
    Parameters for the three-point junction (90° elbow + 15° solids branch).

    Attributes:
        air_diameter: Air leg (Y) diameter [m]
        solids_diameter: Solids chute diameter [m]
        wheel_diameter: Wheel outlet (X) diameter [m]
        stub_length: Length of each leg from bend [m]
        bend_radius: Centerline bend radius of the 90° elbow [m]
        solids_angle_deg: Angle of solids chute from vertical (default 15°) [deg]
        wall_thickness: Wall thickness [m]
        center: Junction center (bend center) (x, y, z) [m]
        resolution: Circumferential segments per pipe
    """
    air_diameter: float
    solids_diameter: float
    wheel_diameter: float
    stub_length: float = 0.06
    bend_radius: float = 0.05
    solids_angle_deg: float = 15.0
    wall_thickness: float = 0.002
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution: int = 24


class ThreePointJunction:
    """
    Three-point connector: 90° elbow (+Y air -> +X wheel), bottom closed;
    air enters from above, curves down to horizontal toward wheel.
    Solids inlet via circular cut on the upper elbow with 15° welded stub.

    Ports:
        air:    (0, +1, 0)   - top of vertical leg (air enters from above)
        solids: (0, sin(15°), cos(15°)) - end of 15° stub on upper elbow
        wheel:  (1, 0, 0)    - end of horizontal leg
    """

    def __init__(self, params: ThreePointJunctionParams):
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
        r_tube = max(p.air_diameter, p.wheel_diameter) / 2
        R = max(p.bend_radius, r_tube + 0.015)
        L = p.stub_length
        angle_rad = np.radians(p.solids_angle_deg)
        L_leg = max(0.001, p.stub_length - R)
        # Air port: top of vertical leg (air enters from above, curving down to wheel)
        air_pos = (cx, cy + R + L_leg, cz)
        # Wheel port: end of horizontal leg
        wheel_pos = (cx + R + L_leg, cy, cz)
        # Solids port: end of 15° stub (on upper part of elbow, outer side toward +Z)
        mid = 0.5 ** 0.5
        r_tube = max(p.air_diameter, p.wheel_diameter) / 2 + p.wall_thickness
        hole_center = np.array([cx + R * mid, cy + R * mid, cz + r_tube])
        dir_solids = np.array([0.0, np.sin(angle_rad), np.cos(angle_rad)])
        solids_pos = tuple(hole_center + L * dir_solids)
        return {
            'air': ConnectionPort(
                position=air_pos,
                direction=(0.0, 1.0, 0.0),
                diameter=p.air_diameter,
                port_type=PortType.CIRCULAR,
                name="air",
            ),
            'solids': ConnectionPort(
                position=solids_pos,
                direction=tuple(dir_solids),
                diameter=p.solids_diameter,
                port_type=PortType.CIRCULAR,
                name="solids",
            ),
            'wheel': ConnectionPort(
                position=wheel_pos,
                direction=(1.0, 0.0, 0.0),
                diameter=p.wheel_diameter,
                port_type=PortType.CIRCULAR,
                name="wheel",
            ),
        }

    def _direction_basis(self, direction: np.ndarray):
        d = direction / (np.linalg.norm(direction) + 1e-12)
        if abs(d[1]) < 0.9:
            perp1 = np.array([0, 1, 0])
        else:
            perp1 = np.array([1, 0, 0])
        perp1 = perp1 - np.dot(perp1, d) * d
        n = np.linalg.norm(perp1)
        if n > 0:
            perp1 = perp1 / n
        perp2 = np.cross(d, perp1)
        n = np.linalg.norm(perp2)
        if n > 0:
            perp2 = perp2 / n
        return d, perp1, perp2

    def _transform(self, local_x: float, local_y: float, z_off: float,
                   direction: np.ndarray, perp1: np.ndarray, perp2: np.ndarray,
                   origin: np.ndarray) -> Tuple[float, float, float]:
        cx, cy, cz = origin[0], origin[1], origin[2]
        return (
            cx + local_x * perp1[0] + local_y * perp2[0] + z_off * direction[0],
            cy + local_x * perp1[1] + local_y * perp2[1] + z_off * direction[1],
            cz + local_x * perp1[2] + local_y * perp2[2] + z_off * direction[2],
        )

    def _add_cylinder(self, all_v: list, all_i: list, all_n: list,
                      radius: float, length: float, direction: np.ndarray,
                      origin: np.ndarray, cap_start: bool, cap_end: bool) -> None:
        """Add cylinder (inner + outer) from origin along direction."""
        p = self.params
        t = p.wall_thickness
        n_seg = p.resolution
        d, perp1, perp2 = self._direction_basis(direction)
        n_div = 2
        for layer in range(n_div + 1):
            z_pos = (layer / n_div) * length
            base_idx = len(all_v)
            for i in range(n_seg):
                th = 2 * np.pi * i / n_seg
                lx = (radius + t) * np.cos(th)
                ly = (radius + t) * np.sin(th)
                pt = self._transform(lx, ly, z_pos, d, perp1, perp2, origin)
                all_v.append(list(pt))
                nx = np.cos(th) * perp1[0] + np.sin(th) * perp2[0]
                ny = np.cos(th) * perp1[1] + np.sin(th) * perp2[1]
                nz = np.cos(th) * perp1[2] + np.sin(th) * perp2[2]
                all_n.append([nx, ny, nz])
            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0, i1 = prev_base + i, prev_base + (i + 1) % n_seg
                    i2, i3 = base_idx + i, base_idx + (i + 1) % n_seg
                    all_i.extend([i0, i2, i1])
                    all_i.extend([i1, i2, i3])
        for layer in range(n_div + 1):
            z_pos = (layer / n_div) * length
            base_idx = len(all_v)
            for i in range(n_seg):
                th = 2 * np.pi * i / n_seg
                lx, ly = radius * np.cos(th), radius * np.sin(th)
                pt = self._transform(lx, ly, z_pos, d, perp1, perp2, origin)
                all_v.append(list(pt))
                nx = np.cos(th) * perp1[0] + np.sin(th) * perp2[0]
                ny = np.cos(th) * perp1[1] + np.sin(th) * perp2[1]
                nz = np.cos(th) * perp1[2] + np.sin(th) * perp2[2]
                all_n.append([-nx, -ny, -nz])
            if layer > 0:
                prev_base = base_idx - n_seg
                for i in range(n_seg):
                    i0, i1 = prev_base + i, prev_base + (i + 1) % n_seg
                    i2, i3 = base_idx + i, base_idx + (i + 1) % n_seg
                    all_i.extend([i0, i1, i2])
                    all_i.extend([i1, i3, i2])
        if cap_start:
            base_idx = len(all_v)
            norm = list(-d)
            for r in [radius, radius + t]:
                for i in range(n_seg):
                    th = 2 * np.pi * i / n_seg
                    pt = self._transform(r * np.cos(th), r * np.sin(th), 0.0, d, perp1, perp2, origin)
                    all_v.append(list(pt))
                    all_n.append(norm)
            for i in range(n_seg):
                i0, i1 = base_idx + i, base_idx + (i + 1) % n_seg
                i2, i3 = base_idx + n_seg + i, base_idx + n_seg + (i + 1) % n_seg
                all_i.extend([i0, i2, i1])
                all_i.extend([i1, i2, i3])
        if cap_end:
            base_idx = len(all_v)
            norm = list(d)
            for r in [radius, radius + t]:
                for i in range(n_seg):
                    th = 2 * np.pi * i / n_seg
                    pt = self._transform(r * np.cos(th), r * np.sin(th), length, d, perp1, perp2, origin)
                    all_v.append(list(pt))
                    all_n.append(norm)
            for i in range(n_seg):
                i0, i1 = base_idx + i, base_idx + (i + 1) % n_seg
                i2, i3 = base_idx + n_seg + i, base_idx + n_seg + (i + 1) % n_seg
                all_i.extend([i0, i1, i2])
                all_i.extend([i1, i3, i2])

    def _add_quarter_torus(self, all_v: list, all_i: list, all_n: list,
                          R: float, r_tube: float, center: np.ndarray) -> None:
        """90° torus bend in XY plane: from +Y curving DOWN to +X.
        Air enters from above (+Y), curves downward through the elbow, exits horizontal (+X).
        R must be > r_tube for a valid elbow (centerline radius of the bend).
        """
        p = self.params
        t = p.wall_thickness
        n_seg = p.resolution
        n_arc = max(12, n_seg)  # Enough segments for a smooth visible elbow
        cx, cy, cz = center[0], center[1], center[2]
        for surf, sign_n in [(r_tube + t, 1), (r_tube, -1)]:
            base_idx = len(all_v)
            for i in range(n_arc + 1):
                theta = 0.5 * np.pi * i / n_arc
                # Sweep from +Y (theta=0) to +X (theta=π/2)
                xc = cx + R * np.sin(theta)
                yc = cy + R * np.cos(theta)
                for j in range(n_seg + 1):
                    phi = 2 * np.pi * j / n_seg
                    dx = np.sin(theta) * np.cos(phi)
                    dy = np.cos(theta) * np.cos(phi)
                    dz = np.sin(phi)
                    pt = [xc + surf * dx, yc + surf * dy, cz + surf * dz]
                    all_v.append(pt)
                    all_n.append([sign_n * dx, sign_n * dy, sign_n * dz])
                if i > 0:
                    prev_base = base_idx + (i - 1) * (n_seg + 1)
                    curr_base = base_idx + i * (n_seg + 1)
                    for j in range(n_seg):
                        i0 = prev_base + j
                        i1 = prev_base + (j + 1)
                        i2 = curr_base + j
                        i3 = curr_base + (j + 1)
                        if sign_n == 1:
                            all_i.extend([i0, i2, i1])
                            all_i.extend([i1, i2, i3])
                        else:
                            all_i.extend([i0, i1, i2])
                            all_i.extend([i1, i3, i2])
        # End caps (annular) at air end (theta=0, +Y) and wheel end (theta=π/2, +X)
        for theta, normal_along in [(0.0, [-1.0, 0.0, 0.0]), (0.5 * np.pi, [0.0, -1.0, 0.0])]:
            base_idx = len(all_v)
            xc = cx + R * np.sin(theta)
            yc = cy + R * np.cos(theta)
            for r in [r_tube, r_tube + t]:
                for j in range(n_seg):
                    phi = 2 * np.pi * j / n_seg
                    ax = np.sin(theta) * np.cos(phi)
                    ay = np.cos(theta) * np.cos(phi)
                    az = np.sin(phi)
                    pt = [xc + r * ax, yc + r * ay, cz + r * az]
                    all_v.append(pt)
                    all_n.append(normal_along)
                for j in range(n_seg):
                    i0 = base_idx + j
                    i1 = base_idx + (j + 1) % n_seg
                    i2 = base_idx + n_seg + j
                    i3 = base_idx + n_seg + (j + 1) % n_seg
                    if theta == 0:
                        all_i.extend([i0, i2, i1])
                        all_i.extend([i1, i2, i3])
                    else:
                        all_i.extend([i0, i1, i2])
                        all_i.extend([i1, i3, i2])
            base_idx += 2 * n_seg

    def generate_mesh(self):
        """90° elbow (+Y->+X, bottom closed) + solids stub at 15° welded to upper elbow."""
        p = self.params
        all_v: list = []
        all_i: list = []
        all_n: list = []
        center = np.array(p.center, dtype=np.float64)
        r_air = p.air_diameter / 2
        r_wheel = p.wheel_diameter / 2
        r_tube = max(p.air_diameter, p.wheel_diameter) / 2
        # Bend radius must exceed tube radius so the 90° torus elbow is valid and visible
        R = max(p.bend_radius, r_tube + 0.015)
        L_leg = max(0.001, p.stub_length - R)
        L = p.stub_length
        angle_rad = np.radians(p.solids_angle_deg)

        # --- Air leg: vertical cylinder from bend upward (air enters from above) ---
        bend_inlet = center + np.array([0.0, R, 0.0])
        self._add_cylinder(
            all_v, all_i, all_n,
            radius=r_air,
            length=L_leg,
            direction=np.array([0.0, 1.0, 0.0]),
            origin=bend_inlet,
            cap_start=False,
            cap_end=True,
        )

        # --- 90° bend (quarter torus): +Y curving DOWN to +X, bottom closed ---
        self._add_quarter_torus(all_v, all_i, all_n, R, r_tube, center)

        # --- Wheel leg: horizontal cylinder from bend outlet toward +X ---
        bend_outlet = center + np.array([R, 0.0, 0.0])
        self._add_cylinder(
            all_v, all_i, all_n,
            radius=r_wheel,
            length=L_leg,
            direction=np.array([1.0, 0.0, 0.0]),
            origin=bend_outlet,
            cap_start=False,
            cap_end=True,
        )

        # --- Solids: circular cut on upper elbow (outer side toward +Z) with 15° welded stub ---
        mid = 0.5 ** 0.5
        hole_center = center + np.array([R * mid, R * mid, r_tube + p.wall_thickness])
        dir_solids = np.array([0.0, np.sin(angle_rad), np.cos(angle_rad)])
        self._add_cylinder(
            all_v, all_i, all_n,
            radius=p.solids_diameter / 2,
            length=L,
            direction=dir_solids,
            origin=hole_center,
            cap_start=False,
            cap_end=True,
        )

        self._vertices = np.array(all_v, dtype=np.float32)
        self._indices = np.array(all_i, dtype=np.int32)
        self._normals = np.array(all_n, dtype=np.float32)
        return self._vertices, self._indices, self._normals
