"""
Centrifugal blower/fan component for air supply.

The centrifugal blower provides the motive force for the air classification
system. Air enters axially through the inlet eye and is accelerated radially
by the impeller, exiting through the scroll/volute casing.

Principle:
- Impeller imparts kinetic energy to air
- Scroll casing converts velocity to static pressure
- Backward-curved blades offer best efficiency (75-85%)
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI


@dataclass
class CentrifugalBlowerParams:
    """Parameters for centrifugal blower/fan."""

    # Impeller geometry
    impeller_diameter: float     # [m] Impeller outer diameter
    impeller_width: float        # [m] Impeller width at outlet
    inlet_diameter: float        # [m] Inlet eye diameter
    hub_diameter: float          # [m] Hub diameter
    num_blades: int              # Number of impeller blades (6-12 typical)
    blade_type: str = "backward_curved"  # "backward_curved", "radial", "forward_curved"
    blade_angle_inlet: float = 30.0   # [deg] Blade angle at inlet
    blade_angle_outlet: float = 45.0  # [deg] Blade angle at outlet (for backward curved)

    # Scroll/volute geometry
    scroll_clearance: float = 0.02    # [m] Gap between impeller and scroll
    scroll_expansion: float = 1.5     # Scroll expansion ratio (outlet/cutoff)

    # Outlet
    outlet_width: float = None   # [m] Outlet width (defaults to impeller_width)
    outlet_height: float = None  # [m] Outlet height (calculated from area)

    # Operating parameters
    rpm: float = 3000.0          # [rpm] Operating speed
    flow_rate: float = 3000.0    # [m³/h] Design flow rate
    pressure_rise: float = 5000.0  # [Pa] Total pressure rise

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Center of impeller

    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 16
    resolution_scroll: int = 36

    def __post_init__(self):
        if self.outlet_width is None:
            self.outlet_width = self.impeller_width
        if self.outlet_height is None:
            # Calculate from continuity (roughly)
            self.outlet_height = self.impeller_width * 1.2

    @property
    def impeller_radius(self) -> float:
        """Impeller outer radius."""
        return self.impeller_diameter / 2

    @property
    def inlet_radius(self) -> float:
        """Inlet eye radius."""
        return self.inlet_diameter / 2

    @property
    def hub_radius(self) -> float:
        """Hub radius."""
        return self.hub_diameter / 2

    @property
    def scroll_inner_radius(self) -> float:
        """Inner radius of scroll at cutoff."""
        return self.impeller_radius + self.scroll_clearance

    @property
    def tip_speed(self) -> float:
        """Impeller tip speed [m/s]."""
        return PI * self.impeller_diameter * self.rpm / 60

    @property
    def specific_speed(self) -> float:
        """Specific speed (dimensionless)."""
        # Ns = N * sqrt(Q) / H^0.75
        # where N = rpm, Q = m³/s, H = m of head
        Q = self.flow_rate / 3600  # m³/s
        H = self.pressure_rise / (1.2 * 9.81)  # m of air (approx)
        return self.rpm * np.sqrt(Q) / (H ** 0.75)

    @property
    def estimated_efficiency(self) -> float:
        """Estimated efficiency based on blade type."""
        if self.blade_type == "backward_curved":
            return 0.80
        elif self.blade_type == "radial":
            return 0.70
        else:  # forward_curved
            return 0.65

    @property
    def shaft_power(self) -> float:
        """Estimated shaft power [W]."""
        Q = self.flow_rate / 3600  # m³/s
        return (Q * self.pressure_rise) / self.estimated_efficiency


class CentrifugalBlower:
    """
    Centrifugal blower/fan for air supply.

    Components:
    - Impeller with blades
    - Scroll/volute casing
    - Inlet cone
    - Outlet duct

    Coordinate system:
    - Origin at center of impeller
    - Inlet along +Z axis
    - Outlet along +X axis (standard arrangement)
    """

    def __init__(self, params: CentrifugalBlowerParams):
        """
        Initialize centrifugal blower.

        Args:
            params: CentrifugalBlowerParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the centrifugal blower.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        # Generate scroll/volute casing
        self._generate_scroll(vertices, indices, normals)

        # Generate impeller
        self._generate_impeller(vertices, indices, normals)

        # Generate inlet cone
        self._generate_inlet(vertices, indices, normals)

        # Generate outlet duct
        self._generate_outlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _generate_scroll(self, vertices: List, indices: List, normals: List):
        """Generate scroll/volute casing."""
        p = self.params
        n_scroll = p.resolution_scroll
        n_axial = p.resolution_axial // 2

        start_idx = len(vertices)

        # Scroll is a spiral that expands as it goes around
        # Start from cutoff (smallest radius) and expand to outlet
        r_cutoff = p.scroll_inner_radius
        r_max = r_cutoff * p.scroll_expansion

        half_width = p.impeller_width / 2 * 1.2  # Slightly wider than impeller

        # Generate scroll surface
        for i in range(n_scroll + 1):
            # Angle from 0 to 360 degrees (full rotation)
            theta = (i / n_scroll) * TWO_PI

            # Radius increases with angle (logarithmic spiral)
            r = r_cutoff + (r_max - r_cutoff) * (i / n_scroll)

            for j in range(n_axial + 1):
                # Axial position
                t_axial = j / n_axial
                z = p.center[2] - half_width + t_axial * 2 * half_width

                x = p.center[0] + r * np.cos(theta)
                y = p.center[1] + r * np.sin(theta)

                vertices.append([x, y, z])
                # Normal pointing outward radially
                normals.append([np.cos(theta), np.sin(theta), 0.0])

        # Generate triangles for scroll
        n_circ = n_axial + 1
        for i in range(n_scroll):
            for j in range(n_axial):
                v0 = start_idx + i * n_circ + j
                v1 = start_idx + i * n_circ + j + 1
                v2 = start_idx + (i + 1) * n_circ + j + 1
                v3 = start_idx + (i + 1) * n_circ + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Add side walls of scroll (front and back plates)
        self._generate_scroll_sidewalls(vertices, indices, normals)

    def _generate_scroll_sidewalls(self, vertices: List, indices: List, normals: List):
        """Generate front and back plates of scroll casing."""
        p = self.params
        n_radial = p.resolution_radial

        half_width = p.impeller_width / 2 * 1.2
        r_inner = p.inlet_radius
        r_outer = p.scroll_inner_radius * p.scroll_expansion

        for side in [-1, 1]:
            start_idx = len(vertices)
            z = p.center[2] + side * half_width

            # Generate annular ring (inner to outer)
            for i in range(2):
                r = r_inner if i == 0 else r_outer
                for j in range(n_radial):
                    theta = (j / n_radial) * TWO_PI
                    x = p.center[0] + r * np.cos(theta)
                    y = p.center[1] + r * np.sin(theta)
                    vertices.append([x, y, z])
                    normals.append([0.0, 0.0, side])

            # Triangles for side wall
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + j
                v1 = start_idx + j_next
                v2 = start_idx + n_radial + j_next
                v3 = start_idx + n_radial + j

                if side > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])

    def _generate_impeller(self, vertices: List, indices: List, normals: List):
        """Generate impeller with blades."""
        p = self.params
        n_radial = p.resolution_radial

        # Hub cylinder
        hub_start = len(vertices)
        hub_half_width = p.impeller_width / 2

        for i in range(2):
            z = p.center[2] - hub_half_width + i * p.impeller_width
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + p.hub_radius * np.cos(theta)
                y = p.center[1] + p.hub_radius * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([np.cos(theta), np.sin(theta), 0.0])

        # Hub triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = hub_start + j
            v1 = hub_start + j_next
            v2 = hub_start + n_radial + j_next
            v3 = hub_start + n_radial + j

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Generate blades
        self._generate_blades(vertices, indices, normals)

        # Shroud (back plate of impeller)
        self._generate_shroud(vertices, indices, normals)

    def _generate_blades(self, vertices: List, indices: List, normals: List):
        """Generate impeller blades."""
        p = self.params
        n_blade_segments = 8  # Segments along blade length

        blade_thickness = 0.003  # 3mm blade thickness
        half_width = p.impeller_width / 2

        for blade in range(p.num_blades):
            blade_start_angle = (blade / p.num_blades) * TWO_PI
            blade_start = len(vertices)

            # Blade is a curved surface from hub to tip
            for i in range(n_blade_segments + 1):
                t = i / n_blade_segments
                r = p.hub_radius + t * (p.impeller_radius - p.hub_radius)

                # Blade angle varies with radius (for backward-curved)
                if p.blade_type == "backward_curved":
                    blade_angle = np.radians(p.blade_angle_inlet +
                                            t * (p.blade_angle_outlet - p.blade_angle_inlet))
                    # Blade curves backward
                    angle_offset = t * blade_angle
                elif p.blade_type == "radial":
                    angle_offset = 0
                else:  # forward_curved
                    blade_angle = np.radians(p.blade_angle_outlet)
                    angle_offset = -t * blade_angle

                theta = blade_start_angle + angle_offset

                for side in [-1, 1]:
                    theta_side = theta + side * blade_thickness / (2 * r)
                    for axial in [-half_width, half_width]:
                        x = p.center[0] + r * np.cos(theta_side)
                        y = p.center[1] + r * np.sin(theta_side)
                        z = p.center[2] + axial

                        vertices.append([x, y, z])
                        # Normal perpendicular to blade
                        nx = -np.sin(theta_side) * side
                        ny = np.cos(theta_side) * side
                        normals.append([nx, ny, 0.0])

            # Triangles for blade (4 vertices per radial segment)
            for i in range(n_blade_segments):
                base = blade_start + i * 4

                # Front face
                indices.extend([base, base + 4, base + 5])
                indices.extend([base, base + 5, base + 1])

                # Back face
                indices.extend([base + 2, base + 3, base + 7])
                indices.extend([base + 2, base + 7, base + 6])

                # Top edge
                indices.extend([base + 1, base + 5, base + 7])
                indices.extend([base + 1, base + 7, base + 3])

                # Bottom edge
                indices.extend([base, base + 2, base + 6])
                indices.extend([base, base + 6, base + 4])

    def _generate_shroud(self, vertices: List, indices: List, normals: List):
        """Generate back shroud/plate of impeller."""
        p = self.params
        n_radial = p.resolution_radial

        shroud_start = len(vertices)
        z = p.center[2] - p.impeller_width / 2 - 0.002  # Slightly behind impeller

        # Annular disk from hub to impeller OD
        for i in range(2):
            r = p.hub_radius if i == 0 else p.impeller_radius
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                y = p.center[1] + r * np.sin(theta)
                vertices.append([x, y, z])
                normals.append([0.0, 0.0, -1.0])

        # Triangles
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = shroud_start + j
            v1 = shroud_start + j_next
            v2 = shroud_start + n_radial + j_next
            v3 = shroud_start + n_radial + j

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _generate_inlet(self, vertices: List, indices: List, normals: List):
        """Generate inlet cone/bell."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial // 2

        inlet_start = len(vertices)

        # Inlet is a conical bell from larger opening to inlet eye
        inlet_bell_length = p.inlet_diameter * 0.5
        inlet_outer_diameter = p.inlet_diameter * 1.3

        z_start = p.center[2] + p.impeller_width / 2 * 1.2
        z_end = z_start + inlet_bell_length

        for i in range(n_axial + 1):
            t = i / n_axial
            z = z_start + t * inlet_bell_length
            r = p.inlet_radius + (inlet_outer_diameter / 2 - p.inlet_radius) * (1 - t)

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                y = p.center[1] + r * np.sin(theta)
                vertices.append([x, y, z])

                # Normal for conical surface
                dr = inlet_outer_diameter / 2 - p.inlet_radius
                slant = np.sqrt(inlet_bell_length ** 2 + dr ** 2)
                n_z = dr / slant
                n_r = inlet_bell_length / slant
                normals.append([n_r * np.cos(theta), n_r * np.sin(theta), n_z])

        # Triangles
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = inlet_start + i * n_radial + j
                v1 = inlet_start + i * n_radial + j_next
                v2 = inlet_start + (i + 1) * n_radial + j_next
                v3 = inlet_start + (i + 1) * n_radial + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

    def _generate_outlet(self, vertices: List, indices: List, normals: List):
        """
        Generate rectangular outlet duct with curled scroll-side edge.

        The scroll-side edge is curved to match the scroll body's cylindrical
        surface for proper welded connection in real-world fabrication.
        The flange-side is flat rectangular for standard connector attachment.
        """
        p = self.params
        n_curl_segments = 8  # Segments for the curved scroll-side edge

        outlet_start = len(vertices)

        # Outlet position (tangent to scroll at largest radius)
        r_outlet = p.scroll_inner_radius * p.scroll_expansion
        outlet_length = p.outlet_height * 1.5

        hw = p.outlet_width / 2  # Half width (Z direction)
        hh = p.outlet_height / 2  # Half height (Y direction)
        scroll_half_width = p.impeller_width / 2 * 1.2  # Scroll axial half-width

        # X positions
        x_start = p.center[0] + r_outlet  # At scroll outer edge
        x_end = x_start + outlet_length  # At flange

        # Generate curled scroll-side edge vertices
        # The edge curves inward to match the scroll's cylindrical surface
        curl_bottom = []  # Bottom edge vertices (Y = center - hh)
        curl_top = []     # Top edge vertices (Y = center + hh)

        for i in range(n_curl_segments + 1):
            t = i / n_curl_segments
            # Z position along the scroll width
            z = p.center[2] - scroll_half_width + t * 2 * scroll_half_width

            # Curl follows scroll surface: deeper curl at center, less at edges
            z_normalized = (z - p.center[2]) / scroll_half_width  # -1 to +1
            curl_depth = 0.015 * (1 - z_normalized ** 2)  # Parabolic curl, max 15mm

            x_curled = x_start - curl_depth

            curl_bottom.append([x_curled, p.center[1] - hh, z])
            curl_top.append([x_curled, p.center[1] + hh, z])

        # Add curl vertices to mesh
        curl_bottom_start = len(vertices)
        for v in curl_bottom:
            vertices.append(v)
            normals.append([-1.0, 0.0, 0.0])

        curl_top_start = len(vertices)
        for v in curl_top:
            vertices.append(v)
            normals.append([-1.0, 0.0, 0.0])

        # Flange-side vertices (flat rectangular)
        flange_start = len(vertices)
        flange_corners = [
            [x_end, p.center[1] - hh, p.center[2] - hw],  # 0: Bottom-back
            [x_end, p.center[1] + hh, p.center[2] - hw],  # 1: Top-back
            [x_end, p.center[1] + hh, p.center[2] + hw],  # 2: Top-front
            [x_end, p.center[1] - hh, p.center[2] + hw],  # 3: Bottom-front
        ]
        for corner in flange_corners:
            vertices.append(corner)
            normals.append([1.0, 0.0, 0.0])

        # Connect curled scroll-side to flat flange-side with walls

        # Bottom wall (Y = center - hh)
        for i in range(n_curl_segments):
            v0 = curl_bottom_start + i
            v1 = curl_bottom_start + i + 1

            # Interpolate flange Z positions
            t0 = i / n_curl_segments
            t1 = (i + 1) / n_curl_segments
            fz0 = p.center[2] - hw + t0 * 2 * hw
            fz1 = p.center[2] - hw + t1 * 2 * hw

            f0_idx = len(vertices)
            vertices.append([x_end, p.center[1] - hh, fz0])
            normals.append([0.0, -1.0, 0.0])
            f1_idx = len(vertices)
            vertices.append([x_end, p.center[1] - hh, fz1])
            normals.append([0.0, -1.0, 0.0])

            indices.extend([v0, f0_idx, f1_idx])
            indices.extend([v0, f1_idx, v1])

        # Top wall (Y = center + hh)
        for i in range(n_curl_segments):
            v0 = curl_top_start + i
            v1 = curl_top_start + i + 1

            t0 = i / n_curl_segments
            t1 = (i + 1) / n_curl_segments
            fz0 = p.center[2] - hw + t0 * 2 * hw
            fz1 = p.center[2] - hw + t1 * 2 * hw

            f0_idx = len(vertices)
            vertices.append([x_end, p.center[1] + hh, fz0])
            normals.append([0.0, 1.0, 0.0])
            f1_idx = len(vertices)
            vertices.append([x_end, p.center[1] + hh, fz1])
            normals.append([0.0, 1.0, 0.0])

            indices.extend([v0, f1_idx, f0_idx])
            indices.extend([v0, v1, f1_idx])

        # Back wall (-Z side)
        back_start = len(vertices)
        vertices.extend([
            curl_bottom[0],  # Curl bottom-back
            curl_top[0],     # Curl top-back
            [x_end, p.center[1] + hh, p.center[2] - hw],  # Flange top-back
            [x_end, p.center[1] - hh, p.center[2] - hw],  # Flange bottom-back
        ])
        for _ in range(4):
            normals.append([0.0, 0.0, -1.0])

        indices.extend([back_start, back_start + 1, back_start + 2])
        indices.extend([back_start, back_start + 2, back_start + 3])

        # Front wall (+Z side)
        front_start = len(vertices)
        vertices.extend([
            curl_bottom[-1],  # Curl bottom-front
            curl_top[-1],     # Curl top-front
            [x_end, p.center[1] + hh, p.center[2] + hw],  # Flange top-front
            [x_end, p.center[1] - hh, p.center[2] + hw],  # Flange bottom-front
        ])
        for _ in range(4):
            normals.append([0.0, 0.0, 1.0])

        indices.extend([front_start, front_start + 2, front_start + 1])
        indices.extend([front_start, front_start + 3, front_start + 2])

        # Flange end cap (rectangular face for connector)
        indices.extend([flange_start, flange_start + 1, flange_start + 2])
        indices.extend([flange_start, flange_start + 2, flange_start + 3])

    def get_performance(self, flow_rate: float = None) -> dict:
        """
        Get estimated performance at given flow rate.

        Args:
            flow_rate: Flow rate [m³/h], uses design flow if None

        Returns:
            Dict with velocity, pressure, power estimates
        """
        p = self.params
        Q = (flow_rate if flow_rate else p.flow_rate) / 3600  # m³/s

        return {
            "flow_rate_m3_h": Q * 3600,
            "tip_speed_m_s": p.tip_speed,
            "pressure_rise_Pa": p.pressure_rise,
            "shaft_power_kW": p.shaft_power / 1000,
            "efficiency": p.estimated_efficiency,
            "specific_speed": p.specific_speed,
        }

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the blower geometry."""
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

    @property
    def ports(self) -> dict:
        """
        Get connection ports for the blower.

        Ports:
        - 'inlet': Axial inlet (inlet eye) - air enters here
        - 'outlet': Scroll outlet (rectangular) - air exits here

        Returns:
            Dictionary of port name to ConnectionPort
        """
        from ..connection_ports import ConnectionPort, PortType

        p = self.params

        # Inlet port: at the inlet eye, facing -Z (air comes from -Z direction)
        # Position at center of inlet eye
        inlet_pos = (p.center[0], p.center[1], p.center[2] - p.impeller_width / 2)

        # Outlet port: at scroll outlet, facing +X (standard scroll orientation)
        # Position at center of outlet
        scroll_r = p.scroll_inner_radius * p.scroll_expansion
        outlet_pos = (p.center[0] + scroll_r, p.center[1], p.center[2])

        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=(0.0, 0.0, -1.0),  # Air enters from -Z
                diameter=p.inlet_diameter,
                port_type=PortType.CIRCULAR,
                name="blower_inlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=(1.0, 0.0, 0.0),  # Air exits toward +X
                diameter=np.sqrt(p.outlet_width * p.outlet_height * 4 / PI),  # Equivalent diameter
                width=p.outlet_width,
                height=p.outlet_height,
                port_type=PortType.RECTANGULAR,
                name="blower_outlet",
                compatible_types=[PortType.RECTANGULAR, PortType.CIRCULAR, PortType.FLANGED],
            ),
        }


def create_standard_centrifugal_blower(
    flow_rate: float = 3000,
    pressure_rise: float = 5000
) -> CentrifugalBlower:
    """
    Create a standard centrifugal blower sized for given duty.

    Args:
        flow_rate: Design flow rate [m³/h]
        pressure_rise: Total pressure rise [Pa]

    Returns:
        CentrifugalBlower instance
    """
    # Size impeller based on flow rate
    # Rough sizing: D ~ (Q / (pi * b * u2))^0.5
    # where u2 = tip speed, b = width

    # Estimate tip speed for given pressure (backward-curved, ~80% efficiency)
    # Delta P = rho * u2^2 * psi, where psi ~ 0.4-0.6 for backward curved
    psi = 0.5
    rho = 1.2
    u2 = np.sqrt(pressure_rise / (rho * psi))

    # Target 3000 RPM for standard motor
    rpm = 3000
    impeller_diameter = u2 * 60 / (PI * rpm)

    # Ensure reasonable size
    impeller_diameter = max(impeller_diameter, 0.20)  # Min 200mm
    impeller_diameter = min(impeller_diameter, 1.0)   # Max 1m

    # Width is typically 0.2-0.4 of diameter
    impeller_width = impeller_diameter * 0.25

    params = CentrifugalBlowerParams(
        impeller_diameter=impeller_diameter,
        impeller_width=impeller_width,
        inlet_diameter=impeller_diameter * 0.6,
        hub_diameter=impeller_diameter * 0.25,
        num_blades=10,
        blade_type="backward_curved",
        rpm=rpm,
        flow_rate=flow_rate,
        pressure_rise=pressure_rise,
    )

    return CentrifugalBlower(params)
