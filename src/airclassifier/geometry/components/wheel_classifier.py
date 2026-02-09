"""
Industrial Centrifugal Wheel Classifier for fine particle separation.

Based on Hosokawa Micron Separator MS design:
- Volute/scroll housing for controlled airflow distribution
- Cage-style classifier wheel with shroud rings and radial blades
- Tangential feed inlet with dispersion zone
- Optional secondary air inlet for sealing/classification assist
- Axial fines outlet through wheel center (protein fraction)
- Conical coarse hopper with gravity discharge (starch fraction)

Separation Principle:
At the wheel periphery, particles experience:
- Centrifugal force (outward): F_c = m·ω²·r = (π/6)·d³·ρ_p·ω²·r
- Drag force (inward, from radial airflow): F_d = 3·π·μ·d·v_r (Stokes)

Cut size d50 (where F_c = F_d):
    d50 = sqrt(18·μ·v_r / (Δρ·ω²·r))

    where:
        μ  = air viscosity [Pa·s]
        v_r = radial air velocity through wheel [m/s]
        Δρ = particle density - air density [kg/m³]
        ω  = angular velocity [rad/s]
        r  = wheel radius [m]

Operating range:
- Wheel diameter: 150-300 mm
- RPM: 3000-10000
- G-force at rim: 1000-7000 g
- Target d50: 15-35 μm (vs zigzag's 30-150 μm limit)

Unlike gravity-based classifiers (zigzag: 1g), the wheel achieves
1000-5000g acceleration, enabling cuts at 15-35 μm — essential for
protein/starch separation where the target d50 is 20-25 μm.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class WheelClassifierParams:
    """
    Parameters for industrial centrifugal wheel classifier.

    Modeled after Hosokawa Micron Separator MS series.
    """

    # Wheel geometry (cage-style with shroud rings)
    wheel_diameter: float = 0.20         # [m] Outer diameter of classifier wheel
    wheel_width: float = 0.04            # [m] Axial width (blade height)
    hub_diameter: float = 0.06           # [m] Central hub for fines outlet
    num_blades: int = 24                 # Number of radial blades (16-48 typical)
    blade_thickness: float = 0.002       # [m] Blade thickness
    shroud_thickness: float = 0.003      # [m] Top/bottom shroud disc thickness
    blade_extension: float = 0.005       # [m] Blade tips extend beyond shroud

    # Housing geometry (volute/scroll)
    housing_type: str = "volute"         # "volute" or "cylindrical"
    volute_clearance: float = 0.015      # [m] Gap between wheel and volute
    volute_expansion: float = 1.25       # Expansion ratio (outer/inner radius)
    housing_height: float = 0.10         # [m] Total housing height (above hopper)
    wall_thickness: float = 0.003        # [m] Housing wall thickness

    # Feed inlet (tangential entry)
    feed_inlet_width: float = 0.05       # [m] Rectangular feed width
    feed_inlet_height: float = 0.06      # [m] Rectangular feed height
    feed_inlet_length: float = 0.08      # [m] Inlet duct length
    feed_angular_position: float = PI    # [rad] Position around housing (PI = -X side)

    # Secondary air inlet (optional, for sealing/assist)
    include_secondary_air: bool = False
    secondary_air_diameter: float = 0.025  # [m]
    secondary_air_position: float = PI/2   # [rad] Position (PI/2 = +Z side)

    # Fines outlet (axial, through hub)
    fines_outlet_diameter: float = 0.05  # [m] Axial outlet diameter
    fines_outlet_length: float = 0.06    # [m] Outlet stub length

    # Coarse hopper (conical bottom)
    include_coarse_hopper: bool = True
    coarse_hopper_height: float = 0.08   # [m] Conical hopper height
    coarse_hopper_angle: float = 60.0    # [deg] Half-angle from vertical
    coarse_outlet_diameter: float = 0.04 # [m] Bottom discharge diameter
    coarse_outlet_length: float = 0.05   # [m] Outlet stub length (longer for airlock flange)

    # Motor/drive (mounted on top, shaft through fines outlet)
    include_motor: bool = True           # Include motor housing
    motor_diameter: float = None         # [m] Auto-calculated from power
    motor_length: float = None           # [m] Auto-calculated
    motor_mount_height: float = 0.03     # [m] Height of motor mount flange

    # Operating parameters
    rpm: float = 6000.0                  # [rpm] Operating speed
    target_d50: float = 25e-6            # [m] Target cut size (25 μm)

    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Housing center

    # Mesh resolution
    resolution_radial: int = 32          # Circumferential resolution
    resolution_axial: int = 12           # Axial resolution
    resolution_scroll: int = 48          # Scroll/volute resolution

    def __post_init__(self):
        """Auto-calculate motor dimensions based on wheel power requirements."""
        # Estimate motor power from wheel specs
        # Power ≈ 0.5 * ρ_air * Q * v_tip² * (1 + k_windage)
        # Approximate Q from wheel area and tip speed
        v_tip = PI * self.wheel_diameter * self.rpm / 60
        # Assume flow ~20-30% of wheel swept volume rate
        Q_approx = 0.25 * PI * self.wheel_diameter * self.wheel_width * v_tip
        air_density = 1.2
        k_windage = 0.3
        power_estimate = 0.5 * air_density * Q_approx * v_tip ** 2 * (1 + k_windage)
        power_kw = power_estimate / 1000

        # Auto-calculate motor dimensions based on power
        if self.motor_diameter is None:
            # Motor frame size based on power (IEC standard sizing)
            if power_kw < 2:
                self.motor_diameter = 0.12  # ~IEC 80 frame
            elif power_kw < 5:
                self.motor_diameter = 0.15  # ~IEC 90 frame
            elif power_kw < 10:
                self.motor_diameter = 0.18  # ~IEC 112 frame
            else:
                self.motor_diameter = 0.22  # ~IEC 132 frame

        if self.motor_length is None:
            # Motor length proportional to diameter
            self.motor_length = self.motor_diameter * 1.4

    @property
    def wheel_radius(self) -> float:
        """Wheel outer radius [m]."""
        return self.wheel_diameter / 2

    @property
    def hub_radius(self) -> float:
        """Hub radius [m]."""
        return self.hub_diameter / 2

    @property
    def volute_inner_radius(self) -> float:
        """Inner radius of volute at cutoff [m]."""
        return self.wheel_radius + self.volute_clearance

    @property
    def volute_outer_radius(self) -> float:
        """Outer radius of volute [m]."""
        return self.volute_inner_radius * self.volute_expansion

    @property
    def omega(self) -> float:
        """Angular velocity [rad/s]. Used by the physics kernel and by visualize_geometry.py
        for animation (rotation angle = omega * time)."""
        return TWO_PI * self.rpm / 60.0

    @property
    def tip_speed(self) -> float:
        """Wheel tip speed [m/s]."""
        return self.omega * self.wheel_radius

    @property
    def centrifugal_acceleration(self) -> float:
        """Centrifugal acceleration at wheel rim [m/s²]."""
        return self.omega ** 2 * self.wheel_radius

    @property
    def g_force(self) -> float:
        """Centrifugal force in g's at wheel rim."""
        return self.centrifugal_acceleration / 9.81

    @property
    def blade_gap(self) -> float:
        """Gap between adjacent blades at wheel periphery [m]."""
        circumference = PI * self.wheel_diameter
        blade_arc = self.num_blades * self.blade_thickness
        return (circumference - blade_arc) / self.num_blades

    @property
    def blade_passage_area(self) -> float:
        """Total flow area through blade gaps [m²]."""
        return self.num_blades * self.blade_gap * self.wheel_width

    def calculate_d50(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate cut size d50 for given operating conditions.

        Based on force balance at wheel periphery:
        - Centrifugal: F_c = (π/6)·d³·ρp·ω²·r
        - Drag (Stokes): F_d = 3·π·μ·d·v_r

        Solving F_c = F_d:
        d50 = sqrt(18·μ·v_r / (Δρ·ω²·r))

        Args:
            volumetric_flow: Air flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Cut size d50 [m]
        """
        if self.rpm < 100 or volumetric_flow <= 0:
            return float('inf')

        omega = self.omega
        r = self.wheel_radius
        b = self.wheel_width
        n = self.num_blades
        mu = air_viscosity

        # Radial velocity through blade gaps
        v_r = volumetric_flow / (n * self.blade_gap * b)

        # Force balance
        delta_rho = particle_density - air_density
        if delta_rho <= 0:
            return float('inf')

        d50_sq = 18 * mu * v_r / (delta_rho * omega ** 2 * r)
        return np.sqrt(max(d50_sq, 0))

    def calculate_rpm_for_d50(
        self,
        target_d50: float,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate required RPM to achieve target d50.

        Args:
            target_d50: Target cut size [m]
            volumetric_flow: Air flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Required wheel RPM
        """
        if target_d50 <= 0 or volumetric_flow <= 0:
            return 0.0

        r = self.wheel_radius
        b = self.wheel_width
        n = self.num_blades
        mu = air_viscosity

        v_r = volumetric_flow / (n * self.blade_gap * b)
        delta_rho = particle_density - air_density

        if delta_rho <= 0:
            return 0.0

        omega_sq = 18 * mu * v_r / (target_d50 ** 2 * delta_rho * r)
        omega = np.sqrt(max(omega_sq, 0))

        return omega * 60 / TWO_PI

    def calculate_power(
        self,
        volumetric_flow: float,
        air_density: float = 1.204,
    ) -> float:
        """
        Estimate shaft power for wheel rotation [W].

        Power from accelerating air + windage losses.
        """
        v_tip = self.tip_speed
        k_windage = 0.3  # Windage factor

        P_flow = 0.5 * air_density * volumetric_flow * v_tip ** 2
        return P_flow * (1 + k_windage)


class WheelClassifier:
    """
    Industrial centrifugal wheel classifier geometry.

    Creates mesh for complete classifier assembly including:
    - Volute/scroll housing with tangential inlet
    - Cage-style classifier wheel (two shroud discs + radial blades)
    - Axial fines outlet through wheel center
    - Conical coarse hopper with bottom discharge
    - Optional secondary air inlet

    Flow path:
    1. Feed air + particles enter tangentially into volute
    2. Air spirals inward through wheel blade gaps
    3. Fine particles (F_drag > F_c) pass through → fines outlet (+Y)
    4. Coarse particles (F_c > F_drag) rejected → hopper (-Y)
    """

    def __init__(self, params: WheelClassifierParams):
        """
        Initialize wheel classifier.

        Args:
            params: WheelClassifierParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None
        self._ports = None

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the wheel classifier.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        if self._vertices is not None:
            return self._vertices, self._indices, self._normals

        vertices = []
        indices = []
        normals = []

        p = self.params

        # Generate volute/scroll housing
        if p.housing_type == "volute":
            self._add_volute_housing(vertices, indices, normals)
        else:
            self._add_cylindrical_housing(vertices, indices, normals)

        # Generate cage-style classifier wheel
        self._add_classifier_wheel(vertices, indices, normals)

        # Generate feed inlet
        self._add_feed_inlet(vertices, indices, normals)

        # Generate secondary air inlet (optional)
        if p.include_secondary_air:
            self._add_secondary_air_inlet(vertices, indices, normals)

        # Generate fines outlet (axial, through hub)
        self._add_fines_outlet(vertices, indices, normals)

        # Generate coarse hopper (conical bottom)
        if p.include_coarse_hopper:
            self._add_coarse_hopper(vertices, indices, normals)

        # Generate motor/drive assembly (on top)
        if p.include_motor:
            self._add_motor_drive(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32).reshape(-1, 3)

        return self._vertices, self._indices, self._normals

    def get_static_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate static mesh -- everything EXCEPT the rotating classifier wheel.

        Includes: volute housing, feed inlet, secondary air inlet, fines outlet,
        coarse hopper, motor/drive assembly.

        The classifier wheel (shroud discs, radial blades, hub) is excluded;
        use get_wheel_mesh() for the animated rotating part.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        vertices = []
        indices = []
        normals = []
        p = self.params

        # Housing
        if p.housing_type == "volute":
            self._add_volute_housing(vertices, indices, normals)
        else:
            self._add_cylindrical_housing(vertices, indices, normals)

        # Feed inlet
        self._add_feed_inlet(vertices, indices, normals)

        # Secondary air inlet (optional)
        if p.include_secondary_air:
            self._add_secondary_air_inlet(vertices, indices, normals)

        # Fines outlet (axial, through hub)
        self._add_fines_outlet(vertices, indices, normals)

        # Coarse hopper (conical bottom)
        if p.include_coarse_hopper:
            self._add_coarse_hopper(vertices, indices, normals)

        # Motor/drive assembly
        if p.include_motor:
            self._add_motor_drive(vertices, indices, normals)

        v = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        i = np.array(indices, dtype=np.int32)
        n = np.array(normals, dtype=np.float32).reshape(-1, 3)
        return v, i, n

    def get_wheel_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for ONLY the rotating classifier wheel.

        Includes: top/bottom shroud discs, radial blades, central hub.

        This is the part that spins at operating RPM. The housing and
        everything else is returned by get_static_mesh().

        Returns:
            Tuple of (vertices, indices, normals)
        """
        vertices = []
        indices = []
        normals = []

        self._add_classifier_wheel(vertices, indices, normals)

        v = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        i = np.array(indices, dtype=np.int32)
        n = np.array(normals, dtype=np.float32).reshape(-1, 3)
        return v, i, n

    def _add_volute_housing(self, vertices: List, indices: List, normals: List):
        """
        Generate volute/scroll housing mesh.

        The volute creates a controlled airflow distribution around the wheel,
        similar to a centrifugal blower but in reverse (air flows inward).
        """
        p = self.params
        n_scroll = p.resolution_scroll
        n_axial = p.resolution_axial
        cx, cy, cz = p.center

        # Volute geometry
        r_inner = p.volute_inner_radius
        r_outer = p.volute_outer_radius
        half_height = p.housing_height / 2

        start_idx = len(vertices) // 3

        # Generate volute surface (logarithmic spiral expansion)
        for i in range(n_scroll + 1):
            theta = (i / n_scroll) * TWO_PI
            # Radius increases with angle (spiral)
            t = i / n_scroll
            r = r_inner + (r_outer - r_inner) * t

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            for j in range(n_axial + 1):
                # Axial position (Y direction)
                t_axial = j / n_axial
                y = cy - half_height + t_axial * 2 * half_height

                x = cx + r * cos_t
                z = cz + r * sin_t

                vertices.extend([x, y, z])
                # Normal pointing outward
                normals.extend([cos_t, 0.0, sin_t])

        # Generate triangles for outer wall
        n_circ = n_axial + 1
        for i in range(n_scroll):
            for j in range(n_axial):
                v0 = start_idx + i * n_circ + j
                v1 = start_idx + i * n_circ + j + 1
                v2 = start_idx + (i + 1) * n_circ + j + 1
                v3 = start_idx + (i + 1) * n_circ + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Add top and bottom annular plates (housing ends)
        self._add_housing_end_plates(vertices, indices, normals)

    def _add_housing_end_plates(self, vertices: List, indices: List, normals: List):
        """Add top and bottom annular end plates for volute housing."""
        p = self.params
        n_radial = p.resolution_radial
        cx, cy, cz = p.center

        r_inner = p.hub_radius * 1.2  # Opening for fines outlet
        r_outer = p.volute_outer_radius
        half_height = p.housing_height / 2

        for side, y_pos in [(-1, cy - half_height), (1, cy + half_height)]:
            start_idx = len(vertices) // 3

            # Inner ring
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                x = cx + r_inner * np.cos(theta)
                z = cz + r_inner * np.sin(theta)
                vertices.extend([x, y_pos, z])
                normals.extend([0.0, float(side), 0.0])

            # Outer ring
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                x = cx + r_outer * np.cos(theta)
                z = cz + r_outer * np.sin(theta)
                vertices.extend([x, y_pos, z])
                normals.extend([0.0, float(side), 0.0])

            # Generate quads
            for i in range(n_radial):
                i_next = (i + 1) % n_radial
                v0 = start_idx + i
                v1 = start_idx + i_next
                v2 = start_idx + n_radial + i_next
                v3 = start_idx + n_radial + i

                if side > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v0, v3, v2])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v0, v2, v3])

    def _add_cylindrical_housing(self, vertices: List, indices: List, normals: List):
        """Generate simple cylindrical housing (alternative to volute)."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial
        cx, cy, cz = p.center

        r = p.volute_outer_radius
        half_height = p.housing_height / 2

        start_idx = len(vertices) // 3

        # Outer cylinder wall
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            for j in range(n_axial + 1):
                t = j / n_axial
                y = cy - half_height + t * 2 * half_height

                vertices.extend([cx + r * cos_t, y, cz + r * sin_t])
                normals.extend([cos_t, 0.0, sin_t])

        # Generate quads
        n_circ = n_axial + 1
        for i in range(n_radial):
            for j in range(n_axial):
                v0 = start_idx + i * n_circ + j
                v1 = start_idx + i * n_circ + j + 1
                v2 = start_idx + (i + 1) * n_circ + j + 1
                v3 = start_idx + (i + 1) * n_circ + j

                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])

        # Add end plates
        self._add_housing_end_plates(vertices, indices, normals)

    def _add_classifier_wheel(self, vertices: List, indices: List, normals: List):
        """
        Generate cage-style classifier wheel with shroud rings and radial blades.

        The wheel consists of:
        - Top shroud disc (ring)
        - Bottom shroud disc (ring)
        - Radial blades connecting the shrouds
        - Central hub (for shaft/fines outlet)
        """
        p = self.params
        n_radial = p.resolution_radial
        cx, cy, cz = p.center

        r_wheel = p.wheel_radius
        r_hub = p.hub_radius
        half_width = p.wheel_width / 2
        shroud_t = p.shroud_thickness

        # ================================================================
        # TOP SHROUD (disc with hole for fines)
        # ================================================================
        self._add_shroud_disc(
            vertices, indices, normals,
            cy + half_width, cy + half_width + shroud_t,
            r_hub, r_wheel, n_radial, normal_dir=1
        )

        # ================================================================
        # BOTTOM SHROUD (solid disc with hub)
        # ================================================================
        self._add_shroud_disc(
            vertices, indices, normals,
            cy - half_width - shroud_t, cy - half_width,
            r_hub, r_wheel, n_radial, normal_dir=-1
        )

        # ================================================================
        # RADIAL BLADES
        # ================================================================
        blade_t = p.blade_thickness
        n_blades = p.num_blades

        for b in range(n_blades):
            blade_angle = TWO_PI * b / n_blades
            self._add_radial_blade(
                vertices, indices, normals,
                blade_angle, r_hub, r_wheel + p.blade_extension,
                cy - half_width, cy + half_width,
                blade_t
            )

        # ================================================================
        # HUB CYLINDER
        # ================================================================
        self._add_hub_cylinder(vertices, indices, normals)

    def _add_shroud_disc(
        self,
        vertices: List, indices: List, normals: List,
        y_bottom: float, y_top: float,
        r_inner: float, r_outer: float,
        n_radial: int, normal_dir: int
    ):
        """Add a shroud disc (annular ring with thickness)."""
        p = self.params
        cx, cy, cz = p.center

        # Top face
        start_idx = len(vertices) // 3
        y = y_top if normal_dir > 0 else y_bottom

        for ring, r in enumerate([r_inner, r_outer]):
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                vertices.extend([cx + r * np.cos(theta), y, cz + r * np.sin(theta)])
                normals.extend([0.0, float(normal_dir), 0.0])

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = start_idx + i
            v1 = start_idx + i_next
            v2 = start_idx + n_radial + i_next
            v3 = start_idx + n_radial + i

            if normal_dir > 0:
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])
            else:
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Outer rim (cylindrical edge)
        start_idx = len(vertices) // 3
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            vertices.extend([cx + r_outer * cos_t, y_bottom, cz + r_outer * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

            vertices.extend([cx + r_outer * cos_t, y_top, cz + r_outer * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _add_radial_blade(
        self,
        vertices: List, indices: List, normals: List,
        angle: float, r_inner: float, r_outer: float,
        y_bottom: float, y_top: float, thickness: float
    ):
        """Add a single radial blade."""
        p = self.params
        cx, cy, cz = p.center

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Blade corners (4 corners, 2 faces)
        half_t = thickness / 2

        # Tangential direction (perpendicular to radial)
        tx, tz = -sin_a, cos_a

        # Inner edge coordinates
        x_in = cx + r_inner * cos_a
        z_in = cz + r_inner * sin_a

        # Outer edge coordinates
        x_out = cx + r_outer * cos_a
        z_out = cz + r_outer * sin_a

        # Offset for thickness
        ox, oz = half_t * tx, half_t * tz

        start_idx = len(vertices) // 3

        # Front face (+tangential normal)
        vertices.extend([x_in + ox, y_bottom, z_in + oz])
        vertices.extend([x_in + ox, y_top, z_in + oz])
        vertices.extend([x_out + ox, y_top, z_out + oz])
        vertices.extend([x_out + ox, y_bottom, z_out + oz])
        for _ in range(4):
            normals.extend([tx, 0.0, tz])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Back face (-tangential normal)
        start_idx = len(vertices) // 3
        vertices.extend([x_in - ox, y_bottom, z_in - oz])
        vertices.extend([x_out - ox, y_bottom, z_out - oz])
        vertices.extend([x_out - ox, y_top, z_out - oz])
        vertices.extend([x_in - ox, y_top, z_in - oz])
        for _ in range(4):
            normals.extend([-tx, 0.0, -tz])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Top edge
        start_idx = len(vertices) // 3
        vertices.extend([x_in - ox, y_top, z_in - oz])
        vertices.extend([x_out - ox, y_top, z_out - oz])
        vertices.extend([x_out + ox, y_top, z_out + oz])
        vertices.extend([x_in + ox, y_top, z_in + oz])
        for _ in range(4):
            normals.extend([0.0, 1.0, 0.0])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Bottom edge
        start_idx = len(vertices) // 3
        vertices.extend([x_in - ox, y_bottom, z_in - oz])
        vertices.extend([x_in + ox, y_bottom, z_in + oz])
        vertices.extend([x_out + ox, y_bottom, z_out + oz])
        vertices.extend([x_out - ox, y_bottom, z_out - oz])
        for _ in range(4):
            normals.extend([0.0, -1.0, 0.0])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Outer tip edge
        start_idx = len(vertices) // 3
        vertices.extend([x_out - ox, y_bottom, z_out - oz])
        vertices.extend([x_out + ox, y_bottom, z_out + oz])
        vertices.extend([x_out + ox, y_top, z_out + oz])
        vertices.extend([x_out - ox, y_top, z_out - oz])
        for _ in range(4):
            normals.extend([cos_a, 0.0, sin_a])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

    def _add_hub_cylinder(self, vertices: List, indices: List, normals: List):
        """Add central hub cylinder."""
        p = self.params
        n_radial = p.resolution_radial
        cx, cy, cz = p.center

        r_hub = p.hub_radius
        half_width = p.wheel_width / 2

        start_idx = len(vertices) // 3

        # Hub outer surface
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            vertices.extend([cx + r_hub * cos_t, cy - half_width, cz + r_hub * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

            vertices.extend([cx + r_hub * cos_t, cy + half_width, cz + r_hub * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            # Inward-facing normals (inside hub)
            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

    def _add_feed_inlet(self, vertices: List, indices: List, normals: List):
        """
        Add tangential feed inlet with proper blend into volute housing.

        Like a centrifugal blower outlet, the inlet duct transitions smoothly
        from the curved volute surface to a rectangular duct. The inlet
        "cuts into" the volute rather than being placed on the outside.
        """
        p = self.params
        cx, cy, cz = p.center
        n_blend = 8  # Resolution for curved transition

        w = p.feed_inlet_width
        h = p.feed_inlet_height
        length = p.feed_inlet_length
        theta = p.feed_angular_position  # Position around housing

        r_volute = p.volute_outer_radius
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Tangential direction (perpendicular to radial)
        tan_x = -sin_t
        tan_z = cos_t

        # Half dimensions
        hw = w / 2  # Half width (along tangential)
        hh = h / 2  # Half height (along Y)

        # ================================================================
        # CURVED TRANSITION ZONE (blends volute curve to rectangular duct)
        # ================================================================
        # The transition wraps around a portion of the volute at theta
        # and blends from circular arc to rectangular cross-section.
        # Short blend so feed duct meets volute body flush (fit and weld).
        blend_length = min(r_volute * 0.08, 0.015)  # Short transition at volute
        angle_span = PI / 5  # Angular extent of blend on volute (36 degrees)

        transition_start = len(vertices) // 3

        # Generate blend profiles from volute surface to duct entrance
        for i in range(n_blend + 1):
            t = i / n_blend  # 0 = at volute, 1 = at duct start

            # Radial distance from center (moves outward along radial)
            r_current = r_volute + t * blend_length

            for j in range(n_blend + 1):
                s = j / n_blend  # 0 = bottom of profile, 1 = top

                # At volute (t=0): profile follows volute curve (arc)
                # At duct (t=1): profile is flat/rectangular

                # Y position
                y = cy - hh + s * 2 * hh

                # Blend from arc to flat
                if t < 0.3:
                    # Arc section: angle varies within angle_span
                    arc_blend = t / 0.3
                    # Arc angle at this Y position
                    y_rel = (s - 0.5) * 2  # -1 to +1
                    local_angle = theta + y_rel * angle_span / 2 * (1 - arc_blend)
                    local_cos = np.cos(local_angle)
                    local_sin = np.sin(local_angle)
                    x = cx + r_current * local_cos
                    z = cz + r_current * local_sin
                    # Normal blends from radial to tangential
                    nx = local_cos * (1 - arc_blend) + cos_t * arc_blend
                    nz = local_sin * (1 - arc_blend) + sin_t * arc_blend
                else:
                    # Rectangular section
                    # Tangential offset based on s (width position)
                    tan_offset = (s - 0.5) * 2 * hw
                    x = cx + r_current * cos_t + tan_offset * tan_x
                    z = cz + r_current * sin_t + tan_offset * tan_z
                    nx = cos_t
                    nz = sin_t

                vertices.extend([x, y, z])
                # Normalize
                n_len = np.sqrt(nx * nx + nz * nz)
                if n_len > 0.001:
                    normals.extend([nx / n_len, 0.0, nz / n_len])
                else:
                    normals.extend([cos_t, 0.0, sin_t])

        # Generate triangles for transition surface
        pts_per_row = n_blend + 1
        for i in range(n_blend):
            for j in range(n_blend):
                v0 = transition_start + i * pts_per_row + j
                v1 = transition_start + i * pts_per_row + j + 1
                v2 = transition_start + (i + 1) * pts_per_row + j + 1
                v3 = transition_start + (i + 1) * pts_per_row + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # ================================================================
        # RECTANGULAR DUCT (extends from blend to inlet flange; fit and weld)
        # ================================================================
        duct_start_r = r_volute + blend_length  # Starts at end of short blend
        duct_end_r = r_volute + length

        # Start position (connected to blend zone)
        x_start = cx + duct_start_r * cos_t
        z_start = cz + duct_start_r * sin_t

        # End position (inlet flange)
        x_end = cx + duct_end_r * cos_t
        z_end = cz + duct_end_r * sin_t

        y0 = cy - hh
        y1 = cy + hh

        # Corners at duct start
        xs0 = x_start - hw * tan_x
        zs0 = z_start - hw * tan_z
        xs1 = x_start + hw * tan_x
        zs1 = z_start + hw * tan_z

        # Corners at duct end (flange)
        xe0 = x_end - hw * tan_x
        ze0 = z_end - hw * tan_z
        xe1 = x_end + hw * tan_x
        ze1 = z_end + hw * tan_z

        # Bottom face
        start_idx = len(vertices) // 3
        vertices.extend([xs0, y0, zs0])
        vertices.extend([xs1, y0, zs1])
        vertices.extend([xe1, y0, ze1])
        vertices.extend([xe0, y0, ze0])
        for _ in range(4):
            normals.extend([0.0, -1.0, 0.0])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Top face
        start_idx = len(vertices) // 3
        vertices.extend([xs0, y1, zs0])
        vertices.extend([xe0, y1, ze0])
        vertices.extend([xe1, y1, ze1])
        vertices.extend([xs1, y1, zs1])
        for _ in range(4):
            normals.extend([0.0, 1.0, 0.0])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Left side wall
        start_idx = len(vertices) // 3
        vertices.extend([xs0, y0, zs0])
        vertices.extend([xe0, y0, ze0])
        vertices.extend([xe0, y1, ze0])
        vertices.extend([xs0, y1, zs0])
        for _ in range(4):
            normals.extend([-tan_x, 0.0, -tan_z])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Right side wall
        start_idx = len(vertices) // 3
        vertices.extend([xs1, y0, zs1])
        vertices.extend([xs1, y1, zs1])
        vertices.extend([xe1, y1, ze1])
        vertices.extend([xe1, y0, ze1])
        for _ in range(4):
            normals.extend([tan_x, 0.0, tan_z])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # Inlet flange face (outer opening)
        start_idx = len(vertices) // 3
        vertices.extend([xe0, y0, ze0])
        vertices.extend([xe1, y0, ze1])
        vertices.extend([xe1, y1, ze1])
        vertices.extend([xe0, y1, ze0])
        for _ in range(4):
            normals.extend([cos_t, 0.0, sin_t])

        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx, start_idx + 2, start_idx + 3])

        # ================================================================
        # SIDE WALLS for transition blend (close the sides)
        # ================================================================
        # Left side wall of transition
        start_idx = len(vertices) // 3
        for i in range(n_blend + 1):
            t = i / n_blend
            r_current = r_volute + t * blend_length

            if t < 0.3:
                arc_blend = t / 0.3
                local_angle = theta - angle_span / 2 * (1 - arc_blend)
                x = cx + r_current * np.cos(local_angle)
                z = cz + r_current * np.sin(local_angle)
            else:
                x = cx + r_current * cos_t - hw * tan_x
                z = cz + r_current * sin_t - hw * tan_z

            vertices.extend([x, cy - hh, z])
            normals.extend([-tan_x, 0.0, -tan_z])
            vertices.extend([x, cy + hh, z])
            normals.extend([-tan_x, 0.0, -tan_z])

        for i in range(n_blend):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v3, v2])
            indices.extend([v0, v2, v1])

        # Right side wall of transition
        start_idx = len(vertices) // 3
        for i in range(n_blend + 1):
            t = i / n_blend
            r_current = r_volute + t * blend_length

            if t < 0.3:
                arc_blend = t / 0.3
                local_angle = theta + angle_span / 2 * (1 - arc_blend)
                x = cx + r_current * np.cos(local_angle)
                z = cz + r_current * np.sin(local_angle)
            else:
                x = cx + r_current * cos_t + hw * tan_x
                z = cz + r_current * sin_t + hw * tan_z

            vertices.extend([x, cy - hh, z])
            normals.extend([tan_x, 0.0, tan_z])
            vertices.extend([x, cy + hh, z])
            normals.extend([tan_x, 0.0, tan_z])

        for i in range(n_blend):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _add_secondary_air_inlet(self, vertices: List, indices: List, normals: List):
        """Add secondary air inlet port."""
        p = self.params
        cx, cy, cz = p.center
        n_radial = 16

        r = p.volute_outer_radius
        d = p.secondary_air_diameter
        theta = p.secondary_air_position

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Position on housing
        port_x = cx + r * cos_t
        port_z = cz + r * sin_t
        port_length = d * 1.5

        start_idx = len(vertices) // 3

        # Cylindrical inlet stub
        for i in range(n_radial + 1):
            phi = (i / n_radial) * TWO_PI

            for j in range(2):
                t = j  # 0 = at housing, 1 = at opening
                dist = t * port_length

                x = port_x + dist * cos_t + (d/2) * np.cos(phi) * sin_t
                y = cy + (d/2) * np.sin(phi)
                z = port_z + dist * sin_t - (d/2) * np.cos(phi) * cos_t

                vertices.extend([x, y, z])
                # Normal perpendicular to cylinder
                nx = np.cos(phi) * sin_t
                nz = -np.cos(phi) * cos_t
                ny = np.sin(phi)
                normals.extend([nx, ny, nz])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _add_fines_outlet(self, vertices: List, indices: List, normals: List):
        """Add axial fines outlet through wheel hub."""
        p = self.params
        cx, cy, cz = p.center
        n_radial = p.resolution_radial

        r = p.fines_outlet_diameter / 2
        y_base = cy + p.housing_height / 2
        length = p.fines_outlet_length

        start_idx = len(vertices) // 3

        # Cylindrical outlet duct (extending upward)
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # At housing top
            vertices.extend([cx + r * cos_t, y_base, cz + r * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

            # Extended upward
            vertices.extend([cx + r * cos_t, y_base + length, cz + r * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _add_coarse_hopper(self, vertices: List, indices: List, normals: List):
        """Add conical coarse hopper at bottom."""
        p = self.params
        cx, cy, cz = p.center
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial // 2

        r_top = p.volute_inner_radius  # Top of hopper at housing bottom
        r_bottom = p.coarse_outlet_diameter / 2
        y_top = cy - p.housing_height / 2
        y_bottom = y_top - p.coarse_hopper_height

        start_idx = len(vertices) // 3

        # Conical surface
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            for j in range(n_axial + 1):
                t = j / n_axial
                r = r_top + (r_bottom - r_top) * t
                y = y_top + (y_bottom - y_top) * t

                vertices.extend([cx + r * cos_t, y, cz + r * sin_t])

                # Normal perpendicular to cone surface
                cone_angle = np.arctan2(r_top - r_bottom, p.coarse_hopper_height)
                nr = np.cos(cone_angle)
                ny = np.sin(cone_angle)
                normals.extend([nr * cos_t, ny, nr * sin_t])

        n_circ = n_axial + 1
        for i in range(n_radial):
            for j in range(n_axial):
                v0 = start_idx + i * n_circ + j
                v1 = start_idx + i * n_circ + j + 1
                v2 = start_idx + (i + 1) * n_circ + j + 1
                v3 = start_idx + (i + 1) * n_circ + j

                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])

        # Outlet stub at bottom
        outlet_length = p.coarse_outlet_length
        start_idx = len(vertices) // 3

        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            vertices.extend([cx + r_bottom * cos_t, y_bottom, cz + r_bottom * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

            vertices.extend([cx + r_bottom * cos_t, y_bottom - outlet_length, cz + r_bottom * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Add flange at bottom for airlock connection
        self._add_coarse_outlet_flange(vertices, indices, normals)

    def _add_coarse_outlet_flange(self, vertices: List, indices: List, normals: List):
        """Add flanged connection at coarse outlet for airlock."""
        p = self.params
        cx, cy, cz = p.center
        n_radial = p.resolution_radial

        r_outlet = p.coarse_outlet_diameter / 2
        r_flange = r_outlet * 1.6  # Flange OD is larger
        flange_thickness = 0.008  # 8mm thick flange

        y_outlet = cy - p.housing_height / 2 - p.coarse_hopper_height - p.coarse_outlet_length
        y_flange_top = y_outlet
        y_flange_bottom = y_outlet - flange_thickness

        start_idx = len(vertices) // 3

        # Flange top face (annular ring)
        for ring, r in enumerate([r_outlet, r_flange]):
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                vertices.extend([cx + r * np.cos(theta), y_flange_top, cz + r * np.sin(theta)])
                normals.extend([0.0, 1.0, 0.0])

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = start_idx + i
            v1 = start_idx + i_next
            v2 = start_idx + n_radial + i_next
            v3 = start_idx + n_radial + i

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Flange outer rim (cylinder)
        start_idx = len(vertices) // 3
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            vertices.extend([cx + r_flange * cos_t, y_flange_top, cz + r_flange * sin_t])
            normals.extend([cos_t, 0.0, sin_t])
            vertices.extend([cx + r_flange * cos_t, y_flange_bottom, cz + r_flange * sin_t])
            normals.extend([cos_t, 0.0, sin_t])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

        # Flange bottom face (annular ring)
        start_idx = len(vertices) // 3
        for ring, r in enumerate([r_outlet, r_flange]):
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                vertices.extend([cx + r * np.cos(theta), y_flange_bottom, cz + r * np.sin(theta)])
                normals.extend([0.0, -1.0, 0.0])

        for i in range(n_radial):
            i_next = (i + 1) % n_radial
            v0 = start_idx + i
            v1 = start_idx + i_next
            v2 = start_idx + n_radial + i_next
            v3 = start_idx + n_radial + i

            indices.extend([v0, v2, v1])
            indices.extend([v0, v3, v2])

    def _add_motor_drive(self, vertices: List, indices: List, normals: List):
        """
        Add side-mounted motor with belt drive to classifier wheel.

        The motor is mounted to the SIDE of the housing (not on top) to keep
        the fines outlet clear for ductwork connection to cyclones. A belt
        drive transmits power from the motor to the wheel shaft.

        Layout (top view, looking down -Y):

                        +Z
                         │
            ┌────────────┼────────────┐
            │            │            │
            │   WHEEL    │  HOUSING   │
            │  (center)  │            │
            │            │            │
            └────────────┼────────────┘
        -X ──────────────┼──────────────── +X
                         │     ╔═════╗
                         │     ║MOTOR║ ◄── Side-mounted
                         │     ║     ║     (belt drive)
                         │     ╚══╤══╝
                        -Z        │
                               V-BELT to wheel shaft

        Side view (+X direction):

            FINES OUTLET (clear for ductwork)
                  ↑
            ┌─────┴─────┐
            │  HOUSING  │────────────┐
            │  (wheel)  │   BELT     │
            └─────┬─────┘   GUARD    │
                  │      ┌──────────┐│
              HOPPER     │  MOTOR   ││
                  │      │ ════════ ││ ◄── Cooling fins
                  ↓      └────┬─────┘│
            COARSE OUT     FEET   ───┘
        """
        p = self.params
        cx, cy, cz = p.center
        n_radial = 16

        motor_r = p.motor_diameter / 2
        motor_len = p.motor_length

        # Motor position: to the side (+X, -Z quadrant) of housing
        # Offset so motor doesn't overlap with housing or hopper
        housing_r = p.volute_outer_radius
        motor_offset_x = housing_r + motor_r + 0.05  # Gap between housing and motor
        motor_offset_z = -motor_r - 0.02  # Slightly toward -Z

        motor_cx = cx + motor_offset_x
        motor_cz = cz + motor_offset_z

        # Motor vertical position: centered on wheel height
        motor_cy = cy  # Same height as wheel center

        # ================================================================
        # MOTOR BODY (cylindrical, horizontal orientation along Z)
        # ================================================================
        # Motor axis is along Z (shaft points toward housing)
        motor_z_start = motor_cz - motor_len / 2  # Away from housing
        motor_z_end = motor_cz + motor_len / 2    # Toward housing

        start_idx = len(vertices) // 3

        # Motor cylinder (axis along Z)
        for i in range(n_radial + 1):
            theta = (i / n_radial) * TWO_PI
            # Rotate around Z axis: X and Y vary, Z is axial
            dx = motor_r * np.cos(theta)
            dy = motor_r * np.sin(theta)

            # Start end of motor
            vertices.extend([motor_cx + dx, motor_cy + dy, motor_z_start])
            normals.extend([np.cos(theta), np.sin(theta), 0.0])

            # End of motor (toward housing)
            vertices.extend([motor_cx + dx, motor_cy + dy, motor_z_end])
            normals.extend([np.cos(theta), np.sin(theta), 0.0])

        for i in range(n_radial):
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + (i + 1) * 2 + 1
            v3 = start_idx + (i + 1) * 2

            indices.extend([v0, v1, v2])
            indices.extend([v0, v2, v3])

        # Motor end caps
        for z_pos, z_dir in [(motor_z_start, -1.0), (motor_z_end, 1.0)]:
            start_idx = len(vertices) // 3
            for i in range(n_radial):
                theta = (i / n_radial) * TWO_PI
                vertices.extend([motor_cx + motor_r * np.cos(theta),
                                motor_cy + motor_r * np.sin(theta), z_pos])
                normals.extend([0.0, 0.0, z_dir])

            center_idx = len(vertices) // 3
            vertices.extend([motor_cx, motor_cy, z_pos])
            normals.extend([0.0, 0.0, z_dir])

            for i in range(n_radial):
                i_next = (i + 1) % n_radial
                if z_dir > 0:
                    indices.extend([start_idx + i, start_idx + i_next, center_idx])
                else:
                    indices.extend([start_idx + i, center_idx, start_idx + i_next])

        # ================================================================
        # MOTOR MOUNT / FEET
        # ================================================================
        foot_width = motor_r * 1.5
        foot_depth = motor_len * 0.8
        foot_height = 0.02
        foot_y = motor_cy - motor_r - foot_height / 2

        self._add_box(
            vertices, indices, normals,
            motor_cx, foot_y, motor_cz,
            foot_width, foot_height, foot_depth
        )

        # ================================================================
        # BELT GUARD (covers belt drive area)
        # ================================================================
        # Simple box representing belt guard between motor and housing
        guard_width = motor_offset_x - housing_r + 0.02
        guard_height = motor_r * 1.5
        guard_depth = motor_r * 1.2

        guard_cx = cx + housing_r + guard_width / 2 - 0.01
        guard_cy = cy
        guard_cz = cz

        self._add_box(
            vertices, indices, normals,
            guard_cx, guard_cy, guard_cz,
            guard_width, guard_height, guard_depth
        )

        # ================================================================
        # TERMINAL BOX (on back of motor)
        # ================================================================
        box_width = motor_r * 0.5
        box_height = motor_r * 0.4
        box_depth = motor_r * 0.25

        # Terminal box on top of motor
        box_y = motor_cy + motor_r + box_height / 2

        self._add_box(
            vertices, indices, normals,
            motor_cx, box_y, motor_cz,
            box_width, box_height, box_depth
        )

    def _add_box(self, vertices: List, indices: List, normals: List,
                 cx: float, cy: float, cz: float,
                 dx: float, dy: float, dz: float):
        """Add a simple box shape (for terminal box, etc.)."""
        hx, hy, hz = dx / 2, dy / 2, dz / 2

        start_idx = len(vertices) // 3

        # 8 corners
        corners = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz + hz),
            (cx - hx, cy - hy, cz + hz),
            (cx - hx, cy + hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]

        # 6 faces with normals
        faces = [
            ([0, 1, 2, 3], [0.0, -1.0, 0.0]),  # Bottom
            ([4, 7, 6, 5], [0.0, 1.0, 0.0]),   # Top
            ([0, 4, 5, 1], [0.0, 0.0, -1.0]),  # Back
            ([2, 6, 7, 3], [0.0, 0.0, 1.0]),   # Front
            ([0, 3, 7, 4], [-1.0, 0.0, 0.0]),  # Left
            ([1, 5, 6, 2], [1.0, 0.0, 0.0]),   # Right
        ]

        for face_indices, normal in faces:
            face_start = len(vertices) // 3
            for fi in face_indices:
                vertices.extend(corners[fi])
                normals.extend(normal)

            indices.extend([face_start, face_start + 1, face_start + 2])
            indices.extend([face_start, face_start + 2, face_start + 3])

    @property
    def ports(self) -> Dict[str, ConnectionPort]:
        """Connection ports for assembly integration."""
        if self._ports is not None:
            return self._ports

        p = self.params
        cx, cy, cz = p.center

        theta = p.feed_angular_position
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        r = p.volute_outer_radius + p.feed_inlet_length

        # Feed inlet (tangential, from zigzag fines)
        inlet_x = cx + r * cos_t
        inlet_z = cz + r * sin_t
        inlet = ConnectionPort(
            position=(inlet_x, cy, inlet_z),
            direction=(cos_t, 0.0, sin_t),  # Radially outward
            width=p.feed_inlet_width,
            height=p.feed_inlet_height,
            port_type=PortType.RECTANGULAR,
            name="inlet"
        )

        # Fines outlet (axial, to cyclones)
        fines_y = cy + p.housing_height / 2 + p.fines_outlet_length
        fines = ConnectionPort(
            position=(cx, fines_y, cz),
            direction=(0.0, 1.0, 0.0),  # +Y
            diameter=p.fines_outlet_diameter,
            port_type=PortType.CIRCULAR,
            name="fines_outlet"
        )

        # Coarse outlet (bottom, to airlock/collection)
        coarse_y = cy - p.housing_height / 2 - p.coarse_hopper_height - p.coarse_outlet_length
        coarse = ConnectionPort(
            position=(cx, coarse_y, cz),
            direction=(0.0, -1.0, 0.0),  # -Y
            diameter=p.coarse_outlet_diameter,
            port_type=PortType.CIRCULAR,
            name="coarse_outlet"
        )

        self._ports = {
            'inlet': inlet,
            'fines_outlet': fines,
            'coarse_outlet': coarse,
        }

        # Optional secondary air
        if p.include_secondary_air:
            theta_s = p.secondary_air_position
            cos_s, sin_s = np.cos(theta_s), np.sin(theta_s)
            r_s = p.volute_outer_radius + p.secondary_air_diameter * 1.5

            self._ports['secondary_air'] = ConnectionPort(
                position=(cx + r_s * cos_s, cy, cz + r_s * sin_s),
                direction=(cos_s, 0.0, sin_s),
                diameter=p.secondary_air_diameter,
                port_type=PortType.CIRCULAR,
                name="secondary_air"
            )

        return self._ports

    def get_operating_summary(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> Dict:
        """Get operating characteristics at given flow rate."""
        p = self.params

        d50 = p.calculate_d50(
            volumetric_flow, particle_density, air_density, air_viscosity
        )
        power = p.calculate_power(volumetric_flow, air_density)
        rpm_for_target = p.calculate_rpm_for_d50(
            p.target_d50, volumetric_flow, particle_density, air_density, air_viscosity
        )

        return {
            'wheel_diameter_mm': p.wheel_diameter * 1000,
            'num_blades': p.num_blades,
            'rpm': p.rpm,
            'tip_speed_m_s': p.tip_speed,
            'g_force': p.g_force,
            'd50_um': d50 * 1e6,
            'target_d50_um': p.target_d50 * 1e6,
            'rpm_for_target': rpm_for_target,
            'power_kW': power / 1000,
            'flow_m3_h': volumetric_flow * 3600,
            'blade_gap_mm': p.blade_gap * 1000,
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_standard_wheel_classifier(
    wheel_diameter: float = 0.20,
    target_d50_um: float = 25.0,
    rpm: float = 6000.0,
) -> WheelClassifier:
    """
    Create a standard wheel classifier for protein/starch separation.

    Based on Hosokawa Micron Separator MS design principles.

    Args:
        wheel_diameter: Wheel diameter [m] (default 200mm)
        target_d50_um: Target cut size [um] (default 25 um)
        rpm: Operating speed [RPM] (default 6000)

    Returns:
        WheelClassifier instance
    """
    params = WheelClassifierParams(
        wheel_diameter=wheel_diameter,
        wheel_width=wheel_diameter * 0.2,
        hub_diameter=wheel_diameter * 0.3,
        num_blades=24,
        blade_thickness=0.002,
        shroud_thickness=0.003,
        volute_clearance=wheel_diameter * 0.075,
        volute_expansion=1.25,
        housing_height=wheel_diameter * 0.5,
        feed_inlet_width=wheel_diameter * 0.25,
        feed_inlet_height=wheel_diameter * 0.3,
        feed_inlet_length=wheel_diameter * 0.4,
        fines_outlet_diameter=wheel_diameter * 0.25,
        coarse_hopper_height=wheel_diameter * 0.4,
        coarse_outlet_diameter=wheel_diameter * 0.2,
        rpm=rpm,
        target_d50=target_d50_um * 1e-6,
    )

    return WheelClassifier(params)


def create_high_speed_wheel_classifier(
    wheel_diameter: float = 0.15,
    target_d50_um: float = 15.0,
) -> WheelClassifier:
    """
    Create a high-speed wheel classifier for fine cuts (d50 < 20 um).

    For protein concentration where very fine separation is needed.

    Args:
        wheel_diameter: Wheel diameter [m] (default 150mm)
        target_d50_um: Target cut size [um] (default 15 um)

    Returns:
        WheelClassifier instance
    """
    params = WheelClassifierParams(
        wheel_diameter=wheel_diameter,
        wheel_width=wheel_diameter * 0.15,
        hub_diameter=wheel_diameter * 0.35,
        num_blades=36,  # More blades for sharper cut
        blade_thickness=0.0015,
        shroud_thickness=0.002,
        volute_clearance=wheel_diameter * 0.08,
        volute_expansion=1.2,
        housing_height=wheel_diameter * 0.45,
        feed_inlet_width=wheel_diameter * 0.2,
        feed_inlet_height=wheel_diameter * 0.25,
        feed_inlet_length=wheel_diameter * 0.35,
        fines_outlet_diameter=wheel_diameter * 0.22,
        coarse_hopper_height=wheel_diameter * 0.35,
        coarse_outlet_diameter=wheel_diameter * 0.15,
        rpm=10000.0,  # Higher speed for finer cut
        target_d50=target_d50_um * 1e-6,
    )

    return WheelClassifier(params)


def create_large_capacity_wheel_classifier(
    wheel_diameter: float = 0.30,
    target_d50_um: float = 35.0,
) -> WheelClassifier:
    """
    Create a large capacity wheel classifier for high throughput.

    For production-scale processing where throughput > 500 kg/h.

    Args:
        wheel_diameter: Wheel diameter [m] (default 300mm)
        target_d50_um: Target cut size [um] (default 35 um)

    Returns:
        WheelClassifier instance
    """
    params = WheelClassifierParams(
        wheel_diameter=wheel_diameter,
        wheel_width=wheel_diameter * 0.22,
        hub_diameter=wheel_diameter * 0.28,
        num_blades=32,
        blade_thickness=0.003,
        shroud_thickness=0.004,
        volute_clearance=wheel_diameter * 0.07,
        volute_expansion=1.3,
        housing_height=wheel_diameter * 0.55,
        feed_inlet_width=wheel_diameter * 0.3,
        feed_inlet_height=wheel_diameter * 0.35,
        feed_inlet_length=wheel_diameter * 0.45,
        fines_outlet_diameter=wheel_diameter * 0.28,
        coarse_hopper_height=wheel_diameter * 0.45,
        coarse_outlet_diameter=wheel_diameter * 0.22,
        rpm=5000.0,  # Lower speed for larger wheel
        target_d50=target_d50_um * 1e-6,
    )

    return WheelClassifier(params)
