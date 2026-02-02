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

    # Motor parameters
    motor_type: str = "electric"     # Motor type (electric, belt_driven)
    motor_diameter: float = None     # [m] Motor housing diameter (auto-calculated)
    motor_length: float = None       # [m] Motor body length (auto-calculated)

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
        
        # Auto-calculate motor dimensions based on shaft power
        if self.motor_diameter is None:
            # Motor frame size based on power (IEC standard sizing)
            power_kw = self.shaft_power / 1000
            if power_kw < 3:
                self.motor_diameter = 0.15  # ~IEC 90 frame
            elif power_kw < 7.5:
                self.motor_diameter = 0.20  # ~IEC 112 frame
            elif power_kw < 15:
                self.motor_diameter = 0.25  # ~IEC 132 frame
            else:
                self.motor_diameter = 0.30  # ~IEC 160 frame
        
        if self.motor_length is None:
            # Motor length proportional to diameter
            self.motor_length = self.motor_diameter * 1.5

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
    - Impeller with blades (ANIMATED - rotates based on RPM)
    - Scroll/volute casing (static)
    - Inlet cone (static)
    - Outlet duct (static)
    - Electric motor BELOW scroll with bearing pedestal (static)
    - Motor cooling fins, terminal box, mounting feet (static)

    Coordinate system:
    - Origin at center of impeller
    - Inlet along +Z axis (front)
    - Outlet along +X axis (standard arrangement)
    - Motor mounted BELOW scroll (-Y direction, industrial belt-drive style)
    
    Layout (side view):
                ┌─────────────┐
                │   SCROLL    │◄── Inlet (+Z)
                │  (impeller) │
                └──────┬──────┘
                       │ Bearing pedestal
                ┌──────┴──────┐
                │    MOTOR    │ ◄── Axis along Z
                └─────────────┘
    
    Drive mechanism:
    - Electric motor mounted below scroll on bearing pedestal
    - Belt/coupling connects motor shaft to impeller shaft
    - Motor provides torque to rotate impeller at specified RPM
    - Shaft power = (Q * ΔP) / η where Q=flow, ΔP=pressure rise, η=efficiency
    
    Animation (belt drive system):
    - Motor pulley rotates FAST (at motor_rpm = impeller_rpm * pulley_ratio)
    - Belt transmits rotation
    - Driven pulley rotates SLOW (same speed as impeller)
    - Impeller rotates with driven pulley (same shaft, around Z-axis)
    
    Animation methods:
    - update_animation(dt, rpm): Update all rotating components
    - get_impeller_mesh(angle): Get rotating impeller mesh
    - get_driven_pulley_mesh(angle): Get rotating driven pulley mesh
    - get_motor_pulley_mesh(angle): Get rotating motor pulley mesh
    - get_static_mesh(): Get non-moving parts (scroll, motor body, belt, supports)
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
        
        # Animation state for belt drive system
        self._impeller_angle = 0.0      # Impeller/driven pulley angle [radians]
        self._motor_angle = 0.0         # Motor pulley angle [radians]
        
        # Belt drive ratio (motor pulley smaller = motor runs faster)
        # driven_diameter / motor_diameter determines the speed reduction
        self._pulley_ratio = 2.0  # Motor spins 2x faster than impeller
        
        # Cached separate meshes for animation
        self._static_vertices = None
        self._static_indices = None
        self._static_normals = None
        self._impeller_vertices = None
        self._impeller_indices = None
        self._impeller_normals = None
        
        # Drive train mesh caches (for animation)
        self._motor_pulley_vertices = None
        self._motor_pulley_indices = None
        self._motor_pulley_normals = None
        self._driven_pulley_vertices = None
        self._driven_pulley_indices = None
        self._driven_pulley_normals = None

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
        
        # Generate electric motor (drives impeller)
        self._generate_motor(vertices, indices, normals)

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
        
        # Add proper outlet casing extending toward +X direction
        self._generate_scroll_outlet_casing(vertices, indices, normals)

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

    def _generate_scroll_outlet_casing(self, vertices: List, indices: List, normals: List):
        """
        Generate scroll outlet casing extending toward +X direction.
        
        This creates the proper housing that transitions from the circular scroll
        volute to a rectangular outlet duct. In real centrifugal blowers, this is
        the "discharge casing" or "scroll tongue" area where air exits tangentially.
        
        Geometry (top view, looking at -Y):
        
                     +Z
                      │
            ┌─────────┼─────────┐
            │  SCROLL │ CASING  │════════► +X (to ductwork)
            │    ○    │         │  Outlet casing
            │ impeller│         │
            └─────────┼─────────┘
                      │
                     -Z
        
        The outlet casing:
        1. Extends from the scroll outer edge at theta=0 (where spiral ends)
        2. Creates a rectangular channel toward +X
        3. Has top, bottom, front and back walls
        4. Ends with a flange face for duct connection
        """
        p = self.params
        n_segments = 8  # Resolution for transition
        
        # Scroll geometry
        r_scroll = p.scroll_inner_radius * p.scroll_expansion
        half_width = p.impeller_width / 2 * 1.2  # Scroll axial half-width
        
        # Outlet casing dimensions
        # Height matches the scroll width (Z dimension becomes Y in outlet)
        casing_height = p.outlet_height if p.outlet_height else half_width * 2 * 0.8
        casing_width = p.outlet_width if p.outlet_width else half_width * 2
        casing_length = p.outlet_height * 2.0  # Length extending toward +X
        
        hw = casing_width / 2   # Half width (Z direction)
        hh = casing_height / 2  # Half height (Y direction)
        
        # Starting position: tangent to scroll at theta=0 (+X side)
        # The scroll at theta=0 has radius r_scroll at position (center_x + r_scroll, center_y)
        x_start = p.center[0] + r_scroll  # At scroll outer edge
        x_end = x_start + casing_length    # At flange/outlet face
        y_center = p.center[1]             # Centered vertically on scroll axis
        z_center = p.center[2]             # Centered axially
        
        # ============================================================
        # Generate transition from scroll to rectangular casing
        # ============================================================
        # The transition blends from the curved scroll surface to flat walls
        
        casing_start = len(vertices)
        
        # Generate casing as a series of profiles from scroll to outlet
        for i in range(n_segments + 1):
            t = i / n_segments  # 0 = at scroll, 1 = at outlet
            
            # X position along casing length
            x = x_start + t * casing_length
            
            # Blend from scroll curvature to rectangular
            # At t=0 (scroll): profile follows scroll curve
            # At t=1 (outlet): profile is rectangular
            
            # Top edge Y position
            # At scroll: top follows scroll curve
            # At outlet: flat at y_center + hh
            y_scroll_top = p.center[1] + np.sqrt(max(0, r_scroll**2 - (x_start - p.center[0])**2)) if t == 0 else 0
            y_top = p.center[1] + hh  # Target rectangular top
            if t < 0.3:
                # Transition zone: blend from scroll curve to flat
                blend = t / 0.3
                y_actual_top = y_scroll_top * (1 - blend) + y_top * blend if y_scroll_top > 0 else y_top
            else:
                y_actual_top = y_top
            
            # Bottom edge Y position  
            y_scroll_bottom = p.center[1] - np.sqrt(max(0, r_scroll**2 - (x_start - p.center[0])**2)) if t == 0 else 0
            y_bottom = p.center[1] - hh  # Target rectangular bottom
            if t < 0.3:
                blend = t / 0.3
                y_actual_bottom = y_scroll_bottom * (1 - blend) + y_bottom * blend if y_scroll_bottom < 0 else y_bottom
            else:
                y_actual_bottom = y_bottom
            
            # Width (Z) - stays roughly constant but can taper slightly
            z_half = half_width * (1 - t * 0.1) + hw * t * 0.1  # Slight blend to outlet width
            z_half = hw  # Keep width constant for cleaner geometry
            
            # Add 4 corner vertices for this profile
            # Order: bottom-back, bottom-front, top-front, top-back
            vertices.append([x, y_actual_bottom, z_center - z_half])  # 0: bottom-back
            normals.append([0.0, -1.0, 0.0])
            vertices.append([x, y_actual_bottom, z_center + z_half])  # 1: bottom-front
            normals.append([0.0, -1.0, 0.0])
            vertices.append([x, y_actual_top, z_center + z_half])     # 2: top-front
            normals.append([0.0, 1.0, 0.0])
            vertices.append([x, y_actual_top, z_center - z_half])     # 3: top-back
            normals.append([0.0, 1.0, 0.0])
        
        # Generate wall surfaces connecting profiles
        pts_per_profile = 4
        for i in range(n_segments):
            base = casing_start + i * pts_per_profile
            next_base = casing_start + (i + 1) * pts_per_profile
            
            # Bottom wall (connects vertices 0 and 1 between profiles)
            v0, v1 = base + 0, base + 1
            v2, v3 = next_base + 0, next_base + 1
            indices.extend([v0, v2, v3])
            indices.extend([v0, v3, v1])
            
            # Top wall (connects vertices 2 and 3 between profiles)
            v0, v1 = base + 3, base + 2
            v2, v3 = next_base + 3, next_base + 2
            indices.extend([v0, v1, v3])
            indices.extend([v0, v3, v2])
            
            # Back wall (-Z side, connects vertices 0 and 3 between profiles)
            v0, v1 = base + 0, base + 3
            v2, v3 = next_base + 0, next_base + 3
            indices.extend([v0, v1, v3])
            indices.extend([v0, v3, v2])
            
            # Front wall (+Z side, connects vertices 1 and 2 between profiles)
            v0, v1 = base + 1, base + 2
            v2, v3 = next_base + 1, next_base + 2
            indices.extend([v0, v3, v1])
            indices.extend([v0, v2, v3])
        
        # ============================================================
        # End cap / flange face at outlet (for duct connection)
        # ============================================================
        flange_start = len(vertices)
        
        # Flange is at x_end, rectangular
        flange_verts = [
            [x_end, y_center - hh, z_center - hw],  # 0: bottom-back
            [x_end, y_center - hh, z_center + hw],  # 1: bottom-front
            [x_end, y_center + hh, z_center + hw],  # 2: top-front
            [x_end, y_center + hh, z_center - hw],  # 3: top-back
        ]
        
        for v in flange_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])  # Facing +X
        
        # Flange face triangles
        indices.extend([flange_start, flange_start + 1, flange_start + 2])
        indices.extend([flange_start, flange_start + 2, flange_start + 3])
        
        # ============================================================
        # Connect casing to scroll body at the starting edge
        # ============================================================
        # Add a curved transition piece that blends into the scroll surface
        self._generate_scroll_casing_blend(vertices, indices, normals, x_start, y_center, z_center, hw, hh, r_scroll, half_width)

    def _generate_scroll_casing_blend(self, vertices: List, indices: List, normals: List,
                                       x_start: float, y_center: float, z_center: float,
                                       hw: float, hh: float, r_scroll: float, scroll_half_width: float):
        """
        Generate a smooth blend between the circular scroll and rectangular casing.
        
        This creates a curved surface that transitions from the scroll's cylindrical
        shape to the casing's rectangular cross-section.
        """
        p = self.params
        n_blend = 6  # Resolution for blend
        
        blend_start = len(vertices)
        
        # The blend covers the area from the scroll surface to the casing entrance
        # It wraps around the +X portion of the scroll (from about -45° to +45° around X axis)
        
        angle_range = PI / 3  # 60 degrees total (30° above and below)
        
        for i in range(n_blend + 1):
            t = i / n_blend  # 0 = bottom of blend, 1 = top of blend
            
            # Angle on scroll circle (from -angle_range/2 to +angle_range/2, centered at theta=0)
            angle = -angle_range / 2 + t * angle_range
            
            # Position on scroll surface
            scroll_x = p.center[0] + r_scroll * np.cos(angle)
            scroll_y = p.center[1] + r_scroll * np.sin(angle)
            
            # Target position on casing entrance
            casing_y = y_center - hh + t * 2 * hh
            
            # For each point, generate front and back vertices (along Z)
            for z_sign in [-1, 1]:
                z = z_center + z_sign * scroll_half_width  # At scroll width
                
                vertices.append([scroll_x, scroll_y, z])
                # Normal points outward from scroll center
                normals.append([np.cos(angle), np.sin(angle), 0.0])
        
        # Generate triangles for the blend surface
        for i in range(n_blend):
            base = blend_start + i * 2
            next_base = blend_start + (i + 1) * 2
            
            # Back face (-Z)
            indices.extend([base, next_base, next_base + 1])
            indices.extend([base, next_base + 1, base + 1])
        
        # Add side caps for the blend (connect to scroll sidewalls)
        # These fill the gap between the blend edge and the casing walls
        for z_sign, z_dir in [(-1, -1.0), (1, 1.0)]:
            cap_start = len(vertices)
            z = z_center + z_sign * scroll_half_width
            
            # Triangle fan from scroll edge to casing corner
            # Center point at casing corner
            casing_corner_y = y_center - hh if z_sign == -1 else y_center + hh
            vertices.append([x_start, y_center, z])
            normals.append([0.0, 0.0, z_dir])
            
            # Add points along the scroll edge
            for i in range(n_blend + 1):
                t = i / n_blend
                angle = -angle_range / 2 + t * angle_range
                scroll_x = p.center[0] + r_scroll * np.cos(angle)
                scroll_y = p.center[1] + r_scroll * np.sin(angle)
                
                vertices.append([scroll_x, scroll_y, z])
                normals.append([0.0, 0.0, z_dir])
            
            # Create triangles
            for i in range(n_blend):
                if z_sign > 0:
                    indices.extend([cap_start, cap_start + 1 + i, cap_start + 2 + i])
                else:
                    indices.extend([cap_start, cap_start + 2 + i, cap_start + 1 + i])

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
        """
        Generate inlet cone/bell on the -Z side of the scroll.
        
        The inlet bell is a FLARED CONE that:
        - Starts SMALL at the scroll (connects to impeller eye)
        - Flares OUTWARD to LARGE at the opening (receives air from duct)
        
        Position: -Z side of scroll, so duct from filter/elbow can connect.
        
        Geometry (side view, looking at XZ plane):
        
            [FILTER] → [ELBOW] → [DUCT] →  INLET BELL  → [SCROLL] → [OUTLET]
                                           
                        ← Air flow direction
                        
            z_end (outer)     ╭────────────╮   LARGE opening (bell mouth) - toward duct
                              │   ╱    ╲   │   Flared cone
                              │  ╱      ╲  │
            z_start (scroll)  ╰─╱────────╲─╯   SMALL (impeller eye diameter) - at scroll
                                │        │
                              [IMPELLER in scroll]
                              
            ←───────────────────────────────→  Z axis
            -Z (toward filter)      +Z (toward outlet/dampers)
        """
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial // 2

        inlet_start = len(vertices)

        # Inlet bell dimensions
        inlet_bell_length = p.inlet_diameter * 0.5
        inlet_outer_diameter = p.inlet_diameter * 1.3  # Flared opening (larger)

        # Position: -Z side of scroll, extending toward -Z (toward incoming duct)
        scroll_half_width = p.impeller_width / 2 * 1.2
        z_scroll_edge = p.center[2] - scroll_half_width  # At -Z edge of scroll
        z_bell_opening = z_scroll_edge - inlet_bell_length  # Bell opening (toward -Z, toward duct)

        for i in range(n_axial + 1):
            t = i / n_axial
            # Go from scroll edge (t=0) to bell opening (t=1)
            z = z_scroll_edge - t * inlet_bell_length
            
            # Bell flares OUT from scroll to opening
            # At t=0 (z_scroll_edge): r = inlet_radius (small, matches impeller eye)
            # At t=1 (z_bell_opening): r = inlet_outer_diameter/2 (large, flared bell)
            r = p.inlet_radius + (inlet_outer_diameter / 2 - p.inlet_radius) * t

            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                y = p.center[1] + r * np.sin(theta)
                vertices.append([x, y, z])

                # Normal for conical surface (points outward and toward +Z since cone expands toward -Z)
                dr = inlet_outer_diameter / 2 - p.inlet_radius
                slant = np.sqrt(inlet_bell_length ** 2 + dr ** 2)
                n_z = dr / slant  # Positive Z component since cone expands toward -Z
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
        Generate outlet reinforcement/flange details.
        
        The main outlet casing is now generated by _generate_scroll_outlet_casing()
        which creates a proper rectangular duct extending from the scroll toward +X.
        
        This method adds additional flange details and reinforcement ribs around
        the outlet opening for a more realistic appearance.
        
        Geometry (top view, looking at -Y):
        
                     +Z
                      │
            ┌─────────┼─────────┐
            │  SCROLL │ CASING  │══[FLANGE]══► +X (to ductwork)
            │    ○    │         │  
            │ impeller│         │
            └─────────┼─────────┘
                      │
                     -Z
        """
        p = self.params
        
        # Scroll geometry - must match _generate_scroll_outlet_casing
        r_scroll = p.scroll_inner_radius * p.scroll_expansion
        half_width = p.impeller_width / 2 * 1.2
        
        # Outlet casing dimensions - must match _generate_scroll_outlet_casing
        casing_height = p.outlet_height if p.outlet_height else half_width * 2 * 0.8
        casing_width = p.outlet_width if p.outlet_width else half_width * 2
        casing_length = p.outlet_height * 2.0
        
        hw = casing_width / 2
        hh = casing_height / 2
        
        x_start = p.center[0] + r_scroll
        x_end = x_start + casing_length
        y_center = p.center[1]
        z_center = p.center[2]
        
        # ============================================================
        # Add flange ring around outlet opening
        # ============================================================
        flange_thickness = 0.015  # 15mm thick flange
        flange_width = 0.025     # 25mm wide flange lip
        
        flange_start = len(vertices)
        
        # Outer flange rectangle
        flange_outer_verts = [
            [x_end, y_center - hh - flange_width, z_center - hw - flange_width],
            [x_end, y_center - hh - flange_width, z_center + hw + flange_width],
            [x_end, y_center + hh + flange_width, z_center + hw + flange_width],
            [x_end, y_center + hh + flange_width, z_center - hw - flange_width],
            [x_end + flange_thickness, y_center - hh - flange_width, z_center - hw - flange_width],
            [x_end + flange_thickness, y_center - hh - flange_width, z_center + hw + flange_width],
            [x_end + flange_thickness, y_center + hh + flange_width, z_center + hw + flange_width],
            [x_end + flange_thickness, y_center + hh + flange_width, z_center - hw - flange_width],
        ]
        
        for v in flange_outer_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])
        
        # Flange front face (annular - outer rectangle minus inner opening)
        # Add inner rectangle vertices for the opening
        inner_start = len(vertices)
        inner_verts = [
            [x_end + flange_thickness, y_center - hh, z_center - hw],
            [x_end + flange_thickness, y_center - hh, z_center + hw],
            [x_end + flange_thickness, y_center + hh, z_center + hw],
            [x_end + flange_thickness, y_center + hh, z_center - hw],
        ]
        
        for v in inner_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])
        
        # Front face of flange (4 trapezoids around the opening)
        # Bottom strip
        indices.extend([flange_start + 4, flange_start + 5, inner_start + 1])
        indices.extend([flange_start + 4, inner_start + 1, inner_start + 0])
        # Top strip
        indices.extend([inner_start + 3, inner_start + 2, flange_start + 6])
        indices.extend([inner_start + 3, flange_start + 6, flange_start + 7])
        # Left strip (-Z)
        indices.extend([flange_start + 4, inner_start + 0, inner_start + 3])
        indices.extend([flange_start + 4, inner_start + 3, flange_start + 7])
        # Right strip (+Z)
        indices.extend([inner_start + 1, flange_start + 5, flange_start + 6])
        indices.extend([inner_start + 1, flange_start + 6, inner_start + 2])
        
        # Outer edges of flange
        # Bottom
        indices.extend([flange_start + 0, flange_start + 4, flange_start + 5])
        indices.extend([flange_start + 0, flange_start + 5, flange_start + 1])
        # Top
        indices.extend([flange_start + 2, flange_start + 6, flange_start + 7])
        indices.extend([flange_start + 2, flange_start + 7, flange_start + 3])
        # Left (-Z)
        indices.extend([flange_start + 0, flange_start + 3, flange_start + 7])
        indices.extend([flange_start + 0, flange_start + 7, flange_start + 4])
        # Right (+Z)
        indices.extend([flange_start + 1, flange_start + 5, flange_start + 6])
        indices.extend([flange_start + 1, flange_start + 6, flange_start + 2])

    def _generate_motor(self, vertices: List, indices: List, normals: List):
        """
        Generate electric motor assembly with complete belt drive system.
        
        Industrial centrifugal blowers have the motor mounted BELOW the scroll
        housing, with a belt drive connecting motor shaft to impeller shaft.
        
        DRIVE TRAIN (side view):
        
                    ┌─────────────┐
                    │   SCROLL    │◄── Inlet (+Z)
                    │  (impeller) │
                    └──────┬──────┘
                           │ Impeller Shaft
                    ┌──────┴──────┐
                    │   BEARING   │◄── Bearing Housing (supports shaft)
                    │   HOUSING   │
                    └──────┬──────┘
                    ┌──────┴──────┐
                    │   DRIVEN    │◄── Large Pulley (on impeller shaft)
                    │   PULLEY    │
                    └──────┬──────┘
                           ║
                        V-BELT ════╗  ◄── Transmits power
                           ║       ║
                    ┌──────┴──────┐║
                    │   MOTOR     │║◄── Small Pulley (on motor shaft)
                    │   PULLEY    │╝
                    └──────┬──────┘
                    ┌──────┴──────┐
                    │    MOTOR    │
                    │ ═══════════ │◄── Cooling fins
                    └──────┬──────┘
                        ═══╧═══    ◄── Mounting feet
        
        Belt Drive Principle:
        - Motor pulley (small) rotates at motor RPM
        - Driven pulley (large) rotates at impeller RPM
        - Ratio = D_driven / D_motor determines speed reduction
        - V-belt transmits torque from motor to impeller
        
        Components:
        1. Impeller shaft (extends from hub through scroll back)
        2. Bearing housing (supports shaft, mounts to scroll)
        3. Driven pulley (large, on impeller shaft)
        4. V-belt (connects pulleys)
        5. Motor pulley (small, on motor shaft)
        6. Motor housing with cooling fins
        7. Terminal box and mounting feet
        """
        p = self.params
        cx, cy, cz = p.center
        
        # Motor is mounted BELOW the scroll (-Y direction)
        # Shaft extends HORIZONTALLY from scroll back (-Z direction)
        scroll_outer_radius = p.scroll_inner_radius * p.scroll_expansion
        scroll_half_width = p.impeller_width / 2 * 1.2
        
        motor_radius = p.motor_diameter / 2
        motor_length = p.motor_length
        
        # Belt drive geometry - VERTICAL pulleys (like wheels)
        # Pulleys rotate around Z-axis (same as impeller)
        driven_pulley_diameter = motor_radius * 2.2  # Large pulley (slower)
        motor_pulley_diameter = motor_radius * 1.1   # Small pulley (faster)
        pulley_width = 0.035  # 35mm wide V-belt pulley
        
        # Shaft diameter based on torque requirements
        shaft_diameter = 0.03  # 30mm shaft
        
        # The shaft exits through the BACK of the scroll (along -Z)
        scroll_back_z = cz - scroll_half_width
        shaft_length = 0.10  # Shaft extends 100mm behind scroll
        shaft_end_z = scroll_back_z - shaft_length  # Where pulley attaches
        
        # Bearing housing at scroll back
        bearing_depth = 0.04  # 40mm deep bearing housing
        
        # Driven pulley position (at end of shaft)
        driven_pulley_z = shaft_end_z - pulley_width / 2
        
        # Motor position - BELOW the scroll, shaft parallel to impeller shaft
        # Motor center is below and slightly behind scroll
        belt_center_distance = 0.18  # Vertical distance between pulley centers
        motor_cy = cy - scroll_outer_radius - motor_radius - 0.08  # Below scroll
        motor_cz = driven_pulley_z  # Same Z as driven pulley (aligned for belt)
        
        # Motor extends along Z axis (shaft toward +Z, body toward -Z)
        motor_z_end = motor_cz + motor_length / 2    # Toward scroll
        motor_z_start = motor_cz - motor_length / 2  # Away from scroll
        
        # Motor pulley at drive end (+Z side of motor)
        motor_pulley_z = motor_z_end + 0.01 + pulley_width / 2
        
        n_segments = 16
        
        # ============================================================
        # 1. IMPELLER SHAFT (extends HORIZONTALLY from scroll back, along -Z)
        # ============================================================
        # The shaft connects the impeller hub to the driven pulley
        # Shaft axis is along Z (same as impeller rotation axis)
        shaft_start = len(vertices)
        shaft_radius = shaft_diameter / 2
        
        # Generate shaft cylinder (horizontal, along -Z)
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + shaft_radius * nx
            y = cy + shaft_radius * ny
            
            # At scroll back
            vertices.append([x, y, scroll_back_z])
            normals.append([nx, ny, 0.0])
            
            # At shaft end (where pulley attaches)
            vertices.append([x, y, shaft_end_z])
            normals.append([nx, ny, 0.0])
        
        # Connect shaft cylinder
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = shaft_start + i * 2
            v1 = shaft_start + i * 2 + 1
            v2 = shaft_start + i_next * 2
            v3 = shaft_start + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # ============================================================
        # 2. BEARING HOUSING (supports shaft, mounts to scroll back)
        # ============================================================
        bearing_start = len(vertices)
        bearing_outer_radius = shaft_radius * 3
        
        # Bearing housing cylinder (at scroll back, extends along -Z)
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + bearing_outer_radius * nx
            y = cy + bearing_outer_radius * ny
            
            # At scroll back
            vertices.append([x, y, scroll_back_z])
            normals.append([nx, ny, 0.0])
            
            # Bearing front face
            vertices.append([x, y, scroll_back_z - bearing_depth])
            normals.append([nx, ny, 0.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = bearing_start + i * 2
            v1 = bearing_start + i * 2 + 1
            v2 = bearing_start + i_next * 2
            v3 = bearing_start + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # Bearing front cap (annular, at -Z end)
        bearing_cap_start = len(vertices)
        bearing_cap_z = scroll_back_z - bearing_depth
        
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            # Outer edge
            vertices.append([cx + bearing_outer_radius * np.cos(theta), 
                           cy + bearing_outer_radius * np.sin(theta), bearing_cap_z])
            normals.append([0.0, 0.0, -1.0])
            # Inner edge (shaft hole)
            vertices.append([cx + shaft_radius * 1.5 * np.cos(theta),
                           cy + shaft_radius * 1.5 * np.sin(theta), bearing_cap_z])
            normals.append([0.0, 0.0, -1.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = bearing_cap_start + i * 2
            v1 = bearing_cap_start + i * 2 + 1
            v2 = bearing_cap_start + i_next * 2
            v3 = bearing_cap_start + i_next * 2 + 1
            indices.extend([v0, v1, v2])
            indices.extend([v1, v3, v2])
        
        # ============================================================
        # 3. DRIVEN PULLEY (VERTICAL disc, like a wheel)
        # ============================================================
        # Pulley rotates around Z-axis, faces are in XY plane
        pulley_start = len(vertices)
        driven_radius = driven_pulley_diameter / 2
        
        pulley_front_z = driven_pulley_z + pulley_width / 2  # Toward scroll
        pulley_back_z = driven_pulley_z - pulley_width / 2   # Away from scroll
        
        # Pulley outer rim (cylinder around Z axis)
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + driven_radius * nx
            y = cy + driven_radius * ny
            
            # Front face
            vertices.append([x, y, pulley_front_z])
            normals.append([nx, ny, 0.0])
            # Back face
            vertices.append([x, y, pulley_back_z])
            normals.append([nx, ny, 0.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = pulley_start + i * 2
            v1 = pulley_start + i * 2 + 1
            v2 = pulley_start + i_next * 2
            v3 = pulley_start + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # Pulley front and back faces (annular discs with V-groove represented)
        for cap_z, cap_dir in [(pulley_front_z, 1.0), (pulley_back_z, -1.0)]:
            cap_start = len(vertices)
            for i in range(n_segments):
                theta = (i / n_segments) * TWO_PI
                # Outer edge
                vertices.append([cx + driven_radius * np.cos(theta),
                               cy + driven_radius * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
                # Inner edge (hub)
                vertices.append([cx + shaft_radius * 2 * np.cos(theta),
                               cy + shaft_radius * 2 * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
            
            for i in range(n_segments):
                i_next = (i + 1) % n_segments
                v0 = cap_start + i * 2
                v1 = cap_start + i * 2 + 1
                v2 = cap_start + i_next * 2
                v3 = cap_start + i_next * 2 + 1
                
                if cap_dir > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v1, v3, v2])
        
        # ============================================================
        # 4. V-BELT (wraps around vertical pulleys)
        # ============================================================
        # Belt wraps around outer circumference of both pulleys
        # Both pulleys at same Z, belt runs vertically (Y direction)
        belt_width = pulley_width * 0.8  # Belt slightly narrower than pulley
        motor_pulley_radius = motor_pulley_diameter / 2
        
        # Belt Z position (same as pulleys)
        belt_z = driven_pulley_z
        
        # Belt runs from driven pulley (at cy) down to motor pulley (at motor_cy)
        # Left and right sides of belt (tangent to pulleys)
        
        belt_start = len(vertices)
        
        # Left side of belt (-X tangent points)
        belt_left_x = cx - driven_radius  # Tangent to driven pulley
        motor_belt_left_x = cx - motor_pulley_radius  # Tangent to motor pulley
        
        belt_left_verts = [
            [belt_left_x, cy, belt_z - belt_width/2],
            [belt_left_x, cy, belt_z + belt_width/2],
            [motor_belt_left_x, motor_cy, belt_z + belt_width/2],
            [motor_belt_left_x, motor_cy, belt_z - belt_width/2],
        ]
        
        for v in belt_left_verts:
            vertices.append(v)
            normals.append([-1.0, 0.0, 0.0])
        
        indices.extend([belt_start, belt_start + 1, belt_start + 2])
        indices.extend([belt_start, belt_start + 2, belt_start + 3])
        
        # Right side of belt (+X tangent points)
        belt_right_start = len(vertices)
        belt_right_x = cx + driven_radius
        motor_belt_right_x = cx + motor_pulley_radius
        
        belt_right_verts = [
            [belt_right_x, cy, belt_z - belt_width/2],
            [belt_right_x, cy, belt_z + belt_width/2],
            [motor_belt_right_x, motor_cy, belt_z + belt_width/2],
            [motor_belt_right_x, motor_cy, belt_z - belt_width/2],
        ]
        
        for v in belt_right_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])
        
        indices.extend([belt_right_start, belt_right_start + 2, belt_right_start + 1])
        indices.extend([belt_right_start, belt_right_start + 3, belt_right_start + 2])
        
        # ============================================================
        # 5. MOTOR PULLEY (VERTICAL disc, same Z as driven pulley)
        # ============================================================
        motor_pulley_start = len(vertices)
        
        mp_front_z = belt_z + pulley_width / 2
        mp_back_z = belt_z - pulley_width / 2
        
        # Motor pulley outer rim
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + motor_pulley_radius * nx
            y = motor_cy + motor_pulley_radius * ny
            
            vertices.append([x, y, mp_front_z])
            normals.append([nx, ny, 0.0])
            vertices.append([x, y, mp_back_z])
            normals.append([nx, ny, 0.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = motor_pulley_start + i * 2
            v1 = motor_pulley_start + i * 2 + 1
            v2 = motor_pulley_start + i_next * 2
            v3 = motor_pulley_start + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # Motor pulley faces (annular)
        for cap_z, cap_dir in [(mp_front_z, 1.0), (mp_back_z, -1.0)]:
            cap_start = len(vertices)
            for i in range(n_segments):
                theta = (i / n_segments) * TWO_PI
                # Outer edge
                vertices.append([cx + motor_pulley_radius * np.cos(theta),
                               motor_cy + motor_pulley_radius * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
                # Inner edge (hub)
                vertices.append([cx + shaft_radius * 2 * np.cos(theta),
                               motor_cy + shaft_radius * 2 * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
            
            for i in range(n_segments):
                i_next = (i + 1) % n_segments
                v0 = cap_start + i * 2
                v1 = cap_start + i * 2 + 1
                v2 = cap_start + i_next * 2
                v3 = cap_start + i_next * 2 + 1
                
                if cap_dir > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v1, v3, v2])
        
        # ============================================================
        # 5b. SUPPORT FRAME (connects motor to scroll)
        # ============================================================
        scroll_bottom_y = cy - scroll_outer_radius
        
        # Vertical support columns
        support_width = 0.025
        support_spacing = driven_radius + 0.02
        
        for side in [-1, 1]:
            support_start = len(vertices)
            sx = cx + side * support_spacing
            
            hw = support_width / 2
            
            support_top_y = scroll_bottom_y
            support_bottom_y = motor_cy - motor_radius - 0.03
            
            support_verts = [
                [sx - hw, support_bottom_y, belt_z - hw],
                [sx + hw, support_bottom_y, belt_z - hw],
                [sx + hw, support_bottom_y, belt_z + hw],
                [sx - hw, support_bottom_y, belt_z + hw],
                [sx - hw, support_top_y, belt_z - hw],
                [sx + hw, support_top_y, belt_z - hw],
                [sx + hw, support_top_y, belt_z + hw],
                [sx - hw, support_top_y, belt_z + hw],
            ]
            
            for v in support_verts:
                vertices.append(v)
                normals.append([float(side), 0.0, 0.0])
            
            # Front (-Z)
            indices.extend([support_start + 0, support_start + 4, support_start + 5])
            indices.extend([support_start + 0, support_start + 5, support_start + 1])
            # Back (+Z)
            indices.extend([support_start + 2, support_start + 6, support_start + 7])
            indices.extend([support_start + 2, support_start + 7, support_start + 3])
            # Outer
            if side > 0:
                indices.extend([support_start + 1, support_start + 5, support_start + 6])
                indices.extend([support_start + 1, support_start + 6, support_start + 2])
            else:
                indices.extend([support_start + 0, support_start + 3, support_start + 7])
                indices.extend([support_start + 0, support_start + 7, support_start + 4])
        
        # Base plate
        base_start = len(vertices)
        base_width = support_spacing * 2 + support_width * 2
        base_depth = motor_length * 0.9
        base_height = 0.015
        base_y = motor_cy - motor_radius - 0.03 - base_height
        
        base_verts = [
            [cx - base_width/2, base_y, belt_z - base_depth/2],
            [cx + base_width/2, base_y, belt_z - base_depth/2],
            [cx + base_width/2, base_y, belt_z + base_depth/2],
            [cx - base_width/2, base_y, belt_z + base_depth/2],
            [cx - base_width/2, base_y + base_height, belt_z - base_depth/2],
            [cx + base_width/2, base_y + base_height, belt_z - base_depth/2],
            [cx + base_width/2, base_y + base_height, belt_z + base_depth/2],
            [cx - base_width/2, base_y + base_height, belt_z + base_depth/2],
        ]
        
        for v in base_verts:
            vertices.append(v)
            normals.append([0.0, -1.0, 0.0])
        
        indices.extend([base_start + 4, base_start + 5, base_start + 6])
        indices.extend([base_start + 4, base_start + 6, base_start + 7])
        indices.extend([base_start + 0, base_start + 2, base_start + 1])
        indices.extend([base_start + 0, base_start + 3, base_start + 2])
        indices.extend([base_start + 0, base_start + 1, base_start + 5])
        indices.extend([base_start + 0, base_start + 5, base_start + 4])
        indices.extend([base_start + 2, base_start + 3, base_start + 7])
        indices.extend([base_start + 2, base_start + 7, base_start + 6])
        
        # ============================================================
        # 6. MOTOR HOUSING (cylindrical body along Z axis)
        # ============================================================
        motor_start = len(vertices)
        
        # Generate cylinder surface (axis along Z)
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            # Cylinder around Z-axis, offset to motor_cy
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + motor_radius * nx
            y = motor_cy + motor_radius * ny
            
            # Non-drive end (-Z)
            vertices.append([x, y, motor_z_start])
            normals.append([nx, ny, 0.0])
            
            # Drive end (+Z)
            vertices.append([x, y, motor_z_end])
            normals.append([nx, ny, 0.0])
        
        # Connect cylinder wall
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = motor_start + i * 2
            v1 = motor_start + i * 2 + 1
            v2 = motor_start + i_next * 2
            v3 = motor_start + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # ============================================================
        # 7. MOTOR END CAPS
        # ============================================================
        # Non-drive end cap (-Z)
        nde_start = len(vertices)
        vertices.append([cx, motor_cy, motor_z_start])
        normals.append([0.0, 0.0, -1.0])
        nde_center = nde_start
        
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            x = cx + motor_radius * 0.9 * np.cos(theta)
            y = motor_cy + motor_radius * 0.9 * np.sin(theta)
            vertices.append([x, y, motor_z_start])
            normals.append([0.0, 0.0, -1.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            indices.extend([nde_center, nde_start + 1 + i, nde_start + 1 + i_next])
        
        # Drive end cap (+Z)
        de_start = len(vertices)
        vertices.append([cx, motor_cy, motor_z_end])
        normals.append([0.0, 0.0, 1.0])
        de_center = de_start
        
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI
            x = cx + motor_radius * 0.9 * np.cos(theta)
            y = motor_cy + motor_radius * 0.9 * np.sin(theta)
            vertices.append([x, y, motor_z_end])
            normals.append([0.0, 0.0, 1.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            indices.extend([de_center, de_start + 1 + i_next, de_start + 1 + i])
        
        # ============================================================
        # 8. COOLING FINS (ribs on motor body)
        # ============================================================
        n_fins = 8
        fin_height = motor_radius * 0.12
        fin_width = 0.006  # 6mm wide fins
        
        for f in range(n_fins):
            theta = (f / n_fins) * TWO_PI
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            # Perpendicular direction for fin width
            px = -np.sin(theta)
            py = np.cos(theta)
            
            fin_start = len(vertices)
            
            r_inner = motor_radius
            r_outer = motor_radius + fin_height
            
            # Fin along the motor length (Z direction)
            z1 = motor_z_start + motor_length * 0.1
            z2 = motor_z_end - motor_length * 0.1
            
            fin_verts = [
                [cx + r_inner * nx + fin_width/2 * px, motor_cy + r_inner * ny + fin_width/2 * py, z1],
                [cx + r_inner * nx - fin_width/2 * px, motor_cy + r_inner * ny - fin_width/2 * py, z1],
                [cx + r_inner * nx - fin_width/2 * px, motor_cy + r_inner * ny - fin_width/2 * py, z2],
                [cx + r_inner * nx + fin_width/2 * px, motor_cy + r_inner * ny + fin_width/2 * py, z2],
                [cx + r_outer * nx + fin_width/2 * px, motor_cy + r_outer * ny + fin_width/2 * py, z1],
                [cx + r_outer * nx - fin_width/2 * px, motor_cy + r_outer * ny - fin_width/2 * py, z1],
                [cx + r_outer * nx - fin_width/2 * px, motor_cy + r_outer * ny - fin_width/2 * py, z2],
                [cx + r_outer * nx + fin_width/2 * px, motor_cy + r_outer * ny + fin_width/2 * py, z2],
            ]
            
            for vert in fin_verts:
                vertices.append(vert)
                normals.append([nx, ny, 0.0])
            
            # Outer face of fin
            indices.extend([fin_start + 4, fin_start + 5, fin_start + 6])
            indices.extend([fin_start + 4, fin_start + 6, fin_start + 7])
            
            # Side faces
            indices.extend([fin_start + 0, fin_start + 4, fin_start + 7])
            indices.extend([fin_start + 0, fin_start + 7, fin_start + 3])
            indices.extend([fin_start + 1, fin_start + 6, fin_start + 5])
            indices.extend([fin_start + 1, fin_start + 2, fin_start + 6])
        
        # ============================================================
        # 9. TERMINAL BOX (on +X side of motor)
        # ============================================================
        box_width = motor_radius * 0.5
        box_depth = motor_length * 0.25
        box_height = motor_radius * 0.35
        
        box_cx = cx + motor_radius + box_width / 2
        box_cy = motor_cy
        box_cz = motor_cz
        
        box_start = len(vertices)
        hw = box_width / 2
        hd = box_depth / 2
        hh = box_height / 2
        
        box_verts = [
            [box_cx - hw, box_cy - hh, box_cz - hd],
            [box_cx + hw, box_cy - hh, box_cz - hd],
            [box_cx + hw, box_cy - hh, box_cz + hd],
            [box_cx - hw, box_cy - hh, box_cz + hd],
            [box_cx - hw, box_cy + hh, box_cz - hd],
            [box_cx + hw, box_cy + hh, box_cz - hd],
            [box_cx + hw, box_cy + hh, box_cz + hd],
            [box_cx - hw, box_cy + hh, box_cz + hd],
        ]
        
        for v in box_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])
        
        # Box faces
        indices.extend([box_start + 4, box_start + 5, box_start + 6])
        indices.extend([box_start + 4, box_start + 6, box_start + 7])
        indices.extend([box_start + 0, box_start + 1, box_start + 2])
        indices.extend([box_start + 0, box_start + 2, box_start + 3])
        indices.extend([box_start + 2, box_start + 6, box_start + 7])
        indices.extend([box_start + 2, box_start + 7, box_start + 3])
        indices.extend([box_start + 0, box_start + 4, box_start + 5])
        indices.extend([box_start + 0, box_start + 5, box_start + 1])
        indices.extend([box_start + 1, box_start + 5, box_start + 6])
        indices.extend([box_start + 1, box_start + 6, box_start + 2])
        
        # ============================================================
        # 10. MOUNTING FEET (two feet on bottom of motor)
        # ============================================================
        foot_width = motor_radius * 0.35
        foot_length = motor_length * 0.25
        foot_height = motor_radius * 0.15
        
        for z_offset in [-motor_length * 0.3, motor_length * 0.3]:
            foot_start = len(vertices)
            foot_cx = cx
            foot_cy = motor_cy - motor_radius - foot_height / 2
            foot_cz = motor_cz + z_offset
            
            hfw = foot_width / 2
            hfl = foot_length / 2
            hfh = foot_height / 2
            
            foot_verts = [
                [foot_cx - hfw, foot_cy - hfh, foot_cz - hfl],
                [foot_cx + hfw, foot_cy - hfh, foot_cz - hfl],
                [foot_cx + hfw, foot_cy - hfh, foot_cz + hfl],
                [foot_cx - hfw, foot_cy - hfh, foot_cz + hfl],
                [foot_cx - hfw, foot_cy + hfh, foot_cz - hfl],
                [foot_cx + hfw, foot_cy + hfh, foot_cz - hfl],
                [foot_cx + hfw, foot_cy + hfh, foot_cz + hfl],
                [foot_cx - hfw, foot_cy + hfh, foot_cz + hfl],
            ]
            
            for v in foot_verts:
                vertices.append(v)
                normals.append([0.0, -1.0, 0.0])
            
            # Foot faces
            indices.extend([foot_start + 0, foot_start + 2, foot_start + 1])
            indices.extend([foot_start + 0, foot_start + 3, foot_start + 2])
            indices.extend([foot_start + 2, foot_start + 6, foot_start + 7])
            indices.extend([foot_start + 2, foot_start + 7, foot_start + 3])
            indices.extend([foot_start + 0, foot_start + 4, foot_start + 5])
            indices.extend([foot_start + 0, foot_start + 5, foot_start + 1])
            indices.extend([foot_start + 0, foot_start + 3, foot_start + 7])
            indices.extend([foot_start + 0, foot_start + 7, foot_start + 4])
            indices.extend([foot_start + 1, foot_start + 5, foot_start + 6])
            indices.extend([foot_start + 1, foot_start + 6, foot_start + 2])

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
    
    # =========================================================================
    # ANIMATION METHODS
    # =========================================================================
    
    def update_animation(self, dt: float, rpm: float):
        """
        Update belt drive system animation.
        
        Animation chain:
        1. Motor pulley rotates at motor_rpm (faster)
        2. Belt transmits rotation
        3. Driven pulley rotates at impeller_rpm (slower due to ratio)
        4. Impeller rotates with driven pulley (same shaft)
        
        Speed relationship:
        - motor_rpm = impeller_rpm * pulley_ratio
        - motor spins FASTER than impeller
        
        Args:
            dt: Time step [seconds]
            rpm: Current IMPELLER rotational speed [RPM]
        """
        # Impeller/driven pulley angular velocity
        impeller_omega = rpm * TWO_PI / 60.0
        
        # Motor spins faster by the pulley ratio
        motor_omega = impeller_omega * self._pulley_ratio
        
        # Update angles
        self._impeller_angle += impeller_omega * dt
        self._motor_angle += motor_omega * dt
        
        # Keep angles in [0, 2π] range
        self._impeller_angle = self._impeller_angle % TWO_PI
        self._motor_angle = self._motor_angle % TWO_PI
    
    def get_impeller_angle(self) -> float:
        """Get current impeller/driven pulley rotation angle [radians]."""
        return self._impeller_angle
    
    def get_motor_angle(self) -> float:
        """Get current motor pulley rotation angle [radians]."""
        return self._motor_angle
    
    def get_pulley_ratio(self) -> float:
        """Get belt drive pulley ratio (motor_rpm / impeller_rpm)."""
        return self._pulley_ratio
    
    def set_impeller_angle(self, angle: float):
        """
        Set impeller rotation angle directly.
        
        Args:
            angle: Rotation angle [radians]
        """
        self._impeller_angle = angle % TWO_PI
    
    def set_motor_angle(self, angle: float):
        """
        Set motor pulley rotation angle directly.
        
        Args:
            angle: Rotation angle [radians]
        """
        self._motor_angle = angle % TWO_PI
    
    def get_impeller_transform(self, angle: float = None) -> np.ndarray:
        """
        Get 4x4 transformation matrix for impeller rotation.
        
        The impeller rotates around the Z-axis centered at params.center.
        
        Args:
            angle: Rotation angle [radians]. Uses current angle if None.
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        if angle is None:
            angle = self._impeller_angle
        
        p = self.params
        cx, cy, cz = p.center
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotation around Z-axis centered at (cx, cy, cz):
        # 1. Translate to origin
        # 2. Rotate around Z
        # 3. Translate back
        # Combined into single matrix:
        transform = np.array([
            [cos_a, -sin_a, 0, cx - cx*cos_a + cy*sin_a],
            [sin_a,  cos_a, 0, cy - cx*sin_a - cy*cos_a],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def get_static_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for static (non-moving) parts of the blower.
        
        Static parts:
        - Scroll/volute casing
        - Inlet cone
        - Outlet duct
        
        Returns:
            Tuple of (vertices, indices, normals)
        """
        if self._static_vertices is None:
            self._generate_separated_meshes()
        
        return self._static_vertices, self._static_indices, self._static_normals
    
    def get_impeller_mesh(self, angle: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for the impeller (animated part) at specified rotation.
        
        The impeller includes:
        - Hub cylinder
        - Blades
        - Back shroud
        
        Args:
            angle: Rotation angle [radians]. Uses current angle if None.
            
        Returns:
            Tuple of (vertices, indices, normals) with rotation applied
        """
        if self._impeller_vertices is None:
            self._generate_separated_meshes()
        
        if angle is None:
            angle = self._impeller_angle
        
        if abs(angle) < 1e-6:
            # No rotation needed
            return self._impeller_vertices.copy(), self._impeller_indices.copy(), self._impeller_normals.copy()
        
        # Apply rotation transform to vertices and normals
        p = self.params
        cx, cy, cz = p.center
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotate vertices around Z-axis at center
        rotated_vertices = self._impeller_vertices.copy()
        rotated_normals = self._impeller_normals.copy()
        
        for i in range(len(rotated_vertices)):
            # Translate to origin
            x = rotated_vertices[i, 0] - cx
            y = rotated_vertices[i, 1] - cy
            
            # Rotate
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            
            # Translate back
            rotated_vertices[i, 0] = new_x + cx
            rotated_vertices[i, 1] = new_y + cy
            
            # Rotate normals (no translation)
            nx = rotated_normals[i, 0]
            ny = rotated_normals[i, 1]
            rotated_normals[i, 0] = nx * cos_a - ny * sin_a
            rotated_normals[i, 1] = nx * sin_a + ny * cos_a
        
        return rotated_vertices, self._impeller_indices.copy(), rotated_normals
    
    def get_driven_pulley_mesh(self, angle: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for the driven pulley (animated, same rotation as impeller).
        
        The driven pulley is on the same shaft as the impeller, so it rotates
        at the same speed.
        
        Args:
            angle: Rotation angle [radians]. Uses current impeller angle if None.
            
        Returns:
            Tuple of (vertices, indices, normals) with rotation applied
        """
        if angle is None:
            angle = self._impeller_angle
        
        p = self.params
        cx, cy, cz = p.center
        
        # Drive train geometry (must match _generate_motor)
        scroll_half_width = p.impeller_width / 2 * 1.2
        scroll_back_z = cz - scroll_half_width
        shaft_length = 0.10
        shaft_end_z = scroll_back_z - shaft_length
        
        motor_radius = p.motor_diameter / 2
        driven_pulley_diameter = motor_radius * 2.2
        pulley_width = 0.035
        shaft_diameter = 0.03
        shaft_radius = shaft_diameter / 2
        
        driven_pulley_z = shaft_end_z - pulley_width / 2
        driven_radius = driven_pulley_diameter / 2
        pulley_front_z = driven_pulley_z + pulley_width / 2
        pulley_back_z = driven_pulley_z - pulley_width / 2
        
        n_segments = 16
        vertices = []
        indices = []
        normals = []
        
        # Generate pulley cylinder
        start_idx = 0
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI + angle  # Apply rotation
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + driven_radius * nx
            y = cy + driven_radius * ny
            
            vertices.append([x, y, pulley_front_z])
            normals.append([nx, ny, 0.0])
            vertices.append([x, y, pulley_back_z])
            normals.append([nx, ny, 0.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + i_next * 2
            v3 = start_idx + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # Pulley faces
        for cap_z, cap_dir in [(pulley_front_z, 1.0), (pulley_back_z, -1.0)]:
            cap_start = len(vertices)
            for i in range(n_segments):
                theta = (i / n_segments) * TWO_PI + angle
                vertices.append([cx + driven_radius * np.cos(theta),
                               cy + driven_radius * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
                vertices.append([cx + shaft_radius * 2 * np.cos(theta),
                               cy + shaft_radius * 2 * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
            
            for i in range(n_segments):
                i_next = (i + 1) % n_segments
                v0 = cap_start + i * 2
                v1 = cap_start + i * 2 + 1
                v2 = cap_start + i_next * 2
                v3 = cap_start + i_next * 2 + 1
                
                if cap_dir > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v1, v3, v2])
        
        return (np.array(vertices, dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(normals, dtype=np.float32))
    
    def get_motor_pulley_mesh(self, angle: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh for the motor pulley (animated, faster rotation than impeller).
        
        The motor pulley rotates faster than the impeller by the pulley ratio.
        
        Args:
            angle: Rotation angle [radians]. Uses current motor angle if None.
            
        Returns:
            Tuple of (vertices, indices, normals) with rotation applied
        """
        if angle is None:
            angle = self._motor_angle
        
        p = self.params
        cx, cy, cz = p.center
        
        # Drive train geometry (must match _generate_motor)
        scroll_outer_radius = p.scroll_inner_radius * p.scroll_expansion
        scroll_half_width = p.impeller_width / 2 * 1.2
        scroll_back_z = cz - scroll_half_width
        shaft_length = 0.10
        shaft_end_z = scroll_back_z - shaft_length
        
        motor_radius = p.motor_diameter / 2
        motor_pulley_diameter = motor_radius * 1.1
        pulley_width = 0.035
        shaft_diameter = 0.03
        shaft_radius = shaft_diameter / 2
        
        driven_pulley_z = shaft_end_z - pulley_width / 2
        belt_z = driven_pulley_z
        
        belt_center_distance = 0.18
        motor_cy = cy - scroll_outer_radius - motor_radius - 0.08
        motor_pulley_radius = motor_pulley_diameter / 2
        
        mp_front_z = belt_z + pulley_width / 2
        mp_back_z = belt_z - pulley_width / 2
        
        n_segments = 16
        vertices = []
        indices = []
        normals = []
        
        # Generate motor pulley cylinder
        start_idx = 0
        for i in range(n_segments):
            theta = (i / n_segments) * TWO_PI + angle  # Apply rotation
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            x = cx + motor_pulley_radius * nx
            y = motor_cy + motor_pulley_radius * ny
            
            vertices.append([x, y, mp_front_z])
            normals.append([nx, ny, 0.0])
            vertices.append([x, y, mp_back_z])
            normals.append([nx, ny, 0.0])
        
        for i in range(n_segments):
            i_next = (i + 1) % n_segments
            v0 = start_idx + i * 2
            v1 = start_idx + i * 2 + 1
            v2 = start_idx + i_next * 2
            v3 = start_idx + i_next * 2 + 1
            
            indices.extend([v0, v2, v1])
            indices.extend([v1, v2, v3])
        
        # Motor pulley faces
        for cap_z, cap_dir in [(mp_front_z, 1.0), (mp_back_z, -1.0)]:
            cap_start = len(vertices)
            for i in range(n_segments):
                theta = (i / n_segments) * TWO_PI + angle
                vertices.append([cx + motor_pulley_radius * np.cos(theta),
                               motor_cy + motor_pulley_radius * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
                vertices.append([cx + shaft_radius * 2 * np.cos(theta),
                               motor_cy + shaft_radius * 2 * np.sin(theta), cap_z])
                normals.append([0.0, 0.0, cap_dir])
            
            for i in range(n_segments):
                i_next = (i + 1) % n_segments
                v0 = cap_start + i * 2
                v1 = cap_start + i * 2 + 1
                v2 = cap_start + i_next * 2
                v3 = cap_start + i_next * 2 + 1
                
                if cap_dir > 0:
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    indices.extend([v0, v1, v2])
                    indices.extend([v1, v3, v2])
        
        return (np.array(vertices, dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(normals, dtype=np.float32))
    
    def _generate_separated_meshes(self):
        """Generate separate meshes for static and animated parts."""
        p = self.params
        
        # Generate static parts (scroll, inlet, outlet, motor)
        static_verts = []
        static_indices = []
        static_normals = []
        
        self._generate_scroll(static_verts, static_indices, static_normals)
        self._generate_inlet(static_verts, static_indices, static_normals)
        self._generate_outlet(static_verts, static_indices, static_normals)
        self._generate_motor(static_verts, static_indices, static_normals)
        
        self._static_vertices = np.array(static_verts, dtype=np.float32)
        self._static_indices = np.array(static_indices, dtype=np.int32)
        self._static_normals = np.array(static_normals, dtype=np.float32)
        
        # Generate impeller parts (at angle=0) - ANIMATED, driven by motor
        impeller_verts = []
        impeller_indices = []
        impeller_normals = []
        
        self._generate_impeller(impeller_verts, impeller_indices, impeller_normals)
        
        self._impeller_vertices = np.array(impeller_verts, dtype=np.float32)
        self._impeller_indices = np.array(impeller_indices, dtype=np.int32)
        self._impeller_normals = np.array(impeller_normals, dtype=np.float32)

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
        - 'inlet': Axial inlet (inlet eye) - air enters here from -Z side
        - 'outlet': Scroll outlet (rectangular) - air exits here toward +X direction

        Returns:
            Dictionary of port name to ConnectionPort
        """
        from ..connection_ports import ConnectionPort, PortType

        p = self.params

        # Inlet port: at the bell opening on -Z side (where duct connects)
        # The inlet bell extends from scroll edge to bell opening:
        #   z_scroll_edge = center - impeller_width/2 * 1.2 (scroll edge at -Z)
        #   z_bell_opening = z_scroll_edge - inlet_bell_length (bell opening, further -Z)
        inlet_bell_length = p.inlet_diameter * 0.5
        scroll_half_width = p.impeller_width / 2 * 1.2
        inlet_z = p.center[2] - scroll_half_width - inlet_bell_length
        inlet_pos = (p.center[0], p.center[1], inlet_z)

        # Outlet port: at end of scroll outlet casing, facing +X
        # The outlet casing extends from the scroll outer edge toward +X
        # Position matches _generate_scroll_outlet_casing and _generate_outlet geometry:
        #   x_start = center[0] + r_scroll (at scroll outer edge)
        #   casing_length = outlet_height * 2.0
        #   flange_thickness = 0.015
        #   x_end = x_start + casing_length + flange_thickness (at flange face)
        scroll_r = p.scroll_inner_radius * p.scroll_expansion
        
        # Outlet casing dimensions - must match _generate_scroll_outlet_casing
        casing_height = p.outlet_height if p.outlet_height else scroll_half_width * 2 * 0.8
        casing_width = p.outlet_width if p.outlet_width else scroll_half_width * 2
        casing_length = p.outlet_height * 2.0
        flange_thickness = 0.015  # Must match _generate_outlet
        
        x_start = p.center[0] + scroll_r
        outlet_x_end = x_start + casing_length + flange_thickness
        outlet_y_center = p.center[1]  # Centered vertically on scroll axis
        outlet_z_center = p.center[2]  # Centered axially
        
        outlet_pos = (outlet_x_end, outlet_y_center, outlet_z_center)

        return {
            'inlet': ConnectionPort(
                position=inlet_pos,
                direction=(0.0, 0.0, -1.0),  # Air enters from -Z direction (duct approaches from -Z)
                diameter=p.inlet_diameter,
                port_type=PortType.CIRCULAR,
                name="blower_inlet",
                compatible_types=[PortType.CIRCULAR, PortType.FLANGED],
            ),
            'outlet': ConnectionPort(
                position=outlet_pos,
                direction=(1.0, 0.0, 0.0),  # Air exits toward +X
                diameter=np.sqrt(casing_width * casing_height * 4 / PI),  # Equivalent diameter
                width=casing_width,
                height=casing_height,
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
