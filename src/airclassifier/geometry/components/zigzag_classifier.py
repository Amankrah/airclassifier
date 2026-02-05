"""
Zigzag air classifier component for particle separation.

The zigzag classifier is the primary separation device in legume protein
fractionation. It uses a series of deflector plates to create turbulent
recirculation zones where particles are separated based on their terminal velocity.

Separation Mechanism:
---------------------
1. Air flows upward through the channel at velocity v_air
2. Particles are fed at a middle stage and experience:
   - Gravity force: F_g = m × g (downward)
   - Drag force: F_d = 0.5 × ρ_air × C_d × A_p × v_rel² (opposing motion)
3. At each deflector plate:
   - Flow accelerates through the throat (constriction)
   - Recirculation zone forms behind the plate
   - In recirculation zone: local velocity << bulk velocity
   - Particles with v_terminal > v_local settle downward
   - Particles with v_terminal < v_local are carried upward

Key Physics:
- Terminal velocity: v_t = √(4 × d_p × g × (ρ_p - ρ_f) / (3 × C_d × ρ_f))
- Stokes number: St = ρ_p × d_p² × U / (18 × μ × L)
- Cut size d50: Particle size where 50% goes to each outlet

References:
- Kaiser (1963): Original zigzag classifier theory
- Senden (1979): Stochastic model for particle trajectories
- Tomas & Gröger (2000): Grade efficiency correlations
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


class ZoneType(Enum):
    """Flow zone types in zigzag classifier."""
    THROAT = 1          # Accelerating flow between plates
    RECIRCULATION = 2   # Eddy zone behind deflector plate
    SEPARATION = 3      # Main separation zone (lower velocity)
    TRANSPORT = 4       # Straight channel sections
    INLET = 5           # Air inlet section
    OUTLET = 6          # Fines/coarse outlet section


@dataclass
class DeflectorPlate:
    """
    A single deflector plate in the zigzag classifier.

    Geometry:
    - Plate extends from wall into channel at an angle
    - Creates throat (constriction) and recirculation zone behind it
    - Alternating plates create the zigzag flow pattern
    """
    stage: int                          # Stage number (1-indexed)
    side: str                           # 'left' or 'right' wall
    base_position: Tuple[float, float]  # (x, y) of plate base on wall
    tip_position: Tuple[float, float]   # (x, y) of plate tip in channel
    angle: float                        # Plate angle from vertical [rad]
    length: float                       # Plate length [m]
    thickness: float                    # Plate thickness [m]

    @property
    def normal(self) -> Tuple[float, float]:
        """Normal vector pointing into separation zone (downstream)."""
        # For upward flow, downstream is above the plate
        if self.side == 'left':
            return (np.cos(self.angle), np.sin(self.angle))
        else:
            return (-np.cos(self.angle), np.sin(self.angle))


@dataclass
class SeparationZone:
    """
    Recirculation/separation zone behind a deflector plate.

    This is where the actual particle separation occurs:
    - Flow velocity is reduced due to recirculation
    - Particles experience multiple separation attempts
    - Heavy particles settle, light particles are re-entrained
    """
    stage: int
    plate: DeflectorPlate
    # Zone boundaries (approximate elliptical region behind plate)
    center: Tuple[float, float]         # Zone center (x, y)
    width: float                        # Zone width (perpendicular to flow)
    height: float                       # Zone height (along flow direction)
    # Flow characteristics
    velocity_ratio: float = 0.3         # v_zone / v_bulk (typically 0.2-0.4)
    turbulence_intensity: float = 0.25  # u'/U (fluctuation/mean)
    residence_time_factor: float = 2.5  # Increased residence vs transport


@dataclass
class ZigzagClassifierParams:
    """
    Parameters for zigzag air classifier with deflector plates.

    The zigzag classifier is characterized by:
    - Channel dimensions (width, depth, height per stage)
    - Deflector plate geometry (angle, length, thickness)
    - Number of stages and feed location
    - Inlet/outlet dimensions

    Design Guidelines (Senden, 1979):
    - Plate angle: 30-60° from vertical (45° typical)
    - Plate length: 0.4-0.6 × channel width (creates 40-60% blockage)
    - Stage height / channel width ratio: 1.0-2.0
    - Number of stages: 3-7 (more stages = sharper separation)
    """

    # Channel geometry
    channel_width: float         # [m] Width of main channel (b)
    channel_depth: float         # [m] Depth of channel into page (d)
    num_stages: int              # Number of deflector plate stages (3-7 typical)
    stage_height: float          # [m] Vertical distance between plates (h)

    # Deflector plate geometry - THE KEY DESIGN PARAMETERS
    plate_angle: float           # [rad] Plate angle from vertical (30-60°, ~0.52-1.05 rad)
    plate_length_ratio: float    # Plate length / channel width (0.4-0.6 typical)
    plate_thickness: float           # [m] Plate thickness

    # Inlet/outlet dimensions
    air_inlet_width: float       # [m] Bottom air inlet width
    air_inlet_height: float      # [m] Bottom air inlet height
    fines_outlet_width: float    # [m] Top fines outlet width
    fines_outlet_height: float   # [m] Top fines outlet height
    coarse_outlet_width: float   # [m] Bottom coarse outlet width
    coarse_outlet_height: float  # [m] Bottom coarse outlet height

    # Construction
    wall_thickness: float        # [m] Wall thickness
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bottom center

    # Feed entry (optional - only used for side-fed zigzag configurations)
    # For venturi-fed systems, particles enter via air_inlet from venturi below
    feed_stage: int = 1          # Stage number for feed entry (1-indexed, middle typical)
    feed_width: float = 0.0      # [m] Width of feed inlet (0 = no feed inlet)
    feed_angle: float = 0.0      # [rad] Feed angle from horizontal (0 = horizontal entry)
    include_feed_inlet: bool = False  # Whether to include side feed inlet geometry

    # Mesh resolution
    resolution_plate: int = 8    # Vertices along each plate edge
    resolution_depth: int = 4    # Vertices along depth

    # Separation zone parameters (from literature)
    recirculation_length_ratio: float = 1.5  # Separation zone length / plate length
    velocity_ratio_in_zone: float = 0.3      # v_zone / v_bulk
    turbulence_intensity: float = 0.25       # u'/U in separation zones

    def __post_init__(self):
        if self.include_feed_inlet:
            if self.feed_stage < 1 or self.feed_stage > self.num_stages:
                raise ValueError(f"Feed stage must be between 1 and {self.num_stages}")
            if self.feed_width <= 0:
                raise ValueError("Feed width must be > 0 when include_feed_inlet is True")
        if self.plate_angle <= 0 or self.plate_angle >= PI / 2:
            raise ValueError("Plate angle should be between 0 and 90° (0 to π/2 rad)")
        if self.plate_length_ratio <= 0 or self.plate_length_ratio >= 1.0:
            raise ValueError("Plate length ratio must be between 0 and 1")
        if self.num_stages < 2:
            raise ValueError("Need at least 2 stages for classification")

    @property
    def plate_length(self) -> float:
        """Actual plate length [m]."""
        return self.plate_length_ratio * self.channel_width

    @property
    def throat_width(self) -> float:
        """Width of throat (constriction) between plate tip and opposite wall [m]."""
        # Horizontal projection of plate into channel
        horizontal_extent = self.plate_length * np.sin(self.plate_angle)
        return self.channel_width - horizontal_extent

    @property
    def blockage_ratio(self) -> float:
        """Fraction of channel blocked by plate (0-1)."""
        return 1.0 - (self.throat_width / self.channel_width)

    @property
    def total_height(self) -> float:
        """Total height of classifier channel section."""
        return self.num_stages * self.stage_height

    @property
    def channel_cross_section_area(self) -> float:
        """Cross-sectional area of main channel [m²]."""
        return self.channel_width * self.channel_depth

    @property
    def throat_cross_section_area(self) -> float:
        """Cross-sectional area at throat [m²]."""
        return self.throat_width * self.channel_depth

    @property
    def velocity_ratio_throat(self) -> float:
        """Velocity ratio: throat velocity / bulk velocity."""
        return self.channel_width / self.throat_width

    @property
    def feed_height(self) -> float:
        """Height of feed entry point [m]."""
        return (self.feed_stage - 0.5) * self.stage_height

    @property
    def separation_zone_height(self) -> float:
        """Approximate height of recirculation zone behind each plate [m]."""
        return self.recirculation_length_ratio * self.plate_length * np.cos(self.plate_angle)


class ZigzagClassifier:
    """
    Zigzag air classifier with proper deflector plate geometry.

    Real-World Design Features:
    ---------------------------
    1. Deflector plates protrude from alternating walls
    2. Plates create throat (accelerating flow) and recirculation zones
    3. Separation occurs in low-velocity zones behind plates
    4. Multiple stages increase separation sharpness

    Flow Zones:
    -----------
    - THROAT: Constricted area between plate tip and opposite wall
              High velocity, particles accelerated upward
    - RECIRCULATION: Eddy zone directly behind each plate
                     Low velocity, high turbulence, particles can settle
    - SEPARATION: Extended zone where classification occurs
                  Particles with v_t > v_local fall, others rise
    - TRANSPORT: Straight sections between deflectors

    Coordinate System:
    ------------------
    - Origin at bottom center of channel (before inlet box)
    - Y-axis pointing upward (flow direction)
    - X-axis across channel width
    - Z-axis into channel depth
    """

    def __init__(self, params: ZigzagClassifierParams):
        """
        Initialize zigzag classifier with deflector plates.

        Args:
            params: ZigzagClassifierParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Store computed geometry
        self.deflector_plates: List[DeflectorPlate] = []
        self.separation_zones: List[SeparationZone] = []

        # Pre-calculate deflector plate positions and separation zones
        self._calculate_deflector_geometry()

    def _calculate_deflector_geometry(self):
        """
        Calculate the geometry of each deflector plate and its separation zone.

        Plate arrangement:
        - Plates alternate between left and right walls
        - First plate on left wall at stage 1
        - Each plate creates a recirculation zone on its downstream side
        """
        p = self.params
        self.deflector_plates = []
        self.separation_zones = []

        x_left_wall = p.center[0] - p.channel_width / 2
        x_right_wall = p.center[0] + p.channel_width / 2
        y_base = p.center[1]

        for stage in range(1, p.num_stages + 1):
            # Plate at middle of each stage
            y_plate_base = y_base + (stage - 0.5) * p.stage_height

            # Alternate sides: odd stages from left, even from right
            if stage % 2 == 1:  # Left wall
                side = 'left'
                x_base = x_left_wall
                # Plate extends into channel (positive x direction)
                x_tip = x_base + p.plate_length * np.sin(p.plate_angle)
                y_tip = y_plate_base + p.plate_length * np.cos(p.plate_angle)
            else:  # Right wall
                side = 'right'
                x_base = x_right_wall
                # Plate extends into channel (negative x direction)
                x_tip = x_base - p.plate_length * np.sin(p.plate_angle)
                y_tip = y_plate_base + p.plate_length * np.cos(p.plate_angle)

            plate = DeflectorPlate(
                stage=stage,
                side=side,
                base_position=(x_base, y_plate_base),
                tip_position=(x_tip, y_tip),
                angle=p.plate_angle,
                length=p.plate_length,
                thickness=p.plate_thickness
            )
            self.deflector_plates.append(plate)

            # Create separation zone behind (above) the plate
            # Zone is approximately elliptical, centered behind plate
            zone_center_x = (x_base + x_tip) / 2
            zone_center_y = y_tip + p.separation_zone_height / 2

            zone = SeparationZone(
                stage=stage,
                plate=plate,
                center=(zone_center_x, zone_center_y),
                width=p.plate_length * np.sin(p.plate_angle) * 0.8,
                height=p.separation_zone_height,
                velocity_ratio=p.velocity_ratio_in_zone,
                turbulence_intensity=p.turbulence_intensity,
                residence_time_factor=2.5
            )
            self.separation_zones.append(zone)

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the zigzag classifier.

        Mesh includes:
        - Left and right channel walls
        - Front and back channel walls
        - Deflector plates (both sides of each plate)
        - Air inlet, fines outlet, coarse outlet boxes
        - Feed inlet tube

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        z_front = p.center[2] - p.channel_depth / 2
        z_back = p.center[2] + p.channel_depth / 2

        x_left = p.center[0] - p.channel_width / 2
        x_right = p.center[0] + p.channel_width / 2
        y_base = p.center[1]
        y_top = y_base + p.total_height

        # Generate main channel walls (straight rectangular sections)
        self._add_channel_walls(vertices, indices, normals,
                               x_left, x_right, y_base, y_top, z_front, z_back)

        # Generate deflector plates
        for plate in self.deflector_plates:
            self._add_deflector_plate(vertices, indices, normals, plate, z_front, z_back)

        # Add air inlet box at bottom
        self._add_inlet_box(vertices, indices, normals, 'air_inlet')

        # Add fines outlet box at top
        self._add_inlet_box(vertices, indices, normals, 'fines_outlet')

        # Add coarse outlet box at bottom
        self._add_inlet_box(vertices, indices, normals, 'coarse_outlet')

        # Add feed inlet at feed stage (only if enabled for side-fed configurations)
        # For venturi-fed systems, particles enter via air_inlet from below
        if self.params.include_feed_inlet:
            self._add_feed_inlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _add_channel_walls(self, vertices: List, indices: List, normals: List,
                           x_left: float, x_right: float,
                           y_base: float, y_top: float,
                           z_front: float, z_back: float):
        """Add the main channel walls (left, right, front, back)."""

        # Left wall - simple rectangle from bottom to top
        start_idx = len(vertices)
        vertices.extend([
            [x_left, y_base, z_front],
            [x_left, y_base, z_back],
            [x_left, y_top, z_front],
            [x_left, y_top, z_back],
        ])
        normals.extend([[1.0, 0.0, 0.0]] * 4)  # Normal into channel
        indices.extend([start_idx, start_idx + 2, start_idx + 1])
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 3])

        # Right wall
        start_idx = len(vertices)
        vertices.extend([
            [x_right, y_base, z_front],
            [x_right, y_base, z_back],
            [x_right, y_top, z_front],
            [x_right, y_top, z_back],
        ])
        normals.extend([[-1.0, 0.0, 0.0]] * 4)  # Normal into channel
        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 1, start_idx + 3, start_idx + 2])

        # Front wall (z = z_front)
        start_idx = len(vertices)
        vertices.extend([
            [x_left, y_base, z_front],
            [x_right, y_base, z_front],
            [x_left, y_top, z_front],
            [x_right, y_top, z_front],
        ])
        normals.extend([[0.0, 0.0, -1.0]] * 4)
        indices.extend([start_idx, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 1, start_idx + 3, start_idx + 2])

        # Back wall (z = z_back)
        start_idx = len(vertices)
        vertices.extend([
            [x_left, y_base, z_back],
            [x_right, y_base, z_back],
            [x_left, y_top, z_back],
            [x_right, y_top, z_back],
        ])
        normals.extend([[0.0, 0.0, 1.0]] * 4)
        indices.extend([start_idx, start_idx + 2, start_idx + 1])
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 3])

    def _add_deflector_plate(self, vertices: List, indices: List, normals: List,
                             plate: DeflectorPlate, z_front: float, z_back: float):
        """
        Add a deflector plate as a 3D solid with proper geometry.

        Each plate is a rectangular prism tilted at the plate angle:
        - Base attached to wall
        - Extends into channel at plate_angle from vertical
        - Has thickness for realistic collision detection
        """
        p = self.params

        # Plate geometry
        x_base, y_base = plate.base_position
        x_tip, y_tip = plate.tip_position
        t = plate.thickness / 2  # Half thickness

        # Direction vectors
        # Along plate (from base to tip)
        dx = x_tip - x_base
        dy = y_tip - y_base
        length = np.sqrt(dx**2 + dy**2)
        along_x, along_y = dx / length, dy / length

        # Perpendicular to plate (for thickness)
        perp_x, perp_y = -along_y, along_x

        # Create plate vertices (8 corners of the rectangular prism)
        # Bottom face (at base)
        v0 = [x_base - perp_x * t, y_base - perp_y * t, z_front]  # base, front, -thick
        v1 = [x_base + perp_x * t, y_base + perp_y * t, z_front]  # base, front, +thick
        v2 = [x_base - perp_x * t, y_base - perp_y * t, z_back]   # base, back, -thick
        v3 = [x_base + perp_x * t, y_base + perp_y * t, z_back]   # base, back, +thick

        # Top face (at tip)
        v4 = [x_tip - perp_x * t, y_tip - perp_y * t, z_front]    # tip, front, -thick
        v5 = [x_tip + perp_x * t, y_tip + perp_y * t, z_front]    # tip, front, +thick
        v6 = [x_tip - perp_x * t, y_tip - perp_y * t, z_back]     # tip, back, -thick
        v7 = [x_tip + perp_x * t, y_tip + perp_y * t, z_back]     # tip, back, +thick

        start_idx = len(vertices)
        vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])

        # Normals for each vertex (will be overwritten per face in proper rendering)
        # For now, use approximate face normals
        normals.extend([[0.0, 0.0, 0.0]] * 8)

        # Plate faces
        # Front face (z = z_front): v0, v1, v4, v5
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 4])

        # Back face (z = z_back): v2, v3, v6, v7
        indices.extend([start_idx + 2, start_idx + 6, start_idx + 3])
        indices.extend([start_idx + 3, start_idx + 6, start_idx + 7])

        # Top face (at tip): v4, v5, v6, v7
        indices.extend([start_idx + 4, start_idx + 5, start_idx + 7])
        indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])

        # Bottom face (at base): v0, v1, v2, v3 - usually hidden by wall
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 1])
        indices.extend([start_idx + 1, start_idx + 2, start_idx + 3])

        # Side faces (the "thick" sides)
        # Side 1 (-thickness side): v0, v2, v4, v6
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 2])
        indices.extend([start_idx + 2, start_idx + 4, start_idx + 6])

        # Side 2 (+thickness side): v1, v3, v5, v7
        indices.extend([start_idx + 1, start_idx + 3, start_idx + 5])
        indices.extend([start_idx + 3, start_idx + 7, start_idx + 5])

    def _add_inlet_box(self, vertices: List, indices: List, normals: List, inlet_type: str):
        """Add inlet/outlet box geometry."""
        p = self.params

        x_center = p.center[0]
        z_center = p.center[2]

        if inlet_type == 'air_inlet':
            width = p.air_inlet_width
            height = p.air_inlet_height
            y_base = p.center[1] - height
        elif inlet_type == 'fines_outlet':
            width = p.fines_outlet_width
            height = p.fines_outlet_height
            y_base = p.center[1] + p.total_height
        else:  # coarse_outlet
            width = p.coarse_outlet_width
            height = p.coarse_outlet_height
            y_base = p.center[1] - height - p.air_inlet_height

        depth = p.channel_depth
        start_idx = len(vertices)
        hw, hd = width / 2, depth / 2

        # 8 corners of box
        box_verts = [
            [x_center - hw, y_base, z_center - hd],
            [x_center + hw, y_base, z_center - hd],
            [x_center + hw, y_base + height, z_center - hd],
            [x_center - hw, y_base + height, z_center - hd],
            [x_center - hw, y_base, z_center + hd],
            [x_center + hw, y_base, z_center + hd],
            [x_center + hw, y_base + height, z_center + hd],
            [x_center - hw, y_base + height, z_center + hd],
        ]

        for v in box_verts:
            vertices.append(v)
            normals.append([0.0, 0.0, 0.0])

        # Box faces
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 3])
        indices.extend([start_idx + 5, start_idx + 4, start_idx + 7])
        indices.extend([start_idx + 5, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 0, start_idx + 3])
        indices.extend([start_idx + 4, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])

    def _add_feed_inlet(self, vertices: List, indices: List, normals: List):
        """Add feed inlet at the specified stage with proper angle."""
        p = self.params

        # Find the plate at feed stage to position feed inlet appropriately
        feed_plate = self.deflector_plates[p.feed_stage - 1]

        # Feed inlet positioned on opposite wall from the plate at this stage
        if feed_plate.side == 'left':
            # Plate is on left, so feed from right
            x_wall = p.center[0] + p.channel_width / 2
            feed_dir_x = 1.0  # Points outward from wall
        else:
            # Plate is on right, so feed from left
            x_wall = p.center[0] - p.channel_width / 2
            feed_dir_x = -1.0

        y_center = p.center[1] + (p.feed_stage - 0.5) * p.stage_height
        z_center = p.center[2]

        feed_length = p.channel_width * 0.3
        hw = p.feed_width / 2
        hd = p.channel_depth / 2

        start_idx = len(vertices)

        # Account for feed angle (rotation around Z axis)
        cos_a = np.cos(p.feed_angle)
        sin_a = np.sin(p.feed_angle)

        # Simple rectangular feed tube
        x_outer = x_wall + feed_dir_x * feed_length

        feed_verts = [
            [x_wall, y_center - hw, z_center - hd],
            [x_wall, y_center + hw, z_center - hd],
            [x_wall, y_center + hw, z_center + hd],
            [x_wall, y_center - hw, z_center + hd],
            [x_outer, y_center - hw, z_center - hd],
            [x_outer, y_center + hw, z_center - hd],
            [x_outer, y_center + hw, z_center + hd],
            [x_outer, y_center - hw, z_center + hd],
        ]

        for v in feed_verts:
            vertices.append(v)
            normals.append([feed_dir_x, 0.0, 0.0])

        # Feed tube faces
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 4])
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 1])
        indices.extend([start_idx + 3, start_idx + 2, start_idx + 6])
        indices.extend([start_idx + 3, start_idx + 6, start_idx + 7])
        indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 5])

    # =========================================================================
    # Physics and Separation Calculations
    # =========================================================================

    def get_bulk_velocity(self, volumetric_flow: float) -> float:
        """
        Calculate bulk air velocity in main channel.

        Args:
            volumetric_flow: Volumetric flow rate [m³/s]

        Returns:
            Bulk velocity in main channel [m/s]
        """
        return volumetric_flow / self.params.channel_cross_section_area

    def get_throat_velocity(self, volumetric_flow: float) -> float:
        """
        Calculate air velocity at throat (constriction).

        The throat is the narrowest point between plate tip and opposite wall.
        Velocity is higher here due to continuity.

        Args:
            volumetric_flow: Volumetric flow rate [m³/s]

        Returns:
            Velocity at throat [m/s]
        """
        return volumetric_flow / self.params.throat_cross_section_area

    def get_separation_zone_velocity(self, volumetric_flow: float) -> float:
        """
        Calculate effective velocity in separation (recirculation) zones.

        This is significantly lower than bulk velocity due to recirculation.
        This is where the actual particle separation occurs.

        Args:
            volumetric_flow: Volumetric flow rate [m³/s]

        Returns:
            Effective velocity in separation zones [m/s]
        """
        v_bulk = self.get_bulk_velocity(volumetric_flow)
        return v_bulk * self.params.velocity_ratio_in_zone

    def calculate_terminal_velocity(
        self,
        particle_diameter: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate terminal (settling) velocity for a particle.

        Uses Schiller-Naumann correlation for intermediate Reynolds numbers.

        Args:
            particle_diameter: Particle diameter [m]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Terminal velocity [m/s]
        """
        g = 9.81
        d_p = particle_diameter
        rho_p = particle_density
        rho_f = air_density
        mu = air_viscosity

        # First estimate using Stokes law
        v_t_stokes = (d_p ** 2) * (rho_p - rho_f) * g / (18 * mu)

        # Check Reynolds number
        Re_p = rho_f * v_t_stokes * d_p / mu

        if Re_p < 0.1:
            # Stokes regime
            return v_t_stokes
        elif Re_p < 1000:
            # Intermediate regime - iterate with Schiller-Naumann
            v_t = v_t_stokes
            for _ in range(20):  # Iterate to convergence
                Re_p = rho_f * v_t * d_p / mu
                C_d = (24 / Re_p) * (1 + 0.15 * Re_p ** 0.687)
                v_t_new = np.sqrt(4 * d_p * g * (rho_p - rho_f) / (3 * C_d * rho_f))
                if abs(v_t_new - v_t) / v_t < 1e-6:
                    break
                v_t = v_t_new
            return v_t
        else:
            # Newton regime (C_d ≈ 0.44)
            C_d = 0.44
            return np.sqrt(4 * d_p * g * (rho_p - rho_f) / (3 * C_d * rho_f))

    def calculate_cut_size_d50(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate the cut size (d50) for given flow rate.

        d50 is the particle size where 50% goes to fines and 50% to coarse.
        This occurs when terminal velocity equals the effective velocity
        in the separation zones.

        Args:
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            d50 cut size [m]
        """
        v_sep = self.get_separation_zone_velocity(volumetric_flow)
        g = 9.81

        # For Stokes regime: v_t = d² × (ρ_p - ρ_f) × g / (18 × μ)
        # Rearrange: d50 = √(18 × μ × v_sep / (g × (ρ_p - ρ_f)))
        d50_stokes = np.sqrt(
            18 * air_viscosity * v_sep / (g * (particle_density - air_density))
        )

        # Check if we're in Stokes regime
        Re_p = air_density * v_sep * d50_stokes / air_viscosity

        if Re_p < 0.1:
            return d50_stokes
        else:
            # Need to iterate for intermediate regime
            d50 = d50_stokes
            for _ in range(20):
                v_t = self.calculate_terminal_velocity(d50, particle_density,
                                                        air_density, air_viscosity)
                # Adjust d50 based on ratio
                ratio = v_sep / v_t
                d50 = d50 * np.sqrt(ratio)
                if abs(ratio - 1.0) < 1e-6:
                    break
            return d50

    def calculate_stokes_number(
        self,
        particle_diameter: float,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate Stokes number for particle at throat.

        St = τ_p × U / L

        Where:
        - τ_p = particle relaxation time = ρ_p × d_p² / (18 × μ)
        - U = characteristic velocity (throat velocity)
        - L = characteristic length (plate length)

        St >> 1: Particle continues straight (inertia dominated)
        St << 1: Particle follows flow (drag dominated)
        St ~ 1: Transition, critical for separation

        Args:
            particle_diameter: Particle diameter [m]
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Stokes number (dimensionless)
        """
        tau_p = particle_density * (particle_diameter ** 2) / (18 * air_viscosity)
        U = self.get_throat_velocity(volumetric_flow)
        L = self.params.plate_length

        return tau_p * U / L

    def calculate_grade_efficiency(
        self,
        particle_diameter: float,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate grade efficiency (probability of going to coarse outlet).

        Uses the Rosin-Rammler-Sperling-Bennett (RRSB) distribution model
        adapted for zigzag classifiers.

        G(d) = 1 - exp(-0.693 × (d/d50)^n)

        Where n is the sharpness parameter (typically 2-5 for zigzag).
        Higher n = sharper cut.

        Args:
            particle_diameter: Particle diameter [m]
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Grade efficiency [0-1] (probability to coarse outlet)
        """
        d50 = self.calculate_cut_size_d50(volumetric_flow, particle_density,
                                          air_density, air_viscosity)

        # Sharpness parameter increases with number of stages
        n = 1.5 + 0.5 * self.params.num_stages

        # RRSB equation
        G = 1.0 - np.exp(-0.693 * (particle_diameter / d50) ** n)

        return np.clip(G, 0.0, 1.0)

    def calculate_required_flow_for_d50(
        self,
        target_d50: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate required flow rate to achieve target cut size.

        Args:
            target_d50: Desired cut size [m]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            Required volumetric flow rate [m³/s]
        """
        g = 9.81

        # Terminal velocity for target d50
        v_t = self.calculate_terminal_velocity(target_d50, particle_density,
                                                air_density, air_viscosity)

        # Separation zone velocity should equal terminal velocity
        v_sep_required = v_t

        # Bulk velocity = v_sep / velocity_ratio
        v_bulk = v_sep_required / self.params.velocity_ratio_in_zone

        # Flow rate
        Q = v_bulk * self.params.channel_cross_section_area
        return Q

    def calculate_pressure_drop(
        self,
        volumetric_flow: float,
        air_density: float = 1.204,
    ) -> float:
        """
        Estimate pressure drop across the classifier.

        Uses empirical correlation for zigzag channels with deflector plates.
        ΔP = K × 0.5 × ρ × v²

        Where K depends on number of stages and blockage ratio.

        Args:
            volumetric_flow: Volumetric flow rate [m³/s]
            air_density: Air density [kg/m³]

        Returns:
            Pressure drop [Pa]
        """
        v_bulk = self.get_bulk_velocity(volumetric_flow)

        # Loss coefficient per stage (empirical)
        # Accounts for: entry loss, deflector drag, expansion loss
        K_stage = 1.5 + 2.0 * self.params.blockage_ratio ** 2

        # Total loss coefficient
        K_total = K_stage * self.params.num_stages + 0.5  # +0.5 for entry/exit

        # Pressure drop
        delta_P = K_total * 0.5 * air_density * v_bulk ** 2

        return delta_P

    def validate_operating_conditions(
        self,
        volumetric_flow: float,
        min_particle_size: float = 5e-6,
        max_particle_size: float = 100e-6,
        particle_density: float = 1420.0,
    ) -> dict:
        """
        Validate if flow rate is appropriate for particle separation.

        Args:
            volumetric_flow: Flow rate [m³/s]
            min_particle_size: Smallest particle to separate [m]
            max_particle_size: Largest particle to separate [m]
            particle_density: Particle density [kg/m³]

        Returns:
            Dictionary with validation results and recommendations
        """
        d50 = self.calculate_cut_size_d50(volumetric_flow, particle_density)
        v_bulk = self.get_bulk_velocity(volumetric_flow)
        v_throat = self.get_throat_velocity(volumetric_flow)
        v_sep = self.get_separation_zone_velocity(volumetric_flow)
        delta_P = self.calculate_pressure_drop(volumetric_flow)

        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "d50_um": d50 * 1e6,
            "bulk_velocity_m_s": v_bulk,
            "throat_velocity_m_s": v_throat,
            "separation_zone_velocity_m_s": v_sep,
            "pressure_drop_Pa": delta_P,
            "volumetric_flow_m3_h": volumetric_flow * 3600,
        }

        # Check d50 range
        if d50 > max_particle_size:
            result["valid"] = False
            result["errors"].append(
                f"Cut size ({d50*1e6:.1f} µm) > max particle ({max_particle_size*1e6:.1f} µm). "
                "ALL particles will go to fines. Reduce flow rate."
            )

        if d50 < min_particle_size:
            result["valid"] = False
            result["errors"].append(
                f"Cut size ({d50*1e6:.1f} µm) < min particle ({min_particle_size*1e6:.1f} µm). "
                "ALL particles will go to coarse. Increase flow rate."
            )

        # Check velocity limits
        if v_throat > 25.0:
            result["errors"].append(
                f"Throat velocity ({v_throat:.1f} m/s) exceeds 25 m/s. "
                "Risk of particle attrition and excessive pressure drop."
            )
            result["valid"] = False
        elif v_throat > 15.0:
            result["warnings"].append(
                f"Throat velocity ({v_throat:.1f} m/s) is high. "
                "May cause particle attrition for fragile materials."
            )

        if v_bulk > 8.0:
            result["warnings"].append(
                f"Bulk velocity ({v_bulk:.1f} m/s) is high. "
                "Separation efficiency may decrease."
            )

        # Check pressure drop
        if delta_P > 2000:
            result["warnings"].append(
                f"Pressure drop ({delta_P:.0f} Pa) is high. "
                "Consider larger channel or fewer stages."
            )

        # Stokes number check for typical particles
        St_min = self.calculate_stokes_number(min_particle_size, volumetric_flow, particle_density)
        St_max = self.calculate_stokes_number(max_particle_size, volumetric_flow, particle_density)
        result["stokes_number_range"] = (St_min, St_max)

        if St_max < 0.1:
            result["warnings"].append(
                f"Max Stokes number ({St_max:.3f}) is very low. "
                "All particles follow the flow - poor inertial separation."
            )

        # Calculate recommended flow range
        Q_for_max = self.calculate_required_flow_for_d50(max_particle_size * 0.8, particle_density)
        Q_for_min = self.calculate_required_flow_for_d50(min_particle_size * 1.2, particle_density)
        result["recommended_flow_range_m3_s"] = (Q_for_min, Q_for_max)
        result["recommended_flow_range_m3_h"] = (Q_for_min * 3600, Q_for_max * 3600)

        # Grade efficiency at size extremes
        G_min = self.calculate_grade_efficiency(min_particle_size, volumetric_flow, particle_density)
        G_max = self.calculate_grade_efficiency(max_particle_size, volumetric_flow, particle_density)
        result["grade_efficiency_at_min_size"] = G_min
        result["grade_efficiency_at_max_size"] = G_max

        return result

    # =========================================================================
    # Zone Classification for Physics Simulation
    # =========================================================================

    def get_zone_at_position(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[ZoneType, Optional[int], float]:
        """
        Determine which flow zone a position is in.

        This is used by the physics simulation to apply zone-specific
        velocity fields and turbulence models.

        Args:
            x, y, z: Position coordinates [m]

        Returns:
            Tuple of (ZoneType, stage_number or None, velocity_factor)
            velocity_factor: Multiplier for bulk velocity (0-1 for separation zones)
        """
        p = self.params

        # Check bounds
        x_left = p.center[0] - p.channel_width / 2
        x_right = p.center[0] + p.channel_width / 2
        y_base = p.center[1]
        y_top = y_base + p.total_height

        # Outside main channel
        if y < y_base:
            return (ZoneType.INLET, None, 1.0)
        if y > y_top:
            return (ZoneType.OUTLET, None, 1.0)

        # Determine stage
        stage = int((y - y_base) / p.stage_height) + 1
        stage = min(stage, p.num_stages)

        # Check if in separation zone
        for zone in self.separation_zones:
            if zone.stage == stage:
                # Check if inside elliptical separation zone
                cx, cy = zone.center
                dx = (x - cx) / (zone.width / 2)
                dy = (y - cy) / (zone.height / 2)
                if dx**2 + dy**2 <= 1.0:
                    return (ZoneType.SEPARATION, stage, zone.velocity_ratio)

        # Check if in throat region
        plate = self.deflector_plates[stage - 1] if stage <= len(self.deflector_plates) else None
        if plate:
            tip_x, tip_y = plate.tip_position
            throat_y_min = tip_y - p.stage_height * 0.2
            throat_y_max = tip_y + p.stage_height * 0.2

            if throat_y_min <= y <= throat_y_max:
                return (ZoneType.THROAT, stage, p.velocity_ratio_throat)

        # Default: transport zone
        return (ZoneType.TRANSPORT, stage, 1.0)

    def get_velocity_field(
        self,
        x: float,
        y: float,
        z: float,
        volumetric_flow: float
    ) -> Tuple[float, float, float]:
        """
        Get the local velocity vector at a position.

        This provides a simplified velocity field for particle tracking.
        For accurate results, use CFD-derived fields.

        Args:
            x, y, z: Position coordinates [m]
            volumetric_flow: Volumetric flow rate [m³/s]

        Returns:
            (v_x, v_y, v_z) velocity components [m/s]
        """
        zone_type, stage, v_factor = self.get_zone_at_position(x, y, z)

        v_bulk = self.get_bulk_velocity(volumetric_flow)
        v_local = v_bulk * v_factor

        # Primary flow is upward (positive Y)
        v_y = v_local

        # Add lateral component in throat regions (flow deflection)
        if zone_type == ZoneType.THROAT and stage:
            plate = self.deflector_plates[stage - 1]
            if plate.side == 'left':
                # Flow deflects right at left-side plate
                v_x = v_local * np.sin(self.params.plate_angle) * 0.3
            else:
                # Flow deflects left at right-side plate
                v_x = -v_local * np.sin(self.params.plate_angle) * 0.3
        else:
            v_x = 0.0

        v_z = 0.0  # No depth-wise flow in 2D approximation

        return (v_x, v_y, v_z)

    def get_turbulence_intensity(self, x: float, y: float, z: float) -> float:
        """
        Get turbulence intensity at a position.

        Returns u'/U where u' is RMS fluctuation velocity.

        Args:
            x, y, z: Position coordinates [m]

        Returns:
            Turbulence intensity (dimensionless, typically 0.05-0.30)
        """
        zone_type, stage, _ = self.get_zone_at_position(x, y, z)

        if zone_type == ZoneType.SEPARATION:
            return self.params.turbulence_intensity
        elif zone_type == ZoneType.THROAT:
            return 0.15  # Moderate turbulence in accelerating flow
        elif zone_type == ZoneType.RECIRCULATION:
            return 0.30  # High turbulence in recirculating flow
        else:
            return 0.08  # Low turbulence in transport zones

    # =========================================================================
    # Geometry Access Methods
    # =========================================================================

    def get_stage_center(self, stage: int) -> Tuple[float, float, float]:
        """Get the center position of a stage."""
        if stage < 1 or stage > self.params.num_stages:
            raise ValueError(f"Stage must be between 1 and {self.params.num_stages}")

        p = self.params
        x = p.center[0]
        y = p.center[1] + (stage - 0.5) * p.stage_height
        z = p.center[2]

        return (x, y, z)

    def get_plate_at_stage(self, stage: int) -> Optional[DeflectorPlate]:
        """Get the deflector plate at a given stage."""
        if stage < 1 or stage > len(self.deflector_plates):
            return None
        return self.deflector_plates[stage - 1]

    def get_separation_zone_at_stage(self, stage: int) -> Optional[SeparationZone]:
        """Get the separation zone at a given stage."""
        if stage < 1 or stage > len(self.separation_zones):
            return None
        return self.separation_zones[stage - 1]

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the classifier geometry."""
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
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for the zigzag classifier.

        Ports:
        - air_inlet: Main air inlet at bottom (rectangular, -Y direction)
        - feed_inlet: Particle feed inlet at feed stage
        - fines_outlet: Top outlet for light particles (+Y direction)
        - coarse_outlet: Bottom outlet for heavy particles (-Y direction)
        """
        p = self.params

        # Air inlet at bottom center (where venturi outlet connects)
        # In venturi-fed systems, this is where particles + air enter
        air_inlet = ConnectionPort(
            position=(p.center[0], p.center[1] - p.air_inlet_height, p.center[2]),
            direction=(0.0, -1.0, 0.0),
            width=p.air_inlet_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="air_inlet"
        )

        # Fines outlet at top (light particles exit here)
        fines_outlet = ConnectionPort(
            position=(p.center[0], p.center[1] + p.total_height + p.fines_outlet_height, p.center[2]),
            direction=(0.0, 1.0, 0.0),
            width=p.fines_outlet_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="fines_outlet"
        )

        # Coarse outlet at bottom (heavy particles fall here)
        coarse_outlet = ConnectionPort(
            position=(p.center[0], p.center[1] - p.air_inlet_height - p.coarse_outlet_height, p.center[2]),
            direction=(0.0, -1.0, 0.0),
            width=p.coarse_outlet_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="coarse_outlet"
        )

        ports = {
            'air_inlet': air_inlet,
            'fines_outlet': fines_outlet,
            'coarse_outlet': coarse_outlet
        }

        # Feed inlet only for side-fed configurations (not venturi-fed)
        if p.include_feed_inlet and p.feed_stage >= 1 and p.feed_stage <= len(self.deflector_plates):
            feed_plate = self.deflector_plates[p.feed_stage - 1]
            if feed_plate.side == 'left':
                x_feed = p.center[0] + p.channel_width / 2 + p.channel_width * 0.3
                feed_dir = (1.0, 0.0, 0.0)
            else:
                x_feed = p.center[0] - p.channel_width / 2 - p.channel_width * 0.3
                feed_dir = (-1.0, 0.0, 0.0)

            y_feed = p.center[1] + (p.feed_stage - 0.5) * p.stage_height

            feed_inlet = ConnectionPort(
                position=(x_feed, y_feed, p.center[2]),
                direction=feed_dir,
                width=p.feed_width,
                height=p.channel_depth,
                port_type=PortType.RECTANGULAR,
                name="feed_inlet"
            )
            ports['feed_inlet'] = feed_inlet

        return ports


def create_standard_zigzag_classifier(
    channel_width: float = 0.15,
    num_stages: int = 5,
    channel_depth: float = 0.30,
    plate_angle_deg: float = 45.0,
    plate_length_ratio: float = 0.5,
) -> ZigzagClassifier:
    """
    Create a standard zigzag classifier with typical industrial proportions.

    Default design based on Senden (1979) and industrial practice:
    - 45° plate angle (good balance of separation and pressure drop)
    - 50% blockage ratio (plate extends halfway across channel)
    - Stage height = 1.5 × channel width

    Args:
        channel_width: Width of zigzag channel [m] (default 0.15m = 150mm)
        num_stages: Number of stages (default 5, range 3-7 typical)
        channel_depth: Depth of channel [m] (default 0.30m = 300mm)
        plate_angle_deg: Plate angle from vertical [degrees] (default 45°)
        plate_length_ratio: Plate length / channel width (default 0.5)

    Returns:
        ZigzagClassifier instance with proper deflector plate geometry
    """
    params = ZigzagClassifierParams(
        channel_width=channel_width,
        channel_depth=channel_depth,
        num_stages=num_stages,
        stage_height=channel_width * 1.5,
        plate_angle=np.radians(plate_angle_deg),
        plate_length_ratio=plate_length_ratio,
        plate_thickness=0.003,
        feed_stage=(num_stages + 1) // 2,
        feed_width=channel_width * 0.5,
        feed_angle=0.0,
        air_inlet_width=channel_width,
        air_inlet_height=channel_width * 0.5,
        fines_outlet_width=channel_width,
        fines_outlet_height=channel_width * 0.5,
        coarse_outlet_width=channel_width * 0.5,
        coarse_outlet_height=channel_width * 0.3,
        wall_thickness=0.003,
    )

    return ZigzagClassifier(params)


def create_high_efficiency_zigzag_classifier(
    channel_width: float = 0.15,
    num_stages: int = 7,
    channel_depth: float = 0.30,
) -> ZigzagClassifier:
    """
    Create a high-efficiency zigzag classifier with more stages and higher blockage.

    Design features:
    - 7 stages for sharper separation
    - 60° plate angle (more aggressive flow deflection)
    - 55% blockage ratio (larger separation zones)

    Args:
        channel_width: Width of zigzag channel [m]
        num_stages: Number of stages (default 7)
        channel_depth: Depth of channel [m]

    Returns:
        ZigzagClassifier instance optimized for separation efficiency
    """
    params = ZigzagClassifierParams(
        channel_width=channel_width,
        channel_depth=channel_depth,
        num_stages=num_stages,
        stage_height=channel_width * 1.3,  # Closer spacing
        plate_angle=np.radians(60.0),       # Steeper plates
        plate_length_ratio=0.55,            # Higher blockage
        plate_thickness=0.003,
        feed_stage=(num_stages + 1) // 2,
        feed_width=channel_width * 0.5,
        feed_angle=0.0,
        air_inlet_width=channel_width,
        air_inlet_height=channel_width * 0.5,
        fines_outlet_width=channel_width,
        fines_outlet_height=channel_width * 0.5,
        coarse_outlet_width=channel_width * 0.5,
        coarse_outlet_height=channel_width * 0.3,
        wall_thickness=0.003,
        velocity_ratio_in_zone=0.25,        # Lower velocity in zones
        turbulence_intensity=0.30,          # Higher turbulence
    )

    return ZigzagClassifier(params)


def create_low_pressure_zigzag_classifier(
    channel_width: float = 0.20,
    num_stages: int = 4,
    channel_depth: float = 0.40,
) -> ZigzagClassifier:
    """
    Create a low-pressure-drop zigzag classifier.

    Design features:
    - Larger channel dimensions
    - Fewer stages (4)
    - 30° plate angle (gentler deflection)
    - 40% blockage ratio (more open flow path)

    Trade-off: Lower separation efficiency for reduced pressure drop.

    Args:
        channel_width: Width of zigzag channel [m]
        num_stages: Number of stages (default 4)
        channel_depth: Depth of channel [m]

    Returns:
        ZigzagClassifier instance optimized for low pressure drop
    """
    params = ZigzagClassifierParams(
        channel_width=channel_width,
        channel_depth=channel_depth,
        num_stages=num_stages,
        stage_height=channel_width * 1.8,  # Taller stages
        plate_angle=np.radians(30.0),      # Gentler plates
        plate_length_ratio=0.40,           # Lower blockage
        plate_thickness=0.003,
        feed_stage=(num_stages + 1) // 2,
        feed_width=channel_width * 0.5,
        feed_angle=0.0,
        air_inlet_width=channel_width,
        air_inlet_height=channel_width * 0.5,
        fines_outlet_width=channel_width,
        fines_outlet_height=channel_width * 0.5,
        coarse_outlet_width=channel_width * 0.5,
        coarse_outlet_height=channel_width * 0.3,
        wall_thickness=0.003,
        velocity_ratio_in_zone=0.35,       # Higher velocity in zones
        turbulence_intensity=0.20,         # Lower turbulence
    )

    return ZigzagClassifier(params)
