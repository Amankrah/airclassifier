"""
Zigzag air classifier component for particle separation.

The zigzag classifier is the primary separation device in legume protein
fractionation. It uses a series of zigzag channels to create turbulent
zones where particles are separated based on their terminal velocity.

Principle:
- Air flows upward through the zigzag channel
- Heavy particles (starch) fall against the airflow → coarse outlet
- Light particles (protein) are carried up → fines outlet
- The zigzag geometry creates turbulent mixing at each stage
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
import warp as wp

from ...utils.constants import PI, TWO_PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class ZigzagClassifierParams:
    """Parameters for zigzag air classifier."""

    # Channel geometry
    channel_width: float         # [m] Width of zigzag channel
    channel_depth: float         # [m] Depth of channel (into page)
    num_stages: int              # Number of zigzag stages (3-7 typical)
    stage_height: float          # [m] Height per stage
    zigzag_angle: float          # [rad] Interior angle of zigzag (typically 120° = 2.094 rad)

    # Feed entry
    feed_stage: int              # Stage number for feed entry (1-indexed, middle typical)
    feed_width: float            # [m] Width of feed inlet

    # Inlet/outlet dimensions
    air_inlet_width: float       # [m] Bottom air inlet width
    air_inlet_height: float      # [m] Bottom air inlet height
    fines_outlet_width: float    # [m] Top fines outlet width
    fines_outlet_height: float   # [m] Top fines outlet height
    coarse_outlet_width: float   # [m] Bottom coarse outlet width
    coarse_outlet_height: float  # [m] Bottom coarse outlet height

    # Construction
    wall_thickness: float = 0.003  # [m] Wall thickness (3mm default)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bottom center

    # Mesh resolution
    resolution_length: int = 10   # Points along channel length
    resolution_depth: int = 4     # Points along depth

    def __post_init__(self):
        if self.feed_stage < 1 or self.feed_stage > self.num_stages:
            raise ValueError(f"Feed stage must be between 1 and {self.num_stages}")
        if self.zigzag_angle <= PI / 2 or self.zigzag_angle >= PI:
            raise ValueError("Zigzag angle should be between 90° and 180°")

    @property
    def total_height(self) -> float:
        """Total height of classifier."""
        return self.num_stages * self.stage_height

    @property
    def horizontal_offset(self) -> float:
        """Horizontal offset per stage due to zigzag."""
        # For zigzag angle θ, horizontal offset = stage_height / tan(θ/2)
        return self.stage_height / np.tan(self.zigzag_angle / 2)

    @property
    def total_width(self) -> float:
        """Total width including zigzag offsets."""
        return self.channel_width + (self.num_stages - 1) * self.horizontal_offset

    @property
    def channel_cross_section_area(self) -> float:
        """Cross-sectional area of channel."""
        return self.channel_width * self.channel_depth

    @property
    def feed_height(self) -> float:
        """Height of feed entry point."""
        return (self.feed_stage - 0.5) * self.stage_height


class ZigzagClassifier:
    """
    Zigzag air classifier for particle separation.

    The classifier consists of:
    - Multiple zigzag stages creating turbulent separation zones
    - Bottom air inlet for upward airflow
    - Feed inlet at middle stage
    - Top fines outlet (protein-rich)
    - Bottom coarse outlet (starch-rich)

    Coordinate system:
    - Origin at bottom center of air inlet
    - Y-axis pointing upward
    - X-axis along channel width
    - Z-axis along channel depth
    """

    def __init__(self, params: ZigzagClassifierParams):
        """
        Initialize zigzag classifier.

        Args:
            params: ZigzagClassifierParams defining the geometry
        """
        self.params = params
        self._vertices = None
        self._indices = None
        self._normals = None

        # Pre-calculate stage positions
        self._calculate_stage_geometry()

    def _calculate_stage_geometry(self):
        """Calculate the geometry of each zigzag stage."""
        p = self.params
        self.stage_corners = []

        # Start at bottom left of channel
        x_left = p.center[0] - p.channel_width / 2
        x_right = p.center[0] + p.channel_width / 2
        y_base = p.center[1]

        for stage in range(p.num_stages + 1):
            y = y_base + stage * p.stage_height

            # Alternate direction of zigzag
            if stage % 2 == 0:
                # Even stages: channel is at base position
                offset = 0
            else:
                # Odd stages: channel is offset
                offset = p.horizontal_offset

            corners = {
                'left': (x_left + offset, y),
                'right': (x_right + offset, y),
                'stage': stage
            }
            self.stage_corners.append(corners)

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the zigzag classifier.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []

        z_front = p.center[2] - p.channel_depth / 2
        z_back = p.center[2] + p.channel_depth / 2

        # Generate vertices for left and right walls
        for wall_side in ['left', 'right']:
            wall_start_idx = len(vertices)

            for i, corners in enumerate(self.stage_corners):
                x = corners[wall_side][0]
                y = corners[wall_side][1]

                # Front and back vertices at this stage level
                vertices.append([x, y, z_front])
                vertices.append([x, y, z_back])

                # Normal pointing inward (into channel)
                if wall_side == 'left':
                    normals.append([1.0, 0.0, 0.0])
                    normals.append([1.0, 0.0, 0.0])
                else:
                    normals.append([-1.0, 0.0, 0.0])
                    normals.append([-1.0, 0.0, 0.0])

            # Generate triangles for this wall
            for i in range(len(self.stage_corners) - 1):
                v0 = wall_start_idx + i * 2
                v1 = wall_start_idx + i * 2 + 1
                v2 = wall_start_idx + (i + 1) * 2
                v3 = wall_start_idx + (i + 1) * 2 + 1

                if wall_side == 'left':
                    # Left wall faces right (+X direction)
                    indices.extend([v0, v2, v1])
                    indices.extend([v1, v2, v3])
                else:
                    # Right wall faces left (-X direction)
                    indices.extend([v0, v1, v2])
                    indices.extend([v1, v3, v2])

        # Generate front and back walls
        for z_pos, normal_z in [(z_front, -1.0), (z_back, 1.0)]:
            wall_start_idx = len(vertices)

            # Create vertices along zigzag path
            for i, corners in enumerate(self.stage_corners):
                x_left = corners['left'][0]
                x_right = corners['right'][0]
                y = corners['left'][1]

                vertices.append([x_left, y, z_pos])
                vertices.append([x_right, y, z_pos])
                normals.append([0.0, 0.0, normal_z])
                normals.append([0.0, 0.0, normal_z])

            # Generate triangles
            for i in range(len(self.stage_corners) - 1):
                bl = wall_start_idx + i * 2       # Bottom left
                br = wall_start_idx + i * 2 + 1   # Bottom right
                tl = wall_start_idx + (i + 1) * 2 # Top left
                tr = wall_start_idx + (i + 1) * 2 + 1  # Top right

                if normal_z < 0:  # Front wall
                    indices.extend([bl, br, tr])
                    indices.extend([bl, tr, tl])
                else:  # Back wall
                    indices.extend([bl, tr, br])
                    indices.extend([bl, tl, tr])

        # Add air inlet box at bottom
        self._add_inlet_box(vertices, indices, normals, 'air_inlet')

        # Add fines outlet box at top
        self._add_inlet_box(vertices, indices, normals, 'fines_outlet')

        # Add coarse outlet box at bottom
        self._add_inlet_box(vertices, indices, normals, 'coarse_outlet')

        # Add feed inlet at feed stage
        self._add_feed_inlet(vertices, indices, normals)

        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)

        return self._vertices, self._indices, self._normals

    def _add_inlet_box(self, vertices: List, indices: List, normals: List, inlet_type: str):
        """Add inlet/outlet box geometry."""
        p = self.params

        if inlet_type == 'air_inlet':
            width = p.air_inlet_width
            height = p.air_inlet_height
            x_center = self.stage_corners[0]['left'][0] + p.channel_width / 2
            y_base = p.center[1] - height
            z_center = p.center[2]
        elif inlet_type == 'fines_outlet':
            width = p.fines_outlet_width
            height = p.fines_outlet_height
            x_center = self.stage_corners[-1]['left'][0] + p.channel_width / 2
            y_base = p.center[1] + p.total_height
            z_center = p.center[2]
        else:  # coarse_outlet
            width = p.coarse_outlet_width
            height = p.coarse_outlet_height
            x_center = self.stage_corners[0]['left'][0] + p.channel_width / 2
            y_base = p.center[1] - height - p.air_inlet_height
            z_center = p.center[2]

        depth = p.channel_depth

        # Box vertices
        start_idx = len(vertices)
        hw, hd = width / 2, depth / 2

        # 8 corners of box
        box_verts = [
            [x_center - hw, y_base, z_center - hd],          # 0: front bottom left
            [x_center + hw, y_base, z_center - hd],          # 1: front bottom right
            [x_center + hw, y_base + height, z_center - hd], # 2: front top right
            [x_center - hw, y_base + height, z_center - hd], # 3: front top left
            [x_center - hw, y_base, z_center + hd],          # 4: back bottom left
            [x_center + hw, y_base, z_center + hd],          # 5: back bottom right
            [x_center + hw, y_base + height, z_center + hd], # 6: back top right
            [x_center - hw, y_base + height, z_center + hd], # 7: back top left
        ]

        for v in box_verts:
            vertices.append(v)
            normals.append([0.0, 0.0, 0.0])  # Will be set per face

        # Box faces (excluding top for inlet, bottom for outlet)
        # Front face
        indices.extend([start_idx + 0, start_idx + 1, start_idx + 2])
        indices.extend([start_idx + 0, start_idx + 2, start_idx + 3])
        # Back face
        indices.extend([start_idx + 5, start_idx + 4, start_idx + 7])
        indices.extend([start_idx + 5, start_idx + 7, start_idx + 6])
        # Left face
        indices.extend([start_idx + 4, start_idx + 0, start_idx + 3])
        indices.extend([start_idx + 4, start_idx + 3, start_idx + 7])
        # Right face
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])

    def _add_feed_inlet(self, vertices: List, indices: List, normals: List):
        """Add feed inlet at the specified stage."""
        p = self.params

        # Feed position at the feed stage
        stage_idx = p.feed_stage - 1
        corners = self.stage_corners[stage_idx]

        # Feed inlet on the right side of the channel
        x_right = corners['right'][0]
        y_center = corners['right'][1] + p.stage_height / 2
        z_center = p.center[2]

        feed_length = p.channel_width * 0.3  # Feed tube length
        hw = p.feed_width / 2
        hd = p.channel_depth / 2

        start_idx = len(vertices)

        # Simple rectangular feed tube pointing into channel
        feed_verts = [
            [x_right, y_center - hw, z_center - hd],
            [x_right, y_center + hw, z_center - hd],
            [x_right, y_center + hw, z_center + hd],
            [x_right, y_center - hw, z_center + hd],
            [x_right + feed_length, y_center - hw, z_center - hd],
            [x_right + feed_length, y_center + hw, z_center - hd],
            [x_right + feed_length, y_center + hw, z_center + hd],
            [x_right + feed_length, y_center - hw, z_center + hd],
        ]

        for v in feed_verts:
            vertices.append(v)
            normals.append([1.0, 0.0, 0.0])

        # Feed tube faces
        # Top
        indices.extend([start_idx + 1, start_idx + 5, start_idx + 6])
        indices.extend([start_idx + 1, start_idx + 6, start_idx + 2])
        # Bottom
        indices.extend([start_idx + 0, start_idx + 3, start_idx + 7])
        indices.extend([start_idx + 0, start_idx + 7, start_idx + 4])
        # Front
        indices.extend([start_idx + 0, start_idx + 4, start_idx + 5])
        indices.extend([start_idx + 0, start_idx + 5, start_idx + 1])
        # Back
        indices.extend([start_idx + 3, start_idx + 2, start_idx + 6])
        indices.extend([start_idx + 3, start_idx + 6, start_idx + 7])
        # Outer end
        indices.extend([start_idx + 4, start_idx + 7, start_idx + 6])
        indices.extend([start_idx + 4, start_idx + 6, start_idx + 5])

    def get_stage_center(self, stage: int) -> Tuple[float, float, float]:
        """
        Get the center position of a stage.

        Args:
            stage: Stage number (1-indexed)

        Returns:
            (x, y, z) center position
        """
        if stage < 1 or stage > self.params.num_stages:
            raise ValueError(f"Stage must be between 1 and {self.params.num_stages}")

        corners = self.stage_corners[stage - 1]
        x = (corners['left'][0] + corners['right'][0]) / 2
        y = corners['left'][1] + self.params.stage_height / 2
        z = self.params.center[2]

        return (x, y, z)

    def get_air_velocity(self, volumetric_flow: float) -> float:
        """
        Calculate air velocity in channel for given flow rate.

        Args:
            volumetric_flow: Volumetric flow rate [m³/s]

        Returns:
            Air velocity in channel [m/s]
        """
        return volumetric_flow / self.params.channel_cross_section_area

    def calculate_cut_size_d50(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
        air_density: float = 1.204,
        air_viscosity: float = 1.82e-5,
    ) -> float:
        """
        Calculate the cut size (d50) for given flow rate.

        Particles smaller than d50 go to fines, larger go to coarse.

        Args:
            volumetric_flow: Flow rate [m³/s]
            particle_density: Particle density [kg/m³]
            air_density: Air density [kg/m³]
            air_viscosity: Dynamic viscosity [Pa·s]

        Returns:
            d50 cut size [m]
        """
        v_air = self.get_air_velocity(volumetric_flow)
        g = 9.81
        # Stokes law rearranged for terminal velocity = air velocity
        # v_t = d² × (ρ_p - ρ_f) × g / (18 × μ)
        # d50 = √(18 × μ × v_air / (g × (ρ_p - ρ_f)))
        d50 = np.sqrt(
            18 * air_viscosity * v_air / (g * (particle_density - air_density))
        )
        return d50

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
        # Rearrange: v_air = d50² × (ρ_p - ρ_f) × g / (18 × μ)
        v_air = (
            (target_d50 ** 2)
            * (particle_density - air_density)
            * g
            / (18 * air_viscosity)
        )
        Q = v_air * self.params.channel_cross_section_area
        return Q

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
        v_air = self.get_air_velocity(volumetric_flow)

        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "d50_um": d50 * 1e6,
            "air_velocity_m_s": v_air,
            "volumetric_flow_m3_h": volumetric_flow * 3600,
        }

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

        if v_air > 5.0:
            result["warnings"].append(
                f"Air velocity ({v_air:.1f} m/s) is high. May cause excessive turbulence."
            )

        if v_air > 20.0:
            result["errors"].append(
                f"Air velocity ({v_air:.1f} m/s) is very high. "
                "Zigzag will act as transport duct, not separator."
            )
            result["valid"] = False

        Q_for_max = self.calculate_required_flow_for_d50(
            max_particle_size * 0.8, particle_density
        )
        Q_for_min = self.calculate_required_flow_for_d50(
            min_particle_size * 1.2, particle_density
        )
        result["recommended_flow_range_m3_s"] = (Q_for_min, Q_for_max)
        result["recommended_flow_range_m3_h"] = (
            Q_for_min * 3600,
            Q_for_max * 3600,
        )

        return result

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
        - air_inlet: Main air inlet at bottom (rectangular, -Y direction - air flows up)
        - feed_inlet: Particle feed inlet at feed stage (rectangular, +X direction)
        - fines_outlet: Top outlet for light particles/protein (+Y direction)
        - coarse_outlet: Bottom outlet for heavy particles/starch (-Y direction)
        """
        p = self.params
        
        # Air inlet at bottom center
        # Position at bottom of air inlet box
        x_bottom = self.stage_corners[0]['left'][0] + p.channel_width / 2
        air_inlet = ConnectionPort(
            position=(x_bottom, p.center[1] - p.air_inlet_height, p.center[2]),
            direction=(0.0, -1.0, 0.0),  # Faces down (air comes from below)
            width=p.air_inlet_width,
            height=p.channel_depth,  # Depth becomes height for rectangular
            port_type=PortType.RECTANGULAR,
            name="air_inlet"
        )
        
        # Feed inlet at feed stage (right side of channel)
        stage_idx = p.feed_stage - 1
        corners = self.stage_corners[stage_idx]
        x_right = corners['right'][0]
        y_feed = corners['right'][1] + p.stage_height / 2
        feed_length = p.channel_width * 0.3  # Same as in _add_feed_inlet
        
        feed_inlet = ConnectionPort(
            position=(x_right + feed_length, y_feed, p.center[2]),
            direction=(1.0, 0.0, 0.0),  # Faces outward (+X)
            width=p.feed_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="feed_inlet"
        )
        
        # Fines outlet at top (light particles/protein carried up by air)
        x_top = self.stage_corners[-1]['left'][0] + p.channel_width / 2
        fines_outlet = ConnectionPort(
            position=(x_top, p.center[1] + p.total_height + p.fines_outlet_height, p.center[2]),
            direction=(0.0, 1.0, 0.0),  # Faces up
            width=p.fines_outlet_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="fines_outlet"
        )
        
        # Coarse outlet at bottom (heavy particles/starch fall down)
        coarse_outlet = ConnectionPort(
            position=(x_bottom, p.center[1] - p.air_inlet_height - p.coarse_outlet_height, p.center[2]),
            direction=(0.0, -1.0, 0.0),  # Faces down
            width=p.coarse_outlet_width,
            height=p.channel_depth,
            port_type=PortType.RECTANGULAR,
            name="coarse_outlet"
        )
        
        return {
            'air_inlet': air_inlet,
            'feed_inlet': feed_inlet,
            'fines_outlet': fines_outlet,
            'coarse_outlet': coarse_outlet
        }


def create_standard_zigzag_classifier(
    channel_width: float = 0.15,
    num_stages: int = 5,
    channel_depth: float = 0.30
) -> ZigzagClassifier:
    """
    Create a standard zigzag classifier with typical proportions.

    Args:
        channel_width: Width of zigzag channel [m]
        num_stages: Number of stages (3-7 typical)
        channel_depth: Depth of channel [m]

    Returns:
        ZigzagClassifier instance
    """
    params = ZigzagClassifierParams(
        channel_width=channel_width,
        channel_depth=channel_depth,
        num_stages=num_stages,
        stage_height=channel_width * 1.5,  # Typical ratio
        zigzag_angle=np.radians(120),      # 120° typical
        feed_stage=(num_stages + 1) // 2,  # Middle stage
        feed_width=channel_width * 0.5,
        air_inlet_width=channel_width,
        air_inlet_height=channel_width * 0.5,
        fines_outlet_width=channel_width,
        fines_outlet_height=channel_width * 0.5,
        coarse_outlet_width=channel_width * 0.5,
        coarse_outlet_height=channel_width * 0.3,
    )

    return ZigzagClassifier(params)
