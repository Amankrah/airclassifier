"""
Complete air classifier system assembly.

This module provides the master assembly that integrates all system phases:
- Phase 1: Classification (Zigzag + Cyclones + Bag Filter)
- Phase 2: Feed System (Hopper + Airlock + Screw Feeder + Deagglomerator)
- Phase 3: Air System (Blower + Filter + Damper)
- Phase 4: Ductwork (Ducts + Transitions connecting the 3 systems)
- Phase 5: Exhaust (Silencer + Stack)

The complete system positions all components in 3D space with proper
duct connections between:
1. Air System outlet → Venturi air_inlet
2. Feed System outlet → Venturi solids_inlet
3. Bag Filter clean_air_outlet → Exhaust silencer
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class CompleteSystemParams:
    """
    Parameters for complete air classifier system.

    Attributes:
        throughput_kg_h: Design throughput [kg/h]
        cut_size_um: Target cut size [μm]
        air_flow_m3_h: Design air flow rate [m³/h]

        # Layout parameters
        feed_position: Feed system position offset (x, y, z) [m]
        classifier_position: Classification system position (x, y, z) [m]
        air_system_position: Air system position (x, y, z) [m]

        # Include flags
        include_feed_system: Whether to include feed system
        include_air_system: Whether to include air system
        include_ductwork: Whether to include connecting ductwork
        include_support_structure: Whether to include support frame
        include_exhaust: Whether to include silencer and stack

        # Sizing
        classifier_width: Zigzag classifier width [m]
        cyclone_diameter: Primary cyclone diameter [m]
    """
    throughput_kg_h: float = 500.0
    cut_size_um: float = 20.0
    air_flow_m3_h: float = 3000.0

    # Layout positions - optimized for protein separation
    # Feed positioned in +Y (behind classifier), elevated for steep gravity chute
    # Outlet at ~20° from vertical for optimal powder flow into venturi
    feed_position: Tuple[float, float, float] = (0.0, 1.0, 3.5)  # +Y behind, +X right, elevated
    classifier_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    air_system_position: Tuple[float, float, float] = (0.0, -3.0, 0.0)

    # Include flags
    include_feed_system: bool = True
    include_air_system: bool = True
    include_ductwork: bool = True
    include_support_structure: bool = False
    include_exhaust: bool = True

    # Optional custom classification system params (venturi throat, cyclone sizes, etc.)
    classification_params: Optional[Any] = None

    # Sizing parameters
    classifier_width: float = 0.15
    cyclone_diameter: float = 0.3
    hopper_diameter: float = 0.6
    main_duct_diameter: float = 0.2
    frame_width: float = 4.0
    frame_depth: float = 3.0
    frame_height: float = 3.5
    stack_height: float = 4.0


class CompleteClassifierAssembly:
    """
    Complete air classifier system assembly.
    
    Integrates all system phases into a single unified assembly
    with proper positioning and connections.
    """
    
    def __init__(self, params: CompleteSystemParams):
        """
        Initialize complete classifier assembly.

        Args:
            params: System parameters
        """
        self.params = params
        self._subsystems: Dict[str, Any] = {}
        self._components: Dict[str, Any] = {}
        self._duct_connections: List[Tuple[Any, Tuple[float, float, float]]] = []
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._build_system()
    
    def _build_system(self):
        """Build the complete system assembly."""
        p = self.params

        # Build classification system (core) FIRST
        self._build_classification_system()

        # Build feed system
        if p.include_feed_system:
            self._build_feed_system()

        # Build air system
        if p.include_air_system:
            self._build_air_system()

        # Build exhaust system BEFORE ductwork (so silencer exists for connections)
        if p.include_exhaust:
            self._build_exhaust_system()

        # Build connecting ductwork (after all systems exist)
        if p.include_ductwork:
            self._build_ductwork()
    
    def _build_classification_system(self):
        """
        Build the classification system (zigzag + cyclones + bag filter).
        
        Coordinate System (Y-up):
        - X: Horizontal (width)
        - Y: Vertical (height) - UP
        - Z: Horizontal (depth)
        """
        from .classification import create_standard_classification_system

        p = self.params
        cx, cy, cz = p.classifier_position

        # Position classification - elevate 0.5m above base position
        # Y is the vertical axis, so add elevation to Y
        class_y = cy + 0.5

        classification = create_standard_classification_system(
            device="cpu", params=p.classification_params
        )
        self._subsystems['classification'] = classification

        # Store position offset for mesh transformation
        self._subsystems['classification_offset'] = (cx, class_y, cz)
    
    def _build_feed_system(self):
        """
        Build the feed system (hopper + airlock + screw + deagglomerator).
        
        COORDINATE SYSTEM:
        - X+: From air filter toward deagglomerator (horizontal)
        - Y+: Vertical (upward toward bag filter top outlet)
        - Z+: Distance away from classification system (depth)
        
        Positioning the feed at 15-degree angle:
        - Z_distance: How far the feed is from the classification system
        - Y_rise: Vertical elevation = Z_distance × tan(15°)
        - This creates a 15-degree downward slope from feed to venturi
        
        The feed outlet faces toward the venturi solids_inlet and they are
        connected with an angled shaft duct at 15 degrees from horizontal.
        """
        from .feed_system import create_standard_feed_system
        
        p = self.params
        
        # Create feed system first to get its dimensions
        feed = create_standard_feed_system(device="cpu")
        self._subsystems['feed_system'] = feed
        
        # Get classification system's venturi position for alignment
        classification = self._subsystems.get('classification')
        class_offset = np.array(self._subsystems.get('classification_offset', (0, 0, 0)))
        
        if classification is not None and hasattr(classification, 'venturi'):
            # Get venturi's solids_inlet position
            venturi = classification.venturi
            class_positions = classification.get_component_positions()
            venturi_pos = np.array(class_positions['venturi']) + class_offset
            
            solids_port = venturi.ports['solids_inlet']
            solids_inlet_world = venturi_pos + np.array(solids_port.position)
            
            # Calculate feed position based on 15-degree angle
            angle_deg = 15.0
            angle_rad = np.radians(angle_deg)
            
            # Get feed system outlet position relative to feed origin
            feed_positions = feed.get_component_positions()
            deagg_local_pos = np.array(feed_positions['deagglomerator'])
            deagg_outlet = feed.deagglomerator.ports['outlet']
            outlet_offset = deagg_local_pos + np.array(deagg_outlet.position)
            
            # Target position (venturi solids_inlet)
            target_x, target_y, target_z = solids_inlet_world
            
            # ============================================================
            # Position feed at 15-degree angle from venturi solids_inlet
            # 
            # Z_distance: Horizontal distance away from classification system
            # Y_rise: Vertical elevation = Z_distance × tan(15°)
            # 
            # The feed is positioned:
            # - At same X as target (aligned along X axis)
            # - Above target in Y (elevated)
            # - Away from target in Z (positive Z, away from classifier)
            # ============================================================
            
            # Horizontal distance in Z (away from classification system)
            z_distance = 1.0  # meters - distance from classifier in Z direction
            
            # Vertical rise (Y) based on 15-degree angle
            # Y_rise = Z_distance × tan(15°)
            y_rise = z_distance * np.tan(angle_rad)  # ~0.27m for 1.0m Z distance
            
            # Add extra clearance for elbows and duct routing
            elbow_clearance = 0.2  # clearance for elbow turns
            
            # Feed outlet desired world position:
            # - Same X as target
            # - Above target in Y by the calculated rise + clearance
            # - Away from target in +Z direction
            feed_outlet_target_x = target_x
            feed_outlet_target_y = target_y + y_rise + elbow_clearance
            feed_outlet_target_z = target_z + z_distance
            
            # Calculate feed system origin position
            # (subtract outlet offset from desired outlet position)
            feed_x = feed_outlet_target_x - outlet_offset[0]
            feed_y = feed_outlet_target_y - outlet_offset[1]
            feed_z = feed_outlet_target_z - outlet_offset[2]
            
        else:
            # Fallback to default position if classification not available
            # Y-up coordinate system: Y is vertical (height)
            fx, fy, fz = p.feed_position
            if p.include_support_structure:
                feed_y = fy + p.frame_height + 1.0  # Add height to Y (vertical)
            else:
                feed_y = fy + 1.0  # Add height to Y (vertical)
            feed_x, feed_z = fx, fz
        
        self._subsystems['feed_system_offset'] = (feed_x, feed_y, feed_z)
    
    def _build_air_system(self):
        """Build the air system (blower + filter + damper)."""
        from .air_system import create_standard_air_system
        
        p = self.params
        ax, ay, az = p.air_system_position
        
        air = create_standard_air_system(device="cpu")
        self._subsystems['air_system'] = air
        # Move air system toward +Z (toward classifier/feeder) for proper elbow alignment
        # The Z offset ensures the second elbow output aligns with target_z for the vertical duct
        self._subsystems['air_system_offset'] = (ax, ay, az + 0.6)
    
    def _build_ductwork(self):
        """
        Build connecting ductwork between systems.

        Creates three physical duct connections:
        1. Air System outlet -> Venturi air_inlet (pressurized air supply)
        2. Feed System outlet -> Venturi solids_inlet (powder feed chute)
        3. Bag Filter clean_air_outlet -> Exhaust silencer (clean air to atmosphere)

        Uses the same port-based connection pattern as classification.py and air_system.py.
        """
        p = self.params

        # Store duct sections as list of (component, world_position) tuples
        # Same format as classification.py's _duct_sections
        self._duct_connections = []
        # Air system -> Venturi air_inlet path only (for flow physics)
        self._air_to_venturi_ducts: List[Tuple[Any, Tuple[float, float, float]]] = []
        # Feed system (deagglomerator) -> Venturi solids_inlet path only (for flow physics)
        self._feed_to_venturi_ducts: List[Tuple[Any, Tuple[float, float, float]]] = []

        # Get classification system's venturi ports
        classification = self._subsystems.get('classification')
        if classification is None:
            return

        # Get venturi from classification system
        venturi = classification.venturi
        class_offset = np.array(self._subsystems.get('classification_offset', (0, 0, 0)))

        # Get component positions from classification system
        class_positions = classification.get_component_positions()
        venturi_pos = np.array(class_positions['venturi']) + class_offset

        # 1. Air System -> Venturi air_inlet (primary air supply from -Y direction)
        if p.include_air_system and 'air_system' in self._subsystems:
            self._build_air_to_venturi_connection(venturi, venturi_pos)

        # 2. Feed System -> Venturi solids_inlet (gravity feed from +Z direction)
        if p.include_feed_system and 'feed_system' in self._subsystems:
            self._build_feed_to_solids_inlet(venturi, venturi_pos)

        # 3. Bag Filter -> Exhaust silencer (clean air exhaust)
        if p.include_exhaust and 'silencer' in self._components:
            self._build_bagfilter_to_exhaust_connection(classification, class_offset)
    
    def _build_air_to_venturi_connection(self, venturi, venturi_pos: np.ndarray):
        """
        Build ductwork from Air System outlet to Venturi air_inlet.

        The air system connects to the venturi's main air inlet (air_inlet port).
        For a vertical venturi (axis='y'), air enters from -Y direction.

        Route:
        - Air from blower outlet (+X) -> elbows -> approach venturi in +Y direction

        Coordinates:
        - Air Start: damper outlet, direction +X
        - Target: venturi air_inlet, expects flow from -Y direction (duct approaches in +Y)
        """
        from ..components.ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams
        from ..components.transitions import Transition, TransitionParams

        p = self.params
        gap = 0.005  # 5mm flange gap

        # Get air system outlet position and diameter
        air_system = self._subsystems['air_system']
        air_offset = np.array(self._subsystems.get('air_system_offset', (0, 0, 0)))

        if hasattr(air_system, 'dampers') and air_system.dampers:
            last_damper = air_system.dampers[-1]
            damper_pos = np.array(air_system._damper_positions[-1])
            air_outlet_port = last_damper.ports['outlet']
            air_outlet_world = air_offset + damper_pos + np.array(air_outlet_port.position)
            air_outlet_d = air_outlet_port.diameter
        else:
            air_outlet_world = air_offset + np.array([1.0, 0.0, 0.0])
            air_outlet_d = p.main_duct_diameter

        # Venturi air inlet position (in world coordinates)
        venturi_air_port = venturi.ports['air_inlet']
        venturi_air_world = venturi_pos + np.array(venturi_air_port.position)
        venturi_air_d = venturi_air_port.diameter

        # Duct dimensions
        duct_d = min(air_outlet_d, p.main_duct_diameter)
        R = duct_d * 1.0  # Bend radius
        trans_len = 0.12  # Transition length

        # Reference coordinates
        start_x, start_y, start_z = air_outlet_world
        target_x, target_y, target_z = venturi_air_world

        # ============================================================
        # TARGET-ALIGNED routing: Work backwards from venturi air_inlet
        # to ensure the duct terminates exactly at the target position.
        # 
        # Venturi air_inlet expects air from -Y, so final approach is +Y
        # Final transition outlet must be at (target_x, target_y, target_z)
        # ============================================================

        # Work backwards from target:
        # - Transition outlet at (target_x, target_y, target_z)
        # - Transition inlet at (target_x, target_y - trans_len, target_z)
        # - Final duct5 approaches in +Y at X=target_x, Z=target_z
        # - Elbow4 turns -X to +Y, so elbow4 inlet_y determines final Y
        # - etc.

        # Transition position (outlet at target)
        trans_outlet_y = target_y
        trans_inlet_y = target_y - trans_len

        # Elbow4: -X to +Y turn
        # outlet_x = inlet_x - R, outlet_y = inlet_y + R
        # We want outlet at (target_x, trans_inlet_y - gap - duct5_len)
        # For simplicity, minimize duct5_len (just a small approach)
        duct5_len = 0.05
        e4_outlet_y = trans_inlet_y - gap - duct5_len
        e4_outlet_x = target_x
        
        e4_inlet_x = e4_outlet_x + R
        e4_inlet_y = e4_outlet_y - R
        
        # Route horizontal at target_z level for final approach
        route_z = target_z

        # Step 1: Short duct from damper outlet in +X direction
        d1_len = 0.1
        d1_pos = (start_x + gap, start_y, start_z)

        duct1 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d1_len, wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct1, d1_pos))
        self._air_to_venturi_ducts.append((duct1, d1_pos))

        curr_x = start_x + gap + d1_len

        # Step 2: Elbow1 - turn from +X to -Z (down)
        e1_inlet = (curr_x + gap, start_y, start_z)
        e1_outlet_x = e1_inlet[0] + R
        e1_outlet_z = start_z - R

        elbow1 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(1.0, 0.0, 0.0), rotation_axis=(0.0, -1.0, 0.0),
        ))
        self._duct_connections.append((elbow1, e1_inlet))
        self._air_to_venturi_ducts.append((elbow1, e1_inlet))

        # Step 3: Vertical duct down in -Z to reach route_z level
        # Elbow2 will turn -Z to +Y, dropping by R in Z
        d2_target_z = route_z + R + gap
        d2_len = max(e1_outlet_z - gap - d2_target_z, 0.05)
        d2_pos = (e1_outlet_x, start_y, e1_outlet_z - gap)

        duct2 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d2_len, wall_thickness=0.002,
            direction=(0.0, 0.0, -1.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct2, d2_pos))
        self._air_to_venturi_ducts.append((duct2, d2_pos))

        e2_inlet_z = e1_outlet_z - gap - d2_len

        # Step 4: Elbow2 - turn from -Z to +Y (toward classifier)
        e2_inlet = (e1_outlet_x, start_y, e2_inlet_z - gap)
        e2_outlet_y = start_y + R
        e2_outlet_z = e2_inlet[2] - R  # This should be close to route_z

        elbow2 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 0.0, -1.0), rotation_axis=(-1.0, 0.0, 0.0),
        ))
        self._duct_connections.append((elbow2, e2_inlet))
        self._air_to_venturi_ducts.append((elbow2, e2_inlet))

        # Now we're at (e1_outlet_x, e2_outlet_y, e2_outlet_z) going +Y
        # We need to reach elbow4 inlet at (e4_inlet_x, e4_inlet_y, route_z)
        
        # We need to turn from +Y to -X at some point
        # Elbow3: +Y to -X, outlet_x = inlet_x - R, outlet_y = inlet_y + R
        
        # After elbow3, duct4 goes in -X to reach e4_inlet_x
        # So: e3_outlet_x - gap - d4_len = e4_inlet_x + gap
        # e3_outlet_x = e3_inlet_x - R = e1_outlet_x - R (since duct3 is vertical in Y)
        
        # We need duct4_len: (e1_outlet_x - R - gap) - (e4_inlet_x + gap) = duct4_len
        d4_len = max((e1_outlet_x - R - gap) - (e4_inlet_x + gap), 0.1)
        
        # e3_outlet_y should equal e4_inlet_y (or close)
        # e3_outlet_y = e3_inlet_y + R
        # So e3_inlet_y = e4_inlet_y - R - gap (for duct4 in between)
        
        # Actually, duct4 is horizontal in -X at y = e3_outlet_y
        # So e3_outlet_y should equal e4_inlet_y
        e3_outlet_y = e4_inlet_y
        e3_inlet_y = e3_outlet_y - R
        
        # Duct3 goes from e2_outlet_y to e3_inlet_y in +Y direction
        # Route at target_z level for alignment with venturi
        d3_len = max(e3_inlet_y - (e2_outlet_y + gap) - gap, 0.1)
        d3_pos = (e1_outlet_x, e2_outlet_y + gap, target_z)

        duct3 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d3_len, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct3, d3_pos))
        self._air_to_venturi_ducts.append((duct3, d3_pos))

        d3_end_y = e2_outlet_y + gap + d3_len

        # Step 6: Elbow3 - turn from +Y to -X
        e3_inlet = (e1_outlet_x, d3_end_y + gap, target_z)
        e3_outlet_x = e1_outlet_x - R
        e3_outlet_y_actual = e3_inlet[1] + R

        elbow3 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0), rotation_axis=(0.0, 0.0, -1.0),
        ))
        self._duct_connections.append((elbow3, e3_inlet))
        self._air_to_venturi_ducts.append((elbow3, e3_inlet))

        # Step 7: Horizontal duct in -X direction toward elbow4 (aligned with venturi)
        # CRITICAL: elbow4 must be positioned so its outlet aligns with target_x, target_z
        # Elbow4 turns -X to +Y: outlet_x = inlet_x - R, outlet_y = inlet_y + R
        # We want outlet at (target_x, ?, target_z)
        # So elbow4 inlet_x = target_x + R
        
        e4_inlet_x_aligned = target_x + R
        e4_inlet_y_aligned = e3_outlet_y_actual
        e4_inlet_z_aligned = target_z  # Align with target Z
        
        # Calculate duct4 length to reach the aligned elbow4 position
        d4_len_actual = max(e3_outlet_x - gap - (e4_inlet_x_aligned + gap), 0.05)
        d4_pos = (e3_outlet_x - gap, e3_outlet_y_actual, target_z)  # Route at target_z

        duct4 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d4_len_actual, wall_thickness=0.002,
            direction=(-1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct4, d4_pos))
        self._air_to_venturi_ducts.append((duct4, d4_pos))

        d4_end_x = e3_outlet_x - gap - d4_len_actual

        # Step 8: Elbow4 - turn from -X to +Y (final approach)
        e4_inlet_actual = (e4_inlet_x_aligned, e4_inlet_y_aligned, e4_inlet_z_aligned)
        e4_outlet_x_actual = target_x  # Now properly aligned with venturi
        e4_outlet_y_actual = e3_outlet_y_actual + R
        
        elbow4 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(-1.0, 0.0, 0.0), rotation_axis=(0.0, 0.0, 1.0),
        ))
        self._duct_connections.append((elbow4, e4_inlet_actual))
        self._air_to_venturi_ducts.append((elbow4, e4_inlet_actual))

        # Step 9: Final approach duct in +Y direction
        # Connect from e4_outlet to transition inlet
        d5_len_actual = max(trans_inlet_y - gap - (e4_outlet_y_actual + gap), 0.02)
        
        if d5_len_actual > 0.02:
            d5_pos = (target_x, e4_outlet_y_actual + gap, target_z)
            duct5 = RoundDuct(RoundDuctParams(
                diameter=duct_d, length=d5_len_actual, wall_thickness=0.002,
                direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct5, d5_pos))
            self._air_to_venturi_ducts.append((duct5, d5_pos))
            trans_pos_y = e4_outlet_y_actual + gap + d5_len_actual + gap
        else:
            trans_pos_y = e4_outlet_y_actual + gap

        # Step 10: Transition to venturi diameter
        # Position at target_x, target_z so outlet aligns with venturi air inlet
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(duct_d,), outlet_dimensions=(venturi_air_d,),
            length=trans_len, concentric=True, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        # Place transition aligned with venturi air inlet
        self._duct_connections.append((trans, (target_x, trans_pos_y, target_z)))
        self._air_to_venturi_ducts.append((trans, (target_x, trans_pos_y, target_z)))

    def _build_feed_to_solids_inlet(self, venturi, venturi_pos: np.ndarray):
        """
        Build direct angled shaft duct from Feed System outlet to Venturi solids_inlet.

        COORDINATE SYSTEM:
        - X+: From air filter toward deagglomerator (horizontal)
        - Y+: Vertical (upward toward bag filter top outlet)
        - Z+: Distance away from classification system (depth)
        
        FEED POSITION:
        - Feed is at +Z (away from classifier, positive Z)
        - Feed is elevated in Y (above venturi)
        - Feed outlet points -Y (downward from deagglomerator)
        
        SHAFT DUCT PATH (15-degree angle):
        - Shaft goes from feed toward venturi
        - Direction: -Z (toward classifier) with -Y component (descending)
        - Angle: 15 degrees below horizontal
        
        Path: Feed outlet (-Y) -> Elbow (turn to -Z) -> Angled shaft (15°) -> Venturi inlet
        """
        from ..components.ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams
        from ..components.transitions import Transition, TransitionParams

        p = self.params
        gap = 0.005  # 5mm flange gap

        # Get feed system outlet
        feed_system = self._subsystems['feed_system']
        feed_offset = np.array(self._subsystems.get('feed_system_offset', (0, 0, 0)))
        feed_positions = feed_system.get_component_positions()
        deagg_local_pos = np.array(feed_positions['deagglomerator'])
        deagg_outlet = feed_system.deagglomerator.ports['outlet']

        feed_outlet_world = feed_offset + deagg_local_pos + np.array(deagg_outlet.position)
        feed_outlet_d = deagg_outlet.diameter

        # Get venturi solids inlet position and direction
        solids_port = venturi.ports['solids_inlet']
        solids_inlet_world = venturi_pos + np.array(solids_port.position)
        solids_inlet_d = solids_port.diameter
        solids_inlet_dir = np.array(solids_port.direction)

        # Coordinates
        feed_x, feed_y, feed_z = feed_outlet_world
        target_x, target_y, target_z = solids_inlet_world

        # ============================================================
        # ANGLED SHAFT (geometry-derived angle from horizontal)
        # 
        # Feed is at (feed_x, feed_y, feed_z) - high Y, high Z
        # Target is at (target_x, target_y, target_z) - lower Y, lower Z
        # 
        # Shaft direction: -Z (toward classifier) with -Y (descending)
        # Angle measured from horizontal (Z axis); stored for physics/kinetics.
        # ============================================================

        # Shaft parameters (angle from horizontal for gravity-driven flow)
        shaft_angle_deg = 15.0
        self._feed_chute_angle_deg = shaft_angle_deg
        shaft_angle_rad = np.radians(shaft_angle_deg)
        
        shaft_d = max(min(feed_outlet_d * 0.5, solids_inlet_d * 1.5), 0.035)
        elbow_R = shaft_d * 1.2
        trans_len = 0.05

        # Shaft direction unit vector:
        # - Primary direction: -Z (toward classifier, horizontal)
        # - Vertical component: -Y (descending) at 15 degrees
        # cos(15°) is the horizontal component (Z), sin(15°) is vertical (Y)
        cos_a = np.cos(shaft_angle_rad)
        sin_a = np.sin(shaft_angle_rad)
        
        # Shaft goes toward -Z (classifier) and descends in -Y
        shaft_dir = np.array([0.0, -sin_a, -cos_a])
        shaft_dir = shaft_dir / np.linalg.norm(shaft_dir)
        
        # ============================================================
        # Calculate target position and required drop
        # The shaft must reach the venturi solids_inlet position
        # ============================================================
        
        # Calculate how much we need to drop in Y to reach the inlet
        # The solids_inlet is at target_y, we start near feed_y
        # We need to drop from feed outlet level to slightly above target
        
        # Target: where the shaft should end (at solids_inlet)
        # The shaft approaches at 15 degrees, so we need to account for that
        
        # Z distance from feed to target
        z_distance = feed_z - target_z
        
        # For a 15-degree descent over z_distance:
        # Y drop from shaft = z_distance * tan(15°)
        shaft_y_drop = z_distance * np.tan(shaft_angle_rad)
        
        # Required Y position where shaft starts (after elbows)
        # shaft_start_y - shaft_y_drop = target_y (approximately)
        # So shaft_start_y = target_y + shaft_y_drop
        # Add extra drop offset to ensure shaft end aligns with inlet
        extra_drop = 0.07# Additional drop to properly meet the inlet
        required_shaft_start_y = target_y + shaft_y_drop - extra_drop
        
        # ============================================================
        # Step 1: Short stub from deagglomerator outlet (-Y direction)
        # ============================================================
        stub_len = 0.04
        stub_pos = (feed_x, feed_y - gap, feed_z)
        
        stub_duct = RoundDuct(RoundDuctParams(
            diameter=shaft_d, length=stub_len, wall_thickness=0.002,
            direction=(0.0, -1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((stub_duct, stub_pos))
        self._feed_to_venturi_ducts.append((stub_duct, stub_pos))

        stub_end_y = feed_y - gap - stub_len
        
        # ============================================================
        # Step 2: Calculate vertical drop needed
        # ============================================================
        
        # After first elbow, we'll be at stub_end_y - gap - elbow_R
        # We need to drop further to reach required_shaft_start_y
        after_first_elbow_y = stub_end_y - gap - elbow_R
        
        # Total drop needed from current position to shaft start
        # Add elbow_R for the second elbow that turns toward -Z
        drop_needed = after_first_elbow_y - required_shaft_start_y - elbow_R - gap * 2
        drop_needed = max(drop_needed, 0.05)  # Minimum drop
        
        # ============================================================
        # Step 3: Extended elbow + vertical drop
        # First elbow: -Y continues down, then add drop duct, then second elbow
        # ============================================================
        
        # Keep going -Y with a longer drop before turning
        # Vertical drop duct
        drop_pos = (feed_x, stub_end_y - gap, feed_z)
        
        drop_duct = RoundDuct(RoundDuctParams(
            diameter=shaft_d, length=drop_needed, wall_thickness=0.002,
            direction=(0.0, -1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((drop_duct, drop_pos))
        self._feed_to_venturi_ducts.append((drop_duct, drop_pos))

        drop_end_y = stub_end_y - gap - drop_needed
        
        # ============================================================
        # Step 4: Elbow - turn from -Y to -Z (toward classifier)
        # ============================================================
        e1_pos = (feed_x, drop_end_y - gap, feed_z)
        
        elbow1 = DuctElbow(DuctElbowParams(
            diameter=shaft_d, bend_radius=elbow_R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, -1.0, 0.0),
            rotation_axis=(-1.0, 0.0, 0.0),  # -X axis: -Y turns to -Z
        ))
        self._duct_connections.append((elbow1, e1_pos))
        self._feed_to_venturi_ducts.append((elbow1, e1_pos))

        # After elbow1: now going -Z direction (toward classifier)
        e1_out_y = drop_end_y - gap - elbow_R
        e1_out_z = feed_z - elbow_R
        
        # ============================================================
        # Step 5: Angled shaft duct at 15 degrees
        # Goes from elbow outlet toward venturi solids_inlet
        # Direction: -Z (toward classifier) with -Y descent at 15 degrees
        # ============================================================
        
        shaft_start = np.array([feed_x, e1_out_y - gap, e1_out_z - gap])
        
        # Calculate shaft length based on Z distance to target
        z_travel = e1_out_z - gap - target_z - trans_len
        
        # For 15-degree angle: Z_travel = shaft_len * cos(15°)
        if z_travel > 0.05:
            shaft_len = z_travel / cos_a
            
            shaft_dir_tuple = tuple(shaft_dir)
            
            shaft_duct = RoundDuct(RoundDuctParams(
                diameter=shaft_d, length=shaft_len, wall_thickness=0.002,
                direction=shaft_dir_tuple, center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((shaft_duct, tuple(shaft_start)))
            self._feed_to_venturi_ducts.append((shaft_duct, tuple(shaft_start)))

            # Shaft end position
            shaft_end = shaft_start + shaft_dir * shaft_len
            shaft_end_y = shaft_end[1]
            shaft_end_z = shaft_end[2]
        else:
            shaft_end_y = shaft_start[1]
            shaft_end_z = shaft_start[2]
        
        # ============================================================
        # Step 6: Transition to venturi solids_inlet
        # ============================================================
        
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(shaft_d,), outlet_dimensions=(solids_inlet_d,),
            length=trans_len, concentric=True, wall_thickness=0.002,
            direction=tuple(shaft_dir), center=(0, 0, 0), flanged=True,
        ))
        
        # Position transition to connect to solids_inlet
        trans_pos = (target_x, shaft_end_y - gap, shaft_end_z - gap)
        self._duct_connections.append((trans, trans_pos))
        self._feed_to_venturi_ducts.append((trans, trans_pos))

    def _build_feed_to_venturi_connection_UNUSED(self, venturi, venturi_pos: np.ndarray):
        """
        Build gravity chute from Feed System outlet to Venturi solids_inlet.

        The venturi solids_inlet expects material from +X direction, so the chute
        must approach from the +X side (entering in -X direction).

        Route: Deagglomerator outlet (-Y dir) -> elbow to +X/down -> 
               angled descent toward classifier -> elbow to approach from +X ->
               final approach in -X direction -> transition to venturi

        Feed outlet is at ~(-3.84, -1.97, 1.0) pointing -Y, diameter 96mm
        Venturi solids inlet is at ~(0.02, 0.11, 0.5) expecting from +X, diameter 32mm
        """
        from ..components.ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams
        from ..components.transitions import Transition, TransitionParams

        p = self.params
        gap = 0.01  # 10mm flange gap

        # Get feed system outlet position and diameter
        feed_system = self._subsystems['feed_system']
        feed_offset = np.array(self._subsystems.get('feed_system_offset', (0, 0, 0)))

        # Get component positions within feed system
        feed_positions = feed_system.get_component_positions()
        deagg_local_pos = np.array(feed_positions['deagglomerator'])
        deagg_outlet = feed_system.deagglomerator.ports['outlet']

        # Calculate world position of deagglomerator outlet
        feed_outlet_world = feed_offset + deagg_local_pos + np.array(deagg_outlet.position)
        feed_outlet_d = deagg_outlet.diameter
        feed_outlet_dir = np.array(deagg_outlet.direction)  # -Y direction

        # Venturi solids inlet position (in world coordinates)
        venturi_solids_port = venturi.ports['solids_inlet']
        venturi_solids_world = venturi_pos + np.array(venturi_solids_port.position)
        venturi_solids_d = venturi_solids_port.diameter
        # Venturi expects flow from +X direction, so chute approaches in -X

        # Chute diameter (larger for powder flow)
        chute_d = max(feed_outlet_d * 0.7, venturi_solids_d * 2.5, 0.05)
        R = chute_d * 1.2  # Elbow bend radius
        trans_len = 0.10  # Transition length

        # Reference coordinates
        start_x, start_y, start_z = feed_outlet_world
        target_x, target_y, target_z = venturi_solids_world

        # ============================================================
        # Route: The chute must approach venturi from +X side (in -X direction)
        # 
        # Path: 
        # 1. Short duct from deagg outlet in -Y direction
        # 2. Elbow: -Y to +X (turn toward classifier)
        # 3. Angled descent: go +X and -Z simultaneously toward target level
        # 4. Elbow: approach direction to -X (for venturi entry)
        # 5. Final approach in -X direction
        # 6. Transition to venturi diameter
        # ============================================================

        # Step 1: Short duct from deagglomerator outlet in -Y direction
        d1_len = 0.08
        d1_pos = (start_x, start_y - gap, start_z)

        duct1 = RoundDuct(RoundDuctParams(
            diameter=chute_d, length=d1_len, wall_thickness=0.002,
            direction=(0.0, -1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct1, d1_pos))

        curr_y = start_y - gap - d1_len

        # Step 2: Elbow1 - turn from -Y to +X (toward classifier in X)
        # Elbow from -Y to +X: outlet at (inlet_x + R, inlet_y - R, inlet_z)
        e1_inlet = (start_x, curr_y - gap, start_z)
        e1_outlet_x = start_x + R
        e1_outlet_y = curr_y - gap - R

        elbow1 = DuctElbow(DuctElbowParams(
            diameter=chute_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, -1.0, 0.0), rotation_axis=(0.0, 0.0, -1.0),
        ))
        self._duct_connections.append((elbow1, e1_inlet))

        # Step 3: Angled descent in +X direction with gravity (-Z component)
        # Create an angled chute that descends while going toward +X
        # Calculate angle to drop from current Z to target Z while traveling in X
        
        # We need to go from current position to a point where we can make
        # the final -X approach to the venturi
        # Final approach point: (target_x + approach_dist, target_y, target_z)
        approach_dist = 0.25  # Distance to approach from +X side
        final_approach_x = target_x + approach_dist + R + gap  # Where elbow2 outlet will be

        # Calculate the angled descent vector
        # Account for the two elbows that will add 2R+gap to Y position:
        # elbow2 (+X→+Y): adds R to Y
        # elbow3 (+Y→-X): adds R to Y
        # So descent_end_y = target_y - 2*R - gap to end up at target_y after elbows
        descent_start = np.array([e1_outlet_x, e1_outlet_y, start_z])
        descent_end = np.array([final_approach_x, target_y - 2*R - gap, target_z])  # Account for elbow Y offsets
        
        descent_vec = descent_end - descent_start
        descent_dist = np.linalg.norm(descent_vec)
        
        if descent_dist > 0.1:
            descent_dir = descent_vec / descent_dist
            descent_dir_tuple = (float(descent_dir[0]), float(descent_dir[1]), float(descent_dir[2]))
            
            d2_pos = (e1_outlet_x + gap * descent_dir[0], 
                      e1_outlet_y + gap * descent_dir[1], 
                      start_z + gap * descent_dir[2])
            
            duct2 = RoundDuct(RoundDuctParams(
                diameter=chute_d, length=descent_dist - gap * 2,
                wall_thickness=0.002,
                direction=descent_dir_tuple, center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct2, d2_pos))
            
            # Current position after descent
            curr_pos = descent_end
        else:
            curr_pos = descent_start

        # Step 4: Elbow2 - turn to -X direction for final approach
        # Calculate the incoming direction from descent
        if descent_dist > 0.1:
            incoming_dir = descent_dir
        else:
            incoming_dir = np.array([1.0, 0.0, 0.0])
        
        # We need to turn toward -X
        # Determine rotation axis based on incoming direction
        # For simplicity, use a vertical elbow if incoming has significant Z component
        
        e2_inlet = (curr_pos[0] + gap, curr_pos[1], curr_pos[2])
        
        # Simple approach: use a horizontal elbow that turns to -X
        # Elbow from +X to -X via +Y or -Y
        # Actually, we need to consider the descent angle
        
        # For robustness, calculate proper elbow orientation
        # Incoming roughly +X with some -Z, want to turn to -X
        # This requires a 180-degree turn or two 90-degree turns
        # Use two elbows: first turn to +Y (or -Y), then to -X
        
        # Simplified: Turn from angled descent to -X
        # Elbow turns from current direction to -X
        
        elbow2 = DuctElbow(DuctElbowParams(
            diameter=chute_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(1.0, 0.0, 0.0),  # Approximate as +X incoming
            rotation_axis=(0.0, 0.0, -1.0),   # Turn in XY plane toward -X via +Y
        ))
        self._duct_connections.append((elbow2, e2_inlet))
        
        # Elbow2 outlet position: for +X to +Y turn, outlet at (inlet_x + R, inlet_y + R, inlet_z)
        e2_outlet_x = e2_inlet[0] + R
        e2_outlet_y = e2_inlet[1] + R
        e2_outlet_z = e2_inlet[2]

        # Step 5: Second elbow to turn from +Y to -X
        e3_inlet = (e2_outlet_x, e2_outlet_y + gap, e2_outlet_z)
        
        elbow3 = DuctElbow(DuctElbowParams(
            diameter=chute_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0),  # +Y incoming
            rotation_axis=(0.0, 0.0, 1.0),    # Turn to -X
        ))
        self._duct_connections.append((elbow3, e3_inlet))
        
        # Elbow3 outlet: +Y to -X turn, outlet at (inlet_x - R, inlet_y + R, inlet_z)
        e3_outlet_x = e3_inlet[0] - R
        e3_outlet_y = e3_inlet[1] + R
        e3_outlet_z = e3_inlet[2]

        # Step 6: Final approach in -X direction toward venturi
        # Go from e3_outlet to near venturi inlet
        trans_start_x = target_x + trans_len + gap
        d3_len = max(e3_outlet_x - gap - trans_start_x, 0.05)
        
        if d3_len > 0.05:
            d3_pos = (e3_outlet_x - gap, e3_outlet_y, e3_outlet_z)
            
            duct3 = RoundDuct(RoundDuctParams(
                diameter=chute_d, length=d3_len, wall_thickness=0.002,
                direction=(-1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct3, d3_pos))
            trans_pos_x = e3_outlet_x - gap - d3_len - gap
        else:
            trans_pos_x = e3_outlet_x - gap

        # Step 7: Transition to venturi diameter
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(chute_d,),
            outlet_dimensions=(venturi_solids_d,),
            length=trans_len,
            concentric=True,
            wall_thickness=0.002,
            direction=(-1.0, 0.0, 0.0),  # Approaching in -X direction
            center=(0, 0, 0),
            flanged=True,
        ))
        # Position at the same Y and Z as the ductwork outlet for continuity
        self._duct_connections.append((trans, (trans_pos_x, e3_outlet_y, e3_outlet_z)))

    def _build_bagfilter_to_exhaust_connection(self, classification, class_offset: np.ndarray):
        """
        Build ductwork from Bag Filter clean_air_outlet to Exhaust system (Silencer).

        Uses target-aligned routing: works backwards from silencer inlet to ensure
        the duct arrives at the exact target position.

        Route: Bag filter outlet (+Y) -> elbow to +Z -> rise up
               -> elbow to -Y -> horizontal toward silencer Y (aligned)
               -> elbow to -X -> horizontal toward silencer X (aligned)
               -> elbow to +Z -> rise to silencer -> transition

        Positions:
        - Start: bag filter outlet at ~(3.76, 5.10, 0.5), direction +Y
        - End: silencer inlet at ~(2.5, 0, 3.0), silencer pointing +Z
        """
        from ..components.ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams
        from ..components.transitions import Transition, TransitionParams

        p = self.params
        gap = 0.005  # 5mm flange gap

        # Get bag filter clean air outlet position and diameter
        bag_filter = classification.bag_filter
        bag_filter_pos = np.array(classification._component_positions['bag_filter']) + class_offset

        clean_air_port = bag_filter.ports['clean_air_outlet']
        bag_outlet_world = bag_filter_pos + np.array(clean_air_port.position)
        bag_outlet_d = clean_air_port.diameter

        # Get silencer inlet position
        silencer = self._components.get('silencer')
        if silencer is None:
            return

        silencer_dir = np.array(silencer.params.direction_normalized)
        silencer_center = np.array(silencer.params.center)
        silencer_inlet_world = silencer_center - silencer_dir * (silencer.params.length / 2)
        silencer_inlet_d = silencer.params.diameter

        # Duct dimensions
        duct_d = min(bag_outlet_d, silencer_inlet_d, p.main_duct_diameter)
        R = duct_d * 1.5  # Elbow bend radius
        trans_len = 0.12  # Transition length

        # Reference coordinates
        start_x, start_y, start_z = bag_outlet_world
        target_x, target_y, target_z = silencer_inlet_world

        # ============================================================
        # Work backwards from target to calculate key waypoints
        # This ensures the duct ends exactly at the silencer inlet
        # ============================================================

        # Final approach is vertical (+Z) into silencer
        # Before that, horizontal in -X to reach target_x
        # Before that, horizontal in -Y to reach target_y
        # First, go up (+Z) then turn to -Y

        # Position of last elbow (elbow4) outlet must be at (target_x, target_y, ...)
        # Elbow4 turns from -X to +Z, outlet at (inlet_x - R, inlet_y, inlet_z + R)
        # So elbow4 inlet_x = target_x + R

        # Position of elbow3 outlet determines the Y-level for elbow4
        # Elbow3 turns from -Y to -X, outlet at (inlet_x - R, inlet_y - R, inlet_z)
        # We want elbow3 outlet_y = target_y, so elbow3 inlet_y = target_y + R

        # The Z-level for horizontal routing
        horizontal_z = target_z - R - trans_len - gap  # Account for final rise

        # Step 1: Short duct from bag filter outlet in +Y direction
        d1_len = 0.15
        d1_start_y = start_y + gap
        d1_end_y = d1_start_y + d1_len

        duct1 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d1_len, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct1, (start_x, d1_start_y, start_z)))

        # Step 2: Elbow1 - turn from +Y to +Z (up)
        e1_inlet_y = d1_end_y + gap
        e1_outlet_y = e1_inlet_y + R
        e1_outlet_z = start_z + R

        elbow1 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0), rotation_axis=(-1.0, 0.0, 0.0),
        ))
        self._duct_connections.append((elbow1, (start_x, e1_inlet_y, start_z)))

        # Step 3: Vertical duct up in +Z direction to horizontal routing level
        d2_start_z = e1_outlet_z + gap
        e2_inlet_z = horizontal_z - R - gap  # Need room for elbow2 which adds R to Z
        d2_len = e2_inlet_z - d2_start_z
        d2_len = max(d2_len, 0.1)

        duct2 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d2_len, wall_thickness=0.002,
            direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct2, (start_x, e1_outlet_y, d2_start_z)))

        # Step 4: Elbow2 - turn from +Z to -Y
        # Elbow2: inlet at (x, y, z), outlet at (x, y - R, z + R)
        # For +Z to -Y turn: rotation about -X axis
        e2_inlet_z_actual = d2_start_z + d2_len + gap
        e2_outlet_y = e1_outlet_y - R
        e2_outlet_z = e2_inlet_z_actual + R

        elbow2 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 0.0, 1.0), rotation_axis=(-1.0, 0.0, 0.0),  # -X for +Z→-Y turn
        ))
        self._duct_connections.append((elbow2, (start_x, e1_outlet_y, e2_inlet_z_actual)))

        # Step 5: Horizontal duct in -Y direction
        # End at Y position where elbow3 inlet should be: target_y + R (working backwards)
        d3_start_y = e2_outlet_y - gap
        e3_inlet_y = target_y + R + gap  # Elbow3 turns -Y→-X, outlet_y = inlet_y - R = target_y
        d3_len = d3_start_y - e3_inlet_y
        d3_len = max(d3_len, 0.1)

        duct3 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d3_len, wall_thickness=0.002,
            direction=(0.0, -1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct3, (start_x, d3_start_y, e2_outlet_z)))

        # Step 6: Elbow3 - turn from -Y to -X
        # Elbow3: inlet at (x, y, z), outlet at (x - R, y - R, z)
        e3_inlet_y_actual = d3_start_y - d3_len - gap
        e3_outlet_x = start_x - R
        e3_outlet_y = e3_inlet_y_actual - R

        elbow3 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, -1.0, 0.0), rotation_axis=(0.0, 0.0, 1.0),  # +Z for -Y→-X
        ))
        self._duct_connections.append((elbow3, (start_x, e3_inlet_y_actual, e2_outlet_z)))

        # Step 7: Horizontal duct in -X direction
        # End at X position where elbow4 inlet should be: target_x + R (working backwards)
        d4_start_x = e3_outlet_x - gap
        e4_inlet_x = target_x + R + gap  # Elbow4 turns -X→+Z, outlet_x = inlet_x - R = target_x
        d4_len = d4_start_x - e4_inlet_x
        d4_len = max(d4_len, 0.1)

        duct4 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d4_len, wall_thickness=0.002,
            direction=(-1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct4, (d4_start_x, e3_outlet_y, e2_outlet_z)))

        # Step 8: Elbow4 - turn from -X to +Z (up toward silencer)
        # Elbow4: inlet at (x, y, z), outlet at (x - R, y, z + R)
        e4_inlet_x_actual = d4_start_x - d4_len - gap
        e4_outlet_x = e4_inlet_x_actual - R
        e4_outlet_z = e2_outlet_z + R

        elbow4 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(-1.0, 0.0, 0.0), rotation_axis=(0.0, -1.0, 0.0),  # -Y for -X→+Z
        ))
        self._duct_connections.append((elbow4, (e4_inlet_x_actual, e3_outlet_y, e2_outlet_z)))

        # Step 9: Final vertical duct up to transition
        d5_start_z = e4_outlet_z + gap
        trans_start_z = target_z - trans_len - gap
        d5_len = trans_start_z - d5_start_z - gap
        d5_len = max(d5_len, 0.05)

        if d5_len > 0.05:
            duct5 = RoundDuct(RoundDuctParams(
                diameter=duct_d, length=d5_len, wall_thickness=0.002,
                direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct5, (e4_outlet_x, e3_outlet_y, d5_start_z)))
            trans_pos_z = d5_start_z + d5_len + gap
        else:
            trans_pos_z = d5_start_z

        # Step 10: Transition to silencer diameter - positioned at silencer X, Y
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(duct_d,), outlet_dimensions=(silencer_inlet_d,),
            length=trans_len, concentric=True, wall_thickness=0.002,
            direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
        ))
        # Position transition at target X, target Y (silencer centerline)
        self._duct_connections.append((trans, (target_x, target_y, trans_pos_z)))

    def _build_exhaust_system(self):
        """Build exhaust silencer and stack with proper connections."""
        from ..components.silencer import create_absorptive_silencer
        from ..components.exhaust_stack import create_standard_exhaust_stack
        from ..components.ductwork import RoundDuct, RoundDuctParams
        from ..components.transitions import Transition, TransitionParams
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        # Silencer position (above frame, offset to side)
        # Silencer is vertical, center is at mid-height
        silencer_x = cx + p.frame_width / 2 + 0.5
        silencer_length = 1.0
        silencer_d = p.main_duct_diameter * 1.2
        silencer_z = cz + p.frame_height  # Center Z position
        
        silencer = create_absorptive_silencer(
            diameter=silencer_d,
            length=silencer_length,
            center=(silencer_x, cy, silencer_z),
            direction=(0, 0, 1)
        )
        self._components['silencer'] = silencer
        
        # Store silencer positions for duct connections
        self._silencer_inlet_z = silencer_z - silencer_length / 2  # Bottom of silencer
        self._silencer_outlet_z = silencer_z + silencer_length / 2  # Top of silencer
        self._silencer_x = silencer_x
        self._silencer_y = cy
        self._silencer_d = silencer_d
        
        # Stack on top of silencer - connect directly to silencer outlet
        # Stack base is at silencer outlet position
        stack_d = silencer_d  # Same diameter for direct connection
        stack_z = self._silencer_outlet_z  # Stack starts at silencer top
        
        stack = create_standard_exhaust_stack(
            diameter=stack_d,
            height=p.stack_height,
            center=(silencer_x, cy, stack_z)
        )
        self._components['exhaust_stack'] = stack
    
    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh from all subsystems and components.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        def add_component_mesh(component, position):
            """Add component mesh with position offset using generate_mesh()."""
            nonlocal vertex_offset

            # Use generate_mesh() which returns (vertices, indices, normals)
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add subsystem meshes with offsets
        for name, subsystem in self._subsystems.items():
            if name.endswith('_offset'):
                continue

            verts, idx = subsystem.build_mesh()

            # Apply position offset if available
            offset_key = f'{name}_offset'
            if offset_key in self._subsystems:
                offset = np.array(self._subsystems[offset_key])
                verts = verts + offset

            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add individual components (silencer, stack)
        for name, component in self._components.items():
            # Components have generate_mesh() method
            verts, idx, _ = component.generate_mesh()

            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add duct connections (connecting ductwork between systems)
        # Format: (component, world_position) - same as classification.py
        for duct, position in self._duct_connections:
            add_component_mesh(duct, position)

        if all_vertices:
            self._vertices = np.vstack(all_vertices).astype(np.float32)
            self._indices = np.concatenate(all_indices).astype(np.int32)
        else:
            self._vertices = np.array([], dtype=np.float32).reshape(0, 3)
            self._indices = np.array([], dtype=np.int32)

        return self._vertices, self._indices
    
    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if self._vertices is None:
            self.build_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if self._indices is None:
            self.build_mesh()
        return self._indices
    
    def get_subsystem(self, name: str) -> Any:
        """Get a subsystem by name."""
        return self._subsystems.get(name)
    
    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        return self._components.get(name)
    
    def get_all_subsystem_names(self) -> List[str]:
        """Get names of all subsystems."""
        return [k for k in self._subsystems.keys() if not k.endswith('_offset')]
    
    def get_all_component_names(self) -> List[str]:
        """Get names of all components."""
        return list(self._components.keys())
    
    def get_all_duct_types(self) -> List[str]:
        """Get types of all duct connections."""
        if hasattr(self, '_duct_connections') and self._duct_connections:
            return [type(duct).__name__ for duct, _ in self._duct_connections]
        return []

    def get_duct_count(self) -> int:
        """Get number of duct connection components."""
        if hasattr(self, '_duct_connections'):
            return len(self._duct_connections)
        return 0

    def get_air_to_venturi_ductwork(self) -> List[Tuple[Any, Tuple[float, float, float]]]:
        """
        Get ductwork from Air System outlet to Venturi air_inlet only.

        Returns list of (component, world_position) in flow order:
        duct1 -> elbow1 -> duct2 -> elbow2 -> duct3 -> elbow3 -> duct4 -> elbow4 -> duct5? -> transition.
        Empty if ductwork was not built (e.g. air system or ductwork disabled).
        """
        if hasattr(self, '_air_to_venturi_ducts'):
            return list(self._air_to_venturi_ducts)
        return []

    def get_feed_to_venturi_ductwork(self) -> List[Tuple[Any, Tuple[float, float, float]]]:
        """
        Get ductwork from Feed System (deagglomerator) outlet to Venturi solids_inlet only.

        Returns list of (component, world_position) in flow order:
        stub_duct -> drop_duct -> elbow1 -> shaft_duct? -> transition.
        Empty if ductwork was not built (e.g. feed system or ductwork disabled).
        """
        if hasattr(self, '_feed_to_venturi_ducts'):
            return list(self._feed_to_venturi_ducts)
        return []

    def get_feed_to_venturi_chute_angle_deg(self) -> Optional[float]:
        """
        Chute angle from horizontal [degrees] for the feed-to-venturi shaft segment.
        Set from geometry when feed-to-venturi ductwork is built.
        """
        return getattr(self, '_feed_chute_angle_deg', None)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of entire system."""
        verts = self.vertices
        if len(verts) == 0:
            return np.zeros(3), np.zeros(3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the complete system."""
        bounds_min, bounds_max = self.get_bounds()
        dimensions = bounds_max - bounds_min

        return {
            'design_throughput_kg_h': self.params.throughput_kg_h,
            'design_cut_size_um': self.params.cut_size_um,
            'design_air_flow_m3_h': self.params.air_flow_m3_h,
            'num_subsystems': len(self.get_all_subsystem_names()),
            'num_components': len(self._components),
            'num_duct_connections': len(self._duct_connections),
            'total_vertices': len(self.vertices),
            'total_triangles': len(self.indices) // 3,
            'dimensions_m': list(dimensions),
            'bounds_min': list(bounds_min),
            'bounds_max': list(bounds_max),
            'includes': {
                'feed_system': self.params.include_feed_system,
                'air_system': self.params.include_air_system,
                'ductwork': self.params.include_ductwork,
                'support_structure': self.params.include_support_structure,
                'exhaust': self.params.include_exhaust,
            }
        }
    
    def get_bill_of_materials(self) -> List[Dict[str, Any]]:
        """
        Generate a bill of materials for the system.

        Returns:
            List of component entries with name, type, and quantity
        """
        bom = []

        # Subsystems
        subsystem_types = {
            'classification': 'Classification System',
            'feed_system': 'Feed System',
            'air_system': 'Air System',
            'support_structure': 'Support Structure',
        }

        for name, desc in subsystem_types.items():
            if name in self._subsystems:
                bom.append({
                    'item': desc,
                    'type': 'Subsystem',
                    'quantity': 1
                })

        # Individual components (silencer, stack, etc.)
        for name in self._components.keys():
            bom.append({
                'item': name.replace('_', ' ').title(),
                'type': 'Component',
                'quantity': 1
            })

        # Duct connections
        duct_counts = {}
        for duct, _ in self._duct_connections:
            duct_type = type(duct).__name__
            duct_counts[duct_type] = duct_counts.get(duct_type, 0) + 1

        for item, qty in duct_counts.items():
            bom.append({
                'item': item,
                'type': 'Ductwork',
                'quantity': qty
            })

        return bom
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )
    
    def print_summary(self):
        """Print a comprehensive summary of the complete system."""
        summary = self.get_system_summary()

        print("=" * 70)
        print("COMPLETE AIR CLASSIFIER SYSTEM")
        print("=" * 70)
        print(f"Design Throughput: {summary['design_throughput_kg_h']:.0f} kg/h")
        print(f"Target Cut Size: {summary['design_cut_size_um']:.0f} um")
        print(f"Air Flow Rate: {summary['design_air_flow_m3_h']:.0f} m3/h")
        print("-" * 70)
        print("SYSTEM COMPOSITION:")
        print(f"  Subsystems: {summary['num_subsystems']}")
        print(f"  Components: {summary['num_components']}")
        print(f"  Duct Connections: {summary['num_duct_connections']}")
        print("-" * 70)
        print("INCLUDED SYSTEMS:")
        for system, included in summary['includes'].items():
            status = "[x]" if included else "[ ]"
            print(f"  {status} {system.replace('_', ' ').title()}")
        print("-" * 70)
        print("CONNECTING DUCTWORK:")
        if self._duct_connections:
            for i, (duct, position) in enumerate(self._duct_connections):
                duct_type = type(duct).__name__
                if hasattr(duct, 'params'):
                    if hasattr(duct.params, 'diameter'):
                        dim = f"D={duct.params.diameter*1000:.0f}mm"
                    else:
                        dim = ""
                    if hasattr(duct.params, 'length'):
                        length = f"L={duct.params.length*1000:.0f}mm"
                    elif hasattr(duct.params, 'bend_radius'):
                        length = f"R={duct.params.bend_radius*1000:.0f}mm"
                    else:
                        length = ""
                    print(f"  [{i+1}] {duct_type:20s} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) {length} {dim}")
                else:
                    print(f"  [{i+1}] {duct_type} at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        else:
            print("  (none)")
        print("-" * 70)
        print("GEOMETRY:")
        print(f"  Dimensions: {summary['dimensions_m'][0]:.1f} x {summary['dimensions_m'][1]:.1f} x {summary['dimensions_m'][2]:.1f} m")
        print(f"  Total Vertices: {summary['total_vertices']:,}")
        print(f"  Total Triangles: {summary['total_triangles']:,}")
        print("=" * 70)
    
    def print_bill_of_materials(self):
        """Print the bill of materials."""
        bom = self.get_bill_of_materials()
        
        print("=" * 50)
        print("BILL OF MATERIALS")
        print("=" * 50)
        print(f"{'Item':<30} {'Type':<15} {'Qty':>5}")
        print("-" * 50)
        
        for entry in bom:
            print(f"{entry['item']:<30} {entry['type']:<15} {entry['quantity']:>5}")
        
        print("-" * 50)
        print(f"{'TOTAL ITEMS':<30} {'':<15} {len(bom):>5}")
        print("=" * 50)


# Factory functions

def create_complete_classifier_system(throughput_kg_h: float = 500,
                                      cut_size_um: float = 20,
                                      **kwargs) -> CompleteClassifierAssembly:
    """
    Create a complete air classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        cut_size_um: Target cut size [μm]
        **kwargs: Additional parameters
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        cut_size_um=cut_size_um,
        **kwargs
    )
    return CompleteClassifierAssembly(params)


def create_pilot_scale_system(throughput_kg_h: float = 100) -> CompleteClassifierAssembly:
    """
    Create a pilot-scale classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        air_flow_m3_h=1000,
        classifier_width=0.1,
        cyclone_diameter=0.2,
        hopper_diameter=0.4,
        main_duct_diameter=0.15,
        frame_width=2.5,
        frame_depth=2.0,
        frame_height=2.5,
        stack_height=3.0,
    )
    return CompleteClassifierAssembly(params)


def create_production_scale_system(throughput_kg_h: float = 2000) -> CompleteClassifierAssembly:
    """
    Create a production-scale classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        air_flow_m3_h=8000,
        classifier_width=0.25,
        cyclone_diameter=0.5,
        hopper_diameter=1.0,
        main_duct_diameter=0.35,
        frame_width=6.0,
        frame_depth=4.0,
        frame_height=5.0,
        stack_height=8.0,
    )
    return CompleteClassifierAssembly(params)


def create_minimal_classifier_system() -> CompleteClassifierAssembly:
    """
    Create a minimal classifier with only the classification core.

    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=200,
        include_feed_system=False,
        include_air_system=False,
        include_ductwork=False,
        include_support_structure=False,
        include_exhaust=False,
    )
    return CompleteClassifierAssembly(params)


def create_core_connections_system() -> CompleteClassifierAssembly:
    """
    Create a system focused on the three main duct connections.

    Optimized for protein separation with steep gravity-fed chute (~20°).
    Support structure is excluded to focus on core flow components.

    LAYOUT (top view, Y is depth, X is width):

                        +Y (back)
                          |
                          |   +-------------+
                          |   | FEED SYSTEM |
                          |   | (hopper)    |
                          |   +-------------+
                          |   at (0, +1, 3.5)
                          |         |
                          |         | gravity chute
                          |         v
                          |   +-------------+
                          |   |  VENTURI    |  Classification
                          |   |  (origin)   |  System at (0,0,0)
                          |   +-------------+
                          |         ^
                          |         | air duct
                          |   +-------------+
                          |   | AIR SYSTEM  |
                          |   | (blower)    |
                          |   +-------------+
                          |   at (0, -3, 0)
                       ---+--------------------> +X (right)
                          |
                        -Y (front)

    System Parameters:
    - Throughput: 500 kg/h
    - Air flow: 3000 m³/h
    - Classifier width: 0.15 m
    - Main duct diameter: 0.2 m

    Connections:
    1. Air System outlet -> Venturi air_inlet (vertical from front)
    2. Feed System outlet -> Venturi solids_inlet (gravity chute from above/behind)
    3. Bag Filter clean_air_outlet -> Exhaust silencer

    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=500,
        air_flow_m3_h=3000,
        # Core systems - all enabled
        include_feed_system=True,
        include_air_system=True,
        include_ductwork=True,
        include_exhaust=True,
        include_support_structure=False,
        # Positions optimized for protein separation
        # Feed behind classifier (+Y), elevated for steep ~20° gravity chute
        classifier_position=(0.0, 0.0, 0.0),
        feed_position=(0.0, 1.0, 3.5), # +Y behind, +X right, elevated
        air_system_position=(0.0, -3.0, 0.0),
        # Sizing
        classifier_width=0.15,
        cyclone_diameter=0.3,
        hopper_diameter=0.6,
        main_duct_diameter=0.2,
        frame_width=4.0,
        frame_depth=3.0,
        frame_height=3.5,
        stack_height=4.0,
    )
    return CompleteClassifierAssembly(params)