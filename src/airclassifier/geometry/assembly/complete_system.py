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

from dataclasses import dataclass
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

    # Layout positions
    feed_position: Tuple[float, float, float] = (-4.0, -1.0, 0.0)
    classifier_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    air_system_position: Tuple[float, float, float] = (-1.0, -5.0, 0.0)

    # Include flags
    include_feed_system: bool = True
    include_air_system: bool = True
    include_ductwork: bool = True
    include_support_structure: bool = False
    include_exhaust: bool = True

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

        # Build support structure first (defines the frame of reference)
        if p.include_support_structure:
            self._build_support_structure()

        # Build classification system (core)
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
    
    def _build_support_structure(self):
        """Build support frame and legs."""
        from .support_exhaust import create_standard_support_exhaust
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        support = create_standard_support_exhaust(
            frame_width=p.frame_width,
            frame_depth=p.frame_depth,
            frame_height=p.frame_height,
            has_exhaust_stack=False,  # We'll add this separately
            has_silencer=False,
            center=(cx, cy, cz)
        )
        self._subsystems['support_structure'] = support
    
    def _build_classification_system(self):
        """Build the classification system (zigzag + cyclones + bag filter)."""
        from .classification import create_standard_classification_system
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        # Position classification on top of support frame
        if p.include_support_structure:
            class_z = cz + p.frame_height + 0.5  # 0.5m above frame
        else:
            class_z = cz + 0.5
        
        classification = create_standard_classification_system(device="cpu")
        self._subsystems['classification'] = classification
        
        # Store position offset for mesh transformation
        self._subsystems['classification_offset'] = (cx, cy, class_z)
    
    def _build_feed_system(self):
        """Build the feed system (hopper + airlock + screw + deagglomerator)."""
        from .feed_system import create_standard_feed_system
        
        p = self.params
        fx, fy, fz = p.feed_position
        
        # Position feed system elevated
        if p.include_support_structure:
            feed_z = fz + p.frame_height + 1.0
        else:
            feed_z = fz + 1.0
        
        feed = create_standard_feed_system(device="cpu")
        self._subsystems['feed_system'] = feed
        self._subsystems['feed_system_offset'] = (fx, fy, feed_z)
    
    def _build_air_system(self):
        """Build the air system (blower + filter + damper)."""
        from .air_system import create_standard_air_system
        
        p = self.params
        ax, ay, az = p.air_system_position
        
        air = create_standard_air_system(device="cpu")
        self._subsystems['air_system'] = air
        self._subsystems['air_system_offset'] = (ax, ay, az + 0.5)
    
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

        # 1. Air System -> Venturi air inlet (primary air supply)
        if p.include_air_system and 'air_system' in self._subsystems:
            self._build_air_to_venturi_connection(venturi, venturi_pos)

        # 2. Feed System -> Venturi solids inlet (powder feed)
        if p.include_feed_system and 'feed_system' in self._subsystems:
            self._build_feed_to_venturi_connection(venturi, venturi_pos)

        # 3. Bag Filter -> Exhaust silencer (clean air exhaust)
        if p.include_exhaust and 'silencer' in self._components:
            self._build_bagfilter_to_exhaust_connection(classification, class_offset)
    
    def _build_air_to_venturi_connection(self, venturi, venturi_pos: np.ndarray):
        """
        Build ductwork from Air System outlet to Venturi air_inlet.

        Uses explicit world coordinates for each component (like multi_cyclone.py).

        Route: Damper outlet (+X) -> elbow to +Y -> horizontal toward classifier
               -> elbow to -X -> shift left -> elbow to -Z -> drop down
               -> elbow to +Y -> approach venturi -> transition

        Positions:
        - Start: damper outlet at ~(1.01, -5.0, 0.88), direction +X
        - End: venturi air inlet at ~(0, 0, 0.5), expects flow from -Y direction
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
        R = duct_d * 1.5  # Elbow bend radius

        # Reference coordinates
        start_x, start_y, start_z = air_outlet_world
        target_x, target_y, target_z = venturi_air_world

        # ============================================================
        # Calculate path geometry using explicit coordinates
        # Following multi_cyclone.py pattern
        # ============================================================

        # Step 1: Short duct from damper outlet in +X direction
        d1_start_x = start_x + gap
        d1_len = 0.15
        d1_end_x = d1_start_x + d1_len

        duct1 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d1_len, wall_thickness=0.002,
            direction=(1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct1, (d1_start_x, start_y, start_z)))

        # Step 2: Elbow1 - turn from +X to +Y
        # Inlet at d1_end_x, outlet at (inlet_x + R, inlet_y + R, inlet_z)
        e1_inlet_x = d1_end_x + gap
        e1_outlet_x = e1_inlet_x + R
        e1_outlet_y = start_y + R

        elbow1 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(1.0, 0.0, 0.0), rotation_axis=(0.0, 0.0, -1.0),
        ))
        self._duct_connections.append((elbow1, (e1_inlet_x, start_y, start_z)))

        # Step 3: Long horizontal duct in +Y direction
        # From elbow1 outlet toward target, leaving 1.5m for the remaining path
        d2_start_y = e1_outlet_y + gap
        d2_end_y = target_y - 1.5  # Leave room for final approach
        d2_len = d2_end_y - d2_start_y
        d2_len = max(d2_len, 0.3)

        duct2 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d2_len, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct2, (e1_outlet_x, d2_start_y, start_z)))

        # Step 4: Elbow2 - turn from +Y to -X
        # For +Y to -X: inlet at (x, d2_start_y + d2_len), outlet at (x - R, inlet_y + R)
        e2_inlet_y = d2_start_y + d2_len + gap
        e2_outlet_x = e1_outlet_x - R
        e2_outlet_y = e2_inlet_y + R

        elbow2 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0), rotation_axis=(0.0, 0.0, 1.0),
        ))
        self._duct_connections.append((elbow2, (e1_outlet_x, e2_inlet_y, start_z)))

        # Step 5: Horizontal duct in -X direction toward target X
        d3_start_x = e2_outlet_x - gap
        d3_end_x = target_x + R + 0.1  # Leave room for elbow and final approach
        d3_len = d3_start_x - d3_end_x
        d3_len = max(d3_len, 0.1)

        duct3 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d3_len, wall_thickness=0.002,
            direction=(-1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct3, (d3_start_x, e2_outlet_y, start_z)))

        # Step 6: Elbow3 - turn from -X to -Z (down)
        # For -X to -Z: inlet at (d3_end_x, y, z), outlet at (inlet_x - R, y, z - R)
        e3_inlet_x = d3_start_x - d3_len - gap
        e3_outlet_x = e3_inlet_x - R
        e3_outlet_z = start_z - R

        elbow3 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(-1.0, 0.0, 0.0), rotation_axis=(0.0, 1.0, 0.0),
        ))
        self._duct_connections.append((elbow3, (e3_inlet_x, e2_outlet_y, start_z)))

        # Step 7: Vertical duct down in -Z direction
        d4_start_z = e3_outlet_z - gap
        d4_end_z = target_z + R + 0.15  # Leave room for elbow and transition
        d4_len = d4_start_z - d4_end_z
        d4_len = max(d4_len, 0.1)

        duct4 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d4_len, wall_thickness=0.002,
            direction=(0.0, 0.0, -1.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct4, (e3_outlet_x, e2_outlet_y, d4_start_z)))

        # Step 8: Elbow4 - turn from -Z to +Y (toward venturi)
        # For -Z to +Y: inlet at (x, y, d4_end_z), outlet at (x, y + R, z - R)
        e4_inlet_z = d4_start_z - d4_len - gap
        e4_outlet_y = e2_outlet_y + R
        e4_outlet_z = e4_inlet_z - R

        elbow4 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 0.0, -1.0), rotation_axis=(-1.0, 0.0, 0.0),
        ))
        self._duct_connections.append((elbow4, (e3_outlet_x, e2_outlet_y, e4_inlet_z)))

        # Step 9: Final approach duct in +Y direction
        d5_start_y = e4_outlet_y + gap
        d5_len = target_y - d5_start_y - 0.15  # Leave room for transition
        d5_len = max(d5_len, 0.05)

        if d5_len > 0.05:
            duct5 = RoundDuct(RoundDuctParams(
                diameter=duct_d, length=d5_len, wall_thickness=0.002,
                direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct5, (e3_outlet_x, d5_start_y, e4_outlet_z)))
            trans_start_y = d5_start_y + d5_len + gap
        else:
            trans_start_y = d5_start_y

        # Step 10: Transition to venturi diameter
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(duct_d,), outlet_dimensions=(venturi_air_d,),
            length=0.12, concentric=True, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((trans, (e3_outlet_x, trans_start_y, e4_outlet_z)))

    def _build_feed_to_venturi_connection(self, venturi, venturi_pos: np.ndarray):
        """
        Build gravity chute from Feed System outlet to Venturi solids_inlet.

        Route: Deagglomerator outlet -> Angled gravity chute -> Venturi solids_inlet

        Feed outlet is at (-3.84, -1.97, 1.0) pointing -Y, diameter 96mm
        Venturi solids inlet is at (0.02, 0.11, 0.5) expecting from +X, diameter 32mm

        We create a direct angled chute that follows the gravity vector while
        connecting the two ports. The chute diameter reduces progressively.
        """
        from ..components.ductwork import RoundDuct, RoundDuctParams
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

        # Venturi solids inlet position (in world coordinates)
        venturi_solids_port = venturi.ports['solids_inlet']
        venturi_solids_world = venturi_pos + np.array(venturi_solids_port.position)
        venturi_solids_d = venturi_solids_port.diameter

        # Calculate the direction vector from feed outlet to venturi inlet
        delta = venturi_solids_world - feed_outlet_world
        total_distance = np.linalg.norm(delta)

        if total_distance < 0.1:
            return  # Too close, no duct needed

        # Direction from feed to venturi
        direction = delta / total_distance
        direction_tuple = (float(direction[0]), float(direction[1]), float(direction[2]))

        # Use appropriate chute diameter (larger for powder flow, reducing toward venturi)
        chute_d = max(feed_outlet_d * 0.8, venturi_solids_d * 2.0, 0.06)

        # Calculate segment lengths
        trans1_length = 0.12  # Entry transition
        trans2_length = 0.10  # Exit transition
        main_chute_length = total_distance - trans1_length - trans2_length - gap * 6

        if main_chute_length < 0.1:
            # Not enough space for full assembly, use single transition
            single_trans = Transition(TransitionParams(
                transition_type="round_to_round",
                inlet_dimensions=(feed_outlet_d,),
                outlet_dimensions=(venturi_solids_d,),
                length=total_distance - gap * 2,
                concentric=True,
                wall_thickness=0.002,
                direction=direction_tuple,
                center=(0, 0, 0),
                flanged=True,
            ))
            trans_start = tuple(feed_outlet_world + direction * gap)
            self._duct_connections.append((single_trans, trans_start))
            return

        # Position tracker along chute direction
        pos = np.array(feed_outlet_world, dtype=float) + direction * gap

        # Step 1: Entry transition (feed_outlet_d -> chute_d)
        trans1 = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(feed_outlet_d,),
            outlet_dimensions=(chute_d,),
            length=trans1_length,
            concentric=True,
            wall_thickness=0.002,
            direction=direction_tuple,
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_connections.append((trans1, tuple(pos)))
        pos = pos + direction * (trans1_length + gap)

        # Step 2: Main angled chute
        chute = RoundDuct(RoundDuctParams(
            diameter=chute_d,
            length=main_chute_length,
            wall_thickness=0.002,
            direction=direction_tuple,
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_connections.append((chute, tuple(pos)))
        pos = pos + direction * (main_chute_length + gap)

        # Step 3: Exit transition (chute_d -> venturi_solids_d)
        trans2 = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(chute_d,),
            outlet_dimensions=(venturi_solids_d,),
            length=trans2_length,
            concentric=True,
            wall_thickness=0.002,
            direction=direction_tuple,
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_connections.append((trans2, tuple(pos)))

    def _build_bagfilter_to_exhaust_connection(self, classification, class_offset: np.ndarray):
        """
        Build ductwork from Bag Filter clean_air_outlet to Exhaust system (Silencer).

        Uses explicit world coordinates for each component (like multi_cyclone.py).

        Route: Bag filter outlet (+Y) -> elbow to +Z -> rise up
               -> elbow to -Y -> horizontal toward silencer Y
               -> elbow to -X -> horizontal toward silencer X
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

        # Reference coordinates
        start_x, start_y, start_z = bag_outlet_world
        target_x, target_y, target_z = silencer_inlet_world

        # ============================================================
        # Calculate path geometry using explicit coordinates
        # Following multi_cyclone.py pattern
        # ============================================================

        # Step 1: Short duct from bag filter outlet in +Y direction
        d1_start_y = start_y + gap
        d1_len = 0.2
        d1_end_y = d1_start_y + d1_len

        duct1 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d1_len, wall_thickness=0.002,
            direction=(0.0, 1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct1, (start_x, d1_start_y, start_z)))

        # Step 2: Elbow1 - turn from +Y to +Z (up)
        # For +Y to +Z: inlet at (x, d1_end_y, z), outlet at (x, inlet_y + R, z + R)
        e1_inlet_y = d1_end_y + gap
        e1_outlet_y = e1_inlet_y + R
        e1_outlet_z = start_z + R

        elbow1 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 1.0, 0.0), rotation_axis=(-1.0, 0.0, 0.0),
        ))
        self._duct_connections.append((elbow1, (start_x, e1_inlet_y, start_z)))

        # Step 3: Vertical duct up in +Z direction
        # Rise to a level below silencer where we can route horizontally
        horizontal_z = target_z - R - 0.3  # Leave room for final elbow
        d2_start_z = e1_outlet_z + gap
        d2_len = horizontal_z - d2_start_z
        d2_len = max(d2_len, 0.2)

        duct2 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d2_len, wall_thickness=0.002,
            direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct2, (start_x, e1_outlet_y, d2_start_z)))

        # Step 4: Elbow2 - turn from +Z to -Y
        # For +Z to -Y: inlet at (x, y, d2_start_z + d2_len), outlet at (x, y - R, z + R)
        e2_inlet_z = d2_start_z + d2_len + gap
        e2_outlet_y = e1_outlet_y - R
        e2_outlet_z = e2_inlet_z + R

        elbow2 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, 0.0, 1.0), rotation_axis=(1.0, 0.0, 0.0),
        ))
        self._duct_connections.append((elbow2, (start_x, e1_outlet_y, e2_inlet_z)))

        # Step 5: Horizontal duct in -Y direction toward silencer Y
        d3_start_y = e2_outlet_y - gap
        d3_end_y = target_y + R + 0.2  # Leave room for elbow
        d3_len = d3_start_y - d3_end_y
        d3_len = max(d3_len, 0.2)

        duct3 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d3_len, wall_thickness=0.002,
            direction=(0.0, -1.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct3, (start_x, d3_start_y, e2_outlet_z)))

        # Step 6: Elbow3 - turn from -Y to -X
        # For -Y to -X: inlet at (x, d3_end_y, z), outlet at (x - R, inlet_y - R, z)
        e3_inlet_y = d3_start_y - d3_len - gap
        e3_outlet_x = start_x - R
        e3_outlet_y = e3_inlet_y - R

        elbow3 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(0.0, -1.0, 0.0), rotation_axis=(0.0, 0.0, -1.0),
        ))
        self._duct_connections.append((elbow3, (start_x, e3_inlet_y, e2_outlet_z)))

        # Step 7: Horizontal duct in -X direction toward silencer X
        d4_start_x = e3_outlet_x - gap
        d4_end_x = target_x + R + 0.15  # Leave room for elbow
        d4_len = d4_start_x - d4_end_x
        d4_len = max(d4_len, 0.1)

        duct4 = RoundDuct(RoundDuctParams(
            diameter=duct_d, length=d4_len, wall_thickness=0.002,
            direction=(-1.0, 0.0, 0.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((duct4, (d4_start_x, e3_outlet_y, e2_outlet_z)))

        # Step 8: Elbow4 - turn from -X to +Z (up toward silencer)
        # For -X to +Z: inlet at (d4_end_x, y, z), outlet at (inlet_x - R, y, z + R)
        e4_inlet_x = d4_start_x - d4_len - gap
        e4_outlet_x = e4_inlet_x - R
        e4_outlet_z = e2_outlet_z + R

        elbow4 = DuctElbow(DuctElbowParams(
            diameter=duct_d, bend_radius=R, angle=90.0, wall_thickness=0.002,
            flanged=True, center=(0, 0, 0),
            inlet_direction=(-1.0, 0.0, 0.0), rotation_axis=(0.0, -1.0, 0.0),
        ))
        self._duct_connections.append((elbow4, (e4_inlet_x, e3_outlet_y, e2_outlet_z)))

        # Step 9: Final vertical duct up in +Z direction to silencer
        d5_start_z = e4_outlet_z + gap
        d5_len = target_z - d5_start_z - 0.15  # Leave room for transition
        d5_len = max(d5_len, 0.05)

        if d5_len > 0.05:
            duct5 = RoundDuct(RoundDuctParams(
                diameter=duct_d, length=d5_len, wall_thickness=0.002,
                direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
            ))
            self._duct_connections.append((duct5, (e4_outlet_x, e3_outlet_y, d5_start_z)))
            trans_start_z = d5_start_z + d5_len + gap
        else:
            trans_start_z = d5_start_z

        # Step 10: Transition to silencer diameter
        trans = Transition(TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(duct_d,), outlet_dimensions=(silencer_inlet_d,),
            length=0.12, concentric=True, wall_thickness=0.002,
            direction=(0.0, 0.0, 1.0), center=(0, 0, 0), flanged=True,
        ))
        self._duct_connections.append((trans, (e4_outlet_x, e3_outlet_y, trans_start_z)))

    def _build_exhaust_system(self):
        """Build exhaust silencer and stack."""
        from ..components.silencer import create_absorptive_silencer
        from ..components.exhaust_stack import create_standard_exhaust_stack
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        # Silencer position (above frame, offset to side)
        silencer_x = cx + p.frame_width / 2 + 0.5
        silencer_z = cz + p.frame_height
        
        silencer = create_absorptive_silencer(
            diameter=p.main_duct_diameter * 1.2,
            length=1.0,
            center=(silencer_x, cy, silencer_z),
            direction=(0, 0, 1)
        )
        self._components['silencer'] = silencer
        
        # Stack on top of silencer
        stack_z = silencer_z + 1.0
        stack = create_standard_exhaust_stack(
            diameter=p.main_duct_diameter * 1.2,
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

    LAYOUT (top view, Y is depth, X is width):

                        +Y (back)
                          |
                          |   +-------------+
                          |   | Bag Filter  |
                          |   | (part of    |
                          |   | classif.)   |
                          |   +-------------+
                          |
    +----------+          |   +-------------+
    | FEED     | ---------+-->|  VENTURI    |  Classification
    | SYSTEM   |  chute   |   |  (origin)   |  System at (0,0,0)
    +----------+          |   +-------------+
    at (-4, -1, Z)        |         ^
                          |         | air duct
                          |   +-----+-----+
                          |   | AIR SYSTEM |
                          |   | (blower)   |
                          |   +-----------+
                          |   at (-1, -5, Z)
                       ---+--------------------> +X (right)
                          |
                        -Y (front)

    Connections:
    1. Air System outlet -> Venturi air_inlet
    2. Feed System outlet -> Venturi solids_inlet
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
        # Positions for visible ductwork connections
        classifier_position=(0.0, 0.0, 0.0),
        feed_position=(-4.0, -1.0, 0.0),
        air_system_position=(-1.0, -5.0, 0.0),
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
