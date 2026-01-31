"""
Air System Assembly Module
==========================

This module provides the complete air supply system assembly for air classification.
It combines all Phase 3 components into an integrated air handling train.

SYSTEM OVERVIEW
===============

The air system provides clean, pressurized air to the classification process:
1. Inlet Air Filter - Removes ambient particulates
2. Centrifugal Blower - Provides motive force (pressure/flow)
3. Flow Dampers - Control air flow rate and isolation

MATERIAL FLOW PATH
==================

    AMBIENT AIR
         │
         ▼
    ┌─────────────┐
    │ INLET FILTER │  ← Removes dust/particles from ambient air
    │   (Panel)    │     G4 class: 80% efficiency on 5µm particles
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ DUCT SECTION │  ← Transition: Filter outlet → Blower inlet
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ CENTRIFUGAL │  ← Air enters axially, exits radially
    │   BLOWER    │     Provides pressure rise (5000 Pa typical)
    │  (Scroll)   │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ DUCT SECTION │  ← Transition: Blower outlet → Damper inlet
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   DAMPER    │  ← Flow control (butterfly valve)
    │  (Control)  │     Throttling for flow adjustment
    └──────┬──────┘
           │
           ▼
    TO CLASSIFIER
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import warp as wp

from ..connection_ports import (
    ConnectionPort, PortType, calculate_alignment,
    validate_assembly_connections, print_connection_report
)


@dataclass
class AirSystemParams:
    """
    Parameters for complete air system.

    Combines all Phase 3 components into an air supply system.
    """

    # System air requirements
    flow_rate_m3_h: float = 3000       # [m3/h] Design flow rate
    pressure_rise_Pa: float = 5000      # [Pa] Total system pressure drop

    # Blower parameters
    blower_blade_type: str = "backward_curved"

    # Inlet filter parameters
    filter_type: str = "panel"          # "panel", "bag", "cartridge", "HEPA"
    filter_efficiency_class: str = "G4"

    # Damper parameters
    num_control_dampers: int = 2        # Number of control dampers in system
    damper_type: str = "butterfly"      # "butterfly", "louver", "iris"

    # Layout parameters
    component_spacing: float = 0.3      # [m] Spacing between components
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class AirSystemAssembly:
    """
    Complete air supply system assembly.

    Combines all Phase 3 components:
    - Inlet Air Filter: Clean air supply
    - Centrifugal Blower: Air motive force
    - Flow Dampers: Flow control/isolation

    Process flow:
    Ambient Air -> Filter -> Blower -> Damper -> Classifier

    Coordinate system:
    - Origin at center of system
    - X-axis: Main flow direction
    - Y-axis: Vertical (up)
    - Z-axis: Depth
    """

    def __init__(self, params: AirSystemParams = None, device: str = "cpu"):
        """
        Initialize air system assembly.

        Args:
            params: AirSystemParams (uses defaults if None)
            device: Warp device for mesh operations
        """
        self.params = params or AirSystemParams()
        self.device = device

        # Create components
        self._create_components()

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._mesh_built = False

    def _create_components(self):
        """
        Create all system components with proper geometry-based positioning.

        Layout: Linear arrangement along X axis for clear visualization.

        Flow path:
        Filter → Duct → Elbow → Duct → Blower inlet bell → Blower → Transition → Dampers

        All components centered at Y=0, Z=0 for simplicity.

        Connections:
        - Filter outlet port at end of outlet stub (faces +X)
        - Blower inlet bell geometry at +Z side (center + impeller_width/2 * 1.2)
        - Duct connects to actual inlet bell opening, not port position
        """
        from ..components import (
            create_standard_centrifugal_blower,
            create_standard_inlet_filter,
            create_standard_damper,
        )

        p = self.params
        gap = 0.005  # 5mm gap between flanged connections (tight gasket space)

        # Calculate duct diameter from flow rate and velocity
        target_velocity = 15.0  # m/s typical for main duct
        duct_area = p.flow_rate_m3_h / 3600 / target_velocity
        self._duct_diameter = np.sqrt(4 * duct_area / np.pi)

        # ============================================================
        # 1. INLET AIR FILTER
        # ============================================================
        self.inlet_filter = create_standard_inlet_filter(
            flow_rate=p.flow_rate_m3_h,
            filter_type=p.filter_type,
            efficiency_class=p.filter_efficiency_class
        )
        # Position filter at system origin
        self._filter_position = (p.center[0], p.center[1], p.center[2])

        # Filter outlet port is at the end of the outlet stub (connection flange)
        filter_outlet = self.inlet_filter.ports['outlet']
        filter_outlet_world = (
            self._filter_position[0] + filter_outlet.position[0],
            self._filter_position[1] + filter_outlet.position[1],
            self._filter_position[2] + filter_outlet.position[2],
        )

        # ============================================================
        # 2. CENTRIFUGAL BLOWER
        # ============================================================
        # Blower geometry: inlet bell is on +Z side (at center + width/2*1.2)
        # Blower outlet is on +X side (scroll tangent)
        # Flow path: Filter(+X) → Duct(+X) → Elbow(turn to +Z) → Duct(+Z) → Blower inlet bell

        self.blower = create_standard_centrifugal_blower(
            flow_rate=p.flow_rate_m3_h,
            pressure_rise=p.pressure_rise_Pa
        )

        blower_inlet = self.blower.ports['inlet']
        blower_outlet = self.blower.ports['outlet']

        # Elbow parameters - use filter outlet diameter for smooth transition
        elbow_diameter = filter_outlet.diameter
        elbow_bend_radius = elbow_diameter * 0.7  # Tight bend for compact layout

        # Horizontal duct from filter outlet to elbow inlet
        duct_horiz_length = 0.08  # 80mm horizontal duct

        # Elbow inlet position: after horizontal duct from filter outlet
        elbow_inlet_x = filter_outlet_world[0] + duct_horiz_length + gap
        elbow_inlet_z = filter_outlet_world[2]  # Same Z as filter (Z=0)

        # After 90° elbow turning +X to +Z:
        # Elbow outlet is at: (elbow_inlet_x + R, Y, elbow_inlet_z + R)
        elbow_outlet_x = elbow_inlet_x + elbow_bend_radius
        elbow_outlet_z = elbow_inlet_z + elbow_bend_radius

        # The blower scroll body extends from center-half_width to center+half_width
        # where half_width = impeller_width/2 * 1.2
        # The inlet bell starts at center + half_width (at scroll +Z edge)
        # The duct must be long enough to:
        #   1. Start OUTSIDE the scroll body (with clearance)
        #   2. End at the inlet bell opening
        impeller_width = self.blower.params.impeller_width
        scroll_half_width = impeller_width / 2 * 1.2  # Half depth of scroll body
        scroll_clearance = 0.05  # 50mm clearance before scroll body

        # Duct length = clearance + full scroll depth to reach inlet bell
        # duct starts at elbow_outlet_z, must clear scroll body, then reach inlet bell
        duct_vert_length = scroll_clearance + 2 * scroll_half_width  # clearance + scroll depth

        # Position blower so:
        #   - Duct starts at elbow_outlet_z (with clearance before scroll)
        #   - Duct ends at inlet bell opening (scroll +Z edge)
        duct_end_z = elbow_outlet_z + duct_vert_length
        inlet_bell_offset = scroll_half_width  # Inlet bell starts at center + half_width
        blower_center_z = duct_end_z - inlet_bell_offset

        # Blower center X: inlet bell is at center X, so blower X = elbow outlet X
        blower_center_x = elbow_outlet_x

        self._blower_position = (
            blower_center_x,
            self._filter_position[1],  # Same Y as filter
            blower_center_z
        )

        # Store elbow parameters for duct creation
        self._elbow_params = {
            'diameter': elbow_diameter,
            'bend_radius': elbow_bend_radius,
            'inlet_pos': (elbow_inlet_x, filter_outlet_world[1], elbow_inlet_z),
            'duct_horiz_length': duct_horiz_length,
            'duct_vert_length': duct_vert_length,
        }

        blower_inlet_world = (
            self._blower_position[0] + blower_inlet.position[0],
            self._blower_position[1] + blower_inlet.position[1],
            self._blower_position[2] + blower_inlet.position[2],
        )

        # Blower outlet port is at scroll edge, but outlet duct extends further
        # The actual outlet flange is at: port_position + outlet_duct_length
        # outlet_duct_length = outlet_height * 1.5 (from _generate_outlet)
        blower_outlet_duct_length = self.blower.params.outlet_height * 1.5
        blower_outlet_flange_world = (
            self._blower_position[0] + blower_outlet.position[0] + blower_outlet_duct_length,
            self._blower_position[1] + blower_outlet.position[1],
            self._blower_position[2] + blower_outlet.position[2],
        )

        # ============================================================
        # 3. FLOW DAMPERS
        # ============================================================
        self.dampers: List = []
        self._damper_positions: List = []

        # Position dampers after blower outlet along +X
        # Direct connection: blower outlet flange → transition (with flange rings) → duct → damper
        transition_length = 0.15  # 150mm rect-to-round transition piece
        duct_after_transition = 0.05  # 50mm round duct

        # Start from actual outlet flange position (end of blower's outlet duct)
        # Path: blower flange → transition → duct → damper
        prev_outlet_x = blower_outlet_flange_world[0] + transition_length + duct_after_transition
        prev_outlet_y = blower_outlet_flange_world[1]
        prev_outlet_z = blower_outlet_flange_world[2]

        for i in range(p.num_control_dampers):
            damper = create_standard_damper(
                diameter=self._duct_diameter,
                damper_type=p.damper_type,
                position=1.0  # Fully open
            )
            self.dampers.append(damper)

            damper_inlet = damper.ports['inlet']
            damper_outlet = damper.ports['outlet']
            
            # Position damper: its inlet at previous outlet position
            damper_center_x = prev_outlet_x + abs(damper_inlet.position[0]) + gap
            
            damper_pos = (damper_center_x, prev_outlet_y, prev_outlet_z)
            self._damper_positions.append(damper_pos)
            
            # Update for next component: add small duct section between dampers
            duct_between_dampers = 0.05  # 50mm duct
            prev_outlet_x = damper_pos[0] + damper_outlet.position[0] + duct_between_dampers
        
        # ============================================================
        # 4. CREATE CONNECTING DUCTWORK
        # ============================================================
        self._create_duct_sections(gap, filter_outlet_world, blower_outlet_flange_world,
                                   transition_length, duct_after_transition)

    def _create_duct_sections(self, gap: float, filter_outlet_world: tuple,
                               blower_outlet_world: tuple,
                               transition_length: float, duct_after_transition: float):
        """
        Create duct sections connecting components.

        Flow path:
        1. Filter outlet (at stub flange) → Horizontal duct (+X) → 90° Elbow → Vertical duct (+Z) → Blower inlet
        2. Blower outlet (at outlet duct flange) → Rect-to-Round Transition (with flange rings) → Straight duct → Damper 1
        3. Damper 1 → Straight duct → Damper 2

        Note: Port positions are at the actual connection flanges (end of stubs),
        so ducts connect directly to these positions without overlap.
        The transition piece has square flange rings at both ends for bolted connections.
        """
        from ..components.ductwork import (
            RoundDuct, RoundDuctParams,
            RectangularDuct, RectangularDuctParams,
            DuctElbow, DuctElbowParams,
            RectToRoundTransition, RectToRoundTransitionParams,
        )

        self._duct_sections = []
        x_direction = (1.0, 0.0, 0.0)
        z_direction = (0.0, 0.0, 1.0)

        filter_outlet = self.inlet_filter.ports['outlet']
        blower_inlet = self.blower.ports['inlet']
        blower_outlet = self.blower.ports['outlet']

        # Get elbow parameters
        elbow = self._elbow_params

        # ============================================================
        # 1. FILTER TO BLOWER CONNECTION
        # ============================================================
        # 1a. Horizontal duct from filter outlet flange to elbow inlet
        # Filter outlet port is already at the end of the filter's outlet stub
        duct_horiz_start = filter_outlet_world
        duct_horiz = RoundDuct(RoundDuctParams(
            diameter=elbow['diameter'],
            length=elbow['duct_horiz_length'],
            wall_thickness=0.002,
            direction=x_direction,
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((duct_horiz, duct_horiz_start))

        # 1b. 90° Elbow turning from +X to +Z
        # Elbow inlet receives air from +X direction, outlet sends air in +Z direction
        # With inlet_dir=(1,0,0) and rotation_axis=(0,1,0):
        #   perp2 = cross(inlet_dir, rot_axis) = (0,0,1) → bend turns toward +Z
        elbow_component = DuctElbow(DuctElbowParams(
            diameter=elbow['diameter'],
            bend_radius=elbow['bend_radius'],
            angle=90.0,
            wall_thickness=0.002,
            flanged=True,
            center=(0, 0, 0),
            inlet_direction=(1.0, 0.0, 0.0),   # Air comes from +X direction
            rotation_axis=(0.0, 1.0, 0.0),     # Rotate around +Y to turn toward +Z
        ))
        self._duct_sections.append((elbow_component, elbow['inlet_pos']))

        # 1c. Vertical duct from elbow outlet to blower inlet
        # Elbow outlet position (after 90° bend)
        elbow_outlet_pos = (
            elbow['inlet_pos'][0] + elbow['bend_radius'],
            elbow['inlet_pos'][1],
            elbow['inlet_pos'][2] + elbow['bend_radius'],
        )

        # Vertical duct connects elbow outlet to blower inlet (on +Z side of blower)
        duct_vert = RoundDuct(RoundDuctParams(
            diameter=elbow['diameter'],
            length=elbow['duct_vert_length'],
            wall_thickness=0.002,
            direction=z_direction,
            center=(0, 0, 0),
            flanged=True,
        ))
        self._duct_sections.append((duct_vert, elbow_outlet_pos))
        
        # ============================================================
        # 2. BLOWER TO DAMPER CONNECTION
        # ============================================================
        # Blower outlet is rectangular, faces +X
        # Damper inlet is circular, faces -X
        # Connection: Blower outlet flange → Rect-to-Round Transition (with flange rings) → Straight duct → Damper

        if self.dampers:
            damper_inlet = self.dampers[0].ports['inlet']
            damper_inlet_world = (
                self._damper_positions[0][0] + damper_inlet.position[0],
                self._damper_positions[0][1] + damper_inlet.position[1],
                self._damper_positions[0][2] + damper_inlet.position[2],
            )

            # 2a. Rect-to-round transition connects directly to blower outlet flange
            # Transition has square flange rings at both ends for fitting
            transition_start = blower_outlet_world

            transition = RectToRoundTransition(RectToRoundTransitionParams(
                rect_width=blower_outlet.width,
                rect_height=blower_outlet.height,
                round_diameter=self._duct_diameter,
                length=transition_length,
                wall_thickness=0.002,
                center=(0, 0, 0),
                direction=x_direction,
                flanged=True,  # Add flange rings at both ends
            ))
            self._duct_sections.append((transition, transition_start))

            # 2b. Straight duct from transition end to damper inlet
            duct2_start_x = blower_outlet_world[0] + transition_length
            duct2_end_x = damper_inlet_world[0]
            duct2_length = duct2_end_x - duct2_start_x

            if duct2_length > 0.01:
                # Position is START of duct
                duct2_start = (duct2_start_x, blower_outlet_world[1], blower_outlet_world[2])

                duct2 = RoundDuct(RoundDuctParams(
                    diameter=self._duct_diameter,
                    length=duct2_length,
                    wall_thickness=0.002,
                    direction=x_direction,
                    center=(0, 0, 0),
                    flanged=True,
                ))
                self._duct_sections.append((duct2, duct2_start))
        
        # ============================================================
        # 3. BETWEEN DAMPERS
        # ============================================================
        for i in range(len(self.dampers) - 1):
            damper_outlet = self.dampers[i].ports['outlet']
            damper_outlet_world = (
                self._damper_positions[i][0] + damper_outlet.position[0],
                self._damper_positions[i][1] + damper_outlet.position[1],
                self._damper_positions[i][2] + damper_outlet.position[2],
            )
            
            next_damper_inlet = self.dampers[i+1].ports['inlet']
            next_damper_inlet_world = (
                self._damper_positions[i+1][0] + next_damper_inlet.position[0],
                self._damper_positions[i+1][1] + next_damper_inlet.position[1],
                self._damper_positions[i+1][2] + next_damper_inlet.position[2],
            )
            
            duct_length = next_damper_inlet_world[0] - damper_outlet_world[0]
            
            if duct_length > 0.01:
                # Position is START of duct (at previous damper outlet)
                duct_start = damper_outlet_world
                
                duct = RoundDuct(RoundDuctParams(
                    diameter=self.dampers[i].params.diameter,
                    length=duct_length,
                    wall_thickness=0.002,
                    direction=x_direction,
                    center=(0, 0, 0),
                    flanged=True,
                ))
                self._duct_sections.append((duct, duct_start))

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components including duct sections.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Helper to add component mesh with position offset
        def add_component_mesh(component, position):
            nonlocal vertex_offset
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add main components
        add_component_mesh(self.inlet_filter, self._filter_position)
        add_component_mesh(self.blower, self._blower_position)

        for damper, pos in zip(self.dampers, self._damper_positions):
            add_component_mesh(damper, pos)
        
        # Add duct sections
        for duct, pos in self._duct_sections:
            add_component_mesh(duct, pos)

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the entire system.

        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        if not self._mesh_built:
            self.build_mesh()

        min_corner = self._combined_vertices.min(axis=0)
        max_corner = self._combined_vertices.max(axis=0)

        return min_corner, max_corner

    def get_system_extent(self) -> np.ndarray:
        """
        Get system extent (dimensions) in each axis.

        Returns:
            Array of [width, height, depth]
        """
        min_c, max_c = self.get_bounds()
        return max_c - min_c

    def get_component(self, name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            name: Component name ('inlet_filter', 'blower', 'damper_0', 'damper_1', etc.)

        Returns:
            Component instance
        """
        components = {
            'inlet_filter': self.inlet_filter,
            'blower': self.blower,
        }
        for i, damper in enumerate(self.dampers):
            components[f'damper_{i}'] = damper

        if name not in components:
            raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
        return components[name]

    def get_component_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get positions of all components.

        Returns:
            Dictionary of component names to positions
        """
        positions = {
            'inlet_filter': self._filter_position,
            'blower': self._blower_position,
        }
        for i, pos in enumerate(self._damper_positions):
            positions[f'damper_{i}'] = pos
        return positions

    def set_damper_position(self, damper_index: int, position: float):
        """
        Set a damper's position.

        Args:
            damper_index: Index of damper (0, 1, ...)
            position: Position 0=closed, 1=fully open
        """
        if damper_index < 0 or damper_index >= len(self.dampers):
            raise IndexError(f"Damper index {damper_index} out of range (0-{len(self.dampers)-1})")
        self.dampers[damper_index].set_position(position)
        # Invalidate mesh
        self._mesh_built = False
        self._combined_vertices = None
        self._combined_indices = None

    def get_total_pressure_drop(self, flow_rate: float = None) -> float:
        """
        Estimate total system pressure drop.

        Args:
            flow_rate: Flow rate [m3/h], uses design flow if None

        Returns:
            Estimated pressure drop [Pa]
        """
        Q = flow_rate if flow_rate else self.params.flow_rate_m3_h

        # Filter pressure drop
        dp_filter = self.inlet_filter.get_pressure_drop(Q, loading=0.5)

        # Damper pressure drops
        dp_dampers = sum(d.get_pressure_drop(Q) for d in self.dampers)

        return dp_filter + dp_dampers

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get air system performance summary.

        Returns:
            Dictionary with performance metrics
        """
        p = self.params

        blower_perf = self.blower.get_performance()
        total_dp = self.get_total_pressure_drop()

        return {
            'design_flow_rate_m3_h': p.flow_rate_m3_h,
            'blower_pressure_rise_Pa': p.pressure_rise_Pa,
            'blower_power_kW': blower_perf['shaft_power_kW'],
            'blower_efficiency': blower_perf['efficiency'],
            'estimated_system_dp_Pa': total_dp,
            'filter_type': p.filter_type,
            'filter_class': p.filter_efficiency_class,
            'num_dampers': len(self.dampers),
        }

    def to_warp_mesh(self) -> wp.Mesh:
        """
        Create a Warp mesh from the system geometry.

        Returns:
            wp.Mesh object
        """
        if not self._mesh_built:
            self.build_mesh()

        points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
        indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)

        return wp.Mesh(points=points, indices=indices)

    def print_summary(self):
        """Print summary of the air system."""
        p = self.params
        perf = self.get_performance_summary()

        print("=" * 60)
        print("Air System Assembly Summary")
        print("=" * 60)

        print("\n1. INLET AIR FILTER")
        print(f"   Filter type:     {p.filter_type}")
        print(f"   Efficiency class: {p.filter_efficiency_class}")
        print(f"   Housing size:    {self.inlet_filter.params.housing_width*1000:.0f} x "
              f"{self.inlet_filter.params.housing_height*1000:.0f} mm")

        print("\n2. CENTRIFUGAL BLOWER")
        print(f"   Impeller dia:    {self.blower.params.impeller_diameter*1000:.0f} mm")
        print(f"   Design flow:     {p.flow_rate_m3_h:.0f} m3/h")
        print(f"   Pressure rise:   {p.pressure_rise_Pa:.0f} Pa")
        print(f"   Shaft power:     {perf['blower_power_kW']:.1f} kW")
        print(f"   Efficiency:      {perf['blower_efficiency']*100:.0f}%")

        print("\n3. FLOW DAMPERS")
        print(f"   Number:          {len(self.dampers)}")
        print(f"   Type:            {p.damper_type}")
        if self.dampers:
            print(f"   Diameter:        {self.dampers[0].params.diameter*1000:.0f} mm")

        print("-" * 60)
        extent = self.get_system_extent()
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print(f"Total system dP: {perf['estimated_system_dp_Pa']:.0f} Pa")

        if self._mesh_built:
            n_verts = len(self._combined_vertices)
            n_tris = len(self._combined_indices) // 3
            print(f"Total mesh:    {n_verts} vertices, {n_tris} triangles")
        print("=" * 60)

    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_vertices

    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_indices


def create_standard_air_system(device: str = "cpu") -> AirSystemAssembly:
    """
    Create a standard air system with default parameters.

    Args:
        device: Warp device

    Returns:
        AirSystemAssembly instance
    """
    return AirSystemAssembly(device=device)


def create_air_system_for_classifier(
    flow_rate_m3_h: float = 3000,
    system_pressure_drop_Pa: float = 5000,
    device: str = "cpu"
) -> AirSystemAssembly:
    """
    Create an air system sized for a specific classifier.

    Args:
        flow_rate_m3_h: Required flow rate [m3/h]
        system_pressure_drop_Pa: Total system pressure drop to overcome [Pa]
        device: Warp device

    Returns:
        AirSystemAssembly configured for given requirements
    """
    params = AirSystemParams(
        flow_rate_m3_h=flow_rate_m3_h,
        pressure_rise_Pa=system_pressure_drop_Pa * 1.2,  # 20% margin
        filter_type="panel",
        filter_efficiency_class="G4",
        num_control_dampers=2,
        damper_type="butterfly",
    )

    return AirSystemAssembly(params, device=device)
