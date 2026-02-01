"""
Debug script for complete system connection analysis.

Provides comprehensive diagnostics for:
- Connection point positions and orientations
- Duct routing path analysis (all segments)
- Port compatibility checks (diameter/direction)
- Subsystem introspection
- Alignment and gap calculations
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def format_vector(v, precision=3) -> str:
    """Format a vector for display."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return f"({v[0]:.{precision}f}, {v[1]:.{precision}f}, {v[2]:.{precision}f})"
    return str(v)


def format_mm(value_m: float) -> str:
    """Format meters as millimeters."""
    return f"{value_m * 1000:.0f} mm"


def direction_name(d) -> str:
    """Get human-readable direction name."""
    d = np.array(d)
    directions = {
        (1, 0, 0): "+X (right)",
        (-1, 0, 0): "-X (left)",
        (0, 1, 0): "+Y (forward)",
        (0, -1, 0): "-Y (back)",
        (0, 0, 1): "+Z (up)",
        (0, 0, -1): "-Z (down)",
    }
    for key, name in directions.items():
        if np.allclose(d, key, atol=0.1):
            return name
    return format_vector(d)


def check_port_compatibility(port1_d: float, port2_d: float, 
                              port1_dir: np.ndarray, port2_dir: np.ndarray) -> Dict[str, Any]:
    """Check if two ports are compatible for connection."""
    diameter_match = abs(port1_d - port2_d) < 0.001
    diameter_ratio = min(port1_d, port2_d) / max(port1_d, port2_d) if max(port1_d, port2_d) > 0 else 0
    
    # Directions should be opposite for proper connection
    dir_dot = np.dot(np.array(port1_dir), np.array(port2_dir))
    directions_aligned = dir_dot < -0.9  # Should be anti-parallel
    
    return {
        'diameter_match': diameter_match,
        'diameter_ratio': diameter_ratio,
        'diameter_diff_mm': abs(port1_d - port2_d) * 1000,
        'directions_aligned': directions_aligned,
        'direction_dot': dir_dot,
        'needs_transition': not diameter_match,
    }


def print_section_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_subsection(title: str, char: str = "-", width: int = 60):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def analyze_port(port, name: str, local_offset: np.ndarray = None, world_offset: np.ndarray = None):
    """Analyze and print port details."""
    print(f"\n  {name}:")
    print(f"    Position (local):  {format_vector(port.position)}")
    
    if local_offset is not None:
        adjusted = np.array(port.position) + np.array(local_offset)
        print(f"    Position (adj.):   {format_vector(adjusted)}")
    
    if world_offset is not None:
        world_pos = np.array(port.position)
        if local_offset is not None:
            world_pos = world_pos + np.array(local_offset)
        world_pos = world_pos + np.array(world_offset)
        print(f"    Position (world):  {format_vector(world_pos)}")
    
    print(f"    Direction:         {direction_name(port.direction)}")
    print(f"    Diameter:          {format_mm(port.diameter)}")
    return port


def analyze_subsystem(subsystem, name: str, offset: np.ndarray):
    """Analyze a subsystem's components and ports."""
    print_subsection(f"Subsystem: {name}")
    print(f"  World Offset: {format_vector(offset)}")
    
    # Get component positions if available
    if hasattr(subsystem, 'get_component_positions'):
        positions = subsystem.get_component_positions()
        print(f"\n  Components ({len(positions)}):")
        for comp_name, pos in positions.items():
            world_pos = np.array(pos) + offset
            print(f"    - {comp_name:20s} local={format_vector(pos):30s} world={format_vector(world_pos)}")
    
    # List available ports
    if hasattr(subsystem, 'ports') and subsystem.ports:
        print(f"\n  Ports:")
        for port_name, port in subsystem.ports.items():
            print(f"    - {port_name}: D={format_mm(port.diameter)}, dir={direction_name(port.direction)}")


def analyze_duct_chain(ducts: List[Tuple[Any, Tuple]], connection_name: str):
    """Analyze a chain of duct components."""
    print_subsection(f"Duct Chain: {connection_name}")
    print(f"  Total segments: {len(ducts)}")
    
    total_length = 0
    for i, (duct, position) in enumerate(ducts):
        duct_type = type(duct).__name__
        pos_str = format_vector(position)
        
        # Extract duct properties
        props = []
        if hasattr(duct, 'params'):
            p = duct.params
            if hasattr(p, 'diameter'):
                props.append(f"D={format_mm(p.diameter)}")
            if hasattr(p, 'length'):
                props.append(f"L={format_mm(p.length)}")
                total_length += p.length
            if hasattr(p, 'bend_radius'):
                props.append(f"R={format_mm(p.bend_radius)}")
            if hasattr(p, 'angle'):
                props.append(f"θ={p.angle:.0f}°")
            if hasattr(p, 'direction'):
                props.append(f"dir={direction_name(p.direction)}")
            if hasattr(p, 'inlet_direction'):
                props.append(f"in={direction_name(p.inlet_direction)}")
            if hasattr(p, 'inlet_dimensions') and hasattr(p, 'outlet_dimensions'):
                in_d = p.inlet_dimensions[0] if p.inlet_dimensions else 0
                out_d = p.outlet_dimensions[0] if p.outlet_dimensions else 0
                props.append(f"{format_mm(in_d)}→{format_mm(out_d)}")
        
        props_str = ", ".join(props) if props else ""
        print(f"  [{i+1:2d}] {duct_type:20s} @ {pos_str:35s} {props_str}")
    
    if total_length > 0:
        print(f"\n  Total straight length: {format_mm(total_length)}")


def get_connection_endpoints(system) -> Dict[str, Dict[str, Any]]:
    """Extract connection endpoint information from the system."""
    endpoints = {}
    
    classification = system._subsystems.get('classification')
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    
    feed_system = system._subsystems.get('feed_system')
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    
    air_system = system._subsystems.get('air_system')
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    
    # Connection 1: Air System -> Venturi air_inlet
    if air_system and classification:
        # Air system outlet (from last damper)
        if hasattr(air_system, 'dampers') and air_system.dampers:
            last_damper = air_system.dampers[-1]
            damper_local_pos = np.array(air_system._damper_positions[-1])
            outlet_port = last_damper.ports['outlet']
            outlet_world = air_offset + damper_local_pos + np.array(outlet_port.position)
            
            endpoints['air_outlet'] = {
                'position': outlet_world,
                'direction': np.array(outlet_port.direction),
                'diameter': outlet_port.diameter,
                'component': 'Damper (Air System)',
            }
        
        # Venturi air inlet
        venturi = classification.venturi
        class_positions = classification.get_component_positions()
        venturi_local = np.array(class_positions['venturi'])
        air_inlet = venturi.ports['air_inlet']
        inlet_world = class_offset + venturi_local + np.array(air_inlet.position)
        
        endpoints['venturi_air_inlet'] = {
            'position': inlet_world,
            'direction': np.array(air_inlet.direction),
            'diameter': air_inlet.diameter,
            'component': 'Venturi (Classification)',
        }
    
    # Connection 2: Feed System -> Venturi solids_inlet
    if feed_system and classification:
        # Feed system outlet (from deagglomerator)
        feed_positions = feed_system.get_component_positions()
        deagg_local = np.array(feed_positions['deagglomerator'])
        deagg_outlet = feed_system.deagglomerator.ports['outlet']
        outlet_world = feed_offset + deagg_local + np.array(deagg_outlet.position)
        
        endpoints['feed_outlet'] = {
            'position': outlet_world,
            'direction': np.array(deagg_outlet.direction),
            'diameter': deagg_outlet.diameter,
            'component': 'Deagglomerator (Feed System)',
        }
        
        # Venturi solids inlet
        venturi = classification.venturi
        solids_inlet = venturi.ports['solids_inlet']
        venturi_local = np.array(class_positions['venturi'])
        inlet_world = class_offset + venturi_local + np.array(solids_inlet.position)
        
        endpoints['venturi_solids_inlet'] = {
            'position': inlet_world,
            'direction': np.array(solids_inlet.direction),
            'diameter': solids_inlet.diameter,
            'component': 'Venturi (Classification)',
        }
    
    # Connection 3: Bag Filter -> Silencer
    if classification:
        bag_filter = classification.bag_filter
        bf_local = np.array(classification._component_positions['bag_filter'])
        clean_air_port = bag_filter.ports['clean_air_outlet']
        bf_outlet_world = class_offset + bf_local + np.array(clean_air_port.position)
        
        endpoints['bagfilter_outlet'] = {
            'position': bf_outlet_world,
            'direction': np.array(clean_air_port.direction),
            'diameter': clean_air_port.diameter,
            'component': 'Bag Filter (Classification)',
        }
    
    silencer = system._components.get('silencer')
    if silencer:
        silencer_dir = np.array(silencer.params.direction_normalized)
        silencer_center = np.array(silencer.params.center)
        silencer_length = silencer.params.length
        silencer_inlet = silencer_center - silencer_dir * (silencer_length / 2)
        silencer_outlet = silencer_center + silencer_dir * (silencer_length / 2)
        
        endpoints['silencer_inlet'] = {
            'position': silencer_inlet,
            'direction': -silencer_dir,  # Inlet direction is opposite of silencer direction
            'diameter': silencer.params.diameter,
            'component': 'Silencer (Exhaust)',
        }
        
        endpoints['silencer_outlet'] = {
            'position': silencer_outlet,
            'direction': silencer_dir,  # Outlet direction matches silencer direction
            'diameter': silencer.params.diameter,
            'component': 'Silencer (Exhaust)',
        }
    
    # Connection 4: Silencer -> Exhaust Stack
    exhaust_stack = system._components.get('exhaust_stack')
    if exhaust_stack:
        stack_center = np.array(exhaust_stack.params.center)
        stack_d = exhaust_stack.params.diameter
        # Stack inlet is at the base (center position for vertical stack)
        
        endpoints['stack_inlet'] = {
            'position': stack_center,
            'direction': np.array([0.0, 0.0, -1.0]),  # Expects flow from below
            'diameter': stack_d,
            'component': 'Exhaust Stack',
        }
        
        # Stack outlet is at the top
        stack_top = stack_center + np.array([0, 0, exhaust_stack.params.height])
        endpoints['stack_outlet'] = {
            'position': stack_top,
            'direction': np.array([0.0, 0.0, 1.0]),  # Flow exits upward
            'diameter': stack_d,
            'component': 'Exhaust Stack (Top)',
        }
    
    return endpoints


def analyze_connection(name: str, from_key: str, to_key: str, endpoints: Dict):
    """Analyze a single connection between two endpoints."""
    print_subsection(f"Connection: {name}")
    
    if from_key not in endpoints or to_key not in endpoints:
        print("  ERROR: Missing endpoint data")
        return
    
    from_ep = endpoints[from_key]
    to_ep = endpoints[to_key]
    
    print(f"\n  FROM: {from_ep['component']}")
    print(f"    Position:  {format_vector(from_ep['position'])}")
    print(f"    Direction: {direction_name(from_ep['direction'])} (flow out)")
    print(f"    Diameter:  {format_mm(from_ep['diameter'])}")
    
    print(f"\n  TO: {to_ep['component']}")
    print(f"    Position:  {format_vector(to_ep['position'])}")
    print(f"    Direction: {direction_name(to_ep['direction'])} (expects from)")
    print(f"    Diameter:  {format_mm(to_ep['diameter'])}")
    
    # Calculate delta and distance
    delta = to_ep['position'] - from_ep['position']
    distance = np.linalg.norm(delta)
    
    print(f"\n  ROUTING:")
    print(f"    Delta (X,Y,Z): {format_vector(delta)}")
    print(f"    Distance:      {format_mm(distance)}")
    print(f"    ΔX: {delta[0]:+.3f} m ({format_mm(abs(delta[0]))})")
    print(f"    ΔY: {delta[1]:+.3f} m ({format_mm(abs(delta[1]))})")
    print(f"    ΔZ: {delta[2]:+.3f} m ({format_mm(abs(delta[2]))})")
    
    # Port compatibility check
    compat = check_port_compatibility(
        from_ep['diameter'], to_ep['diameter'],
        from_ep['direction'], to_ep['direction']
    )
    
    print(f"\n  COMPATIBILITY:")
    print(f"    Diameter match:     {'✓' if compat['diameter_match'] else '✗'} (diff: {compat['diameter_diff_mm']:.1f} mm)")
    print(f"    Diameter ratio:     {compat['diameter_ratio']:.2f}")
    print(f"    Needs transition:   {'Yes' if compat['needs_transition'] else 'No'}")
    print(f"    Directions aligned: {'✓' if compat['directions_aligned'] else '✗'} (dot: {compat['direction_dot']:.2f})")


def categorize_duct_connections(system) -> Dict[str, List[Tuple[Any, Tuple]]]:
    """Categorize duct connections by their connection type."""
    categories = {
        'air_to_venturi': [],
        'feed_to_venturi': [],
        'bagfilter_to_exhaust': [],
    }
    
    if not hasattr(system, '_duct_connections') or not system._duct_connections:
        return categories
    
    # Get reference positions to categorize ducts
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    
    # Simple heuristic: categorize by starting position proximity
    for duct, pos in system._duct_connections:
        pos_arr = np.array(pos)
        
        # Distance to each system's offset
        dist_to_air = np.linalg.norm(pos_arr - air_offset)
        dist_to_feed = np.linalg.norm(pos_arr - feed_offset)
        dist_to_class = np.linalg.norm(pos_arr - class_offset)
        
        # Also check Z position and duct characteristics
        if hasattr(duct, 'params'):
            p = duct.params
            # Air to venturi ducts start near air system
            if dist_to_air < 3.0 and pos_arr[1] < -2.0:  # Y < -2 is toward air system
                categories['air_to_venturi'].append((duct, pos))
            # Feed to venturi ducts have significant X offset and use diagonal direction
            elif pos_arr[0] < -2.0:  # X < -2 is toward feed system
                categories['feed_to_venturi'].append((duct, pos))
            # Bag filter to exhaust ducts are near classification with high Z
            else:
                categories['bagfilter_to_exhaust'].append((duct, pos))
    
    return categories


def analyze_all_ports(system):
    """Analyze all available ports in the system."""
    print_section_header("ALL SYSTEM PORTS")
    
    classification = system._subsystems.get('classification')
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    
    if classification:
        print("\n  CLASSIFICATION SYSTEM PORTS:")
        class_positions = classification.get_component_positions()
        
        # Venturi ports
        if hasattr(classification, 'venturi'):
            venturi = classification.venturi
            venturi_pos = np.array(class_positions.get('venturi', (0, 0, 0)))
            print(f"\n    Venturi (at {format_vector(venturi_pos + class_offset)}):")
            for port_name, port in venturi.ports.items():
                world_pos = venturi_pos + class_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
        
        # Bag Filter ports
        if hasattr(classification, 'bag_filter'):
            bf = classification.bag_filter
            bf_pos = np.array(classification._component_positions.get('bag_filter', (0, 0, 0)))
            print(f"\n    Bag Filter (at {format_vector(bf_pos + class_offset)}):")
            for port_name, port in bf.ports.items():
                world_pos = bf_pos + class_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
        
        # Cyclone ports
        if hasattr(classification, 'cyclone'):
            cyc = classification.cyclone
            cyc_pos = np.array(class_positions.get('cyclone', (0, 0, 0)))
            print(f"\n    Cyclone (at {format_vector(cyc_pos + class_offset)}):")
            for port_name, port in cyc.ports.items():
                world_pos = cyc_pos + class_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
    
    # Air System
    air_system = system._subsystems.get('air_system')
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    
    if air_system:
        print("\n  AIR SYSTEM PORTS:")
        
        # Blower ports
        if hasattr(air_system, 'blower'):
            blower = air_system.blower
            blower_pos = np.array(air_system._blower_position) if hasattr(air_system, '_blower_position') else np.zeros(3)
            print(f"\n    Blower (at {format_vector(blower_pos + air_offset)}):")
            for port_name, port in blower.ports.items():
                world_pos = blower_pos + air_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
        
        # Damper ports
        if hasattr(air_system, 'dampers') and air_system.dampers:
            for i, (damper, damper_pos) in enumerate(zip(air_system.dampers, air_system._damper_positions)):
                damper_pos = np.array(damper_pos)
                print(f"\n    Damper {i+1} (at {format_vector(damper_pos + air_offset)}):")
                for port_name, port in damper.ports.items():
                    world_pos = damper_pos + air_offset + np.array(port.position)
                    print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
    
    # Feed System
    feed_system = system._subsystems.get('feed_system')
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    
    if feed_system:
        print("\n  FEED SYSTEM PORTS:")
        feed_positions = feed_system.get_component_positions()
        
        # Hopper ports
        if hasattr(feed_system, 'hopper'):
            hopper = feed_system.hopper
            hopper_pos = np.array(feed_positions.get('hopper', (0, 0, 0)))
            print(f"\n    Hopper (at {format_vector(hopper_pos + feed_offset)}):")
            for port_name, port in hopper.ports.items():
                world_pos = hopper_pos + feed_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
        
        # Deagglomerator ports
        if hasattr(feed_system, 'deagglomerator'):
            deagg = feed_system.deagglomerator
            deagg_pos = np.array(feed_positions.get('deagglomerator', (0, 0, 0)))
            print(f"\n    Deagglomerator (at {format_vector(deagg_pos + feed_offset)}):")
            for port_name, port in deagg.ports.items():
                world_pos = deagg_pos + feed_offset + np.array(port.position)
                print(f"      {port_name:15s} D={format_mm(port.diameter):8s} dir={direction_name(port.direction):15s} @ {format_vector(world_pos)}")
    
    # Exhaust System Components
    silencer = system._components.get('silencer')
    exhaust_stack = system._components.get('exhaust_stack')
    
    if silencer or exhaust_stack:
        print("\n  EXHAUST SYSTEM:")
        
        if silencer:
            silencer_center = np.array(silencer.params.center)
            silencer_dir = np.array(silencer.params.direction_normalized)
            silencer_len = silencer.params.length
            silencer_d = silencer.params.diameter
            inlet_pos = silencer_center - silencer_dir * (silencer_len / 2)
            outlet_pos = silencer_center + silencer_dir * (silencer_len / 2)
            
            print(f"\n    Silencer (center at {format_vector(silencer_center)}):")
            print(f"      inlet           D={format_mm(silencer_d):8s} dir={direction_name(-silencer_dir):15s} @ {format_vector(inlet_pos)}")
            print(f"      outlet          D={format_mm(silencer_d):8s} dir={direction_name(silencer_dir):15s} @ {format_vector(outlet_pos)}")
        
        if exhaust_stack:
            stack_center = np.array(exhaust_stack.params.center)
            stack_d = exhaust_stack.params.diameter
            stack_h = exhaust_stack.params.height
            stack_top = stack_center + np.array([0, 0, stack_h])
            
            print(f"\n    Exhaust Stack (base at {format_vector(stack_center)}):")
            print(f"      inlet (base)    D={format_mm(stack_d):8s} dir={direction_name([0,0,-1]):15s} @ {format_vector(stack_center)}")
            print(f"      outlet (top)    D={format_mm(stack_d):8s} dir={direction_name([0,0,1]):15s} @ {format_vector(stack_top)}")


def print_system_layout(system):
    """Print ASCII art layout of the system."""
    print_section_header("SYSTEM LAYOUT (Top View, +Y up, +X right)")
    
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    
    silencer = system._components.get('silencer')
    silencer_pos = np.array(silencer.params.center) if silencer else np.zeros(3)
    
    layout = f"""
                                    +Y (back)
                                      │
                                      │
    ┌──────────────┐                  │     ┌─────────────────┐
    │   FEED       │                  │     │   BAG FILTER    │
    │   SYSTEM     │                  │     │                 │
    │   ({feed_offset[0]:.1f}, {feed_offset[1]:.1f})   │    ╔════════════════════╗         └────────┬────────┘
    └──────┬───────┘                  │     ║   CLASSIFICATION ║                  │
           │                          │     ║     SYSTEM       ║                  │ (duct to exhaust)
           │ chute                    │     ║   ({class_offset[0]:.1f}, {class_offset[1]:.1f})      ║                  │
           │                          │     ║                  ║                  ▼
           └──────────────────────────┼────►║     VENTURI      ║           ┌──────────────┐
                                      │     ╚════════════════════╝           │   SILENCER   │
                                      │              ▲                       │ ({silencer_pos[0]:.1f}, {silencer_pos[1]:.1f})   │
                                      │              │                       └──────────────┘
                                      │              │ air duct
                                      │     ┌───────┴────────┐
                                      │     │   AIR SYSTEM   │
                                      │     │   (blower)     │
                                      │     │ ({air_offset[0]:.1f}, {air_offset[1]:.1f})       │
                                      │     └────────────────┘
                         ─────────────┴────────────────────────────────────► +X (right)
                                      │
                                    -Y (front)

    Key Connections:
    1. Air System outlet ──────► Venturi air_inlet
    2. Feed System outlet ─────► Venturi solids_inlet
    3. Bag Filter outlet ──────► Silencer inlet
    """
    print(layout)


def main():
    """Run complete system debug analysis."""
    from airclassifier.geometry.assembly.complete_system import create_core_connections_system
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPLETE SYSTEM CONNECTION DEBUG ANALYSIS".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Create the system
    print("\nCreating core connections system...")
    system = create_core_connections_system()
    
    # System summary
    print_section_header("SYSTEM CONFIGURATION")
    p = system.params
    print(f"  Throughput:      {p.throughput_kg_h} kg/h")
    print(f"  Air Flow:        {p.air_flow_m3_h} m³/h")
    print(f"  Main Duct Dia:   {format_mm(p.main_duct_diameter)}")
    print(f"\n  Included Systems:")
    print(f"    [{'✓' if p.include_feed_system else ' '}] Feed System")
    print(f"    [{'✓' if p.include_air_system else ' '}] Air System")
    print(f"    [{'✓' if p.include_ductwork else ' '}] Ductwork")
    print(f"    [{'✓' if p.include_exhaust else ' '}] Exhaust")
    print(f"    [{'✓' if p.include_support_structure else ' '}] Support Structure")
    
    # Offsets
    print_section_header("SUBSYSTEM OFFSETS")
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    
    print(f"  Classification:  {format_vector(class_offset)}")
    print(f"  Feed System:     {format_vector(feed_offset)}")
    print(f"  Air System:      {format_vector(air_offset)}")
    
    # System layout
    print_system_layout(system)
    
    # Analyze all ports
    analyze_all_ports(system)
    
    # Connection endpoint analysis
    print_section_header("CONNECTION ENDPOINT ANALYSIS")
    endpoints = get_connection_endpoints(system)
    
    # Analyze each connection
    analyze_connection(
        "Air System → Venturi (Air Supply)",
        'air_outlet', 'venturi_air_inlet', endpoints
    )
    
    analyze_connection(
        "Feed System → Venturi (Solids Feed)",
        'feed_outlet', 'venturi_solids_inlet', endpoints
    )
    
    analyze_connection(
        "Bag Filter → Silencer (Clean Air Exhaust)",
        'bagfilter_outlet', 'silencer_inlet', endpoints
    )
    
    analyze_connection(
        "Silencer → Exhaust Stack (Atmospheric Discharge)",
        'silencer_outlet', 'stack_inlet', endpoints
    )
    
    # Duct connection chain analysis
    print_section_header("DUCT CONNECTION CHAINS")
    
    if hasattr(system, '_duct_connections') and system._duct_connections:
        print(f"\n  Total duct segments: {len(system._duct_connections)}")
        
        # List all ducts with indices
        print("\n  Complete Duct Inventory:")
        for i, (duct, pos) in enumerate(system._duct_connections):
            duct_type = type(duct).__name__
            props = []
            if hasattr(duct, 'params'):
                p = duct.params
                if hasattr(p, 'diameter'):
                    props.append(f"D={format_mm(p.diameter)}")
                if hasattr(p, 'length'):
                    props.append(f"L={format_mm(p.length)}")
                if hasattr(p, 'bend_radius'):
                    props.append(f"R={format_mm(p.bend_radius)}")
                if hasattr(p, 'inlet_dimensions') and hasattr(p, 'outlet_dimensions'):
                    props.append(f"trans")
            
            props_str = " | ".join(props) if props else ""
            print(f"    [{i+1:2d}] {duct_type:20s} @ {format_vector(pos):35s} {props_str}")
        
        # Try to categorize ducts
        categories = categorize_duct_connections(system)
        
        print("\n  Categorized Connections:")
        for cat_name, ducts in categories.items():
            if ducts:
                print(f"\n    {cat_name}: {len(ducts)} segments")
    else:
        print("\n  No duct connections found.")
    
    # Bill of Materials
    print_section_header("DUCTWORK BILL OF MATERIALS")
    
    if hasattr(system, '_duct_connections') and system._duct_connections:
        duct_counts = {}
        total_length = 0
        
        for duct, _ in system._duct_connections:
            duct_type = type(duct).__name__
            duct_counts[duct_type] = duct_counts.get(duct_type, 0) + 1
            
            if hasattr(duct, 'params') and hasattr(duct.params, 'length'):
                total_length += duct.params.length
        
        print(f"\n  {'Component':<25} {'Quantity':>10}")
        print(f"  {'-'*25} {'-'*10}")
        for item, qty in sorted(duct_counts.items()):
            print(f"  {item:<25} {qty:>10}")
        print(f"  {'-'*25} {'-'*10}")
        print(f"  {'TOTAL':<25} {sum(duct_counts.values()):>10}")
        print(f"\n  Total straight duct length: {format_mm(total_length)}")
    
    # Geometry summary
    print_section_header("GEOMETRY SUMMARY")
    
    # Build mesh to get stats
    verts, indices = system.build_mesh()
    bounds_min, bounds_max = system.get_bounds()
    dimensions = bounds_max - bounds_min
    
    print(f"  Vertices:    {len(verts):,}")
    print(f"  Triangles:   {len(indices)//3:,}")
    print(f"  Bounds Min:  {format_vector(bounds_min)}")
    print(f"  Bounds Max:  {format_vector(bounds_max)}")
    print(f"  Dimensions:  {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} m")
    print(f"               ({format_mm(dimensions[0])} x {format_mm(dimensions[1])} x {format_mm(dimensions[2])})")
    
    # Final summary
    print_section_header("ROUTING REQUIREMENTS SUMMARY")
    
    if 'air_outlet' in endpoints and 'venturi_air_inlet' in endpoints:
        delta = endpoints['venturi_air_inlet']['position'] - endpoints['air_outlet']['position']
        print(f"\n  1. AIR → VENTURI:")
        print(f"     From: {format_vector(endpoints['air_outlet']['position'])} ({direction_name(endpoints['air_outlet']['direction'])})")
        print(f"     To:   {format_vector(endpoints['venturi_air_inlet']['position'])} (expects {direction_name(endpoints['venturi_air_inlet']['direction'])})")
        print(f"     Delta: {format_vector(delta)}, Distance: {format_mm(np.linalg.norm(delta))}")
        print(f"     Note: Duct must approach venturi from BELOW (-Y)")
    
    if 'feed_outlet' in endpoints and 'venturi_solids_inlet' in endpoints:
        delta = endpoints['venturi_solids_inlet']['position'] - endpoints['feed_outlet']['position']
        print(f"\n  2. FEED → VENTURI:")
        print(f"     From: {format_vector(endpoints['feed_outlet']['position'])} ({direction_name(endpoints['feed_outlet']['direction'])})")
        print(f"     To:   {format_vector(endpoints['venturi_solids_inlet']['position'])} (expects {direction_name(endpoints['venturi_solids_inlet']['direction'])})")
        print(f"     Delta: {format_vector(delta)}, Distance: {format_mm(np.linalg.norm(delta))}")
        print(f"     Note: Gravity chute, angled for powder flow")
    
    if 'bagfilter_outlet' in endpoints and 'silencer_inlet' in endpoints:
        delta = endpoints['silencer_inlet']['position'] - endpoints['bagfilter_outlet']['position']
        print(f"\n  3. BAG FILTER → SILENCER:")
        print(f"     From: {format_vector(endpoints['bagfilter_outlet']['position'])} ({direction_name(endpoints['bagfilter_outlet']['direction'])})")
        print(f"     To:   {format_vector(endpoints['silencer_inlet']['position'])} (expects {direction_name(endpoints['silencer_inlet']['direction'])})")
        print(f"     Delta: {format_vector(delta)}, Distance: {format_mm(np.linalg.norm(delta))}")
        print(f"     Note: Routes through elbows up to silencer height")
    
    if 'silencer_outlet' in endpoints and 'stack_inlet' in endpoints:
        delta = endpoints['stack_inlet']['position'] - endpoints['silencer_outlet']['position']
        distance = np.linalg.norm(delta)
        print(f"\n  4. SILENCER → EXHAUST STACK:")
        print(f"     From: {format_vector(endpoints['silencer_outlet']['position'])} ({direction_name(endpoints['silencer_outlet']['direction'])})")
        print(f"     To:   {format_vector(endpoints['stack_inlet']['position'])} (expects {direction_name(endpoints['stack_inlet']['direction'])})")
        print(f"     Delta: {format_vector(delta)}, Distance: {format_mm(distance)}")
        if distance < 0.01:
            print(f"     Status: ✓ DIRECTLY CONNECTED (no gap)")
        else:
            print(f"     Status: ✗ GAP DETECTED - needs connection duct")
    
    print("\n" + "█" * 80)
    print("█" + "  DEBUG ANALYSIS COMPLETE".center(78) + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
