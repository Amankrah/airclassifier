"""
Debug script for complete system connection analysis.

Provides comprehensive diagnostics for:
- Connection point positions and orientations
- Duct routing path analysis with chain tracing
- Port compatibility checks (diameter/direction)
- Gap and alignment calculations between duct segments
- No magic numbers - uses actual component data
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# COORDINATE SYSTEM (per user specification):
#   X+: From air filter toward deagglomerator (horizontal)
#   Y+: Vertical (upward toward bag filter top outlet)  
#   Z+: Distance away from classification system (depth)
# =============================================================================


def format_vector(v, precision=4) -> str:
    """Format a vector for display."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return f"({v[0]:.{precision}f}, {v[1]:.{precision}f}, {v[2]:.{precision}f})"
    return str(v)


def format_mm(value_m: float) -> str:
    """Format meters as millimeters."""
    return f"{value_m * 1000:.1f}mm"


def direction_name(d) -> str:
    """Get human-readable direction name based on coordinate system."""
    d = np.array(d)
    d_norm = d / (np.linalg.norm(d) + 1e-9)
    
    # Primary axis directions
    directions = {
        (1, 0, 0): "+X (toward deagglomerator)",
        (-1, 0, 0): "-X (toward air filter)",
        (0, 1, 0): "+Y (up/vertical)",
        (0, -1, 0): "-Y (down/vertical)",
        (0, 0, 1): "+Z (away from classifier)",
        (0, 0, -1): "-Z (toward classifier)",
    }
    
    for key, name in directions.items():
        if np.allclose(d_norm, key, atol=0.1):
            return name
    
    # Describe angled directions
    desc = []
    if abs(d_norm[0]) > 0.1:
        desc.append(f"{'+'if d_norm[0]>0 else '-'}X:{abs(d_norm[0]):.2f}")
    if abs(d_norm[1]) > 0.1:
        desc.append(f"{'+'if d_norm[1]>0 else '-'}Y:{abs(d_norm[1]):.2f}")
    if abs(d_norm[2]) > 0.1:
        desc.append(f"{'+'if d_norm[2]>0 else '-'}Z:{abs(d_norm[2]):.2f}")
    
    return f"angled ({', '.join(desc)})"


@dataclass
class PortInfo:
    """Information about a connection port."""
    name: str
    component: str
    position: np.ndarray
    direction: np.ndarray
    diameter: float


@dataclass  
class DuctSegment:
    """Information about a duct segment."""
    index: int
    duct_type: str
    position: np.ndarray
    diameter: float
    length: Optional[float]
    direction: Optional[np.ndarray]
    bend_radius: Optional[float]
    angle: Optional[float]
    inlet_direction: Optional[np.ndarray]
    rotation_axis: Optional[np.ndarray]
    inlet_diameter: Optional[float]
    outlet_diameter: Optional[float]
    duct_obj: Any


def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_subheader(title: str, char: str = "-", width: int = 60):
    """Print a formatted subheader."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def extract_duct_segment(duct, position: Tuple, index: int) -> DuctSegment:
    """Extract information from a duct component."""
    pos = np.array(position)
    duct_type = type(duct).__name__
    
    diameter = None
    length = None
    direction = None
    bend_radius = None
    angle = None
    inlet_direction = None
    rotation_axis = None
    inlet_d = None
    outlet_d = None
    
    if hasattr(duct, 'params'):
        p = duct.params
        if hasattr(p, 'diameter'):
            diameter = p.diameter
        if hasattr(p, 'length'):
            length = p.length
        if hasattr(p, 'direction'):
            direction = np.array(p.direction) if p.direction else None
        if hasattr(p, 'bend_radius'):
            bend_radius = p.bend_radius
        if hasattr(p, 'angle'):
            angle = p.angle
        if hasattr(p, 'inlet_direction'):
            inlet_direction = np.array(p.inlet_direction) if p.inlet_direction else None
        if hasattr(p, 'rotation_axis'):
            rotation_axis = np.array(p.rotation_axis) if p.rotation_axis else None
        if hasattr(p, 'inlet_dimensions') and p.inlet_dimensions:
            inlet_d = p.inlet_dimensions[0]
        if hasattr(p, 'outlet_dimensions') and p.outlet_dimensions:
            outlet_d = p.outlet_dimensions[0]
    
    return DuctSegment(
        index=index,
        duct_type=duct_type,
        position=pos,
        diameter=diameter,
        length=length,
        direction=direction,
        bend_radius=bend_radius,
        angle=angle,
        inlet_direction=inlet_direction,
        rotation_axis=rotation_axis,
        inlet_diameter=inlet_d,
        outlet_diameter=outlet_d,
        duct_obj=duct
    )


def compute_duct_endpoint(seg: DuctSegment) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute the inlet and outlet positions of a duct segment.
    Returns: (inlet_pos, outlet_pos, outlet_direction)
    """
    inlet_pos = seg.position.copy()
    outlet_dir = None
    
    if seg.duct_type == "RoundDuct" and seg.direction is not None and seg.length:
        # Straight duct: outlet = inlet + direction * length
        d = np.array(seg.direction)
        d = d / (np.linalg.norm(d) + 1e-9)
        outlet_pos = inlet_pos + d * seg.length
        outlet_dir = d
        
    elif seg.duct_type == "DuctElbow" and seg.inlet_direction is not None and seg.bend_radius:
        # Elbow: compute outlet based on bend geometry
        # For 90-degree bend with rotation axis:
        # - inlet direction rotates 90° around rotation_axis to get outlet direction
        # - outlet position = inlet + R * in_dir + R * out_dir
        R = seg.bend_radius
        in_dir = np.array(seg.inlet_direction)
        in_dir = in_dir / (np.linalg.norm(in_dir) + 1e-9)
        
        if seg.rotation_axis is not None:
            # Compute outlet direction by rotating inlet_dir 90° around rotation_axis
            # The rotation_axis convention in DuctElbow: positive rotation follows right-hand rule
            # But the actual turn direction is opposite to cross product
            rot_axis = np.array(seg.rotation_axis)
            rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-9)
            
            # For 90-degree elbow: outlet_dir is perpendicular to inlet_dir
            # The cross product gives perpendicular vector, but we need to negate it
            # because the DuctElbow convention has rotation_axis pointing such that
            # the turn goes in the -cross direction
            cross = np.cross(rot_axis, in_dir)
            out_dir = -cross  # Negate to match DuctElbow convention
            out_dir = out_dir / (np.linalg.norm(out_dir) + 1e-9)
            
            # Outlet position: move R along inlet direction to bend center, then R along outlet direction
            outlet_pos = inlet_pos + in_dir * R + out_dir * R
            outlet_dir = out_dir
        else:
            # Fallback: estimate outlet as inlet + 2R in inlet direction
            outlet_pos = inlet_pos + in_dir * R
            outlet_dir = in_dir
            
    elif seg.duct_type == "Transition" and seg.direction is not None and seg.length:
        d = np.array(seg.direction)
        d = d / (np.linalg.norm(d) + 1e-9)
        outlet_pos = inlet_pos + d * seg.length
        outlet_dir = d
    else:
        outlet_pos = inlet_pos.copy()
        outlet_dir = seg.direction
    
    return inlet_pos, outlet_pos, outlet_dir


def get_all_ports(system) -> Dict[str, PortInfo]:
    """Extract all connection ports from the system."""
    ports = {}
    
    # Classification system
    classification = system._subsystems.get('classification')
    class_offset = np.array(system._subsystems.get('classification_offset', (0, 0, 0)))
    
    if classification:
        class_positions = classification.get_component_positions()
        
        # Venturi ports
        if hasattr(classification, 'venturi'):
            venturi = classification.venturi
            venturi_pos = np.array(class_positions.get('venturi', (0, 0, 0)))
            for port_name, port in venturi.ports.items():
                world_pos = class_offset + venturi_pos + np.array(port.position)
                ports[f'venturi_{port_name}'] = PortInfo(
                    name=port_name,
                    component='Venturi',
                    position=world_pos,
                    direction=np.array(port.direction),
                    diameter=port.diameter
                )
        
        # Bag filter ports
        if hasattr(classification, 'bag_filter'):
            bf = classification.bag_filter
            bf_pos = np.array(classification._component_positions.get('bag_filter', (0, 0, 0)))
            for port_name, port in bf.ports.items():
                world_pos = class_offset + bf_pos + np.array(port.position)
                ports[f'bagfilter_{port_name}'] = PortInfo(
                    name=port_name,
                    component='BagFilter',
                    position=world_pos,
                    direction=np.array(port.direction),
                    diameter=port.diameter
                )
        
        # Cyclone ports
        if hasattr(classification, 'cyclone'):
            cyc = classification.cyclone
            cyc_pos = np.array(class_positions.get('cyclone', (0, 0, 0)))
            for port_name, port in cyc.ports.items():
                world_pos = class_offset + cyc_pos + np.array(port.position)
                ports[f'cyclone_{port_name}'] = PortInfo(
                    name=port_name,
                    component='Cyclone',
                    position=world_pos,
                    direction=np.array(port.direction),
                    diameter=port.diameter
                )
    
    # Feed system
    feed_system = system._subsystems.get('feed_system')
    feed_offset = np.array(system._subsystems.get('feed_system_offset', (0, 0, 0)))
    
    if feed_system:
        feed_positions = feed_system.get_component_positions()
        
        if hasattr(feed_system, 'deagglomerator'):
            deagg = feed_system.deagglomerator
            deagg_pos = np.array(feed_positions.get('deagglomerator', (0, 0, 0)))
            for port_name, port in deagg.ports.items():
                world_pos = feed_offset + deagg_pos + np.array(port.position)
                ports[f'deagglomerator_{port_name}'] = PortInfo(
                    name=port_name,
                    component='Deagglomerator',
                    position=world_pos,
                    direction=np.array(port.direction),
                    diameter=port.diameter
                )
        
        if hasattr(feed_system, 'hopper'):
            hopper = feed_system.hopper
            hopper_pos = np.array(feed_positions.get('hopper', (0, 0, 0)))
            for port_name, port in hopper.ports.items():
                world_pos = feed_offset + hopper_pos + np.array(port.position)
                ports[f'hopper_{port_name}'] = PortInfo(
                    name=port_name,
                    component='Hopper',
                    position=world_pos,
                    direction=np.array(port.direction),
                    diameter=port.diameter
                )
    
    # Air system
    air_system = system._subsystems.get('air_system')
    air_offset = np.array(system._subsystems.get('air_system_offset', (0, 0, 0)))
    
    if air_system:
        if hasattr(air_system, 'dampers') and air_system.dampers:
            for i, (damper, damper_pos) in enumerate(zip(air_system.dampers, air_system._damper_positions)):
                dpos = np.array(damper_pos)
                for port_name, port in damper.ports.items():
                    world_pos = air_offset + dpos + np.array(port.position)
                    ports[f'damper{i+1}_{port_name}'] = PortInfo(
                        name=port_name,
                        component=f'Damper{i+1}',
                        position=world_pos,
                        direction=np.array(port.direction),
                        diameter=port.diameter
                    )
    
    # Silencer
    silencer = system._components.get('silencer')
    if silencer:
        center = np.array(silencer.params.center)
        direction = np.array(silencer.params.direction_normalized)
        length = silencer.params.length
        diameter = silencer.params.diameter
        
        inlet_pos = center - direction * (length / 2)
        outlet_pos = center + direction * (length / 2)
        
        ports['silencer_inlet'] = PortInfo(
            name='inlet', component='Silencer',
            position=inlet_pos, direction=-direction, diameter=diameter
        )
        ports['silencer_outlet'] = PortInfo(
            name='outlet', component='Silencer',
            position=outlet_pos, direction=direction, diameter=diameter
        )
    
    # Exhaust stack
    exhaust_stack = system._components.get('exhaust_stack')
    if exhaust_stack:
        center = np.array(exhaust_stack.params.center)
        diameter = exhaust_stack.params.diameter
        height = exhaust_stack.params.height
        
        ports['stack_inlet'] = PortInfo(
            name='inlet', component='ExhaustStack',
            position=center, direction=np.array([0, -1, 0]), diameter=diameter
        )
        ports['stack_outlet'] = PortInfo(
            name='outlet', component='ExhaustStack',
            position=center + np.array([0, height, 0]),
            direction=np.array([0, 1, 0]), diameter=diameter
        )
    
    return ports


def get_duct_segments(system) -> List[DuctSegment]:
    """Extract all duct segments from the system."""
    segments = []
    
    if hasattr(system, '_duct_connections') and system._duct_connections:
        for i, (duct, pos) in enumerate(system._duct_connections):
            seg = extract_duct_segment(duct, pos, i)
            segments.append(seg)
    
    return segments


def find_nearest_port(position: np.ndarray, ports: Dict[str, PortInfo], 
                      max_distance: float = 1.0) -> Optional[Tuple[str, PortInfo, float]]:
    """Find the nearest port to a position."""
    nearest = None
    min_dist = float('inf')
    
    for name, port in ports.items():
        dist = np.linalg.norm(position - port.position)
        if dist < min_dist and dist < max_distance:
            min_dist = dist
            nearest = (name, port, dist)
    
    return nearest


def analyze_duct_chain(segments: List[DuctSegment], ports: Dict[str, PortInfo]):
    """Analyze duct chains and their connections to ports."""
    print_subheader("DUCT CHAIN ANALYSIS")
    
    if not segments:
        print("  No duct segments found.")
        return
    
    print(f"\n  Total duct segments: {len(segments)}")
    
    # Group segments by proximity (chains)
    # First, find which port each segment starts near
    print("\n  DUCT SEGMENT DETAILS:")
    print(f"  {'#':>3} {'Type':<20} {'Position':<35} {'Properties'}")
    print(f"  {'-'*3} {'-'*20} {'-'*35} {'-'*40}")
    
    for seg in segments:
        props = []
        if seg.diameter:
            props.append(f"D={format_mm(seg.diameter)}")
        if seg.length:
            props.append(f"L={format_mm(seg.length)}")
        if seg.bend_radius:
            props.append(f"R={format_mm(seg.bend_radius)}")
        if seg.angle:
            props.append(f"θ={seg.angle:.0f}°")
        if seg.direction is not None:
            props.append(f"dir={direction_name(seg.direction)}")
        if seg.inlet_direction is not None:
            props.append(f"in={direction_name(seg.inlet_direction)}")
        if seg.inlet_diameter and seg.outlet_diameter:
            props.append(f"{format_mm(seg.inlet_diameter)}→{format_mm(seg.outlet_diameter)}")
        
        props_str = ", ".join(props) if props else "-"
        
        # Compute and show outlet position
        _, outlet_pos, out_dir = compute_duct_endpoint(seg)
        outlet_str = format_vector(outlet_pos)
        
        print(f"  {seg.index+1:>3} {seg.duct_type:<20} IN:{format_vector(seg.position):<32}")
        print(f"      {'':20} OUT:{outlet_str:<32} {props_str}")
    
    # Find start/end connections
    print("\n  PORT PROXIMITY ANALYSIS:")
    
    if segments:
        first_seg = segments[0]
        last_seg = segments[-1]
        
        # Check first segment start
        first_nearest = find_nearest_port(first_seg.position, ports)
        if first_nearest:
            name, port, dist = first_nearest
            print(f"\n  Chain START near: {name}")
            print(f"    Port position:  {format_vector(port.position)}")
            print(f"    Duct position:  {format_vector(first_seg.position)}")
            print(f"    Gap distance:   {format_mm(dist)}")
            print(f"    Port direction: {direction_name(port.direction)}")
            print(f"    Port diameter:  {format_mm(port.diameter)}")
        
        # Check last segment end
        last_inlet, last_outlet, _ = compute_duct_endpoint(last_seg)
        last_nearest = find_nearest_port(last_outlet, ports)
        if last_nearest:
            name, port, dist = last_nearest
            print(f"\n  Chain END near: {name}")
            print(f"    Port position:  {format_vector(port.position)}")
            print(f"    Duct endpoint:  {format_vector(last_outlet)}")
            print(f"    Gap distance:   {format_mm(dist)}")
            print(f"    Port direction: {direction_name(port.direction)}")
            print(f"    Port diameter:  {format_mm(port.diameter)}")


def analyze_connection_path(from_port_key: str, to_port_key: str, 
                            ports: Dict[str, PortInfo], segments: List[DuctSegment]):
    """Analyze the duct path between two ports."""
    if from_port_key not in ports or to_port_key not in ports:
        print(f"  ERROR: Port not found ({from_port_key} or {to_port_key})")
        return
    
    from_port = ports[from_port_key]
    to_port = ports[to_port_key]
    
    print(f"\n  FROM: {from_port.component}.{from_port.name}")
    print(f"    World Position: {format_vector(from_port.position)}")
    print(f"    Direction:      {direction_name(from_port.direction)}")
    print(f"    Diameter:       {format_mm(from_port.diameter)}")
    
    print(f"\n  TO: {to_port.component}.{to_port.name}")
    print(f"    World Position: {format_vector(to_port.position)}")
    print(f"    Direction:      {direction_name(to_port.direction)}")
    print(f"    Diameter:       {format_mm(to_port.diameter)}")
    
    # Calculate deltas
    delta = to_port.position - from_port.position
    distance = np.linalg.norm(delta)
    
    print(f"\n  SPATIAL RELATIONSHIP:")
    print(f"    Delta vector: {format_vector(delta)}")
    print(f"    ΔX: {delta[0]:+.4f} m  ({format_mm(abs(delta[0]))})")
    print(f"    ΔY: {delta[1]:+.4f} m  ({format_mm(abs(delta[1]))})")
    print(f"    ΔZ: {delta[2]:+.4f} m  ({format_mm(abs(delta[2]))})")
    print(f"    Direct distance: {format_mm(distance)}")
    
    # Direction compatibility
    dot = np.dot(from_port.direction, to_port.direction)
    print(f"\n  DIRECTION COMPATIBILITY:")
    print(f"    From direction: {direction_name(from_port.direction)}")
    print(f"    To direction:   {direction_name(to_port.direction)}")
    print(f"    Dot product:    {dot:.3f}")
    print(f"    Status:         {'✓ Anti-parallel (good)' if dot < -0.8 else '⚠ Not aligned (needs turns)'}")
    
    # Diameter compatibility
    d_diff = abs(from_port.diameter - to_port.diameter)
    d_ratio = min(from_port.diameter, to_port.diameter) / max(from_port.diameter, to_port.diameter)
    
    print(f"\n  DIAMETER COMPATIBILITY:")
    print(f"    From diameter: {format_mm(from_port.diameter)}")
    print(f"    To diameter:   {format_mm(to_port.diameter)}")
    print(f"    Difference:    {format_mm(d_diff)}")
    print(f"    Ratio:         {d_ratio:.2f}")
    print(f"    Status:        {'✓ Match' if d_diff < 0.001 else '⚠ Needs transition'}")
    
    # Find ducts that might connect these ports
    nearby_start = [s for s in segments if np.linalg.norm(s.position - from_port.position) < 0.5]
    nearby_end = []
    for s in segments:
        _, outlet, _ = compute_duct_endpoint(s)
        if np.linalg.norm(outlet - to_port.position) < 0.5:
            nearby_end.append(s)
    
    if nearby_start:
        print(f"\n  DUCTS NEAR START ({len(nearby_start)}):")
        for s in nearby_start:
            dist = np.linalg.norm(s.position - from_port.position)
            print(f"    [{s.index+1}] {s.duct_type} at {format_mm(dist)} away")
    
    if nearby_end:
        print(f"\n  DUCTS NEAR END ({len(nearby_end)}):")
        for s in nearby_end:
            _, outlet, _ = compute_duct_endpoint(s)
            dist = np.linalg.norm(outlet - to_port.position)
            print(f"    [{s.index+1}] {s.duct_type} at {format_mm(dist)} away")


def print_all_ports(ports: Dict[str, PortInfo]):
    """Print all ports in a formatted table."""
    print_subheader("ALL SYSTEM PORTS")
    
    # Group by component
    by_component: Dict[str, List[Tuple[str, PortInfo]]] = {}
    for key, port in ports.items():
        comp = port.component
        if comp not in by_component:
            by_component[comp] = []
        by_component[comp].append((key, port))
    
    for comp, port_list in sorted(by_component.items()):
        print(f"\n  {comp}:")
        for key, port in port_list:
            print(f"    {port.name:20s} D={format_mm(port.diameter):>10s}  "
                  f"dir={direction_name(port.direction):30s}  "
                  f"@ {format_vector(port.position)}")


def print_subsystem_offsets(system):
    """Print subsystem offset positions."""
    print_subheader("SUBSYSTEM OFFSETS")
    
    offsets = [
        ('classification', system._subsystems.get('classification_offset', (0, 0, 0))),
        ('feed_system', system._subsystems.get('feed_system_offset', (0, 0, 0))),
        ('air_system', system._subsystems.get('air_system_offset', (0, 0, 0))),
    ]
    
    for name, offset in offsets:
        offset = np.array(offset)
        print(f"  {name:20s}: {format_vector(offset)}")
        print(f"    X = {offset[0]:+.4f} m  (horizontal toward deagglomerator)")
        print(f"    Y = {offset[1]:+.4f} m  (vertical height)")
        print(f"    Z = {offset[2]:+.4f} m  (depth from classifier)")


def main():
    """Run complete system debug analysis."""
    from airclassifier.geometry.assembly.complete_system import create_core_connections_system
    
    print("\n" + "=" * 80)
    print("  COMPLETE SYSTEM CONNECTION DIAGNOSTICS")
    print("  Coordinate System: X(horizontal) Y(vertical) Z(depth)")
    print("=" * 80)
    
    # Create the system
    print("\nCreating core connections system...")
    system = create_core_connections_system()
    
    # System configuration
    print_header("SYSTEM CONFIGURATION")
    p = system.params
    print(f"  Throughput:       {p.throughput_kg_h} kg/h")
    print(f"  Air Flow:         {p.air_flow_m3_h} m³/h")
    print(f"  Main Duct Dia:    {format_mm(p.main_duct_diameter)}")
    print(f"\n  Enabled Systems:")
    print(f"    Feed System:      {'Yes' if p.include_feed_system else 'No'}")
    print(f"    Air System:       {'Yes' if p.include_air_system else 'No'}")
    print(f"    Ductwork:         {'Yes' if p.include_ductwork else 'No'}")
    print(f"    Exhaust:          {'Yes' if p.include_exhaust else 'No'}")
    
    # Subsystem offsets
    print_subsystem_offsets(system)
    
    # Extract all ports
    ports = get_all_ports(system)
    print_all_ports(ports)
    
    # Extract all duct segments
    segments = get_duct_segments(system)
    
    # Analyze duct chains
    analyze_duct_chain(segments, ports)
    
    # Analyze specific connections
    print_header("CONNECTION PATH ANALYSIS")
    
    # 1. Air System → Venturi air inlet
    print_subheader("Connection 1: Air System → Venturi Air Inlet")
    analyze_connection_path('damper1_outlet', 'venturi_air_inlet', ports, segments)
    
    # 2. Feed System → Venturi solids inlet
    print_subheader("Connection 2: Feed System → Venturi Solids Inlet")
    analyze_connection_path('deagglomerator_outlet', 'venturi_solids_inlet', ports, segments)
    
    # 3. Bag Filter → Silencer
    print_subheader("Connection 3: Bag Filter → Silencer")
    analyze_connection_path('bagfilter_clean_air_outlet', 'silencer_inlet', ports, segments)
    
    # 4. Silencer → Exhaust Stack
    print_subheader("Connection 4: Silencer → Exhaust Stack")
    analyze_connection_path('silencer_outlet', 'stack_inlet', ports, segments)
    
    # Gap analysis between consecutive ducts - identify separate chains
    print_header("DUCT CHAIN SEPARATION & GAP ANALYSIS")
    
    if len(segments) > 1:
        # First, identify chain breaks (gaps > 500mm indicate separate duct chains)
        CHAIN_BREAK_THRESHOLD = 0.5  # 500mm
        
        chain_breaks = []
        gaps_data = []
        
        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]
            
            _, outlet_a, out_dir_a = compute_duct_endpoint(seg_a)
            inlet_b = seg_b.position
            
            gap = np.linalg.norm(outlet_a - inlet_b)
            gaps_data.append((i, seg_a, seg_b, outlet_a, inlet_b, gap))
            
            if gap > CHAIN_BREAK_THRESHOLD:
                chain_breaks.append(i)
        
        # Define chains
        chains = []
        start_idx = 0
        for break_idx in chain_breaks:
            chains.append((start_idx, break_idx + 1))
            start_idx = break_idx + 1
        chains.append((start_idx, len(segments)))
        
        print(f"\n  Found {len(chains)} separate duct chains:")
        for chain_num, (start, end) in enumerate(chains):
            chain_segs = segments[start:end]
            print(f"\n  CHAIN {chain_num + 1}: Segments [{start+1}..{end}] ({end-start} segments)")
            
            # Find what ports this chain connects
            first_seg = chain_segs[0]
            last_seg = chain_segs[-1]
            _, last_outlet, _ = compute_duct_endpoint(last_seg)
            
            start_port = find_nearest_port(first_seg.position, ports, max_distance=0.5)
            end_port = find_nearest_port(last_outlet, ports, max_distance=0.5)
            
            if start_port:
                print(f"    Start: near {start_port[0]} ({format_mm(start_port[2])} away)")
            if end_port:
                print(f"    End:   near {end_port[0]} ({format_mm(end_port[2])} away)")
            
            # Show gaps within this chain
            print(f"\n    Segment-to-segment gaps within chain:")
            chain_gaps = [(i, d) for i, sa, sb, oa, ib, d in gaps_data if start <= i < end - 1]
            
            all_ok = True
            for gap_idx, gap_dist in chain_gaps:
                seg_a = segments[gap_idx]
                seg_b = segments[gap_idx + 1]
                status = "✓" if gap_dist < 0.02 else "⚠"
                if gap_dist >= 0.02:
                    all_ok = False
                print(f"      [{gap_idx+1}→{gap_idx+2}] {seg_a.duct_type[:12]:<12} → "
                      f"{seg_b.duct_type[:12]:<12} gap={format_mm(gap_dist):>10} {status}")
            
            if all_ok and chain_gaps:
                print(f"    All {len(chain_gaps)} connections within tolerance (< 20mm)")
        
        # Chain break summary
        if chain_breaks:
            print(f"\n  Chain breaks detected at segments: {[b+1 for b in chain_breaks]}")
    
    # Bill of Materials
    print_header("DUCTWORK SUMMARY")
    
    if segments:
        duct_counts: Dict[str, int] = {}
        total_length = 0.0
        
        for seg in segments:
            duct_counts[seg.duct_type] = duct_counts.get(seg.duct_type, 0) + 1
            if seg.length:
                total_length += seg.length
        
        print(f"\n  {'Component':<25} {'Count':>8}")
        print(f"  {'-'*25} {'-'*8}")
        for item, qty in sorted(duct_counts.items()):
            print(f"  {item:<25} {qty:>8}")
        print(f"  {'-'*25} {'-'*8}")
        print(f"  {'TOTAL':<25} {sum(duct_counts.values()):>8}")
        print(f"\n  Total straight duct length: {format_mm(total_length)}")
    
    # Build mesh for bounds
    print_header("GEOMETRY BOUNDS")
    try:
        verts, indices = system.build_mesh()
        bounds_min, bounds_max = system.get_bounds()
        dims = bounds_max - bounds_min
        
        print(f"  Vertices:     {len(verts):,}")
        print(f"  Triangles:    {len(indices)//3:,}")
        print(f"  Bounds Min:   {format_vector(bounds_min)}")
        print(f"  Bounds Max:   {format_vector(bounds_max)}")
        print(f"  Dimensions:   {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
    except Exception as e:
        print(f"  Could not compute bounds: {e}")
    
    print("\n" + "=" * 80)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
