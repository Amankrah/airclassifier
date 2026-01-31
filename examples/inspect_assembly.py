#!/usr/bin/env python
"""
Assembly Inspection Tool

Inspects the positional fitting of components in assemblies to verify
proper port-to-port alignment and connections.

This tool:
1. Validates connection alignment between components
2. Analyzes port-to-port connections
3. Generates detailed reports on assembly quality

Usage:
    python examples/inspect_assembly.py                  # Interactive menu
    python examples/inspect_assembly.py --feed           # Inspect feed system
    python examples/inspect_assembly.py --air            # Inspect air system
    python examples/inspect_assembly.py --feed --detailed # Detailed feed analysis
    python examples/inspect_assembly.py --air --detailed  # Detailed air analysis
    python examples/inspect_assembly.py --drop           # Feed system vertical drop
    python examples/inspect_assembly.py --flow           # Air system flow path
    python examples/inspect_assembly.py --validate       # Validate all assemblies
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from airclassifier.geometry import (
    ConnectionPort,
    PortType,
    calculate_alignment,
    validate_assembly_connections,
    print_connection_report,
)
from airclassifier.geometry.assembly import (
    create_standard_feed_system,
    create_standard_air_system,
    create_standard_classification_system,
    create_complete_classifier_system,
)


def inspect_feed_system():
    """Inspect feed system component positioning."""
    print("\n" + "=" * 70)
    print("FEED SYSTEM ASSEMBLY INSPECTION")
    print("=" * 70)
    
    feed = create_standard_feed_system()
    
    # Print component positions
    print("\n--- Component Positions ---")
    positions = feed.get_component_positions()
    for name, pos in positions.items():
        print(f"{name:20s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m")
    
    # Print port locations
    print("\n--- Connection Ports ---")
    
    components = {
        'hopper': (feed.hopper, feed._hopper_position),
        'airlock': (feed.airlock, feed._airlock_position),
        'feeder': (feed.feeder, feed._feeder_position),
        'deagglomerator': (feed.deagglomerator, feed._deagglomerator_position),
    }
    
    for comp_name, (comp, pos) in components.items():
        if hasattr(comp, 'ports'):
            print(f"\n  {comp_name.upper()}:")
            for port_name, port in comp.ports.items():
                world_pos = port.get_world_position(pos)
                print(f"    {port_name:12s}: pos=({world_pos[0]:7.3f}, {world_pos[1]:7.3f}, {world_pos[2]:7.3f})")
                print(f"                  dir=({port.direction[0]:5.2f}, {port.direction[1]:5.2f}, {port.direction[2]:5.2f})")
                print(f"                  D={port.diameter*1000:.1f}mm, type={port.port_type.value}")
    
    # Validate connections
    print("\n--- Connection Validation ---")
    feed.print_connection_report()
    
    # Show final outlet position
    outlet_pos, outlet_dir = feed.get_system_outlet()
    print(f"\n--- System Outlet (to Classifier) ---")
    print(f"Position: ({outlet_pos[0]:.3f}, {outlet_pos[1]:.3f}, {outlet_pos[2]:.3f}) m")
    print(f"Direction: ({outlet_dir[0]:.2f}, {outlet_dir[1]:.2f}, {outlet_dir[2]:.2f})")
    
    return feed


def inspect_component_ports(component, name: str):
    """Inspect ports of a single component."""
    print(f"\n--- {name} Ports ---")
    
    if not hasattr(component, 'ports'):
        print(f"  No ports defined for {name}")
        return
    
    for port_name, port in component.ports.items():
        print(f"\n  {port_name}:")
        print(f"    Position: ({port.position[0]:.4f}, {port.position[1]:.4f}, {port.position[2]:.4f}) m")
        print(f"    Direction: ({port.direction[0]:.2f}, {port.direction[1]:.2f}, {port.direction[2]:.2f})")
        print(f"    Diameter: {port.diameter*1000:.1f} mm")
        print(f"    Type: {port.port_type.value}")
        print(f"    Area: {port.area*10000:.2f} cm²")
        print(f"    Compatible: {[t.value for t in port.compatible_types]}")


def check_alignment_between_components(comp_a, port_a_name, comp_b, port_b_name, gap=0.0):
    """Check alignment between two component ports."""
    port_a = comp_a.ports[port_a_name]
    port_b = comp_b.ports[port_b_name]
    
    alignment = calculate_alignment(
        source_port=port_a,
        target_port=port_b,
        source_position=(0, 0, 0),  # Assume comp_a at origin
        gap=gap
    )
    
    print(f"\n--- Alignment: {port_a_name} -> {port_b_name} ---")
    print(f"  Target position offset: ({alignment.position_offset[0]:.4f}, "
          f"{alignment.position_offset[1]:.4f}, {alignment.position_offset[2]:.4f}) m")
    print(f"  Gap: {alignment.gap*1000:.1f} mm")
    print(f"  Aligned: {alignment.is_aligned}")
    print(f"  Message: {alignment.message}")
    
    # Check compatibility
    compatible, reason = port_a.is_compatible(port_b)
    print(f"  Port compatible: {compatible} - {reason}")
    
    return alignment


def detailed_feed_system_analysis():
    """Run detailed analysis of ACTUAL ASSEMBLY component fitting."""
    from airclassifier.geometry.assembly import create_standard_feed_system
    
    print("\n" + "=" * 70)
    print("DETAILED FEED SYSTEM COMPONENT ANALYSIS (FROM ASSEMBLY)")
    print("=" * 70)
    
    # Use actual assembly components (with matched diameters)
    feed = create_standard_feed_system()
    hopper = feed.hopper
    airlock = feed.airlock
    feeder = feed.feeder
    deagglomerator = feed.deagglomerator
    
    # Inspect each component's ports
    inspect_component_ports(hopper, "Feed Hopper (Assembly)")
    inspect_component_ports(airlock, "Rotary Airlock (Assembly)")
    inspect_component_ports(feeder, "Screw Feeder (Assembly)")
    inspect_component_ports(deagglomerator, "De-agglomerator (Assembly)")
    
    # Check alignments with assembly's gap
    gap = feed.params.component_spacing
    
    print("\n" + "=" * 70)
    print("PORT-TO-PORT ALIGNMENT ANALYSIS")
    print("=" * 70)
    
    print(f"\n(Using assembly gap: {gap*1000:.1f} mm)")
    
    print("\n1. Hopper discharge -> Airlock inlet")
    check_alignment_between_components(hopper, 'discharge', airlock, 'inlet', gap=gap)
    
    print("\n2. Airlock outlet -> Feeder inlet")
    check_alignment_between_components(airlock, 'outlet', feeder, 'inlet', gap=gap)
    
    print("\n3. Feeder outlet -> Deagglomerator inlet")
    check_alignment_between_components(feeder, 'outlet', deagglomerator, 'inlet', gap=gap)
    
    # Diameter matching analysis
    print("\n" + "=" * 70)
    print("PORT DIAMETER MATCHING (ASSEMBLY)")
    print("=" * 70)
    
    connections = [
        ("Hopper discharge", hopper.ports['discharge'].diameter,
         "Airlock inlet", airlock.ports['inlet'].diameter),
        ("Airlock outlet", airlock.ports['outlet'].diameter,
         "Feeder inlet", feeder.ports['inlet'].diameter),
        ("Feeder outlet", feeder.ports['outlet'].diameter,
         "Deagglomerator inlet", deagglomerator.ports['inlet'].diameter),
    ]
    
    print(f"\n{'Connection':<30} {'Source D':<12} {'Target D':<12} {'Match':<10}")
    print("-" * 70)
    
    for src_name, src_d, tgt_name, tgt_d in connections:
        match_pct = min(src_d, tgt_d) / max(src_d, tgt_d) * 100
        status = "[OK]" if match_pct > 70 else "[!] Mismatch"
        print(f"{src_name:>15} -> {tgt_name:<12} {src_d*1000:8.1f}mm {tgt_d*1000:8.1f}mm {status:>12} ({match_pct:.0f}%)")


def inspect_air_system():
    """Inspect air system component positioning."""
    print("\n" + "=" * 70)
    print("AIR SYSTEM ASSEMBLY INSPECTION")
    print("=" * 70)
    
    air = create_standard_air_system()
    
    # Print component positions
    print("\n--- Component Positions ---")
    positions = air.get_component_positions()
    for name, pos in positions.items():
        print(f"{name:20s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m")
    
    # Print port locations
    print("\n--- Connection Ports ---")
    
    components = {
        'inlet_filter': (air.inlet_filter, air._filter_position),
        'blower': (air.blower, air._blower_position),
    }
    for i, (damper, pos) in enumerate(zip(air.dampers, air._damper_positions)):
        components[f'damper_{i}'] = (damper, pos)
    
    for comp_name, (comp, pos) in components.items():
        if hasattr(comp, 'ports'):
            print(f"\n  {comp_name.upper()}:")
            for port_name, port in comp.ports.items():
                world_pos = port.get_world_position(pos)
                print(f"    {port_name:12s}: pos=({world_pos[0]:7.3f}, {world_pos[1]:7.3f}, {world_pos[2]:7.3f})")
                print(f"                  dir=({port.direction[0]:5.2f}, {port.direction[1]:5.2f}, {port.direction[2]:5.2f})")
                if port.port_type.value == "rectangular":
                    print(f"                  W={port.width*1000:.1f}mm, H={port.height*1000:.1f}mm, type={port.port_type.value}")
                else:
                    print(f"                  D={port.diameter*1000:.1f}mm, type={port.port_type.value}")
    
    # Print duct sections
    print("\n--- Duct Sections ---")
    print(f"  Total duct sections: {len(air._duct_sections)}")
    for i, (duct, pos) in enumerate(air._duct_sections):
        duct_type = type(duct).__name__
        print(f"  {i+1}. {duct_type}: pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
        
        # Different info based on duct type
        if hasattr(duct.params, 'diameter') and hasattr(duct.params, 'length'):
            print(f"           D={duct.params.diameter*1000:.1f}mm, L={duct.params.length*1000:.1f}mm")
        elif hasattr(duct.params, 'diameter') and hasattr(duct.params, 'angle'):
            # Elbow
            print(f"           D={duct.params.diameter*1000:.1f}mm, angle={duct.params.angle}°, R={duct.params.bend_radius*1000:.1f}mm")
        elif hasattr(duct.params, 'rect_width'):
            # Rect-to-round transition
            print(f"           Rect: {duct.params.rect_width*1000:.1f}x{duct.params.rect_height*1000:.1f}mm -> Round: D={duct.params.round_diameter*1000:.1f}mm, L={duct.params.length*1000:.1f}mm")
    
    # Print summary
    air.print_summary()
    
    return air


def detailed_air_system_analysis():
    """Run detailed analysis of air system component fitting."""
    print("\n" + "=" * 70)
    print("DETAILED AIR SYSTEM COMPONENT ANALYSIS")
    print("=" * 70)
    
    air = create_standard_air_system()
    inlet_filter = air.inlet_filter
    blower = air.blower
    dampers = air.dampers
    
    # Inspect each component's ports
    inspect_component_ports(inlet_filter, "Inlet Air Filter")
    inspect_component_ports(blower, "Centrifugal Blower")
    for i, damper in enumerate(dampers):
        inspect_component_ports(damper, f"Flow Damper {i+1}")
    
    # Check alignments with assembly's gap
    gap = air.params.component_spacing
    
    print("\n" + "=" * 70)
    print("PORT-TO-PORT ALIGNMENT ANALYSIS")
    print("=" * 70)
    
    print(f"\n(Using assembly gap: {gap*1000:.1f} mm)")
    
    print("\n1. Filter outlet -> Blower inlet")
    # Check if connected via elbow (perpendicular ports)
    has_elbow = any(type(duct).__name__ == 'DuctElbow' for duct, _ in air._duct_sections)
    if has_elbow:
        print("\n--- Alignment: outlet -> inlet (via 90 deg elbow) ---")
        filter_out = inlet_filter.ports['outlet']
        blower_in = blower.ports['inlet']
        # Check perpendicularity (dot product ~0)
        dot = sum(a*b for a, b in zip(filter_out.direction, blower_in.direction))
        if abs(dot) < 0.1:
            print(f"  Filter outlet direction: ({filter_out.direction[0]:.2f}, {filter_out.direction[1]:.2f}, {filter_out.direction[2]:.2f})")
            print(f"  Blower inlet direction:  ({blower_in.direction[0]:.2f}, {blower_in.direction[1]:.2f}, {blower_in.direction[2]:.2f})")
            print(f"  Ports are perpendicular (dot={dot:.2f}) - connected via 90 deg elbow")
            print(f"  Aligned: True (via elbow)")
        else:
            check_alignment_between_components(inlet_filter, 'outlet', blower, 'inlet', gap=gap)
    else:
        check_alignment_between_components(inlet_filter, 'outlet', blower, 'inlet', gap=gap)
    
    if dampers:
        print("\n2. Blower outlet -> Damper 1 inlet")
        check_alignment_between_components(blower, 'outlet', dampers[0], 'inlet', gap=gap)
        
        for i in range(len(dampers) - 1):
            print(f"\n{i+3}. Damper {i+1} outlet -> Damper {i+2} inlet")
            check_alignment_between_components(dampers[i], 'outlet', dampers[i+1], 'inlet', gap=gap)
    
    # Diameter matching analysis
    print("\n" + "=" * 70)
    print("PORT DIAMETER MATCHING")
    print("=" * 70)
    
    # Check for rect-to-round transition
    transition_output_d = None
    for duct, _ in air._duct_sections:
        if type(duct).__name__ == 'RectToRoundTransition':
            transition_output_d = duct.params.round_diameter
            break
    
    connections = [
        ("Filter outlet", inlet_filter.ports['outlet'].diameter,
         "Blower inlet", blower.ports['inlet'].diameter, False),
    ]
    
    if dampers:
        # Blower outlet is rectangular, convert to equivalent diameter
        blower_out = blower.ports['outlet']
        equiv_d = blower_out.diameter  # Already calculated as equivalent
        damper_d = dampers[0].ports['inlet'].diameter
        
        # Check if transition bridges this connection
        has_transition = (transition_output_d is not None and 
                         abs(transition_output_d - damper_d) < 0.001)
        
        connections.append(
            ("Blower outlet (eq)", equiv_d,
             "Damper 1 inlet", damper_d, has_transition)
        )
        
        for i in range(len(dampers) - 1):
            connections.append(
                (f"Damper {i+1} outlet", dampers[i].ports['outlet'].diameter,
                 f"Damper {i+2} inlet", dampers[i+1].ports['inlet'].diameter, False)
            )
    
    print(f"\n{'Connection':<35} {'Source D':<12} {'Target D':<12} {'Match':<15}")
    print("-" * 75)
    
    for src_name, src_d, tgt_name, tgt_d, via_transition in connections:
        match_pct = min(src_d, tgt_d) / max(src_d, tgt_d) * 100 if max(src_d, tgt_d) > 0 else 0
        
        if via_transition:
            status = "[OK] via transition"
        elif match_pct > 90:
            status = f"[OK] ({match_pct:.0f}%)"
        elif match_pct > 70:
            status = f"[~] ({match_pct:.0f}%)"
        else:
            status = f"[!] Mismatch ({match_pct:.0f}%)"
        
        print(f"{src_name:>17} -> {tgt_name:<15} {src_d*1000:8.1f}mm {tgt_d*1000:8.1f}mm {status:>18}")


def calculate_air_system_flow_path():
    """Calculate flow path length through the air system."""
    air = create_standard_air_system()
    
    print("\n" + "=" * 70)
    print("AIR SYSTEM FLOW PATH ANALYSIS")
    print("=" * 70)
    
    positions = air.get_component_positions()
    
    # Get X positions (horizontal flow direction)
    filter_inlet_x = positions['inlet_filter'][0] - air.inlet_filter.params.housing_depth/2
    filter_outlet_x = positions['inlet_filter'][0] + air.inlet_filter.params.housing_depth/2
    
    blower_inlet_x = positions['blower'][0] + air.blower.ports['inlet'].position[0]
    blower_outlet_x = positions['blower'][0] + air.blower.ports['outlet'].position[0]
    
    print(f"\nAir flow path (X positions):")
    print(f"  Filter inlet:      X = {filter_inlet_x*1000:8.1f} mm  (Ambient air entry)")
    print(f"  Filter outlet:     X = {filter_outlet_x*1000:8.1f} mm")
    print(f"  Blower inlet:      X = {blower_inlet_x*1000:8.1f} mm")
    print(f"  Blower outlet:     X = {blower_outlet_x*1000:8.1f} mm")
    
    if air.dampers:
        for i, (damper, pos) in enumerate(zip(air.dampers, air._damper_positions)):
            damper_inlet_x = pos[0] + damper.ports['inlet'].position[0]
            damper_outlet_x = pos[0] + damper.ports['outlet'].position[0]
            print(f"  Damper {i+1} inlet:    X = {damper_inlet_x*1000:8.1f} mm")
            print(f"  Damper {i+1} outlet:   X = {damper_outlet_x*1000:8.1f} mm")
    
    # Calculate total flow path length
    last_damper_pos = air._damper_positions[-1] if air.dampers else positions['blower']
    last_component = air.dampers[-1] if air.dampers else air.blower
    last_outlet = last_component.ports['outlet']
    
    total_length = (last_damper_pos[0] + last_outlet.position[0]) - filter_inlet_x
    print(f"\n  Total flow path length: {total_length*1000:.1f} mm ({total_length:.3f} m)")
    
    # Print duct section lengths
    print(f"\nDuct sections:")
    total_duct_length = 0
    for i, (duct, pos) in enumerate(air._duct_sections):
        duct_type = type(duct).__name__
        if hasattr(duct.params, 'length'):
            print(f"  Section {i+1} ({duct_type}): {duct.params.length*1000:.1f} mm")
            total_duct_length += duct.params.length
        elif hasattr(duct.params, 'bend_radius'):
            # Elbow - arc length
            arc_length = duct.params.bend_radius * np.radians(duct.params.angle)
            print(f"  Section {i+1} ({duct_type}): arc {arc_length*1000:.1f} mm (R={duct.params.bend_radius*1000:.1f}mm)")
            total_duct_length += arc_length
        else:
            print(f"  Section {i+1} ({duct_type})")
    print(f"  Total ductwork: {total_duct_length*1000:.1f} mm")
    
    # Estimate pressure drop
    perf = air.get_performance_summary()
    print(f"\n--- Performance ---")
    print(f"  Design flow rate: {perf['design_flow_rate_m3_h']:.0f} m³/h")
    print(f"  Blower pressure:  {perf['blower_pressure_rise_Pa']:.0f} Pa")
    print(f"  Blower power:     {perf['blower_power_kW']:.2f} kW")
    print(f"  Blower efficiency: {perf['blower_efficiency']*100:.0f}%")
    print(f"  Estimated system dP: {perf['estimated_system_dp_Pa']:.0f} Pa")


def calculate_total_drop_height():
    """Calculate the total vertical drop through the feed system."""
    feed = create_standard_feed_system()
    
    print("\n" + "=" * 70)
    print("VERTICAL DROP ANALYSIS")
    print("=" * 70)
    
    positions = feed.get_component_positions()
    
    # Get Y positions (vertical)
    hopper_top = positions['hopper'][1] + feed.hopper.params.total_height
    hopper_discharge = positions['hopper'][1]
    airlock_inlet = positions['airlock'][1] + feed.airlock.ports['inlet'].position[1]
    airlock_outlet = positions['airlock'][1] + feed.airlock.ports['outlet'].position[1]
    feeder_inlet = positions['feeder'][1] + feed.feeder.ports['inlet'].position[1]
    feeder_outlet = positions['feeder'][1] + feed.feeder.ports['outlet'].position[1]
    deagg_inlet = positions['deagglomerator'][1] + feed.deagglomerator.ports['inlet'].position[1]
    deagg_outlet = positions['deagglomerator'][1] + feed.deagglomerator.ports['outlet'].position[1]
    
    print(f"\nMaterial flow path (Y positions):")
    print(f"  Hopper top:        Y = {hopper_top*1000:8.1f} mm")
    print(f"  Hopper discharge:  Y = {hopper_discharge*1000:8.1f} mm")
    print(f"  Airlock inlet:     Y = {airlock_inlet*1000:8.1f} mm")
    print(f"  Airlock outlet:    Y = {airlock_outlet*1000:8.1f} mm")
    print(f"  Feeder inlet:      Y = {feeder_inlet*1000:8.1f} mm")
    print(f"  Feeder outlet:     Y = {feeder_outlet*1000:8.1f} mm")
    print(f"  Deagg inlet:       Y = {deagg_inlet*1000:8.1f} mm")
    print(f"  Deagg outlet:      Y = {deagg_outlet*1000:8.1f} mm")
    
    total_drop = hopper_top - deagg_outlet
    print(f"\n  Total vertical drop: {total_drop*1000:.1f} mm ({total_drop:.3f} m)")
    
    # Check for upward flows (problems)
    transitions = [
        ("Hopper->Airlock", hopper_discharge - airlock_inlet),
        ("Airlock->Feeder", airlock_outlet - feeder_inlet),
        ("Feeder->Deagg", feeder_outlet - deagg_inlet),
    ]
    
    print(f"\nGravity flow check:")
    for name, drop in transitions:
        if drop < 0:
            print(f"  ⚠ {name}: Material must flow UP {-drop*1000:.1f} mm!")
        else:
            print(f"  ✓ {name}: Drop of {drop*1000:.1f} mm")


def run_all_validations():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ASSEMBLY VALIDATION")
    print("=" * 70)
    
    # Feed system
    print("\n[1/3] Validating Feed System...")
    try:
        feed = create_standard_feed_system()
        results = feed.validate_connections()
        valid = sum(1 for r in results if r.get('is_valid', False))
        total = len(results)
        print(f"       Result: {valid}/{total} connections valid")
        if valid < total:
            feed.print_connection_report()
    except Exception as e:
        print(f"       ERROR: {e}")
    
    # Air system
    print("\n[2/3] Validating Air System...")
    try:
        air = create_standard_air_system()
        air.build_mesh()
        bounds = air.get_bounds()
        extent = air.get_system_extent()
        print(f"       System bounds: ({bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}) to ({bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f})")
        print(f"       System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print(f"       Components: Filter + Blower + {len(air.dampers)} Dampers")
        print(f"       Duct sections: {len(air._duct_sections)}")
        perf = air.get_performance_summary()
        print(f"       Flow rate: {perf['design_flow_rate_m3_h']:.0f} m³/h")
        print(f"       Pressure rise: {perf['blower_pressure_rise_Pa']:.0f} Pa")
    except Exception as e:
        print(f"       ERROR: {e}")
    
    # Complete system (if available)
    print("\n[3/3] Validating Complete System...")
    try:
        complete = create_complete_classifier_system()
        # Build mesh to ensure all components are created
        complete.build_mesh()
        bounds = complete.get_bounds()
        print(f"       System bounds: {bounds[0]} to {bounds[1]}")
        print(f"       Subsystems: {len(complete.get_all_subsystem_names())}")
        print(f"       Components: {len(complete.get_all_component_names())}")
        print(f"       Instruments: {len(complete.get_all_instrument_names())}")
    except Exception as e:
        print(f"       ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


def interactive_menu():
    """Run interactive menu."""
    while True:
        print("\n" + "=" * 60)
        print("ASSEMBLY INSPECTION TOOL")
        print("=" * 60)
        print("\n--- Feed System ---")
        print("  1. Feed System - Quick Inspection")
        print("  2. Feed System - Detailed Analysis")
        print("  3. Feed System - Vertical Drop Analysis")
        print("\n--- Air System ---")
        print("  4. Air System - Quick Inspection")
        print("  5. Air System - Detailed Analysis")
        print("  6. Air System - Flow Path Analysis")
        print("\n--- General ---")
        print("  7. Run All Validations")
        print("  0. Exit")
        print()
        
        try:
            choice = input("Enter choice (0-7): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            inspect_feed_system()
        elif choice == "2":
            detailed_feed_system_analysis()
        elif choice == "3":
            calculate_total_drop_height()
        elif choice == "4":
            inspect_air_system()
        elif choice == "5":
            detailed_air_system_analysis()
        elif choice == "6":
            calculate_air_system_flow_path()
        elif choice == "7":
            run_all_validations()
        else:
            print("Invalid choice. Please enter 0-7.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect assembly component positioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_assembly.py                # Interactive menu
  python inspect_assembly.py --feed         # Inspect feed system
  python inspect_assembly.py --air          # Inspect air system
  python inspect_assembly.py --validate     # Validate all assemblies
  python inspect_assembly.py --feed --detailed  # Detailed feed analysis
  python inspect_assembly.py --air --detailed   # Detailed air analysis
        """
    )
    
    # Feed system options
    parser.add_argument("--feed", "-f", action="store_true",
                       help="Inspect feed system")
    parser.add_argument("--drop", action="store_true",
                       help="Feed system vertical drop analysis")
    
    # Air system options
    parser.add_argument("--air", "-a", action="store_true",
                       help="Inspect air system")
    parser.add_argument("--flow", action="store_true",
                       help="Air system flow path analysis")
    
    # General options
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Detailed component analysis (use with --feed or --air)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Run all validations")
    
    args = parser.parse_args()
    
    if not any([args.feed, args.air, args.drop, args.flow, args.validate]):
        interactive_menu()
        return
    
    # Feed system inspections
    if args.feed:
        if args.detailed:
            detailed_feed_system_analysis()
        else:
            inspect_feed_system()
    
    if args.drop:
        calculate_total_drop_height()
    
    # Air system inspections
    if args.air:
        if args.detailed:
            detailed_air_system_analysis()
        else:
            inspect_air_system()
    
    if args.flow:
        calculate_air_system_flow_path()
    
    # Validations
    if args.validate:
        run_all_validations()


if __name__ == "__main__":
    main()
