#!/usr/bin/env python
"""
Assembly Inspection Tool

Inspects the positional fitting of components in assemblies to verify
proper port-to-port alignment and connections.

This tool:
1. Validates connection alignment between components
2. Visualizes components with connection ports highlighted
3. Generates detailed reports on assembly quality

Usage:
    python examples/inspect_assembly.py                  # Interactive menu
    python examples/inspect_assembly.py --feed           # Inspect feed system
    python examples/inspect_assembly.py --complete       # Inspect complete system
    python examples/inspect_assembly.py --validate       # Validate all and report
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
    print("\n[1/2] Validating Feed System...")
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
    
    # Complete system (if available)
    print("\n[2/2] Validating Complete System...")
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
        print("\nSelect inspection:")
        print("  1. Feed System - Quick Inspection")
        print("  2. Feed System - Detailed Analysis")
        print("  3. Feed System - Vertical Drop Analysis")
        print("  4. Individual Component Ports")
        print("  5. Run All Validations")
        print("  0. Exit")
        print()
        
        try:
            choice = input("Enter choice (0-5): ").strip()
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
            detailed_feed_system_analysis()
        elif choice == "5":
            run_all_validations()
        else:
            print("Invalid choice. Please enter 0-5.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect assembly component positioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--feed", "-f", action="store_true",
                       help="Inspect feed system")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Detailed component analysis")
    parser.add_argument("--drop", action="store_true",
                       help="Vertical drop analysis")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Run all validations")
    
    args = parser.parse_args()
    
    if not any([args.feed, args.detailed, args.drop, args.validate]):
        interactive_menu()
        return
    
    if args.feed:
        inspect_feed_system()
    
    if args.detailed:
        detailed_feed_system_analysis()
    
    if args.drop:
        calculate_total_drop_height()
    
    if args.validate:
        run_all_validations()


if __name__ == "__main__":
    main()
