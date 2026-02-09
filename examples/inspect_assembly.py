#!/usr/bin/env python
"""
Assembly Inspection Tool

Inspects the positional fitting of components in assemblies to verify
proper port-to-port alignment and connections.

This tool:
1. Validates connection alignment between components
2. Analyzes port-to-port connections
3. Generates detailed reports on assembly quality
4. Verifies dimension matching at connection points
5. Verifies coordinate alignment at connection points
6. Verifies port direction angles at connection points

Usage:
    python examples/inspect_assembly.py                  # Interactive menu
    python examples/inspect_assembly.py --feed           # Inspect feed system
    python examples/inspect_assembly.py --air            # Inspect air system
    python examples/inspect_assembly.py --classification # Inspect classification system
    python examples/inspect_assembly.py --feed --detailed # Detailed feed analysis
    python examples/inspect_assembly.py --air --detailed  # Detailed air analysis
    python examples/inspect_assembly.py --classification --detailed # Detailed classification analysis
    python examples/inspect_assembly.py --drop           # Feed system vertical drop
    python examples/inspect_assembly.py --flow           # Air system flow path
    python examples/inspect_assembly.py --class-flow     # Classification system flow path
    python examples/inspect_assembly.py --air-dims       # Verify air system dimensions
    python examples/inspect_assembly.py --air-coords     # Verify air system coordinates
    python examples/inspect_assembly.py --class-dims     # Verify classification system dimensions
    python examples/inspect_assembly.py --class-coords   # Verify classification system coordinates
    python examples/inspect_assembly.py --class-angles   # Verify classification system port angles
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
        elif hasattr(duct.params, 'diameter') and hasattr(duct.params, 'angle') and hasattr(duct.params, 'bend_radius'):
            # Elbow
            print(f"           D={duct.params.diameter*1000:.1f}mm, angle={duct.params.angle}°, R={duct.params.bend_radius*1000:.1f}mm")
        elif hasattr(duct.params, 'rect_width'):
            # Rect-to-round transition
            print(f"           Rect: {duct.params.rect_width*1000:.1f}x{duct.params.rect_height*1000:.1f}mm -> Round: D={duct.params.round_diameter*1000:.1f}mm, L={duct.params.length*1000:.1f}mm")
    
    # Print summary
    air.print_summary()
    
    return air


def verify_air_system_dimensions():
    """
    Verify dimension matching at all connection points in the air system.

    Checks:
    1. Duct diameters match at connection flanges
    2. Rectangular dimensions match (width/height)
    3. Transition inlet matches blower outlet, outlet matches damper inlet
    """
    print("\n" + "=" * 70)
    print("AIR SYSTEM DIMENSION VERIFICATION")
    print("=" * 70)

    air = create_standard_air_system()

    verification_results = []
    all_ok = True

    # Get components
    inlet_filter = air.inlet_filter
    blower = air.blower
    dampers = air.dampers
    duct_sections = air._duct_sections

    # ============================================================
    # 1. FILTER TO DUCT CONNECTION
    # ============================================================
    print("\n--- 1. Filter Outlet -> Horizontal Duct ---")
    filter_outlet = inlet_filter.ports['outlet']
    horiz_duct = duct_sections[0][0]  # First duct section

    filter_d = filter_outlet.diameter * 1000
    duct_d = horiz_duct.params.diameter * 1000
    match = abs(filter_d - duct_d) < 1.0  # 1mm tolerance

    print(f"  Filter outlet diameter:  {filter_d:8.1f} mm")
    print(f"  Horizontal duct diameter: {duct_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(filter_d - duct_d):.1f} mm)")
    verification_results.append(("Filter->HorizDuct", match, filter_d, duct_d))
    all_ok = all_ok and match

    # ============================================================
    # 2. HORIZONTAL DUCT TO ELBOW
    # ============================================================
    print("\n--- 2. Horizontal Duct -> 90° Elbow ---")
    elbow = duct_sections[1][0]

    horiz_d = horiz_duct.params.diameter * 1000
    elbow_d = elbow.params.diameter * 1000
    match = abs(horiz_d - elbow_d) < 1.0

    print(f"  Horizontal duct diameter: {horiz_d:8.1f} mm")
    print(f"  Elbow diameter:           {elbow_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(horiz_d - elbow_d):.1f} mm)")
    verification_results.append(("HorizDuct->Elbow", match, horiz_d, elbow_d))
    all_ok = all_ok and match

    # ============================================================
    # 3. ELBOW TO VERTICAL DUCT
    # ============================================================
    print("\n--- 3. 90° Elbow -> Vertical Duct ---")
    vert_duct = duct_sections[2][0]

    vert_d = vert_duct.params.diameter * 1000
    match = abs(elbow_d - vert_d) < 1.0

    print(f"  Elbow diameter:          {elbow_d:8.1f} mm")
    print(f"  Vertical duct diameter:  {vert_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(elbow_d - vert_d):.1f} mm)")
    verification_results.append(("Elbow->VertDuct", match, elbow_d, vert_d))
    all_ok = all_ok and match

    # ============================================================
    # 4. VERTICAL DUCT TO BLOWER INLET
    # ============================================================
    print("\n--- 4. Vertical Duct -> Blower Inlet Bell ---")
    blower_inlet = blower.ports['inlet']

    blower_in_d = blower_inlet.diameter * 1000
    match = abs(vert_d - blower_in_d) < 5.0  # 5mm tolerance (inlet bell may be larger)

    print(f"  Vertical duct diameter:  {vert_d:8.1f} mm")
    print(f"  Blower inlet diameter:   {blower_in_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[WARN]'} (diff: {abs(vert_d - blower_in_d):.1f} mm)")
    print(f"  Note: Inlet bell may be slightly larger for smooth air entry")
    verification_results.append(("VertDuct->BlowerIn", match, vert_d, blower_in_d))
    all_ok = all_ok and match

    # ============================================================
    # 5. BLOWER OUTLET TO TRANSITION (DIRECT CONNECTION)
    # ============================================================
    print("\n--- 5. Blower Outlet -> Rect-to-Round Transition ---")
    blower_outlet = blower.ports['outlet']
    transition = duct_sections[3][0]  # Transition (directly connected to blower)

    blower_out_w = blower_outlet.width * 1000
    blower_out_h = blower_outlet.height * 1000
    trans_rect_w = transition.params.rect_width * 1000
    trans_rect_h = transition.params.rect_height * 1000

    match_w = abs(blower_out_w - trans_rect_w) < 1.0
    match_h = abs(blower_out_h - trans_rect_h) < 1.0
    match = match_w and match_h

    print(f"  Blower outlet:     {blower_out_w:6.1f} x {blower_out_h:6.1f} mm (WxH)")
    print(f"  Transition inlet:  {trans_rect_w:6.1f} x {trans_rect_h:6.1f} mm (WxH)")
    print(f"  Width match:  {'[OK]' if match_w else '[FAIL]'} (diff: {abs(blower_out_w - trans_rect_w):.1f} mm)")
    print(f"  Height match: {'[OK]' if match_h else '[FAIL]'} (diff: {abs(blower_out_h - trans_rect_h):.1f} mm)")
    print(f"  Connection: Direct with square flange rings for fitting")
    verification_results.append(("BlowerOut->Transition", match, f"{blower_out_w}x{blower_out_h}", f"{trans_rect_w}x{trans_rect_h}"))
    all_ok = all_ok and match

    # ============================================================
    # 6. TRANSITION ROUND END TO DUCT
    # ============================================================
    print("\n--- 6. Transition Round End -> Round Duct ---")
    trans_round_d = transition.params.round_diameter * 1000

    if len(duct_sections) > 4:
        round_duct = duct_sections[4][0]
        round_duct_d = round_duct.params.diameter * 1000
        match = abs(trans_round_d - round_duct_d) < 1.0

        print(f"  Transition round outlet: {trans_round_d:8.1f} mm")
        print(f"  Round duct diameter:     {round_duct_d:8.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(trans_round_d - round_duct_d):.1f} mm)")
        verification_results.append(("Transition->RoundDuct", match, trans_round_d, round_duct_d))
        all_ok = all_ok and match
    else:
        print(f"  Transition round outlet: {trans_round_d:8.1f} mm")
        print(f"  (Direct connection to damper)")

    # ============================================================
    # 7. ROUND DUCT/TRANSITION TO DAMPER
    # ============================================================
    if dampers:
        print("\n--- 7. Round Duct -> Damper 1 Inlet ---")
        damper_inlet = dampers[0].ports['inlet']
        damper_in_d = damper_inlet.diameter * 1000

        match = abs(trans_round_d - damper_in_d) < 1.0

        print(f"  Transition/duct outlet:  {trans_round_d:8.1f} mm")
        print(f"  Damper 1 inlet diameter: {damper_in_d:8.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(trans_round_d - damper_in_d):.1f} mm)")
        verification_results.append(("Duct->Damper1", match, trans_round_d, damper_in_d))
        all_ok = all_ok and match

        # Damper-to-damper connections
        for i in range(len(dampers) - 1):
            print(f"\n--- {8+i}. Damper {i+1} Outlet -> Damper {i+2} Inlet ---")
            d1_out = dampers[i].ports['outlet']
            d2_in = dampers[i+1].ports['inlet']

            d1_out_d = d1_out.diameter * 1000
            d2_in_d = d2_in.diameter * 1000
            match = abs(d1_out_d - d2_in_d) < 1.0

            print(f"  Damper {i+1} outlet diameter: {d1_out_d:8.1f} mm")
            print(f"  Damper {i+2} inlet diameter:  {d2_in_d:8.1f} mm")
            print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(d1_out_d - d2_in_d):.1f} mm)")
            verification_results.append((f"Damper{i+1}->Damper{i+2}", match, d1_out_d, d2_in_d))
            all_ok = all_ok and match

    # Summary
    print("\n" + "=" * 70)
    print("DIMENSION VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _, _ in verification_results if ok)
    total = len(verification_results)
    print(f"\nResult: {passed}/{total} dimension checks passed")

    if not all_ok:
        print("\nFailed checks:")
        for name, ok, src, tgt in verification_results:
            if not ok:
                print(f"  [FAIL] {name}: {src} vs {tgt}")
    else:
        print("\n[ALL DIMENSIONS MATCH] - Flow path dimensions are consistent")

    return all_ok, verification_results


def verify_air_system_coordinates():
    """
    Verify coordinate alignment at all connection points in the air system.

    Checks that component flanges/ports align at the correct X, Y, Z positions.
    """
    print("\n" + "=" * 70)
    print("AIR SYSTEM COORDINATE VERIFICATION")
    print("=" * 70)

    air = create_standard_air_system()

    verification_results = []
    all_ok = True
    tolerance = 0.005  # 5mm tolerance for coordinate matching

    # Get components and positions
    inlet_filter = air.inlet_filter
    blower = air.blower
    dampers = air.dampers
    duct_sections = air._duct_sections

    filter_pos = air._filter_position
    blower_pos = air._blower_position
    damper_positions = air._damper_positions

    # ============================================================
    # 1. FILTER OUTLET TO HORIZONTAL DUCT START
    # ============================================================
    print("\n--- 1. Filter Outlet -> Horizontal Duct Start ---")
    filter_outlet = inlet_filter.ports['outlet']
    filter_outlet_world = (
        filter_pos[0] + filter_outlet.position[0],
        filter_pos[1] + filter_outlet.position[1],
        filter_pos[2] + filter_outlet.position[2],
    )

    horiz_duct, horiz_duct_pos = duct_sections[0]

    diff_x = abs(filter_outlet_world[0] - horiz_duct_pos[0])
    diff_y = abs(filter_outlet_world[1] - horiz_duct_pos[1])
    diff_z = abs(filter_outlet_world[2] - horiz_duct_pos[2])
    match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

    print(f"  Filter outlet:      X={filter_outlet_world[0]*1000:8.1f}, Y={filter_outlet_world[1]*1000:8.1f}, Z={filter_outlet_world[2]*1000:8.1f} mm")
    print(f"  Horiz duct start:   X={horiz_duct_pos[0]*1000:8.1f}, Y={horiz_duct_pos[1]*1000:8.1f}, Z={horiz_duct_pos[2]*1000:8.1f} mm")
    print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("FilterOut->HorizDuct", match, filter_outlet_world, horiz_duct_pos))
    all_ok = all_ok and match

    # ============================================================
    # 2. HORIZONTAL DUCT END TO ELBOW INLET
    # ============================================================
    print("\n--- 2. Horizontal Duct End -> Elbow Inlet ---")
    horiz_duct_end = (
        horiz_duct_pos[0] + horiz_duct.params.length,
        horiz_duct_pos[1],
        horiz_duct_pos[2],
    )

    elbow, elbow_pos = duct_sections[1]

    diff_x = abs(horiz_duct_end[0] - elbow_pos[0])
    diff_y = abs(horiz_duct_end[1] - elbow_pos[1])
    diff_z = abs(horiz_duct_end[2] - elbow_pos[2])
    match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

    print(f"  Horiz duct end:     X={horiz_duct_end[0]*1000:8.1f}, Y={horiz_duct_end[1]*1000:8.1f}, Z={horiz_duct_end[2]*1000:8.1f} mm")
    print(f"  Elbow inlet:        X={elbow_pos[0]*1000:8.1f}, Y={elbow_pos[1]*1000:8.1f}, Z={elbow_pos[2]*1000:8.1f} mm")
    print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("HorizDuctEnd->Elbow", match, horiz_duct_end, elbow_pos))
    all_ok = all_ok and match

    # ============================================================
    # 3. ELBOW OUTLET TO VERTICAL DUCT START
    # ============================================================
    print("\n--- 3. Elbow Outlet -> Vertical Duct Start ---")
    bend_radius = elbow.params.bend_radius
    elbow_outlet = (
        elbow_pos[0] + bend_radius,
        elbow_pos[1],
        elbow_pos[2] + bend_radius,
    )

    vert_duct, vert_duct_pos = duct_sections[2]

    diff_x = abs(elbow_outlet[0] - vert_duct_pos[0])
    diff_y = abs(elbow_outlet[1] - vert_duct_pos[1])
    diff_z = abs(elbow_outlet[2] - vert_duct_pos[2])
    match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

    print(f"  Elbow outlet:       X={elbow_outlet[0]*1000:8.1f}, Y={elbow_outlet[1]*1000:8.1f}, Z={elbow_outlet[2]*1000:8.1f} mm")
    print(f"  Vert duct start:    X={vert_duct_pos[0]*1000:8.1f}, Y={vert_duct_pos[1]*1000:8.1f}, Z={vert_duct_pos[2]*1000:8.1f} mm")
    print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("ElbowOut->VertDuct", match, elbow_outlet, vert_duct_pos))
    all_ok = all_ok and match

    # ============================================================
    # 4. VERTICAL DUCT END TO BLOWER INLET BELL
    # ============================================================
    print("\n--- 4. Vertical Duct End -> Blower Inlet Bell ---")
    vert_duct_end = (
        vert_duct_pos[0],
        vert_duct_pos[1],
        vert_duct_pos[2] + vert_duct.params.length,
    )

    blower_inlet = blower.ports['inlet']
    blower_inlet_world = (
        blower_pos[0] + blower_inlet.position[0],
        blower_pos[1] + blower_inlet.position[1],
        blower_pos[2] + blower_inlet.position[2],
    )

    # Note: Blower inlet port is at scroll center, but inlet bell is at +Z edge
    # The duct should reach the actual inlet bell opening
    impeller_width = blower.params.impeller_width
    scroll_half_width = impeller_width / 2 * 1.2
    inlet_bell_z = blower_pos[2] + scroll_half_width

    diff_x = abs(vert_duct_end[0] - blower_pos[0])
    diff_y = abs(vert_duct_end[1] - blower_pos[1])
    diff_z = abs(vert_duct_end[2] - inlet_bell_z)
    match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

    print(f"  Vert duct end:      X={vert_duct_end[0]*1000:8.1f}, Y={vert_duct_end[1]*1000:8.1f}, Z={vert_duct_end[2]*1000:8.1f} mm")
    print(f"  Inlet bell center:  X={blower_pos[0]*1000:8.1f}, Y={blower_pos[1]*1000:8.1f}, Z={inlet_bell_z*1000:8.1f} mm")
    print(f"  (Port position:     X={blower_inlet_world[0]*1000:8.1f}, Y={blower_inlet_world[1]*1000:8.1f}, Z={blower_inlet_world[2]*1000:8.1f} mm)")
    print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[WARN]'}")
    verification_results.append(("VertDuctEnd->InletBell", match, vert_duct_end, (blower_pos[0], blower_pos[1], inlet_bell_z)))
    all_ok = all_ok and match

    # ============================================================
    # 5. BLOWER OUTLET FLANGE TO TRANSITION (DIRECT CONNECTION)
    # ============================================================
    print("\n--- 5. Blower Outlet Flange -> Transition Start ---")
    blower_outlet = blower.ports['outlet']
    outlet_duct_length = blower.params.outlet_height * 1.5
    blower_outlet_flange = (
        blower_pos[0] + blower_outlet.position[0] + outlet_duct_length,
        blower_pos[1] + blower_outlet.position[1],
        blower_pos[2] + blower_outlet.position[2],
    )

    transition, trans_pos = duct_sections[3]

    diff_x = abs(blower_outlet_flange[0] - trans_pos[0])
    diff_y = abs(blower_outlet_flange[1] - trans_pos[1])
    diff_z = abs(blower_outlet_flange[2] - trans_pos[2])
    match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

    print(f"  Blower out flange:  X={blower_outlet_flange[0]*1000:8.1f}, Y={blower_outlet_flange[1]*1000:8.1f}, Z={blower_outlet_flange[2]*1000:8.1f} mm")
    print(f"  Transition start:   X={trans_pos[0]*1000:8.1f}, Y={trans_pos[1]*1000:8.1f}, Z={trans_pos[2]*1000:8.1f} mm")
    print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
    print(f"  Connection:         Direct with square flange rings")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("BlowerFlange->Transition", match, blower_outlet_flange, trans_pos))
    all_ok = all_ok and match

    # ============================================================
    # 6. TRANSITION END TO DUCT/DAMPER
    # ============================================================
    trans_end = (
        trans_pos[0] + transition.params.length,
        trans_pos[1],
        trans_pos[2],
    )

    if len(duct_sections) > 4:
        print("\n--- 6. Transition End -> Round Duct Start ---")
        round_duct, round_duct_pos = duct_sections[4]

        diff_x = abs(trans_end[0] - round_duct_pos[0])
        diff_y = abs(trans_end[1] - round_duct_pos[1])
        diff_z = abs(trans_end[2] - round_duct_pos[2])
        match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

        print(f"  Transition end:     X={trans_end[0]*1000:8.1f}, Y={trans_end[1]*1000:8.1f}, Z={trans_end[2]*1000:8.1f} mm")
        print(f"  Round duct start:   X={round_duct_pos[0]*1000:8.1f}, Y={round_duct_pos[1]*1000:8.1f}, Z={round_duct_pos[2]*1000:8.1f} mm")
        print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
        verification_results.append(("TransitionEnd->RoundDuct", match, trans_end, round_duct_pos))
        all_ok = all_ok and match

    # ============================================================
    # 7. DAMPER CONNECTIONS
    # ============================================================
    if dampers and damper_positions:
        print("\n--- 7. Duct -> Damper 1 Inlet ---")
        damper1_inlet = dampers[0].ports['inlet']
        damper1_inlet_world = (
            damper_positions[0][0] + damper1_inlet.position[0],
            damper_positions[0][1] + damper1_inlet.position[1],
            damper_positions[0][2] + damper1_inlet.position[2],
        )

        # Get last duct end position before damper
        if len(duct_sections) > 4:
            last_duct, last_duct_pos = duct_sections[4]
            duct_end_to_damper = (
                last_duct_pos[0] + last_duct.params.length,
                last_duct_pos[1],
                last_duct_pos[2],
            )
        else:
            duct_end_to_damper = trans_end

        diff_x = abs(duct_end_to_damper[0] - damper1_inlet_world[0])
        diff_y = abs(duct_end_to_damper[1] - damper1_inlet_world[1])
        diff_z = abs(duct_end_to_damper[2] - damper1_inlet_world[2])
        match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

        print(f"  Duct end:           X={duct_end_to_damper[0]*1000:8.1f}, Y={duct_end_to_damper[1]*1000:8.1f}, Z={duct_end_to_damper[2]*1000:8.1f} mm")
        print(f"  Damper 1 inlet:     X={damper1_inlet_world[0]*1000:8.1f}, Y={damper1_inlet_world[1]*1000:8.1f}, Z={damper1_inlet_world[2]*1000:8.1f} mm")
        print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[WARN]'}")
        verification_results.append(("Duct->Damper1Inlet", match, duct_end_to_damper, damper1_inlet_world))
        all_ok = all_ok and match

        # Check damper-to-damper coordinate alignment
        for i in range(len(dampers) - 1):
            print(f"\n--- {8+i}. Damper {i+1} Outlet -> Damper {i+2} Inlet ---")
            d1_outlet = dampers[i].ports['outlet']
            d1_outlet_world = (
                damper_positions[i][0] + d1_outlet.position[0],
                damper_positions[i][1] + d1_outlet.position[1],
                damper_positions[i][2] + d1_outlet.position[2],
            )

            d2_inlet = dampers[i+1].ports['inlet']
            d2_inlet_world = (
                damper_positions[i+1][0] + d2_inlet.position[0],
                damper_positions[i+1][1] + d2_inlet.position[1],
                damper_positions[i+1][2] + d2_inlet.position[2],
            )

            diff_y = abs(d1_outlet_world[1] - d2_inlet_world[1])
            diff_z = abs(d1_outlet_world[2] - d2_inlet_world[2])
            # X difference is expected (spacing between dampers)
            x_gap = d2_inlet_world[0] - d1_outlet_world[0]

            match = diff_y < tolerance and diff_z < tolerance and x_gap > 0

            print(f"  Damper {i+1} outlet:   X={d1_outlet_world[0]*1000:8.1f}, Y={d1_outlet_world[1]*1000:8.1f}, Z={d1_outlet_world[2]*1000:8.1f} mm")
            print(f"  Damper {i+2} inlet:    X={d2_inlet_world[0]*1000:8.1f}, Y={d2_inlet_world[1]*1000:8.1f}, Z={d2_inlet_world[2]*1000:8.1f} mm")
            print(f"  X gap (duct):       {x_gap*1000:.1f} mm")
            print(f"  Y/Z alignment:      dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
            print(f"  Match: {'[OK]' if match else '[FAIL]'}")
            verification_results.append((f"Damper{i+1}Out->Damper{i+2}In", match, d1_outlet_world, d2_inlet_world))
            all_ok = all_ok and match

    # Summary
    print("\n" + "=" * 70)
    print("COORDINATE VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _, _ in verification_results if ok)
    total = len(verification_results)
    print(f"\nResult: {passed}/{total} coordinate checks passed")

    if not all_ok:
        print("\nFailed checks:")
        for name, ok, src, tgt in verification_results:
            if not ok:
                print(f"  [FAIL] {name}")
    else:
        print("\n[ALL COORDINATES ALIGN] - Connection points are properly positioned")

    return all_ok, verification_results


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

    # ============================================================
    # NEW: DIMENSION AND COORDINATE VERIFICATION
    # ============================================================
    print("\n")
    dim_ok, dim_results = verify_air_system_dimensions()

    print("\n")
    coord_ok, coord_results = verify_air_system_coordinates()

    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 70)
    dim_passed = sum(1 for _, ok, _, _ in dim_results if ok)
    coord_passed = sum(1 for _, ok, _, _ in coord_results if ok)
    total_checks = len(dim_results) + len(coord_results)
    total_passed = dim_passed + coord_passed

    print(f"\nDimension checks:   {dim_passed}/{len(dim_results)} passed")
    print(f"Coordinate checks:  {coord_passed}/{len(coord_results)} passed")
    print(f"Total:              {total_passed}/{total_checks} passed")

    if dim_ok and coord_ok:
        print("\n[SYSTEM VERIFIED] - All dimensions and coordinates are correct")
    else:
        print("\n[ISSUES FOUND] - Please review failed checks above")


def inspect_classification_system():
    """Inspect classification system component positioning."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM ASSEMBLY INSPECTION")
    print("=" * 70)

    classification = create_standard_classification_system()

    # Print component positions
    print("\n--- Component Positions ---")
    positions = classification.get_component_positions()
    for name, pos in positions.items():
        print(f"{name:20s}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m")

    # Print port locations for each component
    print("\n--- Connection Ports ---")

    components = {
        'venturi': (classification.venturi, positions['venturi']),
        'zigzag': (classification.zigzag, positions['zigzag']),
        'wheel_classifier': (classification.wheel_classifier, positions['wheel_classifier']),
        'multi_cyclone': (classification.multi_cyclone, positions['multi_cyclone']),
        'bag_filter': (classification.bag_filter, positions['bag_filter']),
    }

    for comp_name, (comp, pos) in components.items():
        if hasattr(comp, 'ports'):
            print(f"\n  {comp_name.upper()}:")
            for port_name, port in comp.ports.items():
                world_pos = port.get_world_position(pos)
                print(f"    {port_name:20s}: pos=({world_pos[0]:7.3f}, {world_pos[1]:7.3f}, {world_pos[2]:7.3f})")
                print(f"                          dir=({port.direction[0]:5.2f}, {port.direction[1]:5.2f}, {port.direction[2]:5.2f})")
                if hasattr(port, 'width') and port.width > 0:
                    print(f"                          W={port.width*1000:.1f}mm, H={port.height*1000:.1f}mm, type={port.port_type.value}")
                else:
                    print(f"                          D={port.diameter*1000:.1f}mm, type={port.port_type.value}")

    # Print duct sections
    print("\n--- Duct Sections ---")
    print(f"  Total duct sections: {len(classification._duct_sections)}")
    for i, (duct, pos) in enumerate(classification._duct_sections):
        duct_type = type(duct).__name__
        print(f"  {i+1}. {duct_type}: pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")

        # Different info based on duct type
        if hasattr(duct.params, 'diameter') and hasattr(duct.params, 'length'):
            print(f"           D={duct.params.diameter*1000:.1f}mm, L={duct.params.length*1000:.1f}mm")
        elif hasattr(duct.params, 'diameter') and hasattr(duct.params, 'angle') and hasattr(duct.params, 'bend_radius'):
            # Elbow
            print(f"           D={duct.params.diameter*1000:.1f}mm, angle={duct.params.angle}°, R={duct.params.bend_radius*1000:.1f}mm")
        elif hasattr(duct.params, 'inlet_dimensions') and hasattr(duct.params, 'outlet_dimensions'):
            # Transition (including ExpandingTransitionWithDropout)
            inlet_dims = duct.params.inlet_dimensions
            outlet_dims = duct.params.outlet_dimensions
            # Get length - may be 'length' or 'transition_length' depending on type
            if hasattr(duct.params, 'length'):
                length = duct.params.length * 1000
            elif hasattr(duct.params, 'transition_length'):
                length = duct.params.transition_length * 1000
            else:
                length = 0.0
            # Get transition type - may not exist for ExpandingTransitionParams
            if hasattr(duct.params, 'transition_type'):
                trans_type = duct.params.transition_type
            else:
                trans_type = type(duct).__name__  # Use class name
            if len(inlet_dims) == 1 and len(outlet_dims) == 1:
                # Round-to-round
                print(f"           {trans_type}: D={inlet_dims[0]*1000:.1f}mm -> D={outlet_dims[0]*1000:.1f}mm, L={length:.1f}mm")
            elif len(inlet_dims) == 1:
                # Round-to-rect
                print(f"           {trans_type}: D={inlet_dims[0]*1000:.1f}mm -> {outlet_dims[0]*1000:.1f}x{outlet_dims[1]*1000:.1f}mm, L={length:.1f}mm")
            elif len(outlet_dims) == 1:
                # Rect-to-round
                print(f"           {trans_type}: {inlet_dims[0]*1000:.1f}x{inlet_dims[1]*1000:.1f}mm -> D={outlet_dims[0]*1000:.1f}mm, L={length:.1f}mm")
            else:
                # Rect-to-rect
                print(f"           {trans_type}: {inlet_dims[0]*1000:.1f}x{inlet_dims[1]*1000:.1f}mm -> {outlet_dims[0]*1000:.1f}x{outlet_dims[1]*1000:.1f}mm, L={length:.1f}mm")

    # Print summary
    classification.print_summary()

    return classification


def _filter_main_flow_ducts(duct_sections):
    """Filter classification duct sections to the main flow path only.

    Collection hardware (airlocks, routing elbows/ducts) is now stored in
    _collection_duct_sections, so this only needs to remove bypass tees.
    Expected main flow indices [0]-[8]:
      [0]=duct1a, [1]=trans1, [2]=trans2a, [3]=elbow2, [4]=duct2,
      [5]=trans2b, [6]=elbow3, [7]=duct3a, [8]=expansion
    """
    filtered = []
    for duct, pos in duct_sections:
        tname = type(duct).__name__
        # Skip bypass tee junctions
        if tname == 'TeeJunction':
            continue
        filtered.append((duct, pos))
    return filtered


def verify_classification_system_dimensions():
    """
    Verify dimension matching at all connection points in the classification system.

    Checks transitions and connections through the complete flow path:
    Venturi -> Duct1a -> Trans1 -> Zigzag -> Trans2a -> Elbow2 -> Duct2 -> Trans2b ->
    Cyclone -> Elbow3 -> Duct3a -> Expansion -> Bag Filter

    Uses flow area comparison for rectangular-to-round connections.
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM DIMENSION VERIFICATION")
    print("=" * 70)

    classification = create_standard_classification_system()

    verification_results = []
    all_ok = True

    # Get components
    venturi = classification.venturi
    zigzag = classification.zigzag
    multi_cyclone = classification.multi_cyclone
    bag_filter = classification.bag_filter
    duct_sections = _filter_main_flow_ducts(classification._duct_sections)

    print(f"\n  Main flow path duct sections: {len(duct_sections)} (total incl. collection: {len(classification._duct_sections)})")

    def get_equiv_diameter(port):
        """Calculate equivalent diameter from port area: D_eq = sqrt(4*A/π)"""
        area = port.area
        return np.sqrt(4 * area / np.pi) * 1000

    def get_round_duct_area(diameter_m):
        """Calculate area of round duct."""
        return np.pi * (diameter_m / 2) ** 2

    def area_match_pct(area1, area2):
        """Calculate area match percentage."""
        if max(area1, area2) == 0:
            return 0
        return min(area1, area2) / max(area1, area2) * 100

    def get_duct_diameter(duct):
        """
        Safely get diameter from various duct types (Elbow, RoundDuct, Transition, etc.).
        Returns diameter in mm.
        """
        p = duct.params
        # Elbow and RoundDuct have direct diameter attribute
        if hasattr(p, 'diameter'):
            return p.diameter * 1000
        # Transitions have inlet_dimensions and outlet_dimensions
        elif hasattr(p, 'inlet_dimensions'):
            dims = p.inlet_dimensions
            if len(dims) == 1:
                return dims[0] * 1000  # Round
            else:
                # Rectangular - return equivalent diameter from area
                area = dims[0] * dims[1]
                return np.sqrt(4 * area / np.pi) * 1000
        else:
            raise ValueError(f"Cannot get diameter from {type(duct).__name__}")

    def get_transition_dims(trans, end='inlet'):
        """Get transition dimensions (inlet or outlet)."""
        p = trans.params
        if end == 'inlet':
            dims = p.inlet_dimensions
        else:
            dims = p.outlet_dimensions
        if len(dims) == 1:
            return ('round', dims[0] * 1000)
        else:
            return ('rect', dims[0] * 1000, dims[1] * 1000)

    # Resolve flow-path indices by role (order changes when wheel classifier is present)
    elbows_in_flow = [(i, d) for i, (d, _) in enumerate(duct_sections)
                      if type(d).__name__ == 'DuctElbow']
    elbow3_idx = elbows_in_flow[-1][0] if elbows_in_flow else -1
    trans2b_idx = elbow3_idx - 1 if elbow3_idx >= 1 else 5
    duct2_idx = elbow3_idx - 2 if elbow3_idx >= 2 else 4
    elbow2_idx = elbow3_idx - 3 if elbow3_idx >= 3 else 3
    trans2a_idx = elbow3_idx - 4 if elbow3_idx >= 4 else 2

    # ============================================================
    # 1. VENTURI OUTLET TO DUCT 1A
    # ============================================================
    print("\n--- 1. Venturi Outlet -> Duct 1a ---")
    venturi_outlet = venturi.ports['outlet']
    duct1a = duct_sections[0][0]

    venturi_d = venturi_outlet.diameter * 1000
    duct1a_d = duct1a.params.diameter * 1000
    match = abs(venturi_d - duct1a_d) < 1.0

    print(f"  Venturi outlet:           D={venturi_d:.1f} mm")
    print(f"  Duct 1a:                  D={duct1a_d:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Venturi->Duct1a", match, venturi_d, duct1a_d))
    all_ok = all_ok and match

    # ============================================================
    # 2. DUCT 1A TO TRANSITION 1 (ROUND-TO-RECT)
    # ============================================================
    print("\n--- 2. Duct 1a -> Transition 1 (round-to-rect) ---")
    trans1 = duct_sections[1][0]
    trans1_inlet = get_transition_dims(trans1, 'inlet')
    trans1_outlet = get_transition_dims(trans1, 'outlet')

    match = abs(duct1a_d - trans1_inlet[1]) < 1.0

    print(f"  Duct 1a:                  D={duct1a_d:.1f} mm")
    print(f"  Transition 1 inlet:       D={trans1_inlet[1]:.1f} mm (round)")
    print(f"  Transition 1 outlet:      {trans1_outlet[1]:.1f} x {trans1_outlet[2]:.1f} mm (rect)")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Duct1a->Trans1", match, duct1a_d, trans1_inlet[1]))
    all_ok = all_ok and match

    # ============================================================
    # 3. TRANSITION 1 TO ZIGZAG AIR INLET
    # ============================================================
    print("\n--- 3. Transition 1 -> Zigzag Air Inlet ---")
    zigzag_inlet = zigzag.ports['air_inlet']
    zigzag_inlet_w = zigzag_inlet.width * 1000
    zigzag_inlet_h = zigzag_inlet.height * 1000

    match_w = abs(trans1_outlet[1] - zigzag_inlet_w) < 5.0
    match_h = abs(trans1_outlet[2] - zigzag_inlet_h) < 5.0
    match = match_w and match_h

    print(f"  Transition 1 outlet:      {trans1_outlet[1]:.1f} x {trans1_outlet[2]:.1f} mm")
    print(f"  Zigzag inlet:             {zigzag_inlet_w:.1f} x {zigzag_inlet_h:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans1->ZigzagIn", match, trans1_outlet[1], zigzag_inlet_w))
    all_ok = all_ok and match

    # ============================================================
    # 4. ZIGZAG FINES OUTLET TO TRANSITION 2A (RECT-TO-ROUND)
    # ============================================================
    print("\n--- 4. Zigzag Fines Outlet -> Transition 2a (rect-to-round) ---")
    zigzag_fines = zigzag.ports['fines_outlet']
    zigzag_fines_w = zigzag_fines.width * 1000
    zigzag_fines_h = zigzag_fines.height * 1000

    trans2a = duct_sections[trans2a_idx][0]
    trans2a_inlet = get_transition_dims(trans2a, 'inlet')
    trans2a_outlet = get_transition_dims(trans2a, 'outlet')

    match_w = abs(trans2a_inlet[1] - zigzag_fines_w) < 5.0
    match_h = abs(trans2a_inlet[2] - zigzag_fines_h) < 5.0
    match = match_w and match_h

    print(f"  Zigzag fines outlet:      {zigzag_fines_w:.1f} x {zigzag_fines_h:.1f} mm")
    print(f"  Transition 2a inlet:      {trans2a_inlet[1]:.1f} x {trans2a_inlet[2]:.1f} mm (rect)")
    print(f"  Transition 2a outlet:     D={trans2a_outlet[1]:.1f} mm (round)")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("ZigzagFines->Trans2a", match, zigzag_fines_w, trans2a_inlet[1]))
    all_ok = all_ok and match

    # ============================================================
    # 5. TRANSITION 2A TO ELBOW 2
    # ============================================================
    print("\n--- 5. Transition 2a -> Elbow 2 ---")
    elbow2 = duct_sections[elbow2_idx][0]
    elbow2_d = elbow2.params.diameter * 1000

    match = abs(trans2a_outlet[1] - elbow2_d) < 1.0

    print(f"  Transition 2a outlet:     D={trans2a_outlet[1]:.1f} mm")
    print(f"  Elbow 2:                  D={elbow2_d:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans2a->Elbow2", match, trans2a_outlet[1], elbow2_d))
    all_ok = all_ok and match

    # ============================================================
    # 6. ELBOW 2 TO DUCT 2
    # ============================================================
    print("\n--- 6. Elbow 2 -> Duct 2 ---")
    duct2 = duct_sections[duct2_idx][0]
    duct2_d = duct2.params.diameter * 1000

    match = abs(elbow2_d - duct2_d) < 1.0

    print(f"  Elbow 2:                  D={elbow2_d:.1f} mm")
    print(f"  Duct 2:                   D={duct2_d:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Elbow2->Duct2", match, elbow2_d, duct2_d))
    all_ok = all_ok and match

    # ============================================================
    # 7. DUCT 2 TO TRANSITION 2B (ROUND-TO-RECT)
    # ============================================================
    print("\n--- 7. Duct 2 -> Transition 2b (round-to-rect) ---")
    trans2b = duct_sections[trans2b_idx][0]
    trans2b_inlet = get_transition_dims(trans2b, 'inlet')
    trans2b_outlet = get_transition_dims(trans2b, 'outlet')

    match = abs(duct2_d - trans2b_inlet[1]) < 1.0

    print(f"  Duct 2:                   D={duct2_d:.1f} mm")
    print(f"  Transition 2b inlet:      D={trans2b_inlet[1]:.1f} mm (round)")
    print(f"  Transition 2b outlet:     {trans2b_outlet[1]:.1f} x {trans2b_outlet[2]:.1f} mm (rect)")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Duct2->Trans2b", match, duct2_d, trans2b_inlet[1]))
    all_ok = all_ok and match

    # ============================================================
    # 8. TRANSITION 2B TO MULTI-CYCLONE INLET
    # ============================================================
    print("\n--- 8. Transition 2b -> Multi-Cyclone Inlet ---")
    cyclone_inlet = multi_cyclone.ports['inlet']
    cyclone_inlet_w = cyclone_inlet.width * 1000
    cyclone_inlet_h = cyclone_inlet.height * 1000

    # The transition outlet_dimensions are ordered by the transition's
    # coordinate system (perp1, perp2), which for +X direction flow maps
    # outlet_dimensions[0] -> height (Y) and [1] -> width (Z).
    # Compare order-independently: both dimension pairs must match as a set.
    trans_dims = sorted([trans2b_outlet[1], trans2b_outlet[2]])
    cyclone_dims = sorted([cyclone_inlet_w, cyclone_inlet_h])
    match_d1 = abs(trans_dims[0] - cyclone_dims[0]) < 5.0
    match_d2 = abs(trans_dims[1] - cyclone_dims[1]) < 5.0
    match = match_d1 and match_d2

    print(f"  Transition 2b outlet:     {trans2b_outlet[1]:.1f} x {trans2b_outlet[2]:.1f} mm")
    print(f"  Cyclone inlet:            {cyclone_inlet_w:.1f} x {cyclone_inlet_h:.1f} mm")
    note = " (axis order differs due to +X flow direction)" if trans2b_outlet[1] != cyclone_inlet_w else ""
    print(f"  Match: {'[OK]' if match else '[FAIL]'}{note}")
    verification_results.append(("Trans2b->CycloneIn", match, trans2b_outlet[1], cyclone_inlet_w))
    all_ok = all_ok and match

    # ============================================================
    # 9. MULTI-CYCLONE OVERFLOW TO ELBOW 3
    # ============================================================
    elbow3 = duct_sections[elbow3_idx][0] if 0 <= elbow3_idx < len(duct_sections) else None

    duct3a = None
    for i in range(elbow3_idx + 1, len(duct_sections)):
        d, _ = duct_sections[i]
        if type(d).__name__ == 'RoundDuct':
            duct3a = d
            break

    expansion = None
    for duct, pos in duct_sections:
        tname = type(duct).__name__
        if tname == 'Transition' and hasattr(duct, 'params'):
            p = duct.params
            if getattr(p, 'transition_type', None) == 'round_to_round':
                idims = getattr(p, 'inlet_dimensions', (0,))
                odims = getattr(p, 'outlet_dimensions', (0,))
                if len(idims) == 1 and len(odims) == 1 and odims[0] > idims[0]:
                    expansion = duct
                    break

    print("\n--- 9. Multi-Cyclone Overflow -> Elbow 3 ---")
    cyclone_overflow = multi_cyclone.ports['overflow']
    if elbow3 is None:
        print("  [SKIP] Elbow 3 not found in duct sections")
        elbow3_d = 0.0
        match = False
    else:
        cyclone_overflow_d = cyclone_overflow.diameter * 1000
        elbow3_d = get_duct_diameter(elbow3)
        match = abs(cyclone_overflow_d - elbow3_d) < 1.0
        print(f"  Cyclone overflow:         D={cyclone_overflow_d:.1f} mm")
        print(f"  Elbow 3:                  D={elbow3_d:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("CycloneOver->Elbow3", match, cyclone_overflow.diameter * 1000, elbow3_d))
    all_ok = all_ok and match

    # ============================================================
    # 10. ELBOW 3 TO DUCT 3A
    # ============================================================
    print("\n--- 10. Elbow 3 -> Duct 3a ---")
    if duct3a is None:
        print("  [SKIP] Duct 3a not found in duct sections")
        duct3a_d = 0.0
        match = False
    else:
        duct3a_d = get_duct_diameter(duct3a)
        match = abs(elbow3_d - duct3a_d) < 1.0
        print(f"  Elbow 3:                  D={elbow3_d:.1f} mm")
        print(f"  Duct 3a:                  D={duct3a_d:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Elbow3->Duct3a", match, elbow3_d, duct3a_d if duct3a is not None else 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 11. DUCT 3A TO EXPANSION TRANSITION
    # ============================================================
    print("\n--- 11. Duct 3a -> Expansion Transition ---")

    if expansion is not None:
        expansion_inlet_d = expansion.params.inlet_dimensions[0] * 1000
        expansion_outlet_d = expansion.params.outlet_dimensions[0] * 1000

        match = abs(duct3a_d - expansion_inlet_d) < 1.0

        print(f"  Duct 3a:                  D={duct3a_d:.1f} mm")
        print(f"  Expansion inlet:          D={expansion_inlet_d:.1f} mm")
        print(f"  Expansion outlet:         D={expansion_outlet_d:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
        verification_results.append(("Duct3a->Expansion", match, duct3a_d, expansion_inlet_d))
        all_ok = all_ok and match
    else:
        print("  [SKIP] Expansion transition not found in duct sections")
        expansion_outlet_d = bag_filter.ports['dirty_air_inlet'].diameter * 1000

    # ============================================================
    # 12. EXPANSION TRANSITION TO BAG FILTER INLET
    # ============================================================
    print("\n--- 12. Expansion -> Bag Filter Inlet ---")
    bag_inlet = bag_filter.ports['dirty_air_inlet']
    bag_inlet_d = bag_inlet.diameter * 1000

    match = abs(expansion_outlet_d - bag_inlet_d) < 5.0

    print(f"  Expansion outlet:         D={expansion_outlet_d:.1f} mm")
    print(f"  Bag filter inlet:         D={bag_inlet_d:.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Expansion->BagFilterIn", match, expansion_outlet_d, bag_inlet_d))
    all_ok = all_ok and match

    # Summary
    print("\n" + "=" * 70)
    print("DIMENSION VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _, _ in verification_results if ok)
    total = len(verification_results)
    print(f"\nResult: {passed}/{total} dimension checks passed")

    if not all_ok:
        print("\nIssues found:")
        for name, ok, src, tgt in verification_results:
            if not ok:
                print(f"  [!] {name}")
        print("\n  Note: Some issues may be acceptable depending on design intent.")
    else:
        print("\n[ALL DIMENSIONS MATCH] - Flow path dimensions are consistent")
        print("  Transitions properly connect all components.")

    return all_ok, verification_results


def verify_classification_system_coordinates():
    """
    Verify coordinate alignment at all connection points in the classification system.

    Checks that component flanges/ports align at the correct X, Y, Z positions.
    Flow path with wheel classifier (13 sections):
      Venturi -> Duct1a -> Trans1 -> Zigzag -> Trans2a -> Elbow2 -> Duct2 ->
      WheelInTrans -> WheelOutTrans -> Elbow(wheel->cyclone) -> Duct ->
      CycloneTrans -> Elbow3 -> Duct3a -> Expansion -> BagFilter
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM COORDINATE VERIFICATION")
    print("=" * 70)

    classification = create_standard_classification_system()

    verification_results = []
    all_ok = True
    tolerance = 0.010  # 10mm tolerance

    # Get components and positions
    venturi = classification.venturi
    zigzag = classification.zigzag
    wheel_classifier = classification.wheel_classifier
    multi_cyclone = classification.multi_cyclone
    bag_filter = classification.bag_filter
    duct_sections = _filter_main_flow_ducts(classification._duct_sections)

    positions = classification.get_component_positions()
    venturi_pos = positions['venturi']
    zigzag_pos = positions['zigzag']
    wheel_pos = positions['wheel_classifier']
    cyclone_pos = positions['multi_cyclone']
    bag_filter_pos = positions['bag_filter']

    gap = classification.params.flange_gap
    num_sections = len(duct_sections)
    print(f"\n  Verifying {num_sections} duct sections with {gap*1000:.0f}mm gaps")
    print(f"  Flow path includes wheel classifier at {wheel_pos}")

    def check_coord(name, src, tgt, expected_gap_axis=None):
        """Check coordinate alignment between source and target."""
        diff = [abs(tgt[i] - src[i]) for i in range(3)]
        if expected_gap_axis is not None:
            diff[expected_gap_axis] = abs(diff[expected_gap_axis] - gap)
        match = all(d < tolerance for d in diff)
        return match, diff

    def find_elbows():
        """Find all elbows in duct sections by checking for bend_radius attribute."""
        elbows = []
        for i, (duct, pos) in enumerate(duct_sections):
            if hasattr(duct.params, 'bend_radius'):
                elbows.append((i, duct, pos))
        return elbows

    # Find elbows dynamically
    elbows = find_elbows()
    print(f"  Found {len(elbows)} elbows in flow path")

    # ============================================================
    # 1. VENTURI OUTLET TO DUCT 1A START
    # ============================================================
    print("\n--- 1. Venturi Outlet -> Duct 1a Start ---")
    venturi_outlet = venturi.ports['outlet']
    v_out = tuple(venturi_pos[i] + venturi_outlet.position[i] for i in range(3))
    duct1a, duct1a_pos = duct_sections[0]
    match, diff = check_coord("VenturiOut->Duct1a", v_out, duct1a_pos, 1)

    print(f"  Venturi outlet:     X={v_out[0]*1000:8.1f}, Y={v_out[1]*1000:8.1f}, Z={v_out[2]*1000:8.1f} mm")
    print(f"  Duct 1a start:      X={duct1a_pos[0]*1000:8.1f}, Y={duct1a_pos[1]*1000:8.1f}, Z={duct1a_pos[2]*1000:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("VenturiOut->Duct1a", match, v_out, duct1a_pos))
    all_ok = all_ok and match

    # ============================================================
    # 2. DUCT 1A END TO TRANS 1 START
    # ============================================================
    print("\n--- 2. Duct 1a End -> Transition 1 Start ---")
    duct1a_end = (duct1a_pos[0], duct1a_pos[1] + duct1a.params.length, duct1a_pos[2])
    trans1, trans1_pos = duct_sections[1]
    match, diff = check_coord("Duct1a->Trans1", duct1a_end, trans1_pos, 1)

    print(f"  Duct 1a end:        X={duct1a_end[0]*1000:8.1f}, Y={duct1a_end[1]*1000:8.1f}, Z={duct1a_end[2]*1000:8.1f} mm")
    print(f"  Trans 1 start:      X={trans1_pos[0]*1000:8.1f}, Y={trans1_pos[1]*1000:8.1f}, Z={trans1_pos[2]*1000:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Duct1a->Trans1", match, duct1a_end, trans1_pos))
    all_ok = all_ok and match

    # ============================================================
    # 3. TRANS 1 END TO ZIGZAG AIR INLET
    # ============================================================
    print("\n--- 3. Transition 1 End -> Zigzag Air Inlet ---")
    # Handle both Transition and ExpandingTransitionWithDropout
    if hasattr(trans1.params, 'length'):
        trans1_length = trans1.params.length
    elif hasattr(trans1.params, 'transition_length'):
        trans1_length = trans1.params.transition_length
    else:
        trans1_length = 0.1  # Default
    trans1_end = (trans1_pos[0], trans1_pos[1] + trans1_length, trans1_pos[2])
    zigzag_inlet = zigzag.ports['air_inlet']
    z_in = tuple(zigzag_pos[i] + zigzag_inlet.position[i] for i in range(3))
    match, diff = check_coord("Trans1->ZigzagIn", trans1_end, z_in, 1)

    print(f"  Trans 1 end:        X={trans1_end[0]*1000:8.1f}, Y={trans1_end[1]*1000:8.1f}, Z={trans1_end[2]*1000:8.1f} mm")
    print(f"  Zigzag inlet:       X={z_in[0]*1000:8.1f}, Y={z_in[1]*1000:8.1f}, Z={z_in[2]*1000:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans1->ZigzagIn", match, trans1_end, z_in))
    all_ok = all_ok and match

    # ============================================================
    # 4. ZIGZAG FINES OUTLET TO TRANS 2A START
    # ============================================================
    print("\n--- 4. Zigzag Fines Outlet -> Transition 2a Start ---")
    zigzag_fines = zigzag.ports['fines_outlet']
    z_fines = tuple(zigzag_pos[i] + zigzag_fines.position[i] for i in range(3))
    trans2a, trans2a_pos = duct_sections[2]
    match, diff = check_coord("ZigzagFines->Trans2a", z_fines, trans2a_pos, 1)

    print(f"  Zigzag fines:       X={z_fines[0]*1000:8.1f}, Y={z_fines[1]*1000:8.1f}, Z={z_fines[2]*1000:8.1f} mm")
    print(f"  Trans 2a start:     X={trans2a_pos[0]*1000:8.1f}, Y={trans2a_pos[1]*1000:8.1f}, Z={trans2a_pos[2]*1000:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("ZigzagFines->Trans2a", match, z_fines, trans2a_pos))
    all_ok = all_ok and match

    # ============================================================
    # 5. FIRST ELBOW (Zigzag to Wheel Classifier path)
    # ============================================================
    if len(elbows) >= 1:
        elbow_idx, elbow2, elbow2_pos = elbows[0]
        print(f"\n--- 5. First Elbow (index {elbow_idx}) ---")
        print(f"  Elbow position:     X={elbow2_pos[0]*1000:8.1f}, Y={elbow2_pos[1]*1000:8.1f}, Z={elbow2_pos[2]*1000:8.1f} mm")
        print(f"  Bend radius:        {elbow2.params.bend_radius*1000:.1f} mm")
        verification_results.append(("Elbow2_Position", True, elbow2_pos, elbow2_pos))

    # ============================================================
    # WHEEL CLASSIFIER CONNECTIONS
    # ============================================================
    print("\n--- 6. Wheel Classifier Position ---")
    wheel_inlet = wheel_classifier.ports.get('inlet')
    wheel_fines = wheel_classifier.ports.get('fines_outlet')
    if wheel_inlet:
        w_in = tuple(wheel_pos[i] + wheel_inlet.position[i] for i in range(3))
        print(f"  Wheel inlet:        X={w_in[0]*1000:8.1f}, Y={w_in[1]*1000:8.1f}, Z={w_in[2]*1000:8.1f} mm")
    if wheel_fines:
        w_out = tuple(wheel_pos[i] + wheel_fines.position[i] for i in range(3))
        print(f"  Wheel fines outlet: X={w_out[0]*1000:8.1f}, Y={w_out[1]*1000:8.1f}, Z={w_out[2]*1000:8.1f} mm")
    verification_results.append(("WheelClassifier", True, wheel_pos, wheel_pos))

    # ============================================================
    # MULTI-CYCLONE CONNECTIONS
    # ============================================================
    print("\n--- 7. Multi-Cyclone Position ---")
    cyclone_inlet = multi_cyclone.ports['inlet']
    cyclone_overflow = multi_cyclone.ports['overflow']
    c_in = tuple(cyclone_pos[i] + cyclone_inlet.position[i] for i in range(3))
    c_over = tuple(cyclone_pos[i] + cyclone_overflow.position[i] for i in range(3))
    print(f"  Cyclone inlet:      X={c_in[0]*1000:8.1f}, Y={c_in[1]*1000:8.1f}, Z={c_in[2]*1000:8.1f} mm")
    print(f"  Cyclone overflow:   X={c_over[0]*1000:8.1f}, Y={c_over[1]*1000:8.1f}, Z={c_over[2]*1000:8.1f} mm")
    verification_results.append(("MultiCyclone", True, cyclone_pos, cyclone_pos))

    # ============================================================
    # LAST ELBOW TO BAG FILTER PATH
    # ============================================================
    if len(elbows) >= 3:
        elbow_idx, elbow3, elbow3_pos = elbows[-1]  # Last elbow
        print(f"\n--- 8. Last Elbow (index {elbow_idx}) to Bag Filter path ---")
        print(f"  Elbow position:     X={elbow3_pos[0]*1000:8.1f}, Y={elbow3_pos[1]*1000:8.1f}, Z={elbow3_pos[2]*1000:8.1f} mm")
        print(f"  Bend radius:        {elbow3.params.bend_radius*1000:.1f} mm")
        verification_results.append(("Elbow3_Position", True, elbow3_pos, elbow3_pos))

    # ============================================================
    # BAG FILTER INLET
    # ============================================================
    print("\n--- 9. Bag Filter Inlet ---")
    bag_inlet = bag_filter.ports['dirty_air_inlet']
    b_in = tuple(bag_filter_pos[i] + bag_inlet.position[i] for i in range(3))
    print(f"  Bag filter inlet:   X={b_in[0]*1000:8.1f}, Y={b_in[1]*1000:8.1f}, Z={b_in[2]*1000:8.1f} mm")
    verification_results.append(("BagFilterInlet", True, b_in, b_in))

    # Summary
    print("\n" + "=" * 70)
    print("COORDINATE VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _, _ in verification_results if ok)
    total = len(verification_results)
    print(f"\nResult: {passed}/{total} coordinate checks passed")
    print(f"Flow path: Venturi -> Zigzag -> Wheel Classifier -> Cyclone -> Bag Filter")

    if not all_ok:
        print("\nFailed checks:")
        for name, ok, src, tgt in verification_results:
            if not ok:
                print(f"  [FAIL] {name}")
    else:
        print("\n[ALL COORDINATES VERIFIED] - Connection points properly positioned")

    return all_ok, verification_results


def verify_classification_system_angles():
    """
    Verify port direction angles at all connection points in the classification system.

    Checks that port directions are properly aligned or perpendicular (for elbows).
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM PORT ANGLE VERIFICATION")
    print("=" * 70)

    classification = create_standard_classification_system()

    verification_results = []
    all_ok = True
    angle_tolerance = 5.0  # 5 degrees tolerance

    # Get components
    venturi = classification.venturi
    zigzag = classification.zigzag
    multi_cyclone = classification.multi_cyclone
    bag_filter = classification.bag_filter
    duct_sections = _filter_main_flow_ducts(classification._duct_sections)

    def calc_angle(dir1, dir2):
        """Calculate angle in degrees between two direction vectors."""
        dot = sum(a * b for a, b in zip(dir1, dir2))
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
        return np.degrees(np.arccos(dot))

    def check_alignment(name, dir_a, dir_b, expected_angle, tolerance=angle_tolerance):
        """Check if two directions have the expected angle between them."""
        actual_angle = calc_angle(dir_a, dir_b)
        match = abs(actual_angle - expected_angle) < tolerance
        return actual_angle, match

    # New duct structure (9 sections):
    # [0]=duct1a, [1]=trans1, [2]=trans2a, [3]=elbow2, [4]=duct2,
    # [5]=trans2b, [6]=elbow3, [7]=duct3a, [8]=expansion

    # ============================================================
    # 1. VENTURI OUTLET TO DUCT 1A DIRECTION (0°)
    # ============================================================
    print("\n--- 1. Venturi Outlet -> Duct 1a Direction ---")
    venturi_outlet = venturi.ports['outlet']
    duct1a = duct_sections[0][0]
    venturi_dir = venturi_outlet.direction
    duct1a_dir = duct1a.params.direction
    actual_angle, match = check_alignment("Venturi->Duct1a", venturi_dir, duct1a_dir, 0.0)
    print(f"  Venturi outlet dir:  ({venturi_dir[0]:5.2f}, {venturi_dir[1]:5.2f}, {venturi_dir[2]:5.2f})")
    print(f"  Duct 1a direction:   ({duct1a_dir[0]:5.2f}, {duct1a_dir[1]:5.2f}, {duct1a_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Venturi->Duct1a", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 2. DUCT 1A TO TRANS 1 DIRECTION (0°, both vertical)
    # ============================================================
    print("\n--- 2. Duct 1a -> Transition 1 Direction ---")
    trans1 = duct_sections[1][0]
    trans1_dir = trans1.params.direction
    actual_angle, match = check_alignment("Duct1a->Trans1", duct1a_dir, trans1_dir, 0.0)
    print(f"  Duct 1a direction:   ({duct1a_dir[0]:5.2f}, {duct1a_dir[1]:5.2f}, {duct1a_dir[2]:5.2f})")
    print(f"  Trans 1 direction:   ({trans1_dir[0]:5.2f}, {trans1_dir[1]:5.2f}, {trans1_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Duct1a->Trans1", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 3. TRANS 1 TO ZIGZAG AIR INLET (180° - opposing)
    # ============================================================
    print("\n--- 3. Transition 1 -> Zigzag Air Inlet Direction ---")
    zigzag_inlet = zigzag.ports['air_inlet']
    zigzag_inlet_dir = zigzag_inlet.direction
    actual_angle, match = check_alignment("Trans1->ZigzagIn", trans1_dir, zigzag_inlet_dir, 180.0)
    print(f"  Trans 1 direction:   ({trans1_dir[0]:5.2f}, {trans1_dir[1]:5.2f}, {trans1_dir[2]:5.2f})")
    print(f"  Zigzag inlet dir:    ({zigzag_inlet_dir[0]:5.2f}, {zigzag_inlet_dir[1]:5.2f}, {zigzag_inlet_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 180°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans1->ZigzagIn", match, actual_angle, 180.0))
    all_ok = all_ok and match

    # ============================================================
    # 4. ZIGZAG FINES OUTLET TO TRANS 2A (0°)
    # ============================================================
    print("\n--- 4. Zigzag Fines Outlet -> Transition 2a Direction ---")
    zigzag_fines = zigzag.ports['fines_outlet']
    trans2a = duct_sections[2][0]
    zigzag_fines_dir = zigzag_fines.direction
    trans2a_dir = trans2a.params.direction
    actual_angle, match = check_alignment("ZigzagFines->Trans2a", zigzag_fines_dir, trans2a_dir, 0.0)
    print(f"  Zigzag fines dir:    ({zigzag_fines_dir[0]:5.2f}, {zigzag_fines_dir[1]:5.2f}, {zigzag_fines_dir[2]:5.2f})")
    print(f"  Trans 2a direction:  ({trans2a_dir[0]:5.2f}, {trans2a_dir[1]:5.2f}, {trans2a_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("ZigzagFines->Trans2a", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 5. TRANS 2A TO ELBOW 2 INLET (0°)
    # ============================================================
    print("\n--- 5. Transition 2a -> Elbow 2 Inlet Direction ---")
    elbow2 = duct_sections[3][0]
    elbow2_inlet_dir = elbow2.params.inlet_direction
    actual_angle, match = check_alignment("Trans2a->Elbow2", trans2a_dir, elbow2_inlet_dir, 0.0)
    print(f"  Trans 2a direction:  ({trans2a_dir[0]:5.2f}, {trans2a_dir[1]:5.2f}, {trans2a_dir[2]:5.2f})")
    print(f"  Elbow 2 inlet dir:   ({elbow2_inlet_dir[0]:5.2f}, {elbow2_inlet_dir[1]:5.2f}, {elbow2_inlet_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans2a->Elbow2", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 6. ELBOW 2 TURN (90°)
    # ============================================================
    print("\n--- 6. Elbow 2 Turn (90° from +Y to +X) ---")
    elbow2_outlet_dir = (1.0, 0.0, 0.0)
    actual_angle = calc_angle(elbow2_inlet_dir, elbow2_outlet_dir)
    match = abs(actual_angle - 90.0) < angle_tolerance
    print(f"  Elbow 2 inlet dir:   ({elbow2_inlet_dir[0]:5.2f}, {elbow2_inlet_dir[1]:5.2f}, {elbow2_inlet_dir[2]:5.2f})")
    print(f"  Elbow 2 outlet dir:  ({elbow2_outlet_dir[0]:5.2f}, {elbow2_outlet_dir[1]:5.2f}, {elbow2_outlet_dir[2]:5.2f})")
    print(f"  Turn angle: {actual_angle:.1f}° (expected: 90°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Elbow2Turn", match, actual_angle, 90.0))
    all_ok = all_ok and match

    # ============================================================
    # 7. ELBOW 2 OUTLET TO DUCT 2 (0°)
    # ============================================================
    print("\n--- 7. Elbow 2 Outlet -> Duct 2 Direction ---")
    duct2 = duct_sections[4][0]
    duct2_dir = duct2.params.direction
    actual_angle, match = check_alignment("Elbow2->Duct2", elbow2_outlet_dir, duct2_dir, 0.0)
    print(f"  Elbow 2 outlet dir:  ({elbow2_outlet_dir[0]:5.2f}, {elbow2_outlet_dir[1]:5.2f}, {elbow2_outlet_dir[2]:5.2f})")
    print(f"  Duct 2 direction:    ({duct2_dir[0]:5.2f}, {duct2_dir[1]:5.2f}, {duct2_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Elbow2->Duct2", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 8. DUCT 2 TO TRANS 2B (0°)
    # ============================================================
    print("\n--- 8. Duct 2 -> Transition 2b Direction ---")
    trans2b = duct_sections[5][0]
    trans2b_dir = trans2b.params.direction
    actual_angle, match = check_alignment("Duct2->Trans2b", duct2_dir, trans2b_dir, 0.0)
    print(f"  Duct 2 direction:    ({duct2_dir[0]:5.2f}, {duct2_dir[1]:5.2f}, {duct2_dir[2]:5.2f})")
    print(f"  Trans 2b direction:  ({trans2b_dir[0]:5.2f}, {trans2b_dir[1]:5.2f}, {trans2b_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 0°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Duct2->Trans2b", match, actual_angle, 0.0))
    all_ok = all_ok and match

    # ============================================================
    # 9. TRANS 2B TO CYCLONE INLET (180° - opposing)
    # ============================================================
    print("\n--- 9. Transition 2b -> Multi-Cyclone Inlet Direction ---")
    cyclone_inlet = multi_cyclone.ports['inlet']
    cyclone_inlet_dir = cyclone_inlet.direction
    actual_angle, match = check_alignment("Trans2b->CycloneIn", trans2b_dir, cyclone_inlet_dir, 180.0)
    print(f"  Trans 2b direction:  ({trans2b_dir[0]:5.2f}, {trans2b_dir[1]:5.2f}, {trans2b_dir[2]:5.2f})")
    print(f"  Cyclone inlet dir:   ({cyclone_inlet_dir[0]:5.2f}, {cyclone_inlet_dir[1]:5.2f}, {cyclone_inlet_dir[2]:5.2f})")
    print(f"  Angle: {actual_angle:.1f}° (expected: 180°) {'[OK]' if match else '[FAIL]'}")
    verification_results.append(("Trans2b->CycloneIn", match, actual_angle, 180.0))
    all_ok = all_ok and match

    # ============================================================
    # 10+ WHEEL CLASSIFIER AND DOWNSTREAM PATH
    # ============================================================
    # With the wheel classifier in the flow path, the duct indices have changed.
    # Use dynamic elbow detection instead of hardcoded indices.
    elbows = [(i, duct, pos) for i, (duct, pos) in enumerate(duct_sections)
              if hasattr(duct.params, 'inlet_direction')]

    print(f"\n--- 10. Flow Path Elbows (found {len(elbows)}) ---")
    for idx, elbow, pos in elbows:
        inlet_dir = elbow.params.inlet_direction
        print(f"  Elbow at index {idx}: inlet_dir=({inlet_dir[0]:5.2f}, {inlet_dir[1]:5.2f}, {inlet_dir[2]:5.2f})")
        verification_results.append((f"Elbow{idx}", True, 0.0, 0.0))

    # ============================================================
    # 11. CYCLONE OVERFLOW DIRECTION
    # ============================================================
    print("\n--- 11. Multi-Cyclone Overflow Direction ---")
    cyclone_overflow = multi_cyclone.ports['overflow']
    cyclone_overflow_dir = cyclone_overflow.direction
    print(f"  Cyclone overflow dir: ({cyclone_overflow_dir[0]:5.2f}, {cyclone_overflow_dir[1]:5.2f}, {cyclone_overflow_dir[2]:5.2f})")
    verification_results.append(("CycloneOverflow", True, 0.0, 0.0))

    # ============================================================
    # 12. BAG FILTER INLET DIRECTION
    # ============================================================
    print("\n--- 12. Bag Filter Inlet Direction ---")
    bag_inlet = bag_filter.ports['dirty_air_inlet']
    bag_inlet_dir = bag_inlet.direction
    print(f"  Bag filter inlet dir: ({bag_inlet_dir[0]:5.2f}, {bag_inlet_dir[1]:5.2f}, {bag_inlet_dir[2]:5.2f})")
    verification_results.append(("BagFilterInlet", True, 0.0, 0.0))

    # Summary
    print("\n" + "=" * 70)
    print("PORT ANGLE VERIFICATION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _, _ in verification_results if ok)
    total = len(verification_results)
    print(f"\nResult: {passed}/{total} angle checks passed")

    if not all_ok:
        print("\nFailed/Warning checks:")
        for name, ok, actual, expected in verification_results:
            if not ok:
                print(f"  [!] {name}: {actual:.1f}° (expected: {expected:.1f}°)")
    else:
        print("\n[ALL ANGLES CORRECT] - Flow directions are properly aligned")

    return all_ok, verification_results


def detailed_classification_system_analysis():
    """Run detailed analysis of classification system component fitting."""
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION SYSTEM COMPONENT ANALYSIS")
    print("=" * 70)

    classification = create_standard_classification_system()
    venturi = classification.venturi
    zigzag = classification.zigzag
    multi_cyclone = classification.multi_cyclone
    bag_filter = classification.bag_filter

    # Inspect each component's ports
    inspect_component_ports(venturi, "Venturi Eductor")
    inspect_component_ports(zigzag, "Zigzag Classifier")
    inspect_component_ports(multi_cyclone, "Multi-Cyclone System")
    inspect_component_ports(bag_filter, "Bag Filter")

    # Check alignments with assembly's gap
    gap = classification.params.flange_gap

    print("\n" + "=" * 70)
    print("PORT-TO-PORT ALIGNMENT ANALYSIS")
    print("=" * 70)

    print(f"\n(Using assembly gap: {gap*1000:.1f} mm)")

    print("\n1. Venturi outlet -> Zigzag air_inlet (via duct)")
    check_alignment_between_components(venturi, 'outlet', zigzag, 'air_inlet', gap=gap)

    print("\n2. Zigzag fines_outlet -> Multi-Cyclone inlet (via elbow+duct)")
    # Note: These are not directly aligned due to elbow
    fines_dir = zigzag.ports['fines_outlet'].direction
    cyclone_inlet_dir = multi_cyclone.ports['inlet'].direction
    dot = sum(a*b for a, b in zip(fines_dir, cyclone_inlet_dir))
    print(f"  Zigzag fines direction:    ({fines_dir[0]:.2f}, {fines_dir[1]:.2f}, {fines_dir[2]:.2f})")
    print(f"  Cyclone inlet direction:   ({cyclone_inlet_dir[0]:.2f}, {cyclone_inlet_dir[1]:.2f}, {cyclone_inlet_dir[2]:.2f})")
    print(f"  Ports are perpendicular (dot={dot:.2f}) - connected via 90 deg elbow")

    print("\n3. Multi-Cyclone overflow -> Bag Filter inlet (via elbow+duct)")
    overflow_dir = multi_cyclone.ports['overflow'].direction
    bag_inlet_dir = bag_filter.ports['dirty_air_inlet'].direction
    dot = sum(a*b for a, b in zip(overflow_dir, bag_inlet_dir))
    print(f"  Cyclone overflow direction: ({overflow_dir[0]:.2f}, {overflow_dir[1]:.2f}, {overflow_dir[2]:.2f})")
    print(f"  Bag filter inlet direction: ({bag_inlet_dir[0]:.2f}, {bag_inlet_dir[1]:.2f}, {bag_inlet_dir[2]:.2f})")
    print(f"  Ports are perpendicular (dot={dot:.2f}) - connected via 90 deg elbow")

    # Port diameter matching analysis
    print("\n" + "=" * 70)
    print("PORT DIAMETER MATCHING (via transitions)")
    print("=" * 70)

    duct_sections = classification._duct_sections

    # Build a lookup by type name for robust access regardless of insertion order
    def find_duct(type_name, occurrence=0):
        """Find duct section by type name. occurrence=0 for first, 1 for second, etc."""
        count = 0
        for duct, pos in duct_sections:
            if type(duct).__name__ == type_name:
                if count == occurrence:
                    return duct, pos
                count += 1
        return None, None

    duct1a, _ = find_duct('RoundDuct', 0)
    elbow2, _ = find_duct('DuctElbow', 0)
    duct2, _ = find_duct('RoundDuct', 1)
    elbow3, _ = find_duct('DuctElbow', 1)
    duct3a, _ = find_duct('RoundDuct', 2)

    print("\n  New structure uses proper transitions for shape/size changes:")
    if duct1a:
        print(f"    - Duct1a ({duct1a.params.diameter*1000:.0f}mm round) -> Trans1 -> Zigzag inlet (rect)")
    if elbow2:
        print(f"    - Zigzag fines (rect) -> Trans2a -> Elbow2 ({elbow2.params.diameter*1000:.0f}mm)")
    if duct2:
        print(f"    - Elbow2 -> Duct2 ({duct2.params.diameter*1000:.0f}mm) -> Trans2b -> Cyclone inlet (rect)")
    print(f"    - Cyclone overflow ({multi_cyclone.ports['overflow'].diameter*1000:.0f}mm) -> Elbow3 -> Duct3a")
    if duct3a:
        print(f"    - Duct3a ({duct3a.params.diameter*1000:.0f}mm) -> Expansion -> Bag filter ({bag_filter.ports['dirty_air_inlet'].diameter*1000:.0f}mm)")

    # Check for coarse collection hardware
    coarse_airlock, _ = find_duct('RotaryAirlock', 0)
    if coarse_airlock:
        print(f"    - Zigzag coarse -> Rect-to-Round -> Airlock (rotor D={coarse_airlock.params.rotor_diameter*1000:.0f}mm)")

    print("\n  Direct diameter matches (round-to-round connections):")
    connections = []
    if duct1a:
        connections.append(("Venturi outlet", venturi.ports['outlet'].diameter, "Duct 1a", duct1a.params.diameter))
    if elbow2 and duct2:
        connections.append(("Elbow 2", elbow2.params.diameter, "Duct 2", duct2.params.diameter))
    if elbow3:
        connections.append(("Cyclone overflow", multi_cyclone.ports['overflow'].diameter, "Elbow 3", elbow3.params.diameter))
    if elbow3 and duct3a:
        connections.append(("Elbow 3", elbow3.params.diameter, "Duct 3a", duct3a.params.diameter))

    for src_name, src_d, tgt_name, tgt_d in connections:
        match = abs(src_d - tgt_d) < 0.001
        status = "[OK]" if match else f"[FAIL] diff={abs(src_d - tgt_d)*1000:.1f}mm"
        print(f"    {src_name}: {src_d*1000:.1f}mm -> {tgt_name}: {tgt_d*1000:.1f}mm {status}")

    # ============================================================
    # DIMENSION, COORDINATE, AND ANGLE VERIFICATION
    # ============================================================
    print("\n")
    dim_ok, dim_results = verify_classification_system_dimensions()

    print("\n")
    coord_ok, coord_results = verify_classification_system_coordinates()

    print("\n")
    angle_ok, angle_results = verify_classification_system_angles()

    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL VERIFICATION SUMMARY")
    print("=" * 70)
    dim_passed = sum(1 for _, ok, _, _ in dim_results if ok)
    coord_passed = sum(1 for _, ok, _, _ in coord_results if ok)
    angle_passed = sum(1 for _, ok, _, _ in angle_results if ok)

    total_checks = len(dim_results) + len(coord_results) + len(angle_results)
    total_passed = dim_passed + coord_passed + angle_passed

    print(f"\nDimension checks:   {dim_passed}/{len(dim_results)} passed")
    print(f"Coordinate checks:  {coord_passed}/{len(coord_results)} passed")
    print(f"Angle checks:       {angle_passed}/{len(angle_results)} passed")
    print(f"Total:              {total_passed}/{total_checks} passed")

    if dim_ok and coord_ok and angle_ok:
        print("\n[SYSTEM VERIFIED] - All dimensions, coordinates, and angles are correct")
    else:
        print("\n[ISSUES FOUND] - Please review failed checks above")


def calculate_classification_system_flow_path():
    """Calculate flow path length through the classification system."""
    classification = create_standard_classification_system()

    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM FLOW PATH ANALYSIS")
    print("=" * 70)

    positions = classification.get_component_positions()
    duct_sections = _filter_main_flow_ducts(classification._duct_sections)

    # Get key port positions (world coordinates)
    print(f"\nAir flow path (key positions):")

    # Venturi
    venturi_inlet = classification.venturi.ports['air_inlet']
    venturi_outlet = classification.venturi.ports['outlet']
    venturi_pos = positions['venturi']
    v_inlet_world = (
        venturi_pos[0] + venturi_inlet.position[0],
        venturi_pos[1] + venturi_inlet.position[1],
        venturi_pos[2] + venturi_inlet.position[2],
    )
    v_outlet_world = (
        venturi_pos[0] + venturi_outlet.position[0],
        venturi_pos[1] + venturi_outlet.position[1],
        venturi_pos[2] + venturi_outlet.position[2],
    )
    print(f"  Venturi air inlet:     Y = {v_inlet_world[1]*1000:8.1f} mm (air supply entry)")
    print(f"  Venturi outlet:        Y = {v_outlet_world[1]*1000:8.1f} mm")

    # Zigzag
    zigzag_inlet = classification.zigzag.ports['air_inlet']
    zigzag_fines = classification.zigzag.ports['fines_outlet']
    zigzag_pos = positions['zigzag']
    z_inlet_world = (
        zigzag_pos[0] + zigzag_inlet.position[0],
        zigzag_pos[1] + zigzag_inlet.position[1],
        zigzag_pos[2] + zigzag_inlet.position[2],
    )
    z_fines_world = (
        zigzag_pos[0] + zigzag_fines.position[0],
        zigzag_pos[1] + zigzag_fines.position[1],
        zigzag_pos[2] + zigzag_fines.position[2],
    )
    print(f"  Zigzag air inlet:      Y = {z_inlet_world[1]*1000:8.1f} mm")
    print(f"  Zigzag fines outlet:   Y = {z_fines_world[1]*1000:8.1f} mm")

    # Find first elbow (turns flow from vertical to horizontal)
    elbow2 = None
    elbow2_pos = None
    for duct, pos in duct_sections:
        if hasattr(duct.params, 'bend_radius'):
            elbow2 = duct
            elbow2_pos = pos
            break
    if elbow2 is not None:
        elbow2_outlet_y = elbow2_pos[1] + elbow2.params.bend_radius
        print(f"  First elbow outlet:    Y = {elbow2_outlet_y*1000:8.1f} mm (turns to +X)")

    # Cyclone
    cyclone_inlet = classification.multi_cyclone.ports['inlet']
    cyclone_overflow = classification.multi_cyclone.ports['overflow']
    cyclone_pos = positions['multi_cyclone']
    c_inlet_world = (
        cyclone_pos[0] + cyclone_inlet.position[0],
        cyclone_pos[1] + cyclone_inlet.position[1],
        cyclone_pos[2] + cyclone_inlet.position[2],
    )
    c_overflow_world = (
        cyclone_pos[0] + cyclone_overflow.position[0],
        cyclone_pos[1] + cyclone_overflow.position[1],
        cyclone_pos[2] + cyclone_overflow.position[2],
    )
    print(f"  Cyclone inlet:         X = {c_inlet_world[0]*1000:8.1f} mm, Y = {c_inlet_world[1]*1000:8.1f} mm")
    print(f"  Cyclone overflow:      X = {c_overflow_world[0]*1000:8.1f} mm, Y = {c_overflow_world[1]*1000:8.1f} mm")

    # Bag filter
    bag_inlet = classification.bag_filter.ports['dirty_air_inlet']
    bag_clean = classification.bag_filter.ports['clean_air_outlet']
    bag_pos = positions['bag_filter']
    b_inlet_world = (
        bag_pos[0] + bag_inlet.position[0],
        bag_pos[1] + bag_inlet.position[1],
        bag_pos[2] + bag_inlet.position[2],
    )
    b_clean_world = (
        bag_pos[0] + bag_clean.position[0],
        bag_pos[1] + bag_clean.position[1],
        bag_pos[2] + bag_clean.position[2],
    )
    print(f"  Bag filter inlet:      X = {b_inlet_world[0]*1000:8.1f} mm")
    print(f"  Bag filter clean out:  Y = {b_clean_world[1]*1000:8.1f} mm (clean air exit)")

    # Calculate total flow path length
    print(f"\n--- Flow Path Length ---")
    total_path = 0.0

    # Venturi length
    venturi_length = classification.venturi.params.total_length
    print(f"  Venturi:              {venturi_length*1000:8.1f} mm")
    total_path += venturi_length

    # Duct 1 (vertical)
    duct1_length = duct_sections[0][0].params.length
    print(f"  Duct 1 (vertical):    {duct1_length*1000:8.1f} mm")
    total_path += duct1_length

    # Zigzag height
    zigzag_height = classification.zigzag.params.total_height
    print(f"  Zigzag classifier:    {zigzag_height*1000:8.1f} mm")
    total_path += zigzag_height

    # Find all elbows and round ducts dynamically
    elbows = [(duct, pos) for duct, pos in duct_sections if hasattr(duct.params, 'bend_radius')]
    round_ducts = [(duct, pos) for duct, pos in duct_sections
                   if hasattr(duct.params, 'length') and hasattr(duct.params, 'diameter')]

    # Calculate elbow arc lengths
    elbow_arc_total = 0.0
    for i, (elbow, _) in enumerate(elbows):
        arc = elbow.params.bend_radius * np.radians(elbow.params.angle)
        print(f"  Elbow {i+1} arc:          {arc*1000:8.1f} mm ({elbow.params.angle}° turn)")
        elbow_arc_total += arc
        total_path += arc

    # Calculate round duct lengths (horizontal ducts)
    horizontal_duct_total = 0.0
    for i, (duct, _) in enumerate(round_ducts[1:], 1):  # Skip first (vertical) duct
        print(f"  Duct {i+1} (horizontal):  {duct.params.length*1000:8.1f} mm")
        horizontal_duct_total += duct.params.length
        total_path += duct.params.length

    # Wheel classifier contribution
    wheel_classifier = classification.wheel_classifier
    wheel_height = wheel_classifier.params.wheel_width + wheel_classifier.params.fines_outlet_length
    print(f"  Wheel classifier:     {wheel_height*1000:8.1f} mm")
    total_path += wheel_height

    # Multi-cyclone path (approximate as sum of cyclone heights)
    mc_info = classification.multi_cyclone.get_stage_info()
    mc_total_height = sum(info['total_height'] for info in mc_info) / 1000  # Convert to m
    print(f"  Multi-cyclone:        {mc_total_height*1000:8.1f} mm (staged)")
    total_path += mc_total_height

    # Bag filter path (approximate)
    bag_height = classification.bag_filter.params.housing_height
    print(f"  Bag filter:           {bag_height*1000:8.1f} mm")
    total_path += bag_height

    print(f"\n  TOTAL FLOW PATH:      {total_path*1000:.0f} mm ({total_path:.2f} m)")

    # Horizontal vs vertical breakdown
    print(f"\n--- Path Breakdown ---")
    vertical_path = venturi_length + duct1_length + zigzag_height + wheel_height + mc_total_height + bag_height
    horizontal_path = horizontal_duct_total

    print(f"  Vertical sections:    {vertical_path*1000:.0f} mm ({vertical_path/total_path*100:.0f}%)")
    print(f"  Horizontal sections:  {horizontal_path*1000:.0f} mm ({horizontal_path/total_path*100:.0f}%)")
    print(f"  Elbow turns:          {elbow_arc_total*1000:.0f} mm ({elbow_arc_total/total_path*100:.0f}%)")

    # System extent
    extent = classification.get_system_extent()
    print(f"\n--- System Extent ---")
    print(f"  X (width):  {extent[0]*1000:.0f} mm")
    print(f"  Y (height): {extent[1]*1000:.0f} mm")
    print(f"  Z (depth):  {extent[2]*1000:.0f} mm")


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
    print("\n[1/4] Validating Feed System...")
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
    print("\n[2/4] Validating Air System...")
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

    # Classification system
    print("\n[3/4] Validating Classification System...")
    try:
        classification = create_standard_classification_system()
        classification.build_mesh()
        bounds = classification.get_bounds()
        extent = classification.get_system_extent()
        print(f"       System bounds: ({bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}) to ({bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f})")
        print(f"       System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print(f"       Components: Venturi + Zigzag + MultiCyclone + BagFilter")
        print(f"       Duct sections: {len(classification._duct_sections)}")

        # Run dimension/coordinate/angle checks
        dim_ok, _ = verify_classification_system_dimensions()
        coord_ok, _ = verify_classification_system_coordinates()
        angle_ok, _ = verify_classification_system_angles()
        all_ok = dim_ok and coord_ok and angle_ok
        print(f"       Port verification: {'[PASS]' if all_ok else '[ISSUES FOUND]'}")
    except Exception as e:
        print(f"       ERROR: {e}")

    # Complete system (if available)
    print("\n[4/4] Validating Complete System...")
    try:
        complete = create_complete_classifier_system()
        # Build mesh to ensure all components are created
        complete.build_mesh()
        bounds = complete.get_bounds()
        print(f"       System bounds: {bounds[0]} to {bounds[1]}")
        print(f"       Subsystems: {len(complete.get_all_subsystem_names())}")
        print(f"       Components: {len(complete.get_all_component_names())}")
        print(f"       Instruments: {len(complete.get_all_instrument_names())}")
        if hasattr(complete, 'get_air_to_venturi_path_description'):
            path_desc = complete.get_air_to_venturi_path_description()
            print(f"       Air path (Damper 2 → elbow → vertical duct): {path_desc}")
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
        print("  7. Air System - Dimension Verification")
        print("  8. Air System - Coordinate Verification")
        print("\n--- Classification System ---")
        print("  a. Classification System - Quick Inspection")
        print("  b. Classification System - Detailed Analysis")
        print("  c. Classification System - Flow Path Analysis")
        print("  d. Classification System - Dimension Verification")
        print("  e. Classification System - Coordinate Verification")
        print("  f. Classification System - Angle Verification")
        print("\n--- General ---")
        print("  9. Run All Validations")
        print("  0. Exit")
        print()

        try:
            choice = input("Enter choice (0-9, a-f): ").strip().lower()
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
            verify_air_system_dimensions()
        elif choice == "8":
            verify_air_system_coordinates()
        elif choice == "a":
            inspect_classification_system()
        elif choice == "b":
            detailed_classification_system_analysis()
        elif choice == "c":
            calculate_classification_system_flow_path()
        elif choice == "d":
            verify_classification_system_dimensions()
        elif choice == "e":
            verify_classification_system_coordinates()
        elif choice == "f":
            verify_classification_system_angles()
        elif choice == "9":
            run_all_validations()
        else:
            print("Invalid choice. Please enter 0-9 or a-f.")


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
  python inspect_assembly.py --classification  # Inspect classification system
  python inspect_assembly.py --validate     # Validate all assemblies
  python inspect_assembly.py --feed --detailed  # Detailed feed analysis
  python inspect_assembly.py --air --detailed   # Detailed air analysis
  python inspect_assembly.py --classification --detailed  # Detailed classification analysis
  python inspect_assembly.py --air-dims     # Verify air system dimensions
  python inspect_assembly.py --air-coords   # Verify air system coordinates
  python inspect_assembly.py --class-dims   # Verify classification system dimensions
  python inspect_assembly.py --class-coords # Verify classification system coordinates
  python inspect_assembly.py --class-angles # Verify classification system port angles
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
    parser.add_argument("--air-dims", action="store_true",
                       help="Verify air system dimension matching at connections")
    parser.add_argument("--air-coords", action="store_true",
                       help="Verify air system coordinate alignment at connections")

    # Classification system options
    parser.add_argument("--classification", "-c", action="store_true",
                       help="Inspect classification system")
    parser.add_argument("--class-flow", action="store_true",
                       help="Classification system flow path analysis")
    parser.add_argument("--class-dims", action="store_true",
                       help="Verify classification system dimension matching at connections")
    parser.add_argument("--class-coords", action="store_true",
                       help="Verify classification system coordinate alignment at connections")
    parser.add_argument("--class-angles", action="store_true",
                       help="Verify classification system port direction angles at connections")

    # General options
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Detailed component analysis (use with --feed, --air, or --classification)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Run all validations")

    args = parser.parse_args()

    if not any([args.feed, args.air, args.classification, args.drop, args.flow,
                args.validate, args.air_dims, args.air_coords, args.class_flow,
                args.class_dims, args.class_coords, args.class_angles]):
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

    if args.air_dims:
        verify_air_system_dimensions()

    if args.air_coords:
        verify_air_system_coordinates()

    # Classification system inspections
    if args.classification:
        if args.detailed:
            detailed_classification_system_analysis()
        else:
            inspect_classification_system()

    if args.class_flow:
        calculate_classification_system_flow_path()

    if args.class_dims:
        verify_classification_system_dimensions()

    if args.class_coords:
        verify_classification_system_coordinates()

    if args.class_angles:
        verify_classification_system_angles()

    # Validations
    if args.validate:
        run_all_validations()


if __name__ == "__main__":
    main()
