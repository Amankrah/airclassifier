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

Usage:
    python examples/inspect_assembly.py                  # Interactive menu
    python examples/inspect_assembly.py --feed           # Inspect feed system
    python examples/inspect_assembly.py --air            # Inspect air system
    python examples/inspect_assembly.py --feed --detailed # Detailed feed analysis
    python examples/inspect_assembly.py --air --detailed  # Detailed air analysis
    python examples/inspect_assembly.py --drop           # Feed system vertical drop
    python examples/inspect_assembly.py --flow           # Air system flow path
    python examples/inspect_assembly.py --air-dims       # Verify air system dimensions
    python examples/inspect_assembly.py --air-coords     # Verify air system coordinates
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
    print("\n--- 1. Filter Outlet → Horizontal Duct ---")
    filter_outlet = inlet_filter.ports['outlet']
    horiz_duct = duct_sections[0][0]  # First duct section

    filter_d = filter_outlet.diameter * 1000
    duct_d = horiz_duct.params.diameter * 1000
    match = abs(filter_d - duct_d) < 1.0  # 1mm tolerance

    print(f"  Filter outlet diameter:  {filter_d:8.1f} mm")
    print(f"  Horizontal duct diameter: {duct_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(filter_d - duct_d):.1f} mm)")
    verification_results.append(("Filter→HorizDuct", match, filter_d, duct_d))
    all_ok = all_ok and match

    # ============================================================
    # 2. HORIZONTAL DUCT TO ELBOW
    # ============================================================
    print("\n--- 2. Horizontal Duct → 90° Elbow ---")
    elbow = duct_sections[1][0]

    horiz_d = horiz_duct.params.diameter * 1000
    elbow_d = elbow.params.diameter * 1000
    match = abs(horiz_d - elbow_d) < 1.0

    print(f"  Horizontal duct diameter: {horiz_d:8.1f} mm")
    print(f"  Elbow diameter:           {elbow_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(horiz_d - elbow_d):.1f} mm)")
    verification_results.append(("HorizDuct→Elbow", match, horiz_d, elbow_d))
    all_ok = all_ok and match

    # ============================================================
    # 3. ELBOW TO VERTICAL DUCT
    # ============================================================
    print("\n--- 3. 90° Elbow → Vertical Duct ---")
    vert_duct = duct_sections[2][0]

    vert_d = vert_duct.params.diameter * 1000
    match = abs(elbow_d - vert_d) < 1.0

    print(f"  Elbow diameter:          {elbow_d:8.1f} mm")
    print(f"  Vertical duct diameter:  {vert_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(elbow_d - vert_d):.1f} mm)")
    verification_results.append(("Elbow→VertDuct", match, elbow_d, vert_d))
    all_ok = all_ok and match

    # ============================================================
    # 4. VERTICAL DUCT TO BLOWER INLET
    # ============================================================
    print("\n--- 4. Vertical Duct → Blower Inlet Bell ---")
    blower_inlet = blower.ports['inlet']

    blower_in_d = blower_inlet.diameter * 1000
    match = abs(vert_d - blower_in_d) < 5.0  # 5mm tolerance (inlet bell may be larger)

    print(f"  Vertical duct diameter:  {vert_d:8.1f} mm")
    print(f"  Blower inlet diameter:   {blower_in_d:8.1f} mm")
    print(f"  Match: {'[OK]' if match else '[WARN]'} (diff: {abs(vert_d - blower_in_d):.1f} mm)")
    print(f"  Note: Inlet bell may be slightly larger for smooth air entry")
    verification_results.append(("VertDuct→BlowerIn", match, vert_d, blower_in_d))
    all_ok = all_ok and match

    # ============================================================
    # 5. BLOWER OUTLET TO TRANSITION (DIRECT CONNECTION)
    # ============================================================
    print("\n--- 5. Blower Outlet → Rect-to-Round Transition ---")
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
    verification_results.append(("BlowerOut→Transition", match, f"{blower_out_w}x{blower_out_h}", f"{trans_rect_w}x{trans_rect_h}"))
    all_ok = all_ok and match

    # ============================================================
    # 6. TRANSITION ROUND END TO DUCT
    # ============================================================
    print("\n--- 6. Transition Round End → Round Duct ---")
    trans_round_d = transition.params.round_diameter * 1000

    if len(duct_sections) > 4:
        round_duct = duct_sections[4][0]
        round_duct_d = round_duct.params.diameter * 1000
        match = abs(trans_round_d - round_duct_d) < 1.0

        print(f"  Transition round outlet: {trans_round_d:8.1f} mm")
        print(f"  Round duct diameter:     {round_duct_d:8.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(trans_round_d - round_duct_d):.1f} mm)")
        verification_results.append(("Transition→RoundDuct", match, trans_round_d, round_duct_d))
        all_ok = all_ok and match
    else:
        print(f"  Transition round outlet: {trans_round_d:8.1f} mm")
        print(f"  (Direct connection to damper)")

    # ============================================================
    # 7. ROUND DUCT/TRANSITION TO DAMPER
    # ============================================================
    if dampers:
        print("\n--- 7. Round Duct → Damper 1 Inlet ---")
        damper_inlet = dampers[0].ports['inlet']
        damper_in_d = damper_inlet.diameter * 1000

        match = abs(trans_round_d - damper_in_d) < 1.0

        print(f"  Transition/duct outlet:  {trans_round_d:8.1f} mm")
        print(f"  Damper 1 inlet diameter: {damper_in_d:8.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(trans_round_d - damper_in_d):.1f} mm)")
        verification_results.append(("Duct→Damper1", match, trans_round_d, damper_in_d))
        all_ok = all_ok and match

        # Damper-to-damper connections
        for i in range(len(dampers) - 1):
            print(f"\n--- {8+i}. Damper {i+1} Outlet → Damper {i+2} Inlet ---")
            d1_out = dampers[i].ports['outlet']
            d2_in = dampers[i+1].ports['inlet']

            d1_out_d = d1_out.diameter * 1000
            d2_in_d = d2_in.diameter * 1000
            match = abs(d1_out_d - d2_in_d) < 1.0

            print(f"  Damper {i+1} outlet diameter: {d1_out_d:8.1f} mm")
            print(f"  Damper {i+2} inlet diameter:  {d2_in_d:8.1f} mm")
            print(f"  Match: {'[OK]' if match else '[FAIL]'} (diff: {abs(d1_out_d - d2_in_d):.1f} mm)")
            verification_results.append((f"Damper{i+1}→Damper{i+2}", match, d1_out_d, d2_in_d))
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
    print("\n--- 1. Filter Outlet → Horizontal Duct Start ---")
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
    verification_results.append(("FilterOut→HorizDuct", match, filter_outlet_world, horiz_duct_pos))
    all_ok = all_ok and match

    # ============================================================
    # 2. HORIZONTAL DUCT END TO ELBOW INLET
    # ============================================================
    print("\n--- 2. Horizontal Duct End → Elbow Inlet ---")
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
    verification_results.append(("HorizDuctEnd→Elbow", match, horiz_duct_end, elbow_pos))
    all_ok = all_ok and match

    # ============================================================
    # 3. ELBOW OUTLET TO VERTICAL DUCT START
    # ============================================================
    print("\n--- 3. Elbow Outlet → Vertical Duct Start ---")
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
    verification_results.append(("ElbowOut→VertDuct", match, elbow_outlet, vert_duct_pos))
    all_ok = all_ok and match

    # ============================================================
    # 4. VERTICAL DUCT END TO BLOWER INLET BELL
    # ============================================================
    print("\n--- 4. Vertical Duct End → Blower Inlet Bell ---")
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
    verification_results.append(("VertDuctEnd→InletBell", match, vert_duct_end, (blower_pos[0], blower_pos[1], inlet_bell_z)))
    all_ok = all_ok and match

    # ============================================================
    # 5. BLOWER OUTLET FLANGE TO TRANSITION (DIRECT CONNECTION)
    # ============================================================
    print("\n--- 5. Blower Outlet Flange → Transition Start ---")
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
    verification_results.append(("BlowerFlange→Transition", match, blower_outlet_flange, trans_pos))
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
        print("\n--- 6. Transition End → Round Duct Start ---")
        round_duct, round_duct_pos = duct_sections[4]

        diff_x = abs(trans_end[0] - round_duct_pos[0])
        diff_y = abs(trans_end[1] - round_duct_pos[1])
        diff_z = abs(trans_end[2] - round_duct_pos[2])
        match = diff_x < tolerance and diff_y < tolerance and diff_z < tolerance

        print(f"  Transition end:     X={trans_end[0]*1000:8.1f}, Y={trans_end[1]*1000:8.1f}, Z={trans_end[2]*1000:8.1f} mm")
        print(f"  Round duct start:   X={round_duct_pos[0]*1000:8.1f}, Y={round_duct_pos[1]*1000:8.1f}, Z={round_duct_pos[2]*1000:8.1f} mm")
        print(f"  Difference:         dX={diff_x*1000:.1f}, dY={diff_y*1000:.1f}, dZ={diff_z*1000:.1f} mm")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
        verification_results.append(("TransitionEnd→RoundDuct", match, trans_end, round_duct_pos))
        all_ok = all_ok and match

    # ============================================================
    # 7. DAMPER CONNECTIONS
    # ============================================================
    if dampers and damper_positions:
        print("\n--- 7. Duct → Damper 1 Inlet ---")
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
        verification_results.append(("Duct→Damper1Inlet", match, duct_end_to_damper, damper1_inlet_world))
        all_ok = all_ok and match

        # Check damper-to-damper coordinate alignment
        for i in range(len(dampers) - 1):
            print(f"\n--- {8+i}. Damper {i+1} Outlet → Damper {i+2} Inlet ---")
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
            verification_results.append((f"Damper{i+1}Out→Damper{i+2}In", match, d1_outlet_world, d2_inlet_world))
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
        print("  7. Air System - Dimension Verification")
        print("  8. Air System - Coordinate Verification")
        print("\n--- General ---")
        print("  9. Run All Validations")
        print("  0. Exit")
        print()

        try:
            choice = input("Enter choice (0-9): ").strip()
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
        elif choice == "9":
            run_all_validations()
        else:
            print("Invalid choice. Please enter 0-9.")


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
  python inspect_assembly.py --air-dims     # Verify air system dimensions
  python inspect_assembly.py --air-coords   # Verify air system coordinates
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

    # General options
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Detailed component analysis (use with --feed or --air)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Run all validations")

    args = parser.parse_args()

    if not any([args.feed, args.air, args.drop, args.flow, args.validate,
                args.air_dims, args.air_coords]):
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

    # Validations
    if args.validate:
        run_all_validations()


if __name__ == "__main__":
    main()
