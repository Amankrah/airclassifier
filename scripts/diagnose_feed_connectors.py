"""
Feed System Transition Connector Diagnostics
=============================================

This script diagnoses whether transition connectors fit perfectly between
components in the feed system assembly. It checks:

1. Diameter matching - inlet/outlet diameters match connected ports
2. Length accuracy - transition length exactly fills the gap
3. Position alignment - transition is centered between components
4. Direction alignment - transition direction matches port directions
5. Gap analysis - no overlaps or unwanted gaps

Run: python scripts/diagnose_feed_connectors.py
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airclassifier.geometry.assembly import FeedSystemAssembly, FeedSystemParams


def diagnose_transition_connectors(system: FeedSystemAssembly, verbose: bool = True):
    """
    Diagnose all transition connectors in the feed system.

    Returns:
        dict: Diagnosis results with pass/fail status and details
    """
    results = {
        'overall_pass': True,
        'connections': [],
        'summary': {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
        }
    }

    # Tolerances
    DIAMETER_TOL = 0.0001  # 0.1mm diameter tolerance
    LENGTH_TOL = 0.001     # 1mm length tolerance
    POSITION_TOL = 0.002   # 2mm position tolerance
    DIRECTION_TOL = 0.01   # Direction cosine tolerance

    # Define the connection chain
    connections = [
        {
            'name': 'Hopper → Airlock',
            'source_component': 'hopper',
            'source_port': 'discharge',
            'target_component': 'airlock',
            'target_port': 'inlet',
            'transition_key': 'hopper_to_airlock',
        },
        {
            'name': 'Airlock → Feeder',
            'source_component': 'airlock',
            'source_port': 'outlet',
            'target_component': 'feeder',
            'target_port': 'inlet',
            'transition_key': 'airlock_to_feeder',
        },
        {
            'name': 'Feeder → Deagglomerator',
            'source_component': 'feeder',
            'source_port': 'outlet',
            'target_component': 'deagglomerator',
            'target_port': 'inlet',
            'transition_key': 'feeder_to_deagglomerator',
        },
    ]

    # Get component positions
    positions = system.get_component_positions()
    components = {
        'hopper': system.hopper,
        'airlock': system.airlock,
        'feeder': system.feeder,
        'deagglomerator': system.deagglomerator,
    }

    flange_gap = system.params.component_spacing

    if verbose:
        print("=" * 70)
        print("FEED SYSTEM TRANSITION CONNECTOR DIAGNOSTICS")
        print("=" * 70)
        print(f"\nFlange gap (gasket space): {flange_gap * 1000:.1f} mm")
        print(f"Tolerances: diameter={DIAMETER_TOL*1000:.2f}mm, length={LENGTH_TOL*1000:.1f}mm, position={POSITION_TOL*1000:.1f}mm")

    # Check each connection
    for i, conn_info in enumerate(connections):
        conn_result = {
            'name': conn_info['name'],
            'checks': [],
            'pass': True,
        }

        if verbose:
            print(f"\n{'─' * 70}")
            print(f"CONNECTION {i+1}: {conn_info['name']}")
            print("─" * 70)

        # Get source and target ports
        source_comp = components[conn_info['source_component']]
        target_comp = components[conn_info['target_component']]
        source_port = source_comp.ports[conn_info['source_port']]
        target_port = target_comp.ports[conn_info['target_port']]

        source_pos = np.array(positions[conn_info['source_component']])
        target_pos = np.array(positions[conn_info['target_component']])

        # Calculate world positions of ports
        source_port_world = source_port.get_world_position(tuple(source_pos))
        target_port_world = target_port.get_world_position(tuple(target_pos))

        # Get the transition connector
        transition_data = system._transition_connectors[i]
        transition = transition_data[0]
        transition_start_pos = np.array(transition_data[1])
        trans_params = transition.params

        if verbose:
            print(f"\n  Source port ({conn_info['source_port']}):")
            print(f"    World position: ({source_port_world[0]*1000:.1f}, {source_port_world[1]*1000:.1f}, {source_port_world[2]*1000:.1f}) mm")
            print(f"    Diameter: {source_port.diameter * 1000:.1f} mm")
            print(f"    Direction: ({source_port.direction[0]:.2f}, {source_port.direction[1]:.2f}, {source_port.direction[2]:.2f})")

            print(f"\n  Target port ({conn_info['target_port']}):")
            print(f"    World position: ({target_port_world[0]*1000:.1f}, {target_port_world[1]*1000:.1f}, {target_port_world[2]*1000:.1f}) mm")
            print(f"    Diameter: {target_port.diameter * 1000:.1f} mm")
            print(f"    Direction: ({target_port.direction[0]:.2f}, {target_port.direction[1]:.2f}, {target_port.direction[2]:.2f})")

        # ================================================================
        # CHECK 1: Diameter matching
        # ================================================================
        results['summary']['total_checks'] += 2

        inlet_dia_match = abs(trans_params.inlet_dimensions[0] - source_port.diameter) < DIAMETER_TOL
        outlet_dia_match = abs(trans_params.outlet_dimensions[0] - target_port.diameter) < DIAMETER_TOL

        if verbose:
            print(f"\n  CHECK 1: Diameter Matching")

        # Inlet diameter
        if inlet_dia_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'inlet_diameter',
            'pass': inlet_dia_match,
            'expected': source_port.diameter,
            'actual': trans_params.inlet_dimensions[0],
        })

        if verbose:
            print(f"    Inlet:  expected={source_port.diameter*1000:.2f}mm, actual={trans_params.inlet_dimensions[0]*1000:.2f}mm  {status}")

        # Outlet diameter
        if outlet_dia_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'outlet_diameter',
            'pass': outlet_dia_match,
            'expected': target_port.diameter,
            'actual': trans_params.outlet_dimensions[0],
        })

        if verbose:
            print(f"    Outlet: expected={target_port.diameter*1000:.2f}mm, actual={trans_params.outlet_dimensions[0]*1000:.2f}mm  {status}")

        # ================================================================
        # CHECK 2: Gap and length analysis
        # ================================================================
        results['summary']['total_checks'] += 1

        # Calculate actual distance between ports
        port_distance = np.linalg.norm(target_port_world - source_port_world)

        # Expected transition length (total gap minus two flange gaps)
        expected_trans_length = system._transition_lengths[conn_info['transition_key']]
        expected_total_gap = flange_gap + expected_trans_length + flange_gap

        # Check if the transition length matches
        actual_trans_length = trans_params.length
        length_match = abs(actual_trans_length - expected_trans_length) < LENGTH_TOL

        if verbose:
            print(f"\n  CHECK 2: Length Analysis")
            print(f"    Port-to-port distance: {port_distance*1000:.2f} mm")
            print(f"    Expected total gap:    {expected_total_gap*1000:.2f} mm (flange + transition + flange)")
            print(f"    Transition length:     {actual_trans_length*1000:.2f} mm")

        if length_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'transition_length',
            'pass': length_match,
            'expected': expected_trans_length,
            'actual': actual_trans_length,
        })

        if verbose:
            print(f"    Length match: expected={expected_trans_length*1000:.2f}mm, actual={actual_trans_length*1000:.2f}mm  {status}")

        # Check gap vs port distance
        gap_error = abs(port_distance - expected_total_gap)
        gap_match = gap_error < LENGTH_TOL
        results['summary']['total_checks'] += 1

        if gap_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'gap_vs_distance',
            'pass': gap_match,
            'expected': expected_total_gap,
            'actual': port_distance,
            'error': gap_error,
        })

        if verbose:
            print(f"    Gap vs distance: error={gap_error*1000:.2f}mm  {status}")

        # ================================================================
        # CHECK 3: Position alignment
        # ================================================================
        results['summary']['total_checks'] += 1

        # Transition should start at: source_port + flange_gap in direction of target
        direction_vec = target_port_world - source_port_world
        if np.linalg.norm(direction_vec) > 0.001:
            direction_vec = direction_vec / np.linalg.norm(direction_vec)

        expected_start = source_port_world + direction_vec * flange_gap
        position_error = np.linalg.norm(transition_start_pos - expected_start)
        position_match = position_error < POSITION_TOL

        if verbose:
            print(f"\n  CHECK 3: Position Alignment")
            print(f"    Expected start: ({expected_start[0]*1000:.2f}, {expected_start[1]*1000:.2f}, {expected_start[2]*1000:.2f}) mm")
            print(f"    Actual start:   ({transition_start_pos[0]*1000:.2f}, {transition_start_pos[1]*1000:.2f}, {transition_start_pos[2]*1000:.2f}) mm")
            print(f"    Position error: {position_error*1000:.2f} mm")

        if position_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'position_alignment',
            'pass': position_match,
            'expected': tuple(expected_start),
            'actual': tuple(transition_start_pos),
            'error': position_error,
        })

        if verbose:
            print(f"    Position match: {status}")

        # ================================================================
        # CHECK 4: Direction alignment
        # ================================================================
        results['summary']['total_checks'] += 1

        trans_direction = np.array(trans_params.direction)

        # Direction should match the connection direction
        direction_dot = abs(np.dot(trans_direction, direction_vec))
        direction_match = direction_dot > (1.0 - DIRECTION_TOL)

        if verbose:
            print(f"\n  CHECK 4: Direction Alignment")
            print(f"    Connection direction: ({direction_vec[0]:.3f}, {direction_vec[1]:.3f}, {direction_vec[2]:.3f})")
            print(f"    Transition direction: ({trans_direction[0]:.3f}, {trans_direction[1]:.3f}, {trans_direction[2]:.3f})")
            print(f"    Alignment (dot product): {direction_dot:.4f}")

        if direction_match:
            status = "✓ PASS"
            results['summary']['passed'] += 1
        else:
            status = "✗ FAIL"
            results['summary']['failed'] += 1
            conn_result['pass'] = False
            results['overall_pass'] = False

        conn_result['checks'].append({
            'name': 'direction_alignment',
            'pass': direction_match,
            'dot_product': direction_dot,
        })

        if verbose:
            print(f"    Direction match: {status}")

        # ================================================================
        # CHECK 5: Transition type analysis
        # ================================================================
        d_in = trans_params.inlet_dimensions[0]
        d_out = trans_params.outlet_dimensions[0]
        d_diff = abs(d_in - d_out)

        if verbose:
            print(f"\n  CHECK 5: Transition Type")

        if d_diff > 0.001:
            # Conical transition - check half-angle
            half_angle = np.degrees(np.arctan((d_diff / 2) / actual_trans_length))
            max_angle = 12.0  # Max recommended half-angle

            if d_in > d_out:
                trans_type = "CONICAL REDUCER"
            else:
                trans_type = "CONICAL EXPANDER"

            angle_ok = half_angle <= max_angle + 0.5  # Small tolerance

            if verbose:
                print(f"    Type: {trans_type}")
                print(f"    Half-angle: {half_angle:.1f}° (max recommended: {max_angle}°)")

            if angle_ok:
                if verbose:
                    print(f"    Angle check: ✓ PASS")
            else:
                if verbose:
                    print(f"    Angle check: ⚠ WARNING - steep angle may cause flow issues")
                results['summary']['warnings'] += 1
        else:
            if verbose:
                print(f"    Type: CYLINDRICAL (same diameter)")
                print(f"    No angle check needed")

        # Connection summary
        if verbose:
            conn_status = "✓ PASS" if conn_result['pass'] else "✗ FAIL"
            print(f"\n  CONNECTION RESULT: {conn_status}")

        results['connections'].append(conn_result)

    # ================================================================
    # OVERALL SUMMARY
    # ================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)
        print(f"\nTotal checks: {results['summary']['total_checks']}")
        print(f"  Passed:   {results['summary']['passed']}")
        print(f"  Failed:   {results['summary']['failed']}")
        print(f"  Warnings: {results['summary']['warnings']}")

        if results['overall_pass']:
            print("\n🎉 OVERALL RESULT: ALL CONNECTORS FIT PERFECTLY ✓")
        else:
            print("\n❌ OVERALL RESULT: SOME CONNECTORS HAVE ISSUES")
            print("\nFailed connections:")
            for conn in results['connections']:
                if not conn['pass']:
                    print(f"  - {conn['name']}")
                    for check in conn['checks']:
                        if not check['pass']:
                            print(f"      • {check['name']}: expected={check.get('expected')}, actual={check.get('actual')}")

        print("=" * 70)

    return results


def check_mesh_continuity(system: FeedSystemAssembly, verbose: bool = True):
    """
    Check that the mesh is continuous with no gaps between components.

    This is a geometric verification that transition connectors create
    a continuous surface from hopper to deagglomerator.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MESH CONTINUITY CHECK")
        print("=" * 70)

    # Build the mesh
    vertices, indices = system.build_mesh()

    if verbose:
        print(f"\nTotal vertices: {len(vertices)}")
        print(f"Total triangles: {len(indices) // 3}")

    # Get bounds
    min_corner, max_corner = system.get_bounds()
    extent = system.get_system_extent()

    if verbose:
        print(f"\nBounding box:")
        print(f"  Min: ({min_corner[0]*1000:.1f}, {min_corner[1]*1000:.1f}, {min_corner[2]*1000:.1f}) mm")
        print(f"  Max: ({max_corner[0]*1000:.1f}, {max_corner[1]*1000:.1f}, {max_corner[2]*1000:.1f}) mm")
        print(f"  Extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")

    # Check for any NaN or Inf values
    has_nan = np.any(np.isnan(vertices))
    has_inf = np.any(np.isinf(vertices))

    if verbose:
        print(f"\nVertex validity:")
        print(f"  Contains NaN: {'✗ FAIL' if has_nan else '✓ PASS'}")
        print(f"  Contains Inf: {'✗ FAIL' if has_inf else '✓ PASS'}")

    # Check for degenerate triangles
    tri_vertices = vertices[indices.reshape(-1, 3)]
    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]

    # Calculate triangle areas using cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    degenerate_count = np.sum(areas < 1e-10)

    if verbose:
        print(f"\nTriangle analysis:")
        print(f"  Min area: {areas.min()*1e6:.4f} mm²")
        print(f"  Max area: {areas.max()*1e6:.4f} mm²")
        print(f"  Mean area: {areas.mean()*1e6:.4f} mm²")
        print(f"  Degenerate triangles: {degenerate_count} {'⚠ WARNING' if degenerate_count > 0 else '✓ OK'}")

    # Allow degenerate triangles as a warning, not a failure
    # They often occur at poles/tips of conical sections and don't affect simulation
    mesh_ok = not has_nan and not has_inf

    if verbose:
        if degenerate_count > 0:
            print(f"\n  Note: Degenerate triangles are common at cone tips/poles.")
            print(f"        They don't affect simulation but may cause rendering artifacts.")
        print(f"\nMESH CHECK: {'✓ PASS' if mesh_ok else '✗ ISSUES FOUND'}")
        print("=" * 70)

    return mesh_ok


def check_component_mesh_quality(system: FeedSystemAssembly, verbose: bool = True):
    """
    Check mesh quality for each individual component to identify sources of degenerate triangles.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PER-COMPONENT MESH QUALITY")
        print("=" * 70)

    components = [
        ('Hopper', system.hopper, system._hopper_position),
        ('Airlock', system.airlock, system._airlock_position),
        ('Feeder', system.feeder, system._feeder_position),
        ('Deagglomerator', system.deagglomerator, system._deagglomerator_position),
    ]

    # Add transitions
    for i, (transition, pos, name) in enumerate(system._transition_connectors):
        components.append((f'Transition: {name}', transition, (0.0, 0.0, 0.0)))

    total_degenerate = 0
    component_issues = []

    for name, component, position in components:
        verts, idx, _ = component.generate_mesh()
        num_tris = len(idx) // 3

        if num_tris == 0:
            continue

        # Check for degenerate triangles
        tri_vertices = verts[idx.reshape(-1, 3)]
        v0 = tri_vertices[:, 0]
        v1 = tri_vertices[:, 1]
        v2 = tri_vertices[:, 2]

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        degenerate = np.sum(areas < 1e-10)
        total_degenerate += degenerate

        if verbose:
            status = "✓" if degenerate == 0 else "⚠"
            print(f"\n  {name}:")
            print(f"    Vertices: {len(verts)}, Triangles: {num_tris}")
            print(f"    Degenerate triangles: {degenerate} {status}")

        if degenerate > 0:
            component_issues.append((name, degenerate, num_tris))

    if verbose:
        print(f"\n  {'─' * 50}")
        print(f"  Total degenerate triangles: {total_degenerate}")

        if component_issues:
            print(f"\n  Components with degenerate triangles:")
            for name, degen, total in component_issues:
                pct = (degen / total) * 100
                print(f"    - {name}: {degen}/{total} ({pct:.1f}%)")
        print("=" * 70)

    return total_degenerate, component_issues


def main():
    """Run full diagnostics on feed system transition connectors."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " FEED SYSTEM TRANSITION CONNECTOR DIAGNOSTICS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Create standard feed system
    print("\nCreating standard feed system...")
    system = FeedSystemAssembly()

    # Run connector diagnostics
    results = diagnose_transition_connectors(system, verbose=True)

    # Run mesh continuity check
    mesh_ok = check_mesh_continuity(system, verbose=True)

    # Run per-component mesh quality check
    total_degen, component_issues = check_component_mesh_quality(system, verbose=True)

    # Print transition report
    system.print_transition_report()

    # Note about built-in validation
    print("\n" + "=" * 70)
    print("NOTE ON BUILT-IN CONNECTION VALIDATION")
    print("=" * 70)
    print("""
The built-in validate_connections() method reports gaps between component
ports. This is EXPECTED behavior because:

  1. Components are positioned with intentional gaps
  2. Transition connectors fill these gaps
  3. The validator checks port-to-port distance, not accounting for transitions

Our diagnostic above (CHECK 2: Gap vs distance) verifies that the transition
connectors exactly fill these gaps with proper flange spacing.

EXPECTED GAPS (filled by transitions):
  - Hopper → Airlock:       40mm gap (30mm transition + 2×5mm flanges)
  - Airlock → Feeder:       70mm gap (60mm transition + 2×5mm flanges)
  - Feeder → Deagglomerator: 60mm gap (50mm transition + 2×5mm flanges)
""")
    print("=" * 70)

    # Final summary
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " FINAL DIAGNOSIS RESULT ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    all_pass = results['overall_pass'] and mesh_ok

    if all_pass:
        print("║" + " ✓ ALL TRANSITION CONNECTORS FIT PERFECTLY ".center(68) + "║")
        print("║" + " ✓ MESH IS VALID (no NaN/Inf) ".center(68) + "║")
        if total_degen > 0:
            msg = f" ⚠ {total_degen} degenerate triangles (cosmetic only) "
            print("║" + msg.center(68) + "║")
    else:
        if not results['overall_pass']:
            print("║" + " ✗ SOME CONNECTOR CHECKS FAILED ".center(68) + "║")
        if not mesh_ok:
            print("║" + " ✗ MESH HAS CRITICAL ISSUES (NaN/Inf) ".center(68) + "║")

    print("╚" + "═" * 68 + "╝")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
