#!/usr/bin/env python
"""
Mill Assembly Flow Inspection

Inspects the material flow path through the hammer mill assembly with a
connection-by-connection outlet-to-inlet investigation (similar to
examples/inspect_assembly.py for the classifier/air systems).

This script:
1. Builds the hammer mill assembly from MillConfig
2. Traces the flow path: external → feed chute inlet → feed chute outlet
   → housing feed inlet → [internal milling] → housing discharge outlet → external
3. For each connection (outlet → inlet), reports:
   - Port positions (world coordinates)
   - Port dimensions where available (width × depth or equivalent)
   - Alignment check (distance between outlet and next inlet)
4. Summarizes dimension matching at each connection

Usage:
    python examples/inspect_mill_flow.py              # Full flow inspection
    python examples/inspect_mill_flow.py --flow       # Flow path only
    python examples/inspect_mill_flow.py --dims       # Dimension verification only
    python examples/inspect_mill_flow.py --coords     # Coordinate alignment only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from airclassifier.milling.geometry.assembly.machine import create_hammer_mill_assembly


def _distance_mm(p: tuple, q: tuple) -> float:
    """Euclidean distance between two (x,y,z) points in metres, returned in mm."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))) * 1000.0


def inspect_mill_flow_path():
    """Connection-by-connection outlet-to-inlet flow inspection for the mill."""
    assembly = create_hammer_mill_assembly()
    fc = assembly.feed_chute_geometry
    hp = assembly.housing_geometry
    fp = fc.params
    hpp = hp.params

    # All geometry is in the same world frame (housing center = origin)
    fc_ports = fc.ports
    housing_ports = hp.ports

    print("\n" + "=" * 70)
    print("HAMMER MILL FLOW PATH — CONNECTION BY CONNECTION")
    print("=" * 70)
    print("\nFlow: External → Feed chute inlet → Feed chute outlet → Housing feed inlet")
    print("      → [Internal milling] → Housing discharge outlet → External")
    print()

    tolerance_mm = 5.0  # X/Z alignment tolerance
    flange_y_tolerance_mm = 35.0  # Y offset allowed (housing feed_inlet above chute outlet by design)
    verification = []

    # --- 1. External / upstream → Feed chute INLET ---
    inlet_pos = fc_ports["inlet"]
    print("--- 1. External / upstream → Feed chute INLET ---")
    print(f"  Feed chute inlet:  X={inlet_pos[0]*1000:8.1f}, Y={inlet_pos[1]*1000:8.1f}, Z={inlet_pos[2]*1000:8.1f} mm")
    print(f"  Inlet dimensions:  W={fp.inlet_width_m*1000:.1f} mm (X), D={fp.inlet_depth_m*1000:.1f} mm (Z)")
    print("  (Upstream: pretreatment outfeed or hopper connects here)")
    print()

    # --- 2. Feed chute OUTLET → Housing FEED INLET ---
    chute_outlet_pos = fc_ports["outlet"]
    housing_feed_in_pos = housing_ports["feed_inlet"]
    dist_mm = _distance_mm(chute_outlet_pos, housing_feed_in_pos)
    dx_mm = abs(chute_outlet_pos[0] - housing_feed_in_pos[0]) * 1000
    dy_mm = abs(chute_outlet_pos[1] - housing_feed_in_pos[1]) * 1000
    dz_mm = abs(chute_outlet_pos[2] - housing_feed_in_pos[2]) * 1000
    # X/Z must align; Y may have small offset (housing feed_inlet above chute outlet for flange)
    match = (dx_mm <= tolerance_mm and dz_mm <= tolerance_mm and dy_mm <= flange_y_tolerance_mm)
    verification.append(("Feed chute outlet → Housing feed inlet", match, dist_mm))

    print("--- 2. Feed chute OUTLET → Housing FEED INLET ---")
    print(f"  Feed chute outlet: X={chute_outlet_pos[0]*1000:8.1f}, Y={chute_outlet_pos[1]*1000:8.1f}, Z={chute_outlet_pos[2]*1000:8.1f} mm")
    print(f"  Outlet dimensions: W={fp.outlet_width_m*1000:.1f} mm (X), D={fp.outlet_depth_m*1000:.1f} mm (Z)")
    print(f"  Housing feed inlet: X={housing_feed_in_pos[0]*1000:8.1f}, Y={housing_feed_in_pos[1]*1000:8.1f}, Z={housing_feed_in_pos[2]*1000:8.1f} mm")
    print(f"  Feed opening:      W={hpp.feed_opening_width_m*1000:.1f} mm (X), D={hpp.feed_opening_depth_m*1000:.1f} mm (Z)")
    print(f"  Distance (outlet→inlet): {dist_mm:.2f} mm  {'[OK]' if match else '[CHECK]'}")
    if dy_mm > tolerance_mm and dy_mm <= flange_y_tolerance_mm:
        print("  (Note: Housing feed_inlet is slightly above chute outlet by design for flange; Y offset accepted.)")
    print()

    # --- 3. Internal (housing feed inlet → milling chamber → discharge) ---
    print("--- 3. Internal flow (housing) ---")
    print("  Material enters at feed opening, passes through rotor/screen, exits at discharge.")
    print()

    # --- 4. Housing DISCHARGE OUTLET → External / downstream ---
    discharge_pos = housing_ports["discharge_outlet"]
    print("--- 4. Housing DISCHARGE OUTLET → External / downstream ---")
    print(f"  Discharge outlet:  X={discharge_pos[0]*1000:8.1f}, Y={discharge_pos[1]*1000:8.1f}, Z={discharge_pos[2]*1000:8.1f} mm")
    print(f"  Discharge opening: W={hpp.discharge_opening_width_m*1000:.1f} mm (X), D={hpp.discharge_opening_depth_m*1000:.1f} mm (Z)")
    print("  (Downstream: classifier inlet or conveyor connects here)")
    print()

    # Summary
    print("=" * 70)
    print("FLOW CONNECTION SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in verification if ok)
    total = len(verification)
    print(f"\nPosition alignment (X/Z ≤{tolerance_mm} mm, Y ≤{flange_y_tolerance_mm} mm): {passed}/{total} connections OK")
    for name, ok, dist in verification:
        print(f"  {'[OK]' if ok else '[CHECK]'} {name}: distance = {dist:.2f} mm")
    print()
    return assembly


def verify_mill_flow_dimensions():
    """Verify dimension matching at each flow connection."""
    assembly = create_hammer_mill_assembly()
    fc = assembly.feed_chute_geometry
    hp = assembly.housing_geometry
    fp = fc.params
    hpp = hp.params

    print("\n" + "=" * 70)
    print("MILL FLOW PATH — DIMENSION VERIFICATION")
    print("=" * 70)

    results = []
    tol_mm = 2.0

    # Feed chute outlet vs housing feed opening
    print("\n--- Feed chute outlet ↔ Housing feed inlet ---")
    w_chute = fp.outlet_width_m * 1000
    d_chute = fp.outlet_depth_m * 1000
    w_feed = hpp.feed_opening_width_m * 1000
    d_feed = hpp.feed_opening_depth_m * 1000
    match_w = abs(w_chute - w_feed) <= tol_mm
    match_d = abs(d_chute - d_feed) <= tol_mm
    match = match_w and match_d
    results.append(("Chute outlet ↔ Feed opening", match))

    print(f"  Chute outlet:     W={w_chute:.1f} mm, D={d_chute:.1f} mm")
    print(f"  Housing opening:  W={w_feed:.1f} mm, D={d_feed:.1f} mm")
    print(f"  Width match:  {'[OK]' if match_w else '[FAIL]'} (diff: {abs(w_chute - w_feed):.1f} mm)")
    print(f"  Depth match:  {'[OK]' if match_d else '[FAIL]'} (diff: {abs(d_chute - d_feed):.1f} mm)")

    # Discharge (informational; downstream must match)
    print("\n--- Housing discharge outlet (for downstream connection) ---")
    print(f"  Discharge opening: W={hpp.discharge_opening_width_m*1000:.1f} mm, D={hpp.discharge_opening_depth_m*1000:.1f} mm")

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in results if ok)
    print(f"Dimension checks: {passed}/{len(results)} passed")
    print("=" * 70 + "\n")
    return results


def verify_mill_flow_coordinates():
    """Verify coordinate alignment at the critical outlet→inlet connection."""
    assembly = create_hammer_mill_assembly()
    fc = assembly.feed_chute_geometry
    hp = assembly.housing_geometry
    fc_ports = fc.ports
    housing_ports = hp.ports

    print("\n" + "=" * 70)
    print("MILL FLOW PATH — COORDINATE ALIGNMENT")
    print("=" * 70)

    tolerance_mm = 5.0   # X/Z
    flange_y_mm = 35.0   # Y (housing feed_inlet above chute outlet by design)
    chute_out = fc_ports["outlet"]
    feed_in = housing_ports["feed_inlet"]
    dx = abs(chute_out[0] - feed_in[0]) * 1000
    dy = abs(chute_out[1] - feed_in[1]) * 1000
    dz = abs(chute_out[2] - feed_in[2]) * 1000
    match = (dx <= tolerance_mm and dz <= tolerance_mm and dy <= flange_y_mm)

    print("\n--- Feed chute outlet → Housing feed inlet ---")
    print(f"  Chute outlet:     X={chute_out[0]*1000:8.1f}, Y={chute_out[1]*1000:8.1f}, Z={chute_out[2]*1000:8.1f} mm")
    print(f"  Housing feed in:  X={feed_in[0]*1000:8.1f}, Y={feed_in[1]*1000:8.1f}, Z={feed_in[2]*1000:8.1f} mm")
    print(f"  Difference:       dX={dx:.1f}, dY={dy:.1f}, dZ={dz:.1f} mm")
    print(f"  Tolerance:        X/Z ≤{tolerance_mm} mm, Y ≤{flange_y_mm} mm (flange)")
    print(f"  Aligned: {'[OK]' if match else '[CHECK]'}")

    print("\n" + "=" * 70 + "\n")
    return match


def print_component_ports_summary():
    """Print a short summary of all component ports in the mill assembly."""
    assembly = create_hammer_mill_assembly()
    fc = assembly.feed_chute_geometry
    hp = assembly.housing_geometry

    print("\n" + "=" * 70)
    print("MILL ASSEMBLY — COMPONENT PORTS SUMMARY")
    print("=" * 70)

    print("\n--- Feed chute ---")
    for name, pos in fc.ports.items():
        print(f"  {name:12s}: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")

    print("\n--- Housing ---")
    for name, pos in hp.ports.items():
        print(f"  {name:16s}: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")

    print("\n--- Assembly (pipeline integration) ---")
    ports = assembly.ports
    for name, pos in ports.items():
        print(f"  {name:14s}: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect hammer mill assembly flow path (outlet-to-inlet connections)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--flow", "-f", action="store_true",
                        help="Flow path connection-by-connection inspection (default if no other option)")
    parser.add_argument("--dims", action="store_true",
                        help="Verify dimension matching at connections")
    parser.add_argument("--coords", action="store_true",
                        help="Verify coordinate alignment at connections")
    parser.add_argument("--ports", action="store_true",
                        help="Print component ports summary only")
    args = parser.parse_args()

    any_flag = args.flow or args.dims or args.coords or args.ports
    if not any_flag:
        args.flow = True

    if args.ports:
        print_component_ports_summary()
    if args.flow:
        inspect_mill_flow_path()
    if args.dims:
        verify_mill_flow_dimensions()
    if args.coords:
        verify_mill_flow_coordinates()


if __name__ == "__main__":
    main()
