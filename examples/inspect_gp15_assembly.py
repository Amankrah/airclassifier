#!/usr/bin/env python
"""
GP-15 Assembly Inspection Tool
================================

Inspects the positional fitting of all GP-15 pretreatment machine
components to verify proper alignment, opening dimensions, and
physical connections.

Checks:
  1. Component bounding boxes and positions
  2. Infeed tunnel → oven infeed wall flush alignment
  3. Outfeed tunnel → oven outfeed wall flush alignment
  4. EMU air duct → oven back wall connection
  5. EMU extraction duct → oven ceiling connection
  6. Generator RF conduit → oven back wall penetration
  7. RF conduit → generator cabinet face connection
  8. Opening height matching (tunnels vs oven openings)
  9. Electrode containment within oven/RF zone
  10. Hopper position relative to bed start
  11. Height proportionality (floor → deck → oven ceiling → EMU top)

Usage:
    python examples/inspect_gp15_assembly.py
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from airclassifier.pretreatment.geometry.assembly import (
    create_gp15_machine,
    COMPONENT_COLORS,
)


def _label(name: str) -> str:
    return COMPONENT_COLORS.get(name, {}).get("label", name)


def check_flush(name_a, val_a, name_b, val_b, tol=0.02):
    """Check two values are flush (within tolerance)."""
    diff = abs(val_a - val_b)
    ok = diff <= tol
    status = "OK" if ok else "FAIL"
    print(f"    {name_a} = {val_a:.4f}  vs  {name_b} = {val_b:.4f}"
          f"  diff={diff*1000:.1f} mm  [{status}]")
    return ok


def check_contains(outer_name, outer_min, outer_max, inner_name, inner_min, inner_max, axis):
    """Check inner range is fully inside outer range."""
    ok = inner_min >= outer_min - 0.005 and inner_max <= outer_max + 0.005
    status = "OK" if ok else "FAIL"
    margin_lo = inner_min - outer_min
    margin_hi = outer_max - inner_max
    print(f"    {inner_name} {axis}=[{inner_min:.3f}..{inner_max:.3f}]"
          f"  inside {outer_name} {axis}=[{outer_min:.3f}..{outer_max:.3f}]"
          f"  margins=({margin_lo*1000:.1f}, {margin_hi*1000:.1f}) mm  [{status}]")
    return ok


def check_reach(name_a, val_a, name_b, val_b, direction=">="):
    """Check that A reaches B (A >= B or A <= B)."""
    if direction == ">=":
        ok = val_a >= val_b - 0.02
    else:
        ok = val_a <= val_b + 0.02
    status = "OK" if ok else "FAIL"
    overlap = val_a - val_b if direction == ">=" else val_b - val_a
    print(f"    {name_a} = {val_a:.3f}  {direction}  {name_b} = {val_b:.3f}"
          f"  overlap={overlap*1000:.1f} mm  [{status}]")
    return ok


def main():
    print("=" * 75)
    print("GP-15 PRETREATMENT MACHINE — ASSEMBLY INSPECTION")
    print("=" * 75)

    machine = create_gp15_machine()
    meshes = machine.generate_all_meshes()
    p = machine.params
    cp = p.conveyor_params
    op = p.oven_params
    assert op is not None

    # Collect bounding boxes
    bounds = {}
    for name, (v, t, meta) in meshes.items():
        bounds[name] = {
            "x_min": float(v[:, 0].min()), "x_max": float(v[:, 0].max()),
            "y_min": float(v[:, 1].min()), "y_max": float(v[:, 1].max()),
            "z_min": float(v[:, 2].min()), "z_max": float(v[:, 2].max()),
            "verts": v.shape[0], "tris": t.shape[0],
        }

    # Key reference levels
    bed_start = cp.nose_length_m
    bed_end = cp.frame_length_m - cp.nose_length_m
    bed_center = (bed_start + bed_end) / 2
    floor_y = -(cp.frame_height_m + cp.leg_height_m)
    deck_y = 0.0
    oven_ceil_y = op.oven_height_m
    oven_z_back = op.oven_width_m

    n_pass = 0
    n_fail = 0

    def tally(ok):
        nonlocal n_pass, n_fail
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    # ────────────────────────────────────────────────────────────
    # 1. COMPONENT POSITIONS
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("1. COMPONENT BOUNDING BOXES")
    print("─" * 75)
    print(f"{'Component':30s}  {'X range':18s}  {'Y range':18s}  {'Z range':18s}  V      T")
    for name, b in bounds.items():
        label = _label(name)
        print(f"  {label:28s}  {b['x_min']:6.3f}..{b['x_max']:6.3f}"
              f"  {b['y_min']:6.3f}..{b['y_max']:6.3f}"
              f"  {b['z_min']:6.3f}..{b['z_max']:6.3f}"
              f"  {b['verts']:>5}  {b['tris']:>5}")
    print(f"\n  Reference levels:")
    print(f"    Floor:        Y = {floor_y:.3f}")
    print(f"    Deck/belt:    Y = {deck_y:.3f}")
    print(f"    Oven ceiling: Y = {oven_ceil_y:.3f}")
    print(f"    Oven back:    Z = {oven_z_back:.3f}")
    print(f"    Bed:          X = {bed_start:.3f} .. {bed_end:.3f}"
          f"  ({(bed_end - bed_start) * 100:.1f} cm)")
    print(f"    Bed centre:   X = {bed_center:.3f}")

    # ────────────────────────────────────────────────────────────
    # 2. OVEN CENTRED ON BED
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("2. OVEN CENTRED ON BED")
    print("─" * 75)
    ov = bounds["oven_chamber"]
    oven_center = (ov["x_min"] + ov["x_max"]) / 2
    ok = abs(oven_center - bed_center) < 0.02
    print(f"    Oven centre X = {oven_center:.3f}  Bed centre X = {bed_center:.3f}"
          f"  diff = {abs(oven_center - bed_center) * 1000:.1f} mm"
          f"  [{'OK' if ok else 'FAIL'}]")
    tally(ok)
    infeed_run = ov["x_min"] - bed_start
    outfeed_run = bed_end - ov["x_max"]
    ok2 = abs(infeed_run - outfeed_run) < 0.05
    print(f"    Infeed run:  {infeed_run * 100:.1f} cm    Outfeed run: {outfeed_run * 100:.1f} cm"
          f"  balanced={ok2}")
    tally(ok2)

    # ────────────────────────────────────────────────────────────
    # 3. INFEED TUNNEL → OVEN INFEED WALL
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("3. INFEED TUNNEL → OVEN INFEED WALL (flush at X)")
    print("─" * 75)
    if "infeed_tunnel" in bounds:
        it = bounds["infeed_tunnel"]
        ok = check_flush("Tunnel X_max", it["x_max"],
                         "Oven X_min", ov["x_min"])
        tally(ok)
        # Height match
        tunnel_h = it["y_max"] - max(it["y_min"], 0)
        oven_open_h = op.opening_height_m
        ok2 = abs(tunnel_h - oven_open_h) < 0.02
        print(f"    Tunnel height = {tunnel_h * 1000:.1f} mm  vs"
              f"  Oven opening = {oven_open_h * 1000:.1f} mm"
              f"  [{'OK' if ok2 else 'FAIL'}]")
        tally(ok2)

    # ────────────────────────────────────────────────────────────
    # 4. OUTFEED TUNNEL → OVEN OUTFEED WALL
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("4. OUTFEED TUNNEL → OVEN OUTFEED WALL (flush at X)")
    print("─" * 75)
    if "outfeed_tunnel" in bounds:
        ot = bounds["outfeed_tunnel"]
        ok = check_flush("Tunnel X_min", ot["x_min"],
                         "Oven X_max", ov["x_max"])
        tally(ok)
        tunnel_h = ot["y_max"] - max(ot["y_min"], 0)
        ok2 = abs(tunnel_h - oven_open_h) < 0.02
        print(f"    Tunnel height = {tunnel_h * 1000:.1f} mm  vs"
              f"  Oven opening = {oven_open_h * 1000:.1f} mm"
              f"  [{'OK' if ok2 else 'FAIL'}]")
        tally(ok2)

    # ────────────────────────────────────────────────────────────
    # 5. EMU → OVEN BACK WALL (air duct connection)
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("5. EMU → OVEN BACK WALL (air supply + extraction duct)")
    print("─" * 75)
    if "emu_housing" in bounds:
        em = bounds["emu_housing"]
        ok = check_reach("EMU Z_min", em["z_min"],
                         "Oven Z_back", oven_z_back, "<=")
        tally(ok)
        gap = em["z_min"] - oven_z_back
        if gap > 0.05:
            print(f"    ⚠  Air duct spans {gap * 100:.1f} cm gap"
                  f" (duct mesh extends from EMU to oven)")
        # EMU top above oven ceiling (for extraction fan)
        ok2 = check_reach("EMU Y_max", em["y_max"],
                          "Oven ceiling", oven_ceil_y, ">=")
        tally(ok2)
        clearance = em["y_max"] - oven_ceil_y
        print(f"    Extraction fan clearance above oven: {clearance * 100:.1f} cm")

    # ────────────────────────────────────────────────────────────
    # 6. RF FEED → OVEN BACK WALL (copper conduit penetration)
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("6. RF FEED CONDUIT → OVEN BACK WALL (penetration)")
    print("─" * 75)
    if "rf_feed" in bounds:
        rf = bounds["rf_feed"]
        ok = check_reach("RF Z_max", rf["z_max"],
                         "Oven Z_back", oven_z_back, ">=")
        tally(ok)
        # Must also reach inside oven
        ok2 = check_reach("RF Z_min", rf["z_min"],
                          "Oven Z_back", oven_z_back, "<=")
        tally(ok2)

    # ────────────────────────────────────────────────────────────
    # 7. RF FEED → GENERATOR CABINET (physical contact)
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("7. RF FEED CONDUIT → GENERATOR (physical contact)")
    print("─" * 75)
    if "rf_feed" in bounds and "generator" in bounds:
        rf = bounds["rf_feed"]
        gen = bounds["generator"]
        ok = check_reach("RF Z_max", rf["z_max"],
                         "Generator Z_min", gen["z_min"], ">=")
        tally(ok)

    # ────────────────────────────────────────────────────────────
    # 8. ELECTRODES INSIDE OVEN & RF ZONE
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("8. ELECTRODES CONTAINED WITHIN OVEN / RF ZONE")
    print("─" * 75)
    for elec_name in ["upper_electrode", "lower_electrode"]:
        if elec_name in bounds:
            el = bounds[elec_name]
            label = _label(elec_name)
            print(f"\n  {label}:")
            ok = check_contains("Oven", ov["x_min"], ov["x_max"],
                                label, el["x_min"], el["x_max"], "X")
            tally(ok)
            ok2 = check_contains("Oven", ov["z_min"], ov["z_max"],
                                 label, el["z_min"], el["z_max"], "Z")
            tally(ok2)
            ok3 = check_contains("Oven", ov["y_min"], ov["y_max"],
                                 label, el["y_min"], el["y_max"], "Y")
            tally(ok3)

    # ────────────────────────────────────────────────────────────
    # 9. HOPPER POSITION RELATIVE TO BED
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("9. HOPPER POSITION RELATIVE TO BED START")
    print("─" * 75)
    if "infeed_hopper" in bounds:
        hp = bounds["infeed_hopper"]
        hopper_front = hp["x_max"]
        offset_from_bed = hopper_front - bed_start
        print(f"    Hopper front X = {hopper_front:.3f}"
              f"  Bed start X = {bed_start:.3f}")
        print(f"    Offset from bed start: {offset_from_bed * 100:.1f} cm"
              f"  (target ~15 cm)")
        ok = 10 <= offset_from_bed * 100 <= 25
        print(f"    [{('OK' if ok else 'WARN — check hopper position')}]")
        tally(ok)

    # ────────────────────────────────────────────────────────────
    # 10. HEIGHT PROPORTIONALITY
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("10. HEIGHT PROPORTIONALITY (floor → deck → ceiling)")
    print("─" * 75)

    print(f"\n    Floor-to-deck:   {abs(floor_y) * 100:.1f} cm"
          f"  (frame {cp.frame_height_m * 100:.0f} + legs {cp.leg_height_m * 100:.0f})")
    print(f"    Deck-to-ceiling: {oven_ceil_y * 100:.1f} cm  (oven height)")
    print(f"    Total floor-to-ceiling: {(oven_ceil_y - floor_y) * 100:.1f} cm")

    if "generator" in bounds:
        gen = bounds["generator"]
        gen_top_vs_ceil = gen["y_max"] - oven_ceil_y
        ok = abs(gen_top_vs_ceil) < 0.05
        print(f"\n    Generator top Y = {gen['y_max']:.3f}"
              f"  vs oven ceiling Y = {oven_ceil_y:.3f}"
              f"  diff = {gen_top_vs_ceil * 1000:.0f} mm"
              f"  [{'OK — flush' if ok else 'WARN'}]")
        tally(ok)

    if "emu_housing" in bounds:
        em = bounds["emu_housing"]
        emu_above = em["y_max"] - oven_ceil_y
        ok = emu_above > 0.3
        print(f"    EMU top Y = {em['y_max']:.3f}"
              f"  (above ceiling by {emu_above * 100:.1f} cm"
              f"  for extraction fan)  [{'OK' if ok else 'WARN — too short'}]")
        tally(ok)

    # ────────────────────────────────────────────────────────────
    # 11. BELT WITHIN FRAME
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("11. BELT CONTAINED WITHIN CONVEYOR FRAME")
    print("─" * 75)
    if "belt" in bounds:
        bl = bounds["belt"]
        fr = bounds["conveyor_frame"]
        ok = check_contains("Frame", fr["z_min"], fr["z_max"],
                            "Belt", bl["z_min"], bl["z_max"], "Z")
        tally(ok)

    # ────────────────────────────────────────────────────────────
    # 12. PORT-TO-PORT CONNECTIONS (OVEN ↔ EMU)
    # ────────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print("12. PORT-TO-PORT CONNECTIONS (OVEN <-> EMU)")
    print("─" * 75)

    oven_ports = machine.oven.ports
    emu_ports = machine.emu.ports

    print("\n  OVEN ports:")
    for pname, port in oven_ports.items():
        pos = port.position
        d = port.direction
        dim = (f"D={port.diameter*1000:.0f}mm" if port.port_type.value == "circular"
               else f"W={port.width*1000:.0f}mm H={port.height*1000:.0f}mm")
        print(f"    {pname:16s}  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
              f"  dir=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f})  {dim}")

    print("\n  EMU ports:")
    for pname, port in emu_ports.items():
        pos = port.position
        d = port.direction
        dim = (f"D={port.diameter*1000:.0f}mm" if port.port_type.value == "circular"
               else f"W={port.width*1000:.0f}mm H={port.height*1000:.0f}mm")
        print(f"    {pname:16s}  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
              f"  dir=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f})  {dim}")

    # Check air_supply ↔ air_to_oven
    print("\n  --- Air Supply Connection ---")
    oven_as = oven_ports['air_supply']
    emu_ao = emu_ports['air_to_oven']
    # X and Y should match at the oven back wall (Z = oven_width)
    dx = abs(oven_as.position[0] - emu_ao.position[0])
    dy = abs(oven_as.position[1] - emu_ao.position[1])
    dz = abs(oven_as.position[2] - emu_ao.position[2])
    ok = dx < 0.05 and dy < 0.05 and dz < 0.05
    print(f"    Oven air_supply  pos=({oven_as.position[0]:.3f}, {oven_as.position[1]:.3f}, {oven_as.position[2]:.3f})")
    print(f"    EMU  air_to_oven pos=({emu_ao.position[0]:.3f}, {emu_ao.position[1]:.3f}, {emu_ao.position[2]:.3f})")
    print(f"    Offset: dX={dx*1000:.1f}mm  dY={dy*1000:.1f}mm  dZ={dz*1000:.1f}mm  [{'OK' if ok else 'FAIL'}]")
    tally(ok)

    # Check directions are opposing
    dot = sum(a * b for a, b in zip(oven_as.direction, emu_ao.direction))
    ok2 = dot < -0.9  # should be anti-parallel
    print(f"    Directions: oven={oven_as.direction}  emu={emu_ao.direction}"
          f"  dot={dot:.2f}  [{'OK opposing' if ok2 else 'FAIL — not opposing'}]")
    tally(ok2)

    # Check extraction ↔ extraction_inlet
    print("\n  --- Extraction Connection ---")
    oven_ex = oven_ports['extraction']
    emu_ei = emu_ports['extraction_inlet']
    dx = abs(oven_ex.position[0] - emu_ei.position[0])
    dy = abs(oven_ex.position[1] - emu_ei.position[1])
    # Z can differ (duct spans the gap), but X and Y should match
    ok = dx < 0.05 and dy < 0.05
    print(f"    Oven extraction       pos=({oven_ex.position[0]:.3f}, {oven_ex.position[1]:.3f}, {oven_ex.position[2]:.3f})")
    print(f"    EMU  extraction_inlet pos=({emu_ei.position[0]:.3f}, {emu_ei.position[1]:.3f}, {emu_ei.position[2]:.3f})")
    print(f"    Offset: dX={dx*1000:.1f}mm  dY={dy*1000:.1f}mm  (Z spans gap via duct)")
    print(f"    [{'OK' if ok else 'FAIL'}]")
    tally(ok)

    # Check diameter match
    oven_d = oven_ex.diameter if hasattr(oven_ex, 'diameter') and oven_ex.diameter else 0
    emu_d = emu_ei.diameter if hasattr(emu_ei, 'diameter') and emu_ei.diameter else 0
    if oven_d > 0 and emu_d > 0:
        ok = abs(oven_d - emu_d) < 0.05
        print(f"    Oven extraction D={oven_d*1000:.0f}mm  EMU inlet D={emu_d*1000:.0f}mm"
              f"  [{'OK' if ok else 'FAIL — size mismatch'}]")
        tally(ok)

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    total = n_pass + n_fail
    print("\n" + "=" * 75)
    print(f"INSPECTION SUMMARY:  {n_pass}/{total} checks passed"
          f"  ({n_fail} failed)")
    print("=" * 75)

    if n_fail > 0:
        print("\n⚠  Some checks failed — review the FAIL items above.")
    else:
        print("\n✓  All assembly connections verified.")

    return n_fail == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
