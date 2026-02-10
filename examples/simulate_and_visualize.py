#!/usr/bin/env python
"""
GP-15 Simulation + Visualization
=================================

Runs a complete RF dielectric heating simulation on the GP-15 machine
assembly and visualizes both the 3D machine geometry and the simulation
results (temperature and moisture fields).

This demonstrates the full pretreatment pipeline integration described
in the engineering guide:

    1. Build GP-15 machine assembly (geometry, §2–3)
    2. Configure physics simulation (GP15Simulator, §7.1)
    3. Load a processing recipe (§7.4)
    4. Run coupled multi-physics simulation (§6.2, 9-step loop)
    5. Visualize machine + field overlays (§9.3)
    6. Display outfeed cross-section (pipeline output to milling, §9.1)

Architecture::

    GP15Simulator
        ├── GP15MachineAssembly    ← 3D geometry (single source of truth)
        │     ├── ConveyorBelt     ← Frame, rollers, belt loop
        │     ├── OvenChamber      ← Sheet-metal enclosure
        │     ├── Electrodes       ← Upper (movable) + lower (fixed)
        │     ├── InfeedHopper     ← Gravity-fed material entry
        │     └── ...              ← EMU, generator, tunnels, bin
        │
        └── CoupledSimulator       ← Multi-physics engine
              ├── 1. ADVECT        ← Belt transport (TVD advection)
              ├── 2. RF FIELD      ← Laplace solve for |E|²
              ├── 3. HEATING       ← P_v = 2πfε₀ε″|E|²
              ├── 4. EVAPORATION   ← Temperature-driven drying
              ├── 5. THERMAL       ← Heat equation (FDM)
              ├── 6. MOISTURE      ← Diffusion + evaporation
              ├── 7. PROPERTIES    ← Nonlinear ε′, ε″, ρ·cₚ, k
              ├── 8. CONTROLLER    ← PLC logic (MRH/MRL, gap, temp)
              └── 9. RECORD        ← KPIs and outfeed state

Usage:
    python examples/simulate_and_visualize.py
    python examples/simulate_and_visualize.py --duration 120
    python examples/simulate_and_visualize.py --gap 100 --speed 0.3
    python examples/simulate_and_visualize.py --plots-only
    python examples/simulate_and_visualize.py --gap 150 --bed-depth 60 --duration 90
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import time
from pathlib import Path

import numpy as np

# Where to store/load latest calibration (coupling_factor, k_evap, gap_rate)
_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "utility_docs" / "calibration_latest.json"


def main():
    # Single source of truth: config/controller/calibration all read from this file
    os.environ.setdefault("AIRCLASSIFIER_CALIBRATION_FILE", str(_CALIBRATION_FILE))

    parser = argparse.ArgumentParser(
        description="Run GP-15 RF heating simulation and visualize results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python examples/simulate_and_visualize.py                        # Run#1 defaults (61 kg)
    python examples/simulate_and_visualize.py --mass 30 --speed 0.5  # 30 kg at 0.5 m/min
    python examples/simulate_and_visualize.py --duration 120         # Override: 120 s fixed
    python examples/simulate_and_visualize.py --plots-only           # Batch mode (no 3D)
    python examples/simulate_and_visualize.py --gap 80 --bed-depth 50 --mass 100

    # Calibrate model against actual PLC data, then simulate:
    python examples/simulate_and_visualize.py --calibrate "utility_docs/Run1 RF data(in).csv"
""",
    )
    parser.add_argument("--mass", type=float, default=61.0,
                        help="Run mass in kg (default 61, from Run#1)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override: simulation duration in seconds (computed from mass if omitted)")
    parser.add_argument("--gap", type=float, default=75.0,
                        help="Electrode gap in mm (default 75, from Run#1)")
    parser.add_argument("--bed-depth", type=float, default=25.0,
                        help="Material bed depth / feeder gap in mm (default 25, from Run#1)")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="Belt speed in m/min (default 0.2, from Run#1)")
    parser.add_argument("--moisture", type=float, default=0.10,
                        help="Initial moisture wet basis fraction (default 0.10)")
    parser.add_argument("--temp", type=float, default=17.6,
                        help="Initial temperature in C (default 17.6, from Run#1)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip 3D PyVista view, show only matplotlib plots")
    parser.add_argument("--calibrate", type=str, default=None, metavar="CSV",
                        help="Calibrate model against PLC CSV data before running "
                             "(e.g., --calibrate utility_docs/Run1\\ RF\\ data\\(in\\).csv)")
    parser.add_argument("--cal-duration", type=float, default=300,
                        help="Calibration window in seconds (default 300). "
                             "Use 0 for full PLC recording.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU device (default: auto-detect CUDA)")
    parser.add_argument("--dark", action="store_true",
                        help="Dark theme for 3D viewer")
    args = parser.parse_args()

    # ── 1. Configure ─────────────────────────────────────────────────
    from airclassifier.pretreatment import (
        GP15Simulator,
        MachineConfig,
        MaterialProperties,
        Recipe,
    )
    from airclassifier.pretreatment.geometry.assembly import COMPONENT_COLORS

    config = MachineConfig()
    material = MaterialProperties(
        initial_moisture_wb=args.moisture,
        initial_temperature_c=args.temp,
        bed_depth_m=args.bed_depth / 1000.0,
    )

    # Gap rate to apply to controller after simulator is created (from JSON or calibration result)
    from airclassifier.pretreatment.calibration_store import get_calibration_defaults
    gap_rate_to_apply = get_calibration_defaults()[2]

    # ── 1b. Calibrate against PLC data (optional) ───────────────────
    if args.calibrate:
        from airclassifier.pretreatment.calibration import (
            CalibrationOptimizer, load_plc_data,
        )
        print(f"Loading PLC data: {args.calibrate}")
        plc = load_plc_data(args.calibrate)
        print(f"  {plc.n_samples} samples, {plc.duration_s:.0f} s")
        print(f"  Ia: {plc.anode_current_a.min():.2f}-{plc.anode_current_a.max():.2f} A")
        print(f"  Temp: {plc.product_temp_c.min():.0f}-{plc.product_temp_c.max():.0f} C")
        print()

        cal_dur = args.cal_duration if args.cal_duration > 0 else plc.duration_s
        print(f"  Calibration window: {cal_dur:.0f} s ({cal_dur/60:.1f} min)")

        cal = CalibrationOptimizer(
            plc, config=config, material=material,
            sim_duration_s=cal_dur,
        )

        print("Baseline fit (before calibration):")
        baseline = cal.evaluate_current()
        print(f"  T_sim={baseline['T_sim_final']:.1f} vs T_plc={baseline['T_plc_final']:.1f} C")
        print(f"  gap_sim={baseline['gap_sim_final']:.1f} vs gap_plc={baseline['gap_plc_final']:.1f} mm")
        print(f"  loss={baseline['loss_total']:.1f}")
        print()

        print("Running calibration optimizer...")
        cal_result = cal.run(maxiter=15)
        print()
        print(cal_result)
        print()

        # Apply calibrated parameters and persist for future runs (single source of truth)
        cal_result.apply(config, material)
        gap_rate_to_apply = cal_result.gap_adjust_rate_mm_s
        from airclassifier.pretreatment.calibration_store import save_calibration
        save_calibration(cal_result, _CALIBRATION_FILE)
        print(f"Applied: coupling={config.oscillator_coupling_factor:.4f}, "
              f"k_evap={material.k_evap:.2e}, gap_rate={gap_rate_to_apply:.4f} mm/s")
        print(f"Saved to {_CALIBRATION_FILE} (used when running without --calibrate)")
        print()

    # ── 2. Create simulator (builds machine assembly + physics) ──────
    print("=" * 60)
    print("  GP-15 RF Dielectric Heating -- Simulation")
    print("=" * 60)
    print()
    print("Creating GP-15 simulator ...")
    print("  Architecture: GP15Simulator -> GP15MachineAssembly")
    print("                             -> CoupledSimulator (9-step loop)")

    device = "cpu" if args.cpu else None  # None = auto-detect CUDA
    sim = GP15Simulator(
        config=config,
        material=material,
        device=device,
        use_tvd=True,
        enable_controller=True,
        enable_corrections=False,  # skip corrections for speed
    )
    print(f"  Device:  {sim._device}")
    print(f"  Physics: {'GPU (Warp CUDA kernels)' if getattr(sim._sim, '_use_gpu', False) else 'CPU (NumPy)'}")
    sim._sim.update_parameters(gap_adjust_rate=gap_rate_to_apply)

    # ── 3. Load recipe ───────────────────────────────────────────────
    recipe = Recipe(
        name="example_run",
        recipe_number=1,
        electrode_gap_mm=args.gap,
        belt_speed_m_per_min=args.speed,
        run_mass_kg=args.mass,
        extraction_fan_hz=35.0,
        heater_bank_1_on=True,
        heater_bank_2_on=True,
    )
    sim.load_recipe(recipe)

    # Compute run duration from mass (or use explicit override)
    if args.duration is not None:
        run_duration = args.duration
    else:
        run_duration = sim.compute_run_duration(recipe)
    args.duration = run_duration

    # ── 4. Print assembly info ───────────────────────────────────────
    info = sim.assembly.get_assembly_info()
    nx, ny, nz = sim.grid_shape
    dx, dy, dz = sim.cell_sizes
    belt_stack = config.belt_stack_thickness_m
    air_gap = max(0, args.gap / 1000.0 - args.bed_depth / 1000.0 - belt_stack)
    residence_s = config.oven_length_m / (args.speed / 60.0) if args.speed > 0 else 0

    # Throughput from manual Chapter 5 formula
    rho_bulk = material.bulk_density(material.initial_moisture_wb)
    throughput_kg_h = rho_bulk * material.bed_depth_m * config.belt_width_m * (args.speed / 60.0) * 3600

    print()
    print(f"  Machine:           {info['machine']}")
    print(f"  RF zone:           {info['rf_zone_length_m']:.2f} m  "
          f"(x = {info['rf_zone_x_start_m']:.2f} - {info['rf_zone_x_end_m']:.2f} m)")
    print(f"  Belt width:        {info['belt_width_m'] * 1000:.0f} mm")
    print(f"  Electrode gap:     {args.gap:.0f} mm")
    print(f"  Bed depth:         {args.bed_depth:.0f} mm (feeder gap)")
    print(f"  Belt stack:        {belt_stack * 1000:.1f} mm")
    print(f"  Air gap:           {air_gap * 1000:.0f} mm")
    print(f"  Residence time:    {residence_s:.1f} s")
    print(f"  Simulation grid:   {nx} x {ny} x {nz} = {nx * ny * nz:,} cells")
    print(f"  Cell sizes:        dx={dx * 1000:.1f} mm  dy={dy * 1000:.1f} mm  "
          f"dz={dz * 1000:.1f} mm")
    print(f"  Initial moisture:  {args.moisture:.0%} (wet basis)")
    print(f"  Initial temp:      {args.temp:.1f} C")
    print(f"  Run mass:          {args.mass:.1f} kg")
    print(f"  Throughput:        {throughput_kg_h:.0f} kg/h")
    print(f"  Run duration:      {run_duration:.0f} s ({run_duration/60:.1f} min)")
    print()

    # ── 5. Run simulation or launch live 3D ─────────────────────────
    if not args.plots_only:
        try:
            import pyvista as pv
            print(f"Running LIVE simulation  |  {args.mass:.1f} kg  |  "
                  f"{run_duration:.0f} s ({run_duration/60:.1f} min)  |  "
                  f"belt {args.speed} m/min ...")
            print("  3D window will update in real-time.")
            print()
            result = _run_live_3d(sim, recipe, args, info, material)
        except ImportError:
            print("PyVista not installed -- falling back to batch mode.")
            print("Install with: pip install pyvista")
            args.plots_only = True
        except Exception as e:
            print(f"Live 3D failed ({e}) -- falling back to batch mode.")
            args.plots_only = True

    if args.plots_only:
        print(f"Running simulation for {run_duration:.0f} s ({run_duration/60:.1f} min)  |  "
              f"{args.mass:.1f} kg  |  belt {args.speed} m/min ...")
        t0 = time.time()
        result = sim.run(duration_s=run_duration, adaptive_dt=True)
        elapsed = time.time() - t0
        _print_results(sim, result, elapsed)

    # ── 6. Matplotlib plots ──────────────────────────────────────────
    import matplotlib.pyplot as plt

    outlet = sim.get_outlet_conditions()
    ts = result.time_series
    if not ts.get("time_s"):
        print("No time-series data -- skipping plots.")
        return

    t_arr = np.array(ts["time_s"])
    t_min = t_arr / 60.0  # time in minutes for readability

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(
        f"GP-15 Digital Twin -- {args.mass:.0f} kg Whole Yellow Pea  |  "
        f"Gap {args.gap:.0f} mm  |  Bed {args.bed_depth:.0f} mm  |  "
        f"Belt {args.speed} m/min  |  {args.duration:.0f} s",
        fontsize=12, fontweight="bold",
    )

    # ── Row 1: Temperature, Moisture, Electrode Gap ──────────────

    # [0,0] Temperature
    ax = axes[0, 0]
    ax.fill_between(t_min, ts["T_mean_c"], ts["T_max_c"],
                    alpha=0.15, color="red", label="T range")
    ax.plot(t_min, ts["T_outfeed_c"], "r-", linewidth=2, label="T outfeed")
    ax.plot(t_min, ts["T_mean_c"], "r-", linewidth=0.8, alpha=0.5, label="T mean")
    ax.axhline(70, color="orange", linestyle=":", alpha=0.6, label="Denaturation 70\u00b0C")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Temperature [\u00b0C]")
    ax.set_title("Material Temperature")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # [0,1] Moisture
    ax = axes[0, 1]
    ax.plot(t_min, np.array(ts["M_outfeed_wb"]) * 100, "b-", linewidth=2,
            label="M outfeed")
    ax.plot(t_min, np.array(ts["M_mean_wb"]) * 100, "b-", linewidth=0.8,
            alpha=0.4, label="M mean (all)")
    ax.axhline(material.target_moisture_wb * 100, color="green",
               linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Target {material.target_moisture_wb:.0%}")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Moisture [% wb]")
    ax.set_title("Moisture Content")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # [0,2] Electrode gap (MRH controller)
    ax = axes[0, 2]
    ax.plot(t_min, ts["electrode_gap_mm"], "g-", linewidth=2, label="Actual gap")
    ax.axhline(args.gap, color="gray", linestyle="--", alpha=0.5,
               label=f"Setpoint {args.gap:.0f} mm")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Electrode Gap [mm]")
    ax.set_title("MRH Gap Control")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 2: RF Power, Anode Current, Energy Balance ───────────

    # [1,0] RF Power + Evaporative power
    ax = axes[1, 0]
    ax.plot(t_min, ts["rf_power_kw"], "m-", linewidth=2, label="RF power (in)")
    ax.plot(t_min, ts["evap_power_kw"], "c-", linewidth=1.5,
            alpha=0.8, label="Evap. cooling")
    ax.axhline(config.max_rf_power_kw, color="gray", linestyle=":",
               alpha=0.4, label=f"Rated max {config.max_rf_power_kw} kW")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Energy Balance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [1,1] Anode current
    ax = axes[1, 1]
    ax.plot(t_min, ts["anode_current_a"], "k-", linewidth=2, label="Ia")
    ax.axhline(recipe.mrh_amps, color="red", linestyle="--", alpha=0.6,
               linewidth=1.5, label=f"MRH = {recipe.mrh_amps} A")
    ax.axhline(recipe.mrl_amps, color="orange", linestyle="--", alpha=0.6,
               linewidth=1.5, label=f"MRL = {recipe.mrl_amps} A")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Anode Current [A]")
    ax.set_title("Anode Current (Ia)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # [1,2] Cumulative energy consumed
    ax = axes[1, 2]
    ax.plot(t_min, ts["total_energy_kwh"], "m-", linewidth=2, label="RF energy")
    ax2 = ax.twinx()
    ax2.plot(t_min, np.array(ts["water_removed_kg"]) * 1000, "c-",
             linewidth=1.5, label="Water removed")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Energy [kWh]", color="m")
    ax2.set_ylabel("Water Removed [g]", color="c")
    ax.set_title("Cumulative Totals")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Row 3: Specific energy, Mass in bin, Outfeed cross-section ─

    # [2,0] Specific energy
    ax = axes[2, 0]
    se = np.array(ts["specific_energy_kwh_per_kg"])
    valid = se > 0
    if valid.any():
        ax.plot(t_min[valid], se[valid], "k-", linewidth=1.5)
        ax.axhline(1.0, color="green", linestyle="--", alpha=0.5,
                   label="Manual target: 1.0 kWh/kg")
        ax.axhline(1.0/0.6, color="orange", linestyle=":", alpha=0.5,
                   label="Low S/V factor: 1.67 kWh/kg")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("kWh / kg water")
    ax.set_title("Specific Energy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # [2,1] Collected mass in bin
    ax = axes[2, 1]
    if hasattr(sim.particles, 'collected_mass_kg'):
        collected_kg = sim.particles.collected_mass_kg
        n_collected = sim.particles.collected_count
        ax.bar(["Infeed\n(run mass)", "Collected\n(bin)"],
               [args.mass, collected_kg],
               color=["#4169E1", "#DAA520"], alpha=0.8)
        ax.set_ylabel("Mass [kg]")
        ax.set_title(f"Material Accounting ({n_collected} particles)")
    else:
        ax.text(0.5, 0.5, "No particle data", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis="y")

    # [2,2] Outfeed temperature cross-section
    if outlet.temperature_field is not None:
        im = axes[2, 2].imshow(
            outlet.temperature_field,
            aspect="auto", origin="lower", cmap="hot",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        axes[2, 2].set_xlabel("Z -- belt width [mm]")
        axes[2, 2].set_ylabel("Y -- gap [mm]")
        axes[2, 2].set_title(
            f"Outfeed T  (avg {outlet.avg_temperature_c:.1f}\u00b0C, "
            f"max {outlet.max_temperature_c:.1f}\u00b0C)")
        plt.colorbar(im, ax=axes[2, 2], label="\u00b0C", shrink=0.8)
    else:
        axes[2, 2].text(0.5, 0.5, "No field data", ha="center", va="center",
                        transform=axes[2, 2].transAxes)

    plt.tight_layout()

    # ── 7. Outfeed cross-section (\u00a79.1) ──────────────────────────────
    if outlet.temperature_field is not None and outlet.moisture_field is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
        fig2.suptitle(
            f"Outfeed Cross-Section  --  Pipeline Output to Milling  |  "
            f"Residence {outlet.residence_time_s:.0f} s  |  "
            f"Throughput {outlet.throughput_kg_per_hr:.0f} kg/h",
            fontsize=11, fontweight="bold",
        )
        im_t = axes2[0].imshow(
            outlet.temperature_field, aspect="auto", origin="lower",
            cmap="hot",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        axes2[0].set_xlabel("Z -- belt width [mm]")
        axes2[0].set_ylabel("Y -- gap [mm]")
        axes2[0].set_title(
            f"Temperature  (avg {outlet.avg_temperature_c:.1f}\u00b0C, "
            f"max {outlet.max_temperature_c:.1f}\u00b0C)")
        plt.colorbar(im_t, ax=axes2[0], label="\u00b0C")

        im_m = axes2[1].imshow(
            outlet.moisture_field * 100, aspect="auto", origin="lower",
            cmap="Blues",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        axes2[1].set_xlabel("Z -- belt width [mm]")
        axes2[1].set_ylabel("Y -- gap [mm]")
        axes2[1].set_title(
            f"Moisture  (avg {outlet.avg_moisture_wb:.1%}, "
            f"CV {outlet.moisture_uniformity:.3f})  |  "
            f"Spec. energy {outlet.specific_energy_kwh_per_kg:.2f} kWh/kg water")
        plt.colorbar(im_m, ax=axes2[1], label="% wb")
        plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Print results to console
# ─────────────────────────────────────────────────────────────────────

def _print_results(sim, result, elapsed):
    outlet = sim.get_outlet_conditions()
    ts = result.time_series
    print()
    print("-" * 60)
    print("  RESULTS")
    print("-" * 60)
    print(f"  Outfeed moisture:          {outlet.avg_moisture_wb:.2%}")
    print(f"  Outfeed temperature:       {outlet.avg_temperature_c:.1f} C")
    print(f"  Max temperature:           {outlet.max_temperature_c:.1f} C")
    print(f"  Moisture uniformity (CV):  {outlet.moisture_uniformity:.4f}")
    print()
    print(f"  RF energy consumed:        {result.energy_consumed_kwh:.4f} kWh")
    print(f"  Specific energy:           {outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water")
    print(f"  Throughput:                {result.throughput_kg_per_h:.0f} kg/h")
    # Final gap from controller
    if ts.get("electrode_gap_mm"):
        final_gap = ts["electrode_gap_mm"][-1]
        print(f"  Final electrode gap:       {final_gap:.1f} mm")
    # Particle mass collected
    if hasattr(sim.particles, 'collected_mass_kg'):
        print(f"  Mass collected (bin):      {sim.particles.collected_mass_kg:.2f} kg")
    print()
    print(f"  Simulation wall-clock:     {elapsed:.2f} s")
    n_steps = len(ts.get("time_s", []))
    if n_steps > 0:
        print(f"  Timesteps completed:       {n_steps}")
        print(f"  Speed:                     {n_steps / max(elapsed, 0.001):.0f} steps/s")
    print("-" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────
#  Live 3D Simulation (PyVista)
# ─────────────────────────────────────────────────────────────────────

def _mesh_to_polydata(v, t):
    """Build PyVista PolyData from (vertices, triangles)."""
    import pyvista as pv
    n = t.shape[0]
    faces = np.empty((n, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = t
    return pv.PolyData(v.copy(), faces.ravel())


def _collect_roller_entries(layout):
    """Flatten roller layout dict into (name, x, y, r) tuples."""
    entries = []
    for key in ("head", "tail", "tension", "tracker_1", "tracker_2",
                "driven_sprocket", "drive_sprocket"):
        if key in layout:
            x, y, r, _kind = layout[key]
            entries.append((key, x, y, r))
    for key in ("return_rollers_before", "return_rollers_after",
                "carrying_idlers"):
        for i, (x, y, r, _k) in enumerate(layout.get(key, [])):
            entries.append((f"{key}_{i}", x, y, r))
    return entries


def _run_live_3d(sim, recipe, args, info, material):
    """Run the simulation live with coupled conveyor belt animation.

    All simulation logic (physics stepping, particle updates, conveyor
    control) is handled by ``GP15Simulator`` and its internal
    ``CoupledSimulator``.  This function only does visualization:
    reading state from the simulator's public API and updating
    PyVista meshes.
    """
    import math
    import pyvista as pv
    from airclassifier.pretreatment.geometry.assembly import COMPONENT_COLORS
    from airclassifier.pretreatment.kernels.transport import rotate_mesh_around_z_axis

    # Ensure simulator is ready (builds mask, connects particles)
    sim._ensure_initialized()

    # Use simulator's get_mesh() which skips the static material_bed
    # (particles replace it) and attaches field data
    meshes = sim.get_mesh()
    particle_sys = sim.particles     # owned by the simulator

    # ── Build plotter ─────────────────────────────────────────────
    plotter = pv.Plotter()
    bg = "#1a1a2e" if args.dark else "white"
    plotter.set_background(bg)
    plotter.camera.up = (0, 1, 0)

    xray_opacities = {
        "conveyor_frame": 0.08, "oven_chamber": 0.06,
        "rollers": 0.60, "belt": 0.40,
        "upper_electrode": 0.30, "lower_electrode": 0.25,
        "infeed_hopper": 0.70, "infeed_tunnel": 0.20,
        "outfeed_tunnel": 0.20, "collection_bin": 0.55,
        "emu_housing": 0.10, "generator": 0.20, "rf_feed": 0.80,
    }

    animated_names = {"rollers", "belt", "fields"}

    # ── Add STATIC machine geometry ───────────────────────────────
    for name, item in meshes.items():
        if name in animated_names:
            continue
        if not isinstance(item, tuple) or len(item) != 3:
            continue
        v, t, meta = item
        style = COMPONENT_COLORS.get(name, {})
        color = style.get("color", "#888888")
        label = style.get("label", name)
        opacity = xray_opacities.get(name, style.get("opacity", 0.8))
        if opacity < 0.01:
            continue
        pd = _mesh_to_polydata(v, t)
        plotter.add_mesh(pd, color=color, opacity=opacity,
                         smooth_shading=True, label=label)

    # ── Add ANIMATED rollers ──────────────────────────────────────
    rollers_v, rollers_t, _ = meshes["rollers"]
    rollers_base = rollers_v.copy()
    rollers_pd = _mesh_to_polydata(rollers_v, rollers_t)
    plotter.add_mesh(rollers_pd, color="#808090", opacity=0.85,
                     smooth_shading=True, label="Rollers", name="rollers")

    conveyor_geom = sim.assembly.conveyor
    conv_params = conveyor_geom.params
    roller_layout = conveyor_geom._roller_layout
    roller_entries = _collect_roller_entries(roller_layout)

    roller_z_max = conv_params.belt_z1 + 0.05
    roller_z_min = conv_params.belt_z0 - 0.05
    z_in_range = ((rollers_base[:, 2] >= roller_z_min) &
                  (rollers_base[:, 2] <= roller_z_max))

    roller_vert_masks = {}
    for rname, cx, cy, r in roller_entries:
        if "sprocket" in rname:
            continue
        dx_r = rollers_base[:, 0] - cx
        dy_r = rollers_base[:, 1] - cy
        dist_xy = np.sqrt(dx_r * dx_r + dy_r * dy_r)
        mask = (dist_xy < r * 1.6) & z_in_range
        if mask.any():
            roller_vert_masks[rname] = (mask, cx, cy, r)

    # ── Add ANIMATED belt ─────────────────────────────────────────
    belt_v, belt_t, _ = meshes["belt"]
    belt_pd = _mesh_to_polydata(belt_v, belt_t)

    belt_path_xy = belt_v[0::4, :2]
    _seg = np.diff(belt_path_xy, axis=0)
    _seg_lens = np.sqrt(_seg[:, 0] ** 2 + _seg[:, 1] ** 2)
    belt_arc = np.concatenate([[0.0], np.cumsum(_seg_lens)])
    belt_total_len = belt_arc[-1]
    belt_arc_per_vert = np.repeat(belt_arc, 4)

    head_circ = 2.0 * math.pi * conv_params.head_roller_radius_m
    n_belt_bands = max(1, round(belt_total_len / head_circ))
    belt_band_len = belt_total_len / n_belt_bands

    belt_pd.point_data["motion"] = (
        0.5 + 0.5 * np.cos(2.0 * math.pi * belt_arc_per_vert / belt_band_len)
    ).astype(np.float32)

    plotter.add_mesh(
        belt_pd, scalars="motion",
        cmap=["#3458a5", "#5a7fd4"],
        clim=[0, 1], opacity=0.92,
        show_edges=True, edge_color="#2a4a9a", line_width=0.4,
        smooth_shading=True, name="belt",
        show_scalar_bar=False, label="Belt (PTFE)",
    )

    # ── Add particle point cloud ──────────────────────────────────
    # Fixed-size buffer; dead particles hidden off-screen.
    particle_cloud = pv.PolyData(particle_sys.pos.copy())
    particle_cloud["Temperature"] = particle_sys.temperature.copy()

    T_ambient = material.initial_temperature_c
    plotter.add_mesh(
        particle_cloud, scalars="Temperature",
        cmap="hot", clim=[T_ambient, T_ambient + 15],
        point_size=5.0, render_points_as_spheres=True,
        opacity=0.90, show_scalar_bar=False,
        name="particles", label="Material Particles",
    )

    # ── Build rectilinear grid for temperature field ──────────────
    nx, ny, nz = sim.grid_shape
    dx, dy, dz = sim.cell_sizes
    x0, y0, z0 = sim.get_field_world_origin()

    x_coords = np.linspace(x0, x0 + nx * dx, nx + 1)
    y_coords = np.linspace(y0, y0 + ny * dy, ny + 1)
    z_coords = np.linspace(z0, z0 + nz * dz, nz + 1)

    field_grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)
    mask_flat = sim.material_mask.flatten(order="F")
    field_grid.cell_data["Temperature"] = sim.temperature_field.flatten(order="F")
    field_grid.cell_data["zone"] = mask_flat
    mat_indices = np.where(mask_flat == 1)[0]

    mat_grid = field_grid.threshold(value=1, scalars="zone")
    plotter.add_mesh(
        mat_grid, scalars="Temperature",
        cmap="hot", clim=[T_ambient, T_ambient + 15],
        opacity=0.90, show_scalar_bar=True,
        scalar_bar_args={
            "title": "Temperature [\u00b0C]",
            "position_x": 0.82, "width": 0.12,
        },
        label="Temperature Field",
    )

    # ── Legend, axes, title, camera ────────────────────────────────
    legend_bg = (0.1, 0.1, 0.15, 0.8) if args.dark else "white"
    plotter.add_legend(loc="upper left", bcolor=legend_bg)
    plotter.add_title("GP-15 Live  |  Initializing...", font_size=10)
    plotter.add_axes()
    plotter.reset_camera()
    plotter.camera.azimuth = -55
    plotter.camera.elevation = 18
    plotter.camera.zoom(1.1)

    # ── Smooth adaptive pacing ──────────────────────────────────────
    # Steps per frame ramps smoothly from slow (transient visible)
    # to fast (steady state).  No abrupt jump that causes flicker.
    v_belt_init = recipe.belt_speed_m_per_min / 60.0
    residence_s = sim.config.oven_length_m / max(v_belt_init, 1e-6)
    transient_sim_s = 2.0 * residence_s
    target_fps = 20.0
    frame_dt = 1.0 / target_fps
    steps_min = max(1, int(transient_sim_s / (15.0 * target_fps * 0.3)))
    steps_max = 60

    t_end = args.duration
    t0_wall = time.time()
    conv_ctrl = sim.conveyor  # public accessor

    plotter.show(interactive_update=True, auto_close=False)

    # ── Render loop (visualization only — sim logic in simulator) ─
    try:
        while True:
            t_frame_start = time.perf_counter()

            try:
                plotter.iren.process_events()
            except Exception:
                break

            # Smooth ramp: steps_min during transient, linearly
            # increasing to steps_max over 2x the transient period.
            t_sim = sim.sim_time
            ramp = min(t_sim / max(transient_sim_s * 2, 1.0), 1.0)
            steps = int(steps_min + ramp * (steps_max - steps_min))

            # Step the simulator (physics + particles via coupling loop)
            finished = False
            for _ in range(steps):
                if sim.sim_time >= t_end - 1e-12:
                    finished = True
                    break
                dt = sim.compute_stable_dt()
                dt = min(dt, t_end - sim.sim_time)
                sim.step(dt)

            # ── Visual updates (read-only from simulator state) ───

            # Rollers
            animated_verts = rollers_base.copy()
            for rname, (mask, cx, cy, r) in roller_vert_masks.items():
                angle = -conv_ctrl.roller_angle(r)
                animated_verts[mask] = rotate_mesh_around_z_axis(
                    rollers_base[mask], cx, cy, angle,
                )
            rollers_pd.points = animated_verts

            # Belt scroll
            _phase = conv_ctrl.state.belt_position_m % belt_total_len
            _shifted = (belt_arc_per_vert - _phase + belt_total_len) % belt_total_len
            belt_pd.point_data["motion"] = (
                0.5 + 0.5 * np.cos(2.0 * math.pi * _shifted / belt_band_len)
            ).astype(np.float32)

            # Particles (already stepped by coupling loop step 10)
            updated_pos = particle_sys.pos.copy()
            dead = (particle_sys.state == particle_sys._STATE_DEAD)
            updated_pos[dead] = [0.0, -100.0, 0.0]
            particle_cloud.points = updated_pos
            particle_cloud.point_data["Temperature"] = particle_sys.temperature.copy()

            # Temperature field
            T_flat = sim.temperature_field.flatten(order="F")
            mat_grid.cell_data["Temperature"] = T_flat[mat_indices]

            # Title
            hist = sim.history
            if hist:
                last = hist[-1]
                hopper_n = particle_sys.hopper_count
                riding_n = particle_sys.riding_count
                title = (f"GP-15 Live  |  t={sim.sim_time:.0f}/{t_end:.0f} s  |  "
                         f"T_out={last.T_outfeed_c:.1f} C  |  "
                         f"M_out={last.M_outfeed_wb:.1%}  |  "
                         f"P={last.rf_power_kw:.1f} kW  |  "
                         f"Hopper:{hopper_n}  Belt:{riding_n}")
            else:
                title = f"GP-15 Live  |  t={sim.sim_time:.0f}/{t_end:.0f} s"

            if finished:
                elapsed_w = time.time() - t0_wall
                riding_n = particle_sys.riding_count
                hopper_n = particle_sys.hopper_count
                if riding_n > 0 or hopper_n > 0:
                    title = (f"GP-15 Wind-down  |  Belt clearing: "
                             f"{riding_n} on belt, {hopper_n} in hopper  |  "
                             f"wall={elapsed_w:.1f} s")
                else:
                    title = (f"GP-15 DONE  |  {sim.sim_time:.0f} s  |  "
                             f"M_out={hist[-1].M_outfeed_wb:.1%}  |  "
                             f"wall={elapsed_w:.1f} s  |  Belt clear")

            plotter.add_title(title, font_size=10)

            try:
                plotter.render()
            except Exception:
                break

            elapsed_frame = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, frame_dt - elapsed_frame))

            if finished:
                riding_n = particle_sys.riding_count
                falling_n = int(np.sum(particle_sys.state == particle_sys._STATE_FALLING))
                hopper_n = particle_sys.hopper_count
                belt_active = riding_n + falling_n + hopper_n

                if belt_active > 0:
                    # ── Wind-down: step particles in large time chunks
                    # No physics needed — just belt transport + free-fall.
                    # Use large dt (Courant limit for advection only) and
                    # many sub-steps per frame to clear the belt quickly.
                    v_belt = conv_ctrl.state.belt_speed_m_per_s
                    if v_belt > 0:
                        dt_wind = 0.8 * dx / v_belt  # Courant-safe for advection
                    else:
                        dt_wind = 1.0
                    for _ in range(200):  # many sub-steps per render frame
                        particle_sys.step(
                            dt_sim=dt_wind,
                            belt_speed_m_per_s=v_belt,
                        )
                        if particle_sys.riding_count == 0 and particle_sys.hopper_count == 0:
                            break
                    conv_ctrl.step(dt_wind * 200)
                    finished = False  # keep looping until belt clear
                else:
                    # Belt is fully clear — show final state and stop
                    plotter.show()
                    break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            plotter.close()
        except Exception:
            pass

    elapsed = time.time() - t0_wall
    result = sim.build_result()
    _print_results(sim, result, elapsed)
    return result


if __name__ == "__main__":
    main()
