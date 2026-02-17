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

    # Run#2 defaults (90 kg)
    python examples/simulate_and_visualize.py --mass 90 --gap 75 --bed-depth 35 --speed 0.2 --temp 17.0 --duration 2820 --moisture 0.118067

    # Run#3 defaults (60.5 kg)
    python examples/simulate_and_visualize.py --mass 60.5 --gap 75 --bed-depth 25--speed 0.2 --temp 17.0 --duration 2820

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
    timing = sim.compute_run_timing(recipe)
    user_specified_duration = args.duration is not None
    if user_specified_duration:
        run_duration = args.duration
    else:
        run_duration = timing["total_duration_s"]
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
    # Show timing breakdown
    feed_s = timing["feed_time_s"]
    oven_s = timing["oven_clearing_s"]
    oven_m = timing["oven_clearing_m"]
    belt_m = timing["belt_length_m"]
    if not user_specified_duration:
        # Auto-calculated duration with breakdown
        print(f"  Run duration:      {run_duration:.0f} s ({run_duration/60:.1f} min)")
        print(f"    Feed time:       {feed_s:.0f} s  (hopper → belt)")
        print(f"    Oven clearing:   {oven_s:.0f} s  ({oven_m:.2f} m to oven exit)")
        print(f"    Belt wind-down:  after physics  ({belt_m:.2f} m to bin)")
    else:
        # User override - show both actual and calculated
        calc_dur = timing["total_duration_s"]
        print(f"  Run duration:      {run_duration:.0f} s ({run_duration/60:.1f} min)  [user override]")
        print(f"    (Calculated:     {calc_dur:.0f} s = {feed_s:.0f}s feed + {oven_s:.0f}s oven clearing)")
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

    # [0,0] Temperature & protein quality (pretreatment) — not drying.
    # Secondary axis: weighted globulin native loss (7S+11S). For pretreatment we
    # want to keep this LOW (target typically <15%); high % = over-processing.
    ax = axes[0, 0]
    ax.fill_between(t_min, ts["T_mean_c"], ts["T_max_c"],
                    alpha=0.15, color="red", label="Trange")
    if "T_outfeed_sensor_c" in ts:
        ax.plot(t_min, ts["T_outfeed_sensor_c"], "r-", linewidth=2, label="Tsensor (P75)")
    ax.plot(t_min, ts["T_outfeed_c"], "r--", linewidth=1.2, alpha=0.6, label="Toutfeed (mean)")
    ax.axhline(76, color="orange", linestyle=":", alpha=0.6, label="Legumin onset ~76\u00b0C")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Temperature [\u00b0C]")
    ax.set_title("Temperature & protein quality (pretreatment)")
    ax.grid(True, alpha=0.3)
    # Secondary Y-axis: globulin native loss (7S+11S weighted). Lower = better preservation.
    PRETREATMENT_MAX_DENAT_PCT = 15.0  # typical target: keep denatured fraction below this
    if "protein_denaturation" in ts:
        ax_d = ax.twinx()
        denat_pct = np.array(ts["protein_denaturation"]) * 100
        ax_d.plot(t_min, denat_pct, "k-", linewidth=1.5, alpha=0.7,
                  label="Globulin native loss [%]")
        ax_d.axhline(PRETREATMENT_MAX_DENAT_PCT, color="green", linestyle="--",
                     alpha=0.6, linewidth=1, label=f"Target max {PRETREATMENT_MAX_DENAT_PCT:.0f}%")
        ax_d.set_ylabel("Globulin native loss [%]\n(7S+11S; lower = better)", color="k", fontsize=8)
        ax_d.set_ylim(bottom=0)
        ax_d.tick_params(axis="y", labelsize=7)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_d.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")
    else:
        ax.legend(fontsize=6, loc="upper left")

    # [0,1] Moisture
    ax = axes[0, 1]
    ax.plot(t_min, np.array(ts["M_outfeed_wb"]) * 100, "b-", linewidth=2,
            label="M outfeed")
    ax.plot(t_min, np.array(ts["M_mean_wb"]) * 100, "b-", linewidth=0.8,
            alpha=0.4, label="M mean (all)")
    target_pct = material.target_moisture_wb * 100
    ax.axhline(target_pct, color="green",
               linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Target {target_pct:.0f}%")
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
        dispatched_kg = sim.particles.dispatched_mass_kg
        collected_kg = sim.particles.collected_mass_kg
        n_collected = sim.particles.collected_count
        ax.bar(["Input", "Collected"],
               [dispatched_kg, collected_kg],
               color=["#4169E1", "#DAA520"], alpha=0.8)
        ax.set_ylabel("Mass [kg]")
        # Show mass balance percentage
        if dispatched_kg > 0:
            balance_pct = (collected_kg - dispatched_kg) / dispatched_kg * 100
            ax.set_title(f"Mass Balance ({n_collected} particles, {balance_pct:+.1f}%)")
        else:
            ax.set_title(f"Mass Balance ({n_collected} particles)")
    else:
        ax.text(0.5, 0.5, "No particle data", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis="y")

    # [2,2] Outfeed temperature cross-section (at oven exit; matches time-series peak)
    if outlet.temperature_field is not None:
        im = axes[2, 2].imshow(
            outlet.temperature_field,
            aspect="auto", origin="lower", cmap="hot",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        axes[2, 2].set_xlabel("Z -- belt width [mm]")
        axes[2, 2].set_ylabel("Y -- gap [mm]")
        if getattr(outlet, "at_peak_processing_snapshot", False):
            axes[2, 2].set_title(
                f"Outfeed T at oven exit (peak)  "
                f"sensor {outlet.sensor_temperature_c:.1f}\u00b0C, max {outlet.max_temperature_c:.1f}\u00b0C")
        else:
            axes[2, 2].set_title(
                f"Outfeed T  (sensor {outlet.sensor_temperature_c:.1f}\u00b0C, "
                f"max {outlet.max_temperature_c:.1f}\u00b0C)")
        plt.colorbar(im, ax=axes[2, 2], label="\u00b0C", shrink=0.8)
    else:
        axes[2, 2].text(0.5, 0.5, "No field data", ha="center", va="center",
                        transform=axes[2, 2].transAxes)

    plt.tight_layout()

    # ── 7. Outfeed cross-section (\u00a79.1) ──────────────────────────────
    # When run has finished, cross-section shows at-oven-exit (peak) snapshot so it
    # matches the time-series and Run#1 strip data (82–93°C); not cooled bin state.
    if outlet.temperature_field is not None and outlet.moisture_field is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
        peak_note = " at oven exit (peak)" if getattr(outlet, "at_peak_processing_snapshot", False) else ""
        fig2.suptitle(
            f"Outfeed Cross-Section{peak_note}  --  Pipeline Output to Milling  |  "
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
            f"Temperature  (sensor {outlet.sensor_temperature_c:.1f}\u00b0C, "
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
    print(f"  Outfeed temperature:       {outlet.sensor_temperature_c:.1f} C")
    print(f"  Max temperature:           {outlet.max_temperature_c:.1f} C")
    print(f"  Moisture uniformity (CV):  {outlet.moisture_uniformity:.4f}")
    print()
    # Phase 4: protein quality (weighted 7S+11S native loss). Pretreatment target: keep low (<15%).
    denat = outlet.protein_denaturation_fraction
    print(f"  Protein quality (globulin native loss): {denat:.1%}  (pretreatment target typically <15%)")
    if hasattr(sim.particles, 'vicilin_native'):
        collected = sim.particles.state == 2  # STATE_COLLECTED
        riding = sim.particles.state == 0     # STATE_RIDING
        active = collected if collected.any() else riding
        if active.any():
            vic = 1.0 - float(np.mean(sim.particles.vicilin_native[active]))
            leg = 1.0 - float(np.mean(sim.particles.legumin_native[active]))
            print(f"    Vicilin (7S):            {vic:.1%}  (onset 62 C)")
            print(f"    Legumin (11S):           {leg:.1%}  (onset 76 C)")
    print()
    print(f"  RF energy consumed:        {result.energy_consumed_kwh:.4f} kWh")
    print(f"  Specific energy:           {outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water")
    print(f"  Throughput:                {result.throughput_kg_per_h:.0f} kg/h")
    # Final gap from controller
    if ts.get("electrode_gap_mm"):
        final_gap = ts["electrode_gap_mm"][-1]
        print(f"  Final electrode gap:       {final_gap:.1f} mm")
    # Particle mass accounting
    if hasattr(sim.particles, 'collected_mass_kg'):
        dispatched = sim.particles.dispatched_mass_kg
        collected_kg = sim.particles.collected_mass_kg
        print(f"  Mass input:                {dispatched:.2f} kg")
        print(f"  Mass collected:            {collected_kg:.2f} kg")
        if dispatched > 0:
            mass_balance = (collected_kg - dispatched) / dispatched * 100
            print(f"  Mass balance:              {mass_balance:+.1f}%")
    print()
    # Desirability score (Derringer-Suich composite, 0-10 with 5 dimensions)
    try:
        from airclassifier.pretreatment.desirability import score_desirability
        recipe = sim._recipe
        run_mass = recipe.run_mass_kg if recipe else 0.0
        ds = score_desirability(
            outfeed_temperature_c=outlet.avg_temperature_c,
            max_temperature_c=outlet.max_temperature_c,
            outfeed_moisture_wb=outlet.avg_moisture_wb,
            initial_moisture_wb=sim.material.initial_moisture_wb,
            energy_kwh=result.energy_consumed_kwh,
            run_mass_kg=max(run_mass, 0.001),
        )
        print(f"  Desirability score:        {ds.overall_10:.1f} / 10")
        print(f"    Thermal treatment:       {ds.d_thermal:.2f}")
        print(f"    Flavour (LOX kill):      {ds.d_flavour:.2f}")
        print(f"    Protein preservation:    {ds.d_protein:.2f}")
        print(f"    Moisture retention:      {ds.d_moisture:.2f}")
        print(f"    Energy efficiency:       {ds.d_energy:.2f}")
    except Exception as e:
        print(f"  Desirability score:        (unavailable: {e})")
    print()
    print(f"  Simulation wall-clock:     {elapsed:.2f} s")
    n_steps = len(ts.get("time_s", []))
    if n_steps > 0:
        print(f"  Timesteps completed:       {n_steps}")
        print(f"  Speed:                     {n_steps / max(elapsed, 0.001):.0f} steps/s")
    print("-" * 60)

    # ── Debug: Gap control timeline ──────────────────────────────────
    print()
    print("-" * 60)
    print("  GAP CONTROL DEBUG")
    print("-" * 60)
    if ts.get("electrode_gap_mm") and ts.get("anode_current_a"):
        gap_arr = np.array(ts["electrode_gap_mm"])
        ia_arr = np.array(ts["anode_current_a"])
        t_arr = np.array(ts["time_s"])

        # Find key events
        gap_max_idx = np.argmax(gap_arr)
        gap_max = gap_arr[gap_max_idx]
        gap_max_t = t_arr[gap_max_idx]

        # Find when gap started returning (first decrease after peak)
        gap_return_start_idx = None
        for i in range(gap_max_idx + 1, len(gap_arr)):
            if gap_arr[i] < gap_arr[i-1] - 0.1:
                gap_return_start_idx = i
                break

        # Find when Ia dropped below MRL
        mrl = 1.5  # default MRL
        ia_below_mrl_idx = None
        for i in range(len(ia_arr)):
            if ia_arr[i] < mrl:
                ia_below_mrl_idx = i
                break

        print(f"  Gap peak:         {gap_max:.1f} mm at t={gap_max_t:.0f}s ({gap_max_t/60:.1f} min)")
        print(f"  Gap final:        {gap_arr[-1]:.1f} mm")
        if gap_return_start_idx:
            print(f"  Gap return start: t={t_arr[gap_return_start_idx]:.0f}s ({t_arr[gap_return_start_idx]/60:.1f} min)")
            print(f"    Ia at return:   {ia_arr[gap_return_start_idx]:.3f} A")
        if ia_below_mrl_idx:
            print(f"  Ia < MRL first:   t={t_arr[ia_below_mrl_idx]:.0f}s ({t_arr[ia_below_mrl_idx]/60:.1f} min)")
            print(f"    Ia value:       {ia_arr[ia_below_mrl_idx]:.3f} A")
            print(f"    Gap at that t:  {gap_arr[ia_below_mrl_idx]:.1f} mm")

        # Batch exhaustion timing
        if hasattr(sim, '_sim') and hasattr(sim._sim, '_batch_exhausted'):
            print(f"  Batch exhausted:  {sim._sim._batch_exhausted}")
        if hasattr(sim, '_sim') and hasattr(sim._sim, 'controller'):
            print(f"  Controller batch: {sim._sim.controller._batch_exhausted}")

        # Show Ia trajectory at key points
        print()
        print("  Ia trajectory:")
        print(f"    {'t(s)':>8}  {'t(min)':>8}  {'Ia(A)':>8}  {'Gap(mm)':>10}  {'State':>12}")
        # Sample at regular intervals + key events
        sample_times = [0, 60, 120, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]
        if gap_max_t not in sample_times:
            sample_times.append(gap_max_t)
        if gap_return_start_idx and t_arr[gap_return_start_idx] not in sample_times:
            sample_times.append(t_arr[gap_return_start_idx])
        sample_times = sorted(set(sample_times))

        ctrl_states = ts.get("controller_state", [])
        for t in sample_times:
            if t > t_arr[-1]:
                break
            idx = np.searchsorted(t_arr, t)
            idx = min(idx, len(t_arr) - 1)
            state = ctrl_states[idx] if idx < len(ctrl_states) else ""
            marker = " <-- peak" if abs(t - gap_max_t) < 1 else ""
            print(f"    {t_arr[idx]:8.0f}  {t_arr[idx]/60:8.1f}  {ia_arr[idx]:8.3f}  {gap_arr[idx]:10.1f}  {state:>12}{marker}")

    # ── Debug: Particle system state ─────────────────────────────────
    print()
    print("-" * 60)
    print("  PARTICLE SYSTEM DEBUG")
    print("-" * 60)
    if hasattr(sim, 'particles'):
        ps = sim.particles
        print(f"  Total particles:   {ps.cfg.max_particles}")
        print(f"  Hopper count:      {ps.hopper_count}")
        print(f"  Riding count:      {ps.riding_count}")
        print(f"  Collected count:   {ps.collected_count}")
        print(f"  Dispatched mass:   {ps.dispatched_mass_kg:.2f} kg")
        print(f"  Collected mass:    {ps.collected_mass_kg:.2f} kg")
        print(f"  Run mass target:   {ps.cfg.run_mass_kg:.2f} kg")
        if ps.cfg.run_mass_kg > 0:
            dispatch_pct = ps.dispatched_mass_kg / ps.cfg.run_mass_kg * 100
            print(f"  Dispatch progress: {dispatch_pct:.1f}%")
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

    # Debug tracking
    _last_debug_t = 0.0
    _debug_interval = 60.0  # Print debug every 60 sim seconds
    _batch_exhausted_reported = False
    _gap_return_reported = False
    _gap_peak = 0.0
    _gap_peak_t = 0.0
    hist = None  # Initialize to avoid UnboundLocalError

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

                # ── Periodic debug output ────────────────────────────────
                t_sim = sim.sim_time
                if t_sim - _last_debug_t >= _debug_interval:
                    _last_debug_t = t_sim
                    hist = sim.history
                    if hist:
                        last = hist[-1]
                        ps = particle_sys
                        batch_exh = sim._sim._batch_exhausted if hasattr(sim._sim, '_batch_exhausted') else False
                        ctrl_batch = sim._sim.controller._batch_exhausted if hasattr(sim._sim.controller, '_batch_exhausted') else False
                        # Particle position diagnostics
                        alive = (ps.state != ps._STATE_DEAD)
                        y_vals = ps.pos[alive, 1] if alive.any() else np.array([0.0])
                        n_below = int(np.sum(y_vals < 0))
                        states = np.bincount(ps.state[alive].astype(int), minlength=5)
                        state_str = (f"H={states[4]} R={states[0]} F={states[1]} "
                                     f"C={states[2]} D={states[3]}")
                        print(f"[DEBUG t={t_sim:6.0f}s] Ia={last.anode_current_a:.3f}A  "
                              f"Gap={last.electrode_gap_mm:.1f}mm  "
                              f"Hopper={ps.hopper_count}  Belt={ps.riding_count}  "
                              f"Disp={ps.dispatched_mass_kg:.1f}kg  "
                              f"batch_exh={batch_exh}  ctrl_batch={ctrl_batch}")
                        print(f"         Y: min={y_vals.min():.4f} max={y_vals.max():.4f} "
                              f"below_belt={n_below}  states=[{state_str}]")

                # Track gap peak (get fresh history)
                hist = sim.history
                if hist:
                    current_gap = hist[-1].electrode_gap_mm
                    if current_gap > _gap_peak:
                        _gap_peak = current_gap
                        _gap_peak_t = t_sim

                # Report batch exhaustion once
                if not _batch_exhausted_reported:
                    if hasattr(sim._sim, '_batch_exhausted') and sim._sim._batch_exhausted:
                        _batch_exhausted_reported = True
                        ps = particle_sys
                        print(f"\n[EVENT t={t_sim:.0f}s] BATCH EXHAUSTED - "
                              f"Dispatched={ps.dispatched_mass_kg:.2f}kg  "
                              f"Hopper={ps.hopper_count}  Belt={ps.riding_count}\n")

                # Report gap return start once
                if not _gap_return_reported and _gap_peak > recipe.electrode_gap_mm + 1:
                    if hist:
                        current_gap = hist[-1].electrode_gap_mm
                        if current_gap < _gap_peak - 1:
                            _gap_return_reported = True
                            last = hist[-1]
                            print(f"\n[EVENT t={t_sim:.0f}s] GAP RETURN STARTED - "
                                  f"Peak={_gap_peak:.1f}mm at t={_gap_peak_t:.0f}s  "
                                  f"Current={current_gap:.1f}mm  Ia={last.anode_current_a:.3f}A\n")

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
                         f"T_sensor={last.T_outfeed_sensor_c:.1f} C  |  "
                         f"M_out={last.M_outfeed_wb:.1%}  |  "
                         f"P={last.rf_power_kw:.1f} kW  |  "
                         f"Hopper:{hopper_n}  Belt:{riding_n}")
            else:
                title = f"GP-15 Live  |  t={sim.sim_time:.0f}/{t_end:.0f} s"

            if finished:
                elapsed_w = time.time() - t0_wall
                if hist:
                    title = (f"GP-15 DONE  |  {sim.sim_time:.0f} s  |  "
                             f"M_out={hist[-1].M_outfeed_wb:.1%}  |  "
                             f"wall={elapsed_w:.1f} s")
                else:
                    title = (f"GP-15 DONE  |  {sim.sim_time:.0f} s  |  "
                             f"wall={elapsed_w:.1f} s")

            plotter.add_title(title, font_size=10)

            try:
                plotter.render()
            except Exception:
                break

            elapsed_frame = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, frame_dt - elapsed_frame))

            if finished:
                # ── Run-out wind-down (Manual p.54) ───────────────────
                # Physics simulation is done.  Step particles only (belt
                # transport + free-fall) until all material has exited
                # the GP-15 and landed in the collection bin.
                riding_n = particle_sys.riding_count
                falling_n = int(np.sum(particle_sys.state == particle_sys._STATE_FALLING))
                belt_active = riding_n + falling_n

                if belt_active > 0:
                    v_belt = conv_ctrl.state.belt_speed_m_per_s
                    if v_belt > 0:
                        dt_wind = 0.8 * dx / v_belt
                    else:
                        dt_wind = 1.0
                    for _ in range(200):
                        particle_sys.step(
                            dt_sim=dt_wind,
                            belt_speed_m_per_s=v_belt,
                        )
                        if particle_sys.riding_count == 0:
                            break
                    conv_ctrl.step(dt_wind * 200)

                    # Update visual
                    updated_pos = particle_sys.pos.copy()
                    dead = (particle_sys.state == particle_sys._STATE_DEAD)
                    updated_pos[dead] = [0.0, -100.0, 0.0]
                    particle_cloud.points = updated_pos
                    elapsed_w = time.time() - t0_wall
                    remaining = particle_sys.riding_count
                    plotter.add_title(
                        f"GP-15 Run-out  |  {remaining} on belt  |  "
                        f"wall={elapsed_w:.1f} s", font_size=10)
                    finished = False  # keep looping until belt clear
                else:
                    # Belt fully clear — show final state and stop
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
