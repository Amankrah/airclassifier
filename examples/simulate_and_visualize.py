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

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run GP-15 RF heating simulation and visualize results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python examples/simulate_and_visualize.py
    python examples/simulate_and_visualize.py --duration 120
    python examples/simulate_and_visualize.py --gap 100 --speed 0.3
    python examples/simulate_and_visualize.py --plots-only
    python examples/simulate_and_visualize.py --gap 150 --bed-depth 60
""",
    )
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds (default 60)")
    parser.add_argument("--gap", type=float, default=80.0,
                        help="Electrode gap in mm (default 80)")
    parser.add_argument("--bed-depth", type=float, default=50.0,
                        help="Material bed depth in mm (default 50)")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Belt speed in m/min (default 0.5)")
    parser.add_argument("--moisture", type=float, default=0.10,
                        help="Initial moisture wet basis fraction (default 0.10)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip 3D PyVista view, show only matplotlib plots")
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
        bed_depth_m=args.bed_depth / 1000.0,
    )

    # ── 2. Create simulator (builds machine assembly + physics) ──────
    print("=" * 60)
    print("  GP-15 RF Dielectric Heating — Simulation")
    print("=" * 60)
    print()
    print("Creating GP-15 simulator ...")
    print("  Architecture: GP15Simulator → GP15MachineAssembly")
    print("                             → CoupledSimulator (9-step loop)")

    sim = GP15Simulator(
        config=config,
        material=material,
        device="cpu",
        use_tvd=True,
        enable_controller=True,
        enable_corrections=False,  # skip corrections for speed
    )

    # ── 3. Load recipe ───────────────────────────────────────────────
    recipe = Recipe(
        name="example_run",
        recipe_number=1,
        electrode_gap_mm=args.gap,
        belt_speed_m_per_min=args.speed,
        extraction_fan_hz=35.0,
        heater_bank_1_on=True,
        heater_bank_2_on=True,
    )
    sim.load_recipe(recipe)

    # ── 4. Print assembly info ───────────────────────────────────────
    info = sim.assembly.get_assembly_info()
    nx, ny, nz = sim.grid_shape
    dx, dy, dz = sim.cell_sizes
    belt_stack = config.belt_stack_thickness_m
    air_gap = max(0, args.gap / 1000.0 - args.bed_depth / 1000.0 - belt_stack)
    residence_s = config.oven_length_m / (args.speed / 60.0) if args.speed > 0 else 0

    print()
    print(f"  Machine:           {info['machine']}")
    print(f"  RF zone:           {info['rf_zone_length_m']:.2f} m  "
          f"(x = {info['rf_zone_x_start_m']:.2f} – {info['rf_zone_x_end_m']:.2f} m)")
    print(f"  Belt width:        {info['belt_width_m'] * 1000:.0f} mm")
    print(f"  Electrode gap:     {args.gap:.0f} mm")
    print(f"  Bed depth:         {args.bed_depth:.0f} mm")
    print(f"  Belt stack:        {belt_stack * 1000:.1f} mm")
    print(f"  Air gap:           {air_gap * 1000:.0f} mm")
    print(f"  Residence time:    {residence_s:.1f} s")
    print(f"  Simulation grid:   {nx} × {ny} × {nz} = {nx * ny * nz:,} cells")
    print(f"  Cell sizes:        dx={dx * 1000:.1f} mm  dy={dy * 1000:.1f} mm  "
          f"dz={dz * 1000:.1f} mm")
    print(f"  Initial moisture:  {args.moisture:.0%} (wet basis)")
    print()

    # ── 5. Run simulation ────────────────────────────────────────────
    print(f"Running simulation for {args.duration:.0f} s "
          f"(belt speed {args.speed} m/min) ...")
    t0 = time.time()
    result = sim.run(duration_s=args.duration, adaptive_dt=True)
    elapsed = time.time() - t0

    # ── 6. Get outlet conditions (§9.1) ──────────────────────────────
    outlet = sim.get_outlet_conditions()

    # ── 7. Print results ─────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  RESULTS")
    print("─" * 60)
    print(f"  Final moisture (mean):     {result.final_moisture_mean_wb:.2%}")
    print(f"  Final temperature (mean):  {result.final_temperature_mean_c:.1f} °C")
    print(f"  RF energy consumed:        {result.energy_consumed_kwh:.4f} kWh")
    print(f"  Throughput:                {result.throughput_kg_per_h:.0f} kg/h")
    print()
    print(f"  Outfeed moisture:          {outlet.avg_moisture_wb:.2%}")
    print(f"  Outfeed temperature:       {outlet.avg_temperature_c:.1f} °C")
    print(f"  Moisture uniformity (CV):  {outlet.moisture_uniformity:.4f}")
    print(f"  Max temperature:           {outlet.max_temperature_c:.1f} °C")
    print(f"  Specific energy:           {outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water")
    print()
    print(f"  Simulation wall-clock:     {elapsed:.2f} s")
    n_steps = len(result.time_series.get("time_s", []))
    if n_steps > 0:
        print(f"  Timesteps completed:       {n_steps}")
        print(f"  Speed:                     {n_steps / max(elapsed, 0.001):.0f} steps/s")
    print("─" * 60)
    print()

    # ── 8. Matplotlib plots ──────────────────────────────────────────
    import matplotlib.pyplot as plt

    ts = result.time_series
    if not ts.get("time_s"):
        print("No time-series data — skipping plots.")
        return

    t_arr = np.array(ts["time_s"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"GP-15 Simulation — Gap {args.gap:.0f} mm  |  "
        f"Bed {args.bed_depth:.0f} mm  |  "
        f"Belt {args.speed} m/min  |  {args.duration:.0f} s",
        fontsize=12, fontweight="bold",
    )

    # Temperature
    ax = axes[0, 0]
    ax.plot(t_arr, ts["T_mean_c"], "r-", linewidth=1.5, label="T mean")
    ax.plot(t_arr, ts["T_max_c"], "r--", linewidth=1.0, alpha=0.6, label="T max")
    ax.axhline(70, color="orange", linestyle=":", alpha=0.5, label="Denaturation onset")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Material Temperature (§4.2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Moisture
    ax = axes[0, 1]
    ax.plot(t_arr, np.array(ts["M_mean_wb"]) * 100, "b-", linewidth=1.5, label="M mean")
    ax.axhline(material.target_moisture_wb * 100, color="green",
               linestyle="--", alpha=0.5, label=f"Target {material.target_moisture_wb:.0%}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Moisture [% wb]")
    ax.set_title("Moisture Content (§4.3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RF Power
    ax = axes[1, 0]
    ax.plot(t_arr, ts["rf_power_kw"], "m-", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RF Power [kW]")
    ax.set_title("RF Power Delivered (§4.1)")
    ax.grid(True, alpha=0.3)

    # Anode current
    ax = axes[1, 1]
    ax.plot(t_arr, ts["anode_current_a"], "k-", linewidth=1.5, label="Ia")
    ax.axhline(recipe.mrh_amps, color="red", linestyle="--", alpha=0.5,
               label=f"MRH = {recipe.mrh_amps} A")
    ax.axhline(recipe.mrl_amps, color="orange", linestyle="--", alpha=0.5,
               label=f"MRL = {recipe.mrl_amps} A")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Anode Current [A]")
    ax.set_title("Anode Current (§8.4)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── 9. Outfeed cross-section (§9.1) ──────────────────────────────
    if outlet.temperature_field is not None and outlet.moisture_field is not None:
        fig2, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4))
        fig2.suptitle(
            "Outfeed Cross-Section (x = L_oven)  —  Pipeline Output to Milling (§9.1)",
            fontsize=11, fontweight="bold",
        )

        # Temperature Y-Z cross-section
        im_t = ax_t.imshow(
            outlet.temperature_field,
            aspect="auto",
            origin="lower",
            cmap="hot",
            extent=[0, info["belt_width_m"] * 1000,
                    0, args.gap],
        )
        ax_t.set_xlabel("Z — across belt [mm]")
        ax_t.set_ylabel("Y — electrode gap [mm]")
        ax_t.set_title(f"Temperature  (avg {outlet.avg_temperature_c:.1f} °C)")
        plt.colorbar(im_t, ax=ax_t, label="°C")

        # Moisture Y-Z cross-section
        im_m = ax_m.imshow(
            outlet.moisture_field * 100,
            aspect="auto",
            origin="lower",
            cmap="Blues",
            extent=[0, info["belt_width_m"] * 1000,
                    0, args.gap],
        )
        ax_m.set_xlabel("Z — across belt [mm]")
        ax_m.set_ylabel("Y — electrode gap [mm]")
        ax_m.set_title(f"Moisture  (avg {outlet.avg_moisture_wb:.1%},  "
                        f"CV {outlet.moisture_uniformity:.3f})")
        plt.colorbar(im_m, ax=ax_m, label="% wb")

        plt.tight_layout()

    # ── 10. 3D visualization (optional) ──────────────────────────────
    if not args.plots_only:
        try:
            import pyvista as pv
            _show_3d(sim, recipe, args, info)
        except ImportError:
            print("PyVista not installed — skipping 3D visualization.")
            print("Install with: pip install pyvista")
        except Exception as e:
            print(f"3D visualization failed: {e}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  3D Visualization (PyVista)
# ─────────────────────────────────────────────────────────────────────

def _show_3d(sim, recipe, args, info):
    """Render the machine assembly with simulation field overlay."""
    import pyvista as pv
    from airclassifier.pretreatment.geometry.assembly import COMPONENT_COLORS

    meshes = sim.get_mesh()

    # ── Plotter ──────────────────────────────────────────────────
    plotter = pv.Plotter(shape=(1, 1))
    bg = "#1a1a2e" if args.dark else "white"
    plotter.set_background(bg)
    plotter.camera.up = (0, 1, 0)

    # X-ray opacities so the field inside is visible
    xray_opacities = {
        "conveyor_frame": 0.08,
        "oven_chamber": 0.06,
        "rollers": 0.60,
        "belt": 0.40,
        "upper_electrode": 0.70,
        "lower_electrode": 0.60,
        "material_bed": 0.0,   # hide static bed — replaced by field
        "infeed_hopper": 0.70,
        "infeed_tunnel": 0.20,
        "outfeed_tunnel": 0.20,
        "collection_bin": 0.55,
        "emu_housing": 0.10,
        "generator": 0.20,
        "rf_feed": 0.80,
    }

    # ── Add machine components ───────────────────────────────────
    for name, item in meshes.items():
        if name == "fields":
            continue  # handle separately

        v, t, meta = item
        style = COMPONENT_COLORS.get(name, {})
        color = style.get("color", "#888888")
        label = style.get("label", name)
        opacity = xray_opacities.get(name, style.get("opacity", 0.8))

        if opacity < 0.01:
            continue  # skip invisible components

        n_faces = t.shape[0]
        faces = np.empty((n_faces, 4), dtype=np.int64)
        faces[:, 0] = 3
        faces[:, 1:] = t
        pd = pv.PolyData(v.copy(), faces.ravel())

        plotter.add_mesh(
            pd,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            label=label,
        )

    # ── Add simulation temperature field as coloured volume ──────
    fields = meshes.get("fields")
    if fields is not None:
        T_field = fields["temperature"]
        nx, ny, nz = fields["grid_shape"]
        dx, dy, dz = fields["cell_sizes"]

        # World-coordinate origin of the simulation domain
        x0, y0, z0 = sim.get_field_world_origin()

        # Build rectilinear grid in world coordinates
        x_coords = np.linspace(x0, x0 + nx * dx, nx + 1)
        y_coords = np.linspace(y0, y0 + ny * dy, ny + 1)
        z_coords = np.linspace(z0, z0 + nz * dz, nz + 1)

        grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)

        # Temperature as cell data (nx × ny × nz in Fortran order for VTK)
        T_flat = T_field.flatten(order="F")
        grid.cell_data["Temperature [°C]"] = T_flat

        # Only show material cells (non-ambient temperature)
        T_ambient = 22.0
        threshold = grid.threshold(
            value=T_ambient + 0.5,
            scalars="Temperature [°C]",
        )
        if threshold.n_cells > 0:
            plotter.add_mesh(
                threshold,
                scalars="Temperature [°C]",
                cmap="hot",
                opacity=0.85,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Temperature [°C]",
                    "position_x": 0.82,
                    "width": 0.12,
                },
                label="Temperature Field",
            )

    # ── Legend & title ────────────────────────────────────────────
    legend_bg = (0.1, 0.1, 0.15, 0.8) if args.dark else "white"
    plotter.add_legend(loc="upper left", bcolor=legend_bg)
    plotter.add_title(
        f"GP-15 Simulation  —  "
        f"Gap {args.gap:.0f} mm  |  Bed {args.bed_depth:.0f} mm  |  "
        f"{args.duration:.0f} s  |  "
        f"M: {sim.get_outlet_conditions().avg_moisture_wb:.1%}",
        font_size=10,
    )
    plotter.add_axes()

    # ── Camera ───────────────────────────────────────────────────
    plotter.reset_camera()
    plotter.camera.azimuth = -55
    plotter.camera.elevation = 18
    plotter.camera.zoom(1.1)

    plotter.show()


if __name__ == "__main__":
    main()
