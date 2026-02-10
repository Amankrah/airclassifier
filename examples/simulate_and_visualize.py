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
        bed_depth_m=args.bed_depth / 1000.0,
    )

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
          f"(x = {info['rf_zone_x_start_m']:.2f} - {info['rf_zone_x_end_m']:.2f} m)")
    print(f"  Belt width:        {info['belt_width_m'] * 1000:.0f} mm")
    print(f"  Electrode gap:     {args.gap:.0f} mm")
    print(f"  Bed depth:         {args.bed_depth:.0f} mm")
    print(f"  Belt stack:        {belt_stack * 1000:.1f} mm")
    print(f"  Air gap:           {air_gap * 1000:.0f} mm")
    print(f"  Residence time:    {residence_s:.1f} s")
    print(f"  Simulation grid:   {nx} x {ny} x {nz} = {nx * ny * nz:,} cells")
    print(f"  Cell sizes:        dx={dx * 1000:.1f} mm  dy={dy * 1000:.1f} mm  "
          f"dz={dz * 1000:.1f} mm")
    print(f"  Initial moisture:  {args.moisture:.0%} (wet basis)")
    print()

    # ── 5. Run simulation or launch live 3D ─────────────────────────
    if not args.plots_only:
        try:
            import pyvista as pv
            print(f"Running LIVE simulation for {args.duration:.0f} s "
                  f"(belt speed {args.speed} m/min) ...")
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
        print(f"Running simulation for {args.duration:.0f} s "
              f"(belt speed {args.speed} m/min) ...")
        t0 = time.time()
        result = sim.run(duration_s=args.duration, adaptive_dt=True)
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

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"GP-15 Simulation -- Gap {args.gap:.0f} mm  |  "
        f"Bed {args.bed_depth:.0f} mm  |  "
        f"Belt {args.speed} m/min  |  {args.duration:.0f} s",
        fontsize=12, fontweight="bold",
    )

    # Temperature
    ax = axes[0, 0]
    ax.plot(t_arr, ts["T_mean_c"], "r-", linewidth=1.0, alpha=0.5, label="T mean (all)")
    ax.plot(t_arr, ts["T_outfeed_c"], "r-", linewidth=1.5, label="T outfeed")
    ax.plot(t_arr, ts["T_max_c"], "r--", linewidth=1.0, alpha=0.4, label="T max")
    ax.axhline(70, color="orange", linestyle=":", alpha=0.5, label="Denaturation onset")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [\u00b0C]")
    ax.set_title("Material Temperature (\u00a74.2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Moisture
    ax = axes[0, 1]
    ax.plot(t_arr, np.array(ts["M_mean_wb"]) * 100, "b-", linewidth=1.0,
            alpha=0.5, label="M mean (all)")
    ax.plot(t_arr, np.array(ts["M_outfeed_wb"]) * 100, "b-", linewidth=1.5,
            label="M outfeed")
    ax.axhline(material.target_moisture_wb * 100, color="green",
               linestyle="--", alpha=0.5, label=f"Target {material.target_moisture_wb:.0%}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Moisture [% wb]")
    ax.set_title("Moisture Content (\u00a74.3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # RF Power
    ax = axes[1, 0]
    ax.plot(t_arr, ts["rf_power_kw"], "m-", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RF Power [kW]")
    ax.set_title("RF Power Delivered (\u00a74.1)")
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
    ax.set_title("Anode Current (\u00a78.4)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ── 7. Outfeed cross-section (\u00a79.1) ──────────────────────────────
    if outlet.temperature_field is not None and outlet.moisture_field is not None:
        fig2, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11, 4))
        fig2.suptitle(
            "Outfeed Cross-Section (x = L_oven)  --  Pipeline Output to Milling (\u00a79.1)",
            fontsize=11, fontweight="bold",
        )
        im_t = ax_t.imshow(
            outlet.temperature_field, aspect="auto", origin="lower",
            cmap="hot",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        ax_t.set_xlabel("Z -- across belt [mm]")
        ax_t.set_ylabel("Y -- electrode gap [mm]")
        ax_t.set_title(f"Temperature  (avg {outlet.avg_temperature_c:.1f} \u00b0C)")
        plt.colorbar(im_t, ax=ax_t, label="\u00b0C")

        im_m = ax_m.imshow(
            outlet.moisture_field * 100, aspect="auto", origin="lower",
            cmap="Blues",
            extent=[0, info["belt_width_m"] * 1000, 0, args.gap],
        )
        ax_m.set_xlabel("Z -- across belt [mm]")
        ax_m.set_ylabel("Y -- electrode gap [mm]")
        ax_m.set_title(f"Moisture  (avg {outlet.avg_moisture_wb:.1%},  "
                        f"CV {outlet.moisture_uniformity:.3f})")
        plt.colorbar(im_m, ax=ax_m, label="% wb")
        plt.tight_layout()

    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Print results to console
# ─────────────────────────────────────────────────────────────────────

def _print_results(sim, result, elapsed):
    outlet = sim.get_outlet_conditions()
    print()
    print("-" * 60)
    print("  RESULTS")
    print("-" * 60)
    print(f"  Final moisture (mean):     {result.final_moisture_mean_wb:.2%}")
    print(f"  Final temperature (mean):  {result.final_temperature_mean_c:.1f} C")
    print(f"  RF energy consumed:        {result.energy_consumed_kwh:.4f} kWh")
    print(f"  Throughput:                {result.throughput_kg_per_h:.0f} kg/h")
    print()
    print(f"  Outfeed moisture:          {outlet.avg_moisture_wb:.2%}")
    print(f"  Outfeed temperature:       {outlet.avg_temperature_c:.1f} C")
    print(f"  Moisture uniformity (CV):  {outlet.moisture_uniformity:.4f}")
    print(f"  Max temperature:           {outlet.max_temperature_c:.1f} C")
    print(f"  Specific energy:           {outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water")
    print()
    print(f"  Simulation wall-clock:     {elapsed:.2f} s")
    n_steps = len(result.time_series.get("time_s", []))
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

    # ── Adaptive pacing ───────────────────────────────────────────
    v_belt_init = recipe.belt_speed_m_per_min / 60.0
    residence_s = sim.config.oven_length_m / max(v_belt_init, 1e-6)
    transient_sim_s = 2.0 * residence_s
    target_fps = 20.0
    frame_dt = 1.0 / target_fps
    steps_transient = max(1, int(transient_sim_s / (15.0 * target_fps * 0.3)))
    steps_steady = 100

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

            # Adaptive steps: slow during transient, fast after
            t_sim = sim.sim_time
            steps = steps_transient if t_sim < transient_sim_s else steps_steady

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
                title = (f"GP-15 Live  |  t={sim.sim_time:.0f}/{t_end:.0f} s  |  "
                         f"T_out={last.T_outfeed_c:.1f} C  |  "
                         f"M_out={last.M_outfeed_wb:.1%}  |  "
                         f"P={last.rf_power_kw:.1f} kW  |  "
                         f"Belt {conv_ctrl.state.belt_speed_m_per_min:.1f} m/min")
            else:
                title = f"GP-15 Live  |  t={sim.sim_time:.0f}/{t_end:.0f} s"

            if finished:
                elapsed_w = time.time() - t0_wall
                title = (f"GP-15 DONE  |  {sim.sim_time:.0f} s  |  "
                         f"M_out={hist[-1].M_outfeed_wb:.1%}  |  "
                         f"wall={elapsed_w:.1f} s")

            plotter.add_title(title, font_size=10)

            try:
                plotter.render()
            except Exception:
                break

            elapsed_frame = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, frame_dt - elapsed_frame))

            if finished:
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
