"""
Live Hammer Mill Simulation Visualization
==========================================

Watch particles flow through the hammer mill in real-time.
Shows rotor spinning, particles being fed, impacted, broken, and discharged.

Similar architecture to the GP-15 pretreatment visualization.

Usage:
    python examples/visualize_mill_simulation.py
    python examples/visualize_mill_simulation.py --duration 10
    python examples/visualize_mill_simulation.py --feed-rate 800
    python examples/visualize_mill_simulation.py --static  # No animation
"""

import argparse
import math
import time
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("PyVista not installed. Install with: pip install pyvista")
    exit(1)

from airclassifier.milling import (
    HammerMillSimulator,
    MillConfig,
    MillRecipe,
    COMPONENT_COLORS,
)


# ── Component styling (matches visualize_hammer_mill.py) ────────────────────
COMPONENT_STYLE = {
    "rotor":              {"color": (0.6, 0.6, 0.65),   "opacity": 0.95, "label": "Rotor (steel)"},
    "hammers":            {"color": (0.85, 0.70, 0.15), "opacity": 1.0,  "label": "Hammers (brass)"},
    "hammer_pins":        {"color": (0.55, 0.55, 0.58), "opacity": 1.0,  "label": "Hammer pins"},
    "screen":             {"color": (0.4, 0.45, 0.5),   "opacity": 0.6,  "label": "Screen"},
    "housing":            {"color": (0.35, 0.45, 0.55), "opacity": 0.3,  "label": "Housing"},
    "feed_chute":         {"color": (0.5, 0.6, 0.65),   "opacity": 0.75, "label": "Feed Chute"},
    "drive_motor":        {"color": (0.32, 0.34, 0.38), "opacity": 0.95, "label": "Motor"},
    "drive_base":         {"color": (0.5, 0.52, 0.55),  "opacity": 0.95, "label": "Base plate"},
    "drive_feet":         {"color": (0.45, 0.47, 0.5),  "opacity": 0.95, "label": "Motor feet"},
    "drive_pulley_motor": {"color": (0.28, 0.30, 0.32), "opacity": 1.0,  "label": "Motor pulley"},
    "drive_pulley_mill":  {"color": (0.62, 0.63, 0.66), "opacity": 1.0,  "label": "Mill pulley"},
    "drive_shaft":        {"color": (0.35, 0.37, 0.4),  "opacity": 1.0,  "label": "Motor shaft"},
    "drive_belt":         {"color": (0.22, 0.22, 0.24), "opacity": 1.0,  "label": "Belt"},
}


def _mesh_to_polydata(verts: np.ndarray, tris: np.ndarray) -> "pv.PolyData":
    """Convert vertices/triangles to PyVista PolyData."""
    n = tris.shape[0]
    faces = np.empty((n, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = tris
    return pv.PolyData(verts.copy(), faces.ravel())


def _build_belt_path(dp):
    """Build belt centerline as a closed polyline in Y-Z plane.

    The path follows the belt loop: mill wrap -> straight -> motor wrap -> straight.
    Returns (path_yz, cum_s, total_len, belt_x).
    """
    groove_frac = 0.12
    R_mill = dp.mill_pulley_radius_m * (1 - groove_frac)
    R_motor = dp.pulley_radius_m * (1 - groove_frac)
    my, mz = dp.motor_y_offset_m, dp.motor_z_offset_m
    d = math.sqrt(my**2 + mz**2)
    phi = math.atan2(mz, my)
    n_arc = 24

    pts = []
    # 1. Mill wrap (far side from motor): phi+pi/2 -> phi+3pi/2
    for i in range(n_arc + 1):
        a = (phi + math.pi / 2) + i * math.pi / n_arc
        pts.append((R_mill * math.cos(a), R_mill * math.sin(a)))
    # 2. Straight to motor -tangent
    a0 = phi - math.pi / 2
    pts.append((my + R_motor * math.cos(a0), mz + R_motor * math.sin(a0)))
    # 3. Motor wrap (far side from mill): phi-pi/2 -> phi+pi/2
    for i in range(1, n_arc + 1):
        a = (phi - math.pi / 2) + i * math.pi / n_arc
        pts.append((my + R_motor * math.cos(a), mz + R_motor * math.sin(a)))
    # 4. Close loop back to start
    pts.append(pts[0])

    path_yz = np.array(pts, dtype=np.float64)
    diffs = np.diff(path_yz, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    cum_s = np.zeros(len(path_yz))
    cum_s[1:] = np.cumsum(seg_lens)
    belt_x = dp.mill_pulley_x_m + dp.mill_pulley_width_m / 2
    return path_yz, cum_s, cum_s[-1], belt_x


def _sample_belt_markers(path_yz, cum_s, total_len, belt_x, phase, n=15):
    """Return (n, 3) marker positions at given phase along belt path."""
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        s = (phase + i * total_len / n) % total_len
        idx = min(int(np.searchsorted(cum_s, s)) - 1, len(path_yz) - 2)
        idx = max(idx, 0)
        seg_len = cum_s[idx + 1] - cum_s[idx]
        t = (s - cum_s[idx]) / max(seg_len, 1e-10)
        out[i, 0] = belt_x
        out[i, 1] = path_yz[idx, 0] + t * (path_yz[idx + 1, 0] - path_yz[idx, 0])
        out[i, 2] = path_yz[idx, 1] + t * (path_yz[idx + 1, 1] - path_yz[idx, 1])
    return out


def run_live_simulation(args):
    """Run simulation with live 3D visualization."""

    print("=" * 60)
    print("  Hammer Mill - Live Simulation")
    print("=" * 60)

    # Configuration - geometry must be consistent:
    # hammer tip radius = rotor_radius + hammer_length < screen_inner_radius
    config = MillConfig(
        rotor_rpm=args.rpm,
        rotor_diameter_m=0.20,  # Hub diameter (radius = 0.10m)
        rotor_length_m=0.30,
        hammer_rows=4,
        hammers_per_row=4,
        hammer_length_m=0.08,  # Tip radius = 0.10 + 0.08 = 0.18m
        screen_inner_radius_m=0.188,  # Just larger than tip
        housing_inner_radius_m=0.20,
        screen_aperture_mm=args.aperture,
    )

    recipe = MillRecipe(
        name="Live Demo",
        rotor_rpm=args.rpm,
        screen_aperture_mm=args.aperture,
        feed_rate_kg_per_hr=args.feed_rate,
        feed_d50_um=3000,  # 3mm feed particles
    )

    # Print configuration
    print(f"\nMill Configuration:")
    print(f"  Rotor RPM:          {config.rotor_rpm}")
    print(f"  Rotor diameter:     {config.rotor_diameter_m * 100:.0f} cm")
    print(f"  Hammer tip radius:  {config.rotor_diameter_m/2 + config.hammer_length_m:.3f} m")
    print(f"  Hammer tip speed:   {config.hammer_tip_speed:.1f} m/s")
    print(f"  Screen aperture:    {config.screen_aperture_mm} mm")
    print(f"  Total hammers:      {config.total_hammers}")
    print(f"\nRecipe:")
    print(f"  Feed rate:          {recipe.feed_rate_kg_per_hr} kg/hr")
    print(f"  Feed size (d50):    {recipe.feed_d50_um} um")
    print(f"  Duration:           {args.duration} s")

    # Create simulator
    print("\nCreating simulator...")
    sim = HammerMillSimulator.create(config=config, recipe=recipe)
    sim.initialize(initial_holdup_kg=args.initial_holdup)

    # Get geometry meshes
    print("Building geometry...")
    meshes = sim.get_geometry_meshes()

    # ══════════════════════════════════════════════════════════════════
    # Build PyVista Plotter
    # ══════════════════════════════════════════════════════════════════
    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("#1a1a2e" if args.dark else "white")

    # Store mesh data for animation
    mesh_actors = {}
    original_verts = {}

    # Rotor-axis components (rotate around Y=0, Z=0 at rotor speed)
    rotor_animated = {"rotor", "hammers", "hammer_pins", "drive_pulley_mill"}

    # Motor-axis components (rotate around motor shaft center, faster speed)
    motor_animated = {"drive_pulley_motor", "drive_shaft"}

    # Drive params for motor animation center and pulley ratio
    dp = sim.assembly.drive_params
    motor_center_y = dp.motor_y_offset_m
    motor_center_z = dp.motor_z_offset_m
    pulley_ratio = dp.mill_pulley_radius_m / dp.pulley_radius_m

    # Belt path for marker animation
    belt_path_yz, belt_cum_s, belt_total_len, belt_x = _build_belt_path(dp)
    belt_pitch_r = dp.mill_pulley_radius_m * (1 - 0.12)  # belt pitch radius
    n_belt_markers = 15

    # ── Add geometry meshes with proper colors ────────────────────────
    print("\nAdding components:")
    for name, (verts, tris, meta) in meshes.items():
        style = COMPONENT_STYLE.get(name, {})
        if style:
            rgb = style["color"]
            opacity = style["opacity"]
            label = style["label"]
        else:
            color_rgba = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 1.0))
            rgb = color_rgba[:3]
            opacity = color_rgba[3] if len(color_rgba) > 3 else 0.8
            label = name.replace("_", " ").title()

        pd = _mesh_to_polydata(verts, tris)
        plotter.add_mesh(
            pd,
            color=rgb,
            opacity=opacity,
            smooth_shading=True,
            name=name,
            label=label,
        )

        mesh_actors[name] = pd
        original_verts[name] = verts.copy()

        print(f"  {name:20} {len(verts):5} verts, {len(tris):5} tris, "
              f"opacity={opacity:.2f}")

    # ── Create particle point cloud ───────────────────────────────────
    # Initialize with positions (may be empty)
    initial_pos = sim.get_particle_positions()
    if len(initial_pos) == 0:
        initial_pos = np.array([[0.15, 0.0, 0.0]])  # Dummy point

    particle_cloud = pv.PolyData(initial_pos)
    initial_sizes = sim.get_particle_sizes()
    if len(initial_sizes) == 0:
        initial_sizes = np.array([0.003])

    # Color particles by size (larger = red, smaller = blue)
    particle_cloud["Size"] = initial_sizes * 1000  # mm

    plotter.add_mesh(
        particle_cloud,
        scalars="Size",
        cmap="coolwarm_r",  # Blue (small) to Red (large)
        clim=[0.0, 3.0],    # 0-3mm range
        point_size=8,
        render_points_as_spheres=True,
        opacity=0.9,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Particle Size [mm]",
            "position_x": 0.82,
            "width": 0.12,
        },
        name="particles",
        label="Particles",
    )

    # ── Belt markers (small flat dots that travel along the belt loop) ──
    belt_marker_pos = _sample_belt_markers(
        belt_path_yz, belt_cum_s, belt_total_len, belt_x, 0.0, n_belt_markers,
    )
    belt_marker_cloud = pv.PolyData(belt_marker_pos)
    plotter.add_mesh(
        belt_marker_cloud,
        color=(0.45, 0.45, 0.40),  # subtle gray-brown (blends with belt)
        point_size=4,
        render_points_as_spheres=False,  # flat squares — distinct from particles
        opacity=0.85,
        name="belt_markers",
    )

    # ── Add axes and legend ───────────────────────────────────────────
    plotter.add_axes(
        xlabel="X (rotor axis)",
        ylabel="Y (vertical)",
        zlabel="Z (lateral)",
        line_width=2,
    )

    legend_bg = (0.1, 0.1, 0.15, 0.8) if args.dark else "white"
    plotter.add_legend(loc="upper left", bcolor=legend_bg)

    # Title — use a vtkTextActor (tuple position) so we can call
    # SetInput() safely each frame without actor churn.
    text_color = "white" if args.dark else "black"
    title_actor = plotter.add_text(
        "Hammer Mill Live  |  Initializing...",
        position="upper_left",
        font_size=10,
        color=text_color,
        name="sim_title",
    )

    # Camera: auto-frame the full assembly (mill + drive), Y-up
    plotter.reset_camera()
    plotter.camera.up = (0, 1, 0)
    plotter.camera.zoom(1.4)

    # ══════════════════════════════════════════════════════════════════
    # Simulation state
    # ══════════════════════════════════════════════════════════════════
    dt = 0.002  # 2ms timestep
    omega = config.rotor_angular_velocity
    t_end = args.duration
    t0_wall = time.time()

    theta = 0.0
    step_count = 0
    total_impacts = 0
    total_breakage = 0
    total_discharged = 0

    # Adaptive pacing: fewer steps early (see transient), more later
    steps_min = 2
    steps_max = 15
    transient_s = 0.5

    # Pre-allocate particle buffer (fixed size → no VTK reallocation)
    max_particles = 2000
    particle_buf = np.zeros((max_particles, 3), dtype=np.float32)
    size_buf = np.zeros(max_particles, dtype=np.float32)
    particle_buf[:, 1] = -10.0  # off-screen
    particle_cloud.points = particle_buf.copy()
    particle_cloud["Size"] = size_buf.copy()

    # Non-blocking show
    plotter.show(interactive_update=True, auto_close=False)

    print("\n" + "=" * 60)
    print("  LIVE SIMULATION RUNNING")
    print("  Close window to stop")
    print("=" * 60 + "\n")

    # ══════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════
    target_fps = 30.0
    frame_dt = 1.0 / target_fps
    sim_done = False

    try:
        while not sim_done:
            t_frame_start = time.perf_counter()

            # Pump VTK events (window close, mouse, etc.)
            try:
                plotter.iren.process_events()
            except Exception:
                break
            if plotter.render_window is None:
                break

            sim_time = sim.engine.time_s

            # ── Adaptive stepping ────────────────────────────────────
            ramp = min(sim_time / max(transient_s, 0.01), 1.0)
            steps = int(steps_min + ramp * (steps_max - steps_min))

            for _ in range(steps):
                if sim.engine.time_s >= t_end:
                    break
                state = sim.step(dt)
                step_count += 1
                theta += omega * dt
                total_impacts += state.num_impacts
                total_breakage += state.num_breakage_events
                total_discharged += state.num_discharged

            # ── Rotate rotor-axis components (rotor, hammers, pins, mill pulley) ──
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            for name in rotor_animated:
                if name in mesh_actors:
                    ov = original_verts[name]
                    nv = ov.copy()
                    nv[:, 1] = cos_t * ov[:, 1] - sin_t * ov[:, 2]
                    nv[:, 2] = sin_t * ov[:, 1] + cos_t * ov[:, 2]
                    mesh_actors[name].points = nv

            # ── Rotate motor-axis components (motor pulley, shaft) ─────
            motor_theta = theta * pulley_ratio
            cos_m = np.cos(motor_theta)
            sin_m = np.sin(motor_theta)
            for name in motor_animated:
                if name in mesh_actors:
                    ov = original_verts[name]
                    nv = ov.copy()
                    y_c = ov[:, 1] - motor_center_y
                    z_c = ov[:, 2] - motor_center_z
                    nv[:, 1] = cos_m * y_c - sin_m * z_c + motor_center_y
                    nv[:, 2] = sin_m * y_c + cos_m * z_c + motor_center_z
                    mesh_actors[name].points = nv

            # ── Advance belt markers along the belt path ──────────────
            belt_phase = theta * belt_pitch_r
            belt_marker_cloud.points = _sample_belt_markers(
                belt_path_yz, belt_cum_s, belt_total_len, belt_x,
                belt_phase, n_belt_markers,
            )

            # ── Update particles (fixed-size buffer) ─────────────────
            positions = sim.get_particle_positions()
            sizes = sim.get_particle_sizes()
            n = min(len(positions), max_particles)

            particle_buf[:] = 0.0
            particle_buf[:, 1] = -10.0
            size_buf[:] = 0.0
            if n > 0:
                particle_buf[:n] = positions[:n]
                size_buf[:n] = sizes[:n] * 1000
            particle_cloud.points = particle_buf.copy()
            particle_cloud["Size"] = size_buf.copy()

            # ── Update title (in-place, no actor churn) ──────────────
            sim_time = sim.engine.time_s  # re-read after stepping
            n_particles = len(positions)
            last = sim.history[-1] if sim.history else None
            if last:
                title_actor.SetText(
                    2,
                    f"t={sim_time:.2f}/{t_end:.0f}s  |  "
                    f"Particles: {n_particles}  |  "
                    f"Holdup: {last.holdup_kg*1000:.0f}g  |  "
                    f"Power: {last.power_kw:.1f}kW  |  "
                    f"Impacts: {total_impacts}",
                )
            else:
                title_actor.SetText(
                    2, f"t={sim_time:.2f}s  |  Particles: {n_particles}"
                )

            # ── Render frame ─────────────────────────────────────────
            try:
                plotter.render()
            except Exception:
                break

            # ── Check if simulation finished ─────────────────────────
            if sim.engine.time_s >= t_end:
                sim_done = True
                elapsed_wall = time.time() - t0_wall

                # Clear particles and belt markers from view
                particle_buf[:] = 0.0
                particle_buf[:, 1] = -10.0
                size_buf[:] = 0.0
                particle_cloud.points = particle_buf.copy()
                particle_cloud["Size"] = size_buf.copy()
                offscreen = np.zeros((n_belt_markers, 3), dtype=np.float32)
                offscreen[:, 1] = -10.0
                belt_marker_cloud.points = offscreen

                title_actor.SetText(
                    2,
                    f"DONE  |  {sim_time:.1f}s  |  "
                    f"Impacts: {total_impacts}  |  "
                    f"Discharged: {total_discharged}  |  "
                    f"Wall: {elapsed_wall:.1f}s",
                )
                plotter.render()
                print("\nSimulation complete!")
                print(f"  Total impacts:     {total_impacts}")
                print(f"  Total breakage:    {total_breakage}")
                print(f"  Total discharged:  {total_discharged}")
                print(f"  Wall-clock time:   {elapsed_wall:.1f}s")

            # ── Frame-rate limiter ───────────────────────────────────
            elapsed_frame = time.perf_counter() - t_frame_start
            time.sleep(max(0.001, frame_dt - elapsed_frame))

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Keep window open after sim finishes — pump events until closed
    if sim_done and plotter.render_window is not None:
        print("Close the window to see final results.")
        try:
            plotter.show()
        except Exception:
            pass

    try:
        plotter.close()
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════════
    # Final Results
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0_wall
    result = sim.build_result_from_engine(duration_s=sim.engine.time_s, dt=dt)

    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nProduct PSD:")
    print(f"  d10:  {result.d10_um:.0f} um")
    print(f"  d50:  {result.d50_um:.0f} um")
    print(f"  d90:  {result.d90_um:.0f} um")
    print(f"\nProcess KPIs:")
    print(f"  Mean power:        {result.mean_power_kw:.1f} kW")
    print(f"  Throughput:        {result.throughput_kg_per_hr:.0f} kg/hr")
    if result.specific_energy_kwh_per_t > 0:
        print(f"  Specific energy:   {result.specific_energy_kwh_per_t:.1f} kWh/t")
    print(f"\nSimulation:")
    print(f"  Duration:          {sim.engine.time_s:.2f}s")
    print(f"  Steps:             {step_count}")
    print(f"  Wall-clock:        {elapsed:.1f}s")
    print(f"  Speed:             {step_count/max(elapsed, 0.001):.0f} steps/s")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Live hammer mill simulation visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds (default: 5)")
    parser.add_argument("--feed-rate", type=float, default=800,
                        help="Feed rate in kg/hr (default: 800)")
    parser.add_argument("--rpm", type=float, default=3000,
                        help="Rotor RPM (default: 3000)")
    parser.add_argument("--aperture", type=float, default=1.5,
                        help="Screen aperture in mm (default: 1.5)")
    parser.add_argument("--initial-holdup", type=float, default=0.05,
                        help="Initial material in mill in kg (default: 0.05)")
    parser.add_argument("--dark", action="store_true",
                        help="Dark theme for viewer")
    parser.add_argument("--static", action="store_true",
                        help="Static view only (no simulation)")
    args = parser.parse_args()

    if args.static:
        # Just show static geometry with same styling
        from airclassifier.milling import create_hammer_mill_machine
        config = MillConfig()
        assembly = create_hammer_mill_machine(config=config)
        meshes = assembly.get_component_meshes()

        plotter = pv.Plotter(window_size=(1400, 900))
        plotter.set_background("#1a1a2e" if args.dark else "white")

        for name, (verts, tris, meta) in meshes.items():
            style = COMPONENT_STYLE.get(name, {})
            if style:
                rgb, opacity, label = style["color"], style["opacity"], style["label"]
            else:
                c = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 1.0))
                rgb, opacity, label = c[:3], (c[3] if len(c) > 3 else 0.8), name
            pd = _mesh_to_polydata(verts, tris)
            plotter.add_mesh(pd, color=rgb, opacity=opacity,
                             smooth_shading=True, label=label)

        text_color = "white" if args.dark else "black"
        plotter.add_legend(bcolor=(0.1, 0.1, 0.15, 0.8) if args.dark else "white")
        plotter.add_axes(xlabel="X (rotor axis)", ylabel="Y (vertical)",
                         zlabel="Z (lateral)", line_width=2)
        plotter.add_text("Hammer Mill - Static View", position="upper_left",
                         font_size=12, color=text_color)
        plotter.reset_camera()
        plotter.camera.up = (0, 1, 0)
        plotter.camera.zoom(1.4)
        plotter.show()
    else:
        run_live_simulation(args)


if __name__ == "__main__":
    main()
