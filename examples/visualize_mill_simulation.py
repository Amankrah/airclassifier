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


def _mesh_to_polydata(verts: np.ndarray, tris: np.ndarray) -> "pv.PolyData":
    """Convert vertices/triangles to PyVista PolyData."""
    n = tris.shape[0]
    faces = np.empty((n, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = tris
    return pv.PolyData(verts.copy(), faces.ravel())


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

    # Component opacity settings (X-ray view)
    xray_opacities = {
        "housing": 0.25,      # Very transparent to see inside
        "screen": 0.50,       # Semi-transparent
        "rotor": 0.95,        # Nearly opaque
        "hammers": 0.95,      # Nearly opaque
        "feed_chute": 0.70,   # Mostly visible
        "drive": 0.80,        # Visible
    }

    # Store mesh data for animation
    mesh_actors = {}
    original_verts = {}
    animated_components = {"rotor", "hammers"}

    # ── Add geometry meshes with proper colors ────────────────────────
    print("\nAdding components:")
    for name, (verts, tris, meta) in meshes.items():
        # Get color from COMPONENT_COLORS
        color_rgba = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 1.0))
        rgb = color_rgba[:3]

        # Use xray opacity for better visibility
        opacity = xray_opacities.get(name, color_rgba[3] if len(color_rgba) > 3 else 0.8)

        # Create PyVista mesh
        pd = _mesh_to_polydata(verts, tris)

        # Add to plotter
        actor = plotter.add_mesh(
            pd,
            color=rgb,
            opacity=opacity,
            smooth_shading=True,
            name=name,
            label=name.replace("_", " ").title(),
        )

        mesh_actors[name] = pd
        original_verts[name] = verts.copy()

        print(f"  {name:15} {len(verts):5} verts, {len(tris):5} tris, "
              f"color={rgb}, opacity={opacity:.2f}")

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

    # ── Add axes and legend ───────────────────────────────────────────
    plotter.add_axes(
        xlabel="X (rotor axis)",
        ylabel="Y (vertical)",
        zlabel="Z (lateral)",
        line_width=2,
    )

    legend_bg = (0.1, 0.1, 0.15, 0.8) if args.dark else "white"
    plotter.add_legend(loc="upper left", bcolor=legend_bg)

    # Title
    plotter.add_title("Hammer Mill Live  |  Initializing...", font_size=10)

    # Camera position - closer to the mill for better view
    plotter.camera_position = [
        (0.45, 0.25, 0.45),  # Camera position (closer)
        (0.15, 0.0, 0.0),    # Focal point (center of rotor)
        (0, 1, 0),           # Up vector
    ]
    plotter.camera.zoom(0.8)  # Zoom in (smaller value = closer)

    # ══════════════════════════════════════════════════════════════════
    # Simulation Loop State
    # ══════════════════════════════════════════════════════════════════
    dt = 0.002  # 2ms timestep
    omega = config.rotor_angular_velocity
    t_end = args.duration
    t0_wall = time.time()

    # Tracking
    theta = [0.0]
    step_count = [0]
    total_impacts = [0]
    total_breakage = [0]
    total_discharged = [0]

    # Adaptive pacing: fewer steps early (see transient), more later
    steps_min = 2
    steps_max = 15
    transient_s = 0.5  # Ramp up over 0.5s

    # Show initial view
    plotter.show(interactive_update=True, auto_close=False)

    # ══════════════════════════════════════════════════════════════════
    # Animation/Simulation Loop
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  LIVE SIMULATION RUNNING")
    print("  Close window to stop")
    print("=" * 60 + "\n")

    target_fps = 30.0
    frame_dt = 1.0 / target_fps

    try:
        while True:
            t_frame_start = time.perf_counter()

            # Process UI events
            try:
                plotter.iren.process_events()
            except Exception:
                break

            # Check if simulation finished
            sim_time = sim.engine.time_s
            if sim_time >= t_end:
                # Final update
                elapsed_wall = time.time() - t0_wall
                plotter.add_title(
                    f"Hammer Mill DONE  |  {sim_time:.1f}s  |  "
                    f"Impacts: {total_impacts[0]}  |  "
                    f"Discharged: {total_discharged[0]}  |  "
                    f"Wall: {elapsed_wall:.1f}s",
                    font_size=10,
                )
                plotter.render()
                print("\nSimulation complete!")
                print(f"  Total impacts:     {total_impacts[0]}")
                print(f"  Total breakage:    {total_breakage[0]}")
                print(f"  Total discharged:  {total_discharged[0]}")
                print(f"  Wall-clock time:   {elapsed_wall:.1f}s")

                # Keep window open
                plotter.show()
                break

            # Adaptive stepping
            ramp = min(sim_time / max(transient_s, 0.01), 1.0)
            steps = int(steps_min + ramp * (steps_max - steps_min))

            # Run simulation steps
            for _ in range(steps):
                if sim.engine.time_s >= t_end:
                    break

                state = sim.step(dt)
                step_count[0] += 1
                theta[0] += omega * dt

                # Track statistics
                total_impacts[0] += state.num_impacts
                total_breakage[0] += state.num_breakage_events
                total_discharged[0] += state.num_discharged

            # ── Update rotor/hammer animation ─────────────────────────
            cos_t = np.cos(theta[0])
            sin_t = np.sin(theta[0])

            for name in animated_components:
                if name in mesh_actors:
                    verts = original_verts[name]
                    new_verts = verts.copy()
                    # Rotate around X axis (rotor axis)
                    new_verts[:, 1] = cos_t * verts[:, 1] - sin_t * verts[:, 2]
                    new_verts[:, 2] = sin_t * verts[:, 1] + cos_t * verts[:, 2]
                    mesh_actors[name].points = new_verts

            # ── Update particle positions and sizes ───────────────────
            positions = sim.get_particle_positions()
            sizes = sim.get_particle_sizes()

            if len(positions) > 0:
                particle_cloud.points = positions
                particle_cloud["Size"] = sizes * 1000  # mm
            else:
                # Hide particles if none
                particle_cloud.points = np.array([[0.0, -1.0, 0.0]])
                particle_cloud["Size"] = np.array([0.0])

            # ── Update title with live stats ──────────────────────────
            n_particles = len(positions)
            last_state = sim.history[-1] if sim.history else None

            if last_state:
                title = (
                    f"Hammer Mill Live  |  t={sim_time:.2f}/{t_end:.0f}s  |  "
                    f"Particles: {n_particles}  |  "
                    f"Holdup: {last_state.holdup_kg*1000:.0f}g  |  "
                    f"Power: {last_state.power_kw:.1f}kW  |  "
                    f"Impacts: {total_impacts[0]}"
                )
            else:
                title = f"Hammer Mill Live  |  t={sim_time:.2f}s  |  Particles: {n_particles}"

            plotter.add_title(title, font_size=10)

            # Render
            try:
                plotter.render()
            except Exception:
                break

            # Frame rate control
            elapsed_frame = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, frame_dt - elapsed_frame))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
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
    print(f"  Steps:             {step_count[0]}")
    print(f"  Wall-clock:        {elapsed:.1f}s")
    print(f"  Speed:             {step_count[0]/max(elapsed, 0.001):.0f} steps/s")
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
        # Just show static geometry
        from airclassifier.milling import create_hammer_mill_machine
        config = MillConfig()
        assembly = create_hammer_mill_machine(config=config)
        meshes = assembly.get_component_meshes()

        plotter = pv.Plotter()
        plotter.set_background("white")

        for name, (verts, tris, meta) in meshes.items():
            color_rgba = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 1.0))
            pd = _mesh_to_polydata(verts, tris)
            plotter.add_mesh(pd, color=color_rgba[:3], opacity=color_rgba[3],
                             smooth_shading=True, label=name)

        plotter.add_legend()
        plotter.add_axes()
        plotter.add_title("Hammer Mill - Static View")
        plotter.show()
    else:
        run_live_simulation(args)


if __name__ == "__main__":
    main()
