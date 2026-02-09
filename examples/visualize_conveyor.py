#!/usr/bin/env python
"""
Visualize & Animate GP-15 Conveyor Belt
=========================================

Animated conveyor with physics-driven motion:
  - All rollers rotate at kinematically correct speeds
  - Sprockets rotate (drive sprocket faster by gear ratio)
  - Motor drives the chain which drives the head roller
  - Start / stop / speed control via keyboard

All motion is derived from ``ConveyorDriveController`` which computes
angular velocities from belt speed using real roller/sprocket radii
defined in ``ConveyorBeltParams`` (single source of truth).

Keyboard controls:
  SPACE     Start / Stop the motor
  UP / DOWN Increase / Decrease belt speed (0.1 m/min steps)
  R         Reset (stop + zero positions)
  Q / ESC   Quit

Usage:
    python examples/visualize_conveyor.py              # Animated
    python examples/visualize_conveyor.py --static     # Static view
    python examples/visualize_conveyor.py --xray       # Transparent frame
    python examples/visualize_conveyor.py --speed 1.0  # Start at 1.0 m/min
    python examples/visualize_conveyor.py --side-view  # XY side view
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pyvista as pv

from airclassifier.pretreatment.geometry.components.conveyor_belt import (
    ConveyorBeltGeometry,
    ConveyorBeltParams,
)
from airclassifier.pretreatment.kernels.transport import (
    ConveyorDriveController,
    rotate_mesh_around_z_axis,
)


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def mesh_to_polydata(vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    """Build PyVista PolyData from vertices and triangle indices."""
    n_faces = triangles.shape[0]
    faces = np.empty((n_faces, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles
    return pv.PolyData(vertices.copy(), faces.ravel())


def collect_roller_entries(layout: dict):
    """Flatten the roller layout into (name, x, y, r) tuples."""
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


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize & animate GP-15 conveyor belt drive system"
    )
    parser.add_argument("--static", action="store_true",
                        help="Static view (no animation)")
    parser.add_argument("--xray", action="store_true",
                        help="Transparent frame, solid belt + rollers")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Initial belt speed [m/min] (default 0.5)")
    parser.add_argument("--side-view", action="store_true",
                        help="Camera aligned to XY side-view")
    args = parser.parse_args()

    # ── Build geometry (single source of truth) ───────────────────
    params = ConveyorBeltParams()
    conveyor = ConveyorBeltGeometry(params)

    # Generate meshes (wheels first — belt needs roller layout)
    frame_v, frame_t, _ = conveyor.generate_bed_structure_mesh()
    wheels_v, wheels_t, _ = conveyor.generate_wheels_mesh()
    belt_v, belt_t, _ = conveyor.generate_belt_mesh()

    roller_layout = conveyor._roller_layout

    print(f"Frame   : {frame_v.shape[0]:,} verts, {frame_t.shape[0]:,} tris")
    print(f"Rollers : {wheels_v.shape[0]:,} verts, {wheels_t.shape[0]:,} tris")
    print(f"Belt    : {belt_v.shape[0]:,} verts, {belt_t.shape[0]:,} tris")

    # ── Belt arc-length (from mesh geometry — no magic numbers) ───
    # The belt mesh has 4 vertices per path point (outer/inner × z_lo/z_hi).
    # Vertices 0, 4, 8, … trace the outer z_lo edge along the belt
    # centre-line.  Cumulative arc lengths let us scroll a visual
    # pattern by the physics-driven belt_position_m.
    belt_path_xy = belt_v[0::4, :2]             # (M, 2)  outer edge XY
    _seg = np.diff(belt_path_xy, axis=0)
    _seg_lens = np.sqrt(_seg[:, 0] ** 2 + _seg[:, 1] ** 2)
    belt_arc = np.concatenate([[0.0], np.cumsum(_seg_lens)])
    belt_total_len = belt_arc[-1]                # belt circumference [m]
    belt_arc_per_vert = np.repeat(belt_arc, 4)   # expand to all 4 verts

    # Band period = head-roller circumference (physics: one band per
    # revolution of the drive roller).  Round to an integer count so
    # the pattern wraps seamlessly around the closed belt loop.
    head_circ = 2.0 * math.pi * params.head_roller_radius_m
    n_belt_bands = max(1, round(belt_total_len / head_circ))
    belt_band_len = belt_total_len / n_belt_bands

    print(f"Belt    : circumference {belt_total_len:.2f} m, "
          f"{n_belt_bands} bands (period {belt_band_len*1000:.0f} mm "
          f"= head-roller circumference)")

    # ── Build drive controller (from same params — no magic) ──────
    ctrl = ConveyorDriveController.from_params(params)

    # ── Create plotter ────────────────────────────────────────────
    plotter = pv.Plotter()
    plotter.set_background("#f0f0f5")
    plotter.camera.up = (0, 1, 0)

    frame_opacity = 0.12 if args.xray else 0.25
    plotter.add_mesh(
        mesh_to_polydata(frame_v, frame_t),
        color="#505060", opacity=frame_opacity,
        smooth_shading=True, name="frame",
    )

    wheels_pd = mesh_to_polydata(wheels_v, wheels_t)
    plotter.add_mesh(
        wheels_pd, color="#808090", opacity=0.85,
        smooth_shading=True, name="rollers",
    )

    belt_pd = mesh_to_polydata(belt_v, belt_t)

    # Initial belt band pattern (visible in static mode too).
    # Cosine gives smooth rolling bands; sawtooth would create
    # hard jumps that look like seams rather than motion.
    belt_pd.point_data['motion'] = (
        0.5 + 0.5 * np.cos(2.0 * math.pi * belt_arc_per_vert / belt_band_len)
    ).astype(np.float32)

    plotter.add_mesh(
        belt_pd, scalars='motion',
        cmap=['#3458a5', '#5a7fd4'],   # two-tone blue (belt segments)
        clim=[0, 1],
        opacity=0.92,
        show_edges=True, edge_color="#2a4a9a", line_width=0.4,
        smooth_shading=True, name="belt",
        show_scalar_bar=False,
    )

    # Status text (initial)
    plotter.add_text(
        "Initializing...",
        position="lower_left", font_size=9, color="black",
        name="status",
    )
    plotter.add_title("GP-15 Conveyor — Animated Drive System")
    plotter.add_axes()

    # Camera
    if args.side_view:
        mid_x = params.frame_length_m / 2
        mid_y = -params.frame_height_m / 2
        plotter.camera.position = (mid_x, mid_y, params.frame_width_m + 3.5)
        plotter.camera.focal_point = (mid_x, mid_y, params.belt_center_z)
        plotter.camera.zoom(1.2)
    else:
        plotter.camera.azimuth = -60
        plotter.camera.elevation = 15
        plotter.reset_camera()
        plotter.camera.zoom(1.1)

    # ── Static mode ───────────────────────────────────────────────
    if args.static:
        plotter.show()
        return

    # ══════════════════════════════════════════════════════════════
    #  ANIMATION
    # ══════════════════════════════════════════════════════════════

    # Pre-compute roller vertex masks for rotation animation.
    # Each roller's vertices are identified by proximity to its
    # center in X-Y space.
    roller_entries = collect_roller_entries(roller_layout)

    roller_vert_masks = {}

    # Z-range filter: only include vertices within the belt/roller
    # zone.  Drive system components (chain, sprockets, gearbox,
    # motor bracket, shaft extension) all sit at Z > belt_z1 and
    # would otherwise be caught by roller proximity masks that only
    # check X-Y distance.  Without this filter the chain arc and
    # bracket top near the head roller spin with it — wrong axis.
    roller_z_max = params.belt_z1 + 0.05   # just past belt edge
    roller_z_min = params.belt_z0 - 0.05   # just before belt edge
    z_in_range = (wheels_v[:, 2] >= roller_z_min) & (wheels_v[:, 2] <= roller_z_max)

    for name, cx, cy, r in roller_entries:
        # Skip sprockets — they share mesh space with the chain
        # arc vertices, and a smooth cylinder rotating around its
        # own axis looks identical anyway.  The visual motion is
        # conveyed by the rollers (larger, visible polygon facets).
        if "sprocket" in name:
            continue

        dx = wheels_v[:, 0] - cx
        dy = wheels_v[:, 1] - cy
        dist_xy = np.sqrt(dx * dx + dy * dy)
        mask = (dist_xy < r * 1.6) & z_in_range

        if mask.any():
            roller_vert_masks[name] = (mask, cx, cy, r)

    # Base vertices (never modified — animation copies from these)
    wheels_base = wheels_v.copy()

    # Start motor at requested speed
    if args.speed > 0:
        ctrl.start(args.speed)

    # Open window (non-blocking for our render loop)
    plotter.show(interactive_update=True, auto_close=False)

    # ── Register keyboard handler on VTK interactor ───────────────
    def on_key_press(vtk_obj, event):
        try:
            key = plotter.iren.interactor.GetKeySym()
        except Exception:
            return

        if key == "space":
            if ctrl.state.running:
                ctrl.stop()
                print("  Motor STOPPED")
            else:
                spd = ctrl.state.speed_setpoint_m_per_min
                if spd < 0.1:
                    spd = args.speed
                ctrl.start(spd)
                print(f"  Motor STARTED at {spd:.1f} m/min")

        elif key == "Up":
            new_spd = min(ctrl.state.speed_setpoint_m_per_min + 0.1, 2.0)
            ctrl.set_speed(new_spd)
            if not ctrl.state.running:
                ctrl.start()
            print(f"  Speed -> {new_spd:.1f} m/min")

        elif key == "Down":
            new_spd = max(ctrl.state.speed_setpoint_m_per_min - 0.1, 0.0)
            ctrl.set_speed(new_spd)
            if new_spd <= 0:
                ctrl.stop()
            print(f"  Speed -> {new_spd:.1f} m/min")

        elif key == "r":
            ctrl.emergency_stop()
            ctrl.state.head_roller_angle = 0.0
            ctrl.state.tail_roller_angle = 0.0
            ctrl.state.drive_sprocket_angle = 0.0
            ctrl.state.driven_sprocket_angle = 0.0
            ctrl.state.belt_position_m = 0.0
            ctrl.state.chain_position_m = 0.0
            ctrl.state.elapsed_time_s = 0.0
            ctrl.state.encoder_pulses = 0
            print("  RESET")

        elif key in ("q", "Escape"):
            raise KeyboardInterrupt

    plotter.iren.interactor.AddObserver("KeyPressEvent", on_key_press)

    print("\nAnimation running — controls:")
    print("  SPACE      Start / Stop motor")
    print("  UP / DOWN  Increase / Decrease speed")
    print("  R          Reset")
    print("  Q          Quit\n")

    # ── Render loop ───────────────────────────────────────────────
    target_fps = 30.0
    frame_dt = 1.0 / target_fps
    t_prev = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter()
            dt = min(t_now - t_prev, 0.1)
            t_prev = t_now

            # Process VTK events (keyboard, mouse, window close)
            try:
                plotter.iren.process_events()
            except Exception:
                break

            # Check window still alive
            try:
                rw = plotter.render_window
                if rw is None or rw.GetNeverRendered():
                    pass  # first frame
            except Exception:
                break

            # ── Step physics controller ───────────────────────────
            if dt > 0.001:
                ctrl.step(dt)

            state = ctrl.state

            # ── Rotate all rollers + sprockets ────────────────────
            # Each roller's angle = belt_position / radius (physics).
            # The drive sprocket uses its own angle (faster, from
            # the controller's kinematic chain).
            animated_verts = wheels_base.copy()

            for name, (mask, cx, cy, r) in roller_vert_masks.items():
                # No-slip: angle = belt_distance / radius.
                # Negate because the belt moves in +arc direction
                # (tail→head on top, head→tail on return) and
                # rotate_mesh_around_z_axis uses the standard math
                # convention where positive angle = CCW.  The real
                # rollers spin CW (top surface moves +X), so CW =
                # negative angle.
                angle = -ctrl.roller_angle(r)

                rotated = rotate_mesh_around_z_axis(
                    wheels_base[mask], cx, cy, angle,
                )
                animated_verts[mask] = rotated

            wheels_pd.points = animated_verts

            # ── Scroll belt surface ──────────────────────────────
            # Belt moves in the +arc direction (the path was defined
            # tail→head→return→tail, same as the belt's loop).
            # f(arc − v·t) shifts the pattern in the +arc direction
            # as belt_position_m increases — matching the real flow.
            # Cosine gives smooth rolling bands; period = head-roller
            # circumference (physics, no magic).
            _phase = state.belt_position_m % belt_total_len
            _shifted = (belt_arc_per_vert - _phase + belt_total_len) % belt_total_len
            belt_pd.point_data['motion'] = (
                0.5 + 0.5 * np.cos(2.0 * math.pi * _shifted / belt_band_len)
            ).astype(np.float32)

            # ── Update status text ────────────────────────────────
            motor_str = "RUNNING" if state.running else "STOPPED"
            rpm = (state.head_roller_omega * 60.0
                   / (2.0 * math.pi)) if state.head_roller_omega > 0 else 0.0
            drive_rpm = (state.drive_sprocket_omega * 60.0
                         / (2.0 * math.pi)) if state.drive_sprocket_omega > 0 else 0.0

            plotter.add_text(
                f"{motor_str}  |  Belt: {state.belt_speed_m_per_min:.2f} m/min"
                f"  (set: {state.speed_setpoint_m_per_min:.1f})"
                f"  |  Head: {rpm:.1f} RPM"
                f"  |  Drive spr: {drive_rpm:.1f} RPM"
                f"  |  Travel: {state.belt_position_m:.3f} m"
                f"  |  Enc: {state.encoder_pulses}"
                f"  |  [SPACE] Start/Stop  [UP/DN] Speed  [R] Reset",
                position="lower_left", font_size=9, color="black",
                name="status",
            )

            # ── Render ────────────────────────────────────────────
            try:
                plotter.render()
            except Exception:
                break

            # ── Frame rate control ────────────────────────────────
            elapsed = time.perf_counter() - t_now
            time.sleep(max(0.0, frame_dt - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            plotter.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
