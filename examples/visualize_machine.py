#!/usr/bin/env python
"""
Visualize GP-15 Machine Assembly
=================================

Renders the full GP-15 RF dielectric heating machine assembled from
the component library: conveyor frame + rollers + belt loop, oven
chamber, upper & lower electrodes, and material bed.

Usage:
    python examples/visualize_machine.py                       # Default 3D view
    python examples/visualize_machine.py --xray                # Transparent oven/frame
    python examples/visualize_machine.py --exploded            # Exploded view
    python examples/visualize_machine.py --wireframe           # Wireframe mode
    python examples/visualize_machine.py --side-view           # Side section (XY)
    python examples/visualize_machine.py --no-bed              # No material bed
    python examples/visualize_machine.py --gap 150             # Electrode gap (mm)
    python examples/visualize_machine.py --bed-depth 60        # Bed depth (mm)
    python examples/visualize_machine.py --gap 100 --xray      # Combined options
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pyvista as pv

from airclassifier.pretreatment.geometry.assembly import (
    create_gp15_machine,
    COMPONENT_COLORS,
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


def print_mesh_stats(meshes: dict) -> None:
    """Print per-component vertex/triangle counts."""
    total_v = total_t = 0
    print("\n  Component               Vertices   Triangles")
    print("  " + "─" * 50)
    for name, (v, t, meta) in meshes.items():
        label = COMPONENT_COLORS.get(name, {}).get("label", name)
        print(f"  {label:24s}  {v.shape[0]:>7,}    {t.shape[0]:>7,}")
        total_v += v.shape[0]
        total_t += t.shape[0]
    print("  " + "─" * 50)
    print(f"  {'TOTAL':24s}  {total_v:>7,}    {total_t:>7,}")
    print()


# ─────────────────────────────────────────────────────────────────────
#  Render modes
# ─────────────────────────────────────────────────────────────────────

def add_solid(plotter: pv.Plotter, meshes: dict, xray: bool = False) -> None:
    """Add all components in solid/surface mode."""
    # Opacity overrides for x-ray mode
    xray_opacities = {
        "conveyor_frame": 0.08,
        "oven_chamber": 0.06,
        "rollers": 0.80,
        "belt": 0.92,
        "upper_electrode": 0.90,
        "lower_electrode": 0.85,
        "material_bed": 0.80,
        "infeed_hopper": 0.88,
        "infeed_tunnel": 0.30,
        "emu_housing": 0.15,
        "generator": 0.25,
        "rf_feed": 0.95,
    }

    for name, (v, t, meta) in meshes.items():
        style = COMPONENT_COLORS.get(name, {})
        color = style.get("color", "#888888")
        label = style.get("label", name)
        opacity = (xray_opacities.get(name, style.get("opacity", 0.8))
                   if xray else style.get("opacity", 0.8))

        pd = mesh_to_polydata(v, t)

        kwargs = dict(
            color=color,
            opacity=opacity,
            smooth_shading=True,
            label=label,
        )

        # Belt gets special edge rendering
        if name == "belt":
            kwargs.update(
                show_edges=True,
                edge_color="#2a4a9a",
                line_width=0.4,
                backface_params=dict(color="#5B7FD9"),
            )

        plotter.add_mesh(pd, **kwargs)


def add_wireframe(plotter: pv.Plotter, meshes: dict) -> None:
    """Add all components in wireframe mode."""
    wire_colors = {
        "conveyor_frame": "#303040",
        "rollers": "#606060",
        "belt": "#2255CC",
        "oven_chamber": "#8B6914",
        "upper_electrode": "#C0392B",
        "lower_electrode": "#7B2D8E",
        "material_bed": "#DAA520",
        "infeed_hopper": "#707078",
        "infeed_tunnel": "#505058",
        "emu_housing": "#909098",
        "generator": "#607080",
        "rf_feed": "#CD7F32",
    }
    for name, (v, t, meta) in meshes.items():
        label = COMPONENT_COLORS.get(name, {}).get("label", name)
        color = wire_colors.get(name, "#888888")
        lw = 1.5 if name == "belt" else 1.0
        plotter.add_mesh(
            mesh_to_polydata(v, t),
            style="wireframe",
            color=color,
            line_width=lw,
            label=label,
        )


def add_exploded(plotter: pv.Plotter, meshes: dict) -> None:
    """Add components with vertical offsets for exploded view."""
    # Y offsets to separate layers vertically
    explode_offsets = {
        "conveyor_frame": 0.0,
        "rollers": 0.0,
        "belt": 0.0,
        "oven_chamber": 0.40,
        "upper_electrode": 0.25,
        "lower_electrode": -0.10,
        "material_bed": 0.08,
        "infeed_hopper": 0.10,
        "infeed_tunnel": 0.05,
        "emu_housing": 0.50,
        "generator": 0.0,
        "rf_feed": 0.15,
    }

    for name, (v, t, meta) in meshes.items():
        style = COMPONENT_COLORS.get(name, {})
        color = style.get("color", "#888888")
        label = style.get("label", name)
        opacity = style.get("opacity", 0.8)

        pd = mesh_to_polydata(v, t)

        dy = explode_offsets.get(name, 0.0)
        if abs(dy) > 1e-6:
            pd.translate([0, dy, 0], inplace=True)

        kwargs = dict(
            color=color,
            opacity=min(opacity + 0.10, 1.0),
            smooth_shading=True,
            label=label,
        )
        if name == "belt":
            kwargs.update(
                show_edges=True,
                edge_color="#2a4a9a",
                line_width=0.4,
                backface_params=dict(color="#5B7FD9"),
            )
        plotter.add_mesh(pd, **kwargs)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize the GP-15 RF machine assembly (3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python examples/visualize_machine.py
    python examples/visualize_machine.py --xray
    python examples/visualize_machine.py --exploded
    python examples/visualize_machine.py --wireframe
    python examples/visualize_machine.py --side-view
    python examples/visualize_machine.py --gap 150 --bed-depth 60 --xray
""",
    )
    parser.add_argument("--gap", type=float, default=200,
                        help="Electrode gap in mm (default 200)")
    parser.add_argument("--bed-depth", type=float, default=40,
                        help="Material bed depth in mm (default 40)")
    parser.add_argument("--no-bed", action="store_true",
                        help="Omit material bed")
    parser.add_argument("--wireframe", action="store_true",
                        help="Wireframe rendering")
    parser.add_argument("--xray", action="store_true",
                        help="X-ray mode: transparent oven/frame, solid internals")
    parser.add_argument("--exploded", action="store_true",
                        help="Exploded view with vertical separation")
    parser.add_argument("--side-view", action="store_true",
                        help="Camera aligned to XY side view")
    parser.add_argument("--dark", action="store_true",
                        help="Dark background theme")
    args = parser.parse_args()

    # ── Build machine ─────────────────────────────────────────────
    gap_m = args.gap / 1000.0
    bed_m = args.bed_depth / 1000.0

    print(f"Building GP-15 assembly  (gap={args.gap:.0f} mm, "
          f"bed={args.bed_depth:.0f} mm) ...")

    machine = create_gp15_machine(
        electrode_gap_m=gap_m,
        bed_depth_m=bed_m,
    )

    info = machine.get_assembly_info()
    meshes = machine.generate_all_meshes(include_bed=not args.no_bed)

    print_mesh_stats(meshes)
    print(f"  Oven position:     x = {info['oven_x_start_m']:.2f} – "
          f"{info['oven_x_end_m']:.2f} m")
    print(f"  RF zone:           x = {info['rf_zone_x_start_m']:.2f} – "
          f"{info['rf_zone_x_end_m']:.2f} m")
    print(f"  Electrode gap:     {info['electrode_gap_m']*1000:.0f} mm")
    print(f"  Air gap:           {info['air_gap_m']*1000:.0f} mm")
    print(f"  Belt stack:        {info['belt_stack_thickness_m']*1000:.1f} mm")
    print()

    # ── Plotter ───────────────────────────────────────────────────
    plotter = pv.Plotter()
    bg = "#1a1a2e" if args.dark else "white"
    plotter.set_background(bg)
    plotter.camera.up = (0, 1, 0)

    # ── Add meshes with chosen mode ───────────────────────────────
    if args.wireframe:
        add_wireframe(plotter, meshes)
    elif args.exploded:
        add_exploded(plotter, meshes)
    else:
        add_solid(plotter, meshes, xray=args.xray)

    # ── Legend & title ────────────────────────────────────────────
    legend_bg = (0.1, 0.1, 0.15, 0.8) if args.dark else "white"
    plotter.add_legend(loc="upper left", bcolor=legend_bg)

    mode_tag = ""
    if args.xray:
        mode_tag = " [X-RAY]"
    elif args.exploded:
        mode_tag = " [EXPLODED]"
    elif args.wireframe:
        mode_tag = " [WIREFRAME]"

    plotter.add_title(
        f"GP-15 Machine Assembly{mode_tag}  —  "
        f"Gap {args.gap:.0f} mm  |  Bed {args.bed_depth:.0f} mm  |  "
        f"Air gap {info['air_gap_m']*1000:.0f} mm",
        font_size=10,
    )
    plotter.add_axes()

    # ── Camera ────────────────────────────────────────────────────
    if args.side_view:
        cp = machine.params.conveyor_params
        mid_x = cp.frame_length_m / 2
        mid_y = -cp.frame_height_m / 2 + 0.15
        z_cam = cp.frame_width_m + 4.0
        plotter.camera.position = (mid_x, mid_y, z_cam)
        plotter.camera.focal_point = (mid_x, mid_y, cp.belt_center_z)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.zoom(1.2)
    else:
        plotter.reset_camera()
        plotter.camera.azimuth = -55
        plotter.camera.elevation = 18
        plotter.camera.zoom(1.1)

    plotter.show()


if __name__ == "__main__":
    main()
