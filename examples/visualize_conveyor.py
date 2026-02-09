#!/usr/bin/env python
"""
Visualize GP-15 Conveyor Belt
==============================

Shows the realistic GP-15 conveyor: structural steel frame with legs,
head/tail/tension/tracker/return rollers, continuous belt loop,
attenuation ducts, and optional material bed.

Usage:
    python examples/visualize_conveyor.py                  # Default (with material bed)
    python examples/visualize_conveyor.py --no-bed         # No material bed
    python examples/visualize_conveyor.py --wireframe      # Wireframe mode
    python examples/visualize_conveyor.py --side-view      # XY cross-section
    python examples/visualize_conveyor.py --belt-only      # Belt loop + rollers only
    python examples/visualize_conveyor.py --xray           # Transparent frame, solid belt
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pyvista as pv

from airclassifier.pretreatment.geometry.components.conveyor_belt import (
    ConveyorBeltGeometry,
    ConveyorBeltParams,
)


def mesh_to_polydata(vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    """Build PyVista PolyData from vertices and triangle indices."""
    n_faces = triangles.shape[0]
    faces = np.empty((n_faces, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles
    return pv.PolyData(vertices, faces.ravel())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize GP-15 conveyor belt geometry (real-world structure)"
    )
    parser.add_argument("--no-bed", action="store_true",
                        help="Omit material bed (product) mesh")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render as wireframe")
    parser.add_argument("--belt-only", action="store_true",
                        help="Show belt loop and rollers only (no frame)")
    parser.add_argument("--side-view", action="store_true",
                        help="Camera aligned to XY side-view (belt loop profile)")
    parser.add_argument("--xray", action="store_true",
                        help="Transparent frame, solid belt + rollers")
    parser.add_argument("--bed-depth", type=float, default=0.04,
                        help="Material bed depth in m (default 0.04)")
    args = parser.parse_args()

    # Build conveyor with GP-15 params
    params = ConveyorBeltParams()
    conveyor = ConveyorBeltGeometry(params)

    # ── Generate meshes (wheels first — belt reads roller layout from it) ──
    bed_struct_v, bed_struct_t, bed_meta = conveyor.generate_bed_structure_mesh()
    wheels_v, wheels_t, wheels_meta      = conveyor.generate_wheels_mesh()
    belt_v, belt_t, belt_meta            = conveyor.generate_belt_mesh()

    print(f"Frame     — vertices: {bed_struct_v.shape[0]:,}  triangles: {bed_struct_t.shape[0]:,}")
    print(f"Rollers   — vertices: {wheels_v.shape[0]:,}  triangles: {wheels_t.shape[0]:,}")
    print(f"Belt loop — vertices: {belt_v.shape[0]:,}  triangles: {belt_t.shape[0]:,}")

    frame_pd  = mesh_to_polydata(bed_struct_v, bed_struct_t)
    wheels_pd = mesh_to_polydata(wheels_v, wheels_t)
    belt_pd   = mesh_to_polydata(belt_v, belt_t)

    # ── Plotter ──────────────────────────────────────────────────────
    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.camera.up = (0, 1, 0)

    if args.wireframe:
        # ── Wireframe mode ───────────────────────────────────────────
        if not args.belt_only:
            plotter.add_mesh(frame_pd, style="wireframe", color="black",
                             line_width=1, label="Frame (steel)")
        plotter.add_mesh(wheels_pd, style="wireframe", color="gray",
                         line_width=1, label="Rollers")
        plotter.add_mesh(belt_pd, style="wireframe", color="#2255CC",
                         line_width=1.2, label="Belt loop (PTFE)")
        if not args.no_bed and not args.belt_only:
            mb_v, mb_t, _ = conveyor.generate_bed_mesh(args.bed_depth)
            plotter.add_mesh(mesh_to_polydata(mb_v, mb_t),
                             style="wireframe", color="orange",
                             line_width=1, label="Material bed")
    elif args.xray:
        # ── X-ray: transparent frame, solid belt ─────────────────────
        if not args.belt_only:
            plotter.add_mesh(frame_pd, color="#505060", opacity=0.12,
                             label="Frame (steel)")
        plotter.add_mesh(wheels_pd, color="#808090", opacity=0.85,
                         smooth_shading=True, label="Rollers")
        plotter.add_mesh(belt_pd, color="#4169E1", opacity=0.95,
                         show_edges=True, edge_color="#2a4a9a",
                         line_width=0.4, smooth_shading=True,
                         backface_params=dict(color="#5B7FD9"),
                         label="Belt loop (PTFE)")
        if not args.no_bed and not args.belt_only:
            mb_v, mb_t, _ = conveyor.generate_bed_mesh(args.bed_depth)
            plotter.add_mesh(mesh_to_polydata(mb_v, mb_t),
                             color="#DAA520", opacity=0.80,
                             smooth_shading=True, label="Material bed")
    else:
        # ── Solid mode ───────────────────────────────────────────────
        if not args.belt_only:
            plotter.add_mesh(frame_pd, color="#505060",
                             opacity=0.30 if not args.belt_only else 0.0,
                             smooth_shading=True, label="Frame (steel)")
        plotter.add_mesh(wheels_pd, color="#808090", opacity=0.85,
                         smooth_shading=True, label="Rollers")
        plotter.add_mesh(belt_pd, color="#4169E1", opacity=0.92,
                         show_edges=True, edge_color="#2a4a9a",
                         line_width=0.4, smooth_shading=True,
                         backface_params=dict(color="#5B7FD9"),
                         label="Belt loop (PTFE)")
        if not args.no_bed and not args.belt_only:
            mb_v, mb_t, _ = conveyor.generate_bed_mesh(args.bed_depth)
            plotter.add_mesh(mesh_to_polydata(mb_v, mb_t),
                             color="#DAA520", opacity=0.85,
                             smooth_shading=True, label="Material bed")

    # ── Legend, title, camera ────────────────────────────────────────
    plotter.add_legend(loc="upper left", bcolor="white")
    plotter.add_title(
        "GP-15 Conveyor — 5.5 m frame, 800 mm belt, "
        "head/tail/tension/tracker rollers"
    )
    plotter.add_axes()
    plotter.reset_camera()

    if args.side_view:
        mid_x = params.frame_length_m / 2
        mid_y = -params.frame_height_m / 2
        z_cam = params.frame_width_m + 3.5
        plotter.camera.position = (mid_x, mid_y, z_cam)
        plotter.camera.focal_point = (mid_x, mid_y, params.belt_center_z)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.zoom(1.2)
    else:
        plotter.camera.azimuth = -60
        plotter.camera.elevation = 15
        plotter.camera.zoom(1.1)

    plotter.show()


if __name__ == "__main__":
    main()