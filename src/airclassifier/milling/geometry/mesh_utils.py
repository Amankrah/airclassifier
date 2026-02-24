"""
Mesh Utilities for Hammer Mill Geometry
========================================

Shared mesh generation helper functions for the hammer mill
geometry components. Includes primitives for cylinders, arcs,
boxes, and compound shapes used in mill construction.

Coordinate system:
    X = along rotor axis (shaft direction)
    Y = vertical (up)
    Z = lateral (perpendicular to shaft)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def box_mesh(
    x0: float, y0: float, z0: float,
    lx: float, ly: float, lz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an axis-aligned box mesh.

    Args:
        x0, y0, z0: Minimum corner position.
        lx, ly, lz: Dimensions along each axis.

    Returns:
        (vertices [8, 3], triangles [12, 3])
    """
    verts = np.array([
        [x0,      y0,      z0],
        [x0 + lx, y0,      z0],
        [x0 + lx, y0 + ly, z0],
        [x0,      y0 + ly, z0],
        [x0,      y0,      z0 + lz],
        [x0 + lx, y0,      z0 + lz],
        [x0 + lx, y0 + ly, z0 + lz],
        [x0,      y0 + ly, z0 + lz],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2], [0, 2, 3],  # -Z face
        [4, 6, 5], [4, 7, 6],  # +Z face
        [0, 4, 5], [0, 5, 1],  # -Y face
        [2, 6, 7], [2, 7, 3],  # +Y face
        [0, 3, 7], [0, 7, 4],  # -X face
        [1, 5, 6], [1, 6, 2],  # +X face
    ], dtype=np.int32)
    return verts, tris


def cylinder_mesh(
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    resolution: int = 24,
    axis: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a cylinder mesh along the specified axis.

    Args:
        center: Center of the cylinder base (start point).
        radius: Cylinder radius.
        height: Cylinder height (length along axis).
        resolution: Number of radial segments.
        axis: 'x', 'y', or 'z' for cylinder axis.

    Returns:
        (vertices, triangles)
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts = []
    if axis == "x":
        # Cylinder along X axis (rotor shaft direction)
        for i in range(resolution):
            verts.append([cx, cy + radius * cos_a[i], cz + radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + height, cy + radius * cos_a[i], cz + radius * sin_a[i]])
        verts.append([cx, cy, cz])                    # Base center
        verts.append([cx + height, cy, cz])           # End center
    elif axis == "y":
        # Cylinder along Y axis (vertical)
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy, cz + radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy + height, cz + radius * sin_a[i]])
        verts.append([cx, cy, cz])
        verts.append([cx, cy + height, cz])
    else:  # axis == "z"
        # Cylinder along Z axis
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy + radius * sin_a[i], cz])
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy + radius * sin_a[i], cz + height])
        verts.append([cx, cy, cz])
        verts.append([cx, cy, cz + height])

    verts = np.array(verts, dtype=np.float32)

    # Generate triangles
    tris = []
    n = resolution
    base_center = 2 * n
    end_center = 2 * n + 1

    for i in range(n):
        i_next = (i + 1) % n
        # Side faces (two triangles per quad)
        tris.append([i, i_next, i_next + n])
        tris.append([i, i_next + n, i + n])
        # Base cap
        tris.append([base_center, i_next, i])
        # End cap
        tris.append([end_center, i + n, i_next + n])

    tris = np.array(tris, dtype=np.int32)
    return verts, tris


def frustum_mesh(
    center: Tuple[float, float, float],
    radius_base: float,
    radius_top: float,
    height: float,
    resolution: int = 24,
    axis: str = "z",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a truncated cone (frustum) mesh along the given axis.

    Base (radius_base) is at center; top (radius_top) is at center + height.

    Args:
        center: Center of the base.
        radius_base: Radius at the base.
        radius_top: Radius at the top.
        height: Length along the axis.
        resolution: Number of radial segments.
        axis: 'x', 'y', or 'z'.

    Returns:
        (vertices, triangles) for the lateral surface (no caps).
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts = []
    if axis == "x":
        for i in range(resolution):
            verts.append([cx, cy + radius_base * cos_a[i], cz + radius_base * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + height, cy + radius_top * cos_a[i], cz + radius_top * sin_a[i]])
    elif axis == "y":
        for i in range(resolution):
            verts.append([cx + radius_base * cos_a[i], cy, cz + radius_base * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + radius_top * cos_a[i], cy + height, cz + radius_top * sin_a[i]])
    else:  # axis == "z"
        for i in range(resolution):
            verts.append([cx + radius_base * cos_a[i], cy + radius_base * sin_a[i], cz])
        for i in range(resolution):
            verts.append([cx + radius_top * cos_a[i], cy + radius_top * sin_a[i], cz + height])

    verts = np.array(verts, dtype=np.float32)
    tris = []
    n = resolution
    for i in range(n):
        i_next = (i + 1) % n
        tris.append([i, i_next, i_next + n])
        tris.append([i, i_next + n, i + n])
    tris = np.array(tris, dtype=np.int32)
    return verts, tris


def disc_mesh(
    center: Tuple[float, float, float],
    inner_radius: float,
    outer_radius: float,
    thickness: float,
    resolution: int = 24,
    axis: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an annular disc (ring) mesh.

    Used for rotor discs that support the hammers.

    Args:
        center: Center of the disc.
        inner_radius: Inner radius (shaft hole).
        outer_radius: Outer radius.
        thickness: Disc thickness.
        resolution: Number of radial segments.
        axis: 'x', 'y', or 'z' for disc normal axis.

    Returns:
        (vertices, triangles)
    """
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    cx, cy, cz = center
    half_t = thickness / 2.0
    verts = []

    if axis == "x":
        # Disc with normal along X
        for i in range(resolution):
            # Front face - inner ring
            verts.append([cx - half_t, cy + inner_radius * cos_a[i], cz + inner_radius * sin_a[i]])
        for i in range(resolution):
            # Front face - outer ring
            verts.append([cx - half_t, cy + outer_radius * cos_a[i], cz + outer_radius * sin_a[i]])
        for i in range(resolution):
            # Back face - inner ring
            verts.append([cx + half_t, cy + inner_radius * cos_a[i], cz + inner_radius * sin_a[i]])
        for i in range(resolution):
            # Back face - outer ring
            verts.append([cx + half_t, cy + outer_radius * cos_a[i], cz + outer_radius * sin_a[i]])
    elif axis == "y":
        # Disc with normal along Y
        for i in range(resolution):
            verts.append([cx + inner_radius * cos_a[i], cy - half_t, cz + inner_radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + outer_radius * cos_a[i], cy - half_t, cz + outer_radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + inner_radius * cos_a[i], cy + half_t, cz + inner_radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + outer_radius * cos_a[i], cy + half_t, cz + outer_radius * sin_a[i]])
    else:  # axis == "z"
        for i in range(resolution):
            verts.append([cx + inner_radius * cos_a[i], cy + inner_radius * sin_a[i], cz - half_t])
        for i in range(resolution):
            verts.append([cx + outer_radius * cos_a[i], cy + outer_radius * sin_a[i], cz - half_t])
        for i in range(resolution):
            verts.append([cx + inner_radius * cos_a[i], cy + inner_radius * sin_a[i], cz + half_t])
        for i in range(resolution):
            verts.append([cx + outer_radius * cos_a[i], cy + outer_radius * sin_a[i], cz + half_t])

    verts = np.array(verts, dtype=np.float32)
    n = resolution

    tris = []
    for i in range(n):
        i_next = (i + 1) % n

        # Front face (annulus between inner and outer)
        tris.append([i, i_next, i_next + n])
        tris.append([i, i_next + n, i + n])

        # Back face
        tris.append([2*n + i, 2*n + i + n, 2*n + i_next + n])
        tris.append([2*n + i, 2*n + i_next + n, 2*n + i_next])

        # Outer edge (connects front outer to back outer)
        tris.append([i + n, i_next + n, 3*n + i_next])
        tris.append([i + n, 3*n + i_next, 3*n + i])

        # Inner edge (connects front inner to back inner)
        tris.append([i, 2*n + i, 2*n + i_next])
        tris.append([i, 2*n + i_next, i_next])

    tris = np.array(tris, dtype=np.int32)
    return verts, tris


def arc_surface_mesh(
    center: Tuple[float, float, float],
    inner_radius: float,
    outer_radius: float,
    start_angle_rad: float,
    end_angle_rad: float,
    length: float,
    radial_resolution: int = 16,
    axial_resolution: int = 8,
    axis: str = "x",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a curved arc surface mesh (for screen geometry).

    Creates a section of a cylinder shell between two radii and
    two angles.

    Args:
        center: Center of the arc (on the axis).
        inner_radius: Inner radius of the arc.
        outer_radius: Outer radius of the arc.
        start_angle_rad: Start angle [rad] (0 = +Y in YZ plane for X-axis).
        end_angle_rad: End angle [rad].
        length: Length along the axis.
        radial_resolution: Angular segments.
        axial_resolution: Segments along length.
        axis: Axis along which the arc extends.

    Returns:
        (vertices, triangles)
    """
    cx, cy, cz = center
    angles = np.linspace(start_angle_rad, end_angle_rad, radial_resolution + 1)
    positions = np.linspace(0, length, axial_resolution + 1)

    verts = []

    if axis == "x":
        # Inner surface
        for x_pos in positions:
            for angle in angles:
                y = cy + inner_radius * math.cos(angle)
                z = cz + inner_radius * math.sin(angle)
                verts.append([cx + x_pos, y, z])
        # Outer surface
        for x_pos in positions:
            for angle in angles:
                y = cy + outer_radius * math.cos(angle)
                z = cz + outer_radius * math.sin(angle)
                verts.append([cx + x_pos, y, z])
    elif axis == "y":
        for y_pos in positions:
            for angle in angles:
                x = cx + inner_radius * math.cos(angle)
                z = cz + inner_radius * math.sin(angle)
                verts.append([x, cy + y_pos, z])
        for y_pos in positions:
            for angle in angles:
                x = cx + outer_radius * math.cos(angle)
                z = cz + outer_radius * math.sin(angle)
                verts.append([x, cy + y_pos, z])
    else:  # axis == "z"
        for z_pos in positions:
            for angle in angles:
                x = cx + inner_radius * math.cos(angle)
                y = cy + inner_radius * math.sin(angle)
                verts.append([x, y, cz + z_pos])
        for z_pos in positions:
            for angle in angles:
                x = cx + outer_radius * math.cos(angle)
                y = cy + outer_radius * math.sin(angle)
                verts.append([x, y, cz + z_pos])

    verts = np.array(verts, dtype=np.float32)

    # Generate triangles
    tris = []
    n_ang = radial_resolution + 1
    n_ax = axial_resolution + 1

    # Inner surface quads
    for i in range(axial_resolution):
        for j in range(radial_resolution):
            v0 = i * n_ang + j
            v1 = i * n_ang + j + 1
            v2 = (i + 1) * n_ang + j + 1
            v3 = (i + 1) * n_ang + j
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])

    # Outer surface quads (offset by n_ang * n_ax vertices)
    offset = n_ang * n_ax
    for i in range(axial_resolution):
        for j in range(radial_resolution):
            v0 = offset + i * n_ang + j
            v1 = offset + i * n_ang + j + 1
            v2 = offset + (i + 1) * n_ang + j + 1
            v3 = offset + (i + 1) * n_ang + j
            # Reverse winding for outer surface
            tris.append([v0, v2, v1])
            tris.append([v0, v3, v2])

    # End caps (connect inner to outer at start and end angles)
    # Start angle edge
    for i in range(axial_resolution):
        v_in_0 = i * n_ang
        v_in_1 = (i + 1) * n_ang
        v_out_0 = offset + i * n_ang
        v_out_1 = offset + (i + 1) * n_ang
        tris.append([v_in_0, v_out_0, v_out_1])
        tris.append([v_in_0, v_out_1, v_in_1])

    # End angle edge
    for i in range(axial_resolution):
        v_in_0 = i * n_ang + radial_resolution
        v_in_1 = (i + 1) * n_ang + radial_resolution
        v_out_0 = offset + i * n_ang + radial_resolution
        v_out_1 = offset + (i + 1) * n_ang + radial_resolution
        tris.append([v_in_0, v_in_1, v_out_1])
        tris.append([v_in_0, v_out_1, v_out_0])

    tris = np.array(tris, dtype=np.int32)
    return verts, tris


def concat_meshes(
    parts: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple (vertices, triangles) meshes into one.

    Properly offsets triangle indices for each part.

    Args:
        parts: List of (vertices, triangles) tuples.

    Returns:
        (combined_vertices, combined_triangles)
    """
    if not parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
    all_v = []
    all_t = []
    offset = 0
    for v, t in parts:
        all_v.append(v)
        all_t.append(t + offset)
        offset += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_t).astype(np.int32)


def translate_mesh(
    verts: np.ndarray,
    tris: np.ndarray,
    tx: float, ty: float, tz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Translate vertices by (tx, ty, tz).

    Returns:
        (translated_vertices, triangles_copy)
    """
    offset = np.array([tx, ty, tz], dtype=np.float32)
    return verts + offset, np.array(tris, dtype=np.int32, copy=True)


def rotate_mesh_around_x(
    verts: np.ndarray,
    tris: np.ndarray,
    angle_rad: float,
    pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate vertices around the X axis.

    Args:
        verts: Vertex array.
        tris: Triangle array.
        angle_rad: Rotation angle [rad].
        pivot: Pivot point.

    Returns:
        (rotated_vertices, triangles_copy)
    """
    px, py, pz = pivot
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Translate to pivot
    v = verts.copy()
    v[:, 1] -= py
    v[:, 2] -= pz

    # Rotate around X
    new_y = cos_a * v[:, 1] - sin_a * v[:, 2]
    new_z = sin_a * v[:, 1] + cos_a * v[:, 2]
    v[:, 1] = new_y
    v[:, 2] = new_z

    # Translate back
    v[:, 1] += py
    v[:, 2] += pz

    return v.astype(np.float32), np.array(tris, dtype=np.int32, copy=True)


def multi_box_mesh(boxes: List[Tuple[float, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multiple boxes and concatenate.

    Args:
        boxes: List of (x0, y0, z0, lx, ly, lz) tuples.

    Returns:
        (vertices, triangles) combined mesh.
    """
    return concat_meshes([box_mesh(*b) for b in boxes])
