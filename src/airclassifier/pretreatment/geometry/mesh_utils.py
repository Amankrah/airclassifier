"""
Mesh Utilities
==============

Shared mesh generation helper functions for the GP-15 pretreatment
geometry components.
"""

from __future__ import annotations

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


def hollow_box_mesh(
    x0: float, y0: float, z0: float,
    lx: float, ly: float, lz: float,
    wall: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a hollow box (4 walls, open ends along X).

    Used for tunnel-like shapes.

    Args:
        x0, y0, z0: Minimum corner position.
        lx, ly, lz: Outer dimensions.
        wall: Wall thickness.

    Returns:
        (vertices, triangles)
    """
    parts = [
        box_mesh(x0, y0, z0, lx, wall, lz),                    # bottom
        box_mesh(x0, y0 + ly - wall, z0, lx, wall, lz),        # top
        box_mesh(x0, y0, z0, lx, ly, wall),                    # -Z wall
        box_mesh(x0, y0, z0 + lz - wall, lx, ly, wall),        # +Z wall
    ]
    return concat_meshes(parts)


def multi_box_mesh(boxes: List[Tuple[float, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multiple boxes and concatenate.

    Args:
        boxes: List of (x0, y0, z0, lx, ly, lz) tuples.

    Returns:
        (vertices, triangles) combined mesh.
    """
    return concat_meshes([box_mesh(*b) for b in boxes])


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


def cylinder_mesh(
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    resolution: int = 16,
    axis: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a cylinder mesh.

    Args:
        center: Center of the cylinder base.
        radius: Cylinder radius.
        height: Cylinder height.
        resolution: Number of radial segments.
        axis: 'x', 'y', or 'z' for cylinder axis.

    Returns:
        (vertices, triangles)
    """
    cx, cy, cz = center
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Generate vertices
    verts = []
    if axis == "y":
        # Base circle
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy, cz + radius * sin_a[i]])
        # Top circle
        for i in range(resolution):
            verts.append([cx + radius * cos_a[i], cy + height, cz + radius * sin_a[i]])
        # Centers for caps
        verts.append([cx, cy, cz])                 # Base center
        verts.append([cx, cy + height, cz])        # Top center
    elif axis == "x":
        for i in range(resolution):
            verts.append([cx, cy + radius * cos_a[i], cz + radius * sin_a[i]])
        for i in range(resolution):
            verts.append([cx + height, cy + radius * cos_a[i], cz + radius * sin_a[i]])
        verts.append([cx, cy, cz])
        verts.append([cx + height, cy, cz])
    else:  # axis == "z"
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
    top_center = 2 * n + 1

    for i in range(n):
        i_next = (i + 1) % n
        # Side faces (two triangles per quad)
        tris.append([i, i_next, i_next + n])
        tris.append([i, i_next + n, i + n])
        # Base cap
        tris.append([base_center, i_next, i])
        # Top cap
        tris.append([top_center, i + n, i_next + n])

    tris = np.array(tris, dtype=np.int32)
    return verts, tris
