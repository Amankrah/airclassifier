"""
3D Field Visualization Helpers
==============================

Utilities for rendering pretreatment simulation fields in the
Air Classifier Designer's PyVista viewport (engineering guide §9.3).

Provides:
- ``fields_to_pyvista_grid`` — convert NumPy fields to PyVista grid
- ``create_bed_slice`` — cross-section through the material bed
- ``build_pyvista_scene`` — complete oven + fields for the 3D viewport
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..simulator import GP15Simulator


def fields_to_pyvista_grid(
    grid_shape: Tuple[int, int, int],
    cell_sizes: Tuple[float, float, float],
    T: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    P_v: Optional[np.ndarray] = None,
    cell_is_material: Optional[np.ndarray] = None,
):
    """Convert 3D field arrays to a PyVista UniformGrid for visualization.

    Args:
        grid_shape: (nx, ny, nz).
        cell_sizes: (dx, dy, dz) in metres.
        T: Temperature field [degC].
        M: Moisture field [wet basis fraction].
        P_v: RF power density [W/m^3].
        cell_is_material: Material mask (1=material, 0=air, 2=belt).

    Returns:
        pyvista.ImageData with scalar arrays attached.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("PyVista is required for visualization")

    nx, ny, nz = grid_shape
    dx, dy, dz = cell_sizes

    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
    )

    if T is not None:
        grid.cell_data["Temperature [C]"] = T.flatten(order="F")
    if M is not None:
        grid.cell_data["Moisture [wb]"] = M.flatten(order="F")
    if P_v is not None:
        grid.cell_data["RF Power [W/m3]"] = P_v.flatten(order="F")
    if cell_is_material is not None:
        grid.cell_data["Zone"] = cell_is_material.flatten(order="F").astype(float)

    return grid


def create_bed_slice(
    grid,
    axis: str = "y",
    origin: Optional[float] = None,
):
    """Create a cross-section slice through the material bed.

    Args:
        grid: PyVista grid from fields_to_pyvista_grid.
        axis: Slice normal axis ("x", "y", or "z").
        origin: Position along the axis. None = center.

    Returns:
        pyvista.PolyData slice.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("PyVista is required")

    bounds = grid.bounds
    if origin is None:
        if axis == "x":
            origin = (bounds[0] + bounds[1]) / 2
        elif axis == "y":
            origin = (bounds[2] + bounds[3]) / 2
        else:
            origin = (bounds[4] + bounds[5]) / 2

    normal = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    point = {"x": (origin, 0, 0), "y": (0, origin, 0), "z": (0, 0, origin)}[axis]

    return grid.slice(normal=normal, origin=point)


def build_pyvista_scene(
    simulator: "GP15Simulator",
    scalar: str = "Temperature [C]",
    show_edges: bool = True,
    opacity_material: float = 0.8,
    opacity_structure: float = 0.15,
) -> Dict[str, Any]:
    """Build a complete PyVista scene from a GP15Simulator instance.

    Returns a dict of named mesh actors that can be added to the
    Air Classifier Designer's existing PyVista viewport (§9.3).

    The scene contains:
    - Oven walls (wireframe, semi-transparent)
    - Upper and lower electrode plates
    - Conveyor belt
    - Material bed with field color mapping

    Args:
        simulator: A GP15Simulator instance (may or may not have run).
        scalar: Field name to color the material bed by.
        show_edges: Show grid edges on the field mesh.
        opacity_material: Opacity for the material bed field mesh.
        opacity_structure: Opacity for structural components.

    Returns:
        Dict mapping component names to PyVista mesh objects::

            {
                "field_grid": pv.ImageData,       # 3D field grid
                "bed_slice_y": pv.PolyData,       # Y-normal slice
                "bed_slice_x": pv.PolyData,       # X-normal slice
                "oven_walls": pv.PolyData,        # Oven structure
                "upper_electrode": pv.PolyData,
                "lower_electrode": pv.PolyData,
                "belt": pv.PolyData,
            }
    """
    try:
        import pyvista as pv
    except ImportError:
        raise RuntimeError("PyVista is required for visualization")

    meshes = simulator.get_mesh()
    scene: Dict[str, Any] = {}

    # Structural meshes
    for name in ("oven", "upper_electrode", "lower_electrode", "belt"):
        if name in meshes:
            v = meshes[name]["vertices"]
            t = meshes[name]["triangles"]
            faces = np.column_stack([
                np.full(len(t), 3, dtype=np.int32), t
            ]).ravel()
            scene[name] = pv.PolyData(v, faces)

    # Field grid (if simulation data is available)
    if "fields" in meshes:
        f = meshes["fields"]
        grid = fields_to_pyvista_grid(
            grid_shape=f["grid_shape"],
            cell_sizes=f["cell_sizes"],
            T=f.get("temperature"),
            M=f.get("moisture"),
            P_v=f.get("power_density"),
        )
        scene["field_grid"] = grid

        # Slices
        try:
            scene["bed_slice_y"] = create_bed_slice(grid, axis="y")
            scene["bed_slice_x"] = create_bed_slice(grid, axis="x")
        except Exception:
            pass  # Slicing may fail on very small grids

    return scene
