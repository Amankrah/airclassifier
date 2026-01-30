"""
Mesh generation utilities for cyclone air classifier.

Provides functions for generating computational meshes including:
- Structured grids for SDF evaluation
- Volumetric mesh generation
- Adaptive mesh refinement based on geometry
- Export utilities for visualization tools
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import warp as wp

from ..utils.constants import PI


@dataclass
class GridParams:
    """Parameters for structured grid generation."""

    # Domain bounds
    min_corner: Tuple[float, float, float]  # [m] Minimum corner (x, y, z)
    max_corner: Tuple[float, float, float]  # [m] Maximum corner (x, y, z)

    # Resolution
    nx: int = 50  # Number of cells in x direction
    ny: int = 100  # Number of cells in y direction
    nz: int = 50  # Number of cells in z direction

    @property
    def dx(self) -> float:
        """Cell size in x direction."""
        return (self.max_corner[0] - self.min_corner[0]) / self.nx

    @property
    def dy(self) -> float:
        """Cell size in y direction."""
        return (self.max_corner[1] - self.min_corner[1]) / self.ny

    @property
    def dz(self) -> float:
        """Cell size in z direction."""
        return (self.max_corner[2] - self.min_corner[2]) / self.nz

    @property
    def cell_volume(self) -> float:
        """Volume of a single cell."""
        return self.dx * self.dy * self.dz

    @property
    def total_cells(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny * self.nz

    @classmethod
    def from_cyclone_bounds(
        cls,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        resolution: float = 0.01,
        padding: float = 0.02
    ) -> "GridParams":
        """
        Create grid parameters from cyclone bounding box.

        Args:
            min_corner: Minimum corner of bounding box
            max_corner: Maximum corner of bounding box
            resolution: Target cell size [m]
            padding: Extra padding around geometry [m]

        Returns:
            GridParams instance
        """
        # Add padding
        min_padded = min_corner - padding
        max_padded = max_corner + padding

        # Calculate number of cells
        extent = max_padded - min_padded
        nx = max(10, int(np.ceil(extent[0] / resolution)))
        ny = max(10, int(np.ceil(extent[1] / resolution)))
        nz = max(10, int(np.ceil(extent[2] / resolution)))

        return cls(
            min_corner=tuple(min_padded),
            max_corner=tuple(max_padded),
            nx=nx,
            ny=ny,
            nz=nz
        )


class MeshGenerator:
    """
    Mesh generation utilities for cyclone geometry.

    Provides methods for creating structured grids, extracting
    surface meshes, and exporting to various formats.
    """

    def __init__(self, grid_params: Optional[GridParams] = None):
        """
        Initialize mesh generator.

        Args:
            grid_params: Optional grid parameters. If None, must be
                        provided when calling generation methods.
        """
        self.params = grid_params
        self._grid_points = None
        self._cell_centers = None

    def generate_structured_grid(
        self,
        params: Optional[GridParams] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a structured Cartesian grid.

        Args:
            params: Grid parameters (uses self.params if None)

        Returns:
            Tuple of (X, Y, Z) meshgrid arrays with shape (ny, nx, nz)
        """
        p = params or self.params
        if p is None:
            raise ValueError("Grid parameters must be provided")

        # Create 1D arrays
        x = np.linspace(p.min_corner[0], p.max_corner[0], p.nx + 1)
        y = np.linspace(p.min_corner[1], p.max_corner[1], p.ny + 1)
        z = np.linspace(p.min_corner[2], p.max_corner[2], p.nz + 1)

        # Create 3D meshgrid (note: indexing='ij' for consistent ordering)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self._grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        return X, Y, Z

    def generate_cell_centers(
        self,
        params: Optional[GridParams] = None
    ) -> np.ndarray:
        """
        Generate cell center coordinates.

        Args:
            params: Grid parameters

        Returns:
            Array of shape (nx*ny*nz, 3) with cell center coordinates
        """
        p = params or self.params
        if p is None:
            raise ValueError("Grid parameters must be provided")

        # Cell center coordinates
        x = np.linspace(
            p.min_corner[0] + p.dx/2,
            p.max_corner[0] - p.dx/2,
            p.nx
        )
        y = np.linspace(
            p.min_corner[1] + p.dy/2,
            p.max_corner[1] - p.dy/2,
            p.ny
        )
        z = np.linspace(
            p.min_corner[2] + p.dz/2,
            p.max_corner[2] - p.dz/2,
            p.nz
        )

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        self._cell_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        return self._cell_centers

    def generate_cylindrical_grid(
        self,
        r_min: float,
        r_max: float,
        y_min: float,
        y_max: float,
        nr: int = 20,
        ntheta: int = 48,
        ny: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a cylindrical grid suitable for cyclone geometry.

        Args:
            r_min: Minimum radius [m]
            r_max: Maximum radius [m]
            y_min: Minimum height [m]
            y_max: Maximum height [m]
            nr: Number of radial divisions
            ntheta: Number of angular divisions
            ny: Number of axial divisions

        Returns:
            Tuple of (points, connectivity) where:
            - points: (N, 3) array of vertex coordinates
            - connectivity: (M, 8) array of hexahedral cell vertices
        """
        # Create 1D arrays
        r = np.linspace(r_min, r_max, nr + 1)
        theta = np.linspace(0, 2 * PI, ntheta + 1)[:-1]  # Periodic
        y = np.linspace(y_min, y_max, ny + 1)

        # Generate points
        points_list = []
        for j in range(ny + 1):
            for i in range(nr + 1):
                for k in range(ntheta):
                    x = r[i] * np.cos(theta[k])
                    z = r[i] * np.sin(theta[k])
                    points_list.append([x, y[j], z])

        points = np.array(points_list, dtype=np.float32)

        # Generate hexahedral connectivity
        cells = []
        for j in range(ny):
            for i in range(nr):
                for k in range(ntheta):
                    # Vertex indices (with periodic wrapping in theta)
                    k_next = (k + 1) % ntheta
                    n_ring = (nr + 1) * ntheta

                    v0 = j * n_ring + i * ntheta + k
                    v1 = j * n_ring + i * ntheta + k_next
                    v2 = j * n_ring + (i + 1) * ntheta + k_next
                    v3 = j * n_ring + (i + 1) * ntheta + k
                    v4 = (j + 1) * n_ring + i * ntheta + k
                    v5 = (j + 1) * n_ring + i * ntheta + k_next
                    v6 = (j + 1) * n_ring + (i + 1) * ntheta + k_next
                    v7 = (j + 1) * n_ring + (i + 1) * ntheta + k

                    cells.append([v0, v1, v2, v3, v4, v5, v6, v7])

        connectivity = np.array(cells, dtype=np.int32)

        return points, connectivity

    def generate_adaptive_points(
        self,
        sdf_func,
        params: GridParams,
        near_surface_resolution: float = 0.002,
        surface_thickness: float = 0.01
    ) -> np.ndarray:
        """
        Generate points with adaptive refinement near surfaces.

        Args:
            sdf_func: Signed distance function callable
            params: Base grid parameters
            near_surface_resolution: Resolution near surfaces [m]
            surface_thickness: Distance from surface considered "near" [m]

        Returns:
            Array of shape (N, 3) with adaptively placed points
        """
        # First generate coarse grid
        coarse_centers = self.generate_cell_centers(params)

        # Evaluate SDF at all points
        sdf_values = np.array([sdf_func(p) for p in coarse_centers])

        # Find points near surface
        near_surface_mask = np.abs(sdf_values) < surface_thickness

        # Keep coarse points away from surface
        far_points = coarse_centers[~near_surface_mask]

        # Generate fine points near surface
        if np.any(near_surface_mask):
            near_surface_points = coarse_centers[near_surface_mask]

            # Create fine grid around each near-surface point
            fine_points_list = [far_points]

            # Refinement offsets
            n_refine = int(np.ceil(params.dx / near_surface_resolution))
            offsets = np.linspace(-params.dx/2, params.dx/2, n_refine)

            for p in near_surface_points:
                for dx in offsets[::2]:  # Skip some for efficiency
                    for dy in offsets[::2]:
                        for dz in offsets[::2]:
                            fine_points_list.append(
                                np.array([[p[0] + dx, p[1] + dy, p[2] + dz]])
                            )

            all_points = np.vstack(fine_points_list)
        else:
            all_points = far_points

        return all_points

    def extract_surface_mesh(
        self,
        sdf_values: np.ndarray,
        params: GridParams,
        iso_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract isosurface mesh using marching cubes algorithm.

        Args:
            sdf_values: SDF values on grid of shape (nx, ny, nz)
            params: Grid parameters
            iso_value: Isosurface value (0 for surface)

        Returns:
            Tuple of (vertices, triangles)
        """
        try:
            from skimage import measure
        except ImportError:
            raise ImportError(
                "scikit-image required for marching cubes. "
                "Install with: pip install scikit-image"
            )

        # Run marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            sdf_values,
            level=iso_value,
            spacing=(params.dx, params.dy, params.dz)
        )

        # Offset vertices to correct position
        verts[:, 0] += params.min_corner[0]
        verts[:, 1] += params.min_corner[1]
        verts[:, 2] += params.min_corner[2]

        return verts.astype(np.float32), faces.astype(np.int32)

    @staticmethod
    def combine_meshes(
        meshes: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine multiple meshes into one.

        Args:
            meshes: List of (vertices, indices) tuples

        Returns:
            Combined (vertices, indices) tuple
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        for vertices, indices in meshes:
            all_vertices.append(vertices)
            all_indices.append(indices + vertex_offset)
            vertex_offset += len(vertices)

        combined_vertices = np.vstack(all_vertices)
        combined_indices = np.concatenate(all_indices)

        return combined_vertices, combined_indices

    @staticmethod
    def compute_mesh_quality(
        vertices: np.ndarray,
        triangles: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute mesh quality metrics.

        Args:
            vertices: Vertex positions (N, 3)
            triangles: Triangle indices (M, 3)

        Returns:
            Dictionary of quality metrics
        """
        # Triangle areas
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0

        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Edge lengths
        e01 = np.linalg.norm(edge1, axis=1)
        e02 = np.linalg.norm(edge2, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)

        # Aspect ratio (ratio of longest to shortest edge)
        all_edges = np.stack([e01, e02, e12], axis=1)
        min_edge = np.min(all_edges, axis=1)
        max_edge = np.max(all_edges, axis=1)
        aspect_ratio = max_edge / (min_edge + 1e-10)

        return {
            "num_vertices": len(vertices),
            "num_triangles": len(triangles),
            "total_area": float(np.sum(areas)),
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "mean_area": float(np.mean(areas)),
            "min_edge_length": float(np.min(min_edge)),
            "max_edge_length": float(np.max(max_edge)),
            "mean_aspect_ratio": float(np.mean(aspect_ratio)),
            "max_aspect_ratio": float(np.max(aspect_ratio)),
        }


def generate_cyclone_mesh(
    assembly,
    resolution: float = 0.005,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a complete mesh for a cyclone assembly.

    Args:
        assembly: CycloneAssembly instance
        resolution: Target mesh resolution [m]
        device: Warp device for mesh creation

    Returns:
        Tuple of (vertices, triangles)
    """
    # Build mesh from assembly
    vertices, indices = assembly.build_mesh()

    return vertices, indices


def export_mesh_vtk(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str,
    point_data: Optional[Dict[str, np.ndarray]] = None,
    cell_data: Optional[Dict[str, np.ndarray]] = None
):
    """
    Export mesh to VTK format for ParaView visualization.

    Args:
        vertices: Vertex positions (N, 3)
        triangles: Triangle indices (M, 3)
        filename: Output filename (should end with .vtk)
        point_data: Optional dict of arrays to attach to vertices
        cell_data: Optional dict of arrays to attach to cells
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        # Fallback to manual VTK file writing
        _write_vtk_legacy(vertices, triangles, filename, point_data)
        return

    # Create VTK mesh
    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(v[0], v[1], v[2])

    cells = vtk.vtkCellArray()
    for tri in triangles:
        cells.InsertNextCell(3)
        for idx in tri:
            cells.InsertCellPoint(int(idx))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Add point data
    if point_data:
        for name, data in point_data.items():
            arr = numpy_support.numpy_to_vtk(data)
            arr.SetName(name)
            polydata.GetPointData().AddArray(arr)

    # Add cell data
    if cell_data:
        for name, data in cell_data.items():
            arr = numpy_support.numpy_to_vtk(data)
            arr.SetName(name)
            polydata.GetCellData().AddArray(arr)

    # Write file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def _write_vtk_legacy(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str,
    point_data: Optional[Dict[str, np.ndarray]] = None
):
    """Write VTK legacy format without VTK library."""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Cyclone mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Vertices
        f.write(f"POINTS {len(vertices)} float\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Triangles
        n_tris = len(triangles)
        f.write(f"POLYGONS {n_tris} {n_tris * 4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        # Point data
        if point_data:
            f.write(f"POINT_DATA {len(vertices)}\n")
            for name, data in point_data.items():
                if data.ndim == 1:
                    f.write(f"SCALARS {name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.6f}\n")
                elif data.shape[1] == 3:
                    f.write(f"VECTORS {name} float\n")
                    for v in data:
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def export_mesh_stl(
    vertices: np.ndarray,
    triangles: np.ndarray,
    filename: str,
    binary: bool = True
):
    """
    Export mesh to STL format.

    Args:
        vertices: Vertex positions (N, 3)
        triangles: Triangle indices (M, 3)
        filename: Output filename (should end with .stl)
        binary: Use binary format (smaller file size)
    """
    # Compute face normals
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    if binary:
        import struct
        with open(filename, 'wb') as f:
            # Header (80 bytes)
            header = b'Cyclone mesh - binary STL'
            f.write(header.ljust(80, b'\0'))

            # Number of triangles
            f.write(struct.pack('<I', len(triangles)))

            # Triangle data
            for i, tri in enumerate(triangles):
                # Normal
                f.write(struct.pack('<3f', *normals[i]))
                # Vertices
                f.write(struct.pack('<3f', *vertices[tri[0]]))
                f.write(struct.pack('<3f', *vertices[tri[1]]))
                f.write(struct.pack('<3f', *vertices[tri[2]]))
                # Attribute byte count
                f.write(struct.pack('<H', 0))
    else:
        with open(filename, 'w') as f:
            f.write("solid cyclone\n")
            for i, tri in enumerate(triangles):
                n = normals[i]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for vi in tri:
                    v = vertices[vi]
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid cyclone\n")


def create_sampling_points(
    n_points: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    method: str = "uniform"
) -> np.ndarray:
    """
    Create sampling points within a bounding box.

    Args:
        n_points: Number of points to generate
        bounds_min: Minimum corner of bounding box
        bounds_max: Maximum corner of bounding box
        method: Sampling method ("uniform", "halton", "sobol")

    Returns:
        Array of shape (n_points, 3) with sample positions
    """
    extent = bounds_max - bounds_min

    if method == "uniform":
        points = np.random.random((n_points, 3))
        points = bounds_min + points * extent

    elif method == "halton":
        # Halton sequence for quasi-random sampling
        points = np.zeros((n_points, 3))
        bases = [2, 3, 5]  # Prime bases for each dimension

        for dim in range(3):
            points[:, dim] = _halton_sequence(n_points, bases[dim])

        points = bounds_min + points * extent

    elif method == "sobol":
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=3, scramble=True)
            points = sampler.random(n_points)
            points = bounds_min + points * extent
        except ImportError:
            # Fallback to uniform
            points = np.random.random((n_points, 3))
            points = bounds_min + points * extent
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return points.astype(np.float32)


def _halton_sequence(n: int, base: int) -> np.ndarray:
    """Generate Halton sequence."""
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        k = i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        result[i] = r
    return result
