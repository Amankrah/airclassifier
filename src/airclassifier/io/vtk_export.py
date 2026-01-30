"""
VTK export utilities for ParaView visualization.

Provides functions to export particle data and mesh geometry
to VTK format for visualization in ParaView or other VTK-compatible
software.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


def write_vtk_particles(
    filename: str,
    positions: np.ndarray,
    diameters: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    is_active: Optional[np.ndarray] = None,
    additional_scalars: Optional[Dict[str, np.ndarray]] = None,
    additional_vectors: Optional[Dict[str, np.ndarray]] = None
):
    """
    Write particle data to VTK polydata file (.vtp or .vtk).

    Args:
        filename: Output filename (should end in .vtp or .vtk)
        positions: Particle positions array (N, 3)
        diameters: Particle diameters (N,)
        velocities: Particle velocities (N, 3)
        is_active: Particle active status (N,)
        additional_scalars: Dict of name -> scalar array
        additional_vectors: Dict of name -> vector array (N, 3)
    """
    n_particles = len(positions)

    # Filter out invalid positions
    valid_mask = np.all(np.isfinite(positions), axis=1)
    if is_active is not None:
        valid_mask &= (is_active != 0)

    positions = positions[valid_mask]
    n_valid = len(positions)

    if n_valid == 0:
        print(f"Warning: No valid particles to export to {filename}")
        return

    # Determine format from extension
    path = Path(filename)
    use_xml = path.suffix.lower() == '.vtp'

    if use_xml:
        _write_vtp(filename, positions, diameters, velocities, is_active,
                  additional_scalars, additional_vectors, valid_mask)
    else:
        _write_vtk_legacy(filename, positions, diameters, velocities, is_active,
                         additional_scalars, additional_vectors, valid_mask)


def _write_vtk_legacy(
    filename: str,
    positions: np.ndarray,
    diameters: Optional[np.ndarray],
    velocities: Optional[np.ndarray],
    is_active: Optional[np.ndarray],
    additional_scalars: Optional[Dict[str, np.ndarray]],
    additional_vectors: Optional[Dict[str, np.ndarray]],
    valid_mask: np.ndarray
):
    """Write legacy VTK format."""
    n = len(positions)

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Particle data from cyclone simulation\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n} float\n")
        for p in positions:
            f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")

        # Vertices (one per point)
        f.write(f"\nVERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        # Point data
        f.write(f"\nPOINT_DATA {n}\n")

        # Diameters
        if diameters is not None:
            d_valid = diameters[valid_mask]
            f.write("SCALARS diameter float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for d in d_valid:
                f.write(f"{d:.6e}\n")

        # Active status
        if is_active is not None:
            a_valid = is_active[valid_mask]
            f.write("SCALARS status int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for a in a_valid:
                f.write(f"{int(a)}\n")

        # Velocities
        if velocities is not None:
            v_valid = velocities[valid_mask]
            f.write("VECTORS velocity float\n")
            for v in v_valid:
                f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

        # Additional scalars
        if additional_scalars:
            for name, data in additional_scalars.items():
                d_valid = data[valid_mask]
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in d_valid:
                    f.write(f"{val:.6e}\n")

        # Additional vectors
        if additional_vectors:
            for name, data in additional_vectors.items():
                v_valid = data[valid_mask]
                f.write(f"VECTORS {name} float\n")
                for v in v_valid:
                    f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")


def _write_vtp(
    filename: str,
    positions: np.ndarray,
    diameters: Optional[np.ndarray],
    velocities: Optional[np.ndarray],
    is_active: Optional[np.ndarray],
    additional_scalars: Optional[Dict[str, np.ndarray]],
    additional_vectors: Optional[Dict[str, np.ndarray]],
    valid_mask: np.ndarray
):
    """Write XML VTK polydata format (.vtp)."""
    n = len(positions)

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfVerts="{n}">\n')

        # Point data
        f.write('      <PointData>\n')

        if diameters is not None:
            d_valid = diameters[valid_mask]
            f.write('        <DataArray type="Float32" Name="diameter" format="ascii">\n')
            f.write('          ')
            for d in d_valid:
                f.write(f'{d:.6e} ')
            f.write('\n        </DataArray>\n')

        if is_active is not None:
            a_valid = is_active[valid_mask]
            f.write('        <DataArray type="Int32" Name="status" format="ascii">\n')
            f.write('          ')
            for a in a_valid:
                f.write(f'{int(a)} ')
            f.write('\n        </DataArray>\n')

        if velocities is not None:
            v_valid = velocities[valid_mask]
            f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
            f.write('          ')
            for v in v_valid:
                f.write(f'{v[0]:.6e} {v[1]:.6e} {v[2]:.6e} ')
            f.write('\n        </DataArray>\n')

        if additional_scalars:
            for name, data in additional_scalars.items():
                d_valid = data[valid_mask]
                f.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
                f.write('          ')
                for val in d_valid:
                    f.write(f'{val:.6e} ')
                f.write('\n        </DataArray>\n')

        if additional_vectors:
            for name, data in additional_vectors.items():
                v_valid = data[valid_mask]
                f.write(f'        <DataArray type="Float32" Name="{name}" NumberOfComponents="3" format="ascii">\n')
                f.write('          ')
                for v in v_valid:
                    f.write(f'{v[0]:.6e} {v[1]:.6e} {v[2]:.6e} ')
                f.write('\n        </DataArray>\n')

        f.write('      </PointData>\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write('          ')
        for p in positions:
            f.write(f'{p[0]:.6e} {p[1]:.6e} {p[2]:.6e} ')
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')

        # Vertices
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write('          ')
        for i in range(n):
            f.write(f'{i} ')
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write('          ')
        for i in range(1, n + 1):
            f.write(f'{i} ')
        f.write('\n        </DataArray>\n')
        f.write('      </Verts>\n')

        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')


def write_vtk_mesh(
    filename: str,
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: Optional[np.ndarray] = None
):
    """
    Write triangle mesh to VTK file.

    Args:
        filename: Output filename (.vtk or .vtu)
        vertices: Vertex positions (N, 3)
        indices: Triangle indices (M, 3) or flat array (M*3,)
        normals: Vertex normals (N, 3)
    """
    n_verts = len(vertices)

    # Reshape indices if flat
    if indices.ndim == 1:
        indices = indices.reshape(-1, 3)
    n_tris = len(indices)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Cyclone mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n_verts} float\n")
        for v in vertices:
            f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

        # Polygons (triangles)
        f.write(f"\nPOLYGONS {n_tris} {n_tris * 4}\n")
        for tri in indices:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

        # Normals
        if normals is not None:
            f.write(f"\nPOINT_DATA {n_verts}\n")
            f.write("NORMALS normals float\n")
            for n in normals:
                f.write(f"{n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")


def write_vtk_time_series(
    base_filename: str,
    time_steps: List[float],
    positions_list: List[np.ndarray],
    diameters: Optional[np.ndarray] = None,
    velocities_list: Optional[List[np.ndarray]] = None,
    is_active_list: Optional[List[np.ndarray]] = None
):
    """
    Write time series of particle data for animation.

    Creates numbered files and a .pvd collection file.

    Args:
        base_filename: Base name for output files (without extension)
        time_steps: List of time values
        positions_list: List of position arrays for each time step
        diameters: Particle diameters (constant)
        velocities_list: List of velocity arrays
        is_active_list: List of active status arrays
    """
    base_path = Path(base_filename)
    output_dir = base_path.parent
    base_name = base_path.stem

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write individual time step files
    vtp_files = []
    for i, (t, positions) in enumerate(zip(time_steps, positions_list)):
        vtp_filename = output_dir / f"{base_name}_{i:04d}.vtp"
        vtp_files.append(vtp_filename.name)

        velocities = velocities_list[i] if velocities_list else None
        is_active = is_active_list[i] if is_active_list else None

        write_vtk_particles(
            str(vtp_filename),
            positions,
            diameters=diameters,
            velocities=velocities,
            is_active=is_active
        )

    # Write PVD collection file
    pvd_filename = output_dir / f"{base_name}.pvd"
    with open(pvd_filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')

        for i, (t, vtp_file) in enumerate(zip(time_steps, vtp_files)):
            f.write(f'    <DataSet timestep="{t:.6e}" file="{vtp_file}"/>\n')

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    print(f"Wrote {len(vtp_files)} VTP files and collection: {pvd_filename}")


def export_simulation_results(
    output_dir: str,
    positions: np.ndarray,
    diameters: np.ndarray,
    velocities: np.ndarray,
    is_active: np.ndarray,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_indices: Optional[np.ndarray] = None,
    prefix: str = "cyclone"
):
    """
    Export complete simulation results to VTK files.

    Args:
        output_dir: Output directory
        positions: Particle positions
        diameters: Particle diameters
        velocities: Particle velocities
        is_active: Particle status
        mesh_vertices: Optional cyclone mesh vertices
        mesh_indices: Optional cyclone mesh indices
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export particles
    particles_file = output_path / f"{prefix}_particles.vtp"
    write_vtk_particles(
        str(particles_file),
        positions,
        diameters=diameters,
        velocities=velocities,
        is_active=is_active
    )
    print(f"Exported particles to: {particles_file}")

    # Export mesh if provided
    if mesh_vertices is not None and mesh_indices is not None:
        mesh_file = output_path / f"{prefix}_mesh.vtk"
        write_vtk_mesh(str(mesh_file), mesh_vertices, mesh_indices)
        print(f"Exported mesh to: {mesh_file}")

    # Export separated particles (collected vs escaped)
    collected_mask = is_active == -1
    escaped_mask = is_active == -2

    if np.any(collected_mask):
        collected_file = output_path / f"{prefix}_collected.vtp"
        write_vtk_particles(
            str(collected_file),
            positions[collected_mask],
            diameters=diameters[collected_mask] if diameters is not None else None,
            velocities=velocities[collected_mask] if velocities is not None else None
        )
        print(f"Exported collected particles to: {collected_file}")

    if np.any(escaped_mask):
        escaped_file = output_path / f"{prefix}_escaped.vtp"
        write_vtk_particles(
            str(escaped_file),
            positions[escaped_mask],
            diameters=diameters[escaped_mask] if diameters is not None else None,
            velocities=velocities[escaped_mask] if velocities is not None else None
        )
        print(f"Exported escaped particles to: {escaped_file}")
