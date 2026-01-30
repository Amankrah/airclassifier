"""
Tests for mesh generation utilities.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import tempfile
import os

from airclassifier.geometry.mesh_generator import (
    GridParams,
    MeshGenerator,
    generate_cyclone_mesh,
    export_mesh_vtk,
    export_mesh_stl,
    create_sampling_points,
)
from airclassifier.geometry.assembly import CycloneAssembly, CycloneGeometryParams


class TestGridParams:
    """Tests for GridParams dataclass."""

    def test_default_grid_params(self):
        """Test creation with default values."""
        params = GridParams(
            min_corner=(-0.2, -1.0, -0.2),
            max_corner=(0.2, 0.1, 0.2),
            nx=40,
            ny=100,
            nz=40
        )

        assert params.nx == 40
        assert params.ny == 100
        assert params.nz == 40

    def test_cell_sizes(self):
        """Test cell size calculations."""
        params = GridParams(
            min_corner=(0.0, 0.0, 0.0),
            max_corner=(1.0, 2.0, 1.0),
            nx=10,
            ny=20,
            nz=10
        )

        assert_allclose(params.dx, 0.1, rtol=1e-6)
        assert_allclose(params.dy, 0.1, rtol=1e-6)
        assert_allclose(params.dz, 0.1, rtol=1e-6)

    def test_cell_volume(self):
        """Test cell volume calculation."""
        params = GridParams(
            min_corner=(0.0, 0.0, 0.0),
            max_corner=(1.0, 1.0, 1.0),
            nx=10,
            ny=10,
            nz=10
        )

        expected_volume = 0.1 * 0.1 * 0.1
        assert_allclose(params.cell_volume, expected_volume, rtol=1e-6)

    def test_total_cells(self):
        """Test total cell count."""
        params = GridParams(
            min_corner=(0.0, 0.0, 0.0),
            max_corner=(1.0, 1.0, 1.0),
            nx=10,
            ny=20,
            nz=15
        )

        assert params.total_cells == 10 * 20 * 15

    def test_from_cyclone_bounds(self):
        """Test creation from cyclone bounding box."""
        min_corner = np.array([-0.2, -1.2, -0.2])
        max_corner = np.array([0.2, 0.1, 0.2])

        params = GridParams.from_cyclone_bounds(
            min_corner, max_corner,
            resolution=0.02
        )

        # Check bounds include padding
        assert params.min_corner[0] < min_corner[0]
        assert params.max_corner[1] > max_corner[1]

        # Check resolution is approximately correct
        assert params.dx < 0.025


class TestMeshGenerator:
    """Tests for MeshGenerator class."""

    @pytest.fixture
    def grid_params(self):
        """Create grid parameters."""
        return GridParams(
            min_corner=(-0.2, -1.0, -0.2),
            max_corner=(0.2, 0.1, 0.2),
            nx=20,
            ny=50,
            nz=20
        )

    @pytest.fixture
    def generator(self, grid_params):
        """Create mesh generator."""
        return MeshGenerator(grid_params)

    def test_generator_creation(self, generator, grid_params):
        """Test generator creation."""
        assert generator.params == grid_params

    def test_generate_structured_grid(self, generator):
        """Test structured grid generation."""
        X, Y, Z = generator.generate_structured_grid()

        # Check shapes (node-centered grid has +1 in each dimension)
        assert X.shape == (21, 51, 21)
        assert Y.shape == (21, 51, 21)
        assert Z.shape == (21, 51, 21)

        # Check bounds
        assert_allclose(X[0, 0, 0], -0.2, rtol=1e-6)
        assert_allclose(X[-1, 0, 0], 0.2, rtol=1e-6)

    def test_generate_cell_centers(self, generator):
        """Test cell center generation."""
        centers = generator.generate_cell_centers()

        # Check shape (20*50*20 cells)
        expected_cells = 20 * 50 * 20
        assert centers.shape == (expected_cells, 3)

        # Check bounds are inside domain
        assert np.all(centers[:, 0] > -0.2)
        assert np.all(centers[:, 0] < 0.2)

    def test_generate_cylindrical_grid(self, generator):
        """Test cylindrical grid generation."""
        points, connectivity = generator.generate_cylindrical_grid(
            r_min=0.02,
            r_max=0.15,
            y_min=-0.5,
            y_max=0.0,
            nr=5,
            ntheta=12,
            ny=10
        )

        # Check points shape
        assert points.shape[1] == 3

        # Check connectivity (hexahedral cells)
        assert connectivity.shape[1] == 8

        # All indices should be valid
        assert np.all(connectivity >= 0)
        assert np.all(connectivity < len(points))

    def test_combine_meshes(self):
        """Test mesh combination."""
        # Create two simple meshes
        mesh1 = (
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32)
        )
        mesh2 = (
            np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32)
        )

        combined = MeshGenerator.combine_meshes([mesh1, mesh2])

        # Check combined vertices
        assert len(combined[0]) == 6

        # Check indices are offset correctly
        assert np.all(combined[1][0] == [0, 1, 2])  # First mesh
        assert np.all(combined[1][1] == [3, 4, 5])  # Second mesh

    def test_compute_mesh_quality(self):
        """Test mesh quality computation."""
        # Create simple triangle mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.866, 0],  # Equilateral triangle
        ], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)

        quality = MeshGenerator.compute_mesh_quality(vertices, triangles)

        assert quality["num_vertices"] == 3
        assert quality["num_triangles"] == 1
        assert quality["total_area"] > 0
        assert quality["mean_aspect_ratio"] >= 1.0


class TestGenerateCycloneMesh:
    """Tests for generate_cyclone_mesh function."""

    def test_generates_valid_mesh(self):
        """Test that function generates valid mesh."""
        assembly = CycloneAssembly(
            CycloneGeometryParams.from_diameter(0.3),
            device="cpu"
        )

        vertices, triangles = generate_cyclone_mesh(assembly, resolution=0.01)

        # Check shapes
        assert vertices.shape[1] == 3
        assert triangles.shape[1] == 3

        # Check indices valid
        assert np.all(triangles >= 0)
        assert np.all(triangles < len(vertices))


class TestMeshExport:
    """Tests for mesh export functions."""

    @pytest.fixture
    def simple_mesh(self):
        """Create simple test mesh."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        triangles = np.array([
            [0, 1, 2],
            [0, 1, 3],
        ], dtype=np.int32)
        return vertices, triangles

    def test_export_vtk_legacy(self, simple_mesh):
        """Test VTK legacy format export."""
        vertices, triangles = simple_mesh

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            filename = f.name

        try:
            export_mesh_vtk(vertices, triangles, filename)

            # Check file was created
            assert os.path.exists(filename)

            # Check file has content
            with open(filename, 'r') as f:
                content = f.read()
                assert 'vtk' in content.lower()
                assert 'POINTS' in content
                assert 'POLYGONS' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_vtk_with_data(self, simple_mesh):
        """Test VTK export with point data."""
        vertices, triangles = simple_mesh

        point_data = {
            "pressure": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            filename = f.name

        try:
            export_mesh_vtk(vertices, triangles, filename, point_data=point_data)

            with open(filename, 'r') as f:
                content = f.read()
                assert 'POINT_DATA' in content
                assert 'pressure' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_stl_ascii(self, simple_mesh):
        """Test STL ASCII format export."""
        vertices, triangles = simple_mesh

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            export_mesh_stl(vertices, triangles, filename, binary=False)

            assert os.path.exists(filename)

            with open(filename, 'r') as f:
                content = f.read()
                assert 'solid' in content
                assert 'facet normal' in content
                assert 'endsolid' in content
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_export_stl_binary(self, simple_mesh):
        """Test STL binary format export."""
        vertices, triangles = simple_mesh

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            filename = f.name

        try:
            export_mesh_stl(vertices, triangles, filename, binary=True)

            assert os.path.exists(filename)

            # Binary STL has 80 byte header + 4 byte count + 50 bytes per triangle
            file_size = os.path.getsize(filename)
            expected_size = 80 + 4 + 50 * len(triangles)
            assert file_size == expected_size
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestSamplingPoints:
    """Tests for create_sampling_points function."""

    def test_uniform_sampling(self):
        """Test uniform random sampling."""
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([1.0, 1.0, 1.0])

        points = create_sampling_points(1000, bounds_min, bounds_max, method="uniform")

        assert points.shape == (1000, 3)

        # Check bounds
        assert np.all(points >= bounds_min)
        assert np.all(points <= bounds_max)

    def test_halton_sampling(self):
        """Test Halton sequence sampling."""
        bounds_min = np.array([-1.0, -1.0, -1.0])
        bounds_max = np.array([1.0, 1.0, 1.0])

        points = create_sampling_points(500, bounds_min, bounds_max, method="halton")

        assert points.shape == (500, 3)

        # Check bounds
        assert np.all(points >= bounds_min)
        assert np.all(points <= bounds_max)

        # Halton should have better coverage than random
        # (lower discrepancy)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError):
            create_sampling_points(100, bounds_min, bounds_max, method="invalid")
