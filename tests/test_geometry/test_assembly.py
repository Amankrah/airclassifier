"""
Tests for CycloneAssembly class.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.assembly import (
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
)
from airclassifier.utils.constants import PI


class TestCycloneGeometryParams:
    """Tests for CycloneGeometryParams dataclass."""

    def test_from_diameter_creates_standard_proportions(self):
        """Test that from_diameter creates correct proportions."""
        D = 0.3
        params = CycloneGeometryParams.from_diameter(D)

        # Standard proportions
        assert_allclose(params.cylinder_diameter, D, rtol=1e-6)
        assert_allclose(params.cylinder_height, 1.5 * D, rtol=1e-6)
        assert_allclose(params.cone_height, 2.5 * D, rtol=1e-6)
        assert_allclose(params.inlet_width, 0.25 * D, rtol=1e-6)
        assert_allclose(params.inlet_height, 0.5 * D, rtol=1e-6)
        assert_allclose(params.vortex_finder_diameter, 0.5 * D, rtol=1e-6)

    def test_total_height(self):
        """Test total height property."""
        params = CycloneGeometryParams.from_diameter(0.3)
        expected = params.cylinder_height + params.cone_height
        assert_allclose(params.total_height, expected, rtol=1e-6)

    def test_aspect_ratio(self):
        """Test aspect ratio property."""
        params = CycloneGeometryParams.from_diameter(0.3)
        expected = params.total_height / params.cylinder_diameter
        assert_allclose(params.aspect_ratio, expected, rtol=1e-6)

    def test_custom_parameters(self):
        """Test that custom parameters override defaults."""
        D = 0.3
        params = CycloneGeometryParams.from_diameter(
            D,
            cylinder_height=2.0 * D,  # Override
            inlet_width=0.3 * D  # Override
        )

        assert_allclose(params.cylinder_height, 2.0 * D, rtol=1e-6)
        assert_allclose(params.inlet_width, 0.3 * D, rtol=1e-6)
        # Others should keep defaults
        assert_allclose(params.cone_height, 2.5 * D, rtol=1e-6)


class TestCycloneAssembly:
    """Tests for CycloneAssembly class."""

    @pytest.fixture
    def assembly(self):
        """Create standard cyclone assembly."""
        params = CycloneGeometryParams.from_diameter(0.3)
        return CycloneAssembly(params, device="cpu")

    def test_assembly_creation(self, assembly):
        """Test assembly is created with all components."""
        assert assembly.body is not None
        assert assembly.inlet is not None
        assert assembly.vortex_finder is not None
        assert assembly.dust_outlet is not None
        assert assembly.overflow is not None

    def test_build_mesh(self, assembly):
        """Test mesh building combines all components."""
        vertices, indices = assembly.build_mesh()

        # Should have significant number of vertices
        assert len(vertices) > 100

        # Check shapes
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        # Indices are returned as flat array for Warp compatibility
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

        # Check indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_get_bounds(self, assembly):
        """Test bounding box calculation."""
        min_corner, max_corner = assembly.get_bounds()

        # Check dimensions
        assert len(min_corner) == 3
        assert len(max_corner) == 3

        # max should be greater than min
        assert np.all(max_corner > min_corner)

        # Check approximate sizes
        D = assembly.params.cylinder_diameter
        extent = max_corner - min_corner

        # X and Z extent should be roughly diameter (possibly with inlet)
        assert extent[0] >= D
        assert extent[2] >= D

        # Y extent should be total height (plus margins)
        assert extent[1] >= assembly.params.total_height

    def test_get_inlet_conditions(self, assembly):
        """Test inlet boundary condition parameters."""
        inlet_cond = assembly.get_inlet_conditions()

        assert "position" in inlet_cond
        assert "direction" in inlet_cond
        assert "width" in inlet_cond
        assert "height" in inlet_cond
        assert "area" in inlet_cond

        # Direction should be unit vector
        direction = np.array(inlet_cond["direction"])
        assert_allclose(np.linalg.norm(direction), 1.0, rtol=1e-6)

        # Area should match width * height
        expected_area = inlet_cond["width"] * inlet_cond["height"]
        assert_allclose(inlet_cond["area"], expected_area, rtol=1e-6)

    def test_get_outlet_conditions(self, assembly):
        """Test outlet boundary condition parameters."""
        outlet_cond = assembly.get_outlet_conditions()

        assert "overflow" in outlet_cond
        assert "underflow" in outlet_cond

    def test_classify_position_outside(self, assembly):
        """Test position classification - outside."""
        # Far from cyclone
        point = np.array([1.0, 0.0, 0.0])
        region = assembly.classify_position(point)
        assert region == "outside"

    def test_classify_position_cylinder(self, assembly):
        """Test position classification - cylinder region."""
        # Center of cylinder
        D = assembly.params.cylinder_diameter
        y = -assembly.params.cylinder_height / 2
        point = np.array([0.0, y, 0.0])
        region = assembly.classify_position(point)
        assert region == "cylinder"

    def test_classify_position_cone(self, assembly):
        """Test position classification - cone region."""
        # Center of cone
        y = -assembly.params.cylinder_height - assembly.params.cone_height / 2
        point = np.array([0.0, y, 0.0])
        region = assembly.classify_position(point)
        assert region == "cone"

    def test_classify_position_vortex_finder(self, assembly):
        """Test position classification - vortex finder region."""
        # Center of VF insertion region
        point = np.array([0.0, -assembly.params.vortex_finder_length / 2, 0.0])
        region = assembly.classify_position(point)
        assert region == "vortex_finder"


class TestCreateStandardCyclone:
    """Tests for create_standard_cyclone function."""

    def test_creates_valid_assembly(self):
        """Test that function creates valid assembly."""
        assembly = create_standard_cyclone(0.3, device="cpu")

        assert isinstance(assembly, CycloneAssembly)
        assert assembly.params.cylinder_diameter == 0.3

    def test_different_sizes(self):
        """Test creation with different diameters."""
        sizes = [0.1, 0.3, 0.5, 1.0]

        for D in sizes:
            assembly = create_standard_cyclone(D, device="cpu")
            assert assembly.params.cylinder_diameter == D

            # Proportions should be consistent
            assert_allclose(
                assembly.params.cylinder_height / D, 1.5, rtol=1e-6
            )


class TestAssemblyMeshQuality:
    """Tests for mesh quality in assembly."""

    @pytest.fixture
    def assembly(self):
        """Create assembly for testing."""
        params = CycloneGeometryParams.from_diameter(0.3)
        return CycloneAssembly(params, device="cpu")

    def test_no_degenerate_triangles(self, assembly):
        """Test that mesh has no zero-area triangles."""
        vertices, indices = assembly.build_mesh()

        # Reshape flat indices to triangles
        triangles = indices.reshape(-1, 3)

        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]

            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

            assert area > 1e-12, f"Degenerate triangle: {tri}"

    def test_mesh_extent_matches_params(self, assembly):
        """Test that mesh extent matches geometry parameters."""
        vertices, _ = assembly.build_mesh()

        # Find mesh extent
        mesh_min = vertices.min(axis=0)
        mesh_max = vertices.max(axis=0)

        # Check radial extent roughly matches diameter
        x_extent = mesh_max[0] - mesh_min[0]
        z_extent = mesh_max[2] - mesh_min[2]
        D = assembly.params.cylinder_diameter

        # Should be at least diameter (might be larger due to inlet)
        assert x_extent >= D * 0.9
        assert z_extent >= D * 0.9

    def test_print_summary_runs(self, assembly, capsys):
        """Test that print_summary produces output."""
        assembly.print_summary()

        captured = capsys.readouterr()
        assert "Cyclone" in captured.out
        assert "mm" in captured.out
