"""
Tests for Signed Distance Function (SDF) utilities.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.sdf import (
    CycloneSDF,
    CycloneSDFParams,
    SDFField,
    create_cyclone_sdf,
)
from airclassifier.geometry.assembly import CycloneAssembly, CycloneGeometryParams


class TestCycloneSDFParams:
    """Tests for CycloneSDFParams dataclass."""

    def test_default_params(self):
        """Test default parameter creation."""
        params = CycloneSDFParams()

        assert params.cylinder_radius > 0
        assert params.cylinder_height > 0
        assert params.cone_height > 0

    def test_from_assembly(self):
        """Test creation from CycloneAssembly."""
        assembly = CycloneAssembly(
            CycloneGeometryParams.from_diameter(0.3),
            device="cpu"
        )

        sdf_params = CycloneSDFParams.from_assembly(assembly)

        # Check values match
        assert_allclose(
            sdf_params.cylinder_radius,
            assembly.params.cylinder_diameter / 2,
            rtol=1e-6
        )
        assert_allclose(
            sdf_params.cylinder_height,
            assembly.params.cylinder_height,
            rtol=1e-6
        )

    def test_total_height(self):
        """Test total height property."""
        params = CycloneSDFParams(
            cylinder_height=0.45,
            cone_height=0.75
        )
        assert_allclose(params.total_height, 1.2, rtol=1e-6)


class TestCycloneSDF:
    """Tests for CycloneSDF class."""

    @pytest.fixture
    def sdf(self):
        """Create SDF instance with default parameters."""
        params = CycloneSDFParams(
            center=(0.0, 0.0, 0.0),
            cylinder_radius=0.15,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_bottom_radius=0.05625,
            vortex_finder_radius=0.075,
            vortex_finder_bottom_y=0.15,
        )
        return CycloneSDF(params)

    def test_sdf_creation(self, sdf):
        """Test SDF creation."""
        assert sdf is not None
        assert sdf.params.cylinder_radius == 0.15

    def test_sdf_inside_cylinder(self, sdf):
        """Test SDF value inside cylinder."""
        # Point at cylinder center
        point = np.array([0.0, -0.2, 0.0])
        dist = sdf.evaluate(point)

        # Should be negative (inside)
        assert dist < 0

        # Distance should be roughly the radius
        assert_allclose(abs(dist), 0.15, atol=0.01)

    def test_sdf_on_cylinder_wall(self, sdf):
        """Test SDF value on cylinder wall."""
        # Point on cylinder wall
        point = np.array([0.15, -0.2, 0.0])
        dist = sdf.evaluate(point)

        # Should be approximately zero
        assert_allclose(dist, 0.0, atol=0.001)

    def test_sdf_outside_cylinder(self, sdf):
        """Test SDF value outside cylinder."""
        # Point outside cylinder
        point = np.array([0.2, -0.2, 0.0])
        dist = sdf.evaluate(point)

        # Should be positive (outside)
        assert dist > 0

    def test_sdf_in_cone(self, sdf):
        """Test SDF value in cone section."""
        # Point at cone center
        y = -sdf.params.cylinder_height - sdf.params.cone_height / 2
        point = np.array([0.0, y, 0.0])
        dist = sdf.evaluate(point)

        # Should be negative (inside)
        assert dist < 0

    def test_sdf_in_vortex_finder(self, sdf):
        """Test SDF value in vortex finder region."""
        # Point inside vortex finder
        point = np.array([0.0, -0.05, 0.0])  # Near top, inside VF
        dist = sdf.evaluate(point)

        # Should be negative (inside VF tube)
        assert dist < 0

    def test_sdf_gradient(self, sdf):
        """Test gradient calculation."""
        point = np.array([0.1, -0.2, 0.0])
        grad = sdf.gradient(point)

        # Gradient should be non-zero vector
        assert np.linalg.norm(grad) > 0.1

    def test_sdf_normal(self, sdf):
        """Test surface normal calculation."""
        # Point on cylinder wall
        point = np.array([0.15, -0.2, 0.0])
        normal = sdf.normal(point)

        # Should be unit vector
        assert_allclose(np.linalg.norm(normal), 1.0, rtol=1e-5)

        # Should point outward (radially for cylinder)
        # X component should be positive for point on +X wall
        assert normal[0] > 0.9

    def test_is_inside(self, sdf):
        """Test is_inside method."""
        inside_point = np.array([0.0, -0.2, 0.0])
        outside_point = np.array([0.2, -0.2, 0.0])

        assert sdf.is_inside(inside_point)
        assert not sdf.is_inside(outside_point)

    def test_classify_region_cylinder(self, sdf):
        """Test region classification - cylinder."""
        point = np.array([0.05, -0.2, 0.0])
        region = sdf.classify_region(point)
        assert region == "cylinder"

    def test_classify_region_cone(self, sdf):
        """Test region classification - cone."""
        y = -sdf.params.cylinder_height - sdf.params.cone_height / 2
        point = np.array([0.0, y, 0.0])
        region = sdf.classify_region(point)
        assert region == "cone"

    def test_classify_region_vortex_finder(self, sdf):
        """Test region classification - vortex finder."""
        point = np.array([0.0, -0.05, 0.0])
        region = sdf.classify_region(point)
        assert region == "vortex_finder"

    def test_classify_region_outside(self, sdf):
        """Test region classification - outside."""
        point = np.array([0.5, -0.2, 0.0])
        region = sdf.classify_region(point)
        assert region == "outside"

    def test_evaluate_batch(self, sdf):
        """Test batch evaluation."""
        points = np.array([
            [0.0, -0.2, 0.0],
            [0.15, -0.2, 0.0],
            [0.2, -0.2, 0.0],
        ])

        distances = sdf.evaluate_batch(points)

        assert len(distances) == 3
        assert distances[0] < 0  # Inside
        assert_allclose(distances[1], 0.0, atol=0.001)  # On surface
        assert distances[2] > 0  # Outside


class TestSDFField:
    """Tests for SDFField class."""

    @pytest.fixture
    def sdf_field(self):
        """Create SDF field instance."""
        params = CycloneSDFParams(
            center=(0.0, 0.0, 0.0),
            cylinder_radius=0.15,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_bottom_radius=0.05625,
            vortex_finder_radius=0.075,
            vortex_finder_bottom_y=0.15,
        )
        sdf = CycloneSDF(params)

        bounds_min = np.array([-0.2, -0.5, -0.2])
        bounds_max = np.array([0.2, 0.1, 0.2])

        return SDFField(sdf, bounds_min, bounds_max, resolution=20)

    def test_field_creation(self, sdf_field):
        """Test field creation."""
        assert sdf_field.resolution == (20, 20, 20)

    def test_field_compute(self, sdf_field):
        """Test field computation."""
        field = sdf_field.compute(device="cpu")

        # Check shape
        assert field.shape == (20, 20, 20)

        # Should have both positive and negative values
        assert np.min(field) < 0  # Inside points
        assert np.max(field) > 0  # Outside points

    def test_field_interpolate(self, sdf_field):
        """Test trilinear interpolation."""
        sdf_field.compute(device="cpu")

        # Interpolate at grid point
        point = np.array([0.0, -0.2, 0.0])
        interp_val = sdf_field.interpolate(point)

        # Compare with direct SDF evaluation
        direct_val = sdf_field.sdf.evaluate(point)

        # Should be reasonably close (not exact due to grid resolution)
        assert_allclose(interp_val, direct_val, atol=0.05)

    def test_get_cell_size(self, sdf_field):
        """Test cell size calculation."""
        cell_size = sdf_field.get_cell_size()

        assert len(cell_size) == 3
        assert np.all(cell_size > 0)

        # Check consistency with resolution
        extent = sdf_field.bounds_max - sdf_field.bounds_min
        expected_size = extent / (np.array(sdf_field.resolution) - 1)
        assert_allclose(cell_size, expected_size, rtol=1e-6)


class TestCreateCycloneSDF:
    """Tests for create_cyclone_sdf function."""

    def test_creates_valid_sdf(self):
        """Test that function creates valid SDF."""
        assembly = CycloneAssembly(
            CycloneGeometryParams.from_diameter(0.3),
            device="cpu"
        )

        sdf = create_cyclone_sdf(assembly)

        assert isinstance(sdf, CycloneSDF)

        # Test evaluation
        point = np.array([0.0, -0.2, 0.0])
        dist = sdf.evaluate(point)
        assert dist < 0  # Should be inside


class TestSDFSymmetry:
    """Tests for SDF axial symmetry."""

    @pytest.fixture
    def sdf(self):
        """Create SDF instance."""
        params = CycloneSDFParams(
            center=(0.0, 0.0, 0.0),
            cylinder_radius=0.15,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_bottom_radius=0.05625,
        )
        return CycloneSDF(params)

    def test_rotational_symmetry(self, sdf):
        """Test that SDF is rotationally symmetric about Y axis."""
        y = -0.2
        r = 0.1

        # Test points at same radius but different angles
        angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        distances = []

        for theta in angles:
            point = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            distances.append(sdf.evaluate(point))

        # All distances should be equal
        assert_allclose(distances, distances[0], rtol=1e-6)

    def test_symmetry_in_cone(self, sdf):
        """Test rotational symmetry in cone section."""
        y = -sdf.params.cylinder_height - sdf.params.cone_height / 2
        r = 0.05

        angles = [0, np.pi/3, 2*np.pi/3, np.pi]
        distances = []

        for theta in angles:
            point = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            distances.append(sdf.evaluate(point))

        assert_allclose(distances, distances[0], rtol=1e-6)
