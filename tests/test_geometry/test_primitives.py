"""
Tests for geometry primitives (cylinder, cone, tube).
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from airclassifier.geometry.primitives import (
    Cylinder, CylinderParams,
    Cone, ConeParams,
    Tube, TubeParams,
)
from airclassifier.utils.constants import PI


class TestCylinder:
    """Tests for Cylinder primitive."""

    @pytest.fixture
    def cylinder_params(self):
        """Create default cylinder parameters."""
        return CylinderParams(
            radius=0.1,
            height=0.3,
            center=(0.0, 0.0, 0.0),
            axis="y",
            resolution_radial=24,
            resolution_axial=10
        )

    @pytest.fixture
    def cylinder(self, cylinder_params):
        """Create cylinder instance."""
        return Cylinder(cylinder_params)

    def test_cylinder_creation(self, cylinder):
        """Test cylinder is created with correct parameters."""
        assert cylinder.params.radius == 0.1
        assert cylinder.params.height == 0.3
        assert cylinder.params.axis == "y"

    def test_cylinder_volume(self, cylinder):
        """Test cylinder volume calculation."""
        expected_volume = PI * 0.1**2 * 0.3
        assert_allclose(cylinder.volume, expected_volume, rtol=1e-6)

    def test_cylinder_surface_area(self, cylinder):
        """Test cylinder surface area calculation."""
        # Lateral surface area only (no caps)
        expected_area = 2 * PI * 0.1 * 0.3
        assert_allclose(cylinder.lateral_surface_area, expected_area, rtol=1e-6)

    def test_cylinder_mesh_generation(self, cylinder):
        """Test mesh generation produces valid output."""
        vertices, indices, normals = cylinder.generate_mesh()

        # Check shapes
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 2
        assert indices.shape[1] == 3
        assert normals.shape == vertices.shape

        # Check normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert_allclose(norms, 1.0, rtol=1e-5)

        # Check indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_cylinder_mesh_vertices_on_surface(self, cylinder):
        """Test that mesh vertices lie on cylinder surface."""
        vertices, _, _ = cylinder.generate_mesh()

        # For Y-axis cylinder, radial distance should equal radius
        r = np.sqrt(vertices[:, 0]**2 + vertices[:, 2]**2)
        assert_allclose(r, cylinder.params.radius, rtol=1e-5)

    def test_cylinder_normals_point_outward(self, cylinder):
        """Test that normals point outward from cylinder."""
        vertices, _, normals = cylinder.generate_mesh()

        # For Y-axis cylinder, radial component of normal should be positive
        # (pointing away from axis)
        for i, (v, n) in enumerate(zip(vertices, normals)):
            r_vec = np.array([v[0], 0, v[2]])
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 1e-6:
                r_unit = r_vec / r_mag
                # Normal should have positive component in radial direction
                dot = np.dot(n, r_unit)
                assert dot > 0.9, f"Normal at vertex {i} points inward"


class TestCone:
    """Tests for Cone (frustum) primitive."""

    @pytest.fixture
    def cone_params(self):
        """Create default cone parameters."""
        return ConeParams(
            top_radius=0.15,
            bottom_radius=0.05,
            height=0.4,
            center=(0.0, 0.0, 0.0),
            axis="y",
            resolution_radial=24,
            resolution_axial=16
        )

    @pytest.fixture
    def cone(self, cone_params):
        """Create cone instance."""
        return Cone(cone_params)

    def test_cone_creation(self, cone):
        """Test cone is created with correct parameters."""
        assert cone.params.top_radius == 0.15
        assert cone.params.bottom_radius == 0.05
        assert cone.params.height == 0.4

    def test_cone_volume(self, cone):
        """Test frustum volume calculation."""
        r1, r2, h = 0.15, 0.05, 0.4
        expected_volume = (PI * h / 3.0) * (r1**2 + r1*r2 + r2**2)
        assert_allclose(cone.volume, expected_volume, rtol=1e-6)

    def test_cone_slant_height(self, cone):
        """Test slant height calculation."""
        dr = 0.15 - 0.05  # Difference in radii
        h = 0.4
        expected_slant = np.sqrt(h**2 + dr**2)
        assert_allclose(cone.slant_height, expected_slant, rtol=1e-6)

    def test_cone_half_angle(self, cone):
        """Test half-angle calculation."""
        dr = 0.15 - 0.05
        h = 0.4
        expected_angle = np.arctan2(dr, h)
        assert_allclose(cone.half_angle, expected_angle, rtol=1e-6)

    def test_cone_mesh_generation(self, cone):
        """Test mesh generation produces valid output."""
        vertices, indices, normals = cone.generate_mesh()

        # Check shapes
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 2
        assert indices.shape[1] == 3
        assert normals.shape == vertices.shape

        # Check normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        assert_allclose(norms, 1.0, rtol=1e-5)

    def test_cone_radius_interpolation(self, cone):
        """Test radius varies linearly along height."""
        vertices, _, _ = cone.generate_mesh()

        # Get unique y values
        y_values = np.unique(np.round(vertices[:, 1], 6))

        for y in y_values:
            mask = np.abs(vertices[:, 1] - y) < 1e-6
            verts_at_y = vertices[mask]

            # Calculate radial distances
            r = np.sqrt(verts_at_y[:, 0]**2 + verts_at_y[:, 2]**2)

            # Expected radius at this height
            # Assuming y=0 is top, y=-height is bottom
            t = -y / cone.params.height  # Normalized position
            expected_r = cone.params.top_radius * (1 - t) + cone.params.bottom_radius * t

            assert_allclose(r, expected_r, rtol=1e-4)


class TestTube:
    """Tests for Tube (hollow cylinder) primitive."""

    @pytest.fixture
    def tube_params(self):
        """Create default tube parameters."""
        return TubeParams(
            inner_radius=0.03,
            outer_radius=0.05,
            height=0.2,
            center=(0.0, 0.0, 0.0),
            axis="y",
            resolution_radial=24,
            resolution_axial=8
        )

    @pytest.fixture
    def tube(self, tube_params):
        """Create tube instance."""
        return Tube(tube_params)

    def test_tube_creation(self, tube):
        """Test tube is created with correct parameters."""
        assert tube.params.inner_radius == 0.03
        assert tube.params.outer_radius == 0.05
        assert tube.params.height == 0.2

    def test_tube_wall_thickness(self, tube):
        """Test wall thickness calculation."""
        expected_thickness = 0.05 - 0.03
        assert_allclose(tube.wall_thickness, expected_thickness, rtol=1e-6)

    def test_tube_volume(self, tube):
        """Test tube volume (material volume)."""
        expected_volume = PI * (0.05**2 - 0.03**2) * 0.2
        assert_allclose(tube.volume, expected_volume, rtol=1e-6)

    def test_tube_mesh_has_inner_and_outer_surfaces(self, tube):
        """Test tube mesh includes both inner and outer surfaces."""
        vertices, _, _ = tube.generate_mesh()

        # Calculate radial distances
        r = np.sqrt(vertices[:, 0]**2 + vertices[:, 2]**2)

        # Should have vertices at both inner and outer radii
        inner_verts = np.sum(np.abs(r - tube.params.inner_radius) < 1e-5)
        outer_verts = np.sum(np.abs(r - tube.params.outer_radius) < 1e-5)

        assert inner_verts > 0, "No vertices found at inner radius"
        assert outer_verts > 0, "No vertices found at outer radius"


class TestMeshQuality:
    """Tests for mesh quality across primitives."""

    @pytest.fixture
    def primitives(self):
        """Create list of primitives to test."""
        cylinder = Cylinder(CylinderParams(
            radius=0.1, height=0.3, resolution_radial=24, resolution_axial=10
        ))
        cone = Cone(ConeParams(
            top_radius=0.15, bottom_radius=0.05, height=0.4,
            resolution_radial=24, resolution_axial=16
        ))
        tube = Tube(TubeParams(
            inner_radius=0.03, outer_radius=0.05, height=0.2,
            resolution_radial=24, resolution_axial=8
        ))
        return [cylinder, cone, tube]

    def test_mesh_watertight(self, primitives):
        """Test that meshes have consistent edge connectivity."""
        for primitive in primitives:
            vertices, indices, _ = primitive.generate_mesh()

            # Count edge usage (each edge should appear exactly 2 times
            # in a watertight mesh)
            edge_count = {}
            for tri in indices:
                edges = [
                    tuple(sorted([tri[0], tri[1]])),
                    tuple(sorted([tri[1], tri[2]])),
                    tuple(sorted([tri[2], tri[0]])),
                ]
                for edge in edges:
                    edge_count[edge] = edge_count.get(edge, 0) + 1

            # For these primitives (without caps), boundary edges appear once
            # and interior edges appear twice
            edge_usages = list(edge_count.values())
            assert all(u in [1, 2] for u in edge_usages)

    def test_mesh_triangles_non_degenerate(self, primitives):
        """Test that triangles have non-zero area."""
        for primitive in primitives:
            vertices, indices, _ = primitive.generate_mesh()

            for tri in indices:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

                assert area > 1e-10, f"Degenerate triangle found: {tri}"
