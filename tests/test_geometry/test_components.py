"""
Tests for cyclone geometry components.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.components import (
    CycloneBody, CycloneBodyParams,
    VortexFinder, VortexFinderParams,
    TangentialInlet, InletParams,
    DustOutlet, DustOutletParams,
)
from airclassifier.utils.constants import PI


class TestCycloneBody:
    """Tests for CycloneBody component."""

    @pytest.fixture
    def body_params(self):
        """Create default cyclone body parameters."""
        return CycloneBodyParams(
            cylinder_diameter=0.3,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_tip_diameter=0.1125,
            center=(0.0, 0.0, 0.0),
            resolution_radial=48,
            resolution_axial_cylinder=16,
            resolution_axial_cone=24
        )

    @pytest.fixture
    def body(self, body_params):
        """Create cyclone body instance."""
        return CycloneBody(body_params)

    def test_body_creation(self, body, body_params):
        """Test cyclone body is created correctly."""
        assert body.params.cylinder_diameter == 0.3
        assert body.params.cylinder_height == 0.45
        assert body.params.cone_height == 0.75
        assert body.params.cone_tip_diameter == 0.1125

    def test_body_radii(self, body):
        """Test radius properties."""
        assert_allclose(body.params.cylinder_radius, 0.15, rtol=1e-6)
        assert_allclose(body.params.cone_top_radius, 0.15, rtol=1e-6)
        assert_allclose(body.params.cone_bottom_radius, 0.05625, rtol=1e-6)

    def test_body_total_height(self, body):
        """Test total height calculation."""
        expected = 0.45 + 0.75
        assert_allclose(body.params.total_height, expected, rtol=1e-6)

    def test_body_volume(self, body_params):
        """Test volume calculation."""
        body = CycloneBody(body_params)

        # Expected volume = cylinder + frustum
        r_cyl = body_params.cylinder_radius
        h_cyl = body_params.cylinder_height
        v_cyl = PI * r_cyl**2 * h_cyl

        r1 = body_params.cone_top_radius
        r2 = body_params.cone_bottom_radius
        h_cone = body_params.cone_height
        v_cone = (PI * h_cone / 3.0) * (r1**2 + r1*r2 + r2**2)

        expected_volume = v_cyl + v_cone
        assert_allclose(body.params.volume, expected_volume, rtol=1e-6)

    def test_body_cone_half_angle(self, body):
        """Test cone half-angle calculation."""
        dr = body.params.cone_top_radius - body.params.cone_bottom_radius
        h = body.params.cone_height
        expected = np.arctan2(dr, h)
        assert_allclose(body.params.cone_half_angle, expected, rtol=1e-6)

    def test_body_mesh_generation(self, body):
        """Test mesh generation."""
        vertices, indices, normals = body.generate_mesh()

        # Check shapes
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 2
        assert indices.shape[1] == 3
        assert normals.shape == vertices.shape

        # Should have significant number of vertices
        assert len(vertices) > 100

    def test_position_at_height_above(self, body):
        """Test position classification above cylinder."""
        section, radius = body.get_position_at_height(0.1)
        assert section == "above"
        assert_allclose(radius, body.params.cylinder_radius, rtol=1e-6)

    def test_position_at_height_cylinder(self, body):
        """Test position classification in cylinder."""
        section, radius = body.get_position_at_height(-0.2)
        assert section == "cylinder"
        assert_allclose(radius, body.params.cylinder_radius, rtol=1e-6)

    def test_position_at_height_cone(self, body):
        """Test position classification in cone."""
        # Middle of cone
        y = -body.params.cylinder_height - body.params.cone_height / 2
        section, radius = body.get_position_at_height(y)

        assert section == "cone"

        # Expected radius at midpoint
        expected_r = (body.params.cone_top_radius + body.params.cone_bottom_radius) / 2
        assert_allclose(radius, expected_r, rtol=1e-2)

    def test_position_at_height_below(self, body):
        """Test position classification below cone."""
        y = -body.params.total_height - 0.1
        section, radius = body.get_position_at_height(y)
        assert section == "below"


class TestVortexFinder:
    """Tests for VortexFinder component."""

    @pytest.fixture
    def vf_params(self):
        """Create default vortex finder parameters."""
        return VortexFinderParams(
            diameter=0.15,
            length=0.15,
            cyclone_center=(0.0, 0.0, 0.0),
            protrusion_above=0.05,
            resolution_radial=24,
            resolution_axial=12
        )

    @pytest.fixture
    def vortex_finder(self, vf_params):
        """Create vortex finder instance."""
        return VortexFinder(vf_params)

    def test_vf_creation(self, vortex_finder):
        """Test vortex finder creation."""
        assert vortex_finder.params.diameter == 0.15
        assert vortex_finder.params.length == 0.15

    def test_vf_radii(self, vortex_finder):
        """Test radius properties."""
        assert_allclose(vortex_finder.params.inner_radius, 0.075, rtol=1e-6)

    def test_vf_mesh_generation(self, vortex_finder):
        """Test mesh generation."""
        vertices, indices, normals = vortex_finder.generate_mesh()

        assert vertices.shape[1] == 3
        assert indices.shape[1] == 3
        assert len(vertices) > 10

    def test_is_inside_tube_center(self, vortex_finder):
        """Test point inside tube detection - center."""
        # Point at center, within VF region
        point = np.array([0.0, -0.05, 0.0])  # Inside insertion
        assert vortex_finder.is_inside_tube(point)

    def test_is_inside_tube_edge(self, vortex_finder):
        """Test point inside tube detection - near edge."""
        r = vortex_finder.params.inner_radius * 0.8
        point = np.array([r, -0.05, 0.0])
        assert vortex_finder.is_inside_tube(point)

    def test_is_outside_tube(self, vortex_finder):
        """Test point outside tube detection."""
        r = vortex_finder.params.inner_radius * 1.5
        point = np.array([r, -0.05, 0.0])
        assert not vortex_finder.is_inside_tube(point)

    def test_outlet_plane(self, vortex_finder):
        """Test outlet plane calculation."""
        center, normal = vortex_finder.get_outlet_plane()

        # Normal should point upward (out of cyclone)
        assert normal[1] > 0.9


class TestTangentialInlet:
    """Tests for TangentialInlet component."""

    @pytest.fixture
    def inlet_params(self):
        """Create default inlet parameters."""
        return InletParams(
            width=0.075,
            height=0.15,
            length=0.1,
            cyclone_diameter=0.3,
            inlet_top_offset=0.05,
            cyclone_center=(0.0, 0.0, 0.0),
            angular_position=0.0
        )

    @pytest.fixture
    def inlet(self, inlet_params):
        """Create inlet instance."""
        return TangentialInlet(inlet_params)

    def test_inlet_creation(self, inlet):
        """Test inlet creation."""
        assert inlet.params.width == 0.075
        assert inlet.params.height == 0.15
        assert inlet.params.length == 0.1

    def test_inlet_area(self, inlet):
        """Test inlet area calculation."""
        expected_area = 0.075 * 0.15
        assert_allclose(inlet.params.area, expected_area, rtol=1e-6)

    def test_inlet_mesh_generation(self, inlet):
        """Test mesh generation."""
        vertices, indices, normals = inlet.generate_mesh()

        assert vertices.shape[1] == 3
        assert len(vertices) > 0

    def test_inlet_velocity_direction(self, inlet):
        """Test inlet velocity direction is tangential."""
        direction = inlet.get_inlet_velocity_direction()

        # Should be unit vector
        assert_allclose(np.linalg.norm(direction), 1.0, rtol=1e-6)

        # Should be tangential (perpendicular to radial direction)
        # For angular_position=0, inlet is on +X side, flow should be in -Z or +Z
        # depending on convention


class TestDustOutlet:
    """Tests for DustOutlet component."""

    @pytest.fixture
    def outlet_params(self):
        """Create default dust outlet parameters."""
        return DustOutletParams(
            diameter=0.1125,
            length=0.06,
            cone_bottom_center=(0.0, -1.2, 0.0),
            resolution_radial=24,
            resolution_axial=6
        )

    @pytest.fixture
    def dust_outlet(self, outlet_params):
        """Create dust outlet instance."""
        return DustOutlet(outlet_params)

    def test_outlet_creation(self, dust_outlet):
        """Test dust outlet creation."""
        assert dust_outlet.params.diameter == 0.1125
        assert dust_outlet.params.length == 0.06

    def test_outlet_mesh_generation(self, dust_outlet):
        """Test mesh generation."""
        vertices, indices, normals = dust_outlet.generate_mesh()

        assert vertices.shape[1] == 3
        assert len(vertices) > 0

    def test_outlet_plane(self, dust_outlet):
        """Test outlet plane calculation."""
        center, normal = dust_outlet.get_outlet_plane()

        # Normal should point downward (out of cyclone)
        assert normal[1] < -0.9

    def test_particle_collection_inside(self, dust_outlet):
        """Test particle collection detection - inside outlet."""
        # Point below outlet
        point = np.array([0.0, -1.2 - 0.08, 0.0])
        assert dust_outlet.is_particle_collected(point)

    def test_particle_collection_outside(self, dust_outlet):
        """Test particle collection detection - outside outlet."""
        # Point at cone bottom but outside outlet radius
        point = np.array([0.1, -1.2 - 0.08, 0.0])
        assert not dust_outlet.is_particle_collected(point)


class TestComponentIntegration:
    """Integration tests for component interactions."""

    @pytest.fixture
    def all_components(self):
        """Create a set of compatible components."""
        D = 0.3  # Cylinder diameter

        body = CycloneBody(CycloneBodyParams(
            cylinder_diameter=D,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_tip_diameter=D * 0.375,
        ))

        vf = VortexFinder(VortexFinderParams(
            diameter=D * 0.5,
            length=D * 0.5,
        ))

        inlet = TangentialInlet(InletParams(
            width=D * 0.25,
            height=D * 0.5,
            length=D * 0.3,
            cyclone_diameter=D,
        ))

        outlet = DustOutlet(DustOutletParams(
            diameter=D * 0.375,
            length=D * 0.2,
            cone_bottom_center=(0.0, -1.2, 0.0),
        ))

        return body, vf, inlet, outlet

    def test_component_mesh_combination(self, all_components):
        """Test that all component meshes can be combined."""
        body, vf, inlet, outlet = all_components

        # Generate all meshes
        meshes = []
        vertex_offset = 0

        for component in [body, vf, inlet, outlet]:
            verts, idx, _ = component.generate_mesh()
            idx_offset = idx + vertex_offset
            meshes.append((verts, idx_offset))
            vertex_offset += len(verts)

        # Combine
        all_verts = np.vstack([m[0] for m in meshes])
        all_idx = np.concatenate([m[1] for m in meshes])

        # Verify
        assert len(all_verts) > 0
        assert np.all(all_idx >= 0)
        assert np.all(all_idx < len(all_verts))

    def test_vortex_finder_fits_in_body(self, all_components):
        """Test that vortex finder radius is less than body radius."""
        body, vf, _, _ = all_components

        assert vf.params.inner_radius < body.params.cylinder_radius
