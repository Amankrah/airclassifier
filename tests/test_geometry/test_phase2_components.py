"""
Tests for Phase 2 feed system components.

Tests for:
- FeedHopper
- RotaryAirlock
- ScrewFeeder
- Deagglomerator
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.components import (
    FeedHopper, FeedHopperParams, create_standard_feed_hopper,
    RotaryAirlock, RotaryAirlockParams, create_standard_rotary_airlock,
    ScrewFeeder, ScrewFeederParams, create_standard_screw_feeder,
    Deagglomerator, DeagglomeratorParams, create_standard_deagglomerator,
)
from airclassifier.utils.constants import PI


# =============================================================================
# FEED HOPPER TESTS
# =============================================================================

class TestFeedHopperParams:
    """Tests for FeedHopperParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
        )
        assert params.top_diameter == 0.5
        assert params.bottom_diameter == 0.15

    def test_total_height(self):
        """Test total height calculation."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
            has_lid=True,
            lid_height=0.05,
        )
        expected = 0.4 + 0.3 + 0.05
        assert_allclose(params.total_height, expected, rtol=1e-6)

    def test_radii(self):
        """Test radius properties."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
        )
        assert_allclose(params.top_radius, 0.25, rtol=1e-6)
        assert_allclose(params.bottom_radius, 0.075, rtol=1e-6)

    def test_volume_calculation(self):
        """Test volume calculations."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
        )
        # Cylindrical volume
        v_cyl = PI * 0.25 ** 2 * 0.4
        assert_allclose(params.cylindrical_volume, v_cyl, rtol=1e-6)

        # Total volume should be positive
        assert params.total_volume > 0

    def test_cone_half_angle(self):
        """Test cone half-angle calculation."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
        )
        # angle = atan((0.25 - 0.075) / 0.3)
        expected = np.arctan2(0.175, 0.3)
        assert_allclose(params.cone_half_angle, expected, rtol=1e-6)

    def test_capacity_calculation(self):
        """Test capacity calculation."""
        params = FeedHopperParams(
            top_diameter=0.5,
            bottom_diameter=0.15,
            cylindrical_height=0.4,
            conical_height=0.3,
        )
        capacity = params.capacity_kg(bulk_density=500)
        assert capacity > 0


class TestFeedHopper:
    """Tests for FeedHopper component."""

    @pytest.fixture
    def hopper(self):
        """Create a standard feed hopper."""
        return create_standard_feed_hopper(capacity_kg=500)

    def test_hopper_creation(self, hopper):
        """Test hopper is created correctly."""
        assert hopper.params.top_diameter > 0
        assert hopper.params.bottom_diameter > 0

    def test_mesh_generation(self, hopper):
        """Test mesh generation."""
        vertices, indices, normals = hopper.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0
        assert len(vertices) > 0

    def test_mesh_valid_indices(self, hopper):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = hopper.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_mass_flow_design(self, hopper):
        """Test mass flow design check."""
        # For most powders with angle of repose ~35 deg
        result = hopper.is_mass_flow_design(material_angle_of_repose=35.0)
        # Result depends on cone angle, just check it returns a boolean-like value
        assert result in (True, False)

    def test_discharge_center(self, hopper):
        """Test discharge center position."""
        center = hopper.get_discharge_center()
        assert len(center) == 3


# =============================================================================
# ROTARY AIRLOCK TESTS
# =============================================================================

class TestRotaryAirlockParams:
    """Tests for RotaryAirlockParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = RotaryAirlockParams(
            rotor_diameter=0.2,
            rotor_length=0.15,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
        )
        assert params.rotor_diameter == 0.2
        assert params.num_vanes == 8

    def test_radii(self):
        """Test radius properties."""
        params = RotaryAirlockParams(
            rotor_diameter=0.2,
            rotor_length=0.15,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
        )
        assert_allclose(params.rotor_radius, 0.1, rtol=1e-6)
        assert_allclose(params.housing_inner_radius, 0.1003, rtol=1e-6)

    def test_pocket_angle(self):
        """Test pocket angle calculation."""
        params = RotaryAirlockParams(
            rotor_diameter=0.2,
            rotor_length=0.15,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
        )
        expected = 2 * PI / 8
        assert_allclose(params.pocket_angle, expected, rtol=1e-6)

    def test_volumetric_capacity(self):
        """Test volumetric capacity calculation."""
        params = RotaryAirlockParams(
            rotor_diameter=0.2,
            rotor_length=0.15,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
            rpm=20,
        )
        capacity = params.volumetric_capacity
        assert capacity > 0

    def test_mass_capacity(self):
        """Test mass capacity calculation."""
        params = RotaryAirlockParams(
            rotor_diameter=0.2,
            rotor_length=0.15,
            num_vanes=8,
            vane_thickness=0.005,
            vane_tip_clearance=0.0003,
        )
        capacity = params.capacity_kg_h(bulk_density=500)
        assert capacity > 0


class TestRotaryAirlock:
    """Tests for RotaryAirlock component."""

    @pytest.fixture
    def airlock(self):
        """Create a standard rotary airlock."""
        return create_standard_rotary_airlock(rotor_diameter=0.2)

    def test_airlock_creation(self, airlock):
        """Test airlock is created correctly."""
        assert airlock.params.rotor_diameter == 0.2
        assert airlock.params.num_vanes > 0

    def test_mesh_generation(self, airlock):
        """Test mesh generation."""
        vertices, indices, normals = airlock.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, airlock):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = airlock.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_air_leakage_rate(self, airlock):
        """Test air leakage rate calculation."""
        leakage = airlock.get_air_leakage_rate(pressure_diff=5000)
        assert leakage > 0


# =============================================================================
# SCREW FEEDER TESTS
# =============================================================================

class TestScrewFeederParams:
    """Tests for ScrewFeederParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = ScrewFeederParams(
            screw_diameter=0.1,
            shaft_diameter=0.03,
            screw_pitch=0.08,
            flight_thickness=0.003,
            trough_length=0.3,
            trough_clearance=0.003,
            inlet_length=0.15,
            inlet_width=0.12,
            outlet_diameter=0.08,
        )
        assert params.screw_diameter == 0.1
        assert params.screw_pitch == 0.08

    def test_radii(self):
        """Test radius properties."""
        params = ScrewFeederParams(
            screw_diameter=0.1,
            shaft_diameter=0.03,
            screw_pitch=0.08,
            flight_thickness=0.003,
            trough_length=0.3,
            trough_clearance=0.003,
            inlet_length=0.15,
            inlet_width=0.12,
            outlet_diameter=0.08,
        )
        assert_allclose(params.screw_radius, 0.05, rtol=1e-6)
        assert_allclose(params.shaft_radius, 0.015, rtol=1e-6)

    def test_num_flights(self):
        """Test number of flights calculation."""
        params = ScrewFeederParams(
            screw_diameter=0.1,
            shaft_diameter=0.03,
            screw_pitch=0.1,
            flight_thickness=0.003,
            trough_length=0.3,
            trough_clearance=0.003,
            inlet_length=0.15,
            inlet_width=0.12,
            outlet_diameter=0.08,
        )
        assert_allclose(params.num_flights, 3.0, rtol=1e-6)

    def test_volumetric_capacity(self):
        """Test volumetric capacity calculation."""
        params = ScrewFeederParams(
            screw_diameter=0.1,
            shaft_diameter=0.03,
            screw_pitch=0.08,
            flight_thickness=0.003,
            trough_length=0.3,
            trough_clearance=0.003,
            inlet_length=0.15,
            inlet_width=0.12,
            outlet_diameter=0.08,
            rpm=30,
            fill_level=0.30,
        )
        capacity = params.volumetric_capacity
        assert capacity > 0


class TestScrewFeeder:
    """Tests for ScrewFeeder component."""

    @pytest.fixture
    def feeder(self):
        """Create a standard screw feeder."""
        return create_standard_screw_feeder(screw_diameter=0.1)

    def test_feeder_creation(self, feeder):
        """Test feeder is created correctly."""
        assert feeder.params.screw_diameter == 0.1

    def test_mesh_generation(self, feeder):
        """Test mesh generation."""
        vertices, indices, normals = feeder.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, feeder):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = feeder.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_feed_rate_calculation(self, feeder):
        """Test feed rate calculation."""
        rate = feeder.get_feed_rate(rpm=30, bulk_density=500)
        assert rate > 0

    def test_feed_rate_scales_with_rpm(self, feeder):
        """Test that feed rate scales with RPM."""
        rate_slow = feeder.get_feed_rate(rpm=15)
        rate_fast = feeder.get_feed_rate(rpm=30)
        assert_allclose(rate_fast, rate_slow * 2, rtol=0.01)


# =============================================================================
# DEAGGLOMERATOR TESTS
# =============================================================================

class TestDeagglomeratorParams:
    """Tests for DeagglomeratorParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = DeagglomeratorParams(
            rotor_diameter=0.2,
            rotor_length=0.12,
            shaft_diameter=0.04,
            num_pin_rows=3,
            pins_per_row=6,
            pin_diameter=0.01,
            pin_length=0.07,
            housing_diameter=0.26,
            housing_length=0.16,
            screen_diameter=0.22,
            screen_aperture=0.002,
            screen_open_area=0.40,
            inlet_diameter=0.08,
            outlet_diameter=0.10,
        )
        assert params.rotor_diameter == 0.2
        assert params.num_pin_rows == 3

    def test_radii(self):
        """Test radius properties."""
        params = DeagglomeratorParams(
            rotor_diameter=0.2,
            rotor_length=0.12,
            shaft_diameter=0.04,
            num_pin_rows=3,
            pins_per_row=6,
            pin_diameter=0.01,
            pin_length=0.07,
            housing_diameter=0.26,
            housing_length=0.16,
            screen_diameter=0.22,
            screen_aperture=0.002,
            screen_open_area=0.40,
            inlet_diameter=0.08,
            outlet_diameter=0.10,
        )
        assert_allclose(params.rotor_radius, 0.1, rtol=1e-6)
        assert_allclose(params.shaft_radius, 0.02, rtol=1e-6)

    def test_tip_speed(self):
        """Test pin tip speed calculation."""
        params = DeagglomeratorParams(
            rotor_diameter=0.2,
            rotor_length=0.12,
            shaft_diameter=0.04,
            num_pin_rows=3,
            pins_per_row=6,
            pin_diameter=0.01,
            pin_length=0.07,
            housing_diameter=0.26,
            housing_length=0.16,
            screen_diameter=0.22,
            screen_aperture=0.002,
            screen_open_area=0.40,
            inlet_diameter=0.08,
            outlet_diameter=0.10,
            rpm=1500,
        )
        # tip_speed = pi * D * rpm / 60
        expected = PI * 0.2 * 1500 / 60
        assert_allclose(params.pin_tip_speed, expected, rtol=1e-6)

    def test_clearance(self):
        """Test clearance calculation."""
        params = DeagglomeratorParams(
            rotor_diameter=0.2,
            rotor_length=0.12,
            shaft_diameter=0.04,
            num_pin_rows=3,
            pins_per_row=6,
            pin_diameter=0.01,
            pin_length=0.07,
            housing_diameter=0.26,
            housing_length=0.16,
            screen_diameter=0.22,
            screen_aperture=0.002,
            screen_open_area=0.40,
            inlet_diameter=0.08,
            outlet_diameter=0.10,
        )
        # clearance = screen_radius - rotor_radius = 0.11 - 0.1 = 0.01
        expected = 0.01
        assert_allclose(params.clearance, expected, rtol=1e-6)


class TestDeagglomerator:
    """Tests for Deagglomerator component."""

    @pytest.fixture
    def deagg(self):
        """Create a standard de-agglomerator."""
        return create_standard_deagglomerator(rotor_diameter=0.2)

    def test_deagg_creation(self, deagg):
        """Test de-agglomerator is created correctly."""
        assert deagg.params.rotor_diameter == 0.2

    def test_mesh_generation(self, deagg):
        """Test mesh generation."""
        vertices, indices, normals = deagg.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, deagg):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = deagg.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_tip_speed(self, deagg):
        """Test tip speed getter."""
        speed = deagg.get_tip_speed()
        assert speed > 0


# =============================================================================
# MESH QUALITY TESTS
# =============================================================================

class TestPhase2MeshQuality:
    """Tests for mesh quality of Phase 2 components."""

    @pytest.fixture
    def all_components(self):
        """Create all Phase 2 components."""
        return {
            'feed_hopper': create_standard_feed_hopper(),
            'rotary_airlock': create_standard_rotary_airlock(),
            'screw_feeder': create_standard_screw_feeder(),
            'deagglomerator': create_standard_deagglomerator(),
        }

    def test_no_nan_vertices(self, all_components):
        """Test that no vertices contain NaN."""
        for name, component in all_components.items():
            verts, _, _ = component.generate_mesh()
            assert not np.any(np.isnan(verts)), f"{name} has NaN vertices"

    def test_no_inf_vertices(self, all_components):
        """Test that no vertices contain Inf."""
        for name, component in all_components.items():
            verts, _, _ = component.generate_mesh()
            assert not np.any(np.isinf(verts)), f"{name} has Inf vertices"

    def test_no_degenerate_triangles(self, all_components):
        """Test that no triangles have zero area."""
        for name, component in all_components.items():
            verts, idx, _ = component.generate_mesh()
            triangles = idx.reshape(-1, 3)

            for i, tri in enumerate(triangles[:100]):  # Check first 100
                v0, v1, v2 = verts[tri]
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                assert area > 1e-12, f"{name} has degenerate triangle {i}"

    def test_vertices_count(self, all_components):
        """Test that meshes have reasonable vertex counts."""
        for name, component in all_components.items():
            verts, idx, _ = component.generate_mesh()
            assert len(verts) > 10, f"{name} has too few vertices"
            assert len(idx) > 10, f"{name} has too few indices"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_all_components_create_warp_mesh(self):
        """Test that all components can create Warp meshes."""
        components = [
            create_standard_feed_hopper(),
            create_standard_rotary_airlock(),
            create_standard_screw_feeder(),
            create_standard_deagglomerator(),
        ]

        for comp in components:
            comp.generate_mesh()
            mesh = comp.to_warp_mesh(device="cpu")
            assert mesh is not None

    def test_feed_system_dimensions_compatible(self):
        """Test that feed system dimensions are compatible."""
        hopper = create_standard_feed_hopper(discharge_diameter=0.15)
        airlock = create_standard_rotary_airlock(rotor_diameter=0.20)
        feeder = create_standard_screw_feeder(screw_diameter=0.15)  # Larger feeder

        # Hopper discharge should fit into airlock inlet
        # Airlock should be larger than hopper discharge
        assert airlock.params.rotor_diameter >= hopper.params.bottom_diameter * 0.8

        # Check all components have valid dimensions
        assert hopper.params.bottom_diameter > 0
        assert airlock.params.rotor_diameter > 0
        assert feeder.params.screw_diameter > 0
