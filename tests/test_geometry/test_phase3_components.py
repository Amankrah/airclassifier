"""
Tests for Phase 3 air system components.

Tests for:
- CentrifugalBlower
- InletAirFilter
- FlowDamper
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.components import (
    CentrifugalBlower, CentrifugalBlowerParams, create_standard_centrifugal_blower,
    InletAirFilter, InletAirFilterParams, create_standard_inlet_filter,
    FlowDamper, DamperParams, create_standard_damper,
)
from airclassifier.utils.constants import PI


# =============================================================================
# CENTRIFUGAL BLOWER TESTS
# =============================================================================

class TestCentrifugalBlowerParams:
    """Tests for CentrifugalBlowerParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = CentrifugalBlowerParams(
            impeller_diameter=0.4,
            impeller_width=0.1,
            inlet_diameter=0.24,
            hub_diameter=0.1,
            num_blades=10,
        )
        assert params.impeller_diameter == 0.4
        assert params.num_blades == 10

    def test_radii(self):
        """Test radius properties."""
        params = CentrifugalBlowerParams(
            impeller_diameter=0.4,
            impeller_width=0.1,
            inlet_diameter=0.24,
            hub_diameter=0.1,
            num_blades=10,
        )
        assert_allclose(params.impeller_radius, 0.2, rtol=1e-6)
        assert_allclose(params.inlet_radius, 0.12, rtol=1e-6)
        assert_allclose(params.hub_radius, 0.05, rtol=1e-6)

    def test_tip_speed(self):
        """Test tip speed calculation."""
        params = CentrifugalBlowerParams(
            impeller_diameter=0.4,
            impeller_width=0.1,
            inlet_diameter=0.24,
            hub_diameter=0.1,
            num_blades=10,
            rpm=3000,
        )
        # tip_speed = pi * D * rpm / 60
        expected = PI * 0.4 * 3000 / 60
        assert_allclose(params.tip_speed, expected, rtol=1e-6)

    def test_shaft_power(self):
        """Test shaft power calculation."""
        params = CentrifugalBlowerParams(
            impeller_diameter=0.4,
            impeller_width=0.1,
            inlet_diameter=0.24,
            hub_diameter=0.1,
            num_blades=10,
            flow_rate=3000,
            pressure_rise=5000,
        )
        power = params.shaft_power
        assert power > 0

    def test_specific_speed(self):
        """Test specific speed calculation."""
        params = CentrifugalBlowerParams(
            impeller_diameter=0.4,
            impeller_width=0.1,
            inlet_diameter=0.24,
            hub_diameter=0.1,
            num_blades=10,
            rpm=3000,
            flow_rate=3000,
            pressure_rise=5000,
        )
        ns = params.specific_speed
        assert ns > 0


class TestCentrifugalBlower:
    """Tests for CentrifugalBlower component."""

    @pytest.fixture
    def blower(self):
        """Create a standard centrifugal blower."""
        return create_standard_centrifugal_blower(flow_rate=3000, pressure_rise=5000)

    def test_blower_creation(self, blower):
        """Test blower is created correctly."""
        assert blower.params.impeller_diameter > 0
        assert blower.params.num_blades > 0

    def test_mesh_generation(self, blower):
        """Test mesh generation."""
        vertices, indices, normals = blower.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0
        assert len(vertices) > 0

    def test_mesh_valid_indices(self, blower):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = blower.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_performance_output(self, blower):
        """Test performance calculation."""
        perf = blower.get_performance()
        assert 'flow_rate_m3_h' in perf
        assert 'tip_speed_m_s' in perf
        assert 'shaft_power_kW' in perf
        assert perf['efficiency'] > 0


# =============================================================================
# INLET AIR FILTER TESTS
# =============================================================================

class TestInletAirFilterParams:
    """Tests for InletAirFilterParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = InletAirFilterParams(
            housing_width=0.5,
            housing_height=0.5,
            housing_depth=0.2,
        )
        assert params.housing_width == 0.5
        assert params.housing_height == 0.5

    def test_face_area(self):
        """Test face area calculation."""
        params = InletAirFilterParams(
            housing_width=0.5,
            housing_height=0.6,
            housing_depth=0.2,
        )
        expected = 0.5 * 0.6
        assert_allclose(params.face_area, expected, rtol=1e-6)

    def test_max_flow_rate(self):
        """Test max flow rate calculation."""
        params = InletAirFilterParams(
            housing_width=0.5,
            housing_height=0.5,
            housing_depth=0.2,
            filter_type="panel",
        )
        max_flow = params.max_flow_rate()
        assert max_flow > 0

    def test_filter_area_auto_calculation(self):
        """Test that filter area is auto-calculated."""
        params = InletAirFilterParams(
            housing_width=0.5,
            housing_height=0.5,
            housing_depth=0.2,
        )
        assert params.filter_area is not None
        assert params.filter_area > 0


class TestInletAirFilter:
    """Tests for InletAirFilter component."""

    @pytest.fixture
    def filter_panel(self):
        """Create a panel filter."""
        return create_standard_inlet_filter(flow_rate=3000, filter_type="panel")

    @pytest.fixture
    def filter_bag(self):
        """Create a bag filter."""
        return create_standard_inlet_filter(flow_rate=3000, filter_type="bag")

    def test_filter_creation(self, filter_panel):
        """Test filter is created correctly."""
        assert filter_panel.params.housing_width > 0
        assert filter_panel.params.housing_height > 0

    def test_mesh_generation(self, filter_panel):
        """Test mesh generation."""
        vertices, indices, normals = filter_panel.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, filter_panel):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = filter_panel.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_pressure_drop(self, filter_panel):
        """Test pressure drop calculation."""
        dp_clean = filter_panel.get_pressure_drop(flow_rate=3000, loading=0)
        dp_loaded = filter_panel.get_pressure_drop(flow_rate=3000, loading=1)

        assert dp_clean > 0
        assert dp_loaded > dp_clean

    def test_efficiency_by_class(self, filter_panel):
        """Test efficiency lookup."""
        eff = filter_panel.get_efficiency()
        assert isinstance(eff, dict)

    def test_different_filter_types_have_different_sizes(self, filter_panel, filter_bag):
        """Test that different filter types result in different housing sizes."""
        # Bag filters need larger housing for same flow rate
        assert filter_bag.params.housing_width != filter_panel.params.housing_width


# =============================================================================
# FLOW DAMPER TESTS
# =============================================================================

class TestDamperParams:
    """Tests for DamperParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = DamperParams(
            diameter=0.3,
            damper_type="butterfly",
        )
        assert params.diameter == 0.3
        assert params.damper_type == "butterfly"

    def test_radius(self):
        """Test radius property."""
        params = DamperParams(diameter=0.3)
        assert_allclose(params.radius, 0.15, rtol=1e-6)

    def test_flow_area_open(self):
        """Test full open flow area."""
        params = DamperParams(diameter=0.3)
        expected = PI * 0.15 ** 2
        assert_allclose(params.flow_area_open, expected, rtol=1e-6)

    def test_flow_area_closed(self):
        """Test flow area when closed."""
        params = DamperParams(diameter=0.3, position=0)
        area = params.flow_area()
        assert area == 0 or area < params.flow_area_open * 0.01

    def test_flow_area_varies_with_position(self):
        """Test that flow area changes with position."""
        params = DamperParams(diameter=0.3)

        area_closed = params.flow_area(0)
        area_half = params.flow_area(0.5)
        area_open = params.flow_area(1)

        assert area_closed < area_half < area_open

    def test_cv_calculation(self):
        """Test flow coefficient calculation."""
        params = DamperParams(diameter=0.3, position=1)
        cv = params.cv()
        assert cv > 0


class TestFlowDamper:
    """Tests for FlowDamper component."""

    @pytest.fixture
    def damper_butterfly(self):
        """Create a butterfly damper."""
        return create_standard_damper(diameter=0.3, damper_type="butterfly")

    @pytest.fixture
    def damper_louver(self):
        """Create a louver damper."""
        return create_standard_damper(diameter=0.3, damper_type="louver")

    @pytest.fixture
    def damper_iris(self):
        """Create an iris damper."""
        return create_standard_damper(diameter=0.3, damper_type="iris")

    def test_damper_creation(self, damper_butterfly):
        """Test damper is created correctly."""
        assert damper_butterfly.params.diameter == 0.3
        assert damper_butterfly.params.damper_type == "butterfly"

    def test_mesh_generation_butterfly(self, damper_butterfly):
        """Test butterfly damper mesh generation."""
        vertices, indices, normals = damper_butterfly.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_generation_louver(self, damper_louver):
        """Test louver damper mesh generation."""
        vertices, indices, normals = damper_louver.generate_mesh()

        assert len(vertices) > 0
        assert len(indices) > 0

    def test_mesh_generation_iris(self, damper_iris):
        """Test iris damper mesh generation."""
        vertices, indices, normals = damper_iris.generate_mesh()

        assert len(vertices) > 0
        assert len(indices) > 0

    def test_mesh_valid_indices(self, damper_butterfly):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = damper_butterfly.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_set_position(self, damper_butterfly):
        """Test setting damper position."""
        damper_butterfly.set_position(0.5)
        assert damper_butterfly.params.position == 0.5

        damper_butterfly.set_position(1.5)  # Should clamp
        assert damper_butterfly.params.position == 1.0

        damper_butterfly.set_position(-0.5)  # Should clamp
        assert damper_butterfly.params.position == 0.0

    def test_pressure_drop(self, damper_butterfly):
        """Test pressure drop calculation."""
        damper_butterfly.set_position(1.0)
        dp_open = damper_butterfly.get_pressure_drop(flow_rate=3000)

        damper_butterfly.set_position(0.5)
        dp_half = damper_butterfly.get_pressure_drop(flow_rate=3000)

        assert dp_open < dp_half  # More resistance when throttled


# =============================================================================
# MESH QUALITY TESTS
# =============================================================================

class TestPhase3MeshQuality:
    """Tests for mesh quality of Phase 3 components."""

    @pytest.fixture
    def all_components(self):
        """Create all Phase 3 components."""
        return {
            'centrifugal_blower': create_standard_centrifugal_blower(),
            'inlet_filter': create_standard_inlet_filter(),
            'damper_butterfly': create_standard_damper(damper_type="butterfly"),
            'damper_louver': create_standard_damper(damper_type="louver"),
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

class TestPhase3Integration:
    """Integration tests for Phase 3 components."""

    def test_all_components_create_warp_mesh(self):
        """Test that all components can create Warp meshes."""
        components = [
            create_standard_centrifugal_blower(),
            create_standard_inlet_filter(),
            create_standard_damper(),
        ]

        for comp in components:
            comp.generate_mesh()
            mesh = comp.to_warp_mesh(device="cpu")
            assert mesh is not None

    def test_air_system_sizing_reasonable(self):
        """Test that air system components are sized reasonably."""
        blower = create_standard_centrifugal_blower(flow_rate=3000)
        filter_unit = create_standard_inlet_filter(flow_rate=3000)
        damper = create_standard_damper(diameter=0.3)

        # Blower should have reasonable impeller size
        assert 0.1 < blower.params.impeller_diameter < 2.0

        # Filter should be reasonably sized
        assert filter_unit.params.housing_width > 0.1
        assert filter_unit.params.housing_height > 0.1

        # Damper should fit typical duct sizes
        assert damper.params.diameter > 0

    def test_efficiency_estimates_reasonable(self):
        """Test that efficiency estimates are in valid range."""
        blower = create_standard_centrifugal_blower()

        # Efficiency should be between 0 and 1
        assert 0 < blower.params.estimated_efficiency <= 1.0

        # Backward curved should be most efficient
        blower_bc = create_standard_centrifugal_blower()
        assert blower_bc.params.estimated_efficiency >= 0.75
