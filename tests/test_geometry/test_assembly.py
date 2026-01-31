"""
Tests for assembly modules.

Tests for:
- CycloneAssembly
- ClassificationSystemAssembly
- FeedSystemAssembly
- AirSystemAssembly
- DuctworkSystemAssembly
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.assembly import (
    # Cyclone
    CycloneAssembly, CycloneGeometryParams, create_standard_cyclone,
    # Classification System
    ClassificationSystemAssembly, ClassificationSystemParams,
    create_standard_classification_system, create_protein_separation_system,
    # Feed System
    FeedSystemAssembly, FeedSystemParams,
    create_standard_feed_system, create_feed_system_for_throughput,
    # Air System
    AirSystemAssembly, AirSystemParams,
    create_standard_air_system, create_air_system_for_classifier,
    # Ductwork System
    DuctworkSystemAssembly, DuctworkSystemParams,
    create_standard_ductwork, create_ductwork_for_classifier, create_simple_duct_run,
)


# =============================================================================
# CYCLONE ASSEMBLY TESTS
# =============================================================================

class TestCycloneGeometryParams:
    """Tests for CycloneGeometryParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = CycloneGeometryParams(
            cylinder_diameter=0.3,
            cylinder_height=0.45,
            cone_height=0.75,
            cone_tip_diameter=0.1125,
            inlet_width=0.075,
            inlet_height=0.15,
            vortex_finder_diameter=0.15,
            vortex_finder_length=0.15,
        )
        assert params.cylinder_diameter == 0.3

    def test_from_diameter(self):
        """Test creating params from diameter."""
        params = CycloneGeometryParams.from_diameter(0.3)
        assert params.cylinder_diameter == 0.3
        assert_allclose(params.inlet_width, 0.075, rtol=1e-6)

    def test_total_height(self):
        """Test total height calculation."""
        params = CycloneGeometryParams.from_diameter(0.3)
        expected = params.cylinder_height + params.cone_height
        assert_allclose(params.total_height, expected, rtol=1e-6)


class TestCycloneAssembly:
    """Tests for CycloneAssembly."""

    @pytest.fixture
    def cyclone(self):
        """Create a standard cyclone."""
        return create_standard_cyclone(diameter=0.3, device="cpu")

    def test_cyclone_creation(self, cyclone):
        """Test cyclone is created correctly."""
        assert cyclone.params.cylinder_diameter == 0.3
        assert cyclone.body is not None
        assert cyclone.inlet is not None
        assert cyclone.vortex_finder is not None
        assert cyclone.dust_outlet is not None

    def test_build_mesh(self, cyclone):
        """Test mesh building."""
        verts, idx = cyclone.build_mesh()

        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0

    def test_get_bounds(self, cyclone):
        """Test bounding box."""
        min_c, max_c = cyclone.get_bounds()

        assert len(min_c) == 3
        assert len(max_c) == 3
        assert np.all(max_c > min_c)

    def test_get_warp_mesh(self, cyclone):
        """Test Warp mesh creation."""
        mesh = cyclone.get_warp_mesh()
        assert mesh is not None


# =============================================================================
# CLASSIFICATION SYSTEM TESTS
# =============================================================================

class TestClassificationSystemParams:
    """Tests for ClassificationSystemParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = ClassificationSystemParams()
        assert params.zigzag_channel_width == 0.15
        assert params.zigzag_num_stages == 5

    def test_params_custom(self):
        """Test creating params with custom values."""
        params = ClassificationSystemParams(
            zigzag_channel_width=0.2,
            zigzag_num_stages=7,
        )
        assert params.zigzag_channel_width == 0.2
        assert params.zigzag_num_stages == 7


class TestClassificationSystemAssembly:
    """Tests for ClassificationSystemAssembly."""

    @pytest.fixture
    def system(self):
        """Create a standard classification system."""
        return create_standard_classification_system(device="cpu")

    def test_system_creation(self, system):
        """Test system is created correctly."""
        assert system.venturi is not None
        assert system.zigzag is not None
        assert system.multi_cyclone is not None
        assert system.bag_filter is not None

    def test_build_mesh(self, system):
        """Test mesh building."""
        verts, idx = system.build_mesh()

        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
        assert len(verts) > 1000  # Should have significant geometry

    def test_get_bounds(self, system):
        """Test bounding box."""
        min_c, max_c = system.get_bounds()

        assert len(min_c) == 3
        assert len(max_c) == 3
        assert np.all(max_c > min_c)

    def test_get_component(self, system):
        """Test getting components by name."""
        venturi = system.get_component('venturi')
        assert venturi is not None

        with pytest.raises(KeyError):
            system.get_component('invalid')

    def test_get_component_positions(self, system):
        """Test getting component positions."""
        positions = system.get_component_positions()
        assert 'venturi' in positions
        assert 'zigzag' in positions
        assert 'multi_cyclone' in positions
        assert 'bag_filter' in positions

    def test_protein_separation_system(self):
        """Test protein separation system factory."""
        system = create_protein_separation_system(throughput_kg_h=200)
        assert system is not None
        # Scaled up system should be larger
        extent = system.get_system_extent()
        assert extent[0] > 0


# =============================================================================
# FEED SYSTEM TESTS
# =============================================================================

class TestFeedSystemParams:
    """Tests for FeedSystemParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = FeedSystemParams()
        assert params.hopper_capacity_kg == 500
        assert params.feeder_target_rate_kg_h == 500

    def test_params_custom(self):
        """Test creating params with custom values."""
        params = FeedSystemParams(
            hopper_capacity_kg=1000,
            feeder_target_rate_kg_h=750,
        )
        assert params.hopper_capacity_kg == 1000


class TestFeedSystemAssembly:
    """Tests for FeedSystemAssembly."""

    @pytest.fixture
    def feed_system(self):
        """Create a standard feed system."""
        return create_standard_feed_system(device="cpu")

    def test_system_creation(self, feed_system):
        """Test system is created correctly."""
        assert feed_system.hopper is not None
        assert feed_system.airlock is not None
        assert feed_system.feeder is not None
        assert feed_system.deagglomerator is not None

    def test_build_mesh(self, feed_system):
        """Test mesh building."""
        verts, idx = feed_system.build_mesh()

        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
        assert len(verts) > 100

    def test_get_bounds(self, feed_system):
        """Test bounding box."""
        min_c, max_c = feed_system.get_bounds()

        assert len(min_c) == 3
        assert len(max_c) == 3
        assert np.all(max_c > min_c)

    def test_get_component(self, feed_system):
        """Test getting components by name."""
        hopper = feed_system.get_component('hopper')
        assert hopper is not None

    def test_get_feed_rate(self, feed_system):
        """Test feed rate calculation."""
        rate = feed_system.get_feed_rate()
        assert rate > 0

    def test_throughput_scaling(self):
        """Test throughput-based sizing."""
        small = create_feed_system_for_throughput(throughput_kg_h=250)
        large = create_feed_system_for_throughput(throughput_kg_h=1000)

        # Larger system should have bigger hopper
        assert large.params.hopper_capacity_kg > small.params.hopper_capacity_kg


# =============================================================================
# AIR SYSTEM TESTS
# =============================================================================

class TestAirSystemParams:
    """Tests for AirSystemParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = AirSystemParams()
        assert params.flow_rate_m3_h == 3000
        assert params.pressure_rise_Pa == 5000

    def test_params_custom(self):
        """Test creating params with custom values."""
        params = AirSystemParams(
            flow_rate_m3_h=5000,
            pressure_rise_Pa=8000,
        )
        assert params.flow_rate_m3_h == 5000


class TestAirSystemAssembly:
    """Tests for AirSystemAssembly."""

    @pytest.fixture
    def air_system(self):
        """Create a standard air system."""
        return create_standard_air_system(device="cpu")

    def test_system_creation(self, air_system):
        """Test system is created correctly."""
        assert air_system.inlet_filter is not None
        assert air_system.blower is not None
        assert len(air_system.dampers) > 0

    def test_build_mesh(self, air_system):
        """Test mesh building."""
        verts, idx = air_system.build_mesh()

        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
        assert len(verts) > 100

    def test_get_bounds(self, air_system):
        """Test bounding box."""
        min_c, max_c = air_system.get_bounds()

        assert len(min_c) == 3
        assert len(max_c) == 3
        assert np.all(max_c > min_c)

    def test_get_component(self, air_system):
        """Test getting components by name."""
        blower = air_system.get_component('blower')
        assert blower is not None

        damper = air_system.get_component('damper_0')
        assert damper is not None

    def test_set_damper_position(self, air_system):
        """Test setting damper position."""
        air_system.set_damper_position(0, 0.5)
        assert air_system.dampers[0].params.position == 0.5

    def test_get_total_pressure_drop(self, air_system):
        """Test pressure drop calculation."""
        dp = air_system.get_total_pressure_drop()
        assert dp > 0

    def test_get_performance_summary(self, air_system):
        """Test performance summary."""
        perf = air_system.get_performance_summary()
        assert 'design_flow_rate_m3_h' in perf
        assert 'blower_power_kW' in perf

    def test_classifier_sizing(self):
        """Test sizing for classifier."""
        system = create_air_system_for_classifier(
            flow_rate_m3_h=5000,
            system_pressure_drop_Pa=6000
        )
        assert system.params.flow_rate_m3_h == 5000
        # Pressure rise should have margin over system drop
        assert system.params.pressure_rise_Pa > 6000


# =============================================================================
# DUCTWORK SYSTEM TESTS
# =============================================================================

class TestDuctworkSystemParams:
    """Tests for DuctworkSystemParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = DuctworkSystemParams()
        assert params.main_duct_diameter == 0.2
        assert params.total_length == 5.0

    def test_params_custom(self):
        """Test creating params with custom values."""
        params = DuctworkSystemParams(
            main_duct_diameter=0.25,
            total_length=10.0,
            num_elbows=3,
        )
        assert params.main_duct_diameter == 0.25
        assert params.num_elbows == 3

    def test_equivalent_length(self):
        """Test equivalent length calculation."""
        params = DuctworkSystemParams(
            main_duct_diameter=0.2,
            total_length=5.0,
            num_elbows=2,
            num_45_elbows=1,
        )
        L_eq = params.total_equivalent_length
        # Should be greater than total_length due to fittings
        assert L_eq > params.total_length


class TestDuctworkSystemAssembly:
    """Tests for DuctworkSystemAssembly."""

    @pytest.fixture
    def ductwork(self):
        """Create a standard ductwork system."""
        return create_standard_ductwork(
            main_diameter=0.2,
            total_length=5.0,
            num_elbows=2
        )

    def test_system_creation(self, ductwork):
        """Test system is created correctly."""
        assert len(ductwork._components) > 0
        assert ductwork.params.main_duct_diameter == 0.2

    def test_build_mesh(self, ductwork):
        """Test mesh building."""
        verts, idx = ductwork.build_mesh()

        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
        assert len(verts) > 100

    def test_get_bounds(self, ductwork):
        """Test bounding box."""
        min_c, max_c = ductwork.get_bounds()

        assert len(min_c) == 3
        assert len(max_c) == 3
        assert np.all(max_c >= min_c)

    def test_get_component(self, ductwork):
        """Test getting components by name."""
        duct = ductwork.get_component('duct_0')
        assert duct is not None

    def test_get_component_names(self, ductwork):
        """Test getting component names."""
        names = ductwork.get_component_names()
        assert len(names) > 0
        assert 'duct_0' in names

    def test_get_total_pressure_drop(self, ductwork):
        """Test pressure drop calculation."""
        flow_rate = 0.1  # m³/s
        dp = ductwork.get_total_pressure_drop(flow_rate)
        assert dp > 0

    def test_get_system_summary(self, ductwork):
        """Test system summary."""
        summary = ductwork.get_system_summary()
        assert 'main_diameter_m' in summary
        assert 'total_length_m' in summary
        assert 'num_components' in summary

    def test_ductwork_for_classifier(self):
        """Test ductwork sized for classifier."""
        ductwork = create_ductwork_for_classifier(
            flow_rate_m3_h=3000,
            target_velocity=15.0,
            run_length=8.0,
            num_turns=3,
            include_diverter=True
        )
        assert ductwork.params.has_diverter
        assert ductwork.params.num_elbows == 3

    def test_simple_duct_run(self):
        """Test simple straight duct run."""
        ductwork = create_simple_duct_run(
            diameter=0.2,
            length=2.0
        )
        assert ductwork.params.num_elbows == 0
        assert ductwork.params.has_diverter == False

    def test_ductwork_with_diverter(self):
        """Test ductwork with diverter."""
        ductwork = create_standard_ductwork(
            main_diameter=0.2,
            total_length=5.0,
            has_diverter=True
        )
        assert ductwork.params.has_diverter
        assert 'diverter' in ductwork.get_component_names()


# =============================================================================
# MESH QUALITY TESTS
# =============================================================================

class TestAssemblyMeshQuality:
    """Tests for mesh quality of assemblies."""

    @pytest.fixture
    def all_assemblies(self):
        """Create all assembly types."""
        return {
            'cyclone': create_standard_cyclone(0.3, device="cpu"),
            'classification': create_standard_classification_system(device="cpu"),
            'feed': create_standard_feed_system(device="cpu"),
            'air': create_standard_air_system(device="cpu"),
            'ductwork': create_standard_ductwork(main_diameter=0.2, total_length=3.0),
        }

    def test_no_nan_vertices(self, all_assemblies):
        """Test that no vertices contain NaN."""
        for name, assembly in all_assemblies.items():
            verts, _ = assembly.build_mesh()
            assert not np.any(np.isnan(verts)), f"{name} has NaN vertices"

    def test_no_inf_vertices(self, all_assemblies):
        """Test that no vertices contain Inf."""
        for name, assembly in all_assemblies.items():
            verts, _ = assembly.build_mesh()
            assert not np.any(np.isinf(verts)), f"{name} has Inf vertices"

    def test_valid_indices(self, all_assemblies):
        """Test that all indices are valid."""
        for name, assembly in all_assemblies.items():
            verts, idx = assembly.build_mesh()
            assert np.all(idx >= 0), f"{name} has negative indices"
            assert np.all(idx < len(verts)), f"{name} has out-of-range indices"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAssemblyIntegration:
    """Integration tests for assembly modules."""

    def test_all_assemblies_create_warp_mesh(self):
        """Test that all assemblies can create Warp meshes."""
        assemblies = [
            create_standard_cyclone(0.3, device="cpu"),
            create_standard_classification_system(device="cpu"),
            create_standard_feed_system(device="cpu"),
            create_standard_air_system(device="cpu"),
            create_standard_ductwork(main_diameter=0.2, total_length=3.0),
        ]

        for assembly in assemblies:
            # CycloneAssembly uses get_warp_mesh, others use to_warp_mesh
            if hasattr(assembly, 'to_warp_mesh'):
                mesh = assembly.to_warp_mesh()
            else:
                mesh = assembly.get_warp_mesh()
            assert mesh is not None

    def test_imports_from_geometry_package(self):
        """Test that assemblies can be imported from geometry package."""
        from airclassifier.geometry import (
            CycloneAssembly,
            ClassificationSystemAssembly,
            FeedSystemAssembly,
            AirSystemAssembly,
            DuctworkSystemAssembly,
        )
        assert CycloneAssembly is not None
        assert ClassificationSystemAssembly is not None
        assert FeedSystemAssembly is not None
        assert AirSystemAssembly is not None
        assert DuctworkSystemAssembly is not None
