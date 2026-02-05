"""
Tests for Phase 1 classification system components.

Tests for:
- ZigzagClassifier
- VenturiEducator
- MultiCycloneSystem
- BagFilter
- ClassificationSystemAssembly
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from airclassifier.geometry.components import (
    ZigzagClassifier, ZigzagClassifierParams, create_standard_zigzag_classifier,
    VenturiEducator, VenturiEducatorParams, create_standard_venturi_eductor,
    MultiCycloneSystem, MultiCycloneParams, CycloneStageParams,
    create_protein_separation_cyclones, create_two_stage_cyclones,
    BagFilter, BagFilterParams, create_standard_bag_filter,
)
from airclassifier.geometry.assembly import (
    ClassificationSystemAssembly, ClassificationSystemParams,
    create_standard_classification_system, create_protein_separation_system,
)
from airclassifier.utils.constants import PI


# =============================================================================
# ZIGZAG CLASSIFIER TESTS
# =============================================================================

class TestZigzagClassifierParams:
    """Tests for ZigzagClassifierParams."""

    def test_default_params_creation(self):
        """Test creating params with valid values."""
        params = ZigzagClassifierParams(
            channel_width=0.15,
            channel_depth=0.30,
            num_stages=5,
            stage_height=0.225,
            plate_angle=np.radians(30.0),
            plate_length_ratio=0.5,
            plate_thickness=0.003,
            feed_stage=3,
            feed_width=0.075,
            feed_angle=0.0,
            air_inlet_width=0.15,
            air_inlet_height=0.075,
            fines_outlet_width=0.15,
            fines_outlet_height=0.075,
            coarse_outlet_width=0.075,
            coarse_outlet_height=0.045,
            wall_thickness=0.003,
        )
        assert params.num_stages == 5
        assert params.channel_width == 0.15

    def test_total_height(self):
        """Test total height calculation."""
        params = ZigzagClassifierParams(
            channel_width=0.15,
            channel_depth=0.30,
            num_stages=5,
            stage_height=0.2,
            plate_angle=np.radians(30.0),
            plate_length_ratio=0.5,
            plate_thickness=0.003,
            feed_stage=3,
            feed_width=0.075,
            feed_angle=0.0,
            air_inlet_width=0.15,
            air_inlet_height=0.075,
            fines_outlet_width=0.15,
            fines_outlet_height=0.075,
            coarse_outlet_width=0.075,
            coarse_outlet_height=0.045,
            wall_thickness=0.003,
        )
        assert_allclose(params.total_height, 1.0, rtol=1e-6)

    def test_invalid_feed_stage(self):
        """Test that invalid feed stage raises error."""
        with pytest.raises(ValueError, match="Feed stage must be between"):
            ZigzagClassifierParams(
                channel_width=0.15,
                channel_depth=0.30,
                num_stages=5,
                stage_height=0.2,
                plate_angle=np.radians(30.0),
                plate_length_ratio=0.5,
                plate_thickness=0.003,
                feed_stage=10,  # Invalid: > num_stages
                feed_width=0.075,
                feed_angle=0.0,
                air_inlet_width=0.15,
                air_inlet_height=0.075,
                fines_outlet_width=0.15,
                fines_outlet_height=0.075,
                coarse_outlet_width=0.075,
                coarse_outlet_height=0.045,
                wall_thickness=0.003,
            )

    def test_channel_cross_section_area(self):
        """Test cross-section area calculation."""
        params = ZigzagClassifierParams(
            channel_width=0.15,
            channel_depth=0.30,
            num_stages=5,
            stage_height=0.2,
            plate_angle=np.radians(30.0),
            plate_length_ratio=0.5,
            plate_thickness=0.003,
            feed_stage=3,
            feed_width=0.075,
            feed_angle=0.0,
            air_inlet_width=0.15,
            air_inlet_height=0.075,
            fines_outlet_width=0.15,
            fines_outlet_height=0.075,
            coarse_outlet_width=0.075,
            coarse_outlet_height=0.045,
            wall_thickness=0.003,
        )
        expected_area = 0.15 * 0.30
        assert_allclose(params.channel_cross_section_area, expected_area, rtol=1e-6)


class TestZigzagClassifier:
    """Tests for ZigzagClassifier component."""

    @pytest.fixture
    def classifier(self):
        """Create a standard zigzag classifier."""
        return create_standard_zigzag_classifier(channel_width=0.15, num_stages=5)

    def test_classifier_creation(self, classifier):
        """Test classifier is created correctly."""
        assert classifier.params.num_stages == 5
        assert classifier.params.channel_width == 0.15

    def test_mesh_generation(self, classifier):
        """Test mesh generation."""
        vertices, indices, normals = classifier.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0
        assert len(vertices) > 0

    def test_mesh_valid_indices(self, classifier):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = classifier.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_stage_corners_calculated(self, classifier):
        """Test that stage corners are calculated."""
        assert len(classifier.stage_corners) == classifier.params.num_stages + 1

    def test_get_stage_center(self, classifier):
        """Test getting stage center position."""
        center = classifier.get_stage_center(3)
        assert len(center) == 3
        # Center should be within stage bounds
        assert center[1] >= 0

    def test_get_stage_center_invalid(self, classifier):
        """Test invalid stage raises error."""
        with pytest.raises(ValueError):
            classifier.get_stage_center(10)

    def test_air_velocity_calculation(self, classifier):
        """Test air velocity calculation."""
        flow_rate = 0.5  # m3/s
        velocity = classifier.get_air_velocity(flow_rate)

        expected = flow_rate / classifier.params.channel_cross_section_area
        assert_allclose(velocity, expected, rtol=1e-6)

    def test_warp_mesh_creation(self, classifier):
        """Test Warp mesh can be created."""
        # Generate mesh first
        classifier.generate_mesh()
        mesh = classifier.to_warp_mesh(device="cpu")
        assert mesh is not None


# =============================================================================
# VENTURI EDUCTOR TESTS
# =============================================================================

class TestVenturiEducatorParams:
    """Tests for VenturiEducatorParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = VenturiEducatorParams(
            inlet_diameter=0.1,
            throat_diameter=0.05,
            outlet_diameter=0.09,
            convergent_angle=np.radians(12),
            divergent_angle=np.radians(5),
            solids_inlet_diameter=0.04,
            solids_inlet_angle=np.radians(45),
            solids_inlet_position=0.015,
        )
        assert params.inlet_diameter == 0.1
        assert params.throat_diameter == 0.05

    def test_area_ratio(self):
        """Test area ratio calculation."""
        params = VenturiEducatorParams(
            inlet_diameter=0.1,
            throat_diameter=0.05,
            outlet_diameter=0.09,
            convergent_angle=np.radians(12),
            divergent_angle=np.radians(5),
            solids_inlet_diameter=0.04,
            solids_inlet_angle=np.radians(45),
            solids_inlet_position=0.015,
        )
        # Area ratio = (D_inlet / D_throat)^2 = (0.1/0.05)^2 = 4
        assert_allclose(params.area_ratio, 4.0, rtol=1e-6)

    def test_calculated_lengths(self):
        """Test that lengths are calculated from angles."""
        params = VenturiEducatorParams(
            inlet_diameter=0.1,
            throat_diameter=0.05,
            outlet_diameter=0.09,
            convergent_angle=np.radians(12),
            divergent_angle=np.radians(5),
            solids_inlet_diameter=0.04,
            solids_inlet_angle=np.radians(45),
            solids_inlet_position=0.015,
        )
        # Convergent length should be calculated
        assert params.convergent_length > 0
        assert params.divergent_length > 0
        assert params.throat_length > 0


class TestVenturiEducator:
    """Tests for VenturiEducator component."""

    @pytest.fixture
    def eductor(self):
        """Create a standard venturi eductor."""
        return create_standard_venturi_eductor(inlet_diameter=0.1, throat_ratio=0.5)

    def test_eductor_creation(self, eductor):
        """Test eductor is created correctly."""
        assert eductor.params.inlet_diameter == 0.1
        assert_allclose(eductor.params.throat_diameter, 0.05, rtol=1e-6)

    def test_mesh_generation(self, eductor):
        """Test mesh generation."""
        vertices, indices, normals = eductor.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0
        assert len(vertices) > 0

    def test_mesh_valid_indices(self, eductor):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = eductor.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_throat_velocity(self, eductor):
        """Test throat velocity calculation using continuity."""
        inlet_velocity = 10.0  # m/s
        throat_velocity = eductor.get_velocity_at_throat(inlet_velocity)

        # v_throat = v_inlet * (A_inlet / A_throat) = v_inlet * area_ratio
        expected = inlet_velocity * eductor.params.area_ratio
        assert_allclose(throat_velocity, expected, rtol=1e-6)

    def test_pressure_drop(self, eductor):
        """Test pressure drop calculation."""
        inlet_velocity = 10.0
        density = 1.2

        pressure_drop = eductor.get_pressure_drop_at_throat(inlet_velocity, density)

        # Should be positive (pressure drop = P_inlet - P_throat > 0 since velocity increases)
        assert pressure_drop > 0

    def test_total_length(self, eductor):
        """Test total length is sum of sections."""
        p = eductor.params
        expected = p.convergent_length + p.throat_length + p.divergent_length
        assert_allclose(p.total_length, expected, rtol=1e-6)


# =============================================================================
# MULTI-CYCLONE SYSTEM TESTS
# =============================================================================

class TestCycloneStageParams:
    """Tests for CycloneStageParams."""

    def test_stage_params_creation(self):
        """Test creating stage params."""
        params = CycloneStageParams(
            name="primary",
            diameter=0.4,
            design_d50=40e-6,
        )
        assert params.name == "primary"
        assert params.diameter == 0.4
        assert params.design_d50 == 40e-6


class TestMultiCycloneSystem:
    """Tests for MultiCycloneSystem."""

    @pytest.fixture
    def three_stage_system(self):
        """Create a 3-stage protein separation system."""
        return create_protein_separation_cyclones()

    @pytest.fixture
    def two_stage_system(self):
        """Create a 2-stage cyclone system."""
        return create_two_stage_cyclones()

    def test_three_stage_creation(self, three_stage_system):
        """Test 3-stage system creation."""
        assert three_stage_system.params.num_stages == 3
        assert "primary" in three_stage_system._cyclones
        assert "secondary" in three_stage_system._cyclones
        assert "tertiary" in three_stage_system._cyclones

    def test_two_stage_creation(self, two_stage_system):
        """Test 2-stage system creation."""
        assert two_stage_system.params.num_stages == 2

    def test_mesh_generation(self, three_stage_system):
        """Test mesh generation."""
        vertices, indices, normals = three_stage_system.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, three_stage_system):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = three_stage_system.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_get_cyclone(self, three_stage_system):
        """Test getting individual cyclone."""
        primary = three_stage_system.get_cyclone("primary")
        assert primary is not None
        assert primary.params.cylinder_diameter == 0.4

    def test_get_cyclone_invalid(self, three_stage_system):
        """Test getting invalid cyclone raises error."""
        with pytest.raises(KeyError):
            three_stage_system.get_cyclone("quaternary")

    def test_system_bounds(self, three_stage_system):
        """Test system bounds calculation."""
        min_b, max_b = three_stage_system.get_system_bounds()

        assert len(min_b) == 3
        assert len(max_b) == 3
        assert np.all(max_b > min_b)

    def test_stage_info(self, three_stage_system):
        """Test stage info retrieval."""
        info = three_stage_system.get_stage_info()

        assert len(info) == 3
        assert info[0]['name'] == 'primary'
        assert info[0]['diameter'] == 400  # mm

    def test_cyclone_diameters_decreasing(self, three_stage_system):
        """Test that cyclone diameters decrease in series."""
        info = three_stage_system.get_stage_info()

        diameters = [i['diameter'] for i in info]
        assert diameters[0] > diameters[1] > diameters[2]


# =============================================================================
# BAG FILTER TESTS
# =============================================================================

class TestBagFilterParams:
    """Tests for BagFilterParams."""

    def test_params_creation(self):
        """Test creating params with valid values."""
        params = BagFilterParams(
            housing_width=2.0,
            housing_depth=2.0,
            housing_height=3.5,
            num_bags_x=6,
            num_bags_z=6,
            bag_diameter=0.15,
            bag_length=2.0,
            bag_spacing_x=0.3,
            bag_spacing_z=0.3,
            clean_air_plenum_height=0.5,
            tube_sheet_thickness=0.02,
            dirty_air_section_height=0.5,
            hopper_height=0.8,
            hopper_outlet_width=0.2,
            hopper_outlet_depth=0.2,
            hopper_angle=np.radians(60),
            dirty_air_inlet_diameter=0.3,
            clean_air_outlet_diameter=0.4,
        )
        assert params.num_bags_x == 6
        assert params.num_bags_z == 6

    def test_num_bags(self):
        """Test total bag count."""
        params = BagFilterParams(
            housing_width=2.0,
            housing_depth=2.0,
            housing_height=3.5,
            num_bags_x=6,
            num_bags_z=6,
            bag_diameter=0.15,
            bag_length=2.0,
            bag_spacing_x=0.3,
            bag_spacing_z=0.3,
            clean_air_plenum_height=0.5,
            tube_sheet_thickness=0.02,
            dirty_air_section_height=0.5,
            hopper_height=0.8,
            hopper_outlet_width=0.2,
            hopper_outlet_depth=0.2,
            hopper_angle=np.radians(60),
            dirty_air_inlet_diameter=0.3,
            clean_air_outlet_diameter=0.4,
        )
        assert params.num_bags == 36

    def test_total_filter_area(self):
        """Test filter area calculation."""
        params = BagFilterParams(
            housing_width=2.0,
            housing_depth=2.0,
            housing_height=3.5,
            num_bags_x=6,
            num_bags_z=6,
            bag_diameter=0.15,
            bag_length=2.0,
            bag_spacing_x=0.3,
            bag_spacing_z=0.3,
            clean_air_plenum_height=0.5,
            tube_sheet_thickness=0.02,
            dirty_air_section_height=0.5,
            hopper_height=0.8,
            hopper_outlet_width=0.2,
            hopper_outlet_depth=0.2,
            hopper_angle=np.radians(60),
            dirty_air_inlet_diameter=0.3,
            clean_air_outlet_diameter=0.4,
        )
        # Single bag area = pi * D * L
        single_bag = PI * 0.15 * 2.0
        expected = 36 * single_bag
        assert_allclose(params.total_filter_area, expected, rtol=1e-6)


class TestBagFilter:
    """Tests for BagFilter component."""

    @pytest.fixture
    def bag_filter(self):
        """Create a standard bag filter."""
        return create_standard_bag_filter(flow_rate_m3s=1.0, air_to_cloth=2.0)

    def test_bag_filter_creation(self, bag_filter):
        """Test bag filter is created correctly."""
        assert bag_filter.params.num_bags > 0
        assert bag_filter.params.total_filter_area > 0

    def test_mesh_generation(self, bag_filter):
        """Test mesh generation."""
        vertices, indices, normals = bag_filter.generate_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, bag_filter):
        """Test that all indices reference valid vertices."""
        vertices, indices, _ = bag_filter.generate_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_bag_positions_calculated(self, bag_filter):
        """Test bag positions are calculated."""
        assert len(bag_filter.bag_positions) == bag_filter.params.num_bags

    def test_air_to_cloth_ratio(self, bag_filter):
        """Test air-to-cloth ratio calculation."""
        ratio = bag_filter.params.get_air_to_cloth(1.0)
        # Should be close to target of 2.0
        assert 1.0 < ratio < 3.0

    def test_mesh_with_bags(self, bag_filter):
        """Test mesh generation with bags included."""
        vertices, indices, _ = bag_filter.generate_mesh(include_bags=True)
        n_with_bags = len(vertices)

        # Generate without bags
        bag_filter._vertices = None
        bag_filter._indices = None
        vertices_no_bags, _, _ = bag_filter.generate_mesh(include_bags=False)
        n_without_bags = len(vertices_no_bags)

        # With bags should have more vertices
        assert n_with_bags > n_without_bags


# =============================================================================
# CLASSIFICATION SYSTEM ASSEMBLY TESTS
# =============================================================================

class TestClassificationSystemParams:
    """Tests for ClassificationSystemParams."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ClassificationSystemParams()

        assert params.zigzag_num_stages == 5
        assert params.venturi_throat_ratio == 0.5
        assert params.primary_cyclone_diameter == 0.40

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ClassificationSystemParams(
            zigzag_num_stages=7,
            primary_cyclone_diameter=0.5,
        )

        assert params.zigzag_num_stages == 7
        assert params.primary_cyclone_diameter == 0.5


class TestClassificationSystemAssembly:
    """Tests for ClassificationSystemAssembly."""

    @pytest.fixture
    def system(self):
        """Create a standard classification system."""
        return create_standard_classification_system()

    def test_system_creation(self, system):
        """Test system is created with all components."""
        assert system.venturi is not None
        assert system.zigzag is not None
        assert system.multi_cyclone is not None
        assert system.bag_filter is not None

    def test_mesh_generation(self, system):
        """Test combined mesh generation."""
        vertices, indices = system.build_mesh()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert indices.ndim == 1
        assert len(indices) % 3 == 0

    def test_mesh_valid_indices(self, system):
        """Test that all indices reference valid vertices."""
        vertices, indices = system.build_mesh()

        assert np.all(indices >= 0)
        assert np.all(indices < len(vertices))

    def test_system_bounds(self, system):
        """Test system bounds calculation."""
        min_b, max_b = system.get_bounds()

        assert len(min_b) == 3
        assert len(max_b) == 3
        assert np.all(max_b > min_b)

    def test_system_extent(self, system):
        """Test system extent calculation."""
        extent = system.get_system_extent()

        assert len(extent) == 3
        assert np.all(extent > 0)

    def test_get_component(self, system):
        """Test getting individual component."""
        venturi = system.get_component('venturi')
        assert venturi is system.venturi

    def test_get_component_invalid(self, system):
        """Test getting invalid component raises error."""
        with pytest.raises(KeyError):
            system.get_component('invalid')

    def test_component_positions(self, system):
        """Test component positions are available."""
        positions = system.get_component_positions()

        assert 'venturi' in positions
        assert 'zigzag' in positions
        assert 'multi_cyclone' in positions
        assert 'bag_filter' in positions

        for name, pos in positions.items():
            assert len(pos) == 3

    def test_vertices_property(self, system):
        """Test vertices property triggers build."""
        verts = system.vertices
        assert verts.shape[1] == 3

    def test_indices_property(self, system):
        """Test indices property triggers build."""
        idx = system.indices
        assert idx.ndim == 1
        assert len(idx) % 3 == 0


class TestCreateProteinSeparationSystem:
    """Tests for create_protein_separation_system factory."""

    def test_default_throughput(self):
        """Test system creation with default throughput."""
        system = create_protein_separation_system()
        assert system is not None

    def test_scaled_throughput(self):
        """Test system scaling with different throughput."""
        small = create_protein_separation_system(throughput_kg_h=50)
        large = create_protein_separation_system(throughput_kg_h=200)

        # Large system should have larger components
        assert large.params.primary_cyclone_diameter > small.params.primary_cyclone_diameter


# =============================================================================
# MESH QUALITY TESTS
# =============================================================================

class TestPhase1MeshQuality:
    """Tests for mesh quality of Phase 1 components."""

    @pytest.fixture
    def all_components(self):
        """Create all Phase 1 components."""
        return {
            'zigzag': create_standard_zigzag_classifier(),
            'venturi': create_standard_venturi_eductor(),
            'multi_cyclone': create_protein_separation_cyclones(),
            'bag_filter': create_standard_bag_filter(),
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
                assert area > 1e-10, f"{name} has degenerate triangle {i}"

    def test_consistent_winding(self, all_components):
        """Test basic winding consistency (all normals should have consistent direction)."""
        for name, component in all_components.items():
            verts, idx, normals = component.generate_mesh()

            # Check that normals are unit vectors (where non-zero)
            norms = np.linalg.norm(normals, axis=1)
            non_zero = norms > 1e-6
            if np.any(non_zero):
                assert np.allclose(norms[non_zero], 1.0, atol=0.1), \
                    f"{name} has non-unit normals"
