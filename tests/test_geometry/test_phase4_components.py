"""
Tests for Phase 4 ductwork components.

Tests cover:
- Round and Rectangular Ducts
- Transitions (round-to-round, round-to-rect, rect-to-round, rect-to-rect)
- Elbows (round, rectangular, mitered)
- Diverter Valves (flap, rotating, plug)
"""

import pytest
import numpy as np

from airclassifier.geometry.components.ductwork import (
    RoundDuct,
    RoundDuctParams,
    RectangularDuct,
    RectangularDuctParams,
    create_standard_round_duct,
    create_standard_rectangular_duct,
    create_duct_for_flow,
)
from airclassifier.geometry.components.transitions import (
    Transition,
    TransitionParams,
    create_round_reducer,
    create_round_to_rect_transition,
    create_rect_to_round_transition,
)
from airclassifier.geometry.components.elbows import (
    Elbow,
    ElbowParams,
    create_90_degree_elbow,
    create_45_degree_elbow,
    create_mitered_elbow,
    create_elbow_with_vanes,
)
from airclassifier.geometry.components.diverter import (
    DiverterValve,
    DiverterValveParams,
    create_flap_diverter,
    create_rotating_diverter,
    create_plug_diverter,
    create_diverter_for_classifier,
)


# ============================================================================
# Round Duct Tests
# ============================================================================

class TestRoundDuctParams:
    """Tests for RoundDuctParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = RoundDuctParams(
            diameter=0.2,
            length=1.0,
        )
        assert params.diameter == 0.2
        assert params.length == 1.0
        assert params.wall_thickness == 0.002  # default
    
    def test_radius(self):
        """Test radius property."""
        params = RoundDuctParams(diameter=0.2, length=1.0)
        assert params.radius == 0.1
    
    def test_outer_diameter(self):
        """Test outer diameter calculation."""
        params = RoundDuctParams(diameter=0.2, length=1.0, wall_thickness=0.003)
        assert abs(params.outer_diameter - 0.206) < 1e-9
    
    def test_cross_section_area(self):
        """Test cross-sectional area calculation."""
        params = RoundDuctParams(diameter=0.2, length=1.0)
        expected = np.pi * 0.1 ** 2
        assert abs(params.cross_section_area - expected) < 1e-6
    
    def test_hydraulic_diameter(self):
        """Test hydraulic diameter (should equal diameter for round)."""
        params = RoundDuctParams(diameter=0.2, length=1.0)
        assert params.hydraulic_diameter == 0.2
    
    def test_velocity_calculation(self):
        """Test flow velocity calculation."""
        params = RoundDuctParams(diameter=0.2, length=1.0)
        flow_rate = 0.1  # m³/s
        velocity = params.get_velocity(flow_rate)
        expected = flow_rate / params.cross_section_area
        assert abs(velocity - expected) < 1e-6
    
    def test_pressure_drop(self):
        """Test pressure drop calculation."""
        params = RoundDuctParams(diameter=0.2, length=10.0)
        dp = params.get_pressure_drop(0.1)  # 0.1 m³/s
        assert dp > 0


class TestRoundDuct:
    """Tests for RoundDuct class."""
    
    def test_duct_creation(self):
        """Test round duct creation."""
        params = RoundDuctParams(diameter=0.2, length=1.0)
        duct = RoundDuct(params)
        assert duct.params.diameter == 0.2
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        duct = create_standard_round_duct(diameter=0.2, length=1.0)
        verts, inds, norms = duct.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
        assert len(norms) == len(verts)
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        duct = create_standard_round_duct(diameter=0.2, length=1.0)
        verts = duct.vertices
        inds = duct.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_inlet_outlet_positions(self):
        """Test inlet and outlet position calculations."""
        duct = create_standard_round_duct(
            diameter=0.2, 
            length=1.0,
            center=(0, 0, 0),
            direction=(0, 0, 1)
        )
        
        inlet = duct.get_inlet_position()
        outlet = duct.get_outlet_position()
        
        assert inlet == (0, 0, 0)
        assert abs(outlet[2] - 1.0) < 1e-6
    
    def test_bounds(self):
        """Test bounding box calculation."""
        duct = create_standard_round_duct(diameter=0.2, length=1.0)
        bounds_min, bounds_max = duct.get_bounds()
        
        assert bounds_min[2] < bounds_max[2]


# ============================================================================
# Rectangular Duct Tests
# ============================================================================

class TestRectangularDuctParams:
    """Tests for RectangularDuctParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = RectangularDuctParams(
            width=0.3,
            height=0.2,
            length=1.0,
        )
        assert params.width == 0.3
        assert params.height == 0.2
    
    def test_cross_section_area(self):
        """Test cross-sectional area calculation."""
        params = RectangularDuctParams(width=0.3, height=0.2, length=1.0)
        expected = 0.3 * 0.2
        assert abs(params.cross_section_area - expected) < 1e-6
    
    def test_hydraulic_diameter(self):
        """Test hydraulic diameter calculation."""
        params = RectangularDuctParams(width=0.3, height=0.2, length=1.0)
        # Dh = 4A/P = 4*0.06/(2*0.5) = 0.24
        expected = 4 * 0.06 / 1.0
        assert abs(params.hydraulic_diameter - expected) < 1e-6
    
    def test_aspect_ratio(self):
        """Test aspect ratio."""
        params = RectangularDuctParams(width=0.3, height=0.2, length=1.0)
        assert abs(params.aspect_ratio - 1.5) < 1e-9


class TestRectangularDuct:
    """Tests for RectangularDuct class."""
    
    def test_duct_creation(self):
        """Test rectangular duct creation."""
        params = RectangularDuctParams(width=0.3, height=0.2, length=1.0)
        duct = RectangularDuct(params)
        assert duct.params.width == 0.3
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        duct = create_standard_rectangular_duct(width=0.3, height=0.2, length=1.0)
        verts, inds, norms = duct.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        duct = create_standard_rectangular_duct(width=0.3, height=0.2, length=1.0)
        verts = duct.vertices
        inds = duct.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


class TestDuctFactoryFunctions:
    """Tests for duct factory functions."""
    
    def test_create_duct_for_flow_round(self):
        """Test creating duct sized for flow rate."""
        duct = create_duct_for_flow(
            flow_rate_m3_h=1000,
            velocity_target=15.0,
            length=2.0,
            duct_type="round"
        )
        assert duct.params.diameter > 0
        assert duct.params.length == 2.0
    
    def test_create_duct_for_flow_rectangular(self):
        """Test creating rectangular duct for flow rate."""
        duct = create_duct_for_flow(
            flow_rate_m3_h=1000,
            velocity_target=15.0,
            length=2.0,
            duct_type="rectangular"
        )
        assert duct.params.width > 0
        assert duct.params.height > 0


# ============================================================================
# Transition Tests
# ============================================================================

class TestTransitionParams:
    """Tests for TransitionParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(0.2,),
            outlet_dimensions=(0.15,),
            length=0.3,
        )
        assert params.transition_type == "round_to_round"
        assert params.inlet_dimensions == (0.2,)
    
    def test_is_expansion(self):
        """Test expansion detection."""
        params = TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(0.15,),
            outlet_dimensions=(0.2,),
            length=0.3,
        )
        assert params.is_expansion
        assert not params.is_contraction
    
    def test_is_contraction(self):
        """Test contraction detection."""
        params = TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(0.2,),
            outlet_dimensions=(0.15,),
            length=0.3,
        )
        assert params.is_contraction
        assert not params.is_expansion
    
    def test_area_ratio(self):
        """Test area ratio calculation."""
        params = TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(0.2,),
            outlet_dimensions=(0.2,),
            length=0.3,
        )
        assert abs(params.area_ratio - 1.0) < 1e-6
    
    def test_pressure_loss_coefficient(self):
        """Test pressure loss coefficient estimation."""
        params = TransitionParams(
            transition_type="round_to_round",
            inlet_dimensions=(0.15,),
            outlet_dimensions=(0.2,),
            length=0.3,
        )
        K = params.get_pressure_loss_coefficient()
        assert K > 0


class TestTransition:
    """Tests for Transition class."""
    
    def test_round_to_round_creation(self):
        """Test round-to-round transition creation."""
        trans = create_round_reducer(
            inlet_diameter=0.2,
            outlet_diameter=0.15,
            length=0.3
        )
        assert trans.params.transition_type == "round_to_round"
    
    def test_round_to_round_mesh(self):
        """Test round-to-round mesh generation."""
        trans = create_round_reducer(
            inlet_diameter=0.2,
            outlet_diameter=0.15,
        )
        verts, inds, norms = trans.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_round_to_rect_creation(self):
        """Test round-to-rectangular transition."""
        trans = create_round_to_rect_transition(
            inlet_diameter=0.2,
            outlet_width=0.25,
            outlet_height=0.15,
        )
        assert trans.params.transition_type == "round_to_rect"
    
    def test_rect_to_round_creation(self):
        """Test rectangular-to-round transition."""
        trans = create_rect_to_round_transition(
            inlet_width=0.25,
            inlet_height=0.15,
            outlet_diameter=0.2,
        )
        assert trans.params.transition_type == "rect_to_round"
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        trans = create_round_reducer(
            inlet_diameter=0.2,
            outlet_diameter=0.15,
        )
        verts = trans.vertices
        inds = trans.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


# ============================================================================
# Elbow Tests
# ============================================================================

class TestElbowParams:
    """Tests for ElbowParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.3,
            bend_angle=np.pi / 2,
        )
        assert params.elbow_type == "round"
        assert params.bend_angle == np.pi / 2
    
    def test_r_d_ratio(self):
        """Test R/D ratio calculation."""
        params = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.3,
        )
        assert abs(params.r_d_ratio - 1.5) < 1e-9
    
    def test_is_tight_radius(self):
        """Test tight radius detection."""
        # Tight radius (R/D < 1.5)
        params_tight = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.2,  # R/D = 1.0
        )
        assert params_tight.is_tight_radius
        
        # Not tight (R/D >= 1.5)
        params_normal = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.4,  # R/D = 2.0
        )
        assert not params_normal.is_tight_radius
    
    def test_arc_length(self):
        """Test arc length calculation."""
        params = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.3,
            bend_angle=np.pi / 2,  # 90°
        )
        expected = 0.3 * np.pi / 2
        assert abs(params.arc_length - expected) < 1e-6
    
    def test_outlet_direction(self):
        """Test outlet direction calculation for 90° bend."""
        params = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.3,
            bend_angle=np.pi / 2,
            inlet_direction=(0, 0, 1),
            bend_axis=(1, 0, 0),
        )
        outlet_dir = params.outlet_direction
        # After 90° rotation around X-axis, (0,0,1) becomes (0,-1,0) or similar
        assert len(outlet_dir) == 3
    
    def test_pressure_loss_coefficient(self):
        """Test pressure loss coefficient estimation."""
        params = ElbowParams(
            elbow_type="round",
            diameter=0.2,
            bend_radius=0.3,
        )
        K = params.get_pressure_loss_coefficient()
        assert K > 0


class TestElbow:
    """Tests for Elbow class."""
    
    def test_90_degree_elbow_creation(self):
        """Test 90-degree elbow creation."""
        elbow = create_90_degree_elbow(diameter=0.2)
        assert abs(elbow.params.bend_angle - np.pi / 2) < 1e-6
    
    def test_45_degree_elbow_creation(self):
        """Test 45-degree elbow creation."""
        elbow = create_45_degree_elbow(diameter=0.2)
        assert abs(elbow.params.bend_angle - np.pi / 4) < 1e-6
    
    def test_mitered_elbow_creation(self):
        """Test mitered elbow creation."""
        elbow = create_mitered_elbow(diameter=0.2, num_gores=5)
        assert elbow.params.elbow_type == "mitered"
        assert elbow.params.num_gores == 5
    
    def test_elbow_with_vanes(self):
        """Test elbow with turning vanes."""
        elbow = create_elbow_with_vanes(diameter=0.2, num_vanes=3)
        assert elbow.params.turning_vanes
        assert elbow.params.num_vanes == 3
    
    def test_round_elbow_mesh(self):
        """Test round elbow mesh generation."""
        elbow = create_90_degree_elbow(diameter=0.2)
        verts, inds, norms = elbow.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mitered_elbow_mesh(self):
        """Test mitered elbow mesh generation."""
        elbow = create_mitered_elbow(diameter=0.2, num_gores=5)
        verts, inds, norms = elbow.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        elbow = create_90_degree_elbow(diameter=0.2)
        verts = elbow.vertices
        inds = elbow.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_inlet_outlet_positions(self):
        """Test inlet and outlet positions."""
        elbow = create_90_degree_elbow(diameter=0.2, center=(0, 0, 0))
        
        inlet = elbow.get_inlet_position()
        outlet = elbow.get_outlet_position()
        
        # Positions should be different after bend
        assert inlet != outlet


# ============================================================================
# Diverter Tests
# ============================================================================

class TestDiverterValveParams:
    """Tests for DiverterValveParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = DiverterValveParams(
            inlet_diameter=0.2,
            outlet_angle=np.pi / 3,
        )
        assert params.inlet_diameter == 0.2
        assert params.outlet_angle == np.pi / 3
    
    def test_default_outlet_diameters(self):
        """Test that outlet diameters default to inlet diameter."""
        params = DiverterValveParams(inlet_diameter=0.2)
        assert params.outlet1_diameter == 0.2
        assert params.outlet2_diameter == 0.2
    
    def test_radii_properties(self):
        """Test radius properties."""
        params = DiverterValveParams(inlet_diameter=0.2)
        assert params.inlet_radius == 0.1
        assert params.outlet1_radius == 0.1
        assert params.outlet2_radius == 0.1
    
    def test_area_properties(self):
        """Test area properties."""
        params = DiverterValveParams(inlet_diameter=0.2)
        expected_area = np.pi * 0.1 ** 2
        assert abs(params.inlet_area - expected_area) < 1e-6
    
    def test_flow_split(self):
        """Test flow split based on position."""
        params = DiverterValveParams(inlet_diameter=0.2, position=0.0)
        frac1, frac2 = params.get_flow_split()
        assert frac1 == 1.0
        assert frac2 == 0.0
        
        params.position = 1.0
        frac1, frac2 = params.get_flow_split()
        assert frac1 == 0.0
        assert frac2 == 1.0
        
        params.position = 0.5
        frac1, frac2 = params.get_flow_split()
        assert frac1 == 0.5
        assert frac2 == 0.5
    
    def test_pressure_loss_coefficient(self):
        """Test pressure loss coefficient estimation."""
        params = DiverterValveParams(inlet_diameter=0.2)
        K = params.get_pressure_loss_coefficient()
        assert K > 0


class TestDiverterValve:
    """Tests for DiverterValve class."""
    
    def test_flap_diverter_creation(self):
        """Test flap diverter creation."""
        diverter = create_flap_diverter(inlet_diameter=0.2)
        assert diverter.params.blade_type == "flap"
    
    def test_rotating_diverter_creation(self):
        """Test rotating diverter creation."""
        diverter = create_rotating_diverter(inlet_diameter=0.2)
        assert diverter.params.blade_type == "rotating"
    
    def test_plug_diverter_creation(self):
        """Test plug diverter creation."""
        diverter = create_plug_diverter(inlet_diameter=0.2)
        assert diverter.params.blade_type == "plug"
    
    def test_mesh_generation_flap(self):
        """Test flap diverter mesh generation."""
        diverter = create_flap_diverter(inlet_diameter=0.2)
        verts, inds, norms = diverter.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_generation_rotating(self):
        """Test rotating diverter mesh generation."""
        diverter = create_rotating_diverter(inlet_diameter=0.2)
        verts, inds, norms = diverter.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_generation_plug(self):
        """Test plug diverter mesh generation."""
        diverter = create_plug_diverter(inlet_diameter=0.2)
        verts, inds, norms = diverter.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        diverter = create_flap_diverter(inlet_diameter=0.2)
        verts = diverter.vertices
        inds = diverter.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_set_position(self):
        """Test setting diverter position."""
        diverter = create_flap_diverter(inlet_diameter=0.2)
        
        # Set to outlet2
        diverter.set_position(1.0)
        assert diverter.params.position == 1.0
        
        # Set to outlet1
        diverter.set_position(0.0)
        assert diverter.params.position == 0.0
        
        # Clamp to valid range
        diverter.set_position(1.5)
        assert diverter.params.position == 1.0
        
        diverter.set_position(-0.5)
        assert diverter.params.position == 0.0
    
    def test_outlet_positions(self):
        """Test outlet position calculations."""
        diverter = create_flap_diverter(inlet_diameter=0.2)
        
        inlet = diverter.get_inlet_position()
        outlet1 = diverter.get_outlet1_position()
        outlet2 = diverter.get_outlet2_position()
        
        # All positions should be different
        assert inlet != outlet1
        assert inlet != outlet2
        assert outlet1 != outlet2
    
    def test_diverter_for_classifier(self):
        """Test creating diverter sized for classifier."""
        diverter = create_diverter_for_classifier(inlet_diameter=0.2)
        
        # Protein outlet should be smaller
        assert diverter.params.outlet1_diameter < diverter.params.inlet_diameter


# ============================================================================
# Mesh Quality Tests
# ============================================================================

class TestPhase4MeshQuality:
    """Tests for mesh quality across all Phase 4 components."""
    
    @pytest.fixture
    def all_components(self):
        """Create all Phase 4 components."""
        return [
            create_standard_round_duct(diameter=0.2, length=1.0),
            create_standard_rectangular_duct(width=0.3, height=0.2, length=1.0),
            create_round_reducer(inlet_diameter=0.2, outlet_diameter=0.15),
            create_round_to_rect_transition(inlet_diameter=0.2, outlet_width=0.25, outlet_height=0.15),
            create_90_degree_elbow(diameter=0.2),
            create_45_degree_elbow(diameter=0.2),
            create_mitered_elbow(diameter=0.2, num_gores=5),
            create_flap_diverter(inlet_diameter=0.2),
            create_rotating_diverter(inlet_diameter=0.2),
            create_plug_diverter(inlet_diameter=0.2),
        ]
    
    def test_no_nan_vertices(self, all_components):
        """Test that no vertices contain NaN values."""
        for component in all_components:
            verts = component.vertices
            assert not np.any(np.isnan(verts)), f"NaN found in {type(component).__name__}"
    
    def test_no_inf_vertices(self, all_components):
        """Test that no vertices contain infinite values."""
        for component in all_components:
            verts = component.vertices
            assert not np.any(np.isinf(verts)), f"Inf found in {type(component).__name__}"
    
    def test_no_degenerate_triangles(self, all_components):
        """Test that no triangles are degenerate (zero area)."""
        for component in all_components:
            verts = component.vertices
            inds = component.indices.reshape(-1, 3)
            
            for tri in inds[:100]:  # Check first 100 triangles
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = np.linalg.norm(cross) / 2
                assert area > 1e-10, f"Degenerate triangle in {type(component).__name__}"
    
    def test_vertices_count(self, all_components):
        """Test that all components have reasonable vertex counts."""
        for component in all_components:
            verts = component.vertices
            # Should have at least some vertices but not too many
            assert len(verts) >= 24, f"Too few vertices in {type(component).__name__}"
            assert len(verts) < 100000, f"Too many vertices in {type(component).__name__}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase4Integration:
    """Integration tests for Phase 4 components."""
    
    def test_all_components_create_warp_mesh(self):
        """Test that all components can create Warp meshes."""
        pytest.importorskip("warp")
        
        components = [
            create_standard_round_duct(diameter=0.2, length=1.0),
            create_round_reducer(inlet_diameter=0.2, outlet_diameter=0.15),
            create_90_degree_elbow(diameter=0.2),
            create_flap_diverter(inlet_diameter=0.2),
        ]
        
        for component in components:
            mesh = component.to_warp_mesh()
            assert mesh is not None
    
    def test_duct_system_sizing_reasonable(self):
        """Test that duct sizing is reasonable for typical air flows."""
        # Typical classifier flow: 2000 m³/h at 15 m/s
        duct = create_duct_for_flow(
            flow_rate_m3_h=2000,
            velocity_target=15.0,
            duct_type="round"
        )
        
        # Diameter should be reasonable (100-300mm range)
        assert 0.1 <= duct.params.diameter <= 0.3
        
        # Check actual velocity
        flow_rate_m3_s = 2000 / 3600
        actual_velocity = duct.params.get_velocity(flow_rate_m3_s)
        assert 10 <= actual_velocity <= 20
    
    def test_elbow_directions_perpendicular(self):
        """Test that 90° elbow inlet and outlet are perpendicular."""
        elbow = create_90_degree_elbow(
            diameter=0.2,
            center=(0, 0, 0),
            inlet_direction=(0, 0, 1),
            bend_axis=(1, 0, 0)
        )
        
        inlet_dir = np.array(elbow.params.inlet_direction_normalized)
        outlet_dir = np.array(elbow.params.outlet_direction)
        
        # Dot product should be close to 0 for perpendicular
        dot = np.dot(inlet_dir, outlet_dir)
        assert abs(dot) < 0.1  # Allow some tolerance
    
    def test_transition_connects_different_sizes(self):
        """Test that transitions properly connect different duct sizes."""
        trans = create_round_reducer(
            inlet_diameter=0.25,
            outlet_diameter=0.2
        )
        
        # Check inlet and outlet areas
        inlet_area = np.pi * (0.25/2)**2
        outlet_area = np.pi * (0.2/2)**2
        
        assert abs(trans.params.inlet_area - inlet_area) < 1e-6
        assert abs(trans.params.outlet_area - outlet_area) < 1e-6
        assert trans.params.is_contraction
