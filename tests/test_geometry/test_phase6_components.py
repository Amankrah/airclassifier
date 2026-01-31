"""
Tests for Phase 6 Support & Exhaust components.

Tests cover:
- Equipment Legs (tubular, channel, adjustable)
- Structural Frames
- Silencers
- Exhaust Stacks
- SupportExhaustAssembly
"""

import pytest
import numpy as np

from airclassifier.geometry.components.supports import (
    EquipmentLegs,
    EquipmentLegParams,
    create_tubular_legs,
    create_adjustable_legs,
    create_channel_legs,
    StructuralFrame,
    StructuralFrameParams,
    create_standard_frame,
    create_equipment_skid,
    create_mezzanine_frame,
)
from airclassifier.geometry.components.silencer import (
    Silencer,
    SilencerParams,
    create_absorptive_silencer,
    create_splitter_silencer,
    create_reactive_silencer,
)
from airclassifier.geometry.components.exhaust_stack import (
    ExhaustStack,
    ExhaustStackParams,
    create_standard_exhaust_stack,
    create_tall_stack,
    create_short_vent_stack,
)
from airclassifier.geometry.assembly import (
    SupportExhaustAssembly,
    SupportExhaustParams,
    create_standard_support_exhaust,
    create_compact_support,
    create_industrial_support,
)


# ============================================================================
# Equipment Legs Tests
# ============================================================================

class TestEquipmentLegParams:
    """Tests for EquipmentLegParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = EquipmentLegParams(
            leg_type="tubular",
            num_legs=4,
            leg_height=0.5,
        )
        assert params.leg_type == "tubular"
        assert params.num_legs == 4
        assert params.leg_height == 0.5
    
    def test_leg_radius(self):
        """Test leg radius property."""
        params = EquipmentLegParams(leg_diameter=0.076)
        assert params.leg_radius == 0.038
    
    def test_leg_positions(self):
        """Test leg position calculation."""
        params = EquipmentLegParams(num_legs=4, mounting_diameter=1.0)
        positions = params.get_leg_positions()
        assert len(positions) == 4


class TestEquipmentLegs:
    """Tests for EquipmentLegs class."""
    
    def test_tubular_legs_creation(self):
        """Test tubular legs creation."""
        legs = create_tubular_legs(num_legs=4, height=0.5)
        assert legs.params.leg_type == "tubular"
        assert legs.params.num_legs == 4
    
    def test_adjustable_legs_creation(self):
        """Test adjustable legs creation."""
        legs = create_adjustable_legs(num_legs=4, height=0.5)
        assert legs.params.leg_type == "adjustable"
    
    def test_channel_legs_creation(self):
        """Test channel legs creation."""
        legs = create_channel_legs(num_legs=4, height=0.5)
        assert legs.params.leg_type == "channel"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        legs = create_tubular_legs(num_legs=4, height=0.5)
        verts, inds, norms = legs.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
        assert len(norms) == len(verts)
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        legs = create_tubular_legs(num_legs=4, height=0.5)
        verts = legs.vertices
        inds = legs.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_load_capacity(self):
        """Test total load capacity calculation."""
        legs = create_tubular_legs(num_legs=4, height=0.5, load_capacity=500)
        assert legs.get_total_load_capacity() == 2000


# ============================================================================
# Structural Frame Tests
# ============================================================================

class TestStructuralFrameParams:
    """Tests for StructuralFrameParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = StructuralFrameParams(
            width=2.0,
            depth=2.0,
            height=3.0,
        )
        assert params.width == 2.0
        assert params.height == 3.0
    
    def test_column_positions(self):
        """Test column position calculation."""
        params = StructuralFrameParams(width=2.0, depth=2.0)
        positions = params.get_column_positions()
        assert len(positions) == 4


class TestStructuralFrame:
    """Tests for StructuralFrame class."""
    
    def test_standard_frame_creation(self):
        """Test standard frame creation."""
        frame = create_standard_frame(width=2.0, depth=2.0, height=3.0)
        assert frame.params.width == 2.0
    
    def test_equipment_skid_creation(self):
        """Test equipment skid creation."""
        skid = create_equipment_skid(width=1.5, depth=1.0, height=0.3)
        assert skid.params.has_bracing == False
    
    def test_mezzanine_frame_creation(self):
        """Test mezzanine frame creation."""
        frame = create_mezzanine_frame(width=4.0, depth=3.0, height=3.0)
        assert frame.params.has_bracing == True
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        frame = create_standard_frame(width=2.0, depth=2.0, height=3.0)
        verts, inds, norms = frame.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        frame = create_standard_frame(width=2.0, depth=2.0, height=3.0)
        verts = frame.vertices
        inds = frame.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_platform_area(self):
        """Test platform area calculation."""
        frame = create_standard_frame(
            width=2.0, depth=2.0, height=3.0,
            platform_levels=[3.0]
        )
        area = frame.get_platform_area()
        assert area > 0


# ============================================================================
# Silencer Tests
# ============================================================================

class TestSilencerParams:
    """Tests for SilencerParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = SilencerParams(
            silencer_type="absorptive",
            diameter=0.3,
            length=1.0,
        )
        assert params.silencer_type == "absorptive"
        assert params.diameter == 0.3
    
    def test_radii(self):
        """Test radius properties."""
        params = SilencerParams(diameter=0.3)
        assert params.inner_radius == 0.15
        assert params.outer_radius > params.inner_radius
    
    def test_pressure_drop(self):
        """Test pressure drop calculation."""
        params = SilencerParams(diameter=0.3, num_splitters=0)
        dp = params.get_pressure_drop(velocity=10)
        assert dp > 0


class TestSilencer:
    """Tests for Silencer class."""
    
    def test_absorptive_silencer_creation(self):
        """Test absorptive silencer creation."""
        silencer = create_absorptive_silencer(diameter=0.3, length=1.0)
        assert silencer.params.silencer_type == "absorptive"
    
    def test_splitter_silencer_creation(self):
        """Test splitter silencer creation."""
        silencer = create_splitter_silencer(diameter=0.5, length=1.5, num_splitters=2)
        assert silencer.params.num_splitters == 2
    
    def test_reactive_silencer_creation(self):
        """Test reactive silencer creation."""
        silencer = create_reactive_silencer(diameter=0.3, length=0.6)
        assert silencer.params.silencer_type == "reactive"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        silencer = create_absorptive_silencer(diameter=0.3, length=1.0)
        verts, inds, norms = silencer.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        silencer = create_absorptive_silencer(diameter=0.3, length=1.0)
        verts = silencer.vertices
        inds = silencer.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_insertion_loss(self):
        """Test insertion loss estimation."""
        silencer = create_absorptive_silencer(diameter=0.3, length=1.0, insertion_loss=15)
        loss = silencer.get_insertion_loss(frequency=500)
        assert loss > 0


# ============================================================================
# Exhaust Stack Tests
# ============================================================================

class TestExhaustStackParams:
    """Tests for ExhaustStackParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = ExhaustStackParams(
            diameter=0.3,
            height=3.0,
        )
        assert params.diameter == 0.3
        assert params.height == 3.0
    
    def test_radius(self):
        """Test radius property."""
        params = ExhaustStackParams(diameter=0.3)
        assert params.radius == 0.15
    
    def test_cross_sectional_area(self):
        """Test cross-sectional area."""
        params = ExhaustStackParams(diameter=0.3)
        area = params.cross_sectional_area
        expected = np.pi * (0.15 - params.wall_thickness) ** 2
        assert abs(area - expected) < 1e-6
    
    def test_flow_rate_for_velocity(self):
        """Test flow rate calculation."""
        params = ExhaustStackParams(diameter=0.3, discharge_velocity=15)
        flow = params.get_flow_rate_for_velocity()
        assert flow > 0


class TestExhaustStack:
    """Tests for ExhaustStack class."""
    
    def test_standard_stack_creation(self):
        """Test standard stack creation."""
        stack = create_standard_exhaust_stack(diameter=0.3, height=3.0)
        assert stack.params.diameter == 0.3
    
    def test_tall_stack_creation(self):
        """Test tall stack creation."""
        stack = create_tall_stack(diameter=0.4, height=10.0)
        assert stack.params.guy_wire_lugs == True
    
    def test_short_vent_creation(self):
        """Test short vent stack creation."""
        stack = create_short_vent_stack(diameter=0.25, height=1.5)
        assert stack.params.cap_type == "chinese_hat"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        stack = create_standard_exhaust_stack(diameter=0.3, height=3.0)
        verts, inds, norms = stack.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        stack = create_standard_exhaust_stack(diameter=0.3, height=3.0)
        verts = stack.vertices
        inds = stack.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_cap_types(self):
        """Test different cap types."""
        for cap_type in ["conical", "chinese_hat", "H_cap"]:
            stack = create_standard_exhaust_stack(
                diameter=0.3, height=3.0, cap_type=cap_type
            )
            assert stack.params.cap_type == cap_type
            assert len(stack.vertices) > 0
    
    def test_exit_velocity(self):
        """Test exit velocity calculation."""
        stack = create_standard_exhaust_stack(diameter=0.3, height=3.0)
        velocity = stack.get_exit_velocity(flow_rate=1.0)
        assert velocity > 0


# ============================================================================
# SupportExhaustAssembly Tests
# ============================================================================

class TestSupportExhaustParams:
    """Tests for SupportExhaustParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = SupportExhaustParams(
            frame_width=2.5,
            frame_height=3.0,
        )
        assert params.frame_width == 2.5
        assert params.frame_height == 3.0
    
    def test_total_height(self):
        """Test total height calculation."""
        params = SupportExhaustParams(
            frame_height=3.0,
            leg_height=0.5,
            stack_height=4.0,
            has_legs=True,
            has_exhaust_stack=True
        )
        assert params.total_height == 3.0 + 0.5 + 4.0


class TestSupportExhaustAssembly:
    """Tests for SupportExhaustAssembly class."""
    
    @pytest.fixture
    def assembly(self):
        """Create a standard support/exhaust assembly."""
        return create_standard_support_exhaust(frame_height=3.0, stack_height=4.0)
    
    def test_assembly_creation(self, assembly):
        """Test assembly is created correctly."""
        assert len(assembly._components) > 0
    
    def test_build_mesh(self, assembly):
        """Test mesh building."""
        verts, idx = assembly.build_mesh()
        
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
    
    def test_get_bounds(self, assembly):
        """Test bounding box calculation."""
        min_c, max_c = assembly.get_bounds()
        
        assert len(min_c) == 3
        assert len(max_c) == 3
        # Stack should extend above frame
        assert max_c[2] > assembly.params.frame_height
    
    def test_get_component(self, assembly):
        """Test getting components by name."""
        frame = assembly.get_component('frame')
        assert frame is not None
    
    def test_get_component_names(self, assembly):
        """Test getting component names."""
        names = assembly.get_component_names()
        assert 'frame' in names
    
    def test_get_system_summary(self, assembly):
        """Test getting system summary."""
        summary = assembly.get_system_summary()
        assert 'frame_dimensions' in summary
        assert 'total_height_m' in summary
        assert 'num_components' in summary
    
    def test_compact_support(self):
        """Test compact support creation."""
        assembly = create_compact_support(frame_width=1.5, frame_depth=1.5)
        assert assembly.params.has_silencer == False
    
    def test_industrial_support(self):
        """Test industrial support creation."""
        assembly = create_industrial_support(frame_width=4.0, frame_depth=3.0)
        assert assembly.params.num_platform_levels == 2
        assert assembly.params.has_silencer == True


# ============================================================================
# Mesh Quality Tests
# ============================================================================

class TestPhase6MeshQuality:
    """Tests for mesh quality across all Phase 6 components."""
    
    @pytest.fixture
    def all_components(self):
        """Create all Phase 6 components."""
        return [
            create_tubular_legs(num_legs=4, height=0.5),
            create_adjustable_legs(num_legs=4, height=0.5),
            create_channel_legs(num_legs=4, height=0.5),
            create_standard_frame(width=2.0, depth=2.0, height=3.0),
            create_equipment_skid(width=1.5, depth=1.0, height=0.3),
            create_absorptive_silencer(diameter=0.3, length=1.0),
            create_splitter_silencer(diameter=0.5, length=1.5, num_splitters=2),
            create_standard_exhaust_stack(diameter=0.3, height=3.0),
            create_tall_stack(diameter=0.4, height=10.0),
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
    
    def test_vertices_count(self, all_components):
        """Test that all components have reasonable vertex counts."""
        for component in all_components:
            verts = component.vertices
            assert len(verts) >= 10, f"Too few vertices in {type(component).__name__}"
            assert len(verts) < 100000, f"Too many vertices in {type(component).__name__}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase6Integration:
    """Integration tests for Phase 6 components."""
    
    def test_all_components_create_warp_mesh(self):
        """Test that all components can create Warp meshes."""
        pytest.importorskip("warp")
        
        components = [
            create_tubular_legs(num_legs=4, height=0.5),
            create_standard_frame(width=2.0, depth=2.0, height=3.0),
            create_absorptive_silencer(diameter=0.3, length=1.0),
            create_standard_exhaust_stack(diameter=0.3, height=3.0),
        ]
        
        for component in components:
            mesh = component.to_warp_mesh()
            assert mesh is not None
    
    def test_assembly_creates_warp_mesh(self):
        """Test that assembly can create Warp mesh."""
        pytest.importorskip("warp")
        
        assembly = create_standard_support_exhaust(frame_height=3.0)
        mesh = assembly.to_warp_mesh()
        assert mesh is not None
    
    def test_imports_from_components_package(self):
        """Test that Phase 6 components can be imported from components."""
        from airclassifier.geometry.components import (
            EquipmentLegs,
            StructuralFrame,
            Silencer,
            ExhaustStack,
        )
        assert EquipmentLegs is not None
        assert StructuralFrame is not None
        assert Silencer is not None
        assert ExhaustStack is not None
    
    def test_imports_from_assembly_package(self):
        """Test that assembly can be imported from assembly package."""
        from airclassifier.geometry.assembly import (
            SupportExhaustAssembly,
            create_standard_support_exhaust,
        )
        assert SupportExhaustAssembly is not None
        assert create_standard_support_exhaust is not None
