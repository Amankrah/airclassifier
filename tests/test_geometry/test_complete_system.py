"""
Tests for Complete Air Classifier System Integration.

Tests cover:
- CompleteClassifierAssembly creation and configuration
- Subsystem integration (all 6 phases)
- Instrumentation mounting
- Mesh generation and validity
- Bill of materials generation
- System summaries
"""

import pytest
import numpy as np

from airclassifier.geometry.assembly import (
    CompleteClassifierAssembly,
    CompleteSystemParams,
    create_complete_classifier_system,
    create_pilot_scale_system,
    create_production_scale_system,
    create_minimal_classifier_system,
)


# ============================================================================
# CompleteSystemParams Tests
# ============================================================================

class TestCompleteSystemParams:
    """Tests for CompleteSystemParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = CompleteSystemParams(
            throughput_kg_h=500,
            cut_size_um=20,
        )
        assert params.throughput_kg_h == 500
        assert params.cut_size_um == 20
    
    def test_default_includes(self):
        """Test that all systems are included by default."""
        params = CompleteSystemParams()
        assert params.include_feed_system == True
        assert params.include_air_system == True
        assert params.include_ductwork == True
        assert params.include_safety == True
        assert params.include_instrumentation == True
        assert params.include_support_structure == True
        assert params.include_exhaust == True
    
    def test_custom_positions(self):
        """Test custom position parameters."""
        params = CompleteSystemParams(
            feed_position=(-3.0, 0.0, 0.0),
            classifier_position=(0.0, 0.0, 0.0),
        )
        assert params.feed_position == (-3.0, 0.0, 0.0)
    
    def test_sizing_parameters(self):
        """Test sizing parameters."""
        params = CompleteSystemParams(
            classifier_width=0.2,
            cyclone_diameter=0.4,
            main_duct_diameter=0.25,
        )
        assert params.classifier_width == 0.2
        assert params.cyclone_diameter == 0.4
        assert params.main_duct_diameter == 0.25


# ============================================================================
# CompleteClassifierAssembly Tests
# ============================================================================

class TestCompleteClassifierAssembly:
    """Tests for CompleteClassifierAssembly class."""
    
    @pytest.fixture
    def full_system(self):
        """Create a full system with all components."""
        return create_complete_classifier_system(throughput_kg_h=500)
    
    @pytest.fixture
    def minimal_system(self):
        """Create a minimal system."""
        return create_minimal_classifier_system()
    
    def test_full_system_creation(self, full_system):
        """Test that full system is created correctly."""
        assert len(full_system.get_all_subsystem_names()) > 0
    
    def test_minimal_system_creation(self, minimal_system):
        """Test that minimal system has only classification."""
        subsystems = minimal_system.get_all_subsystem_names()
        assert 'classification' in subsystems
        assert 'feed_system' not in subsystems
    
    def test_build_mesh(self, full_system):
        """Test mesh building."""
        verts, idx = full_system.build_mesh()
        
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(idx) % 3 == 0
        assert len(verts) > 0
    
    def test_mesh_valid_indices(self, full_system):
        """Test that mesh indices are valid."""
        verts = full_system.vertices
        idx = full_system.indices
        
        assert idx.min() >= 0
        assert idx.max() < len(verts)
    
    def test_get_bounds(self, full_system):
        """Test bounding box calculation."""
        min_c, max_c = full_system.get_bounds()
        
        assert len(min_c) == 3
        assert len(max_c) == 3
        # System should have positive extent
        assert np.all(max_c > min_c)
    
    def test_get_subsystem(self, full_system):
        """Test getting subsystems by name."""
        classification = full_system.get_subsystem('classification')
        assert classification is not None
    
    def test_get_all_subsystem_names(self, full_system):
        """Test getting all subsystem names."""
        names = full_system.get_all_subsystem_names()
        assert 'classification' in names
        assert 'feed_system' in names
        assert 'air_system' in names
        assert 'support_structure' in names
    
    def test_get_all_component_names(self, full_system):
        """Test getting all component names."""
        names = full_system.get_all_component_names()
        assert 'silencer' in names
        assert 'exhaust_stack' in names
    
    def test_get_all_instrument_names(self, full_system):
        """Test getting all instrument names."""
        names = full_system.get_all_instrument_names()
        # Should have pressure, temp, sample, sight glass, grounding, explosion vents
        assert any('pressure' in name for name in names)
        assert any('temp' in name for name in names)
        assert any('sample' in name for name in names)
        assert any('grounding' in name for name in names)
    
    def test_get_system_summary(self, full_system):
        """Test getting system summary."""
        summary = full_system.get_system_summary()
        
        assert 'design_throughput_kg_h' in summary
        assert 'num_subsystems' in summary
        assert 'num_components' in summary
        assert 'num_instruments' in summary
        assert 'total_vertices' in summary
        assert 'instrument_breakdown' in summary
        assert 'includes' in summary
    
    def test_get_bill_of_materials(self, full_system):
        """Test bill of materials generation."""
        bom = full_system.get_bill_of_materials()
        
        assert len(bom) > 0
        assert all('item' in entry for entry in bom)
        assert all('type' in entry for entry in bom)
        assert all('quantity' in entry for entry in bom)
    
    def test_includes_classification(self, full_system):
        """Test that classification system is always included."""
        assert full_system.get_subsystem('classification') is not None
    
    def test_includes_feed_system(self, full_system):
        """Test that feed system is included."""
        assert full_system.get_subsystem('feed_system') is not None
    
    def test_includes_air_system(self, full_system):
        """Test that air system is included."""
        assert full_system.get_subsystem('air_system') is not None
    
    def test_includes_support_structure(self, full_system):
        """Test that support structure is included."""
        assert full_system.get_subsystem('support_structure') is not None


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_complete_classifier_system(self):
        """Test standard system creation."""
        system = create_complete_classifier_system(
            throughput_kg_h=500,
            cut_size_um=25
        )
        assert system.params.throughput_kg_h == 500
        assert system.params.cut_size_um == 25
    
    def test_create_pilot_scale_system(self):
        """Test pilot scale system creation."""
        system = create_pilot_scale_system(throughput_kg_h=100)
        assert system.params.throughput_kg_h == 100
        assert system.params.frame_width < 4.0  # Smaller frame
    
    def test_create_production_scale_system(self):
        """Test production scale system creation."""
        system = create_production_scale_system(throughput_kg_h=2000)
        assert system.params.throughput_kg_h == 2000
        assert system.params.frame_width > 4.0  # Larger frame
    
    def test_create_minimal_classifier_system(self):
        """Test minimal system creation."""
        system = create_minimal_classifier_system()
        assert system.params.include_feed_system == False
        assert system.params.include_safety == False


# ============================================================================
# Instrumentation Integration Tests
# ============================================================================

class TestInstrumentationIntegration:
    """Tests for instrumentation integration."""
    
    @pytest.fixture
    def system_with_instrumentation(self):
        """Create system with instrumentation."""
        return create_complete_classifier_system(
            throughput_kg_h=500,
            include_instrumentation=True
        )
    
    def test_pressure_ports_created(self, system_with_instrumentation):
        """Test that pressure ports are created."""
        instruments = system_with_instrumentation.get_all_instrument_names()
        pressure_ports = [n for n in instruments if 'pressure' in n]
        assert len(pressure_ports) >= 3  # At least inlet, outlet, filter
    
    def test_temperature_ports_created(self, system_with_instrumentation):
        """Test that temperature ports are created."""
        instruments = system_with_instrumentation.get_all_instrument_names()
        temp_ports = [n for n in instruments if 'temp' in n]
        assert len(temp_ports) >= 2
    
    def test_sample_ports_created(self, system_with_instrumentation):
        """Test that sample ports are created."""
        instruments = system_with_instrumentation.get_all_instrument_names()
        sample_ports = [n for n in instruments if 'sample' in n]
        assert len(sample_ports) >= 2  # Feed and product
    
    def test_explosion_vents_created(self, system_with_instrumentation):
        """Test that explosion vents are created."""
        instruments = system_with_instrumentation.get_all_instrument_names()
        vents = [n for n in instruments if 'explosion_vent' in n]
        assert len(vents) >= 2  # Cyclone and filter
    
    def test_grounding_points_created(self, system_with_instrumentation):
        """Test that grounding points are created."""
        instruments = system_with_instrumentation.get_all_instrument_names()
        grounds = [n for n in instruments if 'grounding' in n]
        assert len(grounds) >= 3


# ============================================================================
# Mesh Quality Tests
# ============================================================================

class TestSystemMeshQuality:
    """Tests for mesh quality of complete system."""
    
    @pytest.fixture
    def system(self):
        """Create a full system."""
        return create_complete_classifier_system()
    
    def test_no_nan_vertices(self, system):
        """Test that no vertices contain NaN values."""
        verts = system.vertices
        assert not np.any(np.isnan(verts))
    
    def test_no_inf_vertices(self, system):
        """Test that no vertices contain infinite values."""
        verts = system.vertices
        assert not np.any(np.isinf(verts))
    
    def test_reasonable_vertex_count(self, system):
        """Test reasonable vertex count."""
        verts = system.vertices
        assert len(verts) > 1000  # Complex system
        assert len(verts) < 1000000  # Not excessive
    
    def test_valid_triangle_indices(self, system):
        """Test that all triangle indices are valid."""
        idx = system.indices
        assert len(idx) % 3 == 0  # Divisible by 3
        assert np.all(idx >= 0)


# ============================================================================
# System Configuration Tests
# ============================================================================

class TestSystemConfiguration:
    """Tests for different system configurations."""
    
    def test_system_without_feed(self):
        """Test system without feed system."""
        system = create_complete_classifier_system(
            include_feed_system=False
        )
        assert system.get_subsystem('feed_system') is None
    
    def test_system_without_air(self):
        """Test system without air system."""
        system = create_complete_classifier_system(
            include_air_system=False
        )
        assert system.get_subsystem('air_system') is None
    
    def test_system_without_safety(self):
        """Test system without safety equipment."""
        system = create_complete_classifier_system(
            include_safety=False
        )
        instruments = system.get_all_instrument_names()
        vents = [n for n in instruments if 'explosion_vent' in n]
        assert len(vents) == 0
    
    def test_system_without_instrumentation(self):
        """Test system without instrumentation."""
        system = create_complete_classifier_system(
            include_instrumentation=False
        )
        instruments = system.get_all_instrument_names()
        # Should only have safety items if safety enabled
        assert len([n for n in instruments if 'pressure' in n]) == 0
    
    def test_system_without_support(self):
        """Test system without support structure."""
        system = create_complete_classifier_system(
            include_support_structure=False
        )
        assert system.get_subsystem('support_structure') is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestSystemIntegration:
    """Integration tests for complete system."""
    
    def test_warp_mesh_creation(self):
        """Test that system can create Warp mesh."""
        pytest.importorskip("warp")
        
        system = create_complete_classifier_system()
        mesh = system.to_warp_mesh()
        assert mesh is not None
    
    def test_imports_from_assembly_package(self):
        """Test imports from assembly package."""
        from airclassifier.geometry.assembly import (
            CompleteClassifierAssembly,
            create_complete_classifier_system,
        )
        assert CompleteClassifierAssembly is not None
        assert create_complete_classifier_system is not None
    
    def test_imports_from_geometry_package(self):
        """Test imports from geometry package."""
        from airclassifier.geometry import (
            CompleteClassifierAssembly,
            create_complete_classifier_system,
            create_pilot_scale_system,
            create_production_scale_system,
        )
        assert CompleteClassifierAssembly is not None
    
    def test_all_scale_systems_build(self):
        """Test that all scale systems build correctly."""
        systems = [
            create_minimal_classifier_system(),
            create_pilot_scale_system(),
            create_complete_classifier_system(),
            create_production_scale_system(),
        ]
        
        for system in systems:
            verts, idx = system.build_mesh()
            assert len(verts) > 0
            assert len(idx) > 0
    
    def test_production_larger_than_pilot(self):
        """Test that production system is larger than pilot."""
        pilot = create_pilot_scale_system()
        production = create_production_scale_system()
        
        pilot_bounds = pilot.get_bounds()
        prod_bounds = production.get_bounds()
        
        pilot_volume = np.prod(pilot_bounds[1] - pilot_bounds[0])
        prod_volume = np.prod(prod_bounds[1] - prod_bounds[0])
        
        assert prod_volume > pilot_volume
