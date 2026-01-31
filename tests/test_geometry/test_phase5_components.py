"""
Tests for Phase 5 Safety & Instrumentation components.

Tests cover:
- Explosion Vents (rupture panels, hinged doors, recoil vents)
- Grounding Points
- Pressure Ports
- Temperature Ports (thermowells)
- Sample Ports
- Sight Glasses
- SafetyInstrumentationAssembly
"""

import pytest
import numpy as np

from airclassifier.geometry.components.safety import (
    ExplosionVent,
    ExplosionVentParams,
    create_rupture_panel,
    create_hinged_explosion_door,
    create_recoil_vent,
    calculate_vent_area,
    GroundingPoint,
    GroundingPointParams,
    create_weld_stud_ground,
    create_threaded_ground,
    create_grounding_system,
)
from airclassifier.geometry.components.instrumentation import (
    PressurePort,
    PressurePortParams,
    create_flush_pressure_port,
    create_extended_pressure_port,
    create_averaging_pressure_port,
    TemperaturePort,
    TemperaturePortParams,
    create_threaded_thermowell,
    create_flanged_thermowell,
    create_weld_thermowell,
    SamplePort,
    SamplePortParams,
    create_ball_valve_sample_port,
    create_isokinetic_sample_port,
    SightGlass,
    SightGlassParams,
    create_standard_sight_glass,
    create_illuminated_sight_glass,
)
from airclassifier.geometry.assembly import (
    SafetyInstrumentationAssembly,
    SafetyInstrumentationParams,
    create_standard_safety_instrumentation,
    create_minimal_instrumentation,
    create_full_instrumentation,
)


# ============================================================================
# Explosion Vent Tests
# ============================================================================

class TestExplosionVentParams:
    """Tests for ExplosionVentParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = ExplosionVentParams(
            vent_area=0.05,
            vent_type="rupture_panel",
        )
        assert params.vent_area == 0.05
        assert params.vent_type == "rupture_panel"
    
    def test_diameter_calculation(self):
        """Test diameter calculation from area."""
        params = ExplosionVentParams(vent_area=0.1, shape="circular")
        expected_d = 2 * np.sqrt(0.1 / np.pi)
        assert abs(params.diameter - expected_d) < 1e-6
    
    def test_actual_area(self):
        """Test actual area calculation."""
        params = ExplosionVentParams(vent_area=0.05, shape="circular")
        assert abs(params.actual_area - params.vent_area) < 0.01
    
    def test_reduced_pressure_estimate(self):
        """Test reduced pressure estimation."""
        params = ExplosionVentParams(vent_area=0.1)
        Pred = params.get_reduced_pressure(Kst=150, volume=1.0)
        assert Pred > params.static_burst_pressure
        assert Pred <= 2.0  # Should be capped at 2.0


class TestExplosionVent:
    """Tests for ExplosionVent class."""
    
    def test_rupture_panel_creation(self):
        """Test rupture panel creation."""
        vent = create_rupture_panel(vent_area=0.05)
        assert vent.params.vent_type == "rupture_panel"
    
    def test_hinged_door_creation(self):
        """Test hinged door creation."""
        vent = create_hinged_explosion_door(vent_area=0.05)
        assert vent.params.vent_type == "hinged_door"
    
    def test_recoil_vent_creation(self):
        """Test recoil vent creation."""
        vent = create_recoil_vent(vent_area=0.05)
        assert vent.params.vent_type == "recoil"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        vent = create_rupture_panel(vent_area=0.05)
        verts, inds, norms = vent.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
        assert len(norms) == len(verts)
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        vent = create_rupture_panel(vent_area=0.05)
        verts = vent.vertices
        inds = vent.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_vent_area_calculation(self):
        """Test vent area calculation function."""
        area = calculate_vent_area(volume=1.0, Kst=150)
        assert area > 0
        
        # Larger volume needs more area
        area_large = calculate_vent_area(volume=5.0, Kst=150)
        assert area_large > area


# ============================================================================
# Grounding Point Tests
# ============================================================================

class TestGroundingPointParams:
    """Tests for GroundingPointParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = GroundingPointParams(
            location=(1.0, 0.0, 0.5),
            stud_diameter=0.010,
        )
        assert params.location == (1.0, 0.0, 0.5)
        assert params.stud_diameter == 0.010
    
    def test_stud_radius(self):
        """Test stud radius property."""
        params = GroundingPointParams(stud_diameter=0.010)
        assert params.stud_radius == 0.005


class TestGroundingPoint:
    """Tests for GroundingPoint class."""
    
    def test_weld_stud_creation(self):
        """Test weld stud grounding point creation."""
        ground = create_weld_stud_ground(
            location=(0.5, 0.0, 0.3),
            stud_size="M10"
        )
        assert ground.params.stud_type == "weld_stud"
        assert ground.params.stud_diameter == 0.010
    
    def test_threaded_ground_creation(self):
        """Test threaded grounding point creation."""
        ground = create_threaded_ground(location=(0.5, 0.0, 0.3))
        assert ground.params.stud_type == "threaded"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        ground = create_weld_stud_ground(location=(0, 0, 0))
        verts, inds, norms = ground.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        ground = create_weld_stud_ground(location=(0, 0, 0))
        verts = ground.vertices
        inds = ground.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)
    
    def test_grounding_system(self):
        """Test creating multiple grounding points."""
        locations = [(0, 0.5, 0.3), (0.5, 0, 0.3), (0, -0.5, 0.3), (-0.5, 0, 0.3)]
        grounds = create_grounding_system(locations)
        
        assert len(grounds) == 4
        for ground in grounds:
            assert ground.vertices is not None


# ============================================================================
# Pressure Port Tests
# ============================================================================

class TestPressurePortParams:
    """Tests for PressurePortParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = PressurePortParams(
            port_type="flush_mount",
            connection_size="1/2 NPT",
        )
        assert params.port_type == "flush_mount"
    
    def test_port_radius(self):
        """Test port radius property."""
        params = PressurePortParams(port_diameter=0.013)
        assert params.port_radius == 0.0065


class TestPressurePort:
    """Tests for PressurePort class."""
    
    def test_flush_mount_creation(self):
        """Test flush mount pressure port creation."""
        port = create_flush_pressure_port(location=(0.5, 0, 0.3))
        assert port.params.port_type == "flush_mount"
    
    def test_extended_creation(self):
        """Test extended pressure port creation."""
        port = create_extended_pressure_port(
            location=(0.5, 0, 0.3),
            extension_length=0.075
        )
        assert port.params.port_type == "extended"
        assert port.params.extension_length == 0.075
    
    def test_averaging_creation(self):
        """Test averaging pressure port creation."""
        port = create_averaging_pressure_port(location=(0.5, 0, 0.3))
        assert port.params.port_type == "averaging"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        port = create_flush_pressure_port(location=(0, 0, 0))
        verts, inds, norms = port.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        port = create_flush_pressure_port(location=(0, 0, 0))
        verts = port.vertices
        inds = port.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


# ============================================================================
# Temperature Port Tests
# ============================================================================

class TestTemperaturePortParams:
    """Tests for TemperaturePortParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = TemperaturePortParams(
            thermowell_diameter=0.016,
            immersion_length=0.100,
        )
        assert params.thermowell_diameter == 0.016
        assert params.immersion_length == 0.100
    
    def test_thermowell_radius(self):
        """Test thermowell radius property."""
        params = TemperaturePortParams(thermowell_diameter=0.016)
        assert params.thermowell_radius == 0.008


class TestTemperaturePort:
    """Tests for TemperaturePort class."""
    
    def test_threaded_thermowell_creation(self):
        """Test threaded thermowell creation."""
        port = create_threaded_thermowell(
            location=(0.5, 0, 0.3),
            immersion_length=0.1
        )
        assert port.params.connection_type == "threaded"
    
    def test_flanged_thermowell_creation(self):
        """Test flanged thermowell creation."""
        port = create_flanged_thermowell(
            location=(0.5, 0, 0.3),
            immersion_length=0.15
        )
        assert port.params.connection_type == "flanged"
    
    def test_weld_thermowell_creation(self):
        """Test weld thermowell creation."""
        port = create_weld_thermowell(location=(0.5, 0, 0.3))
        assert port.params.connection_type == "weld"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        port = create_threaded_thermowell(location=(0, 0, 0))
        verts, inds, norms = port.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        port = create_threaded_thermowell(location=(0, 0, 0))
        verts = port.vertices
        inds = port.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


# ============================================================================
# Sample Port Tests
# ============================================================================

class TestSamplePortParams:
    """Tests for SamplePortParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = SamplePortParams(
            port_diameter=0.025,
            valve_type="ball",
        )
        assert params.port_diameter == 0.025
        assert params.valve_type == "ball"
    
    def test_port_radius(self):
        """Test port radius property."""
        params = SamplePortParams(port_diameter=0.025)
        assert params.port_radius == 0.0125


class TestSamplePort:
    """Tests for SamplePort class."""
    
    def test_ball_valve_creation(self):
        """Test ball valve sample port creation."""
        port = create_ball_valve_sample_port(location=(0.5, 0, 0.3))
        assert port.params.valve_type == "ball"
    
    def test_isokinetic_creation(self):
        """Test isokinetic sample port creation."""
        port = create_isokinetic_sample_port(location=(0.5, 0, 0.3))
        assert port.params.sample_type == "isokinetic"
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        port = create_ball_valve_sample_port(location=(0, 0, 0))
        verts, inds, norms = port.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        port = create_ball_valve_sample_port(location=(0, 0, 0))
        verts = port.vertices
        inds = port.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


# ============================================================================
# Sight Glass Tests
# ============================================================================

class TestSightGlassParams:
    """Tests for SightGlassParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = SightGlassParams(
            glass_diameter=0.100,
            glass_type="borosilicate",
        )
        assert params.glass_diameter == 0.100
        assert params.glass_type == "borosilicate"
    
    def test_glass_radius(self):
        """Test glass radius property."""
        params = SightGlassParams(glass_diameter=0.100)
        assert params.glass_radius == 0.050
    
    def test_flange_diameter(self):
        """Test flange diameter from size."""
        params = SightGlassParams(flange_size="DN100")
        assert params.flange_diameter == 0.190


class TestSightGlass:
    """Tests for SightGlass class."""
    
    def test_standard_creation(self):
        """Test standard sight glass creation."""
        glass = create_standard_sight_glass(
            location=(0.5, 0, 0.5),
            diameter=0.1
        )
        assert glass.params.glass_diameter == 0.1
        assert not glass.params.light_port
    
    def test_illuminated_creation(self):
        """Test illuminated sight glass creation."""
        glass = create_illuminated_sight_glass(
            location=(0.5, 0, 0.5),
            diameter=0.1
        )
        assert glass.params.light_port
    
    def test_mesh_generation(self):
        """Test mesh generation."""
        glass = create_standard_sight_glass(location=(0, 0, 0))
        verts, inds, norms = glass.generate_mesh()
        
        assert len(verts) > 0
        assert len(inds) > 0
    
    def test_mesh_valid_indices(self):
        """Test that mesh indices are valid."""
        glass = create_standard_sight_glass(location=(0, 0, 0))
        verts = glass.vertices
        inds = glass.indices
        
        assert inds.min() >= 0
        assert inds.max() < len(verts)


# ============================================================================
# SafetyInstrumentationAssembly Tests
# ============================================================================

class TestSafetyInstrumentationParams:
    """Tests for SafetyInstrumentationParams dataclass."""
    
    def test_params_creation(self):
        """Test basic parameter creation."""
        params = SafetyInstrumentationParams(
            vessel_volume=2.0,
            Kst=150,
        )
        assert params.vessel_volume == 2.0
        assert params.Kst == 150
    
    def test_required_vent_area(self):
        """Test required vent area calculation."""
        params = SafetyInstrumentationParams(vessel_volume=1.0)
        area = params.required_vent_area
        assert area > 0
        
        # Larger volume needs more area
        params_large = SafetyInstrumentationParams(vessel_volume=5.0)
        assert params_large.required_vent_area > area


class TestSafetyInstrumentationAssembly:
    """Tests for SafetyInstrumentationAssembly class."""
    
    @pytest.fixture
    def assembly(self):
        """Create a standard safety/instrumentation assembly."""
        return create_standard_safety_instrumentation(vessel_volume=1.0)
    
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
    
    def test_get_component(self, assembly):
        """Test getting components by name."""
        vent = assembly.get_component('explosion_vent_0')
        assert vent is not None
    
    def test_get_component_names(self, assembly):
        """Test getting component names."""
        names = assembly.get_component_names()
        assert len(names) > 0
        assert 'explosion_vent_0' in names
    
    def test_get_components_by_type(self, assembly):
        """Test getting components by type."""
        grounds = assembly.get_components_by_type('grounding_point')
        assert len(grounds) > 0
    
    def test_get_system_summary(self, assembly):
        """Test getting system summary."""
        summary = assembly.get_system_summary()
        assert 'vessel_volume_m3' in summary
        assert 'required_vent_area_m2' in summary
        assert 'total_components' in summary
    
    def test_minimal_instrumentation(self):
        """Test minimal instrumentation package."""
        assembly = create_minimal_instrumentation(vessel_volume=0.5)
        assert assembly.params.num_explosion_vents == 1
        assert assembly.params.num_grounding_points == 2
    
    def test_full_instrumentation(self):
        """Test full instrumentation package."""
        assembly = create_full_instrumentation(vessel_volume=5.0)
        assert assembly.params.num_explosion_vents >= 1
        assert assembly.params.num_pressure_ports == 4
        assert assembly.params.num_temp_ports == 4


# ============================================================================
# Mesh Quality Tests
# ============================================================================

class TestPhase5MeshQuality:
    """Tests for mesh quality across all Phase 5 components."""
    
    @pytest.fixture
    def all_components(self):
        """Create all Phase 5 components."""
        return [
            create_rupture_panel(vent_area=0.05),
            create_hinged_explosion_door(vent_area=0.05),
            create_recoil_vent(vent_area=0.05),
            create_weld_stud_ground(location=(0, 0, 0)),
            create_threaded_ground(location=(0, 0, 0)),
            create_flush_pressure_port(location=(0, 0, 0)),
            create_extended_pressure_port(location=(0, 0, 0)),
            create_threaded_thermowell(location=(0, 0, 0)),
            create_flanged_thermowell(location=(0, 0, 0)),
            create_ball_valve_sample_port(location=(0, 0, 0)),
            create_isokinetic_sample_port(location=(0, 0, 0)),
            create_standard_sight_glass(location=(0, 0, 0)),
            create_illuminated_sight_glass(location=(0, 0, 0)),
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
        """Test that no triangles are degenerate."""
        for component in all_components:
            verts = component.vertices
            inds = component.indices.reshape(-1, 3)
            
            for tri in inds[:50]:  # Check first 50 triangles
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
            assert len(verts) >= 10, f"Too few vertices in {type(component).__name__}"
            assert len(verts) < 50000, f"Too many vertices in {type(component).__name__}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 components."""
    
    def test_all_components_create_warp_mesh(self):
        """Test that all components can create Warp meshes."""
        pytest.importorskip("warp")
        
        components = [
            create_rupture_panel(vent_area=0.05),
            create_weld_stud_ground(location=(0, 0, 0)),
            create_flush_pressure_port(location=(0, 0, 0)),
            create_threaded_thermowell(location=(0, 0, 0)),
            create_ball_valve_sample_port(location=(0, 0, 0)),
            create_standard_sight_glass(location=(0, 0, 0)),
        ]
        
        for component in components:
            mesh = component.to_warp_mesh()
            assert mesh is not None
    
    def test_assembly_creates_warp_mesh(self):
        """Test that assembly can create Warp mesh."""
        pytest.importorskip("warp")
        
        assembly = create_standard_safety_instrumentation(vessel_volume=1.0)
        mesh = assembly.to_warp_mesh()
        assert mesh is not None
    
    def test_vent_sizing_scales_with_volume(self):
        """Test that vent area scales with vessel volume."""
        small = SafetyInstrumentationParams(vessel_volume=0.5)
        large = SafetyInstrumentationParams(vessel_volume=5.0)
        
        assert large.required_vent_area > small.required_vent_area
    
    def test_imports_from_components_package(self):
        """Test that Phase 5 components can be imported from components."""
        from airclassifier.geometry.components import (
            ExplosionVent,
            GroundingPoint,
            PressurePort,
            TemperaturePort,
            SamplePort,
            SightGlass,
        )
        assert ExplosionVent is not None
        assert GroundingPoint is not None
        assert PressurePort is not None
        assert TemperaturePort is not None
        assert SamplePort is not None
        assert SightGlass is not None
    
    def test_imports_from_assembly_package(self):
        """Test that assembly can be imported from assembly package."""
        from airclassifier.geometry.assembly import (
            SafetyInstrumentationAssembly,
            create_standard_safety_instrumentation,
        )
        assert SafetyInstrumentationAssembly is not None
        assert create_standard_safety_instrumentation is not None
