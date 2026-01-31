"""
Tests for geometry visualization module.

Tests cover:
- VisualizationRequest creation and validation
- GeometryVisualizer for components, assemblies, and complete systems
- Matplotlib fallback rendering
- Export functionality (STL, VTK)
"""

import pytest
import numpy as np
import tempfile
import os

from airclassifier.visualization.geometry_viz import (
    VisualizationRequest,
    VisualizationType,
    RenderBackend,
    GeometryVisualizer,
    MatplotlibRenderer,
    visualize_geometry,
    quick_render,
    PYVISTA_AVAILABLE,
    WARP_AVAILABLE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class MockComponent:
    """Mock geometry component for testing."""
    
    def __init__(self, name="test_component"):
        self.name = name
        # Simple cube mesh
        self.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)
        self.indices = np.array([
            0, 1, 2, 0, 2, 3,  # bottom
            4, 5, 6, 4, 6, 7,  # top
            0, 1, 5, 0, 5, 4,  # front
            2, 3, 7, 2, 7, 6,  # back
            0, 3, 7, 0, 7, 4,  # left
            1, 2, 6, 1, 6, 5,  # right
        ], dtype=np.int32)


class MockAssembly:
    """Mock assembly for testing."""
    
    def __init__(self, name="test_assembly"):
        self.name = name
        self._components = {
            "part_a": MockComponent("part_a"),
            "part_b": MockComponent("part_b"),
        }
    
    def build_mesh(self):
        """Build combined mesh."""
        all_verts = []
        all_idx = []
        offset = 0
        
        for comp in self._components.values():
            all_verts.append(comp.vertices + np.array([offset, 0, 0]))
            all_idx.append(comp.indices + offset * 8 // 2)
            offset += 2
        
        return (
            np.vstack(all_verts).astype(np.float32),
            np.concatenate(all_idx).astype(np.int32)
        )
    
    def get_component(self, name):
        return self._components.get(name)
    
    def get_all_component_names(self):
        return list(self._components.keys())


class MockCompleteSystem:
    """Mock complete system for testing."""
    
    def __init__(self):
        self._subsystems = {
            "feed_system": MockAssembly("feed_system"),
            "classification": MockAssembly("classification"),
        }
        self._components = {
            "silencer": MockComponent("silencer"),
        }
        self._instrumentation = {
            "pressure_inlet": MockComponent("pressure_inlet"),
        }
    
    def build_mesh(self):
        """Build complete mesh."""
        all_verts = []
        all_idx = []
        offset = 0
        
        for sub in self._subsystems.values():
            verts, idx = sub.build_mesh()
            all_verts.append(verts + np.array([0, offset, 0]))
            all_idx.append(idx + len(all_verts[-1]) if len(all_verts) > 1 else idx)
            offset += 3
        
        for comp in self._components.values():
            all_verts.append(comp.vertices + np.array([0, offset, 0]))
            all_idx.append(comp.indices + sum(len(v) for v in all_verts[:-1]))
            offset += 2
        
        return (
            np.vstack(all_verts).astype(np.float32),
            np.concatenate(all_idx).astype(np.int32)
        )
    
    def get_subsystem(self, name):
        return self._subsystems.get(name)
    
    def get_all_subsystem_names(self):
        return list(self._subsystems.keys())
    
    def get_component(self, name):
        return self._components.get(name)
    
    def get_all_component_names(self):
        return list(self._components.keys())
    
    def get_instrument(self, name):
        return self._instrumentation.get(name)
    
    def get_all_instrument_names(self):
        return list(self._instrumentation.keys())


@pytest.fixture
def mock_component():
    return MockComponent()


@pytest.fixture
def mock_assembly():
    return MockAssembly()


@pytest.fixture
def mock_complete_system():
    return MockCompleteSystem()


@pytest.fixture
def visualizer():
    return GeometryVisualizer(use_gpu=False)


# =============================================================================
# VisualizationRequest Tests
# =============================================================================

class TestVisualizationRequest:
    """Tests for VisualizationRequest dataclass."""
    
    def test_default_values(self):
        """Test default parameter values."""
        request = VisualizationRequest()
        
        assert request.target_type == "component"
        assert request.backend == "pyvista"
        assert request.show == True
        assert request.opacity == 0.8
        assert request.show_edges == True
        assert request.window_size == (1200, 900)
    
    def test_custom_values(self):
        """Test custom parameter values."""
        request = VisualizationRequest(
            target_type="assembly",
            backend="matplotlib",
            opacity=0.5,
            color="red",
            show=False
        )
        
        assert request.target_type == "assembly"
        assert request.backend == "matplotlib"
        assert request.opacity == 0.5
        assert request.color == "red"
        assert request.show == False
    
    def test_component_request(self, mock_component):
        """Test request with component."""
        request = VisualizationRequest(
            target_type="component",
            component=mock_component,
            target_name="test"
        )
        
        assert request.component is not None
        assert request.target_name == "test"
    
    def test_assembly_request(self, mock_assembly):
        """Test request with assembly."""
        request = VisualizationRequest(
            target_type="assembly",
            assembly=mock_assembly
        )
        
        assert request.assembly is not None


# =============================================================================
# GeometryVisualizer Tests
# =============================================================================

class TestGeometryVisualizer:
    """Tests for GeometryVisualizer class."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        viz = GeometryVisualizer(use_gpu=False)
        assert viz.use_gpu == False
    
    def test_render_component_matplotlib(self, visualizer, mock_component):
        """Test rendering component with matplotlib."""
        request = VisualizationRequest(
            target_type="component",
            component=mock_component,
            backend="matplotlib",
            show=False
        )
        
        results = visualizer.render(request)
        
        assert results["success"] == True
        assert "figure" in results
    
    def test_render_component_no_component(self, visualizer):
        """Test rendering with no component."""
        request = VisualizationRequest(
            target_type="component",
            component=None,
            backend="matplotlib",
            show=False
        )
        
        results = visualizer.render(request)
        
        assert results["success"] == False
        assert "No component" in results["message"]
    
    def test_render_assembly_matplotlib(self, visualizer, mock_assembly):
        """Test rendering assembly with matplotlib."""
        request = VisualizationRequest(
            target_type="assembly",
            assembly=mock_assembly,
            backend="matplotlib",
            show=False
        )
        
        results = visualizer.render(request)
        
        assert results["success"] == True
    
    def test_render_complete_system_matplotlib(self, visualizer, mock_complete_system):
        """Test rendering complete system with matplotlib."""
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=mock_complete_system,
            backend="matplotlib",
            show=False
        )
        
        results = visualizer.render(request)
        
        assert results["success"] == True
    
    def test_unknown_target_type(self, visualizer):
        """Test handling unknown target type."""
        request = VisualizationRequest(
            target_type="unknown",
            backend="matplotlib",
            show=False
        )
        
        results = visualizer.render(request)
        
        assert results["success"] == False
        assert "Unknown target type" in results["message"]
    
    def test_visualize_component_convenience(self, visualizer, mock_component):
        """Test convenience method for component visualization."""
        results = visualizer.visualize_component(
            mock_component,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_visualize_assembly_convenience(self, visualizer, mock_assembly):
        """Test convenience method for assembly visualization."""
        results = visualizer.visualize_assembly(
            mock_assembly,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_visualize_system_convenience(self, visualizer, mock_complete_system):
        """Test convenience method for system visualization."""
        results = visualizer.visualize_system(
            mock_complete_system,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Tests for export functionality."""
    
    def test_export_to_stl(self, visualizer, mock_component):
        """Test STL export."""
        # Use workspace directory for sandbox compatibility
        path = "test_export.stl"
        try:
            result = visualizer.export_to_stl(mock_component, path)
            assert result == path
            assert os.path.exists(path)
        except PermissionError:
            pytest.skip("Permission denied in sandbox environment")
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_export_assembly_to_stl(self, visualizer, mock_assembly):
        """Test STL export of assembly."""
        path = "test_assembly_export.stl"
        try:
            result = visualizer.export_to_stl(mock_assembly, path)
            assert result == path
            assert os.path.exists(path)
        except PermissionError:
            pytest.skip("Permission denied in sandbox environment")
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    @pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista required")
    def test_export_to_vtk(self, visualizer, mock_component):
        """Test VTK export."""
        path = "test_export.vtk"
        try:
            result = visualizer.export_to_vtk(mock_component, path)
            assert result == path
            assert os.path.exists(path)
        except PermissionError:
            pytest.skip("Permission denied in sandbox environment")
        finally:
            if os.path.exists(path):
                os.remove(path)


# =============================================================================
# Legacy Function Tests
# =============================================================================

class TestMatplotlibRenderer:
    """Tests for matplotlib renderer."""
    
    def test_matplotlib_renderer_creation(self):
        """Test matplotlib renderer creation."""
        renderer = MatplotlibRenderer()
        assert renderer is not None
    
    def test_matplotlib_render_mesh(self, mock_component):
        """Test matplotlib mesh rendering."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            renderer = MatplotlibRenderer()
            result = renderer.render_mesh(
                mock_component.vertices,
                mock_component.indices,
                title="Test Mesh",
                show=False
            )
            
            assert result["figure"] is not None
            assert result["axes"] is not None
            
            import matplotlib.pyplot as plt
            plt.close(result["figure"])
        except Exception as e:
            if "tk" in str(e).lower() or "display" in str(e).lower():
                pytest.skip(f"Display not available: {e}")
            raise


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_visualize_geometry_component(self, mock_component):
        """Test auto-detection of component."""
        results = visualize_geometry(
            mock_component,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_visualize_geometry_assembly(self, mock_assembly):
        """Test auto-detection of assembly."""
        results = visualize_geometry(
            mock_assembly,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_visualize_geometry_system(self, mock_complete_system):
        """Test auto-detection of complete system."""
        results = visualize_geometry(
            mock_complete_system,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_quick_render(self, mock_component):
        """Test quick render function."""
        results = quick_render(
            mock_component,
            show=False
        )
        
        assert results["success"] == True


# =============================================================================
# PyVista Integration Tests (if available)
# =============================================================================

@pytest.mark.skipif(not PYVISTA_AVAILABLE, reason="PyVista required")
class TestPyVistaIntegration:
    """Tests for PyVista backend."""
    
    def test_pyvista_renderer_creation(self, mock_component):
        """Test PyVista renderer creation."""
        from airclassifier.visualization.geometry_viz import PyVistaRenderer
        
        renderer = PyVistaRenderer()
        request = VisualizationRequest(show=False)
        plotter = renderer.create_plotter(request)
        
        assert plotter is not None
        renderer.close()
    
    def test_pyvista_add_mesh(self, mock_component):
        """Test adding mesh to PyVista renderer."""
        from airclassifier.visualization.geometry_viz import PyVistaRenderer
        
        renderer = PyVistaRenderer()
        request = VisualizationRequest(show=False)
        renderer.create_plotter(request)
        
        renderer.add_mesh(
            "test",
            mock_component.vertices,
            mock_component.indices,
            color="blue"
        )
        
        assert "test" in renderer._mesh_actors
        renderer.close()
    
    def test_render_component_pyvista(self, mock_component):
        """Test rendering component with PyVista."""
        viz = GeometryVisualizer(use_gpu=False)
        
        request = VisualizationRequest(
            target_type="component",
            component=mock_component,
            backend="pyvista",
            show=False
        )
        
        results = viz.render(request)
        
        assert results["success"] == True
        assert "plotter" in results


# =============================================================================
# Warp Integration Tests (if available)
# =============================================================================

@pytest.mark.skipif(not WARP_AVAILABLE, reason="NVIDIA Warp required")
class TestWarpIntegration:
    """Tests for Warp GPU acceleration."""
    
    def test_warp_processor_creation(self):
        """Test Warp processor creation."""
        from airclassifier.visualization.geometry_viz import WarpMeshProcessor
        
        processor = WarpMeshProcessor(device="cpu")
        assert processor is not None
    
    def test_warp_mesh_creation(self, mock_component):
        """Test creating Warp mesh."""
        from airclassifier.visualization.geometry_viz import WarpMeshProcessor
        
        processor = WarpMeshProcessor(device="cpu")
        mesh = processor.create_warp_mesh(
            mock_component.vertices,
            mock_component.indices
        )
        
        assert mesh is not None
    
    def test_warp_face_normals(self, mock_component):
        """Test computing face normals."""
        from airclassifier.visualization.geometry_viz import WarpMeshProcessor
        
        processor = WarpMeshProcessor(device="cpu")
        normals = processor.compute_face_normals(
            mock_component.vertices,
            mock_component.indices
        )
        
        num_faces = len(mock_component.indices) // 3
        assert normals.shape == (num_faces, 3)
    
    def test_warp_decimate(self, mock_component):
        """Test mesh decimation."""
        from airclassifier.visualization.geometry_viz import WarpMeshProcessor
        
        processor = WarpMeshProcessor(device="cpu")
        new_verts, new_idx = processor.decimate_mesh(
            mock_component.vertices,
            mock_component.indices,
            target_reduction=0.5
        )
        
        assert len(new_idx) <= len(mock_component.indices)


# =============================================================================
# Integration with Real Components
# =============================================================================

class TestRealComponentIntegration:
    """Tests with real geometry components."""
    
    def test_with_real_cyclone(self):
        """Test visualization with real cyclone assembly."""
        from airclassifier.geometry.assembly import CycloneAssembly
        from airclassifier.geometry.assembly.cyclone import CycloneGeometryParams
        
        # Provide all required parameters for cyclone geometry
        params = CycloneGeometryParams(
            cylinder_diameter=0.3,
            cylinder_height=0.3,
            cone_height=0.4,
            cone_tip_diameter=0.05,
            inlet_width=0.1,
            inlet_height=0.15,
            vortex_finder_diameter=0.1,
            vortex_finder_length=0.15
        )
        cyclone = CycloneAssembly(params)
        viz = GeometryVisualizer(use_gpu=False)
        
        results = viz.visualize_assembly(
            cyclone,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_with_real_feed_system(self):
        """Test visualization with real feed system assembly."""
        from airclassifier.geometry.assembly import create_standard_feed_system
        
        feed_system = create_standard_feed_system()
        viz = GeometryVisualizer(use_gpu=False)
        
        results = viz.visualize_assembly(
            feed_system,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
    
    def test_with_real_complete_system(self):
        """Test visualization with real complete system."""
        from airclassifier.geometry.assembly import create_complete_classifier_system
        
        system = create_complete_classifier_system()
        viz = GeometryVisualizer(use_gpu=False)
        
        results = viz.visualize_system(
            system,
            backend="matplotlib",
            show=False
        )
        
        assert results["success"] == True
