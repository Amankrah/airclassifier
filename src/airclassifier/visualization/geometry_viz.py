"""
Geometry visualization module for air classifier systems.

Provides GPU-accelerated 3D visualization using NVIDIA Warp and PyVista
for individual components, assembled systems, and complete classifier systems.

Usage:
    from airclassifier.visualization import GeometryVisualizer, VisualizationRequest
    
    # Visualize a single component
    request = VisualizationRequest(
        target_type="component",
        target_name="cyclone_body",
        component=my_cyclone_component
    )
    visualizer = GeometryVisualizer()
    visualizer.render(request)
    
    # Visualize an assembled system
    request = VisualizationRequest(
        target_type="assembly",
        target_name="feed_system",
        assembly=my_feed_assembly
    )
    visualizer.render(request)
    
    # Visualize the complete system
    request = VisualizationRequest(
        target_type="complete_system",
        complete_system=my_complete_system
    )
    visualizer.render(request)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
import numpy as np

# Try to import GPU libraries
try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    wp = None
    WARP_AVAILABLE = False

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False


class VisualizationType(Enum):
    """Types of geometry visualization targets."""
    COMPONENT = "component"
    COMPONENTS = "components"  # Multiple components
    ASSEMBLY = "assembly"
    COMPLETE_SYSTEM = "complete_system"


class RenderBackend(Enum):
    """Available rendering backends."""
    PYVISTA = "pyvista"
    MATPLOTLIB = "matplotlib"
    EXPORT_ONLY = "export_only"


@dataclass
class VisualizationRequest:
    """
    Request payload for geometry visualization.
    
    Attributes:
        target_type: Type of visualization ("component", "components", "assembly", "complete_system")
        target_name: Optional name for the visualization
        
        # For single component
        component: Single geometry component object
        
        # For multiple components
        components: List of component objects
        component_names: Optional names for each component
        component_colors: Optional colors for each component
        
        # For assembly
        assembly: Assembly object (FeedSystemAssembly, AirSystemAssembly, etc.)
        
        # For complete system
        complete_system: CompleteClassifierAssembly object
        
        # Rendering options
        backend: Rendering backend ("pyvista", "matplotlib", "export_only")
        show: Whether to display interactively
        save_path: Path to save rendered image
        export_stl: Whether to export STL file
        export_vtk: Whether to export VTK file
        
        # Visual settings
        opacity: Surface opacity (0-1)
        color: Default color for surfaces
        colormap: Colormap for multi-component rendering
        show_edges: Whether to show mesh edges
        edge_color: Color of edges
        background_color: Background color
        window_size: Window size (width, height)
        camera_position: Camera position preset or coordinates
        lighting: Lighting preset
        
        # Annotations
        show_bounds: Show bounding box
        show_axes: Show coordinate axes
        show_labels: Show component labels
        title: Title for the visualization
    """
    # Target specification
    target_type: str = "component"
    target_name: str = ""
    
    # Target objects (one of these should be provided)
    component: Any = None
    components: List[Any] = field(default_factory=list)
    component_names: List[str] = field(default_factory=list)
    component_colors: List[str] = field(default_factory=list)
    assembly: Any = None
    complete_system: Any = None
    
    # Backend and output
    backend: str = "pyvista"
    show: bool = True
    save_path: Optional[str] = None
    export_stl: bool = False
    export_vtk: bool = False
    stl_path: Optional[str] = None
    vtk_path: Optional[str] = None
    
    # Visual settings
    opacity: float = 0.8
    color: str = "steelblue"
    colormap: str = "coolwarm"
    show_edges: bool = True
    edge_color: str = "darkgray"
    background_color: str = "white"
    window_size: Tuple[int, int] = (1200, 900)
    camera_position: str = "iso"
    lighting: str = "three_lights"
    
    # Annotations
    show_bounds: bool = False
    show_axes: bool = True
    show_labels: bool = True
    title: Optional[str] = None
    
    # Coordinate system
    up_axis: str = "Y"  # "Y" for Y-up (graphics convention), "Z" for Z-up (engineering)


class WarpMeshProcessor:
    """
    GPU-accelerated mesh processing using NVIDIA Warp.
    
    Provides fast mesh operations for visualization:
    - Mesh normals computation
    - Mesh decimation for performance
    - Ray casting for visibility
    - Distance field computation
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Warp mesh processor.
        
        Args:
            device: Warp device ("cuda" or "cpu")
        """
        if not WARP_AVAILABLE:
            raise ImportError("NVIDIA Warp is required for GPU acceleration. "
                            "Install with: pip install warp-lang")
        
        self.device = device
        wp.init()
    
    def create_warp_mesh(self, vertices: np.ndarray, indices: np.ndarray) -> Any:
        """
        Create a Warp mesh from vertices and indices.
        
        Args:
            vertices: Mesh vertices (N, 3)
            indices: Triangle indices (M,) or (M, 3)
            
        Returns:
            wp.Mesh object
        """
        verts = np.ascontiguousarray(vertices, dtype=np.float32)
        idx = np.ascontiguousarray(indices.flatten(), dtype=np.int32)
        
        return wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=self.device),
            indices=wp.array(idx, dtype=wp.int32, device=self.device)
        )
    
    def compute_face_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Compute face normals.
        
        Args:
            vertices: Mesh vertices (N, 3)
            indices: Triangle indices (M*3,)
            
        Returns:
            Face normals (M, 3)
        """
        triangles = indices.reshape(-1, 3)
        num_faces = len(triangles)
        
        normals = np.zeros((num_faces, 3), dtype=np.float32)
        
        for i, tri in enumerate(triangles):
            v0, v1, v2 = vertices[tri]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals[i] = normal / norm
        
        return normals
    
    def decimate_mesh(self, vertices: np.ndarray, indices: np.ndarray, 
                     target_reduction: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce mesh complexity for faster rendering.
        
        Args:
            vertices: Mesh vertices
            indices: Triangle indices
            target_reduction: Target reduction ratio (0-1)
            
        Returns:
            Decimated (vertices, indices)
        """
        triangles = indices.reshape(-1, 3)
        num_tris = len(triangles)
        target_tris = int(num_tris * (1 - target_reduction))
        
        if target_tris >= num_tris:
            return vertices, indices
        
        step = max(1, num_tris // target_tris)
        selected_tris = triangles[::step]
        
        unique_verts = np.unique(selected_tris.flatten())
        vert_map = {old: new for new, old in enumerate(unique_verts)}
        
        new_verts = vertices[unique_verts]
        new_tris = np.array([[vert_map[v] for v in tri] for tri in selected_tris])
        
        return new_verts, new_tris.flatten()


class PyVistaRenderer:
    """
    High-quality 3D renderer using PyVista.
    
    Provides publication-quality rendering with:
    - Multiple mesh support with distinct colors
    - Interactive camera controls
    - Export to various formats
    - Screenshot and animation support
    """
    
    def __init__(self):
        """Initialize PyVista renderer."""
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for 3D visualization. "
                            "Install with: pip install pyvista")
        
        self.plotter = None
        self._mesh_actors = {}
    
    def create_plotter(self, request: VisualizationRequest) -> Any:
        """
        Create a PyVista plotter with configured settings.
        
        Args:
            request: Visualization request with settings
            
        Returns:
            Configured PyVista Plotter
        """
        self.plotter = pv.Plotter(
            window_size=request.window_size,
            off_screen=not request.show
        )
        
        self.plotter.set_background(request.background_color)
        
        # Set coordinate convention based on up_axis setting
        # Y-up: common in graphics/CAD/game engines
        # Z-up: common in engineering/scientific (VTK default)
        if request.up_axis.upper() == "Y":
            self.plotter.camera.up = (0, 1, 0)  # Y-up
        else:
            self.plotter.camera.up = (0, 0, 1)  # Z-up (default VTK)
        
        if request.show_axes:
            self.plotter.add_axes()
        
        if request.title:
            self.plotter.add_title(request.title, font_size=14)
        
        return self.plotter
    
    def add_mesh(self, name: str, vertices: np.ndarray, indices: np.ndarray,
                color: str = "steelblue", opacity: float = 0.8,
                show_edges: bool = True, edge_color: str = "darkgray",
                label: Optional[str] = None):
        """
        Add a mesh to the plotter.
        
        Args:
            name: Unique name for the mesh
            vertices: Mesh vertices (N, 3)
            indices: Triangle indices (M,)
            color: Surface color
            opacity: Surface opacity
            show_edges: Show mesh edges
            edge_color: Edge color
            label: Optional label for legend
        """
        if self.plotter is None:
            raise RuntimeError("Plotter not initialized. Call create_plotter first.")
        
        faces = indices.reshape(-1, 3)
        pv_faces = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).flatten()
        
        mesh = pv.PolyData(vertices, pv_faces)
        
        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            label=label,
            smooth_shading=True
        )
        
        self._mesh_actors[name] = actor
    
    def render(self, request: VisualizationRequest) -> Optional[str]:
        """
        Render the scene.
        
        Args:
            request: Visualization request
            
        Returns:
            Path to saved image if save_path specified
        """
        if isinstance(request.camera_position, str):
            self.plotter.camera_position = request.camera_position
        else:
            self.plotter.camera_position = request.camera_position
        
        if request.show_bounds:
            self.plotter.add_bounding_box()
        
        save_result = None
        if request.save_path:
            self.plotter.screenshot(request.save_path)
            save_result = request.save_path
        
        if request.show:
            self.plotter.show()
        
        return save_result
    
    def export_mesh(self, vertices: np.ndarray, indices: np.ndarray,
                   stl_path: Optional[str] = None,
                   vtk_path: Optional[str] = None):
        """
        Export mesh to file formats.
        
        Args:
            vertices: Mesh vertices
            indices: Triangle indices
            stl_path: Path for STL export
            vtk_path: Path for VTK export
        """
        faces = indices.reshape(-1, 3)
        pv_faces = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).flatten()
        
        mesh = pv.PolyData(vertices, pv_faces)
        
        if stl_path:
            mesh.save(stl_path)
        
        if vtk_path:
            mesh.save(vtk_path)
    
    def close(self):
        """Close the plotter."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None


class MatplotlibRenderer:
    """
    Matplotlib-based 3D renderer (fallback when PyVista unavailable).
    
    Provides basic 3D visualization with:
    - Surface rendering
    - Wireframe option
    - Screenshot export
    """
    
    def __init__(self):
        """Initialize matplotlib renderer."""
        self.fig = None
        self.ax = None
    
    def render_mesh(self, vertices: np.ndarray, indices: np.ndarray,
                   title: str = "Geometry",
                   show_wireframe: bool = True,
                   alpha: float = 0.6,
                   color: str = "steelblue",
                   max_faces: int = 3000,
                   show: bool = True,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Render mesh using matplotlib 3D.
        
        Args:
            vertices: Mesh vertices (N, 3)
            indices: Triangle indices
            title: Plot title
            show_wireframe: Show wireframe edges
            alpha: Surface transparency
            color: Face color
            max_faces: Maximum faces to render
            show: Display interactively
            save_path: Path to save image
            
        Returns:
            Dictionary with figure and axes
        """
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        triangles = indices.reshape(-1, 3)
        verts_mm = vertices * 1000  # Convert to mm
        
        mesh_faces = []
        for tri in triangles:
            face = [verts_mm[tri[0]], verts_mm[tri[1]], verts_mm[tri[2]]]
            mesh_faces.append(face)
        
        if len(mesh_faces) > max_faces:
            step = len(mesh_faces) // max_faces
            mesh_faces = mesh_faces[::step]
        
        mesh = Poly3DCollection(mesh_faces, alpha=alpha)
        mesh.set_facecolor(color)
        if show_wireframe:
            mesh.set_edgecolor('darkblue')
            mesh.set_linewidth(0.1)
        else:
            mesh.set_edgecolor(color)
        
        self.ax.add_collection3d(mesh)
        
        # Set axis limits
        x_min, x_max = verts_mm[:, 0].min(), verts_mm[:, 0].max()
        y_min, y_max = verts_mm[:, 1].min(), verts_mm[:, 1].max()
        z_min, z_max = verts_mm[:, 2].min(), verts_mm[:, 2].max()
        
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.set_xlabel('X (mm)', fontsize=12)
        self.ax.set_ylabel('Y (mm)', fontsize=12)
        self.ax.set_zlabel('Z (mm)', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return {"figure": self.fig, "axes": self.ax}


class GeometryVisualizer:
    """
    Main geometry visualization class.
    
    Handles visualization requests for:
    - Individual components
    - Multiple components
    - Assembled systems
    - Complete classifier systems
    
    Uses NVIDIA Warp for GPU acceleration and PyVista for rendering.
    
    Example:
        visualizer = GeometryVisualizer()
        
        # Visualize a component
        request = VisualizationRequest(
            target_type="component",
            component=my_cyclone,
            show=True
        )
        visualizer.render(request)
        
        # Visualize complete system
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=my_system,
            export_stl=True,
            stl_path="system.stl"
        )
        visualizer.render(request)
    """
    
    # Default colors for multi-component rendering
    DEFAULT_COLORS = [
        "#4A90D9",  # Blue
        "#5CB85C",  # Green
        "#F0AD4E",  # Orange
        "#D9534F",  # Red
        "#9B59B6",  # Purple
        "#1ABC9C",  # Teal
        "#E74C3C",  # Crimson
        "#3498DB",  # Sky blue
        "#2ECC71",  # Emerald
        "#F39C12",  # Amber
    ]
    
    def __init__(self, use_gpu: bool = True, device: str = "cuda"):
        """
        Initialize geometry visualizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration via Warp
            device: Warp device ("cuda" or "cpu")
        """
        self.use_gpu = use_gpu and WARP_AVAILABLE
        self.device = device
        
        self._warp_processor = None
        self._pyvista_renderer = None
        self._matplotlib_renderer = None
    
    @property
    def warp_processor(self) -> Optional[WarpMeshProcessor]:
        """Get or create Warp mesh processor."""
        if self.use_gpu and self._warp_processor is None:
            try:
                self._warp_processor = WarpMeshProcessor(self.device)
            except ImportError:
                self.use_gpu = False
        return self._warp_processor
    
    @property
    def pyvista_renderer(self) -> PyVistaRenderer:
        """Get or create PyVista renderer."""
        if self._pyvista_renderer is None:
            self._pyvista_renderer = PyVistaRenderer()
        return self._pyvista_renderer
    
    @property
    def matplotlib_renderer(self) -> MatplotlibRenderer:
        """Get or create matplotlib renderer."""
        if self._matplotlib_renderer is None:
            self._matplotlib_renderer = MatplotlibRenderer()
        return self._matplotlib_renderer
    
    def render(self, request: VisualizationRequest) -> Dict[str, Any]:
        """
        Render geometry based on the request.
        
        Args:
            request: Visualization request payload
            
        Returns:
            Dictionary with results (files created, plotter, etc.)
        """
        results = {
            "success": True,
            "files": [],
            "message": ""
        }
        
        # Determine backend
        if request.backend == "pyvista" and not PYVISTA_AVAILABLE:
            request.backend = "matplotlib"
            results["message"] = "PyVista not available, falling back to matplotlib"
        
        # Route to appropriate handler
        target_type = request.target_type.lower()
        
        if target_type == "component":
            return self._render_component(request, results)
        elif target_type == "components":
            return self._render_components(request, results)
        elif target_type == "assembly":
            return self._render_assembly(request, results)
        elif target_type == "complete_system":
            return self._render_complete_system(request, results)
        else:
            results["success"] = False
            results["message"] = f"Unknown target type: {target_type}"
            return results
    
    def _render_component(self, request: VisualizationRequest, 
                         results: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single component."""
        component = request.component
        
        if component is None:
            results["success"] = False
            results["message"] = "No component provided"
            return results
        
        vertices = component.vertices
        indices = component.indices
        name = request.target_name or type(component).__name__
        
        if request.backend == "pyvista":
            return self._render_with_pyvista(
                [(name, vertices, indices, request.color)],
                request, results
            )
        else:
            return self._render_with_matplotlib(
                vertices, indices, request, results
            )
    
    def _render_components(self, request: VisualizationRequest,
                          results: Dict[str, Any]) -> Dict[str, Any]:
        """Render multiple components."""
        components = request.components
        
        if not components:
            results["success"] = False
            results["message"] = "No components provided"
            return results
        
        mesh_data = []
        colors = request.component_colors or self.DEFAULT_COLORS
        names = request.component_names or []
        
        for i, comp in enumerate(components):
            name = names[i] if i < len(names) else f"component_{i}"
            color = colors[i % len(colors)]
            mesh_data.append((name, comp.vertices, comp.indices, color))
        
        if request.backend == "pyvista":
            return self._render_with_pyvista(mesh_data, request, results)
        else:
            all_verts, all_idx = self._combine_meshes(mesh_data)
            return self._render_with_matplotlib(all_verts, all_idx, request, results)
    
    def _render_assembly(self, request: VisualizationRequest,
                        results: Dict[str, Any]) -> Dict[str, Any]:
        """Render an assembled system."""
        assembly = request.assembly
        
        if assembly is None:
            results["success"] = False
            results["message"] = "No assembly provided"
            return results
        
        vertices, indices = assembly.build_mesh()
        name = request.target_name or type(assembly).__name__
        
        mesh_data = [(name, vertices, indices, request.color)]
        
        if request.backend == "pyvista":
            return self._render_with_pyvista(mesh_data, request, results)
        else:
            return self._render_with_matplotlib(vertices, indices, request, results)
    
    def _render_complete_system(self, request: VisualizationRequest,
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """Render the complete classifier system."""
        system = request.complete_system
        
        if system is None:
            results["success"] = False
            results["message"] = "No complete system provided"
            return results
        
        vertices, indices = system.build_mesh()
        name = request.target_name or "Complete Air Classifier System"
        
        # Extract subsystems for colored rendering
        mesh_data = []
        
        if hasattr(system, 'get_all_subsystem_names'):
            subsystem_names = system.get_all_subsystem_names()
            colors = self.DEFAULT_COLORS
            color_idx = 0
            
            for sub_name in subsystem_names:
                subsystem = system.get_subsystem(sub_name)
                if subsystem is not None:
                    try:
                        sub_verts, sub_idx = subsystem.build_mesh()
                        color = colors[color_idx % len(colors)]
                        mesh_data.append((sub_name, sub_verts, sub_idx, color))
                        color_idx += 1
                    except Exception:
                        pass
            
            if hasattr(system, 'get_all_component_names'):
                for comp_name in system.get_all_component_names():
                    comp = system.get_component(comp_name)
                    if comp is not None:
                        try:
                            color = colors[color_idx % len(colors)]
                            mesh_data.append((comp_name, comp.vertices, comp.indices, color))
                            color_idx += 1
                        except Exception:
                            pass
            
            if hasattr(system, 'get_all_instrument_names'):
                for inst_name in system.get_all_instrument_names():
                    inst = system.get_instrument(inst_name)
                    if inst is not None:
                        try:
                            mesh_data.append((inst_name, inst.vertices, inst.indices, "#808080"))
                        except Exception:
                            pass
            
            # Add duct connections (silver/gray color for visibility)
            if hasattr(system, 'get_all_duct_names'):
                duct_color = "#A0A0A0"  # Silver/gray for ductwork
                for duct_name in system.get_all_duct_names():
                    try:
                        duct_verts, duct_idx = system.get_duct_mesh(duct_name)
                        mesh_data.append((duct_name, duct_verts, duct_idx, duct_color))
                    except Exception:
                        pass
        
        if not mesh_data:
            mesh_data = [(name, vertices, indices, request.color)]
        
        if request.title is None:
            request.title = name
        
        if request.backend == "pyvista":
            return self._render_with_pyvista(mesh_data, request, results)
        else:
            return self._render_with_matplotlib(vertices, indices, request, results)
    
    def _render_with_pyvista(self, mesh_data: List[Tuple[str, np.ndarray, np.ndarray, str]],
                            request: VisualizationRequest,
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """Render using PyVista backend."""
        renderer = self.pyvista_renderer
        renderer.create_plotter(request)
        
        for name, verts, idx, color in mesh_data:
            label = name if request.show_labels else None
            renderer.add_mesh(
                name, verts, idx,
                color=color,
                opacity=request.opacity,
                show_edges=request.show_edges,
                edge_color=request.edge_color,
                label=label
            )
        
        if request.export_stl or request.export_vtk:
            all_verts, all_idx = self._combine_meshes(mesh_data)
            
            if request.export_stl:
                stl_path = request.stl_path or "geometry.stl"
                renderer.export_mesh(all_verts, all_idx, stl_path=stl_path)
                results["files"].append(stl_path)
            
            if request.export_vtk:
                vtk_path = request.vtk_path or "geometry.vtk"
                renderer.export_mesh(all_verts, all_idx, vtk_path=vtk_path)
                results["files"].append(vtk_path)
        
        if request.show_labels and len(mesh_data) > 1:
            renderer.plotter.add_legend()
        
        save_result = renderer.render(request)
        if save_result:
            results["files"].append(save_result)
        
        results["plotter"] = renderer.plotter
        results["message"] = f"Rendered {len(mesh_data)} mesh(es)"
        
        return results
    
    def _render_with_matplotlib(self, vertices: np.ndarray, indices: np.ndarray,
                               request: VisualizationRequest,
                               results: Dict[str, Any]) -> Dict[str, Any]:
        """Render using matplotlib backend."""
        renderer = self.matplotlib_renderer
        render_result = renderer.render_mesh(
            vertices, indices,
            title=request.title or request.target_name or "Geometry",
            show_wireframe=request.show_edges,
            alpha=request.opacity,
            color=request.color,
            show=request.show,
            save_path=request.save_path
        )
        
        if request.save_path:
            results["files"].append(request.save_path)
        
        results["figure"] = render_result["figure"]
        results["axes"] = render_result["axes"]
        results["message"] = "Rendered with matplotlib"
        
        return results
    
    def _combine_meshes(self, mesh_data: List[Tuple[str, np.ndarray, np.ndarray, str]]
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multiple meshes into one."""
        all_verts = []
        all_idx = []
        vertex_offset = 0
        
        for _, verts, idx, _ in mesh_data:
            all_verts.append(verts)
            all_idx.append(idx + vertex_offset)
            vertex_offset += len(verts)
        
        return (
            np.vstack(all_verts).astype(np.float32),
            np.concatenate(all_idx).astype(np.int32)
        )
    
    def visualize_component(self, component: Any, name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Convenience method to visualize a single component.
        
        Args:
            component: Geometry component with vertices and indices
            name: Optional name
            **kwargs: Additional VisualizationRequest parameters
            
        Returns:
            Render results
        """
        request = VisualizationRequest(
            target_type="component",
            target_name=name or type(component).__name__,
            component=component,
            **kwargs
        )
        return self.render(request)
    
    def visualize_assembly(self, assembly: Any, name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Convenience method to visualize an assembly.
        
        Args:
            assembly: Assembly object
            name: Optional name
            **kwargs: Additional VisualizationRequest parameters
            
        Returns:
            Render results
        """
        request = VisualizationRequest(
            target_type="assembly",
            target_name=name or type(assembly).__name__,
            assembly=assembly,
            **kwargs
        )
        return self.render(request)
    
    def visualize_system(self, system: Any, **kwargs) -> Dict[str, Any]:
        """
        Convenience method to visualize a complete system.
        
        Args:
            system: CompleteClassifierAssembly object
            **kwargs: Additional VisualizationRequest parameters
            
        Returns:
            Render results
        """
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=system,
            title="Complete Air Classifier System",
            **kwargs
        )
        return self.render(request)
    
    def export_to_stl(self, target: Any, path: str) -> str:
        """
        Export geometry to STL file.
        
        Args:
            target: Component, assembly, or complete system
            path: Output file path
            
        Returns:
            Path to created file
        """
        if hasattr(target, 'build_mesh'):
            vertices, indices = target.build_mesh()
        elif hasattr(target, 'vertices') and hasattr(target, 'indices'):
            vertices = target.vertices
            indices = target.indices
        else:
            raise ValueError("Target must have vertices/indices or build_mesh()")
        
        from ..geometry.mesh_generator import export_mesh_stl
        triangles = indices.reshape(-1, 3)
        export_mesh_stl(vertices, triangles, path, binary=True)
        
        return path
    
    def export_to_vtk(self, target: Any, path: str) -> str:
        """
        Export geometry to VTK file.
        
        Args:
            target: Component, assembly, or complete system
            path: Output file path
            
        Returns:
            Path to created file
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista required for VTK export")
        
        if hasattr(target, 'build_mesh'):
            vertices, indices = target.build_mesh()
        elif hasattr(target, 'vertices') and hasattr(target, 'indices'):
            vertices = target.vertices
            indices = target.indices
        else:
            raise ValueError("Target must have vertices/indices or build_mesh()")
        
        renderer = self.pyvista_renderer
        renderer.export_mesh(vertices, indices, vtk_path=path)
        
        return path


# =============================================================================
# Convenience functions
# =============================================================================

def visualize_geometry(target: Any, **kwargs) -> Dict[str, Any]:
    """
    Quick visualization of any geometry target.
    
    Automatically detects target type (component, assembly, or complete system)
    and renders appropriately.
    
    Args:
        target: Component, assembly, or complete system
        **kwargs: Additional VisualizationRequest parameters
        
    Returns:
        Render results dictionary
    """
    visualizer = GeometryVisualizer()
    
    target_type = "component"
    
    if hasattr(target, 'get_all_subsystem_names'):
        target_type = "complete_system"
    elif hasattr(target, 'build_mesh') and hasattr(target, 'get_component'):
        target_type = "assembly"
    
    if target_type == "complete_system":
        return visualizer.visualize_system(target, **kwargs)
    elif target_type == "assembly":
        return visualizer.visualize_assembly(target, **kwargs)
    else:
        return visualizer.visualize_component(target, **kwargs)


def quick_render(target: Any, show: bool = True, save_path: str = None) -> Dict[str, Any]:
    """
    Quick render with minimal configuration.
    
    Args:
        target: Component, assembly, or complete system
        show: Whether to display interactively
        save_path: Optional path to save image
        
    Returns:
        Render results dictionary
    """
    return visualize_geometry(target, show=show, save_path=save_path)
