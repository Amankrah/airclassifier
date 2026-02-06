"""
3D Viewport Widget
==================

PyVista-based 3D visualization widget for displaying air classifier
geometry, particles, and flow fields.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QPushButton,
    QComboBox, QLabel, QSlider, QCheckBox, QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

# Try to import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not available. 3D viewport will be limited.")


class Viewport3D(QWidget):
    """
    3D viewport widget for visualizing air classifier geometry and simulation.

    Features:
    - Interactive 3D view with rotation/zoom/pan
    - Component geometry visualization
    - Particle rendering during simulation
    - Velocity field visualization
    - Cross-section views
    """

    # Signals
    view_changed = Signal()
    component_clicked = Signal(str)  # component_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._meshes: Dict[str, Any] = {}          # PyVista meshes by component ID
        self._actors: Dict[str, Any] = {}          # VTK actors by component ID
        self._particles_actor = None
        self._flow_actors: List[Any] = []
        self._show_particles = True
        self._show_flow = False
        self._show_wireframe = False
        self._component_colors: Dict[str, str] = {}

        self._setup_ui()
        self._setup_viewer()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)

        # View controls
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom"
        ])
        self.view_combo.currentTextChanged.connect(self._set_view)
        toolbar.addWidget(QLabel(" View: "))
        toolbar.addWidget(self.view_combo)
        toolbar.addSeparator()

        # Display options
        self.wireframe_btn = QPushButton("Wireframe")
        self.wireframe_btn.setCheckable(True)
        self.wireframe_btn.toggled.connect(self._toggle_wireframe)
        toolbar.addWidget(self.wireframe_btn)

        self.particles_btn = QPushButton("Particles")
        self.particles_btn.setCheckable(True)
        self.particles_btn.setChecked(True)
        self.particles_btn.toggled.connect(self._toggle_particles)
        toolbar.addWidget(self.particles_btn)

        self.flow_btn = QPushButton("Flow Field")
        self.flow_btn.setCheckable(True)
        self.flow_btn.toggled.connect(self._toggle_flow)
        toolbar.addWidget(self.flow_btn)

        toolbar.addSeparator()

        # Reset view
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        toolbar.addWidget(reset_btn)

        # Fit all
        fit_btn = QPushButton("Fit All")
        fit_btn.clicked.connect(self._fit_all)
        toolbar.addWidget(fit_btn)

        toolbar.addSeparator()

        # Opacity slider
        toolbar.addWidget(QLabel(" Opacity: "))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setMaximumWidth(100)
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        toolbar.addWidget(self.opacity_slider)

        layout.addWidget(toolbar)

        # 3D viewer container
        self.viewer_container = QFrame()
        self.viewer_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer_container)

    def _setup_viewer(self):
        """Setup the PyVista viewer."""
        if not HAS_PYVISTA:
            # Fallback to placeholder
            placeholder = QLabel("3D Viewport\n(PyVista not installed)\n\nInstall with: pip install pyvistaqt")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background-color: #1e1e1e; color: #808080; font-size: 14px;")
            self.viewer_layout.addWidget(placeholder)
            self.plotter = None
            return

        # Create PyVista Qt interactor
        self.plotter = QtInteractor(self.viewer_container)
        self.plotter.set_background('#1e1e1e')
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()

        # Set up lighting
        self.plotter.enable_shadows()

        # Set Y as the vertical (up) axis
        self.plotter.camera.up = (0, 1, 0)

        self.viewer_layout.addWidget(self.plotter.interactor)

        # Initial view
        self._reset_view()

    def add_mesh(self, component_id: str, vertices: np.ndarray, faces: np.ndarray,
                 color: str = "#4ec9b0", opacity: float = 0.8) -> bool:
        """
        Add a mesh to the viewport.

        Args:
            component_id: Unique identifier for the component
            vertices: Nx3 array of vertex positions (geometry uses Z-up, will be transformed to Y-up)
            faces: Mx3 array of face indices (triangles)
            color: Hex color string
            opacity: Mesh opacity (0-1)

        Returns:
            True if successful
        """
        if not HAS_PYVISTA or self.plotter is None:
            return False

        try:
            # Transform from Z-up (geometry) to Y-up (viewport) coordinate system
            # Geometry uses: X=right, Y=depth, Z=up
            # Viewport uses: X=right, Y=up, Z=depth (into screen)
            # Transform: (x, y, z) -> (x, z, -y) to maintain right-handedness
            transformed_vertices = vertices.copy().astype(np.float64)
            transformed_vertices[:, 1] = vertices[:, 2]   # new Y = old Z (up)
            transformed_vertices[:, 2] = -vertices[:, 1]  # new Z = -old Y (depth, negated)

            # Create PyVista mesh
            # PyVista expects faces in format: [n_verts, v0, v1, v2, n_verts, v0, ...]
            n_faces = len(faces)
            pv_faces = np.zeros((n_faces, 4), dtype=np.int64)
            pv_faces[:, 0] = 3  # triangles
            pv_faces[:, 1:] = faces
            pv_faces = pv_faces.flatten()

            mesh = pv.PolyData(transformed_vertices, pv_faces)
            mesh.compute_normals(inplace=True)

            # Remove old mesh if exists
            if component_id in self._actors:
                self.plotter.remove_actor(self._actors[component_id])

            # Add to plotter
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity * self.opacity_slider.value() / 100,
                smooth_shading=True,
                name=component_id,
            )

            self._meshes[component_id] = mesh
            self._actors[component_id] = actor
            self._component_colors[component_id] = color

            return True

        except Exception as e:
            print(f"Error adding mesh: {e}")
            return False

    def remove_mesh(self, component_id: str):
        """Remove a mesh from the viewport."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        if component_id in self._actors:
            self.plotter.remove_actor(self._actors[component_id])
            del self._actors[component_id]

        if component_id in self._meshes:
            del self._meshes[component_id]

        if component_id in self._component_colors:
            del self._component_colors[component_id]

    def clear(self):
        """Clear all meshes from the viewport."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        self.plotter.clear()
        self._meshes.clear()
        self._actors.clear()
        self._component_colors.clear()
        self._particles_actor = None
        self._flow_actors.clear()

        # Re-add axes
        self.plotter.add_axes()

    @Slot(dict)
    def update_assembly(self, assembly_data: Dict[str, Any]):
        """
        Update viewport from assembly canvas data.

        Args:
            assembly_data: Dictionary with component meshes and positions
        """
        if not HAS_PYVISTA or self.plotter is None:
            return

        # Clear existing
        self.clear()

        # Add each component
        components = assembly_data.get("components", {})
        for comp_id, comp_data in components.items():
            mesh_data = comp_data.get("mesh")
            if mesh_data:
                self.add_mesh(
                    comp_id,
                    mesh_data["vertices"],
                    mesh_data["faces"],
                    color=comp_data.get("color", "#4ec9b0"),
                    opacity=comp_data.get("opacity", 0.8),
                )

        self._fit_all()

    def update_particles(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None,
                         colors: Optional[np.ndarray] = None, sizes: Optional[np.ndarray] = None):
        """
        Update particle positions in the viewport.

        Args:
            positions: Nx3 array of particle positions
            velocities: Optional Nx3 array of velocities for coloring
            colors: Optional Nx3 array of RGB colors (0-1)
            sizes: Optional N array of particle sizes
        """
        if not HAS_PYVISTA or self.plotter is None:
            return

        if not self._show_particles:
            return

        # Remove old particles
        if self._particles_actor is not None:
            self.plotter.remove_actor(self._particles_actor)
            self._particles_actor = None

        if len(positions) == 0:
            return

        # Transform from Z-up (simulation) to Y-up (viewport)
        # (x, y, z) -> (x, z, -y) to maintain right-handedness
        transformed_positions = positions.copy().astype(np.float64)
        transformed_positions[:, 1] = positions[:, 2]   # new Y = old Z
        transformed_positions[:, 2] = -positions[:, 1]  # new Z = -old Y

        # Create point cloud
        points = pv.PolyData(transformed_positions)

        # Add scalar data for coloring
        if velocities is not None:
            speeds = np.linalg.norm(velocities, axis=1)
            points["velocity"] = speeds
            scalars = "velocity"
        elif colors is not None:
            points["rgb"] = colors
            scalars = "rgb"
        else:
            scalars = None

        # Determine point size
        point_size = 5.0
        if sizes is not None:
            point_size = max(3.0, np.mean(sizes) * 10000)  # Scale for visibility

        # Add to plotter
        self._particles_actor = self.plotter.add_mesh(
            points,
            scalars=scalars,
            render_points_as_spheres=True,
            point_size=point_size,
            cmap="plasma" if scalars == "velocity" else None,
            show_scalar_bar=scalars == "velocity",
        )

    def update_flow_field(self, origins: np.ndarray, vectors: np.ndarray,
                          magnitudes: Optional[np.ndarray] = None):
        """
        Update flow field visualization.

        Args:
            origins: Nx3 array of vector origins
            vectors: Nx3 array of vector directions
            magnitudes: Optional N array of magnitudes for coloring
        """
        if not HAS_PYVISTA or self.plotter is None:
            return

        if not self._show_flow:
            return

        # Remove old flow actors
        for actor in self._flow_actors:
            self.plotter.remove_actor(actor)
        self._flow_actors.clear()

        if len(origins) == 0:
            return

        # Transform from Z-up (simulation) to Y-up (viewport)
        # (x, y, z) -> (x, z, -y) to maintain right-handedness
        transformed_origins = origins.copy().astype(np.float64)
        transformed_origins[:, 1] = origins[:, 2]   # new Y = old Z
        transformed_origins[:, 2] = -origins[:, 1]  # new Z = -old Y

        transformed_vectors = vectors.copy().astype(np.float64)
        transformed_vectors[:, 1] = vectors[:, 2]   # new Y = old Z
        transformed_vectors[:, 2] = -vectors[:, 1]  # new Z = -old Y

        # Create glyphs
        points = pv.PolyData(transformed_origins)
        points["vectors"] = transformed_vectors
        if magnitudes is not None:
            points["magnitude"] = magnitudes

        arrows = points.glyph(
            orient="vectors",
            scale="magnitude" if magnitudes is not None else False,
            factor=0.05,
        )

        actor = self.plotter.add_mesh(
            arrows,
            scalars="magnitude" if magnitudes is not None else None,
            cmap="coolwarm",
            show_scalar_bar=magnitudes is not None,
        )
        self._flow_actors.append(actor)

    def rebuild_from_canvas(self, canvas):
        """Rebuild viewport from assembly canvas."""
        assembly_data = canvas.get_assembly_data()
        self.update_assembly(assembly_data)

    @Slot(object, object)
    def update_from_backend_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        """
        Update viewport from simulation backend mesh data.

        This is connected to SimulationBackend.mesh_updated signal to display
        the CompleteClassifierAssembly geometry.

        Args:
            vertices: Nx3 array of vertex positions
            indices: Flat array of face indices (triangles, length divisible by 3)
        """
        if not HAS_PYVISTA or self.plotter is None:
            return

        if vertices is None or indices is None:
            return

        if len(vertices) == 0 or len(indices) == 0:
            return

        try:
            # Clear existing meshes
            self.clear()

            # Reshape indices to Mx3 triangles
            faces = np.array(indices).reshape(-1, 3)

            # Add as single combined mesh
            self.add_mesh(
                component_id="assembly",
                vertices=np.array(vertices),
                faces=faces,
                color="#4ec9b0",
                opacity=0.85,
            )

            # Fit to view
            self._fit_all()

        except Exception as e:
            print(f"Error updating viewport from backend mesh: {e}")

    def update_mesh_from_assembly(self, assembly):
        """
        Update viewport directly from an assembly object.

        Works with both ClassificationSystemAssembly and CompleteClassifierAssembly.

        Args:
            assembly: Assembly object with build_mesh() method
        """
        if assembly is None:
            return

        try:
            vertices, indices = assembly.build_mesh()
            self.update_from_backend_mesh(vertices, indices)
        except Exception as e:
            print(f"Error updating mesh from assembly: {e}")

    def _set_view(self, view_name: str):
        """Set camera to predefined view. Uses Y-up coordinate convention."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        # Y-up coordinate system view vectors:
        # - Y is vertical (up/down)
        # - Z is depth (front/back)
        # - X is horizontal (left/right)
        views = {
            "Isometric": (1, 1, 1),     # View from above-front-right
            "Front": (0, 0, 1),          # Looking toward -Z
            "Back": (0, 0, -1),          # Looking toward +Z
            "Left": (-1, 0, 0),          # Looking toward +X
            "Right": (1, 0, 0),          # Looking toward -X
            "Top": (0, 1, 0),            # Looking down from +Y
            "Bottom": (0, -1, 0),        # Looking up from -Y
        }

        if view_name in views:
            self.plotter.view_vector(views[view_name], viewup=(0, 1, 0))
            self._fit_all()

    def _reset_view(self):
        """Reset to default isometric view with Y as vertical axis."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        # Set Y-up isometric view
        self.plotter.view_vector((1, 1, 1), viewup=(0, 1, 0))
        self._fit_all()
        self.view_combo.setCurrentText("Isometric")

    def _fit_all(self):
        """Fit all geometry in view."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        self.plotter.reset_camera()

    def _toggle_wireframe(self, enabled: bool):
        """Toggle wireframe display mode."""
        self._show_wireframe = enabled

        if not HAS_PYVISTA or self.plotter is None:
            return

        for comp_id, actor in self._actors.items():
            if enabled:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()

        self.plotter.render()

    def _toggle_particles(self, enabled: bool):
        """Toggle particle visibility."""
        self._show_particles = enabled

        if self._particles_actor is not None:
            self._particles_actor.SetVisibility(enabled)
            if HAS_PYVISTA and self.plotter:
                self.plotter.render()

    def _toggle_flow(self, enabled: bool):
        """Toggle flow field visibility."""
        self._show_flow = enabled

        for actor in self._flow_actors:
            actor.SetVisibility(enabled)

        if HAS_PYVISTA and self.plotter:
            self.plotter.render()

    def _set_opacity(self, value: int):
        """Set mesh opacity (0-100)."""
        opacity = value / 100.0

        if not HAS_PYVISTA or self.plotter is None:
            return

        for comp_id, actor in self._actors.items():
            actor.GetProperty().SetOpacity(opacity)

        self.plotter.render()

    def export_mesh(self, file_path: str):
        """
        Export combined mesh to file.

        Args:
            file_path: Output file path (.stl, .obj, .vtk)
        """
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista not available")

        if not self._meshes:
            raise ValueError("No meshes to export")

        # Combine all meshes
        combined = pv.PolyData()
        for mesh in self._meshes.values():
            combined = combined.merge(mesh)

        # Save
        path = Path(file_path)
        if path.suffix.lower() == '.stl':
            combined.save(str(path))
        elif path.suffix.lower() == '.obj':
            combined.save(str(path))
        elif path.suffix.lower() == '.vtk':
            combined.save(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def save_state(self) -> Dict[str, Any]:
        """Save viewport state."""
        state = {
            "show_particles": self._show_particles,
            "show_flow": self._show_flow,
            "show_wireframe": self._show_wireframe,
            "opacity": self.opacity_slider.value(),
        }

        if HAS_PYVISTA and self.plotter:
            camera = self.plotter.camera
            state["camera"] = {
                "position": list(camera.position),
                "focal_point": list(camera.focal_point),
                "up": list(camera.up),
            }

        return state

    def load_state(self, state: Dict[str, Any]):
        """Load viewport state."""
        self._show_particles = state.get("show_particles", True)
        self._show_flow = state.get("show_flow", False)
        self._show_wireframe = state.get("show_wireframe", False)

        self.particles_btn.setChecked(self._show_particles)
        self.flow_btn.setChecked(self._show_flow)
        self.wireframe_btn.setChecked(self._show_wireframe)
        self.opacity_slider.setValue(state.get("opacity", 80))

        if HAS_PYVISTA and self.plotter and "camera" in state:
            cam = state["camera"]
            self.plotter.camera.position = cam["position"]
            self.plotter.camera.focal_point = cam["focal_point"]
            # Use saved up vector, default to Y-up if not present
            self.plotter.camera.up = cam.get("up", (0, 1, 0))
            self.plotter.render()
