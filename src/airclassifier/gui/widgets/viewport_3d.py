"""
3D Viewport Widget
==================

PyVista-based 3D visualization widget for displaying air classifier
geometry, particles, and flow fields.

Coordinate System (Y-up, matching all geometry assemblies):
- X: Horizontal (width)
- Y: Vertical (height) - UP
- Z: Horizontal (depth)

No coordinate transforms are applied -- geometry is already Y-up.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QPushButton,
    QComboBox, QLabel, QSlider, QCheckBox, QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont

from ..theme import COLORS

# Try to import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not available. 3D viewport will be limited.")


# ---------------------------------------------------------------------------
# Component colors -- matching visualize_geometry.py
# ---------------------------------------------------------------------------
COMPONENT_RENDER_COLORS = {
    "cyclone":          "#4A90D9",
    "multicyclone":     "#E74C3C",
    "blower":           "#27AE60",
    "deagglomerator":   "#9B59B6",
    "hopper":           "#F0AD4E",
    "airlock":          "#3498DB",
    "zigzag":           "#2ECC71",
    "venturi":          "#1ABC9C",
    "bagfilter":        "#F39C12",
    "wheel":            "#FF6B6B",
    # fallback
    "default":          "#4A90D9",
}

# Default mesh opacity -- matches visualize_geometry.py
_DEFAULT_OPACITY = 0.85


class Viewport3D(QWidget):
    """
    3D viewport widget for visualizing air classifier geometry and simulation.

    Coordinate System (Y-up):
        X = right, Y = up, Z = depth
    No axis transforms are applied; geometry already uses Y-up.
    """

    # Signals
    view_changed = Signal()
    component_clicked = Signal(str)  # component_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._meshes: Dict[str, Any] = {}
        self._actors: Dict[str, Any] = {}
        self._particles_actor = None
        self._flow_actors: List[Any] = []
        self._show_particles = True
        self._show_flow = False
        self._show_wireframe = False
        self._show_edges = True
        self._component_colors: Dict[str, str] = {}

        self._setup_ui()
        self._setup_viewer()

    # ================================================================
    #  UI
    # ================================================================

    def _setup_ui(self):
        """Setup the widget UI with a modern toolbar."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet(f"""
            QToolBar {{
                background: {COLORS.BG_BASE};
                border-bottom: 1px solid {COLORS.BORDER_SUBTLE};
                spacing: 3px;
                padding: 2px 4px;
            }}
        """)

        # View combo
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Isometric", "Front", "Back", "Left", "Right", "Top", "Bottom"
        ])
        self.view_combo.setFixedWidth(100)
        self.view_combo.currentTextChanged.connect(self._set_view)
        toolbar.addWidget(QLabel(" View: "))
        toolbar.addWidget(self.view_combo)
        toolbar.addSeparator()

        # Toggle buttons
        self.edges_btn = QPushButton("Edges")
        self.edges_btn.setCheckable(True)
        self.edges_btn.setChecked(True)
        self.edges_btn.toggled.connect(self._toggle_edges)
        toolbar.addWidget(self.edges_btn)

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

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        toolbar.addWidget(reset_btn)

        fit_btn = QPushButton("Fit All")
        fit_btn.clicked.connect(self._fit_all)
        toolbar.addWidget(fit_btn)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Opacity: "))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(_DEFAULT_OPACITY * 100))
        self.opacity_slider.setFixedWidth(90)
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        toolbar.addWidget(self.opacity_slider)

        layout.addWidget(toolbar)

        # 3D viewer container
        self.viewer_container = QFrame()
        self.viewer_container.setFrameShape(QFrame.Shape.NoFrame)
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer_container)

    def _setup_viewer(self):
        """Setup the PyVista viewer or show a styled placeholder."""
        if not HAS_PYVISTA:
            placeholder = QFrame()
            placeholder.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS.BG_DARKEST};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 6px;
                }}
            """)
            p_layout = QVBoxLayout(placeholder)
            p_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            icon_lbl = QLabel("3D Viewport")
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_lbl.setStyleSheet(
                f"font-size: 16pt; font-weight: 700; color: {COLORS.TEXT_MUTED};"
                " border: none; background: transparent;"
            )
            p_layout.addWidget(icon_lbl)

            hint = QLabel("PyVista is not installed.\nInstall with: pip install pyvistaqt")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet(
                f"font-size: 10pt; color: {COLORS.TEXT_DISABLED};"
                " border: none; background: transparent;"
            )
            p_layout.addWidget(hint)

            self.viewer_layout.addWidget(placeholder)
            self.plotter = None
            return

        # ---- PyVista setup (matching visualize_geometry.py style) ----
        self.plotter = QtInteractor(self.viewer_container)
        self.plotter.set_background(COLORS.BG_DARKEST)
        self.plotter.add_axes()
        self.plotter.enable_anti_aliasing()

        # Y-up coordinate system (same as geometry code)
        self.plotter.camera.up = (0, 1, 0)

        self.viewer_layout.addWidget(self.plotter.interactor)

        # Initial view -- match visualize_geometry.py camera
        self._reset_view()

    # ================================================================
    #  Mesh management
    # ================================================================

    def add_mesh(
        self,
        component_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: str = "#4A90D9",
        opacity: float = _DEFAULT_OPACITY,
    ) -> bool:
        """
        Add a mesh to the viewport.

        Vertices are used AS-IS (Y-up, matching all geometry assemblies).
        No coordinate transform is applied.

        Args:
            component_id: Unique identifier for the component
            vertices: Nx3 array of vertex positions (Y-up)
            faces: Mx3 array of triangle indices
            color: Hex color string
            opacity: Mesh opacity (0-1)
        """
        if not HAS_PYVISTA or self.plotter is None:
            return False

        try:
            verts = np.asarray(vertices, dtype=np.float64)

            # Build PyVista face array: [3, v0, v1, v2, 3, v0, v1, v2, ...]
            n_faces = len(faces)
            pv_faces = np.zeros((n_faces, 4), dtype=np.int64)
            pv_faces[:, 0] = 3  # triangles
            pv_faces[:, 1:] = faces
            pv_faces = pv_faces.flatten()

            mesh = pv.PolyData(verts, pv_faces)
            mesh.compute_normals(inplace=True)

            # Remove old mesh if it exists for this id
            if component_id in self._actors:
                self.plotter.remove_actor(self._actors[component_id])

            effective_opacity = opacity * self.opacity_slider.value() / 100

            # Render style matching visualize_geometry.py:
            # solid + edge overlay for depth cues
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=effective_opacity,
                show_edges=self._show_edges,
                edge_color="gray",
                smooth_shading=True,
                name=component_id,
            )

            self._meshes[component_id] = mesh
            self._actors[component_id] = actor
            self._component_colors[component_id] = color
            return True

        except Exception as e:
            print(f"Error adding mesh '{component_id}': {e}")
            return False

    def remove_mesh(self, component_id: str):
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
        if not HAS_PYVISTA or self.plotter is None:
            return
        self.plotter.clear()
        self._meshes.clear()
        self._actors.clear()
        self._component_colors.clear()
        self._particles_actor = None
        self._flow_actors.clear()
        self.plotter.add_axes()

    # ================================================================
    #  High-level update helpers
    # ================================================================

    @Slot(dict)
    def update_assembly(self, assembly_data: Dict[str, Any]):
        """Update viewport from assembly canvas data."""
        if not HAS_PYVISTA or self.plotter is None:
            return
        self.clear()
        components = assembly_data.get("components", {})
        for comp_id, comp_data in components.items():
            mesh_data = comp_data.get("mesh")
            if mesh_data:
                self.add_mesh(
                    comp_id,
                    mesh_data["vertices"],
                    mesh_data["faces"],
                    color=comp_data.get("color", "#4A90D9"),
                    opacity=comp_data.get("opacity", _DEFAULT_OPACITY),
                )
        self._fit_all()

    def update_particles(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
    ):
        """
        Update particle positions.  No coordinate transform -- Y-up as-is.
        """
        if not HAS_PYVISTA or self.plotter is None or not self._show_particles:
            return

        if self._particles_actor is not None:
            self.plotter.remove_actor(self._particles_actor)
            self._particles_actor = None

        if len(positions) == 0:
            return

        points = pv.PolyData(np.asarray(positions, dtype=np.float64))

        scalars = None
        if velocities is not None:
            points["velocity"] = np.linalg.norm(velocities, axis=1)
            scalars = "velocity"
        elif colors is not None:
            points["rgb"] = colors
            scalars = "rgb"

        point_size = 5.0
        if sizes is not None:
            point_size = max(3.0, np.mean(sizes) * 10000)

        self._particles_actor = self.plotter.add_mesh(
            points,
            scalars=scalars,
            render_points_as_spheres=True,
            point_size=point_size,
            cmap="plasma" if scalars == "velocity" else None,
            show_scalar_bar=scalars == "velocity",
        )

    def update_flow_field(
        self,
        origins: np.ndarray,
        vectors: np.ndarray,
        magnitudes: Optional[np.ndarray] = None,
    ):
        """
        Update flow field visualization.  No coordinate transform.
        """
        if not HAS_PYVISTA or self.plotter is None or not self._show_flow:
            return

        for actor in self._flow_actors:
            self.plotter.remove_actor(actor)
        self._flow_actors.clear()

        if len(origins) == 0:
            return

        points = pv.PolyData(np.asarray(origins, dtype=np.float64))
        points["vectors"] = np.asarray(vectors, dtype=np.float64)
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
        assembly_data = canvas.get_assembly_data()
        self.update_assembly(assembly_data)

    @Slot(object, object)
    def update_from_backend_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        """
        Display mesh from SimulationBackend.

        Vertices are Y-up, used as-is (no axis swap).
        """
        if not HAS_PYVISTA or self.plotter is None:
            return
        if vertices is None or indices is None or len(vertices) == 0 or len(indices) == 0:
            return
        try:
            self.clear()
            faces = np.array(indices).reshape(-1, 3)
            self.add_mesh(
                "assembly",
                np.array(vertices),
                faces,
                color=COMPONENT_RENDER_COLORS["default"],
                opacity=_DEFAULT_OPACITY,
            )
            self._fit_all()
        except Exception as e:
            print(f"Error updating viewport from backend mesh: {e}")

    def update_mesh_from_assembly(self, assembly):
        if assembly is None:
            return
        try:
            vertices, indices = assembly.build_mesh()
            self.update_from_backend_mesh(vertices, indices)
        except Exception as e:
            print(f"Error updating mesh from assembly: {e}")

    # ================================================================
    #  Camera / View helpers
    # ================================================================

    def _set_view(self, view_name: str):
        """Set camera to a predefined view (Y-up)."""
        if not HAS_PYVISTA or self.plotter is None:
            return

        # Y-up view vectors
        views = {
            "Isometric": None,  # handled specially with azimuth/elevation
            "Front":  (0, 0, 1),
            "Back":   (0, 0, -1),
            "Left":   (-1, 0, 0),
            "Right":  (1, 0, 0),
            "Top":    (0, 1, 0.001),   # slight offset to avoid gimbal
            "Bottom": (0, -1, 0.001),
        }

        if view_name == "Isometric":
            self._reset_view()
        elif view_name in views:
            self.plotter.view_vector(views[view_name], viewup=(0, 1, 0))
            self._fit_all()

    def _reset_view(self):
        """Reset to default isometric view matching visualize_geometry.py."""
        if not HAS_PYVISTA or self.plotter is None:
            return
        self.plotter.reset_camera()
        self.plotter.camera.up = (0, 1, 0)
        self.plotter.camera.azimuth = -170
        self.plotter.camera.elevation = -20
        self.view_combo.setCurrentText("Isometric")

    def _fit_all(self):
        if not HAS_PYVISTA or self.plotter is None:
            return
        self.plotter.reset_camera()

    # ================================================================
    #  Display toggles
    # ================================================================

    def _toggle_edges(self, enabled: bool):
        """Toggle edge visibility on solid meshes."""
        self._show_edges = enabled
        if not HAS_PYVISTA or self.plotter is None:
            return
        # Re-add meshes with updated edge visibility
        for comp_id in list(self._meshes.keys()):
            mesh = self._meshes[comp_id]
            color = self._component_colors.get(comp_id, "#4A90D9")
            if comp_id in self._actors:
                self.plotter.remove_actor(self._actors[comp_id])
            opacity = self.opacity_slider.value() / 100
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                show_edges=enabled,
                edge_color="gray",
                smooth_shading=True,
                name=comp_id,
            )
            self._actors[comp_id] = actor
        self.plotter.render()

    def _toggle_wireframe(self, enabled: bool):
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
        self._show_particles = enabled
        if self._particles_actor is not None:
            self._particles_actor.SetVisibility(enabled)
            if HAS_PYVISTA and self.plotter:
                self.plotter.render()

    def _toggle_flow(self, enabled: bool):
        self._show_flow = enabled
        for actor in self._flow_actors:
            actor.SetVisibility(enabled)
        if HAS_PYVISTA and self.plotter:
            self.plotter.render()

    def _set_opacity(self, value: int):
        opacity = value / 100.0
        if not HAS_PYVISTA or self.plotter is None:
            return
        for comp_id, actor in self._actors.items():
            actor.GetProperty().SetOpacity(opacity)
        self.plotter.render()

    # ================================================================
    #  Export / state
    # ================================================================

    def export_mesh(self, file_path: str):
        if not HAS_PYVISTA:
            raise RuntimeError("PyVista not available")
        if not self._meshes:
            raise ValueError("No meshes to export")
        combined = pv.PolyData()
        for mesh in self._meshes.values():
            combined = combined.merge(mesh)
        combined.save(str(file_path))

    def save_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "show_particles": self._show_particles,
            "show_flow": self._show_flow,
            "show_wireframe": self._show_wireframe,
            "show_edges": self._show_edges,
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
        self._show_particles = state.get("show_particles", True)
        self._show_flow = state.get("show_flow", False)
        self._show_wireframe = state.get("show_wireframe", False)
        self._show_edges = state.get("show_edges", True)
        self.particles_btn.setChecked(self._show_particles)
        self.flow_btn.setChecked(self._show_flow)
        self.wireframe_btn.setChecked(self._show_wireframe)
        self.edges_btn.setChecked(self._show_edges)
        self.opacity_slider.setValue(state.get("opacity", int(_DEFAULT_OPACITY * 100)))
        if HAS_PYVISTA and self.plotter and "camera" in state:
            cam = state["camera"]
            self.plotter.camera.position = cam["position"]
            self.plotter.camera.focal_point = cam["focal_point"]
            self.plotter.camera.up = cam.get("up", (0, 1, 0))
            self.plotter.render()
