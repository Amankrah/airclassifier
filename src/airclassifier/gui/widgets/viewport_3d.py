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
from ..runtime import pyvista_unavailable_message

# Try to import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception as e:
    HAS_PYVISTA = False
    _PYVISTA_ERROR = str(e)
    print(f"Warning: PyVista not available ({e}). 3D viewport will be limited.")


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

        toolbar.addSeparator()

        # Cinematic camera controls
        self.cinematic_btn = QPushButton("Cinematic")
        self.cinematic_btn.setCheckable(True)
        self.cinematic_btn.setChecked(False)
        self.cinematic_btn.setToolTip(
            "Enable cinematic camera — automatic camera movement\n"
            "while the simulation runs (game-style showcase).\n"
            "Move the mouse to temporarily take back control."
        )
        self.cinematic_btn.toggled.connect(self._toggle_cinematic)
        toolbar.addWidget(self.cinematic_btn)

        self.cinematic_mode_combo = QComboBox()
        self.cinematic_mode_combo.addItems(["Orbit", "Showcase", "Flythrough"])
        self.cinematic_mode_combo.setFixedWidth(100)
        self.cinematic_mode_combo.setToolTip(
            "Orbit: smooth rotation around the assembly\n"
            "Showcase: guided tour of key viewpoints\n"
            "Flythrough: scripted spiral sweep"
        )
        self.cinematic_mode_combo.currentTextChanged.connect(self._on_cinematic_mode_changed)
        toolbar.addWidget(self.cinematic_mode_combo)

        layout.addWidget(toolbar)

        # 3D viewer container
        self.viewer_container = QFrame()
        self.viewer_container.setFrameShape(QFrame.Shape.NoFrame)
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer_container)

    def _setup_viewer(self):
        """Setup the PyVista viewer or show a styled placeholder."""
        self._cinematic_camera = None  # CinematicCameraController

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

            hint = QLabel(pyvista_unavailable_message())
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

        # Hook into the VTK interactor to detect mouse interaction
        # so the cinematic camera pauses when the user grabs the view.
        iren = self.plotter.interactor.GetRenderWindow().GetInteractor()
        iren.AddObserver("StartInteractionEvent", self._on_user_interaction)

        # Create cinematic camera controller
        from .cinematic_camera import CinematicCameraController
        self._cinematic_camera = CinematicCameraController(self.plotter, self)

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
    #  Animation support
    # ================================================================

    def build_with_animation(self, complete_assembly) -> Optional["AnimationController"]:
        """
        Build the assembly with animated parts separated from static geometry.

        Two-pass approach:
        1. Build all subsystem meshes, but for components that have animation
           (blower, dampers, airlocks, screw, deagglomerator, hopper), use
           get_static_mesh()/get_body_mesh() instead of generate_mesh() so
           the static assembly mesh does NOT contain rotors/blades/lid.
        2. Register the animated sub-meshes (rotors, blades, lid) separately
           with the AnimationController.

        This way the animated parts are the ONLY meshes at those locations --
        no overlay/ghost of the static version underneath.

        Args:
            complete_assembly: CompleteClassifierAssembly instance

        Returns:
            AnimationController ready to start(), or None if PyVista unavailable.
        """
        if not HAS_PYVISTA or self.plotter is None:
            return None

        from .animation_controller import AnimationController

        self.clear()
        controller = AnimationController(self.plotter, self)

        try:
            self._build_static_and_animated(complete_assembly, controller)
        except Exception as e:
            import traceback
            print(f"Error building animated assembly: {e}")
            traceback.print_exc()
            # Fallback: build full combined mesh (no animation separation)
            try:
                vertices, indices = complete_assembly.build_mesh()
                if vertices is not None and len(vertices) > 0:
                    faces = np.array(indices).reshape(-1, 3)
                    self.add_mesh("assembly_static", np.array(vertices), faces,
                                  color=COMPONENT_RENDER_COLORS["default"], opacity=0.85)
            except Exception:
                pass

        # Render all animated parts at their initial resting position (angle=0)
        # so they are visible in the viewport immediately -- even before the
        # user clicks Run Simulation.  When animation starts, the tick loop
        # takes over and updates these same actors each frame.
        controller.render_initial_state()

        self._fit_all()
        return controller

    def _build_static_and_animated(self, complete_assembly, controller):
        """
        Build the assembly in two layers:
        - Static mesh: everything that doesn't move (housings, ducts, trough, etc.)
        - Animated meshes: rotors, blades, lid, wheel (registered with controller)

        For each animated component, we use get_static_mesh() for the static
        layer and register get_rotor_mesh()/get_lid_mesh() with the controller.
        For all other components, we use the full generate_mesh() / build_mesh().
        """
        subs = getattr(complete_assembly, '_subsystems', {})
        get_sub = getattr(complete_assembly, 'get_subsystem', None)
        if get_sub is None:
            # No subsystem access -- fall back to combined mesh
            vertices, indices = complete_assembly.build_mesh()
            if vertices is not None and len(vertices) > 0:
                faces = np.array(indices).reshape(-1, 3)
                self.add_mesh("assembly_static", np.array(vertices), faces,
                              color=COMPONENT_RENDER_COLORS["default"], opacity=0.85)
            return

        all_static_verts = []
        all_static_indices = []
        vertex_offset = 0
        C_default = COMPONENT_RENDER_COLORS["default"]

        def _add_static(verts, idx):
            """Add a mesh chunk to the combined static mesh."""
            nonlocal vertex_offset
            if verts is None or len(verts) == 0:
                return
            all_static_verts.append(np.asarray(verts, dtype=np.float32))
            all_static_indices.append(np.asarray(idx, dtype=np.int32) + vertex_offset)
            vertex_offset += len(verts)

        def _sub_offset(name):
            key = f"{name}_offset"
            return np.array(subs.get(key, (0, 0, 0)), dtype=np.float64)

        # ================================================================
        # CLASSIFICATION SUBSYSTEM -- build component-by-component
        # so the wheel classifier is EXCLUDED from the static mesh
        # (only the animated version renders the wheel).
        # ================================================================
        classification = get_sub("classification")
        cls_offset = _sub_offset("classification")
        if classification is not None:
            cls_positions = classification.get_component_positions() if hasattr(classification, 'get_component_positions') else {}

            # Static components: everything EXCEPT the wheel classifier
            # (venturi, zigzag, multi_cyclone, bag_filter, ducts, collection hardware)
            for comp_name in ['venturi', 'zigzag', 'multi_cyclone', 'bag_filter']:
                comp = getattr(classification, comp_name, None) or getattr(classification, comp_name.replace('_', ''), None)
                if comp is None:
                    continue
                cpos = np.array(cls_positions.get(comp_name, (0, 0, 0)), dtype=np.float64) + cls_offset
                try:
                    cv, ci, _ = comp.generate_mesh()
                    if cv is not None and len(cv) > 0:
                        _add_static(cv + cpos, ci)
                except Exception as e:
                    print(f"  Static: classification {comp_name} failed: {e}")

            # Wheel classifier: housing goes in static mesh, blades are animated.
            wheel = getattr(classification, 'wheel_classifier', None) or getattr(classification, 'wheel', None)
            wheel_pos_key = 'wheel_classifier' if 'wheel_classifier' in cls_positions else 'wheel'

            if wheel is not None:
                wpos = np.array(cls_positions.get(wheel_pos_key, (0, 0, 0)), dtype=np.float64) + cls_offset
                rpm = getattr(wheel.params, 'rpm', 8000.0) if hasattr(wheel, 'params') else 8000.0

                # Static: housing, motor, hopper, inlets, outlets (everything except blades)
                try:
                    if hasattr(wheel, 'get_static_mesh'):
                        sv, si, _ = wheel.get_static_mesh()
                    else:
                        sv, si, _ = wheel.generate_mesh()
                    if sv is not None and len(sv) > 0:
                        _add_static(sv + wpos, si)
                except Exception as e:
                    print(f"  Static: wheel housing failed: {e}")

                # Animated: only the classifier wheel (shrouds + blades + hub)
                try:
                    if hasattr(wheel, 'get_wheel_mesh'):
                        wv, wi, _ = wheel.get_wheel_mesh()
                    else:
                        wv, wi, _ = wheel.generate_mesh()
                    if wv is not None and len(wv) > 0:
                        verts_world = wv.copy().astype(np.float64) + wpos
                        wheel_pv = self._numpy_to_pv(verts_world, wi)
                        controller.register_wheel(wheel_pv, wpos, rpm, color="#FF6B6B")
                        print(f"  Animation: wheel blades ({rpm:.0f} RPM) [housing is static]")
                except Exception as e:
                    print(f"  Animation: wheel blades failed: {e}")

            # Duct sections (static only -- no animated parts)
            for duct_list_attr in ['_duct_sections', '_collection_duct_sections']:
                for duct, position in getattr(classification, duct_list_attr, []):
                    try:
                        dv, di, _ = duct.generate_mesh()
                        if dv is not None and len(dv) > 0:
                            _add_static(dv + np.array(position, dtype=np.float64) + cls_offset, di)
                    except Exception:
                        pass

            # Classification airlocks: static housing + animated rotor
            for attr_name, label in [
                ('coarse_airlock', 'class_coarse_airlock'),
                ('dropout_airlock', 'class_dropout_airlock'),
                ('wheel_coarse_airlock', 'class_wheel_airlock'),
            ]:
                al = getattr(classification, attr_name, None)
                if al is not None:
                    pos = np.array(cls_positions.get(attr_name, (0, 0, 0)), dtype=np.float64) + cls_offset
                    # Static: airlock housing
                    try:
                        if hasattr(al, 'get_static_mesh'):
                            sv, si, _ = al.get_static_mesh()
                        else:
                            sv, si, _ = al.generate_mesh()
                        if sv is not None and len(sv) > 0:
                            _add_static(sv + pos, si)
                    except Exception:
                        pass
                    # Animated: rotor
                    rpm = getattr(al.params, 'rpm', 20.0) if hasattr(al, 'params') else 20.0
                    controller.register_airlock(al, pos, rpm, name=label, phase="classification", color="#3498DB")
                    print(f"  Animation: {label} ({rpm:.0f} RPM)")

        # ================================================================
        # FEED SYSTEM -- use get_body_mesh/get_static_mesh for animated parts
        # ================================================================
        feed = get_sub("feed_system")
        feed_offset = _sub_offset("feed_system")
        if feed is not None:
            feed_positions = feed.get_component_positions() if hasattr(feed, 'get_component_positions') else {}

            def _fpos(key):
                return np.array(feed_positions.get(key, (0, 0, 0)), dtype=np.float64) + feed_offset

            # Hopper: use get_body_mesh() (without lid) for static, register lid separately
            hopper = getattr(feed, 'hopper', None)
            if hopper is not None:
                hp = _fpos('hopper')
                try:
                    if hasattr(hopper, 'get_body_mesh'):
                        bv, bi, _ = hopper.get_body_mesh()
                        _add_static(bv + hp, bi)
                        # Register lid for animation
                        controller.register_hopper_lid(hopper, hp, color="#F0AD4E")
                        print(f"  Animation: hopper lid (hinge-open)")
                    else:
                        v, i, _ = hopper.generate_mesh()
                        _add_static(v + hp, i)
                except Exception as e:
                    print(f"  Static: hopper failed: {e}")

            # Airlock: use get_static_mesh() for housing, register rotor
            airlock = getattr(feed, 'airlock', None)
            if airlock is not None:
                ap = _fpos('airlock')
                try:
                    sv, si, _ = airlock.get_static_mesh()
                    _add_static(sv + ap, si)
                    rpm = getattr(airlock.params, 'rpm', 20.0) if hasattr(airlock, 'params') else 20.0
                    controller.register_airlock(airlock, ap, rpm, name="feed_airlock", phase="feed", color="#3498DB")
                    print(f"  Animation: feed airlock ({rpm:.0f} RPM)")
                except Exception as e:
                    print(f"  Static: airlock failed: {e}")

            # Screw feeder: use get_static_mesh() for trough, register screw
            screw = getattr(feed, 'feeder', None)
            if screw is not None:
                sp = _fpos('feeder')
                try:
                    sv, si, _ = screw.get_static_mesh()
                    _add_static(sv + sp, si)
                    rpm = getattr(screw.params, 'rpm', 60.0) if hasattr(screw, 'params') else 60.0
                    controller.register_screw(screw, sp, rpm, color="#2ECC71")
                    print(f"  Animation: screw feeder ({rpm:.0f} RPM)")
                except Exception as e:
                    print(f"  Static: screw failed: {e}")

            # Deagglomerator: use get_static_mesh() for housing, register rotor
            deagg = getattr(feed, 'deagglomerator', None)
            if deagg is not None:
                dp = _fpos('deagglomerator')
                try:
                    sv, si, _ = deagg.get_static_mesh()
                    _add_static(sv + dp, si)
                    rpm = getattr(deagg.params, 'rpm', 1500.0) if hasattr(deagg, 'params') else 1500.0
                    controller.register_deagglomerator(deagg, dp, rpm, color="#9B59B6")
                    print(f"  Animation: deagglomerator ({rpm:.0f} RPM)")
                except Exception as e:
                    print(f"  Static: deagg failed: {e}")

            # Feed system transition connectors (static only)
            # Note: transition mesh positions are already baked into
            # TransitionParams.center during feed assembly construction,
            # so we only apply the feed_offset (subsystem offset within
            # the complete assembly) -- NOT the connector_data[1] position.
            # This matches feed_system.build_mesh() which uses (0,0,0) offset.
            try:
                for connector_data in getattr(feed, '_transition_connectors', []):
                    connector = connector_data[0]
                    cv, ci, _ = connector.generate_mesh()
                    _add_static(cv + feed_offset, ci)
            except Exception:
                pass

        # ================================================================
        # AIR SYSTEM -- build component-by-component so animated parts
        # (blower impeller, damper blades) are EXCLUDED from static mesh.
        # ================================================================
        air = get_sub("air_system")
        air_offset = _sub_offset("air_system")
        if air is not None:
            air_positions = air.get_component_positions() if hasattr(air, 'get_component_positions') else {}

            def _apos(key):
                return np.array(air_positions.get(key, (0, 0, 0)), dtype=np.float64) + air_offset

            # --- Inlet filter: fully static ---
            inlet_filter = getattr(air, 'inlet_filter', None)
            if inlet_filter is not None:
                fp = np.array(getattr(air, '_filter_position', (0, 0, 0)), dtype=np.float64) + air_offset
                try:
                    fv, fi, _ = inlet_filter.generate_mesh()
                    if fv is not None and len(fv) > 0:
                        _add_static(fv + fp, fi)
                except Exception as e:
                    print(f"  Static: air inlet_filter failed: {e}")

            # --- Blower: static housing only (impeller is animated) ---
            blower = getattr(air, 'blower', None)
            if blower is not None:
                bp = np.array(getattr(air, '_blower_position', (0, 0, 0)), dtype=np.float64) + air_offset
                try:
                    if hasattr(blower, 'get_static_mesh'):
                        sv, si, _ = blower.get_static_mesh()
                    else:
                        sv, si, _ = blower.generate_mesh()
                    if sv is not None and len(sv) > 0:
                        _add_static(sv + bp, si)
                except Exception as e:
                    print(f"  Static: air blower housing failed: {e}")
                # Register blower impeller for animation
                rpm = getattr(blower.params, 'rpm', 3000.0) if hasattr(blower, 'params') else 3000.0
                controller.register_blower(blower, bp, rpm, color="#27AE60")
                print(f"  Animation: blower ({rpm:.0f} RPM) [no static duplicate]")

            # --- Dampers: static housing only (blades are animated) ---
            dampers = getattr(air, 'dampers', []) or getattr(air, '_dampers', [])
            damper_positions_list = getattr(air, '_damper_positions', [])
            for i, damper in enumerate(dampers):
                if damper is None:
                    continue
                dp = np.array(damper_positions_list[i], dtype=np.float64) + air_offset if i < len(damper_positions_list) else _apos(f'damper_{i}')
                try:
                    if hasattr(damper, 'get_static_mesh'):
                        sv, si, _ = damper.get_static_mesh()
                    else:
                        sv, si, _ = damper.generate_mesh()
                    if sv is not None and len(sv) > 0:
                        _add_static(sv + dp, si)
                except Exception as e:
                    print(f"  Static: air damper {i} housing failed: {e}")
                # Register blade for animation
                controller.register_damper(damper, dp, index=i, color="#CD853F")
                print(f"  Animation: damper {i} [no static duplicate]")

            # --- Duct sections: fully static ---
            for duct, pos in getattr(air, '_duct_sections', []):
                try:
                    dv, di, _ = duct.generate_mesh()
                    if dv is not None and len(dv) > 0:
                        _add_static(dv + np.array(pos, dtype=np.float64) + air_offset, di)
                except Exception:
                    pass

        # ================================================================
        # EXHAUST + DUCT CONNECTIONS (static only)
        # ================================================================
        # Individual components (silencer, stack)
        for name, component in getattr(complete_assembly, '_components', {}).items():
            try:
                cv, ci, _ = component.generate_mesh()
                _add_static(cv, ci)
            except Exception:
                pass

        # Connecting ductwork
        for duct, position in getattr(complete_assembly, '_duct_connections', []):
            try:
                cv, ci, _ = duct.generate_mesh()
                _add_static(cv + np.array(position), ci)
            except Exception:
                pass

        # ================================================================
        # Combine all static geometry into one mesh
        # ================================================================
        if all_static_verts:
            combined_v = np.vstack(all_static_verts).astype(np.float32)
            combined_i = np.concatenate(all_static_indices).astype(np.int32)
            faces = combined_i.reshape(-1, 3)
            self.add_mesh("assembly_static", combined_v, faces,
                          color=C_default, opacity=0.85)

    @staticmethod
    def _numpy_to_pv(vertices: np.ndarray, indices: np.ndarray):
        """Convert numpy vertices+indices to PyVista PolyData."""
        faces = indices.reshape(-1, 3)
        n = len(faces)
        pv_faces = np.zeros((n, 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = faces
        return pv.PolyData(vertices.astype(np.float64), pv_faces.flatten())

    # _register_animated_parts replaced by _build_static_and_animated above

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
    #  Cinematic camera
    # ================================================================

    def _toggle_cinematic(self, enabled: bool):
        """Toggle the cinematic camera on/off."""
        if self._cinematic_camera is None:
            return
        if enabled:
            self._cinematic_camera.start()
        else:
            self._cinematic_camera.stop()

    def _on_cinematic_mode_changed(self, mode_text: str):
        """Switch the cinematic camera mode from the combo box."""
        if self._cinematic_camera is None:
            return
        from .cinematic_camera import CameraMode
        mode_map = {
            "Orbit": CameraMode.ORBIT,
            "Showcase": CameraMode.SHOWCASE,
            "Flythrough": CameraMode.FLYTHROUGH,
        }
        mode = mode_map.get(mode_text, CameraMode.ORBIT)
        self._cinematic_camera.set_mode(mode)

    def _on_user_interaction(self, obj=None, event=None):
        """Called by VTK when the user starts a mouse interaction.

        Pauses the cinematic camera for a few seconds so the user can
        freely rotate/pan/zoom, then the cinematic camera resumes.
        """
        if self._cinematic_camera is not None and self._cinematic_camera.is_running:
            self._cinematic_camera.pause_for_interaction()

    def start_cinematic(self):
        """Programmatic API: start the cinematic camera (e.g. from MainWindow)."""
        if self._cinematic_camera is not None:
            self.cinematic_btn.setChecked(True)  # also triggers _toggle_cinematic

    def stop_cinematic(self):
        """Programmatic API: stop the cinematic camera."""
        if self._cinematic_camera is not None:
            self.cinematic_btn.setChecked(False)

    @property
    def cinematic_enabled(self) -> bool:
        return self.cinematic_btn.isChecked()

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
