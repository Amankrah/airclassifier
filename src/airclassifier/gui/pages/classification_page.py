"""
Classification Page — Air Classification Full-Page View
========================================================

Self-contained page for air classification simulation with:

  1. **3D Viewport** — PyVista-based geometry preview with animated
     rotating/moving components (wheel, blower, dampers, lid).
  2. **Simulation Control** — Full-featured ``SimulationControlPanel``
     (particle config, physics, recirculation, etc.).
  3. **Results Panel** — KPI cards, separation chart, collection table.

The page owns the entire classification lifecycle: build assembly,
start/pause/stop simulation, manage animation controllers, and
display results.  ``MainWindow`` is now a thin shell that routes
mode switches and project-level operations.

Architecture::

    ClassificationPage
        ├── QSplitter (vertical)
        │   ├── Viewport3D (3D geometry + animation)
        │   └── QTabWidget
        │       ├── Simulation Control  (SimulationControlPanel)
        │       └── Results             (ResultsPanel)
        ├── AnimationController  (wheel spin, damper motion, lid servo)
        └── SimulationBackend    (build + mesh generation)

Usage::

    from airclassifier.gui.pages import ClassificationPage

    page = ClassificationPage()
    stacked.addWidget(page)
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..panels.simulation_control import SimulationControlPanel
from ..panels.results_panel import ResultsPanel
from ..widgets.viewport_3d import Viewport3D
from ..theme import COLORS


class ClassificationPage(QWidget):
    """Full-page air classification simulation and visualization.

    Signals
    -------
    simulation_state_changed : Signal(str)
        ``"idle"`` | ``"running"`` | ``"paused"`` | ``"completed"``
    simulation_finished : Signal(dict)
        Emitted with results when a classification run ends.
    log_message : Signal(str)
        Forwarded log messages for the status bar.
    """

    simulation_state_changed = Signal(str)
    simulation_finished = Signal(dict)
    log_message = Signal(str)

    # Timing constants (match real machine startup/shutdown)
    STARTUP_PREAMBLE_MS = 8000   # 8 s startup animation
    SHUTDOWN_DURATION_MS = 4500  # 3 s shutdown + 1.5 s buffer

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._built_backend = None
        self._animation_controller = None
        self._anim_generation = 0
        self._simulation_state: str = "idle"

        self._build_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────
    #  UI Construction
    # ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        # Top: 3D Viewport
        self.viewport = Viewport3D()
        self.viewport.setMinimumHeight(350)
        splitter.addWidget(self.viewport)

        # Bottom: tabbed simulation controls + results
        bottom_tabs = QTabWidget()
        bottom_tabs.setMaximumHeight(380)

        self.sim_control = SimulationControlPanel()
        bottom_tabs.addTab(self.sim_control, "Simulation")

        self.results_panel = ResultsPanel()
        bottom_tabs.addTab(self.results_panel, "Results")

        splitter.addWidget(bottom_tabs)
        splitter.setSizes([600, 350])

    def _connect_signals(self):
        """Wire simulation control signals."""
        self.sim_control.run_requested.connect(self.run_simulation)
        self.sim_control.pause_requested.connect(self.pause_simulation)
        self.sim_control.stop_requested.connect(self.stop_simulation)

        self.sim_control.simulation_results_ready.connect(
            self.results_panel.set_results
        )
        self.sim_control.simulation_results_ready.connect(
            self._on_simulation_finished
        )

        self.sim_control.sim_time_updated.connect(self._on_sim_time_updated)
        self.sim_control.component_state_updated.connect(
            self._on_component_state_updated
        )

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    def build_system(self, assembly_params: Dict[str, Any]):
        """Build and display the classification assembly in the 3D viewport.

        Parameters
        ----------
        assembly_params : dict
            Parameters from the Assembly Config dialog (mode, geometry
            overrides, subsystem toggles, etc.).
        """
        from ..simulation_backend import SimulationConfig, SimulationBackend

        settings = self.sim_control.get_settings()
        p = assembly_params

        self.viewport.clear()
        self._stop_animation()

        if not p.get("enable_classification", True):
            self._built_backend = None
            return False

        config = SimulationConfig(
            assembly_data={},
            use_preclassification=p.get(
                "use_preclassification", settings.use_preclassification
            ),
            wheel_diameter=p.get("wheel_diameter", settings.wheel_diameter),
            wheel_rpm=p.get("wheel_rpm", settings.wheel_rpm),
            include_feed_system=p.get(
                "include_feed_system", settings.include_feed_system
            ),
            include_air_system=p.get(
                "include_air_system", settings.include_air_system
            ),
            include_exhaust=p.get("include_exhaust", settings.include_exhaust),
            venturi_inlet_diameter=p.get("venturi_inlet_diameter", 0.08),
            venturi_throat_ratio=p.get("venturi_throat_ratio", 0.5),
            zigzag_channel_width=p.get("zigzag_channel_width", 0.15),
            zigzag_channel_depth=p.get("zigzag_channel_depth", 0.25),
            zigzag_num_stages=p.get("zigzag_num_stages", 5),
            primary_cyclone_diameter=p.get("primary_cyclone_diameter", 0.30),
            secondary_cyclone_diameter=p.get("secondary_cyclone_diameter", 0.20),
            tertiary_cyclone_diameter=p.get("tertiary_cyclone_diameter", 0.12),
            device="cpu",
        )

        backend = SimulationBackend(config)
        backend._build_assembly_from_gui()

        vertices, indices = backend.get_mesh()
        if vertices is None or len(vertices) == 0:
            self._built_backend = None
            return False

        self.viewport.update_from_backend_mesh(vertices, indices)

        # Attach animation controller
        self._stop_animation()
        assembly_obj = (
            getattr(backend, "_complete_assembly", None)
            or getattr(backend, "_assembly", None)
        )
        if assembly_obj is not None:
            ctrl = self.viewport.build_with_animation(assembly_obj)
            if ctrl is not None:
                self._animation_controller = ctrl
                self._log("Animation: registered rotating components")

        self._built_backend = backend

        mode = (
            "Full System"
            if p.get("use_preclassification", True)
            else "Wheel-Only"
        )
        summary = backend.get_system_summary()
        self._log(
            f"Classifier built: {summary.get('mode', mode)} — "
            f"{len(vertices):,} vertices"
        )
        return True

    def sync_settings_from_params(self, params: Dict[str, Any]):
        """Push assembly dialog params into the simulation settings."""
        s = self.sim_control.get_settings()

        if "wheel_rpm" in params:
            s.wheel_rpm = params["wheel_rpm"]
        if "wheel_diameter" in params:
            s.wheel_diameter = params["wheel_diameter"]
        if "use_preclassification" in params:
            s.use_preclassification = params["use_preclassification"]
        if "include_feed_system" in params:
            s.include_feed_system = params["include_feed_system"]
        if "include_air_system" in params:
            s.include_air_system = params["include_air_system"]
        if "include_exhaust" in params:
            s.include_exhaust = params["include_exhaust"]
        if (
            "venturi_throat_ratio" in params
            and "venturi_inlet_diameter" in params
        ):
            throat_mm = (
                params["venturi_inlet_diameter"]
                * params["venturi_throat_ratio"]
                * 1000
            )
            s.venturi_throat_diameter_mm = throat_mm
        if "zigzag_channel_width" in params:
            s.zigzag_width_mm = params["zigzag_channel_width"] * 1000
        if "zigzag_channel_depth" in params:
            s.zigzag_depth_mm = params["zigzag_channel_depth"] * 1000

        self.sim_control.set_settings(s)

    # ──────────────────────────────────────────────────────────────
    #  Simulation Lifecycle
    # ──────────────────────────────────────────────────────────────

    @Slot()
    def run_simulation(self):
        """Start the classification simulation.

        Sequence: startup preamble animation → classification physics
        → shutdown animation.
        """
        if self._built_backend is None:
            self._log("No system built — cannot run simulation.")
            return

        self._anim_generation += 1
        self._simulation_state = "running"
        self.simulation_state_changed.emit("running")

        # Reset KPI cards
        self.sim_control.progress_bar.setValue(0)
        self.sim_control.card_time.set_value("0.000 s")
        self.sim_control.card_particles.set_value("0")
        self.sim_control.card_fines.set_value("0")
        self.sim_control.card_coarse.set_value("0")
        self.sim_control.card_efficiency.set_value("--")

        # Cinematic camera
        if self.viewport.cinematic_enabled:
            self.viewport.start_cinematic()

        # Start mechanical animation preamble
        self._start_animation()

        self._log(
            f"Startup preamble: {self.STARTUP_PREAMBLE_MS / 1000:.0f}s "
            "(air → feed → classification)..."
        )
        gen = self._anim_generation
        QTimer.singleShot(
            self.STARTUP_PREAMBLE_MS,
            lambda g=gen: self._start_simulation_after_preamble(g),
        )

    def _start_simulation_after_preamble(self, generation: int = -1):
        if generation >= 0 and generation != self._anim_generation:
            return
        if self._simulation_state != "running":
            return
        self._log("Preamble complete — starting classification physics")
        self.sim_control.start_simulation({})

    @Slot()
    def pause_simulation(self):
        self._simulation_state = "paused"
        self.simulation_state_changed.emit("paused")
        self.sim_control.pause_simulation()

    @Slot()
    def stop_simulation(self):
        self._simulation_state = "idle"
        self.simulation_state_changed.emit("idle")
        self._stop_animation()
        self.viewport.stop_cinematic()
        self.sim_control.stop_simulation()

    # ──────────────────────────────────────────────────────────────
    #  Animation
    # ──────────────────────────────────────────────────────────────

    def _start_animation(self):
        if self._animation_controller is None:
            return

        if self._built_backend is not None:
            try:
                subs = self._built_backend.create_subsidiary_simulators()
                air_s = subs.get("air_sim")
                feed_s = subs.get("feed_sim")
                if air_s or feed_s:
                    self._animation_controller.set_subsidiary_simulators(
                        air_s, feed_s
                    )
                    parts = []
                    if air_s:
                        parts.append("air (blower+dampers)")
                    if feed_s:
                        parts.append("feed (lid servo)")
                    self._log(f"Animation physics: {', '.join(parts)}")
            except Exception as e:
                self._log(f"Animation physics init skipped: {e}")

        from ..widgets.animation_controller import AnimationTimeline

        timeline = AnimationTimeline(
            air_start_time=0.0,
            air_ramp_duration=2.0,
            feed_start_time=3.0,
            feed_ramp_duration=2.0,
            classification_start_time=5.0,
            classification_ramp_duration=2.0,
            steady_time=8.0,
        )
        self._animation_controller.start(timeline)
        self._log("Animation started: Air → Feed → Classification")

    def _stop_animation(self):
        _SHUTDOWN_ANIM_S = 3.0
        _FORCE_STOP_MS = int(_SHUTDOWN_ANIM_S * 1000) + 1500

        if self._animation_controller is not None:
            phase = self._animation_controller.phase.value
            if phase in (
                "steady_state",
                "classification",
                "feed_startup",
                "air_startup",
            ):
                gen = self._anim_generation
                self._animation_controller.begin_shutdown(
                    duration=_SHUTDOWN_ANIM_S
                )
                QTimer.singleShot(
                    _FORCE_STOP_MS,
                    lambda g=gen: self._force_stop_animation(g),
                )
            else:
                self._animation_controller.stop()
                self._animation_controller.render_initial_state()

    def _force_stop_animation(self, generation: int = -1):
        if generation >= 0 and generation != self._anim_generation:
            return
        if self._animation_controller is not None:
            self._animation_controller.stop()
            self._animation_controller.render_initial_state()

    # ──────────────────────────────────────────────────────────────
    #  Internal signal handlers
    # ──────────────────────────────────────────────────────────────

    @Slot(float)
    def _on_sim_time_updated(self, sim_time: float):
        if self._animation_controller is not None:
            anim_time = sim_time + self.STARTUP_PREAMBLE_MS / 1000.0
            self._animation_controller.sync_to_sim_time(anim_time)

    @Slot(dict)
    def _on_component_state_updated(self, component_state: dict):
        if self._animation_controller is not None:
            preamble_s = self.STARTUP_PREAMBLE_MS / 1000.0
            component_state = dict(component_state)
            component_state["sim_time"] = (
                component_state.get("sim_time", 0.0) + preamble_s
            )
            self._animation_controller.update_from_physics(component_state)

    @Slot(dict)
    def _on_simulation_finished(self, results: Dict[str, Any]):
        """Classification physics done → play shutdown sequence."""
        self._log("Classification complete — shutting down system...")
        self._stop_animation()

        gen = self._anim_generation
        QTimer.singleShot(
            self.SHUTDOWN_DURATION_MS,
            lambda g=gen: self._on_shutdown_complete(g),
        )

    def _on_shutdown_complete(self, generation: int = -1):
        if generation >= 0 and generation != self._anim_generation:
            return
        self._simulation_state = "idle"
        self.simulation_state_changed.emit("idle")
        self.viewport.stop_cinematic()
        self._log("System shutdown complete — dampers closed, lid closed.")
        self.simulation_finished.emit(
            self.results_panel._results if self.results_panel._results else {}
        )

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.sim_control._log(msg)
        self.log_message.emit(msg)

    @property
    def is_built(self) -> bool:
        return self._built_backend is not None

    def cleanup(self):
        """Stop simulation and release resources."""
        if self._simulation_state == "running":
            self.stop_simulation()
