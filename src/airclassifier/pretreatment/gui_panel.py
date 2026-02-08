"""
Pretreatment GUI Panel
======================

A PySide6 panel widget for the Air Classifier Designer that provides
recipe editing, run/pause/stop controls, real-time KPI cards, and
3D field visualization for the GP-15 RF pretreatment simulation.

Follows the same patterns as ``gui/panels/simulation_control.py``:
- Inherits from ``QWidget`` (not ``QDockWidget``)
- Uses ``QThread`` + worker for background simulation
- Emits signals consumed by ``MainWindow``
- Communicates with ``Viewport3D`` via mesh data

Integration in MainWindow._create_dock_widgets()::

    from airclassifier.pretreatment.gui_panel import PretreatmentPanel

    self.pretreatment_panel = PretreatmentPanel()
    pretreatment_dock = QDockWidget("Pretreatment", self)
    pretreatment_dock.setObjectName("PretreatmentDock")
    pretreatment_dock.setWidget(self.pretreatment_panel)
    self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, pretreatment_dock)
    self.tabifyDockWidget(sim_dock, pretreatment_dock)
    self._view_menu.addAction(pretreatment_dock.toggleViewAction())

Engineering guide §9.3.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

try:
    from PySide6.QtCore import Qt, Signal, QThread, QObject, QTimer
    from PySide6.QtWidgets import (
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QDoubleSpinBox,
        QCheckBox,
        QComboBox,
        QProgressBar,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

from .config import MachineConfig, MaterialProperties, Recipe
from .physics.coupling import StepState


if _HAS_PYSIDE6:

    # ── KPI card widget ──────────────────────────────────────────────

    class _StatCard(QFrame):
        """Small KPI display card matching the project's style."""

        def __init__(self, title: str, unit: str = "", parent=None):
            super().__init__(parent)
            self.setFrameShape(QFrame.Shape.StyledPanel)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(6, 4, 6, 4)

            self._title = QLabel(title)
            self._title.setStyleSheet("font-size: 10px; color: #888;")
            layout.addWidget(self._title)

            self._value = QLabel("--")
            self._value.setStyleSheet("font-size: 18px; font-weight: bold;")
            layout.addWidget(self._value)

            self._unit = unit

        def set_value(self, val: float, fmt: str = ".1f"):
            self._value.setText(f"{val:{fmt}} {self._unit}")

    # ── Background simulation worker ─────────────────────────────────

    class _PretreatmentWorker(QObject):
        """Runs the GP-15 simulation in a background thread."""

        progress_updated = Signal(int, float, dict)   # percent, time, stats
        simulation_completed = Signal(dict)            # results
        simulation_error = Signal(str)                 # error
        log_message = Signal(str)                      # log

        def __init__(self, config, material, recipe, duration_s):
            super().__init__()
            self._config = config
            self._material = material
            self._recipe = recipe
            self._duration_s = duration_s
            self._running = True

        def run(self):
            """Execute the simulation (runs in QThread)."""
            try:
                from .simulator import GP15Simulator

                self.log_message.emit("Initializing GP-15 simulator...")
                sim = GP15Simulator(
                    config=self._config,
                    material=self._material,
                    enable_controller=True,
                    enable_corrections=True,
                    use_tvd=True,
                )
                sim.load_recipe(self._recipe)

                self.log_message.emit(
                    f"Running for {self._duration_s:.0f} s "
                    f"(gap={self._recipe.electrode_gap_mm:.0f} mm, "
                    f"speed={self._recipe.belt_speed_m_per_min:.2f} m/min)..."
                )

                # Compute stable dt and total steps
                sim._ensure_initialized()
                dt = sim._sim.compute_stable_dt(self._recipe)
                total_steps = max(1, int(self._duration_s / dt))
                report_every = max(1, total_steps // 100)

                for step_i in range(total_steps):
                    if not self._running:
                        break

                    actual_dt = min(dt, self._duration_s - sim._sim._time)
                    if actual_dt <= 0:
                        break
                    state = sim.step(actual_dt)

                    if step_i % report_every == 0:
                        pct = int(100 * step_i / total_steps)
                        self.progress_updated.emit(pct, state.time_s, {
                            "T_mean_c": state.T_mean_c,
                            "T_max_c": state.T_max_c,
                            "M_mean_wb": state.M_mean_wb,
                            "rf_power_kw": state.rf_power_kw,
                            "anode_current_a": state.anode_current_a,
                        })

                outlet = sim.get_outlet_conditions()
                meshes = sim.get_mesh()

                results = {
                    "outlet": outlet,
                    "meshes": meshes,
                    "time_series": {
                        "time_s": [s.time_s for s in sim._sim._history],
                        "T_mean_c": [s.T_mean_c for s in sim._sim._history],
                        "M_mean_wb": [s.M_mean_wb for s in sim._sim._history],
                        "rf_power_kw": [s.rf_power_kw for s in sim._sim._history],
                    },
                }

                self.log_message.emit(
                    f"Complete. Outlet: M={outlet.avg_moisture_wb:.1%}, "
                    f"T={outlet.avg_temperature_c:.1f} °C"
                )
                self.progress_updated.emit(100, sim._sim._time, {})
                self.simulation_completed.emit(results)

            except Exception as e:
                self.simulation_error.emit(str(e))

        def stop(self):
            self._running = False

    # ── Main panel widget ────────────────────────────────────────────

    class PretreatmentPanel(QWidget):
        """Pretreatment simulation control panel.

        Provides recipe editing, simulation run controls, live KPI
        cards, and a log tab.  Follows the same architecture as
        ``SimulationControlPanel``.
        """

        # Signals consumed by MainWindow
        run_requested = Signal()
        stop_requested = Signal()
        simulation_results_ready = Signal(dict)
        sim_time_updated = Signal(float)
        mesh_updated = Signal(object, object)  # vertices, indices

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._worker: Optional[_PretreatmentWorker] = None
            self._thread: Optional[QThread] = None
            self._build_ui()

        def _build_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)

            self._tabs = QTabWidget()
            layout.addWidget(self._tabs)

            # ── Control tab ───────────────────────────────────────
            control_widget = QWidget()
            ctrl_layout = QVBoxLayout(control_widget)

            # Material preset
            mat_grp = QGroupBox("Material")
            mat_form = QFormLayout()
            self._material_combo = QComboBox()
            self._material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
            mat_form.addRow("Preset:", self._material_combo)

            self._moisture_spin = QDoubleSpinBox()
            self._moisture_spin.setRange(0.01, 0.25)
            self._moisture_spin.setValue(0.10)
            self._moisture_spin.setDecimals(3)
            self._moisture_spin.setSingleStep(0.01)
            mat_form.addRow("Inlet moisture:", self._moisture_spin)
            mat_grp.setLayout(mat_form)
            ctrl_layout.addWidget(mat_grp)

            # Recipe
            recipe_grp = QGroupBox("Recipe")
            recipe_form = QFormLayout()

            self._gap_spin = QDoubleSpinBox()
            self._gap_spin.setRange(20, 300)
            self._gap_spin.setValue(80)
            self._gap_spin.setSuffix(" mm")
            recipe_form.addRow("Electrode gap:", self._gap_spin)

            self._speed_spin = QDoubleSpinBox()
            self._speed_spin.setRange(0.1, 2.0)
            self._speed_spin.setValue(0.5)
            self._speed_spin.setDecimals(2)
            self._speed_spin.setSuffix(" m/min")
            recipe_form.addRow("Belt speed:", self._speed_spin)

            self._fan_spin = QDoubleSpinBox()
            self._fan_spin.setRange(5, 60)
            self._fan_spin.setValue(30)
            self._fan_spin.setSuffix(" Hz")
            recipe_form.addRow("Extraction fan:", self._fan_spin)

            self._temp_check = QCheckBox("Enable")
            recipe_form.addRow("Temp control:", self._temp_check)

            self._duration_spin = QDoubleSpinBox()
            self._duration_spin.setRange(1, 3600)
            self._duration_spin.setValue(120)
            self._duration_spin.setSuffix(" s")
            recipe_form.addRow("Duration:", self._duration_spin)

            recipe_grp.setLayout(recipe_form)
            ctrl_layout.addWidget(recipe_grp)

            # Buttons
            btn_layout = QHBoxLayout()
            self._run_btn = QPushButton("▶ Run")
            self._run_btn.clicked.connect(self._on_run)
            self._stop_btn = QPushButton("■ Stop")
            self._stop_btn.clicked.connect(self._on_stop)
            self._stop_btn.setEnabled(False)
            btn_layout.addWidget(self._run_btn)
            btn_layout.addWidget(self._stop_btn)
            ctrl_layout.addLayout(btn_layout)

            # Progress
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            ctrl_layout.addWidget(self._progress)

            # KPI cards
            kpi_grp = QGroupBox("Live KPIs")
            kpi_layout = QHBoxLayout()
            self._card_T = _StatCard("Temperature", "°C")
            self._card_M = _StatCard("Moisture", "%")
            self._card_P = _StatCard("RF Power", "kW")
            self._card_Ia = _StatCard("Anode I", "A")
            kpi_layout.addWidget(self._card_T)
            kpi_layout.addWidget(self._card_M)
            kpi_layout.addWidget(self._card_P)
            kpi_layout.addWidget(self._card_Ia)
            kpi_grp.setLayout(kpi_layout)
            ctrl_layout.addWidget(kpi_grp)

            ctrl_layout.addStretch()
            self._tabs.addTab(control_widget, "Control")

            # ── Log tab ───────────────────────────────────────────
            self._log_text = QTextEdit()
            self._log_text.setReadOnly(True)
            self._tabs.addTab(self._log_text, "Log")

        # ── Public API ────────────────────────────────────────────

        def get_recipe(self) -> Recipe:
            """Build a Recipe from current UI values."""
            return Recipe(
                name="gui_recipe",
                recipe_number=0,
                electrode_gap_mm=self._gap_spin.value(),
                belt_speed_m_per_min=self._speed_spin.value(),
                extraction_fan_hz=self._fan_spin.value(),
                temp_control_enabled=self._temp_check.isChecked(),
            )

        def get_material(self) -> MaterialProperties:
            """Build MaterialProperties from current UI values."""
            from .materials.presets import get_material_preset
            mat = get_material_preset(self._material_combo.currentText())
            mat.initial_moisture_wb = self._moisture_spin.value()
            return mat

        def update_kpis(self, stats: dict):
            """Update KPI cards from a stats dict."""
            if "T_mean_c" in stats:
                self._card_T.set_value(stats["T_mean_c"])
            if "M_mean_wb" in stats:
                self._card_M.set_value(stats["M_mean_wb"] * 100, ".2f")
            if "rf_power_kw" in stats:
                self._card_P.set_value(stats["rf_power_kw"], ".2f")
            if "anode_current_a" in stats:
                self._card_Ia.set_value(stats["anode_current_a"], ".2f")

        # ── Simulation lifecycle ──────────────────────────────────

        def start_simulation(self):
            """Launch the background simulation thread."""
            config = MachineConfig()
            material = self.get_material()
            recipe = self.get_recipe()
            duration = self._duration_spin.value()

            self._thread = QThread()
            self._worker = _PretreatmentWorker(config, material, recipe, duration)
            self._worker.moveToThread(self._thread)

            self._thread.started.connect(self._worker.run)
            self._worker.progress_updated.connect(self._on_progress)
            self._worker.simulation_completed.connect(self._on_completed)
            self._worker.simulation_error.connect(self._on_error)
            self._worker.log_message.connect(self._log)

            self._thread.start()
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._log("Simulation started.")

        def _on_run(self):
            self.run_requested.emit()
            self.start_simulation()

        def _on_stop(self):
            if self._worker:
                self._worker.stop()
            self.stop_requested.emit()
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._log("Simulation stopped by user.")

        def _on_progress(self, pct: int, time_s: float, stats: dict):
            self._progress.setValue(pct)
            self.sim_time_updated.emit(time_s)
            if stats:
                self.update_kpis(stats)

        def _on_completed(self, results: dict):
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._progress.setValue(100)
            self.simulation_results_ready.emit(results)

            if self._thread:
                self._thread.quit()
                self._thread.wait()

            self._log("Simulation complete.")

        def _on_error(self, msg: str):
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._log(f"ERROR: {msg}")

            if self._thread:
                self._thread.quit()
                self._thread.wait()

        def _log(self, msg: str):
            self._log_text.append(msg)

else:
    class PretreatmentPanel:  # type: ignore[no-redef]
        """Placeholder — PySide6 not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PySide6 is required for the pretreatment GUI panel. "
                "Install with: pip install PySide6"
            )
