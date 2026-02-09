"""
Pretreatment GUI Panel — GP-15 RF Heating
==========================================

Full-quality simulation control panel for the GP-15 RF dielectric
heating machine.  Matches the UI/UX quality of the classifier
``SimulationControlPanel`` — uses the same theme, stat cards, tab
layout, settings forms, and log view.

Geometry: When a run completes, ``simulation_results_ready`` carries
meshes from ``GP15Simulator.get_mesh()``, which uses the assembled
machine geometry (geometry.machine.build_gp15_machine_meshes). The
main window adds those meshes to the 3D viewport with their color and
opacity, so the same oven, conveyor, electrode components and envelope
shown on Build are shown after a run.

Integration::

    from airclassifier.pretreatment.gui_panel import PretreatmentPanel

    self.pretreatment_panel = PretreatmentPanel()
    dock = QDockWidget("Pretreatment (GP-15)", self)
    dock.setWidget(self.pretreatment_panel)
    self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
"""

from __future__ import annotations

from typing import Optional, Dict, Any

try:
    from PySide6.QtCore import Qt, Signal, QThread, QObject
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSpinBox,
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
    # Import theme colors from the main GUI
    try:
        from ..gui.theme import COLORS
    except Exception:
        # Fallback if theme not available
        class _Colors:
            BG_DARKEST = "#141417"
            BG_DARKER = "#1a1a1f"
            BG_DARK = "#1e1e24"
            BG_BASE = "#242429"
            BG_ELEVATED = "#2a2a31"
            BG_SURFACE = "#303038"
            BG_HOVER = "#363640"
            BORDER = "#3a3a45"
            BORDER_SUBTLE = "#2f2f39"
            BORDER_FOCUS = "#4a9eff"
            TEXT_PRIMARY = "#e4e4eb"
            TEXT_SECONDARY = "#a0a0b0"
            TEXT_MUTED = "#6b6b7b"
            TEXT_DISABLED = "#505060"
            ACCENT = "#4a9eff"
            SUCCESS = "#3dd68c"
            WARNING = "#f0b429"
            DANGER = "#ef5350"
            INFO = "#4fc3f7"
        COLORS = _Colors()

    # ── Stat Card (matching SimulationControlPanel._StatCard) ─────

    class _StatCard(QFrame):
        """Themed KPI display card."""

        def __init__(self, title: str, accent: str = "", parent=None):
            super().__init__(parent)
            self.setStyleSheet(f"""
                _StatCard {{
                    background: {COLORS.BG_DARK};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 6px;
                }}
            """)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(10, 6, 10, 6)
            layout.setSpacing(2)

            self._value_label = QLabel("--")
            color = accent or COLORS.ACCENT
            self._value_label.setStyleSheet(
                f"font-size: 14pt; font-weight: 700; color: {color};"
            )
            self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(self._value_label)

            self._title_label = QLabel(title)
            self._title_label.setStyleSheet(
                f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            )
            self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(self._title_label)

        def set_value(self, text: str):
            self._value_label.setText(text)

    # ── Background Worker ────────────────────────────────────────

    class _PretreatmentWorker(QObject):
        """Runs the GP-15 simulation on a background QThread."""

        progress_updated = Signal(int, float, dict)
        simulation_completed = Signal(dict)
        simulation_error = Signal(str)
        log_message = Signal(str)

        def __init__(self, config, material, recipe, duration_s, settings):
            super().__init__()
            self._config = config
            self._material = material
            self._recipe = recipe
            self._duration_s = duration_s
            self._settings = settings
            self._running = True

        def run(self):
            try:
                from .simulator import GP15Simulator

                s = self._settings
                self.log_message.emit(
                    f"GP-15 initializing | gap={self._recipe.electrode_gap_mm:.0f} mm "
                    f"| speed={self._recipe.belt_speed_m_per_min:.2f} m/min "
                    f"| material={self._material.name}"
                )

                sim = GP15Simulator(
                    config=self._config,
                    material=self._material,
                    enable_controller=s.get("controller", True),
                    enable_corrections=s.get("corrections", True),
                    use_tvd=s.get("tvd", True),
                    use_fdm=s.get("fdm", False),
                    oscillator_efficiency=s.get("efficiency", 0.56),
                )
                sim.load_recipe(self._recipe)
                sim._ensure_initialized()

                dt = sim._sim.compute_stable_dt(self._recipe)
                total_steps = max(1, int(self._duration_s / dt))
                report_every = max(1, total_steps // 100)

                self.log_message.emit(
                    f"Running {total_steps} steps (dt={dt:.4f} s, "
                    f"total={self._duration_s:.0f} s)..."
                )

                for step_i in range(total_steps):
                    if not self._running:
                        self.log_message.emit("Stopped by user.")
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
                            "electrode_gap_mm": state.electrode_gap_mm,
                        })

                outlet = sim.get_outlet_conditions()
                meshes = sim.get_mesh()

                self.log_message.emit(
                    f"Complete | M_out={outlet.avg_moisture_wb:.1%} "
                    f"| T_out={outlet.avg_temperature_c:.1f} °C "
                    f"| CV={outlet.moisture_uniformity:.3f} "
                    f"| E={outlet.total_energy_kwh:.3f} kWh"
                )

                self.progress_updated.emit(100, sim._sim._time, {})
                self.simulation_completed.emit({
                    "outlet": outlet,
                    "meshes": meshes,
                    "time_series": {
                        "time_s": [s.time_s for s in sim._sim._history],
                        "T_mean_c": [s.T_mean_c for s in sim._sim._history],
                        "M_mean_wb": [s.M_mean_wb for s in sim._sim._history],
                        "rf_power_kw": [s.rf_power_kw for s in sim._sim._history],
                        "anode_current_a": [s.anode_current_a for s in sim._sim._history],
                    },
                })

            except Exception as e:
                import traceback
                self.simulation_error.emit(f"{e}\n{traceback.format_exc()}")

        def stop(self):
            self._running = False

    # ── Scrollable helper ────────────────────────────────────────

    def _scrollable(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    # ── Main Panel ───────────────────────────────────────────────

    class PretreatmentPanel(QWidget):
        """GP-15 RF Pretreatment simulation panel.

        Three tabs: Control, Settings, Log.  Matches the UI quality of
        SimulationControlPanel.
        """

        # Signals for MainWindow
        run_requested = Signal()
        stop_requested = Signal()
        simulation_results_ready = Signal(dict)
        sim_time_updated = Signal(float)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._worker: Optional[_PretreatmentWorker] = None
            self._thread: Optional[QThread] = None
            self._build_ui()

        # ── UI Construction ───────────────────────────────────────

        def _build_ui(self):
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(6, 6, 6, 6)
            main_layout.setSpacing(6)

            self._tabs = QTabWidget()
            main_layout.addWidget(self._tabs)

            self._tabs.addTab(self._build_control_tab(), "Control")
            self._tabs.addTab(_scrollable(self._build_settings_tab()), "Settings")
            self._tabs.addTab(self._build_log_tab(), "Log")

        def _build_control_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.setSpacing(8)

            # ── Buttons ───────────────────────────────────────────
            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(6)

            self._run_btn = QPushButton("Run")
            self._run_btn.setProperty("cssClass", "success")
            self._run_btn.setMinimumHeight(34)
            self._run_btn.clicked.connect(self._on_run)
            btn_layout.addWidget(self._run_btn)

            self._stop_btn = QPushButton("Stop")
            self._stop_btn.setProperty("cssClass", "danger")
            self._stop_btn.setMinimumHeight(34)
            self._stop_btn.setEnabled(False)
            self._stop_btn.clicked.connect(self._on_stop)
            btn_layout.addWidget(self._stop_btn)

            layout.addLayout(btn_layout)

            # ── Progress ──────────────────────────────────────────
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setTextVisible(True)
            self._progress.setFormat("%p%")
            self._progress.setFixedHeight(16)
            layout.addWidget(self._progress)

            # ── KPI cards ─────────────────────────────────────────
            cards = QGridLayout()
            cards.setSpacing(6)

            self._card_time = _StatCard("Simulation Time", COLORS.ACCENT)
            self._card_T = _StatCard("Temperature (mean)", COLORS.DANGER)
            self._card_M = _StatCard("Moisture (mean)", COLORS.INFO)
            self._card_P = _StatCard("RF Power", COLORS.WARNING)
            self._card_Ia = _StatCard("Anode Current", COLORS.SUCCESS)
            self._card_gap = _StatCard("Electrode Gap", COLORS.TEXT_SECONDARY)

            cards.addWidget(self._card_time, 0, 0)
            cards.addWidget(self._card_T, 0, 1)
            cards.addWidget(self._card_M, 1, 0)
            cards.addWidget(self._card_P, 1, 1)
            cards.addWidget(self._card_Ia, 2, 0)
            cards.addWidget(self._card_gap, 2, 1)

            layout.addLayout(cards)
            layout.addStretch()
            return w

        def _build_settings_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.setSpacing(8)

            # ── Material ──────────────────────────────────────────
            mat_grp = QGroupBox("Material")
            mat_form = QFormLayout()
            mat_form.setContentsMargins(10, 14, 10, 10)

            self._material_combo = QComboBox()
            self._material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
            mat_form.addRow("Preset:", self._material_combo)

            self._moisture_spin = QDoubleSpinBox()
            self._moisture_spin.setRange(0.01, 0.25)
            self._moisture_spin.setValue(0.10)
            self._moisture_spin.setDecimals(3)
            self._moisture_spin.setSingleStep(0.005)
            mat_form.addRow("Inlet moisture (wb):", self._moisture_spin)

            self._bed_depth_spin = QDoubleSpinBox()
            self._bed_depth_spin.setRange(10, 100)
            self._bed_depth_spin.setValue(40)
            self._bed_depth_spin.setSuffix(" mm")
            mat_form.addRow("Bed depth:", self._bed_depth_spin)

            mat_grp.setLayout(mat_form)
            layout.addWidget(mat_grp)

            # ── Recipe ────────────────────────────────────────────
            recipe_grp = QGroupBox("Recipe (GP-15 HMI)")
            recipe_form = QFormLayout()
            recipe_form.setContentsMargins(10, 14, 10, 10)

            self._gap_spin = QDoubleSpinBox()
            self._gap_spin.setRange(20, 300)
            self._gap_spin.setValue(80)
            self._gap_spin.setSuffix(" mm")
            recipe_form.addRow("Electrode gap:", self._gap_spin)

            self._speed_spin = QDoubleSpinBox()
            self._speed_spin.setRange(0.1, 2.0)
            self._speed_spin.setValue(0.50)
            self._speed_spin.setDecimals(2)
            self._speed_spin.setSingleStep(0.05)
            self._speed_spin.setSuffix(" m/min")
            recipe_form.addRow("Belt speed:", self._speed_spin)

            self._fan_spin = QDoubleSpinBox()
            self._fan_spin.setRange(5, 60)
            self._fan_spin.setValue(30)
            self._fan_spin.setSuffix(" Hz")
            recipe_form.addRow("Extraction fan:", self._fan_spin)

            self._mrh_spin = QDoubleSpinBox()
            self._mrh_spin.setRange(0.5, 3.0)
            self._mrh_spin.setValue(2.6)
            self._mrh_spin.setDecimals(2)
            self._mrh_spin.setSuffix(" A")
            recipe_form.addRow("MRH (overcurrent):", self._mrh_spin)

            self._heater_check = QCheckBox("Both banks on")
            self._heater_check.setChecked(True)
            recipe_form.addRow("Heaters:", self._heater_check)

            self._temp_check = QCheckBox("Enable")
            recipe_form.addRow("Temp control:", self._temp_check)

            self._temp_sp_spin = QDoubleSpinBox()
            self._temp_sp_spin.setRange(30, 100)
            self._temp_sp_spin.setValue(60)
            self._temp_sp_spin.setSuffix(" °C")
            recipe_form.addRow("Temp setpoint:", self._temp_sp_spin)

            recipe_grp.setLayout(recipe_form)
            layout.addWidget(recipe_grp)

            # ── Time ──────────────────────────────────────────────
            time_grp = QGroupBox("Simulation Time")
            time_form = QFormLayout()
            time_form.setContentsMargins(10, 14, 10, 10)

            self._duration_spin = QDoubleSpinBox()
            self._duration_spin.setRange(1, 3600)
            self._duration_spin.setValue(120)
            self._duration_spin.setSuffix(" s")
            time_form.addRow("Duration:", self._duration_spin)

            hint = QLabel("Residence time at 0.5 m/min ≈ 180 s")
            hint.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
            time_form.addRow("", hint)

            time_grp.setLayout(time_form)
            layout.addWidget(time_grp)

            # ── Physics ───────────────────────────────────────────
            phys_grp = QGroupBox("Physics Options")
            phys_form = QFormLayout()
            phys_form.setContentsMargins(10, 14, 10, 10)

            self._tvd_check = QCheckBox("Van Leer TVD advection")
            self._tvd_check.setChecked(True)
            phys_form.addRow(self._tvd_check)

            self._fdm_check = QCheckBox("FDM E-field (per-cell eps')")
            self._fdm_check.setChecked(False)
            phys_form.addRow(self._fdm_check)

            self._ctrl_check = QCheckBox("PLC controller (MRH/MRL)")
            self._ctrl_check.setChecked(True)
            phys_form.addRow(self._ctrl_check)

            self._corr_check = QCheckBox("Fringe + perforation corrections")
            self._corr_check.setChecked(True)
            phys_form.addRow(self._corr_check)

            self._eff_spin = QDoubleSpinBox()
            self._eff_spin.setRange(0.10, 1.0)
            self._eff_spin.setValue(0.56)
            self._eff_spin.setDecimals(2)
            self._eff_spin.setSingleStep(0.05)
            phys_form.addRow("Oscillator efficiency:", self._eff_spin)

            phys_grp.setLayout(phys_form)
            layout.addWidget(phys_grp)

            layout.addStretch()
            return w

        def _build_log_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)
            layout.setSpacing(6)

            self._log_text = QTextEdit()
            self._log_text.setReadOnly(True)
            self._log_text.setFont(QFont("Cascadia Code, Consolas, monospace", 9))
            self._log_text.setStyleSheet(f"""
                QTextEdit {{
                    background: {COLORS.BG_DARKEST};
                    color: {COLORS.TEXT_SECONDARY};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 4px;
                }}
            """)
            layout.addWidget(self._log_text)

            clear_btn = QPushButton("Clear")
            clear_btn.setProperty("cssClass", "ghost")
            clear_btn.clicked.connect(self._log_text.clear)
            layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

            return w

        # ── Public API ────────────────────────────────────────────

        def get_recipe(self) -> Recipe:
            """Build a Recipe from current UI values."""
            return Recipe(
                name="gui_recipe",
                recipe_number=0,
                electrode_gap_mm=self._gap_spin.value(),
                belt_speed_m_per_min=self._speed_spin.value(),
                extraction_fan_hz=self._fan_spin.value(),
                mrh_amps=self._mrh_spin.value(),
                heater_bank_1_on=self._heater_check.isChecked(),
                heater_bank_2_on=self._heater_check.isChecked(),
                temp_control_enabled=self._temp_check.isChecked(),
                temp_setpoint_c=self._temp_sp_spin.value(),
            )

        def get_material(self) -> MaterialProperties:
            """Build MaterialProperties from current UI values."""
            from .materials.presets import get_material_preset
            mat = get_material_preset(self._material_combo.currentText())
            mat.initial_moisture_wb = self._moisture_spin.value()
            mat.bed_depth_m = self._bed_depth_spin.value() / 1000.0
            return mat

        def get_settings(self) -> dict:
            """Return physics settings dict for the worker."""
            return {
                "tvd": self._tvd_check.isChecked(),
                "fdm": self._fdm_check.isChecked(),
                "controller": self._ctrl_check.isChecked(),
                "corrections": self._corr_check.isChecked(),
                "efficiency": self._eff_spin.value(),
            }

        def update_kpis(self, stats: dict):
            """Update KPI cards from a stats dict."""
            if "T_mean_c" in stats:
                self._card_T.set_value(f"{stats['T_mean_c']:.1f} °C")
            if "M_mean_wb" in stats:
                self._card_M.set_value(f"{stats['M_mean_wb'] * 100:.2f} %")
            if "rf_power_kw" in stats:
                self._card_P.set_value(f"{stats['rf_power_kw']:.2f} kW")
            if "anode_current_a" in stats:
                self._card_Ia.set_value(f"{stats['anode_current_a']:.2f} A")
            if "electrode_gap_mm" in stats:
                self._card_gap.set_value(f"{stats['electrode_gap_mm']:.0f} mm")

        # ── Simulation Lifecycle ──────────────────────────────────

        def start_simulation(self):
            """Launch the background simulation thread."""
            config = MachineConfig()
            material = self.get_material()
            recipe = self.get_recipe()
            duration = self._duration_spin.value()
            settings = self.get_settings()

            self._thread = QThread()
            self._worker = _PretreatmentWorker(
                config, material, recipe, duration, settings,
            )
            self._worker.moveToThread(self._thread)

            self._thread.started.connect(self._worker.run)
            self._worker.progress_updated.connect(self._on_progress)
            self._worker.simulation_completed.connect(self._on_completed)
            self._worker.simulation_error.connect(self._on_error)
            self._worker.log_message.connect(self._log)

            self._thread.start()
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._card_time.set_value("0.0 s")
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
            self._log("Stopped by user.")

        def _on_progress(self, pct: int, time_s: float, stats: dict):
            self._progress.setValue(pct)
            self._card_time.set_value(f"{time_s:.1f} s")
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
