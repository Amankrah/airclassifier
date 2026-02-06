"""
Simulation Control Panel
========================

Panel for controlling simulation execution and displaying progress.
Uses the real ClassificationFlowPhysicsSimulator from classification_flow_physics.py.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import traceback

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QTabWidget, QTextEdit,
    QGridLayout, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QObject
from PySide6.QtGui import QColor, QFont, QTextCursor

from ..theme import COLORS


# ============================================================================
# Settings dataclass
# ============================================================================

@dataclass
class SimulationSettings:
    """Settings for simulation execution."""
    # Time settings
    total_time: float = 10.0            # [s] Total simulation time
    dt: float = 0.001                   # [s] Time step
    output_interval: float = 0.1        # [s] Output interval

    # Particle settings
    num_particles: int = 5000           # Number of particles
    particle_feed_rate: float = 1000.0  # [particles/s]
    continuous_feeding: bool = True

    # Physics settings
    turbulence_intensity: float = 0.15
    restitution: float = 0.3
    friction: float = 0.4

    # Compute settings
    device: str = "cuda"                # "cuda" or "cpu"
    precision: str = "float32"          # "float32" or "float64"

    # Material selection
    material_source: str = "yellow_pea"
    material_fraction: str = "whole"

    # Visualization
    show_particles: bool = True
    show_velocity_field: bool = False
    particle_color_mode: str = "velocity"  # "velocity", "size", "type", "zone"

    # Assembly mode
    use_preclassification: bool = True  # True = venturi+zigzag+wheel, False = wheel-only

    # Wheel classifier parameters
    wheel_diameter: float = 0.20        # [m]
    wheel_rpm: float = 8000.0           # [RPM]

    # Air flow
    air_flow_m3s: float = 0.491         # [m3/s] default ~1768 m3/h

    # Complete system options
    include_feed_system: bool = True
    include_air_system: bool = True
    include_exhaust: bool = True


# ============================================================================
# Worker that uses the REAL simulation engine
# ============================================================================

class SimulationWorker(QObject):
    """
    Worker that runs ClassificationFlowPhysicsSimulator on a background thread.

    Emits progress at output_interval so the GUI stays responsive.
    """

    progress_updated = Signal(int, float, dict)  # (percent, sim_time, stats)
    simulation_completed = Signal(dict)           # (results)
    simulation_error = Signal(str)                # (error_message)
    log_message = Signal(str)

    def __init__(self, settings: SimulationSettings):
        super().__init__()
        self.settings = settings
        self._is_running = False
        self._is_paused = False

    def run(self):
        """Build assembly + config, then run the simulation loop."""
        try:
            self._is_running = True
            s = self.settings

            self.log_message.emit("Importing simulation engine...")

            from ..simulation_backend import SimulationBackend, SimulationConfig
            from ...simulation.classification_flow_physics import (
                ClassificationFlowPhysicsSimulator,
                ClassificationFlowConfig,
            )
            from ...geometry.assembly.classification import (
                ClassificationSystemAssembly,
                ClassificationSystemParams,
            )
            from ...particles import FluidConfig, ParticleMaterial

            # --- Build classification assembly ---
            self.log_message.emit("Building classification assembly...")
            params = ClassificationSystemParams()
            params.use_preclassification = s.use_preclassification
            assembly = ClassificationSystemAssembly(params=params)

            # --- Material ---
            material = None
            fluid = FluidConfig.air_at_stp()
            if s.material_source in ("yellow_pea", "faba_bean", "oat"):
                fraction = s.material_fraction if s.material_fraction != "whole" else "whole"
                material = ParticleMaterial.create_food_powder(s.material_source, fraction)
                self.log_message.emit(f"Material: {material.name} ({fraction})")
            elif s.material_source in ("protein", "starch", "fiber"):
                material = ParticleMaterial.create_food_powder(s.material_source, s.material_source)
                self.log_message.emit(f"Material: {material.name}")

            # --- Config ---
            config = ClassificationFlowConfig(
                num_particles=s.num_particles,
                air_flow_rate_m3s=s.air_flow_m3s,
                dt=s.dt,
                turbulent_intensity=s.turbulence_intensity,
                restitution=s.restitution,
                friction=s.friction,
                device=s.device,
                continuous_feeding=s.continuous_feeding,
                particle_feed_rate=s.particle_feed_rate,
                fluid_config=fluid,
                material=material,
                wheel_rpm=s.wheel_rpm if s.wheel_rpm else None,
            )

            self.log_message.emit(
                f"Config: {s.num_particles} particles, dt={s.dt}s, "
                f"Q={s.air_flow_m3s:.3f} m3/s, device={s.device}"
            )

            # --- Create simulator ---
            self.log_message.emit("Initializing Warp simulator...")
            sim = ClassificationFlowPhysicsSimulator(assembly, config)

            total_steps = int(s.total_time / s.dt)
            output_steps = max(1, int(s.output_interval / s.dt))

            self.log_message.emit(
                f"Running {total_steps:,} steps ({s.total_time}s at dt={s.dt}s)..."
            )

            import time as _time
            t_start = _time.perf_counter()

            for step in range(total_steps):
                if not self._is_running:
                    break

                while self._is_paused and self._is_running:
                    QThread.msleep(100)

                sim.step()

                if step % output_steps == 0:
                    progress = int(100 * step / total_steps)
                    sim_time = step * s.dt

                    counts = sim.get_separation_counts()
                    total_collected = (
                        counts.get("coarse", 0)
                        + counts.get("wheel_coarse", 0)
                        + counts.get("cyclone_1", 0)
                        + counts.get("cyclone_2", 0)
                        + counts.get("cyclone_3_protein", 0)
                        + counts.get("bagfilter", 0)
                    )
                    active = counts.get("active", 0)
                    fines = (
                        counts.get("cyclone_1", 0)
                        + counts.get("cyclone_2", 0)
                        + counts.get("cyclone_3_protein", 0)
                        + counts.get("bagfilter", 0)
                    )
                    coarse = counts.get("coarse", 0) + counts.get("wheel_coarse", 0)
                    eff = 0.0
                    if total_collected > 0:
                        eff = 100.0 * fines / total_collected

                    stats = {
                        "active_particles": active,
                        "collected_fines": fines,
                        "collected_coarse": coarse,
                        "separation_efficiency": eff,
                        **counts,
                    }
                    self.progress_updated.emit(progress, sim_time, stats)

            # --- Done ---
            elapsed = _time.perf_counter() - t_start
            counts = sim.get_separation_counts()
            fines = (
                counts.get("cyclone_1", 0)
                + counts.get("cyclone_2", 0)
                + counts.get("cyclone_3_protein", 0)
                + counts.get("bagfilter", 0)
            )
            coarse = counts.get("coarse", 0) + counts.get("wheel_coarse", 0)
            total_collected = fines + coarse
            eff = 100.0 * fines / total_collected if total_collected > 0 else 0.0

            results = {
                "total_time": s.total_time,
                "wall_time_s": elapsed,
                "particles_processed": total_collected,
                "separation_efficiency": eff,
                "fines_collected": fines,
                "coarse_collected": coarse,
                **counts,
            }
            self.simulation_completed.emit(results)

        except Exception as e:
            self.simulation_error.emit(f"{e}\n{traceback.format_exc()}")

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False


# ============================================================================
# Reusable KPI card
# ============================================================================

class _StatCard(QFrame):
    """Compact metric card showing a value with a label."""

    def __init__(self, label: str, initial_value: str = "--",
                 accent: str = COLORS.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self._value_label = QLabel(initial_value)
        self._value_label.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {accent};"
            " border: none; background: transparent;"
        )
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._value_label)

        title = QLabel(label)
        title.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

    def set_value(self, text: str):
        self._value_label.setText(text)


# ============================================================================
# Helper: wrap any widget in a QScrollArea
# ============================================================================

def _scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


# ============================================================================
# Main panel
# ============================================================================

class SimulationControlPanel(QWidget):
    """
    Panel for controlling simulation execution.

    Uses ClassificationFlowPhysicsSimulator for real physics.
    """

    run_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._settings = SimulationSettings()
        self._worker: Optional[SimulationWorker] = None
        self._thread: Optional[QThread] = None

        self._setup_ui()
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._create_control_tab(), "Control")
        tabs.addTab(_scrollable(self._create_settings_tab()), "Settings")
        tabs.addTab(self._create_log_tab(), "Log")

    # ---------------------------------------------------------------- control

    def _create_control_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("cssClass", "success")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.run_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(34)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        btn_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("cssClass", "danger")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        # KPI cards
        grid = QGridLayout()
        grid.setSpacing(6)

        self.card_time = _StatCard("Simulation Time", "0.000 s", COLORS.ACCENT)
        grid.addWidget(self.card_time, 0, 0)

        self.card_particles = _StatCard("Active Particles", "0", COLORS.INFO)
        grid.addWidget(self.card_particles, 0, 1)

        self.card_fines = _StatCard("Fines Collected", "0", COLORS.SUCCESS)
        grid.addWidget(self.card_fines, 1, 0)

        self.card_coarse = _StatCard("Coarse Collected", "0", COLORS.WARNING)
        grid.addWidget(self.card_coarse, 1, 1)

        self.card_efficiency = _StatCard("Separation Efficiency", "--", COLORS.CAT_CLASSIFICATION)
        grid.addWidget(self.card_efficiency, 2, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()
        return widget

    # --------------------------------------------------------------- settings

    def _create_settings_tab(self) -> QWidget:
        """Create the scrollable settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # --- Time ---
        time_group = QGroupBox("Time Settings")
        time_form = QFormLayout(time_group)
        time_form.setContentsMargins(10, 14, 10, 10)

        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setRange(0.1, 600.0)
        self.total_time_spin.setValue(self._settings.total_time)
        self.total_time_spin.setSuffix(" s")
        self.total_time_spin.valueChanged.connect(lambda v: setattr(self._settings, 'total_time', v))
        time_form.addRow("Total Time:", self.total_time_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(self._settings.dt)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.valueChanged.connect(lambda v: setattr(self._settings, 'dt', v))
        time_form.addRow("Time Step:", self.dt_spin)

        self.output_spin = QDoubleSpinBox()
        self.output_spin.setRange(0.01, 10.0)
        self.output_spin.setValue(self._settings.output_interval)
        self.output_spin.setSuffix(" s")
        self.output_spin.valueChanged.connect(lambda v: setattr(self._settings, 'output_interval', v))
        time_form.addRow("Output Interval:", self.output_spin)

        layout.addWidget(time_group)

        # --- Particles ---
        particle_group = QGroupBox("Particle Settings")
        particle_form = QFormLayout(particle_group)
        particle_form.setContentsMargins(10, 14, 10, 10)

        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(100, 500000)
        self.num_particles_spin.setSingleStep(1000)
        self.num_particles_spin.setValue(self._settings.num_particles)
        self.num_particles_spin.valueChanged.connect(lambda v: setattr(self._settings, 'num_particles', v))
        particle_form.addRow("Number of Particles:", self.num_particles_spin)

        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(10, 100000)
        self.feed_rate_spin.setValue(self._settings.particle_feed_rate)
        self.feed_rate_spin.setSuffix(" /s")
        self.feed_rate_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_feed_rate', v))
        particle_form.addRow("Feed Rate:", self.feed_rate_spin)

        self.continuous_check = QCheckBox("Enable")
        self.continuous_check.setChecked(self._settings.continuous_feeding)
        self.continuous_check.stateChanged.connect(
            lambda s: setattr(self._settings, 'continuous_feeding', s == Qt.CheckState.Checked.value)
        )
        particle_form.addRow("Continuous Feeding:", self.continuous_check)

        layout.addWidget(particle_group)

        # --- Material ---
        material_group = QGroupBox("Material")
        material_form = QFormLayout(material_group)
        material_form.setContentsMargins(10, 14, 10, 10)

        self.material_combo = QComboBox()
        self.material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        self.material_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_source', v))
        material_form.addRow("Source:", self.material_combo)

        self.fraction_combo = QComboBox()
        self.fraction_combo.addItems(["whole", "protein", "starch", "fiber"])
        self.fraction_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_fraction', v))
        material_form.addRow("Fraction:", self.fraction_combo)

        layout.addWidget(material_group)

        # --- Physics ---
        physics_group = QGroupBox("Physics")
        physics_form = QFormLayout(physics_group)
        physics_form.setContentsMargins(10, 14, 10, 10)

        self.turbulence_spin = QDoubleSpinBox()
        self.turbulence_spin.setRange(0.0, 0.5)
        self.turbulence_spin.setDecimals(2)
        self.turbulence_spin.setSingleStep(0.01)
        self.turbulence_spin.setValue(self._settings.turbulence_intensity)
        self.turbulence_spin.valueChanged.connect(lambda v: setattr(self._settings, 'turbulence_intensity', v))
        physics_form.addRow("Turbulence Intensity:", self.turbulence_spin)

        self.restitution_spin = QDoubleSpinBox()
        self.restitution_spin.setRange(0.0, 1.0)
        self.restitution_spin.setDecimals(2)
        self.restitution_spin.setValue(self._settings.restitution)
        self.restitution_spin.valueChanged.connect(lambda v: setattr(self._settings, 'restitution', v))
        physics_form.addRow("Restitution:", self.restitution_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setValue(self._settings.friction)
        self.friction_spin.valueChanged.connect(lambda v: setattr(self._settings, 'friction', v))
        physics_form.addRow("Friction:", self.friction_spin)

        layout.addWidget(physics_group)

        # --- Compute ---
        compute_group = QGroupBox("Compute")
        compute_form = QFormLayout(compute_group)
        compute_form.setContentsMargins(10, 14, 10, 10)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'device', v))
        compute_form.addRow("Device:", self.device_combo)

        layout.addWidget(compute_group)

        # --- Assembly Mode ---
        assembly_group = QGroupBox("Assembly Mode")
        assembly_form = QFormLayout(assembly_group)
        assembly_form.setContentsMargins(10, 14, 10, 10)

        self.assembly_mode_combo = QComboBox()
        self.assembly_mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.assembly_mode_combo.currentIndexChanged.connect(self._on_assembly_mode_changed)
        assembly_form.addRow("Mode:", self.assembly_mode_combo)

        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(1000, 20000)
        self.wheel_rpm_spin.setSingleStep(500)
        self.wheel_rpm_spin.setValue(self._settings.wheel_rpm)
        self.wheel_rpm_spin.setSuffix(" RPM")
        self.wheel_rpm_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_rpm', v))
        assembly_form.addRow("Wheel Speed:", self.wheel_rpm_spin)

        self.wheel_diameter_spin = QDoubleSpinBox()
        self.wheel_diameter_spin.setRange(0.05, 0.50)
        self.wheel_diameter_spin.setDecimals(3)
        self.wheel_diameter_spin.setSingleStep(0.01)
        self.wheel_diameter_spin.setValue(self._settings.wheel_diameter)
        self.wheel_diameter_spin.setSuffix(" m")
        self.wheel_diameter_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_diameter', v))
        assembly_form.addRow("Wheel Diameter:", self.wheel_diameter_spin)

        self.air_flow_spin = QDoubleSpinBox()
        self.air_flow_spin.setRange(0.01, 2.0)
        self.air_flow_spin.setDecimals(3)
        self.air_flow_spin.setSingleStep(0.01)
        self.air_flow_spin.setValue(self._settings.air_flow_m3s)
        self.air_flow_spin.setSuffix(" m\u00b3/s")
        self.air_flow_spin.valueChanged.connect(lambda v: setattr(self._settings, 'air_flow_m3s', v))
        assembly_form.addRow("Air Flow Rate:", self.air_flow_spin)

        layout.addWidget(assembly_group)

        return widget

    def _on_assembly_mode_changed(self, index: int):
        self._settings.use_preclassification = (index == 0)
        mode_name = "Full System" if index == 0 else "Wheel-Only"
        self._log(f"Assembly mode: {mode_name}")

    # ------------------------------------------------------------------- log

    def _create_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 9pt;
                background: {COLORS.BG_DARKEST};
                color: {COLORS.TEXT_SECONDARY};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setProperty("cssClass", "ghost")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        return widget

    # --------------------------------------------------------- button handlers

    def _on_run_clicked(self):
        if self._worker and self._worker._is_paused:
            self._worker.resume()
            self.run_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
        else:
            self.run_requested.emit()

    def _on_pause_clicked(self):
        if self._worker:
            self._worker.pause()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_requested.emit()

    def _on_stop_clicked(self):
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.stop_requested.emit()

    # ----------------------------------------------------------- simulation

    def start_simulation(self, assembly_data: Dict[str, Any]):
        """Start a new simulation using the real physics engine."""
        self._log("=" * 50)
        self._log("Starting ClassificationFlowPhysicsSimulator...")
        self._log(f"  Device:     {self._settings.device}")
        self._log(f"  Particles:  {self._settings.num_particles:,}")
        self._log(f"  Time:       {self._settings.total_time}s  (dt={self._settings.dt}s)")
        self._log(f"  Air flow:   {self._settings.air_flow_m3s:.3f} m\u00b3/s "
                   f"({self._settings.air_flow_m3s * 3600:.0f} m\u00b3/h)")
        self._log(f"  Material:   {self._settings.material_source} / {self._settings.material_fraction}")
        mode = "Full System" if self._settings.use_preclassification else "Wheel-Only"
        self._log(f"  Mode:       {mode}")
        self._log("=" * 50)

        # UI state
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Worker + thread
        self._thread = QThread()
        self._worker = SimulationWorker(self._settings)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.simulation_completed.connect(self._on_completed)
        self._worker.simulation_error.connect(self._on_error)
        self._worker.log_message.connect(self._log)

        self._thread.start()
        self._update_timer.start(200)

    def pause_simulation(self):
        if self._worker:
            self._worker.pause()
        self._log("Simulation paused")

    def stop_simulation(self):
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self._log("Simulation stopped")

    def _cleanup_thread(self):
        self._update_timer.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    @Slot(int, float, dict)
    def _on_progress(self, progress: int, sim_time: float, stats: Dict[str, Any]):
        self.progress_bar.setValue(progress)
        self.card_time.set_value(f"{sim_time:.3f} s")
        self.card_particles.set_value(f"{stats.get('active_particles', 0):,}")
        self.card_fines.set_value(f"{stats.get('collected_fines', 0):,}")
        self.card_coarse.set_value(f"{stats.get('collected_coarse', 0):,}")

        eff = stats.get("separation_efficiency", 0)
        if eff > 0:
            self.card_efficiency.set_value(f"{eff:.1f}%")

    @Slot(dict)
    def _on_completed(self, results: Dict[str, Any]):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        self._log("=" * 50)
        self._log("Simulation completed!")
        self._log(f"  Wall time:  {results.get('wall_time_s', 0):.1f}s")
        self._log(f"  Processed:  {results.get('particles_processed', 0):,}")
        self._log(f"  Fines:      {results.get('fines_collected', 0):,}")
        self._log(f"  Coarse:     {results.get('coarse_collected', 0):,}")
        self._log(f"  Efficiency: {results.get('separation_efficiency', 0):.1f}%")
        self._log("=" * 50)

    @Slot(str)
    def _on_error(self, error: str):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self._log(f"ERROR:\n{error}")

    def _update_display(self):
        pass

    def _log(self, message: str):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    # ----------------------------------------------------------- get/set

    def get_settings(self) -> SimulationSettings:
        return self._settings

    def set_settings(self, settings: SimulationSettings):
        self._settings = settings
        self.total_time_spin.setValue(settings.total_time)
        self.dt_spin.setValue(settings.dt)
        self.output_spin.setValue(settings.output_interval)
        self.num_particles_spin.setValue(settings.num_particles)
        self.feed_rate_spin.setValue(settings.particle_feed_rate)
        self.continuous_check.setChecked(settings.continuous_feeding)
        self.device_combo.setCurrentText(settings.device)
        self.material_combo.setCurrentText(settings.material_source)
        self.fraction_combo.setCurrentText(settings.material_fraction)
