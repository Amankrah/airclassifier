"""
Simulation Control Panel
========================

Panel for controlling simulation execution and displaying progress.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QTabWidget, QTextEdit,
    QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QObject
from PySide6.QtGui import QColor, QFont, QTextCursor

from ..theme import COLORS


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

    # Complete system options
    include_feed_system: bool = True
    include_air_system: bool = True
    include_exhaust: bool = True


class SimulationWorker(QObject):
    """Worker object for running simulation in background thread."""

    progress_updated = Signal(int, float, dict)  # (step, time, stats)
    simulation_completed = Signal(dict)           # (results)
    simulation_error = Signal(str)                # (error_message)

    def __init__(self, assembly_data: Dict[str, Any], settings: SimulationSettings):
        super().__init__()
        self.assembly_data = assembly_data
        self.settings = settings
        self._is_running = False
        self._is_paused = False

    def run(self):
        """Run the simulation."""
        try:
            self._is_running = True
            self._setup_simulation()

            total_steps = int(self.settings.total_time / self.settings.dt)
            output_steps = int(self.settings.output_interval / self.settings.dt)

            for step in range(total_steps):
                if not self._is_running:
                    break

                while self._is_paused and self._is_running:
                    QThread.msleep(100)

                # Run simulation step
                self._step_simulation()

                # Report progress
                if step % output_steps == 0:
                    progress = int(100 * step / total_steps)
                    current_time = step * self.settings.dt
                    stats = self._get_stats()
                    self.progress_updated.emit(progress, current_time, stats)

            # Finalize
            results = self._finalize_simulation()
            self.simulation_completed.emit(results)

        except Exception as e:
            self.simulation_error.emit(str(e))

    def _setup_simulation(self):
        """Setup simulation from assembly data."""
        pass

    def _step_simulation(self):
        """Execute one simulation step."""
        pass

    def _get_stats(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        return {
            "active_particles": 0,
            "collected_fines": 0,
            "collected_coarse": 0,
            "separation_efficiency": 0.0,
        }

    def _finalize_simulation(self) -> Dict[str, Any]:
        """Finalize and return results."""
        return {
            "total_time": self.settings.total_time,
            "particles_processed": 0,
            "separation_efficiency": 0.0,
            "grade_efficiency_curve": [],
        }

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False


# --------------------------------------------------------------------------
# Reusable KPI / stat card widget
# --------------------------------------------------------------------------

class _StatCard(QFrame):
    """A compact metric card showing a value with a label."""

    def __init__(self, label: str, initial_value: str = "--", accent: str = COLORS.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self._value_label = QLabel(initial_value)
        self._value_label.setStyleSheet(f"font-size: 16pt; font-weight: 700; color: {accent}; border: none; background: transparent;")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._value_label)

        title = QLabel(label)
        title.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; border: none; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

    def set_value(self, text: str):
        self._value_label.setText(text)


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

class SimulationControlPanel(QWidget):
    """
    Panel for controlling simulation execution.

    Provides controls for:
    - Start/pause/stop simulation
    - Configure simulation parameters
    - Monitor progress and statistics
    """

    # Signals
    run_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)  # SimulationSettings

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._settings = SimulationSettings()
        self._worker: Optional[SimulationWorker] = None
        self._thread: Optional[QThread] = None

        self._setup_ui()
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Control tab
        control_tab = self._create_control_tab()
        tabs.addTab(control_tab, "Control")

        # Settings tab
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "Settings")

        # Log tab
        log_tab = self._create_log_tab()
        tabs.addTab(log_tab, "Log")

    # ---------------------------------------------------------------- control

    def _create_control_tab(self) -> QWidget:
        """Create the simulation control tab with dashboard cards."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # --- button row ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(6)

        self.run_btn = QPushButton("  Run")
        self.run_btn.setProperty("cssClass", "success")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self.run_btn)

        self.pause_btn = QPushButton("  Pause")
        self.pause_btn.setMinimumHeight(34)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        button_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("  Stop")
        self.stop_btn.setProperty("cssClass", "danger")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # --- progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        # --- KPI cards grid ---
        cards_grid = QGridLayout()
        cards_grid.setSpacing(6)

        self.card_time = _StatCard("Simulation Time", "0.000 s", COLORS.ACCENT)
        cards_grid.addWidget(self.card_time, 0, 0)

        self.card_particles = _StatCard("Active Particles", "0", COLORS.INFO)
        cards_grid.addWidget(self.card_particles, 0, 1)

        self.card_fines = _StatCard("Fines Collected", "0", COLORS.SUCCESS)
        cards_grid.addWidget(self.card_fines, 1, 0)

        self.card_coarse = _StatCard("Coarse Collected", "0", COLORS.WARNING)
        cards_grid.addWidget(self.card_coarse, 1, 1)

        self.card_efficiency = _StatCard("Separation Efficiency", "--", COLORS.CAT_CLASSIFICATION)
        cards_grid.addWidget(self.card_efficiency, 2, 0, 1, 2)

        layout.addLayout(cards_grid)

        layout.addStretch()
        return widget

    # --------------------------------------------------------------- settings

    def _create_settings_tab(self) -> QWidget:
        """Create the simulation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Time settings
        time_group = QGroupBox("Time Settings")
        time_layout = QFormLayout(time_group)

        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setRange(0.1, 600.0)
        self.total_time_spin.setValue(self._settings.total_time)
        self.total_time_spin.setSuffix(" s")
        self.total_time_spin.valueChanged.connect(lambda v: setattr(self._settings, 'total_time', v))
        time_layout.addRow("Total Time:", self.total_time_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(self._settings.dt)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.valueChanged.connect(lambda v: setattr(self._settings, 'dt', v))
        time_layout.addRow("Time Step:", self.dt_spin)

        self.output_spin = QDoubleSpinBox()
        self.output_spin.setRange(0.01, 10.0)
        self.output_spin.setValue(self._settings.output_interval)
        self.output_spin.setSuffix(" s")
        self.output_spin.valueChanged.connect(lambda v: setattr(self._settings, 'output_interval', v))
        time_layout.addRow("Output Interval:", self.output_spin)

        layout.addWidget(time_group)

        # Particle settings
        particle_group = QGroupBox("Particle Settings")
        particle_layout = QFormLayout(particle_group)

        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(100, 100000)
        self.num_particles_spin.setValue(self._settings.num_particles)
        self.num_particles_spin.valueChanged.connect(lambda v: setattr(self._settings, 'num_particles', v))
        particle_layout.addRow("Number of Particles:", self.num_particles_spin)

        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(10, 10000)
        self.feed_rate_spin.setValue(self._settings.particle_feed_rate)
        self.feed_rate_spin.setSuffix(" /s")
        self.feed_rate_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_feed_rate', v))
        particle_layout.addRow("Feed Rate:", self.feed_rate_spin)

        self.continuous_check = QCheckBox()
        self.continuous_check.setChecked(self._settings.continuous_feeding)
        self.continuous_check.stateChanged.connect(lambda s: setattr(self._settings, 'continuous_feeding', s == Qt.CheckState.Checked.value))
        particle_layout.addRow("Continuous Feeding:", self.continuous_check)

        layout.addWidget(particle_group)

        # Material settings
        material_group = QGroupBox("Material")
        material_layout = QFormLayout(material_group)

        self.material_combo = QComboBox()
        self.material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        self.material_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_source', v))
        material_layout.addRow("Source:", self.material_combo)

        self.fraction_combo = QComboBox()
        self.fraction_combo.addItems(["whole", "protein", "starch", "fiber"])
        self.fraction_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_fraction', v))
        material_layout.addRow("Fraction:", self.fraction_combo)

        layout.addWidget(material_group)

        # Compute settings
        compute_group = QGroupBox("Compute")
        compute_layout = QFormLayout(compute_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'device', v))
        compute_layout.addRow("Device:", self.device_combo)

        layout.addWidget(compute_group)

        # Assembly mode settings
        assembly_group = QGroupBox("Assembly Mode")
        assembly_layout = QFormLayout(assembly_group)

        self.assembly_mode_combo = QComboBox()
        self.assembly_mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.assembly_mode_combo.currentIndexChanged.connect(self._on_assembly_mode_changed)
        assembly_layout.addRow("Mode:", self.assembly_mode_combo)

        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(1000, 20000)
        self.wheel_rpm_spin.setValue(self._settings.wheel_rpm)
        self.wheel_rpm_spin.setSuffix(" RPM")
        self.wheel_rpm_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_rpm', v))
        assembly_layout.addRow("Wheel Speed:", self.wheel_rpm_spin)

        self.wheel_diameter_spin = QDoubleSpinBox()
        self.wheel_diameter_spin.setRange(0.05, 0.50)
        self.wheel_diameter_spin.setDecimals(3)
        self.wheel_diameter_spin.setValue(self._settings.wheel_diameter)
        self.wheel_diameter_spin.setSuffix(" m")
        self.wheel_diameter_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_diameter', v))
        assembly_layout.addRow("Wheel Diameter:", self.wheel_diameter_spin)

        layout.addWidget(assembly_group)

        layout.addStretch()
        return widget

    def _on_assembly_mode_changed(self, index: int):
        """Handle assembly mode change."""
        self._settings.use_preclassification = (index == 0)
        mode_name = "Full System" if index == 0 else "Wheel-Only"
        self._log(f"Assembly mode: {mode_name}")

    # ------------------------------------------------------------------- log

    def _create_log_tab(self) -> QWidget:
        """Create the simulation log tab."""
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

    # ----------------------------------------------------------- button handlers

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
        """Start a new simulation."""
        self._log("Starting simulation...")
        self._log(f"  Device: {self._settings.device}")
        self._log(f"  Particles: {self._settings.num_particles}")
        self._log(f"  Time: {self._settings.total_time}s")

        # Update UI
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Create worker and thread
        self._thread = QThread()
        self._worker = SimulationWorker(assembly_data, self._settings)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.simulation_completed.connect(self._on_completed)
        self._worker.simulation_error.connect(self._on_error)

        # Start
        self._thread.start()
        self._update_timer.start(100)

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
            self._thread.wait()
            self._thread = None
            self._worker = None

    @Slot(int, float, dict)
    def _on_progress(self, progress: int, time: float, stats: Dict[str, Any]):
        self.progress_bar.setValue(progress)
        self.card_time.set_value(f"{time:.3f} s")
        self.card_particles.set_value(str(stats.get("active_particles", 0)))
        self.card_fines.set_value(str(stats.get("collected_fines", 0)))
        self.card_coarse.set_value(str(stats.get("collected_coarse", 0)))

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

        self._log("Simulation completed!")
        self._log(f"  Final efficiency: {results.get('separation_efficiency', 0):.1f}%")

    @Slot(str)
    def _on_error(self, error: str):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self._log(f"ERROR: {error}")

    def _update_display(self):
        pass

    def _log(self, message: str):
        self.log_text.append(message)
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

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
