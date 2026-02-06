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
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QObject
from PySide6.QtGui import QColor


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
        # This would initialize the Warp simulation
        pass

    def _step_simulation(self):
        """Execute one simulation step."""
        # This would call the Warp kernels
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
        """Pause the simulation."""
        self._is_paused = True

    def resume(self):
        """Resume the simulation."""
        self._is_paused = False

    def stop(self):
        """Stop the simulation."""
        self._is_running = False
        self._is_paused = False


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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

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

    def _create_control_tab(self) -> QWidget:
        """Create the simulation control tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        self.run_btn.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self.run_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(36)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        button_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336;")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QFormLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addRow("Overall:", self.progress_bar)

        self.time_label = QLabel("0.000 s")
        progress_layout.addRow("Simulation Time:", self.time_label)

        self.step_label = QLabel("0")
        progress_layout.addRow("Step:", self.step_label)

        self.fps_label = QLabel("-- fps")
        progress_layout.addRow("Performance:", self.fps_label)

        layout.addWidget(progress_group)

        # Live statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)

        self.active_particles_label = QLabel("0")
        stats_layout.addRow("Active Particles:", self.active_particles_label)

        self.fines_collected_label = QLabel("0")
        stats_layout.addRow("Fines Collected:", self.fines_collected_label)

        self.coarse_collected_label = QLabel("0")
        stats_layout.addRow("Coarse Collected:", self.coarse_collected_label)

        self.efficiency_label = QLabel("--")
        stats_layout.addRow("Separation Efficiency:", self.efficiency_label)

        layout.addWidget(stats_group)

        layout.addStretch()
        return widget

    def _create_settings_tab(self) -> QWidget:
        """Create the simulation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

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

    def _create_log_tab(self) -> QWidget:
        """Create the simulation log tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10px;")
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        return widget

    def _on_run_clicked(self):
        """Handle run button click."""
        if self._worker and self._worker._is_paused:
            # Resume
            self._worker.resume()
            self.run_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
        else:
            # Start new simulation
            self.run_requested.emit()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        if self._worker:
            self._worker.pause()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_requested.emit()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.stop_requested.emit()

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
        """Pause the running simulation."""
        if self._worker:
            self._worker.pause()
        self._log("Simulation paused")

    def stop_simulation(self):
        """Stop the simulation."""
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self._log("Simulation stopped")

    def _cleanup_thread(self):
        """Clean up worker thread."""
        self._update_timer.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    @Slot(int, float, dict)
    def _on_progress(self, progress: int, time: float, stats: Dict[str, Any]):
        """Handle progress update from worker."""
        self.progress_bar.setValue(progress)
        self.time_label.setText(f"{time:.3f} s")
        self.step_label.setText(str(int(time / self._settings.dt)))

        # Update statistics
        self.active_particles_label.setText(str(stats.get("active_particles", 0)))
        self.fines_collected_label.setText(str(stats.get("collected_fines", 0)))
        self.coarse_collected_label.setText(str(stats.get("collected_coarse", 0)))

        eff = stats.get("separation_efficiency", 0)
        if eff > 0:
            self.efficiency_label.setText(f"{eff:.1f}%")

    @Slot(dict)
    def _on_completed(self, results: Dict[str, Any]):
        """Handle simulation completion."""
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        self._log("Simulation completed!")
        self._log(f"  Final efficiency: {results.get('separation_efficiency', 0):.1f}%")

    @Slot(str)
    def _on_error(self, error: str):
        """Handle simulation error."""
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self._log(f"ERROR: {error}")

    def _update_display(self):
        """Update display during simulation."""
        # Update FPS estimate
        pass

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    def get_settings(self) -> SimulationSettings:
        """Get current simulation settings."""
        return self._settings

    def set_settings(self, settings: SimulationSettings):
        """Set simulation settings."""
        self._settings = settings

        # Update UI
        self.total_time_spin.setValue(settings.total_time)
        self.dt_spin.setValue(settings.dt)
        self.output_spin.setValue(settings.output_interval)
        self.num_particles_spin.setValue(settings.num_particles)
        self.feed_rate_spin.setValue(settings.particle_feed_rate)
        self.continuous_check.setChecked(settings.continuous_feeding)
        self.device_combo.setCurrentText(settings.device)
        self.material_combo.setCurrentText(settings.material_source)
        self.fraction_combo.setCurrentText(settings.material_fraction)
