"""
Simulation Control Panel
========================

Panel for controlling simulation execution and displaying progress.
Uses the real ClassificationFlowPhysicsSimulator from classification_flow_physics.py.

Settings are aligned with ClassificationFlowConfig and run_classification_flow.py
defaults so the GUI produces the same results as the CLI example.
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
# Settings dataclass -- aligned with ClassificationFlowConfig + CLI defaults
# ============================================================================

@dataclass
class SimulationSettings:
    """
    Settings for simulation execution.

    Defaults match ``run_classification_flow.py`` and ``ClassificationFlowConfig``
    so the GUI produces identical results to the CLI.
    """

    # --- Time (CLI: --time 360, --dt 0.001) ---
    total_time: float = 360.0           # [s]  CLI default 360
    dt: float = 0.001                   # [s]  ClassificationFlowConfig default
    output_interval: float = 1.0        # [s]  GUI refresh rate (every 1000 steps)

    # --- Particles (CLI: --particles 100000) ---
    num_particles: int = 5000           # GUI-friendly default (CLI is 100k)
    particle_feed_rate: float = 0.0     # [/s]  0 = auto (engine computes from mass flow)
    continuous_feeding: bool = True      # CLI --full-system default

    # --- Particle size (CLI: --particle-dia 50, --particle-std 30) ---
    particle_diameter_um: float = 50.0  # [um]  mean diameter when no material preset
    particle_std_um: float = 30.0       # [um]  std dev when no material preset

    # --- Physics (ClassificationFlowConfig defaults) ---
    turbulence_intensity: float = 0.15  # CLI --turbulence 0.15
    restitution: float = 0.3            # ClassificationFlowConfig default
    friction: float = 0.4               # ClassificationFlowConfig default
    bypass_ratio: float = 0.0           # CLI --bypass-ratio 0.0
    max_loading_ratio: float = 2.0      # CLI --max-loading 2.0

    # --- Compute (CLI: --device cuda) ---
    device: str = "cuda"
    precision: str = "float32"

    # --- Material (CLI: --material yellow_pea) ---
    material_source: str = "yellow_pea" # yellow_pea, faba_bean, oat, or "none"
    material_fraction: str = "whole"    # whole, protein, starch, fiber

    # --- Visualization ---
    show_particles: bool = True
    show_velocity_field: bool = False
    particle_color_mode: str = "velocity"

    # --- Assembly mode (CLI: default preclassification, --wheel-only) ---
    use_preclassification: bool = True

    # --- Wheel classifier (CLI: --wheel-rpm defaults from geometry ~8000) ---
    wheel_diameter: float = 0.20        # [m]
    wheel_rpm: float = 8000.0           # [RPM]

    # --- Air flow (CLI default: 1768 m³/h ≈ 0.491 m³/s) ---
    air_flow_m3s: float = 0.491         # [m³/s]

    # --- Complete system (used by Build Full System) ---
    include_feed_system: bool = True
    include_air_system: bool = True
    include_exhaust: bool = True


# ============================================================================
# Worker -- uses the REAL ClassificationFlowPhysicsSimulator
# ============================================================================

class SimulationWorker(QObject):
    """
    Worker that runs ClassificationFlowPhysicsSimulator on a background thread.

    Follows the same orchestration as ``run_classification_flow.py``:
    1. Build ClassificationSystemAssembly
    2. Create ClassificationFlowConfig
    3. Create ClassificationFlowPhysicsSimulator
    4. Initialize particles (whole flour / material / generic)
    5. Step loop with progress reporting
    """

    progress_updated = Signal(int, float, dict)  # (percent, sim_time, stats)
    simulation_completed = Signal(dict)           # final results
    simulation_error = Signal(str)                # error + traceback
    log_message = Signal(str)

    def __init__(self, settings: SimulationSettings):
        super().__init__()
        self.settings = settings
        self._is_running = False
        self._is_paused = False

    # ----------------------------------------------------------------

    def run(self):
        """Build assembly, init particles, run simulation -- matching CLI flow."""
        try:
            self._is_running = True
            s = self.settings

            self.log_message.emit("Importing simulation engine...")

            from ...simulation.classification_flow_physics import (
                ClassificationFlowPhysicsSimulator,
                ClassificationFlowConfig,
            )
            from ...geometry.assembly.classification import (
                ClassificationSystemAssembly,
                ClassificationSystemParams,
            )
            from ...particles import FluidConfig, ParticleMaterial

            # ==============================================================
            # 1. Build classification assembly
            # ==============================================================
            self.log_message.emit("Building classification assembly...")
            params = ClassificationSystemParams()
            params.use_preclassification = s.use_preclassification
            assembly = ClassificationSystemAssembly(params=params)

            mode_str = "Full System" if s.use_preclassification else "Wheel-Only"
            self.log_message.emit(f"  Mode: {mode_str}")
            if assembly.venturi is not None:
                self.log_message.emit("  Components: Venturi + Zigzag + Wheel + Cyclones + Bag")
            else:
                self.log_message.emit("  Components: Junction + Wheel + Cyclones + Bag")

            # ==============================================================
            # 2. Material + FluidConfig
            # ==============================================================
            material = None
            fluid = FluidConfig.air_at_stp()
            use_material = s.material_source not in ("none", "")

            if use_material:
                if s.material_source in ("yellow_pea", "faba_bean", "oat"):
                    fraction = s.material_fraction if s.material_fraction != "whole" else "whole"
                    material = ParticleMaterial.create_food_powder(s.material_source, fraction)
                elif s.material_source in ("protein", "starch", "fiber"):
                    material = ParticleMaterial.create_food_powder("yellow_pea", s.material_source)
                if material:
                    self.log_message.emit(f"  Material: {material.name}")

            # ==============================================================
            # 3. Build ClassificationFlowConfig (same fields as CLI)
            # ==============================================================
            config = ClassificationFlowConfig(
                num_particles=s.num_particles,
                air_flow_rate_m3s=s.air_flow_m3s,
                bypass_ratio=s.bypass_ratio,
                dt=s.dt,
                turbulent_intensity=s.turbulence_intensity,
                restitution=s.restitution,
                friction=s.friction,
                device=s.device,
                continuous_feeding=s.continuous_feeding,
                particle_feed_rate=s.particle_feed_rate,
                max_loading_ratio=s.max_loading_ratio,
                fluid_config=fluid,
                material=material,
                wheel_rpm=s.wheel_rpm,
            )

            self.log_message.emit(
                f"  Particles: {s.num_particles:,}   dt={s.dt}s   "
                f"Q={s.air_flow_m3s:.3f} m\u00b3/s ({s.air_flow_m3s * 3600:.0f} m\u00b3/h)"
            )
            self.log_message.emit(
                f"  Device: {s.device}   Wheel: {s.wheel_rpm:.0f} RPM   "
                f"Bypass: {s.bypass_ratio:.0%}"
            )

            # ==============================================================
            # 4. Create simulator
            # ==============================================================
            self.log_message.emit("Initializing Warp simulator...")
            sim = ClassificationFlowPhysicsSimulator(assembly, config)

            # ==============================================================
            # 5. Initialize particles (critical -- same as run_classification_flow.py)
            # ==============================================================
            if s.use_preclassification:
                self.log_message.emit("Initializing particles at venturi solids inlet...")
            else:
                self.log_message.emit("Initializing particles at wheel inlet (15\u00b0 solids chute)...")

            if use_material and material is not None:
                if s.material_source in ("yellow_pea", "faba_bean", "oat") and s.material_fraction == "whole":
                    # Whole flour population (protein + starch + fiber)
                    sim.initialize_whole_flour_population(
                        source=s.material_source,
                        num_particles=s.num_particles,
                        initial_velocity=(0.0, 0.5, 0.0),
                    )
                    self.log_message.emit(f"  Whole flour population: {s.material_source}")
                else:
                    # Single fraction via material
                    sim.initialize_particles_from_material(
                        material=material,
                        num_particles=s.num_particles,
                        initial_velocity=(0.0, 0.5, 0.0),
                    )
                    self.log_message.emit(f"  Material population: {material.name}")
            else:
                # Generic particles from diameter/std
                mean_dia_m = s.particle_diameter_um * 1e-6
                std_dia_m = s.particle_std_um * 1e-6
                sim.initialize_particles(
                    num_particles=s.num_particles,
                    mean_diameter=mean_dia_m,
                    diameter_std=std_dia_m,
                    initial_velocity=(0.0, 0.5, 0.0),
                )
                self.log_message.emit(
                    f"  Generic particles: d={s.particle_diameter_um:.0f} \u00b1 "
                    f"{s.particle_std_um:.0f} \u00b5m"
                )

            # ==============================================================
            # 6. Step loop
            # ==============================================================
            total_steps = int(s.total_time / s.dt)
            output_steps = max(1, int(s.output_interval / s.dt))

            self.log_message.emit(
                f"Running {total_steps:,} steps ({s.total_time:.0f}s)..."
            )

            import time as _time
            t_start = _time.perf_counter()

            for step in range(total_steps):
                if not self._is_running:
                    break

                while self._is_paused and self._is_running:
                    QThread.msleep(100)

                sim.step()

                # Progress report at output_interval
                if step > 0 and step % output_steps == 0:
                    progress = int(100 * step / total_steps)
                    sim_time = step * s.dt

                    counts = sim.get_separation_counts()
                    fines = (
                        counts.get("cyclone_1", 0)
                        + counts.get("cyclone_2", 0)
                        + counts.get("cyclone_3_protein", 0)
                        + counts.get("bagfilter", 0)
                    )
                    coarse = counts.get("coarse", 0) + counts.get("wheel_coarse", 0)
                    total_collected = fines + coarse
                    active = counts.get("active", 0)
                    eff = 100.0 * fines / total_collected if total_collected > 0 else 0.0

                    stats = {
                        "active_particles": active,
                        "collected_fines": fines,
                        "collected_coarse": coarse,
                        "separation_efficiency": eff,
                        **counts,
                    }
                    self.progress_updated.emit(progress, sim_time, stats)

            # ==============================================================
            # 7. Final results
            # ==============================================================
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

    # ----------------------------------------------------------------

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
# Helper
# ============================================================================

def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap *widget* in a frameless QScrollArea."""
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

    Uses ClassificationFlowPhysicsSimulator for real Warp-based physics.
    Settings mirror ``run_classification_flow.py`` CLI parameters.
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

    # ================================================================
    # Control tab
    # ================================================================

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

    # ================================================================
    # Settings tab (scrollable)
    # ================================================================

    def _create_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # ---- Time ----
        g = QGroupBox("Time Settings")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setRange(0.1, 3600.0)
        self.total_time_spin.setDecimals(1)
        self.total_time_spin.setSingleStep(10)
        self.total_time_spin.setValue(self._settings.total_time)
        self.total_time_spin.setSuffix(" s")
        self.total_time_spin.valueChanged.connect(lambda v: setattr(self._settings, 'total_time', v))
        f.addRow("Total Time:", self.total_time_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(self._settings.dt)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.valueChanged.connect(lambda v: setattr(self._settings, 'dt', v))
        f.addRow("Time Step (dt):", self.dt_spin)

        self.output_spin = QDoubleSpinBox()
        self.output_spin.setRange(0.01, 60.0)
        self.output_spin.setDecimals(2)
        self.output_spin.setValue(self._settings.output_interval)
        self.output_spin.setSuffix(" s")
        self.output_spin.valueChanged.connect(lambda v: setattr(self._settings, 'output_interval', v))
        f.addRow("Output Interval:", self.output_spin)

        layout.addWidget(g)

        # ---- Particles ----
        g = QGroupBox("Particles")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(100, 500000)
        self.num_particles_spin.setSingleStep(1000)
        self.num_particles_spin.setValue(self._settings.num_particles)
        self.num_particles_spin.valueChanged.connect(lambda v: setattr(self._settings, 'num_particles', v))
        f.addRow("Count:", self.num_particles_spin)

        self.continuous_check = QCheckBox("Continuous")
        self.continuous_check.setChecked(self._settings.continuous_feeding)
        self.continuous_check.setToolTip("Activate particles gradually at feed rate instead of all at t=0")
        self.continuous_check.stateChanged.connect(
            lambda s: setattr(self._settings, 'continuous_feeding', s == Qt.CheckState.Checked.value)
        )
        f.addRow("Feeding:", self.continuous_check)

        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(0, 500000)
        self.feed_rate_spin.setDecimals(0)
        self.feed_rate_spin.setSingleStep(100)
        self.feed_rate_spin.setValue(self._settings.particle_feed_rate)
        self.feed_rate_spin.setSuffix("  /s")
        self.feed_rate_spin.setToolTip("0 = auto-compute from mass flow (recommended)")
        self.feed_rate_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_feed_rate', v))
        f.addRow("Feed Rate:", self.feed_rate_spin)

        self.particle_dia_spin = QDoubleSpinBox()
        self.particle_dia_spin.setRange(1.0, 500.0)
        self.particle_dia_spin.setDecimals(1)
        self.particle_dia_spin.setSingleStep(5)
        self.particle_dia_spin.setValue(self._settings.particle_diameter_um)
        self.particle_dia_spin.setSuffix(" \u00b5m")
        self.particle_dia_spin.setToolTip("Mean diameter (used only when Material = none)")
        self.particle_dia_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_diameter_um', v))
        f.addRow("Mean Diameter:", self.particle_dia_spin)

        self.particle_std_spin = QDoubleSpinBox()
        self.particle_std_spin.setRange(0.0, 200.0)
        self.particle_std_spin.setDecimals(1)
        self.particle_std_spin.setSingleStep(5)
        self.particle_std_spin.setValue(self._settings.particle_std_um)
        self.particle_std_spin.setSuffix(" \u00b5m")
        self.particle_std_spin.setToolTip("Std deviation (used only when Material = none)")
        self.particle_std_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_std_um', v))
        f.addRow("Diameter Std Dev:", self.particle_std_spin)

        layout.addWidget(g)

        # ---- Material ----
        g = QGroupBox("Material")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.material_combo = QComboBox()
        self.material_combo.addItems(["yellow_pea", "faba_bean", "oat", "none"])
        self.material_combo.setToolTip("Food powder preset (provides realistic size distribution)")
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        f.addRow("Source:", self.material_combo)

        self.fraction_combo = QComboBox()
        self.fraction_combo.addItems(["whole", "protein", "starch", "fiber"])
        self.fraction_combo.setToolTip("Whole flour or single fraction")
        self.fraction_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_fraction', v))
        f.addRow("Fraction:", self.fraction_combo)

        layout.addWidget(g)

        # ---- Air / Assembly ----
        g = QGroupBox("Air & Assembly")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.assembly_mode_combo = QComboBox()
        self.assembly_mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.assembly_mode_combo.currentIndexChanged.connect(self._on_assembly_mode_changed)
        f.addRow("Mode:", self.assembly_mode_combo)

        self.air_flow_spin = QDoubleSpinBox()
        self.air_flow_spin.setRange(0.001, 5.0)
        self.air_flow_spin.setDecimals(3)
        self.air_flow_spin.setSingleStep(0.01)
        self.air_flow_spin.setValue(self._settings.air_flow_m3s)
        self.air_flow_spin.setSuffix("  m\u00b3/s")
        self.air_flow_spin.setToolTip("Default 0.491 = 1768 m\u00b3/h (air system at 2500 RPM)")
        self.air_flow_spin.valueChanged.connect(lambda v: setattr(self._settings, 'air_flow_m3s', v))
        f.addRow("Air Flow Rate:", self.air_flow_spin)

        self.bypass_spin = QDoubleSpinBox()
        self.bypass_spin.setRange(0.0, 0.99)
        self.bypass_spin.setDecimals(3)
        self.bypass_spin.setSingleStep(0.01)
        self.bypass_spin.setValue(self._settings.bypass_ratio)
        self.bypass_spin.setToolTip("Fraction of air bypassing venturi+zigzag (0 = no bypass)")
        self.bypass_spin.valueChanged.connect(lambda v: setattr(self._settings, 'bypass_ratio', v))
        f.addRow("Bypass Ratio:", self.bypass_spin)

        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(500, 20000)
        self.wheel_rpm_spin.setSingleStep(500)
        self.wheel_rpm_spin.setDecimals(0)
        self.wheel_rpm_spin.setValue(self._settings.wheel_rpm)
        self.wheel_rpm_spin.setSuffix("  RPM")
        self.wheel_rpm_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_rpm', v))
        f.addRow("Wheel Speed:", self.wheel_rpm_spin)

        self.wheel_diameter_spin = QDoubleSpinBox()
        self.wheel_diameter_spin.setRange(0.05, 0.50)
        self.wheel_diameter_spin.setDecimals(3)
        self.wheel_diameter_spin.setSingleStep(0.01)
        self.wheel_diameter_spin.setValue(self._settings.wheel_diameter)
        self.wheel_diameter_spin.setSuffix("  m")
        self.wheel_diameter_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_diameter', v))
        f.addRow("Wheel Diameter:", self.wheel_diameter_spin)

        layout.addWidget(g)

        # ---- Physics ----
        g = QGroupBox("Physics")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.turbulence_spin = QDoubleSpinBox()
        self.turbulence_spin.setRange(0.0, 0.5)
        self.turbulence_spin.setDecimals(2)
        self.turbulence_spin.setSingleStep(0.01)
        self.turbulence_spin.setValue(self._settings.turbulence_intensity)
        self.turbulence_spin.setToolTip("Fraction of mean velocity for zigzag mixing (15% typical)")
        self.turbulence_spin.valueChanged.connect(lambda v: setattr(self._settings, 'turbulence_intensity', v))
        f.addRow("Turbulence Intensity:", self.turbulence_spin)

        self.restitution_spin = QDoubleSpinBox()
        self.restitution_spin.setRange(0.0, 1.0)
        self.restitution_spin.setDecimals(2)
        self.restitution_spin.setSingleStep(0.05)
        self.restitution_spin.setValue(self._settings.restitution)
        self.restitution_spin.setToolTip("Particle-wall restitution (0=perfectly inelastic, 1=elastic)")
        self.restitution_spin.valueChanged.connect(lambda v: setattr(self._settings, 'restitution', v))
        f.addRow("Restitution:", self.restitution_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setSingleStep(0.05)
        self.friction_spin.setValue(self._settings.friction)
        self.friction_spin.setToolTip("Particle-wall friction coefficient")
        self.friction_spin.valueChanged.connect(lambda v: setattr(self._settings, 'friction', v))
        f.addRow("Friction:", self.friction_spin)

        self.max_loading_spin = QDoubleSpinBox()
        self.max_loading_spin.setRange(0.1, 10.0)
        self.max_loading_spin.setDecimals(1)
        self.max_loading_spin.setSingleStep(0.5)
        self.max_loading_spin.setValue(self._settings.max_loading_ratio)
        self.max_loading_spin.setToolTip("Max solids/air mass ratio for venturi entrainment cap")
        self.max_loading_spin.valueChanged.connect(lambda v: setattr(self._settings, 'max_loading_ratio', v))
        f.addRow("Max Loading Ratio:", self.max_loading_spin)

        layout.addWidget(g)

        # ---- Compute ----
        g = QGroupBox("Compute")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'device', v))
        f.addRow("Device:", self.device_combo)

        layout.addWidget(g)

        return widget

    # ---- settings callbacks ----

    def _on_material_changed(self, source: str):
        self._settings.material_source = source
        use_material = source not in ("none", "")
        # Enable/disable diameter fields based on material selection
        self.particle_dia_spin.setEnabled(not use_material)
        self.particle_std_spin.setEnabled(not use_material)
        self.fraction_combo.setEnabled(use_material)

    def _on_assembly_mode_changed(self, index: int):
        self._settings.use_preclassification = (index == 0)
        # Bypass only relevant with preclassification
        self.bypass_spin.setEnabled(index == 0)
        mode_name = "Full System" if index == 0 else "Wheel-Only"
        self._log(f"Assembly mode: {mode_name}")

    # ================================================================
    # Log tab
    # ================================================================

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

    # ================================================================
    # Button handlers
    # ================================================================

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

    # ================================================================
    # Simulation lifecycle
    # ================================================================

    def start_simulation(self, assembly_data: Dict[str, Any]):
        """Start a new simulation using the real physics engine."""
        s = self._settings
        self._log("=" * 56)
        self._log("CLASSIFICATION FLOW PHYSICS SIMULATION")
        self._log("=" * 56)
        self._log(f"  Device:     {s.device}")
        self._log(f"  Particles:  {s.num_particles:,}")
        self._log(f"  Time:       {s.total_time:.0f}s  (dt={s.dt}s, "
                   f"{int(s.total_time / s.dt):,} steps)")
        self._log(f"  Air flow:   {s.air_flow_m3s:.3f} m\u00b3/s "
                   f"({s.air_flow_m3s * 3600:.0f} m\u00b3/h)")
        self._log(f"  Material:   {s.material_source} / {s.material_fraction}")
        mode = "Full System" if s.use_preclassification else "Wheel-Only"
        self._log(f"  Mode:       {mode}")
        if s.bypass_ratio > 0:
            self._log(f"  Bypass:     {s.bypass_ratio:.1%}")
        self._log(f"  Wheel:      {s.wheel_rpm:.0f} RPM, \u00d8{s.wheel_diameter*1000:.0f} mm")
        self._log("=" * 56)

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

        wall = results.get('wall_time_s', 0)
        steps = int(self._settings.total_time / self._settings.dt)
        self._log("=" * 56)
        self._log("SIMULATION COMPLETED")
        self._log(f"  Wall time:   {wall:.1f}s  ({steps / wall:.0f} steps/s)" if wall > 0 else "  Wall time:   --")
        self._log(f"  Processed:   {results.get('particles_processed', 0):,}")
        self._log(f"  Fines:       {results.get('fines_collected', 0):,}")
        self._log(f"    Cyclone 1: {results.get('cyclone_1', 0):,}")
        self._log(f"    Cyclone 2: {results.get('cyclone_2', 0):,}")
        self._log(f"    Cyclone 3: {results.get('cyclone_3_protein', 0):,}")
        self._log(f"    Bag filter: {results.get('bagfilter', 0):,}")
        self._log(f"  Coarse:      {results.get('coarse_collected', 0):,}")
        self._log(f"    Zigzag:    {results.get('coarse', 0):,}")
        self._log(f"    Wheel:     {results.get('wheel_coarse', 0):,}")
        self._log(f"  Escaped:     {results.get('escaped', 0):,}")
        self._log(f"  Efficiency:  {results.get('separation_efficiency', 0):.1f}%")
        self._log("=" * 56)

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

    # ================================================================
    # Get / Set
    # ================================================================

    def get_settings(self) -> SimulationSettings:
        return self._settings

    def set_settings(self, settings: SimulationSettings):
        self._settings = settings
        # Sync all widgets
        self.total_time_spin.setValue(settings.total_time)
        self.dt_spin.setValue(settings.dt)
        self.output_spin.setValue(settings.output_interval)
        self.num_particles_spin.setValue(settings.num_particles)
        self.feed_rate_spin.setValue(settings.particle_feed_rate)
        self.continuous_check.setChecked(settings.continuous_feeding)
        self.particle_dia_spin.setValue(settings.particle_diameter_um)
        self.particle_std_spin.setValue(settings.particle_std_um)
        self.device_combo.setCurrentText(settings.device)
        self.material_combo.setCurrentText(settings.material_source)
        self.fraction_combo.setCurrentText(settings.material_fraction)
        self.air_flow_spin.setValue(settings.air_flow_m3s)
        self.bypass_spin.setValue(settings.bypass_ratio)
        self.wheel_rpm_spin.setValue(settings.wheel_rpm)
        self.wheel_diameter_spin.setValue(settings.wheel_diameter)
        self.turbulence_spin.setValue(settings.turbulence_intensity)
        self.restitution_spin.setValue(settings.restitution)
        self.friction_spin.setValue(settings.friction)
        self.max_loading_spin.setValue(settings.max_loading_ratio)
        idx = 0 if settings.use_preclassification else 1
        self.assembly_mode_combo.setCurrentIndex(idx)
