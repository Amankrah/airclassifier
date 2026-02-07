"""
Simulation Settings Dialog
==========================

Dialog for configuring simulation parameters.
"""

from typing import Optional
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QDialogButtonBox, QLabel, QSlider,
)
from PySide6.QtCore import Qt

from ..panels.simulation_control import SimulationSettings


class SimulationSettingsDialog(QDialog):
    """Dialog for configuring simulation settings."""

    def __init__(self, settings: SimulationSettings, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        self.setMinimumWidth(500)

        self._settings = SimulationSettings(
            total_time=settings.total_time,
            dt=settings.dt,
            output_interval=settings.output_interval,
            num_particles=settings.num_particles,
            particle_feed_rate=settings.particle_feed_rate,
            continuous_feeding=settings.continuous_feeding,
            turbulence_base=settings.turbulence_base,
            restitution=settings.restitution,
            friction=settings.friction,
            device=settings.device,
            precision=settings.precision,
            material_source=settings.material_source,
            material_fraction=settings.material_fraction,
            show_particles=settings.show_particles,
            show_velocity_field=settings.show_velocity_field,
            particle_color_mode=settings.particle_color_mode,
            recirculate_passes=settings.recirculate_passes,
            recirculate_fractions=settings.recirculate_fractions,
            attrition_factor=settings.attrition_factor,
            attrition_min_um=settings.attrition_min_um,
            recirculate_wheel_rpm=settings.recirculate_wheel_rpm,
            recirculate_time=settings.recirculate_time,
        )

        self._setup_ui()
        self._load_values()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Time tab
        time_tab = self._create_time_tab()
        tabs.addTab(time_tab, "Time")

        # Particles tab
        particles_tab = self._create_particles_tab()
        tabs.addTab(particles_tab, "Particles")

        # Physics tab
        physics_tab = self._create_physics_tab()
        tabs.addTab(physics_tab, "Physics")

        # Compute tab
        compute_tab = self._create_compute_tab()
        tabs.addTab(compute_tab, "Compute")

        # Recirculation tab
        recirc_tab = self._create_recirculation_tab()
        tabs.addTab(recirc_tab, "Recirculation")

        # Visualization tab
        viz_tab = self._create_visualization_tab()
        tabs.addTab(viz_tab, "Visualization")

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._restore_defaults)
        layout.addWidget(buttons)

    def _create_time_tab(self) -> QWidget:
        """Create time settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Time Settings")
        form = QFormLayout(group)

        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setRange(0.1, 600.0)
        self.total_time_spin.setDecimals(1)
        self.total_time_spin.setSuffix(" s")
        form.addRow("Total Simulation Time:", self.total_time_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setSuffix(" s")
        form.addRow("Time Step (dt):", self.dt_spin)

        self.output_spin = QDoubleSpinBox()
        self.output_spin.setRange(0.01, 10.0)
        self.output_spin.setDecimals(2)
        self.output_spin.setSuffix(" s")
        form.addRow("Output Interval:", self.output_spin)

        # Info labels
        self.steps_label = QLabel()
        form.addRow("Total Steps:", self.steps_label)

        self.total_time_spin.valueChanged.connect(self._update_info)
        self.dt_spin.valueChanged.connect(self._update_info)

        layout.addWidget(group)
        layout.addStretch()
        return widget

    def _create_particles_tab(self) -> QWidget:
        """Create particle settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Particle count
        count_group = QGroupBox("Particle Count")
        count_form = QFormLayout(count_group)

        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(100, 100000)
        self.num_particles_spin.setSingleStep(1000)
        count_form.addRow("Number of Particles:", self.num_particles_spin)

        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(10, 10000)
        self.feed_rate_spin.setSuffix(" /s")
        count_form.addRow("Feed Rate:", self.feed_rate_spin)

        self.continuous_check = QCheckBox("Feed particles continuously")
        count_form.addRow("Continuous Feeding:", self.continuous_check)

        layout.addWidget(count_group)

        # Material
        material_group = QGroupBox("Material Properties")
        material_form = QFormLayout(material_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        material_form.addRow("Material Source:", self.source_combo)

        self.fraction_combo = QComboBox()
        self.fraction_combo.addItems(["whole", "protein", "starch", "fiber"])
        material_form.addRow("Fraction:", self.fraction_combo)

        layout.addWidget(material_group)

        layout.addStretch()
        return widget

    def _create_physics_tab(self) -> QWidget:
        """Create physics settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Turbulence
        turb_group = QGroupBox("Turbulence")
        turb_form = QFormLayout(turb_group)

        self.turbulence_spin = QDoubleSpinBox()
        self.turbulence_spin.setRange(0.0, 0.5)
        self.turbulence_spin.setDecimals(2)
        self.turbulence_spin.setSingleStep(0.01)
        turb_form.addRow("Turbulence Intensity:", self.turbulence_spin)

        layout.addWidget(turb_group)

        # Collision
        collision_group = QGroupBox("Particle-Wall Collision")
        collision_form = QFormLayout(collision_group)

        self.restitution_spin = QDoubleSpinBox()
        self.restitution_spin.setRange(0.0, 1.0)
        self.restitution_spin.setDecimals(2)
        collision_form.addRow("Restitution Coefficient:", self.restitution_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1.0)
        self.friction_spin.setDecimals(2)
        collision_form.addRow("Friction Coefficient:", self.friction_spin)

        layout.addWidget(collision_group)

        layout.addStretch()
        return widget

    def _create_compute_tab(self) -> QWidget:
        """Create compute settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Compute Device")
        form = QFormLayout(group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        form.addRow("Device:", self.device_combo)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "float64"])
        form.addRow("Precision:", self.precision_combo)

        # GPU info
        gpu_info = QLabel()
        try:
            import warp as wp
            wp.init()
            devices = wp.get_devices()
            cuda_devices = [str(d) for d in devices if "cuda" in str(d).lower()]
            if cuda_devices:
                gpu_info.setText(f"Available: {', '.join(cuda_devices)}")
                gpu_info.setStyleSheet("color: #4ec9b0;")
            else:
                gpu_info.setText("No CUDA devices found")
                gpu_info.setStyleSheet("color: #dcdcaa;")
        except:
            gpu_info.setText("Warp not available")
            gpu_info.setStyleSheet("color: #f14c4c;")

        form.addRow("GPU Status:", gpu_info)

        layout.addWidget(group)

        layout.addStretch()
        return widget

    def _create_recirculation_tab(self) -> QWidget:
        """Create recirculation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Multi-pass
        pass_group = QGroupBox("Multi-Pass Recirculation")
        pass_form = QFormLayout(pass_group)

        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 10)
        self.passes_spin.setToolTip("1 = single pass (no recirculation)")
        pass_form.addRow("Number of Passes:", self.passes_spin)

        self.recirc_fractions_combo = QComboBox()
        self.recirc_fractions_combo.addItems([
            "cy1", "cy1,cy2", "cy2", "wheel_coarse", "cy1,wheel_coarse",
        ])
        self.recirc_fractions_combo.setEditable(True)
        self.recirc_fractions_combo.setToolTip(
            "Fractions to refeed after each pass.\n"
            "Valid: cy1, cy2, cy3, wheel_coarse, zigzag_coarse, bagfilter"
        )
        pass_form.addRow("Refeed Fractions:", self.recirc_fractions_combo)

        layout.addWidget(pass_group)

        # Attrition
        attr_group = QGroupBox("Venturi Attrition")
        attr_form = QFormLayout(attr_group)

        self.attrition_spin = QDoubleSpinBox()
        self.attrition_spin.setRange(0.0, 0.50)
        self.attrition_spin.setDecimals(2)
        self.attrition_spin.setSingleStep(0.05)
        self.attrition_spin.setToolTip(
            "Diameter reduction per pass (0.10 = 10%).\n"
            "Models shear breakup at venturi throat."
        )
        attr_form.addRow("Attrition Rate:", self.attrition_spin)

        self.attrition_min_spin = QDoubleSpinBox()
        self.attrition_min_spin.setRange(1.0, 20.0)
        self.attrition_min_spin.setDecimals(1)
        self.attrition_min_spin.setSuffix(" \u00b5m")
        self.attrition_min_spin.setToolTip("Floor diameter (protein bodies don't break further)")
        attr_form.addRow("Min Diameter:", self.attrition_min_spin)

        layout.addWidget(attr_group)

        # Pass 2+ overrides
        override_group = QGroupBox("Pass 2+ Overrides")
        override_form = QFormLayout(override_group)

        self.recirc_wheel_spin = QDoubleSpinBox()
        self.recirc_wheel_spin.setRange(0, 20000)
        self.recirc_wheel_spin.setDecimals(0)
        self.recirc_wheel_spin.setSuffix(" RPM")
        self.recirc_wheel_spin.setToolTip("Wheel RPM for recirculation passes. 0 = same as main.")
        override_form.addRow("Wheel RPM:", self.recirc_wheel_spin)

        self.recirc_time_spin = QDoubleSpinBox()
        self.recirc_time_spin.setRange(0, 3600)
        self.recirc_time_spin.setDecimals(0)
        self.recirc_time_spin.setSuffix(" s")
        self.recirc_time_spin.setToolTip("Sim time for passes 2+. 0 = same as Total Time.")
        override_form.addRow("Sim Time:", self.recirc_time_spin)

        layout.addWidget(override_group)

        layout.addStretch()
        return widget

    def _create_visualization_tab(self) -> QWidget:
        """Create visualization settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Real-time Visualization")
        form = QFormLayout(group)

        self.show_particles_check = QCheckBox("Show particles")
        form.addRow("Particles:", self.show_particles_check)

        self.show_flow_check = QCheckBox("Show velocity field")
        form.addRow("Flow Field:", self.show_flow_check)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["velocity", "size", "type", "zone"])
        form.addRow("Particle Color Mode:", self.color_mode_combo)

        layout.addWidget(group)

        layout.addStretch()
        return widget

    def _load_values(self):
        """Load current settings into widgets."""
        self.total_time_spin.setValue(self._settings.total_time)
        self.dt_spin.setValue(self._settings.dt)
        self.output_spin.setValue(self._settings.output_interval)
        self.num_particles_spin.setValue(self._settings.num_particles)
        self.feed_rate_spin.setValue(self._settings.particle_feed_rate)
        self.continuous_check.setChecked(self._settings.continuous_feeding)
        self.source_combo.setCurrentText(self._settings.material_source)
        self.fraction_combo.setCurrentText(self._settings.material_fraction)
        self.turbulence_spin.setValue(self._settings.turbulence_base)
        self.restitution_spin.setValue(self._settings.restitution)
        self.friction_spin.setValue(self._settings.friction)
        self.device_combo.setCurrentText(self._settings.device)
        self.precision_combo.setCurrentText(self._settings.precision)
        self.show_particles_check.setChecked(self._settings.show_particles)
        self.show_flow_check.setChecked(self._settings.show_velocity_field)
        self.color_mode_combo.setCurrentText(self._settings.particle_color_mode)
        # Recirculation
        self.passes_spin.setValue(self._settings.recirculate_passes)
        self.recirc_fractions_combo.setCurrentText(self._settings.recirculate_fractions)
        self.attrition_spin.setValue(self._settings.attrition_factor)
        self.attrition_min_spin.setValue(self._settings.attrition_min_um)
        self.recirc_wheel_spin.setValue(self._settings.recirculate_wheel_rpm)
        self.recirc_time_spin.setValue(self._settings.recirculate_time)

        self._update_info()

    def _update_info(self):
        """Update calculated info labels."""
        total_time = self.total_time_spin.value()
        dt = self.dt_spin.value()
        if dt > 0:
            steps = int(total_time / dt)
            self.steps_label.setText(f"{steps:,}")

    def _save_and_close(self):
        """Save settings and close dialog."""
        self._settings.total_time = self.total_time_spin.value()
        self._settings.dt = self.dt_spin.value()
        self._settings.output_interval = self.output_spin.value()
        self._settings.num_particles = self.num_particles_spin.value()
        self._settings.particle_feed_rate = self.feed_rate_spin.value()
        self._settings.continuous_feeding = self.continuous_check.isChecked()
        self._settings.material_source = self.source_combo.currentText()
        self._settings.material_fraction = self.fraction_combo.currentText()
        self._settings.turbulence_base = self.turbulence_spin.value()
        self._settings.restitution = self.restitution_spin.value()
        self._settings.friction = self.friction_spin.value()
        self._settings.device = self.device_combo.currentText()
        self._settings.precision = self.precision_combo.currentText()
        self._settings.show_particles = self.show_particles_check.isChecked()
        self._settings.show_velocity_field = self.show_flow_check.isChecked()
        self._settings.particle_color_mode = self.color_mode_combo.currentText()
        # Recirculation
        self._settings.recirculate_passes = self.passes_spin.value()
        self._settings.recirculate_fractions = self.recirc_fractions_combo.currentText()
        self._settings.attrition_factor = self.attrition_spin.value()
        self._settings.attrition_min_um = self.attrition_min_spin.value()
        self._settings.recirculate_wheel_rpm = self.recirc_wheel_spin.value()
        self._settings.recirculate_time = self.recirc_time_spin.value()

        self.accept()

    def _restore_defaults(self):
        """Restore default settings."""
        defaults = SimulationSettings()
        self._settings = defaults
        self._load_values()

    def get_settings(self) -> SimulationSettings:
        """Get the configured settings."""
        return self._settings
