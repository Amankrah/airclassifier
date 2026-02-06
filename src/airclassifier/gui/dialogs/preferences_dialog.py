"""
Preferences Dialog
==================

Application preferences and settings dialog.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QGroupBox, QPushButton, QDialogButtonBox, QLabel,
    QFileDialog,
)
from PySide6.QtCore import Qt, QSettings


class PreferencesDialog(QDialog):
    """Application preferences dialog."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)

        self._settings = QSettings()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # General tab
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "General")

        # Simulation tab
        sim_tab = self._create_simulation_tab()
        tabs.addTab(sim_tab, "Simulation")

        # Visualization tab
        viz_tab = self._create_visualization_tab()
        tabs.addTab(viz_tab, "Visualization")

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._save_settings)
        layout.addWidget(buttons)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # UI settings
        ui_group = QGroupBox("User Interface")
        ui_layout = QFormLayout(ui_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        ui_layout.addRow("Theme:", self.theme_combo)

        self.auto_save_check = QCheckBox("Enable auto-save")
        ui_layout.addRow("Auto-save:", self.auto_save_check)

        self.auto_save_interval = QSpinBox()
        self.auto_save_interval.setRange(1, 60)
        self.auto_save_interval.setSuffix(" min")
        ui_layout.addRow("Auto-save Interval:", self.auto_save_interval)

        layout.addWidget(ui_group)

        # Paths
        paths_group = QGroupBox("Default Paths")
        paths_layout = QFormLayout(paths_group)

        self.project_path = QLineEdit()
        browse_proj = QPushButton("Browse...")
        browse_proj.clicked.connect(lambda: self._browse_path(self.project_path))
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(self.project_path)
        proj_layout.addWidget(browse_proj)
        paths_layout.addRow("Projects:", proj_layout)

        self.export_path = QLineEdit()
        browse_exp = QPushButton("Browse...")
        browse_exp.clicked.connect(lambda: self._browse_path(self.export_path))
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(self.export_path)
        exp_layout.addWidget(browse_exp)
        paths_layout.addRow("Exports:", exp_layout)

        layout.addWidget(paths_group)

        layout.addStretch()
        return widget

    def _create_simulation_tab(self) -> QWidget:
        """Create simulation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Compute settings
        compute_group = QGroupBox("Compute Device")
        compute_layout = QFormLayout(compute_group)

        self.default_device = QComboBox()
        self.default_device.addItems(["cuda", "cpu"])
        compute_layout.addRow("Default Device:", self.default_device)

        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["float32", "float64"])
        compute_layout.addRow("Precision:", self.precision_combo)

        layout.addWidget(compute_group)

        # Default simulation parameters
        defaults_group = QGroupBox("Default Parameters")
        defaults_layout = QFormLayout(defaults_group)

        self.default_particles = QSpinBox()
        self.default_particles.setRange(100, 100000)
        defaults_layout.addRow("Number of Particles:", self.default_particles)

        self.default_time = QDoubleSpinBox()
        self.default_time.setRange(0.1, 600.0)
        self.default_time.setSuffix(" s")
        defaults_layout.addRow("Simulation Time:", self.default_time)

        self.default_dt = QDoubleSpinBox()
        self.default_dt.setDecimals(4)
        self.default_dt.setRange(0.0001, 0.01)
        self.default_dt.setSuffix(" s")
        defaults_layout.addRow("Time Step:", self.default_dt)

        layout.addWidget(defaults_group)

        layout.addStretch()
        return widget

    def _create_visualization_tab(self) -> QWidget:
        """Create visualization settings tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 3D viewport
        viewport_group = QGroupBox("3D Viewport")
        viewport_layout = QFormLayout(viewport_group)

        self.default_opacity = QSpinBox()
        self.default_opacity.setRange(10, 100)
        self.default_opacity.setSuffix("%")
        viewport_layout.addRow("Default Opacity:", self.default_opacity)

        self.show_axes = QCheckBox("Show coordinate axes")
        viewport_layout.addRow("Axes:", self.show_axes)

        self.enable_shadows = QCheckBox("Enable shadows")
        viewport_layout.addRow("Shadows:", self.enable_shadows)

        self.antialiasing = QCheckBox("Enable antialiasing")
        viewport_layout.addRow("Antialiasing:", self.antialiasing)

        layout.addWidget(viewport_group)

        # Particle rendering
        particles_group = QGroupBox("Particle Rendering")
        particles_layout = QFormLayout(particles_group)

        self.particle_size = QSpinBox()
        self.particle_size.setRange(1, 20)
        particles_layout.addRow("Point Size:", self.particle_size)

        self.color_mode = QComboBox()
        self.color_mode.addItems(["velocity", "size", "type", "zone"])
        particles_layout.addRow("Color Mode:", self.color_mode)

        layout.addWidget(particles_group)

        layout.addStretch()
        return widget

    def _browse_path(self, line_edit: QLineEdit):
        """Open directory browser."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)

    def _load_settings(self):
        """Load settings from QSettings."""
        # General
        self.theme_combo.setCurrentText(self._settings.value("General/theme", "Dark"))
        self.auto_save_check.setChecked(self._settings.value("General/autoSave", True, type=bool))
        self.auto_save_interval.setValue(self._settings.value("General/autoSaveInterval", 5, type=int))
        self.project_path.setText(self._settings.value("General/projectPath", ""))
        self.export_path.setText(self._settings.value("General/exportPath", ""))

        # Simulation
        self.default_device.setCurrentText(self._settings.value("Simulation/device", "cuda"))
        self.precision_combo.setCurrentText(self._settings.value("Simulation/precision", "float32"))
        self.default_particles.setValue(self._settings.value("Simulation/particles", 5000, type=int))
        self.default_time.setValue(self._settings.value("Simulation/time", 10.0, type=float))
        self.default_dt.setValue(self._settings.value("Simulation/dt", 0.001, type=float))

        # Visualization
        self.default_opacity.setValue(self._settings.value("Visualization/opacity", 80, type=int))
        self.show_axes.setChecked(self._settings.value("Visualization/showAxes", True, type=bool))
        self.enable_shadows.setChecked(self._settings.value("Visualization/shadows", True, type=bool))
        self.antialiasing.setChecked(self._settings.value("Visualization/antialiasing", True, type=bool))
        self.particle_size.setValue(self._settings.value("Visualization/particleSize", 5, type=int))
        self.color_mode.setCurrentText(self._settings.value("Visualization/colorMode", "velocity"))

    def _save_settings(self):
        """Save settings to QSettings."""
        # General
        self._settings.setValue("General/theme", self.theme_combo.currentText())
        self._settings.setValue("General/autoSave", self.auto_save_check.isChecked())
        self._settings.setValue("General/autoSaveInterval", self.auto_save_interval.value())
        self._settings.setValue("General/projectPath", self.project_path.text())
        self._settings.setValue("General/exportPath", self.export_path.text())

        # Simulation
        self._settings.setValue("Simulation/device", self.default_device.currentText())
        self._settings.setValue("Simulation/precision", self.precision_combo.currentText())
        self._settings.setValue("Simulation/particles", self.default_particles.value())
        self._settings.setValue("Simulation/time", self.default_time.value())
        self._settings.setValue("Simulation/dt", self.default_dt.value())

        # Visualization
        self._settings.setValue("Visualization/opacity", self.default_opacity.value())
        self._settings.setValue("Visualization/showAxes", self.show_axes.isChecked())
        self._settings.setValue("Visualization/shadows", self.enable_shadows.isChecked())
        self._settings.setValue("Visualization/antialiasing", self.antialiasing.isChecked())
        self._settings.setValue("Visualization/particleSize", self.particle_size.value())
        self._settings.setValue("Visualization/colorMode", self.color_mode.currentText())

    def _save_and_close(self):
        """Save settings and close dialog."""
        self._save_settings()
        self.accept()
