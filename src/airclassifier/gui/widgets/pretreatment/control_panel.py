"""
Pretreatment Control Panel
==========================

Modern glassmorphism-styled control panel for GP-15 RF heating
recipe settings with visual sliders and material presets.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QWidget, QSlider, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QGroupBox,
    QSizePolicy, QFormLayout,
)

from ...theme import COLORS


class LabeledSlider(QFrame):
    """Slider with label showing current value."""

    value_changed = Signal(float)

    def __init__(
        self,
        title: str,
        min_val: float,
        max_val: float,
        default: float,
        unit: str = "",
        decimals: int = 1,
        steps: int = 100,
        accent_color: str = COLORS.ACCENT,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._steps = steps
        self._unit = unit
        self._accent = accent_color

        self._setup_ui(title, default)

    def _setup_ui(self, title: str, default: float):
        self.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header: title + value
        header = QHBoxLayout()
        header.setSpacing(6)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"""
            font-size: 9pt;
            font-weight: 600;
            color: {COLORS.TEXT_SECONDARY};
        """)
        header.addWidget(self._title_label)

        header.addStretch()

        self._value_label = QLabel()
        self._value_label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 700;
            color: {self._accent};
        """)
        header.addWidget(self._value_label)

        layout.addLayout(header)

        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {COLORS.BG_DARKEST};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {self._accent};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS.ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {self._accent};
                border-radius: 3px;
            }}
        """)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

        # Set initial value
        self.set_value(default)

    def _on_slider_changed(self, step: int):
        value = self._min + (self._max - self._min) * step / self._steps
        self._update_label(value)
        self.value_changed.emit(value)

    def _update_label(self, value: float):
        fmt = f"{{:.{self._decimals}f}}"
        text = fmt.format(value)
        if self._unit:
            text += f" {self._unit}"
        self._value_label.setText(text)

    def value(self) -> float:
        step = self._slider.value()
        return self._min + (self._max - self._min) * step / self._steps

    def set_value(self, value: float):
        value = max(self._min, min(self._max, value))
        step = int((value - self._min) / (self._max - self._min) * self._steps)
        self._slider.blockSignals(True)
        self._slider.setValue(step)
        self._slider.blockSignals(False)
        self._update_label(value)


class PretreatmentControlPanel(QFrame):
    """Modern control panel for GP-15 RF heating settings.

    Features:
    - Material preset selector with visual cards
    - Recipe sliders (gap, belt speed, extraction fan)
    - Physics toggle options
    - Run/Stop controls

    Signals:
        run_requested(dict): Emitted with settings when Run is clicked
        stop_requested(): Emitted when Stop is clicked
        settings_changed(dict): Emitted when any setting changes
    """

    run_requested = Signal(dict)
    stop_requested = Signal()
    settings_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_style()
        self._setup_ui()
        self._connect_signals()

    def _setup_style(self):
        self.setObjectName("pretreatControlPanel")
        self.setStyleSheet(f"""
            QFrame#pretreatControlPanel {{
                background: {COLORS.BG_ELEVATED};
                border-left: 1px solid {COLORS.BORDER};
            }}
        """)
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Title
        title = QLabel("GP-15 RF Heating")
        title.setStyleSheet(f"""
            font-size: 13pt;
            font-weight: 700;
            color: {COLORS.PRETREAT_PRIMARY};
        """)
        layout.addWidget(title)

        # Material Section
        material_group = self._build_material_section()
        layout.addWidget(material_group)

        # Recipe Section
        recipe_group = self._build_recipe_section()
        layout.addWidget(recipe_group)

        # Physics Section
        physics_group = self._build_physics_section()
        layout.addWidget(physics_group)

        layout.addStretch()

        # Control Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS.PRETREAT_PRIMARY}, stop:1 {COLORS.PRETREAT_SECONDARY});
                border: none;
                border-radius: 8px;
                font-size: 11pt;
                font-weight: 600;
                color: {COLORS.TEXT_INVERSE};
                padding: 8px 24px;
            }}
            QPushButton:hover {{
                background: {COLORS.PRETREAT_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {COLORS.BG_SURFACE};
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.DANGER};
                border: none;
                border-radius: 8px;
                font-size: 11pt;
                font-weight: 600;
                color: {COLORS.TEXT_INVERSE};
                padding: 8px 20px;
            }}
            QPushButton:hover {{
                background: #ff6b6b;
            }}
            QPushButton:disabled {{
                background: {COLORS.BG_SURFACE};
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._stop_btn)

        layout.addLayout(btn_layout)

    def _build_material_section(self) -> QGroupBox:
        group = QGroupBox("Material")
        group.setStyleSheet(self._get_group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Material preset
        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        preset_row.addWidget(preset_label)

        self._material_combo = QComboBox()
        self._material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        self._material_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 6px 10px;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QComboBox:hover {{
                border-color: {COLORS.ACCENT};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
        """)
        preset_row.addWidget(self._material_combo, 1)
        layout.addLayout(preset_row)

        # Moisture slider
        self._moisture_slider = LabeledSlider(
            title="Inlet Moisture",
            min_val=0.05,
            max_val=0.20,
            default=0.10,
            unit="wb",
            decimals=3,
            accent_color=COLORS.KPI_MOISTURE,
        )
        layout.addWidget(self._moisture_slider)

        # Initial temperature
        self._temp_slider = LabeledSlider(
            title="Initial Temperature",
            min_val=10.0,
            max_val=35.0,
            default=17.6,
            unit="\u00b0C",
            decimals=1,
            accent_color=COLORS.KPI_TEMPERATURE,
        )
        layout.addWidget(self._temp_slider)

        # Bed depth
        self._bed_slider = LabeledSlider(
            title="Bed Depth",
            min_val=15,
            max_val=60,
            default=25,
            unit="mm",
            decimals=0,
            accent_color=COLORS.TEXT_PRIMARY,
        )
        layout.addWidget(self._bed_slider)

        return group

    def _build_recipe_section(self) -> QGroupBox:
        group = QGroupBox("GP-15 Recipe")
        group.setStyleSheet(self._get_group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Electrode gap
        self._gap_slider = LabeledSlider(
            title="Electrode Gap",
            min_val=30,
            max_val=150,
            default=75,
            unit="mm",
            decimals=0,
            accent_color=COLORS.KPI_ELECTRODE_GAP,
        )
        layout.addWidget(self._gap_slider)

        # Belt speed
        self._speed_slider = LabeledSlider(
            title="Belt Speed",
            min_val=0.10,
            max_val=1.0,
            default=0.20,
            unit="m/min",
            decimals=2,
            accent_color=COLORS.ACCENT,
        )
        layout.addWidget(self._speed_slider)

        # Run mass
        self._mass_slider = LabeledSlider(
            title="Run Mass",
            min_val=10,
            max_val=200,
            default=61,
            unit="kg",
            decimals=0,
            accent_color=COLORS.KPI_THROUGHPUT,
        )
        layout.addWidget(self._mass_slider)

        # Extraction fan
        self._fan_slider = LabeledSlider(
            title="Extraction Fan",
            min_val=10,
            max_val=55,
            default=35,
            unit="Hz",
            decimals=0,
            accent_color=COLORS.INFO,
        )
        layout.addWidget(self._fan_slider)

        # MRH threshold
        self._mrh_slider = LabeledSlider(
            title="MRH Threshold",
            min_val=1.0,
            max_val=3.0,
            default=1.7,
            unit="A",
            decimals=2,
            accent_color=COLORS.KPI_ANODE_CURRENT,
        )
        layout.addWidget(self._mrh_slider)

        # Heater toggle
        heater_row = QHBoxLayout()
        self._heater_check = QCheckBox("Both heater banks on")
        self._heater_check.setChecked(True)
        self._heater_check.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        heater_row.addWidget(self._heater_check)
        layout.addLayout(heater_row)

        return group

    def _build_physics_section(self) -> QGroupBox:
        group = QGroupBox("Physics")
        group.setStyleSheet(self._get_group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # Duration
        self._duration_slider = LabeledSlider(
            title="Duration (0=auto)",
            min_val=0,
            max_val=600,
            default=0,
            unit="s",
            decimals=0,
            accent_color=COLORS.TEXT_SECONDARY,
        )
        layout.addWidget(self._duration_slider)

        # Physics toggles
        self._tvd_check = QCheckBox("Van Leer TVD advection")
        self._tvd_check.setChecked(True)
        self._tvd_check.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(self._tvd_check)

        self._ctrl_check = QCheckBox("PLC controller (MRH/MRL)")
        self._ctrl_check.setChecked(True)
        self._ctrl_check.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(self._ctrl_check)

        self._corr_check = QCheckBox("Fringe + perforation corrections")
        self._corr_check.setChecked(False)
        self._corr_check.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(self._corr_check)

        # Oscillator efficiency
        self._eff_slider = LabeledSlider(
            title="Oscillator Efficiency",
            min_val=0.30,
            max_val=0.80,
            default=0.56,
            unit="",
            decimals=2,
            accent_color=COLORS.KPI_EFFICIENCY,
        )
        layout.addWidget(self._eff_slider)

        # Device selector
        device_row = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        device_row.addWidget(device_label)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda"])
        self._device_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 80px;
            }}
        """)
        device_row.addWidget(self._device_combo)
        device_row.addStretch()
        layout.addLayout(device_row)

        return group

    def _get_group_style(self) -> str:
        return f"""
            QGroupBox {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
                font-size: 9pt;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """

    def _connect_signals(self):
        # Connect all value changes to emit settings_changed
        self._material_combo.currentTextChanged.connect(self._emit_settings)
        self._moisture_slider.value_changed.connect(self._emit_settings)
        self._temp_slider.value_changed.connect(self._emit_settings)
        self._bed_slider.value_changed.connect(self._emit_settings)
        self._gap_slider.value_changed.connect(self._emit_settings)
        self._speed_slider.value_changed.connect(self._emit_settings)
        self._mass_slider.value_changed.connect(self._emit_settings)
        self._fan_slider.value_changed.connect(self._emit_settings)
        self._mrh_slider.value_changed.connect(self._emit_settings)

    def _emit_settings(self, *args):
        self.settings_changed.emit(self.get_settings())

    def _on_run(self):
        self.run_requested.emit(self.get_settings())

    def _on_stop(self):
        self.stop_requested.emit()

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings as a dictionary."""
        return {
            # Material
            "material": self._material_combo.currentText(),
            "inlet_moisture": self._moisture_slider.value(),
            "initial_temp_c": self._temp_slider.value(),
            "bed_depth_mm": self._bed_slider.value(),
            # Recipe
            "electrode_gap_mm": self._gap_slider.value(),
            "belt_speed": self._speed_slider.value(),
            "run_mass_kg": self._mass_slider.value(),
            "extraction_fan_hz": self._fan_slider.value(),
            "mrh_amps": self._mrh_slider.value(),
            "heaters_on": self._heater_check.isChecked(),
            # Physics
            "duration_s": self._duration_slider.value(),
            "tvd": self._tvd_check.isChecked(),
            "controller": self._ctrl_check.isChecked(),
            "corrections": self._corr_check.isChecked(),
            "efficiency": self._eff_slider.value(),
            "device": self._device_combo.currentText(),
        }

    def load_settings(self, settings: Dict[str, Any]):
        """Load settings from a dictionary."""
        if "material" in settings:
            idx = self._material_combo.findText(settings["material"])
            if idx >= 0:
                self._material_combo.setCurrentIndex(idx)
        if "inlet_moisture" in settings:
            self._moisture_slider.set_value(settings["inlet_moisture"])
        if "initial_temp_c" in settings:
            self._temp_slider.set_value(settings["initial_temp_c"])
        if "bed_depth_mm" in settings:
            self._bed_slider.set_value(settings["bed_depth_mm"])
        if "electrode_gap_mm" in settings:
            self._gap_slider.set_value(settings["electrode_gap_mm"])
        if "belt_speed" in settings:
            self._speed_slider.set_value(settings["belt_speed"])
        if "run_mass_kg" in settings:
            self._mass_slider.set_value(settings["run_mass_kg"])
        if "extraction_fan_hz" in settings:
            self._fan_slider.set_value(settings["extraction_fan_hz"])
        if "mrh_amps" in settings:
            self._mrh_slider.set_value(settings["mrh_amps"])
        if "heaters_on" in settings:
            self._heater_check.setChecked(settings["heaters_on"])
        if "duration_s" in settings:
            self._duration_slider.set_value(settings["duration_s"])
        if "tvd" in settings:
            self._tvd_check.setChecked(settings["tvd"])
        if "controller" in settings:
            self._ctrl_check.setChecked(settings["controller"])
        if "corrections" in settings:
            self._corr_check.setChecked(settings["corrections"])
        if "efficiency" in settings:
            self._eff_slider.set_value(settings["efficiency"])
        if "device" in settings:
            idx = self._device_combo.findText(settings["device"])
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)

    def set_running(self, running: bool):
        """Update button states based on running status."""
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
