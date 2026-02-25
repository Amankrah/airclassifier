"""
Milling configuration dialog — Hammer / pin mill only.
Per-system config: rotor, hammers, screen, housing, feed & discharge.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDialogButtonBox,
    QDoubleSpinBox, QSpinBox, QScrollArea, QFrame,
)
from PySide6.QtCore import Signal

from ..theme import COLORS


class MillingConfigDialog(QDialog):
    """Configure Hammer Mill (pin mill) only. Emits params for this system."""

    milling_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure Pin Mill (Hammer Mill)")
        self.setMinimumSize(480, 600)
        self._params = current_params or {}
        self._setup_ui()
        self.load_params(self._params)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Pin Mill — Hammer mill (impact + screen classification)")
        header.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Configure machine geometry and operating point. Build and run from the Milling page.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        clayout = QVBoxLayout(content)
        clayout.setSpacing(8)

        _M = (10, 14, 10, 10)

        # --- Group 1: Rotor & Drive ---
        g = QGroupBox("Rotor && Drive")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self.mill_rotor_rpm_spin = QDoubleSpinBox()
        self.mill_rotor_rpm_spin.setRange(500, 6000)
        self.mill_rotor_rpm_spin.setValue(3000)
        self.mill_rotor_rpm_spin.setDecimals(0)
        self.mill_rotor_rpm_spin.setSuffix("  rpm")
        f.addRow("Rotor RPM:", self.mill_rotor_rpm_spin)

        self.mill_rotor_diameter_spin = QDoubleSpinBox()
        self.mill_rotor_diameter_spin.setRange(0.05, 0.50)
        self.mill_rotor_diameter_spin.setValue(0.20)
        self.mill_rotor_diameter_spin.setDecimals(3)
        self.mill_rotor_diameter_spin.setSingleStep(0.01)
        self.mill_rotor_diameter_spin.setSuffix("  m")
        f.addRow("Rotor diameter:", self.mill_rotor_diameter_spin)

        self.mill_rotor_length_spin = QDoubleSpinBox()
        self.mill_rotor_length_spin.setRange(0.10, 1.00)
        self.mill_rotor_length_spin.setValue(0.30)
        self.mill_rotor_length_spin.setDecimals(3)
        self.mill_rotor_length_spin.setSingleStep(0.01)
        self.mill_rotor_length_spin.setSuffix("  m")
        f.addRow("Rotor length:", self.mill_rotor_length_spin)

        self.mill_shaft_diameter_spin = QDoubleSpinBox()
        self.mill_shaft_diameter_spin.setRange(0.02, 0.15)
        self.mill_shaft_diameter_spin.setValue(0.05)
        self.mill_shaft_diameter_spin.setDecimals(3)
        self.mill_shaft_diameter_spin.setSingleStep(0.005)
        self.mill_shaft_diameter_spin.setSuffix("  m")
        f.addRow("Shaft diameter:", self.mill_shaft_diameter_spin)

        self.mill_motor_power_spin = QDoubleSpinBox()
        self.mill_motor_power_spin.setRange(5, 100)
        self.mill_motor_power_spin.setValue(22.0)
        self.mill_motor_power_spin.setDecimals(1)
        self.mill_motor_power_spin.setSuffix("  kW")
        f.addRow("Motor power:", self.mill_motor_power_spin)

        clayout.addWidget(g)

        # --- Group 2: Hammers ---
        g = QGroupBox("Hammers")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self.mill_hammer_rows_spin = QSpinBox()
        self.mill_hammer_rows_spin.setRange(1, 12)
        self.mill_hammer_rows_spin.setValue(4)
        f.addRow("Hammer rows:", self.mill_hammer_rows_spin)

        self.mill_hammers_per_row_spin = QSpinBox()
        self.mill_hammers_per_row_spin.setRange(2, 8)
        self.mill_hammers_per_row_spin.setValue(4)
        f.addRow("Hammers per row:", self.mill_hammers_per_row_spin)

        self.mill_hammer_mass_spin = QDoubleSpinBox()
        self.mill_hammer_mass_spin.setRange(0.05, 2.0)
        self.mill_hammer_mass_spin.setValue(0.35)
        self.mill_hammer_mass_spin.setDecimals(3)
        self.mill_hammer_mass_spin.setSingleStep(0.05)
        self.mill_hammer_mass_spin.setSuffix("  kg")
        f.addRow("Hammer mass:", self.mill_hammer_mass_spin)

        self.mill_hammer_length_spin = QDoubleSpinBox()
        self.mill_hammer_length_spin.setRange(0.02, 0.20)
        self.mill_hammer_length_spin.setValue(0.08)
        self.mill_hammer_length_spin.setDecimals(3)
        self.mill_hammer_length_spin.setSingleStep(0.005)
        self.mill_hammer_length_spin.setSuffix("  m")
        f.addRow("Hammer length:", self.mill_hammer_length_spin)

        self.mill_hammer_width_spin = QDoubleSpinBox()
        self.mill_hammer_width_spin.setRange(0.01, 0.15)
        self.mill_hammer_width_spin.setValue(0.05)
        self.mill_hammer_width_spin.setDecimals(3)
        self.mill_hammer_width_spin.setSingleStep(0.005)
        self.mill_hammer_width_spin.setSuffix("  m")
        f.addRow("Hammer width:", self.mill_hammer_width_spin)

        self.mill_hammer_thickness_spin = QDoubleSpinBox()
        self.mill_hammer_thickness_spin.setRange(0.002, 0.020)
        self.mill_hammer_thickness_spin.setValue(0.008)
        self.mill_hammer_thickness_spin.setDecimals(3)
        self.mill_hammer_thickness_spin.setSingleStep(0.001)
        self.mill_hammer_thickness_spin.setSuffix("  m")
        f.addRow("Hammer thickness:", self.mill_hammer_thickness_spin)

        self.mill_hammer_clearance_spin = QDoubleSpinBox()
        self.mill_hammer_clearance_spin.setRange(0.002, 0.030)
        self.mill_hammer_clearance_spin.setValue(0.008)
        self.mill_hammer_clearance_spin.setDecimals(3)
        self.mill_hammer_clearance_spin.setSingleStep(0.001)
        self.mill_hammer_clearance_spin.setSuffix("  m")
        f.addRow("Hammer clearance:", self.mill_hammer_clearance_spin)

        clayout.addWidget(g)

        # --- Group 3: Screen ---
        g = QGroupBox("Screen")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self.mill_screen_aperture_spin = QDoubleSpinBox()
        self.mill_screen_aperture_spin.setRange(0.3, 2.0)  # Food powder grade
        self.mill_screen_aperture_spin.setValue(0.5)  # 0.5 mm for protein separation
        self.mill_screen_aperture_spin.setDecimals(2)
        self.mill_screen_aperture_spin.setSingleStep(0.1)
        self.mill_screen_aperture_spin.setSuffix("  mm")
        f.addRow("Aperture:", self.mill_screen_aperture_spin)

        self.mill_screen_open_area_spin = QDoubleSpinBox()
        self.mill_screen_open_area_spin.setRange(0.10, 0.80)
        self.mill_screen_open_area_spin.setValue(0.40)
        self.mill_screen_open_area_spin.setDecimals(2)
        self.mill_screen_open_area_spin.setSingleStep(0.05)
        f.addRow("Open area fraction:", self.mill_screen_open_area_spin)

        self.mill_screen_inner_radius_spin = QDoubleSpinBox()
        self.mill_screen_inner_radius_spin.setRange(0.05, 0.50)
        self.mill_screen_inner_radius_spin.setValue(0.188)
        self.mill_screen_inner_radius_spin.setDecimals(3)
        self.mill_screen_inner_radius_spin.setSingleStep(0.005)
        self.mill_screen_inner_radius_spin.setSuffix("  m")
        f.addRow("Inner radius:", self.mill_screen_inner_radius_spin)

        self.mill_screen_thickness_spin = QDoubleSpinBox()
        self.mill_screen_thickness_spin.setRange(0.001, 0.010)
        self.mill_screen_thickness_spin.setValue(0.003)
        self.mill_screen_thickness_spin.setDecimals(3)
        self.mill_screen_thickness_spin.setSingleStep(0.001)
        self.mill_screen_thickness_spin.setSuffix("  m")
        f.addRow("Thickness:", self.mill_screen_thickness_spin)

        clayout.addWidget(g)

        # --- Group 4: Housing ---
        g = QGroupBox("Housing")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self.mill_housing_inner_radius_spin = QDoubleSpinBox()
        self.mill_housing_inner_radius_spin.setRange(0.05, 0.60)
        self.mill_housing_inner_radius_spin.setValue(0.20)
        self.mill_housing_inner_radius_spin.setDecimals(3)
        self.mill_housing_inner_radius_spin.setSingleStep(0.01)
        self.mill_housing_inner_radius_spin.setSuffix("  m")
        f.addRow("Inner radius:", self.mill_housing_inner_radius_spin)

        self.mill_housing_length_spin = QDoubleSpinBox()
        self.mill_housing_length_spin.setRange(0.10, 1.50)
        self.mill_housing_length_spin.setValue(0.40)
        self.mill_housing_length_spin.setDecimals(3)
        self.mill_housing_length_spin.setSingleStep(0.01)
        self.mill_housing_length_spin.setSuffix("  m")
        f.addRow("Length:", self.mill_housing_length_spin)

        self.mill_housing_wall_spin = QDoubleSpinBox()
        self.mill_housing_wall_spin.setRange(0.002, 0.020)
        self.mill_housing_wall_spin.setValue(0.008)
        self.mill_housing_wall_spin.setDecimals(3)
        self.mill_housing_wall_spin.setSingleStep(0.001)
        self.mill_housing_wall_spin.setSuffix("  m")
        f.addRow("Wall thickness:", self.mill_housing_wall_spin)

        clayout.addWidget(g)

        # --- Group 5: Feed & Discharge ---
        g = QGroupBox("Feed && Discharge")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self.mill_seeds_feed_mass_spin = QDoubleSpinBox()
        self.mill_seeds_feed_mass_spin.setRange(0, 10000)
        self.mill_seeds_feed_mass_spin.setValue(0)
        self.mill_seeds_feed_mass_spin.setDecimals(2)
        self.mill_seeds_feed_mass_spin.setSuffix("  kg")
        self.mill_seeds_feed_mass_spin.setSpecialValueText("Continuous")
        self.mill_seeds_feed_mass_spin.setToolTip("Total mass of seeds (yellow peas) to feed into the mill; 0 = continuous")
        f.addRow("Seeds feed mass:", self.mill_seeds_feed_mass_spin)

        self.mill_feed_rate_spin = QDoubleSpinBox()
        self.mill_feed_rate_spin.setRange(10, 2000)
        self.mill_feed_rate_spin.setValue(500)
        self.mill_feed_rate_spin.setDecimals(0)
        self.mill_feed_rate_spin.setSuffix("  kg/h")
        f.addRow("Feed rate:", self.mill_feed_rate_spin)

        self.mill_feed_chute_width_spin = QDoubleSpinBox()
        self.mill_feed_chute_width_spin.setRange(0.05, 0.40)
        self.mill_feed_chute_width_spin.setValue(0.15)
        self.mill_feed_chute_width_spin.setDecimals(3)
        self.mill_feed_chute_width_spin.setSingleStep(0.01)
        self.mill_feed_chute_width_spin.setSuffix("  m")
        f.addRow("Feed chute width:", self.mill_feed_chute_width_spin)

        self.mill_feed_chute_height_spin = QDoubleSpinBox()
        self.mill_feed_chute_height_spin.setRange(0.05, 0.30)
        self.mill_feed_chute_height_spin.setValue(0.12)
        self.mill_feed_chute_height_spin.setDecimals(3)
        self.mill_feed_chute_height_spin.setSingleStep(0.01)
        self.mill_feed_chute_height_spin.setSuffix("  m")
        f.addRow("Feed chute height:", self.mill_feed_chute_height_spin)

        self.mill_discharge_width_spin = QDoubleSpinBox()
        self.mill_discharge_width_spin.setRange(0.05, 0.50)
        self.mill_discharge_width_spin.setValue(0.20)
        self.mill_discharge_width_spin.setDecimals(3)
        self.mill_discharge_width_spin.setSingleStep(0.01)
        self.mill_discharge_width_spin.setSuffix("  m")
        f.addRow("Discharge chute width:", self.mill_discharge_width_spin)

        self.mill_discharge_height_spin = QDoubleSpinBox()
        self.mill_discharge_height_spin.setRange(0.05, 0.40)
        self.mill_discharge_height_spin.setValue(0.15)
        self.mill_discharge_height_spin.setDecimals(3)
        self.mill_discharge_height_spin.setSingleStep(0.01)
        self.mill_discharge_height_spin.setSuffix("  m")
        f.addRow("Discharge chute height:", self.mill_discharge_height_spin)

        clayout.addWidget(g)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Apply & Build")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> Dict[str, Any]:
        return {
            "enable_milling": True,
            # Rotor & Drive
            "mill_rotor_rpm": self.mill_rotor_rpm_spin.value(),
            "mill_rotor_diameter_m": self.mill_rotor_diameter_spin.value(),
            "mill_rotor_length_m": self.mill_rotor_length_spin.value(),
            "mill_shaft_diameter_m": self.mill_shaft_diameter_spin.value(),
            "mill_motor_power_kw": self.mill_motor_power_spin.value(),
            # Hammers
            "mill_hammer_rows": self.mill_hammer_rows_spin.value(),
            "mill_hammers_per_row": self.mill_hammers_per_row_spin.value(),
            "mill_hammer_mass_kg": self.mill_hammer_mass_spin.value(),
            "mill_hammer_length_m": self.mill_hammer_length_spin.value(),
            "mill_hammer_width_m": self.mill_hammer_width_spin.value(),
            "mill_hammer_thickness_m": self.mill_hammer_thickness_spin.value(),
            "mill_hammer_clearance_m": self.mill_hammer_clearance_spin.value(),
            # Screen
            "mill_screen_aperture_mm": self.mill_screen_aperture_spin.value(),
            "mill_screen_open_area": self.mill_screen_open_area_spin.value(),
            "mill_screen_inner_radius_m": self.mill_screen_inner_radius_spin.value(),
            "mill_screen_thickness_m": self.mill_screen_thickness_spin.value(),
            # Housing
            "mill_housing_inner_radius_m": self.mill_housing_inner_radius_spin.value(),
            "mill_housing_length_m": self.mill_housing_length_spin.value(),
            "mill_housing_wall_thickness_m": self.mill_housing_wall_spin.value(),
            # Feed & Discharge
            "mill_seeds_feed_mass_kg": self.mill_seeds_feed_mass_spin.value(),
            "mill_feed_rate_kg_per_hr": self.mill_feed_rate_spin.value(),
            "mill_feed_chute_width_m": self.mill_feed_chute_width_spin.value(),
            "mill_feed_chute_height_m": self.mill_feed_chute_height_spin.value(),
            "mill_discharge_chute_width_m": self.mill_discharge_width_spin.value(),
            "mill_discharge_chute_height_m": self.mill_discharge_height_spin.value(),
        }

    def load_params(self, p: Dict[str, Any]):
        if not p:
            return
        _load = {
            "mill_rotor_rpm": self.mill_rotor_rpm_spin,
            "mill_rotor_diameter_m": self.mill_rotor_diameter_spin,
            "mill_rotor_length_m": self.mill_rotor_length_spin,
            "mill_shaft_diameter_m": self.mill_shaft_diameter_spin,
            "mill_motor_power_kw": self.mill_motor_power_spin,
            "mill_hammer_mass_kg": self.mill_hammer_mass_spin,
            "mill_hammer_length_m": self.mill_hammer_length_spin,
            "mill_hammer_width_m": self.mill_hammer_width_spin,
            "mill_hammer_thickness_m": self.mill_hammer_thickness_spin,
            "mill_hammer_clearance_m": self.mill_hammer_clearance_spin,
            "mill_screen_aperture_mm": self.mill_screen_aperture_spin,
            "mill_screen_open_area": self.mill_screen_open_area_spin,
            "mill_screen_inner_radius_m": self.mill_screen_inner_radius_spin,
            "mill_screen_thickness_m": self.mill_screen_thickness_spin,
            "mill_housing_inner_radius_m": self.mill_housing_inner_radius_spin,
            "mill_housing_length_m": self.mill_housing_length_spin,
            "mill_housing_wall_thickness_m": self.mill_housing_wall_spin,
            "mill_seeds_feed_mass_kg": self.mill_seeds_feed_mass_spin,
            "mill_feed_rate_kg_per_hr": self.mill_feed_rate_spin,
            "mill_feed_chute_width_m": self.mill_feed_chute_width_spin,
            "mill_feed_chute_height_m": self.mill_feed_chute_height_spin,
            "mill_discharge_chute_width_m": self.mill_discharge_width_spin,
            "mill_discharge_chute_height_m": self.mill_discharge_height_spin,
        }
        for key, spin in _load.items():
            if key in p:
                spin.setValue(p[key])
        # QSpinBox (int) entries
        if "mill_hammer_rows" in p:
            self.mill_hammer_rows_spin.setValue(int(p["mill_hammer_rows"]))
        if "mill_hammers_per_row" in p:
            self.mill_hammers_per_row_spin.setValue(int(p["mill_hammers_per_row"]))

    def _on_apply(self):
        self.milling_configured.emit(self.get_params())
        self.accept()
