"""
Pretreatment configuration dialog — GP-15 RF only.
Per-system config: feedstock, recipe, simulation (no classification/milling).
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QDialogButtonBox,
    QDoubleSpinBox, QCheckBox, QComboBox,
)
from PySide6.QtCore import Qt, Signal

from ..theme import COLORS


class PretreatmentConfigDialog(QDialog):
    """Configure RF Pretreatment (GP-15) only. Emits params for this system."""

    pretreatment_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure RF Pretreatment (GP-15)")
        self.setMinimumSize(420, 520)
        self._params = current_params or {}
        self._setup_ui()
        self.load_params(self._params)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("RF Pretreatment — GP-15 dielectric heating")
        header.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Configure feedstock, recipe, and simulation. Build and run from the Pretreatment page.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        _M = (10, 14, 10, 10)

        # Feedstock
        g = QGroupBox("Feedstock")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.pt_material_combo = QComboBox()
        self.pt_material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        f.addRow("Material preset:", self.pt_material_combo)
        self.pt_moisture_spin = QDoubleSpinBox()
        self.pt_moisture_spin.setRange(0.01, 0.25)
        self.pt_moisture_spin.setValue(0.10)
        self.pt_moisture_spin.setDecimals(3)
        self.pt_moisture_spin.setSingleStep(0.005)
        f.addRow("Inlet moisture (wb):", self.pt_moisture_spin)
        self.pt_target_spin = QDoubleSpinBox()
        self.pt_target_spin.setRange(0.01, 0.10)
        self.pt_target_spin.setValue(0.03)
        self.pt_target_spin.setDecimals(3)
        f.addRow("Target moisture (wb):", self.pt_target_spin)
        self.pt_bed_spin = QDoubleSpinBox()
        self.pt_bed_spin.setRange(10, 100)
        self.pt_bed_spin.setValue(40)
        self.pt_bed_spin.setSuffix("  mm")
        f.addRow("Bed depth:", self.pt_bed_spin)
        layout.addWidget(g)

        # Recipe
        g = QGroupBox("GP-15 Recipe")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.pt_gap_spin = QDoubleSpinBox()
        self.pt_gap_spin.setRange(20, 300)
        self.pt_gap_spin.setValue(80)
        self.pt_gap_spin.setSuffix("  mm")
        f.addRow("Electrode gap:", self.pt_gap_spin)
        self.pt_speed_spin = QDoubleSpinBox()
        self.pt_speed_spin.setRange(0.1, 2.0)
        self.pt_speed_spin.setValue(0.50)
        self.pt_speed_spin.setDecimals(2)
        self.pt_speed_spin.setSuffix("  m/min")
        f.addRow("Belt speed:", self.pt_speed_spin)
        self.pt_fan_spin = QDoubleSpinBox()
        self.pt_fan_spin.setRange(5, 60)
        self.pt_fan_spin.setValue(30)
        self.pt_fan_spin.setSuffix("  Hz")
        f.addRow("Extraction fan:", self.pt_fan_spin)
        self.pt_mrh_spin = QDoubleSpinBox()
        self.pt_mrh_spin.setRange(0.5, 3.0)
        self.pt_mrh_spin.setValue(2.6)
        self.pt_mrh_spin.setDecimals(2)
        self.pt_mrh_spin.setSuffix("  A")
        f.addRow("MRH threshold:", self.pt_mrh_spin)
        self.pt_heater_check = QCheckBox("Both banks on")
        self.pt_heater_check.setChecked(True)
        f.addRow("Heaters:", self.pt_heater_check)
        layout.addWidget(g)

        # Simulation
        g = QGroupBox("Simulation")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.pt_duration_spin = QDoubleSpinBox()
        self.pt_duration_spin.setRange(1, 3600)
        self.pt_duration_spin.setValue(120)
        self.pt_duration_spin.setSuffix("  s")
        f.addRow("Duration:", self.pt_duration_spin)
        self.pt_eff_spin = QDoubleSpinBox()
        self.pt_eff_spin.setRange(0.10, 1.0)
        self.pt_eff_spin.setValue(0.56)
        self.pt_eff_spin.setDecimals(2)
        f.addRow("Oscillator efficiency:", self.pt_eff_spin)
        layout.addWidget(g)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Apply & Build")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self) -> Dict[str, Any]:
        p = {
            "enable_pretreatment": True,
            "pt_material": self.pt_material_combo.currentText(),
            "pt_inlet_moisture": self.pt_moisture_spin.value(),
            "pt_target_moisture": self.pt_target_spin.value(),
            "pt_bed_depth_mm": self.pt_bed_spin.value(),
            "pt_electrode_gap_mm": self.pt_gap_spin.value(),
            "pt_belt_speed": self.pt_speed_spin.value(),
            "pt_extraction_fan_hz": self.pt_fan_spin.value(),
            "pt_mrh_amps": self.pt_mrh_spin.value(),
            "pt_heaters_on": self.pt_heater_check.isChecked(),
            "pt_duration_s": self.pt_duration_spin.value(),
            "pt_oscillator_efficiency": self.pt_eff_spin.value(),
        }
        return p

    def load_params(self, p: Dict[str, Any]):
        if not p:
            return
        if p.get("pt_material"):
            idx = self.pt_material_combo.findText(p["pt_material"])
            if idx >= 0:
                self.pt_material_combo.setCurrentIndex(idx)
        if "pt_inlet_moisture" in p:
            self.pt_moisture_spin.setValue(p["pt_inlet_moisture"])
        if "pt_target_moisture" in p:
            self.pt_target_spin.setValue(p["pt_target_moisture"])
        if "pt_bed_depth_mm" in p:
            self.pt_bed_spin.setValue(p["pt_bed_depth_mm"])
        if "pt_electrode_gap_mm" in p:
            self.pt_gap_spin.setValue(p["pt_electrode_gap_mm"])
        if "pt_belt_speed" in p:
            self.pt_speed_spin.setValue(p["pt_belt_speed"])
        if "pt_extraction_fan_hz" in p:
            self.pt_fan_spin.setValue(p["pt_extraction_fan_hz"])
        if "pt_mrh_amps" in p:
            self.pt_mrh_spin.setValue(p["pt_mrh_amps"])
        if "pt_heaters_on" in p:
            self.pt_heater_check.setChecked(p["pt_heaters_on"])
        if "pt_duration_s" in p:
            self.pt_duration_spin.setValue(p["pt_duration_s"])
        if "pt_oscillator_efficiency" in p:
            self.pt_eff_spin.setValue(p["pt_oscillator_efficiency"])

    def _on_apply(self):
        self.pretreatment_configured.emit(self.get_params())
        self.accept()
