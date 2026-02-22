"""
Milling configuration dialog — Hammer / pin mill only.
Per-system config: rotor, screen, feed rate (no pretreatment/classification).
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDialogButtonBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Signal

from ..theme import COLORS


class MillingConfigDialog(QDialog):
    """Configure Hammer Mill (pin mill) only. Emits params for this system."""

    milling_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure Pin Mill (Hammer Mill)")
        self.setMinimumSize(400, 320)
        self._params = current_params or {}
        self._setup_ui()
        self.load_params(self._params)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Pin Mill — Hammer mill (impact + screen classification)")
        header.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Configure rotor speed, screen aperture, and feed rate. Build and run from the Milling page.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        _M = (10, 14, 10, 10)

        g = QGroupBox("Mill Operating Point")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.mill_rotor_rpm_spin = QDoubleSpinBox()
        self.mill_rotor_rpm_spin.setRange(500, 6000)
        self.mill_rotor_rpm_spin.setValue(3000)
        self.mill_rotor_rpm_spin.setDecimals(0)
        self.mill_rotor_rpm_spin.setSuffix("  rpm")
        f.addRow("Rotor RPM:", self.mill_rotor_rpm_spin)
        self.mill_screen_aperture_spin = QDoubleSpinBox()
        self.mill_screen_aperture_spin.setRange(0.5, 5.0)
        self.mill_screen_aperture_spin.setValue(1.5)
        self.mill_screen_aperture_spin.setDecimals(2)
        self.mill_screen_aperture_spin.setSuffix("  mm")
        f.addRow("Screen aperture:", self.mill_screen_aperture_spin)
        self.mill_feed_rate_spin = QDoubleSpinBox()
        self.mill_feed_rate_spin.setRange(10, 2000)
        self.mill_feed_rate_spin.setValue(500)
        self.mill_feed_rate_spin.setDecimals(0)
        self.mill_feed_rate_spin.setSuffix("  kg/h")
        f.addRow("Feed rate:", self.mill_feed_rate_spin)
        layout.addWidget(g)

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
            "mill_rotor_rpm": self.mill_rotor_rpm_spin.value(),
            "mill_screen_aperture_mm": self.mill_screen_aperture_spin.value(),
            "mill_feed_rate_kg_per_hr": self.mill_feed_rate_spin.value(),
        }

    def load_params(self, p: Dict[str, Any]):
        if not p:
            return
        if "mill_rotor_rpm" in p:
            self.mill_rotor_rpm_spin.setValue(p["mill_rotor_rpm"])
        if "mill_screen_aperture_mm" in p:
            self.mill_screen_aperture_spin.setValue(p["mill_screen_aperture_mm"])
        if "mill_feed_rate_kg_per_hr" in p:
            self.mill_feed_rate_spin.setValue(p["mill_feed_rate_kg_per_hr"])

    def _on_apply(self):
        self.milling_configured.emit(self.get_params())
        self.accept()
