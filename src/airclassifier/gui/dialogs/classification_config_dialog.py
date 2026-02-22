"""
Classification configuration dialog — Air classifier only.
Per-system config: mode, subsystems, geometry (no pretreatment/milling).
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QFormLayout, QScrollArea, QFrame,
    QGroupBox, QLabel, QPushButton, QDialogButtonBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
)
from PySide6.QtCore import Qt, Signal

from ..theme import COLORS


def _scrollable(w: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setWidget(w)
    return s


class ClassificationConfigDialog(QDialog):
    """Configure Air Classification only. Emits params for this system."""

    classification_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Configure Air Classification")
        self.setMinimumSize(480, 640)
        self._params = current_params or {}
        self._setup_ui()
        self.load_params(self._params)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QLabel("Air Classification — Zigzag + Wheel + Cyclones")
        header.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Configure mode, subsystems, and geometry. Build and run from the Classification page.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        scroll = _scrollable(self._build_content())
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Apply & Build")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_content(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Mode & subsystems
        g = QGroupBox("Mode & Subsystems")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        f.addRow("Mode:", self.mode_combo)
        self.chk_feed = QCheckBox("Feed System (Hopper → Airlock → Screw → Deagglomerator)")
        self.chk_feed.setChecked(True)
        f.addRow(self.chk_feed)
        self.chk_air = QCheckBox("Air System (Filter → Blower → Dampers)")
        self.chk_air.setChecked(True)
        f.addRow(self.chk_air)
        self.chk_exhaust = QCheckBox("Exhaust (Silencer + Stack)")
        self.chk_exhaust.setChecked(True)
        f.addRow(self.chk_exhaust)
        self.chk_ductwork = QCheckBox("Connecting Ductwork")
        self.chk_ductwork.setChecked(True)
        f.addRow(self.chk_ductwork)
        self.chk_dropout = QCheckBox("Coarse Dropout Hopper")
        self.chk_dropout.setChecked(True)
        f.addRow(self.chk_dropout)
        self.chk_coarse_collect = QCheckBox("Coarse Collection Airlocks")
        self.chk_coarse_collect.setChecked(True)
        f.addRow(self.chk_coarse_collect)
        layout.addWidget(g)

        g = QGroupBox("Design Operating Point")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.throughput_spin = QDoubleSpinBox()
        self.throughput_spin.setRange(10, 5000)
        self.throughput_spin.setValue(500)
        self.throughput_spin.setSuffix("  kg/h")
        f.addRow("Throughput:", self.throughput_spin)
        self.air_flow_h_spin = QDoubleSpinBox()
        self.air_flow_h_spin.setRange(10, 10000)
        self.air_flow_h_spin.setValue(3000)
        self.air_flow_h_spin.setSuffix("  m³/h")
        f.addRow("Design Air Flow:", self.air_flow_h_spin)
        layout.addWidget(g)

        # Venturi & Zigzag
        g = QGroupBox("Venturi Eductor")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.venturi_inlet_spin = QDoubleSpinBox()
        self.venturi_inlet_spin.setRange(20, 300)
        self.venturi_inlet_spin.setValue(80)
        self.venturi_inlet_spin.setSuffix("  mm")
        f.addRow("Inlet Diameter:", self.venturi_inlet_spin)
        self.venturi_throat_spin = QDoubleSpinBox()
        self.venturi_throat_spin.setRange(0.1, 1.0)
        self.venturi_throat_spin.setDecimals(2)
        self.venturi_throat_spin.setValue(0.5)
        f.addRow("Throat Ratio:", self.venturi_throat_spin)
        layout.addWidget(g)

        g = QGroupBox("Zigzag Classifier")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.zz_width_spin = QDoubleSpinBox()
        self.zz_width_spin.setRange(30, 500)
        self.zz_width_spin.setValue(150)
        self.zz_width_spin.setSuffix("  mm")
        f.addRow("Channel Width:", self.zz_width_spin)
        self.zz_depth_spin = QDoubleSpinBox()
        self.zz_depth_spin.setRange(30, 500)
        self.zz_depth_spin.setValue(250)
        self.zz_depth_spin.setSuffix("  mm")
        f.addRow("Channel Depth:", self.zz_depth_spin)
        self.zz_stages_spin = QSpinBox()
        self.zz_stages_spin.setRange(3, 12)
        self.zz_stages_spin.setValue(5)
        f.addRow("Number of Stages:", self.zz_stages_spin)
        layout.addWidget(g)

        # Wheel & Cyclones
        g = QGroupBox("Wheel Classifier")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.wheel_dia_spin = QDoubleSpinBox()
        self.wheel_dia_spin.setRange(50, 500)
        self.wheel_dia_spin.setValue(200)
        self.wheel_dia_spin.setSuffix("  mm")
        f.addRow("Wheel Diameter:", self.wheel_dia_spin)
        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(500, 20000)
        self.wheel_rpm_spin.setValue(975)
        self.wheel_rpm_spin.setDecimals(0)
        self.wheel_rpm_spin.setSuffix("  RPM")
        f.addRow("Wheel Speed:", self.wheel_rpm_spin)
        layout.addWidget(g)

        g = QGroupBox("Multi-Cyclone System (3 stages)")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.cy1_spin = QDoubleSpinBox()
        self.cy1_spin.setRange(100, 600)
        self.cy1_spin.setValue(300)
        self.cy1_spin.setSuffix("  mm")
        f.addRow("Primary ∅:", self.cy1_spin)
        self.cy2_spin = QDoubleSpinBox()
        self.cy2_spin.setRange(50, 400)
        self.cy2_spin.setValue(200)
        self.cy2_spin.setSuffix("  mm")
        f.addRow("Secondary ∅:", self.cy2_spin)
        self.cy3_spin = QDoubleSpinBox()
        self.cy3_spin.setRange(50, 300)
        self.cy3_spin.setValue(120)
        self.cy3_spin.setSuffix("  mm")
        f.addRow("Tertiary ∅:", self.cy3_spin)
        layout.addWidget(g)

        g = QGroupBox("Bag Filter")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.bag_flow_spin = QDoubleSpinBox()
        self.bag_flow_spin.setRange(0.01, 2.0)
        self.bag_flow_spin.setDecimals(2)
        self.bag_flow_spin.setValue(0.15)
        self.bag_flow_spin.setSuffix("  m³/s")
        f.addRow("Design Flow:", self.bag_flow_spin)
        layout.addWidget(g)

        # Sizing
        g = QGroupBox("Feed System Sizing")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.hopper_dia_spin = QDoubleSpinBox()
        self.hopper_dia_spin.setRange(200, 2000)
        self.hopper_dia_spin.setValue(600)
        self.hopper_dia_spin.setSuffix("  mm")
        f.addRow("Hopper Diameter:", self.hopper_dia_spin)
        layout.addWidget(g)

        g = QGroupBox("Ductwork & Exhaust")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)
        self.main_duct_spin = QDoubleSpinBox()
        self.main_duct_spin.setRange(50, 500)
        self.main_duct_spin.setValue(200)
        self.main_duct_spin.setSuffix("  mm")
        f.addRow("Main Duct ∅:", self.main_duct_spin)
        self.stack_height_spin = QDoubleSpinBox()
        self.stack_height_spin.setRange(1, 20)
        self.stack_height_spin.setValue(4)
        self.stack_height_spin.setSuffix("  m")
        f.addRow("Stack Height:", self.stack_height_spin)
        layout.addWidget(g)

        return w

    def get_params(self) -> Dict[str, Any]:
        preclassification = self.mode_combo.currentIndex() == 0
        return {
            "enable_classification": True,
            "use_preclassification": preclassification,
            "include_feed_system": self.chk_feed.isChecked(),
            "include_air_system": self.chk_air.isChecked(),
            "include_exhaust": self.chk_exhaust.isChecked(),
            "include_ductwork": self.chk_ductwork.isChecked(),
            "include_dropout": self.chk_dropout.isChecked(),
            "include_coarse_collection": self.chk_coarse_collect.isChecked(),
            "throughput_kg_h": self.throughput_spin.value(),
            "air_flow_m3_h": self.air_flow_h_spin.value(),
            "venturi_inlet_diameter": self.venturi_inlet_spin.value() / 1000.0,
            "venturi_throat_ratio": self.venturi_throat_spin.value(),
            "zigzag_channel_width": self.zz_width_spin.value() / 1000.0,
            "zigzag_channel_depth": self.zz_depth_spin.value() / 1000.0,
            "zigzag_num_stages": self.zz_stages_spin.value(),
            "wheel_diameter": self.wheel_dia_spin.value() / 1000.0,
            "wheel_rpm": self.wheel_rpm_spin.value(),
            "primary_cyclone_diameter": self.cy1_spin.value() / 1000.0,
            "secondary_cyclone_diameter": self.cy2_spin.value() / 1000.0,
            "tertiary_cyclone_diameter": self.cy3_spin.value() / 1000.0,
            "bag_filter_flow_rate": self.bag_flow_spin.value(),
            "hopper_diameter": self.hopper_dia_spin.value() / 1000.0,
            "main_duct_diameter": self.main_duct_spin.value() / 1000.0,
            "stack_height": self.stack_height_spin.value(),
        }

    def load_params(self, p: Dict[str, Any]):
        if not p:
            return
        if "use_preclassification" in p:
            self.mode_combo.setCurrentIndex(0 if p["use_preclassification"] else 1)
        if "include_feed_system" in p:
            self.chk_feed.setChecked(p["include_feed_system"])
        if "include_air_system" in p:
            self.chk_air.setChecked(p["include_air_system"])
        if "include_exhaust" in p:
            self.chk_exhaust.setChecked(p["include_exhaust"])
        if "include_ductwork" in p:
            self.chk_ductwork.setChecked(p["include_ductwork"])
        if "include_dropout" in p:
            self.chk_dropout.setChecked(p["include_dropout"])
        if "include_coarse_collection" in p:
            self.chk_coarse_collect.setChecked(p["include_coarse_collection"])
        if "throughput_kg_h" in p:
            self.throughput_spin.setValue(p["throughput_kg_h"])
        if "air_flow_m3_h" in p:
            self.air_flow_h_spin.setValue(p["air_flow_m3_h"])
        if "venturi_throat_ratio" in p:
            self.venturi_throat_spin.setValue(p["venturi_throat_ratio"])
        if "venturi_inlet_diameter" in p:
            self.venturi_inlet_spin.setValue(p["venturi_inlet_diameter"] * 1000)
        if "zigzag_channel_width" in p:
            self.zz_width_spin.setValue(p["zigzag_channel_width"] * 1000)
        if "zigzag_channel_depth" in p:
            self.zz_depth_spin.setValue(p["zigzag_channel_depth"] * 1000)
        if "zigzag_num_stages" in p:
            self.zz_stages_spin.setValue(p["zigzag_num_stages"])
        if "wheel_diameter" in p:
            self.wheel_dia_spin.setValue(p["wheel_diameter"] * 1000)
        if "wheel_rpm" in p:
            self.wheel_rpm_spin.setValue(p["wheel_rpm"])
        if "primary_cyclone_diameter" in p:
            self.cy1_spin.setValue(p["primary_cyclone_diameter"] * 1000)
        if "secondary_cyclone_diameter" in p:
            self.cy2_spin.setValue(p["secondary_cyclone_diameter"] * 1000)
        if "tertiary_cyclone_diameter" in p:
            self.cy3_spin.setValue(p["tertiary_cyclone_diameter"] * 1000)
        if "bag_filter_flow_rate" in p:
            self.bag_flow_spin.setValue(p["bag_filter_flow_rate"])
        if "hopper_diameter" in p:
            self.hopper_dia_spin.setValue(p["hopper_diameter"] * 1000)
        if "main_duct_diameter" in p:
            self.main_duct_spin.setValue(p["main_duct_diameter"] * 1000)
        if "stack_height" in p:
            self.stack_height_spin.setValue(p["stack_height"])

    def _on_apply(self):
        self.classification_configured.emit(self.get_params())
        self.accept()
