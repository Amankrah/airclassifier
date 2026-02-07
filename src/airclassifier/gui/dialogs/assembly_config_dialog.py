"""
Assembly Configuration Dialog
==============================

Separate window for configuring the air classifier assembly.
Parameters-driven, matching the real geometry assemblies:

- ClassificationSystemParams  (classification.py)
- CompleteSystemParams         (complete_system.py)

The assemblies are fully programmatic -- components and connections are
determined by parameters, not by drag-and-drop.  This dialog lets the
user set those parameters and immediately preview the resulting geometry.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, fields, asdict

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QDialogButtonBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QMessageBox, QSplitter,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..theme import COLORS


def _scrollable(w: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setWidget(w)
    return s


# --------------------------------------------------------------------------
# Flow diagram widget -- shows the current assembly topology as text
# --------------------------------------------------------------------------

class _FlowDiagram(QFrame):
    """Shows the component flow path as a styled text block."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Flow Path")
        title.setStyleSheet(
            f"font-weight: 600; font-size: 10pt; color: {COLORS.TEXT_PRIMARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(title)

        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setStyleSheet(
            f"font-family: 'Cascadia Code','Consolas',monospace; font-size: 9pt;"
            f" color: {COLORS.TEXT_SECONDARY}; border: none; background: transparent;"
        )
        layout.addWidget(self._text)

    def set_preclassification(self, enabled: bool, include_feed: bool, include_air: bool):
        if enabled:
            lines = []
            if include_air:
                lines.append("Air Filter \u2192 Blower \u2192 Dampers \u2192 Ductwork")
                lines.append("  \u2514\u2192 Venturi (air inlet)")
            if include_feed:
                lines.append("Hopper \u2192 Airlock \u2192 Screw \u2192 Deagglomerator")
                lines.append("  \u2514\u2192 Venturi (solids inlet)")
            lines += [
                "",
                "Venturi \u2192 Transition \u2192 Zigzag Classifier",
                "  \u251c\u2192 Fines \u2192 Wheel Classifier",
                "  \u2502     \u251c\u2192 Fines \u2192 Cyclone 1 \u2192 Cy2 \u2192 Cy3 (protein)",
                "  \u2502     \u2514\u2192 Coarse \u2192 Wheel coarse hopper",
                "  \u2514\u2192 Coarse \u2192 Zigzag coarse airlock",
                "",
                "Cyclone overflow \u2192 Bag Filter \u2192 Clean Air",
            ]
        else:
            lines = []
            if include_air:
                lines.append("Air Filter \u2192 Blower \u2192 Dampers \u2192 Ductwork")
                lines.append("  \u2514\u2192 Junction (air port)")
            if include_feed:
                lines.append("Hopper \u2192 Airlock \u2192 Screw \u2192 Deagglomerator")
                lines.append("  \u2514\u2192 Junction (solids chute 15\u00b0)")
            lines += [
                "",
                "3-Point Junction \u2192 Wheel Classifier",
                "  \u251c\u2192 Fines \u2192 Cyclone 1 \u2192 Cy2 \u2192 Cy3 (protein)",
                "  \u2514\u2192 Coarse \u2192 Wheel coarse hopper",
                "",
                "Cyclone overflow \u2192 Bag Filter \u2192 Clean Air",
            ]
        self._text.setText("\n".join(lines))


# --------------------------------------------------------------------------
# Main dialog
# --------------------------------------------------------------------------

class AssemblyConfigDialog(QDialog):
    """
    Dialog for configuring the air classifier assembly.

    Maps directly to ClassificationSystemParams + CompleteSystemParams.
    Produces a params dict that can be used to build the assembly.
    """

    # Emitted with the resulting params when user clicks Build & Apply
    assembly_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Assembly Configuration")
        self.setMinimumSize(780, 620)
        self.resize(850, 700)

        self._params = current_params or {}
        self._setup_ui()
        self._update_flow_diagram()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QLabel("Configure Air Classifier Assembly")
        header.setStyleSheet(f"font-size: 14pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Parameters drive the geometry -- components and connections are built automatically.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED}; margin-bottom: 8px;")
        layout.addWidget(sub)

        # Splitter: left = tabs, right = flow diagram
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left: parameter tabs
        tabs = QTabWidget()
        tabs.addTab(_scrollable(self._create_system_tab()), "System")
        tabs.addTab(_scrollable(self._create_classification_tab()), "Classification")
        tabs.addTab(_scrollable(self._create_sizing_tab()), "Sizing")
        splitter.addWidget(tabs)

        # Right: flow diagram
        self._flow_diagram = _FlowDiagram()
        self._flow_diagram.setMinimumWidth(280)
        splitter.addWidget(self._flow_diagram)

        splitter.setSizes([480, 300])
        layout.addWidget(splitter, 1)

        # Buttons
        btn_layout = QHBoxLayout()

        self._build_btn = QPushButton("Build && Preview")
        self._build_btn.setProperty("cssClass", "primary")
        self._build_btn.setMinimumHeight(36)
        self._build_btn.clicked.connect(self._on_build_preview)
        btn_layout.addWidget(self._build_btn)

        self._apply_btn = QPushButton("Apply && Close")
        self._apply_btn.setProperty("cssClass", "success")
        self._apply_btn.setMinimumHeight(36)
        self._apply_btn.clicked.connect(self._on_apply_close)
        btn_layout.addWidget(self._apply_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    # ================================================================
    # System tab
    # ================================================================

    def _create_system_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Mode
        g = QGroupBox("Classification Mode")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.mode_combo.setToolTip(
            "Full System: Venturi entrainment + zigzag pre-classification + wheel\n"
            "Wheel-Only: 3-point junction (air + solids chute) \u2192 wheel"
        )
        self.mode_combo.currentIndexChanged.connect(self._update_flow_diagram)
        f.addRow("Mode:", self.mode_combo)
        layout.addWidget(g)

        # Subsystems
        g = QGroupBox("Subsystems")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.chk_feed = QCheckBox("Feed System (Hopper \u2192 Airlock \u2192 Screw \u2192 Deagglomerator)")
        self.chk_feed.setChecked(True)
        self.chk_feed.stateChanged.connect(self._update_flow_diagram)
        f.addRow(self.chk_feed)

        self.chk_air = QCheckBox("Air System (Filter \u2192 Blower \u2192 Dampers)")
        self.chk_air.setChecked(True)
        self.chk_air.stateChanged.connect(self._update_flow_diagram)
        f.addRow(self.chk_air)

        self.chk_exhaust = QCheckBox("Exhaust (Silencer + Stack)")
        self.chk_exhaust.setChecked(True)
        f.addRow(self.chk_exhaust)

        self.chk_ductwork = QCheckBox("Connecting Ductwork")
        self.chk_ductwork.setChecked(True)
        f.addRow(self.chk_ductwork)

        self.chk_dropout = QCheckBox("Coarse Dropout Hopper (on venturi-to-zigzag transition)")
        self.chk_dropout.setChecked(True)
        f.addRow(self.chk_dropout)

        self.chk_coarse_collect = QCheckBox("Coarse Collection Airlocks")
        self.chk_coarse_collect.setChecked(True)
        f.addRow(self.chk_coarse_collect)

        layout.addWidget(g)

        # Design operating point
        g = QGroupBox("Design Operating Point")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.throughput_spin = QDoubleSpinBox()
        self.throughput_spin.setRange(10, 5000)
        self.throughput_spin.setValue(500)
        self.throughput_spin.setSuffix("  kg/h")
        f.addRow("Throughput:", self.throughput_spin)

        self.air_flow_h_spin = QDoubleSpinBox()
        self.air_flow_h_spin.setRange(10, 10000)
        self.air_flow_h_spin.setValue(3000)
        self.air_flow_h_spin.setSuffix("  m\u00b3/h")
        self.air_flow_h_spin.setToolTip("Design air flow rate for geometry sizing")
        f.addRow("Design Air Flow:", self.air_flow_h_spin)

        layout.addWidget(g)

        return w

    # ================================================================
    # Classification tab
    # ================================================================

    def _create_classification_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Venturi
        g = QGroupBox("Venturi Eductor")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.venturi_inlet_spin = QDoubleSpinBox()
        self.venturi_inlet_spin.setRange(20, 300)
        self.venturi_inlet_spin.setValue(80)
        self.venturi_inlet_spin.setSuffix("  mm")
        f.addRow("Inlet Diameter:", self.venturi_inlet_spin)

        self.venturi_throat_spin = QDoubleSpinBox()
        self.venturi_throat_spin.setRange(0.1, 1.0)
        self.venturi_throat_spin.setDecimals(2)
        self.venturi_throat_spin.setSingleStep(0.05)
        self.venturi_throat_spin.setValue(0.5)
        self.venturi_throat_spin.setToolTip("Throat diameter = Inlet \u00d7 Ratio. 0.5 = 40mm for 80mm inlet")
        f.addRow("Throat Ratio:", self.venturi_throat_spin)

        layout.addWidget(g)

        # Zigzag
        g = QGroupBox("Zigzag Classifier")
        f = QFormLayout(g); f.setContentsMargins(*_M)

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

        # Wheel
        g = QGroupBox("Wheel Classifier (Centrifugal)")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.wheel_dia_spin = QDoubleSpinBox()
        self.wheel_dia_spin.setRange(50, 500)
        self.wheel_dia_spin.setValue(200)
        self.wheel_dia_spin.setSuffix("  mm")
        f.addRow("Wheel Diameter:", self.wheel_dia_spin)

        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(500, 20000)
        self.wheel_rpm_spin.setValue(8000)
        self.wheel_rpm_spin.setDecimals(0)
        self.wheel_rpm_spin.setSuffix("  RPM")
        f.addRow("Wheel Speed:", self.wheel_rpm_spin)

        layout.addWidget(g)

        # Cyclones
        g = QGroupBox("Multi-Cyclone System (3 stages)")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.cy1_spin = QDoubleSpinBox()
        self.cy1_spin.setRange(100, 600)
        self.cy1_spin.setValue(300)
        self.cy1_spin.setSuffix("  mm")
        f.addRow("Primary \u00d8:", self.cy1_spin)

        self.cy2_spin = QDoubleSpinBox()
        self.cy2_spin.setRange(50, 400)
        self.cy2_spin.setValue(200)
        self.cy2_spin.setSuffix("  mm")
        f.addRow("Secondary \u00d8:", self.cy2_spin)

        self.cy3_spin = QDoubleSpinBox()
        self.cy3_spin.setRange(50, 300)
        self.cy3_spin.setValue(120)
        self.cy3_spin.setSuffix("  mm")
        f.addRow("Tertiary \u00d8:", self.cy3_spin)

        layout.addWidget(g)

        # Bag filter
        g = QGroupBox("Bag Filter")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.bag_flow_spin = QDoubleSpinBox()
        self.bag_flow_spin.setRange(0.01, 2.0)
        self.bag_flow_spin.setDecimals(2)
        self.bag_flow_spin.setValue(0.15)
        self.bag_flow_spin.setSuffix("  m\u00b3/s")
        f.addRow("Design Flow:", self.bag_flow_spin)

        layout.addWidget(g)

        return w

    # ================================================================
    # Sizing tab
    # ================================================================

    def _create_sizing_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        g = QGroupBox("Feed System Sizing")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.hopper_dia_spin = QDoubleSpinBox()
        self.hopper_dia_spin.setRange(200, 2000)
        self.hopper_dia_spin.setValue(600)
        self.hopper_dia_spin.setSuffix("  mm")
        f.addRow("Hopper Diameter:", self.hopper_dia_spin)

        layout.addWidget(g)

        g = QGroupBox("Ductwork Sizing")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.main_duct_spin = QDoubleSpinBox()
        self.main_duct_spin.setRange(50, 500)
        self.main_duct_spin.setValue(200)
        self.main_duct_spin.setSuffix("  mm")
        f.addRow("Main Duct \u00d8:", self.main_duct_spin)

        layout.addWidget(g)

        g = QGroupBox("Exhaust")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.stack_height_spin = QDoubleSpinBox()
        self.stack_height_spin.setRange(1, 20)
        self.stack_height_spin.setValue(4)
        self.stack_height_spin.setSuffix("  m")
        f.addRow("Stack Height:", self.stack_height_spin)

        layout.addWidget(g)

        return w

    # ================================================================
    # Flow diagram update
    # ================================================================

    def _update_flow_diagram(self):
        preclassification = self.mode_combo.currentIndex() == 0
        self._flow_diagram.set_preclassification(
            preclassification, self.chk_feed.isChecked(), self.chk_air.isChecked()
        )

    # ================================================================
    # Build params dict
    # ================================================================

    def get_params(self) -> Dict[str, Any]:
        """Build a params dict matching CompleteSystemParams + ClassificationSystemParams."""
        preclassification = self.mode_combo.currentIndex() == 0
        return {
            # System
            "use_preclassification": preclassification,
            "include_feed_system": self.chk_feed.isChecked(),
            "include_air_system": self.chk_air.isChecked(),
            "include_exhaust": self.chk_exhaust.isChecked(),
            "include_ductwork": self.chk_ductwork.isChecked(),
            "include_dropout": self.chk_dropout.isChecked(),
            "include_coarse_collection": self.chk_coarse_collect.isChecked(),
            "throughput_kg_h": self.throughput_spin.value(),
            "air_flow_m3_h": self.air_flow_h_spin.value(),
            # Classification
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
            # Sizing
            "hopper_diameter": self.hopper_dia_spin.value() / 1000.0,
            "main_duct_diameter": self.main_duct_spin.value() / 1000.0,
            "stack_height": self.stack_height_spin.value(),
        }

    def load_params(self, p: Dict[str, Any]):
        """Load existing params into the UI."""
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
        if "venturi_throat_ratio" in p:
            self.venturi_throat_spin.setValue(p["venturi_throat_ratio"])
        if "zigzag_channel_width" in p:
            self.zz_width_spin.setValue(p["zigzag_channel_width"] * 1000)
        if "zigzag_channel_depth" in p:
            self.zz_depth_spin.setValue(p["zigzag_channel_depth"] * 1000)
        if "wheel_diameter" in p:
            self.wheel_dia_spin.setValue(p["wheel_diameter"] * 1000)
        if "wheel_rpm" in p:
            self.wheel_rpm_spin.setValue(p["wheel_rpm"])
        self._update_flow_diagram()

    # ================================================================
    # Actions
    # ================================================================

    def _on_build_preview(self):
        """Build the assembly and emit for 3D preview (don't close dialog)."""
        self.assembly_configured.emit(self.get_params())

    def _on_apply_close(self):
        """Apply and close."""
        self.assembly_configured.emit(self.get_params())
        self.accept()
