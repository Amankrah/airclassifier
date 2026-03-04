"""
Assembly Configuration Dialog
==============================

Process configuration for ProteinProcessIO.

Supports two process stages, independently enabled:
  1. **RF Pretreatment** — QMTI GP-15 RF dielectric heating
  2. **Air Classification** — Venturi/Zigzag/Wheel classifier

The dialog lets the user configure both stages and immediately
preview the resulting geometry in the 3D viewport.
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, fields, asdict

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QGroupBox, QLabel, QPushButton, QDialogButtonBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QFrame,
    QRadioButton, QButtonGroup,
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
# Flow diagram widget -- shows the process topology
# --------------------------------------------------------------------------

class _FlowDiagram(QFrame):
    """Shows the process flow path as a styled text block."""

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

        title = QLabel("Process Flow Path")
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

    def update_flow(
        self,
        pretreatment_enabled: bool,
        classification_enabled: bool,
        milling_enabled: bool,
        preclassification: bool,
        include_feed: bool,
        include_air: bool,
    ):
        lines = []

        # Stage 1: RF Pretreatment
        if pretreatment_enabled:
            lines += [
                "━━━ STAGE 1: RF PRETREATMENT (GP-15) ━━━",
                "",
                "Whole Seeds/Beans",
                "  └→ Hopper → Sizing Plate → Belt Infeed",
                "",
                "GP-15 Oven (27.12 MHz RF Field)",
                "  ├→ Upper Electrode (V = V_rf)",
                "  │     ╎  air gap",
                "  │     ╎  material bed (T, M fields)",
                "  │     ╎  belt (PTFE) + wear strips",
                "  └→ Lower Electrode (ground)",
                "",
                "EMU: Extraction Fan + Heaters → humid air out",
                "",
                "Belt Outfeed → Conditioned Material",
                f"  (moisture 8-14% → 2-4% wb)",
            ]

        # Connector to Milling
        if pretreatment_enabled and milling_enabled:
            lines += [
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "  │  conditioned material",
                "  ↓",
            ]

        # Stage 1b: Milling
        if milling_enabled:
            lines += [
                "",
                "━━━ PIN MILL (Hammer Mill) ━━━",
                "",
                "Conditioned / whole seeds → Feed chute → Mill chamber",
                "  Rotor + hammers → Impact + breakage",
                "  Screen → Milled product (PSD)",
                "",
            ]

        # Connector to Classification
        if (pretreatment_enabled or milling_enabled) and classification_enabled:
            lines += [
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "  │  milled flour",
                "  ↓",
            ]

        # Stage 2: Air Classification
        if classification_enabled:
            lines += ["", "━━━ STAGE 2: AIR CLASSIFICATION ━━━", ""]

            if preclassification:
                if include_air:
                    lines.append("Air Filter → Blower → Dampers → Ductwork")
                    lines.append("  └→ Venturi (air inlet)")
                if include_feed:
                    lines.append("Hopper → Airlock → Screw → Deagglomerator")
                    lines.append("  └→ Venturi (solids inlet)")
                lines += [
                    "",
                    "Venturi → Transition → Zigzag Classifier",
                    "  ├→ Fines → Wheel Classifier",
                    "  │     ├→ Fines → Cyclone 1 → Cy2 → Cy3 (protein)",
                    "  │     └→ Coarse → Wheel coarse hopper",
                    "  └→ Coarse → Zigzag coarse airlock",
                    "",
                    "Cyclone overflow → Bag Filter → Clean Air",
                ]
            else:
                if include_air:
                    lines.append("Air Filter → Blower → Dampers → Ductwork")
                    lines.append("  └→ Junction (air port)")
                if include_feed:
                    lines.append("Hopper → Airlock → Screw → Deagglomerator")
                    lines.append("  └→ Junction (solids chute 15°)")
                lines += [
                    "",
                    "3-Point Junction → Wheel Classifier",
                    "  ├→ Fines → Cyclone 1 → Cy2 → Cy3 (protein)",
                    "  └→ Coarse → Wheel coarse hopper",
                    "",
                    "Cyclone overflow → Bag Filter → Clean Air",
                ]

        if not pretreatment_enabled and not classification_enabled and not milling_enabled:
            lines = [
                "No process stages enabled.",
                "",
                "Enable at least one stage above.",
            ]

        self._text.setText("\n".join(lines))


# --------------------------------------------------------------------------
# Main dialog
# --------------------------------------------------------------------------

class AssemblyConfigDialog(QDialog):
    """
    Process configuration dialog for ProteinProcessIO.

    Supports independent RF Pretreatment and Air Classification stages.
    """

    assembly_configured = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Process Configuration — ProteinProcessIO")
        self.setMinimumSize(860, 680)
        self.resize(960, 760)

        self._params = current_params or {}
        self._setup_ui()
        self._connect_quick_settings()
        self._update_stage_visibility()
        self._update_flow_diagram()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QLabel("Configure Process Stages")
        header.setStyleSheet(f"font-size: 14pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(header)
        sub = QLabel("Enable and configure each process stage. The system builds automatically from parameters.")
        sub.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED}; margin-bottom: 8px;")
        layout.addWidget(sub)

        # Splitter: left = tabs, right = flow diagram
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left: parameter tabs
        tabs = QTabWidget()
        tabs.addTab(_scrollable(self._create_stages_tab()), "Stages")
        tabs.addTab(_scrollable(self._create_pretreatment_tab()), "RF Pretreatment")
        tabs.addTab(_scrollable(self._create_milling_tab()), "Milling")
        tabs.addTab(_scrollable(self._create_classification_tab()), "Classification")
        tabs.addTab(_scrollable(self._create_sizing_tab()), "Sizing")
        splitter.addWidget(tabs)

        # Right: flow diagram
        self._flow_diagram = _FlowDiagram()
        self._flow_diagram.setMinimumWidth(300)
        splitter.addWidget(self._flow_diagram)

        splitter.setSizes([520, 340])
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
    # Stages tab  (NEW — process stage selector)
    # ================================================================

    def _create_stages_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Process stage selector (mutually exclusive for now)
        g = QGroupBox("Active Process Stage")
        stage_layout = QVBoxLayout(g)
        stage_layout.setContentsMargins(*_M)

        self._stage_group = QButtonGroup(self)

        self.radio_pretreatment = QRadioButton(
            "RF Pretreatment — QMTI GP-15 (27.12 MHz dielectric heating)"
        )
        self.radio_pretreatment.setChecked(True)
        self._stage_group.addButton(self.radio_pretreatment, 0)
        stage_layout.addWidget(self.radio_pretreatment)

        hint_pre = QLabel("Moisture conditioning of whole seeds: 8–14% → 2–4% wb before milling")
        hint_pre.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; margin-left: 24px;")
        stage_layout.addWidget(hint_pre)

        stage_layout.addSpacing(6)

        self.radio_milling = QRadioButton(
            "Hammer Mill (impact milling, screen classification)"
        )
        self._stage_group.addButton(self.radio_milling, 1)
        stage_layout.addWidget(self.radio_milling)

        hint_mill = QLabel("Size reduction: conditioned seeds → milled flour (PSD for classifier)")
        hint_mill.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; margin-left: 24px;")
        stage_layout.addWidget(hint_mill)

        stage_layout.addSpacing(6)

        self.radio_classification = QRadioButton(
            "Air Classification — Zigzag + Wheel + Cyclones"
        )
        self._stage_group.addButton(self.radio_classification, 2)
        stage_layout.addWidget(self.radio_classification)

        hint_cls = QLabel("Protein-starch separation via particle size / density classification")
        hint_cls.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; margin-left: 24px;")
        stage_layout.addWidget(hint_cls)

        self._stage_group.idToggled.connect(self._on_stage_changed)

        layout.addWidget(g)

        # ---- Classification-specific settings (visible only when classification selected) ----

        # Classification mode
        self._classification_mode_group = QGroupBox("Classification Mode")
        f = QFormLayout(self._classification_mode_group); f.setContentsMargins(*_M)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.mode_combo.currentIndexChanged.connect(self._update_flow_diagram)
        f.addRow("Mode:", self.mode_combo)
        layout.addWidget(self._classification_mode_group)

        # Subsystems (classification)
        self._subsystems_group = QGroupBox("Subsystems")
        f = QFormLayout(self._subsystems_group); f.setContentsMargins(*_M)

        self.chk_feed = QCheckBox("Feed System (Hopper → Airlock → Screw → Deagglomerator)")
        self.chk_feed.setChecked(True)
        self.chk_feed.stateChanged.connect(self._update_flow_diagram)
        f.addRow(self.chk_feed)

        self.chk_air = QCheckBox("Air System (Filter → Blower → Dampers)")
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

        layout.addWidget(self._subsystems_group)

        # Design operating point (classification)
        self._operating_point_group = QGroupBox("Design Operating Point")
        f = QFormLayout(self._operating_point_group); f.setContentsMargins(*_M)

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

        layout.addWidget(self._operating_point_group)

        # ---- RF Pretreatment quick settings (visible only when pretreatment selected) ----
        self._pretreatment_quick_group = QGroupBox("RF Pretreatment Settings")
        f = QFormLayout(self._pretreatment_quick_group); f.setContentsMargins(*_M)

        self.pt_quick_material_combo = QComboBox()
        self.pt_quick_material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        f.addRow("Material preset:", self.pt_quick_material_combo)

        self.pt_quick_moisture_spin = QDoubleSpinBox()
        self.pt_quick_moisture_spin.setRange(0.01, 0.25)
        self.pt_quick_moisture_spin.setValue(0.10)
        self.pt_quick_moisture_spin.setDecimals(3)
        f.addRow("Inlet moisture (wb):", self.pt_quick_moisture_spin)

        self.pt_quick_target_spin = QDoubleSpinBox()
        self.pt_quick_target_spin.setRange(0.01, 0.10)
        self.pt_quick_target_spin.setValue(0.03)
        self.pt_quick_target_spin.setDecimals(3)
        f.addRow("Target moisture (wb):", self.pt_quick_target_spin)

        self.pt_quick_speed_spin = QDoubleSpinBox()
        self.pt_quick_speed_spin.setRange(0.1, 2.0)
        self.pt_quick_speed_spin.setValue(0.50)
        self.pt_quick_speed_spin.setDecimals(2)
        self.pt_quick_speed_spin.setSuffix("  m/min")
        f.addRow("Belt speed:", self.pt_quick_speed_spin)

        self.pt_quick_gap_spin = QDoubleSpinBox()
        self.pt_quick_gap_spin.setRange(20, 300)
        self.pt_quick_gap_spin.setValue(80)
        self.pt_quick_gap_spin.setSuffix("  mm")
        f.addRow("Electrode gap:", self.pt_quick_gap_spin)

        hint = QLabel("See RF Pretreatment tab for full recipe settings")
        hint.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        f.addRow("", hint)

        layout.addWidget(self._pretreatment_quick_group)
        self._pretreatment_quick_group.setVisible(False)

        # ---- Milling quick settings (visible only when milling selected) ----
        self._milling_quick_group = QGroupBox("Milling Settings")
        f = QFormLayout(self._milling_quick_group); f.setContentsMargins(*_M)

        self.mill_quick_rpm_spin = QDoubleSpinBox()
        self.mill_quick_rpm_spin.setRange(500, 6000)
        self.mill_quick_rpm_spin.setValue(3000)
        self.mill_quick_rpm_spin.setDecimals(0)
        self.mill_quick_rpm_spin.setSuffix("  rpm")
        f.addRow("Rotor RPM:", self.mill_quick_rpm_spin)

        self.mill_quick_screen_spin = QDoubleSpinBox()
        self.mill_quick_screen_spin.setRange(0.3, 2.0)  # Food powder grade
        self.mill_quick_screen_spin.setValue(0.5)  # 0.5 mm for protein separation
        self.mill_quick_screen_spin.setDecimals(2)
        self.mill_quick_screen_spin.setSingleStep(0.1)
        self.mill_quick_screen_spin.setSuffix("  mm")
        f.addRow("Screen aperture:", self.mill_quick_screen_spin)

        self.mill_quick_feed_spin = QDoubleSpinBox()
        self.mill_quick_feed_spin.setRange(10, 2000)
        self.mill_quick_feed_spin.setValue(500)
        self.mill_quick_feed_spin.setDecimals(0)
        self.mill_quick_feed_spin.setSuffix("  kg/h")
        f.addRow("Feed rate:", self.mill_quick_feed_spin)

        self.mill_quick_hammers_spin = QSpinBox()
        self.mill_quick_hammers_spin.setRange(4, 48)
        self.mill_quick_hammers_spin.setValue(16)
        self.mill_quick_hammers_spin.setReadOnly(True)
        self.mill_quick_hammers_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        f.addRow("Total hammers:", self.mill_quick_hammers_spin)

        hint = QLabel("See Milling tab for full geometry settings")
        hint.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        f.addRow("", hint)

        layout.addWidget(self._milling_quick_group)
        self._milling_quick_group.setVisible(False)

        # Initially show classification settings (default selection is pretreatment,
        # but we'll call _on_stage_changed to set correct visibility)
        self._update_stage_visibility()

        return w

    # ================================================================
    # RF Pretreatment tab  (NEW)
    # ================================================================

    def _create_pretreatment_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Material
        g = QGroupBox("Feedstock")
        f = QFormLayout(g); f.setContentsMargins(*_M)

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
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.pt_gap_spin = QDoubleSpinBox()
        self.pt_gap_spin.setRange(20, 300)
        self.pt_gap_spin.setValue(80)
        self.pt_gap_spin.setSuffix("  mm")
        f.addRow("Electrode gap:", self.pt_gap_spin)

        self.pt_speed_spin = QDoubleSpinBox()
        self.pt_speed_spin.setRange(0.1, 2.0)
        self.pt_speed_spin.setValue(0.50)
        self.pt_speed_spin.setDecimals(2)
        self.pt_speed_spin.setSingleStep(0.05)
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
        f = QFormLayout(g); f.setContentsMargins(*_M)

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

        hint = QLabel("Residence time at 0.5 m/min ≈ 180 s  |  Efficiency 0.56 matches GP-15 manual")
        hint.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        f.addRow("", hint)

        layout.addWidget(g)
        return w

    # ================================================================
    # Classification tab  (existing, unchanged)
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
        self.wheel_rpm_spin.setValue(975)
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

        # Bag filter
        g = QGroupBox("Bag Filter")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.bag_flow_spin = QDoubleSpinBox()
        self.bag_flow_spin.setRange(0.01, 2.0)
        self.bag_flow_spin.setDecimals(2)
        self.bag_flow_spin.setValue(0.15)
        self.bag_flow_spin.setSuffix("  m³/s")
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
        f.addRow("Main Duct ∅:", self.main_duct_spin)
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

    def _on_stage_changed(self, id_: int, checked: bool):
        """Toggle tab and flow diagram when stage selection changes."""
        if checked:
            self._update_stage_visibility()
            self._update_flow_diagram()

    def _update_stage_visibility(self):
        """Show/hide stage-specific settings based on selected process stage."""
        pretreatment_on = self.radio_pretreatment.isChecked()
        milling_on = self.radio_milling.isChecked()
        classification_on = self.radio_classification.isChecked()

        # Classification-specific groups
        self._classification_mode_group.setVisible(classification_on)
        self._subsystems_group.setVisible(classification_on)
        self._operating_point_group.setVisible(classification_on)

        # RF Pretreatment quick settings
        self._pretreatment_quick_group.setVisible(pretreatment_on)

        # Milling quick settings
        self._milling_quick_group.setVisible(milling_on)

    def _connect_quick_settings(self):
        """Connect quick settings to main settings for bidirectional sync."""
        # RF Pretreatment: quick <-> main
        self.pt_quick_material_combo.currentIndexChanged.connect(
            lambda i: self.pt_material_combo.setCurrentIndex(i)
        )
        self.pt_material_combo.currentIndexChanged.connect(
            lambda i: self.pt_quick_material_combo.setCurrentIndex(i)
        )

        self.pt_quick_moisture_spin.valueChanged.connect(self.pt_moisture_spin.setValue)
        self.pt_moisture_spin.valueChanged.connect(self.pt_quick_moisture_spin.setValue)

        self.pt_quick_target_spin.valueChanged.connect(self.pt_target_spin.setValue)
        self.pt_target_spin.valueChanged.connect(self.pt_quick_target_spin.setValue)

        self.pt_quick_speed_spin.valueChanged.connect(self.pt_speed_spin.setValue)
        self.pt_speed_spin.valueChanged.connect(self.pt_quick_speed_spin.setValue)

        self.pt_quick_gap_spin.valueChanged.connect(self.pt_gap_spin.setValue)
        self.pt_gap_spin.valueChanged.connect(self.pt_quick_gap_spin.setValue)

        # Milling: quick <-> main
        self.mill_quick_rpm_spin.valueChanged.connect(self.mill_rotor_rpm_spin.setValue)
        self.mill_rotor_rpm_spin.valueChanged.connect(self.mill_quick_rpm_spin.setValue)

        self.mill_quick_screen_spin.valueChanged.connect(self.mill_screen_aperture_spin.setValue)
        self.mill_screen_aperture_spin.valueChanged.connect(self.mill_quick_screen_spin.setValue)

        self.mill_quick_feed_spin.valueChanged.connect(self.mill_feed_rate_spin.setValue)
        self.mill_feed_rate_spin.valueChanged.connect(self.mill_quick_feed_spin.setValue)

        # Total hammers = rows * per_row (read-only sync from main to quick)
        def _update_total_hammers():
            total = self.mill_hammer_rows_spin.value() * self.mill_hammers_per_row_spin.value()
            self.mill_quick_hammers_spin.setValue(total)
        self.mill_hammer_rows_spin.valueChanged.connect(lambda: _update_total_hammers())
        self.mill_hammers_per_row_spin.valueChanged.connect(lambda: _update_total_hammers())
        _update_total_hammers()

    def _update_flow_diagram(self):
        pretreatment_on = self.radio_pretreatment.isChecked()
        milling_on = self.radio_milling.isChecked()
        classification_on = self.radio_classification.isChecked()
        self._flow_diagram.update_flow(
            pretreatment_enabled=pretreatment_on,
            classification_enabled=classification_on,
            milling_enabled=milling_on,
            preclassification=(self.mode_combo.currentIndex() == 0),
            include_feed=self.chk_feed.isChecked(),
            include_air=self.chk_air.isChecked(),
        )

    # ================================================================
    # Build params dict
    # ================================================================

    # ================================================================
    # Milling tab
    # ================================================================

    def _create_milling_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # Quick setup button for wizard
        wizard_frame = QFrame()
        wizard_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #14532d, stop:1 #1a3d2e);
                border: 1px solid #3dd68c;
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        wizard_layout = QHBoxLayout(wizard_frame)
        wizard_layout.setContentsMargins(12, 8, 12, 8)

        wizard_label = QLabel("Use the configuration wizard for guided setup with presets")
        wizard_label.setStyleSheet("color: #3dd68c; font-size: 9pt;")
        wizard_layout.addWidget(wizard_label, 1)

        self._wizard_btn = QPushButton("Open Wizard...")
        self._wizard_btn.setStyleSheet(f"""
            QPushButton {{
                background: #3dd68c;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                color: #141417;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #4ade80;
            }}
        """)
        self._wizard_btn.clicked.connect(self._open_milling_wizard)
        wizard_layout.addWidget(self._wizard_btn)

        layout.addWidget(wizard_frame)
        layout.addSpacing(8)

        # --- Rotor & Drive ---
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

        layout.addWidget(g)

        # --- Hammers ---
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

        layout.addWidget(g)

        # --- Screen ---
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

        self.mill_screen_size_ratio_threshold_spin = QDoubleSpinBox()
        self.mill_screen_size_ratio_threshold_spin.setRange(0.02, 0.50)
        self.mill_screen_size_ratio_threshold_spin.setValue(0.06)
        self.mill_screen_size_ratio_threshold_spin.setDecimals(3)
        self.mill_screen_size_ratio_threshold_spin.setSingleStep(0.01)
        self.mill_screen_size_ratio_threshold_spin.setToolTip(
            "Passage: below this ratio (size/aperture) = full; above = taper. Lower = finer discharge."
        )
        f.addRow("Size ratio threshold:", self.mill_screen_size_ratio_threshold_spin)

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

        layout.addWidget(g)

        # --- Housing ---
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

        layout.addWidget(g)

        # --- Feed & Discharge ---
        g = QGroupBox("Feed && Discharge")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

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

        layout.addWidget(g)

        return w

    def get_params(self) -> Dict[str, Any]:
        """Build a params dict with all process configuration."""
        preclassification = self.mode_combo.currentIndex() == 0
        return {
            # Process stages (mutually exclusive)
            "enable_pretreatment": self.radio_pretreatment.isChecked(),
            "enable_milling": self.radio_milling.isChecked(),
            "enable_classification": self.radio_classification.isChecked(),
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
            # RF Pretreatment
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
            # Milling — Rotor & Drive
            "mill_rotor_rpm": self.mill_rotor_rpm_spin.value(),
            "mill_rotor_diameter_m": self.mill_rotor_diameter_spin.value(),
            "mill_rotor_length_m": self.mill_rotor_length_spin.value(),
            "mill_shaft_diameter_m": self.mill_shaft_diameter_spin.value(),
            "mill_motor_power_kw": self.mill_motor_power_spin.value(),
            # Milling — Hammers
            "mill_hammer_rows": self.mill_hammer_rows_spin.value(),
            "mill_hammers_per_row": self.mill_hammers_per_row_spin.value(),
            "mill_hammer_mass_kg": self.mill_hammer_mass_spin.value(),
            "mill_hammer_length_m": self.mill_hammer_length_spin.value(),
            "mill_hammer_width_m": self.mill_hammer_width_spin.value(),
            "mill_hammer_thickness_m": self.mill_hammer_thickness_spin.value(),
            "mill_hammer_clearance_m": self.mill_hammer_clearance_spin.value(),
            # Milling — Screen
            "mill_screen_aperture_mm": self.mill_screen_aperture_spin.value(),
            "mill_screen_open_area": self.mill_screen_open_area_spin.value(),
            "mill_screen_size_ratio_threshold": self.mill_screen_size_ratio_threshold_spin.value(),
            "mill_screen_inner_radius_m": self.mill_screen_inner_radius_spin.value(),
            "mill_screen_thickness_m": self.mill_screen_thickness_spin.value(),
            # Milling — Housing
            "mill_housing_inner_radius_m": self.mill_housing_inner_radius_spin.value(),
            "mill_housing_length_m": self.mill_housing_length_spin.value(),
            "mill_housing_wall_thickness_m": self.mill_housing_wall_spin.value(),
            # Milling — Feed & Discharge
            "mill_feed_rate_kg_per_hr": self.mill_feed_rate_spin.value(),
            "mill_feed_chute_width_m": self.mill_feed_chute_width_spin.value(),
            "mill_feed_chute_height_m": self.mill_feed_chute_height_spin.value(),
            "mill_discharge_chute_width_m": self.mill_discharge_width_spin.value(),
            "mill_discharge_chute_height_m": self.mill_discharge_height_spin.value(),
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
        # Stages (mutually exclusive radio buttons)
        if p.get("enable_pretreatment", False):
            self.radio_pretreatment.setChecked(True)
        elif p.get("enable_milling", False):
            self.radio_milling.setChecked(True)
        elif p.get("enable_classification", True):
            self.radio_classification.setChecked(True)
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
        # Pretreatment
        if "pt_material" in p:
            idx = self.pt_material_combo.findText(p["pt_material"])
            if idx >= 0:
                self.pt_material_combo.setCurrentIndex(idx)
        if "pt_inlet_moisture" in p:
            self.pt_moisture_spin.setValue(p["pt_inlet_moisture"])
        if "pt_electrode_gap_mm" in p:
            self.pt_gap_spin.setValue(p["pt_electrode_gap_mm"])
        if "pt_belt_speed" in p:
            self.pt_speed_spin.setValue(p["pt_belt_speed"])
        # Milling
        _mill_float = {
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
            "mill_screen_size_ratio_threshold": self.mill_screen_size_ratio_threshold_spin,
            "mill_screen_inner_radius_m": self.mill_screen_inner_radius_spin,
            "mill_screen_thickness_m": self.mill_screen_thickness_spin,
            "mill_housing_inner_radius_m": self.mill_housing_inner_radius_spin,
            "mill_housing_length_m": self.mill_housing_length_spin,
            "mill_housing_wall_thickness_m": self.mill_housing_wall_spin,
            "mill_feed_rate_kg_per_hr": self.mill_feed_rate_spin,
            "mill_feed_chute_width_m": self.mill_feed_chute_width_spin,
            "mill_feed_chute_height_m": self.mill_feed_chute_height_spin,
            "mill_discharge_chute_width_m": self.mill_discharge_width_spin,
            "mill_discharge_chute_height_m": self.mill_discharge_height_spin,
        }
        for key, spin in _mill_float.items():
            if key in p:
                spin.setValue(p[key])
        if "mill_hammer_rows" in p:
            self.mill_hammer_rows_spin.setValue(int(p["mill_hammer_rows"]))
        if "mill_hammers_per_row" in p:
            self.mill_hammers_per_row_spin.setValue(int(p["mill_hammers_per_row"]))
        # Classification
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
        self.assembly_configured.emit(self.get_params())

    def _on_apply_close(self):
        self.assembly_configured.emit(self.get_params())
        self.accept()

    def _open_milling_wizard(self):
        """Open the guided milling configuration wizard."""
        try:
            from .milling_wizard import MillingConfigWizard
            wizard = MillingConfigWizard(self, self.get_params())
            wizard.configuration_complete.connect(self._apply_wizard_config)
            wizard.exec()
        except ImportError:
            QMessageBox.warning(
                self,
                "Wizard Unavailable",
                "The milling configuration wizard is not available."
            )

    def _apply_wizard_config(self, params: Dict[str, Any]):
        """Apply configuration from wizard."""
        self.load_params(params)
        # Select milling stage
        self.radio_milling.setChecked(True)
