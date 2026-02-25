"""
Pretreatment Configuration Wizard
=================================

3-step wizard for configuring GP-15 RF dielectric heating
simulation with material presets and visual feedback.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QScreen
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QStackedWidget, QScrollArea,
    QDoubleSpinBox, QCheckBox, QComboBox, QSlider,
    QSizePolicy, QGraphicsDropShadowEffect, QApplication,
)

from ..theme import COLORS


class PresetCard(QFrame):
    """Clickable preset card for material selection."""

    clicked = Signal()

    def __init__(
        self,
        name: str,
        description: str,
        icon: str,
        color: str,
        params: Dict[str, Any],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._name = name
        self._color = color
        self._params = params
        self._selected = False

        self._setup_ui(name, description, icon, color)

    def _setup_ui(self, name: str, description: str, icon: str, color: str):
        self.setFixedSize(200, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 28pt; color: {color};")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Name
        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"""
            font-size: 8pt;
            color: {COLORS.TEXT_MUTED};
        """)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(Qt.GlobalColor.black)
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def _update_style(self):
        if self._selected:
            self.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {COLORS.BG_ELEVATED}, stop:1 {COLORS.BG_SURFACE});
                    border: 2px solid {self._color};
                    border-radius: 12px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS.BG_SURFACE};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 12px;
                }}
                QFrame:hover {{
                    border-color: {COLORS.BORDER};
                    background: {COLORS.BG_HOVER};
                }}
            """)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    @property
    def params(self) -> Dict[str, Any]:
        return self._params


class WizardStep(QFrame):
    """Base class for wizard steps."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def get_params(self) -> Dict[str, Any]:
        return {}

    def load_params(self, params: Dict[str, Any]):
        pass


class Step1MaterialSelection(WizardStep):
    """Step 1: Material preset selection."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_preset: Optional[PresetCard] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Select Material Preset")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Choose a material preset to start with optimized parameters")
        subtitle.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        layout.addWidget(subtitle)

        # Preset cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        presets = [
            {
                "name": "Yellow Pea",
                "description": "Standard legume preset\n10% moisture, 75mm gap",
                "icon": "\U0001F7E1",  # Yellow circle
                "color": COLORS.WARNING,
                "params": {
                    "material": "yellow_pea",
                    "inlet_moisture": 0.10,
                    "bed_depth_mm": 25,
                    "electrode_gap_mm": 75,
                    "belt_speed": 0.20,
                },
            },
            {
                "name": "Faba Bean",
                "description": "High protein legume\n12% moisture, 80mm gap",
                "icon": "\U0001F7E4",  # Brown circle
                "color": "#8B4513",
                "params": {
                    "material": "faba_bean",
                    "inlet_moisture": 0.12,
                    "bed_depth_mm": 30,
                    "electrode_gap_mm": 80,
                    "belt_speed": 0.18,
                },
            },
            {
                "name": "Oat",
                "description": "Cereal grain preset\n11% moisture, 70mm gap",
                "icon": "\U0001F33E",  # Wheat emoji
                "color": COLORS.KPI_EFFICIENCY,
                "params": {
                    "material": "oat",
                    "inlet_moisture": 0.11,
                    "bed_depth_mm": 20,
                    "electrode_gap_mm": 70,
                    "belt_speed": 0.25,
                },
            },
        ]

        self._cards: list[PresetCard] = []
        for preset in presets:
            card = PresetCard(
                name=preset["name"],
                description=preset["description"],
                icon=preset["icon"],
                color=preset["color"],
                params=preset["params"],
            )
            card.clicked.connect(lambda c=card: self._on_card_clicked(c))
            cards_layout.addWidget(card)
            self._cards.append(card)

        cards_layout.addStretch()
        layout.addLayout(cards_layout)
        layout.addStretch()

        # Select first by default
        if self._cards:
            self._on_card_clicked(self._cards[0])

    def _on_card_clicked(self, card: PresetCard):
        for c in self._cards:
            c.set_selected(c is card)
        self._selected_preset = card

    def get_params(self) -> Dict[str, Any]:
        if self._selected_preset:
            return self._selected_preset.params.copy()
        return {}


class Step2RecipeConfig(WizardStep):
    """Step 2: RF recipe configuration."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Configure GP-15 Recipe")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Adjust the RF heating parameters for your run")
        subtitle.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        layout.addWidget(subtitle)

        # Form grid
        form = QGridLayout()
        form.setSpacing(12)
        form.setColumnStretch(1, 1)

        row = 0

        # Electrode gap
        form.addWidget(self._make_label("Electrode Gap"), row, 0)
        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(30, 150)
        self._gap_spin.setValue(75)
        self._gap_spin.setSuffix(" mm")
        self._gap_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._gap_spin, row, 1)
        form.addWidget(self._make_hint("Distance between electrodes"), row, 2)
        row += 1

        # Belt speed
        form.addWidget(self._make_label("Belt Speed"), row, 0)
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 1.0)
        self._speed_spin.setValue(0.20)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setSuffix(" m/min")
        self._speed_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._speed_spin, row, 1)
        form.addWidget(self._make_hint("Conveyor belt speed"), row, 2)
        row += 1

        # Run mass
        form.addWidget(self._make_label("Run Mass"), row, 0)
        self._mass_spin = QDoubleSpinBox()
        self._mass_spin.setRange(10, 200)
        self._mass_spin.setValue(61)
        self._mass_spin.setSuffix(" kg")
        self._mass_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._mass_spin, row, 1)
        form.addWidget(self._make_hint("Total material mass to process"), row, 2)
        row += 1

        # Extraction fan
        form.addWidget(self._make_label("Extraction Fan"), row, 0)
        self._fan_spin = QDoubleSpinBox()
        self._fan_spin.setRange(10, 55)
        self._fan_spin.setValue(35)
        self._fan_spin.setSuffix(" Hz")
        self._fan_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._fan_spin, row, 1)
        form.addWidget(self._make_hint("Moisture extraction rate"), row, 2)
        row += 1

        # MRH threshold
        form.addWidget(self._make_label("MRH Threshold"), row, 0)
        self._mrh_spin = QDoubleSpinBox()
        self._mrh_spin.setRange(1.0, 3.0)
        self._mrh_spin.setValue(1.7)
        self._mrh_spin.setDecimals(2)
        self._mrh_spin.setSuffix(" A")
        self._mrh_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._mrh_spin, row, 1)
        form.addWidget(self._make_hint("Overcurrent protection level"), row, 2)
        row += 1

        # Heaters
        form.addWidget(self._make_label("Heaters"), row, 0)
        self._heater_check = QCheckBox("Both heater banks on")
        self._heater_check.setChecked(True)
        self._heater_check.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        form.addWidget(self._heater_check, row, 1)
        row += 1

        layout.addLayout(form)
        layout.addStretch()

    def _make_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_SECONDARY};
        """)
        return label

    def _make_hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        return label

    def _spin_style(self) -> str:
        return f"""
            QDoubleSpinBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 10pt;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 120px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {COLORS.PRETREAT_PRIMARY};
            }}
        """

    def get_params(self) -> Dict[str, Any]:
        return {
            "electrode_gap_mm": self._gap_spin.value(),
            "belt_speed": self._speed_spin.value(),
            "run_mass_kg": self._mass_spin.value(),
            "extraction_fan_hz": self._fan_spin.value(),
            "mrh_amps": self._mrh_spin.value(),
            "heaters_on": self._heater_check.isChecked(),
        }

    def load_params(self, params: Dict[str, Any]):
        if "electrode_gap_mm" in params:
            self._gap_spin.setValue(params["electrode_gap_mm"])
        if "belt_speed" in params:
            self._speed_spin.setValue(params["belt_speed"])
        if "run_mass_kg" in params:
            self._mass_spin.setValue(params["run_mass_kg"])
        if "extraction_fan_hz" in params:
            self._fan_spin.setValue(params["extraction_fan_hz"])
        if "mrh_amps" in params:
            self._mrh_spin.setValue(params["mrh_amps"])
        if "heaters_on" in params:
            self._heater_check.setChecked(params["heaters_on"])


class Step3SimulationConfig(WizardStep):
    """Step 3: Simulation parameters."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Simulation Settings")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Configure physics options and simulation duration")
        subtitle.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        layout.addWidget(subtitle)

        # Form
        form = QGridLayout()
        form.setSpacing(12)
        form.setColumnStretch(1, 1)

        row = 0

        # Duration
        form.addWidget(self._make_label("Duration"), row, 0)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0, 600)
        self._duration_spin.setValue(0)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._duration_spin, row, 1)
        form.addWidget(self._make_hint("0 = auto-compute from mass & speed"), row, 2)
        row += 1

        # Bed depth
        form.addWidget(self._make_label("Bed Depth"), row, 0)
        self._bed_spin = QDoubleSpinBox()
        self._bed_spin.setRange(15, 60)
        self._bed_spin.setValue(25)
        self._bed_spin.setSuffix(" mm")
        self._bed_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._bed_spin, row, 1)
        form.addWidget(self._make_hint("Material layer thickness"), row, 2)
        row += 1

        # Initial temperature
        form.addWidget(self._make_label("Initial Temp"), row, 0)
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(10, 35)
        self._temp_spin.setValue(17.6)
        self._temp_spin.setDecimals(1)
        self._temp_spin.setSuffix(" \u00b0C")
        self._temp_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._temp_spin, row, 1)
        form.addWidget(self._make_hint("Starting material temperature"), row, 2)
        row += 1

        # Oscillator efficiency
        form.addWidget(self._make_label("Oscillator Eff."), row, 0)
        self._eff_spin = QDoubleSpinBox()
        self._eff_spin.setRange(0.30, 0.80)
        self._eff_spin.setValue(0.56)
        self._eff_spin.setDecimals(2)
        self._eff_spin.setStyleSheet(self._spin_style())
        form.addWidget(self._eff_spin, row, 1)
        form.addWidget(self._make_hint("RF oscillator efficiency"), row, 2)
        row += 1

        layout.addLayout(form)

        # Physics toggles
        physics_label = QLabel("Physics Options")
        physics_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
            margin-top: 12px;
        """)
        layout.addWidget(physics_label)

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

        # Device selector
        device_row = QHBoxLayout()
        device_label = QLabel("Compute Device:")
        device_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 10pt;")
        device_row.addWidget(device_label)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda"])
        self._device_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 6px 12px;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 100px;
            }}
        """)
        device_row.addWidget(self._device_combo)
        device_row.addStretch()
        layout.addLayout(device_row)

        layout.addStretch()

    def _make_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_SECONDARY};
        """)
        return label

    def _make_hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        return label

    def _spin_style(self) -> str:
        return f"""
            QDoubleSpinBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 10pt;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 120px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {COLORS.PRETREAT_PRIMARY};
            }}
        """

    def get_params(self) -> Dict[str, Any]:
        return {
            "duration_s": self._duration_spin.value(),
            "bed_depth_mm": self._bed_spin.value(),
            "initial_temp_c": self._temp_spin.value(),
            "efficiency": self._eff_spin.value(),
            "tvd": self._tvd_check.isChecked(),
            "controller": self._ctrl_check.isChecked(),
            "corrections": self._corr_check.isChecked(),
            "device": self._device_combo.currentText(),
        }

    def load_params(self, params: Dict[str, Any]):
        if "duration_s" in params:
            self._duration_spin.setValue(params["duration_s"])
        if "bed_depth_mm" in params:
            self._bed_spin.setValue(params["bed_depth_mm"])
        if "initial_temp_c" in params:
            self._temp_spin.setValue(params["initial_temp_c"])
        if "efficiency" in params:
            self._eff_spin.setValue(params["efficiency"])
        if "tvd" in params:
            self._tvd_check.setChecked(params["tvd"])
        if "controller" in params:
            self._ctrl_check.setChecked(params["controller"])
        if "corrections" in params:
            self._corr_check.setChecked(params["corrections"])
        if "device" in params:
            idx = self._device_combo.findText(params["device"])
            if idx >= 0:
                self._device_combo.setCurrentIndex(idx)


class PretreatmentConfigWizard(QDialog):
    """3-step wizard for configuring GP-15 RF heating simulation.

    Steps:
    1. Material Preset Selection
    2. GP-15 Recipe Configuration
    3. Simulation Settings

    Signals:
        configuration_complete(dict): Emitted with full config when wizard completes
    """

    configuration_complete = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_step = 0
        self._accumulated_params: Dict[str, Any] = {}

        self.setWindowTitle("GP-15 Configuration Wizard")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self._setup_ui()
        self._center_on_screen()

    def _center_on_screen(self):
        """Center the dialog on the screen or parent window."""
        if self.parent():
            # Center on parent
            parent_geo = self.parent().geometry()
            x = parent_geo.x() + (parent_geo.width() - self.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - self.height()) // 2
            self.move(x, y)
        else:
            # Center on primary screen
            screen = QApplication.primaryScreen()
            if screen:
                screen_geo = screen.availableGeometry()
                x = (screen_geo.width() - self.width()) // 2
                y = (screen_geo.height() - self.height()) // 2
                self.move(x, y)

    def _make_scrollable(self, widget: QWidget) -> QScrollArea:
        """Wrap a widget in a scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {COLORS.BG_DARKEST};
                width: 10px;
                border-radius: 5px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS.BG_HOVER};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS.BORDER};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background: {COLORS.BG_DARKEST};
                height: 10px;
                border-radius: 5px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: {COLORS.BG_HOVER};
                border-radius: 5px;
                min-width: 30px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {COLORS.BORDER};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
        """)
        scroll.setWidget(widget)
        return scroll

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QDialog {{
                background: {COLORS.BG_BASE};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._build_header()
        layout.addWidget(header)

        # Step indicator
        self._step_indicator = self._build_step_indicator()
        layout.addWidget(self._step_indicator)

        # Content stack with scrollable steps
        self._stack = QStackedWidget()
        self._step1 = Step1MaterialSelection()
        self._step2 = Step2RecipeConfig()
        self._step3 = Step3SimulationConfig()

        # Wrap each step in a scroll area
        self._stack.addWidget(self._make_scrollable(self._step1))
        self._stack.addWidget(self._make_scrollable(self._step2))
        self._stack.addWidget(self._make_scrollable(self._step3))
        layout.addWidget(self._stack, 1)

        # Footer
        footer = self._build_footer()
        layout.addWidget(footer)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS.PRETREAT_PRIMARY}, stop:1 {COLORS.PRETREAT_SECONDARY});
            }}
        """)

        layout = QVBoxLayout(header)
        layout.setContentsMargins(24, 12, 24, 12)

        title = QLabel("GP-15 RF Heating Configuration")
        title.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: 700;
            color: {COLORS.TEXT_INVERSE};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Configure your RF dielectric heating simulation")
        subtitle.setStyleSheet(f"""
            font-size: 9pt;
            color: rgba(255, 255, 255, 0.8);
        """)
        layout.addWidget(subtitle)

        return header

    def _build_step_indicator(self) -> QFrame:
        frame = QFrame()
        frame.setFixedHeight(50)
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border-bottom: 1px solid {COLORS.BORDER};
            }}
        """)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(24, 0, 24, 0)

        steps = [
            ("1", "Material"),
            ("2", "Recipe"),
            ("3", "Simulation"),
        ]

        self._step_labels: list[tuple[QLabel, QLabel]] = []
        for i, (num, text) in enumerate(steps):
            step_layout = QHBoxLayout()
            step_layout.setSpacing(8)

            num_label = QLabel(num)
            num_label.setFixedSize(28, 28)
            num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_layout.addWidget(num_label)

            text_label = QLabel(text)
            step_layout.addWidget(text_label)

            self._step_labels.append((num_label, text_label))
            layout.addLayout(step_layout)

            if i < len(steps) - 1:
                line = QFrame()
                line.setFixedHeight(2)
                line.setMinimumWidth(40)
                line.setStyleSheet(f"background: {COLORS.BORDER};")
                layout.addWidget(line, 1)

        layout.addStretch()
        self._update_step_indicator()
        return frame

    def _update_step_indicator(self):
        for i, (num_label, text_label) in enumerate(self._step_labels):
            if i < self._current_step:
                # Completed
                num_label.setStyleSheet(f"""
                    background: {COLORS.SUCCESS};
                    border-radius: 14px;
                    font-weight: 700;
                    color: {COLORS.TEXT_INVERSE};
                """)
                text_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY};")
            elif i == self._current_step:
                # Current
                num_label.setStyleSheet(f"""
                    background: {COLORS.PRETREAT_PRIMARY};
                    border-radius: 14px;
                    font-weight: 700;
                    color: {COLORS.TEXT_INVERSE};
                """)
                text_label.setStyleSheet(f"""
                    color: {COLORS.TEXT_PRIMARY};
                    font-weight: 600;
                """)
            else:
                # Future
                num_label.setStyleSheet(f"""
                    background: {COLORS.BG_SURFACE};
                    border: 1px solid {COLORS.BORDER};
                    border-radius: 14px;
                    color: {COLORS.TEXT_MUTED};
                """)
                text_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setFixedHeight(64)
        footer.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-top: 1px solid {COLORS.BORDER};
            }}
        """)

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 12, 24, 12)

        self._back_btn = QPushButton("Back")
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 10pt;
                color: {COLORS.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        self._back_btn.clicked.connect(self._go_back)
        layout.addWidget(self._back_btn)

        layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                padding: 10px 16px;
                font-size: 10pt;
                color: {COLORS.TEXT_MUTED};
            }}
            QPushButton:hover {{
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.PRETREAT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 10px 32px;
                font-size: 10pt;
                font-weight: 600;
                color: {COLORS.TEXT_INVERSE};
            }}
            QPushButton:hover {{
                background: {COLORS.PRETREAT_SECONDARY};
            }}
        """)
        self._next_btn.clicked.connect(self._go_next)
        layout.addWidget(self._next_btn)

        self._update_buttons()
        return footer

    def _update_buttons(self):
        self._back_btn.setEnabled(self._current_step > 0)
        if self._current_step == 2:
            self._next_btn.setText("Apply & Run")
        else:
            self._next_btn.setText("Next")

    def _go_back(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._stack.setCurrentIndex(self._current_step)
            self._update_step_indicator()
            self._update_buttons()

    def _get_current_step(self) -> Optional[WizardStep]:
        """Get the actual step widget from the current scroll area."""
        current = self._stack.currentWidget()
        if isinstance(current, QScrollArea):
            widget = current.widget()
            if isinstance(widget, WizardStep):
                return widget
        return None

    def _get_step_at(self, index: int) -> Optional[WizardStep]:
        """Get the step widget at a given index."""
        scroll = self._stack.widget(index)
        if isinstance(scroll, QScrollArea):
            widget = scroll.widget()
            if isinstance(widget, WizardStep):
                return widget
        return None

    def _go_next(self):
        # Collect params from current step
        step = self._get_current_step()
        if step:
            params = step.get_params()
            self._accumulated_params.update(params)

        if self._current_step < 2:
            self._current_step += 1
            self._stack.setCurrentIndex(self._current_step)

            # Load accumulated params into next step
            next_step = self._get_current_step()
            if next_step:
                next_step.load_params(self._accumulated_params)

            self._update_step_indicator()
            self._update_buttons()
        else:
            # Final step - emit configuration
            self.configuration_complete.emit(self._accumulated_params)
            self.accept()

    def get_configuration(self) -> Dict[str, Any]:
        """Get the accumulated configuration."""
        return self._accumulated_params.copy()
