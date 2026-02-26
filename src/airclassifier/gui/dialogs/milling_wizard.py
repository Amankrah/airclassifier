"""
Milling Configuration Wizard
============================

Multi-step wizard for configuring hammer mill parameters
with presets and live 3D preview.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QStackedWidget,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox,
    QSlider, QGroupBox, QFormLayout, QSizePolicy,
    QGraphicsDropShadowEffect, QScrollArea, QApplication,
)

from ..theme import COLORS


class MachineScale(Enum):
    """Predefined machine scales."""
    LAB = "lab"
    PILOT = "pilot"
    PRODUCTION = "production"
    CUSTOM = "custom"


@dataclass
class MachinePreset:
    """Preset machine configuration."""
    name: str
    description: str
    rotor_diameter_m: float
    rotor_length_m: float
    motor_power_kw: float
    hammer_rows: int
    hammers_per_row: int
    screen_inner_radius_m: float
    housing_inner_radius_m: float
    default_rpm: int
    default_screen_mm: float


# Predefined machine presets
# Yellow pea flour for protein separation (NIH): 0.75 mm → D50 ~23.7 µm, 2.0 mm → D50 ~31.1 µm.
# Default 0.75 mm targets fine flour (D50 23.7–31.1 µm) for protein separation.
PRESETS = {
    MachineScale.LAB: MachinePreset(
        name="Lab Scale",
        description="Bench-top hammer mill for R&D (150mm rotor, 2.2 kW)",
        rotor_diameter_m=0.10,
        rotor_length_m=0.15,
        motor_power_kw=2.2,
        hammer_rows=2,
        hammers_per_row=3,
        screen_inner_radius_m=0.095,
        housing_inner_radius_m=0.10,
        default_rpm=4500,
        default_screen_mm=0.75,  # NIH: D50 ~24 µm (protein separation)
    ),
    MachineScale.PILOT: MachinePreset(
        name="Pilot Scale",
        description="Pilot plant hammer mill (200mm rotor, 22 kW)",
        rotor_diameter_m=0.20,
        rotor_length_m=0.30,
        motor_power_kw=22.0,
        hammer_rows=4,
        hammers_per_row=4,
        screen_inner_radius_m=0.188,
        housing_inner_radius_m=0.20,
        default_rpm=3000,
        default_screen_mm=0.75,  # NIH: D50 ~24 µm (protein separation)
    ),
    MachineScale.PRODUCTION: MachinePreset(
        name="Production Scale",
        description="Industrial hammer mill (400mm rotor, 75 kW)",
        rotor_diameter_m=0.40,
        rotor_length_m=0.60,
        motor_power_kw=75.0,
        hammer_rows=6,
        hammers_per_row=6,
        screen_inner_radius_m=0.380,
        housing_inner_radius_m=0.40,
        default_rpm=2000,
        default_screen_mm=0.75,  # NIH: D50 ~24 µm (protein separation)
    ),
}


class _WizardPage(QWidget):
    """Base class for wizard pages."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def validate(self) -> bool:
        """Validate page inputs. Override in subclasses."""
        return True

    def get_data(self) -> Dict[str, Any]:
        """Get page data. Override in subclasses."""
        return {}


class PresetSelectionPage(_WizardPage):
    """Step 1: Select machine preset."""

    preset_changed = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_scale = MachineScale.PILOT
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Select Machine Profile")
        title.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Choose a predefined machine configuration or start with custom parameters.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 10pt;")
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # Preset cards
        self._button_group = QButtonGroup(self)

        for scale in [MachineScale.LAB, MachineScale.PILOT, MachineScale.PRODUCTION, MachineScale.CUSTOM]:
            card = self._create_preset_card(scale)
            layout.addWidget(card)

        layout.addStretch()

    def _create_preset_card(self, scale: MachineScale) -> QFrame:
        """Create a preset selection card."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 2px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
                padding: 12px;
            }}
            QFrame:hover {{
                border-color: {COLORS.MILLING_PRIMARY};
            }}
        """)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)

        radio = QRadioButton()
        radio.setStyleSheet(f"""
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
            QRadioButton::indicator:checked {{
                background: {COLORS.MILLING_PRIMARY};
                border: 2px solid {COLORS.MILLING_PRIMARY};
                border-radius: 9px;
            }}
            QRadioButton::indicator:unchecked {{
                background: {COLORS.BG_DARK};
                border: 2px solid {COLORS.BORDER};
                border-radius: 9px;
            }}
        """)
        if scale == MachineScale.PILOT:
            radio.setChecked(True)
        self._button_group.addButton(radio, list(MachineScale).index(scale))
        radio.toggled.connect(lambda checked, s=scale: self._on_preset_selected(s) if checked else None)
        layout.addWidget(radio)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        if scale == MachineScale.CUSTOM:
            name = "Custom Configuration"
            desc = "Define all parameters manually for specialized applications"
        else:
            preset = PRESETS[scale]
            name = preset.name
            desc = preset.description

        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
        """)
        text_layout.addWidget(name_label)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Badge for recommended
        if scale == MachineScale.PILOT:
            badge = QLabel("Recommended")
            badge.setStyleSheet(f"""
                background: {COLORS.MILLING_MUTED};
                color: {COLORS.MILLING_PRIMARY};
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 8pt;
                font-weight: 600;
            """)
            layout.addWidget(badge)

        return card

    def _on_preset_selected(self, scale: MachineScale):
        self._selected_scale = scale
        self.preset_changed.emit(scale)

    def get_data(self) -> Dict[str, Any]:
        return {"scale": self._selected_scale}


class RotorConfigPage(_WizardPage):
    """Step 2: Configure rotor and hammers."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._preset: Optional[MachinePreset] = PRESETS[MachineScale.PILOT]
        self._setup_ui()
        self._apply_preset()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Rotor Configuration")
        title.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)

        # RPM slider with preview
        rpm_group = QGroupBox("Operating Speed")
        rpm_group.setStyleSheet(self._get_group_style())
        rpm_layout = QVBoxLayout(rpm_group)

        rpm_header = QHBoxLayout()
        rpm_label = QLabel("Rotor RPM")
        rpm_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY};")
        rpm_header.addWidget(rpm_label)

        self._rpm_value = QLabel("3000")
        self._rpm_value.setStyleSheet(f"""
            color: {COLORS.MILLING_PRIMARY};
            font-size: 14pt;
            font-weight: 700;
        """)
        rpm_header.addWidget(self._rpm_value)
        rpm_header.addStretch()

        self._tip_speed_label = QLabel("Tip speed: 31.4 m/s")
        self._tip_speed_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        rpm_header.addWidget(self._tip_speed_label)

        rpm_layout.addLayout(rpm_header)

        self._rpm_slider = QSlider(Qt.Orientation.Horizontal)
        self._rpm_slider.setRange(500, 6000)
        self._rpm_slider.setValue(3000)
        self._rpm_slider.setStyleSheet(self._get_slider_style())
        self._rpm_slider.valueChanged.connect(self._on_rpm_changed)
        rpm_layout.addWidget(self._rpm_slider)

        content_layout.addWidget(rpm_group)

        # Hammer configuration
        hammer_group = QGroupBox("Hammer Configuration")
        hammer_group.setStyleSheet(self._get_group_style())
        hammer_layout = QGridLayout(hammer_group)
        hammer_layout.setSpacing(12)

        hammer_layout.addWidget(QLabel("Rows:"), 0, 0)
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 12)
        self._rows_spin.setValue(4)
        self._rows_spin.setStyleSheet(self._get_spin_style())
        self._rows_spin.valueChanged.connect(self._update_hammer_count)
        hammer_layout.addWidget(self._rows_spin, 0, 1)

        hammer_layout.addWidget(QLabel("Per Row:"), 0, 2)
        self._per_row_spin = QSpinBox()
        self._per_row_spin.setRange(2, 8)
        self._per_row_spin.setValue(4)
        self._per_row_spin.setStyleSheet(self._get_spin_style())
        self._per_row_spin.valueChanged.connect(self._update_hammer_count)
        hammer_layout.addWidget(self._per_row_spin, 0, 3)

        self._total_label = QLabel("Total: 16 hammers")
        self._total_label.setStyleSheet(f"""
            color: {COLORS.MILLING_PRIMARY};
            font-weight: 600;
            font-size: 11pt;
        """)
        hammer_layout.addWidget(self._total_label, 1, 0, 1, 4)

        content_layout.addWidget(hammer_group)

        # Rotor dimensions (collapsed by default for non-custom)
        self._dimensions_group = QGroupBox("Rotor Dimensions (Advanced)")
        self._dimensions_group.setStyleSheet(self._get_group_style())
        dim_layout = QFormLayout(self._dimensions_group)
        dim_layout.setSpacing(10)

        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(0.05, 0.60)
        self._diameter_spin.setValue(0.20)
        self._diameter_spin.setDecimals(3)
        self._diameter_spin.setSuffix(" m")
        self._diameter_spin.setStyleSheet(self._get_spin_style())
        self._diameter_spin.valueChanged.connect(self._on_diameter_changed)
        dim_layout.addRow("Rotor Diameter:", self._diameter_spin)

        self._length_spin = QDoubleSpinBox()
        self._length_spin.setRange(0.10, 1.00)
        self._length_spin.setValue(0.30)
        self._length_spin.setDecimals(3)
        self._length_spin.setSuffix(" m")
        self._length_spin.setStyleSheet(self._get_spin_style())
        dim_layout.addRow("Rotor Length:", self._length_spin)

        self._power_spin = QDoubleSpinBox()
        self._power_spin.setRange(1, 150)
        self._power_spin.setValue(22)
        self._power_spin.setDecimals(1)
        self._power_spin.setSuffix(" kW")
        self._power_spin.setStyleSheet(self._get_spin_style())
        dim_layout.addRow("Motor Power:", self._power_spin)

        content_layout.addWidget(self._dimensions_group)
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _get_group_style(self) -> str:
        return f"""
            QGroupBox {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
                margin-top: 14px;
                padding: 12px;
                font-weight: 600;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
        """

    def _get_slider_style(self) -> str:
        return f"""
            QSlider::groove:horizontal {{
                background: {COLORS.BG_DARKEST};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS.MILLING_PRIMARY};
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS.MILLING_MUTED};
                border-radius: 4px;
            }}
        """

    def _get_spin_style(self) -> str:
        return f"""
            QSpinBox, QDoubleSpinBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 80px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {COLORS.MILLING_PRIMARY};
            }}
        """

    def set_preset(self, scale: MachineScale):
        """Apply a preset configuration."""
        if scale == MachineScale.CUSTOM:
            self._preset = None
            self._dimensions_group.setVisible(True)
        else:
            self._preset = PRESETS[scale]
            self._apply_preset()
            self._dimensions_group.setVisible(False)

    def _apply_preset(self):
        if self._preset is None:
            return

        self._rpm_slider.setValue(self._preset.default_rpm)
        self._rows_spin.setValue(self._preset.hammer_rows)
        self._per_row_spin.setValue(self._preset.hammers_per_row)
        self._diameter_spin.setValue(self._preset.rotor_diameter_m)
        self._length_spin.setValue(self._preset.rotor_length_m)
        self._power_spin.setValue(self._preset.motor_power_kw)

    def _on_rpm_changed(self, value: int):
        self._rpm_value.setText(str(value))
        self._update_tip_speed()

    def _on_diameter_changed(self, value: float):
        self._update_tip_speed()

    def _update_tip_speed(self):
        import math
        rpm = self._rpm_slider.value()
        diameter = self._diameter_spin.value()
        tip_speed = (rpm / 60) * math.pi * diameter
        self._tip_speed_label.setText(f"Tip speed: {tip_speed:.1f} m/s")

    def _update_hammer_count(self):
        total = self._rows_spin.value() * self._per_row_spin.value()
        self._total_label.setText(f"Total: {total} hammers")

    def get_data(self) -> Dict[str, Any]:
        return {
            "rotor_rpm": self._rpm_slider.value(),
            "hammer_rows": self._rows_spin.value(),
            "hammers_per_row": self._per_row_spin.value(),
            "rotor_diameter_m": self._diameter_spin.value(),
            "rotor_length_m": self._length_spin.value(),
            "motor_power_kw": self._power_spin.value(),
        }


class ScreenConfigPage(_WizardPage):
    """Step 3: Configure screen and product size."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Screen & Product Size")
        title.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        subtitle = QLabel("Configure screen aperture to control product particle size.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 10pt;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Aperture slider with large display
        aperture_frame = QFrame()
        aperture_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 12px;
                padding: 20px;
            }}
        """)
        aperture_layout = QVBoxLayout(aperture_frame)
        aperture_layout.setSpacing(12)

        self._aperture_value = QLabel("0.75 mm")
        self._aperture_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._aperture_value.setStyleSheet(f"""
            font-size: 32pt;
            font-weight: 700;
            color: {COLORS.MILLING_PRIMARY};
        """)
        aperture_layout.addWidget(self._aperture_value)

        aperture_label = QLabel("Screen Aperture (yellow pea flour, protein separation)")
        aperture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        aperture_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10pt;")
        aperture_layout.addWidget(aperture_label)

        self._aperture_slider = QSlider(Qt.Orientation.Horizontal)
        self._aperture_slider.setRange(30, 200)  # 0.3-2.0 mm (NIH: 0.75→D50 ~24 µm, 2→~31 µm)
        self._aperture_slider.setValue(75)  # 0.75 mm default (NIH, protein separation)
        self._aperture_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {COLORS.BG_DARKEST};
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS.MILLING_PRIMARY};
                width: 24px;
                height: 24px;
                margin: -7px 0;
                border-radius: 12px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS.MILLING_MUTED};
                border-radius: 5px;
            }}
        """)
        self._aperture_slider.valueChanged.connect(self._on_aperture_changed)
        aperture_layout.addWidget(self._aperture_slider)

        # Expected D50 hint (NIH: 0.75 mm → ~23.7 µm, 2 mm → ~31.1 µm)
        self._d50_hint = QLabel("Expected D50: ~23 µm (excellent for protein separation)")
        self._d50_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._d50_hint.setStyleSheet(f"color: {COLORS.KPI_SIZE}; font-size: 11pt; font-weight: 600;")
        aperture_layout.addWidget(self._d50_hint)

        layout.addWidget(aperture_frame)

        # Feed rate
        feed_group = QGroupBox("Feed Settings")
        feed_group.setStyleSheet(f"""
            QGroupBox {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
                margin-top: 14px;
                padding: 12px;
                font-weight: 600;
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)
        feed_layout = QFormLayout(feed_group)
        feed_layout.setSpacing(10)

        self._feed_spin = QDoubleSpinBox()
        self._feed_spin.setRange(10, 2000)
        self._feed_spin.setValue(500)
        self._feed_spin.setSuffix(" kg/h")
        self._feed_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 6px;
                color: {COLORS.TEXT_PRIMARY};
                min-width: 120px;
            }}
        """)
        feed_layout.addRow("Feed Rate:", self._feed_spin)

        layout.addWidget(feed_group)
        layout.addStretch()

    def _on_aperture_changed(self, value: int):
        mm = value / 100
        self._aperture_value.setText(f"{mm:.2f} mm")

        # NIH: 0.75 mm → D50 ~23.7 µm, 2.0 mm → D50 ~31.1 µm (yellow pea, rotor beater mill)
        d50_est = 17.4 + 6.84 * mm
        if d50_est <= 31:
            hint = f"D50: ~{d50_est:.0f} µm (excellent for protein separation)"
        elif d50_est <= 55:
            hint = f"D50: ~{d50_est:.0f} µm (good for starch/protein fractionation)"
        elif d50_est <= 114:
            hint = f"D50: ~{d50_est:.0f} µm (moderate - consider 0.75 mm for protein separation)"
        else:
            hint = f"D50: ~{d50_est:.0f} µm (coarse - recommend 0.75–2.0 mm for protein separation)"
        self._d50_hint.setText(hint)

    def get_data(self) -> Dict[str, Any]:
        return {
            "screen_aperture_mm": self._aperture_slider.value() / 100,
            "feed_rate_kg_per_hr": self._feed_spin.value(),
        }


class MillingConfigWizard(QDialog):
    """Multi-step wizard for hammer mill configuration.

    Signals:
        configuration_complete(dict): Emitted with full configuration
    """

    configuration_complete = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None, current_params: Optional[Dict] = None):
        super().__init__(parent)
        self._current_params = current_params or {}
        self._current_page = 0

        self.setWindowTitle("Configure Hammer Mill")
        self.setMinimumSize(600, 550)
        self.resize(700, 650)

        self._setup_ui()
        self._connect_signals()
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with progress
        header = QFrame()
        header.setStyleSheet(f"background: {COLORS.BG_ELEVATED};")
        header.setFixedHeight(60)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        self._step_labels = []
        for i, title in enumerate(["1. Profile", "2. Rotor", "3. Screen"]):
            lbl = QLabel(title)
            lbl.setStyleSheet(f"""
                font-size: 10pt;
                color: {COLORS.TEXT_MUTED};
                padding: 8px 16px;
            """)
            if i == 0:
                lbl.setStyleSheet(f"""
                    font-size: 10pt;
                    font-weight: 600;
                    color: {COLORS.MILLING_PRIMARY};
                    background: {COLORS.MILLING_MUTED};
                    border-radius: 4px;
                    padding: 8px 16px;
                """)
            header_layout.addWidget(lbl)
            self._step_labels.append(lbl)

        header_layout.addStretch()
        layout.addWidget(header)

        # Page stack with scrollable pages
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {COLORS.BG_BASE};")

        self._preset_page = PresetSelectionPage()
        self._rotor_page = RotorConfigPage()
        self._screen_page = ScreenConfigPage()

        # Wrap each page in a scroll area
        self._stack.addWidget(self._make_scrollable(self._preset_page))
        self._stack.addWidget(self._make_scrollable(self._rotor_page))
        self._stack.addWidget(self._make_scrollable(self._screen_page))

        layout.addWidget(self._stack, 1)

        # Footer with navigation
        footer = QFrame()
        footer.setStyleSheet(f"""
            background: {COLORS.BG_ELEVATED};
            border-top: 1px solid {COLORS.BORDER_SUBTLE};
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 12, 20, 12)

        self._back_btn = QPushButton("Back")
        self._back_btn.setEnabled(False)
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                padding: 10px 24px;
                color: {COLORS.TEXT_SECONDARY};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        footer_layout.addWidget(self._back_btn)

        footer_layout.addStretch()

        self._next_btn = QPushButton("Next")
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.MILLING_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                color: {COLORS.BG_DARKEST};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.MILLING_SECONDARY};
            }}
        """)
        footer_layout.addWidget(self._next_btn)

        layout.addWidget(footer)

    def _connect_signals(self):
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn.clicked.connect(self._go_next)
        self._preset_page.preset_changed.connect(self._on_preset_changed)

    def _get_page_widget(self, index: int) -> Optional[_WizardPage]:
        """Get the actual page widget from a scroll area at given index."""
        scroll = self._stack.widget(index)
        if isinstance(scroll, QScrollArea):
            widget = scroll.widget()
            if isinstance(widget, _WizardPage):
                return widget
        return None

    def _on_preset_changed(self, scale: MachineScale):
        self._rotor_page.set_preset(scale)

    def _go_back(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_navigation()

    def _go_next(self):
        if self._current_page < 2:
            self._current_page += 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_navigation()
        else:
            self._finish()

    def _update_navigation(self):
        self._back_btn.setEnabled(self._current_page > 0)

        if self._current_page == 2:
            self._next_btn.setText("Build & Preview")
        else:
            self._next_btn.setText("Next")

        # Update step indicators
        for i, lbl in enumerate(self._step_labels):
            if i == self._current_page:
                lbl.setStyleSheet(f"""
                    font-size: 10pt;
                    font-weight: 600;
                    color: {COLORS.MILLING_PRIMARY};
                    background: {COLORS.MILLING_MUTED};
                    border-radius: 4px;
                    padding: 8px 16px;
                """)
            elif i < self._current_page:
                lbl.setStyleSheet(f"""
                    font-size: 10pt;
                    color: {COLORS.SUCCESS};
                    padding: 8px 16px;
                """)
            else:
                lbl.setStyleSheet(f"""
                    font-size: 10pt;
                    color: {COLORS.TEXT_MUTED};
                    padding: 8px 16px;
                """)

    def _finish(self):
        # Collect all data
        config = {}
        config.update(self._preset_page.get_data())
        config.update(self._rotor_page.get_data())
        config.update(self._screen_page.get_data())

        # Map to expected parameter names
        params = {
            "enable_milling": True,
            "mill_rotor_rpm": config.get("rotor_rpm", 3000),
            "mill_rotor_diameter_m": config.get("rotor_diameter_m", 0.20),
            "mill_rotor_length_m": config.get("rotor_length_m", 0.30),
            "mill_motor_power_kw": config.get("motor_power_kw", 22.0),
            "mill_hammer_rows": config.get("hammer_rows", 4),
            "mill_hammers_per_row": config.get("hammers_per_row", 4),
            "mill_screen_aperture_mm": config.get("screen_aperture_mm", 0.75),
            "mill_feed_rate_kg_per_hr": config.get("feed_rate_kg_per_hr", 500),
        }

        self.configuration_complete.emit(params)
        self.accept()

    def get_params(self) -> Dict[str, Any]:
        """Get the configured parameters."""
        config = {}
        config.update(self._preset_page.get_data())
        config.update(self._rotor_page.get_data())
        config.update(self._screen_page.get_data())
        return config
