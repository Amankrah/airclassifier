"""
Milling Control Panel
=====================

Modern control panel for hammer mill simulation with
grouped recipe controls and status display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QGroupBox, QFormLayout, QProgressBar, QTextEdit,
    QSizePolicy, QGraphicsDropShadowEffect, QScrollArea,
    QWidget, QSlider, QComboBox,
)

from ...theme import COLORS, ANIMATIONS

try:
    from airclassifier.milling.control.recipe import DEFAULT_RECIPES
except ImportError:
    DEFAULT_RECIPES = {}


class MillingControlPanel(QFrame):
    """Modern control panel for hammer mill simulation.

    Signals:
        run_clicked(): Run button pressed
        stop_clicked(): Stop button pressed
        config_clicked(): Configure button pressed
        recipe_changed(dict): Recipe parameters changed
    """

    run_clicked = Signal()
    stop_clicked = Signal()
    config_clicked = Signal()
    recipe_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_style()
        self._setup_ui()
        self._connect_signals()

    def _setup_style(self):
        """Apply glassmorphism styling."""
        self.setObjectName("controlPanel")
        self.setStyleSheet(f"""
            QFrame#controlPanel {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS.GLASS_START},
                    stop:1 {COLORS.GLASS_END});
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        self.setMinimumWidth(320)
        self.setMaximumWidth(400)

    def _setup_ui(self):
        """Build the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Hammer Mill")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        header.addWidget(title)

        self._config_btn = QPushButton("...")
        self._config_btn.setToolTip("Configure Mill Geometry")
        self._config_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 12px;
                color: {COLORS.TEXT_SECONDARY};
                font-size: 14pt;
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)
        header.addWidget(self._config_btn)
        layout.addLayout(header)

        # Scroll area for recipe and log so all controls are reachable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Simulation buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.SUCCESS_MUTED};
                border: 1px solid {COLORS.SUCCESS};
                border-radius: 8px;
                padding: 10px 20px;
                color: {COLORS.SUCCESS};
                font-size: 11pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.SUCCESS};
                color: {COLORS.BG_DARKEST};
            }}
            QPushButton:disabled {{
                background: {COLORS.BG_DARK};
                border-color: {COLORS.BORDER_SUBTLE};
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        scroll_layout.addLayout(btn_row)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.DANGER_MUTED};
                border: 1px solid {COLORS.DANGER};
                border-radius: 8px;
                padding: 10px 20px;
                color: {COLORS.DANGER};
                font-size: 11pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.DANGER};
                color: {COLORS.TEXT_INVERSE};
            }}
            QPushButton:disabled {{
                background: {COLORS.BG_DARK};
                border-color: {COLORS.BORDER_SUBTLE};
                color: {COLORS.TEXT_DISABLED};
            }}
        """)
        btn_row.addWidget(self._stop_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                height: 20px;
                text-align: center;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS.MILLING_PRIMARY},
                    stop:1 {COLORS.SUCCESS});
                border-radius: 4px;
            }}
        """)
        self._progress.hide()
        scroll_layout.addWidget(self._progress)

        # Recipe section
        recipe_group = self._create_recipe_group()
        scroll_layout.addWidget(recipe_group)

        # Log output
        log_label = QLabel("Log")
        log_label.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED};")
        scroll_layout.addWidget(log_label)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(100)
        self._log_text.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                color: {COLORS.TEXT_SECONDARY};
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9pt;
                padding: 6px;
            }}
        """)
        scroll_layout.addWidget(self._log_text)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

    def _create_recipe_group(self) -> QGroupBox:
        """Create the recipe parameters group."""
        group = QGroupBox("Recipe")
        group.setStyleSheet(f"""
            QGroupBox {{
                background: transparent;
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: 600;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
        """)

        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(12)

        # Recipe preset dropdown
        preset_row = QFormLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Manual (current)")
        self._preset_recipes: List[Dict[str, Any]] = []
        for _k in sorted(DEFAULT_RECIPES.keys()):
            r = DEFAULT_RECIPES[_k]
            self._preset_combo.addItem(r.name)
            self._preset_recipes.append({
                "rotor_rpm": int(r.rotor_rpm),
                "screen_aperture_mm": r.screen_aperture_mm,
                "feed_rate_kg_per_hr": r.feed_rate_kg_per_hr,
                "duration_s": r.run_duration_s,
            })
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addRow("Preset:", self._preset_combo)
        layout.addLayout(preset_row)

        # Build preset combo stylesheet (same as term_mode_combo later)
        self._preset_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 24px 4px 8px;
                color: {COLORS.TEXT_PRIMARY};
                min-height: 24px;
            }}
            QComboBox:hover {{ border-color: {COLORS.MILLING_PRIMARY}; background: {COLORS.BG_SURFACE}; }}
            QComboBox:focus {{ border-color: {COLORS.MILLING_PRIMARY}; }}
        """)

        # RPM with slider
        rpm_row = QVBoxLayout()
        rpm_header = QHBoxLayout()
        rpm_label = QLabel("Rotor RPM")
        rpm_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        rpm_header.addWidget(rpm_label)

        self._rpm_value = QLabel("3000")
        self._rpm_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-weight: 600;")
        rpm_header.addWidget(self._rpm_value)
        rpm_header.addStretch()
        rpm_row.addLayout(rpm_header)

        self._rpm_slider = QSlider(Qt.Orientation.Horizontal)
        self._rpm_slider.setRange(500, 6000)
        self._rpm_slider.setValue(3000)
        self._rpm_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {COLORS.BG_DARKEST};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS.MILLING_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLORS.MILLING_SECONDARY};
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS.MILLING_MUTED};
                border-radius: 3px;
            }}
        """)
        self._rpm_slider.valueChanged.connect(self._on_rpm_changed)
        rpm_row.addWidget(self._rpm_slider)
        layout.addLayout(rpm_row)

        # Screen aperture with slider and d50 estimate
        aperture_row = QVBoxLayout()
        aperture_header = QHBoxLayout()
        aperture_label = QLabel("Screen Aperture")
        aperture_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        aperture_header.addWidget(aperture_label)

        self._aperture_value = QLabel("0.75 mm")
        self._aperture_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-weight: 600;")
        aperture_header.addWidget(self._aperture_value)
        aperture_header.addStretch()
        aperture_row.addLayout(aperture_header)

        self._aperture_slider = QSlider(Qt.Orientation.Horizontal)
        self._aperture_slider.setRange(30, 200)  # 0.3-2.0 mm (NIH: 0.75→D50 ~24 µm, 2→~31 µm)
        self._aperture_slider.setValue(75)  # 0.75 mm default (NIH, protein separation)
        self._aperture_slider.setStyleSheet(self._rpm_slider.styleSheet())
        self._aperture_slider.valueChanged.connect(self._on_aperture_changed)
        aperture_row.addWidget(self._aperture_slider)

        # Estimated D50 and quality hint (NIH: 0.75 mm → ~23.7 µm, 2 mm → ~31.1 µm)
        self._d50_hint = QLabel("D50 ~23 µm • Excellent for protein separation")
        self._d50_hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt; font-style: italic;")
        aperture_row.addWidget(self._d50_hint)
        layout.addLayout(aperture_row)

        # Feed rate and termination controls
        form = QFormLayout()
        form.setSpacing(8)

        self._seeds_feed_mass_spin = QDoubleSpinBox()
        self._seeds_feed_mass_spin.setRange(0, 10000)
        self._seeds_feed_mass_spin.setValue(0)
        self._seeds_feed_mass_spin.setDecimals(2)
        self._seeds_feed_mass_spin.setSuffix(" kg")
        self._seeds_feed_mass_spin.setSpecialValueText("Continuous (0)")
        self._seeds_feed_mass_spin.setToolTip("Total input mass of seeds to feed into the mill [kg]. Set 0 for continuous feeding at the feed rate below.")
        self._seeds_feed_mass_spin.setStyleSheet(self._get_spinbox_style())
        form.addRow("Input mass (seeds):", self._seeds_feed_mass_spin)
        seeds_hint = QLabel("0 = continuous; set e.g. 10 for 10 kg batch")
        seeds_hint.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED};")
        form.addRow("", seeds_hint)

        self._feed_spin = QDoubleSpinBox()
        self._feed_spin.setRange(10, 2000)
        self._feed_spin.setValue(500)
        self._feed_spin.setSuffix(" kg/h")
        self._feed_spin.setStyleSheet(self._get_spinbox_style())
        form.addRow("Feed Rate:", self._feed_spin)

        # Termination mode selector
        self._term_mode_combo = QComboBox()
        self._term_mode_combo.addItems([
            "Time-based",
            "Mass-processed",
            "Steady-state",
            "Target d50",
        ])
        self._term_mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 24px 4px 8px;
                color: {COLORS.TEXT_PRIMARY};
                min-height: 24px;
            }}
            QComboBox:hover {{
                border-color: {COLORS.MILLING_PRIMARY};
                background: {COLORS.BG_SURFACE};
            }}
            QComboBox:focus {{
                border-color: {COLORS.MILLING_PRIMARY};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {COLORS.BORDER_SUBTLE};
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background: {COLORS.BG_SURFACE};
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {COLORS.TEXT_SECONDARY};
            }}
            QComboBox::down-arrow:hover {{
                border-top-color: {COLORS.TEXT_PRIMARY};
            }}
            QComboBox QAbstractItemView {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                selection-background-color: {COLORS.MILLING_MUTED};
                selection-color: {COLORS.TEXT_PRIMARY};
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 8px;
                min-height: 24px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: {COLORS.BG_HOVER};
            }}
        """)
        self._term_mode_combo.currentIndexChanged.connect(self._on_term_mode_changed)
        form.addRow("Termination:", self._term_mode_combo)

        # Duration (for time-based mode)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(1, 600)
        self._duration_spin.setValue(60)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setStyleSheet(self._get_spinbox_style())
        self._duration_label = QLabel("Duration:")
        form.addRow(self._duration_label, self._duration_spin)

        # Target mass (for mass-processed mode)
        self._target_mass_spin = QDoubleSpinBox()
        self._target_mass_spin.setRange(0.1, 100)
        self._target_mass_spin.setValue(1.0)
        self._target_mass_spin.setDecimals(2)
        self._target_mass_spin.setSuffix(" kg")
        self._target_mass_spin.setStyleSheet(self._get_spinbox_style())
        self._target_mass_label = QLabel("Target mass:")
        form.addRow(self._target_mass_label, self._target_mass_spin)
        self._target_mass_label.hide()
        self._target_mass_spin.hide()

        # Target d50 (for target d50 mode) — yellow pea: 23.7–31.1 µm (NIH)
        self._target_d50_spin = QDoubleSpinBox()
        self._target_d50_spin.setRange(15, 500)
        self._target_d50_spin.setValue(25)
        self._target_d50_spin.setSuffix(" µm")
        self._target_d50_spin.setStyleSheet(self._get_spinbox_style())
        self._target_d50_label = QLabel("Target d50:")
        form.addRow(self._target_d50_label, self._target_d50_spin)
        self._target_d50_label.hide()
        self._target_d50_spin.hide()

        # Min run time (for physics-based modes)
        self._min_time_spin = QDoubleSpinBox()
        self._min_time_spin.setRange(1, 60)
        self._min_time_spin.setValue(5)
        self._min_time_spin.setSuffix(" s")
        self._min_time_spin.setStyleSheet(self._get_spinbox_style())
        self._min_time_label = QLabel("Min time:")
        form.addRow(self._min_time_label, self._min_time_spin)
        self._min_time_label.hide()
        self._min_time_spin.hide()

        # Max run time (safety limit)
        self._max_time_spin = QDoubleSpinBox()
        self._max_time_spin.setRange(10, 600)
        self._max_time_spin.setValue(300)
        self._max_time_spin.setSuffix(" s")
        self._max_time_spin.setStyleSheet(self._get_spinbox_style())
        self._max_time_label = QLabel("Max time:")
        form.addRow(self._max_time_label, self._max_time_spin)
        self._max_time_label.hide()
        self._max_time_spin.hide()

        layout.addLayout(form)

        return group

    def _on_preset_changed(self, index: int):
        """Load preset recipe when user selects a preset (index 0 = Manual)."""
        if index <= 0 or index - 1 >= len(self._preset_recipes):
            return
        preset = self._preset_recipes[index - 1].copy()
        # Preset has duration_s; get_recipe uses duration_s for time-based mode
        self._preset_combo.blockSignals(True)
        self.set_recipe(preset)
        self._preset_combo.blockSignals(False)
        self._emit_recipe()

    def _on_term_mode_changed(self, index: int):
        """Handle termination mode selection change."""
        is_time = (index == 0)
        is_mass = (index == 1)
        is_steady = (index == 2)
        is_target_d50 = (index == 3)
        is_physics = not is_time

        # Show/hide duration (only for time-based)
        self._duration_label.setVisible(is_time)
        self._duration_spin.setVisible(is_time)

        # Show/hide target mass (only for mass mode)
        self._target_mass_label.setVisible(is_mass)
        self._target_mass_spin.setVisible(is_mass)

        # Show/hide target d50 (only for target d50 mode)
        self._target_d50_label.setVisible(is_target_d50)
        self._target_d50_spin.setVisible(is_target_d50)

        # Show/hide min/max time (for all physics-based modes)
        self._min_time_label.setVisible(is_physics)
        self._min_time_spin.setVisible(is_physics)
        self._max_time_label.setVisible(is_physics)
        self._max_time_spin.setVisible(is_physics)

        # Force layout update to reflect visibility changes
        self.updateGeometry()
        if self.parentWidget():
            self.parentWidget().updateGeometry()

        self._emit_recipe()

    def _on_rpm_changed(self, value: int):
        """Handle RPM slider change — update label and refresh D50 estimate."""
        self._rpm_value.setText(str(value))
        # D50 depends on tip speed, so re-evaluate when RPM changes
        self._on_aperture_changed(self._aperture_slider.value())

    def _get_hammer_tip_speed(self) -> float:
        """Compute hammer tip speed from current RPM and default geometry."""
        import math
        rpm = self._rpm_slider.value()
        # Default MillConfig geometry: rotor_diameter=0.20, hammer_length=0.08
        tip_radius = 0.10 + 0.08  # rotor_radius + hammer_length
        return tip_radius * rpm * 2.0 * math.pi / 60.0

    def _on_aperture_changed(self, value: int):
        """Handle aperture slider change - update D50 estimate using config model."""
        from airclassifier.milling.config import ScreenConfig
        aperture_mm = value / 100.0
        self._aperture_value.setText(f"{aperture_mm:.2f} mm")

        # Use the canonical D50 model with actual tip speed from RPM slider
        sc = ScreenConfig(aperture_mm=aperture_mm)
        d50_um = sc.estimated_d50_um_at_tip_speed(self._get_hammer_tip_speed())

        if d50_um <= 25:
            quality = "Excellent for protein separation"
            color = COLORS.SUCCESS
        elif d50_um <= 35:
            quality = "Good for starch/protein fractionation"
            color = COLORS.ACCENT
        elif d50_um <= 50:
            quality = "Moderate - consider finer screen or higher RPM"
            color = COLORS.WARNING
        else:
            quality = "Coarse - use 0.3-0.84 mm screen, 5000-7200 RPM"
            color = COLORS.DANGER

        self._d50_hint.setText(f"D50 ~{d50_um:.0f} µm • {quality}")
        self._d50_hint.setStyleSheet(f"color: {color}; font-size: 8pt; font-style: italic;")

        self._emit_recipe()

    def _get_spinbox_style(self) -> str:
        """Get spinbox stylesheet."""
        return f"""
            QDoubleSpinBox, QSpinBox {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS.TEXT_PRIMARY};
                min-height: 24px;
            }}
            QDoubleSpinBox:focus, QSpinBox:focus {{
                border-color: {COLORS.MILLING_PRIMARY};
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {{
                background: {COLORS.BG_SURFACE};
                border: none;
                width: 18px;
            }}
        """

    def _connect_signals(self):
        """Connect internal signals."""
        self._run_btn.clicked.connect(self.run_clicked.emit)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._config_btn.clicked.connect(self.config_clicked.emit)

        # Recipe changes
        self._rpm_slider.valueChanged.connect(self._emit_recipe)
        # Note: _aperture_slider connected to _on_aperture_changed in _create_recipe_group
        self._seeds_feed_mass_spin.valueChanged.connect(self._emit_recipe)
        self._feed_spin.valueChanged.connect(self._emit_recipe)
        self._duration_spin.valueChanged.connect(self._emit_recipe)
        self._target_mass_spin.valueChanged.connect(self._emit_recipe)
        self._target_d50_spin.valueChanged.connect(self._emit_recipe)
        self._min_time_spin.valueChanged.connect(self._emit_recipe)
        self._max_time_spin.valueChanged.connect(self._emit_recipe)

    def _emit_recipe(self):
        """Emit current recipe values."""
        self.recipe_changed.emit(self.get_recipe())

    # Public API

    def get_recipe(self) -> Dict[str, Any]:
        """Get current recipe parameters including termination config."""
        modes = ["time", "mass", "steady_state", "target_d50"]
        term_mode = modes[self._term_mode_combo.currentIndex()]

        return {
            "rotor_rpm": self._rpm_slider.value(),
            "screen_aperture_mm": self._aperture_slider.value() / 100,
            "seeds_feed_mass_kg": self._seeds_feed_mass_spin.value(),
            "feed_rate_kg_per_hr": self._feed_spin.value(),
            "duration_s": self._duration_spin.value(),
            "termination_mode": term_mode,
            "target_mass_kg": self._target_mass_spin.value(),
            "target_d50_um": self._target_d50_spin.value(),
            "min_run_time_s": self._min_time_spin.value(),
            "max_run_time_s": self._max_time_spin.value(),
        }

    def get_termination_mode(self) -> str:
        """Get current termination mode string."""
        modes = ["time", "mass", "steady_state", "target_d50"]
        return modes[self._term_mode_combo.currentIndex()]

    def set_recipe(self, recipe: Dict[str, Any]):
        """Set recipe parameters."""
        if "rotor_rpm" in recipe:
            self._rpm_slider.setValue(int(recipe["rotor_rpm"]))
        if "screen_aperture_mm" in recipe:
            self._aperture_slider.setValue(int(recipe["screen_aperture_mm"] * 100))
        if "seeds_feed_mass_kg" in recipe:
            self._seeds_feed_mass_spin.setValue(recipe["seeds_feed_mass_kg"])
        if "feed_rate_kg_per_hr" in recipe:
            self._feed_spin.setValue(recipe["feed_rate_kg_per_hr"])
        if "duration_s" in recipe:
            self._duration_spin.setValue(recipe["duration_s"])

    def set_running(self, running: bool):
        """Update button states for running simulation."""
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._progress.setVisible(running)

    def set_progress(self, value: int, maximum: int = 100):
        """Update progress bar."""
        self._progress.setMaximum(maximum)
        self._progress.setValue(value)

    def log(self, message: str):
        """Append message to log."""
        self._log_text.append(message)
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear log output."""
        self._log_text.clear()
