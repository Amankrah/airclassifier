"""
Animated KPI Card Widget
========================

Modern KPI display card with sparklines, trend indicators,
and smooth value animations. Glassmorphism styling.
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, Property,
    QTimer, Signal,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QWidget, QSizePolicy, QGraphicsDropShadowEffect,
)

from ...theme import COLORS, ANIMATIONS
from .sparkline import SparklineWidget


class TrendDirection(Enum):
    """Trend direction for KPI values."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class GlassCard(QFrame):
    """Base card with glassmorphism effect."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_style()

    def _setup_style(self):
        self.setObjectName("glassCard")
        self.setStyleSheet(f"""
            QFrame#glassCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS.GLASS_START},
                    stop:1 {COLORS.GLASS_END});
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
            }}
        """)

        # Add subtle shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)


class AnimatedKPICard(GlassCard):
    """Modern KPI card with sparkline and trend visualization.

    Features:
        - Animated value counter
        - Sparkline trend chart
        - Delta percentage badge
        - Trend direction indicator
        - Semantic color coding
        - Pulse animation on update

    Signals:
        value_changed(float): Emitted when value changes
        clicked(): Emitted when card is clicked
    """

    value_changed = Signal(float)
    clicked = Signal()

    def __init__(
        self,
        title: str,
        unit: str = "",
        semantic_color: str = COLORS.ACCENT,
        show_sparkline: bool = True,
        show_delta: bool = True,
        precision: int = 0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._title = title
        self._unit = unit
        self._semantic_color = semantic_color
        self._show_sparkline = show_sparkline
        self._show_delta = show_delta
        self._precision = precision

        # Value state
        self._current_value: float = 0.0
        self._animated_value: float = 0.0
        self._previous_value: float = 0.0
        self._target_value: Optional[float] = None

        # Animation
        self._value_animation: Optional[QPropertyAnimation] = None

        self._setup_ui()

    def _setup_ui(self):
        """Build the card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        # Top row: value + trend
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # Value label with animated counter
        self._value_label = QLabel("--")
        self._value_label.setStyleSheet(f"""
            font-size: 20pt;
            font-weight: 700;
            color: {self._semantic_color};
            background: transparent;
            border: none;
        """)
        top_row.addWidget(self._value_label)

        # Unit label
        if self._unit:
            unit_label = QLabel(self._unit)
            unit_label.setStyleSheet(f"""
                font-size: 10pt;
                color: {COLORS.TEXT_MUTED};
                background: transparent;
                border: none;
                padding-top: 8px;
            """)
            top_row.addWidget(unit_label)

        top_row.addStretch()

        # Delta badge
        if self._show_delta:
            self._delta_badge = QLabel("")
            self._delta_badge.setStyleSheet(f"""
                font-size: 9pt;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 4px;
                background: transparent;
                border: none;
            """)
            self._delta_badge.hide()
            top_row.addWidget(self._delta_badge)

        layout.addLayout(top_row)

        # Sparkline
        if self._show_sparkline:
            self._sparkline = SparklineWidget(
                parent=self,
                line_color=self._semantic_color,
                max_points=30,
            )
            layout.addWidget(self._sparkline)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(6)

        # Trend icon
        self._trend_icon = QLabel("")
        self._trend_icon.setStyleSheet("""
            font-size: 10pt;
            background: transparent;
            border: none;
        """)
        title_row.addWidget(self._trend_icon)

        # Title
        title_label = QLabel(self._title)
        title_label.setStyleSheet(f"""
            font-size: 9pt;
            color: {COLORS.TEXT_SECONDARY};
            background: transparent;
            border: none;
        """)
        title_row.addWidget(title_label)
        title_row.addStretch()

        layout.addLayout(title_row)

        # Sizing
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(100 if self._show_sparkline else 70)

        # Cursor for clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_value(self, value: float, animate: bool = True):
        """Set the KPI value with optional animation.

        Args:
            value: New value to display
            animate: Whether to animate the transition
        """
        self._previous_value = self._current_value
        self._current_value = value

        # Update sparkline
        if self._show_sparkline and hasattr(self, "_sparkline"):
            self._sparkline.add_value(value, animate=animate)

        # Animate value counter
        if animate and self._previous_value != value:
            self._animate_value(value)
        else:
            self._animated_value = value
            self._update_display()

        # Update delta
        if self._show_delta:
            self._update_delta()

        self.value_changed.emit(value)

    def _animate_value(self, target: float):
        """Animate the value counter."""
        if self._value_animation is not None:
            self._value_animation.stop()

        self._value_animation = QPropertyAnimation(self, b"animatedValue")
        self._value_animation.setDuration(ANIMATIONS.SLOW)
        self._value_animation.setStartValue(self._animated_value)
        self._value_animation.setEndValue(target)
        self._value_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._value_animation.start()

        # Trigger pulse effect
        self._pulse_effect()

    def _pulse_effect(self):
        """Brief pulse animation on value change."""
        original_style = self._value_label.styleSheet()
        pulse_style = original_style.replace(
            self._semantic_color,
            COLORS.TEXT_INVERSE
        )
        self._value_label.setStyleSheet(pulse_style)

        QTimer.singleShot(100, lambda: self._value_label.setStyleSheet(original_style))

    def _get_animated_value(self) -> float:
        return self._animated_value

    def _set_animated_value(self, value: float):
        self._animated_value = value
        self._update_display()

    animatedValue = Property(float, _get_animated_value, _set_animated_value)

    def _update_display(self):
        """Update the value label text."""
        if self._precision == 0:
            text = f"{int(self._animated_value):,}"
        else:
            text = f"{self._animated_value:,.{self._precision}f}"
        self._value_label.setText(text)

    def _update_delta(self):
        """Update the delta badge."""
        if not hasattr(self, "_delta_badge") or self._previous_value == 0:
            return

        delta = self._current_value - self._previous_value
        if self._previous_value != 0:
            delta_pct = (delta / abs(self._previous_value)) * 100
        else:
            delta_pct = 0

        # Determine direction
        if abs(delta_pct) < 0.1:
            icon = "~"
            color = COLORS.TEXT_MUTED
        elif delta > 0:
            icon = "+"
            color = COLORS.SUCCESS
        else:
            icon = "-"
            color = COLORS.DANGER

        # Update badge
        if abs(delta_pct) >= 0.1:
            self._delta_badge.setText(f"{icon}{abs(delta_pct):.1f}%")
            self._delta_badge.setStyleSheet(f"""
                font-size: 9pt;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 4px;
                background: {color}22;
                color: {color};
                border: none;
            """)
            self._delta_badge.show()

        # Update trend icon
        self._trend_icon.setText(icon)
        self._trend_icon.setStyleSheet(f"""
            font-size: 10pt;
            color: {color};
            background: transparent;
            border: none;
        """)

    def set_target(self, target: float):
        """Set a target value for comparison."""
        self._target_value = target
        self._update_delta()

    def set_semantic_color(self, color: str):
        """Update the semantic color."""
        self._semantic_color = color
        self._value_label.setStyleSheet(f"""
            font-size: 20pt;
            font-weight: 700;
            color: {color};
            background: transparent;
            border: none;
        """)
        if self._show_sparkline and hasattr(self, "_sparkline"):
            self._sparkline.set_line_color(color)

    @property
    def value(self) -> float:
        return self._current_value

    @property
    def trend(self) -> float:
        """Get sparkline trend percentage."""
        if self._show_sparkline and hasattr(self, "_sparkline"):
            return self._sparkline.trend
        return 0.0

    def clear(self):
        """Reset the card."""
        self._current_value = 0.0
        self._previous_value = 0.0
        self._animated_value = 0.0
        self._value_label.setText("--")
        if self._show_sparkline and hasattr(self, "_sparkline"):
            self._sparkline.clear()
        if self._show_delta and hasattr(self, "_delta_badge"):
            self._delta_badge.hide()

    def mousePressEvent(self, event):
        """Handle click events."""
        self.clicked.emit()
        super().mousePressEvent(event)
