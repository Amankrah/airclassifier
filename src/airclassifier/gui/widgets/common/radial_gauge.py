"""
Radial Gauge Widget
===================

Circular gauge for displaying power, load, or efficiency metrics.
Features smooth animations and configurable appearance.
"""

from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, QRectF, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QConicalGradient
from PySide6.QtWidgets import QWidget, QSizePolicy

from ...theme import COLORS, ANIMATIONS


class RadialGaugeWidget(QWidget):
    """Circular gauge with animated needle/arc and value display.
    
    Features:
        - Animated value changes
        - Gradient arc coloring
        - Center value text
        - Configurable ranges and colors
        - Warning/danger thresholds
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        min_value: float = 0.0,
        max_value: float = 100.0,
        warning_threshold: float = 70.0,
        danger_threshold: float = 90.0,
        unit: str = "%",
        title: str = "",
    ):
        super().__init__(parent)
        
        self._min_value = min_value
        self._max_value = max_value
        self._current_value = min_value
        self._animated_value = min_value
        self._warning_threshold = warning_threshold
        self._danger_threshold = danger_threshold
        self._unit = unit
        self._title = title
        
        # Colors
        self._bg_color = QColor(COLORS.BG_DARK)
        self._track_color = QColor(COLORS.BORDER_SUBTLE)
        self._normal_color = QColor(COLORS.SUCCESS)
        self._warning_color = QColor(COLORS.WARNING)
        self._danger_color = QColor(COLORS.DANGER)
        self._text_color = QColor(COLORS.TEXT_PRIMARY)
        
        # Animation
        self._animation: Optional[QPropertyAnimation] = None
        
        # Sizing
        self.setMinimumSize(80, 80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_value(self, value: float, animate: bool = True):
        """Set the gauge value.
        
        Args:
            value: New value (clamped to min/max)
            animate: Whether to animate the change
        """
        value = max(self._min_value, min(self._max_value, value))
        self._current_value = value
        
        if animate:
            if self._animation is not None:
                self._animation.stop()
            
            self._animation = QPropertyAnimation(self, b"animatedValue")
            self._animation.setDuration(ANIMATIONS.SLOW)
            self._animation.setStartValue(self._animated_value)
            self._animation.setEndValue(value)
            self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._animation.start()
        else:
            self._animated_value = value
            self.update()

    def _get_animated_value(self) -> float:
        return self._animated_value

    def _set_animated_value(self, value: float):
        self._animated_value = value
        self.update()

    animatedValue = Property(float, _get_animated_value, _set_animated_value)

    @property
    def value(self) -> float:
        return self._current_value

    def set_range(self, min_value: float, max_value: float):
        """Set the value range."""
        self._min_value = min_value
        self._max_value = max_value
        self.update()

    def set_thresholds(self, warning: float, danger: float):
        """Set warning and danger thresholds."""
        self._warning_threshold = warning
        self._danger_threshold = danger
        self.update()

    def _get_color_for_value(self, value: float) -> QColor:
        """Get the appropriate color based on value."""
        if value >= self._danger_threshold:
            return self._danger_color
        elif value >= self._warning_threshold:
            return self._warning_color
        return self._normal_color

    def paintEvent(self, event):
        """Render the gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        size = min(self.width(), self.height())
        rect = QRectF(
            (self.width() - size) / 2 + 4,
            (self.height() - size) / 2 + 4,
            size - 8,
            size - 8
        )
        
        arc_width = size * 0.12
        inner_rect = rect.adjusted(arc_width / 2, arc_width / 2, -arc_width / 2, -arc_width / 2)

        # Draw background track (270 degree arc, starting from bottom-left)
        start_angle = 225 * 16  # Qt uses 1/16th of a degree
        span_angle = -270 * 16

        pen = QPen(self._track_color, arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(inner_rect, start_angle, span_angle)

        # Calculate value angle
        value_range = self._max_value - self._min_value
        if value_range == 0:
            value_range = 1
        value_fraction = (self._animated_value - self._min_value) / value_range
        value_span = int(-270 * 16 * value_fraction)

        # Draw value arc with color
        value_color = self._get_color_for_value(self._animated_value)
        pen = QPen(value_color, arc_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(inner_rect, start_angle, value_span)

        # Draw center text
        painter.setPen(self._text_color)
        
        # Value text
        font = QFont()
        font.setPixelSize(int(size * 0.22))
        font.setBold(True)
        painter.setFont(font)
        
        value_text = f"{self._animated_value:.0f}"
        text_rect = QRectF(rect.left(), rect.top() + rect.height() * 0.3, rect.width(), rect.height() * 0.3)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, value_text)
        
        # Unit text
        font.setPixelSize(int(size * 0.10))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(COLORS.TEXT_MUTED))
        
        unit_rect = QRectF(rect.left(), rect.top() + rect.height() * 0.55, rect.width(), rect.height() * 0.15)
        painter.drawText(unit_rect, Qt.AlignmentFlag.AlignCenter, self._unit)
        
        # Title text
        if self._title:
            title_rect = QRectF(rect.left(), rect.top() + rect.height() * 0.70, rect.width(), rect.height() * 0.15)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self._title)

    def enterEvent(self, event):
        """Show tooltip with full value."""
        self.setToolTip(f"{self._title}: {self._current_value:.2f} {self._unit}")
        super().enterEvent(event)
