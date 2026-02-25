"""
Sparkline Widget
================

Minimal trend line visualization for KPI cards.
Shows recent value history as a small inline chart.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional

from PySide6.QtCore import Qt, QPointF, QRectF, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QPainterPath, QLinearGradient, QColor, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy

from ...theme import COLORS, ANIMATIONS


class SparklineWidget(QWidget):
    """Compact sparkline chart showing recent value trends.
    
    Features:
        - Smooth curved line interpolation
        - Gradient fill under curve
        - Animated value additions
        - Auto-scaling Y axis
        - Configurable colors and size
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        max_points: int = 30,
        line_color: str = COLORS.CHART_PRIMARY,
        fill_color: Optional[str] = None,
        show_fill: bool = True,
        line_width: float = 1.5,
    ):
        super().__init__(parent)
        self._max_points = max_points
        self._values: deque = deque(maxlen=max_points)
        self._line_color = QColor(line_color)
        self._fill_color = QColor(fill_color) if fill_color else self._line_color
        self._show_fill = show_fill
        self._line_width = line_width
        
        # Animation state
        self._animated_value = 0.0
        self._target_value = 0.0
        self._animation: Optional[QPropertyAnimation] = None
        
        # Sizing
        self.setMinimumSize(60, 20)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)
        
        # Tooltip
        self.setMouseTracking(True)

    def add_value(self, value: float, animate: bool = True):
        """Add a new value to the sparkline.
        
        Args:
            value: The value to add
            animate: Whether to animate the addition
        """
        if animate and self._values:
            self._target_value = value
            self._start_animation()
        else:
            self._values.append(value)
            self.update()

    def _start_animation(self):
        """Animate the addition of a new value."""
        if self._animation is not None:
            self._animation.stop()
        
        self._animation = QPropertyAnimation(self, b"animatedValue")
        self._animation.setDuration(ANIMATIONS.NORMAL)
        self._animation.setStartValue(self._values[-1] if self._values else 0.0)
        self._animation.setEndValue(self._target_value)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_animation_finished)
        self._animation.start()

    def _on_animation_finished(self):
        """Called when animation completes."""
        self._values.append(self._target_value)
        self.update()

    def _get_animated_value(self) -> float:
        return self._animated_value

    def _set_animated_value(self, value: float):
        self._animated_value = value
        self.update()

    animatedValue = Property(float, _get_animated_value, _set_animated_value)

    def set_values(self, values: List[float]):
        """Set all values at once (no animation)."""
        self._values.clear()
        for v in values[-self._max_points:]:
            self._values.append(v)
        self.update()

    def clear(self):
        """Clear all values."""
        self._values.clear()
        self.update()

    @property
    def values(self) -> List[float]:
        """Get current values as a list."""
        return list(self._values)

    @property
    def trend(self) -> float:
        """Calculate trend as percentage change from first to last."""
        if len(self._values) < 2:
            return 0.0
        first = self._values[0]
        last = self._values[-1]
        if first == 0:
            return 0.0
        return ((last - first) / abs(first)) * 100

    def set_line_color(self, color: str):
        """Set the line color."""
        self._line_color = QColor(color)
        if not self._fill_color:
            self._fill_color = self._line_color
        self.update()

    def paintEvent(self, event):
        """Render the sparkline."""
        if len(self._values) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        padding = 2
        chart_rect = QRectF(
            padding, padding,
            rect.width() - 2 * padding,
            rect.height() - 2 * padding
        )

        # Calculate min/max for scaling
        values = list(self._values)
        if self._animation and self._animation.state() == QPropertyAnimation.State.Running:
            values = values[:-1] + [self._animated_value] if values else [self._animated_value]
        
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val
        
        if value_range == 0:
            value_range = 1  # Prevent division by zero
            min_val -= 0.5
            max_val += 0.5

        # Build path
        path = QPainterPath()
        n = len(values)
        
        for i, val in enumerate(values):
            x = chart_rect.left() + (i / (n - 1)) * chart_rect.width()
            y = chart_rect.bottom() - ((val - min_val) / value_range) * chart_rect.height()
            
            if i == 0:
                path.moveTo(x, y)
            else:
                # Smooth curve using quadratic bezier
                prev_x = chart_rect.left() + ((i - 1) / (n - 1)) * chart_rect.width()
                ctrl_x = (prev_x + x) / 2
                path.quadTo(ctrl_x, path.currentPosition().y(), ctrl_x, y)
                path.lineTo(x, y)

        # Draw fill gradient
        if self._show_fill:
            fill_path = QPainterPath(path)
            fill_path.lineTo(chart_rect.right(), chart_rect.bottom())
            fill_path.lineTo(chart_rect.left(), chart_rect.bottom())
            fill_path.closeSubpath()

            gradient = QLinearGradient(0, chart_rect.top(), 0, chart_rect.bottom())
            fill_color = QColor(self._fill_color)
            fill_color.setAlpha(60)
            gradient.setColorAt(0, fill_color)
            fill_color.setAlpha(10)
            gradient.setColorAt(1, fill_color)
            
            painter.fillPath(fill_path, gradient)

        # Draw line
        pen = QPen(self._line_color, self._line_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

        # Draw end point dot
        if values:
            last_x = chart_rect.right()
            last_y = chart_rect.bottom() - ((values[-1] - min_val) / value_range) * chart_rect.height()
            painter.setBrush(self._line_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(last_x, last_y), 2.5, 2.5)

    def enterEvent(self, event):
        """Show tooltip with trend info."""
        if len(self._values) >= 2:
            trend = self.trend
            trend_str = f"+{trend:.1f}%" if trend >= 0 else f"{trend:.1f}%"
            self.setToolTip(f"Trend: {trend_str}\nMin: {min(self._values):.1f}\nMax: {max(self._values):.1f}")
        super().enterEvent(event)
