"""
Time Series Chart
=================

Multi-line time series charts for milling simulation results.
Series are split into separate graphs by scale (same units together).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QPainterPath, QFont, QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy, QToolTip, QFrame, QGridLayout,
    QScrollArea,
)

from ...theme import COLORS


@dataclass
class SeriesConfig:
    """Configuration for a single data series."""
    name: str
    values: List[float]
    color: str
    unit: str
    visible: bool = True
    y_axis: str = "left"  # "left" or "right"


# Minimum height for each individual chart (larger for better readability)
_CHART_CANVAS_MIN_HEIGHT = 220
_CHART_FRAME_MIN_HEIGHT = 260

_SCROLLBAR_STYLE = f"""
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: {COLORS.BG_DARKEST};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS.BG_HOVER};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLORS.BORDER};
    }}
    QScrollBar:horizontal {{
        background: {COLORS.BG_DARKEST};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {COLORS.BG_HOVER};
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {COLORS.BORDER};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        height: 0;
        width: 0;
    }}
"""


class _SingleScaleChart(QFrame):
    """One chart with a single Y-axis scale (same units)."""

    def __init__(self, title: str, unit: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = title
        self._unit = unit
        self._time_values: List[float] = []
        self._series: Dict[str, SeriesConfig] = {}
        self.setMinimumHeight(_CHART_FRAME_MIN_HEIGHT)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        title_lbl = QLabel(f"{title} ({unit})")
        title_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt; font-weight: 600;")
        layout.addWidget(title_lbl)
        self._canvas = _TimeSeriesCanvas(self)
        self._canvas.setMinimumHeight(_CHART_CANVAS_MIN_HEIGHT)
        self._canvas.setMinimumWidth(280)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas, 1)

    def set_data(self, time_values: List[float], series: Dict[str, SeriesConfig]) -> None:
        """Set data; all series share the same Y scale (same unit)."""
        self._time_values = time_values
        self._series = {k: SeriesConfig(
            name=s.name, values=s.values, color=s.color, unit=s.unit,
            visible=s.visible, y_axis="left"
        ) for k, s in series.items()}
        self._canvas.set_data(time_values, self._series)

    def clear(self) -> None:
        self._time_values = []
        self._series = {}
        self._canvas.clear()


class TimeSeriesChart(QWidget):
    """Process timeline with separate charts by scale.

    Each graph shows only series with the same unit so the Y-axis is meaningful:
    - d50 (µm)
    - Holdup (kg)
    - Chamber count (#)
    - Passed & Breakage (#/step)
    - Throughput (kg/h)
    - Power (kW)
    """

    cursor_moved = Signal(float, dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._time_values: List[float] = []
        self._series: Dict[str, SeriesConfig] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("Process Timeline")
        title.setStyleSheet(f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(title)

        # Grid of same-scale charts: each chart in its own scroll area (individually scrollable)
        grid = QGridLayout()
        grid.setSpacing(8)

        def _wrap_scroll(chart: _SingleScaleChart) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(chart)
            scroll.setStyleSheet(_SCROLLBAR_STYLE)
            scroll.setMinimumHeight(_CHART_FRAME_MIN_HEIGHT + 8)
            return scroll

        self._chart_d50 = _SingleScaleChart("d50", "µm")
        self._chart_holdup = _SingleScaleChart("Holdup", "kg")
        self._chart_chamber = _SingleScaleChart("Chamber count", "#")
        self._chart_per_step = _SingleScaleChart("Passed & Breakage", "#/step")
        self._chart_throughput = _SingleScaleChart("Throughput", "kg/h")
        self._chart_power = _SingleScaleChart("Power", "kW")

        grid.addWidget(_wrap_scroll(self._chart_d50), 0, 0)
        grid.addWidget(_wrap_scroll(self._chart_holdup), 0, 1)
        grid.addWidget(_wrap_scroll(self._chart_chamber), 0, 2)
        grid.addWidget(_wrap_scroll(self._chart_per_step), 1, 0)
        grid.addWidget(_wrap_scroll(self._chart_throughput), 1, 1)
        grid.addWidget(_wrap_scroll(self._chart_power), 1, 2)

        layout.addLayout(grid)

    def set_data(self, history: List) -> None:
        """Populate all charts from simulation history."""
        if not history:
            self.clear()
            return

        step = max(1, len(history) // 500)
        sampled = history[::step]
        self._time_values = [s.time_s for s in sampled]

        d50_values = [s.d50_m * 1e6 for s in sampled]
        holdup_values = [s.holdup_kg for s in sampled]
        chamber_values = [float(s.num_particles) for s in sampled]
        passed_values = [float(s.num_passed_screen) for s in sampled]
        breakage_values = [float(s.num_breakage_events) for s in sampled]
        throughput_values = [s.discharge_rate_kg_per_s * 3600 for s in sampled]
        power_values = [s.power_kw for s in sampled]

        self._series["d50"] = SeriesConfig("d50", d50_values, COLORS.KPI_SIZE, "µm", True, "left")
        self._series["holdup"] = SeriesConfig("Holdup", holdup_values, COLORS.MILLING_PRIMARY, "kg", True, "left")
        self._series["chamber_count"] = SeriesConfig("Chamber", chamber_values, COLORS.SUCCESS, "#", True, "left")
        self._series["passed"] = SeriesConfig("Passed", passed_values, COLORS.INFO, "#/step", True, "left")
        self._series["breakage"] = SeriesConfig("Breakage", breakage_values, COLORS.WARNING, "#/step", True, "left")
        self._series["throughput"] = SeriesConfig("Throughput", throughput_values, COLORS.KPI_THROUGHPUT, "kg/h", True, "left")
        self._series["power"] = SeriesConfig("Power", power_values, COLORS.KPI_POWER, "kW", True, "left")

        self._chart_d50.set_data(self._time_values, {"d50": self._series["d50"]})
        self._chart_holdup.set_data(self._time_values, {"holdup": self._series["holdup"]})
        self._chart_chamber.set_data(self._time_values, {"chamber_count": self._series["chamber_count"]})
        self._chart_per_step.set_data(self._time_values, {
            "passed": self._series["passed"],
            "breakage": self._series["breakage"],
        })
        self._chart_throughput.set_data(self._time_values, {"throughput": self._series["throughput"]})
        self._chart_power.set_data(self._time_values, {"power": self._series["power"]})

    def clear(self) -> None:
        self._time_values = []
        self._series = {}
        self._chart_d50.clear()
        self._chart_holdup.clear()
        self._chart_chamber.clear()
        self._chart_per_step.clear()
        self._chart_throughput.clear()
        self._chart_power.clear()


class _TimeSeriesCanvas(QWidget):
    """Internal canvas for rendering the time series chart."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._time_values: List[float] = []
        self._series: Dict[str, SeriesConfig] = {}
        self._hover_index = -1

        self.setMouseTracking(True)
        self.setStyleSheet(f"background: {COLORS.BG_DARKEST}; border-radius: 6px;")

    def set_data(self, time_values: List[float], series: Dict[str, SeriesConfig]):
        """Set chart data."""
        self._time_values = time_values
        self._series = series
        self.update()

    def clear(self):
        """Clear all data."""
        self._time_values = []
        self._series = {}
        self.update()

    def paintEvent(self, event):
        """Render the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        padding = {"left": 55, "right": 55, "top": 15, "bottom": 30}

        chart_x = padding["left"]
        chart_y = padding["top"]
        chart_width = rect.width() - padding["left"] - padding["right"]
        chart_height = rect.height() - padding["top"] - padding["bottom"]

        # Background
        painter.fillRect(rect, QColor(COLORS.BG_DARKEST))

        if not self._time_values or not self._series:
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No time series data")
            return

        # Calculate scales
        t_min = min(self._time_values)
        t_max = max(self._time_values)
        t_range = t_max - t_min if t_max != t_min else 1

        # Get visible series
        left_series = [s for s in self._series.values() if s.visible and s.y_axis == "left"]
        right_series = [s for s in self._series.values() if s.visible and s.y_axis == "right"]

        # Calculate Y ranges
        def get_range(series_list):
            if not series_list:
                return 0, 100
            all_vals = []
            for s in series_list:
                all_vals.extend([v for v in s.values if v is not None and not np.isnan(v)])
            if not all_vals:
                return 0, 100
            y_min = min(0, min(all_vals))  # Include 0
            y_max = max(all_vals) * 1.1  # 10% padding
            return y_min, max(y_max, 0.01)

        left_min, left_max = get_range(left_series)
        right_min, right_max = get_range(right_series)

        # Draw grid
        painter.setPen(QPen(QColor(COLORS.CHART_GRID), 1))
        for i in range(5):
            y = chart_y + chart_height * (1 - i / 4)
            painter.drawLine(int(chart_x), int(y), int(chart_x + chart_width), int(y))

        # Draw vertical grid (5 time divisions)
        for i in range(6):
            x = chart_x + chart_width * i / 5
            painter.drawLine(int(x), int(chart_y), int(x), int(chart_y + chart_height))

        # Draw series
        for series in self._series.values():
            if not series.visible or not series.values:
                continue

            color = QColor(series.color)
            pen = QPen(color, 2)
            painter.setPen(pen)

            # Determine Y scale
            if series.y_axis == "left":
                y_min, y_max = left_min, left_max
            else:
                y_min, y_max = right_min, right_max

            y_range = y_max - y_min if y_max != y_min else 1

            # Draw line path
            path = QPainterPath()
            first_point = True

            for i, (t, v) in enumerate(zip(self._time_values, series.values)):
                if v is None or np.isnan(v):
                    continue

                x = chart_x + (t - t_min) / t_range * chart_width
                y = chart_y + chart_height - (v - y_min) / y_range * chart_height

                if first_point:
                    path.moveTo(x, y)
                    first_point = False
                else:
                    path.lineTo(x, y)

            painter.drawPath(path)

            # Draw gradient fill under curve (subtle)
            fill_path = QPainterPath(path)
            if not first_point:
                # Close the path to bottom
                last_t = self._time_values[-1]
                first_t = self._time_values[0]
                fill_path.lineTo(chart_x + (last_t - t_min) / t_range * chart_width, chart_y + chart_height)
                fill_path.lineTo(chart_x + (first_t - t_min) / t_range * chart_width, chart_y + chart_height)
                fill_path.closeSubpath()

                gradient = QLinearGradient(0, chart_y, 0, chart_y + chart_height)
                fill_color = QColor(color)
                fill_color.setAlpha(40)
                gradient.setColorAt(0, fill_color)
                fill_color.setAlpha(5)
                gradient.setColorAt(1, fill_color)
                painter.fillPath(fill_path, gradient)

        # Draw axes
        painter.setPen(QPen(QColor(COLORS.CHART_AXIS), 1))
        # Left Y axis
        painter.drawLine(int(chart_x), int(chart_y), int(chart_x), int(chart_y + chart_height))
        # Right Y axis
        painter.drawLine(int(chart_x + chart_width), int(chart_y),
                        int(chart_x + chart_width), int(chart_y + chart_height))
        # X axis
        painter.drawLine(int(chart_x), int(chart_y + chart_height),
                        int(chart_x + chart_width), int(chart_y + chart_height))

        # Axis labels
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)

        # Left Y axis labels
        if left_series:
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            for i in range(5):
                val = left_min + (left_max - left_min) * i / 4
                y = chart_y + chart_height * (1 - i / 4)
                fmt = f"{val:.1f}" if abs(val) < 1000 and (left_max - left_min) < 100 else f"{val:.0f}"
                painter.drawText(int(chart_x - 50), int(y + 4), fmt)

        # Right Y axis labels (power/throughput)
        if right_series:
            painter.setPen(QColor(COLORS.KPI_POWER))
            for i in range(5):
                val = right_min + (right_max - right_min) * i / 4
                y = chart_y + chart_height * (1 - i / 4)
                painter.drawText(int(chart_x + chart_width + 5), int(y + 4), f"{val:.1f}")

        # X axis labels (time)
        painter.setPen(QColor(COLORS.TEXT_MUTED))
        for i in range(6):
            t = t_min + t_range * i / 5
            x = chart_x + chart_width * i / 5
            painter.drawText(int(x - 10), int(rect.height() - 5), f"{t:.0f}s")

        # Draw crosshair at hover position
        if 0 <= self._hover_index < len(self._time_values):
            t = self._time_values[self._hover_index]
            x = chart_x + (t - t_min) / t_range * chart_width

            painter.setPen(QPen(QColor(COLORS.TEXT_MUTED), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(x), int(chart_y), int(x), int(chart_y + chart_height))

            # Draw value dots on each line
            for series in self._series.values():
                if not series.visible or self._hover_index >= len(series.values):
                    continue

                v = series.values[self._hover_index]
                if v is None or np.isnan(v):
                    continue

                if series.y_axis == "left":
                    y_min, y_max = left_min, left_max
                else:
                    y_min, y_max = right_min, right_max

                y_range = y_max - y_min if y_max != y_min else 1
                y = chart_y + chart_height - (v - y_min) / y_range * chart_height

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(series.color))
                painter.drawEllipse(QPointF(x, y), 4, 4)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hover detection."""
        if not self._time_values:
            return

        rect = self.rect()
        padding = {"left": 55, "right": 55, "top": 15, "bottom": 30}
        chart_x = padding["left"]
        chart_width = rect.width() - padding["left"] - padding["right"]

        x = event.position().x()
        if x < chart_x or x > chart_x + chart_width:
            if self._hover_index >= 0:
                self._hover_index = -1
                self.update()
            return

        # Find nearest data point
        t_min = min(self._time_values)
        t_max = max(self._time_values)
        t_range = t_max - t_min if t_max != t_min else 1

        t_cursor = t_min + (x - chart_x) / chart_width * t_range

        # Binary search for closest index
        idx = min(range(len(self._time_values)),
                  key=lambda i: abs(self._time_values[i] - t_cursor))

        if idx != self._hover_index:
            self._hover_index = idx
            self.update()

            # Show tooltip
            t = self._time_values[idx]
            tooltip_lines = [f"Time: {t:.1f}s"]
            for series in self._series.values():
                if series.visible and idx < len(series.values):
                    v = series.values[idx]
                    if v is not None and not np.isnan(v):
                        tooltip_lines.append(f"{series.name}: {v:.1f} {series.unit}")

            QToolTip.showText(
                event.globalPosition().toPoint(),
                "\n".join(tooltip_lines)
            )

    def leaveEvent(self, event):
        """Clear hover state."""
        self._hover_index = -1
        self.update()
