"""
Time Series Chart
=================

Multi-line time series chart for milling simulation results.
Displays d50, power, and throughput over time with hover tooltips.
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
    QCheckBox, QSizePolicy, QToolTip, QFrame,
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


class TimeSeriesChart(QWidget):
    """Multi-line time series chart with dual Y-axes.

    Features:
        - Multiple series (d50, power, throughput)
        - Dual Y-axes for different units
        - Hover crosshair with tooltips
        - Series visibility toggles
        - Glassmorphism styling
    """

    cursor_moved = Signal(float, dict)  # time, values_dict

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Data
        self._time_values: List[float] = []
        self._series: Dict[str, SeriesConfig] = {}

        # Hover state
        self._hover_x = -1

        self._setup_ui()

    def _setup_ui(self):
        """Build the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar with series toggles
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        self._title_label = QLabel("Process Timeline")
        self._title_label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
        """)
        toolbar.addWidget(self._title_label)
        toolbar.addStretch()

        # Series checkboxes
        self._d50_check = QCheckBox("d50")
        self._d50_check.setChecked(True)
        self._d50_check.setStyleSheet(f"color: {COLORS.KPI_SIZE}; font-size: 9pt;")
        self._d50_check.toggled.connect(lambda v: self._set_series_visible("d50", v))
        toolbar.addWidget(self._d50_check)

        self._power_check = QCheckBox("Power")
        self._power_check.setChecked(True)
        self._power_check.setStyleSheet(f"color: {COLORS.KPI_POWER}; font-size: 9pt;")
        self._power_check.toggled.connect(lambda v: self._set_series_visible("power", v))
        toolbar.addWidget(self._power_check)

        self._throughput_check = QCheckBox("Throughput")
        self._throughput_check.setChecked(True)
        self._throughput_check.setStyleSheet(f"color: {COLORS.KPI_THROUGHPUT}; font-size: 9pt;")
        self._throughput_check.toggled.connect(lambda v: self._set_series_visible("throughput", v))
        toolbar.addWidget(self._throughput_check)

        layout.addLayout(toolbar)

        # Chart canvas
        self._canvas = _TimeSeriesCanvas(self)
        self._canvas.setMinimumHeight(200)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._canvas, 1)

        # Legend
        legend = QHBoxLayout()
        legend.setSpacing(16)
        legend.addStretch()

        for name, color, unit in [
            ("d50", COLORS.KPI_SIZE, "µm"),
            ("Power", COLORS.KPI_POWER, "kW"),
            ("Throughput", COLORS.KPI_THROUGHPUT, "kg/h"),
        ]:
            indicator = QLabel("●")
            indicator.setStyleSheet(f"color: {color}; font-size: 12pt;")
            legend.addWidget(indicator)

            label = QLabel(f"{name} ({unit})")
            label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt;")
            legend.addWidget(label)

        legend.addStretch()
        layout.addLayout(legend)

    def set_data(self, history: List) -> None:
        """Populate from simulation history.

        Args:
            history: List of MillingStepState objects
        """
        if not history:
            self.clear()
            return

        # Sample history for performance (max 500 points)
        step = max(1, len(history) // 500)
        sampled = history[::step]

        # Extract time series
        self._time_values = [s.time_s for s in sampled]

        # d50 in micrometers
        d50_values = [s.d50_m * 1e6 for s in sampled]
        self._series["d50"] = SeriesConfig(
            name="d50",
            values=d50_values,
            color=COLORS.KPI_SIZE,
            unit="µm",
            visible=True,
            y_axis="left",
        )

        # Power in kW
        power_values = [s.power_kw for s in sampled]
        self._series["power"] = SeriesConfig(
            name="Power",
            values=power_values,
            color=COLORS.KPI_POWER,
            unit="kW",
            visible=True,
            y_axis="right",
        )

        # Throughput in kg/h (convert from kg/s)
        throughput_values = [s.discharge_rate_kg_per_s * 3600 for s in sampled]
        self._series["throughput"] = SeriesConfig(
            name="Throughput",
            values=throughput_values,
            color=COLORS.KPI_THROUGHPUT,
            unit="kg/h",
            visible=True,
            y_axis="right",
        )

        self._update_canvas()

    def _set_series_visible(self, name: str, visible: bool):
        """Toggle series visibility."""
        if name in self._series:
            self._series[name].visible = visible
            self._update_canvas()

    def _update_canvas(self):
        """Update canvas with current data."""
        self._canvas.set_data(self._time_values, self._series)

    def clear(self):
        """Clear all data."""
        self._time_values = []
        self._series = {}
        self._canvas.clear()


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

        # Left Y axis labels (d50)
        if left_series:
            painter.setPen(QColor(COLORS.KPI_SIZE))
            for i in range(5):
                val = left_min + (left_max - left_min) * i / 4
                y = chart_y + chart_height * (1 - i / 4)
                painter.drawText(int(chart_x - 50), int(y + 4), f"{val:.0f}")

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
