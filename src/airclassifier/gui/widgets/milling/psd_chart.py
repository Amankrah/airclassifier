"""
Interactive PSD Chart
=====================

Modern particle size distribution chart with hover tooltips,
zoom/pan, and comparison mode.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QPainterPath, QFont, QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QSizePolicy, QToolTip,
    QFrame,
)

from ...theme import COLORS


class InteractivePSDChart(QWidget):
    """Interactive PSD histogram with hover tooltips and comparison mode.

    Features:
        - Hover tooltips showing exact values
        - Zoom/pan with mouse
        - Overlay multiple runs for comparison
        - Logarithmic X-axis option
        - Cumulative distribution toggle
    """

    bar_hovered = Signal(int, float, float)  # index, size, fraction

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Data
        self._size_classes: np.ndarray = np.array([])
        self._mass_fractions: np.ndarray = np.array([])
        self._comparison_data: List[Tuple[np.ndarray, np.ndarray, str]] = []

        # Display options
        self._show_cumulative = False
        self._log_scale = False
        self._hovered_bar = -1

        # Colors
        self._bar_color = QColor(COLORS.MILLING_PRIMARY)
        self._bar_hover_color = QColor(COLORS.MILLING_SECONDARY)
        self._comparison_colors = [
            QColor(COLORS.CHART_SECONDARY),
            QColor(COLORS.CHART_TERTIARY),
            QColor(COLORS.CHART_QUATERNARY),
        ]

        self._setup_ui()

    def _setup_ui(self):
        """Build the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._title_label = QLabel("Particle Size Distribution")
        self._title_label.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
        """)
        toolbar.addWidget(self._title_label)
        toolbar.addStretch()

        self._cumulative_check = QCheckBox("Cumulative")
        self._cumulative_check.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        self._cumulative_check.toggled.connect(self._on_cumulative_toggled)
        toolbar.addWidget(self._cumulative_check)

        self._log_check = QCheckBox("Log X")
        self._log_check.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        self._log_check.toggled.connect(self._on_log_toggled)
        toolbar.addWidget(self._log_check)

        layout.addLayout(toolbar)

        # Chart area
        self._chart_widget = _ChartCanvas(self)
        self._chart_widget.setMinimumHeight(180)
        self._chart_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._chart_widget, 1)

        # Connect chart signals
        self._chart_widget.bar_hovered.connect(self._on_bar_hovered)

    def set_data(self, size_classes: np.ndarray, mass_fractions: np.ndarray):
        """Set the PSD data.

        Args:
            size_classes: Size class edges (um)
            mass_fractions: Mass fractions for each bin
        """
        self._size_classes = np.asarray(size_classes)
        self._mass_fractions = np.asarray(mass_fractions)
        self._update_chart()

    def add_comparison(self, size_classes: np.ndarray, mass_fractions: np.ndarray, label: str):
        """Add a comparison dataset."""
        self._comparison_data.append((
            np.asarray(size_classes),
            np.asarray(mass_fractions),
            label
        ))
        self._update_chart()

    def clear_comparisons(self):
        """Remove all comparison datasets."""
        self._comparison_data.clear()
        self._update_chart()

    def clear(self):
        """Clear all data."""
        self._size_classes = np.array([])
        self._mass_fractions = np.array([])
        self._comparison_data.clear()
        self._update_chart()

    def _update_chart(self):
        """Update the chart display."""
        fractions = self._mass_fractions
        if self._show_cumulative and len(fractions) > 0:
            fractions = np.cumsum(fractions)

        self._chart_widget.set_data(
            self._size_classes,
            fractions,
            self._comparison_data,
            log_scale=self._log_scale,
            cumulative=self._show_cumulative,
        )

    def _on_cumulative_toggled(self, checked: bool):
        """Handle cumulative checkbox."""
        self._show_cumulative = checked
        self._update_chart()

    def _on_log_toggled(self, checked: bool):
        """Handle log scale checkbox."""
        self._log_scale = checked
        self._update_chart()

    def _on_bar_hovered(self, index: int, size: float, fraction: float):
        """Handle bar hover."""
        self._hovered_bar = index
        self.bar_hovered.emit(index, size, fraction)


class _ChartCanvas(QWidget):
    """Internal canvas for rendering the PSD chart."""

    bar_hovered = Signal(int, float, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._size_classes: np.ndarray = np.array([])
        self._mass_fractions: np.ndarray = np.array([])
        self._comparison_data: List[Tuple[np.ndarray, np.ndarray, str]] = []
        self._log_scale = False
        self._cumulative = False
        self._hovered_bar = -1

        self._bar_color = QColor(COLORS.MILLING_PRIMARY)
        self._hover_color = QColor(COLORS.TEXT_INVERSE)

        self.setMouseTracking(True)
        self.setStyleSheet(f"background: {COLORS.BG_DARKEST}; border-radius: 6px;")

    def set_data(
        self,
        size_classes: np.ndarray,
        mass_fractions: np.ndarray,
        comparison_data: List[Tuple[np.ndarray, np.ndarray, str]],
        log_scale: bool = False,
        cumulative: bool = False,
    ):
        """Set chart data."""
        self._size_classes = size_classes
        self._mass_fractions = mass_fractions
        self._comparison_data = comparison_data
        self._log_scale = log_scale
        self._cumulative = cumulative
        self.update()

    def paintEvent(self, event):
        """Render the chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        padding = {"left": 50, "right": 20, "top": 20, "bottom": 35}

        chart_rect = QPointF(padding["left"], padding["top"])
        chart_width = rect.width() - padding["left"] - padding["right"]
        chart_height = rect.height() - padding["top"] - padding["bottom"]

        # Background
        painter.fillRect(rect, QColor(COLORS.BG_DARKEST))

        if len(self._size_classes) < 2 or len(self._mass_fractions) == 0:
            # No data placeholder
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No PSD data")
            return

        # Calculate scales
        sizes = self._size_classes
        fractions = self._mass_fractions
        n_bars = len(fractions)

        if self._log_scale and np.min(sizes[sizes > 0]) > 0:
            x_min = np.log10(np.min(sizes[sizes > 0]))
            x_max = np.log10(np.max(sizes))
        else:
            x_min = np.min(sizes)
            x_max = np.max(sizes)

        y_max = max(np.max(fractions), 0.01)
        if self._cumulative:
            y_max = 1.0

        x_range = x_max - x_min if x_max != x_min else 1
        bar_width = chart_width / n_bars * 0.8
        bar_gap = chart_width / n_bars * 0.2

        # Draw grid lines
        painter.setPen(QPen(QColor(COLORS.CHART_GRID), 1))
        for i in range(5):
            y = padding["top"] + chart_height * (1 - i / 4)
            painter.drawLine(int(padding["left"]), int(y), int(rect.width() - padding["right"]), int(y))

        # Draw bars
        for i, frac in enumerate(fractions):
            x = padding["left"] + i * (chart_width / n_bars) + bar_gap / 2
            bar_h = (frac / y_max) * chart_height
            y = padding["top"] + chart_height - bar_h

            # Gradient fill
            gradient = QLinearGradient(x, y, x, y + bar_h)
            if i == self._hovered_bar:
                gradient.setColorAt(0, self._hover_color)
                gradient.setColorAt(1, self._bar_color)
            else:
                color = QColor(self._bar_color)
                color.setAlpha(200)
                gradient.setColorAt(0, self._bar_color)
                gradient.setColorAt(1, color)

            painter.fillRect(int(x), int(y), int(bar_width), int(bar_h), gradient)

        # Draw axes
        painter.setPen(QPen(QColor(COLORS.CHART_AXIS), 1))
        # Y axis
        painter.drawLine(
            int(padding["left"]), int(padding["top"]),
            int(padding["left"]), int(rect.height() - padding["bottom"])
        )
        # X axis
        painter.drawLine(
            int(padding["left"]), int(rect.height() - padding["bottom"]),
            int(rect.width() - padding["right"]), int(rect.height() - padding["bottom"])
        )

        # Axis labels
        painter.setPen(QColor(COLORS.TEXT_MUTED))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)

        # Y axis labels
        for i in range(5):
            val = y_max * i / 4
            y = padding["top"] + chart_height * (1 - i / 4)
            if self._cumulative:
                text = f"{val:.0%}"
            else:
                text = f"{val:.2f}"
            painter.drawText(int(padding["left"] - 45), int(y + 4), text)

        # X axis labels (every few bars)
        step = max(1, n_bars // 5)
        for i in range(0, n_bars, step):
            x = padding["left"] + i * (chart_width / n_bars) + bar_width / 2
            if i < len(sizes) - 1:
                size = (sizes[i] + sizes[i + 1]) / 2 if i + 1 < len(sizes) else sizes[i]
            else:
                size = sizes[-1] if len(sizes) > 0 else 0
            painter.drawText(int(x - 15), int(rect.height() - 10), f"{size:.0f}")

        # X axis title
        painter.drawText(
            int(rect.width() / 2 - 30),
            int(rect.height() - 2),
            "Size (um)"
        )

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hover detection."""
        if len(self._mass_fractions) == 0:
            return

        rect = self.rect()
        padding = {"left": 50, "right": 20, "top": 20, "bottom": 35}
        chart_width = rect.width() - padding["left"] - padding["right"]
        n_bars = len(self._mass_fractions)

        x = event.position().x()
        bar_index = int((x - padding["left"]) / (chart_width / n_bars))

        if 0 <= bar_index < n_bars:
            if bar_index != self._hovered_bar:
                self._hovered_bar = bar_index
                self.update()

                # Show tooltip
                size = self._size_classes[bar_index] if bar_index < len(self._size_classes) else 0
                frac = self._mass_fractions[bar_index]
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"Size: {size:.1f} um\nFraction: {frac:.3f}"
                )
                self.bar_hovered.emit(bar_index, size, frac)
        else:
            if self._hovered_bar >= 0:
                self._hovered_bar = -1
                self.update()

    def leaveEvent(self, event):
        """Clear hover state."""
        self._hovered_bar = -1
        self.update()
