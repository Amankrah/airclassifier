"""
Results Overlay Panel
=====================

Slide-in results panel that overlays the 3D viewport while keeping
the simulation context visible. Features tabbed results views.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

from PySide6.QtCore import (
    Qt, Signal, Slot, QPropertyAnimation, QEasingCurve,
    Property, QParallelAnimationGroup,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QWidget, QScrollArea,
    QSizePolicy, QGraphicsDropShadowEffect, QGridLayout,
)

from ...theme import COLORS, ANIMATIONS
from ..common import AnimatedKPICard, GlassCard


class ResultsSummaryTab(QWidget):
    """Summary tab with KPI cards."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # KPI grid
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(8)

        self._throughput_card = AnimatedKPICard(
            title="Throughput",
            unit="kg/h",
            semantic_color=COLORS.KPI_THROUGHPUT,
            show_sparkline=False,
        )
        kpi_grid.addWidget(self._throughput_card, 0, 0)

        self._d50_card = AnimatedKPICard(
            title="d50",
            unit="um",
            semantic_color=COLORS.KPI_SIZE,
            show_sparkline=False,
        )
        kpi_grid.addWidget(self._d50_card, 0, 1)

        self._power_card = AnimatedKPICard(
            title="Power",
            unit="kW",
            semantic_color=COLORS.KPI_POWER,
            precision=2,
            show_sparkline=False,
        )
        kpi_grid.addWidget(self._power_card, 1, 0)

        self._specific_energy_card = AnimatedKPICard(
            title="Specific Energy",
            unit="kWh/t",
            semantic_color=COLORS.KPI_EFFICIENCY,
            precision=2,
            show_sparkline=False,
        )
        kpi_grid.addWidget(self._specific_energy_card, 1, 1)

        layout.addLayout(kpi_grid)

        # Additional stats
        stats_frame = GlassCard()
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(6)

        stats_title = QLabel("Particle Size Distribution")
        stats_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
            background: transparent;
        """)
        stats_layout.addWidget(stats_title)

        self._d10_label = QLabel("d10: --")
        self._d10_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_layout.addWidget(self._d10_label)

        self._d50_label = QLabel("d50: --")
        self._d50_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_layout.addWidget(self._d50_label)

        self._d90_label = QLabel("d90: --")
        self._d90_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_layout.addWidget(self._d90_label)

        self._span_label = QLabel("Span: --")
        self._span_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_layout.addWidget(self._span_label)

        layout.addWidget(stats_frame)
        layout.addStretch()

    def update_results(self, result: Dict[str, Any]):
        """Update display with result data."""
        outlet = result.get("outlet")
        result_obj = result.get("result")

        if outlet:
            self._throughput_card.set_value(outlet.throughput_kg_per_hr, animate=False)
            self._d50_card.set_value(outlet.d50_um, animate=False)
            self._power_card.set_value(outlet.power_kw, animate=False)
            self._specific_energy_card.set_value(outlet.specific_energy_kwh_per_t, animate=False)

            self._d10_label.setText(f"d10: {outlet.d10_um:.1f} um")
            self._d50_label.setText(f"d50: {outlet.d50_um:.1f} um")
            self._d90_label.setText(f"d90: {outlet.d90_um:.1f} um")

            # Calculate span
            if outlet.d50_um > 0:
                span = (outlet.d90_um - outlet.d10_um) / outlet.d50_um
                self._span_label.setText(f"Span: {span:.2f}")


class ResultsPSDTab(QWidget):
    """PSD histogram tab."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Placeholder for PSD chart
        self._chart_placeholder = QLabel("PSD Chart")
        self._chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chart_placeholder.setStyleSheet(f"""
            background: {COLORS.BG_DARKEST};
            border: 1px solid {COLORS.BORDER_SUBTLE};
            border-radius: 6px;
            color: {COLORS.TEXT_MUTED};
            min-height: 200px;
        """)
        layout.addWidget(self._chart_placeholder)

        # Will be replaced by InteractivePSDChart
        self._chart = None

    def set_chart(self, chart_widget):
        """Set the actual chart widget."""
        if self._chart_placeholder:
            layout = self.layout()
            layout.removeWidget(self._chart_placeholder)
            self._chart_placeholder.deleteLater()
            self._chart_placeholder = None

        self._chart = chart_widget
        self.layout().addWidget(chart_widget)

    def update_psd(self, size_classes, mass_fractions):
        """Update PSD data."""
        if self._chart:
            self._chart.set_data(size_classes, mass_fractions)


class ResultsTimeSeriesTab(QWidget):
    """Time series plot tab."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Placeholder for time series
        placeholder = QLabel("Time Series Plot\n(d50, power, throughput vs time)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"""
            background: {COLORS.BG_DARKEST};
            border: 1px solid {COLORS.BORDER_SUBTLE};
            border-radius: 6px;
            color: {COLORS.TEXT_MUTED};
            min-height: 200px;
        """)
        layout.addWidget(placeholder)
        layout.addStretch()


class ResultsOverlay(QFrame):
    """Slide-in results panel overlaying the viewport.

    Signals:
        closed(): Emitted when overlay is closed
        export_clicked(): Emitted when export button is clicked
    """

    closed = Signal()
    export_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_visible = False
        self._animation: Optional[QPropertyAnimation] = None
        self._target_width = 320

        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        """Apply glassmorphism styling."""
        self.setObjectName("resultsOverlay")
        self.setStyleSheet(f"""
            QFrame#resultsOverlay {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(15, 23, 42, 0.95),
                    stop:1 rgba(30, 41, 59, 0.98));
                border-left: 1px solid rgba(255, 255, 255, 0.1);
            }}
        """)

        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setXOffset(-5)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        self.setFixedWidth(self._target_width)

    def _setup_ui(self):
        """Build the overlay UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-bottom: 1px solid {COLORS.BORDER_SUBTLE};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Results")
        title.setStyleSheet(f"""
            font-size: 12pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()

        close_btn = QPushButton("X")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {COLORS.TEXT_MUTED};
                font-size: 12pt;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                color: {COLORS.TEXT_PRIMARY};
                background: {COLORS.BG_HOVER};
                border-radius: 4px;
            }}
        """)
        close_btn.clicked.connect(self.hide_panel)
        header_layout.addWidget(close_btn)

        layout.addWidget(header)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: transparent;
                border: none;
            }}
            QTabBar::tab {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-bottom: none;
                padding: 6px 12px;
                margin-right: 2px;
                color: {COLORS.TEXT_MUTED};
                font-size: 9pt;
            }}
            QTabBar::tab:selected {{
                background: {COLORS.BG_ELEVATED};
                color: {COLORS.TEXT_PRIMARY};
            }}
            QTabBar::tab:hover {{
                background: {COLORS.BG_HOVER};
            }}
        """)

        self._summary_tab = ResultsSummaryTab()
        self._psd_tab = ResultsPSDTab()
        self._timeseries_tab = ResultsTimeSeriesTab()

        self._tabs.addTab(self._summary_tab, "Summary")
        self._tabs.addTab(self._psd_tab, "PSD")
        self._tabs.addTab(self._timeseries_tab, "Time Series")

        layout.addWidget(self._tabs, 1)

        # Footer with export button
        footer = QFrame()
        footer.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-top: 1px solid {COLORS.BORDER_SUBTLE};
            }}
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 10)

        export_btn = QPushButton("Export Results...")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.ACCENT_MUTED};
                border: 1px solid {COLORS.ACCENT};
                border-radius: 6px;
                padding: 8px 16px;
                color: {COLORS.ACCENT};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.ACCENT};
                color: {COLORS.TEXT_INVERSE};
            }}
        """)
        export_btn.clicked.connect(self.export_clicked.emit)
        footer_layout.addWidget(export_btn)

        layout.addWidget(footer)

        # Initially hidden (off screen)
        self.hide()

    def show_panel(self):
        """Slide in the results panel."""
        if self._is_visible:
            return

        self._is_visible = True
        self.show()
        self.raise_()

        # Animate slide-in
        if self._animation:
            self._animation.stop()

        parent = self.parent()
        if parent:
            start_x = parent.width()
            end_x = parent.width() - self._target_width
            self.move(start_x, 0)
            self.setFixedHeight(parent.height())

            self._animation = QPropertyAnimation(self, b"pos")
            self._animation.setDuration(ANIMATIONS.NORMAL)
            self._animation.setStartValue(self.pos())
            self._animation.setEndValue(self.pos().__class__(end_x, 0))
            self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._animation.start()

    def hide_panel(self):
        """Slide out the results panel."""
        if not self._is_visible:
            return

        self._is_visible = False

        if self._animation:
            self._animation.stop()

        parent = self.parent()
        if parent:
            end_x = parent.width()

            self._animation = QPropertyAnimation(self, b"pos")
            self._animation.setDuration(ANIMATIONS.NORMAL)
            self._animation.setStartValue(self.pos())
            self._animation.setEndValue(self.pos().__class__(end_x, 0))
            self._animation.setEasingCurve(QEasingCurve.Type.InCubic)
            self._animation.finished.connect(self._on_hide_finished)
            self._animation.start()

    def _on_hide_finished(self):
        """Called when hide animation completes."""
        self.hide()
        self.closed.emit()

    def toggle(self):
        """Toggle panel visibility."""
        if self._is_visible:
            self.hide_panel()
        else:
            self.show_panel()

    def update_results(self, results: Dict[str, Any]):
        """Update all tabs with new results."""
        self._summary_tab.update_results(results)

        result = results.get("result")
        if result and hasattr(result, "psd_size_classes_m"):
            self._psd_tab.update_psd(
                result.psd_size_classes_m * 1e6,
                result.psd_mass_fractions
            )

    @property
    def is_visible(self) -> bool:
        return self._is_visible
