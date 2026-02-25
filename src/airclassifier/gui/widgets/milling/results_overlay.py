"""
Results Overlay Panel
=====================

Comprehensive slide-in results panel with tabbed views for
process analytics, particle size distribution, and time series.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np

from PySide6.QtCore import (
    Qt, Signal, Slot, QPropertyAnimation, QEasingCurve,
    Property, QParallelAnimationGroup,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QWidget, QScrollArea,
    QSizePolicy, QGraphicsDropShadowEffect, QGridLayout,
    QProgressBar, QFormLayout,
)

from ...theme import COLORS, ANIMATIONS
from ..common import AnimatedKPICard, GlassCard
from .psd_chart import InteractivePSDChart
from .time_series_chart import TimeSeriesChart


class ResultsSummaryTab(QWidget):
    """Enhanced summary tab with KPI cards, PSD visual, and process summary."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # KPI grid (2x2)
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
            unit="µm",
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

        # PSD Visual Card with bars
        psd_frame = GlassCard()
        psd_layout = QVBoxLayout(psd_frame)
        psd_layout.setContentsMargins(12, 10, 12, 10)
        psd_layout.setSpacing(8)

        psd_title = QLabel("Particle Size Distribution")
        psd_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
            background: transparent;
        """)
        psd_layout.addWidget(psd_title)

        # D-value bars
        self._d10_bar = self._create_d_value_row("d10", COLORS.KPI_SIZE)
        psd_layout.addLayout(self._d10_bar["layout"])

        self._d50_bar = self._create_d_value_row("d50", COLORS.MILLING_PRIMARY)
        psd_layout.addLayout(self._d50_bar["layout"])

        self._d90_bar = self._create_d_value_row("d90", COLORS.WARNING)
        psd_layout.addLayout(self._d90_bar["layout"])

        # Span
        span_row = QHBoxLayout()
        span_label = QLabel("Span:")
        span_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        span_row.addWidget(span_label)
        self._span_value = QLabel("--")
        self._span_value.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt; font-weight: 600;")
        span_row.addWidget(self._span_value)
        span_row.addStretch()
        psd_layout.addLayout(span_row)

        layout.addWidget(psd_frame)

        # Process Summary Card
        process_frame = GlassCard()
        process_layout = QVBoxLayout(process_frame)
        process_layout.setContentsMargins(12, 10, 12, 10)
        process_layout.setSpacing(6)

        process_title = QLabel("Process Summary")
        process_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_PRIMARY};
            background: transparent;
        """)
        process_layout.addWidget(process_title)

        # Stats grid
        stats_form = QFormLayout()
        stats_form.setSpacing(4)
        stats_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._duration_label = QLabel("--")
        self._duration_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Duration:"), self._duration_label)

        self._total_mass_label = QLabel("--")
        self._total_mass_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Total Mass:"), self._total_mass_label)

        self._residence_label = QLabel("--")
        self._residence_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Residence Time:"), self._residence_label)

        self._efficiency_label = QLabel("--")
        self._efficiency_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Screen Efficiency:"), self._efficiency_label)

        self._particles_passed_label = QLabel("--")
        self._particles_passed_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Particles Passed:"), self._particles_passed_label)

        self._particles_retained_label = QLabel("--")
        self._particles_retained_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Particles in Chamber:"), self._particles_retained_label)

        self._holdup_label = QLabel("--")
        self._holdup_label.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt;")
        stats_form.addRow(self._styled_label("Holdup (kg):"), self._holdup_label)

        process_layout.addLayout(stats_form)
        layout.addWidget(process_frame)

        layout.addStretch()

    def _styled_label(self, text: str) -> QLabel:
        """Create a styled form label."""
        label = QLabel(text)
        label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        return label

    def _create_d_value_row(self, name: str, color: str) -> Dict:
        """Create a row with label, bar, and value for d-values."""
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel(f"{name}:")
        label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt; min-width: 30px;")
        row.addWidget(label)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS.BG_DARKEST};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 4px;
            }}
        """)
        row.addWidget(bar, 1)

        value = QLabel("-- µm")
        value.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt; min-width: 60px;")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(value)

        return {"layout": row, "bar": bar, "value": value}

    def update_results(self, result: Dict[str, Any]):
        """Update display with result data."""
        outlet = result.get("outlet")
        result_obj = result.get("result")

        if outlet:
            self._throughput_card.set_value(outlet.throughput_kg_per_hr, animate=False)
            self._d50_card.set_value(outlet.d50_um, animate=False)
            self._power_card.set_value(outlet.power_kw, animate=False)
            self._specific_energy_card.set_value(outlet.specific_energy_kwh_per_t, animate=False)

            # Update PSD bars (scale to max of 500 µm for visualization)
            max_size = max(500, outlet.d90_um * 1.2)

            self._d10_bar["value"].setText(f"{outlet.d10_um:.0f} µm")
            self._d10_bar["bar"].setValue(int(outlet.d10_um / max_size * 100))

            self._d50_bar["value"].setText(f"{outlet.d50_um:.0f} µm")
            self._d50_bar["bar"].setValue(int(outlet.d50_um / max_size * 100))

            self._d90_bar["value"].setText(f"{outlet.d90_um:.0f} µm")
            self._d90_bar["bar"].setValue(int(outlet.d90_um / max_size * 100))

            # Span
            if outlet.d50_um > 0:
                span = (outlet.d90_um - outlet.d10_um) / outlet.d50_um
                self._span_value.setText(f"{span:.2f}")

            # Residence time
            self._residence_label.setText(f"{outlet.mean_residence_time_s:.1f} s")

        if result_obj:
            # Duration from history
            if hasattr(result_obj, "history") and result_obj.history:
                duration = result_obj.history[-1].time_s if result_obj.history else 0
                self._duration_label.setText(f"{duration:.1f} s")

                # Total mass
                total_mass = sum(s.discharge_rate_kg_per_s for s in result_obj.history) * 0.001  # dt approximation
                if hasattr(result_obj, "psd_total_mass_kg"):
                    total_mass = result_obj.psd_total_mass_kg
                self._total_mass_label.setText(f"{total_mass:.2f} kg")

                # Screen efficiency (% passed)
                total_passed = sum(s.num_passed_screen for s in result_obj.history)
                total_particles = sum(s.num_fed for s in result_obj.history)
                if total_particles > 0:
                    efficiency = total_passed / total_particles * 100
                    self._efficiency_label.setText(f"{efficiency:.1f}%")
                # Particle tracking
                self._particles_passed_label.setText(f"{getattr(result_obj, 'total_particles_passed', total_passed):,}")
                last_state = result_obj.history[-1] if result_obj.history else None
                if last_state is not None:
                    self._particles_retained_label.setText(f"{last_state.num_particles:,}")
                    self._holdup_label.setText(f"{last_state.holdup_kg:.2f} kg")
                elif hasattr(result_obj, "total_particles_retained"):
                    self._particles_retained_label.setText(f"{result_obj.total_particles_retained:,}")
                    self._holdup_label.setText(f"{getattr(result_obj, 'holdup_kg_final', 0):.2f} kg")


class ResultsPSDTab(QWidget):
    """PSD histogram tab with InteractivePSDChart and statistics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Interactive PSD chart
        self._chart = InteractivePSDChart()
        self._chart.setMinimumHeight(200)
        layout.addWidget(self._chart, 1)

        # Statistics panel
        stats_frame = GlassCard()
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(12, 8, 12, 8)
        stats_layout.setSpacing(16)

        # Mode
        mode_col = QVBoxLayout()
        mode_label = QLabel("Mode")
        mode_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt;")
        mode_col.addWidget(mode_label)
        self._mode_value = QLabel("--")
        self._mode_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11pt; font-weight: 600;")
        mode_col.addWidget(self._mode_value)
        stats_layout.addLayout(mode_col)

        # Mean
        mean_col = QVBoxLayout()
        mean_label = QLabel("Mean")
        mean_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt;")
        mean_col.addWidget(mean_label)
        self._mean_value = QLabel("--")
        self._mean_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11pt; font-weight: 600;")
        mean_col.addWidget(self._mean_value)
        stats_layout.addLayout(mean_col)

        # Std Dev
        std_col = QVBoxLayout()
        std_label = QLabel("Std Dev")
        std_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt;")
        std_col.addWidget(std_label)
        self._std_value = QLabel("--")
        self._std_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11pt; font-weight: 600;")
        std_col.addWidget(self._std_value)
        stats_layout.addLayout(std_col)

        # CV
        cv_col = QVBoxLayout()
        cv_label = QLabel("CV")
        cv_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt;")
        cv_col.addWidget(cv_label)
        self._cv_value = QLabel("--")
        self._cv_value.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11pt; font-weight: 600;")
        cv_col.addWidget(self._cv_value)
        stats_layout.addLayout(cv_col)

        stats_layout.addStretch()
        layout.addWidget(stats_frame)

    def update_psd(self, size_classes, mass_fractions):
        """Update PSD data and calculate statistics."""
        size_classes = np.asarray(size_classes)
        mass_fractions = np.asarray(mass_fractions)

        self._chart.set_data(size_classes, mass_fractions)

        if len(mass_fractions) > 0 and len(size_classes) > 1:
            # Calculate midpoints
            midpoints = (size_classes[:-1] + size_classes[1:]) / 2
            if len(midpoints) == len(mass_fractions):
                # Mode (size with highest fraction)
                mode_idx = np.argmax(mass_fractions)
                mode = midpoints[mode_idx]
                self._mode_value.setText(f"{mode:.0f} µm")

                # Mean
                total = np.sum(mass_fractions)
                if total > 0:
                    mean = np.sum(midpoints * mass_fractions) / total
                    self._mean_value.setText(f"{mean:.0f} µm")

                    # Variance and Std Dev
                    variance = np.sum(mass_fractions * (midpoints - mean) ** 2) / total
                    std = np.sqrt(variance)
                    self._std_value.setText(f"{std:.0f} µm")

                    # CV (coefficient of variation)
                    if mean > 0:
                        cv = std / mean * 100
                        self._cv_value.setText(f"{cv:.1f}%")


class ResultsTimeSeriesTab(QWidget):
    """Time series chart tab with d50, power, throughput over time."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Time series chart in scroll area so graphs are scrollable
        self._chart = TimeSeriesChart()
        self._chart.setMinimumHeight(520)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._chart)
        scroll.setStyleSheet(f"""
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
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        layout.addWidget(scroll, 1)

    def update_from_history(self, history: List):
        """Update chart from simulation history."""
        self._chart.set_data(history)


class ResultsAnalyticsTab(QWidget):
    """Particle analytics tab with breakage, screen, and energy statistics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Breakage Statistics Card
        breakage_frame = GlassCard()
        breakage_layout = QVBoxLayout(breakage_frame)
        breakage_layout.setContentsMargins(12, 10, 12, 10)
        breakage_layout.setSpacing(6)

        breakage_title = QLabel("Breakage Statistics")
        breakage_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.MILLING_PRIMARY};
            background: transparent;
        """)
        breakage_layout.addWidget(breakage_title)

        self._impacts_label = self._create_stat_row("Total Impacts:")
        breakage_layout.addLayout(self._impacts_label["layout"])

        self._breakage_events_label = self._create_stat_row("Breakage Events:")
        breakage_layout.addLayout(self._breakage_events_label["layout"])

        self._breakage_rate_label = self._create_stat_row("Breakage Rate:")
        breakage_layout.addLayout(self._breakage_rate_label["layout"])

        self._size_reduction_label = self._create_stat_row("Avg Size Reduction:")
        breakage_layout.addLayout(self._size_reduction_label["layout"])

        layout.addWidget(breakage_frame)

        # Screen Performance Card
        screen_frame = GlassCard()
        screen_layout = QVBoxLayout(screen_frame)
        screen_layout.setContentsMargins(12, 10, 12, 10)
        screen_layout.setSpacing(6)

        screen_title = QLabel("Screen Performance")
        screen_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.KPI_THROUGHPUT};
            background: transparent;
        """)
        screen_layout.addWidget(screen_title)

        self._particles_passed_label = self._create_stat_row("Particles Passed:")
        screen_layout.addLayout(self._particles_passed_label["layout"])

        self._particles_retained_label = self._create_stat_row("Particles Retained:")
        screen_layout.addLayout(self._particles_retained_label["layout"])

        self._holdup_label = self._create_stat_row("Holdup (kg):")
        screen_layout.addLayout(self._holdup_label["layout"])

        self._passage_rate_label = self._create_stat_row("Passage Rate:")
        screen_layout.addLayout(self._passage_rate_label["layout"])

        self._screen_aperture_label = self._create_stat_row("Screen Aperture:")
        screen_layout.addLayout(self._screen_aperture_label["layout"])

        layout.addWidget(screen_frame)

        # Energy Efficiency Card
        energy_frame = GlassCard()
        energy_layout = QVBoxLayout(energy_frame)
        energy_layout.setContentsMargins(12, 10, 12, 10)
        energy_layout.setSpacing(6)

        energy_title = QLabel("Energy Efficiency")
        energy_title.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.KPI_POWER};
            background: transparent;
        """)
        energy_layout.addWidget(energy_title)

        self._total_energy_label = self._create_stat_row("Total Energy:")
        energy_layout.addLayout(self._total_energy_label["layout"])

        self._specific_energy_label = self._create_stat_row("Specific Energy:")
        energy_layout.addLayout(self._specific_energy_label["layout"])

        self._peak_power_label = self._create_stat_row("Peak Power:")
        energy_layout.addLayout(self._peak_power_label["layout"])

        self._mean_power_label = self._create_stat_row("Mean Power:")
        energy_layout.addLayout(self._mean_power_label["layout"])

        self._load_factor_label = self._create_stat_row("Load Factor:")
        energy_layout.addLayout(self._load_factor_label["layout"])

        layout.addWidget(energy_frame)

        layout.addStretch()

    def _create_stat_row(self, label_text: str) -> Dict:
        """Create a stat row with label and value."""
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        row.addWidget(label)

        value = QLabel("--")
        value.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt; font-weight: 600;")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(value)

        return {"layout": row, "value": value}

    def update_analytics(self, result: Dict[str, Any]):
        """Update analytics from result data."""
        result_obj = result.get("result")
        outlet = result.get("outlet")

        if result_obj and hasattr(result_obj, "history") and result_obj.history:
            history = result_obj.history

            # Breakage stats
            total_impacts = sum(s.num_impacts for s in history)
            total_breakage = sum(s.num_breakage_events for s in history)
            breakage_rate = (total_breakage / total_impacts * 100) if total_impacts > 0 else 0

            self._impacts_label["value"].setText(f"{total_impacts:,}")
            self._breakage_events_label["value"].setText(f"{total_breakage:,}")
            self._breakage_rate_label["value"].setText(f"{breakage_rate:.1f}%")

            # Average size reduction
            reductions = [s.mean_size_reduction for s in history if s.mean_size_reduction < 1.0]
            if reductions:
                avg_reduction = np.mean(reductions)
                self._size_reduction_label["value"].setText(f"{avg_reduction:.2f}x")

            # Screen stats
            total_passed = sum(s.num_passed_screen for s in history)
            total_fed = sum(s.num_fed for s in history)
            passage_rate = (total_passed / total_fed * 100) if total_fed > 0 else 0

            last_state = history[-1]
            self._particles_passed_label["value"].setText(f"{total_passed:,}")
            self._particles_retained_label["value"].setText(f"{last_state.num_particles:,}")
            self._holdup_label["value"].setText(f"{last_state.holdup_kg:.2f}")
            self._passage_rate_label["value"].setText(f"{passage_rate:.1f}%")

            # Energy stats
            duration = history[-1].time_s if history else 0
            power_values = [s.power_kw for s in history]
            if power_values:
                mean_power = np.mean(power_values)
                peak_power = np.max(power_values)
                total_energy = mean_power * duration / 3600  # kWh
                load_factor = (mean_power / peak_power * 100) if peak_power > 0 else 0

                self._total_energy_label["value"].setText(f"{total_energy:.3f} kWh")
                self._peak_power_label["value"].setText(f"{peak_power:.1f} kW")
                self._mean_power_label["value"].setText(f"{mean_power:.1f} kW")
                self._load_factor_label["value"].setText(f"{load_factor:.1f}%")

        if outlet:
            self._specific_energy_label["value"].setText(f"{outlet.specific_energy_kwh_per_t:.1f} kWh/t")

        # Screen aperture from result config if available
        if result_obj and hasattr(result_obj, "config"):
            aperture = getattr(result_obj.config, "screen_aperture_mm", None)
            if aperture:
                self._screen_aperture_label["value"].setText(f"{aperture:.2f} mm")


class ResultsOverlay(QFrame):
    """Comprehensive slide-in results panel overlaying the viewport.

    Features:
        - Summary: KPI cards + PSD visual + process summary
        - PSD: Interactive histogram with statistics
        - Process: Time series charts (d50, power, throughput)
        - Analytics: Breakage, screen, energy metrics

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
        self._target_width = 420  # Widened for better readability

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

        title = QLabel("Milling Results")
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
                padding: 6px 10px;
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

        # Create tabs
        self._summary_tab = ResultsSummaryTab()
        self._psd_tab = ResultsPSDTab()
        self._timeseries_tab = ResultsTimeSeriesTab()
        self._analytics_tab = ResultsAnalyticsTab()

        # Wrap in scroll areas
        self._tabs.addTab(self._wrap_scroll(self._summary_tab), "Summary")
        self._tabs.addTab(self._wrap_scroll(self._psd_tab), "PSD")
        self._tabs.addTab(self._wrap_scroll(self._timeseries_tab), "Process")
        self._tabs.addTab(self._wrap_scroll(self._analytics_tab), "Analytics")

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

    def _wrap_scroll(self, widget: QWidget) -> QScrollArea:
        """Wrap widget in scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
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
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        scroll.setWidget(widget)
        return scroll

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
        # Summary tab
        self._summary_tab.update_results(results)

        # PSD tab
        result = results.get("result")
        if result and hasattr(result, "psd_size_classes_m"):
            self._psd_tab.update_psd(
                result.psd_size_classes_m * 1e6,
                result.psd_mass_fractions
            )

        # Time series tab
        if result and hasattr(result, "history"):
            self._timeseries_tab.update_from_history(result.history)

        # Analytics tab
        self._analytics_tab.update_analytics(results)

    @property
    def is_visible(self) -> bool:
        return self._is_visible
