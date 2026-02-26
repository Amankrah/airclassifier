"""
Milling Results Page
====================

Full-page comprehensive results display for hammer mill simulation.
Provides detailed visualization of PSD, process timeline, and analytics.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTabWidget, QFrame,
    QScrollArea, QSizePolicy, QProgressBar,
    QFormLayout, QGraphicsDropShadowEffect, QSplitter,
    QRadioButton,
)

from ...theme import COLORS, ANIMATIONS
from ..common import AnimatedKPICard, GlassCard
from .psd_chart import InteractivePSDChart
from .time_series_chart import TimeSeriesChart


class _KPIRow(QFrame):
    """Row of KPI cards with consistent styling."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)

        # Create KPI cards
        self._throughput_card = self._create_kpi("Throughput", "kg/h", COLORS.KPI_THROUGHPUT)
        layout.addWidget(self._throughput_card["frame"])

        self._d50_card = self._create_kpi("Flour d50", "\u00b5m", COLORS.KPI_SIZE)
        layout.addWidget(self._d50_card["frame"])

        self._seed_d50_card = self._create_kpi("Seed d50", "\u00b5m", COLORS.WARNING)
        layout.addWidget(self._seed_d50_card["frame"])

        self._power_card = self._create_kpi("Power", "kW", COLORS.KPI_POWER)
        layout.addWidget(self._power_card["frame"])

        self._energy_card = self._create_kpi("Specific Energy", "kWh/t", COLORS.KPI_EFFICIENCY)
        layout.addWidget(self._energy_card["frame"])

        self._yield_card = self._create_kpi("Yield", "%", COLORS.KPI_THROUGHPUT)
        layout.addWidget(self._yield_card["frame"])

    def _create_kpi(self, title: str, unit: str, color: str) -> Dict:
        """Create a compact KPI display."""
        frame = QFrame()
        frame.setStyleSheet("background: transparent; border: none;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        layout.addWidget(title_label)

        # Value row
        value_row = QHBoxLayout()
        value_label = QLabel("--")
        value_label.setStyleSheet(f"color: {color}; font-size: 18pt; font-weight: 700;")
        value_row.addWidget(value_label)

        unit_label = QLabel(unit)
        unit_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        value_row.addWidget(unit_label)
        value_row.addStretch()

        layout.addLayout(value_row)

        return {"frame": frame, "value": value_label}

    def update_values(self, outlet, result_obj):
        """Update KPI values from result data."""
        if outlet:
            self._throughput_card["value"].setText(f"{outlet.throughput_kg_per_hr:.0f}")
            self._d50_card["value"].setText(f"{outlet.d50_um:.0f}")
            self._power_card["value"].setText(f"{outlet.power_kw:.1f}")
            self._energy_card["value"].setText(f"{outlet.specific_energy_kwh_per_t:.1f}")
        if result_obj:
            if hasattr(result_obj, "retained_d50_um") and result_obj.retained_d50_um > 0:
                self._seed_d50_card["value"].setText(f"{result_obj.retained_d50_um:.0f}")
            # Yield = discharge mass / total fed mass
            fed = getattr(result_obj, "total_mass_fed_kg", 0)
            discharged = getattr(result_obj, "psd_total_mass_kg", 0)
            if fed > 0:
                self._yield_card["value"].setText(f"{discharged / fed * 100:.1f}")


class _PSDPanel(QFrame):
    """Panel containing PSD chart and statistics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Header with discharge vs retained toggle
        header = QHBoxLayout()
        title = QLabel("Particle Size Distribution")
        title.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 12pt; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        self._psd_discharge_btn = QRadioButton("Flour (discharged)")
        self._psd_discharge_btn.setChecked(True)
        self._psd_discharge_btn.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        self._psd_retained_btn = QRadioButton("Seeds (retained)")
        self._psd_retained_btn.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        self._psd_discharge_btn.toggled.connect(self._on_psd_mode_toggled)
        self._psd_retained_btn.toggled.connect(self._on_psd_mode_toggled)
        header.addWidget(self._psd_discharge_btn)
        header.addWidget(self._psd_retained_btn)
        layout.addLayout(header)

        # Chart
        self._chart = InteractivePSDChart()
        self._chart.setMinimumHeight(280)
        layout.addWidget(self._chart, 1)
        self._psd_discharge_data = None
        self._psd_retained_data = None
        self._psd_outlet = None

        # Statistics row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(24)

        self._mode_stat = self._create_stat("Mode", COLORS.MILLING_PRIMARY)
        stats_row.addLayout(self._mode_stat["layout"])

        self._mean_stat = self._create_stat("Mean", COLORS.KPI_SIZE)
        stats_row.addLayout(self._mean_stat["layout"])

        self._std_stat = self._create_stat("Std Dev", COLORS.TEXT_SECONDARY)
        stats_row.addLayout(self._std_stat["layout"])

        self._cv_stat = self._create_stat("CV", COLORS.WARNING)
        stats_row.addLayout(self._cv_stat["layout"])

        self._span_stat = self._create_stat("Span", COLORS.KPI_EFFICIENCY)
        stats_row.addLayout(self._span_stat["layout"])

        stats_row.addStretch()
        layout.addLayout(stats_row)

    def _create_stat(self, label: str, color: str) -> Dict:
        """Create a statistic display."""
        layout = QVBoxLayout()
        layout.setSpacing(2)

        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        layout.addWidget(label_widget)

        value_widget = QLabel("--")
        value_widget.setStyleSheet(f"color: {color}; font-size: 14pt; font-weight: 600;")
        layout.addWidget(value_widget)

        return {"layout": layout, "value": value_widget}

    def _on_psd_mode_toggled(self):
        """Switch PSD chart between discharge and retained."""
        if self._psd_retained_btn.isChecked() and self._psd_retained_data is not None:
            sc, mf = self._psd_retained_data
            self._chart.set_data(np.asarray(sc), np.asarray(mf))
        elif self._psd_discharge_data is not None:
            sc, mf = self._psd_discharge_data
            self._chart.set_data(np.asarray(sc), np.asarray(mf))
        self._update_psd_stats()

    def _update_psd_stats(self):
        """Update PSD statistics for currently displayed data."""
        if self._psd_retained_btn.isChecked() and self._psd_retained_data is not None:
            size_classes, mass_fractions = self._psd_retained_data
        elif self._psd_discharge_data is not None:
            size_classes, mass_fractions = self._psd_discharge_data
        else:
            return
        size_classes = np.asarray(size_classes)
        mass_fractions = np.asarray(mass_fractions)
        if len(mass_fractions) > 0 and len(size_classes) > 1:
            midpoints = (size_classes[:-1] + size_classes[1:]) / 2
            if len(midpoints) == len(mass_fractions):
                mode_idx = np.argmax(mass_fractions)
                self._mode_stat["value"].setText(f"{midpoints[mode_idx]:.0f} µm")
                total = np.sum(mass_fractions)
                if total > 0:
                    mean = np.sum(midpoints * mass_fractions) / total
                    self._mean_stat["value"].setText(f"{mean:.0f} µm")
                    variance = np.sum(mass_fractions * (midpoints - mean) ** 2) / total
                    std = np.sqrt(variance)
                    self._std_stat["value"].setText(f"{std:.0f} µm")
                    if mean > 0:
                        self._cv_stat["value"].setText(f"{std / mean * 100:.1f}%")
        if self._psd_outlet and self._psd_outlet.d50_um > 0 and not self._psd_retained_btn.isChecked():
            span = (self._psd_outlet.d90_um - self._psd_outlet.d10_um) / self._psd_outlet.d50_um
            self._span_stat["value"].setText(f"{span:.2f}")
        elif self._psd_retained_btn.isChecked():
            self._span_stat["value"].setText("--")

    def update_psd(self, size_classes, mass_fractions, outlet,
                   retained_size_classes=None, retained_mass_fractions=None):
        """Update PSD chart and statistics (discharge and optionally retained)."""
        size_classes = np.asarray(size_classes)
        mass_fractions = np.asarray(mass_fractions)
        self._psd_discharge_data = (size_classes, mass_fractions)
        self._psd_outlet = outlet
        if retained_size_classes is not None and retained_mass_fractions is not None:
            self._psd_retained_data = (
                np.asarray(retained_size_classes),
                np.asarray(retained_mass_fractions),
            )
        else:
            self._psd_retained_data = None
        if not self._psd_retained_btn.isChecked():
            self._chart.set_data(size_classes, mass_fractions)
        elif self._psd_retained_data is not None:
            self._chart.set_data(self._psd_retained_data[0], self._psd_retained_data[1])
        self._update_psd_stats()


class _TimeSeriesPanel(QFrame):
    """Panel containing time series chart."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Process Timeline")
        title.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 12pt; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Chart in scroll area so graphs are scrollable
        self._chart = TimeSeriesChart()
        self._chart.setMinimumHeight(560)
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
                width: 10px;
                border-radius: 5px;
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
        """)
        layout.addWidget(scroll, 1)

    def update_from_history(self, history: List):
        """Update chart from simulation history."""
        self._chart.set_data(history)


class _AnalyticsPanel(QFrame):
    """Panel containing process analytics cards."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Process Analytics")
        title.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 12pt; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Four-column layout: Flour Product | Retained Seeds | Breakage | Energy
        columns = QHBoxLayout()
        columns.setSpacing(12)

        # Flour Product Column (discharged through screen)
        flour_col = self._create_analytics_column("Flour Product", COLORS.KPI_THROUGHPUT, [
            ("d10", "flour_d10"),
            ("d50", "flour_d50"),
            ("d90", "flour_d90"),
            ("Mass Discharged", "flour_mass"),
            ("Particles Passed", "flour_count"),
            ("Passage Rate", "passage_rate"),
        ])
        columns.addWidget(flour_col["frame"])
        self._flour_stats = flour_col["stats"]

        # Retained Seeds Column (still in mill chamber)
        retained_col = self._create_analytics_column("Retained Seeds", COLORS.WARNING, [
            ("d10", "seed_d10"),
            ("d50", "seed_d50"),
            ("d90", "seed_d90"),
            ("Holdup Mass", "seed_mass"),
            ("Particles in Mill", "seed_count"),
            ("Mean Residence", "seed_residence"),
        ])
        columns.addWidget(retained_col["frame"])
        self._retained_stats = retained_col["stats"]

        # Breakage Column
        breakage_col = self._create_analytics_column("Breakage", COLORS.MILLING_PRIMARY, [
            ("Total Impacts", "impacts"),
            ("Breakage Events", "breakage_events"),
            ("Breakage Rate", "breakage_rate"),
            ("Avg Size Reduction", "size_reduction"),
            ("Total Fed", "total_fed"),
            ("Screen Aperture", "aperture"),
        ])
        columns.addWidget(breakage_col["frame"])
        self._breakage_stats = breakage_col["stats"]

        # Energy Column
        energy_col = self._create_analytics_column("Energy Efficiency", COLORS.KPI_POWER, [
            ("Total Energy", "total_energy"),
            ("Peak Power", "peak_power"),
            ("Mean Power", "mean_power"),
            ("Load Factor", "load_factor"),
            ("Specific Energy", "specific_energy"),
        ])
        columns.addWidget(energy_col["frame"])
        self._energy_stats = energy_col["stats"]

        layout.addLayout(columns)

    def _create_analytics_column(self, title: str, color: str, stats: List) -> Dict:
        """Create an analytics column with stats."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 10pt; font-weight: 600;")
        layout.addWidget(title_label)

        # Stats
        stats_dict = {}
        for label, key in stats:
            row = QHBoxLayout()
            label_widget = QLabel(label + ":")
            label_widget.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
            row.addWidget(label_widget)

            value_widget = QLabel("--")
            value_widget.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 9pt; font-weight: 600;")
            value_widget.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(value_widget)

            layout.addLayout(row)
            stats_dict[key] = value_widget

        layout.addStretch()
        return {"frame": frame, "stats": stats_dict}

    def update_analytics(self, result: Dict[str, Any]):
        """Update analytics from result data."""
        result_obj = result.get("result")
        outlet = result.get("outlet")

        if not result_obj:
            return

        # --- Flour Product (discharged through screen) ---
        if hasattr(result_obj, "d10_um"):
            self._flour_stats["flour_d10"].setText(f"{result_obj.d10_um:.0f} \u00b5m")
        if hasattr(result_obj, "d50_um"):
            self._flour_stats["flour_d50"].setText(f"{result_obj.d50_um:.0f} \u00b5m")
        if hasattr(result_obj, "d90_um"):
            self._flour_stats["flour_d90"].setText(f"{result_obj.d90_um:.0f} \u00b5m")
        if hasattr(result_obj, "psd_total_mass_kg"):
            self._flour_stats["flour_mass"].setText(f"{result_obj.psd_total_mass_kg:.2f} kg")
        if hasattr(result_obj, "total_particles_passed"):
            self._flour_stats["flour_count"].setText(f"{result_obj.total_particles_passed:,}")

        # --- Retained Seeds (still in mill chamber) ---
        if hasattr(result_obj, "retained_d10_um"):
            self._retained_stats["seed_d10"].setText(f"{result_obj.retained_d10_um:.0f} \u00b5m")
        if hasattr(result_obj, "retained_d50_um"):
            self._retained_stats["seed_d50"].setText(f"{result_obj.retained_d50_um:.0f} \u00b5m")
        if hasattr(result_obj, "retained_d90_um"):
            self._retained_stats["seed_d90"].setText(f"{result_obj.retained_d90_um:.0f} \u00b5m")
        if hasattr(result_obj, "holdup_kg_final"):
            self._retained_stats["seed_mass"].setText(f"{result_obj.holdup_kg_final:.2f} kg")
        if hasattr(result_obj, "total_particles_retained"):
            self._retained_stats["seed_count"].setText(f"{result_obj.total_particles_retained:,}")
        if outlet and hasattr(outlet, "mean_residence_time_s"):
            self._retained_stats["seed_residence"].setText(f"{outlet.mean_residence_time_s:.1f} s")

        if hasattr(result_obj, "history") and result_obj.history:
            history = result_obj.history

            # Passage rate for flour column
            total_passed = sum(s.num_passed_screen for s in history)
            total_fed = sum(s.num_fed for s in history)
            passage_rate = (total_passed / total_fed * 100) if total_fed > 0 else 0
            self._flour_stats["passage_rate"].setText(f"{passage_rate:.1f}%")

            # --- Breakage stats ---
            total_impacts = sum(s.num_impacts for s in history)
            total_breakage = sum(s.num_breakage_events for s in history)
            breakage_rate = (total_breakage / total_impacts * 100) if total_impacts > 0 else 0

            self._breakage_stats["impacts"].setText(f"{total_impacts:,}")
            self._breakage_stats["breakage_events"].setText(f"{total_breakage:,}")
            self._breakage_stats["breakage_rate"].setText(f"{breakage_rate:.1f}%")
            self._breakage_stats["total_fed"].setText(f"{total_fed:,}")

            reductions = [s.mean_size_reduction for s in history if s.mean_size_reduction < 1.0]
            if reductions:
                avg_reduction = np.mean(reductions)
                self._breakage_stats["size_reduction"].setText(f"{avg_reduction:.2f}x")

            # --- Energy stats ---
            duration = history[-1].time_s if history else 0
            power_values = [s.power_kw for s in history]
            if power_values:
                mean_power = np.mean(power_values)
                peak_power = np.max(power_values)
                total_energy = mean_power * duration / 3600
                load_factor = (mean_power / peak_power * 100) if peak_power > 0 else 0

                self._energy_stats["total_energy"].setText(f"{total_energy:.3f} kWh")
                self._energy_stats["peak_power"].setText(f"{peak_power:.1f} kW")
                self._energy_stats["mean_power"].setText(f"{mean_power:.1f} kW")
                self._energy_stats["load_factor"].setText(f"{load_factor:.1f}%")

        if hasattr(result_obj, "specific_energy_kwh_per_t"):
            self._energy_stats["specific_energy"].setText(
                f"{result_obj.specific_energy_kwh_per_t:.1f} kWh/t")

        # Screen aperture from config
        if hasattr(result_obj, "config"):
            aperture = getattr(result_obj.config, "screen_aperture_mm", None)
            if aperture:
                self._breakage_stats["aperture"].setText(f"{aperture:.2f} mm")


class _ProcessSummaryPanel(QFrame):
    """Panel containing process summary information."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Process Summary")
        title.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 12pt; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Stats in rows
        stats_grid = QGridLayout()
        stats_grid.setSpacing(12)

        self._stats = {}

        row_data = [
            ("Duration", "duration", 0, 0),
            ("Total Mass Fed", "total_mass", 0, 1),
            ("Flour Mass (discharged)", "discharge_mass", 1, 0),
            ("Flour d50", "flour_d50", 1, 1),
            ("Seed Mass (retained)", "holdup_kg", 2, 0),
            ("Seed d50", "seed_d50", 2, 1),
            ("Screen Efficiency", "efficiency", 3, 0),
            ("Mean Residence Time", "residence", 3, 1),
        ]

        for label, key, row, col in row_data:
            stat = self._create_stat_item(label)
            stats_grid.addWidget(stat["frame"], row, col)
            self._stats[key] = stat["value"]

        layout.addLayout(stats_grid)

    def _create_stat_item(self, label: str) -> Dict:
        """Create a stat item with label and value."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        layout.addWidget(label_widget)

        value_widget = QLabel("--")
        value_widget.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 14pt; font-weight: 600;")
        layout.addWidget(value_widget)

        return {"frame": frame, "value": value_widget}

    def update_summary(self, result: Dict[str, Any]):
        """Update process summary."""
        result_obj = result.get("result")
        outlet = result.get("outlet")

        if outlet:
            self._stats["residence"].setText(f"{outlet.mean_residence_time_s:.1f} s")

        if result_obj:
            if hasattr(result_obj, "history") and result_obj.history:
                history = result_obj.history
                duration = history[-1].time_s if history else 0
                self._stats["duration"].setText(f"{duration:.1f} s")
                total_fed = sum(s.num_fed for s in history)
                total_passed = sum(s.num_passed_screen for s in history)
                if total_fed > 0:
                    efficiency = total_passed / total_fed * 100
                    self._stats["efficiency"].setText(f"{efficiency:.1f}%")
            else:
                self._stats["duration"].setText("--")
                self._stats["efficiency"].setText("--")

            # Total mass fed
            if hasattr(result_obj, "total_mass_fed_kg"):
                self._stats["total_mass"].setText(f"{result_obj.total_mass_fed_kg:.2f} kg")

            # Flour (discharged) mass and d50
            if hasattr(result_obj, "psd_total_mass_kg"):
                self._stats["discharge_mass"].setText(f"{result_obj.psd_total_mass_kg:.2f} kg")
            if hasattr(result_obj, "d50_um"):
                self._stats["flour_d50"].setText(f"{result_obj.d50_um:.0f} \u00b5m")

            # Seed (retained) mass and d50
            if hasattr(result_obj, "holdup_kg_final"):
                self._stats["holdup_kg"].setText(f"{result_obj.holdup_kg_final:.2f} kg")
            if hasattr(result_obj, "retained_d50_um") and result_obj.retained_d50_um > 0:
                self._stats["seed_d50"].setText(f"{result_obj.retained_d50_um:.0f} \u00b5m")


class MillingResultsPage(QWidget):
    """Full-page comprehensive milling results display.

    Signals:
        back_clicked(): Emitted when back button is clicked
        export_clicked(): Emitted when export button is clicked
    """

    back_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"background: {COLORS.BG_BASE};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header bar
        header = self._create_header()
        layout.addWidget(header)

        # Main content in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {COLORS.BG_BASE};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {COLORS.BG_DARKEST};
                width: 10px;
                border-radius: 5px;
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
        """)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 24)
        content_layout.setSpacing(16)

        # KPI row
        self._kpi_row = _KPIRow()
        content_layout.addWidget(self._kpi_row)

        # PSD and Time Series side by side
        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        self._psd_panel = _PSDPanel()
        charts_row.addWidget(self._psd_panel, 1)

        self._timeseries_panel = _TimeSeriesPanel()
        charts_row.addWidget(self._timeseries_panel, 1)

        content_layout.addLayout(charts_row)

        # Analytics and Summary row
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        self._analytics_panel = _AnalyticsPanel()
        bottom_row.addWidget(self._analytics_panel, 2)

        self._summary_panel = _ProcessSummaryPanel()
        bottom_row.addWidget(self._summary_panel, 1)

        content_layout.addLayout(bottom_row)

        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _create_header(self) -> QFrame:
        """Create header bar with back button and title."""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-bottom: 1px solid {COLORS.BORDER_SUBTLE};
            }}
        """)
        header.setFixedHeight(56)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(16)

        # Back button
        back_btn = QPushButton("← Back to Simulation")
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
                padding: 8px 16px;
                color: {COLORS.TEXT_SECONDARY};
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        layout.addWidget(back_btn)

        # Title
        title = QLabel("Milling Simulation Results")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        layout.addStretch()

        # Export button
        export_btn = QPushButton("Export Results...")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.ACCENT};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                color: {COLORS.TEXT_INVERSE};
                font-size: 10pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.ACCENT_HOVER};
            }}
        """)
        export_btn.clicked.connect(self.export_clicked.emit)
        layout.addWidget(export_btn)

        return header

    def update_results(self, results: Dict[str, Any]):
        """Update all panels with new results."""
        result_obj = results.get("result")
        outlet = results.get("outlet")

        # KPI row
        self._kpi_row.update_values(outlet, result_obj)

        # PSD panel (discharge + retained)
        if result_obj and hasattr(result_obj, "psd_size_classes_m"):
            retained_sc = getattr(result_obj, "retained_psd_size_classes_m", None)
            retained_mf = getattr(result_obj, "retained_psd_mass_fractions", None)
            has_retained = (
                retained_sc is not None and len(retained_sc) > 0
                and retained_mf is not None and len(retained_mf) > 0
            )
            self._psd_panel.update_psd(
                result_obj.psd_size_classes_m * 1e6,
                result_obj.psd_mass_fractions,
                outlet,
                retained_size_classes=retained_sc * 1e6 if has_retained else None,
                retained_mass_fractions=retained_mf if has_retained else None,
            )

        # Time series panel
        if result_obj and hasattr(result_obj, "history"):
            self._timeseries_panel.update_from_history(result_obj.history)

        # Analytics panel
        self._analytics_panel.update_analytics(results)

        # Summary panel
        self._summary_panel.update_summary(results)
