"""
Pretreatment Results Overlay
============================

Slide-in results panel that overlays the 3D viewport,
preserving context while displaying detailed results.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import (
    Qt, Signal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QWidget, QPushButton, QScrollArea,
    QTabWidget, QSizePolicy, QGraphicsDropShadowEffect,
)

from ...theme import COLORS, ANIMATIONS
from ..common import AnimatedKPICard
from .desirability_panel import DesirabilityPanel


class ResultKPICard(QFrame):
    """Static result card for the results overlay."""

    def __init__(
        self,
        title: str,
        accent: str = COLORS.ACCENT,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._accent = accent
        self._setup_ui(title)

    def _setup_ui(self, title: str):
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._value_label = QLabel("--")
        self._value_label.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: 700;
            color: {self._accent};
        """)
        layout.addWidget(self._value_label)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"""
            font-size: 8pt;
            color: {COLORS.TEXT_MUTED};
        """)
        layout.addWidget(self._title_label)

    def set_value(self, text: str):
        """Set the display value."""
        self._value_label.setText(text)


class PretreatmentResultsOverlay(QFrame):
    """Slide-in results panel for pretreatment simulation.

    Features:
    - Animated slide-in from right
    - Semi-transparent backdrop
    - Tabbed interface: Summary | Desirability | Details
    - Export toolbar

    Signals:
        closed(): Emitted when overlay is closed
        export_requested(str): Emitted with format when export is clicked
    """

    closed = Signal()
    export_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._visible = False
        self._results: Optional[Dict[str, Any]] = None

        self._setup_style()
        self._setup_ui()
        self._setup_animations()

        # Start hidden
        self.hide()

    def _setup_style(self):
        self.setObjectName("pretreatResultsOverlay")
        self.setStyleSheet(f"""
            QFrame#pretreatResultsOverlay {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS.BG_ELEVATED}f0, stop:1 {COLORS.BG_ELEVATED});
                border-left: 1px solid {COLORS.BORDER};
            }}
        """)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(-10, 0)
        self.setGraphicsEffect(shadow)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._build_header()
        layout.addWidget(header)

        # Content tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: transparent;
            }}
            QTabBar::tab {{
                background: {COLORS.BG_SURFACE};
                color: {COLORS.TEXT_MUTED};
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS.BG_ELEVATED};
                color: {COLORS.PRETREAT_PRIMARY};
                font-weight: 600;
            }}
            QTabBar::tab:hover:!selected {{
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)

        # Summary tab
        summary_tab = self._build_summary_tab()
        self._tabs.addTab(summary_tab, "Summary")

        # Desirability tab
        desirability_tab = self._build_desirability_tab()
        self._tabs.addTab(desirability_tab, "Desirability")

        # Details tab
        details_tab = self._build_details_tab()
        self._tabs.addTab(details_tab, "Details")

        layout.addWidget(self._tabs, 1)

        # Footer with export buttons
        footer = self._build_footer()
        layout.addWidget(footer)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border-bottom: 1px solid {COLORS.BORDER};
            }}
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)

        close_btn = QPushButton("\u2715")
        close_btn.setFixedSize(32, 32)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                font-size: 14pt;
                color: {COLORS.TEXT_MUTED};
            }}
            QPushButton:hover {{
                background: {COLORS.DANGER};
                color: {COLORS.TEXT_INVERSE};
                border-color: {COLORS.DANGER};
            }}
        """)
        close_btn.clicked.connect(self.slide_out)
        layout.addWidget(close_btn)

        self._title_label = QLabel("GP-15 Simulation Results")
        self._title_label.setStyleSheet(f"""
            font-size: 12pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(self._title_label, 1)

        return header

    def _build_summary_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # KPI Grid
        kpi_section = QLabel("Key Performance Indicators")
        kpi_section.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_SECONDARY};
        """)
        layout.addWidget(kpi_section)

        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(10)

        self._rc_moisture = ResultKPICard("Outfeed Moisture", COLORS.KPI_MOISTURE)
        self._rc_temp = ResultKPICard("Outfeed Temp", COLORS.KPI_TEMPERATURE)
        self._rc_max_temp = ResultKPICard("Max Temperature", COLORS.DANGER)
        self._rc_cv = ResultKPICard("Moisture CV", COLORS.ACCENT)

        kpi_grid.addWidget(self._rc_moisture, 0, 0)
        kpi_grid.addWidget(self._rc_temp, 0, 1)
        kpi_grid.addWidget(self._rc_max_temp, 0, 2)
        kpi_grid.addWidget(self._rc_cv, 0, 3)

        self._rc_energy = ResultKPICard("RF Energy", COLORS.KPI_ENERGY)
        self._rc_specific = ResultKPICard("Specific Energy", COLORS.WARNING)
        self._rc_throughput = ResultKPICard("Throughput", COLORS.KPI_THROUGHPUT)
        self._rc_wall = ResultKPICard("Wall-Clock Time", COLORS.TEXT_SECONDARY)

        kpi_grid.addWidget(self._rc_energy, 1, 0)
        kpi_grid.addWidget(self._rc_specific, 1, 1)
        kpi_grid.addWidget(self._rc_throughput, 1, 2)
        kpi_grid.addWidget(self._rc_wall, 1, 3)

        self._rc_protein = ResultKPICard("Protein Quality", COLORS.SUCCESS)
        self._rc_mass = ResultKPICard("Mass Balance", COLORS.INFO)
        self._rc_gap = ResultKPICard("Final Gap", COLORS.KPI_ELECTRODE_GAP)
        self._rc_speed = ResultKPICard("Sim Speed", COLORS.TEXT_MUTED)

        kpi_grid.addWidget(self._rc_protein, 2, 0)
        kpi_grid.addWidget(self._rc_mass, 2, 1)
        kpi_grid.addWidget(self._rc_gap, 2, 2)
        kpi_grid.addWidget(self._rc_speed, 2, 3)

        layout.addLayout(kpi_grid)

        # Run parameters summary
        params_section = QLabel("Run Parameters")
        params_section.setStyleSheet(f"""
            font-size: 10pt;
            font-weight: 600;
            color: {COLORS.TEXT_SECONDARY};
            margin-top: 8px;
        """)
        layout.addWidget(params_section)

        self._params_label = QLabel()
        self._params_label.setStyleSheet(f"""
            font-size: 9pt;
            color: {COLORS.TEXT_MUTED};
            background: {COLORS.BG_SURFACE};
            border: 1px solid {COLORS.BORDER_SUBTLE};
            border-radius: 6px;
            padding: 12px;
        """)
        self._params_label.setWordWrap(True)
        layout.addWidget(self._params_label)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_desirability_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Desirability panel
        self._desirability_panel = DesirabilityPanel()
        layout.addWidget(self._desirability_panel)

        # Explanation
        explanation = QLabel(
            "The Process Desirability score combines five key dimensions to evaluate "
            "overall pretreatment quality for downstream protein separation and flavour improvement. "
            "Each dimension is scored 0-100% based on target thresholds specific to the material."
        )
        explanation.setStyleSheet(f"""
            font-size: 9pt;
            color: {COLORS.TEXT_MUTED};
            background: {COLORS.BG_SURFACE};
            border: 1px solid {COLORS.BORDER_SUBTLE};
            border-radius: 6px;
            padding: 12px;
        """)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_details_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Placeholder for detailed results
        self._details_text = QLabel("Detailed simulation metrics will appear here after a run completes.")
        self._details_text.setStyleSheet(f"""
            font-size: 9pt;
            color: {COLORS.TEXT_MUTED};
            background: {COLORS.BG_SURFACE};
            border: 1px solid {COLORS.BORDER_SUBTLE};
            border-radius: 6px;
            padding: 16px;
        """)
        self._details_text.setWordWrap(True)
        self._details_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._details_text)

        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setFixedHeight(56)
        footer.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border-top: 1px solid {COLORS.BORDER};
            }}
        """)

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 8, 16, 8)

        export_label = QLabel("Export:")
        export_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
        layout.addWidget(export_label)

        btn_style = f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 9pt;
                color: {COLORS.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
        """

        for fmt in ["CSV", "JSON", "PNG", "PDF"]:
            btn = QPushButton(fmt)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda checked, f=fmt: self.export_requested.emit(f.lower()))
            layout.addWidget(btn)

        layout.addStretch()

        # Full results button
        full_btn = QPushButton("View Full Results")
        full_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.PRETREAT_PRIMARY};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 10pt;
                font-weight: 600;
                color: {COLORS.TEXT_INVERSE};
            }}
            QPushButton:hover {{
                background: {COLORS.PRETREAT_SECONDARY};
            }}
        """)
        full_btn.clicked.connect(lambda: self.export_requested.emit("full"))
        layout.addWidget(full_btn)

        return footer

    def _setup_animations(self):
        self._slide_anim = QPropertyAnimation(self, b"geometry")
        self._slide_anim.setDuration(ANIMATIONS.NORMAL)
        self._slide_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def slide_in(self):
        """Animate the overlay sliding in from the right."""
        if self._visible:
            return

        parent = self.parentWidget()
        if not parent:
            self.show()
            self._visible = True
            return

        width = min(600, parent.width() - 100)
        height = parent.height()

        # Start off-screen
        start_x = parent.width()
        end_x = parent.width() - width

        self.setGeometry(start_x, 0, width, height)
        self.show()
        self.raise_()

        self._slide_anim.setStartValue(self.geometry())
        end_rect = self.geometry()
        end_rect.moveLeft(end_x)
        self._slide_anim.setEndValue(end_rect)
        self._slide_anim.start()

        self._visible = True

    def slide_out(self):
        """Animate the overlay sliding out to the right."""
        if not self._visible:
            return

        parent = self.parentWidget()
        if not parent:
            self.hide()
            self._visible = False
            self.closed.emit()
            return

        end_x = parent.width()

        self._slide_anim.setStartValue(self.geometry())
        end_rect = self.geometry()
        end_rect.moveLeft(end_x)
        self._slide_anim.setEndValue(end_rect)
        self._slide_anim.finished.connect(self._on_slide_out_done)
        self._slide_anim.start()

    def _on_slide_out_done(self):
        self._slide_anim.finished.disconnect(self._on_slide_out_done)
        self.hide()
        self._visible = False
        self.closed.emit()

    def update_results(self, results: Dict[str, Any]):
        """Update the overlay with simulation results."""
        self._results = results

        outlet = results.get("outlet")
        result = results.get("result")

        # Update title
        run_mass = results.get("run_mass_kg", 0)
        gap = results.get("gap_mm", 0)
        bed = results.get("bed_depth_mm", 0)
        duration = results.get("duration_s", 0)
        self._title_label.setText(
            f"GP-15 Results \u2014 {run_mass:.0f} kg | Gap {gap:.0f} mm | "
            f"Bed {bed:.0f} mm | {duration:.0f} s"
        )

        # Update KPIs
        if outlet:
            self._rc_moisture.set_value(f"{outlet.avg_moisture_wb:.1%}")
            self._rc_temp.set_value(f"{outlet.sensor_temperature_c:.1f} \u00b0C")
            self._rc_max_temp.set_value(f"{outlet.max_temperature_c:.1f} \u00b0C")
            self._rc_cv.set_value(f"{outlet.moisture_uniformity:.3f}")
            self._rc_energy.set_value(f"{outlet.total_energy_kwh:.3f} kWh")

            # Specific energy
            water_removed = results.get("initial_moisture", 0.10) - outlet.avg_moisture_wb
            collected_mass = results.get("collected_mass_kg", run_mass)
            if water_removed > 0 and collected_mass > 0:
                water_kg = water_removed * collected_mass
                specific = outlet.total_energy_kwh / max(water_kg, 0.001)
                self._rc_specific.set_value(f"{specific:.2f} kWh/kg")
            else:
                self._rc_specific.set_value("--")

            # Throughput
            if duration > 0:
                throughput = collected_mass / duration * 3600
                self._rc_throughput.set_value(f"{throughput:.0f} kg/h")
            else:
                self._rc_throughput.set_value("--")

        # Wall time and sim speed
        elapsed = results.get("elapsed_s", 0)
        self._rc_wall.set_value(f"{elapsed:.1f} s")

        if elapsed > 0 and duration > 0:
            sim_speed = duration / elapsed
            self._rc_speed.set_value(f"{sim_speed:.1f}x")
        else:
            self._rc_speed.set_value("--")

        # Protein quality
        particle_data = results.get("particle_data", {})
        vicilin = particle_data.get("vicilin_native_mean")
        legumin = particle_data.get("legumin_native_mean")
        if vicilin is not None and legumin is not None:
            loss = 100 - (vicilin + legumin) / 2 * 100
            self._rc_protein.set_value(f"{loss:.1f}% loss")
        else:
            self._rc_protein.set_value("--")

        # Mass balance
        dispatched = results.get("dispatched_mass_kg", 0)
        collected = results.get("collected_mass_kg", 0)
        if dispatched > 0:
            balance = collected / dispatched * 100
            self._rc_mass.set_value(f"{balance:.1f}%")
        else:
            self._rc_mass.set_value("--")

        # Final gap
        ts = results.get("time_series", {})
        gaps = ts.get("electrode_gap_mm", [])
        if gaps:
            self._rc_gap.set_value(f"{gaps[-1]:.0f} mm")
        else:
            self._rc_gap.set_value(f"{gap:.0f} mm")

        # Update parameters summary
        params_text = (
            f"Material: {results.get('material', 'yellow_pea')}  |  "
            f"Initial Moisture: {results.get('initial_moisture', 0.10):.1%}  |  "
            f"Initial Temp: {results.get('initial_temp_c', 17.6):.1f}\u00b0C\n"
            f"Electrode Gap: {gap:.0f} mm  |  "
            f"Belt Speed: {results.get('belt_speed', 0.2):.2f} m/min  |  "
            f"Bed Depth: {bed:.0f} mm\n"
            f"Run Mass: {run_mass:.0f} kg  |  "
            f"Duration: {duration:.0f} s ({duration/60:.1f} min)"
        )
        self._params_label.setText(params_text)

        # Update desirability (if available)
        # This would require computing desirability from outlet data
        # For now, we leave it to be updated separately

    def clear(self):
        """Reset the overlay."""
        self._results = None
        self._title_label.setText("GP-15 Simulation Results")

        for card in [self._rc_moisture, self._rc_temp, self._rc_max_temp,
                     self._rc_cv, self._rc_energy, self._rc_specific,
                     self._rc_throughput, self._rc_wall, self._rc_protein,
                     self._rc_mass, self._rc_gap, self._rc_speed]:
            card.set_value("--")

        self._params_label.setText("")
        self._desirability_panel.clear()

    @property
    def is_visible(self) -> bool:
        return self._visible
