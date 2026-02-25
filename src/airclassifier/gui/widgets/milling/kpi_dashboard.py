"""
Milling KPI Dashboard
=====================

Live KPI dashboard for hammer mill simulation with
semantic color-coded cards and trend visualization.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QWidget, QSizePolicy, QGraphicsDropShadowEffect,
)

from ...theme import COLORS
from ..common import AnimatedKPICard, RadialGaugeWidget


class MillingKPIDashboard(QFrame):
    """Live KPI dashboard for hammer mill simulation.

    Displays key performance indicators with sparklines and trends:
    - Throughput (kg/h)
    - d50 particle size (um)
    - Power consumption (kW)
    - Specific energy (kWh/t)

    Signals:
        card_clicked(str): Emitted when a KPI card is clicked
    """

    card_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        """Apply styling."""
        self.setObjectName("kpiDashboard")
        self.setStyleSheet(f"""
            QFrame#kpiDashboard {{
                background: transparent;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _setup_ui(self):
        """Build the dashboard UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Throughput
        self._throughput_card = AnimatedKPICard(
            title="Throughput",
            unit="kg/h",
            semantic_color=COLORS.KPI_THROUGHPUT,
            show_sparkline=True,
            show_delta=True,
        )
        self._throughput_card.clicked.connect(lambda: self.card_clicked.emit("throughput"))
        layout.addWidget(self._throughput_card)

        # d50
        self._d50_card = AnimatedKPICard(
            title="d50",
            unit="um",
            semantic_color=COLORS.KPI_SIZE,
            show_sparkline=True,
            show_delta=True,
        )
        self._d50_card.clicked.connect(lambda: self.card_clicked.emit("d50"))
        layout.addWidget(self._d50_card)

        # Power
        self._power_card = AnimatedKPICard(
            title="Power",
            unit="kW",
            semantic_color=COLORS.KPI_POWER,
            precision=2,
            show_sparkline=True,
            show_delta=True,
        )
        self._power_card.clicked.connect(lambda: self.card_clicked.emit("power"))
        layout.addWidget(self._power_card)

        # Power gauge (optional compact view)
        self._power_gauge = RadialGaugeWidget(
            min_value=0,
            max_value=50,
            warning_threshold=35,
            danger_threshold=45,
            unit="kW",
            title="Load",
        )
        self._power_gauge.setFixedSize(80, 80)
        self._power_gauge.hide()  # Hidden by default, can be shown
        layout.addWidget(self._power_gauge)

    def update_kpis(
        self,
        throughput: Optional[float] = None,
        d50: Optional[float] = None,
        power: Optional[float] = None,
        animate: bool = True,
    ):
        """Update KPI values.

        Args:
            throughput: Throughput in kg/h
            d50: d50 in micrometers
            power: Power in kW
            animate: Whether to animate value changes
        """
        if throughput is not None:
            self._throughput_card.set_value(throughput, animate=animate)

        if d50 is not None:
            self._d50_card.set_value(d50, animate=animate)

        if power is not None:
            self._power_card.set_value(power, animate=animate)
            self._power_gauge.set_value(power, animate=animate)

    def update_from_state(self, state: Any, animate: bool = True):
        """Update from a MillingStepState object.

        Args:
            state: MillingStepState with discharge_rate_kg_per_s, d50_m, power_kw
            animate: Whether to animate changes
        """
        if state is None:
            return

        throughput = state.discharge_rate_kg_per_s * 3600 if hasattr(state, "discharge_rate_kg_per_s") else None
        d50 = state.d50_m * 1e6 if hasattr(state, "d50_m") else None
        power = state.power_kw if hasattr(state, "power_kw") else None

        self.update_kpis(throughput=throughput, d50=d50, power=power, animate=animate)

    def clear(self):
        """Reset all cards."""
        self._throughput_card.clear()
        self._d50_card.clear()
        self._power_card.clear()
        self._power_gauge.set_value(0, animate=False)

    def show_gauge(self, show: bool = True):
        """Show or hide the power gauge."""
        self._power_gauge.setVisible(show)

    @property
    def throughput(self) -> float:
        return self._throughput_card.value

    @property
    def d50(self) -> float:
        return self._d50_card.value

    @property
    def power(self) -> float:
        return self._power_card.value
