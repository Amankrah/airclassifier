"""
Pretreatment KPI Dashboard
==========================

Live KPI dashboard for GP-15 RF heating simulation with
semantic color-coded cards and trend visualization.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QWidget, QSizePolicy,
)

from ...theme import COLORS
from ..common import AnimatedKPICard, RadialGaugeWidget


class PretreatmentKPIDashboard(QFrame):
    """Live KPI dashboard for GP-15 RF heating simulation.

    Displays key performance indicators with sparklines and trends:
    - Temperature (C)
    - Moisture (% wb)
    - RF Power (kW)
    - Anode Current (A)
    - Electrode Gap (mm)
    - Simulation Time (s)

    Signals:
        card_clicked(str): Emitted when a KPI card is clicked
    """

    card_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        self._setup_style()
        self._setup_ui()

    def _setup_style(self):
        """Apply styling."""
        self.setObjectName("pretreatKPIDashboard")
        self.setStyleSheet(f"""
            QFrame#pretreatKPIDashboard {{
                background: transparent;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _setup_ui(self):
        """Build the dashboard UI."""
        if self._compact:
            self._setup_compact_ui()
        else:
            self._setup_full_ui()

    def _setup_full_ui(self):
        """Full 2-row KPI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Row 1: Temperature, Moisture, RF Power
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        self._temp_card = AnimatedKPICard(
            title="Temperature",
            unit="\u00b0C",
            semantic_color=COLORS.KPI_TEMPERATURE,
            show_sparkline=True,
            show_delta=True,
            precision=1,
        )
        self._temp_card.clicked.connect(lambda: self.card_clicked.emit("temperature"))
        row1.addWidget(self._temp_card)

        self._moisture_card = AnimatedKPICard(
            title="Moisture",
            unit="% wb",
            semantic_color=COLORS.KPI_MOISTURE,
            show_sparkline=True,
            show_delta=True,
            precision=2,
        )
        self._moisture_card.clicked.connect(lambda: self.card_clicked.emit("moisture"))
        row1.addWidget(self._moisture_card)

        self._power_card = AnimatedKPICard(
            title="RF Power",
            unit="kW",
            semantic_color=COLORS.KPI_RF_POWER,
            show_sparkline=True,
            show_delta=True,
            precision=2,
        )
        self._power_card.clicked.connect(lambda: self.card_clicked.emit("power"))
        row1.addWidget(self._power_card)

        layout.addLayout(row1)

        # Row 2: Anode Current, Electrode Gap, Time
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        self._anode_card = AnimatedKPICard(
            title="Anode Current",
            unit="A",
            semantic_color=COLORS.KPI_ANODE_CURRENT,
            show_sparkline=True,
            show_delta=True,
            precision=2,
        )
        self._anode_card.clicked.connect(lambda: self.card_clicked.emit("anode"))
        row2.addWidget(self._anode_card)

        self._gap_card = AnimatedKPICard(
            title="Electrode Gap",
            unit="mm",
            semantic_color=COLORS.KPI_ELECTRODE_GAP,
            show_sparkline=True,
            show_delta=True,
            precision=0,
        )
        self._gap_card.clicked.connect(lambda: self.card_clicked.emit("gap"))
        row2.addWidget(self._gap_card)

        self._time_card = AnimatedKPICard(
            title="Sim Time",
            unit="s",
            semantic_color=COLORS.ACCENT,
            show_sparkline=False,
            show_delta=False,
            precision=1,
        )
        self._time_card.clicked.connect(lambda: self.card_clicked.emit("time"))
        row2.addWidget(self._time_card)

        layout.addLayout(row2)

        # Optional power gauge
        self._power_gauge = RadialGaugeWidget(
            min_value=0,
            max_value=30,
            warning_threshold=20,
            danger_threshold=25,
            unit="kW",
            title="RF Load",
        )
        self._power_gauge.setFixedSize(90, 90)
        self._power_gauge.hide()

    def _setup_compact_ui(self):
        """Compact single-row KPI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._temp_card = AnimatedKPICard(
            title="Temp",
            unit="\u00b0C",
            semantic_color=COLORS.KPI_TEMPERATURE,
            show_sparkline=False,
            show_delta=True,
            precision=1,
        )
        layout.addWidget(self._temp_card)

        self._moisture_card = AnimatedKPICard(
            title="Moisture",
            unit="%",
            semantic_color=COLORS.KPI_MOISTURE,
            show_sparkline=False,
            show_delta=True,
            precision=2,
        )
        layout.addWidget(self._moisture_card)

        self._power_card = AnimatedKPICard(
            title="RF Power",
            unit="kW",
            semantic_color=COLORS.KPI_RF_POWER,
            show_sparkline=False,
            show_delta=True,
            precision=2,
        )
        layout.addWidget(self._power_card)

        self._anode_card = AnimatedKPICard(
            title="Ia",
            unit="A",
            semantic_color=COLORS.KPI_ANODE_CURRENT,
            show_sparkline=False,
            show_delta=False,
            precision=2,
        )
        layout.addWidget(self._anode_card)

        self._gap_card = AnimatedKPICard(
            title="Gap",
            unit="mm",
            semantic_color=COLORS.KPI_ELECTRODE_GAP,
            show_sparkline=False,
            show_delta=False,
            precision=0,
        )
        layout.addWidget(self._gap_card)

        self._time_card = AnimatedKPICard(
            title="Time",
            unit="s",
            semantic_color=COLORS.ACCENT,
            show_sparkline=False,
            show_delta=False,
            precision=1,
        )
        layout.addWidget(self._time_card)

        self._power_gauge = None

    def update_kpis(
        self,
        temperature: Optional[float] = None,
        moisture: Optional[float] = None,
        power: Optional[float] = None,
        anode_current: Optional[float] = None,
        electrode_gap: Optional[float] = None,
        sim_time: Optional[float] = None,
        animate: bool = True,
    ):
        """Update KPI values.

        Args:
            temperature: Temperature in C
            moisture: Moisture in % wb (0.10 = 10%)
            power: RF power in kW
            anode_current: Anode current in A
            electrode_gap: Electrode gap in mm
            sim_time: Simulation time in seconds
            animate: Whether to animate value changes
        """
        if temperature is not None:
            self._temp_card.set_value(temperature, animate=animate)

        if moisture is not None:
            # Convert to percentage if in decimal form
            moisture_pct = moisture * 100 if moisture < 1.0 else moisture
            self._moisture_card.set_value(moisture_pct, animate=animate)

        if power is not None:
            self._power_card.set_value(power, animate=animate)
            if self._power_gauge:
                self._power_gauge.set_value(power, animate=animate)

        if anode_current is not None:
            self._anode_card.set_value(anode_current, animate=animate)

        if electrode_gap is not None:
            self._gap_card.set_value(electrode_gap, animate=animate)

        if sim_time is not None:
            self._time_card.set_value(sim_time, animate=animate)

    def update_from_state(self, state: Any, animate: bool = True):
        """Update from a simulation state object.

        Args:
            state: Simulation state with T_mean_c, M_mean_wb, rf_power_kw, etc.
            animate: Whether to animate changes
        """
        if state is None:
            return

        temperature = getattr(state, "T_mean_c", None) or getattr(state, "T_outfeed_c", None)
        moisture = getattr(state, "M_mean_wb", None) or getattr(state, "M_outfeed_wb", None)
        power = getattr(state, "rf_power_kw", None)
        anode = getattr(state, "anode_current_a", None)
        gap = getattr(state, "electrode_gap_mm", None)
        time_s = getattr(state, "time_s", None)

        self.update_kpis(
            temperature=temperature,
            moisture=moisture,
            power=power,
            anode_current=anode,
            electrode_gap=gap,
            sim_time=time_s,
            animate=animate,
        )

    def clear(self):
        """Reset all cards."""
        self._temp_card.clear()
        self._moisture_card.clear()
        self._power_card.clear()
        self._anode_card.clear()
        self._gap_card.clear()
        self._time_card.clear()
        if self._power_gauge:
            self._power_gauge.set_value(0, animate=False)

    def show_gauge(self, show: bool = True):
        """Show or hide the power gauge."""
        if self._power_gauge:
            self._power_gauge.setVisible(show)

    @property
    def temperature(self) -> float:
        return self._temp_card.value

    @property
    def moisture(self) -> float:
        return self._moisture_card.value

    @property
    def power(self) -> float:
        return self._power_card.value

    @property
    def anode_current(self) -> float:
        return self._anode_card.value

    @property
    def electrode_gap(self) -> float:
        return self._gap_card.value

    @property
    def sim_time(self) -> float:
        return self._time_card.value
