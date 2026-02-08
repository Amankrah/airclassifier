"""
Pretreatment GUI Panel
======================

A PySide6 dock widget for the Air Classifier Designer that provides
recipe selection, run/pause/stop controls, and real-time KPI cards
for the GP-15 RF pretreatment simulation (engineering guide §9.3).

This panel integrates into the existing MainWindow's dock system::

    from airclassifier.pretreatment.gui_panel import PretreatmentPanel
    panel = PretreatmentPanel()
    main_window.addDockWidget(Qt.RightDockWidgetArea, panel)

The panel communicates with GP15Simulator through signals/slots:
    - ``recipe_changed(Recipe)`` — user selects or edits a recipe
    - ``run_requested(float)`` — user clicks Run with duration
    - ``stop_requested()`` — user clicks Stop
    - ``step_completed(StepState)`` — simulation reports a timestep

Phase 4 skeleton — UI layout only.  Full widget implementation
deferred to GUI integration sprint.
"""

from __future__ import annotations

from typing import Optional

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QDockWidget,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QVBoxLayout,
        QWidget,
    )
    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

from .config import MachineConfig, Recipe
from .physics.coupling import StepState


if _HAS_PYSIDE6:

    class PretreatmentPanel(QDockWidget):
        """Pretreatment simulation control panel.

        Provides recipe editing, simulation controls, and live KPI
        display for the GP-15 RF heating digital twin.
        """

        # Signals
        recipe_changed = Signal(object)
        run_requested = Signal(float)
        stop_requested = Signal()

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__("Pretreatment (GP-15)", parent)
            self.setObjectName("pretreatment_panel")
            self._build_ui()

        def _build_ui(self):
            container = QWidget()
            layout = QVBoxLayout(container)

            # ── Recipe group ──────────────────────────────────────
            recipe_grp = QGroupBox("Recipe")
            recipe_form = QFormLayout()

            self._gap_spin = QDoubleSpinBox()
            self._gap_spin.setRange(20, 300)
            self._gap_spin.setValue(80)
            self._gap_spin.setSuffix(" mm")
            recipe_form.addRow("Electrode gap:", self._gap_spin)

            self._speed_spin = QDoubleSpinBox()
            self._speed_spin.setRange(0.1, 2.0)
            self._speed_spin.setValue(0.5)
            self._speed_spin.setDecimals(2)
            self._speed_spin.setSuffix(" m/min")
            recipe_form.addRow("Belt speed:", self._speed_spin)

            self._fan_spin = QDoubleSpinBox()
            self._fan_spin.setRange(5, 60)
            self._fan_spin.setValue(30)
            self._fan_spin.setSuffix(" Hz")
            recipe_form.addRow("Extraction fan:", self._fan_spin)

            self._duration_spin = QDoubleSpinBox()
            self._duration_spin.setRange(1, 3600)
            self._duration_spin.setValue(120)
            self._duration_spin.setSuffix(" s")
            recipe_form.addRow("Duration:", self._duration_spin)

            recipe_grp.setLayout(recipe_form)
            layout.addWidget(recipe_grp)

            # ── Controls ──────────────────────────────────────────
            ctrl_layout = QHBoxLayout()
            self._run_btn = QPushButton("Run")
            self._run_btn.clicked.connect(self._on_run)
            self._stop_btn = QPushButton("Stop")
            self._stop_btn.clicked.connect(self._on_stop)
            self._stop_btn.setEnabled(False)
            ctrl_layout.addWidget(self._run_btn)
            ctrl_layout.addWidget(self._stop_btn)
            layout.addLayout(ctrl_layout)

            # ── KPI display ───────────────────────────────────────
            kpi_grp = QGroupBox("Live KPIs")
            kpi_form = QFormLayout()

            self._lbl_time = QLabel("0.0 s")
            self._lbl_T = QLabel("-- °C")
            self._lbl_M = QLabel("-- %")
            self._lbl_power = QLabel("-- kW")
            self._lbl_current = QLabel("-- A")

            kpi_form.addRow("Time:", self._lbl_time)
            kpi_form.addRow("T (mean):", self._lbl_T)
            kpi_form.addRow("M (mean):", self._lbl_M)
            kpi_form.addRow("RF power:", self._lbl_power)
            kpi_form.addRow("Ia:", self._lbl_current)

            kpi_grp.setLayout(kpi_form)
            layout.addWidget(kpi_grp)

            layout.addStretch()
            self.setWidget(container)

        def update_kpis(self, state: StepState):
            """Update the KPI labels from a simulation step."""
            self._lbl_time.setText(f"{state.time_s:.1f} s")
            self._lbl_T.setText(f"{state.T_mean_c:.1f} °C")
            self._lbl_M.setText(f"{state.M_mean_wb * 100:.2f} %")
            self._lbl_power.setText(f"{state.rf_power_kw:.2f} kW")
            self._lbl_current.setText(f"{state.anode_current_a:.2f} A")

        def get_recipe(self) -> Recipe:
            """Build a Recipe from the current UI values."""
            return Recipe(
                name="gui_recipe",
                recipe_number=0,
                electrode_gap_mm=self._gap_spin.value(),
                belt_speed_m_per_min=self._speed_spin.value(),
                extraction_fan_hz=self._fan_spin.value(),
            )

        def _on_run(self):
            self.run_requested.emit(self._duration_spin.value())
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)

        def _on_stop(self):
            self.stop_requested.emit()
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

else:
    # Stub when PySide6 is not installed
    class PretreatmentPanel:  # type: ignore[no-redef]
        """Placeholder — PySide6 not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PySide6 is required for the pretreatment GUI panel. "
                "Install with: pip install PySide6"
            )
