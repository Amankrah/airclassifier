"""
Pretreatment Page — GP-15 RF Heating Full-Page View
====================================================

Complete simulation + visualization page for the GP-15 RF dielectric
heating machine.  Uses an internal ``QStackedWidget`` to switch between:

  - **Simulation View** — 3D PyVista viewport with live field overlays
    (rollers, belt, particles, temperature), control panel, and KPIs.
  - **Results View** — Full-page detailed results with KPI cards,
    3x3 matplotlib plots, outfeed cross-section, and export toolbar.

Architecture (modular sections)::

    PretreatmentPage
        └── QStackedWidget
            ├── Page 0: Simulation View
            │   ├── QSplitter(H): 3D Viewport + Control Panel
            │   └── Bottom bar: status + "View Full Results" button
            └── Page 1: Results View
                ├── Header: Back + title + Export (CSV/JSON/PNG/PDF)
                └── QScrollArea
                    ├── KPI section (_build_results_kpi_section)
                    ├── Desirability section (_build_desirability_section)
                    ├── 3×3 time-series plots (_build_results_plots_section / _draw_simulation_plots)
                    ├── Particle analysis (_build_results_particle_section / _draw_particle_plots)
                    └── Outfeed cross-section (_build_results_outfeed_section / _draw_outfeed_section)

Reporting is aligned with examples/simulate_and_visualize.py and canonical
PretreatmentResult/OutletState: time_series from result.time_series + controller_state,
sensor temperature for desirability and PDF, specific energy (kWh/kg water), mass balance %,
and "at oven exit (peak)" when applicable.

Usage::

    from airclassifier.gui.pages import PretreatmentPage

    page = PretreatmentPage()
    stacked.addWidget(page)
"""

from __future__ import annotations

import time
import traceback
from typing import Optional, Dict, Any, List

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import COLORS
from ...pretreatment.desirability import (
    DesirabilityProfile,
    DesirabilityResult,
    PROFILES,
    score_desirability,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False


# ═══════════════════════════════════════════════════════════════════════
#  Reusable widgets
# ═══════════════════════════════════════════════════════════════════════

class _StatCard(QFrame):
    """Themed KPI display card."""

    def __init__(self, title: str, accent: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self._value_label = QLabel("--")
        color = accent or COLORS.ACCENT
        self._value_label.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {color};"
            " border: none; background: transparent;"
        )
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._value_label)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._title_label)

    def set_value(self, text: str):
        self._value_label.setText(text)


def _scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


# ═══════════════════════════════════════════════════════════════════════
#  Live Simulation Render helpers
# ═══════════════════════════════════════════════════════════════════════

def _mesh_to_polydata(v, t):
    """Build PyVista PolyData from (vertices, triangles)."""
    import pyvista as pv
    n = t.shape[0]
    faces = np.empty((n, 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = t
    return pv.PolyData(v.copy(), faces.ravel())


def _collect_roller_entries(layout):
    """Flatten roller layout dict into (name, x, y, r) tuples."""
    entries = []
    for key in ("head", "tail", "tension", "tracker_1", "tracker_2",
                "driven_sprocket", "drive_sprocket"):
        if key in layout:
            x, y, r, _kind = layout[key]
            entries.append((key, x, y, r))
    for key in ("return_rollers_before", "return_rollers_after",
                "carrying_idlers"):
        for i, (x, y, r, _k) in enumerate(layout.get(key, [])):
            entries.append((f"{key}_{i}", x, y, r))
    return entries


# ═══════════════════════════════════════════════════════════════════════
#  Pretreatment Page
# ═══════════════════════════════════════════════════════════════════════

class PretreatmentPage(QWidget):
    """Full-page GP-15 RF pretreatment simulation and visualization.

    Matches examples/simulate_and_visualize.py with live 3D visualization:
    - Animated rollers (rotating)
    - Animated belt (scrolling pattern)
    - Particle point cloud (moving with temperature coloring)
    - Temperature field overlay (updating)

    After simulation, switches to a dedicated results page with detailed
    KPIs, 3x3 plot grid, outfeed cross-section, and export toolbar.

    Signals:
        simulation_started: emitted when a run begins.
        simulation_finished(dict): emitted with results when a run ends.
    """

    simulation_started = Signal()
    simulation_finished = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Live simulation state (main thread, not worker thread)
        self._sim: Optional[Any] = None  # GP15Simulator
        self._recipe: Optional[Any] = None
        self._material: Optional[Any] = None
        self._duration_s: float = 0.0
        self._running: bool = False
        self._winding_down: bool = False
        self._render_timer: Optional[QTimer] = None
        self._t0_wall: float = 0.0
        self._assembly_info: Optional[Dict[str, Any]] = None

        # Animated mesh actors (for live updates)
        self._rollers_pd: Optional[Any] = None
        self._rollers_base: Optional[np.ndarray] = None
        self._roller_vert_masks: Dict[str, Any] = {}
        self._belt_pd: Optional[Any] = None
        self._belt_total_len: float = 0.0
        self._belt_arc_per_vert: Optional[np.ndarray] = None
        self._belt_band_len: float = 0.0
        self._particle_cloud: Optional[Any] = None
        self._mat_grid: Optional[Any] = None
        self._mat_indices: Optional[np.ndarray] = None

        self._results: Optional[Dict[str, Any]] = None
        self._build_ui()

    # ──────────────────────────────────────────────────────────────
    #  UI Construction
    # ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._view_stack = QStackedWidget()
        root.addWidget(self._view_stack)

        # Page 0: Simulation View
        sim_view = self._build_simulation_view()
        self._view_stack.addWidget(sim_view)

        # Page 1: Results View
        results_view = self._build_results_view()
        self._view_stack.addWidget(results_view)

        self._view_stack.setCurrentIndex(0)

    # ──────────────────────────────────────────────────────────────
    #  Page 0: Simulation View
    # ──────────────────────────────────────────────────────────────

    def _build_simulation_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content: 3D viewport + control panel side-by-side
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._viewport_widget = self._build_viewport()
        top_splitter.addWidget(self._viewport_widget)

        ctrl_widget = self._build_control_panel()
        top_splitter.addWidget(ctrl_widget)

        top_splitter.setSizes([700, 350])
        layout.addWidget(top_splitter, 1)

        # Bottom bar: status + "View Full Results"
        bottom_bar = self._build_sim_bottom_bar()
        layout.addWidget(bottom_bar)

        return page

    def _build_sim_bottom_bar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-top: 1px solid {COLORS.BORDER};
            }}
        """)
        bar.setFixedHeight(48)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)

        self._results_status_label = QLabel("Run a simulation to generate results")
        self._results_status_label.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;"
            " border: none; background: transparent;"
        )
        layout.addWidget(self._results_status_label)
        layout.addStretch()

        self._view_results_btn = QPushButton("View Full Results")
        self._view_results_btn.setMinimumHeight(32)
        self._view_results_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.ACCENT};
                color: {COLORS.TEXT_INVERSE};
                border: none; border-radius: 6px;
                padding: 6px 20px; font-size: 10pt; font-weight: 600;
            }}
            QPushButton:hover {{ background: {COLORS.ACCENT_HOVER}; }}
            QPushButton:disabled {{ background: {COLORS.BG_SURFACE}; color: {COLORS.TEXT_DISABLED}; }}
        """)
        self._view_results_btn.setEnabled(False)
        self._view_results_btn.clicked.connect(self._show_results_view)
        layout.addWidget(self._view_results_btn)

        return bar

    # ──────────────────────────────────────────────────────────────
    #  3D Viewport
    # ──────────────────────────────────────────────────────────────

    def _build_viewport(self) -> QWidget:
        """Build the 3D viewport using pyvistaqt or a placeholder."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_PYVISTA:
            self._plotter = QtInteractor(container)
            self._plotter.set_background(COLORS.BG_DARKEST)
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.add_axes()
            layout.addWidget(self._plotter.interactor)

            self._plotter.add_text(
                "GP-15 RF Dielectric Heating Machine\n"
                "Configure settings and click Run to start",
                position="upper_left",
                font_size=10,
                color=COLORS.TEXT_MUTED,
                name="placeholder_text",
            )
        else:
            self._plotter = None
            lbl = QLabel(
                "PyVista not available.\n"
                "Install with: pip install pyvista pyvistaqt\n\n"
                "Simulation will run without 3D visualization."
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {COLORS.TEXT_MUTED}; font-size: 11pt; padding: 40px;"
            )
            layout.addWidget(lbl)

        return container

    # ──────────────────────────────────────────────────────────────
    #  Control Panel (right side of simulation view)
    # ──────────────────────────────────────────────────────────────

    def _build_control_panel(self) -> QWidget:
        container = QWidget()
        container.setMinimumWidth(300)
        container.setMaximumWidth(420)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title
        title = QLabel("GP-15 RF Pretreatment")
        title.setStyleSheet(
            f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};"
        )
        layout.addWidget(title)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setProperty("cssClass", "success")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("cssClass", "danger")
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setFixedHeight(16)
        layout.addWidget(self._progress)

        # KPI cards
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(4)

        self._card_time = _StatCard("Sim Time", COLORS.ACCENT)
        self._card_T = _StatCard("Temperature", COLORS.DANGER)
        self._card_M = _StatCard("Moisture", COLORS.INFO)
        self._card_P = _StatCard("RF Power", COLORS.WARNING)
        self._card_Ia = _StatCard("Anode Current", COLORS.SUCCESS)
        self._card_gap = _StatCard("Electrode Gap", COLORS.TEXT_SECONDARY)

        kpi_grid.addWidget(self._card_time, 0, 0)
        kpi_grid.addWidget(self._card_T, 0, 1)
        kpi_grid.addWidget(self._card_M, 1, 0)
        kpi_grid.addWidget(self._card_P, 1, 1)
        kpi_grid.addWidget(self._card_Ia, 2, 0)
        kpi_grid.addWidget(self._card_gap, 2, 1)

        layout.addLayout(kpi_grid)

        # Settings + Log tabs
        ctrl_tabs = QTabWidget()
        ctrl_tabs.addTab(_scrollable(self._build_settings_widget()), "Settings")
        ctrl_tabs.addTab(self._build_log_widget(), "Log")
        layout.addWidget(ctrl_tabs, 1)

        return container

    def _build_settings_widget(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        _M = (8, 12, 8, 8)

        # ── Material ─────────────────────────────────────────────
        g = QGroupBox("Material")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self._material_combo = QComboBox()
        self._material_combo.addItems(["yellow_pea", "faba_bean", "oat"])
        f.addRow("Preset:", self._material_combo)

        self._moisture_spin = QDoubleSpinBox()
        self._moisture_spin.setRange(0.01, 0.25)
        self._moisture_spin.setValue(0.10)
        self._moisture_spin.setDecimals(3)
        self._moisture_spin.setSingleStep(0.005)
        f.addRow("Inlet moisture (wb):", self._moisture_spin)

        self._temp_init_spin = QDoubleSpinBox()
        self._temp_init_spin.setRange(5.0, 40.0)
        self._temp_init_spin.setValue(17.6)
        self._temp_init_spin.setDecimals(1)
        self._temp_init_spin.setSuffix(" \u00b0C")
        f.addRow("Initial temp:", self._temp_init_spin)

        self._bed_depth_spin = QDoubleSpinBox()
        self._bed_depth_spin.setRange(10, 100)
        self._bed_depth_spin.setValue(25)
        self._bed_depth_spin.setSuffix(" mm")
        f.addRow("Bed depth:", self._bed_depth_spin)

        layout.addWidget(g)

        # ── Recipe ───────────────────────────────────────────────
        g = QGroupBox("Recipe (GP-15 HMI)")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(20, 300)
        self._gap_spin.setValue(75)
        self._gap_spin.setSuffix(" mm")
        f.addRow("Electrode gap:", self._gap_spin)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.05, 2.0)
        self._speed_spin.setValue(0.20)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setSingleStep(0.05)
        self._speed_spin.setSuffix(" m/min")
        f.addRow("Belt speed:", self._speed_spin)

        self._mass_spin = QDoubleSpinBox()
        self._mass_spin.setRange(1, 500)
        self._mass_spin.setValue(61.0)
        self._mass_spin.setDecimals(1)
        self._mass_spin.setSuffix(" kg")
        f.addRow("Run mass:", self._mass_spin)

        self._fan_spin = QDoubleSpinBox()
        self._fan_spin.setRange(5, 60)
        self._fan_spin.setValue(35)
        self._fan_spin.setSuffix(" Hz")
        f.addRow("Extraction fan:", self._fan_spin)

        self._mrh_spin = QDoubleSpinBox()
        self._mrh_spin.setRange(0.5, 3.0)
        self._mrh_spin.setValue(1.7)
        self._mrh_spin.setDecimals(2)
        self._mrh_spin.setSuffix(" A")
        f.addRow("MRH (overcurrent):", self._mrh_spin)

        self._heater_check = QCheckBox("Both banks on")
        self._heater_check.setChecked(True)
        f.addRow("Heaters:", self._heater_check)

        layout.addWidget(g)

        # ── Simulation Time ──────────────────────────────────────
        g = QGroupBox("Simulation Time")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0, 7200)
        self._duration_spin.setValue(0)
        self._duration_spin.setSuffix(" s")
        self._duration_spin.setToolTip(
            "0 = auto-compute from run mass and belt speed"
        )
        f.addRow("Duration:", self._duration_spin)

        hint = QLabel("0 = auto-compute from mass & belt speed")
        hint.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
        )
        f.addRow("", hint)

        layout.addWidget(g)

        # ── Physics ──────────────────────────────────────────────
        g = QGroupBox("Physics Options")
        f = QFormLayout(g)
        f.setContentsMargins(*_M)

        self._tvd_check = QCheckBox("Van Leer TVD advection")
        self._tvd_check.setChecked(True)
        f.addRow(self._tvd_check)

        self._ctrl_check = QCheckBox("PLC controller (MRH/MRL)")
        self._ctrl_check.setChecked(True)
        f.addRow(self._ctrl_check)

        self._corr_check = QCheckBox("Fringe + perforation corrections")
        self._corr_check.setChecked(False)
        f.addRow(self._corr_check)

        self._eff_spin = QDoubleSpinBox()
        self._eff_spin.setRange(0.10, 1.0)
        self._eff_spin.setValue(0.56)
        self._eff_spin.setDecimals(2)
        self._eff_spin.setSingleStep(0.05)
        f.addRow("Oscillator eff:", self._eff_spin)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda"])
        f.addRow("Device:", self._device_combo)

        layout.addWidget(g)
        layout.addStretch()
        return w

    def _build_log_widget(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(4)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("Cascadia Code, Consolas, monospace", 9))
        self._log_text.setStyleSheet(f"""
            QTextEdit {{
                background: {COLORS.BG_DARKEST};
                color: {COLORS.TEXT_SECONDARY};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self._log_text)

        clear_btn = QPushButton("Clear")
        clear_btn.setProperty("cssClass", "ghost")
        clear_btn.clicked.connect(self._log_text.clear)
        layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

        return w

    # ──────────────────────────────────────────────────────────────
    #  Page 1: Results View
    # ──────────────────────────────────────────────────────────────

    def _build_results_view(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header banner with back button + export toolbar
        header = self._build_results_header()
        outer.addWidget(header)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)

        # 4x2 KPI result cards
        kpi_section = self._build_results_kpi_section()
        content_layout.addWidget(kpi_section)

        # Process Desirability scoring
        desirability_section = self._build_desirability_section()
        content_layout.addWidget(desirability_section)

        # Matplotlib sections
        if _HAS_MATPLOTLIB:
            plots_section = self._build_results_plots_section()
            content_layout.addWidget(plots_section)

            particle_section = self._build_results_particle_section()
            content_layout.addWidget(particle_section)

            outfeed_section = self._build_results_outfeed_section()
            content_layout.addWidget(outfeed_section)
        else:
            lbl = QLabel("Matplotlib not available. Install with: pip install matplotlib")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11pt; padding: 40px;")
            content_layout.addWidget(lbl)

        content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        return page

    def _build_results_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border-bottom: 1px solid {COLORS.BORDER};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)

        back_btn = QPushButton("Back to Simulation")
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                color: {COLORS.TEXT_PRIMARY};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px; padding: 6px 16px;
                font-size: 9pt;
            }}
            QPushButton:hover {{ background: {COLORS.BG_HOVER}; }}
        """)
        back_btn.clicked.connect(self._show_simulation_view)
        layout.addWidget(back_btn)

        self._results_title_label = QLabel("GP-15 Simulation Results")
        self._results_title_label.setStyleSheet(
            f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(self._results_title_label, 1, Qt.AlignmentFlag.AlignCenter)

        # Export buttons
        _ghost = f"""
            QPushButton {{
                background: transparent;
                color: {COLORS.TEXT_SECONDARY};
                border: 1px solid {COLORS.BORDER};
                border-radius: 4px; padding: 4px 12px;
                font-size: 9pt;
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
                border-color: {COLORS.TEXT_MUTED};
            }}
        """

        for label, slot in [
            ("Export CSV", self._export_csv),
            ("Export JSON", self._export_json),
            ("Export PNG", self._export_plots),
            ("Export PDF", self._export_pdf_report),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(_ghost)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return header

    def _build_results_kpi_section(self) -> QWidget:
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(16, 12, 16, 12)

        section_title = QLabel("Key Performance Indicators")
        section_title.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_SECONDARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(section_title)

        grid = QGridLayout()
        grid.setSpacing(8)

        self._rc_moisture = _StatCard("Outfeed Moisture", COLORS.INFO)
        self._rc_temp = _StatCard("Outfeed Temp (Sensor)", COLORS.DANGER)
        self._rc_max_temp = _StatCard("Max Temperature", COLORS.WARNING)
        self._rc_cv = _StatCard("Moisture Uniformity (CV)", COLORS.ACCENT)
        self._rc_energy = _StatCard("RF Energy Consumed", COLORS.SUCCESS)
        self._rc_specific = _StatCard("Specific Energy (kWh/kg water)", COLORS.TEXT_PRIMARY)
        self._rc_throughput = _StatCard("Throughput", COLORS.CAT_FEED)
        self._rc_wall = _StatCard("Wall-Clock Time", COLORS.TEXT_SECONDARY)
        self._rc_protein = _StatCard("Protein Quality (Native Loss)", COLORS.WARNING)
        self._rc_mass_balance = _StatCard("Mass Balance", COLORS.INFO)
        self._rc_final_gap = _StatCard("Final Electrode Gap", COLORS.SUCCESS)
        self._rc_sim_speed = _StatCard("Simulation Speed", COLORS.TEXT_SECONDARY)

        grid.addWidget(self._rc_moisture, 0, 0)
        grid.addWidget(self._rc_temp, 0, 1)
        grid.addWidget(self._rc_max_temp, 0, 2)
        grid.addWidget(self._rc_cv, 0, 3)
        grid.addWidget(self._rc_energy, 1, 0)
        grid.addWidget(self._rc_specific, 1, 1)
        grid.addWidget(self._rc_throughput, 1, 2)
        grid.addWidget(self._rc_wall, 1, 3)
        grid.addWidget(self._rc_protein, 2, 0)
        grid.addWidget(self._rc_mass_balance, 2, 1)
        grid.addWidget(self._rc_final_gap, 2, 2)
        grid.addWidget(self._rc_sim_speed, 2, 3)

        layout.addLayout(grid)
        return section

    # ── Desirability Section ───────────────────────────────────────────
    def _build_desirability_section(self) -> QWidget:
        """Build the Process Desirability scoring section for results view."""
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(16, 12, 16, 12)

        section_title = QLabel("Process Desirability  (Protein Separation & Flavour)")
        section_title.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_SECONDARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(section_title)

        # Row: overall score (large) + 5 dimension cards
        row = QHBoxLayout()
        row.setSpacing(10)

        # Overall score — large prominent card
        self._ds_overall = QFrame()
        self._ds_overall.setFixedWidth(160)
        self._ds_overall.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        ov_layout = QVBoxLayout(self._ds_overall)
        ov_layout.setContentsMargins(12, 10, 12, 10)
        ov_layout.setSpacing(2)

        self._ds_overall_value = QLabel("--")
        self._ds_overall_value.setStyleSheet(
            f"font-size: 22pt; font-weight: 800; color: {COLORS.ACCENT};"
            " border: none; background: transparent;"
        )
        self._ds_overall_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov_layout.addWidget(self._ds_overall_value)

        ov_subtitle = QLabel("Overall Score")
        ov_subtitle.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        ov_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov_layout.addWidget(ov_subtitle)
        row.addWidget(self._ds_overall)

        # 5 dimension score cards with progress bars
        self._ds_cards: Dict[str, tuple] = {}  # key → (value_label, bar_widget)
        dim_defs = [
            ("d_thermal",     "Thermal\nTreatment",      COLORS.DANGER),
            ("d_flavour",     "Flavour\nImprovement",    COLORS.WARNING),
            ("d_protein",     "Protein\nPreservation",   COLORS.SUCCESS),
            ("d_moisture",    "Moisture\nRetention",     COLORS.INFO),
            ("d_energy",      "Energy\nEfficiency",      COLORS.CAT_FEED),
        ]

        for key, label_text, color in dim_defs:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS.BG_DARK};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 6px;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)
            card_layout.setSpacing(3)

            val_label = QLabel("--")
            val_label.setStyleSheet(
                f"font-size: 13pt; font-weight: 700; color: {color};"
                " border: none; background: transparent;"
            )
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(val_label)

            # Score bar (QProgressBar styled as colored fill)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {COLORS.BG_DARKEST};
                    border: none;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 3px;
                }}
            """)
            card_layout.addWidget(bar)

            title_label = QLabel(label_text)
            title_label.setStyleSheet(
                f"font-size: 7pt; color: {COLORS.TEXT_MUTED};"
                " border: none; background: transparent;"
            )
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(title_label)

            self._ds_cards[key] = (val_label, bar)
            row.addWidget(card, 1)

        layout.addLayout(row)
        return section

    def _update_desirability(self, results: dict):
        """Compute and display desirability scores from simulation results."""
        outlet = results.get("outlet")
        if outlet is None:
            return

        # Determine profile from material name
        material_name = getattr(self._material, "name", "yellow_pea") if self._material else "yellow_pea"
        profile = PROFILES.get(material_name)
        if profile is None:
            profile = PROFILES["yellow_pea"]

        # Get run parameters
        initial_m = results.get("initial_moisture", 0.10)
        run_mass = results.get("run_mass_kg", 0.0)
        result_obj = results.get("result")
        energy_kwh = result_obj.energy_consumed_kwh if result_obj else 0.0

        dr = score_desirability(
            outfeed_temperature_c=outlet.sensor_temperature_c,  # Use sensor-comparable temp
            max_temperature_c=outlet.max_temperature_c,
            outfeed_moisture_wb=outlet.avg_moisture_wb,
            initial_moisture_wb=initial_m,
            energy_kwh=energy_kwh,
            run_mass_kg=run_mass,
            profile=profile,
        )

        # Update overall score with dynamic color
        score_10 = dr.overall_10
        if score_10 >= 7.0:
            ov_color = COLORS.SUCCESS
        elif score_10 >= 4.0:
            ov_color = COLORS.WARNING
        else:
            ov_color = COLORS.DANGER
        self._ds_overall_value.setText(f"{score_10:.1f}")
        self._ds_overall_value.setStyleSheet(
            f"font-size: 22pt; font-weight: 800; color: {ov_color};"
            " border: none; background: transparent;"
        )

        # Update individual dimension cards
        dim_scores = {
            "d_thermal": dr.d_thermal,
            "d_flavour": dr.d_flavour,
            "d_protein": dr.d_protein,
            "d_moisture": dr.d_moisture,
            "d_energy": dr.d_energy,
        }
        for key, score in dim_scores.items():
            if key in self._ds_cards:
                val_label, bar = self._ds_cards[key]
                pct = int(round(score * 100))
                val_label.setText(f"{pct}%")
                bar.setValue(pct)

    def _build_results_plots_section(self) -> QWidget:
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 8, 12, 8)

        section_title = QLabel("Simulation Time-Series")
        section_title.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_SECONDARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(section_title)

        self._plot_figure = Figure(figsize=(16, 11), dpi=100)
        self._plot_figure.patch.set_facecolor(COLORS.BG_DARK)
        self._plot_canvas = FigureCanvas(self._plot_figure)
        self._plot_canvas.setMinimumHeight(750)
        self._plot_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._plot_canvas)

        return section

    def _build_results_outfeed_section(self) -> QWidget:
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 8, 12, 8)

        section_title = QLabel("Outfeed Cross-Section \u2014 Pipeline Output to Milling")
        section_title.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_SECONDARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(section_title)

        self._outfeed_figure = Figure(figsize=(14, 5), dpi=100)
        self._outfeed_figure.patch.set_facecolor(COLORS.BG_DARK)
        self._outfeed_canvas = FigureCanvas(self._outfeed_figure)
        self._outfeed_canvas.setMinimumHeight(350)
        self._outfeed_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._outfeed_canvas)

        return section

    def _build_results_particle_section(self) -> QWidget:
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_BASE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 8, 12, 8)

        section_title = QLabel("Particle System Analysis")
        section_title.setStyleSheet(
            f"font-size: 10pt; font-weight: 600; color: {COLORS.TEXT_SECONDARY};"
            " border: none; background: transparent;"
        )
        layout.addWidget(section_title)

        self._particle_figure = Figure(figsize=(16, 9), dpi=100)
        self._particle_figure.patch.set_facecolor(COLORS.BG_DARK)
        self._particle_canvas = FigureCanvas(self._particle_figure)
        self._particle_canvas.setMinimumHeight(600)
        self._particle_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._particle_canvas)

        return section

    # ──────────────────────────────────────────────────────────────
    #  View Navigation
    # ──────────────────────────────────────────────────────────────

    def _show_results_view(self):
        """Switch to the full-page results view."""
        self._view_stack.setCurrentIndex(1)

    def _show_simulation_view(self):
        """Switch back to the simulation view."""
        self._view_stack.setCurrentIndex(0)

    # ──────────────────────────────────────────────────────────────
    #  Simulation Lifecycle
    # ──────────────────────────────────────────────────────────────

    def run_simulation(self):
        """Start the GP-15 simulation (called by MainWindow F5 when in Pretreatment mode)."""
        if not self._running:
            self._on_run()

    def _on_run(self):
        """Start the GP-15 simulation."""
        from airclassifier.pretreatment import (
            MachineConfig,
            MaterialProperties,
            Recipe,
        )

        # Reset results state
        self._view_results_btn.setEnabled(False)
        self._results_status_label.setText("Simulation running...")
        self._results_status_label.setStyleSheet(
            f"color: {COLORS.WARNING}; font-size: 9pt;"
            " border: none; background: transparent;"
        )
        self._show_simulation_view()

        config = MachineConfig()

        # Material from settings — use MaterialProperties() directly so
        # k_evap and oscillator_coupling_factor come from the calibration
        # store (same as examples/simulate_and_visualize.py).
        mat = MaterialProperties(
            name=self._material_combo.currentText(),
            initial_moisture_wb=self._moisture_spin.value(),
            initial_temperature_c=self._temp_init_spin.value(),
            bed_depth_m=self._bed_depth_spin.value() / 1000.0,
        )

        # Recipe from settings
        recipe = Recipe(
            name="gui_run",
            recipe_number=1,
            electrode_gap_mm=self._gap_spin.value(),
            belt_speed_m_per_min=self._speed_spin.value(),
            run_mass_kg=self._mass_spin.value(),
            extraction_fan_hz=self._fan_spin.value(),
            mrh_amps=self._mrh_spin.value(),
            heater_bank_1_on=self._heater_check.isChecked(),
            heater_bank_2_on=self._heater_check.isChecked(),
        )

        # Duration
        duration = self._duration_spin.value()
        if duration <= 0:
            from airclassifier.pretreatment import GP15Simulator
            temp_sim = GP15Simulator(config=config, material=mat, device="cpu")
            temp_sim.load_recipe(recipe)
            duration = temp_sim.compute_run_duration(recipe)
            if duration <= 0:
                duration = 120.0
            self._log(f"Auto duration: {duration:.0f} s ({duration/60:.1f} min)")

        # Physics settings
        device = self._device_combo.currentText()
        if device == "auto":
            device = None

        settings = {
            "tvd": self._tvd_check.isChecked(),
            "fdm": True,
            "controller": self._ctrl_check.isChecked(),
            "corrections": self._corr_check.isChecked(),
            "efficiency": self._eff_spin.value(),
            "device": device,
        }

        # UI state
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)
        self._card_time.set_value("0.0 s")

        self._log("=" * 50)
        self._log("GP-15 RF DIELECTRIC HEATING SIMULATION")
        self._log("=" * 50)
        self._log(f"  Material:  {self._material_combo.currentText()}")
        self._log(f"  Mass:      {self._mass_spin.value():.1f} kg")
        self._log(
            f"  Gap:       {self._gap_spin.value():.0f} mm  |  "
            f"Bed: {self._bed_depth_spin.value():.0f} mm"
        )
        self._log(f"  Belt:      {self._speed_spin.value():.2f} m/min")
        self._log(f"  Duration:  {duration:.0f} s ({duration/60:.1f} min)")
        self._log("=" * 50)

        # Create simulator in main thread (for live visualization)
        try:
            from airclassifier.pretreatment.simulator import GP15Simulator
            from airclassifier.pretreatment.calibration_store import get_calibration_defaults

            self._log(
                f"GP-15 initialising | gap={recipe.electrode_gap_mm:.0f} mm "
                f"| speed={recipe.belt_speed_m_per_min:.2f} m/min "
                f"| bed={mat.bed_depth_m * 1000:.0f} mm"
            )

            self._sim = GP15Simulator(
                config=config,
                material=mat,
                enable_controller=settings.get("controller", True),
                enable_corrections=settings.get("corrections", False),
                use_tvd=settings.get("tvd", True),
                use_fdm=settings.get("fdm", False),
                oscillator_efficiency=settings.get("efficiency", 0.56),
                device=settings.get("device"),
            )
            self._sim.load_recipe(recipe)
            self._recipe = recipe
            self._material = mat
            self._duration_s = duration

            # Capture assembly info for plots
            try:
                self._assembly_info = self._sim.assembly.get_assembly_info()
            except Exception:
                self._assembly_info = None

            # Apply calibration gap rate
            gap_rate = get_calibration_defaults()[2]
            self._sim._sim.update_parameters(gap_adjust_rate=gap_rate)

            dt = self._sim._sim.compute_stable_dt(recipe)
            total_steps = max(1, int(duration / dt))
            self._log(
                f"Running {total_steps:,} steps "
                f"(dt={dt:.4f} s, total={duration:.0f} s)..."
            )

            # Initialize live visualization
            if not self._init_live_visualization(self._sim):
                self._log("Warning: Live visualization initialization failed")

            # Start render loop
            self._running = True
            self._winding_down = False
            self._t0_wall = time.perf_counter()
            self._render_timer = QTimer(self)
            self._render_timer.timeout.connect(self._render_tick)
            self._render_timer.start(50)  # ~20 fps

            self.simulation_started.emit()

        except Exception as e:
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._running = False

    def _on_stop(self):
        """Stop the running simulation."""
        self._running = False
        self._winding_down = False
        if self._render_timer:
            self._render_timer.stop()
            self._render_timer = None
        if self._sim is not None:
            self._finalize_results()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log("Stopped by user.")

    @Slot(int, float, dict)
    def _on_progress(self, pct: int, time_s: float, kpis: dict):
        """Handle progress updates."""
        self._progress.setValue(pct)
        self._card_time.set_value(f"{time_s:.1f} s")

        if kpis:
            if "T_mean_c" in kpis:
                self._card_T.set_value(f"{kpis['T_mean_c']:.1f} \u00b0C")
            if "M_mean_wb" in kpis:
                self._card_M.set_value(f"{kpis['M_mean_wb'] * 100:.2f} %")
            if "rf_power_kw" in kpis:
                self._card_P.set_value(f"{kpis['rf_power_kw']:.2f} kW")
            if "anode_current_a" in kpis:
                self._card_Ia.set_value(f"{kpis['anode_current_a']:.2f} A")
            if "electrode_gap_mm" in kpis:
                self._card_gap.set_value(f"{kpis['electrode_gap_mm']:.0f} mm")

    def _log(self, msg: str):
        """Append a message to the log view."""
        self._log_text.append(msg)

    # ──────────────────────────────────────────────────────────────
    #  Live 3D Visualization
    # ──────────────────────────────────────────────────────────────

    def _init_live_visualization(self, sim):
        """Initialize live 3D visualization with animated components.

        Matches _run_live_3d from examples/simulate_and_visualize.py.
        """
        if not _HAS_PYVISTA or self._plotter is None:
            return False

        try:
            import pyvista as pv
            import math
            from airclassifier.pretreatment.geometry.assembly import COMPONENT_COLORS

            sim._ensure_initialized()
            meshes = sim.get_mesh()
            particle_sys = sim.particles
            conv_params = sim.assembly.conveyor.params
            roller_layout = sim.assembly.conveyor._roller_layout

            # Clear viewport
            self._plotter.clear()

            xray_opacities = {
                "conveyor_frame": 0.08, "oven_chamber": 0.06,
                "rollers": 0.60, "belt": 0.40,
                "upper_electrode": 0.30, "lower_electrode": 0.25,
                "infeed_hopper": 0.70, "infeed_tunnel": 0.20,
                "outfeed_tunnel": 0.20, "collection_bin": 0.55,
                "emu_housing": 0.10, "generator": 0.20, "rf_feed": 0.80,
            }

            animated_names = {"rollers", "belt", "fields"}

            # Add STATIC machine geometry
            for name, item in meshes.items():
                if name in animated_names:
                    continue
                if not isinstance(item, tuple) or len(item) != 3:
                    continue
                v, t, meta = item
                style = COMPONENT_COLORS.get(name, {})
                color = style.get("color", "#888888")
                label = style.get("label", name)
                opacity = xray_opacities.get(name, style.get("opacity", 0.8))
                if opacity < 0.01:
                    continue
                pd = _mesh_to_polydata(v, t)
                self._plotter.add_mesh(pd, color=color, opacity=opacity,
                                       smooth_shading=True, label=label)

            # Add ANIMATED rollers
            rollers_v, rollers_t, _ = meshes["rollers"]
            self._rollers_base = rollers_v.copy()
            self._rollers_pd = _mesh_to_polydata(rollers_v, rollers_t)
            self._plotter.add_mesh(self._rollers_pd, color="#808090", opacity=0.85,
                                   smooth_shading=True, label="Rollers", name="rollers")

            roller_entries = _collect_roller_entries(roller_layout)
            roller_z_max = conv_params.belt_z1 + 0.05
            roller_z_min = conv_params.belt_z0 - 0.05
            z_in_range = ((self._rollers_base[:, 2] >= roller_z_min) &
                          (self._rollers_base[:, 2] <= roller_z_max))

            self._roller_vert_masks = {}
            for rname, cx, cy, r in roller_entries:
                if "sprocket" in rname:
                    continue
                dx_r = self._rollers_base[:, 0] - cx
                dy_r = self._rollers_base[:, 1] - cy
                dist_xy = np.sqrt(dx_r * dx_r + dy_r * dy_r)
                mask = (dist_xy < r * 1.6) & z_in_range
                if mask.any():
                    self._roller_vert_masks[rname] = (mask, cx, cy, r)

            # Add ANIMATED belt
            belt_v, belt_t, _ = meshes["belt"]
            self._belt_pd = _mesh_to_polydata(belt_v, belt_t)

            belt_path_xy = belt_v[0::4, :2]
            _seg = np.diff(belt_path_xy, axis=0)
            _seg_lens = np.sqrt(_seg[:, 0] ** 2 + _seg[:, 1] ** 2)
            belt_arc = np.concatenate([[0.0], np.cumsum(_seg_lens)])
            self._belt_total_len = belt_arc[-1]
            self._belt_arc_per_vert = np.repeat(belt_arc, 4)

            head_circ = 2.0 * math.pi * conv_params.head_roller_radius_m
            n_belt_bands = max(1, round(self._belt_total_len / head_circ))
            self._belt_band_len = self._belt_total_len / n_belt_bands

            self._belt_pd.point_data["motion"] = (
                0.5 + 0.5 * np.cos(2.0 * math.pi * self._belt_arc_per_vert / self._belt_band_len)
            ).astype(np.float32)

            self._plotter.add_mesh(
                self._belt_pd, scalars="motion",
                cmap=["#3458a5", "#5a7fd4"],
                clim=[0, 1], opacity=0.92,
                show_edges=True, edge_color="#2a4a9a", line_width=0.4,
                smooth_shading=True, name="belt",
                show_scalar_bar=False, label="Belt (PTFE)",
            )

            # Add particle point cloud
            particle_cloud = pv.PolyData(particle_sys.pos.copy())
            particle_cloud["Temperature"] = particle_sys.temperature.copy()
            T_ambient = self._material.initial_temperature_c
            self._particle_cloud = particle_cloud
            self._plotter.add_mesh(
                particle_cloud, scalars="Temperature",
                cmap="hot", clim=[T_ambient, T_ambient + 15],
                point_size=5.0, render_points_as_spheres=True,
                opacity=0.90, show_scalar_bar=False,
                name="particles", label="Material Particles",
            )

            # Build rectilinear grid for temperature field
            nx, ny, nz = sim.grid_shape
            dx, dy, dz = sim.cell_sizes
            x0, y0, z0 = sim.get_field_world_origin()

            x_coords = np.linspace(x0, x0 + nx * dx, nx + 1)
            y_coords = np.linspace(y0, y0 + ny * dy, ny + 1)
            z_coords = np.linspace(z0, z0 + nz * dz, nz + 1)

            field_grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)
            mask_flat = sim.material_mask.flatten(order="F")
            field_grid.cell_data["Temperature"] = sim.temperature_field.flatten(order="F")
            field_grid.cell_data["zone"] = mask_flat
            self._mat_indices = np.where(mask_flat == 1)[0]

            self._mat_grid = field_grid.threshold(value=1, scalars="zone")
            self._plotter.add_mesh(
                self._mat_grid, scalars="Temperature",
                cmap="hot", clim=[T_ambient, T_ambient + 15],
                opacity=0.90, show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Temperature [\u00b0C]",
                    "position_x": 0.82, "width": 0.12,
                },
                label="Temperature Field",
            )

            # Legend, axes, camera
            self._plotter.add_legend(loc="upper left", bcolor=(0.1, 0.1, 0.15, 0.8))
            self._plotter.add_title("GP-15 Live  |  Initializing...", font_size=10)
            self._plotter.add_axes()
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.reset_camera()
            self._plotter.camera.azimuth = 125
            self._plotter.camera.elevation = 18
            self._plotter.camera.zoom(1.1)

            return True
        except Exception as e:
            self._log(f"Live visualization init error: {e}\n{traceback.format_exc()}")
            return False

    def _update_live_visualization(self, sim):
        """Update animated components for one frame."""
        if not _HAS_PYVISTA or self._plotter is None:
            return

        try:
            import math
            from airclassifier.pretreatment.kernels.transport import rotate_mesh_around_z_axis

            conv_ctrl = sim.conveyor
            particle_sys = sim.particles

            # Update rollers
            if self._rollers_pd is not None and self._rollers_base is not None:
                animated_verts = self._rollers_base.copy()
                for rname, (mask, cx, cy, r) in self._roller_vert_masks.items():
                    angle = -conv_ctrl.roller_angle(r)
                    animated_verts[mask] = rotate_mesh_around_z_axis(
                        self._rollers_base[mask], cx, cy, angle,
                    )
                self._rollers_pd.points = animated_verts

            # Update belt scroll
            if self._belt_pd is not None:
                _phase = conv_ctrl.state.belt_position_m % self._belt_total_len
                _shifted = (self._belt_arc_per_vert - _phase + self._belt_total_len) % self._belt_total_len
                self._belt_pd.point_data["motion"] = (
                    0.5 + 0.5 * np.cos(2.0 * math.pi * _shifted / self._belt_band_len)
                ).astype(np.float32)

            # Update particles
            if self._particle_cloud is not None:
                updated_pos = particle_sys.pos.copy()
                dead = (particle_sys.state == particle_sys._STATE_DEAD)
                updated_pos[dead] = [0.0, -100.0, 0.0]
                self._particle_cloud.points = updated_pos
                self._particle_cloud.point_data["Temperature"] = particle_sys.temperature.copy()

            # Update temperature field
            if self._mat_grid is not None and self._mat_indices is not None:
                T_flat = sim.temperature_field.flatten(order="F")
                self._mat_grid.cell_data["Temperature"] = T_flat[self._mat_indices]

            # Update title
            hist = sim.history
            if hist:
                last = hist[-1]
                hopper_n = particle_sys.hopper_count
                riding_n = particle_sys.riding_count
                title = (f"GP-15 Live  |  t={sim.sim_time:.0f}/{self._duration_s:.0f} s  |  "
                         f"T_out={last.T_outfeed_c:.1f} C  |  "
                         f"M_out={last.M_outfeed_wb:.1%}  |  "
                         f"P={last.rf_power_kw:.1f} kW  |  "
                         f"Hopper:{hopper_n}  Belt:{riding_n}")
            else:
                title = f"GP-15 Live  |  t={sim.sim_time:.0f}/{self._duration_s:.0f} s"
            self._plotter.add_title(title, font_size=10)

            self._plotter.render()
        except Exception:
            pass  # Silently ignore visualization errors

    def _render_tick(self):
        """Render loop tick: step simulator and update visualization."""
        if not self._running or self._sim is None:
            return

        # ── Wind-down phase: clear the belt (no physics, just transport)
        if self._winding_down:
            try:
                particle_sys = self._sim.particles
                conv_ctrl = self._sim.conveyor
                v_belt = conv_ctrl.state.belt_speed_m_per_s
                dx, _, _ = self._sim.cell_sizes
                dt_wind = 0.8 * dx / v_belt if v_belt > 0 else 1.0

                # Multiple sub-steps per frame for fast clearing
                for _ in range(200):
                    particle_sys.step(
                        dt_sim=dt_wind,
                        belt_speed_m_per_s=v_belt,
                    )
                    if particle_sys.riding_count == 0 and particle_sys.hopper_count == 0:
                        break
                    conv_ctrl.step(dt_wind)

                self._update_live_visualization(self._sim)

                if particle_sys.riding_count == 0 and particle_sys.hopper_count == 0:
                    self._winding_down = False
                    self._finalize_results()
            except Exception as e:
                self._log(f"Wind-down error: {e}\n{traceback.format_exc()}")
                self._winding_down = False
                self._finalize_results()
            return

        # ── Normal simulation phase
        try:
            # Smooth adaptive pacing (matches example)
            v_belt_init = self._recipe.belt_speed_m_per_min / 60.0
            residence_s = self._sim.config.oven_length_m / max(v_belt_init, 1e-6)
            transient_sim_s = 2.0 * residence_s
            target_fps = 20.0
            steps_min = max(1, int(transient_sim_s / (15.0 * target_fps * 0.3)))
            steps_max = 60

            t_sim = self._sim.sim_time
            ramp = min(t_sim / max(transient_sim_s * 2, 1.0), 1.0)
            steps = int(steps_min + ramp * (steps_max - steps_min))

            # Step the simulator
            finished = False
            for _ in range(steps):
                if self._sim.sim_time >= self._duration_s - 1e-12:
                    finished = True
                    break
                dt = self._sim.compute_stable_dt()
                dt = min(dt, self._duration_s - self._sim.sim_time)
                state = self._sim.step(dt)

                # Update KPIs periodically
                if len(self._sim.history) % 10 == 0:
                    pct = int(100 * self._sim.sim_time / self._duration_s)
                    kpis = {
                        "T_mean_c": state.T_mean_c,
                        "T_max_c": state.T_max_c,
                        "T_outfeed_c": state.T_outfeed_c,
                        "M_mean_wb": state.M_mean_wb,
                        "M_outfeed_wb": state.M_outfeed_wb,
                        "rf_power_kw": state.rf_power_kw,
                        "anode_current_a": state.anode_current_a,
                        "electrode_gap_mm": state.electrode_gap_mm,
                    }
                    self._on_progress(pct, state.time_s, kpis)

            # Update visualization
            self._update_live_visualization(self._sim)

            # Handle completion — enter wind-down phase
            if finished:
                particle_sys = self._sim.particles
                riding_n = particle_sys.riding_count
                hopper_n = particle_sys.hopper_count
                if riding_n > 0 or hopper_n > 0:
                    self._winding_down = True
                    self._log(
                        f"Simulation done — clearing belt "
                        f"({riding_n} riding, {hopper_n} in hopper)..."
                    )
                else:
                    self._finalize_results()

        except Exception as e:
            self._log(f"Render tick error: {e}\n{traceback.format_exc()}")
            self._finalize_results()

    def _finalize_results(self):
        """Build results and switch to the results page.

        Called after the belt is fully cleared (wind-down complete).
        """
        if self._sim is None:
            return

        try:
            particle_sys = self._sim.particles

            # Build results
            elapsed = time.perf_counter() - self._t0_wall
            result = self._sim.build_result()
            outlet = self._sim.get_outlet_conditions()
            meshes = self._sim.get_mesh()

            # Use canonical result.time_series (includes electrode_temperature_c); add controller_state from history for GUI/export
            history = self._sim.history
            ts = dict(result.time_series) if result.time_series else {}
            if history and ts and hasattr(history[0], "controller_state"):
                ts["controller_state"] = [h.controller_state for h in history]

            collected_mass_kg = 0.0
            collected_count = 0
            dispatched_mass_kg = 0.0
            if hasattr(particle_sys, "collected_mass_kg"):
                collected_mass_kg = particle_sys.collected_mass_kg
                collected_count = particle_sys.collected_count
            if hasattr(particle_sys, "dispatched_mass_kg"):
                dispatched_mass_kg = particle_sys.dispatched_mass_kg

            # Capture particle-level data for analysis plots
            particle_data = {}
            if particle_sys is not None:
                collected_mask = (particle_sys.state == particle_sys._STATE_COLLECTED)
                riding_mask = (particle_sys.state == particle_sys._STATE_RIDING)

                particle_data["n_total"] = particle_sys.cfg.max_particles
                particle_data["hopper_count"] = particle_sys.hopper_count
                particle_data["riding_count"] = particle_sys.riding_count
                particle_data["collected_count"] = particle_sys.collected_count
                particle_data["falling_count"] = particle_sys.falling_count

                if collected_mask.any():
                    if hasattr(particle_sys, "T_at_oven_exit"):
                        particle_data["T_collected"] = particle_sys.T_at_oven_exit[collected_mask].copy()
                    else:
                        particle_data["T_collected"] = particle_sys.temperature[collected_mask].copy()
                    if hasattr(particle_sys, "M_at_oven_exit"):
                        particle_data["M_collected"] = particle_sys.M_at_oven_exit[collected_mask].copy() * 100
                if hasattr(particle_sys, "vicilin_native"):
                    active_mask = collected_mask if collected_mask.any() else riding_mask
                    if active_mask.any():
                        particle_data["vicilin_native_mean"] = float(np.mean(particle_sys.vicilin_native[active_mask]))
                        particle_data["legumin_native_mean"] = float(np.mean(particle_sys.legumin_native[active_mask]))
                if hasattr(particle_sys, "T_core"):
                    active_mask = collected_mask if collected_mask.any() else riding_mask
                    if active_mask.any():
                        if not collected_mask.any() or not hasattr(particle_sys, "T_core_at_oven_exit"):
                            particle_data["T_surface"] = particle_sys.temperature[active_mask].copy()
                            particle_data["T_core"] = particle_sys.T_core[active_mask].copy()
                        else:
                            particle_data["T_surface"] = particle_sys.T_at_oven_exit[active_mask].copy()
                            particle_data["T_core"] = particle_sys.T_core_at_oven_exit[active_mask].copy()

            self._results = {
                "outlet": outlet,
                "result": result,
                "meshes": meshes,
                "time_series": ts,
                "elapsed_s": elapsed,
                "n_steps": len(history),
                "collected_mass_kg": collected_mass_kg,
                "collected_count": collected_count,
                "dispatched_mass_kg": dispatched_mass_kg,
                "run_mass_kg": self._recipe.run_mass_kg,
                "gap_mm": self._recipe.electrode_gap_mm,
                "bed_depth_mm": self._material.bed_depth_m * 1000.0,
                "belt_speed": self._recipe.belt_speed_m_per_min,
                "duration_s": self._duration_s,
                "initial_moisture": self._material.initial_moisture_wb,
                "initial_temp_c": self._material.initial_temperature_c,
                "particle_data": particle_data,
            }

            # Stop render timer
            if self._render_timer:
                self._render_timer.stop()
                self._render_timer = None

            self._running = False
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._progress.setValue(100)

            # Update plots and summary on results page
            if _HAS_MATPLOTLIB and ts.get("time_s"):
                self._draw_simulation_plots(ts, self._results)
            self._update_summary(self._results)
            if _HAS_MATPLOTLIB:
                if outlet:
                    self._draw_outfeed_section(self._results)
                self._draw_particle_plots(self._results)

            # Enable "View Full Results" and update status
            self._view_results_btn.setEnabled(True)
            self._results_status_label.setText(
                f"Complete | M={outlet.avg_moisture_wb:.1%} | "
                f"T={outlet.sensor_temperature_c:.1f} \u00b0C | "
                f"E={outlet.total_energy_kwh:.3f} kWh | "
                f"wall={elapsed:.1f} s"
            )
            self._results_status_label.setStyleSheet(
                f"color: {COLORS.SUCCESS}; font-size: 9pt;"
                " border: none; background: transparent;"
            )

            # Update results page header
            self._results_title_label.setText(
                f"GP-15 Results \u2014 {self._recipe.run_mass_kg:.0f} kg  |  "
                f"Gap {self._recipe.electrode_gap_mm:.0f} mm  |  "
                f"Bed {self._material.bed_depth_m*1000:.0f} mm  |  "
                f"Belt {self._recipe.belt_speed_m_per_min:.2f} m/min  |  "
                f"{self._duration_s:.0f} s"
            )

            self._log(
                f"Complete | M_out={outlet.avg_moisture_wb:.1%} "
                f"| T_sensor={outlet.sensor_temperature_c:.1f} \u00b0C "
                f"| CV={outlet.moisture_uniformity:.3f} "
                f"| E={outlet.total_energy_kwh:.3f} kWh "
                f"| wall={elapsed:.1f} s"
            )

            # Auto-switch to results page
            self._show_results_view()

            self.simulation_finished.emit(self._results)

        except Exception as e:
            self._log(f"Finish error: {e}\n{traceback.format_exc()}")
            self._running = False
            if self._render_timer:
                self._render_timer.stop()
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_belt_width_mm(self) -> float:
        """Get belt width in mm from the simulator assembly, or default."""
        if self._assembly_info is not None:
            try:
                return self._assembly_info["belt_width_m"] * 1000
            except (KeyError, TypeError):
                pass
        if self._sim is not None:
            try:
                info = self._sim.assembly.get_assembly_info()
                return info["belt_width_m"] * 1000
            except Exception:
                pass
        return 800.0  # MachineConfig default

    # ──────────────────────────────────────────────────────────────
    #  Matplotlib Plots
    # ──────────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        """Apply dark theme styling to a matplotlib axis."""
        ax.set_facecolor(COLORS.BG_DARKEST)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(COLORS.BORDER)
        ax.title.set_color(COLORS.TEXT_PRIMARY)
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_color(COLORS.TEXT_SECONDARY)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_color(COLORS.TEXT_SECONDARY)
        ax.yaxis.label.set_fontsize(8)

    def _draw_simulation_plots(
        self, ts: dict, results: Optional[dict] = None
    ):
        """Draw the 3x3 simulation plot grid matching the example.

        Layout:
          [0,0] Temperature   [0,1] Moisture      [0,2] Electrode Gap
          [1,0] RF Power      [1,1] Anode Current  [1,2] Cumulative Energy
          [2,0] Specific E    [2,1] Mass Account   [2,2] Outfeed T section
        """
        fig = self._plot_figure
        fig.clear()

        t_arr = np.array(ts["time_s"])
        t_min = t_arr / 60.0

        axes = fig.subplots(3, 3)

        r = results or {}
        gap_mm = r.get("gap_mm", 75)
        bed_mm = r.get("bed_depth_mm", 25)
        belt_speed = r.get("belt_speed", 0.2)
        mass_kg = r.get("run_mass_kg", 61)
        duration = r.get("duration_s", t_arr[-1] if len(t_arr) else 0)
        belt_width_mm = self._get_belt_width_mm()

        fig.suptitle(
            f"GP-15 Digital Twin \u2014 {mass_kg:.0f} kg  |  "
            f"Gap {gap_mm:.0f} mm  |  Bed {bed_mm:.0f} mm  |  "
            f"Belt {belt_speed} m/min  |  {duration:.0f} s",
            fontsize=10, fontweight="bold", color=COLORS.TEXT_PRIMARY,
        )

        # ── [0,0] Temperature + Protein Denaturation ──────────────
        ax = axes[0, 0]
        self._style_ax(ax)
        if "T_mean_c" in ts and "T_max_c" in ts:
            ax.fill_between(
                t_min, ts["T_mean_c"], ts["T_max_c"],
                alpha=0.15, color="red", label="T range",
            )
        if "T_outfeed_sensor_c" in ts:
            ax.plot(t_min, ts["T_outfeed_sensor_c"], "r-", lw=2, label="T sensor (P75)")
        if "T_outfeed_c" in ts:
            ax.plot(t_min, ts["T_outfeed_c"], "r--", lw=1.2, alpha=0.6, label="T outfeed (mean)")
        ax.axhline(76, color="orange", ls=":", alpha=0.6, label="Legumin onset 76\u00b0C")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Temperature [\u00b0C]")
        ax.set_title("Temperature & Protein Quality")
        # Secondary Y-axis: protein denaturation (globulin native loss)
        if "protein_denaturation" in ts:
            denat_pct = np.array(ts["protein_denaturation"]) * 100
            if np.any(denat_pct > 0):
                ax_d = ax.twinx()
                ax_d.plot(t_min, denat_pct, "k-", lw=1.5, alpha=0.7,
                          label="Globulin native loss [%]")
                ax_d.axhline(15, color="green", ls="--", alpha=0.6, lw=1,
                             label="Target max 15%")
                ax_d.set_ylabel("Globulin native loss [%]", color="k", fontsize=7)
                ax_d.set_ylim(bottom=0)
                ax_d.tick_params(colors=COLORS.TEXT_SECONDARY, labelsize=7)
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_d.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper left")
            else:
                ax.legend(fontsize=6, loc="upper left")
        else:
            ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [0,1] Moisture ───────────────────────────────────────
        ax = axes[0, 1]
        self._style_ax(ax)
        if "M_outfeed_wb" in ts:
            ax.plot(
                t_min, np.array(ts["M_outfeed_wb"]) * 100,
                "b-", lw=2, label="M outfeed",
            )
        if "M_mean_wb" in ts:
            ax.plot(
                t_min, np.array(ts["M_mean_wb"]) * 100,
                "b-", lw=0.8, alpha=0.4, label="M mean",
            )
        # Target moisture line (from example)
        if self._material is not None:
            try:
                target = self._material.target_moisture_wb
                ax.axhline(
                    target * 100, color="green", ls="--", alpha=0.7, lw=1.5,
                    label=f"Target {target:.0%}",
                )
            except AttributeError:
                pass
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Moisture [% wb]")
        ax.set_title("Moisture Content")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [0,2] Electrode Gap ──────────────────────────────────
        ax = axes[0, 2]
        self._style_ax(ax)
        if "electrode_gap_mm" in ts:
            ax.plot(
                t_min, ts["electrode_gap_mm"], "g-", lw=2, label="Actual gap",
            )
            ax.axhline(
                gap_mm, color="gray", ls="--", alpha=0.5,
                label=f"Setpoint {gap_mm:.0f} mm",
            )
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Electrode Gap [mm]")
        ax.set_title("MRH Gap Control")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [1,0] RF Power ───────────────────────────────────────
        ax = axes[1, 0]
        self._style_ax(ax)
        if "rf_power_kw" in ts:
            ax.plot(t_min, ts["rf_power_kw"], "m-", lw=2, label="RF power (in)")
        if "evap_power_kw" in ts:
            ax.plot(
                t_min, ts["evap_power_kw"], "c-", lw=1.5,
                alpha=0.8, label="Evap. cooling",
            )
        # Rated max power line (from example)
        if self._sim is not None:
            try:
                max_power = self._sim.config.max_rf_power_kw
                ax.axhline(
                    max_power, color="gray", ls=":", alpha=0.4,
                    label=f"Rated max {max_power} kW",
                )
            except AttributeError:
                pass
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Power [kW]")
        ax.set_title("Energy Balance")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [1,1] Anode Current ──────────────────────────────────
        ax = axes[1, 1]
        self._style_ax(ax)
        if "anode_current_a" in ts:
            ax.plot(t_min, ts["anode_current_a"], color=COLORS.TEXT_PRIMARY, ls="-", lw=2, label="Ia")
            # MRH line
            if self._recipe is not None:
                mrh = self._recipe.mrh_amps
                ax.axhline(mrh, color="red", ls="--", alpha=0.6, lw=1.5,
                            label=f"MRH = {mrh} A")
                # MRL line (from example)
                mrl = self._recipe.mrl_amps
                ax.axhline(mrl, color="orange", ls="--", alpha=0.6, lw=1.5,
                            label=f"MRL = {mrl} A")
            else:
                mrh = self._mrh_spin.value()
                ax.axhline(mrh, color="red", ls="--", alpha=0.6, lw=1.5,
                            label=f"MRH = {mrh} A")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Anode Current [A]")
        ax.set_title("Anode Current (Ia)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [1,2] Cumulative Energy + Water ──────────────────────
        ax = axes[1, 2]
        self._style_ax(ax)
        if "total_energy_kwh" in ts:
            ax.plot(t_min, ts["total_energy_kwh"], "m-", lw=2, label="RF energy")
            ax.set_ylabel("Energy [kWh]", color="m", fontsize=8)
        if "water_removed_kg" in ts:
            ax2 = ax.twinx()
            ax2.plot(
                t_min, np.array(ts["water_removed_kg"]) * 1000,
                "c-", lw=1.5, label="Water removed",
            )
            ax2.set_ylabel("Water Removed [g]", color="c", fontsize=8)
            ax2.tick_params(colors="c", labelsize=7)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")
        ax.set_xlabel("Time [min]")
        ax.set_title("Cumulative Totals")
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [2,0] Specific Energy ────────────────────────────────
        ax = axes[2, 0]
        self._style_ax(ax)
        if "specific_energy_kwh_per_kg" in ts:
            se = np.array(ts["specific_energy_kwh_per_kg"])
            valid = se > 0
            if valid.any():
                ax.plot(t_min[valid], se[valid], color=COLORS.TEXT_PRIMARY, ls="-", lw=1.5)
                ax.axhline(
                    1.0, color="green", ls="--", alpha=0.5,
                    label="Manual target: 1.0 kWh/kg",
                )
                ax.axhline(
                    1.0 / 0.6, color="orange", ls=":", alpha=0.5,
                    label="Low S/V factor: 1.67 kWh/kg",
                )
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("kWh / kg water")
        ax.set_title("Specific Energy (kWh/kg water)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)
        ax.set_ylim(bottom=0)

        # ── [2,1] Material Accounting (Mass Balance) ─────────────
        ax = axes[2, 1]
        self._style_ax(ax)
        collected_kg = r.get("collected_mass_kg", 0)
        collected_n = r.get("collected_count", 0)
        dispatched_kg = r.get("dispatched_mass_kg", 0)
        if collected_kg > 0 or mass_kg > 0:
            ax.bar(
                ["Input", "Collected"],
                [mass_kg, collected_kg],
                color=["#4169E1", "#DAA520"], alpha=0.8,
            )
            ax.set_ylabel("Mass [kg]")
            balance_str = ""
            if dispatched_kg > 0:
                balance_pct = (collected_kg - dispatched_kg) / dispatched_kg * 100
                balance_str = f", {balance_pct:+.1f}%"
            ax.set_title(f"Mass Balance ({collected_n} particles{balance_str})")
        else:
            ax.text(
                0.5, 0.5, "No particle data",
                ha="center", va="center", transform=ax.transAxes,
                color=COLORS.TEXT_MUTED,
            )
        ax.grid(True, alpha=0.2, color=COLORS.BORDER, axis="y")

        # ── [2,2] Outfeed Temperature Cross-Section ──────────────
        ax = axes[2, 2]
        self._style_ax(ax)
        outlet = r.get("outlet")
        if outlet and outlet.temperature_field is not None:
            im = ax.imshow(
                outlet.temperature_field,
                aspect="auto", origin="lower", cmap="hot",
                extent=[0, belt_width_mm, 0, gap_mm],
            )
            ax.set_xlabel("Z \u2014 belt width [mm]")
            ax.set_ylabel("Y \u2014 gap [mm]")
            ax.set_title(
                f"Outfeed T  (sensor {outlet.sensor_temperature_c:.1f}\u00b0C, "
                f"max {outlet.max_temperature_c:.1f}\u00b0C)",
            )
            fig.colorbar(im, ax=ax, label="\u00b0C", shrink=0.8)
        else:
            ax.text(
                0.5, 0.5, "No field data",
                ha="center", va="center", transform=ax.transAxes,
                color=COLORS.TEXT_MUTED,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._plot_canvas.draw()

    def _draw_outfeed_section(self, results: dict):
        """Draw outfeed temperature + moisture cross-sections."""
        fig = self._outfeed_figure
        fig.clear()

        outlet = results.get("outlet")
        if not outlet:
            return

        gap_mm = results.get("gap_mm", 75)
        belt_width_mm = self._get_belt_width_mm()

        if outlet.temperature_field is not None and outlet.moisture_field is not None:
            axes = fig.subplots(1, 2)

            peak_note = " at oven exit (peak)" if getattr(outlet, "at_peak_processing_snapshot", False) else ""
            fig.suptitle(
                f"Outfeed Cross-Section \u2014 Pipeline Output to Milling{peak_note}  |  "
                f"Residence {outlet.residence_time_s:.0f} s  |  "
                f"Throughput {outlet.throughput_kg_per_hr:.0f} kg/h",
                fontsize=9, fontweight="bold", color=COLORS.TEXT_PRIMARY,
            )

            # Temperature
            ax = axes[0]
            self._style_ax(ax)
            im_t = ax.imshow(
                outlet.temperature_field, aspect="auto", origin="lower",
                cmap="hot", extent=[0, belt_width_mm, 0, gap_mm],
            )
            ax.set_xlabel("Z \u2014 belt width [mm]")
            ax.set_ylabel("Y \u2014 gap [mm]")
            ax.set_title(
                f"Temperature  (sensor {outlet.sensor_temperature_c:.1f}\u00b0C, "
                f"max {outlet.max_temperature_c:.1f}\u00b0C)",
            )
            fig.colorbar(im_t, ax=ax, label="\u00b0C")

            # Moisture
            ax = axes[1]
            self._style_ax(ax)
            im_m = ax.imshow(
                outlet.moisture_field * 100, aspect="auto", origin="lower",
                cmap="Blues", extent=[0, belt_width_mm, 0, gap_mm],
            )
            ax.set_xlabel("Z \u2014 belt width [mm]")
            ax.set_ylabel("Y \u2014 gap [mm]")
            ax.set_title(
                f"Moisture  (avg {outlet.avg_moisture_wb:.1%}, "
                f"CV {outlet.moisture_uniformity:.3f})  |  "
                f"Spec. energy {outlet.specific_energy_kwh_per_kg:.2f} kWh/kg water",
            )
            fig.colorbar(im_m, ax=ax, label="% wb")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        self._outfeed_canvas.draw()

    def _draw_particle_plots(self, results: dict):
        """Draw 2x2 particle system analysis plots (matches example Figures 3+5).

        Layout:
          [0,0] Particle state pie chart
          [0,1] Treatment temperature histogram (collected at oven exit)
          [1,0] Protein fraction bars (Vicilin 7S vs Legumin 11S)
          [1,1] Core vs Surface temperature (Biot model)
        """
        fig = self._particle_figure
        fig.clear()

        pd = results.get("particle_data", {})
        if not pd:
            ax = fig.add_subplot(111)
            self._style_ax(ax)
            ax.text(0.5, 0.5, "No particle data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color=COLORS.TEXT_MUTED, fontsize=12)
            self._particle_canvas.draw()
            return

        n_total = pd.get("n_total", 0)
        collected_n = pd.get("collected_count", 0)

        fig.suptitle(
            f"Particle System Analysis  |  "
            f"n={n_total}  |  Collected: {collected_n}",
            fontsize=10, fontweight="bold", color=COLORS.TEXT_PRIMARY,
        )

        axes = fig.subplots(2, 2)

        # ── [0,0] Particle state pie chart ──────────────────────────
        ax = axes[0, 0]
        self._style_ax(ax)
        states = {
            "Hopper": pd.get("hopper_count", 0),
            "Riding": pd.get("riding_count", 0),
            "Falling": pd.get("falling_count", 0),
            "Collected": pd.get("collected_count", 0),
        }
        states_nz = {k: v for k, v in states.items() if v > 0}
        if states_nz:
            color_map = {
                "Hopper": "#FFA500", "Riding": "#4169E1",
                "Falling": "#DC143C", "Collected": "#228B22",
            }
            ax.pie(
                states_nz.values(),
                labels=[f"{k}\n({v})" for k, v in states_nz.items()],
                colors=[color_map[k] for k in states_nz.keys()],
                autopct="%1.1f%%", startangle=90,
                textprops={"color": COLORS.TEXT_PRIMARY, "fontsize": 7},
            )
            ax.set_title(f"Particle States (n={n_total})", fontsize=9,
                         color=COLORS.TEXT_PRIMARY)
        else:
            ax.text(0.5, 0.5, "No particles", ha="center", va="center",
                    transform=ax.transAxes, color=COLORS.TEXT_MUTED)

        # ── [0,1] Treatment temperature histogram ───────────────────
        ax = axes[0, 1]
        self._style_ax(ax)
        T_collected = pd.get("T_collected")
        if T_collected is not None and len(T_collected) > 0:
            ax.hist(T_collected, bins=30, alpha=0.7, color="#228B22",
                    label=f"Collected (n={len(T_collected)})", edgecolor="black",
                    linewidth=0.5)
            ax.axvline(np.mean(T_collected), color="red", ls="--", lw=2,
                       label=f"Mean: {np.mean(T_collected):.1f}\u00b0C")
            ax.axvline(np.percentile(T_collected, 75), color="orange", ls=":",
                       lw=1.5, label=f"P75: {np.percentile(T_collected, 75):.1f}\u00b0C")
            ax.set_xlabel("Temperature [\u00b0C]")
            ax.set_ylabel("Count")
            ax.set_title("Treatment Temperature\n(at oven exit)")
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "No collected particles", ha="center", va="center",
                    transform=ax.transAxes, color=COLORS.TEXT_MUTED)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        # ── [1,0] Protein fraction bars (Vicilin 7S vs Legumin 11S) ─
        ax = axes[1, 0]
        self._style_ax(ax)
        vic_mean = pd.get("vicilin_native_mean")
        leg_mean = pd.get("legumin_native_mean")
        if vic_mean is not None and leg_mean is not None:
            vic_denat_pct = (1.0 - vic_mean) * 100
            leg_denat_pct = (1.0 - leg_mean) * 100
            vic_native_pct = vic_mean * 100
            leg_native_pct = leg_mean * 100

            x = np.arange(2)
            width = 0.35
            ax.bar(x - width / 2, [vic_native_pct, leg_native_pct], width,
                   label="Native (preserved)", color="#228B22", alpha=0.8)
            ax.bar(x + width / 2, [vic_denat_pct, leg_denat_pct], width,
                   label="Denatured (lost)", color="#DC143C", alpha=0.8)

            ax.set_ylabel("Percentage [%]")
            ax.set_title("Protein Fraction Status")
            ax.set_xticks(x)
            ax.set_xticklabels(["Vicilin (7S)\nonset 62\u00b0C",
                                "Legumin (11S)\nonset 76\u00b0C"])
            ax.legend(fontsize=6)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.2, color=COLORS.BORDER, axis="y")

            for i, (nat, den) in enumerate([(vic_native_pct, vic_denat_pct),
                                             (leg_native_pct, leg_denat_pct)]):
                ax.text(i - width / 2, nat + 2, f"{nat:.0f}%", ha="center",
                        fontsize=7, color=COLORS.TEXT_PRIMARY)
                ax.text(i + width / 2, den + 2, f"{den:.0f}%", ha="center",
                        fontsize=7, color=COLORS.TEXT_PRIMARY)
        else:
            ax.text(0.5, 0.5, "No denaturation data", ha="center", va="center",
                    transform=ax.transAxes, color=COLORS.TEXT_MUTED)

        # ── [1,1] Core vs Surface temperature (Biot model) ──────────
        ax = axes[1, 1]
        self._style_ax(ax)
        T_surface = pd.get("T_surface")
        T_core = pd.get("T_core")
        if T_surface is not None and T_core is not None and len(T_surface) > 0:
            ax.scatter(T_surface, T_core, alpha=0.3, s=5, c="blue")
            ax.plot([T_surface.min(), T_surface.max()],
                    [T_surface.min(), T_surface.max()],
                    "k--", alpha=0.5, label="T_core = T_surface")
            lag = np.mean(T_surface - T_core)
            ax.set_title(f"Core vs Surface Temperature\n(Biot model, avg lag: {lag:.1f}\u00b0C)")
            ax.set_xlabel("Surface Temperature [\u00b0C]")
            ax.set_ylabel("Core Temperature [\u00b0C]")
            ax.legend(fontsize=6)
        else:
            ax.text(0.5, 0.5, "No Biot model data", ha="center", va="center",
                    transform=ax.transAxes, color=COLORS.TEXT_MUTED)
        ax.grid(True, alpha=0.2, color=COLORS.BORDER)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        self._particle_canvas.draw()

    # ──────────────────────────────────────────────────────────────
    #  Summary (KPI cards on results page)
    # ──────────────────────────────────────────────────────────────

    def _update_summary(self, results: dict):
        """Update the results page KPI cards with final results."""
        outlet = results.get("outlet")
        if outlet is None:
            return

        self._rc_moisture.set_value(f"{outlet.avg_moisture_wb:.2%}")
        # Use sensor-comparable temperature (75th percentile) - matches PLC/strips
        self._rc_temp.set_value(f"{outlet.sensor_temperature_c:.1f} \u00b0C")
        self._rc_max_temp.set_value(f"{outlet.max_temperature_c:.1f} \u00b0C")
        self._rc_cv.set_value(f"{outlet.moisture_uniformity:.4f}")

        result_obj = results.get("result")
        if result_obj:
            self._rc_energy.set_value(
                f"{result_obj.energy_consumed_kwh:.4f} kWh"
            )
            self._rc_throughput.set_value(
                f"{result_obj.throughput_kg_per_h:.0f} kg/h"
            )

        if outlet.specific_energy_kwh_per_kg > 0:
            self._rc_specific.set_value(
                f"{outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water"
            )

        elapsed = results.get("elapsed_s", 0)
        n_steps = results.get("n_steps", 0)
        wall_str = f"{elapsed:.1f} s"
        self._rc_wall.set_value(wall_str)

        # Simulation speed
        if n_steps > 0 and elapsed > 0:
            self._rc_sim_speed.set_value(f"{n_steps / elapsed:.0f} steps/s")

        # Protein quality (globulin native loss)
        denat = outlet.protein_denaturation_fraction
        particle_data = results.get("particle_data", {})
        if denat > 0 or particle_data.get("vicilin_native_mean") is not None:
            denat_pct = denat * 100
            protein_str = f"{denat_pct:.1f}%"
            vic_mean = particle_data.get("vicilin_native_mean")
            leg_mean = particle_data.get("legumin_native_mean")
            if vic_mean is not None and leg_mean is not None:
                vic_denat = (1.0 - vic_mean) * 100
                leg_denat = (1.0 - leg_mean) * 100
                protein_str += f"  (7S: {vic_denat:.1f}%, 11S: {leg_denat:.1f}%)"
            self._rc_protein.set_value(protein_str)

        # Mass balance
        dispatched_kg = results.get("dispatched_mass_kg", 0.0)
        collected_kg = results.get("collected_mass_kg", 0.0)
        if dispatched_kg > 0:
            balance_pct = (collected_kg - dispatched_kg) / dispatched_kg * 100
            self._rc_mass_balance.set_value(
                f"{collected_kg:.1f} / {dispatched_kg:.1f} kg ({balance_pct:+.1f}%)"
            )
        elif collected_kg > 0:
            self._rc_mass_balance.set_value(f"{collected_kg:.1f} kg")

        # Final electrode gap
        ts = results.get("time_series", {})
        if ts.get("electrode_gap_mm"):
            final_gap = ts["electrode_gap_mm"][-1]
            self._rc_final_gap.set_value(f"{final_gap:.1f} mm")

        # Desirability scoring
        self._update_desirability(results)

    # ──────────────────────────────────────────────────────────────
    #  Export
    # ──────────────────────────────────────────────────────────────

    def _export_plots(self):
        """Export the 3x3 plot grid as image."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        if not _HAS_MATPLOTLIB:
            return

        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Plots",
            str(Path.home() / "gp15_simulation_plots.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if fp:
            self._plot_figure.savefig(
                fp, facecolor=COLORS.BG_DARK, edgecolor="none", dpi=150,
            )

    def _export_csv(self):
        """Export time-series data as CSV."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        if not self._results:
            return

        fp, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            str(Path.home() / "gp15_results.csv"),
            "CSV (*.csv)",
        )
        if fp:
            ts = self._results.get("time_series", {})
            if not ts:
                return
            keys = list(ts.keys())
            with open(fp, "w") as f:
                f.write(",".join(keys) + "\n")
                n = len(ts[keys[0]])
                for i in range(n):
                    row = [str(ts[k][i]) for k in keys]
                    f.write(",".join(row) + "\n")

    def _export_json(self):
        """Export results summary as JSON."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        import json

        if not self._results:
            return

        fp, _ = QFileDialog.getSaveFileName(
            self, "Export JSON",
            str(Path.home() / "gp15_results.json"),
            "JSON (*.json)",
        )
        if fp:
            outlet = self._results.get("outlet")
            summary = {
                "duration_s": self._results.get("duration_s"),
                "elapsed_s": self._results.get("elapsed_s"),
                "gap_mm": self._results.get("gap_mm"),
                "bed_depth_mm": self._results.get("bed_depth_mm"),
                "belt_speed_m_per_min": self._results.get("belt_speed"),
                "run_mass_kg": self._results.get("run_mass_kg"),
                "initial_moisture_wb": self._results.get("initial_moisture"),
                "initial_temp_c": self._results.get("initial_temp_c"),
                "n_steps": self._results.get("n_steps"),
                "collected_mass_kg": self._results.get("collected_mass_kg"),
                "collected_count": self._results.get("collected_count"),
            }
            if outlet:
                summary.update({
                    "avg_moisture_wb": outlet.avg_moisture_wb,
                    "avg_temperature_c": outlet.avg_temperature_c,
                    "max_temperature_c": outlet.max_temperature_c,
                    "moisture_uniformity_cv": outlet.moisture_uniformity,
                    "total_energy_kwh": outlet.total_energy_kwh,
                    "specific_energy_kwh_per_kg": outlet.specific_energy_kwh_per_kg,
                    "throughput_kg_per_hr": outlet.throughput_kg_per_hr,
                    "residence_time_s": outlet.residence_time_s,
                })
            with open(fp, "w") as f:
                json.dump(summary, f, indent=2, default=str)

    def _export_pdf_report(self):
        """Export a complete PDF report with KPIs, plots, and outfeed."""
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path

        if not self._results or not _HAS_MATPLOTLIB:
            return

        fp, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report",
            str(Path.home() / "gp15_report.pdf"),
            "PDF (*.pdf)",
        )
        if not fp:
            return

        try:
            from matplotlib.backends.backend_pdf import PdfPages

            outlet = self._results.get("outlet")
            r = self._results
            ts = r.get("time_series", {})

            with PdfPages(fp) as pdf:
                # Page 1: Title + KPIs
                fig_title = Figure(figsize=(11, 8.5), dpi=100)
                fig_title.patch.set_facecolor("white")
                ax = fig_title.add_subplot(111)
                ax.axis("off")

                title_text = (
                    "GP-15 RF Pretreatment Report\n"
                    "=" * 40 + "\n\n"
                    f"Run Mass:        {r.get('run_mass_kg', 0):.1f} kg\n"
                    f"Electrode Gap:   {r.get('gap_mm', 0):.0f} mm\n"
                    f"Bed Depth:       {r.get('bed_depth_mm', 0):.0f} mm\n"
                    f"Belt Speed:      {r.get('belt_speed', 0):.2f} m/min\n"
                    f"Duration:        {r.get('duration_s', 0):.0f} s "
                    f"({r.get('duration_s', 0)/60:.1f} min)\n"
                    f"Wall-Clock:      {r.get('elapsed_s', 0):.1f} s\n"
                    f"Steps:           {r.get('n_steps', 0):,}\n"
                    "\n"
                )
                if outlet:
                    title_text += (
                        "Results\n"
                        "-" * 40 + "\n"
                        f"Outfeed Moisture:    {outlet.avg_moisture_wb:.2%}\n"
                        f"Outfeed Temperature (sensor P75): {outlet.sensor_temperature_c:.1f} C\n"
                        f"Max Temperature:     {outlet.max_temperature_c:.1f} C\n"
                        f"Moisture CV:         {outlet.moisture_uniformity:.4f}\n"
                        f"RF Energy:           {outlet.total_energy_kwh:.4f} kWh\n"
                        f"Specific Energy:     {outlet.specific_energy_kwh_per_kg:.3f} kWh/kg water\n"
                        f"Throughput:          {outlet.throughput_kg_per_hr:.0f} kg/h\n"
                        f"Residence Time:      {outlet.residence_time_s:.0f} s\n"
                    )

                ax.text(
                    0.1, 0.95, title_text,
                    ha="left", va="top",
                    fontsize=12, family="monospace",
                    transform=ax.transAxes,
                )
                pdf.savefig(fig_title)

                # Page 2: 3x3 simulation plots
                if ts.get("time_s"):
                    pdf.savefig(self._plot_figure)

                # Page 3: Outfeed cross-section
                if outlet and outlet.temperature_field is not None:
                    pdf.savefig(self._outfeed_figure)

            self._log(f"PDF report exported: {fp}")
        except Exception as e:
            self._log(f"PDF export error: {e}")

    # ──────────────────────────────────────────────────────────────
    #  Build System (called from MainWindow)
    # ──────────────────────────────────────────────────────────────

    def build_system(self, assembly_params: Dict[str, Any]) -> bool:
        """Build and display the GP-15 RF pretreatment machine geometry.

        Args:
            assembly_params: Assembly configuration dict with keys:
                - pt_electrode_gap_mm: Electrode gap in mm (default 200)
                - pt_bed_depth_mm: Material bed depth in mm (default 40)

        Returns:
            True if build succeeded, False otherwise.
        """
        if self._plotter is None:
            self._log("3D viewport not available \u2014 cannot build geometry.")
            return False

        try:
            from airclassifier.pretreatment.geometry.assembly.machine import (
                create_gp15_machine,
                COMPONENT_COLORS,
            )

            gap_mm = assembly_params.get("pt_electrode_gap_mm", 200)
            bed_mm = assembly_params.get("pt_bed_depth_mm", 40)

            self._log(f"Building GP-15 machine: gap={gap_mm:.0f} mm, bed={bed_mm:.0f} mm")

            machine = create_gp15_machine(
                electrode_gap_m=gap_mm / 1000.0,
                bed_depth_m=bed_mm / 1000.0,
            )
            meshes = machine.generate_all_meshes()

            self._plotter.clear()

            total_verts = 0
            xray_opacities = {
                "conveyor_frame": 0.08, "oven_chamber": 0.06,
                "rollers": 0.60, "belt": 0.40,
                "upper_electrode": 0.30, "lower_electrode": 0.25,
                "infeed_hopper": 0.70, "infeed_tunnel": 0.20,
                "outfeed_tunnel": 0.20, "collection_bin": 0.55,
                "emu_housing": 0.10, "generator": 0.20, "rf_feed": 0.80,
            }

            for name, (verts, tris, meta) in meshes.items():
                style = COMPONENT_COLORS.get(name, {})
                color = style.get("color", "#888888")
                opacity = xray_opacities.get(name, style.get("opacity", 0.8))
                if opacity < 0.01:
                    continue

                n_tri = tris.shape[0]
                faces = np.empty((n_tri, 4), dtype=np.int64)
                faces[:, 0] = 3
                faces[:, 1:] = tris
                pd = pv.PolyData(verts.copy(), faces.ravel())
                self._plotter.add_mesh(
                    pd,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    label=style.get("label", name),
                )
                total_verts += len(verts)

            self._plotter.add_axes()
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.reset_camera()
            self._plotter.camera.azimuth = 125
            self._plotter.camera.elevation = 18

            self._log(
                f"GP-15 machine built: {len(meshes)} components, "
                f"{total_verts:,} vertices"
            )
            return True

        except Exception as e:
            self._log(f"Build error: {e}\n{traceback.format_exc()}")
            return False

    # ──────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────

    def cleanup(self):
        """Stop simulation and clean up resources."""
        self._running = False
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
