"""
Milling Page — Hammer Mill Full-Page View
=========================================

Modern full-page simulation and visualization for the hammer mill (pin mill).
Features glass-morphism UI, animated KPI cards, slide-in results panel,
and timeline playback controls.

Architecture::
    MillingPage
        ├── Viewport (PyVista 3D)
        ├── KPI Dashboard (animated cards)
        ├── Control Panel (recipe + buttons)
        ├── Timeline Widget (playback)
        └── Results Overlay (slide-in)
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
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
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import COLORS

# Import new modular widgets
try:
    from ..widgets.common import AnimatedKPICard, GlassCard
    from ..widgets.milling import (
        TimelineWidget,
        ResultsOverlay,
        InteractivePSDChart,
        MillingControlPanel,
        MillingKPIDashboard,
    )
    from ..widgets.milling.results_page import MillingResultsPage
    _HAS_NEW_WIDGETS = True
except ImportError:
    _HAS_NEW_WIDGETS = False

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


def _rgba_to_hex(rgba) -> str:
    """Convert (r,g,b,a) in [0,1] to hex color."""
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


# Visual style for each mill component (color + opacity).
_MILL_STYLE = {
    # --- Mill internals ---
    "rotor":              {"color": (0.52, 0.55, 0.62), "opacity": 0.95},
    "hammers":            {"color": (0.85, 0.68, 0.15), "opacity": 1.0},
    "hammer_pins":        {"color": (0.70, 0.72, 0.75), "opacity": 1.0},
    # --- Enclosure ---
    "screen":             {"color": (0.40, 0.58, 0.52), "opacity": 0.55},
    "housing":            {"color": (0.34, 0.42, 0.54), "opacity": 0.30},
    "housing_discharge":  {"color": (0.34, 0.42, 0.54), "opacity": 0.80},
    "feed_chute":         {"color": (0.55, 0.48, 0.38), "opacity": 0.75},
    # --- Drive train ---
    "drive_motor":        {"color": (0.22, 0.30, 0.22), "opacity": 0.95},
    "drive_shaft":        {"color": (0.70, 0.72, 0.75), "opacity": 1.0},
    "drive_base":         {"color": (0.28, 0.28, 0.30), "opacity": 0.95},
    "drive_feet":         {"color": (0.28, 0.28, 0.30), "opacity": 0.95},
    "drive_pulley_motor": {"color": (0.50, 0.50, 0.55), "opacity": 1.0},
    "drive_pulley_mill":  {"color": (0.50, 0.50, 0.55), "opacity": 1.0},
    "drive_belt":         {"color": (0.15, 0.15, 0.15), "opacity": 1.0},
}


class MillingPage(QWidget):
    """Full-page hammer mill simulation and visualization.

    Signals:
        simulation_finished(dict): Emitted when a run ends, with result and outlet.
    """

    simulation_finished = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._sim: Optional[Any] = None
        self._running = False
        self._term_mode: str = "time"  # Current termination mode
        self._render_timer: Optional[QTimer] = None
        self._plotter = None
        self._results: Optional[Dict[str, Any]] = None
        self._last_params: Dict[str, Any] = {}

        # Animation state for live digital twin visualization
        self._mesh_actors: Dict[str, Any] = {}
        self._original_verts: Dict[str, np.ndarray] = {}
        self._theta: float = 0.0
        self._omega: float = 0.0
        self._particle_cloud: Optional[Any] = None
        self._particle_buf: Optional[np.ndarray] = None
        self._size_buf: Optional[np.ndarray] = None
        self._max_particles: int = 2000

        # Belt animation state
        self._belt_marker_cloud: Optional[Any] = None
        self._belt_path_yz: Optional[np.ndarray] = None
        self._belt_cum_s: Optional[np.ndarray] = None
        self._belt_total_len: float = 0.0
        self._belt_x: float = 0.0
        self._belt_pitch_r: float = 0.0
        self._n_belt_markers: int = 15

        # Motor animation state
        self._motor_center_y: float = 0.0
        self._motor_center_z: float = 0.0
        self._pulley_ratio: float = 1.0

        # Component sets for animation
        self._rotor_animated = {"rotor", "hammers", "hammer_pins", "drive_pulley_mill"}
        self._motor_animated = {"drive_pulley_motor", "drive_shaft"}

        # Simulation history for waveform
        self._kpi_history: List[Dict[str, float]] = []

        self._build_ui()

    def _build_ui(self):
        """Build the modern UI layout."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Stacked widget to switch between simulation and results views
        self._main_stack = QStackedWidget()
        root.addWidget(self._main_stack, 1)

        # Page 0: Simulation view
        sim_page = QWidget()
        sim_layout = QVBoxLayout(sim_page)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        sim_layout.setSpacing(0)

        # Main content area
        main_content = QWidget()
        main_layout = QHBoxLayout(main_content)
        main_layout.setContentsMargins(12, 12, 12, 0)
        main_layout.setSpacing(12)

        # Left: 3D Viewport
        viewport_container = self._build_viewport()
        main_layout.addWidget(viewport_container, 3)

        # Right: Control panel
        if _HAS_NEW_WIDGETS:
            self._control_panel = MillingControlPanel()
            self._control_panel.run_clicked.connect(self._on_run)
            self._control_panel.stop_clicked.connect(self._on_stop)
            self._control_panel.config_clicked.connect(self._show_config_wizard)
            main_layout.addWidget(self._control_panel)
        else:
            ctrl_widget = self._build_legacy_control_panel()
            main_layout.addWidget(ctrl_widget)

        sim_layout.addWidget(main_content, 1)

        # KPI Dashboard (below viewport)
        if _HAS_NEW_WIDGETS:
            self._kpi_dashboard = MillingKPIDashboard()
            dashboard_container = QWidget()
            dashboard_layout = QHBoxLayout(dashboard_container)
            dashboard_layout.setContentsMargins(12, 8, 12, 8)
            dashboard_layout.addWidget(self._kpi_dashboard)

            # View Results button
            self._view_results_btn = QPushButton("View Results")
            self._view_results_btn.setEnabled(False)
            self._view_results_btn.setStyleSheet(f"""
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
                QPushButton:disabled {{
                    background: {COLORS.BG_DARK};
                    border-color: {COLORS.BORDER_SUBTLE};
                    color: {COLORS.TEXT_DISABLED};
                }}
            """)
            self._view_results_btn.clicked.connect(self._toggle_results)
            dashboard_layout.addWidget(self._view_results_btn)

            sim_layout.addWidget(dashboard_container)

        # Timeline widget
        if _HAS_NEW_WIDGETS:
            self._timeline = TimelineWidget()
            self._timeline.play_clicked.connect(self._on_run)
            self._timeline.stop_clicked.connect(self._on_stop)
            self._timeline.seek.connect(self._on_seek)
            sim_layout.addWidget(self._timeline)

        self._main_stack.addWidget(sim_page)

        # Page 1: Full-page results view
        if _HAS_NEW_WIDGETS:
            self._results_page = MillingResultsPage()
            self._results_page.back_clicked.connect(self._show_simulation_view)
            self._results_page.export_clicked.connect(self._show_export_dialog)
            self._main_stack.addWidget(self._results_page)

        # Keep overlay for backwards compatibility (but prefer full page)
        if _HAS_NEW_WIDGETS:
            self._results_overlay = ResultsOverlay(self)
            self._results_overlay.export_clicked.connect(self._show_export_dialog)
            self._results_overlay.hide()

    def _build_viewport(self) -> QWidget:
        """Build the 3D viewport container."""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_PYVISTA:
            self._plotter = QtInteractor(container)
            self._plotter.set_background(color="#3a3a44", top="#282830")
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.add_axes()
            layout.addWidget(self._plotter.interactor)
            self._plotter.add_text(
                "Hammer Mill (Pin Mill)\nConfigure and click Run to start",
                position="upper_left", font_size=10, color=COLORS.TEXT_MUTED,
                name="placeholder_text",
            )
        else:
            lbl = QLabel("PyVista not available. Install: pip install pyvista pyvistaqt")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; padding: 40px;")
            layout.addWidget(lbl)

        return container

    def _build_legacy_control_panel(self) -> QWidget:
        """Build legacy control panel for when new widgets aren't available."""
        container = QWidget()
        container.setMinimumWidth(300)
        layout = QVBoxLayout(container)
        layout.setSpacing(6)

        title = QLabel("Hammer Mill")
        title.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        layout.addWidget(title)

        btn_row = QHBoxLayout()
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

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Recipe group
        g = QGroupBox("Recipe")
        f = QFormLayout(g)
        self._rpm_spin = QDoubleSpinBox()
        self._rpm_spin.setRange(500, 6000)
        self._rpm_spin.setValue(3000)
        self._rpm_spin.setSuffix(" rpm")
        f.addRow("Rotor RPM:", self._rpm_spin)

        self._aperture_spin = QDoubleSpinBox()
        self._aperture_spin.setRange(0.3, 2.0)  # Food powder grade
        self._aperture_spin.setValue(0.5)  # 0.5 mm for protein separation
        self._aperture_spin.setDecimals(2)
        self._aperture_spin.setSingleStep(0.1)
        self._aperture_spin.setSuffix(" mm")
        f.addRow("Screen aperture:", self._aperture_spin)

        self._feed_spin = QDoubleSpinBox()
        self._feed_spin.setRange(10, 2000)
        self._feed_spin.setValue(500)
        self._feed_spin.setSuffix(" kg/h")
        f.addRow("Feed rate:", self._feed_spin)

        # Termination mode selector
        self._term_mode_combo = QComboBox()
        self._term_mode_combo.addItems([
            "Time-based",
            "Mass-processed",
            "Steady-state",
            "Target d50",
        ])
        self._term_mode_combo.currentIndexChanged.connect(self._on_term_mode_changed)
        f.addRow("Termination:", self._term_mode_combo)

        # Duration (for time-based mode)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(1, 600)
        self._duration_spin.setValue(60)
        self._duration_spin.setSuffix(" s")
        self._duration_row = (QLabel("Duration:"), self._duration_spin)
        f.addRow(self._duration_row[0], self._duration_row[1])

        # Target mass (for mass-processed mode)
        self._target_mass_spin = QDoubleSpinBox()
        self._target_mass_spin.setRange(0.1, 100)
        self._target_mass_spin.setValue(1.0)
        self._target_mass_spin.setDecimals(2)
        self._target_mass_spin.setSuffix(" kg")
        self._target_mass_row = (QLabel("Target mass:"), self._target_mass_spin)
        f.addRow(self._target_mass_row[0], self._target_mass_row[1])
        self._target_mass_row[0].setVisible(False)
        self._target_mass_row[1].setVisible(False)

        # Target d50 (for target d50 mode)
        self._target_d50_spin = QDoubleSpinBox()
        self._target_d50_spin.setRange(50, 2000)
        self._target_d50_spin.setValue(500)
        self._target_d50_spin.setSuffix(" µm")
        self._target_d50_row = (QLabel("Target d50:"), self._target_d50_spin)
        f.addRow(self._target_d50_row[0], self._target_d50_row[1])
        self._target_d50_row[0].setVisible(False)
        self._target_d50_row[1].setVisible(False)

        # Min run time (for physics-based modes)
        self._min_time_spin = QDoubleSpinBox()
        self._min_time_spin.setRange(1, 60)
        self._min_time_spin.setValue(5)
        self._min_time_spin.setSuffix(" s")
        self._min_time_row = (QLabel("Min run time:"), self._min_time_spin)
        f.addRow(self._min_time_row[0], self._min_time_row[1])
        self._min_time_row[0].setVisible(False)
        self._min_time_row[1].setVisible(False)

        # Max run time (safety limit for physics-based modes)
        self._max_time_spin = QDoubleSpinBox()
        self._max_time_spin.setRange(10, 600)
        self._max_time_spin.setValue(300)
        self._max_time_spin.setSuffix(" s")
        self._max_time_row = (QLabel("Max run time:"), self._max_time_spin)
        f.addRow(self._max_time_row[0], self._max_time_row[1])
        self._max_time_row[0].setVisible(False)
        self._max_time_row[1].setVisible(False)

        layout.addWidget(g)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            f"font-family: monospace; font-size: 9pt; background: {COLORS.BG_DARKEST};"
        )
        layout.addWidget(self._log_text)

        return container

    def _build_belt_path(self, dp) -> tuple:
        """Build belt centerline as a closed polyline in Y-Z plane."""
        groove_frac = 0.12
        R_mill = dp.mill_pulley_radius_m * (1 - groove_frac)
        R_motor = dp.pulley_radius_m * (1 - groove_frac)
        my, mz = dp.motor_y_offset_m, dp.motor_z_offset_m
        phi = math.atan2(mz, my)
        n_arc = 24

        pts = []
        for i in range(n_arc + 1):
            a = (phi + math.pi / 2) + i * math.pi / n_arc
            pts.append((R_mill * math.cos(a), R_mill * math.sin(a)))
        a0 = phi - math.pi / 2
        pts.append((my + R_motor * math.cos(a0), mz + R_motor * math.sin(a0)))
        for i in range(1, n_arc + 1):
            a = (phi - math.pi / 2) + i * math.pi / n_arc
            pts.append((my + R_motor * math.cos(a), mz + R_motor * math.sin(a)))
        pts.append(pts[0])

        path_yz = np.array(pts, dtype=np.float64)
        diffs = np.diff(path_yz, axis=0)
        seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        cum_s = np.zeros(len(path_yz))
        cum_s[1:] = np.cumsum(seg_lens)
        belt_x = dp.mill_pulley_x_m + dp.mill_pulley_width_m / 2
        return path_yz, cum_s, cum_s[-1], belt_x

    def _sample_belt_markers(self, phase: float) -> np.ndarray:
        """Return (n, 3) marker positions at given phase along belt path."""
        if self._belt_path_yz is None:
            return np.zeros((self._n_belt_markers, 3), dtype=np.float32)
        out = np.zeros((self._n_belt_markers, 3), dtype=np.float32)
        for i in range(self._n_belt_markers):
            s = (phase + i * self._belt_total_len / self._n_belt_markers) % self._belt_total_len
            idx = min(int(np.searchsorted(self._belt_cum_s, s)) - 1, len(self._belt_path_yz) - 2)
            idx = max(idx, 0)
            seg_len = self._belt_cum_s[idx + 1] - self._belt_cum_s[idx]
            t = (s - self._belt_cum_s[idx]) / max(seg_len, 1e-10)
            out[i, 0] = self._belt_x
            out[i, 1] = self._belt_path_yz[idx, 0] + t * (self._belt_path_yz[idx + 1, 0] - self._belt_path_yz[idx, 0])
            out[i, 2] = self._belt_path_yz[idx, 1] + t * (self._belt_path_yz[idx + 1, 1] - self._belt_path_yz[idx, 1])
        return out

    def _on_term_mode_changed(self, index: int):
        """Handle termination mode selection change."""
        # Mode indices: 0=Time, 1=Mass, 2=Steady-state, 3=Target d50
        is_time = (index == 0)
        is_mass = (index == 1)
        is_steady = (index == 2)
        is_target_d50 = (index == 3)
        is_physics = not is_time  # Any physics-based mode

        # Show/hide duration (only for time-based)
        self._duration_row[0].setVisible(is_time)
        self._duration_row[1].setVisible(is_time)

        # Show/hide target mass (only for mass mode)
        self._target_mass_row[0].setVisible(is_mass)
        self._target_mass_row[1].setVisible(is_mass)

        # Show/hide target d50 (only for target d50 mode)
        self._target_d50_row[0].setVisible(is_target_d50)
        self._target_d50_row[1].setVisible(is_target_d50)

        # Show/hide min/max time (for all physics-based modes)
        self._min_time_row[0].setVisible(is_physics)
        self._min_time_row[1].setVisible(is_physics)
        self._max_time_row[0].setVisible(is_physics)
        self._max_time_row[1].setVisible(is_physics)

        # Force layout update to reflect visibility changes
        self.updateGeometry()

    def _get_termination_mode(self) -> str:
        """Get current termination mode string."""
        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            return self._control_panel.get_termination_mode()
        modes = ["time", "mass", "steady_state", "target_d50"]
        return modes[self._term_mode_combo.currentIndex()]

    def _log(self, msg: str):
        """Log a message to the control panel."""
        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            self._control_panel.log(msg)
        elif hasattr(self, "_log_text") and self._log_text:
            self._log_text.append(msg)
            self._log_text.verticalScrollBar().setValue(
                self._log_text.verticalScrollBar().maximum()
            )

    def _toggle_results(self):
        """Show the full-page results view."""
        if _HAS_NEW_WIDGETS and hasattr(self, "_results_page"):
            self._main_stack.setCurrentIndex(1)

    def _show_simulation_view(self):
        """Return to the simulation view from results."""
        if hasattr(self, "_main_stack"):
            self._main_stack.setCurrentIndex(0)

    def _show_config_wizard(self):
        """Show the milling configuration wizard."""
        try:
            from ..dialogs.milling_wizard import MillingConfigWizard
            wizard = MillingConfigWizard(self, self._last_params)
            wizard.configuration_complete.connect(self._on_wizard_complete)
            wizard.exec()
        except ImportError:
            self._log("Config wizard not available")

    def _on_wizard_complete(self, params: Dict[str, Any]):
        """Handle wizard completion."""
        self._last_params.update(params)
        self.build_system(self._last_params)

    def _show_export_dialog(self):
        """Show the export dialog."""
        try:
            from ..dialogs.export_dialog import ExportDialog
            dialog = ExportDialog(self, self._results)
            dialog.export_requested.connect(self._do_export)
            dialog.exec()
        except ImportError:
            self._log("Export dialog not available")

    def _do_export(self, config: Dict[str, Any]):
        """Execute the export."""
        self._log(f"Exporting to {config.get('path', 'unknown')}...")
        # TODO: Implement actual export logic

    def _on_seek(self, position: float):
        """Handle timeline seek."""
        # TODO: Implement state replay
        pass

    def run_simulation(self):
        """Start the mill simulation (called by MainWindow F5 when in Milling mode)."""
        if not self._running:
            self._on_run()

    def _on_run(self):
        """Start the simulation."""
        try:
            from airclassifier.milling import HammerMillSimulator, MillConfig, MillRecipe
        except ImportError as e:
            self._log(f"Milling module not available: {e}")
            return

        # Get recipe from control panel
        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            recipe_data = self._control_panel.get_recipe()
            self._control_panel.set_running(True)
        else:
            # Fallback to individual spinboxes
            term_mode = self._get_termination_mode()
            recipe_data = {
                "rotor_rpm": self._rpm_spin.value(),
                "screen_aperture_mm": self._aperture_spin.value(),
                "feed_rate_kg_per_hr": self._feed_spin.value(),
                "duration_s": self._duration_spin.value(),
                "termination_mode": term_mode,
                "target_mass_kg": self._target_mass_spin.value(),
                "target_d50_um": self._target_d50_spin.value(),
                "min_run_time_s": self._min_time_spin.value(),
                "max_run_time_s": self._max_time_spin.value(),
            }
            self._run_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._progress.setVisible(True)

        self._running = True
        self._kpi_history.clear()

        # Reset live KPIs so they update from 0 as the run progresses
        if _HAS_NEW_WIDGETS and hasattr(self, "_kpi_dashboard"):
            self._kpi_dashboard.clear()

        if _HAS_NEW_WIDGETS and hasattr(self, "_timeline"):
            from ..widgets.milling.timeline_widget import PlaybackState
            self._timeline.set_state(PlaybackState.PLAYING)

        p = self._last_params
        config = MillConfig(
            rotor_rpm=p.get("mill_rotor_rpm", recipe_data.get("rotor_rpm", 3000)),
            rotor_diameter_m=p.get("mill_rotor_diameter_m", 0.20),
            rotor_length_m=p.get("mill_rotor_length_m", 0.30),
            shaft_diameter_m=p.get("mill_shaft_diameter_m", 0.05),
            motor_power_kw=p.get("mill_motor_power_kw", 22.0),
            hammer_rows=int(p.get("mill_hammer_rows", 4)),
            hammers_per_row=int(p.get("mill_hammers_per_row", 4)),
            hammer_mass_kg=p.get("mill_hammer_mass_kg", 0.35),
            hammer_length_m=p.get("mill_hammer_length_m", 0.08),
            hammer_width_m=p.get("mill_hammer_width_m", 0.05),
            hammer_thickness_m=p.get("mill_hammer_thickness_m", 0.008),
            hammer_clearance_m=p.get("mill_hammer_clearance_m", 0.008),
            screen_aperture_mm=p.get("mill_screen_aperture_mm", recipe_data.get("screen_aperture_mm", 1.5)),
            screen_open_area=p.get("mill_screen_open_area", 0.40),
            screen_inner_radius_m=p.get("mill_screen_inner_radius_m", 0.188),
            screen_thickness_m=p.get("mill_screen_thickness_m", 0.003),
            housing_inner_radius_m=p.get("mill_housing_inner_radius_m", 0.20),
            housing_length_m=p.get("mill_housing_length_m", 0.40),
            housing_wall_thickness_m=p.get("mill_housing_wall_thickness_m", 0.008),
            feed_rate_kg_per_hr=p.get("mill_feed_rate_kg_per_hr", recipe_data.get("feed_rate_kg_per_hr", 500)),
            feed_chute_width_m=p.get("mill_feed_chute_width_m", 0.15),
            feed_chute_height_m=p.get("mill_feed_chute_height_m", 0.12),
            discharge_chute_width_m=p.get("mill_discharge_chute_width_m", 0.20),
            discharge_chute_height_m=p.get("mill_discharge_chute_height_m", 0.15),
        )

        recipe = MillRecipe(
            rotor_rpm=config.rotor_rpm,
            screen_aperture_mm=config.screen_aperture_mm,
            feed_rate_kg_per_hr=config.feed_rate_kg_per_hr,
            run_duration_s=recipe_data.get("duration_s", 60),
            seeds_feed_mass_kg=recipe_data.get("seeds_feed_mass_kg", p.get("mill_seeds_feed_mass_kg", p.get("mill_yellow_peas_feed_mass_kg", 0.0))),
        )

        self._sim = HammerMillSimulator(config=config)
        self._sim.load_recipe(recipe)
        self._sim.initialize(initial_holdup_kg=0.01)

        # Configure termination mode
        from airclassifier.milling import TerminationConfig

        # Get termination config from recipe_data (works for both new and fallback widgets)
        term_mode = recipe_data.get("termination_mode", "time")
        term_config = TerminationConfig(
            mode=term_mode,
            run_duration_s=recipe_data.get("duration_s", 60),
            target_mass_kg=recipe_data.get("target_mass_kg", 1.0),
            target_d50_um=recipe_data.get("target_d50_um", 500),
            min_run_time_s=recipe_data.get("min_run_time_s", 5.0),
            max_run_time_s=recipe_data.get("max_run_time_s", 300),
        )
        self._sim.set_termination_config(term_config)
        self._term_mode = term_mode  # Store for later

        # Calculate n_steps based on mode
        if term_mode == "time":
            duration_s = recipe_data.get("duration_s", 60)
        else:
            # For physics-based modes, use max_run_time as upper bound
            duration_s = recipe_data.get("max_run_time_s", 300)

        dt = 0.002
        n_steps = int(duration_s / dt)

        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            self._control_panel.set_progress(0, n_steps)
        else:
            self._progress.setMaximum(n_steps)
            self._progress.setValue(0)

        mode_names = {
            "time": "Time-based",
            "mass": "Mass-processed",
            "steady_state": "Steady-state",
            "target_d50": "Target d50",
        }
        self._log(f"Running mill: {mode_names.get(term_mode, term_mode)} mode, max {duration_s:.0f}s, dt={dt}")

        steps_per_frame = [5, 10]
        transient_steps = int(0.5 / dt)
        current_step = [0]
        # Visual rotation tracking to avoid stroboscopic effect
        # At high RPM, physics theta wraps multiple times per frame
        # Track cumulative visual angle separately for smooth animation
        visual_theta = [0.0]

        def _step_batch():
            if not self._running or self._sim is None:
                return

            ramp = min(current_step[0] / max(transient_steps, 1), 1.0)
            batch = int(steps_per_frame[0] + ramp * (steps_per_frame[1] - steps_per_frame[0]))
            batch = min(batch, n_steps - current_step[0])

            done = 0
            for _ in range(batch):
                if not self._running:
                    break
                self._sim.step(dt)
                done += 1

            current_step[0] += done

            # Update progress
            if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
                self._control_panel.set_progress(current_step[0], n_steps)
            else:
                self._progress.setValue(current_step[0])

            # Update timeline
            if _HAS_NEW_WIDGETS and hasattr(self, "_timeline"):
                current_time = current_step[0] * dt
                self._timeline.set_time(current_time, duration_s)

            # Get latest physics state and update live KPIs every frame
            state = self._sim.engine.history[-1] if self._sim.engine.history else None
            if state is not None and _HAS_NEW_WIDGETS and hasattr(self, "_kpi_dashboard"):
                self._kpi_dashboard.update_from_state(state, animate=True)
                # Allow the UI to repaint so Throughput/d50/Power update visibly during the run
                app = QApplication.instance()
                if app is not None:
                    app.processEvents()

            if state:
                throughput = state.discharge_rate_kg_per_s * 3600
                d50 = state.d50_m * 1e6
                power = state.power_kw
                # Store for waveform
                self._kpi_history.append({
                    "throughput": throughput,
                    "d50": d50,
                    "power": power,
                })

                # Update timeline waveform periodically
                if _HAS_NEW_WIDGETS and hasattr(self, "_timeline") and len(self._kpi_history) % 50 == 0:
                    d50_values = [h["d50"] for h in self._kpi_history]
                    self._timeline.set_waveform_data(d50_values[-100:])

            # Animation updates
            # Use visual_theta for smooth animation instead of physics angle
            # At high RPM (3000 RPM = 314 rad/s), physics angle wraps ~2π per frame
            # causing stroboscopic effect (appears frozen). Use slower visual rate.
            # Target: 2 visual revolutions per second for clear motion
            batch_time = done * dt
            visual_omega = 2.0 * 2.0 * math.pi  # 2 rev/s = 4π rad/s
            visual_theta[0] += visual_omega * batch_time

            cos_t = np.cos(visual_theta[0])
            sin_t = np.sin(visual_theta[0])
            for name in self._rotor_animated:
                if name in self._mesh_actors and name in self._original_verts:
                    ov = self._original_verts[name]
                    nv = ov.copy()
                    nv[:, 1] = cos_t * ov[:, 1] - sin_t * ov[:, 2]
                    nv[:, 2] = sin_t * ov[:, 1] + cos_t * ov[:, 2]
                    self._mesh_actors[name].points = nv

            motor_theta = visual_theta[0] * self._pulley_ratio
            cos_m = np.cos(motor_theta)
            sin_m = np.sin(motor_theta)
            for name in self._motor_animated:
                if name in self._mesh_actors and name in self._original_verts:
                    ov = self._original_verts[name]
                    nv = ov.copy()
                    y_c = ov[:, 1] - self._motor_center_y
                    z_c = ov[:, 2] - self._motor_center_z
                    nv[:, 1] = cos_m * y_c - sin_m * z_c + self._motor_center_y
                    nv[:, 2] = sin_m * y_c + cos_m * z_c + self._motor_center_z
                    self._mesh_actors[name].points = nv

            if self._belt_marker_cloud is not None and self._belt_path_yz is not None:
                belt_phase = visual_theta[0] * self._belt_pitch_r
                self._belt_marker_cloud.points = self._sample_belt_markers(belt_phase)

            if self._particle_cloud is not None and self._particle_buf is not None:
                # Get all visible particles (chamber + discharge flow)
                positions, sizes = self._sim.get_all_visible_particles()
                n = min(len(positions), self._max_particles)

                # Reset buffer (unused slots invisible via NaN size)
                self._particle_buf[:] = -1000.0  # Far outside view
                self._size_buf[:] = np.nan  # NaN = invisible

                if n > 0:
                    self._particle_buf[:n] = positions[:n]
                    self._size_buf[:n] = sizes[:n] * 1000  # Convert to mm

                self._particle_cloud.points = self._particle_buf.copy()
                self._particle_cloud["Size"] = self._size_buf.copy()

            if self._plotter is not None:
                try:
                    self._plotter.render()
                except Exception:
                    pass

            # Check termination criteria
            should_stop = False
            stop_reason = ""

            # Time-based termination (always checked)
            if current_step[0] >= n_steps:
                should_stop = True
                stop_reason = f"Duration complete ({current_step[0] * dt:.1f}s)"

            # Physics-based termination (if not time mode)
            if not should_stop and self._term_mode != "time":
                physics_stop, physics_reason = self._sim.check_termination()
                if physics_stop:
                    should_stop = True
                    stop_reason = physics_reason

            if should_stop:
                self._log(f"Simulation complete: {stop_reason}")
                self._finish_run()

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(_step_batch)
        self._render_timer.start(16)

    def _on_stop(self):
        """Stop the simulation."""
        self._running = False
        if _HAS_NEW_WIDGETS and hasattr(self, "_timeline"):
            from ..widgets.milling.timeline_widget import PlaybackState
            self._timeline.set_state(PlaybackState.STOPPED)

    def _finish_run(self):
        """Finalize simulation run."""
        if self._render_timer:
            self._render_timer.stop()
            self._render_timer = None

        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            self._control_panel.set_running(False)
        else:
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._progress.setVisible(False)

        if _HAS_NEW_WIDGETS and hasattr(self, "_timeline"):
            from ..widgets.milling.timeline_widget import PlaybackState
            self._timeline.set_state(PlaybackState.STOPPED)

        # Clear particles (move outside view and set NaN size)
        if self._particle_buf is not None and self._particle_cloud is not None:
            self._particle_buf[:] = -1000.0
            self._size_buf[:] = np.nan
            self._particle_cloud.points = self._particle_buf.copy()
            self._particle_cloud["Size"] = self._size_buf.copy()

        # Reset belt markers to initial positions (phase 0)
        if self._belt_marker_cloud is not None and self._belt_path_yz is not None:
            self._belt_marker_cloud.points = self._sample_belt_markers(0.0)

        if self._plotter is not None:
            try:
                self._plotter.render()
            except Exception:
                pass

        if self._sim is None:
            return

        # Build results
        recipe_data = {}
        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            recipe_data = self._control_panel.get_recipe()
        else:
            recipe_data = {"duration_s": self._duration_spin.value()}

        duration_s = recipe_data.get("duration_s", 60)
        result = self._sim.build_result_from_engine(duration_s=duration_s, dt=0.002)
        outlet = self._sim.get_outlet_conditions()
        self._results = {"result": result, "outlet": outlet}

        # Enable results button
        if hasattr(self, "_view_results_btn"):
            self._view_results_btn.setEnabled(True)

        # Update results views
        if _HAS_NEW_WIDGETS and hasattr(self, "_results_page"):
            self._results_page.update_results(self._results)
        if _HAS_NEW_WIDGETS and hasattr(self, "_results_overlay"):
            self._results_overlay.update_results(self._results)

        self._log(f"Done. d50={outlet.d50_um:.1f} um, throughput={outlet.throughput_kg_per_hr:.0f} kg/h")
        self.simulation_finished.emit(self._results)

    def build_system(self, assembly_params: Dict[str, Any]) -> bool:
        """Build and display hammer mill geometry in the viewport."""
        self._last_params = dict(assembly_params)
        if self._plotter is None:
            self._log("3D viewport not available.")
            return False

        try:
            from airclassifier.milling import create_hammer_mill_machine, MillConfig
            import pyvista as pv

            config = MillConfig(
                rotor_rpm=assembly_params.get("mill_rotor_rpm", 3000),
                rotor_diameter_m=assembly_params.get("mill_rotor_diameter_m", 0.20),
                rotor_length_m=assembly_params.get("mill_rotor_length_m", 0.30),
                shaft_diameter_m=assembly_params.get("mill_shaft_diameter_m", 0.05),
                motor_power_kw=assembly_params.get("mill_motor_power_kw", 22.0),
                hammer_rows=int(assembly_params.get("mill_hammer_rows", 4)),
                hammers_per_row=int(assembly_params.get("mill_hammers_per_row", 4)),
                hammer_mass_kg=assembly_params.get("mill_hammer_mass_kg", 0.35),
                hammer_length_m=assembly_params.get("mill_hammer_length_m", 0.08),
                hammer_width_m=assembly_params.get("mill_hammer_width_m", 0.05),
                hammer_thickness_m=assembly_params.get("mill_hammer_thickness_m", 0.008),
                hammer_clearance_m=assembly_params.get("mill_hammer_clearance_m", 0.008),
                screen_aperture_mm=assembly_params.get("mill_screen_aperture_mm", 1.5),
                screen_open_area=assembly_params.get("mill_screen_open_area", 0.40),
                screen_inner_radius_m=assembly_params.get("mill_screen_inner_radius_m", 0.188),
                screen_thickness_m=assembly_params.get("mill_screen_thickness_m", 0.003),
                housing_inner_radius_m=assembly_params.get("mill_housing_inner_radius_m", 0.20),
                housing_length_m=assembly_params.get("mill_housing_length_m", 0.40),
                housing_wall_thickness_m=assembly_params.get("mill_housing_wall_thickness_m", 0.008),
                feed_rate_kg_per_hr=assembly_params.get("mill_feed_rate_kg_per_hr", 500.0),
                feed_chute_width_m=assembly_params.get("mill_feed_chute_width_m", 0.15),
                feed_chute_height_m=assembly_params.get("mill_feed_chute_height_m", 0.12),
                discharge_chute_width_m=assembly_params.get("mill_discharge_chute_width_m", 0.20),
                discharge_chute_height_m=assembly_params.get("mill_discharge_chute_height_m", 0.15),
            )

            assembly = create_hammer_mill_machine(config=config, build_meshes=True)
            meshes = assembly.get_component_meshes()

            self._plotter.clear()
            self._mesh_actors.clear()
            self._original_verts.clear()
            self._omega = config.rotor_angular_velocity

            dp = assembly.drive_params
            self._motor_center_y = dp.motor_y_offset_m
            self._motor_center_z = dp.motor_z_offset_m
            self._pulley_ratio = dp.mill_pulley_radius_m / dp.pulley_radius_m

            path_yz, cum_s, total_len, belt_x = self._build_belt_path(dp)
            self._belt_path_yz = path_yz
            self._belt_cum_s = cum_s
            self._belt_total_len = total_len
            self._belt_x = belt_x
            self._belt_pitch_r = dp.mill_pulley_radius_m * (1 - 0.12)

            total_verts = 0
            for name, (verts, tris, meta) in meshes.items():
                style = _MILL_STYLE.get(name)
                if style:
                    color = _rgba_to_hex(style["color"])
                    opacity = style["opacity"]
                else:
                    color = _rgba_to_hex((0.5, 0.5, 0.55))
                    opacity = 0.8

                n_tri = tris.shape[0]
                faces = np.empty((n_tri, 4), dtype=np.int64)
                faces[:, 0] = 3
                faces[:, 1:] = tris
                pd = pv.PolyData(verts.copy(), faces.ravel())
                self._plotter.add_mesh(pd, color=color, opacity=opacity, smooth_shading=True, name=name)
                self._mesh_actors[name] = pd
                self._original_verts[name] = verts.copy()
                total_verts += len(verts)

            self._plotter.add_axes()
            self._plotter.reset_camera()
            self._plotter.camera.up = (0, 1, 0)
            # Zoom out slightly to show full mill assembly
            self._plotter.camera.zoom(0.7)

            # Initialize particle buffer - place far outside view so they're invisible
            self._particle_buf = np.zeros((self._max_particles, 3), dtype=np.float32)
            self._particle_buf[:, 0] = -1000.0  # Far outside visible area
            self._size_buf = np.full(self._max_particles, np.nan, dtype=np.float32)  # NaN = invisible

            self._particle_cloud = pv.PolyData(self._particle_buf.copy())
            self._particle_cloud["Size"] = self._size_buf.copy()
            self._plotter.add_mesh(
                self._particle_cloud,
                scalars="Size",
                cmap="coolwarm_r",
                clim=[0.0, 3.0],
                point_size=8,
                render_points_as_spheres=True,
                opacity=0.9,
                nan_opacity=0.0,  # Hide NaN-sized particles
                show_scalar_bar=True,
                scalar_bar_args={"title": "Size [mm]", "position_x": 0.85, "width": 0.10},
                name="particles",
            )

            # Belt markers - initialize at proper positions along belt path
            belt_marker_pos = self._sample_belt_markers(0.0)
            self._belt_marker_cloud = pv.PolyData(belt_marker_pos)
            self._plotter.add_mesh(
                self._belt_marker_cloud,
                color="#505048",  # Dark gray matching belt
                point_size=3,
                render_points_as_spheres=False,
                opacity=0.7,
                name="belt_markers",
            )

            self._log(f"Hammer mill built: {len(meshes)} components, {total_verts:,} vertices")
            return True

        except Exception as e:
            self._log(f"Build error: {e}\n{traceback.format_exc()}")
            return False

    def resizeEvent(self, event):
        """Handle resize to update overlay position."""
        super().resizeEvent(event)
        if _HAS_NEW_WIDGETS and hasattr(self, "_results_overlay"):
            if self._results_overlay.is_visible:
                self._results_overlay.setFixedHeight(self.height())

    def sync_settings_from_params(self, params: Dict[str, Any]):
        """Sync control panel settings from assembly/wizard params.

        This method updates the simulation page's recipe controls to match
        the configuration from the assembly dialog or wizard.

        Args:
            params: Dictionary with mill configuration parameters.
        """
        # Map assembly param keys to recipe keys for control panel
        recipe = {}

        if "mill_rotor_rpm" in params:
            recipe["rotor_rpm"] = params["mill_rotor_rpm"]
        if "mill_screen_aperture_mm" in params:
            recipe["screen_aperture_mm"] = params["mill_screen_aperture_mm"]
        if "mill_seeds_feed_mass_kg" in params:
            recipe["seeds_feed_mass_kg"] = params["mill_seeds_feed_mass_kg"]
        if "mill_feed_rate_kg_per_hr" in params:
            recipe["feed_rate_kg_per_hr"] = params["mill_feed_rate_kg_per_hr"]

        # Apply to new control panel
        if _HAS_NEW_WIDGETS and hasattr(self, "_control_panel"):
            self._control_panel.set_recipe(recipe)
        else:
            # Fallback to legacy spinboxes
            if "mill_rotor_rpm" in params and hasattr(self, "_rpm_spin"):
                self._rpm_spin.setValue(params["mill_rotor_rpm"])
            if "mill_screen_aperture_mm" in params and hasattr(self, "_aperture_spin"):
                self._aperture_spin.setValue(params["mill_screen_aperture_mm"])
            if "mill_seeds_feed_mass_kg" in params and hasattr(self, "_seeds_feed_mass_spin"):
                self._seeds_feed_mass_spin.setValue(params["mill_seeds_feed_mass_kg"])
            if "mill_feed_rate_kg_per_hr" in params and hasattr(self, "_feed_spin"):
                self._feed_spin.setValue(params["mill_feed_rate_kg_per_hr"])

        # Store params for geometry rebuild
        self._last_params.update(params)

    def cleanup(self):
        """Stop simulation and release resources."""
        self._running = False
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
