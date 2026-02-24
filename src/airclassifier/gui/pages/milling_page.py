"""
Milling Page — Hammer Mill Full-Page View
=========================================

Full-page simulation and visualization for the hammer mill (pin mill).
Uses QStackedWidget: Simulation View (3D + controls) and Results View (KPIs, PSD, export).

Architecture::
    MillingPage
        └── QStackedWidget
            ├── Page 0: Simulation View (viewport + control panel)
            └── Page 1: Results View (KPI cards, PSD plot, export)
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtWidgets import (
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
# Matches examples/visualize_hammer_mill.py COMPONENT_STYLE.
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


class _StatCard(QFrame):
    """KPI display card."""

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
        self._value_label = QLabel("--")
        color = accent or COLORS.ACCENT
        self._value_label.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {color};"
            " border: none; background: transparent;"
        )
        layout.addWidget(self._value_label)
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        layout.addWidget(self._title_label)

    def set_value(self, text: str):
        self._value_label.setText(text)


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
        self._render_timer: Optional[QTimer] = None
        self._plotter = None
        self._results: Optional[Dict[str, Any]] = None
        self._last_params: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._view_stack = QStackedWidget()
        root.addWidget(self._view_stack)

        sim_view = self._build_simulation_view()
        self._view_stack.addWidget(sim_view)

        results_view = self._build_results_view()
        self._view_stack.addWidget(results_view)

        self._view_stack.setCurrentIndex(0)

    def _build_simulation_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._viewport_widget = self._build_viewport()
        top_splitter.addWidget(self._viewport_widget)
        ctrl_widget = self._build_control_panel()
        top_splitter.addWidget(ctrl_widget)
        top_splitter.setSizes([700, 350])
        layout.addWidget(top_splitter, 1)

        bottom_bar = QFrame()
        bottom_bar.setStyleSheet(f"QFrame {{ background: {COLORS.BG_ELEVATED}; border-top: 1px solid {COLORS.BORDER}; }}")
        bar_layout = QHBoxLayout(bottom_bar)
        self._view_results_btn = QPushButton("View Full Results")
        self._view_results_btn.setEnabled(False)
        self._view_results_btn.clicked.connect(self._show_results_view)
        bar_layout.addWidget(self._view_results_btn)
        layout.addWidget(bottom_bar)

        return page

    def _build_viewport(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        if _HAS_PYVISTA:
            self._plotter = QtInteractor(container)
            self._plotter.set_background(
                color="#3a3a44", top="#282830"
            )
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.add_axes()
            layout.addWidget(self._plotter.interactor)
            self._plotter.add_text(
                "Hammer Mill (Pin Mill)\nConfigure and click Run to start",
                position="upper_left", font_size=10, color=COLORS.TEXT_MUTED, name="placeholder_text",
            )
        else:
            lbl = QLabel("PyVista not available. Install: pip install pyvista pyvistaqt")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; padding: 40px;")
            layout.addWidget(lbl)
        return container

    def _build_control_panel(self) -> QWidget:
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

        kpi_grid = QGridLayout()
        self._card_throughput = _StatCard("Throughput (kg/h)")
        self._card_d50 = _StatCard("d50 (µm)")
        self._card_power = _StatCard("Power (kW)")
        kpi_grid.addWidget(self._card_throughput, 0, 0)
        kpi_grid.addWidget(self._card_d50, 0, 1)
        kpi_grid.addWidget(self._card_power, 1, 0)
        layout.addLayout(kpi_grid)

        g = QGroupBox("Recipe")
        f = QFormLayout(g)
        self._rpm_spin = QDoubleSpinBox()
        self._rpm_spin.setRange(500, 6000)
        self._rpm_spin.setValue(3000)
        self._rpm_spin.setSuffix(" rpm")
        f.addRow("Rotor RPM:", self._rpm_spin)
        self._aperture_spin = QDoubleSpinBox()
        self._aperture_spin.setRange(0.5, 5.0)
        self._aperture_spin.setValue(1.5)
        self._aperture_spin.setDecimals(2)
        self._aperture_spin.setSuffix(" mm")
        f.addRow("Screen aperture:", self._aperture_spin)
        self._feed_spin = QDoubleSpinBox()
        self._feed_spin.setRange(10, 2000)
        self._feed_spin.setValue(500)
        self._feed_spin.setSuffix(" kg/h")
        f.addRow("Feed rate:", self._feed_spin)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(1, 600)
        self._duration_spin.setValue(60)
        self._duration_spin.setSuffix(" s")
        f.addRow("Duration:", self._duration_spin)
        layout.addWidget(g)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(f"font-family: monospace; font-size: 9pt; background: {COLORS.BG_DARKEST};")
        layout.addWidget(self._log_text)

        return container

    def _build_results_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        back_btn = QPushButton("← Back to Simulation")
        back_btn.clicked.connect(self._show_simulation_view)
        header.addWidget(back_btn)
        self._results_title = QLabel("Milling Results")
        self._results_title.setStyleSheet(f"font-size: 14pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};")
        header.addWidget(self._results_title, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        kpi_section = QGridLayout()
        self._rc_throughput = _StatCard("Throughput (kg/h)")
        self._rc_d50 = _StatCard("d50 (µm)")
        self._rc_power = _StatCard("Power (kW)")
        self._rc_specific = _StatCard("Specific energy (kWh/t)")
        kpi_section.addWidget(self._rc_throughput, 0, 0)
        kpi_section.addWidget(self._rc_d50, 0, 1)
        kpi_section.addWidget(self._rc_power, 1, 0)
        kpi_section.addWidget(self._rc_specific, 1, 1)
        content_layout.addLayout(kpi_section)

        if _HAS_MATPLOTLIB:
            self._psd_fig = Figure(figsize=(6, 3), facecolor=COLORS.BG_DARK)
            self._psd_canvas = FigureCanvas(self._psd_fig)
            self._psd_canvas.setMinimumHeight(220)
            content_layout.addWidget(QLabel("Particle size distribution"))
            content_layout.addWidget(self._psd_canvas)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return page

    def _log(self, msg: str):
        if hasattr(self, "_log_text") and self._log_text:
            self._log_text.append(msg)
            self._log_text.verticalScrollBar().setValue(self._log_text.verticalScrollBar().maximum())

    def _show_results_view(self):
        self._view_stack.setCurrentIndex(1)
        if self._results:
            self._update_results_display()

    def _show_simulation_view(self):
        self._view_stack.setCurrentIndex(0)

    def _update_results_display(self):
        r = self._results
        if not r:
            return
        outlet = r.get("outlet")
        result = r.get("result")
        if outlet:
            self._rc_throughput.set_value(f"{outlet.throughput_kg_per_hr:.0f}")
            self._rc_d50.set_value(f"{outlet.d50_um:.1f}")
            self._rc_power.set_value(f"{outlet.power_kw:.2f}")
            self._rc_specific.set_value(f"{outlet.specific_energy_kwh_per_t:.2f}")
        if _HAS_MATPLOTLIB and result and len(result.psd_size_classes_m) > 0:
            self._psd_fig.clear()
            ax = self._psd_fig.add_subplot(111)
            ax.set_facecolor("#1e1e24")
            x_um = result.psd_size_classes_m * 1e6
            ax.bar(x_um[:-1], result.psd_mass_fractions, width=np.diff(x_um) * 0.9, color=COLORS.ACCENT, alpha=0.8)
            ax.set_xlabel("Size (µm)")
            ax.set_ylabel("Mass fraction")
            ax.set_title("PSD (discharge)")
            self._psd_canvas.draw()

    def run_simulation(self):
        """Start the mill simulation (called by MainWindow F5 when in Milling mode)."""
        if not self._running:
            self._on_run()

    def _on_run(self):
        try:
            from airclassifier.milling import HammerMillSimulator, MillConfig, MillRecipe
        except ImportError as e:
            self._log(f"Milling module not available: {e}")
            return

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._running = True
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._progress.setMaximum(100)

        p = self._last_params
        config = MillConfig(
            rotor_rpm=p.get("mill_rotor_rpm", self._rpm_spin.value()),
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
            screen_aperture_mm=p.get("mill_screen_aperture_mm", self._aperture_spin.value()),
            screen_open_area=p.get("mill_screen_open_area", 0.40),
            screen_inner_radius_m=p.get("mill_screen_inner_radius_m", 0.188),
            screen_thickness_m=p.get("mill_screen_thickness_m", 0.003),
            housing_inner_radius_m=p.get("mill_housing_inner_radius_m", 0.20),
            housing_length_m=p.get("mill_housing_length_m", 0.40),
            housing_wall_thickness_m=p.get("mill_housing_wall_thickness_m", 0.008),
            feed_rate_kg_per_hr=p.get("mill_feed_rate_kg_per_hr", self._feed_spin.value()),
            feed_chute_width_m=p.get("mill_feed_chute_width_m", 0.15),
            feed_chute_height_m=p.get("mill_feed_chute_height_m", 0.12),
            discharge_chute_width_m=p.get("mill_discharge_chute_width_m", 0.20),
            discharge_chute_height_m=p.get("mill_discharge_chute_height_m", 0.15),
        )
        recipe = MillRecipe(
            rotor_rpm=config.rotor_rpm,
            screen_aperture_mm=config.screen_aperture_mm,
            feed_rate_kg_per_hr=config.feed_rate_kg_per_hr,
            run_duration_s=self._duration_spin.value(),
        )
        self._sim = HammerMillSimulator(config=config)
        self._sim.load_recipe(recipe)
        self._sim.initialize(initial_holdup_kg=0.01)

        duration_s = self._duration_spin.value()
        dt = 0.001
        n_steps = int(duration_s / dt)
        self._progress.setMaximum(n_steps)
        self._log(f"Running mill: {duration_s:.0f} s, dt={dt}")

        def _step_batch():
            if not self._running or self._sim is None:
                return
            done = 0
            batch = min(100, n_steps - self._progress.value())
            for _ in range(batch):
                if not self._running:
                    break
                self._sim.step(dt)
                done += 1
            self._progress.setValue(self._progress.value() + done)
            state = self._sim.engine.history[-1] if self._sim.engine.history else None
            if state:
                self._card_throughput.set_value(f"{state.discharge_rate_kg_per_s * 3600:.0f}")
                self._card_d50.set_value(f"{state.d50_m * 1e6:.0f}")
                self._card_power.set_value(f"{state.power_kw:.2f}")
            if self._progress.value() >= n_steps:
                self._finish_run()

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(_step_batch)
        self._render_timer.start(0)

    def _on_stop(self):
        self._running = False

    def _finish_run(self):
        if self._render_timer:
            self._render_timer.stop()
            self._render_timer = None
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)

        if self._sim is None:
            return
        duration_s = self._duration_spin.value()
        result = self._sim.build_result_from_engine(duration_s=duration_s, dt=0.001)
        outlet = self._sim.get_outlet_conditions()
        self._results = {"result": result, "outlet": outlet}
        self._view_results_btn.setEnabled(True)
        self._log(f"Done. d50={outlet.d50_um:.1f} µm, throughput={outlet.throughput_kg_per_hr:.0f} kg/h")
        self.simulation_finished.emit(self._results)

    def build_system(self, assembly_params: Dict[str, Any]) -> bool:
        """Build and display hammer mill geometry in the viewport."""
        self._last_params = dict(assembly_params)
        if self._plotter is None:
            self._log("3D viewport not available.")
            return False
        try:
            from airclassifier.milling import (
                create_hammer_mill_machine,
                MillConfig,
            )
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
                self._plotter.add_mesh(pd, color=color, opacity=opacity, smooth_shading=True)
                total_verts += len(verts)
            self._plotter.add_axes()
            self._plotter.camera.up = (0, 1, 0)
            self._plotter.reset_camera()
            self._log(f"Hammer mill built: {len(meshes)} components, {total_verts:,} vertices")
            return True
        except Exception as e:
            self._log(f"Build error: {e}\n{traceback.format_exc()}")
            return False

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


