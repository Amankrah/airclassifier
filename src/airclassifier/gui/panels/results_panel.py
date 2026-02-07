"""
Results Panel
=============

Panel for displaying simulation results from ClassificationFlowPhysicsSimulator.
Matches the output format of ``run_classification_flow.py``.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QFrame, QScrollArea, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont

from ..theme import COLORS

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --------------------------------------------------------------------------
# Reusable stat card
# --------------------------------------------------------------------------

class _ResultCard(QFrame):
    """Single KPI value with label."""

    def __init__(self, label: str, initial: str = "--", accent: str = COLORS.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 5px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(1)

        self._val = QLabel(initial)
        self._val.setStyleSheet(
            f"font-size: 13pt; font-weight: 700; color: {accent};"
            " border:none; background:transparent;"
        )
        layout.addWidget(self._val)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border:none; background:transparent;"
        )
        layout.addWidget(lbl)

    def set_value(self, text: str):
        self._val.setText(text)


def _scrollable(w: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setFrameShape(QFrame.Shape.NoFrame)
    s.setWidget(w)
    return s


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

class ResultsPanel(QWidget):
    """
    Displays results from ClassificationFlowPhysicsSimulator.

    Tabs:
    - Summary: KPI cards + separation breakdown table + cyclone stats
    - Separation Chart: matplotlib bar chart of collection points
    - Collection Points: table matching CLI output
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._results: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(_scrollable(self._create_summary_tab()), "Summary")

        if HAS_MATPLOTLIB:
            tabs.addTab(self._create_chart_tab(), "Separation Chart")

        tabs.addTab(self._create_collection_tab(), "Collection Points")

    # ================================================================
    # Summary tab
    # ================================================================

    def _create_summary_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # ---- KPI cards row 1 ----
        grid = QGridLayout()
        grid.setSpacing(6)

        self.rc_total = _ResultCard("Total Particles", "--", COLORS.TEXT_PRIMARY)
        grid.addWidget(self.rc_total, 0, 0)

        self.rc_efficiency = _ResultCard("Fines / Collected", "--", COLORS.ACCENT)
        grid.addWidget(self.rc_efficiency, 0, 1)

        self.rc_wall_time = _ResultCard("Wall Time", "--", COLORS.TEXT_SECONDARY)
        grid.addWidget(self.rc_wall_time, 0, 2)

        self.rc_fines = _ResultCard("Fines Collected", "--", COLORS.SUCCESS)
        grid.addWidget(self.rc_fines, 1, 0)

        self.rc_coarse = _ResultCard("Coarse Collected", "--", COLORS.WARNING)
        grid.addWidget(self.rc_coarse, 1, 1)

        self.rc_active = _ResultCard("Still Active", "--", COLORS.INFO)
        grid.addWidget(self.rc_active, 1, 2)

        layout.addLayout(grid)

        # ---- Operating conditions ----
        op_group = QGroupBox("Operating Conditions")
        op_form = QFormLayout(op_group)

        self.lbl_air_flow = QLabel("--")
        op_form.addRow("Air Flow:", self.lbl_air_flow)
        self.lbl_blower = QLabel("--")
        op_form.addRow("Blower:", self.lbl_blower)
        self.lbl_wheel = QLabel("--")
        op_form.addRow("Wheel:", self.lbl_wheel)
        self.lbl_mode = QLabel("--")
        op_form.addRow("Mode:", self.lbl_mode)

        layout.addWidget(op_group)

        # ---- Separation breakdown table ----
        self.sep_table = QTableWidget()
        self.sep_table.setColumnCount(3)
        self.sep_table.setHorizontalHeaderLabels(["Collection Point", "Particles", "%"])
        self.sep_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sep_table.setAlternatingRowColors(True)
        self.sep_table.verticalHeader().setVisible(False)
        self.sep_table.setFixedHeight(220)

        self._sep_rows = [
            ("Zigzag coarse (starch)", "coarse"),
            ("Wheel coarse (starch)", "wheel_coarse"),
            ("Cyclone 1 (fines 1)", "cyclone_1"),
            ("Cyclone 2 (fines 2)", "cyclone_2"),
            ("Cyclone 3 (PROTEIN)", "cyclone_3_protein"),
            ("Bag filter", "bagfilter"),
            ("Escaped (loss)", "escaped"),
            ("Still active", "active"),
        ]
        self.sep_table.setRowCount(len(self._sep_rows))
        for i, (label, _) in enumerate(self._sep_rows):
            self.sep_table.setItem(i, 0, QTableWidgetItem(label))
            self.sep_table.setItem(i, 1, QTableWidgetItem("--"))
            self.sep_table.setItem(i, 2, QTableWidgetItem("--"))

        layout.addWidget(self.sep_table)

        # ---- Cyclone particle size stats ----
        cyc_group = QGroupBox("Cyclone Particle Sizes (design d50 vs actual)")
        cyc_form = QFormLayout(cyc_group)

        self.lbl_cy1 = QLabel("--")
        cyc_form.addRow("Primary (Cy1):", self.lbl_cy1)
        self.lbl_cy2 = QLabel("--")
        cyc_form.addRow("Secondary (Cy2):", self.lbl_cy2)
        self.lbl_cy3 = QLabel("--")
        cyc_form.addRow("Tertiary (Cy3):", self.lbl_cy3)

        layout.addWidget(cyc_group)

        # ---- Empty hint ----
        self._empty_hint = QLabel("Run a simulation to see results here")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10pt; padding: 16px;")
        layout.addWidget(self._empty_hint)

        return widget

    # ================================================================
    # Chart tab (matplotlib)
    # ================================================================

    def _create_chart_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.figure.patch.set_facecolor(COLORS.BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        export_btn = QPushButton("Export Plot")
        export_btn.setProperty("cssClass", "ghost")
        export_btn.clicked.connect(self._export_plot)
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

        self._draw_empty_chart()
        return widget

    def _draw_empty_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS.BG_DARKEST)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY)
        for sp in ax.spines.values():
            sp.set_color(COLORS.BORDER)
        ax.set_title("No Results", color=COLORS.TEXT_MUTED, fontsize=11)
        ax.grid(True, alpha=0.15, color=COLORS.BORDER)
        self.canvas.draw()

    # ================================================================
    # Collection Points tab
    # ================================================================

    def _create_collection_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.collection_table = QTableWidget()
        self.collection_table.setColumnCount(4)
        self.collection_table.setHorizontalHeaderLabels([
            "Collection Point", "Particles", "% of Total", "Cyclone d50 / mean"
        ])
        self.collection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.collection_table.setAlternatingRowColors(True)
        self.collection_table.verticalHeader().setVisible(False)
        layout.addWidget(self.collection_table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        export_btn = QPushButton("Export CSV")
        export_btn.setProperty("cssClass", "ghost")
        export_btn.clicked.connect(lambda: self._export_table(self.collection_table, "collection_points.csv"))
        btn_row.addWidget(export_btn)
        layout.addLayout(btn_row)

        return widget

    # ================================================================
    # Data update
    # ================================================================

    @Slot(dict)
    def set_results(self, results: Dict[str, Any]):
        """Accept results dict from SimulationControlPanel."""
        self._results = results
        self._empty_hint.hide()
        self._update_summary()
        self._update_collection_table()
        if HAS_MATPLOTLIB:
            self._draw_separation_chart()

    def _update_summary(self):
        r = self._results
        if not r:
            return

        n_total = r.get("num_particles", 0)
        fines = r.get("fines_collected", 0)
        coarse = r.get("coarse_collected", 0)
        collected = fines + coarse
        active = r.get("active", 0)
        eff = r.get("separation_efficiency", 0)
        wall = r.get("wall_time_s", 0)

        self.rc_total.set_value(f"{n_total:,}")
        self.rc_efficiency.set_value(f"{eff:.1f}%")
        self.rc_wall_time.set_value(f"{wall:.1f}s")
        self.rc_fines.set_value(f"{fines:,}")
        self.rc_coarse.set_value(f"{coarse:,}")
        self.rc_active.set_value(f"{active:,}")

        # Operating conditions
        q = r.get("air_flow_m3h", 0)
        self.lbl_air_flow.setText(f"{q:.0f} m\u00b3/h ({r.get('air_flow_m3s', 0):.3f} m\u00b3/s)")
        brpm = r.get("blower_rpm", 0)
        self.lbl_blower.setText(f"{brpm:.0f} RPM" if brpm > 0 else "Direct flow")
        self.lbl_wheel.setText(f"{r.get('wheel_rpm', 0):.0f} RPM")
        self.lbl_mode.setText("Full System" if r.get("use_preclassification") else "Wheel-Only")

        # Separation breakdown table
        for i, (label, key) in enumerate(self._sep_rows):
            count = r.get(key, 0)
            pct = 100.0 * count / n_total if n_total > 0 else 0
            self.sep_table.setItem(i, 1, QTableWidgetItem(f"{count:,}"))
            self.sep_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))

        # Cyclone particle size stats
        cs = r.get("cyclone_stats", {})
        for lbl_widget, key in [(self.lbl_cy1, "cyclone_1"), (self.lbl_cy2, "cyclone_2"), (self.lbl_cy3, "cyclone_3_protein")]:
            st = cs.get(key, {})
            n = st.get("count", 0)
            d50 = st.get("design_d50_um")
            mean = st.get("mean_d_um")
            median = st.get("median_d_um")
            parts = [f"N={n:,}"]
            if d50:
                parts.append(f"design d50={d50:.0f} \u00b5m")
            if mean:
                parts.append(f"mean={mean:.1f} \u00b5m")
            if median:
                parts.append(f"median={median:.1f} \u00b5m")
            lbl_widget.setText("  ".join(parts))

    def _update_collection_table(self):
        r = self._results
        if not r:
            return

        n_total = r.get("num_particles", 0)
        cs = r.get("cyclone_stats", {})

        rows = [
            ("Zigzag coarse", r.get("coarse", 0), None),
            ("Wheel coarse", r.get("wheel_coarse", 0), None),
            ("Cyclone 1 (primary)", r.get("cyclone_1", 0), cs.get("cyclone_1")),
            ("Cyclone 2 (secondary)", r.get("cyclone_2", 0), cs.get("cyclone_2")),
            ("Cyclone 3 (protein)", r.get("cyclone_3_protein", 0), cs.get("cyclone_3_protein")),
            ("Bag filter", r.get("bagfilter", 0), None),
            ("Escaped", r.get("escaped", 0), None),
            ("Still active", r.get("active", 0), None),
        ]

        self.collection_table.setRowCount(len(rows))
        for i, (label, count, cyc_st) in enumerate(rows):
            self.collection_table.setItem(i, 0, QTableWidgetItem(label))
            self.collection_table.setItem(i, 1, QTableWidgetItem(f"{count:,}"))
            pct = 100.0 * count / n_total if n_total > 0 else 0
            self.collection_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))
            if cyc_st:
                d50 = cyc_st.get("design_d50_um")
                mean = cyc_st.get("mean_d_um")
                median = cyc_st.get("median_d_um")
                parts = []
                if d50:
                    parts.append(f"d50={d50:.0f}\u00b5m")
                if mean:
                    parts.append(f"mean={mean:.1f}\u00b5m")
                if median:
                    parts.append(f"med={median:.1f}\u00b5m")
                self.collection_table.setItem(i, 3, QTableWidgetItem("  ".join(parts)))
            else:
                self.collection_table.setItem(i, 3, QTableWidgetItem(""))

    def _draw_separation_chart(self):
        r = self._results
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS.BG_DARKEST)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY)
        for sp in ax.spines.values():
            sp.set_color(COLORS.BORDER)

        if not r:
            ax.set_title("No Results", color=COLORS.TEXT_MUTED)
            self.canvas.draw()
            return

        labels = ["Zigzag\ncoarse", "Wheel\ncoarse", "Cyclone 1", "Cyclone 2", "Cyclone 3\n(protein)", "Bag\nfilter"]
        keys = ["coarse", "wheel_coarse", "cyclone_1", "cyclone_2", "cyclone_3_protein", "bagfilter"]
        values = [r.get(k, 0) for k in keys]
        colors = [COLORS.WARNING, "#e8a838", COLORS.CAT_AIR, COLORS.ACCENT, COLORS.SUCCESS, COLORS.CAT_FEED]

        bars = ax.bar(range(len(labels)), values, color=colors, edgecolor=COLORS.BORDER, linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color=COLORS.TEXT_SECONDARY, fontsize=8)
        ax.set_ylabel("Particles", color=COLORS.TEXT_SECONDARY)

        q_h = r.get("air_flow_m3h", 0)
        wrpm = r.get("wheel_rpm", 0)
        ax.set_title(
            f"Separation Results  |  Q={q_h:.0f} m\u00b3/h  Wheel={wrpm:.0f} RPM",
            color=COLORS.TEXT_PRIMARY, fontsize=10,
        )
        ax.grid(True, alpha=0.15, color=COLORS.BORDER, axis='y')

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,}", ha='center', va='bottom',
                    color=COLORS.TEXT_PRIMARY, fontsize=8,
                )

        self.figure.tight_layout()
        self.canvas.draw()

    # ================================================================
    # Export
    # ================================================================

    def _export_plot(self):
        from PySide6.QtWidgets import QFileDialog
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", str(Path.home() / "separation_results.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if fp:
            self.figure.savefig(fp, facecolor=COLORS.BG_DARK, edgecolor='none', dpi=150)

    def _export_table(self, table: QTableWidget, default_name: str):
        from PySide6.QtWidgets import QFileDialog
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Data", str(Path.home() / default_name), "CSV (*.csv)",
        )
        if fp:
            with open(fp, 'w') as f:
                headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
                f.write(",".join(headers) + "\n")
                for row in range(table.rowCount()):
                    cells = [table.item(row, c).text() if table.item(row, c) else "" for c in range(table.columnCount())]
                    f.write(",".join(cells) + "\n")

    def export_results(self, file_path: str):
        import json as _json
        p = Path(file_path)
        if p.suffix == '.json':
            with open(p, 'w') as f:
                _json.dump(self._results, f, indent=2, default=str)
        elif p.suffix == '.csv':
            self._export_table(self.collection_table, str(p))
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

    def clear(self):
        self._results = {}
        if HAS_MATPLOTLIB:
            self._draw_empty_chart()
        if hasattr(self, '_empty_hint'):
            self._empty_hint.show()
