"""
Results Panel
=============

Panel for displaying and analyzing simulation results.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QFrame, QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont

from ..theme import COLORS

# Try to import matplotlib for plotting
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --------------------------------------------------------------------------
# Reusable summary stat widget
# --------------------------------------------------------------------------

class _SummaryItem(QFrame):
    """Compact label/value pair shown in summary grid."""

    def __init__(self, label: str, initial: str = "--", parent=None):
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
        self._val.setStyleSheet(f"font-size: 13pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY}; border:none; background:transparent;")
        layout.addWidget(self._val)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; border:none; background:transparent;")
        layout.addWidget(lbl)

    def set_value(self, text: str):
        self._val.setText(text)


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

class ResultsPanel(QWidget):
    """
    Panel for displaying simulation results.

    Shows:
    - Summary statistics (card grid)
    - Separation efficiency curves
    - Particle distribution plots
    - Tabular data
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

        summary_tab = self._create_summary_tab()
        tabs.addTab(summary_tab, "Summary")

        if HAS_MATPLOTLIB:
            curves_tab = self._create_curves_tab()
            tabs.addTab(curves_tab, "Efficiency Curves")

        data_tab = self._create_data_tab()
        tabs.addTab(data_tab, "Particle Data")

        collection_tab = self._create_collection_tab()
        tabs.addTab(collection_tab, "Collection Points")

    # ---------------------------------------------------------------- summary

    def _create_summary_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Performance cards
        from PySide6.QtWidgets import QGridLayout
        perf_grid = QGridLayout()
        perf_grid.setSpacing(6)

        self.si_total = _SummaryItem("Total Particles")
        perf_grid.addWidget(self.si_total, 0, 0)

        self.si_efficiency = _SummaryItem("Separation Efficiency")
        perf_grid.addWidget(self.si_efficiency, 0, 1)

        self.si_recovery = _SummaryItem("Protein Recovery")
        perf_grid.addWidget(self.si_recovery, 0, 2)

        self.si_purity = _SummaryItem("Protein Purity")
        perf_grid.addWidget(self.si_purity, 1, 0)

        self.si_throughput = _SummaryItem("Throughput")
        perf_grid.addWidget(self.si_throughput, 1, 1)

        self.si_mass_err = _SummaryItem("Mass Balance Error")
        perf_grid.addWidget(self.si_mass_err, 1, 2)

        layout.addLayout(perf_grid)

        # Cut sizes group
        cut_group = QGroupBox("Cut Sizes (d50)")
        cut_layout = QFormLayout(cut_group)

        self.zigzag_d50_label = QLabel("--")
        cut_layout.addRow("Zigzag Classifier:", self.zigzag_d50_label)

        self.wheel_d50_label = QLabel("--")
        cut_layout.addRow("Wheel Classifier:", self.wheel_d50_label)

        self.cyclone_primary_d50_label = QLabel("--")
        cut_layout.addRow("Primary Cyclone:", self.cyclone_primary_d50_label)

        self.cyclone_secondary_d50_label = QLabel("--")
        cut_layout.addRow("Secondary Cyclone:", self.cyclone_secondary_d50_label)

        self.cyclone_tertiary_d50_label = QLabel("--")
        cut_layout.addRow("Tertiary Cyclone:", self.cyclone_tertiary_d50_label)

        layout.addWidget(cut_group)

        # Empty-state hint
        self._empty_hint = QLabel("Run a simulation to see results here")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10pt; padding: 20px;")
        layout.addWidget(self._empty_hint)

        layout.addStretch()
        return widget

    # --------------------------------------------------------------- curves

    def _create_curves_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Grade Efficiency Curve",
            "Tromp Curve",
            "Particle Size Distribution",
            "Collection vs Size",
        ])
        self.plot_type_combo.currentTextChanged.connect(self._update_plot)
        controls_layout.addWidget(QLabel("Plot Type:"))
        controls_layout.addWidget(self.plot_type_combo)
        controls_layout.addStretch()

        export_btn = QPushButton("Export Plot")
        export_btn.setProperty("cssClass", "ghost")
        export_btn.clicked.connect(self._export_plot)
        controls_layout.addWidget(export_btn)

        layout.addLayout(controls_layout)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor(COLORS.BG_DARK)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self._create_empty_plot()
        return widget

    def _create_empty_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS.BG_DARKEST)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(COLORS.BORDER)
        ax.set_xlabel("Particle Size [um]", color=COLORS.TEXT_SECONDARY)
        ax.set_ylabel("Efficiency [%]", color=COLORS.TEXT_SECONDARY)
        ax.set_title("No Results Available", color=COLORS.TEXT_MUTED, fontsize=11)
        ax.grid(True, alpha=0.15, color=COLORS.BORDER)
        self.canvas.draw()

    # ------------------------------------------------------------ data table

    def _create_data_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.particle_table = QTableWidget()
        self.particle_table.setColumnCount(6)
        self.particle_table.setHorizontalHeaderLabels([
            "ID", "Size [um]", "Density [kg/m3]", "Type", "Final Zone", "Residence Time [s]"
        ])
        self.particle_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.particle_table.setAlternatingRowColors(True)
        self.particle_table.verticalHeader().setVisible(False)
        layout.addWidget(self.particle_table)

        export_layout = QHBoxLayout()
        export_layout.addStretch()
        export_btn = QPushButton("Export to CSV")
        export_btn.setProperty("cssClass", "ghost")
        export_btn.clicked.connect(lambda: self._export_table(self.particle_table, "particle_data.csv"))
        export_layout.addWidget(export_btn)
        layout.addLayout(export_layout)

        return widget

    # ------------------------------------------------------- collection table

    def _create_collection_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.collection_table = QTableWidget()
        self.collection_table.setColumnCount(5)
        self.collection_table.setHorizontalHeaderLabels([
            "Collection Point", "Particles", "Mass [g]", "Avg Size [um]", "Composition"
        ])
        self.collection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.collection_table.setAlternatingRowColors(True)
        self.collection_table.verticalHeader().setVisible(False)
        layout.addWidget(self.collection_table)

        collection_points = [
            "Zigzag Coarse",
            "Dropout Hopper",
            "Wheel Coarse",
            "Primary Cyclone",
            "Secondary Cyclone",
            "Tertiary Cyclone",
            "Bag Filter",
            "Clean Air Exit",
        ]
        self.collection_table.setRowCount(len(collection_points))
        for i, point in enumerate(collection_points):
            self.collection_table.setItem(i, 0, QTableWidgetItem(point))
            for j in range(1, 5):
                self.collection_table.setItem(i, j, QTableWidgetItem("--"))

        return widget

    # ============================================================
    # Data update helpers
    # ============================================================

    def set_results(self, results: Dict[str, Any]):
        self._results = results
        self._empty_hint.hide()
        self._update_summary()
        self._update_plot(self.plot_type_combo.currentText() if HAS_MATPLOTLIB else None)
        self._update_data_tables()

    def _update_summary(self):
        if not self._results:
            return

        self.si_total.set_value(str(self._results.get("total_particles", "--")))
        self.si_efficiency.set_value(f"{self._results.get('separation_efficiency', 0):.1f}%")
        self.si_recovery.set_value(f"{self._results.get('protein_recovery', 0):.1f}%")
        self.si_purity.set_value(f"{self._results.get('protein_purity', 0):.1f}%")
        self.si_throughput.set_value(f"{self._results.get('throughput_kg_h', 0):.1f} kg/h")

        mass_balance = self._results.get("mass_balance", {})
        error = mass_balance.get('error', 0)
        self.si_mass_err.set_value(f"{error:.2f}%")

        cut_sizes = self._results.get("cut_sizes", {})
        self.zigzag_d50_label.setText(f"{cut_sizes.get('zigzag', 0) * 1e6:.1f} um")
        self.wheel_d50_label.setText(f"{cut_sizes.get('wheel', 0) * 1e6:.1f} um")
        self.cyclone_primary_d50_label.setText(f"{cut_sizes.get('cyclone_primary', 0) * 1e6:.1f} um")
        self.cyclone_secondary_d50_label.setText(f"{cut_sizes.get('cyclone_secondary', 0) * 1e6:.1f} um")
        self.cyclone_tertiary_d50_label.setText(f"{cut_sizes.get('cyclone_tertiary', 0) * 1e6:.1f} um")

    def _update_plot(self, plot_type: Optional[str]):
        if not HAS_MATPLOTLIB or not plot_type:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS.BG_DARKEST)
        ax.tick_params(colors=COLORS.TEXT_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(COLORS.BORDER)

        if not self._results:
            ax.set_title("No Results Available", color=COLORS.TEXT_MUTED)
            self.canvas.draw()
            return

        if plot_type == "Grade Efficiency Curve":
            self._plot_grade_efficiency(ax)
        elif plot_type == "Tromp Curve":
            self._plot_tromp_curve(ax)
        elif plot_type == "Particle Size Distribution":
            self._plot_psd(ax)
        elif plot_type == "Collection vs Size":
            self._plot_collection_by_size(ax)

        ax.legend(facecolor=COLORS.BG_ELEVATED, edgecolor=COLORS.BORDER, labelcolor=COLORS.TEXT_PRIMARY)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_grade_efficiency(self, ax):
        data = self._results.get("grade_efficiency", {})
        sizes = data.get("sizes", [])
        efficiency = data.get("efficiency", [])
        if sizes and efficiency:
            sizes_um = [s * 1e6 for s in sizes]
            ax.semilogx(sizes_um, efficiency, 'o-', color=COLORS.ACCENT, label='Overall', linewidth=2, markersize=4)
            ax.axhline(y=50, color=COLORS.DANGER, linestyle='--', alpha=0.5, label='d50')
        ax.set_xlabel("Particle Size [um]", color=COLORS.TEXT_SECONDARY)
        ax.set_ylabel("Grade Efficiency [%]", color=COLORS.TEXT_SECONDARY)
        ax.set_title("Grade Efficiency Curve", color=COLORS.TEXT_PRIMARY)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.15, color=COLORS.BORDER)

    def _plot_tromp_curve(self, ax):
        data = self._results.get("tromp_curve", {})
        sizes = data.get("sizes", [])
        partition = data.get("partition", [])
        if sizes and partition:
            sizes_um = [s * 1e6 for s in sizes]
            ax.semilogx(sizes_um, partition, 'o-', color=COLORS.WARNING, label='Partition', linewidth=2, markersize=4)
            ax.axhline(y=50, color=COLORS.DANGER, linestyle='--', alpha=0.5)
        ax.set_xlabel("Particle Size [um]", color=COLORS.TEXT_SECONDARY)
        ax.set_ylabel("To Coarse [%]", color=COLORS.TEXT_SECONDARY)
        ax.set_title("Tromp Curve (Partition)", color=COLORS.TEXT_PRIMARY)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.15, color=COLORS.BORDER)

    def _plot_psd(self, ax):
        data = self._results.get("psd", {})
        for stream, values in data.items():
            sizes = values.get("sizes", [])
            freq = values.get("frequency", [])
            if sizes and freq:
                sizes_um = [s * 1e6 for s in sizes]
                ax.plot(sizes_um, freq, '-', label=stream, linewidth=2)
        ax.set_xlabel("Particle Size [um]", color=COLORS.TEXT_SECONDARY)
        ax.set_ylabel("Frequency [%]", color=COLORS.TEXT_SECONDARY)
        ax.set_title("Particle Size Distribution", color=COLORS.TEXT_PRIMARY)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.15, color=COLORS.BORDER)

    def _plot_collection_by_size(self, ax):
        data = self._results.get("collection_by_size", {})
        sizes = data.get("sizes", [])
        if sizes:
            sizes_um = [s * 1e6 for s in sizes]
            bottom = [0] * len(sizes)
            palette = [COLORS.SUCCESS, COLORS.WARNING, COLORS.CAT_FEED, COLORS.ACCENT, COLORS.CAT_EXHAUST, COLORS.INFO]
            for i, (point, counts) in enumerate(data.get("points", {}).items()):
                color = palette[i % len(palette)]
                ax.bar(range(len(sizes)), counts, bottom=bottom, label=point, color=color, alpha=0.85)
                bottom = [b + c for b, c in zip(bottom, counts)]
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s:.0f}" for s in sizes_um], rotation=45)
        ax.set_xlabel("Particle Size [um]", color=COLORS.TEXT_SECONDARY)
        ax.set_ylabel("Particle Count", color=COLORS.TEXT_SECONDARY)
        ax.set_title("Collection by Size", color=COLORS.TEXT_PRIMARY)
        ax.grid(True, alpha=0.15, color=COLORS.BORDER, axis='y')

    def _update_data_tables(self):
        particles = self._results.get("particles", [])
        self.particle_table.setRowCount(min(len(particles), 1000))
        for i, p in enumerate(particles[:1000]):
            self.particle_table.setItem(i, 0, QTableWidgetItem(str(p.get("id", i))))
            self.particle_table.setItem(i, 1, QTableWidgetItem(f"{p.get('size', 0) * 1e6:.2f}"))
            self.particle_table.setItem(i, 2, QTableWidgetItem(f"{p.get('density', 0):.0f}"))
            self.particle_table.setItem(i, 3, QTableWidgetItem(p.get("type", "--")))
            self.particle_table.setItem(i, 4, QTableWidgetItem(p.get("zone", "--")))
            self.particle_table.setItem(i, 5, QTableWidgetItem(f"{p.get('residence_time', 0):.3f}"))

        collection = self._results.get("collection_summary", {})
        for i in range(self.collection_table.rowCount()):
            point = self.collection_table.item(i, 0).text()
            data = collection.get(point, {})
            self.collection_table.setItem(i, 1, QTableWidgetItem(str(data.get("count", 0))))
            self.collection_table.setItem(i, 2, QTableWidgetItem(f"{data.get('mass', 0):.2f}"))
            self.collection_table.setItem(i, 3, QTableWidgetItem(f"{data.get('avg_size', 0) * 1e6:.1f}"))
            self.collection_table.setItem(i, 4, QTableWidgetItem(data.get("composition", "--")))

    # ============================================================
    # Export
    # ============================================================

    def _export_plot(self):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            str(Path.home() / "efficiency_curve.png"),
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if file_path:
            self.figure.savefig(file_path, facecolor=COLORS.BG_DARK, edgecolor='none', dpi=150)

    def _export_table(self, table: QTableWidget, default_name: str):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            str(Path.home() / default_name),
            "CSV Files (*.csv)"
        )
        if file_path:
            with open(file_path, 'w') as f:
                headers = []
                for col in range(table.columnCount()):
                    headers.append(table.horizontalHeaderItem(col).text())
                f.write(",".join(headers) + "\n")
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    f.write(",".join(row_data) + "\n")

    def export_results(self, file_path: str):
        import json as _json
        from pathlib import Path as _P
        path = _P(file_path)
        if path.suffix == '.json':
            with open(path, 'w') as f:
                _json.dump(self._results, f, indent=2, default=str)
        elif path.suffix == '.csv':
            self._export_table(self.particle_table, str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def clear(self):
        self._results = {}
        self._update_summary()
        if HAS_MATPLOTLIB:
            self._create_empty_plot()
        self.particle_table.setRowCount(0)
        if hasattr(self, '_empty_hint'):
            self._empty_hint.show()
