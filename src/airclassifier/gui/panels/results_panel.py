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
    QHeaderView, QComboBox, QFrame, QSplitter,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor

# Try to import matplotlib for plotting
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ResultsPanel(QWidget):
    """
    Panel for displaying simulation results.

    Shows:
    - Summary statistics
    - Separation efficiency curves
    - Particle distribution plots
    - Tabular data
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._results: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Summary tab
        summary_tab = self._create_summary_tab()
        tabs.addTab(summary_tab, "Summary")

        # Efficiency curves tab
        if HAS_MATPLOTLIB:
            curves_tab = self._create_curves_tab()
            tabs.addTab(curves_tab, "Efficiency Curves")

        # Particle data tab
        data_tab = self._create_data_tab()
        tabs.addTab(data_tab, "Particle Data")

        # Collection data tab
        collection_tab = self._create_collection_tab()
        tabs.addTab(collection_tab, "Collection Points")

    def _create_summary_tab(self) -> QWidget:
        """Create the summary statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Overall performance
        perf_group = QGroupBox("Overall Performance")
        perf_layout = QFormLayout(perf_group)

        self.total_particles_label = QLabel("--")
        perf_layout.addRow("Total Particles Processed:", self.total_particles_label)

        self.sep_efficiency_label = QLabel("--")
        perf_layout.addRow("Separation Efficiency:", self.sep_efficiency_label)

        self.protein_recovery_label = QLabel("--")
        perf_layout.addRow("Protein Recovery:", self.protein_recovery_label)

        self.protein_purity_label = QLabel("--")
        perf_layout.addRow("Protein Purity:", self.protein_purity_label)

        self.throughput_label = QLabel("--")
        perf_layout.addRow("Throughput:", self.throughput_label)

        layout.addWidget(perf_group)

        # Cut sizes
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

        # Mass balance
        mass_group = QGroupBox("Mass Balance")
        mass_layout = QFormLayout(mass_group)

        self.mass_in_label = QLabel("--")
        mass_layout.addRow("Feed Mass:", self.mass_in_label)

        self.mass_out_label = QLabel("--")
        mass_layout.addRow("Total Collected:", self.mass_out_label)

        self.mass_error_label = QLabel("--")
        mass_layout.addRow("Balance Error:", self.mass_error_label)

        layout.addWidget(mass_group)

        layout.addStretch()
        return widget

    def _create_curves_tab(self) -> QWidget:
        """Create the efficiency curves tab with matplotlib plots."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Plot controls
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
        export_btn.clicked.connect(self._export_plot)
        controls_layout.addWidget(export_btn)

        layout.addLayout(controls_layout)

        # Matplotlib canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor('#2d2d30')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize with empty plot
        self._create_empty_plot()

        return widget

    def _create_empty_plot(self):
        """Create an empty placeholder plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#dcdcdc')
        ax.spines['bottom'].set_color('#3e3e42')
        ax.spines['top'].set_color('#3e3e42')
        ax.spines['left'].set_color('#3e3e42')
        ax.spines['right'].set_color('#3e3e42')
        ax.set_xlabel("Particle Size [um]", color='#dcdcdc')
        ax.set_ylabel("Efficiency [%]", color='#dcdcdc')
        ax.set_title("No Results Available", color='#dcdcdc')
        ax.grid(True, alpha=0.3, color='#3e3e42')
        self.canvas.draw()

    def _create_data_tab(self) -> QWidget:
        """Create the particle data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Data table
        self.particle_table = QTableWidget()
        self.particle_table.setColumnCount(6)
        self.particle_table.setHorizontalHeaderLabels([
            "ID", "Size [um]", "Density [kg/m3]", "Type", "Final Zone", "Residence Time [s]"
        ])
        self.particle_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.particle_table.setAlternatingRowColors(True)
        layout.addWidget(self.particle_table)

        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(lambda: self._export_table(self.particle_table, "particle_data.csv"))
        export_layout.addWidget(export_btn)
        layout.addLayout(export_layout)

        return widget

    def _create_collection_tab(self) -> QWidget:
        """Create the collection points data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Collection summary table
        self.collection_table = QTableWidget()
        self.collection_table.setColumnCount(5)
        self.collection_table.setHorizontalHeaderLabels([
            "Collection Point", "Particles", "Mass [g]", "Avg Size [um]", "Composition"
        ])
        self.collection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.collection_table.setAlternatingRowColors(True)
        layout.addWidget(self.collection_table)

        # Pre-populate with collection points
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

    def set_results(self, results: Dict[str, Any]):
        """
        Set simulation results to display.

        Args:
            results: Dictionary containing simulation results
        """
        self._results = results
        self._update_summary()
        self._update_plot(self.plot_type_combo.currentText() if HAS_MATPLOTLIB else None)
        self._update_data_tables()

    def _update_summary(self):
        """Update summary statistics from results."""
        if not self._results:
            return

        # Update labels
        self.total_particles_label.setText(
            str(self._results.get("total_particles", "--"))
        )
        self.sep_efficiency_label.setText(
            f"{self._results.get('separation_efficiency', 0):.1f}%"
        )
        self.protein_recovery_label.setText(
            f"{self._results.get('protein_recovery', 0):.1f}%"
        )
        self.protein_purity_label.setText(
            f"{self._results.get('protein_purity', 0):.1f}%"
        )
        self.throughput_label.setText(
            f"{self._results.get('throughput_kg_h', 0):.1f} kg/h"
        )

        # Cut sizes
        cut_sizes = self._results.get("cut_sizes", {})
        self.zigzag_d50_label.setText(
            f"{cut_sizes.get('zigzag', 0)*1e6:.1f} um"
        )
        self.wheel_d50_label.setText(
            f"{cut_sizes.get('wheel', 0)*1e6:.1f} um"
        )
        self.cyclone_primary_d50_label.setText(
            f"{cut_sizes.get('cyclone_primary', 0)*1e6:.1f} um"
        )
        self.cyclone_secondary_d50_label.setText(
            f"{cut_sizes.get('cyclone_secondary', 0)*1e6:.1f} um"
        )
        self.cyclone_tertiary_d50_label.setText(
            f"{cut_sizes.get('cyclone_tertiary', 0)*1e6:.1f} um"
        )

        # Mass balance
        mass_balance = self._results.get("mass_balance", {})
        self.mass_in_label.setText(
            f"{mass_balance.get('feed', 0):.2f} g"
        )
        self.mass_out_label.setText(
            f"{mass_balance.get('collected', 0):.2f} g"
        )
        error = mass_balance.get('error', 0)
        self.mass_error_label.setText(f"{error:.2f}%")

    def _update_plot(self, plot_type: Optional[str]):
        """Update the matplotlib plot."""
        if not HAS_MATPLOTLIB or not plot_type:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Style for dark theme
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#dcdcdc')
        for spine in ax.spines.values():
            spine.set_color('#3e3e42')

        if not self._results:
            ax.set_title("No Results Available", color='#dcdcdc')
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

        ax.legend(facecolor='#2d2d30', edgecolor='#3e3e42', labelcolor='#dcdcdc')
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_grade_efficiency(self, ax):
        """Plot grade efficiency curve."""
        data = self._results.get("grade_efficiency", {})
        sizes = data.get("sizes", [])
        efficiency = data.get("efficiency", [])

        if sizes and efficiency:
            sizes_um = [s * 1e6 for s in sizes]
            ax.semilogx(sizes_um, efficiency, 'o-', color='#4ec9b0', label='Overall')
            ax.axhline(y=50, color='#f14c4c', linestyle='--', alpha=0.5, label='d50')

        ax.set_xlabel("Particle Size [um]", color='#dcdcdc')
        ax.set_ylabel("Grade Efficiency [%]", color='#dcdcdc')
        ax.set_title("Grade Efficiency Curve", color='#dcdcdc')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, color='#3e3e42')

    def _plot_tromp_curve(self, ax):
        """Plot Tromp (partition) curve."""
        data = self._results.get("tromp_curve", {})
        sizes = data.get("sizes", [])
        partition = data.get("partition", [])

        if sizes and partition:
            sizes_um = [s * 1e6 for s in sizes]
            ax.semilogx(sizes_um, partition, 'o-', color='#dcdcaa', label='Partition')
            ax.axhline(y=50, color='#f14c4c', linestyle='--', alpha=0.5)

        ax.set_xlabel("Particle Size [um]", color='#dcdcdc')
        ax.set_ylabel("To Coarse [%]", color='#dcdcdc')
        ax.set_title("Tromp Curve (Partition)", color='#dcdcdc')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, color='#3e3e42')

    def _plot_psd(self, ax):
        """Plot particle size distribution."""
        data = self._results.get("psd", {})

        for stream, values in data.items():
            sizes = values.get("sizes", [])
            freq = values.get("frequency", [])
            if sizes and freq:
                sizes_um = [s * 1e6 for s in sizes]
                ax.plot(sizes_um, freq, '-', label=stream)

        ax.set_xlabel("Particle Size [um]", color='#dcdcdc')
        ax.set_ylabel("Frequency [%]", color='#dcdcdc')
        ax.set_title("Particle Size Distribution", color='#dcdcdc')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, color='#3e3e42')

    def _plot_collection_by_size(self, ax):
        """Plot collection point distribution by size."""
        data = self._results.get("collection_by_size", {})

        sizes = data.get("sizes", [])
        if sizes:
            sizes_um = [s * 1e6 for s in sizes]
            bottom = [0] * len(sizes)

            colors = ['#4ec9b0', '#dcdcaa', '#ce9178', '#569cd6', '#c586c0', '#9cdcfe']
            for i, (point, counts) in enumerate(data.get("points", {}).items()):
                color = colors[i % len(colors)]
                ax.bar(range(len(sizes)), counts, bottom=bottom, label=point, color=color, alpha=0.8)
                bottom = [b + c for b, c in zip(bottom, counts)]

            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s:.0f}" for s in sizes_um], rotation=45)

        ax.set_xlabel("Particle Size [um]", color='#dcdcdc')
        ax.set_ylabel("Particle Count", color='#dcdcdc')
        ax.set_title("Collection by Size", color='#dcdcdc')
        ax.grid(True, alpha=0.3, color='#3e3e42', axis='y')

    def _update_data_tables(self):
        """Update data tables from results."""
        # Update particle table
        particles = self._results.get("particles", [])
        self.particle_table.setRowCount(min(len(particles), 1000))  # Limit display

        for i, p in enumerate(particles[:1000]):
            self.particle_table.setItem(i, 0, QTableWidgetItem(str(p.get("id", i))))
            self.particle_table.setItem(i, 1, QTableWidgetItem(f"{p.get('size', 0)*1e6:.2f}"))
            self.particle_table.setItem(i, 2, QTableWidgetItem(f"{p.get('density', 0):.0f}"))
            self.particle_table.setItem(i, 3, QTableWidgetItem(p.get("type", "--")))
            self.particle_table.setItem(i, 4, QTableWidgetItem(p.get("zone", "--")))
            self.particle_table.setItem(i, 5, QTableWidgetItem(f"{p.get('residence_time', 0):.3f}"))

        # Update collection table
        collection = self._results.get("collection_summary", {})
        for i in range(self.collection_table.rowCount()):
            point = self.collection_table.item(i, 0).text()
            data = collection.get(point, {})

            self.collection_table.setItem(i, 1, QTableWidgetItem(str(data.get("count", 0))))
            self.collection_table.setItem(i, 2, QTableWidgetItem(f"{data.get('mass', 0):.2f}"))
            self.collection_table.setItem(i, 3, QTableWidgetItem(f"{data.get('avg_size', 0)*1e6:.1f}"))
            self.collection_table.setItem(i, 4, QTableWidgetItem(data.get("composition", "--")))

    def _export_plot(self):
        """Export current plot to file."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot",
            str(Path.home() / "efficiency_curve.png"),
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        if file_path:
            self.figure.savefig(file_path, facecolor='#2d2d30', edgecolor='none', dpi=150)

    def _export_table(self, table: QTableWidget, default_name: str):
        """Export table to CSV."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            str(Path.home() / default_name),
            "CSV Files (*.csv)"
        )
        if file_path:
            with open(file_path, 'w') as f:
                # Header
                headers = []
                for col in range(table.columnCount()):
                    headers.append(table.horizontalHeaderItem(col).text())
                f.write(",".join(headers) + "\n")

                # Data
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    f.write(",".join(row_data) + "\n")

    def export_results(self, file_path: str):
        """Export all results to file."""
        import json
        from pathlib import Path

        path = Path(file_path)
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self._results, f, indent=2, default=str)
        elif path.suffix == '.csv':
            self._export_table(self.particle_table, str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def clear(self):
        """Clear all results."""
        self._results = {}
        self._update_summary()
        if HAS_MATPLOTLIB:
            self._create_empty_plot()
        self.particle_table.setRowCount(0)
