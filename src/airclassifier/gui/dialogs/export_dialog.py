"""
Export Results Dialog
=====================

Dialog for exporting simulation results to various formats
with configurable options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QRadioButton,
    QButtonGroup, QCheckBox, QFileDialog,
    QLineEdit, QGroupBox, QFormLayout,
)

from ..theme import COLORS


class ExportFormat(Enum):
    """Available export formats."""
    CSV = "csv"
    EXCEL = "xlsx"
    JSON = "json"
    PDF = "pdf"


class ExportDialog(QDialog):
    """Dialog for exporting simulation results.

    Signals:
        export_requested(dict): Emitted with export configuration
    """

    export_requested = Signal(dict)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        results: Optional[Dict[str, Any]] = None,
        default_path: Optional[Path] = None,
    ):
        super().__init__(parent)
        self._results = results
        self._default_path = default_path or Path.home() / "milling_results"

        self.setWindowTitle("Export Results")
        self.setMinimumSize(450, 400)
        self.resize(500, 450)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Export Simulation Results")
        title.setStyleSheet(f"""
            font-size: 14pt;
            font-weight: 700;
            color: {COLORS.TEXT_PRIMARY};
        """)
        layout.addWidget(title)

        # Format selection
        format_group = QGroupBox("Export Format")
        format_group.setStyleSheet(self._get_group_style())
        format_layout = QVBoxLayout(format_group)
        format_layout.setSpacing(8)

        self._format_group = QButtonGroup(self)

        formats = [
            (ExportFormat.CSV, "CSV", "Raw data in comma-separated format"),
            (ExportFormat.EXCEL, "Excel (.xlsx)", "Spreadsheet with charts and formatting"),
            (ExportFormat.JSON, "JSON", "Machine-readable structured data"),
            (ExportFormat.PDF, "PDF Report", "Formatted report with charts and summary"),
        ]

        for i, (fmt, name, desc) in enumerate(formats):
            row = QHBoxLayout()

            radio = QRadioButton(name)
            radio.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-weight: 600;")
            if fmt == ExportFormat.EXCEL:
                radio.setChecked(True)
            self._format_group.addButton(radio, i)
            row.addWidget(radio)

            hint = QLabel(desc)
            hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9pt;")
            row.addWidget(hint, 1)

            format_layout.addLayout(row)

        layout.addWidget(format_group)

        # Content selection
        content_group = QGroupBox("Include")
        content_group.setStyleSheet(self._get_group_style())
        content_layout = QVBoxLayout(content_group)
        content_layout.setSpacing(6)

        self._include_psd = QCheckBox("PSD histogram")
        self._include_psd.setChecked(True)
        self._include_psd.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        content_layout.addWidget(self._include_psd)

        self._include_timeseries = QCheckBox("Time series (d50, power, throughput)")
        self._include_timeseries.setChecked(True)
        self._include_timeseries.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        content_layout.addWidget(self._include_timeseries)

        self._include_summary = QCheckBox("Summary statistics")
        self._include_summary.setChecked(True)
        self._include_summary.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        content_layout.addWidget(self._include_summary)

        self._include_config = QCheckBox("Machine configuration")
        self._include_config.setChecked(False)
        self._include_config.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        content_layout.addWidget(self._include_config)

        self._include_screenshot = QCheckBox("3D viewport screenshot")
        self._include_screenshot.setChecked(False)
        self._include_screenshot.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY};")
        content_layout.addWidget(self._include_screenshot)

        layout.addWidget(content_group)

        # Output path
        path_group = QGroupBox("Output")
        path_group.setStyleSheet(self._get_group_style())
        path_layout = QHBoxLayout(path_group)

        self._path_edit = QLineEdit()
        self._path_edit.setText(str(self._default_path))
        self._path_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {COLORS.BG_DARKEST};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QLineEdit:focus {{
                border-color: {COLORS.ACCENT};
            }}
        """)
        path_layout.addWidget(self._path_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 4px;
                padding: 8px 16px;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
            }}
        """)
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(browse_btn)

        layout.addWidget(path_group)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                padding: 10px 24px;
                color: {COLORS.TEXT_SECONDARY};
            }}
            QPushButton:hover {{
                background: {COLORS.BG_HOVER};
                color: {COLORS.TEXT_PRIMARY};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS.ACCENT};
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                color: {COLORS.TEXT_INVERSE};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS.ACCENT_HOVER};
            }}
        """)
        self._export_btn.clicked.connect(self._do_export)
        btn_layout.addWidget(self._export_btn)

        layout.addLayout(btn_layout)

    def _get_group_style(self) -> str:
        return f"""
            QGroupBox {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
                color: {COLORS.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
        """

    def _connect_signals(self):
        pass

    def _browse_path(self):
        """Open file browser."""
        formats = [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON, ExportFormat.PDF]
        selected_format = formats[self._format_group.checkedId()]

        filters = {
            ExportFormat.CSV: "CSV Files (*.csv)",
            ExportFormat.EXCEL: "Excel Files (*.xlsx)",
            ExportFormat.JSON: "JSON Files (*.json)",
            ExportFormat.PDF: "PDF Files (*.pdf)",
        }

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            str(self._default_path),
            filters[selected_format],
        )

        if path:
            self._path_edit.setText(path)

    def _do_export(self):
        """Execute export with current settings."""
        formats = [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON, ExportFormat.PDF]
        selected_format = formats[self._format_group.checkedId()]

        config = {
            "format": selected_format.value,
            "path": self._path_edit.text(),
            "include_psd": self._include_psd.isChecked(),
            "include_timeseries": self._include_timeseries.isChecked(),
            "include_summary": self._include_summary.isChecked(),
            "include_config": self._include_config.isChecked(),
            "include_screenshot": self._include_screenshot.isChecked(),
            "results": self._results,
        }

        self.export_requested.emit(config)
        self.accept()

    def get_export_config(self) -> Dict[str, Any]:
        """Get the export configuration."""
        formats = [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON, ExportFormat.PDF]
        selected_format = formats[self._format_group.checkedId()]

        return {
            "format": selected_format.value,
            "path": self._path_edit.text(),
            "include_psd": self._include_psd.isChecked(),
            "include_timeseries": self._include_timeseries.isChecked(),
            "include_summary": self._include_summary.isChecked(),
            "include_config": self._include_config.isChecked(),
            "include_screenshot": self._include_screenshot.isChecked(),
        }
