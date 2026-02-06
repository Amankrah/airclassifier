"""
Air Classifier Application Entry Point
======================================

Main application class and launcher for the Air Classifier GUI.
"""

import sys
import os
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont, QPalette, QColor, QFontDatabase

from .main_window import MainWindow


class AirClassifierApp(QApplication):
    """
    Main application class for Air Classifier Designer.

    Handles application-level settings, theming, and initialization.
    Uses modern UI design with proper DPI scaling and larger fonts.
    """

    APP_NAME = "Air Classifier Designer"
    APP_VERSION = "1.0.0"
    ORG_NAME = "AirClassifier"
    ORG_DOMAIN = "airclassifier.local"

    # UI Scale factor for larger interface
    UI_SCALE = 1.25  # 25% larger than default

    def __init__(self, argv: list):
        # Enable high DPI scaling before creating app
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        os.environ["QT_SCALE_FACTOR"] = str(self.UI_SCALE)

        super().__init__(argv)

        # Set application metadata
        self.setApplicationName(self.APP_NAME)
        self.setApplicationVersion(self.APP_VERSION)
        self.setOrganizationName(self.ORG_NAME)
        self.setOrganizationDomain(self.ORG_DOMAIN)

        # Initialize settings
        self.settings = QSettings()

        # Apply theme
        self._apply_theme()

        # Create main window
        self.main_window: Optional[MainWindow] = None

    def _apply_theme(self):
        """Apply modern dark theme to the application."""
        # Set fusion style for modern look
        self.setStyle("Fusion")

        # Create dark palette
        palette = QPalette()

        # Base colors
        dark_color = QColor(45, 45, 48)
        darker_color = QColor(30, 30, 32)
        light_color = QColor(62, 62, 66)
        text_color = QColor(220, 220, 220)
        highlight_color = QColor(0, 122, 204)
        disabled_color = QColor(127, 127, 127)

        # Window and background
        palette.setColor(QPalette.ColorRole.Window, dark_color)
        palette.setColor(QPalette.ColorRole.WindowText, text_color)
        palette.setColor(QPalette.ColorRole.Base, darker_color)
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
        palette.setColor(QPalette.ColorRole.ToolTipBase, dark_color)
        palette.setColor(QPalette.ColorRole.ToolTipText, text_color)

        # Text
        palette.setColor(QPalette.ColorRole.Text, text_color)
        palette.setColor(QPalette.ColorRole.PlaceholderText, disabled_color)

        # Buttons
        palette.setColor(QPalette.ColorRole.Button, light_color)
        palette.setColor(QPalette.ColorRole.ButtonText, text_color)

        # Highlights
        palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        # Links
        palette.setColor(QPalette.ColorRole.Link, highlight_color)
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(148, 108, 200))

        # Disabled state
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)

        self.setPalette(palette)

        # Set default font - larger for better readability
        font = QFont("Segoe UI", 11)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)

        # Additional stylesheet for modern, larger UI
        self.setStyleSheet("""
            /* Global font and sizing */
            * {
                font-size: 11pt;
            }
            QMainWindow::separator {
                background: #3e3e42;
                width: 2px;
                height: 2px;
            }
            QMainWindow::separator:hover {
                background: #007acc;
            }
            QDockWidget {
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }
            QDockWidget::title {
                background: #3e3e42;
                padding: 10px;
                font-weight: bold;
                font-size: 12pt;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background: #2d2d30;
            }
            QTabBar::tab {
                background: #2d2d30;
                border: 1px solid #3e3e42;
                padding: 12px 24px;
                margin-right: 2px;
                font-size: 11pt;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background: #3e3e42;
                border-bottom: 3px solid #007acc;
            }
            QTabBar::tab:hover:!selected {
                background: #3e3e42;
            }
            QToolBar {
                background: #2d2d30;
                border: none;
                spacing: 6px;
                padding: 6px;
            }
            QToolBar QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 8px 12px;
                min-width: 80px;
                font-size: 10pt;
            }
            QToolBar QToolButton:hover {
                background: #3e3e42;
                border: 1px solid #007acc;
            }
            QToolBar QToolButton:pressed {
                background: #007acc;
            }
            QStatusBar {
                background: #007acc;
                color: white;
                padding: 6px;
                font-size: 11pt;
            }
            QMenuBar {
                background: #2d2d30;
                border-bottom: 1px solid #3e3e42;
                font-size: 11pt;
            }
            QMenuBar::item {
                padding: 10px 16px;
            }
            QMenuBar::item:selected {
                background: #3e3e42;
            }
            QMenu {
                background: #2d2d30;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 10px 40px 10px 24px;
                font-size: 11pt;
            }
            QMenu::item:selected {
                background: #3e3e42;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5f;
                min-height: 40px;
                border-radius: 5px;
                margin: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #7a7a7f;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #2d2d30;
                height: 14px;
            }
            QScrollBar::handle:horizontal {
                background: #5a5a5f;
                min-width: 40px;
                border-radius: 5px;
                margin: 3px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #7a7a7f;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 16px;
                padding: 16px 12px 12px 12px;
                font-weight: bold;
                font-size: 11pt;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 8px;
                font-size: 11pt;
            }
            QLabel {
                font-size: 11pt;
                padding: 2px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px 12px;
                min-height: 24px;
                font-size: 11pt;
                selection-background-color: #007acc;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus {
                border: 2px solid #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                selection-background-color: #007acc;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 8px 12px;
                min-height: 28px;
            }
            QPushButton {
                background: #3e3e42;
                border: 1px solid #505054;
                border-radius: 4px;
                padding: 10px 20px;
                min-height: 28px;
                min-width: 80px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background: #505054;
                border: 1px solid #007acc;
            }
            QPushButton:pressed {
                background: #007acc;
            }
            QPushButton:disabled {
                background: #2d2d30;
                color: #666;
            }
            QSlider::groove:horizontal {
                background: #3e3e42;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e90ff;
            }
            QProgressBar {
                background: #3e3e42;
                border: none;
                border-radius: 4px;
                text-align: center;
                min-height: 20px;
                font-size: 10pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007acc, stop:1 #00a2ff);
                border-radius: 4px;
            }
            QCheckBox {
                font-size: 11pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QRadioButton {
                font-size: 11pt;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
            QTreeView, QListView, QTableView {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                alternate-background-color: #2d2d30;
                font-size: 11pt;
            }
            QTreeView::item, QListView::item {
                padding: 8px 4px;
                min-height: 28px;
            }
            QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {
                background: #007acc;
            }
            QTreeView::item:hover, QListView::item:hover {
                background: #3e3e42;
            }
            QHeaderView::section {
                background: #3e3e42;
                border: none;
                border-right: 1px solid #2d2d30;
                padding: 10px 8px;
                font-size: 11pt;
                font-weight: bold;
            }
            QSplitter::handle {
                background: #3e3e42;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
            QSplitter::handle:hover {
                background: #007acc;
            }
            /* Form layouts */
            QFormLayout {
                spacing: 12px;
            }
        """)

    def run(self) -> int:
        """
        Initialize and show the main window, then run the event loop.

        Returns:
            Application exit code
        """
        self.main_window = MainWindow()
        self.main_window.show()
        return self.exec()


def launch_app():
    """
    Convenience function to launch the Air Classifier application.

    Usage:
        from airclassifier.gui import launch_app
        launch_app()
    """
    app = AirClassifierApp(sys.argv)
    sys.exit(app.run())


if __name__ == "__main__":
    launch_app()
