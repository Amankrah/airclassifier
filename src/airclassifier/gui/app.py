"""
Air Classifier Application Entry Point
======================================

Main application class and launcher for the Air Classifier GUI.
"""

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont, QPalette, QColor

from .main_window import MainWindow


class AirClassifierApp(QApplication):
    """
    Main application class for Air Classifier Designer.

    Handles application-level settings, theming, and initialization.
    """

    APP_NAME = "Air Classifier Designer"
    APP_VERSION = "1.0.0"
    ORG_NAME = "AirClassifier"
    ORG_DOMAIN = "airclassifier.local"

    def __init__(self, argv: list):
        super().__init__(argv)

        # Set application metadata
        self.setApplicationName(self.APP_NAME)
        self.setApplicationVersion(self.APP_VERSION)
        self.setOrganizationName(self.ORG_NAME)
        self.setOrganizationDomain(self.ORG_DOMAIN)

        # Enable high DPI support
        self.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

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

        # Set default font
        font = QFont("Segoe UI", 9)
        self.setFont(font)

        # Additional stylesheet for fine-tuning
        self.setStyleSheet("""
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
                padding: 6px;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background: #2d2d30;
            }
            QTabBar::tab {
                background: #2d2d30;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3e3e42;
                border-bottom: 2px solid #007acc;
            }
            QTabBar::tab:hover:!selected {
                background: #3e3e42;
            }
            QToolBar {
                background: #2d2d30;
                border: none;
                spacing: 3px;
                padding: 3px;
            }
            QToolButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 4px;
            }
            QToolButton:hover {
                background: #3e3e42;
                border: 1px solid #007acc;
            }
            QToolButton:pressed {
                background: #007acc;
            }
            QStatusBar {
                background: #007acc;
                color: white;
            }
            QMenuBar {
                background: #2d2d30;
                border-bottom: 1px solid #3e3e42;
            }
            QMenuBar::item {
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background: #3e3e42;
            }
            QMenu {
                background: #2d2d30;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 6px 30px 6px 20px;
            }
            QMenu::item:selected {
                background: #3e3e42;
            }
            QScrollBar:vertical {
                background: #2d2d30;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5f;
                min-height: 30px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6a6a6f;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #2d2d30;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background: #5a5a5f;
                min-width: 30px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #6a6a6f;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                selection-background-color: #007acc;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #007acc;
            }
            QSlider::groove:horizontal {
                background: #3e3e42;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e90ff;
            }
            QProgressBar {
                background: #3e3e42;
                border: none;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #007acc;
                border-radius: 3px;
            }
            QTreeView, QListView, QTableView {
                background: #1e1e1e;
                border: 1px solid #3e3e42;
                alternate-background-color: #2d2d30;
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
                padding: 6px;
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
