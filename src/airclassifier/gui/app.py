"""
Air Classifier Application Entry Point
======================================

Main application class and launcher for the Air Classifier GUI.
"""

import sys
import os
from typing import Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings, QLocale
from PySide6.QtGui import QFont, QPalette, QColor, QFontDatabase

# Re-export COLORS so existing `from .app import COLORS` still works
from .theme import COLORS


class AirClassifierApp(QApplication):
    """
    Main application class for ProteinProcessIO.

    Handles application-level settings, theming, and initialization.
    Uses modern UI design with proper DPI scaling and larger fonts.
    """

    APP_NAME = "ProteinProcessIO"
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

        # Force C locale so spin boxes use '.' decimal separator
        QLocale.setDefault(QLocale.c())

        # Set application metadata
        self.setApplicationName(self.APP_NAME)
        self.setApplicationVersion(self.APP_VERSION)
        self.setOrganizationName(self.ORG_NAME)
        self.setOrganizationDomain(self.ORG_DOMAIN)

        # Initialize settings
        self.settings = QSettings()

        # Apply theme
        self._apply_theme()

        # Create main window (lazy import to avoid circular dependency)
        self.main_window = None

    def _apply_theme(self):
        """Apply modern dark theme to the application."""
        # Set fusion style for modern look
        self.setStyle("Fusion")

        # Create dark palette
        palette = QPalette()
        C = COLORS

        # Window and background
        palette.setColor(QPalette.ColorRole.Window, QColor(C.BG_BASE))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(C.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Base, QColor(C.BG_DARK))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(C.BG_ELEVATED))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(C.BG_SURFACE))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(C.TEXT_PRIMARY))

        # Text
        palette.setColor(QPalette.ColorRole.Text, QColor(C.TEXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(C.TEXT_MUTED))

        # Buttons
        palette.setColor(QPalette.ColorRole.Button, QColor(C.BG_SURFACE))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(C.TEXT_PRIMARY))

        # Highlights
        palette.setColor(QPalette.ColorRole.Highlight, QColor(C.ACCENT))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(C.TEXT_INVERSE))

        # Links
        palette.setColor(QPalette.ColorRole.Link, QColor(C.ACCENT))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(C.CAT_EXHAUST))

        # Disabled state
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(C.TEXT_DISABLED))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(C.TEXT_DISABLED))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(C.TEXT_DISABLED))

        self.setPalette(palette)

        # Set default font - larger for better readability
        font = QFont("Segoe UI", 10)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)

        # Build stylesheet from design tokens
        self.setStyleSheet(self._build_stylesheet())

    @staticmethod
    def _build_stylesheet() -> str:
        """Build the application stylesheet from design tokens."""
        C = COLORS
        return f"""
            /* ===== GLOBAL ===== */
            * {{
                font-size: 10pt;
                outline: none;
            }}

            /* ===== MAIN WINDOW SEPARATORS ===== */
            QMainWindow::separator {{
                background: {C.BORDER_SUBTLE};
                width: 1px;
                height: 1px;
            }}
            QMainWindow::separator:hover {{
                background: {C.ACCENT};
            }}

            /* ===== DOCK WIDGETS ===== */
            QDockWidget {{
                font-weight: bold;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(undock.png);
            }}
            QDockWidget::title {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {C.BG_SURFACE}, stop:1 {C.BG_ELEVATED});
                padding: 8px 12px;
                font-weight: 600;
                font-size: 10pt;
                border-bottom: 1px solid {C.BORDER_SUBTLE};
            }}
            QDockWidget::close-button, QDockWidget::float-button {{
                border: none;
                padding: 2px;
            }}
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
                background: {C.BG_HOVER};
                border-radius: 3px;
            }}

            /* ===== TAB WIDGET ===== */
            QTabWidget::pane {{
                border: 1px solid {C.BORDER_SUBTLE};
                border-top: none;
                background: {C.BG_DARK};
                border-radius: 0 0 4px 4px;
            }}
            QTabBar::tab {{
                background: transparent;
                border: none;
                padding: 8px 20px;
                margin-right: 1px;
                font-size: 10pt;
                color: {C.TEXT_SECONDARY};
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {C.TEXT_PRIMARY};
                border-bottom: 2px solid {C.ACCENT};
                background: {C.BG_DARK};
            }}
            QTabBar::tab:hover:!selected {{
                color: {C.TEXT_PRIMARY};
                background: {C.BG_ELEVATED};
                border-bottom: 2px solid {C.BORDER};
            }}

            /* ===== TOOLBARS ===== */
            QToolBar {{
                background: {C.BG_BASE};
                border: none;
                border-bottom: 1px solid {C.BORDER_SUBTLE};
                spacing: 4px;
                padding: 3px 6px;
            }}
            QToolBar::separator {{
                background: {C.BORDER};
                width: 1px;
                margin: 4px 6px;
            }}
            QToolBar QToolButton {{
                background: transparent;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 9pt;
                color: {C.TEXT_SECONDARY};
            }}
            QToolBar QToolButton:hover {{
                background: {C.BG_HOVER};
                color: {C.TEXT_PRIMARY};
                border: 1px solid {C.BORDER};
            }}
            QToolBar QToolButton:pressed {{
                background: {C.ACCENT_MUTED};
                color: {C.ACCENT};
            }}
            QToolBar QToolButton:checked {{
                background: {C.ACCENT_MUTED};
                color: {C.ACCENT};
                border: 1px solid {C.ACCENT};
            }}

            /* ===== STATUS BAR ===== */
            QStatusBar {{
                background: {C.BG_DARKEST};
                border-top: 1px solid {C.BORDER_SUBTLE};
                color: {C.TEXT_SECONDARY};
                padding: 2px 8px;
                font-size: 9pt;
                min-height: 24px;
            }}
            QStatusBar::item {{
                border: none;
            }}
            QStatusBar QLabel {{
                font-size: 9pt;
                padding: 0 8px;
            }}

            /* ===== MENU BAR ===== */
            QMenuBar {{
                background: {C.BG_DARKEST};
                border-bottom: 1px solid {C.BORDER_SUBTLE};
                font-size: 10pt;
                padding: 1px 0;
            }}
            QMenuBar::item {{
                padding: 6px 14px;
                border-radius: 4px;
                margin: 2px 1px;
            }}
            QMenuBar::item:selected {{
                background: {C.BG_HOVER};
            }}
            QMenuBar::item:pressed {{
                background: {C.ACCENT_MUTED};
            }}
            QMenu {{
                background: {C.BG_ELEVATED};
                border: 1px solid {C.BORDER};
                border-radius: 6px;
                padding: 4px 0;
            }}
            QMenu::item {{
                padding: 7px 32px 7px 20px;
                font-size: 10pt;
                border-radius: 3px;
                margin: 1px 4px;
            }}
            QMenu::item:selected {{
                background: {C.ACCENT_MUTED};
                color: {C.ACCENT_HOVER};
            }}
            QMenu::separator {{
                height: 1px;
                background: {C.BORDER_SUBTLE};
                margin: 4px 12px;
            }}
            QMenu::icon {{
                padding-left: 8px;
            }}

            /* ===== SCROLLBARS ===== */
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {C.TEXT_MUTED};
                min-height: 30px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {C.TEXT_SECONDARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 10px;
            }}
            QScrollBar::handle:horizontal {{
                background: {C.TEXT_MUTED};
                min-width: 30px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {C.TEXT_SECONDARY};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: transparent;
            }}

            /* ===== GROUP BOXES ===== */
            QGroupBox {{
                border: 1px solid {C.BORDER_SUBTLE};
                border-radius: 6px;
                margin-top: 12px;
                padding: 14px 10px 10px 10px;
                font-weight: 600;
                font-size: 9pt;
                color: {C.TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {C.TEXT_SECONDARY};
                font-size: 9pt;
            }}

            /* ===== LABELS ===== */
            QLabel {{
                font-size: 10pt;
                padding: 1px;
            }}

            /* ===== TEXT INPUTS ===== */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {C.BG_DARK};
                border: 1px solid {C.BORDER};
                border-radius: 5px;
                padding: 5px 10px;
                min-height: 20px;
                font-size: 10pt;
                color: {C.TEXT_PRIMARY};
                selection-background-color: {C.ACCENT};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border: 1px solid {C.BORDER_FOCUS};
            }}
            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
                background: {C.BG_BASE};
                color: {C.TEXT_DISABLED};
                border-color: {C.BORDER_SUBTLE};
            }}
            QTextEdit {{
                background: {C.BG_DARK};
                border: 1px solid {C.BORDER};
                border-radius: 5px;
                padding: 6px;
                font-size: 10pt;
                color: {C.TEXT_PRIMARY};
                selection-background-color: {C.ACCENT};
            }}
            QTextEdit:focus {{
                border: 1px solid {C.BORDER_FOCUS};
            }}

            /* ===== COMBOBOX DROPDOWN ===== */
            QComboBox::drop-down {{
                border: none;
                width: 24px;
                subcontrol-position: right center;
            }}
            QComboBox::down-arrow {{
                width: 10px;
                height: 10px;
            }}
            QComboBox QAbstractItemView {{
                background: {C.BG_ELEVATED};
                border: 1px solid {C.BORDER};
                border-radius: 4px;
                selection-background-color: {C.ACCENT_MUTED};
                selection-color: {C.ACCENT_HOVER};
                padding: 3px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 5px 10px;
                min-height: 22px;
                border-radius: 3px;
            }}

            /* ===== BUTTONS ===== */
            QPushButton {{
                background: {C.BG_SURFACE};
                border: 1px solid {C.BORDER};
                border-radius: 5px;
                padding: 6px 16px;
                min-height: 22px;
                font-size: 10pt;
                color: {C.TEXT_PRIMARY};
            }}
            QPushButton:hover {{
                background: {C.BG_HOVER};
                border-color: {C.TEXT_MUTED};
            }}
            QPushButton:pressed {{
                background: {C.ACCENT_MUTED};
                border-color: {C.ACCENT};
                color: {C.ACCENT_HOVER};
            }}
            QPushButton:disabled {{
                background: {C.BG_BASE};
                color: {C.TEXT_DISABLED};
                border-color: {C.BORDER_SUBTLE};
            }}
            /* Primary button variant - applied via objectName or setProperty */
            QPushButton[cssClass="primary"] {{
                background: {C.ACCENT};
                color: {C.TEXT_INVERSE};
                border: 1px solid {C.ACCENT};
                font-weight: 600;
            }}
            QPushButton[cssClass="primary"]:hover {{
                background: {C.ACCENT_HOVER};
                border-color: {C.ACCENT_HOVER};
            }}
            QPushButton[cssClass="primary"]:pressed {{
                background: {C.ACCENT_PRESSED};
            }}
            /* Success button */
            QPushButton[cssClass="success"] {{
                background: {C.SUCCESS};
                color: {C.BG_DARKEST};
                border: 1px solid {C.SUCCESS};
                font-weight: 600;
            }}
            QPushButton[cssClass="success"]:hover {{
                background: #4ee69c;
                border-color: #4ee69c;
            }}
            QPushButton[cssClass="success"]:pressed {{
                background: #2db872;
            }}
            /* Danger button */
            QPushButton[cssClass="danger"] {{
                background: {C.DANGER};
                color: {C.TEXT_INVERSE};
                border: 1px solid {C.DANGER};
                font-weight: 600;
            }}
            QPushButton[cssClass="danger"]:hover {{
                background: #ff6e6b;
                border-color: #ff6e6b;
            }}
            QPushButton[cssClass="danger"]:pressed {{
                background: #d32f2f;
            }}
            /* Ghost button */
            QPushButton[cssClass="ghost"] {{
                background: transparent;
                border: 1px solid transparent;
                color: {C.TEXT_SECONDARY};
            }}
            QPushButton[cssClass="ghost"]:hover {{
                background: {C.BG_HOVER};
                color: {C.TEXT_PRIMARY};
            }}

            /* ===== SLIDERS ===== */
            QSlider::groove:horizontal {{
                background: {C.BG_SURFACE};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C.ACCENT};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {C.ACCENT_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {C.ACCENT};
                border-radius: 2px;
            }}

            /* ===== PROGRESS BAR ===== */
            QProgressBar {{
                background: {C.BG_SURFACE};
                border: none;
                border-radius: 4px;
                text-align: center;
                min-height: 16px;
                font-size: 9pt;
                color: {C.TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {C.ACCENT}, stop:1 {C.ACCENT_HOVER});
                border-radius: 4px;
            }}

            /* ===== CHECKBOXES & RADIO ===== */
            QCheckBox {{
                font-size: 10pt;
                spacing: 6px;
                color: {C.TEXT_PRIMARY};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {C.BORDER};
                border-radius: 3px;
                background: {C.BG_DARK};
            }}
            QCheckBox::indicator:checked {{
                background: {C.ACCENT};
                border-color: {C.ACCENT};
            }}
            QCheckBox::indicator:hover {{
                border-color: {C.ACCENT};
            }}
            QRadioButton {{
                font-size: 10pt;
                spacing: 6px;
                color: {C.TEXT_PRIMARY};
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {C.BORDER};
                border-radius: 8px;
                background: {C.BG_DARK};
            }}
            QRadioButton::indicator:checked {{
                background: {C.ACCENT};
                border-color: {C.ACCENT};
            }}

            /* ===== TREE / LIST / TABLE ===== */
            QTreeView, QListView, QTableView {{
                background: {C.BG_DARK};
                border: 1px solid {C.BORDER_SUBTLE};
                border-radius: 4px;
                alternate-background-color: {C.BG_ELEVATED};
                font-size: 10pt;
                outline: none;
            }}
            QTreeView::item, QListView::item {{
                padding: 4px 6px;
                min-height: 24px;
                border-radius: 3px;
                margin: 0 2px;
            }}
            QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
                background: {C.ACCENT_MUTED};
                color: {C.ACCENT_HOVER};
            }}
            QTreeView::item:hover, QListView::item:hover {{
                background: {C.BG_HOVER};
            }}
            QTreeView::branch {{
                background: transparent;
            }}
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {{
                border-image: none;
            }}
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {{
                border-image: none;
            }}
            QHeaderView::section {{
                background: {C.BG_SURFACE};
                border: none;
                border-right: 1px solid {C.BG_DARK};
                border-bottom: 1px solid {C.BORDER_SUBTLE};
                padding: 6px 8px;
                font-size: 9pt;
                font-weight: 600;
                color: {C.TEXT_SECONDARY};
            }}
            QTableWidget {{
                gridline-color: {C.BORDER_SUBTLE};
            }}

            /* ===== SPLITTERS ===== */
            QSplitter::handle {{
                background: {C.BORDER_SUBTLE};
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            QSplitter::handle:vertical {{
                height: 2px;
            }}
            QSplitter::handle:hover {{
                background: {C.ACCENT};
            }}

            /* ===== TOOLTIPS ===== */
            QToolTip {{
                background: {C.BG_SURFACE};
                border: 1px solid {C.BORDER};
                border-radius: 4px;
                color: {C.TEXT_PRIMARY};
                padding: 4px 8px;
                font-size: 9pt;
            }}

            /* ===== DIALOG BUTTONS ===== */
            QDialogButtonBox QPushButton {{
                min-width: 72px;
            }}

            /* ===== FRAMES ===== */
            QFrame[frameShape="4"] /* HLine */ {{
                color: {C.BORDER_SUBTLE};
                max-height: 1px;
            }}
        """

    def run(self) -> int:
        """
        Initialize and show the main window, then run the event loop.

        Returns:
            Application exit code
        """
        from .main_window import MainWindow
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
