"""
Main Window for ProteinProcessIO
========================================

Thin application shell that hosts two self-contained process pages
inside a ``QStackedWidget``:

  - **Page 0 — ClassificationPage**: Air classification with 3D
    viewport, simulation controls, results, and animation.
  - **Page 1 — PretreatmentPage**: GP-15 RF dielectric heating with
    its own 3D viewport, controls, matplotlib plots, and results.

All classification-specific logic (build, run, animate) lives in
``ClassificationPage``.  All pretreatment-specific logic lives in
``PretreatmentPage``.  ``MainWindow`` only handles:

  - Mode switching (toolbar / menu / welcome overlay)
  - Project file operations (new / open / save)
  - Assembly configuration dialog
  - Status bar, menus, help
"""

from typing import Optional, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QStatusBar, QMenuBar,
    QMenu, QMessageBox, QFileDialog,
    QLabel, QProgressBar, QFrame, QPushButton,
    QGraphicsOpacityEffect, QSizePolicy,
    QStackedWidget,
)
from PySide6.QtCore import (
    Qt, QSettings, Signal, Slot, QTimer,
    QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import QAction, QKeySequence, QCloseEvent

from .pages.classification_page import ClassificationPage
from .theme import COLORS

try:
    from .pages.pretreatment_page import PretreatmentPage
    _HAS_PRETREATMENT = True
except Exception:
    _HAS_PRETREATMENT = False

try:
    from .pages.milling_page import MillingPage
    _HAS_MILLING = True
except Exception:
    _HAS_MILLING = False

# Pipeline orchestration
from .pipeline import (
    PipelineState,
    PipelineStage,
    map_pretreatment_to_milling,
    map_milling_to_classification,
)


# ══════════════════════════════════════════════════════════════════════
#  Welcome overlay
# ══════════════════════════════════════════════════════════════════════

class _WelcomeOverlay(QWidget):
    """Translucent overlay shown when no project is loaded."""

    new_project_clicked = Signal()
    open_project_clicked = Signal()
    load_preset_clicked = Signal()
    pretreatment_clicked = Signal()
    milling_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: rgba(20, 20, 23, 210); border-radius: 0;")

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setFixedSize(520, 520)
        card.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_ELEVATED};
                border: 1px solid {COLORS.BORDER};
                border-radius: 12px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 32, 36, 32)
        card_layout.setSpacing(8)

        title = QLabel("ProteinProcessIO")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 18pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY};"
            " border: none; background: transparent;"
        )
        card_layout.addWidget(title)

        subtitle = QLabel("Interactive design & simulation for protein processing")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"font-size: 10pt; color: {COLORS.TEXT_SECONDARY};"
            " border: none; margin-bottom: 16px; background: transparent;"
        )
        card_layout.addWidget(subtitle)

        card_layout.addSpacing(6)

        stage_label = QLabel("Select a process stage to begin:")
        stage_label.setStyleSheet(
            f"font-size: 9pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        card_layout.addWidget(stage_label)
        card_layout.addSpacing(4)

        # Button styles
        _primary = f"""
            QPushButton {{
                background: {COLORS.ACCENT}; color: {COLORS.TEXT_INVERSE};
                border: none; border-radius: 6px; padding: 10px 20px;
                font-size: 11pt; font-weight: 600; min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.ACCENT_HOVER}; }}
            QPushButton:pressed {{ background: {COLORS.ACCENT_PRESSED}; }}
        """
        _secondary = f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE}; color: {COLORS.TEXT_PRIMARY};
                border: 1px solid {COLORS.BORDER}; border-radius: 6px;
                padding: 10px 20px; font-size: 10pt; min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.BG_HOVER}; border-color: {COLORS.TEXT_MUTED}; }}
        """
        _pretreatment = f"""
            QPushButton {{
                background: {COLORS.WARNING_MUTED}; color: {COLORS.WARNING};
                border: 1px solid {COLORS.WARNING}; border-radius: 6px;
                padding: 10px 20px; font-size: 11pt; font-weight: 600; min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.WARNING}; color: {COLORS.BG_DARKEST}; }}
        """

        pt_btn = QPushButton("  RF Pretreatment (GP-15)")
        pt_btn.setStyleSheet(_pretreatment)
        pt_btn.setToolTip("GP-15 RF dielectric heating simulation\nMoisture conditioning: 8–14% → 2–4% wb")
        pt_btn.clicked.connect(self.pretreatment_clicked.emit)
        card_layout.addWidget(pt_btn)

        _milling = f"""
            QPushButton {{
                background: {COLORS.SUCCESS_MUTED}; color: {COLORS.SUCCESS};
                border: 1px solid {COLORS.SUCCESS}; border-radius: 6px;
                padding: 10px 20px; font-size: 11pt; font-weight: 600; min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.SUCCESS}; color: {COLORS.BG_DARKEST}; }}
        """
        mill_btn = QPushButton("  Hammer Mill")
        mill_btn.setStyleSheet(_milling)
        mill_btn.setToolTip("Hammer mill impact milling simulation\nSize reduction: whole seeds → milled flour")
        mill_btn.clicked.connect(self.milling_clicked.emit)
        card_layout.addWidget(mill_btn)

        cls_btn = QPushButton("  Air Classification")
        cls_btn.setStyleSheet(_primary)
        cls_btn.setToolTip("Zigzag + wheel classifier + cyclone separation")
        cls_btn.clicked.connect(self.new_project_clicked.emit)
        card_layout.addWidget(cls_btn)

        card_layout.addSpacing(6)

        cfg_btn = QPushButton("  Configure Assembly...")
        cfg_btn.setStyleSheet(_secondary)
        cfg_btn.clicked.connect(self.load_preset_clicked.emit)
        card_layout.addWidget(cfg_btn)

        open_btn = QPushButton("  Open Existing Project...")
        open_btn.setStyleSheet(_secondary)
        open_btn.clicked.connect(self.open_project_clicked.emit)
        card_layout.addWidget(open_btn)

        card_layout.addSpacing(8)

        hint = QLabel("Use Ctrl+1 / Ctrl+2 to switch between modes anytime")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"font-size: 9pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        card_layout.addWidget(hint)

        outer.addWidget(card)

    def fade_out(self, duration: int = 350):
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._on_fade_finished)
        anim.start()
        self._anim = anim

    def _on_fade_finished(self):
        self.hide()
        self.setParent(None)
        self.deleteLater()


class _StatusSeparator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFixedWidth(1)
        self.setStyleSheet(f"color: {COLORS.BORDER};")


# ══════════════════════════════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Thin application shell.

    All process-specific logic lives in the two page widgets:
      - ``self.classification_page``  (``ClassificationPage``)
      - ``self.pretreatment_page``    (``PretreatmentPage``)

    ``MainWindow`` owns only: mode switching, project I/O, menus,
    toolbars, status bar, and the assembly configuration dialog.
    """

    # Signals
    project_changed = Signal(str)
    assembly_updated = Signal()
    mode_changed = Signal(str)

    # Mode constants
    MODE_CLASSIFICATION = "classification"
    MODE_PRETREATMENT = "pretreatment"
    MODE_MILLING = "milling"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Project state (per-system params for independent configure/build/run)
        self._project_path: Optional[Path] = None
        self._is_modified: bool = False
        self._assembly_config: Dict[str, Any] = {}
        self._assembly_params: Dict[str, Any] = {}
        self._pretreatment_params: Dict[str, Any] = {}
        self._classification_params: Dict[str, Any] = {}
        self._milling_params: Dict[str, Any] = {}
        self._pipeline_state: PipelineState = PipelineState()
        self._current_mode: str = self.MODE_CLASSIFICATION

        # Build UI
        self._setup_window()
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_central_widget()
        self._create_status_bar()
        self._restore_state()
        self._connect_signals()

        # Auto-save
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._auto_save_timer.start(60000)

        QTimer.singleShot(0, self._show_welcome_overlay)

    # ================================================================
    #  Window setup
    # ================================================================

    def _setup_window(self):
        self.setWindowTitle("ProteinProcessIO")
        self.setMinimumSize(1400, 900)
        self.resize(1800, 1100)

    # ================================================================
    #  Welcome overlay
    # ================================================================

    def _show_welcome_overlay(self):
        self._welcome = _WelcomeOverlay(self)
        self._welcome.new_project_clicked.connect(self._dismiss_welcome_and_new)
        self._welcome.open_project_clicked.connect(self._dismiss_welcome_and_open)
        self._welcome.load_preset_clicked.connect(self._dismiss_welcome_and_preset)
        self._welcome.pretreatment_clicked.connect(self._dismiss_welcome_and_pretreatment)
        self._welcome.milling_clicked.connect(self._dismiss_welcome_and_milling)
        self._welcome.setGeometry(self.centralWidget().geometry())
        self._welcome.show()
        self._welcome.raise_()

    def _dismiss_welcome(self):
        if hasattr(self, "_welcome") and self._welcome is not None:
            self._welcome.fade_out()
            self._welcome = None

    def _dismiss_welcome_and_new(self):
        self._dismiss_welcome()
        self.new_project()

    def _dismiss_welcome_and_open(self):
        self._dismiss_welcome()
        self.open_project()

    def _dismiss_welcome_and_preset(self):
        self._dismiss_welcome()
        self.show_assembly_config()

    def _dismiss_welcome_and_pretreatment(self):
        self._dismiss_welcome()
        self.switch_mode(self.MODE_PRETREATMENT)

    def _dismiss_welcome_and_milling(self):
        self._dismiss_welcome()
        self.switch_mode(self.MODE_MILLING)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_welcome") and self._welcome is not None and self._welcome.isVisible():
            self._welcome.setGeometry(self.centralWidget().geometry())

    # ================================================================
    #  Mode switching
    # ================================================================

    @Slot(str)
    def switch_mode(self, mode: str):
        """Switch between Classification, Pretreatment, and Milling pages."""
        if mode == self._current_mode:
            return

        self._current_mode = mode

        if mode == self.MODE_CLASSIFICATION:
            self._central_stack.setCurrentIndex(0)
            self._sim_toolbar.show()
            self._cls_mode_btn.setProperty("cssClass", "primary")
            self._pt_mode_btn.setProperty("cssClass", "")
            self._mill_mode_btn.setProperty("cssClass", "")
            self.statusBar().showMessage("Mode: Air Classification", 3000)
        elif mode == self.MODE_PRETREATMENT:
            self._central_stack.setCurrentIndex(1)
            self._sim_toolbar.hide()
            self._cls_mode_btn.setProperty("cssClass", "")
            self._pt_mode_btn.setProperty("cssClass", "primary")
            self._mill_mode_btn.setProperty("cssClass", "")
            self.statusBar().showMessage("Mode: RF Pretreatment (GP-15)", 3000)
        else:
            # MODE_MILLING
            self._central_stack.setCurrentIndex(2)
            self._sim_toolbar.hide()
            self._cls_mode_btn.setProperty("cssClass", "")
            self._pt_mode_btn.setProperty("cssClass", "")
            self._mill_mode_btn.setProperty("cssClass", "primary")
            self.statusBar().showMessage("Mode: Hammer Mill", 3000)

        # Force style refresh
        for btn in (self._cls_mode_btn, self._pt_mode_btn, self._mill_mode_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self.mode_changed.emit(mode)

    # ================================================================
    #  Actions
    # ================================================================

    def _create_actions(self):
        # File
        self.action_new = QAction("&New Project", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.setStatusTip("Create a new air classifier project")
        self.action_new.triggered.connect(self.new_project)

        self.action_open = QAction("&Open Project...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.open_project)

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self.save_project)

        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.action_save_as.triggered.connect(self.save_project_as)

        self.action_export_mesh = QAction("Export &Mesh...", self)
        self.action_export_mesh.triggered.connect(self.export_mesh)

        self.action_export_results = QAction("Export &Results...", self)
        self.action_export_results.triggered.connect(self.export_results)

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.triggered.connect(self.close)

        # Edit
        self.action_preferences = QAction("&Preferences...", self)
        self.action_preferences.setShortcut(QKeySequence("Ctrl+,"))
        self.action_preferences.triggered.connect(self.show_preferences)

        # View
        self.action_reset_layout = QAction("&Reset Layout", self)
        self.action_reset_layout.triggered.connect(self.reset_layout)

        self.action_fullscreen = QAction("&Fullscreen", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.triggered.connect(self.toggle_fullscreen)

        # Assembly
        self.action_add_component = QAction("&Configure Assembly...", self)
        self.action_add_component.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.action_add_component.setStatusTip("Configure all process stages (pretreatment, classification, milling)")
        self.action_add_component.triggered.connect(self.show_assembly_config)

        self.action_configure_pretreatment = QAction("Configure &Pretreatment (GP-15)...", self)
        self.action_configure_pretreatment.setStatusTip("Configure RF Pretreatment only; then Build current system")
        self.action_configure_pretreatment.triggered.connect(self.show_pretreatment_config)

        self.action_configure_classification = QAction("Configure &Classification...", self)
        self.action_configure_classification.setStatusTip("Configure Air Classification only; then Build current system")
        self.action_configure_classification.triggered.connect(self.show_classification_config)

        self.action_configure_current = QAction("Configure &Current System...", self)
        self.action_configure_current.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.action_configure_current.setStatusTip("Configure the active stage (Pretreatment, Classification, or Milling)")
        self.action_configure_current.triggered.connect(self.show_config_current_system)

        self.action_configure_milling = QAction("Configure &Hammer Mill...", self)
        self.action_configure_milling.setStatusTip("Configure Hammer Mill; then Build current system")
        self.action_configure_milling.triggered.connect(self.show_milling_config)

        self.action_build_system = QAction("&Build Current System", self)
        self.action_build_system.setShortcut(QKeySequence("Ctrl+B"))
        self.action_build_system.setStatusTip("Build the system for the active page (Pretreatment, Classification, or Milling)")
        self.action_build_system.triggered.connect(self.build_full_system)

        # Simulation (toolbar actions — delegate to active page)
        self.action_run_sim = QAction("&Run Simulation", self)
        self.action_run_sim.setShortcut(QKeySequence("F5"))
        self.action_run_sim.triggered.connect(self._run_active_simulation)

        self.action_pause_sim = QAction("&Pause", self)
        self.action_pause_sim.setShortcut(QKeySequence("F6"))
        self.action_pause_sim.setEnabled(False)
        self.action_pause_sim.triggered.connect(self._pause_active_simulation)

        self.action_stop_sim = QAction("&Stop", self)
        self.action_stop_sim.setShortcut(QKeySequence("Shift+F5"))
        self.action_stop_sim.setEnabled(False)
        self.action_stop_sim.triggered.connect(self._stop_active_simulation)

        self.action_sim_settings = QAction("Simulation &Settings...", self)
        self.action_sim_settings.triggered.connect(self.show_simulation_settings)

        # View: toggle results (pretreatment mode)
        self.action_toggle_results = QAction("Toggle &Results View", self)
        self.action_toggle_results.setShortcut(QKeySequence("Ctrl+R"))
        self.action_toggle_results.triggered.connect(self._toggle_pt_results)

        # Help
        self.action_help = QAction("&Documentation", self)
        self.action_help.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.action_help.triggered.connect(self.show_help)

        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self.show_about)

    # ================================================================
    #  Menus
    # ================================================================

    def _create_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_export_mesh)
        file_menu.addAction(self.action_export_results)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_preferences)

        view_menu = menubar.addMenu("&View")
        a1 = QAction("&Classification Mode", self)
        a1.setShortcut(QKeySequence("Ctrl+1"))
        a1.triggered.connect(lambda: self.switch_mode(self.MODE_CLASSIFICATION))
        view_menu.addAction(a1)
        a2 = QAction("&Pretreatment Mode", self)
        a2.setShortcut(QKeySequence("Ctrl+2"))
        a2.triggered.connect(lambda: self.switch_mode(self.MODE_PRETREATMENT))
        view_menu.addAction(a2)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_results)
        view_menu.addSeparator()
        view_menu.addAction(self.action_reset_layout)
        view_menu.addAction(self.action_fullscreen)

        assembly_menu = menubar.addMenu("&Assembly")
        assembly_menu.addAction(self.action_configure_current)
        assembly_menu.addAction(self.action_add_component)
        assembly_menu.addSeparator()
        assembly_menu.addAction(self.action_configure_pretreatment)
        assembly_menu.addAction(self.action_configure_classification)
        assembly_menu.addAction(self.action_configure_milling)
        assembly_menu.addSeparator()
        assembly_menu.addAction(self.action_build_system)

        sim_menu = menubar.addMenu("&Simulation")
        sim_menu.addAction(self.action_run_sim)
        sim_menu.addAction(self.action_pause_sim)
        sim_menu.addAction(self.action_stop_sim)
        sim_menu.addSeparator()
        sim_menu.addAction(self.action_sim_settings)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.action_help)
        help_menu.addSeparator()
        help_menu.addAction(self.action_about)

    # ================================================================
    #  Toolbars
    # ================================================================

    def _create_toolbars(self):
        # Main
        main_tb = QToolBar("Main", self)
        main_tb.setMovable(True)
        main_tb.setObjectName("MainToolbar")
        main_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        main_tb.addAction(self.action_new)
        main_tb.addAction(self.action_open)
        main_tb.addAction(self.action_save)
        main_tb.addSeparator()
        main_tb.addAction(self.action_add_component)
        main_tb.addAction(self.action_build_system)
        self.addToolBar(main_tb)

        # Mode
        mode_tb = QToolBar("Process Stage", self)
        mode_tb.setMovable(True)
        mode_tb.setObjectName("ModeToolbar")
        mode_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._cls_mode_btn = QPushButton("  Air Classification")
        self._cls_mode_btn.setProperty("cssClass", "primary")
        self._cls_mode_btn.setMinimumHeight(28)
        self._cls_mode_btn.clicked.connect(lambda: self.switch_mode(self.MODE_CLASSIFICATION))
        mode_tb.addWidget(self._cls_mode_btn)

        self._pt_mode_btn = QPushButton("  RF Pretreatment")
        self._pt_mode_btn.setMinimumHeight(28)
        self._pt_mode_btn.clicked.connect(lambda: self.switch_mode(self.MODE_PRETREATMENT))
        mode_tb.addWidget(self._pt_mode_btn)

        self._mill_mode_btn = QPushButton("  Hammer Mill")
        self._mill_mode_btn.setMinimumHeight(28)
        self._mill_mode_btn.clicked.connect(lambda: self.switch_mode(self.MODE_MILLING))
        mode_tb.addWidget(self._mill_mode_btn)

        self.addToolBar(mode_tb)

        # Simulation (classification only — hidden in pretreatment mode)
        self._sim_toolbar = QToolBar("Simulation", self)
        self._sim_toolbar.setMovable(True)
        self._sim_toolbar.setObjectName("SimulationToolbar")
        self._sim_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._sim_toolbar.addAction(self.action_run_sim)
        self._sim_toolbar.addAction(self.action_pause_sim)
        self._sim_toolbar.addAction(self.action_stop_sim)
        self.addToolBar(self._sim_toolbar)

    # ================================================================
    #  Central widget (stacked pages)
    # ================================================================

    def _create_central_widget(self):
        """Create the QStackedWidget holding both process pages."""
        self._central_stack = QStackedWidget()

        # Page 0: Classification
        self.classification_page = ClassificationPage()
        self._central_stack.addWidget(self.classification_page)

        # Page 1: Pretreatment
        if _HAS_PRETREATMENT:
            self.pretreatment_page = PretreatmentPage()
            self._central_stack.addWidget(self.pretreatment_page)
        else:
            self.pretreatment_page = None
            placeholder = QLabel(
                "Pretreatment module not available.\n"
                "Check that airclassifier.pretreatment is installed."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 12pt;")
            self._central_stack.addWidget(placeholder)

        # Page 2: Milling
        if _HAS_MILLING:
            self.milling_page = MillingPage()
            self._central_stack.addWidget(self.milling_page)
        else:
            self.milling_page = None
            placeholder_mill = QLabel(
                "Milling module not available.\n"
                "Check that airclassifier.milling is installed."
            )
            placeholder_mill.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder_mill.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 12pt;")
            self._central_stack.addWidget(placeholder_mill)

        self._central_stack.setCurrentIndex(0)
        self.setCentralWidget(self._central_stack)

    # ================================================================
    #  Status bar
    # ================================================================

    def _create_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._component_count_label = QLabel("0 components")
        self._component_count_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        sb.addPermanentWidget(self._component_count_label)

        sb.addPermanentWidget(_StatusSeparator())

        self.gpu_status_label = QLabel("GPU: Checking...")
        self.gpu_status_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        sb.addPermanentWidget(self.gpu_status_label)

        sb.addPermanentWidget(_StatusSeparator())

        self.sim_progress = QProgressBar()
        self.sim_progress.setFixedWidth(140)
        self.sim_progress.setFixedHeight(14)
        self.sim_progress.setTextVisible(False)
        self.sim_progress.setVisible(False)
        sb.addPermanentWidget(self.sim_progress)

        QTimer.singleShot(100, self._check_gpu_status)

    def _check_gpu_status(self):
        try:
            import warp as wp
            wp.init()
            devices = wp.get_devices()
            cuda = [d for d in devices if "cuda" in str(d).lower()]
            if cuda:
                self.gpu_status_label.setText(f"GPU: {cuda[0]} Ready")
                self.gpu_status_label.setStyleSheet(f"color: {COLORS.SUCCESS};")
            else:
                self.gpu_status_label.setText("GPU: CPU Mode (No CUDA)")
                self.gpu_status_label.setStyleSheet(f"color: {COLORS.WARNING};")
        except ImportError:
            self.gpu_status_label.setText("GPU: Warp Not Installed")
            self.gpu_status_label.setStyleSheet(f"color: {COLORS.DANGER};")
        except Exception as e:
            self.gpu_status_label.setText(f"GPU: Error - {str(e)[:30]}")
            self.gpu_status_label.setStyleSheet(f"color: {COLORS.DANGER};")

    def _update_component_count(self):
        p = self._assembly_params or self._classification_params
        count = 1
        if p.get("include_feed_system", True):
            count += 1
        if p.get("include_air_system", True):
            count += 1
        if p.get("include_exhaust", True):
            count += 1
        self._component_count_label.setText(f"{count} subsystems")

    def _merge_assembly_params(self) -> Dict[str, Any]:
        """Merge per-system params for backward compatibility and single-dialog use."""
        out = dict(self._classification_params)
        out.update(self._pretreatment_params)
        out.update(self._milling_params)
        return out

    def _split_assembly_params(self, params: Dict[str, Any]):
        """Split a full params dict into _pretreatment_params and _classification_params."""
        pt_keys = {"enable_pretreatment"} | {k for k in params if k.startswith("pt_")}
        cls_keys = {
            "enable_classification", "use_preclassification",
            "include_feed_system", "include_air_system", "include_exhaust",
            "include_ductwork", "include_dropout", "include_coarse_collection",
            "throughput_kg_h", "air_flow_m3_h",
            "venturi_inlet_diameter", "venturi_throat_ratio",
            "zigzag_channel_width", "zigzag_channel_depth", "zigzag_num_stages",
            "wheel_diameter", "wheel_rpm",
            "primary_cyclone_diameter", "secondary_cyclone_diameter", "tertiary_cyclone_diameter",
            "bag_filter_flow_rate", "hopper_diameter", "main_duct_diameter", "stack_height",
        }
        mill_keys = {"enable_milling"} | {k for k in params if k.startswith("mill_")}
        self._pretreatment_params = {k: params[k] for k in pt_keys if k in params}
        self._classification_params = {k: params[k] for k in cls_keys if k in params}
        self._milling_params = {k: params[k] for k in mill_keys if k in params}
        # Keep any other keys in _assembly_params via merge
        self._assembly_params = self._merge_assembly_params()
        for k, v in params.items():
            if k not in self._pretreatment_params and k not in self._classification_params and k not in self._milling_params:
                self._assembly_params[k] = v

    # ================================================================
    #  State save / restore
    # ================================================================

    def _restore_state(self):
        settings = QSettings()
        geo = settings.value("MainWindow/geometry")
        state = settings.value("MainWindow/state")
        if geo:
            self.restoreGeometry(geo)
        if state:
            self.restoreState(state)

    def _save_state(self):
        settings = QSettings()
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())

    # ================================================================
    #  Signal wiring
    # ================================================================

    def _connect_signals(self):
        # Classification page → status bar updates
        self.classification_page.simulation_state_changed.connect(
            self._on_cls_state_changed
        )
        self.classification_page.simulation_finished.connect(
            lambda _: self.statusBar().showMessage("Classification complete", 5000)
        )

        # Pretreatment page → status bar updates
        if self.pretreatment_page is not None:
            self.pretreatment_page.simulation_finished.connect(
                self._on_pretreatment_finished
            )

        # Milling page → status bar updates
        if self.milling_page is not None:
            self.milling_page.simulation_finished.connect(
                self._on_milling_finished
            )

        # Pipeline transfer signals
        if self.pretreatment_page is not None:
            self.pretreatment_page.transfer_to_milling_requested.connect(
                self._on_transfer_pretreatment_to_milling
            )
        if self.milling_page is not None:
            self.milling_page.transfer_to_classifier_requested.connect(
                self._on_transfer_milling_to_classifier
            )

    @Slot(str)
    def _on_cls_state_changed(self, state: str):
        """Update toolbar button states when classification sim state changes."""
        running = state == "running"
        paused = state == "paused"
        self.action_run_sim.setEnabled(not running or paused)
        self.action_pause_sim.setEnabled(running and not paused)
        self.action_stop_sim.setEnabled(running or paused)
        self.sim_progress.setVisible(running or paused)

    @Slot(dict)
    def _on_pretreatment_finished(self, results: Dict[str, Any]):
        outlet = results.get("outlet")
        if outlet:
            self.statusBar().showMessage(
                f"GP-15 complete: M={outlet.avg_moisture_wb:.1%}, "
                f"T={outlet.avg_temperature_c:.1f} °C, "
                f"E={outlet.total_energy_kwh:.2f} kWh, "
                f"CV={outlet.moisture_uniformity:.4f}",
                10000,
            )

    @Slot(dict)
    def _on_milling_finished(self, results: Dict[str, Any]):
        outlet = results.get("outlet")
        if outlet:
            self.statusBar().showMessage(
                f"Mill complete: d50={outlet.d50_um:.0f} µm, "
                f"throughput={outlet.throughput_kg_per_hr:.0f} kg/h, "
                f"power={outlet.power_kw:.2f} kW",
                10000,
            )

    # ================================================================
    #  Pipeline Transfer Handlers
    # ================================================================

    @Slot(dict)
    def _on_transfer_pretreatment_to_milling(self, pretreatment_results: Dict[str, Any]):
        """Handle transfer from pretreatment to milling stage.

        Maps outlet conditions (moisture, temperature, throughput) to
        configure the hammer mill feed parameters.
        """
        outlet = pretreatment_results.get("outlet")
        if outlet is None:
            QMessageBox.warning(
                self,
                "Transfer Error",
                "No pretreatment results available. Run a pretreatment simulation first.",
            )
            return

        # Map pretreatment outlet to milling parameters
        milling_params = map_pretreatment_to_milling(outlet, self._pretreatment_params)

        # Update milling params
        self._milling_params.update(milling_params)

        # Sync to milling page UI
        if self.milling_page is not None:
            self.milling_page.sync_settings_from_params(self._milling_params)

        # Update pipeline state
        self._pipeline_state.set_stage_result(
            stage=PipelineStage.PRETREATMENT,
            result=pretreatment_results,
            outlet=outlet,
            params_used=dict(self._pretreatment_params),
        )
        self._pipeline_state.mass_after_pretreatment_kg = (
            outlet.throughput_kg_per_hr * (outlet.residence_time_s / 3600.0)
            if outlet.residence_time_s > 0 else 0.0
        )

        # Switch to milling mode
        self.switch_mode(self.MODE_MILLING)

        # Status message
        self.statusBar().showMessage(
            f"Transferred: M={outlet.avg_moisture_wb:.1%}, "
            f"T={outlet.avg_temperature_c:.1f}°C, "
            f"{outlet.throughput_kg_per_hr:.0f} kg/h "
            f"→ Milling configured. Click Run (F5) to start.",
            10000,
        )

        self._set_modified(True)

    @Slot(dict)
    def _on_transfer_milling_to_classifier(self, milling_results: Dict[str, Any]):
        """Handle transfer from milling to classification stage.

        Maps PSD and material properties to configure the air classifier
        particle feed parameters.
        """
        outlet = milling_results.get("outlet")
        if outlet is None:
            QMessageBox.warning(
                self,
                "Transfer Error",
                "No milling results available. Run a milling simulation first.",
            )
            return

        # Map milling outlet to classification parameters
        classifier_params = map_milling_to_classification(outlet, self._milling_params)

        # Update classification params
        self._classification_params.update(classifier_params)

        # Sync to classification page UI
        self.classification_page.sync_settings_from_params(self._classification_params)

        # Update pipeline state
        self._pipeline_state.set_stage_result(
            stage=PipelineStage.MILLING,
            result=milling_results,
            outlet=outlet,
            params_used=dict(self._milling_params),
        )
        self._pipeline_state.mass_after_milling_kg = (
            outlet.throughput_kg_per_hr * (outlet.mean_residence_time_s / 3600.0)
            if outlet.mean_residence_time_s > 0 else 0.0
        )

        # Switch to classification mode
        self.switch_mode(self.MODE_CLASSIFICATION)

        # Rebuild system with new particle properties
        self.build_full_system()

        # Status message
        self.statusBar().showMessage(
            f"Transferred: d50={outlet.d50_um:.0f}µm, "
            f"{outlet.throughput_kg_per_hr:.0f} kg/h "
            f"→ Classifier configured. Click Run (F5) to start.",
            10000,
        )

        self._set_modified(True)

    def _update_window_title(self):
        title = "ProteinProcessIO"
        if self._project_path:
            title = f"{self._project_path.name} - {title}"
        if self._is_modified:
            title = f"*{title}"
        self.setWindowTitle(title)

    def _set_modified(self, modified: bool = True):
        self._is_modified = modified
        self._update_window_title()

    # ================================================================
    #  Simulation delegation (toolbar actions → active page)
    # ================================================================

    def _run_active_simulation(self):
        """Run simulation for the current page (F5 / Run)."""
        if self._current_mode == self.MODE_PRETREATMENT and self.pretreatment_page is not None:
            if not getattr(self.pretreatment_page, "_sim", None) and getattr(
                self.pretreatment_page, "build_system", None
            ):
                self.build_full_system()
            self.pretreatment_page.run_simulation()
        elif self._current_mode == self.MODE_CLASSIFICATION:
            if not self.classification_page.is_built:
                self.build_full_system()
            self.classification_page.run_simulation()
        elif self._current_mode == self.MODE_MILLING and self.milling_page is not None:
            if not getattr(self.milling_page, "_sim", None) and getattr(
                self.milling_page, "build_system", None
            ):
                self.build_full_system()
            self.milling_page.run_simulation()

    def _pause_active_simulation(self):
        if self._current_mode == self.MODE_CLASSIFICATION:
            self.classification_page.pause_simulation()

    def _stop_active_simulation(self):
        if self._current_mode == self.MODE_CLASSIFICATION:
            self.classification_page.stop_simulation()

    # ================================================================
    #  File operations
    # ================================================================

    @Slot()
    def new_project(self):
        if self._is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save changes before creating a new project?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self._project_path = None
        self._assembly_config = {}
        self._assembly_params = {}
        self._pretreatment_params = {}
        self._classification_params = {}
        self._milling_params = {}
        self._pipeline_state = PipelineState()
        self._is_modified = False
        self.classification_page.viewport.clear()
        self._update_window_title()
        self._update_component_count()
        self.statusBar().showMessage("New project created", 3000)

    @Slot()
    def open_project(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Project", str(Path.home()),
            "Air Classifier Project (*.acproj);;All Files (*)",
        )
        if fp:
            self._load_project(Path(fp))

    def _load_project(self, path: Path):
        try:
            import json
            with open(path, "r") as f:
                data = json.load(f)
            self._project_path = path
            self._assembly_config = data.get("assembly", {})
            if "pretreatment_params" in data and "classification_params" in data:
                self._pretreatment_params = data.get("pretreatment_params", {})
                self._classification_params = data.get("classification_params", {})
                self._milling_params = data.get("milling_params", {})
                self._assembly_params = self._merge_assembly_params()
            else:
                self._assembly_params = data.get("assembly_params", {})
                self._split_assembly_params(self._assembly_params)
            # Load pipeline state
            if "pipeline_state" in data:
                self._pipeline_state = PipelineState.from_dict(data["pipeline_state"])
            else:
                self._pipeline_state = PipelineState()
            self.classification_page.viewport.load_state(data.get("viewport", {}))
            self._is_modified = False
            self._update_window_title()
            self._update_component_count()
            self.project_changed.emit(str(path))
            self.statusBar().showMessage(f"Opened: {path.name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")

    @Slot()
    def save_project(self) -> bool:
        if self._project_path:
            return self._save_project_to(self._project_path)
        return self.save_project_as()

    @Slot()
    def save_project_as(self) -> bool:
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Project", str(Path.home() / "untitled.acproj"),
            "Air Classifier Project (*.acproj);;All Files (*)",
        )
        if fp:
            return self._save_project_to(Path(fp))
        return False

    def _save_project_to(self, path: Path) -> bool:
        try:
            import json
            data = {
                "version": "1.0",
                "assembly": self._assembly_config,
                "assembly_params": self._merge_assembly_params(),
                "pretreatment_params": self._pretreatment_params,
                "classification_params": self._classification_params,
                "milling_params": self._milling_params,
                "pipeline_state": self._pipeline_state.to_dict(),
                "viewport": self.classification_page.viewport.save_state(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self._project_path = path
            self._is_modified = False
            self._update_window_title()
            self.statusBar().showMessage(f"Saved: {path.name}", 3000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
            return False

    @Slot()
    def _auto_save(self):
        if self._is_modified and self._project_path:
            self._save_project_to(self._project_path)

    @Slot()
    def export_mesh(self):
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh", str(Path.home() / "classifier.stl"),
            "STL Files (*.stl);;OBJ Files (*.obj);;VTK Files (*.vtk)",
        )
        if fp:
            try:
                self.classification_page.viewport.export_mesh(fp)
                self.statusBar().showMessage(f"Exported: {Path(fp).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    @Slot()
    def export_results(self):
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export Results", str(Path.home() / "results.csv"),
            "CSV Files (*.csv);;JSON Files (*.json);;VTK Files (*.vtk)",
        )
        if fp:
            try:
                self.classification_page.results_panel.export_results(fp)
                self.statusBar().showMessage(f"Exported: {Path(fp).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # ================================================================
    #  Edit / View
    # ================================================================

    @Slot()
    def show_preferences(self):
        from .dialogs.preferences_dialog import PreferencesDialog
        PreferencesDialog(self).exec()

    @Slot()
    def reset_layout(self):
        settings = QSettings()
        settings.remove("MainWindow/state")
        settings.remove("MainWindow/geometry")
        QMessageBox.information(self, "Layout Reset", "Layout will be reset on next start.")

    @Slot(bool)
    def toggle_fullscreen(self, checked: bool):
        self.showFullScreen() if checked else self.showNormal()

    @Slot()
    def _toggle_pt_results(self):
        """Toggle between simulation and results views in pretreatment or milling mode."""
        if self._current_mode == self.MODE_PRETREATMENT and self.pretreatment_page is not None:
            if self.pretreatment_page._view_stack.currentIndex() == 0:
                self.pretreatment_page._show_results_view()
            else:
                self.pretreatment_page._show_simulation_view()
        elif self._current_mode == self.MODE_MILLING and self.milling_page is not None:
            if self.milling_page._view_stack.currentIndex() == 0:
                self.milling_page._show_results_view()
            else:
                self.milling_page._show_simulation_view()

    # ================================================================
    #  Assembly operations
    # ================================================================

    @Slot()
    def show_assembly_config(self):
        """Open full process configuration (all stages)."""
        from .dialogs.assembly_config_dialog import AssemblyConfigDialog
        dialog = AssemblyConfigDialog(self, self._assembly_params)
        dialog.assembly_configured.connect(self._on_assembly_configured)
        dialog.exec()

    @Slot()
    def show_pretreatment_config(self):
        """Open RF Pretreatment-only configuration. Apply & Build builds GP-15."""
        from .dialogs.pretreatment_config_dialog import PretreatmentConfigDialog
        dialog = PretreatmentConfigDialog(self, self._pretreatment_params)
        dialog.pretreatment_configured.connect(self._on_pretreatment_configured)
        dialog.exec()

    def _on_pretreatment_configured(self, params: Dict[str, Any]):
        self._pretreatment_params = params
        self._assembly_params = self._merge_assembly_params()
        self._set_modified(True)
        # Sync settings into pretreatment page controls
        if self.pretreatment_page is not None:
            self.pretreatment_page.sync_settings_from_params(self._pretreatment_params)
        if self._current_mode == self.MODE_PRETREATMENT and self.pretreatment_page is not None:
            self.build_full_system()
        self.statusBar().showMessage("Pretreatment configuration applied", 3000)

    @Slot()
    def show_classification_config(self):
        """Open Air Classification-only configuration. Apply & Build builds classifier."""
        from .dialogs.classification_config_dialog import ClassificationConfigDialog
        dialog = ClassificationConfigDialog(self, self._classification_params)
        dialog.classification_configured.connect(self._on_classification_configured)
        dialog.exec()

    def _on_classification_configured(self, params: Dict[str, Any]):
        self._classification_params = params
        self._assembly_params = self._merge_assembly_params()
        self._set_modified(True)
        self._update_component_count()
        self.classification_page.sync_settings_from_params(self._assembly_params)
        if self._current_mode == self.MODE_CLASSIFICATION:
            self.build_full_system()
        self.statusBar().showMessage("Classification configuration applied", 3000)

    @Slot()
    def show_milling_config(self):
        """Open Hammer Mill configuration. Apply & Build builds the mill."""
        from .dialogs.milling_config_dialog import MillingConfigDialog
        dialog = MillingConfigDialog(self, self._milling_params)
        dialog.milling_configured.connect(self._on_milling_configured)
        dialog.exec()

    def _on_milling_configured(self, params: Dict[str, Any]):
        self._milling_params = params
        self._assembly_params = self._merge_assembly_params()
        self._set_modified(True)
        # Sync settings into milling page controls
        if self.milling_page is not None:
            self.milling_page.sync_settings_from_params(self._milling_params)
        if self._current_mode == self.MODE_MILLING and self.milling_page is not None:
            self.build_full_system()
        self.statusBar().showMessage("Milling configuration applied", 3000)

    @Slot()
    def show_config_current_system(self):
        """Open configuration for the current page (Pretreatment, Classification, or Milling)."""
        if self._current_mode == self.MODE_PRETREATMENT:
            self.show_pretreatment_config()
        elif self._current_mode == self.MODE_CLASSIFICATION:
            self.show_classification_config()
        elif self._current_mode == self.MODE_MILLING:
            self.show_milling_config()

    def _on_assembly_configured(self, params: Dict[str, Any]):
        self._split_assembly_params(params)
        self._set_modified(True)
        self._update_component_count()

        # Switch mode based on selected stage
        if params.get("enable_milling", False):
            self.switch_mode(self.MODE_MILLING)
        elif params.get("enable_pretreatment", False):
            self.switch_mode(self.MODE_PRETREATMENT)
        elif params.get("enable_classification", True):
            self.switch_mode(self.MODE_CLASSIFICATION)

        # Sync params into all page settings
        self.classification_page.sync_settings_from_params(self._assembly_params)

        if self.pretreatment_page is not None:
            self.pretreatment_page.sync_settings_from_params(self._pretreatment_params)

        if self.milling_page is not None:
            self.milling_page.sync_settings_from_params(self._milling_params)

        # Build current system
        self.build_full_system()

    @Slot()
    def build_full_system(self):
        """Build the system for the active mode (uses per-system params)."""
        self.statusBar().showMessage("Building process system...")
        try:
            if self._current_mode == self.MODE_PRETREATMENT:
                if self.pretreatment_page is not None:
                    params = dict(self._pretreatment_params)
                    params.setdefault("enable_pretreatment", True)
                    ok = self.pretreatment_page.build_system(params)
                    if ok:
                        self.statusBar().showMessage("GP-15 machine built successfully", 5000)
                    else:
                        self.statusBar().showMessage("GP-15 build produced no geometry", 3000)
                else:
                    self.statusBar().showMessage("Pretreatment module not available", 3000)
            elif self._current_mode == self.MODE_MILLING:
                if self.milling_page is not None:
                    params = dict(self._milling_params)
                    params.setdefault("enable_milling", True)
                    ok = self.milling_page.build_system(params)
                    if ok:
                        self.statusBar().showMessage("Hammer mill built successfully", 5000)
                    else:
                        self.statusBar().showMessage("Mill build produced no geometry", 3000)
                else:
                    self.statusBar().showMessage("Milling module not available", 3000)
            else:
                params = dict(self._classification_params)
                params.setdefault("enable_classification", True)
                ok = self.classification_page.build_system(params)
                if ok:
                    self.statusBar().showMessage("System built successfully", 5000)
                else:
                    self.statusBar().showMessage("Build produced no geometry", 3000)
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Build Error",
                f"Failed to build system:\n{e}\n\n{traceback.format_exc()}",
            )
            self.statusBar().showMessage("Build failed", 3000)

    # ================================================================
    #  Simulation settings dialog
    # ================================================================

    @Slot()
    def show_simulation_settings(self):
        from .dialogs.simulation_settings_dialog import SimulationSettingsDialog
        sc = self.classification_page.sim_control
        dialog = SimulationSettingsDialog(sc.get_settings(), self)
        if dialog.exec():
            sc.set_settings(dialog.get_settings())

    # ================================================================
    #  Help
    # ================================================================

    @Slot()
    def show_help(self):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://airclassifier.readthedocs.io"))

    @Slot()
    def show_about(self):
        QMessageBox.about(
            self, "About ProteinProcessIO",
            """<h2>ProteinProcessIO</h2>
            <p>Version 1.0.0</p>
            <p>Interactive design and simulation tool for protein processing.</p>
            <p>Powered by NVIDIA Warp for GPU-accelerated multiphysics.</p>
            <hr>
            <p><b>Process Stages:</b></p>
            <ul>
                <li>RF Pretreatment (GP-15 dielectric heating)</li>
                <li>Air Classification (zigzag + wheel + cyclones)</li>
            </ul>
            """,
        )

    # ================================================================
    #  Close
    # ================================================================

    def closeEvent(self, event: QCloseEvent):
        if self._is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Save changes before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self.classification_page.cleanup()
        if self.pretreatment_page is not None:
            self.pretreatment_page.cleanup()
        if self.milling_page is not None:
            self.milling_page.cleanup()

        self._save_state()
        event.accept()
