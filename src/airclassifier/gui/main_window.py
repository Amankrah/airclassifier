"""
Main Window for Air Classifier Designer
========================================

Central window containing all panels, menus, and toolbars.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QToolBar, QStatusBar, QMenuBar,
    QMenu, QMessageBox, QFileDialog,
    QLabel, QProgressBar, QFrame, QPushButton,
    QGraphicsOpacityEffect, QSizePolicy,
)
from PySide6.QtCore import (
    Qt, QSettings, Signal, Slot, QTimer,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
)
from PySide6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent, QFont

from .panels.simulation_control import SimulationControlPanel
from .panels.results_panel import ResultsPanel
from .widgets.viewport_3d import Viewport3D
from .theme import COLORS


class _WelcomeOverlay(QWidget):
    """
    Translucent overlay shown when no project is loaded.

    Provides quick-start actions and guides new users.
    """

    new_project_clicked = Signal()
    open_project_clicked = Signal()
    load_preset_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"background: rgba(20, 20, 23, 210); border-radius: 0;")

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setFixedSize(480, 380)
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

        # Title
        title = QLabel("Air Classifier Designer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 18pt; font-weight: 700; color: {COLORS.TEXT_PRIMARY}; border: none; background: transparent;")
        card_layout.addWidget(title)

        subtitle = QLabel("Interactive design & simulation for air classification systems")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: 10pt; color: {COLORS.TEXT_SECONDARY}; border: none; margin-bottom: 16px; background: transparent;")
        card_layout.addWidget(subtitle)

        card_layout.addSpacing(10)

        # Quick-start buttons
        btn_style_primary = f"""
            QPushButton {{
                background: {COLORS.ACCENT};
                color: {COLORS.TEXT_INVERSE};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: 600;
                min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.ACCENT_HOVER}; }}
            QPushButton:pressed {{ background: {COLORS.ACCENT_PRESSED}; }}
        """
        btn_style_secondary = f"""
            QPushButton {{
                background: {COLORS.BG_SURFACE};
                color: {COLORS.TEXT_PRIMARY};
                border: 1px solid {COLORS.BORDER};
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 10pt;
                min-height: 28px;
            }}
            QPushButton:hover {{ background: {COLORS.BG_HOVER}; border-color: {COLORS.TEXT_MUTED}; }}
        """

        new_btn = QPushButton("  New Project")
        new_btn.setStyleSheet(btn_style_primary)
        new_btn.clicked.connect(self.new_project_clicked.emit)
        card_layout.addWidget(new_btn)

        open_btn = QPushButton("  Open Existing Project...")
        open_btn.setStyleSheet(btn_style_secondary)
        open_btn.clicked.connect(self.open_project_clicked.emit)
        card_layout.addWidget(open_btn)

        config_btn = QPushButton("  Configure Assembly")
        config_btn.setStyleSheet(btn_style_secondary)
        config_btn.clicked.connect(self.load_preset_clicked.emit)
        card_layout.addWidget(config_btn)

        card_layout.addSpacing(10)

        hint = QLabel("Tip: Configure Assembly to set parameters, then Build Full System to preview")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 9pt; color: {COLORS.TEXT_MUTED}; border: none; background: transparent;")
        card_layout.addWidget(hint)

        outer.addWidget(card)

    def fade_out(self, duration: int = 350):
        """Animate the overlay away."""
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._on_fade_finished)
        anim.start()
        self._anim = anim          # prevent GC

    def _on_fade_finished(self):
        self.hide()
        self.setParent(None)
        self.deleteLater()


class _StatusSeparator(QFrame):
    """Thin vertical line for the status bar."""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFixedWidth(1)
        self.setStyleSheet(f"color: {COLORS.BORDER};")


class MainWindow(QMainWindow):
    """
    Main application window for Air Classifier Designer.

    Layout:
    +-------------------------------------------------------------+
    |  Menu Bar                                                     |
    +-------------------------------------------------------------+
    |  Tool Bar                                                     |
    +------------+-------------------------------+-----------------+
    |            |                               |                 |
    | Components |     Central Area              |  Properties     |
    |            |  +-------------------------+  |                 |
    |            |  |   3D Viewport           |  |                 |
    |            |  |                         |  |                 |
    |            |  +-------------------------+  |                 |
    |            |  |   Assembly Canvas       |  |                 |
    |            |  |   (Node Editor)         |  |                 |
    |            |  +-------------------------+  |                 |
    +------------+-------------------------------+-----------------+
    |  Simulation Control / Results Panel (Tabbed)                  |
    +-------------------------------------------------------------+
    |  Status Bar                                                   |
    +-------------------------------------------------------------+
    """

    # Signals
    project_changed = Signal(str)  # Emitted when project path changes
    assembly_updated = Signal()    # Emitted when assembly is modified
    simulation_state_changed = Signal(str)  # "idle", "running", "paused", "completed"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Project state
        self._project_path: Optional[Path] = None
        self._is_modified: bool = False
        self._assembly_config: Dict[str, Any] = {}
        self._simulation_state: str = "idle"
        self._built_backend = None  # set by Build Full System
        self._assembly_params: Dict[str, Any] = {}  # set by Assembly Config dialog
        self._animation_controller = None  # AnimationController instance
        self._anim_generation = 0  # incremented each run; stale timers check this

        # Initialize UI
        self._setup_window()
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_central_widget()
        self._create_dock_widgets()
        self._create_status_bar()
        self._restore_state()

        # Connect signals
        self._connect_signals()

        # Auto-save timer
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._auto_save)
        self._auto_save_timer.start(60000)  # Auto-save every minute

        # Show welcome overlay after UI is ready
        QTimer.singleShot(0, self._show_welcome_overlay)

    # ------------------------------------------------------------------ setup

    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("Air Classifier Designer")
        self.setMinimumSize(1400, 900)
        self.resize(1800, 1100)

        # Enable docking features
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.AnimatedDocks
        )

    def _show_welcome_overlay(self):
        """Show the welcome overlay on top of the central widget."""
        self._welcome = _WelcomeOverlay(self)
        self._welcome.new_project_clicked.connect(self._dismiss_welcome_and_new)
        self._welcome.open_project_clicked.connect(self._dismiss_welcome_and_open)
        self._welcome.load_preset_clicked.connect(self._dismiss_welcome_and_preset)
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

    def resizeEvent(self, event):
        """Keep overlay sized to the central area."""
        super().resizeEvent(event)
        if hasattr(self, "_welcome") and self._welcome is not None and self._welcome.isVisible():
            self._welcome.setGeometry(self.centralWidget().geometry())

    # --------------------------------------------------------------- actions

    def _create_actions(self):
        """Create all application actions."""
        # File actions
        self.action_new = QAction("&New Project", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.setStatusTip("Create a new air classifier project")
        self.action_new.triggered.connect(self.new_project)

        self.action_open = QAction("&Open Project...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.setStatusTip("Open an existing project")
        self.action_open.triggered.connect(self.open_project)

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.setStatusTip("Save the current project")
        self.action_save.triggered.connect(self.save_project)

        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.action_save_as.setStatusTip("Save project to a new file")
        self.action_save_as.triggered.connect(self.save_project_as)

        self.action_export_mesh = QAction("Export &Mesh...", self)
        self.action_export_mesh.setStatusTip("Export geometry as STL/OBJ")
        self.action_export_mesh.triggered.connect(self.export_mesh)

        self.action_export_results = QAction("Export &Results...", self)
        self.action_export_results.setStatusTip("Export simulation results")
        self.action_export_results.triggered.connect(self.export_results)

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.setStatusTip("Exit the application")
        self.action_exit.triggered.connect(self.close)

        # Edit actions
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.triggered.connect(self.undo)

        self.action_redo = QAction("&Redo", self)
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.action_redo.triggered.connect(self.redo)

        self.action_preferences = QAction("&Preferences...", self)
        self.action_preferences.setShortcut(QKeySequence("Ctrl+,"))
        self.action_preferences.triggered.connect(self.show_preferences)

        # View actions
        self.action_reset_layout = QAction("&Reset Layout", self)
        self.action_reset_layout.setStatusTip("Reset window layout to default")
        self.action_reset_layout.triggered.connect(self.reset_layout)

        self.action_fullscreen = QAction("&Fullscreen", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.triggered.connect(self.toggle_fullscreen)

        # Assembly actions
        self.action_add_component = QAction("&Configure Assembly...", self)
        self.action_add_component.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.action_add_component.setStatusTip("Open assembly configuration (components, dimensions, mode)")
        self.action_add_component.triggered.connect(self.add_component)

        self.action_build_system = QAction("&Build Full System", self)
        self.action_build_system.setShortcut(QKeySequence("Ctrl+B"))
        self.action_build_system.setStatusTip("Build and preview the complete classifier assembly in 3D")
        self.action_build_system.triggered.connect(self.build_full_system)

        # Simulation actions
        self.action_run_sim = QAction("&Run Simulation", self)
        self.action_run_sim.setShortcut(QKeySequence("F5"))
        self.action_run_sim.triggered.connect(self.run_simulation)

        self.action_pause_sim = QAction("&Pause", self)
        self.action_pause_sim.setShortcut(QKeySequence("F6"))
        self.action_pause_sim.setEnabled(False)
        self.action_pause_sim.triggered.connect(self.pause_simulation)

        self.action_stop_sim = QAction("&Stop", self)
        self.action_stop_sim.setShortcut(QKeySequence("Shift+F5"))
        self.action_stop_sim.setEnabled(False)
        self.action_stop_sim.triggered.connect(self.stop_simulation)

        self.action_sim_settings = QAction("Simulation &Settings...", self)
        self.action_sim_settings.triggered.connect(self.show_simulation_settings)

        # Help actions
        self.action_help = QAction("&Documentation", self)
        self.action_help.setShortcut(QKeySequence.StandardKey.HelpContents)
        self.action_help.triggered.connect(self.show_help)

        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self.show_about)

    # ----------------------------------------------------------------- menus

    def _create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()

        # File menu
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

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_preferences)

        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.action_reset_layout)
        view_menu.addAction(self.action_fullscreen)
        view_menu.addSeparator()
        # Dock widget toggles will be added here
        self._view_menu = view_menu

        # Assembly menu
        assembly_menu = menubar.addMenu("&Assembly")
        assembly_menu.addAction(self.action_add_component)
        assembly_menu.addSeparator()
        assembly_menu.addAction(self.action_build_system)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        sim_menu.addAction(self.action_run_sim)
        sim_menu.addAction(self.action_pause_sim)
        sim_menu.addAction(self.action_stop_sim)
        sim_menu.addSeparator()
        sim_menu.addAction(self.action_sim_settings)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.action_help)
        help_menu.addSeparator()
        help_menu.addAction(self.action_about)

    # -------------------------------------------------------------- toolbars

    def _create_toolbars(self):
        """Create application toolbars with descriptive labels."""
        # Main toolbar
        main_toolbar = QToolBar("Main", self)
        main_toolbar.setMovable(True)
        main_toolbar.setObjectName("MainToolbar")
        main_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        main_toolbar.addAction(self.action_new)
        main_toolbar.addAction(self.action_open)
        main_toolbar.addAction(self.action_save)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_add_component)  # Configure Assembly
        main_toolbar.addAction(self.action_build_system)

        self.addToolBar(main_toolbar)

        # Simulation toolbar
        sim_toolbar = QToolBar("Simulation", self)
        sim_toolbar.setMovable(True)
        sim_toolbar.setObjectName("SimulationToolbar")
        sim_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        sim_toolbar.addAction(self.action_run_sim)
        sim_toolbar.addAction(self.action_pause_sim)
        sim_toolbar.addAction(self.action_stop_sim)

        self.addToolBar(sim_toolbar)

    # --------------------------------------------------------- central widget

    def _create_central_widget(self):
        """Create the central widget with viewport and assembly canvas."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D Viewport takes the full central area
        self.viewport_3d = Viewport3D()
        self.viewport_3d.setMinimumHeight(400)
        layout.addWidget(self.viewport_3d)

        self.setCentralWidget(central_widget)

    # ------------------------------------------------------------- dock area

    def _create_dock_widgets(self):
        """Create dock widgets for simulation and results."""
        # Simulation Control (bottom)
        self.sim_control = SimulationControlPanel()
        self.sim_control.setMaximumHeight(350)
        sim_dock = QDockWidget("Simulation", self)
        sim_dock.setObjectName("SimulationDock")
        sim_dock.setWidget(self.sim_control)
        sim_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, sim_dock)
        self._view_menu.addAction(sim_dock.toggleViewAction())

        # Results Panel (bottom, tabbed with simulation)
        self.results_panel = ResultsPanel()
        self.results_panel.setMaximumHeight(350)
        results_dock = QDockWidget("Results", self)
        results_dock.setObjectName("ResultsDock")
        results_dock.setWidget(self.results_panel)
        results_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, results_dock)
        self.tabifyDockWidget(sim_dock, results_dock)
        self._view_menu.addAction(results_dock.toggleViewAction())

        # Raise simulation dock by default
        sim_dock.raise_()

    # -------------------------------------------------------------- statusbar

    def _create_status_bar(self):
        """Create a segmented status bar with clear zones."""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Left: transient messages (handled by Qt automatically)

        # Right permanent section: GPU | components | progress
        self._component_count_label = QLabel("0 components")
        self._component_count_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        status_bar.addPermanentWidget(self._component_count_label)

        status_bar.addPermanentWidget(_StatusSeparator())

        # GPU status
        self.gpu_status_label = QLabel("GPU: Checking...")
        self.gpu_status_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED};")
        status_bar.addPermanentWidget(self.gpu_status_label)

        status_bar.addPermanentWidget(_StatusSeparator())

        # Simulation progress
        self.sim_progress = QProgressBar()
        self.sim_progress.setFixedWidth(140)
        self.sim_progress.setFixedHeight(14)
        self.sim_progress.setTextVisible(False)
        self.sim_progress.setVisible(False)
        status_bar.addPermanentWidget(self.sim_progress)

        # Check GPU status
        QTimer.singleShot(100, self._check_gpu_status)

    def _check_gpu_status(self):
        """Check NVIDIA GPU and Warp availability."""
        try:
            import warp as wp
            wp.init()
            devices = wp.get_devices()
            cuda_devices = [d for d in devices if "cuda" in str(d).lower()]
            if cuda_devices:
                self.gpu_status_label.setText(f"GPU: {cuda_devices[0]} Ready")
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
        """Refresh the component count label in the status bar."""
        # Count is based on assembly params (subsystems enabled)
        p = self._assembly_params
        count = 1  # classification is always present
        if p.get("include_feed_system", True):
            count += 1
        if p.get("include_air_system", True):
            count += 1
        if p.get("include_exhaust", True):
            count += 1
        self._component_count_label.setText(f"{count} subsystems")

    # --------------------------------------------------------- state save/restore

    def _restore_state(self):
        """Restore window state from settings."""
        settings = QSettings()
        geometry = settings.value("MainWindow/geometry")
        state = settings.value("MainWindow/state")

        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def _save_state(self):
        """Save window state to settings."""
        settings = QSettings()
        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())

    # --------------------------------------------------------- signal wiring

    def _connect_signals(self):
        """Connect internal signals."""
        # Simulation control signals
        self.sim_control.run_requested.connect(self.run_simulation)
        self.sim_control.pause_requested.connect(self.pause_simulation)
        self.sim_control.stop_requested.connect(self.stop_simulation)

        # Simulation results -> Results panel + stop animation
        self.sim_control.simulation_results_ready.connect(
            self.results_panel.set_results
        )
        self.sim_control.simulation_results_ready.connect(
            self._on_simulation_finished
        )

        # Sync animation time with simulation progress
        self.sim_control.sim_time_updated.connect(self._on_sim_time_updated)

        # Physics-driven animation: forward component state from simulation to animation
        self.sim_control.component_state_updated.connect(self._on_component_state_updated)

    def _update_window_title(self):
        """Update window title based on project state."""
        title = "Air Classifier Designer"
        if self._project_path:
            title = f"{self._project_path.name} - {title}"
        if self._is_modified:
            title = f"*{title}"
        self.setWindowTitle(title)

    def _set_modified(self, modified: bool = True):
        """Set the modified state of the project."""
        self._is_modified = modified
        self._update_window_title()

    # ================================================================
    #  File Operations
    # ================================================================

    @Slot()
    def new_project(self):
        """Create a new project."""
        if self._is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Do you want to save changes before creating a new project?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        self._project_path = None
        self._assembly_config = {}
        self._assembly_params = {}
        self._is_modified = False
        self._built_backend = None
        self.viewport_3d.clear()
        self._update_window_title()
        self._update_component_count()
        self.statusBar().showMessage("New project created", 3000)

    @Slot()
    def open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project",
            str(Path.home()),
            "Air Classifier Project (*.acproj);;All Files (*)"
        )
        if file_path:
            self._load_project(Path(file_path))

    def _load_project(self, path: Path):
        """Load a project from file."""
        try:
            import json
            with open(path, 'r') as f:
                data = json.load(f)

            self._project_path = path
            self._assembly_config = data.get("assembly", {})
            self._assembly_params = data.get("assembly_params", {})
            self.viewport_3d.load_state(data.get("viewport", {}))
            self._is_modified = False
            self._update_window_title()
            self._update_component_count()
            self.project_changed.emit(str(path))
            self.statusBar().showMessage(f"Opened: {path.name}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open project:\n{str(e)}"
            )

    @Slot()
    def save_project(self) -> bool:
        """Save the current project."""
        if self._project_path:
            return self._save_project_to(self._project_path)
        else:
            return self.save_project_as()

    @Slot()
    def save_project_as(self) -> bool:
        """Save project to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            str(Path.home() / "untitled.acproj"),
            "Air Classifier Project (*.acproj);;All Files (*)"
        )
        if file_path:
            return self._save_project_to(Path(file_path))
        return False

    def _save_project_to(self, path: Path) -> bool:
        """Save project to specified path."""
        try:
            import json
            data = {
                "version": "1.0",
                "assembly": self._assembly_config,
                "assembly_params": self._assembly_params,
                "viewport": self.viewport_3d.save_state(),
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            self._project_path = path
            self._is_modified = False
            self._update_window_title()
            self.statusBar().showMessage(f"Saved: {path.name}", 3000)
            return True
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save project:\n{str(e)}"
            )
            return False

    @Slot()
    def _auto_save(self):
        """Auto-save if project has been modified."""
        if self._is_modified and self._project_path:
            self._save_project_to(self._project_path)

    @Slot()
    def export_mesh(self):
        """Export geometry as mesh file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Mesh",
            str(Path.home() / "classifier.stl"),
            "STL Files (*.stl);;OBJ Files (*.obj);;VTK Files (*.vtk)"
        )
        if file_path:
            try:
                self.viewport_3d.export_mesh(file_path)
                self.statusBar().showMessage(f"Exported: {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    @Slot()
    def export_results(self):
        """Export simulation results."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results",
            str(Path.home() / "results.csv"),
            "CSV Files (*.csv);;JSON Files (*.json);;VTK Files (*.vtk)"
        )
        if file_path:
            try:
                self.results_panel.export_results(file_path)
                self.statusBar().showMessage(f"Results exported: {Path(file_path).name}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # ================================================================
    #  Edit Operations
    # ================================================================

    @Slot()
    def undo(self):
        """Undo last action."""
        pass  # Assembly is now params-driven (no undo stack)

    @Slot()
    def redo(self):
        """Redo last undone action."""
        pass  # Assembly is now params-driven (no undo stack)

    @Slot()
    def show_preferences(self):
        """Show preferences dialog."""
        from .dialogs.preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec()

    # ================================================================
    #  View Operations
    # ================================================================

    @Slot()
    def reset_layout(self):
        """Reset window layout to default."""
        settings = QSettings()
        settings.remove("MainWindow/state")
        settings.remove("MainWindow/geometry")

        QMessageBox.information(
            self, "Layout Reset",
            "Layout will be reset on next application start."
        )

    @Slot(bool)
    def toggle_fullscreen(self, checked: bool):
        """Toggle fullscreen mode."""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()

    # ================================================================
    #  Assembly Operations
    # ================================================================

    @Slot()
    def add_component(self):
        """Open the assembly configuration dialog."""
        self.show_assembly_config()

    @Slot()
    def delete_component(self):
        """No-op -- assembly is params-driven now."""
        pass

    @Slot()
    def auto_connect_ports(self):
        """No-op -- connections are automatic in params-driven assemblies."""
        self.statusBar().showMessage("Connections are automatic in the assembly system", 3000)

    @Slot()
    def validate_assembly(self):
        """Build the assembly to validate it."""
        self.build_full_system()

    @Slot()
    def load_preset(self):
        """Open assembly configuration dialog (replaces old preset dialog)."""
        self.show_assembly_config()

    @Slot()
    def show_assembly_config(self):
        """Open the Assembly Configuration dialog."""
        from .dialogs.assembly_config_dialog import AssemblyConfigDialog

        dialog = AssemblyConfigDialog(self, self._assembly_params)
        dialog.assembly_configured.connect(self._on_assembly_configured)
        dialog.exec()

    def _on_assembly_configured(self, params: Dict[str, Any]):
        """Handle new assembly params from the Assembly Config dialog.

        Syncs shared fields (wheel RPM, air flow, mode, etc.) into the
        Simulation Settings so both panels stay consistent.
        """
        self._assembly_params = params
        self._set_modified(True)
        self._update_component_count()

        # Sync assembly params -> simulation settings
        s = self.sim_control.get_settings()
        if "wheel_rpm" in params:
            s.wheel_rpm = params["wheel_rpm"]
        if "wheel_diameter" in params:
            s.wheel_diameter = params["wheel_diameter"]
        if "use_preclassification" in params:
            s.use_preclassification = params["use_preclassification"]
        if "include_feed_system" in params:
            s.include_feed_system = params["include_feed_system"]
        if "include_air_system" in params:
            s.include_air_system = params["include_air_system"]
        if "include_exhaust" in params:
            s.include_exhaust = params["include_exhaust"]
        # Sync geometry overrides that exist in both places
        if "venturi_throat_ratio" in params and "venturi_inlet_diameter" in params:
            throat_mm = params["venturi_inlet_diameter"] * params["venturi_throat_ratio"] * 1000
            s.venturi_throat_diameter_mm = throat_mm
        if "zigzag_channel_width" in params:
            s.zigzag_width_mm = params["zigzag_channel_width"] * 1000
        if "zigzag_channel_depth" in params:
            s.zigzag_depth_mm = params["zigzag_channel_depth"] * 1000
        # Push updated settings back to the UI widgets
        self.sim_control.set_settings(s)

        # Immediately build and preview
        self.build_full_system()

    @Slot()
    def build_full_system(self):
        """Build and display the complete classifier assembly in 3D viewport."""
        from .simulation_backend import SimulationConfig, SimulationBackend

        self.statusBar().showMessage("Building complete system...")

        try:
            settings = self.sim_control.get_settings()
            p = self._assembly_params

            config = SimulationConfig(
                assembly_data={},
                use_preclassification=p.get("use_preclassification", settings.use_preclassification),
                wheel_diameter=p.get("wheel_diameter", settings.wheel_diameter),
                wheel_rpm=p.get("wheel_rpm", settings.wheel_rpm),
                include_feed_system=p.get("include_feed_system", settings.include_feed_system),
                include_air_system=p.get("include_air_system", settings.include_air_system),
                include_exhaust=p.get("include_exhaust", settings.include_exhaust),
                # Classification geometry params from Assembly Config dialog
                venturi_inlet_diameter=p.get("venturi_inlet_diameter", 0.08),
                venturi_throat_ratio=p.get("venturi_throat_ratio", 0.5),
                zigzag_channel_width=p.get("zigzag_channel_width", 0.15),
                zigzag_channel_depth=p.get("zigzag_channel_depth", 0.25),
                zigzag_num_stages=p.get("zigzag_num_stages", 5),
                primary_cyclone_diameter=p.get("primary_cyclone_diameter", 0.30),
                secondary_cyclone_diameter=p.get("secondary_cyclone_diameter", 0.20),
                tertiary_cyclone_diameter=p.get("tertiary_cyclone_diameter", 0.12),
                device="cpu",
            )

            backend = SimulationBackend(config)
            backend._build_assembly_from_gui()

            vertices, indices = backend.get_mesh()
            if vertices is not None and len(vertices) > 0:
                self.viewport_3d.update_from_backend_mesh(vertices, indices)

                # Set up animation controller from the COMPLETE assembly.
                # This only REGISTERS the animated parts (wheel, blower, dampers,
                # lid, etc.) -- it does NOT start any animation or physics.
                # Animation begins only when the user clicks Run Simulation.
                self._stop_animation()
                assembly_obj = getattr(backend, '_complete_assembly', None) or getattr(backend, '_assembly', None)
                if assembly_obj is not None:
                    ctrl = self.viewport_3d.build_with_animation(assembly_obj)
                    if ctrl is not None:
                        self._animation_controller = ctrl
                        self.sim_control._log("Animation: registered rotating components")

                # Keep reference so Run Simulation can skip canvas validation
                self._built_backend = backend

                mode = "Full System" if p.get("use_preclassification", True) else "Wheel-Only"
                msg = f"Built {mode}: {len(vertices):,} vertices, {len(indices)//3:,} triangles"
                self.statusBar().showMessage(msg, 5000)

                summary = backend.get_system_summary()
                self.sim_control._log(f"System built: {summary.get('mode', 'unknown')}")
                self.sim_control._log("Ready to run simulation.")
            else:
                self._built_backend = None
                QMessageBox.warning(self, "Build Failed", "Could not generate mesh geometry.")

        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Build Error",
                f"Failed to build system:\n{e}\n\n{traceback.format_exc()}"
            )
            self.statusBar().showMessage("Build failed", 3000)

    # ================================================================
    #  Simulation Operations
    # ================================================================

    # Startup preamble: animation plays the full startup sequence
    # (air ramp, dampers open, lid open→close, wheel spin-up) BEFORE
    # the classification physics begins.  This matches the real machine
    # where the system must be at steady-state before material is classified.
    STARTUP_PREAMBLE_MS = 8000   # 8 seconds (matches AnimationTimeline.steady_time)
    SHUTDOWN_DURATION_MS = 4500  # 3s shutdown anim + 1.5s buffer for force-stop

    @Slot()
    def run_simulation(self):
        """Start the simulation.

        If a system hasn't been built yet, build it first automatically.

        Sequence:
        1. Start the mechanical animation (startup preamble: blower ramp,
           dampers open, lid open→close, wheel spin-up).
        2. After the preamble completes (~8s), start the classification
           particle physics so the system is at steady-state.
        3. When physics finishes, play the shutdown sequence.
        """
        # Auto-build if not built yet
        if self._built_backend is None:
            self.build_full_system()

        # Bump generation so stale timers from a previous run are ignored
        self._anim_generation += 1

        self._start_simulation_run()

        # Reset the sim-control progress bar and KPI cards NOW so the user
        # doesn't see stale 100% / old values during the 8s preamble.
        self.sim_control.progress_bar.setValue(0)
        self.sim_control.card_time.set_value("0.000 s")
        self.sim_control.card_particles.set_value("0")
        self.sim_control.card_fines.set_value("0")
        self.sim_control.card_coarse.set_value("0")
        self.sim_control.card_efficiency.set_value("--")

        # Start cinematic camera if the user has it enabled
        if self.viewport_3d.cinematic_enabled:
            self.viewport_3d.start_cinematic()

        # Start mechanical animations -- preamble runs first
        self._start_animation()

        # Delay the simulation start until the startup preamble completes
        # so the animation shows the full startup sequence before particles flow.
        self.sim_control._log(
            f"Startup preamble: {self.STARTUP_PREAMBLE_MS / 1000:.0f}s "
            "(air → feed → classification)..."
        )
        gen = self._anim_generation
        QTimer.singleShot(
            self.STARTUP_PREAMBLE_MS,
            lambda g=gen: self._start_simulation_after_preamble(g),
        )

    def _start_simulation_after_preamble(self, generation: int = -1):
        """Called after the startup preamble animation completes."""
        if generation >= 0 and generation != self._anim_generation:
            return  # stale timer from a cancelled/restarted run
        if self._simulation_state != "running":
            return  # User cancelled during preamble
        self.sim_control._log("Preamble complete — starting classification physics")
        self.sim_control.start_simulation({})

    def _start_animation(self):
        """Start the mechanical animation sequence with physics-driven simulators."""
        if self._animation_controller is None:
            return

        # Create subsidiary physics simulators from the built assembly
        # so the animation uses real VFD ramp / damper / lid servo dynamics.
        if self._built_backend is not None:
            try:
                subs = self._built_backend.create_subsidiary_simulators()
                air_s = subs.get("air_sim")
                feed_s = subs.get("feed_sim")
                if air_s or feed_s:
                    self._animation_controller.set_subsidiary_simulators(air_s, feed_s)
                    parts = []
                    if air_s:
                        parts.append("air (blower+dampers)")
                    if feed_s:
                        parts.append("feed (lid servo)")
                    self.sim_control._log(f"Animation physics: {', '.join(parts)}")
            except Exception as e:
                self.sim_control._log(f"Animation physics init skipped: {e}")

        from .widgets.animation_controller import AnimationTimeline
        timeline = AnimationTimeline(
            air_start_time=0.0,
            air_ramp_duration=2.0,
            feed_start_time=3.0,
            feed_ramp_duration=2.0,
            classification_start_time=5.0,
            classification_ramp_duration=2.0,
            steady_time=8.0,
        )
        self._animation_controller.start(timeline)
        self.sim_control._log("Animation started: Air \u2192 Feed \u2192 Classification")

    def _stop_animation(self):
        """Begin shutdown animation (dampers close, lid closes, ramp down).

        The shutdown uses wall-clock time for progress, so the animation
        completes within ``_SHUTDOWN_ANIM_S`` seconds regardless of how
        fast animation ticks fire.  A generous force-stop timer fires
        well after the animation should have completed as a safety net.

        Uses a generation counter so stale timers from a previous run
        cannot interfere with a new run's animation.
        """
        _SHUTDOWN_ANIM_S = 3.0          # animation duration (wall-clock)
        _FORCE_STOP_MS = int(_SHUTDOWN_ANIM_S * 1000) + 1500  # safety buffer

        if self._animation_controller is not None:
            phase = self._animation_controller.phase.value
            if phase in ("steady_state", "classification", "feed_startup", "air_startup"):
                # Graceful shutdown: ramp everything to closed
                gen = self._anim_generation
                self._animation_controller.begin_shutdown(duration=_SHUTDOWN_ANIM_S)
                # Capture generation so the lambda is a no-op if a new run started
                QTimer.singleShot(
                    _FORCE_STOP_MS,
                    lambda g=gen: self._force_stop_animation(g),
                )
            else:
                self._animation_controller.stop()
                self._animation_controller.render_initial_state()

    def _force_stop_animation(self, generation: int = -1):
        """Force-stop after shutdown delay, render parts at resting state.

        This is a safety net -- the AnimationController now auto-completes
        the shutdown when progress reaches 100%.  If it already stopped
        itself, render_initial_state() is still safe to call (idempotent).

        Ignores the call if the generation counter has advanced (a new run
        started since this timer was scheduled).
        """
        if generation >= 0 and generation != self._anim_generation:
            return  # stale timer from a previous run -- ignore
        if self._animation_controller is not None:
            self._animation_controller.stop()
            # render_initial_state() explicitly resets all parts to frac=0
            # (closed dampers, closed lid, stopped rotors) before rendering.
            self._animation_controller.render_initial_state()

    @Slot(float)
    def _on_sim_time_updated(self, sim_time: float):
        """Sync animation time with simulation progress.

        The simulation reports sim_time starting from 0, but the animation
        already spent STARTUP_PREAMBLE_MS on the startup sequence.  Offset
        so the animation stays at steady-state during classification.
        """
        if self._animation_controller is not None:
            anim_time = sim_time + self.STARTUP_PREAMBLE_MS / 1000.0
            self._animation_controller.sync_to_sim_time(anim_time)

    @Slot(dict)
    def _on_component_state_updated(self, component_state: dict):
        """Forward physics-driven component states to animation controller.

        Offsets sim_time by the startup preamble so the animation controller
        sees animation-time (startup already completed) rather than raw sim_time.
        """
        if self._animation_controller is not None:
            # Offset sim_time so animation stays at steady-state
            preamble_s = self.STARTUP_PREAMBLE_MS / 1000.0
            component_state = dict(component_state)  # shallow copy
            component_state["sim_time"] = component_state.get("sim_time", 0.0) + preamble_s
            self._animation_controller.update_from_physics(component_state)

    @Slot(dict)
    def _on_simulation_finished(self, results: Dict[str, Any]):
        """Called when classification physics completes -- play shutdown sequence.

        The shutdown animation (dampers close, lid closes, blower ramps down)
        plays for SHUTDOWN_DURATION_MS before the simulation is declared
        fully complete.  This matches the real machine shutdown procedure.
        """
        self.action_pause_sim.setEnabled(False)
        self.sim_control._log("Classification complete — shutting down system...")

        # Start shutdown animation (dampers close, lid close, blower ramp down)
        self._stop_animation()

        # Keep simulation state as "running" during shutdown so the UI
        # shows the shutdown animation.  After the shutdown completes,
        # _on_shutdown_complete finalizes everything.
        gen = self._anim_generation
        QTimer.singleShot(
            self.SHUTDOWN_DURATION_MS,
            lambda g=gen: self._on_shutdown_complete(g),
        )

    def _on_shutdown_complete(self, generation: int = -1):
        """Called after the shutdown animation finishes."""
        if generation >= 0 and generation != self._anim_generation:
            return  # stale timer from a previous run
        self._simulation_state = "idle"
        self.action_run_sim.setEnabled(True)
        self.action_stop_sim.setEnabled(False)
        self.sim_progress.setVisible(False)
        # Stop cinematic camera when simulation is fully done
        self.viewport_3d.stop_cinematic()
        self.sim_control._log("System shutdown complete — dampers closed, lid closed.")
        self.statusBar().showMessage("Simulation complete", 5000)

    def _start_simulation_run(self):
        """Shared UI state change when a simulation run begins."""
        self._simulation_state = "running"
        self.simulation_state_changed.emit("running")
        self.action_run_sim.setEnabled(False)
        self.action_pause_sim.setEnabled(True)
        self.action_stop_sim.setEnabled(True)
        self.sim_progress.setVisible(True)
        self.sim_progress.setValue(0)

    @Slot()
    def pause_simulation(self):
        """Pause the running simulation."""
        self._simulation_state = "paused"
        self.simulation_state_changed.emit("paused")
        self.action_run_sim.setEnabled(True)
        self.action_pause_sim.setEnabled(False)
        self.sim_control.pause_simulation()

    @Slot()
    def stop_simulation(self):
        """Stop the simulation and animations."""
        self._simulation_state = "idle"
        self.simulation_state_changed.emit("idle")
        self.action_run_sim.setEnabled(True)
        self.action_pause_sim.setEnabled(False)
        self.action_stop_sim.setEnabled(False)
        self.sim_progress.setVisible(False)
        self._stop_animation()
        self.viewport_3d.stop_cinematic()
        self.sim_control.stop_simulation()

    @Slot()
    def show_simulation_settings(self):
        """Show simulation settings dialog."""
        from .dialogs.simulation_settings_dialog import SimulationSettingsDialog
        dialog = SimulationSettingsDialog(self.sim_control.get_settings(), self)
        if dialog.exec():
            self.sim_control.set_settings(dialog.get_settings())

    # ================================================================
    #  Help Operations
    # ================================================================

    @Slot()
    def show_help(self):
        """Show documentation."""
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl("https://airclassifier.readthedocs.io"))

    @Slot()
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Air Classifier Designer",
            f"""<h2>Air Classifier Designer</h2>
            <p>Version 1.0.0</p>
            <p>Interactive design and simulation tool for air classification systems.</p>
            <p>Powered by NVIDIA Warp for GPU-accelerated multiphysics simulation.</p>
            <hr>
            <p><b>Features:</b></p>
            <ul>
                <li>Visual component assembly</li>
                <li>Real-time 3D preview</li>
                <li>CFD + particle dynamics simulation</li>
                <li>Separation efficiency analysis</li>
            </ul>
            """
        )

    # ================================================================
    #  Event Handlers
    # ================================================================

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        if self._is_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Do you want to save changes before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Stop any running simulation
        if self._simulation_state == "running":
            self.stop_simulation()

        # Save window state
        self._save_state()
        event.accept()
