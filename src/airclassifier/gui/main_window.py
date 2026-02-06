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
    QMenu, QMessageBox, QFileDialog, QSplitter,
    QLabel, QProgressBar,
)
from PySide6.QtCore import Qt, QSettings, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent

from .panels.component_palette import ComponentPalette
from .panels.property_editor import PropertyEditor
from .panels.simulation_control import SimulationControlPanel
from .panels.results_panel import ResultsPanel
from .widgets.viewport_3d import Viewport3D
from .widgets.assembly_canvas import AssemblyCanvas


class MainWindow(QMainWindow):
    """
    Main application window for Air Classifier Designer.

    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Menu Bar                                                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Tool Bar                                                       │
    ├────────────┬────────────────────────────────┬───────────────────┤
    │            │                                │                   │
    │ Component  │     Central Area               │   Property        │
    │ Palette    │  ┌──────────────────────────┐  │   Editor          │
    │            │  │   3D Viewport            │  │                   │
    │            │  │                          │  │                   │
    │            │  ├──────────────────────────┤  │                   │
    │            │  │   Assembly Canvas        │  │                   │
    │            │  │   (Node Editor)          │  │                   │
    │            │  └──────────────────────────┘  │                   │
    │            │                                │                   │
    ├────────────┴────────────────────────────────┴───────────────────┤
    │  Simulation Control / Results Panel (Tabbed)                    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Status Bar                                                     │
    └─────────────────────────────────────────────────────────────────┘
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
        self.action_add_component = QAction("&Add Component", self)
        self.action_add_component.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.action_add_component.triggered.connect(self.add_component)

        self.action_delete_component = QAction("&Delete Component", self)
        self.action_delete_component.setShortcut(QKeySequence.StandardKey.Delete)
        self.action_delete_component.triggered.connect(self.delete_component)

        self.action_auto_connect = QAction("Auto-&Connect Ports", self)
        self.action_auto_connect.setStatusTip("Automatically connect compatible ports")
        self.action_auto_connect.triggered.connect(self.auto_connect_ports)

        self.action_validate = QAction("&Validate Assembly", self)
        self.action_validate.setShortcut(QKeySequence("Ctrl+Shift+V"))
        self.action_validate.triggered.connect(self.validate_assembly)

        self.action_load_preset = QAction("Load &Preset...", self)
        self.action_load_preset.setStatusTip("Load a predefined classifier configuration")
        self.action_load_preset.triggered.connect(self.load_preset)

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
        assembly_menu.addAction(self.action_delete_component)
        assembly_menu.addSeparator()
        assembly_menu.addAction(self.action_auto_connect)
        assembly_menu.addAction(self.action_validate)
        assembly_menu.addSeparator()
        assembly_menu.addAction(self.action_load_preset)
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

    def _create_toolbars(self):
        """Create application toolbars."""
        # Main toolbar
        main_toolbar = QToolBar("Main", self)
        main_toolbar.setMovable(True)
        main_toolbar.setObjectName("MainToolbar")

        main_toolbar.addAction(self.action_new)
        main_toolbar.addAction(self.action_open)
        main_toolbar.addAction(self.action_save)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_undo)
        main_toolbar.addAction(self.action_redo)

        self.addToolBar(main_toolbar)

        # Simulation toolbar
        sim_toolbar = QToolBar("Simulation", self)
        sim_toolbar.setMovable(True)
        sim_toolbar.setObjectName("SimulationToolbar")

        sim_toolbar.addAction(self.action_run_sim)
        sim_toolbar.addAction(self.action_pause_sim)
        sim_toolbar.addAction(self.action_stop_sim)
        sim_toolbar.addSeparator()
        sim_toolbar.addAction(self.action_validate)
        sim_toolbar.addAction(self.action_build_system)

        self.addToolBar(sim_toolbar)

    def _create_central_widget(self):
        """Create the central widget with viewport and assembly canvas."""
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create splitter for viewport and assembly canvas
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # 3D Viewport (top) - give it more space
        self.viewport_3d = Viewport3D()
        self.viewport_3d.setMinimumHeight(400)  # Ensure minimum height for viewport
        self.main_splitter.addWidget(self.viewport_3d)

        # Assembly Canvas (bottom) - node editor for schematic view
        self.assembly_canvas = AssemblyCanvas()
        self.assembly_canvas.setMinimumHeight(150)
        self.assembly_canvas.setMaximumHeight(300)  # Limit canvas height
        self.main_splitter.addWidget(self.assembly_canvas)

        # Set initial sizes (80% viewport, 20% canvas)
        self.main_splitter.setSizes([800, 200])
        self.main_splitter.setStretchFactor(0, 4)  # Viewport gets 4x stretch
        self.main_splitter.setStretchFactor(1, 1)  # Canvas gets 1x stretch

        layout.addWidget(self.main_splitter)
        self.setCentralWidget(central_widget)

    def _create_dock_widgets(self):
        """Create all dock widgets."""
        # Component Palette (left) - constrain width to give more room to viewport
        self.component_palette = ComponentPalette()
        self.component_palette.setMinimumWidth(180)
        self.component_palette.setMaximumWidth(280)
        palette_dock = QDockWidget("Components", self)
        palette_dock.setObjectName("ComponentPaletteDock")
        palette_dock.setWidget(self.component_palette)
        palette_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, palette_dock)
        self._view_menu.addAction(palette_dock.toggleViewAction())

        # Property Editor (right) - constrain width to give more room to viewport
        self.property_editor = PropertyEditor()
        self.property_editor.setMinimumWidth(200)
        self.property_editor.setMaximumWidth(320)
        property_dock = QDockWidget("Properties", self)
        property_dock.setObjectName("PropertyEditorDock")
        property_dock.setWidget(self.property_editor)
        property_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, property_dock)
        self._view_menu.addAction(property_dock.toggleViewAction())

        # Simulation Control (bottom) - constrain height to give more room to viewport
        self.sim_control = SimulationControlPanel()
        self.sim_control.setMaximumHeight(350)  # Limit panel height
        sim_dock = QDockWidget("Simulation", self)
        sim_dock.setObjectName("SimulationDock")
        sim_dock.setWidget(self.sim_control)
        sim_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, sim_dock)
        self._view_menu.addAction(sim_dock.toggleViewAction())

        # Results Panel (bottom, tabbed with simulation)
        self.results_panel = ResultsPanel()
        self.results_panel.setMaximumHeight(350)  # Limit panel height
        results_dock = QDockWidget("Results", self)
        results_dock.setObjectName("ResultsDock")
        results_dock.setWidget(self.results_panel)
        results_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, results_dock)
        self.tabifyDockWidget(sim_dock, results_dock)
        self._view_menu.addAction(results_dock.toggleViewAction())

        # Raise simulation dock by default
        sim_dock.raise_()

    def _create_status_bar(self):
        """Create the status bar."""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # GPU status
        self.gpu_status_label = QLabel("GPU: Checking...")
        status_bar.addPermanentWidget(self.gpu_status_label)

        # Simulation progress
        self.sim_progress = QProgressBar()
        self.sim_progress.setMaximumWidth(200)
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
                self.gpu_status_label.setStyleSheet("color: #4ec9b0;")
            else:
                self.gpu_status_label.setText("GPU: CPU Mode (No CUDA)")
                self.gpu_status_label.setStyleSheet("color: #dcdcaa;")
        except ImportError:
            self.gpu_status_label.setText("GPU: Warp Not Installed")
            self.gpu_status_label.setStyleSheet("color: #f14c4c;")
        except Exception as e:
            self.gpu_status_label.setText(f"GPU: Error - {str(e)[:30]}")
            self.gpu_status_label.setStyleSheet("color: #f14c4c;")

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

    def _connect_signals(self):
        """Connect internal signals."""
        # Component palette -> Assembly canvas
        self.component_palette.component_selected.connect(
            self.assembly_canvas.add_component_node
        )

        # Assembly canvas -> Property editor
        self.assembly_canvas.node_selected.connect(
            self.property_editor.set_component
        )

        # Property editor -> Assembly canvas (parameter updates)
        self.property_editor.parameter_changed.connect(
            self.assembly_canvas.update_node_params
        )

        # Assembly canvas -> 3D viewport
        self.assembly_canvas.assembly_changed.connect(
            self.viewport_3d.update_assembly
        )

        # Simulation control signals
        self.sim_control.run_requested.connect(self.run_simulation)
        self.sim_control.pause_requested.connect(self.pause_simulation)
        self.sim_control.stop_requested.connect(self.stop_simulation)

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

    # --- File Operations ---

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
        self._is_modified = False
        self.assembly_canvas.clear()
        self.viewport_3d.clear()
        self.property_editor.clear()
        self._update_window_title()
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
            self.assembly_canvas.load_state(data.get("canvas", {}))
            self.viewport_3d.load_state(data.get("viewport", {}))
            self._is_modified = False
            self._update_window_title()
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
                "canvas": self.assembly_canvas.save_state(),
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

    # --- Edit Operations ---

    @Slot()
    def undo(self):
        """Undo last action."""
        self.assembly_canvas.undo()

    @Slot()
    def redo(self):
        """Redo last undone action."""
        self.assembly_canvas.redo()

    @Slot()
    def show_preferences(self):
        """Show preferences dialog."""
        from .dialogs.preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.exec()

    # --- View Operations ---

    @Slot()
    def reset_layout(self):
        """Reset window layout to default."""
        # Remove saved state
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

    # --- Assembly Operations ---

    @Slot()
    def add_component(self):
        """Add a new component to the assembly."""
        self.component_palette.show_add_dialog()

    @Slot()
    def delete_component(self):
        """Delete selected component."""
        self.assembly_canvas.delete_selected()
        self._set_modified(True)

    @Slot()
    def auto_connect_ports(self):
        """Automatically connect compatible ports."""
        connected = self.assembly_canvas.auto_connect()
        self.statusBar().showMessage(f"Auto-connected {connected} port pairs", 3000)
        if connected > 0:
            self._set_modified(True)

    @Slot()
    def validate_assembly(self):
        """Validate the current assembly."""
        errors = self.assembly_canvas.validate()
        if errors:
            error_text = "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(
                self, "Validation Errors",
                f"Assembly has the following issues:\n\n{error_text}"
            )
        else:
            QMessageBox.information(
                self, "Validation Passed",
                "Assembly is valid and ready for simulation."
            )

    @Slot()
    def load_preset(self):
        """Load a predefined classifier configuration."""
        from .dialogs.preset_dialog import PresetDialog
        dialog = PresetDialog(self)
        if dialog.exec():
            preset = dialog.selected_preset
            self.assembly_canvas.load_preset(preset)
            self.viewport_3d.rebuild_from_canvas(self.assembly_canvas)
            self._set_modified(True)
            self.statusBar().showMessage(f"Loaded preset: {preset['name']}", 3000)

    @Slot()
    def build_full_system(self):
        """
        Build and display the complete classifier assembly in 3D viewport.

        This creates the actual CompleteClassifierAssembly from complete_system.py
        based on current settings (with or without preclassification) and displays
        the full 3D geometry including feed system, air system, and exhaust.
        """
        from .simulation_backend import SimulationConfig, SimulationBackend

        self.statusBar().showMessage("Building complete system...")

        try:
            # Get settings from simulation control panel
            settings = self.sim_control.get_settings()

            # Create config from current settings
            config = SimulationConfig(
                assembly_data=self.assembly_canvas.get_assembly_data(),
                use_preclassification=settings.use_preclassification,
                wheel_diameter=settings.wheel_diameter,
                wheel_rpm=settings.wheel_rpm,
                include_feed_system=settings.include_feed_system,
                include_air_system=settings.include_air_system,
                include_exhaust=settings.include_exhaust,
                device="cpu",  # Use CPU for geometry only
            )

            # Create backend and build assembly (geometry only, no simulation)
            backend = SimulationBackend(config)
            backend._build_assembly_from_gui()

            # Get mesh and display in viewport
            vertices, indices = backend.get_mesh()
            if vertices is not None and len(vertices) > 0:
                self.viewport_3d.update_from_backend_mesh(vertices, indices)

                mode = "Full System" if settings.use_preclassification else "Wheel-Only"
                msg = f"Built {mode}: {len(vertices):,} vertices, {len(indices)//3:,} triangles"
                self.statusBar().showMessage(msg, 5000)

                # Show summary
                summary = backend.get_system_summary()
                self.sim_control._log(f"System built: {summary.get('mode', 'unknown')}")
            else:
                QMessageBox.warning(self, "Build Failed", "Could not generate mesh geometry.")

        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Build Error",
                f"Failed to build system:\n{e}\n\n{traceback.format_exc()}"
            )
            self.statusBar().showMessage("Build failed", 3000)

    # --- Simulation Operations ---

    @Slot()
    def run_simulation(self):
        """Start or resume simulation."""
        # Validate first
        errors = self.assembly_canvas.validate()
        if errors:
            QMessageBox.warning(
                self, "Cannot Run",
                "Please fix assembly errors before running simulation."
            )
            return

        self._simulation_state = "running"
        self.simulation_state_changed.emit("running")
        self.action_run_sim.setEnabled(False)
        self.action_pause_sim.setEnabled(True)
        self.action_stop_sim.setEnabled(True)
        self.sim_progress.setVisible(True)
        self.sim_progress.setValue(0)

        # Start simulation in sim control panel
        assembly_data = self.assembly_canvas.get_assembly_data()
        self.sim_control.start_simulation(assembly_data)

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
        """Stop the simulation."""
        self._simulation_state = "idle"
        self.simulation_state_changed.emit("idle")
        self.action_run_sim.setEnabled(True)
        self.action_pause_sim.setEnabled(False)
        self.action_stop_sim.setEnabled(False)
        self.sim_progress.setVisible(False)
        self.sim_control.stop_simulation()

    @Slot()
    def show_simulation_settings(self):
        """Show simulation settings dialog."""
        from .dialogs.simulation_settings_dialog import SimulationSettingsDialog
        dialog = SimulationSettingsDialog(self.sim_control.get_settings(), self)
        if dialog.exec():
            self.sim_control.set_settings(dialog.get_settings())

    # --- Help Operations ---

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
            """<h2>Air Classifier Designer</h2>
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

    # --- Event Handlers ---

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
