"""
Component Palette Panel
=======================

Tree view of available air classifier components that can be
dragged onto the assembly canvas.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QHBoxLayout, QPushButton, QDialog,
    QFormLayout, QComboBox, QDialogButtonBox, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDrag, QIcon, QColor


@dataclass
class ComponentInfo:
    """Information about an available component."""
    name: str
    category: str
    description: str
    icon: str = ""
    default_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {}


# Define all available components
COMPONENT_LIBRARY: List[ComponentInfo] = [
    # Classification Components
    ComponentInfo(
        name="Venturi Eductor",
        category="Classification",
        description="Entrains particles into airstream using converging-diverging nozzle",
        default_params={
            "inlet_diameter": 0.08,
            "throat_ratio": 0.5,
            "divergent_angle": 7.0,
        }
    ),
    ComponentInfo(
        name="Zigzag Classifier",
        category="Classification",
        description="Gravity-based pre-classifier using zigzag channel for particle separation",
        default_params={
            "channel_width": 0.06,
            "channel_depth": 0.08,
            "num_stages": 5,
            "plate_angle": 60.0,
        }
    ),
    ComponentInfo(
        name="Wheel Classifier",
        category="Classification",
        description="Centrifugal classifier for fine particle separation (d50 ~25um)",
        default_params={
            "wheel_diameter": 0.20,
            "wheel_rpm": 8000.0,
            "num_blades": 24,
            "target_d50": 25e-6,
        }
    ),
    # Cyclone Components
    ComponentInfo(
        name="Cyclone (Primary)",
        category="Cyclones",
        description="Large cyclone for coarse particle collection (d50 ~40um)",
        default_params={
            "diameter": 0.30,
            "cylinder_height_ratio": 1.5,
            "cone_height_ratio": 2.5,
            "inlet_width_ratio": 0.2,
        }
    ),
    ComponentInfo(
        name="Cyclone (Secondary)",
        category="Cyclones",
        description="Medium cyclone for intermediate particle collection (d50 ~20um)",
        default_params={
            "diameter": 0.20,
            "cylinder_height_ratio": 1.5,
            "cone_height_ratio": 2.5,
        }
    ),
    ComponentInfo(
        name="Cyclone (Tertiary)",
        category="Cyclones",
        description="Small cyclone for fine particle collection (d50 ~10um)",
        default_params={
            "diameter": 0.12,
            "cylinder_height_ratio": 1.5,
            "cone_height_ratio": 2.5,
        }
    ),
    ComponentInfo(
        name="Multi-Cyclone System",
        category="Cyclones",
        description="Series arrangement of cyclones for staged particle collection",
        default_params={
            "num_stages": 3,
            "primary_diameter": 0.30,
            "secondary_diameter": 0.20,
            "tertiary_diameter": 0.12,
        }
    ),
    # Filter Components
    ComponentInfo(
        name="Bag Filter",
        category="Filtration",
        description="Final fine particle capture with fabric bags",
        default_params={
            "flow_rate": 0.15,
            "air_to_cloth": 2.5,
            "num_bags": 12,
        }
    ),
    # Feed System Components
    ComponentInfo(
        name="Feed Hopper",
        category="Feed System",
        description="Storage hopper for feed material",
        default_params={
            "diameter": 0.6,
            "cylinder_height": 0.8,
            "cone_angle": 60.0,
        }
    ),
    ComponentInfo(
        name="Rotary Airlock",
        category="Feed System",
        description="Rotary valve for controlled material feeding",
        default_params={
            "rotor_diameter": 0.15,
            "rotor_width": 0.15,
            "num_vanes": 8,
        }
    ),
    ComponentInfo(
        name="Screw Feeder",
        category="Feed System",
        description="Screw conveyor for controlled material transport",
        default_params={
            "screw_diameter": 0.08,
            "pitch": 0.06,
            "length": 0.5,
        }
    ),
    ComponentInfo(
        name="Deagglomerator",
        category="Feed System",
        description="Pin mill or impact mill for breaking up agglomerates",
        default_params={
            "rotor_diameter": 0.15,
            "rpm": 3000,
            "num_pins": 24,
        }
    ),
    # Air System Components
    ComponentInfo(
        name="Centrifugal Blower",
        category="Air System",
        description="Main air mover for the classification system",
        default_params={
            "flow_rate": 0.15,
            "pressure_rise": 3000.0,
            "impeller_diameter": 0.25,
        }
    ),
    ComponentInfo(
        name="Air Filter",
        category="Air System",
        description="Inlet air filter for removing contaminants",
        default_params={
            "flow_rate": 0.15,
            "filter_class": "HEPA",
        }
    ),
    ComponentInfo(
        name="Damper",
        category="Air System",
        description="Flow control damper",
        default_params={
            "diameter": 0.08,
            "max_opening": 90.0,
        }
    ),
    # Ductwork Components
    ComponentInfo(
        name="Round Duct",
        category="Ductwork",
        description="Straight round duct section",
        default_params={
            "diameter": 0.08,
            "length": 0.5,
        }
    ),
    ComponentInfo(
        name="Duct Elbow",
        category="Ductwork",
        description="90-degree duct elbow",
        default_params={
            "diameter": 0.08,
            "bend_radius": 0.12,
            "angle": 90.0,
        }
    ),
    ComponentInfo(
        name="Rect-to-Round Transition",
        category="Ductwork",
        description="Transition from rectangular to round cross-section",
        default_params={
            "inlet_width": 0.06,
            "inlet_height": 0.08,
            "outlet_diameter": 0.08,
            "length": 0.15,
        }
    ),
    ComponentInfo(
        name="Tee Junction",
        category="Ductwork",
        description="Three-way pipe junction",
        default_params={
            "main_diameter": 0.08,
            "branch_diameter": 0.06,
            "angle": 90.0,
        }
    ),
    # Exhaust Components
    ComponentInfo(
        name="Silencer",
        category="Exhaust",
        description="Acoustic silencer for noise reduction",
        default_params={
            "diameter": 0.20,
            "length": 0.6,
        }
    ),
    ComponentInfo(
        name="Exhaust Stack",
        category="Exhaust",
        description="Vertical exhaust stack for clean air discharge",
        default_params={
            "diameter": 0.15,
            "height": 4.0,
        }
    ),
]


class ComponentPalette(QWidget):
    """
    Tree view panel showing available components organized by category.

    Users can drag components from this palette onto the assembly canvas
    to add them to their classifier design.
    """

    # Signal emitted when a component is selected (double-clicked or dragged)
    component_selected = Signal(str, dict)  # (component_name, default_params)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._populate_tree()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter components...")
        self.search_edit.textChanged.connect(self._filter_tree)
        self.search_edit.setClearButtonEnabled(True)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        # Component tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.setToolTipDuration(5000)
        layout.addWidget(self.tree)

        # Quick add button
        self.add_button = QPushButton("Add Selected")
        self.add_button.clicked.connect(self._add_selected)
        layout.addWidget(self.add_button)

    def _populate_tree(self):
        """Populate the tree with components from the library."""
        self.tree.clear()

        # Group components by category
        categories: Dict[str, List[ComponentInfo]] = {}
        for comp in COMPONENT_LIBRARY:
            if comp.category not in categories:
                categories[comp.category] = []
            categories[comp.category].append(comp)

        # Define category order and colors
        category_order = [
            "Classification",
            "Cyclones",
            "Filtration",
            "Feed System",
            "Air System",
            "Ductwork",
            "Exhaust",
        ]
        category_colors = {
            "Classification": QColor(100, 149, 237),  # Cornflower blue
            "Cyclones": QColor(144, 238, 144),        # Light green
            "Filtration": QColor(255, 182, 193),      # Light pink
            "Feed System": QColor(255, 218, 185),     # Peach
            "Air System": QColor(173, 216, 230),      # Light blue
            "Ductwork": QColor(192, 192, 192),        # Silver
            "Exhaust": QColor(221, 160, 221),         # Plum
        }

        # Create tree items
        for category in category_order:
            if category not in categories:
                continue

            # Category item
            cat_item = QTreeWidgetItem([category])
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)

            color = category_colors.get(category, QColor(200, 200, 200))
            cat_item.setBackground(0, color.darker(200))

            self.tree.addTopLevelItem(cat_item)

            # Component items
            for comp in categories[category]:
                comp_item = QTreeWidgetItem([comp.name])
                comp_item.setData(0, Qt.ItemDataRole.UserRole, comp)
                comp_item.setToolTip(0, comp.description)
                cat_item.addChild(comp_item)

            cat_item.setExpanded(True)

    def _filter_tree(self, text: str):
        """Filter tree items based on search text."""
        text = text.lower()

        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_visible = False

            for j in range(cat_item.childCount()):
                comp_item = cat_item.child(j)
                comp: ComponentInfo = comp_item.data(0, Qt.ItemDataRole.UserRole)

                # Match against name, category, or description
                visible = (
                    text in comp.name.lower() or
                    text in comp.category.lower() or
                    text in comp.description.lower()
                )
                comp_item.setHidden(not visible)
                if visible:
                    cat_visible = True

            cat_item.setHidden(not cat_visible)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on tree item."""
        comp = item.data(0, Qt.ItemDataRole.UserRole)
        if comp:
            self.component_selected.emit(comp.name, comp.default_params.copy())

    def _add_selected(self):
        """Add the currently selected component."""
        items = self.tree.selectedItems()
        if items:
            comp = items[0].data(0, Qt.ItemDataRole.UserRole)
            if comp:
                self.component_selected.emit(comp.name, comp.default_params.copy())

    def show_add_dialog(self):
        """Show dialog for adding a component with customization."""
        dialog = ComponentAddDialog(self)
        if dialog.exec():
            name, params = dialog.get_result()
            self.component_selected.emit(name, params)

    def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """Get component info by name."""
        for comp in COMPONENT_LIBRARY:
            if comp.name == name:
                return comp
        return None

    def startDrag(self, supportedActions):
        """Start drag operation for component."""
        items = self.tree.selectedItems()
        if not items:
            return

        item = items[0]
        comp = item.data(0, Qt.ItemDataRole.UserRole)
        if not comp:
            return

        # Create MIME data
        mime_data = QMimeData()
        import json
        data = {
            "name": comp.name,
            "category": comp.category,
            "params": comp.default_params,
        }
        mime_data.setData("application/x-airclassifier-component", json.dumps(data).encode())

        # Create drag
        drag = QDrag(self.tree)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)


class ComponentAddDialog(QDialog):
    """Dialog for adding a component with parameter customization."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Component")
        self.setMinimumWidth(400)

        self._selected_comp: Optional[ComponentInfo] = None
        self._param_widgets: Dict[str, QWidget] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Component selection
        select_group = QGroupBox("Select Component")
        select_layout = QFormLayout(select_group)

        self.category_combo = QComboBox()
        categories = sorted(set(c.category for c in COMPONENT_LIBRARY))
        self.category_combo.addItems(categories)
        self.category_combo.currentTextChanged.connect(self._update_component_list)
        select_layout.addRow("Category:", self.category_combo)

        self.component_combo = QComboBox()
        self.component_combo.currentTextChanged.connect(self._update_params)
        select_layout.addRow("Component:", self.component_combo)

        layout.addWidget(select_group)

        # Parameters group
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout(self.params_group)
        layout.addWidget(self.params_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Initialize
        self._update_component_list(self.category_combo.currentText())

    def _update_component_list(self, category: str):
        """Update component list based on selected category."""
        self.component_combo.clear()
        for comp in COMPONENT_LIBRARY:
            if comp.category == category:
                self.component_combo.addItem(comp.name)

    def _update_params(self, component_name: str):
        """Update parameter widgets based on selected component."""
        # Clear existing widgets
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        self._param_widgets.clear()

        # Find component
        self._selected_comp = None
        for comp in COMPONENT_LIBRARY:
            if comp.name == component_name:
                self._selected_comp = comp
                break

        if not self._selected_comp:
            return

        # Create parameter widgets
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        for param_name, default_value in self._selected_comp.default_params.items():
            label = param_name.replace("_", " ").title()

            if isinstance(default_value, float):
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(-1e9, 1e9)
                widget.setValue(default_value)
            elif isinstance(default_value, int):
                widget = QSpinBox()
                widget.setRange(-1000000, 1000000)
                widget.setValue(default_value)
            else:
                widget = QLineEdit(str(default_value))

            self._param_widgets[param_name] = widget
            self.params_layout.addRow(f"{label}:", widget)

    def get_result(self) -> tuple:
        """Get the selected component and parameters."""
        if not self._selected_comp:
            return "", {}

        params = {}
        for param_name, widget in self._param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                params[param_name] = widget.value()
            elif isinstance(widget, QLineEdit):
                params[param_name] = widget.text()

        return self._selected_comp.name, params
