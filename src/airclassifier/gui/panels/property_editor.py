"""
Property Editor Panel
=====================

Dynamic property editor for configuring component parameters.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame,
    QFormLayout, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QGroupBox, QPushButton,
    QHBoxLayout, QSlider, QColorDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont

from ..theme import COLORS


@dataclass
class PropertyDefinition:
    """Definition of an editable property."""
    name: str
    display_name: str
    type: str  # "float", "int", "bool", "string", "choice", "color"
    default: Any
    min_value: Any = None
    max_value: Any = None
    step: Any = None
    choices: List[str] = None
    unit: str = ""
    tooltip: str = ""
    group: str = "General"

    def __post_init__(self):
        if self.choices is None:
            self.choices = []


# Property definitions for each component type
COMPONENT_PROPERTIES: Dict[str, List[PropertyDefinition]] = {
    "Venturi Eductor": [
        PropertyDefinition("inlet_diameter", "Inlet Diameter", "float", 0.08, 0.02, 0.3, 0.005, unit="m", group="Dimensions", tooltip="Air inlet diameter"),
        PropertyDefinition("throat_ratio", "Throat Ratio", "float", 0.5, 0.3, 0.8, 0.05, group="Dimensions", tooltip="Throat/inlet diameter ratio"),
        PropertyDefinition("divergent_angle", "Divergent Angle", "float", 7.0, 3.0, 15.0, 0.5, unit="deg", group="Dimensions"),
        PropertyDefinition("solids_inlet_diameter", "Solids Inlet Dia", "float", 0.04, 0.02, 0.1, 0.005, unit="m", group="Dimensions"),
        PropertyDefinition("material", "Material", "choice", "Stainless Steel", choices=["Stainless Steel", "Carbon Steel", "Aluminum", "Plastic"], group="Construction"),
    ],
    "Zigzag Classifier": [
        PropertyDefinition("channel_width", "Channel Width", "float", 0.06, 0.03, 0.2, 0.005, unit="m", group="Dimensions"),
        PropertyDefinition("channel_depth", "Channel Depth", "float", 0.08, 0.04, 0.3, 0.005, unit="m", group="Dimensions"),
        PropertyDefinition("num_stages", "Number of Stages", "int", 5, 3, 12, 1, group="Dimensions"),
        PropertyDefinition("plate_angle", "Plate Angle", "float", 60.0, 30.0, 75.0, 1.0, unit="deg", group="Geometry"),
        PropertyDefinition("plate_length_ratio", "Plate Length Ratio", "float", 0.6, 0.4, 0.9, 0.05, group="Geometry"),
        PropertyDefinition("turbulence_intensity", "Turbulence Intensity", "float", 0.15, 0.05, 0.3, 0.01, group="Flow"),
    ],
    "Wheel Classifier": [
        PropertyDefinition("wheel_diameter", "Wheel Diameter", "float", 0.20, 0.1, 0.5, 0.01, unit="m", group="Wheel"),
        PropertyDefinition("wheel_rpm", "Wheel Speed", "float", 8000.0, 1000.0, 15000.0, 100.0, unit="RPM", group="Operation"),
        PropertyDefinition("num_blades", "Number of Blades", "int", 24, 8, 48, 2, group="Wheel"),
        PropertyDefinition("blade_thickness", "Blade Thickness", "float", 0.002, 0.001, 0.005, 0.0005, unit="m", group="Wheel"),
        PropertyDefinition("hub_diameter", "Hub Diameter", "float", 0.08, 0.03, 0.15, 0.005, unit="m", group="Wheel"),
        PropertyDefinition("target_d50", "Target d50", "float", 25e-6, 5e-6, 100e-6, 1e-6, unit="m", group="Separation", tooltip="Target cut size"),
        PropertyDefinition("volute_clearance", "Volute Clearance", "float", 0.01, 0.005, 0.03, 0.001, unit="m", group="Housing"),
    ],
    "Multi-Cyclone System": [
        PropertyDefinition("num_stages", "Number of Stages", "int", 3, 1, 5, 1, group="Configuration"),
        PropertyDefinition("primary_diameter", "Primary Diameter", "float", 0.30, 0.15, 0.5, 0.01, unit="m", group="Primary"),
        PropertyDefinition("secondary_diameter", "Secondary Diameter", "float", 0.20, 0.1, 0.4, 0.01, unit="m", group="Secondary"),
        PropertyDefinition("tertiary_diameter", "Tertiary Diameter", "float", 0.12, 0.08, 0.25, 0.01, unit="m", group="Tertiary"),
        PropertyDefinition("aspect_ratio", "Height/Diameter", "float", 4.0, 3.0, 6.0, 0.5, group="Geometry"),
    ],
    "Bag Filter": [
        PropertyDefinition("flow_rate", "Flow Rate", "float", 0.15, 0.05, 1.0, 0.01, unit="m3/s", group="Flow"),
        PropertyDefinition("air_to_cloth", "Air-to-Cloth Ratio", "float", 2.5, 1.0, 5.0, 0.1, unit="m3/min/m2", group="Flow"),
        PropertyDefinition("num_bags", "Number of Bags", "int", 12, 4, 48, 2, group="Configuration"),
        PropertyDefinition("bag_diameter", "Bag Diameter", "float", 0.15, 0.1, 0.3, 0.01, unit="m", group="Bags"),
        PropertyDefinition("bag_length", "Bag Length", "float", 2.0, 1.0, 4.0, 0.1, unit="m", group="Bags"),
        PropertyDefinition("cleaning_method", "Cleaning Method", "choice", "Pulse Jet", choices=["Pulse Jet", "Reverse Air", "Shaker"], group="Operation"),
    ],
    "Centrifugal Blower": [
        PropertyDefinition("flow_rate", "Flow Rate", "float", 0.15, 0.05, 1.0, 0.01, unit="m3/s", group="Performance"),
        PropertyDefinition("pressure_rise", "Pressure Rise", "float", 3000.0, 1000.0, 10000.0, 100.0, unit="Pa", group="Performance"),
        PropertyDefinition("impeller_diameter", "Impeller Diameter", "float", 0.25, 0.15, 0.5, 0.01, unit="m", group="Dimensions"),
        PropertyDefinition("rpm", "Speed", "float", 3000.0, 1000.0, 6000.0, 100.0, unit="RPM", group="Operation"),
        PropertyDefinition("efficiency", "Efficiency", "float", 0.75, 0.5, 0.9, 0.01, group="Performance"),
    ],
    "Feed Hopper": [
        PropertyDefinition("diameter", "Diameter", "float", 0.6, 0.3, 1.5, 0.05, unit="m", group="Dimensions"),
        PropertyDefinition("cylinder_height", "Cylinder Height", "float", 0.8, 0.3, 2.0, 0.1, unit="m", group="Dimensions"),
        PropertyDefinition("cone_angle", "Cone Angle", "float", 60.0, 30.0, 75.0, 5.0, unit="deg", group="Geometry"),
        PropertyDefinition("outlet_diameter", "Outlet Diameter", "float", 0.1, 0.05, 0.3, 0.01, unit="m", group="Dimensions"),
    ],
    "Round Duct": [
        PropertyDefinition("diameter", "Diameter", "float", 0.08, 0.02, 0.3, 0.005, unit="m", group="Dimensions"),
        PropertyDefinition("length", "Length", "float", 0.5, 0.1, 5.0, 0.1, unit="m", group="Dimensions"),
        PropertyDefinition("wall_thickness", "Wall Thickness", "float", 0.002, 0.001, 0.005, 0.0005, unit="m", group="Dimensions"),
    ],
    "Duct Elbow": [
        PropertyDefinition("diameter", "Diameter", "float", 0.08, 0.02, 0.3, 0.005, unit="m", group="Dimensions"),
        PropertyDefinition("bend_radius", "Bend Radius", "float", 0.12, 0.05, 0.5, 0.01, unit="m", group="Dimensions"),
        PropertyDefinition("angle", "Bend Angle", "float", 90.0, 15.0, 180.0, 15.0, unit="deg", group="Geometry"),
    ],
}

# Category -> accent colour  (matches palette)
_CATEGORY_FOR_TYPE = {
    "Venturi Eductor": "Classification",
    "Zigzag Classifier": "Classification",
    "Wheel Classifier": "Classification",
    "Cyclone (Primary)": "Cyclones",
    "Cyclone (Secondary)": "Cyclones",
    "Cyclone (Tertiary)": "Cyclones",
    "Multi-Cyclone System": "Cyclones",
    "Bag Filter": "Filtration",
    "Feed Hopper": "Feed System",
    "Rotary Airlock": "Feed System",
    "Screw Feeder": "Feed System",
    "Deagglomerator": "Feed System",
    "Centrifugal Blower": "Air System",
    "Air Filter": "Air System",
    "Damper": "Air System",
    "Round Duct": "Ductwork",
    "Duct Elbow": "Ductwork",
    "Rect-to-Round Transition": "Ductwork",
    "Tee Junction": "Ductwork",
    "Silencer": "Exhaust",
    "Exhaust Stack": "Exhaust",
}

_CATEGORY_COLORS_HEX = {
    "Classification": COLORS.CAT_CLASSIFICATION,
    "Cyclones":       COLORS.CAT_CYCLONES,
    "Filtration":     COLORS.CAT_FILTRATION,
    "Feed System":    COLORS.CAT_FEED,
    "Air System":     COLORS.CAT_AIR,
    "Ductwork":       COLORS.CAT_DUCTWORK,
    "Exhaust":        COLORS.CAT_EXHAUST,
}


class PropertyEditor(QWidget):
    """
    Dynamic property editor panel.

    Displays and allows editing of component parameters based on
    the currently selected component in the assembly.
    """

    # Signal emitted when a parameter value changes
    parameter_changed = Signal(str, str, object)  # (component_id, param_name, new_value)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_component_id: Optional[str] = None
        self._current_component_type: Optional[str] = None
        self._widgets: Dict[str, QWidget] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        # Header card (colored bar + title)
        self._header_frame = QFrame()
        self._header_frame.setFixedHeight(52)
        self._header_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_SURFACE};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        header_layout = QHBoxLayout(self._header_frame)
        header_layout.setContentsMargins(0, 0, 12, 0)
        header_layout.setSpacing(10)

        # Accent bar
        self._accent_bar = QFrame()
        self._accent_bar.setFixedWidth(5)
        self._accent_bar.setStyleSheet(f"background: {COLORS.TEXT_MUTED}; border-radius: 3px; border: none;")
        header_layout.addWidget(self._accent_bar)

        # Text
        header_text_layout = QVBoxLayout()
        header_text_layout.setContentsMargins(0, 6, 0, 6)
        header_text_layout.setSpacing(1)

        self.header_label = QLabel("No Selection")
        self.header_label.setStyleSheet(f"font-weight: 600; font-size: 11pt; color: {COLORS.TEXT_PRIMARY}; border: none; background: transparent;")
        header_text_layout.addWidget(self.header_label)

        self._category_label = QLabel("")
        self._category_label.setStyleSheet(f"font-size: 8pt; color: {COLORS.TEXT_MUTED}; border: none; background: transparent;")
        header_text_layout.addWidget(self._category_label)

        header_layout.addLayout(header_text_layout, 1)
        layout.addWidget(self._header_frame)

        layout.addSpacing(6)

        # Scroll area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.props_widget = QWidget()
        self.props_layout = QVBoxLayout(self.props_widget)
        self.props_layout.setContentsMargins(0, 0, 0, 0)
        self.props_layout.setSpacing(6)
        self.props_layout.addStretch()

        scroll.setWidget(self.props_widget)
        layout.addWidget(scroll, 1)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setProperty("cssClass", "ghost")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_btn)

    # ------------------------------------------------------------------

    def clear(self):
        """Clear the property editor."""
        self._current_component_id = None
        self._current_component_type = None
        self._widgets.clear()
        self.header_label.setText("No Selection")
        self._category_label.setText("")
        self._accent_bar.setStyleSheet(f"background: {COLORS.TEXT_MUTED}; border-radius: 3px; border: none;")

        # Remove all widgets except the stretch
        while self.props_layout.count() > 1:
            item = self.props_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @Slot(str, str, dict)
    def set_component(self, component_id: str, component_type: str, params: Dict[str, Any]):
        """
        Set the component to edit.

        Args:
            component_id: Unique identifier for the component
            component_type: Type name of the component
            params: Current parameter values
        """
        self.clear()

        self._current_component_id = component_id
        self._current_component_type = component_type

        # Header
        self.header_label.setText(component_type)
        category = _CATEGORY_FOR_TYPE.get(component_type, "")
        self._category_label.setText(category)
        accent_color = _CATEGORY_COLORS_HEX.get(category, COLORS.TEXT_MUTED)
        self._accent_bar.setStyleSheet(f"background: {accent_color}; border-radius: 3px; border: none;")

        # Get property definitions
        prop_defs = COMPONENT_PROPERTIES.get(component_type, [])
        if not prop_defs:
            prop_defs = self._create_generic_properties(params)

        # Group properties
        groups: Dict[str, List[PropertyDefinition]] = {}
        for prop in prop_defs:
            if prop.group not in groups:
                groups[prop.group] = []
            groups[prop.group].append(prop)

        # Create widgets for each group
        for group_name, props in groups.items():
            group_box = QGroupBox(group_name)
            group_layout = QFormLayout(group_box)
            group_layout.setSpacing(6)
            group_layout.setContentsMargins(10, 14, 10, 10)

            for prop in props:
                widget = self._create_widget(prop, params.get(prop.name, prop.default))
                if widget:
                    # Build label with unit in muted colour
                    if prop.unit:
                        label_widget = QLabel(f"{prop.display_name} <span style='color:{COLORS.TEXT_MUTED}; font-size:8pt;'>[{prop.unit}]</span>")
                        label_widget.setTextFormat(Qt.TextFormat.RichText)
                    else:
                        label_widget = QLabel(f"{prop.display_name}")

                    group_layout.addRow(label_widget, widget)
                    self._widgets[prop.name] = widget

            # Insert before the stretch
            self.props_layout.insertWidget(self.props_layout.count() - 1, group_box)

    def _create_generic_properties(self, params: Dict[str, Any]) -> List[PropertyDefinition]:
        """Create generic property definitions from parameter dict."""
        props = []
        for name, value in params.items():
            display_name = name.replace("_", " ").title()
            if isinstance(value, float):
                props.append(PropertyDefinition(name, display_name, "float", value))
            elif isinstance(value, int):
                props.append(PropertyDefinition(name, display_name, "int", value))
            elif isinstance(value, bool):
                props.append(PropertyDefinition(name, display_name, "bool", value))
            else:
                props.append(PropertyDefinition(name, display_name, "string", str(value)))
        return props

    def _create_widget(self, prop: PropertyDefinition, value: Any) -> Optional[QWidget]:
        """Create an appropriate widget for a property."""
        widget = None

        if prop.type == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            if prop.min_value is not None:
                widget.setMinimum(prop.min_value)
            else:
                widget.setMinimum(-1e9)
            if prop.max_value is not None:
                widget.setMaximum(prop.max_value)
            else:
                widget.setMaximum(1e9)
            if prop.step is not None:
                widget.setSingleStep(prop.step)
            widget.setValue(float(value) if value is not None else prop.default)
            widget.valueChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "int":
            widget = QSpinBox()
            if prop.min_value is not None:
                widget.setMinimum(int(prop.min_value))
            else:
                widget.setMinimum(-1000000)
            if prop.max_value is not None:
                widget.setMaximum(int(prop.max_value))
            else:
                widget.setMaximum(1000000)
            if prop.step is not None:
                widget.setSingleStep(int(prop.step))
            widget.setValue(int(value) if value is not None else prop.default)
            widget.valueChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value) if value is not None else prop.default)
            widget.stateChanged.connect(lambda s, n=prop.name: self._on_value_changed(n, s == Qt.CheckState.Checked.value))

        elif prop.type == "choice":
            widget = QComboBox()
            widget.addItems(prop.choices)
            if value in prop.choices:
                widget.setCurrentText(value)
            elif prop.default in prop.choices:
                widget.setCurrentText(prop.default)
            widget.currentTextChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        elif prop.type == "color":
            widget = QPushButton()
            color = QColor(value) if value else QColor(prop.default)
            widget.setStyleSheet(f"background-color: {color.name()}; border-radius: 4px; min-height: 24px;")
            widget.clicked.connect(lambda checked, n=prop.name, w=widget: self._pick_color(n, w))

        else:  # string
            widget = QLineEdit()
            widget.setText(str(value) if value is not None else str(prop.default))
            widget.textChanged.connect(lambda v, n=prop.name: self._on_value_changed(n, v))

        if widget and prop.tooltip:
            widget.setToolTip(prop.tooltip)

        return widget

    def _on_value_changed(self, param_name: str, value: Any):
        """Handle parameter value change."""
        if self._current_component_id:
            self.parameter_changed.emit(self._current_component_id, param_name, value)

    def _pick_color(self, param_name: str, button: QPushButton):
        """Show color picker dialog."""
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}; border-radius: 4px; min-height: 24px;")
            self._on_value_changed(param_name, color.name())

    def _reset_to_defaults(self):
        """Reset all properties to default values."""
        if not self._current_component_type:
            return

        prop_defs = COMPONENT_PROPERTIES.get(self._current_component_type, [])
        for prop in prop_defs:
            widget = self._widgets.get(prop.name)
            if widget:
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(prop.default)
                elif isinstance(widget, QSpinBox):
                    widget.setValue(prop.default)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(prop.default)
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(prop.default)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(prop.default))

    def get_current_values(self) -> Dict[str, Any]:
        """Get current values of all properties."""
        values = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text()
        return values
