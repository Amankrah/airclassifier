"""
Assembly Canvas Widget
======================

Node-based visual editor for assembling air classifier components.
Uses Qt Graphics Framework for interactive component placement and connection.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import json
import uuid

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem, QGraphicsPathItem, QGraphicsDropShadowEffect,
    QMenu, QInputDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF, QLineF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QLinearGradient, QTransform, QUndoStack, QUndoCommand,
)


# Color scheme for component categories
CATEGORY_COLORS = {
    "Classification": QColor(100, 149, 237),  # Cornflower blue
    "Cyclones": QColor(144, 238, 144),        # Light green
    "Filtration": QColor(255, 182, 193),      # Light pink
    "Feed System": QColor(255, 218, 185),     # Peach
    "Air System": QColor(173, 216, 230),      # Light blue
    "Ductwork": QColor(192, 192, 192),        # Silver
    "Exhaust": QColor(221, 160, 221),         # Plum
}


@dataclass
class PortData:
    """Data for a connection port."""
    name: str
    port_type: str  # "inlet" or "outlet"
    position: Tuple[float, float]  # Relative to node
    connected_to: Optional[str] = None  # Node ID of connected port


@dataclass
class NodeData:
    """Data for a component node."""
    id: str
    component_type: str
    category: str
    params: Dict[str, Any]
    position: Tuple[float, float]
    ports: List[PortData] = field(default_factory=list)


class PortItem(QGraphicsEllipseItem):
    """Visual representation of a connection port."""

    def __init__(self, port_data: PortData, parent_node: 'NodeItem'):
        size = 12
        super().__init__(-size/2, -size/2, size, size, parent_node)

        self.port_data = port_data
        self.parent_node = parent_node
        self._connection: Optional['ConnectionItem'] = None

        # Style based on port type
        if port_data.port_type == "inlet":
            self.setBrush(QBrush(QColor(100, 200, 100)))
        else:
            self.setBrush(QBrush(QColor(200, 100, 100)))

        self.setPen(QPen(QColor(60, 60, 60), 2))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(10)

        # Position relative to parent
        self.setPos(QPointF(*port_data.position))

        # Tooltip
        self.setToolTip(f"{port_data.name} ({port_data.port_type})")

    def hoverEnterEvent(self, event):
        """Highlight on hover."""
        self.setPen(QPen(QColor(255, 255, 100), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Remove highlight."""
        self.setPen(QPen(QColor(60, 60, 60), 2))
        super().hoverLeaveEvent(event)

    def center_scene_pos(self) -> QPointF:
        """Get center position in scene coordinates."""
        return self.scenePos()


class NodeItem(QGraphicsRectItem):
    """Visual representation of a component node."""

    def __init__(self, node_data: NodeData):
        super().__init__()

        self.node_data = node_data
        self.port_items: Dict[str, PortItem] = {}

        # Node dimensions
        self.node_width = 180
        self.node_height = 100

        self.setRect(0, 0, self.node_width, self.node_height)
        self.setPos(QPointF(*node_data.position))

        # Make movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Style
        self._setup_style()

        # Add title
        self._add_title()

        # Add ports
        self._add_ports()

        # Drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setOffset(3, 3)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

    def _setup_style(self):
        """Setup node visual style."""
        # Get category color
        color = CATEGORY_COLORS.get(self.node_data.category, QColor(100, 100, 100))

        # Gradient fill
        gradient = QLinearGradient(0, 0, 0, self.node_height)
        gradient.setColorAt(0, color.lighter(120))
        gradient.setColorAt(1, color.darker(110))
        self.setBrush(QBrush(gradient))

        # Border
        self.setPen(QPen(color.darker(150), 2))

    def _add_title(self):
        """Add title text to node."""
        # Header background
        header = QGraphicsRectItem(0, 0, self.node_width, 25, self)
        color = CATEGORY_COLORS.get(self.node_data.category, QColor(100, 100, 100))
        header.setBrush(QBrush(color.darker(130)))
        header.setPen(QPen(Qt.PenStyle.NoPen))

        # Title text
        title = QGraphicsTextItem(self.node_data.component_type, self)
        title.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        title.setFont(font)

        # Center title
        title_width = title.boundingRect().width()
        title.setPos((self.node_width - title_width) / 2, 3)

        # Node ID (small text)
        id_text = QGraphicsTextItem(self.node_data.id[:8], self)
        id_text.setDefaultTextColor(QColor(150, 150, 150))
        id_text.setFont(QFont("Segoe UI", 7))
        id_text.setPos(5, self.node_height - 18)

    def _add_ports(self):
        """Add port items to node."""
        # Default ports based on component type
        default_ports = self._get_default_ports()

        # Position ports
        inlet_y = 50
        outlet_y = 50

        for port_data in default_ports:
            if port_data.port_type == "inlet":
                port_data.position = (-6, inlet_y)
                inlet_y += 25
            else:
                port_data.position = (self.node_width + 6, outlet_y)
                outlet_y += 25

            port_item = PortItem(port_data, self)
            self.port_items[port_data.name] = port_item
            self.node_data.ports.append(port_data)

    def _get_default_ports(self) -> List[PortData]:
        """Get default ports for this component type."""
        component_ports = {
            "Venturi Eductor": [
                PortData("air_inlet", "inlet", (0, 0)),
                PortData("solids_inlet", "inlet", (0, 0)),
                PortData("outlet", "outlet", (0, 0)),
            ],
            "Zigzag Classifier": [
                PortData("air_inlet", "inlet", (0, 0)),
                PortData("fines_outlet", "outlet", (0, 0)),
                PortData("coarse_outlet", "outlet", (0, 0)),
            ],
            "Wheel Classifier": [
                PortData("inlet", "inlet", (0, 0)),
                PortData("fines_outlet", "outlet", (0, 0)),
                PortData("coarse_outlet", "outlet", (0, 0)),
            ],
            "Multi-Cyclone System": [
                PortData("inlet", "inlet", (0, 0)),
                PortData("overflow", "outlet", (0, 0)),
                PortData("dust_outlet", "outlet", (0, 0)),
            ],
            "Bag Filter": [
                PortData("dirty_air_inlet", "inlet", (0, 0)),
                PortData("clean_air_outlet", "outlet", (0, 0)),
                PortData("dust_outlet", "outlet", (0, 0)),
            ],
            "Centrifugal Blower": [
                PortData("inlet", "inlet", (0, 0)),
                PortData("outlet", "outlet", (0, 0)),
            ],
            "Feed Hopper": [
                PortData("outlet", "outlet", (0, 0)),
            ],
            "Round Duct": [
                PortData("inlet", "inlet", (0, 0)),
                PortData("outlet", "outlet", (0, 0)),
            ],
            "Duct Elbow": [
                PortData("inlet", "inlet", (0, 0)),
                PortData("outlet", "outlet", (0, 0)),
            ],
        }

        return component_ports.get(self.node_data.component_type, [
            PortData("inlet", "inlet", (0, 0)),
            PortData("outlet", "outlet", (0, 0)),
        ])

    def itemChange(self, change, value):
        """Handle item changes (e.g., position)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update node data
            self.node_data.position = (value.x(), value.y())

            # Update connections
            for port_item in self.port_items.values():
                if port_item._connection:
                    port_item._connection.update_path()

        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        """Highlight on hover."""
        self.setPen(QPen(QColor(255, 200, 50), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Remove highlight."""
        color = CATEGORY_COLORS.get(self.node_data.category, QColor(100, 100, 100))
        self.setPen(QPen(color.darker(150), 2))
        super().hoverLeaveEvent(event)


class ConnectionItem(QGraphicsPathItem):
    """Visual representation of a connection between ports."""

    def __init__(self, start_port: PortItem, end_port: PortItem):
        super().__init__()

        self.start_port = start_port
        self.end_port = end_port

        # Register connection with ports
        start_port._connection = self
        end_port._connection = self
        start_port.port_data.connected_to = end_port.parent_node.node_data.id
        end_port.port_data.connected_to = start_port.parent_node.node_data.id

        # Style
        self.setPen(QPen(QColor(200, 200, 200), 3, Qt.PenStyle.SolidLine))
        self.setZValue(-1)

        # Initial path
        self.update_path()

    def update_path(self):
        """Update the path based on port positions."""
        start = self.start_port.center_scene_pos()
        end = self.end_port.center_scene_pos()

        # Create smooth bezier curve
        path = QPainterPath(start)

        # Control points for smooth curve
        dx = abs(end.x() - start.x())
        ctrl_dist = max(50, dx / 2)

        if start.x() < end.x():
            ctrl1 = QPointF(start.x() + ctrl_dist, start.y())
            ctrl2 = QPointF(end.x() - ctrl_dist, end.y())
        else:
            ctrl1 = QPointF(start.x() - ctrl_dist, start.y())
            ctrl2 = QPointF(end.x() + ctrl_dist, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)

    def disconnect(self):
        """Disconnect this connection."""
        self.start_port._connection = None
        self.end_port._connection = None
        self.start_port.port_data.connected_to = None
        self.end_port.port_data.connected_to = None


class AssemblyCanvas(QGraphicsView):
    """
    Node-based visual editor for assembling air classifier components.

    Features:
    - Drag and drop component nodes
    - Connect ports with bezier curves
    - Pan and zoom navigation
    - Undo/redo support
    - Export assembly configuration
    """

    # Signals
    node_selected = Signal(str, str, dict)  # (node_id, component_type, params)
    assembly_changed = Signal(dict)          # assembly_data

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data storage
        self._nodes: Dict[str, NodeItem] = {}
        self._connections: List[ConnectionItem] = []
        self._pending_connection: Optional[PortItem] = None
        self._temp_line: Optional[QGraphicsLineItem] = None

        # Undo stack
        self._undo_stack = QUndoStack(self)

        # Setup
        self._setup_scene()
        self._setup_view()

    def _setup_scene(self):
        """Setup the graphics scene."""
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(-5000, -5000, 10000, 10000)
        self._scene.setBackgroundBrush(QBrush(QColor(30, 30, 32)))
        self.setScene(self._scene)

        # Add grid
        self._draw_grid()

    def _setup_view(self):
        """Setup view properties."""
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        # Accept drops
        self.setAcceptDrops(True)

    def _draw_grid(self):
        """Draw background grid."""
        grid_size = 25
        pen = QPen(QColor(45, 45, 48), 1)

        for x in range(-5000, 5000, grid_size):
            line = self._scene.addLine(x, -5000, x, 5000, pen)
            line.setZValue(-100)

        for y in range(-5000, 5000, grid_size):
            line = self._scene.addLine(-5000, y, 5000, y, pen)
            line.setZValue(-100)

    @Slot(str, dict)
    def add_component_node(self, component_type: str, params: Dict[str, Any]):
        """
        Add a new component node to the canvas.

        Args:
            component_type: Type of component
            params: Component parameters
        """
        # Determine category from component type
        category = self._get_category(component_type)

        # Create node data
        node_id = str(uuid.uuid4())
        position = self._get_spawn_position()

        node_data = NodeData(
            id=node_id,
            component_type=component_type,
            category=category,
            params=params.copy(),
            position=position,
        )

        # Create visual node
        node_item = NodeItem(node_data)
        self._scene.addItem(node_item)
        self._nodes[node_id] = node_item

        # Connect selection signal
        node_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        self.assembly_changed.emit(self.get_assembly_data())

    def _get_category(self, component_type: str) -> str:
        """Get category for component type."""
        categories = {
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
        return categories.get(component_type, "Ductwork")

    def _get_spawn_position(self) -> Tuple[float, float]:
        """Get position for new node (center of view + offset)."""
        center = self.mapToScene(self.viewport().rect().center())
        offset = len(self._nodes) * 20
        return (center.x() + offset, center.y() + offset)

    def delete_selected(self):
        """Delete selected nodes and connections."""
        for item in self._scene.selectedItems():
            if isinstance(item, NodeItem):
                # Remove connections
                for port_item in item.port_items.values():
                    if port_item._connection:
                        conn = port_item._connection
                        conn.disconnect()
                        self._scene.removeItem(conn)
                        if conn in self._connections:
                            self._connections.remove(conn)

                # Remove node
                del self._nodes[item.node_data.id]
                self._scene.removeItem(item)

        self.assembly_changed.emit(self.get_assembly_data())

    @Slot(str, str, object)
    def update_node_params(self, node_id: str, param_name: str, value: Any):
        """Update a node's parameter."""
        if node_id in self._nodes:
            self._nodes[node_id].node_data.params[param_name] = value
            self.assembly_changed.emit(self.get_assembly_data())

    def auto_connect(self) -> int:
        """
        Automatically connect compatible ports.

        Returns:
            Number of connections made
        """
        connections_made = 0
        # Simple auto-connect: connect outlets to inlets of nearby nodes
        # (More sophisticated matching could be added)

        for node_id, node in self._nodes.items():
            for port_name, port_item in node.port_items.items():
                if port_item.port_data.port_type != "outlet":
                    continue
                if port_item._connection is not None:
                    continue

                # Find nearby inlet
                best_inlet = None
                best_dist = float('inf')

                for other_id, other_node in self._nodes.items():
                    if other_id == node_id:
                        continue

                    for other_port_name, other_port in other_node.port_items.items():
                        if other_port.port_data.port_type != "inlet":
                            continue
                        if other_port._connection is not None:
                            continue

                        # Calculate distance
                        dist = (port_item.center_scene_pos() - other_port.center_scene_pos()).manhattanLength()
                        if dist < best_dist and dist < 300:  # Max auto-connect distance
                            best_dist = dist
                            best_inlet = other_port

                if best_inlet:
                    self._create_connection(port_item, best_inlet)
                    connections_made += 1

        if connections_made > 0:
            self.assembly_changed.emit(self.get_assembly_data())

        return connections_made

    def _create_connection(self, start_port: PortItem, end_port: PortItem):
        """Create a connection between two ports."""
        connection = ConnectionItem(start_port, end_port)
        self._scene.addItem(connection)
        self._connections.append(connection)

    def validate(self) -> List[str]:
        """
        Validate the assembly configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for disconnected ports
        for node_id, node in self._nodes.items():
            required_ports = {"inlet", "air_inlet", "dirty_air_inlet"}
            for port_name, port_item in node.port_items.items():
                if port_name in required_ports and port_item._connection is None:
                    errors.append(f"{node.node_data.component_type}: {port_name} not connected")

        # Check for cycles (simple check)
        # (More sophisticated validation could be added)

        # Check for required components
        component_types = [n.node_data.component_type for n in self._nodes.values()]
        if not any("Classifier" in ct for ct in component_types):
            errors.append("No classifier component found")

        return errors

    def get_assembly_data(self) -> Dict[str, Any]:
        """
        Get assembly configuration as dictionary.

        Returns:
            Assembly data including nodes, connections, and parameters
        """
        data = {
            "components": {},
            "connections": [],
        }

        for node_id, node in self._nodes.items():
            data["components"][node_id] = {
                "type": node.node_data.component_type,
                "category": node.node_data.category,
                "params": node.node_data.params,
                "position": node.node_data.position,
            }

        for conn in self._connections:
            data["connections"].append({
                "from_node": conn.start_port.parent_node.node_data.id,
                "from_port": conn.start_port.port_data.name,
                "to_node": conn.end_port.parent_node.node_data.id,
                "to_port": conn.end_port.port_data.name,
            })

        return data

    def save_state(self) -> Dict[str, Any]:
        """Save canvas state for project file."""
        return self.get_assembly_data()

    def load_state(self, state: Dict[str, Any]):
        """Load canvas state from project file."""
        self.clear()

        # Load components
        for node_id, comp_data in state.get("components", {}).items():
            node_data = NodeData(
                id=node_id,
                component_type=comp_data["type"],
                category=comp_data["category"],
                params=comp_data["params"],
                position=tuple(comp_data["position"]),
            )
            node_item = NodeItem(node_data)
            self._scene.addItem(node_item)
            self._nodes[node_id] = node_item

        # Load connections
        for conn_data in state.get("connections", []):
            from_node = self._nodes.get(conn_data["from_node"])
            to_node = self._nodes.get(conn_data["to_node"])
            if from_node and to_node:
                from_port = from_node.port_items.get(conn_data["from_port"])
                to_port = to_node.port_items.get(conn_data["to_port"])
                if from_port and to_port:
                    self._create_connection(from_port, to_port)

    def load_preset(self, preset: Dict[str, Any]):
        """Load a preset configuration."""
        self.load_state(preset.get("assembly", {}))

    def clear(self):
        """Clear all nodes and connections."""
        for conn in self._connections:
            self._scene.removeItem(conn)
        self._connections.clear()

        for node in self._nodes.values():
            self._scene.removeItem(node)
        self._nodes.clear()

        self.assembly_changed.emit(self.get_assembly_data())

    def undo(self):
        """Undo last action."""
        self._undo_stack.undo()

    def redo(self):
        """Redo last undone action."""
        self._undo_stack.redo()

    # Event handlers

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake_event = event
            fake_event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on a port
            item = self.itemAt(event.pos())
            if isinstance(item, PortItem):
                if self._pending_connection is None:
                    # Start connection
                    self._pending_connection = item
                    self._temp_line = self._scene.addLine(QLineF(
                        item.center_scene_pos(),
                        self.mapToScene(event.pos())
                    ), QPen(QColor(255, 255, 100), 2, Qt.PenStyle.DashLine))
                else:
                    # Complete connection
                    if self._is_valid_connection(self._pending_connection, item):
                        self._create_connection(self._pending_connection, item)
                        self.assembly_changed.emit(self.get_assembly_data())

                    # Clean up
                    if self._temp_line:
                        self._scene.removeItem(self._temp_line)
                        self._temp_line = None
                    self._pending_connection = None

                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        super().mouseReleaseEvent(event)

        # Check selection and emit signal
        selected = self._scene.selectedItems()
        if selected and isinstance(selected[0], NodeItem):
            node = selected[0]
            self.node_selected.emit(
                node.node_data.id,
                node.node_data.component_type,
                node.node_data.params
            )

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self._temp_line and self._pending_connection:
            # Update temporary connection line
            start = self._pending_connection.center_scene_pos()
            end = self.mapToScene(event.pos())
            self._temp_line.setLine(QLineF(start, end))

        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor

        self.scale(factor, factor)

    def contextMenuEvent(self, event):
        """Handle right-click context menu."""
        item = self.itemAt(event.pos())

        menu = QMenu(self)

        if isinstance(item, NodeItem):
            # Node context menu
            delete_action = menu.addAction("Delete Node")
            delete_action.triggered.connect(self.delete_selected)

            duplicate_action = menu.addAction("Duplicate Node")
            duplicate_action.triggered.connect(lambda: self._duplicate_node(item))

        elif isinstance(item, ConnectionItem):
            # Connection context menu
            delete_action = menu.addAction("Delete Connection")
            delete_action.triggered.connect(lambda: self._delete_connection(item))

        else:
            # Canvas context menu
            auto_connect_action = menu.addAction("Auto-Connect Ports")
            auto_connect_action.triggered.connect(self.auto_connect)

            menu.addSeparator()

            fit_action = menu.addAction("Fit All")
            fit_action.triggered.connect(self._fit_all)

        menu.exec(event.globalPos())

    def _is_valid_connection(self, port1: PortItem, port2: PortItem) -> bool:
        """Check if connection between two ports is valid."""
        # Can't connect to same node
        if port1.parent_node == port2.parent_node:
            return False

        # Can't connect same type (inlet-inlet or outlet-outlet)
        if port1.port_data.port_type == port2.port_data.port_type:
            return False

        # Can't connect if either already connected
        if port1._connection is not None or port2._connection is not None:
            return False

        return True

    def _duplicate_node(self, node: NodeItem):
        """Duplicate a node."""
        self.add_component_node(
            node.node_data.component_type,
            node.node_data.params.copy()
        )

    def _delete_connection(self, conn: ConnectionItem):
        """Delete a connection."""
        conn.disconnect()
        self._scene.removeItem(conn)
        self._connections.remove(conn)
        self.assembly_changed.emit(self.get_assembly_data())

    def _fit_all(self):
        """Fit all content in view."""
        items_rect = self._scene.itemsBoundingRect()
        self.fitInView(items_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Handle drag move."""
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop."""
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            data = json.loads(event.mimeData().data("application/x-airclassifier-component").data().decode())
            self.add_component_node(data["name"], data["params"])
            event.acceptProposedAction()
