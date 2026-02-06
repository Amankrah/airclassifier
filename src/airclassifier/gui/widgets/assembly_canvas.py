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
import math

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsTextItem, QGraphicsPathItem, QGraphicsDropShadowEffect,
    QMenu, QInputDialog, QMessageBox, QGraphicsProxyWidget,
)
from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF, QLineF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QLinearGradient, QTransform, QUndoStack, QUndoCommand,
)

from ..theme import COLORS


# Color scheme for component categories
CATEGORY_COLORS = {
    "Classification": QColor(COLORS.CAT_CLASSIFICATION),
    "Cyclones":       QColor(COLORS.CAT_CYCLONES),
    "Filtration":     QColor(COLORS.CAT_FILTRATION),
    "Feed System":    QColor(COLORS.CAT_FEED),
    "Air System":     QColor(COLORS.CAT_AIR),
    "Ductwork":       QColor(COLORS.CAT_DUCTWORK),
    "Exhaust":        QColor(COLORS.CAT_EXHAUST),
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


# --------------------------------------------------------------------------
# Port
# --------------------------------------------------------------------------

class PortItem(QGraphicsEllipseItem):
    """Visual representation of a connection port."""

    def __init__(self, port_data: PortData, parent_node: 'NodeItem'):
        size = 10
        super().__init__(-size / 2, -size / 2, size, size, parent_node)

        self.port_data = port_data
        self.parent_node = parent_node
        self._connection: Optional['ConnectionItem'] = None

        # Style based on port type
        if port_data.port_type == "inlet":
            self.setBrush(QBrush(QColor(COLORS.SUCCESS)))
        else:
            self.setBrush(QBrush(QColor(COLORS.WARNING)))

        self.setPen(QPen(QColor(COLORS.BG_DARKEST), 1.5))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(10)

        # Position relative to parent
        self.setPos(QPointF(*port_data.position))

        # Tooltip
        self.setToolTip(f"{port_data.name} ({port_data.port_type})")

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor(COLORS.ACCENT_HOVER), 2.5))
        self.setScale(1.3)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(QColor(COLORS.BG_DARKEST), 1.5))
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def center_scene_pos(self) -> QPointF:
        return self.scenePos()


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class NodeItem(QGraphicsRectItem):
    """Visual representation of a component node with modern styling."""

    NODE_WIDTH = 180
    NODE_HEIGHT = 80
    HEADER_H = 24
    CORNER_R = 8

    def __init__(self, node_data: NodeData):
        super().__init__()

        self.node_data = node_data
        self.port_items: Dict[str, PortItem] = {}

        self.setRect(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT)
        self.setPos(QPointF(*node_data.position))

        # Make movable and selectable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Style
        self._cat_color = CATEGORY_COLORS.get(node_data.category, QColor(100, 100, 100))
        self._setup_style()

        # Title
        self._add_title()

        # Ports
        self._add_ports()

        # Subtle drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

    def _setup_style(self):
        """Setup node visual style -- solid dark card, thin accent top."""
        self.setBrush(QBrush(QColor(COLORS.BG_ELEVATED)))
        self.setPen(QPen(QColor(COLORS.BORDER), 1))

    def paint(self, painter: QPainter, option, widget=None):
        """Custom paint for rounded rect with accent header."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        radius = self.CORNER_R

        # Body
        body_path = QPainterPath()
        body_path.addRoundedRect(r, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(COLORS.BG_ELEVATED)))
        painter.drawPath(body_path)

        # Header accent bar
        header_path = QPainterPath()
        header_path.moveTo(r.left(), r.top() + radius)
        header_path.arcTo(r.left(), r.top(), radius * 2, radius * 2, 180, -90)
        header_path.lineTo(r.right() - radius, r.top())
        header_path.arcTo(r.right() - radius * 2, r.top(), radius * 2, radius * 2, 90, -90)
        header_path.lineTo(r.right(), r.top() + self.HEADER_H)
        header_path.lineTo(r.left(), r.top() + self.HEADER_H)
        header_path.closeSubpath()

        painter.setBrush(QBrush(self._cat_color.darker(130)))
        painter.drawPath(header_path)

        # Border
        is_selected = self.isSelected()
        border_color = QColor(COLORS.ACCENT) if is_selected else QColor(COLORS.BORDER)
        painter.setPen(QPen(border_color, 1.5 if is_selected else 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(r, radius, radius)

    def _add_title(self):
        """Add title text to node."""
        title = QGraphicsTextItem(self.node_data.component_type, self)
        title.setDefaultTextColor(QColor(COLORS.TEXT_INVERSE))
        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        title.setFont(font)

        title_width = title.boundingRect().width()
        title.setPos(max(4, (self.NODE_WIDTH - title_width) / 2), 2)

        # Node ID (small label at bottom)
        id_text = QGraphicsTextItem(self.node_data.id[:8], self)
        id_text.setDefaultTextColor(QColor(COLORS.TEXT_MUTED))
        id_text.setFont(QFont("Segoe UI", 6))
        id_text.setPos(6, self.NODE_HEIGHT - 15)

    def _add_ports(self):
        """Add port items to node."""
        default_ports = self._get_default_ports()

        inlet_y = self.HEADER_H + 12
        outlet_y = self.HEADER_H + 12

        for port_data in default_ports:
            if port_data.port_type == "inlet":
                port_data.position = (-5, inlet_y)
                inlet_y += 20
            else:
                port_data.position = (self.NODE_WIDTH + 5, outlet_y)
                outlet_y += 20

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
        """Snap-to-grid on position change and update connections."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Snap to 25px grid
            grid = 25
            snapped = QPointF(
                round(value.x() / grid) * grid,
                round(value.y() / grid) * grid,
            )
            return snapped

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node_data.position = (value.x(), value.y())
            for port_item in self.port_items.values():
                if port_item._connection:
                    port_item._connection.update_path()

        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.update()
        super().hoverLeaveEvent(event)


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

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
        self.setPen(QPen(QColor(COLORS.TEXT_SECONDARY), 2, Qt.PenStyle.SolidLine))
        self.setZValue(-1)
        self.setAcceptHoverEvents(True)

        self.update_path()

    def update_path(self):
        """Update the path with a smooth cubic bezier."""
        start = self.start_port.center_scene_pos()
        end = self.end_port.center_scene_pos()

        path = QPainterPath(start)
        dx = abs(end.x() - start.x())
        ctrl_dist = max(50, dx * 0.45)

        if start.x() < end.x():
            ctrl1 = QPointF(start.x() + ctrl_dist, start.y())
            ctrl2 = QPointF(end.x() - ctrl_dist, end.y())
        else:
            ctrl1 = QPointF(start.x() - ctrl_dist, start.y())
            ctrl2 = QPointF(end.x() + ctrl_dist, end.y())

        path.cubicTo(ctrl1, ctrl2, end)
        self.setPath(path)

    def hoverEnterEvent(self, event):
        self.setPen(QPen(QColor(COLORS.ACCENT_HOVER), 3))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(QColor(COLORS.TEXT_SECONDARY), 2))
        super().hoverLeaveEvent(event)

    def disconnect(self):
        self.start_port._connection = None
        self.end_port._connection = None
        self.start_port.port_data.connected_to = None
        self.end_port.port_data.connected_to = None


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

class AssemblyCanvas(QGraphicsView):
    """
    Node-based visual editor for assembling air classifier components.

    Features:
    - Drag and drop component nodes
    - Connect ports with bezier curves
    - Pan and zoom navigation (mouse-wheel + middle-button)
    - Snap-to-grid for tidy alignment
    - Undo/redo support
    - Export assembly configuration
    """

    GRID_SIZE = 25
    GRID_COLOR_MINOR = QColor(COLORS.BG_SURFACE)
    GRID_COLOR_MAJOR = QColor(COLORS.BORDER_SUBTLE)

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
        """Setup the graphics scene (no grid items -- grid is drawn in drawBackground)."""
        self._scene = QGraphicsScene()
        self._scene.setSceneRect(-5000, -5000, 10000, 10000)
        self.setScene(self._scene)

    def _setup_view(self):
        """Setup view properties."""
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setAcceptDrops(True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setBackgroundBrush(QBrush(QColor(COLORS.BG_DARKEST)))

    # -- efficient grid via drawBackground (no scene items) --

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw a dot-grid background for a cleaner look."""
        super().drawBackground(painter, rect)

        gs = self.GRID_SIZE
        left = int(math.floor(rect.left() / gs)) * gs
        top = int(math.floor(rect.top() / gs)) * gs
        right = int(math.ceil(rect.right() / gs)) * gs
        bottom = int(math.ceil(rect.bottom() / gs)) * gs

        # Dots instead of lines -- lighter and modern
        painter.setPen(Qt.PenStyle.NoPen)

        for x in range(left, right + 1, gs):
            for y in range(top, bottom + 1, gs):
                is_major = (x % (gs * 4) == 0) and (y % (gs * 4) == 0)
                if is_major:
                    painter.setBrush(QBrush(self.GRID_COLOR_MAJOR))
                    painter.drawEllipse(QPointF(x, y), 1.5, 1.5)
                else:
                    painter.setBrush(QBrush(self.GRID_COLOR_MINOR))
                    painter.drawEllipse(QPointF(x, y), 0.8, 0.8)

    # -- component management --

    @Slot(str, dict)
    def add_component_node(self, component_type: str, params: Dict[str, Any]):
        category = self._get_category(component_type)
        node_id = str(uuid.uuid4())
        position = self._get_spawn_position()

        node_data = NodeData(
            id=node_id,
            component_type=component_type,
            category=category,
            params=params.copy(),
            position=position,
        )

        node_item = NodeItem(node_data)
        self._scene.addItem(node_item)
        self._nodes[node_id] = node_item
        node_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        self.assembly_changed.emit(self.get_assembly_data())

    def _get_category(self, component_type: str) -> str:
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
        center = self.mapToScene(self.viewport().rect().center())
        offset = len(self._nodes) * 25
        gs = self.GRID_SIZE
        x = round((center.x() + offset) / gs) * gs
        y = round((center.y() + offset) / gs) * gs
        return (x, y)

    def delete_selected(self):
        for item in self._scene.selectedItems():
            if isinstance(item, NodeItem):
                for port_item in item.port_items.values():
                    if port_item._connection:
                        conn = port_item._connection
                        conn.disconnect()
                        self._scene.removeItem(conn)
                        if conn in self._connections:
                            self._connections.remove(conn)
                del self._nodes[item.node_data.id]
                self._scene.removeItem(item)

        self.assembly_changed.emit(self.get_assembly_data())

    @Slot(str, str, object)
    def update_node_params(self, node_id: str, param_name: str, value: Any):
        if node_id in self._nodes:
            self._nodes[node_id].node_data.params[param_name] = value
            self.assembly_changed.emit(self.get_assembly_data())

    def auto_connect(self) -> int:
        connections_made = 0
        for node_id, node in self._nodes.items():
            for port_name, port_item in node.port_items.items():
                if port_item.port_data.port_type != "outlet":
                    continue
                if port_item._connection is not None:
                    continue

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
                        dist = (port_item.center_scene_pos() - other_port.center_scene_pos()).manhattanLength()
                        if dist < best_dist and dist < 300:
                            best_dist = dist
                            best_inlet = other_port

                if best_inlet:
                    self._create_connection(port_item, best_inlet)
                    connections_made += 1

        if connections_made > 0:
            self.assembly_changed.emit(self.get_assembly_data())
        return connections_made

    def _create_connection(self, start_port: PortItem, end_port: PortItem):
        connection = ConnectionItem(start_port, end_port)
        self._scene.addItem(connection)
        self._connections.append(connection)

    def validate(self) -> List[str]:
        errors = []
        for node_id, node in self._nodes.items():
            required_ports = {"inlet", "air_inlet", "dirty_air_inlet"}
            for port_name, port_item in node.port_items.items():
                if port_name in required_ports and port_item._connection is None:
                    errors.append(f"{node.node_data.component_type}: {port_name} not connected")

        component_types = [n.node_data.component_type for n in self._nodes.values()]
        if not any("Classifier" in ct for ct in component_types):
            errors.append("No classifier component found")
        return errors

    def get_assembly_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"components": {}, "connections": []}
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
        return self.get_assembly_data()

    def load_state(self, state: Dict[str, Any]):
        self.clear()
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

        for conn_data in state.get("connections", []):
            from_node = self._nodes.get(conn_data["from_node"])
            to_node = self._nodes.get(conn_data["to_node"])
            if from_node and to_node:
                from_port = from_node.port_items.get(conn_data["from_port"])
                to_port = to_node.port_items.get(conn_data["to_port"])
                if from_port and to_port:
                    self._create_connection(from_port, to_port)

    def load_preset(self, preset: Dict[str, Any]):
        self.load_state(preset.get("assembly", {}))

    def clear(self):
        for conn in self._connections:
            self._scene.removeItem(conn)
        self._connections.clear()
        for node in self._nodes.values():
            self._scene.removeItem(node)
        self._nodes.clear()
        self.assembly_changed.emit(self.get_assembly_data())

    def undo(self):
        self._undo_stack.undo()

    def redo(self):
        self._undo_stack.redo()

    # ============================================================
    # Event handlers
    # ============================================================

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake_event = event
            fake_event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, PortItem):
                if self._pending_connection is None:
                    self._pending_connection = item
                    self._temp_line = self._scene.addLine(
                        QLineF(item.center_scene_pos(), self.mapToScene(event.pos())),
                        QPen(QColor(COLORS.ACCENT_HOVER), 2, Qt.PenStyle.DashLine),
                    )
                else:
                    if self._is_valid_connection(self._pending_connection, item):
                        self._create_connection(self._pending_connection, item)
                        self.assembly_changed.emit(self.get_assembly_data())
                    if self._temp_line:
                        self._scene.removeItem(self._temp_line)
                        self._temp_line = None
                    self._pending_connection = None
                return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().mouseReleaseEvent(event)

        selected = self._scene.selectedItems()
        if selected and isinstance(selected[0], NodeItem):
            node = selected[0]
            self.node_selected.emit(
                node.node_data.id,
                node.node_data.component_type,
                node.node_data.params,
            )

    def mouseMoveEvent(self, event):
        if self._temp_line and self._pending_connection:
            start = self._pending_connection.center_scene_pos()
            end = self.mapToScene(event.pos())
            self._temp_line.setLine(QLineF(start, end))
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        self.scale(factor, factor)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)

        if isinstance(item, NodeItem):
            delete_action = menu.addAction("Delete Node")
            delete_action.triggered.connect(self.delete_selected)
            dup_action = menu.addAction("Duplicate Node")
            dup_action.triggered.connect(lambda: self._duplicate_node(item))
        elif isinstance(item, ConnectionItem):
            delete_action = menu.addAction("Delete Connection")
            delete_action.triggered.connect(lambda: self._delete_connection(item))
        else:
            auto_action = menu.addAction("Auto-Connect Ports")
            auto_action.triggered.connect(self.auto_connect)
            menu.addSeparator()
            fit_action = menu.addAction("Fit All")
            fit_action.triggered.connect(self._fit_all)

        menu.exec(event.globalPos())

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts inside the canvas."""
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel pending connection
            if self._temp_line:
                self._scene.removeItem(self._temp_line)
                self._temp_line = None
            self._pending_connection = None
        else:
            super().keyPressEvent(event)

    # ---- helpers ----

    def _is_valid_connection(self, port1: PortItem, port2: PortItem) -> bool:
        if port1.parent_node == port2.parent_node:
            return False
        if port1.port_data.port_type == port2.port_data.port_type:
            return False
        if port1._connection is not None or port2._connection is not None:
            return False
        return True

    def _duplicate_node(self, node: NodeItem):
        self.add_component_node(node.node_data.component_type, node.node_data.params.copy())

    def _delete_connection(self, conn: ConnectionItem):
        conn.disconnect()
        self._scene.removeItem(conn)
        self._connections.remove(conn)
        self.assembly_changed.emit(self.get_assembly_data())

    def _fit_all(self):
        items_rect = self._scene.itemsBoundingRect()
        if not items_rect.isEmpty():
            self.fitInView(items_rect.adjusted(-40, -40, 40, 40), Qt.AspectRatioMode.KeepAspectRatio)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-airclassifier-component"):
            data = json.loads(event.mimeData().data("application/x-airclassifier-component").data().decode())
            self.add_component_node(data["name"], data["params"])
            event.acceptProposedAction()
