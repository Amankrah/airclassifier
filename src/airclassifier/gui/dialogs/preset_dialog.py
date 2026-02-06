"""
Preset Selection Dialog
=======================

Dialog for selecting and loading predefined classifier configurations.
"""

from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QTextEdit, QGroupBox, QDialogButtonBox, QWidget,
    QSplitter,
)
from PySide6.QtCore import Qt


# Predefined classifier configurations
PRESETS: List[Dict[str, Any]] = [
    {
        "name": "Standard Protein Classifier",
        "description": """Complete air classification system for protein extraction from plant-based materials.

Mode: use_preclassification = True (Full System)

Flow path:
1. Venturi eductor entrains powder in high-velocity air
2. Zigzag classifier performs gravity-based pre-separation
3. Wheel classifier achieves fine centrifugal separation
4. Multi-stage cyclones collect separated fractions
5. Bag filter captures ultrafine particles

Features:
- Venturi eductor for particle entrainment
- 5-stage zigzag pre-classifier (d50 ~50µm)
- Wheel classifier for fine cut (d50 ~25µm)
- 3-stage cyclone system for staged collection
- Bag filter for final capture

Suitable for: Yellow peas, faba beans, oats
Throughput: 100-500 kg/h
Target protein purity: >70%""",
        "assembly": {
            "use_preclassification": True,  # Full system with zigzag + wheel
            "components": {
                "venturi-1": {
                    "type": "Venturi Eductor",
                    "category": "Classification",
                    "params": {"inlet_diameter": 0.08, "throat_ratio": 0.5},
                    "position": (100, 100),
                },
                "zigzag-1": {
                    "type": "Zigzag Classifier",
                    "category": "Classification",
                    "params": {"channel_width": 0.06, "num_stages": 5},
                    "position": (350, 100),
                },
                "wheel-1": {
                    "type": "Wheel Classifier",
                    "category": "Classification",
                    "params": {"wheel_diameter": 0.20, "wheel_rpm": 8000},
                    "position": (600, 100),
                },
                "cyclone-1": {
                    "type": "Multi-Cyclone System",
                    "category": "Cyclones",
                    "params": {"num_stages": 3},
                    "position": (850, 100),
                },
                "bagfilter-1": {
                    "type": "Bag Filter",
                    "category": "Filtration",
                    "params": {"flow_rate": 0.15},
                    "position": (1100, 100),
                },
            },
            "connections": [
                {"from_node": "venturi-1", "from_port": "outlet", "to_node": "zigzag-1", "to_port": "air_inlet"},
                {"from_node": "zigzag-1", "from_port": "fines_outlet", "to_node": "wheel-1", "to_port": "inlet"},
                {"from_node": "wheel-1", "from_port": "fines_outlet", "to_node": "cyclone-1", "to_port": "inlet"},
                {"from_node": "cyclone-1", "from_port": "overflow", "to_node": "bagfilter-1", "to_port": "dirty_air_inlet"},
            ],
        },
    },
    {
        "name": "Simple Cyclone Classifier",
        "description": """Basic cyclone-based classifier without wheel classifier.

Mode: use_preclassification = True (uses venturi + zigzag)

Features:
- Venturi eductor for entrainment
- 4-stage zigzag pre-classifier
- Single large cyclone for collection
- Bag filter for fines

Suitable for: Coarser separations (d50 >40µm)
Throughput: 200-800 kg/h
Lower capital cost, simpler operation
Good for initial separation before further processing""",
        "assembly": {
            "use_preclassification": True,
            "components": {
                "venturi-1": {
                    "type": "Venturi Eductor",
                    "category": "Classification",
                    "params": {"inlet_diameter": 0.10, "throat_ratio": 0.5},
                    "position": (100, 100),
                },
                "zigzag-1": {
                    "type": "Zigzag Classifier",
                    "category": "Classification",
                    "params": {"channel_width": 0.08, "num_stages": 4},
                    "position": (350, 100),
                },
                "cyclone-1": {
                    "type": "Cyclone (Primary)",
                    "category": "Cyclones",
                    "params": {"diameter": 0.35},
                    "position": (600, 100),
                },
                "bagfilter-1": {
                    "type": "Bag Filter",
                    "category": "Filtration",
                    "params": {"flow_rate": 0.20},
                    "position": (850, 100),
                },
            },
            "connections": [
                {"from_node": "venturi-1", "from_port": "outlet", "to_node": "zigzag-1", "to_port": "air_inlet"},
                {"from_node": "zigzag-1", "from_port": "fines_outlet", "to_node": "cyclone-1", "to_port": "inlet"},
                {"from_node": "cyclone-1", "from_port": "overflow", "to_node": "bagfilter-1", "to_port": "dirty_air_inlet"},
            ],
        },
    },
    {
        "name": "Wheel-Only Classifier (No Preclassification)",
        "description": """High-performance wheel classifier without pre-classification stages.

Mode: use_preclassification = False

This mode uses a 3-point junction that merges:
- Air inlet (from blower system)
- Solids chute (15° angled feed from hopper)
- Wheel classifier inlet

Features:
- Direct feed to wheel classifier via 3-point junction
- High-speed centrifugal separation (1000-5000g force)
- Multi-cyclone staged collection
- Bag filter for ultrafine capture

Suitable for: Pre-ground materials requiring fine separation
Throughput: 50-200 kg/h
Best protein purity (>75%), highest energy consumption
Cut size: 20-30 µm achievable""",
        "assembly": {
            "use_preclassification": False,  # KEY: Wheel-only mode
            "components": {
                "wheel-1": {
                    "type": "Wheel Classifier",
                    "category": "Classification",
                    "params": {"wheel_diameter": 0.25, "wheel_rpm": 10000, "num_blades": 32},
                    "position": (100, 100),
                },
                "cyclone-1": {
                    "type": "Multi-Cyclone System",
                    "category": "Cyclones",
                    "params": {"num_stages": 3},
                    "position": (400, 100),
                },
                "bagfilter-1": {
                    "type": "Bag Filter",
                    "category": "Filtration",
                    "params": {"flow_rate": 0.12},
                    "position": (700, 100),
                },
            },
            "connections": [
                {"from_node": "wheel-1", "from_port": "fines_outlet", "to_node": "cyclone-1", "to_port": "inlet"},
                {"from_node": "cyclone-1", "from_port": "overflow", "to_node": "bagfilter-1", "to_port": "dirty_air_inlet"},
            ],
        },
    },
    {
        "name": "Lab-Scale Classifier",
        "description": """Small-scale classifier for laboratory testing.

Mode: use_preclassification = True

Features:
- Compact venturi eductor (40mm inlet)
- 3-stage zigzag classifier (30mm channel)
- Single small cyclone (80mm diameter)
- Mini bag filter

Suitable for: Process development, small batches
Throughput: 5-20 kg/h
Low material requirements for testing
Ideal for parameter optimization before scale-up""",
        "assembly": {
            "use_preclassification": True,
            "components": {
                "venturi-1": {
                    "type": "Venturi Eductor",
                    "category": "Classification",
                    "params": {"inlet_diameter": 0.04, "throat_ratio": 0.5},
                    "position": (100, 100),
                },
                "zigzag-1": {
                    "type": "Zigzag Classifier",
                    "category": "Classification",
                    "params": {"channel_width": 0.03, "num_stages": 3},
                    "position": (350, 100),
                },
                "cyclone-1": {
                    "type": "Cyclone (Tertiary)",
                    "category": "Cyclones",
                    "params": {"diameter": 0.08},
                    "position": (600, 100),
                },
                "bagfilter-1": {
                    "type": "Bag Filter",
                    "category": "Filtration",
                    "params": {"flow_rate": 0.03, "num_bags": 4},
                    "position": (850, 100),
                },
            },
            "connections": [
                {"from_node": "venturi-1", "from_port": "outlet", "to_node": "zigzag-1", "to_port": "air_inlet"},
                {"from_node": "zigzag-1", "from_port": "fines_outlet", "to_node": "cyclone-1", "to_port": "inlet"},
                {"from_node": "cyclone-1", "from_port": "overflow", "to_node": "bagfilter-1", "to_port": "dirty_air_inlet"},
            ],
        },
    },
]


class PresetDialog(QDialog):
    """Dialog for selecting classifier presets."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Load Preset Configuration")
        self.setMinimumSize(700, 500)

        self.selected_preset: Optional[Dict[str, Any]] = None
        self._setup_ui()
        self._populate_list()

    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Select a predefined classifier configuration:")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.setMinimumWidth(250)
        self.preset_list.currentItemChanged.connect(self._on_selection_changed)
        self.preset_list.itemDoubleClicked.connect(self._on_double_click)
        splitter.addWidget(self.preset_list)

        # Description panel
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_layout.setContentsMargins(0, 0, 0, 0)

        self.desc_title = QLabel()
        self.desc_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        desc_layout.addWidget(self.desc_title)

        self.desc_text = QTextEdit()
        self.desc_text.setReadOnly(True)
        desc_layout.addWidget(self.desc_text)

        splitter.addWidget(desc_widget)

        # Set splitter sizes
        splitter.setSizes([250, 400])

        layout.addWidget(splitter)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_list(self):
        """Populate preset list."""
        for preset in PRESETS:
            item = QListWidgetItem(preset["name"])
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.preset_list.addItem(item)

        # Select first item
        if self.preset_list.count() > 0:
            self.preset_list.setCurrentRow(0)

    def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle selection change."""
        if current:
            preset = current.data(Qt.ItemDataRole.UserRole)
            self.desc_title.setText(preset["name"])
            self.desc_text.setPlainText(preset["description"])

    def _on_double_click(self, item: QListWidgetItem):
        """Handle double-click on item."""
        self._accept_selection()

    def _accept_selection(self):
        """Accept current selection."""
        current = self.preset_list.currentItem()
        if current:
            self.selected_preset = current.data(Qt.ItemDataRole.UserRole)
            self.accept()
