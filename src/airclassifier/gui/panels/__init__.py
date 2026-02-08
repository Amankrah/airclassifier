"""
GUI Panels for ProteinProcessIO
=======================================

Dockable panel widgets for the main window.
"""

from .component_palette import ComponentPalette
from .property_editor import PropertyEditor
from .simulation_control import SimulationControlPanel
from .results_panel import ResultsPanel

__all__ = [
    "ComponentPalette",
    "PropertyEditor",
    "SimulationControlPanel",
    "ResultsPanel",
]
