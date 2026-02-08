"""
GP-15 Machine Components
========================

Individual geometric components of the GP-15 RF dielectric heating machine.
Each component follows the Params + Geometry pattern and provides:

- ``generate_mesh()`` returning (vertices, triangles, metadata)
- ``@property ports`` returning Dict[str, ConnectionPort] for assembly alignment

Components:
- Generator: RF oscillator cabinet
- OvenChamber: Main processing chamber (refactored from oven.py)
- ConveyorBelt: Belt and material bed (refactored from conveyor.py)
- Electrode: Upper/lower electrode plates
- Tunnel: Attenuation tunnels (infeed/outfeed)
- InfeedHopper: Material hopper with sizing plate
- EMU: Environment Management Unit (extraction duct + heaters)
- HMIPanel: Control console
- Housing: Outer cabinet walls
- SupportLegs: Machine stands
"""

from .generator import GeneratorGeometry, GeneratorParams
from .tunnel import TunnelGeometry, TunnelParams
from .hopper import InfeedHopperGeometry, InfeedHopperParams
from .emu import EMUGeometry, EMUParams
from .hmi_panel import HMIPanelGeometry, HMIPanelParams
from .housing import HousingGeometry, HousingParams
from .support_legs import SupportLegsGeometry, SupportLegsParams
from .oven_chamber import OvenChamberGeometry, OvenChamberParams
from .conveyor_belt import ConveyorBeltGeometry, ConveyorBeltParams
from .electrode import ElectrodeGeometry, ElectrodeParams

__all__ = [
    # New components
    "GeneratorGeometry", "GeneratorParams",
    "TunnelGeometry", "TunnelParams",
    "InfeedHopperGeometry", "InfeedHopperParams",
    "EMUGeometry", "EMUParams",
    "HMIPanelGeometry", "HMIPanelParams",
    "HousingGeometry", "HousingParams",
    "SupportLegsGeometry", "SupportLegsParams",
    # Refactored components
    "OvenChamberGeometry", "OvenChamberParams",
    "ConveyorBeltGeometry", "ConveyorBeltParams",
    "ElectrodeGeometry", "ElectrodeParams",
]
