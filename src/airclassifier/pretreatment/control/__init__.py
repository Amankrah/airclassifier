"""
Pretreatment Control System
============================

Simulation of the GP-15 PLC control logic:
- Recipe system (30-recipe storage, HMI recipe mirror)
- Electrode gap controller (homing, setpoint, MRH/MRL protection)
- Temperature control (optional auto mode with 6 sensors)
- Safety logic (arc detection, recycle, lockout)
"""

from .controller import GP15Controller, ControllerState, ControllerStatus
from .recipe import RecipeStore
from .safety import SafetyEvent, SafetyMonitor, SafetyStatus

__all__ = [
    # Controller
    "GP15Controller",
    "ControllerState",
    "ControllerStatus",
    # Safety
    "SafetyMonitor",
    "SafetyEvent",
    "SafetyStatus",
    # Recipe storage
    "RecipeStore",
]
