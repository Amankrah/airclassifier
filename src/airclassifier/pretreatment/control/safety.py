"""
Safety Logic
============

Arc detection, recycle logic, and lockout conditions for the GP-15.

The GP-15 has several safety systems:
- MRH (Meter Relay High): Overcurrent protection -> RF off, recycle
- Arc detection: Reflected power spike -> immediate RF off
- Recycle limit: Max 4 restarts, then lockout
- Thermal limit: Valve thermal fuse at ~145 degC
- Ambient limit: >40 degC ambient -> warning
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyEvent(Enum):
    """Safety event types."""
    NONE = "none"
    MRH_TRIP = "mrh_trip"
    ARC_DETECTED = "arc_detected"
    RECYCLE_LIMIT = "recycle_limit"
    THERMAL_FUSE = "thermal_fuse"
    AMBIENT_HIGH = "ambient_high"


@dataclass
class SafetyStatus:
    """Current safety system state."""
    event: SafetyEvent = SafetyEvent.NONE
    rf_inhibited: bool = False
    lockout: bool = False
    recycle_count: int = 0
    message: str = ""


class SafetyMonitor:
    """Monitors GP-15 safety conditions each timestep."""

    def __init__(self, max_recycles: int = 4, restart_delay_s: float = 2.0):
        self._max_recycles = max_recycles
        self._restart_delay_s = restart_delay_s
        self.status = SafetyStatus()

    def check(
        self,
        anode_current_a: float,
        mrh_amps: float,
        reflected_power_kw: float = 0.0,
        valve_temp_c: float = 25.0,
        ambient_temp_c: float = 22.0,
    ) -> SafetyStatus:
        """Check all safety conditions. Returns updated status."""
        # TODO: Implement MRH check
        # TODO: Implement arc detection (reflected power spike)
        # TODO: Implement recycle count and lockout
        # TODO: Implement thermal fuse check
        raise NotImplementedError
