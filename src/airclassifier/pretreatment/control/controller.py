"""
PLC Controller
==============

Replicates the GP-15 PLC control logic as a discrete-event controller
that runs at each simulation timestep after the physics solve.

Control functions:
- Electrode gap: homing sequence, setpoint tracking, debounce
- MRH (Meter Relay High): overcurrent trip -> RF off, recycle
- MRL (Meter Relay Low): undercurrent -> belt stop
- Temperature control: optional auto mode with 6 sensor average
- Belt speed: setpoint from recipe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import MachineConfig, Recipe


class ControllerState(Enum):
    """PLC state machine states."""
    IDLE = "idle"
    HOMING = "homing"
    READY = "ready"
    RUNNING = "running"
    MRH_TRIP = "mrh_trip"
    MRL_STOP = "mrl_stop"
    ARC_LOCKOUT = "arc_lockout"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ControllerStatus:
    """Current controller output state."""
    state: ControllerState = ControllerState.IDLE
    electrode_gap_mm: float = 0.0
    belt_speed_m_per_min: float = 0.0
    rf_enabled: bool = False
    heater_bank_1_on: bool = False
    heater_bank_2_on: bool = False
    extraction_fan_hz: float = 0.0
    recycle_count: int = 0
    mrh_active: bool = False
    mrl_active: bool = False


class GP15Controller:
    """Simulates the GP-15 PLC control logic.

    Runs once per simulation timestep to check anode current,
    manage the electrode gap, and enforce safety limits.
    """

    def __init__(self, machine: MachineConfig):
        self._machine = machine
        self.status = ControllerStatus()
        self._recipe: Optional[Recipe] = None
        self._recycle_count = 0

    def load_recipe(self, recipe: Recipe):
        """Load a recipe and configure setpoints."""
        self._recipe = recipe
        self.status.electrode_gap_mm = recipe.electrode_gap_mm
        self.status.belt_speed_m_per_min = recipe.belt_speed_m_per_min
        self.status.rf_enabled = recipe.rf_power_enabled
        self.status.heater_bank_1_on = recipe.heater_bank_1_on
        self.status.heater_bank_2_on = recipe.heater_bank_2_on
        self.status.extraction_fan_hz = recipe.extraction_fan_hz
        self._recycle_count = 0
        self.status.state = ControllerState.READY

    def start(self):
        """Transition from READY to RUNNING."""
        if self.status.state == ControllerState.READY:
            self.status.state = ControllerState.RUNNING

    def step(
        self,
        dt: float,
        anode_current_a: float,
        rf_power_kw: float,
        T_outfeed_c: float,
    ) -> ControllerStatus:
        """Execute one controller timestep.

        Args:
            dt: Timestep [s].
            anode_current_a: Measured anode current [A].
            rf_power_kw: Delivered RF power [kW].
            T_outfeed_c: Average temperature at outfeed [degC].

        Returns:
            Updated controller status.
        """
        # TODO: Implement MRH check (overcurrent trip)
        # TODO: Implement MRL check (undercurrent belt stop)
        # TODO: Implement recycle logic (up to max_recycle_restarts)
        # TODO: Implement temperature control mode
        # TODO: Implement electrode gap debounce
        raise NotImplementedError

    def emergency_stop(self):
        """Trigger emergency stop — RF off, belt stop, lockout."""
        self.status.state = ControllerState.EMERGENCY_STOP
        self.status.rf_enabled = False
        self.status.belt_speed_m_per_min = 0.0
