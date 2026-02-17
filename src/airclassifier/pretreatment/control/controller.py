"""
PLC Controller
==============

Replicates the GP-15 PLC control logic as a discrete-event controller
that runs at each simulation timestep after the physics solve.

Engineering guide §8:
- Electrode gap: homing sequence, setpoint tracking, debounce
- MRH (Meter Relay High): overcurrent trip → RF off, recycle
- MRL (Meter Relay Low): undercurrent → electrode drive stop
- Temperature control: optional auto mode with 6-sensor average
- Recycle: via SafetyMonitor (4 attempts then lockout)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..config import MachineConfig, Recipe
from ..calibration_store import get_calibration_defaults
from .safety import SafetyMonitor, SafetyEvent


class ControllerState(Enum):
    """PLC state machine states."""
    IDLE = "idle"
    HOMING = "homing"
    READY = "ready"
    RUNNING = "running"
    MRH_TRIP = "mrh_trip"
    MRL_STOP = "mrl_stop"
    RECYCLE = "recycle"
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

    Implements the control hierarchy from the engineering guide §8:

    1. **Safety** — SafetyMonitor checks MRH, arc, thermal fuse.
       If a fault occurs, RF is inhibited and a recycle sequence
       starts (up to 4 attempts, then lockout).

    2. **Electrode gap** — Asymmetric open/close with proportional
       MRH response.  During overcurrent the gap opens proportionally
       to how far Ia exceeds MRH (fast response).  In the normal band
       (MRL < Ia < MRH) the gap drifts very slowly toward setpoint
       (matching Run#1 PLC: gap held at 87 mm for ~1200 s).  Under
       MRL with batch clearing, the gap returns to setpoint at a
       moderate close rate (~15% of open rate, matching Run#1 PLC
       close phase of ~0.03 mm/s).

    3. **Temperature control** (optional) — When
       ``recipe.temp_control_enabled`` is True, a closed-loop
       controller adjusts the electrode gap and belt speed to hold
       the outfeed temperature at the recipe setpoint.

    Gap rate ratios derived from Run#1 PLC data analysis::

        Phase            PLC rate   Model
        ─────────────    ─────────  ──────────────────────────────
        MRH opening      0.19 mm/s  gap_rate × (1 + overshoot/MRH)
        Normal band hold  0.001 mm/s  gap_rate × _GAP_DRIFT_RATIO
        MRL closing       0.03 mm/s  gap_rate × _GAP_CLOSE_RATIO
    """

    # ── Asymmetric gap drive ratios ──────────────────────────────────
    # Derived from Run#1 PLC data (61 kg whole yellow pea, gap=75 mm).
    #
    # The PLC shows the electrode gap opens quickly during MRH events
    # but returns to setpoint very slowly:
    #   - Open:   75→87 mm in ~60 s   (0.19 mm/s)
    #   - Hold:   87→86.7 mm in 600 s (0.001 mm/s, normal band drift)
    #   - Close:  84→75 mm in 300 s   (0.03 mm/s, MRL/batch clearing)
    #
    # _GAP_CLOSE_RATIO: fraction of gap_adjust_rate for MRL closing.
    #   0.15 × 0.19 ≈ 0.029 mm/s (matches PLC close phase)
    #
    # _GAP_DRIFT_RATIO: fraction of gap_adjust_rate for normal-band
    #   drift toward setpoint.  Very slow — models the PLC's tendency
    #   to hold the elevated gap during steady-state processing.
    #   0.005 × 0.19 ≈ 0.001 mm/s (matches PLC hold phase)
    _GAP_CLOSE_RATIO = 0.15
    _GAP_DRIFT_RATIO = 0.005

    # Gap adjustment step for temperature control [mm]
    _TEMP_GAP_STEP_MM = 2.0

    # Belt speed adjustment step for temperature control [m/min]
    _TEMP_SPEED_STEP = 0.05

    def __init__(self, machine: MachineConfig, gap_adjust_rate_mm_s: float | None = None):
        self._machine = machine
        self.gap_adjust_rate_mm_s = (
            gap_adjust_rate_mm_s if gap_adjust_rate_mm_s is not None
            else get_calibration_defaults()[2]
        )
        self.status = ControllerStatus()
        self._recipe: Optional[Recipe] = None
        self.safety = SafetyMonitor(
            max_recycles=machine.max_recycle_restarts,
            restart_delay_s=machine.restart_delay_s,
        )

        # Debounce state
        self._debounce_timer = 0.0
        self._gap_command_pending = False

        # Temperature control state
        self._temp_last_correction_time = 0.0
        self._temp_gap_correction_applied = False
        self._sim_time = 0.0

        # Batch mode state: when True, MRL condition means empty belt
        # (return gap to setpoint) rather than low load (hold position)
        self._batch_exhausted = False

    def load_recipe(self, recipe: Recipe):
        """Load a recipe and configure setpoints."""
        self._recipe = recipe
        self.status.electrode_gap_mm = recipe.electrode_gap_mm
        self.status.belt_speed_m_per_min = recipe.belt_speed_m_per_min
        self.status.rf_enabled = recipe.rf_power_enabled
        self.status.heater_bank_1_on = recipe.heater_bank_1_on
        self.status.heater_bank_2_on = recipe.heater_bank_2_on
        self.status.extraction_fan_hz = recipe.extraction_fan_hz
        self.status.state = ControllerState.READY
        self.safety.reset()
        self.safety.update_monitor_interval(
            recipe.belt_speed_m_per_min,
            self._machine.oven_length_m,
        )

    def start(self):
        """Transition from READY to RUNNING."""
        if self.status.state == ControllerState.READY:
            self.status.state = ControllerState.RUNNING

    def set_batch_exhausted(self, exhausted: bool):
        """Signal that the oven chamber has cleared (run-out complete).

        Called when the M=0 front (injected after hopper empties) has
        advected from the hopper through the entire oven chamber to
        the outfeed attenuation duct exit (``oven_x_end``).

        Manual p.54: *"At the end of the production run allow product
        to 'run-out' to ensure there is no product in the GP-15, then
        press the GP-15 OFF button to stop processing."*

        The delay from hopper-empty to this signal accounts for belt
        travel time from hopper discharge to oven exit — material is
        "in the GP-15" until it passes the outfeed attenuation duct,
        not just the electrodes (Engineering Guide §2.2.3).

        When batch_exhausted=True and Ia < MRL, the controller returns
        the gap to setpoint (oven clearing, load won't recover).
        Without this flag, Ia < MRL would hold the gap (momentary
        low load during continuous operation, expect recovery).

        Matches Run#2 PLC: gap returned 94.1→75.2 mm after material
        cleared the oven, not when the hopper emptied.
        """
        self._batch_exhausted = exhausted

    def step(
        self,
        dt: float,
        anode_current_a: float,
        rf_power_kw: float,
        T_outfeed_c: float,
        e_field_max: float = 0.0,
    ) -> ControllerStatus:
        """Execute one controller timestep.

        Args:
            dt: Timestep [s].
            anode_current_a: Measured anode current [A].
            rf_power_kw: Delivered RF power [kW].
            T_outfeed_c: Average outfeed temperature [°C].
            e_field_max: Peak E-field for arc detection [V/m].

        Returns:
            Updated controller status.
        """
        self._sim_time += dt
        recipe = self._recipe
        if recipe is None or self.status.state in (
            ControllerState.IDLE,
            ControllerState.EMERGENCY_STOP,
        ):
            return self.status

        # ── 1. SAFETY CHECK ───────────────────────────────────────────
        safety = self.safety.check(
            anode_current_a=anode_current_a,
            mrh_amps=recipe.mrh_amps,
            e_field_max_v_per_m=e_field_max,
            dt=dt,
        )
        self.status.recycle_count = safety.recycle_count

        if safety.lockout:
            self.status.state = ControllerState.ARC_LOCKOUT
            self.status.rf_enabled = False
            return self.status

        if safety.rf_inhibited:
            # In recycle sequence — RF off, gap to max
            self.status.rf_enabled = False
            self.status.state = ControllerState.RECYCLE
            self.status.electrode_gap_mm = self._machine.electrode_gap_max_m * 1000.0
            return self.status

        # If we were in recycle and safety cleared, restore RF
        # but keep the current gap (don't reset to recipe setpoint —
        # MRH may have opened it, and resetting would cause another trip)
        if self.status.state == ControllerState.RECYCLE:
            self.status.state = ControllerState.RUNNING
            self.status.rf_enabled = recipe.rf_power_enabled

        # ── 2. MRH / MRL GAP CONTROL ─────────────────────────────────
        # Asymmetric gap control with proportional MRH response.
        #
        # Three regimes derived from Run#1 PLC analysis:
        #
        #   MRH (Ia > MRH):  Open gap proportionally to overcurrent.
        #     The further Ia exceeds MRH, the faster the gap opens.
        #     gain = 1 + (Ia - MRH) / MRH  →  e.g. at Ia=2.0, MRH=1.7:
        #     gain = 1.18, rate = 0.19 × 1.18 = 0.22 mm/s.
        #
        #   Normal band (MRL < Ia < MRH):  Very slow drift toward
        #     setpoint.  The PLC holds the elevated gap for ~1200 s
        #     during steady-state processing (87 → 86.7 mm in 600 s).
        #     drift_rate = gap_rate × _GAP_DRIFT_RATIO (0.5%).
        #
        #   MRL (Ia < MRL):  Depends on batch state:
        #     - Batch exhausted: close toward setpoint at moderate rate
        #       (gap_rate × _GAP_CLOSE_RATIO = 15% of open rate).
        #     - Continuous: hold position (existing MRL_STOP).
        self.status.mrh_active = anode_current_a > recipe.mrh_amps
        self.status.mrl_active = anode_current_a < recipe.mrl_amps

        gap_setpoint_mm = recipe.electrode_gap_mm
        gap_max_mm = self._machine.electrode_gap_max_m * 1000.0
        gap_min_mm = self._machine.electrode_gap_min_m * 1000.0

        if self.status.mrh_active:
            # Overcurrent: proportional gap opening.
            # Rate scales with how far Ia exceeds MRH, giving faster
            # response to larger overcurrents while maintaining the
            # base rate at the MRH threshold.
            overshoot = max(0.0, anode_current_a - recipe.mrh_amps)
            gain = 1.0 + overshoot / max(recipe.mrh_amps, 0.1)
            self.status.electrode_gap_mm += self.gap_adjust_rate_mm_s * gain * dt
            self.status.electrode_gap_mm = min(
                self.status.electrode_gap_mm, gap_max_mm,
            )
            self.status.state = ControllerState.MRH_TRIP
        elif self.status.mrl_active:
            # Undercurrent behavior depends on batch mode:
            # - Batch exhausted (empty belt): return gap to setpoint
            #   at the moderate close rate (15% of open rate).
            # - Continuous operation: hold position (MRL_STOP).
            if self._batch_exhausted:
                close_rate = self.gap_adjust_rate_mm_s * self._GAP_CLOSE_RATIO
                if self.status.electrode_gap_mm > gap_setpoint_mm + 0.5:
                    self.status.electrode_gap_mm -= close_rate * dt
                    self.status.electrode_gap_mm = max(
                        self.status.electrode_gap_mm, gap_setpoint_mm,
                    )
                if self.status.state in (
                    ControllerState.MRH_TRIP,
                    ControllerState.MRL_STOP,
                ):
                    self.status.state = ControllerState.RUNNING
            else:
                self.status.state = ControllerState.MRL_STOP
        else:
            # Normal band: very slow drift toward setpoint.
            # The PLC holds the elevated gap during steady-state
            # processing — gap barely moves while material is present.
            drift_rate = self.gap_adjust_rate_mm_s * self._GAP_DRIFT_RATIO
            if self.status.electrode_gap_mm > gap_setpoint_mm + 0.5:
                self.status.electrode_gap_mm -= drift_rate * dt
                self.status.electrode_gap_mm = max(
                    self.status.electrode_gap_mm, gap_setpoint_mm,
                )
            elif self.status.electrode_gap_mm < gap_setpoint_mm - 0.5:
                self.status.electrode_gap_mm += drift_rate * dt
                self.status.electrode_gap_mm = min(
                    self.status.electrode_gap_mm, gap_setpoint_mm,
                )

            if self.status.state in (
                ControllerState.MRH_TRIP,
                ControllerState.MRL_STOP,
            ):
                self.status.state = ControllerState.RUNNING

        # ── 3. DEBOUNCE ───────────────────────────────────────────────
        # Don't alter gap unless debounce timer has elapsed
        if self._gap_command_pending:
            self._debounce_timer += dt
            if self._debounce_timer >= self._machine.electrode_debounce_s:
                self._gap_command_pending = False
                self._debounce_timer = 0.0

        # ── 4. TEMPERATURE CONTROL (optional auto mode) ───────────────
        if (
            recipe.temp_control_enabled
            and self.status.state == ControllerState.RUNNING
        ):
            self._temperature_control_step(dt, T_outfeed_c, recipe)

        return self.status

    def emergency_stop(self):
        """Trigger emergency stop — RF off, belt stop, lockout."""
        self.status.state = ControllerState.EMERGENCY_STOP
        self.status.rf_enabled = False
        self.status.belt_speed_m_per_min = 0.0

    # ------------------------------------------------------------------
    # Temperature control (§8.2)
    # ------------------------------------------------------------------

    def _temperature_control_step(
        self,
        dt: float,
        T_outfeed_c: float,
        recipe: Recipe,
    ):
        """Automatic temperature control using outfeed sensor average.

        Algorithm (from GP-15 manual Screen 14):
        1. If T_avg > T_setpoint: increase gap (reduce power).
           If still too hot after envelope_time: increase belt speed.
        2. If T_avg < T_setpoint: decrease gap (increase power).
           If still too cold after envelope_time: decrease belt speed.
        3. Wait envelope_time before next correction.
        """
        if (self._sim_time - self._temp_last_correction_time
                < recipe.temp_envelope_time_s):
            return  # Wait for envelope time

        self._temp_last_correction_time = self._sim_time
        gap_min_mm = self._machine.electrode_gap_min_m * 1000.0
        gap_max_mm = self._machine.electrode_gap_max_m * 1000.0
        T_sp = recipe.temp_setpoint_c
        delta_T = T_outfeed_c - T_sp

        if delta_T > 1.0:
            # Too hot — increase gap first
            if not self._temp_gap_correction_applied:
                self.status.electrode_gap_mm = min(
                    self.status.electrode_gap_mm + self._TEMP_GAP_STEP_MM,
                    gap_max_mm,
                )
                self._temp_gap_correction_applied = True
            else:
                # Gap already adjusted — increase belt speed
                self.status.belt_speed_m_per_min = min(
                    self.status.belt_speed_m_per_min + self._TEMP_SPEED_STEP,
                    self._machine.belt_speed_max_m_per_min,
                )
                self._temp_gap_correction_applied = False

        elif delta_T < -1.0:
            # Too cold — decrease gap first
            if not self._temp_gap_correction_applied:
                self.status.electrode_gap_mm = max(
                    self.status.electrode_gap_mm - self._TEMP_GAP_STEP_MM,
                    gap_min_mm,
                )
                self._temp_gap_correction_applied = True
            else:
                # Gap already adjusted — decrease belt speed
                self.status.belt_speed_m_per_min = max(
                    self.status.belt_speed_m_per_min - self._TEMP_SPEED_STEP,
                    self._machine.belt_speed_min_m_per_min,
                )
                self._temp_gap_correction_applied = False

        else:
            # Within tolerance
            self._temp_gap_correction_applied = False
