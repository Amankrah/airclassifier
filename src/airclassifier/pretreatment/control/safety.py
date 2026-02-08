"""
Safety Logic
============

Arc detection, recycle logic, and lockout conditions for the GP-15.

The GP-15 has several safety systems (engineering guide §8.3):
- MRH (Meter Relay High): Overcurrent protection → RF off, recycle
- Arc detection: E-field exceeds breakdown → immediate RF off
- Recycle limit: Max 4 restarts, then lockout
- Thermal limit: Valve thermal fuse at ~145 °C
- Ambient limit: >40 °C ambient → warning
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Monitors GP-15 safety conditions each timestep.

    Implements the recycle sequence from the engineering guide §8.3::

        1. Fault detected (MRH trip or arc)
        2. RF power off immediately
        3. Wait restart_delay (typically 2 s)
        4. RF restores
        5. Increment recycle counter
        6. If counter >= max_restarts (4): LOCKOUT
        7. Counter resets after monitor_interval (from belt speed)

    Args:
        max_recycles: Maximum restart attempts before lockout.
        restart_delay_s: Delay between RF off and RF restore [s].
        monitor_interval_s: Time window for recycle counter reset [s].
            If 0, computed from belt speed via
            :meth:`update_monitor_interval`.
    """

    def __init__(
        self,
        max_recycles: int = 4,
        restart_delay_s: float = 2.0,
        monitor_interval_s: float = 0.0,
    ):
        self._max_recycles = max_recycles
        self._restart_delay_s = restart_delay_s
        self._monitor_interval_s = monitor_interval_s
        self._recycle_count = 0
        self._in_recycle = False
        self._recycle_timer = 0.0
        self._monitor_timer = 0.0
        self.status = SafetyStatus()

    def update_monitor_interval(
        self,
        belt_speed_m_per_min: float,
        oven_length_m: float = 1.5,
    ):
        """Compute recycle counter reset interval from belt speed.

        Uses one full belt transit of the oven as the monitoring window.
        """
        v = belt_speed_m_per_min / 60.0  # m/s
        if v > 0:
            self._monitor_interval_s = oven_length_m / v
        else:
            self._monitor_interval_s = 300.0  # 5 min fallback

    def check(
        self,
        anode_current_a: float,
        mrh_amps: float,
        reflected_power_kw: float = 0.0,
        valve_temp_c: float = 25.0,
        ambient_temp_c: float = 22.0,
        e_field_max_v_per_m: float = 0.0,
        arc_threshold_v_per_m: float = 3.0e6,
        dt: float = 0.0,
    ) -> SafetyStatus:
        """Check all safety conditions and manage recycle state.

        Args:
            anode_current_a: Measured anode current [A].
            mrh_amps: MRH threshold from active recipe [A].
            reflected_power_kw: Reflected RF power (arc indicator) [kW].
            valve_temp_c: Triode valve temperature [°C].
            ambient_temp_c: Oven ambient temperature [°C].
            e_field_max_v_per_m: Peak E-field magnitude [V/m].
            arc_threshold_v_per_m: Breakdown threshold [V/m].
            dt: Current timestep [s] (for timer advancement).

        Returns:
            Updated :class:`SafetyStatus`.
        """
        # If already locked out, stay locked out
        if self.status.lockout:
            return self.status

        # --- Advance recycle timer if in recycle ---
        if self._in_recycle:
            self._recycle_timer += dt
            if self._recycle_timer >= self._restart_delay_s:
                # Restart delay elapsed — restore RF
                self._in_recycle = False
                self._recycle_timer = 0.0
                self.status.rf_inhibited = False
                self.status.event = SafetyEvent.NONE
                self.status.message = (
                    f"Recycle {self._recycle_count}/{self._max_recycles} complete"
                )
            return self.status

        # --- Monitor interval: reset recycle counter ---
        if self._monitor_interval_s > 0 and self._recycle_count > 0:
            self._monitor_timer += dt
            if self._monitor_timer >= self._monitor_interval_s:
                self._recycle_count = 0
                self._monitor_timer = 0.0

        # --- Check 1: Thermal fuse (valve temperature) ---
        if valve_temp_c >= 145.0:
            self.status.event = SafetyEvent.THERMAL_FUSE
            self.status.rf_inhibited = True
            self.status.lockout = True
            self.status.message = (
                f"Thermal fuse: valve at {valve_temp_c:.0f} °C >= 145 °C"
            )
            return self.status

        # --- Check 2: MRH (overcurrent) ---
        if anode_current_a > mrh_amps:
            return self._trigger_recycle(
                SafetyEvent.MRH_TRIP,
                f"MRH trip: Ia={anode_current_a:.2f} A > {mrh_amps:.2f} A",
            )

        # --- Check 3: Arc detection ---
        if e_field_max_v_per_m > arc_threshold_v_per_m or reflected_power_kw > 5.0:
            return self._trigger_recycle(
                SafetyEvent.ARC_DETECTED,
                f"Arc detected: E={e_field_max_v_per_m:.0f} V/m "
                f"(threshold {arc_threshold_v_per_m:.0f})",
            )

        # --- Check 4: Ambient temperature warning ---
        if ambient_temp_c > 40.0:
            self.status.event = SafetyEvent.AMBIENT_HIGH
            self.status.message = (
                f"Ambient high: {ambient_temp_c:.1f} °C > 40 °C"
            )
            # Warning only — does not inhibit RF
            return self.status

        # --- All clear ---
        self.status.event = SafetyEvent.NONE
        self.status.rf_inhibited = False
        self.status.message = ""
        self.status.recycle_count = self._recycle_count
        return self.status

    def reset(self):
        """Operator-initiated reset (clears lockout)."""
        self._recycle_count = 0
        self._in_recycle = False
        self._recycle_timer = 0.0
        self._monitor_timer = 0.0
        self.status = SafetyStatus()

    # ------------------------------------------------------------------

    def _trigger_recycle(self, event: SafetyEvent, message: str) -> SafetyStatus:
        """Start a recycle sequence or lock out if max reached."""
        self._recycle_count += 1
        self._monitor_timer = 0.0  # reset monitor window

        if self._recycle_count >= self._max_recycles:
            self.status.event = SafetyEvent.RECYCLE_LIMIT
            self.status.rf_inhibited = True
            self.status.lockout = True
            self.status.recycle_count = self._recycle_count
            self.status.message = (
                f"LOCKOUT: {self._recycle_count} recycles reached — "
                f"operator intervention required"
            )
        else:
            self._in_recycle = True
            self._recycle_timer = 0.0
            self.status.event = event
            self.status.rf_inhibited = True
            self.status.lockout = False
            self.status.recycle_count = self._recycle_count
            self.status.message = message

        return self.status
