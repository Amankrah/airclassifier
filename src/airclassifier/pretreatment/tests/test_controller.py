"""
Tests for the PLC controller and safety logic.

Validates (engineering guide §8):
- MRH trip: RF off when anode current > mrh_amps
- MRL stop: electrode drive stops when current < mrl_amps
- Recycle: RF restarts after MRH trip (up to 4 times)
- Lockout: after 4 recycles, system locks out
- Temperature control: gap adjusts to maintain setpoint
- Safety monitor: arc detection, thermal fuse, recycle counter
"""

import pytest

from airclassifier.pretreatment.config import MachineConfig, Recipe
from airclassifier.pretreatment.control.controller import (
    GP15Controller,
    ControllerState,
)
from airclassifier.pretreatment.control.safety import (
    SafetyMonitor,
    SafetyEvent,
)


# ── Safety Monitor Tests ─────────────────────────────────────────────

class TestSafetyMonitor:
    """SafetyMonitor unit tests."""

    def test_no_fault_clears(self):
        """Normal conditions should report NONE with no RF inhibit."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=2.0)
        status = sm.check(anode_current_a=1.5, mrh_amps=2.6, dt=0.5)
        assert status.event == SafetyEvent.NONE
        assert not status.rf_inhibited
        assert not status.lockout

    def test_mrh_triggers_recycle(self):
        """Overcurrent should trigger MRH trip and inhibit RF."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=2.0)
        status = sm.check(anode_current_a=3.0, mrh_amps=2.6, dt=0.5)
        assert status.event == SafetyEvent.MRH_TRIP
        assert status.rf_inhibited
        assert status.recycle_count == 1

    def test_recycle_restores_rf_after_delay(self):
        """After the restart delay, RF should be restored."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=2.0)
        # Trigger fault
        sm.check(anode_current_a=3.0, mrh_amps=2.6, dt=0.5)
        assert sm.status.rf_inhibited

        # Advance through delay (4 x 0.5s = 2.0s)
        for _ in range(4):
            sm.check(anode_current_a=1.0, mrh_amps=2.6, dt=0.5)

        assert not sm.status.rf_inhibited
        assert sm.status.recycle_count == 1

    def test_lockout_after_max_recycles(self):
        """After max_recycles faults, system should lock out."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=0.1)

        for i in range(4):
            # Trigger fault
            sm.check(anode_current_a=3.0, mrh_amps=2.6, dt=0.5)
            if not sm.status.lockout:
                # Clear the recycle delay
                for _ in range(5):
                    sm.check(anode_current_a=1.0, mrh_amps=2.6, dt=0.1)

        assert sm.status.lockout
        assert sm.status.event == SafetyEvent.RECYCLE_LIMIT

    def test_arc_detection(self):
        """High E-field should trigger arc detection."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=2.0)
        status = sm.check(
            anode_current_a=1.0,
            mrh_amps=2.6,
            e_field_max_v_per_m=5.0e6,
            arc_threshold_v_per_m=3.0e6,
            dt=0.5,
        )
        assert status.event == SafetyEvent.ARC_DETECTED
        assert status.rf_inhibited

    def test_thermal_fuse_lockout(self):
        """Valve over-temperature should cause immediate lockout."""
        sm = SafetyMonitor(max_recycles=4, restart_delay_s=2.0)
        status = sm.check(
            anode_current_a=1.0,
            mrh_amps=2.6,
            valve_temp_c=150.0,
            dt=0.5,
        )
        assert status.event == SafetyEvent.THERMAL_FUSE
        assert status.lockout

    def test_reset_clears_lockout(self):
        """Operator reset should clear lockout."""
        sm = SafetyMonitor(max_recycles=1, restart_delay_s=0.1)
        sm.check(anode_current_a=3.0, mrh_amps=2.6, dt=0.5)
        assert sm.status.lockout
        sm.reset()
        assert not sm.status.lockout
        assert sm.status.recycle_count == 0


# ── Controller Tests ─────────────────────────────────────────────────

class TestMRHProtection:
    """Meter Relay High (overcurrent) protection."""

    def test_mrh_trips_rf(self):
        """RF should turn off when anode current exceeds MRH threshold."""
        config = MachineConfig()
        ctrl = GP15Controller(config)
        recipe = Recipe(name="test", recipe_number=1,
                        electrode_gap_mm=80, belt_speed_m_per_min=0.5,
                        mrh_amps=2.6)
        ctrl.load_recipe(recipe)
        ctrl.start()

        # Normal operation
        status = ctrl.step(dt=0.5, anode_current_a=2.0, rf_power_kw=10.0,
                          T_outfeed_c=50.0)
        assert status.rf_enabled

        # Trigger MRH via safety monitor (overcurrent)
        # The safety monitor triggers recycle, inhibiting RF
        status = ctrl.step(dt=0.5, anode_current_a=3.0, rf_power_kw=15.0,
                          T_outfeed_c=50.0)
        assert not status.rf_enabled

    def test_recycle_after_mrh(self):
        """System should attempt to restart after MRH trip."""
        config = MachineConfig(max_recycle_restarts=4, restart_delay_s=1.0)
        ctrl = GP15Controller(config)
        recipe = Recipe(name="test", recipe_number=1,
                        electrode_gap_mm=80, belt_speed_m_per_min=0.5,
                        mrh_amps=2.6, mrl_amps=0.5)
        ctrl.load_recipe(recipe)
        ctrl.start()

        # Trigger MRH
        ctrl.step(dt=0.5, anode_current_a=3.0, rf_power_kw=15.0,
                  T_outfeed_c=50.0)
        assert not ctrl.status.rf_enabled

        # Wait out the restart delay (2 x 0.5s = 1.0s)
        # Use current above MRL so we don't enter MRL_STOP
        for _ in range(2):
            ctrl.step(dt=0.5, anode_current_a=1.5, rf_power_kw=5.0,
                      T_outfeed_c=50.0)

        # RF should be restored
        assert ctrl.status.rf_enabled
        assert ctrl.status.state == ControllerState.RUNNING

    def test_lockout_after_max_recycles(self):
        """System should lock out after max_recycle_restarts."""
        config = MachineConfig(max_recycle_restarts=4, restart_delay_s=0.1)
        ctrl = GP15Controller(config)
        recipe = Recipe(name="test", recipe_number=1,
                        electrode_gap_mm=80, belt_speed_m_per_min=0.5,
                        mrh_amps=2.6)
        ctrl.load_recipe(recipe)
        ctrl.start()

        for _ in range(4):
            # Trigger overcurrent
            ctrl.step(dt=0.5, anode_current_a=3.0, rf_power_kw=15.0,
                      T_outfeed_c=50.0)
            if ctrl.status.state == ControllerState.ARC_LOCKOUT:
                break
            # Clear recycle delay
            for _ in range(5):
                ctrl.step(dt=0.1, anode_current_a=1.0, rf_power_kw=0.0,
                          T_outfeed_c=50.0)

        assert ctrl.status.state == ControllerState.ARC_LOCKOUT
        assert not ctrl.status.rf_enabled


class TestRecipeLoading:
    """Recipe load and setpoint application."""

    def test_load_recipe(self):
        """Controller should apply recipe setpoints."""
        config = MachineConfig()
        ctrl = GP15Controller(config)
        recipe = Recipe(name="test", recipe_number=1,
                        electrode_gap_mm=80, belt_speed_m_per_min=0.5)
        ctrl.load_recipe(recipe)
        assert ctrl.status.electrode_gap_mm == 80.0
        assert ctrl.status.belt_speed_m_per_min == 0.5
        assert ctrl.status.state == ControllerState.READY


class TestTemperatureControl:
    """Automatic temperature control mode."""

    def test_hot_outfeed_increases_gap(self):
        """When T > setpoint, gap should increase (reduce power)."""
        config = MachineConfig()
        ctrl = GP15Controller(config)
        recipe = Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=80, belt_speed_m_per_min=0.5,
            mrh_amps=2.6, mrl_amps=0.5,
            temp_control_enabled=True,
            temp_setpoint_c=60.0,
            temp_envelope_time_s=0.0,  # immediate correction
        )
        ctrl.load_recipe(recipe)
        ctrl.start()
        gap_before = ctrl.status.electrode_gap_mm

        # Run with hot outfeed (current above MRL)
        ctrl.step(dt=1.0, anode_current_a=1.5, rf_power_kw=8.0,
                  T_outfeed_c=70.0)

        assert ctrl.status.electrode_gap_mm > gap_before

    def test_cold_outfeed_decreases_gap(self):
        """When T < setpoint, gap should decrease (increase power)."""
        config = MachineConfig()
        ctrl = GP15Controller(config)
        recipe = Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=80, belt_speed_m_per_min=0.5,
            mrh_amps=2.6, mrl_amps=0.5,
            temp_control_enabled=True,
            temp_setpoint_c=60.0,
            temp_envelope_time_s=0.0,
        )
        ctrl.load_recipe(recipe)
        ctrl.start()
        gap_before = ctrl.status.electrode_gap_mm

        ctrl.step(dt=1.0, anode_current_a=1.5, rf_power_kw=4.0,
                  T_outfeed_c=50.0)

        assert ctrl.status.electrode_gap_mm < gap_before
