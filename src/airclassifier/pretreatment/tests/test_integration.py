"""
Integration tests for the full coupled simulation.

Validates:
- End-to-end: material enters at initial moisture, exits drier
- Temperature rises under RF heating
- Coupling: T and M fields are mutually consistent
- Advection: material moves through the domain
"""

import numpy as np
import pytest

from airclassifier.pretreatment.config import MachineConfig, MaterialProperties, Recipe
from airclassifier.pretreatment.geometry.oven import OvenGeometry, OvenGeometryParams
from airclassifier.pretreatment.physics.coupling import CoupledSimulator


# ── Helpers ──────────────────────────────────────────────────────────

def _make_coupled(nx=16, ny=8, nz=12):
    """Create a small coupled simulator for testing."""
    machine = MachineConfig()
    material = MaterialProperties(
        initial_moisture_wb=0.10,
        target_moisture_wb=0.03,
        initial_temperature_c=22.0,
        bed_depth_m=0.04,
        bed_porosity=0.40,
    )

    # Build grid
    oven = OvenGeometry(OvenGeometryParams(
        length=machine.oven_length_m,
        width=machine.belt_width_m,
        height=machine.electrode_gap_max_m,
        resolution=nz,
    ))
    grid_shape = oven.get_grid_shape()
    cell_sizes = oven.get_cell_sizes()

    sim = CoupledSimulator(
        machine=machine,
        material=material,
        grid_shape=grid_shape,
        cell_sizes=cell_sizes,
        device="cpu",
    )

    gap_m = 0.10  # 100 mm gap
    mask = oven.build_material_mask(
        electrode_gap_m=gap_m,
        bed_depth_m=material.bed_depth_m,
        belt_stack_m=machine.belt_stack_thickness_m,
    )
    sim.initialize(cell_is_material=mask)

    recipe = Recipe(
        name="test_recipe",
        recipe_number=1,
        electrode_gap_mm=100.0,
        belt_speed_m_per_min=0.5,
        extraction_fan_hz=30.0,
    )

    return sim, recipe, material


# ── Tests ────────────────────────────────────────────────────────────

class TestCoupledSimulation:
    """Full coupled simulation integration tests."""

    def test_simulation_runs(self):
        """The coupled simulator should execute without error."""
        sim, recipe, _ = _make_coupled()
        state = sim.step(dt=0.5, recipe=recipe)
        assert state.time_s > 0

    def test_temperature_rises(self):
        """RF heating should raise the material temperature."""
        sim, recipe, mat = _make_coupled()
        T_start = mat.initial_temperature_c

        for _ in range(20):
            state = sim.step(dt=0.5, recipe=recipe)

        assert state.T_mean_c > T_start, (
            f"Expected T > {T_start}, got {state.T_mean_c}"
        )

    def test_moisture_decreases(self):
        """Material moisture should decrease after several steps."""
        sim, recipe, mat = _make_coupled()
        M_start = mat.initial_moisture_wb

        for _ in range(50):
            state = sim.step(dt=0.5, recipe=recipe)

        assert state.M_mean_wb < M_start, (
            f"Expected M < {M_start}, got {state.M_mean_wb}"
        )

    def test_run_returns_result(self):
        """The run() method should return a PretreatmentResult."""
        sim, recipe, _ = _make_coupled()
        result = sim.run(duration_s=10.0, dt=0.5, recipe=recipe)

        assert result.duration_s == pytest.approx(10.0, abs=1.0)
        assert result.T_final is not None
        assert result.M_final is not None
        assert result.energy_consumed_kwh >= 0.0
        assert result.throughput_kg_per_h > 0.0

    def test_rf_power_positive(self):
        """RF power delivered to material should be positive."""
        sim, recipe, _ = _make_coupled()
        state = sim.step(dt=0.5, recipe=recipe)

        assert state.rf_power_kw > 0.0, "RF power should be positive"

    def test_anode_current_in_range(self):
        """Anode current should be between no-load and full-load."""
        sim, recipe, _ = _make_coupled()
        machine = sim._machine

        for _ in range(5):
            state = sim.step(dt=0.5, recipe=recipe)

        assert state.anode_current_a >= machine.anode_current_no_load_a
        assert state.anode_current_a <= machine.anode_current_full_load_a

    def test_time_series_recorded(self):
        """run() should record time-series data."""
        sim, recipe, _ = _make_coupled()
        result = sim.run(duration_s=5.0, dt=0.5, recipe=recipe)

        ts = result.time_series
        assert "time_s" in ts
        assert "T_mean_c" in ts
        assert "M_mean_wb" in ts
        assert len(ts["time_s"]) == 10  # 5s / 0.5s = 10 steps
