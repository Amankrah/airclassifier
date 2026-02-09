"""
Integration tests for the full pretreatment simulation.

Validates:
- GP15Simulator public API (§7.1)
- End-to-end coupled simulation
- OutletState for pipeline integration (§9.1)
- Manual example validation (§10.1)
- Mesh generation for visualization
"""

import numpy as np
import pytest

from airclassifier.pretreatment.config import MachineConfig, MaterialProperties, Recipe
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.physics.coupling import CoupledSimulator, OutletState
from airclassifier.pretreatment.geometry.oven import OvenGeometry, OvenGeometryParams


# ── Helpers ──────────────────────────────────────────────────────────

def _make_gp15(**kwargs):
    """Create a GP15Simulator with small grid for fast tests."""
    config = MachineConfig()
    material = MaterialProperties(
        initial_moisture_wb=0.10,
        target_moisture_wb=0.03,
        initial_temperature_c=22.0,
        bed_depth_m=0.04,
        bed_porosity=0.40,
    )
    return GP15Simulator(
        config=config,
        material=material,
        enable_controller=False,
        enable_corrections=False,
        use_tvd=False,
        **kwargs,
    )


def _make_coupled(nx=16, ny=8, nz=12):
    """Create a small coupled simulator for direct testing."""
    machine = MachineConfig()
    material = MaterialProperties(
        initial_moisture_wb=0.10,
        target_moisture_wb=0.03,
        initial_temperature_c=22.0,
        bed_depth_m=0.04,
        bed_porosity=0.40,
    )

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

    gap_m = 0.10
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


# ── GP15Simulator public API tests ───────────────────────────────────

class TestGP15Simulator:
    """GP15Simulator public API (§7.1)."""

    def test_create_and_run(self):
        """Simulator should create and run without error."""
        gp15 = _make_gp15()
        gp15.load_recipe(Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=100, belt_speed_m_per_min=0.5,
        ))
        result = gp15.run(duration_s=5.0)
        assert result.duration_s > 0
        assert result.T_final is not None
        assert result.M_final is not None

    def test_outlet_conditions(self):
        """get_outlet_conditions() should return valid OutletState."""
        gp15 = _make_gp15()
        gp15.load_recipe(Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=100, belt_speed_m_per_min=0.5,
        ))
        gp15.run(duration_s=5.0)
        outlet = gp15.get_outlet_conditions()

        assert isinstance(outlet, OutletState)
        assert outlet.avg_temperature_c > 0
        assert 0 <= outlet.avg_moisture_wb <= 1.0
        assert outlet.throughput_kg_per_hr > 0
        assert outlet.residence_time_s > 0

    def test_get_mesh(self):
        """get_mesh() returns assembled machine geometry (same as build_gp15_machine_meshes)."""
        gp15 = _make_gp15()
        gp15.load_recipe(Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=100, belt_speed_m_per_min=0.5,
        ))
        meshes = gp15.get_mesh()

        assert "oven" in meshes
        assert "upper_electrode" in meshes
        assert "lower_electrode" in meshes
        assert "belt" in meshes
        assert "material_bed" in meshes
        # Envelope parts from geometry.machine
        assert "housing" in meshes
        assert "legs" in meshes

        # Each structural mesh has vertices, triangles, color, opacity
        for name in ("oven", "upper_electrode", "lower_electrode", "belt", "material_bed"):
            assert meshes[name]["vertices"].shape[1] == 3
            assert meshes[name]["triangles"].shape[1] == 3
            assert "color" in meshes[name]
            assert "opacity" in meshes[name]

    def test_get_mesh_with_fields(self):
        """After running, get_mesh() should include field data."""
        gp15 = _make_gp15()
        gp15.load_recipe(Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=100, belt_speed_m_per_min=0.5,
        ))
        gp15.run(duration_s=2.0)
        meshes = gp15.get_mesh()

        assert "fields" in meshes
        f = meshes["fields"]
        assert "temperature" in f
        assert "moisture" in f
        assert f["grid_shape"] is not None

    def test_step_interactive(self):
        """step() should work for interactive/real-time use."""
        gp15 = _make_gp15()
        gp15.load_recipe(Recipe(
            name="test", recipe_number=1,
            electrode_gap_mm=100, belt_speed_m_per_min=0.5,
        ))
        state = gp15.step(dt=0.5)
        assert state.time_s > 0
        assert state.rf_power_kw >= 0


# ── CoupledSimulator direct tests ────────────────────────────────────

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

        assert state.T_mean_c > T_start

    def test_moisture_decreases(self):
        """Material moisture should decrease after several steps."""
        sim, recipe, mat = _make_coupled()
        M_start = mat.initial_moisture_wb

        for _ in range(50):
            state = sim.step(dt=0.5, recipe=recipe)

        assert state.M_mean_wb < M_start

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
        assert state.rf_power_kw > 0.0

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


# ── Validation (§10.1) ──────────────────────────────────────────────

class TestManualValidation:
    """Reproduce the GP-15 manual worked example (§10.1).

    Given:
        Throughput       = 600 kg/hr
        Inlet moisture   = 4% (wet basis)
        Outlet moisture  = 3% (wet basis)
        Water removal    = 1 kg/kWh (high surface-to-volume product)

    Water removed = 600 × (0.04 - 0.03) / (1 - 0.03) = 6.19 kg/hr
    RF power to material = 6.19 / 1.0 = 6.19 kW
    Generator power = ~11 kW (at ~56% oscillator efficiency)
    """

    def test_water_removal_calculation(self):
        """Verify the water removal arithmetic from the manual."""
        throughput_kg_hr = 600.0
        M_in = 0.04
        M_out = 0.03

        water_removed = throughput_kg_hr * (M_in - M_out) / (1 - M_out)
        assert water_removed == pytest.approx(6.19, abs=0.01)

    def test_rf_power_requirement(self):
        """RF power to material should be ~6.2 kW for 1 kg/kWh efficiency."""
        water_rate_kg_hr = 6.19
        efficiency_kg_per_kwh = 1.0
        P_rf_kw = water_rate_kg_hr / efficiency_kg_per_kwh
        assert P_rf_kw == pytest.approx(6.19, abs=0.1)

    def test_generator_power_with_efficiency(self):
        """Generator power at 56% efficiency should be ~11 kW."""
        P_rf_kw = 6.19
        oscillator_efficiency = 0.56
        P_gen_kw = P_rf_kw / oscillator_efficiency
        assert P_gen_kw == pytest.approx(11.05, abs=0.5)

    def test_oscillator_efficiency_constant(self):
        """The default oscillator efficiency should match the manual."""
        gp15 = _make_gp15()
        assert gp15._sim._oscillator_efficiency == pytest.approx(0.56)
