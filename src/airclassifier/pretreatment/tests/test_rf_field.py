"""
Tests for the RF electric field solver.

Validates:
- Uniform parallel-plate approximation: E = V/gap
- Series-capacitor voltage division across layered stack
- Power density P_v = 2*pi*f*eps_0*eps'' * |E|^2
- Total RF power matches integral of P_v over material volume
"""

import numpy as np
import pytest

from airclassifier.pretreatment.config import MachineConfig, MaterialProperties
from airclassifier.pretreatment.geometry.oven import OvenGeometry, OvenGeometryParams
from airclassifier.pretreatment.physics.rf_field import RFFieldSolver
from airclassifier.pretreatment.kernels.dielectric_heating import (
    TWO_PI_F_EPS0,
    compute_power_density_np,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_solver(nx=20, ny=10, nz=15):
    """Create a solver with known grid shape."""
    machine = MachineConfig()
    dx = machine.oven_length_m / nx
    dy = machine.electrode_gap_max_m / ny
    dz = machine.belt_width_m / nz
    return RFFieldSolver(
        grid_shape=(nx, ny, nz),
        cell_sizes=(dx, dy, dz),
        machine=machine,
    ), machine


# ── Phase 1: uniform parallel-plate model ────────────────────────────

class TestUniformField:
    """Phase 1: uniform parallel-plate model."""

    def test_uniform_field_no_bed_no_belt(self):
        """With only air between plates, E = V/gap everywhere."""
        solver, machine = _make_solver()
        gap = 0.10  # 100 mm
        V_kv = 5.0  # 5 kV

        E2 = solver.solve(
            electrode_gap_m=gap,
            voltage_kv=V_kv,
            bed_depth_m=0.0,
            belt_stack_m=0.0,
            eps_bed_avg=1.0,
        )
        E_expected = V_kv * 1000.0 / gap  # V/m
        E2_expected = E_expected ** 2

        # All cells should be air — uniform field
        np.testing.assert_allclose(E2, E2_expected, rtol=1e-4)

    def test_series_capacitor_voltage_division(self):
        """E in the material bed matches the series-capacitor formula."""
        solver, machine = _make_solver()
        gap = 0.10
        V_kv = 9.0
        V_total = V_kv * 1000.0

        d_belt = 0.0035
        d_bed = 0.05
        d_air = gap - d_belt - d_bed
        eps_belt = 2.1
        eps_bed = 5.0

        # Analytic: E_bed = V_total / (eps_bed * sum(d_j / eps_j))
        cap_sum = d_air / 1.0 + d_bed / eps_bed + d_belt / eps_belt
        E_bed_expected = V_total / (eps_bed * cap_sum)

        solver.solve(
            electrode_gap_m=gap,
            voltage_kv=V_kv,
            bed_depth_m=d_bed,
            belt_stack_m=d_belt,
            eps_bed_avg=eps_bed,
        )

        assert solver.get_bed_field_strength() == pytest.approx(
            E_bed_expected, rel=1e-4
        )

    def test_field_belt_less_than_air(self):
        """Belt (eps'=2.1) should have a weaker field than air (eps'=1)."""
        solver, machine = _make_solver()
        gap = 0.10
        V_kv = 5.0

        solver.solve(
            electrode_gap_m=gap,
            voltage_kv=V_kv,
            bed_depth_m=0.04,
            belt_stack_m=0.0035,
            eps_bed_avg=5.0,
        )

        assert solver._E_belt < solver._E_air

    def test_power_density_magnitude(self):
        """P_v = 2*pi*f*eps_0*eps''*|E|^2 at known conditions."""
        # Single cell computation
        eps_loss = 1.0
        E_sq = 1e8  # (10 kV/m)^2
        P_v = TWO_PI_F_EPS0 * eps_loss * E_sq

        # Manual: 2*pi*27.12e6*8.854e-12 = ~1.5098e-3
        expected = 1.5098e-3 * 1.0 * 1e8
        assert P_v == pytest.approx(expected, rel=1e-3)

    def test_total_power_matches_integral(self):
        """Integral of P_v over the material volume = RF power [W]."""
        machine = MachineConfig()
        gap = 0.10
        V_kv = 9.0
        d_bed = 0.05
        d_belt = machine.belt_stack_thickness_m

        # Build geometry first so solver uses the same grid shape
        oven = OvenGeometry(OvenGeometryParams(
            length=machine.oven_length_m,
            width=machine.belt_width_m,
            height=machine.electrode_gap_max_m,
            resolution=20,
        ))
        grid_shape = oven.get_grid_shape()
        cell_sizes = oven.get_cell_sizes()

        solver = RFFieldSolver(grid_shape, cell_sizes, machine)
        mask = oven.build_material_mask(gap, d_bed, d_belt)

        # Give the material uniform eps''
        eps_loss = np.where(mask == 1, 1.5, 0.0).astype(np.float32)
        eps_real = np.where(mask == 1, 5.0, 1.0).astype(np.float32)

        solver.solve(
            electrode_gap_m=gap,
            voltage_kv=V_kv,
            eps_real=eps_real,
            cell_is_material=mask,
            bed_depth_m=d_bed,
            belt_stack_m=d_belt,
        )

        P_v = compute_power_density_np(solver.e_field_sq, eps_loss)
        cell_vol = cell_sizes[0] * cell_sizes[1] * cell_sizes[2]

        total_power_integral = float(np.sum(P_v[mask == 1]) * cell_vol)
        total_power_method = solver.compute_total_rf_power(eps_loss, mask, cell_vol)

        assert total_power_integral == pytest.approx(total_power_method, rel=1e-6)
        assert total_power_method > 0.0, "Should have non-zero RF power"


class TestVariablePermittivity:
    """Phase 2 placeholder: FDM with spatially varying eps'."""

    def test_higher_loss_gets_more_power(self):
        """Cells with higher eps'' receive proportionally more P_v."""
        E2 = np.ones((5, 5, 5), dtype=np.float32) * 1e6
        eps_low = np.ones((5, 5, 5), dtype=np.float32) * 0.5
        eps_high = np.ones((5, 5, 5), dtype=np.float32) * 2.0

        P_low = compute_power_density_np(E2, eps_low)
        P_high = compute_power_density_np(E2, eps_high)

        # P_high should be 4x P_low (eps ratio = 4)
        np.testing.assert_allclose(P_high / P_low, 4.0, rtol=1e-6)
