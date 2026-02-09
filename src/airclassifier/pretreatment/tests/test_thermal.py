"""
Tests for the thermal solver.

Validates:
- CFL stability condition scales correctly with grid spacing
- Uniform heating gives linear temperature rise
- Energy conservation: total energy added = integral(P_v * dt)
- Convective BC drives surface towards air temperature
"""

import numpy as np
import pytest

from airclassifier.pretreatment.physics.thermal import ThermalSolver


# ── Helpers ──────────────────────────────────────────────────────────

def _make_solver(nx=10, ny=8, nz=10, dx=0.02, dy=0.005, dz=0.02):
    return ThermalSolver(
        grid_shape=(nx, ny, nz),
        cell_sizes=(dx, dy, dz),
    )


class TestExplicitHeatEquation:
    """Explicit FDM heat equation tests."""

    def test_cfl_timestep(self):
        """CFL dt should decrease with finer grid (halve dx → ¼ dt)."""
        coarse = _make_solver(dx=0.02, dy=0.02, dz=0.02)
        fine = _make_solver(dx=0.01, dy=0.01, dz=0.01)

        k = 0.5
        rho_cp = 1e6

        dt_coarse = coarse.get_cfl_dt(k, rho_cp)
        dt_fine = fine.get_cfl_dt(k, rho_cp)

        # dt scales as dx^2 → fine should be ~4x smaller
        ratio = dt_coarse / dt_fine
        assert ratio == pytest.approx(4.0, rel=1e-3)

    def test_uniform_heating_linear_rise(self):
        """Uniform P_v with no conduction should give linear T rise.

        dT/dt = P_v / (rho*c_p)  →  T(t) = T0 + P_v/(rho*c_p) * t
        """
        nx, ny, nz = 10, 8, 10
        solver = _make_solver(nx=nx, ny=ny, nz=nz)
        solver.initialize(22.0)

        rho_cp_val = 1e6  # J/(m^3*K)
        P_v_val = 5e4     # W/m^3
        dt = 0.1

        rho_cp = np.full((nx, ny, nz), rho_cp_val, dtype=np.float32)
        k_eff = np.full((nx, ny, nz), 0.0, dtype=np.float32)  # no conduction
        P_v = np.full((nx, ny, nz), P_v_val, dtype=np.float32)
        evap = np.zeros((nx, ny, nz), dtype=np.float32)
        mask = np.ones((nx, ny, nz), dtype=np.int32)

        n_steps = 10
        for _ in range(n_steps):
            solver.step(dt, P_v, evap, rho_cp, k_eff, mask, T_inlet_c=22.0)

        t_total = dt * n_steps
        expected_rise = P_v_val / rho_cp_val * t_total
        T_expected = 22.0 + expected_rise

        # Check interior cells (not boundary)
        interior = solver.T[2:-2, 2:-2, 2:-2]
        np.testing.assert_allclose(interior, T_expected, rtol=0.02)

    def test_no_source_stays_constant(self):
        """With zero source and zero conduction, T should not change."""
        nx, ny, nz = 8, 6, 8
        solver = _make_solver(nx=nx, ny=ny, nz=nz)
        solver.initialize(50.0)

        rho_cp = np.full((nx, ny, nz), 1e6, dtype=np.float32)
        k_eff = np.zeros((nx, ny, nz), dtype=np.float32)
        P_v = np.zeros((nx, ny, nz), dtype=np.float32)
        evap = np.zeros((nx, ny, nz), dtype=np.float32)
        mask = np.ones((nx, ny, nz), dtype=np.int32)

        for _ in range(5):
            solver.step(0.1, P_v, evap, rho_cp, k_eff, mask, T_inlet_c=50.0)

        # Interior should remain at 50 degC
        interior = solver.T[1:-1, 1:-1, 1:-1]
        np.testing.assert_allclose(interior, 50.0, atol=1e-6)

    def test_energy_conservation(self):
        """Total energy increase ≈ integral(P_v * dt) when no losses."""
        nx, ny, nz = 12, 8, 12
        dx, dy, dz = 0.02, 0.005, 0.02
        solver = _make_solver(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        solver.initialize(22.0)

        rho_cp_val = 5e5
        P_v_val = 1e4
        dt = 0.05
        cell_vol = dx * dy * dz

        rho_cp = np.full((nx, ny, nz), rho_cp_val, dtype=np.float32)
        k_eff = np.full((nx, ny, nz), 0.001, dtype=np.float32)  # tiny k
        P_v = np.full((nx, ny, nz), P_v_val, dtype=np.float32)
        evap = np.zeros((nx, ny, nz), dtype=np.float32)
        mask = np.ones((nx, ny, nz), dtype=np.int32)

        T_initial = solver.T.copy()
        n_steps = 10
        for _ in range(n_steps):
            solver.step(dt, P_v, evap, rho_cp, k_eff, mask, T_inlet_c=22.0)

        dT = solver.T - T_initial
        energy_gained = float(np.sum(rho_cp * dT * cell_vol))
        energy_input = P_v_val * nx * ny * nz * cell_vol * dt * n_steps

        # Allow 30% tolerance due to boundary effects
        assert energy_gained == pytest.approx(energy_input, rel=0.30)

    def test_convection_bc_cools_surface(self):
        """Convective BC should cool a hot surface towards T_air."""
        nx, ny, nz = 8, 6, 8
        dx, dy, dz = 0.02, 0.01, 0.02
        solver = _make_solver(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        solver.initialize(80.0)  # Start hot

        rho_cp = np.full((nx, ny, nz), 1e6, dtype=np.float32)
        k_eff = np.full((nx, ny, nz), 0.2, dtype=np.float32)

        T_air = 25.0
        h_conv = 50.0  # Strong convection
        j_surface = 4  # Top of bed

        T_before = solver.T[:, j_surface, :].copy()

        solver.apply_convection_bc(
            j_surface=j_surface,
            h_conv=h_conv,
            T_air_c=T_air,
            rho_cp=rho_cp,
            k_eff=k_eff,
            dt=0.5,
        )

        T_after = solver.T[:, j_surface, :]

        # Surface should have cooled (moved towards T_air)
        assert float(np.mean(T_after)) < float(np.mean(T_before))
        assert float(np.mean(T_after)) > T_air  # Not colder than air
