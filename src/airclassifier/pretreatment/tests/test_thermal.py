"""
Tests for the thermal solver.

Validates:
- CFL stability condition
- Energy conservation with RF source
- Steady-state convergence
- Convective BC at bed surface
"""

import pytest


class TestExplicitHeatEquation:
    """Explicit FDM heat equation tests."""

    def test_cfl_timestep(self):
        """CFL dt should decrease with finer grid."""
        # TODO: Implement
        pass

    def test_uniform_heating(self):
        """Uniform P_v should give linear temperature rise."""
        # TODO: Implement
        pass

    def test_energy_conservation(self):
        """Total energy added = integral(P_v * dt) - integral(L_v * evap * dt)."""
        # TODO: Implement
        pass

    def test_convection_bc(self):
        """Surface temperature should approach T_air with high h_conv."""
        # TODO: Implement
        pass
