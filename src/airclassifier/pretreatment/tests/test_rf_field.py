"""
Tests for the RF electric field solver.

Validates:
- Uniform parallel-plate approximation: E = V/gap
- Energy conservation: integral(P_v) = total RF power
- Boundary conditions: phi = V at upper, phi = 0 at lower
"""

import pytest


class TestUniformField:
    """Phase 1: uniform parallel-plate model."""

    def test_uniform_field_magnitude(self):
        """E = V/gap for uniform parallel plates."""
        # TODO: Implement
        pass

    def test_power_density_magnitude(self):
        """P_v = 2*pi*f*eps_0*eps''*|E|^2 at known conditions."""
        # TODO: Implement
        pass

    def test_total_power_matches_input(self):
        """Integral of P_v over volume = input RF power [kW]."""
        # TODO: Implement
        pass


class TestVariablePermittivity:
    """Phase 2: FDM with spatially varying eps'."""

    def test_higher_loss_gets_more_power(self):
        """Cells with higher eps'' should receive more power."""
        # TODO: Implement
        pass
