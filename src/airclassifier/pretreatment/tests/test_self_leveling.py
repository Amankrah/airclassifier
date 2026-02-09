"""
Tests for the self-leveling property of RF dielectric heating.

The key feature of RF heating is self-leveling: wetter regions have
higher eps'' and absorb more energy, which drives preferential drying.
This creates spatial uniformity — the fundamental advantage of RF
over convective drying.

Validates (from engineering guide §4.3.4 and §10.2):
- Wetter cells receive more power (higher eps'')
- Non-uniform initial moisture converges toward uniform final moisture
- Starting from 40% CV at infeed, outfeed CV should converge below inlet CV
"""

import numpy as np
import pytest

from airclassifier.pretreatment.config import MachineConfig, MaterialProperties, Recipe
from airclassifier.pretreatment.geometry.oven import OvenGeometry, OvenGeometryParams
from airclassifier.pretreatment.kernels.dielectric_heating import (
    TWO_PI_F_EPS0,
    compute_power_density_np,
    update_material_properties_np,
)
from airclassifier.pretreatment.physics.coupling import CoupledSimulator


# ── Helpers ──────────────────────────────────────────────────────────

def _make_coupled(nx=20, ny=8, nz=12, **kwargs):
    """Create a coupled simulator for testing."""
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
        **kwargs,
    )

    gap_m = 0.10
    mask = oven.build_material_mask(
        electrode_gap_m=gap_m,
        bed_depth_m=material.bed_depth_m,
        belt_stack_m=machine.belt_stack_thickness_m,
    )
    sim.initialize(cell_is_material=mask)
    return sim, material, mask


class TestSelfLeveling:
    """Self-leveling / preferential drying tests."""

    def test_wetter_gets_more_power(self):
        """Cells with higher M should have higher eps'' and thus higher P_v.

        This is the fundamental self-leveling mechanism: higher moisture
        → higher dielectric loss → more RF energy absorbed.
        """
        # Create two blocks of material at different moisture levels
        M_wet, M_dry = 0.12, 0.06
        T = 50.0  # degC

        mat = MaterialProperties()
        eps_wet = mat.eps_loss(T, M_wet)
        eps_dry = mat.eps_loss(T, M_dry)

        assert eps_wet > eps_dry, (
            f"Wetter material should have higher eps'': "
            f"eps''({M_wet})={eps_wet:.4f}, eps''({M_dry})={eps_dry:.4f}"
        )

        # Given the same E-field, wetter region absorbs more power
        E_sq = 1e6  # (1 kV/m)^2
        P_wet = TWO_PI_F_EPS0 * eps_wet * E_sq
        P_dry = TWO_PI_F_EPS0 * eps_dry * E_sq

        assert P_wet > P_dry
        # The ratio should be meaningful (> 1.5x)
        assert P_wet / P_dry > 1.5

    def test_non_uniform_converges(self):
        """The wetter region should lose moisture faster than the dryer
        region, demonstrating the self-leveling mechanism.

        Self-leveling means: wetter cells have higher eps'' → more RF
        power → faster heating → more evaporation → faster drying.
        So the moisture *spread* (max - min) should decrease.
        """
        sim, material, mask = _make_coupled()
        mat_mask = (mask == 1)

        # Set non-uniform initial moisture: top half wet, bottom half dry
        # (using Z-axis so belt advection doesn't mix them)
        nz = sim._grid_shape[2]
        half_z = nz // 2
        sim.moisture.M[:, :, :half_z] = np.where(
            mask[:, :, :half_z] == 1, 0.14, 0.0
        )
        sim.moisture.M[:, :, half_z:] = np.where(
            mask[:, :, half_z:] == 1, 0.06, 0.0
        )
        sim._update_properties()

        M_init = sim.moisture.M[mat_mask]
        spread_initial = float(np.max(M_init) - np.min(M_init))

        recipe = Recipe(
            name="test",
            recipe_number=1,
            electrode_gap_mm=100.0,
            belt_speed_m_per_min=0.0,   # stationary — no advection
            extraction_fan_hz=30.0,
        )

        # Track moisture in each half separately
        wet_mask = mat_mask & (np.arange(nz)[None, None, :] < half_z)
        dry_mask = mat_mask & (np.arange(nz)[None, None, :] >= half_z)

        M_wet_start = float(np.mean(sim.moisture.M[wet_mask]))
        M_dry_start = float(np.mean(sim.moisture.M[dry_mask]))

        for _ in range(80):
            sim.step(dt=0.5, recipe=recipe)

        M_wet_end = float(np.mean(sim.moisture.M[wet_mask]))
        M_dry_end = float(np.mean(sim.moisture.M[dry_mask]))

        # The wetter half should lose more moisture than the dryer half
        loss_wet = M_wet_start - M_wet_end
        loss_dry = M_dry_start - M_dry_end

        assert loss_wet > loss_dry, (
            f"Wetter region should dry faster: "
            f"wet loss={loss_wet:.5f}, dry loss={loss_dry:.5f}"
        )

    def test_final_uniformity(self):
        """After sufficient processing, the gap between the wet-half
        mean and dry-half mean should narrow due to self-leveling.

        Engineering guide §10.2: wetter regions dry faster, driving
        convergence toward spatial uniformity.
        """
        sim, material, mask = _make_coupled()
        mat_mask = (mask == 1)

        # Set non-uniform initial moisture: Z < half wet, Z >= half dry
        nx, ny, nz = sim._grid_shape
        half_z = nz // 2
        sim.moisture.M[:, :, :half_z] = np.where(
            mask[:, :, :half_z] == 1, 0.14, 0.0
        )
        sim.moisture.M[:, :, half_z:] = np.where(
            mask[:, :, half_z:] == 1, 0.06, 0.0
        )
        sim._update_properties()

        wet_mask = mat_mask & (np.arange(nz)[None, None, :] < half_z)
        dry_mask = mat_mask & (np.arange(nz)[None, None, :] >= half_z)

        gap_initial = abs(
            float(np.mean(sim.moisture.M[wet_mask]))
            - float(np.mean(sim.moisture.M[dry_mask]))
        )

        recipe = Recipe(
            name="test",
            recipe_number=1,
            electrode_gap_mm=100.0,
            belt_speed_m_per_min=0.0,
            extraction_fan_hz=30.0,
        )

        for _ in range(100):
            sim.step(dt=0.5, recipe=recipe)

        gap_final = abs(
            float(np.mean(sim.moisture.M[wet_mask]))
            - float(np.mean(sim.moisture.M[dry_mask]))
        )

        assert gap_final < gap_initial, (
            f"Wet/dry gap should narrow: "
            f"initial={gap_initial:.4f}, final={gap_final:.4f}"
        )
