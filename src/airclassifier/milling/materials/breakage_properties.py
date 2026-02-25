"""
Breakage Properties
===================

Material-specific breakage parameters for the hammer mill.
Provides empirical or fitted parameters for different feedstocks,
including physical seed properties for mass calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from ..config import BreakageParams


def compute_particle_mass(
    diameter_m: float,
    density_kg_m3: float,
    sphericity: float = 1.0,
) -> float:
    """Compute particle mass from physical properties.

    Uses equivalent-sphere volume scaled by sphericity as a volume
    correction factor. For non-spherical seeds (lentils, wheat),
    sphericity < 1 reduces the effective volume.

    Args:
        diameter_m: Equivalent sphere diameter [m]
        density_kg_m3: True particle density [kg/m³]
        sphericity: Shape factor (0-1), used as volume correction

    Returns:
        Particle mass [kg]
    """
    volume = (math.pi / 6.0) * diameter_m ** 3
    return volume * density_kg_m3 * sphericity


@dataclass
class MaterialBreakageProperties:
    """Breakage and physical properties for a specific seed/material.

    These parameters control how the material breaks under impact
    in the hammer mill, and provide physical properties for mass
    calculations.
    """

    name: str = "generic"

    # Base breakage parameters
    breakage_params: BreakageParams = None

    # Physical seed properties
    seed_density_kg_m3: float = 1400.0    # True particle density [kg/m³]
    seed_d50_um: float = 7000.0           # Typical whole seed median diameter [µm]
    thousand_kernel_weight_g: float = 200.0  # TKW [g] — standard agricultural measurement
    sphericity: float = 0.85              # Shape factor (0-1)

    # Material-specific factors
    hardness_factor: float = 1.0          # Harder materials break less easily
    moisture_sensitivity: float = 0.1     # Effect of moisture on breakage
    temperature_sensitivity: float = 0.01 # Effect of temperature

    def __post_init__(self):
        if self.breakage_params is None:
            self.breakage_params = BreakageParams()

    @property
    def seed_mass_kg(self) -> float:
        """Compute single seed mass from physical properties [kg]."""
        return compute_particle_mass(
            self.seed_d50_um * 1e-6,
            self.seed_density_kg_m3,
            self.sphericity,
        )

    def get_effective_params(
        self,
        moisture_wb: float = 0.12,
        temperature_c: float = 25.0,
    ) -> BreakageParams:
        """Get breakage params adjusted for conditions.

        Args:
            moisture_wb: Moisture content (wet basis)
            temperature_c: Temperature [C]

        Returns:
            Adjusted BreakageParams
        """
        base = self.breakage_params

        # Moisture effect: higher moisture = harder to break
        moisture_factor = 1.0 - self.moisture_sensitivity * (moisture_wb - 0.12)

        # Temperature effect: higher temp = easier to break
        temp_factor = 1.0 + self.temperature_sensitivity * (temperature_c - 25.0)

        # Combined adjustment to selection rate
        combined_factor = moisture_factor * temp_factor / self.hardness_factor

        return BreakageParams(
            num_size_classes=base.num_size_classes,
            d_min_um=base.d_min_um,
            d_max_um=base.d_max_um,
            selection_rate_constant=base.selection_rate_constant * combined_factor,
            selection_size_exponent=base.selection_size_exponent,
            selection_reference_size_um=base.selection_reference_size_um,
            breakage_distribution_exponent=base.breakage_distribution_exponent,
            min_impact_energy_j=base.min_impact_energy_j / combined_factor,
            energy_to_breakage_factor=base.energy_to_breakage_factor,
        )


# Pre-defined material breakage properties
# Tuned for fine flour production (d50 ~60µm) for protein-starch separation
# Physical properties from agricultural seed science literature
MATERIAL_LIBRARY: Dict[str, MaterialBreakageProperties] = {
    "yellow_pea": MaterialBreakageProperties(
        name="yellow_pea",
        breakage_params=BreakageParams(
            selection_rate_constant=0.42,
            selection_size_exponent=1.3,
            selection_reference_size_um=500.0,
            breakage_distribution_exponent=0.52,
            min_impact_energy_j=0.0005,
            energy_to_breakage_factor=8.0,
        ),
        seed_density_kg_m3=1400.0,
        seed_d50_um=7000.0,              # ~7mm diameter
        thousand_kernel_weight_g=200.0,   # ~200mg per seed
        sphericity=0.88,
        hardness_factor=0.85,
        moisture_sensitivity=0.15,
    ),
    "chickpea": MaterialBreakageProperties(
        name="chickpea",
        breakage_params=BreakageParams(
            selection_rate_constant=0.38,
            selection_size_exponent=1.25,
            selection_reference_size_um=500.0,
            breakage_distribution_exponent=0.55,
            min_impact_energy_j=0.0006,
            energy_to_breakage_factor=7.0,
        ),
        seed_density_kg_m3=1300.0,
        seed_d50_um=8500.0,              # ~8.5mm diameter
        thousand_kernel_weight_g=350.0,
        sphericity=0.80,
        hardness_factor=1.0,
        moisture_sensitivity=0.12,
    ),
    "lentil": MaterialBreakageProperties(
        name="lentil",
        breakage_params=BreakageParams(
            selection_rate_constant=0.45,
            selection_size_exponent=1.35,
            selection_reference_size_um=500.0,
            breakage_distribution_exponent=0.50,
            min_impact_energy_j=0.0004,
            energy_to_breakage_factor=9.0,
        ),
        seed_density_kg_m3=1400.0,
        seed_d50_um=4500.0,              # ~4.5mm diameter (flat disc)
        thousand_kernel_weight_g=40.0,
        sphericity=0.70,                  # Flat lentil shape
        hardness_factor=0.80,
        moisture_sensitivity=0.10,
    ),
    "faba_bean": MaterialBreakageProperties(
        name="faba_bean",
        breakage_params=BreakageParams(
            selection_rate_constant=0.40,
            selection_size_exponent=1.3,
            selection_reference_size_um=500.0,
            breakage_distribution_exponent=0.53,
            min_impact_energy_j=0.0005,
            energy_to_breakage_factor=8.0,
        ),
        seed_density_kg_m3=1350.0,
        seed_d50_um=9000.0,              # ~9mm diameter
        thousand_kernel_weight_g=400.0,
        sphericity=0.75,                  # Oblate shape
        hardness_factor=0.88,
        moisture_sensitivity=0.14,
    ),
    "wheat": MaterialBreakageProperties(
        name="wheat",
        breakage_params=BreakageParams(
            selection_rate_constant=0.35,
            selection_size_exponent=1.2,
            selection_reference_size_um=600.0,
            breakage_distribution_exponent=0.58,
            min_impact_energy_j=0.0006,
            energy_to_breakage_factor=6.0,
        ),
        seed_density_kg_m3=1350.0,
        seed_d50_um=3500.0,              # ~3.5mm (length-based)
        thousand_kernel_weight_g=35.0,
        sphericity=0.55,                  # Elongated kernel
        hardness_factor=1.0,
        moisture_sensitivity=0.18,
    ),
}


def get_material_properties(name: str) -> MaterialBreakageProperties:
    """Get breakage properties for a material.

    Args:
        name: Material name

    Returns:
        MaterialBreakageProperties
    """
    if name in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[name]
    return MaterialBreakageProperties(name=name)
