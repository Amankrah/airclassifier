"""
Breakage Properties
===================

Material-specific breakage parameters for the hammer mill.
Provides empirical or fitted parameters for different feedstocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..config import BreakageParams


@dataclass
class MaterialBreakageProperties:
    """Breakage properties for a specific material.

    These parameters control how the material breaks under impact
    in the hammer mill.
    """

    name: str = "generic"

    # Base breakage parameters
    breakage_params: BreakageParams = None

    # Material-specific factors
    hardness_factor: float = 1.0          # Harder materials break less easily
    moisture_sensitivity: float = 0.1     # Effect of moisture on breakage
    temperature_sensitivity: float = 0.01 # Effect of temperature

    def __post_init__(self):
        if self.breakage_params is None:
            self.breakage_params = BreakageParams()

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
MATERIAL_LIBRARY: Dict[str, MaterialBreakageProperties] = {
    "yellow_pea": MaterialBreakageProperties(
        name="yellow_pea",
        breakage_params=BreakageParams(
            selection_rate_constant=0.12,
            selection_size_exponent=1.1,
            breakage_distribution_exponent=0.85,
        ),
        hardness_factor=0.9,
        moisture_sensitivity=0.15,
    ),
    "chickpea": MaterialBreakageProperties(
        name="chickpea",
        breakage_params=BreakageParams(
            selection_rate_constant=0.10,
            selection_size_exponent=1.0,
            breakage_distribution_exponent=0.80,
        ),
        hardness_factor=1.1,
        moisture_sensitivity=0.12,
    ),
    "lentil": MaterialBreakageProperties(
        name="lentil",
        breakage_params=BreakageParams(
            selection_rate_constant=0.14,
            selection_size_exponent=1.2,
            breakage_distribution_exponent=0.90,
        ),
        hardness_factor=0.85,
        moisture_sensitivity=0.10,
    ),
    "wheat": MaterialBreakageProperties(
        name="wheat",
        breakage_params=BreakageParams(
            selection_rate_constant=0.13,
            selection_size_exponent=1.15,
            breakage_distribution_exponent=0.82,
        ),
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
