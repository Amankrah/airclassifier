"""
Material Presets
================

Factory functions for creating MaterialProperties instances for
common feedstocks processed in the GP-15 RF heating machine.

IMPORTANT: The GP-15 operates on WHOLE beans, seeds, or groats —
not flour. Flour is only produced after milling, which is the step
BETWEEN pretreatment and air classification:

    Whole seeds --> GP-15 RF drying --> Pin mill (flour) --> Air classifier

Dielectric coefficients are initial estimates for the whole-seed
form and should be updated with measured data.
"""

from __future__ import annotations

from ..config import MaterialProperties


def create_yellow_pea(
    initial_moisture_wb: float = 0.10,
    bed_depth_m: float = 0.05,
) -> MaterialProperties:
    """Whole yellow pea seeds (Pisum sativum).

    Intact or dehulled yellow peas as received from storage.
    Higher dielectric loss than cereals due to the protein matrix's
    water-binding capacity. Typical seed diameter 6-8 mm.

    After RF conditioning to ~3% moisture, the peas are pin-milled
    into whole flour and fed to the air classifier.
    """
    return MaterialProperties(
        name="yellow_pea",
        initial_moisture_wb=initial_moisture_wb,
        target_moisture_wb=0.03,
        initial_temperature_c=22.0,
        dielectric_loss_coeffs=(85.0, 2.5, 0.12, 0.008, 0.02),
        dielectric_const_coeffs=(25.0, -0.05, 2.5),
        c_p_dry=1380.0,
        k_dry=0.18,
        k_moisture_beta=4.0,
        rho_solid=1450.0,          # solid density of whole pea seed
        D_eff_D0=5.7e-4,
        D_eff_Ea=28500.0,
        k_evap=5.0e-5,            # whole seeds: 0.6 kg/kWh (Manual Ch.5)
        T_evap_threshold_c=25.0,   # onset of accelerated evaporation
        bed_depth_m=bed_depth_m,
        bed_porosity=0.40,         # packed bed of whole seeds
    )


def create_faba_bean(
    initial_moisture_wb: float = 0.11,
    bed_depth_m: float = 0.05,
) -> MaterialProperties:
    """Whole faba bean seeds (Vicia faba).

    Intact faba beans. Larger seed size than yellow peas (~8-12 mm).
    Slightly higher moisture binding due to higher tannin content.

    After RF conditioning, beans are pin-milled and air-classified.
    """
    return MaterialProperties(
        name="faba_bean",
        initial_moisture_wb=initial_moisture_wb,
        target_moisture_wb=0.03,
        initial_temperature_c=22.0,
        dielectric_loss_coeffs=(90.0, 2.8, 0.13, 0.009, 0.025),
        dielectric_const_coeffs=(27.0, -0.05, 2.6),
        c_p_dry=1420.0,
        k_dry=0.17,
        k_moisture_beta=4.2,
        rho_solid=1400.0,          # solid density of whole faba bean
        D_eff_D0=4.8e-4,
        D_eff_Ea=29000.0,
        k_evap=4.5e-5,            # whole seeds: 0.6 kg/kWh (Manual Ch.5)
        T_evap_threshold_c=25.0,
        bed_depth_m=bed_depth_m,
        bed_porosity=0.42,         # packed bed porosity (larger seeds)
    )


def create_oat(
    initial_moisture_wb: float = 0.12,
    bed_depth_m: float = 0.05,
) -> MaterialProperties:
    """Oat groats (Avena sativa).

    Dehulled oat kernels. Smaller and more elongated than legume seeds.
    Higher lipid content (~6-8%) affects dielectric properties and
    limits the maximum safe temperature (lipid oxidation above ~80 C).

    After RF conditioning, groats are pin-milled and air-classified.
    """
    return MaterialProperties(
        name="oat",
        initial_moisture_wb=initial_moisture_wb,
        target_moisture_wb=0.04,
        initial_temperature_c=22.0,
        dielectric_loss_coeffs=(75.0, 2.2, 0.10, 0.007, 0.018),
        dielectric_const_coeffs=(22.0, -0.04, 2.3),
        c_p_dry=1350.0,
        k_dry=0.16,
        k_moisture_beta=3.8,
        rho_solid=1350.0,          # solid density of oat groat
        D_eff_D0=6.2e-4,
        D_eff_Ea=27500.0,
        k_evap=4.0e-5,            # whole groats: 0.6 kg/kWh (Manual Ch.5)
        T_evap_threshold_c=25.0,
        bed_depth_m=bed_depth_m,
        bed_porosity=0.38,         # packed bed of oat groats
    )


def get_material_preset(name: str, **kwargs) -> MaterialProperties:
    """Get a MaterialProperties instance by name.

    Args:
        name: "yellow_pea", "faba_bean", or "oat".
        **kwargs: Overrides passed to the factory function.
    """
    factories = {
        "yellow_pea": create_yellow_pea,
        "faba_bean": create_faba_bean,
        "oat": create_oat,
    }
    factory = factories.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown material '{name}'. Options: {list(factories.keys())}")
    return factory(**kwargs)
