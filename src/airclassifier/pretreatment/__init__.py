"""
Pretreatment Module — RF Dielectric Heating Digital Twin
=========================================================

Physics-based simulation of the QMTI GP-15 Radio Frequency dielectric
heating machine for moisture conditioning of whole legume seeds and
cereal groats prior to milling and air classification.

Process chain:  Whole beans/seeds --> GP-15 RF drying --> Pin mill --> Air classifier

The GP-15 heats material volumetrically via a 27.12 MHz RF field in a
parallel-plate capacitor applicator. Water molecules absorb dielectric
energy preferentially, reducing moisture from 8-14% to 2-4% in a single
pass on a continuous conveyor belt.

Main entry points::

    from airclassifier.pretreatment import (
        GP15Simulator,
        PretreatmentResult,
        MachineConfig,
        MaterialProperties,
        Recipe,
    )

    # Quick start
    sim = GP15Simulator(MachineConfig(), MaterialProperties())
    sim.load_recipe(Recipe(name="yellow_pea_standard", recipe_number=1,
                           electrode_gap_mm=80, belt_speed_m_per_min=0.5))
    result = sim.run(duration_s=120.0)

Modules:
    geometry    Oven chamber, electrode, conveyor belt meshes
    physics     RF field, thermal, moisture, airflow solvers
    kernels     Warp GPU kernels (heating, diffusion, advection)
    control     PLC controller, recipe system, safety logic
    materials   Feedstock properties (yellow pea, faba bean, oat)
    io          VTK/CSV export, 3D field visualization helpers
"""

from .config import MachineConfig, MaterialProperties, Recipe

__all__ = [
    # Config
    "MachineConfig",
    "MaterialProperties",
    "Recipe",
    # Simulator (lazy imports to avoid heavy deps at import time)
    # "GP15Simulator",
    # "PretreatmentResult",
]
