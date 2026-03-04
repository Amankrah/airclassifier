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
        OutletState,
        MachineConfig,
        MaterialProperties,
        Recipe,
    )

    # Quick start
    sim = GP15Simulator(MachineConfig(), MaterialProperties())
    sim.load_recipe(Recipe(name="yellow_pea_standard", recipe_number=1,
                           electrode_gap_mm=80, belt_speed_m_per_min=0.5))
    result = sim.run(duration_s=120.0)
    outlet = sim.get_outlet_conditions()

Modules:
    geometry    Oven chamber, electrode, conveyor belt meshes
    physics     RF field, thermal, moisture, airflow solvers
    kernels     Warp GPU kernels (heating, diffusion, advection)
    control     PLC controller, recipe system, safety logic
    materials   Feedstock properties (yellow pea, faba bean, oat)
    io          VTK/CSV export, 3D field visualization helpers
    optimizer   Recipe optimization and sensitivity sweeps
"""

# =============================================================================
# LAZY IMPORTS - Deferred to avoid Warp JIT issues in PyInstaller bundles
# =============================================================================
# Many submodules import NVIDIA Warp which requires source code access.
# By deferring imports, the GUI can start without loading simulation code.

# Safe imports (no warp dependency)
from .config import MachineConfig, MaterialProperties, Recipe
from .materials import get_material_preset


def __getattr__(name):
    """Lazy import handler for module-level attributes."""
    # Calibration
    if name in ("CalibrationOptimizer", "CalibrationResult", "load_plc_data"):
        from .calibration import CalibrationOptimizer, CalibrationResult, load_plc_data
        return {"CalibrationOptimizer": CalibrationOptimizer,
                "CalibrationResult": CalibrationResult,
                "load_plc_data": load_plc_data}[name]

    # Control
    if name in ("GP15Controller", "RecipeStore"):
        from .control import GP15Controller, RecipeStore
        return {"GP15Controller": GP15Controller, "RecipeStore": RecipeStore}[name]

    # Desirability
    if name in ("DesirabilityProfile", "DesirabilityResult", "score_desirability"):
        from .desirability import DesirabilityProfile, DesirabilityResult, score_desirability
        return {"DesirabilityProfile": DesirabilityProfile,
                "DesirabilityResult": DesirabilityResult,
                "score_desirability": score_desirability}[name]

    # Conveyor (kernels - has warp)
    if name in ("ConveyorDriveController", "ConveyorDriveState"):
        from .kernels.transport import ConveyorDriveController, ConveyorDriveState
        return {"ConveyorDriveController": ConveyorDriveController,
                "ConveyorDriveState": ConveyorDriveState}[name]

    # Optimizer
    if name in ("DifferentiableOptimizer", "OptimizationResult",
                "optimize_recipe", "sensitivity_sweep"):
        from .optimizer import (DifferentiableOptimizer, OptimizationResult,
                                optimize_recipe, sensitivity_sweep)
        return {"DifferentiableOptimizer": DifferentiableOptimizer,
                "OptimizationResult": OptimizationResult,
                "optimize_recipe": optimize_recipe,
                "sensitivity_sweep": sensitivity_sweep}[name]

    # Physics (coupling - has warp)
    if name in ("CoupledSimulator", "OutletState", "PretreatmentResult", "StepState"):
        from .physics.coupling import (CoupledSimulator, OutletState,
                                       PretreatmentResult, StepState)
        return {"CoupledSimulator": CoupledSimulator,
                "OutletState": OutletState,
                "PretreatmentResult": PretreatmentResult,
                "StepState": StepState}[name]

    # Simulator
    if name == "GP15Simulator":
        from .simulator import GP15Simulator
        return GP15Simulator

    raise AttributeError(f"module 'airclassifier.pretreatment' has no attribute '{name}'")


__all__ = [
    # ── Public API ─────────────────────────────────────────────────
    "GP15Simulator",
    # ── Config ─────────────────────────────────────────────────────
    "MachineConfig",
    "MaterialProperties",
    "Recipe",
    # ── Results ────────────────────────────────────────────────────
    "PretreatmentResult",
    "StepState",
    "OutletState",
    # ── Materials ──────────────────────────────────────────────────
    "get_material_preset",
    # ── Control ────────────────────────────────────────────────────
    "GP15Controller",
    "RecipeStore",
    # ── Conveyor drive ─────────────────────────────────────────────
    "ConveyorDriveController",
    "ConveyorDriveState",
    # ── Calibration ────────────────────────────────────────────────
    "CalibrationOptimizer",
    "CalibrationResult",
    "load_plc_data",
    # ── Optimizer (§11) ────────────────────────────────────────────
    "OptimizationResult",
    "optimize_recipe",
    "sensitivity_sweep",
    "DifferentiableOptimizer",
    # ── Desirability scoring ───────────────────────────────────────
    "DesirabilityProfile",
    "DesirabilityResult",
    "score_desirability",
    # ── Internal (advanced) ────────────────────────────────────────
    "CoupledSimulator",
]
