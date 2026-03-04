"""
Milling Module
==============

Physics-based digital twin of a horizontal-shaft hammer mill
for ProteinProcessIO.

Process chain: Pretreatment (GP-15) -> Hammer Mill -> Air Classifier

Main entry points:
    - HammerMillSimulator: High-level simulator class
    - run_milling_simulation: Convenience function
    - create_hammer_mill_machine: Geometry factory

Public API:
    - HammerMillSimulator
    - MillingOutletState
    - MillConfig
    - MillRecipe
    - ScreenConfig
    - BreakageParams
    - MillingResult
"""

# =============================================================================
# LAZY IMPORTS - Deferred to avoid Warp JIT issues in PyInstaller bundles
# =============================================================================
# The physics module imports NVIDIA Warp which requires source code access.
# By deferring imports, the GUI can start without loading simulation code.

# Config imports are safe (no warp dependency)
from .config import (
    MillConfig,
    ScreenConfig,
    BreakageParams,
    MillRecipe,
    MillingOutletState,
)


def __getattr__(name):
    """Lazy import handler for module-level attributes."""
    # Simulator imports (depends on physics -> warp)
    if name in ("HammerMillSimulator", "MillingResult", "run_milling_simulation"):
        from .simulator import HammerMillSimulator, MillingResult, run_milling_simulation
        return {
            "HammerMillSimulator": HammerMillSimulator,
            "MillingResult": MillingResult,
            "run_milling_simulation": run_milling_simulation,
        }[name]

    # Geometry imports (depends on warp)
    if name in ("create_hammer_mill_machine", "build_hammer_mill_meshes",
                "HammerMillMachineAssembly", "COMPONENT_COLORS"):
        from .geometry import (
            create_hammer_mill_machine, build_hammer_mill_meshes,
            HammerMillMachineAssembly, COMPONENT_COLORS,
        )
        return {
            "create_hammer_mill_machine": create_hammer_mill_machine,
            "build_hammer_mill_meshes": build_hammer_mill_meshes,
            "HammerMillMachineAssembly": HammerMillMachineAssembly,
            "COMPONENT_COLORS": COMPONENT_COLORS,
        }[name]

    # Physics imports (depends on warp)
    if name in ("CoupledMillingEngine", "MillingStepState", "ImpactSolver",
                "BreakageModel", "ScreenClassifier", "ConvergenceDetector",
                "TerminationConfig"):
        from .physics import (
            CoupledMillingEngine, MillingStepState, ImpactSolver,
            BreakageModel, ScreenClassifier, ConvergenceDetector, TerminationConfig,
        )
        return {
            "CoupledMillingEngine": CoupledMillingEngine,
            "MillingStepState": MillingStepState,
            "ImpactSolver": ImpactSolver,
            "BreakageModel": BreakageModel,
            "ScreenClassifier": ScreenClassifier,
            "ConvergenceDetector": ConvergenceDetector,
            "TerminationConfig": TerminationConfig,
        }[name]

    raise AttributeError(f"module 'airclassifier.milling' has no attribute '{name}'")


__all__ = [
    # Main simulator
    "HammerMillSimulator",
    "run_milling_simulation",
    "MillingResult",
    # Configuration
    "MillConfig",
    "ScreenConfig",
    "BreakageParams",
    "MillRecipe",
    "MillingOutletState",
    # Geometry
    "create_hammer_mill_machine",
    "build_hammer_mill_meshes",
    "HammerMillMachineAssembly",
    "COMPONENT_COLORS",
    # Physics
    "CoupledMillingEngine",
    "MillingStepState",
    "ImpactSolver",
    "BreakageModel",
    "ScreenClassifier",
    # Convergence
    "ConvergenceDetector",
    "TerminationConfig",
]
