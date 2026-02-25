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

from .config import (
    MillConfig,
    ScreenConfig,
    BreakageParams,
    MillRecipe,
    MillingOutletState,
)
from .simulator import (
    HammerMillSimulator,
    MillingResult,
    run_milling_simulation,
)
from .geometry import (
    create_hammer_mill_machine,
    build_hammer_mill_meshes,
    HammerMillMachineAssembly,
    COMPONENT_COLORS,
)
from .physics import (
    CoupledMillingEngine,
    MillingStepState,
    ImpactSolver,
    BreakageModel,
    ScreenClassifier,
    ConvergenceDetector,
    TerminationConfig,
)

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
