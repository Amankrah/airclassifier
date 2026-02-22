"""
Milling Physics Module
======================

Physics simulation for the hammer mill.
Provides solvers for impact, breakage, screen passage, and
the coupled simulation engine.

Components:
    - ImpactSolver: Hammer-particle collision detection
    - BreakageModel: Particle size reduction
    - ScreenClassifier: Screen passage classification
    - CoupledMillingEngine: Orchestrates the simulation
"""

from .impact import ImpactSolver, ImpactStats
from .breakage import BreakageModel, BreakageStats
from .screen_classifier import ScreenClassifier, ScreenStats
from .coupling import (
    CoupledMillingEngine,
    MillingStepState,
    ParticleState,
)

__all__ = [
    # Impact
    "ImpactSolver",
    "ImpactStats",
    # Breakage
    "BreakageModel",
    "BreakageStats",
    # Screen
    "ScreenClassifier",
    "ScreenStats",
    # Coupling
    "CoupledMillingEngine",
    "MillingStepState",
    "ParticleState",
]
