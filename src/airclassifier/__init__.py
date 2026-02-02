"""
Air Classifier - GPU-accelerated cyclone simulation using NVIDIA Warp.

A comprehensive simulation package for modeling particle separation
in cyclone air classifiers.

Modules:
    geometry: Cyclone geometry and mesh generation
    fluid: Flow field representation and fluid dynamics
    particles: Material properties and particle systems
    kinetics: Force calculations and particle dynamics
    simulation: Main simulation orchestration
    visualization: Plotting and rendering utilities
    io: VTK export and data I/O
    utils: Constants, units, and utilities
"""

__version__ = "0.1.0"
__author__ = "Emmanuel Kwofie"

# Convenience imports
from .geometry import (
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
)

from .fluid import (
    CycloneFlowField,
    CycloneFlowParams,
)

from .particles.material import (
    ParticleMaterial,
    MaterialProperties,
)

from .simulation import (
    # Enums
    FlowMode,
    SystemState,
    # System simulators
    AirSystemSimulator,
    FeedSystemSimulator,
    ClassificationSystemSimulator,
    CompleteSystemSimulator,
    # Configs
    AirSystemConfig,
    FeedSystemConfig,
    ClassificationConfig,
    CompleteSystemConfig,
    # Factory functions
    create_air_system_simulator,
    create_feed_system_simulator,
    create_classification_simulator,
    create_complete_system_simulator,
)

# Sub-modules available for import
from . import kinetics
from . import visualization
from . import io
from . import utils

__all__ = [
    # Version
    "__version__",
    # Geometry
    "CycloneAssembly",
    "CycloneGeometryParams",
    "create_standard_cyclone",
    # Fluid
    "CycloneFlowField",
    "CycloneFlowParams",
    # Particles
    "ParticleMaterial",
    "MaterialProperties",
    # Simulation - Enums
    "FlowMode",
    "SystemState",
    # Simulation - System simulators
    "AirSystemSimulator",
    "FeedSystemSimulator",
    "ClassificationSystemSimulator",
    "CompleteSystemSimulator",
    # Simulation - Configs
    "AirSystemConfig",
    "FeedSystemConfig",
    "ClassificationConfig",
    "CompleteSystemConfig",
    # Simulation - Factory functions
    "create_air_system_simulator",
    "create_feed_system_simulator",
    "create_classification_simulator",
    "create_complete_system_simulator",
    # Sub-modules
    "kinetics",
    "visualization",
    "io",
    "utils",
]
