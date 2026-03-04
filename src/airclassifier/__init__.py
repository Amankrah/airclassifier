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


# =============================================================================
# LAZY IMPORTS - Deferred to avoid Warp JIT issues in PyInstaller bundles
# =============================================================================
# These imports trigger NVIDIA Warp which requires source code access.
# By deferring imports, the GUI can start without loading simulation code.

def __getattr__(name):
    """Lazy import handler for package-level attributes."""
    # Geometry imports
    if name in ("CycloneAssembly", "CycloneGeometryParams", "create_standard_cyclone"):
        from .geometry import CycloneAssembly, CycloneGeometryParams, create_standard_cyclone
        return {"CycloneAssembly": CycloneAssembly,
                "CycloneGeometryParams": CycloneGeometryParams,
                "create_standard_cyclone": create_standard_cyclone}[name]

    # Fluid imports
    if name in ("CycloneFlowField", "CycloneFlowParams"):
        from .fluid import CycloneFlowField, CycloneFlowParams
        return {"CycloneFlowField": CycloneFlowField,
                "CycloneFlowParams": CycloneFlowParams}[name]

    # Particle imports
    if name in ("ParticleMaterial", "MaterialProperties"):
        from .particles.material import ParticleMaterial, MaterialProperties
        return {"ParticleMaterial": ParticleMaterial,
                "MaterialProperties": MaterialProperties}[name]

    # Simulation imports
    if name in ("FlowMode", "SystemState", "AirSystemSimulator", "AirSystemConfig",
                "FeedFlowPhysicsSimulator", "FlowPhysicsConfig",
                "AirFlowPhysicsSimulator", "AirFlowPhysicsConfig"):
        from .simulation import (
            FlowMode, SystemState, AirSystemSimulator, AirSystemConfig,
            FeedFlowPhysicsSimulator, FlowPhysicsConfig,
            AirFlowPhysicsSimulator, AirFlowPhysicsConfig,
        )
        return {
            "FlowMode": FlowMode,
            "SystemState": SystemState,
            "AirSystemSimulator": AirSystemSimulator,
            "AirSystemConfig": AirSystemConfig,
            "FeedFlowPhysicsSimulator": FeedFlowPhysicsSimulator,
            "FlowPhysicsConfig": FlowPhysicsConfig,
            "AirFlowPhysicsSimulator": AirFlowPhysicsSimulator,
            "AirFlowPhysicsConfig": AirFlowPhysicsConfig,
        }[name]

    # Sub-module imports
    if name == "kinetics":
        from . import kinetics
        return kinetics
    if name == "visualization":
        from . import visualization
        return visualization
    if name == "io":
        from . import io
        return io
    if name == "utils":
        from . import utils
        return utils

    raise AttributeError(f"module 'airclassifier' has no attribute '{name}'")


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
    # Simulation - Basic
    "AirSystemSimulator",
    "AirSystemConfig",
    # Simulation - Physics-based
    "FeedFlowPhysicsSimulator",
    "FlowPhysicsConfig",
    "AirFlowPhysicsSimulator",
    "AirFlowPhysicsConfig",
    # Sub-modules
    "kinetics",
    "visualization",
    "io",
    "utils",
]
