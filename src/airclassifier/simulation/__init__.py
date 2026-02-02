"""
Simulation module for air classifier systems.

Provides system-level simulation for the complete air classifier:

Subsystem Simulators:
- AirSystemSimulator: Blower, filter, and damper simulation
- FeedSystemSimulator: Material handling (hopper, airlock, feeder, deagglomerator)
- ClassificationSystemSimulator: Particle separation (venturi, zigzag, cyclones, bag filter)

Complete System:
- CompleteSystemSimulator: Full integrated simulation with all systems coupled

Flow Modes:
- ANALYTICAL: Fast analytical flow models (Rankine vortex, etc.)
- CFD: Full Navier-Stokes with CFD-DEM coupling
"""

from .simulator import (
    # Enums
    FlowMode,
    SystemState,
    
    # Base classes
    BaseSimulationConfig,
    BaseSimulationState,
    
    # Air System
    AirSystemConfig,
    AirSystemState,
    AirSystemSimulator,
    
    # Feed System
    FeedSystemConfig,
    FeedSystemState,
    FeedSystemSimulator,
    
    # Classification System
    ClassificationConfig,
    ClassificationState,
    ClassificationSystemSimulator,
    
    # Complete System
    CompleteSystemConfig,
    CompleteSystemState,
    CompleteSystemSimulator,
    
    # Factory functions
    create_air_system_simulator,
    create_feed_system_simulator,
    create_classification_simulator,
    create_complete_system_simulator,
)

from .cfd_dem_coupling import (
    CFDDEMCoupler,
    CFDConfig,
    DEMConfig,
    CycloneCFDParams,
    CouplingMode,
    TurbulenceModelType,
)

__all__ = [
    # Enums
    "FlowMode",
    "SystemState",
    
    # Base classes
    "BaseSimulationConfig",
    "BaseSimulationState",
    
    # Air System Simulator
    "AirSystemConfig",
    "AirSystemState",
    "AirSystemSimulator",
    "create_air_system_simulator",
    
    # Feed System Simulator
    "FeedSystemConfig",
    "FeedSystemState",
    "FeedSystemSimulator",
    "create_feed_system_simulator",
    
    # Classification System Simulator
    "ClassificationConfig",
    "ClassificationState",
    "ClassificationSystemSimulator",
    "create_classification_simulator",
    
    # Complete System Simulator
    "CompleteSystemConfig",
    "CompleteSystemState",
    "CompleteSystemSimulator",
    "create_complete_system_simulator",
    
    # CFD-DEM (advanced)
    "CFDDEMCoupler",
    "CFDConfig",
    "DEMConfig",
    "CycloneCFDParams",
    "CouplingMode",
    "TurbulenceModelType",
]
