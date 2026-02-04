"""
Simulation module for air classifier systems.

Provides physics-based simulation using actual geometry:

Physics-Based Simulators:
- FeedFlowPhysicsSimulator: Material flow through feed system (hopper, airlock, feeder, deagglomerator)
- AirFlowPhysicsSimulator: SPH-based air flow through air system (filter, blower, dampers)
  Uses Smoothed Particle Hydrodynamics for physically accurate air flow simulation

Basic Simulators:
- AirSystemSimulator: Simplified blower, filter, and damper simulation

Advanced:
- CFDDEMCoupler: Full CFD-DEM coupling for detailed analysis
"""

from .simulator import (
    # Enums
    FlowMode,
    SystemState,
    PouringState,

    # Base classes
    BaseSimulationConfig,
    BaseSimulationState,

    # Air System (basic)
    AirSystemConfig,
    AirSystemState,
    AirSystemSimulator,

    # Feed System (wrapper)
    FeedSystemConfig,
    FeedSystemSimulator,
)

from .cfd_dem_coupling import (
    CFDDEMCoupler,
    CFDConfig,
    DEMConfig,
    CycloneCFDParams,
    CouplingMode,
    TurbulenceModelType,
)

from .feed_flow_physics import (
    FeedFlowPhysicsSimulator,
    FlowPhysicsConfig,
    FlowPhysicsState,
    FlowZone,
    LidState,
    SimulationPhase,
    ComponentGeometry,
    extract_geometry,
    create_physics_flow_simulator,
)

from .air_flow_physics import (
    AirFlowPhysicsSimulator,
    AirFlowPhysicsConfig,
    AirFlowPhysicsState,
    BlowerState,
    SystemPhase as AirSystemPhase,
    BlowerGeometry,
    FilterGeometry,
    DamperGeometry,
    DuctSegment,
    extract_air_geometry,
    create_air_flow_simulator,
)

from .classification_flow_physics import (
    ClassificationFlowConfig,
    ClassificationFlowState,
    SimulationPhase as ClassificationSimulationPhase,
    Zone as ClassificationZone,
    ComponentGeometry as ClassificationComponentGeometry,
    ConnectionPath as ClassificationConnectionPath,
    extract_geometry as extract_classification_geometry,
    print_geometry_summary as print_classification_geometry_summary,
)

__all__ = [
    # Enums
    "FlowMode",
    "SystemState",
    "PouringState",
    "LidState",

    # Base classes
    "BaseSimulationConfig",
    "BaseSimulationState",

    # Air System Simulator (basic)
    "AirSystemConfig",
    "AirSystemState",
    "AirSystemSimulator",

    # Feed System Simulator (wrapper)
    "FeedSystemConfig",
    "FeedSystemSimulator",
    
    # CFD-DEM (advanced)
    "CFDDEMCoupler",
    "CFDConfig",
    "DEMConfig",
    "CycloneCFDParams",
    "CouplingMode",
    "TurbulenceModelType",
    
    # Physics-based feed flow (geometry-driven)
    "FeedFlowPhysicsSimulator",
    "FlowPhysicsConfig",
    "FlowPhysicsState",
    "FlowZone",
    "LidState",
    "SimulationPhase",
    "ComponentGeometry",
    "extract_geometry",
    "create_physics_flow_simulator",
    
    # Physics-based air flow with SPH (geometry-driven, physically accurate)
    "AirFlowPhysicsSimulator",
    "AirFlowPhysicsConfig",
    "AirFlowPhysicsState",
    "BlowerState",
    "AirSystemPhase",
    "BlowerGeometry",
    "FilterGeometry",
    "DamperGeometry",
    "DuctSegment",
    "extract_air_geometry",
    "create_air_flow_simulator",
    
    # Classification flow physics (geometry-driven)
    "ClassificationFlowConfig",
    "ClassificationFlowState",
    "ClassificationSimulationPhase",
    "ClassificationZone",
    "ClassificationComponentGeometry",
    "ClassificationConnectionPath",
    "extract_classification_geometry",
    "print_classification_geometry_summary",
]
