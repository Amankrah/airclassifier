"""
Simulation module for air classifier systems.

Provides physics-based simulation using actual geometry:

Physics-Based Simulators:
- FeedFlowPhysicsSimulator: Material flow through feed system (hopper, airlock, feeder, deagglomerator)
- AirFlowPhysicsSimulator: Air flow through air system (filter, blower, dampers)

Basic Simulators:
- AirSystemSimulator: Simplified blower, filter, and damper simulation

Advanced:
- CFDDEMCoupler: Full CFD-DEM coupling for detailed analysis
"""

from .simulator import (
    # Enums
    FlowMode,
    SystemState,
    
    # Base classes
    BaseSimulationConfig,
    BaseSimulationState,
    
    # Air System (basic)
    AirSystemConfig,
    AirSystemState,
    AirSystemSimulator,
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

__all__ = [
    # Enums
    "FlowMode",
    "SystemState",
    
    # Base classes
    "BaseSimulationConfig",
    "BaseSimulationState",
    
    # Air System Simulator (basic)
    "AirSystemConfig",
    "AirSystemState",
    "AirSystemSimulator",
    
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
    
    # Physics-based air flow (geometry-driven)
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
]
