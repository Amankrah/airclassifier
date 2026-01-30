"""
Simulation module for cyclone air classifier.

Provides the main simulation orchestration, time stepping,
and state management.

Two simulation approaches:
- CycloneSimulator: Fast analytical flow field (Rankine vortex)
- CFDDEMCoupler: Full CFD-DEM with Navier-Stokes solver
"""

from .simulator import (
    FlowMode,
    SimulationConfig,
    SimulationState,
    CycloneSimulator,
    create_simulator,
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
    # Factory function
    "create_simulator",
    "FlowMode",
    # Analytical flow simulator
    "SimulationConfig",
    "SimulationState",
    "CycloneSimulator",
    # CFD-DEM coupled simulator
    "CFDDEMCoupler",
    "CFDConfig",
    "DEMConfig",
    "CycloneCFDParams",
    "CouplingMode",
    "TurbulenceModelType",
]
