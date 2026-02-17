"""
Pretreatment Physics Solvers
============================

Multi-physics solvers for the GP-15 RF heating simulation:
- RF electric field (Laplace equation with dielectric coupling)
- Thermal (heat equation with RF source and latent heat sink)
- Moisture (diffusion + evaporation kinetics)
- Airflow (EMU extraction + heater model)
- Coupling orchestrator (timestep sequencing)
"""

from .airflow import AirflowState, EMUAirflowModel
from .coupling import CoupledSimulator, OutletState, PretreatmentResult, StepState
from .moisture import MoistureSolver
from .rf_field import RFFieldSolver
from .thermal import ThermalSolver

__all__ = [
    # Coupling orchestrator
    "CoupledSimulator",
    "PretreatmentResult",
    "StepState",
    "OutletState",
    # Individual solvers
    "RFFieldSolver",
    "ThermalSolver",
    "MoistureSolver",
    "EMUAirflowModel",
    "AirflowState",
]
