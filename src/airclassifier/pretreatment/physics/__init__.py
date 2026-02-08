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
