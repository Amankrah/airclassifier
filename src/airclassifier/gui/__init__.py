"""
Air Classifier GUI Module
=========================

PySide6-based desktop application for interactive air classifier design
and NVIDIA Warp multiphysics simulation.

Features:
- Component assembly with drag-and-drop
- Parameter configuration panels
- 3D visualization with PyVista
- Real-time simulation controls
- Results analysis and export

Assembly Modes:
- Full System (use_preclassification=True): Venturi + Zigzag + Wheel + Cyclones
- Wheel-Only (use_preclassification=False): 3-point junction + Wheel + Cyclones

Material Sources:
- yellow_pea: Yellow peas (Pisum sativum)
- faba_bean: Faba beans (Vicia faba)
- oat: Oats (Avena sativa)

Usage:
    from airclassifier.gui import launch_app
    launch_app()

    # Check resources before simulation
    from airclassifier.gui import check_resources
    status = check_resources()
    if status["simulation"] and status["assemblies"]:
        print("Ready to simulate!")
"""

from .main_window import MainWindow
from .app import AirClassifierApp, launch_app
from .simulation_backend import (
    SimulationConfig,
    SimulationBackend,
    check_resources,
    get_particle_resources,
    get_assembly_resources,
    get_simulation_resources,
    get_available_materials,
    get_available_fractions,
    create_config_from_preset,
    create_mesh_from_assembly,
    MATERIAL_SOURCES,
    PARTICLE_FRACTIONS,
)

__all__ = [
    # Main application
    "MainWindow",
    "AirClassifierApp",
    "launch_app",
    # Simulation backend
    "SimulationConfig",
    "SimulationBackend",
    # Resource helpers
    "check_resources",
    "get_particle_resources",
    "get_assembly_resources",
    "get_simulation_resources",
    "get_available_materials",
    "get_available_fractions",
    # Factory functions
    "create_config_from_preset",
    "create_mesh_from_assembly",
    # Constants
    "MATERIAL_SOURCES",
    "PARTICLE_FRACTIONS",
]
