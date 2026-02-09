"""
GP-15 Machine Assembly
======================

Assembles all GP-15 machine components with port-based alignment.

Classes:
- GP15MachineAssembly: Complete machine with all components
- GP15MachineParams: Assembly configuration

Factory functions:
- create_gp15_machine(): Standard GP-15 configuration
"""

from .machine import (
    GP15MachineAssembly,
    GP15MachineParams,
    COMPONENT_COLORS,
    create_gp15_machine,
    build_gp15_machine_meshes,
)

__all__ = [
    "GP15MachineAssembly",
    "GP15MachineParams",
    "COMPONENT_COLORS",
    "create_gp15_machine",
    "build_gp15_machine_meshes",
]
