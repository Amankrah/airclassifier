"""
GP-15 Machine Geometry - Backward Compatibility
================================================

Re-exports from :mod:`.assembly.machine` for backward compatibility.

New code should use:

    from airclassifier.pretreatment.geometry import create_gp15_machine
    machine = create_gp15_machine()
    meshes = machine.get_component_meshes()

Or import directly from ``assembly.machine``.
"""

from .assembly.machine import (
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
