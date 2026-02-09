"""
Pretreatment Geometry
=====================

Parametric mesh generation for the GP-15 RF heating machine.

Components
----------
Individual machine parts following the Params + Geometry pattern:

.. autosummary::
    :toctree: generated/

    components.generator.GeneratorGeometry
    components.oven_chamber.OvenChamberGeometry
    components.conveyor_belt.ConveyorBeltGeometry
    components.electrode.ElectrodeGeometry
    components.tunnel.TunnelGeometry
    components.hopper.InfeedHopperGeometry
    components.emu.EMUGeometry
    components.hmi_panel.HMIPanelGeometry
    components.housing.HousingGeometry
    components.support_legs.SupportLegsGeometry

Assembly
--------
Complete machine assembly with port-based component alignment:

.. autosummary::
    :toctree: generated/

    assembly.machine.GP15MachineAssembly
    assembly.machine.create_gp15_machine
    assembly.machine.build_gp15_machine_meshes

Utilities
---------
Mesh generation helpers and SDF functions:

.. autosummary::
    :toctree: generated/

    mesh_utils.box_mesh
    mesh_utils.hollow_box_mesh
    mesh_utils.concat_meshes
    mesh_utils.translate_mesh
    sdf.oven_sdf
"""

# Components (new API)
from .components import (
    # Generator
    GeneratorGeometry,
    GeneratorParams,
    # Oven
    OvenChamberGeometry,
    OvenChamberParams,
    # Conveyor
    ConveyorBeltGeometry,
    ConveyorBeltParams,
    # Electrode
    ElectrodeGeometry,
    ElectrodeParams,
    # Tunnel
    TunnelGeometry,
    TunnelParams,
    # Hopper
    InfeedHopperGeometry,
    InfeedHopperParams,
    # EMU
    EMUGeometry,
    EMUParams,
    # HMI Panel
    HMIPanelGeometry,
    HMIPanelParams,
    # Housing
    HousingGeometry,
    HousingParams,
    # Support Legs
    SupportLegsGeometry,
    SupportLegsParams,
)

# Assembly (new API)
from .assembly import (
    GP15MachineAssembly,
    GP15MachineParams,
    COMPONENT_COLORS,
    create_gp15_machine,
    build_gp15_machine_meshes,
)

# Mesh utilities
from .mesh_utils import (
    box_mesh,
    hollow_box_mesh,
    concat_meshes,
    translate_mesh,
    cylinder_mesh,
)

# SDF
from .sdf import oven_sdf

# Backward compatibility aliases
from .components.oven_chamber import (
    OvenGeometry,
    OvenGeometryParams,
)
from .components.conveyor_belt import (
    ConveyorGeometry,
    ConveyorParams,
)

__all__ = [
    # Components
    "GeneratorGeometry",
    "GeneratorParams",
    "OvenChamberGeometry",
    "OvenChamberParams",
    "ConveyorBeltGeometry",
    "ConveyorBeltParams",
    "ElectrodeGeometry",
    "ElectrodeParams",
    "TunnelGeometry",
    "TunnelParams",
    "InfeedHopperGeometry",
    "InfeedHopperParams",
    "EMUGeometry",
    "EMUParams",
    "HMIPanelGeometry",
    "HMIPanelParams",
    "HousingGeometry",
    "HousingParams",
    "SupportLegsGeometry",
    "SupportLegsParams",
    # Assembly
    "GP15MachineAssembly",
    "GP15MachineParams",
    "COMPONENT_COLORS",
    "create_gp15_machine",
    "build_gp15_machine_meshes",
    # Mesh utilities
    "box_mesh",
    "hollow_box_mesh",
    "concat_meshes",
    "translate_mesh",
    "cylinder_mesh",
    # SDF
    "oven_sdf",
    # Backward compatibility
    "OvenGeometry",
    "OvenGeometryParams",
    "ConveyorGeometry",
    "ConveyorParams",
]
