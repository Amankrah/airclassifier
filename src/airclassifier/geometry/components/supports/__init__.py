"""
Support structure components for air classification systems.

This module provides structural support geometries including:
- Equipment legs (tubular, channel, adjustable)
- Structural frames (bolted, welded)
- Platforms and access ladders
"""

from .legs import (
    EquipmentLegs,
    EquipmentLegParams,
    create_tubular_legs,
    create_adjustable_legs,
    create_channel_legs,
)
from .frame import (
    StructuralFrame,
    StructuralFrameParams,
    create_standard_frame,
    create_equipment_skid,
    create_mezzanine_frame,
)

__all__ = [
    # Equipment Legs
    "EquipmentLegs",
    "EquipmentLegParams",
    "create_tubular_legs",
    "create_adjustable_legs",
    "create_channel_legs",
    # Structural Frame
    "StructuralFrame",
    "StructuralFrameParams",
    "create_standard_frame",
    "create_equipment_skid",
    "create_mezzanine_frame",
]
