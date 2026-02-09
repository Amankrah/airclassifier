"""
Conveyor Belt Geometry - Backward Compatibility
================================================

Re-exports from :mod:`.components.conveyor_belt` for backward compatibility.

New code should import directly from ``components.conveyor_belt``.
"""

from .components.conveyor_belt import (
    ConveyorBeltGeometry,
    ConveyorBeltParams,
    # Backward compatibility aliases
    ConveyorGeometry,
    ConveyorParams,
)

__all__ = [
    "ConveyorBeltGeometry",
    "ConveyorBeltParams",
    "ConveyorGeometry",
    "ConveyorParams",
]
