"""
Oven Chamber Geometry - Backward Compatibility
==============================================

Re-exports from :mod:`.components.oven_chamber` for backward compatibility.

New code should import directly from ``components.oven_chamber``.
"""

from .components.oven_chamber import (
    OvenChamberGeometry,
    OvenChamberParams,
    # Backward compatibility aliases
    OvenGeometry,
    OvenGeometryParams,
)

__all__ = [
    "OvenChamberGeometry",
    "OvenChamberParams",
    "OvenGeometry",
    "OvenGeometryParams",
]
