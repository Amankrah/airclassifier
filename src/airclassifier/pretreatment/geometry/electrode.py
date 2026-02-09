"""
Electrode Geometry - Backward Compatibility
============================================

Re-exports from :mod:`.components.electrode` for backward compatibility.

New code should import directly from ``components.electrode``.
"""

from .components.electrode import (
    ElectrodeGeometry,
    ElectrodeParams,
)

__all__ = [
    "ElectrodeGeometry",
    "ElectrodeParams",
]
