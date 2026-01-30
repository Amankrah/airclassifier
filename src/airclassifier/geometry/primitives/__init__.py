"""
Geometric primitives for cyclone construction.

This module provides basic shapes (cylinder, cone, tube) that are
combined to create cyclone components.
"""

from .cylinder import Cylinder, CylinderParams, cylinder_sdf, cylinder_sdf_hollow
from .cone import Cone, ConeParams, cone_sdf, frustum_sdf
from .tube import Tube, TubeParams, RectangularDuct, RectangularDuctParams

__all__ = [
    # Cylinder
    "Cylinder",
    "CylinderParams",
    "cylinder_sdf",
    "cylinder_sdf_hollow",
    # Cone
    "Cone",
    "ConeParams",
    "cone_sdf",
    "frustum_sdf",
    # Tube
    "Tube",
    "TubeParams",
    "RectangularDuct",
    "RectangularDuctParams",
]
