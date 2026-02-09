"""
Pretreatment Materials
======================

Feedstock property definitions and factory functions for the GP-15
RF heating simulation. Provides temperature- and moisture-dependent
dielectric, thermal, and transport properties.

The GP-15 operates on whole beans, seeds, or groats — NOT flour.
Flour is only produced after milling, which follows pretreatment.

Materials:
    yellow_pea  Whole yellow pea seeds (Pisum sativum)
    faba_bean   Whole faba bean seeds (Vicia faba)
    oat         Oat groats (Avena sativa)
"""

from ..config import MaterialProperties
from .presets import (
    create_yellow_pea,
    create_faba_bean,
    create_oat,
    get_material_preset,
)

__all__ = [
    "MaterialProperties",
    "create_yellow_pea",
    "create_faba_bean",
    "create_oat",
    "get_material_preset",
]
