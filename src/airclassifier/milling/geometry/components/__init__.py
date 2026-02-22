"""
Hammer Mill Geometry Components
===============================

Individual geometry components for the hammer mill assembly.
Each component has a Params dataclass and a Geometry class that
generates meshes.

Components:
    - Rotor: main shaft with support discs (animated)
    - Hammers: swinging impact elements (animated with rotor)
    - Screen: curved perforated plate (static)
    - Housing: outer casing (static)
    - FeedChute: inlet duct (static)
    - Drive: motor and belt system (static)
"""

from .rotor import RotorParams, RotorGeometry
from .hammers import HammerParams, HammerGeometry
from .screen import ScreenParams, ScreenGeometry
from .housing import HousingParams, HousingGeometry
from .feed_chute import FeedChuteParams, FeedChuteGeometry
from .drive import DriveParams, DriveGeometry

__all__ = [
    "RotorParams",
    "RotorGeometry",
    "HammerParams",
    "HammerGeometry",
    "ScreenParams",
    "ScreenGeometry",
    "HousingParams",
    "HousingGeometry",
    "FeedChuteParams",
    "FeedChuteGeometry",
    "DriveParams",
    "DriveGeometry",
]
