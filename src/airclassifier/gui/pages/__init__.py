"""
GUI Pages
=========

Full-page views for different process stages.
Each page contains its own 3D viewport, controls, and results.

Pages:
  - **ClassificationPage** — Air classification (zigzag + wheel + cyclones)
  - **PretreatmentPage**   — RF pretreatment (GP-15 dielectric heating)
"""

from .classification_page import ClassificationPage
from .pretreatment_page import PretreatmentPage

__all__ = ["ClassificationPage", "PretreatmentPage"]
