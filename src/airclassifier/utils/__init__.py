"""
Utilities module for cyclone air classifier.

Provides physical constants, unit conversions, and validation utilities.
"""

# Safe imports (no warp dependency)
from .constants import (
    PI,
    TWO_PI,
    GRAVITY,
    AirProperties,
    AirPropertiesAtTemp,
    MaterialDensities,
    FlowRegimes,
    NumericalConstants,
)
from .validation import (
    RUN2_TARGETS,
    ValidationResult,
    compare_sim_to_run2,
    get_run2_targets,
)


def __getattr__(name):
    """Lazy import handler for warp-dependent constants."""
    # Warp constants (lazy loaded)
    if name == "GRAVITY_VEC":
        from .constants import GRAVITY_VEC
        return GRAVITY_VEC
    if name in ("WP_PI", "WP_GRAVITY", "WP_AIR_DENSITY", "WP_AIR_VISCOSITY", "WP_EPSILON"):
        from . import constants
        return getattr(constants, name)
    raise AttributeError(f"module 'airclassifier.utils' has no attribute '{name}'")


__all__ = [
    # Mathematical constants
    "PI",
    "TWO_PI",
    "GRAVITY",
    "GRAVITY_VEC",
    # Air properties
    "AirProperties",
    "AirPropertiesAtTemp",
    # Material densities
    "MaterialDensities",
    # Flow regimes
    "FlowRegimes",
    # Numerical constants
    "NumericalConstants",
    # Warp constants
    "WP_PI",
    "WP_GRAVITY",
    "WP_AIR_DENSITY",
    "WP_AIR_VISCOSITY",
    "WP_EPSILON",
    # Run#2 validation
    "RUN2_TARGETS",
    "ValidationResult",
    "compare_sim_to_run2",
    "get_run2_targets",
]
