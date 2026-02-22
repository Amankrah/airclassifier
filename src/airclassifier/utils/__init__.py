"""
Utilities module for cyclone air classifier.

Provides physical constants, unit conversions, and validation utilities.
"""

from .constants import (
    PI,
    TWO_PI,
    GRAVITY,
    GRAVITY_VEC,
    AirProperties,
    AirPropertiesAtTemp,
    MaterialDensities,
    FlowRegimes,
    NumericalConstants,
    WP_PI,
    WP_GRAVITY,
    WP_AIR_DENSITY,
    WP_AIR_VISCOSITY,
    WP_EPSILON,
)
from .validation import (
    RUN2_TARGETS,
    ValidationResult,
    compare_sim_to_run2,
    get_run2_targets,
)

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
