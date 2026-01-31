"""
Instrumentation components for air classification systems.

This module provides process instrumentation geometries including:
- Pressure transmitter ports
- Temperature ports (thermowells)
- Sample extraction ports
- Sight glasses / inspection ports
"""

from .pressure_port import (
    PressurePort,
    PressurePortParams,
    create_flush_pressure_port,
    create_extended_pressure_port,
    create_averaging_pressure_port,
)
from .temp_port import (
    TemperaturePort,
    TemperaturePortParams,
    create_threaded_thermowell,
    create_flanged_thermowell,
    create_weld_thermowell,
)
from .sample_port import (
    SamplePort,
    SamplePortParams,
    create_ball_valve_sample_port,
    create_isokinetic_sample_port,
)
from .sight_glass import (
    SightGlass,
    SightGlassParams,
    create_standard_sight_glass,
    create_illuminated_sight_glass,
)

__all__ = [
    # Pressure Port
    "PressurePort",
    "PressurePortParams",
    "create_flush_pressure_port",
    "create_extended_pressure_port",
    "create_averaging_pressure_port",
    # Temperature Port
    "TemperaturePort",
    "TemperaturePortParams",
    "create_threaded_thermowell",
    "create_flanged_thermowell",
    "create_weld_thermowell",
    # Sample Port
    "SamplePort",
    "SamplePortParams",
    "create_ball_valve_sample_port",
    "create_isokinetic_sample_port",
    # Sight Glass
    "SightGlass",
    "SightGlassParams",
    "create_standard_sight_glass",
    "create_illuminated_sight_glass",
]
