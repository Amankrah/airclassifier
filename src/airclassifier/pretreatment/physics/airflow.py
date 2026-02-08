"""
EMU Airflow Model
=================

Models the Environment Management Unit (EMU) airflow:
- Extraction fan (variable speed, 5-60 Hz) removes humid air from oven
- Heater banks (2 x 6 kW) pre-heat incoming air
- Convective heat transfer at the material bed surface

The EMU maintains oven humidity and temperature to prevent condensation
and assist surface drying.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MachineConfig, Recipe


@dataclass
class AirflowState:
    """Current EMU airflow state."""
    extraction_flow_m3_per_s: float = 0.0
    air_temperature_c: float = 22.0
    air_humidity_rh: float = 0.50
    convective_htc_w_per_m2k: float = 15.0   # h_conv at bed surface


class EMUAirflowModel:
    """Simple EMU airflow model for convective boundary conditions.

    Computes air temperature inside the oven from heater power and
    extraction rate, then provides the convective HTC at the bed surface.
    """

    def __init__(self, machine: MachineConfig):
        self._machine = machine
        self.state = AirflowState()

    def update(self, recipe: Recipe, dt: float):
        """Update airflow state from current recipe settings.

        Args:
            recipe: Active recipe (fan Hz, heater banks).
            dt: Timestep [s].
        """
        # TODO: Compute extraction flow rate from fan Hz
        # TODO: Compute air temperature from heater power / flow rate
        # TODO: Update convective HTC model
        raise NotImplementedError
