"""
EMU Airflow Model
=================

Models the Environment Management Unit (EMU) airflow:
- Extraction fan (variable speed, 5-60 Hz) removes humid air from oven
- Heater banks (2 x 6 kW) pre-heat incoming air
- Convective heat transfer at the material bed surface

The EMU maintains oven humidity and temperature to prevent condensation
and assist surface drying.

Phase 1: Simplified algebraic model (no transient air dynamics).
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

    The engineering guide (Section 2.4 & 4.2.2) specifies:

    * Extraction fan capacity = 31.1 m^3/min at 60 Hz (VFD).
    * Heater arrays = 2 banks x 6 kW = 12 kW total.
    * T_air = T_ambient + Q_heater / (m_dot_air * c_p_air).
    * h_conv from flat-plate forced-convection correlation.
    """

    # Air properties at ~50 degC
    _RHO_AIR = 1.09         # kg/m^3
    _CP_AIR = 1005.0        # J/(kg*K)
    _K_AIR = 0.028          # W/(m*K)
    _NU_AIR = 1.8e-5        # m^2/s  (kinematic viscosity)
    _PR_AIR = 0.71           # Prandtl number

    def __init__(self, machine: MachineConfig):
        self._machine = machine
        self.state = AirflowState()

    def update(self, recipe: Recipe, dt: float):
        """Update airflow state from current recipe settings.

        Args:
            recipe: Active recipe (fan Hz, heater banks).
            dt: Timestep [s] (unused in Phase 1 — algebraic model).
        """
        m = self._machine

        # --- Extraction volumetric flow rate ---
        # Linear scaling with VFD frequency: Q = Q_max * (f / f_max)
        fan_fraction = max(recipe.extraction_fan_hz, 0.0) / m.extraction_fan_hz_max
        Q_m3_per_s = m.extraction_fan_capacity_m3_per_min / 60.0 * fan_fraction
        self.state.extraction_flow_m3_per_s = Q_m3_per_s

        # Mass flow rate
        m_dot = Q_m3_per_s * self._RHO_AIR  # kg/s

        # --- Heater power ---
        n_banks_on = int(recipe.heater_bank_1_on) + int(recipe.heater_bank_2_on)
        Q_heater_w = n_banks_on * (m.heater_power_total_kw / m.heater_bank_count) * 1000.0

        # Air temperature rise:  T_air = T_ambient + Q / (m_dot * c_p)
        T_ambient = 22.0  # degC default
        if m_dot > 1e-4:
            dT = Q_heater_w / (m_dot * self._CP_AIR)
        else:
            dT = 0.0
        self.state.air_temperature_c = T_ambient + dT

        # --- Convective HTC at bed surface ---
        # Simplified flat-plate forced convection:
        #   Re = U * L / nu,   Nu = 0.664 * Re^0.5 * Pr^(1/3)   (laminar)
        # Air velocity over the bed from extraction fan:
        #   U = Q / A_cross   (A_cross ~ belt_width * oven_height above bed)
        A_cross = m.belt_width_m * 0.15  # approximate flow cross-section
        U_air = Q_m3_per_s / max(A_cross, 1e-4)
        L_char = m.oven_length_m

        Re = U_air * L_char / self._NU_AIR
        if Re > 1.0:
            Nu = 0.664 * Re**0.5 * self._PR_AIR ** (1.0 / 3.0)
        else:
            Nu = 3.66  # minimum for enclosed flow
        h_conv = Nu * self._K_AIR / L_char
        self.state.convective_htc_w_per_m2k = max(h_conv, 5.0)  # floor at 5 W/(m^2*K)
