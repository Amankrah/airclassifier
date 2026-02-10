"""
Configuration Dataclasses
=========================

Machine specifications, material properties, and recipe definitions
for the GP-15 RF dielectric heating simulation.

All values are cross-referenced against the GP-15 Installation and
Operation Manual and verified against machine test reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


# ============================================================================
#  Machine Configuration
# ============================================================================

@dataclass
class MachineConfig:
    """GP-15 machine parameters.

    All values cross-referenced against the Installation and Operation
    Manual. Values marked [TBD] should be measured from the physical
    machine and updated here.
    """

    # --- Generator / Oscillator ---
    rf_frequency_hz: float = 27.12e6
    max_rf_power_kw: float = 15.0
    anode_voltage_no_load_kv: float = 9.18       # Test report: no-load
    anode_voltage_full_load_kv: float = 8.38     # Test report: full-load
    anode_current_no_load_a: float = 0.4         # Test report
    anode_current_full_load_a: float = 2.58      # Test report
    supply_voltage_v: float = 600.0              # 3-phase
    supply_kva_max: float = 42.0

    # --- Oven / Applicator ---
    oven_length_m: float = 1.5                   # [TBD - MEASURE from drawing]
    electrode_gap_min_m: float = 0.02            # [TBD - MEASURE from test report]
    electrode_gap_max_m: float = 0.30            # [TBD - MEASURE from test report]
    electrode_count: int = 2                     # Two plates per electrode
    electrode_perforation: bool = True           # Upper plates are perforated

    # --- Conveyor ---
    belt_width_m: float = 0.8
    belt_speed_min_m_per_min: float = 0.1
    belt_speed_max_m_per_min: float = 2.0
    belt_thickness_m: float = 0.002              # PTFE belt ~2mm
    belt_permittivity_real: float = 2.1          # PTFE eps'
    belt_permittivity_loss: float = 0.0003       # PTFE eps''
    wear_strip_thickness_m: float = 0.001        # 24 Teflon wear strips
    top_sheet_thickness_m: float = 0.0005        # Protective top sheet

    # --- EMU (Environment Management Unit) ---
    heater_power_total_kw: float = 12.0          # 2 banks x 6 x 1 kW
    heater_bank_count: int = 2
    extraction_fan_capacity_m3_per_min: float = 31.1
    extraction_fan_hz_min: float = 5.0
    extraction_fan_hz_max: float = 60.0
    extraction_duct_diameter_m: float = 0.25
    extraction_max_backpressure_pa: float = 250.0
    air_cooling_cfm: float = 1119.0              # Generator cooling

    # --- Control ---
    recipe_capacity: int = 30
    max_recycle_restarts: int = 4
    restart_delay_s: float = 2.0
    electrode_debounce_s: float = 0.5
    ambient_temp_limit_c: float = 40.0

    # --- Machine envelope ---
    machine_length_m: float = 5.5
    machine_width_m: float = 2.9
    machine_height_m: float = 2.2
    machine_weight_kg: float = 2550.0

    @property
    def belt_stack_thickness_m(self) -> float:
        """Total thickness of dielectric layers between material and lower electrode."""
        return self.belt_thickness_m + self.wear_strip_thickness_m + self.top_sheet_thickness_m

    def anode_voltage_kv(self, current_a: float) -> float:
        """Compute anode voltage at given current (linear droop model)."""
        slope = (
            (self.anode_voltage_no_load_kv - self.anode_voltage_full_load_kv)
            / (self.anode_current_full_load_a - self.anode_current_no_load_a)
        )
        return self.anode_voltage_no_load_kv - slope * (current_a - self.anode_current_no_load_a)


# ============================================================================
#  Material Properties
# ============================================================================

@dataclass
class MaterialProperties:
    """Feedstock characterization for the RF simulation.

    Describes the whole beans, seeds, or groats as they enter the GP-15.
    This is the raw material BEFORE milling — not flour.

    Process chain:
        Whole seeds --> GP-15 RF drying --> Pin mill (flour) --> Air classifier

    All property models are parameterized as functions of temperature
    and moisture content. Coefficients are material-specific and should
    be fitted from measured data for the whole-seed form.
    """

    name: str = "yellow_pea"

    # --- Initial conditions ---
    initial_moisture_wb: float = 0.10            # 10% wet basis
    target_moisture_wb: float = 0.03             # 3% target
    initial_temperature_c: float = 22.0          # Ambient

    # --- Dielectric properties at 27.12 MHz ---
    # eps''(T,M) = a1*M^2 + a2*M + a3*M*T + a4*T + a5
    dielectric_loss_coeffs: Tuple[float, ...] = (85.0, 2.5, 0.12, 0.008, 0.02)
    # eps'(T,M) = b1*M + b2*T + b3
    dielectric_const_coeffs: Tuple[float, ...] = (25.0, -0.05, 2.5)

    # --- Thermal properties ---
    c_p_dry: float = 1380.0                      # J/(kg*K)
    c_p_water: float = 4186.0                    # J/(kg*K)
    k_dry: float = 0.18                          # W/(m*K)
    k_moisture_beta: float = 4.0                 # k sensitivity to M
    rho_solid: float = 1450.0                    # kg/m^3 (whole seed solid density)

    # --- Moisture diffusivity: D_eff = D0 * exp(-Ea / (R*T)) ---
    D_eff_D0: float = 5.7e-4                     # m^2/s
    D_eff_Ea: float = 28500.0                    # J/mol

    # --- Evaporation model ---
    # The threshold temperature controls the onset of active moisture
    # removal.  At 40 °C the material barely reaches it with the GP-15's
    # power density, producing zero drying.  Legume seeds lose moisture
    # to dry air at any temperature above the dew point (~10-15 °C).
    # A threshold of 25 °C models the practical onset of accelerated
    # evaporation from RF heating while allowing drying to begin as
    # soon as material warms above ambient.
    k_evap: float = 1.5e-4                       # 1/(degC*s) rate constant
    T_evap_threshold_c: float = 25.0             # degC

    # --- Bed geometry (packed bed of whole seeds on the conveyor) ---
    bed_depth_m: float = 0.05                    # 50 mm typical
    bed_porosity: float = 0.40                   # Void fraction between whole seeds

    def eps_loss(self, T_c: float, M_wb: float) -> float:
        """Dielectric loss factor eps''(T, M)."""
        a1, a2, a3, a4, a5 = self.dielectric_loss_coeffs
        return a1 * M_wb**2 + a2 * M_wb + a3 * M_wb * T_c + a4 * T_c + a5

    def eps_real(self, T_c: float, M_wb: float) -> float:
        """Dielectric constant eps'(T, M)."""
        b1, b2, b3 = self.dielectric_const_coeffs
        return b1 * M_wb + b2 * T_c + b3

    def specific_heat(self, T_c: float, M_wb: float) -> float:
        """Specific heat capacity c_p(M) via linear mixing [J/(kg*K)]."""
        return self.c_p_dry * (1.0 - M_wb) + self.c_p_water * M_wb

    def thermal_conductivity(self, T_c: float, M_wb: float) -> float:
        """Effective thermal conductivity k(M) [W/(m*K)]."""
        k_solid = self.k_dry * (1.0 + self.k_moisture_beta * M_wb)
        k_air = 0.026  # W/(m*K) at ~50 degC
        return k_solid * (1.0 - self.bed_porosity) + k_air * self.bed_porosity

    def bulk_density(self, M_wb: float) -> float:
        """Bulk density rho(M) [kg/m^3]."""
        M_db = M_wb / max(1.0 - M_wb, 1.0e-6)  # dry basis
        return self.rho_solid * (1.0 - self.bed_porosity) * (1.0 + M_db)

    def moisture_diffusivity(self, T_c: float) -> float:
        """Effective moisture diffusivity D_eff(T) [m^2/s]."""
        T_K = T_c + 273.15
        R = 8.314
        return self.D_eff_D0 * math.exp(-self.D_eff_Ea / (R * T_K))


# ============================================================================
#  Recipe — mirrors HMI recipe system
# ============================================================================

@dataclass
class Recipe:
    """GP-15 HMI recipe. Up to 30 can be stored.

    Maps directly to the GP-15 Recipe Edit Screen parameters.
    """

    name: str = "default"
    recipe_number: int = 0                       # 1-30, 0 = manual mode

    # --- Process setpoints ---
    electrode_gap_mm: float = 80.0               # Gap setpoint
    belt_speed_m_per_min: float = 0.5            # Conveyor speed
    rf_power_enabled: bool = True

    # --- Anode current protection ---
    mrh_amps: float = 2.6                        # Meter Relay High (overcurrent trip)
    mrl_amps: float = 2.0                        # Meter Relay Low (drive stop threshold)

    # --- EMU settings ---
    extraction_fan_hz: float = 30.0
    heater_bank_1_on: bool = True
    heater_bank_2_on: bool = True
    heater_fan_hz: float = 30.0

    # --- Temperature control (optional automatic mode) ---
    temp_control_enabled: bool = False
    temp_setpoint_c: float = 60.0
    temp_sensors_active: Tuple[bool, ...] = (True, True, True, True, True, True)
    temp_envelope_time_s: float = 10.0           # Correction interval
