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

from .calibration_store import get_calibration_defaults


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

    # Oscillator-to-electrode coupling factor.
    # Converts the anode DC voltage to the RF voltage appearing
    # across the electrodes:  V_rf = V_anode * coupling_factor.
    #
    # Calibrated against Run#1 PLC data (25-Mar-2025, full 2794 s)
    # using Nelder-Mead local optimization (calibration.py, 289 evals).
    #
    # Run#1: 61 kg whole yellow pea, gap=75mm, speed=0.2 m/min,
    #        bed=25mm, MRH=1.7 A.  PLC shows T reaching 101 C,
    #        Ia=1.65-1.70 A, gap opens 75→87 mm.
    #
    # Calibrated value: k ≈ 0.138 (from 0.181 baseline).
    # k_evap collapsed to ~1e-6 — whole seeds barely evaporate
    # moisture (intact seed coat).  The GP-15 at these settings
    # performs thermal conditioning, not drying.
    # Default from utility_docs/calibration_latest.json (single source of truth).
    oscillator_coupling_factor: float = field(
        default_factory=lambda: get_calibration_defaults()[0]
    )

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
    # Dry fractionation protocol: temper to 13.5–15% before RF for uniform heat
    # penetration, LOX deactivation, and improved protein separation (air classification).
    initial_moisture_wb: float = 0.14            # 14% wet basis (13.5–15% ideal for pretreatment)
    initial_temperature_c: float = 22.0          # Ambient

    # --- Target conditions (context-dependent) ---
    # PRETREATMENT / DRY FRACTIONATION:
    #   - Pre-RF: 13.5–15% moisture (tempering ensures uniform RF heating).
    #   - Post-RF: typically <10% (e.g. 9–10% in studies); seeds then re-tempered to
    #     ~12% for consistent de-hulling and milling before air classification.
    #   - LOX inactivation: >65°C for sufficient time; studies report outfeed ~84°C.
    #   - Minimize protein denaturation (target <15% globulin native loss).
    # This target is the typical post-RF moisture; downstream milling often re-tempers to ~12%.
    target_moisture_wb: float = 0.10             # 10% typical post-RF (re-temper to ~12% for milling)
    # For aggressive drying (storage), set to 0.03-0.05

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
    # removal.  A threshold of 25 C models the practical onset of
    # accelerated evaporation from RF heating.
    #
    # k_evap for whole seeds (GP-15 processes WHOLE beans/seeds, not
    # flour — Manual Chapter 1).  The manual (Chapter 5) states:
    #   - 1.0 kg/kWh for high surface-to-volume (fine powders)
    #   - 0.6 kg/kWh for low surface-to-volume (whole seeds)
    #
    # Whole seeds (6-8 mm dia) have ~100x lower surface-to-volume
    # than flour (50 um).  k_evap is reduced accordingly so the
    # drying rate matches the manual's 0.6 factor.
    # Default from utility_docs/calibration_latest.json (single source of truth).
    k_evap: float = field(default_factory=lambda: get_calibration_defaults()[1])
    T_evap_threshold_c: float = 25.0             # degC

    # --- Bed geometry (packed bed of whole seeds on the conveyor) ---
    bed_depth_m: float = 0.05                    # 50 mm typical
    bed_porosity: float = 0.40                   # Void fraction between whole seeds

    # --- Inter-bed thermal dispersion ─────────────────────────────
    # The extraction fan pulls air through the packed bed (porosity ~40%),
    # creating forced convection in the interstitial channels between
    # peas.  This convective transport dominates over solid conduction
    # and makes the vertical temperature profile much more uniform
    # than pure conduction would predict.
    #
    # k_dispersion is the effective additive conductivity from inter-bed
    # airflow, radiation between grains, and convective mixing.
    # Physically: k_disp ~ ρ_air * cp_air * U_interstitial * d_particle
    # For 8mm peas with moderate airflow (~0.3 m/s):
    #   k_disp ≈ 1.2 * 1000 * 0.3 * 0.008 ≈ 2.9 W/(m·K)
    #
    # Calibrated value: 2.10 W/(m·K), sensitivity dL/dp = 0.008
    # (insensitive — well-determined by temperature data).
    # Validated against Run#2 temperature strip measurements (77-82°C).
    # Default from utility_docs/calibration_latest.json (single source of truth).
    k_dispersion: float = field(default_factory=lambda: get_calibration_defaults()[3])

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
        """Effective thermal conductivity k(T, M) [W/(m*K)].

        Combines solid-framework conduction (parallel mixing rule) with
        inter-bed thermal dispersion from forced airflow through the
        packed bed.  The dispersion term (``k_dispersion``) models
        the convective heat transport in the interstitial channels
        between peas driven by the extraction fan, plus inter-particle
        radiation at elevated temperature.

        Without dispersion, the conduction-only k_eff (~0.15 W/m·K)
        gives a thermal diffusion time of ~85 min across the 35 mm
        bed — far longer than the 7.5 min residence time — producing
        an unrealistically steep vertical gradient (125 °C bottom,
        45 °C top).  The dispersion term raises k_eff to ~0.65 W/m·K,
        reducing diffusion time to ~15 min and matching the Run #2
        temperature strip measurements (77–82 °C at the outfeed).
        """
        k_solid = self.k_dry * (1.0 + self.k_moisture_beta * M_wb)
        k_air = 0.026  # W/(m*K) at ~50 degC
        k_conduction = k_solid * (1.0 - self.bed_porosity) + k_air * self.bed_porosity
        return k_conduction + self.k_dispersion

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

    The ``run_mass_kg`` field sets the total mass of material for
    the production run.  The simulation duration is computed from::

        duration = run_mass_kg / (rho_bulk * bed_depth * belt_width * belt_speed)

    This matches how the real machine operates: the operator loads
    a known mass, sets the belt speed and sizing gate, and the
    machine runs until all material has passed through the oven
    (Manual Chapter 5, Chapter 6).

    Example from actual Run#1 (25-Mar-2025):
        Material:       Whole Yellow Pea
        Run mass:       61 kg
        Feeder gap:     25 mm (bed depth from sizing gate)
        Belt speed:     0.2 m/min
        Electrode gap:  75 mm
        Start temp:     17.6 C
    """

    name: str = "default"
    recipe_number: int = 0                       # 1-30, 0 = manual mode

    # --- Process setpoints ---
    electrode_gap_mm: float = 80.0               # Gap setpoint
    belt_speed_m_per_min: float = 0.5            # Conveyor speed
    rf_power_enabled: bool = True

    # --- Run mass (Manual Chapter 5: "600 kg of product per hour") ---
    # Total mass of material for the production run [kg].
    # If > 0, the simulation duration is computed from mass / throughput.
    # If 0, the caller must specify duration_s explicitly.
    run_mass_kg: float = 0.0

    # --- Anode current protection ---
    # Run#1 PLC data shows: Ia 2nd Limit=1.7, Ia 1st Limit=1.5
    # These are the actual configured limits on the machine.
    # The manual's default MRH=2.6 / MRL=2.0 are the maximum
    # allowed values; the operator sets tighter limits per recipe.
    mrh_amps: float = 1.7                        # Meter Relay High (from Run#1 PLC)
    mrl_amps: float = 1.5                        # Meter Relay Low (from Run#1 PLC)

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
