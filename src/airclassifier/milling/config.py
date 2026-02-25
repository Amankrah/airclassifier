"""
Milling Configuration Dataclasses
==================================

Machine specifications, screen parameters, breakage properties, and recipe
definitions for the hammer mill simulation.

Process chain: Pretreatment (GP-15) -> Hammer Mill -> Air Classifier
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ============================================================================
#  Machine Configuration
# ============================================================================

@dataclass
class MillConfig:
    """Hammer mill machine parameters.

    Describes a horizontal-shaft hammer mill (pin mill / impact mill)
    for dry fractionation lines. Material from the pretreatment stage
    enters, is impacted by rotating hammers against the screen/housing,
    breaks, and exits through the screen apertures.
    """

    # --- Rotor / Drive ---
    rotor_rpm: float = 3000.0                    # Typical: 2000-4000 rpm
    rotor_diameter_m: float = 0.20               # Rotor hub outer diameter
    rotor_length_m: float = 0.30                 # Rotor active length (along shaft)
    shaft_diameter_m: float = 0.05               # Main shaft diameter

    # Power and motor
    motor_power_kw: float = 22.0                 # Typical: 15-55 kW
    motor_efficiency: float = 0.92               # Motor efficiency
    no_load_power_kw: float = 2.5                # Power at idle (bearing/windage)

    # --- Hammer configuration ---
    hammer_rows: int = 4                         # Rows along rotor length
    hammers_per_row: int = 4                     # Hammers per row (evenly spaced)
    hammer_mass_kg: float = 0.35                 # Mass per hammer
    hammer_length_m: float = 0.08                # From pivot to tip
    hammer_width_m: float = 0.05                 # Width (along rotor axis)
    hammer_thickness_m: float = 0.008            # Thickness
    hammer_clearance_m: float = 0.008            # Gap between hammer tip and screen

    # --- Screen ---
    # For protein-starch separation in legumes/pulses:
    #   - Protein bodies: 5-15 µm (fine fraction after air classification)
    #   - Starch granules: 20-40 µm (coarse fraction)
    #   - Air classification cut point: ~22 µm
    # Screen aperture determines product d50 (empirically ~12% of aperture):
    #   - 0.3 mm → d50 ~36 µm (excellent protein separation)
    #   - 0.5 mm → d50 ~60 µm (good for air classification)
    #   - 0.8 mm → d50 ~96 µm (coarse, may need re-milling)
    screen_arc_angle_deg: float = 180.0          # Screen wraps bottom half
    screen_inner_radius_m: float = 0.188         # Inside radius of screen (tip + clearance)
    screen_thickness_m: float = 0.003            # Screen plate thickness
    screen_aperture_mm: float = 0.5              # Hole size [mm] - 0.5mm optimal for protein separation
    screen_open_area: float = 0.40               # Open area fraction (0-1)

    # --- Housing ---
    housing_inner_radius_m: float = 0.20         # Casing inner radius
    housing_length_m: float = 0.40               # Housing length (includes end plates)
    housing_wall_thickness_m: float = 0.008      # Casing wall

    # --- Feed inlet ---
    feed_chute_width_m: float = 0.15             # Chute cross-section width
    feed_chute_height_m: float = 0.12            # Chute cross-section height
    feed_chute_length_m: float = 0.25            # Chute length
    feed_rate_kg_per_hr: float = 500.0           # Nominal feed rate

    # --- Discharge ---
    discharge_chute_width_m: float = 0.20
    discharge_chute_height_m: float = 0.15

    # --- Machine envelope ---
    machine_height_m: float = 0.80
    machine_width_m: float = 0.60
    machine_depth_m: float = 0.50

    @property
    def rotor_angular_velocity(self) -> float:
        """Rotor angular velocity [rad/s]."""
        return self.rotor_rpm * 2.0 * math.pi / 60.0

    @property
    def hammer_tip_speed(self) -> float:
        """Hammer tip linear velocity [m/s]."""
        # Tip radius = rotor radius + hammer length
        tip_radius = self.rotor_diameter_m / 2.0 + self.hammer_length_m
        return tip_radius * self.rotor_angular_velocity

    @property
    def total_hammers(self) -> int:
        """Total number of hammers."""
        return self.hammer_rows * self.hammers_per_row

    @property
    def screen_arc_angle_rad(self) -> float:
        """Screen arc angle in radians."""
        return math.radians(self.screen_arc_angle_deg)

    def tip_to_screen_clearance(self) -> float:
        """Clearance between hammer tip and screen inner surface [m]."""
        tip_radius = self.rotor_diameter_m / 2.0 + self.hammer_length_m
        return self.screen_inner_radius_m - tip_radius

    @property
    def estimated_d50_um(self) -> float:
        """Estimate product d50 [µm] based on screen aperture.

        Based on empirical data: d50 ≈ 12% of screen aperture for legume flour.
        Reference: Fine Grinding and Air Classification of Field Pea (ResearchGate)
        """
        return self.screen_aperture_mm * 1000 * 0.12  # 12% of aperture in µm

    def get_separation_quality(self) -> str:
        """Get protein separation quality assessment based on d50.

        Returns:
            Quality rating and recommendation string.
        """
        d50 = self.estimated_d50_um
        if d50 <= 40:
            return "Excellent - optimal for protein body liberation"
        elif d50 <= 70:
            return "Good - suitable for air classification"
        elif d50 <= 100:
            return "Moderate - may need secondary milling"
        else:
            return "Coarse - recommend finer screen or re-milling"


@dataclass
class ScreenConfig:
    """Screen-specific parameters for aperture and passage model.

    The screen is the curved perforated plate at the bottom of the
    mill chamber. Particles smaller than the aperture can pass through;
    larger particles are retained for further breakage.

    For protein-starch separation (legumes/pulses):
        - 0.3 mm: d50 ~36 µm - Excellent protein liberation
        - 0.5 mm: d50 ~60 µm - Good for air classification
        - 0.8 mm: d50 ~96 µm - Coarse, may need re-milling
    """

    aperture_mm: float = 0.5                     # Nominal hole diameter [mm] - optimal for protein separation
    open_area: float = 0.40                      # Open area fraction
    hole_shape: str = "round"                    # "round", "square", "slotted"

    # Passage model parameters
    passage_probability_factor: float = 1.0      # Tuning factor for passage rate
    size_ratio_threshold: float = 0.8            # d_particle/d_aperture below which passage is likely

    # Screen wear
    wear_factor: float = 1.0                     # 1.0 = new, increases with wear

    @property
    def aperture_m(self) -> float:
        """Aperture size in meters."""
        return self.aperture_mm / 1000.0

    @property
    def estimated_d50_um(self) -> float:
        """Estimate product d50 [µm] based on screen aperture.

        Empirical relationship: d50 ≈ 12% of aperture for legume flour.
        """
        return self.aperture_mm * 1000 * 0.12

    def passage_probability(self, particle_size_m: float) -> float:
        """Compute probability of passage for a given particle size.

        Args:
            particle_size_m: Particle characteristic size [m]

        Returns:
            Probability of passage (0-1)
        """
        ratio = particle_size_m / self.aperture_m
        if ratio > 1.0:
            return 0.0  # Cannot pass if larger than aperture
        elif ratio < self.size_ratio_threshold:
            # High probability of passage for small particles
            return self.passage_probability_factor * self.open_area
        else:
            # Decreasing probability as size approaches aperture
            t = (ratio - self.size_ratio_threshold) / (1.0 - self.size_ratio_threshold)
            return self.passage_probability_factor * self.open_area * (1.0 - t * t)


@dataclass
class BreakageParams:
    """Parameters for hammer-mill impact breakage (selection + breakage function).

    Implements high-fidelity hammer milling physics and kinetics for digital twin:
    - Selection function S(d, E): probability of breakage per hammer impact,
      driven by impact energy E (from hammer tip speed and particle mass) and
      particle size d (larger particles break more readily).
    - Breakage function B(d_daughter | d_parent): Gaudin–Schuhmann daughter size
      distribution from impact comminution (single-impact size reduction).
    - Impact energy is supplied by the impact kernel (hammer–particle collision;
      tip speed and restitution determine E). Breakage is applied only to
      impacted particles, so kinetics are fully coupled to hammer milling.
    """

    # Size classes for PSD (geometric progression)
    num_size_classes: int = 20
    d_min_um: float = 10.0                       # Smallest size class [um]
    d_max_um: float = 5000.0                     # Largest size class [um]

    # Selection function: S(d) = k * (d / d_ref)^alpha
    # Probability of breakage per impact for size d
    # Tuned for legume flour milling (pea, faba bean) to achieve d50 ~60µm
    selection_rate_constant: float = 0.40        # k: base selection rate (increased for fine flour)
    selection_size_exponent: float = 1.3         # alpha: larger particles break more easily
    selection_reference_size_um: float = 500.0   # d_ref: shifted for finer grinding

    # Breakage function: B(d_daughter | d_parent)
    # Cumulative mass fraction finer than d_daughter given breakage of d_parent
    # Uses Gaudin-Schuhmann: B = (d_daughter / d_parent)^gamma
    # Lower gamma = smaller daughter particles (more aggressive grinding)
    breakage_distribution_exponent: float = 0.55  # gamma: tuned for fine flour (d50 ~60µm)

    # Impact energy threshold
    min_impact_energy_j: float = 0.0005          # Below this, no breakage (lowered for fine particles)
    energy_to_breakage_factor: float = 8.0       # Converts impact energy to selection probability

    @property
    def size_classes_um(self) -> Tuple[float, ...]:
        """Size class boundaries in micrometers (geometric progression)."""
        ratio = (self.d_max_um / self.d_min_um) ** (1.0 / self.num_size_classes)
        return tuple(self.d_min_um * (ratio ** i) for i in range(self.num_size_classes + 1))

    @property
    def size_classes_m(self) -> Tuple[float, ...]:
        """Size class boundaries in meters."""
        return tuple(d * 1e-6 for d in self.size_classes_um)

    def selection_probability(self, d_um: float, impact_energy_j: float = 0.01) -> float:
        """Compute selection probability for a particle size.

        Args:
            d_um: Particle size [um]
            impact_energy_j: Impact energy [J]

        Returns:
            Probability of breakage (0-1)
        """
        if impact_energy_j < self.min_impact_energy_j:
            return 0.0

        # Size-dependent selection
        S = self.selection_rate_constant * (d_um / self.selection_reference_size_um) ** self.selection_size_exponent

        # Energy scaling
        energy_factor = min(1.0, impact_energy_j / self.min_impact_energy_j * self.energy_to_breakage_factor)

        return min(1.0, S * energy_factor)


# ============================================================================
#  Recipe — Operating setpoints
# ============================================================================

@dataclass
class MillRecipe:
    """Hammer mill operating recipe.

    Defines the operating setpoints for a milling run. Analogous to
    the GP-15 Recipe for pretreatment.

    Screen aperture guidelines for protein-starch separation:
        - 0.3-0.4 mm: Fine grinding, excellent protein liberation
        - 0.5 mm: Standard for air classification (d50 ~60 µm)
        - 0.8+ mm: Coarse grinding, may need re-milling
    """

    name: str = "default"
    recipe_number: int = 0                       # Recipe slot number

    # --- Operating setpoints ---
    rotor_rpm: float = 3000.0                    # Rotor speed setpoint
    screen_aperture_mm: float = 0.5              # Screen aperture [mm] - optimal for protein separation
    feed_rate_kg_per_hr: float = 500.0           # Target feed rate

    # --- Run parameters ---
    run_mass_kg: float = 0.0                     # Total mass for run (0 = continuous)
    run_duration_s: float = 60.0                 # Duration if run_mass_kg = 0

    # --- Control parameters ---
    power_limit_kw: float = 20.0                 # Max power before feed cutback
    temperature_limit_c: float = 80.0            # Max product temperature

    # --- Feed material (from pretreatment or synthetic) ---
    feed_moisture_wb: float = 0.12               # Feed moisture (wet basis)
    feed_temperature_c: float = 60.0             # Feed temperature
    feed_d50_um: float = 3000.0                  # Feed median size [um] (whole seeds ~3mm)
    # Mass of seeds (e.g. yellow peas) to feed into the mill for milling into powder [kg]
    seeds_feed_mass_kg: float = 0.0             # Total seeds mass; 0 = unlimited (continuous)
    # Internal: mass per particle for simulation (default ~150 mg per whole pea); not user-facing
    feed_particle_mass_kg: float = 0.00015

    @property
    def rotor_omega(self) -> float:
        """Rotor angular velocity [rad/s] from recipe rpm."""
        return self.rotor_rpm * 2.0 * math.pi / 60.0

    def throughput_kg_per_s(self) -> float:
        """Feed rate in kg/s."""
        return self.feed_rate_kg_per_hr / 3600.0


# ============================================================================
#  Outlet State — For pipeline integration
# ============================================================================

@dataclass
class MillingOutletState:
    """Outlet conditions from the hammer mill.

    This is the data passed to the downstream air classifier.
    Analogous to OutletState from pretreatment.
    """

    # --- Particle size distribution ---
    psd_mass_fractions: Tuple[float, ...] = field(default_factory=tuple)
    psd_size_classes_um: Tuple[float, ...] = field(default_factory=tuple)
    d10_um: float = 0.0
    d50_um: float = 0.0
    d90_um: float = 0.0

    # --- Flow rates ---
    throughput_kg_per_hr: float = 0.0
    mass_holdup_kg: float = 0.0                  # Material in mill at end

    # --- Thermal state (passthrough from feed) ---
    avg_temperature_c: float = 25.0
    avg_moisture_wb: float = 0.12

    # --- Power and energy ---
    power_kw: float = 0.0
    specific_energy_kwh_per_t: float = 0.0       # kWh per tonne

    # --- Residence time ---
    mean_residence_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "psd_mass_fractions": list(self.psd_mass_fractions),
            "psd_size_classes_um": list(self.psd_size_classes_um),
            "d10_um": self.d10_um,
            "d50_um": self.d50_um,
            "d90_um": self.d90_um,
            "throughput_kg_per_hr": self.throughput_kg_per_hr,
            "mass_holdup_kg": self.mass_holdup_kg,
            "avg_temperature_c": self.avg_temperature_c,
            "avg_moisture_wb": self.avg_moisture_wb,
            "power_kw": self.power_kw,
            "specific_energy_kwh_per_t": self.specific_energy_kwh_per_t,
            "mean_residence_time_s": self.mean_residence_time_s,
        }
