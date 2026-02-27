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
    for dry fractionation lines. For yellow pea flour for air classification:
    rotor speeds 3,000–7,200 rpm (BAKERpedia); tip speed ~102 m/s with 0.84 mm
    screen gives median ~98 µm, low starch damage (ResearchGate). Goal: break
    cotyledon to release starch granules (15–40 µm) from protein matrix (1–10 µm).
    """

    # --- Rotor / Drive ---
    # Yellow pea: 3,000 rpm standard; 6,000–7,200 rpm for very fine flour / protein enrichment (BAKERpedia)
    rotor_rpm: float = 3000.0                    # Typical: 3,000–7,200 rpm for pea flour
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
    # Yellow pea flour for air classification: small screens 0.84–2 mm (ResearchGate).
    # Tip speed ~102 m/s + 0.84 mm screen → median ~98 µm, low starch damage.
    # NIH: 0.75 mm → D50 ~23.7 µm; 2.0 mm → D50 ~31.1 µm. Dehulled peas improve yield/protein.
    screen_arc_angle_deg: float = 180.0          # Screen wraps bottom half
    screen_inner_radius_m: float = 0.188         # Inside radius of screen (tip + clearance)
    screen_thickness_m: float = 0.003            # Screen plate thickness
    screen_aperture_mm: float = 0.75              # Hole size [mm]; 0.75–0.84 mm fine, up to 2 mm
    screen_open_area: float = 0.40               # Open area fraction (0-1)
    screen_size_ratio_threshold: float = 0.06     # Passage: below d/aperture = full; above = taper (retain coarse for breakage)

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
        """Hammer tip linear velocity [m/s]. Research: ~102 m/s for fine pea flour (0.84 mm screen, D50 ~98 µm)."""
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
        """Estimate product D50 [µm] from screen aperture (yellow pea).

        NIH: 0.75 mm → D50 ~23.7 µm; 2.0 mm → ~31.1 µm. Fit: D50_µm ≈ 17.4 + 6.84 * aperture_mm.
        ResearchGate: tip speed ~102 m/s + 0.84 mm screen → median ~98 µm (low starch damage).
        """
        return 17.4 + 6.84 * self.screen_aperture_mm

    def get_separation_quality(self) -> str:
        """Get protein separation quality assessment based on D50.

        Goal: release starch granules (15–40 µm) from protein matrix (1–10 µm).
        Fine flour improves air classification efficiency (BAKERpedia, ResearchGate).
        """
        d50 = self.estimated_d50_um
        if d50 <= 31:
            return "Excellent - D50 in 24–31 µm range for protein separation"
        elif d50 <= 55:
            return "Good - suitable for starch/protein fractionation"
        elif d50 <= 98:
            return "Moderate - tip speed ~102 m/s + 0.84 mm screen can reach ~98 µm (ResearchGate)"
        elif d50 <= 114:
            return "Moderate - consider finer screen (0.75–0.84 mm) or higher rpm"
        else:
            return "Coarse - use 0.84–2 mm screen, 3,000–7,200 rpm for pea flour"


@dataclass
class ScreenConfig:
    """Screen-specific parameters for aperture and passage model.

    The screen is the curved perforated plate at the bottom of the
    mill chamber. Particles smaller than the aperture can pass through;
    larger particles are retained for further breakage.

    Yellow pea flour (NIH): 0.75 mm → D50 ~23.7 µm; 2.0 mm → D50 ~31.1 µm.
    Default 0.75 mm targets protein separation (fine flour).
    """

    aperture_mm: float = 0.75                   # Nominal hole diameter [mm] — 0.75 mm → D50 ~24 µm (NIH)
    open_area: float = 0.40                      # Open area fraction
    hole_shape: str = "round"                    # "round", "square", "slotted"

    # Passage model parameters (for protein separation: favor fine particles passing)
    passage_probability_factor: float = 1.0      # Tuning factor for passage rate
    # Below this ratio (d/aperture), passage prob is max; above, it tapers. Lower = finer discharge D50 (protein separation).
    size_ratio_threshold: float = 0.06           # 0.06 → particles < ~45 µm pass easily (0.75 mm); coarser retained for breakage

    # Screen wear
    wear_factor: float = 1.0                     # 1.0 = new, increases with wear

    @property
    def aperture_m(self) -> float:
        """Aperture size in meters."""
        return self.aperture_mm / 1000.0

    @property
    def estimated_d50_um(self) -> float:
        """Estimate product D50 [µm] from screen aperture (yellow pea, NIH).

        D50_µm ≈ 17.4 + 6.84 * aperture_mm (0.75 mm → ~23.7 µm, 2 mm → ~31.1 µm).
        """
        return 17.4 + 6.84 * self.aperture_mm

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
    # Yellow pea: 1–10 µm protein bodies, 10–55 µm starch/cell, 55–470 µm cotyledon (NIH).
    num_size_classes: int = 20
    d_min_um: float = 5.0                        # Smallest size [µm] — allow protein-body range (1–10 µm)
    d_max_um: float = 5000.0                     # Largest size class [um]

    # Selection function: S(d) = k * (d / d_ref)^alpha
    # Tuned so discharge D50 reaches 24–43 µm (NIH protein separation). Run 5–10 s for steady state.
    selection_rate_constant: float = 1.0        # k: cap 1.0 — most impacted particles break
    selection_size_exponent: float = 1.4          # alpha: larger particles break more easily
    selection_reference_size_um: float = 100.0   # d_ref: 100 µm+ break readily

    # Breakage function: Gaudin–Schuhmann; lower gamma → smaller daughters per break.
    breakage_distribution_exponent: float = 0.26  # gamma medium — very aggressive for fine flour

    # --- Size-dependent breakage regimes (legume comminution) ---
    regime_coarse_threshold_m: float = 1.0e-3    # Above = coarse
    regime_fine_threshold_m: float = 1.0e-4      # Below = fine (100 µm)

    # Coarse regime (d > 1 mm)
    gamma_coarse: float = 1.0
    clamp_lo_coarse: float = 0.30
    clamp_hi_coarse: float = 0.60

    # Medium regime (100 µm–1 mm): strong size reduction
    clamp_lo_medium: float = 0.10
    clamp_hi_medium: float = 0.26

    # Fine regime (d < 100 µm): target 24–43 µm; lower clamps → more fines so D50 can drop below ~70 µm
    gamma_fine: float = 0.12
    clamp_lo_fine: float = 0.03
    clamp_hi_fine: float = 0.14

    # Impact energy
    min_impact_energy_j: float = 0.00015         # Low so more impacts lead to breakage
    energy_to_breakage_factor: float = 14.0      # Strong coupling from impact energy to selection

    # Multi-fragment breakage (mass-conserving fragmentation)
    # When enabled, each breakage event produces 2-N fragments whose masses
    # sum exactly to the parent mass. The existing kernel computes the primary
    # daughter; secondary fragments are generated as a CPU post-processing step.
    enable_multi_fragment: bool = True
    max_fragments_per_event: int = 6             # N_max: hard cap on total fragments per event
    fragment_count_coefficient: float = 2.0      # C_n: base coefficient for fragment count
    fragment_count_size_exp: float = 0.5         # alpha_n: size-ratio exponent for fragment count
    fragment_count_energy_exp: float = 0.3       # beta_n: energy exponent for fragment count
    fragment_position_noise_m: float = 0.001     # Spatial jitter for secondary fragments [m]
    fragment_velocity_noise_m_per_s: float = 0.5 # Velocity jitter for secondary fragments [m/s]
    max_particle_count: int = 50_000             # Safety cap on total particles in simulation

    @property
    def size_classes_um(self) -> Tuple[float, ...]:
        """Size class boundaries in micrometers (geometric progression)."""
        ratio = (self.d_max_um / self.d_min_um) ** (1.0 / self.num_size_classes)
        return tuple(self.d_min_um * (ratio ** i) for i in range(self.num_size_classes + 1))

    @property
    def size_classes_m(self) -> Tuple[float, ...]:
        """Size class boundaries in meters."""
        return tuple(d * 1e-6 for d in self.size_classes_um)

    @property
    def gamma_medium(self) -> float:
        """Medium regime gamma (alias for breakage_distribution_exponent)."""
        return self.breakage_distribution_exponent

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

    Defines the operating setpoints for a milling run. For yellow pea flour
    (air classification): rotor 3,000–7,200 rpm (BAKERpedia); tip speed ~102 m/s
    with 0.84 mm screen → median ~98 µm, low starch damage (ResearchGate). Screen
    0.84–2 mm for fine grinding; dehulled peas recommended.
    """

    name: str = "default"
    recipe_number: int = 0                       # Recipe slot number

    # --- Operating setpoints ---
    rotor_rpm: float = 3000.0                    # 3,000–7,200 rpm; higher for finer flour / protein enrichment
    screen_aperture_mm: float = 0.75             # 0.84–2 mm for pea flour; 0.75 mm → D50 ~24 µm (NIH), 0.84 mm → ~98 µm
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
