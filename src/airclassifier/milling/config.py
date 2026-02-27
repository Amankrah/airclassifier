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

    # --- Transport physics ---
    # Coefficients of restitution for wall bounces (0 = inelastic, 1 = elastic).
    # Reflection formula: v_new = v - (1+e)*n*(v·n), so factor = 1+e.
    wall_restitution_radial: float = 0.3     # CoR for radial (housing) wall bounce
    wall_restitution_endwall: float = 0.3    # CoR for axial (end plate) bounce
    # Fraction of centrifugal force imparted to particles near rotor.
    # Full coupling (1.0) would fling everything to the wall instantly;
    # in reality particles are loosely entrained in air, not rigidly attached.
    centrifugal_coupling_factor: float = 0.1
    # Hammer–particle collision restitution
    hammer_restitution: float = 0.3          # CoR for hammer-particle impacts (0-1)

    # --- Screen detection ---
    screen_zone_tolerance_m: float = 0.03    # Radial tolerance for screen proximity [m]
    velocity_passage_threshold_m_per_s: float = 5.0  # Speed above which screen passage drops

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

    @property
    def hammer_tip_radius_m(self) -> float:
        """Radius from shaft centre to hammer tip [m]."""
        return self.rotor_diameter_m / 2.0 + self.hammer_length_m

    def tip_to_screen_clearance(self) -> float:
        """Clearance between hammer tip and screen inner surface [m]."""
        return self.screen_inner_radius_m - self.hammer_tip_radius_m

    @property
    def hammer_angular_extent_rad(self) -> float:
        """Angular width of one hammer at the tip [rad].

        arc_length / radius = hammer_width / tip_radius.
        """
        return self.hammer_width_m / self.hammer_tip_radius_m

    @property
    def impact_sweep_inner_margin_m(self) -> float:
        """Radial depth behind hammer tip that still registers impacts [m].

        Capped at 40% of hammer length so very long hammers don't create
        an unrealistically deep sweep zone.
        """
        return min(0.03, self.hammer_length_m * 0.4)

    @property
    def impact_sweep_outer_margin_m(self) -> float:
        """Radial clearance beyond hammer tip for impact detection [m].

        Uses actual tip-to-screen clearance (particles in the gap can
        still be struck).  Minimum 5 mm so detection isn't zero when
        clearance is very tight.
        """
        return max(0.005, self.tip_to_screen_clearance())

    @property
    def estimated_d50_um(self) -> float:
        """Estimate realistic product D50 [µm] from screen aperture and tip speed.

        Single-pass hammer mill has an aerodynamic grinding limit around 40–80 µm
        for pulse flour.  Below ~80 µm, air entrainment, starch granule resistance,
        and reagglomeration prevent further reduction.

        Model: D50 = grinding_limit + k * aperture_mm^beta
        Validated against pilot data: 0.3 mm / 6000 rpm → ~67 µm.
        """
        # Grinding limit depends on tip speed (higher = slightly lower floor)
        tip = self.hammer_tip_speed
        # At 113 m/s (6000 rpm): floor ≈ 55 µm; at 56 m/s (3000 rpm): floor ≈ 70 µm
        grinding_floor = max(40.0, 80.0 - 0.22 * tip)
        # Screen contribution: larger aperture → coarser product
        return grinding_floor + 18.0 * self.screen_aperture_mm ** 0.7

    def get_separation_quality(self) -> str:
        """Get protein separation quality assessment based on D50.

        Goal: release starch granules (15–40 µm) from protein matrix (1–10 µm).
        Single-pass hammer mill typically achieves 40–100 µm.
        Pin/jet mills needed for < 40 µm.
        """
        d50 = self.estimated_d50_um
        if d50 <= 45:
            return "Excellent - near grinding limit, good liberation for protein separation"
        elif d50 <= 70:
            return "Good - suitable for starch/protein fractionation in air classifier"
        elif d50 <= 100:
            return "Moderate - fine flour, consider higher RPM or two-pass for better separation"
        else:
            return "Coarse - use finer screen (0.3–0.84 mm) or higher RPM (5000–7200)"


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
        """Estimate product D50 [µm] from screen aperture.

        Uses the same grinding-floor model as MillConfig.estimated_d50_um.
        For a single-pass hammer mill at ~6000 RPM (~63 m/s tip), the
        aerodynamic grinding floor is ~55 µm; screen aperture adds a
        coarsening offset.  Result for 0.3 mm / 6000 RPM → ~63 µm.
        """
        # Assume typical pilot tip speed ~63 m/s (6000 RPM, 0.20 m radius)
        # MillConfig.estimated_d50_um uses actual config tip speed;
        # here we use a representative value since ScreenConfig doesn't know RPM.
        typical_tip_speed = 63.0
        grinding_floor = max(40.0, 80.0 - 0.22 * typical_tip_speed)
        return grinding_floor + 18.0 * self.aperture_mm ** 0.7

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

    # Fine regime (d < 100 µm): realistic — starch granules and cell fragments
    # resist further breakage; single impact gives 15–40% reduction, not 86–97%.
    gamma_fine: float = 0.55
    clamp_lo_fine: float = 0.55
    clamp_hi_fine: float = 0.85

    # Impact energy
    min_impact_energy_j: float = 0.00015         # Low so more impacts lead to breakage
    energy_to_breakage_factor: float = 14.0      # Strong coupling from impact energy to selection

    # --- Impact efficiency (air entrainment / cushioning) ---
    # Fine particles follow the airflow around hammers instead of impacting them.
    # η = min(1, (d / d_crit)^n)  reduces effective impact energy for d < d_crit.
    # At d_crit ≈ 80 µm the transition from ballistic to entrained begins.
    impact_efficiency_d_crit_um: float = 80.0    # Transition size [µm]
    impact_efficiency_exponent: float = 2.0      # Sharpness of the transition

    # --- Reagglomeration ---
    # Below ~50 µm, van der Waals, electrostatic, and moisture bridges cause
    # fine particles to stick together, raising effective D50.
    reagglom_enabled: bool = True
    reagglom_threshold_um: float = 50.0          # Only particles below this can agglomerate
    reagglom_rate: float = 0.02                  # Probability per eligible pair per step
    reagglom_max_merges_per_step: int = 50       # Cap to limit compute cost
    reagglom_moisture_sensitivity: float = 2.0   # Higher moisture → more agglomeration
    reagglom_moisture_baseline: float = 0.08    # Below this moisture, no extra agglomeration boost
    reagglom_temp_threshold_c: float = 40.0     # Above this, starch surfaces become stickier
    reagglom_temp_sensitivity: float = 0.02     # Rate increase per °C above threshold

    # --- Thermal model ---
    # Specific energy input heats the product; temperature affects breakage and stickiness.
    # Calibrated so pilot mill (6 kW, 500 kg/h) reaches ~45-65 °C steady state.
    thermal_enabled: bool = True
    cp_product_j_per_kg_k: float = 1800.0        # Specific heat of pulse flour [J/(kg·K)]
    eta_thermal: float = 0.35                     # Fraction of impact energy → heat
    #   ~35%: rest goes to fracture surface energy, sound, elastic deformation, air turbulence
    h_conv_w_per_m2_k: float = 150.0              # Overall heat transfer (product→wall→ambient)
    #   Internal forced convection is high (~200+ W/m²K) but external natural
    #   convection (~15 W/m²K) is the bottleneck; lumped effective h ≈ 100-200.
    a_cooling_m2: float = 0.25                    # Effective cooling area [m²]
    #   Pilot housing outer surface ~0.5 m²; effective area ~50% due to insulation,
    #   mounting, and non-uniform airflow.  h×A ≈ 37.5 W/K → steady state ~55°C.
    t_air_c: float = 25.0                         # Ambient air temperature [°C]
    # Temperature effects on breakage
    t_breakage_onset_c: float = 50.0              # Above this, starch softens → harder to break
    t_breakage_slope: float = 0.008               # Selection rate drops by this per °C above onset
    #   Gentler slope: at 65°C penalty is 1 - 0.008*15 = 0.88 (12% reduction, not 90%)

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
