"""
Process Desirability Scoring
============================

Literature-backed Derringer–Suich desirability functions that evaluate
GP-15 RF pretreatment output for downstream protein separation (air
classification) and flavour improvement (lipoxygenase inactivation).

The GP-15 processes **whole seeds** (not flour).  The intact seed coat
prevents rapid moisture loss — Run#2 data (90 kg yellow pea, 35 mm bed,
0.2 m/min) shows only ~1.3 pp moisture reduction (11.8 % → 10.5 % wb).
This is thermal conditioning, not drying.

Each dimension maps a simulation KPI to a 0–1 desirability score:

    d = 0  →  completely unacceptable
    d = 1  →  ideal / on-target

The **overall desirability** is the geometric mean of all dimensions,
scaled 0–10 for display.

Scoring Dimensions
------------------
1. **Thermal Treatment** — target-range (outfeed temperature)
   Ideal outfeed temperature range for LOX inactivation while
   preserving protein.  Run#2 temperature strips: 77–82 °C;
   PLC steady-state outfeed: ~68–70 °C.

2. **Flavour Improvement** — larger-is-better (LOX inactivation)
   LOX fully inactivated at ≥65 °C for ≥4 min in pea.
   Run#2 achieved 24.2 min above 65 °C.
   Loganathan et al. 2009, *Food Chem.* 116, 1038-1043;
   Indrawati et al. 2001, *J. Food Sci.* 66, 686-693.

3. **Protein Preservation** — smaller-is-better (max temperature)
   Vicilin (7S) onset ~62–65 °C, peak ~71 °C; legumin (11S) onset
   ~76–78 °C, peak ~84 °C.  Run#2 strips: 77–82 °C (some vicilin
   denaturation expected, legumin mostly intact).
   Mession et al. 2013, *JAFC* 61, 1196-1204.

4. **Moisture Retention** — target-is-best (minimal drying)
   Whole seeds should retain moisture.  Run#2: 11.8 → 10.5 % wb
   (~1.3 pp loss).  Excessive drying of whole seeds indicates
   thermal damage to the seed coat.

5. **Energy Efficiency** — smaller-is-better (kWh per kg material)
   Energy per kg of material processed (NOT per kg water removed,
   since whole seeds barely dry).  Run#2: ~3.8 kWh for 90 kg
   = ~0.042 kWh/kg material.

References
----------
- Mession et al. (2013) *J. Agric. Food Chem.* 61, 1196-1204.
- Loganathan et al. (2009) *Food Chem.* 116, 1038-1043.
- Indrawati et al. (2001) *J. Food Sci.* 66, 686-693.
- Asavajaru et al. (2025) *Legume Sci.* e70030.
- Run#2 PLC data: 90 kg whole yellow pea, 25-Mar-2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================================
#  Desirability Profile — material-specific thresholds
# ============================================================================

@dataclass
class DesirabilityProfile:
    """Material-specific thresholds for desirability scoring.

    All temperatures in °C, moisture in wet-basis fraction.
    Calibrated against Run#2 PLC + NIR data (25-Mar-2025).
    """

    name: str = "yellow_pea"

    # ── Thermal Treatment (target-range for outfeed temperature) ──
    # The ideal outfeed temperature window for LOX inactivation
    # while preserving protein native structure.
    # Run#2: PLC steady-state ~68-70°C, strips 77-82°C.
    thermal_ideal_low_c: float = 65.0       # LOX kill threshold
    thermal_ideal_high_c: float = 82.0      # below legumin peak
    thermal_no_effect_c: float = 40.0       # no treatment effect
    thermal_excessive_c: float = 100.0      # excessive / scorching

    # ── Flavour (LOX inactivation temperature threshold) ──
    # LOX fully inactivated at ≥65°C for ≥4 min (pea).
    lox_kill_c: float = 65.0
    lox_no_effect_c: float = 40.0

    # ── Protein Preservation (max temperature) ──
    # Vicilin (7S) peak ~71°C; legumin (11S) peak ~84°C.
    # Run#2 strips: 77-82°C → some vicilin denaturation but legumin safe.
    protein_safe_c: float = 71.0            # vicilin peak
    protein_denatured_c: float = 90.0       # well above legumin peak

    # ── Moisture Retention (whole seeds) ──
    # Whole seeds should retain moisture — only ~1.3 pp loss expected.
    # Run#2: 11.8% → 10.5% wb.  >3 pp loss indicates seed coat damage.
    moisture_loss_ideal_pp: float = 0.0     # no loss = perfect
    moisture_loss_acceptable_pp: float = 2.0  # up to 2 pp is fine
    moisture_loss_excessive_pp: float = 5.0   # seed coat damage

    # ── Energy Efficiency (kWh per kg material processed) ──
    # Run#2: ~3.8 kWh for 90 kg = 0.042 kWh/kg.
    # GP-15 rated 15 kW, typical throughput 200-600 kg/h.
    energy_ideal_kwh_per_kg: float = 0.04   # excellent efficiency
    energy_poor_kwh_per_kg: float = 0.15    # poor efficiency


# Pre-configured profiles per material
PROFILES: Dict[str, DesirabilityProfile] = {
    "yellow_pea": DesirabilityProfile(
        name="yellow_pea",
        thermal_ideal_low_c=65.0,
        thermal_ideal_high_c=82.0,
        thermal_no_effect_c=40.0,
        thermal_excessive_c=100.0,
        lox_kill_c=65.0,
        lox_no_effect_c=40.0,
        protein_safe_c=71.0,        # vicilin peak ~71°C
        protein_denatured_c=90.0,   # above legumin peak ~84°C
        moisture_loss_ideal_pp=0.0,
        moisture_loss_acceptable_pp=2.0,
        moisture_loss_excessive_pp=5.0,
        energy_ideal_kwh_per_kg=0.04,
        energy_poor_kwh_per_kg=0.15,
    ),
    "faba_bean": DesirabilityProfile(
        name="faba_bean",
        thermal_ideal_low_c=65.0,
        thermal_ideal_high_c=90.0,   # faba has higher thermal tolerance
        thermal_no_effect_c=40.0,
        thermal_excessive_c=110.0,
        lox_kill_c=65.0,
        lox_no_effect_c=40.0,
        protein_safe_c=90.0,         # faba vicilin Td ~90°C
        protein_denatured_c=105.0,   # faba legumin Td ~100°C + margin
        moisture_loss_ideal_pp=0.0,
        moisture_loss_acceptable_pp=2.0,
        moisture_loss_excessive_pp=5.0,
        energy_ideal_kwh_per_kg=0.04,
        energy_poor_kwh_per_kg=0.15,
    ),
    "red_lentil": DesirabilityProfile(
        name="red_lentil",
        thermal_ideal_low_c=65.0,
        thermal_ideal_high_c=82.0,
        thermal_no_effect_c=40.0,
        thermal_excessive_c=100.0,
        lox_kill_c=65.0,
        lox_no_effect_c=40.0,
        protein_safe_c=75.0,         # lentil globulins similar to pea
        protein_denatured_c=92.0,
        moisture_loss_ideal_pp=0.0,
        moisture_loss_acceptable_pp=2.0,
        moisture_loss_excessive_pp=5.0,
        energy_ideal_kwh_per_kg=0.04,
        energy_poor_kwh_per_kg=0.15,
    ),
}


# ============================================================================
#  Desirability Result
# ============================================================================

@dataclass
class DesirabilityResult:
    """Per-dimension and overall desirability scores (all 0–1)."""

    d_thermal: float = 0.0
    d_flavour: float = 0.0
    d_protein: float = 0.0
    d_moisture: float = 0.0
    d_energy: float = 0.0

    overall: float = 0.0           # geometric mean of all dimensions
    overall_10: float = 0.0        # overall × 10 for display (0–10 scale)

    # Human-readable labels for each dimension
    labels: Dict[str, str] = field(default_factory=lambda: {
        "d_thermal": "Thermal Treatment",
        "d_flavour": "Flavour Improvement",
        "d_protein": "Protein Preservation",
        "d_moisture": "Moisture Retention",
        "d_energy": "Energy Efficiency",
    })

    def dimensions(self) -> Dict[str, float]:
        """Return {label: score} for all dimensions."""
        return {
            self.labels["d_thermal"]: self.d_thermal,
            self.labels["d_flavour"]: self.d_flavour,
            self.labels["d_protein"]: self.d_protein,
            self.labels["d_moisture"]: self.d_moisture,
            self.labels["d_energy"]: self.d_energy,
        }


# ============================================================================
#  Desirability Functions (Derringer–Suich)
# ============================================================================

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(x, hi))


def _target_is_best(
    y: float,
    target: float,
    lower: float,
    upper: float,
    s: float = 1.0,
    t: float = 1.0,
) -> float:
    """Derringer–Suich nominal (target-is-best) desirability.

    d = ((y - L)/(T - L))^s   if  L ≤ y ≤ T
    d = ((U - y)/(U - T))^t   if  T ≤ y ≤ U
    d = 0                     otherwise
    """
    if y <= lower or y >= upper:
        return 0.0
    if y <= target:
        denom = target - lower
        if denom <= 0:
            return 1.0
        return _clamp(((y - lower) / denom) ** s)
    else:
        denom = upper - target
        if denom <= 0:
            return 1.0
        return _clamp(((upper - y) / denom) ** t)


def _target_range(
    y: float,
    ideal_low: float,
    ideal_high: float,
    lower: float,
    upper: float,
    s: float = 1.0,
    t: float = 1.0,
) -> float:
    """Desirability for a target RANGE (plateau between ideal_low and ideal_high).

    d = 1                           if  ideal_low ≤ y ≤ ideal_high
    d = ((y - L)/(ideal_low - L))^s if  L < y < ideal_low
    d = ((U - y)/(U - ideal_high))^t if ideal_high < y < U
    d = 0                           if  y ≤ L or y ≥ U
    """
    if y <= lower or y >= upper:
        return 0.0
    if ideal_low <= y <= ideal_high:
        return 1.0
    if y < ideal_low:
        denom = ideal_low - lower
        if denom <= 0:
            return 1.0
        return _clamp(((y - lower) / denom) ** s)
    else:  # y > ideal_high
        denom = upper - ideal_high
        if denom <= 0:
            return 1.0
        return _clamp(((upper - y) / denom) ** t)


def _larger_is_better(
    y: float,
    lower: float,
    target: float,
    s: float = 1.0,
) -> float:
    """Derringer–Suich larger-is-better desirability.

    d = 0                     if  y ≤ L
    d = ((y - L)/(T - L))^s   if  L < y < T
    d = 1                     if  y ≥ T
    """
    if y >= target:
        return 1.0
    if y <= lower:
        return 0.0
    denom = target - lower
    if denom <= 0:
        return 1.0
    return _clamp(((y - lower) / denom) ** s)


def _smaller_is_better(
    y: float,
    target: float,
    upper: float,
    t: float = 1.0,
) -> float:
    """Derringer–Suich smaller-is-better desirability.

    d = 1                     if  y ≤ T
    d = ((U - y)/(U - T))^t   if  T < y < U
    d = 0                     if  y ≥ U
    """
    if y <= target:
        return 1.0
    if y >= upper:
        return 0.0
    denom = upper - target
    if denom <= 0:
        return 1.0
    return _clamp(((upper - y) / denom) ** t)


# ============================================================================
#  Main Scoring Function
# ============================================================================

def score_desirability(
    outfeed_temperature_c: float,
    max_temperature_c: float,
    outfeed_moisture_wb: float,
    initial_moisture_wb: float,
    energy_kwh: float,
    run_mass_kg: float,
    profile: Optional[DesirabilityProfile] = None,
) -> DesirabilityResult:
    """Score a GP-15 simulation run for downstream process suitability.

    Designed for **whole seed thermal conditioning** — the GP-15
    heats seeds for LOX inactivation and protein conditioning, not
    for drying (whole seeds barely lose moisture).

    Args:
        outfeed_temperature_c: Average outfeed temperature [°C].
        max_temperature_c: Maximum material temperature reached [°C].
        outfeed_moisture_wb: Average outfeed moisture (wet basis fraction).
        initial_moisture_wb: Initial moisture before treatment (wet basis fraction).
        energy_kwh: Total RF energy consumed [kWh].
        run_mass_kg: Total mass of material processed [kg].
        profile: Material-specific thresholds.  Defaults to yellow_pea.

    Returns:
        :class:`DesirabilityResult` with per-dimension and overall scores.
    """
    if profile is None:
        profile = PROFILES["yellow_pea"]

    # 1. Thermal Treatment — target-range (outfeed temperature)
    #    Ideal: outfeed T in [65, 82]°C for yellow pea.
    #    Run#2 PLC steady-state: 68-70°C; strips: 77-82°C.
    d_thermal = _target_range(
        y=outfeed_temperature_c,
        ideal_low=profile.thermal_ideal_low_c,
        ideal_high=profile.thermal_ideal_high_c,
        lower=profile.thermal_no_effect_c,
        upper=profile.thermal_excessive_c,
    )

    # 2. Flavour Improvement — larger-is-better (outfeed temperature)
    #    LOX fully killed at ≥65°C (pea, 4 min exposure).
    #    Run#2 achieved 24.2 min above 65°C — excellent.
    d_flavour = _larger_is_better(
        y=outfeed_temperature_c,
        lower=profile.lox_no_effect_c,
        target=profile.lox_kill_c,
    )

    # 3. Protein Preservation — smaller-is-better (max temperature)
    #    Run#2 strips: 77-82°C — some vicilin denaturation, legumin safe.
    d_protein = _smaller_is_better(
        y=max_temperature_c,
        target=profile.protein_safe_c,
        upper=profile.protein_denatured_c,
    )

    # 4. Moisture Retention — smaller-is-better (moisture loss)
    #    Whole seeds should retain moisture.  Run#2: ~1.3 pp loss.
    #    Excessive loss (>5 pp) indicates seed coat thermal damage.
    moisture_loss_pp = max(0.0, (initial_moisture_wb - outfeed_moisture_wb) * 100.0)
    d_moisture = _smaller_is_better(
        y=moisture_loss_pp,
        target=profile.moisture_loss_acceptable_pp,
        upper=profile.moisture_loss_excessive_pp,
    )

    # 5. Energy Efficiency — smaller-is-better (kWh per kg material)
    #    NOT per kg water removed (whole seeds barely dry).
    #    Run#2: ~3.8 kWh / 90 kg ≈ 0.042 kWh/kg.
    if run_mass_kg > 0:
        energy_per_kg = energy_kwh / run_mass_kg
    else:
        energy_per_kg = profile.energy_poor_kwh_per_kg

    d_energy = _smaller_is_better(
        y=energy_per_kg,
        target=profile.energy_ideal_kwh_per_kg,
        upper=profile.energy_poor_kwh_per_kg,
    )

    # Overall desirability — geometric mean
    scores = [d_thermal, d_flavour, d_protein, d_moisture, d_energy]

    # Geometric mean: if any dimension is 0, overall is 0
    product = 1.0
    n = len(scores)
    for s in scores:
        if s <= 0:
            product = 0.0
            break
        product *= s

    if product > 0:
        overall = product ** (1.0 / n)
    else:
        overall = 0.0

    return DesirabilityResult(
        d_thermal=round(d_thermal, 4),
        d_flavour=round(d_flavour, 4),
        d_protein=round(d_protein, 4),
        d_moisture=round(d_moisture, 4),
        d_energy=round(d_energy, 4),
        overall=round(overall, 4),
        overall_10=round(overall * 10, 2),
    )
