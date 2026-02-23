"""
Run#2 validation targets and comparison helpers.

Compares simulation outputs to physical Run#2 (90 kg, 35 mm bed, 75 mm gap,
0.2 m/min, 17°C, 11.81% wb) NIR/PLC/strip targets. Used to audit coupling,
config, controller, and particle behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Run#2 physical targets (NIR outfeed moisture, temperature strips, PLC gap/Ia)
RUN2_TARGETS = {
    "outfeed_moisture_wb": 0.1053,       # NIR avg of 15 samples
    "outfeed_temp_c_min": 77.0,          # temperature strips range
    "outfeed_temp_c_max": 82.0,
    "gap_peak_mm": 94.1,
    "ia_steady_min": 1.5,                 # Ia steady band (MRL–MRH)
    "ia_steady_max": 1.7,
    "mrh_amps": 1.7,
    "mrl_amps": 1.5,
}


@dataclass
class ValidationResult:
    """Result of comparing sim to Run#2 targets."""
    outfeed_moisture_ok: bool
    outfeed_temp_ok: bool
    gap_triggered_ok: bool
    ia_steady_ok: bool
    mass_balance_ok: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Run#2 validation:",
            f"  Outfeed moisture:  {'OK' if self.outfeed_moisture_ok else 'MISMATCH'} ({self.details.get('outfeed_moisture_note', '')})",
            f"  Outfeed temp:      {'OK' if self.outfeed_temp_ok else 'MISMATCH'} ({self.details.get('outfeed_temp_note', '')})",
            f"  Gap control:        {'OK' if self.gap_triggered_ok else 'NO OPENING'} ({self.details.get('gap_note', '')})",
            f"  Ia steady band:    {'OK' if self.ia_steady_ok else 'LOW'} ({self.details.get('ia_note', '')})",
            f"  Mass balance:      {'OK' if self.mass_balance_ok else 'CHECK'} ({self.details.get('mass_note', '')})",
        ]
        return "\n".join(lines)


def compare_sim_to_run2(
    outlet: Any,
    *,
    ts: Optional[Dict[str, Any]] = None,
    dispatched_kg: float = 0.0,
    collected_kg: Optional[float] = None,
    tolerance_temp_c: float = 3.0,
    tolerance_moisture_pp: float = 0.5,
    tolerance_mass_pct: float = 2.0,
    prefer_particle_stats: bool = True,
) -> ValidationResult:
    """Compare simulation results to Run#2 physical targets.

    Args:
        outlet: Object with at least sensor_temperature_c, avg_moisture_wb
                (e.g. OutletState from get_outlet_conditions()).
        ts: Optional time-series dict with electrode_gap_mm, anode_current_a.
        dispatched_kg: Mass dispatched (run_mass_kg).
        collected_kg: Mass collected (particles.collected_mass_kg); if None, skip mass check.
        tolerance_temp_c: Allowable error on outfeed temp [°C].
        tolerance_moisture_pp: Allowable error on outfeed moisture [percentage points].
        tolerance_mass_pct: Allowable mass balance error [%].
        prefer_particle_stats: If True, use particle-based statistics (Lagrangian)
            over grid-based (Eulerian) when available. Particle stats match the
            physical measurement methodology from Pea RF summary(Run#2).csv.

    Returns:
        ValidationResult with pass/fail flags and short notes.
    """
    details: Dict[str, Any] = {}
    t_min = RUN2_TARGETS["outfeed_temp_c_min"]
    t_max = RUN2_TARGETS["outfeed_temp_c_max"]
    m_tgt = RUN2_TARGETS["outfeed_moisture_wb"]
    gap_peak_tgt = RUN2_TARGETS["gap_peak_mm"]
    ia_lo = RUN2_TARGETS["ia_steady_min"]
    ia_hi = RUN2_TARGETS["ia_steady_max"]

    # Check for particle-based statistics (preferred - matches physical methodology)
    particle_count = getattr(outlet, "particle_count", 0)
    use_particles = prefer_particle_stats and particle_count > 0

    # Outfeed temperature (sensor = 75th percentile, matches strips)
    # Prefer particle P75 temperature when available - this matches
    # temperature strip measurement methodology from physical runs.
    if use_particles:
        T_sensor = getattr(outlet, "particle_p75_temperature_c", 0.0)
        source = "particle"
    else:
        T_sensor = getattr(outlet, "sensor_temperature_c", None) or getattr(outlet, "avg_temperature_c", 0.0)
        source = "grid"

    T_ok = t_min - tolerance_temp_c <= T_sensor <= t_max + tolerance_temp_c
    details["outfeed_temp_sim_c"] = T_sensor
    details["outfeed_temp_target_c"] = f"{t_min}-{t_max}"
    details["outfeed_temp_source"] = source
    if T_ok:
        details["outfeed_temp_note"] = f"{T_sensor:.1f}°C (target {t_min}-{t_max}°C, {source})"
    else:
        status = "too cold" if T_sensor < t_min else "too hot"
        details["outfeed_temp_note"] = f"{T_sensor:.1f}°C vs {t_min}-{t_max}°C ({status}, {source})"

    # Outfeed moisture (wb fraction)
    # Prefer particle mean moisture when available - this matches NIR
    # measurement methodology (sampling individual seeds at oven exit).
    if use_particles:
        M_wb = getattr(outlet, "particle_avg_moisture_wb", 0.0)
        source = "particle"
    else:
        M_wb = getattr(outlet, "avg_moisture_wb", 0.0)
        source = "grid"

    M_pp = M_wb * 100.0
    M_tgt_pp = m_tgt * 100.0
    M_ok = abs(M_pp - M_tgt_pp) <= tolerance_moisture_pp
    details["outfeed_moisture_wb"] = M_wb
    details["outfeed_moisture_target_wb"] = m_tgt
    details["outfeed_moisture_source"] = source
    details["outfeed_moisture_note"] = f"{M_pp:.2f}% vs {M_tgt_pp:.2f}% ({source})"
    if use_particles:
        details["particle_count"] = particle_count

    # Gap: did controller open gap above setpoint? (Run#2 peak 94.1 mm)
    gap_triggered = False
    gap_peak_sim = 75.0
    if ts and "electrode_gap_mm" in ts:
        gap_arr = ts["electrode_gap_mm"]
        if isinstance(gap_arr, (list, tuple)):
            gap_peak_sim = max(gap_arr) if gap_arr else 75.0
        else:
            gap_peak_sim = float(gap_arr)
    gap_triggered = gap_peak_sim >= gap_peak_tgt - 2.0  # within 2 mm of target peak
    details["gap_peak_sim_mm"] = gap_peak_sim
    details["gap_peak_target_mm"] = gap_peak_tgt
    details["gap_note"] = f"peak {gap_peak_sim:.1f} mm (target {gap_peak_tgt} mm)" + (" — no MRH opening" if not gap_triggered else "")

    # Ia: did sim reach steady band 1.5–1.7 A?
    ia_steady_ok = False
    if ts and "anode_current_a" in ts:
        ia_arr = ts["anode_current_a"]
        if isinstance(ia_arr, (list, tuple)) and ia_arr:
            ia_max = max(ia_arr)
            ia_steady_ok = ia_lo - 0.05 <= ia_max <= ia_hi + 0.1
            details["ia_peak_sim_a"] = ia_max
        else:
            details["ia_peak_sim_a"] = None
    details["ia_target_a"] = f"{ia_lo}-{ia_hi}"
    details["ia_note"] = f"peak {details.get('ia_peak_sim_a', 'N/A')} A (target {ia_lo}-{ia_hi} A)" + (" — below MRL, gap never opens" if not ia_steady_ok and not gap_triggered else "")

    # Mass balance
    mass_ok = True
    if dispatched_kg > 0 and collected_kg is not None:
        balance_pct = (collected_kg - dispatched_kg) / dispatched_kg * 100.0
        mass_ok = abs(balance_pct) <= tolerance_mass_pct
        details["mass_balance_pct"] = balance_pct
        details["mass_note"] = f"{balance_pct:+.1f}% (dispatched {dispatched_kg:.1f} kg, collected {collected_kg:.1f} kg)"
    else:
        details["mass_note"] = "not checked"

    return ValidationResult(
        outfeed_moisture_ok=M_ok,
        outfeed_temp_ok=T_ok,
        gap_triggered_ok=gap_triggered,
        ia_steady_ok=ia_steady_ok,
        mass_balance_ok=mass_ok,
        details=details,
    )


def get_run2_targets() -> Dict[str, Any]:
    """Return Run#2 target values for reporting."""
    return dict(RUN2_TARGETS)
