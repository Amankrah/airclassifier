"""
Feed-to-Classification Flow Physics: Ductwork from Deagglomerator to Venturi Solids Inlet
==========================================================================================

Continuation of feed_flow_physics: computes air flow physics and particle kinetics
along the ductwork from the feed system (deagglomerator) outlet to the last transition
that joins the venturi solids inlet.

Physics:
  - Air flow (optional carrier/sweep): velocity, Reynolds, pressure drop (Darcy-Weisbach, K-factors).
Kinetics:
  - Gravity-driven particle transport: terminal velocity (Schiller-Naumann), gravity component
    along segment direction, residence time per segment, slip velocity.
Uses ductwork geometry from CompleteClassifierAssembly (feed-to-venturi solids path only).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..geometry.assembly.complete_system import CompleteClassifierAssembly, CompleteSystemParams
from .air_flow_physics import DuctSegment, calculate_duct_pressure_drop
from .feed_flow_physics import (
    RHO_AIR,
    MU_AIR,
    DEFAULT_PARTICLE_DENSITY,
)
from ..utils.constants import PI, GRAVITY
from ..particles.drag_models import terminal_velocity_schiller_naumann

# Defaults
RHO_F = RHO_AIR
MU_F = MU_AIR
RHO_P_DEFAULT = DEFAULT_PARTICLE_DENSITY

# Friction coefficient powder on steel (literature); used for inclined chute kinetics
FRICTION_POWDER_STEEL = 0.4


def compute_feed_entry_rate_particles_per_s(
    solids_mass_flow_kg_s: float,
    particle_density_kg_m3: float,
    particle_diameter_m: float,
) -> float:
    """
    Compute particle feed entry rate at the solids inlet [particles/s] from mass flow and particle properties.
    Used by classification to know feed entry rate from feedclass/feed_flow_physics (no magic numbers).
    m_particle = rho_p * (pi/6)*d^3; N_dot = m_dot_solids / m_particle.
    """
    if solids_mass_flow_kg_s <= 0 or particle_density_kg_m3 <= 0 or particle_diameter_m <= 0:
        return 0.0
    vol_particle = (PI / 6.0) * (particle_diameter_m ** 3)
    mass_particle = particle_density_kg_m3 * vol_particle
    return solids_mass_flow_kg_s / mass_particle


def compute_venturi_max_throughput_kg_h(
    air_flow_m3_s: float,
    max_loading_ratio: float = 2.0,
    air_density_kg_m3: float = RHO_F,
) -> float:
    """
    Compute the maximum solids throughput [kg/h] for the venturi at given air flow.

    The venturi's solids capacity is limited by the loading ratio:
        mu = m_dot_solids / m_dot_air  <=  max_loading_ratio

    For dilute-phase pneumatic transport, mu < 5 is typical; mu = 2 is
    conservative and ensures stable entrainment.

    Use this to coordinate the feed system throughput with the classification
    system's venturi capacity.  The screw feeder RPM should be set so that
    its discharge rate does not exceed this value.

    Args:
        air_flow_m3_s: Volumetric air flow rate [m³/s]
        max_loading_ratio: Maximum solids/air mass ratio [-] (default 2.0)
        air_density_kg_m3: Air density [kg/m³] (default STP)

    Returns:
        Maximum solids throughput [kg/h]
    """
    m_dot_air = air_flow_m3_s * air_density_kg_m3  # kg/s
    m_dot_solids_max = max_loading_ratio * m_dot_air  # kg/s
    return m_dot_solids_max * 3600.0  # kg/h


@dataclass
class FeedDuctSegmentResult:
    """Geometry and flow result for one duct/elbow/transition segment (feed path)."""
    name: str
    segment_type: str   # "duct" | "elbow" | "transition"
    diameter: float
    length: float
    area: float
    # Direction of flow along segment (unit vector) for gravity component; (0,-1,0) = down
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0)
    k_factor: float = 0.0
    # Filled by compute
    velocity_air: float = 0.0
    reynolds_air: float = 0.0
    pressure_drop_Pa: float = 0.0
    terminal_velocity_vertical: float = 0.0
    gravity_component_along: float = 0.0
    residence_time_s: float = 0.0
    particle_velocity_along: float = 0.0


def _segment_from_component_feed(duct, index: int, prev_direction: Optional[Tuple[float, float, float]] = None) -> Optional[FeedDuctSegmentResult]:
    """Build FeedDuctSegmentResult from ductwork component; include direction for kinetics."""
    from ..geometry.components.ductwork import RoundDuct, DuctElbow
    from ..geometry.components.transitions import Transition

    default_dir = (0.0, -1.0, 0.0) if prev_direction is None else prev_direction

    if isinstance(duct, RoundDuct):
        p = duct.params
        d = p.diameter
        area = PI * (d / 2) ** 2
        dr = getattr(p, "direction", (0.0, -1.0, 0.0))
        if isinstance(dr, (list, tuple)) and len(dr) >= 3:
            dr = np.array(dr, dtype=float)
            n = np.linalg.norm(dr)
            dr = tuple((dr / n).tolist()) if n > 0 else default_dir
        else:
            dr = default_dir
        return FeedDuctSegmentResult(
            name=f"feed_duct_{index}",
            segment_type="duct",
            diameter=d,
            length=p.length,
            area=area,
            direction=dr,
            k_factor=0.0,
        )
    if isinstance(duct, DuctElbow):
        p = duct.params
        d = p.diameter
        area = PI * (d / 2) ** 2
        R_bend = p.bend_radius if p.bend_radius else 1.5 * d
        r_over_d = R_bend / d
        k = 0.22 + 0.04 * (1.0 / max(r_over_d, 0.3))
        # Path length for residence: 90 deg bend = pi*R/2
        path_len = (PI / 2.0) * R_bend
        return FeedDuctSegmentResult(
            name=f"feed_elbow_{index}",
            segment_type="elbow",
            diameter=d,
            length=path_len,
            area=area,
            direction=default_dir,
            k_factor=k,
        )
    if isinstance(duct, Transition):
        p = duct.params
        d_in = p.inlet_dimensions[0] if p.transition_type.startswith("round") else np.sqrt(4 * p.inlet_area / PI)
        area_in = p.inlet_area
        return FeedDuctSegmentResult(
            name=f"feed_transition_{index}",
            segment_type="transition",
            diameter=d_in,
            length=p.length,
            area=area_in,
            direction=default_dir,
            k_factor=0.15 if p.is_contraction else 0.4,
        )
    return None


def extract_feed_to_venturi_segments(
    assembly: CompleteClassifierAssembly,
) -> List[FeedDuctSegmentResult]:
    """Extract flow segments from the feed-to-venturi solids ductwork."""
    path = assembly.get_feed_to_venturi_ductwork()
    segments = []
    prev_dir = (0.0, -1.0, 0.0)
    for i, (comp, _pos) in enumerate(path):
        seg = _segment_from_component_feed(comp, i, prev_dir)
        if seg is not None:
            segments.append(seg)
            prev_dir = seg.direction
    return segments


def compute_ductwork_flow_feed(
    segments: List[FeedDuctSegmentResult],
    volume_flow_rate_m3_s: float,
    rho_air: float = RHO_F,
    mu_air: float = MU_F,
    roughness: float = 0.00015,
) -> None:
    """
    Compute air flow (velocity, Re, pressure drop) for each segment in-place.
    Use 0 for volume_flow_rate_m3_s if no sweep/carrier air.
    """
    Q = volume_flow_rate_m3_s
    for seg in segments:
        v = Q / seg.area if seg.area > 0 else 0.0
        Re = (rho_air * v * seg.diameter / mu_air) if mu_air > 0 and seg.diameter > 0 else 0.0
        if seg.segment_type == "duct":
            ds = DuctSegment(
                name=seg.name,
                diameter=seg.diameter,
                length=seg.length,
                area=seg.area,
                roughness=roughness,
            )
            dp = calculate_duct_pressure_drop(ds, v, rho_air, mu_air)
        else:
            dp = seg.k_factor * (0.5 * rho_air * v ** 2)
        seg.velocity_air = v
        seg.reynolds_air = Re
        seg.pressure_drop_Pa = dp


def compute_particle_kinetics_feed(
    segments: List[FeedDuctSegmentResult],
    particle_diameter_m: float = 50e-6,
    particle_density_kg_m3: float = RHO_P_DEFAULT,
    rho_air: float = RHO_F,
    mu_air: float = MU_AIR,
    gravity: float = GRAVITY,
    friction_coefficient: float = FRICTION_POWDER_STEEL,
) -> None:
    """
    Compute particle kinetics along the chute for each segment in-place from geometry and physics:
    - Terminal velocity (vertical) from Schiller-Naumann
    - Gravity component along segment from dot(g_vec, direction)
    - Angle from horizontal from segment direction; effective acceleration
      a_eff = g * (sin(theta) - mu * cos(theta)) for sliding particle
    - Particle velocity along chute: v = sqrt(2 * a_eff * L) when a_eff > 0, capped by v_term
    - Residence time = L / v
    """
    v_term = terminal_velocity_schiller_naumann(
        diameter=particle_diameter_m,
        particle_density=particle_density_kg_m3,
        fluid_density=rho_air,
        fluid_viscosity=mu_air,
        gravity=gravity,
    )
    g_vec = np.array([0.0, -gravity, 0.0])
    for seg in segments:
        seg.terminal_velocity_vertical = v_term
        dr = np.array(seg.direction, dtype=float)
        g_along = np.dot(g_vec, dr)
        seg.gravity_component_along = float(g_along)
        # Angle from horizontal: sin(theta) = |vertical component| = |dy| for Y-down convention
        dy = dr[1]
        sin_theta = np.clip(np.abs(dy), 0.0, 1.0)
        cos_theta = np.sqrt(1.0 - sin_theta ** 2) if sin_theta < 1.0 else 0.0
        # Effective acceleration along chute: a = g*(sin(theta) - mu*cos(theta))
        a_eff = gravity * (sin_theta - friction_coefficient * cos_theta)
        if a_eff > 1e-6:
            # Particle accelerates: v = sqrt(2*a*L), capped by terminal velocity
            v_along = min(np.sqrt(2.0 * a_eff * seg.length), v_term)
            v_along = max(v_along, 0.05)
        elif abs(g_along) < 1e-6:
            # Horizontal segment: minimal slip velocity
            v_along = 0.1
        else:
            # a_eff <= 0 (e.g. too shallow or friction dominates): slip scale from v_term
            v_along = max(0.05, v_term * 0.2)
        seg.particle_velocity_along = float(v_along)
        seg.residence_time_s = seg.length / seg.particle_velocity_along if seg.particle_velocity_along > 0 else 0.0


def compute_feed_to_venturi_flow(
    assembly: CompleteClassifierAssembly,
    volume_flow_air_m3_s: float = 0.0,
    particle_diameter_m: float = 50e-6,
    particle_density_kg_m3: float = RHO_P_DEFAULT,
    rho_air: float = RHO_F,
    mu_air: float = MU_F,
    solids_mass_flow_kg_s: Optional[float] = None,
    sphericity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One-shot: extract feed-to-venturi ductwork and compute air flow + particle kinetics.

    Args:
        assembly: CompleteClassifierAssembly with ductwork built.
        volume_flow_air_m3_s: Optional sweep/carrier air in chute [m³/s]; 0 = gravity only.
        particle_diameter_m: Particle diameter [m] for kinetics (from feed_flow_physics/material).
        particle_density_kg_m3: Particle density [kg/m³] (from feed_flow_physics/material).
        rho_air, mu_air: Air properties.
        solids_mass_flow_kg_s: Optional solids mass flow [kg/s]; when set, result includes particle_feed_rate_per_s.
        sphericity: Optional particle sphericity [-] (from feed_flow_physics/material); passed through to classification.

    Returns:
        Dict with segments, total_pressure_drop_Pa, total_residence_time_s,
        venturi_solids_inlet_diameter_m, particle_diameter_m, particle_density_kg_m3,
        particle_feed_rate_per_s (if solids_mass_flow_kg_s given), sphericity (if given).
    """
    segments = extract_feed_to_venturi_segments(assembly)
    compute_ductwork_flow_feed(segments, volume_flow_air_m3_s, rho_air, mu_air)
    compute_particle_kinetics_feed(
        segments,
        particle_diameter_m=particle_diameter_m,
        particle_density_kg_m3=particle_density_kg_m3,
        rho_air=rho_air,
        mu_air=mu_air,
    )

    total_dp = sum(s.pressure_drop_Pa for s in segments)
    total_residence = sum(s.residence_time_s for s in segments)
    d_venturi, A_venturi = get_venturi_solids_inlet_from_assembly(assembly)

    seg_dicts = []
    for s in segments:
        seg_dicts.append({
            "name": s.name,
            "type": s.segment_type,
            "diameter_m": s.diameter,
            "length_m": s.length,
            "direction": s.direction,
            "velocity_air_m_s": s.velocity_air,
            "reynolds_air": s.reynolds_air,
            "pressure_drop_Pa": s.pressure_drop_Pa,
            "terminal_velocity_vertical_m_s": s.terminal_velocity_vertical,
            "gravity_component_along_m_s2": s.gravity_component_along,
            "particle_velocity_along_m_s": s.particle_velocity_along,
            "residence_time_s": s.residence_time_s,
        })

    out: Dict[str, Any] = {
        "segments": seg_dicts,
        "total_pressure_drop_Pa": total_dp,
        "total_residence_time_s": total_residence,
        "venturi_solids_inlet_diameter_m": d_venturi,
        "venturi_solids_inlet_area_m2": A_venturi,
        "volume_flow_air_m3_s": volume_flow_air_m3_s,
        "particle_diameter_m": particle_diameter_m,
        "particle_density_kg_m3": particle_density_kg_m3,
        "terminal_velocity_vertical_m_s": segments[0].terminal_velocity_vertical if segments else 0.0,
    }
    if sphericity is not None:
        out["sphericity"] = sphericity
    if solids_mass_flow_kg_s is not None and solids_mass_flow_kg_s > 0:
        out["particle_feed_rate_per_s"] = compute_feed_entry_rate_particles_per_s(
            solids_mass_flow_kg_s, particle_density_kg_m3, particle_diameter_m
        )
    return out


def get_venturi_solids_inlet_from_assembly(assembly: CompleteClassifierAssembly) -> Tuple[float, float]:
    """Get venturi solids inlet diameter and area from the classification system. Returns (diameter_m, area_m2)."""
    classification = assembly.get_subsystem("classification")
    if classification is None or not hasattr(classification, "venturi"):
        return 0.0, 0.0
    venturi = classification.venturi
    if venturi is None:
        return 0.0, 0.0
    port = venturi.ports.get("solids_inlet")
    if port is None:
        return 0.0, 0.0
    d = getattr(port, "diameter", 0.0)
    if d <= 0 and hasattr(port, "width") and hasattr(port, "height"):
        area = port.width * port.height
        d = np.sqrt(4 * area / PI)
    else:
        area = PI * (d / 2) ** 2
    return d, area


def print_feed_ductwork_summary(result: Dict[str, Any]) -> None:
    """Print summary of feed ductwork flow and kinetics."""
    print("=" * 70)
    print("FEED SYSTEM -> VENTURI SOLIDS INLET (ductwork flow + kinetics)")
    print("=" * 70)
    print(f"  Air flow (sweep):  {result.get('volume_flow_air_m3_s', 0) * 3600:.2f} m3/h")
    print(f"  Particle d:        {result.get('particle_diameter_m', 0) * 1e6:.1f} um")
    print(f"  Terminal v (vert): {result.get('terminal_velocity_vertical_m_s', 0):.4f} m/s")
    print(f"  Total dP:          {result.get('total_pressure_drop_Pa', 0):.2f} Pa")
    print(f"  Total residence:   {result.get('total_residence_time_s', 0):.3f} s")
    print(f"  Venturi solids D:  {result.get('venturi_solids_inlet_diameter_m', 0) * 1000:.0f} mm")
    print("-" * 70)
    for s in result.get("segments", []):
        print(f"  {s['name']:22s} {s['type']:10s} L={s['length_m']:.3f}m "
              f"v_air={s['velocity_air_m_s']:.2f} v_part={s['particle_velocity_along_m_s']:.2f} "
              f"t_res={s['residence_time_s']:.3f}s dP={s['pressure_drop_Pa']:.1f}Pa")
    print("=" * 70)


if __name__ == "__main__":
    params = CompleteSystemParams(
        include_feed_system=True,
        include_air_system=False,
        include_exhaust=False,
        include_ductwork=True,
    )
    assembly = CompleteClassifierAssembly(params)

    # Gravity-only (no sweep air)
    result = compute_feed_to_venturi_flow(
        assembly,
        volume_flow_air_m3_s=0.0,
        particle_diameter_m=50e-6,
        particle_density_kg_m3=1420.0,
    )
    print_feed_ductwork_summary(result)

    d_solids, A_solids = get_venturi_solids_inlet_from_assembly(assembly)
    print(f"\nVenturi solids inlet: D = {d_solids*1000:.0f} mm, A = {A_solids*1e4:.2f} cm2")
