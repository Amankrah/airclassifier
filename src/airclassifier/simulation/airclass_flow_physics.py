"""
Air Flow Physics: Ductwork from Air System to Venturi Air Inlet
================================================================

Continuation of air_flow_physics: computes velocity, Reynolds number,
and pressure drop along the ductwork from the air system outlet to the
last transition that joins the venturi air inlet.

Uses ductwork geometry from CompleteClassifierAssembly (air-to-venturi path only).
Flow is assumed incompressible; Q is constant along the path. Pressure drops
from Darcy-Weisbach (straight ducts), K-factor (elbows), and contraction loss (transition).

Why no Warp kernels (@wp.kernel / @wp.func)?
--------------------------------------------
air_flow_physics.py has two layers:

  1. **SPH simulation** (kernels): @wp.func (poly6_kernel, spiky_gradient, ...) and
     @wp.kernel (compute_sph_density_pressure_kernel, integrate_sph_air_kernel, ...)
     simulate air as particles through the *air system* (blower, scroll, dampers)
     on the GPU.

  2. **Steady duct hydraulics** (Python): calculate_friction_factor(),
     calculate_duct_pressure_drop(), DuctSegment, extract_air_geometry() are plain
     Python. They compute system resistance and per-duct v, Re, dP for the *air
     system* duct network (filter -> elbow -> blower -> dampers).

This module (airclass_flow_physics) is a **continuation of (2) only**. It applies
the same steady 1D duct physics to a *different* duct network: air system outlet
-> ducts -> venturi air inlet (geometry from CompleteClassifierAssembly). We do
not run SPH along that path; we only need scalar per-segment results (v = Q/A,
Re, dP). So we reuse the existing Python API from air_flow_physics:

  - DuctSegment, calculate_friction_factor, calculate_duct_pressure_drop

and do not add new kernels. Kernels would be needed only if we were to simulate
air as particles through this duct path (e.g. SPH continuation from blower
outlet to venturi), which is not the goal here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from ..geometry.assembly.complete_system import CompleteClassifierAssembly, CompleteSystemParams
from .air_flow_physics import (
    DuctSegment,
    calculate_friction_factor,
    calculate_duct_pressure_drop,
)
from ..utils.constants import PI
from ..particles import FluidConfig

# Default air properties (STP)
RHO_AIR = 1.204
MU_AIR = 1.82e-5


@dataclass
class DuctSegmentResult:
    """Result for one duct/elbow/transition segment."""
    name: str
    segment_type: str   # "duct" | "elbow" | "transition"
    diameter: float      # [m] characteristic (inlet for transition)
    length: float        # [m] (equivalent for elbow)
    area: float          # [m²] cross-section for velocity
    velocity: float = 0.0
    reynolds: float = 0.0
    pressure_drop: float = 0.0
    k_factor: float = 0.0   # for elbow/transition (0 for straight duct)


def _segment_from_component(duct, index: int) -> Optional[DuctSegmentResult]:
    """
    Build a DuctSegmentResult from a ductwork component (RoundDuct, DuctElbow, or Transition).
    """
    from ..geometry.components.ductwork import RoundDuct, DuctElbow
    from ..geometry.components.transitions import Transition

    if isinstance(duct, RoundDuct):
        p = duct.params
        d = p.diameter
        area = PI * (d / 2) ** 2
        return DuctSegmentResult(
            name=f"duct_{index}",
            segment_type="duct",
            diameter=d,
            length=p.length,
            area=area,
            k_factor=0.0,
        )
    if isinstance(duct, DuctElbow):
        p = duct.params
        d = p.diameter
        area = PI * (d / 2) ** 2
        # Equivalent length for 90° elbow: L_eq = K * D, K ~ 30 for R/D=1 (or use K_loss = 0.3)
        # We'll use K_loss (pressure drop = K * 0.5*rho*v^2) in the solver; store equivalent length for Re
        r_over_d = p.bend_radius / d if p.bend_radius else 1.0
        # Approximate equivalent length (m) for 90° elbow: 20-30 * D depending on R/D
        leq = (20.0 + 10.0 / max(r_over_d, 0.5)) * d * (p.angle / 90.0) if hasattr(p, 'angle') else 30 * d
        # K factor for 90° elbow R/D=1 is about 0.3
        k = 0.22 + 0.04 * (1.0 / max(r_over_d, 0.3))  # ~0.3 for R/D=1
        return DuctSegmentResult(
            name=f"elbow_{index}",
            segment_type="elbow",
            diameter=d,
            length=leq,
            area=area,
            k_factor=k,
        )
    if isinstance(duct, Transition):
        p = duct.params
        d_in = p.inlet_dimensions[0] if p.transition_type.startswith("round") else np.sqrt(4 * p.inlet_area / PI)
        area_in = p.inlet_area
        # Use inlet for velocity (conservative; transition has acceleration in contraction)
        return DuctSegmentResult(
            name=f"transition_{index}",
            segment_type="transition",
            diameter=d_in,
            length=p.length,
            area=area_in,
            # Contraction loss K ~ 0.1-0.5 depending on angle; expansion larger
            k_factor=0.15 if p.is_contraction else 0.4,
        )
    return None


def extract_air_to_venturi_segments(
    assembly: CompleteClassifierAssembly,
) -> List[DuctSegmentResult]:
    """
    Extract flow segments from the air-to-venturi ductwork.

    Args:
        assembly: CompleteClassifierAssembly with ductwork built (include_ductwork=True).

    Returns:
        List of DuctSegmentResult in flow order (air system outlet -> venturi air inlet).
    """
    path = assembly.get_air_to_venturi_ductwork()
    segments = []
    for i, (comp, _pos) in enumerate(path):
        seg = _segment_from_component(comp, i)
        if seg is not None:
            segments.append(seg)
    return segments


def compute_ductwork_flow(
    segments: List[DuctSegmentResult],
    volume_flow_rate_m3_s: float,
    rho: float = RHO_AIR,
    mu: float = MU_AIR,
    roughness: float = 0.00015,
    venturi_inlet_area_m2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute velocity, Reynolds, and pressure drop for each segment and total to venturi inlet.

    Args:
        segments: From extract_air_to_venturi_segments(assembly).
        volume_flow_rate_m3_s: Volumetric flow rate [m³/s] (from air system).
        rho: Air density [kg/m³].
        mu: Dynamic viscosity [Pa·s].
        roughness: Duct roughness [m].
        venturi_inlet_area_m2: If set, venturi_inlet_velocity = Q / this area (outlet of last transition).

    Returns:
        Dict with segments, total_pressure_drop_Pa, venturi_inlet_velocity_m_s, venturi_inlet_diameter_m, etc.
    """
    if not segments:
        return {
            "segments": [],
            "total_pressure_drop_Pa": 0.0,
            "venturi_inlet_velocity_m_s": 0.0,
            "venturi_inlet_diameter_m": 0.0,
            "volume_flow_rate_m3_s": volume_flow_rate_m3_s,
            "volume_flow_rate_m3_h": volume_flow_rate_m3_s * 3600,
        }

    Q = volume_flow_rate_m3_s
    total_dp = 0.0
    results = []

    for seg in segments:
        v = Q / seg.area if seg.area > 0 else 0.0
        Re = (rho * v * seg.diameter / mu) if mu > 0 and seg.diameter > 0 else 0.0

        if seg.segment_type == "duct":
            # Darcy-Weisbach
            ds = DuctSegment(
                name=seg.name,
                diameter=seg.diameter,
                length=seg.length,
                area=seg.area,
                roughness=roughness,
            )
            dp = calculate_duct_pressure_drop(ds, v, rho, mu)
        else:
            # Elbow or transition: dP = K * (rho * v^2 / 2)
            dp = seg.k_factor * (0.5 * rho * v ** 2)

        seg.velocity = v
        seg.reynolds = Re
        seg.pressure_drop = dp
        total_dp += dp
        results.append({
            "name": seg.name,
            "type": seg.segment_type,
            "diameter_m": seg.diameter,
            "length_m": seg.length,
            "velocity_m_s": v,
            "reynolds": Re,
            "pressure_drop_Pa": dp,
        })

    last = segments[-1]
    if venturi_inlet_area_m2 is not None and venturi_inlet_area_m2 > 0:
        venturi_inlet_v = Q / venturi_inlet_area_m2
        venturi_inlet_d = np.sqrt(4 * venturi_inlet_area_m2 / PI)
    else:
        venturi_inlet_v = last.velocity
        venturi_inlet_d = last.diameter

    return {
        "segments": results,
        "total_pressure_drop_Pa": total_dp,
        "venturi_inlet_velocity_m_s": venturi_inlet_v,
        "venturi_inlet_diameter_m": venturi_inlet_d,
        "volume_flow_rate_m3_s": Q,
        "volume_flow_rate_m3_h": Q * 3600,
    }


def compute_air_to_venturi_flow(
    assembly: CompleteClassifierAssembly,
    volume_flow_rate_m3_s: float,
    rho: float = RHO_AIR,
    mu: float = MU_AIR,
) -> Dict[str, Any]:
    """
    One-shot: extract ductwork from assembly and compute flow physics to venturi inlet.

    Args:
        assembly: CompleteClassifierAssembly (with ductwork built).
        volume_flow_rate_m3_s: Flow rate [m³/s] (e.g. from AirFlowPhysicsSimulator).

    Returns:
        Same as compute_ductwork_flow(...), with venturi inlet velocity/diameter from assembly.
    """
    segments = extract_air_to_venturi_segments(assembly)
    d_venturi, A_venturi = get_venturi_air_inlet_from_assembly(assembly)
    return compute_ductwork_flow(
        segments,
        volume_flow_rate_m3_s,
        rho=rho,
        mu=mu,
        venturi_inlet_area_m2=A_venturi if A_venturi > 0 else None,
    )


def get_venturi_air_inlet_from_assembly(assembly: CompleteClassifierAssembly) -> Tuple[float, float]:
    """
    Get venturi air inlet diameter and cross-sectional area from the classification system.

    Returns:
        (diameter_m, area_m2)
    """
    classification = assembly.get_subsystem("classification")
    if classification is None or not hasattr(classification, "venturi"):
        return 0.0, 0.0
    venturi = classification.venturi
    port = venturi.ports.get("air_inlet")
    if port is None:
        return 0.0, 0.0
    d = getattr(port, "diameter", 0.0)
    if d <= 0 and hasattr(port, "width") and hasattr(port, "height"):
        area = port.width * port.height
        d = np.sqrt(4 * area / PI)
    else:
        area = PI * (d / 2) ** 2
    return d, area


def print_ductwork_flow_summary(result: Dict[str, Any]) -> None:
    """Print a summary of the ductwork flow results."""
    print("=" * 60)
    print("AIR SYSTEM -> VENTURI DUCTWORK FLOW")
    print("=" * 60)
    print(f"  Volume flow:    {result.get('volume_flow_rate_m3_s', 0) * 3600:.1f} m³/h")
    print(f"  Total dP:       {result.get('total_pressure_drop_Pa', 0):.1f} Pa")
    print(f"  Segments:      {len(result.get('segments', []))}")
    print("-" * 60)
    for s in result.get("segments", []):
        print(f"  {s['name']:18s} {s['type']:10s} "
              f"v={s['velocity_m_s']:6.2f} m/s  Re={s['reynolds']:8.0f}  dP={s['pressure_drop_Pa']:6.1f} Pa")
    print("=" * 60)


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    # Build complete system with ductwork (air -> venturi only path is built when air_system + ductwork enabled)
    params = CompleteSystemParams(
        include_feed_system=False,
        include_exhaust=False,
        include_ductwork=True,
        include_air_system=True,
    )
    assembly = CompleteClassifierAssembly(params)

    Q_m3_s = 1768.0 / 3600.0  # m³/s (from air system at 2500 RPM)
    result = compute_air_to_venturi_flow(assembly, Q_m3_s)
    print_ductwork_flow_summary(result)

    d_venturi, A_venturi = get_venturi_air_inlet_from_assembly(assembly)
    print(f"\nVenturi air inlet: D = {d_venturi*1000:.0f} mm, A = {A_venturi*1e4:.1f} cm²")
    print(f"Velocity at venturi inlet (Q/A): {Q_m3_s / A_venturi:.2f} m/s")
