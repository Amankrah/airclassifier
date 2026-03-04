"""
Pipeline Data Mappings
======================

Functions to transform outlet state from one simulation stage
into input parameters for the next stage.

Process chain: Pretreatment (GP-15) -> Milling (Hammer Mill) -> Air Classifier
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from airclassifier.pretreatment.physics.coupling import OutletState
    from airclassifier.milling.config import MillingOutletState


def map_pretreatment_to_milling(
    outlet: "OutletState",
    pretreatment_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Map pretreatment outlet state to milling input parameters.

    Transforms GP-15 RF heating outlet conditions into hammer mill feed
    configuration. The treated seeds (with reduced moisture and elevated
    temperature) become the mill input.

    Args:
        outlet: OutletState from pretreatment simulation
            - avg_moisture_wb: Mean moisture at outfeed [wet basis, 0-1]
            - avg_temperature_c: Mean temperature at outfeed [degC]
            - throughput_kg_per_hr: Mass flow rate [kg/h]
            - residence_time_s: Time in oven [s]

        pretreatment_params: Original pretreatment configuration dict
            - pt_run_mass_kg: Total batch mass if finite run

    Returns:
        Dict with milling parameters (mill_* prefixed for sync_settings_from_params):
            - mill_feed_moisture_wb: Feed moisture content
            - mill_feed_temperature_c: Feed temperature
            - mill_feed_rate_kg_per_hr: Feed rate (matches pretreatment throughput)
            - mill_seeds_feed_mass_kg: Total mass to process
            - mill_feed_d50_um: Feed particle size (whole seeds ~3000 um)

    Example:
        >>> outlet = OutletState(avg_moisture_wb=0.08, avg_temperature_c=65.0, ...)
        >>> params = map_pretreatment_to_milling(outlet, {"pt_run_mass_kg": 61.0})
        >>> params["mill_feed_moisture_wb"]
        0.08
    """
    # Batch mass from pretreatment (if finite run) or estimate from throughput
    run_mass = pretreatment_params.get("pt_run_mass_kg", 0.0)
    if run_mass <= 0 and outlet.throughput_kg_per_hr > 0 and outlet.residence_time_s > 0:
        # Estimate from throughput * residence time
        run_mass = outlet.throughput_kg_per_hr * (outlet.residence_time_s / 3600.0)

    return {
        # Core feed conditions from pretreatment outlet
        "mill_feed_moisture_wb": outlet.avg_moisture_wb,
        "mill_feed_temperature_c": outlet.avg_temperature_c,

        # Match throughput to pretreatment output
        "mill_feed_rate_kg_per_hr": outlet.throughput_kg_per_hr,

        # Batch mass for finite runs
        "mill_seeds_feed_mass_kg": run_mass,

        # Feed is whole seeds (yellow pea ~6-8mm diameter)
        # Using 3000 um as representative whole-seed size
        "mill_feed_d50_um": 3000.0,
    }


def map_milling_to_classification(
    outlet: "MillingOutletState",
    milling_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Map milling outlet state to air classifier input parameters.

    Transforms hammer mill flour output (PSD, temperature, moisture) into
    air classifier feed configuration. The flour PSD determines separation
    behavior in the zigzag and wheel classifier stages.

    Args:
        outlet: MillingOutletState from milling simulation
            - d50_um, d10_um, d90_um: Particle size percentiles [um]
            - psd_mass_fractions: Mass fractions per size class
            - psd_size_classes_um: Size class boundaries [um]
            - avg_temperature_c: Product temperature [degC]
            - avg_moisture_wb: Product moisture [wet basis, 0-1]
            - throughput_kg_per_hr: Mass flow rate [kg/h]
            - power_kw: Mill power consumption [kW]

        milling_params: Original milling configuration dict
            (currently unused, reserved for future extensions)

    Returns:
        Dict with classification parameters:
            - particle_diameter_um: Primary particle diameter (d50)
            - particle_diameter_std_um: Approx. std from d90-d10 spread
            - visual_particle_diameter: d50 in meters for visualization
            - psd_d10_um, psd_d50_um, psd_d90_um: Size percentiles
            - psd_mass_fractions: Full PSD mass fractions
            - psd_size_classes_um: PSD size class boundaries
            - feed_temperature_c: Thermal state passthrough
            - feed_moisture_wb: Moisture state passthrough
            - throughput_kg_h: Mass flow for loading calculations
            - solids_mass_flow_kg_s: Mass flow in SI units
            - particle_density: Material density (yellow pea flour ~1450 kg/m3)
            - from_milling_pipeline: Flag indicating data source

    Example:
        >>> outlet = MillingOutletState(d50_um=25.0, d10_um=10.0, d90_um=80.0, ...)
        >>> params = map_milling_to_classification(outlet, {})
        >>> params["particle_diameter_um"]
        25.0
    """
    # Estimate std deviation from span (d90 - d10) / 3.29 for normal distribution
    # This is approximate; actual PSD may be log-normal or Rosin-Rammler
    span = outlet.d90_um - outlet.d10_um
    std_estimate = span / 3.29 if span > 0 else outlet.d50_um * 0.3

    return {
        # Primary particle diameter for drag calculations
        "particle_diameter_um": outlet.d50_um,
        "particle_diameter_std_um": std_estimate,

        # Visual display size (meters)
        "visual_particle_diameter": outlet.d50_um * 1e-6,

        # Full PSD data for multi-size classification
        "psd_d10_um": outlet.d10_um,
        "psd_d50_um": outlet.d50_um,
        "psd_d90_um": outlet.d90_um,
        "psd_mass_fractions": list(outlet.psd_mass_fractions),
        "psd_size_classes_um": list(outlet.psd_size_classes_um),

        # Thermal state passthrough
        "feed_temperature_c": outlet.avg_temperature_c,
        "feed_moisture_wb": outlet.avg_moisture_wb,

        # Mass flow for loading ratio calculations
        "throughput_kg_h": outlet.throughput_kg_per_hr,
        "solids_mass_flow_kg_s": outlet.throughput_kg_per_hr / 3600.0,

        # Standard yellow pea flour density
        # Whole seeds ~1450 kg/m3; flour slightly lower due to porosity
        "particle_density": 1400.0,

        # Flag to indicate this came from the milling pipeline
        "from_milling_pipeline": True,

        # Power consumption info (informational)
        "upstream_mill_power_kw": outlet.power_kw,
    }


def get_pipeline_summary(
    pretreatment_outlet: "OutletState | None" = None,
    milling_outlet: "MillingOutletState | None" = None,
) -> Dict[str, Any]:
    """Generate a summary of pipeline state for display.

    Args:
        pretreatment_outlet: Outlet from pretreatment (if completed)
        milling_outlet: Outlet from milling (if completed)

    Returns:
        Dict with summary metrics for each completed stage
    """
    summary = {
        "pretreatment": None,
        "milling": None,
    }

    if pretreatment_outlet is not None:
        summary["pretreatment"] = {
            "moisture_wb": pretreatment_outlet.avg_moisture_wb,
            "temperature_c": pretreatment_outlet.avg_temperature_c,
            "throughput_kg_h": pretreatment_outlet.throughput_kg_per_hr,
            "energy_kwh": pretreatment_outlet.total_energy_kwh,
            "denaturation_pct": pretreatment_outlet.protein_denaturation_fraction * 100,
        }

    if milling_outlet is not None:
        summary["milling"] = {
            "d50_um": milling_outlet.d50_um,
            "d10_um": milling_outlet.d10_um,
            "d90_um": milling_outlet.d90_um,
            "throughput_kg_h": milling_outlet.throughput_kg_per_hr,
            "power_kw": milling_outlet.power_kw,
            "moisture_wb": milling_outlet.avg_moisture_wb,
            "temperature_c": milling_outlet.avg_temperature_c,
        }

    return summary
