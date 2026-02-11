#!/usr/bin/env python3
"""
Physics-Based Classification Flow Simulation Example
=====================================================

This example demonstrates the physics-based particle separation simulation
through the classification system:

- Venturi entrainment (Bernoulli acceleration)
- Zigzag classification (counter-current air separation)
- Multi-stage cyclone separation (centrifugal forces)
- Bag filter collection (inertial impaction)

The simulation tracks particles through the system and shows:
- Real-time particle positions colored by velocity
- Zone transitions as particles flow through components
- Separation statistics (coarse vs. fines vs. protein)

Usage:
    # Default run (1000 particles, 5 s, 0.3 m3/s)
    python examples/run_classification_flow.py

    # Diagnostics only (no simulation)
    python examples/run_classification_flow.py --no-sim
    python examples/run_classification_flow.py --diagnostics --no-sim

    # Full system: airclass -> feedclass -> classification (no magic numbers)
    python examples/run_classification_flow.py --full-system
    python examples/run_classification_flow.py --full-system --material yellow_pea --no-sim

    # Operating-condition validation (zigzag/cyclone cut sizes vs flow)
    python examples/run_classification_flow.py --validate
    python examples/run_classification_flow.py --diagnostics --validate

    # Run with diagnostics printed before/after
    python examples/run_classification_flow.py --diagnostics
    python examples/run_classification_flow.py -d --particles 2000 --time 10

    # Particle count and time
    python examples/run_classification_flow.py --particles 5000 --time 10
    python examples/run_classification_flow.py -n 500 -t 3

    # Air flow rate (m3/s)
    python examples/run_classification_flow.py --air-flow 0.1
    python examples/run_classification_flow.py --air-flow 0.5 --particles 2000

    # Particle size (when not using --material)
    python examples/run_classification_flow.py --particle-dia 30 --particle-std 15
    python examples/run_classification_flow.py --particle-dia 80 --particle-std 40

    # Food powder materials (whole flour: protein + starch + fiber)
    python examples/run_classification_flow.py --material yellow_pea
    python examples/run_classification_flow.py --material faba_bean --particles 500
    python examples/run_classification_flow.py --material oat --time 8

    # Single fraction (protein, starch, or fiber only)
    python examples/run_classification_flow.py --material protein --particles 1000
    python examples/run_classification_flow.py --material starch
    python examples/run_classification_flow.py --material fiber

    # Target cut size d50 in microns (auto air flow; geometry warning may apply)
    python examples/run_classification_flow.py --target-d50 35
    python examples/run_classification_flow.py --target-d50 40 --particles 2000

    # Zigzag geometry overrides (mm)
    python examples/run_classification_flow.py --zigzag-width 100 --zigzag-depth 150
    python examples/run_classification_flow.py --zigzag-width 120 --diagnostics

    # Without preclassification (wheel-only: no venturi, zigzag, dropout)
    python examples/run_classification_flow.py --without-preclassification
    python examples/run_classification_flow.py --full-system --material yellow_pea --without-preclassification
    python examples/run_classification_flow.py --wheel-only  # alias for --without-preclassification

    # Wheel classifier RPM (main classifier; overrides geometry default)
    python examples/run_classification_flow.py --wheel-rpm 6000
    python examples/run_classification_flow.py --wheel-rpm 10000 --diagnostics

    # Turbulence intensity
    python examples/run_classification_flow.py --turbulence 0.2

    # Compute device
    python examples/run_classification_flow.py --device cpu
    python examples/run_classification_flow.py --device cuda -n 5000

    # 3D visualization (requires pyvista)
    python examples/run_classification_flow.py --visualize
    python examples/run_classification_flow.py -v --particles 2000 --time 10
    python examples/run_classification_flow.py --visualize --material yellow_pea --diagnostics

    # Multi-pass recirculation (refeed Cy1 back through the classifier)
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 500 --wheel-rpm 1300 --recirculate cy1 --passes 3
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 500 --wheel-rpm 1300 --recirculate cy1 cy2 --passes 2
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 500 --wheel-rpm 1300 --recirculate cy1 --passes 3 --recirculate-wheel-rpm 2000
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 500 --wheel-rpm 1300 --recirculate cy1 --passes 3 --recirculate-time 180

    # Combined examples
    python examples/run_classification_flow.py --diagnostics --visualize --particles 5000 --time 10
    python examples/run_classification_flow.py --material yellow_pea --air-flow 0.2 -n 1000 -t 5 -v
    python examples/run_classification_flow.py --no-sim --material faba_bean --diagnostics

Operating point (bench-scale geometry: 40 mm venturi, 200 mm wheel):
    Defaults: --blower-rpm 700 --wheel-rpm 975 (optimized for yellow pea protein recovery,
    wheel-only mode: 30% protein recovery, 59% starch yield, 72% purity).

    - Low blower (e.g. 350 RPM): air cannot carry fines through the wheel; most go to wheel coarse.
    - Medium blower (600–750 RPM): optimal range for wheel-only protein/starch separation.
      Wheel RPM 800–1500 gives d50 ~25–45 µm, good for protein/starch cut.
    - High blower (e.g. 2500 RPM): venturi can choke at the throat (Ma≈1). Use `--throat-diameter`
      to raise choked-flow limit if needed, but be aware high flow can break the cyclone cascade.
    Example: --full-system --material yellow_pea --wheel-only
    Example: --full-system --material yellow_pea --wheel-only --recirculate cy1 --passes 3

Options:
    -n, --particles N     Number of particles (default: 1000)
    -t, --time T          Simulation time in seconds (default: 5)
    --dt                  Time step in seconds (default: 0.001)
    --air-flow            Air flow rate in m3/s (default: 1768 m3/h, from air system at 2500 RPM)
    --particle-dia        Mean particle diameter in microns (default: 50)
    --particle-std        Particle diameter std dev in microns (default: 30)
    --material            Preset: yellow_pea, faba_bean, oat, protein, starch, fiber
    -v, --visualize       Enable 3D visualization (requires pyvista)
    -d, --diagnostics     Print detailed flow path with all calculations
    --validate            Run operating-condition validation (zigzag/cyclone vs flow)
    --full-system         Run airclass -> feedclass -> classification (no magic numbers)
    --without-preclassification  Disable preclassification (wheel-only, same as --wheel-only)
    --no-sim              Print diagnostics only, skip simulation
    --target-d50          Target cut size in microns (auto air flow)
    --zigzag-width        Override zigzag channel width in mm
    --zigzag-depth        Override zigzag channel depth in mm
    --wheel-rpm           Wheel classifier RPM (main classifier; default: 975)
    --turbulence          Turbulent intensity (default: 0.15)
    --device              cuda or cpu (default: cuda)
    --recirculate FRAC    Fractions to refeed: cy1, cy2, cy3, wheel_coarse, zigzag_coarse, bagfilter
    --passes N            Number of classification passes (default: 1; requires --recirculate)
    --recirculate-wheel-rpm RPM  Wheel RPM for passes 2+ (tighter cut on narrower PSD)
    --recirculate-time T  Simulation time for passes 2+ in seconds (default: same as --time)
    --attrition F         Venturi attrition per pass (0.0-0.5, default: 0.10 = 10% breakup)
    --attrition-min D     Min diameter for attrition in µm (default: 5.0 = protein body floor)
"""

import sys
import argparse
import time
import numpy as np

# Optional visualization
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Note: Install pyvista for 3D visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Physics-based classification flow simulation"
    )
    parser.add_argument(
        "--particles", "-n", type=int, default=100000,
        help="Number of particles (default: 100000)"
    )
    parser.add_argument(
        "--time", "-t", type=float, default=360.0,
        help="Simulation time in seconds (default: 360)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001,
        help="Time step in seconds (default: 0.001)"
    )
    # Default matches air system at 2500 RPM (run_air_flow_physics: ~1768 m³/h)
    _AIR_FLOW_DEFAULT_M3_S = 1768.0 / 3600.0  # ~0.491 m³/s
    parser.add_argument(
        "--air-flow", type=float, default=_AIR_FLOW_DEFAULT_M3_S,
        help="Air flow rate in m³/s (default: 1768 m³/h from air system at 2500 RPM)"
    )
    parser.add_argument(
        "--blower-rpm", type=float, default=700,
        help="VFD blower speed in RPM (overrides --air-flow via fan law: "
             "Q = 3000 m³/h × RPM/3000). Design: 3000 RPM = 3000 m³/h. "
             "Default: 700 RPM (optimized for yellow pea protein recovery)."
    )
    parser.add_argument(
        "--bypass-ratio", type=float, default=0.0,
        help="Bypass ratio 0.0-1.0: fraction of total flow bypassing venturi+zigzag. "
             "Bypass merges back before cyclones. E.g. 0.967 = 96.7%% bypass, "
             "3.3%% through classification. (default: 0.0 = no bypass)"
    )
    parser.add_argument(
        "--particle-dia", type=float, default=50.0,
        help="Mean particle diameter in microns (default: 50um for whole flour)"
    )
    parser.add_argument(
        "--particle-std", type=float, default=30.0,
        help="Particle diameter std dev in microns (default: 30um)"
    )
    parser.add_argument(
        "--material", type=str, default=None,
        choices=["yellow_pea", "faba_bean", "oat", "protein", "starch", "fiber"],
        help="Use preset food powder material with realistic size distribution"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Enable 3D visualization (requires pyvista)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)"
    )
    parser.add_argument(
        "--turbulence", type=float, default=0.15,
        help="Base turbulence intensity (default: 0.15). Scales zone-specific "
             "intensities proportionally: zigzag=0.25, cyclone=0.12 at base=0.15."
    )
    parser.add_argument(
        "--diagnostics", "-d", action="store_true",
        help="Print detailed flow path diagnostics with all calculations"
    )
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Only print diagnostics, don't run simulation"
    )
    parser.add_argument(
        "--target-d50", type=float, default=None,
        help="Target cut size in microns (auto-calculates air flow). "
             "For protein/starch: use 30-40um. WARNING: current geometry may need redesign."
    )
    parser.add_argument(
        "--throat-diameter", type=float, default=None,
        help="Override venturi throat diameter in mm (default: 40mm = 80mm inlet × 0.5 ratio). "
             "Smaller throat increases throat velocity/shear at moderate flow; larger throat raises choked-flow limit."
    )
    parser.add_argument(
        "--zigzag-width", type=float, default=None,
        help="Override zigzag channel width in mm (default: 120mm from geometry)"
    )
    parser.add_argument(
        "--zigzag-depth", type=float, default=None,
        help="Override zigzag channel depth in mm (default: 200mm from geometry)"
    )
    parser.add_argument(
        "--wheel-rpm", type=float, default=975,
        help="Wheel classifier speed in RPM (main classifier; default: 975 RPM, "
             "optimized for yellow pea protein recovery with d50=36 µm)."
    )
    parser.add_argument(
        "--wheel-only", action="store_true",
        help="Use wheel-only assembly (no zigzag, venturi, dropout): air inlet + 15° solids chute -> wheel -> cyclones -> bag"
    )
    parser.add_argument(
        "--without-preclassification", action="store_true",
        help="Disable preclassification (same as --wheel-only): no venturi, zigzag, or dropout. "
             "Works with both --full-system and legacy paths."
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run operating-condition validation (zigzag/cyclone cut sizes vs flow)"
    )
    parser.add_argument(
        "--full-system", action="store_true",
        help="Run air flow to venturi (airclass), then feed to venturi (feedclass), then classification from geometry and physics (no magic numbers)"
    )
    parser.add_argument(
        "--batch-feed", action="store_true",
        help="Force batch feeding (all particles active at t=0) instead of continuous. "
             "Default with --full-system is continuous feeding."
    )
    parser.add_argument(
        "--max-loading", type=float, default=2.0,
        help="Max solids loading ratio mu = m_dot_solids / m_dot_air for venturi entrainment cap (default: 2.0)"
    )

    # Multi-pass recirculation
    parser.add_argument(
        "--recirculate", type=str, nargs="+", default=None,
        metavar="FRACTION",
        help="Fractions to recirculate after each pass. Valid: cy1, cy2, cy3, "
             "wheel_coarse, zigzag_coarse, bagfilter. "
             "Example: --recirculate cy1 cy2"
    )
    parser.add_argument(
        "--passes", type=int, default=1,
        help="Number of classification passes (default: 1). "
             "Requires --recirculate to specify which fractions to refeed."
    )
    parser.add_argument(
        "--recirculate-wheel-rpm", type=float, default=None,
        help="Wheel RPM override for recirculation passes 2+. "
             "Useful for tighter cuts on narrower refeed PSD. "
             "Example: --recirculate-wheel-rpm 2000"
    )
    parser.add_argument(
        "--recirculate-time", type=float, default=None,
        help="Simulation time for recirculation passes 2+ in seconds. "
             "Default: same as --time. Fewer particles may need less time."
    )
    parser.add_argument(
        "--attrition", type=float, default=0.10,
        help="Venturi attrition: fraction of breakable diameter removed per "
             "recirculation pass (0.0-0.5, default: 0.10 = 10%%). Models "
             "shear-induced breakup of protein-starch composites at throat. "
             "0.0 disables attrition."
    )
    parser.add_argument(
        "--attrition-min", type=float, default=5.0,
        help="Minimum particle diameter (µm) below which attrition stops "
             "(default: 5.0 µm = individual protein bodies)."
    )
    parser.add_argument(
        "--recirculate-to-wheel", action="store_true",
        help="Feed recirculated particles directly to wheel classifier "
             "(skip venturi+zigzag). Default: particles re-enter through "
             "venturi solids inlet (matching physical machine design)."
    )
    
    args = parser.parse_args()

    # Unify --wheel-only and --without-preclassification into a single flag
    skip_preclassification = args.wheel_only or args.without_preclassification

    # Scale zone-specific turbulence from --turbulence base value.
    # Defaults: zigzag=0.25, cyclone=0.12 at base=0.15.
    _turb_scale = args.turbulence / 0.15
    turb_zigzag = 0.25 * _turb_scale
    turb_cyclone = 0.12 * _turb_scale

    # VFD: convert blower RPM to air flow via actual operating point
    if args.blower_rpm is not None:
        from airclassifier.simulation.airclass_flow_physics import compute_blower_operating_point
        op = compute_blower_operating_point(args.blower_rpm)
        args.air_flow = op["Q_m3_s"]

    print("=" * 70)
    print("PHYSICS-BASED CLASSIFICATION FLOW SIMULATION")
    print("  Protein/Starch Separation via Air Classification")
    print("=" * 70)
    if args.blower_rpm is not None:
        print(f"  VFD: {args.blower_rpm:.0f} RPM -> {op['Q_m3_h']:.0f} m³/h (operating point)")
        print(f"       Fan law would give {op['Q_fan_law_m3_h']:.0f} m³/h (overestimate)")
        print(f"       P = {op['P_operating_Pa']:.0f} Pa, eff = {op['efficiency']:.1%}, W = {op['shaft_power_W']:.0f} W")
    if args.bypass_ratio > 0:
        print(f"  Bypass: {args.bypass_ratio*100:.1f}% around venturi+zigzag")
    print(f"  Wheel RPM (main classifier): {args.wheel_rpm:.0f}")
    
    # Import modules
    from airclassifier.simulation.classification_flow_physics import (
        ClassificationFlowPhysicsSimulator,
        ClassificationFlowConfig,
        compute_venturi_physics_from_air_and_feed,
    )
    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly,
        ClassificationSystemParams,
    )
    from airclassifier.particles import FluidConfig, ParticleMaterial
    
    if args.full_system:
        # Full system: airclass -> feedclass -> classification (geometry + physics, no magic numbers)
        from airclassifier.geometry.assembly.complete_system import (
            CompleteClassifierAssembly,
            CompleteSystemParams,
        )
        from airclassifier.simulation.airclass_flow_physics import (
            compute_air_to_venturi_flow,
            print_ductwork_flow_summary,
        )
        from airclassifier.simulation.feedclass_flow_physics import (
            compute_feed_to_venturi_flow,
            print_feed_ductwork_summary,
            compute_venturi_max_throughput_kg_h,
        )
        if skip_preclassification:
            print("\n[FULL SYSTEM] Air -> Wheel (no preclassification) -> Cyclones -> Bag Filter")
        else:
            print("\n[FULL SYSTEM] Air -> Venturi -> Feed -> Venturi -> Classification")
        fluid = FluidConfig.air_at_stp()
        material = None
        if args.material:
            fraction = "whole" if args.material in ("yellow_pea", "faba_bean", "oat") else args.material
            material = ParticleMaterial.create_food_powder(args.material, fraction)
        # Particle size for feed ductwork: from material size distribution when available, else args
        if material is not None and getattr(material, "size_distribution", None) is not None:
            sd = material.size_distribution
            particle_dia_m = getattr(sd, "d50", None) or (sd.d_min + sd.d_max) / 2.0
        else:
            particle_dia_m = args.particle_dia * 1e-6
        particle_density = material.density if material else 1420.0
        Q_m3s = args.air_flow
        # Build classification params with optional overrides
        classification_params = None
        if args.throat_diameter is not None or skip_preclassification:
            from airclassifier.geometry.assembly.classification import ClassificationSystemParams
            classification_params = ClassificationSystemParams()
            if skip_preclassification:
                classification_params.use_preclassification = False
                print(f"  [Mode] Without preclassification (wheel-only): no venturi, zigzag, dropout")
            if args.throat_diameter is not None:
                throat_m = args.throat_diameter / 1000.0  # mm to m
                classification_params.venturi_throat_ratio = throat_m / classification_params.venturi_inlet_diameter
                print(f"  [Override] Venturi throat: {args.throat_diameter:.1f} mm "
                      f"(ratio={classification_params.venturi_throat_ratio:.3f})")
        # Coordinate feed throughput with venturi capacity
        # The screw feeder rate must not exceed what the venturi can entrain
        venturi_max_kg_h = compute_venturi_max_throughput_kg_h(
            Q_m3s, max_loading_ratio=args.max_loading,
        )
        # Use venturi capacity as throughput (capped by feeder max)
        throughput = min(venturi_max_kg_h, 500.0)  # 500 kg/h feeder design max

        complete_params = CompleteSystemParams(
            air_flow_m3_h=Q_m3s * 3600.0,
            throughput_kg_h=throughput,
            include_feed_system=True,
            include_air_system=True,
            include_exhaust=False,
            include_ductwork=True,
            classification_params=classification_params,
        )
        complete_assembly = CompleteClassifierAssembly(complete_params)
        if skip_preclassification:
            print("\n1. Air system -> Wheel junction (airclass, no venturi)...")
        else:
            print("\n1. Air system -> Venturi air inlet (airclass)...")
        air_result = compute_air_to_venturi_flow(
            complete_assembly, Q_m3s,
            rho=fluid.density, mu=fluid.dynamic_viscosity,
        )
        print_ductwork_flow_summary(air_result)
        if skip_preclassification:
            print("\n2. Feed system -> Wheel junction solids inlet (feedclass, no venturi)...")
        else:
            print("\n2. Feed system -> Venturi solids inlet (feedclass)...")
        solids_mass_flow_kg_s = complete_assembly.params.throughput_kg_h / 3600.0
        sphericity = getattr(material, "sphericity", None) if material else None
        feed_result = compute_feed_to_venturi_flow(
            complete_assembly,
            volume_flow_air_m3_s=0.0,
            particle_diameter_m=particle_dia_m,
            particle_density_kg_m3=particle_density,
            rho_air=fluid.density,
            mu_air=fluid.dynamic_viscosity,
            solids_mass_flow_kg_s=solids_mass_flow_kg_s,
            sphericity=sphericity,
        )
        print_feed_ductwork_summary(feed_result)
        # Material / feed properties used for classification validation (full-system)
        print("\n  Material / feed properties (used for classification validation):")
        if material is not None:
            sd = getattr(material, "size_distribution", None)
            if sd is not None:
                d_min_um = sd.d_min * 1e6
                d_max_um = sd.d_max * 1e6
                d50_um = getattr(sd, "d50", None)
                d50_um = (d50_um * 1e6) if d50_um is not None else (d_min_um + d_max_um) / 2.0
                print(f"    Material:        {material.name} (density={material.density:.0f} kg/m³, sphericity={getattr(material, 'sphericity', 0.75):.2f})")
                print(f"    Size range:      {d_min_um:.1f} – {d_max_um:.1f} µm   d50={d50_um:.1f} µm")
            else:
                print(f"    Material:        {material.name} (density={material.density:.0f} kg/m³, sphericity={getattr(material, 'sphericity', 0.75):.2f})")
            print(f"    Feed rep. d:     {particle_dia_m * 1e6:.1f} µm (feed ductwork and entry rate)")
            if feed_result.get("particle_feed_rate_per_s"):
                print(f"    Particle rate:   {feed_result['particle_feed_rate_per_s']:.0f} particles/s (solids mass flow + rep. d)")
        else:
            print(f"    Material:        generic (density={particle_density:.0f} kg/m³)")
            print(f"    Particle d:       {particle_dia_m * 1e6:.1f} µm")
        classification_assembly = complete_assembly.get_subsystem("classification")
        if not skip_preclassification:
            venturi_physics = compute_venturi_physics_from_air_and_feed(
                air_result, feed_result, classification_assembly,
                solids_mass_flow_kg_s=solids_mass_flow_kg_s,
                rho_air=fluid.density,
            )
            print("\n3. Venturi + classification (from geometry and air/feed results):")
            print(f"   Throat velocity:    {venturi_physics['venturi_throat_velocity_m_s']:.1f} m/s")
            print(f"   Mach at throat:     {venturi_physics['venturi_mach_throat']:.3f}")
            if venturi_physics['venturi_flow_limited']:
                print(f"   *** CHOKED FLOW *** Max: {venturi_physics['venturi_choked_flow_m3h']:.0f} m3/h")
            elif venturi_physics['venturi_mach_throat'] > 0.3:
                print(f"   *** COMPRESSIBLE *** Max: {venturi_physics['venturi_choked_flow_m3h']:.0f} m3/h")
            print(f"   dP (Bernoulli):     {venturi_physics['pressure_drop_bernoulli_Pa']:.0f} Pa ({venturi_physics['pressure_drop_bernoulli_Pa']/1000:.1f} kPa)")
            print(f"   K_venturi:          {venturi_physics['venturi_k_factor']:.1f} Pa/(m3/s)^2")
            print(f"   Particle entry:     {venturi_physics['particle_entry_velocity_m_s']:.2f} m/s")
            print(f"   Loading ratio:      {venturi_physics['loading_ratio']:.4f}")
            print(f"   Momentum transfer: {venturi_physics['momentum_transfer_N']:.1f} N")
            print(f"   dP (solids accel):  {venturi_physics['pressure_drop_solids_Pa']:.1f} Pa")
        else:
            print("\n3. Wheel-only classification (no venturi/zigzag):")
            print(f"   Air flow:           {Q_m3s * 3600:.0f} m³/h ({Q_m3s:.3f} m³/s)")
            print(f"   Solids mass flow:   {solids_mass_flow_kg_s * 3600:.1f} kg/h")
        assembly = classification_assembly
        config = ClassificationFlowConfig.from_air_and_feed_results(
            air_result, feed_result, classification_assembly,
            solids_mass_flow_kg_s=solids_mass_flow_kg_s,
            num_particles_capacity=args.particles,
            simulation_time_s=args.time,
            dt=args.dt,
            device=args.device,
            turbulence_zigzag=turb_zigzag,
            turbulence_cyclone=turb_cyclone,
            fluid_config=fluid,
            material=material,
            bypass_ratio=args.bypass_ratio,
            continuous_feeding=not args.batch_feed,
            max_loading_ratio=args.max_loading,
            wheel_rpm=args.wheel_rpm,
        )
        print("\nCreating classification system assembly (from full system)...")
    else:
        # Classification-only: assembly and config from args (legacy path)
        # Create assembly with optional geometry overrides
        print("\nCreating classification system assembly...")
        
        # Check for zigzag geometry overrides
        from airclassifier.geometry.assembly.classification import ClassificationSystemParams
        
        custom_params = None
        has_overrides = (args.zigzag_width is not None or
                         args.zigzag_depth is not None or
                         args.throat_diameter is not None or
                         skip_preclassification)
        if has_overrides:
            custom_params = ClassificationSystemParams()
            if skip_preclassification:
                custom_params.use_preclassification = False
                print(f"  [Mode] Without preclassification (wheel-only): no zigzag, venturi, dropout; 15° solids chute + air inlet -> wheel")
            if args.throat_diameter is not None:
                throat_m = args.throat_diameter / 1000.0  # mm to m
                custom_params.venturi_throat_ratio = throat_m / custom_params.venturi_inlet_diameter
                print(f"  [Override] Venturi throat: {args.throat_diameter:.1f} mm "
                      f"(ratio={custom_params.venturi_throat_ratio:.3f})")
            if args.zigzag_width is not None:
                custom_params.zigzag_channel_width = args.zigzag_width / 1000.0  # mm to m
                print(f"  [Override] Zigzag width: {args.zigzag_width:.1f} mm")
            if args.zigzag_depth is not None:
                custom_params.zigzag_channel_depth = args.zigzag_depth / 1000.0  # mm to m
                print(f"  [Override] Zigzag depth: {args.zigzag_depth:.1f} mm")
            assembly = ClassificationSystemAssembly(params=custom_params)
        else:
            assembly = ClassificationSystemAssembly()
        
        # Print assembly info
        print(f"\n  Components:")
        if assembly.venturi is not None:
            print(f"    - Venturi eductor (particle entrainment)")
        if assembly.zigzag is not None:
            print(f"    - Zigzag classifier (primary separation)")
        if getattr(assembly.params, 'use_preclassification', True) is False:
            print(f"    - Air inlet + 15° solids chute -> wheel inlet")
        print(f"    - Wheel classifier (centrifugal fine cut)")
        print(f"    - Multi-cyclone system (staged separation)")
        print(f"    - Bag filter (final collection)")
        
        # =========================================================================
        # CALCULATE OPTIMAL AIR FLOW FOR TARGET D50
        # =========================================================================
        if args.target_d50 is not None and assembly.zigzag is not None:
            # Physics constants
            g = 9.81  # m/s^2
            mu = 1.81e-5  # Pa.s (air viscosity)
            rho_p = 1450.0  # kg/m^3 (particle density)
            rho_f = 1.2  # kg/m^3 (air density)
            
            # Get zigzag cross-section
            zz_width = assembly.zigzag.params.channel_width
            zz_depth = assembly.zigzag.params.channel_depth
            zz_area = zz_width * zz_depth
            
            # Target d50 in meters
            d50_target = args.target_d50 * 1e-6
            
            # Calculate required air velocity from d50 formula:
            # d50 = sqrt(18*mu*v_air / (g*(rho_p - rho_f)))
            # => v_air = d50^2 * g * (rho_p - rho_f) / (18 * mu)
            v_air_required = (d50_target**2 * g * (rho_p - rho_f)) / (18 * mu)
            
            # Calculate required volumetric flow rate
            Q_required = v_air_required * zz_area
            
            print(f"\n  TARGET D50 CALCULATION:")
            print(f"    Target d50:          {args.target_d50:.1f} um")
            print(f"    Zigzag area:         {zz_area*1e6:.0f} mm^2 ({zz_width*1000:.0f} x {zz_depth*1000:.0f} mm)")
            print(f"    Required v_air:      {v_air_required*100:.2f} cm/s = {v_air_required:.4f} m/s")
            print(f"    Required Q:          {Q_required*1000:.4f} L/s = {Q_required*3600:.2f} m^3/h")
            
            # Check cyclone inlet velocity at this flow
            cyclone_inlet_area = 0.075 * 0.15  # 75x150 mm from geometry
            v_cyclone = Q_required / cyclone_inlet_area
            print(f"    Cyclone inlet v:     {v_cyclone:.2f} m/s", end="")
            if v_cyclone < 5:
                print(" [WARNING: Too slow for effective cyclone separation!]")
                print(f"\n  DESIGN ISSUE:")
                print(f"    The zigzag and cyclone have incompatible flow requirements:")
                print(f"    - Zigzag needs v={v_air_required*100:.1f} cm/s for d50={args.target_d50}um")
                print(f"    - Cyclone needs v=15-25 m/s for centrifugal separation")
                print(f"    At same flow rate, these cannot both be satisfied.")
                print(f"\n  OPTIONS:")
                
                # Option 1: Make zigzag smaller
                Q_cyclone_target = 15.0 * cyclone_inlet_area  # For 15 m/s cyclone inlet
                zz_area_small = Q_cyclone_target / v_air_required
                if zz_area_small > 0:
                    side_mm = np.sqrt(zz_area_small) * 1000
                    print(f"    1. RESIZE ZIGZAG LARGER (impractical: {side_mm:.0f}x{side_mm:.0f} mm)")
                
                # Option 2: Higher d50 with current geometry
                v_for_15ms_cyclone = 15.0 * cyclone_inlet_area / zz_area
                d50_at_15ms = np.sqrt(18 * mu * v_for_15ms_cyclone / (g * (rho_p - rho_f))) * 1e6
                print(f"    2. ACCEPT HIGHER d50 = {d50_at_15ms:.0f} um (use --air-flow 0.17)")
                print(f"       (all particles < {d50_at_15ms:.0f}um go to fines)")
                
                # Option 3: Two-stage air system
                print(f"    3. ADD SECONDARY AIR INJECTION before cyclones (not implemented)")
                
                # Option 4: Resize zigzag SMALLER for higher velocity at same Q
                # For cyclone at 15 m/s with small Q, make zigzag SMALLER
                # Target: same d50, but with Q that gives cyclone 15 m/s
                print(f"    4. REDESIGN: Smaller zigzag + higher flow for both")
                
                # Calculate: what zigzag size gives d50=35um AND cyclone v=15 m/s?
                # v_zigzag = d50^2 * g * (rho_p - rho_f) / (18 * mu) [fixed by physics]
                # Q = v_zigzag * A_zigzag = v_cyclone * A_cyclone
                # A_zigzag = v_cyclone * A_cyclone / v_zigzag
                # For v_cyclone = 15, A_cyclone = 0.01125, v_zigzag = 0.053:
                # A_zigzag = 15 * 0.01125 / 0.053 = 3.18 m² (too large)
                # So we need a SMALLER cyclone inlet!
                
                # Alternative: what cyclone inlet size works with current zigzag?
                # v_cyclone = Q / A_cyclone, need v_cyclone = 15
                # Q = v_zigzag * A_zigzag = 0.053 * 0.024 = 0.00128 m³/s
                # A_cyclone_needed = Q / 15 = 0.00128 / 15 = 85 mm²
                A_cyclone_needed = Q_required / 15.0
                d_cyclone_needed = np.sqrt(4 * A_cyclone_needed / np.pi) * 1000
                print(f"       For current geometry + d50={args.target_d50}um, need cyclone inlet D={d_cyclone_needed:.0f}mm")
                print(f"       (current cyclone inlet is ~75x150mm = 11250 mm²)")
            else:
                print(" [OK]")
            
            # Override the air flow rate
            args.air_flow = Q_required
            print(f"\n    => Setting air flow to {Q_required*1000:.4f} L/s ({Q_required*3600:.2f} m^3/h)")
        
        # Create physics config (legacy path: material/fluid from example, not created by classification)
        print("\nConfiguring physics simulation...")
        if args.material and args.material in ("yellow_pea", "faba_bean", "oat"):
            # Material created here (same as feed); classification only receives it for separation
            material = ParticleMaterial.create_food_powder(args.material, "whole")
            fluid = FluidConfig.air_at_stp()
            config = ClassificationFlowConfig(
                dt=args.dt,
                air_flow_rate_m3s=args.air_flow,
                bypass_ratio=args.bypass_ratio,
                num_particles=args.particles,
                device=args.device,
                turbulence_zigzag=turb_zigzag,
                turbulence_cyclone=turb_cyclone,
                material=material,
                fluid_config=fluid,
                wheel_rpm=args.wheel_rpm,
            )
            print(f"  Using FluidConfig + {args.material} whole flour")
        elif args.material:
            # Single fraction (protein/starch/fiber) or other preset - use material with FluidConfig
            source = "yellow_pea" if args.material in ("protein", "starch", "fiber") else args.material
            fraction = args.material if args.material in ("protein", "starch", "fiber") else "whole"
            material = ParticleMaterial.create_food_powder(source, fraction)
            config = ClassificationFlowConfig(
                dt=args.dt,
                air_flow_rate_m3s=args.air_flow,
                bypass_ratio=args.bypass_ratio,
                num_particles=args.particles,
                device=args.device,
                turbulence_zigzag=turb_zigzag,
                turbulence_cyclone=turb_cyclone,
                material=material,
                fluid_config=FluidConfig.air_at_stp(),
                wheel_rpm=args.wheel_rpm,
            )
            print(f"  Using FluidConfig + material: {material.name}")
        else:
            config = ClassificationFlowConfig(
                dt=args.dt,
                air_flow_rate_m3s=args.air_flow,
                bypass_ratio=args.bypass_ratio,
                num_particles=args.particles,
                device=args.device,
                turbulence_zigzag=turb_zigzag,
                turbulence_cyclone=turb_cyclone,
                wheel_rpm=args.wheel_rpm,
            )
    
    # Run operating-condition validation when requested
    if args.validate or args.diagnostics:
        total_flow_m3_h = config.air_flow_rate_m3s * 3600.0
        bypass = getattr(config, "bypass_ratio", 0.0)
        classification_flow_m3_h = total_flow_m3_h * (1.0 - bypass)
        rho = config.particle_density
        if config.material is not None and hasattr(config.material, "size_distribution"):
            sd = config.material.size_distribution
            min_um = sd.d_min * 1e6
            max_um = sd.d_max * 1e6
        else:
            min_um, max_um = 5.0, 100.0
        val = assembly.validate_system_configuration(
            air_flow_m3_h=total_flow_m3_h,
            particle_density=rho,
            min_particle_um=min_um,
            max_particle_um=max_um,
            classification_flow_m3_h=classification_flow_m3_h if bypass > 0 else None,
            cyclone_flow_m3_h=total_flow_m3_h if bypass > 0 else None,
        )
        print("\n" + "=" * 70)
        print("OPERATING CONDITION VALIDATION")
        print("=" * 70)
        print(f"  Air flow (total): {total_flow_m3_h:.0f} m3/h   Particle density: {rho:.0f} kg/m3")
        if bypass > 0:
            print(f"  Classification flow (zigzag): {classification_flow_m3_h:.1f} m3/h   Cyclone flow: {total_flow_m3_h:.0f} m3/h")
        print(f"  Particle size range: {min_um:.0f} - {max_um:.0f} um")
        zz = val.get("components", {}).get("zigzag", {})
        cy = val.get("components", {}).get("cyclones", {})
        if zz:
            v_bulk = zz.get('bulk_velocity_m_s', 0)
            v_sep = zz.get('separation_zone_velocity_m_s', 0)
            print(f"  Zigzag d50: {zz.get('d50_um', 0):.1f} um   v_zone: {v_sep:.2f} m/s (v_bulk: {v_bulk:.2f} m/s)")
        if cy and cy.get("stages"):
            for s in cy["stages"]:
                print(f"  {s['name']}: d50={s['actual_d50_um']:.2f} um (design {s['design_d50_um']:.0f} um)")
        for err in val.get("errors", []):
            print(f"  ERROR: {err}")
        for w in val.get("warnings", []):
            print(f"  WARNING: {w}")
        if not val.get("valid") and val.get("recommendation"):
            print(f"  RECOMMENDATION: {val['recommendation']}")
        if cy.get("recommended_flow_m3_h") is not None:
            print(f"  Recommended flow for design d50: {cy['recommended_flow_m3_h']:.0f} m3/h")
        print("=" * 70 + "\n")

    # Create simulator
    simulator = ClassificationFlowPhysicsSimulator(assembly, config)
    
    # =========================================================================
    # PRINT DETAILED FLOW PATH DIAGNOSTICS
    # =========================================================================
    if args.diagnostics or args.no_sim:
        simulator.print_detailed_flow_path()
        
        if args.no_sim:
            print("\n" + "=" * 70)
            print("DIAGNOSTICS ONLY MODE - Simulation skipped")
            print("=" * 70)
            return
    
    # Initialize particles (use integrated particle module when --material set)
    if getattr(assembly.params, 'use_preclassification', True):
        print("\nInitializing particles at venturi solids inlet...")
    else:
        print("\nInitializing particles at wheel inlet (15° solids chute)...")
    
    if args.material and args.material in ("yellow_pea", "faba_bean", "oat"):
        # Whole flour population (protein + starch + fiber) via reusable module
        simulator.initialize_whole_flour_population(
            source=args.material,
            num_particles=args.particles,
        )
    elif args.material and args.material in ("protein", "starch", "fiber"):
        # Single fraction via material
        material = ParticleMaterial.create_food_powder("yellow_pea", args.material)
        simulator.initialize_particles_from_material(
            material=material,
            num_particles=args.particles,
        )
    else:
        mean_dia_m = args.particle_dia * 1e-6
        std_dia_m = args.particle_std * 1e-6
        simulator.initialize_particles(
            num_particles=args.particles,
            mean_diameter=mean_dia_m,
            diameter_std=std_dia_m,
        )
    
    # Visualization setup
    plotter = None
    particle_actor = None
    
    if args.visualize and HAS_PYVISTA:
        print("\nSetting up 3D visualization using actual assembly geometry...")
        pv.set_plot_theme("document")
        plotter = pv.Plotter(title="Classification Flow - Protein Separation")
        plotter.set_background("white")
        plotter.camera.up = (0, 1, 0)

        # Transform from Z-up (geometry) to Y-up (viewport) coordinate system
        def to_y_up(vertices):
            """Transform vertices: (x, y, z) -> (x, z, -y) for Y-up display."""
            v = np.array(vertices, dtype=np.float64)
            result = v.copy()
            result[:, 1] = v[:, 2]    # new Y = old Z (vertical)
            result[:, 2] = -v[:, 1]   # new Z = -old Y (depth)
            return result

        def to_y_up_point(point):
            """Transform single point: (x, y, z) -> (x, z, -y)."""
            p = np.array(point, dtype=np.float64)
            return np.array([p[0], p[2], -p[1]])

        # ============================================
        # BUILD MESH FROM ACTUAL ASSEMBLY
        # ============================================
        print("  Building assembly mesh...")
        assembly.build_mesh()

        # Get component positions
        comp_positions = assembly.get_component_positions()

        # ============================================
        # VENTURI EDUCTOR (when present)
        # ============================================
        if assembly.venturi is not None:
            print("  Adding venturi eductor...")
            v_vent, i_vent, _ = assembly.venturi.generate_mesh()
            v_vent = v_vent + np.array(comp_positions['venturi'])
            v_vent = to_y_up(v_vent)
            faces_vent = np.hstack([[3] + list(face) for face in i_vent.reshape(-1, 3)])
            venturi_mesh = pv.PolyData(v_vent, faces_vent)
            plotter.add_mesh(venturi_mesh, color='#3498DB', opacity=0.5, label='Venturi')

        # ============================================
        # ZIGZAG CLASSIFIER (when present)
        # ============================================
        if assembly.zigzag is not None:
            print("  Adding zigzag classifier...")
            v_zz, i_zz, _ = assembly.zigzag.generate_mesh()
            v_zz = v_zz + np.array(comp_positions['zigzag'])
            v_zz = to_y_up(v_zz)
            faces_zz = np.hstack([[3] + list(face) for face in i_zz.reshape(-1, 3)])
            zigzag_mesh = pv.PolyData(v_zz, faces_zz)
            plotter.add_mesh(zigzag_mesh, color='#2ECC71', opacity=0.5, label='Zigzag')

        # ============================================
        # WHEEL CLASSIFIER (with animation: rotation = omega * time)
        # ============================================
        print("  Adding wheel classifier (animated)...")
        v_wheel, i_wheel, _ = assembly.wheel_classifier.generate_mesh()
        wheel_pos = np.array(comp_positions['wheel_classifier'])
        # Wheel classifier center from params (local 0,0,0 is housing center)
        v_wheel_base = v_wheel + wheel_pos
        v_wheel_base = to_y_up(v_wheel_base)
        wheel_center_world = to_y_up_point(wheel_pos)
        faces_wheel = np.hstack([[3] + list(face) for face in i_wheel.reshape(-1, 3)])
        wheel_mesh_base = pv.PolyData(v_wheel_base, faces_wheel)
        wheel_actor = plotter.add_mesh(wheel_mesh_base, color='#9B59B6', opacity=0.6, label='Wheel')

        # ============================================
        # MULTI-CYCLONE SYSTEM (actual geometry)
        # ============================================
        print("  Adding multi-cyclone system...")
        v_mc, i_mc, _ = assembly.multi_cyclone.generate_mesh()
        v_mc = v_mc + np.array(comp_positions['multi_cyclone'])
        v_mc = to_y_up(v_mc)
        faces_mc = np.hstack([[3] + list(face) for face in i_mc.reshape(-1, 3)])
        cyclone_mesh = pv.PolyData(v_mc, faces_mc)
        plotter.add_mesh(cyclone_mesh, color='#E74C3C', opacity=0.5, label='Cyclones')

        # ============================================
        # BAG FILTER (actual geometry)
        # ============================================
        print("  Adding bag filter...")
        v_bf, i_bf, _ = assembly.bag_filter.generate_mesh()
        v_bf = v_bf + np.array(comp_positions['bag_filter'])
        v_bf = to_y_up(v_bf)
        faces_bf = np.hstack([[3] + list(face) for face in i_bf.reshape(-1, 3)])
        bagfilter_mesh = pv.PolyData(v_bf, faces_bf)
        plotter.add_mesh(bagfilter_mesh, color='#95A5A6', opacity=0.5, label='Bag Filter')

        # ============================================
        # DUCTWORK (actual geometry)
        # ============================================
        print("  Adding ductwork...")
        for idx, (duct, position) in enumerate(assembly._duct_sections):
            v_duct, i_duct, _ = duct.generate_mesh()
            v_duct = v_duct + np.array(position)
            v_duct = to_y_up(v_duct)
            faces_duct = np.hstack([[3] + list(face) for face in i_duct.reshape(-1, 3)])
            duct_mesh = pv.PolyData(v_duct, faces_duct)
            label = 'Ductwork' if idx == 0 else None
            plotter.add_mesh(duct_mesh, color='#7F8C8D', opacity=0.4, label=label)
        
        # ============================================
        # LABELS FOR KEY PORTS
        # ============================================
        print("  Adding port labels...")

        # Get port positions from assembly and transform to Y-up
        try:
            if assembly.zigzag is not None:
                coarse_pos = assembly.get_port_world_position('zigzag', 'coarse_outlet')
                coarse_pos = to_y_up_point(coarse_pos)
                plotter.add_point_labels([coarse_pos - np.array([0, 0.1, 0])],
                                        ["COARSE\n(Starch)"], font_size=12,
                                        text_color='#8B4513', point_size=0)
        except (KeyError, AttributeError):
            pass

        try:
            if assembly.zigzag is not None:
                fines_pos = assembly.get_port_world_position('zigzag', 'fines_outlet')
            else:
                fines_pos = assembly.get_port_world_position('wheel_classifier', 'fines_outlet')
            fines_pos = to_y_up_point(fines_pos)
            plotter.add_point_labels([fines_pos + np.array([0, 0.1, 0])],
                                    ["FINES"], font_size=10,
                                    text_color='#2ECC71', point_size=0)
        except (KeyError, AttributeError):
            pass

        # Cyclone dust outlets
        try:
            for dust_name in ['primary_dust', 'secondary_dust', 'tertiary_dust']:
                dust_pos = assembly.get_port_world_position('multi_cyclone', dust_name)
                dust_pos = to_y_up_point(dust_pos)
                label = "PROTEIN" if 'tertiary' in dust_name else dust_name.split('_')[0].title()
                color = '#9B59B6' if 'tertiary' in dust_name else '#E74C3C'
                plotter.add_point_labels([dust_pos - np.array([0, 0.1, 0])],
                                        [label], font_size=10,
                                        text_color=color, point_size=0)
        except (KeyError, AttributeError):
            pass

        try:
            dust_pos = assembly.get_port_world_position('bag_filter', 'dust_outlet')
            dust_pos = to_y_up_point(dust_pos)
            plotter.add_point_labels([dust_pos - np.array([0, 0.1, 0])],
                                    ["Bag Dust"], font_size=10,
                                    text_color='#95A5A6', point_size=0)
        except (KeyError, AttributeError):
            pass
        
        # ============================================
        # INFO TEXT
        # ============================================
        plotter.add_text(
            "CLASSIFICATION FLOW\nInitializing...",
            position='upper_left', font_size=10, color='black', name='sim_info'
        )
        
        # ============================================
        # SEPARATION STATS TEXT
        # ============================================
        plotter.add_text(
            "Separation:\n---",
            position='upper_right', font_size=10, color='black', name='sep_info'
        )
        
        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_axes()
        plotter.reset_camera()
        plotter.camera.azimuth = -30
        plotter.camera.elevation = 20
        plotter.camera.zoom(0.8)
        plotter.show(interactive_update=True, auto_close=False)
    
    # =========================================================================
    # MULTI-PASS RECIRCULATION SETUP
    # =========================================================================
    num_passes = args.passes if args.recirculate else 1
    recirculate_fractions = args.recirculate or []
    # Validate fraction names
    valid_fractions = {'cy1', 'cy2', 'cy3', 'wheel_coarse', 'zigzag_coarse', 'bagfilter'}
    for frac in recirculate_fractions:
        if frac not in valid_fractions:
            print(f"ERROR: Unknown recirculation fraction '{frac}'. Valid: {', '.join(sorted(valid_fractions))}")
            return

    if num_passes > 1:
        print(f"\n  MULTI-PASS RECIRCULATION: {num_passes} passes")
        print(f"    Recirculating: {', '.join(recirculate_fractions)}")
        if args.recirculate_wheel_rpm is not None:
            print(f"    Pass 2+ wheel RPM: {args.recirculate_wheel_rpm:.0f}")
        if args.recirculate_time is not None:
            print(f"    Pass 2+ sim time: {args.recirculate_time:.1f} s")
        if args.attrition > 0:
            print(f"    Venturi attrition: {args.attrition*100:.0f}% per pass "
                  f"(min {args.attrition_min:.1f} µm)")
        else:
            print(f"    Venturi attrition: disabled")
        if args.recirculate_to_wheel:
            print(f"    Refeed path: wheel → cyclones (skip preclassification)")
        else:
            print(f"    Refeed path: venturi → zigzag → wheel → cyclones (full path)")

    # Cumulative collection across all passes (for final summary)
    cumulative_counts = {
        'coarse': 0, 'wheel_coarse': 0, 'cyclone_1': 0, 'cyclone_2': 0,
        'cyclone_3_protein': 0, 'bagfilter': 0, 'escaped': 0, 'active': 0,
    }
    # Cumulative cyclone particle size data (diameters in µm, for merged stats)
    cumulative_cy_diameters = {'cyclone_1': [], 'cyclone_2': [], 'cyclone_3_protein': []}
    pass_results = []  # Per-pass separation counts
    original_feed_count = args.particles  # Total particles from original feed

    for pass_num in range(1, num_passes + 1):
        pass_sim_time = args.time
        if pass_num > 1 and args.recirculate_time is not None:
            pass_sim_time = args.recirculate_time

        # Run simulation
        print("\n" + "-" * 70)
        if num_passes > 1:
            print(f"RUNNING SIMULATION — PASS {pass_num}/{num_passes}")
        else:
            print("RUNNING SIMULATION")
        print("-" * 70)
        print(f"  Time: {pass_sim_time:.1f} s")
        print(f"  dt:   {args.dt*1000:.2f} ms")
        print(f"  Steps: {int(pass_sim_time / args.dt):,}")
        if args.bypass_ratio > 0:
            Q_class = args.air_flow * (1.0 - args.bypass_ratio)
            print(f"  Air flow: {args.air_flow * 3600:.0f} m³/h total, "
                  f"{Q_class * 3600:.1f} m³/h classification, "
                  f"{args.bypass_ratio*100:.1f}% bypass")
        else:
            print(f"  Air flow: {args.air_flow * 3600:.0f} m³/h")
        if getattr(simulator, 'use_preclassification', True):
            print(f"  Zigzag d50: {simulator.zigzag_d50 * 1e6:.1f} µm")
        print(f"  Wheel d50: {simulator.wheel_d50 * 1e6:.1f} µm")
        if pass_num == 1:
            if config.continuous_feeding:
                print(f"  Feeding: continuous at {config.particle_feed_rate:.0f} particles/s")
                m_per_particle = config.particle_density * (np.pi / 6.0) * config.visual_particle_diameter**3
                print(f"  Feed mass flow: {config.particle_feed_rate * m_per_particle * 3600:.1f} kg/h")
                print(f"  Max loading ratio: {config.max_loading_ratio:.1f}")
            else:
                print(f"  Feeding: batch (all particles active at t=0)")
        else:
            print(f"  Recirculation pass: {simulator.state.total_particles_to_feed} particles from {', '.join(recirculate_fractions)}")
        print("-" * 70)

        total_steps = int(pass_sim_time / args.dt)
        print_interval = max(1, total_steps // 20)  # ~20 console updates

        # Animation timing
        target_fps = 30
        frame_interval = 1.0 / target_fps
        steps_per_frame = max(1, int(frame_interval / args.dt))

        # Start simulation
        start_time = time.time()
        last_wall_time = time.time()
        last_print_step = -print_interval

        step = 0
        early_exit = False
        while step < total_steps:
            # Run multiple simulation steps per visual frame
            frame_steps = min(steps_per_frame, total_steps - step)
            for _ in range(frame_steps):
                simulator.step()
                step += 1

            # Get current state
            zone_counts = simulator.get_zone_counts()
            sep_counts = simulator.get_separation_counts()

            # Early termination: all particles fed and none still active
            all_fed = (simulator.state.particles_fed >= simulator.state.total_particles_to_feed)
            if all_fed and sep_counts['active'] == 0:
                if not early_exit:
                    early_exit = True
                    print(f"  [Early exit] All particles settled at t={simulator.state.time:.1f}s "
                          f"(step {step:,}/{total_steps:,})")
                break

            # Console output at intervals
            if step - last_print_step >= print_interval or step >= total_steps:
                last_print_step = step
                progress = 100.0 * step / total_steps

                active = sep_counts['active']
                coarse = sep_counts['coarse']           # Zigzag coarse (starch)
                wheel_coarse = sep_counts.get('wheel_coarse', 0)  # Wheel coarse (starch)
                cy1 = sep_counts['cyclone_1']
                cy2 = sep_counts['cyclone_2']
                cy3 = sep_counts['cyclone_3_protein']
                bag = sep_counts['bagfilter']

                status = f"  [{progress:5.1f}%] t={simulator.state.time:5.2f}s"
                # Show feed progress if continuous feeding
                if config.continuous_feeding or pass_num > 1:
                    fed = simulator.state.particles_fed
                    total_to_feed = simulator.state.total_particles_to_feed
                    status += f" | Fed:{fed:5d}/{total_to_feed}"
                status += f" | Active:{active:5d} Zc:{coarse:5d} Wc:{wheel_coarse:5d}"
                status += f" Cy1:{cy1:5d} Cy2:{cy2:5d} Cy3:{cy3:5d} Bag:{bag:5d}"
                # Show zone breakdown for active particles
                zz = zone_counts.get('zigzag', 0)
                fp = zone_counts.get('fines_path', 0)
                vent = zone_counts.get('venturi', 0)
                duct_vz = zone_counts.get('duct_v_z', 0)
                wh = zone_counts.get('wheel_housing', 0)
                wf = zone_counts.get('wheel_fines', 0)
                wch = zone_counts.get('wheel_coarse_hopper', 0)
                cy1_z = zone_counts.get('cyclone_1', 0)
                cy2_z = zone_counts.get('cyclone_2', 0)
                cy3_z = zone_counts.get('cyclone_3', 0)
                status += f"  [zz:{zz:4d} fp:{fp:4d} wh:{wh:4d} wf:{wf:4d} wch:{wch:4d} c1:{cy1_z:4d} c2:{cy2_z:4d} c3:{cy3_z:4d}]"
                print(status)

            # Update visualization every frame
            if plotter is not None:
                # Calculate wall-clock dt
                current_wall_time = time.time()
                wall_dt = current_wall_time - last_wall_time
                last_wall_time = current_wall_time

                # ============================================
                # UPDATE PARTICLES
                # ============================================
                positions = simulator.get_positions()
                velocities = simulator.get_velocities()
                diameters = simulator.get_diameters()
                zones = simulator.get_zones()

                if len(positions) > 0:
                    # Filter finite positions
                    finite = np.isfinite(positions).all(axis=1)
                    if np.any(finite):
                        positions = np.asarray(positions[finite], dtype=np.float64)
                        velocities = np.asarray(velocities[finite], dtype=np.float64)
                        diameters = np.asarray(diameters[finite], dtype=np.float64)
                        zones = zones[finite]
                        n_show = len(positions)
                    else:
                        n_show = 0

                    if n_show > 0:
                        # Transform positions to Y-up for display
                        positions = to_y_up(positions)

                        # Create particle point cloud
                        speeds = np.linalg.norm(velocities, axis=1)

                        particle_mesh = pv.PolyData(positions)
                        particle_mesh['velocity'] = speeds

                        # Point size based on particle diameter (scale up for visibility)
                        particle_dia_um = float(np.mean(diameters)) * 1e6
                        point_size = max(6, min(15, int(particle_dia_um / 3)))

                        try:
                            new_actor = plotter.add_mesh(
                                particle_mesh,
                                scalars='velocity',
                                cmap='plasma',  # Velocity coloring
                                point_size=point_size,
                                render_points_as_spheres=True,
                                opacity=0.9,
                                clim=[0, 20.0],
                                show_scalar_bar=True,
                                scalar_bar_args={'title': 'Velocity (m/s)', 'n_labels': 3},
                            )
                            if particle_actor is not None:
                                try:
                                    plotter.remove_actor(particle_actor)
                                except Exception:
                                    pass
                            particle_actor = new_actor
                        except Exception as e:
                            if "plane" not in str(e).lower():
                                raise
                            pass  # Keep previous frame

                # ============================================
                # UPDATE WHEEL AND MOTOR ROTATION (coupled to physics omega * time)
                # ============================================
                try:
                    omega = getattr(simulator, 'wheel_omega', 0.0)
                    t = simulator.state.time
                    angle_rad = omega * t
                    angle_deg = np.degrees(angle_rad)
                    wheel_mesh_rotated = wheel_mesh_base.copy(deep=True)
                    wheel_mesh_rotated.rotate_y(angle_deg, point=wheel_center_world, in_place=True)
                    plotter.remove_actor(wheel_actor)
                    wheel_actor = plotter.add_mesh(wheel_mesh_rotated, color='#9B59B6', opacity=0.6, label='Wheel')
                except Exception:
                    pass

                # ============================================
                # UPDATE INFO TEXT
                # ============================================
                pass_label = f" (Pass {pass_num}/{num_passes})" if num_passes > 1 else ""
                info_text = (
                    f"CLASSIFICATION FLOW{pass_label}\n"
                    f"Protein Separation\n"
                    f"\n"
                    f"Venturi:   {zone_counts.get('venturi', 0):4d}\n"
                    f"Zigzag:    {zone_counts.get('zigzag', 0):4d}\n"
                    f"Cyclone 1: {zone_counts.get('cyclone_1', 0):4d}\n"
                    f"Cyclone 2: {zone_counts.get('cyclone_2', 0):4d}\n"
                    f"Cyclone 3: {zone_counts.get('cyclone_3', 0):4d}\n"
                    f"Bag Filt:  {zone_counts.get('bagfilter', 0):4d}\n"
                    f"\n"
                    f"t = {simulator.state.time:.2f}s"
                )
                plotter.add_text(info_text, position='upper_left', font_size=10,
                                color='black', name='sim_info')

                # ============================================
                # UPDATE SEPARATION STATS
                # ============================================
                total_collected = (sep_counts['coarse'] + sep_counts.get('wheel_coarse', 0) +
                                 sep_counts['cyclone_1'] + sep_counts['cyclone_2'] +
                                 sep_counts['cyclone_3_protein'] + sep_counts['bagfilter'])

                if total_collected > 0:
                    pct_coarse = 100 * sep_counts['coarse'] / total_collected
                    pct_wc = 100 * sep_counts.get('wheel_coarse', 0) / total_collected
                    pct_cy1 = 100 * sep_counts['cyclone_1'] / total_collected
                    pct_cy2 = 100 * sep_counts['cyclone_2'] / total_collected
                    pct_cy3 = 100 * sep_counts['cyclone_3_protein'] / total_collected
                    pct_bag = 100 * sep_counts['bagfilter'] / total_collected
                else:
                    pct_coarse = pct_wc = pct_cy1 = pct_cy2 = pct_cy3 = pct_bag = 0

                sep_text = (
                    f"SEPARATION\n"
                    f"----------\n"
                    f"Zc (zigzag): {sep_counts['coarse']:4d} ({pct_coarse:4.1f}%)\n"
                    f"Wc (wheel):  {sep_counts.get('wheel_coarse', 0):4d} ({pct_wc:4.1f}%)\n"
                    f"Cy1:         {sep_counts['cyclone_1']:4d} ({pct_cy1:4.1f}%)\n"
                    f"Cy2:         {sep_counts['cyclone_2']:4d} ({pct_cy2:4.1f}%)\n"
                    f"Cy3:         {sep_counts['cyclone_3_protein']:4d} ({pct_cy3:4.1f}%)\n"
                    f"Bag:         {sep_counts['bagfilter']:4d} ({pct_bag:4.1f}%)\n"
                    f"----------\n"
                    f"Protein: Cy3+Bag | Starch: Zc+Wc"
                )
                plotter.add_text(sep_text, position='upper_right', font_size=10,
                                color='black', name='sep_info')

                plotter.update()
                time.sleep(0.001)

        elapsed = time.time() - start_time

        print("-" * 70)
        if num_passes > 1:
            print(f"PASS {pass_num}/{num_passes} COMPLETE")
        else:
            print("SIMULATION COMPLETE")
        print("-" * 70)
        print(f"  Wall time: {elapsed:.1f} s")
        print(f"  Sim time:  {simulator.state.time:.2f} s")
        print(f"  Steps:     {simulator.state.step:,}")
        print(f"  Rate:      {simulator.state.step / elapsed:.0f} steps/s")
        if config.continuous_feeding and pass_num == 1:
            fed = simulator.state.particles_fed
            total_to_feed = simulator.state.total_particles_to_feed
            print(f"  Feeding:   {fed}/{total_to_feed} particles fed ({100*fed/max(1,total_to_feed):.1f}%)")
            print(f"  Feed rate: {config.particle_feed_rate:.0f} particles/s")

        # Print pass separation summary
        simulator.print_separation_summary()

        # =====================================================================
        # ACCUMULATE PASS RESULTS
        # =====================================================================
        pass_sep = simulator.get_separation_counts()
        pass_results.append({'pass': pass_num, 'counts': pass_sep})

        # For non-recirculated fractions, add to cumulative totals
        # Recirculated fractions will be re-processed in the next pass
        for key in cumulative_counts:
            if key == 'active':
                continue  # Don't accumulate active (transient)
            # If this fraction is being recirculated, don't add to cumulative yet
            # (it will be re-classified in the next pass)
            # Map separation count keys to recirculate fraction names
            sep_to_frac = {
                'cyclone_1': 'cy1', 'cyclone_2': 'cy2',
                'cyclone_3_protein': 'cy3', 'wheel_coarse': 'wheel_coarse',
                'coarse': 'zigzag_coarse', 'bagfilter': 'bagfilter',
            }
            frac_name = sep_to_frac.get(key)
            if frac_name in recirculate_fractions and pass_num < num_passes:
                # This fraction will be recirculated — don't collect yet
                pass
            else:
                cumulative_counts[key] += pass_sep.get(key, 0)

        # Accumulate cyclone particle size data for cumulative stats
        try:
            cy_stats = simulator.get_cyclone_particle_size_stats()
            zones_np = simulator.get_zones()
            diameters_np = simulator.get_diameters()
            for key, zone_id in [('cyclone_1', 55), ('cyclone_2', 56), ('cyclone_3_protein', 57)]:
                mask = (zones_np == zone_id)
                if np.any(mask):
                    d_um = diameters_np[mask] * 1e6
                    sep_to_frac = {
                        'cyclone_1': 'cy1', 'cyclone_2': 'cy2', 'cyclone_3_protein': 'cy3',
                    }
                    frac_name = sep_to_frac.get(key)
                    if frac_name in recirculate_fractions and pass_num < num_passes:
                        pass  # Will be reclassified
                    else:
                        cumulative_cy_diameters[key].extend(d_um.tolist())
        except Exception:
            pass

        # Remaining active particles at end of pass count as "still active"
        cumulative_counts['active'] += pass_sep.get('active', 0)

        # =====================================================================
        # RECIRCULATION: Extract and re-initialize for next pass
        # =====================================================================
        if pass_num < num_passes and recirculate_fractions:
            print(f"\n{'='*70}")
            print(f"RECIRCULATION: Extracting {', '.join(recirculate_fractions)} for pass {pass_num + 1}")
            print(f"{'='*70}")

            particle_data = simulator.extract_collected_particles(recirculate_fractions)
            frac_counts = particle_data['fraction_counts']
            for frac_name, frac_count in frac_counts.items():
                print(f"  {frac_name}: {frac_count} particles")
            print(f"  Total to recirculate: {particle_data['count']} particles")

            if particle_data['count'] == 0:
                print("  No particles to recirculate — stopping early.")
                break

            # Print size distribution of refeed material
            d_um = particle_data['diameters'] * 1e6
            print(f"  Refeed PSD: {d_um.min():.1f} – {d_um.max():.1f} µm, "
                  f"mean={d_um.mean():.1f} µm, median={np.median(d_um):.1f} µm")

            # Determine wheel RPM for next pass
            next_wheel_rpm = args.recirculate_wheel_rpm  # None = keep current

            # Re-initialize: particles return to feed hopper and flow through
            # the feed system (gravity chute ~21s) before arriving at venturi
            # solids inlet. Use continuous feeding to match real trickle-in.
            # Feed residence time from feedclass physics (if available).
            _feed_res_time = 0.0
            if args.full_system:
                _feed_res_time = feed_result.get('total_residence_time_s', 21.0)
            n_recirc = simulator.reinitialize_from_particles(
                particle_data,
                initial_velocity=None,  # auto from feed kinetics
                continuous_feeding=None,  # auto (continuous when feed time > 0)
                wheel_rpm=next_wheel_rpm,
                attrition_factor=args.attrition,
                attrition_min_diameter_m=args.attrition_min * 1e-6,  # µm to m
                skip_preclassification=args.recirculate_to_wheel,
                feed_residence_time_s=_feed_res_time,
            )
            if n_recirc == 0:
                print("  Recirculation initialization failed — stopping.")
                break

    # =========================================================================
    # FINAL DIAGNOSTICS (after all passes)
    # =========================================================================
    if args.diagnostics:
        print("\n" + "=" * 70)
        print("FINAL DIAGNOSTICS - Zone Distribution (last pass)")
        print("=" * 70)
        zone_counts = simulator.get_zone_counts()
        print(f"\n  Active Particle Distribution by Zone:")
        for zone_name, count in zone_counts.items():
            if count > 0:
                print(f"    {zone_name:20s}: {count:5d}")

        # Print detailed separation analysis (path order; balance = sum of all)
        sep = simulator.get_separation_counts()
        total = sum(sep.values())  # Full balance: every particle in exactly one bucket

        if total > 0:
            print(f"\n  Separation Analysis (particle balance):")
            print(f"    Total (all destinations): {total:5d} particles")
            print(f"    Still in system:          {sep['active']:5d} particles")

            print(f"\n  Collection by Outlet (path order):")
            outlets = [
                ('Zigzag coarse (starch)', 'coarse', '#8B4513'),
                ('Wheel coarse (starch)', 'wheel_coarse', '#A0522D'),
                ('Cyclone 1', 'cyclone_1', '#E74C3C'),
                ('Cyclone 2', 'cyclone_2', '#E67E22'),
                ('Cyclone 3 (Protein)', 'cyclone_3_protein', '#9B59B6'),
                ('Bag Filter', 'bagfilter', '#95A5A6'),
                ('Escaped', 'escaped', '#FF0000'),
            ]

            for label, key, _ in outlets:
                count = sep[key]
                pct = 100.0 * count / total if total > 0 else 0
                bar_len = int(pct / 2)
                bar = '#' * bar_len
                print(f"    {label:20s}: {count:5d} ({pct:5.1f}%) |{bar}")

            # Protein recovery estimate
            protein_outlets = sep['cyclone_3_protein'] + sep['bagfilter']
            starch_outlets = sep['coarse'] + sep.get('wheel_coarse', 0)
            print(f"\n  Protein Recovery Estimate:")
            print(f"    Protein-rich fractions (Cy3 + Bag): {protein_outlets:5d} ({100*protein_outlets/max(1,total):.1f}%)")
            print(f"    Starch-rich fractions (Zc + Wc):     {starch_outlets:5d} ({100*starch_outlets/max(1,total):.1f}%)")

            # Cyclone particle sizes (design d50 vs actual)
            try:
                cy_stats = simulator.get_cyclone_particle_size_stats()
                print(f"\n  Cyclone particle sizes (design d50 vs collected):")
                for key, title in [('cyclone_1', 'Cy1 (coarse fines)'), ('cyclone_2', 'Cy2 (medium)'), ('cyclone_3_protein', 'Cy3 (protein)')]:
                    s = cy_stats.get(key, {})
                    n = s.get('count', 0)
                    design = s.get('design_d50_um')
                    mean_d = s.get('mean_d_um')
                    med_d = s.get('median_d_um')
                    design_str = f"design d50={design:.0f} µm" if design is not None else "design d50=N/A"
                    if n > 0 and mean_d is not None:
                        med_str = f"  median={med_d:.1f} µm" if med_d is not None else ""
                        print(f"    {title:22s}: N={n:5d}  {design_str}  →  mean={mean_d:.1f} µm{med_str}")
                    else:
                        print(f"    {title:22s}: N={n:5d}  {design_str}")
            except Exception:
                pass

        # Per-zone particle size analysis
        zones_arr = simulator.get_zones()
        diameters_arr = simulator.get_diameters()
        is_active_arr = simulator.state.is_active.numpy()[:simulator.state.particles_active]

        print(f"\n  Particle Size Analysis by Zone (active particles):")
        zone_labels = [
            ('Venturi (0-2)', lambda z: (z >= 0) & (z <= 2)),
            ('Duct V→Z (10)', lambda z: z == 10),
            ('Zigzag (20-21)', lambda z: (z == 20) | (z == 21)),
            ('Fines path (22)', lambda z: z == 22),
            ('Elbow/Duct (40-41)', lambda z: (z == 40) | (z == 41)),
            ('Cyclone 1 (50)', lambda z: z == 50),
            ('Cyclone 2 (51)', lambda z: z == 51),
            ('Cyclone 3 (52)', lambda z: z == 52),
        ]
        for label, zone_mask_fn in zone_labels:
            mask = zone_mask_fn(zones_arr) & (is_active_arr == 1)
            n = int(np.sum(mask))
            if n > 0:
                d_um = diameters_arr[mask] * 1e6  # Convert to µm
                print(f"    {label:22s}: n={n:5d}  d=[{np.min(d_um):5.1f}, {np.median(d_um):5.1f}, {np.max(d_um):5.1f}] µm (min/med/max)")

    # =========================================================================
    # CUMULATIVE MULTI-PASS SUMMARY
    # =========================================================================
    if num_passes > 1:
        print("\n" + "=" * 70)
        print(f"CUMULATIVE RESULTS — {len(pass_results)} PASSES")
        print(f"  Recirculated fractions: {', '.join(recirculate_fractions)}")
        print("=" * 70)

        cum_total = sum(cumulative_counts.values())
        print(f"\n  Original feed: {original_feed_count} particles")
        print(f"  Cumulative collection (all passes combined):\n")

        labels = [
            ('coarse', 'Zigzag coarse (starch):  '),
            ('wheel_coarse', 'Wheel coarse (starch):   '),
            ('cyclone_1', 'Cyclone 1 (fines 1):     '),
            ('cyclone_2', 'Cyclone 2 (fines 2):     '),
            ('cyclone_3_protein', 'Cyclone 3 (PROTEIN):     '),
            ('bagfilter', 'Bag filter:               '),
            ('escaped', 'Escaped (loss):           '),
            ('active', 'Still active (residual):   '),
        ]
        for key, label in labels:
            c = cumulative_counts.get(key, 0)
            pct_of_feed = 100.0 * c / max(1, original_feed_count)
            pct_of_cum = 100.0 * c / max(1, cum_total)
            bar_len = int(pct_of_feed / 2)
            bar = '#' * bar_len
            print(f"    {label} {c:5d} ({pct_of_feed:5.1f}% of feed, {pct_of_cum:5.1f}% of collected) |{bar}")

        # Cumulative protein recovery
        protein_cum = cumulative_counts['cyclone_3_protein'] + cumulative_counts['bagfilter']
        starch_cum = cumulative_counts['coarse'] + cumulative_counts['wheel_coarse']
        print(f"\n  Cumulative Protein Recovery:")
        print(f"    Protein-rich (Cy3 + Bag):  {protein_cum:5d} ({100*protein_cum/max(1,original_feed_count):.1f}% of feed)")
        print(f"    Starch-rich (Zc + Wc):     {starch_cum:5d} ({100*starch_cum/max(1,original_feed_count):.1f}% of feed)")

        # Cumulative cyclone particle sizes
        print(f"\n  Cumulative Cyclone Particle Sizes (all passes):")
        for key, title in [('cyclone_1', 'Cy1'), ('cyclone_2', 'Cy2'), ('cyclone_3_protein', 'Cy3 (protein)')]:
            d_list = cumulative_cy_diameters.get(key, [])
            if d_list:
                d_arr = np.array(d_list)
                print(f"    {title:20s}: N={len(d_arr):5d}  mean={d_arr.mean():.1f} µm  median={np.median(d_arr):.1f} µm  "
                      f"range=[{d_arr.min():.1f}, {d_arr.max():.1f}] µm")
            else:
                print(f"    {title:20s}: N=    0")

        # Per-pass breakdown table
        print(f"\n  Per-Pass Breakdown:")
        header = f"    {'Pass':>4s}  {'Feed':>6s}  {'Zc':>6s}  {'Wc':>6s}  {'Cy1':>6s}  {'Cy2':>6s}  {'Cy3':>6s}  {'Bag':>5s}  {'Active':>6s}"
        print(header)
        print(f"    {'----':>4s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'-----':>5s}  {'------':>6s}")
        for pr in pass_results:
            p = pr['pass']
            c = pr['counts']
            feed_n = sum(c.values())
            print(f"    {p:4d}  {feed_n:6d}  {c['coarse']:6d}  {c.get('wheel_coarse',0):6d}  "
                  f"{c['cyclone_1']:6d}  {c['cyclone_2']:6d}  {c['cyclone_3_protein']:6d}  "
                  f"{c['bagfilter']:5d}  {c['active']:6d}")

        print("=" * 70)

    print("=" * 70)

    if plotter is not None:
        print("\nVisualization window open. Close to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
