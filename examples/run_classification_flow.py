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

    # Turbulence intensity
    python examples/run_classification_flow.py --turbulence 0.2

    # Compute device
    python examples/run_classification_flow.py --device cpu
    python examples/run_classification_flow.py --device cuda -n 5000

    # 3D visualization (requires pyvista)
    python examples/run_classification_flow.py --visualize
    python examples/run_classification_flow.py -v --particles 2000 --time 10
    python examples/run_classification_flow.py --visualize --material yellow_pea --diagnostics

    # Combined examples
    python examples/run_classification_flow.py --diagnostics --visualize --particles 5000 --time 10
    python examples/run_classification_flow.py --material yellow_pea --air-flow 0.2 -n 1000 -t 5 -v
    python examples/run_classification_flow.py --no-sim --material faba_bean --diagnostics

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
    --no-sim              Print diagnostics only, skip simulation
    --target-d50          Target cut size in microns (auto air flow)
    --zigzag-width        Override zigzag channel width in mm
    --zigzag-depth        Override zigzag channel depth in mm
    --turbulence          Turbulent intensity (default: 0.15)
    --device              cuda or cpu (default: cuda)
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
        "--blower-rpm", type=float, default=None,
        help="VFD blower speed in RPM (overrides --air-flow via fan law: "
             "Q = 3000 m³/h × RPM/3000). Design: 3000 RPM = 3000 m³/h."
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
        help="Turbulent intensity (default: 0.15)"
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
             "Controls flow restriction and system operating point."
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
    
    args = parser.parse_args()

    # VFD: convert blower RPM to air flow via fan law
    if args.blower_rpm is not None:
        # Fan law: Q ∝ N → Q = Q_design × (N / N_design)
        # Design point: 3000 m³/h at 3000 RPM
        Q_DESIGN_M3H = 3000.0
        RPM_DESIGN = 3000.0
        Q_vfd_m3h = Q_DESIGN_M3H * (args.blower_rpm / RPM_DESIGN)
        args.air_flow = Q_vfd_m3h / 3600.0  # convert to m³/s

    print("=" * 70)
    print("PHYSICS-BASED CLASSIFICATION FLOW SIMULATION")
    print("  Protein/Starch Separation via Air Classification")
    print("=" * 70)
    if args.blower_rpm is not None:
        print(f"  VFD: {args.blower_rpm:.0f} RPM -> {args.air_flow * 3600:.0f} m³/h (fan law)")
    if args.bypass_ratio > 0:
        print(f"  Bypass: {args.bypass_ratio*100:.1f}% around venturi+zigzag")
    
    # Import modules
    from airclassifier.simulation.classification_flow_physics import (
        ClassificationFlowPhysicsSimulator,
        ClassificationFlowConfig,
        compute_venturi_physics_from_air_and_feed,
    )
    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly,
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
        )
        print("\n[FULL SYSTEM] Air -> Venturi -> Feed -> Venturi -> Classification")
        fluid = FluidConfig.air_at_stp()
        material = None
        if args.material:
            fraction = "whole" if args.material in ("yellow_pea", "faba_bean", "oat") else args.material
            material = ParticleMaterial.create_food_powder(args.material, fraction)
        particle_dia_m = args.particle_dia * 1e-6
        particle_density = material.density if material else 1420.0
        Q_m3s = args.air_flow
        # Build classification params with optional throat diameter override
        classification_params = None
        if args.throat_diameter is not None:
            from airclassifier.geometry.assembly.classification import ClassificationSystemParams
            classification_params = ClassificationSystemParams()
            throat_m = args.throat_diameter / 1000.0  # mm to m
            classification_params.venturi_throat_ratio = throat_m / classification_params.venturi_inlet_diameter
            print(f"  [Override] Venturi throat: {args.throat_diameter:.1f} mm "
                  f"(ratio={classification_params.venturi_throat_ratio:.3f})")
        complete_params = CompleteSystemParams(
            air_flow_m3_h=Q_m3s * 3600.0,
            throughput_kg_h=500.0,
            include_feed_system=True,
            include_air_system=True,
            include_exhaust=False,
            include_ductwork=True,
            classification_params=classification_params,
        )
        complete_assembly = CompleteClassifierAssembly(complete_params)
        print("\n1. Air system -> Venturi air inlet (airclass)...")
        air_result = compute_air_to_venturi_flow(
            complete_assembly, Q_m3s,
            rho=fluid.density, mu=fluid.dynamic_viscosity,
        )
        print_ductwork_flow_summary(air_result)
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
        classification_assembly = complete_assembly.get_subsystem("classification")
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
        assembly = classification_assembly
        config = ClassificationFlowConfig.from_air_and_feed_results(
            air_result, feed_result, classification_assembly,
            solids_mass_flow_kg_s=solids_mass_flow_kg_s,
            num_particles_capacity=args.particles,
            simulation_time_s=args.time,
            dt=args.dt,
            device=args.device,
            turbulent_intensity=args.turbulence,
            fluid_config=fluid,
            material=material,
            bypass_ratio=args.bypass_ratio,
            continuous_feeding=not args.batch_feed,
            max_loading_ratio=args.max_loading,
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
                         args.throat_diameter is not None)
        if has_overrides:
            custom_params = ClassificationSystemParams()
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
        print(f"    - Venturi eductor (particle entrainment)")
        print(f"    - Zigzag classifier (primary separation)")
        print(f"    - Multi-cyclone system (staged separation)")
        print(f"    - Bag filter (final collection)")
        
        # =========================================================================
        # CALCULATE OPTIMAL AIR FLOW FOR TARGET D50
        # =========================================================================
        if args.target_d50 is not None:
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
                turbulent_intensity=args.turbulence,
                material=material,
                fluid_config=fluid,
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
                turbulent_intensity=args.turbulence,
                material=material,
                fluid_config=FluidConfig.air_at_stp(),
            )
            print(f"  Using FluidConfig + material: {material.name}")
        else:
            config = ClassificationFlowConfig(
                dt=args.dt,
                air_flow_rate_m3s=args.air_flow,
                bypass_ratio=args.bypass_ratio,
                num_particles=args.particles,
                device=args.device,
                turbulent_intensity=args.turbulence,
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
    print("\nInitializing particles at venturi inlet...")
    
    if args.material and args.material in ("yellow_pea", "faba_bean", "oat"):
        # Whole flour population (protein + starch + fiber) via reusable module
        simulator.initialize_whole_flour_population(
            source=args.material,
            num_particles=args.particles,
            initial_velocity=(0.0, 0.5, 0.0),
        )
    elif args.material and args.material in ("protein", "starch", "fiber"):
        # Single fraction via material
        material = ParticleMaterial.create_food_powder("yellow_pea", args.material)
        simulator.initialize_particles_from_material(
            material=material,
            num_particles=args.particles,
            initial_velocity=(0.0, 0.5, 0.0),
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
        
        # ============================================
        # BUILD MESH FROM ACTUAL ASSEMBLY
        # ============================================
        print("  Building assembly mesh...")
        assembly.build_mesh()
        
        # Get component positions
        comp_positions = assembly.get_component_positions()
        
        # ============================================
        # VENTURI EDUCTOR (actual geometry)
        # ============================================
        print("  Adding venturi eductor...")
        v_vent, i_vent, _ = assembly.venturi.generate_mesh()
        v_vent = v_vent + np.array(comp_positions['venturi'])
        faces_vent = np.hstack([[3] + list(face) for face in i_vent.reshape(-1, 3)])
        venturi_mesh = pv.PolyData(v_vent, faces_vent)
        plotter.add_mesh(venturi_mesh, color='#3498DB', opacity=0.5, label='Venturi')
        
        # ============================================
        # ZIGZAG CLASSIFIER (actual geometry)
        # ============================================
        print("  Adding zigzag classifier...")
        v_zz, i_zz, _ = assembly.zigzag.generate_mesh()
        v_zz = v_zz + np.array(comp_positions['zigzag'])
        faces_zz = np.hstack([[3] + list(face) for face in i_zz.reshape(-1, 3)])
        zigzag_mesh = pv.PolyData(v_zz, faces_zz)
        plotter.add_mesh(zigzag_mesh, color='#2ECC71', opacity=0.5, label='Zigzag')
        
        # ============================================
        # MULTI-CYCLONE SYSTEM (actual geometry)
        # ============================================
        print("  Adding multi-cyclone system...")
        v_mc, i_mc, _ = assembly.multi_cyclone.generate_mesh()
        v_mc = v_mc + np.array(comp_positions['multi_cyclone'])
        faces_mc = np.hstack([[3] + list(face) for face in i_mc.reshape(-1, 3)])
        cyclone_mesh = pv.PolyData(v_mc, faces_mc)
        plotter.add_mesh(cyclone_mesh, color='#E74C3C', opacity=0.5, label='Cyclones')
        
        # ============================================
        # BAG FILTER (actual geometry)
        # ============================================
        print("  Adding bag filter...")
        v_bf, i_bf, _ = assembly.bag_filter.generate_mesh()
        v_bf = v_bf + np.array(comp_positions['bag_filter'])
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
            faces_duct = np.hstack([[3] + list(face) for face in i_duct.reshape(-1, 3)])
            duct_mesh = pv.PolyData(v_duct, faces_duct)
            label = 'Ductwork' if idx == 0 else None
            plotter.add_mesh(duct_mesh, color='#7F8C8D', opacity=0.4, label=label)
        
        # ============================================
        # LABELS FOR KEY PORTS
        # ============================================
        print("  Adding port labels...")
        
        # Get port positions from assembly
        try:
            coarse_pos = assembly.get_port_world_position('zigzag', 'coarse_outlet')
            plotter.add_point_labels([coarse_pos - np.array([0, 0.1, 0])], 
                                    ["COARSE\n(Starch)"], font_size=12, 
                                    text_color='#8B4513', point_size=0)
        except (KeyError, AttributeError):
            pass
        
        try:
            fines_pos = assembly.get_port_world_position('zigzag', 'fines_outlet')
            plotter.add_point_labels([fines_pos + np.array([0, 0.1, 0])], 
                                    ["FINES"], font_size=10,
                                    text_color='#2ECC71', point_size=0)
        except (KeyError, AttributeError):
            pass
        
        # Cyclone dust outlets
        try:
            for dust_name in ['primary_dust', 'secondary_dust', 'tertiary_dust']:
                dust_pos = assembly.get_port_world_position('multi_cyclone', dust_name)
                label = "PROTEIN" if 'tertiary' in dust_name else dust_name.split('_')[0].title()
                color = '#9B59B6' if 'tertiary' in dust_name else '#E74C3C'
                plotter.add_point_labels([dust_pos - np.array([0, 0.1, 0])], 
                                        [label], font_size=10,
                                        text_color=color, point_size=0)
        except (KeyError, AttributeError):
            pass
        
        try:
            dust_pos = assembly.get_port_world_position('bag_filter', 'dust_outlet')
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
    
    # Run simulation
    print("\n" + "-" * 70)
    print("RUNNING SIMULATION")
    print("-" * 70)
    print(f"  Time: {args.time:.1f} s")
    print(f"  dt:   {args.dt*1000:.2f} ms")
    print(f"  Steps: {int(args.time / args.dt):,}")
    if args.bypass_ratio > 0:
        Q_class = args.air_flow * (1.0 - args.bypass_ratio)
        print(f"  Air flow: {args.air_flow * 3600:.0f} m³/h total, "
              f"{Q_class * 3600:.1f} m³/h classification, "
              f"{args.bypass_ratio*100:.1f}% bypass")
    else:
        print(f"  Air flow: {args.air_flow * 3600:.0f} m³/h")
    print(f"  Zigzag d50: {simulator.zigzag_d50 * 1e6:.1f} µm")
    if config.continuous_feeding:
        print(f"  Feeding: continuous at {config.particle_feed_rate:.0f} particles/s")
        m_per_particle = config.particle_density * (np.pi / 6.0) * config.visual_particle_diameter**3
        print(f"  Feed mass flow: {config.particle_feed_rate * m_per_particle * 3600:.1f} kg/h")
        print(f"  Max loading ratio: {config.max_loading_ratio:.1f}")
    else:
        print(f"  Feeding: batch (all particles active at t=0)")
    print("-" * 70)
    
    total_steps = int(args.time / args.dt)
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
    while step < total_steps:
        # Run multiple simulation steps per visual frame
        frame_steps = min(steps_per_frame, total_steps - step)
        for _ in range(frame_steps):
            simulator.step()
            step += 1
        
        # Get current state
        zone_counts = simulator.get_zone_counts()
        sep_counts = simulator.get_separation_counts()
        
        # Console output at intervals
        if step - last_print_step >= print_interval or step >= total_steps:
            last_print_step = step
            progress = 100.0 * step / total_steps
            
            active = sep_counts['active']
            coarse = sep_counts['coarse']
            cy1 = sep_counts['cyclone_1']
            cy2 = sep_counts['cyclone_2']
            cy3 = sep_counts['cyclone_3_protein']
            bag = sep_counts['bagfilter']
            
            status = f"  [{progress:5.1f}%] t={simulator.state.time:5.2f}s"
            # Show feed progress if continuous feeding
            if config.continuous_feeding:
                fed = simulator.state.particles_fed
                total_to_feed = simulator.state.total_particles_to_feed
                status += f" | Fed:{fed:5d}/{total_to_feed}"
            status += f" | Active:{active:5d} Coarse:{coarse:5d}"
            status += f" Cy1:{cy1:5d} Cy2:{cy2:5d} Cy3:{cy3:5d} Bag:{bag:5d}"
            # Show zone breakdown for active particles
            zz = zone_counts.get('zigzag', 0)
            fp = zone_counts.get('fines_path', 0)
            vent = zone_counts.get('venturi', 0)
            duct_vz = zone_counts.get('duct_v_z', 0)
            cy1_z = zone_counts.get('cyclone_1', 0)
            cy2_z = zone_counts.get('cyclone_2', 0)
            cy3_z = zone_counts.get('cyclone_3', 0)
            status += f"  [zz:{zz:5d} fp:{fp:4d} v:{vent:3d} d:{duct_vz:3d} c1:{cy1_z:4d} c2:{cy2_z:4d} c3:{cy3_z:4d}]"
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
            # UPDATE INFO TEXT
            # ============================================
            info_text = (
                f"CLASSIFICATION FLOW\n"
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
            total_collected = (sep_counts['coarse'] + sep_counts['cyclone_1'] + 
                             sep_counts['cyclone_2'] + sep_counts['cyclone_3_protein'] +
                             sep_counts['bagfilter'])
            
            if total_collected > 0:
                pct_coarse = 100 * sep_counts['coarse'] / total_collected
                pct_cy1 = 100 * sep_counts['cyclone_1'] / total_collected
                pct_cy2 = 100 * sep_counts['cyclone_2'] / total_collected
                pct_cy3 = 100 * sep_counts['cyclone_3_protein'] / total_collected
                pct_bag = 100 * sep_counts['bagfilter'] / total_collected
            else:
                pct_coarse = pct_cy1 = pct_cy2 = pct_cy3 = pct_bag = 0
            
            sep_text = (
                f"SEPARATION\n"
                f"----------\n"
                f"Coarse: {sep_counts['coarse']:4d} ({pct_coarse:4.1f}%)\n"
                f"Cy1:    {sep_counts['cyclone_1']:4d} ({pct_cy1:4.1f}%)\n"
                f"Cy2:    {sep_counts['cyclone_2']:4d} ({pct_cy2:4.1f}%)\n"
                f"Cy3:    {sep_counts['cyclone_3_protein']:4d} ({pct_cy3:4.1f}%)\n"
                f"Bag:    {sep_counts['bagfilter']:4d} ({pct_bag:4.1f}%)\n"
                f"----------\n"
                f"Protein: Cy3+Bag\n"
                f"Starch:  Coarse"
            )
            plotter.add_text(sep_text, position='upper_right', font_size=10,
                            color='black', name='sep_info')
            
            plotter.update()
            time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print("SIMULATION COMPLETE")
    print("-" * 70)
    print(f"  Wall time: {elapsed:.1f} s")
    print(f"  Sim time:  {simulator.state.time:.2f} s")
    print(f"  Steps:     {simulator.state.step:,}")
    print(f"  Rate:      {simulator.state.step / elapsed:.0f} steps/s")
    if config.continuous_feeding:
        fed = simulator.state.particles_fed
        total_to_feed = simulator.state.total_particles_to_feed
        print(f"  Feeding:   {fed}/{total_to_feed} particles fed ({100*fed/max(1,total_to_feed):.1f}%)")
        print(f"  Feed rate: {config.particle_feed_rate:.0f} particles/s")
    
    # Print final separation summary
    simulator.print_separation_summary()
    
    # =========================================================================
    # FINAL DIAGNOSTICS
    # =========================================================================
    if args.diagnostics:
        print("\n" + "=" * 70)
        print("FINAL DIAGNOSTICS - Zone Distribution")
        print("=" * 70)
        zone_counts = simulator.get_zone_counts()
        print(f"\n  Active Particle Distribution by Zone:")
        for zone_name, count in zone_counts.items():
            if count > 0:
                print(f"    {zone_name:20s}: {count:5d}")
        
        # Print detailed separation analysis
        sep = simulator.get_separation_counts()
        total = sum(sep.values()) - sep['active']  # Exclude still-active particles
        
        if total > 0:
            print(f"\n  Separation Analysis:")
            print(f"    Total collected:     {total:5d} particles")
            print(f"    Still in system:     {sep['active']:5d} particles")
            
            print(f"\n  Collection by Outlet:")
            outlets = [
                ('Coarse (Starch)', 'coarse', '#8B4513'),
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
            starch_outlets = sep['coarse']
            print(f"\n  Protein Recovery Estimate:")
            print(f"    Protein-rich fractions (Cy3 + Bag): {protein_outlets:5d} ({100*protein_outlets/max(1,total):.1f}%)")
            print(f"    Starch-rich fractions (Coarse):     {starch_outlets:5d} ({100*starch_outlets/max(1,total):.1f}%)")

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

    print("=" * 70)
    
    if plotter is not None:
        print("\nVisualization window open. Close to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
