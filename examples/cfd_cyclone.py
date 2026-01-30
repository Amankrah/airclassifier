"""
CFD-DEM Coupled Cyclone Air Classifier Simulation Example

Demonstrates the full physics CFD-DEM coupling approach where:
- Fluid flow is solved using Navier-Stokes equations (NavierStokesSolver)
- Turbulence is modeled with k-epsilon RANS (KEpsilonModel)
- Particles are tracked with Lagrangian DEM
- Two-way coupling exchanges momentum between fluid and particles

This is more computationally expensive than the analytical flow approach
but provides higher fidelity results, especially for:
- Dense particle flows
- Complex geometries
- Turbulence-dominated separation

Features demonstrated:
- CFD-DEM coupling setup
- Turbulence model configuration
- One-way vs two-way coupling
- Comparison with analytical flow results
"""

import warp as wp
import numpy as np
import time
from pathlib import Path

# Import from our package
from airclassifier import (
    CycloneGeometryParams,
    ParticleMaterial,
    SimulationConfig,
    FlowMode,
    create_simulator,
)
from airclassifier.simulation import (
    CFDDEMCoupler,
    CFDConfig,
    DEMConfig,
    CycloneCFDParams,
    TurbulenceModelType,
    CouplingMode,
)


def run_cfd_simulation():
    """Run CFD-DEM coupled simulation."""
    print("=" * 60)
    print("CFD-DEM COUPLED CYCLONE SIMULATION")
    print("=" * 60)

    # Initialize Warp
    wp.init()
    print(f"\nWarp initialized on: {wp.get_device()}")

    # ==========================================================================
    # 1. DEFINE CYCLONE GEOMETRY
    # ==========================================================================
    print("\n1. Creating cyclone geometry...")

    geometry = CycloneGeometryParams(
        cylinder_diameter=0.3,       # 300 mm
        cylinder_height=0.45,        # 450 mm
        cone_height=0.75,            # 750 mm
        cone_tip_diameter=0.1125,    # 112.5 mm
        inlet_width=0.075,           # 75 mm
        inlet_height=0.15,           # 150 mm
        vortex_finder_diameter=0.15, # 150 mm
        vortex_finder_length=0.15,   # 150 mm
    )

    print(f"  Cyclone diameter: {geometry.cylinder_diameter * 1000:.0f} mm")
    print(f"  Total height: {(geometry.cylinder_height + geometry.cone_height) * 1000:.0f} mm")

    # ==========================================================================
    # 2. DEFINE PARTICLE MATERIAL
    # ==========================================================================
    print("\n2. Defining particle material...")

    material = ParticleMaterial.create(
        material_name="quartz",
        distribution_type="rosin_rammler",
        d50=50.0e-6,        # 50 micron median
        spread=2.0,
        d_min=5.0e-6,       # 5 micron minimum
        d_max=150.0e-6,     # 150 micron maximum
    )

    print(f"  Material: {material.name}")
    print(f"  Density: {material.density} kg/m3")
    print(f"  d50: {material.size_distribution.d50 * 1e6:.1f} um")

    # ==========================================================================
    # 3. OPTION A: USE FACTORY FUNCTION (RECOMMENDED)
    # ==========================================================================
    print("\n3. Creating CFD-DEM coupled simulator (via factory)...")

    # Use the factory function with CFD flow mode
    config = SimulationConfig(
        flow_mode=FlowMode.CFD,      # Full CFD-DEM coupling
        dt=1.0e-4,                   # Larger dt for CFD (CFD substeps handle stability)
        duration=0.2,                # Short demo run
        num_particles=2000,          # Fewer particles for demo
        injection_duration=0.05,
        device="cuda",
        # CFD-specific parameters
        cfd_resolution=(32, 64, 32), # Coarser grid for demo
        cfd_substeps=5,
        turbulence_model="k_epsilon",
        turbulence_intensity=0.05,
        coupling_mode="one_way",     # Start with one-way coupling
        # Wall collision parameters
        wall_restitution=0.7,
        wall_friction=0.3,
    )

    print(f"  Flow mode: {config.flow_mode.value}")
    print(f"  CFD grid: {config.cfd_resolution}")
    print(f"  Turbulence: {config.turbulence_model}")
    print(f"  Coupling: {config.coupling_mode}")

    # Create simulator using factory
    simulator = create_simulator(geometry, material, config)

    print(f"  Simulator type: {type(simulator).__name__}")

    # ==========================================================================
    # 4. RUN SIMULATION
    # ==========================================================================
    print("\n4. Running CFD-DEM simulation...")
    print("   (This may take a while due to CFD solver)")

    start_time = time.time()
    step_count = 0

    def progress(current_time, total_time):
        nonlocal step_count
        step_count += 1
        if step_count % 10 == 0:
            pct = 100 * current_time / total_time
            print(f"    t = {current_time:.4f}s ({pct:.1f}%)")

    # Run the simulation
    simulator.run(duration=config.duration, progress_callback=progress)

    elapsed = time.time() - start_time
    print(f"\n  Simulation completed in {elapsed:.1f}s")

    # ==========================================================================
    # 5. ANALYZE RESULTS
    # ==========================================================================
    print("\n5. Results:")

    results = simulator.get_results()

    print(f"  Time simulated: {results['time']:.3f}s")
    print(f"  CFD steps: {results.get('cfd_steps', 'N/A')}")
    print(f"  DEM steps: {results.get('dem_steps', 'N/A')}")
    print(f"  Particles injected: {results['particles_injected']}")
    print(f"  Particles collected: {results['particles_collected']}")
    print(f"  Particles escaped: {results['particles_escaped']}")
    print(f"  Particles active: {results['particles_active']}")
    print(f"  Collection efficiency: {results['collection_efficiency']:.1%}")

    return results


def run_comparison():
    """Compare analytical vs CFD flow results."""
    print("\n" + "=" * 60)
    print("COMPARISON: ANALYTICAL vs CFD FLOW")
    print("=" * 60)

    # Common parameters
    geometry = CycloneGeometryParams(
        cylinder_diameter=0.3,
        cylinder_height=0.45,
        cone_height=0.75,
        cone_tip_diameter=0.1125,
        inlet_width=0.075,
        inlet_height=0.15,
        vortex_finder_diameter=0.15,
        vortex_finder_length=0.15,
    )

    material = ParticleMaterial.create(
        material_name="quartz",
        distribution_type="rosin_rammler",
        d50=50.0e-6,
        spread=2.0,
        d_min=5.0e-6,
        d_max=150.0e-6,
    )

    # Analytical flow simulation
    print("\n1. Running ANALYTICAL flow simulation...")
    config_analytical = SimulationConfig(
        flow_mode=FlowMode.ANALYTICAL,
        dt=1.0e-5,
        duration=0.2,
        num_particles=2000,
        injection_duration=0.05,
        device="cuda",
    )

    start = time.time()
    sim_analytical = create_simulator(geometry, material, config_analytical)
    sim_analytical.run()
    time_analytical = time.time() - start
    results_analytical = sim_analytical.get_results()

    print(f"  Time: {time_analytical:.1f}s")
    print(f"  Efficiency: {results_analytical['collection_efficiency']:.1%}")

    # CFD flow simulation
    print("\n2. Running CFD flow simulation...")
    config_cfd = SimulationConfig(
        flow_mode=FlowMode.CFD,
        dt=1.0e-4,
        duration=0.2,
        num_particles=2000,
        injection_duration=0.05,
        device="cuda",
        cfd_resolution=(32, 64, 32),
        cfd_substeps=5,
        turbulence_model="k_epsilon",
    )

    start = time.time()
    sim_cfd = create_simulator(geometry, material, config_cfd)
    sim_cfd.run(duration=config_cfd.duration)
    time_cfd = time.time() - start
    results_cfd = sim_cfd.get_results()

    print(f"  Time: {time_cfd:.1f}s")
    print(f"  Efficiency: {results_cfd['collection_efficiency']:.1%}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"                      Analytical      CFD")
    print(f"  Computation time:   {time_analytical:>8.1f}s    {time_cfd:>8.1f}s")
    print(f"  Collection eff:     {results_analytical['collection_efficiency']:>8.1%}    {results_cfd['collection_efficiency']:>8.1%}")
    print(f"  Collected:          {results_analytical['particles_collected']:>8d}    {results_cfd['particles_collected']:>8d}")
    print(f"  Escaped:            {results_analytical['particles_escaped']:>8d}    {results_cfd['particles_escaped']:>8d}")
    print(f"  Active:             {results_analytical['particles_active']:>8d}    {results_cfd['particles_active']:>8d}")
    print(f"\nSpeedup factor: {time_cfd/time_analytical:.1f}x slower for CFD")


def direct_cfd_dem_usage():
    """Demonstrate direct CFDDEMCoupler usage for advanced control."""
    print("\n" + "=" * 60)
    print("DIRECT CFD-DEM COUPLER USAGE")
    print("=" * 60)

    # This shows how to use CFDDEMCoupler directly for more control

    # Cyclone geometry parameters
    cyclone_params = CycloneCFDParams(
        cylinder_diameter=0.3,
        cylinder_height=0.45,
        cone_height=0.75,
        cone_tip_diameter=0.1125,
        inlet_width=0.075,
        inlet_height=0.15,
        inlet_velocity=15.0,
        vortex_finder_diameter=0.15,
        vortex_finder_length=0.15,
    )

    # CFD solver configuration
    cfd_config = CFDConfig(
        domain_size=(0.36, 1.3, 0.36),  # Slightly larger than cyclone
        resolution=(48, 96, 48),         # Grid resolution
        density=1.2,
        kinematic_viscosity=1.5e-5,
        dt=1.0e-4,
        max_pressure_iterations=50,
        turbulence_model=TurbulenceModelType.K_EPSILON,
        turbulence_intensity=0.05,
        coupling_mode=CouplingMode.TWO_WAY,  # Full two-way coupling
        cfd_substeps=5,
    )

    # DEM (particle) configuration
    dem_config = DEMConfig(
        dt=1.0e-5,
        num_particles=1000,
        injection_duration=0.05,
        wall_restitution=0.7,
        wall_friction=0.3,
    )

    # Material
    material = ParticleMaterial.create(
        material_name="quartz",
        distribution_type="rosin_rammler",
        d50=50.0e-6,
        spread=2.0,
    )

    print("\nConfiguration:")
    print(f"  CFD grid: {cfd_config.resolution}")
    print(f"  Turbulence: {cfd_config.turbulence_model.value}")
    print(f"  Coupling: {cfd_config.coupling_mode.value}")
    print(f"  CFD substeps: {cfd_config.cfd_substeps}")
    print(f"  Particles: {dem_config.num_particles}")

    # Create coupler directly
    print("\nCreating CFD-DEM coupler...")
    coupler = CFDDEMCoupler(
        cyclone_params=cyclone_params,
        cfd_config=cfd_config,
        dem_config=dem_config,
        material=material,
        device="cuda",
    )

    # Manual stepping for fine control
    print("\nRunning with manual stepping...")
    num_steps = 100
    for i in range(num_steps):
        coupler.step()
        if (i + 1) % 20 == 0:
            results = coupler.get_results()
            print(f"  Step {i+1}: t={results['time']:.4f}s, "
                  f"collected={results['particles_collected']}, "
                  f"escaped={results['particles_escaped']}")

    print("\nFinal results:")
    results = coupler.get_results()
    print(f"  Efficiency: {results['collection_efficiency']:.1%}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "--direct":
        direct_cfd_dem_usage()
    else:
        run_cfd_simulation()

    print("\n" + "=" * 60)
    print("CFD-DEM SIMULATION COMPLETE")
    print("=" * 60)
    print("\nUsage options:")
    print("  python cfd_cyclone.py           # Run CFD-DEM simulation")
    print("  python cfd_cyclone.py --compare # Compare analytical vs CFD")
    print("  python cfd_cyclone.py --direct  # Direct CFDDEMCoupler usage")


if __name__ == "__main__":
    main()
