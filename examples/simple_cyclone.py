"""
Simple Cyclone Air Classifier Simulation Example

Demonstrates basic usage of the airclassifier package to simulate
particle separation in a standard cyclone geometry.

Features demonstrated:
- Cyclone geometry setup
- Particle material definition with size distribution
- Flow field configuration
- GPU-accelerated simulation with wall collisions
- Grade efficiency curve calculation
- VTK export for ParaView visualization
- Summary plotting
"""

import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from our package
from airclassifier import (
    CycloneAssembly,
    CycloneGeometryParams,
    CycloneFlowField,
    CycloneFlowParams,
    ParticleMaterial,
    CycloneSimulator,
    SimulationConfig,
)
from airclassifier.kinetics import (
    compute_grade_efficiency,
    theoretical_d50_lapple,
    plot_grade_efficiency,
)
from airclassifier.io import export_simulation_results
from airclassifier.visualization import plot_simulation_summary


def main():
    """Run a simple cyclone simulation."""

    # Initialize Warp
    wp.init()
    print(f"Warp initialized on: {wp.get_device()}")

    # ==========================================================================
    # 1. DEFINE CYCLONE GEOMETRY
    # ==========================================================================
    print("\n1. Creating cyclone geometry...")

    # Standard cyclone with 300mm diameter
    geometry = CycloneGeometryParams(
        cylinder_diameter=0.3,      # 300 mm
        cylinder_height=0.45,       # 450 mm (1.5 * D)
        cone_height=0.75,           # 750 mm (2.5 * D)
        cone_tip_diameter=0.1125,   # 112.5 mm (0.375 * D)
        inlet_width=0.075,          # 75 mm (0.25 * D)
        inlet_height=0.15,          # 150 mm (0.5 * D)
        vortex_finder_diameter=0.15,  # 150 mm (0.5 * D)
        vortex_finder_length=0.15,    # 150 mm (0.5 * D)
    )

    # Create cyclone assembly
    cyclone = CycloneAssembly(geometry, device="cuda")
    cyclone.print_summary()

    # ==========================================================================
    # 2. DEFINE PARTICLE MATERIAL
    # ==========================================================================
    print("\n2. Defining particle material...")

    # Quartz particles with Rosin-Rammler size distribution
    material = ParticleMaterial.create(
        material_name="quartz",
        distribution_type="rosin_rammler",
        d50=50.0e-6,        # 50 micron median
        spread=2.0,         # Spread parameter
        d_min=5.0e-6,       # 5 micron minimum
        d_max=150.0e-6,     # 150 micron maximum
    )

    print(f"  Material: {material.name}")
    print(f"  Density: {material.density} kg/m³")
    print(f"  Sphericity: {material.sphericity}")
    print(f"  d50: {material.size_distribution.d50 * 1e6:.1f} μm")

    # Sample and visualize size distribution
    sample_diameters = material.sample_diameters(10000) * 1e6  # Convert to microns

    # ==========================================================================
    # 3. VISUALIZE FLOW FIELD
    # ==========================================================================
    print("\n3. Creating flow field...")

    flow_params = CycloneFlowParams(
        cylinder_radius=geometry.cylinder_diameter / 2.0,
        vortex_finder_radius=geometry.vortex_finder_diameter / 2.0,
        cylinder_height=geometry.cylinder_height,
        cone_height=geometry.cone_height,
        cone_bottom_radius=geometry.cone_tip_diameter / 2.0,
        inlet_velocity=15.0,  # 15 m/s inlet velocity
        inlet_width=geometry.inlet_width,
        inlet_height=geometry.inlet_height,
    )

    flow_field = CycloneFlowField(flow_params)

    print(f"  Inlet velocity: {flow_params.inlet_velocity} m/s")
    print(f"  Max tangential velocity: {flow_field.params.max_tangential_velocity:.1f} m/s")
    print(f"  Volumetric flow rate: {flow_params.volumetric_flow_rate * 3600:.1f} m³/h")

    # ==========================================================================
    # 4. RUN SIMULATION
    # ==========================================================================
    print("\n4. Running simulation...")

    config = SimulationConfig(
        dt=1.0e-5,              # 10 microsecond time step
        duration=0.5,           # 0.5 second simulation
        num_particles=5000,     # 5000 particles
        injection_duration=0.1, # Inject over 0.1 seconds
        device="cuda",
        # Wall collision settings
        include_wall_collisions=True,
        wall_restitution=0.7,   # Coefficient of restitution for wall bounces
        wall_friction=0.3,      # Friction coefficient
    )

    print(f"  Time step: {config.dt * 1e6:.1f} μs")
    print(f"  Duration: {config.duration} s")
    print(f"  Total steps: {config.num_steps}")
    print(f"  Particles: {config.num_particles}")

    # Create and run simulator
    simulator = CycloneSimulator(geometry, material, config)

    # Run with progress updates
    def progress(step, total):
        if step % 5000 == 0:
            print(f"    Step {step}/{total} ({100*step/total:.1f}%)")

    simulator.run(progress_callback=progress)

    # ==========================================================================
    # 5. ANALYZE RESULTS
    # ==========================================================================
    print("\n5. Results:")

    results = simulator.get_results()

    print(f"  Particles injected: {results['particles_injected']}")
    print(f"  Particles collected (underflow): {results['particles_collected']}")
    print(f"  Particles escaped (overflow): {results['particles_escaped']}")
    print(f"  Particles still active: {results['particles_active']}")
    print(f"  Collection efficiency: {results['collection_efficiency']:.1%}")

    # ==========================================================================
    # 6. GRADE EFFICIENCY CURVE
    # ==========================================================================
    print("\n6. Computing grade efficiency curve...")

    # Calculate theoretical d50 cut size using Lapple model
    d50_theoretical = theoretical_d50_lapple(
        cyclone_diameter=geometry.cylinder_diameter,
        inlet_width=geometry.inlet_width,
        inlet_velocity=flow_params.inlet_velocity,
        fluid_viscosity=1.81e-5,  # Air at 20°C
        particle_density=material.density,
        fluid_density=1.2,
        num_turns=5.0,
    )
    print(f"  Theoretical d50 (Lapple): {d50_theoretical * 1e6:.1f} μm")

    # Compute grade efficiency from simulation results
    grade_curve = compute_grade_efficiency(
        diameters=results['diameters'],
        is_active=results['is_active'],
        num_bins=20,
    )

    if grade_curve.d50 is not None:
        print(f"  Simulated d50: {grade_curve.d50 * 1e6:.1f} μm")
    else:
        print(f"  Simulated d50: N/A (insufficient data - no particles collected/escaped)")
    print(f"  Overall efficiency: {grade_curve.overall_efficiency:.1%}")

    # ==========================================================================
    # 7. VTK EXPORT FOR PARAVIEW
    # ==========================================================================
    print("\n7. Exporting VTK files for ParaView...")

    output_dir = Path("vtk_output")
    output_dir.mkdir(exist_ok=True)

    export_simulation_results(
        output_dir=str(output_dir),
        positions=results['positions'],
        diameters=results['diameters'],
        velocities=results['velocities'],
        is_active=results['is_active'],
        prefix="cyclone_sim",
    )

    # ==========================================================================
    # 8. PLOTTING
    # ==========================================================================
    print("\n8. Creating plots...")

    try:
        # Create summary figure
        fig = plot_simulation_summary(
            results=results,
            flow_field=flow_field,
            cyclone_radius=geometry.cylinder_diameter / 2,
            save_path='cyclone_summary.png',
        )

        # Create grade efficiency plot
        d50_str = f"{grade_curve.d50*1e6:.1f} μm" if grade_curve.d50 is not None else "N/A"
        fig_grade = plot_grade_efficiency(
            grade_curve,
            title=f"Grade Efficiency (d50 = {d50_str})",
        )
        if fig_grade is not None:
            fig_grade.savefig('grade_efficiency.png', dpi=150, bbox_inches='tight')
            print("  Saved grade efficiency plot: grade_efficiency.png")

        # Additional custom plot: Size distribution comparison
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Feed size distribution
        ax1 = axes[0]
        ax1.hist(sample_diameters, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Particle Diameter (μm)')
        ax1.set_ylabel('Count')
        ax1.set_title('Feed Size Distribution')
        ax1.axvline(material.size_distribution.d50 * 1e6, color='r',
                   linestyle='--', label=f'd50 = {material.size_distribution.d50*1e6:.1f} μm')
        ax1.legend()

        # Plot 2: Collected vs Escaped size distributions
        ax2 = axes[1]
        is_active = results['is_active']
        diameters_um = results['diameters'] * 1e6

        collected_mask = is_active == -1
        escaped_mask = is_active == -2

        if np.any(collected_mask):
            ax2.hist(diameters_um[collected_mask], bins=30, alpha=0.7,
                    label=f'Collected ({np.sum(collected_mask)})', color='green')
        if np.any(escaped_mask):
            ax2.hist(diameters_um[escaped_mask], bins=30, alpha=0.7,
                    label=f'Escaped ({np.sum(escaped_mask)})', color='red')

        ax2.set_xlabel('Particle Diameter (μm)')
        ax2.set_ylabel('Count')
        ax2.set_title('Separated Size Distributions')
        ax2.legend()

        # Plot 3: Tangential velocity profile
        ax3 = axes[2]
        r_range = np.linspace(0.001, geometry.cylinder_diameter/2 * 0.95, 100)
        v_tan = [flow_field._tangential_velocity(r, geometry.cylinder_diameter/2) for r in r_range]
        ax3.plot(r_range * 1000, v_tan, 'b-', linewidth=2)
        ax3.set_xlabel('Radial Position (mm)')
        ax3.set_ylabel('Tangential Velocity (m/s)')
        ax3.set_title('Tangential Velocity Profile')
        ax3.axvline(flow_params.vortex_finder_radius * 1000, color='r',
                   linestyle='--', label='Vortex finder')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('cyclone_analysis.png', dpi=150)
        print("  Saved analysis plots: cyclone_analysis.png")

        plt.show()

    except ImportError as e:
        print(f"\n  (Plotting not available: {e})")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Collection efficiency: {results['collection_efficiency']:.1%}")
    if grade_curve.d50 is not None:
        print(f"  Simulated d50: {grade_curve.d50 * 1e6:.1f} μm")
    else:
        print(f"  Simulated d50: N/A (insufficient data)")
    print(f"  Theoretical d50: {d50_theoretical * 1e6:.1f} μm")
    print(f"\nOutput files:")
    print(f"  - cyclone_summary.png")
    print(f"  - cyclone_analysis.png")
    print(f"  - grade_efficiency.png")
    print(f"  - vtk_output/ (for ParaView)")


if __name__ == "__main__":
    main()
