"""
Food Protein Separation Simulation using Air Classification.

This example demonstrates GPU-accelerated particle simulation for separating
protein from plant-based food powders (yellow peas, faba beans, oats) using
NVIDIA Warp.

The air classifier uses a combination of:
- Rankine vortex flow (cyclone-like swirling)
- Drag forces (particle-fluid interaction)
- Gravitational separation
- Centrifugal separation

Protein particles (smaller, lighter) are carried upward with the air flow,
while starch particles (larger, denser) are thrown outward and settle down.

Usage:
    python food_protein_separation.py [--source yellow_pea|faba_bean|oat]
                                      [--particles 10000]
                                      [--time 0.5]
                                      [--visualize]
                                      [--save-animation path/to/output.mp4]
"""

import argparse
import time
import numpy as np
import warp as wp

# Import from airclassifier
from airclassifier.particles import (
    WarpParticleSystem,
    ParticleSystemConfig,
    ParticleMaterial,
    ParticleType,
    create_particle_system,
)
from airclassifier.utils.constants import (
    FoodPowderComposition,
    FoodPowderSizeRanges,
    AirProperties,
)

# Optional visualization imports
try:
    from airclassifier.visualization import (
        ParticleVisualizer,
        ParticleVisualizationConfig,
        ColorMode,
        visualize_particles,
        animate_simulation,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Visualization not available. Install pyvista for visualization.")


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("   Food Protein Air Classification Simulation")
    print("   Powered by NVIDIA Warp GPU Computing")
    print("=" * 70)
    print()


def print_material_info(source: str):
    """Print information about the material being processed."""
    source_info = {
        "yellow_pea": {
            "name": "Yellow Pea (Pisum sativum)",
            "protein": FoodPowderComposition.YELLOW_PEA_PROTEIN_CONTENT * 100,
            "starch": FoodPowderComposition.YELLOW_PEA_STARCH_CONTENT * 100,
            "fiber": FoodPowderComposition.YELLOW_PEA_FIBER_CONTENT * 100,
        },
        "faba_bean": {
            "name": "Faba Bean (Vicia faba)",
            "protein": FoodPowderComposition.FABA_BEAN_PROTEIN_CONTENT * 100,
            "starch": FoodPowderComposition.FABA_BEAN_STARCH_CONTENT * 100,
            "fiber": FoodPowderComposition.FABA_BEAN_FIBER_CONTENT * 100,
        },
        "oat": {
            "name": "Oat (Avena sativa)",
            "protein": FoodPowderComposition.OAT_PROTEIN_CONTENT * 100,
            "starch": FoodPowderComposition.OAT_STARCH_CONTENT * 100,
            "fiber": FoodPowderComposition.OAT_FIBER_CONTENT * 100,
        },
    }
    
    info = source_info[source]
    print(f"Material: {info['name']}")
    print(f"  Composition (typical):")
    print(f"    - Protein: {info['protein']:.1f}%")
    print(f"    - Starch:  {info['starch']:.1f}%")
    print(f"    - Fiber:   {info['fiber']:.1f}%")
    print()
    print(f"  Particle size ranges:")
    print(f"    - Protein: {FoodPowderSizeRanges.PROTEIN_D_MIN*1e6:.0f}-{FoodPowderSizeRanges.PROTEIN_D_MAX*1e6:.0f} μm (d50={FoodPowderSizeRanges.PROTEIN_D50*1e6:.0f} μm)")
    print(f"    - Starch:  {FoodPowderSizeRanges.STARCH_D_MIN*1e6:.0f}-{FoodPowderSizeRanges.STARCH_D_MAX*1e6:.0f} μm (d50={FoodPowderSizeRanges.STARCH_D50*1e6:.0f} μm)")
    print(f"    - Fiber:   {FoodPowderSizeRanges.FIBER_D_MIN*1e6:.0f}-{FoodPowderSizeRanges.FIBER_D_MAX*1e6:.0f} μm (d50={FoodPowderSizeRanges.FIBER_D50*1e6:.0f} μm)")
    print()


def create_injection_positions(
    num_particles: int,
    inlet_radius: float = 0.12,
    inlet_height: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Create initial particle positions at the classifier inlet.
    
    Particles are distributed in an annular region representing
    the tangential inlet of a cyclone classifier.
    
    Args:
        num_particles: Number of particles
        inlet_radius: Radial position of inlet
        inlet_height: Height of inlet
        seed: Random seed
        
    Returns:
        Positions array (N, 3)
    """
    rng = np.random.default_rng(seed)
    
    # Distribute particles in a ring (tangential inlet)
    theta = rng.uniform(0, 2 * np.pi, num_particles)
    r = rng.uniform(inlet_radius * 0.8, inlet_radius, num_particles)
    
    positions = np.zeros((num_particles, 3), dtype=np.float32)
    positions[:, 0] = r * np.cos(theta)
    positions[:, 1] = inlet_height + rng.uniform(-0.02, 0.02, num_particles)
    positions[:, 2] = r * np.sin(theta)
    
    return positions


def setup_classifier_geometry(system: WarpParticleSystem):
    """
    Configure the air classifier geometry and flow field.
    
    Sets up:
    - Domain boundaries (cylindrical cyclone shape approximation)
    - Rankine vortex flow parameters
    - Collection zones (fines outlet top, coarse outlet bottom)
    """
    # Domain bounds (approximate cyclone shape)
    system.set_domain_bounds(
        min_bound=(-0.2, -0.8, -0.2),  # Bottom cone
        max_bound=(0.2, 1.5, 0.2),      # Top cylinder
    )
    
    # Vortex parameters (Rankine vortex model)
    # These approximate a typical air classifier cyclone
    system.set_vortex_parameters(
        center=(0.0, 0.3, 0.0),           # Vortex axis center
        core_radius=0.04,                  # Inner vortex core (m)
        max_tangential_velocity=15.0,      # Maximum swirl velocity (m/s)
        axial_velocity_up=5.0,             # Upward velocity in core (m/s)
        axial_velocity_down=-2.0,          # Downward velocity near wall (m/s)
        radial_velocity=-0.4,              # Inward radial velocity (m/s)
    )
    
    # Collection zones
    system.set_collection_zones(
        fines_y_min=1.2,                   # Fines exit above this height
        coarse_y_max=-0.6,                 # Coarse exit below this height
        cyclone_center=(0.0, 0.0, 0.0),    # Cyclone center
        cyclone_bottom_y=-0.4,             # Dust outlet position
        cyclone_radius=0.08,               # Collection radius
    )


def run_simulation(
    source: str = "yellow_pea",
    num_particles: int = 10000,
    simulation_time: float = 0.5,
    dt: float = 1e-5,
    verbose: bool = True,
) -> dict:
    """
    Run the protein separation simulation.
    
    Args:
        source: Material source ("yellow_pea", "faba_bean", "oat")
        num_particles: Number of particles to simulate
        simulation_time: Total simulation time (seconds)
        dt: Time step (seconds)
        verbose: Print progress information
        
    Returns:
        Dictionary with simulation results
    """
    if verbose:
        print(f"Setting up simulation with {num_particles:,} particles...")
    
    # Initialize Warp
    wp.init()
    
    # Create particle system
    config = ParticleSystemConfig(
        max_particles=num_particles * 2,  # Allow headroom
        gravity=9.81,
        include_gravity=True,
        include_drag=True,
        include_centrifugal=True,
        fluid_density=AirProperties.DENSITY,
        fluid_viscosity=AirProperties.DYNAMIC_VISCOSITY,
        device="cuda",
    )
    
    system = WarpParticleSystem(config)
    
    # Setup geometry
    setup_classifier_geometry(system)
    
    # Create injection positions
    positions = create_injection_positions(num_particles)
    
    # Add initial tangential velocity (particles enter with swirl)
    velocities = np.zeros((num_particles, 3), dtype=np.float32)
    for i in range(num_particles):
        x, z = positions[i, 0], positions[i, 2]
        r = np.sqrt(x**2 + z**2)
        if r > 0.01:
            # Tangential direction (counterclockwise)
            velocities[i, 0] = -z / r * 8.0  # Initial tangential velocity
            velocities[i, 2] = x / r * 8.0
            velocities[i, 1] = -0.5  # Slight downward
    
    if verbose:
        print(f"Injecting mixed {source.replace('_', ' ')} powder...")
    
    # Inject mixed powder (protein + starch + fiber fractions)
    system.inject_mixed_powder(positions, source, seed=42)
    
    # Add velocities
    system.velocities = wp.from_numpy(
        velocities[:system.num_particles].astype(np.float32),
        dtype=wp.vec3,
        device=system.device
    )
    
    if verbose:
        initial_stats = system.get_statistics()
        print(f"  Total particles: {initial_stats['total_particles']:,}")
        print(f"  Active particles: {initial_stats['active_particles']:,}")
        print()
    
    # Run simulation
    total_steps = int(simulation_time / dt)
    output_interval = max(1, total_steps // 20)  # 20 progress updates
    
    if verbose:
        print(f"Running simulation ({simulation_time}s, {total_steps:,} steps)...")
        print()
    
    start_time = time.time()
    
    for step in range(total_steps):
        system.step(dt)
        
        if verbose and step % output_interval == 0:
            stats = system.get_statistics()
            progress = (step / total_steps) * 100
            print(f"  Progress: {progress:5.1f}% | "
                  f"Active: {stats['active_particles']:5,} | "
                  f"Fines: {stats['collected_fines']:5,} | "
                  f"Coarse: {stats['collected_coarse']:5,} | "
                  f"Efficiency: {stats['separation_efficiency']:.1%}")
    
    # Synchronize GPU
    system.synchronize()
    
    elapsed_time = time.time() - start_time
    
    # Get final statistics
    final_stats = system.get_statistics()
    
    if verbose:
        print()
        print(f"Simulation completed in {elapsed_time:.2f}s")
        print(f"  Performance: {total_steps / elapsed_time:.0f} steps/s")
        print()
    
    # Add system reference to results for visualization
    final_stats['system'] = system
    final_stats['elapsed_time'] = elapsed_time
    final_stats['source'] = source
    
    return final_stats


def print_results(results: dict):
    """Print detailed simulation results."""
    print("=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print()
    
    print(f"Material: {results['source'].replace('_', ' ').title()}")
    print(f"Simulation time: {results['time']:.4f} s")
    print(f"Computation time: {results['elapsed_time']:.2f} s")
    print()
    
    print("Particle Distribution:")
    print(f"  Total particles:     {results['total_particles']:>8,}")
    print(f"  Active (in flight):  {results['active_particles']:>8,}")
    print(f"  Collected as fines:  {results['collected_fines']:>8,}")
    print(f"  Collected as coarse: {results['collected_coarse']:>8,}")
    print(f"  In cyclone:          {results['collected_cyclone']:>8,}")
    print()
    
    print("Separation Performance:")
    print(f"  Separation efficiency: {results['separation_efficiency']:.1%}")
    print(f"  Protein recovery:      {results['protein_recovery']:.1%}")
    print(f"  Starch rejection:      {results['starch_rejection']:.1%}")
    print()
    
    print("Particle Size Analysis:")
    print(f"  Mean fines diameter:   {results['fines_mean_diameter_um']:.1f} μm")
    print(f"  Mean coarse diameter:  {results['coarse_mean_diameter_um']:.1f} μm")
    print()
    
    # Quality assessment
    print("Quality Assessment:")
    if results['protein_recovery'] > 0.7:
        print("  ✓ Good protein recovery (>70%)")
    else:
        print("  ✗ Low protein recovery (<70%) - consider adjusting air velocity")
    
    if results['starch_rejection'] > 0.6:
        print("  ✓ Good starch rejection (>60%)")
    else:
        print("  ✗ Low starch rejection (<60%) - consider adjusting cut point")
    
    cut_ratio = results['coarse_mean_diameter_um'] / max(1, results['fines_mean_diameter_um'])
    if cut_ratio > 2.0:
        print(f"  ✓ Good size separation (coarse/fines = {cut_ratio:.1f}x)")
    else:
        print(f"  ✗ Poor size separation (coarse/fines = {cut_ratio:.1f}x)")
    
    print()
    print("=" * 70)


def visualize_results(results: dict, save_path: str = None):
    """
    Visualize the simulation results.
    
    Args:
        results: Simulation results dictionary
        save_path: Optional path to save animation
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available. Install pyvista:")
        print("  pip install pyvista")
        return
    
    system = results['system']
    source = results['source']
    
    print("Launching interactive visualization...")
    print("  Color coding: RED = Protein, BLUE = Starch, GREEN = Fiber")
    print()
    
    config = ParticleVisualizationConfig(
        title=f"{source.replace('_', ' ').title()} Protein Separation",
        color_mode=ColorMode.BY_TYPE,
        point_size=6.0,
        show_collection_zones=True,
        show_stats=True,
        window_size=(1400, 900),
    )
    
    visualizer = ParticleVisualizer(config)
    visualizer.show(
        system,
        bounds_min=(-0.2, -0.8, -0.2),
        bounds_max=(0.2, 1.5, 0.2),
    )


def run_animation(
    source: str = "yellow_pea",
    num_particles: int = 5000,
    simulation_time: float = 0.3,
    save_path: str = None,
):
    """
    Run an animated simulation.
    
    Args:
        source: Material source
        num_particles: Number of particles
        simulation_time: Total time
        save_path: Path to save animation
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization not available for animation.")
        return
    
    print(f"Setting up animated simulation...")
    
    # Create system
    config = ParticleSystemConfig(
        max_particles=num_particles * 2,
        device="cuda",
    )
    system = WarpParticleSystem(config)
    setup_classifier_geometry(system)
    
    # Inject particles
    positions = create_injection_positions(num_particles)
    system.inject_mixed_powder(positions, source)
    
    # Add initial velocity
    velocities = np.zeros((num_particles, 3), dtype=np.float32)
    for i in range(num_particles):
        x, z = positions[i, 0], positions[i, 2]
        r = np.sqrt(x**2 + z**2)
        if r > 0.01:
            velocities[i, 0] = -z / r * 8.0
            velocities[i, 2] = x / r * 8.0
            velocities[i, 1] = -0.5
    
    # Configure visualization
    viz_config = ParticleVisualizationConfig(
        title=f"{source.replace('_', ' ').title()} Air Classification",
        color_mode=ColorMode.BY_TYPE,
        point_size=5.0,
        show_collection_zones=True,
        fps=30,
    )
    
    visualizer = ParticleVisualizer(viz_config)
    
    print(f"Starting animation ({simulation_time}s simulation)...")
    if save_path:
        print(f"Saving to: {save_path}")
    
    visualizer.animate(
        system,
        dt=1e-5,
        total_time=simulation_time,
        bounds_min=(-0.2, -0.8, -0.2),
        bounds_max=(0.2, 1.5, 0.2),
        save_path=save_path,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Food Protein Air Classification Simulation"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="yellow_pea",
        choices=["yellow_pea", "faba_bean", "oat"],
        help="Food powder source material"
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=10000,
        help="Number of particles to simulate"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.5,
        help="Simulation time in seconds"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show interactive visualization after simulation"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Run animated simulation"
    )
    parser.add_argument(
        "--save-animation",
        type=str,
        default=None,
        help="Path to save animation video (e.g., output.mp4)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    print_material_info(args.source)
    
    if args.animate:
        run_animation(
            source=args.source,
            num_particles=min(args.particles, 5000),  # Limit for animation
            simulation_time=args.time,
            save_path=args.save_animation,
        )
    else:
        # Run simulation
        results = run_simulation(
            source=args.source,
            num_particles=args.particles,
            simulation_time=args.time,
            verbose=True,
        )
        
        # Print results
        print_results(results)
        
        # Visualize if requested
        if args.visualize:
            visualize_results(results)


if __name__ == "__main__":
    main()
