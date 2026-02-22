"""
Run Hammer Mill Simulation
==========================

Example script to run a hammer mill simulation and display results.

Usage:
    python examples/run_hammer_mill_simulation.py
"""

import numpy as np

from airclassifier.milling import (
    HammerMillSimulator,
    MillConfig,
    MillRecipe,
    run_milling_simulation,
)


def run_example():
    """Run an example hammer mill simulation."""
    print("=" * 60)
    print("Hammer Mill Simulation Example")
    print("=" * 60)

    # Configure the mill
    config = MillConfig(
        rotor_rpm=3000,
        rotor_diameter_m=0.20,
        rotor_length_m=0.30,
        hammer_rows=4,
        hammers_per_row=4,
        hammer_length_m=0.08,
        screen_inner_radius_m=0.188,
        housing_inner_radius_m=0.20,
        screen_aperture_mm=1.5,
        motor_power_kw=22.0,
    )

    # Create a recipe
    recipe = MillRecipe(
        name="Fine Flour",
        rotor_rpm=3000,
        screen_aperture_mm=1.5,
        feed_rate_kg_per_hr=500,
        feed_d50_um=3000,  # 3mm whole seeds
    )

    print(f"\nMill Configuration:")
    print(f"  Rotor RPM: {config.rotor_rpm}")
    print(f"  Hammer tip speed: {config.hammer_tip_speed:.1f} m/s")
    print(f"  Screen aperture: {config.screen_aperture_mm} mm")
    print(f"  Motor power: {config.motor_power_kw} kW")

    print(f"\nRecipe: {recipe.name}")
    print(f"  Feed rate: {recipe.feed_rate_kg_per_hr} kg/hr")
    print(f"  Feed size (d50): {recipe.feed_d50_um} um")

    # Create simulator
    print("\nCreating simulator...")
    sim = HammerMillSimulator.create(config=config, recipe=recipe)

    # Initialize with some material in the mill
    print("Initializing with 100g holdup...")
    sim.initialize(initial_holdup_kg=0.1)

    # Run simulation
    duration = 2.0  # seconds
    dt = 0.002      # 2ms timestep
    print(f"\nRunning simulation for {duration}s with dt={dt*1000:.0f}ms...")

    result = sim.run(duration_s=duration, dt=dt)

    # Display results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nTime series: {len(result.history)} steps")

    if len(result.history) > 0:
        last = result.history[-1]
        print(f"\nFinal state:")
        print(f"  Particles in mill: {last.num_particles}")
        print(f"  Holdup: {last.holdup_kg * 1000:.1f} g")
        print(f"  Power: {last.power_kw:.1f} kW")

        print(f"\nImpact statistics:")
        print(f"  Impacts this step: {last.num_impacts}")
        print(f"  Total impact energy: {last.total_impact_energy_j:.3f} J")

        print(f"\nBreakage statistics:")
        print(f"  Breakage events: {last.num_breakage_events}")
        print(f"  Size reduction ratio: {last.mean_size_reduction:.2f}")

        print(f"\nScreen statistics:")
        print(f"  Particles passed: {last.num_passed_screen}")
        print(f"  Passage rate: {last.screen_passage_rate * 100:.1f}%")

    print(f"\nProduct PSD:")
    print(f"  Total discharged: {result.psd_total_mass_kg * 1000:.1f} g")
    print(f"  d10: {result.d10_um:.0f} um")
    print(f"  d50: {result.d50_um:.0f} um")
    print(f"  d90: {result.d90_um:.0f} um")

    print(f"\nProcess KPIs:")
    print(f"  Mean power: {result.mean_power_kw:.1f} kW")
    print(f"  Throughput: {result.throughput_kg_per_hr:.0f} kg/hr")
    if result.specific_energy_kwh_per_t > 0:
        print(f"  Specific energy: {result.specific_energy_kwh_per_t:.1f} kWh/t")

    # Get outlet state for classifier
    outlet = sim.get_outlet_conditions()
    print(f"\nOutlet state (for classifier):")
    print(f"  d50: {outlet.d50_um:.0f} um")
    print(f"  Throughput: {outlet.throughput_kg_per_hr:.0f} kg/hr")

    return result


def plot_results(result):
    """Plot simulation results using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not installed. Skipping plots.")
        return

    if len(result.history) == 0:
        print("No history to plot.")
        return

    # Extract time series
    times = [s.time_s for s in result.history]
    holdups = [s.holdup_kg * 1000 for s in result.history]  # g
    powers = [s.power_kw for s in result.history]
    impacts = [s.num_impacts for s in result.history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Hammer Mill Simulation Results", fontsize=14)

    # Holdup vs time
    ax = axes[0, 0]
    ax.plot(times, holdups, 'b-')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Holdup (g)")
    ax.set_title("Material in Mill")
    ax.grid(True, alpha=0.3)

    # Power vs time
    ax = axes[0, 1]
    ax.plot(times, powers, 'r-')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Power Draw")
    ax.grid(True, alpha=0.3)

    # Impacts vs time
    ax = axes[1, 0]
    ax.plot(times, impacts, 'g-')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Impacts per step")
    ax.set_title("Impact Events")
    ax.grid(True, alpha=0.3)

    # PSD (if available)
    ax = axes[1, 1]
    if len(result.psd_mass_fractions) > 0:
        sizes_um = result.psd_size_classes_m[:-1] * 1e6
        ax.bar(range(len(result.psd_mass_fractions)), result.psd_mass_fractions)
        ax.set_xlabel("Size class")
        ax.set_ylabel("Mass fraction")
        ax.set_title(f"Product PSD (d50={result.d50_um:.0f} um)")
    else:
        ax.text(0.5, 0.5, "No PSD data", ha='center', va='center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hammer_mill_results.png", dpi=150)
    print("\nPlot saved to: hammer_mill_results.png")
    plt.show()


if __name__ == "__main__":
    result = run_example()
    plot_results(result)
