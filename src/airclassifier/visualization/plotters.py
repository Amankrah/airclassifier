"""
Plotting utilities for cyclone air classifier visualization.

Provides functions for creating various plots of simulation results
using matplotlib and plotly.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np


def plot_particle_positions_2d(
    positions: np.ndarray,
    is_active: Optional[np.ndarray] = None,
    diameters: Optional[np.ndarray] = None,
    plane: str = "xz",
    cyclone_radius: Optional[float] = None,
    ax=None,
    title: str = "Particle Positions",
    colorby: str = "status"
):
    """
    Plot particle positions in a 2D projection.

    Args:
        positions: Particle positions (N, 3)
        is_active: Particle status array
        diameters: Particle diameters (for coloring)
        plane: Projection plane ("xy", "xz", or "yz")
        cyclone_radius: Optional cyclone radius for reference circle
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        colorby: What to color by ("status", "diameter", "velocity")

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Select plane coordinates
    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    labels_map = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}

    idx1, idx2 = plane_map.get(plane, (0, 2))
    xlabel, ylabel = labels_map.get(plane, ("X", "Z"))

    x = positions[:, idx1] * 1000  # Convert to mm
    y = positions[:, idx2] * 1000

    # Color particles
    if colorby == "status" and is_active is not None:
        colors = []
        for status in is_active:
            if status == 1:
                colors.append('blue')      # Active
            elif status == -1:
                colors.append('green')     # Collected
            elif status == -2:
                colors.append('red')       # Escaped
            else:
                colors.append('gray')      # Inactive

        ax.scatter(x, y, c=colors, s=1, alpha=0.5)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Active'),
            Patch(facecolor='green', label='Collected'),
            Patch(facecolor='red', label='Escaped'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    elif colorby == "diameter" and diameters is not None:
        sc = ax.scatter(x, y, c=diameters * 1e6, s=1, alpha=0.5, cmap='viridis')
        plt.colorbar(sc, ax=ax, label='Diameter (μm)')

    else:
        ax.scatter(x, y, c='blue', s=1, alpha=0.5)

    # Draw cyclone outline
    if cyclone_radius is not None and plane == "xz":
        theta = np.linspace(0, 2 * np.pi, 100)
        r_mm = cyclone_radius * 1000
        ax.plot(r_mm * np.cos(theta), r_mm * np.sin(theta), 'k-', linewidth=2)

    ax.set_xlabel(f'{xlabel} (mm)')
    ax.set_ylabel(f'{ylabel} (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_particle_positions_3d(
    positions: np.ndarray,
    is_active: Optional[np.ndarray] = None,
    diameters: Optional[np.ndarray] = None,
    ax=None,
    title: str = "3D Particle Positions",
    max_particles: int = 5000
):
    """
    Plot particle positions in 3D.

    Args:
        positions: Particle positions (N, 3)
        is_active: Particle status array
        diameters: Particle diameters
        ax: Matplotlib 3D axes
        title: Plot title
        max_particles: Max particles to plot (for performance)

    Returns:
        Matplotlib 3D axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Subsample if too many particles
    n = len(positions)
    if n > max_particles:
        idx = np.random.choice(n, max_particles, replace=False)
        positions = positions[idx]
        if is_active is not None:
            is_active = is_active[idx]
        if diameters is not None:
            diameters = diameters[idx]

    x = positions[:, 0] * 1000
    y = positions[:, 1] * 1000
    z = positions[:, 2] * 1000

    if is_active is not None:
        colors = np.where(is_active == 1, 'blue',
                         np.where(is_active == -1, 'green', 'red'))
        ax.scatter(x, y, z, c=colors, s=1, alpha=0.3)
    else:
        ax.scatter(x, y, z, c='blue', s=1, alpha=0.3)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)

    return ax


def plot_size_distribution(
    diameters: np.ndarray,
    num_bins: int = 50,
    ax=None,
    title: str = "Particle Size Distribution",
    log_scale: bool = False
):
    """
    Plot particle size distribution histogram.

    Args:
        diameters: Particle diameters [m]
        num_bins: Number of histogram bins
        ax: Matplotlib axes
        title: Plot title
        log_scale: Use log scale for x-axis

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    d_um = diameters * 1e6  # Convert to microns

    if log_scale:
        bins = np.logspace(np.log10(d_um.min()), np.log10(d_um.max()), num_bins)
    else:
        bins = num_bins

    ax.hist(d_um, bins=bins, edgecolor='black', alpha=0.7)

    if log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Particle Diameter (μm)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics
    d50 = np.median(d_um)
    d_mean = np.mean(d_um)
    ax.axvline(d50, color='r', linestyle='--', label=f'd50 = {d50:.1f} μm')
    ax.axvline(d_mean, color='g', linestyle='--', label=f'Mean = {d_mean:.1f} μm')
    ax.legend()

    return ax


def plot_velocity_field_slice(
    flow_field,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 30,
    ax=None,
    title: str = "Velocity Field"
):
    """
    Plot velocity field vectors at a horizontal slice.

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate of slice
        x_range: (x_min, x_max) range
        z_range: (z_min, z_max) range
        resolution: Grid resolution
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Z = np.meshgrid(x, z)

    # Compute velocities
    U = np.zeros_like(X)
    W = np.zeros_like(Z)
    V_mag = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], y_slice, Z[i, j]])
            vel = flow_field.velocity_at(pos)
            U[i, j] = vel[0]
            W[i, j] = vel[2]
            V_mag[i, j] = np.linalg.norm(vel)

    # Plot as quiver with color by magnitude
    ax.quiver(X * 1000, Z * 1000, U, W, V_mag, cmap='jet', alpha=0.8)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')

    return ax


def plot_tangential_velocity_profile(
    flow_field,
    y_position: float = 0.0,
    ax=None,
    title: str = "Tangential Velocity Profile"
):
    """
    Plot radial profile of tangential velocity.

    Args:
        flow_field: CycloneFlowField instance
        y_position: Y-coordinate for profile
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Radial positions
    r_max = flow_field.params.cylinder_radius * 0.95
    r = np.linspace(0.001, r_max, 100)

    # Compute tangential velocities
    v_tan = []
    for ri in r:
        pos = np.array([ri, y_position, 0.0])
        vel = flow_field.velocity_at(pos)
        # Tangential velocity is in Z direction for point on X-axis
        v_tan.append(abs(vel[2]))

    ax.plot(r * 1000, v_tan, 'b-', linewidth=2)

    # Mark vortex finder radius
    vf_r = flow_field.params.vortex_finder_radius
    ax.axvline(vf_r * 1000, color='r', linestyle='--',
              label=f'Vortex finder r = {vf_r*1000:.1f} mm')

    ax.set_xlabel('Radial Position (mm)')
    ax.set_ylabel('Tangential Velocity (m/s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_simulation_summary(
    results: Dict[str, Any],
    flow_field=None,
    cyclone_radius: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Create summary figure with multiple plots.

    Args:
        results: Simulation results dictionary
        flow_field: Optional CycloneFlowField for velocity plots
        cyclone_radius: Optional cyclone radius
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Size distribution
    plot_size_distribution(
        results['diameters'],
        ax=axes[0, 0],
        title='Particle Size Distribution'
    )

    # Plot 2: XZ positions (top view)
    plot_particle_positions_2d(
        results['positions'],
        is_active=results['is_active'],
        plane='xz',
        cyclone_radius=cyclone_radius,
        ax=axes[0, 1],
        title='Particle Positions (Top View)'
    )

    # Plot 3: XY positions (side view)
    plot_particle_positions_2d(
        results['positions'],
        is_active=results['is_active'],
        plane='xy',
        ax=axes[1, 0],
        title='Particle Positions (Side View)'
    )

    # Plot 4: Tangential velocity or statistics
    if flow_field is not None:
        plot_tangential_velocity_profile(
            flow_field,
            ax=axes[1, 1],
            title='Tangential Velocity Profile'
        )
    else:
        # Show statistics text
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        Simulation Results
        ==================

        Particles injected: {results['particles_injected']}
        Particles collected: {results['particles_collected']}
        Particles escaped: {results['particles_escaped']}
        Particles active: {results.get('particles_active', 'N/A')}

        Collection efficiency: {results['collection_efficiency']:.1%}

        Simulation time: {results['time']:.3f} s
        Total steps: {results['steps']}
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=12, family='monospace', verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary figure to: {save_path}")

    return fig


def create_interactive_3d_plot(
    positions: np.ndarray,
    is_active: Optional[np.ndarray] = None,
    diameters: Optional[np.ndarray] = None
):
    """
    Create interactive 3D plot using plotly.

    Args:
        positions: Particle positions
        is_active: Particle status
        diameters: Particle diameters

    Returns:
        Plotly figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return None

    # Subsample if too many particles
    max_particles = 10000
    n = len(positions)
    if n > max_particles:
        idx = np.random.choice(n, max_particles, replace=False)
        positions = positions[idx]
        if is_active is not None:
            is_active = is_active[idx]
        if diameters is not None:
            diameters = diameters[idx]

    x = positions[:, 0] * 1000
    y = positions[:, 1] * 1000
    z = positions[:, 2] * 1000

    # Color by status or diameter
    if is_active is not None:
        color = np.where(is_active == 1, 0,
                        np.where(is_active == -1, 1, 2))
        colorscale = [[0, 'blue'], [0.5, 'green'], [1, 'red']]
    elif diameters is not None:
        color = diameters * 1e6
        colorscale = 'Viridis'
    else:
        color = 'blue'
        colorscale = None

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=color,
            colorscale=colorscale,
            opacity=0.6
        )
    )])

    fig.update_layout(
        title='3D Particle Positions',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        )
    )

    return fig
