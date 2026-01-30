"""
Flow visualization utilities for cyclone air classifier.

Provides functions for visualizing velocity fields, streamlines,
vortex structures, and particle trajectories.
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
import numpy as np

from ..utils.constants import PI


def plot_velocity_magnitude_slice(
    flow_field,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 50,
    ax=None,
    title: str = "Velocity Magnitude",
    cmap: str = "jet",
    show_colorbar: bool = True
):
    """
    Plot velocity magnitude on a horizontal slice.

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate of the slice
        x_range: (x_min, x_max) range in meters
        z_range: (z_min, z_max) range in meters
        resolution: Grid resolution
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        show_colorbar: Whether to show colorbar

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

    # Compute velocity magnitudes
    V_mag = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], y_slice, Z[i, j]])
            vel = flow_field.velocity_at(pos)
            V_mag[i, j] = np.linalg.norm(vel)

    # Plot
    im = ax.pcolormesh(X * 1000, Z * 1000, V_mag, shading='auto', cmap=cmap)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')

    return ax


def plot_velocity_components_slice(
    flow_field,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 30,
    figsize: Tuple[float, float] = (15, 5)
):
    """
    Plot tangential, axial, and radial velocity components.

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate of the slice
        x_range: (x_min, x_max) range
        z_range: (z_min, z_max) range
        resolution: Grid resolution
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Z = np.meshgrid(x, z)

    # Compute velocity components
    V_tan = np.zeros_like(X)
    V_axial = np.zeros_like(X)
    V_radial = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], y_slice, Z[i, j]])
            vel = flow_field.velocity_at(pos)

            # Radial distance
            r = np.sqrt(pos[0]**2 + pos[2]**2)

            if r > 1e-6:
                # Unit vectors
                radial_unit = np.array([pos[0]/r, 0, pos[2]/r])
                tangent_unit = np.array([-pos[2]/r, 0, pos[0]/r])

                V_tan[i, j] = np.dot(vel, tangent_unit)
                V_radial[i, j] = np.dot(vel, radial_unit)
            else:
                V_tan[i, j] = 0
                V_radial[i, j] = 0

            V_axial[i, j] = vel[1]

    # Plot tangential velocity
    im0 = axes[0].pcolormesh(X * 1000, Z * 1000, V_tan, shading='auto', cmap='RdBu_r')
    plt.colorbar(im0, ax=axes[0], label='m/s')
    axes[0].set_title('Tangential Velocity')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Z (mm)')
    axes[0].set_aspect('equal')

    # Plot axial velocity
    im1 = axes[1].pcolormesh(X * 1000, Z * 1000, V_axial, shading='auto', cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1], label='m/s')
    axes[1].set_title('Axial Velocity')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Z (mm)')
    axes[1].set_aspect('equal')

    # Plot radial velocity
    im2 = axes[2].pcolormesh(X * 1000, Z * 1000, V_radial, shading='auto', cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[2], label='m/s')
    axes[2].set_title('Radial Velocity')
    axes[2].set_xlabel('X (mm)')
    axes[2].set_ylabel('Z (mm)')
    axes[2].set_aspect('equal')

    plt.tight_layout()

    return fig


def plot_velocity_vectors(
    flow_field,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 20,
    ax=None,
    title: str = "Velocity Vectors",
    scale: float = 1.0,
    color_by_magnitude: bool = True
):
    """
    Plot velocity vectors (quiver plot) on a horizontal slice.

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate of the slice
        x_range: (x_min, x_max) range
        z_range: (z_min, z_max) range
        resolution: Number of arrows per dimension
        ax: Matplotlib axes
        title: Plot title
        scale: Arrow scale factor
        color_by_magnitude: Color arrows by velocity magnitude

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

    # Plot
    if color_by_magnitude:
        quiver = ax.quiver(
            X * 1000, Z * 1000, U, W, V_mag,
            cmap='jet', scale=scale * 50, width=0.003
        )
        plt.colorbar(quiver, ax=ax, label='Velocity (m/s)')
    else:
        ax.quiver(X * 1000, Z * 1000, U, W, scale=scale * 50, width=0.003)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')

    return ax


def compute_streamlines(
    flow_field,
    seed_points: np.ndarray,
    max_length: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 1000
) -> List[np.ndarray]:
    """
    Compute streamlines from seed points.

    Args:
        flow_field: CycloneFlowField instance
        seed_points: Array of seed point positions (N, 3)
        max_length: Maximum streamline length
        dt: Integration time step
        max_steps: Maximum number of integration steps

    Returns:
        List of streamline trajectories as (M, 3) arrays
    """
    streamlines = []

    for seed in seed_points:
        trajectory = [seed.copy()]
        pos = seed.copy()
        length = 0.0

        for step in range(max_steps):
            # Get velocity at current position
            vel = flow_field.velocity_at(pos)
            speed = np.linalg.norm(vel)

            if speed < 1e-6:
                break

            # Normalize and advance
            direction = vel / speed
            step_size = min(dt * speed, max_length - length)
            pos = pos + direction * step_size
            length += step_size

            trajectory.append(pos.copy())

            if length >= max_length:
                break

        if len(trajectory) > 1:
            streamlines.append(np.array(trajectory))

    return streamlines


def plot_streamlines_2d(
    streamlines: List[np.ndarray],
    plane: str = "xz",
    ax=None,
    title: str = "Streamlines",
    color: str = "blue",
    linewidth: float = 0.5,
    alpha: float = 0.7,
    show_direction: bool = True
):
    """
    Plot 2D projection of streamlines.

    Args:
        streamlines: List of streamline trajectories
        plane: Projection plane ("xz", "xy", or "yz")
        ax: Matplotlib axes
        title: Plot title
        color: Line color
        linewidth: Line width
        alpha: Line transparency
        show_direction: Show direction arrows

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plane indices
    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    labels_map = {"xy": ("X", "Y"), "xz": ("X", "Z"), "yz": ("Y", "Z")}
    idx1, idx2 = plane_map.get(plane, (0, 2))
    xlabel, ylabel = labels_map.get(plane, ("X", "Z"))

    # Plot streamlines
    for stream in streamlines:
        x = stream[:, idx1] * 1000
        y = stream[:, idx2] * 1000
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)

        # Add direction arrows
        if show_direction and len(stream) > 5:
            mid = len(stream) // 2
            dx = x[mid+1] - x[mid-1]
            dy = y[mid+1] - y[mid-1]
            ax.annotate(
                '', xy=(x[mid], y[mid]),
                xytext=(x[mid] - dx*0.3, y[mid] - dy*0.3),
                arrowprops=dict(arrowstyle='->', color=color, lw=1)
            )

    ax.set_xlabel(f'{xlabel} (mm)')
    ax.set_ylabel(f'{ylabel} (mm)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_streamlines_3d(
    streamlines: List[np.ndarray],
    ax=None,
    title: str = "3D Streamlines",
    color: str = "blue",
    linewidth: float = 0.5,
    alpha: float = 0.7
):
    """
    Plot streamlines in 3D.

    Args:
        streamlines: List of streamline trajectories
        ax: Matplotlib 3D axes
        title: Plot title
        color: Line color or "velocity" for velocity-based coloring
        linewidth: Line width
        alpha: Line transparency

    Returns:
        Matplotlib 3D axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

    for stream in streamlines:
        x = stream[:, 0] * 1000
        y = stream[:, 1] * 1000
        z = stream[:, 2] * 1000
        ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)

    return ax


def plot_radial_velocity_profile(
    flow_field,
    y_position: float,
    r_max: Optional[float] = None,
    component: str = "tangential",
    ax=None,
    title: Optional[str] = None,
    num_points: int = 100
):
    """
    Plot velocity profile as a function of radius.

    Args:
        flow_field: CycloneFlowField instance
        y_position: Y-coordinate for profile
        r_max: Maximum radius (default: cylinder radius)
        component: Velocity component ("tangential", "axial", "radial", "magnitude")
        ax: Matplotlib axes
        title: Plot title
        num_points: Number of points in profile

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if r_max is None:
        r_max = flow_field.params.cylinder_radius * 0.95

    r = np.linspace(0.001, r_max, num_points)
    v = np.zeros(num_points)

    for i, ri in enumerate(r):
        pos = np.array([ri, y_position, 0.0])
        vel = flow_field.velocity_at(pos)

        if component == "tangential":
            # Tangential velocity (z-direction for point on x-axis)
            v[i] = abs(vel[2])
        elif component == "axial":
            v[i] = vel[1]
        elif component == "radial":
            v[i] = vel[0]
        elif component == "magnitude":
            v[i] = np.linalg.norm(vel)
        else:
            v[i] = np.linalg.norm(vel)

    ax.plot(r * 1000, v, 'b-', linewidth=2)

    # Mark vortex finder radius
    vf_r = flow_field.params.vortex_finder_radius
    ax.axvline(vf_r * 1000, color='r', linestyle='--',
               label=f'Vortex finder r = {vf_r*1000:.1f} mm')

    # Mark core radius
    core_r = flow_field.params.core_radius
    ax.axvline(core_r * 1000, color='g', linestyle=':',
               label=f'Core r = {core_r*1000:.1f} mm')

    ax.set_xlabel('Radial Position (mm)')
    ax.set_ylabel(f'{component.capitalize()} Velocity (m/s)')

    if title is None:
        title = f'{component.capitalize()} Velocity Profile at Y = {y_position*1000:.1f} mm'
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_axial_velocity_profile(
    flow_field,
    r_position: float,
    y_range: Optional[Tuple[float, float]] = None,
    ax=None,
    title: Optional[str] = None,
    num_points: int = 100
):
    """
    Plot velocity profile along the cyclone axis.

    Args:
        flow_field: CycloneFlowField instance
        r_position: Radial position for profile
        y_range: (y_min, y_max) range
        ax: Matplotlib axes
        title: Plot title
        num_points: Number of points in profile

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    if y_range is None:
        total_height = flow_field.params.cylinder_height + flow_field.params.cone_height
        y_range = (0, total_height)

    y = np.linspace(y_range[0], y_range[1], num_points)
    v_tan = np.zeros(num_points)
    v_axial = np.zeros(num_points)

    for i, yi in enumerate(y):
        pos = np.array([r_position, yi, 0.0])
        vel = flow_field.velocity_at(pos)
        v_tan[i] = abs(vel[2])  # Tangential (z for point on x-axis)
        v_axial[i] = vel[1]     # Axial

    ax.plot(v_tan, y * 1000, 'b-', linewidth=2, label='Tangential')
    ax.plot(v_axial, y * 1000, 'r-', linewidth=2, label='Axial')

    # Mark cylinder/cone transition
    cyl_h = flow_field.params.cylinder_height
    ax.axhline(cyl_h * 1000, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.9, cyl_h * 1000 + 5, 'Cyl/Cone', ha='right')

    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Axial Position Y (mm)')

    if title is None:
        title = f'Velocity Profile at r = {r_position*1000:.1f} mm'
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_vortex_structure(
    flow_field,
    y_slices: List[float],
    r_max: Optional[float] = None,
    figsize: Tuple[float, float] = (15, 5)
):
    """
    Visualize the vortex structure at multiple heights.

    Args:
        flow_field: CycloneFlowField instance
        y_slices: List of Y positions to visualize
        r_max: Maximum radius for plots
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    n_slices = len(y_slices)
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)

    if n_slices == 1:
        axes = [axes]

    if r_max is None:
        r_max = flow_field.params.cylinder_radius

    for i, y_slice in enumerate(y_slices):
        ax = axes[i]

        # Get local radius at this height
        R_local = flow_field.get_local_radius(y_slice)

        # Create grid
        x_range = (-R_local * 1.1, R_local * 1.1)
        z_range = (-R_local * 1.1, R_local * 1.1)

        plot_velocity_vectors(
            flow_field, y_slice,
            x_range, z_range,
            resolution=15,
            ax=ax,
            title=f'Y = {y_slice*1000:.0f} mm',
            color_by_magnitude=True
        )

        # Draw cyclone boundary
        theta = np.linspace(0, 2*PI, 100)
        ax.plot(R_local * np.cos(theta) * 1000,
                R_local * np.sin(theta) * 1000,
                'k-', linewidth=2)

        # Draw vortex finder if in range
        if y_slice < flow_field.params.vortex_finder_radius:
            vf_r = flow_field.params.vortex_finder_radius
            ax.plot(vf_r * np.cos(theta) * 1000,
                    vf_r * np.sin(theta) * 1000,
                    'r--', linewidth=1.5)

    plt.tight_layout()

    return fig


def create_flow_animation(
    flow_field,
    y_range: Tuple[float, float],
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    num_frames: int = 50,
    resolution: int = 30,
    filename: Optional[str] = None,
    fps: int = 10
):
    """
    Create animation of flow field through different heights.

    Args:
        flow_field: CycloneFlowField instance
        y_range: (y_min, y_max) range for animation
        x_range: (x_min, x_max) range
        z_range: (z_min, z_max) range
        num_frames: Number of animation frames
        resolution: Grid resolution
        filename: Output filename (gif or mp4)
        fps: Frames per second

    Returns:
        Matplotlib animation object
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 10))

    y_values = np.linspace(y_range[0], y_range[1], num_frames)

    # Create initial grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    X, Z = np.meshgrid(x, z)

    def update(frame):
        ax.clear()

        y_slice = y_values[frame]

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

        # Plot
        quiver = ax.quiver(
            X * 1000, Z * 1000, U, W, V_mag,
            cmap='jet', scale=50
        )

        # Draw boundary
        R_local = flow_field.get_local_radius(y_slice)
        theta = np.linspace(0, 2*PI, 100)
        ax.plot(R_local * np.cos(theta) * 1000,
                R_local * np.sin(theta) * 1000,
                'k-', linewidth=2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'Flow Field at Y = {y_slice*1000:.1f} mm')
        ax.set_aspect('equal')
        ax.set_xlim(x_range[0] * 1000, x_range[1] * 1000)
        ax.set_ylim(z_range[0] * 1000, z_range[1] * 1000)

        return [quiver]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000//fps, blit=False)

    if filename:
        if filename.endswith('.gif'):
            anim.save(filename, writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps)
        print(f"Animation saved to: {filename}")

    return anim


def plot_pressure_gradient_estimate(
    flow_field,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 50,
    fluid_density: float = 1.225,
    ax=None
):
    """
    Estimate and plot pressure gradient from velocity field.

    Uses simplified Bernoulli equation for incompressible flow:
    p + 0.5 * rho * v^2 = const

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate of the slice
        x_range: (x_min, x_max) range
        z_range: (z_min, z_max) range
        resolution: Grid resolution
        fluid_density: Fluid density [kg/m3]
        ax: Matplotlib axes

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

    # Compute pressure estimate (relative)
    P = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], y_slice, Z[i, j]])
            vel = flow_field.velocity_at(pos)
            v_sq = np.sum(vel**2)
            # Dynamic pressure (negative for high velocity regions)
            P[i, j] = -0.5 * fluid_density * v_sq

    # Shift to make center reference
    P_center = P[resolution//2, resolution//2]
    P_relative = P - P_center

    # Plot
    im = ax.pcolormesh(X * 1000, Z * 1000, P_relative, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Relative Pressure (Pa)')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'Estimated Pressure Distribution at Y = {y_slice*1000:.1f} mm')
    ax.set_aspect('equal')

    return ax


def plot_flow_summary(
    flow_field,
    y_slice: float = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None
):
    """
    Create a comprehensive flow visualization summary.

    Args:
        flow_field: CycloneFlowField instance
        y_slice: Y-coordinate for slice plots (default: mid-cylinder)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if y_slice is None:
        y_slice = flow_field.params.cylinder_height / 2

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get geometry bounds
    R = flow_field.params.cylinder_radius
    x_range = (-R * 1.1, R * 1.1)
    z_range = (-R * 1.1, R * 1.1)

    # Plot 1: Velocity magnitude
    plot_velocity_magnitude_slice(
        flow_field, y_slice, x_range, z_range,
        resolution=50, ax=axes[0, 0],
        title=f'Velocity Magnitude (Y = {y_slice*1000:.0f} mm)'
    )

    # Add cyclone boundary
    theta = np.linspace(0, 2*PI, 100)
    axes[0, 0].plot(R * np.cos(theta) * 1000, R * np.sin(theta) * 1000, 'k-', lw=2)

    # Plot 2: Velocity vectors
    plot_velocity_vectors(
        flow_field, y_slice, x_range, z_range,
        resolution=15, ax=axes[0, 1],
        title=f'Velocity Vectors (Y = {y_slice*1000:.0f} mm)'
    )
    axes[0, 1].plot(R * np.cos(theta) * 1000, R * np.sin(theta) * 1000, 'k-', lw=2)

    # Plot 3: Tangential velocity profile
    plot_radial_velocity_profile(
        flow_field, y_slice, r_max=R * 0.95,
        component="tangential", ax=axes[1, 0],
        title=f'Tangential Velocity Profile (Y = {y_slice*1000:.0f} mm)'
    )

    # Plot 4: Axial velocity profile
    plot_radial_velocity_profile(
        flow_field, y_slice, r_max=R * 0.95,
        component="axial", ax=axes[1, 1],
        title=f'Axial Velocity Profile (Y = {y_slice*1000:.0f} mm)'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Flow summary saved to: {save_path}")

    return fig


def generate_seed_points_ring(
    center: Tuple[float, float, float],
    radius: float,
    y_offset: float = 0.0,
    num_points: int = 24
) -> np.ndarray:
    """
    Generate seed points in a ring pattern for streamlines.

    Args:
        center: Center of the ring
        radius: Ring radius
        y_offset: Y offset from center
        num_points: Number of points in ring

    Returns:
        Array of shape (num_points, 3)
    """
    theta = np.linspace(0, 2*PI, num_points, endpoint=False)

    points = np.zeros((num_points, 3))
    points[:, 0] = center[0] + radius * np.cos(theta)
    points[:, 1] = center[1] + y_offset
    points[:, 2] = center[2] + radius * np.sin(theta)

    return points


def generate_seed_points_grid(
    x_range: Tuple[float, float],
    y_value: float,
    z_range: Tuple[float, float],
    nx: int = 10,
    nz: int = 10
) -> np.ndarray:
    """
    Generate seed points on a rectangular grid.

    Args:
        x_range: (x_min, x_max) range
        y_value: Y-coordinate
        z_range: (z_min, z_max) range
        nx: Number of points in x
        nz: Number of points in z

    Returns:
        Array of shape (nx*nz, 3)
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    z = np.linspace(z_range[0], z_range[1], nz)
    X, Z = np.meshgrid(x, z)

    points = np.zeros((nx * nz, 3))
    points[:, 0] = X.ravel()
    points[:, 1] = y_value
    points[:, 2] = Z.ravel()

    return points
