"""
Particle visualization and animation system using NVIDIA Warp and PyVista.

Provides real-time and offline visualization of particle simulations for
air classification of food powders (protein separation from peas, beans, oats).

Features:
- Real-time particle rendering with color-coding by type/diameter/velocity
- Animation recording to video/GIF
- Interactive 3D visualization with PyVista
- GPU-accelerated particle data processing
- Flow field visualization (streamlines, glyphs)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable, Union, TYPE_CHECKING
from enum import Enum
from pathlib import Path
import numpy as np
import time

if TYPE_CHECKING:
    import pyvista as pv

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    wp = None
    WARP_AVAILABLE = False

try:
    import pyvista as pv
    from pyvista import themes
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None
    PYVISTA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


class ColorMode(Enum):
    """Particle color-coding mode."""
    UNIFORM = "uniform"             # Single color for all particles
    BY_TYPE = "by_type"             # Color by particle type (protein/starch/fiber)
    BY_DIAMETER = "by_diameter"     # Color by particle size
    BY_VELOCITY = "by_velocity"     # Color by velocity magnitude
    BY_STATE = "by_state"           # Color by collection state
    BY_AGE = "by_age"               # Color by time in simulation


class RenderMode(Enum):
    """Rendering mode for particles."""
    POINTS = "points"               # Simple point cloud
    SPHERES = "spheres"             # Scaled spheres (slower but accurate)
    GLYPHS = "glyphs"               # Glyph-based rendering


@dataclass
class ParticleVisualizationConfig:
    """Configuration for particle visualization."""
    
    # Display settings
    window_size: Tuple[int, int] = (1280, 720)
    background_color: str = "white"
    title: str = "Air Classifier Particle Simulation"
    
    # Particle appearance
    color_mode: ColorMode = ColorMode.BY_TYPE
    render_mode: RenderMode = RenderMode.POINTS
    point_size: float = 5.0
    sphere_scale: float = 1000.0  # Scale factor for sphere radius
    opacity: float = 0.9
    
    # Color schemes
    uniform_color: str = "#4A90D9"
    type_colors: Dict[int, str] = field(default_factory=lambda: {
        0: "#808080",  # WHOLE - Gray
        1: "#E74C3C",  # PROTEIN - Red
        2: "#3498DB",  # STARCH - Blue
        3: "#2ECC71",  # FIBER - Green
        4: "#F39C12",  # OTHER - Orange
    })
    state_colors: Dict[int, str] = field(default_factory=lambda: {
        1: "#4A90D9",   # ACTIVE - Blue
        -1: "#E74C3C",  # FINES - Red
        -2: "#3498DB",  # COARSE - Blue
        -3: "#9B59B6",  # CYCLONE_1 - Purple
        -4: "#8E44AD",  # CYCLONE_2 - Dark Purple
        -5: "#6C3483",  # CYCLONE_3 - Darker Purple
        -6: "#1ABC9C",  # BAG_FILTER - Teal
    })
    colormap_diameter: str = "viridis"
    colormap_velocity: str = "plasma"
    colormap_age: str = "cool"
    
    # Domain visualization
    show_domain_box: bool = True
    domain_box_color: str = "gray"
    domain_box_opacity: float = 0.1
    
    # Collection zones
    show_collection_zones: bool = True
    fines_zone_color: str = "#FFCCCB"
    coarse_zone_color: str = "#ADD8E6"
    
    # Flow field
    show_flow_field: bool = False
    flow_arrow_scale: float = 0.1
    flow_arrow_color: str = "#808080"
    
    # Axes and annotations
    show_axes: bool = True
    show_colorbar: bool = True
    show_stats: bool = True
    
    # Animation
    fps: int = 30
    
    # Coordinate system
    up_axis: str = "Y"  # Y-up for visualization


class ParticleColorMapper:
    """Maps particle properties to colors."""
    
    def __init__(self, config: ParticleVisualizationConfig):
        self.config = config
    
    def get_colors(
        self,
        positions: np.ndarray,
        velocities: np.ndarray = None,
        diameters: np.ndarray = None,
        particle_types: np.ndarray = None,
        states: np.ndarray = None,
        ages: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute colors for particles based on color mode.
        
        Returns:
            RGB colors array (N, 3) with values in [0, 1]
        """
        n = len(positions)
        
        if self.config.color_mode == ColorMode.UNIFORM:
            return self._uniform_colors(n)
        
        elif self.config.color_mode == ColorMode.BY_TYPE:
            return self._type_colors(particle_types, n)
        
        elif self.config.color_mode == ColorMode.BY_DIAMETER:
            return self._diameter_colors(diameters, n)
        
        elif self.config.color_mode == ColorMode.BY_VELOCITY:
            return self._velocity_colors(velocities, n)
        
        elif self.config.color_mode == ColorMode.BY_STATE:
            return self._state_colors(states, n)
        
        elif self.config.color_mode == ColorMode.BY_AGE:
            return self._age_colors(ages, n)
        
        return self._uniform_colors(n)
    
    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        """Convert hex color to RGB array."""
        hex_color = hex_color.lstrip('#')
        return np.array([
            int(hex_color[0:2], 16) / 255.0,
            int(hex_color[2:4], 16) / 255.0,
            int(hex_color[4:6], 16) / 255.0,
        ])
    
    def _uniform_colors(self, n: int) -> np.ndarray:
        rgb = self._hex_to_rgb(self.config.uniform_color)
        return np.tile(rgb, (n, 1))
    
    def _type_colors(self, types: np.ndarray, n: int) -> np.ndarray:
        if types is None:
            return self._uniform_colors(n)
        
        colors = np.zeros((n, 3))
        for type_val, hex_color in self.config.type_colors.items():
            mask = types == type_val
            if np.any(mask):
                colors[mask] = self._hex_to_rgb(hex_color)
        return colors
    
    def _state_colors(self, states: np.ndarray, n: int) -> np.ndarray:
        if states is None:
            return self._uniform_colors(n)
        
        colors = np.zeros((n, 3))
        for state_val, hex_color in self.config.state_colors.items():
            mask = states == state_val
            if np.any(mask):
                colors[mask] = self._hex_to_rgb(hex_color)
        return colors
    
    def _diameter_colors(self, diameters: np.ndarray, n: int) -> np.ndarray:
        if diameters is None:
            return self._uniform_colors(n)
        
        cmap = plt.get_cmap(self.config.colormap_diameter)
        d_min, d_max = diameters.min(), diameters.max()
        if d_max - d_min < 1e-12:
            normalized = np.zeros(n)
        else:
            normalized = (diameters - d_min) / (d_max - d_min)
        
        return cmap(normalized)[:, :3]
    
    def _velocity_colors(self, velocities: np.ndarray, n: int) -> np.ndarray:
        if velocities is None:
            return self._uniform_colors(n)
        
        cmap = plt.get_cmap(self.config.colormap_velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        v_max = speeds.max()
        if v_max < 1e-12:
            normalized = np.zeros(n)
        else:
            normalized = speeds / v_max
        
        return cmap(normalized)[:, :3]
    
    def _age_colors(self, ages: np.ndarray, n: int) -> np.ndarray:
        if ages is None:
            return self._uniform_colors(n)
        
        cmap = plt.get_cmap(self.config.colormap_age)
        age_max = ages.max()
        if age_max < 1e-12:
            normalized = np.zeros(n)
        else:
            normalized = ages / age_max
        
        return cmap(normalized)[:, :3]


class ParticleVisualizer:
    """
    Real-time particle visualization using PyVista.
    
    Provides interactive 3D rendering of particle simulations with
    multiple color-coding options and animation capabilities.
    
    Example:
        >>> from airclassifier.particles import WarpParticleSystem
        >>> from airclassifier.visualization import ParticleVisualizer
        >>> 
        >>> system = WarpParticleSystem()
        >>> visualizer = ParticleVisualizer()
        >>> 
        >>> # Interactive visualization
        >>> visualizer.show(system)
        >>> 
        >>> # Animate simulation
        >>> visualizer.animate(system, dt=1e-5, steps=1000)
    """
    
    def __init__(self, config: ParticleVisualizationConfig = None):
        """
        Initialize the particle visualizer.
        
        Args:
            config: Visualization configuration
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for particle visualization. "
                            "Install with: pip install pyvista")
        
        self.config = config or ParticleVisualizationConfig()
        self.color_mapper = ParticleColorMapper(self.config)
        
        self.plotter = None
        self._particle_actor = None
        self._domain_actor = None
        self._stats_actor = None
        self._frame_count = 0
    
    def create_plotter(self, off_screen: bool = False) -> "pv.Plotter":
        """Create and configure the PyVista plotter."""
        self.plotter = pv.Plotter(
            window_size=self.config.window_size,
            off_screen=off_screen,
            title=self.config.title,
        )
        
        self.plotter.set_background(self.config.background_color)
        
        if self.config.up_axis.upper() == "Y":
            self.plotter.camera.up = (0, 1, 0)
        else:
            self.plotter.camera.up = (0, 0, 1)
        
        if self.config.show_axes:
            self.plotter.add_axes()
        
        return self.plotter
    
    def add_particles(
        self,
        positions: np.ndarray,
        velocities: np.ndarray = None,
        diameters: np.ndarray = None,
        particle_types: np.ndarray = None,
        states: np.ndarray = None,
        ages: np.ndarray = None,
    ):
        """
        Add or update particles in the visualization.
        
        Args:
            positions: Particle positions (N, 3)
            velocities: Particle velocities (N, 3)
            diameters: Particle diameters (N,)
            particle_types: Particle type indices (N,)
            states: Particle states (N,)
            ages: Particle ages (N,)
        """
        if self.plotter is None:
            self.create_plotter()
        
        # Filter active particles only
        if states is not None:
            active_mask = states == 1
            if not np.any(active_mask):
                return
            
            positions = positions[active_mask]
            if velocities is not None:
                velocities = velocities[active_mask]
            if diameters is not None:
                diameters = diameters[active_mask]
            if particle_types is not None:
                particle_types = particle_types[active_mask]
            if ages is not None:
                ages = ages[active_mask]
            states = states[active_mask]
        
        if len(positions) == 0:
            return
        
        # Get colors
        colors = self.color_mapper.get_colors(
            positions, velocities, diameters, particle_types, states, ages
        )
        
        # Create point cloud
        points = pv.PolyData(positions)
        points['colors'] = (colors * 255).astype(np.uint8)
        
        if diameters is not None:
            points['diameters'] = diameters
        
        # Remove old actor
        if self._particle_actor is not None:
            self.plotter.remove_actor(self._particle_actor)
        
        # Add new particles
        if self.config.render_mode == RenderMode.POINTS:
            self._particle_actor = self.plotter.add_points(
                points,
                scalars='colors',
                rgb=True,
                point_size=self.config.point_size,
                render_points_as_spheres=True,
                opacity=self.config.opacity,
            )
        elif self.config.render_mode == RenderMode.SPHERES and diameters is not None:
            # Create spheres at each point
            glyphs = points.glyph(
                geom=pv.Sphere(radius=1.0),
                scale='diameters',
                factor=self.config.sphere_scale,
            )
            self._particle_actor = self.plotter.add_mesh(
                glyphs,
                scalars=np.repeat(colors, 1, axis=0),  # Map colors
                rgb=True,
                opacity=self.config.opacity,
            )
        else:
            self._particle_actor = self.plotter.add_points(
                points,
                scalars='colors',
                rgb=True,
                point_size=self.config.point_size,
                render_points_as_spheres=True,
                opacity=self.config.opacity,
            )
    
    def add_domain_box(
        self,
        bounds_min: Tuple[float, float, float],
        bounds_max: Tuple[float, float, float],
    ):
        """Add domain bounding box visualization."""
        if not self.config.show_domain_box:
            return
        
        box = pv.Box(bounds=(
            bounds_min[0], bounds_max[0],
            bounds_min[1], bounds_max[1],
            bounds_min[2], bounds_max[2],
        ))
        
        if self._domain_actor is not None:
            self.plotter.remove_actor(self._domain_actor)
        
        self._domain_actor = self.plotter.add_mesh(
            box,
            style='wireframe',
            color=self.config.domain_box_color,
            line_width=2,
            opacity=0.5,
        )
    
    def add_collection_zones(
        self,
        fines_y_min: float,
        coarse_y_max: float,
        bounds_min: Tuple[float, float, float],
        bounds_max: Tuple[float, float, float],
    ):
        """Add collection zone visualization."""
        if not self.config.show_collection_zones:
            return
        
        # Fines zone (top)
        fines_box = pv.Box(bounds=(
            bounds_min[0], bounds_max[0],
            fines_y_min, bounds_max[1],
            bounds_min[2], bounds_max[2],
        ))
        self.plotter.add_mesh(
            fines_box,
            color=self.config.fines_zone_color,
            opacity=0.2,
            label="Fines (Protein)",
        )
        
        # Coarse zone (bottom)
        coarse_box = pv.Box(bounds=(
            bounds_min[0], bounds_max[0],
            bounds_min[1], coarse_y_max,
            bounds_min[2], bounds_max[2],
        ))
        self.plotter.add_mesh(
            coarse_box,
            color=self.config.coarse_zone_color,
            opacity=0.2,
            label="Coarse (Starch)",
        )
    
    def add_stats_text(self, stats: Dict[str, Any]):
        """Add statistics text overlay."""
        if not self.config.show_stats:
            return
        
        text = (
            f"Time: {stats.get('time', 0):.4f} s\n"
            f"Active: {stats.get('active_particles', 0):,}\n"
            f"Fines: {stats.get('collected_fines', 0):,}\n"
            f"Coarse: {stats.get('collected_coarse', 0):,}\n"
            f"Efficiency: {stats.get('separation_efficiency', 0):.1%}"
        )
        
        if self._stats_actor is not None:
            self.plotter.remove_actor(self._stats_actor)
        
        self._stats_actor = self.plotter.add_text(
            text,
            position='upper_left',
            font_size=10,
            color='black',
        )
    
    def show(
        self,
        particle_system,
        bounds_min: Tuple[float, float, float] = (-1, -2, -1),
        bounds_max: Tuple[float, float, float] = (1, 3, 1),
    ):
        """
        Show static visualization of current particle state.
        
        Args:
            particle_system: WarpParticleSystem instance
            bounds_min: Domain minimum bounds
            bounds_max: Domain maximum bounds
        """
        self.create_plotter()
        
        # Get particle data
        positions = particle_system.get_positions()
        velocities = particle_system.get_velocities()
        diameters = particle_system.get_diameters()
        states = particle_system.get_states()
        types = particle_system.get_particle_types()
        
        # Add visualization elements
        self.add_domain_box(bounds_min, bounds_max)
        self.add_collection_zones(
            particle_system.fines_y_min,
            particle_system.coarse_y_max,
            bounds_min, bounds_max,
        )
        self.add_particles(positions, velocities, diameters, types, states)
        
        # Add statistics
        stats = particle_system.get_statistics()
        self.add_stats_text(stats)
        
        # Show
        self.plotter.show()
    
    def animate(
        self,
        particle_system,
        dt: float,
        total_time: float = 1.0,
        bounds_min: Tuple[float, float, float] = (-1, -2, -1),
        bounds_max: Tuple[float, float, float] = (1, 3, 1),
        save_path: Optional[str] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Animate the particle simulation in real-time.
        
        Args:
            particle_system: WarpParticleSystem instance
            dt: Time step for simulation
            total_time: Total simulation time
            bounds_min: Domain minimum bounds
            bounds_max: Domain maximum bounds
            save_path: Optional path to save animation (mp4/gif)
            callback: Optional callback function called each frame
        """
        self.create_plotter(off_screen=save_path is not None)
        
        # Setup domain
        self.add_domain_box(bounds_min, bounds_max)
        self.add_collection_zones(
            particle_system.fines_y_min,
            particle_system.coarse_y_max,
            bounds_min, bounds_max,
        )
        
        # Initial particles
        positions = particle_system.get_positions()
        velocities = particle_system.get_velocities()
        diameters = particle_system.get_diameters()
        states = particle_system.get_states()
        types = particle_system.get_particle_types()
        self.add_particles(positions, velocities, diameters, types, states)
        
        # Animation settings
        total_steps = int(total_time / dt)
        frame_interval = max(1, total_steps // (int(total_time * self.config.fps)))
        
        if save_path:
            self.plotter.open_movie(save_path, framerate=self.config.fps)
        
        self._frame_count = 0
        
        def update_callback():
            """Callback for animation update."""
            # Step simulation
            for _ in range(frame_interval):
                particle_system.step(dt)
            
            # Update visualization
            positions = particle_system.get_positions()
            velocities = particle_system.get_velocities()
            diameters = particle_system.get_diameters()
            states = particle_system.get_states()
            types = particle_system.get_particle_types()
            
            self.add_particles(positions, velocities, diameters, types, states)
            
            # Update stats
            stats = particle_system.get_statistics()
            self.add_stats_text(stats)
            
            # User callback
            if callback:
                callback(self._frame_count, stats)
            
            self._frame_count += 1
            
            if save_path:
                self.plotter.write_frame()
            
            # Check termination
            if particle_system.time >= total_time:
                return False
            return True
        
        # Run animation
        if save_path:
            # Offline rendering
            while update_callback():
                pass
            self.plotter.close()
        else:
            # Interactive
            self.plotter.add_callback(update_callback, interval=1000//self.config.fps)
            self.plotter.show()
    
    def close(self):
        """Close the visualizer."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None


class MatplotlibParticleAnimator:
    """
    Matplotlib-based particle animator for simpler visualizations.
    
    Good for 2D projections and when PyVista is not available.
    """
    
    def __init__(self, config: ParticleVisualizationConfig = None):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required")
        
        self.config = config or ParticleVisualizationConfig()
        self.color_mapper = ParticleColorMapper(self.config)
    
    def create_animation(
        self,
        particle_system,
        dt: float,
        total_time: float = 1.0,
        projection: str = "xy",
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """
        Create a matplotlib animation of the simulation.
        
        Args:
            particle_system: WarpParticleSystem instance
            dt: Time step
            total_time: Total simulation time
            projection: Projection plane ("xy", "xz", "yz", "3d")
            save_path: Optional path to save animation
            
        Returns:
            Matplotlib FuncAnimation object
        """
        fig = plt.figure(figsize=(10, 8))
        
        if projection == "3d":
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
        else:
            ax = fig.add_subplot(111)
            if projection == "xy":
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
            elif projection == "xz":
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Z (m)')
            else:  # yz
                ax.set_xlabel('Y (m)')
                ax.set_ylabel('Z (m)')
        
        ax.set_title(self.config.title)
        
        # Initialize scatter
        positions = particle_system.get_positions()
        states = particle_system.get_states()
        types = particle_system.get_particle_types()
        
        active_mask = states == 1
        active_pos = positions[active_mask]
        active_types = types[active_mask]
        
        colors = self.color_mapper._type_colors(active_types, len(active_types))
        
        if projection == "3d":
            scatter = ax.scatter(
                active_pos[:, 0], active_pos[:, 1], active_pos[:, 2],
                c=colors, s=self.config.point_size, alpha=self.config.opacity
            )
        else:
            if projection == "xy":
                scatter = ax.scatter(
                    active_pos[:, 0], active_pos[:, 1],
                    c=colors, s=self.config.point_size, alpha=self.config.opacity
                )
            elif projection == "xz":
                scatter = ax.scatter(
                    active_pos[:, 0], active_pos[:, 2],
                    c=colors, s=self.config.point_size, alpha=self.config.opacity
                )
            else:
                scatter = ax.scatter(
                    active_pos[:, 1], active_pos[:, 2],
                    c=colors, s=self.config.point_size, alpha=self.config.opacity
                )
        
        # Text for stats
        stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        total_steps = int(total_time / dt)
        frame_interval = max(1, total_steps // (int(total_time * self.config.fps)))
        
        def init():
            return scatter, stats_text
        
        def update(frame):
            # Step simulation
            for _ in range(frame_interval):
                particle_system.step(dt)
            
            # Get data
            positions = particle_system.get_positions()
            states = particle_system.get_states()
            types = particle_system.get_particle_types()
            
            active_mask = states == 1
            active_pos = positions[active_mask]
            active_types = types[active_mask]
            
            if len(active_pos) > 0:
                colors = self.color_mapper._type_colors(active_types, len(active_types))
                
                if projection == "3d":
                    scatter._offsets3d = (active_pos[:, 0], active_pos[:, 1], active_pos[:, 2])
                else:
                    if projection == "xy":
                        scatter.set_offsets(active_pos[:, :2])
                    elif projection == "xz":
                        scatter.set_offsets(active_pos[:, [0, 2]])
                    else:
                        scatter.set_offsets(active_pos[:, 1:])
                
                scatter.set_facecolor(colors)
            
            # Update stats
            stats = particle_system.get_statistics()
            stats_text.set_text(
                f"Time: {stats['time']:.3f}s\n"
                f"Active: {stats['active_particles']}\n"
                f"Fines: {stats['collected_fines']}\n"
                f"Coarse: {stats['collected_coarse']}\n"
                f"Efficiency: {stats['separation_efficiency']:.1%}"
            )
            
            return scatter, stats_text
        
        n_frames = int(total_time * self.config.fps)
        anim = animation.FuncAnimation(
            fig, update, init_func=init,
            frames=n_frames, interval=1000//self.config.fps,
            blit=True
        )
        
        if save_path:
            writer = animation.FFMpegWriter(fps=self.config.fps)
            anim.save(save_path, writer=writer)
        
        return anim


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def visualize_particles(
    particle_system,
    color_by: str = "type",
    **kwargs
) -> ParticleVisualizer:
    """
    Quick visualization of a particle system.
    
    Args:
        particle_system: WarpParticleSystem instance
        color_by: How to color particles ("type", "diameter", "velocity", "state")
        **kwargs: Additional ParticleVisualizationConfig parameters
        
    Returns:
        ParticleVisualizer instance
    """
    color_mode_map = {
        "type": ColorMode.BY_TYPE,
        "diameter": ColorMode.BY_DIAMETER,
        "velocity": ColorMode.BY_VELOCITY,
        "state": ColorMode.BY_STATE,
        "age": ColorMode.BY_AGE,
        "uniform": ColorMode.UNIFORM,
    }
    
    config = ParticleVisualizationConfig(
        color_mode=color_mode_map.get(color_by, ColorMode.BY_TYPE),
        **kwargs
    )
    
    visualizer = ParticleVisualizer(config)
    visualizer.show(particle_system)
    
    return visualizer


def animate_simulation(
    particle_system,
    dt: float = 1e-5,
    total_time: float = 0.5,
    color_by: str = "type",
    save_path: Optional[str] = None,
    **kwargs
) -> ParticleVisualizer:
    """
    Animate a particle simulation.
    
    Args:
        particle_system: WarpParticleSystem instance
        dt: Time step [s]
        total_time: Total simulation time [s]
        color_by: How to color particles
        save_path: Optional path to save animation
        **kwargs: Additional configuration
        
    Returns:
        ParticleVisualizer instance
    """
    color_mode_map = {
        "type": ColorMode.BY_TYPE,
        "diameter": ColorMode.BY_DIAMETER,
        "velocity": ColorMode.BY_VELOCITY,
        "state": ColorMode.BY_STATE,
    }
    
    config = ParticleVisualizationConfig(
        color_mode=color_mode_map.get(color_by, ColorMode.BY_TYPE),
        **kwargs
    )
    
    visualizer = ParticleVisualizer(config)
    visualizer.animate(particle_system, dt, total_time, save_path=save_path)
    
    return visualizer


def create_separation_animation(
    source: str = "yellow_pea",
    num_particles: int = 5000,
    total_time: float = 0.5,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a complete protein separation animation.
    
    Args:
        source: "yellow_pea", "faba_bean", or "oat"
        num_particles: Number of particles
        total_time: Simulation time
        save_path: Path to save animation
        
    Returns:
        Dictionary with simulation results
    """
    from ..particles.particle_system import WarpParticleSystem, ParticleSystemConfig
    
    # Create system
    config = ParticleSystemConfig(max_particles=num_particles * 2)
    system = WarpParticleSystem(config)
    
    # Configure for realistic cyclone
    system.set_domain_bounds((-0.5, -1.0, -0.5), (0.5, 2.0, 0.5))
    system.set_vortex_parameters(
        center=(0.0, 0.5, 0.0),
        core_radius=0.05,
        max_tangential_velocity=12.0,
        axial_velocity_up=4.0,
        axial_velocity_down=-1.5,
        radial_velocity=-0.3,
    )
    system.set_collection_zones(
        fines_y_min=1.5,
        coarse_y_max=-0.8,
        cyclone_center=(0.0, 0.0, 0.0),
        cyclone_bottom_y=-0.5,
        cyclone_radius=0.08,
    )
    
    # Generate initial positions (at inlet)
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2*np.pi, num_particles)
    r = rng.uniform(0.1, 0.15, num_particles)
    
    positions = np.zeros((num_particles, 3))
    positions[:, 0] = r * np.cos(theta)
    positions[:, 1] = 0.3 + rng.uniform(-0.05, 0.05, num_particles)
    positions[:, 2] = r * np.sin(theta)
    
    # Inject mixed powder
    system.inject_mixed_powder(positions.astype(np.float32), source)
    
    # Animate
    config = ParticleVisualizationConfig(
        title=f"{source.replace('_', ' ').title()} Protein Separation",
        color_mode=ColorMode.BY_TYPE,
        show_collection_zones=True,
    )
    
    visualizer = ParticleVisualizer(config)
    visualizer.animate(
        system,
        dt=1e-5,
        total_time=total_time,
        bounds_min=(-0.5, -1.0, -0.5),
        bounds_max=(0.5, 2.0, 0.5),
        save_path=save_path,
    )
    
    return system.get_statistics()
