"""
Real-time rendering for cyclone air classifier simulation.

Provides OpenGL-based visualization using pyglet for real-time
particle visualization during simulation.
"""

from typing import Optional, Tuple
import numpy as np


class SimpleRenderer:
    """
    Simple matplotlib-based renderer for quick visualization.

    For more advanced real-time visualization, use the pyglet-based
    renderer or export to VTK for ParaView.
    """

    def __init__(self, cyclone_radius: float, cyclone_height: float):
        """
        Initialize renderer.

        Args:
            cyclone_radius: Cyclone body radius [m]
            cyclone_height: Total cyclone height [m]
        """
        self.cyclone_radius = cyclone_radius
        self.cyclone_height = cyclone_height
        self.fig = None
        self.ax = None
        self._initialized = False

    def initialize(self):
        """Initialize matplotlib figure."""
        import matplotlib.pyplot as plt

        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._initialized = True

    def update(
        self,
        positions: np.ndarray,
        is_active: Optional[np.ndarray] = None,
        title: str = ""
    ):
        """
        Update the display with new particle positions.

        Args:
            positions: Particle positions (N, 3)
            is_active: Particle active status
            title: Plot title
        """
        if not self._initialized:
            self.initialize()

        self.ax.clear()

        # Convert to mm
        x = positions[:, 0] * 1000
        z = positions[:, 2] * 1000

        # Color by status
        if is_active is not None:
            active_mask = is_active == 1
            collected_mask = is_active == -1
            escaped_mask = is_active == -2

            if np.any(active_mask):
                self.ax.scatter(x[active_mask], z[active_mask],
                               c='blue', s=1, alpha=0.5, label='Active')
            if np.any(collected_mask):
                self.ax.scatter(x[collected_mask], z[collected_mask],
                               c='green', s=1, alpha=0.5, label='Collected')
            if np.any(escaped_mask):
                self.ax.scatter(x[escaped_mask], z[escaped_mask],
                               c='red', s=1, alpha=0.5, label='Escaped')
        else:
            self.ax.scatter(x, z, c='blue', s=1, alpha=0.5)

        # Draw cyclone outline
        theta = np.linspace(0, 2 * np.pi, 100)
        r_mm = self.cyclone_radius * 1000
        self.ax.plot(r_mm * np.cos(theta), r_mm * np.sin(theta),
                    'k-', linewidth=2)

        self.ax.set_xlim(-r_mm * 1.2, r_mm * 1.2)
        self.ax.set_ylim(-r_mm * 1.2, r_mm * 1.2)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Z (mm)')
        self.ax.set_title(title)
        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper right')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the renderer."""
        import matplotlib.pyplot as plt
        if self._initialized:
            plt.close(self.fig)
            self._initialized = False


class PygletRenderer:
    """
    Pyglet-based OpenGL renderer for high-performance visualization.

    Provides smooth real-time rendering of large particle counts.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "Cyclone Simulation"
    ):
        """
        Initialize pyglet renderer.

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title
        self.window = None
        self._initialized = False

    def initialize(self):
        """Initialize pyglet window."""
        try:
            import pyglet
            from pyglet import gl
        except ImportError:
            print("Pyglet not available. Install with: pip install pyglet")
            return False

        self.window = pyglet.window.Window(
            width=self.width,
            height=self.height,
            caption=self.title
        )

        # Set up OpenGL
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self._initialized = True
        return True

    def update(
        self,
        positions: np.ndarray,
        is_active: Optional[np.ndarray] = None,
        camera_distance: float = 2.0
    ):
        """
        Update display with new positions.

        Args:
            positions: Particle positions
            is_active: Particle status
            camera_distance: Camera distance from origin
        """
        if not self._initialized:
            if not self.initialize():
                return

        try:
            import pyglet
            from pyglet import gl
        except ImportError:
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set up projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(45.0, self.width / self.height, 0.1, 100.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.gluLookAt(
            camera_distance, camera_distance, camera_distance,
            0, 0, 0,
            0, 1, 0
        )

        # Draw particles as points
        gl.glPointSize(2.0)
        gl.glBegin(gl.GL_POINTS)

        for i, pos in enumerate(positions):
            if is_active is not None:
                if is_active[i] == 1:
                    gl.glColor4f(0.2, 0.2, 1.0, 0.5)  # Blue - active
                elif is_active[i] == -1:
                    gl.glColor4f(0.2, 1.0, 0.2, 0.5)  # Green - collected
                elif is_active[i] == -2:
                    gl.glColor4f(1.0, 0.2, 0.2, 0.5)  # Red - escaped
                else:
                    continue  # Skip inactive
            else:
                gl.glColor4f(0.2, 0.2, 1.0, 0.5)

            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()

        self.window.flip()

    def run(self):
        """Run the pyglet event loop."""
        if self._initialized:
            import pyglet
            pyglet.app.run()

    def close(self):
        """Close the renderer."""
        if self._initialized and self.window:
            self.window.close()
            self._initialized = False


def visualize_results_interactive(
    results: dict,
    cyclone_radius: float
):
    """
    Launch interactive visualization of simulation results.

    Args:
        results: Simulation results dictionary
        cyclone_radius: Cyclone body radius
    """
    from .plotters import create_interactive_3d_plot

    fig = create_interactive_3d_plot(
        results['positions'],
        is_active=results['is_active'],
        diameters=results['diameters']
    )

    if fig is not None:
        fig.show()
