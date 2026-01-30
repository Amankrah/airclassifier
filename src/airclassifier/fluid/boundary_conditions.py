"""
Boundary conditions for cyclone flow simulation.

Provides inlet, outlet, and wall boundary condition implementations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable
import numpy as np
import warp as wp


class BoundaryType(Enum):
    """Types of boundary conditions."""

    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    PERIODIC = "periodic"
    SYMMETRY = "symmetry"


@dataclass
class InletCondition:
    """Inlet boundary condition parameters."""

    velocity: Tuple[float, float, float]  # (vx, vy, vz) [m/s]
    turbulence_intensity: float = 0.05    # Fraction
    length_scale: float = 0.01            # Turbulent length scale [m]

    @property
    def velocity_magnitude(self) -> float:
        return np.sqrt(sum(v**2 for v in self.velocity))

    @property
    def k_inlet(self) -> float:
        """Inlet turbulent kinetic energy."""
        return 1.5 * (self.turbulence_intensity * self.velocity_magnitude)**2

    @property
    def epsilon_inlet(self) -> float:
        """Inlet turbulent dissipation rate."""
        C_mu = 0.09
        return C_mu**(3/4) * self.k_inlet**(3/2) / self.length_scale


@dataclass
class OutletCondition:
    """Outlet boundary condition parameters."""

    pressure: float = 0.0     # Gauge pressure [Pa]
    backflow_velocity: Optional[Tuple[float, float, float]] = None


@dataclass
class WallCondition:
    """Wall boundary condition parameters."""

    no_slip: bool = True           # No-slip (True) or slip (False)
    moving_velocity: Optional[Tuple[float, float, float]] = None
    roughness: float = 0.0         # Wall roughness height [m]


@wp.kernel
def apply_inlet_boundary(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    is_inlet: wp.array3d(dtype=wp.int32),
    inlet_vx: float,
    inlet_vy: float,
    inlet_vz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Apply Dirichlet velocity condition at inlet cells."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if is_inlet[i, j, k] == 1:
        vel_x[i, j, k] = inlet_vx
        vel_y[i, j, k] = inlet_vy
        vel_z[i, j, k] = inlet_vz


@wp.kernel
def apply_outlet_boundary(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    pressure: wp.array3d(dtype=float),
    is_outlet: wp.array3d(dtype=wp.int32),
    outlet_pressure: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Apply pressure outlet boundary condition."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if is_outlet[i, j, k] == 1:
        # Fixed pressure
        pressure[i, j, k] = outlet_pressure

        # Zero gradient for velocity (Neumann)
        # Copy from interior neighbor
        if i > 0 and is_outlet[i-1, j, k] != 1:
            vel_x[i, j, k] = vel_x[i-1, j, k]
            vel_y[i, j, k] = vel_y[i-1, j, k]
            vel_z[i, j, k] = vel_z[i-1, j, k]
        elif i < nx - 1 and is_outlet[i+1, j, k] != 1:
            vel_x[i, j, k] = vel_x[i+1, j, k]
            vel_y[i, j, k] = vel_y[i+1, j, k]
            vel_z[i, j, k] = vel_z[i+1, j, k]


@wp.kernel
def apply_wall_boundary_no_slip(
    vel_x: wp.array3d(dtype=float),
    vel_y: wp.array3d(dtype=float),
    vel_z: wp.array3d(dtype=float),
    is_wall: wp.array3d(dtype=wp.int32),
    wall_vx: float,
    wall_vy: float,
    wall_vz: float,
    nx: int,
    ny: int,
    nz: int,
):
    """Apply no-slip wall boundary condition."""
    i, j, k = wp.tid()

    if i >= nx or j >= ny or k >= nz:
        return

    if is_wall[i, j, k] == 1:
        vel_x[i, j, k] = wall_vx
        vel_y[i, j, k] = wall_vy
        vel_z[i, j, k] = wall_vz


class BoundaryConditionManager:
    """
    Manages boundary conditions for the flow solver.

    Handles identification of boundary cells and application
    of appropriate conditions.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        grid_spacing: Tuple[float, float, float],
        device: str = "cuda"
    ):
        """
        Initialize boundary condition manager.

        Args:
            grid_shape: (nx, ny, nz) grid dimensions
            grid_spacing: (dx, dy, dz) cell sizes
            device: Warp device
        """
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.device = device

        nx, ny, nz = grid_shape

        # Boundary masks
        self.is_inlet = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self.is_outlet = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
        self.is_wall = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)

        # Conditions
        self.inlet_condition: Optional[InletCondition] = None
        self.outlet_condition: Optional[OutletCondition] = None
        self.wall_condition: Optional[WallCondition] = None

    def set_cyclone_boundaries(
        self,
        cylinder_radius: float,
        cone_height: float,
        cone_bottom_radius: float,
        cylinder_height: float,
        inlet_position: Tuple[float, float, float],
        inlet_size: Tuple[float, float, float],
        vortex_finder_radius: float,
        vortex_finder_bottom: float,
    ):
        """
        Set boundary cells for cyclone geometry.

        Args:
            cylinder_radius: Radius of cylindrical section [m]
            cone_height: Height of conical section [m]
            cone_bottom_radius: Radius at bottom of cone [m]
            cylinder_height: Height of cylindrical section [m]
            inlet_position: (x, y, z) position of inlet center [m]
            inlet_size: (width, height, depth) of inlet [m]
            vortex_finder_radius: Radius of vortex finder [m]
            vortex_finder_bottom: Y-position of vortex finder bottom [m]
        """
        nx, ny, nz = self.grid_shape
        dx, dy, dz = self.grid_spacing

        is_inlet_np = np.zeros((nx, ny, nz), dtype=np.int32)
        is_outlet_np = np.zeros((nx, ny, nz), dtype=np.int32)
        is_wall_np = np.zeros((nx, ny, nz), dtype=np.int32)

        # Domain center
        cx = nx * dx / 2
        cz = nz * dz / 2

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = i * dx
                    y = j * dy
                    z = k * dz

                    # Distance from axis
                    r = np.sqrt((x - cx)**2 + (z - cz)**2)

                    # Determine local wall radius based on y position
                    if y > cylinder_height:
                        # Cylindrical section
                        wall_r = cylinder_radius
                    else:
                        # Conical section
                        t = y / cylinder_height if cylinder_height > 0 else 1
                        wall_r = cone_bottom_radius + t * (cylinder_radius - cone_bottom_radius)

                    # Wall cells (outer boundary)
                    if r > wall_r * 0.95:
                        is_wall_np[i, j, k] = 1

                    # Inlet region
                    ix, iy, iz = inlet_position
                    iw, ih, idepth = inlet_size
                    if (abs(x - ix) < iw/2 and
                        abs(y - iy) < ih/2 and
                        abs(z - iz) < idepth/2):
                        is_inlet_np[i, j, k] = 1
                        is_wall_np[i, j, k] = 0

                    # Vortex finder outlet (top center)
                    if y > cylinder_height + cone_height - dy and r < vortex_finder_radius:
                        is_outlet_np[i, j, k] = 1
                        is_wall_np[i, j, k] = 0

                    # Dust outlet (bottom center)
                    if y < dy and r < cone_bottom_radius * 0.8:
                        is_outlet_np[i, j, k] = 1
                        is_wall_np[i, j, k] = 0

        # Copy to GPU
        wp.copy(self.is_inlet, wp.array(is_inlet_np, dtype=wp.int32, device=self.device))
        wp.copy(self.is_outlet, wp.array(is_outlet_np, dtype=wp.int32, device=self.device))
        wp.copy(self.is_wall, wp.array(is_wall_np, dtype=wp.int32, device=self.device))

    def apply_all(
        self,
        vel_x: wp.array,
        vel_y: wp.array,
        vel_z: wp.array,
        pressure: wp.array
    ):
        """
        Apply all boundary conditions.

        Args:
            vel_x, vel_y, vel_z: Velocity components
            pressure: Pressure field
        """
        nx, ny, nz = self.grid_shape

        # Apply inlet
        if self.inlet_condition:
            vx, vy, vz = self.inlet_condition.velocity
            wp.launch(
                kernel=apply_inlet_boundary,
                dim=(nx, ny, nz),
                inputs=[
                    vel_x, vel_y, vel_z, self.is_inlet,
                    vx, vy, vz, nx, ny, nz,
                ],
                device=self.device
            )

        # Apply outlet
        if self.outlet_condition:
            wp.launch(
                kernel=apply_outlet_boundary,
                dim=(nx, ny, nz),
                inputs=[
                    vel_x, vel_y, vel_z, pressure, self.is_outlet,
                    self.outlet_condition.pressure, nx, ny, nz,
                ],
                device=self.device
            )

        # Apply wall
        if self.wall_condition:
            if self.wall_condition.no_slip:
                wall_vel = self.wall_condition.moving_velocity or (0.0, 0.0, 0.0)
                wp.launch(
                    kernel=apply_wall_boundary_no_slip,
                    dim=(nx, ny, nz),
                    inputs=[
                        vel_x, vel_y, vel_z, self.is_wall,
                        wall_vel[0], wall_vel[1], wall_vel[2],
                        nx, ny, nz,
                    ],
                    device=self.device
                )
