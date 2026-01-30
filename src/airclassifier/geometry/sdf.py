"""
Signed Distance Function (SDF) utilities for cyclone air classifier.

Provides unified SDF interface combining all cyclone components,
including:
- Analytical SDF evaluation
- SDF field generation on grids
- Gradient and normal computation
- GPU-accelerated Warp implementations
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Union
import numpy as np
import warp as wp

from ..utils.constants import PI


@dataclass
class CycloneSDFParams:
    """Parameters for cyclone SDF calculation."""

    # Geometry center
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Cyclone body
    cylinder_radius: float = 0.15      # [m]
    cylinder_height: float = 0.45      # [m]
    cone_height: float = 0.75          # [m]
    cone_bottom_radius: float = 0.05625  # [m]

    # Vortex finder
    vortex_finder_radius: float = 0.075   # [m]
    vortex_finder_bottom_y: float = 0.15  # [m] Y position of VF bottom

    # Dust outlet (extension below cone)
    dust_outlet_radius: float = 0.05625   # [m]
    dust_outlet_length: float = 0.1       # [m]

    @classmethod
    def from_assembly(cls, assembly) -> "CycloneSDFParams":
        """
        Create SDF parameters from a CycloneAssembly.

        Args:
            assembly: CycloneAssembly instance

        Returns:
            CycloneSDFParams instance
        """
        p = assembly.params

        # Calculate vortex finder bottom position
        vf_bottom_y = p.center[1] - p.vortex_finder_length

        return cls(
            center=p.center,
            cylinder_radius=p.cylinder_diameter / 2.0,
            cylinder_height=p.cylinder_height,
            cone_height=p.cone_height,
            cone_bottom_radius=p.cone_tip_diameter / 2.0,
            vortex_finder_radius=p.vortex_finder_diameter / 2.0,
            vortex_finder_bottom_y=vf_bottom_y,
            dust_outlet_radius=p.dust_outlet_diameter / 2.0,
            dust_outlet_length=p.dust_outlet_length,
        )

    @property
    def total_height(self) -> float:
        """Total height of cyclone body."""
        return self.cylinder_height + self.cone_height


class CycloneSDF:
    """
    Unified Signed Distance Function for complete cyclone geometry.

    Combines SDFs for:
    - Cyclone body (cylinder + cone)
    - Vortex finder (inner tube)
    - Dust outlet

    Convention:
    - Negative values: inside the cyclone (valid flow region)
    - Positive values: outside or inside walls
    - Zero: on the surface
    """

    def __init__(self, params: CycloneSDFParams):
        """
        Initialize cyclone SDF.

        Args:
            params: CycloneSDFParams defining the geometry
        """
        self.params = params
        self._warp_params = None

    def evaluate(self, point: np.ndarray) -> float:
        """
        Evaluate SDF at a single point.

        Args:
            point: 3D point [x, y, z]

        Returns:
            Signed distance (negative inside, positive outside)
        """
        return self._sdf_numpy(point)

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate SDF at multiple points.

        Args:
            points: Array of shape (N, 3)

        Returns:
            Array of shape (N,) with signed distances
        """
        return np.array([self._sdf_numpy(p) for p in points])

    def gradient(self, point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute SDF gradient (outward normal direction) at a point.

        Args:
            point: 3D point
            eps: Finite difference step size

        Returns:
            Gradient vector (normalized = surface normal)
        """
        grad = np.zeros(3)
        for i in range(3):
            p_plus = point.copy()
            p_minus = point.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            grad[i] = (self.evaluate(p_plus) - self.evaluate(p_minus)) / (2 * eps)
        return grad

    def normal(self, point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute outward surface normal at a point.

        Args:
            point: 3D point (should be near surface for accuracy)
            eps: Finite difference step size

        Returns:
            Unit normal vector pointing outward
        """
        grad = self.gradient(point, eps)
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            return grad / norm
        return np.array([0.0, 1.0, 0.0])

    def _sdf_numpy(self, point: np.ndarray) -> float:
        """Compute SDF using NumPy (CPU)."""
        p = self.params

        # Local coordinates (y positive going down into cyclone)
        local_x = point[0] - p.center[0]
        local_y = p.center[1] - point[1]  # Flip so positive is downward
        local_z = point[2] - p.center[2]

        # Radial distance from axis
        r = np.sqrt(local_x**2 + local_z**2)

        # Start with large positive distance (outside)
        min_dist = 1e10

        # Check if inside vortex finder region (special handling)
        if local_y < p.vortex_finder_bottom_y and local_y > -0.1:
            if r < p.vortex_finder_radius:
                # Inside vortex finder tube - this is valid region
                # Distance to VF inner wall (positive when inside tube)
                dist_vf = p.vortex_finder_radius - r
                # We want negative inside, so negate
                return -dist_vf

        # Above cylinder top
        if local_y < 0:
            if r <= p.cylinder_radius:
                # Above and inside cylinder projection
                return local_y  # Negative (inside) relative to top cap
            else:
                # Outside and above - distance to edge
                return np.sqrt((r - p.cylinder_radius)**2 + local_y**2)

        # In cylinder section
        elif local_y <= p.cylinder_height:
            R_wall = p.cylinder_radius
            # Distance to cylinder wall (negative inside)
            dist_wall = r - R_wall

            # Check if in VF exclusion zone
            if local_y < p.vortex_finder_bottom_y and r < p.vortex_finder_radius:
                # Inside VF region - already handled above
                pass

            return dist_wall

        # In cone section
        elif local_y <= p.total_height:
            cone_y = local_y - p.cylinder_height
            t = cone_y / p.cone_height
            R_wall = p.cylinder_radius * (1 - t) + p.cone_bottom_radius * t

            # Distance to slant surface
            # The cone surface has a normal with radial and axial components
            dr = p.cylinder_radius - p.cone_bottom_radius
            slant = np.sqrt(p.cone_height**2 + dr**2)

            # Perpendicular distance to slant line
            dist_slant = (r - R_wall) * p.cone_height / slant

            return dist_slant

        # Below cone (dust outlet region)
        else:
            y_below = local_y - p.total_height

            if r <= p.dust_outlet_radius:
                # Inside dust outlet pipe
                if y_below <= p.dust_outlet_length:
                    return r - p.dust_outlet_radius  # Distance to outlet wall
                else:
                    return y_below - p.dust_outlet_length  # Below outlet
            else:
                # Outside dust outlet
                return np.sqrt((r - p.dust_outlet_radius)**2 + y_below**2)

    def is_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the cyclone (valid flow region).

        Args:
            point: 3D point

        Returns:
            True if inside cyclone (SDF < 0)
        """
        return self.evaluate(point) < 0

    def classify_region(self, point: np.ndarray) -> str:
        """
        Classify which region of the cyclone a point is in.

        Args:
            point: 3D point

        Returns:
            Region name: "outside", "cylinder", "cone", "vortex_finder",
                        "dust_outlet", "above", "below"
        """
        p = self.params

        local_x = point[0] - p.center[0]
        local_y = p.center[1] - point[1]
        local_z = point[2] - p.center[2]
        r = np.sqrt(local_x**2 + local_z**2)

        # Check vortex finder first
        if local_y < p.vortex_finder_bottom_y and local_y > 0:
            if r < p.vortex_finder_radius:
                return "vortex_finder"

        # Check main regions
        if local_y < 0:
            return "above"
        elif local_y <= p.cylinder_height:
            if r <= p.cylinder_radius:
                return "cylinder"
            return "outside"
        elif local_y <= p.total_height:
            cone_y = local_y - p.cylinder_height
            t = cone_y / p.cone_height
            R_wall = p.cylinder_radius * (1 - t) + p.cone_bottom_radius * t
            if r <= R_wall:
                return "cone"
            return "outside"
        elif local_y <= p.total_height + p.dust_outlet_length:
            if r <= p.dust_outlet_radius:
                return "dust_outlet"
            return "outside"
        else:
            return "below"


# =============================================================================
# WARP SDF FUNCTIONS
# =============================================================================

@wp.struct
class WarpSDFParams:
    """Warp-compatible SDF parameters."""
    center: wp.vec3
    cylinder_radius: float
    cylinder_height: float
    cone_height: float
    cone_bottom_radius: float
    vortex_finder_radius: float
    vortex_finder_bottom_y: float
    dust_outlet_radius: float
    dust_outlet_length: float


@wp.func
def cyclone_sdf(pos: wp.vec3, params: WarpSDFParams) -> float:
    """
    Compute signed distance to cyclone surface.

    Args:
        pos: Query point
        params: Cyclone geometry parameters

    Returns:
        Signed distance (negative inside, positive outside)
    """
    # Local coordinates (y positive going down)
    local_x = pos[0] - params.center[0]
    local_y = params.center[1] - pos[1]
    local_z = pos[2] - params.center[2]

    r = wp.sqrt(local_x * local_x + local_z * local_z)
    total_height = params.cylinder_height + params.cone_height

    # Vortex finder region
    if local_y < params.vortex_finder_bottom_y and local_y > -0.1:
        if r < params.vortex_finder_radius:
            return -(params.vortex_finder_radius - r)

    # Above cylinder
    if local_y < 0.0:
        if r <= params.cylinder_radius:
            return local_y
        else:
            return wp.sqrt((r - params.cylinder_radius) ** 2.0 + local_y ** 2.0)

    # Cylinder section
    if local_y <= params.cylinder_height:
        return r - params.cylinder_radius

    # Cone section
    if local_y <= total_height:
        cone_y = local_y - params.cylinder_height
        t = cone_y / params.cone_height
        R_wall = params.cylinder_radius * (1.0 - t) + params.cone_bottom_radius * t

        dr = params.cylinder_radius - params.cone_bottom_radius
        slant = wp.sqrt(params.cone_height * params.cone_height + dr * dr)
        return (r - R_wall) * params.cone_height / slant

    # Below cone (dust outlet)
    y_below = local_y - total_height
    if r <= params.dust_outlet_radius:
        if y_below <= params.dust_outlet_length:
            return r - params.dust_outlet_radius
        else:
            return y_below - params.dust_outlet_length
    else:
        return wp.sqrt((r - params.dust_outlet_radius) ** 2.0 + y_below ** 2.0)


@wp.func
def cyclone_sdf_gradient(pos: wp.vec3, params: WarpSDFParams) -> wp.vec3:
    """
    Compute SDF gradient (surface normal direction).

    Uses central differences for numerical gradient.
    """
    eps = 1.0e-5

    dx_plus = cyclone_sdf(wp.vec3(pos[0] + eps, pos[1], pos[2]), params)
    dx_minus = cyclone_sdf(wp.vec3(pos[0] - eps, pos[1], pos[2]), params)
    dy_plus = cyclone_sdf(wp.vec3(pos[0], pos[1] + eps, pos[2]), params)
    dy_minus = cyclone_sdf(wp.vec3(pos[0], pos[1] - eps, pos[2]), params)
    dz_plus = cyclone_sdf(wp.vec3(pos[0], pos[1], pos[2] + eps), params)
    dz_minus = cyclone_sdf(wp.vec3(pos[0], pos[1], pos[2] - eps), params)

    grad_x = (dx_plus - dx_minus) / (2.0 * eps)
    grad_y = (dy_plus - dy_minus) / (2.0 * eps)
    grad_z = (dz_plus - dz_minus) / (2.0 * eps)

    return wp.vec3(grad_x, grad_y, grad_z)


@wp.kernel
def compute_sdf_field(
    positions: wp.array(dtype=wp.vec3),
    sdf_values: wp.array(dtype=float),
    params: WarpSDFParams
):
    """
    Compute SDF values at multiple positions (GPU kernel).
    """
    tid = wp.tid()
    sdf_values[tid] = cyclone_sdf(positions[tid], params)


@wp.kernel
def compute_sdf_gradient_field(
    positions: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.vec3),
    params: WarpSDFParams
):
    """
    Compute SDF gradients at multiple positions (GPU kernel).
    """
    tid = wp.tid()
    gradients[tid] = cyclone_sdf_gradient(positions[tid], params)


@wp.kernel
def classify_points_inside(
    positions: wp.array(dtype=wp.vec3),
    inside_flags: wp.array(dtype=wp.int32),
    params: WarpSDFParams
):
    """
    Classify points as inside (1) or outside (0) the cyclone.
    """
    tid = wp.tid()
    sdf = cyclone_sdf(positions[tid], params)
    if sdf < 0.0:
        inside_flags[tid] = 1
    else:
        inside_flags[tid] = 0


def create_warp_sdf_params(params: CycloneSDFParams) -> WarpSDFParams:
    """
    Convert CycloneSDFParams to WarpSDFParams.

    Args:
        params: Python SDF parameters

    Returns:
        Warp-compatible SDF parameters
    """
    warp_params = WarpSDFParams()
    warp_params.center = wp.vec3(*params.center)
    warp_params.cylinder_radius = params.cylinder_radius
    warp_params.cylinder_height = params.cylinder_height
    warp_params.cone_height = params.cone_height
    warp_params.cone_bottom_radius = params.cone_bottom_radius
    warp_params.vortex_finder_radius = params.vortex_finder_radius
    warp_params.vortex_finder_bottom_y = params.vortex_finder_bottom_y
    warp_params.dust_outlet_radius = params.dust_outlet_radius
    warp_params.dust_outlet_length = params.dust_outlet_length
    return warp_params


# =============================================================================
# SDF FIELD GENERATION
# =============================================================================

class SDFField:
    """
    Discretized SDF field on a regular grid.

    Useful for:
    - Fast SDF lookups (trilinear interpolation)
    - Visualization
    - Surface extraction (marching cubes)
    """

    def __init__(
        self,
        sdf: CycloneSDF,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        resolution: Union[int, Tuple[int, int, int]] = 50
    ):
        """
        Initialize SDF field.

        Args:
            sdf: CycloneSDF instance
            bounds_min: Minimum corner of domain
            bounds_max: Maximum corner of domain
            resolution: Grid resolution (int for uniform, tuple for per-axis)
        """
        self.sdf = sdf
        self.bounds_min = np.array(bounds_min, dtype=np.float32)
        self.bounds_max = np.array(bounds_max, dtype=np.float32)

        if isinstance(resolution, int):
            self.resolution = (resolution, resolution, resolution)
        else:
            self.resolution = resolution

        self._field = None
        self._computed = False

    def compute(self, device: str = "cpu") -> np.ndarray:
        """
        Compute SDF values on the grid.

        Args:
            device: Computation device ("cpu" or "cuda")

        Returns:
            3D array of SDF values with shape (nx, ny, nz)
        """
        nx, ny, nz = self.resolution

        # Create grid coordinates
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], nx)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], ny)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        if device == "cuda":
            self._field = self._compute_gpu(positions)
        else:
            self._field = self._compute_cpu(positions)

        self._field = self._field.reshape(nx, ny, nz)
        self._computed = True

        return self._field

    def _compute_cpu(self, positions: np.ndarray) -> np.ndarray:
        """Compute SDF on CPU."""
        return self.sdf.evaluate_batch(positions)

    def _compute_gpu(self, positions: np.ndarray) -> np.ndarray:
        """Compute SDF on GPU using Warp."""
        n = len(positions)
        device = "cuda"

        # Create Warp arrays
        pos_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device=device)
        sdf_wp = wp.zeros(n, dtype=float, device=device)

        # Create Warp parameters
        warp_params = create_warp_sdf_params(self.sdf.params)

        # Launch kernel
        wp.launch(
            kernel=compute_sdf_field,
            dim=n,
            inputs=[pos_wp, sdf_wp, warp_params],
            device=device
        )

        return sdf_wp.numpy()

    def interpolate(self, point: np.ndarray) -> float:
        """
        Interpolate SDF value at an arbitrary point using trilinear interpolation.

        Args:
            point: 3D query point

        Returns:
            Interpolated SDF value
        """
        if not self._computed:
            self.compute()

        # Normalize coordinates to grid space
        nx, ny, nz = self.resolution
        dx = (self.bounds_max - self.bounds_min) / (np.array(self.resolution) - 1)

        # Grid indices (continuous)
        idx = (point - self.bounds_min) / dx

        # Clamp to valid range
        idx = np.clip(idx, 0, np.array(self.resolution) - 1 - 1e-6)

        # Integer indices
        i0 = int(np.floor(idx[0]))
        j0 = int(np.floor(idx[1]))
        k0 = int(np.floor(idx[2]))

        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        k1 = min(k0 + 1, nz - 1)

        # Interpolation weights
        tx = idx[0] - i0
        ty = idx[1] - j0
        tz = idx[2] - k0

        # Trilinear interpolation
        c000 = self._field[i0, j0, k0]
        c001 = self._field[i0, j0, k1]
        c010 = self._field[i0, j1, k0]
        c011 = self._field[i0, j1, k1]
        c100 = self._field[i1, j0, k0]
        c101 = self._field[i1, j0, k1]
        c110 = self._field[i1, j1, k0]
        c111 = self._field[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz

    @property
    def field(self) -> np.ndarray:
        """Get computed SDF field."""
        if not self._computed:
            self.compute()
        return self._field

    def get_cell_size(self) -> np.ndarray:
        """Get grid cell size."""
        return (self.bounds_max - self.bounds_min) / (np.array(self.resolution) - 1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_cyclone_sdf(assembly) -> CycloneSDF:
    """
    Create a CycloneSDF from a CycloneAssembly.

    Args:
        assembly: CycloneAssembly instance

    Returns:
        CycloneSDF instance
    """
    params = CycloneSDFParams.from_assembly(assembly)
    return CycloneSDF(params)


def visualize_sdf_slice(
    sdf: CycloneSDF,
    y_slice: float,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 100,
    ax=None
):
    """
    Visualize SDF field as a 2D slice.

    Args:
        sdf: CycloneSDF instance
        y_slice: Y-coordinate of the slice
        x_range: (min, max) X range
        z_range: (min, max) Z range
        resolution: Grid resolution
        ax: Matplotlib axes (creates new if None)

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

    # Evaluate SDF
    sdf_values = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], y_slice, Z[i, j]])
            sdf_values[i, j] = sdf.evaluate(point)

    # Plot
    levels = np.linspace(-0.1, 0.1, 21)
    contour = ax.contourf(X * 1000, Z * 1000, sdf_values, levels=levels,
                          cmap='RdBu', extend='both')
    ax.contour(X * 1000, Z * 1000, sdf_values, levels=[0], colors='black',
               linewidths=2)

    plt.colorbar(contour, ax=ax, label='SDF (m)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title(f'SDF Field at Y = {y_slice * 1000:.1f} mm')
    ax.set_aspect('equal')

    return ax
