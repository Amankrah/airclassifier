"""
Flow field representation for cyclone air classifier.

Provides analytical velocity profiles based on established cyclone
flow models. These profiles can be used for particle tracking without
full CFD simulation.

Key flow features in a cyclone:
1. Tangential velocity: Rankine vortex (forced + free vortex)
2. Axial velocity: Downward in outer region, upward in inner core
3. Radial velocity: Generally small, inward near walls
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import numpy as np
import warp as wp

from ..utils.constants import PI, AirProperties


@dataclass
class CycloneFlowParams:
    """Parameters for analytical cyclone flow field."""

    # Geometry (required)
    cylinder_radius: float      # [m] Cyclone body radius
    vortex_finder_radius: float # [m] Vortex finder radius
    cylinder_height: float      # [m] Height of cylindrical section
    cone_height: float          # [m] Height of conical section
    cone_bottom_radius: float   # [m] Radius at cone bottom

    # Flow conditions
    inlet_velocity: float       # [m/s] Inlet velocity
    inlet_width: float          # [m] Inlet width
    inlet_height: float         # [m] Inlet height

    # Fluid properties
    fluid_density: float = AirProperties.DENSITY           # [kg/m³]
    fluid_viscosity: float = AirProperties.DYNAMIC_VISCOSITY  # [Pa·s]

    # Vortex parameters (can be fitted to experiments)
    vortex_exponent: float = 0.7  # n in v_tan ~ r^(-n), typically 0.5-0.9
    core_radius_factor: float = 0.4  # Core radius as fraction of vf radius

    # Axial velocity parameters
    axial_profile_exponent: float = 2.0  # Shape of axial profile

    @property
    def core_radius(self) -> float:
        """Radius of forced vortex core."""
        return self.core_radius_factor * self.vortex_finder_radius

    @property
    def inlet_area(self) -> float:
        """Inlet cross-sectional area."""
        return self.inlet_width * self.inlet_height

    @property
    def volumetric_flow_rate(self) -> float:
        """Volumetric flow rate [m³/s]."""
        return self.inlet_velocity * self.inlet_area

    @property
    def max_tangential_velocity(self) -> float:
        """
        Estimated maximum tangential velocity (at core boundary).
        """
        # From momentum balance at inlet
        # v_tan_max ≈ inlet_velocity * (R / r_core)^n
        return self.inlet_velocity * (
            self.cylinder_radius / self.core_radius
        ) ** self.vortex_exponent


class CycloneFlowField:
    """
    Analytical flow field for cyclone air classifier.

    Implements the Rankine combined vortex model with modifications
    for cyclone geometry. Provides velocity at any point in the cyclone.

    Coordinate system:
    - Origin at top center of cylinder
    - Y-axis pointing downward (into cyclone)
    - Tangential direction: counterclockwise when viewed from above
    """

    def __init__(self, params: CycloneFlowParams):
        """
        Initialize flow field.

        Args:
            params: CycloneFlowParams defining the flow
        """
        self.params = params
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        """Precompute coefficients for velocity calculations."""
        p = self.params

        # Tangential velocity coefficient
        # At r = R (wall), v_tan ≈ v_inlet
        # v_tan = C * r^(-n) for r > r_core
        self._v_tan_coeff = p.inlet_velocity * p.cylinder_radius ** p.vortex_exponent

        # Core angular velocity (for forced vortex in core)
        # Continuity at r = r_core: omega * r_core = C * r_core^(-n)
        self._omega_core = self._v_tan_coeff / (p.core_radius ** (1 + p.vortex_exponent))

        # Axial velocity scales
        # Outer region: downward, magnitude ~ volumetric_flow / annular_area
        outer_area = PI * (p.cylinder_radius ** 2 - p.vortex_finder_radius ** 2)
        self._v_axial_outer = -p.volumetric_flow_rate / outer_area

        # Inner region: upward through vortex finder
        inner_area = PI * p.vortex_finder_radius ** 2
        self._v_axial_inner = p.volumetric_flow_rate / inner_area

    def get_local_radius(self, y: float) -> float:
        """
        Get the cyclone wall radius at a given Y position.

        Args:
            y: Y coordinate (positive = depth into cyclone from top)

        Returns:
            Local wall radius [m]
        """
        p = self.params

        if y < 0:
            return p.cylinder_radius
        elif y <= p.cylinder_height:
            return p.cylinder_radius
        elif y <= p.cylinder_height + p.cone_height:
            # In cone - linear interpolation
            cone_y = y - p.cylinder_height
            t = cone_y / p.cone_height
            return p.cylinder_radius * (1 - t) + p.cone_bottom_radius * t
        else:
            return p.cone_bottom_radius

    def velocity_at(self, position: np.ndarray) -> np.ndarray:
        """
        Get velocity vector at a given position.

        Args:
            position: 3D position [x, y, z] in meters
                     (y positive going into cyclone)

        Returns:
            Velocity vector [vx, vy, vz] in m/s
        """
        p = self.params

        x, y, z = position

        # Radial position
        r = np.sqrt(x ** 2 + z ** 2)

        # Handle axis singularity
        if r < 1e-10:
            # On axis - only axial velocity
            return np.array([0.0, self._v_axial_inner, 0.0])

        # Get local wall radius
        R_local = self.get_local_radius(y)

        # Unit vectors
        radial_unit = np.array([x / r, 0.0, z / r])
        # Tangential unit (counterclockwise when viewed from above, i.e., from +Y)
        tangent_unit = np.array([-z / r, 0.0, x / r])
        axial_unit = np.array([0.0, 1.0, 0.0])

        # Calculate velocity components
        v_tan = self._tangential_velocity(r, R_local)
        v_axial = self._axial_velocity(r, y, R_local)
        v_radial = self._radial_velocity(r, y, R_local)

        # Combine into velocity vector
        velocity = v_tan * tangent_unit + v_axial * axial_unit + v_radial * radial_unit

        return velocity

    def _tangential_velocity(self, r: float, R_local: float) -> float:
        """
        Calculate tangential velocity magnitude.

        Uses Rankine vortex model:
        - Forced vortex (solid body rotation) in core: v = omega * r
        - Free vortex outside core: v = C * r^(-n)
        """
        p = self.params
        r_core = p.core_radius

        # Scale core with local radius
        r_core_local = r_core * R_local / p.cylinder_radius

        if r <= r_core_local:
            # Forced vortex (solid body rotation)
            return self._omega_core * r
        else:
            # Free vortex with decay
            return self._v_tan_coeff * r ** (-p.vortex_exponent)

    def _axial_velocity(self, r: float, y: float, R_local: float) -> float:
        """
        Calculate axial velocity (positive = downward).

        Profile: downward in outer annulus, upward in inner core.
        """
        p = self.params
        r_vf = p.vortex_finder_radius

        # Determine axial velocity based on radial position
        # and whether we're above or below vortex finder

        # Above vortex finder bottom (in cylinder with VF)
        vf_bottom_y = p.vortex_finder_radius  # Approximate insertion depth

        if y < vf_bottom_y:
            # In region where vortex finder exists
            if r < r_vf * 0.9:  # Inside vortex finder region
                # Upward flow
                # Profile: parabolic with max at center
                r_norm = r / r_vf
                return self._v_axial_inner * (1.0 - r_norm ** 2)
            else:
                # Outside vortex finder - downward outer vortex
                r_norm = (r - r_vf) / (R_local - r_vf)
                return self._v_axial_outer * (1.0 - r_norm ** p.axial_profile_exponent)
        else:
            # Below vortex finder - full radial extent available for vortex
            # Inner upward flow expands, outer downward contracts
            r_transition = r_vf * (1.0 + 0.5 * (y - vf_bottom_y) / p.cylinder_height)
            r_transition = min(r_transition, R_local * 0.7)

            if r < r_transition:
                # Upward inner flow
                r_norm = r / r_transition
                return self._v_axial_inner * (1.0 - r_norm ** 2) * 0.5
            else:
                # Downward outer flow
                r_norm = (r - r_transition) / (R_local - r_transition + 1e-10)
                return self._v_axial_outer * (1.0 - r_norm ** 1.5)

    def _radial_velocity(self, r: float, y: float, R_local: float) -> float:
        """
        Calculate radial velocity (positive = outward).

        Generally small compared to tangential and axial.
        Inward near walls due to boundary layer.
        """
        p = self.params

        # Simple model: weak inward drift in outer region
        # to maintain mass balance
        if r > p.vortex_finder_radius:
            # Inward drift proportional to distance from axis
            r_norm = (r - p.vortex_finder_radius) / (R_local - p.vortex_finder_radius + 1e-10)
            return -0.1 * p.inlet_velocity * r_norm
        else:
            return 0.0

    def velocity_field_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate velocities for multiple positions.

        Args:
            positions: Array of shape (N, 3) with positions

        Returns:
            Array of shape (N, 3) with velocities
        """
        velocities = np.zeros_like(positions)
        for i in range(len(positions)):
            velocities[i] = self.velocity_at(positions[i])
        return velocities


# =============================================================================
# WARP FUNCTIONS AND KERNELS
# =============================================================================

@wp.struct
class WarpFlowParams:
    """Warp-compatible flow parameters."""
    cylinder_radius: float
    vortex_finder_radius: float
    cylinder_height: float
    cone_height: float
    cone_bottom_radius: float
    core_radius: float
    vortex_exponent: float
    v_tan_coeff: float
    omega_core: float
    v_axial_outer: float
    v_axial_inner: float
    inlet_velocity: float


@wp.func
def wp_get_local_radius(
    y: float,
    cylinder_radius: float,
    cylinder_height: float,
    cone_height: float,
    cone_bottom_radius: float
) -> float:
    """Get local wall radius at given Y position."""
    if y < 0.0:
        return cylinder_radius
    elif y <= cylinder_height:
        return cylinder_radius
    elif y <= cylinder_height + cone_height:
        cone_y = y - cylinder_height
        t = cone_y / cone_height
        return cylinder_radius * (1.0 - t) + cone_bottom_radius * t
    else:
        return cone_bottom_radius


@wp.func
def wp_tangential_velocity(
    r: float,
    r_core: float,
    omega_core: float,
    v_tan_coeff: float,
    vortex_exponent: float
) -> float:
    """Calculate tangential velocity magnitude."""
    if r <= r_core:
        return omega_core * r
    else:
        return v_tan_coeff * wp.pow(r, -vortex_exponent)


@wp.func
def wp_velocity_at(
    pos: wp.vec3,
    params: WarpFlowParams
) -> wp.vec3:
    """
    Calculate velocity at a given position.

    Args:
        pos: Position (x, y, z) where y is depth into cyclone
        params: Flow parameters

    Returns:
        Velocity vector
    """
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Radial position
    r = wp.sqrt(x * x + z * z)

    eps = 1.0e-10
    if r < eps:
        # On axis
        return wp.vec3(0.0, params.v_axial_inner, 0.0)

    # Local wall radius
    R_local = wp_get_local_radius(
        y, params.cylinder_radius, params.cylinder_height,
        params.cone_height, params.cone_bottom_radius
    )

    # Unit vectors
    radial_x = x / r
    radial_z = z / r
    tangent_x = -z / r
    tangent_z = x / r

    # Scale core radius with local geometry
    r_core_local = params.core_radius * R_local / params.cylinder_radius

    # Tangential velocity
    v_tan = wp_tangential_velocity(
        r, r_core_local, params.omega_core,
        params.v_tan_coeff, params.vortex_exponent
    )

    # Axial velocity (simplified)
    r_vf = params.vortex_finder_radius
    if r < r_vf * 0.9:
        r_norm = r / r_vf
        v_axial = params.v_axial_inner * (1.0 - r_norm * r_norm)
    else:
        r_norm = (r - r_vf) / (R_local - r_vf + eps)
        v_axial = params.v_axial_outer * (1.0 - r_norm * r_norm)

    # Radial velocity (small inward drift)
    if r > r_vf:
        r_norm = (r - r_vf) / (R_local - r_vf + eps)
        v_radial = -0.1 * params.inlet_velocity * r_norm
    else:
        v_radial = 0.0

    # Combine components
    vx = v_tan * tangent_x + v_radial * radial_x
    vy = v_axial
    vz = v_tan * tangent_z + v_radial * radial_z

    return wp.vec3(vx, vy, vz)


@wp.kernel
def compute_fluid_velocities(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    params: WarpFlowParams
):
    """
    Kernel to compute fluid velocities at particle positions.

    Args:
        positions: Particle positions
        velocities: Output fluid velocities
        params: Flow parameters
    """
    tid = wp.tid()
    velocities[tid] = wp_velocity_at(positions[tid], params)


def create_warp_flow_params(flow_field: CycloneFlowField) -> WarpFlowParams:
    """
    Convert CycloneFlowField to WarpFlowParams for GPU computation.

    Args:
        flow_field: CycloneFlowField instance

    Returns:
        WarpFlowParams instance
    """
    p = flow_field.params

    params = WarpFlowParams()
    params.cylinder_radius = p.cylinder_radius
    params.vortex_finder_radius = p.vortex_finder_radius
    params.cylinder_height = p.cylinder_height
    params.cone_height = p.cone_height
    params.cone_bottom_radius = p.cone_bottom_radius
    params.core_radius = p.core_radius
    params.vortex_exponent = p.vortex_exponent
    params.v_tan_coeff = flow_field._v_tan_coeff
    params.omega_core = flow_field._omega_core
    params.v_axial_outer = flow_field._v_axial_outer
    params.v_axial_inner = flow_field._v_axial_inner
    params.inlet_velocity = p.inlet_velocity

    return params
