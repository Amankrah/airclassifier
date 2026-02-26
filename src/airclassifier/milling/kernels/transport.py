"""
Transport Kernel
================

Warp kernel for particle advection in the hammer mill chamber.
Handles gravity, drag, centrifugal effects, and boundary constraints.

Particles move through the mill chamber driven by:
    - Gravity (downward)
    - Air drag (velocity-dependent)
    - Centrifugal throw from rotor (radial outward from shaft)
    - Chamber boundary constraints

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    wp = None


# Physical constants
GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.2  # kg/m^3
AIR_VISCOSITY = 1.8e-5  # Pa.s


if WARP_AVAILABLE:
    @wp.kernel
    def transport_step_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        sizes: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        residence_times: wp.array(dtype=float),
        chamber_radius: float,
        chamber_length: float,
        rotor_omega: float,
        dt: float,
        gravity: float,
        drag_coeff: float,
    ):
        """Advect particles in the mill chamber.

        Updates position and velocity based on forces.
        Applies chamber boundary constraints.
        """
        tid = wp.tid()

        pos = positions[tid]
        vel = velocities[tid]
        size = sizes[tid]
        mass = masses[tid]

        # Skip inactive particles (mass <= 0)
        if mass <= 0.0:
            return

        # --- Compute forces ---

        # 1. Gravity (negative Y direction)
        f_gravity = wp.vec3(0.0, -mass * gravity, 0.0)

        # 2. Air drag (opposes velocity)
        # F_drag = -0.5 * rho * Cd * A * |v|^2 * v_hat
        speed = wp.length(vel)
        if speed > 1e-6:
            # Cross-sectional area (sphere approximation)
            area = 3.14159 * size * size * 0.25
            drag_mag = 0.5 * 1.2 * drag_coeff * area * speed * speed
            f_drag = -vel * (drag_mag / speed)
        else:
            f_drag = wp.vec3(0.0, 0.0, 0.0)

        # 3. Centrifugal throw from rotor
        # Particles near the rotor get thrown outward
        # Radial distance from shaft (Y-Z plane)
        r_yz = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        if r_yz > 0.05:  # Only if not at shaft center
            # Radial unit vector
            r_hat = wp.vec3(0.0, pos[1] / r_yz, pos[2] / r_yz)
            # Centrifugal acceleration ~ omega^2 * r (scaled by proximity to rotor)
            proximity_factor = wp.max(0.0, 1.0 - r_yz / chamber_radius)
            centrifugal_accel = rotor_omega * rotor_omega * r_yz * proximity_factor * 0.1
            f_centrifugal = r_hat * (mass * centrifugal_accel)
        else:
            f_centrifugal = wp.vec3(0.0, 0.0, 0.0)

        # --- Total force and acceleration ---
        f_total = f_gravity + f_drag + f_centrifugal
        accel = f_total / mass

        # --- Update velocity (forward Euler) ---
        new_vel = vel + accel * dt

        # --- Update position ---
        new_pos = pos + new_vel * dt

        # --- Chamber boundary constraints ---
        # Cylindrical chamber along X axis
        x = new_pos[0]
        y = new_pos[1]
        z = new_pos[2]

        # X bounds (along rotor)
        x_min = 0.0
        x_max = chamber_length
        if x < x_min:
            x = x_min
            new_vel = wp.vec3(-new_vel[0] * 0.3, new_vel[1], new_vel[2])
        elif x > x_max:
            x = x_max
            new_vel = wp.vec3(-new_vel[0] * 0.3, new_vel[1], new_vel[2])

        # Radial bound (Y-Z plane)
        r = wp.sqrt(y * y + z * z)
        if r > chamber_radius:
            # Push back to boundary
            scale = chamber_radius / r
            y = y * scale
            z = z * scale
            # Reflect radial velocity
            if r > 1e-6:
                r_hat = wp.vec3(0.0, y / r, z / r)
                v_radial = wp.dot(new_vel, r_hat)
                if v_radial > 0.0:
                    new_vel = new_vel - r_hat * (v_radial * 1.3)  # Bounce

        new_pos = wp.vec3(x, y, z)

        # --- Update arrays ---
        positions[tid] = new_pos
        velocities[tid] = new_vel
        residence_times[tid] = residence_times[tid] + dt


def transport_step_warp(
    positions: "wp.array",
    velocities: "wp.array",
    sizes: "wp.array",
    masses: "wp.array",
    residence_times: "wp.array",
    chamber_radius: float,
    chamber_length: float,
    rotor_omega: float,
    dt: float,
    gravity: float = GRAVITY,
    drag_coeff: float = 0.44,
):
    """Launch the transport step kernel.

    Args:
        positions: Particle positions [n, 3]
        velocities: Particle velocities [n, 3]
        sizes: Particle characteristic sizes [n]
        masses: Particle masses [n]
        residence_times: Time each particle has been in mill [n]
        chamber_radius: Mill chamber radius [m]
        chamber_length: Mill chamber length [m]
        rotor_omega: Rotor angular velocity [rad/s]
        dt: Timestep [s]
        gravity: Gravitational acceleration [m/s^2]
        drag_coeff: Drag coefficient
    """
    n = positions.shape[0]
    wp.launch(
        transport_step_kernel,
        dim=n,
        inputs=[
            positions, velocities, sizes, masses, residence_times,
            chamber_radius, chamber_length, rotor_omega, dt, gravity, drag_coeff,
        ],
    )


# Safe bounds to prevent overflow/NaN in NumPy path (chamber typically ~0.2–0.4 m)
_MAX_POS_R = 10.0   # max radial distance [m] before clamping
_MAX_VEL = 500.0   # max velocity magnitude [m/s] before clamping


# NumPy fallback for CPU
def transport_step_np(
    positions: np.ndarray,
    velocities: np.ndarray,
    sizes: np.ndarray,
    masses: np.ndarray,
    residence_times: np.ndarray,
    chamber_radius: float,
    chamber_length: float,
    rotor_omega: float,
    dt: float,
    gravity: float = GRAVITY,
    drag_coeff: float = 0.44,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized NumPy implementation of transport step.

    Returns:
        (new_positions, new_velocities, new_residence_times)
    """
    n = len(positions)
    if n == 0:
        return positions.copy(), velocities.copy(), residence_times.copy()

    # Sanitize inputs to avoid overflow/NaN (can occur from bad state or GPU sync)
    positions = np.nan_to_num(positions, nan=0.0, posinf=_MAX_POS_R, neginf=-_MAX_POS_R)
    velocities = np.nan_to_num(velocities, nan=0.0, posinf=_MAX_VEL, neginf=-_MAX_VEL)
    positions = np.clip(positions, [-_MAX_POS_R, -_MAX_POS_R, -_MAX_POS_R], [_MAX_POS_R, _MAX_POS_R, _MAX_POS_R])
    vel_norm = np.linalg.norm(velocities, axis=1)
    vel_clip = vel_norm > _MAX_VEL
    if vel_clip.any():
        scale = _MAX_VEL / np.where(vel_clip, vel_norm, 1.0)
        velocities = np.where(vel_clip[:, None], velocities * scale[:, None], velocities)

    active = masses > 0.0

    # --- Gravity (Y-component only) ---
    f_gravity = np.zeros_like(positions)
    f_gravity[active, 1] = -masses[active] * gravity

    # --- Air drag (safe speed to avoid div-by-zero and overflow) ---
    speed = np.linalg.norm(velocities, axis=1)
    speed = np.clip(speed, 1e-6, _MAX_VEL)
    area = np.pi * sizes * sizes * 0.25
    drag_mask = active & (speed > 1e-6)
    drag_mag = np.zeros(n, dtype=positions.dtype)
    drag_mag[drag_mask] = (
        0.5 * AIR_DENSITY * drag_coeff * area[drag_mask] * speed[drag_mask] ** 2
    )
    f_drag = np.zeros_like(velocities)
    inv_speed = np.where(drag_mask, 1.0 / speed, 1.0)
    f_drag[drag_mask] = (
        -velocities[drag_mask] * (drag_mag[drag_mask] * inv_speed[drag_mask])[:, None]
    )

    # --- Centrifugal throw (safe r_yz to avoid overflow in sqrt) ---
    yz_sq = positions[:, 1] ** 2 + positions[:, 2] ** 2
    yz_sq = np.minimum(yz_sq, _MAX_POS_R * _MAX_POS_R)
    r_yz = np.sqrt(yz_sq)
    cent_mask = active & (r_yz > 0.05)
    r_hat = np.zeros_like(positions)
    r_yz_safe = np.where(r_yz > 1e-10, r_yz, 1e-10)
    r_hat[cent_mask, 1] = positions[cent_mask, 1] / r_yz_safe[cent_mask]
    r_hat[cent_mask, 2] = positions[cent_mask, 2] / r_yz_safe[cent_mask]
    proximity = np.maximum(0.0, 1.0 - r_yz / np.maximum(chamber_radius, 1e-10))
    cent_accel = rotor_omega ** 2 * r_yz * proximity * 0.1
    f_cent = r_hat * (masses * cent_accel)[:, None]

    # --- Total force → acceleration → Euler integration ---
    accel = np.zeros_like(positions)
    accel[active] = (
        (f_gravity[active] + f_drag[active] + f_cent[active])
        / masses[active, None]
    )
    new_vel = velocities + accel * dt
    new_pos = positions + new_vel * dt

    # Sanitize after integration to prevent overflow propagation
    new_pos = np.nan_to_num(new_pos, nan=0.0, posinf=chamber_length, neginf=0.0)
    new_vel = np.nan_to_num(new_vel, nan=0.0, posinf=_MAX_VEL, neginf=-_MAX_VEL)
    new_pos = np.clip(new_pos, [-_MAX_POS_R, -_MAX_POS_R, -_MAX_POS_R], [_MAX_POS_R, _MAX_POS_R, _MAX_POS_R])

    # --- X boundary clamp ---
    x_lo = new_pos[:, 0] < 0
    x_hi = new_pos[:, 0] > chamber_length
    new_pos[x_lo, 0] = 0.0
    new_vel[x_lo, 0] *= -0.3
    new_pos[x_hi, 0] = chamber_length
    new_vel[x_hi, 0] *= -0.3

    # --- Radial boundary (safe r to avoid overflow in sqrt) ---
    yz_sq_new = new_pos[:, 1] ** 2 + new_pos[:, 2] ** 2
    yz_sq_new = np.minimum(yz_sq_new, _MAX_POS_R * _MAX_POS_R)
    r = np.sqrt(yz_sq_new)
    outside = r > chamber_radius
    safe_r = np.maximum(r, 1e-10)
    scale = np.where(outside, chamber_radius / safe_r, 1.0)
    new_pos[:, 1] *= scale
    new_pos[:, 2] *= scale

    # Reflect radial velocity for particles that were outside
    # Denom: after scaling, radial distance is chamber_radius for outside particles, so use it directly
    reflect = outside & (safe_r > 1e-6)
    if reflect.any():
        r_hat_r = np.zeros_like(new_pos)
        denom = np.where(reflect, np.maximum(safe_r * scale, 1e-10), 1.0)
        r_hat_r[reflect, 1] = new_pos[reflect, 1] / denom[reflect]
        r_hat_r[reflect, 2] = new_pos[reflect, 2] / denom[reflect]
        # Dot product of velocity with radial unit vector
        v_radial = np.sum(new_vel[reflect] * r_hat_r[reflect], axis=1)
        # Only reflect outward-moving particles
        outward = v_radial > 0
        idx = np.where(reflect)[0][outward]
        if len(idx) > 0:
            v_rad_out = v_radial[outward]
            new_vel[idx] -= r_hat_r[idx] * (v_rad_out * 1.3)[:, None]

    # --- Residence time ---
    new_res = residence_times.copy()
    new_res[active] += dt

    return new_pos, new_vel, new_res
