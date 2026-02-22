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
    """NumPy implementation of transport step.

    Returns:
        (new_positions, new_velocities, new_residence_times)
    """
    n = len(positions)
    new_pos = positions.copy()
    new_vel = velocities.copy()
    new_res = residence_times.copy()

    for i in range(n):
        if masses[i] <= 0.0:
            continue

        pos = positions[i]
        vel = velocities[i]
        mass = masses[i]
        size = sizes[i]

        # Gravity
        f_gravity = np.array([0.0, -mass * gravity, 0.0])

        # Air drag
        speed = np.linalg.norm(vel)
        if speed > 1e-6:
            area = math.pi * size * size * 0.25
            drag_mag = 0.5 * AIR_DENSITY * drag_coeff * area * speed * speed
            f_drag = -vel * (drag_mag / speed)
        else:
            f_drag = np.zeros(3)

        # Centrifugal throw
        r_yz = math.sqrt(pos[1] ** 2 + pos[2] ** 2)
        if r_yz > 0.05:
            r_hat = np.array([0.0, pos[1] / r_yz, pos[2] / r_yz])
            proximity_factor = max(0.0, 1.0 - r_yz / chamber_radius)
            centrifugal_accel = rotor_omega ** 2 * r_yz * proximity_factor * 0.1
            f_centrifugal = r_hat * (mass * centrifugal_accel)
        else:
            f_centrifugal = np.zeros(3)

        # Total force and acceleration
        f_total = f_gravity + f_drag + f_centrifugal
        accel = f_total / mass

        # Update
        new_vel[i] = vel + accel * dt
        new_pos[i] = pos + new_vel[i] * dt

        # Boundaries
        x, y, z = new_pos[i]
        vx, vy, vz = new_vel[i]

        # X bounds
        if x < 0:
            x = 0
            vx = -vx * 0.3
        elif x > chamber_length:
            x = chamber_length
            vx = -vx * 0.3

        # Radial bound
        r = math.sqrt(y ** 2 + z ** 2)
        if r > chamber_radius:
            scale = chamber_radius / r
            y *= scale
            z *= scale
            # Reflect
            if r > 1e-6:
                r_hat = np.array([0.0, y / r, z / r])
                v = np.array([vx, vy, vz])
                v_radial = np.dot(v, r_hat)
                if v_radial > 0:
                    v = v - r_hat * (v_radial * 1.3)
                    vx, vy, vz = v

        new_pos[i] = [x, y, z]
        new_vel[i] = [vx, vy, vz]
        new_res[i] += dt

    return new_pos, new_vel, new_res
