"""
Impact Kernel
=============

Warp kernel for hammer-particle collision detection and impact response.
Detects when particles collide with rotating hammers and computes
velocity changes and impact energies.

The hammer tips trace circular paths in the Y-Z plane as the rotor
rotates. Impact occurs when a particle is within the hammer sweep zone.

Physics:
    - Impact velocity depends on relative velocity (particle vs hammer tip)
    - Energy transfer follows coefficient of restitution model
    - Impact energy is tracked for breakage calculations
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


if WARP_AVAILABLE:
    @wp.kernel
    def impact_detection_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        sizes: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        impact_flags: wp.array(dtype=int),
        impact_energies: wp.array(dtype=float),
        rotor_theta: float,
        rotor_omega: float,
        hammer_tip_radius: float,
        hammer_width: float,
        hammer_rows: int,
        hammers_per_row: int,
        row_start_x: float,
        row_spacing: float,
        restitution: float,
        dt: float,
    ):
        """Detect and resolve hammer-particle impacts.

        For each particle, check if it's within the hammer sweep zone.
        If so, compute impact velocity and energy.
        """
        tid = wp.tid()

        pos = positions[tid]
        vel = velocities[tid]
        size = sizes[tid]
        mass = masses[tid]

        # Skip inactive particles
        if mass <= 0.0:
            impact_flags[tid] = 0
            impact_energies[tid] = 0.0
            return

        # Particle radial position in Y-Z plane
        r_particle = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        particle_angle = wp.atan2(pos[2], pos[1])

        # Check if particle is in hammer zone (radially)
        inner_radius = hammer_tip_radius - 0.03  # Inner sweep zone
        outer_radius = hammer_tip_radius + 0.02  # Outer sweep zone (clearance)

        if r_particle < inner_radius or r_particle > outer_radius:
            impact_flags[tid] = 0
            impact_energies[tid] = 0.0
            return

        # Check X position (is particle at a hammer row?)
        x = pos[0]
        hit_row = False
        for row in range(hammer_rows):
            row_x = row_start_x + float(row) * row_spacing
            if wp.abs(x - row_x) < hammer_width * 0.6:
                hit_row = True
                break

        if not hit_row:
            impact_flags[tid] = 0
            impact_energies[tid] = 0.0
            return

        # Check angular position (is a hammer near the particle?)
        # Hammers are evenly spaced angularly
        angular_spacing = 2.0 * 3.14159 / float(hammers_per_row)
        hit_hammer = False

        for h in range(hammers_per_row):
            hammer_angle = rotor_theta + float(h) * angular_spacing
            # Normalize angle
            while hammer_angle > 3.14159:
                hammer_angle -= 2.0 * 3.14159
            while hammer_angle < -3.14159:
                hammer_angle += 2.0 * 3.14159

            angle_diff = wp.abs(particle_angle - hammer_angle)
            if angle_diff > 3.14159:
                angle_diff = 2.0 * 3.14159 - angle_diff

            # Check if particle is within hammer angular extent
            hammer_angular_extent = 0.15  # ~8.6 degrees
            if angle_diff < hammer_angular_extent:
                hit_hammer = True
                break

        if not hit_hammer:
            impact_flags[tid] = 0
            impact_energies[tid] = 0.0
            return

        # --- Impact detected! ---
        impact_flags[tid] = 1

        # Hammer tip velocity (tangential)
        hammer_speed = rotor_omega * hammer_tip_radius
        hammer_vel = wp.vec3(
            0.0,
            -hammer_speed * wp.sin(particle_angle),
            hammer_speed * wp.cos(particle_angle),
        )

        # Relative velocity
        rel_vel = vel - hammer_vel
        rel_speed = wp.length(rel_vel)

        # Impact energy (kinetic energy in relative frame)
        impact_energy = 0.5 * mass * rel_speed * rel_speed
        impact_energies[tid] = impact_energy

        # Velocity update (simplified impact model)
        # New velocity reflects off hammer surface with restitution
        if rel_speed > 0.1:
            # Normal direction (radially outward from shaft)
            n = wp.vec3(0.0, pos[1] / r_particle, pos[2] / r_particle)
            # Velocity component normal to hammer surface
            v_n = wp.dot(rel_vel, n)
            # Reflect and apply restitution
            new_rel_vel = rel_vel - n * (v_n * (1.0 + restitution))
            # Add hammer velocity back
            new_vel = new_rel_vel + hammer_vel * 0.5  # Partial momentum transfer
            velocities[tid] = new_vel


def impact_detection_warp(
    positions: "wp.array",
    velocities: "wp.array",
    sizes: "wp.array",
    masses: "wp.array",
    impact_flags: "wp.array",
    impact_energies: "wp.array",
    rotor_theta: float,
    rotor_omega: float,
    hammer_tip_radius: float,
    hammer_width: float,
    hammer_rows: int,
    hammers_per_row: int,
    row_start_x: float,
    row_spacing: float,
    restitution: float = 0.3,
    dt: float = 0.001,
):
    """Launch impact detection kernel.

    Args:
        positions: Particle positions [n, 3]
        velocities: Particle velocities [n, 3]
        sizes: Particle sizes [n]
        masses: Particle masses [n]
        impact_flags: Output impact indicators [n]
        impact_energies: Output impact energies [n]
        rotor_theta: Current rotor angle [rad]
        rotor_omega: Rotor angular velocity [rad/s]
        hammer_tip_radius: Radius to hammer tip [m]
        hammer_width: Hammer width [m]
        hammer_rows: Number of hammer rows
        hammers_per_row: Hammers per row
        row_start_x: X position of first row
        row_spacing: Spacing between rows
        restitution: Coefficient of restitution
        dt: Timestep [s]
    """
    n = positions.shape[0]
    wp.launch(
        impact_detection_kernel,
        dim=n,
        inputs=[
            positions, velocities, sizes, masses, impact_flags, impact_energies,
            rotor_theta, rotor_omega, hammer_tip_radius, hammer_width,
            hammer_rows, hammers_per_row, row_start_x, row_spacing,
            restitution, dt,
        ],
    )


# NumPy fallback
def impact_detection_np(
    positions: np.ndarray,
    velocities: np.ndarray,
    sizes: np.ndarray,
    masses: np.ndarray,
    rotor_theta: float,
    rotor_omega: float,
    hammer_tip_radius: float,
    hammer_width: float,
    hammer_rows: int,
    hammers_per_row: int,
    row_start_x: float,
    row_spacing: float,
    restitution: float = 0.3,
    dt: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy implementation of impact detection.

    Returns:
        (impact_flags, impact_energies, new_velocities)
    """
    n = len(positions)
    impact_flags = np.zeros(n, dtype=np.int32)
    impact_energies = np.zeros(n, dtype=np.float32)
    new_velocities = velocities.copy()

    inner_radius = hammer_tip_radius - 0.03
    outer_radius = hammer_tip_radius + 0.02
    angular_spacing = 2.0 * math.pi / hammers_per_row
    hammer_angular_extent = 0.15

    for i in range(n):
        if masses[i] <= 0.0:
            continue

        pos = positions[i]
        vel = velocities[i]
        mass = masses[i]

        # Radial check
        r_particle = math.sqrt(pos[1] ** 2 + pos[2] ** 2)
        if r_particle < inner_radius or r_particle > outer_radius:
            continue

        particle_angle = math.atan2(pos[2], pos[1])

        # Row check
        hit_row = False
        for row in range(hammer_rows):
            row_x = row_start_x + row * row_spacing
            if abs(pos[0] - row_x) < hammer_width * 0.6:
                hit_row = True
                break

        if not hit_row:
            continue

        # Angular check
        hit_hammer = False
        for h in range(hammers_per_row):
            hammer_angle = rotor_theta + h * angular_spacing
            # Normalize
            while hammer_angle > math.pi:
                hammer_angle -= 2 * math.pi
            while hammer_angle < -math.pi:
                hammer_angle += 2 * math.pi

            angle_diff = abs(particle_angle - hammer_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            if angle_diff < hammer_angular_extent:
                hit_hammer = True
                break

        if not hit_hammer:
            continue

        # Impact!
        impact_flags[i] = 1

        # Hammer velocity
        hammer_speed = rotor_omega * hammer_tip_radius
        hammer_vel = np.array([
            0.0,
            -hammer_speed * math.sin(particle_angle),
            hammer_speed * math.cos(particle_angle),
        ])

        # Relative velocity
        rel_vel = vel - hammer_vel
        rel_speed = np.linalg.norm(rel_vel)

        # Impact energy
        impact_energies[i] = 0.5 * mass * rel_speed ** 2

        # Velocity update
        if rel_speed > 0.1:
            n_vec = np.array([0.0, pos[1] / r_particle, pos[2] / r_particle])
            v_n = np.dot(rel_vel, n_vec)
            new_rel_vel = rel_vel - n_vec * (v_n * (1.0 + restitution))
            new_velocities[i] = new_rel_vel + hammer_vel * 0.5

    return impact_flags, impact_energies, new_velocities
