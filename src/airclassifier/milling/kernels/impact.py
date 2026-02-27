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
    - Size-dependent impact efficiency: fine particles follow the airflow
      around the hammer (air entrainment / cushioning) and receive reduced
      effective impact energy.  η = min(1, (d / d_crit)^n).
"""

from __future__ import annotations

import math
from typing import Tuple

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
        efficiency_d_crit: float,
        efficiency_exponent: float,
    ):
        """Detect and resolve hammer-particle impacts.

        For each particle, check if it's within the hammer sweep zone.
        If so, compute impact velocity and energy, scaled by size-dependent
        impact efficiency (fine particles are entrained in air, not impacted).
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

        # --- Size-dependent impact efficiency ---
        # Fine particles follow air around the hammer: η = min(1, (d/d_crit)^n)
        # This models air entrainment, cushioning, and reduced collision efficiency.
        eta = 1.0
        if efficiency_d_crit > 0.0 and size < efficiency_d_crit:
            ratio = size / efficiency_d_crit
            eta = wp.pow(ratio, efficiency_exponent)

        # Impact energy scaled by efficiency
        impact_energy = 0.5 * mass * rel_speed * rel_speed * eta
        impact_energies[tid] = impact_energy

        # Velocity update (simplified impact model)
        # Deflection is also reduced for fine particles (they pass through more)
        if rel_speed > 0.1:
            # Normal direction (radially outward from shaft)
            n = wp.vec3(0.0, pos[1] / r_particle, pos[2] / r_particle)
            # Velocity component normal to hammer surface
            v_n = wp.dot(rel_vel, n)
            # Reflect and apply restitution, scaled by efficiency
            new_rel_vel = rel_vel - n * (v_n * (1.0 + restitution) * eta)
            # Add hammer velocity back (partial momentum transfer)
            new_vel = new_rel_vel + hammer_vel * (0.5 * eta)
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
    efficiency_d_crit: float = 80e-6,
    efficiency_exponent: float = 2.0,
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
        efficiency_d_crit: Size below which impact efficiency drops [m]
        efficiency_exponent: Exponent for efficiency taper
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
            efficiency_d_crit, efficiency_exponent,
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
    efficiency_d_crit: float = 80e-6,
    efficiency_exponent: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized NumPy implementation of impact detection.

    Returns:
        (impact_flags, impact_energies, new_velocities)
    """
    n = len(positions)
    impact_flags = np.zeros(n, dtype=np.int32)
    impact_energies = np.zeros(n, dtype=np.float32)
    new_velocities = velocities.copy()

    if n == 0:
        return impact_flags, impact_energies, new_velocities

    inner_radius = hammer_tip_radius - 0.03
    outer_radius = hammer_tip_radius + 0.02
    angular_spacing = 2.0 * math.pi / hammers_per_row
    hammer_angular_extent = 0.15

    # --- Radial filter ---
    r_particle = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    radial_ok = (
        (masses > 0.0)
        & (r_particle >= inner_radius)
        & (r_particle <= outer_radius)
    )

    # --- Row check (vectorized against all hammer rows) ---
    row_positions = row_start_x + np.arange(hammer_rows) * row_spacing  # [R]
    x_dists = np.abs(positions[:, 0, None] - row_positions[None, :])    # [N, R]
    row_ok = np.any(x_dists < hammer_width * 0.6, axis=1)              # [N]

    # --- Angular check (vectorized against all hammers) ---
    particle_angle = np.arctan2(positions[:, 2], positions[:, 1])       # [N]
    hammer_angles = rotor_theta + np.arange(hammers_per_row) * angular_spacing  # [H]
    # Normalize to [-pi, pi]
    hammer_angles = (hammer_angles + math.pi) % (2.0 * math.pi) - math.pi
    angle_diffs = np.abs(particle_angle[:, None] - hammer_angles[None, :])  # [N, H]
    angle_diffs = np.minimum(angle_diffs, 2.0 * math.pi - angle_diffs)
    hammer_ok = np.any(angle_diffs < hammer_angular_extent, axis=1)    # [N]

    # --- Combined hit mask ---
    hit = radial_ok & row_ok & hammer_ok
    if not hit.any():
        return impact_flags, impact_energies, new_velocities

    impact_flags[hit] = 1

    # --- Compute impact physics for hit particles ---
    h_idx = np.where(hit)[0]
    h_pos = positions[h_idx]
    h_vel = velocities[h_idx]
    h_mass = masses[h_idx]
    h_sizes = sizes[h_idx]
    h_r = r_particle[h_idx]
    h_angle = particle_angle[h_idx]

    # Hammer velocity (tangential at particle angle)
    hammer_speed = rotor_omega * hammer_tip_radius
    h_hammer_vel = np.zeros_like(h_pos)
    h_hammer_vel[:, 1] = -hammer_speed * np.sin(h_angle)
    h_hammer_vel[:, 2] = hammer_speed * np.cos(h_angle)

    # Relative velocity and raw kinetic energy
    rel_vel = h_vel - h_hammer_vel
    rel_speed = np.linalg.norm(rel_vel, axis=1)

    # --- Size-dependent impact efficiency ---
    # Fine particles follow air around the hammer: η = min(1, (d/d_crit)^n)
    eta = np.ones(len(h_idx), dtype=np.float64)
    if efficiency_d_crit > 0.0:
        below = h_sizes < efficiency_d_crit
        if below.any():
            ratio = h_sizes[below] / efficiency_d_crit
            eta[below] = np.power(ratio, efficiency_exponent)

    # Impact energy scaled by efficiency
    impact_energies[h_idx] = (0.5 * h_mass * rel_speed ** 2 * eta).astype(np.float32)

    # Velocity update for particles with rel_speed > 0.1
    fast = rel_speed > 0.1
    if fast.any():
        fi = h_idx[fast]
        f_pos = h_pos[fast]
        f_r = h_r[fast]
        f_rel = rel_vel[fast]
        f_hvel = h_hammer_vel[fast]
        f_eta = eta[fast]

        # Normal direction (radially outward from shaft)
        n_vec = np.zeros_like(f_pos)
        n_vec[:, 1] = f_pos[:, 1] / f_r
        n_vec[:, 2] = f_pos[:, 2] / f_r

        # Radial component of relative velocity
        v_n = np.sum(f_rel * n_vec, axis=1)

        # Reflect and apply restitution, scaled by efficiency
        new_rel = f_rel - n_vec * (v_n * (1.0 + restitution) * f_eta)[:, None]
        new_velocities[fi] = new_rel + f_hvel * (0.5 * f_eta)[:, None]

    return impact_flags, impact_energies, new_velocities
