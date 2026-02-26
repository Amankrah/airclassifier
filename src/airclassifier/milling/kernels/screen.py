"""
Screen Kernel
=============

Warp kernel for screen classification in the hammer mill.
Determines which particles pass through the screen aperture
and which are retained for further breakage.

The screen is a curved perforated surface at the bottom of the
mill chamber. Particles smaller than the aperture can pass;
larger particles are retained.

Physics:
    - Passage depends on particle size vs aperture size
    - Position must be in screen zone (radial and angular)
    - Passage is probabilistic (open area, orientation, velocity)
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
    def screen_passage_kernel(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        sizes: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        passage_flags: wp.array(dtype=int),
        rand_states: wp.array(dtype=wp.uint32),
        screen_radius: float,
        screen_start_angle: float,
        screen_arc_angle: float,
        screen_x_start: float,
        screen_x_end: float,
        aperture: float,
        open_area: float,
        passage_factor: float,
        size_ratio_threshold: float,
    ):
        """Test particles for screen passage.

        For each particle in the screen zone, determine if it passes
        through based on size, aperture, and stochastic factors.
        size_ratio_threshold: below d/aperture passage is max; above, quadratic taper (retain coarse for breakage).
        """
        tid = wp.tid()

        pos = positions[tid]
        vel = velocities[tid]
        size = sizes[tid]
        mass = masses[tid]

        # Default: no passage
        passage_flags[tid] = 0

        # Skip inactive particles
        if mass <= 0.0:
            return

        # Check if particle is in screen zone

        # X bounds (along rotor)
        if pos[0] < screen_x_start or pos[0] > screen_x_end:
            return

        # Radial position (screen is at fixed radius)
        r = wp.sqrt(pos[1] * pos[1] + pos[2] * pos[2])
        screen_tolerance = 0.03  # 3cm tolerance around screen radius
        if wp.abs(r - screen_radius) > screen_tolerance:
            return

        # Angular position (screen covers part of circumference)
        angle = wp.atan2(pos[2], pos[1])
        # Normalize to [0, 2*pi]
        if angle < 0.0:
            angle += 2.0 * 3.14159

        screen_end_angle = screen_start_angle + screen_arc_angle
        if angle < screen_start_angle or angle > screen_end_angle:
            return

        # --- Particle is in screen zone ---

        # Size check: can particle fit through aperture?
        if size > aperture:
            return  # Too large to pass

        # Passage probability (match NumPy: below threshold = full; above = quadratic taper)
        size_ratio = size / aperture
        span = 1.0 - size_ratio_threshold
        if span < 0.01:
            span = 0.01
        if size_ratio <= size_ratio_threshold:
            size_prob = 1.0
        else:
            t = (size_ratio - size_ratio_threshold) / span
            size_prob = 1.0 - t * t
            if size_prob < 0.0:
                size_prob = 0.0

        # Velocity factor (very fast particles may not have time to pass)
        speed = wp.length(vel)
        if speed > 5.0:  # m/s threshold
            vel_prob = 5.0 / speed
        else:
            vel_prob = 1.0

        # Total passage probability
        passage_prob = open_area * size_prob * vel_prob * passage_factor
        passage_prob = wp.clamp(passage_prob, 0.0, 1.0)

        # Stochastic passage test
        state = rand_states[tid]
        state = state * wp.uint32(1103515245) + wp.uint32(12345)
        rand_states[tid] = state
        rand_val = float(state & wp.uint32(0x7FFFFFFF)) / float(0x7FFFFFFF)

        if rand_val < passage_prob:
            passage_flags[tid] = 1


def screen_passage_warp(
    positions: "wp.array",
    velocities: "wp.array",
    sizes: "wp.array",
    masses: "wp.array",
    passage_flags: "wp.array",
    rand_states: "wp.array",
    screen_radius: float,
    screen_start_angle: float,
    screen_arc_angle: float,
    screen_x_start: float,
    screen_x_end: float,
    aperture: float,
    open_area: float = 0.4,
    passage_factor: float = 1.0,
    size_ratio_threshold: float = 0.06,
):
    """Launch screen passage kernel.

    Args:
        size_ratio_threshold: Below d/aperture passage is max; above, quadratic taper (retain coarse for breakage).
    """
    n = positions.shape[0]
    wp.launch(
        screen_passage_kernel,
        dim=n,
        inputs=[
            positions, velocities, sizes, masses, passage_flags, rand_states,
            screen_radius, screen_start_angle, screen_arc_angle,
            screen_x_start, screen_x_end, aperture, open_area, passage_factor,
            size_ratio_threshold,
        ],
    )


# NumPy fallback
def screen_passage_np(
    positions: np.ndarray,
    velocities: np.ndarray,
    sizes: np.ndarray,
    masses: np.ndarray,
    screen_radius: float,
    screen_start_angle: float,
    screen_arc_angle: float,
    screen_x_start: float,
    screen_x_end: float,
    aperture: float,
    open_area: float = 0.4,
    passage_factor: float = 1.0,
    size_ratio_threshold: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Vectorized NumPy implementation of screen passage test.

    Returns:
        passage_flags: Array of passage indicators [n]
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(positions)
    passage_flags = np.zeros(n, dtype=np.int32)
    if n == 0:
        return passage_flags

    screen_tolerance = 0.03
    screen_end_angle = screen_start_angle + screen_arc_angle

    active = masses > 0.0

    # X bounds
    x_ok = (positions[:, 0] >= screen_x_start) & (positions[:, 0] <= screen_x_end)

    # Radial check
    r = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    r_ok = np.abs(r - screen_radius) <= screen_tolerance

    # Angular check
    angle = np.arctan2(positions[:, 2], positions[:, 1])
    angle = np.where(angle < 0, angle + 2.0 * math.pi, angle)
    a_ok = (angle >= screen_start_angle) & (angle <= screen_end_angle)

    # Size check
    size_ok = sizes <= aperture

    # Combined: particle is in screen zone and small enough
    in_zone = active & x_ok & r_ok & a_ok & size_ok
    n_candidates = int(in_zone.sum())
    if n_candidates == 0:
        return passage_flags

    # Passage probability (only for candidates). Below threshold = full prob; above = taper (retain coarse for breakage).
    size_ratio = sizes[in_zone] / aperture
    span = max(0.2, 1.0 - size_ratio_threshold)
    size_prob = np.where(
        size_ratio <= size_ratio_threshold,
        1.0,
        np.maximum(0.0, 1.0 - ((size_ratio - size_ratio_threshold) / span) ** 2),
    )

    speed = np.linalg.norm(velocities[in_zone], axis=1)
    vel_prob = np.minimum(1.0, 5.0 / np.maximum(speed, 0.1))

    prob = np.clip(open_area * size_prob * vel_prob * passage_factor, 0.0, 1.0)

    # Stochastic test
    passed = rng.random(n_candidates) < prob
    candidate_idx = np.where(in_zone)[0]
    passage_flags[candidate_idx[passed]] = 1

    return passage_flags


def apply_screen_discharge(
    positions: np.ndarray,
    velocities: np.ndarray,
    sizes: np.ndarray,
    masses: np.ndarray,
    passage_flags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove passed particles from active set and return discharged.

    Args:
        positions: All particle positions
        velocities: All particle velocities
        sizes: All particle sizes
        masses: All particle masses
        passage_flags: Passage indicators (1 = passed)

    Returns:
        (retained_pos, retained_vel, retained_sizes, retained_masses, discharged_sizes)
    """
    passed_mask = passage_flags == 1
    retained_mask = ~passed_mask

    retained_pos = positions[retained_mask]
    retained_vel = velocities[retained_mask]
    retained_sizes = sizes[retained_mask]
    retained_masses = masses[retained_mask]

    discharged_sizes = sizes[passed_mask]

    return retained_pos, retained_vel, retained_sizes, retained_masses, discharged_sizes
