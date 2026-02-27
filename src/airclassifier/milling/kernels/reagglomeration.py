"""
Reagglomeration Kernel
======================

Fine-particle agglomeration model for hammer milling.

Below ~50 µm, van der Waals forces, electrostatic charging from repeated
impacts, and moisture bridges cause fine particles to stick together.
This effectively raises the measured D50 and is a key mechanism limiting
the grinding fineness of single-pass hammer mills.

Implementation: stochastic pair-merge.  Each step, eligible fine particles
are randomly paired and merged with a probability that depends on size
(smaller → stickier) and moisture.  Mass is exactly conserved.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def reagglomeration_step_np(
    sizes: np.ndarray,
    masses: np.ndarray,
    threshold_m: float = 50e-6,
    rate: float = 0.02,
    max_merges: int = 50,
    moisture_wb: float = 0.12,
    moisture_sensitivity: float = 2.0,
    product_temperature_c: float = 25.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Apply stochastic pair-merge reagglomeration to fine particles.

    Particles below *threshold_m* are randomly paired.  For each pair the
    agglomeration probability is::

        P = rate × (d_ref / max(d_i, d_j))^2 × moisture_factor × temp_factor

    If accepted, the two particles merge: the first gets
    ``d_new = (d_i^3 + d_j^3)^{1/3}`` and the combined mass; the second
    is deactivated (mass = 0, size = 0).

    Args:
        sizes:  Particle sizes [n] (modified in place).
        masses: Particle masses [n] (modified in place).
        threshold_m: Only particles below this size can agglomerate [m].
        rate: Base agglomeration probability per eligible pair per step.
        max_merges: Cap on merges per step (limits compute cost).
        moisture_wb: Product moisture (wet basis).  Higher → stickier.
        moisture_sensitivity: Exponent amplifying moisture effect.
        product_temperature_c: Product temperature [°C].
            Above 40 °C starch surfaces become stickier.
        rng: NumPy random generator.

    Returns:
        (sizes, masses, num_merges)  — arrays are modified in place.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find eligible fine particles (alive and below threshold)
    eligible = (masses > 0.0) & (sizes > 0.0) & (sizes < threshold_m)
    elig_idx = np.where(eligible)[0]
    n_elig = len(elig_idx)

    if n_elig < 2:
        return sizes, masses, 0

    # Shuffle and pair up: (0,1), (2,3), (4,5), ...
    rng.shuffle(elig_idx)
    n_pairs = min(n_elig // 2, max_merges)

    # Reference size for probability scaling (half the threshold)
    d_ref = threshold_m * 0.5

    # Moisture factor: higher moisture → more agglomeration
    moisture_factor = 1.0 + moisture_sensitivity * max(0.0, moisture_wb - 0.08)

    # Temperature factor: above 40 °C starch surfaces become stickier
    temp_factor = 1.0 + 0.02 * max(0.0, product_temperature_c - 40.0)

    num_merges = 0
    for k in range(n_pairs):
        i = elig_idx[2 * k]
        j = elig_idx[2 * k + 1]

        d_i = sizes[i]
        d_j = sizes[j]
        d_max = max(d_i, d_j)

        # Probability: smaller particles stick more readily
        prob = rate * (d_ref / d_max) ** 2 * moisture_factor * temp_factor
        prob = min(prob, 1.0)

        if rng.random() < prob:
            # Merge: conserve mass, compute merged diameter
            m_new = masses[i] + masses[j]
            d_new = (d_i ** 3 + d_j ** 3) ** (1.0 / 3.0)

            sizes[i] = d_new
            masses[i] = m_new

            # Deactivate the second particle
            sizes[j] = 0.0
            masses[j] = 0.0

            num_merges += 1

    return sizes, masses, num_merges
