"""
Breakage Kernel
===============

Hammer-mill impact breakage: size reduction driven by hammer–particle collisions.

Physics (high-fidelity digital twin):
    - Selection S(d, E): probability of breakage per impact. E comes from the
      impact kernel (hammer tip speed, particle mass, restitution). Larger
      particles (d) break more readily (size exponent alpha).
    - Breakage B(d_daughter | d_parent): Gaudin–Schuhmann distribution for
      single-impact comminution; gamma controls daughter size (finer vs coarser).
    - Only particles that received an impact in the impact step are candidates;
      breakage is applied in-place, so kinetics are fully coupled to hammer milling.

This kernel operates on:
    1. Lagrangian particles (update individual particle sizes)
    2. PSD bins (population balance on size classes)

Model:
    - Selection: S(d, E) = k * (d/d_ref)^alpha * f(E); min_energy threshold.
    - Breakage: B(d_daughter | d_parent) ~ (d_daughter/d_parent)^gamma
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
    def breakage_kernel(
        sizes: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        impact_flags: wp.array(dtype=int),
        impact_energies: wp.array(dtype=float),
        break_flags: wp.array(dtype=int),
        rand_states: wp.array(dtype=wp.uint32),
        selection_k: float,
        selection_alpha: float,
        reference_size: float,
        min_energy: float,
        energy_scale: float,
        breakage_gamma: float,
        min_size: float,
        # Size-dependent regime parameters
        coarse_threshold: float,
        fine_threshold: float,
        gamma_coarse: float,
        gamma_fine: float,
        clamp_lo_coarse: float,
        clamp_hi_coarse: float,
        clamp_lo_medium: float,
        clamp_hi_medium: float,
        clamp_lo_fine: float,
        clamp_hi_fine: float,
    ):
        """Apply breakage to impacted particles.

        For each particle with an impact, determine if breakage occurs
        (stochastic selection) and if so, reduce the particle size.
        Uses size-dependent breakage regimes (coarse/medium/fine).
        """
        tid = wp.tid()

        # Skip if no impact
        if impact_flags[tid] == 0:
            break_flags[tid] = 0
            return

        size = sizes[tid]
        mass = masses[tid]
        energy = impact_energies[tid]

        # Skip inactive or too-small particles
        if mass <= 0.0 or size < min_size:
            break_flags[tid] = 0
            return

        # --- Selection probability ---
        # S = k * (d / d_ref)^alpha * energy_factor
        size_factor = wp.pow(size / reference_size, selection_alpha)
        energy_factor = wp.min(1.0, energy / min_energy * energy_scale)
        selection_prob = selection_k * size_factor * energy_factor
        selection_prob = wp.clamp(selection_prob, 0.0, 1.0)

        # Stochastic selection (simple LCG random)
        state = rand_states[tid]
        state = state * wp.uint32(1103515245) + wp.uint32(12345)
        rand_states[tid] = state
        rand_val = float(state & wp.uint32(0x7FFFFFFF)) / float(0x7FFFFFFF)

        if rand_val > selection_prob:
            break_flags[tid] = 0
            return

        # --- Breakage occurs! ---
        break_flags[tid] = 1

        # Select regime based on particle size
        gamma = breakage_gamma  # default: medium
        clamp_lo = clamp_lo_medium
        clamp_hi = clamp_hi_medium
        if size > coarse_threshold:
            gamma = gamma_coarse
            clamp_lo = clamp_lo_coarse
            clamp_hi = clamp_hi_coarse
        elif size < fine_threshold:
            gamma = gamma_fine
            clamp_lo = clamp_lo_fine
            clamp_hi = clamp_hi_fine

        # Mean daughter size ~ d_parent * (gamma / (gamma + 1))
        reduction_factor = gamma / (gamma + 1.0)

        # Add some randomness to reduction
        state = state * wp.uint32(1103515245) + wp.uint32(12345)
        rand_states[tid] = state
        rand_factor = 0.5 + float(state & wp.uint32(0x7FFFFFFF)) / float(0x7FFFFFFF)
        reduction_factor = reduction_factor * rand_factor

        # Size-dependent clamp
        new_size = size * wp.clamp(reduction_factor, clamp_lo, clamp_hi)
        new_size = wp.max(new_size, min_size)

        # Update size
        sizes[tid] = new_size

        # Update mass (proportional to volume ~ d^3)
        size_ratio = new_size / size
        new_mass = mass * size_ratio * size_ratio * size_ratio
        masses[tid] = new_mass


def breakage_step_warp(
    sizes: "wp.array",
    masses: "wp.array",
    impact_flags: "wp.array",
    impact_energies: "wp.array",
    break_flags: "wp.array",
    rand_states: "wp.array",
    selection_k: float = 0.15,
    selection_alpha: float = 1.2,
    reference_size: float = 0.001,
    min_energy: float = 0.001,
    energy_scale: float = 5.0,
    breakage_gamma: float = 0.8,
    min_size: float = 1e-5,
    # Size-dependent regime parameters
    coarse_threshold: float = 1.0e-3,
    fine_threshold: float = 1.0e-4,
    gamma_coarse: float = 1.2,
    gamma_fine: float = 0.35,
    clamp_lo_coarse: float = 0.40,
    clamp_hi_coarse: float = 0.70,
    clamp_lo_medium: float = 0.20,
    clamp_hi_medium: float = 0.55,
    clamp_lo_fine: float = 0.15,
    clamp_hi_fine: float = 0.45,
):
    """Launch breakage kernel.

    Args:
        sizes: Particle sizes [n] (updated in place)
        masses: Particle masses [n] (updated in place)
        impact_flags: Impact indicators [n]
        impact_energies: Impact energies [n]
        break_flags: Output breakage indicators [n]
        rand_states: Random number generator states [n]
        selection_k: Base selection rate constant
        selection_alpha: Size exponent in selection function
        reference_size: Reference size for selection [m]
        min_energy: Minimum energy for breakage [J]
        energy_scale: Energy scaling factor
        breakage_gamma: Breakage distribution exponent (medium regime)
        min_size: Minimum particle size [m]
        coarse_threshold: Coarse/medium boundary [m]
        fine_threshold: Medium/fine boundary [m]
        gamma_coarse: Coarse regime Gaudin-Schuhmann exponent
        gamma_fine: Fine regime Gaudin-Schuhmann exponent
        clamp_lo_coarse: Coarse regime min reduction factor
        clamp_hi_coarse: Coarse regime max reduction factor
        clamp_lo_medium: Medium regime min reduction factor
        clamp_hi_medium: Medium regime max reduction factor
        clamp_lo_fine: Fine regime min reduction factor
        clamp_hi_fine: Fine regime max reduction factor
    """
    n = sizes.shape[0]
    wp.launch(
        breakage_kernel,
        dim=n,
        inputs=[
            sizes, masses, impact_flags, impact_energies, break_flags,
            rand_states, selection_k, selection_alpha, reference_size,
            min_energy, energy_scale, breakage_gamma, min_size,
            coarse_threshold, fine_threshold,
            gamma_coarse, gamma_fine,
            clamp_lo_coarse, clamp_hi_coarse,
            clamp_lo_medium, clamp_hi_medium,
            clamp_lo_fine, clamp_hi_fine,
        ],
    )


# NumPy fallback
def breakage_step_np(
    sizes: np.ndarray,
    masses: np.ndarray,
    impact_flags: np.ndarray,
    impact_energies: np.ndarray,
    selection_k: float = 0.15,
    selection_alpha: float = 1.2,
    reference_size: float = 0.001,
    min_energy: float = 0.001,
    energy_scale: float = 5.0,
    breakage_gamma: float = 0.8,
    min_size: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
    # Size-dependent regime parameters
    coarse_threshold: float = 1.0e-3,
    fine_threshold: float = 1.0e-4,
    gamma_coarse: float = 1.2,
    gamma_fine: float = 0.35,
    clamp_lo_coarse: float = 0.40,
    clamp_hi_coarse: float = 0.70,
    clamp_lo_medium: float = 0.20,
    clamp_hi_medium: float = 0.55,
    clamp_lo_fine: float = 0.15,
    clamp_hi_fine: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy implementation of breakage step.

    Returns:
        (new_sizes, new_masses, break_flags)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(sizes)
    new_sizes = sizes.copy()
    new_masses = masses.copy()
    break_flags = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if impact_flags[i] == 0:
            continue

        size = sizes[i]
        mass = masses[i]
        energy = impact_energies[i]

        if mass <= 0.0 or size < min_size:
            continue

        # Selection probability
        size_factor = (size / reference_size) ** selection_alpha
        energy_factor = min(1.0, energy / min_energy * energy_scale)
        selection_prob = min(1.0, selection_k * size_factor * energy_factor)

        if rng.random() > selection_prob:
            continue

        # Breakage!
        break_flags[i] = 1

        # Select regime based on particle size
        if size > coarse_threshold:
            gamma = gamma_coarse
            cl, ch = clamp_lo_coarse, clamp_hi_coarse
        elif size < fine_threshold:
            gamma = gamma_fine
            cl, ch = clamp_lo_fine, clamp_hi_fine
        else:
            gamma = breakage_gamma
            cl, ch = clamp_lo_medium, clamp_hi_medium

        # Daughter size from Gaudin-Schuhmann mean
        reduction_factor = gamma / (gamma + 1.0)
        rand_factor = 0.5 + rng.random()
        reduction_factor = reduction_factor * rand_factor
        reduction_factor = np.clip(reduction_factor, cl, ch)

        new_size = max(size * reduction_factor, min_size)
        new_sizes[i] = new_size

        # Update mass
        size_ratio = new_size / size
        new_masses[i] = mass * size_ratio ** 3

    return new_sizes, new_masses, break_flags


# ---------------------------------------------------------------------------
#  Multi-fragment post-processing (mass-conserving)
# ---------------------------------------------------------------------------

def generate_fragments_np(
    parent_sizes: np.ndarray,
    parent_masses: np.ndarray,
    primary_sizes: np.ndarray,
    primary_masses: np.ndarray,
    break_flags: np.ndarray,
    impact_energies: np.ndarray,
    gamma: float,
    min_size: float,
    max_fragments: int,
    frag_count_coeff: float,
    frag_count_size_exp: float,
    frag_count_energy_exp: float,
    energy_ref: float,
    rng: Optional[np.random.Generator] = None,
    # Size-dependent regime parameters
    coarse_threshold: float = 1.0e-3,
    fine_threshold: float = 1.0e-4,
    gamma_coarse: float = 1.2,
    gamma_fine: float = 0.35,
) -> Tuple[
    np.ndarray, np.ndarray,   # adjusted primary sizes & masses
    np.ndarray, np.ndarray,   # secondary fragment sizes & masses
    np.ndarray,               # parent_indices (index into broken particles)
    int,                      # num_fragments_created
]:
    """Generate secondary fragments and re-normalise for exact mass conservation.

    After the breakage kernel produces a single primary daughter per broken
    particle, this function:
      1. Determines fragment count N from size-reduction ratio and impact energy.
      2. Samples N-1 secondary daughter sizes from a Gaudin-Schuhmann
         inverse distribution using size-dependent gamma.
      3. Re-normalises all N fragment masses so they sum to the parent mass
         (volume-weighted: m_i = m_parent * d_i^3 / sum(d_j^3)).

    The primary daughter size/mass arrays are **modified in place** to reflect
    the re-normalised mass.  Secondary fragment arrays are returned separately
    for the caller to append to particle state.

    Args:
        parent_sizes: Original sizes before kernel [n]
        parent_masses: Original masses before kernel [n]
        primary_sizes: Sizes after kernel (primary daughters) [n]
        primary_masses: Masses after kernel (unused; overwritten) [n]
        break_flags: Breakage indicators from kernel [n]
        impact_energies: Impact energies [n]
        gamma: Breakage distribution exponent (medium regime, Gaudin-Schuhmann)
        min_size: Minimum particle size [m]
        max_fragments: Maximum total fragments per event (N_max)
        frag_count_coeff: C_n coefficient for fragment count
        frag_count_size_exp: alpha_n exponent on size ratio
        frag_count_energy_exp: beta_n exponent on energy ratio
        energy_ref: Reference energy for fragment count scaling [J]
        rng: NumPy random generator
        coarse_threshold: Coarse/medium boundary [m]
        fine_threshold: Medium/fine boundary [m]
        gamma_coarse: Coarse regime Gaudin-Schuhmann exponent
        gamma_fine: Fine regime Gaudin-Schuhmann exponent

    Returns:
        adjusted_primary_sizes: Primary sizes (unchanged)
        adjusted_primary_masses: Primary masses (re-normalised in place)
        frag_sizes: Secondary fragment sizes [M]
        frag_masses: Secondary fragment masses [M]
        parent_indices: Index of each fragment's parent in the *full* array [M]
        num_fragments_created: Total secondary fragments generated
    """
    if rng is None:
        rng = np.random.default_rng()

    broken_idx = np.where(break_flags == 1)[0]
    n_broken = len(broken_idx)

    if n_broken == 0:
        empty = np.zeros(0, dtype=np.float64)
        empty_idx = np.zeros(0, dtype=np.int64)
        return (
            primary_sizes, primary_masses,
            empty, empty,
            empty_idx, 0,
        )

    # Pre-allocate generous upper bound (each broken particle <= max_fragments-1 secondaries)
    max_secondary = n_broken * (max_fragments - 1)
    frag_sizes_buf = np.empty(max_secondary, dtype=np.float64)
    frag_masses_buf = np.empty(max_secondary, dtype=np.float64)
    parent_idx_buf = np.empty(max_secondary, dtype=np.int64)
    frag_cursor = 0

    for k in range(n_broken):
        idx = broken_idx[k]
        d_parent = parent_sizes[idx]
        m_parent = parent_masses[idx]
        d_primary = primary_sizes[idx]
        E = impact_energies[idx]

        # --- Skip fragmentation for very small parents ---
        if d_parent < 2.0 * min_size or d_primary <= min_size:
            # Just ensure mass conservation for the single primary
            primary_masses[idx] = m_parent
            continue

        # --- Per-particle gamma based on parent size regime ---
        if d_parent > coarse_threshold:
            eff_gamma = gamma_coarse
        elif d_parent < fine_threshold:
            eff_gamma = gamma_fine
        else:
            eff_gamma = gamma  # medium
        inv_gamma = 1.0 / eff_gamma if eff_gamma > 0 else 1.0

        # --- Fragment count ---
        # N = clamp(floor(C_n * (d_parent/d_primary)^0.5 * (E/E_ref)^0.3), 2, N_max)
        size_ratio = d_parent / d_primary if d_primary > 0 else 1.0
        energy_ratio = E / energy_ref if energy_ref > 0 else 1.0
        N_float = frag_count_coeff * (size_ratio ** frag_count_size_exp) * (energy_ratio ** frag_count_energy_exp)
        N = max(2, min(max_fragments, int(N_float)))

        # --- Sample secondary sizes from Gaudin-Schuhmann inverse ---
        # d_i = d_parent * u^(1/gamma),  u ~ Uniform(0,1)
        # Clamp to [min_size, d_primary]
        n_secondary = N - 1
        u = rng.uniform(0.0, 1.0, n_secondary)
        sec_sizes = d_parent * np.power(u, inv_gamma)
        sec_sizes = np.clip(sec_sizes, min_size, d_primary)

        # --- Mass conservation: volume-weighted re-normalisation ---
        # all volumes: primary + secondaries
        all_d = np.empty(N, dtype=np.float64)
        all_d[0] = d_primary
        all_d[1:] = sec_sizes

        all_v = all_d ** 3
        v_sum = all_v.sum()
        if v_sum <= 0:
            primary_masses[idx] = m_parent
            continue

        all_m = m_parent * (all_v / v_sum)

        # Write back re-normalised primary mass
        primary_masses[idx] = all_m[0]

        # Store secondary fragments
        end = frag_cursor + n_secondary
        frag_sizes_buf[frag_cursor:end] = sec_sizes
        frag_masses_buf[frag_cursor:end] = all_m[1:]
        parent_idx_buf[frag_cursor:end] = idx
        frag_cursor = end

    # Trim buffers to actual count
    frag_sizes = frag_sizes_buf[:frag_cursor].copy()
    frag_masses = frag_masses_buf[:frag_cursor].copy()
    parent_indices = parent_idx_buf[:frag_cursor].copy()

    return (
        primary_sizes, primary_masses,
        frag_sizes, frag_masses,
        parent_indices, frag_cursor,
    )


# Population balance on PSD bins
def breakage_psd_np(
    psd_masses: np.ndarray,
    size_classes: np.ndarray,
    total_impact_energy: float,
    selection_k: float = 0.15,
    selection_alpha: float = 1.2,
    breakage_gamma: float = 0.8,
    dt: float = 0.001,
    # Size-dependent regime parameters
    coarse_threshold: float = 1.0e-3,
    fine_threshold: float = 1.0e-4,
    gamma_coarse: float = 1.2,
    gamma_fine: float = 0.35,
) -> np.ndarray:
    """Apply breakage to a particle size distribution (population balance).

    This operates on mass fractions in size classes rather than
    individual particles. Uses size-dependent gamma per parent class.

    Args:
        psd_masses: Mass in each size class [n_classes]
        size_classes: Size class midpoints [n_classes]
        total_impact_energy: Total impact energy this timestep
        selection_k: Selection rate constant
        selection_alpha: Size exponent
        breakage_gamma: Breakage distribution exponent (medium regime)
        dt: Timestep
        coarse_threshold: Coarse/medium boundary [m]
        fine_threshold: Medium/fine boundary [m]
        gamma_coarse: Coarse regime Gaudin-Schuhmann exponent
        gamma_fine: Fine regime Gaudin-Schuhmann exponent

    Returns:
        New PSD masses [n_classes]
    """
    n = len(psd_masses)
    new_masses = psd_masses.copy()

    # Build breakage matrix
    # B[i, j] = fraction of mass from class j going to class i
    # Upper triangular: mass only moves to smaller sizes
    B = np.zeros((n, n))
    for j in range(n):
        d_parent = size_classes[j]
        # Select gamma based on parent size regime
        if d_parent > coarse_threshold:
            g = gamma_coarse
        elif d_parent < fine_threshold:
            g = gamma_fine
        else:
            g = breakage_gamma  # medium
        for i in range(j + 1):  # i <= j (smaller or equal)
            d_daughter = size_classes[i]
            # Cumulative breakage function
            ratio = d_daughter / d_parent
            B[i, j] = ratio ** g

    # Normalize columns (mass conservation)
    for j in range(n):
        col_sum = B[:, j].sum()
        if col_sum > 0:
            B[:, j] /= col_sum

    # Selection rates
    d_ref = size_classes[n // 2]
    S = np.array([
        selection_k * (d / d_ref) ** selection_alpha
        for d in size_classes
    ])

    # Energy scaling
    energy_factor = min(1.0, total_impact_energy * 100)
    S = np.clip(S * energy_factor, 0, 1)

    # Apply population balance
    # dm_i/dt = sum_j(S_j * B_ij * m_j) - S_i * m_i
    for j in range(n):
        mass_breaking = S[j] * psd_masses[j] * dt
        new_masses[j] -= mass_breaking
        # Distribute to smaller sizes
        for i in range(j + 1):
            new_masses[i] += B[i, j] * mass_breaking

    return np.maximum(new_masses, 0)
