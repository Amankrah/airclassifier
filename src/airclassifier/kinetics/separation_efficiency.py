"""
Separation efficiency calculations for cyclone air classifier.

Provides grade efficiency curves, cut size calculations, and
separation performance analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


@dataclass
class GradeEfficiencyCurve:
    """
    Grade efficiency curve for cyclone separator.

    The grade efficiency G(d) represents the fraction of particles
    of diameter d that are collected (go to underflow).
    """

    # Particle size bins (midpoints) [m]
    diameters: np.ndarray

    # Collection efficiency for each size bin (0-1)
    efficiencies: np.ndarray

    # Number of particles in each bin (for statistics)
    counts_feed: np.ndarray      # Particles fed
    counts_collected: np.ndarray # Particles collected
    counts_escaped: np.ndarray   # Particles escaped (overflow)

    @property
    def d50(self) -> Optional[float]:
        """
        Cut size (d50) - diameter at which 50% of particles are collected.

        Returns:
            d50 in meters, or None if cannot be determined
        """
        if len(self.diameters) < 2:
            return None

        # Find where efficiency crosses 0.5
        try:
            # Create interpolation function
            f = interp1d(self.diameters, self.efficiencies - 0.5,
                        kind='linear', fill_value='extrapolate')

            # Find root (where efficiency = 0.5)
            d_min, d_max = self.diameters.min(), self.diameters.max()

            # Check if 0.5 is within range
            if f(d_min) * f(d_max) > 0:
                # Doesn't cross 0.5 in range
                if np.mean(self.efficiencies) > 0.5:
                    return d_min * 0.5  # Estimate below range
                else:
                    return d_max * 2.0  # Estimate above range

            d50 = brentq(f, d_min, d_max)
            return d50

        except Exception:
            # Fallback: linear interpolation
            idx = np.searchsorted(self.efficiencies, 0.5)
            if idx == 0 or idx >= len(self.diameters):
                return None
            d1, d2 = self.diameters[idx-1], self.diameters[idx]
            e1, e2 = self.efficiencies[idx-1], self.efficiencies[idx]
            if abs(e2 - e1) < 1e-10:
                return (d1 + d2) / 2
            return d1 + (0.5 - e1) * (d2 - d1) / (e2 - e1)

    @property
    def d25(self) -> Optional[float]:
        """Diameter at 25% efficiency."""
        return self._find_diameter_at_efficiency(0.25)

    @property
    def d75(self) -> Optional[float]:
        """Diameter at 75% efficiency."""
        return self._find_diameter_at_efficiency(0.75)

    @property
    def sharpness_index(self) -> Optional[float]:
        """
        Sharpness index = d25/d75.

        Higher values (closer to 1) indicate sharper separation.
        """
        d25 = self.d25
        d75 = self.d75
        if d25 is None or d75 is None or d75 < 1e-15:
            return None
        return d25 / d75

    @property
    def overall_efficiency(self) -> float:
        """
        Overall mass collection efficiency.

        Weighted average of grade efficiencies by feed count.
        """
        total_feed = np.sum(self.counts_feed)
        if total_feed == 0:
            return 0.0

        total_collected = np.sum(self.counts_collected)
        return total_collected / total_feed

    def _find_diameter_at_efficiency(self, target_eff: float) -> Optional[float]:
        """Find diameter at a given efficiency level."""
        if len(self.diameters) < 2:
            return None

        try:
            f = interp1d(self.diameters, self.efficiencies - target_eff,
                        kind='linear', fill_value='extrapolate')
            d_min, d_max = self.diameters.min(), self.diameters.max()

            if f(d_min) * f(d_max) > 0:
                return None

            return brentq(f, d_min, d_max)
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "diameters_um": (self.diameters * 1e6).tolist(),
            "efficiencies": self.efficiencies.tolist(),
            "counts_feed": self.counts_feed.tolist(),
            "counts_collected": self.counts_collected.tolist(),
            "counts_escaped": self.counts_escaped.tolist(),
            "d50_um": self.d50 * 1e6 if self.d50 else None,
            "d25_um": self.d25 * 1e6 if self.d25 else None,
            "d75_um": self.d75 * 1e6 if self.d75 else None,
            "sharpness_index": self.sharpness_index,
            "overall_efficiency": self.overall_efficiency,
        }


def compute_grade_efficiency(
    diameters: np.ndarray,
    is_active: np.ndarray,
    num_bins: int = 20,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    log_scale: bool = True
) -> GradeEfficiencyCurve:
    """
    Compute grade efficiency curve from simulation results.

    Args:
        diameters: Array of particle diameters [m]
        is_active: Array of particle states:
                   1 = still active
                   -1 = collected (underflow)
                   -2 = escaped (overflow)
        num_bins: Number of size bins
        d_min: Minimum diameter for binning (uses data min if None)
        d_max: Maximum diameter for binning (uses data max if None)
        log_scale: Use logarithmic bin spacing

    Returns:
        GradeEfficiencyCurve object
    """
    # Filter to only finished particles
    finished_mask = (is_active == -1) | (is_active == -2)
    d_finished = diameters[finished_mask]
    status_finished = is_active[finished_mask]

    if len(d_finished) == 0:
        # No finished particles - return empty curve
        return GradeEfficiencyCurve(
            diameters=np.array([]),
            efficiencies=np.array([]),
            counts_feed=np.array([]),
            counts_collected=np.array([]),
            counts_escaped=np.array([])
        )

    # Determine bin edges
    if d_min is None:
        d_min = d_finished.min() * 0.9
    if d_max is None:
        d_max = d_finished.max() * 1.1

    if log_scale and d_min > 0:
        bin_edges = np.logspace(np.log10(d_min), np.log10(d_max), num_bins + 1)
    else:
        bin_edges = np.linspace(d_min, d_max, num_bins + 1)

    # Compute bin midpoints
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count particles in each bin
    collected_mask = status_finished == -1
    escaped_mask = status_finished == -2

    counts_collected, _ = np.histogram(d_finished[collected_mask], bins=bin_edges)
    counts_escaped, _ = np.histogram(d_finished[escaped_mask], bins=bin_edges)
    counts_feed = counts_collected + counts_escaped

    # Compute efficiencies
    efficiencies = np.zeros(num_bins)
    for i in range(num_bins):
        if counts_feed[i] > 0:
            efficiencies[i] = counts_collected[i] / counts_feed[i]
        else:
            # No data - interpolate or use 0
            efficiencies[i] = 0.0

    return GradeEfficiencyCurve(
        diameters=bin_centers,
        efficiencies=efficiencies,
        counts_feed=counts_feed,
        counts_collected=counts_collected,
        counts_escaped=counts_escaped
    )


def theoretical_d50_lapple(
    cyclone_diameter: float,
    inlet_width: float,
    inlet_velocity: float,
    fluid_density: float,
    fluid_viscosity: float,
    particle_density: float,
    num_turns: float = 5.0
) -> float:
    """
    Calculate theoretical d50 using Lapple model.

    d50 = sqrt(9 * mu * b / (2 * pi * N * v_in * (rho_p - rho_f)))

    Args:
        cyclone_diameter: Cyclone body diameter [m]
        inlet_width: Inlet width [m]
        inlet_velocity: Inlet gas velocity [m/s]
        fluid_density: Gas density [kg/m³]
        fluid_viscosity: Gas dynamic viscosity [Pa·s]
        particle_density: Particle density [kg/m³]
        num_turns: Number of effective turns in cyclone

    Returns:
        Theoretical d50 [m]
    """
    b = inlet_width  # Inlet width
    N = num_turns
    v_in = inlet_velocity
    mu = fluid_viscosity
    rho_p = particle_density
    rho_f = fluid_density

    d50_sq = (9 * mu * b) / (2 * np.pi * N * v_in * (rho_p - rho_f))

    if d50_sq < 0:
        return 0.0

    return np.sqrt(d50_sq)


def theoretical_d50_barth(
    cyclone_diameter: float,
    vortex_finder_diameter: float,
    inlet_velocity: float,
    fluid_density: float,
    fluid_viscosity: float,
    particle_density: float,
    friction_factor: float = 0.005
) -> float:
    """
    Calculate theoretical d50 using Barth model.

    More accurate than Lapple for modern cyclone designs.

    Args:
        cyclone_diameter: Cyclone body diameter [m]
        vortex_finder_diameter: Vortex finder diameter [m]
        inlet_velocity: Inlet gas velocity [m/s]
        fluid_density: Gas density [kg/m³]
        fluid_viscosity: Gas dynamic viscosity [Pa·s]
        particle_density: Particle density [kg/m³]
        friction_factor: Wall friction factor

    Returns:
        Theoretical d50 [m]
    """
    D = cyclone_diameter
    Dx = vortex_finder_diameter
    v_in = inlet_velocity
    mu = fluid_viscosity
    rho_p = particle_density
    rho_f = fluid_density
    f = friction_factor

    # Characteristic radius (at vortex finder edge)
    r_x = Dx / 2

    # Estimate tangential velocity at r_x
    v_tan = v_in * (D / Dx) ** 0.5  # Simplified estimate

    # d50 from force balance
    d50_sq = (18 * mu * r_x) / (v_tan ** 2 * (rho_p - rho_f))

    if d50_sq < 0:
        return 0.0

    return np.sqrt(d50_sq)


def rosin_rammler_grade_efficiency(
    d: np.ndarray,
    d50: float,
    m: float = 2.0
) -> np.ndarray:
    """
    Rosin-Rammler type grade efficiency curve.

    G(d) = 1 - exp(-0.693 * (d/d50)^m)

    This is an empirical model that fits many cyclone grade
    efficiency curves well.

    Args:
        d: Particle diameters [m]
        d50: Cut size [m]
        m: Sharpness parameter (higher = sharper cut)

    Returns:
        Grade efficiencies (0-1)
    """
    return 1.0 - np.exp(-0.693 * (d / d50) ** m)


def fit_grade_efficiency_curve(
    measured_curve: GradeEfficiencyCurve
) -> Tuple[float, float]:
    """
    Fit Rosin-Rammler parameters to measured grade efficiency curve.

    Args:
        measured_curve: Measured GradeEfficiencyCurve

    Returns:
        Tuple of (d50, m) parameters
    """
    from scipy.optimize import curve_fit

    def rr_model(d, d50, m):
        return 1.0 - np.exp(-0.693 * (d / d50) ** m)

    # Filter out zero-count bins
    mask = measured_curve.counts_feed > 0
    d = measured_curve.diameters[mask]
    eff = measured_curve.efficiencies[mask]

    if len(d) < 2:
        return measured_curve.d50 or 50e-6, 2.0

    try:
        # Initial guess
        d50_init = measured_curve.d50 or np.median(d)
        m_init = 2.0

        popt, _ = curve_fit(
            rr_model, d, eff,
            p0=[d50_init, m_init],
            bounds=([d.min() * 0.1, 0.5], [d.max() * 10, 10.0])
        )
        return popt[0], popt[1]

    except Exception:
        return measured_curve.d50 or np.median(d), 2.0


def plot_grade_efficiency(
    curve: GradeEfficiencyCurve,
    ax=None,
    show_theoretical: bool = True,
    d50_theoretical: Optional[float] = None,
    title: str = "Grade Efficiency Curve"
):
    """
    Plot grade efficiency curve.

    Args:
        curve: GradeEfficiencyCurve to plot
        ax: Matplotlib axes (creates new figure if None)
        show_theoretical: Show fitted R-R curve
        d50_theoretical: Theoretical d50 to show
        title: Plot title

    Returns:
        Matplotlib figure object, or None if no data to plot
    """
    import matplotlib.pyplot as plt

    # Check if we have data to plot
    if len(curve.diameters) == 0:
        print("Warning: No grade efficiency data to plot (no particles collected/escaped)")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot measured data
    d_um = curve.diameters * 1e6
    ax.scatter(d_um, curve.efficiencies * 100, s=50, c='blue',
              label='Simulation', zorder=5)

    # Connect with line
    ax.plot(d_um, curve.efficiencies * 100, 'b-', alpha=0.5)

    # Plot fitted curve
    if show_theoretical and len(curve.diameters) > 2:
        d50_fit, m_fit = fit_grade_efficiency_curve(curve)
        d_fine = np.logspace(np.log10(curve.diameters.min()),
                            np.log10(curve.diameters.max()), 100)
        eff_fit = rosin_rammler_grade_efficiency(d_fine, d50_fit, m_fit)
        ax.plot(d_fine * 1e6, eff_fit * 100, 'r--',
               label=f'R-R fit (d50={d50_fit*1e6:.1f}μm, m={m_fit:.2f})')

    # Mark d50
    if curve.d50:
        ax.axvline(curve.d50 * 1e6, color='green', linestyle=':',
                  label=f'd50 = {curve.d50*1e6:.1f} μm')
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

    # Theoretical d50
    if d50_theoretical:
        ax.axvline(d50_theoretical * 1e6, color='orange', linestyle='--',
                  label=f'd50 (theoretical) = {d50_theoretical*1e6:.1f} μm')

    ax.set_xlabel('Particle Diameter (μm)')
    ax.set_ylabel('Collection Efficiency (%)')
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_ylim(0, 105)
    ax.set_xlim(d_um.min() * 0.8, d_um.max() * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
