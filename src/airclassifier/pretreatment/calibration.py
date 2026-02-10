"""
Model Calibration Against PLC Data
====================================

Fits simulation model parameters to match real GP-15 PLC recordings.

This is distinct from recipe optimization (optimizer.py):

    Recipe optimization:  "What gap + speed gives the best product?"
    Model calibration:    "What physics parameters make the sim match reality?"

Parameters calibrated (with physical meaning):

    oscillator_coupling_factor  — tank circuit efficiency (Manual §2.2.1)
    k_evap                      — evaporation rate for whole seeds (Manual Ch.5)
    gap_adjust_rate_mm_s        — MRH electrode drive speed (Manual §8.1)

Targets from PLC data (CSV from actual machine runs):

    Product_Temp(t)   — outfeed temperature trajectory [C]
    Ia(t)             — anode current trajectory [A]
    Electrode_Act(t)  — actual electrode gap trajectory [mm]

Usage::

    from airclassifier.pretreatment.calibration import (
        CalibrationOptimizer, load_plc_data,
    )

    plc = load_plc_data("utility_docs/Run1 RF data(in).csv")
    cal = CalibrationOptimizer(plc, material=mat)
    result = cal.run()
    print(result)
    result.apply(config, material, controller)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MachineConfig, MaterialProperties, Recipe
from .simulator import GP15Simulator


# ─────────────────────────────────────────────────────────────────────
#  PLC Data Loading
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PLCRecording:
    """Parsed PLC data from a GP-15 production run."""
    time_s: np.ndarray           # time since start [s]
    anode_current_a: np.ndarray  # Ia [A]
    grid_current_a: np.ndarray   # Ig [A]
    product_temp_c: np.ndarray   # outfeed sensor [C]
    electrode_set_mm: np.ndarray # setpoint [mm]
    electrode_act_mm: np.ndarray # actual [mm]
    conv_speed_m_per_min: np.ndarray
    ia_limit_1: float            # MRL / Ia 1st Limit
    ia_limit_2: float            # MRH / Ia 2nd Limit

    # Run metadata
    duration_s: float = 0.0
    n_samples: int = 0
    sample_interval_s: float = 5.0


def load_plc_data(csv_path: str | Path) -> PLCRecording:
    """Load a GP-15 PLC CSV recording.

    Expected columns (from Run1 RF data format):
        Date, Time, Ia, Ig, Ia 2nd Limit, Ia 1st Limit,
        Conv_Speed, Product_Temp, Electrode_Set, Electrode_Act

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Parsed :class:`PLCRecording`.
    """
    import csv
    from datetime import datetime

    path = Path(csv_path)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    # Parse timestamps to seconds since start
    def _parse_time(date_str: str, time_str: str) -> float:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
        return dt.timestamp()

    t0 = _parse_time(rows[0]["Date"], rows[0]["Time"])
    time_s = np.array([_parse_time(r["Date"], r["Time"]) - t0 for r in rows])

    ia = np.array([float(r["Ia"]) for r in rows])
    ig = np.array([float(r["Ig"]) for r in rows])
    temp = np.array([float(r["Product_Temp"]) for r in rows])
    e_set = np.array([float(r["Electrode_Set"]) for r in rows])
    e_act = np.array([float(r["Electrode_Act"]) for r in rows])
    speed = np.array([float(r["Conv_Speed"]) for r in rows])
    ia_lim1 = float(rows[0]["Ia 1st Limit"])
    ia_lim2 = float(rows[0]["Ia 2nd Limit"])

    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 5.0

    return PLCRecording(
        time_s=time_s,
        anode_current_a=ia,
        grid_current_a=ig,
        product_temp_c=temp,
        electrode_set_mm=e_set,
        electrode_act_mm=e_act,
        conv_speed_m_per_min=speed,
        ia_limit_1=ia_lim1,
        ia_limit_2=ia_lim2,
        duration_s=float(time_s[-1] - time_s[0]),
        n_samples=len(rows),
        sample_interval_s=float(dt),
    )


# ─────────────────────────────────────────────────────────────────────
#  Calibration Result
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Result of model calibration against PLC data."""

    # Optimized parameters
    oscillator_coupling_factor: float = 0.258
    k_evap: float = 5.0e-5
    gap_adjust_rate_mm_s: float = 0.1

    # Fit quality
    loss_total: float = 0.0
    loss_temperature: float = 0.0
    loss_gap: float = 0.0
    n_iterations: int = 0
    converged: bool = False

    # History
    history: List[Dict] = field(default_factory=list)

    def apply(
        self,
        config: MachineConfig,
        material: MaterialProperties,
    ):
        """Apply calibrated parameters to the config and material.

        Modifies the objects in-place so the next simulation uses
        the calibrated values.
        """
        config.oscillator_coupling_factor = self.oscillator_coupling_factor
        material.k_evap = self.k_evap

    def __str__(self) -> str:
        return (
            f"CalibrationResult:\n"
            f"  coupling_factor = {self.oscillator_coupling_factor:.4f}\n"
            f"  k_evap          = {self.k_evap:.2e}\n"
            f"  gap_rate         = {self.gap_adjust_rate_mm_s:.4f} mm/s\n"
            f"  loss_total       = {self.loss_total:.4f}\n"
            f"  loss_temperature = {self.loss_temperature:.4f}\n"
            f"  loss_gap         = {self.loss_gap:.4f}\n"
            f"  iterations       = {self.n_iterations}\n"
            f"  converged        = {self.converged}"
        )


# ─────────────────────────────────────────────────────────────────────
#  Calibration Optimizer
# ─────────────────────────────────────────────────────────────────────

class CalibrationOptimizer:
    """Fit simulation parameters to match real PLC data.

    Uses scipy.optimize.differential_evolution (global optimizer,
    no gradient needed) to minimize the weighted MSE between
    simulated and measured trajectories:

        L = w_T * MSE(T_sim, T_plc)
          + w_gap * MSE(gap_sim, gap_plc)

    where the trajectories are compared at matched time points.

    The optimizer varies:
        - oscillator_coupling_factor (controls RF power level)
        - k_evap (controls drying rate vs sensible heating)
        - gap_adjust_rate_mm_s (controls MRH response speed)

    All other parameters (material properties, machine geometry)
    are held fixed at their known values.

    Args:
        plc_data: Parsed PLC recording from :func:`load_plc_data`.
        config: Machine configuration (held fixed except coupling).
        material: Material properties (held fixed except k_evap).
        recipe_overrides: Override recipe fields from PLC data.
        sim_duration_s: How much of the PLC recording to fit.
            Default: use the full recording duration.
        w_temperature: Weight for temperature MSE.
        w_gap: Weight for electrode gap MSE.
        n_compare_points: Number of time points to compare.
    """

    def __init__(
        self,
        plc_data: PLCRecording,
        config: MachineConfig | None = None,
        material: MaterialProperties | None = None,
        recipe_overrides: Dict | None = None,
        sim_duration_s: float | None = None,
        device: str | None = None,
        w_temperature: float = 1.0,
        w_gap: float = 0.5,
        n_compare_points: int = 50,
    ):
        self._plc = plc_data
        self._config = config or MachineConfig()
        self._material = material or MaterialProperties()
        # Auto-detect CUDA (uses GPU if available — RTX 6000 = ~5x faster)
        self._device = device  # None = auto-detect in GP15Simulator
        self._w_T = w_temperature
        self._w_gap = w_gap
        self._n_pts = n_compare_points

        # Duration to simulate (default: match PLC recording)
        self._sim_duration = sim_duration_s or plc_data.duration_s

        # Build recipe from PLC data
        gap_set = float(np.median(plc_data.electrode_set_mm))
        speed = float(np.median(plc_data.conv_speed_m_per_min))
        overrides = recipe_overrides or {}
        self._recipe = Recipe(
            name="calibration",
            recipe_number=0,
            electrode_gap_mm=overrides.get("electrode_gap_mm", gap_set),
            belt_speed_m_per_min=overrides.get("belt_speed_m_per_min", speed),
            mrh_amps=overrides.get("mrh_amps", plc_data.ia_limit_2),
            mrl_amps=overrides.get("mrl_amps", plc_data.ia_limit_1),
        )

        # Resample PLC data to n_compare_points for MSE computation
        self._plc_times = np.linspace(0, self._sim_duration, n_compare_points)
        self._plc_temp = np.interp(
            self._plc_times, plc_data.time_s, plc_data.product_temp_c,
        )
        self._plc_gap = np.interp(
            self._plc_times, plc_data.time_s, plc_data.electrode_act_mm,
        )

        self._eval_count = 0

    def _simulate(
        self,
        coupling: float,
        k_evap: float,
        gap_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation with candidate parameters.

        Returns:
            (T_sim, gap_sim) arrays at the comparison time points.
        """
        # Temporarily override parameters
        config = MachineConfig()
        config.oscillator_coupling_factor = coupling

        material = MaterialProperties(
            name=self._material.name,
            initial_moisture_wb=self._material.initial_moisture_wb,
            initial_temperature_c=self._material.initial_temperature_c,
            bed_depth_m=self._material.bed_depth_m,
            k_evap=k_evap,
            T_evap_threshold_c=self._material.T_evap_threshold_c,
            dielectric_loss_coeffs=self._material.dielectric_loss_coeffs,
            dielectric_const_coeffs=self._material.dielectric_const_coeffs,
            rho_solid=self._material.rho_solid,
            bed_porosity=self._material.bed_porosity,
        )

        sim = GP15Simulator(
            config=config,
            material=material,
            device=self._device,  # None = auto-detect (CUDA if available)
            enable_controller=True,
            enable_corrections=False,
            use_tvd=False,  # speed over accuracy for calibration
        )
        sim.load_recipe(self._recipe)

        # Override gap adjustment rate
        sim._sim.controller._GAP_ADJUST_RATE_MM_S = gap_rate

        try:
            result = sim.run(duration_s=self._sim_duration, adaptive_dt=True)
        except Exception:
            # Simulation diverged — return penalty values
            return (
                np.full_like(self._plc_times, 100.0),
                np.full_like(self._plc_times, 300.0),
            )

        ts = result.time_series
        if not ts.get("time_s") or len(ts["time_s"]) < 2:
            return (
                np.full_like(self._plc_times, 100.0),
                np.full_like(self._plc_times, 300.0),
            )

        t_sim = np.array(ts["time_s"])
        T_sim = np.array(ts["T_outfeed_c"])
        gap_sim = np.array(ts["electrode_gap_mm"])

        # Interpolate to comparison time points
        T_interp = np.interp(self._plc_times, t_sim, T_sim)
        gap_interp = np.interp(self._plc_times, t_sim, gap_sim)

        return T_interp, gap_interp

    def _objective(self, params: np.ndarray) -> float:
        """Objective function for the optimizer.

        Args:
            params: [coupling_factor, k_evap, gap_rate]

        Returns:
            Weighted MSE loss.
        """
        coupling = float(params[0])
        k_evap = float(params[1])
        gap_rate = float(params[2])

        self._eval_count += 1

        T_sim, gap_sim = self._simulate(coupling, k_evap, gap_rate)

        # MSE losses
        loss_T = float(np.mean((T_sim - self._plc_temp) ** 2))
        loss_gap = float(np.mean((gap_sim - self._plc_gap) ** 2))

        total = self._w_T * loss_T + self._w_gap * loss_gap

        if self._eval_count % 10 == 0:
            print(f"  eval {self._eval_count:3d}: "
                  f"k={coupling:.4f} k_evap={k_evap:.2e} "
                  f"gap_rate={gap_rate:.4f}  "
                  f"L_T={loss_T:.1f} L_gap={loss_gap:.1f} "
                  f"total={total:.1f}")

        return total

    def run(
        self,
        maxiter: int = 30,
        seed: int = 42,
    ) -> CalibrationResult:
        """Run the calibration optimization.

        Uses scipy.optimize.differential_evolution for global
        optimization (robust, derivative-free, handles noisy
        objectives well).

        Args:
            maxiter: Maximum generations for differential evolution.
            seed: Random seed for reproducibility.

        Returns:
            :class:`CalibrationResult` with optimized parameters.
        """
        try:
            from scipy.optimize import differential_evolution
        except ImportError:
            raise ImportError(
                "Calibration requires scipy. Install: pip install scipy"
            )

        # Parameter bounds (physically meaningful ranges)
        bounds = [
            (0.10, 0.40),     # oscillator_coupling_factor
            (1e-6, 5e-4),     # k_evap
            (0.01, 1.0),      # gap_adjust_rate_mm_s
        ]

        # Detect actual device for display
        dev_name = self._device or "auto-detect"
        print(f"Starting calibration against PLC data "
              f"({self._plc.n_samples} samples, {self._plc.duration_s:.0f} s)")
        print(f"  Sim duration: {self._sim_duration:.0f} s")
        print(f"  Device: {dev_name}")
        print(f"  Compare points: {self._n_pts}")
        print(f"  Bounds: coupling=[{bounds[0]}], k_evap=[{bounds[1]}], "
              f"gap_rate=[{bounds[2]}]")
        print()

        self._eval_count = 0
        history = []

        def _callback(xk, convergence):
            history.append({
                "coupling": float(xk[0]),
                "k_evap": float(xk[1]),
                "gap_rate": float(xk[2]),
                "convergence": float(convergence),
            })

        result = differential_evolution(
            self._objective,
            bounds=bounds,
            maxiter=maxiter,
            seed=seed,
            tol=0.01,
            callback=_callback,
            disp=True,
            polish=False,  # skip local polish for speed
            popsize=8,     # small population for faster convergence
        )

        best = result.x
        T_best, gap_best = self._simulate(best[0], best[1], best[2])
        loss_T = float(np.mean((T_best - self._plc_temp) ** 2))
        loss_gap = float(np.mean((gap_best - self._plc_gap) ** 2))

        return CalibrationResult(
            oscillator_coupling_factor=float(best[0]),
            k_evap=float(best[1]),
            gap_adjust_rate_mm_s=float(best[2]),
            loss_total=float(result.fun),
            loss_temperature=loss_T,
            loss_gap=loss_gap,
            n_iterations=result.nit,
            converged=result.success,
            history=history,
        )

    def evaluate_current(self) -> Dict[str, float]:
        """Evaluate the current parameter values against PLC data.

        Useful for checking the baseline fit before running
        calibration.

        Returns:
            Dict with loss components and parameter values.
        """
        coupling = self._config.oscillator_coupling_factor
        k_evap = self._material.k_evap
        gap_rate = 0.1  # current default

        T_sim, gap_sim = self._simulate(coupling, k_evap, gap_rate)
        loss_T = float(np.mean((T_sim - self._plc_temp) ** 2))
        loss_gap = float(np.mean((gap_sim - self._plc_gap) ** 2))

        return {
            "coupling": coupling,
            "k_evap": k_evap,
            "gap_rate": gap_rate,
            "loss_T": loss_T,
            "loss_gap": loss_gap,
            "loss_total": self._w_T * loss_T + self._w_gap * loss_gap,
            "T_sim_final": float(T_sim[-1]),
            "T_plc_final": float(self._plc_temp[-1]),
            "gap_sim_final": float(gap_sim[-1]),
            "gap_plc_final": float(self._plc_gap[-1]),
        }
