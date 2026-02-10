"""
Model Calibration Against PLC Data
====================================

Fits simulation model parameters to match real GP-15 PLC recordings.

Parameters calibrated:
    oscillator_coupling_factor  -- tank circuit efficiency (Manual section 2.2.1)
    k_evap                      -- evaporation rate for whole seeds (Manual Ch.5)
    gap_adjust_rate_mm_s        -- MRH electrode drive speed (Manual section 8.1)

Targets from PLC data:
    Product_Temp(t)   -- outfeed temperature trajectory [C]
    Ia(t)             -- anode current trajectory [A]
    Electrode_Act(t)  -- actual electrode gap trajectory [mm]

Improvements over naive approach:
    - Simulator reuse (reset() instead of re-constructing per evaluation)
    - Ia included in loss function (most informative signal)
    - Loss terms normalized by PLC signal variance (interpretable weights)
    - polish=True for local refinement after DE convergence
    - Sensitivity analysis at the optimum
    - Proper exception handling with logging

Usage::

    from airclassifier.pretreatment.calibration import (
        CalibrationOptimizer, load_plc_data,
    )

    plc = load_plc_data("utility_docs/Run1 RF data(in).csv")
    cal = CalibrationOptimizer(plc, material=mat)
    result = cal.run()
    print(result)
    result.apply(config, material)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration_store import get_calibration_defaults
from .config import MachineConfig, MaterialProperties, Recipe
from .simulator import GP15Simulator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  PLC Data Loading
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PLCRecording:
    """Parsed PLC data from a GP-15 production run."""
    time_s: np.ndarray
    anode_current_a: np.ndarray
    grid_current_a: np.ndarray
    product_temp_c: np.ndarray
    electrode_set_mm: np.ndarray
    electrode_act_mm: np.ndarray
    conv_speed_m_per_min: np.ndarray
    ia_limit_1: float            # MRL
    ia_limit_2: float            # MRH
    duration_s: float = 0.0
    n_samples: int = 0
    sample_interval_s: float = 5.0


def load_plc_data(csv_path: str | Path) -> PLCRecording:
    """Load a GP-15 PLC CSV recording.

    Expected columns:
        Date, Time, Ia, Ig, Ia 2nd Limit, Ia 1st Limit,
        Conv_Speed, Product_Temp, Electrode_Set, Electrode_Act
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
        time_s=time_s, anode_current_a=ia, grid_current_a=ig,
        product_temp_c=temp, electrode_set_mm=e_set,
        electrode_act_mm=e_act, conv_speed_m_per_min=speed,
        ia_limit_1=ia_lim1, ia_limit_2=ia_lim2,
        duration_s=float(time_s[-1] - time_s[0]),
        n_samples=len(rows), sample_interval_s=float(dt),
    )


# ─────────────────────────────────────────────────────────────────────
#  Calibration Result
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Result of model calibration against PLC data.

    Defaults for coupling, k_evap, gap_rate come from
    utility_docs/calibration_latest.json (single source of truth).
    """
    oscillator_coupling_factor: float = field(
        default_factory=lambda: get_calibration_defaults()[0]
    )
    k_evap: float = field(default_factory=lambda: get_calibration_defaults()[1])
    gap_adjust_rate_mm_s: float = field(
        default_factory=lambda: get_calibration_defaults()[2]
    )

    loss_total: float = 0.0
    loss_temperature: float = 0.0
    loss_anode_current: float = 0.0
    loss_gap: float = 0.0
    n_evaluations: int = 0
    n_iterations: int = 0
    converged: bool = False

    # Sensitivity (d_loss / d_param at optimum)
    sensitivity: Dict[str, float] = field(default_factory=dict)

    history: List[Dict] = field(default_factory=list)

    def apply(self, config: MachineConfig, material: MaterialProperties):
        """Apply calibrated parameters in-place."""
        config.oscillator_coupling_factor = self.oscillator_coupling_factor
        material.k_evap = self.k_evap

    def __str__(self) -> str:
        lines = [
            "CalibrationResult:",
            f"  coupling_factor = {self.oscillator_coupling_factor:.4f}",
            f"  k_evap          = {self.k_evap:.2e}",
            f"  gap_rate         = {self.gap_adjust_rate_mm_s:.4f} mm/s",
            f"  loss_total       = {self.loss_total:.4f}",
            f"    L_temperature  = {self.loss_temperature:.4f}",
            f"    L_anode_current = {self.loss_anode_current:.4f}",
            f"    L_gap          = {self.loss_gap:.4f}",
            f"  evaluations      = {self.n_evaluations}",
            f"  iterations       = {self.n_iterations}",
            f"  converged        = {self.converged}",
        ]
        if self.sensitivity:
            lines.append("  sensitivity (dL/dp):")
            for k, v in self.sensitivity.items():
                lines.append(f"    {k:30s} = {v:+.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
#  Calibration Optimizer
# ─────────────────────────────────────────────────────────────────────

class CalibrationOptimizer:
    """Fit simulation parameters to match real PLC data.

    Loss function (normalized MSE):

        L = w_T   * MSE(T_sim, T_plc)   / var(T_plc)
          + w_Ia  * MSE(Ia_sim, Ia_plc) / var(Ia_plc)
          + w_gap * MSE(gap_sim, gap_plc) / var(gap_plc)

    Each term is normalized by the PLC signal's variance so that
    weights are interpretable regardless of unit scales.

    Uses scipy.optimize.differential_evolution with local polish.

    Args:
        plc_data: Parsed PLC recording.
        config: Machine configuration.
        material: Material properties.
        device: Compute device (None = auto-detect CUDA).
        sim_duration_s: How much of the PLC to fit (None = full).
        w_temperature: Weight for temperature MSE.
        w_anode_current: Weight for Ia MSE.
        w_gap: Weight for electrode gap MSE.
        n_compare_points: Resampled comparison points.
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
        w_anode_current: float = 1.0,
        w_gap: float = 1.0,
        n_compare_points: int = 50,
    ):
        self._plc = plc_data
        self._config = config or MachineConfig()
        self._material = material or MaterialProperties()
        self._device = device
        self._w_T = w_temperature
        self._w_Ia = w_anode_current
        self._w_gap = w_gap
        self._n_pts = n_compare_points
        self._sim_duration = sim_duration_s or plc_data.duration_s

        # Recipe from PLC data
        gap_set = float(np.median(plc_data.electrode_set_mm))
        speed = float(np.median(plc_data.conv_speed_m_per_min))
        overrides = recipe_overrides or {}
        self._recipe = Recipe(
            name="calibration", recipe_number=0,
            electrode_gap_mm=overrides.get("electrode_gap_mm", gap_set),
            belt_speed_m_per_min=overrides.get("belt_speed_m_per_min", speed),
            mrh_amps=overrides.get("mrh_amps", plc_data.ia_limit_2),
            mrl_amps=overrides.get("mrl_amps", plc_data.ia_limit_1),
        )

        # Resample PLC data to comparison grid
        self._plc_times = np.linspace(0, self._sim_duration, n_compare_points)
        self._plc_temp = np.interp(
            self._plc_times, plc_data.time_s, plc_data.product_temp_c,
        )
        self._plc_ia = np.interp(
            self._plc_times, plc_data.time_s, plc_data.anode_current_a,
        )
        self._plc_gap = np.interp(
            self._plc_times, plc_data.time_s, plc_data.electrode_act_mm,
        )

        # Normalization: variance of each PLC signal (guard against zero)
        self._var_T = max(float(np.var(self._plc_temp)), 1.0)
        self._var_Ia = max(float(np.var(self._plc_ia)), 0.01)
        self._var_gap = max(float(np.var(self._plc_gap)), 1.0)

        # Persistent simulator (reused across evaluations)
        self._sim: Optional[GP15Simulator] = None
        self._eval_count = 0

    def _get_or_create_sim(
        self,
        coupling: float,
        k_evap: float,
        gap_rate: float,
    ) -> GP15Simulator:
        """Get a simulator, creating once and resetting on reuse.

        On first call: constructs the full GP15Simulator with geometry,
        grid, and solver allocations.

        On subsequent calls: uses ``CoupledSimulator.update_parameters()``
        to propagate new values to all sub-solvers (single source of
        truth), then ``reset()`` to zero fields without re-allocating.
        """
        if self._sim is None:
            # First call: full construction (geometry + arrays)
            config = MachineConfig()
            config.oscillator_coupling_factor = coupling
            mat = self._make_material(k_evap)

            self._sim = GP15Simulator(
                config=config, material=mat,
                device=self._device,
                enable_controller=True,
                enable_corrections=False,
                use_tvd=False,
            )
            self._sim.load_recipe(self._recipe)
            self._sim._ensure_initialized()
            self._sim._sim.controller.gap_adjust_rate_mm_s = gap_rate
        else:
            # Reuse: single-point parameter update + field reset
            self._sim._sim.update_parameters(
                coupling_factor=coupling,
                k_evap=k_evap,
                gap_adjust_rate=gap_rate,
            )
            self._sim._sim.controller.load_recipe(self._recipe)
            self._sim._sim.controller.start()
            self._sim._sim.conveyor.start(
                speed_m_per_min=self._recipe.belt_speed_m_per_min,
            )
            self._sim._sim.reset()

        return self._sim

    def _make_material(self, k_evap: float) -> MaterialProperties:
        m = self._material
        return MaterialProperties(
            name=m.name,
            initial_moisture_wb=m.initial_moisture_wb,
            initial_temperature_c=m.initial_temperature_c,
            bed_depth_m=m.bed_depth_m,
            k_evap=k_evap,
            T_evap_threshold_c=m.T_evap_threshold_c,
            dielectric_loss_coeffs=m.dielectric_loss_coeffs,
            dielectric_const_coeffs=m.dielectric_const_coeffs,
            rho_solid=m.rho_solid,
            bed_porosity=m.bed_porosity,
        )

    def _simulate(
        self, coupling: float, k_evap: float, gap_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one simulation. Returns (T_sim, Ia_sim, gap_sim)."""
        sim = self._get_or_create_sim(coupling, k_evap, gap_rate)

        try:
            result = sim._sim.run(
                duration_s=self._sim_duration,
                recipe=self._recipe,
                adaptive_dt=True,
            )
        except Exception as e:
            logger.warning(f"Simulation failed (k={coupling:.4f}, "
                           f"k_evap={k_evap:.2e}, gap_rate={gap_rate:.4f}): {e}")
            n = len(self._plc_times)
            return np.full(n, 100.0), np.full(n, 0.0), np.full(n, 300.0)

        ts = result.time_series
        if not ts.get("time_s") or len(ts["time_s"]) < 2:
            n = len(self._plc_times)
            return np.full(n, 100.0), np.full(n, 0.0), np.full(n, 300.0)

        t_sim = np.array(ts["time_s"])
        T_sim = np.interp(self._plc_times, t_sim, ts["T_outfeed_c"])
        Ia_sim = np.interp(self._plc_times, t_sim, ts["anode_current_a"])
        gap_sim = np.interp(self._plc_times, t_sim, ts["electrode_gap_mm"])

        return T_sim, Ia_sim, gap_sim

    def _compute_loss(
        self,
        T_sim: np.ndarray,
        Ia_sim: np.ndarray,
        gap_sim: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute normalized weighted loss components.

        Returns (total, L_T, L_Ia, L_gap).
        """
        L_T = float(np.mean((T_sim - self._plc_temp) ** 2)) / self._var_T
        L_Ia = float(np.mean((Ia_sim - self._plc_ia) ** 2)) / self._var_Ia
        L_gap = float(np.mean((gap_sim - self._plc_gap) ** 2)) / self._var_gap
        total = self._w_T * L_T + self._w_Ia * L_Ia + self._w_gap * L_gap
        return total, L_T, L_Ia, L_gap

    def _objective(self, params: np.ndarray) -> float:
        """Objective function for scipy optimizer."""
        coupling, k_evap, gap_rate = float(params[0]), float(params[1]), float(params[2])
        self._eval_count += 1

        T_sim, Ia_sim, gap_sim = self._simulate(coupling, k_evap, gap_rate)
        total, L_T, L_Ia, L_gap = self._compute_loss(T_sim, Ia_sim, gap_sim)

        if self._eval_count % 10 == 0:
            print(f"  eval {self._eval_count:3d}: "
                  f"k={coupling:.4f} k_evap={k_evap:.2e} "
                  f"gap_rate={gap_rate:.4f}  "
                  f"L_T={L_T:.3f} L_Ia={L_Ia:.3f} L_gap={L_gap:.3f} "
                  f"total={total:.3f}")

        return total

    def run(self, maxiter: int = 30, seed: int = 42) -> CalibrationResult:
        """Run calibration with differential evolution + local polish.

        Args:
            maxiter: Maximum DE generations.
            seed: Random seed.

        Returns:
            :class:`CalibrationResult` with optimized parameters.
        """
        try:
            from scipy.optimize import differential_evolution
        except ImportError:
            raise ImportError("Calibration requires scipy: pip install scipy")

        bounds = [
            (0.10, 0.40),     # oscillator_coupling_factor
            (1e-6, 5e-4),     # k_evap
            (0.005, 1.0),     # gap_adjust_rate_mm_s
        ]

        dev_name = self._device or "auto-detect"
        print(f"Calibration: {self._plc.n_samples} PLC samples, "
              f"{self._sim_duration:.0f} s, device={dev_name}")
        print(f"  Normalization: var_T={self._var_T:.1f}, "
              f"var_Ia={self._var_Ia:.4f}, var_gap={self._var_gap:.1f}")
        print(f"  Weights: w_T={self._w_T}, w_Ia={self._w_Ia}, w_gap={self._w_gap}")
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
            tol=0.005,
            callback=_callback,
            disp=True,
            polish=True,      # local Nelder-Mead refinement at the end
            popsize=15,       # robust population (default = 15*n_params)
        )

        best = result.x
        # Final evaluation reuses the warm simulator (reset via _get_or_create_sim)
        T_best, Ia_best, gap_best = self._simulate(best[0], best[1], best[2])
        total, L_T, L_Ia, L_gap = self._compute_loss(T_best, Ia_best, gap_best)

        # ── Sensitivity analysis (finite differences at optimum) ──
        sensitivity = self._compute_sensitivity(best, total)

        return CalibrationResult(
            oscillator_coupling_factor=float(best[0]),
            k_evap=float(best[1]),
            gap_adjust_rate_mm_s=float(best[2]),
            loss_total=float(result.fun),
            loss_temperature=L_T,
            loss_anode_current=L_Ia,
            loss_gap=L_gap,
            n_evaluations=self._eval_count,
            n_iterations=result.nit,
            converged=result.success,
            sensitivity=sensitivity,
            history=history,
        )

    def _compute_sensitivity(
        self, x_opt: np.ndarray, f_opt: float,
    ) -> Dict[str, float]:
        """Finite-difference sensitivity dL/dp at the optimum.

        Reports how much the loss changes per unit change in each
        parameter.  Large sensitivity = well-constrained by the data.
        Small sensitivity = sloppy (data doesn't constrain it).

        Perturbations are clamped to the optimization bounds to
        avoid biased gradient estimates near bound edges.
        """
        names = ["coupling_factor", "k_evap", "gap_rate_mm_s"]
        eps = [0.005, 1e-5, 0.005]
        bounds = [(0.10, 0.40), (1e-6, 5e-4), (0.005, 1.0)]
        sensitivity = {}

        for i, (name, h, (lo, hi)) in enumerate(zip(names, eps, bounds)):
            x_p = x_opt.copy()
            x_m = x_opt.copy()
            x_p[i] = min(x_opt[i] + h, hi)
            x_m[i] = max(x_opt[i] - h, lo)
            actual_h = x_p[i] - x_m[i]

            if actual_h < 1e-12:
                sensitivity[name] = 0.0
                continue

            f_p = self._objective(x_p)
            f_m = self._objective(x_m)
            sensitivity[name] = float((f_p - f_m) / actual_h)

        return sensitivity

    def evaluate_current(self) -> Dict[str, float]:
        """Evaluate the current parameter values against PLC data."""
        coupling = self._config.oscillator_coupling_factor
        k_evap = self._material.k_evap
        gap_rate = get_calibration_defaults()[2]

        T_sim, Ia_sim, gap_sim = self._simulate(coupling, k_evap, gap_rate)
        total, L_T, L_Ia, L_gap = self._compute_loss(T_sim, Ia_sim, gap_sim)

        return {
            "coupling": coupling, "k_evap": k_evap, "gap_rate": gap_rate,
            "loss_T": L_T, "loss_Ia": L_Ia, "loss_gap": L_gap,
            "loss_total": total,
            "T_sim_final": float(T_sim[-1]),
            "T_plc_final": float(self._plc_temp[-1]),
            "Ia_sim_final": float(Ia_sim[-1]),
            "Ia_plc_final": float(self._plc_ia[-1]),
            "gap_sim_final": float(gap_sim[-1]),
            "gap_plc_final": float(self._plc_gap[-1]),
        }
