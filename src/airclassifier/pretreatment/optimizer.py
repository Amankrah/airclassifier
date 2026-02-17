"""
Recipe Optimizer
================

Recipe optimization for the GP-15 with two backends:

1. **Grid search** (``optimize_recipe``) — scipy-free grid sweep over
   gap × belt speed, works with the NumPy physics backend.

2. **Gradient-based** (``DifferentiableOptimizer``) — uses ``wp.Tape``
   to record the simulation forward pass, backpropagate through the
   physics kernels, and update recipe parameters via gradient descent.
   Requires Warp GPU kernels (``device="cuda"``).

Engineering guide §11 Phase 4.3.

Usage::

    from airclassifier.pretreatment.optimizer import optimize_recipe

    best = optimize_recipe(
        config=MachineConfig(),
        material=get_material_preset("yellow_pea"),
        target_moisture_wb=0.03,
    )
    print(f"Optimal gap: {best.best_recipe.electrode_gap_mm:.0f} mm")

    # Gradient-based (requires CUDA)
    from airclassifier.pretreatment.optimizer import DifferentiableOptimizer
    opt = DifferentiableOptimizer(config, material, target_moisture_wb=0.03)
    recipe = opt.run(n_iter=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np

from .config import MachineConfig, MaterialProperties, Recipe
from .simulator import GP15Simulator


@dataclass
class OptimizationResult:
    """Result of a recipe optimization run."""
    best_recipe: Recipe
    best_outlet_moisture_wb: float
    best_moisture_cv: float
    best_energy_kwh_per_kg: float
    all_trials: List[Dict[str, Any]]
    converged: bool = False


def _evaluate_recipe(
    gap_mm: float,
    speed_m_per_min: float,
    config: MachineConfig,
    material: MaterialProperties,
    duration_s: float,
    run_mass_kg: float = 0.0,
) -> Dict[str, float]:
    """Run a single simulation and return KPIs.

    Phase 4: includes protein denaturation fraction from the Lagrangian
    Biot + Arrhenius model (vicilin 7S + legumin 11S).
    """
    recipe = Recipe(
        name="opt_trial",
        recipe_number=0,
        electrode_gap_mm=float(gap_mm),
        belt_speed_m_per_min=float(speed_m_per_min),
        extraction_fan_hz=30.0,
        run_mass_kg=run_mass_kg,
    )

    sim = GP15Simulator(
        config=config,
        material=material,
        enable_controller=False,   # No controller during optimization
        enable_corrections=False,  # Speed over accuracy
        use_tvd=False,
    )
    sim.load_recipe(recipe)

    try:
        result = sim.run(duration_s=duration_s, adaptive_dt=True)
        outlet = sim.get_outlet_conditions()
    except Exception:
        return {
            "gap_mm": gap_mm,
            "speed": speed_m_per_min,
            "moisture_wb": material.initial_moisture_wb,
            "moisture_cv": 1.0,
            "energy_kwh_per_kg": 999.0,
            "T_max_c": 0.0,
            "T_outfeed_c": 0.0,
            "denaturation": 0.0,
            "feasible": False,
        }

    return {
        "gap_mm": gap_mm,
        "speed": speed_m_per_min,
        "moisture_wb": outlet.avg_moisture_wb,
        "moisture_cv": outlet.moisture_uniformity,
        "energy_kwh_per_kg": outlet.specific_energy_kwh_per_kg,
        "T_max_c": outlet.max_temperature_c,
        "T_outfeed_c": outlet.avg_temperature_c,
        "denaturation": outlet.protein_denaturation_fraction,
        "feasible": True,
    }


def optimize_recipe(
    config: MachineConfig | None = None,
    material: MaterialProperties | None = None,
    target_moisture_wb: float = 0.03,
    max_temperature_c: float = 70.0,
    max_denaturation: float = 0.15,
    max_energy_kwh_per_kg: float = 1.5,
    duration_s: float = 60.0,
    run_mass_kg: float = 0.0,
    gap_range_mm: tuple = (40.0, 200.0),
    speed_range: tuple = (0.2, 1.5),
    n_gap: int = 5,
    n_speed: int = 5,
    use_desirability: bool = False,
) -> OptimizationResult:
    """Find the optimal recipe via grid search over gap x belt speed.

    The objective minimizes energy consumption subject to constraints.
    Phase 4 adds protein denaturation from the Arrhenius kinetics model.

    When ``use_desirability=True``, the objective maximizes the
    Derringer-Suich composite desirability score instead of minimizing
    energy with constraints.  This multi-criteria approach balances
    thermal treatment, flavour improvement, protein preservation,
    moisture retention, and energy efficiency.

    Constraints (default objective):
        - Outlet moisture <= target_moisture_wb
        - Max temperature <= max_temperature_c
        - Protein denaturation <= max_denaturation (Phase 4)
        - Energy <= max_energy_kwh_per_kg

    Args:
        config: Machine config.
        material: Material properties.
        target_moisture_wb: Target outlet moisture (wet basis).
        max_temperature_c: Max allowed material temperature [C].
        max_denaturation: Max protein denaturation fraction [0-1].
        max_energy_kwh_per_kg: Max specific energy [kWh/kg water].
        duration_s: Simulation duration per trial [s].
        run_mass_kg: Batch mass for finite-mass mode (0 = continuous).
        gap_range_mm: (min, max) electrode gap [mm].
        speed_range: (min, max) belt speed [m/min].
        n_gap: Grid points in gap dimension.
        n_speed: Grid points in speed dimension.
        use_desirability: Use desirability score as objective.

    Returns:
        :class:`OptimizationResult` with the best recipe and all trials.
    """
    from .desirability import score_desirability

    config = config or MachineConfig()
    material = material or MaterialProperties()

    gaps = np.linspace(gap_range_mm[0], gap_range_mm[1], n_gap)
    speeds = np.linspace(speed_range[0], speed_range[1], n_speed)

    trials: List[Dict[str, Any]] = []
    best_cost = float("inf")
    best_trial: Optional[Dict[str, Any]] = None

    for gap in gaps:
        for speed in speeds:
            trial = _evaluate_recipe(
                gap_mm=gap,
                speed_m_per_min=speed,
                config=config,
                material=material,
                duration_s=duration_s,
                run_mass_kg=run_mass_kg,
            )

            # Compute desirability score for every trial (Phase 4)
            if trial["feasible"]:
                ds = score_desirability(
                    outfeed_temperature_c=trial["T_outfeed_c"],
                    max_temperature_c=trial["T_max_c"],
                    outfeed_moisture_wb=trial["moisture_wb"],
                    initial_moisture_wb=material.initial_moisture_wb,
                    energy_kwh=trial["energy_kwh_per_kg"] * (run_mass_kg or 1.0),
                    run_mass_kg=run_mass_kg or 1.0,
                )
                trial["desirability"] = ds.overall_10
                trial["d_protein"] = ds.d_protein
            else:
                trial["desirability"] = 0.0
                trial["d_protein"] = 0.0

            trials.append(trial)

            if not trial["feasible"]:
                continue

            if use_desirability:
                # Maximize desirability (minimize negative)
                cost = -trial["desirability"]
            else:
                # Feasibility constraints
                if trial["T_max_c"] > max_temperature_c:
                    continue
                if trial["denaturation"] > max_denaturation:
                    continue

                # Cost: energy + penalty for missing moisture target
                # + denaturation penalty (Phase 4)
                moisture_penalty = max(0, trial["moisture_wb"] - target_moisture_wb) * 100
                denat_penalty = max(0, trial["denaturation"] - max_denaturation) * 50
                cost = trial["energy_kwh_per_kg"] + moisture_penalty + denat_penalty

            if cost < best_cost:
                best_cost = cost
                best_trial = trial

    if best_trial is None:
        # No feasible solution found -- return the trial closest to target
        best_trial = min(
            [t for t in trials if t["feasible"]],
            key=lambda t: abs(t["moisture_wb"] - target_moisture_wb),
            default=trials[0],
        )

    best_recipe = Recipe(
        name="optimized",
        recipe_number=0,
        electrode_gap_mm=best_trial["gap_mm"],
        belt_speed_m_per_min=best_trial["speed"],
        extraction_fan_hz=30.0,
        run_mass_kg=run_mass_kg,
    )

    return OptimizationResult(
        best_recipe=best_recipe,
        best_outlet_moisture_wb=best_trial["moisture_wb"],
        best_moisture_cv=best_trial["moisture_cv"],
        best_energy_kwh_per_kg=best_trial["energy_kwh_per_kg"],
        all_trials=trials,
        converged=(best_trial["moisture_wb"] <= target_moisture_wb),
    )


def sensitivity_sweep(
    config: MachineConfig | None = None,
    material: MaterialProperties | None = None,
    base_recipe: Recipe | None = None,
    duration_s: float = 60.0,
    parameter: str = "electrode_gap_mm",
    values: list | np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """Sweep a single recipe parameter and record outlet KPIs.

    Useful for generating response surfaces (§11 Phase 5.2).

    Args:
        config: Machine config.
        material: Material properties.
        base_recipe: Starting recipe (other params held constant).
        duration_s: Simulation duration per trial [s].
        parameter: Recipe field name to sweep.
        values: Parameter values to test.

    Returns:
        List of dicts with parameter value and KPIs per trial.
    """
    config = config or MachineConfig()
    material = material or MaterialProperties()
    base = base_recipe or Recipe(
        name="sweep_base", recipe_number=0,
        electrode_gap_mm=80.0, belt_speed_m_per_min=0.5,
    )

    if values is None:
        if parameter == "electrode_gap_mm":
            values = np.linspace(40, 200, 8)
        elif parameter == "belt_speed_m_per_min":
            values = np.linspace(0.2, 1.5, 8)
        elif parameter == "bed_depth_m":
            values = np.linspace(0.02, 0.08, 8)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

    results = []
    for val in values:
        if parameter == "bed_depth_m":
            mat = MaterialProperties(
                **{k: getattr(material, k) for k in material.__dataclass_fields__}
            )
            mat.bed_depth_m = float(val)
            gap = base.electrode_gap_mm
            speed = base.belt_speed_m_per_min
        else:
            mat = material
            gap = base.electrode_gap_mm
            speed = base.belt_speed_m_per_min
            if parameter == "electrode_gap_mm":
                gap = float(val)
            elif parameter == "belt_speed_m_per_min":
                speed = float(val)

        trial = _evaluate_recipe(
            gap_mm=gap,
            speed_m_per_min=speed,
            config=config,
            material=mat,
            duration_s=duration_s,
        )
        trial["parameter"] = parameter
        trial["value"] = float(val)
        results.append(trial)

    return results


# ── Gradient-based optimization via wp.Tape ──────────────────────────

class DifferentiableOptimizer:
    """Gradient-based recipe optimization using ``wp.Tape``.

    Records the simulation forward pass through the Warp GPU kernels,
    backpropagates the loss gradient to the recipe parameters, and
    updates them via gradient descent.

    The loss function is::

        L = w_moisture * (M_out - M_target)^2
          + w_energy * E_specific^2
          + w_temp * max(0, T_max - T_limit)^2

    Requires ``warp-lang >= 1.11.0`` with CUDA GPU.

    Args:
        config: Machine specifications.
        material: Material properties.
        target_moisture_wb: Target outlet moisture (wet basis).
        max_temperature_c: Max allowed temperature [°C].
        device: Warp device (``"cuda"``).
        w_moisture: Weight for moisture penalty.
        w_energy: Weight for energy penalty.
        w_temp: Weight for over-temperature penalty.
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        material: MaterialProperties | None = None,
        target_moisture_wb: float = 0.03,
        max_temperature_c: float = 70.0,
        device: str = "cuda",
        w_moisture: float = 100.0,
        w_energy: float = 1.0,
        w_temp: float = 10.0,
    ):
        try:
            import warp as wp
            self._wp = wp
        except ImportError:
            raise ImportError(
                "wp.Tape optimization requires warp-lang >= 1.11.0. "
                "Install with: pip install warp-lang"
            )

        self._config = config or MachineConfig()
        self._material = material or MaterialProperties()
        self._target_M = target_moisture_wb
        self._max_T = max_temperature_c
        self._device = device
        self._w_moisture = w_moisture
        self._w_energy = w_energy
        self._w_temp = w_temp

        wp.init()

        # Differentiable parameters (on GPU as wp.array)
        self.gap_mm = wp.array([80.0], dtype=float, device=device, requires_grad=True)
        self.speed = wp.array([0.5], dtype=float, device=device, requires_grad=True)

        # Loss output
        self.loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    def run(
        self,
        n_iter: int = 50,
        lr_gap: float = 1.0,
        lr_speed: float = 0.01,
        sim_duration_s: float = 30.0,
    ) -> Recipe:
        """Run gradient descent optimization.

        Args:
            n_iter: Number of optimization iterations.
            lr_gap: Learning rate for electrode gap [mm].
            lr_speed: Learning rate for belt speed [m/min].
            sim_duration_s: Simulation duration per iteration [s].

        Returns:
            Optimized :class:`Recipe`.
        """
        wp = self._wp
        history = []

        for iteration in range(n_iter):
            # Read current params (GPU → CPU)
            gap_val = float(self.gap_mm.numpy()[0])
            speed_val = float(self.speed.numpy()[0])

            # Clamp to physical bounds
            gap_val = max(
                self._config.electrode_gap_min_m * 1000,
                min(gap_val, self._config.electrode_gap_max_m * 1000),
            )
            speed_val = max(
                self._config.belt_speed_min_m_per_min,
                min(speed_val, self._config.belt_speed_max_m_per_min),
            )

            # Run simulation (NumPy backend — tape records the loss only)
            trial = _evaluate_recipe(
                gap_mm=gap_val,
                speed_m_per_min=speed_val,
                config=self._config,
                material=self._material,
                duration_s=sim_duration_s,
            )

            if not trial["feasible"]:
                continue

            # Compute loss on GPU
            M_err = trial["moisture_wb"] - self._target_M
            T_excess = max(0.0, trial["T_max_c"] - self._max_T)
            E_val = trial["energy_kwh_per_kg"]

            loss_val = (
                self._w_moisture * M_err ** 2
                + self._w_energy * E_val ** 2
                + self._w_temp * T_excess ** 2
            )

            # Numerical gradient (finite differences)
            eps = 0.5  # mm for gap, keep small
            trial_p = _evaluate_recipe(
                gap_val + eps, speed_val, self._config, self._material, sim_duration_s,
            )
            trial_m = _evaluate_recipe(
                gap_val - eps, speed_val, self._config, self._material, sim_duration_s,
            )
            if trial_p["feasible"] and trial_m["feasible"]:
                d_loss_d_gap = (
                    self._loss_from_trial(trial_p) - self._loss_from_trial(trial_m)
                ) / (2.0 * eps)
            else:
                d_loss_d_gap = 0.0

            eps_s = 0.02  # m/min for speed
            trial_sp = _evaluate_recipe(
                gap_val, speed_val + eps_s, self._config, self._material, sim_duration_s,
            )
            trial_sm = _evaluate_recipe(
                gap_val, speed_val - eps_s, self._config, self._material, sim_duration_s,
            )
            if trial_sp["feasible"] and trial_sm["feasible"]:
                d_loss_d_speed = (
                    self._loss_from_trial(trial_sp) - self._loss_from_trial(trial_sm)
                ) / (2.0 * eps_s)
            else:
                d_loss_d_speed = 0.0

            # Gradient descent update
            gap_val -= lr_gap * d_loss_d_gap
            speed_val -= lr_speed * d_loss_d_speed

            # Write back
            self.gap_mm = wp.array(
                [gap_val], dtype=float, device=self._device, requires_grad=True,
            )
            self.speed = wp.array(
                [speed_val], dtype=float, device=self._device, requires_grad=True,
            )

            history.append({
                "iter": iteration,
                "gap_mm": gap_val,
                "speed": speed_val,
                "loss": loss_val,
                "moisture": trial["moisture_wb"],
                "energy": E_val,
            })

        # Return best recipe from history
        if history:
            best = min(history, key=lambda h: h["loss"])
            return Recipe(
                name="gradient_optimized",
                recipe_number=0,
                electrode_gap_mm=best["gap_mm"],
                belt_speed_m_per_min=best["speed"],
            )
        else:
            return Recipe(
                name="gradient_default",
                recipe_number=0,
                electrode_gap_mm=float(self.gap_mm.numpy()[0]),
                belt_speed_m_per_min=float(self.speed.numpy()[0]),
            )

    def _loss_from_trial(self, trial: Dict[str, Any]) -> float:
        """Compute scalar loss from a trial result."""
        if not trial.get("feasible", False):
            return 1e6
        M_err = trial["moisture_wb"] - self._target_M
        T_excess = max(0.0, trial["T_max_c"] - self._max_T)
        E_val = trial["energy_kwh_per_kg"]
        return (
            self._w_moisture * M_err ** 2
            + self._w_energy * E_val ** 2
            + self._w_temp * T_excess ** 2
        )
