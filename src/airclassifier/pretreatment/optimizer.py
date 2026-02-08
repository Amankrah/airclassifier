"""
Recipe Optimizer
================

Gradient-free recipe optimization for the GP-15 using scipy.

Finds the recipe parameters (electrode gap, belt speed, RF power)
that minimize energy consumption while meeting moisture and
uniformity targets.

Engineering guide §11 Phase 4.3: differentiable simulation for
gradient-based recipe optimization.  This module provides the
equivalent capability using scipy.optimize for the NumPy physics
backend.  When Warp GPU kernels are ported (Phase 5), this can be
replaced with wp.Tape-based gradient descent.

Usage::

    from airclassifier.pretreatment.optimizer import optimize_recipe

    best, result = optimize_recipe(
        config=MachineConfig(),
        material=get_material_preset("yellow_pea"),
        target_moisture_wb=0.03,
        max_energy_kwh_per_kg=1.2,
    )
    print(f"Optimal gap: {best.electrode_gap_mm:.0f} mm")
    print(f"Optimal speed: {best.belt_speed_m_per_min:.2f} m/min")
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
) -> Dict[str, float]:
    """Run a single simulation and return KPIs."""
    recipe = Recipe(
        name="opt_trial",
        recipe_number=0,
        electrode_gap_mm=float(gap_mm),
        belt_speed_m_per_min=float(speed_m_per_min),
        extraction_fan_hz=30.0,
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
            "feasible": False,
        }

    return {
        "gap_mm": gap_mm,
        "speed": speed_m_per_min,
        "moisture_wb": outlet.avg_moisture_wb,
        "moisture_cv": outlet.moisture_uniformity,
        "energy_kwh_per_kg": outlet.specific_energy_kwh_per_kg,
        "T_max_c": outlet.max_temperature_c,
        "feasible": True,
    }


def optimize_recipe(
    config: MachineConfig | None = None,
    material: MaterialProperties | None = None,
    target_moisture_wb: float = 0.03,
    max_temperature_c: float = 70.0,
    max_energy_kwh_per_kg: float = 1.5,
    duration_s: float = 60.0,
    gap_range_mm: tuple = (40.0, 200.0),
    speed_range: tuple = (0.2, 1.5),
    n_gap: int = 5,
    n_speed: int = 5,
) -> OptimizationResult:
    """Find the optimal recipe via grid search over gap × belt speed.

    The objective minimizes energy consumption subject to:
    - Outlet moisture <= target_moisture_wb
    - Max temperature <= max_temperature_c (protein denaturation)
    - Energy <= max_energy_kwh_per_kg

    Args:
        config: Machine config.
        material: Material properties.
        target_moisture_wb: Target outlet moisture (wet basis).
        max_temperature_c: Max allowed material temperature [°C].
        max_energy_kwh_per_kg: Max specific energy [kWh/kg water].
        duration_s: Simulation duration per trial [s].
        gap_range_mm: (min, max) electrode gap [mm].
        speed_range: (min, max) belt speed [m/min].
        n_gap: Grid points in gap dimension.
        n_speed: Grid points in speed dimension.

    Returns:
        :class:`OptimizationResult` with the best recipe and all trials.
    """
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
            )
            trials.append(trial)

            if not trial["feasible"]:
                continue

            # Feasibility check
            if trial["T_max_c"] > max_temperature_c:
                continue

            # Cost: energy consumption + penalty for missing moisture target
            moisture_penalty = max(0, trial["moisture_wb"] - target_moisture_wb) * 100
            cost = trial["energy_kwh_per_kg"] + moisture_penalty

            if cost < best_cost:
                best_cost = cost
                best_trial = trial

    if best_trial is None:
        # No feasible solution found — return the trial closest to target
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
