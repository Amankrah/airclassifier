#!/usr/bin/env python3
"""
Classification System Optimizer
================================

Finds the best operating configuration for the air classifier to maximize
protein recovery, starch yield, or a combined objective.

**Default mode is wheel-only** (no venturi/zigzag preclassification), which
avoids the particle trapping problem in the zigzag at high d50.  Use
``--full-system`` to include venturi + zigzag preclassification.

Three optimization strategies:

1. **Grid Search** (``--strategy grid``) — Exhaustive sweep over blower RPM ×
   wheel RPM. Simple, deterministic, good for 2-3 parameters. Default.

2. **Bayesian Optimization** (``--strategy bayesian``) — Uses Optuna's
   Tree-of-Parzen-Estimators to intelligently sample the parameter space.
   Best for high-dimensional searches (5+ parameters). Requires ``pip install optuna``.

3. **Latin Hypercube** (``--strategy lhs``) — Space-filling design that
   covers the search space more evenly than random sampling. Good for
   building response surface models with fewer evaluations.

Objective Functions (``--objective``):

- ``protein_recovery``:  Maximize (Cy3 + Bag) / total_feed
- ``starch_yield``:      Maximize (Zc + Wc) / total_feed
- ``combined``:          Maximize w_protein * protein_recovery + w_starch * starch_yield
                         (weights via --w-protein, --w-starch)
- ``separation_efficiency``: Maximize (fines / (fines + coarse))
- ``protein_purity``:    Maximize fraction of protein particles in Cy3 + Bag

Usage::

    # Quick grid search: wheel-only mode (default), 4×4 = 16 trials
    python examples/optimize_classification.py --material yellow_pea

    # Fine grid with more points
    python examples/optimize_classification.py --material yellow_pea \\
        --blower-rpm-range 400 700 --wheel-rpm-range 1000 4000 \\
        --n-blower 6 --n-wheel 6

    # Bayesian optimization (50 trials, higher-dimensional)
    python examples/optimize_classification.py --material yellow_pea \\
        --strategy bayesian --n-trials 50

    # Optimize with recirculation
    python examples/optimize_classification.py --material yellow_pea \\
        --recirculate cy1 --passes 2

    # Full system (venturi + zigzag + wheel)
    python examples/optimize_classification.py --material yellow_pea --full-system

    # Multi-objective: protein purity
    python examples/optimize_classification.py --material yellow_pea \\
        --objective protein_purity

    # Custom weights for combined objective
    python examples/optimize_classification.py --material yellow_pea \\
        --objective combined --w-protein 3.0 --w-starch 1.0

    # Fewer particles for faster search (trade accuracy for speed)
    python examples/optimize_classification.py --material yellow_pea \\
        --particles 10000 --time 120
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class TrialConfig:
    """A single configuration point to evaluate."""
    blower_rpm: float
    wheel_rpm: float
    wheel_only: bool = True
    passes: int = 1
    recirculate: Optional[List[str]] = None
    recirculate_wheel_rpm: Optional[float] = None
    recirculate_time: Optional[float] = None
    attrition: float = 0.10
    bypass_ratio: float = 0.0
    throat_diameter_mm: Optional[float] = None


@dataclass
class TrialResult:
    """Result from a single simulation trial."""
    config: TrialConfig
    # Raw counts
    zigzag_coarse: int = 0
    wheel_coarse: int = 0
    cyclone_1: int = 0
    cyclone_2: int = 0
    cyclone_3_protein: int = 0
    bagfilter: int = 0
    escaped: int = 0
    active: int = 0
    total_feed: int = 0
    # Derived metrics
    protein_recovery: float = 0.0       # (Cy3 + Bag) / total_feed
    starch_yield: float = 0.0           # (Zc + Wc) / total_feed
    total_collection: float = 0.0       # 1 - (active + escaped) / total_feed
    separation_efficiency: float = 0.0  # fines / (fines + coarse)
    protein_purity: float = 0.0         # protein particles in Cy3+Bag / all in Cy3+Bag
    # Timing
    wall_time_s: float = 0.0
    early_stopped: bool = False
    feasible: bool = True
    error: Optional[str] = None


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    best_trial: TrialResult
    all_trials: List[TrialResult]
    objective: str
    strategy: str
    best_score: float
    wheel_only: bool = True
    material: str = "yellow_pea"
    total_wall_time_s: float = 0.0


# ─── Stdout suppression for noisy simulator output ──────────────────────────

@contextlib.contextmanager
def suppress_stdout():
    """Redirect stdout to devnull to silence simulator diagnostic prints."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ─── Simulation Runner ───────────────────────────────────────────────────────

# Check interval for early termination (in simulation seconds)
_EARLY_STOP_CHECK_INTERVAL_S = 30.0
# If collected counts don't change across two checks after feeding is done, stop
_EARLY_STOP_STALL_CHECKS = 2


def run_single_trial(
    trial_config: TrialConfig,
    material: str,
    num_particles: int,
    sim_time: float,
    dt: float,
    device: str,
    max_loading: float,
) -> TrialResult:
    """Run one complete classification simulation and return metrics.

    Supports both wheel-only mode (no venturi/zigzag — default) and
    full-system mode (venturi + zigzag + wheel).

    Includes early termination: if all particles are fed and no particles
    change state for two consecutive check intervals, the simulation stops
    early to save time.
    """
    from airclassifier.simulation.classification_flow_physics import (
        ClassificationFlowPhysicsSimulator,
        ClassificationFlowConfig,
        compute_venturi_physics_from_air_and_feed,
    )
    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly,
        ClassificationSystemParams,
    )
    from airclassifier.geometry.assembly.complete_system import (
        CompleteClassifierAssembly,
        CompleteSystemParams,
    )
    from airclassifier.simulation.airclass_flow_physics import (
        compute_air_to_venturi_flow,
        compute_blower_operating_point,
    )
    from airclassifier.simulation.feedclass_flow_physics import (
        compute_feed_to_venturi_flow,
        compute_venturi_max_throughput_kg_h,
    )
    from airclassifier.particles import FluidConfig, ParticleMaterial

    result = TrialResult(config=trial_config, total_feed=num_particles)
    t0 = time.time()

    try:
        # ── Air flow from blower RPM ──
        op = compute_blower_operating_point(trial_config.blower_rpm)
        Q_m3s = op["Q_m3_s"]

        # ── Material ──
        fluid = FluidConfig.air_at_stp()
        fraction = "whole" if material in ("yellow_pea", "faba_bean", "oat") else material
        mat = ParticleMaterial.create_food_powder(material, fraction)
        sd = mat.size_distribution
        particle_dia_m = getattr(sd, "d50", None) or (sd.d_min + sd.d_max) / 2.0
        particle_density = mat.density
        sphericity = getattr(mat, "sphericity", 0.75)

        # ── Geometry: wheel-only vs full system ──
        classification_params = ClassificationSystemParams()
        if trial_config.wheel_only:
            classification_params.use_preclassification = False
        if trial_config.throat_diameter_mm is not None:
            throat_m = trial_config.throat_diameter_mm / 1000.0
            classification_params.venturi_throat_ratio = (
                throat_m / classification_params.venturi_inlet_diameter
            )

        # ── Venturi capacity (still used for feed rate capping) ──
        venturi_max_kg_h = compute_venturi_max_throughput_kg_h(
            Q_m3s, max_loading_ratio=max_loading,
        )
        throughput = min(venturi_max_kg_h, 500.0)

        # ── Build complete system ──
        complete_params = CompleteSystemParams(
            air_flow_m3_h=Q_m3s * 3600.0,
            throughput_kg_h=throughput,
            include_feed_system=True,
            include_air_system=True,
            include_exhaust=False,
            include_ductwork=True,
            classification_params=classification_params,
        )

        # Suppress noisy assembly/simulator construction output
        with suppress_stdout():
            complete_assembly = CompleteClassifierAssembly(complete_params)

            # ── Air system ──
            air_result = compute_air_to_venturi_flow(
                complete_assembly, Q_m3s,
                rho=fluid.density, mu=fluid.dynamic_viscosity,
            )

            # ── Feed system ──
            solids_mass_flow_kg_s = throughput / 3600.0
            feed_result = compute_feed_to_venturi_flow(
                complete_assembly,
                volume_flow_air_m3_s=0.0,
                particle_diameter_m=particle_dia_m,
                particle_density_kg_m3=particle_density,
                rho_air=fluid.density,
                mu_air=fluid.dynamic_viscosity,
                solids_mass_flow_kg_s=solids_mass_flow_kg_s,
                sphericity=sphericity,
            )

            # ── Classification config ──
            assembly = complete_assembly.get_subsystem("classification")
            config = ClassificationFlowConfig.from_air_and_feed_results(
                air_result, feed_result, assembly,
                solids_mass_flow_kg_s=solids_mass_flow_kg_s,
                num_particles_capacity=num_particles,
                simulation_time_s=sim_time,
                dt=dt,
                device=device,
                fluid_config=fluid,
                material=mat,
                bypass_ratio=trial_config.bypass_ratio,
                continuous_feeding=True,
                max_loading_ratio=max_loading,
                wheel_rpm=trial_config.wheel_rpm,
            )

            # ── Create simulator ──
            simulator = ClassificationFlowPhysicsSimulator(assembly, config)
            simulator.initialize_whole_flour_population(
                source=material, num_particles=num_particles,
            )

        # ── Recirculation setup ──
        num_passes = trial_config.passes
        recirculate_fractions = trial_config.recirculate or []
        recirc_time = trial_config.recirculate_time or sim_time

        # Feed residence time (for recirculation refeed timing)
        feed_res_time = feed_result.get("total_residence_time_s", 21.0)

        # ── Cumulative counters ──
        cumulative_counts = {
            "coarse": 0, "wheel_coarse": 0,
            "cyclone_1": 0, "cyclone_2": 0,
            "cyclone_3_protein": 0, "bagfilter": 0,
            "escaped": 0, "active": 0,
        }
        sep_to_frac = {
            "cyclone_1": "cy1", "cyclone_2": "cy2",
            "cyclone_3_protein": "cy3", "wheel_coarse": "wheel_coarse",
            "coarse": "zigzag_coarse", "bagfilter": "bagfilter",
        }

        # ── Run simulation passes ──
        for pass_num in range(1, num_passes + 1):
            current_sim_time = sim_time if pass_num == 1 else recirc_time
            n_steps = int(current_sim_time / config.dt)
            check_interval_steps = int(_EARLY_STOP_CHECK_INTERVAL_S / config.dt)

            # Early termination state
            prev_collected = None
            stall_count = 0

            for step_i in range(n_steps):
                simulator.step()

                # ── Early termination check ──
                if check_interval_steps > 0 and (step_i + 1) % check_interval_steps == 0:
                    # Only check after all particles are fed
                    all_fed = (
                        simulator.state.particles_fed
                        >= simulator.state.total_particles_to_feed
                    )
                    if all_fed:
                        snap = simulator.get_separation_counts()
                        collected_now = (
                            snap.get("coarse", 0)
                            + snap.get("wheel_coarse", 0)
                            + snap.get("cyclone_1", 0)
                            + snap.get("cyclone_2", 0)
                            + snap.get("cyclone_3_protein", 0)
                            + snap.get("bagfilter", 0)
                            + snap.get("escaped", 0)
                        )
                        if prev_collected is not None and collected_now == prev_collected:
                            stall_count += 1
                            if stall_count >= _EARLY_STOP_STALL_CHECKS:
                                result.early_stopped = True
                                break
                        else:
                            stall_count = 0
                        prev_collected = collected_now

            # Accumulate results
            pass_sep = simulator.get_separation_counts()
            for key in cumulative_counts:
                if key == "active":
                    continue
                frac_name = sep_to_frac.get(key)
                if frac_name in recirculate_fractions and pass_num < num_passes:
                    pass  # Will be re-processed
                else:
                    cumulative_counts[key] += pass_sep.get(key, 0)
            cumulative_counts["active"] += pass_sep.get("active", 0)

            # Recirculation
            if pass_num < num_passes and recirculate_fractions:
                with suppress_stdout():
                    particle_data = simulator.extract_collected_particles(
                        recirculate_fractions
                    )
                if particle_data["count"] == 0:
                    break

                next_wheel_rpm = trial_config.recirculate_wheel_rpm
                with suppress_stdout():
                    n_recirc = simulator.reinitialize_from_particles(
                        particle_data,
                        initial_velocity=None,
                        continuous_feeding=None,
                        wheel_rpm=next_wheel_rpm,
                        attrition_factor=trial_config.attrition,
                        attrition_min_diameter_m=5.0e-6,
                        skip_preclassification=trial_config.wheel_only,
                        feed_residence_time_s=feed_res_time,
                    )
                if n_recirc == 0:
                    break

        # ── Extract final counts ──
        result.zigzag_coarse = cumulative_counts["coarse"]
        result.wheel_coarse = cumulative_counts["wheel_coarse"]
        result.cyclone_1 = cumulative_counts["cyclone_1"]
        result.cyclone_2 = cumulative_counts["cyclone_2"]
        result.cyclone_3_protein = cumulative_counts["cyclone_3_protein"]
        result.bagfilter = cumulative_counts["bagfilter"]
        result.escaped = cumulative_counts["escaped"]
        result.active = cumulative_counts["active"]

        total = num_particles
        protein_rich = result.cyclone_3_protein + result.bagfilter
        starch_rich = result.zigzag_coarse + result.wheel_coarse
        fines = (result.cyclone_1 + result.cyclone_2
                 + result.cyclone_3_protein + result.bagfilter)
        coarse = result.zigzag_coarse + result.wheel_coarse

        result.protein_recovery = protein_rich / max(1, total)
        result.starch_yield = starch_rich / max(1, total)
        result.total_collection = 1.0 - (result.active + result.escaped) / max(1, total)
        result.separation_efficiency = fines / max(1, fines + coarse)

        # ── Protein purity (requires particle types) ──
        try:
            zones_np = simulator.get_zones()
            types_np = simulator.state.particle_types.numpy()[
                : simulator.state.particles_active
            ]
            # Cy3 zone = 57, Bagfilter zone = 60
            protein_outlet_mask = (zones_np == 57) | (zones_np == 60)
            if np.any(protein_outlet_mask):
                types_in_outlet = types_np[protein_outlet_mask]
                n_protein_particles = int(np.sum(types_in_outlet == 0))
                result.protein_purity = n_protein_particles / max(
                    1, int(np.sum(protein_outlet_mask))
                )
            else:
                result.protein_purity = 0.0
        except Exception:
            result.protein_purity = 0.0

    except Exception as e:
        result.feasible = False
        result.error = str(e)

    result.wall_time_s = time.time() - t0
    return result


# ─── Objective Functions ─────────────────────────────────────────────────────

def compute_score(
    trial: TrialResult,
    objective: str,
    w_protein: float = 2.0,
    w_starch: float = 1.0,
) -> float:
    """Compute a scalar score to maximize for the given objective."""
    if not trial.feasible:
        return -999.0

    if objective == "protein_recovery":
        return trial.protein_recovery
    elif objective == "starch_yield":
        return trial.starch_yield
    elif objective == "combined":
        return w_protein * trial.protein_recovery + w_starch * trial.starch_yield
    elif objective == "separation_efficiency":
        return trial.separation_efficiency
    elif objective == "protein_purity":
        # Combine purity with recovery to avoid trivially empty solutions
        return trial.protein_purity * 0.7 + trial.protein_recovery * 0.3
    else:
        raise ValueError(f"Unknown objective: {objective}")


# ─── Optimization Strategies ────────────────────────────────────────────────

def _format_trial_line(tr: TrialResult, score: float) -> str:
    """Format a single trial result line."""
    status = "OK" if tr.feasible else f"FAIL: {tr.error}"
    early = " [early]" if tr.early_stopped else ""
    return (f"{status}  "
            f"prot={tr.protein_recovery:.3f}  starch={tr.starch_yield:.3f}  "
            f"purity={tr.protein_purity:.3f}  score={score:.4f}  "
            f"({tr.wall_time_s:.1f}s{early})")


def grid_search(
    material: str,
    blower_range: Tuple[float, float],
    wheel_range: Tuple[float, float],
    n_blower: int,
    n_wheel: int,
    objective: str,
    w_protein: float,
    w_starch: float,
    num_particles: int,
    sim_time: float,
    dt: float,
    device: str,
    max_loading: float,
    wheel_only: bool,
    passes: int,
    recirculate: Optional[List[str]],
    recirculate_wheel_rpm: Optional[float],
    recirculate_time: Optional[float],
    attrition: float,
) -> OptimizationResult:
    """Exhaustive grid search over blower RPM × wheel RPM."""
    blower_rpms = np.linspace(blower_range[0], blower_range[1], n_blower)
    wheel_rpms = np.linspace(wheel_range[0], wheel_range[1], n_wheel)

    total_trials = n_blower * n_wheel
    trials: List[TrialResult] = []
    best_score = -float("inf")
    best_trial: Optional[TrialResult] = None

    mode_str = "WHEEL-ONLY" if wheel_only else "FULL SYSTEM (venturi+zigzag+wheel)"
    print(f"\n{'='*70}")
    print(f"GRID SEARCH OPTIMIZATION — {mode_str}")
    print(f"  Blower RPM: {blower_range[0]:.0f} – {blower_range[1]:.0f} ({n_blower} points)")
    print(f"  Wheel RPM:  {wheel_range[0]:.0f} – {wheel_range[1]:.0f} ({n_wheel} points)")
    print(f"  Total trials: {total_trials}")
    print(f"  Objective: {objective}")
    print(f"  Particles: {num_particles}, Time: {sim_time}s")
    if passes > 1:
        print(f"  Passes: {passes}, Recirculate: {recirculate}")
    print(f"{'='*70}\n")

    t_start = time.time()

    for i, blower_rpm in enumerate(blower_rpms):
        for j, wheel_rpm in enumerate(wheel_rpms):
            trial_num = i * n_wheel + j + 1
            tc = TrialConfig(
                blower_rpm=float(blower_rpm),
                wheel_rpm=float(wheel_rpm),
                wheel_only=wheel_only,
                passes=passes,
                recirculate=recirculate,
                recirculate_wheel_rpm=recirculate_wheel_rpm,
                recirculate_time=recirculate_time,
                attrition=attrition,
            )

            print(f"  [{trial_num:3d}/{total_trials}] "
                  f"Blower={blower_rpm:.0f} RPM, Wheel={wheel_rpm:.0f} RPM ... ",
                  end="", flush=True)

            tr = run_single_trial(
                tc, material, num_particles, sim_time, dt, device, max_loading,
            )
            trials.append(tr)

            score = compute_score(tr, objective, w_protein, w_starch)
            print(_format_trial_line(tr, score))

            if score > best_score:
                best_score = score
                best_trial = tr

    total_time = time.time() - t_start
    return OptimizationResult(
        best_trial=best_trial or trials[0],
        all_trials=trials,
        objective=objective,
        strategy="grid",
        best_score=best_score,
        wheel_only=wheel_only,
        material=material,
        total_wall_time_s=total_time,
    )


def bayesian_optimization(
    material: str,
    blower_range: Tuple[float, float],
    wheel_range: Tuple[float, float],
    objective: str,
    w_protein: float,
    w_starch: float,
    num_particles: int,
    sim_time: float,
    dt: float,
    device: str,
    max_loading: float,
    n_trials: int,
    wheel_only: bool,
    passes: int,
    recirculate: Optional[List[str]],
    recirculate_wheel_rpm: Optional[float],
    recirculate_time: Optional[float],
    attrition: float,
    optimize_passes: bool = False,
    optimize_recirculate: bool = False,
) -> OptimizationResult:
    """Bayesian optimization using Optuna's TPE sampler."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: Bayesian optimization requires 'optuna'. Install with:")
        print("       pip install optuna")
        sys.exit(1)

    trials: List[TrialResult] = []
    trial_counter = [0]

    mode_str = "WHEEL-ONLY" if wheel_only else "FULL SYSTEM"
    print(f"\n{'='*70}")
    print(f"BAYESIAN OPTIMIZATION (Optuna TPE) — {mode_str}")
    print(f"  Blower RPM: {blower_range[0]:.0f} – {blower_range[1]:.0f}")
    print(f"  Wheel RPM:  {wheel_range[0]:.0f} – {wheel_range[1]:.0f}")
    print(f"  Max trials: {n_trials}")
    print(f"  Objective: {objective}")
    print(f"  Particles: {num_particles}, Time: {sim_time}s")
    if optimize_passes:
        print(f"  Optimizing passes: 1-4")
    if optimize_recirculate:
        print(f"  Optimizing recirculation fractions")
    print(f"{'='*70}\n")

    t_start = time.time()

    def optuna_objective(trial: optuna.Trial) -> float:
        trial_counter[0] += 1
        blower_rpm = trial.suggest_float("blower_rpm", blower_range[0], blower_range[1])
        wheel_rpm = trial.suggest_float("wheel_rpm", wheel_range[0], wheel_range[1])

        # Optionally optimize discrete choices
        trial_passes = passes
        trial_recirculate = recirculate
        trial_recirc_wheel = recirculate_wheel_rpm
        trial_attrition = attrition

        if optimize_passes:
            trial_passes = trial.suggest_int("passes", 1, 4)
            if trial_passes > 1 and trial_recirculate is None:
                trial_recirculate = ["cy1"]  # Default

        if optimize_recirculate and trial_passes > 1:
            recirc_choice = trial.suggest_categorical(
                "recirculate", ["cy1", "cy2", "cy1+cy2"]
            )
            trial_recirculate = recirc_choice.split("+")
            trial_recirc_wheel = trial.suggest_float(
                "recirculate_wheel_rpm", wheel_range[0], wheel_range[1]
            )

        tc = TrialConfig(
            blower_rpm=blower_rpm,
            wheel_rpm=wheel_rpm,
            wheel_only=wheel_only,
            passes=trial_passes,
            recirculate=trial_recirculate,
            recirculate_wheel_rpm=trial_recirc_wheel,
            recirculate_time=recirculate_time,
            attrition=trial_attrition,
        )

        print(f"  [{trial_counter[0]:3d}/{n_trials}] "
              f"Blower={blower_rpm:.0f}, Wheel={wheel_rpm:.0f}"
              + (f", Passes={trial_passes}" if optimize_passes else "")
              + " ... ", end="", flush=True)

        tr = run_single_trial(
            tc, material, num_particles, sim_time, dt, device, max_loading,
        )
        trials.append(tr)

        score = compute_score(tr, objective, w_protein, w_starch)
        print(_format_trial_line(tr, score))

        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(optuna_objective, n_trials=n_trials)

    # Find the best trial result
    best_idx = max(range(len(trials)),
                   key=lambda i: compute_score(trials[i], objective, w_protein, w_starch))
    best_trial = trials[best_idx]
    best_score = compute_score(best_trial, objective, w_protein, w_starch)

    total_time = time.time() - t_start
    return OptimizationResult(
        best_trial=best_trial,
        all_trials=trials,
        objective=objective,
        strategy="bayesian",
        best_score=best_score,
        wheel_only=wheel_only,
        material=material,
        total_wall_time_s=total_time,
    )


def latin_hypercube_search(
    material: str,
    blower_range: Tuple[float, float],
    wheel_range: Tuple[float, float],
    n_trials: int,
    objective: str,
    w_protein: float,
    w_starch: float,
    num_particles: int,
    sim_time: float,
    dt: float,
    device: str,
    max_loading: float,
    wheel_only: bool,
    passes: int,
    recirculate: Optional[List[str]],
    recirculate_wheel_rpm: Optional[float],
    recirculate_time: Optional[float],
    attrition: float,
) -> OptimizationResult:
    """Latin Hypercube Sampling for space-filling exploration."""
    from scipy.stats.qmc import LatinHypercube

    mode_str = "WHEEL-ONLY" if wheel_only else "FULL SYSTEM"
    print(f"\n{'='*70}")
    print(f"LATIN HYPERCUBE SAMPLING — {mode_str}")
    print(f"  Blower RPM: {blower_range[0]:.0f} – {blower_range[1]:.0f}")
    print(f"  Wheel RPM:  {wheel_range[0]:.0f} – {wheel_range[1]:.0f}")
    print(f"  Samples: {n_trials}")
    print(f"  Objective: {objective}")
    print(f"  Particles: {num_particles}, Time: {sim_time}s")
    print(f"{'='*70}\n")

    t_start = time.time()

    sampler = LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=n_trials)

    # Scale to parameter ranges
    blower_rpms = sample[:, 0] * (blower_range[1] - blower_range[0]) + blower_range[0]
    wheel_rpms = sample[:, 1] * (wheel_range[1] - wheel_range[0]) + wheel_range[0]

    trials: List[TrialResult] = []
    best_score = -float("inf")
    best_trial: Optional[TrialResult] = None

    for i in range(n_trials):
        tc = TrialConfig(
            blower_rpm=float(blower_rpms[i]),
            wheel_rpm=float(wheel_rpms[i]),
            wheel_only=wheel_only,
            passes=passes,
            recirculate=recirculate,
            recirculate_wheel_rpm=recirculate_wheel_rpm,
            recirculate_time=recirculate_time,
            attrition=attrition,
        )

        print(f"  [{i+1:3d}/{n_trials}] "
              f"Blower={blower_rpms[i]:.0f} RPM, Wheel={wheel_rpms[i]:.0f} RPM ... ",
              end="", flush=True)

        tr = run_single_trial(
            tc, material, num_particles, sim_time, dt, device, max_loading,
        )
        trials.append(tr)

        score = compute_score(tr, objective, w_protein, w_starch)
        print(_format_trial_line(tr, score))

        if score > best_score:
            best_score = score
            best_trial = tr

    total_time = time.time() - t_start
    return OptimizationResult(
        best_trial=best_trial or trials[0],
        all_trials=trials,
        objective=objective,
        strategy="lhs",
        best_score=best_score,
        wheel_only=wheel_only,
        material=material,
        total_wall_time_s=total_time,
    )


# ─── Results Reporting ───────────────────────────────────────────────────────

def print_optimization_results(opt_result: OptimizationResult) -> None:
    """Print formatted optimization results with best config and ranking."""
    best = opt_result.best_trial
    bc = best.config

    mode_str = "WHEEL-ONLY" if opt_result.wheel_only else "FULL SYSTEM"
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE — {opt_result.strategy.upper()} — {mode_str}")
    print(f"{'='*70}")
    print(f"  Objective:       {opt_result.objective}")
    print(f"  Trials:          {len(opt_result.all_trials)}")
    n_early = sum(1 for t in opt_result.all_trials if t.early_stopped)
    if n_early > 0:
        print(f"  Early stopped:   {n_early}/{len(opt_result.all_trials)} trials")
    print(f"  Total wall time: {opt_result.total_wall_time_s:.1f} s "
          f"({opt_result.total_wall_time_s/60:.1f} min)")
    print(f"  Best score:      {opt_result.best_score:.4f}")

    print(f"\n  {'─'*50}")
    print(f"  BEST CONFIGURATION:")
    print(f"  {'─'*50}")
    print(f"    Mode:           {'wheel-only' if bc.wheel_only else 'full system'}")
    print(f"    Blower RPM:     {bc.blower_rpm:.0f}")
    print(f"    Wheel RPM:      {bc.wheel_rpm:.0f}")
    if bc.passes > 1:
        print(f"    Passes:         {bc.passes}")
        print(f"    Recirculate:    {bc.recirculate}")
        if bc.recirculate_wheel_rpm is not None:
            print(f"    Recirc. wheel:  {bc.recirculate_wheel_rpm:.0f} RPM")

    print(f"\n  RESULTS:")
    print(f"    Protein recovery:     {best.protein_recovery:.3f}  ({best.protein_recovery*100:.1f}%)")
    print(f"    Starch yield:         {best.starch_yield:.3f}  ({best.starch_yield*100:.1f}%)")
    print(f"    Protein purity:       {best.protein_purity:.3f}  ({best.protein_purity*100:.1f}%)")
    print(f"    Separation eff.:      {best.separation_efficiency:.3f}")
    print(f"    Total collection:     {best.total_collection:.3f}  ({best.total_collection*100:.1f}%)")

    print(f"\n  COLLECTION BREAKDOWN (of {best.total_feed} particles):")
    labels = [
        ("Wheel coarse (starch)", best.wheel_coarse),
    ]
    if not opt_result.wheel_only:
        labels.insert(0, ("Zigzag coarse (starch)", best.zigzag_coarse))
    labels += [
        ("Cyclone 1 (fines)", best.cyclone_1),
        ("Cyclone 2 (fines)", best.cyclone_2),
        ("Cyclone 3 (protein)", best.cyclone_3_protein),
        ("Bag filter", best.bagfilter),
        ("Escaped", best.escaped),
        ("Still active", best.active),
    ]
    for label, count in labels:
        pct = 100.0 * count / max(1, best.total_feed)
        bar = "#" * int(pct / 2)
        print(f"    {label:25s} {count:6d} ({pct:5.1f}%) |{bar}")

    # ── Top-5 ranking ──
    scored = [
        (compute_score(t, opt_result.objective), t)
        for t in opt_result.all_trials if t.feasible
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  {'─'*50}")
    print(f"  TOP 5 CONFIGURATIONS:")
    print(f"  {'─'*50}")
    print(f"  {'Rank':>4s}  {'Blower':>7s}  {'Wheel':>7s}  {'ProtRec':>7s}  "
          f"{'StchYld':>7s}  {'Purity':>7s}  {'Collect':>7s}  {'Score':>7s}")
    print(f"  {'────':>4s}  {'───────':>7s}  {'───────':>7s}  {'───────':>7s}  "
          f"{'───────':>7s}  {'───────':>7s}  {'───────':>7s}  {'───────':>7s}")
    for rank, (score, t) in enumerate(scored[:5], 1):
        print(f"  {rank:4d}  {t.config.blower_rpm:7.0f}  {t.config.wheel_rpm:7.0f}  "
              f"{t.protein_recovery:7.3f}  {t.starch_yield:7.3f}  "
              f"{t.protein_purity:7.3f}  {t.total_collection:7.3f}  {score:7.4f}")

    print(f"\n{'='*70}")

    # ── CLI command to reproduce best ──
    cmd_parts = [
        "python examples/run_classification_flow.py",
        "--full-system",
        f"--material {opt_result.material}",
        f"--blower-rpm {bc.blower_rpm:.0f}",
        f"--wheel-rpm {bc.wheel_rpm:.0f}",
    ]
    if bc.wheel_only:
        cmd_parts.append("--wheel-only")
    if bc.passes > 1 and bc.recirculate:
        cmd_parts.append(f"--recirculate {' '.join(bc.recirculate)}")
        cmd_parts.append(f"--passes {bc.passes}")
        if bc.recirculate_wheel_rpm is not None:
            cmd_parts.append(f"--recirculate-wheel-rpm {bc.recirculate_wheel_rpm:.0f}")

    print(f"\n  Reproduce best with:")
    print(f"    {' '.join(cmd_parts)}")
    print()


def save_results(opt_result: OptimizationResult, filepath: str) -> None:
    """Save optimization results to JSON for analysis."""
    data = {
        "objective": opt_result.objective,
        "strategy": opt_result.strategy,
        "best_score": opt_result.best_score,
        "wheel_only": opt_result.wheel_only,
        "material": opt_result.material,
        "total_wall_time_s": opt_result.total_wall_time_s,
        "n_trials": len(opt_result.all_trials),
        "best_config": {
            "blower_rpm": opt_result.best_trial.config.blower_rpm,
            "wheel_rpm": opt_result.best_trial.config.wheel_rpm,
            "wheel_only": opt_result.best_trial.config.wheel_only,
            "passes": opt_result.best_trial.config.passes,
            "recirculate": opt_result.best_trial.config.recirculate,
        },
        "best_metrics": {
            "protein_recovery": opt_result.best_trial.protein_recovery,
            "starch_yield": opt_result.best_trial.starch_yield,
            "protein_purity": opt_result.best_trial.protein_purity,
            "separation_efficiency": opt_result.best_trial.separation_efficiency,
            "total_collection": opt_result.best_trial.total_collection,
        },
        "trials": [
            {
                "blower_rpm": t.config.blower_rpm,
                "wheel_rpm": t.config.wheel_rpm,
                "passes": t.config.passes,
                "protein_recovery": t.protein_recovery,
                "starch_yield": t.starch_yield,
                "protein_purity": t.protein_purity,
                "separation_efficiency": t.separation_efficiency,
                "total_collection": t.total_collection,
                "early_stopped": t.early_stopped,
                "feasible": t.feasible,
                "wall_time_s": t.wall_time_s,
            }
            for t in opt_result.all_trials
        ],
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {filepath}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimize air classifier operating configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick grid search — wheel-only (default)
  python examples/optimize_classification.py --material yellow_pea

  # Bayesian optimization with 30 trials
  python examples/optimize_classification.py --material yellow_pea --strategy bayesian --n-trials 30

  # Full system (venturi+zigzag+wheel) instead of wheel-only
  python examples/optimize_classification.py --material yellow_pea --full-system

  # Optimize for protein purity with recirculation
  python examples/optimize_classification.py --material yellow_pea --objective protein_purity --recirculate cy1 --passes 2

  # Fast exploration (fewer particles)
  python examples/optimize_classification.py --material yellow_pea --particles 10000 --time 120
""",
    )

    # Material
    parser.add_argument(
        "--material", type=str, required=True,
        choices=["yellow_pea", "faba_bean", "oat"],
        help="Material to classify (required)",
    )

    # Mode: wheel-only (default) vs full-system
    parser.add_argument(
        "--full-system", action="store_true",
        help="Use full system (venturi + zigzag + wheel). Default is wheel-only.",
    )

    # Strategy
    parser.add_argument(
        "--strategy", type=str, default="grid",
        choices=["grid", "bayesian", "lhs"],
        help="Optimization strategy (default: grid)",
    )

    # Objective
    parser.add_argument(
        "--objective", type=str, default="protein_recovery",
        choices=["protein_recovery", "starch_yield", "combined",
                 "separation_efficiency", "protein_purity"],
        help="Objective to maximize (default: protein_recovery)",
    )
    parser.add_argument("--w-protein", type=float, default=2.0,
                        help="Weight for protein recovery in 'combined' objective (default: 2.0)")
    parser.add_argument("--w-starch", type=float, default=1.0,
                        help="Weight for starch yield in 'combined' objective (default: 1.0)")

    # Parameter ranges
    parser.add_argument(
        "--blower-rpm-range", type=float, nargs=2, default=[550, 850],
        metavar=("MIN", "MAX"),
        help="Blower RPM search range (default: 550 850, centered on optimum 700)",
    )
    parser.add_argument(
        "--wheel-rpm-range", type=float, nargs=2, default=[800, 1500],
        metavar=("MIN", "MAX"),
        help="Wheel RPM search range (default: 800 1500, centered on optimum 975)",
    )
    parser.add_argument("--n-blower", type=int, default=4,
                        help="Grid points for blower RPM (default: 4)")
    parser.add_argument("--n-wheel", type=int, default=4,
                        help="Grid points for wheel RPM (default: 4)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of trials for bayesian/lhs (default: 30)")

    # Simulation settings
    parser.add_argument("--particles", "-n", type=int, default=50000,
                        help="Particles per trial (default: 50000, lower = faster)")
    parser.add_argument("--time", "-t", type=float, default=240.0,
                        help="Simulation time per trial in seconds (default: 240)")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Time step (default: 0.001)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--max-loading", type=float, default=2.0,
                        help="Max venturi loading ratio (default: 2.0)")

    # Recirculation
    parser.add_argument("--recirculate", type=str, nargs="+", default=None,
                        help="Fractions to recirculate: cy1, cy2, cy3, etc.")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of classification passes (default: 1)")
    parser.add_argument("--recirculate-wheel-rpm", type=float, default=None,
                        help="Wheel RPM for passes 2+")
    parser.add_argument("--recirculate-time", type=float, default=None,
                        help="Sim time for passes 2+ (default: same as --time)")
    parser.add_argument("--attrition", type=float, default=0.10,
                        help="Attrition per pass (default: 0.10)")

    # Bayesian extras
    parser.add_argument("--optimize-passes", action="store_true",
                        help="(Bayesian) Also optimize number of passes (1-4)")
    parser.add_argument("--optimize-recirculate", action="store_true",
                        help="(Bayesian) Also optimize recirculation fractions")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    wheel_only = not args.full_system

    print("=" * 70)
    print("AIR CLASSIFIER CONFIGURATION OPTIMIZER")
    print(f"  Mode: {'WHEEL-ONLY (no venturi/zigzag)' if wheel_only else 'FULL SYSTEM (venturi+zigzag+wheel)'}")
    print("=" * 70)

    # ── Run optimization ──
    if args.strategy == "grid":
        opt_result = grid_search(
            material=args.material,
            blower_range=tuple(args.blower_rpm_range),
            wheel_range=tuple(args.wheel_rpm_range),
            n_blower=args.n_blower,
            n_wheel=args.n_wheel,
            objective=args.objective,
            w_protein=args.w_protein,
            w_starch=args.w_starch,
            num_particles=args.particles,
            sim_time=args.time,
            dt=args.dt,
            device=args.device,
            max_loading=args.max_loading,
            wheel_only=wheel_only,
            passes=args.passes,
            recirculate=args.recirculate,
            recirculate_wheel_rpm=args.recirculate_wheel_rpm,
            recirculate_time=args.recirculate_time,
            attrition=args.attrition,
        )

    elif args.strategy == "bayesian":
        opt_result = bayesian_optimization(
            material=args.material,
            blower_range=tuple(args.blower_rpm_range),
            wheel_range=tuple(args.wheel_rpm_range),
            objective=args.objective,
            w_protein=args.w_protein,
            w_starch=args.w_starch,
            num_particles=args.particles,
            sim_time=args.time,
            dt=args.dt,
            device=args.device,
            max_loading=args.max_loading,
            n_trials=args.n_trials,
            wheel_only=wheel_only,
            passes=args.passes,
            recirculate=args.recirculate,
            recirculate_wheel_rpm=args.recirculate_wheel_rpm,
            recirculate_time=args.recirculate_time,
            attrition=args.attrition,
            optimize_passes=args.optimize_passes,
            optimize_recirculate=args.optimize_recirculate,
        )

    elif args.strategy == "lhs":
        opt_result = latin_hypercube_search(
            material=args.material,
            blower_range=tuple(args.blower_rpm_range),
            wheel_range=tuple(args.wheel_rpm_range),
            n_trials=args.n_trials,
            objective=args.objective,
            w_protein=args.w_protein,
            w_starch=args.w_starch,
            num_particles=args.particles,
            sim_time=args.time,
            dt=args.dt,
            device=args.device,
            max_loading=args.max_loading,
            wheel_only=wheel_only,
            passes=args.passes,
            recirculate=args.recirculate,
            recirculate_wheel_rpm=args.recirculate_wheel_rpm,
            recirculate_time=args.recirculate_time,
            attrition=args.attrition,
        )

    # ── Print results ──
    print_optimization_results(opt_result)

    # ── Save results ──
    if args.output:
        save_results(opt_result, args.output)


if __name__ == "__main__":
    main()
