"""Calibrate simulation against Run#1 PLC data, validate against Run#2.

Run#1 (calibration):  61 kg, 25mm bed, 0.2 m/min, 17.6°C, 11.7% M
Run#2 (validation):   90 kg, 35mm bed, 0.2 m/min, 17.0°C, 11.8% M

Parameters fitted (4):
  oscillator_coupling_factor  -- tank circuit efficiency
  k_evap                      -- evaporation rate for whole seeds
  gap_adjust_rate_mm_s        -- MRH electrode drive speed
  k_dispersion                -- inter-bed thermal dispersion [W/(m·K)]

Uses the infeed-delay + empty-grid initialisation so the Ia ramp-up
matches the real PLC recording (~6 min idle before material arrives).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from airclassifier.pretreatment.calibration import (
    CalibrationOptimizer, load_plc_data, PLCRecording,
)
from airclassifier.pretreatment.config import MachineConfig, MaterialProperties

# ── Load Run#1 PLC data (calibration set) ─────────────────────────────
plc_full = load_plc_data("utility_docs/Run1 RF data(in).csv")
print(f"Run#1 PLC: {plc_full.n_samples} samples, {plc_full.duration_s:.0f} s")
print(f"  Ia range: {plc_full.anode_current_a.min():.2f}-{plc_full.anode_current_a.max():.2f} A")
print(f"  Temp range: {plc_full.product_temp_c.min():.0f}-{plc_full.product_temp_c.max():.0f} C")

# ── Use FULL PLC recording (no trimming) ─────────────────────────────
# The simulation now models the infeed delay naturally (empty oven at
# t=0, material arrives at grid after belt transit ~211s).  The PLC
# also starts with idle Ia (~0.22A) before material reaches the RF zone
# (~410s).  Using the full recording lets the optimizer fit the entire
# lifecycle: idle → ramp → steady → run-out → idle.
plc = plc_full

# Find material arrival for reference only
arrival_idx = 0
for i in range(len(plc.anode_current_a)):
    if plc.anode_current_a[i] > 0.5:
        arrival_idx = i
        break
print(f"\nMaterial arrival at index {arrival_idx}, "
      f"t={plc.time_s[arrival_idx]:.0f}s into recording")
print(f"Using full PLC: {plc.n_samples} samples, {plc.duration_s:.0f} s")
print(f"  Ia at t=0: {plc.anode_current_a[0]:.2f} A (idle)")
print(f"  Gap at t=0: {plc.electrode_act_mm[0]:.1f} mm")

# Show PLC trajectory for reference
print(f"\nPLC trajectory (full recording):")
print(f"  {'t(s)':>6}  {'Ia(A)':>6}  {'Gap(mm)':>8}  {'T(C)':>6}")
for offset_s in [0, 60, 120, 300, 450, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]:
    idx = offset_s // 5
    if idx >= plc.n_samples:
        break
    print(f"  {offset_s:6d}  {plc.anode_current_a[idx]:6.2f}  "
          f"{plc.electrode_act_mm[idx]:8.1f}  {plc.product_temp_c[idx]:6.0f}")

# ── Configure for Run#1 ─────────────────────────────────────────────
# Run#1: 61 kg, 25mm feeder gap, 0.2 m/min, 17.6°C starting temp
# NIR raw moisture: 11.37%, 12.33%, 11.51% → avg 11.74%
material = MaterialProperties(
    bed_depth_m=0.025,            # 25mm feeder gap (Run#1)
    initial_moisture_wb=0.1174,   # 11.74% wb (NIR average)
    initial_temperature_c=17.6,   # 17.6°C starting temp (Run#1 avg)
)
config = MachineConfig()

# Recipe overrides for Run#1 (61 kg batch)
recipe_overrides = {
    "run_mass_kg": 61.0,
}

# ── Run calibration ──────────────────────────────────────────────────
# Fit against the full PLC recording so the optimizer sees every phase:
#   1. Ia ramp (material filling RF zone)
#   2. Steady-state processing (gap at peak, Ia at MRH band)
#   3. Run-out (Ia drop, gap return to setpoint)
# Truncating would miss the run-out dynamics that constrain gap_rate.
cal_duration = None  # None = full PLC duration
cal_duration_display = cal_duration or plc.duration_s
print(f"\n{'='*60}")
print(f"Calibrating against {cal_duration_display:.0f}s of Run#1 PLC data")
print(f"  Material: bed=25mm, M=11.74%, T0=17.6°C, mass=61kg")
print(f"  Recipe from PLC: gap={np.median(plc.electrode_set_mm):.0f}mm, "
      f"speed={np.median(plc.conv_speed_m_per_min):.1f} m/min, "
      f"MRH={plc.ia_limit_2:.1f}A, MRL={plc.ia_limit_1:.1f}A")
print(f"  Parameters: coupling, k_evap, gap_rate, k_dispersion")
print(f"{'='*60}")

t_start = time.time()

cal = CalibrationOptimizer(
    plc, config=config, material=material,
    recipe_overrides=recipe_overrides,
    sim_duration_s=cal_duration,
    device=None,             # auto-detect CUDA
    w_temperature=0.5,       # IR sensor is biased vs bulk temp
    w_anode_current=1.5,     # most reliable PLC signal
    w_gap=1.0,               # reliable signal
    n_compare_points=60,
)

# Baseline evaluation (warms simulator; time used for ETA)
print("\nBaseline (current calibration):")
t_baseline = time.perf_counter()
baseline = cal.evaluate_current()
baseline_sec = time.perf_counter() - t_baseline
print(f"  coupling={baseline['coupling']:.4f}, k_evap={baseline['k_evap']:.2e}, "
      f"k_disp={baseline['k_dispersion']:.2f}, gap_rate={baseline['gap_rate']:.4f}")
print(f"  T_sim={baseline['T_sim_final']:.1f}°C vs T_plc={baseline['T_plc_final']:.0f}°C")
print(f"  Ia_sim={baseline['Ia_sim_final']:.3f}A vs Ia_plc={baseline['Ia_plc_final']:.2f}A")
print(f"  gap_sim={baseline['gap_sim_final']:.1f}mm vs gap_plc={baseline['gap_plc_final']:.1f}mm")
print(f"  loss={baseline['loss_total']:.3f} (T={baseline['loss_T']:.3f}, "
      f"Ia={baseline['loss_Ia']:.3f}, gap={baseline['loss_gap']:.3f})")
est_100 = baseline_sec * 100 / 60
est_300 = baseline_sec * 300 / 60
print(f"  Baseline eval: {baseline_sec:.1f} s  →  est. ~{est_100:.0f} min (100 evals) to ~{est_300:.0f} min (300 evals)")

# Run optimizer
# "nelder-mead" from baseline: ~5x faster than DE, good when baseline is close
# "de": global search, more robust but slower
print(f"\nRunning calibration (method=nelder-mead, maxiter=150)...")
cal_result = cal.run(method="nelder-mead", maxiter=150, seed=42)

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"Calibration complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*60}")
print(cal_result)

# ── Save result ──────────────────────────────────────────────────────
from pathlib import Path
from airclassifier.pretreatment.calibration_store import save_calibration
cal_file = Path("utility_docs/calibration_latest.json")
save_calibration(cal_result, cal_file)
print(f"\nSaved to {cal_file}")
print(f"  coupling_factor = {cal_result.oscillator_coupling_factor:.6f}")
print(f"  k_evap          = {cal_result.k_evap:.2e}")
print(f"  k_dispersion    = {cal_result.k_dispersion:.4f} W/(m·K)")
print(f"  gap_rate         = {cal_result.gap_adjust_rate_mm_s:.6f} mm/s")

# ── Validation reminder ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("NEXT: Validate against Run#2")
print("  python examples/simulate_and_visualize.py \\")
print("    --mass 90 --gap 75 --bed-depth 35 --speed 0.2 \\")
print("    --temp 17.0 --moisture 0.118067")
print()
print("Run#2 targets (NIR):")
print("  Outfeed moisture:  10.53% wb (avg of 15 samples)")
print("  Temperature strips: 77-82°C")
print("  Gap peak:          94.1 mm")
print("  Ia steady:         1.5-1.7 A")
print(f"{'='*60}")
