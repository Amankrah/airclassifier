"""Recalibrate simulation against Run#2 PLC data with all bug fixes.

Trims PLC data to start from material arrival (Ia > 0.5A) so that
t=0 in PLC aligns with t=0 in simulation (material already loaded).

Uses corrected FDM solver with:
  - Red-Black GS-SOR field solve
  - _T_new initialization fix
  - j=0 full energy balance (RF + conduction + Robin)
  - eps' floor at 1.5
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from airclassifier.pretreatment.calibration import (
    CalibrationOptimizer, load_plc_data, PLCRecording,
)
from airclassifier.pretreatment.config import MachineConfig, MaterialProperties

# ── Load PLC data ────────────────────────────────────────────────────
plc_full = load_plc_data("utility_docs/Run2 RF data(in).csv")
print(f"Full PLC: {plc_full.n_samples} samples, {plc_full.duration_s:.0f} s")
print(f"  Ia range: {plc_full.anode_current_a.min():.2f}-{plc_full.anode_current_a.max():.2f} A")
print(f"  Temp range: {plc_full.product_temp_c.min():.0f}-{plc_full.product_temp_c.max():.0f} C")

# ── Trim to material arrival (Ia > 0.5A) ────────────────────────────
arrival_idx = None
for i in range(len(plc_full.anode_current_a)):
    if plc_full.anode_current_a[i] > 0.5:
        arrival_idx = i
        break

if arrival_idx is None:
    raise ValueError("No material arrival detected (Ia never > 0.5A)")

print(f"\nMaterial arrival at index {arrival_idx}, "
      f"t={plc_full.time_s[arrival_idx]:.0f}s into recording")

# Create trimmed PLCRecording starting from material arrival
t_offset = plc_full.time_s[arrival_idx]
plc = PLCRecording(
    time_s=plc_full.time_s[arrival_idx:] - t_offset,
    anode_current_a=plc_full.anode_current_a[arrival_idx:],
    grid_current_a=plc_full.grid_current_a[arrival_idx:],
    product_temp_c=plc_full.product_temp_c[arrival_idx:],
    electrode_set_mm=plc_full.electrode_set_mm[arrival_idx:],
    electrode_act_mm=plc_full.electrode_act_mm[arrival_idx:],
    conv_speed_m_per_min=plc_full.conv_speed_m_per_min[arrival_idx:],
    ia_limit_1=plc_full.ia_limit_1,
    ia_limit_2=plc_full.ia_limit_2,
    duration_s=float(plc_full.time_s[-1] - t_offset),
    n_samples=len(plc_full.time_s) - arrival_idx,
    sample_interval_s=plc_full.sample_interval_s,
)

print(f"Trimmed PLC: {plc.n_samples} samples, {plc.duration_s:.0f} s")
print(f"  Ia at t=0: {plc.anode_current_a[0]:.2f} A")
print(f"  Gap at t=0: {plc.electrode_act_mm[0]:.1f} mm")

# Show PLC trajectory for reference
print(f"\nPLC trajectory (from material arrival):")
print(f"  {'t(s)':>6}  {'Ia(A)':>6}  {'Gap(mm)':>8}  {'T(C)':>6}")
for offset_s in [0, 30, 60, 90, 120, 180, 300, 450, 600, 900, 1200]:
    idx = offset_s // 5
    if idx >= plc.n_samples:
        break
    print(f"  {offset_s:6d}  {plc.anode_current_a[idx]:6.2f}  "
          f"{plc.electrode_act_mm[idx]:8.1f}  {plc.product_temp_c[idx]:6.0f}")

# ── Configure for Run#2 ─────────────────────────────────────────────
material = MaterialProperties(
    bed_depth_m=0.035,           # 35mm feeder gap
    initial_moisture_wb=0.118,   # 11.8% wb
    initial_temperature_c=17.0,  # 17°C starting temp
)
config = MachineConfig()

# ── Run calibration ──────────────────────────────────────────────────
# Use 900s window to capture initial transient + approach to steady-state
cal_duration = 900.0
print(f"\n{'='*60}")
print(f"Calibrating against {cal_duration:.0f}s of Run#2 PLC data")
print(f"  FDM solver: enabled (corrected)")
print(f"  Material: bed=35mm, M=11.8%, T0=17°C")
print(f"  Recipe from PLC: gap={np.median(plc.electrode_set_mm):.0f}mm, "
      f"speed={np.median(plc.conv_speed_m_per_min):.1f} m/min, "
      f"MRH={plc.ia_limit_2:.1f}A, MRL={plc.ia_limit_1:.1f}A")
print(f"{'='*60}")

t_start = time.time()

cal = CalibrationOptimizer(
    plc, config=config, material=material,
    sim_duration_s=cal_duration,
    device="cpu",
    w_temperature=0.5,   # lower weight: IR sensor is biased vs bulk temp
    w_anode_current=1.5, # higher weight: most reliable PLC signal
    w_gap=1.0,           # reliable signal
    n_compare_points=60,
)

# Baseline evaluation
print("\nBaseline (current calibration):")
baseline = cal.evaluate_current()
print(f"  coupling={baseline['coupling']:.4f}, k_evap={baseline['k_evap']:.2e}, "
      f"gap_rate={baseline['gap_rate']:.4f}")
print(f"  T_sim={baseline['T_sim_final']:.1f}°C vs T_plc={baseline['T_plc_final']:.0f}°C")
print(f"  Ia_sim={baseline['Ia_sim_final']:.3f}A vs Ia_plc={baseline['Ia_plc_final']:.2f}A")
print(f"  gap_sim={baseline['gap_sim_final']:.1f}mm vs gap_plc={baseline['gap_plc_final']:.1f}mm")
print(f"  loss={baseline['loss_total']:.3f} (T={baseline['loss_T']:.3f}, "
      f"Ia={baseline['loss_Ia']:.3f}, gap={baseline['loss_gap']:.3f})")

# Run optimizer
print(f"\nRunning differential evolution (maxiter=20, popsize=15)...")
cal_result = cal.run(maxiter=20, seed=42)

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
print(f"  gap_rate         = {cal_result.gap_adjust_rate_mm_s:.6f} mm/s")
