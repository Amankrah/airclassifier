"""Run#2 simulation diagnostic with all fixes applied.

Checks: P_rf, Ia, gap dynamics, outfeed temperature.
Target: T_out ≈ 68-70°C, Ia ≈ 1.56A, gap ~90mm (or 75mm if MRH doesn't trip).
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

# Run#2 conditions
mat = MaterialProperties(
    bed_depth_m=0.035,
    initial_moisture_wb=0.118,
    initial_temperature_c=17.0,
)
recipe = Recipe(
    name="Run#2",
    electrode_gap_mm=75.0,
    belt_speed_m_per_min=0.2,
    mrh_amps=1.7,
    mrl_amps=1.5,
)

sim = GP15Simulator(
    material=mat,
    device="cpu",
    use_fdm=True,
    enable_controller=True,
    enable_corrections=False,
)
sim.load_recipe(recipe)
sim._ensure_initialized()
cs = sim._sim

print("=== Run#2 Simulation (FDM + all fixes) ===")
print(f"Grid: {cs._grid_shape}, Material cells: {np.sum(cs.cell_is_material == 1)}")
print(f"V_rf coupling: {cs._machine.oscillator_coupling_factor:.4f}")
print(f"Belt speed: {recipe.belt_speed_m_per_min} m/min")
print(f"Residence time: {cs._machine.oven_length_m / (recipe.belt_speed_m_per_min / 60.0):.0f}s")
print()

# Run for 600s, report every 30s
t_start = time.time()
t_report = 0
n_steps = 0

while cs._time < 600:
    dt = cs.compute_stable_dt(recipe)
    dt = min(dt, 600 - cs._time)
    state = cs.step(dt, recipe)
    n_steps += 1

    if cs._time >= t_report:
        elapsed = time.time() - t_start
        print(f"  t={state.time_s:6.1f}s  P_rf={state.rf_power_kw:5.2f}kW  "
              f"Ia={state.anode_current_a:.3f}A  Gap={state.electrode_gap_mm:5.1f}mm  "
              f"T_out={state.T_outfeed_c:5.2f}°C  T_mean={state.T_mean_c:5.2f}°C  "
              f"T_elec={state.electrode_temperature_c:.1f}°C  "
              f"[wall={elapsed:.1f}s, steps={n_steps}]")
        t_report = cs._time + 30

elapsed = time.time() - t_start
print(f"\nDone: {n_steps} steps in {elapsed:.1f}s ({n_steps/elapsed:.1f} steps/s)")

# Final state
mat_mask = (cs.cell_is_material == 1)
print(f"\n=== Final State (t={cs._time:.0f}s) ===")
print(f"T_outfeed = {state.T_outfeed_c:.1f}°C  (target: 68-70°C)")
print(f"T_mean = {state.T_mean_c:.1f}°C")
print(f"T_max = {state.T_max_c:.1f}°C")
print(f"Ia = {state.anode_current_a:.3f}A  (target: ~1.56A)")
print(f"Gap = {state.electrode_gap_mm:.1f}mm  (target: ~90mm or 75mm)")
print(f"P_rf = {state.rf_power_kw:.2f}kW")
print(f"M_outfeed = {state.M_outfeed_wb:.4f}  (initial: {mat.initial_moisture_wb})")
print(f"T_electrode = {state.electrode_temperature_c:.1f}°C")
