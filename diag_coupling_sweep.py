"""Sweep coupling factor to find value matching Run#2 PLC steady-state.

Target PLC steady-state:
  - Ia ≈ 1.55A at gap ≈ 94mm  →  P_rf ≈ 7.9 kW
  - T_out ≈ 77-82°C (temperature strips)
  - Gap stabilizes around 94mm (not 105mm)
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

def run_sim(coupling_factor, duration_s=500):
    """Run simulation with given coupling factor, return final state."""
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
    cs.update_parameters(coupling_factor=coupling_factor)

    # Run simulation
    states = []
    while cs._time < duration_s:
        dt = cs.compute_stable_dt(recipe)
        dt = min(dt, duration_s - cs._time)
        state = cs.step(dt, recipe)
        if len(states) == 0 or state.time_s >= states[-1].time_s + 30:
            states.append(state)

    return states

# Test coupling factors
print("=" * 80)
print("Coupling factor sweep for Run#2 match")
print("PLC targets: Ia≈1.55A, Gap≈94mm, T_out≈79°C (strips)")
print("=" * 80)

for cf in [0.174, 0.165, 0.155, 0.145, 0.135, 0.125]:
    t_start = time.time()
    states = run_sim(cf, duration_s=500)
    elapsed = time.time() - t_start

    # Find peak values and steady-state (last states)
    peak_ia = max(s.anode_current_a for s in states)
    peak_gap = max(s.electrode_gap_mm for s in states)
    last = states[-1]

    print(f"\n  coupling_factor = {cf:.4f}  ({elapsed:.1f}s)")
    print(f"    Peak Ia={peak_ia:.3f}A  Peak Gap={peak_gap:.1f}mm")
    print(f"    Final (t={last.time_s:.0f}s): Ia={last.anode_current_a:.3f}A  Gap={last.electrode_gap_mm:.1f}mm  "
          f"T_out={last.T_outfeed_c:.1f}°C  T_mean={last.T_mean_c:.1f}°C  P_rf={last.rf_power_kw:.2f}kW  "
          f"T_elec={last.electrode_temperature_c:.1f}°C")

    # Show trajectory
    print(f"    Trajectory:")
    for s in states[::3]:  # every 3rd state (every 90s)
        print(f"      t={s.time_s:5.0f}s  Ia={s.anode_current_a:.3f}A  Gap={s.electrode_gap_mm:6.1f}mm  "
              f"T_out={s.T_outfeed_c:5.1f}°C  P_rf={s.rf_power_kw:.2f}kW  [{s.controller_state}]")
