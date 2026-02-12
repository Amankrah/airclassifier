"""Run longer simulations at promising coupling factors to find steady-state.

Target PLC steady-state (t=1200s from material arrival):
  - Ia ≈ 1.50-1.55A
  - Gap ≈ 93-94mm
  - T_out ≈ 77-82°C (temp strips)
  - P_rf ≈ 7.5-8.5 kW
"""
import sys, time
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

def run_sim(coupling_factor, duration_s=1200):
    """Run simulation with given coupling factor."""
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

    states = []
    t_report = 0
    while cs._time < duration_s:
        dt = cs.compute_stable_dt(recipe)
        dt = min(dt, duration_s - cs._time)
        state = cs.step(dt, recipe)
        if cs._time >= t_report:
            states.append(state)
            t_report = cs._time + 60  # every 60s

    return states

for cf in [0.150, 0.148, 0.145, 0.142]:
    t_start = time.time()
    states = run_sim(cf, duration_s=1200)
    elapsed = time.time() - t_start

    peak_ia = max(s.anode_current_a for s in states)
    peak_gap = max(s.electrode_gap_mm for s in states)

    print(f"\n{'='*70}")
    print(f"  coupling_factor = {cf:.4f}  (computed in {elapsed:.1f}s)")
    print(f"  Peak Ia={peak_ia:.3f}A  Peak Gap={peak_gap:.1f}mm")
    print(f"  {'t(s)':>6}  {'Ia(A)':>7}  {'Gap(mm)':>8}  {'T_out(C)':>9}  {'T_mean(C)':>9}  {'P_rf(kW)':>9}  {'T_elec(C)':>9}  {'State':>10}")

    for s in states:
        print(f"  {s.time_s:6.0f}  {s.anode_current_a:7.3f}  {s.electrode_gap_mm:8.1f}  "
              f"{s.T_outfeed_c:9.1f}  {s.T_mean_c:9.1f}  {s.rf_power_kw:9.2f}  "
              f"{s.electrode_temperature_c:9.1f}  {s.controller_state:>10}")

    # Compare with PLC at similar times
    print(f"\n  PLC comparison (from Run2 RF data):")
    print(f"  t=450s: PLC Ia=1.66A Gap=94.1mm T=43°C | Sim Ia={states[7].anode_current_a:.3f}A Gap={states[7].electrode_gap_mm:.1f}mm T={states[7].T_outfeed_c:.1f}°C")
    last = states[-1]
    print(f"  t=900s: PLC Ia=1.58A Gap=93.9mm T=77°C | Sim Ia={last.anode_current_a:.3f}A Gap={last.electrode_gap_mm:.1f}mm T={last.T_outfeed_c:.1f}°C")
