"""Final comparison: simulation vs Run#2 PLC data with all fixes.

Fixes applied:
1. Ia fraction formula (P_rf directly, not P_gen)
2. FDM solver Red-Black GS-SOR (stable)
3. _T_new initialization (no first-step cold start)
4. eps' floor at 1.5 (physical minimum)
5. j=0 full energy balance (RF + conduction + Robin BC)
"""
import sys, time, csv
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

def run_sim(coupling_factor, duration_s=1200):
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
        material=mat, device="cpu", use_fdm=True,
        enable_controller=True, enable_corrections=False,
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
            t_report = cs._time + 60
    return states

# Load PLC data
plc_data = []
with open("utility_docs/Run2 RF data(in).csv") as f:
    for row in csv.DictReader(f):
        plc_data.append({
            'Ia': float(row['Ia']),
            'Product_Temp': float(row['Product_Temp']),
            'Electrode_Act': float(row['Electrode_Act']),
        })
plc_t0 = next(i for i, d in enumerate(plc_data) if d['Ia'] > 0.5)

for cf in [0.174, 0.155, 0.150, 0.145, 0.140]:
    t_start = time.time()
    states = run_sim(cf, duration_s=1200)
    elapsed = time.time() - t_start

    peak_ia = max(s.anode_current_a for s in states)
    peak_gap = max(s.electrode_gap_mm for s in states)
    last = states[-1]

    print(f"\n{'='*75}")
    print(f"  cf={cf:.4f}  ({elapsed:.0f}s)  Peak: Ia={peak_ia:.3f}A Gap={peak_gap:.0f}mm")
    print(f"  {'t(s)':>6}  {'Ia':>6}  {'Gap':>6}  {'T_out':>6}  {'T_mean':>6}  {'P_rf':>6}  {'T_elec':>7}  {'State':>10}")
    for s in states:
        print(f"  {s.time_s:6.0f}  {s.anode_current_a:6.3f}  {s.electrode_gap_mm:6.1f}  "
              f"{s.T_outfeed_c:6.1f}  {s.T_mean_c:6.1f}  {s.rf_power_kw:6.2f}  "
              f"{s.electrode_temperature_c:7.1f}  {s.controller_state:>10}")

    # Side-by-side at key PLC times
    print(f"\n  PLC comparison:")
    for t_sec in [60, 300, 450, 600, 900, 1200]:
        plc_idx = plc_t0 + t_sec // 5
        if plc_idx >= len(plc_data):
            continue
        p = plc_data[plc_idx]
        sim_s = next((s for s in states if s.time_s >= t_sec), states[-1])
        print(f"    t={t_sec:4d}s  PLC: Ia={p['Ia']:.2f}A Gap={p['Electrode_Act']:.0f}mm T={p['Product_Temp']:.0f}°C"
              f"  |  Sim: Ia={sim_s.anode_current_a:.3f}A Gap={sim_s.electrode_gap_mm:.0f}mm T={sim_s.T_outfeed_c:.1f}°C")

print("\n\nPLC targets: Ia=1.50-1.55A, Gap=93-94mm, T_strips=77-82°C")
