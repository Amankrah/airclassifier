"""Compare simulation output against Run#2 PLC data.

Reads the real PLC CSV and runs the simulation side-by-side.
Reports Ia, gap, T_out at key time points for comparison.
"""
import sys, time, csv
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

# ── Load PLC data ─────────────────────────────────────────────────────
plc_data = []
with open("utility_docs/Run2 RF data(in).csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        plc_data.append({
            'time_str': row['Time'],
            'Ia': float(row['Ia']),
            'Product_Temp': float(row['Product_Temp']),
            'Electrode_Act': float(row['Electrode_Act']),
            'Conv_Speed': float(row['Conv_Speed']),
        })

# Find material arrival (Ia > 0.5A)
plc_t0_idx = None
for i, d in enumerate(plc_data):
    if d['Ia'] > 0.5:
        plc_t0_idx = i
        break

print(f"PLC: Material arrival at index {plc_t0_idx}, time {plc_data[plc_t0_idx]['time_str']}")
print(f"PLC: Total records: {len(plc_data)}, 5s intervals")
print()

# Extract PLC time series from material arrival, every 30s (6 records)
print("=== PLC Data (from material arrival, every 30s) ===")
print(f"{'t(s)':>6}  {'Ia(A)':>6}  {'Gap(mm)':>8}  {'T_prod(C)':>9}")
for offset_s in range(0, 1860, 30):
    idx = plc_t0_idx + offset_s // 5
    if idx >= len(plc_data):
        break
    d = plc_data[idx]
    print(f"{offset_s:6d}  {d['Ia']:6.2f}  {d['Electrode_Act']:8.1f}  {d['Product_Temp']:9.0f}")

# ── Run simulation ────────────────────────────────────────────────────
print()
print("=" * 60)
print("=== Running Simulation (FDM, no corrections) ===")

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
machine = cs._machine

print(f"Grid: {cs._grid_shape}, Cell sizes: {cs._cell_sizes}")
print(f"Coupling factor: {machine.oscillator_coupling_factor:.6f}")
print(f"Gap adjust rate: {cs.controller.gap_adjust_rate_mm_s:.4f} mm/s")
print(f"Oven length: {machine.oven_length_m}m")
print(f"Residence time: {machine.oven_length_m / (recipe.belt_speed_m_per_min / 60.0):.0f}s")
print()

# Run for 600s, report every 30s
print(f"{'t(s)':>6}  {'Ia(A)':>6}  {'Gap(mm)':>8}  {'T_out(C)':>9}  {'T_mean(C)':>9}  {'P_rf(kW)':>9}  {'T_elec(C)':>9}  {'State':>10}")
t_start = time.time()
t_report = 0
n_steps = 0

while cs._time < 600:
    dt = cs.compute_stable_dt(recipe)
    dt = min(dt, 600 - cs._time)
    state = cs.step(dt, recipe)
    n_steps += 1

    if cs._time >= t_report:
        print(f"{state.time_s:6.1f}  {state.anode_current_a:6.3f}  {state.electrode_gap_mm:8.1f}  "
              f"{state.T_outfeed_c:9.2f}  {state.T_mean_c:9.2f}  {state.rf_power_kw:9.2f}  "
              f"{state.electrode_temperature_c:9.1f}  {state.controller_state:>10}")
        t_report = cs._time + 30

elapsed = time.time() - t_start
print(f"\nDone: {n_steps} steps in {elapsed:.1f}s")

# ── Side-by-side comparison at key points ─────────────────────────────
print()
print("=" * 60)
print("=== Side-by-Side Comparison ===")
print(f"{'t(s)':>6}  {'PLC Ia':>7}  {'Sim Ia':>7}  {'PLC Gap':>8}  {'Sim Gap':>8}  {'PLC T':>6}  {'Sim T':>6}")

# Get simulation history
sim_history = cs._history

for offset_s in [0, 30, 60, 90, 120, 180, 300, 450, 600]:
    # PLC data
    plc_idx = plc_t0_idx + offset_s // 5
    if plc_idx < len(plc_data):
        plc = plc_data[plc_idx]
        plc_ia = plc['Ia']
        plc_gap = plc['Electrode_Act']
        plc_t = plc['Product_Temp']
    else:
        plc_ia = plc_gap = plc_t = float('nan')

    # Simulation data (find closest time)
    sim_state = None
    for s in sim_history:
        if s.time_s >= offset_s:
            sim_state = s
            break
    if sim_state is None and sim_history:
        sim_state = sim_history[-1]

    if sim_state:
        print(f"{offset_s:6d}  {plc_ia:7.2f}  {sim_state.anode_current_a:7.3f}  "
              f"{plc_gap:8.1f}  {sim_state.electrode_gap_mm:8.1f}  "
              f"{plc_t:6.0f}  {sim_state.T_outfeed_c:6.1f}")

# Key metrics
print()
print("=== Key Metrics Comparison ===")
# PLC peak Ia
plc_peak_ia = max(d['Ia'] for d in plc_data[plc_t0_idx:plc_t0_idx+400])
plc_peak_gap = max(d['Electrode_Act'] for d in plc_data[plc_t0_idx:plc_t0_idx+400])
plc_peak_temp = max(d['Product_Temp'] for d in plc_data[plc_t0_idx:plc_t0_idx+400])

# Simulation peaks
sim_peak_ia = max(s.anode_current_a for s in sim_history)
sim_peak_gap = max(s.electrode_gap_mm for s in sim_history)
sim_peak_temp = max(s.T_outfeed_c for s in sim_history)

print(f"Peak Ia:          PLC={plc_peak_ia:.2f}A   Sim={sim_peak_ia:.3f}A")
print(f"Peak Gap:         PLC={plc_peak_gap:.1f}mm  Sim={sim_peak_gap:.1f}mm")
print(f"Peak Product T:   PLC={plc_peak_temp:.0f}°C  Sim={sim_peak_temp:.1f}°C")
print(f"Temp strips:      77, 77, 82, 77, 82°C (avg 79°C)")
