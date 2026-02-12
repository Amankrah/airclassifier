"""Quick check: does j=0 heat up and does electrode temperature rise?"""
import sys
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

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

# Check cell zones at j=0,1,2,3
nx, ny, nz = cs._grid_shape
print(f"Grid: ({nx},{ny},{nz}), dy={cs._cell_sizes[1]:.4f}")
for j in range(min(ny, 6)):
    zone_counts = {0: 0, 1: 0, 2: 0}
    for z in [0, 1, 2]:
        if z in [0, 1, 2]:
            zone_counts[z] = int(np.sum(cs.cell_is_material[:, j, :] == z))
    print(f"  j={j}: zone0(air)={zone_counts[0]}, zone1(mat)={zone_counts[1]}, zone2(belt)={zone_counts[2]}")

print()
print("=== Running 300s, monitoring j=0 and electrode temperature ===")
t_report = 0
while cs._time < 300:
    dt = cs.compute_stable_dt(recipe)
    dt = min(dt, 300 - cs._time)
    state = cs.step(dt, recipe)
    if cs._time >= t_report:
        mat_mask_j0 = (cs.cell_is_material[:, 0, :] == 1)
        T_j0_avg = float(np.mean(cs.thermal.T[:, 0, :][mat_mask_j0]))
        T_j1_avg = float(np.mean(cs.thermal.T[:, 1, :][cs.cell_is_material[:, 1, :] == 1]))
        print(f"  t={state.time_s:5.0f}s  T_j0={T_j0_avg:5.1f}°C  T_j1={T_j1_avg:5.1f}°C  "
              f"T_elec={state.electrode_temperature_c:.2f}°C  T_out={state.T_outfeed_c:.1f}°C  "
              f"Ia={state.anode_current_a:.3f}A  Gap={state.electrode_gap_mm:.1f}mm")
        t_report = cs._time + 30
