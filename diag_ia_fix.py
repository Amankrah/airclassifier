"""Quick diagnostic: verify the Ia fraction fix and P_rf_theoretical values."""
import sys
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties

# Run#2 conditions
recipe = Recipe(
    name="Run#2",
    electrode_gap_mm=75.0,
    belt_speed_m_per_min=0.2,
    mrh_amps=1.7,
    mrl_amps=1.5,
)

mat = MaterialProperties(
    bed_depth_m=0.035,
    initial_moisture_wb=0.118,
    initial_temperature_c=17.0,
)

sim = GP15Simulator(
    material=mat,
    device="cpu",
    use_fdm=True,
    enable_controller=True,
)
sim.load_recipe(recipe)
sim._ensure_initialized()

cs = sim._sim
machine = cs._machine
mat = cs._material
mat_mask = (cs.cell_is_material == 1)
dx, dy, dz = cs._cell_sizes

print(f"Grid: {cs._grid_shape}, Cell sizes: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}")
print(f"Material cells: {np.sum(mat_mask)}")
print(f"V_rf coupling: {machine.oscillator_coupling_factor:.4f}")
print(f"Oscillator efficiency: {cs._oscillator_efficiency:.3f}")
print(f"max_rf_power_kw: {machine.max_rf_power_kw}")
print()

# Test at various gap widths with the Phase 1 solver
print("=== Phase 1 (series-capacitor) P_rf at different gaps ===")
for gap_mm in [75, 90, 100, 120, 150, 200]:
    gap_m = gap_mm / 1000.0
    V_rf = 9.18 * machine.oscillator_coupling_factor  # no-load voltage
    cs.rf.solve(
        electrode_gap_m=gap_m,
        voltage_kv=V_rf,
        eps_real=cs.eps_real,
        cell_is_material=cs.cell_is_material,
        bed_depth_m=mat.bed_depth_m,
        belt_stack_m=machine.belt_stack_thickness_m,
    )
    from airclassifier.pretreatment.kernels.dielectric_heating import TWO_PI_F_EPS0, compute_power_density_np
    P_v = np.zeros_like(cs.rf.e_field_sq)
    compute_power_density_np(cs.rf.e_field_sq, cs.eps_loss, out=P_v)
    P_rf = float(np.sum(P_v[mat_mask]) * cs._cell_vol) / 1000.0
    frac = P_rf / max(machine.max_rf_power_kw, 0.01)
    Ia = 0.4 + 2.18 * frac
    print(f"  Gap={gap_mm:3d}mm: V_rf={V_rf:.3f}kV  E_bed={cs.rf._E_bed:.0f}V/m  P_rf={P_rf:.2f}kW  frac={frac:.3f}  Ia={min(Ia,3.0):.3f}A")

# Test Phase 2 FDM solver
print()
print("=== Phase 2 (FDM) P_rf at different gaps ===")
for gap_mm in [75, 90, 100, 120, 150, 200]:
    gap_m = gap_mm / 1000.0
    V_rf = 9.18 * machine.oscillator_coupling_factor
    j_upper = min(29, max(1, round(gap_m / dy)))
    cs.rf.solve_fdm(
        gap_m, V_rf, cs.eps_real,
        bed_depth_m=mat.bed_depth_m,
        belt_stack_m=machine.belt_stack_thickness_m,
    )
    P_v = np.zeros_like(cs.rf.e_field_sq)
    compute_power_density_np(cs.rf.e_field_sq, cs.eps_loss, out=P_v)
    P_rf = float(np.sum(P_v[mat_mask]) * cs._cell_vol) / 1000.0
    frac = P_rf / max(machine.max_rf_power_kw, 0.01)
    Ia = 0.4 + 2.18 * frac
    E_bed_rms = float(np.sqrt(np.mean(cs.rf.e_field_sq[mat_mask])))
    print(f"  Gap={gap_mm:3d}mm: j_upper={j_upper:2d}  E_bed_rms={E_bed_rms:.0f}V/m  P_rf={P_rf:.2f}kW  frac={frac:.3f}  Ia={min(Ia,3.0):.3f}A")

# Show MRH thresholds
print()
print("=== MRH Analysis ===")
print(f"MRH threshold: {recipe.mrh_amps:.1f}A")
P_mrh = (recipe.mrh_amps - 0.4) / 2.18 * machine.max_rf_power_kw
print(f"P_rf at MRH threshold: {P_mrh:.2f}kW")
print(f"For MRH to disengage, P_rf_theoretical must drop below {P_mrh:.2f}kW")

# Run a few simulation steps to see the dynamics
print()
print("=== Simulation dynamics (5 steps) ===")
for i in range(5):
    dt = cs.compute_stable_dt(recipe)
    state = cs.step(dt, recipe)
    print(f"  Step {i+1}: t={state.time_s:.2f}s  dt={dt:.3f}s  P_rf={state.rf_power_kw:.2f}kW  Ia={state.anode_current_a:.3f}A  Gap={state.electrode_gap_mm:.1f}mm  T_out={state.T_outfeed_c:.2f}°C")
