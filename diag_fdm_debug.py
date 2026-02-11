"""Debug the FDM solver: check eps array, potential profile, convergence."""
import sys
sys.path.insert(0, "src")

import numpy as np
from airclassifier.pretreatment.simulator import GP15Simulator
from airclassifier.pretreatment.config import Recipe, MaterialProperties
from airclassifier.pretreatment.kernels.field_solve import (
    solve_laplace_jacobi, compute_gradient_sq_np
)
from airclassifier.pretreatment.kernels.dielectric_heating import (
    TWO_PI_F_EPS0, compute_power_density_np
)

# Setup
mat = MaterialProperties(
    bed_depth_m=0.035,
    initial_moisture_wb=0.118,
    initial_temperature_c=17.0,
)
recipe = Recipe(electrode_gap_mm=75.0, belt_speed_m_per_min=0.2)
sim = GP15Simulator(material=mat, device="cpu", use_fdm=True, enable_controller=True)
sim.load_recipe(recipe)
sim._ensure_initialized()

cs = sim._sim
machine = cs._machine
dx, dy, dz = cs._cell_sizes
nx, ny, nz = cs._grid_shape
mat_mask = (cs.cell_is_material == 1)

print(f"Grid: ({nx},{ny},{nz}), Cell sizes: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}")
print(f"Material cells: {np.sum(mat_mask)}")
print()

# Check eps_real profile along Y axis (at center X, Z)
ix, iz = nx // 2, nz // 2
print("=== eps_real along Y (center slice) ===")
for j in range(min(ny, 15)):
    y_mm = (j + 0.5) * dy * 1000
    eps_val = cs.eps_real[ix, j, iz]
    is_mat = cs.cell_is_material[ix, j, iz]
    print(f"  j={j:2d}  y={y_mm:5.1f}mm  eps_real={eps_val:.4f}  is_material={is_mat}")

print()
print("=== Direct FDM solve (fresh, not reusing potential) ===")
for gap_mm in [75, 100, 150]:
    gap_m = gap_mm / 1000.0
    V_rf = 9.18 * machine.oscillator_coupling_factor  # no-load
    V_total = V_rf * 1000.0
    j_upper = min(ny - 1, max(1, round(gap_m / dy)))

    # Solve from SCRATCH (zeros)
    phi = solve_laplace_jacobi(
        eps=cs.eps_real,
        V_upper=V_total,
        dx=dx, dy=dy, dz=dz,
        max_iter=500,
        tol=1e-5,
        phi_init=None,  # force linear ramp initial guess
        j_upper=j_upper,
    )

    E2 = compute_gradient_sq_np(phi, dx, dy, dz)
    E2[:, j_upper:, :] = 0.0

    # Print potential profile along Y
    print(f"\n  Gap={gap_mm}mm, j_upper={j_upper}:")
    for j in range(min(j_upper + 3, ny)):
        phi_val = phi[ix, j, iz]
        e2_val = E2[ix, j, iz]
        print(f"    j={j:2d}  phi={phi_val:8.1f}V  E={np.sqrt(e2_val):8.1f}V/m  {'MAT' if cs.cell_is_material[ix,j,iz]==1 else 'AIR'}")

    # Total P_rf
    P_v = np.zeros_like(E2)
    compute_power_density_np(E2, cs.eps_loss, out=P_v)
    P_rf = float(np.sum(P_v[mat_mask]) * cs._cell_vol) / 1000.0
    E_bed_rms = float(np.sqrt(np.mean(E2[mat_mask])))
    print(f"    P_rf = {P_rf:.2f} kW,  E_bed_rms = {E_bed_rms:.0f} V/m")

# Also check: what does a pure 1D analytical solution give?
print()
print("=== Analytical 1D series-capacitor model ===")
for gap_mm in [75, 100, 150]:
    gap_m = gap_mm / 1000.0
    V_total = 9.18 * machine.oscillator_coupling_factor * 1000.0
    d_belt = machine.belt_stack_thickness_m
    d_bed = mat.bed_depth_m
    d_air = max(gap_m - d_belt - d_bed, 0.0)
    eps_belt = machine.belt_permittivity_real
    eps_bed = float(np.mean(cs.eps_real[mat_mask]))
    eps_air = 1.0
    cap_sum = d_air / eps_air + d_bed / eps_bed + d_belt / eps_belt
    if cap_sum > 0:
        D = V_total / cap_sum
        E_bed = D / eps_bed
        E_air = D / eps_air
        P_v_bed = TWO_PI_F_EPS0 * float(np.mean(cs.eps_loss[mat_mask])) * E_bed**2
        vol_bed = 1.5 * 0.8 * d_bed  # approximate bed volume
        P_rf = P_v_bed * vol_bed / 1000.0
        print(f"  Gap={gap_mm}mm: eps_bed={eps_bed:.2f} E_bed={E_bed:.0f}V/m E_air={E_air:.0f}V/m P_rf≈{P_rf:.2f}kW")
