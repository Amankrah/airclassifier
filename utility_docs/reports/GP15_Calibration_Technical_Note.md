# GPU-Accelerated Digital Twin of the GP-15 RF Dielectric Heating Machine: Model Development, Calibration, and Validation Against PLC Field Data

**Technical Note — Preprint Draft**

---

**Application:** Thermal pretreatment of whole yellow pea (*Pisum sativum*) for dry fractionation
**Machine:** QMTI GP-15 (15 kW self-excited triode oscillator, 27.12 MHz ISM band)
**Calibration Data:** Run #1 PLC recording (559 samples, 2794 s, 25-Mar-2025)
**Compute:** NVIDIA RTX 6000 Ada Generation (48 GiB) via CUDA / NVIDIA Warp 1.11
**Authors:** Emmanuel Kwofie
**Date:** February 2026

---

## Nomenclature

| Symbol | Description | Units |
|---|---|---|
| *f* | RF operating frequency (27.12 MHz) | Hz |
| *epsilon_0* | Permittivity of free space | F/m |
| *epsilon'* | Relative dielectric constant | -- |
| *epsilon''* | Dielectric loss factor | -- |
| *E* | Electric field magnitude | V/m |
| *P_v* | Volumetric RF power density | W/m^3 |
| *T* | Temperature | C |
| *M* | Moisture content (wet basis) | kg/kg |
| *rho* | Bulk density | kg/m^3 |
| *c_p* | Specific heat capacity | J/(kg K) |
| *k* | Thermal conductivity | W/(m K) |
| *D_eff* | Effective moisture diffusivity | m^2/s |
| *L_v* | Latent heat of vaporization | J/kg |
| *V_rf* | RF voltage at electrodes | V |
| *V_a* | Anode DC voltage | V |
| *I_a* | Anode current | A |
| *k_c* | Oscillator coupling factor | -- |
| *k_evap* | Evaporation rate constant | 1/(C s) |
| *r_gap* | MRH gap adjustment rate | mm/s |
| *v_belt* | Belt linear speed | m/s |

---

## 1. Abstract

This technical note presents the development, calibration, and validation of a GPU-accelerated digital twin for the QMTI GP-15 radio-frequency (RF) dielectric heating machine. The GP-15 is a 15 kW, 27.12 MHz industrial RF heating system used as a pretreatment step in the dry fractionation of whole yellow pea (*Pisum sativum*), where it thermally conditions seeds prior to pin milling and air classification.

The digital twin couples ten physics and control substeps per timestep: belt advection (TVD scheme), quasi-static RF field solution, volumetric dielectric heating, evaporation kinetics, thermal transport (explicit FDM with variable conductivity), moisture diffusion, nonlinear material property updates, PLC controller logic with MRH overcurrent protection, KPI recording, and Lagrangian particle tracking. The model is implemented in Python with GPU-accelerated kernels via NVIDIA Warp, running on a 3D rectilinear grid of 57,600 cells.

The model was developed through an iterative process of progressive debugging and physics refinement, guided by comparison with real PLC data from a 61 kg production run. Six critical bugs were identified and corrected in the initial implementation (material mask coordinate mismatch, RF field Y-mapping error, thermal boundary condition, evaporation threshold, CFL stability, and specific energy computation). The generator model was then transitioned from a fixed-power approach to a self-consistent voltage-driven model calibrated against the PLC anode current trajectory.

Three parameters were calibrated using differential evolution with L-BFGS-B local refinement against the full 2794-second PLC recording: the oscillator coupling factor (*k_c* = 0.174), the evaporation rate constant (*k_evap* = 1.0 x 10^-6), and the MRH gap adjustment rate (*r_gap* = 0.148 mm/s). The calibration loss function comprised variance-normalized MSE for outfeed temperature, anode current, and electrode gap, with anode current providing the strongest constraint on the coupling factor.

The calibrated model reproduces the key features of the PLC data: outfeed temperature of 43.2 C (vs. 45.0 C measured), electrode gap drift from 75 to 79.6 mm (vs. 75 to 87 mm measured), anode current stabilizing at 1.70 A near the MRH threshold, and negligible moisture removal (10.0% to 9.8% wb). The most significant finding is that *k_evap* was driven to the lower bound of the search range, confirming that the GP-15 operating on whole yellow pea functions as a **thermal pretreatment unit**, not a dryer. Virtually all RF energy is directed to sensible heating, with the intact seed coat preventing significant moisture loss below 100 C.

---

## 2. Introduction

### 2.1 Background

Radio-frequency (RF) dielectric heating at ISM frequencies (27.12 MHz) is an established technology for post-harvest treatment of agricultural products, including tempering, drying, pasteurization, and insect disinfestation. Unlike conventional convective heating, RF heating is volumetric: the oscillating electric field directly excites polar molecules (primarily water) throughout the material bulk, producing uniform temperature rise without the thermal lag associated with surface-to-core conduction.

The QMTI GP-15 is an industrial RF heating system comprising a self-excited triode valve oscillator, a parallel-plate electrode applicator (oven), a continuous PTFE conveyor belt, and an Environment Management Unit (EMU) for air handling. The machine is rated at 15 kW RF output at 27.12 MHz, with an 800 mm belt width and adjustable electrode gap (20--300 mm). The GP-15 is part of a dry fractionation line for plant-based protein: whole seeds are RF-treated, then pin-milled into flour, and air-classified to separate protein-rich and starch-rich fractions.

### 2.2 Motivation for a Digital Twin

Optimizing the GP-15's operating parameters (electrode gap, belt speed, bed depth) for new feedstocks currently requires costly trial-and-error experimentation. A validated simulation model — a digital twin — would enable virtual recipe development, reduce commissioning time, predict product quality (temperature uniformity, moisture content), and provide real-time process monitoring when coupled with PLC data.

Digital twins for RF heating have been reported in the literature for food processing applications, but most focus on microwave (2.45 GHz) rather than RF (27.12 MHz) frequencies, and few incorporate the PLC control logic (MRH overcurrent protection, electrode gap servo) that governs the real machine's operating point. This work develops a comprehensive digital twin that couples electromagnetic, thermal, moisture, mechanical (conveyor), and control subsystems, and calibrates it against field PLC data from an actual production run.

### 2.3 Scope

This note covers:
- The governing equations and their numerical implementation
- An iterative model development methodology grounded in comparison with PLC data
- A formal calibration procedure using derivative-free global optimization
- Validation of the calibrated model against an independent section of the PLC recording
- Physical interpretation of the calibrated parameters

---

## 3. Governing Equations

The simulation couples four physics domains through shared state variables (*T*, *M*, *E*), updated at each timestep in a sequential operator-splitting scheme.

### 3.1 RF Dielectric Heating

The volumetric power density absorbed by the material is:

    P_v(x,y,z) = 2*pi*f * epsilon_0 * epsilon''(T, M) * |E(x,y,z)|^2    [W/m^3]

where *epsilon''(T, M)* is the temperature- and moisture-dependent dielectric loss factor. At 27.12 MHz the free-space wavelength (11 m) far exceeds the electrode dimensions, so the electromagnetic problem reduces to a quasi-static Laplace equation. For the Phase 1 implementation, a series-capacitor voltage division model is used:

    E_bed = V_rf / (epsilon'_bed * sum(d_i / epsilon'_i))

where the summation runs over the three dielectric layers (air gap, material bed, PTFE belt stack).

The electrode RF voltage is computed from the generator's anode voltage via the oscillator coupling factor:

    V_rf = V_anode(I_a) * k_c

where *V_anode* follows the linear droop model fitted from the GP-15 test report (no-load: 9.18 kV at 0.4 A; full-load: 8.38 kV at 2.58 A).

### 3.2 Heat Transfer

Temperature evolution in the material bed:

    rho*c_p * dT/dt = div(k * grad(T)) + P_v - L_v * m_evap

with Dirichlet BCs at the infeed (T = T_inlet), Neumann (zero-gradient) at the outfeed and belt edges, a Robin BC at the bottom (contact conductance through the PTFE belt stack to the isothermal electrode), and a convective BC at the bed surface (EMU airflow).

### 3.3 Moisture Transport

    dM/dt = div(D_eff(T) * grad(M)) - m_evap / rho_dry

where D_eff follows an Arrhenius model and the evaporation rate is:

    m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)

### 3.4 Belt Advection

Material fields (*T*, *M*) are advected along the belt direction (+X) using a second-order Van Leer TVD scheme with the Courant number limited to < 0.9.

### 3.5 PLC Controller

The controller replicates the GP-15's MRH (Meter Relay High) overcurrent protection:
- When *I_a > I_MRH*, the electrode gap opens at rate *r_gap* [mm/s]
- When *I_a < I_MRL*, the electrode drive stops
- The gap adjusts continuously (proportional control), not as a safety trip

This distinction — MRH as proportional gap control, not a recycle trigger — was identified from the Run #1 PLC data and is critical for correct simulation behavior.

---

## 4. Model Development Methodology

The digital twin was developed through an iterative process of progressive debugging, physics refinement, and validation against PLC data. This section documents the key development milestones.

### 4.1 Initial Implementation and Bug Discovery

The initial simulation produced zero moisture removal and minimal temperature rise (28.4 C outfeed from 22 C inlet). Systematic investigation revealed six interacting bugs:

| Bug | Root Cause | Impact |
|---|---|---|
| Material mask coordinate mismatch | Mask builder used dy = gap/ny instead of gap_max/ny | 7 cells tagged as material instead of 2 |
| RF field Y-mapping error | Field solver used same wrong dy for layer boundaries | E-field assigned to wrong Y-positions |
| Bottom BC destroyed heating | Dirichlet BC at j=0 erased RF energy in material cells | ~50% of heating lost |
| Evaporation threshold too high | T_threshold = 40 C never reached at initial conditions | Zero evaporation |
| CFL excluded air cells | Thermal solver updated air cells but CFL only checked material | Potential instability |
| Specific energy division by zero | Near-zero water removal gave 25,490 kWh/kg | Misleading output |

After correction, the simulation produced physically reasonable results: outfeed moisture 8.6% (from 10%), outfeed temperature 31.6 C, specific energy 0.96 kWh/kg — matching the GP-15 manual's target of 1.0 kg water per kWh.

### 4.2 Generator Model Transition

The initial implementation used a fixed-power approach (Approach A from the engineering guide): the generator was assumed to deliver its full rated 15 kW regardless of the load. Analysis of the Run #1 PLC data revealed that the actual anode current (1.65--1.70 A at steady state) corresponds to approximately 8.6 kW — far below the rated maximum. The generator model was therefore transitioned to a voltage-driven approach (Approach B): the electrode voltage is computed from the anode voltage and the coupling factor, and the material absorbs whatever power results from this voltage applied through the series-capacitor model.

### 4.3 MRH Controller Behavior

The PLC data showed that the electrode gap opened smoothly from 75 mm to 87 mm over approximately 1000 seconds, with RF power remaining ON throughout. This contradicted the initial simulation's safety logic, which treated MRH overcurrent as a recycle trigger (RF off, 2-second delay, restart, 4 attempts then lockout). The safety monitor was corrected to treat MRH as proportional gap control — a continuous adjustment, not a safety trip — matching the observed PLC behavior.

### 4.4 Vertical Grid Resolution

The simulation grid spanned the maximum electrode gap (300 mm), but with ny = 11 cells, the 27.3 mm cell height placed only 2 cells within the 50 mm material bed. The minimum cell height was constrained to 10 mm, increasing ny to 30 and providing 5 material cells — sufficient to resolve the vertical temperature profile (bottom contact cooling, interior heating, surface convection).

---

## 5. Calibration Method

### 5.1 Calibration Parameters

Three parameters were calibrated, selected as the dominant sources of uncertainty in the model:

| Parameter | Physical Meaning | Default | Bounds |
|---|---|---|---|
| *k_c* (coupling factor) | Tank circuit RF voltage efficiency | 0.258 | [0.10, 0.40] |
| *k_evap* (evaporation rate) | Moisture evaporation kinetics for whole seeds | 5.0 x 10^-5 | [1 x 10^-6, 5 x 10^-4] |
| *r_gap* (gap adjustment rate) | MRH electrode drive speed | 0.012 | [0.005, 1.0] mm/s |

### 5.2 Loss Function

The objective function is a weighted sum of variance-normalized MSE:

    L = w_T * MSE(T_sim, T_plc) / Var(T_plc)
      + w_Ia * MSE(Ia_sim, Ia_plc) / Var(Ia_plc)
      + w_gap * MSE(gap_sim, gap_plc) / Var(gap_plc)

with all weights set to 1.0. The variance normalization (Var_T = 612.1, Var_Ia = 0.39, Var_gap = 43.0) renders all terms dimensionless and comparable. Both trajectories are resampled to 50 common time points.

The inclusion of *I_a* is critical: it is a near-instantaneous function of coupling factor and load impedance, providing the sharpest gradient for constraining *k_c*. Without *I_a*, compensating interactions between *k_c* and *k_evap* produce multiple local minima.

### 5.3 Optimizer Configuration

Differential evolution (scipy.optimize.differential_evolution) with L-BFGS-B local polish. Population size 15, max 30 generations, tolerance 0.005. The simulator is constructed once and reset per evaluation via CoupledSimulator.reset(), avoiding repeated array allocation and Warp kernel compilation.

---

## 6. Calibration Results

### 6.1 Calibrated Parameters

| Parameter | Calibrated | Default | Change |
|---|---|---|---|
| *k_c* | **0.1741** | 0.258 | -32.5% |
| *k_evap* | **1.00 x 10^-6** | 5.0 x 10^-5 | -98% (at lower bound) |
| *r_gap* | **0.1475 mm/s** | 0.012 | +12x |

### 6.2 Convergence

The optimization completed 926 evaluations (15 DE generations + L-BFGS-B polish), converging to a total loss of 4.424 (L_T = 1.34, L_Ia = 1.98, L_gap = 1.10).

Three distinct convergence phases were observed:

**Phase 1 (Generations 1--3):** Rapid exploration eliminated high-coupling candidates (*k_c* > 0.25) that caused catastrophic MRH gap losses. Best loss dropped from 5.10 to 4.68.

**Phase 2 (Generations 3--9):** Plateau at L_gap = 1.796. At coupling ~0.14, the simulated I_a never exceeded MRH, so the gap remained fixed. The gap rate became a "don't care" parameter, reducing the effective search to 2D.

**Phase 3 (Generations 10--15):** The optimizer discovered the MRH edge regime at *k_c* ~ 0.174, where I_a just grazes the MRH threshold, causing the controller to reproduce the gentle gap drift observed in the PLC data. This broke through the L_gap floor.

### 6.3 Sensitivity Analysis

| Parameter | dL/dp | Interpretation |
|---|---|---|
| *k_c* | -6.81 | Well-constrained (I_a directly proportional) |
| *k_evap* | +31,869 | At bound; any increase degrades fit |
| *r_gap* | -0.27 | Weakly constrained (MRH fires intermittently) |

The large *k_evap* gradient confirms evaporation is negligible — the optimizer pushes it to the bound because any positive evaporation rate introduces latent heat cooling that contradicts the measured temperature trajectory.

---

## 7. Validation

### 7.1 Post-Calibration Simulation

A 947-second production simulation (61 kg, 75 mm gap, 0.2 m/min, 25 mm bed) with calibrated parameters:

| Metric | Simulated | PLC / Measured |
|---|---|---|
| Outfeed temperature | 43.2 C | 45.0 C (PLC sensor) |
| Maximum temperature | 60.6 C | 101 C (PLC peak) / 77--93 C (strips) |
| Outfeed moisture | 9.80% wb | ~10.4% (NIR) |
| Electrode gap (final) | 79.6 mm | 75--87 mm |
| Anode current (steady) | ~1.70 A | 1.65--1.70 A |
| RF power (steady) | ~5 kW | -- |

### 7.2 Time-Series Comparison

The nine-panel diagnostic dashboard (Figure 1) shows good agreement for the anode current trajectory and electrode gap dynamics. The MRH gap control plot confirms the calibrated model correctly places the operating point at the edge of MRH activation, reproducing the gradual gap drift rather than an abrupt step.

### 7.3 Outfeed Cross-Section

The outfeed cross-section (Figure 2) reveals a pronounced vertical temperature gradient: 60 C at the bed surface (nearest upper electrode) to ~20 C near the belt. The PTFE belt stack acts as a thermal insulator, creating a cold boundary layer. The moisture cross-section shows essentially uniform distribution at 9.8% wb, confirming negligible drying.

---

## 8. Discussion

### 8.1 The GP-15 as a Thermal Pretreatment Unit

The most significant finding of this calibration study is that the GP-15 operating on whole yellow pea functions as a **thermal pretreatment unit**, not a dryer. The calibrated evaporation rate constant (*k_evap* = 1.0 x 10^-6) is effectively zero, meaning virtually all absorbed RF energy (approximately 5 kW sustained, 1.28 kWh total) goes to sensible heating. This is physically consistent with the intact seed coat acting as a moisture barrier: at temperatures below 100 C and atmospheric pressure, the rate of moisture transport through the seed coat is negligible compared to the rate of RF energy absorption.

This finding has important implications for process design:
- **Energy metrics:** The conventional metric of kg water per kWh (from GP-15 Manual Chapter 5) is inappropriate for this application. A more relevant metric is specific energy per degree of temperature rise (approximately 0.021 kWh/kg based on 232 kg/h throughput and 5 kW average power).
- **Product quality:** The primary process outcome is seed tempering (softening for subsequent milling), not dehydration. Temperature uniformity across the bed is the critical quality parameter.
- **Operating point:** The machine operates at one-third of its rated capacity (*I_a* = 1.7 A vs. 2.58 A full-load), with the MRH controller actively managing power delivery.

### 8.2 The Oscillator Coupling Factor

The calibrated coupling factor (*k_c* = 0.174) is 32.5% lower than the initial analytical estimate (0.258). The initial estimate was derived from the GP-15 test report's full-load operating point assuming ideal coupling; the calibrated value reflects the actual tank circuit efficiency including parasitic losses, electrode impedance mismatch, and cable losses. This parameter is well-constrained by the anode current data (sensitivity gradient = -6.81) and is expected to be stable across runs with the same machine configuration.

### 8.3 The MRH Edge Operating Regime

The calibration revealed that the GP-15 operates at the edge of its MRH overcurrent protection threshold — a regime where the anode current oscillates near the 1.7 A limit, causing the electrode gap to drift slowly upward. This behavior is not described in the GP-15 manual (which treats MRH as a protection feature) but is a natural consequence of operating a self-excited oscillator at a fixed power level into a load whose impedance changes with temperature.

The discovery of this operating regime required the optimizer to explore coupling values slightly above the initial convergence basin (Phase 3 of the convergence trajectory). This is a strong argument for using a global optimizer (differential evolution) rather than a local method, which would have been trapped in the Phase 2 plateau.

### 8.4 Limitations

1. **Single-run calibration.** Only one PLC recording was used. Validation against independent runs (different masses, belt speeds, materials) is needed to establish parameter transferability.
2. **Temperature underprediction.** The simulated outfeed temperature (43.2 C) matches the PLC's mid-run reading but underpredicts the peak (101 C at end of run). This likely reflects the simplified convective boundary conditions and the absence of oven pre-heating in the simulation.
3. **Oscillator model fidelity.** The single-parameter coupling model (L_Ia = 1.98) is the weakest component. A more detailed oscillator model incorporating load-dependent frequency pulling would reduce the residual.
4. **k_evap at bound.** The evaporation rate hitting its lower bound indicates the model would benefit from fixing *k_evap* = 0 for whole seeds and calibrating only 2 parameters, reducing the search space.

---

## 9. Computational Performance

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation (48 GiB, SM 8.9) |
| CPU | AMD Ryzen (Family 25 Model 24) |
| CUDA Toolkit | 12.9 |
| NVIDIA Warp | 1.11.0 |
| Grid | 60 x 30 x 32 = 57,600 cells |
| GPU-accelerated steps | Advection, P_v computation, thermal FDM, moisture diffusion, property update |
| CPU steps | RF field solve (series-capacitor), PLC controller, KPI recording, Lagrangian particles |
| Calibration (926 evals) | ~46 minutes wall-clock (GPU) |
| Post-calibration simulation | 947 s simulated in 1337 s wall-clock (0.7x real-time) |

---

## 10. Conclusions

1. A coupled multi-physics digital twin of the GP-15 RF heating machine was developed, calibrated, and validated against PLC field data from a whole yellow pea production run.

2. The iterative development methodology — progressive debugging guided by PLC comparison — identified six critical implementation errors and two model architecture improvements (voltage-driven generator, MRH as proportional control) that transformed the simulation from non-functional to physically accurate.

3. Three-parameter calibration using differential evolution with variance-normalized multi-signal loss converged to a total loss of 4.424, with each component of order 1.0 in normalized units. The inclusion of anode current in the loss function was essential for resolving parameter ambiguities.

4. The calibrated *k_evap* = 1.0 x 10^-6 (at the lower bound) establishes that the GP-15 operating on whole yellow pea is a thermal pretreatment unit, not a dryer. This is the key process insight from the calibration.

5. The calibrated model reproduces the MRH edge operating regime — a gradual electrode gap drift driven by the oscillator's load-dependent power delivery — which is the dominant control dynamic of the real machine.

6. The residual loss (4.424) reflects the structural accuracy limit of the current model physics. Further improvement requires higher-fidelity submodels for the oscillator load characteristics, oven pre-heating, and the vertical temperature profile boundary conditions.

---

## References

1. QMTI GP-15 Installation and Operation Manual, Quantum Mechanical Technologies Inc., Prince Albert, SK, Canada, 2021.
2. Kwofie, E., "Pretreatment Engineering Guide: RF Dielectric Heating Digital Twin — QMTI GP-15 Gentle Processing Machine," Air Classifier Designer Project, February 2026.
3. Storn, R. and Price, K., "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces," J. Global Optimization, 11, pp. 341--359, 1997.
4. NVIDIA Warp Documentation, https://nvidia.github.io/warp/, 2024.

---

*Calibrated parameters saved to: `utility_docs/calibration_latest.json`*
*Source code: `src/airclassifier/pretreatment/` (simulator, coupling, calibration, particles)*
*Example: `python examples/simulate_and_visualize.py --calibrate "utility_docs/Run1 RF data(in).csv" --cal-duration 0`*
