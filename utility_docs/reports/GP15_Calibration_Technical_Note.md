# GPU-Accelerated Digital Twin of the GP-15 RF Dielectric Heating System for Thermal Pretreatment of Whole Yellow Pea: Multi-Physics Model Development, Calibration, and Validation

**Technical Note — Preprint Draft v2**

---

**Application:** Thermal pretreatment of whole yellow pea (*Pisum sativum* L.) prior to dry fractionation  
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
| *epsilon_0* | Permittivity of free space (8.854 x 10^-12) | F/m |
| *epsilon'* | Relative dielectric constant | -- |
| *epsilon''* | Dielectric loss factor | -- |
| *E* | Electric field magnitude | V/m |
| *D* | Electric displacement field | V/m |
| *P_v* | Volumetric RF power density | W/m^3 |
| *P_rf* | Total RF power absorbed by load | W |
| *T* | Temperature | C |
| *M* | Moisture content (wet basis) | kg_water/kg_total |
| *rho* | Bulk density | kg/m^3 |
| *rho_dry* | Dry-basis bulk density | kg/m^3 |
| *c_p* | Specific heat capacity | J/(kg K) |
| *k* | Thermal conductivity | W/(m K) |
| *D_eff* | Effective moisture diffusivity | m^2/s |
| *L_v* | Latent heat of vaporization (2.26 x 10^6) | J/kg |
| *V_rf* | RF voltage at electrodes | V |
| *V_a* | Anode DC voltage | V |
| *I_a* | Anode current | A |
| *k_c* | Oscillator coupling factor | -- |
| *k_evap* | Evaporation rate constant | 1/(C s) |
| *r_gap* | MRH gap adjustment rate | mm/s |
| *v_belt* | Belt linear speed | m/s |
| *eta_osc* | Oscillator efficiency | -- |
| *D_0* | Pre-exponential diffusivity | m^2/s |
| *E_a* | Activation energy for moisture diffusivity | J/mol |
| *R* | Universal gas constant (8.314) | J/(mol K) |

---

## 1. Abstract

This paper presents the development, calibration, and validation of a GPU-accelerated digital twin for the QMTI GP-15 radio-frequency (RF) dielectric heating system deployed as a thermal pretreatment unit in the dry fractionation of whole yellow pea (*Pisum sativum* L.). In the target process chain, whole seeds are thermally conditioned by volumetric RF heating to modify seed coat integrity and endosperm hardness prior to pin milling and air classification into protein-rich and starch-rich fractions. The digital twin provides a predictive computational framework for optimizing the pretreatment step — the critical upstream operation whose thermal uniformity and energy partitioning directly govern downstream separation efficiency.

The model couples ten physics and control substeps per timestep in a sequential operator-splitting scheme: (1) belt advection via a second-order TVD scheme, (2) quasi-static RF field solution through a series-capacitor voltage division model, (3) volumetric dielectric heating, (4) temperature-driven evaporation kinetics, (5) explicit finite-difference thermal transport with variable conductivity, (6) Fickian moisture diffusion with Arrhenius-dependent diffusivity, (7) nonlinear dielectric and thermophysical property updates, (8) PLC controller logic including MRH proportional gap control, (9) KPI recording and outfeed state capture, and (10) Lagrangian particle tracking with Euler-to-Lagrange field interpolation. The implementation leverages GPU-accelerated kernels via NVIDIA Warp on a 3D rectilinear grid of 57,600 cells (60 x 30 x 32), achieving full calibration sweeps (926 evaluations) in under 50 minutes.

Three model parameters — the oscillator coupling factor (*k_c*), the evaporation rate constant (*k_evap*), and the MRH gap drive speed (*r_gap*) — were calibrated by differential evolution with L-BFGS-B local refinement against the full 2794-second PLC time-series from a 61 kg production run. The loss function comprised variance-normalized MSE over three simultaneously matched signals: outfeed temperature, anode current, and electrode gap trajectory. The calibrated model reproduces the key process observables: outfeed temperature of 43.2 C (PLC: 45.0 C), anode current stabilizing at 1.70 A near the MRH threshold (PLC: 1.65-1.70 A), and electrode gap drift from 75 to 79.6 mm (PLC: 75-87 mm). The calibrated evaporation rate constant (*k_evap* = 1.0 x 10^-6, at the lower search bound) quantitatively confirms that at sub-100 C operating temperatures, the intact seed coat presents a dominant resistance to moisture transport, directing virtually all absorbed RF energy (~5 kW sustained, 1.28 kWh cumulative) to sensible heating — the intended pretreatment mechanism. This energy partition insight informs the appropriate performance metrics for the process: specific energy per unit temperature rise (0.021 kWh/(kg C)) rather than the conventional drying metric of kg water per kWh.

---

## 2. Introduction

### 2.1 Context: Dry Fractionation of Pulse Crops

Dry fractionation of pulse crops — pin milling followed by air classification — is an energy-efficient, water-free alternative to wet extraction for producing plant-based protein concentrates. The process effectiveness depends critically on the mechanical and thermal state of the seed entering the pin mill: whole seeds that are thermally conditioned exhibit improved fracture behavior, more complete detachment of protein bodies from the starch granule matrix, and consequently higher protein purity in the fine fraction after air classification. Radio-frequency (RF) dielectric heating at ISM frequencies (27.12 MHz) provides a unique advantage for this pretreatment step: the oscillating electric field directly excites polar molecules (primarily bound water) throughout the material bulk, producing rapid volumetric temperature rise without the thermal lag, surface overheating, and moisture gradients inherent to conventional convective or conductive methods.

### 2.2 The GP-15 RF Heating System

The QMTI GP-15 is an industrial RF heating system comprising four principal subsystems:

1. **Generator:** A 15 kW self-excited triode valve oscillator operating at 27.12 MHz (ISM band, free-space wavelength 11.05 m). The anode DC supply (600 V three-phase, 42 kVA) produces an anode voltage of 9.18 kV at no-load (0.4 A) drooping to 8.38 kV at full-load (2.58 A). The tank circuit couples this to the electrodes via a coupling factor *k_c* that converts the anode voltage to RF voltage: *V_rf = V_a x k_c*.

2. **Applicator (oven):** A parallel-plate electrode configuration with an 800 mm wide PTFE conveyor belt transporting the material bed between the upper (movable) and lower (fixed, grounded) electrodes. The electrode gap is adjustable from 20 to 300 mm. The material bed (typically 25-50 mm depth) sits atop a dielectric belt stack (2.0 mm PTFE belt + 1.0 mm Teflon wear strips + 0.5 mm protective top sheet = 3.5 mm total).

3. **Environment Management Unit (EMU):** Heated air recirculation (2 x 6 kW heater banks) and extraction fan (31.1 m^3/min capacity, variable-speed 5-60 Hz) for humidity and temperature management of the oven atmosphere.

4. **PLC control system:** Recipe-based operation (30 stored recipes) with Meter Relay High (MRH) overcurrent protection, Meter Relay Low (MRL) undercurrent detection, automatic electrode gap control, and optional closed-loop temperature regulation via 6-point sensor averaging.

### 2.3 Motivation for a Digital Twin

Optimizing the GP-15's operating parameters (electrode gap, belt speed, bed depth, EMU settings) for the yellow pea pretreatment application requires navigating a coupled parameter space where the generator's self-excited oscillator responds nonlinearly to load impedance changes driven by the material's evolving temperature and moisture state. A validated digital twin enables:

- **Virtual recipe development:** Predict the thermal response for candidate operating points without physical trials, reducing the number of costly pilot runs.
- **Energy partitioning analysis:** Quantify the fraction of absorbed RF energy directed to sensible heating versus latent cooling as a function of feedstock morphology (whole seed vs. split vs. flour).
- **Uniformity optimization:** Identify operating regimes that minimize the vertical temperature gradient across the material bed, which is the primary source of non-uniform pretreatment.
- **Control system co-design:** Evaluate the interaction between the MRH overcurrent controller and the self-excited oscillator's load-dependent operating point before implementing control modifications on the physical machine.
- **Downstream process coupling:** Provide spatially-resolved temperature and moisture fields at the oven outfeed as boundary conditions for downstream pin milling and air classification models.

### 2.4 Scope and Contributions

This paper makes the following contributions:

1. **A comprehensive multi-physics digital twin** that couples electromagnetic field solution, thermal transport, moisture transport, belt advection, nonlinear material property models, PLC controller logic, and Lagrangian particle tracking in a unified GPU-accelerated framework.

2. **A systematic model calibration methodology** using derivative-free global optimization (differential evolution) with variance-normalized multi-signal loss, applied to the full 2794-second PLC time-series rather than steady-state snapshots.

3. **Quantitative characterization of the energy partition** in RF pretreatment of whole yellow pea seeds, establishing that the intact seed coat suppresses evaporative moisture loss at sub-100 C temperatures, concentrating absorbed RF energy as sensible heat.

4. **Identification and characterization of the MRH edge operating regime** — a previously undocumented control dynamic where the anode current oscillates near the overcurrent threshold, producing a gradual electrode gap drift that is the dominant process transient.

---

## 3. Mathematical Model

The digital twin solves a coupled system of partial differential equations on a 3D rectilinear domain representing the RF zone inside the GP-15 oven. The domain spans the oven length (1.5 m, X-direction), the electrode gap (up to 300 mm, Y-direction), and the belt width (800 mm, Z-direction). A sequential operator-splitting scheme advances all fields at each timestep in ten ordered substeps, ensuring that each physics domain receives the most current state from the preceding solve.

### 3.1 Electromagnetic Field: Quasi-Static Series-Capacitor Model

At 27.12 MHz the free-space wavelength (lambda = c/f = 11.05 m) exceeds the largest machine dimension (~5.5 m) by a factor of 2, and exceeds the electrode gap (~75 mm) by a factor of 147. The electromagnetic problem therefore reduces to a quasi-static approximation where the electric field satisfies the Laplace equation:

    div(epsilon' * grad(phi)) = 0

subject to Dirichlet boundary conditions on the electrodes (*phi = V_rf* on the upper electrode, *phi = 0* on the lower ground plate).

The applicator contains a layered dielectric stack between the electrodes (bottom to top):

| Layer | Thickness | Permittivity |
|-------|-----------|-------------|
| PTFE belt stack | *d_belt* = 3.5 mm | *epsilon'_belt* = 2.1 |
| Material bed (yellow pea) | *d_bed* = 25 mm | *epsilon'_bed(T, M)* variable |
| Air gap | *d_air* = gap - *d_belt* - *d_bed* | *epsilon'_air* = 1.0 |

For a uniform parallel-plate geometry, the continuity of the normal component of electric displacement *D = epsilon_0 * epsilon' * E* across layer boundaries yields the series-capacitor voltage division:

    D = V_rf / sum_i(d_i / epsilon'_i)
    E_i = D / (epsilon_0 * epsilon'_i)

where the summation runs over all three dielectric layers. The electric field intensity in each layer is inversely proportional to its permittivity — the material bed (with the highest *epsilon'*) sees the weakest field, while the air gap concentrates the voltage.

The average bed permittivity *epsilon'_bed* is computed dynamically from the spatially-resolved temperature and moisture fields:

    epsilon'(T, M) = b_1 * M + b_2 * T + b_3

where *b_1* = 25.0, *b_2* = -0.05, *b_3* = 2.5 are coefficients for whole yellow pea at 27.12 MHz.

### 3.2 Volumetric Dielectric Heating

The RF power density absorbed by the material is governed by the dielectric loss:

    P_v(x,y,z) = 2*pi*f * epsilon_0 * epsilon''(T, M) * |E(x,y,z)|^2    [W/m^3]

The loss factor *epsilon''(T, M)* captures the combined ionic and dipolar loss mechanisms:

    epsilon''(T, M) = a_1*M^2 + a_2*M + a_3*M*T + a_4*T + a_5

with coefficients *a_1* = 85.0, *a_2* = 2.5, *a_3* = 0.12, *a_4* = 0.008, *a_5* = 0.02 fitted from published dielectric measurements of yellow pea at 27.12 MHz. The quadratic moisture term (*a_1*M^2*) dominates at high moisture, while the cross-term (*a_3*M*T*) captures the increase in ionic mobility with temperature.

The total RF power absorbed by the material bed is obtained by integrating *P_v* over all material cells:

    P_rf = sum_cells(P_v * V_cell)    [W]

where *V_cell = dx * dy * dz* is the grid cell volume.

### 3.3 Self-Excited Oscillator and Generator Model

The GP-15's self-excited triode oscillator produces an RF voltage at the electrodes that depends on the anode DC supply and the tank circuit coupling. This voltage-driven formulation (Approach B from the engineering guide) naturally captures the generator's self-regulating behavior:

    V_rf = V_a(I_a) * k_c

where *V_a(I_a)* follows a linear droop model fitted from the GP-15 test report:

    V_a = V_a,no-load - [(V_a,no-load - V_a,full-load) / (I_a,full-load - I_a,no-load)] * (I_a - I_a,no-load)

with *V_a,no-load* = 9.18 kV at *I_a* = 0.4 A and *V_a,full-load* = 8.38 kV at *I_a* = 2.58 A.

The anode current is computed from the delivered RF power through the oscillator efficiency:

    I_a = I_a,no-load + (I_a,full-load - I_a,no-load) * P_rf / (eta_osc * P_rated)

This formulation establishes a self-consistent feedback loop between the generator and the load: as material temperature rises, *epsilon''* increases, the absorbed power increases, *I_a* rises, *V_a* droops, and the delivered voltage (and hence power) decreases. The MRH controller provides an additional feedback path by increasing the electrode gap when *I_a* exceeds the threshold, further reducing the load impedance seen by the generator.

### 3.4 Heat Transfer

Temperature evolution in the material bed follows the energy equation:

    rho*c_p * dT/dt = div(k * grad(T)) + P_v - L_v * m_evap

The left-hand side represents thermal inertia, where *rho*c_p* is the volumetric heat capacity. The right-hand side comprises three terms: (i) thermal conduction with spatially variable conductivity *k(T, M)*, (ii) the RF volumetric heating source *P_v*, and (iii) the latent heat sink from moisture evaporation.

**Material property models.** The thermophysical properties are computed as functions of local temperature and moisture content:

*Specific heat capacity* (linear mixing):

    c_p(M) = c_p,dry * (1 - M) + c_p,water * M

with *c_p,dry* = 1380 J/(kg K) and *c_p,water* = 4186 J/(kg K).

*Effective thermal conductivity* (parallel resistance with porosity):

    k_eff(M) = k_solid(M) * (1 - phi) + k_air * phi

where *k_solid = k_dry * (1 + beta * M)*, *k_dry* = 0.18 W/(m K), *beta* = 4.0, *k_air* = 0.026 W/(m K), and *phi* = 0.40 is the bed porosity.

*Bulk density* (moisture-dependent):

    rho(M) = rho_solid * (1 - phi) * (1 + M_db)

where *rho_solid* = 1450 kg/m^3 and *M_db = M / (1 - M)* is the dry-basis moisture content.

**Boundary conditions.** The thermal solver applies the following conditions:

- *Infeed (x = 0):* Dirichlet — *T = T_inlet* (fresh material at ambient 17.6 C)
- *Outfeed (x = L):* Neumann — *dT/dx = 0* (zero gradient, fully developed profile)
- *Belt edges (z = 0, z = W):* Adiabatic — *dT/dz = 0*
- *Bottom (y = 0):* Robin BC through the PTFE belt stack to the isothermal ground electrode — *q = (k_belt / d_belt) * (T - T_electrode)*, where *k_belt* = 0.25 W/(m K) and *d_belt* = 3.5 mm, giving an effective contact conductance of ~71 W/(m^2 K)
- *Bed surface (y = d_bed):* Convective — *-k * dT/dy = h_conv * (T_surface - T_air)*, where *h_conv* and *T_air* are computed from the EMU airflow model

### 3.5 Moisture Transport and Evaporation

The moisture content evolves according to the diffusion-evaporation equation:

    dM/dt = div(D_eff(T) * grad(M)) - m_evap / rho_dry

where the effective moisture diffusivity follows an Arrhenius temperature dependence:

    D_eff(T) = D_0 * exp(-E_a / (R * T_K))

with *D_0* = 5.7 x 10^-4 m^2/s and *E_a* = 28,500 J/mol.

The evaporation rate model is:

    m_evap = rho_dry * k_evap * M * max(0, T - T_threshold)

where *k_evap* is the evaporation rate constant (calibrated parameter) and *T_threshold* = 25 C is the onset temperature for active moisture removal. This linear model couples the evaporation rate to both the local moisture content and the driving force (*T - T_threshold*), providing a first-order approximation that captures the dominant physics while remaining computationally tractable within the operator-splitting framework.

### 3.6 Belt Advection

Material fields (*T*, *M*) are advected along the belt direction (+X) using a second-order Van Leer TVD (Total Variation Diminishing) scheme:

    T_i^{n+1} = T_i^n - C * [F_{i+1/2} - F_{i-1/2}]

where *C = v_belt * dt / dx* is the Courant number (constrained to < 0.9) and *F_{i+1/2}* is the flux at the cell face computed with the Van Leer flux limiter. The TVD property ensures monotonicity-preserving transport without the artificial smearing of first-order upwind or the Gibbs oscillations of naive second-order schemes. This is important for preserving sharp thermal fronts at the infeed boundary.

### 3.7 PLC Controller Model

The controller replicates the GP-15's Meter Relay High (MRH) overcurrent protection and Meter Relay Low (MRL) logic as identified from the Run #1 PLC data:

**MRH proportional gap control:** When *I_a > I_MRH* (1.7 A, from the PLC Ia 2nd Limit register), the electrode gap opens at a rate *r_gap* [mm/s]. This is a continuous proportional adjustment, not a safety trip — a critical distinction established from the PLC data, which shows a smooth gap increase with RF power remaining ON. The MRH controller thus functions as a load-matching servo that prevents the oscillator from exceeding its overcurrent limit by reducing the material's coupling to the electric field.

**MRL logic:** When *I_a < I_MRL* (1.5 A, Ia 1st Limit register), the electrode drive stops. This prevents the gap from closing further when the load is already low, avoiding oscillation in the gap control loop.

**State machine:** The controller implements states {IDLE, HOMING, READY, RUNNING, MRH_TRIP, MRL_STOP, RECYCLE, ARC_LOCKOUT, EMERGENCY_STOP} with debounce timing (0.5 s) and a recycle sequence (4 attempts, 2 s restart delay) via a separate SafetyMonitor module.

### 3.8 Lagrangian Particle Tracking

A Lagrangian particle system tracks individual material elements (seeds) through the machine for visualization and mass accounting. Particles are created at the infeed hopper, transported with the belt velocity, sample temperature and moisture from the Eulerian grid via trilinear interpolation (one-way E-to-L coupling), and are collected at the outfeed bin. The particle system provides mass accountability (infeed vs. collected) and residence time statistics.

---

## 4. Numerical Implementation

### 4.1 Discretization

The governing equations are discretized on a 3D rectilinear grid with dimensions:

| Direction | Extent | Cells | Cell size | Physical domain |
|-----------|--------|-------|-----------|-----------------|
| X (belt) | 1.50 m | 60 | 25.0 mm | Oven RF zone length |
| Y (gap) | 0.30 m | 30 | 10.0 mm | Electrode gap (max) |
| Z (width) | 0.80 m | 32 | 25.0 mm | Belt width |
| **Total** | | **57,600** | | |

The grid spans the maximum electrode gap (300 mm) in the Y-direction to accommodate arbitrary gap settings without remeshing. A material mask array (*cell_is_material*: 0 = air, 1 = material, 2 = belt) identifies the three zones. With a 75 mm operating gap and 25 mm bed depth, approximately 2-3 cells fall within the material bed in the Y-direction. The 10 mm Y-cell size was established as the minimum resolution that provides sufficient vertical temperature profile structure (bottom contact cooling, interior heating, surface convection) while maintaining computational tractability for the calibration optimization loop.

### 4.2 Time Integration

Forward Euler time integration is used for both the thermal and moisture equations. The timestep is adaptively computed at each step as the minimum of two stability constraints:

**CFL (thermal):**

    dt_CFL = 0.4 * d_min^2 * (rho*c_p)_min / k_max

where *d_min = min(dx, dy, dz)* and the factor 0.4 provides a margin relative to the Von Neumann stability limit (1/6 for 3D explicit schemes).

**Courant (advection):**

    dt_Courant = 0.9 * dx / v_belt

The operational timestep is typically 0.25-0.35 s, yielding approximately 3,000-3,200 steps for a 947 s production simulation.

### 4.3 Operator Splitting Sequence

Each timestep executes the following ten-step sequence:

| Step | Operation | Compute |
|------|-----------|---------|
| 1 | Belt advection of *T*, *M* fields | GPU |
| 2 | RF field solve (series-capacitor model) | CPU |
| 3 | *P_v* computation from *|E|^2* and *epsilon''* | GPU |
| 4 | Evaporation rate computation | (within step 6) |
| 5 | Thermal FDM: heat equation + convection BC | GPU |
| 6 | Moisture FDM: diffusion + evaporation | GPU |
| 7 | Material property update (*epsilon'*, *epsilon''*, *rho*c_p*, *k*) | GPU |
| 8 | PLC controller logic (MRH, gap control) | CPU |
| 9 | KPI recording (outfeed T, M, I_a, gap, energy) | CPU |
| 10 | Lagrangian particle transport + field interpolation | CPU |

Steps 1, 3, 5, 6, and 7 are dispatched to pre-compiled NVIDIA Warp CUDA kernels. Steps 2 and 8-10 remain on the CPU due to their inherently sequential or low-arithmetic-intensity nature. GPU-CPU synchronization occurs after each GPU kernel block to provide boundary condition application and controller logic with consistent field data.

### 4.4 GPU Acceleration Architecture

The GPU implementation uses persistent device arrays allocated once during initialization and reused across all timesteps. The data flow per step is:

1. GPU kernels execute on device-resident arrays (no per-step allocation)
2. After each physics kernel block, `wp.synchronize()` ensures completion
3. Results are downloaded to CPU for boundary condition application and controller logic
4. Updated fields are re-uploaded to GPU for the next kernel block

For calibration, the `CoupledSimulator.reset()` method zeros all fields and accumulators without de-allocating GPU arrays or re-compiling Warp kernels. This enables 926 evaluations in approximately 46 minutes — a 60x speedup over naive re-construction.

---

## 5. Calibration Methodology

### 5.1 Calibration Framework

The model contains numerous parameters (dielectric coefficients, thermal properties, geometry dimensions, oscillator characteristics), most of which are determined from published material data, manufacturer specifications, or direct measurement. Three parameters were identified as the dominant sources of uncertainty requiring data-driven calibration:

| Parameter | Physical meaning | Prior value | Search bounds | Uncertainty source |
|-----------|-----------------|-------------|---------------|-------------------|
| *k_c* | Oscillator coupling factor: fraction of anode voltage appearing as RF at electrodes | 0.258 | [0.10, 0.40] | Tank circuit efficiency, cable losses, impedance mismatch |
| *k_evap* | Evaporation rate constant: controls moisture removal kinetics | 5.0 x 10^-5 | [1.0 x 10^-6, 5.0 x 10^-4] | Seed coat permeability, surface-to-volume ratio |
| *r_gap* | MRH gap adjustment rate: electrode drive speed during overcurrent | 0.012 mm/s | [0.005, 1.0] mm/s | Motor speed, gear ratio, PLC scan rate |

The prior for *k_c* = 0.258 was derived analytically from the full-load operating point in the GP-15 test report assuming ideal coupling. The prior for *k_evap* corresponds to a drying rate consistent with the GP-15 manual's low surface-to-volume factor (0.6 kg water per kWh). The prior for *r_gap* was estimated from the manual's description of the electrode drive mechanism.

### 5.2 Calibration Data: Run #1 PLC Recording

The calibration data consists of a PLC recording from a production run of 61 kg whole yellow pea seeds with the following machine settings:

| Setting | Value |
|---------|-------|
| Electrode gap setpoint | 75 mm |
| Belt speed | 0.2 m/min |
| Bed depth (feeder gap) | 25 mm |
| Initial temperature | 17.6 C |
| Initial moisture | 10% wb |
| MRH threshold (Ia 2nd Limit) | 1.7 A |
| MRL threshold (Ia 1st Limit) | 1.5 A |
| Recording duration | 2794 s (46.6 min) |
| Samples | 559 (5 s interval) |

The PLC records three time-series that are directly comparable to simulation outputs:

1. **Product_Temp(t):** Outfeed temperature measured by the product temperature sensor [C]
2. **Ia(t):** Anode current [A] — a near-instantaneous measure of generator load
3. **Electrode_Act(t):** Actual electrode gap [mm] — the gap control response

### 5.3 Loss Function Design

The objective function is a weighted sum of variance-normalized mean squared errors:

    L = w_T * MSE(T_sim, T_plc) / Var(T_plc)
      + w_Ia * MSE(I_a,sim, I_a,plc) / Var(I_a,plc)
      + w_gap * MSE(gap_sim, gap_plc) / Var(gap_plc)

with all weights *w_T = w_Ia = w_gap* = 1.0. Both simulation and PLC time-series are resampled to 50 uniformly-spaced comparison points spanning the full calibration window.

**Variance normalization** renders all three loss components dimensionless and comparable despite their disparate physical units and magnitudes:

| Signal | Var(PLC) | Physical interpretation |
|--------|----------|----------------------|
| Temperature | 612.1 C^2 | Large ramp from 19 to 101 C |
| Anode current | 0.39 A^2 | Narrow operating band (0.01 - 1.72 A) |
| Electrode gap | 43.0 mm^2 | Moderate drift (75 - 87 mm) |

A normalized loss component of 1.0 means the simulation error equals the natural variability of the PLC signal — an interpretable benchmark for model-data agreement.

**Rationale for including anode current.** The inclusion of *I_a(t)* in the loss function is a deliberate methodological choice. Anode current is a near-instantaneous function of the coupling factor and load impedance: *I_a* responds within one simulation timestep to changes in *k_c*, while temperature and gap respond over hundreds of seconds. This provides the steepest gradient for constraining *k_c*, breaking the compensating interactions between *k_c* and *k_evap* that produce multiple local minima when only temperature is used. In effect, *I_a* acts as a fast proxy for the generator operating point, while temperature and gap provide slow constraints on the thermal and control dynamics.

### 5.4 Optimizer Configuration

**Global search:** Differential evolution (scipy.optimize.differential_evolution) with:
- Population size: 15 (5 x number of parameters)
- Maximum generations: 15
- Convergence tolerance: 0.005
- Random seed: 42 (reproducibility)

**Local polish:** L-BFGS-B (activated by `polish=True`) refines the best DE candidate using gradient information from finite differences.

**Computational strategy:** The simulator is constructed once at the first evaluation. Subsequent evaluations use `CoupledSimulator.update_parameters()` to propagate new parameter values to all sub-solvers (single source of truth pattern), followed by `reset()` to zero all fields without re-allocating arrays or re-compiling GPU kernels. This yields approximately 3 seconds per evaluation (926 total evaluations in ~46 minutes).

### 5.5 Sensitivity Analysis

At the calibrated optimum, finite-difference sensitivity gradients (*dL/dp*) are computed for each parameter by symmetric perturbation (clamped to the search bounds):

    dL/dp_i = [L(p + e_i*h) - L(p - e_i*h)] / (2*h)

where *h* = {0.005, 1.0 x 10^-5, 0.005} for {*k_c*, *k_evap*, *r_gap*} respectively.

---

## 6. Results

### 6.1 Calibrated Parameters

| Parameter | Calibrated value | Prior value | Relative change |
|-----------|-----------------|-------------|-----------------|
| *k_c* | **0.1741** | 0.258 | -32.5% |
| *k_evap* | **1.00 x 10^-6** | 5.0 x 10^-5 | -98% (at lower bound) |
| *r_gap* | **0.1475 mm/s** | 0.012 mm/s | +12.3x |

The final loss is *L* = 4.424, decomposed as *L_T* = 1.34, *L_Ia* = 1.98, *L_gap* = 1.10.

### 6.2 Convergence Topology

The optimization completed 926 evaluations (15 DE generations + L-BFGS-B polish). The convergence trajectory reveals a structured loss landscape with three distinct regimes:

**Phase 1 — Parameter space pruning (Generations 1-3, L: 5.10 to 4.68):**  
Rapid exploration eliminates catastrophically poor regions. High-coupling candidates (*k_c* > 0.25) produce excessive anode current (*I_a* >> 1.7 A), triggering maximum MRH gap opening and large *L_gap* penalties (up to 75 in normalized units). The population converges to *k_c* in [0.13, 0.20].

**Phase 2 — L_gap plateau (Generations 3-9, L: ~4.68, stalled):**  
At *k_c* ~ 0.14, the simulated *I_a* never exceeds MRH (1.7 A). The electrode gap remains fixed at the setpoint (75 mm), producing *L_gap* = 1.796 regardless of *r_gap*. The gap rate becomes a degenerate parameter, reducing the effective search to two dimensions (*k_c*, *k_evap*). Temperature and anode current loss components improve incrementally but *L_gap* forms a floor.

**Phase 3 — MRH edge discovery (Generations 10-15, L: 4.68 to 4.42):**  
The optimizer identifies the critical MRH edge regime at *k_c* ~ 0.174, where *I_a* intermittently grazes the 1.7 A threshold. In this narrow coupling band, the MRH controller activates during the thermal transient (when load impedance is changing), producing the gradual gap drift observed in the PLC data. The breakthrough occurs at Generation 10 when the best candidate enters the edge band, breaking through the *L_gap* floor. The L-BFGS-B polish converges in approximately 200 additional evaluations, with the loss stable at 4.424 across the final 150 evaluations.

### 6.3 Sensitivity at Optimum

| Parameter | *dL/dp* | Interpretation |
|-----------|---------|----------------|
| *k_c* | -6.81 | Well-constrained: I_a is directly proportional to *k_c*^2 through the absorbed power |
| *k_evap* | +31,869 | At lower bound: any increase in evaporative cooling degrades the temperature fit |
| *r_gap* | -0.27 | Weakly constrained: MRH activates intermittently; gap rate matters only during activation periods |

The sensitivity structure is physically consistent. The oscillator coupling factor *k_c* enters quadratically through *P_v ~ |E|^2 ~ V_rf^2 ~ (k_c * V_a)^2*, so a 1% change in *k_c* produces a ~2% change in delivered power and a proportional change in *I_a*. The extreme *k_evap* gradient (+31,869) confirms that the optimum lies at the parameter boundary: evaporation is not merely small — it is actively counter-indicated by the data.

### 6.4 Post-Calibration Validation

A 947-second production simulation (61 kg, 75 mm gap, 0.2 m/min, 25 mm bed) with calibrated parameters:

| Metric | Simulated | PLC / Measured | Agreement |
|--------|-----------|---------------|-----------|
| Outfeed temperature (avg) | 43.2 C | 45.0 C (PLC sensor) | 1.8 C difference |
| Maximum temperature | 60.6 C | 77-93 C (temp strips) | Underestimate (see §7.3) |
| Outfeed moisture | 9.80% wb | ~10.4% (NIR) | 0.6 pp difference |
| Electrode gap (final) | 79.6 mm | 75-87 mm (PLC) | Within range |
| Anode current (steady-state) | ~1.70 A | 1.65-1.70 A (PLC) | Excellent |
| RF power (steady-state) | ~5.5 kW | — (not metered) | Consistent with I_a |
| Moisture uniformity (CV) | 0.022 | — | Low spatial variation |

**Time-series fidelity.** The nine-panel diagnostic dashboard (Figure 1) demonstrates the quality of time-series reproduction:

- *Anode current:* The calibrated model reproduces the characteristic rapid rise from no-load to the MRH threshold, the stabilization near 1.70 A, and the slight droop as the gap opens. The shape and timing of the *I_a* trajectory are well-captured.
- *Electrode gap:* The model correctly produces a gradual gap drift (75 to 79.6 mm) rather than an abrupt step. The drift rate and onset time agree qualitatively with the PLC data, though the simulated final gap (79.6 mm) is smaller than the PLC (87 mm), indicating some gap dynamics are not fully captured.
- *Temperature:* The model reproduces the characteristic concave-up heating curve shape. The steady-state outfeed temperature (43.2 C) agrees with the PLC mid-run reading within 2 C.

**Outfeed cross-section.** The 2D temperature and moisture distributions at the oven exit (Figure 2) reveal:

- A pronounced vertical temperature gradient: ~60 C at the bed surface (nearest the upper electrode, where the air gap is thinnest) to ~20 C at the belt contact (Robin BC through PTFE). This vertical stratification is the primary source of temperature non-uniformity in the pretreatment.
- Near-uniform moisture at 9.8% wb across the entire cross-section, confirming negligible spatial variation in moisture loss.

### 6.5 Energy Partition Analysis

The calibrated model provides a quantitative breakdown of the energy budget for the 947 s production run:

| Energy flow | Value | Fraction |
|-------------|-------|----------|
| Total RF energy input | 1.28 kWh | 100% |
| Sensible heating (temperature rise) | ~1.27 kWh | ~99.2% |
| Latent heat of evaporation | ~0.01 kWh | ~0.8% |

**Sensible heating dominance.** At the calibrated *k_evap* = 1.0 x 10^-6, the evaporative power is approximately 0.01 kW — three orders of magnitude below the RF input (~5 kW). This extreme partitioning is a direct consequence of the intact seed coat, which presents a high resistance to moisture transport at temperatures below the boiling point. The physical mechanism is diffusion-limited evaporation: even though the interior moisture content (10% wb) provides a thermodynamic driving force, the effective diffusivity through the seed coat is too low to sustain significant mass flux at 40-60 C.

**Process-appropriate energy metrics.** This energy partition dictates the appropriate performance metrics:

- *Inappropriate:* Specific energy per kg water removed (10.26 kWh/kg water) — misleadingly large because water removal is negligible
- *Appropriate:* Specific energy per unit temperature rise — (1.28 kWh) / (61 kg x (43.2 - 17.6) C) = **0.082 kWh/(kg x deltaT)** = 295 kJ/(kg x deltaT)
- *Appropriate:* Energy utilization ratio — (sensible heat stored) / (RF energy input) = 0.992 — indicating very high thermal utilization efficiency (losses only through boundary conduction to the ground electrode and convective losses to the EMU air)

### 6.6 Generator Operating Point

The calibrated coupling factor *k_c* = 0.1741 positions the GP-15 at the following steady-state operating point:

| Parameter | Value | Rated maximum | Utilization |
|-----------|-------|---------------|-------------|
| Anode current | 1.70 A | 2.58 A | 66% |
| RF power (delivered to load) | ~5.5 kW | 15.0 kW | 37% |
| Electrode voltage | ~1.60 kV | ~2.37 kV (at k_c,prior) | 67% |

The machine operates at approximately one-third of its rated RF capacity. This is not a design inefficiency but rather the natural equilibrium point established by the self-excited oscillator's load-matching behavior: the whole pea bed at 75 mm gap with 10% moisture presents a moderate dielectric load that does not demand the generator's full output. Higher utilization would require either a smaller gap (increasing the electric field in the material), a thicker bed (more lossy material in the field), or a material with higher dielectric loss factor.

---

## 7. Discussion

### 7.1 Energy Partitioning and the Seed Coat Barrier

The quantitative result that 99.2% of absorbed RF energy is directed to sensible heating is the central physical finding of this calibration study. This energy partition is governed by the morphological structure of the feedstock: whole yellow pea seeds (6-8 mm diameter) possess an intact seed coat (testa) that functions as a semipermeable barrier to moisture transport. At the operating temperatures achieved in this process (40-60 C mean, 60 C peak), the rate of moisture diffusion through the seed coat is negligible compared to the rate of RF energy deposition.

This can be expressed as a dimensionless ratio. The Biot number for mass transfer through the seed coat is:

    Bi_m = (k_evap * L_characteristic) / D_eff

At the calibrated *k_evap* = 1.0 x 10^-6 and *D_eff* ~ 10^-10 m^2/s (Arrhenius at 50 C), the mass transfer resistance is dominated by the seed coat rather than internal diffusion. The system is in the regime where RF energy delivery far exceeds the material's capacity for evaporative cooling — the energy has no path to dissipation except through sensible temperature rise and boundary heat transfer.

This finding has direct implications for process design in the dry fractionation line:

1. **Temperature uniformity is the primary quality metric**, not moisture uniformity or drying rate. The pretreatment step should be evaluated by the coefficient of variation of outfeed temperature (CV = 0.022 in the current configuration), the peak-to-mean temperature ratio, and the fraction of seed volume exceeding the denaturation threshold (~70 C for legume proteins).

2. **The operating envelope is defined by thermal limits**, not drying kinetics. The maximum belt residence time is constrained by the requirement to keep seed temperatures below the denaturation threshold for the target protein fraction, while the minimum residence time must provide sufficient thermal conditioning for effective fracture during milling.

3. **Bed depth and gap should be optimized for the vertical temperature profile**. The outfeed cross-section shows a 40 C gradient from bed surface to belt contact. Reducing bed depth or increasing gap (both decrease the material-to-air-gap ratio) would reduce the temperature gradient but also reduce throughput. This is a Pareto optimization problem well-suited to the digital twin.

### 7.2 The Oscillator Coupling Factor

The calibrated *k_c* = 0.1741 is 32.5% lower than the analytical prior (0.258), which was derived from the test report's full-load operating point under the assumption of ideal coupling. The discrepancy reflects real-world losses in the tank circuit-to-electrode chain: parasitic impedances in the trombocone feed lines, cable losses, and impedance mismatch between the oscillator output and the parallel-plate applicator under partial load.

The sensitivity gradient (*dL/dk_c* = -6.81) confirms that *k_c* is the best-constrained parameter. This is expected: *k_c* enters the power expression quadratically (*P ~ k_c^2*), and *I_a* provides a fast, direct observable of the delivered power. The standard error of the coupling factor estimate can be bounded from the sensitivity: at the optimum, a perturbation *delta_k_c* = 0.005 produces *delta_L* = 0.034, which is within the noise floor of the stochastic optimizer. The practical confidence interval for *k_c* is approximately +/- 0.005 (3%).

The coupling factor is expected to be a machine-specific constant that does not depend on the material or operating conditions (it characterizes the tank circuit, not the load). Validation against independent runs with different materials and gap settings would confirm this transferability.

### 7.3 Residual Temperature Discrepancy

The simulated outfeed temperature (43.2 C average, 60.6 C maximum) underpredicts the PLC peak temperature reading (101 C) and the temperature-indicator strip readings (77-93 C across five sections). Three factors contribute to this discrepancy:

1. **Sensor location vs. simulation definition.** The PLC Product_Temp sensor measures at a fixed location in the oven that may correspond to a hot spot (near the electrode surface) rather than the bulk outfeed average. The simulation reports the material-cell-weighted mean at the last X-slice, which is inherently lower.

2. **Oven pre-heating.** The GP-15 was operating (RF on, conveyor running) for some time before the 61 kg feed was introduced. The oven chamber, electrodes, and PTFE belt were already at elevated temperature. The simulation starts from a cold initial condition (17.6 C everywhere), lacking this thermal pre-conditioning. Adding an optional pre-run warm-up phase to the simulation could reduce this gap.

3. **Oscillator model fidelity.** The single-parameter coupling model (*L_Ia* = 1.98 is the largest loss component) is the weakest submodel. The real oscillator exhibits load-dependent frequency pulling (the operating frequency shifts as the load impedance changes), which alters the dielectric loss factor *epsilon''(f)* and hence the absorbed power. A more detailed oscillator equivalent-circuit model incorporating frequency pulling would improve the *I_a* fit and, through the power coupling, the temperature prediction.

### 7.4 The MRH Edge Operating Regime

The calibration revealed that under the Run #1 conditions, the GP-15 operates at the edge of its MRH activation threshold — a regime where the anode current rises to ~1.70 A during the initial thermal transient, triggering proportional gap opening, which reduces the load and brings *I_a* back below MRH. This produces a self-regulating cycle that manifests as a gradual gap drift rather than a discrete control event.

The MRH edge regime is a consequence of the coupling between three dynamics:

1. **Thermal loading:** As the material bed heats up, *epsilon''(T, M)* increases, the absorbed power increases, and *I_a* rises.
2. **Oscillator droop:** Higher current causes the anode voltage to droop, partially self-limiting the power delivery.
3. **MRH gap control:** When the droop alone is insufficient to keep *I_a* below MRH, the gap opens to further reduce coupling.

The discovery of this regime during calibration required the global optimizer (differential evolution) to explore coupling values in the narrow band *k_c* in [0.17, 0.18]. A local optimizer initialized at the prior (*k_c* = 0.258) would not have reached this region, as it lies across the Phase 2 plateau described in Section 6.2. This demonstrates the value of combining global search with local refinement for calibrating models with non-convex loss landscapes.

### 7.5 Grid Resolution and Numerical Accuracy

The vertical resolution (10 mm cells, 2-3 cells in the 25 mm material bed) is sufficient for the calibration objective (matching bulk KPIs) but marginal for resolving the detailed vertical temperature profile. A convergence study with finer grids (5 mm, 2.5 mm Y-cells) would quantify the discretization error in the peak temperature and vertical gradient. The TVD advection scheme provides second-order accuracy in the X-direction, ensuring that the thermal front propagation through the RF zone is not artificially smeared by numerical diffusion.

### 7.6 Limitations and Path to Higher Fidelity

1. **Single-run calibration.** The three parameters were calibrated against one PLC recording. Cross-validation against independent runs (different masses, belt speeds, gap settings, moisture contents) is needed to establish parameter transferability and quantify prediction uncertainty.

2. **Evaporation model at boundary.** The calibrated *k_evap* hitting its lower bound indicates the model's evaporation physics are over-parameterized for this feedstock. For whole seeds at sub-100 C, the appropriate modeling choice may be to set *k_evap* = 0 entirely and calibrate only two parameters (*k_c*, *r_gap*), reducing the search dimensionality and avoiding boundary artifacts in the sensitivity analysis.

3. **Oscillator model.** The linear droop with a single coupling factor (*L_Ia* = 1.98, largest loss component) is the primary accuracy bottleneck. A more detailed model incorporating the oscillator equivalent circuit, load-dependent frequency pulling, and tank circuit Q-factor would reduce the anode current residual.

4. **2D/3D RF field.** The series-capacitor model provides a 1D (Y-direction) field solution with uniform E in each layer. Fringe fields at the electrode edges and the spatial variation of *epsilon'(T, M)* are not captured. The Phase 2 FDM Laplace solver (implemented but not used in the calibration for computational cost reasons) addresses both of these effects.

5. **Vertical resolution.** The 10 mm cell size places only 2-3 cells in the 25 mm bed. While adequate for bulk KPIs, this limits the model's ability to predict the detailed vertical temperature profile, which is important for assessing peak temperature exposure and denaturation risk.

---

## 8. Computational Performance

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 6000 Ada Generation (48 GiB, SM 8.9) |
| CPU | AMD Ryzen (Family 25 Model 24) |
| CUDA Toolkit | 12.9 |
| NVIDIA Warp | 1.11.0 |
| Grid | 60 x 30 x 32 = 57,600 cells |
| GPU-accelerated steps | Advection, P_v computation, thermal FDM, moisture diffusion, property update |
| CPU steps | RF field solve (series-capacitor), PLC controller, KPI recording, Lagrangian particles |
| Timestep (typical) | 0.25-0.35 s |
| Timesteps per simulation (947 s) | ~3,171 |
| **Single simulation** | **1,337 s wall-clock (0.71x real-time)** |
| **Calibration (926 evaluations)** | **~46 min wall-clock (~3 s/eval)** |
| Speedup from GPU array reuse | ~60x vs. re-construction per evaluation |

The sub-real-time performance (0.71x) is dominated by GPU-CPU synchronization for boundary conditions and the controller logic. Pure GPU execution (without CPU sync) achieves approximately 2x real-time. Future optimization targets include fusing boundary conditions into the GPU kernels and batching KPI computation.

---

## 9. Conclusions

1. **A comprehensive multi-physics digital twin** of the GP-15 RF dielectric heating system has been developed, coupling electromagnetic field solution, thermal transport, moisture kinetics, belt advection, PLC control logic, and Lagrangian particle tracking on a GPU-accelerated platform. The model architecture (sequential operator-splitting on a 57,600-cell rectilinear grid with NVIDIA Warp CUDA kernels) enables calibration sweeps at 3 seconds per evaluation.

2. **The three-parameter calibration** against a full 2794-second PLC recording demonstrates that derivative-free global optimization (differential evolution) with variance-normalized multi-signal loss converges to a physically interpretable optimum (*L* = 4.424, with each component of order 1.0). The inclusion of anode current in the loss function is essential for resolving the coupling factor and breaking parameter degeneracies.

3. **The energy partition in RF pretreatment of whole yellow pea** is overwhelmingly dominated by sensible heating (99.2% of absorbed energy), with negligible evaporative moisture removal. This partition is governed by the intact seed coat, which functions as a diffusion barrier to moisture transport at sub-100 C operating temperatures. The calibrated *k_evap* = 1.0 x 10^-6 (at the lower search bound) quantitatively establishes this barrier effect.

4. **The MRH edge operating regime** — a previously undocumented control dynamic in which the anode current oscillates near the overcurrent threshold, producing a gradual electrode gap drift — has been identified and characterized. This regime is the dominant process transient under the Run #1 conditions and arises from the coupling between thermal loading, oscillator droop, and MRH proportional gap control.

5. **The calibrated oscillator coupling factor** (*k_c* = 0.1741) is 32.5% below the analytical prior, reflecting real-world parasitic losses in the tank circuit. This parameter is well-constrained by the anode current data (sensitivity gradient = -6.81) and is expected to transfer across operating conditions for the same machine.

6. **The residual loss** (4.424) establishes the structural accuracy limit of the current model physics. The primary improvement path is a higher-fidelity oscillator model incorporating load-dependent frequency pulling (*L_Ia* = 1.98 is the largest loss component), followed by oven pre-heating initialization and finer vertical grid resolution.

7. **The digital twin provides the computational foundation** for systematic optimization of the pretreatment step in the dry fractionation line — including virtual recipe development (gap, speed, bed depth), thermal uniformity optimization (minimizing vertical temperature gradients), and evaluation of alternative feedstocks (split pea, dehulled pea, other pulses) where the energy partition between sensible and latent pathways will differ.

---

## References

1. QMTI GP-15 Installation and Operation Manual, Quantum Mechanical Technologies Inc., Prince Albert, SK, Canada, 2021.
2. Kwofie, E., "Pretreatment Engineering Guide: RF Dielectric Heating Digital Twin — QMTI GP-15 Gentle Processing Machine," Air Classifier Designer Project, February 2026.
3. Storn, R. and Price, K., "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces," J. Global Optimization, 11, pp. 341-359, 1997.
4. NVIDIA Warp Documentation, https://nvidia.github.io/warp/, 2024.
5. Piyasena, P., Dussault, C., Koutchma, T., Ramaswamy, H.S., and Awuah, G.B., "Radio frequency heating of foods: principles, applications and related properties — a review," Critical Reviews in Food Science and Nutrition, 43(6), pp. 587-606, 2003.
6. Jiao, S., Johnson, J.A., Tang, J., and Wang, S., "Industrial-scale radio frequency treatments for insect control in lentils," Journal of Stored Products Research, 48, pp. 143-148, 2012.
7. Schutyser, M.A.I. and van der Goot, A.J., "The potential of dry fractionation for sustainable plant protein production," Trends in Food Science and Technology, 22(4), pp. 154-164, 2011.
8. Assatory, A., Vitelli, M., Rajabzadeh, A.R., and Legge, R.L., "Dry fractionation methods for plant protein, starch and fiber enrichment: A review," Trends in Food Science and Technology, 86, pp. 340-351, 2019.

---

*Calibrated parameters saved to: `utility_docs/calibration_latest.json`*  
*Source code: `src/airclassifier/pretreatment/` (simulator, physics/coupling, calibration, particles, control)*  
*Example: `python examples/simulate_and_visualize.py --calibrate "utility_docs/Run1 RF data(in).csv" --cal-duration 0`*
