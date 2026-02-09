# `pretreatment` — Engineering Guide

**RF Dielectric Heating Digital Twin**
*QMTI GP-15 Gentle Processing Machine — NVIDIA Warp Implementation*

Air Classifier Designer Project · Pretreatment Module
Emmanuel Kwofie · February 2026

---

## 1. Purpose and Scope

This document is the build specification for the `pretreatment` module — a physics-based digital twin of the QMTI GP-15 Radio Frequency dielectric heating machine. It gives you, the engineer, everything required to implement the simulation from scratch: the governing equations, the machine geometry, the material property models, the control logic, every Warp kernel signature, and the data flow into your downstream milling and classification stages.

The GP-15 sits at the head of the dry fractionation line. Raw whole flour (yellow pea, faba bean, oat) enters with moisture content in the range of 8–14% wet basis. The RF field heats the material volumetrically — water molecules absorb dielectric energy preferentially — reducing moisture to 2–4% in a single pass. The conditioned material then feeds the pin mill and air classifier. Moisture content and its spatial uniformity at the GP-15 outlet directly control particle size distribution after milling and protein separation efficiency in the classifier. This module must therefore produce accurate, spatially-resolved temperature and moisture fields at the outfeed, not merely bulk averages.

The simulation engine is NVIDIA Warp. All heavy computation — field solves, heat transfer, moisture diffusion, material advection — runs as JIT-compiled GPU kernels. The module supports differentiable simulation via `wp.Tape` for gradient-based recipe optimization and integrates with the existing Air Classifier Designer GUI (PySide6 + PyVista) for real-time 3D visualization.

---

## 2. GP-15 Machine Anatomy

This section documents every subsystem of the physical machine that the simulation must represent. Values are cross-referenced against the GP-15 Installation and Operation Manual and verified against machine photographs. Where a value could not be verified, it is marked **[TBD — MEASURE]**.

### 2.1 Generator and Oscillator

The RF energy source is a self-excited triode valve oscillator operating in a tuned LC tank circuit.

| Parameter | Value | Source |
|---|---|---|
| Operating frequency | 27.12 MHz (ISM band) | Manual Appendix B |
| Maximum RF output power | 15 kW | Manual Appendix B |
| Anode voltage (no-load) | 9.18 kV DC, Ia = 0.4 A | Manual Appendix B, Test Report |
| Anode voltage (full-load) | 8.38 kV DC, Ia = 2.58 A | Manual Appendix B, Test Report |
| Nominal anode voltage (nameplate) | 9.5 kV DC | Manual Appendix B |
| Electrical supply | 600 V, 3-phase, 60 Hz, 42 kVA max | Manual Appendix B |
| Grid resistors | 6 × 200 Ω in series (1200 Ω total) | Manual parts list |
| Valve thermal fuse | Operates at ~145°C | Manual Appendix B |
| Air cooling (valve/tank circuit) | 1119 CFM | Manual Appendix B |

**Critical modeling note — voltage droop under load:** The anode voltage is not constant. It drops from 9.18 kV at idle to 8.38 kV at full rated power (2.58 A anode current). The simulation must model this load-dependent voltage to correctly predict anode current from delivered RF power. A linear interpolation between the no-load and full-load operating points from the test report is the minimum-viable model:

```
V_anode(Ia) = 9.18 - (9.18 - 8.38) / (2.58 - 0.4) × (Ia - 0.4)   [kV]
            = 9.18 - 0.367 × (Ia - 0.4)                              [kV]
```

The generator is not spatially simulated. It is modeled as a parameterized power source: total RF power (kW) is an input, and the field distribution within the applicator is computed to deliver that total power dissipation.

### 2.2 Oven / Applicator (Primary Simulation Volume)

The oven is the rectangular chamber where RF processing occurs. It contains the electrode system, the conveyor belt, and the material bed. This is the primary simulation domain.

#### 2.2.1 Upper Electrode Assembly

The upper electrode is **not** a single monolithic plate. It consists of the following components (from the manual parts list and machine photographs):

| Component | Part Number | Quantity | Description |
|---|---|---|---|
| Perforated electrode plates | 11.05.433 | 2 | Two separate plates with circular perforation arrays, side by side along the conveyor direction |
| Electrode frame beams | 11.05.432 | 6 | Longitudinal beams forming the electrode support frame |
| Silicon supports | 10.00.091 | 8 | Electrical insulation between electrode plates and frame |
| Feed strips | 10.07.874 | 6 × 18 inches | Copper strips connecting the oscillator tank circuit to the electrode plates |
| Lead screw nuts | 10.07.201 | 4 | Drive mechanism — electrode frame moves vertically on 4 lead screws |
| Tuning bridges | — | 8 | Part of the impedance matching network above the electrode |
| Trombocones | — | 12 | Tunable coupling elements between oscillator and electrode |
| Tuning feed plates | — | 4 | RF distribution plates above the electrode |

**Implications for the field model:**

The two-plate layout creates a center seam where field uniformity may be reduced. The circular perforations create localized field concentrations at each hole edge (field enhancement around conductor edges). The tuning/coupling structure (bridges, trombocones, feed plates) means the RF field distribution is not purely determined by the electrode gap — it has spatial variations imposed by the feed strip locations and coupling geometry.

For Phase 1, a uniform parallel-plate capacitor model is acceptable. For Phase 3+, the simulation should add correction factors: a perforation correction (estimated 3–8% field non-uniformity depending on hole size/pitch), a seam correction at the center joint, and a feed-strip proximity correction near the 6 copper strip attachment points.

#### 2.2.2 Lower Electrode Assembly

The lower electrode sits beneath the conveyor belt. The material rides the belt, which rides the lower electrode.

| Component | Part Number | Quantity | Description |
|---|---|---|---|
| Bottom electrode trays | 21-0042-0180 | 2 | Removable trays with U-handles, side by side |
| PET supports | various | multiple | Polyester insulating supports for the trays |
| Teflon wear strips | 41-0005-0692 | 24 | PTFE strips between belt and electrode surface |
| Top sheets | 41-0135-0005 | 2 | Protective sheets over the electrode trays |
| Copper chokes | 41-0296-0005 | 4 | 18-turn copper chokes for RF impedance matching |

**Layered stack from bottom to top within the simulation domain:**

```
  Upper electrode (perforated plates)      ← Boundary: V = V_upper
         |
     Air gap (variable, electrode gap minus bed depth)
         |
     Material bed (granular, porous)       ← Simulation volume
         |
     Conveyor belt (PTFE, ~2mm)            ← Dielectric layer: ε' ≈ 2.1, ε'' ≈ 0.0003
         |
     Teflon wear strips (~1mm total)       ← Dielectric layer: ε' ≈ 2.1, ε'' ≈ 0.0003
         |
     Top sheet (~0.5mm)                    ← Dielectric layer
         |
  Lower electrode (solid trays)            ← Boundary: V = 0 (ground)
```

These intermediate dielectric layers (belt + wear strips + top sheet ≈ 3.5 mm total) reduce the effective electric field in the material. The simulation must account for this stack in the capacitor model by computing the voltage division across the series of dielectric layers.

#### 2.2.3 Oven Dimensions

| Parameter | Value | Status |
|---|---|---|
| Active RF zone length | **[TBD — MEASURE]** from drawing #59-0001-0005_REV1 | **CRITICAL** — placeholder 1.5 m used until verified |
| Belt width (usable) | 800 mm | Confirmed — manual |
| Machine envelope (L × W × H) | 5.5 × 2.9 × 2.2 m | Confirmed — manual |
| Machine weight | ~2550 kg total (~1500 kg per zone) | Manual Appendix B |
| Electrode gap range | **[TBD — MEASURE]** from commissioned Test Report | Placeholder: 20–300 mm |
| Attenuation ducts | Infeed + outfeed tunnels | Prevent RF leakage; define entry/exit BCs |

**The oven length and electrode gap range are the two most critical geometric parameters for the simulation.** Residence time is directly proportional to oven length. Power density is inversely proportional to gap squared. Both must be measured from the actual machine or its engineering drawings before the simulation can be validated.

### 2.3 Conveyor System

| Parameter | Value | Source |
|---|---|---|
| Belt type | Modular link, plain reinforced PTFE (Teflon) | Manual |
| Belt width | 800 mm | Manual |
| Belt speed range | 0.1 – 2.0 m/min (typical) | Manual |
| Belt length (total loop) | ~16 m | Parts list (QTY of belt sections) |
| Belt material dielectric | ε' ≈ 2.1, ε'' ≈ 0.0003 at 27 MHz | PTFE literature values |
| Belt thickness | ~2 mm (estimated) | **[TBD — MEASURE]** |

The belt material matters. PTFE has very low dielectric loss (nearly transparent to RF), but it introduces a dielectric layer between the material and the lower electrode that must be included in the field calculation. See Section 4.1.3 for the series-capacitor voltage division.

### 2.4 Environment Management Unit (EMU)

The EMU manages the air environment inside the oven to remove evaporated moisture and prevent condensation.

| Component | Specification | Source |
|---|---|---|
| Heater arrays | 2 independent banks, each 6 × 1 kW = 12 kW total | Manual |
| Heater fans | 2, each with VFD speed control | Manual |
| Extraction fan capacity | 31.1 m³/min (1100 ft³/min) maximum | Manual |
| Extraction fan speed range | 5 – 60 Hz (VFD controlled) | Manual |
| Minimum extraction duct diameter | 250 mm | Manual |
| Maximum extraction backpressure | 250 Pa (1" WG) | Manual |
| Air inlet filters | Fitted to heater bank inlets | Manual |

The heater arrays serve two purposes: (1) maintaining warm ambient air to prevent condensation of evaporated moisture on the oven walls and product surface, and (2) providing supplemental convective heating. The heaters operate independently of the RF system — they can be on when RF is off. The simulation should model the heater power as an additional convective boundary condition at the material surface, separate from the RF volumetric source.

### 2.5 Infeed System

| Component | Part Number | Description |
|---|---|---|
| Infeed hopper | 2-813-05 | Gravity-fed hopper above belt entry |
| Sizing plate | 21-0042-0185 | Controls material bed depth on belt |
| Sizing plate (teeth type) | 21-0042-0186 | Alternative plate for specific products |

The sizing plate is critical: it determines the bed depth entering the oven, which directly controls the volume of material exposed to the RF field per unit time. Bed depth is a derived parameter — it depends on hopper settings and product flow characteristics — not a pure constant.

### 2.6 Temperature Sensing

| Component | Specification | Source |
|---|---|---|
| Optical temperature sensors | 6, mounted at outfeed | Manual Screen 14 |
| Sensor selection | Individual sensors can be enabled/disabled | Manual |
| Temperature calculation | Average of active sensors | Manual |

These sensors enable the automatic temperature control mode described in Section 8.

### 2.7 Control Console (HMI / PLC)

| Parameter | Value | Source |
|---|---|---|
| Recipe capacity | 30 stored recipes | Manual |
| Control modes | Manual, Recipe, Automatic Temperature | Manual |
| Anode current monitoring | Continuous, with MRH and MRL thresholds | Manual |
| Permitted recycle restarts | 4 (typical) | Manual |
| Restart delay | 2 seconds (typical) | Manual |
| Electrode debounce | 0.5 s after button release | Manual |
| Ambient temperature limit | 40–43°C maximum | Manual |
| Sound level | < 75 dB when fully installed | Manual |

---

## 3. Coordinate System and Simulation Domain

The simulation uses the same **Y-up** coordinate system as the Air Classifier Designer project.

```
        Y (up — vertical, electrode gap direction)
        |
        |    Z (depth — across belt width, 800 mm)
        |   /
        |  /
        | /
        +------------- X (length — conveyor direction, infeed → outfeed)
```

| Axis | Physical Direction | Range |
|---|---|---|
| X | Conveyor travel direction (infeed at x=0, outfeed at x=L_oven) | 0 to L_oven (active RF zone) |
| Y | Vertical — lower electrode at y=0, upper electrode at y=gap | 0 to electrode_gap |
| Z | Across belt width | 0 to 0.8 m |

### 3.1 Domain Decomposition

The simulation volume between the electrodes contains three distinct material zones stacked vertically:

```
y = gap     ┌─────────────────────────────────┐  Upper electrode (V = V_rf)
            │         Air gap                  │  ε' ≈ 1.0, ε'' ≈ 0
            │                                  │
y = d_bed   ├─────────────────────────────────┤
            │      Material bed (porous)       │  ε'(M,T), ε''(M,T)
            │      (the process zone)          │
y = d_belt  ├─────────────────────────────────┤
            │   Belt + wear strips + top sheet │  ε' ≈ 2.1, ε'' ≈ 0.0003
y = 0       └─────────────────────────────────┘  Lower electrode (V = 0, ground)
```

Where `d_belt` ≈ 3.5 mm (combined belt + wear strip + top sheet thickness) and `d_bed` = `d_belt` + bed depth (typically 20–80 mm depending on product and sizing plate).

### 3.2 Grid Specification

The domain is discretized as a structured hexahedral grid, compatible with `warp.fem.Grid3D` or `warp.fem.Hexmesh`.

| Parameter | Typical Value | Notes |
|---|---|---|
| nx (X, conveyor direction) | 60–120 | Must resolve residence time gradients |
| ny (Y, vertical / gap direction) | 15–30 | Must resolve bed depth and air gap separately |
| nz (Z, across belt width) | 20–40 | Must capture edge effects near belt edges |
| dx | L_oven / nx | Uniform spacing in X |
| dy | gap / ny | Non-uniform recommended: finer in bed, coarser in air gap |
| dz | 0.8 / nz | Uniform spacing in Z |
| Total cells | 18,000 – 144,000 | Moderate GPU memory requirement |

For cells that lie entirely within the belt/wear-strip layer, the simulation assigns PTFE dielectric properties and zero thermal source (RF passes through but deposits negligible energy). For cells within the material bed, full coupled physics applies. For cells in the air gap, a simplified convective heat transfer model is used.

---

## 4. Physics Models

The pretreatment simulation couples four physics domains. Each domain is modeled independently and coupled at each timestep through shared state variables (temperature T, moisture content M, electric field E).

### 4.1 RF Dielectric Heating (Volumetric Heat Source)

#### 4.1.1 Governing Equation

The volumetric power density absorbed by the material at each point is:

```
P_v(x,y,z) = 2π · f · ε₀ · ε''_eff(T, M) · |E(x,y,z)|²    [W/m³]
```

where:
- `f` = 27.12 × 10⁶ Hz (operating frequency)
- `ε₀` = 8.854 × 10⁻¹² F/m (permittivity of free space)
- `ε''_eff(T, M)` = effective dielectric loss factor of the material, dependent on temperature and moisture content
- `|E(x,y,z)|` = magnitude of the electric field at position (x,y,z)

This equation is the foundation of the entire simulation. The RF field selectively heats water molecules (which dominate the loss factor), making the process inherently self-leveling: wetter regions absorb more energy, heat faster, dry faster, and converge toward uniform moisture. This self-leveling behavior is the GP-15's primary advantage and must be faithfully reproduced.

#### 4.1.2 Quasi-Static Field Approximation

At 27.12 MHz, the free-space wavelength is:

```
λ = c / f = 3 × 10⁸ / 27.12 × 10⁶ ≈ 11.06 m
```

The electrode dimensions (~0.8 m wide, ~1.5 m long) are much smaller than λ, so the electromagnetic problem reduces to a quasi-static electrostatic problem. The electric field satisfies:

```
∇ · (ε(x,y,z) · ∇φ) = 0    (no free charges)
```

with boundary conditions:
- `φ = V_rf` on the upper electrode surface
- `φ = 0` on the lower electrode surface (ground)
- `∂φ/∂n = 0` on the lateral boundaries (fringe field approximation)

The electric field is then `E = -∇φ`, and `|E|² = |∇φ|²`.

#### 4.1.3 Series-Capacitor Model (Layered Dielectric Stack)

For the uniform-field approximation (Phase 1), the multi-layer stack between the electrodes acts as capacitors in series. The voltage across each layer is proportional to its thickness divided by its permittivity:

```
V_total = V_air + V_bed + V_belt

V_layer_i = V_total × (d_i / ε'_i) / Σ(d_j / ε'_j)

E_layer_i = V_layer_i / d_i = V_total / (ε'_i × Σ(d_j / ε'_j))
```

For the three-layer stack:

| Layer | Thickness d | Relative permittivity ε' | Loss factor ε'' |
|---|---|---|---|
| Air gap | gap - d_bed - d_belt | 1.0 | ~0 |
| Material bed | d_bed (typically 20-80 mm) | ε'_bed(M, T) — typically 2–15 | ε''_bed(M, T) — typically 0.05–2.0 |
| Belt + wear strips | d_belt ≈ 3.5 mm | 2.1 (PTFE) | 0.0003 |

The field in the material bed is:

```
E_bed = V_total / (ε'_bed × (d_air/1.0 + d_bed/ε'_bed + d_belt/2.1))
```

Since ε'' for PTFE and air are negligible, only the material bed absorbs significant RF energy. The total RF power delivered to the material is:

```
P_rf_total = ∫∫∫_bed P_v dV = 2π·f·ε₀ × ∫∫∫_bed ε''_eff · |E_bed|² dV
```

This integral must equal the generator's delivered power (constrained by the anode current and voltage). In the simulation, one of two approaches is used:

**Approach A — Power-constrained (recommended for initial implementation):**
Given total RF power P_total (from recipe or anode current model), solve for V_rf such that the volume integral of P_v equals P_total. This requires a single scalar iteration per timestep.

**Approach B — Voltage-driven:**
Given V_anode from the generator model, compute V_rf from the oscillator coupling efficiency, then compute P_v directly. More physically accurate but requires modeling the oscillator-electrode coupling.

#### 4.1.4 Fringe Field Correction

At the electrode edges, the electric field is not uniform — it bulges outward (fringe field). For the GP-15's aspect ratio (width/gap > 2 in most operating conditions), the fringe field correction is modest. Two approaches:

**Analytic correction (Phase 1):** Apply a multiplicative correction factor to the field strength near the edges. The Palmer formula for parallel-plate fringe capacitance gives:

```
C_fringe / C_uniform ≈ 1 + (d/π·w) × [1 + ln(2π·w/d)]
```

where d = gap and w = electrode width. This provides a first-order estimate of the edge effect.

**2D Laplace pre-solve (Phase 3):** Solve the 2D Laplace equation in the Y-Z plane (across belt width × gap) once for each electrode gap setting. Store the resulting field correction map. Apply it as a multiplier to the 1D uniform field model. This captures the actual fringe profile including the effect of the grounded attenuation duct walls.

#### 4.1.5 Perforation Correction (Phase 3+)

The upper electrode plates have circular perforations. Each hole acts as a field singularity: the field concentrates at the hole edges and drops inside the hole. The effective field correction can be modeled as:

```
E_eff(x,z) = E_uniform × η_perf(x,z)
```

where `η_perf` is a correction map computed from the perforation geometry (hole diameter, pitch, pattern). For a regular array of circular holes with diameter d_hole and pitch p:

```
η_perf ≈ 1 / (1 - π·d_hole²/(4·p²))    (area-average)
```

with local enhancement at hole edges of approximately `η_edge ≈ η_perf × (1 + d_hole/(2·gap))`.

### 4.2 Heat Transfer

#### 4.2.1 Governing Equation

Temperature evolution in the material bed is governed by the heat equation with a volumetric RF source and evaporative sink:

```
ρ(M) · c_p(T, M) · ∂T/∂t = ∇·(k(T, M) · ∇T) + P_v(x,y,z) - L_v · ṁ_evap(T, M)
```

where:
- `ρ(M)` = bulk density of the material bed [kg/m³], moisture-dependent
- `c_p(T, M)` = specific heat capacity [J/(kg·K)], depends on both T and M
- `k(T, M)` = effective thermal conductivity [W/(m·K)], depends on both T and M
- `P_v` = RF volumetric power density [W/m³] (from Section 4.1)
- `L_v` = latent heat of vaporization of water ≈ 2.26 × 10⁶ J/kg (temperature-dependent)
- `ṁ_evap` = local evaporation rate [kg/(m³·s)] (from Section 4.3)

#### 4.2.2 Boundary Conditions

| Surface | Condition | Physical Meaning |
|---|---|---|
| Top of bed (y = d_bed) | Convective: `-k ∂T/∂y = h_conv · (T_surface - T_air)` | EMU airflow carries heat away from surface |
| Bottom of bed (y = d_belt) | Conductive through belt to lower electrode (isothermal sink) | Belt contact — approximate as specified contact conductance |
| Infeed (x = 0) | Dirichlet: `T = T_inlet` (ambient, ~20–25°C) | Fresh material entering at ambient temperature |
| Outfeed (x = L_oven) | Neumann: `∂T/∂x = 0` | Zero-gradient outflow (advection-dominated) |
| Belt edges (z = 0, z = 0.8) | Adiabatic: `∂T/∂z = 0` | Insulated edges (conservative approximation) |

The convective heat transfer coefficient `h_conv` depends on the extraction fan airflow velocity over the material surface. For forced convection over a flat bed:

```
h_conv = Nu · k_air / L_char
```

where `Nu` is the Nusselt number from a flat-plate forced convection correlation (laminar or turbulent depending on the Reynolds number), `k_air` is the thermal conductivity of air, and `L_char` is a characteristic length (bed length in the flow direction).

The EMU heater arrays add up to 12 kW of thermal power to the air stream. When heaters are active, the air temperature `T_air` is elevated above ambient. The simulation should model `T_air` as:

```
T_air = T_ambient + Q_heater / (ṁ_air · c_p_air)
```

where `ṁ_air` is the mass flow rate from the extraction fan (derived from the 31.1 m³/min capacity scaled by VFD frequency) and `Q_heater` is the total heater power (0 to 12 kW).

#### 4.2.3 Numerical Method

**Explicit finite difference (Phase 1):** Second-order central differences for the Laplacian, forward Euler time integration. Stable under the CFL condition:

```
dt < dx² / (2·α·d)
```

where `α = k/(ρ·c_p)` is the thermal diffusivity and `d` is the spatial dimension (2 in 2D, 3 in 3D). For typical material properties (α ≈ 1.5 × 10⁻⁷ m²/s) and dx = 15 mm, the stability limit is dt < ~0.75 s, which is comfortable for the physical timescale (residence time ~45–900 s).

**Implicit finite difference (Phase 3, if needed):** Backward Euler with the heat equation discretized as a sparse linear system, solved using `warp.sparse.cg()`. Unconditionally stable, allows larger timesteps for steady-state seeking.

### 4.3 Moisture Transport and Drying Kinetics

This is the core process model. The GP-15 exists to remove moisture, and the simulation must capture both the rate and spatial uniformity of drying.

#### 4.3.1 Governing Equation

```
∂M/∂t = ∇·(D_eff(T) · ∇M) - ṁ_evap(T, M) / ρ_dry
```

where:
- `M` = local moisture content [kg water / kg wet material] (wet basis)
- `D_eff(T)` = effective moisture diffusivity [m²/s], strongly temperature-dependent
- `ṁ_evap` = volumetric evaporation rate [kg/(m³·s)]
- `ρ_dry` = dry-basis bulk density [kg/m³]

#### 4.3.2 Evaporation Rate Model

The evaporation rate couples the thermal and moisture fields. It is the mechanism through which the latent heat sink appears in the heat equation:

**At the surface (top of bed):**

```
ṁ_evap_surface = h_m · ρ_air · (w_sat(T_surface) - w_air)    [kg/(m²·s)]
```

where:
- `h_m` = mass transfer coefficient [m/s] (from heat-mass transfer analogy: `h_m = h_conv / (ρ_air · c_p_air · Le^(2/3))`, where Le is the Lewis number ≈ 0.85 for water in air)
- `w_sat(T)` = saturation humidity ratio at surface temperature [kg water / kg dry air]
- `w_air` = humidity ratio of the oven air

**Within the bed (internal evaporation from RF heating):**

When the material temperature exceeds a threshold near the boiling point at the local partial pressure, internal evaporation occurs. For temperatures below 100°C (the normal operating range of the GP-15), the internal evaporation rate is modeled as:

```
ṁ_evap_internal = ρ_dry · k_evap · M · max(0, T - T_threshold)    [kg/(m³·s)]
```

where `k_evap` is an empirical rate constant [1/(°C·s)] and `T_threshold` is the temperature above which active evaporation occurs (~40°C for thin-layer drying of legume flours). This simplified model captures the essential physics: evaporation rate increases with temperature and with available moisture.

**Calibration target:** The GP-15 manual specifies water removal efficiency of:
- 1.0 kg water per kWh of RF energy for high surface-to-volume materials
- 0.6 kg water per kWh for low surface-to-volume feedstock

These values constrain `k_evap` during model calibration.

#### 4.3.3 Moisture Diffusivity

The effective diffusivity of moisture within the porous material bed follows an Arrhenius-type temperature dependence:

```
D_eff(T) = D_0 · exp(-E_a / (R · T))
```

where:
- `D_0` = pre-exponential factor [m²/s] (material-dependent, typically 10⁻⁴ to 10⁻² for food powders)
- `E_a` = activation energy [J/mol] (typically 20–50 kJ/mol for legume flours)
- `R` = 8.314 J/(mol·K) (universal gas constant)
- `T` = absolute temperature [K]

Literature values for yellow pea flour: `D_0 ≈ 5.7 × 10⁻⁴ m²/s`, `E_a ≈ 28.5 kJ/mol` (these values should be replaced with measured data when available).

#### 4.3.4 Self-Leveling Mechanism

The self-leveling of RF drying is emergent from the coupling between the dielectric loss factor and moisture content. The causal chain is:

```
Higher moisture M  →  Higher ε''(M)  →  Higher P_v = 2πfε₀ε''|E|²
                   →  Higher ∂T/∂t   →  Higher ṁ_evap(T,M)
                   →  Faster moisture decrease  →  M converges to neighbors
```

This mechanism must be reproduced without any artificial smoothing. It arises naturally from the physics coupling and serves as a key validation target: starting from a spatially non-uniform initial moisture field, the simulation should show convergence toward uniformity over the residence time.

### 4.4 Material Transport (Conveyor Kinematics)

Material enters at the infeed (x = 0), rides the belt at velocity `v_belt`, and exits at the outfeed (x = L_oven).

#### 4.4.1 Eulerian Advection

In the Eulerian frame (fixed grid), the material fields (T, M, and all derived properties) are advected along the positive X-axis at the belt velocity:

```
∂φ/∂t + v_belt · ∂φ/∂x = S(φ)
```

where φ is any transported scalar (T, M) and S(φ) is the source/sink term from the physics models.

The advection is implemented as a first-order upwind scheme (Phase 1) or a second-order TVD scheme with a flux limiter (Phase 2):

**Upwind (Phase 1):**
```
φ_new[i] = φ[i] - v_belt · dt/dx · (φ[i] - φ[i-1])
```

**Van Leer TVD (Phase 2):**
```
r = (φ[i] - φ[i-1]) / (φ[i+1] - φ[i])
ψ(r) = (r + |r|) / (1 + |r|)              # Van Leer limiter
flux = v_belt · (φ[i] + 0.5·ψ(r)·(φ[i+1] - φ[i]))
```

#### 4.4.2 Boundary Conditions for Transport

| Boundary | Condition | Implementation |
|---|---|---|
| Infeed (x = 0) | `φ = φ_inlet` (fresh material) | Inject new material at ambient T and initial moisture M₀ |
| Outfeed (x = L_oven) | Zero-gradient (advection outflow) | `φ[nx] = φ[nx-1]` — material exits freely |

#### 4.4.3 Residence Time

The residence time is the total time material spends in the active RF zone:

```
t_res = L_oven / v_belt
```

| Belt Speed | Residence Time (L=1.5m) | Typical Application |
|---|---|---|
| 0.1 m/min | 900 s (15 min) | High moisture reduction, thick bed |
| 0.5 m/min | 180 s (3 min) | Moderate drying |
| 1.0 m/min | 90 s (1.5 min) | Light conditioning |
| 2.0 m/min | 45 s | Minimal treatment, thin bed |

The residence time × power density product determines total energy delivered per unit volume of material. This is the primary design variable for the process.

---

## 5. Material Properties

All material properties are functions of temperature and/or moisture content. These functional dependencies are critical for simulation accuracy — they drive the nonlinear coupling between the physics domains.

### 5.1 Dielectric Properties

The dielectric loss factor `ε''` is the single most important material property for the simulation. It determines how much RF energy is absorbed.

#### 5.1.1 Loss Factor Model

For legume flours at 27 MHz, the loss factor is modeled as:

```
ε''(T, M) = a₁·M² + a₂·M + a₃·M·T + a₄·T + a₅
```

where M is moisture content (wet basis, fraction) and T is temperature (°C). Typical coefficients for yellow pea flour (from published dielectric characterization studies of legume powders):

| Coefficient | Value | Units |
|---|---|---|
| a₁ | 85.0 | dimensionless |
| a₂ | 2.5 | dimensionless |
| a₃ | 0.12 | 1/°C |
| a₄ | 0.008 | 1/°C |
| a₅ | 0.02 | dimensionless |

At typical operating conditions: M = 0.10, T = 60°C → ε'' ≈ 1.9. At M = 0.03, T = 60°C → ε'' ≈ 0.22. This 8.6× ratio between wet and dry material is the physical basis for the self-leveling effect.

The dielectric constant (real part) `ε'` is modeled similarly:

```
ε'(T, M) = b₁·M + b₂·T + b₃
```

Typical values: b₁ = 25.0, b₂ = -0.05, b₃ = 2.5. At M = 0.10 → ε' ≈ 5.0.

**These coefficients are placeholders.** They must be replaced with values measured for the actual feedstock using a network analyzer at 27 MHz, or fitted from published data for the specific legume/oilseed species being processed.

#### 5.1.2 Loss Tangent

The loss tangent provides a useful diagnostic:

```
tan δ = ε'' / ε'
```

For efficient RF heating, `tan δ > 0.01` is needed. For protein-rich legume flours at 8–12% moisture, `tan δ` is typically 0.05–0.3, well within the efficient range.

### 5.2 Thermal Properties

| Property | Model | Typical Range | Dependence |
|---|---|---|---|
| Specific heat c_p | `c_p = c_p_dry·(1-M) + c_p_water·M` | 1200–2800 J/(kg·K) | Linear mixing of dry solid and water |
| Thermal conductivity k | `k = k_dry·(1 + β·M)·(1 - φ_bed)` + `k_air·φ_bed` | 0.1–0.5 W/(m·K) | Moisture, porosity |
| Bulk density ρ | `ρ = ρ_solid·(1-φ_bed)·(1+M/(1-M))` | 400–800 kg/m³ | Porosity, moisture |

Constituent property values:

| Property | Value | Source |
|---|---|---|
| c_p_dry (legume flour) | 1300–1500 J/(kg·K) | Literature |
| c_p_water | 4186 J/(kg·K) | Standard |
| k_dry (legume flour) | 0.15–0.25 W/(m·K) | Literature |
| β (moisture sensitivity of k) | 3.0–5.0 | Empirical |
| ρ_solid (legume flour particle) | 1350–1500 kg/m³ | Measured |
| φ_bed (bed porosity) | 0.35–0.45 | Depends on packing |
| L_v (latent heat at 60°C) | 2.36 × 10⁶ J/kg | Standard thermodynamic tables |

### 5.3 Material Presets

Matching the Air Classifier Designer's material system, the pretreatment module provides presets for the three target feedstocks:

```python
MATERIAL_PRESETS = {
    "yellow_pea": MaterialProperties(
        name="Yellow Pea Flour (Pisum sativum)",
        dielectric_loss_coeffs=(85.0, 2.5, 0.12, 0.008, 0.02),
        dielectric_const_coeffs=(25.0, -0.05, 2.5),
        c_p_dry=1380.0,       # J/(kg·K)
        k_dry=0.18,           # W/(m·K)
        k_moisture_beta=4.0,
        rho_solid=1450.0,     # kg/m³
        D_eff_D0=5.7e-4,      # m²/s
        D_eff_Ea=28500.0,     # J/mol
        bed_porosity=0.40,
    ),
    "faba_bean": MaterialProperties(
        name="Faba Bean Flour (Vicia faba)",
        dielectric_loss_coeffs=(90.0, 2.8, 0.13, 0.009, 0.025),
        dielectric_const_coeffs=(27.0, -0.04, 2.6),
        c_p_dry=1420.0,
        k_dry=0.20,
        k_moisture_beta=3.8,
        rho_solid=1480.0,
        D_eff_D0=4.9e-4,
        D_eff_Ea=30200.0,
        bed_porosity=0.38,
    ),
    "oat": MaterialProperties(
        name="Oat Flour (Avena sativa)",
        dielectric_loss_coeffs=(78.0, 2.2, 0.11, 0.007, 0.03),
        dielectric_const_coeffs=(22.0, -0.06, 2.8),
        c_p_dry=1350.0,
        k_dry=0.17,
        k_moisture_beta=4.5,
        rho_solid=1400.0,
        D_eff_D0=6.2e-4,
        D_eff_Ea=27800.0,
        bed_porosity=0.42,
    ),
}
```

---

## 6. Module Architecture

The `pretreatment` module is a self-contained Python package within the Air Classifier Designer project. It follows the same patterns as the existing `airclassifier` package: parametric geometry, GPU-accelerated physics via Warp, and integration with the PySide6/PyVista GUI.

### 6.1 Package Structure

```
airclassifier/
├── src/airclassifier/
│   ├── pretreatment/                    # <<< NEW MODULE
│   │   ├── __init__.py                  # Public API: GP15Simulator, PretreatmentResult
│   │   ├── config.py                    # Machine, material, recipe dataclasses
│   │   │
│   │   ├── geometry/
│   │   │   ├── __init__.py
│   │   │   ├── oven.py                  # Oven chamber: electrode plates, walls, grid
│   │   │   ├── conveyor.py              # Belt geometry, material bed mesh
│   │   │   ├── electrode.py             # Electrode detail: perforations, seam, feed strips
│   │   │   └── sdf.py                   # Signed distance fields for oven boundaries
│   │   │
│   │   ├── physics/
│   │   │   ├── __init__.py
│   │   │   ├── rf_field.py              # RF electric field solver (Laplace + capacitor)
│   │   │   ├── thermal.py               # Heat equation solver (FDM / FEM)
│   │   │   ├── moisture.py              # Moisture diffusion + evaporation kinetics
│   │   │   ├── airflow.py               # EMU airflow + heater model
│   │   │   └── coupling.py              # Multi-physics coupling orchestrator
│   │   │
│   │   ├── kernels/
│   │   │   ├── __init__.py
│   │   │   ├── dielectric_heating.py    # P_v computation, loss factor update
│   │   │   ├── heat_transfer.py         # Conduction, convection BC, RF source
│   │   │   ├── drying.py                # Moisture diffusion, evaporation
│   │   │   ├── transport.py             # Material advection on conveyor
│   │   │   └── field_solve.py           # Laplace solver kernels (Jacobi/CG)
│   │   │
│   │   ├── control/
│   │   │   ├── __init__.py
│   │   │   ├── recipe.py                # 30-recipe storage, HMI recipe mirror
│   │   │   ├── controller.py            # PLC logic: gap control, MRH/MRL, temp control
│   │   │   └── safety.py                # Arc detection, recycle logic, lockout
│   │   │
│   │   ├── materials/
│   │   │   ├── __init__.py
│   │   │   ├── properties.py            # MaterialProperties class + property functions
│   │   │   └── presets.py               # Yellow pea, faba bean, oat presets
│   │   │
│   │   ├── io/
│   │   │   ├── __init__.py
│   │   │   ├── export.py                # VTK, CSV, NumPy export
│   │   │   └── visualization.py         # 3D field rendering helpers
│   │   │
│   │   └── tests/
│   │       ├── test_rf_field.py
│   │       ├── test_thermal.py
│   │       ├── test_moisture.py
│   │       ├── test_self_leveling.py
│   │       ├── test_controller.py
│   │       └── test_integration.py
│   │
│   ├── geometry/                         # Existing classifier geometry
│   ├── fluid/                            # Existing SPH flow
│   ├── particles/                        # Existing particle system
│   ├── kinetics/                         # Existing force models
│   ├── simulation/                       # Existing simulation orchestration
│   └── gui/                              # Existing PySide6 GUI
```

### 6.2 Data Flow Per Timestep

Each simulation timestep follows this sequence of coupled operations:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TIMESTEP (dt)                                                       │
│                                                                     │
│  1. ADVECT ─────── transport.advect_material(T, M, v_belt, dt)     │
│     │              Shift T and M fields by v_belt·dt along X.       │
│     │              Inject fresh material at infeed boundary.         │
│     │                                                               │
│  2. RF FIELD ───── rf_field.solve(electrode_gap, ε', ε'')          │
│     │              Solve ∇·(ε∇φ) = 0 with electrode BCs.           │
│     │              Compute |E|² field.                              │
│     │                                                               │
│  3. HEATING ────── dielectric_heating.compute_Pv(|E|², ε'', f)    │
│     │              P_v = 2π·f·ε₀·ε''·|E|² at each cell.           │
│     │                                                               │
│  4. EVAPORATION ── drying.compute_evap(T, M, h_m, T_air)          │
│     │              ṁ_evap from surface and internal models.         │
│     │                                                               │
│  5. THERMAL ────── heat_transfer.step(T, P_v, ṁ_evap, BCs, dt)   │
│     │              Advance T by dt with RF source and latent sink.  │
│     │                                                               │
│  6. MOISTURE ───── drying.step(M, D_eff(T), ṁ_evap, dt)           │
│     │              Advance M by dt with diffusion and evaporation.  │
│     │                                                               │
│  7. PROPERTIES ─── update ε'(T,M), ε''(T,M), ρ(M), c_p(T,M),    │
│     │              k(T,M), D_eff(T) for next timestep.              │
│     │                                                               │
│  8. CONTROLLER ─── controller.step(P_rf, I_a, T_outfeed)          │
│     │              Check MRH/MRL, adjust gap, check recycle,        │
│     │              temperature control mode.                        │
│     │                                                               │
│  9. RECORD ─────── Log outfeed state, KPIs, diagnostics.          │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Warp Computation Strategy

| Computation | Warp Feature | Implementation |
|---|---|---|
| RF field (Laplace equation) | `warp.fem.Grid3D` + `warp.sparse.cg` | FEM assembly on structured hex grid, CG solve with Dirichlet BCs on electrodes |
| Heat equation (explicit FDM) | Custom `@wp.kernel` with `wp.array3d` | 3D central-difference Laplacian, forward Euler, CFL-constrained dt |
| Heat equation (implicit, Phase 3) | `warp.sparse.cg` or `warp.sparse.bicgstab` | Backward Euler, sparse system assembly, iterative solve |
| Moisture diffusion | Custom `@wp.kernel` with `wp.array3d` | Same FDM approach as heat equation |
| Evaporation | Custom `@wp.kernel` | Per-cell: ṁ_evap from local T, M, surface conditions |
| Dielectric coupling (P_v) | Custom `@wp.kernel` | Embarrassingly parallel: P_v = 2πfε₀ε''|E|² per cell |
| Material advection | Custom `@wp.kernel` | Upwind or TVD scheme along X-axis |
| Property update | Custom `@wp.kernel` | Per-cell recompute of ε', ε'', ρ, c_p, k, D_eff |
| Gradient/optimization | `wp.Tape` (autodiff) | Differentiable for recipe optimization |
| Performance | `wp.capture_begin/end` (CUDA graphs) | Capture steady-state loop for minimal launch overhead |

---

## 7. Key Class Designs

### 7.1 `GP15Simulator` — Main Entry Point

```python
class GP15Simulator:
    """Digital twin of the QMTI GP-15 RF dielectric heating machine.
    
    Orchestrates geometry, physics solvers, control logic, and the
    coupled simulation loop. Provides the public API for the pretreatment
    module.
    """
    
    def __init__(
        self,
        config: MachineConfig,
        material: MaterialProperties,
        device: str = "cuda"
    ):
        """Initialize the digital twin.
        
        Args:
            config: Machine specifications (electrode dimensions, power, etc.)
            material: Feedstock properties (dielectric, thermal, moisture)
            device: Warp device ("cuda" or "cpu")
        """
    
    def load_recipe(self, recipe: Recipe) -> None:
        """Load a processing recipe (mirrors HMI recipe system).
        
        Sets electrode gap, belt speed, RF power, extraction fan,
        heater settings, MRH/MRL thresholds.
        """
    
    def run(self, duration_s: float, dt: float = 0.1) -> PretreatmentResult:
        """Run the full simulation for the specified duration.
        
        Executes the coupled physics loop, returns complete results
        including time-series of all fields and KPIs.
        """
    
    def step(self, dt: float) -> StepState:
        """Advance one timestep. For interactive / real-time GUI use."""
    
    def get_outlet_conditions(self) -> OutletState:
        """Get material state at the outfeed cross-section.
        
        Returns temperature and moisture fields at x = L_oven,
        averaged KPIs, and throughput metrics. This is the interface
        to the downstream milling module.
        """
    
    def get_mesh(self) -> tuple:
        """Return PyVista-compatible mesh for 3D visualization.
        
        Returns (vertices, faces, field_data) suitable for adding
        to the Air Classifier Designer's PyVista viewport.
        """
```

### 7.2 `MachineConfig` — GP-15 Specifications

```python
@dataclass
class MachineConfig:
    """GP-15 machine parameters. All values cross-referenced against
    the Installation and Operation Manual.
    """
    # Generator / Oscillator
    rf_frequency_hz: float = 27.12e6
    max_rf_power_kw: float = 15.0
    anode_voltage_no_load_kv: float = 9.18      # Test report: no-load
    anode_voltage_full_load_kv: float = 8.38     # Test report: full-load
    anode_current_no_load_a: float = 0.4         # Test report
    anode_current_full_load_a: float = 2.58      # Test report
    supply_voltage_v: float = 600.0              # 3-phase
    supply_kva_max: float = 42.0
    
    # Oven / Applicator
    oven_length_m: float = 1.5                   # [TBD — MEASURE from drawing]
    electrode_gap_min_m: float = 0.02            # [TBD — MEASURE from test report]
    electrode_gap_max_m: float = 0.30            # [TBD — MEASURE from test report]
    electrode_count: int = 2                     # Two plates per electrode (upper and lower)
    electrode_perforation: bool = True           # Upper plates are perforated
    
    # Conveyor
    belt_width_m: float = 0.8
    belt_speed_min_m_per_min: float = 0.1
    belt_speed_max_m_per_min: float = 2.0
    belt_thickness_m: float = 0.002              # PTFE belt ~2mm
    belt_permittivity_real: float = 2.1          # PTFE ε'
    belt_permittivity_loss: float = 0.0003       # PTFE ε''
    wear_strip_thickness_m: float = 0.001        # 24 Teflon wear strips
    top_sheet_thickness_m: float = 0.0005        # Protective top sheet
    
    # EMU (Environment Management Unit)
    heater_power_total_kw: float = 12.0          # 2 banks × 6 × 1 kW
    heater_bank_count: int = 2
    extraction_fan_capacity_m3_per_min: float = 31.1
    extraction_fan_hz_min: float = 5.0
    extraction_fan_hz_max: float = 60.0
    extraction_duct_diameter_m: float = 0.25
    extraction_max_backpressure_pa: float = 250.0
    air_cooling_cfm: float = 1119.0              # Generator cooling
    
    # Control
    recipe_capacity: int = 30
    max_recycle_restarts: int = 4
    restart_delay_s: float = 2.0
    electrode_debounce_s: float = 0.5
    ambient_temp_limit_c: float = 40.0
    
    # Machine envelope
    machine_length_m: float = 5.5
    machine_width_m: float = 2.9
    machine_height_m: float = 2.2
    machine_weight_kg: float = 2550.0
    
    @property
    def belt_stack_thickness_m(self) -> float:
        """Total thickness of dielectric layers between material and lower electrode."""
        return self.belt_thickness_m + self.wear_strip_thickness_m + self.top_sheet_thickness_m
    
    def anode_voltage_kv(self, current_a: float) -> float:
        """Compute anode voltage at given current (linear droop model)."""
        slope = (self.anode_voltage_no_load_kv - self.anode_voltage_full_load_kv) / \
                (self.anode_current_full_load_a - self.anode_current_no_load_a)
        return self.anode_voltage_no_load_kv - slope * (current_a - self.anode_current_no_load_a)
```

### 7.3 `MaterialProperties`

```python
@dataclass
class MaterialProperties:
    """Feedstock characterization for the RF simulation.
    
    All property models are parameterized as functions of temperature
    and moisture content. Coefficients are material-specific and should
    be fitted from measured data.
    """
    name: str = "protein_feedstock"
    
    # Initial conditions
    initial_moisture_wb: float = 0.10            # 10% wet basis
    target_moisture_wb: float = 0.03             # 3% target
    initial_temperature_c: float = 22.0          # Ambient
    
    # Dielectric properties at 27.12 MHz
    # ε''(T,M) = a1·M² + a2·M + a3·M·T + a4·T + a5
    dielectric_loss_coeffs: tuple = (85.0, 2.5, 0.12, 0.008, 0.02)
    # ε'(T,M) = b1·M + b2·T + b3
    dielectric_const_coeffs: tuple = (25.0, -0.05, 2.5)
    
    # Thermal properties
    c_p_dry: float = 1380.0                      # J/(kg·K)
    c_p_water: float = 4186.0                    # J/(kg·K)
    k_dry: float = 0.18                          # W/(m·K)
    k_moisture_beta: float = 4.0                 # k sensitivity to M
    rho_solid: float = 1450.0                    # kg/m³ (particle density)
    
    # Moisture diffusivity: D_eff = D0 · exp(-Ea / (R·T))
    D_eff_D0: float = 5.7e-4                     # m²/s
    D_eff_Ea: float = 28500.0                    # J/mol
    
    # Evaporation model
    k_evap: float = 1.5e-4                       # 1/(°C·s) rate constant
    T_evap_threshold_c: float = 40.0             # °C
    
    # Bed geometry
    bed_depth_m: float = 0.05                    # 50 mm typical
    bed_porosity: float = 0.40                   # Void fraction
    
    def eps_loss(self, T_c: float, M_wb: float) -> float:
        """Dielectric loss factor ε''(T, M)."""
        a1, a2, a3, a4, a5 = self.dielectric_loss_coeffs
        return a1*M_wb**2 + a2*M_wb + a3*M_wb*T_c + a4*T_c + a5
    
    def eps_real(self, T_c: float, M_wb: float) -> float:
        """Dielectric constant ε'(T, M)."""
        b1, b2, b3 = self.dielectric_const_coeffs
        return b1*M_wb + b2*T_c + b3
    
    def specific_heat(self, T_c: float, M_wb: float) -> float:
        """Specific heat capacity c_p(M) via linear mixing [J/(kg·K)]."""
        return self.c_p_dry * (1.0 - M_wb) + self.c_p_water * M_wb
    
    def thermal_conductivity(self, T_c: float, M_wb: float) -> float:
        """Effective thermal conductivity k(M) [W/(m·K)]."""
        k_solid = self.k_dry * (1.0 + self.k_moisture_beta * M_wb)
        k_air = 0.026  # W/(m·K) at ~50°C
        return k_solid * (1.0 - self.bed_porosity) + k_air * self.bed_porosity
    
    def bulk_density(self, M_wb: float) -> float:
        """Bulk density ρ(M) [kg/m³]."""
        M_db = M_wb / (1.0 - M_wb)  # Convert to dry basis
        return self.rho_solid * (1.0 - self.bed_porosity) * (1.0 + M_db)
    
    def moisture_diffusivity(self, T_c: float) -> float:
        """Effective moisture diffusivity D_eff(T) [m²/s]."""
        T_K = T_c + 273.15
        R = 8.314
        return self.D_eff_D0 * math.exp(-self.D_eff_Ea / (R * T_K))
```

### 7.4 `Recipe` — Mirrors HMI Recipe System

```python
@dataclass
class Recipe:
    """GP-15 HMI recipe. Up to 30 can be stored.
    
    Maps directly to the GP-15 Recipe Edit Screen parameters.
    """
    name: str
    recipe_number: int                           # 1-30, 0 = manual mode
    
    # Process setpoints
    electrode_gap_mm: float                      # Gap setpoint
    belt_speed_m_per_min: float                  # Conveyor speed
    rf_power_enabled: bool = True
    
    # Anode current protection
    mrh_amps: float = 2.6                        # Meter Relay High (overcurrent trip)
    mrl_amps: float = 2.0                        # Meter Relay Low (drive stop threshold)
    
    # EMU settings
    extraction_fan_hz: float = 30.0
    heater_bank_1_on: bool = True
    heater_bank_2_on: bool = True
    heater_fan_hz: float = 30.0
    
    # Temperature control (optional automatic mode)
    temp_control_enabled: bool = False
    temp_setpoint_c: float = 60.0
    temp_sensors_active: tuple = (True, True, True, True, True, True)  # 6 sensors
    temp_envelope_time_s: float = 10.0           # Correction interval
```

---

## 8. Warp Kernel Architecture

All GPU kernels follow Warp best practices: pre-allocated arrays, explicit `@wp.kernel` decorators, no allocations inside loops, and CUDA graph compatibility.

### 8.1 Kernel: Dielectric Heating (P_v Computation)

```python
@wp.kernel
def compute_power_density(
    e_field_sq: wp.array3d(dtype=float),        # |E|² at each cell [V²/m²]
    eps_loss: wp.array3d(dtype=float),           # ε'' at each cell
    power_density: wp.array3d(dtype=float),      # Output: P_v [W/m³]
    two_pi_f_eps0: float,                        # 2π · 27.12e6 · 8.854e-12
):
    """Compute volumetric RF power density at each grid cell.
    
    P_v = 2π · f · ε₀ · ε'' · |E|²
    
    Called every timestep after the E-field and material properties
    have been updated.
    """
    i, j, k = wp.tid()
    E2 = e_field_sq[i, j, k]
    loss = eps_loss[i, j, k]
    power_density[i, j, k] = two_pi_f_eps0 * loss * E2
```

### 8.2 Kernel: Property Update

```python
@wp.kernel
def update_material_properties(
    T: wp.array3d(dtype=float),                  # Temperature [°C]
    M: wp.array3d(dtype=float),                  # Moisture [wet basis fraction]
    eps_loss: wp.array3d(dtype=float),            # Output: ε''
    eps_real: wp.array3d(dtype=float),            # Output: ε'
    rho_cp: wp.array3d(dtype=float),              # Output: ρ·c_p [J/(m³·K)]
    k_eff: wp.array3d(dtype=float),               # Output: k [W/(m·K)]
    cell_is_material: wp.array3d(dtype=int),      # 1 if material, 0 if air/belt
    # Loss factor coefficients: ε'' = a1·M² + a2·M + a3·M·T + a4·T + a5
    a1: float, a2: float, a3: float, a4: float, a5: float,
    # Dielectric constant: ε' = b1·M + b2·T + b3
    b1: float, b2: float, b3: float,
    # Thermal
    c_p_dry: float, c_p_water: float,
    k_dry: float, k_beta: float, k_air: float,
    rho_solid: float, porosity: float,
):
    """Update all material properties from current T and M fields.
    
    Must be called after every thermal and moisture solve step so that
    the coupling between fields is maintained.
    """
    i, j, k_idx = wp.tid()
    
    if cell_is_material[i, j, k_idx] == 0:
        eps_loss[i, j, k_idx] = 0.0
        eps_real[i, j, k_idx] = 1.0  # Air
        rho_cp[i, j, k_idx] = 1.2 * 1005.0  # Air: ρ·c_p
        k_eff[i, j, k_idx] = 0.026  # Air conductivity
        return
    
    temp = T[i, j, k_idx]
    moist = M[i, j, k_idx]
    
    # Dielectric loss factor
    eps_loss[i, j, k_idx] = a1*moist*moist + a2*moist + a3*moist*temp + a4*temp + a5
    
    # Dielectric constant
    eps_real[i, j, k_idx] = b1*moist + b2*temp + b3
    
    # Specific heat (linear mixing)
    cp = c_p_dry * (1.0 - moist) + c_p_water * moist
    
    # Bulk density
    M_db = moist / wp.max(1.0 - moist, 1.0e-6)
    rho = rho_solid * (1.0 - porosity) * (1.0 + M_db)
    
    rho_cp[i, j, k_idx] = rho * cp
    
    # Thermal conductivity
    k_solid = k_dry * (1.0 + k_beta * moist)
    k_eff[i, j, k_idx] = k_solid * (1.0 - porosity) + k_air * porosity
```

### 8.3 Kernel: Explicit Heat Equation Step

```python
@wp.kernel
def heat_conduction_step(
    T: wp.array3d(dtype=float),                  # Temperature [°C]
    T_new: wp.array3d(dtype=float),              # Output temperature
    P_v: wp.array3d(dtype=float),                # RF source [W/m³]
    evap_rate: wp.array3d(dtype=float),          # Evaporation rate [kg/(m³·s)]
    rho_cp: wp.array3d(dtype=float),             # ρ·c_p [J/(m³·K)]
    k_eff: wp.array3d(dtype=float),              # k [W/(m·K)]
    L_v: float,                                  # Latent heat [J/kg]
    dx: float, dy: float, dz: float, dt: float,
    nx: int, ny: int, nz: int,
):
    """Advance temperature field by one explicit FDM timestep.
    
    T_new = T + dt/(ρ·c_p) × [∇·(k∇T) + P_v - L_v·ṁ_evap]
    
    Uses second-order central differences for the Laplacian with
    variable conductivity. Boundary cells use one-sided differences.
    """
    i, j, k = wp.tid()
    
    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        T_new[i, j, k] = T[i, j, k]  # Boundary handling in separate kernel
        return
    
    # Central difference Laplacian with variable k
    # ∂/∂x(k·∂T/∂x) ≈ (k_{i+½}(T_{i+1}-T_i) - k_{i-½}(T_i-T_{i-1})) / dx²
    k_xp = 0.5 * (k_eff[i, j, k] + k_eff[i+1, j, k])
    k_xm = 0.5 * (k_eff[i, j, k] + k_eff[i-1, j, k])
    lap_x = (k_xp * (T[i+1, j, k] - T[i, j, k]) - k_xm * (T[i, j, k] - T[i-1, j, k])) / (dx * dx)
    
    k_yp = 0.5 * (k_eff[i, j, k] + k_eff[i, j+1, k])
    k_ym = 0.5 * (k_eff[i, j, k] + k_eff[i, j-1, k])
    lap_y = (k_yp * (T[i, j+1, k] - T[i, j, k]) - k_ym * (T[i, j, k] - T[i, j-1, k])) / (dy * dy)
    
    k_zp = 0.5 * (k_eff[i, j, k] + k_eff[i, j, k+1])
    k_zm = 0.5 * (k_eff[i, j, k] + k_eff[i, j, k-1])
    lap_z = (k_zp * (T[i, j, k+1] - T[i, j, k]) - k_zm * (T[i, j, k] - T[i, j, k-1])) / (dz * dz)
    
    laplacian = lap_x + lap_y + lap_z
    source = P_v[i, j, k]
    sink = L_v * evap_rate[i, j, k]
    
    rc = wp.max(rho_cp[i, j, k], 1.0)  # Guard against zero
    T_new[i, j, k] = T[i, j, k] + dt / rc * (laplacian + source - sink)
```

### 8.4 Kernel: Moisture Diffusion and Evaporation

```python
@wp.kernel
def moisture_step(
    M: wp.array3d(dtype=float),                  # Moisture content [wet basis fraction]
    M_new: wp.array3d(dtype=float),              # Output moisture
    T: wp.array3d(dtype=float),                  # Temperature [°C]
    evap_rate: wp.array3d(dtype=float),          # Output: evaporation rate [kg/(m³·s)]
    cell_is_material: wp.array3d(dtype=int),
    rho_dry: wp.array3d(dtype=float),            # Dry-basis bulk density
    D0: float, Ea: float, R_gas: float,          # Diffusivity Arrhenius params
    k_evap: float, T_threshold: float,           # Evaporation rate params
    dx: float, dy: float, dz: float, dt: float,
    nx: int, ny: int, nz: int,
):
    """Advance moisture field by one timestep.
    
    ∂M/∂t = ∇·(D_eff·∇M) - ṁ_evap/ρ_dry
    
    Evaporation rate: ṁ_evap = ρ_dry · k_evap · M · max(0, T - T_threshold)
    """
    i, j, k = wp.tid()
    
    if cell_is_material[i, j, k] == 0:
        M_new[i, j, k] = 0.0
        evap_rate[i, j, k] = 0.0
        return
    
    if i <= 0 or i >= nx - 1 or j <= 0 or j >= ny - 1 or k <= 0 or k >= nz - 1:
        M_new[i, j, k] = M[i, j, k]
        evap_rate[i, j, k] = 0.0
        return
    
    temp = T[i, j, k]
    moist = M[i, j, k]
    
    # Moisture diffusivity: D_eff = D0 · exp(-Ea/(R·T_K))
    T_K = temp + 273.15
    D_eff = D0 * wp.exp(-Ea / (R_gas * T_K))
    
    # Diffusion (central differences, constant D within cell)
    lap_M = D_eff * (
        (M[i+1,j,k] - 2.0*moist + M[i-1,j,k]) / (dx*dx) +
        (M[i,j+1,k] - 2.0*moist + M[i,j-1,k]) / (dy*dy) +
        (M[i,j,k+1] - 2.0*moist + M[i,j,k-1]) / (dz*dz)
    )
    
    # Evaporation
    rho_d = wp.max(rho_dry[i, j, k], 1.0)
    dT = wp.max(temp - T_threshold, 0.0)
    m_evap = rho_d * k_evap * wp.max(moist, 0.0) * dT
    evap_rate[i, j, k] = m_evap
    
    # Update moisture
    M_new[i, j, k] = wp.max(moist + dt * (lap_M - m_evap / rho_d), 0.0)
```

### 8.5 Kernel: Conveyor Advection (Upwind)

```python
@wp.kernel
def advect_material(
    field: wp.array3d(dtype=float),              # Scalar field to advect (T or M)
    field_new: wp.array3d(dtype=float),          # Output
    v_belt_dx_dt: float,                         # v_belt · dt / dx (Courant number)
    inlet_value: float,                          # Value at infeed boundary
    nx: int, ny: int, nz: int,
):
    """Advect a scalar field along the positive X-axis (conveyor direction).
    
    First-order upwind scheme. Courant number must be < 1 for stability.
    Infeed boundary injects fresh material at inlet_value.
    """
    i, j, k = wp.tid()
    
    if i == 0:
        # Infeed: inject fresh material
        field_new[i, j, k] = inlet_value
    elif i < nx:
        # Upwind advection
        field_new[i, j, k] = field[i, j, k] - v_belt_dx_dt * (field[i, j, k] - field[i-1, j, k])
    else:
        field_new[i, j, k] = field[i, j, k]
```

### 8.6 Kernel: Convective Boundary Condition (Top of Bed)

```python
@wp.kernel
def apply_convection_bc(
    T: wp.array3d(dtype=float),
    j_surface: int,                              # Y-index of bed surface
    h_conv: float,                               # Convective HTC [W/(m²·K)]
    T_air: float,                                # Oven air temperature [°C]
    k_eff: wp.array3d(dtype=float),
    dy: float, dt: float,
    rho_cp: wp.array3d(dtype=float),
    nx: int, nz: int,
):
    """Apply convective heat transfer at the material bed surface.
    
    -k·∂T/∂y|_surface = h·(T_surface - T_air)
    
    Modifies T at the surface cells to include convective flux.
    """
    i, k = wp.tid()
    j = j_surface
    
    if i >= nx or k >= nz:
        return
    
    T_s = T[i, j, k]
    q_conv = h_conv * (T_s - T_air)
    rc = wp.max(rho_cp[i, j, k], 1.0)
    
    # Surface flux applied over half-cell thickness
    T[i, j, k] = T_s - dt * q_conv / (rc * dy * 0.5)
```

---

## 8. Control System Simulation

The GP-15's PLC logic is replicated as a discrete-event controller that runs at each simulation timestep, after the physics solve.

### 8.1 Electrode Gap Control

The electrode gap determines RF power density (`E ∝ V/gap`, so `P_v ∝ 1/gap²`). The controller manages the gap through several modes:

**Homing sequence (startup):**
1. Drive electrode to upper limit switch
2. Reset encoder to zero
3. Drive to maximum gap position
4. Ready for setpoint commands

**Setpoint tracking:**
When a recipe is loaded, the electrode drives to the recipe's gap setpoint when RF power is enabled.

**MRH protection (overcurrent):**
If anode current exceeds the MRH threshold (typically 2.6 A), the electrode gap automatically increases (reduces power). This is a safety-critical feature that prevents valve damage.

**MRL threshold (drive stop):**
The electrode drive stops moving when anode current falls below the MRL threshold (typically 2.0 A). This prevents the gap from overshooting during MRH corrections.

**Debounce logic:**
Electrode position is not altered until control buttons have been released for 0.5 seconds, preventing accidental adjustments.

### 8.2 Automatic Temperature Control Mode

The GP-15 supports automatic closed-loop temperature control using the 6 optical sensors at the outfeed:

```python
class TemperatureController:
    """Automatic temperature control using outfeed sensor array.
    
    Algorithm (from GP-15 manual Screen 14):
    1. Compute average temperature from active sensors
    2. If T_avg > T_setpoint:
       a. Increase electrode gap (reduce power)
       b. If still too hot after envelope_time: increase conveyor speed
    3. If T_avg < T_setpoint:
       a. Decrease electrode gap (increase power)
       b. If still too cold after envelope_time: decrease conveyor speed
    4. Wait envelope_time before next correction
    """
    
    def __init__(self, setpoint_c: float, envelope_time_s: float,
                 active_sensors: tuple):
        self.setpoint = setpoint_c
        self.envelope_time = envelope_time_s
        self.active_sensors = active_sensors
        self.last_correction_time = 0.0
        self.gap_correction_applied = False
```

### 8.3 Recycle Logic (Arc Recovery)

```python
class RecycleController:
    """GP-15 automatic recycling for arc/flashover recovery.
    
    Sequence:
    1. Arc detected (E-field exceeds breakdown threshold)
    2. RF power off immediately
    3. Electrode drives to maximum gap
    4. Wait restart_delay (typically 2 seconds)
    5. RF restores, electrode returns to setpoint
    6. Increment recycle counter
    7. If counter >= max_restarts (4): LOCKOUT — operator intervention required
    
    Counter reset: automatic after monitor_interval, which is calculated
    from conveyor speed. Parameterized, not hard-coded.
    """
    
    def __init__(self, config: MachineConfig):
        self.max_restarts = config.max_recycle_restarts
        self.restart_delay = config.restart_delay_s
        self.counter = 0
        self.locked_out = False
        self.in_recycle = False
        self.recycle_timer = 0.0
        self.monitor_interval = 0.0  # Computed from belt speed
    
    def update_monitor_interval(self, belt_speed_m_per_min: float):
        """Compute recycle counter reset interval from conveyor speed.
        
        The exact formula is not specified in the manual. It depends on
        conveyor speed, max restarts, and restart interval. This should
        be parameterized from the HMI 'Monitor Interval' setting.
        """
        # Placeholder: reset after one full belt transit of the oven
        if belt_speed_m_per_min > 0:
            self.monitor_interval = 60.0 * 1.5 / belt_speed_m_per_min  # L_oven/v_belt
```

### 8.4 Anode Current Model

The anode current is the primary operator feedback signal. It indicates how much RF power is being delivered to the load:

```python
def compute_anode_current(P_rf_total_kw: float, config: MachineConfig) -> float:
    """Compute anode current from total RF power delivered.
    
    Uses the linear droop model fitted to the GP-15 test report data:
    - No-load: V_a = 9.18 kV, I_a = 0.4 A, P = 0 kW
    - Full-load: V_a = 8.38 kV, I_a = 2.58 A, P = 15 kW
    
    The relationship P = V_a × I_a × η (where η is the oscillator
    efficiency) gives a quadratic in I_a. For the simulation,
    a simpler linear interpolation on the test report data is used.
    """
    # Linear interpolation: I_a = I_a_idle + (I_a_full - I_a_idle) × (P / P_max)
    fraction = min(P_rf_total_kw / config.max_rf_power_kw, 1.0)
    I_a = config.anode_current_no_load_a + \
          (config.anode_current_full_load_a - config.anode_current_no_load_a) * fraction
    return I_a
```

---

## 9. Integration with Air Classifier Pipeline

### 9.1 Output Interface

The pretreatment module's output feeds directly into the milling stage. The `OutletState` dataclass captures the material condition at the GP-15 outfeed:

```python
@dataclass
class OutletState:
    """Material state at the GP-15 outfeed. Input to the milling module."""
    
    # Spatially-resolved fields at the outfeed cross-section (Y × Z)
    temperature_field: np.ndarray                # [ny, nz] in °C
    moisture_field: np.ndarray                   # [ny, nz] wet basis fraction
    
    # Bulk averages
    avg_temperature_c: float                     # Mean outfeed temperature
    avg_moisture_wb: float                       # Mean outfeed moisture
    moisture_uniformity: float                   # Coefficient of variation (std/mean)
    
    # Process metrics
    throughput_kg_per_hr: float                  # Mass flow rate
    total_energy_kwh: float                      # Total RF energy delivered
    specific_energy_kwh_per_kg: float            # Energy per kg water removed
    residence_time_s: float                      # L_oven / v_belt
    
    # Quality indicators
    max_temperature_c: float                     # Peak temperature (protein denaturation risk)
    protein_denaturation_fraction: float          # Estimated from time-temperature history
```

The downstream milling module consumes primarily `avg_moisture_wb` and `moisture_uniformity`. Non-uniform drying produces inconsistent particle sizes during milling, which degrades protein separation in the classifier. The `max_temperature_c` flags potential protein denaturation — the simulation should warn if any material exceeds ~70°C for extended periods (denaturation onset for pea protein).

### 9.2 Pipeline Integration

```python
from airclassifier.pretreatment import GP15Simulator, MachineConfig, Recipe
from airclassifier.pretreatment.materials.presets import MATERIAL_PRESETS
from airclassifier.simulation.classification_flow_physics import ClassificationFlowPhysics

# Stage 1: RF Pretreatment
config = MachineConfig()
material = MATERIAL_PRESETS["yellow_pea"]
material.initial_moisture_wb = 0.10
material.bed_depth_m = 0.04

gp15 = GP15Simulator(config=config, material=material, device="cuda")
gp15.load_recipe(Recipe(
    name="yellow_pea_standard",
    recipe_number=1,
    electrode_gap_mm=80,
    belt_speed_m_per_min=0.5,
    extraction_fan_hz=35.0,
))
result = gp15.run(duration_s=300)
outlet = gp15.get_outlet_conditions()

print(f"Outlet moisture: {outlet.avg_moisture_wb:.1%}")
print(f"Uniformity (CV): {outlet.moisture_uniformity:.3f}")
print(f"Energy efficiency: {outlet.specific_energy_kwh_per_kg:.2f} kWh/kg water")

# Stage 2: Feed to milling (existing project module)
# The pretreated material moisture and temperature inform
# the milling energy requirement and particle size distribution

# Stage 3: Air Classification (existing project)
# classifier = ClassificationFlowPhysics(...)
```

### 9.3 GUI Integration

The pretreatment module integrates with the existing PySide6/PyVista GUI through:

1. **3D viewport**: The oven geometry (electrodes, belt, material bed) renders as a PyVista mesh alongside the existing classifier assembly. Temperature and moisture fields are visualized as color maps on the material bed mesh.

2. **Simulation panel**: A new "Pretreatment" tab in the simulation control dock provides recipe selection, run/pause/stop controls, and real-time KPI cards (outlet moisture, temperature, power, anode current).

3. **Assembly configuration**: The oven appears as an optional upstream component in the assembly configuration dialog, connected to the classifier's feed hopper via the process pipeline.

---

## 10. Validation Targets

### 10.1 Manual Example Calculation

The GP-15 manual provides a worked example that serves as the primary validation target:

```
Given:
  Throughput           = 600 kg/hr
  Inlet moisture       = 4% (wet basis)
  Outlet moisture      = 3% (wet basis)
  Water removal rate   = 1 kg/kWh (high surface-to-volume product)

Water removed = 600 × (0.04 - 0.03) / (1 - 0.03) = 6.19 kg/hr
RF power required = 6.19 / 1.0 = 6.19 kW

Manual answer: approximately 11 kW (includes system losses and ~56% efficiency)
```

The simulation must reproduce this result: given 600 kg/hr at 4% → 3% moisture, the total RF power delivered to the material should be ~6.2 kW, and the generator power (accounting for oscillator and coupling losses) should be ~11 kW.

### 10.2 Self-Leveling Verification

Starting from a deliberately non-uniform initial moisture field (e.g., one half at 12%, other half at 8%), the simulation should show convergence toward uniform moisture over the residence time. The moisture CV at the outfeed should be less than 5% when starting from a 40% CV at the infeed.

### 10.3 Operating Curve Validation

| Operating Point | Expected Ia | Expected Va | P_rf |
|---|---|---|---|
| No load (empty belt) | 0.4 A | 9.18 kV | ~0 kW |
| Light load | ~1.0 A | ~8.9 kV | ~4 kW |
| Medium load | ~1.8 A | ~8.6 kV | ~9 kW |
| Full rated power | 2.58 A | 8.38 kV | 15 kW |

### 10.4 Energy Balance

At every timestep, verify:
```
P_rf_input = P_heating + P_evaporation + P_convection_loss + P_conduction_loss

Where:
  P_heating     = ∫ ρ·c_p·∂T/∂t dV
  P_evaporation = ∫ L_v·ṁ_evap dV
  P_convection  = ∫ h·(T_s - T_air) dA
  P_conduction  = boundary conduction losses
```

The energy balance error should be less than 0.1% per timestep.

---

## 11. Development Phases

### Phase 1: Foundation (Weeks 1–3)

1. Project scaffolding: package structure, config dataclasses, test harness, `wp.init()` on target device.
2. Geometry module: oven grid generation using `warp.fem.Grid3D`, parameterized by electrode gap, belt width, oven length. Cell tagging (material / air / belt).
3. RF field solver: uniform parallel-plate capacitor model with series-capacitor voltage division for the layered stack. Validate: E_bed matches analytic formula.
4. Basic thermal kernel: explicit FDM heat conduction with constant properties. Validate against 1D analytic solution (semi-infinite slab with constant surface flux).

### Phase 2: Core Physics (Weeks 4–6)

1. Dielectric heating coupling: connect E-field to thermal solver via P_v. Implement moisture-dependent ε''.
2. Moisture transport: diffusion + evaporation model. Validate self-leveling behavior with non-uniform initial conditions.
3. Property coupling: temperature- and moisture-dependent ρ, c_p, k, D_eff, ε''. Full nonlinear coupling at each timestep.
4. Conveyor transport: material advection along X-axis. Infeed injection, outfeed collection, Courant number control.

### Phase 3: Control and Realism (Weeks 7–9)

1. Controller module: recipe system (30 recipes), electrode gap control with MRH/MRL logic, anode current model with voltage droop.
2. Recycle/safety logic: arc detection (E-field threshold), recycle sequence with counter and lockout, parameterized monitor interval.
3. EMU airflow model: forced convection BCs, extraction fan capacity (31.1 m³/min), heater array power (12 kW), condensation avoidance.
4. Temperature control mode: 6-sensor feedback, gap + speed corrections, envelope time logic.
5. 2D Laplace fringe field correction: pre-solve Y-Z cross-section, store as correction map.
6. Calibration: tune k_evap against 1 kg/kWh target, tune oscillator efficiency against 11 kW manual example.

### Phase 4: Integration and Optimization (Weeks 10–12)

1. Pipeline integration: OutletState → milling module. Full pretreatment → milling → classification workflow.
2. GUI integration: PyVista 3D rendering of oven + fields, simulation control panel, KPI dashboard.
3. Differentiable simulation: enable `wp.Tape` for gradient-based recipe optimization (minimize energy for target moisture + uniformity).
4. Performance: CUDA graph capture for steady-state loop. Profile kernel launches, tune block_dim.

### Phase 5: Validation and Documentation (Weeks 13–14)

1. Validation suite: reproduce manual examples, operating curves, energy balance checks.
2. Sensitivity analysis: sweep electrode gap, belt speed, bed depth. Document response surfaces.
3. Electrode geometry refinements: perforation correction factors, center seam model, feed strip proximity effects.
4. API documentation, Jupyter notebooks for each physics module, integration tests.

---

## 12. Technical Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Dielectric property data unavailable for target feedstock | Inaccurate RF heating prediction | Use literature values for similar legume isolates. Plan experimental characterization with network analyzer at 27 MHz. Implement property fitting framework for measured data. |
| Oven length and gap range unverified | All residence-time and power-density calculations wrong | **[CRITICAL]** Obtain engineering drawing #59-0001-0005_REV1 and commissioned Test Report before Phase 2 validation. |
| Numerical instability in explicit thermal solver | Simulation divergence | Enforce CFL: `dt < dx²/(2α·d)`. Implement adaptive timestepping. Fall back to implicit solver (`warp.sparse.cg`) if needed. |
| RF field fringe effects significant | Underestimated edge heating, overheated belt edges | Phase 1: uniform field. Phase 3: 2D Laplace correction. Validate against known electrode geometries. |
| Coupling instability (RF ↔ thermal ↔ moisture) | Non-physical oscillations, energy non-conservation | Operator splitting with sufficiently small dt. Sub-cycle fast physics. Monitor energy balance at every step; halt if error > 1%. |
| Material bed is granular, not continuum | Continuum assumption breaks for coarse particles | Treat bed as effective continuum with porosity correction. Add DEM option if experimental validation shows continuum model is inadequate. |
| Warp FEM on 3D grid too slow for interactive use | Long iteration times, poor GUI responsiveness | Use explicit FDM kernels (bypass FEM assembly overhead). Use CUDA graph capture. Reduce resolution adaptively for interactive mode. |
| Electrode perforation effects stronger than expected | Non-uniform heating pattern missed by uniform model | Phase 3+ perforation correction. If insufficient, compute 3D field on fine sub-grid near electrode, project onto coarse simulation grid. |

---

## 13. Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | >= 3.10 | Runtime |
| `warp-lang` | >= 1.11.0 | GPU simulation (kernels, FEM, sparse solvers, autodiff) |
| `numpy` | >= 2.0.0 | Array interchange, post-processing |
| `scipy` | >= 1.15.0 | Material property interpolation, optimization |
| `PySide6` | >= 6.5.0 | GUI framework (shared with Air Classifier Designer) |
| `pyvista` | >= 0.42.0 | 3D visualization (shared) |
| `pyvistaqt` | >= 0.11.0 | PyVista-Qt bridge (shared) |
| `matplotlib` | >= 3.7.0 | 2D plots, diagnostics |
| `vtk` | >= 9.2.0 | VTK export for ParaView |
| `pytest` | >= 7.0 | Testing framework |
| CUDA Toolkit | >= 11.8 | GPU acceleration via Warp |

**Hardware:** NVIDIA GPU with >= 8 GB VRAM (RTX 3070+). The simulation grid at 80×25×30 = 60,000 cells requires ~50 MB GPU memory for all field arrays. Gradient computation (autodiff) approximately doubles memory requirements.

---

## 14. Success Criteria

The pretreatment module is complete when:

1. **Accuracy**: Moisture reduction from 4% → 3% at 600 kg/hr requires ~11 kW generator power (matching GP-15 manual example).
2. **Self-leveling**: Starting from 40% CV moisture non-uniformity at infeed, outfeed CV < 5%.
3. **Operating curves**: Anode current and voltage match test report data at no-load, half-load, and full-load.
4. **Energy conservation**: Energy balance error < 0.1% per timestep.
5. **Controller fidelity**: MRH/MRL gap control, recycle sequence (4 restarts + lockout), temperature control mode all function correctly.
6. **Pipeline integration**: OutletState feeds milling module; full pretreatment → milling → classification workflow produces plausible results.
7. **Performance**: GPU simulation runs faster than real-time at 60,000-cell resolution on RTX 3070.
8. **GUI**: 3D oven visualization with live temperature/moisture fields renders in the Air Classifier Designer viewport.
