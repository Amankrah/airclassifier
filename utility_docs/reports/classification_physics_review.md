# Classification Flow Physics — Module Review

**Files:** `classification_flow_physics.py` (6,428 lines), `classification.py` (1,958 lines)  
**Date:** 2026-02-07  
**Scope:** Physics implementation, separation logic, integration boundaries, bugs

---

## Module Architecture

### Two Classification Paths

The system supports two operating modes controlled by `use_preclassification`:

**With preclassification** (default): Air Supply → Venturi → Zigzag → Wheel → Cyclones → Bag Filter. Particles enter at zone 0 (venturi solids inlet). The zigzag pre-classifies coarse starch (>~50 µm) before the wheel makes the fine cut (~25 µm).

**Without preclassification** (wheel-only): Air+Solids → Three-Point Junction → Wheel → Cyclones → Bag Filter. No venturi, zigzag, or dropout. Particles enter at zone 34 (wheel housing). Simpler path for when the feed is already partially classified or when throughput is paramount.

### Zone Map

| Zone | Component | Role |
|------|-----------|------|
| -1 | INACTIVE | Deactivated particles |
| 0–2 | Venturi (inlet/throat/divergent) | Particle entrainment |
| 10 | Duct venturi→zigzag | Transport with dropout |
| 20–23 | Zigzag (entry/stages/fines/coarse) | Pre-classification by terminal velocity |
| 30 | Coarse outlet | Collected starch (zigzag reject) |
| 34–37 | Wheel (housing/fines/hopper/collected) | Centrifugal fine cut |
| 40–41 | Elbow + duct zigzag→cyclone | Transport |
| 50–52 | Cyclones (primary/secondary/tertiary) | Staged fines collection |
| 55–57 | Cyclone dust outlets | Collected fractions |
| 60–61 | Elbow + duct cyclone→bag | Transport |
| 70 | Bag filter | Final capture |
| 75 | Bag filter dust | Collected ultra-fines |
| 80 | Clean air exit | Exhaust |

### Configuration System

`ClassificationFlowConfig` dataclass with `from_air_and_feed_results()` factory. Particle parameters (num_particles, particle_density, visual_particle_diameter, sphericity) from feed_result. Air parameters (density, viscosity, flow rate) from air_result. Continuous feeding mode activates particles gradually at `particle_feed_rate` instead of all at t=0. Max loading ratio caps venturi entrainment at μ=2.0 (dilute-phase pneumatic transport limit).

### Geometry Extraction

`extract_geometry(assembly)` pulls all dimensions from `ClassificationSystemAssembly` with no hardcoding. Returns `ComponentGeometry` dataclass for each component plus `ConnectionPath` for ducts. This is well-designed — geometry changes in the assembly automatically propagate to the physics simulator without manual synchronization.

---

## Physics Implementation — Warp GPU Kernels

### Drag Models

Four drag coefficient correlations with automatic regime selection:

- **Stokes** (Re < 0.1): C_d = 24/Re. Exact for creeping flow.
- **Schiller-Naumann** (0.1 < Re < 1000): C_d = (24/Re)(1 + 0.15 Re^0.687). Workhorse model for most flour particles.
- **Haider-Levenspiel**: Non-spherical correction using sphericity φ. Starch φ=0.8–0.9 (granules), protein φ=0.6–0.8 (irregular agglomerates). This is the right model for flour classification.
- **Terminal velocity**: Stokes-regime base with iterative intermediate-regime correction. Typical values: 10 µm protein v_t ≈ 0.005 m/s, 50 µm starch v_t ≈ 0.12 m/s, 100 µm starch v_t ≈ 0.4 m/s.

All formulas are physically correct and well-implemented.

### Venturi Physics

Continuity equation Q = A×v applied through converging-throat-diverging geometry. At throat: v_throat = v_inlet × (D_inlet/D_throat)². For the default geometry (D_inlet=80mm, D_throat=40mm): area ratio = 4, so v_throat = 4 × v_inlet. Bernoulli pressure drop computed correctly. Loading ratio μ = ṁ_solids/ṁ_air checked against choked-flow limit (Ma=1 at throat, C_d=0.985).

Radial containment and zone transitions are handled correctly. The venturi requires vel[1] > 0.5 m/s before allowing transition to duct (zone 10), preventing batch-initialized particles from teleporting past the venturi before drag accelerates them. Good defensive design.

### Zigzag Classifier Physics

Deflector plate physics modeled with three velocity zones per stage:

1. **Throat** at plate tip: v_throat = v_mean × (channel_width/throat_width) from continuity. This is where the acceleration happens.
2. **Separation zone** behind plate: v_zone = v_mean × velocity_ratio_zone (recirculation region). This is where particles separate based on terminal velocity vs local air velocity.
3. **Transport** between zones: v = v_mean (bulk flow carrying fines upward).

Plates alternate left/right at each stage, positioned at vertical center. Turbulent dispersion uses air velocity as reference (not particle velocity), which is correct — slow particles near d50 need turbulence most to break equilibrium.

Cut size formula: d50 = sqrt(18 × μ × v_air / (g × Δρ)). At v_air = 2 m/s → d50 ≈ 22 µm.

### Wheel Classifier Physics

Radial velocity from continuity through blade gaps: open arc at radius r = 2πr - n_blades × t_blade, flow area A(r) = open_arc × wheel_width, v_r = -Q/A(r) (inward = negative, drawing fines through).

Tangential velocity: solid-body rotation v_tan = ω×r within wheel geometry (hub to rim). Creates centrifugal force F_c = m×ω²×r pushing particles outward.

Separation force ratio: F_c/F_d where F_c = m×ω²×r (centrifugal, outward) and F_d = 3π×μ×d×|v_r| (Stokes drag, inward). F_c/F_d > 1 → COARSE (rejected), F_c/F_d < 1 → FINES (passed through).

Rotating blade collision detection with angular spacing = 2π/n_blades, particle angle relative to rotating frame, collision if arc distance < blade_thickness/2.

### Cyclone Physics — Rankine Vortex

Tangential velocity: inner core (r < r_vf) solid body v_tan = v_inlet × (R/r_core) × (r/r_core); outer region (r ≥ r_vf) free vortex v_tan = v_inlet × (R/r) from angular momentum conservation. This is the standard Rankine vortex model used in cyclone design.

Radial velocity: outer region inward flow v_r = -0.15 × v_inlet × (0.5 + 0.5 × (1-r_frac)); inner region negligible (primarily axial). Cone section intensifies inward flow.

Axial velocity: outer region DOWNWARD (carries large particles to dust outlet); inner region UPWARD (carries small particles to vortex finder).

Cut size via Lapple equation: d50 = sqrt(9μW / (2πNv_in Δρ)) where N = number of spiral turns (5–6). For D=300mm, W=75mm, v_in=15 m/s → d50 ≈ 5–10 µm.

### Wall Collision

Inelastic reflection: normal component v_n' = -e × v_n (restitution e = 0.2–0.4 for flour on steel), tangent component v_t' = (1-μ) × v_t (friction μ = 0.3–0.5). Appropriate for the particle-wall interactions in this system.

### Main Kernel Integration

Semi-implicit Euler: vel += accel × dt, pos += vel × dt. Velocity clamping at 50 m/s for stability. Post-integration containment kernel enforces global system bounds.

---

## Bugs

### BUG 1 — Zigzag velocity cap defeats separation physics [CRITICAL]

**Lines 2667–2676.** The code caps upward air velocity at `v_zone_max = v_air_zigzag × zigzag_velocity_ratio_zone`:

```python
v_zone_max = v_air_zigzag * zigzag_velocity_ratio_zone
if v_air[1] > v_zone_max:
    v_air = wp.vec3(v_air[0], v_zone_max, v_air[2])
```

The comment explains this was added because "high bulk/throat velocities lift all particles upward and prevent coarse classification." This reveals a fundamental misunderstanding of zigzag separation. The throat velocity (v_bulk × channel_width/throat_width) is real — particles physically experience this high velocity at the constriction between deflector plates. Capping it to zone velocity (~0.3 × v_bulk) means particles never see the throat acceleration that drives the separation mechanism.

The comment also states "zone velocity is the time-averaged effective velocity experienced by particles as they bounce between deflector plates." This is not how zigzag classifiers work. Particles experience a spatially varying velocity field (throat → zone → transport), and their trajectory through this field determines whether they rise or fall. Separation occurs naturally from force balance: particles with v_t < v_local rise, particles with v_t > v_local fall.

**Fix:** Remove the velocity cap entirely. If all particles are carried upward, the problem is elsewhere — likely the air velocity is too high for the desired d50, or the deflector plate geometry needs adjustment. The correct remedy is to tune air flow rate or plate geometry, not to artificially suppress the velocity field.

### BUG 2 — Zigzag separation logic is purely geometric [HIGH]

**Lines 2770–2778.** The zigzag separation decision checks only particle position:

```python
if pos[1] >= zigzag_fines_outlet_y - particle_radius * 2.0:
    zone = 22  # Fines
elif pos[1] <= zigzag_coarse_outlet_y + particle_radius * 2.0:
    zone = 30  # Coarse
```

A heavy particle momentarily pushed to the top by turbulence or initial momentum gets classified as fines, even if its terminal velocity is 10× the air velocity. Compare with the duct venturi→zigzag transition (zone 10, lines 2620–2633) which correctly checks terminal velocity before routing to coarse.

**Fix:** Add terminal velocity check at the fines outlet. A particle should only transition to fines (zone 22) if both (1) it has reached the fines outlet geometry AND (2) its terminal velocity is below the local air velocity (v_t < v_air_zigzag × safety_factor). Otherwise it will fall back and should remain in zigzag stages.

### BUG 3 — Wheel separation ignores radial position [HIGH]

**Lines 2858–2869.** The wheel classifier routes particles to coarse only when they are physically at or beyond the wheel rim AND force_ratio > 1.0:

```python
if r <= wheel_hub_radius:
    zone = 35  # FINES (inside hub)
elif r >= wheel_radius and force_ratio > 1.0:
    zone = 36  # COARSE (at rim with F_c > F_d)
```

A particle at r = 0.9 × wheel_radius with force_ratio = 2.0 (clearly coarse) won't be classified because it hasn't reached the rim geometrically. It circulates in a "dead zone" between hub and rim indefinitely. While centrifugal force should push it outward eventually, the kernel doesn't enforce this trajectory — it just waits for geometric boundary crossing.

**Fix:** Use force ratio at any radius within the wheel annulus. If F_c/F_d > 1 at current r, the particle is on an outward trajectory → route to coarse hopper. If F_c/F_d < 1, particle is on an inward trajectory → allow it to continue toward hub. This matches the real physics: the force ratio determines trajectory direction, not geometric position.

### BUG 4 — Cyclone separation is purely geometric [MEDIUM]

**Lines 3207–3218.** Cyclone collection checks only position:

```python
if at_wall and below_cylinder:
    zone = 55  # Collected in primary dust outlet
elif in_core and above_vf:
    zone = 51  # Move to secondary cyclone
```

Any particle touching the wall below the cylinder is collected, regardless of size. Any particle in the core above the vortex finder goes to the secondary, regardless of size. This defeats staged cyclone separation: a 5 µm particle that momentarily contacts the wall (due to turbulence) gets collected by the primary cyclone (design d50 ~40 µm) even though it should pass through with 99%+ probability.

Real cyclones have grade efficiency curves: η(d) = 1 - exp(-0.693 × (d/d50)^n). Small particles touching the wall get re-entrained; large particles stick and spiral down.

**Fix:** Add a Stokes number or force ratio check. At minimum, particles at the wall should only be collected if their aerodynamic diameter exceeds some fraction of the stage d50 (e.g., d > 0.5 × d50). Better: implement the grade efficiency curve as a probabilistic collection check each timestep a particle contacts the wall.

### BUG 5 — Venturi particle entry velocity ignored [MEDIUM]

Venturi kernel computes air velocity from continuity but initializes particles with hardcoded `initial_velocity` from config (default 0.5 m/s upward). The feed system delivers particles at specific velocities (deagglomerator exit, gravity drop), but this information is not used.

If feed delivers particles at 5 m/s downward (gravity drop from hopper) but the venturi kernel assumes 0.5 m/s upward, particles will have wrong initial momentum and wrong entrainment behavior. The `reinitialize_from_particles()` method does compute a feed chute terminal velocity (0.05 m/s along inlet direction), but this is only used for recirculation passes, not the initial pass.

**Fix:** Accept particle entry velocity from `feed_result` or `transfer_data` in the initial `initialize_particles()` path. Use actual feed system exit velocity for zone 0 initialization.

### BUG 6 — Continuous feeding partially implemented [MEDIUM]

The `step()` method (lines 5670–5689) does implement continuous particle activation with a feed accumulator, and `reinitialize_from_particles()` correctly sets up continuous mode for recirculation passes. However, the initial `initialize_particles()` method (line 5304) and `initialize_whole_flour_population()` (line 5564) always activate all particles at t=0 regardless of the `continuous_feeding` config flag.

This means continuous feeding works for recirculation passes (pass 2+) but not for the initial pass. The first pass always runs in batch mode.

**Fix:** Apply the same continuous activation logic in `initialize_particles()` and `initialize_whole_flour_population()`. Set `is_active = 0` for all particles, `particles_active = 0`, and let `step()` activate them at `particle_feed_rate`.

### BUG 7 — Deflector plate edge collision incomplete [LOW]

**Lines 2698–2768.** Collision detection checks plate surface overlap but not the plate tip edge. Particles can pass through the gap between plate tip and opposite wall without collision or deflection. This creates a minor "leak" path for particles with specific horizontal trajectories at plate tip height.

**Fix:** Add edge collision check at the plate tip point. If particle center is within particle_radius of the tip and moving toward it, apply collision normal.

### BUG 8 — Bag filter instant collection [LOW]

**Lines 3461–3467.** All particles entering the bag filter are instantly collected (zone 70 → 75). No physics simulation inside. This is reasonable for >99.9% collection efficiency, but it means bag filter statistics are trivially complete — every particle that reaches the bag filter is counted as collected, with no size-dependent efficiency.

**Fix (optional):** Add simple probabilistic collection: P_collect = 1 - exp(-k × d/v_face). Particles with random() > P_collect pass to clean air (zone 80). This would allow the simulation to predict emissions for regulatory compliance estimates.

---

## Design Issues (Not Bugs)

### ISSUE 1 — No velocity continuity between zones

Each zone computes its own air velocity field independently. For example, air velocity at the zigzag fines outlet is set to `v_air_cyclone_inlet * 0.8` (line 2975) — a hardcoded factor, not derived from zigzag outlet velocity or duct continuity. If you change zigzag geometry or air flow rate, connecting duct velocities don't update consistently. Particles experience velocity discontinuities at zone boundaries.

### ISSUE 2 — Uniform turbulent intensity everywhere

`turbulent_intensity` (default 0.15) is applied identically in zigzag and cyclones. Real turbulence characteristics differ significantly: zigzag recirculation zones have high intensity (0.2–0.3), cyclones have moderate swirling turbulence (0.1–0.15), ducts have low turbulence (0.05–0.1). Using the same intensity everywhere over-predicts turbulence in ducts and under-predicts it in zigzag separation zones.

### ISSUE 3 — Hash grid allocated but unused

`_setup_hash_grid` (line 3860) sets up particle neighbor search infrastructure but no kernel uses it. Particles are treated as non-interacting. For dilute-phase flow (μ < 2.0) this is acceptable, but the allocated GPU memory is wasted.

### ISSUE 4 — Wheel volumetric flow origin unclear

`wheel_volumetric_flow` is used in `compute_wheel_radial_velocity` but its derivation path from total air flow rate to wheel-specific flow is not explicit in `_compute_derived_parameters`. It should account for bypass, leakage, and flow split between wheel and housing.

---

## Multi-Pass Recirculation

The recirculation API (`extract_collected_particles` → `reinitialize_from_particles`) is well-designed:

- Extracts particles from any combination of collection zones (cy1, cy2, cy3, wheel_coarse, zigzag_coarse, bagfilter)
- Applies venturi attrition: d_new = d_min + (1-factor)(d_old - d_min), exponential decay toward minimum
- Supports wheel RPM override per pass
- Computes entry velocity from feed system kinetics (gravity chute terminal velocity along inlet direction)
- Uses pass-varying RNG seed for unique particle arrangements
- Y-spread at venturi inlet prevents all particles starting at same cross-section

The `skip_preclassification` flag allows experimental bypass (directly to wheel at zone 34) for testing purposes.

One issue: the RNG seed (`n * 31 + 137`) varies only with particle count, not with pass number. If two consecutive passes have the same particle count, they get identical random positions. A pass counter would be more robust.

---

## Integration Boundaries

### Air System → Classification

`airclass_flow_physics` computes venturi inlet velocity from air flow rate and passes it as `v_air_venturi_inlet`. The venturi K-factor should be included in blower operating point calculation (identified as air system bug #6 in the air system review). The handoff is otherwise clean.

### Feed System → Classification

The feed system delivers particles with positions, velocities, diameters, and masses via `inject_particles_from_feed()`. **CRITICAL**: the feed system exports visual-scale diameters (300× physical, e.g., 15mm instead of 50µm) without rescaling. Classification expects physical-scale diameters. This was identified in the feed-classification integration audit and remains the most critical cross-module bug.

### Classification → Cyclones (internal)

Particles transition from zigzag fines (zone 22) or wheel fines (zone 35) to cyclone inlet (zone 50). The velocity field should be continuous but uses hardcoded factors at the boundary (design issue 1).

---

## Geometry Assembly (classification.py)

The `ClassificationSystemAssembly` is clean and well-organized. Key points:

- Three distinct coarse collection points: dropout hopper (venturi→zigzag transition), zigzag coarse outlet (pre-classifier reject), wheel coarse outlet (centrifugal reject). Each with rotary airlock.
- Staged cyclone design with explicit d50 targets: primary ~40 µm, secondary ~20 µm, tertiary ~10 µm.
- `validate_system_configuration()` checks zigzag operating conditions and cyclone staging against specified air flow and particle size range. Can accept separate flows for zigzag and cyclones when bypass is used.
- Port-based connection system with world-position transforms for assembly positioning.
- Comprehensive `print_summary()` with all dimensions, port positions, and flow path documentation.

---

## Structural Assessment

The classification flow physics module is well-designed overall. The zone-based architecture is clear and extensible, geometry extraction has zero hardcoding, and the Warp kernels implement correct physics formulas. The docstrings and comments are thorough and demonstrate genuine understanding of the separation physics.

The main systematic issue is that separation decisions are too geometric and not enough physics-based. The zigzag, wheel, and cyclone separators all route particles based primarily on whether they've reached a geometric boundary, rather than checking whether the local force balance supports that classification. This makes the simulation work for visualization but undermines quantitative accuracy — particles that shouldn't be at a boundary (due to turbulent fluctuation or initial momentum) get misclassified.

---

## Recommended Fix Priority

**Priority 1 — Blocking for quantitative accuracy:**
1. Fix feed system diameter scaling (300× visual→physical)
2. Remove zigzag velocity cap (Bug 1)
3. Add terminal velocity check to zigzag separation (Bug 2)

**Priority 2 — High impact:**
4. Fix wheel classifier force-ratio-at-any-radius (Bug 3)
5. Add cyclone grade efficiency check (Bug 4)
6. Use feed system entry velocity in venturi (Bug 5)

**Priority 3 — Medium impact:**
7. Enable continuous feeding for initial pass (Bug 6)
8. Add zone velocity coupling (Design Issue 1)
9. Add zone-specific turbulence intensities (Design Issue 2)

**Priority 4 — Low impact / quality-of-life:**
10. Deflector plate edge collision (Bug 7)
11. Bag filter collection probability (Bug 8)
12. Free unused hash grid memory (Design Issue 3)
