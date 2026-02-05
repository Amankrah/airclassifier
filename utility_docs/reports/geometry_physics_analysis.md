# Geometry and Physics Analysis: Real-World Practicality

## Executive Summary

This document analyzes the geometric and physics implementations of the air classifier system to determine their real-world practicality and correctness. The analysis covers:

1. Zigzag Classifier Geometry
2. Cyclone Assembly Geometry
3. Bag Filter Geometry
4. Multi-Cyclone System
5. Complete System Integration
6. Physics Implementation Accuracy

**Last Updated:** Analysis based on current codebase review

---

## 1. Zigzag Classifier Analysis

### Current Implementation (`zigzag_classifier.py`)

**Geometry Description:**
- Alternating left-right horizontal offsets per stage
- Channel width × depth cross-section (rectangular)
- Interior zigzag angle (typically 120°)
- Feed inlet at middle stage (configurable)
- Air inlet at bottom, fines outlet at top, coarse outlet at bottom
- Simple rectangular feed tube extending from channel wall

**Physics (in `classification_flow_physics.py`):**
- Terminal velocity-based separation (Stokes drag)
- Schiller-Naumann drag correlation for intermediate Reynolds numbers
- Cut size (d50) calculation using Stokes law:
  ```
  d50 = √(18 × μ × v_air / (g × (ρ_p - ρ_f)))
  ```
- Turbulent dispersion implemented with configurable intensity (15% default)
- Stage-dependent velocity variation (±30% sinusoidal)

### Real-World Assessment

#### Geometry Issues:

| Issue | Current State | Real-World Design |
|-------|---------------|-------------------|
| **Channel Shape** | Simple rectangular with alternating X-offset | Industrial zigzags use angled deflector plates creating triangular/trapezoidal zones |
| **Stage Geometry** | 2D offset pattern - stages just shift left/right | Real zigzags have shelf-like deflectors protruding into the channel creating distinct separation zones |
| **Air Distribution** | Assumed uniform flow through rectangular channel | Real systems have perforated plates, distribution manifolds for uniform cross-flow |
| **Feed Injection** | Simple rectangular port on side | Real systems use dispersing nozzles, often with secondary air injection |
| **Wall Interaction** | Smooth walls assumed | Real walls often roughened or have anti-adhesion coatings |

#### Physics Strengths:

| Feature | Current State | Assessment |
|---------|---------------|------------|
| **Drag Model** | Schiller-Naumann correlation | ✓ Correct for intermediate Re (0.1 < Re < 1000) |
| **Turbulent Dispersion** | ✓ Implemented (`compute_turbulent_dispersion()`) | Stochastic velocity fluctuations added in zigzag stages |
| **Buoyancy** | ✓ Archimedes correction (ρ_p - ρ_f) | Correct physics |
| **Zone Tracking** | ✓ Per-zone particle states | Excellent for debugging and analysis |
| **d50 Calculation** | ✓ Stokes law | Correct for small particles |

#### Physics Weaknesses:

| Issue | Current State | Impact |
|-------|---------------|--------|
| **Particle-Particle Collisions** | Hash grid infrastructure exists but NOT actively used in classification kernel | HIGH at realistic loading (~5-10% v/v) |
| **Wall Effects** | Simple inelastic restitution/friction | Real wall bounce, sliding, agglomeration critical |
| **Re-entrainment** | Not modeled | Settled particles can be re-entrained by turbulence |

#### Missing Critical Geometry Features:

1. **Deflector Plates**: Real zigzag classifiers have triangular or trapezoidal deflector plates creating turbulent mixing zones - NOT simple channel offsets
2. **Multiple Feed Points**: Industrial units often have distributed feed
3. **Secondary Air**: Many designs inject secondary air at stages
4. **Variable Geometry**: Industrial units have adjustable deflector angles

### Recommended Geometry Improvements:

```
CURRENT (Simplified):                 REAL-WORLD (Deflector-based):

   ____                                  ____
  |    |  <- simple offset             |    \  <- deflector plate
  |____|                               |_____\
      ____                                  ____
     |    | <- alternating                 /    |
     |____|                               /_____| <- creates turbulent zones
```

**Specific Changes Needed:**
1. Replace simple offset with angled deflector geometry
2. Model the recirculation zones behind deflectors
3. Add proper inlet plenum with distribution
4. Consider non-uniform velocity profiles

---

## 2. Cyclone Assembly Analysis

### Current Implementation (`cyclone.py`)

**Geometry Description:**
- Standard Stairmand/Lapple proportions:
  - Cylinder height: 1.5D
  - Cone height: 2.5D
  - Inlet width: 0.25D
  - Inlet height: 0.5D
  - Vortex finder: 0.5D
- Tangential inlet with configurable angular position
- Dust outlet at cone apex
- Complete component integration (body, inlet, vortex finder, dust outlet, overflow)

**Physics (in `classification_flow_physics.py`):**
- Lapple equation for d50:
  ```
  d50 = √(9 × μ × W / (π × N × v_i × (ρ_p - ρ_g)))
  ```
- Pressure drop estimation using velocity head
- Collection efficiency based on d50 ratio
- 3D velocity field: tangential + radial + axial components
- Centrifugal acceleration computed

### Real-World Assessment

#### Geometry - MOSTLY CORRECT:

| Aspect | Current | Assessment |
|--------|---------|------------|
| **Proportions** | Standard ratios (Stairmand) | ✓ Correct for high-efficiency design |
| **Tangential Inlet** | Rectangular, tangential entry | ✓ Correct |
| **Vortex Finder** | Cylindrical, correct insertion | ✓ Correct |
| **Cone Angle** | Derived from height/tip diameter | ✓ Reasonable |
| **Inlet Angular Position** | Configurable (0, π, etc.) | ✓ Good for series arrangement |

#### Geometry Issues:

| Issue | Current State | Real-World Improvement |
|-------|---------------|------------------------|
| **Inlet Scroll** | Simple rectangular duct | Real systems use scroll inlets for smoother entry |
| **Vortex Breaker** | Not included | Dust outlet should have vortex breaker cone |
| **Roof** | Flat assumed | May need conical roof for gas flow |
| **Wall Thickness** | Uniform | Wear-prone areas need thicker walls or liners |

#### Physics Issues:

| Issue | Current State | Real-World Consideration |
|-------|---------------|--------------------------|
| **Tangential Velocity** | ✓ Rankine vortex model (solid body + free vortex) | Correct flow pattern |
| **Radial Velocity** | ✓ Inward flow modeled | Correct for drag balance |
| **Axial Velocity** | ✓ Outer down, inner up (double helix) | Correct flow pattern |
| **Centrifugal Force** | ✓ `compute_centrifugal_acceleration()` | Correct physics |
| **Swirl Decay** | NOT modeled | Inner vortex dissipates along cone - affects trajectories |
| **Short-Circuiting** | Not considered | Direct path inlet→VF reduces efficiency |
| **Re-entrainment** | Simple model | Dust pickup from cone walls is major loss |
| **Secondary Flow** | Not modeled | Flow patterns near vortex finder lip critical |

### Recommended Improvements:

1. **Add Vortex Breaker**: Critical for preventing re-entrainment at dust outlet
2. **Model Swirl Decay**: Velocity reduces along cone - affects particle trajectories
3. **Short-Circuit Modeling**: Some gas bypasses separation zone
4. **Consider High-Efficiency Variants**:
   - Extended vortex finder
   - Double-scroll inlet
   - Helical roof

---

## 3. Bag Filter Analysis

### Current Implementation (`bag_filter.py`)

**Geometry Description:**
- Full rectangular housing with hopper
- Filter bags hanging from tube sheet (configurable array)
- Dirty air section, tube sheet, clean air plenum
- **Pulse-jet cleaning system geometry** (NEW - fully modeled):
  - Compressed air tank (reservoir) mounted on top
  - Main feed pipe from tank into housing
  - Header pipes across each bag row
  - Blow tubes extending down above each bag
  - Conical nozzles directing pulses into bags

**Physics:**
- Air-to-cloth ratio calculation
- Collection efficiency > 99.9% for particles > 1 μm (design basis)
- Particle settling to hopper

### Real-World Assessment

#### Geometry Strengths:

| Feature | Implementation | Assessment |
|---------|----------------|------------|
| **Housing** | Full rectangular box with hopper | ✓ Correct industrial design |
| **Filter Bags** | Cylindrical, hanging from tube sheet | ✓ Standard configuration |
| **Tube Sheet** | Proper separation of dirty/clean air | ✓ Critical feature present |
| **Hopper** | Pyramidal frustum with outlet | ✓ Proper dust collection |
| **Pulse-Jet System** | ✓ Tank, headers, blow tubes, nozzles | **EXCELLENT** - Complete geometry |
| **Ports** | Dirty inlet, clean outlet, dust outlet | ✓ Complete connection set |

#### Geometry Issues:

| Issue | Current State | Real-World Improvement |
|-------|---------------|------------------------|
| **Bag Cages** | Not modeled | Real bags have internal wire cages |
| **Venturi Nozzles** | Simple conical | Industrial systems use specific venturi designs |
| **Cleaning Sequence** | Not modeled | Real systems clean rows sequentially |
| **Filter Media** | Simplified as cylinder | Real bags have specific weave/felt properties |

#### Physics Simplifications:

| Issue | Current State | Impact |
|-------|---------------|--------|
| **Filter Cake** | Not modeled | Cake buildup affects pressure drop and efficiency |
| **Pressure Drop** | Not calculated | Critical for blower sizing |
| **Re-entrainment** | Not modeled during pulse | Some dust re-entrains during cleaning |
| **Particle Interception** | Simplified | Real mechanisms include inertia, interception, diffusion |

---

## 4. Multi-Cyclone System Analysis

### Current Implementation (`multi_cyclone.py`)

**Configuration:**
- Series or parallel arrangement
- 2-3 stages with decreasing diameter
- Internal connecting ductwork (elbows, transitions) - **FULLY MODELED**
- Design d50 targets: 40μm → 20μm → 10μm (typical)
- Uses `CycloneAssembly` for each stage (proper components)

**Connecting Ductwork (Series Arrangement):**
- Elbow 1: VF (+Y) → horizontal (+X)
- Horizontal duct toward next cyclone
- Elbow 2: +X → down (-Y)
- Vertical duct
- Elbow 3: -Y → +X toward inlet
- Round-to-rectangular transition into cyclone inlet

### Real-World Assessment

#### Design Strengths:

| Feature | Implementation | Assessment |
|---------|----------------|------------|
| **Series arrangement** | Correct for progressive separation | ✓ Standard for protein/starch |
| **Decreasing diameter** | Reduces d50 at each stage | ✓ Correct physics |
| **Duct connections** | Elbows and transitions modeled | ✓ **EXCELLENT** - Proper geometry |
| **Inlet Orientation** | All inlets face upstream (-X) for series flow | ✓ Correct flow path |
| **Performance Calculations** | d50, pressure drop, efficiency | ✓ Engineering-accurate |

#### Issues:

| Issue | Current State | Real-World Design |
|-------|---------------|-------------------|
| **Pressure Balance** | Not actively calculated through complete path | Series cyclones need careful pressure balancing |
| **Flow Distribution** | Assumed even | May need orifice plates or dampers |
| **Dust Sealing** | Open outlets | Each dust outlet needs rotary valve or double-dump |
| **Re-entrainment Path** | Not blocked | Downstream cyclone can pull dust from upstream |
| **Bypass** | Not included | Need bypass for startup/shutdown |

#### Missing Real-World Components:

1. **Isolation Valves**: Between stages for maintenance
2. **Pressure Taps**: Monitoring each stage ΔP
3. **Sight Glasses**: Visual inspection ports
4. **Dust Level Sensors**: In collection hoppers
5. **Air Locks**: Rotary valves at each dust outlet

---

## 5. Complete System Integration Analysis

### Current Implementation (`complete_system.py` + `classification_flow_physics.py`)

**System Architecture:**
- **Phase 1**: Classification System (Venturi + Zigzag + Cyclones + Bag Filter)
- **Phase 2**: Feed System (Hopper + Airlock + Screw Feeder + Deagglomerator)
- **Phase 3**: Air System (Blower + Filter + Damper)
- **Phase 4**: Ductwork (Connecting ducts, elbows, transitions)
- **Phase 5**: Exhaust (Silencer + Stack)

**Physics Simulation Flow:**
```
VENTURI (zones 0-2)     Particle entrainment into airstream
       |
DUCT_V_Z (zone 10)      Vertical duct to zigzag (round-to-rect transition)
       |
ZIGZAG (zones 20-23)    Primary separation by terminal velocity
       |__________________
       |                  |
FINES (+Y)          COARSE (-Y)
zone 22              zone 30 (collected starch)
       |
ELBOW (zone 40)         90° turn from vertical to horizontal
       |
DUCT (zone 41)          Horizontal to cyclones (round-to-rect)
       |
CYCLONES (zones 50-52)  Staged centrifugal separation
|     |     |
DUST  DUST  DUST         (zones 55-57, collected)
             |
ELBOW (zone 60)
       |
DUCT (zone 61)          To bag filter (expansion)
       |
BAG FILTER (zone 70)    Final fines capture
|           |
DUST        CLEAN AIR
zone 75     zone 80
```

### Real-World Assessment

#### Strengths:

| Feature | Implementation | Assessment |
|---------|----------------|------------|
| **Modular Design** | Separate subsystems with clear interfaces | ✓ Excellent for maintenance and testing |
| **Port-Based Connections** | Standardized connection ports for all components | ✓ Industry-standard approach |
| **Ductwork Geometry** | Realistic elbows, transitions, and routing | ✓ Proper flow path modeling |
| **Zone Tracking** | Complete particle tracking through all zones | ✓ Excellent for debugging |
| **Coordinate System** | Consistent Y-up coordinate system throughout | ✓ Prevents confusion |
| **Geometry Extraction** | `extract_geometry()` pulls real dimensions | ✓ No magic numbers in physics |

#### Geometry Issues:

| Issue | Current State | Real-World Improvement |
|-------|---------------|------------------------|
| **Duct Flanges** | Modeled but not detailed | Real systems need gasket details, bolt patterns |
| **Support Structure** | Optional, basic frame | Real systems need structural analysis, vibration isolation |
| **Expansion Joints** | Not included | Thermal expansion requires flexible connections |
| **Access Ports** | Not modeled | Maintenance requires inspection ports, manholes |
| **Insulation** | Not included | High-temperature systems need thermal insulation |

#### Flow Path Issues:

| Issue | Current State | Real-World Consideration |
|-------|---------------|--------------------------|
| **Pressure Losses** | Not calculated through complete ductwork | Elbows and transitions add significant ΔP |
| **Flow Distribution** | Assumed uniform | Real systems may need flow straighteners |
| **Leakage** | Perfect seals assumed | Flange connections have small leakage |
| **Flow Reversal** | Not prevented | Need check valves or backflow prevention |
| **Startup/Shutdown** | Not modeled | Bypass valves needed for safe operation |

---

## 6. Physics Implementation Assessment

### Classification Flow Physics (`classification_flow_physics.py`)

**Implemented Physics - COMPREHENSIVE:**

| Feature | Implementation | Assessment |
|---------|----------------|------------|
| **Drag Model** | Schiller-Naumann correlation | ✓ Correct for intermediate Re (0.1 < Re < 1000) |
| **Haider-Levenspiel** | Available as `@wp.func` for non-spherical particles | ✓ Available but not used in main kernel |
| **Buoyancy** | Archimedes correction (ρ_p - ρ_f) | ✓ Correct |
| **Zone Tracking** | Complete per-zone states (venturi, zigzag, cyclone, bag) | ✓ Excellent for debugging and analysis |
| **Material Properties** | From ParticleMaterial/ParticlePhysicsConfig | ✓ Flexible, realistic food powder data |
| **Turbulent Dispersion** | `compute_turbulent_dispersion()` with configurable intensity | ✓ **IMPLEMENTED** (15% default) |
| **Centrifugal Forces** | Full 3D cyclone velocity field (tangential + radial + axial) | ✓ Properly modeled |
| **Venturi Physics** | Bernoulli-based air velocity with continuity | ✓ Correct pressure drop and entrainment |
| **Wall Collisions** | Inelastic with restitution and friction | ✓ `reflect_velocity_inelastic()` |
| **Terminal Velocity** | `compute_terminal_velocity()` with regime detection | ✓ Stokes to intermediate |
| **Stokes Number** | `compute_stokes_number()` for inertia vs drag | ✓ Available for analysis |
| **Cut Size** | `compute_cut_size_zigzag()` and `compute_cut_size_cyclone()` | ✓ Lapple equation |
| **Grade Efficiency** | `compute_separation_probability()` with sharpness parameter | ✓ Rosin-Rammler model |

### Weaknesses/Missing Physics:

| Missing Feature | Impact | Priority | Status |
|-----------------|--------|----------|--------|
| **Particle-Particle Collisions** | HIGH at realistic loading (~5-10% v/v) | HIGH | Hash grid infrastructure EXISTS (`ParticleCollisionHandler`) but NOT used in classification kernel |
| **Agglomeration** | Fine powders clump, changes effective d50 | HIGH | Not implemented |
| **Electrostatics** | Tribocharging in dry systems | MEDIUM | Not implemented |
| **Humidity Effects** | Particle adhesion changes | MEDIUM | Not implemented |
| **Non-Spherical Drag in Kernel** | Shape factors for real particles | MEDIUM | Haider-Levenspiel function exists but main kernel uses Schiller-Naumann |
| **Wall Roughness** | Affects collision dynamics | LOW | Not implemented (smooth walls assumed) |
| **Two-Way Coupling** | Particles affect fluid | LOW (dilute) | Not implemented (one-way coupling) |
| **Swirl Decay in Cyclones** | Velocity reduces along cone | MEDIUM | Not modeled (constant tangential velocity assumed) |
| **Re-entrainment** | Dust pickup from walls | MEDIUM | Not modeled |

### Hash Grid / Collision Infrastructure Status:

**File:** `src/airclassifier/particles/interactions/particle_particle.py`

| Component | Status |
|-----------|--------|
| `ParticleCollisionHandler` class | ✓ EXISTS (Lines 208-309) |
| `wp.HashGrid` creation | ✓ EXISTS (128³ grid) |
| `detect_particle_collisions` kernel | ✓ EXISTS (Lines 83-148) |
| `separate_overlapping_particles` kernel | ✓ EXISTS (Lines 150-206) |
| `particle_particle_collision` function | ✓ EXISTS (Lines 26-81) |
| Default enabled | ✗ NO (`enable_collisions: bool = False`) |
| Integration in classification_flow_physics | ✗ NOT ACTIVE (grid created but collisions not computed) |

---

## 7. Practical Recommendations

### Immediate Improvements (High Impact, Lower Effort)

1. **Zigzag Deflectors**:
   - Replace simple offset geometry with triangular deflector plates
   - This is the core separation mechanism - current geometry won't separate well

2. **Vortex Breakers**:
   - Add to all cyclone dust outlets
   - Simple cone geometry, prevents major efficiency loss

3. **Activate Particle-Particle Collisions**:
   - Hash grid infrastructure already exists in `ParticleCollisionHandler`
   - Need to integrate into `classification_physics_kernel` or call separately
   - At realistic loadings (~5-10% v/v), this is essential for accuracy
   - Enable with `ParticleCollisionParams(enable_collisions=True)`

### Medium-Term Improvements

4. **Use Non-Spherical Drag in Main Kernel**:
   - `drag_coefficient_haider_levenspiel()` function exists
   - Need to call it in `compute_drag_acceleration()` based on particle sphericity
   - Protein/starch particles are not spheres - sphericity ~0.75
   - Expected impact: 10-30% change in drag at high Re

5. **Agglomeration Model**:
   - Fine particles (<20μm) tend to clump
   - Affects effective d50 significantly
   - Can use simplified probability-based model

6. **Swirl Decay in Cyclones**:
   - Model velocity reduction along cone length
   - Affects particle trajectories and collection efficiency
   - Simple linear or power-law decay

### System-Level Additions

7. **Dust Sealing**:
   - Model rotary valves at dust outlets
   - Critical for preventing bypass flow

8. **Pressure Drop Calculation**:
   - Calculate pressure losses through complete ductwork system
   - Include elbows, transitions, and length losses
   - Critical for blower sizing and system balance
   - Series cyclones must be balanced

9. **Complete System Flow Validation**:
   - Verify mass balance: Feed = Coarse + Fines + Carryover
   - Check air flow continuity through all connections
   - Validate particle transfer from feed to classification system

---

## 8. Comparison: Current vs. Industrial Reality

### Zigzag Classifier

| Metric | Current Sim | Industrial Reality |
|--------|-------------|-------------------|
| Geometry Accuracy | **LOW** (~40%) | Needs deflector plates |
| Physics Accuracy | **GOOD** (~80%) | Turbulent dispersion ✓, but P-P collisions missing |
| Separation Prediction | Moderate accuracy | Real efficiency may be lower due to geometry simplification |

### Cyclone System

| Metric | Current Sim | Industrial Reality |
|--------|-------------|-------------------|
| Geometry Accuracy | **HIGH** (~85%) | Good proportions, missing vortex breaker |
| Physics Accuracy | **GOOD** (~75%) | Full 3D velocity field, missing swirl decay |
| d50 Prediction | Good (Lapple) | Within engineering accuracy |

### Bag Filter

| Metric | Current Sim | Industrial Reality |
|--------|-------------|-------------------|
| Geometry Accuracy | **EXCELLENT** (~90%) | Full pulse-jet system geometry |
| Physics Accuracy | MEDIUM (~60%) | Simplified collection model |
| Collection Prediction | Good for > 1 μm | Missing cake buildup effects |

### Multi-Cyclone System

| Metric | Current Sim | Industrial Reality |
|--------|-------------|-------------------|
| Geometry Accuracy | **EXCELLENT** (~90%) | Full ductwork with elbows/transitions |
| Flow Path | ✓ Correct | Good series arrangement |
| Missing | Dust sealing, pressure balance | Need rotary valves |

### Overall System

| Metric | Current Sim | Industrial Reality |
|--------|-------------|-------------------|
| Flow Path | ✓ Complete | All zones tracked |
| System Integration | ✓ Excellent | Proper port-based connections |
| Ductwork | ✓ Realistic | Elbows, transitions, expansions |
| Mass Balance | Tracked by zone | Need better validation |
| Collection Points | ✓ Defined | Need dust sealing (rotary valves) |
| Pressure Drop | NOT calculated | Critical for blower sizing |

---

## 9. Priority Action Summary

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **CRITICAL** | Redesign zigzag geometry with proper deflector plates | HIGH | Fixes core separation mechanism |
| **HIGH** | Activate particle-particle collisions | LOW | Infrastructure exists - just enable |
| **HIGH** | Add vortex breakers to cyclones | LOW | Simple geometry addition |
| **HIGH** | Calculate pressure drop through ductwork | MEDIUM | Critical for blower sizing |
| **MEDIUM** | Use Haider-Levenspiel drag in main kernel | LOW | Function exists - change drag call |
| **MEDIUM** | Model swirl decay in cyclones | MEDIUM | Linear decay factor |
| **MEDIUM** | Add rotary valves at dust outlets | MEDIUM | Prevent re-entrainment |

### Expected Outcome After Improvements:

With the recommended changes, the simulation should achieve:
- **Zigzag**: Within 20% of experimental d50 (after deflector redesign)
- **Cyclone**: Within 15% of Lapple correlation (already good)
- **Complete System**: Realistic mass balance and pressure drop predictions
- **Physics**: Suitable for engineering design decisions
- **Ductwork**: Accurate pressure loss calculations for blower sizing

---

## Appendix A: Reference Dimensions

### Typical Industrial Zigzag Classifier:
- Channel width: 100-300 mm
- Number of stages: 4-8
- Deflector angle: 30-60° from horizontal
- Air velocity: 1-5 m/s
- Feed loading: 0.1-1.0 kg solids/kg air

### Typical Industrial Cyclone (High-Efficiency):
- Cylinder diameter: 150-500 mm (pilot), 0.5-2m (production)
- Aspect ratio (H/D): 3-4
- Inlet velocity: 15-25 m/s
- Pressure drop: 500-2500 Pa
- d50 range: 2-20 μm (depending on size)

### Typical Multi-Cyclone Configuration:
- Primary: d50 = 30-50 μm (coarse collection)
- Secondary: d50 = 15-25 μm (medium fraction)
- Tertiary: d50 = 5-15 μm (fine collection)

---

## Appendix B: Code Reference Summary

### Key Files:
- **Zigzag Geometry**: `src/airclassifier/geometry/components/zigzag_classifier.py`
- **Cyclone Assembly**: `src/airclassifier/geometry/assembly/cyclone.py`
- **Bag Filter**: `src/airclassifier/geometry/components/bag_filter.py`
- **Multi-Cyclone**: `src/airclassifier/geometry/components/multi_cyclone.py`
- **Physics Simulation**: `src/airclassifier/simulation/classification_flow_physics.py`
- **Particle Collisions**: `src/airclassifier/particles/interactions/particle_particle.py`

### Key Functions:
- `compute_drag_acceleration()` - Main drag calculation (Schiller-Naumann)
- `drag_coefficient_haider_levenspiel()` - Non-spherical drag (available but unused)
- `compute_turbulent_dispersion()` - Zigzag turbulence (✓ active)
- `compute_cyclone_tangential_velocity()` - Cyclone flow field
- `compute_centrifugal_acceleration()` - Cyclone separation force
- `reflect_velocity_inelastic()` - Wall collision response
- `ParticleCollisionHandler` - P-P collision handler (exists, not integrated)

---

## Appendix C: Literature References

1. Shapiro & Galperin (2005) - "Air classification of solid particles: a review"
2. Hoffmann & Stein (2008) - "Gas Cyclones and Swirl Tubes"
3. Lapple (1951) - "Processes use many collector types"
4. Stairmand (1951) - "The design and performance of cyclone separators"
5. Rhodes (2008) - "Introduction to Particle Technology"
6. Haider & Levenspiel (1989) - "Drag coefficient and terminal velocity of spherical and non-spherical particles"
