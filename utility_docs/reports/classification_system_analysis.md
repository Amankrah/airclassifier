# Classification System Simulation Analysis Report

## Executive Summary

**⛔ CRITICAL: The classification system is NOT working as intended.**

The system completely fails to separate protein from starch. Instead of achieving fractionation, nearly all material (99.8%) ends up in a single collection point (Cyclone 1).

| Metric | Status | Notes |
|--------|--------|-------|
| Protein Recovery | ❌ FAILED | 0% recovered (target: 70-90%) |
| Starch Recovery | ❌ FAILED | 0.2% recovered |
| Zigzag Separation | ❌ BYPASSED | d50 = 695 µm >> particle size |
| Cyclone Separation | ❌ FAILED | All material in Cy1, none in Cy2/Cy3 |
| Material Loss | ✅ None | 0% escaped |

---

## 1. The Core Problem

### What Should Happen (Protein/Starch Separation)

```
Feed (mixed flour)
    │
    ▼
┌─────────────────┐
│    ZIGZAG       │  ← Should separate at d50 ≈ 20-40 µm
│   CLASSIFIER    │
└────┬───────┬────┘
     │       │
   FINES   COARSE
  (small)  (large)
     │       │
     │       └──► STARCH FRACTION (large granules 15-60 µm)
     │
     ▼
┌─────────────────┐
│   CYCLONES      │  ← Should do staged collection
│  (Cy1→Cy2→Cy3)  │
└────┬───────┬────┘
     │       │
   Cy3     Cy1+Cy2
  (finest) (medium)
     │       │
     └──► PROTEIN FRACTION (small particles 10-30 µm)
```

### What Actually Happened

```
Feed (100,000 particles)
    │
    ▼
┌─────────────────┐
│    ZIGZAG       │  ← d50 = 695 µm (WAY too high!)
│   (BYPASSED)    │     All particles < 695 µm pass through
└────┬───────┬────┘
     │       │
   99.85%   0.15%
     │       │
     │       └──► COARSE: 153 particles (only largest fiber)
     │
     ▼
┌─────────────────┐
│   CYCLONE 1     │  ← d50 = 0.8 µm (captures EVERYTHING!)
│   (99.85%)      │
└────────────────┘
     │
     ▼
   NOTHING reaches Cy2, Cy3, or Bag Filter
```

---

## 2. Quantitative Analysis

### Simulation Results

| Collection Point | Expected | Actual | Status |
|-----------------|----------|--------|--------|
| Coarse (Starch) | 55% (starch granules) | 153 (0.15%) | ❌ |
| Cyclone 1 | ~20% (coarse fines) | 99,847 (99.85%) | ❌ |
| Cyclone 2 | ~25% (medium fines) | 0 (0%) | ❌ |
| Cyclone 3 (Protein) | ~25% (fine protein) | 0 (0%) | ❌ |
| Bag Filter | ~5% (ultra-fines) | 0 (0%) | ❌ |
| Escaped | 0% | 0 (0%) | ✅ |

### Why This Happened: The Physics

**Zigzag Cut Size Calculation:**
```
d50 = √(18 × µ × v_air / (g × (ρ_p - ρ_f)))
    = √(18 × 1.82e-5 × 20.46 / (9.81 × (1420 - 1.2)))
    = √(6.72e-3 / 13918)
    = 695 µm
```

**Problem:** Your flour particles are 5-100 µm, but the cut size is 695 µm.
- ALL particles have terminal velocity < air velocity
- ALL particles are carried upward to the fines outlet
- The zigzag acts as a transport duct, not a separator

**Cyclone Cut Size:**
```
d50 = 0.8 µm (calculated from geometry and flow)
```

**Problem:** With d50 = 0.8 µm, the primary cyclone captures virtually everything.
- Particles down to ~1-2 µm are collected with >99% efficiency
- Nothing passes through to Cyclone 2 or Cyclone 3

---

## 3. Root Cause Analysis

### Mismatch Between Systems

| Parameter | Air System Output | Classification Need | Mismatch |
|-----------|------------------|---------------------|----------|
| Flow Rate | 1,768 m³/h | ~36-100 m³/h | **17-49× too high** |
| Zigzag Velocity | 20.5 m/s | ~0.5-2 m/s | **10-40× too high** |
| Cyclone Inlet Velocity | 173.7 m/s | ~15-25 m/s | **7-12× too high** |

### The Air System is Oversized for Classification

Your air system was designed for:
- 3,000 m³/h design flow
- 1,768 m³/h at 2,500 RPM operating point

But protein/starch separation requires:
- For d50 = 35 µm: **Q ≈ 4.5 m³/h** (zigzag velocity ~0.05 m/s)
- For d50 = 100 µm: **Q ≈ 37 m³/h** (zigzag velocity ~0.4 m/s)

**Your air flow is ~50× higher than needed for separation!**

---

## 4. Component-by-Component Assessment

### Venturi Eductor ✅ Working (as transport)

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Inlet velocity | 97.7 m/s | Very high, good entrainment |
| Throat velocity | 390.8 m/s | Extremely high |
| Throat vacuum | 86.2 kPa | Excellent suction |
| Reynolds number | 1,031,320 | Fully turbulent |

The venturi successfully entrains all particles. It's doing its job, but the downstream flow rate is too high for classification.

### Zigzag Classifier ❌ BYPASSED

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Channel size | 120 × 200 mm | Standard pilot scale |
| Stages | 5 | Good for separation |
| Air velocity | 20.46 m/s | **FAR too high** |
| Reynolds number | 202,499 | Turbulent (expected) |
| Cut size (d50) | 695 µm | **Should be 20-40 µm** |

**Result:** Only 153 particles (0.15%) went to coarse outlet — these were likely the very largest fiber particles. Everything else passed straight through.

### Primary Cyclone ❌ OVER-COLLECTING

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Diameter | 300 mm | Appropriate size |
| Inlet velocity | 173.7 m/s | **Extremely high** |
| Tangential velocity | 173.7 m/s | Supersonic-level forces |
| Centrifugal acceleration | 201,133 m/s² (20,503 g) | Extreme |
| Cut size (d50) | 0.8 µm | **Captures everything** |

**Result:** 99,847 particles (99.85%) collected here. Nothing passed through to downstream cyclones.

### Secondary & Tertiary Cyclones ❌ STARVED

| Cyclone | Collected | Reason |
|---------|-----------|--------|
| Secondary (200mm) | 0 | Nothing reached it |
| Tertiary (120mm) | 0 | Nothing reached it |

These cyclones received no material because the primary cyclone captured everything.

### Bag Filter ❌ UNUSED

| Parameter | Collected | Assessment |
|-----------|-----------|------------|
| Particles | 0 | Nothing reached it |
| Expected function | Final fine capture | Unused |

---

## 5. Required Corrections

### Option A: Reduce Air Flow (Recommended)

To achieve proper separation with current geometry:

| Target d50 | Required Q | Required v_zigzag | RPM (estimate) |
|------------|------------|-------------------|----------------|
| 35 µm | 4.5 m³/h | 0.05 m/s | ~8 RPM |
| 50 µm | 9.1 m³/h | 0.11 m/s | ~15 RPM |
| 100 µm | 36.6 m³/h | 0.42 m/s | ~60 RPM |

**Problem:** These flow rates are far below your blower's operating range. The blower is simply too large.

### Option B: Resize Classification System

Keep air flow at ~1,768 m³/h but redesign the classifier:

| Parameter | Current | Needed (approx) |
|-----------|---------|-----------------|
| Zigzag channel area | 240 cm² | ~24,000 cm² (100×) |
| Channel width | 120 mm | ~1,200 mm |
| Channel depth | 200 mm | ~2,000 mm |

**Problem:** This would require an industrial-scale zigzag, not pilot scale.

### Option C: Add Secondary Air Supply (Best Practical Option)

Use the main blower for transport/cyclone operation, but add a separate low-flow air supply for the zigzag:

```
Main Blower (1,768 m³/h) → Venturi → [Bypass valve]
                                         │
                             ┌───────────┴───────────┐
                             │                       │
                        Main flow              Bleed to zigzag
                       (transport)              (~50 m³/h)
                             │                       │
                             └───────────────────────┘
                                         │
                                    Zigzag Classifier
```

### Option D: Different Separation Technology

At these high flow rates, consider:
- **Air table** instead of zigzag
- **Fluidized bed classifier**
- **Turbo classifier** (centrifugal)

These can handle higher flow rates with appropriate cut sizes.

---

## 6. Simulation Performance

Despite the separation failure, the simulation itself ran well:

| Metric | Value | Assessment |
|--------|-------|------------|
| Particles | 100,000 | Large-scale simulation |
| Simulation time | 180 s | Complete processing |
| Wall time | 23.1 s | Fast (7,778 steps/s) |
| Material loss | 0% | Proper boundary handling |
| Steady state | Reached by t=9s | Fast equilibration |

The physics engine and particle tracking are working correctly — the problem is purely a **design mismatch**.

---

## 7. Comparison with Feed and Air Systems

| System | Status | Key Issue |
|--------|--------|-----------|
| Feed System | ⚠️ Partial | 19.3% airlock retention |
| Air System | ✅ Good | Stable, efficient |
| Classification | ❌ Failed | Air flow 50× too high |

**The classification system is the bottleneck** — the other systems work but are feeding into a non-functional separator.

---

## 8. Recommended Action Plan

### Immediate (Before Next Simulation)

1. **Calculate required air flow** for your target cut size:
   ```python
   # For d50 = 35 µm (protein/starch separation)
   Q_required = A_zigzag * v_air
   # where v_air = d50² * (ρ_p - ρ_f) * g / (18 * µ)
   ```

2. **Either:**
   - Reduce blower RPM drastically (likely impractical)
   - Add flow control bypass
   - Redesign zigzag channel dimensions

### Short-term

3. **Run parametric study** varying air flow to find operating window:
   ```
   --flow 10 --flow 50 --flow 100 --flow 500 m³/h
   ```

4. **Verify cyclone staging** at lower flow rates:
   - Primary should collect coarse fines
   - Secondary should collect medium
   - Tertiary should collect protein-rich fines

### Medium-term

5. **Consider two-stage operation:**
   - Stage 1: High flow for transport and fiber rejection
   - Stage 2: Low flow for protein/starch separation

6. **Add instrumentation points** to simulation:
   - Cut size validation
   - Particle size distribution at each stage
   - Collection efficiency curves

---

## 9. Conclusion

**The classification system fundamentally does not work at current operating conditions.**

| What Works | What Fails |
|------------|------------|
| Particle entrainment | Zigzag separation |
| Material transport | Cyclone staging |
| No material loss | Protein recovery |
| Fast simulation | Starch recovery |

**Root cause:** Air flow rate (1,768 m³/h) is approximately **50× too high** for the zigzag classifier geometry and particle sizes involved.

**The system needs to be redesigned or operated at dramatically lower air flow rates** to achieve protein/starch separation.

---

## Quick Reference: Required Flow Rates

| Separation Goal | Cut Size | Required Flow | Zigzag Velocity |
|-----------------|----------|---------------|-----------------|
| Protein enrichment | 20 µm | 1.5 m³/h | 0.02 m/s |
| Protein/starch split | 35 µm | 4.5 m³/h | 0.05 m/s |
| Fiber rejection | 100 µm | 37 m³/h | 0.42 m/s |
| **Current operation** | **695 µm** | **1,768 m³/h** | **20.5 m/s** |

---

*Analysis based on 180-second simulation with 100,000 yellow pea flour particles*  
*Simulation rate: 7,778 steps/s on NVIDIA RTX 6000 Ada*
