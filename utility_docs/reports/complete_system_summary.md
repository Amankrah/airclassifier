# Air Classifier System - Complete Analysis Summary

## Overall System Status: ⚠️ REQUIRES REDESIGN

| Subsystem | Status | Primary Issue |
|-----------|--------|---------------|
| Feed System | ⚠️ Partial | 19.3% particle retention in airlock |
| Air System | ✅ Good | Working well, stable operation |
| Classification | ❌ Failed | Air flow 50× too high for separation |

---

## Critical Finding

**The three subsystems are not properly matched:**

```
Feed System          Air System           Classification System
(755 kg/h)    →     (1,768 m³/h)    →    (d50 = 695 µm)
     ↓                    ↓                      ↓
  Working            Working              NOT separating
  (mostly)           (well)               anything
```

The air system provides **50× more airflow** than the classification system needs. This causes:
- Zigzag classifier acts as transport duct (not separator)
- Primary cyclone captures 99.85% of all material
- Zero protein recovery
- Zero meaningful starch/protein separation

---

## Subsystem Summaries

### 1. Feed System

| Metric | Value | Status |
|--------|-------|--------|
| Throughput | 755 kg/h | ✅ Above 500 kg/h target |
| Particle transport | 80.5% exited | ⚠️ Acceptable |
| Airlock retention | 19.3% stuck | ⚠️ Needs attention |
| Hopper emptying | 3 seconds | ✅ Fast |

**Action needed:** Investigate airlock geometry, consider increasing RPM from 20 to 25-30.

### 2. Air System

| Metric | Value | Status |
|--------|-------|--------|
| Flow rate | 1,768 m³/h | ✅ Stable |
| Pressure rise | 1,736 Pa | ✅ Adequate |
| Efficiency | 68.6% | ✅ Good for partial load |
| Stability | Rock-solid | ✅ Excellent |

**Action needed:** None for the air system itself, but flow must be reduced/bypassed for classification.

### 3. Classification System

| Metric | Value | Status |
|--------|-------|--------|
| Protein recovery | 0% | ❌ Complete failure |
| Zigzag cut size | 695 µm | ❌ Should be 20-40 µm |
| Cyclone staging | None | ❌ Cy1 captures everything |
| Material loss | 0% | ✅ Good containment |

**Action needed:** Major redesign or dramatically reduced air flow.

---

## Recommended Solutions

### Option 1: Add Flow Bypass (Recommended)

```
Blower → Main Line (high flow for transport)
             │
             ├─→ Bypass valve → Exhaust
             │
             └─→ Classification line (~50 m³/h)
```

This allows:
- High flow for venturi entrainment and cyclone operation
- Controlled low flow through zigzag for actual separation

### Option 2: Separate Air Systems

- **Transport air:** Main blower at current settings
- **Classification air:** Small dedicated blower (50-100 m³/h) for zigzag

### Option 3: Resize Classification System

Increase zigzag channel area by ~100× to match current flow.
- Channel width: 120 mm → ~1,200 mm
- Channel depth: 200 mm → ~2,000 mm

**Not recommended** — would require industrial-scale equipment.

### Option 4: Different Technology

At 1,768 m³/h, consider:
- Turbo air classifier (centrifugal)
- Multi-stage cyclone-only separation
- Air table classifier

---

## Flow Rate Requirements for Separation

| Separation Target | Cut Size (d50) | Required Flow | Current Flow |
|-------------------|----------------|---------------|--------------|
| Protein enrichment | 20 µm | 1.5 m³/h | 1,768 m³/h ❌ |
| Protein/starch split | 35 µm | 4.5 m³/h | 1,768 m³/h ❌ |
| Fiber rejection only | 100 µm | 37 m³/h | 1,768 m³/h ❌ |

---

## Next Steps

1. **Immediate:** Add flow control or bypass to classification system
2. **Run test:** Simulate at Q = 50 m³/h to verify separation occurs
3. **Feed system:** Run airlock diagnostics to find particle trap locations
4. **Integration:** Re-simulate full system with matched flow rates

---

## Simulation Performance

All simulations ran efficiently on your NVIDIA RTX 6000 Ada:

| Simulation | Particles | Wall Time | Rate |
|------------|-----------|-----------|------|
| Feed system | 2,000 | 2.7 s | 4,528 steps/s |
| Air system | 1,000 SPH | 12.2 s | 4,971 steps/s |
| Classification | 100,000 | 23.1 s | 7,778 steps/s |

The physics engines work correctly — this is a **design problem**, not a simulation problem.
