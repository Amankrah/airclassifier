# Feed System Simulation Analysis Report

## Executive Summary

Your feed system simulation shows **mixed results** — the system functions but has some concerning behaviors that should be addressed before production use.

| Metric | Status | Notes |
|--------|--------|-------|
| Particle Transport | ⚠️ Partial | 80.5% throughput, 19.3% stuck in airlock |
| Mass Flow Rate | ✅ Good | 755 kg/h achieved (target: 500 kg/h) |
| Physics Validation | ✅ Correct | Reynolds numbers and flow regimes realistic |
| Geometry Validation | ✅ OK | Particles fit through all passages |

---

## 1. Flow Performance Analysis

### Final Particle Distribution (60s simulation)

```
Exited successfully:    1,609 particles (80.5%)  ✅
Stuck in Airlock:         385 particles (19.3%)  ⚠️ CONCERN
In Deagglomerator:          2 particles (0.1%)   
Inactive:                   4 particles (0.2%)   
```

### Key Concern: Airlock Retention

**385 particles (19.3%) remain trapped in the rotary airlock after 60 seconds.** This indicates potential issues:

1. **Vane pocket geometry** — Particles may be lodging in dead zones between vanes
2. **Clearance too tight** — 0.3mm vane tip clearance may cause jamming with larger particles (up to 30mm in your distribution)
3. **RPM may be insufficient** — At 20 RPM, each pocket only cycles 20 times in 60 seconds

### Recommendations:
- Review vane pocket geometry for particle-trapping dead zones
- Consider increasing airlock RPM from 20 to 25-30 RPM
- Verify the 0.3mm vane clearance is appropriate for your particle size distribution

---

## 2. Mass Flow Rate Analysis

### Achieved vs Design Rates

| Component | Design Rate | Calculated Rate | Status |
|-----------|-------------|-----------------|--------|
| Airlock (volumetric) | 1,168 kg/h | 3,854 kg/h | Over-capacity |
| Screw Feeder | 500 kg/h | 1,315 kg/h | Over-capacity |
| **Actual Throughput** | 500 kg/h | **755 kg/h** | ✅ 151% of target |

The system achieves **755 kg/h** actual throughput, which is **51% above your 500 kg/h target**. This suggests:
- The system has adequate capacity headroom
- Airlock and feeder are not the bottleneck (despite particle retention)
- Deagglomerator is processing material efficiently

---

## 3. Physics Validation

### Particle Properties (Yellow Pea Flour)

| Property | Value | Assessment |
|----------|-------|------------|
| Density | 1,420 kg/m³ | Realistic for legume flour |
| Sphericity | 0.70 | Appropriate for milled particles |
| Restitution | 0.30 | Correct for soft organic particles |
| Friction | 0.50 | Typical for flour-on-metal |

### Aerodynamic Analysis

```
Terminal Velocity:     22.4 m/s (mean), range 12.2 - 33.3 m/s
Reynolds Number:       28,222 (mean), range 3,634 - 65,934
Flow Regime:           100% Newton regime (Re > 1000)
```

**Assessment: ✅ Physically correct**

All particles are in the Newton (inertial) regime, which is expected for:
- Large particles (4.5 - 30mm diameter)
- High terminal velocities (>12 m/s)
- Air at standard conditions

The Schiller-Naumann drag correlation is appropriate for this regime.

### Settling Time Estimates

```
Hopper height: 1,576 mm (1.58 m)
Settling time: 0.05 - 0.13 seconds (mean: 0.08s)
```

These very short settling times indicate particles fall rapidly through the hopper — consistent with the fast hopper emptying observed in simulation (hopper emptied by t=3s).

---

## 4. Simulation Timeline Analysis

| Time (s) | Hopper | Airlock | Feeder | Deagg | Exited | Observation |
|----------|--------|---------|--------|-------|--------|-------------|
| 0.03 | 2,000 | 0 | 0 | 0 | 0 | Initial state |
| 3.03 | 0 | 1,120 | 19 | 323 | 525 | Hopper empties rapidly |
| 6.03 | 0 | 1,004 | 9 | 107 | 874 | Steady processing |
| 15.03 | 0 | 565 | 2 | 7 | 1,421 | Flow slowing |
| 30.03 | 0 | 482 | 1 | 3 | 1,510 | Diminishing returns |
| 60.00 | 0 | 385 | 0 | 2 | 1,609 | Asymptotic plateau |

### Key Observations:

1. **Rapid initial discharge** — Hopper empties in ~3 seconds (consistent with calculated settling times)
2. **Exponential decay in airlock** — From 1,120 → 385 particles follows diminishing evacuation rate
3. **Feeder works efficiently** — Never accumulates more than 19 particles
4. **Deagglomerator is not a bottleneck** — Peak of 323 particles, quickly processes down to 2

---

## 5. Component-by-Component Assessment

### Feed Hopper ✅ WORKING WELL
- Empties completely and rapidly
- Mass flow design (cone angle) is effective
- 150mm discharge adequate for particle sizes

### Rotary Airlock ⚠️ NEEDS ATTENTION
- Retains 19.3% of particles after 60s
- May have dead zones or insufficient clearance
- Consider geometry optimization or RPM increase

### Screw Feeder ✅ WORKING WELL
- Maintains low inventory (0-19 particles)
- Effective axial transport at 8 cm/s
- No accumulation or jamming observed

### Deagglomerator ✅ WORKING WELL
- Processes material faster than incoming rate
- Screen aperture (1.0mm) appropriate
- 15.7 m/s tip speed provides adequate impact energy

---

## 6. Particle Composition at Exit

### Input vs Output Composition

| Component | Input | Output | Change |
|-----------|-------|--------|--------|
| Protein | 500 (25%) | 171 (10.6%) | ⚠️ -14.4% |
| Starch | 1,100 (55%) | 1,038 (64.5%) | +9.5% |
| Fiber | 400 (20%) | 400 (24.9%) | +4.9% |

**Note:** The shift in composition suggests **selective retention of protein particles in the airlock**. This could indicate:
- Protein particles have different physical properties (size, density, adhesion)
- There may be segregation occurring based on particle characteristics
- Worth investigating if this affects downstream classification

---

## 7. Recommendations

### Immediate Actions

1. **Investigate airlock retention** — The 19.3% retention rate is the primary concern. Run diagnostic simulations to identify where particles are lodging.

2. **Test higher airlock RPM** — Increase from 20 RPM to 25-30 RPM and measure impact on retention.

3. **Review particle-airlock clearance** — With particles up to 30mm diameter and only 0.3mm vane clearance, verify no mechanical interference.

### Future Improvements

1. **Continuous feed mode** — Current simulation uses pre-loaded batch. Test with continuous particle injection to validate steady-state behavior.

2. **Longer simulation time** — Run to 120-180 seconds to see if airlock eventually clears or reaches permanent equilibrium.

3. **Size distribution optimization** — The 4.5-30mm particle range is very wide. Consider narrower distribution for more consistent flow.

---

## 8. Conclusion

**The feed system fundamentally works** — it successfully transports 80.5% of material through all four stages at above-design throughput (755 kg/h vs 500 kg/h target).

**However**, the 19.3% particle retention in the rotary airlock is a significant issue that should be resolved before production use. This could lead to:
- Reduced effective throughput
- Material degradation from extended residence time
- Potential jamming under continuous operation

**Priority action:** Focus diagnostic efforts on the airlock geometry and operating parameters.

---

*Analysis based on 60-second simulation with 2,000 yellow pea flour particles*
*Simulation rate: 4,528 steps/s on NVIDIA RTX 6000 Ada*
