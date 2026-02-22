# Run#2 Validation Report

## Calibration Parameters Used

From `calibration_latest.json` (calibrated on Run#1):

| Parameter | Value |
|-----------|-------|
| coupling_factor | 0.1294 |
| k_evap | 1.42e-06 |
| k_dispersion | 0.100 W/(m·K) |
| gap_adjust_rate | 0.2737 mm/s |

## Run#2 Conditions

| Parameter | Value |
|-----------|-------|
| Mass | 90 kg |
| Bed depth | 35 mm |
| Initial gap | 75 mm |
| Belt speed | 0.2 m/min |
| Initial temp | 17.0°C |
| Initial moisture | 11.81% wb |

## Results Comparison

| Metric | Target (NIR/PLC) | Simulation | Error | Status |
|--------|------------------|------------|-------|--------|
| Outfeed moisture | 10.53% wb | 11.58% wb | +1.05% | **Too wet** |
| Outfeed temperature | 77-82°C | 64.5°C | -12.5 to -17.5°C | **Too cold** |
| Max temperature | ~85°C (est) | 87.3°C | +2°C | OK |
| Gap peak | 94.1 mm | 75.0 mm | -19.1 mm | **No movement** |
| Ia steady | 1.5-1.7 A | 1.44-1.46 A | -0.04 to -0.26 A | Slightly low |

## Critical Issue: Gap Controller Not Triggering

The electrode gap remained fixed at 75 mm throughout the entire 2155 s simulation:

```
Gap peak:    75.0 mm at t=0s
Gap final:   75.0 mm
```

**Root cause:** The simulated anode current peaked at ~1.46 A, which is below the MRL threshold of 1.5 A. Since Ia never exceeded MRL, the controller never triggered gap opening.

### Run#2 PLC Data Confirmation

| Parameter | Physical (PLC) | Simulation | Error |
|-----------|----------------|------------|-------|
| MRH | 1.7 A | 1.7 A | - |
| MRL | 1.5 A | 1.5 A | - |
| Ia peak | **1.72 A** | **1.46 A** | **-0.26 A** |
| Gap peak | **94.1 mm** | **75.0 mm** | **-19.1 mm** |
| Samples > MRL | 306 (54%) | 0 (0%) | - |

The physical Run#2 sustained Ia > 1.5 A for **54% of the run** (306 of 564 samples), triggering continuous gap adjustment. The simulation never reached this threshold.

**Ia deficit: 0.26 A (15% lower than physical)**

## Ia Trajectory Analysis

### Physical Run#2 (PLC Data)

| Time | Ia (A) | Gap (mm) | Temp (°C) | Notes |
|------|--------|----------|-----------|-------|
| 14:20:28 | 0.21 | 106.8 | 27 | Electrode driving down |
| 14:25:28 | 0.30 | 75.1 | 37 | Gap at setpoint |
| 14:30:28 | **1.69** | 79.2 | 41 | Ia > MRL, gap opening |
| 14:35:28 | 1.64 | **94.1** | 44 | Gap peak |
| 14:40:28 | 1.59 | 93.9 | 89 | Steady state |
| 14:45:28 | 1.51 | 93.4 | 68 | Ia still > MRL |
| 14:50:28 | 1.50 | 91.6 | 68 | Gap closing |
| 14:55:28 | 1.55 | 80.0 | 68 | Material exiting |
| 15:00:28 | 0.32 | 75.2 | 84 | Runout |
| 15:05:28 | 0.31 | 75.2 | 45 | Final |

### Simulation

| Time (s) | Sim Ia (A) | PLC Ia (A) | Deficit |
|----------|------------|------------|---------|
| 300 | 1.29 | ~1.3 | OK |
| 600 | 1.44 | **1.69** | **-0.25 A** |
| 900 | 1.45 | 1.64 | -0.19 A |
| 1200 | 1.18 | 1.51 | -0.33 A |
| 1500 | 0.43 | 0.32 | +0.11 A |

**The simulation consistently under-predicts Ia by 0.2-0.3 A during the heating phase.**

## Hypothesis: Coupling Factor Too Low for Thicker Beds

| Parameter | Run#1 (Calibration) | Run#2 (Validation) | Change |
|-----------|---------------------|---------------------|--------|
| Bed depth | 25 mm | 35 mm | +40% |
| Mass | 61 kg | 90 kg | +47% |
| Peak Ia (physical) | 1.72 A | 1.72 A | Same |
| Peak Ia (simulation) | 1.69 A | **1.46 A** | **-14%** |

The simulation correctly predicts Ia for Run#1 but **under-predicts by 0.26 A for Run#2**.

Possible causes:

1. **Non-linear bed depth effect**: The E-field intensity scales differently with bed depth. At 35mm, more material is exposed to the fringe field region, increasing effective coupling.

2. **Capacitance change**: Thicker dielectric layer (35mm vs 25mm) changes the effective capacitance of the oscillator-electrode-material system. The series capacitor model may not capture this fully.

3. **Field uniformity**: 25mm bed has more uniform field penetration; 35mm bed has stronger surface heating, which may produce higher average dielectric loss.

## Cascade Effects

Because the gap controller never triggered:
1. Gap stayed at 75 mm instead of opening to 94 mm
2. Higher-than-optimal power density in material
3. But Ia being low means overall power delivery was insufficient
4. Net result: underheating and under-drying

## Recommendations

### Option A: Bed-Depth Dependent Coupling (Quick Fix)

Add bed depth correction factor to coupling:
```python
k_eff = k_base * (1 + alpha * (bed_depth - 25) / 25)
```

Required correction:
- Current k = 0.1294
- Target Ia = 1.69 A (vs sim 1.46 A)
- Ia ratio = 1.69 / 1.46 = 1.16
- Since Ia ∝ k, need k_eff ≈ 0.1294 × 1.16 = **0.150**
- alpha ≈ (0.150 / 0.1294 - 1) / 0.4 = **0.40**

### Option B: Recalibrate on Run#2 (Recommended)

Run calibration directly on Run#2 PLC data:
```bash
python diag_calibrate.py --plc "utility_docs/Run2 RF data(in).csv" \
    --mass 90 --bed-depth 35 --moisture 0.1181 --temp 17.0
```

Compare the calibrated coupling factor to Run#1's 0.1294. If k₂ ≈ 0.15, this confirms the bed-depth dependency hypothesis.

### Option C: Multi-Run Calibration (Best)

Calibrate simultaneously on both Run#1 and Run#2 with a bed-depth-aware coupling model. This would give a single set of parameters that generalizes across bed depths.

## Mass Balance

| Metric | Value |
|--------|-------|
| Mass input | 90.00 kg |
| Mass collected | 89.01 kg |
| Mass balance | -1.1% |

The mass balance is acceptable (within 2% tolerance).

## Protein Quality (Informational)

| Metric | Value | Target |
|--------|-------|--------|
| Globulin native loss | 22.8% | <15% |
| Vicilin (7S) loss | 35.2% | <25% |
| Legumin (11S) loss | 16.2% | <20% |

Note: These values are computed based on the simulated temperature profile. With actual temperatures being lower in the physical run, real protein damage may differ.

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| MRH/MRL thresholds | ✓ Confirmed | Same as Run#1 (1.7/1.5 A) |
| Ia prediction | ✗ Under-predicts | 1.46 A vs 1.72 A physical (-15%) |
| Gap control | ✗ Not triggered | Never exceeded MRL threshold |
| Temperature | ✗ Too low | 64.5°C vs 77-82°C target |
| Moisture | ✗ Too wet | 11.58% vs 10.53% target |
| Mass balance | ✓ OK | -1.1% (within tolerance) |

**Root cause:** Coupling factor calibrated on 25mm beds under-predicts power absorption for 35mm beds.

**Recommended fix:** Recalibrate on Run#2 PLC data to determine bed-depth-dependent coupling correction.

---

*Generated: 2026-02-19*
*Calibration source: Run#1 (2794 s, Nelder-Mead, 129 evals)*
*Validation data: Run#2 (564 samples, 2820 s)*
