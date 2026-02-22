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

**Root cause:** The simulated anode current peaks at ~1.46 A, which is **below MRH (1.7 A)**. The controller opens the gap only when `Ia > MRH`. Since the sim never exceeds 1.7 A, the gap never opens.

### Run#2 CSV data (why gap and temperature differ)

Source: `utility_docs/Run2 RF data(in).csv`, `utility_docs/Pea RF summary(Run#2).csv`.

**Physical Run#2 (from CSV):**

| Time (approx) | Ia (A) | Electrode_Act (mm) | Product_Temp (°C) |
|---------------|--------|--------------------|-------------------|
| 14:20:28      | 0.21   | 106.8              | 27 (idle)         |
| 14:25:28      | 0.30   | 75.1               | 37 (setpoint)     |
| **14:28:08**  | **1.72** | 75.1 → **MRH trip** | 39                |
| 14:28:13–14:31 | 1.65–1.70 | **75 → 79 → 82** (opening) | 39–42 |
| 14:32–14:35   | 1.66–1.71 | **87–94.1** (peak gap) | 42–43 |
| 14:35–14:40   | 1.59–1.69 | **94.1** (held)   | 43–89 (PLC sensor rises) |
| 14:39–14:41   | 1.55–1.62 | 93.9              | **94–103** (PLC)  |

- **Temperature strips** (Pea RF summary): outfeed strips **77, 77, 82, 77, 82** °C.
- **NIR outfeed moisture:** **10.53%** wb (avg of 15 samples).

So in the real run, **Ia hits 1.72 A** (row 94 in CSV), **exceeds MRH 1.7 A**, the relay trips, and the gap opens from 75 mm to **94.1 mm**. Product_Temp (PLC) and strips then reach 77–103°C. In the simulation, **Ia never reaches 1.7 A** (peaks ~1.46 A), so the controller never opens the gap, and the sim under-delivers power → outfeed stays ~65°C and moisture stays higher (11.59%).

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

## Code audit: coupling, config, controller, particles

Trace of where each Run#2 metric comes from and why mismatches occur.

### Outfeed temperature (sim 65°C vs target 77–82°C)

- **Source in code:** `coupling.py` → `_record_step()`:
  - Outfeed slice: `T_out_cells = self.thermal.T[-1, :, :][outfeed_mat]`
  - When `has_material`: `T_outfeed_sensor = np.percentile(T_out_cells, 75)` (sensor‑comparable)
  - `get_outlet_conditions()` returns `sensor_temperature_c` from peak snapshot or `_last_valid_T_outfeed_sensor`
- **Example:** `simulate_and_visualize.py` prints `outlet.sensor_temperature_c` as “Outfeed temperature”.
- **Diagnosis:** Sim thermal solution produces a cooler outfeed cross‑section than the physical strips. Possible causes: (1) overall RF power too low (Ia/gap cascade), (2) convective cooling or thermal inertia not matched, (3) 75th‑percentile vs strip placement/sampling difference.

### Anode current Ia (sim peak ~1.45 A vs target 1.5–1.7 A)

- **Source in code:** `coupling.py` → same step:
  - `fraction = P_rf_kw_theoretical / machine.max_rf_power_kw`
  - `I_a = no_load + (full_load - no_load) * fraction`, then RC‑filtered
  - `P_rf_theoretical` comes from RF field + dielectric heating (coupling factor, E², σ'').
- **Diagnosis:** At Run#2 conditions (90 kg, 35 mm bed), simulated load gives P_rf such that Ia stays just below 1.5 A. Calibration was on Run#1 (61 kg, 25 mm); coupling factor is not scaled for thicker/heavier beds, so Ia is under‑predicted and never reaches MRH (1.7 A).

### Gap never opening (sim 75 mm vs target peak 94.1 mm)

- **Source in code:** `controller.py` → `_gap_step()`:
  - `mrh_active = (anode_current_a > recipe.mrh_amps)` (1.7 A)
  - Only when `mrh_active`: gap increases at `gap_adjust_rate_mm_s * gain * dt`
  - `mrl_active = (anode_current_a < recipe.mrl_amps)` (1.5 A); when batch exhausted, gap closes toward setpoint.
- **Config:** `Recipe` defaults `mrh_amps=1.7`, `mrl_amps=1.5` (`config.py`). Example does not override them.
- **Diagnosis:** Because sim Ia never exceeds 1.7 A (and barely reaches 1.5 A), `mrh_active` is never True, so the controller never opens the gap. Cascade: under‑predicted Ia → no MRH trip → gap fixed at 75 mm → different power density and thermal outcome than Run#2.

### Outfeed moisture (sim 11.59% vs target 10.53% wb)

- **Source in code:** Same `_record_step()`: `M_outfeed_mean` from `moisture.M[-1, :, :][outfeed_mat]`; when belt has cleared, `get_outlet_conditions()` uses `_moisture_at_batch_exhausted`.
- **Diagnosis:** Sim retains slightly more moisture (less drying). Consistent with underheating (lower T and possibly lower effective drying rate).

### Mass balance (dispatched 90 kg, collected 89 kg, −1.1%)

- **Source in code:** `particles.py`:
  - Dispatch: `_dispatched_mass_kg += mass_per_particle` (throughput/spawn_rate or run_mass_kg/max_particles).
  - Collection: `mass_ratio = (1 - M_initial) / (1 - M_landed)`; `_total_collected_kg += sum(mass_per_particle * mass_ratio)`.
- **Diagnosis:** Collected < dispatched is correct when moisture is lost. The small deficit is within tolerance; possible minor contribution from particle‑level moisture interpolation and rounding.

### Config and calibration loading

- **MachineConfig:** `oscillator_coupling_factor` uses `default_factory=lambda: get_calibration_defaults()[0]`, so `calibration_latest.json` is loaded when the config is created.
- **MaterialProperties:** `k_evap` (and `k_dispersion`) similarly use `get_calibration_defaults()`.
- **Example:** Only `gap_adjust_rate` is passed explicitly via `sim._sim.update_parameters(gap_adjust_rate=...)`; coupling and k_evap come from config/material built with defaults, so calibration is applied for non‑`--calibrate` runs.

### Validation helper

- `src/airclassifier/utils/validation.py`: `RUN2_TARGETS`, `compare_sim_to_run2(outlet, ts=..., dispatched_kg=..., collected_kg=...)` return a `ValidationResult` with pass/fail and short notes for outfeed T, moisture, gap, Ia, and mass balance.

### Figure 5 / particle treatment temperature (fixed)

**Particle Treatment Analysis** (e.g. Figure 5) plots treatment temperature and moisture **at oven exit**. Previously, `T_at_oven_exit` was captured when a particle crossed **oven_x_end** (3.6 m). By then the particle had already left the RF grid (grid ends at ~2.96 m) and was no longer interpolated, so its `self.temperature` had been cooling for ~200 s. That made the particle histogram show mean ~41°C and a large 20–25°C spike while the grid outfeed reported ~65°C.

**Fix (particles.py):** Capture treatment temperature when the particle first crosses **grid_x_end** (RF zone exit), not oven_x_end. At that moment `self.temperature` still holds the last interpolated value (outfeed cell). The particle histogram and the grid outfeed metric (sensor P75) now refer to the same physical moment (exit from RF zone).

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
