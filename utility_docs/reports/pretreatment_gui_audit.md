# Pretreatment GUI Pipeline & Result-Reporting Audit

**Purpose:** Audit the complete pipeline of the pretreatment page in the GUI and document how result reporting does or does not align with `examples/simulate_and_visualize.py`.

**References:**
- GUI: `src/airclassifier/gui/pages/pretreatment_page.py`
- Example: `examples/simulate_and_visualize.py`
- Backend: `src/airclassifier/pretreatment/simulator.py`, `physics/coupling.py`
- Panel (dock): `src/airclassifier/pretreatment/gui_panel.py`

---

## 1. Complete Pipeline: Pretreatment Page (GUI)

### 1.1 Entry and Build

1. **Build system**  
   `PretreatmentPage.build_system(assembly_params)` is called from the main window with assembly params (e.g. `pt_electrode_gap_mm`, `pt_bed_depth_mm`). It builds the GP-15 machine via the same geometry path as the example and stores `_assembly_info`, then updates the 3D viewport.

2. **Control inputs**  
   User sets run mass, gap, bed depth, belt speed, duration (or auto from mass), MRH/MRL, and material (e.g. yellow pea). Recipe and material are built from these and passed into the simulator when **Run** is clicked.

3. **Simulation start**  
   On **Run**:
   - `_run_simulation()` starts a timer-driven loop (main thread).
   - `GP15Simulator` is created from config/material/recipe (same as example).
   - `sim.run(duration_s=..., adaptive_dt=True)` runs the coupled simulation.
   - Live KPIs are taken from the latest history step and shown in the control-panel cards (`_card_T`, `_card_M`, `_card_P`, etc.).
   - A render timer updates the 3D view (rollers, belt, particles, temperature overlay).

4. **Wind-down**  
   After `sim.run()` returns, the code waits for the belt to clear (particles collected, wind-down). When done, `_finalize_results()` is called.

5. **Result assembly (`_finalize_results`)**
   - `result = self._sim.build_result()` → `PretreatmentResult` (from coupling `_build_result()`).
   - `outlet = self._sim.get_outlet_conditions()` → `OutletState`.
   - `meshes = self._sim.get_mesh()`.
   - **Time series:** Built from **`result.time_series`** (canonical, includes `electrode_temperature_c`); **`controller_state`** is appended from `history` for GUI/export.
   - **Particle data:** From `sim.particles` (collected/riding counts, `T_collected`, `M_collected`, vicilin/legumin, T_surface/T_core if available).
   - `_results` dict is set with: `outlet`, `result`, `meshes`, `time_series`, `elapsed_s`, `n_steps`, `collected_mass_kg`, `dispatched_mass_kg`, `run_mass_kg`, `gap_mm`, `bed_depth_mm`, `belt_speed`, `duration_s`, `initial_moisture`, `initial_temp_c`, `particle_data`.

6. **Results view**
   - `_draw_simulation_plots(ts, _results)` → 3×3 matplotlib grid.
   - `_update_summary(_results)` → KPI cards.
   - `_draw_outfeed_section(_results)` → outfeed T + M cross-sections.
   - `_draw_particle_plots(_results)` → 2×2 particle analysis.
   - `_update_desirability(_results)` → desirability overall + 5 dimensions.
   - Stacked widget switches to the “Results” page; user can export CSV/JSON/PNG/PDF.

### 1.2 Data Flow Summary

```
User inputs → Recipe + Material + Config
    → GP15Simulator.run(duration_s, adaptive_dt=True)
    → CoupledSimulator (physics) → _history (StepState list)
    → build_result() → PretreatmentResult (time_series from _history in coupling)
    → get_outlet_conditions() → OutletState
GUI: builds its own time_series from sim.history (same step list, but different keys than result.time_series in one place — see §2)
    → _results dict → _update_summary, _draw_* → KPI cards, 3×3 plots, outfeed, particle, desirability
```

---

## 2. Misalignments: GUI vs simulate_and_visualize

### 2.1 Time-Series Keys (Export / CSV)

- **Coupling `_build_result()`** puts in `result.time_series`:  
  `electrode_temperature_c` and does **not** include `controller_state`.
- **GUI** builds `ts` from `history` and includes `controller_state` but does **not** include `electrode_temperature_c`.

So:

- **GUI CSV export** (and any use of `_results["time_series"]`) has `controller_state` but is **missing** `electrode_temperature_c`.
- **Example** uses `result.time_series` from `sim.build_result()`, which has `electrode_temperature_c` and no `controller_state`.

**Recommendation:** Either build GUI `time_series` from `result.time_series` after `build_result()`, or append `electrode_temperature_c` from history so CSV/plots match the canonical result and the example.

---

### 2.2 Desirability Input: Outfeed Temperature

- **Example** (`_print_results` → `score_desirability`):  
  `outfeed_temperature_c=outlet.avg_temperature_c`
- **GUI** (`_update_desirability`):  
  `outfeed_temperature_c=outlet.sensor_temperature_c`

So the **desirability score** in the GUI uses the sensor (75th percentile) temperature, while the example uses the average outfeed temperature. Scores can differ slightly.

**Recommendation:** Decide a single convention (e.g. sensor vs mean) and use it in both the example and the GUI, and document it (e.g. in desirability docstring).

---

### 2.3 PDF Report vs KPI Cards (Outfeed Temperature)

- **KPI cards:** “Outfeed Temp (Sensor)” shows `outlet.sensor_temperature_c` (aligned with example console “Outfeed temperature” which prints `sensor_temperature_c`).
- **PDF report** (page 1 text): Uses `outlet.avg_temperature_c` in the “Results” block (“Outfeed Temperature: … C”).

So the **PDF** uses mean temperature while the **cards and example** use sensor temperature.

**Recommendation:** Use `outlet.sensor_temperature_c` in the PDF results block as well, and optionally add a line “Outfeed temp (sensor, P75): … °C” so it matches the cards and the example.

---

### 2.4 Specific Energy Unit Label

- **Example:** Console and outfeed subplot label: “kWh/kg **water**” (specific energy per kg water removed).
- **GUI:** KPI card title is “Specific Energy”; value is “X.XXX kWh/kg”. Outfeed moisture subplot title uses “Spec. energy … kWh/kg” without “water”.

So the GUI does not spell out “per kg water” and can be read as “per kg product”.

**Recommendation:** Use “Specific Energy (kWh/kg water)” in the KPI card and “kWh/kg water” in the outfeed plot so it matches the example and avoids ambiguity.

---

### 2.5 Mass Balance / Material Accounting Plot

- **Example** (fig2 [1,1]): Bar labels “Input” and “Collected”; title includes mass balance: “Mass Balance (n particles, **+X.X%**)”.
- **GUI** (3×3 [2,1]): Bar labels “Infeed (run mass)” and “Collected (bin)”; title is “Material Accounting (n particles)” with **no balance %**.

So the GUI plot does not show the mass balance percentage that the example and the KPI card show.

**Recommendation:** Add balance % to the [2,1] plot title when `dispatched_kg > 0`, e.g. “Material Accounting (n particles, ±X.X%)”, and optionally use “Input” / “Collected” for consistency with the example.

---

### 2.6 Outfeed Cross-Section: “at oven exit” Note

- **Example:** When `outlet.at_peak_processing_snapshot` is True, the outfeed figure suptitle includes “at oven exit (peak)” and the temperature subplot title can say “Outfeed T at oven exit (peak)”.
- **GUI:** Outfeed section suptitle is “Outfeed Cross-Section — Pipeline Output to Milling” with residence and throughput, but does **not** add “at oven exit (peak)” when the snapshot is the peak-processing one.

So the GUI does not indicate when the cross-section is the peak (oven-exit) snapshot rather than end-of-run.

**Recommendation:** If `outlet.at_peak_processing_snapshot` is True, append “ at oven exit (peak)” (or similar) to the outfeed section title so behaviour matches the example.

---

### 2.7 Dock Panel (gui_panel) vs Full Results

- **PretreatmentPage** stores the full `PretreatmentResult` in `_results["result"]` and uses it for energy, throughput, and desirability.
- **gui_panel** worker, on completion, emits `simulation_completed` with `outlet`, `meshes`, and a **time_series** built from `sim._sim._history` — and does **not** call `sim.build_result()` or include the `PretreatmentResult` object.

So any consumer of `simulation_results_ready` from the dock panel does **not** receive `result.energy_consumed_kwh` or `result.throughput_kg_per_h`; they only have `outlet` (e.g. `outlet.total_energy_kwh`, `outlet.throughput_kg_per_hr`). Naming differs (`throughput_kg_per_h` vs `throughput_kg_per_hr`). Logic is similar but not identical (result vs outlet).

**Recommendation:** If the dock panel is used for result display or export, either (a) call `sim.build_result()` and include `result` in the emitted dict and use it for energy/throughput, or (b) document that only `outlet` is provided and that energy/throughput should be taken from `outlet` with the correct attribute names.

---

## 3. Aligned Behaviour (No Change Needed)

- **Outfeed moisture, sensor temp, max temp, CV:** Same source (`outlet`) and same semantics in GUI and example.
- **RF energy:** GUI uses `result.energy_consumed_kwh` for the KPI card, matching the example’s `result.energy_consumed_kwh`.
- **Throughput:** GUI uses `result.throughput_kg_per_h`, matching the example’s `result.throughput_kg_per_h` (example prints “Throughput” from result).
- **Protein quality:** Both use `outlet.protein_denaturation_fraction` and, when available, vicilin/legumin from particles.
- **Mass balance KPI card:** Both use dispatched vs collected and show balance % the same way.
- **Final electrode gap:** From last value of `electrode_gap_mm` in time series in both.
- **3×3 plot layout:** Same logical content (temperature+protein, moisture, gap, RF power, anode current, cumulative energy, specific energy, mass account, outfeed T cross-section); minor label/title differences only as noted above.
- **Outfeed 1×2 (T + M):** Same extent (belt width × gap), same fields and colorbars; only the “at peak” note and specific-energy unit label differ (§2.4, §2.6).
- **Particle analysis:** Same concepts (state pie, T histogram at oven exit, Vicilin/Legumin bars, core vs surface); GUI uses `particle_data` derived from the same particle system.

---

## 4. Summary Table

| Item | Example | GUI | Aligned? |
|------|--------|-----|----------|
| Time-series source | `result.time_series` (coupling) | Manual from `history` | No: GUI has `controller_state`, lacks `electrode_temperature_c` |
| Desirability outfeed T | `outlet.avg_temperature_c` | `outlet.sensor_temperature_c` | No |
| PDF “Outfeed Temperature” | N/A (console only) | `outlet.avg_temperature_c` | No vs cards (cards use sensor) |
| Specific energy label | “kWh/kg water” | “kWh/kg” | No |
| Mass plot title | “Mass Balance (n, ±X.X%)” | “Material Accounting (n)” | No (no % in GUI) |
| Outfeed section “at peak” note | Yes when applicable | No | No |
| Dock panel result object | N/A | No `build_result()` in worker | No (if dock used for reporting) |
| KPI moisture, T sensor, max T, CV, energy, throughput, protein, mass balance, gap | Same semantics | Same | Yes |
| 3×3 and outfeed/particle content | Same | Same | Yes (aside from labels above) |

---

## 5. Recommended Changes (Concise)

1. **Time series:** Build GUI `time_series` from `result.time_series` after `build_result()`, or add `electrode_temperature_c` from history and document that `controller_state` is GUI-only (or add it to coupling and result.time_series).
2. **Desirability:** Unify on one outfeed temperature (e.g. always `sensor_temperature_c` in both example and GUI) and document.
3. **PDF:** Use `outlet.sensor_temperature_c` in the PDF “Outfeed Temperature” line to match KPI and example.
4. **Labels:** Use “Specific Energy (kWh/kg water)” and “kWh/kg water” in GUI card and outfeed plot.
5. **Mass plot:** Add mass balance % to the [2,1] title and optionally align bar labels with the example.
6. **Outfeed section:** When `outlet.at_peak_processing_snapshot` is True, add “ at oven exit (peak)” to the section title.
7. **Dock panel:** Include `build_result()` in the worker and pass `result` in `simulation_completed`, or document outlet-only and attribute names.

These changes will align the GUI’s result reporting with the example and with the canonical `PretreatmentResult`/`OutletState` usage.

---

## 6. Implementation Status (Completed)

| # | Recommendation | Status |
|---|----------------|--------|
| 1 | Time series from result.time_series + controller_state | Done |
| 2 | Desirability: sensor temperature in example and GUI | Done |
| 3 | PDF: Outfeed Temperature (sensor P75) | Done |
| 4 | Specific Energy (kWh/kg water) labels | Done |
| 5 | Mass Balance plot with % and Input/Collected | Done |
| 6 | Outfeed at oven exit (peak) note | Done |
| 7 | Dock: build_result() and result in simulation_completed | Done |

---

## 7. Re-audit Verification (All Issues Fixed)

**Date:** Re-audit performed to confirm all seven recommendations are implemented.

### 7.1 Time series (Recommendation 1)

- **Pretreatment page** (`pretreatment_page.py` ~1559–1563): After `result = self._sim.build_result()`, `ts = dict(result.time_series)`; then `ts["controller_state"]` is added from `history`. CSV/export now includes `electrode_temperature_c` (from result) and `controller_state` (from history). **Fixed.**
- **Dock panel** (`gui_panel.py` ~201–220): Worker calls `result = sim.build_result()`, then `ts = dict(result.time_series)` and adds `controller_state` from history; emit includes `"result": result`, `"time_series": ts`. **Fixed.**

### 7.2 Desirability outfeed temperature (Recommendation 2)

- **GUI** (`pretreatment_page.py` ~896): `_update_desirability` uses `outfeed_temperature_c=outlet.sensor_temperature_c` with comment "Use sensor-comparable temp". **Fixed.**
- **Example** (`simulate_and_visualize.py` ~946): `score_desirability(..., outfeed_temperature_c=outlet.sensor_temperature_c, ...)` with comment "Align with GUI: sensor (P75) for desirability". **Fixed.**

### 7.3 PDF "Outfeed Temperature" (Recommendation 3)

- **PDF report** (`pretreatment_page.py` ~2399): Title text uses `"Outfeed Temperature (sensor P75): {outlet.sensor_temperature_c:.1f} C"`. **Fixed.**

### 7.4 Specific energy labels (Recommendation 4)

- **KPI card** (~733): `_StatCard("Specific Energy (kWh/kg water)", ...)`. **Fixed.**
- **3×3 plot** (~1931): Axis title `"Specific Energy (kWh/kg water)"`; reference lines "1.0 kWh/kg", "1.67 kWh/kg". **Fixed.**
- **Outfeed moisture subplot** (~2039): `f"Spec. energy ... kWh/kg water"`. **Fixed.**
- **_update_summary** (~2215): Value format `f"{...:.3f} kWh/kg water"`. **Fixed.**
- **PDF** (~2403): `"Specific Energy: ... kWh/kg water"`. **Fixed.**

### 7.5 Mass Balance plot (Recommendation 5)

- **3×3 [2,1]** (~1936–1953): Bar labels `["Input", "Collected"]`; when `dispatched_kg > 0`, `balance_pct` is computed and title is `f"Mass Balance ({collected_n} particles{balance_str})"` with `balance_str = f", {balance_pct:+.1f}%"`. **Fixed.**

### 7.6 Outfeed section "at peak" note (Recommendation 6)

- **_draw_outfeed_section** (~2004–2006): `peak_note = " at oven exit (peak)" if getattr(outlet, "at_peak_processing_snapshot", False) else ""`; suptitle includes `f"... Pipeline Output to Milling{peak_note}  |  ..."`. **Fixed.**

### 7.7 Dock panel result (Recommendation 7)

- **Worker** (`gui_panel.py` ~201–220): `result = sim.build_result()`; emit dict includes `"outlet"`, `"result"`, `"meshes"`, `"time_series"` (from result + controller_state). **Fixed.**

### 7.8 Re-audit summary table

| # | Issue | Status | Evidence |
|---|--------|--------|----------|
| 1 | Time series: electrode_temperature_c + controller_state | **Fixed** | result.time_series used; controller_state from history (page + dock) |
| 2 | Desirability: sensor temp in GUI and example | **Fixed** | sensor_temperature_c in both |
| 3 | PDF Outfeed Temperature (sensor P75) | **Fixed** | Line 2399 |
| 4 | Specific Energy (kWh/kg water) | **Fixed** | Card, plot, outfeed, PDF |
| 5 | Mass Balance plot: Input/Collected + % | **Fixed** | [2,1] title and labels |
| 6 | Outfeed "at oven exit (peak)" | **Fixed** | peak_note in _draw_outfeed_section |
| 7 | Dock: build_result() and result in emit | **Fixed** | gui_panel worker lines 201–220 |

**Conclusion:** All seven audit recommendations have been implemented. GUI result reporting is aligned with `examples/simulate_and_visualize.py` and with the canonical `PretreatmentResult`/`OutletState` usage.
