# Calibration report

## GPU acceleration

When an NVIDIA GPU is available (Warp detects `cuda:0`), the simulation physics loop runs on the GPU via pre-compiled Warp CUDA kernels. The following per-timestep operations are GPU-accelerated:

| Step | Operation | GPU kernel |
|------|-----------|------------|
| 1 | Advection (belt transport of T, M) | `advect_material_wp_kernel` |
| 3 | RF power density P_v = 2πfε₀ε″\|E\|² | `_compute_power_density_kernel` |
| 5 | Thermal FDM (heat equation + convection BC) | `heat_conduction_step`, `apply_convection_bc` |
| 6 | Moisture diffusion + evaporation | `moisture_step` |
| 7 | Material property update (ε′, ε″, ρcₚ, k) | `_update_properties_kernel` |

**Remaining on CPU:** RF field solve (step 2, series-capacitor model — lightweight), PLC controller logic (step 8), KPI recording (step 9), Lagrangian particles (step 10), and boundary condition application after each GPU kernel.

**Data flow:** GPU arrays are persistent across timesteps. After each physics step the results are synced to CPU for boundary conditions, controller logic, and KPI computation.

**Calibration:** The same GPU-accelerated path is used during calibration (each `differential_evolution` evaluation runs a full simulation on GPU). The optimizer itself (scipy) runs on CPU between evaluations. Pass `--cpu` to force CPU-only mode.

## Stored calibration (automatic)

Calibration cannot be run every time. The latest calibrated parameters are **saved automatically** after each calibration run and **loaded automatically** when you run without `--calibrate`.

- **File:** `utility_docs/calibration_latest.json`
- **Parameters stored:** `oscillator_coupling_factor`, `k_evap`, `gap_adjust_rate_mm_s`
- **When calibrating:** Run with `--calibrate "utility_docs/Run1 RF data(in).csv"` (optionally `--cal-duration 0` for full run). The result is written to the file and applied to the simulator (including controller gap rate).
- **When not calibrating:** Running `simulate_and_visualize.py` without `--calibrate` loads from `calibration_latest.json` if present and applies all three parameters so simulations use the latest calibration.

API: `airclassifier.pretreatment.calibration_store.save_calibration(result, path)` and `load_calibration(path)`.

## Parameter comparison (300 s vs full 2794 s calibration)

| Parameter        | 300s calibration | Full 2794s calibration | Change              |
|------------------|------------------|--------------------------|---------------------|
| coupling_factor  | 0.201            | 0.181                    | Lower (less power)  |
| k_evap           | 5.9e-5           | 1.01e-6                  | 59× lower          |
| gap_rate         | 0.012 mm/s       | 0.649 mm/s               | 54× faster         |

The full-window calibration is preferred when possible; the stored file allows reusing that result without re-running calibration.

---

## Run#1: Simulation vs physical machine (2794 s)

Comparison of a **2794 s simulation** (no calibration in that run; uses `calibration_latest.json`) with the **physical Run#1** data in `utility_docs/Run1 RF data(in).csv` and the Pea RF summary sheet.

| Quantity | Physical (Run#1) | Simulation (--duration 2794) | Notes |
|----------|------------------|------------------------------|--------|
| **Run duration** | ~2794 s (10:59–11:45) | 2794 s (46.6 min) | Match |
| **Infeed mass** | 61 kg | 61.0 kg | Match |
| **Belt speed** | 0.2 m/min | 0.2 m/min | Match |
| **Initial temp** | 17.6 °C | 17.6 °C | Match |
| **Electrode gap setpoint** | 75 mm | 75 mm | Match |
| **Product temp (PLC)** | 19–101 °C over run | Outfeed avg 54.4 °C, max 79.0 °C | Sim cooler; PLC peak 101 °C during heating phase |
| **Outfeed temp (strips)** | 77–93 °C (sections 1–5) | — | Strip readings vs sim bulk outfeed |
| **Anode current Ia** | 0.01–1.72 A | Stabilizes ~1.68–1.69 A (Figure 1) | Good agreement; both near MRH 1.7 A |
| **Electrode gap actual** | Opens to ~87 mm, ends ~75 mm | Final 82.7 mm | Sim gap stays open; physical closes to setpoint at end |
| **Outfeed moisture** | NIR avg 10.45% wb | 9.79% wb | Close; sim slightly drier |
| **Throughput** | 61 kg / 46.76 min ≈ 78 kg/h (run average) | 232 kg/h (theoretical belt throughput) | Different definitions |
| **RF energy** | — | 3.7951 kWh | No physical meter in summary |
| **Specific energy** | — | 9.98 kWh/kg water | — |

**Summary**

- **Anode current** and **electrode gap dynamics** (opening under load) are in good agreement; calibration has aligned Ia and gap trajectory with the PLC.
- **Outfeed temperature**: The physical machine reaches 77–101 °C (PLC and strips); the simulation gives a lower bulk outfeed (54.4 °C avg, 79 °C max). Remaining gap can be from: (1) PLC Product_Temp vs sim bulk/outfeed definition, (2) very low calibrated k_evap (whole seeds, minimal evaporation) reducing heating, or (3) thermal inertia / cooling not fully matched.
- **Moisture**: 9.79% (sim) vs 10.45% (NIR) is close; sim is slightly drier.
- **Electrode gap at end**: Simulation ends at 82.7 mm; PLC ends at 75.2 mm. The physical controller likely returns gap to setpoint when load drops; the sim keeps the last MRH-driven position.

For tighter temperature match, options include: re-calibrating with full 2794 s window (already in `calibration_latest.json`), checking sensor location vs outfeed definition, or reviewing k_evap / coupling for whole-seed thermal response.

---

## Experimental setup vs GP-15 manual

In the actual experiment the RF was run for some time (conveyor and RF on, electrode brought to setpoint) before or at the start of the 61 kg run. The **GP-15 Installation and Operation Manual** describes the following, which affect comparison with simulation:

**Normal operation (Manual Page 57)**  
- Operator presses **RF ON**, then GP-15 is switched ON.  
- **The electrode is driven down from the max gap point to the electrode gap setpoint.**  
So there is a period at start where the machine is on and the electrode is moving down to 75 mm; the system may already be running (with or without product) before the logged "production" window.

**End of run (Manual Page 52)**  
- *"At the end of the production run **allow product to 'run-out' of the oven**, then press the GP-15 OFF button to stop processing."*  
So the physical run includes a run-out phase after the last feed; the PLC log may include this tail (e.g. falling Product_Temp at end).

**Running empty (Manual Page 58 – Optimum GP-15 machine use)**  
- *"**The longer the GP-15 machine runs empty, the less energy efficient the system becomes. At the same time the service life of the triode valve life is being reduced.**"*  
The manual therefore discourages prolonged empty running; any initial "RF on for some time" is typically short (e.g. electrode drive-down, stabilisation), not an extended empty warm-up.

**Implication for simulation**  
If in Run#1 the RF (and conveyor) were on for some time before the 61 kg feed or before the PLC log start (10:59:01), the oven and electrode could already be in a different state than "cold start" at t = 0 in the simulation. The simulation currently starts from a cold initial condition; adding an optional short "pre-run" or "RF on / electrode to setpoint" phase could better match the experimental procedure when comparing to PLC data.
