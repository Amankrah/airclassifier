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


(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/simulate_and_visualize.py --calibrate "utility_docs/Run1 RF data(in).csv" --cal-duration 0
Loading PLC data: utility_docs/Run1 RF data(in).csv
  559 samples, 2794 s
  Ia: 0.01-1.72 A
  Temp: 19-101 C

  Calibration window: 2794 s (46.6 min)
Baseline fit (before calibration):
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0
  Physics: GPU (Warp)
Module airclassifier.pretreatment.kernels.transport 21f9392 load on device 'cuda:0' took 589.90 ms  (compiled)
Module airclassifier.pretreatment.kernels.dielectric_heating 859e3c8 load on device 'cuda:0' took 245.50 ms  (compiled)
Module airclassifier.pretreatment.kernels.heat_transfer fd9ea03 load on device 'cuda:0' took 294.74 ms  (compiled)
Module airclassifier.pretreatment.kernels.drying b18a1ab load on device 'cuda:0' took 216.16 ms  (compiled)
  T_sim=43.5 vs T_plc=45.0 C
  gap_sim=82.4 vs gap_plc=75.2 mm
  loss=4.5

Running calibration optimizer...
Calibration: 559 PLC samples, 2794 s, device=auto-detect
  Normalization: var_T=612.1, var_Ia=0.3895, var_gap=43.0
  Weights: w_T=1.0, w_Ia=1.0, w_gap=1.0

  eval  10: k=0.1936 k_evap=1.66e-04 gap_rate=0.3203  L_T=2.824 L_Ia=2.088 L_gap=1.190 total=6.102
  eval  20: k=0.3350 k_evap=3.67e-04 gap_rate=0.0625  L_T=3.028 L_Ia=5.318 L_gap=29.058 total=37.404
  eval  30: k=0.1322 k_evap=1.07e-04 gap_rate=0.0928  L_T=2.894 L_Ia=0.985 L_gap=1.796 total=5.675
  eval  40: k=0.3576 k_evap=3.61e-04 gap_rate=0.5513  L_T=3.033 L_Ia=3.211 L_gap=49.992 total=56.236
  eval  50: k=0.1798 k_evap=2.76e-04 gap_rate=0.4050  L_T=2.986 L_Ia=1.867 L_gap=1.796 total=6.648
  eval  60: k=0.2462 k_evap=1.77e-05 gap_rate=0.2275  L_T=1.711 L_Ia=2.286 L_gap=11.544 total=15.541
  eval  70: k=0.2359 k_evap=3.12e-04 gap_rate=0.1897  L_T=3.002 L_Ia=2.222 L_gap=3.412 total=8.635
  eval  80: k=0.1164 k_evap=1.56e-04 gap_rate=0.2693  L_T=3.036 L_Ia=1.020 L_gap=1.796 total=5.852
  eval  90: k=0.3104 k_evap=7.92e-05 gap_rate=0.0485  L_T=2.517 L_Ia=7.146 L_gap=27.569 total=37.232
differential_evolution step 1: f(x)= 5.095931122403392
  eval 100: k=0.2139 k_evap=8.57e-05 gap_rate=0.4225  L_T=2.520 L_Ia=2.144 L_gap=2.062 total=6.726
  eval 110: k=0.1543 k_evap=1.48e-04 gap_rate=0.0547  L_T=2.896 L_Ia=1.199 L_gap=1.796 total=5.891
  eval 120: k=0.1450 k_evap=2.96e-04 gap_rate=0.1436  L_T=3.075 L_Ia=1.055 L_gap=1.796 total=5.926
  eval 130: k=0.1473 k_evap=2.70e-05 gap_rate=0.0815  L_T=2.208 L_Ia=1.164 L_gap=1.796 total=5.168
differential_evolution step 2: f(x)= 4.81577358783477
  eval 140: k=0.1326 k_evap=9.00e-05 gap_rate=0.4330  L_T=2.838 L_Ia=0.985 L_gap=1.796 total=5.619
  eval 150: k=0.1514 k_evap=4.14e-04 gap_rate=0.2041  L_T=3.109 L_Ia=1.121 L_gap=1.796 total=6.025
  eval 160: k=0.1436 k_evap=1.34e-05 gap_rate=0.1450  L_T=2.028 L_Ia=1.115 L_gap=1.796 total=4.939
  eval 170: k=0.2119 k_evap=2.57e-04 gap_rate=0.5744  L_T=2.958 L_Ia=2.123 L_gap=1.369 total=6.450
  eval 180: k=0.3211 k_evap=1.93e-05 gap_rate=0.6621  L_T=1.742 L_Ia=2.772 L_gap=49.815 total=54.329
differential_evolution step 3: f(x)= 4.682530315602346
  eval 190: k=0.1687 k_evap=6.78e-05 gap_rate=0.1115  L_T=2.440 L_Ia=1.728 L_gap=1.796 total=5.965
  eval 200: k=0.1312 k_evap=1.29e-04 gap_rate=0.2220  L_T=2.948 L_Ia=0.984 L_gap=1.796 total=5.728
  eval 210: k=0.1502 k_evap=8.11e-05 gap_rate=0.8711  L_T=2.689 L_Ia=1.159 L_gap=1.796 total=5.644
  eval 220: k=0.3852 k_evap=1.92e-04 gap_rate=0.4273  L_T=2.883 L_Ia=3.827 L_gap=74.956 total=81.665
differential_evolution step 4: f(x)= 4.682530315602346
  eval 230: k=0.1670 k_evap=9.94e-05 gap_rate=0.5395  L_T=2.669 L_Ia=1.576 L_gap=1.796 total=6.042
  eval 240: k=0.1376 k_evap=1.07e-04 gap_rate=0.2275  L_T=2.867 L_Ia=1.008 L_gap=1.796 total=5.671
  eval 250: k=0.1493 k_evap=1.37e-04 gap_rate=0.3530  L_T=2.892 L_Ia=1.121 L_gap=1.796 total=5.809
  eval 260: k=0.1524 k_evap=1.23e-05 gap_rate=0.9477  L_T=1.832 L_Ia=1.348 L_gap=1.796 total=4.975
  eval 270: k=0.1266 k_evap=3.53e-05 gap_rate=0.4738  L_T=2.572 L_Ia=0.969 L_gap=1.796 total=5.338
differential_evolution step 5: f(x)= 4.682530315602346
  eval 280: k=0.1936 k_evap=2.63e-05 gap_rate=0.9566  L_T=1.873 L_Ia=2.082 L_gap=1.323 total=5.277
  eval 290: k=0.1301 k_evap=2.90e-06 gap_rate=0.6005  L_T=2.082 L_Ia=0.970 L_gap=1.796 total=4.848
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 300: k=0.1281 k_evap=1.73e-05 gap_rate=0.7311  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
  eval 310: k=0.1392 k_evap=2.06e-04 gap_rate=0.6739  L_T=3.027 L_Ia=1.014 L_gap=1.796 total=5.838
differential_evolution step 6: f(x)= 4.682530315602346
differential_evolution step 6: f(x)= 4.682530315602346
  eval 320: k=0.1480 k_evap=7.44e-05 gap_rate=0.9127  L_T=2.666 L_Ia=1.127 L_gap=1.796 total=5.589
  eval 330: k=0.1480 k_evap=8.27e-05 gap_rate=0.1113  L_T=2.713 L_Ia=1.121 L_gap=1.796 total=5.630
  eval 330: k=0.1480 k_evap=8.27e-05 gap_rate=0.1113  L_T=2.713 L_Ia=1.121 L_gap=1.796 total=5.630
  eval 340: k=0.1351 k_evap=1.34e-05 gap_rate=0.8984  L_T=2.180 L_Ia=1.002 L_gap=1.796 total=4.978
  eval 350: k=0.1514 k_evap=4.56e-04 gap_rate=0.3714  L_T=3.119 L_Ia=1.119 L_gap=1.796 total=6.034
  eval 360: k=0.1326 k_evap=1.46e-04 gap_rate=0.4686  L_T=2.973 L_Ia=0.987 L_gap=1.796 total=5.756
differential_evolution step 7: f(x)= 4.682530315602346
  eval 370: k=0.1493 k_evap=1.86e-05 gap_rate=0.2649  L_T=2.026 L_Ia=1.230 L_gap=1.796 total=5.053
  eval 380: k=0.1432 k_evap=1.33e-05 gap_rate=0.5084  L_T=2.033 L_Ia=1.107 L_gap=1.796 total=4.936
  eval 390: k=0.1281 k_evap=1.73e-05 gap_rate=0.7713  L_T=2.356 L_Ia=0.966 L_gap=1.796 total=5.118
  eval 400: k=0.1495 k_evap=2.70e-05 gap_rate=0.2301  L_T=2.175 L_Ia=1.212 L_gap=1.796 total=5.183
differential_evolution step 8: f(x)= 4.682530315602346
  eval 410: k=0.1490 k_evap=1.03e-05 gap_rate=0.2957  L_T=1.855 L_Ia=1.249 L_gap=1.796 total=4.900
  eval 420: k=0.1460 k_evap=9.87e-05 gap_rate=0.9912  L_T=2.795 L_Ia=1.088 L_gap=1.796 total=5.679
  eval 430: k=0.1436 k_evap=7.08e-06 gap_rate=0.7047  L_T=1.891 L_Ia=1.128 L_gap=1.796 total=4.815
  eval 440: k=0.1524 k_evap=4.65e-05 gap_rate=0.2933  L_T=2.398 L_Ia=1.244 L_gap=1.796 total=5.438
  eval 450: k=0.1628 k_evap=2.97e-04 gap_rate=0.3210  L_T=3.041 L_Ia=1.333 L_gap=1.796 total=6.170
differential_evolution step 9: f(x)= 4.682530315602346
  eval 460: k=0.1535 k_evap=3.37e-04 gap_rate=0.4249  L_T=3.079 L_Ia=1.154 L_gap=1.796 total=6.029
  eval 470: k=0.1434 k_evap=4.69e-06 gap_rate=0.9634  L_T=1.838 L_Ia=1.130 L_gap=1.796 total=4.764
  eval 480: k=0.1340 k_evap=1.73e-05 gap_rate=0.6015  L_T=2.263 L_Ia=0.992 L_gap=1.796 total=5.051
  eval 490: k=0.1473 k_evap=4.79e-06 gap_rate=0.0815  L_T=1.753 L_Ia=1.224 L_gap=1.796 total=4.774
differential_evolution step 10: f(x)= 4.5815003328503305
  eval 500: k=0.1741 k_evap=2.82e-04 gap_rate=0.1281  L_T=3.005 L_Ia=1.654 L_gap=1.796 total=6.455
  eval 510: k=0.1934 k_evap=3.36e-06 gap_rate=0.0603  L_T=1.398 L_Ia=2.078 L_gap=1.760 total=5.236
  eval 520: k=0.1673 k_evap=1.30e-05 gap_rate=0.9248  L_T=1.620 L_Ia=1.904 L_gap=1.500 total=5.025
  eval 590: k=0.1690 k_evap=1.02e-04 gap_rate=0.1551  L_T=2.667 L_Ia=1.644 L_gap=1.796 total=6.107
  eval 600: k=0.2120 k_evap=1.88e-05 gap_rate=0.4177  L_T=1.733 L_Ia=2.135 L_gap=3.330 total=7.197
  eval 610: k=0.1436 k_evap=3.35e-05 gap_rate=0.2756  L_T=2.350 L_Ia=1.089 L_gap=1.796 total=5.235
  eval 620: k=0.1444 k_evap=5.04e-06 gap_rate=0.0765  L_T=1.826 L_Ia=1.149 L_gap=1.796 total=4.771
  eval 630: k=0.1463 k_evap=4.85e-04 gap_rate=0.4030  L_T=3.131 L_Ia=1.062 L_gap=1.796 total=5.989
differential_evolution step 13: f(x)= 4.441527869390225
  eval 640: k=0.1799 k_evap=8.08e-06 gap_rate=0.7540  L_T=1.507 L_Ia=2.020 L_gap=1.094 total=4.620
  eval 650: k=0.1462 k_evap=8.11e-05 gap_rate=0.7678  L_T=2.716 L_Ia=1.096 L_gap=1.796 total=5.609
  eval 660: k=0.2003 k_evap=3.83e-06 gap_rate=0.0634  L_T=1.409 L_Ia=2.096 L_gap=2.436 total=5.941
  eval 670: k=0.1473 k_evap=3.57e-04 gap_rate=0.2284  L_T=3.097 L_Ia=1.075 L_gap=1.796 total=5.968
differential_evolution step 14: f(x)= 4.441527869390225
  eval 680: k=0.1731 k_evap=4.99e-04 gap_rate=0.2957  L_T=3.101 L_Ia=1.572 L_gap=1.796 total=6.469
  eval 690: k=0.1951 k_evap=7.74e-05 gap_rate=0.0340  L_T=2.463 L_Ia=2.112 L_gap=1.128 total=5.704
  eval 700: k=0.1638 k_evap=8.03e-06 gap_rate=0.4616  L_T=1.501 L_Ia=1.872 L_gap=1.670 total=5.043
  eval 710: k=0.1772 k_evap=1.60e-04 gap_rate=0.1785  L_T=2.827 L_Ia=1.869 L_gap=1.796 total=6.492
  eval 720: k=0.1913 k_evap=1.06e-05 gap_rate=0.5480  L_T=1.561 L_Ia=2.076 L_gap=1.435 total=5.072
differential_evolution step 15: f(x)= 4.426078652982303
Polishing solution with 'L-BFGS-B'
  eval 730: k=0.1578 k_evap=1.03e-06 gap_rate=0.1268  L_T=1.387 L_Ia=1.685 L_gap=1.796 total=4.868
  eval 740: k=0.1738 k_evap=1.04e-06 gap_rate=0.1472  L_T=1.338 L_Ia=1.983 L_gap=1.107 total=4.428
  eval 750: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.335 L_Ia=1.988 L_gap=1.106 total=4.429
  eval 760: k=0.1741 k_evap=1.02e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.984 L_gap=1.103 total=4.425
  eval 770: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.984 L_gap=1.103 total=4.424
  eval 780: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.984 L_gap=1.103 total=4.424
  eval 790: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 800: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.335 L_Ia=1.989 L_gap=1.106 total=4.429
  eval 810: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 820: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 830: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 840: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 850: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 860: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 870: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 880: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 890: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 900: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 910: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424
  eval 920: k=0.1741 k_evap=1.00e-06 gap_rate=0.1475  L_T=1.338 L_Ia=1.983 L_gap=1.103 total=4.424

CalibrationResult:
  coupling_factor = 0.1741
  k_evap          = 1.00e-06
  gap_rate         = 0.1475 mm/s
  loss_total       = 4.4243
    L_temperature  = 1.3377
    L_anode_current = 1.9834
    L_gap          = 1.1032
  evaluations      = 926
  iterations       = 15
  converged        = False
  sensitivity (dL/dp):
    coupling_factor                = -6.8097
    k_evap                         = +31868.8187
    gap_rate_mm_s                  = -0.2663

Applied: coupling=0.1741, k_evap=1.00e-06, gap_rate=0.1475 mm/s
Saved to C:\Users\Windows\Desktop\Dev_Projects\airclassifier\utility_docs\calibration_latest.json (used when running without --calibrate)

============================================================
  GP-15 RF Dielectric Heating -- Simulation
============================================================

Creating GP-15 simulator ...
  Architecture: GP15Simulator -> GP15MachineAssembly
                             -> CoupledSimulator (9-step loop)
  Device:  cuda
  Physics: GPU (Warp)

  Machine:           GP-15 RF Dielectric Heating Machine
  RF zone:           1.50 m  (x = 1.46 - 2.96 m)
  Belt width:        800 mm
  Electrode gap:     75 mm
  Bed depth:         25 mm (feeder gap)
  Belt stack:        3.5 mm
  Air gap:           46 mm
  Residence time:    450.0 s
  Simulation grid:   60 x 30 x 32 = 57,600 cells
  Cell sizes:        dx=25.0 mm  dy=10.0 mm  dz=25.0 mm
  Initial moisture:  10% (wet basis)
  Initial temp:      17.6 C
  Run mass:          61.0 kg
  Throughput:        232 kg/h
  Run duration:      947 s (15.8 min)

Running LIVE simulation  |  61.0 kg  |  947 s (15.8 min)  |  belt 0.2 m/min ...
  3D window will update in real-time.


------------------------------------------------------------
  RESULTS
------------------------------------------------------------
  Outfeed moisture:          9.80%
  Outfeed temperature:       43.2 C
  Max temperature:           60.6 C
  Moisture uniformity (CV):  0.0218

  RF energy consumed:        1.2818 kWh
  Specific energy:           10.259 kWh/kg water
  Throughput:                232 kg/h
  Final electrode gap:       79.6 mm
  Mass collected (bin):      5.19 kg

  Simulation wall-clock:     1336.95 s
  Timesteps completed:       3171
  Speed:                     2 steps/s
------------------------------------------------------------

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 



(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python diag_calibrate_run2.py
Full PLC: 564 samples, 2815 s
  Ia range: 0.21-1.72 A
  Temp range: 27-103 C

Material arrival at index 78, t=390s into recording
Trimmed PLC: 486 samples, 2425 s
  Ia at t=0: 0.53 A
  Gap at t=0: 75.1 mm

PLC trajectory (from material arrival):
    t(s)   Ia(A)   Gap(mm)    T(C)
       0    0.53      75.1      38
      30    0.93      75.1      39
      60    1.48      75.1      39
      90    1.70      76.3      40
     120    1.68      77.8      40
     180    1.70      79.2      41
     300    1.70      86.0      42
     450    1.66      94.1      43
     600    1.59      94.1      46
     900    1.58      93.9      77
    1200    1.51      93.2      72

============================================================
Calibrating against 900s of Run#2 PLC data
  FDM solver: enabled (corrected)
  Material: bed=35mm, M=11.8%, T0=17°C
  Recipe from PLC: gap=75mm, speed=0.2 m/min, MRH=1.7A, MRL=1.5A
============================================================

Baseline (current calibration):
  Physics: CPU (NumPy)
  coupling=0.1741, k_evap=1.00e-06, gap_rate=0.1475
  T_sim=41.2°C vs T_plc=77°C
  Ia_sim=1.046A vs Ia_plc=1.58A
  gap_sim=105.0mm vs gap_plc=93.9mm
  loss=8.253 (T=1.994, Ia=3.551, gap=1.928)

Running differential evolution (maxiter=20, popsize=15)...
Calibration: 486 PLC samples, 900 s, device=cpu
  Normalization: var_T=342.3, var_Ia=0.0456, var_gap=53.7
  Weights: w_T=0.5, w_Ia=1.5, w_gap=1.0

  eval  10: k=0.1936 k_evap=1.66e-04 gap_rate=0.3203  L_T=1.840 L_Ia=1.593 L_gap=0.850 total=4.160
  eval  20: k=0.3350 k_evap=3.67e-04 gap_rate=0.0625  L_T=1.145 L_Ia=34.745 L_gap=6.200 total=58.889
  eval  30: k=0.1322 k_evap=1.07e-04 gap_rate=0.0928  L_T=2.309 L_Ia=3.925 L_gap=4.271 total=11.313
  eval  40: k=0.3576 k_evap=3.61e-04 gap_rate=0.5513  L_T=2.180 L_Ia=7.441 L_gap=36.061 total=48.313
  eval  50: k=0.2486 k_evap=8.19e-05 gap_rate=0.5704  L_T=1.706 L_Ia=4.060 L_gap=9.653 total=16.596
  eval  60: k=0.2044 k_evap=7.81e-05 gap_rate=0.2160  L_T=1.582 L_Ia=2.162 L_gap=1.109 total=5.143
  eval  70: k=0.2359 k_evap=2.91e-04 gap_rate=0.3304  L_T=2.262 L_Ia=4.053 L_gap=3.968 total=11.179
  eval  80: k=0.1303 k_evap=3.83e-04 gap_rate=0.1830  L_T=2.663 L_Ia=4.969 L_gap=4.271 total=13.056
  eval  90: k=0.3104 k_evap=2.19e-05 gap_rate=0.2783  L_T=2.326 L_Ia=15.275 L_gap=91.556 total=115.631
differential_evolution step 1: f(x)= 3.227525019856481
  eval 100: k=0.1812 k_evap=9.89e-05 gap_rate=0.0088  L_T=1.082 L_Ia=6.992 L_gap=2.305 total=13.333
  eval 110: k=0.1286 k_evap=2.75e-04 gap_rate=0.6339  L_T=2.626 L_Ia=5.253 L_gap=4.271 total=13.463
  eval 120: k=0.1703 k_evap=3.64e-04 gap_rate=0.6251  L_T=2.299 L_Ia=2.380 L_gap=0.952 total=5.673
  eval 130: k=0.1054 k_evap=3.97e-04 gap_rate=0.5317  L_T=2.873 L_Ia=11.435 L_gap=4.271 total=22.860
differential_evolution step 2: f(x)= 3.14328609896196
  eval 140: k=0.1429 k_evap=1.19e-04 gap_rate=0.0221  L_T=2.087 L_Ia=1.803 L_gap=4.271 total=8.019
  eval 150: k=0.1313 k_evap=1.09e-04 gap_rate=0.4333  L_T=2.239 L_Ia=3.942 L_gap=4.271 total=11.303
  eval 160: k=0.1838 k_evap=4.13e-04 gap_rate=0.4655  L_T=2.329 L_Ia=1.726 L_gap=1.068 total=4.822
  eval 170: k=0.1785 k_evap=8.66e-05 gap_rate=0.4304  L_T=1.668 L_Ia=1.244 L_gap=1.025 total=3.725
  eval 180: k=0.2146 k_evap=2.11e-04 gap_rate=0.8779  L_T=2.156 L_Ia=1.873 L_gap=1.412 total=5.300
differential_evolution step 3: f(x)= 2.415296120512104
  eval 190: k=0.1517 k_evap=1.19e-04 gap_rate=0.2129  L_T=1.868 L_Ia=0.756 L_gap=4.271 total=6.340
  eval 200: k=0.1975 k_evap=9.59e-05 gap_rate=0.5233  L_T=1.901 L_Ia=2.113 L_gap=0.852 total=4.972
  eval 210: k=0.1488 k_evap=3.42e-05 gap_rate=0.5312  L_T=1.565 L_Ia=2.031 L_gap=1.325 total=5.154
  eval 220: k=0.1837 k_evap=5.31e-05 gap_rate=0.0913  L_T=1.602 L_Ia=2.229 L_gap=0.205 total=4.349
differential_evolution step 4: f(x)= 2.415296120512104
  eval 230: k=0.1981 k_evap=1.29e-04 gap_rate=0.3066  L_T=2.037 L_Ia=2.469 L_gap=0.772 total=5.494
  eval 240: k=0.1457 k_evap=7.04e-05 gap_rate=0.2485  L_T=1.649 L_Ia=1.003 L_gap=4.271 total=6.599
  eval 250: k=0.1488 k_evap=1.46e-04 gap_rate=0.4694  L_T=2.098 L_Ia=1.182 L_gap=4.271 total=7.093
  eval 260: k=0.1515 k_evap=4.58e-05 gap_rate=0.4624  L_T=1.635 L_Ia=1.851 L_gap=1.395 total=4.989
  eval 270: k=0.1657 k_evap=2.11e-04 gap_rate=0.2328  L_T=2.056 L_Ia=0.720 L_gap=4.271 total=6.379
differential_evolution step 5: f(x)= 2.415296120512104
  eval 280: k=0.1660 k_evap=1.45e-04 gap_rate=0.6860  L_T=2.152 L_Ia=2.205 L_gap=0.899 total=5.282
  eval 290: k=0.1508 k_evap=3.19e-04 gap_rate=0.2340  L_T=2.366 L_Ia=1.344 L_gap=4.271 total=7.470
  eval 300: k=0.1814 k_evap=1.04e-04 gap_rate=0.5969  L_T=1.821 L_Ia=1.342 L_gap=1.054 total=3.978
  eval 310: k=0.1709 k_evap=1.69e-04 gap_rate=0.6284  L_T=2.248 L_Ia=2.193 L_gap=0.959 total=5.372
differential_evolution step 6: f(x)= 2.415296120512104
  eval 320: k=0.1738 k_evap=1.64e-04 gap_rate=0.5749  L_T=2.203 L_Ia=1.959 L_gap=0.968 total=5.008
  eval 330: k=0.1527 k_evap=4.90e-04 gap_rate=0.1270  L_T=2.470 L_Ia=1.324 L_gap=4.271 total=7.491
  eval 340: k=0.1586 k_evap=7.83e-05 gap_rate=0.4661  L_T=1.878 L_Ia=1.755 L_gap=1.240 total=4.812
  eval 350: k=0.1785 k_evap=3.72e-05 gap_rate=0.4272  L_T=1.495 L_Ia=2.176 L_gap=0.206 total=4.218
  eval 360: k=0.1537 k_evap=2.11e-04 gap_rate=0.0462  L_T=2.262 L_Ia=0.995 L_gap=4.271 total=6.894
differential_evolution step 7: f(x)= 2.415296120512104
  eval 370: k=0.1184 k_evap=2.52e-04 gap_rate=0.2699  L_T=2.747 L_Ia=7.827 L_gap=4.271 total=17.384
  eval 380: k=0.1632 k_evap=4.26e-04 gap_rate=0.7262  L_T=2.334 L_Ia=0.698 L_gap=4.271 total=6.486
  eval 390: k=0.1814 k_evap=3.91e-05 gap_rate=0.4201  L_T=1.489 L_Ia=2.158 L_gap=0.277 total=4.259
  eval 400: k=0.1641 k_evap=6.19e-05 gap_rate=0.8601  L_T=1.693 L_Ia=1.729 L_gap=0.936 total=4.376
differential_evolution step 8: f(x)= 2.415296120512104
  eval 410: k=0.1826 k_evap=1.01e-04 gap_rate=0.3792  L_T=1.753 L_Ia=1.292 L_gap=1.050 total=3.865
  eval 420: k=0.1562 k_evap=3.89e-04 gap_rate=0.4462  L_T=2.417 L_Ia=1.006 L_gap=4.271 total=6.989
  eval 430: k=0.1586 k_evap=9.86e-06 gap_rate=0.4661  L_T=1.068 L_Ia=0.637 L_gap=0.165 total=1.655
  eval 440: k=0.1721 k_evap=2.70e-05 gap_rate=0.2750  L_T=1.484 L_Ia=2.180 L_gap=0.127 total=4.138
  eval 450: k=0.1790 k_evap=3.75e-05 gap_rate=0.6743  L_T=1.486 L_Ia=2.226 L_gap=0.241 total=4.323
differential_evolution step 9: f(x)= 1.655378523922549
  eval 460: k=0.1557 k_evap=2.12e-05 gap_rate=0.7854  L_T=1.263 L_Ia=1.278 L_gap=0.916 total=3.465
  eval 470: k=0.1557 k_evap=1.07e-05 gap_rate=0.4337  L_T=1.128 L_Ia=0.731 L_gap=0.923 total=2.584
  eval 480: k=0.1814 k_evap=1.35e-04 gap_rate=0.4389  L_T=1.981 L_Ia=1.454 L_gap=1.035 total=4.206
  eval 490: k=0.1667 k_evap=3.31e-05 gap_rate=0.3836  L_T=1.191 L_Ia=0.925 L_gap=0.944 total=2.927
differential_evolution step 10: f(x)= 1.655378523922549
  eval 500: k=0.1629 k_evap=1.50e-04 gap_rate=0.8711  L_T=1.901 L_Ia=0.676 L_gap=4.271 total=6.236
  eval 510: k=0.1630 k_evap=3.36e-05 gap_rate=0.6520  L_T=1.261 L_Ia=1.140 L_gap=0.939 total=3.279
  eval 520: k=0.1601 k_evap=2.25e-04 gap_rate=0.2258  L_T=2.139 L_Ia=0.653 L_gap=4.271 total=6.320
  eval 530: k=0.1662 k_evap=1.17e-04 gap_rate=0.5983  L_T=2.075 L_Ia=2.153 L_gap=0.929 total=5.196
  eval 540: k=0.1790 k_evap=4.08e-04 gap_rate=0.5790  L_T=2.469 L_Ia=2.123 L_gap=1.028 total=5.447
differential_evolution step 11: f(x)= 1.655378523922549
  eval 550: k=0.1553 k_evap=2.12e-05 gap_rate=0.9634  L_T=1.265 L_Ia=1.332 L_gap=0.899 total=3.530
  eval 560: k=0.1706 k_evap=4.84e-05 gap_rate=0.4337  L_T=1.336 L_Ia=1.063 L_gap=0.978 total=3.241
  eval 570: k=0.1658 k_evap=3.14e-05 gap_rate=0.6115  L_T=1.197 L_Ia=0.964 L_gap=0.934 total=2.978
  eval 580: k=0.1644 k_evap=3.31e-05 gap_rate=0.2130  L_T=1.247 L_Ia=0.916 L_gap=0.930 total=2.927
differential_evolution step 12: f(x)= 1.655378523922549
  eval 590: k=0.1635 k_evap=2.88e-05 gap_rate=0.7610  L_T=1.204 L_Ia=1.003 L_gap=0.911 total=3.017
  eval 600: k=0.1669 k_evap=1.73e-05 gap_rate=0.7233  L_T=1.439 L_Ia=2.413 L_gap=0.130 total=4.469
  eval 610: k=0.1694 k_evap=4.66e-05 gap_rate=0.7755  L_T=1.362 L_Ia=1.141 L_gap=0.951 total=3.343
  eval 620: k=0.1596 k_evap=9.87e-06 gap_rate=0.5178  L_T=1.516 L_Ia=2.788 L_gap=0.098 total=5.038
  eval 630: k=0.1790 k_evap=1.21e-06 gap_rate=0.7383  L_T=1.593 L_Ia=3.213 L_gap=2.160 total=7.777
differential_evolution step 13: f(x)= 1.655378523922549
  eval 640: k=0.1549 k_evap=2.73e-05 gap_rate=0.5882  L_T=1.380 L_Ia=1.571 L_gap=0.909 total=3.955
  eval 650: k=0.1442 k_evap=1.38e-05 gap_rate=0.3875  L_T=1.414 L_Ia=2.097 L_gap=1.166 total=5.019
  eval 660: k=0.1422 k_evap=1.20e-04 gap_rate=0.2179  L_T=2.120 L_Ia=1.939 L_gap=4.271 total=8.240
  eval 670: k=0.1291 k_evap=2.28e-05 gap_rate=0.1239  L_T=1.404 L_Ia=2.998 L_gap=4.271 total=9.470
differential_evolution step 14: f(x)= 1.655378523922549
  eval 680: k=0.1547 k_evap=2.46e-06 gap_rate=0.5545  L_T=1.596 L_Ia=3.060 L_gap=0.087 total=5.474
  eval 690: k=0.1836 k_evap=1.10e-04 gap_rate=0.4222  L_T=1.827 L_Ia=1.330 L_gap=1.052 total=3.960
  eval 700: k=0.1517 k_evap=1.38e-05 gap_rate=0.1410  L_T=1.306 L_Ia=1.024 L_gap=0.935 total=3.123
  eval 710: k=0.1767 k_evap=4.98e-05 gap_rate=0.5440  L_T=1.268 L_Ia=1.101 L_gap=1.010 total=3.295
  eval 720: k=0.1679 k_evap=2.42e-05 gap_rate=0.6667  L_T=1.002 L_Ia=0.865 L_gap=0.954 total=2.752
differential_evolution step 15: f(x)= 1.655378523922549
  eval 730: k=0.1577 k_evap=5.72e-06 gap_rate=0.3992  L_T=1.571 L_Ia=2.861 L_gap=0.085 total=5.161
  eval 740: k=0.1593 k_evap=1.04e-05 gap_rate=0.2783  L_T=1.553 L_Ia=2.745 L_gap=0.083 total=4.977
  eval 750: k=0.1569 k_evap=6.76e-06 gap_rate=0.4887  L_T=1.575 L_Ia=2.947 L_gap=0.090 total=5.298
  eval 760: k=0.1588 k_evap=6.19e-06 gap_rate=0.3938  L_T=1.531 L_Ia=2.694 L_gap=0.093 total=4.900
differential_evolution step 16: f(x)= 1.655378523922549
  eval 770: k=0.1629 k_evap=6.34e-06 gap_rate=0.6501  L_T=1.459 L_Ia=2.331 L_gap=0.128 total=4.354
  eval 780: k=0.1552 k_evap=1.64e-05 gap_rate=0.8323  L_T=1.190 L_Ia=1.083 L_gap=0.887 total=3.107
  eval 790: k=0.1548 k_evap=9.62e-06 gap_rate=0.4431  L_T=1.161 L_Ia=0.750 L_gap=0.902 total=2.608
  eval 800: k=0.1684 k_evap=3.09e-04 gap_rate=0.5712  L_T=2.183 L_Ia=0.935 L_gap=3.719 total=6.213
  eval 810: k=0.1575 k_evap=2.69e-04 gap_rate=0.4601  L_T=2.288 L_Ia=0.817 L_gap=4.271 total=6.640
differential_evolution step 17: f(x)= 1.655378523922549
  eval 820: k=0.1557 k_evap=1.29e-05 gap_rate=0.4707  L_T=1.144 L_Ia=0.827 L_gap=0.914 total=2.727
  eval 830: k=0.1542 k_evap=1.03e-05 gap_rate=0.3875  L_T=1.177 L_Ia=0.815 L_gap=0.912 total=2.722
  eval 840: k=0.1517 k_evap=1.51e-05 gap_rate=0.2091  L_T=1.300 L_Ia=1.168 L_gap=0.935 total=3.337
  eval 850: k=0.1670 k_evap=2.75e-05 gap_rate=0.4512  L_T=1.099 L_Ia=0.830 L_gap=0.950 total=2.745
differential_evolution step 18: f(x)= 1.6408921882032474
  eval 860: k=0.1630 k_evap=2.32e-05 gap_rate=0.5791  L_T=1.126 L_Ia=0.823 L_gap=0.934 total=2.731
  eval 870: k=0.1632 k_evap=1.50e-05 gap_rate=0.4423  L_T=1.504 L_Ia=2.620 L_gap=0.105 total=4.787
  eval 880: k=0.1556 k_evap=1.80e-05 gap_rate=0.4224  L_T=1.223 L_Ia=1.057 L_gap=0.899 total=3.096
  eval 890: k=0.1694 k_evap=1.22e-05 gap_rate=0.3821  L_T=1.390 L_Ia=1.923 L_gap=0.160 total=3.741
  eval 900: k=0.1658 k_evap=2.17e-05 gap_rate=0.6016  L_T=1.021 L_Ia=0.800 L_gap=0.933 total=2.643
differential_evolution step 19: f(x)= 1.6408921882032474
  eval 910: k=0.1603 k_evap=1.08e-05 gap_rate=0.4707  L_T=1.524 L_Ia=2.763 L_gap=0.098 total=5.005
  eval 920: k=0.1576 k_evap=1.06e-05 gap_rate=0.4672  L_T=1.100 L_Ia=0.659 L_gap=0.907 total=2.445
  eval 930: k=0.1517 k_evap=1.63e-06 gap_rate=0.0874  L_T=1.944 L_Ia=2.995 L_gap=0.095 total=5.560
  eval 940: k=0.1649 k_evap=5.06e-06 gap_rate=0.5246  L_T=1.431 L_Ia=1.989 L_gap=0.147 total=3.845
differential_evolution step 20: f(x)= 1.5389264391903226
Polishing solution with 'L-BFGS-B'
  eval 950: k=0.4000 k_evap=1.00e-06 gap_rate=1.0000  L_T=2.011 L_Ia=9.223 L_gap=235.404 total=250.244
  eval 960: k=0.2426 k_evap=1.27e-05 gap_rate=0.4337  L_T=1.422 L_Ia=4.231 L_gap=17.787 total=24.845
  eval 970: k=0.1647 k_evap=1.84e-05 gap_rate=0.1535  L_T=1.550 L_Ia=2.392 L_gap=0.072 total=4.435
  eval 980: k=0.1627 k_evap=1.86e-05 gap_rate=0.1465  L_T=1.578 L_Ia=2.530 L_gap=0.071 total=4.655
  eval 990: k=0.1621 k_evap=1.86e-05 gap_rate=0.1444  L_T=1.125 L_Ia=0.606 L_gap=0.067 total=1.538
  eval 1000: k=0.1621 k_evap=6.90e-06 gap_rate=0.1444  L_T=1.624 L_Ia=2.198 L_gap=0.080 total=4.188
  eval 1010: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.066 total=1.536
  eval 1020: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1030: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.066 total=1.536
  eval 1040: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1050: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.535
  eval 1060: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1070: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.535
  eval 1080: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1090: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.535
  eval 1100: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1110: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.535
  eval 1120: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536
  eval 1130: k=0.4000 k_evap=1.00e-06 gap_rate=1.0000  L_T=2.011 L_Ia=9.223 L_gap=235.404 total=250.244
  eval 1140: k=0.1934 k_evap=1.62e-05 gap_rate=0.2568  L_T=1.445 L_Ia=2.683 L_gap=2.453 total=7.200
  eval 1150: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.535
  eval 1160: k=0.1621 k_evap=1.85e-05 gap_rate=0.1444  L_T=1.124 L_Ia=0.605 L_gap=0.065 total=1.536

============================================================
Calibration complete in 23278s (388.0 min)
============================================================
CalibrationResult:
  coupling_factor = 0.1621
  k_evap          = 1.85e-05
  gap_rate         = 0.1444 mm/s
  loss_total       = 1.5355
    L_temperature  = 1.1243
    L_anode_current = 0.6053
    L_gap          = 0.0653
  evaluations      = 1167
  iterations       = 20
  converged        = False
  sensitivity (dL/dp):
    coupling_factor                = +142.7488
    k_evap                         = -73970.5757
    gap_rate_mm_s                  = -312.8367

Saved to utility_docs\calibration_latest.json
  coupling_factor = 0.162133
  k_evap          = 1.85e-05
  gap_rate         = 0.144400 mm/s
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 












(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> Remove-Item -Recurse -Force "$env:LOCALAPPDATA\NVIDIA\warp\Cache\1.11.0"
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python diag_calibrate_run2.py
Full PLC: 564 samples, 2815 s
  Ia range: 0.21-1.72 A
  Temp range: 27-103 C

Material arrival at index 78, t=390s into recording
Trimmed PLC: 486 samples, 2425 s
  Ia at t=0: 0.53 A
  Gap at t=0: 75.1 mm

PLC trajectory (from material arrival):
    t(s)   Ia(A)   Gap(mm)    T(C)
       0    0.53      75.1      38
      30    0.93      75.1      39
      60    1.48      75.1      39
      90    1.70      76.3      40
     120    1.68      77.8      40
     180    1.70      79.2      41
     300    1.70      86.0      42
     450    1.66      94.1      43
     600    1.59      94.1      46
     900    1.58      93.9      77
    1200    1.51      93.2      72
    1500    1.54      87.7      68
    1800    1.24      75.2      70
    2100    0.31      75.2      98
    2400    0.31      75.2      45

============================================================
Calibrating against 1200s of Run#2 PLC data
  FDM solver: enabled (corrected)
  Material: bed=35mm, M=11.8%, T0=17°C
  Recipe from PLC: gap=75mm, speed=0.2 m/min, MRH=1.7A, MRL=1.5A
============================================================

Baseline (current calibration):
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0
  Physics: GPU (Warp)
Module airclassifier.pretreatment.kernels.transport 21f9392 load on device 'cuda:0' took 249.77 ms  (compiled)
Module airclassifier.pretreatment.kernels.field_solve e938513 load on device 'cuda:0' took 81.12 ms  (compiled)
Module airclassifier.pretreatment.kernels.heat_transfer fd9ea03 load on device 'cuda:0' took 71.22 ms  (compiled)
Module airclassifier.pretreatment.kernels.drying b18a1ab load on device 'cuda:0' took 62.88 ms  (compiled)
Module airclassifier.pretreatment.kernels.dielectric_heating 513d4b9 load on device 'cuda:0' took 63.99 ms  (compiled)
  coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444
  T_sim=68.2°C vs T_plc=72°C
  Ia_sim=1.647A vs Ia_plc=1.51A
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
Module airclassifier.pretreatment.kernels.dielectric_heating 513d4b9 load on device 'cuda:0' took 63.99 ms  (compiled)
  coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444
  T_sim=68.2°C vs T_plc=72°C
  Ia_sim=1.647A vs Ia_plc=1.51A
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
  coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444
  T_sim=68.2°C vs T_plc=72°C
  Ia_sim=1.647A vs Ia_plc=1.51A
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
  T_sim=68.2°C vs T_plc=72°C
  Ia_sim=1.647A vs Ia_plc=1.51A
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
  Ia_sim=1.647A vs Ia_plc=1.51A
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
  gap_sim=85.0mm vs gap_plc=93.2mm
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)
  loss=6.824 (T=0.966, Ia=3.435, gap=1.188)
  Baseline eval: 30.7 s  →  est. ~51 min (100 evals) to ~153 min (300 evals)

Running calibration (method=nelder-mead, maxiter=150)...

Running calibration (method=nelder-mead, maxiter=150)...
Calibration: 486 PLC samples, 1200 s, device=auto-detect
  Normalization: var_T=322.6, var_Ia=0.0375, var_gap=45.9
Running calibration (method=nelder-mead, maxiter=150)...
Calibration: 486 PLC samples, 1200 s, device=auto-detect
  Normalization: var_T=322.6, var_Ia=0.0375, var_gap=45.9
  Weights: w_T=0.5, w_Ia=1.5, w_gap=1.0
Calibration: 486 PLC samples, 1200 s, device=auto-detect
  Normalization: var_T=322.6, var_Ia=0.0375, var_gap=45.9
  Weights: w_T=0.5, w_Ia=1.5, w_gap=1.0
  Normalization: var_T=322.6, var_Ia=0.0375, var_gap=45.9
  Weights: w_T=0.5, w_Ia=1.5, w_gap=1.0

  Weights: w_T=0.5, w_Ia=1.5, w_gap=1.0


  Estimated max evals: 300 (ETA after first evals)
  Method: Nelder-Mead from baseline (coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444)
  eval  10: k=0.1832 k_evap=1.93e-05 gap_rate=0.1310  L_T=1.112 L_Ia=2.567 L_gap=0.397 total=4.804  (33.5s  ETA ~140.1 min)
  Estimated max evals: 300 (ETA after first evals)
  Method: Nelder-Mead from baseline (coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444)
  eval  10: k=0.1832 k_evap=1.93e-05 gap_rate=0.1310  L_T=1.112 L_Ia=2.567 L_gap=0.397 total=4.804  (33.5s  ETA ~140.1 min)
  Method: Nelder-Mead from baseline (coupling=0.1621, k_evap=1.85e-05, gap_rate=0.1444)
  eval  10: k=0.1832 k_evap=1.93e-05 gap_rate=0.1310  L_T=1.112 L_Ia=2.567 L_gap=0.397 total=4.804  (33.5s  ETA ~140.1 min)
  eval  20: k=0.1780 k_evap=1.93e-05 gap_rate=0.1380  L_T=1.579 L_Ia=1.688 L_gap=0.285 total=3.606  (33.7s  ETA ~136.0 min)
  eval  30: k=0.1809 k_evap=1.93e-05 gap_rate=0.1342  L_T=1.298 L_Ia=1.720 L_gap=0.363 total=3.591  (32.4s  ETA ~136.7 min)
  eval  10: k=0.1832 k_evap=1.93e-05 gap_rate=0.1310  L_T=1.112 L_Ia=2.567 L_gap=0.397 total=4.804  (33.5s  ETA ~140.1 min)
  eval  20: k=0.1780 k_evap=1.93e-05 gap_rate=0.1380  L_T=1.579 L_Ia=1.688 L_gap=0.285 total=3.606  (33.7s  ETA ~136.0 min)
  eval  30: k=0.1809 k_evap=1.93e-05 gap_rate=0.1342  L_T=1.298 L_Ia=1.720 L_gap=0.363 total=3.591  (32.4s  ETA ~136.7 min)
  eval  40: k=0.1807 k_evap=1.92e-05 gap_rate=0.1368  L_T=1.381 L_Ia=1.738 L_gap=0.354 total=3.650  (38.5s  ETA ~134.2 min)
  eval  20: k=0.1780 k_evap=1.93e-05 gap_rate=0.1380  L_T=1.579 L_Ia=1.688 L_gap=0.285 total=3.606  (33.7s  ETA ~136.0 min)
  eval  30: k=0.1809 k_evap=1.93e-05 gap_rate=0.1342  L_T=1.298 L_Ia=1.720 L_gap=0.363 total=3.591  (32.4s  ETA ~136.7 min)
  eval  40: k=0.1807 k_evap=1.92e-05 gap_rate=0.1368  L_T=1.381 L_Ia=1.738 L_gap=0.354 total=3.650  (38.5s  ETA ~134.2 min)
  eval  30: k=0.1809 k_evap=1.93e-05 gap_rate=0.1342  L_T=1.298 L_Ia=1.720 L_gap=0.363 total=3.591  (32.4s  ETA ~136.7 min)
  eval  40: k=0.1807 k_evap=1.92e-05 gap_rate=0.1368  L_T=1.381 L_Ia=1.738 L_gap=0.354 total=3.650  (38.5s  ETA ~134.2 min)
  eval  50: k=0.1811 k_evap=1.92e-05 gap_rate=0.1365  L_T=1.233 L_Ia=1.637 L_gap=0.370 total=3.443  (32.0s  ETA ~132.9 min)
  eval  40: k=0.1807 k_evap=1.92e-05 gap_rate=0.1368  L_T=1.381 L_Ia=1.738 L_gap=0.354 total=3.650  (38.5s  ETA ~134.2 min)
  eval  50: k=0.1811 k_evap=1.92e-05 gap_rate=0.1365  L_T=1.233 L_Ia=1.637 L_gap=0.370 total=3.443  (32.0s  ETA ~132.9 min)
  eval  50: k=0.1811 k_evap=1.92e-05 gap_rate=0.1365  L_T=1.233 L_Ia=1.637 L_gap=0.370 total=3.443  (32.0s  ETA ~132.9 min)
  eval  60: k=0.1811 k_evap=1.92e-05 gap_rate=0.1365  L_T=1.248 L_Ia=1.800 L_gap=0.367 total=3.692  (32.4s  ETA ~127.8 min)
  eval  70: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.262 L_Ia=1.619 L_gap=0.365 total=3.424  (30.2s  ETA ~123.6 min)
  eval  60: k=0.1811 k_evap=1.92e-05 gap_rate=0.1365  L_T=1.248 L_Ia=1.800 L_gap=0.367 total=3.692  (32.4s  ETA ~127.8 min)
  eval  70: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.262 L_Ia=1.619 L_gap=0.365 total=3.424  (30.2s  ETA ~123.6 min)
  eval  80: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.654 L_gap=0.366 total=3.478  (30.9s  ETA ~117.6 min)
  eval  70: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.262 L_Ia=1.619 L_gap=0.365 total=3.424  (30.2s  ETA ~123.6 min)
  eval  80: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.654 L_gap=0.366 total=3.478  (30.9s  ETA ~117.6 min)
  eval  90: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.366 total=3.419  (39.7s  ETA ~112.6 min)
  eval 100: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.365 total=3.419  (41.2s  ETA ~109.8 min)
  eval  80: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.654 L_gap=0.366 total=3.478  (30.9s  ETA ~117.6 min)
  eval  90: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.366 total=3.419  (39.7s  ETA ~112.6 min)
  eval 100: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.365 total=3.419  (41.2s  ETA ~109.8 min)

  eval  90: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.366 total=3.419  (39.7s  ETA ~112.6 min)
  eval 100: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.365 total=3.419  (41.2s  ETA ~109.8 min)

  eval 100: k=0.1811 k_evap=1.92e-05 gap_rate=0.1364  L_T=1.264 L_Ia=1.614 L_gap=0.365 total=3.419  (41.2s  ETA ~109.8 min)

  Calibration wall time: 3687 s (61.4 min), 109 evals, ~33.4 s/eval


  Calibration wall time: 3687 s (61.4 min), 109 evals, ~33.4 s/eval

============================================================
  Calibration wall time: 3687 s (61.4 min), 109 evals, ~33.4 s/eval

============================================================

============================================================
Calibration complete in 3718s (62.0 min)
============================================================
Calibration complete in 3718s (62.0 min)
Calibration complete in 3718s (62.0 min)
============================================================
CalibrationResult:
============================================================
CalibrationResult:
  coupling_factor = 0.1811
CalibrationResult:
  coupling_factor = 0.1811
  coupling_factor = 0.1811
  k_evap          = 1.92e-05
  gap_rate         = 0.1364 mm/s
  k_evap          = 1.92e-05
  gap_rate         = 0.1364 mm/s
  loss_total       = 3.4188
  gap_rate         = 0.1364 mm/s
  loss_total       = 3.4188
  loss_total       = 3.4188
    L_temperature  = 1.2651
    L_anode_current = 1.6776
    L_gap          = 0.3654
  evaluations      = 109
  iterations       = 43
  converged        = True
  sensitivity (dL/dp):
    coupling_factor                = +606.7023
    k_evap                         = -315191.6649
    gap_rate_mm_s                  = -13.0989

Saved to utility_docs\calibration_latest.json
  coupling_factor = 0.181097
  k_evap          = 1.92e-05
  gap_rate         = 0.136421 mm/s
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier>