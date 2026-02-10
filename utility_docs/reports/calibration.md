# Calibration report

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

## Example run (300 s calibration)

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/simulate_and_visualize.py --calibrate "utility_docs/Run1 RF data(in).csv"
Loading PLC data: utility_docs/Run1 RF data(in).csv
  559 samples, 2794 s
  Ia: 0.01-1.72 A
  Temp: 19-101 C

Baseline fit (before calibration):
  T_sim=30.5 vs T_plc=31.0 C
  gap_sim=103.4 vs gap_plc=75.2 mm
  loss=161.9

Running calibration optimizer...
Starting calibration against PLC data (559 samples, 2794 s)
  Sim duration: 300 s
  Compare points: 50
  Bounds: coupling=[(0.1, 0.4)], k_evap=[(1e-06, 0.0005)], gap_rate=[(0.01, 1.0)]

  eval  10: k=0.2538 k_evap=2.00e-04 gap_rate=0.1173  L_T=19.1 L_gap=286.8 total=162.5
  eval  20: k=0.1892 k_evap=1.53e-04 gap_rate=0.7100  L_T=21.4 L_gap=25.1 total=33.9
  eval  30: k=0.2650 k_evap=4.74e-04 gap_rate=0.5667  L_T=31.2 L_gap=501.8 total=282.2
  eval  40: k=0.3719 k_evap=4.15e-04 gap_rate=0.1201  L_T=15.6 L_gap=451.0 total=241.1
differential_evolution step 1: f(x)= 33.89707213698678
  eval  50: k=0.3960 k_evap=9.35e-05 gap_rate=0.4443  L_T=5.2 L_gap=3220.0 total=1615.2
  eval  60: k=0.1748 k_evap=2.14e-04 gap_rate=0.6370  L_T=28.0 L_gap=21.4 total=38.8
  eval  70: k=0.1197 k_evap=7.17e-05 gap_rate=0.3917  L_T=43.3 L_gap=21.4 total=54.1
differential_evolution step 2: f(x)= 26.59350596910513
  eval  80: k=0.1793 k_evap=3.00e-04 gap_rate=0.4383  L_T=30.5 L_gap=21.4 total=41.2
  eval  90: k=0.2453 k_evap=3.09e-04 gap_rate=0.8702  L_T=28.1 L_gap=309.0 total=182.6
differential_evolution step 3: f(x)= 21.697459926818084
  eval 100: k=0.2241 k_evap=2.18e-04 gap_rate=0.5854  L_T=24.4 L_gap=146.9 total=97.9
  eval 110: k=0.2281 k_evap=9.95e-05 gap_rate=0.8881  L_T=15.7 L_gap=200.1 total=115.7
  eval 120: k=0.2436 k_evap=2.60e-04 gap_rate=0.1066  L_T=23.4 L_gap=212.4 total=129.7
differential_evolution step 4: f(x)= 21.697459926818084
  eval 130: k=0.2143 k_evap=1.27e-04 gap_rate=0.4826  L_T=18.4 L_gap=102.5 total=69.7
  eval 140: k=0.2042 k_evap=8.77e-05 gap_rate=0.7007  L_T=15.0 L_gap=64.8 total=47.4
differential_evolution step 5: f(x)= 21.697459926818084
  eval 150: k=0.2179 k_evap=1.77e-05 gap_rate=0.2090  L_T=8.9 L_gap=163.2 total=90.5
  eval 160: k=0.2084 k_evap=1.68e-04 gap_rate=0.3788  L_T=21.8 L_gap=70.3 total=56.9
differential_evolution step 6: f(x)= 21.697459926818084
  eval 170: k=0.1451 k_evap=1.92e-04 gap_rate=0.4103  L_T=36.4 L_gap=21.4 total=47.1
  eval 180: k=0.2044 k_evap=1.96e-04 gap_rate=0.6096  L_T=23.7 L_gap=54.1 total=50.8
  eval 190: k=0.2115 k_evap=1.22e-04 gap_rate=0.0700  L_T=15.5 L_gap=64.4 total=47.7
differential_evolution step 7: f(x)= 21.697459926818084
  eval 200: k=0.1990 k_evap=1.10e-04 gap_rate=0.0216  L_T=14.9 L_gap=27.9 total=28.9
  eval 210: k=0.2007 k_evap=1.20e-04 gap_rate=0.1834  L_T=18.0 L_gap=48.2 total=42.1
differential_evolution step 8: f(x)= 21.58733550985906
  eval 220: k=0.1827 k_evap=2.19e-05 gap_rate=0.1260  L_T=11.4 L_gap=26.0 total=24.5
  eval 230: k=0.1900 k_evap=2.85e-05 gap_rate=0.1875  L_T=10.6 L_gap=36.0 total=28.6
  eval 240: k=0.2079 k_evap=1.27e-05 gap_rate=0.0689  L_T=10.0 L_gap=78.9 total=49.5
differential_evolution step 9: f(x)= 21.58733550985906
  eval 250: k=0.2037 k_evap=1.34e-05 gap_rate=0.1857  L_T=10.1 L_gap=85.4 total=52.8
  eval 260: k=0.2177 k_evap=1.75e-05 gap_rate=0.8551  L_T=9.8 L_gap=165.1 total=92.4
differential_evolution step 10: f(x)= 21.58733550985906
  eval 270: k=0.2331 k_evap=3.30e-05 gap_rate=0.3893  L_T=8.8 L_gap=274.9 total=146.2
  eval 280: k=0.2275 k_evap=1.84e-05 gap_rate=0.0620  L_T=10.1 L_gap=133.0 total=76.5
differential_evolution step 11: f(x)= 21.58733550985906
  eval 290: k=0.1843 k_evap=2.40e-05 gap_rate=0.8676  L_T=11.2 L_gap=27.9 total=25.2
  eval 300: k=0.2096 k_evap=5.71e-05 gap_rate=0.0190  L_T=8.2 L_gap=28.6 total=22.5
  eval 310: k=0.2116 k_evap=2.82e-04 gap_rate=0.1616  L_T=27.2 L_gap=73.1 total=63.7
differential_evolution step 12: f(x)= 21.58733550985906
  eval 320: k=0.2107 k_evap=1.13e-04 gap_rate=0.1815  L_T=16.8 L_gap=85.9 total=59.7
  eval 330: k=0.1850 k_evap=2.08e-05 gap_rate=0.1516  L_T=11.1 L_gap=28.9 total=25.5
differential_evolution step 13: f(x)= 21.58733550985906
  eval 340: k=0.2119 k_evap=2.41e-04 gap_rate=0.1118  L_T=24.9 L_gap=69.2 total=59.5
  eval 350: k=0.2373 k_evap=1.40e-05 gap_rate=0.6889  L_T=9.4 L_gap=342.5 total=180.6
  eval 360: k=0.1915 k_evap=1.86e-05 gap_rate=0.7571  L_T=10.4 L_gap=42.0 total=31.4
differential_evolution step 14: f(x)= 21.58733550985906
  eval 370: k=0.2038 k_evap=7.01e-05 gap_rate=0.1240  L_T=13.0 L_gap=65.8 total=45.9
  eval 380: k=0.2087 k_evap=6.53e-05 gap_rate=0.8212  L_T=12.8 L_gap=88.9 total=57.2
differential_evolution step 15: f(x)= 21.251560574910847

CalibrationResult:
  coupling_factor = 0.2014
  k_evap          = 5.90e-05
  gap_rate         = 0.0123 mm/s
  loss_total       = 21.2516
  loss_temperature = 9.3683
  loss_gap         = 23.7665
  iterations       = 15
  converged        = False

Applied: coupling=0.2014, k_evap=5.90e-05

============================================================
  GP-15 RF Dielectric Heating -- Simulation
============================================================

Creating GP-15 simulator ...
  Architecture: GP15Simulator -> GP15MachineAssembly
                             -> CoupledSimulator (9-step loop)
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0
  Device:  cuda

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
  Outfeed moisture:          8.27%
  Outfeed temperature:       31.3 C
  Max temperature:           40.1 C
  Moisture uniformity (CV):  0.0441

  RF energy consumed:        1.2895 kWh
  Specific energy:           1.222 kWh/kg water
  Throughput:                232 kg/h
  Final electrode gap:       84.1 mm
  Mass collected (bin):      5.19 kg

  Simulation wall-clock:     34.70 s
  Timesteps completed:       3171
  Speed:                     91 steps/s
------------------------------------------------------------

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 


After first improvement:

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
  T_sim=31.1 vs T_plc=45.0 C
  gap_sim=84.1 vs gap_plc=75.2 mm
  loss=1444.5

Running calibration optimizer...
Starting calibration against PLC data (559 samples, 2794 s)
  Sim duration: 2794 s
  Device: auto-detect
  Compare points: 50
  Bounds: coupling=[(0.1, 0.4)], k_evap=[(1e-06, 0.0005)], gap_rate=[(0.01, 1.0)]

  eval  10: k=0.2538 k_evap=2.00e-04 gap_rate=0.1173  L_T=1771.1 L_gap=301.8 total=1922.0
  eval  20: k=0.1892 k_evap=1.53e-04 gap_rate=0.7100  L_T=1712.7 L_gap=55.8 total=1740.6
  eval  30: k=0.2795 k_evap=3.86e-05 gap_rate=0.7009  L_T=1265.8 L_gap=939.9 total=1735.7
  eval  40: k=0.3766 k_evap=4.75e-04 gap_rate=0.7848  L_T=1887.5 L_gap=2704.9 total=3240.0
differential_evolution step 1: f(x)= 1292.8952714901611
  eval  50: k=0.2542 k_evap=2.12e-04 gap_rate=0.2331  L_T=1782.2 L_gap=309.3 total=1936.8
  eval  60: k=0.1580 k_evap=3.97e-04 gap_rate=0.2765  L_T=1897.8 L_gap=77.3 total=1936.4
  eval  70: k=0.2751 k_evap=2.35e-05 gap_rate=0.9669  L_T=1118.9 L_gap=952.2 total=1595.0
differential_evolution step 2: f(x)= 988.6707097718651
  eval  80: k=0.1940 k_evap=1.02e-05 gap_rate=0.6165  L_T=952.2 L_gap=69.1 total=986.8
  eval  90: k=0.2796 k_evap=3.26e-04 gap_rate=0.6541  L_T=1847.4 L_gap=582.8 total=2138.8
differential_evolution step 3: f(x)= 986.7959020174382
  eval 100: k=0.1966 k_evap=2.02e-04 gap_rate=0.5924  L_T=1772.5 L_gap=49.8 total=1797.4
  eval 110: k=0.2620 k_evap=3.63e-04 gap_rate=0.9056  L_T=1860.4 L_gap=362.4 total=2041.6
  eval 120: k=0.1670 k_evap=1.84e-05 gap_rate=0.3621  L_T=1056.6 L_gap=70.9 total=1092.1
differential_evolution step 4: f(x)= 965.7904626558379
  eval 130: k=0.2339 k_evap=8.76e-06 gap_rate=0.0665  L_T=993.8 L_gap=426.0 total=1206.8
  eval 140: k=0.2117 k_evap=4.55e-04 gap_rate=0.3235  L_T=1883.8 L_gap=56.5 total=1912.0
differential_evolution step 5: f(x)= 965.7904626558379
  eval 150: k=0.1911 k_evap=9.67e-06 gap_rate=0.7009  L_T=941.3 L_gap=61.1 total=971.9
  eval 160: k=0.2135 k_evap=8.43e-05 gap_rate=0.4522  L_T=1539.4 L_gap=87.4 total=1583.1
differential_evolution step 6: f(x)= 965.7904626558379
  eval 170: k=0.2266 k_evap=2.25e-04 gap_rate=0.3327  L_T=1792.4 L_gap=110.2 total=1847.5
  eval 180: k=0.2243 k_evap=1.35e-05 gap_rate=0.3080  L_T=993.6 L_gap=254.4 total=1120.8
  eval 190: k=0.2148 k_evap=2.35e-05 gap_rate=0.3723  L_T=1115.1 L_gap=151.1 total=1190.7
differential_evolution step 7: f(x)= 922.6433178900511
  eval 200: k=0.2069 k_evap=1.44e-05 gap_rate=0.1604  L_T=1004.6 L_gap=119.3 total=1064.2
  eval 210: k=0.1707 k_evap=1.40e-05 gap_rate=0.3580  L_T=1001.3 L_gap=57.9 total=1030.2
differential_evolution step 8: f(x)= 872.8180332028586
  eval 220: k=0.1319 k_evap=1.57e-05 gap_rate=0.5982  L_T=1394.9 L_gap=77.3 total=1433.5
  eval 230: k=0.1969 k_evap=6.66e-06 gap_rate=0.7593  L_T=900.8 L_gap=83.3 total=942.5
  eval 240: k=0.2006 k_evap=3.19e-06 gap_rate=0.3299  L_T=849.2 L_gap=105.5 total=902.0
differential_evolution step 9: f(x)= 851.464372299085
  eval 250: k=0.1510 k_evap=9.08e-06 gap_rate=0.4964  L_T=1093.2 L_gap=77.3 total=1131.9
  eval 260: k=0.2011 k_evap=1.50e-04 gap_rate=0.6784  L_T=1708.4 L_gap=49.5 total=1733.2
differential_evolution step 10: f(x)= 851.464372299085
  eval 270: k=0.1583 k_evap=1.58e-04 gap_rate=0.7906  L_T=1780.0 L_gap=77.3 total=1818.7
  eval 280: k=0.2208 k_evap=1.18e-05 gap_rate=0.8513  L_T=975.0 L_gap=228.3 total=1089.1
differential_evolution step 11: f(x)= 851.464372299085
  eval 290: k=0.1810 k_evap=3.73e-06 gap_rate=0.6488  L_T=860.7 L_gap=47.8 total=884.6
  eval 300: k=0.1691 k_evap=2.83e-04 gap_rate=0.2598  L_T=1850.7 L_gap=77.3 total=1889.4
  eval 310: k=0.2148 k_evap=6.62e-05 gap_rate=0.5931  L_T=1455.8 L_gap=102.9 total=1507.3
differential_evolution step 12: f(x)= 851.464372299085
  eval 320: k=0.1887 k_evap=2.49e-04 gap_rate=0.4046  L_T=1810.3 L_gap=61.6 total=1841.1
  eval 330: k=0.2057 k_evap=3.61e-06 gap_rate=0.9019  L_T=853.8 L_gap=133.6 total=920.7
differential_evolution step 13: f(x)= 838.334952507118
  eval 340: k=0.1973 k_evap=1.21e-06 gap_rate=0.9902  L_T=825.9 L_gap=94.9 total=873.4
  eval 350: k=0.1644 k_evap=1.30e-06 gap_rate=0.7333  L_T=822.6 L_gap=63.3 total=854.2
  eval 360: k=0.1841 k_evap=3.52e-04 gap_rate=0.0395  L_T=1857.4 L_gap=77.3 total=1896.1
differential_evolution step 14: f(x)= 838.334952507118
  eval 370: k=0.1825 k_evap=3.31e-06 gap_rate=0.7087  L_T=849.3 L_gap=49.2 total=873.9
  eval 380: k=0.1644 k_evap=1.05e-06 gap_rate=0.5191  L_T=811.8 L_gap=64.4 total=844.0
differential_evolution step 15: f(x)= 838.334952507118

CalibrationResult:
  coupling_factor = 0.1810
  k_evap          = 1.01e-06
  gap_rate         = 0.6488 mm/s
  loss_total       = 838.3350
  loss_temperature = 814.1130
  loss_gap         = 48.4439
  iterations       = 15
  converged        = False

Applied: coupling=0.1810, k_evap=1.01e-06

============================================================
  GP-15 RF Dielectric Heating -- Simulation
============================================================

Creating GP-15 simulator ...
  Architecture: GP15Simulator -> GP15MachineAssembly
                             -> CoupledSimulator (9-step loop)
  Device:  cuda

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
  Outfeed moisture:          9.77%
  Outfeed temperature:       57.6 C
  Max temperature:           84.1 C
  Moisture uniformity (CV):  0.0243

  RF energy consumed:        1.3966 kWh
  Specific energy:           9.823 kWh/kg water
  Throughput:                232 kg/h
  Final electrode gap:       83.4 mm
  Mass collected (bin):      5.19 kg

  Simulation wall-clock:     746.56 s
  Timesteps completed:       3171
  Speed:                     4 steps/s
------------------------------------------------------------

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 


