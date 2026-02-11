(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================

======================================================================
GRID SEARCH OPTIMIZATION — WHEEL-ONLY
  Blower RPM: 400 – 800 (4 points)
  Wheel RPM:  1000 – 4000 (4 points)
  Total trials: 16
  Objective: protein_recovery
  Particles: 50000, Time: 240.0s
======================================================================

  [  1/16] Blower=400 RPM, Wheel=1000 RPM ... Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 1.89 ms  (cached)
OK  prot=0.088  starch=0.676  purity=0.538  score=0.0879  (39.0s)
  [  2/16] Blower=400 RPM, Wheel=2000 RPM ... OK  prot=0.000  starch=0.859  purity=1.000  score=0.0000  (38.7s)
  [  3/16] Blower=400 RPM, Wheel=3000 RPM ... OK  prot=0.001  starch=0.859  purity=1.000  score=0.0009  (39.2s)
  [  4/16] Blower=400 RPM, Wheel=4000 RPM ... OK  prot=0.000  starch=0.859  purity=0.000  score=0.0000  (39.2s)
  [  5/16] Blower=533 RPM, Wheel=1000 RPM ... OK  prot=0.167  starch=0.636  purity=0.581  score=0.1670  (39.2s)
  [  6/16] Blower=533 RPM, Wheel=2000 RPM ... OK  prot=0.047  starch=0.849  purity=1.000  score=0.0469  (39.2s)
  [  7/16] Blower=533 RPM, Wheel=3000 RPM ... OK  prot=0.051  starch=0.859  purity=1.000  score=0.0509  (39.2s)
  [  8/16] Blower=533 RPM, Wheel=4000 RPM ... OK  prot=0.000  starch=0.859  purity=0.000  score=0.0000  (39.0s)
  [  9/16] Blower=667 RPM, Wheel=1000 RPM ... OK  prot=0.294  starch=0.602  purity=0.709  score=0.2944  (39.1s)
  [ 10/16] Blower=667 RPM, Wheel=2000 RPM ... OK  prot=0.147  starch=0.837  purity=1.000  score=0.1471  (38.7s)
  [ 11/16] Blower=667 RPM, Wheel=3000 RPM ... OK  prot=0.084  starch=0.859  purity=1.000  score=0.0842  (39.1s)
  [ 12/16] Blower=667 RPM, Wheel=4000 RPM ... OK  prot=0.047  starch=0.859  purity=1.000  score=0.0474  (39.3s)
  [ 13/16] Blower=800 RPM, Wheel=1000 RPM ... OK  prot=0.264  starch=0.572  purity=0.769  score=0.2643  (39.4s)
  [ 14/16] Blower=800 RPM, Wheel=2000 RPM ... OK  prot=0.175  starch=0.822  purity=0.996  score=0.1748  (39.1s)
  [ 15/16] Blower=800 RPM, Wheel=3000 RPM ... OK  prot=0.107  starch=0.859  purity=1.000  score=0.1065  (39.1s)
  [ 16/16] Blower=800 RPM, Wheel=4000 RPM ... OK  prot=0.060  starch=0.859  purity=1.000  score=0.0604  (39.2s)

======================================================================
OPTIMIZATION COMPLETE — GRID — WHEEL-ONLY
======================================================================
  Objective:       protein_recovery
  Trials:          16
  Total wall time: 626.9 s (10.4 min)
  Best score:      0.2944

  ──────────────────────────────────────────────────
  BEST CONFIGURATION:
  ──────────────────────────────────────────────────
    Mode:           wheel-only
    Blower RPM:     667
    Wheel RPM:      1000

  RESULTS:
    Protein recovery:     0.294  (29.4%)
    Starch yield:         0.602  (60.2%)
    Protein purity:       0.709  (70.9%)
    Separation eff.:      0.392
    Total collection:     0.989  (98.9%)

  COLLECTION BREAKDOWN (of 50000 particles):
    Wheel coarse (starch)      30087 ( 60.2%) |##############################
    Cyclone 1 (fines)           4302 (  8.6%) |####
    Cyclone 2 (fines)            363 (  0.7%) |
    Cyclone 3 (protein)        14523 ( 29.0%) |##############
    Bag filter                   196 (  0.4%) |
    Escaped                        0 (  0.0%) |
    Still active                 529 (  1.1%) |

  ──────────────────────────────────────────────────
  TOP 5 CONFIGURATIONS:
  ──────────────────────────────────────────────────
  Rank   Blower    Wheel  ProtRec  StchYld   Purity  Collect    Score
  ────  ───────  ───────  ───────  ───────  ───────  ───────  ───────
     1      667     1000    0.294    0.602    0.709    0.989   0.2944
     2      800     1000    0.264    0.572    0.769    1.000   0.2643
     3      800     2000    0.175    0.822    0.996    1.000   0.1748
     4      533     1000    0.167    0.636    0.581    0.868   0.1670
     5      667     2000    0.147    0.837    1.000    0.984   0.1471

======================================================================

  Reproduce best with:
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 667 --wheel-rpm 1000 --wheel-only

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier>  python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 667 --wheel-rpm 1000 --wheel-only
======================================================================
PHYSICS-BASED CLASSIFICATION FLOW SIMULATION
  Protein/Starch Separation via Air Classification
======================================================================
  VFD: 667 RPM -> 472 m³/h (operating point)
       Fan law would give 667 m³/h (overestimate)
       P = 124 Pa, eff = 68.6%, W = 24 W
  Wheel RPM (main classifier): 1000 (override)

[FULL SYSTEM] Air -> Wheel (no preclassification) -> Cyclones -> Bag Filter
  [Mode] Without preclassification (wheel-only): no venturi, zigzag, dropout

1. Air system -> Wheel junction (airclass, no venturi)...
============================================================
AIR SYSTEM -> VENTURI DUCTWORK FLOW
============================================================
  Volume flow:    471.6 m³/h
  Total dP:       5.6 Pa
  Segments:      4
------------------------------------------------------------
  duct_0             duct       v=  4.17 m/s  Re=   55024  dP=   0.1 Pa
  elbow_1            elbow      v=  4.17 m/s  Re=   55024  dP=   3.1 Pa
  duct_2             duct       v=  4.17 m/s  Re=   55024  dP=   0.8 Pa
  transition_3       transition v=  4.17 m/s  Re=   55024  dP=   1.6 Pa
============================================================

2. Feed system -> Wheel junction solids inlet (feedclass, no venturi)...
======================================================================
FEED SYSTEM -> VENTURI SOLIDS INLET (ductwork flow + kinetics)
======================================================================
  Air flow (sweep):  0.00 m3/h
  Particle d:        35.0 um
  Terminal v (vert): 0.0502 m/s
  Total dP:          0.00 Pa
  Total residence:   21.906 s
  Venturi solids D:  0 mm
----------------------------------------------------------------------
  feed_duct_0            duct       L=1.099m v_air=0.00 v_part=0.05 t_res=21.906s dP=0.0Pa
======================================================================

  Material / feed properties (used for classification validation):
    Material:        yellow_pea (density=1420 kg/m³, sphericity=0.70)
    Size range:      2.0 – 500.0 µm   d50=35.0 µm
    Feed rep. d:     35.0 µm (feed ductwork and entry rate)
    Particle rate:   4356888079 particles/s (solids mass flow + rep. d)

3. Wheel-only classification (no venturi/zigzag):
   Air flow:           472 m³/h (0.131 m³/s)
   Solids mass flow:   500.0 kg/h
  [Feed cap] Requested 184411.1 kg/h exceeds venturi capacity 1135.7 kg/h (mu=2.0)
             Capped to 1135.7 kg/h
  [Continuous feed] 100000 sim particles over 120s = 833 particles/s
  [Physical rate] 1135.7 kg/h (capped at mu=2.0)

Creating classification system assembly (from full system)...
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0

  *** VENTURI COMPRESSIBILITY WARNING ***
      Throat velocity: 104 m/s
      Mach number:     0.30 (>0.3 - compressible regime)
      Bernoulli (incompressible) approximation has >5% error
      Max choked flow: 1528 m3/h

    Wheel Classifier (main classifier - centrifugal):
      Diameter:        200 mm
      RPM:             1000
      Tip speed:       10.5 m/s
      G-force (rim):   112 g
      d50:             34.5 um
      Hub radius:      30.0 mm
      Blades:          24

    Particle Entry: Zone 34 (wheel housing - wheel-only mode)

  Classification Physics Parameters:

    Air Flow:
      Flow rate:       472 m3/h
      Venturi inlet:   66.7 m/s
      Venturi throat:  104.3 m/s (D=40.0mm, Ma=0.304)
      Venturi dP:      3863 Pa (3.9 kPa)
      Zigzag bulk:     0.00 m/s
      Zigzag ZONE:     0.00 m/s (30% of bulk)
      Cyclone (series, rectangular tangential inlet):
        Primary    D=300mm  inlet=75x150mm  v=11.6 m/s (weak vortex)
        Secondary  D=200mm  inlet=50x100mm  v=26.2 m/s
        Tertiary   D=120mm  inlet=30x60mm  v=72.8 m/s

    Venturi Throat Analysis:
      Throat diameter: 40.0 mm
      Throat area:     1256.6 mm2
      Max flow (Ma=1): 1528 m3/h
      K_venturi:       225072.9 Pa/(m3/s)2
      *** Ma=0.30 > 0.3: compressible regime ***

    Cut Sizes (d50) - based on ZONE velocity:
      Zigzag:          0.0 um (at v_zone=0.00 m/s)
      (if bulk):       0.0 um (wrong - ignores zone effect)
      Cy1 (primary)    4.9 um (weak vortex)
      Cy2 (secondary)  2.7 um
      Cy3 (tertiary)   1.2 um

    Multi-Stage Sharpening (0 stages):
      Each stage is a separation opportunity
      Effective cut sharpness increases with stages

    For protein separation:
      Protein:         ~10-30 um (should go to fines)
      Starch:          ~15-60 um (should go to coarse)
      Status: Zigzag d50 (0.0um) in protein range - good!

  ClassificationFlowPhysicsSimulator initialized
    Device: cuda
    Max particles: 100000

Initializing particles at wheel inlet (15° solids chute)...

  Pre-allocated 100000 particles as yellow_pea whole flour (continuous feeding)
    Feed rate: 833 particles/s
    Mass flow: 0.0 kg/h  (0.0000 kg/s)
    Time to feed all 100000 particles: 120.0 s
    Protein: 25000 (25%)  Starch: 55000 (55%)  Fiber: 20000 (20%)
    Diameter range: 5.0 - 100.0 um  Total mass: 0.01 g
    Initial zone:   34

----------------------------------------------------------------------
RUNNING SIMULATION
----------------------------------------------------------------------
  Time: 360.0 s
  dt:   1.00 ms
  Steps: 360,000
  Air flow: 472 m³/h
  Wheel d50: 34.5 µm
  Feeding: continuous at 833 particles/s
  Feed mass flow: 0.0 kg/h
  Max loading ratio: 2.0
----------------------------------------------------------------------
Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 2.03 ms  (cached)
  [  0.0%] t= 0.03s | Fed:   27/100000 | Active:   27 Zc:    0 Wc:    0 Cy1:    0 Cy2:    0 Cy3:    0 Bag:    0  [zz:   0 fp:   0 wh:   2 wf:   2 wch:  20 c1:   0 c2:   0 c3:   0]
  [  5.0%] t=18.05s | Fed:15042/100000 | Active: 3530 Zc:    0 Wc: 7989 Cy1: 1279 Cy2:   38 Cy3: 2194 Bag:   12  [zz:   0 fp:   0 wh: 112 wf:   2 wch:1086 c1:1555 c2:  84 c3:   4]
  [ 10.0%] t=36.07s | Fed:30057/100000 | Active: 4481 Zc:    0 Wc:17079 Cy1: 2525 Cy2:   99 Cy3: 5831 Bag:   42  [zz:   0 fp:   0 wh: 212 wf:   1 wch:1132 c1:2372 c2:  91 c3:   0]
  [ 15.0%] t=54.09s | Fed:45072/100000 | Active: 5090 Zc:    0 Wc:26070 Cy1: 3831 Cy2:  178 Cy3: 9819 Bag:   84  [zz:   0 fp:   0 wh: 318 wf:   1 wch:1099 c1:2887 c2:  90 c3:   5]
  [ 20.0%] t=72.10s | Fed:60087/100000 | Active: 5425 Zc:    0 Wc:35153 Cy1: 5082 Cy2:  275 Cy3:14028 Bag:  124  [zz:   0 fp:   0 wh: 432 wf:   0 wch:1118 c1:3114 c2: 106 c3:   5]
  [ 25.0%] t=90.12s | Fed:75102/100000 | Active: 5752 Zc:    0 Wc:44317 Cy1: 6374 Cy2:  357 Cy3:18131 Bag:  171  [zz:   0 fp:   0 wh: 527 wf:   0 wch:1103 c1:3289 c2: 103 c3:   5]
  [ 30.0%] t=108.14s | Fed:90117/100000 | Active: 5976 Zc:    0 Wc:53384 Cy1: 7665 Cy2:  452 Cy3:22406 Bag:  234  [zz:   0 fp:   0 wh: 630 wf:   1 wch:1064 c1:3480 c2: 104 c3:   7]
  [ 35.0%] t=126.16s | Fed:100000/100000 | Active: 3694 Zc:    0 Wc:60325 Cy1: 8612 Cy2:  537 Cy3:26543 Bag:  289  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   6 c1:2942 c2:  36 c3:   5]
  [ 40.0%] t=144.18s | Fed:100000/100000 | Active: 2356 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  583 Cy3:27795 Bag:  322  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1:1638 c2:  11 c3:   2]
  [ 45.1%] t=162.20s | Fed:100000/100000 | Active: 1692 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  602 Cy3:28417 Bag:  345  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 979 c2:   8 c3:   1]
  [ 50.1%] t=180.21s | Fed:100000/100000 | Active: 1390 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  607 Cy3:28705 Bag:  354  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 677 c2:   9 c3:   0]
  [ 55.1%] t=198.23s | Fed:100000/100000 | Active: 1200 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  611 Cy3:28884 Bag:  361  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 494 c2:   2 c3:   0]
  [ 60.1%] t=216.25s | Fed:100000/100000 | Active: 1092 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  613 Cy3:28990 Bag:  361  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 386 c2:   2 c3:   0]
  [ 65.1%] t=234.27s | Fed:100000/100000 | Active: 1031 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  613 Cy3:29049 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 325 c2:   2 c3:   0]
  [ 70.1%] t=252.29s | Fed:100000/100000 | Active:  987 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  616 Cy3:29090 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 279 c2:   4 c3:   0]
  [ 75.1%] t=270.30s | Fed:100000/100000 | Active:  942 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  616 Cy3:29135 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 237 c2:   1 c3:   0]
  [ 80.1%] t=288.32s | Fed:100000/100000 | Active:  910 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  617 Cy3:29166 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 206 c2:   0 c3:   0]
  [ 85.1%] t=306.34s | Fed:100000/100000 | Active:  885 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  617 Cy3:29191 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 181 c2:   0 c3:   0]
  [ 90.1%] t=324.36s | Fed:100000/100000 | Active:  865 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  617 Cy3:29211 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 161 c2:   0 c3:   0]
  [ 95.1%] t=342.37s | Fed:100000/100000 | Active:  849 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  617 Cy3:29227 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 145 c2:   0 c3:   0]
  [100.0%] t=360.00s | Fed:100000/100000 | Active:  834 Zc:    0 Wc:60331 Cy1: 8613 Cy2:  617 Cy3:29242 Bag:  363  [zz:   0 fp:   0 wh: 704 wf:   0 wch:   0 c1: 130 c2:   0 c3:   0]
----------------------------------------------------------------------
SIMULATION COMPLETE
----------------------------------------------------------------------
  Wall time: 73.1 s
  Sim time:  360.00 s
  Steps:     360,000
  Rate:      4925 steps/s
  Feeding:   100000/100000 particles fed (100.0%)
  Feed rate: 833 particles/s

  Separation Results (t = 360.000s):
  ==================================================
    Zigzag coarse (starch):       0 (  0.0%)
    Wheel coarse (starch):    60331 ( 60.3%)
    Cyclone 1 (fines 1):       8613 (  8.6%)
    Cyclone 2 (fines 2):        617 (  0.6%)
    Cyclone 3 (PROTEIN):      29242 ( 29.2%)
    Bag filter:                  363 (  0.4%)
    Escaped (loss):                0 (  0.0%)
    Still active:                 834 (  0.8%)
  ==================================================

  Cyclone particle sizes (design d50 vs actual mean):
    cyclone_1            N= 8613  (design d50=40 µm, mean=18.3 µm)
    cyclone_2            N=  617  (design d50=20 µm, mean=12.2 µm)
    cyclone_3_protein    N=29242  (design d50=10 µm, mean=9.7 µm)
  ==================================================
    Total (balance):       100000  (slots used: 100000)
  ==================================================

    Protein recovery (cy3 + bag): 29605
    Starch recovery (zigzag + wheel coarse): 60331
======================================================================
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 


Run 2:

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea  --blower-rpm-range 550 750 --wheel-rpm-range 800 1500  --n-blower 5 --n-wheel 5
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================

======================================================================
GRID SEARCH OPTIMIZATION — WHEEL-ONLY
  Blower RPM: 550 – 750 (5 points)
  Wheel RPM:  800 – 1500 (5 points)
  Total trials: 25
  Objective: protein_recovery
  Particles: 50000, Time: 240.0s
======================================================================

  [  1/25] Blower=550 RPM, Wheel=800 RPM ... Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 1.77 ms  (cached)
OK  prot=0.178  starch=0.561  purity=0.582  score=0.1776  (39.8s)
  [  2/25] Blower=550 RPM, Wheel=975 RPM ... OK  prot=0.179  starch=0.624  purity=0.582  score=0.1790  (39.5s)
  [  3/25] Blower=550 RPM, Wheel=1150 RPM ... OK  prot=0.179  starch=0.671  purity=0.582  score=0.1785  (39.5s)
  [  4/25] Blower=550 RPM, Wheel=1325 RPM ... OK  prot=0.168  starch=0.709  purity=0.573  score=0.1682  (39.5s)
  [  5/25] Blower=550 RPM, Wheel=1500 RPM ... OK  prot=0.140  starch=0.742  purity=0.594  score=0.1403  (39.8s)
  [  6/25] Blower=600 RPM, Wheel=800 RPM ... OK  prot=0.231  starch=0.545  purity=0.644  score=0.2313  (39.8s)
  [  7/25] Blower=600 RPM, Wheel=975 RPM ... OK  prot=0.234  starch=0.612  purity=0.646  score=0.2337  (39.4s)
  [  8/25] Blower=600 RPM, Wheel=1150 RPM ... OK  prot=0.235  starch=0.659  purity=0.648  score=0.2351  (43.8s)
  [  9/25] Blower=600 RPM, Wheel=1325 RPM ... OK  prot=0.228  starch=0.698  purity=0.643  score=0.2279  (42.1s)
  [ 10/25] Blower=600 RPM, Wheel=1500 RPM ... OK  prot=0.200  starch=0.730  purity=0.662  score=0.1997  (41.2s)
  [ 11/25] Blower=650 RPM, Wheel=800 RPM ... OK  prot=0.292  starch=0.531  purity=0.707  score=0.2918  (45.5s)
  [ 12/25] Blower=650 RPM, Wheel=975 RPM ... OK  prot=0.286  starch=0.598  purity=0.702  score=0.2858  (40.4s)
  [ 13/25] Blower=650 RPM, Wheel=1150 RPM ... OK  prot=0.285  starch=0.648  purity=0.701  score=0.2848  (40.1s)
  [ 14/25] Blower=650 RPM, Wheel=1325 RPM ... OK  prot=0.282  starch=0.686  purity=0.699  score=0.2823  (39.8s)
  [ 15/25] Blower=650 RPM, Wheel=1500 RPM ... OK  prot=0.257  starch=0.720  purity=0.718  score=0.2571  (40.4s)
  [ 16/25] Blower=700 RPM, Wheel=800 RPM ... OK  prot=0.300  starch=0.517  purity=0.720  score=0.2996  (59.6s)
  [ 17/25] Blower=700 RPM, Wheel=975 RPM ... OK  prot=0.301  starch=0.586  purity=0.721  score=0.3008  (49.9s)
  [ 18/25] Blower=700 RPM, Wheel=1150 RPM ... OK  prot=0.300  starch=0.637  purity=0.720  score=0.3004  (40.7s)
  [ 19/25] Blower=700 RPM, Wheel=1325 RPM ... OK  prot=0.298  starch=0.676  purity=0.720  score=0.2985  (40.5s)
  [ 20/25] Blower=700 RPM, Wheel=1500 RPM ... OK  prot=0.279  starch=0.710  purity=0.726  score=0.2790  (47.3s)
  [ 21/25] Blower=750 RPM, Wheel=800 RPM ... OK  prot=0.284  starch=0.504  purity=0.744  score=0.2845  (41.0s)
  [ 22/25] Blower=750 RPM, Wheel=975 RPM ... OK  prot=0.284  starch=0.574  purity=0.744  score=0.2839  (54.4s)
  [ 23/25] Blower=750 RPM, Wheel=1150 RPM ... OK  prot=0.285  starch=0.627  purity=0.745  score=0.2846  (40.8s)
  [ 24/25] Blower=750 RPM, Wheel=1325 RPM ... OK  prot=0.283  starch=0.667  purity=0.746  score=0.2833  (40.8s)
  [ 25/25] Blower=750 RPM, Wheel=1500 RPM ... OK  prot=0.280  starch=0.701  purity=0.741  score=0.2804  (40.8s)

======================================================================
OPTIMIZATION COMPLETE — GRID — WHEEL-ONLY
======================================================================
  Objective:       protein_recovery
  Trials:          25
  Total wall time: 1067.6 s (17.8 min)
  Best score:      0.3008

  ──────────────────────────────────────────────────
  BEST CONFIGURATION:
  ──────────────────────────────────────────────────
    Mode:           wheel-only
    Blower RPM:     700
    Wheel RPM:      975

  RESULTS:
    Protein recovery:     0.301  (30.1%)
    Starch yield:         0.586  (58.6%)
    Protein purity:       0.721  (72.1%)
    Separation eff.:      0.413
    Total collection:     0.998  (99.8%)

  COLLECTION BREAKDOWN (of 50000 particles):
    Wheel coarse (starch)      29300 ( 58.6%) |#############################
    Cyclone 1 (fines)           4962 (  9.9%) |####
    Cyclone 2 (fines)            612 (  1.2%) |
    Cyclone 3 (protein)        14824 ( 29.6%) |##############
    Bag filter                   215 (  0.4%) |
    Escaped                        0 (  0.0%) |
    Still active                  87 (  0.2%) |

  ──────────────────────────────────────────────────
  TOP 5 CONFIGURATIONS:
  ──────────────────────────────────────────────────
  Rank   Blower    Wheel  ProtRec  StchYld   Purity  Collect    Score
  ────  ───────  ───────  ───────  ───────  ───────  ───────  ───────
     1      700      975    0.301    0.586    0.721    0.998   0.3008
     2      700     1150    0.300    0.637    0.720    0.998   0.3004
     3      700      800    0.300    0.517    0.720    0.997   0.2996
     4      700     1325    0.298    0.676    0.720    0.997   0.2985
     5      650      800    0.292    0.531    0.707    0.987   0.2918

======================================================================

  Reproduce best with:
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 700 --wheel-rpm 975 --wheel-only

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 700 --wheel-rpm 975 --wheel-only
======================================================================
PHYSICS-BASED CLASSIFICATION FLOW SIMULATION
  Protein/Starch Separation via Air Classification
======================================================================
  VFD: 700 RPM -> 495 m³/h (operating point)
       Fan law would give 700 m³/h (overestimate)
       P = 136 Pa, eff = 68.6%, W = 27 W
  Wheel RPM (main classifier): 975 (override)

[FULL SYSTEM] Air -> Wheel (no preclassification) -> Cyclones -> Bag Filter
  [Mode] Without preclassification (wheel-only): no venturi, zigzag, dropout

1. Air system -> Wheel junction (airclass, no venturi)...
============================================================
AIR SYSTEM -> VENTURI DUCTWORK FLOW
============================================================
  Volume flow:    495.0 m³/h
  Total dP:       6.1 Pa
  Segments:      4
------------------------------------------------------------
  duct_0             duct       v=  4.38 m/s  Re=   57746  dP=   0.1 Pa
  elbow_1            elbow      v=  4.38 m/s  Re=   57746  dP=   3.4 Pa
  duct_2             duct       v=  4.38 m/s  Re=   57746  dP=   0.9 Pa
  transition_3       transition v=  4.38 m/s  Re=   57746  dP=   1.7 Pa
============================================================

2. Feed system -> Wheel junction solids inlet (feedclass, no venturi)...
======================================================================
FEED SYSTEM -> VENTURI SOLIDS INLET (ductwork flow + kinetics)
======================================================================
  Air flow (sweep):  0.00 m3/h
  Particle d:        35.0 um
  Terminal v (vert): 0.0502 m/s
  Total dP:          0.00 Pa
  Total residence:   21.906 s
  Venturi solids D:  0 mm
----------------------------------------------------------------------
  feed_duct_0            duct       L=1.099m v_air=0.00 v_part=0.05 t_res=21.906s dP=0.0Pa
======================================================================

  Material / feed properties (used for classification validation):
    Material:        yellow_pea (density=1420 kg/m³, sphericity=0.70)
    Size range:      2.0 – 500.0 µm   d50=35.0 µm
    Feed rep. d:     35.0 µm (feed ductwork and entry rate)
    Particle rate:   4356888079 particles/s (solids mass flow + rep. d)

3. Wheel-only classification (no venturi/zigzag):
   Air flow:           495 m³/h (0.137 m³/s)
   Solids mass flow:   500.0 kg/h
  [Feed cap] Requested 184411.1 kg/h exceeds venturi capacity 1191.9 kg/h (mu=2.0)
             Capped to 1191.9 kg/h
  [Continuous feed] 100000 sim particles over 120s = 833 particles/s
  [Physical rate] 1191.9 kg/h (capped at mu=2.0)

Creating classification system assembly (from full system)...
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0

  *** VENTURI COMPRESSIBILITY WARNING ***
      Throat velocity: 109 m/s
      Mach number:     0.32 (>0.3 - compressible regime)
      Bernoulli (incompressible) approximation has >5% error
      Max choked flow: 1528 m3/h

    Wheel Classifier (main classifier - centrifugal):
      Diameter:        200 mm
      RPM:             975
      Tip speed:       10.2 m/s
      G-force (rim):   106 g
      d50:             36.3 um
      Hub radius:      30.0 mm
      Blades:          24

    Particle Entry: Zone 34 (wheel housing - wheel-only mode)

  Classification Physics Parameters:

    Air Flow:
      Flow rate:       495 m3/h
      Venturi inlet:   70.0 m/s
      Venturi throat:  109.4 m/s (D=40.0mm, Ma=0.319)
      Venturi dP:      4255 Pa (4.3 kPa)
      Zigzag bulk:     0.00 m/s
      Zigzag ZONE:     0.00 m/s (30% of bulk)
      Cyclone (series, rectangular tangential inlet):
        Primary    D=300mm  inlet=75x150mm  v=12.2 m/s (weak vortex)
        Secondary  D=200mm  inlet=50x100mm  v=27.5 m/s
        Tertiary   D=120mm  inlet=30x60mm  v=76.4 m/s

    Venturi Throat Analysis:
      Throat diameter: 40.0 mm
      Throat area:     1256.6 mm2
      Max flow (Ma=1): 1528 m3/h
      K_venturi:       225072.9 Pa/(m3/s)2
      *** Ma=0.32 > 0.3: compressible regime ***

    Cut Sizes (d50) - based on ZONE velocity:
      Zigzag:          0.0 um (at v_zone=0.00 m/s)
      (if bulk):       0.0 um (wrong - ignores zone effect)
      Cy1 (primary)    4.8 um (weak vortex)
      Cy2 (secondary)  2.6 um
      Cy3 (tertiary)   1.2 um

    Multi-Stage Sharpening (0 stages):
      Each stage is a separation opportunity
      Effective cut sharpness increases with stages

    For protein separation:
      Protein:         ~10-30 um (should go to fines)
      Starch:          ~15-60 um (should go to coarse)
      Status: Zigzag d50 (0.0um) in protein range - good!

  ClassificationFlowPhysicsSimulator initialized
    Device: cuda
    Max particles: 100000

Initializing particles at wheel inlet (15° solids chute)...

  Pre-allocated 100000 particles as yellow_pea whole flour (continuous feeding)
    Feed rate: 833 particles/s
    Mass flow: 0.0 kg/h  (0.0000 kg/s)
    Time to feed all 100000 particles: 120.0 s
    Protein: 25000 (25%)  Starch: 55000 (55%)  Fiber: 20000 (20%)
    Diameter range: 5.0 - 100.0 um  Total mass: 0.01 g
    Initial zone:   34

----------------------------------------------------------------------
RUNNING SIMULATION
----------------------------------------------------------------------
  Time: 360.0 s
  dt:   1.00 ms
  Steps: 360,000
  Air flow: 495 m³/h
  Wheel d50: 36.3 µm
  Feeding: continuous at 833 particles/s
  Feed mass flow: 0.0 kg/h
  Max loading ratio: 2.0
----------------------------------------------------------------------
Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 2.08 ms  (cached)
  [  0.0%] t= 0.03s | Fed:   27/100000 | Active:   27 Zc:    0 Wc:    0 Cy1:    0 Cy2:    0 Cy3:    0 Bag:    0  [zz:   0 fp:   0 wh:   2 wf:   1 wch:  20 c1:   0 c2:   0 c3:   0]
  [  5.0%] t=18.05s | Fed:15042/100000 | Active: 3284 Zc:    0 Wc: 7855 Cy1: 1492 Cy2:   56 Cy3: 2341 Bag:   14  [zz:   0 fp:   0 wh:  13 wf:   2 wch: 973 c1:1591 c2:  81 c3:   2]
  [ 10.0%] t=36.07s | Fed:30057/100000 | Active: 4074 Zc:    0 Wc:16668 Cy1: 2967 Cy2:  175 Cy3: 6128 Bag:   45  [zz:   0 fp:   0 wh:  16 wf:   1 wch:1037 c1:2328 c2:  97 c3:   5]
  [ 15.0%] t=54.09s | Fed:45072/100000 | Active: 4457 Zc:    0 Wc:25427 Cy1: 4447 Cy2:  333 Cy3:10315 Bag:   93  [zz:   0 fp:   0 wh:  22 wf:   1 wch:1016 c1:2709 c2:  89 c3:   5]
  [ 20.0%] t=72.10s | Fed:60087/100000 | Active: 4610 Zc:    0 Wc:34288 Cy1: 5892 Cy2:  510 Cy3:14639 Bag:  148  [zz:   0 fp:   0 wh:  32 wf:   0 wch:1031 c1:2893 c2:  69 c3:   4]
  [ 25.0%] t=90.12s | Fed:75102/100000 | Active: 4778 Zc:    0 Wc:43212 Cy1: 7394 Cy2:  705 Cy3:18807 Bag:  206  [zz:   0 fp:   0 wh:  30 wf:   1 wch:1002 c1:3008 c2:  88 c3:   2]
  [ 30.0%] t=108.14s | Fed:90117/100000 | Active: 4757 Zc:    0 Wc:52020 Cy1: 8903 Cy2:  892 Cy3:23289 Bag:  256  [zz:   0 fp:   0 wh:  35 wf:   1 wch: 968 c1:3056 c2:  97 c3:   6]
  [ 35.0%] t=126.16s | Fed:100000/100000 | Active: 2518 Zc:    0 Wc:58712 Cy1: 9980 Cy2: 1083 Cy3:27391 Bag:  316  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:2442 c2:  42 c3:   2]
  [ 40.0%] t=144.18s | Fed:100000/100000 | Active: 1208 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1196 Cy3:28557 Bag:  346  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:1169 c2:   8 c3:   0]
  [ 45.1%] t=162.20s | Fed:100000/100000 | Active:  685 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1256 Cy3:29012 Bag:  354  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 649 c2:   5 c3:   0]
  [ 50.1%] t=180.21s | Fed:100000/100000 | Active:  455 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1293 Cy3:29202 Bag:  357  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 422 c2:   2 c3:   0]
  [ 55.1%] t=198.23s | Fed:100000/100000 | Active:  333 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1298 Cy3:29319 Bag:  357  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 295 c2:   6 c3:   1]
  [ 60.1%] t=216.25s | Fed:100000/100000 | Active:  263 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29385 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 232 c2:   0 c3:   0]
  [ 65.1%] t=234.27s | Fed:100000/100000 | Active:  216 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29432 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 185 c2:   0 c3:   0]
  [ 70.1%] t=252.29s | Fed:100000/100000 | Active:  181 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29467 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 150 c2:   0 c3:   0]
  [ 75.1%] t=270.30s | Fed:100000/100000 | Active:  152 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29496 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1: 121 c2:   0 c3:   0]
  [ 80.1%] t=288.32s | Fed:100000/100000 | Active:  127 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29521 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:  96 c2:   0 c3:   0]
  [ 85.1%] t=306.34s | Fed:100000/100000 | Active:  111 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29537 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:  80 c2:   0 c3:   0]
  [ 90.1%] t=324.36s | Fed:100000/100000 | Active:  100 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29548 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:  69 c2:   0 c3:   0]
  [ 95.1%] t=342.37s | Fed:100000/100000 | Active:   90 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29558 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:  58 c2:   1 c3:   0]
  [100.0%] t=360.00s | Fed:100000/100000 | Active:   82 Zc:    0 Wc:58712 Cy1: 9981 Cy2: 1301 Cy3:29566 Bag:  358  [zz:   0 fp:   0 wh:  31 wf:   0 wch:   0 c1:  51 c2:   0 c3:   0]
----------------------------------------------------------------------
SIMULATION COMPLETE
----------------------------------------------------------------------
  Wall time: 76.7 s
  Sim time:  360.00 s
  Steps:     360,000
  Rate:      4695 steps/s
  Feeding:   100000/100000 particles fed (100.0%)
  Feed rate: 833 particles/s

  Separation Results (t = 360.000s):
  ==================================================
    Zigzag coarse (starch):       0 (  0.0%)
    Wheel coarse (starch):    58712 ( 58.7%)
    Cyclone 1 (fines 1):       9981 ( 10.0%)
    Cyclone 2 (fines 2):       1301 (  1.3%)
    Cyclone 3 (PROTEIN):      29566 ( 29.6%)
    Bag filter:                  358 (  0.4%)
    Escaped (loss):                0 (  0.0%)
    Still active:                  82 (  0.1%)
  ==================================================

  Cyclone particle sizes (design d50 vs actual mean):
    cyclone_1            N= 9981  (design d50=40 µm, mean=18.9 µm)
    cyclone_2            N= 1301  (design d50=20 µm, mean=13.3 µm)
    cyclone_3_protein    N=29566  (design d50=10 µm, mean=9.6 µm)
  ==================================================
    Total (balance):       100000  (slots used: 100000)
  ==================================================

    Protein recovery (cy3 + bag): 29924
    Starch recovery (zigzag + wheel coarse): 58712
======================================================================
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 

PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> & C:/Users/Windows/Desktop/Dev_Projects/airclassifier/venv/Scripts/Activate.ps1
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea  --objective protein_purity
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================

======================================================================
GRID SEARCH OPTIMIZATION — WHEEL-ONLY
  Blower RPM: 550 – 850 (4 points)
  Wheel RPM:  800 – 1500 (4 points)
  Total trials: 16
  Objective: protein_purity
  Particles: 50000, Time: 240.0s
======================================================================

  [  1/16] Blower=550 RPM, Wheel=800 RPM ... Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 2.67 ms  (cached)
OK  prot=0.178  starch=0.561  purity=0.582  score=0.4609  (60.3s)
  [  2/16] Blower=550 RPM, Wheel=1033 RPM ... OK  prot=0.178  starch=0.641  purity=0.584  score=0.4620  (49.4s)
  [  3/16] Blower=550 RPM, Wheel=1267 RPM ... OK  prot=0.175  starch=0.697  purity=0.575  score=0.4551  (52.4s)
  [  4/16] Blower=550 RPM, Wheel=1500 RPM ... OK  prot=0.140  starch=0.742  purity=0.594  score=0.4580  (58.0s)
  [  5/16] Blower=650 RPM, Wheel=800 RPM ... OK  prot=0.292  starch=0.531  purity=0.707  score=0.5821  (46.8s)
  [  6/16] Blower=650 RPM, Wheel=1033 RPM ... OK  prot=0.286  starch=0.617  purity=0.702  score=0.5773  (42.8s)
  [  7/16] Blower=650 RPM, Wheel=1267 RPM ... OK  prot=0.285  starch=0.674  purity=0.702  score=0.5771  (42.5s)
  [  8/16] Blower=650 RPM, Wheel=1500 RPM ... OK  prot=0.257  starch=0.720  purity=0.718  score=0.5797  (42.6s)
  [  9/16] Blower=750 RPM, Wheel=800 RPM ... OK  prot=0.284  starch=0.504  purity=0.744  score=0.6062  (42.4s)
  [ 10/16] Blower=750 RPM, Wheel=1033 RPM ... OK  prot=0.284  starch=0.593  purity=0.743  score=0.6056  (46.7s)
  [ 11/16] Blower=750 RPM, Wheel=1267 RPM ... OK  prot=0.284  starch=0.655  purity=0.744  score=0.6058  (42.5s)
  [ 12/16] Blower=750 RPM, Wheel=1500 RPM ... OK  prot=0.280  starch=0.701  purity=0.741  score=0.6030  (38.9s)
  [ 13/16] Blower=850 RPM, Wheel=800 RPM ... OK  prot=0.242  starch=0.477  purity=0.796  score=0.6300  (45.8s)
  [ 14/16] Blower=850 RPM, Wheel=1033 RPM ... OK  prot=0.242  starch=0.573  purity=0.795  score=0.6290  (51.0s)
  [ 15/16] Blower=850 RPM, Wheel=1267 RPM ... OK  prot=0.241  starch=0.637  purity=0.798  score=0.6311  (38.6s)
  [ 16/16] Blower=850 RPM, Wheel=1500 RPM ... OK  prot=0.241  starch=0.683  purity=0.796  score=0.6297  (38.5s)

======================================================================
OPTIMIZATION COMPLETE — GRID — WHEEL-ONLY
======================================================================
  Objective:       protein_purity
  Trials:          16
  Total wall time: 740.5 s (12.3 min)
  Best score:      0.6311

  ──────────────────────────────────────────────────
  BEST CONFIGURATION:
  ──────────────────────────────────────────────────
    Mode:           wheel-only
    Blower RPM:     850
    Wheel RPM:      1267

  RESULTS:
    Protein recovery:     0.241  (24.1%)
    Starch yield:         0.637  (63.7%)
    Protein purity:       0.798  (79.8%)
    Separation eff.:      0.363
    Total collection:     1.000  (100.0%)

  COLLECTION BREAKDOWN (of 50000 particles):
    Wheel coarse (starch)      31840 ( 63.7%) |###############################
    Cyclone 1 (fines)           1857 (  3.7%) |#
    Cyclone 2 (fines)           4248 (  8.5%) |####
    Cyclone 3 (protein)        11834 ( 23.7%) |###########
    Bag filter                   221 (  0.4%) |
    Escaped                        0 (  0.0%) |
    Still active                   0 (  0.0%) |

  ──────────────────────────────────────────────────
  TOP 5 CONFIGURATIONS:
  ──────────────────────────────────────────────────
  Rank   Blower    Wheel  ProtRec  StchYld   Purity  Collect    Score
  ────  ───────  ───────  ───────  ───────  ───────  ───────  ───────
     1      850     1267    0.241    0.637    0.798    1.000   0.6311
     2      850      800    0.242    0.477    0.796    1.000   0.6300
     3      850     1500    0.241    0.683    0.796    1.000   0.6297
     4      850     1033    0.242    0.573    0.795    1.000   0.6290
     5      750      800    0.284    0.504    0.744    1.000   0.6062

======================================================================

  Reproduce best with:
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 850 --wheel-rpm 1267 --wheel-only

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 



PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> & C:/Users/Windows/Desktop/Dev_Projects/airclassifier/venv/Scripts/Activate.ps1
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea  --objective protein_purity
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================

======================================================================
GRID SEARCH OPTIMIZATION — WHEEL-ONLY
  Blower RPM: 550 – 850 (4 points)
  Wheel RPM:  800 – 1500 (4 points)
  Total trials: 16
  Objective: protein_purity
  Particles: 50000, Time: 240.0s
======================================================================

  [  1/16] Blower=550 RPM, Wheel=800 RPM ... Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 2.67 ms  (cached)
OK  prot=0.178  starch=0.561  purity=0.582  score=0.4609  (60.3s)
  [  2/16] Blower=550 RPM, Wheel=1033 RPM ... OK  prot=0.178  starch=0.641  purity=0.584  score=0.4620  (49.4s)
  [  3/16] Blower=550 RPM, Wheel=1267 RPM ... OK  prot=0.175  starch=0.697  purity=0.575  score=0.4551  (52.4s)
  [  4/16] Blower=550 RPM, Wheel=1500 RPM ... OK  prot=0.140  starch=0.742  purity=0.594  score=0.4580  (58.0s)
  [  5/16] Blower=650 RPM, Wheel=800 RPM ... OK  prot=0.292  starch=0.531  purity=0.707  score=0.5821  (46.8s)
  [  6/16] Blower=650 RPM, Wheel=1033 RPM ... OK  prot=0.286  starch=0.617  purity=0.702  score=0.5773  (42.8s)
  [  7/16] Blower=650 RPM, Wheel=1267 RPM ... OK  prot=0.285  starch=0.674  purity=0.702  score=0.5771  (42.5s)
  [  8/16] Blower=650 RPM, Wheel=1500 RPM ... OK  prot=0.257  starch=0.720  purity=0.718  score=0.5797  (42.6s)
  [  9/16] Blower=750 RPM, Wheel=800 RPM ... OK  prot=0.284  starch=0.504  purity=0.744  score=0.6062  (42.4s)
  [ 10/16] Blower=750 RPM, Wheel=1033 RPM ... OK  prot=0.284  starch=0.593  purity=0.743  score=0.6056  (46.7s)
  [ 11/16] Blower=750 RPM, Wheel=1267 RPM ... OK  prot=0.284  starch=0.655  purity=0.744  score=0.6058  (42.5s)
  [ 12/16] Blower=750 RPM, Wheel=1500 RPM ... OK  prot=0.280  starch=0.701  purity=0.741  score=0.6030  (38.9s)
  [ 13/16] Blower=850 RPM, Wheel=800 RPM ... OK  prot=0.242  starch=0.477  purity=0.796  score=0.6300  (45.8s)
  [ 14/16] Blower=850 RPM, Wheel=1033 RPM ... OK  prot=0.242  starch=0.573  purity=0.795  score=0.6290  (51.0s)
  [ 15/16] Blower=850 RPM, Wheel=1267 RPM ... OK  prot=0.241  starch=0.637  purity=0.798  score=0.6311  (38.6s)
  [ 16/16] Blower=850 RPM, Wheel=1500 RPM ... OK  prot=0.241  starch=0.683  purity=0.796  score=0.6297  (38.5s)

======================================================================
OPTIMIZATION COMPLETE — GRID — WHEEL-ONLY
======================================================================
  Objective:       protein_purity
  Trials:          16
  Total wall time: 740.5 s (12.3 min)
  Best score:      0.6311

  ──────────────────────────────────────────────────
  BEST CONFIGURATION:
  ──────────────────────────────────────────────────
    Mode:           wheel-only
    Blower RPM:     850
    Wheel RPM:      1267

  RESULTS:
    Protein recovery:     0.241  (24.1%)
    Starch yield:         0.637  (63.7%)
    Protein purity:       0.798  (79.8%)
    Separation eff.:      0.363
    Total collection:     1.000  (100.0%)

  COLLECTION BREAKDOWN (of 50000 particles):
    Wheel coarse (starch)      31840 ( 63.7%) |###############################
    Cyclone 1 (fines)           1857 (  3.7%) |#
    Cyclone 2 (fines)           4248 (  8.5%) |####
    Cyclone 3 (protein)        11834 ( 23.7%) |###########
    Bag filter                   221 (  0.4%) |
    Escaped                        0 (  0.0%) |
    Still active                   0 (  0.0%) |

  ──────────────────────────────────────────────────
  TOP 5 CONFIGURATIONS:
  ──────────────────────────────────────────────────
  Rank   Blower    Wheel  ProtRec  StchYld   Purity  Collect    Score
  ────  ───────  ───────  ───────  ───────  ───────  ───────  ───────
     1      850     1267    0.241    0.637    0.798    1.000   0.6311
     2      850      800    0.242    0.477    0.796    1.000   0.6300
     3      850     1500    0.241    0.683    0.796    1.000   0.6297
     4      850     1033    0.242    0.573    0.795    1.000   0.6290
     5      750      800    0.284    0.504    0.744    1.000   0.6062

======================================================================

  Reproduce best with:
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 850 --wheel-rpm 1267 --wheel-only

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea  --strategy bayesian --n-trials 50
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================
ERROR: Bayesian optimization requires 'optuna'. Install with:
       pip install optuna
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier>  pip install optuna
Collecting optuna
  Downloading optuna-4.7.0-py3-none-any.whl.metadata (17 kB)
Collecting alembic>=1.5.0 (from optuna)
  Downloading alembic-1.18.4-py3-none-any.whl.metadata (7.2 kB)
Collecting colorlog (from optuna)
  Downloading colorlog-6.10.1-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: numpy in .\venv\Lib\site-packages (from optuna) (2.4.1)
Requirement already satisfied: packaging>=20.0 in .\venv\Lib\site-packages (from optuna) (26.0)
Collecting sqlalchemy>=1.4.2 (from optuna)
  Downloading sqlalchemy-2.0.46-cp313-cp313-win_amd64.whl.metadata (9.8 kB)
Collecting tqdm (from optuna)
  Downloading tqdm-4.67.3-py3-none-any.whl.metadata (57 kB)
Requirement already satisfied: PyYAML in .\venv\Lib\site-packages (from optuna) (6.0.3)
Collecting Mako (from alembic>=1.5.0->optuna)
  Using cached mako-1.3.10-py3-none-any.whl.metadata (2.9 kB)
Requirement already satisfied: typing-extensions>=4.12 in .\venv\Lib\site-packages (from alembic>=1.5.0->optuna) (4.15.0)
Collecting greenlet>=1 (from sqlalchemy>=1.4.2->optuna)
  Downloading greenlet-3.3.1-cp313-cp313-win_amd64.whl.metadata (3.8 kB)
Requirement already satisfied: colorama in .\venv\Lib\site-packages (from colorlog->optuna) (0.4.6)
Requirement already satisfied: MarkupSafe>=0.9.2 in .\venv\Lib\site-packages (from Mako->alembic>=1.5.0->optuna) (3.0.3)
Downloading optuna-4.7.0-py3-none-any.whl (413 kB)
Downloading alembic-1.18.4-py3-none-any.whl (263 kB)
Downloading sqlalchemy-2.0.46-cp313-cp313-win_amd64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 28.0 MB/s  0:00:00
Downloading greenlet-3.3.1-cp313-cp313-win_amd64.whl (227 kB)
Downloading colorlog-6.10.1-py3-none-any.whl (11 kB)
Using cached mako-1.3.10-py3-none-any.whl (78 kB)
Downloading tqdm-4.67.3-py3-none-any.whl (78 kB)
Installing collected packages: tqdm, Mako, greenlet, colorlog, sqlalchemy, alembic, optuna
Successfully installed Mako-1.3.10 alembic-1.18.4 colorlog-6.10.1 greenlet-3.3.1 optuna-4.7.0 sqlalchemy-2.0.46 tqdm-4.67.3                                                                                                                  

[notice] A new release of pip is available: 26.0 -> 26.0.1
[notice] To update, run: python.exe -m pip install --upgrade pip
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> python examples/optimize_classification.py --material yellow_pea  --strategy bayesian --n-trials 50
======================================================================
AIR CLASSIFIER CONFIGURATION OPTIMIZER
  Mode: WHEEL-ONLY (no venturi/zigzag)
======================================================================

======================================================================
BAYESIAN OPTIMIZATION (Optuna TPE) — WHEEL-ONLY
  Blower RPM: 550 – 850
  Wheel RPM:  800 – 1500
  Max trials: 50
  Objective: protein_recovery
  Particles: 50000, Time: 240.0s
======================================================================

  [  1/50] Blower=663, Wheel=1306 ... Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 1.64 ms  (cached)
OK  prot=0.290  starch=0.679  purity=0.705  score=0.2903  (39.1s)
  [  2/50] Blower=566, Wheel=1366 ... OK  prot=0.180  starch=0.714  purity=0.601  score=0.1798  (39.3s)
  [  3/50] Blower=662, Wheel=1206 ... OK  prot=0.291  starch=0.659  purity=0.703  score=0.2906  (51.9s)
  [  4/50] Blower=766, Wheel=897 ... OK  prot=0.278  starch=0.542  purity=0.753  score=0.2779  (66.1s)
  [  5/50] Blower=677, Wheel=951 ... OK  prot=0.299  starch=0.584  purity=0.714  score=0.2988  (46.8s)
  [  6/50] Blower=714, Wheel=1336 ... OK  prot=0.297  starch=0.676  purity=0.727  score=0.2972  (39.2s)
  [  7/50] Blower=759, Wheel=1292 ... OK  prot=0.281  starch=0.659  purity=0.747  score=0.2812  (38.8s)
  [  8/50] Blower=702, Wheel=900 ... OK  prot=0.300  starch=0.559  purity=0.721  score=0.2996  (39.0s)
  [  9/50] Blower=789, Wheel=886 ... OK  prot=0.269  starch=0.532  purity=0.762  score=0.2690  (38.9s)
  [ 10/50] Blower=667, Wheel=1233 ... OK  prot=0.292  starch=0.664  purity=0.704  score=0.2917  (39.2s)
  [ 11/50] Blower=835, Wheel=1057 ... OK  prot=0.250  starch=0.584  purity=0.782  score=0.2501  (38.9s)
  [ 12/50] Blower=620, Wheel=1014 ... OK  prot=0.252  starch=0.618  purity=0.668  score=0.2524  (39.4s)
  [ 13/50] Blower=706, Wheel=806 ... OK  prot=0.298  starch=0.518  purity=0.726  score=0.2976  (39.0s)
  [ 14/50] Blower=607, Wheel=999 ... OK  prot=0.241  starch=0.617  purity=0.653  score=0.2411  (40.1s)
  [ 15/50] Blower=730, Wheel=1120 ... OK  prot=0.292  starch=0.623  purity=0.734  score=0.2923  (41.9s)
  [ 16/50] Blower=627, Wheel=1450 ... OK  prot=0.239  starch=0.716  purity=0.690  score=0.2393  (41.2s)
  [ 17/50] Blower=553, Wheel=925 ... OK  prot=0.182  starch=0.607  purity=0.587  score=0.1825  (39.3s)
  [ 18/50] Blower=672, Wheel=816 ... OK  prot=0.301  starch=0.532  purity=0.712  score=0.3011  (39.1s)
  [ 19/50] Blower=731, Wheel=832 ... OK  prot=0.291  starch=0.523  purity=0.734  score=0.2910  (38.8s)
  [ 20/50] Blower=805, Wheel=1102 ... OK  prot=0.263  starch=0.603  purity=0.770  score=0.2630  (42.9s)
  [ 21/50] Blower=636, Wheel=848 ... OK  prot=0.273  starch=0.555  purity=0.688  score=0.2731  (67.7s)
  [ 22/50] Blower=686, Wheel=956 ... OK  prot=0.301  starch=0.583  purity=0.714  score=0.3011  (67.6s)
  [ 23/50] Blower=693, Wheel=986 ... OK  prot=0.301  starch=0.591  purity=0.719  score=0.3008  (67.5s)
  [ 24/50] Blower=687, Wheel=968 ... OK  prot=0.301  starch=0.587  purity=0.717  score=0.3006  (67.6s)
  [ 25/50] Blower=589, Wheel=1051 ... OK  prot=0.222  starch=0.636  purity=0.630  score=0.2218  (68.6s)
  [ 26/50] Blower=648, Wheel=802 ... OK  prot=0.290  starch=0.532  purity=0.706  score=0.2902  (67.4s)
  [ 27/50] Blower=742, Wheel=1170 ... OK  prot=0.287  starch=0.633  purity=0.740  score=0.2868  (46.2s)
  [ 28/50] Blower=701, Wheel=865 ... OK  prot=0.300  starch=0.545  purity=0.722  score=0.2997  (39.0s)
  [ 29/50] Blower=601, Wheel=974 ... OK  prot=0.235  starch=0.611  purity=0.646  score=0.2348  (39.2s)
  [ 30/50] Blower=657, Wheel=1057 ... OK  prot=0.289  starch=0.622  purity=0.704  score=0.2894  (39.1s)
  [ 31/50] Blower=688, Wheel=939 ... OK  prot=0.301  starch=0.576  purity=0.716  score=0.3008  (38.9s)
  [ 32/50] Blower=686, Wheel=939 ... OK  prot=0.301  starch=0.576  purity=0.716  score=0.3010  (39.2s)
  [ 33/50] Blower=680, Wheel=1027 ... OK  prot=0.298  starch=0.608  purity=0.711  score=0.2984  (38.8s)
  [ 34/50] Blower=721, Wheel=932 ... OK  prot=0.295  starch=0.566  purity=0.731  score=0.2948  (69.7s)
  [ 35/50] Blower=647, Wheel=848 ... OK  prot=0.288  starch=0.552  purity=0.702  score=0.2878  (78.5s)
  [ 36/50] Blower=669, Wheel=1095 ... OK  prot=0.294  starch=0.629  purity=0.709  score=0.2944  (78.3s)
  [ 37/50] Blower=752, Wheel=903 ... OK  prot=0.283  starch=0.548  purity=0.745  score=0.2832  (78.2s)
  [ 38/50] Blower=686, Wheel=979 ... OK  prot=0.300  starch=0.591  purity=0.716  score=0.3000  (79.1s)
  [ 39/50] Blower=782, Wheel=877 ... OK  prot=0.271  starch=0.530  purity=0.759  score=0.2705  (71.6s)
  [ 40/50] Blower=650, Wheel=1177 ... OK  prot=0.286  starch=0.654  purity=0.702  score=0.2857  (54.9s)
  [ 41/50] Blower=720, Wheel=1245 ... OK  prot=0.295  starch=0.656  purity=0.730  score=0.2950  (58.5s)
  [ 42/50] Blower=692, Wheel=952 ... OK  prot=0.301  starch=0.580  purity=0.719  score=0.3008  (55.3s)
  [ 43/50] Blower=668, Wheel=923 ... OK  prot=0.297  starch=0.575  purity=0.710  score=0.2967  (56.1s)
  [ 44/50] Blower=708, Wheel=909 ... OK  prot=0.298  starch=0.561  purity=0.726  score=0.2980  (58.0s)
  [ 45/50] Blower=688, Wheel=1035 ... OK  prot=0.300  starch=0.608  purity=0.715  score=0.2999  (44.1s)
  [ 46/50] Blower=675, Wheel=996 ... OK  prot=0.297  starch=0.599  purity=0.711  score=0.2967  (39.0s)
  [ 47/50] Blower=738, Wheel=952 ... OK  prot=0.288  starch=0.569  purity=0.739  score=0.2879  (39.4s)
  [ 48/50] Blower=633, Wheel=829 ... OK  prot=0.268  starch=0.548  purity=0.686  score=0.2683  (39.0s)
  [ 49/50] Blower=698, Wheel=884 ... OK  prot=0.301  starch=0.553  purity=0.721  score=0.3006  (39.0s)
  [ 50/50] Blower=661, Wheel=1086 ... OK  prot=0.291  starch=0.628  purity=0.702  score=0.2910  (38.9s)

======================================================================
OPTIMIZATION COMPLETE — BAYESIAN — WHEEL-ONLY
======================================================================
  Objective:       protein_recovery
  Trials:          50
  Total wall time: 2504.5 s (41.7 min)
  Best score:      0.3011

  ──────────────────────────────────────────────────
  BEST CONFIGURATION:
  ──────────────────────────────────────────────────
    Mode:           wheel-only
    Blower RPM:     672
    Wheel RPM:      816

  RESULTS:
    Protein recovery:     0.301  (30.1%)
    Starch yield:         0.532  (53.2%)
    Protein purity:       0.712  (71.2%)
    Separation eff.:      0.465
    Total collection:     0.995  (99.5%)

  COLLECTION BREAKDOWN (of 50000 particles):
    Wheel coarse (starch)      26588 ( 53.2%) |##########################
    Cyclone 1 (fines)           7768 ( 15.5%) |#######
    Cyclone 2 (fines)            319 (  0.6%) |
    Cyclone 3 (protein)        14846 ( 29.7%) |##############
    Bag filter                   210 (  0.4%) |
    Escaped                        0 (  0.0%) |
    Still active                 269 (  0.5%) |

  ──────────────────────────────────────────────────
  TOP 5 CONFIGURATIONS:
  ──────────────────────────────────────────────────
  Rank   Blower    Wheel  ProtRec  StchYld   Purity  Collect    Score
  ────  ───────  ───────  ───────  ───────  ───────  ───────  ───────
     1      672      816    0.301    0.532    0.712    0.995   0.3011
     2      686      956    0.301    0.583    0.714    0.996   0.3011
     3      686      939    0.301    0.576    0.716    0.996   0.3010
     4      693      986    0.301    0.591    0.719    0.997   0.3008
     5      688      939    0.301    0.576    0.716    0.996   0.3008

======================================================================

  Reproduce best with:
    python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 672 --wheel-rpm 816 --wheel-only

(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier>  python examples/run_classification_flow.py --full-system --material yellow_pea --blower-rpm 672 --wheel-rpm 816 --wheel-only
======================================================================
PHYSICS-BASED CLASSIFICATION FLOW SIMULATION
  Protein/Starch Separation via Air Classification
======================================================================
  VFD: 672 RPM -> 475 m³/h (operating point)
       Fan law would give 672 m³/h (overestimate)
       P = 125 Pa, eff = 68.6%, W = 24 W
  Wheel RPM (main classifier): 816

[FULL SYSTEM] Air -> Wheel (no preclassification) -> Cyclones -> Bag Filter
  [Mode] Without preclassification (wheel-only): no venturi, zigzag, dropout

1. Air system -> Wheel junction (airclass, no venturi)...
============================================================
AIR SYSTEM -> VENTURI DUCTWORK FLOW
============================================================
  Volume flow:    475.2 m³/h
  Total dP:       5.7 Pa
  Segments:      4
------------------------------------------------------------
  duct_0             duct       v=  4.20 m/s  Re=   55436  dP=   0.1 Pa
  elbow_1            elbow      v=  4.20 m/s  Re=   55436  dP=   3.1 Pa
  duct_2             duct       v=  4.20 m/s  Re=   55436  dP=   0.8 Pa
  transition_3       transition v=  4.20 m/s  Re=   55436  dP=   1.6 Pa
============================================================

2. Feed system -> Wheel junction solids inlet (feedclass, no venturi)...
======================================================================
FEED SYSTEM -> VENTURI SOLIDS INLET (ductwork flow + kinetics)
======================================================================
  Air flow (sweep):  0.00 m3/h
  Particle d:        35.0 um
  Terminal v (vert): 0.0502 m/s
  Total dP:          0.00 Pa
  Total residence:   21.906 s
  Venturi solids D:  0 mm
----------------------------------------------------------------------
  feed_duct_0            duct       L=1.099m v_air=0.00 v_part=0.05 t_res=21.906s dP=0.0Pa
======================================================================

  Material / feed properties (used for classification validation):
    Material:        yellow_pea (density=1420 kg/m³, sphericity=0.70)
    Size range:      2.0 – 500.0 µm   d50=35.0 µm
    Feed rep. d:     35.0 µm (feed ductwork and entry rate)
    Particle rate:   4356888079 particles/s (solids mass flow + rep. d)

3. Wheel-only classification (no venturi/zigzag):
   Air flow:           475 m³/h (0.132 m³/s)
   Solids mass flow:   500.0 kg/h
  [Feed cap] Requested 184411.1 kg/h exceeds venturi capacity 1144.2 kg/h (mu=2.0)
             Capped to 1144.2 kg/h
  [Continuous feed] 100000 sim particles over 120s = 833 particles/s
  [Physical rate] 1144.2 kg/h (capped at mu=2.0)

Creating classification system assembly (from full system)...
Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 12.0
   Devices:
     "cpu"      : "AMD64 Family 25 Model 24 Stepping 1, AuthenticAMD"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
   Kernel cache:
     \\?\C:\Users\Windows\AppData\Local\NVIDIA\warp\Cache\1.11.0

  *** VENTURI COMPRESSIBILITY WARNING ***
      Throat velocity: 105 m/s
      Mach number:     0.31 (>0.3 - compressible regime)
      Bernoulli (incompressible) approximation has >5% error
      Max choked flow: 1528 m3/h

    Wheel Classifier (main classifier - centrifugal):
      Diameter:        200 mm
      RPM:             816
      Tip speed:       8.5 m/s
      G-force (rim):   74 g
      d50:             42.5 um
      Hub radius:      30.0 mm
      Blades:          24

    Particle Entry: Zone 34 (wheel housing - wheel-only mode)

  Classification Physics Parameters:

    Air Flow:
      Flow rate:       475 m3/h
      Venturi inlet:   67.2 m/s
      Venturi throat:  105.0 m/s (D=40.0mm, Ma=0.306)
      Venturi dP:      3921 Pa (3.9 kPa)
      Zigzag bulk:     0.00 m/s
      Zigzag ZONE:     0.00 m/s (30% of bulk)
      Cyclone (series, rectangular tangential inlet):
        Primary    D=300mm  inlet=75x150mm  v=11.7 m/s (weak vortex)
        Secondary  D=200mm  inlet=50x100mm  v=26.4 m/s
        Tertiary   D=120mm  inlet=30x60mm  v=73.3 m/s

    Venturi Throat Analysis:
      Throat diameter: 40.0 mm
      Throat area:     1256.6 mm2
      Max flow (Ma=1): 1528 m3/h
      K_venturi:       225072.9 Pa/(m3/s)2
      *** Ma=0.31 > 0.3: compressible regime ***

    Cut Sizes (d50) - based on ZONE velocity:
      Zigzag:          0.0 um (at v_zone=0.00 m/s)
      (if bulk):       0.0 um (wrong - ignores zone effect)
      Cy1 (primary)    4.9 um (weak vortex)
      Cy2 (secondary)  2.6 um
      Cy3 (tertiary)   1.2 um

    Multi-Stage Sharpening (0 stages):
      Each stage is a separation opportunity
      Effective cut sharpness increases with stages

    For protein separation:
      Protein:         ~10-30 um (should go to fines)
      Starch:          ~15-60 um (should go to coarse)
      Status: Zigzag d50 (0.0um) in protein range - good!

  ClassificationFlowPhysicsSimulator initialized
    Device: cuda
    Max particles: 100000

Initializing particles at wheel inlet (15° solids chute)...

  Pre-allocated 100000 particles as yellow_pea whole flour (continuous feeding)
    Feed rate: 833 particles/s
    Mass flow: 0.0 kg/h  (0.0000 kg/s)
    Time to feed all 100000 particles: 120.0 s
    Protein: 25000 (25%)  Starch: 55000 (55%)  Fiber: 20000 (20%)
    Diameter range: 5.0 - 100.0 um  Total mass: 0.01 g
    Initial zone:   34

----------------------------------------------------------------------
RUNNING SIMULATION
----------------------------------------------------------------------
  Time: 360.0 s
  dt:   1.00 ms
  Steps: 360,000
  Air flow: 475 m³/h
  Wheel d50: 42.5 µm
  Feeding: continuous at 833 particles/s
  Feed mass flow: 0.0 kg/h
  Max loading ratio: 2.0
----------------------------------------------------------------------
Module airclassifier.simulation.classification_flow_physics 5804443 load on device 'cuda:0' took 1.72 ms  (cached)
  [  0.0%] t= 0.03s | Fed:   27/100000 | Active:   27 Zc:    0 Wc:    0 Cy1:    0 Cy2:    0 Cy3:    0 Bag:    0  [zz:   0 fp:   0 wh:   4 wf:   0 wch:  19 c1:   0 c2:   0 c3:   0]
  [  5.0%] t=18.05s | Fed:15042/100000 | Active: 3210 Zc:    0 Wc: 7328 Cy1: 2247 Cy2:   38 Cy3: 2203 Bag:   16  [zz:   0 fp:   0 wh:  52 wf:   1 wch: 754 c1:1604 c2: 105 c3:   4]
  [ 10.0%] t=36.07s | Fed:30057/100000 | Active: 4057 Zc:    0 Wc:15368 Cy1: 4583 Cy2:  104 Cy3: 5898 Bag:   47  [zz:   0 fp:   0 wh: 100 wf:   1 wch: 759 c1:2412 c2: 105 c3:   1]
  [ 15.0%] t=54.09s | Fed:45072/100000 | Active: 4640 Zc:    0 Wc:23300 Cy1: 6881 Cy2:  208 Cy3: 9930 Bag:  113  [zz:   0 fp:   0 wh: 166 wf:   0 wch: 770 c1:2890 c2: 123 c3:   4]
  [ 20.0%] t=72.10s | Fed:60087/100000 | Active: 4883 Zc:    0 Wc:31368 Cy1: 9159 Cy2:  305 Cy3:14198 Bag:  174  [zz:   0 fp:   0 wh: 213 wf:   2 wch: 783 c1:3136 c2: 102 c3:   5]
  [ 25.0%] t=90.12s | Fed:75102/100000 | Active: 5160 Zc:    0 Wc:39512 Cy1:11470 Cy2:  411 Cy3:18319 Bag:  230  [zz:   0 fp:   0 wh: 262 wf:   0 wch: 764 c1:3311 c2: 101 c3:   3]
  [ 30.0%] t=108.14s | Fed:90117/100000 | Active: 5272 Zc:    0 Wc:47475 Cy1:13861 Cy2:  513 Cy3:22713 Bag:  283  [zz:   0 fp:   0 wh: 309 wf:   1 wch: 725 c1:3446 c2: 110 c3:   5]
  [ 35.0%] t=126.16s | Fed:100000/100000 | Active: 3193 Zc:    0 Wc:53371 Cy1:15529 Cy2:  614 Cy3:26956 Bag:  337  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1:2830 c2:  31 c3:   1]
  [ 40.0%] t=144.18s | Fed:100000/100000 | Active: 1829 Zc:    0 Wc:53371 Cy1:15529 Cy2:  655 Cy3:28244 Bag:  372  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1:1486 c2:  12 c3:   0]
  [ 45.1%] t=162.20s | Fed:100000/100000 | Active: 1256 Zc:    0 Wc:53371 Cy1:15529 Cy2:  671 Cy3:28787 Bag:  386  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 916 c2:   8 c3:   1]
  [ 50.1%] t=180.21s | Fed:100000/100000 | Active:  953 Zc:    0 Wc:53371 Cy1:15529 Cy2:  680 Cy3:29074 Bag:  393  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 617 c2:   5 c3:   0]
  [ 55.1%] t=198.23s | Fed:100000/100000 | Active:  806 Zc:    0 Wc:53371 Cy1:15529 Cy2:  685 Cy3:29214 Bag:  395  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 474 c2:   1 c3:   0]
  [ 60.1%] t=216.25s | Fed:100000/100000 | Active:  711 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29304 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 377 c2:   3 c3:   0]
  [ 65.1%] t=234.27s | Fed:100000/100000 | Active:  648 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29367 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 316 c2:   1 c3:   0]
  [ 70.1%] t=252.29s | Fed:100000/100000 | Active:  604 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29411 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 271 c2:   2 c3:   0]
  [ 75.1%] t=270.30s | Fed:100000/100000 | Active:  574 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29441 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 241 c2:   2 c3:   0]
  [ 80.1%] t=288.32s | Fed:100000/100000 | Active:  541 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29474 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 209 c2:   1 c3:   0]
  [ 85.1%] t=306.34s | Fed:100000/100000 | Active:  519 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29496 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 187 c2:   1 c3:   0]
  [ 90.1%] t=324.36s | Fed:100000/100000 | Active:  497 Zc:    0 Wc:53371 Cy1:15529 Cy2:  688 Cy3:29518 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 166 c2:   0 c3:   0]
  [ 95.1%] t=342.37s | Fed:100000/100000 | Active:  473 Zc:    0 Wc:53371 Cy1:15529 Cy2:  689 Cy3:29541 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 142 c2:   0 c3:   0]
  [100.0%] t=360.00s | Fed:100000/100000 | Active:  460 Zc:    0 Wc:53371 Cy1:15529 Cy2:  689 Cy3:29554 Bag:  397  [zz:   0 fp:   0 wh: 331 wf:   0 wch:   0 c1: 128 c2:   1 c3:   0]
----------------------------------------------------------------------
SIMULATION COMPLETE
----------------------------------------------------------------------
  Wall time: 73.5 s
  Sim time:  360.00 s
  Steps:     360,000
  Rate:      4900 steps/s
  Feeding:   100000/100000 particles fed (100.0%)
  Feed rate: 833 particles/s

  Separation Results (t = 360.000s):
  ==================================================
    Zigzag coarse (starch):       0 (  0.0%)
    Wheel coarse (starch):    53371 ( 53.4%)
    Cyclone 1 (fines 1):      15529 ( 15.5%)
    Cyclone 2 (fines 2):        689 (  0.7%)
    Cyclone 3 (PROTEIN):      29554 ( 29.6%)
    Bag filter:                  397 (  0.4%)
    Escaped (loss):                0 (  0.0%)
    Still active:                 460 (  0.5%)
  ==================================================

  Cyclone particle sizes (design d50 vs actual mean):
    cyclone_1            N=15529  (design d50=40 µm, mean=20.7 µm)
    cyclone_2            N=  689  (design d50=20 µm, mean=12.2 µm)
    cyclone_3_protein    N=29554  (design d50=10 µm, mean=9.7 µm)
  ==================================================
    Total (balance):       100000  (slots used: 100000)
  ==================================================

    Protein recovery (cy3 + bag): 29951
    Starch recovery (zigzag + wheel coarse): 53371
======================================================================
(venv) PS C:\Users\Windows\Desktop\Dev_Projects\airclassifier> 


