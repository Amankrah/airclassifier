# Hammer Mill Module — Build and Integration Guide

**Real-world hammer mill digital twin for the Air Classifier Designer**

This document specifies how to build the milling module separately (like the pretreatment module), implement geometry component-by-component with animations, implement physics and kinetics with NVIDIA Warp kernels, couple product flow with physics into a simulator, and pipe the module into the GUI.

Process chain: **Pretreatment (GP-15) → Hammer Mill → Air Classifier**

Reference: `utility_docs/NVIDIA_Warp_Developer_Guide.md` for Warp primitives, kernel patterns, and best practices.

---

## 1. Module Purpose and Scope

The **milling** module is a physics-based digital twin of a horizontal-shaft hammer mill (e.g. pin mill / impact mill) used in dry fractionation lines. Conditioned material from the GP-15 (or synthetic feed) enters the mill; the rotor with swinging hammers impacts the feed against the screen and housing. Particles break by impact and attrition. The screen defines the maximum product size; undersize exits as milled product for the air classifier. The module must produce:

- **Particle size distribution (PSD)** at mill discharge
- **Throughput and power draw** (for energy and process KPIs)
- **Residence time and breakage kinetics** (optional: population balance)

The simulation engine is **NVIDIA Warp**. Heavy computation — impact detection, breakage, screen classification, PSD updates — runs as JIT-compiled GPU kernels. The module supports pipeline integration: it consumes `OutletState` from pretreatment and produces a **MillingOutletState** (PSD, mass flow, moisture/temperature passthrough) for the classifier.

---

## 2. Recommended Package Structure

Mirror the pretreatment layout so the GUI and pipeline can integrate milling the same way:

```
src/airclassifier/milling/
├── BUILD_AND_INTEGRATE.md     # This file
├── __init__.py                # Public API: HammerMillSimulator, MillingOutletState, config, etc.
├── config.py                  # MillConfig, ScreenConfig, Recipe (rpm, gap, screen aperture)
├── simulator.py               # HammerMillSimulator — wraps geometry + CoupledMillingEngine + I/O
├── geometry/
│   ├── __init__.py
│   ├── machine.py             # create_hammer_mill_machine(), high-level factory
│   ├── mesh_utils.py           # Shared mesh helpers (cylinders, boxes, arcs)
│   └── assembly/
│       ├── __init__.py
│       └── machine.py         # HammerMillMachineAssembly, build_hammer_mill_meshes()
│   └── components/            # One module per physical component
│       ├── __init__.py
│       ├── rotor.py            # RotorGeometry, RotorParams — shaft + hammer pins + discs
│       ├── hammers.py          # HammerGeometry, HammerParams — swinging hammers (count, mass, tip speed)
│       ├── screen.py           # ScreenGeometry, ScreenParams — curved screen, aperture, open area
│       ├── housing.py          # HousingGeometry, HousingParams — casing, feed chute, discharge
│       ├── feed_chute.py       # FeedChuteGeometry, FeedChuteParams — inlet from pretreatment
│       └── drive.py            # DriveGeometry, DriveParams — motor, belt guard (optional)
├── physics/
│   ├── __init__.py
│   ├── coupling.py            # CoupledMillingEngine — orchestrates step sequence (see §5)
│   ├── impact.py              # ImpactSolver — collision detection hammer–particle, screen–particle
│   ├── breakage.py            # BreakageKernel — size reduction model (selection + breakage function)
│   └── screen_classifier.py   # ScreenClassifier — passage through aperture (PSD split)
├── kernels/                   # Warp JIT kernels
│   ├── __init__.py
│   ├── impact.py              # wp.kernel: distance to hammer surface, impact impulse
│   ├── breakage.py            # wp.kernel: apply breakage matrix / PSD update
│   ├── screen.py              # wp.kernel: particle–screen test, pass/retain
│   └── transport.py           # wp.kernel: particle advection in mill chamber, residence
├── control/
│   ├── __init__.py
│   └── recipe.py              # MillRecipe, speed setpoints, screen selection
├── materials/
│   ├── __init__.py
│   └── breakage_properties.py # Breakage parameters (Bond, drop-weight, or empirical)
├── io/
│   ├── __init__.py
│   └── export.py              # CSV/JSON PSD export, VTK for particles (optional)
└── tests/
    ├── __init__.py
    └── test_integration.py    # End-to-end mill run
```

---

## 3. Geometry: Component-by-Component Build and Animations

### 3.1 Design Principle

- Each **component** has a **Params** dataclass (dimensions, positions) and a **Geometry** class that produces **vertices and triangles** (and optional metadata for animation).
- **Assembly** composes components with a single world frame (e.g. Y-up, X along rotor axis, Z up for vertical mill or vice versa). Use **connection ports** (like pretreatment) so the feed chute aligns to the pretreatment outfeed and the discharge aligns to the classifier inlet.
- **Animations** are driven by simulation time: rotor angle θ(t) = θ0 + ω·t; hammers rotate with the rotor; optional belt/motor animation via a scale or placeholder mesh.

### 3.2 Components to Implement

| Component    | Params (examples) | Geometry output | Animation |
|-------------|--------------------|------------------|-----------|
| **Rotor**   | shaft_radius_m, length_m, disc_count, disc_positions | Cylinder + discs (or simplified drum) | Rotate around axis by θ(t) |
| **Hammers** | count_per_row, rows, arm_length_m, hammer_mass_kg, tip_width_m | One mesh per hammer (or instanced boxes) | Same rotation as rotor; optional swing angle if free-swinging |
| **Screen**  | inner_radius_m, arc_angle_deg, aperture_mm, thickness_m | Curved perforated surface (simplified: curved quad mesh + texture or rings) | None (fixed to housing) |
| **Housing** | length_m, radius_m, wall_thickness_m, feed_inlet_rect, discharge_rect | Casing (cylinder or box), feed opening, discharge opening | None |
| **Feed chute** | length_m, cross_section_m, flange_offset_m | Duct from “pretreatment outfeed” port to mill inlet | None |
| **Drive**    | motor_box_m, pulley_radius_m | Simple box + cylinder for motor/pulley | Optional belt rotation |

### 3.3 Assembly and Ports

- **HammerMillMachineAssembly** holds:
  - List of components and their transforms (position, rotation) in world space.
  - **Ports**: `infeed_port` (for connecting to pretreatment outfeed), `outfeed_port` (for classifier feed).
- **Parameter chain**: `MillConfig` → `RotorParams`, `ScreenParams`, `HousingParams`, `FeedChuteParams` derived from config so that screen aperture, rotor length, and housing radius are consistent.
- **build_hammer_mill_meshes()** returns a dict: `name -> (vertices, triangles, metadata)`. Metadata can include `animation_axis`, `animation_type` ("rotate"), `pivot` for the GUI to drive animations.

### 3.4 Animation in the GUI

- The GUI (e.g. **MillingPage**) gets mesh names and metadata from `build_hammer_mill_meshes()`.
- For each mesh with `animation_type == "rotate"`, the viewport applies a rotation around `animation_axis` by angle **θ = ω·t** (or from simulator-provided θ each frame).
- Rotor and hammers share the same θ. Drive pulley can use the same θ scaled by a ratio. Use the same pattern as pretreatment (e.g. `rotate_mesh_around_z_axis` or equivalent in the 3D view).

---

## 4. Physics and Kinetics (Warp Kernels)

### 4.1 Physics Overview

- **Product flow**: Particles (or parcels) enter at the feed chute; they move inside the chamber, are struck by hammers, break, and either stay in the mill or pass through the screen.
- **Kinetics**: Breakage is modeled by a **selection function** (probability of breakage per size class) and a **breakage function** (daughter PSD given breakage). Optionally use a **population balance** on size classes; or a Lagrangian particle set with size/mass attributes that get updated by breakage kernels.
- **Screen**: Particles with characteristic size &lt; aperture pass; others are retained and can be re-impacted (simplified: one pass per timestep with a passage probability by size).

### 4.2 Warp Primitives to Use (see Warp Developer Guide)

- **wp.Mesh** / **wp.Bvh**: For hammer and screen geometry; use `wp.mesh_query_point()` or ray/point queries for collision/clearance.
- **wp.HashGrid** or **wp.Bvh**: For particle–hammer and particle–screen proximity (neighbor or overlap tests).
- **wp.array**: Particle positions, velocities, sizes, masses; PSD bins; breakage state.
- **@wp.kernel**: All per-particle or per-cell work (impact, breakage, screen test, advection).
- **wp.atomic_add**: For binning PSD or counting passage events.
- **Random**: `wp.rand_init`, `wp.randf` for stochastic breakage and passage.

### 4.3 Kernel Design (High-Level)

| Kernel file   | Responsibility | Inputs / Outputs |
|---------------|----------------|-------------------|
| **impact**    | Distance from particle to hammer surface; impact impulse (normal/tangent); optional damage accumulation | positions, velocities, hammer_mesh or AABBs, ω, dt → new velocities, impact flags |
| **breakage**  | Apply selection + breakage: update particle size/mass or bin masses | sizes, masses, selection_prob, breakage_matrix → new sizes/masses or PSD bins |
| **screen**    | Test particle vs screen aperture (and optionally screen mesh); set pass/retain | positions, sizes, aperture, screen_sdf or mesh → pass_mask, discharge indices |
| **transport** | Advect particles in chamber (gravity, drag, centrifugal throw); optional residence time | positions, velocities, dt, chamber_bounds → new positions, residence |

### 4.4 Coupling Order (CoupledMillingEngine)

Each timestep, run in order:

1. **FEED** — Inject new particles from inlet (from pretreatment OutletState or synthetic) into chamber.
2. **TRANSPORT** — Advect particles (velocity update, position update, bounds).
3. **IMPACT** — Hammer–particle and (optional) housing–particle; update velocities and impact count.
4. **BREAKAGE** — Apply selection and breakage to impacted particles (update size/mass or PSD).
5. **SCREEN** — Test passage; mark particles that pass through screen.
6. **DISCHARGE** — Move passed particles to discharge buffer; update PSD and mass flow.
7. **RECORD** — Log chamber holdup, discharge rate, power (optional), PSD.

Power draw can be derived from impact energy (sum of ΔE per impact) plus a baseline (no-load); optional kernel to accumulate power.

---

## 5. Product Flow Coupling and Simulator

### 5.1 Inlet: From Pretreatment

- **OutletState** (from `airclassifier.pretreatment`) provides:
  - `temperature_field`, `moisture_field` (optional for mill moisture/temp passthrough),
  - `avg_temperature_c`, `avg_moisture_wb`, `throughput_kg_per_hr`.
- The milling module converts this to **feed**: particle stream or size distribution. If pretreatment uses Lagrangian particles, their final positions/sizes at outfeed can seed the mill feed; otherwise use a **synthetic PSD** (e.g. single size class for “conditioned whole seeds” or a prescribed PSD).
- **HammerMillSimulator.set_inlet_state(outlet_state: OutletState)** (or equivalent) sets the feed rate and optional T/M for the next run.

### 5.2 Outlet: To Classifier

- **MillingOutletState** (like `OutletState` in pretreatment) should expose:
  - **psd** — mass per size class or cumulative PSD (d50, d90, etc.),
  - **throughput_kg_per_hr**, **mass_holdup_kg**,
  - **avg_moisture_wb**, **avg_temperature_c** (passthrough from feed or simple model),
  - **power_kw** (optional).
- The classifier (and GUI) will consume this for feed PSD and flow rate.

### 5.3 Simulator Class

- **HammerMillSimulator**:
  - Holds **HammerMillMachineAssembly** (geometry) and **CoupledMillingEngine** (physics).
  - **load_recipe(MillRecipe)** — rpm, screen aperture, feed rate limits.
  - **run(duration_s)** or **step(dt)** — advance coupling loop; return **MillingResult** (PSD, throughput, power, time series).
  - **get_outlet_conditions()** → **MillingOutletState** for pipeline.
  - **get_geometry_meshes()** → same dict as `build_hammer_mill_meshes()` for the GUI to render and animate.

---

## 6. GUI Integration

### 6.1 New Page and Mode

- Add **MillingPage** under `gui/pages/` (mirror **PretreatmentPage**):
  - **Simulation View**: 3D viewport (PyVista) + control panel (recipe, run/stop, KPIs: throughput, PSD summary, power).
  - **Results View**: KPI cards, PSD plot (matplotlib), export (CSV/JSON).
- Add **MODE_MILLING** in **MainWindow**; add a third mode button “Milling” (or “Pin Mill”) next to Classification and Pretreatment.
- **QStackedWidget**: index 0 = Classification, 1 = Pretreatment, 2 = Milling. On mode switch, set current index and show/hide toolbars as needed.

### 6.2 Building the Mill in the Viewport

- When the user selects Milling mode (or opens Assembly Config with Milling enabled), call **MillingPage.build_system(assembly_params)** (or equivalent).
- **build_system**:
  - Build **HammerMillMachineAssembly** from params.
  - Call **build_hammer_mill_meshes()** and add each mesh to the PyVista scene with **COMPONENT_COLORS** (or milling-specific colors).
  - Register animated meshes (rotor, hammers) with the animation controller; each frame, set rotation from **θ = ω·t** (or from simulator if running live).

### 6.3 Assembly Config Dialog

- Add a **“Milling”** tab or stage option (e.g. radio “RF Pretreatment”, “Pin Mill”, “Air Classification” or checkboxes for which stages are enabled).
- **Milling** tab: parameters such as rotor rpm, screen aperture, feed rate, machine type (if multiple mill types exist). On Apply/Build, pass **assembly_params** including milling params to the main window so **MillingPage.build_system** receives them.

### 6.4 Pipeline Flow in GUI

- If **enable_pretreatment** and **enable_milling** are both true, the flow diagram can show: Pretreatment → Milling → Classification.
- When running in “full pipeline” mode (future), outlet of pretreatment → inlet of milling → outlet of milling → inlet of classifier. For now, document that **MillingPage** can accept **OutletState** from pretreatment (e.g. via a shared project state or “Load from Pretreatment” button) so the mill feed matches the last pretreatment run.

### 6.5 Signals and Cleanup

- **MillingPage** should emit **simulation_finished(results)** (like PretreatmentPage) so the main window can show results or pass to the next stage.
- On mode switch or close, call **MillingPage.cleanup()** to release Warp arrays, stop timers, and clear the viewport.

---

## 7. Implementation Order (Suggested)

1. **Config and geometry**
   - `config.py`: MillConfig, ScreenConfig, MillRecipe.
   - `geometry/components/`: Rotor, Hammers, Screen, Housing, FeedChute (Params + Geometry + meshes).
   - `geometry/assembly/machine.py`: HammerMillMachineAssembly, build_hammer_mill_meshes(), COMPONENT_COLORS.

2. **Kernels (Warp)**
   - `kernels/transport.py`: Chamber advection, bounds.
   - `kernels/impact.py`: Hammer–particle distance/impact (mesh or BVH).
   - `kernels/breakage.py`: Selection + breakage (PSD or particle attributes).
   - `kernels/screen.py`: Passage test by size vs aperture.

3. **Physics and coupling**
   - `physics/impact.py`, `physics/breakage.py`, `physics/screen_classifier.py` (Python wrappers that call Warp kernels).
   - `physics/coupling.py`: CoupledMillingEngine (FEED → TRANSPORT → IMPACT → BREAKAGE → SCREEN → DISCHARGE → RECORD).

4. **Simulator and API**
   - `simulator.py`: HammerMillSimulator (geometry + CoupledMillingEngine, run(), get_outlet_conditions()).
   - `__init__.py`: Export HammerMillSimulator, MillingOutletState, MillConfig, MillRecipe, etc.

5. **GUI**
   - `gui/pages/milling_page.py`: MillingPage (viewport, controls, results, build_system, cleanup).
   - `gui/pages/__init__.py`: Export MillingPage.
   - `gui/main_window.py`: MODE_MILLING, third stack index, milling button, build_system for milling, cleanup.
   - `gui/dialogs/assembly_config_dialog.py`: Milling tab, enable_milling, milling params in assembly_params.

6. **Pipeline**
   - Accept OutletState in HammerMillSimulator; produce MillingOutletState for classifier (and optional “Load from Pretreatment” in MillingPage).

---

## 8. References

- **NVIDIA Warp**: `utility_docs/NVIDIA_Warp_Developer_Guide.md` — kernels, Mesh/BVH/HashGrid, arrays, atomics, random.
- **Pretreatment**: `src/airclassifier/pretreatment/` — geometry components and assembly, CoupledSimulator, GP15Simulator, OutletState, PretreatmentPage.
- **Engineering guide**: `utility_docs/pretreatment_engineering_guide.md` — coordinate system, parameter chain, physics coupling order.

---

*This document is the single specification for building and integrating the hammer mill module into the Air Classifier Designer.*
