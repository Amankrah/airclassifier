# Milling Module Audit — BUILD_AND_INTEGRATE.md Compliance

**Date:** 2026-02-22  
**Reference:** `src/airclassifier/milling/BUILD_AND_INTEGRATE.md`

This audit checks whether the milling module is built and integrated according to the guide. Each section of the guide is checked for presence and correctness.

---

## Summary

| Section | Status | Notes |
|--------|--------|------|
| §1 Purpose and scope | ✅ | Module produces PSD, throughput, power; uses Warp; pipeline in/out specified |
| §2 Package structure | ✅ | All listed files and folders exist |
| §3 Geometry (components, assembly, animations) | ✅ | All 6 components, ports, animation metadata |
| §4 Physics and Warp kernels | ✅ | 4 kernels with @wp.kernel + NumPy fallbacks; coupling order correct |
| §5 Product flow and simulator | ✅ | `set_inlet_state(OutletState)` added; rest present |
| §6 GUI integration | ✅ | MillingPage, MODE_MILLING, Assembly Config Milling tab implemented |
| §7 Implementation order | — | Used as checklist; config/geometry/kernels/physics/simulator done |
| §8 References | — | Doc-only |

**Overall:** The milling module is **fully aligned** with the guide. **(1)** `set_inlet_state(OutletState)` has been added to `CoupledMillingEngine` and `HammerMillSimulator`. **(2)** GUI integration (§6) is implemented: MillingPage, MODE_MILLING, third stack index, Pin Mill button, Assembly Config Milling tab and enable_milling, build_system/cleanup, and simulation_finished signal.

---

## 1. Module Purpose and Scope (§1)

- **PSD at discharge:** ✅ `MillingResult` and `MillingOutletState` expose `psd_mass_fractions`, `psd_size_classes_um`, `d10_um`, `d50_um`, `d90_um`.
- **Throughput and power:** ✅ `throughput_kg_per_hr`, `mean_power_kw`, `power_kw` in result/outlet.
- **Residence time / breakage kinetics:** ✅ `mean_residence_time_s`, breakage stats in `MillingStepState`.
- **Warp as engine:** ✅ Kernels in `kernels/` use `@wp.kernel` with NumPy fallbacks.
- **Pipeline:** ✅ Consumes feed (recipe-based); produces `MillingOutletState` for classifier. **Gap:** No explicit consumption of pretreatment `OutletState` (see §5.1 below).

---

## 2. Recommended Package Structure (§2)

All paths from the guide exist:

| Path | Present | Notes |
|------|--------|------|
| `milling/BUILD_AND_INTEGRATE.md` | ✅ | |
| `milling/__init__.py` | ✅ | Exports simulator, config, geometry, physics |
| `milling/config.py` | ✅ | MillConfig, ScreenConfig, BreakageParams, MillRecipe, MillingOutletState |
| `milling/simulator.py` | ✅ | HammerMillSimulator, MillingResult, run_milling_simulation |
| `milling/geometry/` | ✅ | |
| `milling/geometry/machine.py` | ✅ | create_hammer_mill_machine() |
| `milling/geometry/mesh_utils.py` | ✅ | |
| `milling/geometry/assembly/machine.py` | ✅ | HammerMillMachineAssembly, build_hammer_mill_meshes(), COMPONENT_COLORS |
| `milling/geometry/components/rotor.py` | ✅ | RotorParams, RotorGeometry |
| `milling/geometry/components/hammers.py` | ✅ | HammerParams, HammerGeometry |
| `milling/geometry/components/screen.py` | ✅ | ScreenParams, ScreenGeometry |
| `milling/geometry/components/housing.py` | ✅ | HousingParams, HousingGeometry |
| `milling/geometry/components/feed_chute.py` | ✅ | FeedChuteParams, FeedChuteGeometry |
| `milling/geometry/components/drive.py` | ✅ | DriveParams, DriveGeometry |
| `milling/physics/coupling.py` | ✅ | CoupledMillingEngine |
| `milling/physics/impact.py` | ✅ | ImpactSolver |
| `milling/physics/breakage.py` | ✅ | BreakageModel |
| `milling/physics/screen_classifier.py` | ✅ | ScreenClassifier |
| `milling/kernels/impact.py` | ✅ | Warp + NumPy |
| `milling/kernels/breakage.py` | ✅ | Warp + NumPy |
| `milling/kernels/screen.py` | ✅ | Warp + NumPy |
| `milling/kernels/transport.py` | ✅ | Warp + NumPy |
| `milling/control/recipe.py` | ✅ | RecipeStore; MillRecipe lives in config |
| `milling/materials/breakage_properties.py` | ✅ | |
| `milling/io/export.py` | ✅ | CSV, JSON, VTK export |
| `milling/tests/test_integration.py` | ✅ | Config, geometry, kernels, physics, simulator |

---

## 3. Geometry: Component-by-Component and Animations (§3)

### 3.1 Design principle

- **Params + Geometry per component:** ✅ Each component has a `*Params` dataclass and a `*Geometry` class with `generate_mesh()` returning vertices, triangles, and metadata.
- **Single world frame / connection ports:** ✅ Assembly uses Y-up, X along rotor; `assembly.ports` returns `infeed_port` and `outfeed_port` (e.g. from feed_chute inlet and housing discharge).
- **Parameter chain:** ✅ `MillConfig` → `RotorParams`, `HammerParams`, `ScreenParams`, `HousingParams`, `FeedChuteParams`, `DriveParams` via `from_mill_config` / `from_housing` / `from_rotor`.

### 3.2 Components

| Component | Params | Geometry | Mesh output | Animation |
|-----------|--------|----------|-------------|----------|
| Rotor | RotorParams | RotorGeometry | shaft + discs | ✅ `animation_type: "rotate"`, `animation_axis: "x"` |
| Hammers | HammerParams | HammerGeometry | hammer meshes | ✅ `animation_type: "rotate"`, `animation_axis: "x"` |
| Screen | ScreenParams | ScreenGeometry | curved surface | ✅ Static |
| Housing | HousingParams | HousingGeometry | casing, openings | ✅ Static |
| Feed chute | FeedChuteParams | FeedChuteGeometry | duct | ✅ Static |
| Drive | DriveParams | DriveGeometry | motor/pulley | ✅ Static (optional rotate noted in drive.py) |

### 3.3 Assembly and build_hammer_mill_meshes()

- **HammerMillMachineAssembly:** ✅ Holds all component params and geometries; builds from config.
- **build_hammer_mill_meshes(config, resolution):** ✅ Returns `name -> (vertices, triangles, metadata)`.
- **Metadata:** ✅ Includes `animation_type`, `animation_axis`, `pivot` where applicable (rotor, hammers).
- **get_animated_components():** ✅ Returns dict of components with animation metadata for GUI.

### 3.4 Animation in GUI

- **Guide:** GUI gets meshes and metadata from `build_hammer_mill_meshes()` and applies rotation θ = ω·t for `animation_type == "rotate"`.
- **Implementation:** Geometry and metadata are ready. **GUI side:** No MillingPage exists yet to consume them (see §6).

---

## 4. Physics and Kinetics — Warp Kernels (§4)

### 4.1–4.2 Physics overview and Warp primitives

- Product flow (feed → transport → impact → breakage → screen → discharge) is implemented in `CoupledMillingEngine`.
- Kernels use `wp.array`, `@wp.kernel`; NumPy fallbacks used when Warp is unavailable. No `wp.Mesh`/`wp.Bvh` in kernels (impact uses analytical hammer-sweep zone); acceptable per guide “hammer_mesh or AABBs”.

### 4.3 Kernel design

| Kernel | Responsibility | Implemented |
|--------|----------------|------------|
| impact | Distance/impulse hammer–particle; impact flags and energies | ✅ `impact_detection_kernel` (Warp) + `impact_detection_np` |
| breakage | Selection + breakage; size/mass update | ✅ `breakage_step_*` (Warp + NumPy) |
| screen | Particle vs aperture; pass/retain | ✅ `screen_passage_*` (Warp + NumPy) |
| transport | Advect in chamber; residence | ✅ `transport_step_*` (Warp + NumPy) |

### 4.4 Coupling order (CoupledMillingEngine.step)

Order in `physics/coupling.py` matches the guide exactly:

1. **FEED** — Inject new particles from inlet  
2. **TRANSPORT** — Advect particles  
3. **IMPACT** — Hammer–particle collisions  
4. **BREAKAGE** — Apply size reduction to impacted particles  
5. **SCREEN** — Test passage  
6. **DISCHARGE** — Move passed particles to outlet  
7. **RECORD** — Log state and KPIs  

---

## 5. Product Flow Coupling and Simulator (§5)

### 5.1 Inlet from pretreatment

- **Guide:** “HammerMillSimulator.set_inlet_state(outlet_state: OutletState) (or equivalent) sets the feed rate and optional T/M for the next run.”
- **Current:** No `set_inlet_state(OutletState)`. Feed is driven by `MillRecipe` (`feed_rate_kg_per_hr`, `feed_moisture_wb`, `feed_temperature_c`, `feed_d50_um`). Engine uses `_feed_rate_kg_per_s` and recipe-based feed.
- **Done:** `set_inlet_state(outlet_state)` implemented on `CoupledMillingEngine` and `HammerMillSimulator`; feed rate and T/M passthrough from outlet; `MillingResult.to_outlet_state(avg_temperature_c=..., avg_moisture_wb=...)` and `get_outlet_conditions()` use engine inlet T/M.

### 5.2 Outlet to classifier

- **MillingOutletState:** ✅ Exposes `psd_mass_fractions`, `psd_size_classes_um`, `d10_um`, `d50_um`, `d90_um`, `throughput_kg_per_hr`, `mass_holdup_kg`, `avg_moisture_wb`, `avg_temperature_c`, `power_kw`, `specific_energy_kwh_per_t`, `mean_residence_time_s` (in result/to_outlet_state).

### 5.3 Simulator class

| Requirement | Status |
|-------------|--------|
| Holds HammerMillMachineAssembly and CoupledMillingEngine | ✅ |
| load_recipe(MillRecipe) | ✅ |
| run(duration_s) / step(dt) | ✅ |
| Return MillingResult (PSD, throughput, power, time series) | ✅ |
| get_outlet_conditions() → MillingOutletState | ✅ |
| get_geometry_meshes() → same dict as build_hammer_mill_meshes() | ✅ Uses assembly.get_component_meshes() |

Additional helpers present: `get_animated_components()`, `get_rotor_angle()`, `get_particle_positions()`, `get_particle_sizes()`.

---

## 6. GUI Integration (§6)

**Status: ✅ Implemented.**

| Requirement | Status |
|-------------|--------|
| **MillingPage** under `gui/pages/` (Simulation View + Results View) | ✅ `milling_page.py` with viewport, control panel, results view |
| **MODE_MILLING** in MainWindow | ✅ |
| Third mode button “Pin Mill” | ✅ |
| QStackedWidget index 2 for Milling | ✅ Stack: 0=Classification, 1=Pretreatment, 2=Milling |
| **MillingPage.build_system(assembly_params)** | ✅ Builds mill from mill_rotor_rpm, mill_screen_aperture_mm |
| Build mill meshes, COMPONENT_COLORS | ✅ |
| **Assembly Config:** “Milling” tab and stage option | ✅ Milling tab + radio “Pin Mill” in Stages |
| enable_milling, milling params in assembly_params | ✅ mill_rotor_rpm, mill_screen_aperture_mm, mill_feed_rate_kg_per_hr |
| Flow diagram: Pretreatment → Milling → Classification | ✅ update_flow() includes milling_enabled |
| **MillingPage.simulation_finished(results)** signal | ✅ Connected to _on_milling_finished |
| **MillingPage.cleanup()** on mode switch/close | ✅ Called in closeEvent |

---

## 7. Implementation Order (§7)

Used as a checklist:

1. **Config and geometry** — ✅ Done (config, all components, assembly, build_hammer_mill_meshes, COMPONENT_COLORS).
2. **Kernels (Warp)** — ✅ Done (transport, impact, breakage, screen).
3. **Physics and coupling** — ✅ Done (impact, breakage, screen_classifier wrappers; CoupledMillingEngine with 7-step order).
4. **Simulator and API** — ✅ Done (HammerMillSimulator, run, get_outlet_conditions, get_geometry_meshes; __init__ exports).
5. **GUI** — ✅ Done (MillingPage, MODE_MILLING, Assembly Config Milling tab and enable_milling).
6. **Pipeline** — ✅ Inlet via `set_inlet_state(OutletState)`; outlet via `get_outlet_conditions()`.

---

## 8. References (§8)

Documentation references are correct. No code changes required.

---

## Completed action items

1. **`set_inlet_state(OutletState)`** — Implemented on `CoupledMillingEngine` and `HammerMillSimulator`; feed rate and T/M passthrough; `MillingResult.to_outlet_state(avg_temperature_c=..., avg_moisture_wb=...)` and `get_outlet_conditions()` use engine inlet state.
2. **GUI integration (§6)** — Implemented: `gui/pages/milling_page.py`, MODE_MILLING, third stack page, Pin Mill button, build_system/cleanup, simulation_finished → _on_milling_finished; Assembly Config Milling tab and enable_milling (mill_rotor_rpm, mill_screen_aperture_mm, mill_feed_rate_kg_per_hr); flow diagram updated for milling.

The milling module is now fully aligned with BUILD_AND_INTEGRATE.md.
