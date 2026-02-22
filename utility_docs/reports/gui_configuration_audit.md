# GUI Configuration Audit: Configuring and Running Three Systems Independently

**Purpose:** Audit how the GUI configures the process stages (Pretreatment, Classification, Milling) and recommend a more user-friendly way to configure each system independently and run them separately.

**References:**
- Main window: `src/airclassifier/gui/main_window.py`
- Assembly config dialog: `src/airclassifier/gui/dialogs/assembly_config_dialog.py`
- Classification page: `src/airclassifier/gui/pages/classification_page.py`
- Pretreatment page: `src/airclassifier/gui/pages/pretreatment_page.py`
- Milling module (no GUI yet): `src/airclassifier/milling/`

---

## 1. The Three Systems

| System | Description | GUI today | Backend |
|--------|-------------|-----------|---------|
| **RF Pretreatment** | GP-15 dielectric heating (moisture conditioning) | Full page: build, run, 3D, results, export | `pretreatment` (GP15Simulator, coupling) |
| **Air Classification** | Zigzag + wheel + cyclones (protein–starch separation) | Full page: build, run, 3D, animation, results | `simulation_backend`, classification flow |
| **Milling** | Hammer mill (size reduction after pretreatment) | **None** | `milling` (HammerMillSimulator, config, geometry) |

Process chain (conceptual): **Pretreatment → Milling → Classification**.

---

## 2. Current Configuration and Run Model

### 2.1 Single Assembly Config Dialog

- **Entry:** Menu “Assembly → Configure Assembly...” (Ctrl+Shift+A) or welcome “Load preset”.
- **Content:** One dialog with four tabs:
  1. **Stages** — “Active Process Stage” (radio: **RF Pretreatment** or **Air Classification**), Classification mode (Full / Wheel-only), Subsystems (feed, air, exhaust, ductwork, etc.), Design operating point (throughput, air flow).
  2. **RF Pretreatment** — Material, moisture, bed depth, electrode gap, belt speed, fan, MRH, heaters, duration, oscillator efficiency.
  3. **Classification** — Venturi, zigzag, wheel, cyclones, bag filter, etc.
  4. **Sizing** — Hopper, main duct, stack height.

- **Output:** One flat `assembly_params` dict passed to `MainWindow._on_assembly_configured(params)`. All systems’ parameters live in this single dict (e.g. `enable_pretreatment`, `enable_classification`, `pt_*`, `venturi_*`, `wheel_*`, …).

### 2.2 “Active Stage” Is Mutually Exclusive

- In the Stages tab, the user picks **one** “Active Process Stage”: either RF Pretreatment or Air Classification.
- On Apply/Build:
  - `enable_pretreatment` / `enable_classification` drive which mode is switched to (e.g. if pretreatment is checked, app switches to Pretreatment mode).
  - **Build Full System** (Ctrl+B) builds only the **current** mode’s page (pretreatment page or classification page).
- So at any time only one system is “active”; the other is not built. The user cannot have both built and switch between running each without re-opening the dialog and switching the radio.

### 2.3 Run Behavior

- **Classification mode:** F5 “Run Simulation” runs the classification page’s simulation (and toolbar Run/Pause/Stop apply to classification only). Simulation toolbar is visible.
- **Pretreatment mode:** F5 is not wired to pretreatment; the user runs simulation via the **Run** button on the pretreatment page itself. Simulation toolbar is hidden in this mode.
- So “run” is effectively per-page, but “configure” and “build” are global and mode-dependent.

### 2.4 Project and Sync

- **Project file** stores a single `assembly_params` (and `assembly`, viewport state). There is no separate storage per system (e.g. `pretreatment_params`, `classification_params`, `milling_params`).
- When assembly config is applied, **classification** page gets `sync_settings_from_params(params)` so its internal settings (e.g. sim control) are updated from the shared params. Pretreatment page reads from the same params when it builds (e.g. `pt_electrode_gap_mm`, `pt_bed_depth_mm` from `assembly_params`).

### 2.5 Component Count (Status Bar)

- “X subsystems” is computed from classification-centric flags: `include_feed_system`, `include_air_system`, `include_exhaust`. It does not reflect pretreatment or milling.

---

## 3. Pain Points (User-Unfriendly Aspects)

1. **One big dialog for everything** — Pretreatment and classification parameters are in different tabs but live in one blob. Users who only care about one system still see the other’s options and the “Stages” logic.
2. **Mutually exclusive “active stage”** — Cannot keep both systems configured and built, then run one or the other. Switching stage rebuilds the other system’s context.
3. **“Build Full System” is misleading** — It only builds the **current** mode’s system, not “all” systems. Name suggests building the entire process chain.
4. **No per-system “Configure & Build”** — There is no “Configure Pretreatment only” or “Build Classification only” that leaves the other system untouched. Every build is tied to the global dialog and current mode.
5. **Run semantics differ by mode** — In classification mode, F5 runs classification; in pretreatment mode, Run is only on the page. No consistent “Run current system” or explicit “Run Pretreatment” / “Run Classification”.
6. **Milling is absent** — The third system (milling) has no GUI; it cannot be configured or run from the app.
7. **Single params blob in project** — Saving/loading does not separate by system; loading a project overwrites all params, so per-system presets (e.g. “GP-15 Run#2” vs “Wheel-only 500 kg/h”) are not first-class.
8. **Stages tab mixes concerns** — “Which stage is active” (radio) is mixed with classification subsystems (feed, air, exhaust) and design point. Users configuring only pretreatment still see classification options in the first tab.

---

## 4. Recommended Direction: Independent Configuration and Run

Goal: **Configure each system independently, build per system, and run each system separately**, with clear context and minimal cognitive load.

### 4.1 Per-System Configuration (Not One Monolithic Dialog)

- **Option A — Per-system config panels/dialogs**
  - **Pretreatment:** “Configure Pretreatment” (or a dedicated panel on the Pretreatment page) with only GP-15 / feedstock / recipe / simulation parameters. No classification or milling fields.
  - **Classification:** “Configure Classification” with only classifier geometry, mode (full / wheel-only), subsystems (feed, air, exhaust, etc.), and design point.
  - **Milling (future):** “Configure Milling” with only mill/screen/breakage parameters.
  - A **Process flow** view (optional) can show “Pretreatment → Milling → Classification” and which stages are enabled, without embedding all parameters there.

- **Option B — Single dialog but clearly scoped by system**
  - Keep one dialog but restructure: e.g. “Configure: [Pretreatment] [Classification] [Milling]” as top-level choice; only the selected system’s parameters are shown. No “active stage” radio that also switches app mode; “Apply” only updates that system’s params and optionally triggers “Build this system only”.

Recommendation: **Option A** is more user-friendly: each page owns its configuration entry point and parameters, so the user always knows “I am configuring Pretreatment” when on the Pretreatment page.

### 4.2 Independent Enable, Build, and Run

- **Enable** — Allow more than one system to be “enabled” at once (e.g. checkboxes: Enable Pretreatment, Enable Classification, Enable Milling). The flow diagram can show all enabled stages. This replaces the single “active stage” radio.
- **Build** — Provide explicit actions:
  - “Build Pretreatment” (only builds GP-15 geometry on the pretreatment page),
  - “Build Classification” (only builds classifier on the classification page),
  - “Build Milling” (when milling has a page),
  - and optionally “Build All” (builds every enabled system).
  - “Build Full System” (Ctrl+B) should be redefined to “Build current page’s system” or “Build All” so the name matches behavior.
- **Run** — Explicit and consistent:
  - “Run” (or F5) = run the **current page’s** simulation (Classification page or Pretreatment page or Milling page). No hidden difference between modes.
  - Optional: toolbar or menu entries “Run Pretreatment”, “Run Classification”, “Run Milling” that switch to that page and run (or run in background if desired later).

### 4.3 Navigation and Context

- **Mode switching** — Keep clear “Classification | Pretreatment | Milling” (tabs or toolbar). When the user is on a page, the window title or status bar can show “ProteinProcessIO — [Classification]” so context is obvious.
- **Config entry** — On each page, a “Configure” or “Settings” button opens **that system’s** config (panel or dialog). No need to open a global “Assembly config” unless the user wants to see the overall process flow or enable/disable stages.

### 4.4 Stored Config Per System

- **Project file** — Store three (or two until milling exists) separate dicts, e.g. `pretreatment_params`, `classification_params`, `milling_params`. Optionally keep a single `assembly_params` for backward compatibility and derive it from the three when saving old format.
- **Presets** — Allow “Save preset” / “Load preset” **per system** (e.g. “GP-15 Run#2”, “Wheel-only 500 kg/h”, “Mill screen 2 mm”). Loading a preset updates only that system’s params and optionally builds that system only.

### 4.5 Milling Integration (When Ready)

- Add a **Milling** page (or tab) with its own 3D view (if needed), controls, and results.
- Add “Configure Milling” (panel or dialog) using `MillConfig`, `ScreenConfig`, `BreakageParams`, `MillRecipe`.
- Add “Build Milling” and “Run Milling” so milling can be configured and run independently, and the process flow shows Pretreatment → Milling → Classification.

---

## 5. Summary: Current vs Recommended

| Aspect | Current | Recommended |
|--------|--------|-------------|
| Config UI | One dialog, all systems in tabs; “Active stage” radio | Per-system config (panel or dialog per system); optional process-flow view |
| Enable | One “active” stage (mutually exclusive) | Multiple systems can be enabled; build/run per system |
| Build | “Build Full System” = build current mode only | “Build [System]” per page; “Build All” optional; rename so it’s clear |
| Run | F5 = classification only; pretreatment Run on page | F5 (and Run) = run **current page**; optional “Run [System]” actions |
| Project | Single `assembly_params` | `pretreatment_params`, `classification_params`, `milling_params` (and optional legacy blob) |
| Presets | Global preset (classification-oriented) | Per-system presets (e.g. GP-15 Run#2, Wheel-only 500 kg/h) |
| Milling | No GUI | Milling page + Configure Milling + Build/Run Milling |

---

## 6. Implementation Hints (No Code Yet)

1. **Split params in MainWindow**  
   Maintain `_pretreatment_params`, `_classification_params`, `_milling_params` (or keep `_assembly_params` as a merged view for backward compatibility). On load/save, read/write per-system keys.

2. **Pretreatment config**  
   Add “Configure” on Pretreatment page that opens a dialog (or inline panel) with only `pt_*` and related fields. On Apply, update `_pretreatment_params` and optionally call `pretreatment_page.build_system(pretreatment_params_only)`.

3. **Classification config**  
   Similarly, “Configure” on Classification page (or reuse a simplified Assembly dialog scoped to “Classification” only) updates `_classification_params` and builds only classification when “Build” is clicked.

4. **Assembly dialog evolution**  
   Either (a) replace the current dialog with a “Process flow” dialog that only enables/disables stages and links to per-system config, or (b) keep one dialog but make the first tab “Which systems to enable” (checkboxes) and then “Configure Pretreatment”, “Configure Classification”, “Configure Milling” as sub-tabs that only show that system’s params.

5. **Build actions**  
   “Build Full System” → “Build current system” (current page) or add “Build Classification”, “Build Pretreatment”, “Build Milling” and “Build All” so behavior is explicit.

6. **Run**  
   Unify so F5 runs the active page’s simulation (classification or pretreatment); ensure pretreatment page’s Run button and F5 both trigger the same behavior when in pretreatment mode.

7. **Milling**  
   When adding the milling UI, add `MillingPage`, a milling config dialog/panel, and Build/Run for milling; add `_milling_params` to project and to any process-flow view.

---

**Conclusion:** The current GUI uses a single assembly config and a single “active stage,” which makes it harder to configure and run Pretreatment and Classification independently. A more user-friendly approach is: **per-system configuration**, **independent enable/build/run per system**, **clear navigation (which system I’m on)**, and **per-system stored params and presets**. Milling should be integrated the same way when it is added to the GUI.
