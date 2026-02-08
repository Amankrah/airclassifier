"""
Simulation Control Panel
========================

Panel for controlling simulation execution and displaying progress.
Uses the real ClassificationFlowPhysicsSimulator from classification_flow_physics.py.

Settings are aligned with ClassificationFlowConfig and run_classification_flow.py
defaults so the GUI produces the same results as the CLI example.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import traceback

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QProgressBar, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QFrame, QTabWidget, QTextEdit,
    QGridLayout, QSizePolicy, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QObject
from PySide6.QtGui import QColor, QFont, QTextCursor

from ..theme import COLORS


# ============================================================================
# Settings dataclass -- aligned with ClassificationFlowConfig + CLI defaults
# ============================================================================

@dataclass
class SimulationSettings:
    """
    Settings for simulation execution.

    Defaults match ``run_classification_flow.py`` and ``ClassificationFlowConfig``
    so the GUI produces identical results to the CLI.
    """

    # --- Time (CLI: --time 360, --dt 0.001) ---
    total_time: float = 360.0           # [s]  CLI default 360
    dt: float = 0.001                   # [s]  ClassificationFlowConfig default
    output_interval: float = 1.0        # [s]  GUI refresh rate (every 1000 steps)

    # --- Particles (CLI: --particles 100000) ---
    num_particles: int = 5000           # GUI-friendly default (CLI is 100k)
    particle_feed_rate: float = 0.0     # [/s]  0 = auto (engine computes from mass flow)
    continuous_feeding: bool = True      # CLI --full-system default

    # --- Particle size (CLI: --particle-dia 50, --particle-std 30) ---
    particle_diameter_um: float = 50.0  # [um]  mean diameter when no material preset
    particle_std_um: float = 30.0       # [um]  std dev when no material preset

    # --- Physics (ClassificationFlowConfig defaults) ---
    turbulence_base: float = 0.15  # Base turbulence (scales zigzag=0.25, cyclone=0.12)
    restitution: float = 0.3            # ClassificationFlowConfig default
    friction: float = 0.4               # ClassificationFlowConfig default
    bypass_ratio: float = 0.0           # CLI --bypass-ratio 0.0
    max_loading_ratio: float = 2.0      # CLI --max-loading 2.0

    # --- Compute (CLI: --device cuda) ---
    device: str = "cuda"
    precision: str = "float32"

    # --- Material (CLI: --material yellow_pea) ---
    material_source: str = "yellow_pea" # yellow_pea, faba_bean, oat, or "none"
    material_fraction: str = "whole"    # whole, protein, starch, fiber

    # --- Visualization ---
    show_particles: bool = True
    show_velocity_field: bool = False
    particle_color_mode: str = "velocity"

    # --- Recirculation (CLI: --recirculate, --passes, --attrition) ---
    recirculate_passes: int = 1                    # Number of passes (1 = single pass)
    recirculate_fractions: str = "cy1"             # Comma-separated: cy1,cy2,wheel_coarse,...
    attrition_factor: float = 0.10                 # Diameter reduction per pass (0.0 = off)
    attrition_min_um: float = 5.0                  # [µm] Floor diameter for attrition
    recirculate_wheel_rpm: float = 0.0             # Wheel RPM for passes 2+ (0 = same)
    recirculate_time: float = 0.0                  # Sim time for passes 2+ (0 = same as total_time)

    # --- Assembly mode (CLI: default preclassification, --wheel-only) ---
    use_preclassification: bool = True

    # --- Wheel classifier (CLI: --wheel-rpm defaults from geometry ~8000) ---
    wheel_diameter: float = 0.20        # [m]
    wheel_rpm: float = 8000.0           # [RPM]

    # --- Air flow (CLI default: 1768 m³/h ≈ 0.491 m³/s) ---
    air_flow_m3s: float = 0.491         # [m³/s]
    blower_rpm: float = 0.0             # [RPM] 0 = use air_flow_m3s directly; >0 overrides via operating point

    # --- Geometry overrides (CLI: --throat-diameter, --zigzag-width/depth) ---
    venturi_throat_diameter_mm: float = 0.0   # [mm] 0 = use geometry default (40mm)
    zigzag_width_mm: float = 0.0              # [mm] 0 = use geometry default (120mm)
    zigzag_depth_mm: float = 0.0              # [mm] 0 = use geometry default (200mm)

    # --- Complete system (used by Build Full System) ---
    include_feed_system: bool = True
    include_air_system: bool = True
    include_exhaust: bool = True


# ============================================================================
# Worker -- uses the REAL ClassificationFlowPhysicsSimulator
# ============================================================================

class SimulationWorker(QObject):
    """
    Worker that runs ClassificationFlowPhysicsSimulator on a background thread.

    Follows the same orchestration as ``run_classification_flow.py``:
    1. Build ClassificationSystemAssembly
    2. Create ClassificationFlowConfig
    3. Create ClassificationFlowPhysicsSimulator
    4. Initialize particles (whole flour / material / generic)
    5. Step loop with progress reporting
    """

    progress_updated = Signal(int, float, dict)  # (percent, sim_time, stats)
    component_state_updated = Signal(dict)       # physics-driven component states
    simulation_completed = Signal(dict)           # final results
    simulation_error = Signal(str)                # error + traceback
    log_message = Signal(str)

    def __init__(self, settings: SimulationSettings):
        super().__init__()
        self.settings = settings
        self._is_running = False
        self._is_paused = False

        # Populated during run() for physics-driven animation
        self._blower_operating_point: dict = {}  # from airclass compute_blower_operating_point()
        self._air_flow_m3s: float = 0.0          # resolved air flow rate

        # Subsidiary physics simulators (stepped alongside classification sim)
        self._air_sim = None   # AirFlowPhysicsSimulator (blower ramp, damper positions)
        self._feed_sim = None  # FeedFlowPhysicsSimulator (lid angle, feed phase)

    # ----------------------------------------------------------------

    def run(self):
        """Build assembly, init particles, run simulation -- matching CLI flow."""
        try:
            self._is_running = True
            s = self.settings

            self.log_message.emit("Importing simulation engine...")

            from ...simulation.classification_flow_physics import (
                ClassificationFlowPhysicsSimulator,
                ClassificationFlowConfig,
            )
            from ...geometry.assembly.classification import (
                ClassificationSystemAssembly,
                ClassificationSystemParams,
            )
            from ...particles import FluidConfig, ParticleMaterial

            # ==============================================================
            # 0. Resolve blower RPM -> air flow
            # ==============================================================
            air_flow = s.air_flow_m3s
            if s.blower_rpm > 0:
                from ...simulation.airclass_flow_physics import compute_blower_operating_point
                op = compute_blower_operating_point(s.blower_rpm)
                air_flow = op["Q_m3_s"]
                self._blower_operating_point = op  # store for physics-driven animation
                self.log_message.emit(
                    f"  Blower: {s.blower_rpm:.0f} RPM \u2192 {op['Q_m3_h']:.0f} m\u00b3/h "
                    f"({air_flow:.3f} m\u00b3/s)"
                )
            self._air_flow_m3s = air_flow

            # ==============================================================
            # 1. Build classification assembly with geometry overrides
            # ==============================================================
            self.log_message.emit("Building classification assembly...")
            params = ClassificationSystemParams()
            params.use_preclassification = s.use_preclassification
            # Geometry overrides (same as CLI --throat-diameter, --zigzag-width/depth)
            if s.venturi_throat_diameter_mm > 0:
                throat_m = s.venturi_throat_diameter_mm / 1000.0
                params.venturi_throat_ratio = throat_m / params.venturi_inlet_diameter
                self.log_message.emit(f"  Venturi throat: {s.venturi_throat_diameter_mm:.1f} mm")
            if s.zigzag_width_mm > 0:
                params.zigzag_channel_width = s.zigzag_width_mm / 1000.0
                self.log_message.emit(f"  Zigzag width: {s.zigzag_width_mm:.0f} mm")
            if s.zigzag_depth_mm > 0:
                params.zigzag_channel_depth = s.zigzag_depth_mm / 1000.0
                self.log_message.emit(f"  Zigzag depth: {s.zigzag_depth_mm:.0f} mm")
            assembly = ClassificationSystemAssembly(params=params)

            mode_str = "Full System" if s.use_preclassification else "Wheel-Only"
            self.log_message.emit(f"  Mode: {mode_str}")
            if assembly.venturi is not None:
                self.log_message.emit("  Components: Venturi + Zigzag + Wheel + Cyclones + Bag")
            else:
                self.log_message.emit("  Components: Junction + Wheel + Cyclones + Bag")

            # ==============================================================
            # 2. Material + FluidConfig
            # ==============================================================
            material = None
            fluid = FluidConfig.air_at_stp()
            use_material = s.material_source not in ("none", "")

            if use_material:
                if s.material_source in ("yellow_pea", "faba_bean", "oat"):
                    fraction = s.material_fraction if s.material_fraction != "whole" else "whole"
                    material = ParticleMaterial.create_food_powder(s.material_source, fraction)
                elif s.material_source in ("protein", "starch", "fiber"):
                    material = ParticleMaterial.create_food_powder("yellow_pea", s.material_source)
                if material:
                    self.log_message.emit(f"  Material: {material.name}")

            # ==============================================================
            # 3. Build ClassificationFlowConfig (same fields as CLI)
            # ==============================================================

            # Auto-compute feed rate when continuous feeding is on but rate is 0
            feed_rate = s.particle_feed_rate
            if s.continuous_feeding and feed_rate <= 0 and s.num_particles > 0:
                feed_duration = min(s.total_time * 0.5, 120.0)
                feed_duration = max(feed_duration, 1.0)
                feed_rate = s.num_particles / feed_duration
                self.log_message.emit(
                    f"  Auto feed rate: {feed_rate:.0f} /s "
                    f"({s.num_particles} over {feed_duration:.0f}s)"
                )

            _ts = s.turbulence_base / 0.15  # scale from base
            config = ClassificationFlowConfig(
                num_particles=s.num_particles,
                air_flow_rate_m3s=air_flow,
                bypass_ratio=s.bypass_ratio,
                dt=s.dt,
                turbulence_zigzag=0.25 * _ts,
                turbulence_cyclone=0.12 * _ts,
                restitution=s.restitution,
                friction=s.friction,
                device=s.device,
                continuous_feeding=s.continuous_feeding,
                particle_feed_rate=feed_rate,
                max_loading_ratio=s.max_loading_ratio,
                fluid_config=fluid,
                material=material,
                wheel_rpm=s.wheel_rpm,
            )

            self.log_message.emit(
                f"  Particles: {s.num_particles:,}   dt={s.dt}s   "
                f"Q={air_flow:.3f} m\u00b3/s ({air_flow * 3600:.0f} m\u00b3/h)"
            )
            self.log_message.emit(
                f"  Device: {s.device}   Wheel: {s.wheel_rpm:.0f} RPM   "
                f"Bypass: {s.bypass_ratio:.0%}"
            )

            # ==============================================================
            # 4. Create simulator
            # ==============================================================
            self.log_message.emit("Initializing Warp simulator...")
            sim = ClassificationFlowPhysicsSimulator(assembly, config)

            # ==============================================================
            # 5. Initialize particles (critical -- same as run_classification_flow.py)
            # ==============================================================
            if s.use_preclassification:
                self.log_message.emit("Initializing particles at venturi solids inlet...")
            else:
                self.log_message.emit("Initializing particles at wheel inlet (15\u00b0 solids chute)...")

            if use_material and material is not None:
                if s.material_source in ("yellow_pea", "faba_bean", "oat") and s.material_fraction == "whole":
                    # Whole flour population (protein + starch + fiber)
                    sim.initialize_whole_flour_population(
                        source=s.material_source,
                        num_particles=s.num_particles,
                    )
                    self.log_message.emit(f"  Whole flour population: {s.material_source}")
                else:
                    # Single fraction via material
                    sim.initialize_particles_from_material(
                        material=material,
                        num_particles=s.num_particles,
                    )
                    self.log_message.emit(f"  Material population: {material.name}")
            else:
                # Generic particles from diameter/std
                mean_dia_m = s.particle_diameter_um * 1e-6
                std_dia_m = s.particle_std_um * 1e-6
                sim.initialize_particles(
                    num_particles=s.num_particles,
                    mean_diameter=mean_dia_m,
                    diameter_std=std_dia_m,
                )
                self.log_message.emit(
                    f"  Generic particles: d={s.particle_diameter_um:.0f} \u00b1 "
                    f"{s.particle_std_um:.0f} \u00b5m"
                )

            # ==============================================================
            # 6. Multi-pass recirculation loop
            # ==============================================================
            num_passes = s.recirculate_passes if s.recirculate_passes > 1 else 1
            recirc_fractions = [
                f.strip() for f in s.recirculate_fractions.split(",") if f.strip()
            ] if num_passes > 1 else []

            # Cumulative collection across passes
            cumulative = {
                'coarse': 0, 'wheel_coarse': 0, 'cyclone_1': 0,
                'cyclone_2': 0, 'cyclone_3_protein': 0, 'bagfilter': 0,
                'escaped': 0, 'active': 0,
            }
            pass_results_list = []
            # Cumulative cyclone diameter data for merged stats
            import numpy as _np
            cumul_cy_diameters = {'cyclone_1': [], 'cyclone_2': [], 'cyclone_3_protein': []}

            if num_passes > 1:
                self.log_message.emit(
                    f"Multi-pass recirculation: {num_passes} passes, "
                    f"fractions=[{', '.join(recirc_fractions)}], "
                    f"attrition={s.attrition_factor*100:.0f}%"
                )

            import time as _time
            t_start = _time.perf_counter()

            for pass_num in range(1, num_passes + 1):
                if not self._is_running:
                    break

                pass_sim_time = s.total_time
                if pass_num > 1 and s.recirculate_time > 0:
                    pass_sim_time = s.recirculate_time

                total_steps = int(pass_sim_time / s.dt)
                output_steps = max(1, int(s.output_interval / s.dt))

                if num_passes > 1:
                    self.log_message.emit(f"--- Pass {pass_num}/{num_passes} "
                                          f"({total_steps:,} steps, {pass_sim_time:.0f}s) ---")
                else:
                    self.log_message.emit(f"Running {total_steps:,} steps ({s.total_time:.0f}s)...")

                for step in range(total_steps):
                    if not self._is_running:
                        break

                    while self._is_paused and self._is_running:
                        QThread.msleep(100)

                    sim.step()

                    # Progress report at output_interval
                    if step > 0 and step % output_steps == 0:
                        # Overall progress across all passes
                        pass_frac = (pass_num - 1) / num_passes
                        step_frac = step / total_steps / num_passes
                        progress = int(100 * (pass_frac + step_frac))
                        sim_time = step * s.dt

                        counts = sim.get_separation_counts()
                        fines = (
                            counts.get("cyclone_1", 0)
                            + counts.get("cyclone_2", 0)
                            + counts.get("cyclone_3_protein", 0)
                            + counts.get("bagfilter", 0)
                        )
                        coarse = counts.get("coarse", 0) + counts.get("wheel_coarse", 0)
                        total_collected = fines + coarse
                        active = counts.get("active", 0)
                        eff = 100.0 * fines / total_collected if total_collected > 0 else 0.0

                        stats = {
                            "active_particles": active,
                            "collected_fines": fines,
                            "collected_coarse": coarse,
                            "separation_efficiency": eff,
                            "pass_number": pass_num,
                            "total_passes": num_passes,
                            **counts,
                        }
                        self.progress_updated.emit(progress, sim_time, stats)

                        # Emit physics-driven component states for animation
                        comp_state = self._get_component_states(sim, sim_time)
                        if comp_state:
                            self.component_state_updated.emit(comp_state)

                    # Early termination: all fed and none active
                    all_fed = (sim.state.particles_fed >= sim.state.total_particles_to_feed)
                    if step > 0 and all_fed and sim.get_separation_counts().get('active', 0) == 0:
                        self.log_message.emit(
                            f"  [Early exit] All particles settled at "
                            f"t={sim.state.time:.1f}s (step {step:,}/{total_steps:,})"
                        )
                        break

                # Pass complete — force-collect active particles by current zone
                # Active particles are still in transit; assign them to the
                # collection bin matching their current zone so no particles
                # are "lost" in the results.
                n_active_before = sim.get_separation_counts().get('active', 0)
                if n_active_before > 0:
                    zones_np = sim.get_zones()
                    is_active_np = sim.state.is_active.numpy()[:sim.state.particles_active]
                    active_mask = is_active_np == 1
                    active_zones = zones_np[active_mask]
                    # Map zone → collection zone for force-collect
                    # Particles in cyclone body → their dust outlet
                    # Particles in zigzag → fines (they're small refeed particles)
                    # Particles in ducts/venturi → fines (in transit to cyclones)
                    import warp as _wp
                    zone_arr = sim.state.zones.numpy().copy()
                    active_arr = sim.state.is_active.numpy().copy()
                    n_forced = 0
                    for idx in range(sim.state.particles_active):
                        if active_arr[idx] != 1:
                            continue
                        z = zone_arr[idx]
                        # Force-assign based on current zone
                        if z == 50:        # In Cy1 body
                            zone_arr[idx] = 55  # → Cy1 dust
                        elif z == 51:      # In Cy2 body
                            zone_arr[idx] = 56  # → Cy2 dust
                        elif z == 52:      # In Cy3 body
                            zone_arr[idx] = 57  # → Cy3 dust
                        elif z in (40, 41):  # In ducts to cyclones
                            zone_arr[idx] = 55  # → Cy1 (next destination)
                        elif z == 70:      # In bag filter
                            zone_arr[idx] = 75  # → Bag dust
                        elif z in (20, 21, 22):  # In zigzag / fines path
                            zone_arr[idx] = 55  # → Cy1 (fines path)
                        elif z in (0, 1, 2, 10):  # In venturi/duct
                            zone_arr[idx] = 55  # → Cy1 (still in transit)
                        elif z in (34, 35):  # In wheel
                            zone_arr[idx] = 55  # → Cy1 (fines path)
                        elif z == 36:      # In wheel coarse hopper
                            zone_arr[idx] = 37  # → Wheel coarse
                        else:
                            zone_arr[idx] = 55  # Default: Cy1
                        active_arr[idx] = 0
                        n_forced += 1
                    sim.state.zones = _wp.array(zone_arr, dtype=_wp.int32, device=sim.device)
                    sim.state.is_active = _wp.array(active_arr, dtype=_wp.int32, device=sim.device)
                    self.log_message.emit(
                        f"  [Force-collect] {n_forced} active particles assigned to nearest bin"
                    )

                # Now get final pass counts (with force-collected particles)
                pass_counts = sim.get_separation_counts()
                pass_results_list.append({'pass': pass_num, 'counts': pass_counts})

                # Accumulate cyclone diameter data for cumulative stats
                try:
                    zones_np = sim.get_zones()
                    diameters_np = sim.get_diameters()
                    sep_to_frac_cy = {
                        'cyclone_1': 'cy1', 'cyclone_2': 'cy2', 'cyclone_3_protein': 'cy3',
                    }
                    for key, zone_id in [('cyclone_1', 55), ('cyclone_2', 56), ('cyclone_3_protein', 57)]:
                        mask = (zones_np == zone_id)
                        if _np.any(mask):
                            d_um = diameters_np[mask] * 1e6
                            frac_name = sep_to_frac_cy.get(key)
                            if frac_name in recirc_fractions and pass_num < num_passes:
                                pass  # Will be reclassified
                            else:
                                cumul_cy_diameters[key].extend(d_um.tolist())
                except Exception:
                    pass

                # Map sep count keys to fraction names for recirculation check
                sep_to_frac = {
                    'cyclone_1': 'cy1', 'cyclone_2': 'cy2',
                    'cyclone_3_protein': 'cy3', 'wheel_coarse': 'wheel_coarse',
                    'coarse': 'zigzag_coarse', 'bagfilter': 'bagfilter',
                }
                for key in cumulative:
                    if key == 'active':
                        continue
                    frac_name = sep_to_frac.get(key)
                    if frac_name in recirc_fractions and pass_num < num_passes:
                        pass  # Will be recirculated
                    else:
                        cumulative[key] += pass_counts.get(key, 0)
                # Active should be 0 after force-collect
                cumulative['active'] += pass_counts.get('active', 0)

                if num_passes > 1:
                    self.log_message.emit(
                        f"  Pass {pass_num}: Zc={pass_counts.get('coarse',0)} "
                        f"Wc={pass_counts.get('wheel_coarse',0)} "
                        f"Cy1={pass_counts.get('cyclone_1',0)} "
                        f"Cy3={pass_counts.get('cyclone_3_protein',0)} "
                        f"Active={pass_counts.get('active',0)}"
                    )

                # Recirculation: extract and reinitialize for next pass
                if pass_num < num_passes and recirc_fractions and self._is_running:
                    particle_data = sim.extract_collected_particles(recirc_fractions)
                    n_recirc = particle_data['count']
                    if n_recirc == 0:
                        self.log_message.emit("  No particles to recirculate — stopping.")
                        break

                    self.log_message.emit(
                        f"  Recirculating {n_recirc} particles "
                        f"(mean {particle_data['diameters'].mean()*1e6:.1f} µm)"
                    )

                    next_wheel = s.recirculate_wheel_rpm if s.recirculate_wheel_rpm > 0 else None
                    # Particles return to feed hopper and trickle through
                    # the feed system (~21s residence) before reaching
                    # the venturi solids inlet — matching real machine.
                    sim.reinitialize_from_particles(
                        particle_data,
                        initial_velocity=None,  # auto from feed kinetics
                        continuous_feeding=None,  # auto (continuous)
                        wheel_rpm=next_wheel,
                        attrition_factor=s.attrition_factor,
                        attrition_min_diameter_m=s.attrition_min_um * 1e-6,
                        feed_residence_time_s=21.0,  # gravity chute transit
                    )

            # ==============================================================
            # 7. Final results (cumulative across all passes)
            # ==============================================================
            elapsed = _time.perf_counter() - t_start

            # Use cumulative counts when multi-pass, otherwise last pass counts
            if num_passes > 1:
                final_counts = cumulative
            else:
                final_counts = sim.get_separation_counts()

            fines = (
                final_counts.get("cyclone_1", 0)
                + final_counts.get("cyclone_2", 0)
                + final_counts.get("cyclone_3_protein", 0)
                + final_counts.get("bagfilter", 0)
            )
            coarse = final_counts.get("coarse", 0) + final_counts.get("wheel_coarse", 0)
            total_collected = fines + coarse
            eff = 100.0 * fines / total_collected if total_collected > 0 else 0.0

            # Cyclone particle size stats — cumulative across all passes
            cyclone_stats = {}
            if num_passes > 1:
                # Build cumulative stats from collected diameter data
                for key in ('cyclone_1', 'cyclone_2', 'cyclone_3_protein'):
                    d_list = cumul_cy_diameters.get(key, [])
                    entry = {'count': len(d_list), 'mean_d_um': None, 'median_d_um': None, 'design_d50_um': None}
                    if d_list:
                        d_arr = _np.array(d_list)
                        entry['mean_d_um'] = float(d_arr.mean())
                        entry['median_d_um'] = float(_np.median(d_arr))
                    # Get design d50 from last pass stats
                    try:
                        last_stats = sim.get_cyclone_particle_size_stats()
                        entry['design_d50_um'] = last_stats.get(key, {}).get('design_d50_um')
                    except Exception:
                        pass
                    cyclone_stats[key] = entry
            else:
                try:
                    cyclone_stats = sim.get_cyclone_particle_size_stats()
                except Exception:
                    pass

            # Geometry info for results panel
            geo = getattr(sim, 'geometry', {})

            results = {
                "total_time": s.total_time,
                "wall_time_s": elapsed,
                "particles_processed": total_collected,
                "separation_efficiency": eff,
                "fines_collected": fines,
                "coarse_collected": coarse,
                "air_flow_m3s": air_flow,
                "air_flow_m3h": air_flow * 3600,
                "blower_rpm": s.blower_rpm,
                "wheel_rpm": s.wheel_rpm,
                "use_preclassification": s.use_preclassification,
                "num_particles": s.num_particles,
                "cyclone_stats": cyclone_stats,
                "geometry": geo,
                "num_passes": num_passes,
                "pass_results": pass_results_list,
                **final_counts,
            }
            self.simulation_completed.emit(results)

        except Exception as e:
            self.simulation_error.emit(f"{e}\n{traceback.format_exc()}")

    # ----------------------------------------------------------------

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False

    # ----------------------------------------------------------------
    # Subsidiary physics simulators
    # ----------------------------------------------------------------

    def _create_subsidiary_simulators(self, classification_assembly, air_flow_m3s: float, s):
        """
        Create lightweight AirFlowPhysicsSimulator and FeedFlowPhysicsSimulator
        to step alongside the classification simulation.

        These provide real physics state for animation:
        - Air: blower VFD ramp (S-curve), damper positions, system phase
        - Feed: lid servo angle, feed phase, component angular velocities

        Both simulators run in lightweight mode:
        - Air: enable_sph=False (no GPU air particles, just state machine)
        - Feed: num_particles=0, enable_pouring=False (lid animation only)
        """
        # Build a CompleteClassifierAssembly to get air/feed subsystem assemblies
        try:
            from ...geometry.assembly.complete_system import (
                CompleteClassifierAssembly,
                CompleteSystemParams,
            )
            from ...geometry.assembly.classification import ClassificationSystemParams

            cls_params = ClassificationSystemParams(
                use_preclassification=s.use_preclassification,
                wheel_diameter=getattr(s, 'wheel_diameter', 0.3),
                wheel_rpm=s.wheel_rpm,
            )
            # Apply geometry overrides that were used for the classification assembly
            if s.venturi_throat_diameter_mm > 0:
                throat_m = s.venturi_throat_diameter_mm / 1000.0
                cls_params.venturi_throat_ratio = throat_m / cls_params.venturi_inlet_diameter
            if s.zigzag_width_mm > 0:
                cls_params.zigzag_channel_width = s.zigzag_width_mm / 1000.0
            if s.zigzag_depth_mm > 0:
                cls_params.zigzag_channel_depth = s.zigzag_depth_mm / 1000.0

            complete_params = CompleteSystemParams(
                classification_params=cls_params,
                air_flow_m3_h=air_flow_m3s * 3600.0,
                include_feed_system=s.include_feed_system,
                include_air_system=s.include_air_system,
                include_exhaust=False,
                include_ductwork=True,
            )
            complete_assembly = CompleteClassifierAssembly(complete_params)
        except Exception as e:
            self.log_message.emit(f"  Subsidiary sims: could not build complete assembly: {e}")
            return

        # --- Air Flow Physics Simulator (lightweight: no SPH particles) ---
        if s.include_air_system:
            try:
                from ...simulation.air_flow_physics import (
                    AirFlowPhysicsSimulator,
                    AirFlowPhysicsConfig,
                )
                air_assembly = complete_assembly.get_subsystem("air_system")
                if air_assembly is not None:
                    blower_rpm = self._blower_operating_point.get("rpm", s.blower_rpm) if self._blower_operating_point else s.blower_rpm
                    air_config = AirFlowPhysicsConfig(
                        target_rpm=blower_rpm,
                        dt=0.01,              # coarse dt (state machine only)
                        total_time=s.total_time,
                        ramp_time=2.0,        # VFD ramp time
                        damper_ramp_time=2.0,  # damper open time
                        enable_sph=False,     # NO GPU air particles
                        device="cpu",         # CPU-only (no Warp kernels)
                    )
                    self._air_sim = AirFlowPhysicsSimulator(air_assembly, air_config)
                    self._air_sim.start_system()  # begin startup immediately
                    self.log_message.emit(
                        f"  Air physics: blower {blower_rpm:.0f} RPM, "
                        f"ramp=2.0s, dampers=2.0s [lightweight, no SPH]"
                    )
            except Exception as e:
                self.log_message.emit(f"  Air physics init failed (animation will use fallback): {e}")
                self._air_sim = None

        # --- Feed Flow Physics Simulator (lightweight: lid animation only) ---
        if s.include_feed_system:
            try:
                from ...simulation.feed_flow_physics import (
                    FeedFlowPhysicsSimulator,
                    FlowPhysicsConfig,
                )
                feed_assembly = complete_assembly.get_subsystem("feed_system")
                if feed_assembly is not None:
                    feed_config = FlowPhysicsConfig(
                        dt=0.01,              # coarse dt (lid servo only)
                        total_time=s.total_time,
                        animate_lid=True,
                        lid_open_angle=90.0,
                        lid_animation_time=2.0,   # 2s to fully open (45 deg/s)
                        enable_pouring=False,     # NO particle pouring
                        num_particles=0,          # NO feed particles
                        device="cpu",             # CPU-only
                    )
                    self._feed_sim = FeedFlowPhysicsSimulator(feed_assembly, feed_config)
                    self.log_message.emit(
                        "  Feed physics: lid servo 90\u00b0/2s, "
                        f"airlock={feed_config.airlock_rpm:.0f} RPM, "
                        f"screw={feed_config.feeder_rpm:.0f} RPM, "
                        f"deagg={feed_config.deagg_rpm:.0f} RPM [lightweight, no particles]"
                    )
            except Exception as e:
                self.log_message.emit(f"  Feed physics init failed (animation will use fallback): {e}")
                self._feed_sim = None

        # Fast-forward subsidiary sims through the startup preamble so they
        # start at steady-state.  The preamble animation was already shown to
        # the user by the AnimationController before the worker was launched.
        self._fast_forward_subsidiary_sims()

    def _fast_forward_subsidiary_sims(self):
        """Fast-forward air/feed sims through the startup preamble to steady state."""
        PREAMBLE = 8.0  # matches AnimationTimeline.steady_time

        # Air: ramp blower to full speed, open dampers fully
        if self._air_sim is not None:
            try:
                dt = self._air_sim.config.dt
                while self._air_sim.state.time < PREAMBLE:
                    self._air_sim.step()
            except Exception:
                pass

        # Feed: open lid and let it reach fully open.
        # Lid STAYS OPEN for the entire classification phase and only
        # closes during the shutdown sequence after the simulation ends.
        if self._feed_sim is not None:
            try:
                feed_start = 3.0
                dt = self._feed_sim.config.dt

                # Open lid
                self._feed_sim.open_lid()
                while self._feed_sim.state.time < (PREAMBLE - feed_start):
                    self._feed_sim.step()
                # Lid is now fully open at 90° -- stays open
            except Exception:
                pass

    def _step_subsidiary_simulators(self, sim_time: float):
        """
        Advance air and feed physics simulators during the classification phase.

        The sims were already fast-forwarded through the startup preamble
        in _fast_forward_subsidiary_sims().  During classification (sim_time
        0→total_time), they are at steady state: blower at full RPM, dampers
        open, lid closed.  We keep stepping them to maintain their time.
        """
        PREAMBLE = 8.0  # already fast-forwarded this far
        target = PREAMBLE + sim_time

        if self._air_sim is not None:
            try:
                air_dt = self._air_sim.config.dt
                while self._air_sim.state.time < target - air_dt * 0.5:
                    self._air_sim.step()
            except Exception:
                pass

        if self._feed_sim is not None:
            try:
                # Feed sim time is offset by feed_start (3s)
                feed_target = target - 3.0
                feed_dt = self._feed_sim.config.dt
                while self._feed_sim.state.time < feed_target - feed_dt * 0.5:
                    self._feed_sim.step()
            except Exception:
                pass

    def _get_component_states(self, sim, sim_time: float) -> dict:
        """
        Extract component states during the classification phase.

        The startup preamble (air ramp, dampers open, lid open/close) was
        already played by the AnimationController BEFORE the simulation
        started.  During classification, every component is at steady state:
        - Wheel: spinning at physics omega (from ClassificationFlowPhysicsSimulator)
        - Blower: full RPM (from airclass_flow_physics operating point)
        - Dampers: fully open (position 1.0)
        - Lid: fully open (90°)
        - Feed components: full speed
        - All ramp fractions: 1.0

        The shutdown animation (closing dampers, lid, ramp-down) is handled
        by AnimationController.begin_shutdown() AFTER the simulation ends.
        """
        import math
        TWO_PI = 2.0 * math.pi
        s = self.settings
        component_state = {"sim_time": float(sim_time)}

        # ==================================================================
        # WHEEL -- from ClassificationFlowPhysicsSimulator (live physics)
        # angle = wheel_omega * time  (same as Warp kernel, line 1962)
        # ==================================================================
        wheel_omega = getattr(sim, 'wheel_omega', 0.0)
        component_state["wheel_omega"] = float(wheel_omega)
        component_state["wheel_angle_rad"] = float(wheel_omega * sim_time) % TWO_PI

        cls_state = getattr(sim, 'state', None)
        if cls_state is not None:
            phase = getattr(cls_state, 'phase', None)
            component_state["phase"] = phase.value if phase else "running"

        # ==================================================================
        # STEADY STATE -- everything fully running during classification.
        # Preamble already completed; shutdown handled after sim ends.
        # ==================================================================
        op = self._blower_operating_point
        blower_design_rpm = op.get("rpm", s.blower_rpm) if op else s.blower_rpm

        component_state["blower_rpm"] = float(blower_design_rpm)
        component_state["blower_ramp_frac"] = 1.0
        component_state["air_flow_m3s"] = float(self._air_flow_m3s)
        component_state["damper_positions"] = [1.0, 1.0]   # fully open
        component_state["lid_angle_deg"] = 90.0              # fully open
        component_state["feed_ramp_frac"] = 1.0
        component_state["classification_ramp_frac"] = 1.0

        return component_state


# ============================================================================
# Reusable KPI card
# ============================================================================

class _StatCard(QFrame):
    """Compact metric card showing a value with a label."""

    def __init__(self, label: str, initial_value: str = "--",
                 accent: str = COLORS.TEXT_PRIMARY, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS.BG_DARK};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self._value_label = QLabel(initial_value)
        self._value_label.setStyleSheet(
            f"font-size: 14pt; font-weight: 700; color: {accent};"
            " border: none; background: transparent;"
        )
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._value_label)

        title = QLabel(label)
        title.setStyleSheet(
            f"font-size: 8pt; color: {COLORS.TEXT_MUTED};"
            " border: none; background: transparent;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

    def set_value(self, text: str):
        self._value_label.setText(text)


# ============================================================================
# Helper
# ============================================================================

def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap *widget* in a frameless QScrollArea."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


# ============================================================================
# Main panel
# ============================================================================

class SimulationControlPanel(QWidget):
    """
    Panel for controlling simulation execution.

    Uses ClassificationFlowPhysicsSimulator for real Warp-based physics.
    Settings mirror ``run_classification_flow.py`` CLI parameters.
    """

    run_requested = Signal()
    pause_requested = Signal()
    stop_requested = Signal()
    settings_changed = Signal(object)
    simulation_results_ready = Signal(dict)  # emitted with full results dict
    sim_time_updated = Signal(float)         # emitted with simulation time each progress tick
    component_state_updated = Signal(dict)   # physics-driven component states for animation

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._settings = SimulationSettings()
        self._worker: Optional[SimulationWorker] = None
        self._thread: Optional[QThread] = None

        self._setup_ui()
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._create_control_tab(), "Control")
        tabs.addTab(_scrollable(self._create_settings_tab()), "Settings")
        tabs.addTab(self._create_log_tab(), "Log")

    # ================================================================
    # Control tab
    # ================================================================

    def _create_control_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("cssClass", "success")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._on_run_clicked)
        btn_row.addWidget(self.run_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setMinimumHeight(34)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        btn_row.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("cssClass", "danger")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        # KPI cards
        grid = QGridLayout()
        grid.setSpacing(6)

        self.card_time = _StatCard("Simulation Time", "0.000 s", COLORS.ACCENT)
        grid.addWidget(self.card_time, 0, 0)

        self.card_particles = _StatCard("Active Particles", "0", COLORS.INFO)
        grid.addWidget(self.card_particles, 0, 1)

        self.card_fines = _StatCard("Fines Collected", "0", COLORS.SUCCESS)
        grid.addWidget(self.card_fines, 1, 0)

        self.card_coarse = _StatCard("Coarse Collected", "0", COLORS.WARNING)
        grid.addWidget(self.card_coarse, 1, 1)

        self.card_efficiency = _StatCard("Separation Efficiency", "--", COLORS.CAT_CLASSIFICATION)
        grid.addWidget(self.card_efficiency, 2, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()
        return widget

    # ================================================================
    # Settings tab (scrollable)
    # ================================================================

    def _create_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        _M = (10, 14, 10, 10)

        # ---- Time ----
        g = QGroupBox("Time Settings")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.total_time_spin = QDoubleSpinBox()
        self.total_time_spin.setRange(0.1, 3600.0)
        self.total_time_spin.setDecimals(1)
        self.total_time_spin.setSingleStep(10)
        self.total_time_spin.setValue(self._settings.total_time)
        self.total_time_spin.setSuffix(" s")
        self.total_time_spin.valueChanged.connect(self._on_time_changed)
        f.addRow("Total Time:", self.total_time_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.0001, 0.01)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(self._settings.dt)
        self.dt_spin.setSuffix(" s")
        self.dt_spin.valueChanged.connect(self._on_time_changed)
        f.addRow("Time Step (dt):", self.dt_spin)

        self.output_spin = QDoubleSpinBox()
        self.output_spin.setRange(0.01, 60.0)
        self.output_spin.setDecimals(2)
        self.output_spin.setValue(self._settings.output_interval)
        self.output_spin.setSuffix(" s")
        self.output_spin.valueChanged.connect(lambda v: setattr(self._settings, 'output_interval', v))
        f.addRow("Output Interval:", self.output_spin)

        layout.addWidget(g)

        # ---- Particles ----
        g = QGroupBox("Particles")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(100, 500000)
        self.num_particles_spin.setSingleStep(1000)
        self.num_particles_spin.setValue(self._settings.num_particles)
        self.num_particles_spin.valueChanged.connect(lambda v: setattr(self._settings, 'num_particles', v))
        f.addRow("Count:", self.num_particles_spin)

        self.continuous_check = QCheckBox("Continuous")
        self.continuous_check.setChecked(self._settings.continuous_feeding)
        self.continuous_check.setToolTip("Activate particles gradually at feed rate instead of all at t=0")
        self.continuous_check.stateChanged.connect(
            lambda s: setattr(self._settings, 'continuous_feeding', s == Qt.CheckState.Checked.value)
        )
        f.addRow("Feeding:", self.continuous_check)

        self.feed_rate_spin = QDoubleSpinBox()
        self.feed_rate_spin.setRange(0, 500000)
        self.feed_rate_spin.setDecimals(0)
        self.feed_rate_spin.setSingleStep(100)
        self.feed_rate_spin.setValue(self._settings.particle_feed_rate)
        self.feed_rate_spin.setSuffix("  /s")
        self.feed_rate_spin.setToolTip("0 = auto-compute from mass flow (recommended)")
        self.feed_rate_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_feed_rate', v))
        f.addRow("Feed Rate:", self.feed_rate_spin)

        self.particle_dia_spin = QDoubleSpinBox()
        self.particle_dia_spin.setRange(1.0, 500.0)
        self.particle_dia_spin.setDecimals(1)
        self.particle_dia_spin.setSingleStep(5)
        self.particle_dia_spin.setValue(self._settings.particle_diameter_um)
        self.particle_dia_spin.setSuffix(" \u00b5m")
        self.particle_dia_spin.setToolTip("Mean diameter (used only when Material = none)")
        self.particle_dia_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_diameter_um', v))
        f.addRow("Mean Diameter:", self.particle_dia_spin)

        self.particle_std_spin = QDoubleSpinBox()
        self.particle_std_spin.setRange(0.0, 200.0)
        self.particle_std_spin.setDecimals(1)
        self.particle_std_spin.setSingleStep(5)
        self.particle_std_spin.setValue(self._settings.particle_std_um)
        self.particle_std_spin.setSuffix(" \u00b5m")
        self.particle_std_spin.setToolTip("Std deviation (used only when Material = none)")
        self.particle_std_spin.valueChanged.connect(lambda v: setattr(self._settings, 'particle_std_um', v))
        f.addRow("Diameter Std Dev:", self.particle_std_spin)

        layout.addWidget(g)

        # ---- Material ----
        g = QGroupBox("Material")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.material_combo = QComboBox()
        self.material_combo.addItems(["yellow_pea", "faba_bean", "oat", "none"])
        self.material_combo.setToolTip("Food powder preset (provides realistic size distribution)")
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        f.addRow("Source:", self.material_combo)

        self.fraction_combo = QComboBox()
        self.fraction_combo.addItems(["whole", "protein", "starch", "fiber"])
        self.fraction_combo.setToolTip("Whole flour or single fraction")
        self.fraction_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'material_fraction', v))
        f.addRow("Fraction:", self.fraction_combo)

        layout.addWidget(g)

        # ---- Air / Assembly ----
        g = QGroupBox("Air & Assembly")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.assembly_mode_combo = QComboBox()
        self.assembly_mode_combo.addItems([
            "Full System (Venturi + Zigzag + Wheel)",
            "Wheel-Only (Direct Feed)"
        ])
        self.assembly_mode_combo.currentIndexChanged.connect(self._on_assembly_mode_changed)
        f.addRow("Mode:", self.assembly_mode_combo)

        self.blower_rpm_spin = QDoubleSpinBox()
        self.blower_rpm_spin.setRange(0, 5000)
        self.blower_rpm_spin.setSingleStep(100)
        self.blower_rpm_spin.setDecimals(0)
        self.blower_rpm_spin.setValue(self._settings.blower_rpm)
        self.blower_rpm_spin.setSuffix("  RPM")
        self.blower_rpm_spin.setToolTip(
            "VFD blower speed. 0 = use Air Flow Rate directly.\n"
            "Design: 3000 RPM = 3000 m\u00b3/h.\n"
            "Recommended: 400-600 RPM for bench-scale."
        )
        self.blower_rpm_spin.valueChanged.connect(self._on_blower_rpm_changed)
        f.addRow("Blower RPM:", self.blower_rpm_spin)

        self.air_flow_spin = QDoubleSpinBox()
        self.air_flow_spin.setRange(0.001, 5.0)
        self.air_flow_spin.setDecimals(3)
        self.air_flow_spin.setSingleStep(0.01)
        self.air_flow_spin.setValue(self._settings.air_flow_m3s)
        self.air_flow_spin.setSuffix("  m\u00b3/s")
        self.air_flow_spin.setToolTip("Direct air flow rate. Overridden when Blower RPM > 0.")
        self.air_flow_spin.valueChanged.connect(lambda v: setattr(self._settings, 'air_flow_m3s', v))
        f.addRow("Air Flow Rate:", self.air_flow_spin)

        self.bypass_spin = QDoubleSpinBox()
        self.bypass_spin.setRange(0.0, 0.99)
        self.bypass_spin.setDecimals(3)
        self.bypass_spin.setSingleStep(0.01)
        self.bypass_spin.setValue(self._settings.bypass_ratio)
        self.bypass_spin.setToolTip("Fraction of air bypassing venturi+zigzag (0 = no bypass)")
        self.bypass_spin.valueChanged.connect(lambda v: setattr(self._settings, 'bypass_ratio', v))
        f.addRow("Bypass Ratio:", self.bypass_spin)

        self.wheel_rpm_spin = QDoubleSpinBox()
        self.wheel_rpm_spin.setRange(500, 20000)
        self.wheel_rpm_spin.setSingleStep(500)
        self.wheel_rpm_spin.setDecimals(0)
        self.wheel_rpm_spin.setValue(self._settings.wheel_rpm)
        self.wheel_rpm_spin.setSuffix("  RPM")
        self.wheel_rpm_spin.valueChanged.connect(self._on_wheel_rpm_changed)
        f.addRow("Wheel Speed:", self.wheel_rpm_spin)

        self.wheel_diameter_spin = QDoubleSpinBox()
        self.wheel_diameter_spin.setRange(0.05, 0.50)
        self.wheel_diameter_spin.setDecimals(3)
        self.wheel_diameter_spin.setSingleStep(0.01)
        self.wheel_diameter_spin.setValue(self._settings.wheel_diameter)
        self.wheel_diameter_spin.setSuffix("  m")
        self.wheel_diameter_spin.valueChanged.connect(lambda v: setattr(self._settings, 'wheel_diameter', v))
        f.addRow("Wheel Diameter:", self.wheel_diameter_spin)

        layout.addWidget(g)

        # ---- Geometry Overrides ----
        g = QGroupBox("Geometry Overrides")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        hint = QLabel("0 = use geometry defaults")
        hint.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8pt; border: none; background: transparent;")
        f.addRow(hint)

        self.throat_dia_spin = QDoubleSpinBox()
        self.throat_dia_spin.setRange(0, 200)
        self.throat_dia_spin.setDecimals(1)
        self.throat_dia_spin.setSingleStep(5)
        self.throat_dia_spin.setValue(self._settings.venturi_throat_diameter_mm)
        self.throat_dia_spin.setSuffix("  mm")
        self.throat_dia_spin.setToolTip("Venturi throat diameter. Default 40mm (80mm inlet \u00d7 0.5 ratio)")
        self.throat_dia_spin.valueChanged.connect(lambda v: setattr(self._settings, 'venturi_throat_diameter_mm', v))
        f.addRow("Venturi Throat \u00d8:", self.throat_dia_spin)

        self.zz_width_spin = QDoubleSpinBox()
        self.zz_width_spin.setRange(0, 500)
        self.zz_width_spin.setDecimals(0)
        self.zz_width_spin.setSingleStep(10)
        self.zz_width_spin.setValue(self._settings.zigzag_width_mm)
        self.zz_width_spin.setSuffix("  mm")
        self.zz_width_spin.setToolTip("Zigzag channel width. Default 120mm from geometry.")
        self.zz_width_spin.valueChanged.connect(lambda v: setattr(self._settings, 'zigzag_width_mm', v))
        f.addRow("Zigzag Width:", self.zz_width_spin)

        self.zz_depth_spin = QDoubleSpinBox()
        self.zz_depth_spin.setRange(0, 500)
        self.zz_depth_spin.setDecimals(0)
        self.zz_depth_spin.setSingleStep(10)
        self.zz_depth_spin.setValue(self._settings.zigzag_depth_mm)
        self.zz_depth_spin.setSuffix("  mm")
        self.zz_depth_spin.setToolTip("Zigzag channel depth. Default 200mm from geometry.")
        self.zz_depth_spin.valueChanged.connect(lambda v: setattr(self._settings, 'zigzag_depth_mm', v))
        f.addRow("Zigzag Depth:", self.zz_depth_spin)

        layout.addWidget(g)

        # ---- Physics ----
        g = QGroupBox("Physics")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.turbulence_spin = QDoubleSpinBox()
        self.turbulence_spin.setRange(0.0, 0.5)
        self.turbulence_spin.setDecimals(2)
        self.turbulence_spin.setSingleStep(0.01)
        self.turbulence_spin.setValue(self._settings.turbulence_base)
        self.turbulence_spin.setToolTip("Base turbulence (0.15 default). Scales zone-specific: zigzag=0.25, cyclone=0.12")
        self.turbulence_spin.valueChanged.connect(lambda v: setattr(self._settings, 'turbulence_base', v))
        f.addRow("Turbulence (base):", self.turbulence_spin)

        self.restitution_spin = QDoubleSpinBox()
        self.restitution_spin.setRange(0.0, 1.0)
        self.restitution_spin.setDecimals(2)
        self.restitution_spin.setSingleStep(0.05)
        self.restitution_spin.setValue(self._settings.restitution)
        self.restitution_spin.setToolTip("Particle-wall restitution (0=perfectly inelastic, 1=elastic)")
        self.restitution_spin.valueChanged.connect(lambda v: setattr(self._settings, 'restitution', v))
        f.addRow("Restitution:", self.restitution_spin)

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setSingleStep(0.05)
        self.friction_spin.setValue(self._settings.friction)
        self.friction_spin.setToolTip("Particle-wall friction coefficient")
        self.friction_spin.valueChanged.connect(lambda v: setattr(self._settings, 'friction', v))
        f.addRow("Friction:", self.friction_spin)

        self.max_loading_spin = QDoubleSpinBox()
        self.max_loading_spin.setRange(0.1, 10.0)
        self.max_loading_spin.setDecimals(1)
        self.max_loading_spin.setSingleStep(0.5)
        self.max_loading_spin.setValue(self._settings.max_loading_ratio)
        self.max_loading_spin.setToolTip("Max solids/air mass ratio for venturi entrainment cap")
        self.max_loading_spin.valueChanged.connect(lambda v: setattr(self._settings, 'max_loading_ratio', v))
        f.addRow("Max Loading Ratio:", self.max_loading_spin)

        layout.addWidget(g)

        # ---- Recirculation ----
        g = QGroupBox("Multi-Pass Recirculation")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 10)
        self.passes_spin.setValue(self._settings.recirculate_passes)
        self.passes_spin.setToolTip("Number of classification passes (1 = single pass, no recirculation)")
        self.passes_spin.valueChanged.connect(self._on_passes_changed)
        f.addRow("Passes:", self.passes_spin)

        self.recirc_fractions_combo = QComboBox()
        self.recirc_fractions_combo.addItems([
            "cy1", "cy1,cy2", "cy2", "wheel_coarse", "cy1,wheel_coarse",
        ])
        self.recirc_fractions_combo.setEditable(True)
        self.recirc_fractions_combo.setCurrentText(self._settings.recirculate_fractions)
        self.recirc_fractions_combo.setToolTip(
            "Fractions to refeed: cy1, cy2, cy3, wheel_coarse, zigzag_coarse, bagfilter.\n"
            "Comma-separated for multiple."
        )
        self.recirc_fractions_combo.currentTextChanged.connect(
            lambda v: setattr(self._settings, 'recirculate_fractions', v)
        )
        f.addRow("Refeed Fractions:", self.recirc_fractions_combo)

        self.attrition_spin = QDoubleSpinBox()
        self.attrition_spin.setRange(0.0, 0.50)
        self.attrition_spin.setDecimals(2)
        self.attrition_spin.setSingleStep(0.05)
        self.attrition_spin.setValue(self._settings.attrition_factor)
        self.attrition_spin.setToolTip(
            "Venturi attrition: fraction of breakable diameter removed per pass.\n"
            "Models shear breakup of protein-starch composites at throat.\n"
            "0.10 = 10% per pass. 0 = disabled."
        )
        self.attrition_spin.valueChanged.connect(lambda v: setattr(self._settings, 'attrition_factor', v))
        f.addRow("Attrition Rate:", self.attrition_spin)

        self.attrition_min_spin = QDoubleSpinBox()
        self.attrition_min_spin.setRange(1.0, 20.0)
        self.attrition_min_spin.setDecimals(1)
        self.attrition_min_spin.setSingleStep(1.0)
        self.attrition_min_spin.setValue(self._settings.attrition_min_um)
        self.attrition_min_spin.setSuffix(" \u00b5m")
        self.attrition_min_spin.setToolTip("Minimum diameter below which attrition stops (protein body floor)")
        self.attrition_min_spin.valueChanged.connect(lambda v: setattr(self._settings, 'attrition_min_um', v))
        f.addRow("Attrition Min \u00d8:", self.attrition_min_spin)

        self.recirc_wheel_spin = QDoubleSpinBox()
        self.recirc_wheel_spin.setRange(0, 20000)
        self.recirc_wheel_spin.setSingleStep(500)
        self.recirc_wheel_spin.setDecimals(0)
        self.recirc_wheel_spin.setValue(self._settings.recirculate_wheel_rpm)
        self.recirc_wheel_spin.setSuffix("  RPM")
        self.recirc_wheel_spin.setToolTip("Wheel RPM for passes 2+. 0 = same as main wheel RPM.")
        self.recirc_wheel_spin.valueChanged.connect(lambda v: setattr(self._settings, 'recirculate_wheel_rpm', v))
        f.addRow("Pass 2+ Wheel:", self.recirc_wheel_spin)

        self.recirc_time_spin = QDoubleSpinBox()
        self.recirc_time_spin.setRange(0, 3600)
        self.recirc_time_spin.setDecimals(0)
        self.recirc_time_spin.setSingleStep(30)
        self.recirc_time_spin.setValue(self._settings.recirculate_time)
        self.recirc_time_spin.setSuffix("  s")
        self.recirc_time_spin.setToolTip("Simulation time for passes 2+. 0 = same as Total Time.")
        self.recirc_time_spin.valueChanged.connect(lambda v: setattr(self._settings, 'recirculate_time', v))
        f.addRow("Pass 2+ Time:", self.recirc_time_spin)

        # Initially disable recirculation controls when passes=1
        self._set_recirc_controls_enabled(self._settings.recirculate_passes > 1)

        layout.addWidget(g)

        # ---- Compute ----
        g = QGroupBox("Compute")
        f = QFormLayout(g); f.setContentsMargins(*_M)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.device_combo.currentTextChanged.connect(lambda v: setattr(self._settings, 'device', v))
        f.addRow("Device:", self.device_combo)

        layout.addWidget(g)

        return widget

    # ---- settings callbacks ----

    def _on_blower_rpm_changed(self, rpm: float):
        self._settings.blower_rpm = rpm
        if rpm > 0:
            try:
                from ...simulation.airclass_flow_physics import compute_blower_operating_point
                op = compute_blower_operating_point(rpm)
                q = op["Q_m3_s"]
                self._settings.air_flow_m3s = q
                self.air_flow_spin.setValue(q)
                self.air_flow_spin.setEnabled(False)
                # Dynamic tooltips
                self.blower_rpm_spin.setToolTip(
                    f"VFD blower at {rpm:.0f} RPM\n"
                    f"Flow: {op['Q_m3_h']:.0f} m\u00b3/h ({q:.3f} m\u00b3/s)\n"
                    f"Pressure: {op['P_operating_Pa']:.0f} Pa\n"
                    f"Power: {op['shaft_power_W']:.0f} W  Efficiency: {op['efficiency']:.1%}"
                )
                self.air_flow_spin.setToolTip(
                    f"Computed from blower at {rpm:.0f} RPM: {op['Q_m3_h']:.0f} m\u00b3/h"
                )
                self._log(
                    f"Blower {rpm:.0f} RPM \u2192 {op['Q_m3_h']:.0f} m\u00b3/h "
                    f"({q:.3f} m\u00b3/s), P={op['P_operating_Pa']:.0f} Pa"
                )
            except Exception as e:
                self._log(f"Blower RPM error: {e}")
        else:
            self.air_flow_spin.setEnabled(True)
            self.blower_rpm_spin.setToolTip(
                "VFD blower speed. 0 = use Air Flow Rate directly.\n"
                "Design: 3000 RPM = 3000 m\u00b3/h.\n"
                "Recommended: 400-600 RPM for bench-scale."
            )
            self.air_flow_spin.setToolTip(
                "Direct air flow rate (m\u00b3/s). Set Blower RPM > 0 to compute from fan curve."
            )

    def _on_time_changed(self, _=None):
        """Update time-related tooltips when total_time or dt changes."""
        t = self.total_time_spin.value()
        dt = self.dt_spin.value()
        self._settings.total_time = t
        self._settings.dt = dt
        if dt > 0:
            steps = int(t / dt)
            self.total_time_spin.setToolTip(f"{t:.1f}s = {steps:,} steps at dt={dt}s")
            self.dt_spin.setToolTip(f"{dt}s per step, {steps:,} total steps for {t:.1f}s")
        else:
            self.total_time_spin.setToolTip(f"{t:.1f}s")
            self.dt_spin.setToolTip("")

    def _on_wheel_rpm_changed(self, rpm: float):
        self._settings.wheel_rpm = rpm
        import math
        omega = 2 * math.pi * rpm / 60.0
        dia_m = self._settings.wheel_diameter
        r = dia_m / 2.0
        tip_speed = omega * r
        g_force = (omega ** 2 * r) / 9.81
        self.wheel_rpm_spin.setToolTip(
            f"Wheel classifier at {rpm:.0f} RPM\n"
            f"\u00d8 {dia_m * 1000:.0f} mm   \u03c9 = {omega:.1f} rad/s\n"
            f"Tip speed: {tip_speed:.1f} m/s\n"
            f"G-force at rim: {g_force:.0f} g"
        )

    def _on_material_changed(self, source: str):
        self._settings.material_source = source
        use_material = source not in ("none", "")
        # Enable/disable diameter fields based on material selection
        self.particle_dia_spin.setEnabled(not use_material)
        self.particle_std_spin.setEnabled(not use_material)
        self.fraction_combo.setEnabled(use_material)

    def _on_assembly_mode_changed(self, index: int):
        self._settings.use_preclassification = (index == 0)
        # Bypass only relevant with preclassification
        self.bypass_spin.setEnabled(index == 0)
        mode_name = "Full System" if index == 0 else "Wheel-Only"
        self._log(f"Assembly mode: {mode_name}")

    def _on_passes_changed(self, value: int):
        self._settings.recirculate_passes = value
        self._set_recirc_controls_enabled(value > 1)
        if value > 1:
            self._log(f"Recirculation: {value} passes")

    def _set_recirc_controls_enabled(self, enabled: bool):
        """Enable/disable recirculation sub-controls based on passes count."""
        self.recirc_fractions_combo.setEnabled(enabled)
        self.attrition_spin.setEnabled(enabled)
        self.attrition_min_spin.setEnabled(enabled)
        self.recirc_wheel_spin.setEnabled(enabled)
        self.recirc_time_spin.setEnabled(enabled)

    # ================================================================
    # Log tab
    # ================================================================

    def _create_log_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 9pt;
                background: {COLORS.BG_DARKEST};
                color: {COLORS.TEXT_SECONDARY};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
            }}
        """)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setProperty("cssClass", "ghost")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        return widget

    # ================================================================
    # Button handlers
    # ================================================================

    def _on_run_clicked(self):
        if self._worker and self._worker._is_paused:
            self._worker.resume()
            self.run_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
        else:
            self.run_requested.emit()

    def _on_pause_clicked(self):
        if self._worker:
            self._worker.pause()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_requested.emit()

    def _on_stop_clicked(self):
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.stop_requested.emit()

    # ================================================================
    # Simulation lifecycle
    # ================================================================

    def start_simulation(self, assembly_data: Dict[str, Any]):
        """Start a new simulation using the real physics engine."""
        s = self._settings
        self._log("=" * 56)
        self._log("CLASSIFICATION FLOW PHYSICS SIMULATION")
        self._log("=" * 56)
        self._log(f"  Device:     {s.device}")
        self._log(f"  Particles:  {s.num_particles:,}")
        self._log(f"  Time:       {s.total_time:.0f}s  (dt={s.dt}s, "
                   f"{int(s.total_time / s.dt):,} steps)")
        self._log(f"  Air flow:   {s.air_flow_m3s:.3f} m\u00b3/s "
                   f"({s.air_flow_m3s * 3600:.0f} m\u00b3/h)")
        self._log(f"  Material:   {s.material_source} / {s.material_fraction}")
        mode = "Full System" if s.use_preclassification else "Wheel-Only"
        self._log(f"  Mode:       {mode}")
        if s.bypass_ratio > 0:
            self._log(f"  Bypass:     {s.bypass_ratio:.1%}")
        self._log(f"  Wheel:      {s.wheel_rpm:.0f} RPM, \u00d8{s.wheel_diameter*1000:.0f} mm")
        if s.recirculate_passes > 1:
            self._log(f"  Recirc:     {s.recirculate_passes} passes, "
                       f"fractions=[{s.recirculate_fractions}], "
                       f"attrition={s.attrition_factor*100:.0f}%")
        self._log("=" * 56)

        # UI state
        self.run_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Worker + thread
        self._thread = QThread()
        self._worker = SimulationWorker(self._settings)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.component_state_updated.connect(self.component_state_updated)
        self._worker.simulation_completed.connect(self._on_completed)
        self._worker.simulation_error.connect(self._on_error)
        self._worker.log_message.connect(self._log)

        self._thread.start()
        self._update_timer.start(200)

    def pause_simulation(self):
        if self._worker:
            self._worker.pause()
        self._log("Simulation paused")

    def stop_simulation(self):
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        self._log("Simulation stopped")

    def _cleanup_thread(self):
        self._update_timer.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None

    @Slot(int, float, dict)
    def _on_progress(self, progress: int, sim_time: float, stats: Dict[str, Any]):
        self.progress_bar.setValue(progress)
        pass_num = stats.get("pass_number", 0)
        total_passes = stats.get("total_passes", 1)
        if total_passes > 1 and pass_num > 0:
            self.card_time.set_value(f"{sim_time:.3f} s  (P{pass_num}/{total_passes})")
        else:
            self.card_time.set_value(f"{sim_time:.3f} s")
        self.card_particles.set_value(f"{stats.get('active_particles', 0):,}")
        self.card_fines.set_value(f"{stats.get('collected_fines', 0):,}")
        self.card_coarse.set_value(f"{stats.get('collected_coarse', 0):,}")

        eff = stats.get("separation_efficiency", 0)
        if eff > 0:
            self.card_efficiency.set_value(f"{eff:.1f}%")

        # Forward sim time to animation controller via main window
        self.sim_time_updated.emit(sim_time)

    @Slot(dict)
    def _on_completed(self, results: Dict[str, Any]):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        wall = results.get('wall_time_s', 0)
        steps = int(self._settings.total_time / self._settings.dt)
        self._log("=" * 56)
        self._log("SIMULATION COMPLETED")
        self._log(f"  Wall time:   {wall:.1f}s  ({steps / wall:.0f} steps/s)" if wall > 0 else "  Wall time:   --")
        self._log(f"  Processed:   {results.get('particles_processed', 0):,}")
        self._log(f"  Fines:       {results.get('fines_collected', 0):,}")
        self._log(f"    Cyclone 1: {results.get('cyclone_1', 0):,}")
        self._log(f"    Cyclone 2: {results.get('cyclone_2', 0):,}")
        self._log(f"    Cyclone 3: {results.get('cyclone_3_protein', 0):,}")
        self._log(f"    Bag filter: {results.get('bagfilter', 0):,}")
        self._log(f"  Coarse:      {results.get('coarse_collected', 0):,}")
        self._log(f"    Zigzag:    {results.get('coarse', 0):,}")
        self._log(f"    Wheel:     {results.get('wheel_coarse', 0):,}")
        self._log(f"  Escaped:     {results.get('escaped', 0):,}")
        self._log(f"  Efficiency:  {results.get('separation_efficiency', 0):.1f}%")
        # Cyclone size stats
        for key, st in results.get("cyclone_stats", {}).items():
            n = st.get("count", 0)
            d50 = st.get("design_d50_um")
            mean = st.get("mean_d_um")
            median = st.get("median_d_um")
            d50_s = f"d50={d50:.0f}\u00b5m" if d50 else ""
            mean_s = f"mean={mean:.1f}\u00b5m" if mean else ""
            med_s = f"median={median:.1f}\u00b5m" if median else ""
            self._log(f"    {key}: N={n:,}  {d50_s}  {mean_s}  {med_s}")
        # Multi-pass breakdown
        n_passes = results.get("num_passes", 1)
        if n_passes > 1:
            self._log(f"\n  Multi-Pass ({n_passes} passes):")
            for pr in results.get("pass_results", []):
                c = pr['counts']
                self._log(
                    f"    Pass {pr['pass']}: "
                    f"Zc={c.get('coarse',0):,} Wc={c.get('wheel_coarse',0):,} "
                    f"Cy1={c.get('cyclone_1',0):,} Cy3={c.get('cyclone_3_protein',0):,} "
                    f"Active={c.get('active',0):,}"
                )
        self._log("=" * 56)
        # Forward results to the Results panel
        self.simulation_results_ready.emit(results)

    @Slot(str)
    def _on_error(self, error: str):
        self._cleanup_thread()
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._log(f"ERROR:\n{error}")

    def _update_display(self):
        pass

    def _log(self, message: str):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    # ================================================================
    # Get / Set
    # ================================================================

    def get_settings(self) -> SimulationSettings:
        return self._settings

    def set_settings(self, settings: SimulationSettings):
        self._settings = settings
        # Sync all widgets
        self.total_time_spin.setValue(settings.total_time)
        self.dt_spin.setValue(settings.dt)
        self.output_spin.setValue(settings.output_interval)
        self.num_particles_spin.setValue(settings.num_particles)
        self.feed_rate_spin.setValue(settings.particle_feed_rate)
        self.continuous_check.setChecked(settings.continuous_feeding)
        self.particle_dia_spin.setValue(settings.particle_diameter_um)
        self.particle_std_spin.setValue(settings.particle_std_um)
        self.device_combo.setCurrentText(settings.device)
        self.material_combo.setCurrentText(settings.material_source)
        self.fraction_combo.setCurrentText(settings.material_fraction)
        self.air_flow_spin.setValue(settings.air_flow_m3s)
        self.bypass_spin.setValue(settings.bypass_ratio)
        self.wheel_rpm_spin.setValue(settings.wheel_rpm)
        self.wheel_diameter_spin.setValue(settings.wheel_diameter)
        self.turbulence_spin.setValue(settings.turbulence_base)
        self.restitution_spin.setValue(settings.restitution)
        self.friction_spin.setValue(settings.friction)
        self.max_loading_spin.setValue(settings.max_loading_ratio)
        self.blower_rpm_spin.setValue(settings.blower_rpm)
        self.throat_dia_spin.setValue(settings.venturi_throat_diameter_mm)
        self.zz_width_spin.setValue(settings.zigzag_width_mm)
        self.zz_depth_spin.setValue(settings.zigzag_depth_mm)
        idx = 0 if settings.use_preclassification else 1
        self.assembly_mode_combo.setCurrentIndex(idx)
        # Recirculation
        self.passes_spin.setValue(settings.recirculate_passes)
        self.recirc_fractions_combo.setCurrentText(settings.recirculate_fractions)
        self.attrition_spin.setValue(settings.attrition_factor)
        self.attrition_min_spin.setValue(settings.attrition_min_um)
        self.recirc_wheel_spin.setValue(settings.recirculate_wheel_rpm)
        self.recirc_time_spin.setValue(settings.recirculate_time)
        self._set_recirc_controls_enabled(settings.recirculate_passes > 1)
        # Refresh dynamic tooltips
        self._on_time_changed()
        self._on_wheel_rpm_changed(settings.wheel_rpm)
        if settings.blower_rpm > 0:
            self._on_blower_rpm_changed(settings.blower_rpm)
