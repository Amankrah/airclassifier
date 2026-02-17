"""
GP15Simulator — Main Entry Point
==================================

Digital twin of the QMTI GP-15 RF dielectric heating machine.
Orchestrates the assembled machine geometry with the coupled physics
engine, control logic, and pipeline integration.  Provides the public
API for the pretreatment module (engineering guide §7.1).

Architecture (engineering guide §6.1)::

    GP15Simulator
        ├── GP15MachineAssembly    ← 3D geometry (conveyor, oven, electrodes)
        ├── CoupledSimulator       ← Multi-physics engine (9-step loop §6.2)
        │     ├── RFFieldSolver    ← RF field (§4.1)
        │     ├── ThermalSolver    ← Heat transfer (§4.2)
        │     ├── MoistureSolver   ← Drying kinetics (§4.3)
        │     ├── EMUAirflowModel  ← Convective BCs (§2.4)
        │     ├── ConveyorDrive    ← Belt kinematics (§4.4)
        │     └── GP15Controller   ← PLC / safety logic (§8)
        └── OutletState            ← Pipeline output to milling (§9.1)

The machine assembly provides the **single source of truth** for all
geometric dimensions.  The simulation grid is derived from the
assembly's oven chamber geometry, ensuring perfect consistency
between the 3D render and the physics computation.

Usage::

    from airclassifier.pretreatment import GP15Simulator, MachineConfig, Recipe
    from airclassifier.pretreatment.materials.presets import get_material_preset

    gp15 = GP15Simulator(MachineConfig(), get_material_preset("yellow_pea"))
    gp15.load_recipe(Recipe(
        name="yellow_pea_standard", recipe_number=1,
        electrode_gap_mm=80, belt_speed_m_per_min=0.5,
    ))
    result = gp15.run(duration_s=180.0)
    outlet = gp15.get_outlet_conditions()
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np

from .config import MachineConfig, MaterialProperties, Recipe
from .control.recipe import RecipeStore
from .geometry.machine import (
    GP15MachineAssembly,
    COMPONENT_COLORS,
    create_gp15_machine,
)
from .geometry.oven import OvenGeometryParams
from .io.export import export_csv_timeseries, export_numpy_snapshot, export_vtk
from .particles import MaterialParticleSystem
from .physics.coupling import (
    CoupledSimulator,
    OutletState,
    PretreatmentResult,
    StepState,
)


def _detect_device() -> str:
    """Return ``"cuda"`` if an NVIDIA GPU is available, else ``"cpu"``.

    Checks for CUDA via Warp first (primary compute engine), then
    falls back to a basic ``torch.cuda`` / env-var check.
    """
    try:
        import warp as wp
        wp.init()
        if wp.is_cuda_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class GP15Simulator:
    """Digital twin of the QMTI GP-15 RF dielectric heating machine.

    Orchestrates the assembled machine geometry (§2) with the coupled
    physics engine (§4–6) and PLC control logic (§8).

    The ``GP15MachineAssembly`` is the **single source of truth** for
    all geometric dimensions.  The simulation grid (nx × ny × nz) is
    derived from the assembly's oven chamber, so the 3D visualisation
    and the physics computation are always consistent.

    The assembly follows the parameter chain from §6.1::

        MachineConfig
          → OvenChamberParams.from_machine(config)
              → ElectrodeParams.from_oven(oven)
              → InfeedHopperParams.from_oven(oven)
              → EMUParams.from_oven(oven)
              → GeneratorParams.from_oven(oven)

    This ensures belt width, RF zone length, electrode gap range, and
    all dependent dimensions propagate consistently through geometry
    and physics.

    Args:
        config: Machine specifications (generator, oven, conveyor, EMU).
        material: Feedstock properties (dielectric, thermal, moisture).
        device: Compute device (``"cuda"`` or ``"cpu"``).
        use_fdm: Phase 2 FDM RF field solver.
        use_tvd: Phase 2 Van Leer TVD advection (§4.4.1).
        power_constrained: Phase 2 Approach A voltage iteration (§4.1.3).
        target_power_kw: Target RF power for power-constrained mode.
        enable_controller: Phase 3 full PLC control logic (§8).
        oscillator_efficiency: Generator efficiency (default 0.56, §10.1).
        enable_corrections: Phase 3 fringe + perforation corrections (§4.1.4–5).
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        material: MaterialProperties | None = None,
        device: str | None = None,
        *,
        use_fdm: bool = True,
        use_tvd: bool = True,
        power_constrained: bool = False,
        target_power_kw: float | None = None,
        enable_controller: bool = True,
        oscillator_efficiency: float = 0.56,
        enable_corrections: bool = True,
    ):
        self.config = config or MachineConfig()
        self.material = material or MaterialProperties()
        # Auto-detect CUDA: use GPU if available, fall back to CPU.
        if device is None:
            device = _detect_device()
        self._device = device
        self._recipe: Optional[Recipe] = None

        # ── Machine assembly (single source of truth for geometry) ──
        # Derive oven params from MachineConfig so the simulation grid
        # (RF zone length, belt width, gap range) matches the physics
        # engine exactly.  The assembly's parameter chain (§6.1)
        # propagates these values to electrodes, hopper, EMU, and
        # generator automatically via __post_init__.
        oven_params = OvenGeometryParams.from_machine(self.config)
        self._assembly = create_gp15_machine(
            electrode_gap_m=self.config.electrode_gap_max_m,
            bed_depth_m=self.material.bed_depth_m,
            oven_params=oven_params,
        )

        # ── Simulation grid from assembly's oven geometry ───────────
        # The oven component provides get_grid_shape() and
        # get_cell_sizes() which discretize the RF zone (§3.2).
        self._grid_shape = self._assembly.oven.get_grid_shape()
        self._cell_sizes = self._assembly.oven.get_cell_sizes()

        # ── Physics engine (§6.2 nine-step coupling loop) ───────────
        self._sim = CoupledSimulator(
            machine=self.config,
            material=self.material,
            grid_shape=self._grid_shape,
            cell_sizes=self._cell_sizes,
            device=device,
            use_fdm=use_fdm,
            use_tvd=use_tvd,
            power_constrained=power_constrained,
            target_power_kw=target_power_kw,
            enable_controller=enable_controller,
            oscillator_efficiency=oscillator_efficiency,
            enable_corrections=enable_corrections,
        )

        # ── Lagrangian particle system (visual + E-L coupling) ────
        # Particles are created from the assembly geometry and the
        # pretreatment material.  Density and sphericity come from the
        # particles.material system; inlet T and M from the
        # pretreatment MaterialProperties.  See particles.py for the
        # full data-flow trace.
        self._particles = MaterialParticleSystem.from_assembly(
            self._assembly, self.material,
        )

        self._initialized = False

        # ── Recipe store (mirrors HMI 30-recipe system) ───────────
        self._recipe_store = RecipeStore(capacity=self.config.recipe_capacity)

    # ── Component accessors ─────────────────────────────────────────

    @property
    def recipe_store(self) -> RecipeStore:
        """The 30-slot recipe store (mirrors GP-15 HMI system)."""
        return self._recipe_store

    @property
    def assembly(self) -> GP15MachineAssembly:
        """The underlying machine assembly (geometry)."""
        return self._assembly

    @property
    def particles(self) -> MaterialParticleSystem:
        """The Lagrangian particle system for material visualization."""
        return self._particles

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Simulation grid dimensions ``(nx, ny, nz)``."""
        return self._grid_shape

    @property
    def cell_sizes(self) -> Tuple[float, float, float]:
        """Simulation cell sizes ``(dx, dy, dz)`` in metres."""
        return self._cell_sizes

    @property
    def sim_time(self) -> float:
        """Current simulation time [s]."""
        return self._sim._time

    @property
    def conveyor(self):
        """The conveyor drive controller (for animation state)."""
        return self._sim.conveyor

    @property
    def history(self):
        """Time-series of :class:`StepState` snapshots."""
        return self._sim._history

    @property
    def temperature_field(self) -> np.ndarray:
        """Current 3D temperature field (nx, ny, nz) [C]."""
        return self._sim.thermal.T

    @property
    def moisture_field(self) -> np.ndarray:
        """Current 3D moisture field (nx, ny, nz) [wet-basis]."""
        return self._sim.moisture.M

    @property
    def material_mask(self) -> np.ndarray:
        """3D material zone mask (nx, ny, nz): 0=air, 1=material, 2=belt."""
        return self._sim.cell_is_material

    def compute_stable_dt(self) -> float:
        """Compute a stable timestep from CFL and Courant constraints."""
        self._ensure_initialized()
        recipe = self._recipe or Recipe()
        return self._sim.compute_stable_dt(recipe)

    def step_n(self, n_steps: int, recipe: Recipe | None = None) -> StepState:
        """Advance *n_steps* simulation timesteps with adaptive dt.

        Convenience for the live visualization loop.  Returns the
        last :class:`StepState`.
        """
        self._ensure_initialized()
        recipe = recipe or self._recipe or Recipe()
        state = None
        for _ in range(n_steps):
            dt = self._sim.compute_stable_dt(recipe)
            remaining = max(0.0, 1e12)  # no end-time limit
            dt = min(dt, remaining)
            state = self._sim.step(dt, recipe)
        return state

    def build_result(self) -> PretreatmentResult:
        """Build a :class:`PretreatmentResult` from current state."""
        return self._sim._build_result()

    def compute_run_duration(self, recipe: Recipe | None = None) -> float:
        """Compute the production run duration from mass and throughput.

        The total run duration consists of two phases:

        1. **Feed time**: Time to dispatch all material from hopper onto belt
           ``t_feed = run_mass_kg / throughput``

        2. **Run-out**: Time for the last material to travel from hopper
           discharge through the GP-15 into the collection bin.
           ``t_runout = belt_length / v_belt``

        From the GP-15 manual (p. 54): *"At the end of the production run
        allow product to 'run-out' to ensure there is no product in the
        GP-15, then press the GP-15 OFF button to stop processing."*

        Uses the throughput formula from the GP-15 manual (Chapter 5)::

            throughput = rho_bulk * bed_depth * belt_width * belt_speed

        Returns 0.0 if ``run_mass_kg`` is not set on the recipe.

        Returns:
            Estimated run duration [s].
        """
        recipe = recipe or self._recipe or Recipe()
        if recipe.run_mass_kg <= 0:
            return 0.0

        mat = self.material
        v_belt = recipe.belt_speed_m_per_min / 60.0  # m/s
        if v_belt <= 0:
            return 0.0

        rho_bulk = mat.bulk_density(mat.initial_moisture_wb)
        bed_cross = mat.bed_depth_m * self.config.belt_width_m
        throughput_kg_per_s = rho_bulk * bed_cross * v_belt

        if throughput_kg_per_s <= 0:
            return 0.0

        # ── Feed time: hopper → belt dispatch ──────────────────────────
        feed_time_s = recipe.run_mass_kg / throughput_kg_per_s

        # ── Run-out: last material reaches the collection bin ─────────
        # Particles must travel from hopper spawn to head roller (where
        # they fall off the belt), then fall into the collection bin.
        #
        # Manual p.54: "allow product to 'run-out' to ensure there
        # is no product in the GP-15, then press the GP-15 OFF button."
        #
        # The simulation must run until ALL particles have been collected,
        # not just until they exit the oven RF zone.
        if self._particles is not None:
            pcfg = self._particles.cfg
            hopper_to_head_roller = pcfg.head_roller_x - pcfg.spawn_x
        else:
            cp = self._assembly.conveyor.params
            op = self._assembly.oven.params
            spawn_x = op.oven_x_start_m - 0.35
            head_x = cp.frame_length_m - cp.nose_length_m
            hopper_to_head_roller = head_x - spawn_x

        belt_runout_s = hopper_to_head_roller / v_belt

        # Add buffer for falling phase (~1-2 seconds for gravity drop)
        fall_buffer_s = 2.0

        return feed_time_s + belt_runout_s + fall_buffer_s

    def compute_run_timing(self, recipe: Recipe | None = None) -> dict:
        """Compute detailed timing breakdown for a production run.

        Returns a dictionary with:
            - feed_time_s: Time to dispatch all mass from hopper [s]
            - oven_clearing_s: Time for material to exit oven RF zone [s]
            - belt_runout_s: Time for material to reach head roller [s]
            - fall_buffer_s: Buffer for falling into collection bin [s]
            - total_duration_s: Total run duration (feed + runout + fall) [s]
            - oven_clearing_m: Distance from spawn to oven exit [m]
            - belt_length_m: Distance from hopper discharge to head roller [m]
            - throughput_kg_per_s: Mass flow rate [kg/s]

        This is useful for displaying timing information in the UI
        and for understanding why the simulation takes a certain time.
        """
        recipe = recipe or self._recipe or Recipe()
        result = {
            "feed_time_s": 0.0,
            "oven_clearing_s": 0.0,
            "belt_runout_s": 0.0,
            "fall_buffer_s": 0.0,
            "total_duration_s": 0.0,
            "oven_clearing_m": 0.0,
            "belt_length_m": 0.0,
            "throughput_kg_per_s": 0.0,
        }

        if recipe.run_mass_kg <= 0:
            return result

        mat = self.material
        v_belt = recipe.belt_speed_m_per_min / 60.0  # m/s
        if v_belt <= 0:
            return result

        rho_bulk = mat.bulk_density(mat.initial_moisture_wb)
        bed_cross = mat.bed_depth_m * self.config.belt_width_m
        throughput_kg_per_s = rho_bulk * bed_cross * v_belt

        if throughput_kg_per_s <= 0:
            return result

        # Feed time
        feed_time_s = recipe.run_mass_kg / throughput_kg_per_s

        # Belt distances and timing
        if self._particles is not None:
            pcfg = self._particles.cfg
            oven_clearing_m = pcfg.oven_x_end - pcfg.spawn_x
            belt_length_m = pcfg.head_roller_x - pcfg.spawn_x
        else:
            cp = self._assembly.conveyor.params
            op = self._assembly.oven.params
            spawn_x = op.oven_x_start_m - 0.35
            head_x = cp.frame_length_m - cp.nose_length_m
            oven_clearing_m = op.oven_x_end_m - spawn_x
            belt_length_m = head_x - spawn_x

        oven_clearing_s = oven_clearing_m / v_belt
        belt_runout_s = belt_length_m / v_belt

        # Buffer for falling phase (~1-2s for gravity drop into bin)
        fall_buffer_s = 2.0

        result["feed_time_s"] = feed_time_s
        result["oven_clearing_s"] = oven_clearing_s
        result["belt_runout_s"] = belt_runout_s
        result["fall_buffer_s"] = fall_buffer_s
        # Total: feed + belt travel to head roller + fall into bin
        result["total_duration_s"] = feed_time_s + belt_runout_s + fall_buffer_s
        result["oven_clearing_m"] = oven_clearing_m
        result["belt_length_m"] = belt_length_m
        result["throughput_kg_per_s"] = throughput_kg_per_s

        return result

    # ------------------------------------------------------------------
    # Public API  (§7.1)
    # ------------------------------------------------------------------

    def load_recipe(self, recipe: Recipe) -> None:
        """Load a processing recipe (mirrors HMI recipe system).

        Sets electrode gap, belt speed, RF power, extraction fan,
        heater settings, MRH/MRL thresholds.  Updates the assembly's
        electrode gap so subsequent ``get_mesh()`` calls render the
        upper electrode at the correct position.

        When ``run_mass_kg > 0``, configures the particle system for
        finite-mass feed: particles start in the hopper, dispatch onto
        the belt, and the belt clears at the end of the run.

        The recipe is also stored in the :attr:`recipe_store` (30-slot
        capacity, mirrors the GP-15 HMI) so it can be recalled later
        via ``recipe_store.load(recipe_number)``.
        """
        self._recipe = recipe
        # Persist in the 30-slot recipe store (HMI mirror)
        if recipe.recipe_number >= 0:
            self._recipe_store.store(recipe)
        # Sync assembly geometry with recipe setpoints
        self._assembly.set_electrode_gap(recipe.electrode_gap_mm / 1000.0)
        self._assembly.set_bed_depth(self.material.bed_depth_m)

        # Configure particle system for finite mass feed (hopper → belt → bin)
        if self._particles is not None and recipe.run_mass_kg > 0:
            v_belt = recipe.belt_speed_m_per_min / 60.0
            rho_bulk = self.material.bulk_density(self.material.initial_moisture_wb)
            bed_cross = self.material.bed_depth_m * self.config.belt_width_m
            throughput = rho_bulk * bed_cross * v_belt
            self._particles.set_run_mass(
                recipe.run_mass_kg,
                throughput_kg_per_s=throughput,
            )

    def run(
        self,
        duration_s: float | None = None,
        dt: float | None = None,
        adaptive_dt: bool = True,
    ) -> PretreatmentResult:
        """Run the full simulation.

        If ``duration_s`` is not provided, it is computed from the
        recipe's ``run_mass_kg`` using the throughput formula from
        the GP-15 manual (Chapter 5)::

            duration = mass / (rho_bulk * bed_depth * belt_width * belt_speed)

        This matches how the real machine operates: the operator
        loads a known mass, and the machine runs until all material
        has passed through.

        Args:
            duration_s: Total simulation time [s].  If ``None``,
                computed from ``recipe.run_mass_kg``.
            dt: Fixed timestep [s].  If ``None``, auto-computed.
            adaptive_dt: Recompute ``dt`` each step for stability.

        Returns:
            :class:`PretreatmentResult` with time-series and final fields.
        """
        self._ensure_initialized()
        recipe = self._recipe or Recipe()

        if duration_s is None:
            duration_s = self.compute_run_duration(recipe)
            if duration_s <= 0:
                raise ValueError(
                    "Cannot compute run duration: set recipe.run_mass_kg "
                    "or pass duration_s explicitly."
                )

        return self._sim.run(
            duration_s=duration_s,
            dt=dt,
            recipe=recipe,
            adaptive_dt=adaptive_dt,
        )

    def step(self, dt: float) -> StepState:
        """Advance one timestep.  For interactive / real-time GUI use."""
        self._ensure_initialized()
        recipe = self._recipe or Recipe()
        return self._sim.step(dt, recipe)

    def get_outlet_conditions(self) -> OutletState:
        """Get material state at the outfeed cross-section.

        Returns temperature and moisture fields at x = L_oven,
        averaged KPIs, and throughput metrics.  This is the interface
        to the downstream milling module (§9.1).
        """
        return self._sim.get_outlet_conditions(self._recipe)

    def get_mesh(self) -> Dict[str, Any]:
        """Return all component meshes for 3D visualization (§9.3).

        Uses the machine assembly's ``generate_all_meshes()`` directly,
        ensuring perfect consistency between geometry and physics.
        The assembly holds the correct electrode gap from
        ``load_recipe``, so the upper electrode renders at the recipe
        setpoint.

        When the simulation has been run, adds a ``"fields"`` entry
        containing the 3D temperature, moisture, and power density
        arrays for colour-mapping the material bed.

        Returns:
            Dict of ``{name: (vertices, triangles, metadata)}``
            plus optional ``"fields"`` dict with simulation data.
        """
        meshes = self._assembly.generate_all_meshes(skip_material_bed=True)

        # Attach field data if simulation has run
        if self._initialized:
            meshes["fields"] = {
                "temperature": self._sim.thermal.T.copy(),
                "moisture": self._sim.moisture.M.copy(),
                "power_density": self._sim.P_v.copy(),
                "grid_shape": self._grid_shape,
                "cell_sizes": self._cell_sizes,
            }

        return meshes

    def get_field_world_origin(self) -> Tuple[float, float, float]:
        """World-coordinate origin of the simulation field volume.

        The simulation grid covers the RF zone inside the oven (§3).
        This method returns the (x, y, z) offset needed to place the
        field arrays in the machine assembly's world frame.

        Returns:
            ``(x0, y0, z0)`` in metres — the lower-left-near corner
            of the simulation domain in assembly coordinates.
        """
        info = self._assembly.get_assembly_info()
        return (
            info["rf_zone_x_start_m"],
            0.0,  # y = 0 is deck-plate / lower electrode
            self._assembly.oven.params.conveyor_belt_z0_m,
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_vtk(self, filepath: str) -> None:
        """Export current fields as VTK structured grid."""
        if not self._initialized:
            raise RuntimeError("Simulation has not been run yet.")
        export_vtk(
            filepath,
            self._grid_shape,
            self._cell_sizes,
            {
                "Temperature": self._sim.thermal.T,
                "Moisture": self._sim.moisture.M,
                "RF_Power": self._sim.P_v,
            },
        )

    def export_csv(self, filepath: str) -> None:
        """Export time-series KPIs as CSV.

        Exports all 16 time-series fields from :class:`StepState` for
        calibration comparison and gap control analysis:

        time_s, T_mean_c, T_max_c, T_outfeed_c, T_outfeed_sensor_c,
        M_mean_wb, M_outfeed_wb, rf_power_kw, evap_power_kw, anode_current_a,
        electrode_gap_mm, total_energy_kwh, water_removed_kg,
        specific_energy_kwh_per_kg, electrode_temperature_c,
        protein_denaturation (Phase 4).
        """
        if not self._sim._history:
            raise RuntimeError("No time-series data — run the simulation first.")
        h = self._sim._history
        ts = {
            "time_s": [s.time_s for s in h],
            "T_mean_c": [s.T_mean_c for s in h],
            "T_max_c": [s.T_max_c for s in h],
            "T_outfeed_c": [s.T_outfeed_c for s in h],
            "T_outfeed_sensor_c": [s.T_outfeed_sensor_c for s in h],
            "M_mean_wb": [s.M_mean_wb for s in h],
            "M_outfeed_wb": [s.M_outfeed_wb for s in h],
            "rf_power_kw": [s.rf_power_kw for s in h],
            "evap_power_kw": [s.evap_power_kw for s in h],
            "anode_current_a": [s.anode_current_a for s in h],
            "electrode_gap_mm": [s.electrode_gap_mm for s in h],
            "total_energy_kwh": [s.total_energy_kwh for s in h],
            "water_removed_kg": [s.water_removed_kg for s in h],
            "specific_energy_kwh_per_kg": [s.specific_energy_kwh_per_kg for s in h],
            "electrode_temperature_c": [s.electrode_temperature_c for s in h],
            "protein_denaturation": [s.protein_denaturation for s in h],
        }
        export_csv_timeseries(filepath, ts)

    def export_snapshot(self, directory: str) -> None:
        """Save current fields as NumPy .npy files."""
        if not self._initialized:
            raise RuntimeError("Simulation has not been run yet.")
        export_numpy_snapshot(
            directory,
            self._sim._time,
            self._sim.thermal.T,
            self._sim.moisture.M,
            self._sim.P_v,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        """Lazy initialization: build mask and set initial conditions."""
        if self._initialized:
            return
        gap_m = (self._recipe.electrode_gap_mm / 1000.0
                 if self._recipe else self.config.electrode_gap_max_m)

        # Build material mask from the assembly's oven geometry (§3.1)
        mask = self._assembly.oven.build_material_mask(
            electrode_gap_m=gap_m,
            bed_depth_m=self.material.bed_depth_m,
            belt_stack_m=self.config.belt_stack_thickness_m,
        )
        self._sim.initialize(cell_is_material=mask, electrode_gap_m=gap_m)

        # Connect particle system to the coupling loop (step 10)
        self._sim._particles = self._particles
        self._sim._grid_origin = self.get_field_world_origin()

        if self._recipe:
            self._sim.controller.load_recipe(self._recipe)
            self._sim.controller.start()
        self._initialized = True
