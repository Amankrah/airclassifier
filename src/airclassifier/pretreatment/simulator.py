"""
GP15Simulator — Main Entry Point
==================================

Digital twin of the QMTI GP-15 RF dielectric heating machine.
Orchestrates geometry, physics solvers, control logic, and the
coupled simulation loop.  Provides the public API for the
pretreatment module (engineering guide §7.1).

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

from typing import Optional, Dict, Any

import numpy as np

from .config import MachineConfig, MaterialProperties, Recipe
from .geometry.conveyor import ConveyorGeometry, ConveyorParams
from .geometry.electrode import ElectrodeGeometry, ElectrodeParams
from .geometry.oven import OvenGeometry, OvenGeometryParams
from .io.export import export_csv_timeseries, export_numpy_snapshot, export_vtk
from .physics.coupling import (
    CoupledSimulator,
    OutletState,
    PretreatmentResult,
    StepState,
)


class GP15Simulator:
    """Digital twin of the QMTI GP-15 RF dielectric heating machine.

    Orchestrates geometry, physics solvers, control logic, and the
    coupled simulation loop.  Provides the public API for the
    pretreatment module.

    Args:
        config: Machine specifications (electrode dimensions, power, etc.)
        material: Feedstock properties (dielectric, thermal, moisture).
        device: Warp device (``"cuda"`` or ``"cpu"``).
        use_fdm: Phase 2 FDM RF field solver.
        use_tvd: Phase 2 Van Leer TVD advection.
        power_constrained: Phase 2 power-constrained voltage iteration.
        target_power_kw: Target RF power for power-constrained mode.
        enable_controller: Phase 3 full PLC control logic.
        oscillator_efficiency: Generator efficiency (default 0.56, §10.1).
        enable_corrections: Phase 3 fringe + perforation corrections.
    """

    def __init__(
        self,
        config: MachineConfig | None = None,
        material: MaterialProperties | None = None,
        device: str = "cpu",
        *,
        use_fdm: bool = False,
        use_tvd: bool = True,
        power_constrained: bool = False,
        target_power_kw: float | None = None,
        enable_controller: bool = True,
        oscillator_efficiency: float = 0.56,
        enable_corrections: bool = True,
    ):
        self.config = config or MachineConfig()
        self.material = material or MaterialProperties()
        self._device = device
        self._recipe: Optional[Recipe] = None

        # Build geometry
        self._oven = OvenGeometry(OvenGeometryParams.from_machine(self.config))
        self._grid_shape = self._oven.get_grid_shape()
        self._cell_sizes = self._oven.get_cell_sizes()

        # Build simulator
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

        # Geometry helpers (for visualization)
        self._electrode = ElectrodeGeometry(
            ElectrodeParams.from_machine(self.config)
        )
        self._conveyor = ConveyorGeometry(
            ConveyorParams.from_machine(self.config)
        )

        self._initialized = False

    # ------------------------------------------------------------------
    # Public API  (§7.1)
    # ------------------------------------------------------------------

    def load_recipe(self, recipe: Recipe) -> None:
        """Load a processing recipe (mirrors HMI recipe system).

        Sets electrode gap, belt speed, RF power, extraction fan,
        heater settings, MRH/MRL thresholds.
        """
        self._recipe = recipe
        # Controller will receive the recipe at run() time

    def run(
        self,
        duration_s: float,
        dt: float | None = None,
        adaptive_dt: bool = True,
    ) -> PretreatmentResult:
        """Run the full simulation for the specified duration.

        Executes the coupled physics loop, returns complete results
        including time-series of all fields and KPIs.

        Args:
            duration_s: Total simulation time [s].
            dt: Fixed timestep [s].  If ``None``, auto-computed.
            adaptive_dt: Recompute ``dt`` each step for stability.

        Returns:
            :class:`PretreatmentResult` with time-series and final fields.
        """
        self._ensure_initialized()
        recipe = self._recipe or Recipe()
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
        """Return PyVista-compatible mesh data for 3D visualization.

        Returns a dict containing mesh components and field data
        suitable for adding to the Air Classifier Designer's
        PyVista viewport (§9.3).
        """
        gap_m = (self._recipe.electrode_gap_mm / 1000.0
                 if self._recipe else self.config.electrode_gap_max_m)

        meshes: Dict[str, Any] = {}

        # Oven walls
        v, t, m = self._oven.generate_mesh()
        meshes["oven"] = {"vertices": v, "triangles": t, "metadata": m}

        # Electrodes
        v, t, m = self._electrode.generate_upper_mesh(gap_m)
        meshes["upper_electrode"] = {"vertices": v, "triangles": t, "metadata": m}
        v, t, m = self._electrode.generate_lower_mesh()
        meshes["lower_electrode"] = {"vertices": v, "triangles": t, "metadata": m}

        # Conveyor belt
        v, t, m = self._conveyor.generate_belt_mesh()
        meshes["belt"] = {"vertices": v, "triangles": t, "metadata": m}

        # Material bed
        v, t, m = self._conveyor.generate_bed_mesh(self.material.bed_depth_m)
        meshes["bed"] = {"vertices": v, "triangles": t, "metadata": m}

        # Field data (if simulation has run)
        if self._initialized:
            meshes["fields"] = {
                "temperature": self._sim.thermal.T.copy(),
                "moisture": self._sim.moisture.M.copy(),
                "power_density": self._sim.P_v.copy(),
                "grid_shape": self._grid_shape,
                "cell_sizes": self._cell_sizes,
            }

        return meshes

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
        """Export time-series KPIs as CSV."""
        if not self._sim._history:
            raise RuntimeError("No time-series data — run the simulation first.")
        ts = {
            "time_s": [s.time_s for s in self._sim._history],
            "T_mean_c": [s.T_mean_c for s in self._sim._history],
            "T_max_c": [s.T_max_c for s in self._sim._history],
            "M_mean_wb": [s.M_mean_wb for s in self._sim._history],
            "rf_power_kw": [s.rf_power_kw for s in self._sim._history],
            "anode_current_a": [s.anode_current_a for s in self._sim._history],
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
        mask = self._oven.build_material_mask(
            electrode_gap_m=gap_m,
            bed_depth_m=self.material.bed_depth_m,
            belt_stack_m=self.config.belt_stack_thickness_m,
        )
        self._sim.initialize(cell_is_material=mask, electrode_gap_m=gap_m)
        if self._recipe:
            self._sim.controller.load_recipe(self._recipe)
            self._sim.controller.start()
        self._initialized = True
