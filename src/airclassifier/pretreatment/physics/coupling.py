"""
Multi-Physics Coupling Orchestrator
====================================

Sequences the coupled physics solvers each timestep:

    1. ADVECT — shift T and M fields by belt velocity
    2. RF FIELD — solve Laplace for |E|^2
    3. HEATING — compute P_v from |E|^2 and eps''
    4. EVAPORATION — (computed inside moisture.step)
    5. THERMAL — advance T with RF source and latent sink
    6. MOISTURE — advance M with diffusion and evaporation
    7. PROPERTIES — update eps', eps'', rho, c_p, k, D_eff
    8. CONTROLLER — PLC logic (MRH/MRL, gap, temperature control)
    9. RECORD — log outfeed state and KPIs

Phase 2 additions:
    - Phase 2 FDM RF field solver (Jacobi / SOR)
    - Power-constrained voltage iteration (Approach A)
    - Van Leer TVD advection
    - Adaptive timestep from CFL + Courant limits
    - OutletState for pipeline integration

Phase 3 additions:
    - GP15Controller with MRH/MRL, recycle, temperature control
    - SafetyMonitor with arc detection and lockout
    - Electrode perforation + fringe field corrections
    - Oscillator efficiency for calibration (§10.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

from ..config import MachineConfig, MaterialProperties, Recipe
from ..control.controller import GP15Controller, ControllerState
from ..geometry.electrode import ElectrodeGeometry, ElectrodeParams
from ..geometry.oven import OvenGeometry, OvenGeometryParams
from ..kernels.dielectric_heating import (
    TWO_PI_F_EPS0,
    compute_power_density_np,
    update_material_properties_np,
)
from ..kernels.transport import advect_material_np, advect_material_tvd_np
from .rf_field import RFFieldSolver
from .thermal import ThermalSolver
from .moisture import MoistureSolver
from .airflow import EMUAirflowModel


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class StepState:
    """State snapshot after a single simulation step."""
    time_s: float = 0.0
    T_mean_c: float = 0.0
    T_max_c: float = 0.0
    M_mean_wb: float = 0.0
    M_min_wb: float = 0.0
    rf_power_kw: float = 0.0
    anode_current_a: float = 0.0
    electrode_gap_mm: float = 0.0
    belt_speed_m_per_min: float = 0.0


@dataclass
class PretreatmentResult:
    """Complete simulation results."""
    duration_s: float = 0.0
    final_moisture_mean_wb: float = 0.0
    final_temperature_mean_c: float = 0.0
    energy_consumed_kwh: float = 0.0
    throughput_kg_per_h: float = 0.0
    time_series: Dict[str, Any] = field(default_factory=dict)
    # Full 3D fields at final time
    T_final: Optional[np.ndarray] = None
    M_final: Optional[np.ndarray] = None


@dataclass
class OutletState:
    """Material state at the GP-15 outfeed. Input to the milling module.

    See engineering guide §9.1.
    """
    # Spatially-resolved fields at the outfeed cross-section (Y x Z)
    temperature_field: Optional[np.ndarray] = None  # [ny, nz] degC
    moisture_field: Optional[np.ndarray] = None      # [ny, nz] wet-basis

    # Bulk averages
    avg_temperature_c: float = 0.0
    avg_moisture_wb: float = 0.0
    moisture_uniformity: float = 0.0   # CV = std / mean

    # Process metrics
    throughput_kg_per_hr: float = 0.0
    total_energy_kwh: float = 0.0
    specific_energy_kwh_per_kg: float = 0.0   # kWh per kg water removed
    residence_time_s: float = 0.0

    # Quality indicators
    max_temperature_c: float = 0.0
    protein_denaturation_fraction: float = 0.0


class CoupledSimulator:
    """Orchestrates the multi-physics coupling loop.

    This is the internal engine.  Users should use ``GP15Simulator``
    instead, which wraps this with geometry, control, and I/O.

    Phase 2 capabilities:
        - FDM RF field solver with per-cell eps' (``use_fdm=True``)
        - Power-constrained voltage iteration (``power_constrained=True``)
        - Van Leer TVD advection (``use_tvd=True``)
        - Adaptive timestep from CFL + Courant limits
        - OutletState for pipeline integration
    """

    def __init__(
        self,
        machine: MachineConfig,
        material: MaterialProperties,
        grid_shape: tuple,
        cell_sizes: tuple,
        device: str = "cpu",
        *,
        use_fdm: bool = False,
        use_tvd: bool = False,
        power_constrained: bool = False,
        target_power_kw: float | None = None,
        enable_controller: bool = False,
        oscillator_efficiency: float = 0.56,
        enable_corrections: bool = False,
    ):
        self._machine = machine
        self._material = material
        self._grid_shape = grid_shape
        self._cell_sizes = cell_sizes
        self._device = device
        self._time = 0.0

        # Phase 2 options
        self._use_fdm = use_fdm
        self._use_tvd = use_tvd
        self._power_constrained = power_constrained
        self._target_power_kw = target_power_kw

        # Phase 3 options
        self._enable_controller = enable_controller
        self._oscillator_efficiency = oscillator_efficiency
        self._enable_corrections = enable_corrections

        # Cell volume
        dx, dy, dz = cell_sizes
        self._cell_vol = dx * dy * dz

        # Solvers
        self.rf = RFFieldSolver(grid_shape, cell_sizes, machine, device)
        self.thermal = ThermalSolver(grid_shape, cell_sizes, device)
        self.moisture = MoistureSolver(grid_shape, cell_sizes, device)
        self.airflow = EMUAirflowModel(machine)

        # Phase 3: controller
        self.controller = GP15Controller(machine)

        # Phase 3: electrode correction fields
        self._perf_correction: np.ndarray | None = None
        self._fringe_correction: np.ndarray | None = None
        if enable_corrections:
            elec = ElectrodeGeometry(ElectrodeParams.from_machine(machine))
            self._perf_correction = elec.get_perforation_correction_field(grid_shape)
            # Fringe correction is gap-dependent, computed per solve

        # Material property arrays (3-D)
        self.eps_loss = np.zeros(grid_shape, dtype=np.float32)
        self.eps_real = np.zeros(grid_shape, dtype=np.float32)
        self.rho_cp = np.zeros(grid_shape, dtype=np.float32)
        self.k_eff = np.zeros(grid_shape, dtype=np.float32)
        self.rho_dry = np.zeros(grid_shape, dtype=np.float32)
        self.cell_is_material = np.zeros(grid_shape, dtype=np.int32)

        # RF heating field
        self.P_v = np.zeros(grid_shape, dtype=np.float32)

        # KPI accumulators
        self._total_rf_energy_j = 0.0
        self._history: List[StepState] = []

        # Cached bed surface index
        self._j_surface: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(
        self,
        cell_is_material: np.ndarray | None = None,
        electrode_gap_m: float | None = None,
    ):
        """Set initial conditions for T, M, and material properties.

        Args:
            cell_is_material: Pre-built material mask.  If ``None``, one
                is generated from oven geometry and material properties.
            electrode_gap_m: Electrode gap for mask generation [m].
        """
        mat = self._material
        self.thermal.initialize(mat.initial_temperature_c)
        self.moisture.initialize(mat.initial_moisture_wb)

        # Build cell_is_material
        if cell_is_material is not None:
            self.cell_is_material[:] = cell_is_material
        else:
            oven = OvenGeometry(OvenGeometryParams(
                length=self._machine.oven_length_m,
                width=self._machine.belt_width_m,
                height=self._machine.electrode_gap_max_m,
                resolution=self._grid_shape[2],
            ))
            gap = electrode_gap_m or self._machine.electrode_gap_max_m
            self.cell_is_material[:] = oven.build_material_mask(
                electrode_gap_m=gap,
                bed_depth_m=mat.bed_depth_m,
                belt_stack_m=self._machine.belt_stack_thickness_m,
            )

        # Initial property fill
        self._update_properties()

        # Compute dry-basis density (only in material cells)
        mat_mask = (self.cell_is_material == 1)
        self.rho_dry[mat_mask] = mat.rho_solid * (1.0 - mat.bed_porosity)
        self.rho_dry[~mat_mask] = 1.0  # dummy guard

        # Cache bed surface index
        self._j_surface = self._find_bed_surface_j()

    # ------------------------------------------------------------------
    # Adaptive timestep
    # ------------------------------------------------------------------

    def compute_stable_dt(self, recipe: Recipe) -> float:
        """Compute a stable timestep from CFL and Courant constraints.

        CFL (thermal):  dt < factor * dmin^2 * rho_cp_min / k_max
        Courant (advection):  dt < 0.9 * dx / v_belt

        Returns:
            The minimum of the two constraints [s].
        """
        dx, dy, dz = self._cell_sizes
        mat_mask = (self.cell_is_material == 1)

        # CFL from thermal diffusivity
        k_max = float(np.max(self.k_eff[mat_mask])) if mat_mask.any() else 0.2
        rho_cp_min = float(np.min(self.rho_cp[mat_mask])) if mat_mask.any() else 1e5
        dt_cfl = self.thermal.get_cfl_dt(k_max, rho_cp_min)

        # Courant from advection
        v_belt = recipe.belt_speed_m_per_min / 60.0  # m/s
        if v_belt > 0:
            dt_courant = 0.9 * dx / v_belt
        else:
            dt_courant = dt_cfl  # no advection limit

        return min(dt_cfl, dt_courant)

    # ------------------------------------------------------------------
    # Single timestep
    # ------------------------------------------------------------------

    def step(self, dt: float, recipe: Recipe) -> StepState:
        """Execute one coupled physics timestep.

        Follows the 9-step sequence from the engineering guide (§6.2).
        """
        dx, dy, dz = self._cell_sizes
        mat = self._material
        machine = self._machine
        mat_mask = (self.cell_is_material == 1)

        # ── 1. ADVECT ─────────────────────────────────────────────────
        v_belt = recipe.belt_speed_m_per_min / 60.0  # m/s
        if v_belt > 0 and v_belt * dt / dx < 1.0:
            _advect = advect_material_tvd_np if self._use_tvd else advect_material_np
            self.thermal.T = _advect(
                self.thermal.T, v_belt, dx, dt,
                inlet_value=mat.initial_temperature_c,
            )
            self.moisture.M = _advect(
                self.moisture.M, v_belt, dx, dt,
                inlet_value=mat.initial_moisture_wb,
            )
            # Zero moisture in non-material cells after advection
            self.moisture.M[~mat_mask] = 0.0

        # ── 2. RF FIELD ───────────────────────────────────────────────
        gap_m = recipe.electrode_gap_mm / 1000.0

        # Update fringe correction for current gap (Phase 3)
        if self._enable_corrections:
            elec = ElectrodeGeometry(ElectrodeParams.from_machine(machine))
            self._fringe_correction = elec.get_fringe_field_correction(
                self._grid_shape, gap_m,
            )

        if self._power_constrained and self._target_power_kw:
            # Phase 2 Approach A: iterate V_rf to match target power
            V_rf_kv = self.rf.solve_power_constrained(
                electrode_gap_m=gap_m,
                target_power_kw=self._target_power_kw,
                eps_real=self.eps_real,
                eps_loss=self.eps_loss,
                cell_is_material=self.cell_is_material,
                cell_volume_m3=self._cell_vol,
                bed_depth_m=mat.bed_depth_m,
                belt_stack_m=machine.belt_stack_thickness_m,
                use_fdm=self._use_fdm,
            )
        elif self._use_fdm:
            # Phase 2 FDM: voltage-driven with per-cell eps'
            V_rf_kv = machine.anode_voltage_kv(machine.anode_current_no_load_a)
            self.rf.solve_fdm(gap_m, V_rf_kv, self.eps_real)
        else:
            # Phase 1: uniform parallel-plate
            V_rf_kv = machine.anode_voltage_kv(machine.anode_current_no_load_a)
            self.rf.solve(
                electrode_gap_m=gap_m,
                voltage_kv=V_rf_kv,
                eps_real=self.eps_real,
                cell_is_material=self.cell_is_material,
                bed_depth_m=mat.bed_depth_m,
                belt_stack_m=machine.belt_stack_thickness_m,
            )

        # ── 3. HEATING ────────────────────────────────────────────────
        # Apply electrode corrections to |E|^2 if enabled
        e_field = self.rf.e_field_sq
        if self._enable_corrections:
            if self._perf_correction is not None:
                # Perforation: (nx, nz) → broadcast over Y
                e_field = e_field * self._perf_correction[:, np.newaxis, :]**2
            if self._fringe_correction is not None:
                # Fringe: (nz,) → broadcast over X, Y
                e_field = e_field * self._fringe_correction[np.newaxis, np.newaxis, :]**2

        compute_power_density_np(e_field, self.eps_loss, out=self.P_v)
        P_rf_w = float(np.sum(self.P_v[mat_mask]) * self._cell_vol)
        P_rf_kw = P_rf_w / 1000.0

        # ── 4. EVAPORATION (computed inside moisture.step) ────────────

        # ── 5. THERMAL ────────────────────────────────────────────────
        self.thermal.step(
            dt=dt,
            P_v=self.P_v,
            evap_rate=self.moisture.evap_rate,
            rho_cp=self.rho_cp,
            k_eff=self.k_eff,
            cell_is_material=self.cell_is_material,
            T_inlet_c=mat.initial_temperature_c,
        )

        # Convective BC at bed surface
        self.airflow.update(recipe, dt)
        if self._j_surface > 0:
            self.thermal.apply_convection_bc(
                j_surface=self._j_surface,
                h_conv=self.airflow.state.convective_htc_w_per_m2k,
                T_air_c=self.airflow.state.air_temperature_c,
                rho_cp=self.rho_cp,
                k_eff=self.k_eff,
                dt=dt,
            )

        # ── 6. MOISTURE ───────────────────────────────────────────────
        self.moisture.step(
            dt=dt,
            T=self.thermal.T,
            cell_is_material=self.cell_is_material,
            rho_dry=self.rho_dry,
            D0=mat.D_eff_D0,
            Ea=mat.D_eff_Ea,
            k_evap=mat.k_evap,
            T_threshold=mat.T_evap_threshold_c,
            M_inlet_wb=mat.initial_moisture_wb,
        )

        # ── 7. PROPERTIES ─────────────────────────────────────────────
        self._update_properties()

        # ── 8. CONTROLLER ──────────────────────────────────────────────
        # Anode current from delivered power (generator model §8.4)
        # Generator power = RF power / oscillator efficiency
        P_gen_kw = P_rf_kw / max(self._oscillator_efficiency, 0.01)
        fraction = min(P_gen_kw / machine.max_rf_power_kw, 1.0) if P_gen_kw > 0 else 0.0
        I_a = (
            machine.anode_current_no_load_a
            + (machine.anode_current_full_load_a - machine.anode_current_no_load_a) * fraction
        )

        if self._enable_controller:
            T_outfeed = float(np.mean(
                self.thermal.T[-1, :, :][self.cell_is_material[-1, :, :] == 1]
            )) if np.any(self.cell_is_material[-1, :, :] == 1) else mat.initial_temperature_c

            ctrl_status = self.controller.step(
                dt=dt,
                anode_current_a=I_a,
                rf_power_kw=P_rf_kw,
                T_outfeed_c=T_outfeed,
                e_field_max=float(np.sqrt(np.max(self.rf.e_field_sq))),
            )
            # Controller may override recipe setpoints
            recipe = Recipe(
                name=recipe.name,
                recipe_number=recipe.recipe_number,
                electrode_gap_mm=ctrl_status.electrode_gap_mm,
                belt_speed_m_per_min=ctrl_status.belt_speed_m_per_min,
                rf_power_enabled=ctrl_status.rf_enabled,
                mrh_amps=recipe.mrh_amps,
                mrl_amps=recipe.mrl_amps,
                extraction_fan_hz=ctrl_status.extraction_fan_hz,
                heater_bank_1_on=ctrl_status.heater_bank_1_on,
                heater_bank_2_on=ctrl_status.heater_bank_2_on,
                heater_fan_hz=recipe.heater_fan_hz,
                temp_control_enabled=recipe.temp_control_enabled,
                temp_setpoint_c=recipe.temp_setpoint_c,
                temp_sensors_active=recipe.temp_sensors_active,
                temp_envelope_time_s=recipe.temp_envelope_time_s,
            )

        # ── 9. RECORD ─────────────────────────────────────────────────
        self._time += dt
        self._total_rf_energy_j += P_rf_w * dt

        T_mat = self.thermal.T[mat_mask]
        M_mat = self.moisture.M[mat_mask]
        T_mean = float(np.mean(T_mat)) if T_mat.size else mat.initial_temperature_c
        T_max = float(np.max(T_mat)) if T_mat.size else mat.initial_temperature_c
        M_mean = float(np.mean(M_mat)) if M_mat.size else mat.initial_moisture_wb
        M_min = float(np.min(M_mat)) if M_mat.size else 0.0

        state = StepState(
            time_s=self._time,
            T_mean_c=T_mean,
            T_max_c=T_max,
            M_mean_wb=M_mean,
            M_min_wb=M_min,
            rf_power_kw=P_rf_kw,
            anode_current_a=I_a,
            electrode_gap_mm=recipe.electrode_gap_mm,
            belt_speed_m_per_min=recipe.belt_speed_m_per_min,
        )
        self._history.append(state)
        return state

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(
        self,
        duration_s: float,
        dt: float | None = None,
        recipe: Recipe | None = None,
        adaptive_dt: bool = False,
    ) -> PretreatmentResult:
        """Run the full simulation for *duration_s* seconds.

        Args:
            duration_s: Total simulation time [s].
            dt: Fixed timestep [s].  If ``None`` and ``adaptive_dt``
                is ``True``, the timestep is computed from CFL/Courant.
            recipe: Processing recipe.
            adaptive_dt: If ``True``, recompute ``dt`` each step.

        Returns:
            A :class:`PretreatmentResult` with time-series and final fields.
        """
        if recipe is None:
            recipe = Recipe()

        # Load recipe into controller (Phase 3)
        if self._enable_controller:
            self.controller.load_recipe(recipe)
            self.controller.start()

        t_end = self._time + duration_s

        while self._time < t_end - 1e-12:
            if adaptive_dt or dt is None:
                _dt = self.compute_stable_dt(recipe)
            else:
                _dt = dt
            # Don't overshoot
            _dt = min(_dt, t_end - self._time)
            self.step(_dt, recipe)

        mat_mask = (self.cell_is_material == 1)
        M_final = self.moisture.M[mat_mask]
        T_final = self.thermal.T[mat_mask]

        # Throughput estimate
        belt_speed = recipe.belt_speed_m_per_min / 60.0
        bed_cross = self._material.bed_depth_m * self._machine.belt_width_m
        rho_bulk = self._material.bulk_density(self._material.initial_moisture_wb)
        throughput_kg_h = rho_bulk * bed_cross * belt_speed * 3600.0

        result = PretreatmentResult(
            duration_s=self._time,
            final_moisture_mean_wb=float(np.mean(M_final)) if M_final.size else 0.0,
            final_temperature_mean_c=float(np.mean(T_final)) if T_final.size else 0.0,
            energy_consumed_kwh=self._total_rf_energy_j / 3.6e6,
            throughput_kg_per_h=throughput_kg_h,
            time_series={
                "time_s": [s.time_s for s in self._history],
                "T_mean_c": [s.T_mean_c for s in self._history],
                "T_max_c": [s.T_max_c for s in self._history],
                "M_mean_wb": [s.M_mean_wb for s in self._history],
                "rf_power_kw": [s.rf_power_kw for s in self._history],
                "anode_current_a": [s.anode_current_a for s in self._history],
            },
            T_final=self.thermal.T.copy(),
            M_final=self.moisture.M.copy(),
        )
        return result

    # ------------------------------------------------------------------
    # Outlet conditions  (§9.1)
    # ------------------------------------------------------------------

    def get_outlet_conditions(self, recipe: Recipe | None = None) -> OutletState:
        """Collect material state at the outfeed cross-section (x = L_oven).

        Returns an :class:`OutletState` containing the Y-Z temperature
        and moisture fields at the last X-cell, bulk averages, process
        metrics, and quality indicators.
        """
        if recipe is None:
            recipe = Recipe()

        mat = self._material
        machine = self._machine
        mat_mask_yz = (self.cell_is_material[-1, :, :] == 1)

        T_yz = self.thermal.T[-1, :, :]
        M_yz = self.moisture.M[-1, :, :]

        T_mat = T_yz[mat_mask_yz]
        M_mat = M_yz[mat_mask_yz]

        avg_T = float(np.mean(T_mat)) if T_mat.size else mat.initial_temperature_c
        avg_M = float(np.mean(M_mat)) if M_mat.size else mat.initial_moisture_wb
        max_T = float(np.max(T_mat)) if T_mat.size else mat.initial_temperature_c

        # Moisture CV (coefficient of variation)
        if M_mat.size and avg_M > 1e-8:
            moisture_cv = float(np.std(M_mat) / avg_M)
        else:
            moisture_cv = 0.0

        # Throughput
        v_belt = recipe.belt_speed_m_per_min / 60.0
        bed_cross = mat.bed_depth_m * machine.belt_width_m
        rho_bulk = mat.bulk_density(mat.initial_moisture_wb)
        throughput_kg_h = rho_bulk * bed_cross * v_belt * 3600.0

        # Residence time
        residence_s = machine.oven_length_m / v_belt if v_belt > 0 else 0.0

        # Energy metrics
        total_kwh = self._total_rf_energy_j / 3.6e6
        water_removed_kg_h = throughput_kg_h * max(mat.initial_moisture_wb - avg_M, 0.0)
        if water_removed_kg_h > 0:
            specific_energy = total_kwh / (water_removed_kg_h / 3600.0 * residence_s)
        else:
            specific_energy = 0.0

        # Protein denaturation estimate (simplified: fraction of time T > 70 C)
        # For now, report 0 — detailed model in Phase 4
        denaturation = 0.0

        return OutletState(
            temperature_field=T_yz.copy(),
            moisture_field=M_yz.copy(),
            avg_temperature_c=avg_T,
            avg_moisture_wb=avg_M,
            moisture_uniformity=moisture_cv,
            throughput_kg_per_hr=throughput_kg_h,
            total_energy_kwh=total_kwh,
            specific_energy_kwh_per_kg=specific_energy,
            residence_time_s=residence_s,
            max_temperature_c=max_T,
            protein_denaturation_fraction=denaturation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_properties(self):
        """Recompute all material properties from current T and M."""
        mat = self._material
        a = mat.dielectric_loss_coeffs
        b = mat.dielectric_const_coeffs

        update_material_properties_np(
            T=self.thermal.T,
            M=self.moisture.M,
            cell_is_material=self.cell_is_material,
            a1=a[0], a2=a[1], a3=a[2], a4=a[3], a5=a[4],
            b1=b[0], b2=b[1], b3=b[2],
            c_p_dry=mat.c_p_dry,
            c_p_water=mat.c_p_water,
            k_dry=mat.k_dry,
            k_beta=mat.k_moisture_beta,
            rho_solid=mat.rho_solid,
            porosity=mat.bed_porosity,
            eps_loss_out=self.eps_loss,
            eps_real_out=self.eps_real,
            rho_cp_out=self.rho_cp,
            k_eff_out=self.k_eff,
        )

    def _find_bed_surface_j(self) -> int:
        """Find the Y-index of the top-most material cell (bed surface)."""
        ny = self._grid_shape[1]
        for j in range(ny - 1, -1, -1):
            if np.any(self.cell_is_material[:, j, :] == 1):
                return j
        return 0
