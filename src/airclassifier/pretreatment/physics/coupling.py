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
    8. CONTROLLER — PLC logic (placeholder for Phase 3)
    9. RECORD — log outfeed state and KPIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

from ..config import MachineConfig, MaterialProperties, Recipe
from ..geometry.oven import OvenGeometry, OvenGeometryParams
from ..kernels.dielectric_heating import (
    TWO_PI_F_EPS0,
    compute_power_density_np,
    update_material_properties_np,
)
from ..kernels.transport import advect_material_np
from .rf_field import RFFieldSolver
from .thermal import ThermalSolver
from .moisture import MoistureSolver
from .airflow import EMUAirflowModel


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


class CoupledSimulator:
    """Orchestrates the multi-physics coupling loop.

    This is the internal engine.  Users should use ``GP15Simulator``
    instead, which wraps this with geometry, control, and I/O.
    """

    def __init__(
        self,
        machine: MachineConfig,
        material: MaterialProperties,
        grid_shape: tuple,
        cell_sizes: tuple,
        device: str = "cpu",
    ):
        self._machine = machine
        self._material = material
        self._grid_shape = grid_shape
        self._cell_sizes = cell_sizes
        self._device = device
        self._time = 0.0

        # Cell volume
        dx, dy, dz = cell_sizes
        self._cell_vol = dx * dy * dz

        # Solvers
        self.rf = RFFieldSolver(grid_shape, cell_sizes, machine, device)
        self.thermal = ThermalSolver(grid_shape, cell_sizes, device)
        self.moisture = MoistureSolver(grid_shape, cell_sizes, device)
        self.airflow = EMUAirflowModel(machine)

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

        # Compute dry-basis density once (only in material cells)
        mat_mask = (self.cell_is_material == 1)
        M_wb = self.moisture.M[mat_mask]
        M_db = M_wb / np.maximum(1.0 - M_wb, 1e-6)
        self.rho_dry[mat_mask] = (
            mat.rho_solid * (1.0 - mat.bed_porosity)
        )
        self.rho_dry[~mat_mask] = 1.0  # dummy guard

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
            self.thermal.T = advect_material_np(
                self.thermal.T, v_belt, dx, dt,
                inlet_value=mat.initial_temperature_c,
            )
            self.moisture.M = advect_material_np(
                self.moisture.M, v_belt, dx, dt,
                inlet_value=mat.initial_moisture_wb,
            )
            # Zero moisture in non-material cells after advection
            self.moisture.M[~mat_mask] = 0.0

        # ── 2. RF FIELD ───────────────────────────────────────────────
        gap_m = recipe.electrode_gap_mm / 1000.0
        # Compute voltage from power-constrained approach (Phase 1):
        # Use a nominal voltage to start — we'll scale later.
        # For now, use a fraction of the anode voltage at idle.
        V_rf_kv = machine.anode_voltage_kv(
            machine.anode_current_no_load_a
        )
        self.rf.solve(
            electrode_gap_m=gap_m,
            voltage_kv=V_rf_kv,
            eps_real=self.eps_real,
            cell_is_material=self.cell_is_material,
            bed_depth_m=mat.bed_depth_m,
            belt_stack_m=machine.belt_stack_thickness_m,
        )

        # ── 3. HEATING ────────────────────────────────────────────────
        compute_power_density_np(
            self.rf.e_field_sq, self.eps_loss, out=self.P_v,
        )
        # Total RF power delivered [W]
        P_rf_w = float(np.sum(self.P_v[mat_mask]) * self._cell_vol)
        P_rf_kw = P_rf_w / 1000.0

        # ── 4. EVAPORATION (computed inside moisture.step) ────────────
        # (done in step 6)

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
        j_surface = self._find_bed_surface_j()
        if j_surface > 0:
            self.thermal.apply_convection_bc(
                j_surface=j_surface,
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

        # ── 8. CONTROLLER (placeholder for Phase 3) ───────────────────
        # Anode current from delivered power
        fraction = min(P_rf_kw / machine.max_rf_power_kw, 1.0) if P_rf_kw > 0 else 0.0
        I_a = (
            machine.anode_current_no_load_a
            + (machine.anode_current_full_load_a - machine.anode_current_no_load_a) * fraction
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

    def run(self, duration_s: float, dt: float, recipe: Recipe) -> PretreatmentResult:
        """Run the full simulation for *duration_s* seconds.

        Returns:
            A :class:`PretreatmentResult` with time-series and final fields.
        """
        n_steps = max(1, int(duration_s / dt))
        for _ in range(n_steps):
            self.step(dt, recipe)

        mat_mask = (self.cell_is_material == 1)
        M_final = self.moisture.M[mat_mask]
        T_final = self.thermal.T[mat_mask]

        # Throughput estimate
        belt_speed = recipe.belt_speed_m_per_min / 60.0  # m/s
        bed_cross = self._material.bed_depth_m * self._machine.belt_width_m
        rho_bulk = self._material.bulk_density(self._material.initial_moisture_wb)
        throughput_kg_s = rho_bulk * bed_cross * belt_speed
        throughput_kg_h = throughput_kg_s * 3600.0

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
