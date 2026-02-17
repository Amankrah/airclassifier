"""
Multi-Physics Coupling Orchestrator
====================================

Sequences the coupled physics solvers each timestep:

    1.  ADVECT     — shift T and M fields by belt velocity
    2.  RF FIELD   — solve Laplace for |E|^2
    3.  HEATING    — compute P_v from |E|^2 and eps''
    4.  EVAPORATION — (computed inside moisture.step)
    5.  THERMAL    — advance T with RF source and latent sink
    6.  MOISTURE   — advance M with diffusion and evaporation
    7.  PROPERTIES — update eps', eps'', rho, c_p, k, D_eff
    8.  CONTROLLER — PLC logic (MRH/MRL, gap, temperature control)
    9.  RECORD     — log outfeed state and KPIs
    10. PARTICLES  — Lagrangian tracers: belt transport + E-L field interpolation

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
from ..control.controller import GP15Controller, ControllerState, ControllerStatus
from ..geometry.electrode import ElectrodeGeometry, ElectrodeParams
from ..geometry.oven import OvenGeometry, OvenGeometryParams
from ..kernels.dielectric_heating import (
    TWO_PI_F_EPS0,
    compute_power_density_np,
    update_material_properties_np,
)
from ..kernels.transport import (
    advect_material_np,
    advect_material_tvd_np,
    ConveyorDriveController,
)
from .rf_field import RFFieldSolver
from .thermal import ThermalSolver
from .moisture import MoistureSolver
from .airflow import EMUAirflowModel

# ── GPU kernel imports (Warp / CUDA) ─────────────────────────────────
# When Warp is installed and CUDA is available, the simulation loop
# dispatches to these pre-compiled GPU kernels instead of NumPy.
_GPU_KERNELS_OK = False
try:
    import warp as wp
    from ..kernels.heat_transfer import (
        heat_conduction_step as _k_thermal,
        apply_convection_bc as _k_conv_bc,
    )
    from ..kernels.drying import moisture_step as _k_moisture
    from ..kernels.dielectric_heating import (
        _compute_power_density_kernel as _k_power_density,
        _update_properties_kernel as _k_update_props,
    )
    from ..kernels.transport import advect_material_wp_kernel as _k_advect
    _GPU_KERNELS_OK = True
except (ImportError, AttributeError):
    pass


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class StepState:
    """State snapshot after a single simulation step."""
    time_s: float = 0.0
    # Temperature
    T_mean_c: float = 0.0
    T_max_c: float = 0.0
    T_outfeed_c: float = 0.0
    T_outfeed_sensor_c: float = 0.0     # 75th percentile - matches PLC/strip sensors
    # Moisture
    M_mean_wb: float = 0.0
    M_min_wb: float = 0.0
    M_outfeed_wb: float = 0.0
    # RF system
    rf_power_kw: float = 0.0
    anode_current_a: float = 0.0
    # Controller
    electrode_gap_mm: float = 0.0
    belt_speed_m_per_min: float = 0.0
    controller_state: str = ""
    # Energy balance
    evap_power_kw: float = 0.0          # latent heat sink
    total_energy_kwh: float = 0.0       # cumulative RF energy
    water_removed_kg: float = 0.0       # cumulative water removed
    specific_energy_kwh_per_kg: float = 0.0
    # Electrode
    electrode_temperature_c: float = 0.0
    # Protein quality (Phase 4)
    protein_denaturation: float = 0.0   # fraction denatured [0-1]


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
    sensor_temperature_c: float = 0.0  # 75th percentile - matches PLC/strip sensors
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

    # When True, temperature_field is the peak-processing snapshot (at oven exit),
    # not the end-of-run state (Run#1 strip 82–93°C; avoids confusion with cooled bin).
    at_peak_processing_snapshot: bool = False


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
        self.thermal = ThermalSolver(
            grid_shape, cell_sizes, device,
            belt_stack_m=machine.belt_stack_thickness_m,
        )
        self.moisture = MoistureSolver(grid_shape, cell_sizes, device)
        self.airflow = EMUAirflowModel(machine)

        # Phase 3: controller
        self.controller = GP15Controller(machine)

        # Conveyor drive controller (motor, VFD ramp, kinematics)
        # Uses ConveyorBeltParams as single source of truth for all
        # roller radii, sprocket radii, and encoder PPR.
        from ..geometry.components.conveyor_belt import ConveyorBeltParams
        self.conveyor = ConveyorDriveController.from_params(
            ConveyorBeltParams(), machine,
        )

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

        # Lagrangian particle system (set by GP15Simulator after init)
        self._particles = None
        # Grid world origin (set by GP15Simulator for E-L interpolation)
        self._grid_origin = (0.0, 0.0, 0.0)

        # Batch mode tracking: when run_mass_kg > 0, the infeed stops
        # injecting fresh material once the batch is exhausted.  This
        # enables proper Ia drop → gap return behavior for batch runs.
        self._batch_exhausted = False
        self._batch_exhausted_time: float = 0.0  # time when batch exhausted

        # ── RF zone clearing tracking ──────────────────────────────────
        # Separate from batch_exhausted: tracks when the M=0 front has
        # actually reached the RF zone.  Used for controller gap-return
        # logic (should wait for RF zone to clear, not hopper to empty).
        #
        # The M=0 front advects from the hopper discharge to the RF zone
        # at belt speed.  Once it reaches the RF zone start, material
        # begins clearing and Ia starts dropping.
        self._rf_zone_clearing = False
        self._rf_zone_clear_time: float = 0.0  # time when RF zone fully clears

        # Track last valid outlet values for time series after batch exhausts.
        # When M=0 advects through the grid, we use these instead of the grid values.
        self._last_valid_M_outfeed = material.initial_moisture_wb
        self._last_valid_T_outfeed = material.initial_temperature_c
        self._last_valid_T_outfeed_sensor = material.initial_temperature_c  # 75th percentile

        # Snapshot of moisture at batch exhaustion time - represents steady-state
        # processing value BEFORE the M=0 front starts affecting outfeed readings.
        # This is the value we report as "outfeed moisture" for the run.
        self._moisture_at_batch_exhausted: float = material.initial_moisture_wb

        # Track last valid grid-wide temperatures for time series after belt clears.
        # During clearing, material remaining on belt cools in post-RF zone, but we
        # want to show representative PROCESSING temperatures, not clearing artifacts.
        self._last_valid_T_mean: float = material.initial_temperature_c
        self._last_valid_T_max: float = material.initial_temperature_c
        # Peak outfeed snapshot during processing (for cross-section when belt has cleared).
        # Run#1 strip: 82–93°C at oven exit; PLC Product_Temp peaks then cools after batch.
        # We store the T/M field snapshot when sensor T is highest so the heatmap matches.
        self._peak_outfeed_sensor_T: float = material.initial_temperature_c
        self._peak_outfeed_T_yz: Optional[np.ndarray] = None
        self._peak_outfeed_M_yz: Optional[np.ndarray] = None

        # KPI accumulators
        self._total_rf_energy_j = 0.0
        self._cumulative_water_removed_kg = 0.0  # integrated over time (batch-correct)
        self._history: List[StepState] = []

        # Generator operating point — tracks the previous timestep's
        # delivered RF power for the self-consistent Approach A solve.
        self._last_P_rf_kw: float = 0.0

        # ── Plate ammeter filtering (§8.4) ─────────────────────────────
        # The real plate ammeter has an RC time constant that smooths
        # rapid fluctuations.  Without this filter, the simulation shows
        # ~10x larger Ia oscillations than the real machine (Run#2 shows
        # ±0.02-0.05A stability, simulation was showing ±0.3-0.6A).
        self._Ia_filtered: float = machine.anode_current_no_load_a
        self._AMMETER_TAU: float = 0.5  # ammeter time constant [s]

        # ── Property smoothing (numerical stability) ───────────────────
        # Exponential smoothing on eps'' prevents discrete cell advection
        # from causing step-changes in the total dielectric load.  The
        # real material bed has thermal inertia that smooths property
        # changes; this filter approximates that effect.
        self._eps_loss_smoothed: np.ndarray | None = None
        self._PROPERTY_SMOOTH_TAU: float = 1.0  # property smoothing time constant [s]

        # ── RF power smoothing (generator inertia) ─────────────────────
        # The oscillator tank circuit has electrical inertia that prevents
        # instantaneous power changes.  This smooths the P_rf signal used
        # for voltage droop calculation, preventing feedback oscillations.
        self._P_rf_smoothed: float = 0.0
        self._RF_POWER_TAU: float = 0.3  # RF power smoothing time constant [s]

        # Cached bed surface index
        self._j_surface: int = 0

        # Lumped lower electrode / tray temperature model.
        # The aluminum trays on the conveyor absorb heat from the bed
        # through the PTFE belt and lose it via natural convection to
        # the oven air.  This prevents the Robin BC from draining
        # excessive heat to a fixed cold-temperature sink.
        self._T_electrode_c: float = material.initial_temperature_c
        self._ELECTRODE_MASS_KG: float = 15.0     # aluminum trays
        self._ELECTRODE_CP: float = 900.0          # J/(kg·K) aluminum
        self._ELECTRODE_H_LOSS: float = 5.0        # W/(m²·K) natural convection

        # ── GPU acceleration ─────────────────────────────────────────
        # When device="cuda" and Warp GPU kernels are available, the
        # physics loop (advection, thermal, moisture, power density,
        # material properties) runs on the GPU via Warp kernels.
        # Data is synced to CPU only for controller logic, KPI
        # recording, and boundary condition application.
        self._use_gpu = (device == "cuda" and _GPU_KERNELS_OK)
        self._wp_device = "cuda:0"
        self._gpu: dict = {}  # persistent GPU arrays, allocated in initialize()

    # ------------------------------------------------------------------
    # Initialisation / Reset
    # ------------------------------------------------------------------

    def update_parameters(
        self,
        coupling_factor: float | None = None,
        k_evap: float | None = None,
        gap_adjust_rate: float | None = None,
    ):
        """Update calibratable parameters on all sub-solvers.

        Single entry point that propagates to every component that
        caches material or machine properties.  Avoids the fragile
        pattern of poking multiple objects independently.

        Args:
            coupling_factor: Oscillator-to-electrode coupling.
            k_evap: Evaporation rate constant [1/(C*s)].
            gap_adjust_rate: MRH gap drive speed [mm/s].
        """
        if coupling_factor is not None:
            self._machine.oscillator_coupling_factor = coupling_factor
        if k_evap is not None:
            self._material.k_evap = k_evap
        if gap_adjust_rate is not None:
            self.controller.gap_adjust_rate_mm_s = gap_adjust_rate

    # ------------------------------------------------------------------
    # GPU array management
    # ------------------------------------------------------------------

    def _alloc_gpu_arrays(self):
        """Allocate persistent Warp GPU arrays (called once from initialize)."""
        if not self._use_gpu:
            return
        nx, ny, nz = self._grid_shape
        shape = (nx, ny, nz)
        d = self._wp_device
        f = wp.float32
        self._gpu = {
            'T':        wp.zeros(shape, dtype=f, device=d),
            'T_buf':    wp.zeros(shape, dtype=f, device=d),
            'M':        wp.zeros(shape, dtype=f, device=d),
            'M_buf':    wp.zeros(shape, dtype=f, device=d),
            'P_v':      wp.zeros(shape, dtype=f, device=d),
            'evap_rate': wp.zeros(shape, dtype=f, device=d),
            'e_field_sq': wp.zeros(shape, dtype=f, device=d),
            'eps_loss':  wp.zeros(shape, dtype=f, device=d),
            'eps_real':  wp.zeros(shape, dtype=f, device=d),
            'rho_cp':    wp.zeros(shape, dtype=f, device=d),
            'k_eff':     wp.zeros(shape, dtype=f, device=d),
            'rho_dry':   wp.zeros(shape, dtype=f, device=d),
            'cell_mask': wp.zeros(shape, dtype=wp.int32, device=d),
        }

    def _upload_to_gpu(self, name: str, np_array: np.ndarray):
        """Copy a NumPy array to the named persistent GPU array."""
        g = self._gpu[name]
        cpu = wp.array(np_array, dtype=g.dtype, device="cpu")
        wp.copy(g, cpu)

    def _sync_all_to_gpu(self):
        """Upload all physics fields from CPU to GPU."""
        self._upload_to_gpu('T', self.thermal.T)
        self._upload_to_gpu('M', self.moisture.M)
        self._upload_to_gpu('P_v', self.P_v)
        self._upload_to_gpu('evap_rate', self.moisture.evap_rate)
        self._upload_to_gpu('eps_loss', self.eps_loss)
        self._upload_to_gpu('eps_real', self.eps_real)
        self._upload_to_gpu('rho_cp', self.rho_cp)
        self._upload_to_gpu('k_eff', self.k_eff)
        self._upload_to_gpu('rho_dry', self.rho_dry)
        self._upload_to_gpu('cell_mask', self.cell_is_material)

    def _sync_physics_from_gpu(self):
        """Download key physics fields from GPU to CPU."""
        wp.synchronize()
        g = self._gpu
        self.thermal.T[:]          = g['T'].numpy()
        self.moisture.M[:]         = g['M'].numpy()
        self.P_v[:]                = g['P_v'].numpy()
        self.moisture.evap_rate[:] = g['evap_rate'].numpy()
        self.eps_loss[:]           = g['eps_loss'].numpy()
        self.eps_real[:]           = g['eps_real'].numpy()
        self.rho_cp[:]             = g['rho_cp'].numpy()
        self.k_eff[:]              = g['k_eff'].numpy()

    def _apply_thermal_bcs_cpu(self, dt: float):
        """Apply thermal boundary conditions on CPU after GPU kernel.

        The GPU thermal kernel handles the interior FDM update but
        leaves boundary cells as copies.  This method applies the
        full set of BCs from the ThermalSolver (infeed Dirichlet,
        outfeed Neumann, belt adiabatic, bottom Robin, top Neumann).
        """
        T = self.thermal.T
        mat = self._material
        T_inlet = mat.initial_temperature_c
        dy = self._cell_sizes[1]

        # NaN / Inf guard and physical clamp
        np.nan_to_num(T, copy=False, nan=T_inlet, posinf=200.0, neginf=T_inlet)
        np.clip(T, 0.0, 200.0, out=T)

        # Infeed (x=0): Dirichlet
        T[0, :, :] = T_inlet
        # Outfeed (x=-1): Neumann
        T[-1, :, :] = T[-2, :, :]
        # Belt edges (z): adiabatic
        T[:, :, 0] = T[:, :, 1]
        T[:, :, -1] = T[:, :, -2]

        # Bottom (j=0): Full energy balance for boundary cell.
        # The GPU kernel skips boundary cells (T_new = T at j<=0),
        # so j=0 needs the complete update here:
        #   1. Conduction from j=1 (one-sided difference)
        #   2. Robin BC: heat loss through PTFE belt to electrode
        #   3. RF source (P_v)
        #   4. Latent heat sink (evaporation)
        T_elec = self._T_electrode_c
        mat_j0 = (self.cell_is_material[:, 0, :] == 1)
        if np.any(mat_j0):
            K_BELT = self.thermal._K_BELT
            D_BELT = self.thermal._D_BELT
            h_contact = K_BELT / max(D_BELT, 1e-6)
            rc_0 = np.maximum(self.rho_cp[:, 0, :], 1.0)
            # Conduction from j=1 (one-sided, interior neighbor)
            k_y_half = 0.5 * (self.k_eff[:, 0, :] + self.k_eff[:, 1, :])
            cond_j1 = k_y_half * (T[:, 1, :] - T[:, 0, :]) / (dy * dy)
            # Robin BC flux (belt contact to electrode)
            robin = h_contact * (T[:, 0, :] - T_elec) / (dy * 0.5)
            # RF source and latent sink at j=0
            src_j0 = self.P_v[:, 0, :]
            snk_j0 = 2.26e6 * self.moisture.evap_rate[:, 0, :]
            # Forward Euler update
            T[:, 0, :] = np.where(
                mat_j0,
                T[:, 0, :] + dt / rc_0 * (cond_j1 - robin + src_j0 - snk_j0),
                T_elec,
            )
        else:
            T[:, 0, :] = T_elec

        # Top (j=-1): Neumann
        T[:, -1, :] = T[:, -2, :]

    def _apply_moisture_bcs_cpu(self):
        """Apply moisture boundary conditions on CPU after GPU kernel.

        The GPU moisture kernel zeros non-material cells and handles
        interior diffusion + evaporation.  This method adds the BC
        set from the MoistureSolver (infeed, outfeed, edges).

        For batch runs, once the batch is exhausted the infeed injects
        M=0 (empty cells) instead of fresh wet material.
        """
        M = self.moisture.M
        mat = self._material
        cell_mask = self.cell_is_material

        # NaN guard
        np.nan_to_num(M, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(M, 0.0, 1.0, out=M)

        # Non-material cells = 0
        M[cell_mask != 1] = 0.0

        # Infeed: batch-aware moisture injection
        if self._batch_exhausted:
            # Batch exhausted: inject empty (M=0)
            M[0, :, :] = 0.0
        else:
            # Normal: inject fresh wet material
            M[0, :, :] = np.where(cell_mask[0, :, :] == 1,
                                   mat.initial_moisture_wb, 0.0)
        # Outfeed
        M[-1, :, :] = M[-2, :, :]
        # Z edges
        M[:, :, 0] = M[:, :, 1]
        M[:, :, -1] = M[:, :, -2]
        # Y edges
        M[:, 0, :] = M[:, 1, :]
        M[:, -1, :] = M[:, -2, :]
        # Re-enforce non-material = 0
        M[cell_mask != 1] = 0.0

    def reset(self):
        """Reset all fields and accumulators for a fresh run.

        Re-uses existing array allocations (no re-construction).
        Called by the calibration optimizer to avoid creating a new
        GP15Simulator for every parameter evaluation.
        """
        mat = self._material
        self.thermal.initialize(mat.initial_temperature_c)
        self.moisture.initialize(mat.initial_moisture_wb)
        self.moisture.evap_rate[:] = 0.0
        self.P_v[:] = 0.0
        self._time = 0.0
        self._total_rf_energy_j = 0.0
        self._cumulative_water_removed_kg = 0.0
        self._last_P_rf_kw = 0.0
        self._Ia_filtered = self._machine.anode_current_no_load_a
        self._P_rf_smoothed = 0.0
        self._history.clear()
        self._update_properties()
        # Initialize smoothed eps_loss from freshly computed values
        if self._eps_loss_smoothed is not None:
            self._eps_loss_smoothed[:] = self.eps_loss

        # Re-stamp rho_dry from current material (in case porosity
        # or rho_solid changed between evaluations)
        mat_mask = (self.cell_is_material == 1)
        self.rho_dry[mat_mask] = mat.rho_solid * (1.0 - mat.bed_porosity)
        self.rho_dry[~mat_mask] = 1.0

        # Reset controller
        self.controller.status = ControllerStatus()
        self.controller.safety.reset()
        self.controller._sim_time = 0.0
        self.controller._batch_exhausted = False

        # Reset conveyor
        self.conveyor.state.belt_position_m = 0.0
        self.conveyor.state.elapsed_time_s = 0.0

        # Reset batch tracking
        self._batch_exhausted = False
        self._batch_exhausted_time = 0.0
        self._rf_zone_clearing = False
        self._rf_zone_clear_time = 0.0
        self._last_valid_M_outfeed = self._material.initial_moisture_wb
        self._last_valid_T_outfeed = self._material.initial_temperature_c
        self._last_valid_T_outfeed_sensor = self._material.initial_temperature_c
        self._moisture_at_batch_exhausted = self._material.initial_moisture_wb
        self._last_valid_T_mean = self._material.initial_temperature_c
        self._last_valid_T_max = self._material.initial_temperature_c
        self._peak_outfeed_sensor_T = self._material.initial_temperature_c
        self._peak_outfeed_T_yz = None
        self._peak_outfeed_M_yz = None

        # Reset particle system mass tracking.
        # Without this, dispatched_mass_kg retains the previous run's
        # value, causing batch_exhausted to trigger immediately at t=0
        # on subsequent evaluations (e.g. during calibration).
        if self._particles is not None:
            self._particles._dispatched_mass_kg = 0.0
            self._particles._total_collected_kg = 0.0
            self._particles._initial_belt_mass_kg = 0.0
            self._particles._batch_exhausted = False
            self._particles._spawn_accumulator = 0.0
            self._particles._alive_count = 0
            self._particles.state[:] = self._particles._STATE_DEAD
            self._particles.pos[:] = 0.0
            self._particles.vel[:] = 0.0
            self._particles.age[:] = 0.0
            # Phase 4: reset Biot + denaturation state
            self._particles.T_core[:] = self._material.initial_temperature_c
            self._particles.vicilin_native[:] = 1.0
            self._particles.legumin_native[:] = 1.0
            # Re-initialize particle distribution (hopper fill / belt prefill)
            self._particles._init_particles()

        # Re-sync GPU arrays with reset state
        if self._use_gpu and self._gpu:
            self._sync_all_to_gpu()

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

        # For batch runs the oven starts EMPTY — material must travel
        # from hopper to the RF zone before the grid sees any moisture.
        # Initialise M = 0 and let advection fill naturally from x = 0.
        # The infeed boundary condition handles the transit delay
        # (see step() → batch-aware infeed).
        if self._particles is not None and self._particles.cfg.run_mass_kg > 0:
            self.moisture.initialize(0.0)
        else:
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

        # Initialize smoothed eps_loss for property filtering
        # This smooths out discrete advection effects that cause
        # step-changes in the total dielectric load.
        self._eps_loss_smoothed = self.eps_loss.copy()

        # Compute dry-basis density (only in material cells)
        mat_mask = (self.cell_is_material == 1)
        self.rho_dry[mat_mask] = mat.rho_solid * (1.0 - mat.bed_porosity)
        self.rho_dry[~mat_mask] = 1.0  # dummy guard

        # Cache bed surface index
        self._j_surface = self._find_bed_surface_j()

        # Reset electrode temperature to ambient
        self._T_electrode_c = mat.initial_temperature_c

        # Allocate GPU arrays and upload initial state
        if self._use_gpu:
            self._alloc_gpu_arrays()
            self._sync_all_to_gpu()

    # ------------------------------------------------------------------
    # Adaptive timestep
    # ------------------------------------------------------------------

    def compute_stable_dt(self, recipe: Recipe) -> float:
        """Compute a stable timestep from CFL and Courant constraints.

        CFL (thermal):  dt < factor * dmin^2 * rho_cp_min / k_max
        Courant (advection):  dt < 0.9 * dx / v_belt

        Uses the *actual* belt speed from the conveyor drive controller
        (which includes VFD ramp), falling back to the recipe setpoint
        for the initial estimate before the conveyor has started.

        The CFL must consider ALL cell types (material, air, belt)
        because the thermal solver updates every cell.  Air cells have
        much smaller rho*c_p than material, requiring smaller timesteps
        to remain stable.

        Returns:
            The minimum of the two constraints [s].
        """
        dx, dy, dz = self._cell_sizes

        # CFL from thermal diffusivity — worst case across ALL cells
        # (the thermal solver updates air cells too)
        active = (self.rho_cp > 1.0)  # any cell with valid properties
        if active.any():
            k_max = float(np.max(self.k_eff[active]))
            rho_cp_min = float(np.min(self.rho_cp[active]))
        else:
            k_max = 0.2
            rho_cp_min = 1e5
        dt_cfl = self.thermal.get_cfl_dt(k_max, rho_cp_min)

        # Courant from advection — use actual belt speed (with ramp)
        v_belt = self.conveyor.state.belt_speed_m_per_s
        if v_belt <= 0:
            # Conveyor not yet started: use recipe setpoint for estimate
            v_belt = recipe.belt_speed_m_per_min / 60.0
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
        When ``device="cuda"`` and Warp kernels are available, steps
        1 and 3–7 run on the GPU; steps 2 (RF field) and 8–10
        (controller, recording, particles) remain on the CPU.
        """
        dx, dy, dz = self._cell_sizes
        nx, ny, nz = self._grid_shape
        mat = self._material
        machine = self._machine
        mat_mask = (self.cell_is_material == 1)

        # ── 1. ADVECT ─────────────────────────────────────────────────
        # Update conveyor drive: set speed from recipe, step kinematics
        self.conveyor.set_speed(recipe.belt_speed_m_per_min)
        if not self.conveyor.state.running:
            self.conveyor.start()
        self.conveyor.step(dt)
        v_belt = self.conveyor.state.belt_speed_m_per_s  # actual (ramped)

        # ── Batch-aware infeed boundary condition ─────────────────────
        # For finite-mass runs (run_mass_kg > 0), once the particle system
        # has dispatched all material, the infeed should inject "empty"
        # cells (M≈0) instead of fresh wet material.  This causes:
        #   - ε'' → low as empty cells advect through RF zone
        #   - P_rf → 0 as material clears
        #   - Ia → idle (0.4A) matching real machine behavior
        #   - Gap return when MRL < Ia < MRH (normal band)
        #
        # Without this, the Eulerian grid continuously injects fresh
        # material and Ia never drops to idle, preventing gap return.
        if self._particles is not None:
            cfg = self._particles.cfg
            if cfg.run_mass_kg > 0:
                # Check if batch is exhausted (hopper empty)
                dispatched = self._particles.dispatched_mass_kg
                if dispatched >= cfg.run_mass_kg and not self._batch_exhausted:
                    self._batch_exhausted = True
                    self._batch_exhausted_time = self._time
                    # Snapshot current outfeed moisture - this is the steady-state
                    # value BEFORE the M=0 front starts affecting outfeed.
                    self._moisture_at_batch_exhausted = self._last_valid_M_outfeed
                    # Signal particles: stop moisture interpolation from grid
                    # (grid now has M=0 behind the material)
                    self._particles.set_batch_exhausted(True)

                # ── RF zone clearing detection ─────────────────────────────
                # The M=0 front advects from hopper discharge at belt speed.
                # We signal the controller only when M=0 has reached the RF
                # zone END (oven has cleared), not when it reaches the RF
                # zone start (material still in oven).
                #
                # So ctrl_batch = True only after the oven is empty; gap
                # return (Ia < MRL → return to setpoint) then correctly
                # means "belt clearing" with no material left in the RF zone.
                if self._batch_exhausted and not self._rf_zone_clearing:
                    # Time for M=0 front to travel from hopper to OVEN EXIT.
                    # Material is "in the GP-15" until it exits the outfeed
                    # attenuation duct (Manual p.54), not just the anode plate.
                    hopper_to_oven_end = cfg.oven_x_end - cfg.spawn_x
                    if v_belt > 0 and hopper_to_oven_end > 0:
                        advect_time_to_oven_end = hopper_to_oven_end / v_belt
                        time_since_exhaust = self._time - self._batch_exhausted_time
                        if time_since_exhaust >= advect_time_to_oven_end:
                            self._rf_zone_clearing = True
                            if self._enable_controller:
                                self.controller.set_batch_exhausted(True)

        # Determine infeed values based on batch state AND belt transit.
        #
        # The belt transit from hopper (spawn_x) to grid inlet
        # (rf_zone_x_start) applies SYMMETRICALLY to both:
        #   - Initial fill: material hasn't reached the grid yet
        #   - Batch exhaust: M=0 front hasn't reached the grid yet
        #
        # When the hopper empties, material between hopper and grid
        # inlet is still wet.  The M=0 front reaches grid x=0 only
        # after transit_time = (grid_origin_x - spawn_x) / v_belt.
        T_inlet = mat.initial_temperature_c
        if self._particles is not None and self._particles.cfg.run_mass_kg > 0:
            hopper_to_grid = self._grid_origin[0] - self._particles.cfg.spawn_x
            transit_time = hopper_to_grid / v_belt if (hopper_to_grid > 0 and v_belt > 0) else 0.0

            if self._batch_exhausted:
                # M=0 front left the hopper at _batch_exhausted_time.
                # It reaches grid x=0 after transit_time.
                time_since_exhaust = self._time - self._batch_exhausted_time
                if time_since_exhaust >= transit_time:
                    M_inlet = 0.0  # M=0 front has reached the grid
                else:
                    M_inlet = mat.initial_moisture_wb  # still wet belt
            elif self._time < transit_time:
                M_inlet = 0.0  # initial fill: material not here yet
            else:
                M_inlet = mat.initial_moisture_wb  # normal operation
        else:
            # Continuous mode or no particles — inject fresh material
            M_inlet = mat.initial_moisture_wb

        if v_belt > 0 and v_belt * dt / dx < 1.0:
            if self._use_gpu:
                # GPU upwind advection (Warp kernel)
                C = float(v_belt * dt / dx)
                g = self._gpu
                wp.launch(_k_advect, dim=(nx, ny, nz),
                          inputs=[g['T'], g['T_buf'], C,
                                  float(T_inlet),
                                  nx, ny, nz],
                          device=self._wp_device)
                g['T'], g['T_buf'] = g['T_buf'], g['T']
                wp.launch(_k_advect, dim=(nx, ny, nz),
                          inputs=[g['M'], g['M_buf'], C,
                                  float(M_inlet),
                                  nx, ny, nz],
                          device=self._wp_device)
                g['M'], g['M_buf'] = g['M_buf'], g['M']
            else:
                _advect = advect_material_tvd_np if self._use_tvd else advect_material_np
                self.thermal.T = _advect(
                    self.thermal.T, v_belt, dx, dt,
                    inlet_value=T_inlet,
                )
                self.moisture.M = _advect(
                    self.moisture.M, v_belt, dx, dt,
                    inlet_value=M_inlet,
                )
                # Zero moisture in non-material cells after advection
                self.moisture.M[~mat_mask] = 0.0

        # ── 2. RF FIELD ───────────────────────────────────────────────
        # Use the controller's current gap (which may differ from the
        # recipe setpoint if MRH has opened the gap).  Run#1 PLC data
        # shows gap opening from 75mm to 87mm under MRH control.
        if self._enable_controller:
            gap_m = self.controller.status.electrode_gap_mm / 1000.0
        else:
            gap_m = recipe.electrode_gap_mm / 1000.0

        # Update fringe correction for current gap (Phase 3)
        if self._enable_corrections:
            elec = ElectrodeGeometry(ElectrodeParams.from_machine(machine))
            self._fringe_correction = elec.get_fringe_field_correction(
                self._grid_shape, gap_m,
            )

        if self._power_constrained and self._target_power_kw:
            # Explicit Approach A: user-specified target power
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
            V_rf_kv = self._compute_electrode_voltage(machine)
            self.rf.solve_fdm(
                gap_m, V_rf_kv, self.eps_real,
                bed_depth_m=mat.bed_depth_m,
                belt_stack_m=machine.belt_stack_thickness_m,
            )
        else:
            # ── Approach B — voltage-driven (§4.1.3) ─────────────
            #
            # The generator produces an RF voltage at the electrodes
            # determined by the anode voltage and the oscillator
            # coupling factor (tank circuit, trombocones, feed strips):
            #
            #     V_rf = V_anode * oscillator_coupling_factor
            #
            # The material absorbs whatever power results from this
            # voltage applied through the series-capacitor model.
            # This naturally self-regulates:
            #
            #   - Thin bed + large gap → most voltage drops across
            #     the air gap → low P_rf (matches Run#1: Ia=0.29 A)
            #   - Thick bed + small gap → more voltage in material
            #     → high P_rf (matches full-load: Ia=2.58 A)
            #
            # The voltage droop under load is included: as more power
            # is absorbed, V_anode drops (linear droop model from
            # Manual Appendix E test report), reducing V_rf.
            V_rf_kv = self._compute_electrode_voltage(machine)
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

        # ── Generator power limiting ─────────────────────────────────
        # The triode oscillator physically cannot deliver more than its
        # rated maximum (max_rf_power_kw).  When the voltage-driven model
        # (Approach B) produces a theoretical P_rf exceeding the generator
        # capacity, the oscillator saturates.  Since the Laplace equation
        # is linear, scaling |E|^2 by (P_max / P_theoretical) correctly
        # reduces the delivered power without re-solving.
        #
        # The anode current Ia is computed from the UNCLAMPED demand:
        # the plate ammeter reflects the oscillator stress, which is what
        # the MRH relay sees.  But Ia is capped at the tube's physical
        # limit (~3.0 A for the YL-1057 triode).
        P_rf_max_kw = machine.max_rf_power_kw

        if self._use_gpu:
            g = self._gpu
            # Upload the (possibly corrected) e_field to GPU
            self._upload_to_gpu('e_field_sq', e_field.astype(np.float32))

            # Compute P_v on GPU — uses the GPU's eps_loss which is
            # up-to-date from the previous step's property update kernel.
            # The CPU's self.eps_loss is stale on the GPU path.
            wp.launch(_k_power_density, dim=(nx, ny, nz),
                      inputs=[g['e_field_sq'], g['eps_loss'], g['P_v'],
                              TWO_PI_F_EPS0, nx, ny, nz],
                      device=self._wp_device)

            # Download P_v for power limiting sum (CPU reduction)
            wp.synchronize()
            self.P_v[:] = g['P_v'].numpy()
            P_rf_w_theoretical = float(np.sum(self.P_v[mat_mask]) * self._cell_vol)
            P_rf_kw_theoretical = P_rf_w_theoretical / 1000.0

            if P_rf_kw_theoretical > P_rf_max_kw and P_rf_kw_theoretical > 0:
                scale = P_rf_max_kw / P_rf_kw_theoretical
                self.P_v *= scale
                # Re-upload the scaled P_v for the thermal kernel
                self._upload_to_gpu('P_v', self.P_v)
                P_rf_w = P_rf_max_kw * 1000.0
                P_rf_kw = P_rf_max_kw
            else:
                P_rf_w = P_rf_w_theoretical
                P_rf_kw = P_rf_kw_theoretical
        else:
            # CPU path: compute P_v from NumPy arrays
            compute_power_density_np(e_field, self.eps_loss, out=self.P_v)
            P_rf_w_theoretical = float(np.sum(self.P_v[mat_mask]) * self._cell_vol)
            P_rf_kw_theoretical = P_rf_w_theoretical / 1000.0

            if P_rf_kw_theoretical > P_rf_max_kw and P_rf_kw_theoretical > 0:
                scale = P_rf_max_kw / P_rf_kw_theoretical
                self.P_v *= scale
                P_rf_w = P_rf_max_kw * 1000.0
                P_rf_kw = P_rf_max_kw
            else:
                P_rf_w = P_rf_w_theoretical
                P_rf_kw = P_rf_kw_theoretical

        if self._use_gpu:
            g = self._gpu

            # ── GPU BATCH: thermal + convection + moisture + properties
            # Launch all kernels without sync — Warp serialises them on
            # the same CUDA stream, so ordering is guaranteed.  We sync
            # once after the batch, halving CPU/GPU round-trips.

            # 5. Thermal (reads P_v produced above — same stream, ordered)
            wp.launch(_k_thermal, dim=(nx, ny, nz),
                      inputs=[g['T'], g['T_buf'], g['P_v'], g['evap_rate'],
                              g['rho_cp'], g['k_eff'],
                              2.26e6, dx, dy, dz, dt, nx, ny, nz],
                      device=self._wp_device)
            g['T'], g['T_buf'] = g['T_buf'], g['T']

            # Convective BC on GPU
            self.airflow.update(recipe, dt)
            if self._j_surface > 0:
                wp.launch(_k_conv_bc, dim=(nx, nz),
                          inputs=[g['T'], self._j_surface,
                                  float(self.airflow.state.convective_htc_w_per_m2k),
                                  float(self.airflow.state.air_temperature_c),
                                  g['rho_cp'], dy, dt, nx, nz],
                          device=self._wp_device)

            # 6. Moisture (reads T produced above — same stream, ordered)
            wp.launch(_k_moisture, dim=(nx, ny, nz),
                      inputs=[g['M'], g['M_buf'], g['T'], g['evap_rate'],
                              g['cell_mask'], g['rho_dry'],
                              float(mat.D_eff_D0), float(mat.D_eff_Ea),
                              8.314, float(mat.k_evap),
                              float(mat.T_evap_threshold_c),
                              dx, dy, dz, dt, nx, ny, nz],
                      device=self._wp_device)
            g['M'], g['M_buf'] = g['M_buf'], g['M']

            # 7. Properties (reads T, M produced above — same stream)
            a = mat.dielectric_loss_coeffs
            b = mat.dielectric_const_coeffs
            wp.launch(_k_update_props, dim=(nx, ny, nz),
                      inputs=[g['T'], g['M'], g['cell_mask'],
                              g['eps_loss'], g['eps_real'],
                              g['rho_cp'], g['k_eff'],
                              float(a[0]), float(a[1]), float(a[2]),
                              float(a[3]), float(a[4]),
                              float(b[0]), float(b[1]), float(b[2]),
                              float(mat.c_p_dry), float(mat.c_p_water),
                              float(mat.k_dry), float(mat.k_moisture_beta),
                              float(mat.rho_solid), float(mat.bed_porosity),
                              float(mat.k_dispersion),
                              nx, ny, nz],
                      device=self._wp_device)

            # ── SINGLE SYNC: download all results at once ────────────
            self._sync_physics_from_gpu()

            # Apply BCs on CPU (boundary rows only — cheap)
            self._apply_thermal_bcs_cpu(dt)
            self._apply_moisture_bcs_cpu()

            # ── Property smoothing (GPU path) ─────────────────────────
            # Same smoothing as CPU path: prevents discrete cell advection
            # from causing step-changes in total dielectric load.
            if self._eps_loss_smoothed is not None:
                alpha_eps = dt / (self._PROPERTY_SMOOTH_TAU + dt)
                self._eps_loss_smoothed[:] = (
                    alpha_eps * self.eps_loss
                    + (1.0 - alpha_eps) * self._eps_loss_smoothed
                )
                self.eps_loss[:] = self._eps_loss_smoothed

            # Re-upload corrected T and M (boundary rows changed)
            self._upload_to_gpu('T', self.thermal.T)
            self._upload_to_gpu('M', self.moisture.M)
            # Also upload smoothed eps_loss for next step
            self._upload_to_gpu('eps_loss', self.eps_loss)
        # (else: CPU path — P_v already computed above with generator limiting)

        # ── 4. EVAPORATION (computed inside moisture.step) ────────────

        # ── 5–7. THERMAL + MOISTURE + PROPERTIES (CPU path) ──────────
        if not self._use_gpu:
            self.thermal.step(
                dt=dt,
                P_v=self.P_v,
                evap_rate=self.moisture.evap_rate,
                rho_cp=self.rho_cp,
                k_eff=self.k_eff,
                cell_is_material=self.cell_is_material,
                T_inlet_c=mat.initial_temperature_c,
                T_electrode_c=self._T_electrode_c,
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

            # Batch-aware moisture inlet: use M_inlet computed earlier
            # (0 if batch exhausted, initial_moisture_wb otherwise)
            self.moisture.step(
                dt=dt,
                T=self.thermal.T,
                cell_is_material=self.cell_is_material,
                rho_dry=self.rho_dry,
                D0=mat.D_eff_D0,
                Ea=mat.D_eff_Ea,
                k_evap=mat.k_evap,
                T_threshold=mat.T_evap_threshold_c,
                M_inlet_wb=M_inlet,
            )

            self._update_properties()

            # ── 7b. PROPERTY SMOOTHING (numerical stability) ───────────
            # Apply exponential smoothing to eps'' to prevent discrete cell
            # advection from causing step-changes in the total dielectric
            # load.  The real material bed has thermal inertia that smooths
            # property changes; this filter approximates that effect.
            #
            # Run#2 shows Ia varying by ±0.02-0.05A; without this the
            # simulation showed ±0.3-0.6A oscillations caused by discrete
            # cells entering/leaving the RF zone.
            if self._eps_loss_smoothed is not None:
                alpha_eps = dt / (self._PROPERTY_SMOOTH_TAU + dt)
                self._eps_loss_smoothed[:] = (
                    alpha_eps * self.eps_loss
                    + (1.0 - alpha_eps) * self._eps_loss_smoothed
                )
                # Use smoothed eps_loss for power calculations
                self.eps_loss[:] = self._eps_loss_smoothed

        # ── 5b. ELECTRODE TEMPERATURE ────────────────────────────────
        # Lumped thermal model for the lower electrode / aluminum trays.
        # Heat flows from bed bottom through the PTFE belt into the
        # trays, and the trays lose heat via natural convection.
        #   Q_in  = h_contact * A * (T_bed_bottom_avg - T_electrode)
        #   Q_out = h_loss * A * (T_electrode - T_ambient)
        #   dT = (Q_in - Q_out) * dt / (m * c_p)
        T_bed_j0 = self.thermal.T[:, 0, :]
        mat_j0 = (self.cell_is_material[:, 0, :] == 1)
        if np.any(mat_j0):
            T_bed_avg = float(np.mean(T_bed_j0[mat_j0]))
        else:
            T_bed_avg = self._T_electrode_c
        h_contact = self.thermal._K_BELT / max(self.thermal._D_BELT, 1e-6)
        belt_area = self._machine.oven_length_m * self._machine.belt_width_m
        Q_in = h_contact * belt_area * (T_bed_avg - self._T_electrode_c)
        Q_out = self._ELECTRODE_H_LOSS * belt_area * (
            self._T_electrode_c - mat.initial_temperature_c
        )
        dT_elec = (Q_in - Q_out) * dt / (
            self._ELECTRODE_MASS_KG * self._ELECTRODE_CP
        )
        self._T_electrode_c += dT_elec

        # ── 8. CONTROLLER ──────────────────────────────────────────────
        # Anode current from THEORETICAL demand (generator model §8.4).
        # The plate ammeter reflects the oscillator's loading, which is
        # what the MRH relay sees.  Use the UNCLAMPED theoretical P_rf
        # (before generator limiting) so the MRH relay detects overload.
        #
        # The fraction interpolates linearly between no-load and full-load
        # operating points from the test report:
        #   fraction = 0 → Ia = 0.4 A (no-load)
        #   fraction = 1 → Ia = 2.58 A (full-load, P_rf = 15 kW)
        #
        # fraction uses P_rf directly (NOT P_gen = P_rf/eta) because the
        # test-report endpoints already encode the efficiency:
        #   no-load:  P_rf=0, Ia=0.4A
        #   full-load: P_rf=15kW, Ia=2.58A
        # Dividing by efficiency would double-count it, inflating Ia
        # by 1/eta ≈ 1.79x.
        #
        # Cap Ia at the tube's maximum plate current (~3.0 A for the
        # YL-1057 triode) — the tube physically can't draw more.
        fraction = P_rf_kw_theoretical / max(machine.max_rf_power_kw, 0.01) if P_rf_kw_theoretical > 0 else 0.0
        I_a_instant = (
            machine.anode_current_no_load_a
            + (machine.anode_current_full_load_a - machine.anode_current_no_load_a) * fraction
        )
        I_a_instant = min(I_a_instant, 3.0)  # tube maximum plate current

        # ── Plate ammeter filtering (matches real instrument dynamics) ──
        # The real plate ammeter has an RC time constant (~0.5s) that
        # smooths rapid fluctuations.  Run#2 shows Ia varying by only
        # ±0.02-0.05A during steady operation; without this filter the
        # simulation showed ±0.3-0.6A oscillations (10x too large).
        #
        # Exponential moving average: α = dt / (τ + dt)
        alpha_Ia = dt / (self._AMMETER_TAU + dt)
        self._Ia_filtered = alpha_Ia * I_a_instant + (1.0 - alpha_Ia) * self._Ia_filtered
        I_a = self._Ia_filtered

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

        # ── RF power smoothing for generator feedback ──────────────────
        # The oscillator tank circuit has electrical inertia that prevents
        # instantaneous response to load changes.  Using the CURRENT P_rf
        # (instead of previous timestep) eliminates the one-step lag that
        # was causing feedback oscillations.  The smoothing filter then
        # models the tank circuit's response time.
        #
        # This fixes the voltage droop feedback lag: V_rf now responds to
        # the current load state rather than being one timestep behind.
        alpha_rf = dt / (self._RF_POWER_TAU + dt)
        self._P_rf_smoothed = alpha_rf * P_rf_kw + (1.0 - alpha_rf) * self._P_rf_smoothed
        self._last_P_rf_kw = self._P_rf_smoothed  # smoothed value for generator model

        T_mat = self.thermal.T[mat_mask]
        M_mat = self.moisture.M[mat_mask]
        T_mean = float(np.mean(T_mat)) if T_mat.size else mat.initial_temperature_c
        T_max = float(np.max(T_mat)) if T_mat.size else mat.initial_temperature_c
        M_mean = float(np.mean(M_mat)) if M_mat.size else mat.initial_moisture_wb
        M_min = float(np.min(M_mat)) if M_mat.size else 0.0

        # Outfeed cross-section: last X-slice (index -1). Neumann BC sets T[-1]=T[-2],
        # so this slice is the temperature at the oven exit (no gradient at outlet).
        outfeed_mat = (self.cell_is_material[-1, :, :] == 1)
        T_out_cells = self.thermal.T[-1, :, :][outfeed_mat]
        M_out_cells = self.moisture.M[-1, :, :][outfeed_mat]

        # For time series: detect when the M=0 clearing front has reached
        # the outfeed grid slice and freeze to last-valid values.
        #
        # Approach: compute advection time from hopper discharge to the
        # outfeed.  Once batch_exhausted AND elapsed time >= advection time,
        # the M=0 front has (physically) arrived at the outlet slice.  This
        # is *predictive* — we freeze at the exact arrival instant, before
        # M=0 cells mix into the slice average and create an artificial dip.
        #
        # Previous approach used a 50% moisture threshold which was
        # *reactive* and triggered after the clearing artifact had already
        # contaminated the slice average for several timesteps.
        outfeed_clear = False
        if self._batch_exhausted and v_belt > 0:
            # Distance from hopper discharge to outfeed (world coordinates),
            # consistent with RF-zone clearing logic (§ batch-aware infeed).
            if self._particles is not None:
                outfeed_world_x = self._grid_origin[0] + machine.oven_length_m
                hopper_to_outfeed = outfeed_world_x - self._particles.cfg.spawn_x
            else:
                hopper_to_outfeed = machine.oven_length_m
            if hopper_to_outfeed > 0:
                advect_time = hopper_to_outfeed / v_belt
                time_since_exhaust = self._time - self._batch_exhausted_time
                if time_since_exhaust >= advect_time:
                    outfeed_clear = True

        # Check if outfeed has meaningful moisture content.
        # The timing-based outfeed_clear can be off due to geometry approximations.
        # Add a reactive threshold: if mean moisture < 50% of initial, the M=0 front
        # has arrived regardless of the timing calculation.
        M_outfeed_mean = float(np.mean(M_out_cells)) if M_out_cells.size > 0 else 0.0
        M_clearing_threshold = mat.initial_moisture_wb * 0.5
        outfeed_moisture_low = (M_outfeed_mean < M_clearing_threshold)

        has_material = (not outfeed_clear) and (M_out_cells.size > 0) and (not outfeed_moisture_low)

        if has_material:
            T_outfeed = float(np.mean(T_out_cells))
            M_outfeed = M_outfeed_mean
            # Update last valid values for use after belt clears
            self._last_valid_T_outfeed = T_outfeed
            self._last_valid_M_outfeed = M_outfeed
            # Also update grid-wide T values (for time series range plot)
            self._last_valid_T_mean = T_mean
            self._last_valid_T_max = T_max
        else:
            # M=0 front at outlet (belt clearing) — hold last valid values
            T_outfeed = self._last_valid_T_outfeed
            M_outfeed = self._last_valid_M_outfeed
            # Use held values for grid temperatures (prevents range dropping during clearing)
            T_mean = self._last_valid_T_mean
            T_max = self._last_valid_T_max

        # Sensor-comparable temperature: The PLC's Product_Temp sensor is an
        # IR pyrometer or thermocouple that measures surface/exposed temperatures,
        # NOT the bulk volume average.  Temperature strips on peas similarly
        # measure the surface of individual particles.
        #
        # In a packed bed with vertical temperature gradient (RF heating +
        # convective cooling), surface-based sensors see a distribution biased
        # toward exposed/hot regions.  The 75th percentile of outfeed cell
        # temperatures better represents what these sensors measure than the
        # volume mean.
        #
        # Validation (Run#2): strips showed 77-82°C, simulation mean was 50.6°C,
        # but simulation max was 73.4°C.  The 75th percentile gives a value
        # between mean and max that matches sensor physics.
        if has_material and T_out_cells.size > 0:
            T_outfeed_sensor = float(np.percentile(T_out_cells, 75))
            self._last_valid_T_outfeed_sensor = T_outfeed_sensor
            # Snapshot at peak sensor T for cross-section when belt later clears (Run#1: 82–93°C at oven exit)
            if T_outfeed_sensor > self._peak_outfeed_sensor_T:
                self._peak_outfeed_sensor_T = T_outfeed_sensor
                self._peak_outfeed_T_yz = self.thermal.T[-1, :, :].copy()
                self._peak_outfeed_M_yz = self.moisture.M[-1, :, :].copy()
        else:
            # Use last valid sensor (75th percentile) when belt has cleared, not mean
            T_outfeed_sensor = self._last_valid_T_outfeed_sensor

        # Evaporative power (latent heat sink)
        evap_rate_mat = self.moisture.evap_rate[mat_mask]
        evap_power_w = float(np.sum(evap_rate_mat)) * self._cell_vol * 2.26e6
        evap_power_kw = evap_power_w / 1000.0

        # Cumulative energy and water removal
        total_energy_kwh = self._total_rf_energy_j / 3.6e6
        v_belt = self.conveyor.state.belt_speed_m_per_s
        bed_cross = mat.bed_depth_m * self._machine.belt_width_m
        rho_bulk = mat.bulk_density(mat.initial_moisture_wb)
        throughput_kg_s = rho_bulk * bed_cross * v_belt
        # Water removed: integrate over time (batch-correct).
        # Once batch exhausts, use the steady-state moisture snapshot to prevent
        # the M=0 front from artificially inflating water removal calculation.
        # The M=0 front mixing at outfeed would drop M_outfeed, creating a false
        # spike in delta_M and cumulative water removed.
        M_for_delta = (self._moisture_at_batch_exhausted if self._batch_exhausted
                       else M_outfeed)
        delta_M_step = max(mat.initial_moisture_wb - M_for_delta, 0.0)
        self._cumulative_water_removed_kg += throughput_kg_s * delta_M_step * dt
        water_removed_kg = self._cumulative_water_removed_kg
        spec_energy = (total_energy_kwh / max(water_removed_kg, 1e-6)
                       if water_removed_kg > 0.001 else 0.0)

        # Controller state
        ctrl_state = ""
        if self._enable_controller:
            ctrl_state = self.controller.status.state.value

        state = StepState(
            time_s=self._time,
            T_mean_c=T_mean,
            T_max_c=T_max,
            T_outfeed_c=T_outfeed,
            T_outfeed_sensor_c=T_outfeed_sensor,
            M_mean_wb=M_mean,
            M_min_wb=M_min,
            M_outfeed_wb=M_outfeed,
            rf_power_kw=P_rf_kw,
            anode_current_a=I_a,
            electrode_gap_mm=(self.controller.status.electrode_gap_mm
                              if self._enable_controller else recipe.electrode_gap_mm),
            belt_speed_m_per_min=self.conveyor.state.belt_speed_m_per_min,
            controller_state=ctrl_state,
            evap_power_kw=evap_power_kw,
            total_energy_kwh=total_energy_kwh,
            water_removed_kg=water_removed_kg,
            specific_energy_kwh_per_kg=spec_energy,
            electrode_temperature_c=self._T_electrode_c,
            protein_denaturation=(
                self._particles.mean_denaturation
                if self._particles is not None else 0.0
            ),
        )
        self._history.append(state)

        # ── 10. PARTICLES ─────────────────────────────────────────────
        # Lagrangian tracer update: move particles with belt, sample
        # T and M from the Eulerian grid via trilinear interpolation.
        # One-way coupling (E→L): particles don't affect the grid.
        if self._particles is not None:
            self._particles.step(
                dt_sim=dt,
                belt_speed_m_per_s=self.conveyor.state.belt_speed_m_per_s,
                T_field=self.thermal.T,
                M_field=self.moisture.M,
                cell_is_material=self.cell_is_material,
                grid_origin=self._grid_origin,
                cell_sizes=self._cell_sizes,
            )

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

        # Start conveyor drive at recipe belt speed
        self.conveyor.start(speed_m_per_min=recipe.belt_speed_m_per_min)

        t_end = self._time + duration_s

        while self._time < t_end - 1e-12:
            if adaptive_dt or dt is None:
                _dt = self.compute_stable_dt(recipe)
            else:
                _dt = dt
            # Don't overshoot
            _dt = min(_dt, t_end - self._time)
            self.step(_dt, recipe)

        return self._build_result()

    def _build_result(self) -> PretreatmentResult:
        """Assemble a :class:`PretreatmentResult` from current state.

        Called by :meth:`run` at the end and also by the live 3-D
        visualisation callback after the PyVista window closes.
        """
        mat_mask = (self.cell_is_material == 1)
        M_final = self.moisture.M[mat_mask]
        T_final = self.thermal.T[mat_mask]

        # Throughput estimate (use actual belt speed from conveyor drive)
        belt_speed = self.conveyor.state.belt_speed_m_per_s
        bed_cross = self._material.bed_depth_m * self._machine.belt_width_m
        rho_bulk = self._material.bulk_density(self._material.initial_moisture_wb)
        throughput_kg_h = rho_bulk * bed_cross * belt_speed * 3600.0

        return PretreatmentResult(
            duration_s=self._time,
            final_moisture_mean_wb=float(np.mean(M_final)) if M_final.size else 0.0,
            final_temperature_mean_c=float(np.mean(T_final)) if T_final.size else 0.0,
            energy_consumed_kwh=self._total_rf_energy_j / 3.6e6,
            throughput_kg_per_h=throughput_kg_h,
            time_series={
                "time_s": [s.time_s for s in self._history],
                "T_mean_c": [s.T_mean_c for s in self._history],
                "T_max_c": [s.T_max_c for s in self._history],
                "T_outfeed_c": [s.T_outfeed_c for s in self._history],
                "T_outfeed_sensor_c": [s.T_outfeed_sensor_c for s in self._history],
                "M_mean_wb": [s.M_mean_wb for s in self._history],
                "M_outfeed_wb": [s.M_outfeed_wb for s in self._history],
                "rf_power_kw": [s.rf_power_kw for s in self._history],
                "evap_power_kw": [s.evap_power_kw for s in self._history],
                "anode_current_a": [s.anode_current_a for s in self._history],
                "electrode_gap_mm": [s.electrode_gap_mm for s in self._history],
                "total_energy_kwh": [s.total_energy_kwh for s in self._history],
                "water_removed_kg": [s.water_removed_kg for s in self._history],
                "specific_energy_kwh_per_kg": [s.specific_energy_kwh_per_kg for s in self._history],
                "electrode_temperature_c": [s.electrode_temperature_c for s in self._history],
                "protein_denaturation": [s.protein_denaturation for s in self._history],
            },
            T_final=self.thermal.T.copy(),
            M_final=self.moisture.M.copy(),
        )

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

        # For outlet measurements, only include cells with ACTUAL material.
        # During batch clearing, some cells in the geometric material zone
        # may be empty (M ≈ 0). We detect actual material by M > threshold.
        # Threshold is 1% of initial moisture - anything below is "empty".
        M_threshold = mat.initial_moisture_wb * 0.01
        has_material = mat_mask_yz & (M_yz > M_threshold)

        T_mat = T_yz[has_material]
        M_mat = M_yz[has_material]

        # For finite mass mode: if belt has cleared (no material at outlet),
        # show the PEAK PROCESSING snapshot so the cross-section matches the
        # time-series and Run#1 strip data (82–93°C at oven exit). Do NOT use
        # the current grid (cold/empty) or particle data (cooled in bin).
        use_peak_snapshot = False
        use_last_valid_sensor = False  # belt cleared, no peak: use 75th percentile fallback
        if T_mat.size == 0:
            if (self._peak_outfeed_T_yz is not None and
                    self._peak_outfeed_M_yz is not None):
                # Temperature: use peak snapshot (matches strip sensor readings 82-93°C)
                T_yz = self._peak_outfeed_T_yz.copy()
                peak_has = mat_mask_yz & (self._peak_outfeed_M_yz > M_threshold)
                T_peak = T_yz[peak_has]
                avg_T = float(np.mean(T_peak)) if T_peak.size else self._last_valid_T_outfeed
                max_T = float(np.max(T_peak)) if T_peak.size else self._peak_outfeed_sensor_T
                sensor_T = (float(np.percentile(T_peak, 75)) if T_peak.size
                            else self._peak_outfeed_sensor_T)

                # Moisture: use BATCH EXHAUSTED snapshot (not peak-T snapshot which is driest,
                # nor last_valid which gets dragged down by M=0 front mixing at outfeed).
                # The batch-exhausted snapshot captures steady-state processing moisture
                # BEFORE the M=0 front affects outfeed readings.
                avg_M = self._moisture_at_batch_exhausted
                M_yz = self._peak_outfeed_M_yz.copy()
                M_yz[mat_mask_yz] = avg_M  # Fill heatmap with representative value
                moisture_cv = 0.0  # CV not meaningful for single value
                use_peak_snapshot = True
            else:
                T_mat = np.array([self._last_valid_T_outfeed])
                M_mat = np.array([self._moisture_at_batch_exhausted])
                use_last_valid_sensor = True

        if not use_peak_snapshot:
            # Compute averages from current slice or fallback T_mat/M_mat
            avg_T = float(np.mean(T_mat)) if T_mat.size else mat.initial_temperature_c
            avg_M = float(np.mean(M_mat)) if M_mat.size else mat.initial_moisture_wb
            max_T = float(np.max(T_mat)) if T_mat.size else mat.initial_temperature_c
            sensor_T = (self._last_valid_T_outfeed_sensor if use_last_valid_sensor
                        else (float(np.percentile(T_mat, 75)) if T_mat.size
                              else mat.initial_temperature_c))
            if M_mat.size and avg_M > 1e-8:
                moisture_cv = float(np.std(M_mat) / avg_M)
            else:
                moisture_cv = 0.0
            # When belt cleared (no peak snapshot): fill material zone for heatmap
            if T_mat.size == 0:
                T_yz = T_yz.copy()
                M_yz = M_yz.copy()
                T_yz[mat_mask_yz] = avg_T
                M_yz[mat_mask_yz] = avg_M

        # Throughput (use actual belt speed from conveyor drive)
        v_belt = self.conveyor.state.belt_speed_m_per_s
        if v_belt <= 0:
            v_belt = recipe.belt_speed_m_per_min / 60.0  # fallback
        bed_cross = mat.bed_depth_m * machine.belt_width_m
        rho_bulk = mat.bulk_density(mat.initial_moisture_wb)
        throughput_kg_h = rho_bulk * bed_cross * v_belt * 3600.0

        # Residence time
        residence_s = machine.oven_length_m / v_belt if v_belt > 0 else 0.0

        # Energy metrics
        total_kwh = self._total_rf_energy_j / 3.6e6
        water_removed_kg_h = throughput_kg_h * max(mat.initial_moisture_wb - avg_M, 0.0)
        # Specific energy: average RF power [kW] / water removal rate [kg/h]
        # gives kWh per kg water removed.  Guard against division by zero
        # when no water has been removed (e.g. simulation too short or
        # drying model inactive).
        if water_removed_kg_h > 1e-6:
            avg_power_kw = (self._total_rf_energy_j / max(self._time, 1.0)) / 1000.0
            specific_energy = avg_power_kw / water_removed_kg_h
        else:
            specific_energy = float("inf") if total_kwh > 0 else 0.0

        # Protein denaturation (Phase 4: Arrhenius kinetics on Lagrangian particles).
        # Uses the Biot core temperature model — denaturation happens inside
        # the seed, not at the surface.  Two-fraction model: vicilin (7S, onset
        # 62 C) + legumin (11S, onset 76 C) weighted by pea globulin composition.
        if self._particles is not None:
            denaturation = self._particles.mean_denaturation
        else:
            denaturation = 0.0

        return OutletState(
            temperature_field=T_yz.copy(),
            moisture_field=M_yz.copy(),
            avg_temperature_c=avg_T,
            sensor_temperature_c=sensor_T,
            avg_moisture_wb=avg_M,
            moisture_uniformity=moisture_cv,
            throughput_kg_per_hr=throughput_kg_h,
            total_energy_kwh=total_kwh,
            specific_energy_kwh_per_kg=specific_energy,
            residence_time_s=residence_s,
            max_temperature_c=max_T,
            protein_denaturation_fraction=denaturation,
            at_peak_processing_snapshot=use_peak_snapshot,
        )

    # ------------------------------------------------------------------
    # Generator model
    # ------------------------------------------------------------------

    def _compute_electrode_voltage(self, machine: MachineConfig) -> float:
        """Compute the RF voltage at the electrodes (Approach B, §4.1.3).

        The GP-15's self-excited triode oscillator produces an RF
        voltage that depends on the anode DC supply and the tank
        circuit coupling:

            V_rf = V_anode × oscillator_coupling_factor

        The anode voltage droops under load.  The oscillator cannot
        deliver more than its rated maximum (15 kW) — beyond that
        the triode de-excites and the tank circuit detuning limits
        the output.  The droop model is valid between no-load and
        full-load; we clamp the fraction at 1.0 to prevent the
        voltage computation from extrapolating into the unphysical
        de-excitation region.

        The MRH gap control (which sees the UNCLAMPED Ia from actual
        delivered power) reduces the gap to bring the operating point
        back into the rated range.

        Returns:
            V_rf [kV] at the electrodes.
        """
        k = machine.oscillator_coupling_factor

        # Anode current from smoothed RF power (tank circuit inertia).
        # The smoothing eliminates the one-timestep feedback lag that was
        # causing ±0.3-0.6A oscillations in Ia (Run#2 shows ±0.02-0.05A).
        #
        # Clamp fraction at 1.0 for the VOLTAGE computation:
        # the oscillator's tank circuit physically limits the output
        # voltage to the range covered by the droop model.
        #
        # fraction uses P_rf directly (same reasoning as step 8 Ia):
        # the test-report endpoints already encode efficiency.
        fraction = min(self._last_P_rf_kw / max(machine.max_rf_power_kw, 0.01), 1.0) if self._last_P_rf_kw > 0 else 0.0
        I_a = (machine.anode_current_no_load_a
               + (machine.anode_current_full_load_a - machine.anode_current_no_load_a) * fraction)

        V_a_kv = machine.anode_voltage_kv(I_a)
        V_rf_kv = V_a_kv * k
        return V_rf_kv

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
            k_dispersion=mat.k_dispersion,
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
