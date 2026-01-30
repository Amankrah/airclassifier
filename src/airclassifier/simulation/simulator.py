"""
Main simulator for cyclone air classifier.

Orchestrates the simulation by combining:
- Geometry (cyclone assembly)
- Flow field (analytical or CFD)
- Particle system
- Force calculations
- Time integration

Two simulation modes are available:
- ANALYTICAL: Fast Rankine vortex flow field (CycloneSimulator)
- CFD: Full Navier-Stokes with CFD-DEM coupling (CFDDEMCoupler)

Use create_simulator() factory function to get the appropriate simulator
based on configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum
import numpy as np
import warp as wp
import yaml

from ..geometry import CycloneAssembly, CycloneGeometryParams
from ..fluid import (
    CycloneFlowField,
    CycloneFlowParams,
    create_warp_flow_params,
    WarpFlowParams,
    wp_velocity_at,
)
from ..particles.material import ParticleMaterial, MaterialProperties, SizeDistributionParams
from ..kinetics.forces import (
    wp_drag_force_schiller_naumann,
    wp_gravity_acceleration,
    wp_centrifugal_acceleration,
)
from ..particles.interactions.particle_wall import (
    handle_wall_collisions_sdf,
    WallCollisionParams,
)
from ..utils.constants import GRAVITY, AirProperties


class FlowMode(Enum):
    """Flow field simulation mode."""
    ANALYTICAL = "analytical"  # Fast Rankine vortex (default)
    CFD = "cfd"                # Full Navier-Stokes CFD-DEM coupling


@dataclass
class SimulationConfig:
    """Configuration for the cyclone simulation."""

    # Flow mode
    flow_mode: FlowMode = FlowMode.ANALYTICAL  # ANALYTICAL or CFD

    # Time parameters
    dt: float = 1.0e-5              # [s] Time step
    duration: float = 1.0           # [s] Total simulation time
    output_interval: float = 0.01   # [s] Output interval

    # Particle parameters
    num_particles: int = 10000      # Number of particles to simulate
    injection_duration: float = 0.1 # [s] Duration over which to inject particles

    # Physics options
    include_gravity: bool = True
    include_drag: bool = True
    include_centrifugal: bool = True  # Note: automatically included via particle motion
    include_virtual_mass: bool = False
    include_wall_collisions: bool = True

    # Wall collision parameters
    wall_restitution: float = 0.8
    wall_friction: float = 0.3

    # CFD-specific parameters (only used if flow_mode == CFD)
    cfd_resolution: Tuple[int, int, int] = (64, 128, 64)
    cfd_substeps: int = 10
    turbulence_model: str = "k_epsilon"  # "none", "k_epsilon", "smagorinsky"
    turbulence_intensity: float = 0.05
    coupling_mode: str = "one_way"  # "one_way" or "two_way"

    # Device
    device: str = "cuda"

    # Output
    output_directory: str = "./results"
    save_trajectories: bool = False  # Can be memory intensive

    @property
    def num_steps(self) -> int:
        """Total number of time steps."""
        return int(self.duration / self.dt)

    @property
    def output_steps(self) -> int:
        """Steps between outputs."""
        return max(1, int(self.output_interval / self.dt))


@dataclass
class SimulationState:
    """Current state of the simulation."""

    # Time
    time: float = 0.0
    step: int = 0

    # Particle arrays (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    is_active: Optional[wp.array] = None

    # Statistics
    particles_injected: int = 0
    particles_collected: int = 0     # In dust outlet
    particles_escaped: int = 0       # Through overflow
    particles_active: int = 0

    # Collection tracking
    collected_diameters: List[float] = field(default_factory=list)
    escaped_diameters: List[float] = field(default_factory=list)


class CycloneSimulator:
    """
    Main simulator for cyclone air classifier particle separation.

    Performs Lagrangian particle tracking through an analytically
    defined flow field in a cyclone geometry.
    """

    def __init__(
        self,
        geometry_params: CycloneGeometryParams,
        material: ParticleMaterial,
        config: SimulationConfig,
    ):
        """
        Initialize the simulator.

        Args:
            geometry_params: Cyclone geometry parameters
            material: Particle material definition
            config: Simulation configuration
        """
        self.config = config
        self.material = material
        self.device = config.device

        # Initialize Warp
        wp.init()

        # Create geometry
        self.cyclone = CycloneAssembly(geometry_params, device=config.device)

        # Create flow field
        self._setup_flow_field(geometry_params)

        # Initialize state
        self.state = SimulationState()

        # Pre-allocate arrays
        self._allocate_arrays()

        # Store geometry info for kernels
        self._setup_kernel_params()

    def _setup_flow_field(self, geom: CycloneGeometryParams):
        """Set up the analytical flow field."""
        flow_params = CycloneFlowParams(
            cylinder_radius=geom.cylinder_diameter / 2.0,
            vortex_finder_radius=geom.vortex_finder_diameter / 2.0,
            cylinder_height=geom.cylinder_height,
            cone_height=geom.cone_height,
            cone_bottom_radius=geom.cone_tip_diameter / 2.0,
            inlet_velocity=15.0,  # Default, can be overridden
            inlet_width=geom.inlet_width,
            inlet_height=geom.inlet_height,
            fluid_density=AirProperties.DENSITY,
            fluid_viscosity=AirProperties.DYNAMIC_VISCOSITY,
        )

        self.flow_field = CycloneFlowField(flow_params)
        self.flow_params_wp = create_warp_flow_params(self.flow_field)

    def _allocate_arrays(self):
        """Pre-allocate particle arrays."""
        n = self.config.num_particles

        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)

        # Temporary arrays for computation
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._fluid_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)

    def _setup_kernel_params(self):
        """Set up parameters for kernels."""
        geom = self.cyclone.params

        self.axis_center = wp.vec3(
            geom.center[0],
            geom.center[1],
            geom.center[2]
        )

        # Geometry parameters for wall collisions
        self.cylinder_radius = geom.cylinder_diameter / 2.0
        self.cylinder_height = geom.cylinder_height
        self.cone_height = geom.cone_height
        self.cone_bottom_radius = geom.cone_tip_diameter / 2.0
        self.vf_radius = geom.vortex_finder_diameter / 2.0
        self.vf_bottom_y = geom.center[1] - geom.vortex_finder_length

        # Outlet positions
        self.dust_outlet_y = geom.center[1] - geom.cylinder_height - geom.cone_height
        self.dust_outlet_radius = geom.dust_outlet_diameter / 2.0
        self.overflow_y = geom.center[1] + 0.05  # Vortex finder top
        self.overflow_radius = geom.vortex_finder_diameter / 2.0

    def inject_particles(self, n_inject: int):
        """
        Inject new particles at the inlet.

        Args:
            n_inject: Number of particles to inject
        """
        if self.state.particles_injected >= self.config.num_particles:
            return

        n_inject = min(n_inject, self.config.num_particles - self.state.particles_injected)

        # Sample diameters
        diameters = self.material.sample_diameters(
            n_inject,
            seed=self.state.step + 42
        )

        # Get inlet position and velocity
        inlet = self.cyclone.get_inlet_conditions()
        inlet_pos = inlet["position"]
        inlet_dir = inlet["direction"]
        inlet_width = inlet["width"]
        inlet_height = inlet["height"]

        # Generate random positions within inlet
        rng = np.random.default_rng(self.state.step + 123)

        # Spread particles across inlet cross-section
        positions = np.zeros((n_inject, 3), dtype=np.float32)
        velocities = np.zeros((n_inject, 3), dtype=np.float32)

        for i in range(n_inject):
            # Random offset within inlet
            offset_w = (rng.random() - 0.5) * inlet_width * 0.8
            offset_h = (rng.random() - 0.5) * inlet_height * 0.8

            # Position at inlet
            positions[i] = inlet_pos + np.array([0, offset_h, offset_w])

            # Initial velocity (same as inlet flow)
            velocities[i] = inlet_dir * self.flow_field.params.inlet_velocity

        # Copy to device
        start_idx = self.state.particles_injected
        end_idx = start_idx + n_inject

        # Update arrays
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)

        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_inject)

        # Mark as active
        active_flags = wp.array(np.ones(n_inject, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_inject)

        self.state.particles_injected += n_inject
        self.state.particles_active += n_inject

    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt

        # Inject particles if still in injection phase
        if self.state.time < self.config.injection_duration:
            inject_rate = self.config.num_particles / self.config.injection_duration
            n_inject = int(inject_rate * dt) + (1 if np.random.random() < (inject_rate * dt % 1) else 0)
            if n_inject > 0:
                self.inject_particles(n_inject)

        # Skip if no active particles
        if self.state.particles_active == 0:
            self.state.time += dt
            self.state.step += 1
            return

        n = self.state.particles_injected

        # Compute fluid velocities at particle positions
        wp.launch(
            kernel=compute_fluid_velocities_kernel,
            dim=n,
            inputs=[
                self.state.positions,
                self._fluid_velocities,
                self.state.is_active,
                self.flow_params_wp
            ],
            device=self.device
        )

        # Compute accelerations
        wp.launch(
            kernel=compute_particle_accelerations,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self._fluid_velocities,
                self.state.is_active,
                self._accelerations,
                self.material.density,
                self.flow_field.params.fluid_density,
                self.flow_field.params.fluid_viscosity,
                float(GRAVITY),
                self.axis_center,
                self.config.include_drag,
                self.config.include_gravity,
            ],
            device=self.device
        )

        # Integrate (Euler for simplicity, can upgrade to RK4)
        wp.launch(
            kernel=integrate_euler,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self._accelerations,
                self.state.is_active,
                dt
            ],
            device=self.device
        )

        # Handle wall collisions
        if self.config.include_wall_collisions:
            wp.launch(
                kernel=handle_wall_collisions_sdf,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.diameters,
                    self.state.is_active,
                    self.axis_center,
                    self.cylinder_radius,
                    self.cylinder_height,
                    self.cone_height,
                    self.cone_bottom_radius,
                    self.vf_radius,
                    self.vf_bottom_y,
                    self.config.wall_restitution,
                    self.config.wall_friction,
                ],
                device=self.device
            )

        # Check for particles exiting
        wp.launch(
            kernel=check_particle_exits,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.is_active,
                self.axis_center,
                self.dust_outlet_y,
                self.dust_outlet_radius,
                self.overflow_y,
                self.overflow_radius,
            ],
            device=self.device
        )

        # Update time
        self.state.time += dt
        self.state.step += 1

    def run(self, progress_callback=None):
        """
        Run the full simulation.

        Args:
            progress_callback: Optional function called with (step, total_steps)
        """
        total_steps = self.config.num_steps

        for step in range(total_steps):
            self.step()

            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)

        wp.synchronize()

    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results.

        Returns:
            Dictionary with results
        """
        # Copy data back to CPU
        positions = self.state.positions.numpy()
        velocities = self.state.velocities.numpy()
        diameters = self.state.diameters.numpy()
        is_active = self.state.is_active.numpy()

        # Count collected vs escaped
        n_collected = np.sum((is_active == -1))  # Marked as collected
        n_escaped = np.sum((is_active == -2))    # Marked as escaped
        n_active = np.sum((is_active == 1))

        return {
            "time": self.state.time,
            "steps": self.state.step,
            "particles_injected": self.state.particles_injected,
            "particles_collected": int(n_collected),
            "particles_escaped": int(n_escaped),
            "particles_active": int(n_active),
            "collection_efficiency": n_collected / max(1, n_collected + n_escaped),
            "positions": positions,
            "velocities": velocities,
            "diameters": diameters,
            "is_active": is_active,
        }

    @classmethod
    def from_config(cls, config_path: str) -> "CycloneSimulator":
        """
        Create simulator from YAML configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            CycloneSimulator instance
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Parse geometry
        geom_cfg = cfg.get('geometry', {})
        body_cfg = geom_cfg.get('body', {})
        inlet_cfg = geom_cfg.get('inlet', {})
        vf_cfg = geom_cfg.get('vortex_finder', {})
        do_cfg = geom_cfg.get('dust_outlet', {})

        geometry_params = CycloneGeometryParams(
            cylinder_diameter=body_cfg.get('cylinder_diameter', 0.3),
            cylinder_height=body_cfg.get('cylinder_height', 0.6),
            cone_height=body_cfg.get('cone_height', 0.9),
            cone_tip_diameter=body_cfg.get('cone_tip_diameter', 0.075),
            inlet_width=inlet_cfg.get('width', 0.075),
            inlet_height=inlet_cfg.get('height', 0.15),
            vortex_finder_diameter=vf_cfg.get('diameter', 0.15),
            vortex_finder_length=vf_cfg.get('length', 0.3),
            dust_outlet_diameter=do_cfg.get('diameter', 0.075),
        )

        # Parse material
        mat_cfg = cfg.get('particles', {}).get('material', {})
        size_cfg = cfg.get('particles', {}).get('size_distribution', {})

        material = ParticleMaterial.create(
            material_name=mat_cfg.get('name', 'quartz'),
            distribution_type=size_cfg.get('type', 'rosin_rammler'),
            d50=size_cfg.get('d50', 50.0e-6),
            spread=size_cfg.get('spread', 2.0),
            d_min=size_cfg.get('d_min', 1.0e-6),
            d_max=size_cfg.get('d_max', 200.0e-6),
        )

        # Parse simulation config
        sim_cfg = cfg.get('simulation', {})

        config = SimulationConfig(
            dt=sim_cfg.get('dt', 1.0e-5),
            duration=sim_cfg.get('duration', 1.0),
            output_interval=sim_cfg.get('output_interval', 0.01),
            num_particles=cfg.get('particles', {}).get('num_particles', 10000),
            device=sim_cfg.get('device', 'cuda'),
        )

        return cls(geometry_params, material, config)


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def compute_fluid_velocities_kernel(
    positions: wp.array(dtype=wp.vec3),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    params: WarpFlowParams
):
    """Compute fluid velocities at particle positions."""
    tid = wp.tid()

    if is_active[tid] != 1:
        fluid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    fluid_velocities[tid] = wp_velocity_at(positions[tid], params)


@wp.kernel
def compute_particle_accelerations(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    accelerations: wp.array(dtype=wp.vec3),
    rho_p: float,
    rho_f: float,
    mu_f: float,
    g: float,
    axis_center: wp.vec3,
    include_drag: bool,
    include_gravity: bool,
):
    """Compute total acceleration for each particle."""
    tid = wp.tid()

    if is_active[tid] != 1:
        accelerations[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    v_f = fluid_velocities[tid]

    acc = wp.vec3(0.0, 0.0, 0.0)

    # Drag acceleration
    if include_drag:
        v_rel = v_f - vel
        v_rel_mag = wp.length(v_rel)
        eps = 1.0e-10

        if v_rel_mag > eps:
            # Reynolds number
            Re = rho_f * v_rel_mag * d / mu_f

            # Schiller-Naumann drag coefficient
            if Re < eps:
                Cd = 24.0 / eps
            else:
                Cd = (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))

            # Particle mass
            volume = (3.141592653589793 / 6.0) * d * d * d
            mass = rho_p * volume

            # Projected area
            A_p = (3.141592653589793 / 4.0) * d * d

            # Drag acceleration
            a_drag_mag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag / mass
            acc = acc + v_rel * (a_drag_mag / v_rel_mag)

    # Gravity with buoyancy
    if include_gravity:
        a_grav = (1.0 - rho_f / rho_p) * g
        acc = acc + wp.vec3(0.0, -a_grav, 0.0)

    accelerations[tid] = acc


@wp.kernel
def integrate_euler(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    dt: float
):
    """Euler integration step."""
    tid = wp.tid()

    if is_active[tid] != 1:
        return

    vel = velocities[tid]
    acc = accelerations[tid]
    pos = positions[tid]

    # Update velocity
    new_vel = vel + acc * dt

    # Update position
    new_pos = pos + new_vel * dt

    velocities[tid] = new_vel
    positions[tid] = new_pos


@wp.kernel
def check_particle_exits(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    axis_center: wp.vec3,
    dust_outlet_y: float,
    dust_outlet_radius: float,
    overflow_y: float,
    overflow_radius: float,
):
    """Check if particles have exited through outlets."""
    tid = wp.tid()

    if is_active[tid] != 1:
        return

    pos = positions[tid]

    # Check dust outlet (bottom)
    if pos[1] < dust_outlet_y:
        dx = pos[0] - axis_center[0]
        dz = pos[2] - axis_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        if r < dust_outlet_radius:
            is_active[tid] = -1  # Collected
            return

    # Check overflow (top)
    if pos[1] > overflow_y:
        dx = pos[0] - axis_center[0]
        dz = pos[2] - axis_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        if r < overflow_radius:
            is_active[tid] = -2  # Escaped through overflow
            return


def create_simulator(
    geometry_params: CycloneGeometryParams,
    material: ParticleMaterial,
    config: SimulationConfig,
) -> Union["CycloneSimulator", "CFDDEMCoupler"]:
    """
    Factory function to create the appropriate simulator.

    Creates either a CycloneSimulator (analytical flow) or CFDDEMCoupler
    (full CFD-DEM coupling) based on the configuration.

    Args:
        geometry_params: Cyclone geometry parameters
        material: Particle material definition
        config: Simulation configuration (includes flow_mode)

    Returns:
        Simulator instance (CycloneSimulator or CFDDEMCoupler)

    Example:
        >>> # Analytical flow (fast)
        >>> config = SimulationConfig(flow_mode=FlowMode.ANALYTICAL)
        >>> sim = create_simulator(geometry_params, material, config)

        >>> # Full CFD (accurate)
        >>> config = SimulationConfig(flow_mode=FlowMode.CFD)
        >>> sim = create_simulator(geometry_params, material, config)
    """
    if config.flow_mode == FlowMode.ANALYTICAL:
        return CycloneSimulator(geometry_params, material, config)

    elif config.flow_mode == FlowMode.CFD:
        # Import here to avoid circular imports
        from .cfd_dem_coupling import (
            CFDDEMCoupler,
            CFDConfig,
            DEMConfig,
            CycloneCFDParams,
            TurbulenceModelType,
            CouplingMode,
        )

        # Convert geometry params to CFD params
        cyclone_cfd_params = CycloneCFDParams(
            cylinder_diameter=geometry_params.cylinder_diameter,
            cylinder_height=geometry_params.cylinder_height,
            cone_height=geometry_params.cone_height,
            cone_tip_diameter=geometry_params.cone_tip_diameter,
            inlet_width=geometry_params.inlet_width,
            inlet_height=geometry_params.inlet_height,
            inlet_velocity=15.0,  # Default, can be parameterized
            vortex_finder_diameter=geometry_params.vortex_finder_diameter,
            vortex_finder_length=geometry_params.vortex_finder_length,
        )

        # Map turbulence model string to enum
        turb_map = {
            "none": TurbulenceModelType.NONE,
            "k_epsilon": TurbulenceModelType.K_EPSILON,
            "smagorinsky": TurbulenceModelType.SMAGORINSKY,
        }
        turb_model = turb_map.get(config.turbulence_model, TurbulenceModelType.K_EPSILON)

        # Map coupling mode string to enum
        coupling_map = {
            "one_way": CouplingMode.ONE_WAY,
            "two_way": CouplingMode.TWO_WAY,
        }
        coupling = coupling_map.get(config.coupling_mode, CouplingMode.ONE_WAY)

        # Create CFD config
        cfd_config = CFDConfig(
            domain_size=(
                geometry_params.cylinder_diameter * 1.2,
                geometry_params.cylinder_height + geometry_params.cone_height + 0.1,
                geometry_params.cylinder_diameter * 1.2,
            ),
            resolution=config.cfd_resolution,
            dt=config.dt,
            turbulence_model=turb_model,
            turbulence_intensity=config.turbulence_intensity,
            coupling_mode=coupling,
            cfd_substeps=config.cfd_substeps,
        )

        # Create DEM config
        dem_config = DEMConfig(
            dt=config.dt,
            num_particles=config.num_particles,
            injection_duration=config.injection_duration,
            wall_restitution=config.wall_restitution,
            wall_friction=config.wall_friction,
        )

        return CFDDEMCoupler(
            cyclone_params=cyclone_cfd_params,
            cfd_config=cfd_config,
            dem_config=dem_config,
            material=material,
            device=config.device,
        )

    else:
        raise ValueError(f"Unknown flow mode: {config.flow_mode}")


def main():
    """Entry point for command-line usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m airclassifier.simulation.simulator <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    print("Creating simulator...")
    sim = CycloneSimulator.from_config(config_path)

    print("Running simulation...")
    sim.run(progress_callback=lambda s, t: print(f"Step {s}/{t}") if s % 1000 == 0 else None)

    print("\nResults:")
    results = sim.get_results()
    print(f"  Particles injected: {results['particles_injected']}")
    print(f"  Particles collected: {results['particles_collected']}")
    print(f"  Particles escaped: {results['particles_escaped']}")
    print(f"  Collection efficiency: {results['collection_efficiency']:.2%}")


if __name__ == "__main__":
    main()
