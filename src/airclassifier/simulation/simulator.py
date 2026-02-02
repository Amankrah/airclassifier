"""
System-level simulators for air classifier.

Provides separate simulation modes for each subsystem and a complete
integrated simulation:

1. AirSystemSimulator: Flow through blower, filter, dampers
2. FeedSystemSimulator: Material flow through hopper, airlock, feeder, deagglomerator
3. ClassificationSystemSimulator: Particle separation through venturi, zigzag, cyclones, bag filter
4. CompleteSystemSimulator: Full integrated simulation with all systems running

Each simulator can operate independently or be coupled for full system analysis.

Two flow modes are available:
- ANALYTICAL: Fast Rankine vortex / analytical flow models
- CFD: Full Navier-Stokes with CFD-DEM coupling
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import warp as wp
import yaml

from ..utils.constants import GRAVITY, AirProperties, PI


class FlowMode(Enum):
    """Flow field simulation mode."""
    ANALYTICAL = "analytical"  # Fast analytical flow models
    CFD = "cfd"                # Full Navier-Stokes CFD-DEM coupling


class SystemState(Enum):
    """Operating state of a system."""
    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


# =============================================================================
# BASE CONFIGURATION AND STATE
# =============================================================================

@dataclass
class BaseSimulationConfig:
    """Base configuration for all simulators."""
    
    # Flow mode
    flow_mode: FlowMode = FlowMode.ANALYTICAL
    
    # Time parameters
    dt: float = 1.0e-5              # [s] Time step
    duration: float = 1.0           # [s] Total simulation time
    output_interval: float = 0.01   # [s] Output interval
    
    # Physics options
    include_gravity: bool = True
    include_drag: bool = True
    include_wall_collisions: bool = True
    
    # Wall collision parameters
    wall_restitution: float = 0.8
    wall_friction: float = 0.3
    
    # Device
    device: str = "cuda"
    
    # Output
    output_directory: str = "./results"
    
    @property
    def num_steps(self) -> int:
        """Total number of time steps."""
        return int(self.duration / self.dt)
    
    @property
    def output_steps(self) -> int:
        """Steps between outputs."""
        return max(1, int(self.output_interval / self.dt))


@dataclass
class BaseSimulationState:
    """Base state for all simulators."""
    
    # Time
    time: float = 0.0
    step: int = 0
    
    # System state
    system_state: SystemState = SystemState.OFF


# =============================================================================
# AIR SYSTEM SIMULATOR
# =============================================================================

@dataclass
class AirSystemConfig(BaseSimulationConfig):
    """Configuration for air system simulation."""
    
    # Blower parameters
    blower_rpm: float = 3000.0       # [RPM] Blower speed
    blower_ramp_time: float = 0.5    # [s] Time to reach full speed
    
    # Damper settings (0=closed, 1=fully open)
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Flow parameters
    design_flow_rate_m3_h: float = 3000.0    # [m³/h] Design flow rate
    design_pressure_rise_Pa: float = 5000.0  # [Pa] Design pressure rise


@dataclass
class AirSystemState(BaseSimulationState):
    """State for air system simulation."""
    
    # Blower state
    blower_rpm: float = 0.0
    blower_target_rpm: float = 0.0
    
    # Flow state
    flow_rate_m3_h: float = 0.0
    pressure_Pa: float = 0.0
    
    # Damper states
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Energy tracking
    power_consumption_kW: float = 0.0
    total_energy_kWh: float = 0.0


class AirSystemSimulator:
    """
    Simulator for the air system (blower + filter + dampers).
    
    Simulates:
    - Blower startup/shutdown ramp
    - Flow rate based on blower speed and damper positions
    - Pressure drop through filter and dampers
    - Power consumption
    
    Flow path:
    Ambient Air -> Filter -> Blower -> Damper(s) -> Outlet
    """
    
    def __init__(
        self,
        assembly: Any,  # AirSystemAssembly
        config: AirSystemConfig = None,
    ):
        """
        Initialize air system simulator.
        
        Args:
            assembly: AirSystemAssembly instance
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or AirSystemConfig()
        self.state = AirSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Get system parameters from assembly
        self._setup_system_parameters()
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.blower = self.assembly.blower
        self.dampers = self.assembly.dampers
        self.inlet_filter = self.assembly.inlet_filter
        
        # Blower characteristics
        self.max_rpm = 3600.0  # Typical max
        self.design_rpm = self.config.blower_rpm
        
        # Flow coefficient (simplified fan law)
        self.flow_coefficient = self.config.design_flow_rate_m3_h / self.design_rpm
        
    def start(self):
        """Start the air system (begin blower ramp-up)."""
        self.state.system_state = SystemState.STARTING
        self.state.blower_target_rpm = self.config.blower_rpm
        
    def stop(self):
        """Stop the air system (begin blower ramp-down)."""
        self.state.system_state = SystemState.STOPPING
        self.state.blower_target_rpm = 0.0
        
    def set_damper_position(self, damper_index: int, position: float):
        """Set damper position (0=closed, 1=fully open)."""
        if 0 <= damper_index < len(self.state.damper_positions):
            self.state.damper_positions[damper_index] = max(0.0, min(1.0, position))
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Update blower RPM (ramp up/down)
        if self.state.system_state in [SystemState.STARTING, SystemState.RUNNING]:
            ramp_rate = self.design_rpm / self.config.blower_ramp_time
            
            if self.state.blower_rpm < self.state.blower_target_rpm:
                self.state.blower_rpm = min(
                    self.state.blower_rpm + ramp_rate * dt,
                    self.state.blower_target_rpm
                )
            
            if self.state.blower_rpm >= self.state.blower_target_rpm * 0.99:
                self.state.system_state = SystemState.RUNNING
                
        elif self.state.system_state == SystemState.STOPPING:
            ramp_rate = self.design_rpm / self.config.blower_ramp_time
            self.state.blower_rpm = max(
                self.state.blower_rpm - ramp_rate * dt,
                0.0
            )
            
            if self.state.blower_rpm <= 0.01:
                self.state.system_state = SystemState.OFF
                self.state.blower_rpm = 0.0
        
        # Calculate flow rate (simplified fan law: Q ∝ N)
        damper_factor = np.prod(self.state.damper_positions)  # Combined damper effect
        self.state.flow_rate_m3_h = (
            self.flow_coefficient * self.state.blower_rpm * damper_factor
        )
        
        # Calculate pressure (simplified: P ∝ N²)
        rpm_ratio = self.state.blower_rpm / self.design_rpm if self.design_rpm > 0 else 0
        self.state.pressure_Pa = self.config.design_pressure_rise_Pa * rpm_ratio ** 2
        
        # Calculate power (simplified: W ∝ N³)
        design_power_kW = 5.0  # Typical blower power
        self.state.power_consumption_kW = design_power_kW * rpm_ratio ** 3
        self.state.total_energy_kWh += self.state.power_consumption_kW * dt / 3600.0
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        # Auto-start the system
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "blower_rpm": self.state.blower_rpm,
            "flow_rate_m3_h": self.state.flow_rate_m3_h,
            "pressure_Pa": self.state.pressure_Pa,
            "power_consumption_kW": self.state.power_consumption_kW,
            "total_energy_kWh": self.state.total_energy_kWh,
            "damper_positions": self.state.damper_positions.copy(),
        }
    
    def get_flow_velocity(self, position: np.ndarray) -> np.ndarray:
        """
        Get flow velocity at a position in the air system.
        
        Args:
            position: 3D position [x, y, z]
            
        Returns:
            Velocity vector [vx, vy, vz]
        """
        # Simplified analytical model
        # Assume flow follows the duct centerline
        flow_rate_m3_s = self.state.flow_rate_m3_h / 3600.0
        duct_area = np.pi * (self.assembly._duct_diameter / 2) ** 2
        
        if duct_area > 0 and flow_rate_m3_s > 0:
            velocity_magnitude = flow_rate_m3_s / duct_area
        else:
            velocity_magnitude = 0.0
        
        # Assume flow in +X direction through main duct
        return np.array([velocity_magnitude, 0.0, 0.0])


# =============================================================================
# FEED SYSTEM SIMULATOR
# =============================================================================

@dataclass
class FeedSystemConfig(BaseSimulationConfig):
    """Configuration for feed system simulation."""
    
    # Feed parameters
    feed_rate_kg_h: float = 500.0    # [kg/h] Target feed rate
    material_bulk_density: float = 500.0  # [kg/m³] Material density
    
    # Component speeds
    airlock_rpm: float = 20.0        # [RPM] Rotary airlock speed
    feeder_rpm: float = 60.0         # [RPM] Screw feeder speed
    deagg_rpm: float = 1500.0        # [RPM] Deagglomerator speed
    
    # Ramp times
    ramp_time: float = 2.0           # [s] Time to reach full speed
    
    # Particle tracking
    num_particles: int = 5000        # Number of particles to track
    injection_duration: float = 0.5  # [s] Duration over which to inject


@dataclass
class FeedSystemState(BaseSimulationState):
    """State for feed system simulation."""
    
    # Component speeds
    airlock_rpm: float = 0.0
    feeder_rpm: float = 0.0
    deagg_rpm: float = 0.0
    
    # Material flow
    mass_flow_rate_kg_h: float = 0.0
    hopper_mass_kg: float = 0.0
    
    # Particle tracking (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    is_active: Optional[wp.array] = None
    
    # Statistics
    particles_injected: int = 0
    particles_discharged: int = 0


class FeedSystemSimulator:
    """
    Simulator for the feed system (hopper + airlock + feeder + deagglomerator).
    
    Simulates:
    - Material discharge from hopper (gravity flow)
    - Rotary airlock volumetric metering
    - Screw feeder controlled dosing
    - Deagglomerator lump breaking
    
    Flow path:
    Hopper -> Airlock -> Screw Feeder -> Deagglomerator -> Outlet
    """
    
    def __init__(
        self,
        assembly: Any,  # FeedSystemAssembly
        config: FeedSystemConfig = None,
    ):
        """
        Initialize feed system simulator.
        
        Args:
            assembly: FeedSystemAssembly instance
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or FeedSystemConfig()
        self.state = FeedSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Initialize hopper mass
        self.state.hopper_mass_kg = self.assembly.params.hopper_capacity_kg
        
        # Setup system
        self._setup_system_parameters()
        self._allocate_arrays()
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.hopper = self.assembly.hopper
        self.airlock = self.assembly.airlock
        self.feeder = self.assembly.feeder
        self.deagglomerator = self.assembly.deagglomerator
        
        # Get component positions
        self.positions = self.assembly.get_component_positions()
        
        # Calculate volumetric flow from airlock
        self.airlock_pocket_volume = (
            np.pi * (self.airlock.params.rotor_diameter / 2) ** 2 *
            self.airlock.params.rotor_length / self.airlock.params.num_vanes
        )
    
    def _allocate_arrays(self):
        """Pre-allocate particle arrays."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
        
        # Temporary arrays
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
    
    def start(self):
        """Start the feed system."""
        self.state.system_state = SystemState.STARTING
    
    def stop(self):
        """Stop the feed system."""
        self.state.system_state = SystemState.STOPPING
    
    def inject_particles(self, n_inject: int):
        """Inject particles at hopper discharge."""
        if self.state.particles_injected >= self.config.num_particles:
            return
        
        n_inject = min(n_inject, self.config.num_particles - self.state.particles_injected)
        
        # Generate random particle sizes (log-normal distribution)
        rng = np.random.default_rng(self.state.step + 42)
        mean_diameter = 50e-6  # 50 microns
        diameters = rng.lognormal(
            mean=np.log(mean_diameter),
            sigma=0.5,
            size=n_inject
        ).astype(np.float32)
        diameters = np.clip(diameters, 5e-6, 500e-6)
        
        # Get hopper discharge position
        hopper_pos = np.array(self.positions['hopper'])
        discharge_port = self.hopper.ports['discharge']
        discharge_pos = hopper_pos + np.array(discharge_port.position)
        
        # Generate random positions within discharge area
        positions = np.zeros((n_inject, 3), dtype=np.float32)
        velocities = np.zeros((n_inject, 3), dtype=np.float32)
        
        discharge_radius = discharge_port.diameter / 2 * 0.8
        
        for i in range(n_inject):
            r = rng.random() * discharge_radius
            theta = rng.random() * 2 * np.pi
            
            positions[i, 0] = discharge_pos[0] + r * np.cos(theta)
            positions[i, 1] = discharge_pos[1]
            positions[i, 2] = discharge_pos[2] + r * np.sin(theta)
            
            # Initial velocity (small downward)
            velocities[i, 1] = -0.1
        
        # Copy to device
        start_idx = self.state.particles_injected
        
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters, dtype=float, device=self.device)
        
        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_inject)
        
        active_flags = wp.array(np.ones(n_inject, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_inject)
        
        self.state.particles_injected += n_inject
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Update component speeds (ramp up/down)
        if self.state.system_state == SystemState.STARTING:
            ramp_rate = 1.0 / self.config.ramp_time
            
            self.state.airlock_rpm = min(
                self.state.airlock_rpm + self.config.airlock_rpm * ramp_rate * dt,
                self.config.airlock_rpm
            )
            self.state.feeder_rpm = min(
                self.state.feeder_rpm + self.config.feeder_rpm * ramp_rate * dt,
                self.config.feeder_rpm
            )
            self.state.deagg_rpm = min(
                self.state.deagg_rpm + self.config.deagg_rpm * ramp_rate * dt,
                self.config.deagg_rpm
            )
            
            if (self.state.airlock_rpm >= self.config.airlock_rpm * 0.99 and
                self.state.feeder_rpm >= self.config.feeder_rpm * 0.99):
                self.state.system_state = SystemState.RUNNING
                
        elif self.state.system_state == SystemState.STOPPING:
            ramp_rate = 1.0 / self.config.ramp_time
            
            self.state.airlock_rpm = max(0, self.state.airlock_rpm - self.config.airlock_rpm * ramp_rate * dt)
            self.state.feeder_rpm = max(0, self.state.feeder_rpm - self.config.feeder_rpm * ramp_rate * dt)
            self.state.deagg_rpm = max(0, self.state.deagg_rpm - self.config.deagg_rpm * ramp_rate * dt)
            
            if self.state.airlock_rpm <= 0.01:
                self.state.system_state = SystemState.OFF
        
        # Calculate mass flow rate
        if self.state.system_state == SystemState.RUNNING:
            # Airlock volumetric flow
            vol_flow_m3_s = self.airlock_pocket_volume * (self.state.airlock_rpm / 60.0)
            self.state.mass_flow_rate_kg_h = (
                vol_flow_m3_s * self.config.material_bulk_density * 3600.0
            )
            
            # Deplete hopper
            mass_removed = self.state.mass_flow_rate_kg_h * dt / 3600.0
            self.state.hopper_mass_kg = max(0, self.state.hopper_mass_kg - mass_removed)
        else:
            self.state.mass_flow_rate_kg_h = 0.0
        
        # Inject particles if running
        if self.state.system_state == SystemState.RUNNING:
            if self.state.time < self.config.injection_duration:
                inject_rate = self.config.num_particles / self.config.injection_duration
                n_inject = int(inject_rate * dt) + (1 if np.random.random() < (inject_rate * dt % 1) else 0)
                if n_inject > 0:
                    self.inject_particles(n_inject)
        
        # Update particle positions (gravity + simple drag)
        n = self.state.particles_injected
        if n > 0:
            wp.launch(
                kernel=feed_particle_update_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.diameters,
                    self.state.is_active,
                    dt,
                    float(GRAVITY),
                    AirProperties.DENSITY,
                    AirProperties.DYNAMIC_VISCOSITY,
                ],
                device=self.device
            )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "airlock_rpm": self.state.airlock_rpm,
            "feeder_rpm": self.state.feeder_rpm,
            "deagg_rpm": self.state.deagg_rpm,
            "mass_flow_rate_kg_h": self.state.mass_flow_rate_kg_h,
            "hopper_mass_kg": self.state.hopper_mass_kg,
            "particles_injected": self.state.particles_injected,
        }


# =============================================================================
# CLASSIFICATION SYSTEM SIMULATOR
# =============================================================================

@dataclass
class ClassificationConfig(BaseSimulationConfig):
    """Configuration for classification system simulation."""
    
    # Particle parameters
    num_particles: int = 10000       # Number of particles to simulate
    injection_duration: float = 0.1  # [s] Duration over which to inject
    
    # Flow parameters
    inlet_velocity: float = 15.0     # [m/s] Air inlet velocity
    
    # Separation targets
    target_cut_size_um: float = 20.0  # [μm] Target cut size


@dataclass
class ClassificationState(BaseSimulationState):
    """State for classification system simulation."""
    
    # Particle arrays (on device)
    positions: Optional[wp.array] = None
    velocities: Optional[wp.array] = None
    diameters: Optional[wp.array] = None
    is_active: Optional[wp.array] = None
    
    # Statistics
    particles_injected: int = 0
    particles_in_fines: int = 0       # Fine fraction (protein)
    particles_in_coarse: int = 0      # Coarse fraction (starch)
    particles_in_cyclone_1: int = 0   # Primary cyclone
    particles_in_cyclone_2: int = 0   # Secondary cyclone
    particles_in_cyclone_3: int = 0   # Tertiary cyclone
    particles_in_bag_filter: int = 0  # Bag filter
    particles_active: int = 0
    
    # Size tracking
    fines_diameters: List[float] = field(default_factory=list)
    coarse_diameters: List[float] = field(default_factory=list)


class ClassificationSystemSimulator:
    """
    Simulator for the classification system.
    
    Simulates particle separation through:
    - Venturi Eductor (entrainment)
    - Zigzag Classifier (primary separation)
    - Multi-Cyclone System (staged collection)
    - Bag Filter (fine capture)
    
    Tracks particles and computes separation efficiency.
    """
    
    def __init__(
        self,
        assembly: Any,  # ClassificationSystemAssembly
        config: ClassificationConfig = None,
        material: Any = None,  # Optional ParticleMaterial
    ):
        """
        Initialize classification system simulator.
        
        Args:
            assembly: ClassificationSystemAssembly instance
            config: Simulation configuration
            material: Particle material definition (optional)
        """
        self.assembly = assembly
        self.config = config or ClassificationConfig()
        self.material = material
        self.state = ClassificationState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Setup system
        self._setup_system_parameters()
        self._allocate_arrays()
        self._setup_flow_field()
    
    def _setup_system_parameters(self):
        """Extract parameters from assembly."""
        self.venturi = self.assembly.venturi
        self.zigzag = self.assembly.zigzag
        self.multi_cyclone = self.assembly.multi_cyclone
        self.bag_filter = self.assembly.bag_filter
        
        # Get component positions
        self.component_positions = self.assembly.get_component_positions()
        
        # Get system bounds for domain setup
        self.bounds_min, self.bounds_max = self.assembly.get_bounds()
    
    def _allocate_arrays(self):
        """Pre-allocate particle arrays."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
        
        # Temporary arrays
        self._accelerations = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._fluid_velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
    
    def _setup_flow_field(self):
        """Setup analytical flow field for the classification system."""
        # Store flow parameters for the analytical model
        self.flow_params = {
            'inlet_velocity': self.config.inlet_velocity,
            'fluid_density': AirProperties.DENSITY,
            'fluid_viscosity': AirProperties.DYNAMIC_VISCOSITY,
        }
        
        # Venturi parameters
        venturi_pos = np.array(self.component_positions['venturi'])
        self.venturi_center = wp.vec3(venturi_pos[0], venturi_pos[1], venturi_pos[2])
        self.venturi_inlet_d = self.venturi.params.inlet_diameter
        self.venturi_throat_d = self.venturi.params.throat_diameter
        
        # Zigzag parameters
        zigzag_pos = np.array(self.component_positions['zigzag'])
        self.zigzag_center = wp.vec3(zigzag_pos[0], zigzag_pos[1], zigzag_pos[2])
        self.zigzag_width = self.zigzag.params.channel_width
        self.zigzag_height = self.zigzag.params.total_height
        
        # Cyclone parameters (primary)
        cyclone_pos = np.array(self.component_positions['multi_cyclone'])
        self.cyclone_center = wp.vec3(cyclone_pos[0], cyclone_pos[1], cyclone_pos[2])
    
    def start(self):
        """Start the classification system."""
        self.state.system_state = SystemState.RUNNING
    
    def stop(self):
        """Stop the classification system."""
        self.state.system_state = SystemState.OFF
    
    def inject_particles(self, n_inject: int):
        """Inject particles at the venturi inlet."""
        if self.state.particles_injected >= self.config.num_particles:
            return
        
        n_inject = min(n_inject, self.config.num_particles - self.state.particles_injected)
        
        # Generate particle sizes
        if self.material is not None:
            diameters = self.material.sample_diameters(n_inject, seed=self.state.step + 42)
        else:
            # Default: log-normal distribution
            rng = np.random.default_rng(self.state.step + 42)
            mean_d = 30e-6  # 30 microns
            diameters = rng.lognormal(np.log(mean_d), 0.7, n_inject).astype(np.float32)
            diameters = np.clip(diameters, 1e-6, 200e-6)
        
        # Get venturi solids inlet position
        venturi_pos = np.array(self.component_positions['venturi'])
        solids_inlet = self.venturi.ports['solids_inlet']
        inlet_pos = venturi_pos + np.array(solids_inlet.position)
        inlet_dir = np.array(solids_inlet.direction)
        inlet_radius = solids_inlet.diameter / 2 * 0.8
        
        # Generate positions at inlet
        rng = np.random.default_rng(self.state.step + 123)
        positions = np.zeros((n_inject, 3), dtype=np.float32)
        velocities = np.zeros((n_inject, 3), dtype=np.float32)
        
        for i in range(n_inject):
            # Random position within inlet area
            r = rng.random() * inlet_radius
            theta = rng.random() * 2 * np.pi
            
            # Calculate perpendicular directions
            if abs(inlet_dir[1]) < 0.9:
                perp1 = np.cross(inlet_dir, [0, 1, 0])
            else:
                perp1 = np.cross(inlet_dir, [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(inlet_dir, perp1)
            
            positions[i] = inlet_pos + r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            
            # Initial velocity in inlet direction
            velocities[i] = inlet_dir * 2.0  # 2 m/s entry velocity
        
        # Copy to device
        start_idx = self.state.particles_injected
        
        positions_wp = wp.array(positions, dtype=wp.vec3, device=self.device)
        velocities_wp = wp.array(velocities, dtype=wp.vec3, device=self.device)
        diameters_wp = wp.array(diameters.astype(np.float32), dtype=float, device=self.device)
        
        wp.copy(self.state.positions, positions_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.velocities, velocities_wp, dest_offset=start_idx, count=n_inject)
        wp.copy(self.state.diameters, diameters_wp, dest_offset=start_idx, count=n_inject)
        
        active_flags = wp.array(np.ones(n_inject, dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.copy(self.state.is_active, active_flags, dest_offset=start_idx, count=n_inject)
        
        self.state.particles_injected += n_inject
        self.state.particles_active += n_inject
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Inject particles during injection phase
        if self.state.time < self.config.injection_duration:
            inject_rate = self.config.num_particles / self.config.injection_duration
            n_inject = int(inject_rate * dt) + (1 if np.random.random() < (inject_rate * dt % 1) else 0)
            if n_inject > 0:
                self.inject_particles(n_inject)
        
        # Skip if no active particles
        if self.state.particles_active == 0 and self.state.particles_injected >= self.config.num_particles:
            self.state.time += dt
            self.state.step += 1
            return
        
        n = self.state.particles_injected
        if n == 0:
            self.state.time += dt
            self.state.step += 1
            return
        
        # Compute fluid velocities at particle positions
        wp.launch(
            kernel=classification_fluid_velocity_kernel,
            dim=n,
            inputs=[
                self.state.positions,
                self._fluid_velocities,
                self.state.is_active,
                self.venturi_center,
                self.zigzag_center,
                self.cyclone_center,
                self.venturi_inlet_d,
                float(self.config.inlet_velocity),
            ],
            device=self.device
        )
        
        # Compute accelerations
        wp.launch(
            kernel=classification_particle_accelerations,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self._fluid_velocities,
                self.state.is_active,
                self._accelerations,
                2500.0,  # Particle density (flour ~2500 kg/m³)
                AirProperties.DENSITY,
                AirProperties.DYNAMIC_VISCOSITY,
                float(GRAVITY),
                self.config.include_drag,
                self.config.include_gravity,
            ],
            device=self.device
        )
        
        # Integrate
        wp.launch(
            kernel=integrate_euler_kernel,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self._accelerations,
                self.state.is_active,
                dt,
            ],
            device=self.device
        )
        
        # Check particle collection zones
        wp.launch(
            kernel=classification_check_collection,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.is_active,
                self.zigzag_center,
                self.zigzag_height,
                self.cyclone_center,
            ],
            device=self.device
        )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        self.state.system_state = SystemState.RUNNING
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        # Copy data back to CPU
        is_active = self.state.is_active.numpy()
        diameters = self.state.diameters.numpy()
        
        # Count particles in each zone
        # is_active encoding: 1=active, -1=coarse, -2=fines, -3=cyclone1, -4=cyclone2, -5=cyclone3, -6=bagfilter
        n_coarse = np.sum(is_active == -1)
        n_fines = np.sum(is_active == -2)
        n_cyclone1 = np.sum(is_active == -3)
        n_cyclone2 = np.sum(is_active == -4)
        n_cyclone3 = np.sum(is_active == -5)
        n_bagfilter = np.sum(is_active == -6)
        n_active = np.sum(is_active == 1)
        
        # Get diameters by fraction
        coarse_d = diameters[is_active == -1] if n_coarse > 0 else np.array([])
        fines_d = diameters[is_active == -2] if n_fines > 0 else np.array([])
        
        return {
            "time": self.state.time,
            "steps": self.state.step,
            "particles_injected": self.state.particles_injected,
            "particles_active": int(n_active),
            "particles_coarse": int(n_coarse),
            "particles_fines": int(n_fines),
            "particles_cyclone_1": int(n_cyclone1),
            "particles_cyclone_2": int(n_cyclone2),
            "particles_cyclone_3": int(n_cyclone3),
            "particles_bag_filter": int(n_bagfilter),
            "separation_efficiency": float(n_fines / max(1, n_coarse + n_fines)),
            "mean_coarse_diameter_um": float(np.mean(coarse_d) * 1e6) if len(coarse_d) > 0 else 0.0,
            "mean_fines_diameter_um": float(np.mean(fines_d) * 1e6) if len(fines_d) > 0 else 0.0,
        }


# =============================================================================
# COMPLETE SYSTEM SIMULATOR
# =============================================================================

@dataclass
class CompleteSystemConfig(BaseSimulationConfig):
    """Configuration for complete system simulation."""
    
    # Air system config
    blower_rpm: float = 3000.0
    damper_positions: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Feed system config
    feed_rate_kg_h: float = 500.0
    airlock_rpm: float = 20.0
    feeder_rpm: float = 60.0
    deagg_rpm: float = 1500.0
    
    # Classification config
    num_particles: int = 10000
    injection_duration: float = 0.1
    inlet_velocity: float = 15.0
    
    # Startup sequence
    startup_sequence: List[str] = field(default_factory=lambda: ["air", "classification", "feed"])
    startup_delay: float = 0.5  # [s] Delay between system startups


@dataclass
class CompleteSystemState(BaseSimulationState):
    """State for complete system simulation."""
    
    # Subsystem states
    air_system_state: SystemState = SystemState.OFF
    feed_system_state: SystemState = SystemState.OFF
    classification_state: SystemState = SystemState.OFF
    
    # Flow coupling
    total_flow_rate_m3_h: float = 0.0
    system_pressure_Pa: float = 0.0
    
    # Material tracking
    feed_rate_kg_h: float = 0.0
    particles_in_system: int = 0
    
    # Energy
    total_power_kW: float = 0.0


class CompleteSystemSimulator:
    """
    Complete air classifier system simulator.
    
    Integrates and couples all three subsystems:
    - Air System: Provides motive force
    - Feed System: Delivers material
    - Classification System: Separates particles
    
    Handles startup sequence, inter-system coupling, and complete
    material balance tracking.
    """
    
    def __init__(
        self,
        assembly: Any,  # CompleteClassifierAssembly
        config: CompleteSystemConfig = None,
        material: Any = None,  # Optional ParticleMaterial
    ):
        """
        Initialize complete system simulator.
        
        Args:
            assembly: CompleteClassifierAssembly instance
            config: Simulation configuration
            material: Particle material definition
        """
        self.assembly = assembly
        self.config = config or CompleteSystemConfig()
        self.material = material
        self.state = CompleteSystemState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Create subsystem simulators
        self._create_subsystem_simulators()
        
        # Startup tracking
        self._startup_index = 0
        self._startup_timer = 0.0
    
    def _create_subsystem_simulators(self):
        """Create simulators for each subsystem."""
        # Get subsystems from assembly
        air_system = self.assembly.get_subsystem('air_system')
        feed_system = self.assembly.get_subsystem('feed_system')
        classification = self.assembly.get_subsystem('classification')
        
        # Air system simulator
        if air_system is not None:
            air_config = AirSystemConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                blower_rpm=self.config.blower_rpm,
                damper_positions=self.config.damper_positions,
                device=self.config.device,
            )
            self.air_simulator = AirSystemSimulator(air_system, air_config)
        else:
            self.air_simulator = None
        
        # Feed system simulator
        if feed_system is not None:
            feed_config = FeedSystemConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                feed_rate_kg_h=self.config.feed_rate_kg_h,
                airlock_rpm=self.config.airlock_rpm,
                feeder_rpm=self.config.feeder_rpm,
                deagg_rpm=self.config.deagg_rpm,
                device=self.config.device,
            )
            self.feed_simulator = FeedSystemSimulator(feed_system, feed_config)
        else:
            self.feed_simulator = None
        
        # Classification simulator
        if classification is not None:
            class_config = ClassificationConfig(
                dt=self.config.dt,
                duration=self.config.duration,
                num_particles=self.config.num_particles,
                injection_duration=self.config.injection_duration,
                inlet_velocity=self.config.inlet_velocity,
                device=self.config.device,
            )
            self.classification_simulator = ClassificationSystemSimulator(
                classification, class_config, self.material
            )
        else:
            self.classification_simulator = None
    
    def start(self):
        """Start the complete system (initiates startup sequence)."""
        self.state.system_state = SystemState.STARTING
        self._startup_index = 0
        self._startup_timer = 0.0
        # Start the first system immediately
        if self.config.startup_sequence:
            first_system = self.config.startup_sequence[0]
            self.start_system(first_system)
            self._startup_index = 1
    
    def start_system(self, system_name: str):
        """Start a specific subsystem."""
        if system_name == "air" and self.air_simulator:
            self.air_simulator.start()
            self.state.air_system_state = SystemState.STARTING
        elif system_name == "feed" and self.feed_simulator:
            self.feed_simulator.start()
            self.state.feed_system_state = SystemState.STARTING
        elif system_name == "classification" and self.classification_simulator:
            self.classification_simulator.start()
            self.state.classification_state = SystemState.RUNNING
    
    def stop_all(self):
        """Stop all subsystems."""
        if self.air_simulator:
            self.air_simulator.stop()
        if self.feed_simulator:
            self.feed_simulator.stop()
        if self.classification_simulator:
            self.classification_simulator.stop()
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        
        # Handle startup sequence
        if self._startup_index < len(self.config.startup_sequence):
            self._startup_timer += dt
            
            if self._startup_timer >= self.config.startup_delay:
                system_name = self.config.startup_sequence[self._startup_index]
                self.start_system(system_name)
                self._startup_index += 1
                self._startup_timer = 0.0
        
        # Step each subsystem
        if self.air_simulator:
            self.air_simulator.step()
            self.state.air_system_state = self.air_simulator.state.system_state
            
        if self.feed_simulator:
            self.feed_simulator.step()
            self.state.feed_system_state = self.feed_simulator.state.system_state
            
        if self.classification_simulator:
            # Couple air velocity to classification
            if self.air_simulator and self.air_simulator.state.system_state == SystemState.RUNNING:
                # Scale inlet velocity based on actual flow rate
                flow_ratio = self.air_simulator.state.flow_rate_m3_h / self.air_simulator.config.design_flow_rate_m3_h
                self.classification_simulator.config.inlet_velocity = self.config.inlet_velocity * flow_ratio
            
            self.classification_simulator.step()
            self.state.classification_state = self.classification_simulator.state.system_state
        
        # Update coupled state
        if self.air_simulator:
            self.state.total_flow_rate_m3_h = self.air_simulator.state.flow_rate_m3_h
            self.state.system_pressure_Pa = self.air_simulator.state.pressure_Pa
            self.state.total_power_kW = self.air_simulator.state.power_consumption_kW
            
        if self.feed_simulator:
            self.state.feed_rate_kg_h = self.feed_simulator.state.mass_flow_rate_kg_h
            
        if self.classification_simulator:
            self.state.particles_in_system = self.classification_simulator.state.particles_active
        
        # Update time
        self.state.time += dt
        self.state.step += 1
        
        # Update system state
        if (self.state.air_system_state == SystemState.RUNNING and
            self.state.classification_state == SystemState.RUNNING):
            self.state.system_state = SystemState.RUNNING
    
    def run(self, progress_callback=None):
        """Run the full simulation."""
        total_steps = self.config.num_steps
        
        # Auto-start the system
        self.start()
        
        for step in range(total_steps):
            self.step()
            
            if progress_callback and step % self.config.output_steps == 0:
                progress_callback(step, total_steps)
        
        wp.synchronize()
    
    def get_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results."""
        results = {
            "time": self.state.time,
            "steps": self.state.step,
            "system_state": self.state.system_state.value,
            "total_flow_rate_m3_h": self.state.total_flow_rate_m3_h,
            "system_pressure_Pa": self.state.system_pressure_Pa,
            "total_power_kW": self.state.total_power_kW,
            "feed_rate_kg_h": self.state.feed_rate_kg_h,
        }
        
        # Add subsystem results
        if self.air_simulator:
            results["air_system"] = self.air_simulator.get_results()
            
        if self.feed_simulator:
            results["feed_system"] = self.feed_simulator.get_results()
            
        if self.classification_simulator:
            results["classification"] = self.classification_simulator.get_results()
        
        return results


# =============================================================================
# WARP KERNELS
# =============================================================================

@wp.kernel
def feed_particle_update_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    diameters: wp.array(dtype=float),
    is_active: wp.array(dtype=wp.int32),
    dt: float,
    g: float,
    rho_f: float,
    mu_f: float,
):
    """Update particle positions in feed system with gravity and drag."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    
    # Gravity
    acc = wp.vec3(0.0, -g, 0.0)
    
    # Simple Stokes drag (assume low Re in feed system)
    if d > 1.0e-8:
        drag_coeff = 18.0 * mu_f / (2500.0 * d * d)  # Stokes drag
        acc = acc - vel * drag_coeff
    
    # Integrate
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos


@wp.kernel
def classification_fluid_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    fluid_velocities: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    venturi_center: wp.vec3,
    zigzag_center: wp.vec3,
    cyclone_center: wp.vec3,
    venturi_d: float,
    inlet_velocity: float,
):
    """Compute fluid velocities at particle positions in classification system."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        fluid_velocities[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    
    # Determine which region particle is in and apply appropriate flow model
    # Region detection based on Y position relative to component centers
    
    y_venturi = venturi_center[1]
    y_zigzag = zigzag_center[1]
    y_cyclone = cyclone_center[1]
    
    # Venturi region: strong vertical upward flow
    if pos[1] < y_zigzag - 0.5:
        # Accelerated flow through venturi throat
        v_mag = inlet_velocity * 1.5  # Throat acceleration
        fluid_velocities[tid] = wp.vec3(0.0, v_mag, 0.0)
        return
    
    # Zigzag region: oscillating upward flow
    if pos[1] < y_zigzag + 1.0:
        # Zigzag creates sinusoidal horizontal component
        phase = pos[1] * 10.0  # ~10 oscillations per meter
        v_horiz = inlet_velocity * 0.3 * wp.sin(phase)
        v_vert = inlet_velocity * 0.8
        fluid_velocities[tid] = wp.vec3(v_horiz, v_vert, 0.0)
        return
    
    # Cyclone region: swirling flow (Rankine vortex)
    dx = pos[0] - cyclone_center[0]
    dz = pos[2] - cyclone_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    cyclone_r = 0.15  # Approximate cyclone radius
    
    if r > 0.001:
        # Tangential velocity (free vortex outside core)
        v_tan = inlet_velocity * 0.8 * cyclone_r / wp.max(r, cyclone_r * 0.3)
        
        # Radial velocity (inward)
        v_rad = -inlet_velocity * 0.1
        
        # Axial velocity (downward near wall, upward in center)
        if r < cyclone_r * 0.5:
            v_axial = inlet_velocity * 0.5  # Upward in core
        else:
            v_axial = -inlet_velocity * 0.3  # Downward near wall
        
        # Convert to Cartesian
        cos_theta = dx / r
        sin_theta = dz / r
        
        vx = -v_tan * sin_theta + v_rad * cos_theta
        vz = v_tan * cos_theta + v_rad * sin_theta
        
        fluid_velocities[tid] = wp.vec3(vx, v_axial, vz)
    else:
        fluid_velocities[tid] = wp.vec3(0.0, inlet_velocity * 0.5, 0.0)


@wp.kernel
def classification_particle_accelerations(
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
    include_drag: bool,
    include_gravity: bool,
):
    """Compute particle accelerations in classification system."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        accelerations[tid] = wp.vec3(0.0, 0.0, 0.0)
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    d = diameters[tid]
    v_f = fluid_velocities[tid]
    
    acc = wp.vec3(0.0, 0.0, 0.0)
    
    # Drag force (Schiller-Naumann)
    if include_drag:
        v_rel = v_f - vel
        v_rel_mag = wp.length(v_rel)
        eps = 1.0e-10
        
        if v_rel_mag > eps and d > eps:
            # Reynolds number
            Re = rho_f * v_rel_mag * d / mu_f
            
            # Drag coefficient
            if Re < eps:
                Cd = 24.0 / eps
            else:
                Cd = (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))
            
            # Particle properties
            volume = (3.141592653589793 / 6.0) * d * d * d
            mass = rho_p * volume
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
def integrate_euler_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    dt: float,
):
    """Euler integration step."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    vel = velocities[tid]
    acc = accelerations[tid]
    pos = positions[tid]
    
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    
    velocities[tid] = new_vel
    positions[tid] = new_pos


@wp.kernel
def classification_check_collection(
    positions: wp.array(dtype=wp.vec3),
    is_active: wp.array(dtype=wp.int32),
    zigzag_center: wp.vec3,
    zigzag_height: float,
    cyclone_center: wp.vec3,
):
    """Check if particles have been collected in various zones."""
    tid = wp.tid()
    
    if is_active[tid] != 1:
        return
    
    pos = positions[tid]
    
    # Coarse outlet (below zigzag)
    if pos[1] < zigzag_center[1] - zigzag_height:
        is_active[tid] = -1  # Coarse fraction
        return
    
    # Check cyclone collection (particles that fall through dust outlet)
    # Primary cyclone
    dx = pos[0] - cyclone_center[0]
    dz = pos[2] - cyclone_center[2]
    r = wp.sqrt(dx * dx + dz * dz)
    
    if pos[1] < cyclone_center[1] - 0.5 and r < 0.1:
        is_active[tid] = -3  # Primary cyclone
        return
    
    # Fines (top exit)
    if pos[1] > zigzag_center[1] + zigzag_height + 1.0:
        is_active[tid] = -2  # Fines fraction
        return


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_air_system_simulator(
    assembly: Any,
    blower_rpm: float = 3000.0,
    damper_positions: List[float] = None,
    device: str = "cuda",
) -> AirSystemSimulator:
    """
    Create an air system simulator.
    
    Args:
        assembly: AirSystemAssembly instance
        blower_rpm: Blower operating speed [RPM]
        damper_positions: List of damper positions (0-1)
        device: Warp device
        
    Returns:
        AirSystemSimulator instance
    """
    config = AirSystemConfig(
        blower_rpm=blower_rpm,
        damper_positions=damper_positions or [1.0, 1.0],
        device=device,
    )
    return AirSystemSimulator(assembly, config)


def create_feed_system_simulator(
    assembly: Any,
    feed_rate_kg_h: float = 500.0,
    device: str = "cuda",
) -> FeedSystemSimulator:
    """
    Create a feed system simulator.
    
    Args:
        assembly: FeedSystemAssembly instance
        feed_rate_kg_h: Target feed rate [kg/h]
        device: Warp device
        
    Returns:
        FeedSystemSimulator instance
    """
    config = FeedSystemConfig(
        feed_rate_kg_h=feed_rate_kg_h,
        device=device,
    )
    return FeedSystemSimulator(assembly, config)


def create_classification_simulator(
    assembly: Any,
    num_particles: int = 10000,
    inlet_velocity: float = 15.0,
    material: Any = None,
    device: str = "cuda",
) -> ClassificationSystemSimulator:
    """
    Create a classification system simulator.
    
    Args:
        assembly: ClassificationSystemAssembly instance
        num_particles: Number of particles to simulate
        inlet_velocity: Air inlet velocity [m/s]
        material: ParticleMaterial instance
        device: Warp device
        
    Returns:
        ClassificationSystemSimulator instance
    """
    config = ClassificationConfig(
        num_particles=num_particles,
        inlet_velocity=inlet_velocity,
        device=device,
    )
    return ClassificationSystemSimulator(assembly, config, material)


def create_complete_system_simulator(
    assembly: Any,
    blower_rpm: float = 3000.0,
    feed_rate_kg_h: float = 500.0,
    num_particles: int = 10000,
    material: Any = None,
    device: str = "cuda",
) -> CompleteSystemSimulator:
    """
    Create a complete system simulator.
    
    Args:
        assembly: CompleteClassifierAssembly instance
        blower_rpm: Blower operating speed [RPM]
        feed_rate_kg_h: Target feed rate [kg/h]
        num_particles: Number of particles to simulate
        material: ParticleMaterial instance
        device: Warp device
        
    Returns:
        CompleteSystemSimulator instance
    """
    config = CompleteSystemConfig(
        blower_rpm=blower_rpm,
        feed_rate_kg_h=feed_rate_kg_h,
        num_particles=num_particles,
        device=device,
    )
    return CompleteSystemSimulator(assembly, config, material)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Entry point for command-line usage."""
    import sys
    
    print("=" * 70)
    print("Air Classifier System Simulators")
    print("=" * 70)
    print("\nAvailable simulators:")
    print("  1. AirSystemSimulator       - Blower, filter, dampers")
    print("  2. FeedSystemSimulator      - Hopper, airlock, feeder, deagglomerator")
    print("  3. ClassificationSimulator  - Venturi, zigzag, cyclones, bag filter")
    print("  4. CompleteSystemSimulator  - Full integrated system")
    print("\nUsage:")
    print("  from airclassifier.simulation.simulator import (")
    print("      create_air_system_simulator,")
    print("      create_feed_system_simulator,")
    print("      create_classification_simulator,")
    print("      create_complete_system_simulator,")
    print("  )")
    print("\nExample:")
    print("  from airclassifier.geometry.assembly import create_standard_air_system")
    print("  assembly = create_standard_air_system()")
    print("  sim = create_air_system_simulator(assembly)")
    print("  sim.run()")
    print("  results = sim.get_results()")
    print("=" * 70)


if __name__ == "__main__":
    main()
