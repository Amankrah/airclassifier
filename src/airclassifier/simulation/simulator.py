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



def main():
    """Entry point for command-line usage."""
    import sys
    
    print("=" * 70)
    print("Air Classifier System Simulators")
    print("=" * 70)
    print("\nAvailable simulators:")
    print("  1. AirSystemSimulator       - Blower, filter, dampers")
    print("=" * 70)


if __name__ == "__main__":
    main()
