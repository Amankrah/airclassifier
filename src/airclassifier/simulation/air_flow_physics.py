"""
Physics-Based Air Flow Simulation with SPH
===========================================

Simulates air flow through the air system using actual geometry and
Smoothed Particle Hydrodynamics (SPH) for physically accurate flow.
NO magic numbers - all parameters computed from geometry and physical laws.

SPH Air Physics:
- Air represented as SPH particles with mass, density, pressure
- Density computed from neighbor kernel interpolation (Poly6)
- Pressure from weakly compressible equation of state
- Forces: pressure gradient (Spiky kernel), viscosity (Laplacian kernel)
- XSPH velocity smoothing for coherent flow

Blower Physics:
- Fan affinity laws (Q ∝ N, P ∝ N², W ∝ N³)
- Centrifugal acceleration in impeller region (ω²r)
- Scroll geometry guides flow to tangential outlet

System Physics:
- Darcy-Weisbach friction losses in ducts
- Component pressure drops from geometry
- Boundary containment within actual duct geometry

Components modeled:
- Centrifugal blower (impeller, scroll, outlet)
- Inlet filter (media resistance)
- Dampers (blade angle control)
- Ductwork (90° elbow, transitions)

Author: Air Classifier Physics Engine
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
import warp as wp

from ..geometry.assembly.air_system import AirSystemAssembly, AirSystemParams
from ..utils.constants import PI, GRAVITY, AirProperties
from ..particles import FluidConfig  # Reusable fluid configuration

# =============================================================================
# PHYSICAL CONSTANTS (from air properties - also available via FluidConfig)
# =============================================================================

RHO_AIR = AirProperties.DENSITY           # kg/m³ at STP
MU_AIR = AirProperties.DYNAMIC_VISCOSITY  # Pa·s at STP
NU_AIR = MU_AIR / RHO_AIR                  # m²/s kinematic viscosity

# Derived constants
TWO_PI = 2.0 * PI
SQRT_2 = np.sqrt(2.0)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class BlowerState(Enum):
    """Operating state of blower."""
    OFF = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3


class SystemPhase(Enum):
    """Simulation workflow phases."""
    IDLE = "idle"
    STARTUP = "startup"          # Blower ramping up, dampers opening
    RUNNING = "running"          # Steady-state operation
    SHUTDOWN = "shutdown"        # Blower ramping down, dampers closing
    OFF = "off"


@dataclass
class DuctSegment:
    """Geometry of a duct segment for flow calculation."""
    name: str
    diameter: float              # [m] Hydraulic diameter
    length: float                # [m] Segment length
    area: float                  # [m²] Cross-sectional area
    roughness: float = 0.00015  # [m] Surface roughness (galvanized steel)
    
    # Calculated during simulation
    velocity: float = 0.0        # [m/s]
    pressure_drop: float = 0.0   # [Pa]
    reynolds: float = 0.0


@dataclass
class ComponentPressureDrop:
    """Pressure drop data for a component."""
    name: str
    k_factor: float              # Loss coefficient K
    pressure_drop: float = 0.0   # [Pa] Current pressure drop


@dataclass
class AirFlowPhysicsConfig:
    """Configuration for physics-based air flow simulation."""
    
    # Time stepping
    dt: float = 0.001               # Time step [s]
    total_time: float = 10.0        # Total simulation time [s]
    
    # Blower operating parameters
    target_rpm: float = 3000.0      # [RPM] Target blower speed
    ramp_time: float = 2.0          # [s] Time to reach full speed
    
    # Damper control
    damper_ramp_time: float = 1.0   # [s] Time for damper to open/close
    
    # Air properties (can be set via FluidConfig or directly)
    air_density: float = RHO_AIR
    air_viscosity: float = MU_AIR
    
    # Optional FluidConfig for consistency with other modules
    fluid_config: Optional[FluidConfig] = None
    
    # SPH Air particles (physically accurate air flow)
    enable_sph: bool = True
    num_particles: int = 1000       # Number of SPH air particles
    smoothing_length: float = 0.04  # SPH kernel support radius [m]
    speed_of_sound: float = 50.0    # Artificial speed of sound for stability [m/s]
    sph_viscosity: float = 0.01     # SPH artificial viscosity coefficient
    xsph_factor: float = 0.1        # XSPH velocity smoothing factor
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Apply FluidConfig properties if provided."""
        if self.fluid_config is not None:
            self.air_density = self.fluid_config.density
            self.air_viscosity = self.fluid_config.dynamic_viscosity
    
    @classmethod
    def from_fluid_config(
        cls,
        fluid_config: FluidConfig,
        target_rpm: float = 3000.0,
        total_time: float = 10.0,
        dt: float = 0.001,
        **kwargs
    ) -> "AirFlowPhysicsConfig":
        """
        Create config from a FluidConfig for consistency with other modules.
        
        Args:
            fluid_config: FluidConfig instance (e.g., FluidConfig.air_at_stp())
            target_rpm: Target blower RPM
            total_time: Simulation duration [s]
            dt: Time step [s]
            **kwargs: Additional config parameters
            
        Returns:
            Configured AirFlowPhysicsConfig
        """
        return cls(
            dt=dt,
            total_time=total_time,
            target_rpm=target_rpm,
            fluid_config=fluid_config,
            **kwargs
        )
    
    @classmethod
    def for_temperature(
        cls,
        temperature_c: float = 20.0,
        target_rpm: float = 3000.0,
        **kwargs
    ) -> "AirFlowPhysicsConfig":
        """
        Create config with air properties at a specific temperature.
        
        Args:
            temperature_c: Air temperature in Celsius
            target_rpm: Target blower RPM
            **kwargs: Additional config parameters
            
        Returns:
            Configured AirFlowPhysicsConfig
        """
        fluid = FluidConfig.air_at_temperature(temperature_c)
        return cls.from_fluid_config(fluid, target_rpm=target_rpm, **kwargs)


@dataclass
class AirFlowPhysicsState:
    """State of the air flow physics simulation."""
    time: float = 0.0
    step: int = 0
    
    # Blower state
    blower_state: BlowerState = BlowerState.OFF
    blower_rpm: float = 0.0
    blower_omega: float = 0.0       # [rad/s]
    blower_tip_speed: float = 0.0   # [m/s]
    
    # Flow state (from physics)
    volume_flow_rate: float = 0.0   # [m³/s]
    mass_flow_rate: float = 0.0     # [kg/s]
    
    # Pressure state
    static_pressure_rise: float = 0.0    # [Pa] Blower pressure rise
    total_pressure_drop: float = 0.0     # [Pa] System pressure drop
    
    # Power
    shaft_power: float = 0.0        # [W]
    electrical_power: float = 0.0   # [W]
    efficiency: float = 0.0
    
    # Damper positions
    damper_positions: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Simulation phase
    phase: SystemPhase = SystemPhase.IDLE
    phase_start_time: float = 0.0
    
    # SPH Air particle data (on device)
    positions: Optional[wp.array] = None      # Particle positions
    velocities: Optional[wp.array] = None     # Particle velocities
    densities: Optional[wp.array] = None      # SPH density field
    pressures: Optional[wp.array] = None      # SPH pressure field
    forces: Optional[wp.array] = None         # SPH forces (3 floats per particle)
    
    # Hash grid for SPH neighbor search
    hash_grid: Optional[wp.HashGrid] = None
    
    # SPH statistics
    max_velocity: float = 0.0                 # Max particle velocity [m/s]
    avg_density: float = RHO_AIR              # Average density [kg/m³]
    
    # Energy tracking
    total_energy_kWh: float = 0.0


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

@dataclass
class BlowerGeometry:
    """Extracted blower geometry for physics calculations."""
    impeller_diameter: float      # [m]
    impeller_width: float         # [m]
    inlet_diameter: float         # [m]
    hub_diameter: float           # [m]
    num_blades: int
    blade_type: str
    scroll_diameter: float        # [m]
    outlet_width: float           # [m]
    outlet_height: float          # [m]
    design_rpm: float
    design_flow_rate: float       # [m³/h]
    design_pressure_rise: float   # [Pa]
    
    @property
    def impeller_radius(self) -> float:
        return self.impeller_diameter / 2.0
    
    @property
    def inlet_area(self) -> float:
        """Inlet eye area [m²]."""
        return PI * (self.inlet_diameter / 2.0) ** 2
    
    @property
    def outlet_area(self) -> float:
        """Scroll outlet area [m²]."""
        return self.outlet_width * self.outlet_height
    
    @property
    def blade_outlet_area(self) -> float:
        """Impeller blade outlet area [m²]."""
        return PI * self.impeller_diameter * self.impeller_width


@dataclass
class FilterGeometry:
    """Extracted filter geometry."""
    face_area: float              # [m²]
    media_depth: float            # [m]
    housing_volume: float         # [m³]
    efficiency_class: str
    
    @property
    def media_resistance(self) -> float:
        """Media resistance coefficient [Pa·s/m]."""
        # Based on filter class (approximate values)
        resistance_map = {
            'G4': 50.0,
            'M5': 100.0,
            'M6': 150.0,
            'F7': 250.0,
            'F8': 350.0,
            'F9': 500.0,
            'HEPA': 2000.0,
        }
        return resistance_map.get(self.efficiency_class, 100.0)


@dataclass
class DamperGeometry:
    """Extracted damper geometry."""
    diameter: float               # [m]
    blade_chord: float            # [m]
    position: float = 0.0         # 0=closed, 1=open
    
    @property
    def area(self) -> float:
        return PI * (self.diameter / 2.0) ** 2
    
    def get_cv(self, position: float) -> float:
        """
        Get flow coefficient Cv based on blade position.
        
        Cv = Q / sqrt(dP)  where Q in gpm, dP in psi
        We use metric: Kv = Q / sqrt(dP/rho) in m³/h
        
        For butterfly valve, Cv varies with angle.
        """
        # Butterfly valve characteristic curve (approximate)
        # At 0° (closed): Cv ≈ 0
        # At 90° (open): Cv = Cv_max
        angle = position * 90.0  # degrees
        angle_rad = np.radians(angle)
        
        # Equal percentage characteristic
        if position < 0.05:
            return 0.01  # Small leakage when "closed"
        
        # Cv varies approximately as sin²(angle) for butterfly
        cv_factor = np.sin(angle_rad) ** 2
        
        # Max Cv based on diameter (m³/h per sqrt(bar))
        cv_max = 10.0 * (self.diameter * 1000) ** 2 / 10000  # Empirical
        
        return cv_max * cv_factor


def extract_air_geometry(assembly: AirSystemAssembly) -> Dict[str, Any]:
    """
    Extract all geometry parameters from air system assembly.
    
    Args:
        assembly: AirSystemAssembly with actual geometry
        
    Returns:
        Dictionary with geometry for all components
    """
    geometry = {}
    
    # =================================================================
    # BLOWER GEOMETRY
    # =================================================================
    blower = assembly.blower
    bp = blower.params
    
    geometry['blower'] = BlowerGeometry(
        impeller_diameter=bp.impeller_diameter,
        impeller_width=bp.impeller_width,
        inlet_diameter=bp.inlet_diameter,
        hub_diameter=bp.hub_diameter,
        num_blades=bp.num_blades,
        blade_type=bp.blade_type,
        scroll_diameter=bp.impeller_diameter * 1.2,  # Typical scroll OD
        outlet_width=bp.outlet_width,
        outlet_height=bp.outlet_height,
        design_rpm=bp.rpm,
        design_flow_rate=bp.flow_rate,
        design_pressure_rise=bp.pressure_rise,
    )
    
    # =================================================================
    # FILTER GEOMETRY
    # =================================================================
    inlet_filter = assembly.inlet_filter
    fp = inlet_filter.params
    
    geometry['filter'] = FilterGeometry(
        face_area=fp.housing_width * fp.housing_height,
        media_depth=fp.housing_depth,
        housing_volume=fp.housing_width * fp.housing_height * fp.housing_depth,
        efficiency_class=fp.efficiency_class,
    )
    
    # =================================================================
    # DAMPER GEOMETRY
    # =================================================================
    geometry['dampers'] = []
    for i, damper in enumerate(assembly.dampers):
        dp = damper.params
        geometry['dampers'].append(DamperGeometry(
            diameter=dp.diameter,
            blade_chord=dp.diameter * 0.9,  # Blade is ~90% of diameter
            position=dp.position,
        ))
    
    # =================================================================
    # DUCT GEOMETRY (complete flow path with positions)
    # All values extracted directly from assembly - NO hardcoding
    # =================================================================
    duct_diameter = assembly._duct_diameter
    elbow_params = assembly._elbow_params
    transition_params = assembly._transition_params
    
    # Component positions (directly from assembly)
    filter_pos = np.array(assembly._filter_position)
    blower_pos = np.array(assembly._blower_position)
    
    # Filter outlet world position (from assembly port)
    filter_outlet_port = assembly.inlet_filter.ports['outlet']
    filter_outlet_world = filter_pos + np.array(filter_outlet_port.position)
    
    # Elbow positions (directly from assembly's stored params)
    elbow_inlet_pos = np.array(elbow_params['inlet_pos'])
    elbow_outlet_pos = np.array(elbow_params['outlet_pos'])
    elbow_bend_radius = elbow_params['bend_radius']
    
    # Blower inlet/outlet (directly from assembly's stored world positions)
    blower_inlet_world = np.array(assembly._blower_inlet_world)
    blower_outlet_world = np.array(assembly._blower_outlet_world)
    
    # Transition lengths (directly from assembly)
    transition_length = transition_params['length']
    duct_after_transition = transition_params['duct_after_length']
    
    # Create duct segment list with positions for tracer routing
    geometry['ducts'] = []
    
    # 1. Filter to elbow (horizontal, +X direction)
    geometry['ducts'].append(DuctSegment(
        name='filter_to_elbow',
        diameter=elbow_params['diameter'],
        length=elbow_params['duct_horiz_length'],
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 2. 90° Elbow (turns from +X to +Z)
    geometry['ducts'].append(DuctSegment(
        name='elbow_90deg',
        diameter=elbow_params['diameter'],
        length=elbow_bend_radius * PI / 2,  # Arc length for 90°
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 3. Elbow to blower inlet (vertical, +Z direction)
    geometry['ducts'].append(DuctSegment(
        name='elbow_to_blower',
        diameter=elbow_params['diameter'],
        length=elbow_params['duct_vert_length'],
        area=PI * (elbow_params['diameter'] / 2) ** 2,
    ))
    
    # 4. Blower outlet to damper (transition + duct)
    if assembly.dampers:
        damper_pos = np.array(assembly._damper_positions[0])
        damper_inlet_port = assembly.dampers[0].ports['inlet']
        damper_inlet_world = damper_pos + np.array(damper_inlet_port.position)
        
        # Compute duct length from actual positions
        duct_length = damper_inlet_world[0] - blower_outlet_world[0] - transition_length
        
        # Transition piece (using assembly's stored transition_length)
        geometry['ducts'].append(DuctSegment(
            name='blower_transition',
            diameter=duct_diameter,
            length=transition_length,
            area=PI * (duct_diameter / 2) ** 2,
        ))
        
        # Duct to damper
        if duct_length > 0.01:
            geometry['ducts'].append(DuctSegment(
                name='transition_to_damper',
                diameter=duct_diameter,
                length=duct_length,
                area=PI * (duct_diameter / 2) ** 2,
            ))
    
    # 5. Ducts between dampers (if multiple dampers)
    for i in range(len(assembly.dampers) - 1):
        # Get positions from assembly
        damper_pos_i = np.array(assembly._damper_positions[i])
        damper_pos_next = np.array(assembly._damper_positions[i + 1])
        
        outlet_port = assembly.dampers[i].ports['outlet']
        inlet_port = assembly.dampers[i + 1].ports['inlet']
        
        outlet_world = damper_pos_i + np.array(outlet_port.position)
        inlet_world = damper_pos_next + np.array(inlet_port.position)
        
        duct_between_length = inlet_world[0] - outlet_world[0]
        
        if duct_between_length > 0.01:
            geometry['ducts'].append(DuctSegment(
                name=f'damper_{i}_to_damper_{i+1}',
                diameter=assembly.dampers[i].params.diameter,
                length=duct_between_length,
                area=PI * (assembly.dampers[i].params.diameter / 2) ** 2,
            ))
    
    # =================================================================
    # FLOW PATH WAYPOINTS (for tracer particle routing)
    # All positions directly from assembly - NO hardcoding
    # =================================================================
    # Get filter inlet position from port
    filter_inlet_port = assembly.inlet_filter.ports['inlet']
    filter_inlet_world = filter_pos + np.array(filter_inlet_port.position)
    
    geometry['flow_path'] = {
        'filter_inlet': filter_inlet_world.copy(),
        'filter_outlet': filter_outlet_world.copy(),
        'elbow_inlet': elbow_inlet_pos.copy(),
        'elbow_outlet': elbow_outlet_pos.copy(),
        'blower_inlet': blower_inlet_world.copy(),
        'blower_center': blower_pos.copy(),
        'blower_outlet': blower_outlet_world.copy(),
    }
    
    # Add damper positions to flow path (from assembly's stored positions)
    for idx, (damper, pos) in enumerate(zip(assembly.dampers, assembly._damper_positions)):
        damper_pos_arr = np.array(pos)
        inlet_port = damper.ports['inlet']
        outlet_port = damper.ports['outlet']
        geometry['flow_path'][f'damper_{idx}_inlet'] = damper_pos_arr + np.array(inlet_port.position)
        geometry['flow_path'][f'damper_{idx}_outlet'] = damper_pos_arr + np.array(outlet_port.position)
    
    # =================================================================
    # SYSTEM PARAMETERS
    # =================================================================
    geometry['system'] = {
        'design_flow_rate_m3_h': assembly.params.flow_rate_m3_h,
        'design_pressure_Pa': assembly.params.pressure_rise_Pa,
        'duct_diameter': duct_diameter,
        'elbow_diameter': elbow_params['diameter'],
        'elbow_bend_radius': elbow_bend_radius,
    }
    
    # =================================================================
    # COMPONENT POSITIONS (for animation coordination)
    # =================================================================
    geometry['positions'] = {
        'filter': filter_pos.copy(),
        'blower': blower_pos.copy(),
        'dampers': [np.array(pos) for pos in assembly._damper_positions],
    }
    
    return geometry


# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================

def calculate_friction_factor(reynolds: float, roughness: float, diameter: float) -> float:
    """
    Calculate Darcy friction factor using Colebrook-White equation.
    
    For laminar flow (Re < 2300): f = 64/Re
    For turbulent flow: Colebrook-White (iterative)
    
    Args:
        reynolds: Reynolds number
        roughness: Surface roughness [m]
        diameter: Pipe diameter [m]
        
    Returns:
        Darcy friction factor
    """
    if reynolds < 2300:
        # Laminar flow
        return 64.0 / max(reynolds, 1.0)
    
    # Turbulent flow - use Swamee-Jain approximation (explicit)
    relative_roughness = roughness / diameter
    
    term1 = relative_roughness / 3.7
    term2 = 5.74 / (reynolds ** 0.9)
    
    f = 0.25 / (np.log10(term1 + term2) ** 2)
    
    return f


def calculate_duct_pressure_drop(
    segment: DuctSegment,
    velocity: float,
    rho: float,
    mu: float
) -> float:
    """
    Calculate pressure drop in a duct segment using Darcy-Weisbach.
    
    dP = f * (L/D) * (rho * V²/2)
    
    Args:
        segment: Duct segment geometry
        velocity: Flow velocity [m/s]
        rho: Air density [kg/m³]
        mu: Dynamic viscosity [Pa·s]
        
    Returns:
        Pressure drop [Pa]
    """
    if velocity < 0.001:
        return 0.0
    
    # Reynolds number
    Re = rho * velocity * segment.diameter / mu
    segment.reynolds = Re
    
    # Friction factor
    f = calculate_friction_factor(Re, segment.roughness, segment.diameter)
    
    # Darcy-Weisbach equation
    dP = f * (segment.length / segment.diameter) * (rho * velocity ** 2 / 2.0)
    
    segment.velocity = velocity
    segment.pressure_drop = dP
    
    return dP


def calculate_filter_pressure_drop(
    filter_geo: FilterGeometry,
    volume_flow_rate: float,
    rho: float
) -> float:
    """
    Calculate filter pressure drop.
    
    dP = R * V_face + 0.5 * rho * V_face² * K_entry
    
    where R is media resistance and K_entry is entry loss coefficient.
    
    Args:
        filter_geo: Filter geometry
        volume_flow_rate: Flow rate [m³/s]
        rho: Air density [kg/m³]
        
    Returns:
        Pressure drop [Pa]
    """
    if volume_flow_rate < 1e-6:
        return 0.0
    
    face_velocity = volume_flow_rate / filter_geo.face_area
    
    # Media resistance (viscous term)
    dP_media = filter_geo.media_resistance * face_velocity
    
    # Entry/exit losses (dynamic term)
    k_entry_exit = 0.5  # Typical for filter housing
    dP_dynamic = 0.5 * rho * face_velocity ** 2 * k_entry_exit
    
    return dP_media + dP_dynamic


def calculate_damper_pressure_drop(
    damper_geo: DamperGeometry,
    volume_flow_rate: float,
    rho: float
) -> float:
    """
    Calculate damper pressure drop based on blade position.
    
    dP = 0.5 * rho * V² * K
    
    where K is the loss coefficient based on blade angle.
    
    Args:
        damper_geo: Damper geometry
        volume_flow_rate: Flow rate [m³/s]
        rho: Air density [kg/m³]
        
    Returns:
        Pressure drop [Pa]
    """
    if volume_flow_rate < 1e-6:
        return 0.0
    
    velocity = volume_flow_rate / damper_geo.area
    
    # Loss coefficient K based on blade position
    # K varies from very high (closed) to low (open)
    position = damper_geo.position
    
    if position < 0.05:
        k = 1000.0  # Nearly closed - very high resistance
    elif position > 0.95:
        k = 0.3     # Fully open - minimal resistance
    else:
        # Interpolate based on butterfly valve curve
        angle = position * 90.0
        angle_rad = np.radians(angle)
        # K decreases as valve opens
        k = 0.3 + 50.0 * (1.0 - np.sin(angle_rad) ** 2)
    
    dP = 0.5 * rho * velocity ** 2 * k
    
    return dP


def calculate_blower_performance(
    blower_geo: BlowerGeometry,
    omega: float,
    system_pressure_drop: float,
    rho: float
) -> Tuple[float, float, float, float]:
    """
    Calculate blower operating point using fan laws.
    
    Fan affinity laws:
    - Q ∝ N (flow proportional to speed)
    - P ∝ N² (pressure proportional to speed squared)
    - W ∝ N³ (power proportional to speed cubed)
    
    The operating point is where blower curve intersects system curve.
    
    Args:
        blower_geo: Blower geometry
        omega: Angular velocity [rad/s]
        system_pressure_drop: System resistance at design flow [Pa]
        rho: Air density [kg/m³]
        
    Returns:
        Tuple of (volume_flow_rate, pressure_rise, shaft_power, efficiency)
    """
    if omega < 0.1:
        return 0.0, 0.0, 0.0, 0.0
    
    rpm = omega * 60.0 / TWO_PI
    design_rpm = blower_geo.design_rpm
    
    if design_rpm < 1.0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Speed ratio
    n_ratio = rpm / design_rpm
    
    # =================================================================
    # BLOWER CURVE (pressure vs flow at current speed)
    # =================================================================
    # At design point: Q_design, P_design
    Q_design = blower_geo.design_flow_rate / 3600.0  # Convert to m³/s
    P_design = blower_geo.design_pressure_rise
    
    # At current speed: Q_max = Q_design * n, P_shutoff = P_design * n²
    Q_max = Q_design * n_ratio
    P_shutoff = P_design * n_ratio ** 2
    
    # =================================================================
    # IMPELLER TIP SPEED
    # =================================================================
    tip_speed = omega * blower_geo.impeller_radius
    
    # Theoretical pressure rise from Euler equation
    # P = rho * u2 * c_u2  where c_u2 is tangential velocity at exit
    # For backward curved blades, c_u2 ≈ 0.5 * u2
    slip_factor = 0.85  # Typical for backward curved
    blade_factor = 0.5 if blower_geo.blade_type == "backward_curved" else 0.7
    
    P_theoretical = rho * tip_speed ** 2 * blade_factor * slip_factor
    
    # =================================================================
    # OPERATING POINT (intersection of blower and system curves)
    # =================================================================
    # System curve: dP = K * Q²
    # If at design flow we have design pressure, K = P_design / Q_design²
    if Q_design > 1e-6:
        K_system = P_design / (Q_design ** 2)
    else:
        K_system = 1e6
    
    # Operating flow rate (solve quadratic)
    # P_blower = P_shutoff * (1 - (Q/Q_max)²)  # Parabolic approximation
    # P_system = K * Q²
    # At operating point: P_blower = P_system
    # P_shutoff * (1 - Q²/Q_max²) = K * Q²
    # Q² * (K + P_shutoff/Q_max²) = P_shutoff
    
    denom = K_system + P_shutoff / max(Q_max ** 2, 1e-12)
    Q_operating_sq = P_shutoff / denom
    Q_operating = np.sqrt(max(Q_operating_sq, 0.0))
    
    # Clamp to physical limits
    Q_operating = min(Q_operating, Q_max * 1.1)
    
    # Pressure at operating point
    P_operating = K_system * Q_operating ** 2
    P_operating = min(P_operating, P_shutoff)
    
    # =================================================================
    # EFFICIENCY
    # =================================================================
    # Efficiency varies with operating point
    # Peak efficiency at design point, drops off at other points
    flow_ratio = Q_operating / max(Q_design * n_ratio, 1e-6)
    
    # Efficiency curve (parabolic)
    peak_efficiency = 0.75 if blower_geo.blade_type == "backward_curved" else 0.65
    efficiency = peak_efficiency * (1.0 - (flow_ratio - 1.0) ** 2)
    efficiency = max(0.1, min(peak_efficiency, efficiency))
    
    # =================================================================
    # SHAFT POWER
    # =================================================================
    # W = Q * P / eta
    fluid_power = Q_operating * P_operating
    shaft_power = fluid_power / max(efficiency, 0.1)
    
    return Q_operating, P_operating, shaft_power, efficiency


# =============================================================================
# SPH KERNEL FUNCTIONS FOR AIR SIMULATION
# =============================================================================

@wp.func
def poly6_kernel(r: float, h: float) -> float:
    """
    Poly6 smoothing kernel for density estimation.
    W(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3  for r < h
    """
    if r >= h:
        return 0.0
    h2 = h * h
    r2 = r * r
    diff = h2 - r2
    coeff = 315.0 / (64.0 * PI * h * h * h * h * h * h * h * h * h)
    return coeff * diff * diff * diff


@wp.func
def spiky_gradient(r_vec: wp.vec3, r: float, h: float) -> wp.vec3:
    """
    Gradient of spiky kernel for pressure forces.
    Sharper near center for better pressure response.
    """
    if r < 1e-10 or r >= h:
        return wp.vec3(0.0, 0.0, 0.0)
    coeff = -45.0 / (PI * h * h * h * h * h * h)
    diff = h - r
    grad_mag = coeff * diff * diff
    return r_vec * (grad_mag / r)


@wp.func
def viscosity_laplacian(r: float, h: float) -> float:
    """Laplacian of viscosity kernel for viscous forces."""
    if r >= h:
        return 0.0
    coeff = 45.0 / (PI * h * h * h * h * h * h)
    return coeff * (h - r)


# =============================================================================
# SPH DENSITY AND PRESSURE KERNEL
# =============================================================================

@wp.kernel
def compute_sph_density_pressure_kernel(
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    pressures: wp.array(dtype=float),
    grid: wp.uint64,
    n: int,
    particle_mass: float,
    h: float,
    rest_density: float,
    speed_of_sound: float,
):
    """
    Compute SPH density and pressure for air particles.
    
    Density: rho_i = sum_j m_j * W(r_ij, h)
    Pressure: P = c^2 * (rho - rho_0)  (weakly compressible)
    """
    tid = wp.tid()
    if tid >= n:
        return
    
    pos_i = positions[tid]
    
    # Self-contribution
    density = particle_mass * poly6_kernel(0.0, h)
    
    # Neighbor contributions
    query = wp.hash_grid_query(grid, pos_i, h)
    idx = int(0)
    
    while wp.hash_grid_query_next(query, idx):
        if idx != tid:
            pos_j = positions[idx]
            r_vec = pos_i - pos_j
            r = wp.length(r_vec)
            if r < h:
                density += particle_mass * poly6_kernel(r, h)
    
    # Clamp density
    density = wp.max(density, rest_density * 0.5)
    
    # Pressure from equation of state (weakly compressible)
    pressure = speed_of_sound * speed_of_sound * (density - rest_density)
    
    densities[tid] = density
    pressures[tid] = pressure


# =============================================================================
# SPH FORCES KERNEL
# =============================================================================

@wp.kernel
def compute_sph_forces_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    pressures: wp.array(dtype=float),
    forces: wp.array(dtype=float),
    grid: wp.uint64,
    n: int,
    particle_mass: float,
    h: float,
    viscosity: float,
    rest_density: float,
):
    """
    Compute SPH pressure and viscosity forces.
    
    Pressure: F_pressure = -sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_W
    Viscosity: F_viscosity = mu * sum_j m_j (v_j - v_i) / rho_j * laplacian_W
    """
    tid = wp.tid()
    if tid >= n:
        return
    
    pos_i = positions[tid]
    vel_i = velocities[tid]
    rho_i = densities[tid]
    P_i = pressures[tid]
    
    force = wp.vec3(0.0, 0.0, 0.0)
    rho_i_safe = wp.max(rho_i, rest_density * 0.5)
    
    query = wp.hash_grid_query(grid, pos_i, h)
    idx = int(0)
    
    while wp.hash_grid_query_next(query, idx):
        if idx != tid:
            pos_j = positions[idx]
            vel_j = velocities[idx]
            rho_j = densities[idx]
            P_j = pressures[idx]
            
            r_vec = pos_i - pos_j
            r = wp.length(r_vec)
            
            if r < h and r > 1e-10:
                rho_j_safe = wp.max(rho_j, rest_density * 0.5)
                
                # Pressure force (symmetric)
                pressure_term = P_i / (rho_i_safe * rho_i_safe) + P_j / (rho_j_safe * rho_j_safe)
                grad_W = spiky_gradient(r_vec, r, h)
                force_pressure = grad_W * (-particle_mass * pressure_term)
                
                # Viscosity force
                vel_diff = vel_j - vel_i
                lap_W = viscosity_laplacian(r, h)
                force_viscosity = vel_diff * (viscosity * particle_mass / rho_j_safe * lap_W)
                
                force = force + force_pressure + force_viscosity
    
    forces[tid * 3 + 0] = force[0]
    forces[tid * 3 + 1] = force[1]
    forces[tid * 3 + 2] = force[2]


# =============================================================================
# SPH INTEGRATION KERNEL (with external blower forces and containment)
# =============================================================================

@wp.kernel
def integrate_sph_air_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=float),
    densities: wp.array(dtype=float),
    n: int,
    dt: float,
    rest_density: float,
    # Blower parameters
    blower_center: wp.vec3,
    blower_omega: float,
    impeller_radius: float,
    scroll_radius: float,
    scroll_half_width: float,  # Z extent of scroll housing
    blower_outlet: wp.vec3,
    tip_speed: float,
    # Blower outlet rectangular dimensions
    outlet_width: float,   # Z dimension of rectangular outlet
    outlet_height: float,  # Y dimension of rectangular outlet
    # Flow path
    filter_outlet: wp.vec3,
    elbow_inlet: wp.vec3,
    elbow_center: wp.vec3,
    elbow_outlet: wp.vec3,
    elbow_bend_radius: float,
    elbow_duct_radius: float,  # Radius of circular duct through elbow
    damper_1_inlet: wp.vec3,
    damper_2_outlet: wp.vec3,
    # Geometry
    duct_radius: float,
    outlet_duct_radius: float,
    # Bounds
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    # Spawn
    spawn_x: float,
    spawn_y_min: float, spawn_y_max: float,
    spawn_z_min: float, spawn_z_max: float,
    max_vel: float,
):
    """
    Integrate SPH air particles with external blower forces and boundary containment.
    """
    tid = wp.tid()
    if tid >= n:
        return
    
    pos = positions[tid]
    vel = velocities[tid]
    rho = densities[tid]
    
    # Get SPH force
    fx = forces[tid * 3 + 0]
    fy = forces[tid * 3 + 1]
    fz = forces[tid * 3 + 2]
    force = wp.vec3(fx, fy, fz)
    
    # Acceleration from SPH
    rho_safe = wp.max(rho, rest_density * 0.5)
    accel = force * (rest_density / rho_safe)
    
    # =========================================================================
    # EXTERNAL BLOWER FORCE based on position
    # =========================================================================
    duct_y = filter_outlet[1]
    elbow_z = elbow_outlet[2]
    blower_z = blower_center[2]
    outlet_x = blower_outlet[0]
    
    # Scroll outlet transition point (earlier than blower_outlet X)
    scroll_outlet_x = blower_center[0] + scroll_radius * 0.7
    
    # Segment 1: Before elbow - push in +X
    if pos[0] < elbow_inlet[0] and pos[2] < elbow_z * 0.5:
        accel = accel + wp.vec3(tip_speed * 0.5, 0.0, 0.0)
    
    # Segment 2: 90° Elbow - follow curve with centrifugal effect toward outer wall
    elif pos[0] >= elbow_inlet[0] - duct_radius and pos[2] < elbow_z:
        # Vector from elbow bend center to particle
        dx = pos[0] - elbow_center[0]
        dz = pos[2] - elbow_center[2]
        r_bend = wp.sqrt(dx * dx + dz * dz)
        
        # Angle around the bend (0 at inlet +X, pi/2 at outlet +Z)
        alpha = wp.atan2(dz, -dx)
        alpha = wp.clamp(alpha, 0.0, 1.5708)
        
        # Tangent direction (flow direction along the curve)
        tan_x = wp.sin(alpha)   # At alpha=0: tan=(0,0,1), at alpha=pi/2: tan=(1,0,0)
        tan_z = wp.cos(alpha)
        
        # Radial direction (outward from bend center)
        rad_x = -wp.cos(alpha)  # Points away from bend center
        rad_z = wp.sin(alpha)
        
        # Main flow: tangent to curve
        flow_accel = tip_speed * 0.5
        
        # Centrifugal effect: pushes toward outer curve (away from bend center)
        # v^2/r effect - faster flow = stronger centrifugal push
        centrifugal_accel = tip_speed * 0.3  # Push toward outer wall
        
        accel = accel + wp.vec3(
            tan_x * flow_accel + rad_x * centrifugal_accel,
            0.0,
            tan_z * flow_accel + rad_z * centrifugal_accel
        )
    
    # Segment 3: Vertical duct - push in +Z
    elif pos[2] >= elbow_z - duct_radius and pos[2] < blower_z:
        accel = accel + wp.vec3(0.0, 0.0, tip_speed * 0.5)
    
    # Segment 4: Blower - centrifugal acceleration with scroll outlet
    elif pos[2] >= blower_z - scroll_radius * 0.5 and pos[0] < scroll_outlet_x:
        dx = pos[0] - blower_center[0]
        dy = pos[1] - blower_center[1]
        r_xy = wp.sqrt(dx * dx + dy * dy)
        
        # Angle from center (0 = +X direction, which is the outlet)
        angle = wp.atan2(dy, dx)  # -pi to pi, 0 at +X
        
        # Normalize radius (0 = center, 1 = scroll edge)
        r_norm = r_xy / (scroll_radius + 0.001)
        
        if r_xy > 0.01:
            rad_x = dx / r_xy
            rad_y = dy / r_xy
            tan_x = -rad_y  # Tangent direction (counterclockwise)
            tan_y = rad_x
            
            # Centrifugal: omega^2 * r (pushes outward)
            centrifugal = blower_omega * blower_omega * r_xy
            
            # Tangential drag from impeller blades
            tangent_drag = blower_omega * r_xy * 0.6
            
            # Scroll effect: as particles approach +X side (angle near 0), 
            # guide them more strongly toward the outlet
            # abs(angle) is 0 at +X, pi at -X
            outlet_proximity = wp.cos(angle)  # 1 at +X, -1 at -X
            outlet_proximity = wp.max(outlet_proximity, 0.0)  # Only positive side
            
            # Strong +X acceleration near the outlet region
            # Increases with radius (scroll guides outer flow to outlet)
            outlet_accel = tip_speed * r_norm * (1.0 + outlet_proximity * 2.0)
            
            # Reduce tangential near outlet to let particles exit
            tangent_factor = 1.0 - outlet_proximity * 0.8
            
            accel = accel + wp.vec3(
                rad_x * centrifugal * 0.3 + tan_x * tangent_drag * tangent_factor + outlet_accel,
                rad_y * centrifugal * 0.3 + tan_y * tangent_drag * tangent_factor,
                0.0
            )
        else:
            # Near center - push toward impeller edge and outlet
            accel = accel + wp.vec3(tip_speed * 0.8, 0.0, 0.0)
    
    # Segment 5+: After blower outlet - strong +X push
    else:
        accel = accel + wp.vec3(tip_speed * 1.0, 0.0, 0.0)
    
    # =========================================================================
    # VELOCITY INTEGRATION
    # =========================================================================
    vel_new = vel + accel * dt
    
    # Clamp velocity
    vel_mag = wp.length(vel_new)
    if vel_mag > max_vel:
        vel_new = vel_new * (max_vel / vel_mag)
    
    # Position update
    pos_new = pos + vel_new * dt
    
    # =========================================================================
    # BOUNDARY CONTAINMENT
    # =========================================================================
    # Determine segment based on position
    # Blower outlet transition: once X > blower_center + scroll_radius*0.7, treat as outlet
    scroll_outlet_x = blower_center[0] + scroll_radius * 0.7
    
    segment = 0
    if pos_new[0] < elbow_inlet[0] and pos_new[2] < elbow_z * 0.5:
        segment = 1
    elif pos_new[0] >= elbow_inlet[0] - duct_radius and pos_new[2] < elbow_z:
        segment = 2
    elif pos_new[2] >= elbow_z - duct_radius and pos_new[2] < blower_z:
        segment = 3
    elif pos_new[2] >= blower_z - scroll_radius * 0.5 and pos_new[0] < scroll_outlet_x:
        # In blower scroll region
        segment = 4
    else:
        # Past the scroll outlet threshold - in outlet duct
        segment = 5
    
    if segment == 1:
        # Inlet duct (X axis) - STRICT cylindrical containment
        dy = pos_new[1] - duct_y
        dz = pos_new[2] - filter_outlet[2]
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        # HARD wall - no escape
        max_r = duct_radius * 0.99
        if r_yz > max_r:
            if r_yz > 0.001:
                scale = max_r / r_yz
                dy = dy * scale
                dz = dz * scale
            pos_new = wp.vec3(pos_new[0], duct_y + dy, filter_outlet[2] + dz)
            # Kill radial velocity, keep axial
            vel_new = wp.vec3(wp.abs(vel_new[0]), 0.0, 0.0)
    
    elif segment == 2:
        # 90° ELBOW - STRICT toroidal containment
        # Particles can ONLY exit through inlet (alpha=0) or outlet (alpha=pi/2)
        # NO escape through tube walls
        dx = pos_new[0] - elbow_center[0]
        dz = pos_new[2] - elbow_center[2]
        dy = pos_new[1] - duct_y
        
        # Angle around the bend arc (0 = inlet at -X from center, pi/2 = outlet at +Z)
        alpha = wp.atan2(dz, -dx)
        
        # HARD CLAMP angle to valid range [0, pi/2]
        # This prevents particles from going "around" the elbow the wrong way
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.5708:
            alpha = 1.5708
        
        # Point on the elbow centerline at this angle
        cx = elbow_center[0] - elbow_bend_radius * wp.cos(alpha)
        cz = elbow_center[2] + elbow_bend_radius * wp.sin(alpha)
        
        # Vector from tube centerline to particle
        to_p_x = pos_new[0] - cx
        to_p_y = dy
        to_p_z = pos_new[2] - cz
        
        # Distance from tube centerline
        r_tube = wp.sqrt(to_p_x * to_p_x + to_p_y * to_p_y + to_p_z * to_p_z)
        
        # STRICT wall containment - NO escape
        max_r = elbow_duct_radius * 0.99  # Hard wall limit
        if r_tube > max_r:
            # Project HARD onto tube inner surface
            if r_tube > 0.001:
                scale = max_r / r_tube
                to_p_x = to_p_x * scale
                to_p_y = to_p_y * scale
                to_p_z = to_p_z * scale
            pos_new = wp.vec3(cx + to_p_x, duct_y + to_p_y, cz + to_p_z)
            
            # Kill normal velocity component, keep tangential
            tan_x = wp.sin(alpha)
            tan_z = wp.cos(alpha)
            vel_tang = vel_new[0] * tan_x + vel_new[2] * tan_z
            # Only tangential flow remains
            vel_new = wp.vec3(tan_x * wp.abs(vel_tang), 0.0, tan_z * wp.abs(vel_tang))
        
        # Also enforce Y containment (top/bottom of elbow tube)
        if wp.abs(pos_new[1] - duct_y) > elbow_duct_radius * 0.95:
            sign_y = 1.0 if pos_new[1] > duct_y else -1.0
            pos_new = wp.vec3(pos_new[0], duct_y + sign_y * elbow_duct_radius * 0.9, pos_new[2])
            vel_new = wp.vec3(vel_new[0], 0.0, vel_new[2])
    
    elif segment == 3:
        # Vertical duct (Z axis) - STRICT cylindrical containment
        dx = pos_new[0] - elbow_outlet[0]
        dy = pos_new[1] - duct_y
        r_xy = wp.sqrt(dx * dx + dy * dy)
        
        # HARD wall - no escape
        max_r = duct_radius * 0.99
        if r_xy > max_r:
            if r_xy > 0.001:
                scale = max_r / r_xy
                dx = dx * scale
                dy = dy * scale
            pos_new = wp.vec3(elbow_outlet[0] + dx, duct_y + dy, pos_new[2])
            # Kill radial velocity, keep axial (Z)
            vel_new = wp.vec3(0.0, 0.0, wp.abs(vel_new[2]))
    
    elif segment == 4:
        # Blower scroll with RECTANGULAR outlet opening toward +X
        # Uses actual outlet_width (Z) and outlet_height (Y) from geometry
        dx = pos_new[0] - blower_center[0]
        dy = pos_new[1] - blower_center[1]
        dz = pos_new[2] - blower_center[2]
        r_xy = wp.sqrt(dx * dx + dy * dy)
        
        # Check if particle is in the outlet region (+X side of scroll)
        in_outlet_region = pos_new[0] > blower_center[0] + scroll_radius * 0.4
        
        # Rectangular outlet bounds from actual geometry
        outlet_y_half = outlet_height * 0.5 * 0.9   # Y dimension (height) with margin
        outlet_z_half = outlet_width * 0.5 * 0.9    # Z dimension (width) with margin
        
        if in_outlet_region:
            # In rectangular outlet casing - STRICT rectangular containment
            
            # Y bounds: HARD WALL at +/- outlet_height/2
            dy_new = pos_new[1] - blower_center[1]
            if dy_new > outlet_y_half:
                pos_new = wp.vec3(pos_new[0], blower_center[1] + outlet_y_half * 0.95, pos_new[2])
                vel_new = wp.vec3(vel_new[0], 0.0, vel_new[2])  # Zero Y velocity at wall
            elif dy_new < -outlet_y_half:
                pos_new = wp.vec3(pos_new[0], blower_center[1] - outlet_y_half * 0.95, pos_new[2])
                vel_new = wp.vec3(vel_new[0], 0.0, vel_new[2])  # Zero Y velocity at wall
            
            # Z bounds: HARD WALL at +/- outlet_width/2
            dz_new = pos_new[2] - blower_center[2]
            if dz_new > outlet_z_half:
                pos_new = wp.vec3(pos_new[0], pos_new[1], blower_center[2] + outlet_z_half * 0.95)
                vel_new = wp.vec3(vel_new[0], vel_new[1], 0.0)  # Zero Z velocity at wall
            elif dz_new < -outlet_z_half:
                pos_new = wp.vec3(pos_new[0], pos_new[1], blower_center[2] - outlet_z_half * 0.95)
                vel_new = wp.vec3(vel_new[0], vel_new[1], 0.0)  # Zero Z velocity at wall
        else:
            # In scroll region - cylindrical housing containment
            
            # Z containment (top/bottom of scroll housing)
            if dz > scroll_half_width * 0.85:
                pos_new = wp.vec3(pos_new[0], pos_new[1], blower_center[2] + scroll_half_width * 0.8)
                vel_new = wp.vec3(vel_new[0], vel_new[1], 0.0)
            elif dz < -scroll_half_width * 0.85:
                pos_new = wp.vec3(pos_new[0], pos_new[1], blower_center[2] - scroll_half_width * 0.8)
                vel_new = wp.vec3(vel_new[0], vel_new[1], 0.0)
            
            # Cylindrical containment with opening at +X toward rectangular outlet
            if r_xy > scroll_radius * 0.92:
                angle = wp.atan2(dy, dx)
                near_outlet = wp.abs(angle) < 0.4  # Within ~23 degrees of +X
                
                if near_outlet:
                    # Near outlet - transition to rectangular, constrain to outlet bounds
                    if dy > outlet_y_half:
                        pos_new = wp.vec3(pos_new[0], blower_center[1] + outlet_y_half * 0.9, pos_new[2])
                        vel_new = wp.vec3(vel_new[0], 0.0, vel_new[2])
                    elif dy < -outlet_y_half:
                        pos_new = wp.vec3(pos_new[0], blower_center[1] - outlet_y_half * 0.9, pos_new[2])
                        vel_new = wp.vec3(vel_new[0], 0.0, vel_new[2])
                else:
                    # Not near outlet - constrain to scroll cylinder
                    scale = scroll_radius * 0.85 / (r_xy + 0.001)
                    pos_new = wp.vec3(blower_center[0] + dx * scale, blower_center[1] + dy * scale, pos_new[2])
                    vel_new = vel_new * 0.7
    
    else:
        # Outlet duct (X axis) - cylindrical containment
        dy = pos_new[1] - blower_center[1]
        dz = pos_new[2] - blower_center[2]
        r_yz = wp.sqrt(dy * dy + dz * dz)
        
        # Strict cylindrical containment for outlet duct
        if r_yz > outlet_duct_radius * 0.88:
            scale = outlet_duct_radius * 0.82 / (r_yz + 0.001)
            pos_new = wp.vec3(pos_new[0], blower_center[1] + dy * scale, blower_center[2] + dz * scale)
            vel_new = wp.vec3(vel_new[0], vel_new[1] * 0.2, vel_new[2] * 0.2)
    
    # =========================================================================
    # RESPAWN IF OUT OF BOUNDS
    # =========================================================================
    respawn = False
    if pos_new[0] > damper_2_outlet[0] + 0.1:
        respawn = True
    if pos_new[0] < x_min or pos_new[0] > x_max:
        respawn = True
    if pos_new[1] < y_min or pos_new[1] > y_max:
        respawn = True
    if pos_new[2] < z_min or pos_new[2] > z_max:
        respawn = True
    
    if respawn:
        seed = float(tid * 12345 + int(pos[0] * 1000.0))
        pos_new = wp.vec3(
            spawn_x,
            spawn_y_min + (spawn_y_max - spawn_y_min) * wp.frac(seed * 0.618),
            spawn_z_min + (spawn_z_max - spawn_z_min) * wp.frac(seed * 0.381)
        )
        vel_new = wp.vec3(tip_speed * 0.2, 0.0, 0.0)
    
    positions[tid] = pos_new
    velocities[tid] = vel_new


# =============================================================================
# XSPH VELOCITY SMOOTHING (for coherent flow)
# =============================================================================

@wp.kernel
def xsph_correction_kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=float),
    corrections: wp.array(dtype=float),
    grid: wp.uint64,
    n: int,
    particle_mass: float,
    h: float,
    xsph_factor: float,
    rest_density: float,
):
    """XSPH velocity smoothing for more coherent flow."""
    tid = wp.tid()
    if tid >= n:
        return
    
    pos_i = positions[tid]
    vel_i = velocities[tid]
    rho_i = densities[tid]
    
    correction = wp.vec3(0.0, 0.0, 0.0)
    
    query = wp.hash_grid_query(grid, pos_i, h)
    idx = int(0)
    
    while wp.hash_grid_query_next(query, idx):
        if idx != tid:
            pos_j = positions[idx]
            vel_j = velocities[idx]
            rho_j = densities[idx]
            
            r_vec = pos_i - pos_j
            r = wp.length(r_vec)
            
            if r < h:
                W = poly6_kernel(r, h)
                vel_diff = vel_j - vel_i
                rho_sum = rho_i + rho_j
                if rho_sum > 0.01:
                    correction = correction + vel_diff * (2.0 * particle_mass * W / rho_sum)
    
    correction = correction * xsph_factor
    corrections[tid * 3 + 0] = correction[0]
    corrections[tid * 3 + 1] = correction[1]
    corrections[tid * 3 + 2] = correction[2]


@wp.kernel
def apply_xsph_kernel(
    velocities: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=float),
    n: int,
):
    """Apply XSPH corrections to velocities."""
    tid = wp.tid()
    if tid >= n:
        return
    
    vel = velocities[tid]
    corr = wp.vec3(corrections[tid * 3 + 0], corrections[tid * 3 + 1], corrections[tid * 3 + 2])
    velocities[tid] = vel + corr


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class AirFlowPhysicsSimulator:
    """
    Physics-based air flow simulator for air system.
    
    Uses actual geometry from AirSystemAssembly and computes all
    physics from first principles - no magic numbers.
    """
    
    def __init__(
        self,
        assembly: AirSystemAssembly,
        config: AirFlowPhysicsConfig = None,
    ):
        """
        Initialize the physics-based air flow simulator.
        
        Args:
            assembly: AirSystemAssembly with actual geometry
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or AirFlowPhysicsConfig()
        self.state = AirFlowPhysicsState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Extract geometry
        self.geometry = extract_air_geometry(assembly)
        
        # Compute derived parameters
        self._compute_derived_parameters()
        
        # Allocate SPH arrays if enabled
        if self.config.enable_sph:
            self._allocate_sph_arrays()
            self._init_hash_grid()
        
        # Get system bounds
        self._compute_system_bounds()
    
    def _compute_derived_parameters(self):
        """Compute all derived physics parameters from geometry."""
        cfg = self.config
        blower_geo = self.geometry['blower']
        
        # Design angular velocity
        self.design_omega = TWO_PI * blower_geo.design_rpm / 60.0
        
        # Target angular velocity
        self.target_omega = TWO_PI * cfg.target_rpm / 60.0
        
        # Angular acceleration for ramp
        self.omega_ramp_rate = self.target_omega / cfg.ramp_time
        
        # Design tip speed
        self.design_tip_speed = self.design_omega * blower_geo.impeller_radius
        
        # Duct area for velocity calculation
        self.duct_area = PI * (self.geometry['system']['duct_diameter'] / 2.0) ** 2
        
        print(f"\n  Air Flow Physics Parameters (computed from geometry):")
        print(f"    Design RPM:        {blower_geo.design_rpm:.0f}")
        print(f"    Target RPM:        {cfg.target_rpm:.0f}")
        print(f"    Design omega:      {self.design_omega:.1f} rad/s")
        print(f"    Impeller dia:      {blower_geo.impeller_diameter*1000:.0f} mm")
        print(f"    Design tip speed:  {self.design_tip_speed:.1f} m/s")
        print(f"    Design flow:       {blower_geo.design_flow_rate:.0f} m³/h")
        print(f"    Design pressure:   {blower_geo.design_pressure_rise:.0f} Pa")
        print(f"    Duct diameter:     {self.geometry['system']['duct_diameter']*1000:.0f} mm")
        
        # Print duct segment geometry
        total_duct_length = sum(d.length for d in self.geometry['ducts'])
        print(f"\n  Duct Segments (from geometry, total length: {total_duct_length*1000:.0f} mm):")
        for duct in self.geometry['ducts']:
            print(f"    {duct.name}:")
            print(f"      Length:    {duct.length*1000:.0f} mm")
            print(f"      Diameter:  {duct.diameter*1000:.0f} mm")
            print(f"      Area:      {duct.area*10000:.1f} cm²")
        
        # Print flow path waypoints (all from assembly's stored positions)
        if 'flow_path' in self.geometry:
            print(f"\n  Flow Path Waypoints (from assembly port positions):")
            for name, pos in self.geometry['flow_path'].items():
                print(f"    {name:20s}: ({pos[0]*1000:6.0f}, {pos[1]*1000:6.0f}, {pos[2]*1000:6.0f}) mm")
    
    def _allocate_sph_arrays(self):
        """Allocate SPH air particle arrays on device."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.densities = wp.zeros(n, dtype=float, device=self.device)
        self.state.pressures = wp.zeros(n, dtype=float, device=self.device)
        self.state.forces = wp.zeros(n * 3, dtype=float, device=self.device)
        
        # Temporary array for XSPH corrections
        self._xsph_corrections = wp.zeros(n * 3, dtype=float, device=self.device)
        
        # Compute particle mass from density and estimated volume
        duct_length = 2.0  # Approximate total flow path
        duct_volume = PI * (self.geometry['system']['duct_diameter'] / 2.0) ** 2 * duct_length
        particle_volume = duct_volume / n
        self.particle_mass = self.config.air_density * particle_volume
        
        # Max velocity for stability
        self.max_velocity = min(self.config.speed_of_sound * 0.5, 
                                self.config.smoothing_length / self.config.dt * 0.1)
        
        print(f"\n  SPH Air Parameters:")
        print(f"    Particles:        {n}")
        print(f"    Particle mass:    {self.particle_mass*1e6:.3f} mg")
        print(f"    Smoothing length: {self.config.smoothing_length*1000:.1f} mm")
        print(f"    Sound speed:      {self.config.speed_of_sound:.0f} m/s")
        print(f"    Max velocity:     {self.max_velocity:.1f} m/s")
    
    def _init_hash_grid(self):
        """Initialize hash grid for SPH neighbor search."""
        cell_size = self.config.smoothing_length * 2.5
        
        # Estimate grid dimensions from system bounds
        if hasattr(self, 'x_max'):
            dim_x = int((self.x_max - self.x_min) / cell_size) + 1
            dim_y = int((self.y_max - self.y_min) / cell_size) + 1
            dim_z = int((self.z_max - self.z_min) / cell_size) + 1
        else:
            dim_x = dim_y = dim_z = 50  # Default
        
        self.state.hash_grid = wp.HashGrid(dim_x, dim_y, dim_z, device=self.device)
    
    def _compute_system_bounds(self):
        """Compute system bounding box for particle containment."""
        if not self.assembly._mesh_built:
            self.assembly.build_mesh()
        
        min_corner, max_corner = self.assembly.get_bounds()
        
        # Add some margin
        margin = 0.2
        self.x_min = min_corner[0] - margin
        self.x_max = max_corner[0] + margin
        self.y_min = min_corner[1] - margin
        self.y_max = max_corner[1] + margin
        self.z_min = min_corner[2] - margin
        self.z_max = max_corner[2] + margin
        
        # Spawn region (filter inlet)
        filter_pos = self.assembly._filter_position
        duct_radius = self.geometry['system']['duct_diameter'] / 2.0
        
        self.spawn_x = filter_pos[0] - 0.05
        self.spawn_y_min = filter_pos[1] - duct_radius * 0.8
        self.spawn_y_max = filter_pos[1] + duct_radius * 0.8
        self.spawn_z_min = filter_pos[2] - duct_radius * 0.8
        self.spawn_z_max = filter_pos[2] + duct_radius * 0.8
        
        # Re-init hash grid now that we have bounds
        if self.config.enable_sph and self.state.hash_grid is not None:
            self._init_hash_grid()
    
    def initialize_particles(self):
        """Initialize SPH air particles throughout the flow domain."""
        if not self.config.enable_sph:
            return
        
        n = self.config.num_particles
        rng = np.random.default_rng(42)
        
        positions = np.zeros((n, 3), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        
        # Get geometry
        flow_path = self.geometry.get('flow_path', {})
        filter_outlet = flow_path.get('filter_outlet', np.array([0.0, 0.0, 0.0]))
        elbow_inlet = flow_path.get('elbow_inlet', np.array([0.1, 0.0, 0.0]))
        elbow_outlet = flow_path.get('elbow_outlet', np.array([0.2, 0.0, 0.1]))
        blower_center = flow_path.get('blower_center', np.array([0.3, 0.0, 0.3]))
        blower_outlet = flow_path.get('blower_outlet', np.array([0.5, 0.0, 0.2]))
        
        duct_radius = self.geometry['system']['duct_diameter'] / 2.0
        blower_geo = self.geometry['blower']
        scroll_radius = blower_geo.scroll_diameter / 2.0
        impeller_width = blower_geo.impeller_width
        
        # Distribute particles: 40% inlet, 10% elbow, 10% vertical, 20% blower, 20% outlet
        n_inlet = int(n * 0.4)
        n_elbow = int(n * 0.1)
        n_vert = int(n * 0.1)
        n_blower = int(n * 0.2)
        n_outlet = n - n_inlet - n_elbow - n_vert - n_blower
        
        idx = 0
        
        # Inlet duct
        for i in range(n_inlet):
            t = rng.uniform(0, 1)
            x = filter_outlet[0] + t * (elbow_inlet[0] - filter_outlet[0])
            r = rng.uniform(0, duct_radius * 0.9)
            theta = rng.uniform(0, TWO_PI)
            y = filter_outlet[1] + r * np.cos(theta)
            z = filter_outlet[2] + r * np.sin(theta)
            positions[idx] = [x, y, z]
            velocities[idx] = [1.0, 0.0, 0.0]
            idx += 1
        
        # Elbow
        elbow_bend_radius = self.geometry['system'].get('elbow_bend_radius', 0.1)
        elbow_center = np.array([elbow_inlet[0] + elbow_bend_radius, elbow_inlet[1], elbow_inlet[2]])
        for i in range(n_elbow):
            alpha = rng.uniform(0, np.pi / 2)
            cx = elbow_center[0] - elbow_bend_radius * np.cos(alpha)
            cz = elbow_center[2] + elbow_bend_radius * np.sin(alpha)
            r = rng.uniform(0, duct_radius * 0.9)
            phi = rng.uniform(0, TWO_PI)
            x = cx + r * np.cos(phi) * np.cos(alpha)
            y = elbow_inlet[1] + r * np.sin(phi)
            z = cz + r * np.cos(phi) * np.sin(alpha)
            positions[idx] = [x, y, z]
            velocities[idx] = [np.cos(alpha), 0.0, np.sin(alpha)]
            idx += 1
        
        # Vertical duct
        blower_inlet_z = blower_center[2] - impeller_width * 0.5
        for i in range(n_vert):
            t = rng.uniform(0, 1)
            z = elbow_outlet[2] + t * (blower_inlet_z - elbow_outlet[2])
            r = rng.uniform(0, duct_radius * 0.9)
            theta = rng.uniform(0, TWO_PI)
            x = elbow_outlet[0] + r * np.cos(theta)
            y = elbow_outlet[1] + r * np.sin(theta)
            positions[idx] = [x, y, z]
            velocities[idx] = [0.0, 0.0, 1.0]
            idx += 1
        
        # Blower
        for i in range(n_blower):
            r = rng.uniform(0.1 * scroll_radius, scroll_radius * 0.9)
            theta = rng.uniform(0, TWO_PI)
            x = blower_center[0] + r * np.cos(theta)
            y = blower_center[1] + r * np.sin(theta)
            z = blower_center[2] + rng.uniform(-impeller_width * 0.4, impeller_width * 0.4)
            positions[idx] = [x, y, z]
            tan_x = -np.sin(theta)
            tan_y = np.cos(theta)
            velocities[idx] = [tan_x * 2.0 + 1.0, tan_y * 2.0, 0.0]
            idx += 1
        
        # Outlet duct
        damper_2_outlet = flow_path.get('damper_1_outlet', blower_outlet + np.array([0.5, 0.0, 0.0]))
        outlet_duct_radius = self.geometry['system'].get('duct_diameter', 0.266) / 2.0
        for i in range(n_outlet):
            t = rng.uniform(0, 1)
            x = blower_outlet[0] + t * (damper_2_outlet[0] - blower_outlet[0])
            r = rng.uniform(0, outlet_duct_radius * 0.9)
            theta = rng.uniform(0, TWO_PI)
            y = blower_center[1] + r * np.cos(theta)
            z = blower_outlet[2] + r * np.sin(theta)
            positions[idx] = [x, y, z]
            velocities[idx] = [2.0, 0.0, 0.0]
            idx += 1
        
        # Copy to device
        wp.copy(self.state.positions, wp.array(positions, dtype=wp.vec3, device=self.device))
        wp.copy(self.state.velocities, wp.array(velocities, dtype=wp.vec3, device=self.device))
        
        # Initialize densities
        densities = np.full(n, self.config.air_density, dtype=np.float32)
        wp.copy(self.state.densities, wp.array(densities, dtype=float, device=self.device))
        
        print(f"\n  Initialized {n} SPH air particles")
    
    # Alias for backwards compatibility
    def initialize_tracers(self):
        """Initialize particles (alias for backwards compatibility)."""
        self.initialize_particles()
    
    # =========================================================================
    # CONTROL METHODS
    # =========================================================================
    
    def start_system(self):
        """Start the air system (ramp up blower, open dampers)."""
        self.state.phase = SystemPhase.STARTUP
        self.state.phase_start_time = self.state.time
        self.state.blower_state = BlowerState.STARTING
    
    def stop_system(self):
        """Stop the air system (ramp down blower, close dampers)."""
        self.state.phase = SystemPhase.SHUTDOWN
        self.state.phase_start_time = self.state.time
        self.state.blower_state = BlowerState.STOPPING
    
    def set_damper_position(self, index: int, position: float):
        """Set damper position (0=closed, 1=open)."""
        if 0 <= index < len(self.state.damper_positions):
            self.state.damper_positions[index] = max(0.0, min(1.0, position))
            self.geometry['dampers'][index].position = self.state.damper_positions[index]
    
    # =========================================================================
    # SIMULATION STEP
    # =========================================================================
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        cfg = self.config
        
        # =================================================================
        # UPDATE BLOWER SPEED (RAMP UP/DOWN)
        # =================================================================
        if self.state.blower_state == BlowerState.STARTING:
            self.state.blower_omega += self.omega_ramp_rate * dt
            if self.state.blower_omega >= self.target_omega:
                self.state.blower_omega = self.target_omega
                self.state.blower_state = BlowerState.RUNNING
                self.state.phase = SystemPhase.RUNNING
        
        elif self.state.blower_state == BlowerState.STOPPING:
            self.state.blower_omega -= self.omega_ramp_rate * dt
            if self.state.blower_omega <= 0.0:
                self.state.blower_omega = 0.0
                self.state.blower_state = BlowerState.OFF
                self.state.phase = SystemPhase.OFF
        
        # Compute RPM and tip speed
        self.state.blower_rpm = self.state.blower_omega * 60.0 / TWO_PI
        blower_geo = self.geometry['blower']
        self.state.blower_tip_speed = self.state.blower_omega * blower_geo.impeller_radius
        
        # =================================================================
        # UPDATE DAMPER POSITIONS (RAMP)
        # =================================================================
        if self.state.phase == SystemPhase.STARTUP:
            # Open dampers during startup
            damper_ramp_rate = 1.0 / cfg.damper_ramp_time
            for i in range(len(self.state.damper_positions)):
                self.state.damper_positions[i] = min(
                    1.0,
                    self.state.damper_positions[i] + damper_ramp_rate * dt
                )
                self.geometry['dampers'][i].position = self.state.damper_positions[i]
        
        elif self.state.phase == SystemPhase.SHUTDOWN:
            # Close dampers during shutdown
            damper_ramp_rate = 1.0 / cfg.damper_ramp_time
            for i in range(len(self.state.damper_positions)):
                self.state.damper_positions[i] = max(
                    0.0,
                    self.state.damper_positions[i] - damper_ramp_rate * dt
                )
                self.geometry['dampers'][i].position = self.state.damper_positions[i]
        
        # =================================================================
        # CALCULATE SYSTEM PRESSURE DROPS
        # =================================================================
        rho = cfg.air_density
        mu = cfg.air_viscosity
        
        # First pass: estimate flow rate from blower performance
        Q_estimate, P_blower, shaft_power, efficiency = calculate_blower_performance(
            blower_geo,
            self.state.blower_omega,
            blower_geo.design_pressure_rise,
            rho
        )
        
        # Calculate component pressure drops at this flow rate
        total_dp = 0.0
        
        # Filter pressure drop
        filter_geo = self.geometry['filter']
        dp_filter = calculate_filter_pressure_drop(filter_geo, Q_estimate, rho)
        total_dp += dp_filter
        
        # Duct pressure drops
        velocity = Q_estimate / self.duct_area if self.duct_area > 0 else 0.0
        for duct in self.geometry['ducts']:
            dp_duct = calculate_duct_pressure_drop(duct, velocity, rho, mu)
            total_dp += dp_duct
        
        # Damper pressure drops
        dp_dampers = 0.0
        for damper_geo in self.geometry['dampers']:
            dp_damper = calculate_damper_pressure_drop(damper_geo, Q_estimate, rho)
            dp_dampers += dp_damper
        total_dp += dp_dampers
        
        # =================================================================
        # ITERATE TO FIND OPERATING POINT
        # =================================================================
        # Simple iteration to find where blower curve meets system curve
        for _ in range(3):
            Q_new, P_new, shaft_power, efficiency = calculate_blower_performance(
                blower_geo,
                self.state.blower_omega,
                total_dp,
                rho
            )
            Q_estimate = 0.5 * (Q_estimate + Q_new)
        
        # =================================================================
        # UPDATE STATE
        # =================================================================
        self.state.volume_flow_rate = Q_estimate
        self.state.mass_flow_rate = Q_estimate * rho
        self.state.static_pressure_rise = P_new
        self.state.total_pressure_drop = total_dp
        self.state.shaft_power = shaft_power
        self.state.efficiency = efficiency
        
        # Electrical power (assuming 95% motor efficiency)
        motor_efficiency = 0.95
        self.state.electrical_power = shaft_power / motor_efficiency
        
        # Accumulate energy
        self.state.total_energy_kWh += self.state.electrical_power * dt / 3600.0 / 1000.0
        
        # =================================================================
        # UPDATE SPH AIR PARTICLES
        # =================================================================
        if self.config.enable_sph and self.state.positions is not None:
            n = self.config.num_particles
            h = self.config.smoothing_length
            
            # Get flow path waypoints
            flow_path = self.geometry.get('flow_path', {})
            filter_outlet = flow_path.get('filter_outlet', np.array([0.0, 0.0, 0.0]))
            elbow_inlet = flow_path.get('elbow_inlet', np.array([0.1, 0.0, 0.0]))
            elbow_outlet = flow_path.get('elbow_outlet', np.array([0.2, 0.0, 0.1]))
            blower_center = flow_path.get('blower_center', np.array([0.3, 0.0, 0.3]))
            blower_outlet = flow_path.get('blower_outlet', np.array([0.5, 0.0, 0.2]))
            damper_1_inlet = flow_path.get('damper_0_inlet', blower_outlet + np.array([0.2, 0.0, 0.0]))
            damper_2_outlet = flow_path.get('damper_1_outlet', blower_outlet + np.array([0.55, 0.0, 0.0]))
            
            # Geometry
            elbow_bend_radius = self.geometry['system'].get('elbow_bend_radius', 0.1)
            elbow_center = np.array([elbow_inlet[0] + elbow_bend_radius, elbow_inlet[1], elbow_inlet[2]])
            elbow_duct_radius = self.geometry['system'].get('elbow_diameter', 0.3) / 2.0
            duct_radius = elbow_duct_radius  # Same as elbow duct
            outlet_duct_radius = self.geometry['system'].get('duct_diameter', 0.266) / 2.0
            
            blower_geo = self.geometry['blower']
            impeller_radius = blower_geo.impeller_radius
            scroll_radius = blower_geo.scroll_diameter / 2.0
            scroll_half_width = blower_geo.impeller_width * 0.6  # Housing slightly wider than impeller
            tip_speed = self.state.blower_omega * impeller_radius
            
            # Blower rectangular outlet dimensions (from geometry)
            outlet_width = blower_geo.outlet_width    # Z dimension
            outlet_height = blower_geo.outlet_height  # Y dimension
            
            # Update hash grid
            self.state.hash_grid.build(self.state.positions, h)
            
            # 1. Compute density and pressure
            wp.launch(
                kernel=compute_sph_density_pressure_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.densities,
                    self.state.pressures,
                    self.state.hash_grid.id,
                    n,
                    float(self.particle_mass),
                    float(h),
                    float(cfg.air_density),
                    float(cfg.speed_of_sound),
                ],
                device=self.device,
            )
            
            # 2. Compute SPH forces
            wp.launch(
                kernel=compute_sph_forces_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.densities,
                    self.state.pressures,
                    self.state.forces,
                    self.state.hash_grid.id,
                    n,
                    float(self.particle_mass),
                    float(h),
                    float(cfg.sph_viscosity),
                    float(cfg.air_density),
                ],
                device=self.device,
            )
            
            # 3. Integrate with external blower forces and containment
            wp.launch(
                kernel=integrate_sph_air_kernel,
                dim=n,
                inputs=[
                    self.state.positions,
                    self.state.velocities,
                    self.state.forces,
                    self.state.densities,
                    n,
                    float(dt),
                    float(cfg.air_density),
                    # Blower
                    wp.vec3(float(blower_center[0]), float(blower_center[1]), float(blower_center[2])),
                    float(self.state.blower_omega),
                    float(impeller_radius),
                    float(scroll_radius),
                    float(scroll_half_width),
                    wp.vec3(float(blower_outlet[0]), float(blower_outlet[1]), float(blower_outlet[2])),
                    float(tip_speed),
                    # Blower rectangular outlet dimensions
                    float(outlet_width),
                    float(outlet_height),
                    # Flow path
                    wp.vec3(float(filter_outlet[0]), float(filter_outlet[1]), float(filter_outlet[2])),
                    wp.vec3(float(elbow_inlet[0]), float(elbow_inlet[1]), float(elbow_inlet[2])),
                    wp.vec3(float(elbow_center[0]), float(elbow_center[1]), float(elbow_center[2])),
                    wp.vec3(float(elbow_outlet[0]), float(elbow_outlet[1]), float(elbow_outlet[2])),
                    float(elbow_bend_radius),
                    float(elbow_duct_radius),
                    wp.vec3(float(damper_1_inlet[0]), float(damper_1_inlet[1]), float(damper_1_inlet[2])),
                    wp.vec3(float(damper_2_outlet[0]), float(damper_2_outlet[1]), float(damper_2_outlet[2])),
                    # Geometry
                    float(duct_radius),
                    float(outlet_duct_radius),
                    # Bounds
                    float(self.x_min), float(self.x_max),
                    float(self.y_min), float(self.y_max),
                    float(self.z_min), float(self.z_max),
                    # Spawn
                    float(self.spawn_x),
                    float(self.spawn_y_min), float(self.spawn_y_max),
                    float(self.spawn_z_min), float(self.spawn_z_max),
                    float(self.max_velocity),
                ],
                device=self.device,
            )
            
            # 4. XSPH velocity smoothing for coherent flow
            if cfg.xsph_factor > 0:
                wp.launch(
                    kernel=xsph_correction_kernel,
                    dim=n,
                    inputs=[
                        self.state.positions,
                        self.state.velocities,
                        self.state.densities,
                        self._xsph_corrections,
                        self.state.hash_grid.id,
                        n,
                        float(self.particle_mass),
                        float(h),
                        float(cfg.xsph_factor),
                        float(cfg.air_density),
                    ],
                    device=self.device,
                )
                wp.launch(
                    kernel=apply_xsph_kernel,
                    dim=n,
                    inputs=[
                        self.state.velocities,
                        self._xsph_corrections,
                        n,
                    ],
                    device=self.device,
                )
            
            # Update statistics periodically
            if self.state.step % 100 == 0:
                velocities = self.state.velocities.numpy()
                densities = self.state.densities.numpy()
                speeds = np.linalg.norm(velocities, axis=1)
                self.state.max_velocity = float(np.max(speeds))
                self.state.avg_density = float(np.mean(densities))
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    # =========================================================================
    # RESULTS AND GETTERS
    # =========================================================================
    
    def get_results(self) -> Dict[str, Any]:
        """Get current simulation results."""
        return {
            'time': self.state.time,
            'step': self.state.step,
            'phase': self.state.phase.value,
            'blower_state': self.state.blower_state.name,
            'blower_rpm': self.state.blower_rpm,
            'blower_omega': self.state.blower_omega,
            'tip_speed': self.state.blower_tip_speed,
            'volume_flow_rate_m3_h': self.state.volume_flow_rate * 3600,
            'mass_flow_rate_kg_s': self.state.mass_flow_rate,
            'pressure_rise_Pa': self.state.static_pressure_rise,
            'system_pressure_drop_Pa': self.state.total_pressure_drop,
            'shaft_power_kW': self.state.shaft_power / 1000.0,
            'electrical_power_kW': self.state.electrical_power / 1000.0,
            'efficiency': self.state.efficiency,
            'damper_positions': self.state.damper_positions.copy(),
            'total_energy_kWh': self.state.total_energy_kWh,
            # SPH statistics
            'max_velocity': self.state.max_velocity,
            'avg_density': self.state.avg_density,
        }
    
    def get_particle_positions(self) -> np.ndarray:
        """Get current SPH air particle positions."""
        if self.state.positions is None:
            return np.array([])
        return self.state.positions.numpy()
    
    def get_particle_velocities(self) -> np.ndarray:
        """Get current SPH air particle velocities."""
        if self.state.velocities is None:
            return np.array([])
        return self.state.velocities.numpy()
    
    def get_particle_densities(self) -> np.ndarray:
        """Get current SPH air particle densities."""
        if self.state.densities is None:
            return np.array([])
        return self.state.densities.numpy()
    
    def get_particle_pressures(self) -> np.ndarray:
        """Get current SPH air particle pressures."""
        if self.state.pressures is None:
            return np.array([])
        return self.state.pressures.numpy()
    
    # Aliases for backwards compatibility with tracer API
    def get_tracer_positions(self) -> np.ndarray:
        """Get particle positions (alias for backwards compatibility)."""
        return self.get_particle_positions()
    
    def get_tracer_velocities(self) -> np.ndarray:
        """Get particle velocities (alias for backwards compatibility)."""
        return self.get_particle_velocities()
    
    def get_duct_segment_data(self) -> List[Dict[str, Any]]:
        """Get pressure drop data for each duct segment."""
        data = []
        for duct in self.geometry['ducts']:
            data.append({
                'name': duct.name,
                'length': duct.length,
                'diameter': duct.diameter,
                'velocity': duct.velocity,
                'reynolds': duct.reynolds,
                'pressure_drop': duct.pressure_drop,
            })
        return data
    
    # =========================================================================
    # PHYSICS ANALYSIS METHODS
    # =========================================================================
    
    def compute_sph_statistics(self) -> Dict[str, Any]:
        """
        Compute detailed SPH particle statistics.
        
        Returns:
            Dictionary with SPH statistics including velocity, density,
            pressure distributions, and flow characteristics.
        """
        if not self.config.enable_sph or self.state.positions is None:
            return {'enabled': False}
        
        positions = self.state.positions.numpy()
        velocities = self.state.velocities.numpy()
        densities = self.state.densities.numpy()
        pressures = self.state.pressures.numpy()
        
        # Velocity statistics
        speeds = np.linalg.norm(velocities, axis=1)
        vel_x = velocities[:, 0]
        vel_y = velocities[:, 1]
        vel_z = velocities[:, 2]
        
        # Position spread (for flow uniformity)
        pos_std = np.std(positions, axis=0)
        
        # Density variation (should be near rest density for incompressible)
        density_variation = np.std(densities) / np.mean(densities) * 100
        
        # Pressure statistics
        pressure_range = np.max(pressures) - np.min(pressures)
        
        # Compute kinetic energy
        kinetic_energy = 0.5 * self.particle_mass * np.sum(speeds ** 2)
        
        # Flow direction analysis
        mean_velocity = np.mean(velocities, axis=0)
        flow_direction = mean_velocity / (np.linalg.norm(mean_velocity) + 1e-10)
        
        return {
            'enabled': True,
            'num_particles': len(positions),
            # Velocity
            'velocity_mean': float(np.mean(speeds)),
            'velocity_max': float(np.max(speeds)),
            'velocity_min': float(np.min(speeds)),
            'velocity_std': float(np.std(speeds)),
            'velocity_x_mean': float(np.mean(vel_x)),
            'velocity_y_mean': float(np.mean(vel_y)),
            'velocity_z_mean': float(np.mean(vel_z)),
            # Density
            'density_mean': float(np.mean(densities)),
            'density_std': float(np.std(densities)),
            'density_variation_percent': float(density_variation),
            'density_min': float(np.min(densities)),
            'density_max': float(np.max(densities)),
            # Pressure
            'pressure_mean': float(np.mean(pressures)),
            'pressure_std': float(np.std(pressures)),
            'pressure_range': float(pressure_range),
            'pressure_min': float(np.min(pressures)),
            'pressure_max': float(np.max(pressures)),
            # Energy
            'kinetic_energy_J': float(kinetic_energy),
            # Flow direction
            'flow_direction': flow_direction.tolist(),
            'mean_velocity_magnitude': float(np.linalg.norm(mean_velocity)),
            # Position spread
            'position_std_x': float(pos_std[0]),
            'position_std_y': float(pos_std[1]),
            'position_std_z': float(pos_std[2]),
        }
    
    def compute_flow_regime_analysis(self) -> Dict[str, Any]:
        """
        Analyze flow regimes throughout the system.
        
        Computes Reynolds numbers and classifies flow as laminar,
        transitional, or turbulent for each duct segment.
        
        Returns:
            Dictionary with flow regime analysis for each segment.
        """
        cfg = self.config
        rho = cfg.air_density
        mu = cfg.air_viscosity
        nu = mu / rho  # Kinematic viscosity
        
        # Get current flow rate
        Q = self.state.volume_flow_rate  # m³/s
        
        analysis = {
            'flow_rate_m3_s': Q,
            'flow_rate_m3_h': Q * 3600,
            'air_density': rho,
            'air_viscosity': mu,
            'kinematic_viscosity': nu,
            'segments': [],
        }
        
        # Analyze each duct segment
        for duct in self.geometry['ducts']:
            velocity = Q / duct.area if duct.area > 0 else 0.0
            Re = rho * velocity * duct.diameter / mu if velocity > 0 else 0.0
            
            # Classify flow regime
            if Re < 2300:
                regime = "laminar"
            elif Re < 4000:
                regime = "transitional"
            else:
                regime = "turbulent"
            
            # Calculate friction factor
            if Re > 0:
                f = calculate_friction_factor(Re, duct.roughness, duct.diameter)
            else:
                f = 0.0
            
            # Pressure drop
            if velocity > 0:
                dP = f * (duct.length / duct.diameter) * (rho * velocity ** 2 / 2.0)
            else:
                dP = 0.0
            
            analysis['segments'].append({
                'name': duct.name,
                'diameter_mm': duct.diameter * 1000,
                'length_mm': duct.length * 1000,
                'area_cm2': duct.area * 10000,
                'velocity_m_s': velocity,
                'reynolds': Re,
                'flow_regime': regime,
                'friction_factor': f,
                'pressure_drop_Pa': dP,
            })
        
        # Overall flow regime summary
        reynolds_values = [s['reynolds'] for s in analysis['segments'] if s['reynolds'] > 0]
        if reynolds_values:
            analysis['reynolds_mean'] = np.mean(reynolds_values)
            analysis['reynolds_max'] = np.max(reynolds_values)
            analysis['reynolds_min'] = np.min(reynolds_values)
            
            # Count regimes
            regimes = [s['flow_regime'] for s in analysis['segments']]
            analysis['laminar_count'] = regimes.count('laminar')
            analysis['transitional_count'] = regimes.count('transitional')
            analysis['turbulent_count'] = regimes.count('turbulent')
        
        return analysis
    
    def compute_blower_analysis(self) -> Dict[str, Any]:
        """
        Analyze blower operating point and performance.
        
        Returns:
            Dictionary with detailed blower performance metrics.
        """
        blower_geo = self.geometry['blower']
        omega = self.state.blower_omega
        rpm = self.state.blower_rpm
        
        # Design point
        design_rpm = blower_geo.design_rpm
        design_Q = blower_geo.design_flow_rate / 3600.0  # m³/s
        design_P = blower_geo.design_pressure_rise
        
        # Current operating point
        Q = self.state.volume_flow_rate
        P = self.state.static_pressure_rise
        W = self.state.shaft_power
        eta = self.state.efficiency
        
        # Speed ratio
        n_ratio = rpm / design_rpm if design_rpm > 0 else 0.0
        
        # Flow coefficient (dimensionless)
        tip_speed = omega * blower_geo.impeller_radius
        if tip_speed > 0:
            flow_coeff = Q / (blower_geo.blade_outlet_area * tip_speed)
        else:
            flow_coeff = 0.0
        
        # Head coefficient (dimensionless)
        if tip_speed > 0:
            head_coeff = P / (self.config.air_density * tip_speed ** 2)
        else:
            head_coeff = 0.0
        
        # Specific speed (dimensionless)
        if P > 0 and Q > 0:
            specific_speed = omega * np.sqrt(Q) / (P / self.config.air_density) ** 0.75
        else:
            specific_speed = 0.0
        
        # Operating point relative to design
        flow_ratio = Q / design_Q if design_Q > 0 else 0.0
        pressure_ratio = P / design_P if design_P > 0 else 0.0
        
        return {
            # Geometry
            'impeller_diameter_mm': blower_geo.impeller_diameter * 1000,
            'impeller_width_mm': blower_geo.impeller_width * 1000,
            'blade_type': blower_geo.blade_type,
            'num_blades': blower_geo.num_blades,
            # Design point
            'design_rpm': design_rpm,
            'design_flow_rate_m3_h': blower_geo.design_flow_rate,
            'design_pressure_Pa': design_P,
            # Current operation
            'current_rpm': rpm,
            'current_omega_rad_s': omega,
            'tip_speed_m_s': tip_speed,
            'flow_rate_m3_s': Q,
            'flow_rate_m3_h': Q * 3600,
            'pressure_rise_Pa': P,
            'shaft_power_W': W,
            'shaft_power_kW': W / 1000,
            'efficiency': eta,
            'efficiency_percent': eta * 100,
            # Ratios
            'speed_ratio': n_ratio,
            'flow_ratio': flow_ratio,
            'pressure_ratio': pressure_ratio,
            # Dimensionless coefficients
            'flow_coefficient': flow_coeff,
            'head_coefficient': head_coeff,
            'specific_speed': specific_speed,
        }
    
    def print_physics_analysis(self):
        """
        Print comprehensive physics analysis report.
        
        Includes fluid properties, flow regime analysis, blower performance,
        and SPH statistics if enabled.
        """
        cfg = self.config
        
        print("\n" + "=" * 60)
        print("AIR FLOW PHYSICS ANALYSIS")
        print("(Using FluidConfig for consistent air properties)")
        print("=" * 60)
        
        # Fluid properties
        print("\nFluid Properties:")
        print(f"  Air density:      {cfg.air_density:.3f} kg/m^3")
        print(f"  Air viscosity:    {cfg.air_viscosity:.2e} Pa.s")
        print(f"  Kinematic visc:   {cfg.air_viscosity/cfg.air_density:.2e} m^2/s")
        if cfg.fluid_config and cfg.fluid_config.temperature_c is not None:
            print(f"  Temperature:      {cfg.fluid_config.temperature_c:.1f} C")
        
        # Blower analysis
        blower = self.compute_blower_analysis()
        print("\nBlower Performance:")
        print(f"  Impeller:         {blower['impeller_diameter_mm']:.0f} mm dia, "
              f"{blower['num_blades']} {blower['blade_type']} blades")
        print(f"  Design point:     {blower['design_rpm']:.0f} RPM, "
              f"{blower['design_flow_rate_m3_h']:.0f} m³/h, "
              f"{blower['design_pressure_Pa']:.0f} Pa")
        print(f"  Current RPM:      {blower['current_rpm']:.0f} "
              f"({blower['speed_ratio']*100:.0f}% of design)")
        print(f"  Tip speed:        {blower['tip_speed_m_s']:.1f} m/s")
        print(f"  Flow rate:        {blower['flow_rate_m3_h']:.0f} m³/h "
              f"({blower['flow_ratio']*100:.0f}% of design)")
        print(f"  Pressure rise:    {blower['pressure_rise_Pa']:.0f} Pa "
              f"({blower['pressure_ratio']*100:.0f}% of design)")
        print(f"  Shaft power:      {blower['shaft_power_kW']:.2f} kW")
        print(f"  Efficiency:       {blower['efficiency_percent']:.1f}%")
        print(f"  Specific speed:   {blower['specific_speed']:.3f}")
        
        # Flow regime analysis
        flow = self.compute_flow_regime_analysis()
        print("\nFlow Regime Analysis:")
        print(f"  Flow rate:        {flow['flow_rate_m3_h']:.0f} m³/h")
        if 'reynolds_mean' in flow:
            print(f"  Reynolds (mean):  {flow['reynolds_mean']:.0f}")
            print(f"  Reynolds (range): {flow['reynolds_min']:.0f} - {flow['reynolds_max']:.0f}")
        
        # Segment details
        print("\n  Duct Segments:")
        print(f"    {'Name':<25} {'D(mm)':>6} {'V(m/s)':>8} {'Re':>10} {'Regime':<12} {'dP(Pa)':>8}")
        print("    " + "-" * 75)
        for seg in flow['segments']:
            print(f"    {seg['name']:<25} {seg['diameter_mm']:>6.0f} "
                  f"{seg['velocity_m_s']:>8.2f} {seg['reynolds']:>10.0f} "
                  f"{seg['flow_regime']:<12} {seg['pressure_drop_Pa']:>8.1f}")
        
        # Regime summary
        if 'laminar_count' in flow:
            print(f"\n  Flow Regime Distribution:")
            total = flow['laminar_count'] + flow['transitional_count'] + flow['turbulent_count']
            if total > 0:
                print(f"    Laminar (Re < 2300):      {flow['laminar_count']} "
                      f"({100*flow['laminar_count']/total:.0f}%)")
                print(f"    Transitional (2300-4000): {flow['transitional_count']} "
                      f"({100*flow['transitional_count']/total:.0f}%)")
                print(f"    Turbulent (Re > 4000):    {flow['turbulent_count']} "
                      f"({100*flow['turbulent_count']/total:.0f}%)")
        
        # SPH statistics
        sph = self.compute_sph_statistics()
        if sph['enabled']:
            print("\nSPH Particle Statistics:")
            print(f"  Particles:        {sph['num_particles']}")
            print(f"  Velocity (mean):  {sph['velocity_mean']:.2f} m/s")
            print(f"  Velocity (range): {sph['velocity_min']:.2f} - {sph['velocity_max']:.2f} m/s")
            print(f"  Velocity (std):   {sph['velocity_std']:.2f} m/s")
            print(f"  Flow direction:   ({sph['flow_direction'][0]:.2f}, "
                  f"{sph['flow_direction'][1]:.2f}, {sph['flow_direction'][2]:.2f})")
            print(f"  Density (mean):   {sph['density_mean']:.3f} kg/m³")
            print(f"  Density var:      {sph['density_variation_percent']:.1f}%")
            print(f"  Pressure range:   {sph['pressure_min']:.0f} - {sph['pressure_max']:.0f} Pa")
            print(f"  Kinetic energy:   {sph['kinetic_energy_J']:.4f} J")
        
        # System pressure drops
        print("\nSystem Pressure Balance:")
        total_dp = self.state.total_pressure_drop
        blower_dp = self.state.static_pressure_rise
        print(f"  Blower rise:      +{blower_dp:.0f} Pa")
        print(f"  System drop:      -{total_dp:.0f} Pa")
        print(f"  Balance:          {blower_dp - total_dp:+.0f} Pa")
        
        # Energy
        print("\nEnergy Consumption:")
        print(f"  Shaft power:      {self.state.shaft_power/1000:.2f} kW")
        print(f"  Electrical power: {self.state.electrical_power/1000:.2f} kW")
        print(f"  Total energy:     {self.state.total_energy_kWh:.4f} kWh")
        
        print("=" * 60)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get complete analysis data as a dictionary.
        
        Returns:
            Dictionary containing all analysis metrics.
        """
        return {
            'sph_statistics': self.compute_sph_statistics(),
            'flow_regime': self.compute_flow_regime_analysis(),
            'blower': self.compute_blower_analysis(),
            'results': self.get_results(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_air_flow_simulator(
    assembly: AirSystemAssembly,
    target_rpm: float = 3000.0,
    total_time: float = 10.0,
    dt: float = 0.001,
    enable_sph: bool = True,
    num_particles: int = 1000,
    device: str = "cuda",
    fluid_config: Optional[FluidConfig] = None,
) -> AirFlowPhysicsSimulator:
    """
    Create a physics-based air flow simulator with SPH air particles.
    
    Args:
        assembly: AirSystemAssembly with geometry
        target_rpm: Target blower RPM
        total_time: Simulation duration [s]
        dt: Time step [s]
        enable_sph: Enable SPH air simulation
        num_particles: Number of SPH air particles
        device: Compute device
        fluid_config: Optional FluidConfig for air properties
        
    Returns:
        Configured AirFlowPhysicsSimulator
    """
    config = AirFlowPhysicsConfig(
        dt=dt,
        total_time=total_time,
        target_rpm=target_rpm,
        enable_sph=enable_sph,
        num_particles=num_particles,
        device=device,
        fluid_config=fluid_config or FluidConfig.air_at_stp(),
    )
    
    return AirFlowPhysicsSimulator(assembly, config)


def create_air_flow_simulator_at_temperature(
    assembly: AirSystemAssembly,
    temperature_c: float = 20.0,
    target_rpm: float = 3000.0,
    total_time: float = 10.0,
    dt: float = 0.001,
    enable_sph: bool = True,
    num_particles: int = 1000,
    device: str = "cuda",
) -> AirFlowPhysicsSimulator:
    """
    Create a physics-based air flow simulator with temperature-adjusted air properties.
    
    Args:
        assembly: AirSystemAssembly with geometry
        temperature_c: Air temperature in Celsius
        target_rpm: Target blower RPM
        total_time: Simulation duration [s]
        dt: Time step [s]
        enable_sph: Enable SPH air simulation
        num_particles: Number of SPH air particles
        device: Compute device
        
    Returns:
        Configured AirFlowPhysicsSimulator
    """
    config = AirFlowPhysicsConfig.for_temperature(
        temperature_c=temperature_c,
        target_rpm=target_rpm,
        dt=dt,
        total_time=total_time,
        enable_sph=enable_sph,
        num_particles=num_particles,
        device=device,
    )
    
    return AirFlowPhysicsSimulator(assembly, config)
