"""
Classification Flow Physics Module
==================================

Physics-based simulation for the classification system using NVIDIA Warp.

This module simulates particle separation in the classification system:
- Venturi Eductor: Particle entrainment into airstream
- Zigzag Classifier: Primary separation by terminal velocity
- Multi-Cyclone System: Staged collection of fines
- Bag Filter: Final fine particle capture

Physics implemented:
- Two-phase flow: air velocity field + particle dynamics
- Drag: Schiller-Naumann correlation with relative velocity
- Gravity with buoyancy correction
- Inelastic wall collisions with restitution and friction
- Centrifugal effects in cyclones
- Turbulent dispersion in zigzag stages

Coordinate System (Y-up):
- Origin at venturi air inlet (bottom of system)
- Y-axis: Vertical (up) - main flow direction through venturi/zigzag
- X-axis: Horizontal (right)
- Z-axis: Depth (into page)

NO magic numbers - all dimensions derived from actual geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from enum import Enum
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None
    print("Warning: Warp not available. GPU simulation disabled.")

from ..geometry.assembly.classification import (
    ClassificationSystemAssembly,
    ClassificationSystemParams,
    create_standard_classification_system,
)
from ..utils.constants import PI, TWO_PI, GRAVITY


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class ClassificationFlowConfig:
    """
    Configuration for classification flow simulation.
    
    All parameters have physical meaning - no magic numbers.
    """
    # Particle parameters
    num_particles: int = 5000
    particle_density: float = 1450.0       # [kg/m³] Typical flour/starch
    visual_particle_diameter: float = 0.002  # [m] 2mm for visualization
    
    # Air properties
    air_density: float = 1.2               # [kg/m³] At ~20°C, 1 atm
    air_viscosity: float = 1.81e-5         # [Pa·s] Dynamic viscosity
    
    # Air flow rate - determines all air velocities
    air_flow_rate_m3s: float = 0.1         # [m³/s] Volumetric flow rate (360 m³/h)
    
    # Collision parameters
    restitution: float = 0.3               # Coefficient of restitution (inelastic)
    friction: float = 0.4                  # Friction coefficient
    
    # Simulation timing
    dt: float = 0.001                      # [s] Time step (1ms for stability)
    
    # Feed rate from feed system
    particle_feed_rate: float = 100.0      # [particles/s] From deagglomerator
    
    # Turbulence parameters (for zigzag mixing)
    turbulent_intensity: float = 0.15      # Fraction of mean velocity (15%)
    
    # Compute device
    device: str = "cuda"                   # Warp device ('cuda' or 'cpu')
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dt > 0.005:
            print(f"Warning: dt={self.dt}s may be too large for stability")
        if self.particle_density < self.air_density:
            print(f"Warning: particle density < air density - particles will float")
        if self.air_flow_rate_m3s < 0.01:
            print(f"Warning: Low air flow rate may cause poor separation")


# =============================================================================
# SIMULATION STATE
# =============================================================================

class SimulationPhase(Enum):
    """Phases of the classification simulation."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class ClassificationFlowState:
    """State of the classification flow simulation."""
    # Particle arrays (Warp arrays on device)
    positions: Any = None      # wp.array(dtype=wp.vec3)
    velocities: Any = None     # wp.array(dtype=wp.vec3)
    diameters: Any = None      # wp.array(dtype=float)
    masses: Any = None         # wp.array(dtype=float)
    zones: Any = None          # wp.array(dtype=wp.int32) - which component
    is_active: Any = None      # wp.array(dtype=wp.int32) - 1=active, 0=inactive
    
    # Simulation state
    time: float = 0.0
    step: int = 0
    phase: SimulationPhase = SimulationPhase.IDLE
    
    # Particle feed tracking
    particles_fed: int = 0
    total_particles_to_feed: int = 0
    last_feed_time: float = 0.0
    
    # Collection tracking
    collected_fines: int = 0      # Protein-rich (light particles)
    collected_coarse: int = 0     # Starch-rich (heavy particles)
    collected_cyclone: Dict[str, int] = field(default_factory=dict)
    collected_bagfilter: int = 0
    exited_clean_air: int = 0     # Particles that escaped with clean air


# =============================================================================
# ZONE DEFINITIONS
# =============================================================================

class Zone(Enum):
    """
    Zone IDs for particle tracking through classification system.
    
    Flow path:
    VENTURI -> DUCT_V_Z -> ZIGZAG -> (COARSE_OUT or FINES_PATH)
    FINES_PATH: ELBOW1 -> DUCT_Z_C -> CYCLONE_PRIMARY -> ... -> BAG_FILTER
    """
    # Inactive
    INACTIVE = -1
    
    # Venturi eductor
    VENTURI_INLET = 0           # Entering via solids inlet
    VENTURI_THROAT = 1          # In throat region
    VENTURI_DIVERGENT = 2       # In divergent section
    
    # Venturi to Zigzag duct
    DUCT_VENTURI_ZIGZAG = 10
    
    # Zigzag classifier
    ZIGZAG_ENTRY = 20           # Entering zigzag
    ZIGZAG_STAGES = 21          # In zigzag stages (separation)
    ZIGZAG_FINES = 22           # Moving toward fines outlet
    ZIGZAG_COARSE = 23          # Moving toward coarse outlet
    
    # Coarse collection
    COARSE_OUTLET = 30          # Collected as coarse (starch)
    
    # Fines path to cyclones
    ELBOW_ZIGZAG_CYCLONE = 40   # Elbow after zigzag
    DUCT_ZIGZAG_CYCLONE = 41    # Horizontal duct to cyclones
    
    # Multi-cyclone system
    CYCLONE_PRIMARY = 50        # Primary cyclone (coarse fines)
    CYCLONE_SECONDARY = 51      # Secondary cyclone (medium)
    CYCLONE_TERTIARY = 52       # Tertiary cyclone (fine protein)
    
    # Cyclone dust outlets
    DUST_PRIMARY = 55
    DUST_SECONDARY = 56
    DUST_TERTIARY = 57
    
    # Cyclone to bag filter
    ELBOW_CYCLONE_BAG = 60
    DUCT_CYCLONE_BAG = 61
    
    # Bag filter
    BAG_FILTER = 70             # In bag filter
    DUST_BAGFILTER = 75         # Collected in bag filter hopper
    
    # Exit
    CLEAN_AIR_EXIT = 80         # Escaped with clean air (should be rare)
    EXITED = 99                 # Exited system (collected)


# =============================================================================
# GEOMETRY EXTRACTION
# =============================================================================

@dataclass
class ComponentGeometry:
    """Extracted geometry from a classification system component."""
    center: np.ndarray          # World position of component center
    axis: str                   # Main axis ('x', 'y', 'z')
    
    # Cylindrical/housing parameters
    radius: float = 0.0         # Main radius
    length: float = 0.0         # Axial length
    
    # Port parameters (computed from actual ports)
    inlet_pos: np.ndarray = None
    inlet_dir: np.ndarray = None
    inlet_diameter: float = 0.0
    inlet_width: float = 0.0    # For rectangular ports
    inlet_height: float = 0.0
    
    outlet_pos: np.ndarray = None
    outlet_dir: np.ndarray = None
    outlet_diameter: float = 0.0
    outlet_width: float = 0.0
    outlet_height: float = 0.0
    
    # Venturi-specific
    throat_diameter: float = 0.0
    throat_start: float = 0.0
    throat_end: float = 0.0
    solids_inlet_pos: np.ndarray = None
    solids_inlet_dir: np.ndarray = None
    solids_inlet_diameter: float = 0.0
    
    # Zigzag-specific
    channel_width: float = 0.0
    channel_depth: float = 0.0
    num_stages: int = 0
    stage_height: float = 0.0
    total_height: float = 0.0
    fines_outlet_pos: np.ndarray = None
    coarse_outlet_pos: np.ndarray = None
    
    # Cyclone-specific
    cylinder_diameter: float = 0.0
    cylinder_height: float = 0.0
    cone_height: float = 0.0
    vortex_finder_diameter: float = 0.0
    dust_outlet_diameter: float = 0.0
    
    # Bag filter-specific
    housing_width: float = 0.0
    housing_depth: float = 0.0
    housing_height: float = 0.0


@dataclass
class ConnectionPath:
    """Geometry for the path between two components."""
    name: str
    start_pos: np.ndarray       # World position of start
    end_pos: np.ndarray         # World position of end
    direction: np.ndarray       # Unit vector from start to end
    length: float               # Distance
    start_diameter: float       # Diameter/size at start
    end_diameter: float         # Diameter/size at end
    avg_radius: float           # Average radius for containment
    is_elbow: bool = False      # True if this is an elbow section
    bend_radius: float = 0.0    # Bend radius for elbows


def extract_geometry(assembly: ClassificationSystemAssembly) -> Dict[str, Any]:
    """
    Extract all geometry parameters from a ClassificationSystemAssembly.
    
    This function computes all dimensions from the actual component geometry,
    ensuring no magic numbers are used in the simulation.
    
    Args:
        assembly: ClassificationSystemAssembly instance
        
    Returns:
        Dictionary mapping component names to their geometry
    """
    positions = assembly.get_component_positions()
    geometry = {}
    
    # =========================================================================
    # VENTURI EDUCTOR GEOMETRY
    # =========================================================================
    venturi = assembly.venturi
    venturi_pos = np.array(positions['venturi'])
    venturi_ports = venturi.ports
    vp = venturi.params
    
    geometry['venturi'] = ComponentGeometry(
        center=venturi_pos,
        axis=vp.axis,  # Typically 'y' - vertical
        radius=vp.inlet_diameter / 2,
        length=vp.total_length,
        # Air inlet (bottom)
        inlet_pos=venturi_pos + np.array(venturi_ports['air_inlet'].position),
        inlet_dir=np.array(venturi_ports['air_inlet'].direction),
        inlet_diameter=venturi_ports['air_inlet'].diameter,
        # Outlet (top)
        outlet_pos=venturi_pos + np.array(venturi_ports['outlet'].position),
        outlet_dir=np.array(venturi_ports['outlet'].direction),
        outlet_diameter=venturi_ports['outlet'].diameter,
        # Venturi-specific
        throat_diameter=vp.throat_diameter,
        throat_start=vp.throat_start_position,
        throat_end=vp.throat_end_position,
        # Solids inlet
        solids_inlet_pos=venturi_pos + np.array(venturi_ports['solids_inlet'].position),
        solids_inlet_dir=np.array(venturi_ports['solids_inlet'].direction),
        solids_inlet_diameter=venturi_ports['solids_inlet'].diameter,
    )
    
    # =========================================================================
    # ZIGZAG CLASSIFIER GEOMETRY
    # =========================================================================
    zigzag = assembly.zigzag
    zigzag_pos = np.array(positions['zigzag'])
    zigzag_ports = zigzag.ports
    zp = zigzag.params
    
    geometry['zigzag'] = ComponentGeometry(
        center=zigzag_pos,
        axis='y',  # Vertical classifier
        length=zp.total_height,
        total_height=zp.total_height,
        # Channel geometry
        channel_width=zp.channel_width,
        channel_depth=zp.channel_depth,
        num_stages=zp.num_stages,
        stage_height=zp.stage_height,
        # Air inlet (bottom)
        inlet_pos=zigzag_pos + np.array(zigzag_ports['air_inlet'].position),
        inlet_dir=np.array(zigzag_ports['air_inlet'].direction),
        inlet_width=zigzag_ports['air_inlet'].width,
        inlet_height=zigzag_ports['air_inlet'].height if hasattr(zigzag_ports['air_inlet'], 'height') else zp.channel_depth,
        # Fines outlet (top)
        fines_outlet_pos=zigzag_pos + np.array(zigzag_ports['fines_outlet'].position),
        outlet_pos=zigzag_pos + np.array(zigzag_ports['fines_outlet'].position),
        outlet_dir=np.array(zigzag_ports['fines_outlet'].direction),
        outlet_width=zigzag_ports['fines_outlet'].width,
        outlet_height=zigzag_ports['fines_outlet'].height if hasattr(zigzag_ports['fines_outlet'], 'height') else zp.channel_depth,
        # Coarse outlet (bottom)
        coarse_outlet_pos=zigzag_pos + np.array(zigzag_ports['coarse_outlet'].position),
    )
    
    # =========================================================================
    # MULTI-CYCLONE SYSTEM GEOMETRY
    # =========================================================================
    multi_cyclone = assembly.multi_cyclone
    cyclone_pos = np.array(positions['multi_cyclone'])
    cyclone_ports = multi_cyclone.ports
    
    # Get primary cyclone params for main geometry
    primary_cyclone = multi_cyclone._cyclones[multi_cyclone.params.stages[0].name]
    cp = primary_cyclone.params
    
    geometry['multi_cyclone'] = ComponentGeometry(
        center=cyclone_pos,
        axis='y',  # Vertical cyclones
        # Primary cyclone dimensions
        cylinder_diameter=cp.cylinder_diameter,
        radius=cp.cylinder_diameter / 2,
        cylinder_height=cp.cylinder_height,
        cone_height=cp.cone_height,
        length=cp.cylinder_height + cp.cone_height,
        vortex_finder_diameter=cp.vortex_finder_diameter,
        dust_outlet_diameter=cp.dust_outlet_diameter,
        # Inlet (tangential, rectangular)
        inlet_pos=cyclone_pos + np.array(cyclone_ports['inlet'].position),
        inlet_dir=np.array(cyclone_ports['inlet'].direction),
        inlet_width=cyclone_ports['inlet'].width,
        inlet_height=cyclone_ports['inlet'].height,
        # Overflow outlet (top)
        outlet_pos=cyclone_pos + np.array(cyclone_ports['overflow'].position),
        outlet_dir=np.array(cyclone_ports['overflow'].direction),
        outlet_diameter=cyclone_ports['overflow'].diameter,
    )
    
    # Store individual cyclone positions and params
    geometry['cyclone_stages'] = {}
    for stage in multi_cyclone.params.stages:
        stage_cyclone = multi_cyclone._cyclones[stage.name]
        stage_pos = multi_cyclone._cyclone_positions[stage.name]
        stage_params = stage_cyclone.params
        
        geometry['cyclone_stages'][stage.name] = {
            'position': np.array(stage_pos),
            'diameter': stage_params.cylinder_diameter,
            'cylinder_height': stage_params.cylinder_height,
            'cone_height': stage_params.cone_height,
            'vortex_finder_diameter': stage_params.vortex_finder_diameter,
            'dust_outlet_diameter': stage_params.dust_outlet_diameter,
            'dust_outlet_pos': cyclone_pos + np.array(cyclone_ports[f'dust_outlet_{stage.name}'].position),
        }
    
    # =========================================================================
    # BAG FILTER GEOMETRY
    # =========================================================================
    bag_filter = assembly.bag_filter
    bag_pos = np.array(positions['bag_filter'])
    bag_ports = bag_filter.ports
    bp = bag_filter.params
    
    geometry['bag_filter'] = ComponentGeometry(
        center=bag_pos,
        axis='y',  # Vertical filter
        # Housing dimensions
        housing_width=bp.housing_width,
        housing_depth=bp.housing_depth,
        housing_height=bp.housing_height,
        length=bp.housing_height,
        # Dirty air inlet (side)
        inlet_pos=bag_pos + np.array(bag_ports['dirty_air_inlet'].position),
        inlet_dir=np.array(bag_ports['dirty_air_inlet'].direction),
        inlet_diameter=bag_ports['dirty_air_inlet'].diameter,
        # Clean air outlet (top)
        outlet_pos=bag_pos + np.array(bag_ports['clean_air_outlet'].position),
        outlet_dir=np.array(bag_ports['clean_air_outlet'].direction),
        outlet_diameter=bag_ports['clean_air_outlet'].diameter,
        # Dust outlet (bottom)
        coarse_outlet_pos=bag_pos + np.array(bag_ports['dust_outlet'].position),
    )
    
    # =========================================================================
    # DUCT SECTIONS (from assembly)
    # =========================================================================
    geometry['ducts'] = []
    if hasattr(assembly, '_duct_sections'):
        for duct, position in assembly._duct_sections:
            duct_info = {
                'position': np.array(position),
                'type': type(duct).__name__,
            }
            if hasattr(duct, 'params'):
                if hasattr(duct.params, 'diameter'):
                    duct_info['diameter'] = duct.params.diameter
                if hasattr(duct.params, 'length'):
                    duct_info['length'] = duct.params.length
                if hasattr(duct.params, 'bend_radius'):
                    duct_info['bend_radius'] = duct.params.bend_radius
                    duct_info['is_elbow'] = True
                if hasattr(duct.params, 'direction'):
                    duct_info['direction'] = np.array(duct.params.direction)
            geometry['ducts'].append(duct_info)
    
    # =========================================================================
    # CONNECTION PATHS (computed from actual port positions)
    # =========================================================================
    connections = {}
    
    # Venturi outlet -> Zigzag air inlet
    venturi_out = geometry['venturi'].outlet_pos
    zigzag_in = geometry['zigzag'].inlet_pos
    conn_vec = zigzag_in - venturi_out
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['venturi_to_zigzag'] = {
        'start_pos': venturi_out.copy(),
        'end_pos': zigzag_in.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['venturi'].outlet_diameter,
        'end_diameter': geometry['zigzag'].inlet_width,  # Rectangular
        'avg_radius': geometry['venturi'].outlet_diameter / 2,
    }
    
    # Zigzag fines -> Cyclone inlet
    zigzag_fines = geometry['zigzag'].fines_outlet_pos
    cyclone_in = geometry['multi_cyclone'].inlet_pos
    conn_vec = cyclone_in - zigzag_fines
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['zigzag_to_cyclone'] = {
        'start_pos': zigzag_fines.copy(),
        'end_pos': cyclone_in.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['zigzag'].outlet_width,
        'end_diameter': geometry['multi_cyclone'].inlet_width,
        'avg_radius': (geometry['zigzag'].outlet_width + geometry['multi_cyclone'].inlet_width) / 4,
    }
    
    # Cyclone overflow -> Bag filter inlet
    cyclone_out = geometry['multi_cyclone'].outlet_pos
    bag_in = geometry['bag_filter'].inlet_pos
    conn_vec = bag_in - cyclone_out
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    
    connections['cyclone_to_bagfilter'] = {
        'start_pos': cyclone_out.copy(),
        'end_pos': bag_in.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': geometry['multi_cyclone'].outlet_diameter,
        'end_diameter': geometry['bag_filter'].inlet_diameter,
        'avg_radius': (geometry['multi_cyclone'].outlet_diameter + geometry['bag_filter'].inlet_diameter) / 4,
    }
    
    geometry['connections'] = connections
    
    return geometry


def print_geometry_summary(geometry: Dict[str, Any]):
    """Print a summary of extracted geometry for debugging."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION SYSTEM GEOMETRY SUMMARY")
    print("=" * 70)
    
    # Venturi
    v = geometry['venturi']
    print("\n1. VENTURI EDUCTOR")
    print(f"   Center:           ({v.center[0]*1000:.1f}, {v.center[1]*1000:.1f}, {v.center[2]*1000:.1f}) mm")
    print(f"   Axis:             {v.axis}")
    print(f"   Inlet diameter:   {v.inlet_diameter*1000:.1f} mm")
    print(f"   Throat diameter:  {v.throat_diameter*1000:.1f} mm")
    print(f"   Outlet diameter:  {v.outlet_diameter*1000:.1f} mm")
    print(f"   Total length:     {v.length*1000:.1f} mm")
    print(f"   Solids inlet pos: ({v.solids_inlet_pos[0]*1000:.1f}, {v.solids_inlet_pos[1]*1000:.1f}, {v.solids_inlet_pos[2]*1000:.1f}) mm")
    
    # Zigzag
    z = geometry['zigzag']
    print("\n2. ZIGZAG CLASSIFIER")
    print(f"   Center:           ({z.center[0]*1000:.1f}, {z.center[1]*1000:.1f}, {z.center[2]*1000:.1f}) mm")
    print(f"   Channel width:    {z.channel_width*1000:.1f} mm")
    print(f"   Channel depth:    {z.channel_depth*1000:.1f} mm")
    print(f"   Stages:           {z.num_stages}")
    print(f"   Stage height:     {z.stage_height*1000:.1f} mm")
    print(f"   Total height:     {z.total_height*1000:.1f} mm")
    
    # Multi-cyclone
    c = geometry['multi_cyclone']
    print("\n3. MULTI-CYCLONE SYSTEM")
    print(f"   Center:           ({c.center[0]*1000:.1f}, {c.center[1]*1000:.1f}, {c.center[2]*1000:.1f}) mm")
    print(f"   Primary diameter: {c.cylinder_diameter*1000:.1f} mm")
    print(f"   Inlet (WxH):      {c.inlet_width*1000:.1f} x {c.inlet_height*1000:.1f} mm")
    print(f"   Overflow dia:     {c.outlet_diameter*1000:.1f} mm")
    
    for name, stage in geometry['cyclone_stages'].items():
        print(f"   {name.title():12s} D={stage['diameter']*1000:.0f}mm at ({stage['position'][0]*1000:.0f}, {stage['position'][1]*1000:.0f}, {stage['position'][2]*1000:.0f})")
    
    # Bag filter
    b = geometry['bag_filter']
    print("\n4. BAG FILTER")
    print(f"   Center:           ({b.center[0]*1000:.1f}, {b.center[1]*1000:.1f}, {b.center[2]*1000:.1f}) mm")
    print(f"   Housing (WxDxH):  {b.housing_width*1000:.1f} x {b.housing_depth*1000:.1f} x {b.housing_height*1000:.1f} mm")
    print(f"   Inlet diameter:   {b.inlet_diameter*1000:.1f} mm")
    print(f"   Outlet diameter:  {b.outlet_diameter*1000:.1f} mm")
    
    # Connections
    print("\n5. CONNECTION PATHS")
    for name, conn in geometry['connections'].items():
        print(f"   {name}:")
        print(f"     Length:     {conn['length']*1000:.1f} mm")
        print(f"     Direction:  ({conn['direction'][0]:.2f}, {conn['direction'][1]:.2f}, {conn['direction'][2]:.2f})")
    
    print("\n" + "=" * 70)


# =============================================================================
# WARP PHYSICS FUNCTIONS
# =============================================================================
#
# FUNDAMENTAL PHYSICS FOR PROTEIN/STARCH SEPARATION
# =================================================
#
# Air classification separates particles based on their TERMINAL VELOCITY:
# - Terminal velocity: v_t = sqrt(4 * d_p * g * (rho_p - rho_f) / (3 * C_d * rho_f))
# - Particles with v_t < v_air rise (fines/protein)
# - Particles with v_t > v_air fall (coarse/starch)
#
# Key dimensionless numbers:
# - Reynolds number: Re = rho_f * v_rel * d_p / mu
# - Stokes number: St = rho_p * d_p^2 * v / (18 * mu * L)
#   St >> 1: inertia-dominated (particle doesn't follow flow)
#   St << 1: drag-dominated (particle follows flow)
#
# Separation mechanisms by component:
# 1. VENTURI: Entrainment via Bernoulli pressure drop at throat
# 2. ZIGZAG: Counter-current separation with turbulent mixing stages
# 3. CYCLONE: Centrifugal vs drag force balance
# 4. BAG FILTER: Inertial impaction and interception
#
# =============================================================================

if wp is not None:

    # -------------------------------------------------------------------------
    # PARTICLE REYNOLDS NUMBER
    # -------------------------------------------------------------------------
    @wp.func
    def compute_particle_reynolds(
        diameter: float,
        v_rel_mag: float,
        rho_f: float,
        mu_f: float
    ) -> float:
        """
        Particle Reynolds number: Re_p = rho_f * |v_rel| * d_p / mu
        
        Determines the drag regime:
        - Re < 0.1: Stokes (creeping flow)
        - 0.1 < Re < 1000: Intermediate (Schiller-Naumann)
        - Re > 1000: Newton (turbulent wake)
        
        For flour particles (10-100 um) in air at 10-20 m/s:
        Re typically 0.01 - 100 (Stokes to intermediate)
        """
        eps = 1.0e-10
        return rho_f * v_rel_mag * diameter / wp.max(mu_f, eps)

    # -------------------------------------------------------------------------
    # DRAG COEFFICIENT MODELS
    # -------------------------------------------------------------------------
    @wp.func
    def drag_coefficient_stokes(Re: float) -> float:
        """
        Stokes drag: C_d = 24/Re
        Valid for Re < 0.1 (very small particles, slow relative motion)
        
        For protein-rich fines (d_p ~ 10-30 um), often in Stokes regime.
        """
        eps = 1.0e-10
        if Re < eps:
            return 24.0 / eps
        return 24.0 / Re

    @wp.func
    def drag_coefficient_schiller_naumann(Re: float) -> float:
        """
        Schiller-Naumann correlation: C_d = (24/Re) * (1 + 0.15 * Re^0.687)
        
        Valid for 0.1 < Re < 1000 (intermediate regime).
        Most flour/starch particles operate in this regime.
        
        This correlation smoothly transitions from Stokes to turbulent.
        """
        eps = 1.0e-10
        if Re < eps:
            return 24.0 / eps
        return (24.0 / Re) * (1.0 + 0.15 * wp.pow(Re, 0.687))

    @wp.func
    def drag_coefficient_haider_levenspiel(Re: float, sphericity: float) -> float:
        """
        Haider-Levenspiel correlation for NON-SPHERICAL particles.
        
        Flour/protein particles are NOT perfectly spherical:
        - Starch granules: phi = 0.8-0.9 (rounded)
        - Protein particles: phi = 0.6-0.8 (irregular, fibrous)
        
        C_d = (24/Re)(1 + A*Re^B) + C/(1 + D/Re)
        
        Where A, B, C, D are functions of sphericity phi.
        Non-spherical particles have HIGHER drag -> lower terminal velocity.
        """
        eps = 1.0e-10
        if Re < eps:
            Re = eps
        
        phi = sphericity
        
        # Correlation coefficients (Haider & Levenspiel, 1989)
        A = wp.exp(2.3288 - 6.4581 * phi + 2.4486 * phi * phi)
        B = 0.0964 + 0.5565 * phi
        C = wp.exp(4.905 - 13.8944 * phi + 18.4222 * phi * phi - 10.2599 * phi * phi * phi)
        D = wp.exp(1.4681 + 12.2584 * phi - 20.7322 * phi * phi + 15.8855 * phi * phi * phi)
        
        return (24.0 / Re) * (1.0 + A * wp.pow(Re, B)) + C / (1.0 + D / Re)

    # -------------------------------------------------------------------------
    # TERMINAL VELOCITY
    # -------------------------------------------------------------------------
    @wp.func
    def compute_terminal_velocity(
        diameter: float,
        rho_p: float,
        rho_f: float,
        mu_f: float,
        g: float
    ) -> float:
        """
        Terminal velocity: the settling velocity where drag = gravity - buoyancy.
        
        This is THE key parameter for air classification:
        - v_t determines if particle rises or falls in the airstream
        - Separation occurs at the "cut size" where v_t = v_air
        
        For Stokes regime (small particles):
        v_t = d_p^2 * g * (rho_p - rho_f) / (18 * mu)
        
        For intermediate regime, solved iteratively (here using approximation).
        
        Typical values for flour at 20C:
        - 10 um protein: v_t = 0.005 m/s
        - 50 um starch:  v_t = 0.12 m/s
        - 100 um starch: v_t = 0.4 m/s
        """
        eps = 1.0e-10
        
        # Buoyancy-corrected density difference
        delta_rho = rho_p - rho_f
        if delta_rho < eps:
            return 0.0  # Neutrally buoyant
        
        # Stokes terminal velocity (valid for Re < 0.1)
        v_stokes = diameter * diameter * g * delta_rho / (18.0 * mu_f)
        
        # Check Reynolds number
        Re = rho_f * v_stokes * diameter / mu_f
        
        if Re < 0.1:
            return v_stokes
        
        # For intermediate regime, use iterative correction
        # Approximate: v_t = v_stokes / (1 + 0.15 * Re^0.687)^0.5
        correction = 1.0 + 0.15 * wp.pow(Re, 0.687)
        return v_stokes / wp.sqrt(correction)

    # -------------------------------------------------------------------------
    # STOKES NUMBER (Inertia vs Drag)
    # -------------------------------------------------------------------------
    @wp.func
    def compute_stokes_number(
        diameter: float,
        rho_p: float,
        v_char: float,
        mu_f: float,
        L_char: float
    ) -> float:
        """
        Stokes number: St = tau_p * v / L = (rho_p * d_p^2 * v) / (18 * mu * L)
        
        tau_p = particle relaxation time (time to respond to flow changes)
        
        Physical meaning:
        - St >> 1: Particle has high inertia, doesn't follow flow (impacts walls)
        - St << 1: Particle follows flow closely (carried by air)
        - St ~ 1: Intermediate behavior (ideal for separation)
        
        In cyclones: St determines separation efficiency
        - Large St -> particle spirals to wall -> collected
        - Small St -> particle follows air to vortex finder -> escapes
        """
        eps = 1.0e-10
        tau_p = rho_p * diameter * diameter / (18.0 * mu_f)
        return tau_p * v_char / wp.max(L_char, eps)

    # -------------------------------------------------------------------------
    # GRAVITY WITH BUOYANCY
    # -------------------------------------------------------------------------
    @wp.func
    def compute_gravity_buoyancy(
        rho_p: float,
        rho_f: float,
        g: float
    ) -> wp.vec3:
        """
        Gravitational acceleration with buoyancy correction.
        
        a_g = g * (1 - rho_f/rho_p) in -Y direction
        
        For flour in air:
        - rho_p = 1400 kg/m^3 (protein/starch)
        - rho_f = 1.2 kg/m^3 (air)
        - Buoyancy factor: (1 - 1.2/1400) = 0.999 (negligible buoyancy)
        
        But buoyancy is more significant for:
        - Very light particles (e.g., hollow fibers)
        - Denser fluids (e.g., fluidized beds)
        """
        buoyancy_factor = 1.0 - rho_f / rho_p
        return wp.vec3(0.0, -g * buoyancy_factor, 0.0)

    # -------------------------------------------------------------------------
    # DRAG ACCELERATION (Two-Phase Flow)
    # -------------------------------------------------------------------------
    @wp.func
    def compute_drag_acceleration(
        v_particle: wp.vec3,
        v_fluid: wp.vec3,
        diameter: float,
        mass: float,
        rho_f: float,
        mu_f: float
    ) -> wp.vec3:
        """
        Drag acceleration from fluid-particle relative velocity.
        
        F_drag = 0.5 * C_d * rho_f * A_p * |v_rel|^2 * (v_rel / |v_rel|)
        a_drag = F_drag / m_p
        
        Key insight for separation:
        - Drag acts in direction of RELATIVE velocity (v_fluid - v_particle)
        - If v_fluid > v_particle (upward air): drag pushes particle UP
        - This is why light particles rise and heavy particles fall
        
        The drag-to-weight ratio determines separation:
        - F_drag/F_gravity > 1 -> particle rises
        - F_drag/F_gravity < 1 -> particle falls
        """
        v_rel = v_fluid - v_particle
        v_rel_mag = wp.length(v_rel)
        eps = 1.0e-10
        
        if v_rel_mag < eps:
            return wp.vec3(0.0, 0.0, 0.0)
        
        # Reynolds number
        Re = compute_particle_reynolds(diameter, v_rel_mag, rho_f, mu_f)
        
        # Drag coefficient (Schiller-Naumann for most flour particles)
        Cd = drag_coefficient_schiller_naumann(Re)
        
        # Projected area (sphere)
        A_p = PI / 4.0 * diameter * diameter
        
        # Drag force magnitude
        F_drag = 0.5 * Cd * rho_f * A_p * v_rel_mag * v_rel_mag
        
        # Acceleration (in direction of relative velocity)
        a_mag = F_drag / mass
        
        return v_rel * (a_mag / v_rel_mag)

    # -------------------------------------------------------------------------
    # VENTURI ENTRAINMENT PHYSICS
    # -------------------------------------------------------------------------
    @wp.func
    def compute_venturi_air_velocity(
        pos: wp.vec3,
        venturi_center: wp.vec3,
        inlet_diameter: float,
        throat_diameter: float,
        outlet_diameter: float,
        throat_start: float,
        throat_end: float,
        total_length: float,
        v_inlet: float,
        axis: int  # 0=X, 1=Y, 2=Z
    ) -> wp.vec3:
        """
        Compute air velocity field in venturi using continuity equation.
        
        BERNOULLI PRINCIPLE:
        - v1*A1 = v2*A2 (continuity)
        - At throat: v_throat = v_inlet * (D_inlet/D_throat)^2
        - Pressure drops at throat (Bernoulli): dP = 0.5*rho*(v_throat^2 - v_inlet^2)
        
        This pressure drop draws particles into the air stream.
        
        For D_inlet=80mm, D_throat=40mm:
        - Area ratio = 4
        - v_throat = 4 * v_inlet
        - If v_inlet = 15 m/s -> v_throat = 60 m/s
        """
        # Position along venturi axis
        if axis == 1:  # Y-axis (vertical)
            axial_pos = pos[1] - venturi_center[1]
        elif axis == 0:  # X-axis
            axial_pos = pos[0] - venturi_center[0]
        else:  # Z-axis
            axial_pos = pos[2] - venturi_center[2]
        
        # Clamp to venturi length
        axial_pos = wp.clamp(axial_pos, 0.0, total_length)
        
        # Determine diameter at this position
        if axial_pos < throat_start:
            # Convergent section: linear interpolation
            t = axial_pos / throat_start
            D = inlet_diameter + t * (throat_diameter - inlet_diameter)
        elif axial_pos < throat_end:
            # Throat section: constant diameter
            D = throat_diameter
        else:
            # Divergent section: linear interpolation
            t = (axial_pos - throat_end) / (total_length - throat_end)
            D = throat_diameter + t * (outlet_diameter - throat_diameter)
        
        # Continuity: v * A = constant -> v = v_inlet * (D_inlet/D)^2
        eps = 1.0e-6
        D = wp.max(D, eps)
        area_ratio = (inlet_diameter / D) * (inlet_diameter / D)
        v_local = v_inlet * area_ratio
        
        # Velocity vector along axis
        if axis == 1:
            return wp.vec3(0.0, v_local, 0.0)
        elif axis == 0:
            return wp.vec3(v_local, 0.0, 0.0)
        else:
            return wp.vec3(0.0, 0.0, v_local)

    # -------------------------------------------------------------------------
    # ZIGZAG CLASSIFIER PHYSICS
    # -------------------------------------------------------------------------
    @wp.func
    def compute_zigzag_air_velocity(
        pos: wp.vec3,
        zigzag_center: wp.vec3,
        channel_width: float,
        total_height: float,
        num_stages: int,
        v_mean: float,
        stage_height: float
    ) -> wp.vec3:
        """
        Compute air velocity field in zigzag classifier.
        
        SEPARATION PRINCIPLE:
        - Air flows upward at mean velocity v_mean
        - At each zigzag stage, flow accelerates/decelerates
        - Particles with v_terminal < v_mean rise (fines)
        - Particles with v_terminal > v_mean fall (coarse)
        
        The zigzag geometry creates:
        1. Velocity variations (acceleration at constrictions)
        2. Turbulent recirculation zones
        3. Multiple separation stages (each a "cut")
        
        This returns the LOCAL air velocity at the particle position.
        """
        # Height in classifier (from bottom)
        local_y = pos[1] - zigzag_center[1]
        
        # Base upward velocity
        v_y = v_mean
        
        # Stage-dependent velocity variation
        # At each zigzag, flow accelerates on the inside of the turn
        if num_stages > 0 and stage_height > 0.0:
            stage_num = int(local_y / stage_height)
            pos_in_stage = local_y - float(stage_num) * stage_height
            
            # Velocity varies sinusoidally within each stage (simplified model)
            # Maximum at stage center, minimum at transitions
            t = pos_in_stage / stage_height
            v_variation = 0.3 * wp.sin(t * 2.0 * PI)  # +/-30% variation
            v_y = v_mean * (1.0 + v_variation)
        
        return wp.vec3(0.0, v_y, 0.0)

    @wp.func
    def compute_turbulent_dispersion(
        vel: wp.vec3,
        turbulent_intensity: float,
        seed: int,
        tid: int
    ) -> wp.vec3:
        """
        Add turbulent velocity fluctuations for zigzag mixing.
        
        TURBULENT DISPERSION:
        In real zigzag classifiers, turbulence is essential for:
        1. Keeping particles suspended
        2. Promoting mixing between stages
        3. Creating probability-based separation (not deterministic)
        
        v' = I * v_mean * random_direction
        
        Where I = turbulent intensity (typically 0.1-0.2 for zigzag)
        
        This causes some "misclassification" - a few protein particles
        end up in coarse, and vice versa. This is physically realistic.
        """
        # Simple pseudo-random based on particle ID and seed
        # In practice, use Warp's random functions
        phase = float(tid * 17 + seed * 31) * 0.1
        
        # Random velocity components (simplified Gaussian-like)
        v_mag = wp.length(vel)
        fluctuation = turbulent_intensity * v_mag
        
        vx = fluctuation * wp.sin(phase * 1.1)
        vy = fluctuation * wp.cos(phase * 2.3)
        vz = fluctuation * wp.sin(phase * 0.7 + 1.5)
        
        return wp.vec3(vx, vy, vz)

    # -------------------------------------------------------------------------
    # CYCLONE SEPARATION PHYSICS
    # -------------------------------------------------------------------------
    @wp.func
    def compute_cyclone_tangential_velocity(
        pos: wp.vec3,
        cyclone_center: wp.vec3,
        inlet_velocity: float,
        cyclone_radius: float
    ) -> wp.vec3:
        """
        Compute tangential velocity field in cyclone.
        
        CYCLONE FLOW PATTERN:
        1. Outer vortex: spirals downward along walls
        2. Inner vortex: spirals upward through core to vortex finder
        
        Tangential velocity profile (Rankine vortex model):
        - Inner region (r < r_core): v_tan = omega * r (solid body rotation)
        - Outer region (r > r_core): v_tan = Gamma / (2*pi * r) (free vortex)
        
        For separation:
        - Centrifugal force: F_c = m * v_tan^2 / r (pushes particles out)
        - Drag force: toward center (air flows inward)
        - Balance determines particle trajectory
        """
        # Radial position from cyclone axis
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        
        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)
        
        # Core radius (typically 0.3-0.5 of cyclone radius)
        r_core = cyclone_radius * 0.4
        
        # Tangential velocity (Rankine vortex)
        if r < r_core:
            # Solid body rotation in core
            v_tan = inlet_velocity * r / r_core
        else:
            # Free vortex in outer region (conserved angular momentum)
            v_tan = inlet_velocity * r_core / r
        
        # Tangential direction (perpendicular to radial in XZ plane)
        # Counter-clockwise when viewed from above
        tan_x = -dz / r
        tan_z = dx / r
        
        return wp.vec3(v_tan * tan_x, 0.0, v_tan * tan_z)

    @wp.func
    def compute_cyclone_radial_velocity(
        pos: wp.vec3,
        cyclone_center: wp.vec3,
        cyclone_radius: float,
        vortex_finder_radius: float,
        cylinder_height: float,
        cone_height: float,
        v_inlet: float
    ) -> wp.vec3:
        """
        Compute radial velocity component in cyclone.
        
        RADIAL FLOW:
        - Outer region: slight inward flow toward core
        - Inner region: stronger inward flow toward vortex finder
        
        This radial drag competes with centrifugal force:
        - Large particles: centrifugal > drag -> move to wall -> collected
        - Small particles: drag > centrifugal -> follow air -> escape
        """
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        local_y = pos[1] - cyclone_center[1]
        
        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)
        
        # Radial velocity magnitude (inward, toward axis)
        # Stronger near vortex finder, weaker near walls
        r_normalized = r / cyclone_radius
        v_radial_mag = -0.1 * v_inlet * (1.0 - r_normalized)  # Inward
        
        # In cone section, radial flow intensifies
        total_height = cylinder_height + cone_height
        if local_y < -cylinder_height:  # In cone
            cone_factor = 1.0 + (-local_y - cylinder_height) / cone_height
            v_radial_mag = v_radial_mag * cone_factor
        
        # Radial direction (inward)
        return wp.vec3(v_radial_mag * dx / r, 0.0, v_radial_mag * dz / r)

    @wp.func
    def compute_cyclone_axial_velocity(
        pos: wp.vec3,
        cyclone_center: wp.vec3,
        cyclone_radius: float,
        vortex_finder_radius: float,
        v_inlet: float
    ) -> wp.vec3:
        """
        Compute axial (vertical) velocity in cyclone.
        
        AXIAL FLOW PATTERN:
        - Outer region (r > r_vf): DOWNWARD toward dust outlet
        - Inner region (r < r_vf): UPWARD toward vortex finder
        
        This creates the characteristic "double helix" flow:
        - Dirty air spirals down along wall
        - Clean air spirals up through center
        
        Particles that reach the wall spiral down and are collected.
        Particles that stay in the core escape with clean air.
        """
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        
        # Transition radius (approximately vortex finder radius)
        r_transition = vortex_finder_radius * 1.2
        
        if r > r_transition:
            # Outer region: downward flow
            v_axial = -0.2 * v_inlet
        else:
            # Inner region: upward flow (toward vortex finder)
            # Stronger toward center
            inner_factor = 1.0 - r / r_transition
            v_axial = 0.5 * v_inlet * inner_factor
        
        return wp.vec3(0.0, v_axial, 0.0)

    @wp.func
    def compute_centrifugal_acceleration(
        pos: wp.vec3,
        vel: wp.vec3,
        axis_center: wp.vec3,
        axis_dir: wp.vec3
    ) -> wp.vec3:
        """
        Centrifugal acceleration: a_c = v_tan^2 / r (outward)
        
        SEPARATION MECHANISM:
        - Centrifugal force is proportional to mass * v^2 / r
        - Larger/denser particles experience more centrifugal force
        - They migrate to the outer wall and spiral down
        
        This is why cyclones separate by AERODYNAMIC diameter:
        d_ae = d_p * sqrt(rho_p / rho_ref)
        
        Both size AND density matter for separation.
        """
        # Vector from axis to particle
        to_pos = pos - axis_center
        
        # Remove axial component to get radial vector
        axial = wp.dot(to_pos, axis_dir) * axis_dir
        radial = to_pos - axial
        r = wp.length(radial)
        
        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)
        
        # Tangential velocity component
        radial_unit = radial / r
        tangent = wp.cross(axis_dir, radial_unit)
        v_tan = wp.dot(vel, tangent)
        
        # Centrifugal acceleration (outward)
        a_centrifugal = v_tan * v_tan / r
        
        return radial_unit * a_centrifugal

    # -------------------------------------------------------------------------
    # WALL COLLISION PHYSICS
    # -------------------------------------------------------------------------
    @wp.func
    def reflect_velocity_inelastic(
        vel: wp.vec3,
        normal: wp.vec3,
        restitution: float,
        friction: float
    ) -> wp.vec3:
        """
        Inelastic wall collision with friction.
        
        COLLISION MODEL:
        - Normal component: v_n' = -e * v_n (restitution)
        - Tangent component: v_t' = (1-mu) * v_t (friction)
        
        For flour particles hitting steel walls:
        - e = 0.2-0.4 (significant energy loss)
        - mu = 0.3-0.5 (moderate friction)
        
        After collision, particles lose energy and may:
        - Stick to wall (if v_n' = 0)
        - Slide along wall (friction-dominated)
        - Bounce off (restitution-dominated)
        """
        n_len = wp.length(normal)
        eps = 1.0e-10
        
        if n_len < eps:
            return vel
        
        n = normal / n_len
        
        # Decompose velocity
        v_n = wp.dot(vel, n)
        v_normal = n * v_n
        v_tangent = vel - v_normal
        
        # Only reflect if moving into surface
        if v_n >= 0.0:
            return vel
        
        # Apply restitution to normal component
        v_normal_new = -restitution * v_normal
        
        # Apply friction to tangential component
        v_tangent_new = v_tangent * (1.0 - friction)
        
        return v_normal_new + v_tangent_new

    # -------------------------------------------------------------------------
    # CUT SIZE CALCULATION
    # -------------------------------------------------------------------------
    @wp.func
    def compute_cut_size_zigzag(
        v_air: float,
        rho_p: float,
        rho_f: float,
        mu_f: float,
        g: float
    ) -> float:
        """
        Calculate the cut size (d50) for zigzag classifier.
        
        CUT SIZE: particle diameter where 50% goes to fines, 50% to coarse.
        
        For Stokes regime:
        d50 = sqrt(18 * mu * v_air / (g * (rho_p - rho_f)))
        
        Example: For flour in air at v_air = 2 m/s:
        d50 = sqrt(18 * 1.81e-5 * 2 / (9.81 * (1400 - 1.2)))
            = sqrt(6.5e-4 / 13700)
            = sqrt(4.7e-8)
            = 22 um
        
        Particles < 22 um -> fines (protein-rich)
        Particles > 22 um -> coarse (starch-rich)
        """
        delta_rho = rho_p - rho_f
        eps = 1.0e-10
        
        if delta_rho < eps or v_air < eps:
            return 0.0
        
        d50_squared = 18.0 * mu_f * v_air / (g * delta_rho)
        return wp.sqrt(d50_squared)

    @wp.func
    def compute_cut_size_cyclone(
        inlet_width: float,
        inlet_velocity: float,
        num_spirals: float,
        rho_p: float,
        rho_f: float,
        mu_f: float
    ) -> float:
        """
        Calculate the cut size (d50) for cyclone separator.
        
        Lapple equation:
        d50 = sqrt(9 * mu * W / (2 * pi * N * v_in * (rho_p - rho_f)))
        
        Where:
        - W = inlet width
        - N = number of spiral turns (typically 5-6)
        - v_in = inlet velocity
        
        Example: For a primary cyclone (D=300mm, W=75mm, v_in=15m/s):
        d50 = 5-10 um
        
        Smaller cyclones (D=120mm) have d50 = 2-3 um.
        """
        delta_rho = rho_p - rho_f
        eps = 1.0e-10
        
        if delta_rho < eps or inlet_velocity < eps or num_spirals < eps:
            return 0.0
        
        d50_squared = (9.0 * mu_f * inlet_width) / (2.0 * PI * num_spirals * inlet_velocity * delta_rho)
        return wp.sqrt(d50_squared)

    # -------------------------------------------------------------------------
    # SEPARATION PROBABILITY
    # -------------------------------------------------------------------------
    @wp.func
    def compute_separation_probability(
        diameter: float,
        d50: float,
        sharpness: float
    ) -> float:
        """
        Probability that a particle goes to fines outlet.
        
        Grade efficiency curve (Rosin-Rammler model):
        eta(d) = 1 - exp(-0.693 * (d/d50)^n)
        
        Where n = sharpness parameter (2-4 for zigzag, 3-5 for cyclone)
        
        This gives:
        - d << d50: eta -> 0 (all to fines)
        - d = d50:  eta = 0.5 (50/50 split)
        - d >> d50: eta -> 1 (all to coarse)
        
        In reality, separation is probabilistic, not deterministic.
        Some protein ends up in coarse fraction (especially if agglomerated).
        """
        eps = 1.0e-10
        
        if d50 < eps:
            return 0.5  # No separation
        
        ratio = diameter / d50
        
        # Rosin-Rammler grade efficiency
        probability_to_coarse = 1.0 - wp.exp(-0.693 * wp.pow(ratio, sharpness))
        
        # Return probability to FINES (protein)
        return 1.0 - probability_to_coarse


# =============================================================================
# MAIN CLASSIFICATION PHYSICS KERNEL
# =============================================================================
#
# This kernel simulates particle motion through the classification system:
#
#   VENTURI (zones 0-2)     Particle entrainment into airstream
#        |
#   DUCT_V_Z (zone 10)      Vertical duct to zigzag
#        |
#   ZIGZAG (zones 20-23)    Primary separation by terminal velocity
#        |__________________
#        |                  |
#   FINES (+Y)          COARSE (-Y)
#   zone 22              zone 30 (collected starch)
#        |
#   ELBOW (zone 40)
#        |
#   DUCT (zone 41)          Horizontal to cyclones
#        |
#   CYCLONES (zones 50-52)  Staged centrifugal separation
#   |     |     |
#  DUST  DUST  DUST         (zones 55-57, collected)
#              |
#   ELBOW (zone 60)
#        |
#   DUCT (zone 61)          To bag filter
#        |
#   BAG FILTER (zone 70)    Final fines capture
#   |           |
#  DUST        CLEAN AIR
#  zone 75     zone 80 (should be minimal)
#
# =============================================================================

if wp is not None:

    @wp.kernel
    def classification_physics_kernel(
        # Particle state arrays
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        diameters: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        zones: wp.array(dtype=wp.int32),
        is_active: wp.array(dtype=wp.int32),
        num_particles: int,
        
        # =====================================================================
        # VENTURI GEOMETRY (axis='y', vertical)
        # =====================================================================
        venturi_center: wp.vec3,
        venturi_inlet_diameter: float,
        venturi_throat_diameter: float,
        venturi_outlet_diameter: float,
        venturi_throat_start: float,      # Axial position where throat begins
        venturi_throat_end: float,        # Axial position where throat ends
        venturi_total_length: float,
        venturi_solids_inlet_pos: wp.vec3,
        venturi_solids_inlet_radius: float,
        
        # =====================================================================
        # ZIGZAG GEOMETRY
        # =====================================================================
        zigzag_center: wp.vec3,
        zigzag_channel_width: float,
        zigzag_channel_depth: float,
        zigzag_total_height: float,
        zigzag_num_stages: int,
        zigzag_stage_height: float,
        zigzag_inlet_y: float,            # Bottom of zigzag (air inlet)
        zigzag_fines_outlet_y: float,     # Top (fines exit)
        zigzag_coarse_outlet_y: float,    # Bottom (coarse exit)
        
        # =====================================================================
        # CYCLONE GEOMETRY (primary cyclone - others computed from this)
        # =====================================================================
        cyclone_primary_center: wp.vec3,
        cyclone_primary_radius: float,
        cyclone_primary_cylinder_height: float,
        cyclone_primary_cone_height: float,
        cyclone_primary_vf_radius: float,  # Vortex finder radius
        cyclone_primary_dust_y: float,     # Y position of dust outlet
        
        cyclone_secondary_center: wp.vec3,
        cyclone_secondary_radius: float,
        cyclone_secondary_cylinder_height: float,
        cyclone_secondary_cone_height: float,
        cyclone_secondary_vf_radius: float,
        cyclone_secondary_dust_y: float,
        
        cyclone_tertiary_center: wp.vec3,
        cyclone_tertiary_radius: float,
        cyclone_tertiary_cylinder_height: float,
        cyclone_tertiary_cone_height: float,
        cyclone_tertiary_vf_radius: float,
        cyclone_tertiary_dust_y: float,
        
        # =====================================================================
        # BAG FILTER GEOMETRY
        # =====================================================================
        bagfilter_center: wp.vec3,
        bagfilter_half_width: float,
        bagfilter_half_depth: float,
        bagfilter_height: float,
        bagfilter_inlet_y: float,
        bagfilter_outlet_y: float,
        bagfilter_dust_y: float,
        
        # =====================================================================
        # DUCT/CONNECTION GEOMETRY
        # =====================================================================
        duct_venturi_zigzag_start: wp.vec3,
        duct_venturi_zigzag_end: wp.vec3,
        duct_venturi_zigzag_radius: float,
        
        duct_zigzag_cyclone_start: wp.vec3,
        duct_zigzag_cyclone_end: wp.vec3,
        duct_zigzag_cyclone_radius: float,
        
        duct_cyclone_bag_start: wp.vec3,
        duct_cyclone_bag_end: wp.vec3,
        duct_cyclone_bag_radius: float,
        
        # =====================================================================
        # PHYSICS PARAMETERS
        # =====================================================================
        dt: float,
        gravity: float,
        rho_p: float,               # Particle density
        rho_f: float,               # Air density
        mu_f: float,                # Air viscosity
        restitution: float,
        friction: float,
        
        # Air velocities
        v_air_venturi_inlet: float,  # Inlet air velocity
        v_air_zigzag: float,         # Mean upward velocity in zigzag
        v_air_cyclone_inlet: float,  # Inlet velocity to cyclones
        
        # Turbulence
        turbulent_intensity: float,
        
        # Random seed for turbulent dispersion
        random_seed: int,
    ):
        """
        Main physics kernel for classification system.
        
        PROTEIN SEPARATION PHYSICS:
        ===========================
        
        1. VENTURI ENTRAINMENT
           - Air accelerates through throat (Bernoulli)
           - Low pressure draws in particles from solids inlet
           - Particles accelerate and mix with air
        
        2. ZIGZAG SEPARATION (counter-current)
           - Air flows UP at velocity v_air_zigzag
           - Particles with v_terminal < v_air → rise → FINES (protein)
           - Particles with v_terminal > v_air → fall → COARSE (starch)
           - Cut size d50 = sqrt(18*mu*v_air / (g*(rho_p-rho_f)))
        
        3. CYCLONE SEPARATION (centrifugal)
           - Tangential inlet creates swirling flow
           - Centrifugal force: F_c = m*v_tan²/r → pushes particles OUT
           - Drag force: F_d → pushes particles IN (toward vortex finder)
           - Large particles → wall → dust outlet
           - Small particles → vortex finder → next stage
        
        4. BAG FILTER (inertial impaction)
           - Remaining fines captured on filter bags
           - Clean air exits through top
        """
        tid = wp.tid()
        
        if tid >= num_particles:
            return
        
        if is_active[tid] == 0:
            return
        
        pos = positions[tid]
        vel = velocities[tid]
        zone = zones[tid]
        d = diameters[tid]
        m = masses[tid]
        particle_radius = d * 0.5
        
        # =====================================================================
        # COMPUTE BASE ACCELERATIONS
        # =====================================================================
        # Gravity with buoyancy
        a_gravity = compute_gravity_buoyancy(rho_p, rho_f, gravity)
        
        # Air velocity depends on zone (computed below)
        v_air = wp.vec3(0.0, 0.0, 0.0)
        
        # =====================================================================
        # ZONE 0: VENTURI INLET (entering via solids inlet)
        # =====================================================================
        if zone == 0:
            # Particle is entering through solids inlet tube
            # Move toward throat region
            
            # Distance from solids inlet axis
            to_inlet = pos - venturi_solids_inlet_pos
            
            # Simple model: particle moves toward venturi axis
            # Air velocity at throat draws particle in
            local_y = pos[1] - venturi_center[1]
            
            # Compute air velocity using continuity
            v_air = compute_venturi_air_velocity(
                pos, venturi_center,
                venturi_inlet_diameter, venturi_throat_diameter, venturi_outlet_diameter,
                venturi_throat_start, venturi_throat_end, venturi_total_length,
                v_air_venturi_inlet, 1  # axis=Y
            )
            
            # Transition to throat when reaching throat region
            if local_y >= venturi_throat_start and local_y <= venturi_throat_end:
                zone = 1
        
        # =====================================================================
        # ZONE 1: VENTURI THROAT (high velocity, entrainment)
        # =====================================================================
        elif zone == 1:
            local_y = pos[1] - venturi_center[1]
            
            # Air velocity in throat (maximum)
            v_air = compute_venturi_air_velocity(
                pos, venturi_center,
                venturi_inlet_diameter, venturi_throat_diameter, venturi_outlet_diameter,
                venturi_throat_start, venturi_throat_end, venturi_total_length,
                v_air_venturi_inlet, 1
            )
            
            # Radial containment (cylindrical throat)
            dx = pos[0] - venturi_center[0]
            dz = pos[2] - venturi_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            throat_r = venturi_throat_diameter / 2.0
            
            if r + particle_radius > throat_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - throat_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to divergent section
            if local_y > venturi_throat_end:
                zone = 2
        
        # =====================================================================
        # ZONE 2: VENTURI DIVERGENT (pressure recovery)
        # =====================================================================
        elif zone == 2:
            local_y = pos[1] - venturi_center[1]
            
            # Air velocity (decelerating as diameter increases)
            v_air = compute_venturi_air_velocity(
                pos, venturi_center,
                venturi_inlet_diameter, venturi_throat_diameter, venturi_outlet_diameter,
                venturi_throat_start, venturi_throat_end, venturi_total_length,
                v_air_venturi_inlet, 1
            )
            
            # Radial containment (expanding cone)
            dx = pos[0] - venturi_center[0]
            dz = pos[2] - venturi_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            
            # Radius increases linearly from throat to outlet
            t = (local_y - venturi_throat_end) / (venturi_total_length - venturi_throat_end + 0.001)
            t = wp.clamp(t, 0.0, 1.0)
            local_radius = venturi_throat_diameter / 2.0 + t * (venturi_outlet_diameter - venturi_throat_diameter) / 2.0
            
            if r + particle_radius > local_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - local_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to duct when exiting venturi
            if local_y > venturi_total_length - particle_radius:
                zone = 10  # Enter duct to zigzag
        
        # =====================================================================
        # ZONE 10: DUCT - VENTURI TO ZIGZAG
        # =====================================================================
        elif zone == 10:
            # Cylindrical duct, flow direction is +Y (upward)
            
            # Progress along duct
            duct_length = wp.length(duct_venturi_zigzag_end - duct_venturi_zigzag_start)
            progress = (pos[1] - duct_venturi_zigzag_start[1]) / (duct_venturi_zigzag_end[1] - duct_venturi_zigzag_start[1] + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            # Center at this height
            center_x = duct_venturi_zigzag_start[0] + progress * (duct_venturi_zigzag_end[0] - duct_venturi_zigzag_start[0])
            center_z = duct_venturi_zigzag_start[2] + progress * (duct_venturi_zigzag_end[2] - duct_venturi_zigzag_start[2])
            
            dx = pos[0] - center_x
            dz = pos[2] - center_z
            r = wp.sqrt(dx * dx + dz * dz)
            
            # Air velocity (constant through duct)
            v_air = wp.vec3(0.0, v_air_zigzag, 0.0)
            
            # Radial containment
            if r + particle_radius > duct_venturi_zigzag_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - duct_venturi_zigzag_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to zigzag
            if pos[1] >= zigzag_inlet_y - particle_radius * 2.0:
                zone = 20  # Enter zigzag
        
        # =====================================================================
        # ZONE 20-21: ZIGZAG CLASSIFIER (PRIMARY SEPARATION)
        # Note: Zone 22 (fines) and 23 (coarse) are handled separately for transitions
        # =====================================================================
        elif zone == 20 or zone == 21:
            # ZIGZAG SEPARATION PHYSICS:
            # - Air flows upward at v_air_zigzag
            # - Gravity pulls particles down
            # - Terminal velocity determines fate:
            #   v_t < v_air → particle rises (fines/protein)
            #   v_t > v_air → particle falls (coarse/starch)
            
            local_y = pos[1] - zigzag_center[1]
            local_x = pos[0] - zigzag_center[0]
            local_z = pos[2] - zigzag_center[2]
            
            # Compute air velocity with stage-dependent variation
            v_air = compute_zigzag_air_velocity(
                pos, zigzag_center,
                zigzag_channel_width, zigzag_total_height, zigzag_num_stages,
                v_air_zigzag, zigzag_stage_height
            )
            
            # Add turbulent dispersion (essential for realistic separation)
            v_turb = compute_turbulent_dispersion(vel, turbulent_intensity, random_seed, tid)
            v_air = v_air + v_turb
            
            # Channel wall containment (rectangular cross-section)
            half_w = zigzag_channel_width / 2.0
            half_d = zigzag_channel_depth / 2.0
            
            # X walls
            if local_x + particle_radius > half_w:
                pos = wp.vec3(zigzag_center[0] + half_w - particle_radius - 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(-1.0, 0.0, 0.0), restitution, friction)
            elif local_x - particle_radius < -half_w:
                pos = wp.vec3(zigzag_center[0] - half_w + particle_radius + 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(1.0, 0.0, 0.0), restitution, friction)
            
            # Z walls
            if local_z + particle_radius > half_d:
                pos = wp.vec3(pos[0], pos[1], zigzag_center[2] + half_d - particle_radius - 0.001)
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, -1.0), restitution, friction)
            elif local_z - particle_radius < -half_d:
                pos = wp.vec3(pos[0], pos[1], zigzag_center[2] - half_d + particle_radius + 0.001)
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, 1.0), restitution, friction)
            
            # SEPARATION LOGIC based on particle position
            # Rising particles (fines) move toward top
            if pos[1] >= zigzag_fines_outlet_y - particle_radius * 2.0:
                zone = 22  # Fines path (toward cyclones)
            # Falling particles (coarse) move toward bottom
            elif pos[1] <= zigzag_coarse_outlet_y + particle_radius * 2.0:
                zone = 30  # Coarse outlet (collected starch)
            else:
                zone = 21  # Still in stages
        
        # =====================================================================
        # ZONE 30: COARSE OUTLET (Starch collection)
        # =====================================================================
        elif zone == 30:
            # Particle has fallen to coarse outlet - it's collected starch
            # Keep zone = 30 for statistics, just deactivate
            is_active[tid] = 0
        
        # =====================================================================
        # ZONE 22/40: FINES PATH - ZIGZAG TO CYCLONE
        # =====================================================================
        elif zone == 22 or zone == 40:
            # Transition from zigzag fines outlet to cyclone duct
            zone = 41  # Enter horizontal duct to cyclones
        
        # =====================================================================
        # ZONE 41: DUCT - ZIGZAG TO CYCLONE (horizontal)
        # =====================================================================
        elif zone == 41:
            # Horizontal duct toward primary cyclone inlet
            
            # Progress along duct
            dx_duct = duct_zigzag_cyclone_end[0] - duct_zigzag_cyclone_start[0]
            progress = (pos[0] - duct_zigzag_cyclone_start[0]) / (dx_duct + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            # Center at this position
            center_y = duct_zigzag_cyclone_start[1] + progress * (duct_zigzag_cyclone_end[1] - duct_zigzag_cyclone_start[1])
            center_z = duct_zigzag_cyclone_start[2] + progress * (duct_zigzag_cyclone_end[2] - duct_zigzag_cyclone_start[2])
            
            dy = pos[1] - center_y
            dz = pos[2] - center_z
            r = wp.sqrt(dy * dy + dz * dz)
            
            # Air velocity (horizontal toward cyclone)
            v_air = wp.vec3(v_air_cyclone_inlet, 0.0, 0.0)
            
            # Radial containment
            if r + particle_radius > duct_zigzag_cyclone_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(0.0, -dy / r, -dz / r)
                    push = r + particle_radius - duct_zigzag_cyclone_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to primary cyclone
            if pos[0] >= cyclone_primary_center[0] - cyclone_primary_radius - particle_radius * 2.0:
                zone = 50  # Enter primary cyclone
        
        # =====================================================================
        # ZONE 50: PRIMARY CYCLONE (coarse fines)
        # =====================================================================
        elif zone == 50:
            # CYCLONE SEPARATION PHYSICS:
            # - Tangential inlet creates swirling flow
            # - Outer vortex spirals DOWN (dirty air with particles)
            # - Inner vortex spirals UP (clean air)
            # - Centrifugal force pushes large particles to wall
            # - Small particles follow air to vortex finder
            
            dx = pos[0] - cyclone_primary_center[0]
            dz = pos[2] - cyclone_primary_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - cyclone_primary_center[1]
            
            # Compute cyclone velocity field
            v_tan = compute_cyclone_tangential_velocity(
                pos, cyclone_primary_center, v_air_cyclone_inlet, cyclone_primary_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_primary_center, cyclone_primary_radius, cyclone_primary_vf_radius,
                cyclone_primary_cylinder_height, cyclone_primary_cone_height, v_air_cyclone_inlet
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_primary_center, cyclone_primary_radius, cyclone_primary_vf_radius,
                v_air_cyclone_inlet
            )
            
            v_air = v_tan + v_rad + v_axial
            
            # Centrifugal acceleration (pushes particles outward)
            a_centrifugal = compute_centrifugal_acceleration(
                pos, vel, cyclone_primary_center, wp.vec3(0.0, 1.0, 0.0)
            )
            
            # Wall containment (cylinder + cone)
            total_height = cyclone_primary_cylinder_height + cyclone_primary_cone_height
            
            # Local radius (cylinder or cone)
            if local_y >= -cyclone_primary_cylinder_height:
                # In cylinder section
                wall_r = cyclone_primary_radius
            else:
                # In cone section - radius decreases linearly
                cone_progress = (-local_y - cyclone_primary_cylinder_height) / cyclone_primary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                # Cone tip is smaller (assume 0.3 of cylinder radius)
                wall_r = cyclone_primary_radius * (1.0 - 0.7 * cone_progress)
            
            # Radial containment
            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # SEPARATION DECISION:
            # - Particle at wall AND below cylinder → dust outlet
            # - Particle in core AND above certain height → vortex finder → next stage
            
            at_wall = r > wall_r * 0.8
            in_core = r < cyclone_primary_vf_radius * 1.5
            below_cylinder = local_y < -cyclone_primary_cylinder_height
            above_vf = local_y > 0.0
            
            if at_wall and below_cylinder and local_y < cyclone_primary_dust_y + particle_radius * 3.0:
                zone = 55  # Collected in primary dust outlet
            elif in_core and above_vf:
                zone = 51  # Move to secondary cyclone
        
        # =====================================================================
        # ZONE 51: SECONDARY CYCLONE (medium fines)
        # =====================================================================
        elif zone == 51:
            dx = pos[0] - cyclone_secondary_center[0]
            dz = pos[2] - cyclone_secondary_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - cyclone_secondary_center[1]
            
            # Cyclone velocity field
            v_tan = compute_cyclone_tangential_velocity(
                pos, cyclone_secondary_center, v_air_cyclone_inlet * 0.8, cyclone_secondary_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_secondary_center, cyclone_secondary_radius, cyclone_secondary_vf_radius,
                cyclone_secondary_cylinder_height, cyclone_secondary_cone_height, v_air_cyclone_inlet * 0.8
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_secondary_center, cyclone_secondary_radius, cyclone_secondary_vf_radius,
                v_air_cyclone_inlet * 0.8
            )
            
            v_air = v_tan + v_rad + v_axial
            
            a_centrifugal = compute_centrifugal_acceleration(
                pos, vel, cyclone_secondary_center, wp.vec3(0.0, 1.0, 0.0)
            )
            
            # Wall containment
            if local_y >= -cyclone_secondary_cylinder_height:
                wall_r = cyclone_secondary_radius
            else:
                cone_progress = (-local_y - cyclone_secondary_cylinder_height) / cyclone_secondary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                wall_r = cyclone_secondary_radius * (1.0 - 0.7 * cone_progress)
            
            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Separation decision
            at_wall = r > wall_r * 0.8
            in_core = r < cyclone_secondary_vf_radius * 1.5
            below_cylinder = local_y < -cyclone_secondary_cylinder_height
            above_vf = local_y > 0.0
            
            if at_wall and below_cylinder and local_y < cyclone_secondary_dust_y + particle_radius * 3.0:
                zone = 56  # Collected in secondary dust outlet
            elif in_core and above_vf:
                zone = 52  # Move to tertiary cyclone
        
        # =====================================================================
        # ZONE 52: TERTIARY CYCLONE (fine protein)
        # =====================================================================
        elif zone == 52:
            dx = pos[0] - cyclone_tertiary_center[0]
            dz = pos[2] - cyclone_tertiary_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - cyclone_tertiary_center[1]
            
            # Cyclone velocity field (smallest cyclone, highest velocity)
            v_tan = compute_cyclone_tangential_velocity(
                pos, cyclone_tertiary_center, v_air_cyclone_inlet * 0.6, cyclone_tertiary_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_tertiary_center, cyclone_tertiary_radius, cyclone_tertiary_vf_radius,
                cyclone_tertiary_cylinder_height, cyclone_tertiary_cone_height, v_air_cyclone_inlet * 0.6
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_tertiary_center, cyclone_tertiary_radius, cyclone_tertiary_vf_radius,
                v_air_cyclone_inlet * 0.6
            )
            
            v_air = v_tan + v_rad + v_axial
            
            a_centrifugal = compute_centrifugal_acceleration(
                pos, vel, cyclone_tertiary_center, wp.vec3(0.0, 1.0, 0.0)
            )
            
            # Wall containment
            if local_y >= -cyclone_tertiary_cylinder_height:
                wall_r = cyclone_tertiary_radius
            else:
                cone_progress = (-local_y - cyclone_tertiary_cylinder_height) / cyclone_tertiary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                wall_r = cyclone_tertiary_radius * (1.0 - 0.7 * cone_progress)
            
            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Separation decision
            at_wall = r > wall_r * 0.8
            in_core = r < cyclone_tertiary_vf_radius * 1.5
            below_cylinder = local_y < -cyclone_tertiary_cylinder_height
            above_vf = local_y > 0.0
            
            if at_wall and below_cylinder and local_y < cyclone_tertiary_dust_y + particle_radius * 3.0:
                zone = 57  # Collected in tertiary dust outlet (fine protein)
            elif in_core and above_vf:
                zone = 60  # Move to bag filter path
        
        # =====================================================================
        # ZONES 55-57: CYCLONE DUST OUTLETS (collected)
        # =====================================================================
        elif zone == 55 or zone == 56 or zone == 57:
            # Particle collected in cyclone dust outlet
            # Keep zone for statistics, just deactivate
            is_active[tid] = 0
        
        # =====================================================================
        # ZONE 60-61: CYCLONE TO BAG FILTER PATH
        # =====================================================================
        elif zone == 60 or zone == 61:
            # Duct from tertiary cyclone overflow to bag filter
            
            # Progress along duct
            dx_duct = duct_cyclone_bag_end[0] - duct_cyclone_bag_start[0]
            progress = (pos[0] - duct_cyclone_bag_start[0]) / (dx_duct + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            center_y = duct_cyclone_bag_start[1] + progress * (duct_cyclone_bag_end[1] - duct_cyclone_bag_start[1])
            center_z = duct_cyclone_bag_start[2] + progress * (duct_cyclone_bag_end[2] - duct_cyclone_bag_start[2])
            
            dy = pos[1] - center_y
            dz = pos[2] - center_z
            r = wp.sqrt(dy * dy + dz * dz)
            
            # Air velocity
            v_air = wp.vec3(v_air_cyclone_inlet * 0.5, 0.0, 0.0)
            
            # Radial containment
            if r + particle_radius > duct_cyclone_bag_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(0.0, -dy / r, -dz / r)
                    push = r + particle_radius - duct_cyclone_bag_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to bag filter
            if pos[0] >= bagfilter_center[0] - bagfilter_half_width - particle_radius * 2.0:
                zone = 70  # Enter bag filter
        
        # =====================================================================
        # ZONE 70: BAG FILTER (final fines capture)
        # =====================================================================
        elif zone == 70:
            # BAG FILTER PHYSICS:
            # - Dirty air enters from side
            # - Bags capture particles via inertial impaction, interception, diffusion
            # - Clean air exits through top
            # - Collected dust falls to hopper
            
            local_x = pos[0] - bagfilter_center[0]
            local_y = pos[1] - bagfilter_center[1]
            local_z = pos[2] - bagfilter_center[2]
            
            # Simplified: air flows through filter, particles captured
            # Small velocity upward
            v_air = wp.vec3(0.0, 0.5, 0.0)
            
            # Box containment
            if local_x + particle_radius > bagfilter_half_width:
                pos = wp.vec3(bagfilter_center[0] + bagfilter_half_width - particle_radius - 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(-1.0, 0.0, 0.0), restitution, friction)
            elif local_x - particle_radius < -bagfilter_half_width:
                pos = wp.vec3(bagfilter_center[0] - bagfilter_half_width + particle_radius + 0.001, pos[1], pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(1.0, 0.0, 0.0), restitution, friction)
            
            if local_z + particle_radius > bagfilter_half_depth:
                pos = wp.vec3(pos[0], pos[1], bagfilter_center[2] + bagfilter_half_depth - particle_radius - 0.001)
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, -1.0), restitution, friction)
            elif local_z - particle_radius < -bagfilter_half_depth:
                pos = wp.vec3(pos[0], pos[1], bagfilter_center[2] - bagfilter_half_depth + particle_radius + 0.001)
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, 1.0), restitution, friction)
            
            # Particles settle to hopper (collected)
            if pos[1] <= bagfilter_dust_y + particle_radius * 2.0:
                zone = 75  # Collected in bag filter hopper
            # Very small particles might escape with clean air (rare)
            elif pos[1] >= bagfilter_outlet_y - particle_radius * 2.0:
                # Check if particle is small enough to escape
                # In practice, bag filter captures > 99.9% of particles
                zone = 80  # Escaped with clean air (should be rare)
        
        # =====================================================================
        # ZONES 75, 80, 99: COLLECTION / EXIT
        # =====================================================================
        elif zone == 75:
            # Collected in bag filter - keep zone for statistics
            is_active[tid] = 0
        
        elif zone == 80:
            # Escaped with clean air - keep zone for statistics
            is_active[tid] = 0
        
        elif zone == 99:
            # Legacy exit zone - deactivate
            is_active[tid] = 0
        
        # =====================================================================
        # COMPUTE DRAG AND INTEGRATE
        # =====================================================================
        if is_active[tid] == 1:
            # Drag acceleration (uses zone-specific v_air computed above)
            a_drag = compute_drag_acceleration(vel, v_air, d, m, rho_f, mu_f)
            
            # Add centrifugal acceleration if in cyclone
            accel = a_gravity + a_drag
            if zone == 50 or zone == 51 or zone == 52:
                accel = accel + a_centrifugal
            
            # Semi-implicit Euler integration
            vel = vel + accel * dt
            
            # Velocity damping for stability
            v_mag = wp.length(vel)
            max_vel = 50.0  # Limit to 50 m/s
            if v_mag > max_vel:
                vel = vel * (max_vel / v_mag)
            
            pos = pos + vel * dt
        
        # =====================================================================
        # WRITE BACK
        # =====================================================================
        positions[tid] = pos
        velocities[tid] = vel
        zones[tid] = zone


    # =========================================================================
    # PARTICLE INITIALIZATION KERNEL
    # =========================================================================
    @wp.kernel
    def init_classification_particles(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        diameters: wp.array(dtype=float),
        masses: wp.array(dtype=float),
        zones: wp.array(dtype=wp.int32),
        is_active: wp.array(dtype=wp.int32),
        num_particles: int,
        # Solids inlet position (from venturi eductor)
        inlet_center: wp.vec3,
        inlet_radius: float,
        # Initial velocity (from feed system)
        initial_velocity: wp.vec3,
        # Particle properties
        mean_diameter: float,
        diameter_std: float,
        density: float,
        # Random seed
        random_seed: int,
    ):
        """
        Initialize particles at the venturi solids inlet.
        
        Particles arrive from the feed system with a distribution of sizes
        representing the flour mixture (protein + starch).
        
        Size distribution:
        - Protein particles: ~10-30 microns (smaller)
        - Starch particles: ~15-60 microns (larger)
        - Using log-normal distribution to capture this
        """
        tid = wp.tid()
        
        if tid >= num_particles:
            return
        
        # Random position within inlet circle (cylindrical distribution)
        state = wp.rand_init(random_seed, tid)
        r_rand = wp.sqrt(wp.randf(state)) * inlet_radius * 0.8
        theta = wp.randf(state) * 2.0 * 3.14159265359
        
        x = inlet_center[0] + r_rand * wp.cos(theta)
        y = inlet_center[1]
        z = inlet_center[2] + r_rand * wp.sin(theta)
        
        positions[tid] = wp.vec3(x, y, z)
        
        # Initial velocity (inherits from feed system)
        # Add small random perturbation
        vx = initial_velocity[0] + (wp.randf(state) - 0.5) * 0.1
        vy = initial_velocity[1] + (wp.randf(state) - 0.5) * 0.1
        vz = initial_velocity[2] + (wp.randf(state) - 0.5) * 0.1
        velocities[tid] = wp.vec3(vx, vy, vz)
        
        # Particle diameter (log-normal distribution for realistic flour)
        # ln(d) ~ N(ln(mean), cv) where cv = std/mean
        ln_mean = wp.log(mean_diameter)
        ln_std = diameter_std / mean_diameter  # coefficient of variation
        ln_d = ln_mean + ln_std * (wp.randf(state) - 0.5) * 2.0  # Simplified normal
        d = wp.exp(ln_d)
        
        # Clamp to physical bounds
        d = wp.clamp(d, 5.0e-6, 100.0e-6)  # 5-100 microns
        diameters[tid] = d
        
        # Mass from density and volume
        vol = 3.14159265359 / 6.0 * d * d * d
        masses[tid] = density * vol
        
        # Start in zone 0 (venturi inlet)
        zones[tid] = 0
        is_active[tid] = 1


    # =========================================================================
    # SEPARATION STATISTICS KERNEL
    # =========================================================================
    @wp.kernel
    def count_separation_results(
        zones: wp.array(dtype=wp.int32),
        is_active: wp.array(dtype=wp.int32),
        num_particles: int,
        # Output counts (using atomic adds)
        count_coarse: wp.array(dtype=wp.int32),       # Zone 30: Coarse starch from zigzag
        count_cyclone_1: wp.array(dtype=wp.int32),    # Zone 55: Primary cyclone
        count_cyclone_2: wp.array(dtype=wp.int32),    # Zone 56: Secondary cyclone
        count_cyclone_3: wp.array(dtype=wp.int32),    # Zone 57: Tertiary cyclone (fine protein)
        count_bagfilter: wp.array(dtype=wp.int32),    # Zone 75: Bag filter
        count_escaped: wp.array(dtype=wp.int32),      # Zone 80: Escaped with clean air
        count_active: wp.array(dtype=wp.int32),       # Still in system
    ):
        """
        Count particles by final destination for separation analysis.
        
        PROTEIN SEPARATION QUALITY METRICS:
        ===================================
        
        Ideal outcome for protein separation:
        - High protein in cyclone_3 (zone 57) and bagfilter (zone 75)
        - High starch in coarse (zone 30) and cyclone_1 (zone 55)
        - Minimal escaped (zone 80)
        
        Grade efficiency: What fraction of each size class ends up where?
        Separation efficiency: How pure is each collected fraction?
        """
        tid = wp.tid()
        
        if tid >= num_particles:
            return
        
        zone = zones[tid]
        active = is_active[tid]
        
        # Count by zone (works for both active and inactive particles)
        if zone == 30:
            wp.atomic_add(count_coarse, 0, 1)
        elif zone == 55:
            wp.atomic_add(count_cyclone_1, 0, 1)
        elif zone == 56:
            wp.atomic_add(count_cyclone_2, 0, 1)
        elif zone == 57:
            wp.atomic_add(count_cyclone_3, 0, 1)
        elif zone == 75:
            wp.atomic_add(count_bagfilter, 0, 1)
        elif zone == 80:
            wp.atomic_add(count_escaped, 0, 1)
        elif active == 1:
            wp.atomic_add(count_active, 0, 1)


    # =========================================================================
    # POST-INTEGRATION CONTAINMENT KERNEL
    # =========================================================================
    @wp.kernel
    def post_integration_containment(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        diameters: wp.array(dtype=float),
        zones: wp.array(dtype=wp.int32),
        is_active: wp.array(dtype=wp.int32),
        num_particles: int,
        # System bounds (bounding box for entire classification system)
        system_min: wp.vec3,
        system_max: wp.vec3,
        # Restitution for wall bounces
        restitution: float,
    ):
        """
        Ensure particles stay within system bounds after integration.
        
        This is a safety net to prevent particles from escaping due to
        numerical instabilities or large time steps.
        """
        tid = wp.tid()
        
        if tid >= num_particles:
            return
        
        if is_active[tid] == 0:
            return
        
        pos = positions[tid]
        vel = velocities[tid]
        particle_radius = diameters[tid] * 0.5
        
        # Clamp to system bounds
        # X bounds
        if pos[0] < system_min[0] + particle_radius:
            pos = wp.vec3(system_min[0] + particle_radius + 0.001, pos[1], pos[2])
            if vel[0] < 0.0:
                vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
        elif pos[0] > system_max[0] - particle_radius:
            pos = wp.vec3(system_max[0] - particle_radius - 0.001, pos[1], pos[2])
            if vel[0] > 0.0:
                vel = wp.vec3(-vel[0] * restitution, vel[1], vel[2])
        
        # Y bounds
        if pos[1] < system_min[1] + particle_radius:
            pos = wp.vec3(pos[0], system_min[1] + particle_radius + 0.001, pos[2])
            if vel[1] < 0.0:
                vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
        elif pos[1] > system_max[1] - particle_radius:
            pos = wp.vec3(pos[0], system_max[1] - particle_radius - 0.001, pos[2])
            if vel[1] > 0.0:
                vel = wp.vec3(vel[0], -vel[1] * restitution, vel[2])
        
        # Z bounds
        if pos[2] < system_min[2] + particle_radius:
            pos = wp.vec3(pos[0], pos[1], system_min[2] + particle_radius + 0.001)
            if vel[2] < 0.0:
                vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
        elif pos[2] > system_max[2] - particle_radius:
            pos = wp.vec3(pos[0], pos[1], system_max[2] - particle_radius - 0.001)
            if vel[2] > 0.0:
                vel = wp.vec3(vel[0], vel[1], -vel[2] * restitution)
        
        positions[tid] = pos
        velocities[tid] = vel


    # =========================================================================
    # GRADE EFFICIENCY KERNEL
    # =========================================================================
    @wp.kernel
    def compute_grade_efficiency_kernel(
        diameters: wp.array(dtype=float),
        zones: wp.array(dtype=wp.int32),
        is_active: wp.array(dtype=wp.int32),
        num_particles: int,
        # Bin boundaries for particle sizes
        bin_edges: wp.array(dtype=float),
        num_bins: int,
        # Output: counts per bin per destination
        # Shape: (num_bins, 6) where 6 = coarse, cy1, cy2, cy3, bag, escaped
        coarse_counts: wp.array(dtype=wp.int32),
        cyclone1_counts: wp.array(dtype=wp.int32),
        cyclone2_counts: wp.array(dtype=wp.int32),
        cyclone3_counts: wp.array(dtype=wp.int32),
        bag_counts: wp.array(dtype=wp.int32),
        escaped_counts: wp.array(dtype=wp.int32),
    ):
        """
        Compute grade efficiency by tracking particle fates by size class.
        
        GRADE EFFICIENCY CURVE:
        =======================
        
        G(d) = fraction of particles of diameter d that report to fines
        
        For protein separation:
        - Want G(d_protein) → high (protein goes to fines)
        - Want G(d_starch) → low (starch goes to coarse)
        
        This kernel bins particles by diameter and tracks where they went.
        """
        tid = wp.tid()
        
        if tid >= num_particles:
            return
        
        if is_active[tid] == 1:
            return  # Still active, not yet collected
        
        d = diameters[tid]
        zone = zones[tid]
        
        # Find size bin using dynamic variable (required by Warp for mutation in loops)
        # We use a 'found' flag to avoid overwriting once we find the right bin
        bin_idx = int(0)
        found = int(0)
        for i in range(num_bins):
            if found == 0:
                if d >= bin_edges[i] and d < bin_edges[i + 1]:
                    bin_idx = i
                    found = 1
        
        # Increment appropriate counter
        if zone == 30:
            wp.atomic_add(coarse_counts, bin_idx, 1)
        elif zone == 55:
            wp.atomic_add(cyclone1_counts, bin_idx, 1)
        elif zone == 56:
            wp.atomic_add(cyclone2_counts, bin_idx, 1)
        elif zone == 57:
            wp.atomic_add(cyclone3_counts, bin_idx, 1)
        elif zone == 75:
            wp.atomic_add(bag_counts, bin_idx, 1)
        elif zone == 80:
            wp.atomic_add(escaped_counts, bin_idx, 1)


# =============================================================================
# SIMULATOR CLASS
# =============================================================================

class ClassificationFlowPhysicsSimulator:
    """
    Physics-based particle separation simulator for the classification system.
    
    Simulates the complete protein/starch separation process:
    
    1. VENTURI ENTRAINMENT
       - Particles enter through solids inlet
       - Air accelerates through throat, entraining particles
       
    2. ZIGZAG CLASSIFICATION
       - Counter-current air flow (up) vs. gravity (down)
       - Light particles (protein) rise → fines outlet
       - Heavy particles (starch) fall → coarse outlet
       
    3. CYCLONE SEPARATION (staged)
       - Primary: removes coarsest fines
       - Secondary: removes medium fines  
       - Tertiary: collects finest protein
       
    4. BAG FILTER
       - Final capture of remaining fines
       
    SEPARATION PHYSICS:
    ===================
    - Cut size (d50) = particle size with 50% probability to each outlet
    - Terminal velocity determines zigzag separation
    - Centrifugal vs. drag determines cyclone separation
    - Sharpness factor determines how clean the separation is
    """
    
    def __init__(
        self,
        assembly: 'ClassificationSystemAssembly',
        config: ClassificationFlowConfig = None,
    ):
        """
        Initialize the classification flow simulator.
        
        Args:
            assembly: ClassificationSystemAssembly with geometry
            config: Simulation configuration
        """
        self.assembly = assembly
        self.config = config or ClassificationFlowConfig()
        self.state = ClassificationFlowState()
        
        # Initialize Warp
        wp.init()
        self.device = self.config.device
        
        # Extract geometry
        self.geometry = extract_geometry(assembly)
        self._compute_derived_parameters()
        
        # Allocate arrays
        self._allocate_arrays()
        
        # Hash grid for particle collisions
        self._setup_hash_grid()
        
        # Separation statistics arrays
        self._setup_statistics_arrays()
        
        print(f"\n  ClassificationFlowPhysicsSimulator initialized")
        print(f"    Device: {self.device}")
        print(f"    Max particles: {self.config.num_particles}")
    
    def _compute_derived_parameters(self):
        """Compute physics parameters from geometry and config."""
        cfg = self.config
        geo = self.geometry
        
        # Helper to safely get geometry attributes with defaults
        def get_geo_attr(component_name: str, attr: str, default):
            """Get attribute from geometry dict with fallback to default."""
            comp = geo.get(component_name)
            if comp is None:
                return default
            val = getattr(comp, attr, None)
            if val is None:
                return default
            return val
        
        def get_geo_center(component_name: str, default: list):
            """Get center from geometry component."""
            comp = geo.get(component_name)
            if comp is None or comp.center is None:
                return np.array(default)
            return np.array(comp.center)
        
        # =====================================================================
        # VENTURI GEOMETRY
        # =====================================================================
        self.venturi_center = get_geo_center('venturi', [0, 0, 0])
        self.venturi_inlet_diameter = get_geo_attr('venturi', 'inlet_diameter', 0.1)
        self.venturi_throat_diameter = get_geo_attr('venturi', 'throat_diameter', 0.05)
        self.venturi_outlet_diameter = get_geo_attr('venturi', 'outlet_diameter', 0.08)
        self.venturi_total_length = get_geo_attr('venturi', 'length', 0.3)
        
        # Venturi throat region (estimated if not provided)
        self.venturi_throat_start = get_geo_attr('venturi', 'throat_start', self.venturi_total_length * 0.3)
        self.venturi_throat_end = get_geo_attr('venturi', 'throat_end', self.venturi_total_length * 0.5)
        
        # Solids inlet position
        solids_inlet = get_geo_attr('venturi', 'solids_inlet_pos', None)
        if solids_inlet is None:
            solids_inlet = self.venturi_center.copy()
        self.venturi_solids_inlet_pos = np.array(solids_inlet)
        self.venturi_solids_inlet_radius = get_geo_attr('venturi', 'solids_inlet_diameter', 0.05) / 2.0
        
        # =====================================================================
        # ZIGZAG GEOMETRY
        # =====================================================================
        self.zigzag_center = get_geo_center('zigzag', [0, 0.5, 0])
        self.zigzag_channel_width = get_geo_attr('zigzag', 'channel_width', 0.15)
        self.zigzag_channel_depth = get_geo_attr('zigzag', 'channel_depth', 0.15)
        self.zigzag_total_height = get_geo_attr('zigzag', 'total_height', 1.5)
        if self.zigzag_total_height == 0:
            self.zigzag_total_height = get_geo_attr('zigzag', 'length', 1.5)
        self.zigzag_num_stages = get_geo_attr('zigzag', 'num_stages', 12)
        self.zigzag_stage_height = self.zigzag_total_height / max(1, self.zigzag_num_stages)
        
        # Zigzag inlet/outlet positions
        self.zigzag_inlet_y = self.zigzag_center[1] - self.zigzag_total_height / 2
        self.zigzag_fines_outlet_y = self.zigzag_center[1] + self.zigzag_total_height / 2
        self.zigzag_coarse_outlet_y = self.zigzag_inlet_y - 0.05  # Below inlet
        
        # =====================================================================
        # PRIMARY CYCLONE
        # =====================================================================
        self.cyclone_primary_center = get_geo_center('cyclone_primary', [0.5, 0.5, 0])
        self.cyclone_primary_radius = get_geo_attr('cyclone_primary', 'radius', 0.15)
        if self.cyclone_primary_radius == 0:
            self.cyclone_primary_radius = get_geo_attr('cyclone_primary', 'cylinder_diameter', 0.30) / 2.0
        self.cyclone_primary_cylinder_height = get_geo_attr('cyclone_primary', 'cylinder_height', 0.3)
        self.cyclone_primary_cone_height = get_geo_attr('cyclone_primary', 'cone_height', 0.4)
        self.cyclone_primary_vf_radius = get_geo_attr('cyclone_primary', 'vortex_finder_diameter', self.cyclone_primary_radius * 0.8) / 2.0
        self.cyclone_primary_dust_y = self.cyclone_primary_center[1] - self.cyclone_primary_cylinder_height - self.cyclone_primary_cone_height
        
        # =====================================================================
        # SECONDARY CYCLONE (smaller)
        # =====================================================================
        self.cyclone_secondary_center = get_geo_center('cyclone_secondary', [0.8, 0.5, 0])
        self.cyclone_secondary_radius = get_geo_attr('cyclone_secondary', 'radius', 0.12)
        if self.cyclone_secondary_radius == 0:
            self.cyclone_secondary_radius = get_geo_attr('cyclone_secondary', 'cylinder_diameter', 0.24) / 2.0
        self.cyclone_secondary_cylinder_height = get_geo_attr('cyclone_secondary', 'cylinder_height', 0.25)
        self.cyclone_secondary_cone_height = get_geo_attr('cyclone_secondary', 'cone_height', 0.35)
        self.cyclone_secondary_vf_radius = get_geo_attr('cyclone_secondary', 'vortex_finder_diameter', self.cyclone_secondary_radius * 0.8) / 2.0
        self.cyclone_secondary_dust_y = self.cyclone_secondary_center[1] - self.cyclone_secondary_cylinder_height - self.cyclone_secondary_cone_height
        
        # =====================================================================
        # TERTIARY CYCLONE (smallest)
        # =====================================================================
        self.cyclone_tertiary_center = get_geo_center('cyclone_tertiary', [1.1, 0.5, 0])
        self.cyclone_tertiary_radius = get_geo_attr('cyclone_tertiary', 'radius', 0.10)
        if self.cyclone_tertiary_radius == 0:
            self.cyclone_tertiary_radius = get_geo_attr('cyclone_tertiary', 'cylinder_diameter', 0.20) / 2.0
        self.cyclone_tertiary_cylinder_height = get_geo_attr('cyclone_tertiary', 'cylinder_height', 0.2)
        self.cyclone_tertiary_cone_height = get_geo_attr('cyclone_tertiary', 'cone_height', 0.3)
        self.cyclone_tertiary_vf_radius = get_geo_attr('cyclone_tertiary', 'vortex_finder_diameter', self.cyclone_tertiary_radius * 0.8) / 2.0
        self.cyclone_tertiary_dust_y = self.cyclone_tertiary_center[1] - self.cyclone_tertiary_cylinder_height - self.cyclone_tertiary_cone_height
        
        # =====================================================================
        # BAG FILTER
        # =====================================================================
        self.bagfilter_center = get_geo_center('bagfilter', [1.5, 0.5, 0])
        self.bagfilter_half_width = get_geo_attr('bagfilter', 'housing_width', 0.4) / 2
        self.bagfilter_half_depth = get_geo_attr('bagfilter', 'housing_depth', 0.4) / 2
        self.bagfilter_height = get_geo_attr('bagfilter', 'housing_height', 1.0)
        self.bagfilter_inlet_y = self.bagfilter_center[1]
        self.bagfilter_outlet_y = self.bagfilter_center[1] + self.bagfilter_height / 2
        self.bagfilter_dust_y = self.bagfilter_center[1] - self.bagfilter_height / 2
        
        # =====================================================================
        # DUCT/CONNECTION GEOMETRY
        # =====================================================================
        connections = geo.get('connections', {})
        
        # Venturi to zigzag duct
        conn_v_z = connections.get('venturi_to_zigzag', {})
        self.duct_venturi_zigzag_start = np.array(conn_v_z.get('start', self.venturi_center + [0, self.venturi_total_length, 0]))
        self.duct_venturi_zigzag_end = np.array(conn_v_z.get('end', [self.zigzag_center[0], self.zigzag_inlet_y, self.zigzag_center[2]]))
        self.duct_venturi_zigzag_radius = conn_v_z.get('radius', 0.04)
        
        # Zigzag to cyclone duct
        conn_z_c = connections.get('zigzag_to_cyclone', {})
        self.duct_zigzag_cyclone_start = np.array(conn_z_c.get('start', [self.zigzag_center[0], self.zigzag_fines_outlet_y, self.zigzag_center[2]]))
        self.duct_zigzag_cyclone_end = np.array(conn_z_c.get('end', self.cyclone_primary_center))
        self.duct_zigzag_cyclone_radius = conn_z_c.get('radius', 0.04)
        
        # Cyclone to bag filter duct
        conn_c_b = connections.get('cyclone_to_bagfilter', {})
        self.duct_cyclone_bag_start = np.array(conn_c_b.get('start', self.cyclone_tertiary_center + [0, 0.1, 0]))
        self.duct_cyclone_bag_end = np.array(conn_c_b.get('end', self.bagfilter_center))
        self.duct_cyclone_bag_radius = conn_c_b.get('radius', 0.04)
        
        # =====================================================================
        # AIR VELOCITIES
        # =====================================================================
        # Compute from volumetric flow rate and cross-sectional areas
        Q_air = cfg.air_flow_rate_m3s
        
        # Venturi inlet velocity
        A_venturi_inlet = np.pi * (self.venturi_inlet_diameter / 2) ** 2
        self.v_air_venturi_inlet = Q_air / A_venturi_inlet
        
        # Zigzag air velocity (upward)
        A_zigzag = self.zigzag_channel_width * self.zigzag_channel_depth
        self.v_air_zigzag = Q_air / A_zigzag
        
        # Cyclone inlet velocity
        A_cyclone_inlet = np.pi * (self.cyclone_primary_radius * 0.2) ** 2  # Tangential inlet
        self.v_air_cyclone_inlet = Q_air / A_cyclone_inlet
        
        # =====================================================================
        # CUT SIZE CALCULATION
        # =====================================================================
        # d50 = sqrt(18 * mu * v_air / (g * (rho_p - rho_f)))
        g = 9.81
        rho_p = cfg.particle_density
        rho_f = cfg.air_density
        mu = cfg.air_viscosity
        
        self.zigzag_d50 = np.sqrt(18 * mu * self.v_air_zigzag / (g * (rho_p - rho_f)))
        
        # Cyclone d50 (approximate - depends on design)
        # d50 ≈ sqrt(9*mu*W / (2*pi*N*v_in*(rho_p-rho_f)))
        N_turns = 5  # Effective turns
        W = self.cyclone_primary_radius * 0.2  # Inlet width
        self.cyclone_d50 = np.sqrt(9 * mu * W / (2 * np.pi * N_turns * self.v_air_cyclone_inlet * (rho_p - rho_f)))
        
        # =====================================================================
        # SYSTEM BOUNDS
        # =====================================================================
        all_centers = [
            self.venturi_center,
            self.zigzag_center,
            self.cyclone_primary_center,
            self.cyclone_secondary_center,
            self.cyclone_tertiary_center,
            self.bagfilter_center,
        ]
        all_centers = np.array(all_centers)
        
        self.system_min = np.min(all_centers, axis=0) - np.array([1.0, 1.0, 1.0])
        self.system_max = np.max(all_centers, axis=0) + np.array([1.0, 2.0, 1.0])
        
        # =====================================================================
        # PRINT SUMMARY
        # =====================================================================
        print(f"\n  Classification Physics Parameters:")
        print(f"\n    Air Flow:")
        print(f"      Flow rate:       {Q_air * 3600:.0f} m³/h")
        print(f"      Venturi inlet:   {self.v_air_venturi_inlet:.1f} m/s")
        print(f"      Zigzag:          {self.v_air_zigzag:.2f} m/s")
        print(f"      Cyclone inlet:   {self.v_air_cyclone_inlet:.1f} m/s")
        
        print(f"\n    Cut Sizes (d50):")
        print(f"      Zigzag:          {self.zigzag_d50 * 1e6:.1f} µm")
        print(f"      Cyclone:         {self.cyclone_d50 * 1e6:.1f} µm")
        
        print(f"\n    For protein separation:")
        print(f"      Protein:         ~10-30 µm (should go to fines)")
        print(f"      Starch:          ~15-60 µm (should go to coarse)")
        
        if self.zigzag_d50 * 1e6 < 15:
            print(f"      Status: Zigzag d50 ({self.zigzag_d50*1e6:.1f}µm) < 15µm - good for protein recovery")
        else:
            print(f"      WARNING: Zigzag d50 ({self.zigzag_d50*1e6:.1f}µm) > 15µm - may lose protein to coarse")
            print(f"               Consider increasing air velocity or reducing channel size")
    
    def _allocate_arrays(self):
        """Allocate particle arrays on device."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.masses = wp.zeros(n, dtype=float, device=self.device)
        self.state.zones = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
    
    def _setup_hash_grid(self):
        """Setup hash grid for particle collisions."""
        extent = self.system_max - self.system_min
        max_extent = max(extent)
        
        grid_dim = max(32, int(max_extent / 0.05))
        
        self._hash_grid = wp.HashGrid(
            dim_x=grid_dim,
            dim_y=grid_dim,
            dim_z=grid_dim,
            device=self.device
        )
    
    def _setup_statistics_arrays(self):
        """Setup arrays for separation statistics."""
        # Single-element arrays for atomic counters
        self._count_coarse = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone1 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone2 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone3 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_bagfilter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_escaped = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_active = wp.zeros(1, dtype=wp.int32, device=self.device)
    
    def initialize_particles(
        self,
        num_particles: int = None,
        mean_diameter: float = 30e-6,   # 30 µm (flour average)
        diameter_std: float = 15e-6,    # 15 µm std dev
        initial_velocity: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    ):
        """
        Initialize particles at the venturi solids inlet.
        
        Args:
            num_particles: Number of particles (default: config value)
            mean_diameter: Mean particle diameter [m] (default 30µm)
            diameter_std: Standard deviation [m] (default 15µm)
            initial_velocity: Initial velocity from feed system [m/s]
        """
        n = num_particles or self.config.num_particles
        n = min(n, self.config.num_particles)
        
        cfg = self.config
        
        # Use the Warp kernel for initialization
        wp.launch(
            kernel=init_classification_particles,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self.state.masses,
                self.state.zones,
                self.state.is_active,
                n,
                wp.vec3(*self.venturi_solids_inlet_pos),
                float(self.venturi_solids_inlet_radius),
                wp.vec3(*initial_velocity),
                float(mean_diameter),
                float(diameter_std),
                float(cfg.particle_density),
                42,  # Random seed
            ],
            device=self.device
        )
        
        self.state.particles_active = n
        
        # Get diameter stats for logging
        diameters = self.state.diameters.numpy()[:n]
        
        print(f"\n  Initialized {n} particles at venturi inlet")
        print(f"    Diameter range: {diameters.min()*1e6:.1f} - {diameters.max()*1e6:.1f} µm")
        print(f"    Mean diameter:  {diameters.mean()*1e6:.1f} µm")
        print(f"    Inlet position: ({self.venturi_solids_inlet_pos[0]*1000:.0f}, {self.venturi_solids_inlet_pos[1]*1000:.0f}, {self.venturi_solids_inlet_pos[2]*1000:.0f}) mm")
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        cfg = self.config
        
        n = self.state.particles_active
        
        if n == 0:
            self.state.time += dt
            self.state.step += 1
            return
        
        # Random seed for turbulent dispersion (changes each step)
        random_seed = self.state.step * 1337 + 42
        
        # Launch main physics kernel
        wp.launch(
            kernel=classification_physics_kernel,
            dim=n,
            inputs=[
                # Particle state
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self.state.masses,
                self.state.zones,
                self.state.is_active,
                n,
                
                # Venturi geometry
                wp.vec3(*self.venturi_center),
                float(self.venturi_inlet_diameter),
                float(self.venturi_throat_diameter),
                float(self.venturi_outlet_diameter),
                float(self.venturi_throat_start),
                float(self.venturi_throat_end),
                float(self.venturi_total_length),
                wp.vec3(*self.venturi_solids_inlet_pos),
                float(self.venturi_solids_inlet_radius),
                
                # Zigzag geometry
                wp.vec3(*self.zigzag_center),
                float(self.zigzag_channel_width),
                float(self.zigzag_channel_depth),
                float(self.zigzag_total_height),
                int(self.zigzag_num_stages),
                float(self.zigzag_stage_height),
                float(self.zigzag_inlet_y),
                float(self.zigzag_fines_outlet_y),
                float(self.zigzag_coarse_outlet_y),
                
                # Primary cyclone
                wp.vec3(*self.cyclone_primary_center),
                float(self.cyclone_primary_radius),
                float(self.cyclone_primary_cylinder_height),
                float(self.cyclone_primary_cone_height),
                float(self.cyclone_primary_vf_radius),
                float(self.cyclone_primary_dust_y),
                
                # Secondary cyclone
                wp.vec3(*self.cyclone_secondary_center),
                float(self.cyclone_secondary_radius),
                float(self.cyclone_secondary_cylinder_height),
                float(self.cyclone_secondary_cone_height),
                float(self.cyclone_secondary_vf_radius),
                float(self.cyclone_secondary_dust_y),
                
                # Tertiary cyclone
                wp.vec3(*self.cyclone_tertiary_center),
                float(self.cyclone_tertiary_radius),
                float(self.cyclone_tertiary_cylinder_height),
                float(self.cyclone_tertiary_cone_height),
                float(self.cyclone_tertiary_vf_radius),
                float(self.cyclone_tertiary_dust_y),
                
                # Bag filter
                wp.vec3(*self.bagfilter_center),
                float(self.bagfilter_half_width),
                float(self.bagfilter_half_depth),
                float(self.bagfilter_height),
                float(self.bagfilter_inlet_y),
                float(self.bagfilter_outlet_y),
                float(self.bagfilter_dust_y),
                
                # Ducts
                wp.vec3(*self.duct_venturi_zigzag_start),
                wp.vec3(*self.duct_venturi_zigzag_end),
                float(self.duct_venturi_zigzag_radius),
                
                wp.vec3(*self.duct_zigzag_cyclone_start),
                wp.vec3(*self.duct_zigzag_cyclone_end),
                float(self.duct_zigzag_cyclone_radius),
                
                wp.vec3(*self.duct_cyclone_bag_start),
                wp.vec3(*self.duct_cyclone_bag_end),
                float(self.duct_cyclone_bag_radius),
                
                # Physics parameters
                float(dt),
                float(9.81),  # Gravity
                float(cfg.particle_density),
                float(cfg.air_density),
                float(cfg.air_viscosity),
                float(cfg.restitution),
                float(cfg.friction),
                
                # Air velocities
                float(self.v_air_venturi_inlet),
                float(self.v_air_zigzag),
                float(self.v_air_cyclone_inlet),
                
                # Turbulence
                float(cfg.turbulent_intensity),
                
                # Random seed
                random_seed,
            ],
            device=self.device
        )
        
        # Post-integration containment
        wp.launch(
            kernel=post_integration_containment,
            dim=n,
            inputs=[
                self.state.positions,
                self.state.velocities,
                self.state.diameters,
                self.state.zones,
                self.state.is_active,
                n,
                wp.vec3(*self.system_min),
                wp.vec3(*self.system_max),
                float(cfg.restitution),
            ],
            device=self.device
        )
        
        # Update time
        self.state.time += dt
        self.state.step += 1
    
    def get_separation_counts(self) -> Dict[str, int]:
        """
        Get particle counts by collection location.
        
        Returns:
            Dictionary with counts for each outlet
        """
        # Reset counters
        self._count_coarse.zero_()
        self._count_cyclone1.zero_()
        self._count_cyclone2.zero_()
        self._count_cyclone3.zero_()
        self._count_bagfilter.zero_()
        self._count_escaped.zero_()
        self._count_active.zero_()
        
        n = self.state.particles_active
        
        wp.launch(
            kernel=count_separation_results,
            dim=n,
            inputs=[
                self.state.zones,
                self.state.is_active,
                n,
                self._count_coarse,
                self._count_cyclone1,
                self._count_cyclone2,
                self._count_cyclone3,
                self._count_bagfilter,
                self._count_escaped,
                self._count_active,
            ],
            device=self.device
        )
        
        return {
            'coarse': int(self._count_coarse.numpy()[0]),
            'cyclone_1': int(self._count_cyclone1.numpy()[0]),
            'cyclone_2': int(self._count_cyclone2.numpy()[0]),
            'cyclone_3_protein': int(self._count_cyclone3.numpy()[0]),
            'bagfilter': int(self._count_bagfilter.numpy()[0]),
            'escaped': int(self._count_escaped.numpy()[0]),
            'active': int(self._count_active.numpy()[0]),
        }
    
    def get_zone_counts(self) -> Dict[str, int]:
        """Get particle counts by current zone."""
        zones = self.state.zones.numpy()[:self.state.particles_active]
        is_active = self.state.is_active.numpy()[:self.state.particles_active]
        
        active_zones = zones[is_active == 1]
        
        return {
            'venturi': int(np.sum((active_zones >= 0) & (active_zones <= 2))),
            'duct_v_z': int(np.sum(active_zones == 10)),
            'zigzag': int(np.sum((active_zones >= 20) & (active_zones <= 23))),
            'coarse_outlet': int(np.sum(active_zones == 30)),
            'elbow_z_c': int(np.sum(active_zones == 40)),
            'duct_z_c': int(np.sum(active_zones == 41)),
            'cyclone_1': int(np.sum(active_zones == 50)),
            'cyclone_2': int(np.sum(active_zones == 51)),
            'cyclone_3': int(np.sum(active_zones == 52)),
            'dust_cy1': int(np.sum(active_zones == 55)),
            'dust_cy2': int(np.sum(active_zones == 56)),
            'dust_cy3': int(np.sum(active_zones == 57)),
            'duct_c_b': int(np.sum((active_zones >= 60) & (active_zones <= 61))),
            'bagfilter': int(np.sum(active_zones == 70)),
            'bagfilter_dust': int(np.sum(active_zones == 75)),
            'clean_air': int(np.sum(active_zones == 80)),
            'exited': int(np.sum(active_zones == 99)),
            'inactive': int(np.sum(is_active == 0)),
        }
    
    def get_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.state.positions.numpy()[:self.state.particles_active]
    
    def get_velocities(self) -> np.ndarray:
        """Get current particle velocities."""
        return self.state.velocities.numpy()[:self.state.particles_active]
    
    def get_diameters(self) -> np.ndarray:
        """Get particle diameters."""
        return self.state.diameters.numpy()[:self.state.particles_active]
    
    def get_zones(self) -> np.ndarray:
        """Get particle zones."""
        return self.state.zones.numpy()[:self.state.particles_active]
    
    def print_separation_summary(self):
        """Print a summary of separation results."""
        counts = self.get_separation_counts()
        total = sum(counts.values())
        
        print(f"\n  Separation Results (t = {self.state.time:.3f}s):")
        print(f"  {'='*50}")
        
        # Coarse (starch)
        pct = 100 * counts['coarse'] / max(1, total)
        print(f"    COARSE (starch):      {counts['coarse']:5d} ({pct:5.1f}%)")
        
        # Cyclone fractions
        for i, key in enumerate(['cyclone_1', 'cyclone_2', 'cyclone_3_protein'], 1):
            pct = 100 * counts[key] / max(1, total)
            label = 'PROTEIN' if i == 3 else f'fines {i}'
            print(f"    Cyclone {i} ({label:8s}): {counts[key]:5d} ({pct:5.1f}%)")
        
        # Bag filter
        pct = 100 * counts['bagfilter'] / max(1, total)
        print(f"    Bag filter:           {counts['bagfilter']:5d} ({pct:5.1f}%)")
        
        # Escaped
        pct = 100 * counts['escaped'] / max(1, total)
        print(f"    Escaped (loss):       {counts['escaped']:5d} ({pct:5.1f}%)")
        
        # Still active
        pct = 100 * counts['active'] / max(1, total)
        print(f"    Still active:         {counts['active']:5d} ({pct:5.1f}%)")
        
        print(f"  {'='*50}")
        
        # Separation quality indicators
        protein_collected = counts['cyclone_3_protein'] + counts['bagfilter']
        starch_collected = counts['coarse']
        
        print(f"\n    Protein recovery (cy3 + bag): {protein_collected}")
        print(f"    Starch recovery (coarse):     {starch_collected}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_classification_simulator(
    air_flow_rate_m3s: float = 0.1,
    particle_density: float = 1450.0,
    num_particles: int = 10000,
    device: str = "cuda",
) -> Tuple['ClassificationSystemAssembly', ClassificationFlowPhysicsSimulator]:
    """
    Create a classification system and flow simulator.
    
    Args:
        air_flow_rate_m3s: Air volumetric flow rate [m³/s]
        particle_density: Particle density [kg/m³] (flour ~1450)
        num_particles: Number of simulation particles
        device: Warp device ('cuda' or 'cpu')
        
    Returns:
        Tuple of (assembly, simulator)
    """
    from ..geometry.assembly.classification import ClassificationSystemAssembly
    
    # Create assembly
    assembly = ClassificationSystemAssembly()
    
    # Create config
    config = ClassificationFlowConfig(
        air_flow_rate_m3s=air_flow_rate_m3s,
        particle_density=particle_density,
        num_particles=num_particles,
        device=device,
    )
    
    # Create simulator
    simulator = ClassificationFlowPhysicsSimulator(assembly, config)
    
    return assembly, simulator
