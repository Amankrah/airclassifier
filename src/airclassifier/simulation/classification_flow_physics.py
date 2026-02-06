"""
Classification Flow Physics Module
==================================

Physics-based simulation for the classification system using NVIDIA Warp.

This module simulates particle separation in the classification system:
- Venturi Eductor: Particle entrainment into airstream
- Zigzag Classifier: Primary separation by terminal velocity (d50 > 50 um)
- Wheel Classifier: Centrifugal separation for fines (d50 ~ 25 um)
- Multi-Cyclone System: Staged collection of fines
- Bag Filter: Final fine particle capture

Flow Path (mandatory):
  Air Supply -> Venturi -> Zigzag (pre-classifier) -> Wheel Classifier (main) -> Cyclones -> Bag Filter

The zigzag is the pre-classifier (gravity-based coarse/fines split). The wheel
classifier is the MAIN classifier and provides the fine cut between zigzag
and cyclones. It uses centrifugal force (1000-5000g) to achieve:
- Coarse rejection: High-inertia particles (starch) -> wheel coarse outlet
- Fine passage: Low-inertia particles (protein) -> wheel fines outlet -> cyclones

Physics implemented:
- Two-phase flow: air velocity field + particle dynamics
- Drag: Schiller-Naumann correlation with relative velocity
- Gravity with buoyancy correction
- Inelastic wall collisions with restitution and friction
- Centrifugal effects in cyclones and wheel classifier
- Turbulent dispersion in zigzag stages
- Wheel classifier blade collisions and separation physics

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
from ..particles import (
    FluidConfig,
    ParticlePhysicsConfig,
    ParticleMaterial,
    create_particle_population,
    create_whole_flour_population,
)


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class ClassificationFlowConfig:
    """
    Configuration for classification flow simulation.

    Classification does not create air or feed - it performs separation on the air and
    feed flow coming in. Particle parameters (num_particles, particle_density,
    visual_particle_diameter, sphericity) are provided by feedclass_flow_physics
    through feed_flow_physics via feed_result. Air parameters (air_density,
    air_viscosity, air_flow_rate_m3s) come from air_result. Particle feed entry
    rate at the solids inlet is computed from solids mass flow and particle
    properties (Warp kernel feed_entry_rate_particles_per_s / feedclass
    compute_feed_entry_rate_particles_per_s). Use from_air_and_feed_results() to
    build config from airclass and feedclass results only. Direct construction is
    for legacy/standalone use only.
    """
    # Particle parameters: provided by feedclass/feed_flow_physics via feed_result (legacy defaults for direct construction)
    num_particles: int = 5000              # Capacity; set from feed_result when using from_air_and_feed_results
    particle_density: float = 1450.0      # [kg/m3] From feed_result
    visual_particle_diameter: float = 0.002  # [m] From feed_result particle_diameter_m
    sphericity: float = 0.75               # [-] From feed_result or material

    # Air properties: from air_result when using from_air_and_feed_results (legacy defaults for direct construction)
    air_density: float = 1.2               # [kg/m3] From air_result / FluidConfig
    air_viscosity: float = 1.81e-5        # [Pa*s] From air_result / FluidConfig

    # Optional; when provided, __post_init__ copies density/viscosity into fields above
    fluid_config: Optional[FluidConfig] = None
    material: Optional[ParticleMaterial] = None

    # Air flow rate [m3/s]; from air_result when using from_air_and_feed_results
    air_flow_rate_m3s: float = 0.0

    # Bypass ratio: fraction of total flow that bypasses venturi+zigzag (0.0-1.0)
    # 0.0 = no bypass (all flow through classification), 0.967 = 96.7% bypass
    # Bypass flow merges back before cyclones, so cyclones see full Q_total
    bypass_ratio: float = 0.0

    # Collision parameters
    restitution: float = 0.3               # Coefficient of restitution (inelastic)
    friction: float = 0.4                   # Friction coefficient

    # Simulation timing
    dt: float = 0.001                      # [s] Time step (1ms for stability)

    # Feed entry rate at solids inlet [particles/s]; from feed_result (particle_feed_rate_per_s)
    # or computed from solids_mass_flow via Warp feed_entry_rate_particles_per_s kernel
    particle_feed_rate: float = 0.0        # 0 = not set; set by from_air_and_feed_results from feed_result

    # Continuous feeding: activate particles gradually instead of all at t=0
    # When True, particles are pre-allocated but inactive; step() activates them at particle_feed_rate
    continuous_feeding: bool = False

    # Max solids loading ratio (mu = m_dot_solids / m_dot_air) for venturi entrainment cap
    # Dilute-phase pneumatic transport: mu < 5 typical; conservative cap at mu=2
    max_loading_ratio: float = 2.0

    # Turbulence parameters (for zigzag mixing)
    turbulent_intensity: float = 0.15      # Fraction of mean velocity (15%)

    # Wheel classifier (main classifier) operating speed [RPM]. When set, overrides
    # assembly wheel RPM for this run (e.g. for sensitivity or target d50).
    wheel_rpm: Optional[float] = None      # None = use assembly default
    
    # Compute device
    device: str = "cuda"                   # Warp device ('cuda' or 'cpu')
    
    def __post_init__(self):
        """Apply FluidConfig/Material and validate."""
        if self.fluid_config is not None:
            self.air_density = self.fluid_config.density
            self.air_viscosity = self.fluid_config.dynamic_viscosity
        if self.material is not None:
            self.particle_density = self.material.density
            self.sphericity = getattr(self.material, 'sphericity', self.sphericity)
        if self.dt > 0.005:
            print(f"Warning: dt={self.dt}s may be too large for stability")
        if self.particle_density > 0 and self.air_density > 0 and self.particle_density < self.air_density:
            print(f"Warning: particle density < air density - particles will float")
        if self.air_flow_rate_m3s > 0 and self.air_flow_rate_m3s < 0.01:
            print(f"Warning: Low air flow rate may cause poor separation")

    @classmethod
    def from_air_and_feed_results(
        cls,
        air_result: Dict[str, Any],
        feed_result: Dict[str, Any],
        classification_assembly: ClassificationSystemAssembly,
        solids_mass_flow_kg_s: Optional[float] = None,
        num_particles_capacity: Optional[int] = None,
        fluid_config: Optional[FluidConfig] = None,
        material: Optional[ParticleMaterial] = None,
        simulation_time_s: float = 180.0,
        **kwargs
    ) -> "ClassificationFlowConfig":
        """
        Build config from airclass and feedclass results only (no magic numbers).
        Particle params from feed_result (feedclass/feed_flow_physics). Air params from
        air_result. Particle feed entry rate at solids inlet from feed_result
        (particle_feed_rate_per_s) or computed from solids_mass_flow_kg_s via
        feedclass compute_feed_entry_rate_particles_per_s (same formula as Warp
        feed_entry_rate_particles_per_s kernel).
        """
        from .feedclass_flow_physics import compute_feed_entry_rate_particles_per_s

        Q_m3s = air_result.get("volume_flow_rate_m3_s", 0.0)
        if Q_m3s <= 0:
            Q_m3s = air_result.get("volume_flow_rate_m3_h", 0.0) / 3600.0
        rho_air = kwargs.pop("rho_air", 1.204)
        mu_air = kwargs.pop("mu_air", 1.82e-5)
        if fluid_config is not None:
            rho_air = fluid_config.density
            mu_air = fluid_config.dynamic_viscosity

        particle_density = feed_result.get("particle_density_kg_m3", 0.0)
        particle_dia_m = feed_result.get("particle_diameter_m", 0.0)
        sphericity = feed_result.get("sphericity", 0.75)
        if material is not None:
            particle_density = material.density
            sphericity = getattr(material, "sphericity", sphericity)
            if hasattr(material, "size_distribution") and material.size_distribution is not None:
                sd = material.size_distribution
                particle_dia_m = (sd.d_min + sd.d_max) / 2.0
        if particle_density <= 0:
            particle_density = 1420.0
        if particle_dia_m <= 0:
            particle_dia_m = 50e-6

        # Feed entry rate at solids inlet: from feed_result or computed from solids mass flow
        particle_feed_rate = feed_result.get("particle_feed_rate_per_s", 0.0)
        if particle_feed_rate <= 0 and solids_mass_flow_kg_s is not None and solids_mass_flow_kg_s > 0:
            particle_feed_rate = compute_feed_entry_rate_particles_per_s(
                solids_mass_flow_kg_s, particle_density, particle_dia_m
            )

        # Venturi entrainment capacity (loading ratio limit)
        # mu = m_dot_solids / m_dot_air
        max_loading = kwargs.pop("max_loading_ratio", 2.0)
        continuous = kwargs.pop("continuous_feeding", True)
        m_dot_air = Q_m3s * rho_air
        m_per_particle = particle_density * (np.pi / 6.0) * particle_dia_m**3

        if particle_feed_rate > 0 and m_dot_air > 0 and m_per_particle > 0:
            m_dot_solids_requested = particle_feed_rate * m_per_particle
            m_dot_solids_max = max_loading * m_dot_air
            m_dot_solids_capped = min(m_dot_solids_requested, m_dot_solids_max)
            if m_dot_solids_requested > m_dot_solids_max:
                print(f"  [Feed cap] Requested {m_dot_solids_requested*3600:.1f} kg/h "
                      f"exceeds venturi capacity {m_dot_solids_max*3600:.1f} kg/h (mu={max_loading:.1f})")
                print(f"             Capped to {m_dot_solids_capped*3600:.1f} kg/h")
        else:
            m_dot_solids_capped = 0.0

        # Capacity: from caller or auto-sized
        n_cap = num_particles_capacity
        if n_cap is None or n_cap <= 0:
            if particle_feed_rate > 0:
                n_cap = max(5000, int(particle_feed_rate * simulation_time_s * 1.2))
            else:
                n_cap = 10000

        # Simulation feed rate: spread N sim particles over a feeding window
        # Each sim particle is a statistical representative; the physical rate
        # (~billions/s for 50um particles) would dump them all in < 1 ms.
        # Instead, compute feed duration from: total sim mass / capped mass flow rate
        # then cap to a reasonable window (e.g., half the simulation time).
        sim_feed_rate = 0.0
        if continuous and n_cap > 0 and m_per_particle > 0:
            total_sim_mass = n_cap * m_per_particle
            if m_dot_solids_capped > 0:
                # Physical feed time for the sim particle mass
                phys_feed_time = total_sim_mass / m_dot_solids_capped
                # But sim particles are a tiny fraction of real mass;
                # scale feed duration so particles trickle in over meaningful time.
                # Use: feed all particles in first half of sim, max 120s
                feed_duration = min(simulation_time_s * 0.5, 120.0)
                feed_duration = max(feed_duration, 1.0)  # at least 1s
            else:
                feed_duration = min(simulation_time_s * 0.5, 120.0)
                feed_duration = max(feed_duration, 1.0)
            sim_feed_rate = n_cap / feed_duration
            print(f"  [Continuous feed] {n_cap} sim particles over {feed_duration:.0f}s "
                  f"= {sim_feed_rate:.0f} particles/s")
            if m_dot_solids_capped > 0:
                print(f"  [Physical rate] {m_dot_solids_capped*3600:.1f} kg/h "
                      f"(capped at mu={max_loading:.1f})")

        return cls(
            air_flow_rate_m3s=Q_m3s,
            air_density=rho_air,
            air_viscosity=mu_air,
            particle_density=particle_density,
            visual_particle_diameter=particle_dia_m,
            sphericity=sphericity,
            num_particles=n_cap,
            particle_feed_rate=sim_feed_rate if continuous else particle_feed_rate,
            continuous_feeding=continuous,
            max_loading_ratio=max_loading,
            fluid_config=fluid_config or FluidConfig.air_at_stp(),
            material=material,
            **kwargs
        )


# =============================================================================
# VENTURI PHYSICS FROM AIR + FEED (geometry and first principles)
# =============================================================================

def get_venturi_geometry_from_assembly(assembly: ClassificationSystemAssembly) -> Dict[str, Any]:
    """Extract venturi geometry from classification assembly (no magic numbers)."""
    venturi = assembly.venturi
    vp = venturi.params
    return {
        "inlet_diameter_m": vp.inlet_diameter,
        "inlet_area_m2": vp.inlet_area,
        "throat_diameter_m": vp.throat_diameter,
        "throat_area_m2": vp.throat_area,
        "outlet_diameter_m": vp.outlet_diameter,
        "total_length_m": vp.total_length,
        "throat_start_m": vp.throat_start_position,
        "throat_end_m": vp.throat_end_position,
        "solids_inlet_diameter_m": getattr(
            venturi.ports.get("solids_inlet"),
            "diameter",
            vp.solids_inlet_diameter if hasattr(vp, "solids_inlet_diameter") else 0.04,
        ),
    }


def compute_venturi_physics_from_air_and_feed(
    air_result: Dict[str, Any],
    feed_result: Dict[str, Any],
    classification_assembly: ClassificationSystemAssembly,
    solids_mass_flow_kg_s: Optional[float] = None,
    rho_air: float = 1.204,
) -> Dict[str, Any]:
    """
    Compute venturi and air-particle interaction from airclass + feedclass results and geometry.
    No magic numbers: continuity (Q = A*v), Bernoulli, momentum transfer, loading ratio.
    """
    geo = get_venturi_geometry_from_assembly(classification_assembly)
    Q_m3s = air_result.get("volume_flow_rate_m3_s", 0.0)
    if Q_m3s <= 0:
        Q_m3s = air_result.get("volume_flow_rate_m3_h", 0.0) / 3600.0
    v_inlet = air_result.get("venturi_inlet_velocity_m_s", 0.0)
    if v_inlet <= 0 and geo["inlet_area_m2"] > 0:
        v_inlet = Q_m3s / geo["inlet_area_m2"]
    A_throat = geo["throat_area_m2"]
    v_throat = Q_m3s / A_throat if A_throat > 0 else 0.0
    segments = feed_result.get("segments", [])
    particle_entry_v = segments[-1]["particle_velocity_along_m_s"] if segments else 0.1
    m_dot_air = Q_m3s * rho_air
    loading_ratio = 0.0
    momentum_transfer_N = 0.0
    pressure_drop_solids_Pa = 0.0
    mixture_density_kg_m3 = rho_air
    if solids_mass_flow_kg_s is not None and solids_mass_flow_kg_s > 0 and m_dot_air > 0:
        loading_ratio = solids_mass_flow_kg_s / m_dot_air
        momentum_transfer_N = solids_mass_flow_kg_s * (v_throat - particle_entry_v)
        if A_throat > 0 and v_throat > 0:
            pressure_drop_solids_Pa = momentum_transfer_N / (A_throat * v_throat)
        particle_density = feed_result.get("particle_density_kg_m3", 1420.0)
        vol_flow_solids = solids_mass_flow_kg_s / particle_density
        vol_flow_air = Q_m3s
        solid_vol_frac = vol_flow_solids / (vol_flow_air + vol_flow_solids) if (vol_flow_air + vol_flow_solids) > 0 else 0.0
        mixture_density_kg_m3 = (1.0 - solid_vol_frac) * rho_air + solid_vol_frac * particle_density
    # Compressibility / choked flow check
    speed_of_sound = 343.0  # m/s at ~20 C
    mach_throat = v_throat / speed_of_sound if speed_of_sound > 0 else 0.0
    Cd_venturi = 0.985
    Q_choked = A_throat * speed_of_sound * Cd_venturi
    flow_limited = Q_m3s > Q_choked

    # Venturi K-factor for system curve: dP = K * Q^2
    A_inlet = geo["inlet_area_m2"]
    if A_throat > 0 and A_inlet > 0:
        k_venturi = 0.5 * rho_air * (1.0 / A_throat**2 - 1.0 / A_inlet**2)
    else:
        k_venturi = 0.0

    # Bernoulli pressure drop (air only, incompressible)
    pressure_drop_bernoulli_Pa = 0.5 * rho_air * (v_throat**2 - v_inlet**2)

    return {
        "volume_flow_rate_m3_s": Q_m3s,
        "venturi_inlet_velocity_m_s": v_inlet,
        "venturi_throat_velocity_m_s": v_throat,
        "venturi_throat_area_m2": A_throat,
        "particle_entry_velocity_m_s": particle_entry_v,
        "loading_ratio": loading_ratio,
        "momentum_transfer_N": momentum_transfer_N,
        "pressure_drop_solids_Pa": pressure_drop_solids_Pa,
        "pressure_drop_bernoulli_Pa": pressure_drop_bernoulli_Pa,
        "mixture_density_kg_m3": mixture_density_kg_m3,
        "venturi_geometry": geo,
        # System curve feedback for blower operating point
        "venturi_k_factor": k_venturi,
        "venturi_choked_flow_m3s": Q_choked,
        "venturi_choked_flow_m3h": Q_choked * 3600.0,
        "venturi_mach_throat": mach_throat,
        "venturi_flow_limited": flow_limited,
    }


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
    
    # Optional per-particle type/density (for material-based populations)
    particle_types: Any = None   # wp.array(dtype=wp.int32) - 0=protein, 1=starch, 2=fiber
    densities: Any = None        # wp.array(dtype=float) - per-particle density
    
    # Number of active particles (slots 0..particles_active-1)
    particles_active: int = 0
    
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

    # Wheel classifier (mandatory, between zigzag fines and cyclones)
    # Centrifugal separator for fine particle classification (d50 ~ 20-25 um)
    WHEEL_HOUSING = 34          # In annular chamber around wheel
    WHEEL_FINES_OUTLET = 35     # Through hub center to cyclones (protein)
    WHEEL_COARSE_HOPPER = 36    # In conical hopper (starch rejected by wheel)
    WHEEL_COARSE_COLLECTED = 37 # Collected at wheel coarse outlet

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
    # Deflector plate geometry (for proper separation physics)
    plate_angle: float = 0.0            # [rad] Plate angle from vertical
    plate_length: float = 0.0           # [m] Length of deflector plate
    plate_length_ratio: float = 0.0     # Plate length / channel width
    throat_width: float = 0.0           # [m] Width at throat (constriction)
    blockage_ratio: float = 0.0         # Fraction of channel blocked by plate
    velocity_ratio_throat: float = 0.0  # v_throat / v_bulk (continuity)
    velocity_ratio_in_zone: float = 0.0 # v_zone / v_bulk (from ZigzagClassifierParams)
    recirculation_length_ratio: float = 0.0  # Separation zone length / plate length (from params)
    turbulence_intensity_zigzag: float = 0.0  # Turbulence intensity in separation zones
    
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
    
    When use_preclassification is False, venturi and zigzag are omitted;
    geometry['wheel_only_entry'] provides air_inlet_pos, solids_inlet_pos, junction for particle entry.
    
    Args:
        assembly: ClassificationSystemAssembly instance
        
    Returns:
        Dictionary mapping component names to their geometry
    """
    positions = assembly.get_component_positions()
    geometry = {}
    use_preclassification = getattr(assembly.params, 'use_preclassification', True)

    # =========================================================================
    # VENTURI EDUCTOR GEOMETRY (only when use_preclassification)
    # =========================================================================
    if use_preclassification and assembly.venturi is not None:
        venturi = assembly.venturi
        venturi_pos = np.array(positions['venturi'])
        venturi_ports = venturi.ports
        vp = venturi.params
        geometry['venturi'] = ComponentGeometry(
            center=venturi_pos,
            axis=vp.axis,  # Typically 'y' - vertical
            radius=vp.inlet_diameter / 2,
            length=vp.total_length,
            inlet_pos=venturi_pos + np.array(venturi_ports['air_inlet'].position),
            inlet_dir=np.array(venturi_ports['air_inlet'].direction),
            inlet_diameter=venturi_ports['air_inlet'].diameter,
            outlet_pos=venturi_pos + np.array(venturi_ports['outlet'].position),
            outlet_dir=np.array(venturi_ports['outlet'].direction),
            outlet_diameter=venturi_ports['outlet'].diameter,
            throat_diameter=vp.throat_diameter,
            throat_start=vp.throat_start_position,
            throat_end=vp.throat_end_position,
            solids_inlet_pos=venturi_pos + np.array(venturi_ports['solids_inlet'].position),
            solids_inlet_dir=np.array(venturi_ports['solids_inlet'].direction),
            solids_inlet_diameter=venturi_ports['solids_inlet'].diameter,
        )
    else:
        # Wheel-only assembly: no venturi; particle entry at solids chute / wheel inlet
        geometry['venturi'] = None
        wheel_only_solids = getattr(assembly, '_wheel_only_solids_inlet_pos', None)
        wheel_only_air = getattr(assembly, '_wheel_only_air_inlet_pos', None)
        wheel_only_junction = getattr(assembly, '_wheel_only_junction', None)
        wheel_inlet_pos = None
        if assembly.wheel_classifier is not None and 'wheel_classifier' in positions:
            wheel_pos = np.array(positions['wheel_classifier'])
            wheel_inlet_pos = wheel_pos + np.array(assembly.wheel_classifier.ports['inlet'].position)
        geometry['wheel_only_entry'] = {
            'air_inlet_pos': wheel_only_air if wheel_only_air is not None else np.array([0.0, 0.0, 0.0]),
            'solids_inlet_pos': wheel_only_solids if wheel_only_solids is not None else (wheel_inlet_pos if wheel_inlet_pos is not None else np.array([0.0, 0.4, 0.0])),
            'junction': wheel_only_junction if wheel_only_junction is not None else np.array([0.0, 0.4, 0.0]),
            'solids_inlet_diameter': getattr(assembly.params, 'wheel_only_solids_chute_diameter', 0.05),
            'solids_inlet_radius': getattr(assembly.params, 'wheel_only_solids_chute_diameter', 0.05) / 2.0,
        }
        if geometry['wheel_only_entry']['solids_inlet_pos'] is None:
            geometry['wheel_only_entry']['solids_inlet_pos'] = geometry['wheel_only_entry']['junction'].copy()
        if wheel_inlet_pos is not None:
            geometry['wheel_only_entry']['wheel_inlet_pos'] = wheel_inlet_pos

    # =========================================================================
    # ZIGZAG CLASSIFIER GEOMETRY (only when use_preclassification)
    # =========================================================================
    if use_preclassification and assembly.zigzag is not None:
        zigzag = assembly.zigzag
        zigzag_pos = np.array(positions['zigzag'])
        zigzag_ports = zigzag.ports
        zp = zigzag.params
        geometry['zigzag'] = ComponentGeometry(
            center=zigzag_pos,
            axis='y',  # Vertical classifier
            length=zp.total_height,
            total_height=zp.total_height,
            channel_width=zp.channel_width,
            channel_depth=zp.channel_depth,
            num_stages=zp.num_stages,
            stage_height=zp.stage_height,
            plate_angle=zp.plate_angle,
            plate_length=zp.plate_length,
            plate_length_ratio=zp.plate_length_ratio,
            throat_width=zp.throat_width,
            blockage_ratio=zp.blockage_ratio,
            velocity_ratio_throat=zp.velocity_ratio_throat,
            velocity_ratio_in_zone=zp.velocity_ratio_in_zone,
            recirculation_length_ratio=zp.recirculation_length_ratio,
            turbulence_intensity_zigzag=zp.turbulence_intensity,
            inlet_pos=zigzag_pos + np.array(zigzag_ports['air_inlet'].position),
            inlet_dir=np.array(zigzag_ports['air_inlet'].direction),
            inlet_width=zigzag_ports['air_inlet'].width,
            inlet_height=zigzag_ports['air_inlet'].height if hasattr(zigzag_ports['air_inlet'], 'height') else zp.channel_depth,
            fines_outlet_pos=zigzag_pos + np.array(zigzag_ports['fines_outlet'].position),
            outlet_pos=zigzag_pos + np.array(zigzag_ports['fines_outlet'].position),
            outlet_dir=np.array(zigzag_ports['fines_outlet'].direction),
            outlet_width=zigzag_ports['fines_outlet'].width,
            outlet_height=zigzag_ports['fines_outlet'].height if hasattr(zigzag_ports['fines_outlet'], 'height') else zp.channel_depth,
            coarse_outlet_pos=zigzag_pos + np.array(zigzag_ports['coarse_outlet'].position),
        )
    else:
        geometry['zigzag'] = None

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
            'inlet_width': stage_params.inlet_width,
            'inlet_height': stage_params.inlet_height,
            'design_d50': stage.design_d50,  # [m] target cut size for staged collection
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
    all_duct_sections = list(assembly._duct_sections)
    if hasattr(assembly, '_collection_duct_sections'):
        all_duct_sections.extend(assembly._collection_duct_sections)
    for duct, position in all_duct_sections:
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
    # DROPOUT HOPPER GEOMETRY (if present)
    # =========================================================================
    geometry['dropout'] = None
    for duct, position in assembly._duct_sections:
        if type(duct).__name__ == 'ExpandingTransitionWithDropout':
            dropout_pos = np.array(position)
            dp = duct.params
            dropout_port = duct.ports['dropout']
            geometry['dropout'] = {
                'position': dropout_pos,
                'inlet_diameter': dp.inlet_diameter,
                'transition_length': dp.transition_length,
                'hopper_height': dp.hopper_height,
                'hopper_outlet_diameter': dp.hopper_outlet_diameter,
                'hopper_outlet_pos': dropout_pos + np.array(dropout_port.position),
            }
            break

    # =========================================================================
    # BYPASS GEOMETRY (if present)
    # =========================================================================
    geometry['bypass'] = None
    if hasattr(assembly, '_bypass_split_tee'):
        split_pos = np.array(assembly._bypass_split_tee_pos)
        merge_pos = np.array(assembly._bypass_merge_tee_pos)
        split_branch = assembly._bypass_split_tee.ports['branch']
        merge_branch = assembly._bypass_merge_tee.ports['branch']

        geometry['bypass'] = {
            'split_pos': split_pos,
            'merge_pos': merge_pos,
            'split_branch_world': split_pos + np.array(split_branch.position),
            'merge_branch_world': merge_pos + np.array(merge_branch.position),
            'diameter': assembly.params.bypass_diameter,
        }

    # =========================================================================
    # COARSE COLLECTION HARDWARE (if present; only with zigzag)
    # =========================================================================
    geometry['coarse_collection'] = None
    if (use_preclassification and assembly.zigzag is not None
            and hasattr(assembly.params, 'include_coarse_collection') and assembly.params.include_coarse_collection):
        zigzag_coarse_port = assembly.zigzag.ports['coarse_outlet']
        zigzag_pos = assembly._component_positions['zigzag']
        coarse_world = zigzag_pos + np.array(zigzag_coarse_port.position)
        geometry['coarse_collection'] = {
            'port_position': coarse_world,
            'port_width': zigzag_coarse_port.width,
            'port_height': zigzag_coarse_port.height,
            'airlock_rotor_d': assembly.params.coarse_airlock_rotor_d,
        }

    # DROPOUT COLLECTION HARDWARE (if present)
    geometry['dropout_collection'] = None
    if (hasattr(assembly.params, 'include_dropout_collection')
            and assembly.params.include_dropout_collection
            and geometry.get('dropout') is not None):
        geometry['dropout_collection'] = {
            'hopper_outlet_pos': geometry['dropout']['hopper_outlet_pos'],
            'hopper_outlet_d': geometry['dropout']['hopper_outlet_diameter'],
            'airlock_rotor_d': assembly.params.dropout_airlock_rotor_d,
        }

    # =========================================================================
    # WHEEL CLASSIFIER GEOMETRY (mandatory - between zigzag and cyclones)
    # =========================================================================
    # The wheel classifier is required for fine particle separation (d50 ~ 25 um)
    wheel = assembly.wheel_classifier
    wheel_pos = np.array(assembly._component_positions.get('wheel_classifier', [0, 0, 0]))
    wp_params = wheel.params

    # Extract port positions
    wheel_ports = wheel.ports
    inlet_port = wheel_ports.get('inlet')
    fines_port = wheel_ports.get('fines_outlet')
    coarse_port = wheel_ports.get('coarse_outlet')

    geometry['wheel_classifier'] = {
        'position': wheel_pos,
        'wheel_diameter': wp_params.wheel_diameter,
        'wheel_radius': wp_params.wheel_diameter / 2.0,
        'hub_diameter': wp_params.hub_diameter,
        'hub_radius': wp_params.hub_diameter / 2.0,
        'wheel_width': wp_params.wheel_width,
        'num_blades': wp_params.num_blades,
        'blade_thickness': wp_params.blade_thickness,
        'blade_gap': wp_params.blade_gap,
        'blade_passage_area': wp_params.blade_passage_area,
        'rpm': wp_params.rpm,
        'omega': wp_params.rpm * 2.0 * np.pi / 60.0,  # rad/s
        'housing_radius': wp_params.wheel_diameter / 2.0 + wp_params.volute_clearance,
        'hopper_height': wp_params.coarse_hopper_height,
        'hopper_half_angle': np.radians(wp_params.coarse_hopper_angle),
        'fines_outlet_diameter': wp_params.fines_outlet_diameter,
        'coarse_outlet_diameter': wp_params.coarse_outlet_diameter,
        # Port world positions
        'inlet_pos': wheel_pos + np.array(inlet_port.position) if inlet_port else wheel_pos,
        'fines_outlet_pos': wheel_pos + np.array(fines_port.position) if fines_port else wheel_pos + np.array([0, 0.1, 0]),
        'coarse_outlet_pos': wheel_pos + np.array(coarse_port.position) if coarse_port else wheel_pos + np.array([0, -0.1, 0]),
    }

    # =========================================================================
    # CONNECTION PATHS (computed from actual port positions)
    # =========================================================================
    connections = {}
    
    # Venturi outlet -> Zigzag air inlet (or wheel-only: air/solids junction -> wheel inlet)
    if geometry.get('venturi') is not None and geometry.get('zigzag') is not None:
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
            'end_diameter': geometry['zigzag'].inlet_width,
            'avg_radius': geometry['venturi'].outlet_diameter / 2,
        }
        zigzag_fines = geometry['zigzag'].fines_outlet_pos
    else:
        # Wheel-only: duct from junction to wheel inlet; "zigzag_to_cyclone" = wheel fines to cyclone
        wo = geometry.get('wheel_only_entry', {})
        junction = wo.get('junction', np.array([0.0, 0.4, 0.0]))
        wheel_inlet_pos = wo.get('wheel_inlet_pos')
        if wheel_inlet_pos is None and geometry.get('wheel_classifier') is not None:
            wg = geometry['wheel_classifier']
            wheel_inlet_pos = np.array(wg['inlet_pos'])
        if wheel_inlet_pos is None:
            wheel_inlet_pos = junction + np.array([0.3, 0.0, 0.0])
        air_inlet = wo.get('air_inlet_pos', np.array([0.0, 0.0, 0.0]))
        conn_vec = junction - air_inlet
        conn_len = float(np.linalg.norm(conn_vec)) or 0.4
        conn_dir = conn_vec / max(np.linalg.norm(conn_vec), 1e-9)
        connections['venturi_to_zigzag'] = {
            'start_pos': air_inlet.copy(),
            'end_pos': np.array(wheel_inlet_pos),
            'direction': (np.array(wheel_inlet_pos) - np.array(junction)) / max(np.linalg.norm(np.array(wheel_inlet_pos) - np.array(junction)), 1e-9),
            'length': float(np.linalg.norm(np.array(wheel_inlet_pos) - np.array(junction))) or 0.3,
            'start_diameter': 0.07,
            'end_diameter': 0.06,
            'avg_radius': 0.03,
        }
        zigzag_fines = np.array(geometry['wheel_classifier']['fines_outlet_pos'])

    # Zigzag fines (or wheel fines when wheel-only) -> Cyclone inlet
    cyclone_in = geometry['multi_cyclone'].inlet_pos
    conn_vec = cyclone_in - np.asarray(zigzag_fines)
    conn_len = float(np.linalg.norm(conn_vec))
    conn_dir = conn_vec / max(conn_len, 1e-6)
    if geometry.get('zigzag') is not None:
        start_d = geometry['zigzag'].outlet_width
    else:
        start_d = geometry['wheel_classifier']['fines_outlet_diameter']
    connections['zigzag_to_cyclone'] = {
        'start_pos': np.asarray(zigzag_fines).copy(),
        'end_pos': cyclone_in.copy(),
        'direction': conn_dir,
        'length': conn_len,
        'start_diameter': start_d,
        'end_diameter': geometry['multi_cyclone'].inlet_width,
        'avg_radius': (start_d + geometry['multi_cyclone'].inlet_width) / 4,
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
    
    # Dropout hopper
    dropout = geometry.get('dropout')
    if dropout is not None:
        print("\n5. COARSE DROPOUT HOPPER")
        print(f"   Position:         ({dropout['position'][0]*1000:.1f}, {dropout['position'][1]*1000:.1f}, {dropout['position'][2]*1000:.1f}) mm")
        print(f"   Inlet diameter:   {dropout['inlet_diameter']*1000:.1f} mm")
        print(f"   Transition length:{dropout['transition_length']*1000:.1f} mm")
        print(f"   Hopper height:    {dropout['hopper_height']*1000:.1f} mm")
        print(f"   Hopper outlet D:  {dropout['hopper_outlet_diameter']*1000:.1f} mm")
        print(f"   Discharge pos:    ({dropout['hopper_outlet_pos'][0]*1000:.1f}, {dropout['hopper_outlet_pos'][1]*1000:.1f}, {dropout['hopper_outlet_pos'][2]*1000:.1f}) mm")

    # Bypass
    bypass = geometry.get('bypass')
    if bypass is not None:
        print("\n6. BYPASS DUCT")
        print(f"   Split position:   ({bypass['split_pos'][0]*1000:.1f}, {bypass['split_pos'][1]*1000:.1f}, {bypass['split_pos'][2]*1000:.1f}) mm")
        print(f"   Merge position:   ({bypass['merge_pos'][0]*1000:.1f}, {bypass['merge_pos'][1]*1000:.1f}, {bypass['merge_pos'][2]*1000:.1f}) mm")
        print(f"   Bypass diameter:  {bypass['diameter']*1000:.1f} mm")

    # Coarse collection hardware
    coarse_coll = geometry.get('coarse_collection')
    if coarse_coll is not None:
        pos = coarse_coll['port_position']
        print("\n7. COARSE COLLECTION (Zigzag)")
        print(f"   Port position:    ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
        print(f"   Port size:        {coarse_coll['port_width']*1000:.1f} x {coarse_coll['port_height']*1000:.1f} mm")
        print(f"   Airlock rotor D:  {coarse_coll['airlock_rotor_d']*1000:.1f} mm")

    # Dropout collection hardware
    dropout_coll = geometry.get('dropout_collection')
    if dropout_coll is not None:
        pos = dropout_coll['hopper_outlet_pos']
        print("\n8. DROPOUT COLLECTION (Hopper)")
        print(f"   Outlet position:  ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
        print(f"   Outlet diameter:  {dropout_coll['hopper_outlet_d']*1000:.1f} mm")
        print(f"   Airlock rotor D:  {dropout_coll['airlock_rotor_d']*1000:.1f} mm")

    # Connections
    section_num = 5 + (1 if dropout else 0) + (1 if bypass else 0) + (1 if coarse_coll else 0) + (1 if dropout_coll else 0)
    print(f"\n{section_num}. CONNECTION PATHS")
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
    # FEED ENTRY RATE AT SOLIDS INLET (from feedclass / solids mass flow)
    # -------------------------------------------------------------------------
    @wp.func
    def particle_mass_from_density_diameter(rho_p: float, d_p: float) -> float:
        """
        Mass per particle [kg]: m = rho_p * (pi/6) * d_p^3.
        Used to compute feed entry rate at solids inlet: N_dot = m_dot_solids / m_particle.
        """
        return rho_p * (PI / 6.0) * (d_p * d_p * d_p)

    @wp.func
    def feed_entry_rate_particles_per_s(
        solids_mass_flow_kg_s: float,
        rho_p: float,
        d_p: float,
    ) -> float:
        """
        Particle feed entry rate at solids inlet [particles/s] from mass flow and particle properties.
        N_dot = m_dot_solids / m_particle; m_particle from particle_mass_from_density_diameter.
        """
        m_p = particle_mass_from_density_diameter(rho_p, d_p)
        if m_p <= 0.0:
            return 0.0
        return solids_mass_flow_kg_s / m_p

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
        stage_height: float,
        plate_angle: float,
        plate_length: float,
        throat_width: float,
        velocity_ratio_zone: float,
        recirculation_length_ratio: float,
    ) -> wp.vec3:
        """
        Compute air velocity field in zigzag classifier (matches ZigzagClassifierParams).

        DEFLECTOR PLATE PHYSICS (from zigzag_classifier.py):
        1. THROAT: Constriction at plate tip; continuity v_throat = v_mean * (channel_width/throat_width).
        2. SEPARATION ZONE: Recirculation behind plate; height = recirculation_length_ratio * plate_vertical;
           width = 0.8 * plate_horizontal (SeparationZone in component); v_zone = v_mean * velocity_ratio_zone.
        3. TRANSPORT: Straight channel between zones; v = v_mean (bulk).

        Plate at center of each stage (y_plate_base = (stage - 0.5)*stage_height). Throat at plate tip +/- 0.2*stage_height (component).
        """
        local_y = pos[1] - zigzag_center[1]
        local_x = pos[0] - zigzag_center[0]

        v_y = v_mean
        v_x = 0.0

        if num_stages > 0 and stage_height > 0.0:
            stage_num = int(local_y / stage_height)
            pos_in_stage = local_y - float(stage_num) * stage_height

            plate_on_left = (stage_num % 2) == 0
            plate_horizontal = plate_length * wp.sin(plate_angle)
            plate_vertical = plate_length * wp.cos(plate_angle)
            half_width = channel_width / 2.0

            # Plate at vertical center of stage (zigzag_classifier: y_plate_base = (stage - 0.5)*stage_height)
            plate_center_in_stage = 0.5 * stage_height
            plate_tip_in_stage = plate_center_in_stage + plate_vertical

            # Separation zone height from ZigzagClassifierParams.separation_zone_height
            separation_height = recirculation_length_ratio * plate_vertical
            dy_from_plate_center = pos_in_stage - plate_center_in_stage

            # Separation zone: above plate center, within separation_height; lateral extent 0.8*plate_horizontal (SeparationZone.width)
            zone_lateral_extent = plate_horizontal * 0.8
            in_separation_zone = False
            if dy_from_plate_center > 0.0 and dy_from_plate_center < separation_height:
                if plate_on_left:
                    if local_x >= -half_width and local_x <= -half_width + zone_lateral_extent:
                        in_separation_zone = True
                else:
                    if local_x >= half_width - zone_lateral_extent and local_x <= half_width:
                        in_separation_zone = True

            # Throat at plate tip +/- 0.2*stage_height (zigzag_classifier get_zone_at: throat_y_min/max = tip_y +/- 0.2*stage_height)
            throat_half_span = 0.2 * stage_height
            throat_y_min = plate_tip_in_stage - throat_half_span
            throat_y_max = plate_tip_in_stage + throat_half_span
            in_throat_zone = pos_in_stage >= throat_y_min and pos_in_stage <= throat_y_max

            if in_separation_zone:
                v_y = v_mean * velocity_ratio_zone
            elif in_throat_zone:
                v_y = v_mean * (channel_width / throat_width)
                v_throat = v_y
                v_x = v_throat * wp.sin(plate_angle)
                if not plate_on_left:
                    v_x = -v_x
            else:
                v_y = v_mean

        return wp.vec3(v_x, v_y, 0.0)

    @wp.func
    def compute_turbulent_dispersion(
        v_ref: float,
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

        v' = I * v_ref * random_direction

        Where I = turbulent intensity (typically 0.1-0.2 for zigzag)
        v_ref = reference air velocity (zone velocity, NOT particle velocity)

        Using air velocity as reference ensures turbulence doesn't vanish
        for slow particles near d50 - those particles NEED turbulence most
        to break the equilibrium and get classified either way.
        """
        # Proper random numbers using Warp's RNG (changes every timestep via seed)
        state = wp.rand_init(seed, tid)

        # Fluctuation based on AIR velocity, not particle velocity
        fluctuation = turbulent_intensity * v_ref

        # Uniform random in [-1, 1] for each component
        vx = fluctuation * (wp.randf(state) - 0.5) * 2.0
        vy = fluctuation * (wp.randf(state) - 0.5) * 2.0
        vz = fluctuation * (wp.randf(state) - 0.5) * 2.0

        return wp.vec3(vx, vy, vz)

    # -------------------------------------------------------------------------
    # CYCLONE SEPARATION PHYSICS
    # -------------------------------------------------------------------------
    @wp.func
    def compute_cyclone_tangential_velocity(
        pos: wp.vec3,
        cyclone_center: wp.vec3,
        inlet_velocity: float,
        cyclone_radius: float,
        vortex_finder_radius: float,
    ) -> wp.vec3:
        """
        Compute tangential velocity field in cyclone (matches CycloneGeometryParams).

        Rankine vortex: inner core = solid body, outer = free vortex (angular momentum).
        Core radius = vortex_finder_radius (inner/outer vortex boundary from geometry).
        Boundary: v_tan(R) = v_inlet (tangential inlet).
        """
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)

        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)

        r_core = vortex_finder_radius

        if r < r_core:
            v_tan = inlet_velocity * cyclone_radius / r_core * r / r_core
        else:
            v_tan = inlet_velocity * cyclone_radius / r

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

        RADIAL FLOW (inward toward vortex finder):
        - Outer region (r > r_vf): inward flow; r_transition = vortex_finder_radius (geometry).
        - Magnitude scaled by inlet flow; profile (0.5 + 0.5*(1-r_frac)) gives peak near transition.
        - Inner region (r <= r_vf): negligible radial (flow primarily axial upward).
        """
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        local_y = pos[1] - cyclone_center[1]

        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)

        r_transition = vortex_finder_radius

        if r > r_transition:
            r_frac = (r - r_transition) / (cyclone_radius - r_transition + eps)
            r_frac = wp.clamp(r_frac, 0.0, 1.0)
            v_radial_mag = -0.15 * v_inlet * (0.5 + 0.5 * (1.0 - r_frac))
        else:
            # Inner vortex: negligible radial (primarily axial upward)
            v_radial_mag = 0.0

        # In cone section, converging walls intensify inward flow
        if local_y < -cylinder_height:
            cone_progress = (-local_y - cylinder_height) / (cone_height + eps)
            cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
            v_radial_mag = v_radial_mag * (1.0 + cone_progress)

        # Radial direction (negative magnitude = inward)
        return wp.vec3(v_radial_mag * dx / r, 0.0, v_radial_mag * dz / r)

    @wp.func
    def compute_cyclone_axial_velocity(
        pos: wp.vec3,
        cyclone_center: wp.vec3,
        cyclone_radius: float,
        vortex_finder_radius: float,
        cylinder_height: float,
        cone_height: float,
        v_inlet: float
    ) -> wp.vec3:
        """
        Compute axial (vertical) velocity in cyclone.

        AXIAL FLOW PATTERN (height-dependent):
        - Outer region (r > r_vf): DOWNWARD toward dust outlet
          Accelerates in cone section (converging walls)
        - Inner region (r < r_vf): UPWARD toward vortex finder
          Strongest near the top where air exits through vortex finder

        Double helix flow pattern:
        - Outer spiral goes DOWN (carries large particles to dust outlet)
        - Inner spiral goes UP (carries small particles to vortex finder exit)
        """
        dx = pos[0] - cyclone_center[0]
        dz = pos[2] - cyclone_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        local_y = pos[1] - cyclone_center[1]

        eps = 1.0e-6
        r_transition = vortex_finder_radius

        if r > r_transition:
            # Outer region: downward flow
            v_axial = -0.2 * v_inlet
            # Stronger in cone section (converging walls accelerate downward flow)
            if local_y < -cylinder_height:
                cone_progress = (-local_y - cylinder_height) / (cone_height + eps)
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                v_axial = v_axial * (1.0 + 0.5 * cone_progress)
        else:
            # Inner region: upward flow toward vortex finder
            # Stronger toward center and stronger near the top
            inner_factor = 1.0 - r / r_transition
            v_axial = 0.5 * v_inlet * inner_factor
            # Strengthen near the top (approaching vortex finder exit)
            if local_y > -cylinder_height * 0.5:
                # Upper half: boost upward flow (vortex finder suction)
                height_factor = 1.0 + (local_y + cylinder_height * 0.5) / (cylinder_height * 0.5 + eps)
                height_factor = wp.clamp(height_factor, 1.0, 1.5)
                v_axial = v_axial * height_factor

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
    # WHEEL CLASSIFIER PHYSICS (Centrifugal Separator)
    # -------------------------------------------------------------------------
    @wp.func
    def compute_wheel_radial_velocity(
        pos: wp.vec3,
        wheel_center: wp.vec3,
        wheel_radius: float,
        hub_radius: float,
        volumetric_flow: float,
        wheel_width: float,
        num_blades: int,
        blade_thickness: float,
    ) -> float:
        """
        Radial inward air velocity through wheel blade gaps.

        CONTINUITY IN CLASSIFIER WHEEL (matches wheel_classifier.WheelClassifierParams):
        Air flows through the open area between radial blades, not the full annulus.
        At radius r: open circumferential length = 2*PI*r - num_blades*blade_thickness
        Flow area A(r) = (2*PI*r - num_blades*blade_thickness) * wheel_width
        Continuity: Q = |v_r| * A(r), so v_r = -Q / A(r) (inward = negative).

        Cut size d50 from force balance: d50 = sqrt(18*mu*v_r / (Delta_rho*omega^2*r)).

        Args:
            pos: Particle position [m]
            wheel_center: Center of wheel [m]
            wheel_radius: Outer radius of wheel [m]
            hub_radius: Inner hub radius (fines outlet) [m]
            volumetric_flow: Air flow rate through wheel [m3/s]
            wheel_width: Axial width of wheel (blade height) [m]
            num_blades: Number of radial blades
            blade_thickness: Blade thickness [m]

        Returns:
            Radial velocity magnitude (negative = inward) [m/s]
        """
        dx = pos[0] - wheel_center[0]
        dz = pos[2] - wheel_center[2]
        r = wp.sqrt(dx * dx + dz * dz)
        r = wp.clamp(r, hub_radius, wheel_radius)

        eps = 1.0e-6
        if wheel_width < eps:
            return 0.0

        # Blade blockage: open arc at radius r = 2*PI*r - num_blades*blade_thickness
        open_arc = 2.0 * PI * r - float(num_blades) * blade_thickness
        area = open_arc * wheel_width
        area = wp.max(area, eps)
        v_radial = -volumetric_flow / area
        return v_radial

    @wp.func
    def compute_wheel_tangential_velocity(
        pos: wp.vec3,
        wheel_center: wp.vec3,
        wheel_radius: float,
        hub_radius: float,
        omega: float,
    ) -> wp.vec3:
        """
        Tangential velocity from rotating wheel (drives air rotation).

        WHEEL ROTATION:
        The cage wheel rotates at omega rad/s, entraining air in tangential motion.
        At radius r, the air tangential velocity is approximately:
        v_tan = omega*r (solid body rotation, assuming good blade coupling)

        The rotating air creates centrifugal force on particles:
        F_c = m*v_tan^2/r = m*omega^2*r

        Args:
            pos: Particle position [m]
            wheel_center: Center of wheel [m]
            wheel_radius: Outer radius [m]
            hub_radius: Inner hub radius [m]
            omega: Angular velocity [rad/s]

        Returns:
            Tangential velocity vector [m/s]
        """
        dx = pos[0] - wheel_center[0]
        dz = pos[2] - wheel_center[2]
        r = wp.sqrt(dx * dx + dz * dz)

        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)

        # Solid-body rotation only within wheel geometry (hub to rim)
        if r < hub_radius or r > wheel_radius:
            v_tan = 0.0
        else:
            v_tan = omega * r

        # Tangential direction: perpendicular to radial in XZ plane
        # Counter-clockwise when viewed from above (+Y)
        tan_x = -dz / r
        tan_z = dx / r

        return wp.vec3(v_tan * tan_x, 0.0, v_tan * tan_z)

    @wp.func
    def check_wheel_blade_collision(
        pos: wp.vec3,
        wheel_center: wp.vec3,
        num_blades: int,
        blade_thickness: float,
        wheel_radius: float,
        hub_radius: float,
        omega: float,
        time: float,
    ) -> wp.vec3:
        """
        Check if particle collides with rotating blade, return collision normal.

        BLADE COLLISION:
        The wheel has num_blades radial blades rotating at omega.
        Each blade sweeps an angle of 2*PI/num_blades.
        We check if the particle is within blade_thickness/2 of any blade.

        Args:
            pos: Particle position [m]
            wheel_center: Center of wheel [m]
            num_blades: Number of radial blades
            blade_thickness: Thickness of each blade [m]
            wheel_radius: Outer radius [m]
            hub_radius: Inner hub radius [m]
            omega: Angular velocity [rad/s]
            time: Current simulation time [s]

        Returns:
            Collision normal (blade surface normal) or zero vector if no collision
        """
        dx = pos[0] - wheel_center[0]
        dz = pos[2] - wheel_center[2]
        r = wp.sqrt(dx * dx + dz * dz)

        # Blade radial extent: hub to wheel rim (geometry from wheel_classifier)
        if r < hub_radius or r > wheel_radius:
            return wp.vec3(0.0, 0.0, 0.0)

        eps = 1.0e-6
        if r < eps:
            return wp.vec3(0.0, 0.0, 0.0)

        # Blade angular spacing
        blade_angle_step = TWO_PI / float(num_blades)

        # Current wheel rotation angle
        current_rotation = omega * time
        # Normalize to [0, 2*PI)
        current_rotation = current_rotation - wp.floor(current_rotation / TWO_PI) * TWO_PI

        # Particle angle in fixed frame
        particle_angle = wp.atan2(dz, dx)
        if particle_angle < 0.0:
            particle_angle = particle_angle + TWO_PI

        # Particle angle relative to rotating wheel
        rel_angle = particle_angle - current_rotation
        # Normalize to [0, 2*PI)
        rel_angle = rel_angle - wp.floor(rel_angle / TWO_PI) * TWO_PI

        # Find nearest blade index
        blade_idx = int(rel_angle / blade_angle_step + 0.5)
        nearest_blade_rel_angle = float(blade_idx) * blade_angle_step

        angle_to_blade = rel_angle - nearest_blade_rel_angle
        arc_dist = r * wp.abs(angle_to_blade)

        # Collision if particle center within half blade thickness of blade surface
        half_thickness = blade_thickness * 0.5
        if arc_dist < half_thickness:
            # Collision! Return blade normal (perpendicular to blade)
            # Blade is radial, so normal is tangential
            # Direction depends on which side of blade
            if angle_to_blade > 0.0:
                # Particle on CCW side of blade - push CW
                normal_angle = particle_angle - PI * 0.5
            else:
                # Particle on CW side of blade - push CCW
                normal_angle = particle_angle + PI * 0.5

            return wp.vec3(wp.cos(normal_angle), 0.0, wp.sin(normal_angle))

        return wp.vec3(0.0, 0.0, 0.0)

    @wp.func
    def compute_wheel_separation_force_ratio(
        d_p: float,
        rho_p: float,
        rho_f: float,
        mu_f: float,
        omega: float,
        r: float,
        v_radial: float,
    ) -> float:
        """
        Compute ratio of centrifugal to drag force for separation decision.

        SEPARATION PHYSICS:
        At the wheel periphery, particles experience:
        - Centrifugal force (outward): F_c = m*omega^2*r = (PI/6)*d^3*rho_p*omega^2*r
        - Drag force (inward from radial airflow): F_d = 3*PI*mu*d*v_r (Stokes)

        The force ratio F_c/F_d determines fate:
        - F_c/F_d > 1: Particle thrown outward -> COARSE (starch)
        - F_c/F_d < 1: Particle carried inward -> FINES (protein)

        Cut size (d50) where F_c = F_d:
        d50^2 = 18*mu*v_r / (Delta_rho*omega^2*r)
        d50 = sqrt(18*mu*v_r / (Delta_rho*omega^2*r))

        Args:
            d_p: Particle diameter [m]
            rho_p: Particle density [kg/m3]
            rho_f: Fluid density [kg/m3]
            mu_f: Fluid viscosity [Pa*s]
            omega: Angular velocity [rad/s]
            r: Radial position [m]
            v_radial: Radial air velocity (inward, negative) [m/s]

        Returns:
            Force ratio F_c/F_d (>1 = coarse, <1 = fines)
        """
        eps = 1.0e-10

        # Mass of particle
        m_p = PI / 6.0 * d_p * d_p * d_p * rho_p

        # Centrifugal force (outward)
        F_c = m_p * omega * omega * r

        # Drag force (Stokes, toward center)
        # F_d = 3*PI*mu*d*|v_r| for Stokes regime
        v_r_mag = wp.abs(v_radial)
        F_d = 3.0 * PI * mu_f * d_p * v_r_mag

        # Force ratio
        if F_d < eps:
            return 1000.0  # No drag: centrifugal dominates -> coarse

        return F_c / F_d

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
        # ZIGZAG GEOMETRY (with deflector plate parameters)
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
        # Deflector plate parameters (NEW)
        zigzag_plate_angle: float,        # Plate angle from vertical [rad]
        zigzag_plate_length: float,       # Plate length [m]
        zigzag_throat_width: float,       # Constriction width [m]
        zigzag_velocity_ratio_zone: float, # v_zone / v_bulk (from ZigzagClassifierParams)
        zigzag_recirculation_length_ratio: float,  # Separation zone length / plate length
        
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
        bagfilter_inlet_radius: float,   # Dirty air inlet radius [m] (from BagFilterParams)
        
        # =====================================================================
        # DUCT/CONNECTION GEOMETRY
        # =====================================================================
        duct_venturi_zigzag_start: wp.vec3,
        duct_venturi_zigzag_end: wp.vec3,
        duct_venturi_zigzag_radius: float,
        
        # Zigzag to cyclone path (includes 90 deg elbow)
        duct_zigzag_cyclone_start: wp.vec3,    # Fines outlet position
        duct_zigzag_cyclone_end: wp.vec3,      # Cyclone inlet position
        duct_zigzag_cyclone_radius: float,
        elbow_zigzag_cyclone_pos: wp.vec3,     # Position where elbow turns
        elbow_zigzag_cyclone_bend_radius: float,
        
        # Cyclone to bag filter path (includes 90 deg elbow)
        duct_cyclone_bag_start: wp.vec3,       # Cyclone overflow position
        duct_cyclone_bag_end: wp.vec3,         # Bag filter inlet position
        duct_cyclone_bag_radius: float,
        elbow_cyclone_bag_pos: wp.vec3,        # Position where elbow turns
        elbow_cyclone_bag_bend_radius: float,
        
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
        v_air_cyclone_inlet: float,  # Inlet velocity to primary cyclone
        v_air_cyclone_secondary_inlet: float,  # Inlet velocity to secondary cyclone
        v_air_cyclone_tertiary_inlet: float,   # Inlet velocity to tertiary cyclone

        # Cyclone cone tip ratios (dust_outlet_D / cyclone_D)
        cyclone_primary_cone_tip_ratio: float,
        cyclone_secondary_cone_tip_ratio: float,
        cyclone_tertiary_cone_tip_ratio: float,
        
        # Turbulence
        turbulent_intensity: float,

        # =====================================================================
        # WHEEL CLASSIFIER GEOMETRY (mandatory centrifugal separator)
        # =====================================================================
        wheel_enabled: int,                    # Always 1 (wheel classifier is mandatory)
        wheel_center: wp.vec3,                 # Center of wheel housing
        wheel_radius: float,                   # Outer radius of classifier wheel
        wheel_hub_radius: float,               # Inner hub radius (fines outlet)
        wheel_width: float,                    # Axial width of wheel (blade height)
        wheel_num_blades: int,                 # Number of radial blades
        wheel_blade_thickness: float,          # Blade thickness [m]
        wheel_omega: float,                    # Angular velocity [rad/s]
        wheel_housing_radius: float,           # Inner radius of volute housing
        wheel_hopper_y_bottom: float,          # Bottom of coarse hopper [m]
        wheel_hopper_half_angle_rad: float,    # Hopper cone half-angle [rad]
        wheel_coarse_outlet_radius: float,    # Coarse outlet radius [m] (geometry)
        wheel_fines_outlet_y: float,           # Y position of fines outlet
        wheel_coarse_outlet_y: float,          # Y position of coarse outlet
        wheel_inlet_y: float,                  # Y position of feed inlet
          wheel_volumetric_flow: float,          # Air flow through wheel [m3/s]

        # Random seed for turbulent dispersion
        random_seed: int,

        # Current simulation time (for rotating blade collision)
        sim_time: float,
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
           - Particles with v_terminal < v_air -> rise -> FINES (protein)
           - Particles with v_terminal > v_air -> fall -> COARSE (starch)
           - Cut size d50 = sqrt(18*mu*v_air / (g*(rho_p-rho_f)))
        
        3. CYCLONE SEPARATION (centrifugal)
           - Tangential inlet creates swirling flow
           - Centrifugal force: F_c = m*v_tan^2/r -> pushes particles OUT
           - Drag force: F_d -> pushes particles IN (toward vortex finder)
           - Large particles -> wall -> dust outlet
           - Small particles -> vortex finder -> next stage
        
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
            
            # Transition: throat when in throat region; skip to divergent if already past throat; exit if past venturi
            if local_y >= venturi_throat_start and local_y <= venturi_throat_end:
                zone = 1
            elif local_y > venturi_throat_end:
                zone = 2  # Already past throat (e.g. spawned at solids inlet downstream of throat)
            if local_y > venturi_total_length - particle_radius:
                zone = 10  # Already past venturi outlet
        
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
        # ZONE 10: DUCT - VENTURI TO ZIGZAG (includes round-to-rect transition)
        # From assembly:
        #   RoundDuct: pos=(0, 0.322, 0) D=72mm, L=36mm (venturi outlet)
        #   Transition: pos=(0, 0.363, 0) round D=72mm -> rect 120x200mm, L=301mm
        #   -> Zigzag air_inlet at (0, 0.669, 0)
        # =====================================================================
        elif zone == 10:
            # Flow direction is +Y (upward)
            # Cross-section transitions from circular to rectangular
            
            # Progress along duct (0 = venturi outlet, 1 = zigzag inlet)
            duct_dy = duct_venturi_zigzag_end[1] - duct_venturi_zigzag_start[1]
            progress = (pos[1] - duct_venturi_zigzag_start[1]) / (duct_dy + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            # Center at this height (may shift slightly)
            center_x = duct_venturi_zigzag_start[0] + progress * (duct_venturi_zigzag_end[0] - duct_venturi_zigzag_start[0])
            center_z = duct_venturi_zigzag_start[2] + progress * (duct_venturi_zigzag_end[2] - duct_venturi_zigzag_start[2])
            
            dx = pos[0] - center_x
            dz = pos[2] - center_z
            
            # Air velocity from continuity: Q = v * A = const
            # Duct transitions from circular (D=72mm) to rectangular (120x200mm)
            # so velocity is higher in the narrow round section
            A_zigzag = zigzag_channel_width * zigzag_channel_depth
            if progress < 0.1:
                # Round duct: A = pi * r^2
                A_local = PI * duct_venturi_zigzag_radius * duct_venturi_zigzag_radius
            else:
                # Transition section: morphing cross-section
                trans_p = (progress - 0.1) / 0.9
                trans_p = wp.clamp(trans_p, 0.0, 1.0)
                hw = duct_venturi_zigzag_radius + trans_p * (zigzag_channel_width / 2.0 - duct_venturi_zigzag_radius)
                hd = duct_venturi_zigzag_radius + trans_p * (zigzag_channel_depth / 2.0 - duct_venturi_zigzag_radius)
                A_local = 4.0 * hw * hd
            v_air_duct = v_air_zigzag * (A_zigzag / wp.max(A_local, 1.0e-6))
            v_air = wp.vec3(0.0, v_air_duct, 0.0)

            # Transition from circular to rectangular cross-section
            # First ~10% is round duct, then transition to rectangular
            if progress < 0.1:
                # Circular section (D=72mm -> radius=36mm)
                r = wp.sqrt(dx * dx + dz * dz)
                local_radius = duct_venturi_zigzag_radius
                
                if r + particle_radius > local_radius:
                    if r > 1.0e-6:
                        normal = wp.vec3(-dx / r, 0.0, -dz / r)
                        push = r + particle_radius - local_radius + 0.001
                        pos = pos + normal * push
                        vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            else:
                # Transition to rectangular (morphing from circle to rect)
                # At end: 120mm x 200mm (half_w=60mm, half_d=100mm)
                trans_progress = (progress - 0.1) / 0.9
                trans_progress = wp.clamp(trans_progress, 0.0, 1.0)
                
                # Dimensions morph from circular to rectangular
                # Start: radius ~36mm (equivalent to ~64mm sides)
                # End: 120mm x 200mm (60mm x 100mm half-dimensions)
                half_w = duct_venturi_zigzag_radius + trans_progress * (zigzag_channel_width / 2.0 - duct_venturi_zigzag_radius)
                half_d = duct_venturi_zigzag_radius + trans_progress * (zigzag_channel_depth / 2.0 - duct_venturi_zigzag_radius)
                
                # Rectangular containment (with rounded corners early in transition)
                corner_blend = 1.0 - trans_progress  # 1.0=circular, 0.0=rectangular
                
                # X walls
                if dx + particle_radius > half_w:
                    pos = wp.vec3(center_x + half_w - particle_radius - 0.001, pos[1], pos[2])
                    vel = reflect_velocity_inelastic(vel, wp.vec3(-1.0, 0.0, 0.0), restitution, friction)
                elif dx - particle_radius < -half_w:
                    pos = wp.vec3(center_x - half_w + particle_radius + 0.001, pos[1], pos[2])
                    vel = reflect_velocity_inelastic(vel, wp.vec3(1.0, 0.0, 0.0), restitution, friction)
                
                # Z walls
                if dz + particle_radius > half_d:
                    pos = wp.vec3(pos[0], pos[1], center_z + half_d - particle_radius - 0.001)
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, -1.0), restitution, friction)
                elif dz - particle_radius < -half_d:
                    pos = wp.vec3(pos[0], pos[1], center_z - half_d + particle_radius + 0.001)
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, 1.0), restitution, friction)
            
            # Terminal velocity: can this particle be carried at this point in the duct?
            # v_air_duct is local (high in round section, drops to v_air_zigzag at zigzag).
            # If v_t > local v_air_duct and particle is not moving up, it cannot progress
            # and will oscillate or fall - send to coarse so the duct drains.
            v_t = compute_terminal_velocity(d, rho_p, rho_f, mu_f, gravity)
            if v_t > v_air_duct and vel[1] <= 0.0:
                zone = 30  # Coarse (cannot be carried at this duct cross-section)
            # Transition to zigzag when reaching inlet
            elif pos[1] >= zigzag_inlet_y - particle_radius * 2.0:
                zone = 20  # Enter zigzag
            # Particles that fall back below duct start (very heavy, v_t > v_air_round)
            elif pos[1] < duct_venturi_zigzag_start[1] - particle_radius:
                zone = 30  # Coarse (fell through duct)
            # Fallback: particles falling in lower duct (progress < 0.15) with
            # downward velocity - pre-classified by duct expansion.
            elif progress < 0.15 and vel[1] < -0.01:
                zone = 30  # Coarse (pre-classified by duct expansion)
        
        # =====================================================================
        # ZONE 20-21: ZIGZAG CLASSIFIER (PRIMARY SEPARATION)
        # Note: Zone 22 (fines) and 23 (coarse) are handled separately for transitions
        # =====================================================================
        elif zone == 20 or zone == 21:
            # ZIGZAG SEPARATION PHYSICS:
            # - Air flows upward at v_air_zigzag
            # - Gravity pulls particles down
            # - Terminal velocity determines fate:
            #   v_t < v_air -> particle rises (fines/protein)
            #   v_t > v_air -> particle falls (coarse/starch)
            
            local_y = pos[1] - zigzag_center[1]
            local_x = pos[0] - zigzag_center[0]
            local_z = pos[2] - zigzag_center[2]
            
            # Compute air velocity with deflector plate physics
            v_air = compute_zigzag_air_velocity(
                pos, zigzag_center,
                zigzag_channel_width, zigzag_total_height, zigzag_num_stages,
                v_air_zigzag, zigzag_stage_height,
                zigzag_plate_angle, zigzag_plate_length, zigzag_throat_width,
                zigzag_velocity_ratio_zone, zigzag_recirculation_length_ratio
            )

            # Add turbulent dispersion (essential for realistic separation)
            # Use ZONE velocity as reference so turbulence doesn't vanish
            # for slow particles near d50 (those need it most)
            v_zone_ref = v_air_zigzag * zigzag_velocity_ratio_zone
            v_turb = compute_turbulent_dispersion(v_zone_ref, turbulent_intensity, random_seed, tid)
            v_air = v_air + v_turb

            # CAP upward velocity at zone velocity for proper separation.
            # In zigzag classifiers, the effective velocity that determines the
            # cut size (d50) is the zone velocity, not bulk or throat velocity.
            # Without this cap, the high bulk/throat velocities lift all particles
            # upward and prevent coarse classification. The zone velocity is the
            # time-averaged effective velocity experienced by particles as they
            # bounce between deflector plates through recirculation zones.
            v_zone_max = v_air_zigzag * zigzag_velocity_ratio_zone
            if v_air[1] > v_zone_max:
                v_air = wp.vec3(v_air[0], v_zone_max, v_air[2])

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

            # =========================================================
            # DEFLECTOR PLATE COLLISION DETECTION
            # =========================================================
            # Plates alternate from left/right walls at each stage
            # Plate at stage n: base on wall, tip extends into channel
            if zigzag_stage_height > 0.0 and zigzag_plate_length > 0.0:
                stage_num = int(local_y / zigzag_stage_height)
                pos_in_stage = local_y - float(stage_num) * zigzag_stage_height

                # Plate is at vertical center of stage
                plate_y_local = 0.5 * zigzag_stage_height
                plate_on_left = (stage_num % 2) == 0

                # Plate geometry
                plate_horizontal = zigzag_plate_length * wp.sin(zigzag_plate_angle)
                plate_vertical = zigzag_plate_length * wp.cos(zigzag_plate_angle)

                # Check if particle is near plate height
                dy_from_plate_base = pos_in_stage - plate_y_local
                if dy_from_plate_base >= -particle_radius and dy_from_plate_base <= plate_vertical + particle_radius:
                    # Particle is at plate height - check lateral collision
                    # Plate extends from wall into channel
                    if plate_on_left:
                        # Plate from left wall (-half_w) extending right
                        plate_tip_x = -half_w + plate_horizontal
                        # Check if particle is to the left of plate tip and overlapping plate
                        if local_x < plate_tip_x + particle_radius:
                            # Distance along plate direction
                            t_along = dy_from_plate_base / (plate_vertical + 1e-6)
                            t_along = wp.clamp(t_along, 0.0, 1.0)
                            plate_x_at_y = -half_w + t_along * plate_horizontal

                            if local_x < plate_x_at_y + particle_radius:
                                # Collision with plate - reflect off plate surface
                                # Plate normal points into channel (+X, -Y direction)
                                nx = wp.cos(zigzag_plate_angle)
                                ny = -wp.sin(zigzag_plate_angle)
                                normal = wp.normalize(wp.vec3(nx, ny, 0.0))

                                # Push particle away from plate
                                push_dist = particle_radius + 0.001
                                pos = wp.vec3(
                                    zigzag_center[0] + plate_x_at_y + push_dist * nx,
                                    pos[1] + push_dist * ny,
                                    pos[2]
                                )
                                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
                    else:
                        # Plate from right wall (+half_w) extending left
                        plate_tip_x = half_w - plate_horizontal
                        # Check if particle is to the right of plate tip and overlapping plate
                        if local_x > plate_tip_x - particle_radius:
                            t_along = dy_from_plate_base / (plate_vertical + 1e-6)
                            t_along = wp.clamp(t_along, 0.0, 1.0)
                            plate_x_at_y = half_w - t_along * plate_horizontal

                            if local_x > plate_x_at_y - particle_radius:
                                # Collision with plate - reflect off plate surface
                                # Plate normal points into channel (-X, -Y direction)
                                nx = -wp.cos(zigzag_plate_angle)
                                ny = -wp.sin(zigzag_plate_angle)
                                normal = wp.normalize(wp.vec3(nx, ny, 0.0))

                                # Push particle away from plate
                                push_dist = particle_radius + 0.001
                                pos = wp.vec3(
                                    zigzag_center[0] + plate_x_at_y + push_dist * nx,
                                    pos[1] + push_dist * ny,
                                    pos[2]
                                )
                                vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

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
        # ZONE 34: WHEEL CLASSIFIER HOUSING (Annular chamber around wheel)
        # Centrifugal separation: F_c vs F_d determines fines/coarse
        # =====================================================================
        elif zone == 34:
            # WHEEL CLASSIFIER SEPARATION PHYSICS:
            # - Rotating cage wheel creates centrifugal force on particles
            # - Air flows radially inward through blade gaps (drag force)
            # - F_c = m*omega^2*r (outward, proportional to d^3)
            # - F_d = 3*PI*mu*d*v_r (inward, proportional to d)
            # - Small particles (F_d > F_c): carried inward -> FINES (protein)
            # - Large particles (F_c > F_d): thrown outward -> COARSE (starch)

            # Position relative to wheel center (XZ plane)
            dx = pos[0] - wheel_center[0]
            dz = pos[2] - wheel_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - wheel_center[1]

            # Compute air velocity field in wheel housing
            # 1. Tangential component from rotating wheel
            v_tan = compute_wheel_tangential_velocity(
                pos, wheel_center, wheel_radius, wheel_hub_radius, wheel_omega
            )

            # 2. Radial component (inward through blade gaps; area from blade geometry)
            v_radial_mag = compute_wheel_radial_velocity(
                pos, wheel_center, wheel_radius, wheel_hub_radius,
                wheel_volumetric_flow, wheel_width,
                wheel_num_blades, wheel_blade_thickness
            )
            # Radial direction (toward center)
            eps = 1.0e-6
            if r > eps:
                radial_dir = wp.vec3(dx / r, 0.0, dz / r)
            else:
                radial_dir = wp.vec3(1.0, 0.0, 0.0)
            v_rad = radial_dir * v_radial_mag

            # Total air velocity
            v_air = v_tan + v_rad

            # Check blade collision (rotating blades)
            blade_normal = check_wheel_blade_collision(
                pos, wheel_center, wheel_num_blades, wheel_blade_thickness,
                wheel_radius, wheel_hub_radius, wheel_omega, sim_time
            )
            if wp.length(blade_normal) > 0.5:
                # Collision with blade - reflect velocity
                vel = reflect_velocity_inelastic(vel, blade_normal, restitution, friction)
                # Push away from blade
                pos = pos + blade_normal * (particle_radius + 0.001)

            # Wall containment - volute housing (outer boundary)
            if r + particle_radius > wheel_housing_radius:
                if r > eps:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wheel_housing_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            # Top/bottom shroud containment
            wheel_half_width = wheel_width * 0.5
            if local_y > wheel_half_width - particle_radius:
                pos = wp.vec3(pos[0], wheel_center[1] + wheel_half_width - particle_radius - 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, -1.0, 0.0), restitution, friction)
            elif local_y < -wheel_half_width + particle_radius:
                pos = wp.vec3(pos[0], wheel_center[1] - wheel_half_width + particle_radius + 0.001, pos[2])
                vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 1.0, 0.0), restitution, friction)

            # SEPARATION DECISION: force balance F_c vs F_d (no magic thresholds)
            # F_c/F_d > 1: centrifugal wins -> COARSE; F_c/F_d < 1: drag wins -> FINES
            force_ratio = compute_wheel_separation_force_ratio(
                d, rho_p, rho_f, mu_f, wheel_omega, r, v_radial_mag
            )

            if r <= wheel_hub_radius:
                # Geometrically inside hub -> FINES (protein)
                zone = 35
            elif r >= wheel_radius and force_ratio > 1.0:
                # At or beyond wheel rim and F_c > F_d -> COARSE (starch)
                zone = 36

        # =====================================================================
        # ZONE 35: WHEEL FINES OUTLET (Through hub to cyclones)
        # =====================================================================
        elif zone == 35:
            # Particle passed through wheel hub -> moving to fines outlet
            # Axial flow upward through hub center

            # Air velocity: upward through fines outlet
            v_fines = wheel_volumetric_flow / (PI * wheel_hub_radius * wheel_hub_radius + 1.0e-6)
            v_air = wp.vec3(0.0, v_fines, 0.0)

            # Hub wall containment (cylindrical)
            dx = pos[0] - wheel_center[0]
            dz = pos[2] - wheel_center[2]
            r = wp.sqrt(dx * dx + dz * dz)

            if r + particle_radius > wheel_hub_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wheel_hub_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            # Exit to cyclone path when reaching fines outlet
            if pos[1] > wheel_fines_outlet_y:
                zone = 40  # Enter elbow toward cyclones
                # Teleport to elbow inlet position
                pos = wp.vec3(
                    elbow_zigzag_cyclone_pos[0],
                    elbow_zigzag_cyclone_pos[1] - 0.01,
                    elbow_zigzag_cyclone_pos[2]
                )

        # =====================================================================
        # ZONE 36: WHEEL COARSE HOPPER (Gravity settling in conical hopper)
        # =====================================================================
        elif zone == 36:
            # Particle rejected by wheel, settling in conical hopper below

            # Position relative to hopper center (same X-Z as wheel, Y is below)
            dx = pos[0] - wheel_center[0]
            dz = pos[2] - wheel_center[2]
            r_xz = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - wheel_center[1]

            # No air flow in hopper (gravity settling)
            v_air = wp.vec3(0.0, 0.0, 0.0)

            # Conical wall containment
            # Hopper tapers from wheel_radius at wheel bottom to coarse outlet
            # Local radius at current height (below wheel center)
            wheel_bottom_y = -wheel_width * 0.5
            hopper_height = wheel_bottom_y - wheel_hopper_y_bottom
            if hopper_height > 0.01:
                # Progress through hopper (0 = wheel bottom, 1 = outlet)
                y_from_wheel_bottom = wheel_bottom_y - local_y
                hopper_progress = y_from_wheel_bottom / hopper_height
                hopper_progress = wp.clamp(hopper_progress, 0.0, 1.0)

                # Cone geometry: top = housing radius, bottom = coarse outlet radius
                hopper_top_radius = wheel_housing_radius
                hopper_bottom_radius = hopper_top_radius - hopper_height * wp.tan(wheel_hopper_half_angle_rad)
                hopper_bottom_radius = wp.max(hopper_bottom_radius, wheel_coarse_outlet_radius)

                local_wall_radius = hopper_top_radius - hopper_progress * (hopper_top_radius - hopper_bottom_radius)

                # Conical wall collision
                if r_xz + particle_radius > local_wall_radius:
                    if r_xz > 1.0e-6:
                        # Normal points inward and upward (cone surface normal)
                        cone_angle = wheel_hopper_half_angle_rad
                        n_r = wp.cos(cone_angle)
                        n_y = wp.sin(cone_angle)
                        normal = wp.vec3(-dx / r_xz * n_r, n_y, -dz / r_xz * n_r)
                        normal = wp.normalize(normal)
                        push = r_xz + particle_radius - local_wall_radius + 0.001
                        pos = pos + normal * push
                        vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            # Collection at coarse outlet
            if pos[1] < wheel_coarse_outlet_y + particle_radius:
                zone = 37  # Collected at coarse outlet

        # =====================================================================
        # ZONE 37: WHEEL COARSE COLLECTED (at coarse outlet)
        # =====================================================================
        elif zone == 37:
            # Particle collected at wheel coarse outlet - deactivate
            is_active[tid] = 0

        # =====================================================================
        # ZONE 22: FINES PATH - VERTICAL TRANSITION AFTER ZIGZAG
        # From assembly: Zigzag fines at (0.104, 1.689, 0) -> Transition -> Elbow at (0.104, 1.849, 0)
        # OR if wheel classifier enabled: -> Wheel classifier inlet
        # =====================================================================
        elif zone == 22:
            # Particle moving up from zigzag fines outlet toward elbow (or wheel)
            # Vertical duct section: fines outlet -> elbow inlet (or wheel inlet)

            local_x = pos[0] - duct_zigzag_cyclone_start[0]
            local_z = pos[2] - duct_zigzag_cyclone_start[2]
            r = wp.sqrt(local_x * local_x + local_z * local_z)

            # Air velocity is upward in this vertical section
            v_air = wp.vec3(0.0, v_air_cyclone_inlet * 0.8, 0.0)

            # Radial containment
            if r + particle_radius > duct_zigzag_cyclone_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(-local_x / r, 0.0, -local_z / r)
                    push = r + particle_radius - duct_zigzag_cyclone_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            # Transition depends on whether wheel classifier is enabled
            if wheel_enabled == 1:
                # Route to wheel classifier
                if pos[1] >= wheel_inlet_y - particle_radius * 2.0:
                    zone = 34  # Enter wheel classifier housing
                    # Teleport to wheel housing tangential inlet
                    pos = wp.vec3(
                        wheel_center[0] + wheel_housing_radius - particle_radius * 2.0,
                        wheel_center[1],
                        wheel_center[2]
                    )
                    # Tangential inlet velocity (tangent to housing)
                    vel = wp.vec3(0.0, 0.0, v_air_cyclone_inlet * 0.5)
            else:
                # Original path: transition to elbow when reaching elbow height
                if pos[1] >= elbow_zigzag_cyclone_pos[1] - particle_radius * 2.0:
                    zone = 40  # Enter elbow
        
        # =====================================================================
        # ZONE 40: 90 deg ELBOW - ZIGZAG TO CYCLONE (turns from +Y to +X)
        # From assembly: Elbow at (0.104, 1.849, 0) D=119.7mm, 90 deg, R=179.5mm
        # =====================================================================
        elif zone == 40:
            # In 90 deg elbow turning from vertical (+Y) to horizontal (+X)
            # Model as curved path with centripetal acceleration
            
            # Elbow center is offset from inlet by bend radius in X direction
            elbow_center_x = elbow_zigzag_cyclone_pos[0] + elbow_zigzag_cyclone_bend_radius
            elbow_center_y = elbow_zigzag_cyclone_pos[1]
            elbow_center_z = elbow_zigzag_cyclone_pos[2]
            
            # Position relative to elbow center
            dx = pos[0] - elbow_center_x
            dy = pos[1] - elbow_center_y
            
            # Angle in elbow (0 = inlet from below, 90 deg = exit to right)
            angle = wp.atan2(dy, -dx)  # Angle from -X axis
            
            # Air velocity follows the elbow curve
            # Tangential direction changes through the elbow
            v_tan_mag = v_air_cyclone_inlet
            v_air = wp.vec3(
                v_tan_mag * wp.sin(angle),   # X component
                v_tan_mag * wp.cos(angle),   # Y component
                0.0
            )
            
            # Distance from elbow centerline
            r_from_axis = wp.sqrt(dx * dx + dy * dy)
            dz = pos[2] - elbow_center_z
            
            # Containment to elbow cross-section
            # Check radial distance from bend path
            r_deviation = wp.abs(r_from_axis - elbow_zigzag_cyclone_bend_radius)
            cross_r = wp.sqrt(r_deviation * r_deviation + dz * dz)
            
            if cross_r + particle_radius > duct_zigzag_cyclone_radius:
                if cross_r > 1.0e-6:
                    # Push toward centerline
                    normal_r = (r_from_axis - elbow_zigzag_cyclone_bend_radius) / (r_deviation + 1.0e-6)
                    normal = wp.vec3(
                        -normal_r * dx / (r_from_axis + 1.0e-6),
                        -normal_r * dy / (r_from_axis + 1.0e-6),
                        -dz / (cross_r + 1.0e-6)
                    )
                    push = cross_r + particle_radius - duct_zigzag_cyclone_radius + 0.001
                    pos = pos + wp.normalize(normal) * push
                    vel = reflect_velocity_inelastic(vel, wp.normalize(normal), restitution, friction)
            
            # Transition to horizontal duct when angle > ~80 deg (exiting elbow)
            if angle > 1.4 or pos[0] > elbow_zigzag_cyclone_pos[0] + elbow_zigzag_cyclone_bend_radius * 0.7:
                zone = 41  # Enter horizontal duct to cyclones
        
        # =====================================================================
        # ZONE 41: DUCT - ZIGZAG TO CYCLONE (horizontal, includes round-to-rect transition)
        # From assembly:
        #   After elbow: RoundDuct at (0.288, 2.029, 0) D=119.7mm L=150mm
        #   Then: Transition at (0.443, 2.029, 0) round D=119.7mm -> rect 150x75mm L=100mm
        #   -> Cyclone inlet at (0.548, 2.029, 0) W=75mm H=150mm
        # =====================================================================
        elif zone == 41:
            # Horizontal duct toward primary cyclone inlet
            # Flow direction is +X
            
            # Progress along horizontal section (from elbow exit to cyclone inlet)
            x_start = elbow_zigzag_cyclone_pos[0] + elbow_zigzag_cyclone_bend_radius
            dx_duct = duct_zigzag_cyclone_end[0] - x_start
            progress = (pos[0] - x_start) / (dx_duct + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            # Center at this X position (Y is constant at cyclone inlet height)
            center_y = duct_zigzag_cyclone_end[1]
            center_z = duct_zigzag_cyclone_start[2] + progress * (duct_zigzag_cyclone_end[2] - duct_zigzag_cyclone_start[2])
            
            dy = pos[1] - center_y
            dz = pos[2] - center_z
            
            # Air velocity (horizontal toward cyclone)
            v_air = wp.vec3(v_air_cyclone_inlet, 0.0, 0.0)
            
            # Cross-section: Round duct -> Round-to-rect transition -> Rectangular inlet
            # First ~60% is round (D=119.7mm), then transition to rect (75x150mm)
            if progress < 0.6:
                # Circular section
                r = wp.sqrt(dy * dy + dz * dz)
                
                if r + particle_radius > duct_zigzag_cyclone_radius:
                    if r > 1.0e-6:
                        normal = wp.vec3(0.0, -dy / r, -dz / r)
                        push = r + particle_radius - duct_zigzag_cyclone_radius + 0.001
                        pos = pos + normal * push
                        vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            else:
                # Transition to rectangular (morph from circle to rect)
                trans_progress = (progress - 0.6) / 0.4
                trans_progress = wp.clamp(trans_progress, 0.0, 1.0)
                
                # Cyclone inlet is 75mm (W) x 150mm (H) -> half_w=37.5mm, half_h=75mm
                # Note: W is in Z direction (depth), H is in Y direction (height)
                cyclone_inlet_half_w = 0.0375  # 37.5mm
                cyclone_inlet_half_h = 0.075   # 75mm
                
                # Morph dimensions
                half_y = duct_zigzag_cyclone_radius + trans_progress * (cyclone_inlet_half_h - duct_zigzag_cyclone_radius)
                half_z = duct_zigzag_cyclone_radius + trans_progress * (cyclone_inlet_half_w - duct_zigzag_cyclone_radius)
                
                # Rectangular containment
                if dy + particle_radius > half_y:
                    pos = wp.vec3(pos[0], center_y + half_y - particle_radius - 0.001, pos[2])
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, -1.0, 0.0), restitution, friction)
                elif dy - particle_radius < -half_y:
                    pos = wp.vec3(pos[0], center_y - half_y + particle_radius + 0.001, pos[2])
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 1.0, 0.0), restitution, friction)
                
                if dz + particle_radius > half_z:
                    pos = wp.vec3(pos[0], pos[1], center_z + half_z - particle_radius - 0.001)
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, -1.0), restitution, friction)
                elif dz - particle_radius < -half_z:
                    pos = wp.vec3(pos[0], pos[1], center_z - half_z + particle_radius + 0.001)
                    vel = reflect_velocity_inelastic(vel, wp.vec3(0.0, 0.0, 1.0), restitution, friction)
            
            # Transition to primary cyclone when reaching inlet
            if pos[0] >= duct_zigzag_cyclone_end[0] - particle_radius * 2.0:
                zone = 50  # Enter primary cyclone
                # Teleport to cyclone tangential inlet (at wall, top of cylinder)
                # Without this, particles are placed OUTSIDE the cyclone body
                # (duct end ~0.445m vs cyclone center ~0.685m, R=0.15m)
                # and wall containment slams them to the wall immediately.
                pos = wp.vec3(
                    cyclone_primary_center[0] + cyclone_primary_radius - particle_radius * 2.0,
                    cyclone_primary_center[1],
                    cyclone_primary_center[2]
                )
                # Tangential inlet velocity (CCW viewed from above)
                # At position (+R, 0, 0) from center, tangential direction is (0, 0, +1)
                vel = wp.vec3(0.0, -0.2 * v_air_cyclone_inlet, v_air_cyclone_inlet)
        
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
                pos, cyclone_primary_center, v_air_cyclone_inlet, cyclone_primary_radius,
                cyclone_primary_vf_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_primary_center, cyclone_primary_radius, cyclone_primary_vf_radius,
                cyclone_primary_cylinder_height, cyclone_primary_cone_height, v_air_cyclone_inlet
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_primary_center, cyclone_primary_radius, cyclone_primary_vf_radius,
                cyclone_primary_cylinder_height, cyclone_primary_cone_height,
                v_air_cyclone_inlet
            )
            
            v_air = v_tan + v_rad + v_axial

            # Wall containment (cylinder + cone)
            total_height = cyclone_primary_cylinder_height + cyclone_primary_cone_height
            
            # Local radius (cylinder or cone)
            if local_y >= -cyclone_primary_cylinder_height:
                # In cylinder section
                wall_r = cyclone_primary_radius
            else:
                # In cone section - radius decreases linearly to dust outlet
                cone_progress = (-local_y - cyclone_primary_cylinder_height) / cyclone_primary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                # Cone tapers from cylinder radius to tip (dust_outlet_D/2)
                # tip_ratio = dust_outlet_D / D, so tip_r = tip_ratio * R
                wall_r = cyclone_primary_radius * (1.0 - (1.0 - cyclone_primary_cone_tip_ratio) * cone_progress)

            # Radial containment
            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            # SEPARATION: geometry-based (CycloneGeometryParams)
            # In core = inside vortex finder radius (inner vortex -> overflow)
            # At wall = at cyclone wall (spirals to dust outlet)
            in_core = r <= cyclone_primary_vf_radius
            at_wall = r >= wall_r * 0.99
            below_cylinder = local_y < -cyclone_primary_cylinder_height
            above_vf = local_y > 0.0

            if at_wall and below_cylinder:
                zone = 55  # Collected in primary dust outlet
            elif in_core and above_vf:
                zone = 51  # Move to secondary cyclone
                pos = wp.vec3(
                    cyclone_secondary_center[0] + cyclone_secondary_radius - particle_radius * 2.0,
                    cyclone_secondary_center[1],
                    cyclone_secondary_center[2]
                )
                vel = wp.vec3(0.0, -0.2 * v_air_cyclone_secondary_inlet, v_air_cyclone_secondary_inlet)
        
        # =====================================================================
        # ZONE 51: SECONDARY CYCLONE (medium fines)
        # =====================================================================
        elif zone == 51:
            dx = pos[0] - cyclone_secondary_center[0]
            dz = pos[2] - cyclone_secondary_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - cyclone_secondary_center[1]

            # Cyclone velocity field (computed from actual secondary inlet velocity)
            v_tan = compute_cyclone_tangential_velocity(
                pos, cyclone_secondary_center, v_air_cyclone_secondary_inlet, cyclone_secondary_radius,
                cyclone_secondary_vf_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_secondary_center, cyclone_secondary_radius, cyclone_secondary_vf_radius,
                cyclone_secondary_cylinder_height, cyclone_secondary_cone_height, v_air_cyclone_secondary_inlet
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_secondary_center, cyclone_secondary_radius, cyclone_secondary_vf_radius,
                cyclone_secondary_cylinder_height, cyclone_secondary_cone_height,
                v_air_cyclone_secondary_inlet
            )

            v_air = v_tan + v_rad + v_axial

            # Wall containment
            if local_y >= -cyclone_secondary_cylinder_height:
                wall_r = cyclone_secondary_radius
            else:
                cone_progress = (-local_y - cyclone_secondary_cylinder_height) / cyclone_secondary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                wall_r = cyclone_secondary_radius * (1.0 - (1.0 - cyclone_secondary_cone_tip_ratio) * cone_progress)

            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            in_core = r <= cyclone_secondary_vf_radius
            at_wall = r >= wall_r * 0.99
            below_cylinder = local_y < -cyclone_secondary_cylinder_height
            above_vf = local_y > 0.0

            if at_wall and below_cylinder:
                zone = 56  # Collected in secondary dust outlet
            elif in_core and above_vf:
                zone = 52  # Move to tertiary cyclone
                # Teleport to tertiary cyclone tangential inlet
                pos = wp.vec3(
                    cyclone_tertiary_center[0] + cyclone_tertiary_radius - particle_radius * 2.0,
                    cyclone_tertiary_center[1],
                    cyclone_tertiary_center[2]
                )
                vel = wp.vec3(0.0, -0.2 * v_air_cyclone_tertiary_inlet, v_air_cyclone_tertiary_inlet)
        
        # =====================================================================
        # ZONE 52: TERTIARY CYCLONE (fine protein)
        # =====================================================================
        elif zone == 52:
            dx = pos[0] - cyclone_tertiary_center[0]
            dz = pos[2] - cyclone_tertiary_center[2]
            r = wp.sqrt(dx * dx + dz * dz)
            local_y = pos[1] - cyclone_tertiary_center[1]

            # Cyclone velocity field (smallest cyclone, highest inlet velocity)
            v_tan = compute_cyclone_tangential_velocity(
                pos, cyclone_tertiary_center, v_air_cyclone_tertiary_inlet, cyclone_tertiary_radius,
                cyclone_tertiary_vf_radius
            )
            v_rad = compute_cyclone_radial_velocity(
                pos, cyclone_tertiary_center, cyclone_tertiary_radius, cyclone_tertiary_vf_radius,
                cyclone_tertiary_cylinder_height, cyclone_tertiary_cone_height, v_air_cyclone_tertiary_inlet
            )
            v_axial = compute_cyclone_axial_velocity(
                pos, cyclone_tertiary_center, cyclone_tertiary_radius, cyclone_tertiary_vf_radius,
                cyclone_tertiary_cylinder_height, cyclone_tertiary_cone_height,
                v_air_cyclone_tertiary_inlet
            )

            v_air = v_tan + v_rad + v_axial

            # Wall containment
            if local_y >= -cyclone_tertiary_cylinder_height:
                wall_r = cyclone_tertiary_radius
            else:
                cone_progress = (-local_y - cyclone_tertiary_cylinder_height) / cyclone_tertiary_cone_height
                cone_progress = wp.clamp(cone_progress, 0.0, 1.0)
                wall_r = cyclone_tertiary_radius * (1.0 - (1.0 - cyclone_tertiary_cone_tip_ratio) * cone_progress)

            if r + particle_radius > wall_r:
                if r > 1.0e-6:
                    normal = wp.vec3(-dx / r, 0.0, -dz / r)
                    push = r + particle_radius - wall_r + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)

            in_core = r <= cyclone_tertiary_vf_radius
            at_wall = r >= wall_r * 0.99
            below_cylinder = local_y < -cyclone_tertiary_cylinder_height
            above_vf = local_y > 0.0

            if at_wall and below_cylinder:
                zone = 57  # Collected in tertiary dust outlet (fine protein)
            elif in_core and above_vf:
                zone = 60  # Move to bag filter path
                # Teleport to elbow inlet (bag filter path)
                pos = wp.vec3(
                    elbow_cyclone_bag_pos[0],
                    elbow_cyclone_bag_pos[1],
                    elbow_cyclone_bag_pos[2]
                )
                vel = wp.vec3(0.0, v_air_cyclone_inlet * 0.5, 0.0)
        
        # =====================================================================
        # ZONES 55-57: CYCLONE DUST OUTLETS (collected)
        # =====================================================================
        elif zone == 55 or zone == 56 or zone == 57:
            # Particle collected in cyclone dust outlet
            # Keep zone for statistics, just deactivate
            is_active[tid] = 0
        
        # =====================================================================
        # ZONE 60: 90 deg ELBOW - CYCLONE OVERFLOW TO BAG FILTER (turns from +Y to +X)
        # From assembly: Elbow at (2.148, 2.209, 0) D=60mm, 90 deg, R=90mm
        # =====================================================================
        elif zone == 60:
            # In 90 deg elbow turning from vertical (+Y) to horizontal (+X)
            
            # Elbow center is offset from inlet by bend radius in X direction
            elbow_center_x = elbow_cyclone_bag_pos[0] + elbow_cyclone_bag_bend_radius
            elbow_center_y = elbow_cyclone_bag_pos[1]
            elbow_center_z = elbow_cyclone_bag_pos[2]
            
            # Position relative to elbow center
            dx = pos[0] - elbow_center_x
            dy = pos[1] - elbow_center_y
            
            # Angle in elbow
            angle = wp.atan2(dy, -dx)
            
            # Air velocity follows the elbow curve
            v_tan_mag = v_air_cyclone_inlet * 0.5
            v_air = wp.vec3(
                v_tan_mag * wp.sin(angle),
                v_tan_mag * wp.cos(angle),
                0.0
            )
            
            # Distance from elbow centerline
            r_from_axis = wp.sqrt(dx * dx + dy * dy)
            dz = pos[2] - elbow_center_z
            
            # Containment to elbow cross-section
            r_deviation = wp.abs(r_from_axis - elbow_cyclone_bag_bend_radius)
            cross_r = wp.sqrt(r_deviation * r_deviation + dz * dz)
            
            if cross_r + particle_radius > duct_cyclone_bag_radius:
                if cross_r > 1.0e-6:
                    normal_r = (r_from_axis - elbow_cyclone_bag_bend_radius) / (r_deviation + 1.0e-6)
                    normal = wp.vec3(
                        -normal_r * dx / (r_from_axis + 1.0e-6),
                        -normal_r * dy / (r_from_axis + 1.0e-6),
                        -dz / (cross_r + 1.0e-6)
                    )
                    push = cross_r + particle_radius - duct_cyclone_bag_radius + 0.001
                    pos = pos + wp.normalize(normal) * push
                    vel = reflect_velocity_inelastic(vel, wp.normalize(normal), restitution, friction)
            
            # Transition to horizontal duct when exiting elbow
            if angle > 1.4 or pos[0] > elbow_cyclone_bag_pos[0] + elbow_cyclone_bag_bend_radius * 0.7:
                zone = 61  # Enter horizontal duct to bag filter
        
        # =====================================================================
        # ZONE 61: HORIZONTAL DUCT - CYCLONE TO BAG FILTER
        # From assembly: Duct at (2.243, 2.299, 0) D=60mm L=100mm -> Expansion to D=300mm L=565mm
        # =====================================================================
        elif zone == 61:
            # Horizontal duct from elbow exit toward bag filter inlet
            # Includes expansion transition from 60mm to 300mm
            
            # Progress along horizontal duct (X direction)
            dx_duct = duct_cyclone_bag_end[0] - elbow_cyclone_bag_pos[0] - elbow_cyclone_bag_bend_radius
            x_start = elbow_cyclone_bag_pos[0] + elbow_cyclone_bag_bend_radius
            progress = (pos[0] - x_start) / (dx_duct + 0.001)
            progress = wp.clamp(progress, 0.0, 1.0)
            
            # Center at this X position (constant Y and Z for horizontal duct)
            center_y = duct_cyclone_bag_end[1]  # Constant height
            center_z = duct_cyclone_bag_end[2]
            
            # Duct expands from elbow radius to bag filter inlet radius (geometry)
            if progress < 0.15:
                local_radius = duct_cyclone_bag_radius
            else:
                expansion_progress = (progress - 0.15) / 0.85
                expansion_progress = wp.clamp(expansion_progress, 0.0, 1.0)
                local_radius = duct_cyclone_bag_radius + expansion_progress * (bagfilter_inlet_radius - duct_cyclone_bag_radius)
            
            dy = pos[1] - center_y
            dz = pos[2] - center_z
            r = wp.sqrt(dy * dy + dz * dz)
            
            # Air velocity (horizontal toward bag filter)
            v_air = wp.vec3(v_air_cyclone_inlet * 0.5, 0.0, 0.0)
            
            # Radial containment with expanding radius
            if r + particle_radius > local_radius:
                if r > 1.0e-6:
                    normal = wp.vec3(0.0, -dy / r, -dz / r)
                    push = r + particle_radius - local_radius + 0.001
                    pos = pos + normal * push
                    vel = reflect_velocity_inelastic(vel, normal, restitution, friction)
            
            # Transition to bag filter when reaching inlet
            if pos[0] >= duct_cyclone_bag_end[0] - particle_radius * 2.0:
                zone = 70  # Enter bag filter
        
        # =====================================================================
        # ZONE 70: BAG FILTER (final fines capture)
        # =====================================================================
        elif zone == 70:
            # BAG FILTER: Capture on entry.
            # Real bag filters capture >99.9% of particles via inertial
            # impaction, interception, and diffusion on the filter media.
            # Simulating internal trajectories is unnecessary - any particle
            # that reaches the bag filter is effectively collected.
            zone = 75
        
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
            
            # In Lagrangian particle tracking (lab frame), centrifugal effect
            # emerges naturally from drag redirecting particles toward curved
            # air streamlines. Do NOT add explicit centrifugal acceleration -
            # that would double-count the outward tendency and push all particles
            # to the wall regardless of size.
            accel = a_gravity + a_drag
            
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
        count_wheel_coarse: wp.array(dtype=wp.int32), # Zone 37: Wheel classifier coarse reject
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
        elif zone == 37:
            wp.atomic_add(count_wheel_coarse, 0, 1)
        elif zone == 55:
            wp.atomic_add(count_cyclone_1, 0, 1)
        elif zone == 56:
            wp.atomic_add(count_cyclone_2, 0, 1)
        elif zone == 57:
            wp.atomic_add(count_cyclone_3, 0, 1)
        elif zone == 75:
            wp.atomic_add(count_bagfilter, 0, 1)
        elif zone == 80 or zone == 99:
            wp.atomic_add(count_escaped, 0, 1)  # 80 = clean air exit, 99 = legacy exit
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
        - Want G(d_protein) -> high (protein goes to fines)
        - Want G(d_starch) -> low (starch goes to coarse)
        
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
       - Light particles (protein) rise -> fines outlet
       - Heavy particles (starch) fall -> coarse outlet
       
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
        
        # Operating-condition validation (zigzag/cyclone cut sizes vs flow)
        self._validation_result = None
        try:
            Q_total = self.config.air_flow_rate_m3s
            bypass = getattr(self.config, "bypass_ratio", 0.0)
            Q_class = Q_total * (1.0 - bypass)
            total_flow_m3_h = Q_total * 3600.0
            classification_flow_m3_h = Q_class * 3600.0
            rho = self.config.particle_density
            min_um = 5.0
            max_um = 100.0
            if self.config.material is not None and hasattr(
                self.config.material, "size_distribution"
            ):
                sd = self.config.material.size_distribution
                min_um = sd.d_min * 1e6
                max_um = sd.d_max * 1e6
            self._validation_result = assembly.validate_system_configuration(
                air_flow_m3_h=total_flow_m3_h,
                particle_density=rho,
                min_particle_um=min_um,
                max_particle_um=max_um,
                classification_flow_m3_h=classification_flow_m3_h if bypass > 0 else None,
                cyclone_flow_m3_h=total_flow_m3_h if bypass > 0 else None,
            )
            if not self._validation_result.get("valid", True):
                rec = self._validation_result.get("recommendation", "")
                msg = (rec[:80] + "...") if len(rec) > 80 else rec
                print(f"\n  WARNING: Operating conditions mismatch. Run with --validate for details. {msg}")
                # Bench-scale geometry (40 mm venturi, 200 mm wheel): sweet spot from analysis
                print(f"  Operating point hint: For this geometry, try wheel RPM 2000–4000 and blower "
                      f"400–600 RPM to get transport + selectivity without venturi choke or wheel overload.")
        except Exception:
            pass  # Validation is best-effort; do not block init

        print(f"\n  ClassificationFlowPhysicsSimulator initialized")
        print(f"    Device: {self.device}")
        print(f"    Max particles: {self.config.num_particles}")
        
        # Store flag for detailed output (can be disabled for batch runs)
        self._print_detailed_flow = True
    
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
        
        def get_geo_center(component_name: str, default):
            """Get center from geometry component."""
            comp = geo.get(component_name)
            if comp is None or comp.center is None:
                if default is None:
                    return None
                return np.array(default)
            return np.array(comp.center)
        
        # =====================================================================
        # VENTURI GEOMETRY
        # Axial position is measured from the AIR INLET (local_y = 0 at inlet,
        # local_y = total_length at outlet). Throat start/end are axial distances from inlet.
        # =====================================================================
        venturi_geo = geo.get('venturi')
        if venturi_geo is not None and venturi_geo.inlet_pos is not None:
            self.venturi_center = np.array(venturi_geo.inlet_pos, dtype=np.float64)
        else:
            self.venturi_center = get_geo_center('venturi', [0, 0, 0])
        self.venturi_inlet_diameter = get_geo_attr('venturi', 'inlet_diameter', 0.1)
        self.venturi_throat_diameter = get_geo_attr('venturi', 'throat_diameter', 0.05)
        self.venturi_outlet_diameter = get_geo_attr('venturi', 'outlet_diameter', 0.08)
        self.venturi_total_length = get_geo_attr('venturi', 'length', 0.3)
        # Throat axial bounds from component (distance from air inlet along axis)
        self.venturi_throat_start = get_geo_attr('venturi', 'throat_start', self.venturi_total_length * 0.3)
        self.venturi_throat_end = get_geo_attr('venturi', 'throat_end', self.venturi_total_length * 0.5)
        
        # Solids inlet position and direction (for feed->classification connection)
        # Wheel-only: use wheel_only_entry so particles spawn at solids chute / wheel inlet
        wheel_only = geo.get('wheel_only_entry')
        self.use_preclassification = (venturi_geo is not None)
        if wheel_only is not None:
            self.venturi_solids_inlet_pos = np.array(wheel_only['solids_inlet_pos'], dtype=np.float64)
            self.venturi_solids_inlet_radius = float(wheel_only.get('solids_inlet_radius', 0.025))
            # Direction from solids chute toward junction (downward at 15°)
            junc = np.array(wheel_only['junction'])
            self.venturi_solids_inlet_dir = junc - self.venturi_solids_inlet_pos
            norm = np.linalg.norm(self.venturi_solids_inlet_dir)
            if norm > 1e-9:
                self.venturi_solids_inlet_dir = self.venturi_solids_inlet_dir / norm
            else:
                self.venturi_solids_inlet_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            self.venturi_center = np.array(wheel_only['air_inlet_pos'], dtype=np.float64)
            self.venturi_total_length = float(np.linalg.norm(junc - self.venturi_center)) or 0.4
            self.venturi_inlet_diameter = 2.0 * self.venturi_solids_inlet_radius
            self.venturi_throat_diameter = self.venturi_inlet_diameter * 0.8
            self.venturi_outlet_diameter = self.venturi_inlet_diameter * 0.9
            self.venturi_throat_start = self.venturi_total_length * 0.2
            self.venturi_throat_end = self.venturi_total_length * 0.5
        else:
            solids_inlet = get_geo_attr('venturi', 'solids_inlet_pos', None)
            if solids_inlet is None:
                solids_inlet = self.venturi_center.copy()
            self.venturi_solids_inlet_pos = np.array(solids_inlet)
            solids_inlet_dir = get_geo_attr('venturi', 'solids_inlet_dir', None)
            if solids_inlet_dir is None:
                solids_inlet_dir = np.array([0.0, 1.0, 0.0])  # Default: +Y into venturi
            self.venturi_solids_inlet_dir = np.array(solids_inlet_dir, dtype=np.float64)
            self.venturi_solids_inlet_radius = get_geo_attr('venturi', 'solids_inlet_diameter', 0.05) / 2.0
        
        # =====================================================================
        # ZIGZAG GEOMETRY (using actual port positions from assembly)
        # From assembly inspection:
        #   Position: (0.000, 0.729, 0.000) m
        #   air_inlet: pos=(0.000, 0.669, 0.000) dir=(0, -1, 0)
        #   fines_outlet: pos=(0.104, 1.689, 0.000) dir=(0, 1, 0)
        #   coarse_outlet: pos=(0.000, 0.633, 0.000) dir=(0, -1, 0)
        #   Channel: 120mm x 200mm, 5 stages, total height 900mm
        # =====================================================================
        self.zigzag_center = get_geo_center('zigzag', [0, 0.5, 0])
        self.zigzag_channel_width = get_geo_attr('zigzag', 'channel_width', 0.12)  # 120mm
        self.zigzag_channel_depth = get_geo_attr('zigzag', 'channel_depth', 0.20)  # 200mm
        self.zigzag_total_height = get_geo_attr('zigzag', 'total_height', 0.90)    # 900mm
        if self.zigzag_total_height == 0:
            self.zigzag_total_height = get_geo_attr('zigzag', 'length', 0.90)
        self.zigzag_num_stages = get_geo_attr('zigzag', 'num_stages', 5)  # 5 stages
        self.zigzag_stage_height = self.zigzag_total_height / max(1, self.zigzag_num_stages)
        if not self.use_preclassification and geo.get('wheel_classifier') is not None:
            self.zigzag_center = np.array(geo['wheel_classifier']['position'])
            self.zigzag_total_height = 0.0
            self.zigzag_channel_width = 0.0
            self.zigzag_channel_depth = 0.0
            self.zigzag_num_stages = 0
            self.zigzag_stage_height = 0.0

        # NEW: Deflector plate parameters for proper separation physics
        self.zigzag_plate_angle = get_geo_attr('zigzag', 'plate_angle', np.radians(45))  # 45 deg default
        self.zigzag_plate_length = get_geo_attr('zigzag', 'plate_length', self.zigzag_channel_width * 0.5)
        self.zigzag_throat_width = get_geo_attr('zigzag', 'throat_width', self.zigzag_channel_width * 0.5)
        self.zigzag_blockage_ratio = get_geo_attr('zigzag', 'blockage_ratio', 0.5)
        self.zigzag_velocity_ratio_throat = get_geo_attr('zigzag', 'velocity_ratio_throat', 2.0)  # v_throat/v_bulk
        self.zigzag_velocity_ratio_zone = get_geo_attr('zigzag', 'velocity_ratio_in_zone', 0.3)  # v_zone/v_bulk
        self.zigzag_recirculation_length_ratio = get_geo_attr('zigzag', 'recirculation_length_ratio', 1.5)  # from ZigzagClassifierParams
        self.zigzag_turbulence_intensity = get_geo_attr('zigzag', 'turbulence_intensity_zigzag', 0.25)
        
        # Zigzag inlet/outlet positions - USE ACTUAL PORT POSITIONS from geometry
        # The inlet_pos and outlet_pos are world coordinates, not relative to center
        zigzag_geo = geo.get('zigzag')
        if zigzag_geo is not None:
            # Use actual port Y positions from extracted geometry
            if zigzag_geo.inlet_pos is not None:
                self.zigzag_inlet_y = float(zigzag_geo.inlet_pos[1])
            else:
                self.zigzag_inlet_y = self.zigzag_center[1] - self.zigzag_total_height / 2
            
            if zigzag_geo.fines_outlet_pos is not None:
                self.zigzag_fines_outlet_y = float(zigzag_geo.fines_outlet_pos[1])
            else:
                self.zigzag_fines_outlet_y = self.zigzag_center[1] + self.zigzag_total_height / 2
            
            if zigzag_geo.coarse_outlet_pos is not None:
                self.zigzag_coarse_outlet_y = float(zigzag_geo.coarse_outlet_pos[1])
            else:
                self.zigzag_coarse_outlet_y = self.zigzag_inlet_y - 0.05
        else:
            # Fallback calculations
            self.zigzag_inlet_y = self.zigzag_center[1] - self.zigzag_total_height / 2
            self.zigzag_fines_outlet_y = self.zigzag_center[1] + self.zigzag_total_height / 2
            self.zigzag_coarse_outlet_y = self.zigzag_inlet_y - 0.05
        
        # =====================================================================
        # CYCLONE GEOMETRY (from cyclone_stages in extracted geometry)
        # From assembly inspection:
        #   multi_cyclone position: (1.423, 2.154, 0.000) m
        #   Primary:   D=300mm, d50=40um, H=1200mm, dust at (0.788, 0.894, 0)
        #   Secondary: D=200mm, d50=20um, H=800mm,  dust at (1.588, 1.314, 0)
        #   Tertiary:  D=120mm, d50=10um, H=480mm,  dust at (2.148, 1.650, 0)
        #   inlet:    pos=(0.548, 2.029, 0.000) dir=(-1, 0, 0) W=75mm H=150mm
        #   overflow: pos=(2.148, 2.204, 0.000) dir=(0, 1, 0) D=60mm
        # =====================================================================
        cyclone_stages = geo.get('cyclone_stages', {})
        multi_cyclone_center = get_geo_center('multi_cyclone', [1.423, 2.154, 0])
        
        # Helper to get cyclone stage data
        def get_cyclone_stage(stage_name: str, defaults: dict):
            """Get cyclone stage parameters with fallbacks."""
            stage = cyclone_stages.get(stage_name, {})
            if stage:
                return {
                    'position': stage.get('position', defaults['position']),
                    'diameter': stage.get('diameter', defaults['diameter']),
                    'cylinder_height': stage.get('cylinder_height', defaults['cylinder_height']),
                    'cone_height': stage.get('cone_height', defaults['cone_height']),
                    'vortex_finder_diameter': stage.get('vortex_finder_diameter', defaults['vortex_finder_diameter']),
                    'dust_outlet_diameter': stage.get('dust_outlet_diameter', defaults.get('dust_outlet_diameter')),
                    'dust_outlet_pos': stage.get('dust_outlet_pos', defaults.get('dust_outlet_pos')),
                    'inlet_width': stage.get('inlet_width', defaults.get('inlet_width')),
                    'inlet_height': stage.get('inlet_height', defaults.get('inlet_height')),
                }
            return defaults
        
        # PRIMARY CYCLONE (D=300mm)
        # Stairmand proportions: inlet W=0.25D, H=0.5D, dust_outlet=0.375D
        primary_defaults = {
            'position': multi_cyclone_center,
            'diameter': 0.30,
            'cylinder_height': 0.30,  # ~300mm cylinder
            'cone_height': 0.90,      # ~900mm cone (total H=1200mm)
            'vortex_finder_diameter': 0.12,
            'dust_outlet_diameter': 0.30 * 0.375,  # Stairmand: 0.375D
            'inlet_width': 0.30 * 0.25,   # Stairmand: 0.25D = 75mm
            'inlet_height': 0.30 * 0.5,   # Stairmand: 0.5D = 150mm
        }
        primary = get_cyclone_stage('primary', primary_defaults)
        if cyclone_stages and 'position' in cyclone_stages.get('primary', {}):
            self.cyclone_primary_center = np.array(primary['position']) + multi_cyclone_center
        else:
            self.cyclone_primary_center = np.array(multi_cyclone_center)
        self.cyclone_primary_radius = primary['diameter'] / 2.0
        self.cyclone_primary_cylinder_height = primary['cylinder_height']
        self.cyclone_primary_cone_height = primary['cone_height']
        self.cyclone_primary_vf_radius = primary['vortex_finder_diameter'] / 2.0
        D_pri = primary['diameter']
        self.cyclone_primary_inlet_width = primary.get('inlet_width') or D_pri * 0.25
        self.cyclone_primary_inlet_height = primary.get('inlet_height') or D_pri * 0.5
        self.cyclone_primary_dust_outlet_diameter = primary.get('dust_outlet_diameter') or D_pri * 0.375
        self.cyclone_primary_cone_tip_ratio = self.cyclone_primary_dust_outlet_diameter / D_pri
        dust_pos = primary.get('dust_outlet_pos')
        if dust_pos is not None:
            self.cyclone_primary_dust_y = float(dust_pos[1])
        else:
            self.cyclone_primary_dust_y = self.cyclone_primary_center[1] - self.cyclone_primary_cylinder_height - self.cyclone_primary_cone_height

        # SECONDARY CYCLONE (D=200mm)
        secondary_defaults = {
            'position': multi_cyclone_center + np.array([0.4, 0, 0]),
            'diameter': 0.20,
            'cylinder_height': 0.20,
            'cone_height': 0.60,
            'vortex_finder_diameter': 0.08,
            'dust_outlet_diameter': 0.20 * 0.375,
            'inlet_width': 0.20 * 0.25,
            'inlet_height': 0.20 * 0.5,
        }
        secondary = get_cyclone_stage('secondary', secondary_defaults)
        if cyclone_stages and 'position' in cyclone_stages.get('secondary', {}):
            self.cyclone_secondary_center = np.array(secondary['position']) + multi_cyclone_center
        else:
            self.cyclone_secondary_center = np.array(multi_cyclone_center) + np.array([0.4, 0, 0])
        self.cyclone_secondary_radius = secondary['diameter'] / 2.0
        self.cyclone_secondary_cylinder_height = secondary['cylinder_height']
        self.cyclone_secondary_cone_height = secondary['cone_height']
        self.cyclone_secondary_vf_radius = secondary['vortex_finder_diameter'] / 2.0
        D_sec = secondary['diameter']
        self.cyclone_secondary_inlet_width = secondary.get('inlet_width') or D_sec * 0.25
        self.cyclone_secondary_inlet_height = secondary.get('inlet_height') or D_sec * 0.5
        self.cyclone_secondary_dust_outlet_diameter = secondary.get('dust_outlet_diameter') or D_sec * 0.375
        self.cyclone_secondary_cone_tip_ratio = self.cyclone_secondary_dust_outlet_diameter / D_sec
        dust_pos = secondary.get('dust_outlet_pos')
        if dust_pos is not None:
            self.cyclone_secondary_dust_y = float(dust_pos[1])
        else:
            self.cyclone_secondary_dust_y = self.cyclone_secondary_center[1] - self.cyclone_secondary_cylinder_height - self.cyclone_secondary_cone_height

        # TERTIARY CYCLONE (D=120mm)
        tertiary_defaults = {
            'position': multi_cyclone_center + np.array([0.725, 0, 0]),
            'diameter': 0.12,
            'cylinder_height': 0.12,
            'cone_height': 0.36,
            'vortex_finder_diameter': 0.05,
            'dust_outlet_diameter': 0.12 * 0.375,
            'inlet_width': 0.12 * 0.25,
            'inlet_height': 0.12 * 0.5,
        }
        tertiary = get_cyclone_stage('tertiary', tertiary_defaults)
        if cyclone_stages and 'position' in cyclone_stages.get('tertiary', {}):
            self.cyclone_tertiary_center = np.array(tertiary['position']) + multi_cyclone_center
        else:
            self.cyclone_tertiary_center = np.array(multi_cyclone_center) + np.array([0.725, 0, 0])
        self.cyclone_tertiary_radius = tertiary['diameter'] / 2.0
        self.cyclone_tertiary_cylinder_height = tertiary['cylinder_height']
        self.cyclone_tertiary_cone_height = tertiary['cone_height']
        self.cyclone_tertiary_vf_radius = tertiary['vortex_finder_diameter'] / 2.0
        D_ter = tertiary['diameter']
        self.cyclone_tertiary_inlet_width = tertiary.get('inlet_width') or D_ter * 0.25
        self.cyclone_tertiary_inlet_height = tertiary.get('inlet_height') or D_ter * 0.5
        self.cyclone_tertiary_dust_outlet_diameter = tertiary.get('dust_outlet_diameter') or D_ter * 0.375
        self.cyclone_tertiary_cone_tip_ratio = self.cyclone_tertiary_dust_outlet_diameter / D_ter
        dust_pos = tertiary.get('dust_outlet_pos')
        if dust_pos is not None:
            self.cyclone_tertiary_dust_y = float(dust_pos[1])
        else:
            self.cyclone_tertiary_dust_y = self.cyclone_tertiary_center[1] - self.cyclone_tertiary_cylinder_height - self.cyclone_tertiary_cone_height
        
        # =====================================================================
        # BAG FILTER
        # =====================================================================
        # Try both 'bag_filter' and 'bagfilter' keys for compatibility
        self.bagfilter_center = get_geo_center('bag_filter', None)
        if self.bagfilter_center is None:
            self.bagfilter_center = get_geo_center('bagfilter', [1.5, 0.5, 0])
        
        self.bagfilter_half_width = get_geo_attr('bag_filter', 'housing_width', None)
        if self.bagfilter_half_width is None:
            self.bagfilter_half_width = get_geo_attr('bagfilter', 'housing_width', 0.4)
        self.bagfilter_half_width = self.bagfilter_half_width / 2
        
        self.bagfilter_half_depth = get_geo_attr('bag_filter', 'housing_depth', None)
        if self.bagfilter_half_depth is None:
            self.bagfilter_half_depth = get_geo_attr('bagfilter', 'housing_depth', 0.4)
        self.bagfilter_half_depth = self.bagfilter_half_depth / 2
        
        self.bagfilter_height = get_geo_attr('bag_filter', 'housing_height', None)
        if self.bagfilter_height is None:
            self.bagfilter_height = get_geo_attr('bagfilter', 'housing_height', 1.0)
        
        # Inlet/outlet/dust Y from BagFilterParams ports (dirty_air_inlet, clean_air_outlet, dust_outlet)
        bag_filter_geo_for_ports = geo.get('bag_filter')
        if bag_filter_geo_for_ports is not None and bag_filter_geo_for_ports.inlet_pos is not None:
            self.bagfilter_inlet_y = float(bag_filter_geo_for_ports.inlet_pos[1])
        else:
            self.bagfilter_inlet_y = self.bagfilter_center[1]
        if bag_filter_geo_for_ports is not None and bag_filter_geo_for_ports.outlet_pos is not None:
            self.bagfilter_outlet_y = float(bag_filter_geo_for_ports.outlet_pos[1])
        else:
            self.bagfilter_outlet_y = self.bagfilter_center[1] + self.bagfilter_height / 2
        if bag_filter_geo_for_ports is not None and getattr(bag_filter_geo_for_ports, 'coarse_outlet_pos', None) is not None:
            self.bagfilter_dust_y = float(bag_filter_geo_for_ports.coarse_outlet_pos[1])
        else:
            self.bagfilter_dust_y = self.bagfilter_center[1] - self.bagfilter_height / 2
        
        # Bag filter dirty air inlet radius (duct expansion in zone 61; from BagFilterParams)
        self.bagfilter_inlet_radius = get_geo_attr('bag_filter', 'inlet_diameter', 0.3) / 2.0
        
        # =====================================================================
        # DUCT/CONNECTION GEOMETRY (using actual port positions)
        # =====================================================================
        connections = geo.get('connections', {})
        
        # Get actual port positions from geometry for accurate particle flow
        venturi_geo = geo.get('venturi')
        zigzag_geo = geo.get('zigzag')
        multi_cyclone_geo = geo.get('multi_cyclone')
        bag_filter_geo = geo.get('bag_filter')
        
        # Venturi to zigzag duct
        # From assembly: Venturi outlet at Y=0.317m, Zigzag air_inlet at Y=0.669m
        conn_v_z = connections.get('venturi_to_zigzag', {})
        
        # Use actual port positions if available (extract_geometry stores as 'start_pos'/'end_pos')
        if 'start_pos' in conn_v_z:
            self.duct_venturi_zigzag_start = np.array(conn_v_z['start_pos'])
        else:
            # Fallback: venturi outlet position
            self.duct_venturi_zigzag_start = venturi_geo.outlet_pos if venturi_geo else self.venturi_center + np.array([0, self.venturi_total_length, 0])
        
        if 'end_pos' in conn_v_z:
            self.duct_venturi_zigzag_end = np.array(conn_v_z['end_pos'])
        else:
            # Fallback: zigzag inlet position
            self.duct_venturi_zigzag_end = zigzag_geo.inlet_pos if zigzag_geo else np.array([self.zigzag_center[0], self.zigzag_inlet_y, self.zigzag_center[2]])
        
        # Use actual outlet diameter / 2 as radius
        if 'start_diameter' in conn_v_z:
            self.duct_venturi_zigzag_radius = conn_v_z['start_diameter'] / 2.0
        else:
            self.duct_venturi_zigzag_radius = 0.036  # 72mm diameter / 2
        
        # Zigzag to cyclone path (includes 90 deg elbow!)
        # From assembly: Fines outlet at (0.104, 1.689, 0) -> Elbow at (0.104, 1.849, 0) -> horizontal to cyclone inlet at (0.548, 2.029, 0)
        conn_z_c = connections.get('zigzag_to_cyclone', {})
        
        if 'start_pos' in conn_z_c:
            self.duct_zigzag_cyclone_start = np.array(conn_z_c['start_pos'])
        else:
            self.duct_zigzag_cyclone_start = zigzag_geo.fines_outlet_pos if zigzag_geo else np.array([self.zigzag_center[0], self.zigzag_fines_outlet_y, self.zigzag_center[2]])
        
        if 'end_pos' in conn_z_c:
            self.duct_zigzag_cyclone_end = np.array(conn_z_c['end_pos'])
        else:
            self.duct_zigzag_cyclone_end = multi_cyclone_geo.inlet_pos if multi_cyclone_geo else self.cyclone_primary_center
        
        # Compute elbow position (where vertical duct meets horizontal)
        # From assembly: Elbow at (0.104, 1.849, 0) with R=179.5mm
        # After elbow, horizontal duct is at Y=2.029m
        self.elbow_zigzag_cyclone_pos = np.array([
            self.duct_zigzag_cyclone_start[0],  # X same as fines outlet
            self.duct_zigzag_cyclone_end[1],    # Y at cyclone inlet height
            self.duct_zigzag_cyclone_start[2]   # Z same as fines outlet
        ])
        self.elbow_zigzag_cyclone_radius = 0.060  # ~120mm diameter / 2
        self.elbow_zigzag_cyclone_bend_radius = 0.180  # 180mm bend radius
        
        # Use actual dimensions
        if 'avg_radius' in conn_z_c:
            self.duct_zigzag_cyclone_radius = conn_z_c['avg_radius']
        else:
            self.duct_zigzag_cyclone_radius = 0.060  # ~120mm diameter duct
        
        # Cyclone to bag filter path (also includes 90 deg elbow!)
        # From assembly: Overflow at (2.148, 2.204, 0) -> Elbow at (2.148, 2.209, 0) -> horizontal to bag filter inlet at (2.918, 2.299, 0)
        conn_c_b = connections.get('cyclone_to_bagfilter', {})
        
        if 'start_pos' in conn_c_b:
            self.duct_cyclone_bag_start = np.array(conn_c_b['start_pos'])
        else:
            self.duct_cyclone_bag_start = multi_cyclone_geo.outlet_pos if multi_cyclone_geo else self.cyclone_tertiary_center + np.array([0, 0.1, 0])
        
        if 'end_pos' in conn_c_b:
            self.duct_cyclone_bag_end = np.array(conn_c_b['end_pos'])
        else:
            self.duct_cyclone_bag_end = bag_filter_geo.inlet_pos if bag_filter_geo else self.bagfilter_center
        
        # Elbow for cyclone to bag filter path
        self.elbow_cyclone_bag_pos = np.array([
            self.duct_cyclone_bag_start[0],  # X same as overflow
            self.duct_cyclone_bag_end[1],    # Y at bag filter inlet height
            self.duct_cyclone_bag_start[2]   # Z same as overflow
        ])
        self.elbow_cyclone_bag_radius = 0.030  # 60mm diameter / 2
        self.elbow_cyclone_bag_bend_radius = 0.090  # 90mm bend radius
        
        # Use actual dimensions
        if 'avg_radius' in conn_c_b:
            self.duct_cyclone_bag_radius = conn_c_b['avg_radius']
        else:
            self.duct_cyclone_bag_radius = 0.030  # 60mm diameter initially, expands to 300mm
        
        # =====================================================================
        # AIR VELOCITIES
        # =====================================================================
        # Compute from volumetric flow rate and cross-sectional areas
        #
        # BYPASS FLOW SPLIT:
        #   Q_total  = blower output (cfg.air_flow_rate_m3s)
        #   Q_class  = Q_total * (1 - bypass_ratio)  -> through venturi + zigzag
        #   Q_bypass = Q_total * bypass_ratio         -> bypass duct (around zigzag)
        #   Q_cyclone = Q_total                       -> cyclones (after merge)
        #
        Q_total = cfg.air_flow_rate_m3s
        bypass_ratio = cfg.bypass_ratio
        Q_class = Q_total * (1.0 - bypass_ratio)  # through venturi + zigzag
        Q_bypass = Q_total * bypass_ratio           # around classification
        self._Q_total = Q_total
        self._Q_class = Q_class
        self._Q_bypass = Q_bypass
        self._bypass_ratio = bypass_ratio

        if bypass_ratio > 0:
            print(f"\n  BYPASS FLOW SPLIT:")
            print(f"      Total (blower):    {Q_total * 3600:.0f} m3/h")
            print(f"      Classification:    {Q_class * 3600:.1f} m3/h ({(1-bypass_ratio)*100:.1f}%)")
            print(f"      Bypass:            {Q_bypass * 3600:.1f} m3/h ({bypass_ratio*100:.1f}%)")
            print(f"      Cyclone (merged):  {Q_total * 3600:.0f} m3/h")

        # Use Q_class for venturi + zigzag, Q_total for cyclones
        Q_air = Q_class

        # Venturi inlet velocity
        A_venturi_inlet = np.pi * (self.venturi_inlet_diameter / 2) ** 2
        self.v_air_venturi_inlet = Q_air / A_venturi_inlet

        # =====================================================================
        # VENTURI THROAT COMPRESSIBILITY / CHOKED FLOW CHECK
        # =====================================================================
        A_venturi_throat = np.pi * (self.venturi_throat_diameter / 2) ** 2
        self._A_venturi_throat = A_venturi_throat
        v_throat_requested = Q_air / A_venturi_throat if A_venturi_throat > 0 else 0.0

        # Speed of sound in air at ~20 C
        speed_of_sound = 343.0  # m/s
        mach_throat = v_throat_requested / speed_of_sound if speed_of_sound > 0 else 0.0

        # Discharge coefficient for well-designed venturi (ISO 5167)
        Cd_venturi = 0.985

        # Maximum (choked) flow: Mach 1 at throat
        Q_choked = A_venturi_throat * speed_of_sound * Cd_venturi  # m3/s
        self.venturi_Q_choked_m3s = Q_choked
        self.venturi_Q_choked_m3h = Q_choked * 3600.0
        self.venturi_mach_requested = mach_throat
        self.venturi_flow_limited = False

        # Venturi pressure drop (Bernoulli, incompressible):
        # dP = 0.5 * rho * (v_throat^2 - v_inlet^2) = 0.5 * rho * Q^2 * (1/A_t^2 - 1/A_i^2)
        # This is the K-factor for the system curve
        if A_venturi_throat > 0 and A_venturi_inlet > 0:
            self.venturi_k_factor = 0.5 * cfg.air_density * (1.0 / A_venturi_throat**2 - 1.0 / A_venturi_inlet**2)
        else:
            self.venturi_k_factor = 0.0

        if mach_throat >= 1.0:
            # CHOKED FLOW - cap at sonic limit
            Q_air = Q_choked
            self.venturi_flow_limited = True
            print(f"\n  *** VENTURI CHOKED FLOW ***")
            print(f"      Requested:       {cfg.air_flow_rate_m3s * 3600:.0f} m3/h")
            print(f"      Throat velocity: {v_throat_requested:.0f} m/s > speed of sound ({speed_of_sound:.0f} m/s)")
            print(f"      Mach number:     {mach_throat:.2f} (SUPERSONIC - impossible)")
            print(f"      Throat diameter: {self.venturi_throat_diameter*1000:.1f} mm")
            print(f"      Throat area:     {A_venturi_throat*1e6:.1f} mm^2")
            print(f"      Max choked flow: {Q_choked * 3600:.0f} m3/h (Ma=1, Cd={Cd_venturi})")
            print(f"      Flow CAPPED to:  {Q_air * 3600:.0f} m3/h")
            # Recalculate inlet velocity at capped flow
            self.v_air_venturi_inlet = Q_air / A_venturi_inlet
        elif mach_throat > 0.3:
            # Compressibility effects significant (>5% density change)
            print(f"\n  *** VENTURI COMPRESSIBILITY WARNING ***")
            print(f"      Throat velocity: {v_throat_requested:.0f} m/s")
            print(f"      Mach number:     {mach_throat:.2f} (>0.3 - compressible regime)")
            print(f"      Bernoulli (incompressible) approximation has >5% error")
            print(f"      Max choked flow: {Q_choked * 3600:.0f} m3/h")

        self.v_air_venturi_throat = Q_air / A_venturi_throat if A_venturi_throat > 0 else 0.0
        self.venturi_mach_actual = self.v_air_venturi_throat / speed_of_sound
        self.venturi_pressure_drop_Pa = 0.5 * cfg.air_density * (
            self.v_air_venturi_throat**2 - self.v_air_venturi_inlet**2
        )

        # Zigzag air velocity (upward) - uses potentially capped Q_air (skip when wheel-only)
        A_zigzag = max(self.zigzag_channel_width * self.zigzag_channel_depth, 1e-12)
        self.v_air_zigzag = Q_air / A_zigzag if self.use_preclassification else 0.0

        # Cyclone inlet velocities - uses Q_total (bypass merges back before cyclones)
        # Each cyclone has a RECTANGULAR tangential inlet (Stairmand: W=0.25D, H=0.5D)
        # In series arrangement, same Q flows through each stage (minus collected dust)
        # Velocity increases through series as cyclones get smaller
        A_pri = self.cyclone_primary_inlet_width * self.cyclone_primary_inlet_height
        A_sec = self.cyclone_secondary_inlet_width * self.cyclone_secondary_inlet_height
        A_ter = self.cyclone_tertiary_inlet_width * self.cyclone_tertiary_inlet_height
        self._A_cyclone_inlet = A_pri  # For backwards compatibility
        self.v_air_cyclone_inlet = Q_total / max(A_pri, 1e-6)
        self.v_air_cyclone_secondary_inlet = Q_total / max(A_sec, 1e-6)
        self.v_air_cyclone_tertiary_inlet = Q_total / max(A_ter, 1e-6)

        # =====================================================================
        # CUT SIZE CALCULATION
        # =====================================================================
        # IMPORTANT: Separation occurs in the ZONE (recirculation region), not bulk flow!
        # v_zone = v_bulk * velocity_ratio_zone (typically 0.3)
        # d50 = sqrt(18 * mu * v_zone / (g * (rho_p - rho_f)))
        g = 9.81
        rho_p = cfg.particle_density
        rho_f = cfg.air_density
        mu = cfg.air_viscosity

        # Zone velocity where separation actually occurs
        self.v_air_zigzag_zone = self.v_air_zigzag * self.zigzag_velocity_ratio_zone

        # d50 based on ZONE velocity (where separation happens), not bulk
        self.zigzag_d50 = np.sqrt(18 * mu * self.v_air_zigzag_zone / (g * (rho_p - rho_f)))

        # For reference: what d50 would be if using bulk velocity (wrong!)
        self.zigzag_d50_bulk = np.sqrt(18 * mu * self.v_air_zigzag / (g * (rho_p - rho_f)))

        # Cyclone d50 per stage (Lapple equation)
        # d50 = sqrt(9*mu*W / (2*pi*N*v_in*(rho_p-rho_f)))
        N_turns = 5  # Effective turns
        W_pri = self.cyclone_primary_inlet_width
        W_sec = self.cyclone_secondary_inlet_width
        W_ter = self.cyclone_tertiary_inlet_width
        self.cyclone_d50 = np.sqrt(9 * mu * W_pri / (2 * np.pi * N_turns * self.v_air_cyclone_inlet * (rho_p - rho_f)))
        self.cyclone_secondary_d50 = np.sqrt(9 * mu * W_sec / (2 * np.pi * N_turns * self.v_air_cyclone_secondary_inlet * (rho_p - rho_f)))
        self.cyclone_tertiary_d50 = np.sqrt(9 * mu * W_ter / (2 * np.pi * N_turns * self.v_air_cyclone_tertiary_inlet * (rho_p - rho_f)))

        # Cyclone inlet velocity validation
        # Below ~5 m/s: no vortex forms, cyclone acts as gravity settler, Lapple equation invalid
        # 5-15 m/s: weak vortex, Lapple d50 unreliable (actual d50 much coarser)
        # 15-25 m/s: proper vortex, Lapple d50 valid
        # >25 m/s: excessive pressure drop, possible re-entrainment
        self.cyclone_min_vortex_velocity = 5.0    # m/s - absolute minimum for any vortex
        self.cyclone_good_vortex_velocity = 15.0  # m/s - minimum for reliable Lapple d50
        self.cyclone_vortex_ok = self.v_air_cyclone_inlet >= self.cyclone_min_vortex_velocity
        self.cyclone_d50_reliable = self.v_air_cyclone_inlet >= self.cyclone_good_vortex_velocity

        # =====================================================================
        # WHEEL CLASSIFIER PARAMETERS (main classifier - centrifugal separation)
        # =====================================================================
        wheel_geo = geo['wheel_classifier']  # Wheel classifier is mandatory
        self.wheel_enabled = True  # Always enabled

        self.wheel_center = np.array(wheel_geo['position'])
        self.wheel_radius = wheel_geo['wheel_radius']
        self.wheel_hub_radius = wheel_geo['hub_radius']
        self.wheel_width = wheel_geo['wheel_width']
        self.wheel_num_blades = wheel_geo['num_blades']
        self.wheel_blade_thickness = wheel_geo['blade_thickness']
        # Config wheel_rpm overrides assembly RPM when set
        wheel_rpm = getattr(cfg, 'wheel_rpm', None)
        if wheel_rpm is not None and wheel_rpm > 0:
            self.wheel_omega = float(wheel_rpm) * 2.0 * np.pi / 60.0
        else:
            self.wheel_omega = wheel_geo['omega']
        self.wheel_housing_radius = wheel_geo['housing_radius']
        self.wheel_hopper_height = wheel_geo['hopper_height']
        self.wheel_hopper_half_angle = wheel_geo['hopper_half_angle']
        self.wheel_fines_outlet_diameter = wheel_geo['fines_outlet_diameter']
        self.wheel_coarse_outlet_diameter = wheel_geo['coarse_outlet_diameter']
        self.wheel_inlet_pos = wheel_geo['inlet_pos']
        self.wheel_fines_outlet_pos = wheel_geo['fines_outlet_pos']
        self.wheel_coarse_outlet_pos = wheel_geo['coarse_outlet_pos']

        # Air flow through wheel (uses classification flow, not total)
        self.wheel_volumetric_flow = Q_air

        # Wheel d50 from force balance (matches wheel_classifier.calculate_d50):
        # v_r through blade passage area; d50 = sqrt(18*mu*v_r / (Delta_rho*omega^2*r))
        blade_passage_area = (
            (2.0 * np.pi * self.wheel_radius - self.wheel_num_blades * self.wheel_blade_thickness)
            * self.wheel_width
        )
        blade_passage_area = max(blade_passage_area, 1.0e-12)
        v_radial = Q_air / blade_passage_area
        omega_sq = self.wheel_omega ** 2
        delta_rho = rho_p - rho_f
        self.wheel_d50 = np.sqrt(18.0 * mu * v_radial / (delta_rho * omega_sq * self.wheel_radius))

        # G-force at wheel rim
        self.wheel_g_force = omega_sq * self.wheel_radius / g

        # Tip speed
        self.wheel_tip_speed = self.wheel_omega * self.wheel_radius

        wheel_rpm_display = (float(getattr(cfg, 'wheel_rpm', 0)) or wheel_geo['rpm'])
        print(f"\n    Wheel Classifier (main classifier - centrifugal):")
        print(f"      Diameter:        {self.wheel_radius * 2 * 1000:.0f} mm")
        print(f"      RPM:             {wheel_rpm_display:.0f}")
        print(f"      Tip speed:       {self.wheel_tip_speed:.1f} m/s")
        print(f"      G-force (rim):   {self.wheel_g_force:.0f} g")
        print(f"      d50:             {self.wheel_d50 * 1e6:.1f} um")
        print(f"      Hub radius:      {self.wheel_hub_radius * 1000:.1f} mm")
        print(f"      Blades:          {self.wheel_num_blades}")

        # =====================================================================
        # SYSTEM BOUNDS
        # =====================================================================
        all_centers = [
            self.venturi_center,
            self.zigzag_center,
            self.wheel_center,  # Wheel classifier is mandatory
            self.cyclone_primary_center,
            self.cyclone_secondary_center,
            self.cyclone_tertiary_center,
            self.bagfilter_center,
        ]
        all_centers = np.array(all_centers)
        
        self.system_min = np.min(all_centers, axis=0) - np.array([1.0, 1.0, 1.0])
        self.system_max = np.max(all_centers, axis=0) + np.array([1.0, 2.0, 1.0])
        
        # =====================================================================
        # STORE COMPUTED VALUES FOR DETAILED FLOW PRINTING
        # =====================================================================
        self._Q_air = Q_air
        self._A_venturi_inlet = A_venturi_inlet
        self._A_zigzag = A_zigzag
        self._A_cyclone_inlet = A_pri
        self._g = g
        self._rho_p = rho_p
        self._rho_f = rho_f
        self._mu = mu
        
        # =====================================================================
        # PRINT SUMMARY
        # =====================================================================
        print(f"\n  Classification Physics Parameters:")
        print(f"\n    Air Flow:")
        if bypass_ratio > 0:
            print(f"      Total (blower):  {Q_total * 3600:.0f} m3/h")
            print(f"      Bypass:          {Q_bypass * 3600:.1f} m3/h ({bypass_ratio*100:.1f}% around zigzag)")
            if self.venturi_flow_limited:
                print(f"      Classification:  {Q_air * 3600:.0f} m3/h (CAPPED from {Q_class * 3600:.1f} m3/h - choked)")
            else:
                print(f"      Classification:  {Q_air * 3600:.1f} m3/h ({(1-bypass_ratio)*100:.1f}% through venturi+zigzag)")
            print(f"      Cyclone (merge): {Q_total * 3600:.0f} m3/h")
        elif self.venturi_flow_limited:
            print(f"      Flow rate:       {Q_air * 3600:.0f} m3/h (CAPPED from {cfg.air_flow_rate_m3s * 3600:.0f} m3/h - choked)")
        else:
            print(f"      Flow rate:       {Q_air * 3600:.0f} m3/h")
        print(f"      Venturi inlet:   {self.v_air_venturi_inlet:.1f} m/s")
        print(f"      Venturi throat:  {self.v_air_venturi_throat:.1f} m/s (D={self.venturi_throat_diameter*1000:.1f}mm, Ma={self.venturi_mach_actual:.3f})")
        print(f"      Venturi dP:      {self.venturi_pressure_drop_Pa:.0f} Pa ({self.venturi_pressure_drop_Pa/1000:.1f} kPa)")
        print(f"      Zigzag bulk:     {self.v_air_zigzag:.2f} m/s")
        print(f"      Zigzag ZONE:     {self.v_air_zigzag_zone:.2f} m/s ({self.zigzag_velocity_ratio_zone:.0%} of bulk)")
        print(f"      Cyclone (series, rectangular tangential inlet):")
        for label, v_in, D, W, H in [
            ("Primary",   self.v_air_cyclone_inlet,            self.cyclone_primary_radius*2,   self.cyclone_primary_inlet_width,   self.cyclone_primary_inlet_height),
            ("Secondary", self.v_air_cyclone_secondary_inlet,  self.cyclone_secondary_radius*2, self.cyclone_secondary_inlet_width, self.cyclone_secondary_inlet_height),
            ("Tertiary",  self.v_air_cyclone_tertiary_inlet,   self.cyclone_tertiary_radius*2,  self.cyclone_tertiary_inlet_width,  self.cyclone_tertiary_inlet_height),
        ]:
            status = ""
            if v_in < self.cyclone_min_vortex_velocity:
                status = " *** NO VORTEX ***"
            elif v_in < self.cyclone_good_vortex_velocity:
                status = " (weak vortex)"
            print(f"        {label:10s} D={D*1000:.0f}mm  inlet={W*1000:.0f}x{H*1000:.0f}mm  v={v_in:.1f} m/s{status}")

        print(f"\n    Venturi Throat Analysis:")
        print(f"      Throat diameter: {self.venturi_throat_diameter*1000:.1f} mm")
        print(f"      Throat area:     {self._A_venturi_throat*1e6:.1f} mm2")
        print(f"      Max flow (Ma=1): {self.venturi_Q_choked_m3h:.0f} m3/h")
        print(f"      K_venturi:       {self.venturi_k_factor:.1f} Pa/(m3/s)2")
        if self.venturi_mach_actual > 0.3:
            print(f"      *** Ma={self.venturi_mach_actual:.2f} > 0.3: compressible regime ***")
        if self.venturi_flow_limited:
            print(f"      *** FLOW CHOKED at venturi throat ***")

        print(f"\n    Cut Sizes (d50) - based on ZONE velocity:")
        print(f"      Zigzag:          {self.zigzag_d50 * 1e6:.1f} um (at v_zone={self.v_air_zigzag_zone:.2f} m/s)")
        print(f"      (if bulk):       {self.zigzag_d50_bulk * 1e6:.1f} um (wrong - ignores zone effect)")
        for label, d50, v_in in [
            ("Cy1 (primary)",   self.cyclone_d50,            self.v_air_cyclone_inlet),
            ("Cy2 (secondary)", self.cyclone_secondary_d50,  self.v_air_cyclone_secondary_inlet),
            ("Cy3 (tertiary)",  self.cyclone_tertiary_d50,   self.v_air_cyclone_tertiary_inlet),
        ]:
            if v_in >= self.cyclone_good_vortex_velocity:
                print(f"      {label:16s} {d50*1e6:.1f} um")
            elif v_in >= self.cyclone_min_vortex_velocity:
                print(f"      {label:16s} {d50*1e6:.1f} um (weak vortex)")
            else:
                print(f"      {label:16s} {d50*1e6:.1f} um *** NO VORTEX (v={v_in:.1f} m/s) ***")

        print(f"\n    Multi-Stage Sharpening ({self.zigzag_num_stages} stages):")
        print(f"      Each stage is a separation opportunity")
        print(f"      Effective cut sharpness increases with stages")

        print(f"\n    For protein separation:")
        print(f"      Protein:         ~10-30 um (should go to fines)")
        print(f"      Starch:          ~15-60 um (should go to coarse)")

        if self.zigzag_d50 * 1e6 < 35:
            print(f"      Status: Zigzag d50 ({self.zigzag_d50*1e6:.1f}um) in protein range - good!")
        elif self.zigzag_d50 * 1e6 < 60:
            print(f"      Status: Zigzag d50 ({self.zigzag_d50*1e6:.1f}um) in starch range - partial separation")
        else:
            print(f"      WARNING: Zigzag d50 ({self.zigzag_d50*1e6:.1f}um) > 60um - poor separation")
            print(f"               Consider: reduce channel size, increase air flow, or more stages")
    
    def _allocate_arrays(self):
        """Allocate particle arrays on device."""
        n = self.config.num_particles
        
        self.state.positions = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.state.diameters = wp.zeros(n, dtype=float, device=self.device)
        self.state.masses = wp.zeros(n, dtype=float, device=self.device)
        self.state.zones = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.state.is_active = wp.zeros(n, dtype=wp.int32, device=self.device)
        # Optional: per-particle type and density (for material/transfer)
        self.state.particle_types = wp.zeros(n, dtype=wp.int32, device=self.device)
        self.state.densities = wp.zeros(n, dtype=float, device=self.device)
    
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
        self._count_wheel_coarse = wp.zeros(1, dtype=wp.int32, device=self.device)  # Wheel classifier reject
        self._count_cyclone1 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone2 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_cyclone3 = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_bagfilter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_escaped = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count_active = wp.zeros(1, dtype=wp.int32, device=self.device)
    
    def print_detailed_flow_path(self):
        """
        Print detailed material flow path with all underlying computed calculations.
        
        Shows exact coordinates, velocities, areas, Reynolds numbers, and 
        separation physics for each flow segment.
        """
        print("\n" + "=" * 80)
        print("DETAILED MATERIAL FLOW PATH - CLASSIFICATION SYSTEM")
        print("=" * 80)
        
        Q = self._Q_air
        rho_f = self._rho_f
        rho_p = self._rho_p
        mu = self._mu
        g = self._g
        
        # Helper functions
        def compute_Re_circular(v, D):
            """Reynolds number for circular duct."""
            return rho_f * v * D / mu
        
        def compute_Re_rectangular(v, W, H):
            """Reynolds number for rectangular duct using hydraulic diameter."""
            D_h = 4 * W * H / (2 * (W + H))
            return rho_f * v * D_h / mu
        
        def terminal_velocity(d_p):
            """Terminal velocity for Stokes regime."""
            return (d_p ** 2 * (rho_p - rho_f) * g) / (18 * mu)
        
        def flow_regime(Re):
            """Determine flow regime."""
            if Re < 2300:
                return "LAMINAR"
            elif Re < 4000:
                return "TRANSITIONAL"
            else:
                return "TURBULENT"
        
        segment_num = 0
        
        # =====================================================================
        # SEGMENT 1: VENTURI EDUCTOR
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: VENTURI EDUCTOR (Particle Entrainment)")
        print(f"{'-' * 80}")
        
        print(f"\n  GEOMETRY:")
        print(f"    Center position:     ({self.venturi_center[0]*1000:.1f}, {self.venturi_center[1]*1000:.1f}, {self.venturi_center[2]*1000:.1f}) mm")
        print(f"    Inlet diameter:      {self.venturi_inlet_diameter*1000:.1f} mm")
        print(f"    Throat diameter:     {self.venturi_throat_diameter*1000:.1f} mm")
        print(f"    Outlet diameter:     {self.venturi_outlet_diameter*1000:.1f} mm")
        print(f"    Total length:        {self.venturi_total_length*1000:.1f} mm")
        print(f"    Throat region:       Y = {self.venturi_throat_start*1000:.1f} to {self.venturi_throat_end*1000:.1f} mm")
        
        print(f"\n  SOLIDS INLET (Feed entry point):")
        print(f"    Position:            ({self.venturi_solids_inlet_pos[0]*1000:.1f}, {self.venturi_solids_inlet_pos[1]*1000:.1f}, {self.venturi_solids_inlet_pos[2]*1000:.1f}) mm")
        print(f"    Diameter:            {self.venturi_solids_inlet_radius*2*1000:.1f} mm")
        print(f"    Direction:           Angled into throat (Coanda effect)")
        
        print(f"\n  FLOW CALCULATIONS:")
        A_inlet = np.pi * (self.venturi_inlet_diameter/2)**2
        A_throat = np.pi * (self.venturi_throat_diameter/2)**2
        A_outlet = np.pi * (self.venturi_outlet_diameter/2)**2
        v_inlet = Q / A_inlet
        v_throat = Q / A_throat
        v_outlet = Q / A_outlet
        
        print(f"    Volumetric flow Q:   {Q*1000:.2f} L/s = {Q*3600:.0f} m3/h")
        print(f"    Inlet  (A={A_inlet*1e4:.2f} cm2):  v = Q/A = {v_inlet:.2f} m/s")
        print(f"    Throat (A={A_throat*1e4:.2f} cm2):  v = Q/A = {v_throat:.2f} m/s")
        print(f"    Outlet (A={A_outlet*1e4:.2f} cm2):  v = Q/A = {v_outlet:.2f} m/s")
        
        Re_inlet = compute_Re_circular(v_inlet, self.venturi_inlet_diameter)
        Re_throat = compute_Re_circular(v_throat, self.venturi_throat_diameter)
        print(f"\n  REYNOLDS NUMBERS:")
        print(f"    Re_inlet  = rho*v*D/mu = {Re_inlet:.0f} ({flow_regime(Re_inlet)})")
        print(f"    Re_throat = rho*v*D/mu = {Re_throat:.0f} ({flow_regime(Re_throat)})")
        
        print(f"\n  BERNOULLI PRESSURE DROP AT THROAT:")
        delta_P = 0.5 * rho_f * (v_throat**2 - v_inlet**2)
        print(f"    dP = 0.5*rho*(v2_throat - v2_inlet)")
        print(f"       = 0.5 x {rho_f} x ({v_throat:.2f}^2 - {v_inlet:.2f}^2)")
        print(f"       = {delta_P:.1f} Pa = {delta_P/1000:.3f} kPa")
        print(f"    This suction draws particles from solids inlet")

        print(f"\n  COMPRESSIBILITY CHECK (throat):")
        speed_of_sound = 343.0
        Ma = v_throat / speed_of_sound
        print(f"    v_throat:            {v_throat:.1f} m/s")
        print(f"    Speed of sound:      {speed_of_sound:.0f} m/s (air at ~20C)")
        print(f"    Mach number:         {Ma:.3f}")
        if Ma >= 1.0:
            Q_choked = A_throat * speed_of_sound * 0.985
            print(f"    *** CHOKED FLOW (Ma >= 1) ***")
            print(f"    Max physical flow:   {Q_choked*3600:.0f} m3/h")
            print(f"    Requested flow:      {Q*3600:.0f} m3/h (EXCEEDS LIMIT)")
        elif Ma > 0.3:
            print(f"    *** COMPRESSIBLE REGIME (Ma > 0.3) ***")
            print(f"    Incompressible Bernoulli has >5% error")
            rho_ratio = (1 + 0.2 * Ma**2)**(-2.5)  # isentropic density ratio
            print(f"    Isentropic rho_throat/rho_inlet: {rho_ratio:.3f}")
        else:
            print(f"    Incompressible regime (Ma < 0.3) - Bernoulli valid")

        print(f"\n  SYSTEM CURVE CONTRIBUTION:")
        print(f"    K_venturi = 0.5*rho*(1/A_t^2 - 1/A_i^2)")
        print(f"              = {self.venturi_k_factor:.1f} Pa/(m3/s)^2")
        print(f"    dP at Q   = K*Q^2 = {self.venturi_k_factor * Q**2:.0f} Pa")

        print(f"\n  PARTICLE ENTRY (Zone 0 -> 1 -> 2):")
        print(f"    Zone 0: Solids inlet -> entering throat")
        print(f"    Zone 1: Throat region (high velocity entrainment)")
        print(f"    Zone 2: Divergent section -> outlet")
        print(f"    Exit at Y = {self.venturi_total_length*1000:.1f} mm -> Zone 10")
        
        # =====================================================================
        # SEGMENT 2: VENTURI -> ZIGZAG DUCT
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: DUCT - VENTURI TO ZIGZAG (Round -> Rectangular Transition)")
        print(f"{'-' * 80}")
        
        print(f"\n  PATH COORDINATES:")
        print(f"    Start (venturi outlet): ({self.duct_venturi_zigzag_start[0]*1000:.1f}, {self.duct_venturi_zigzag_start[1]*1000:.1f}, {self.duct_venturi_zigzag_start[2]*1000:.1f}) mm")
        print(f"    End (zigzag inlet):     ({self.duct_venturi_zigzag_end[0]*1000:.1f}, {self.duct_venturi_zigzag_end[1]*1000:.1f}, {self.duct_venturi_zigzag_end[2]*1000:.1f}) mm")
        
        duct_length_vz = np.linalg.norm(self.duct_venturi_zigzag_end - self.duct_venturi_zigzag_start)
        print(f"    Path length:            {duct_length_vz*1000:.1f} mm")
        print(f"    Flow direction:         +Y (upward)")
        
        print(f"\n  CROSS-SECTION TRANSITION:")
        print(f"    Start: Circular D = {self.duct_venturi_zigzag_radius*2*1000:.1f} mm (A = {np.pi*(self.duct_venturi_zigzag_radius)**2*1e4:.2f} cm2)")
        print(f"    End:   Rectangular {self.zigzag_channel_width*1000:.0f} x {self.zigzag_channel_depth*1000:.0f} mm (A = {self._A_zigzag*1e4:.2f} cm2)")
        
        v_duct_start = Q / (np.pi * self.duct_venturi_zigzag_radius**2)
        v_duct_end = Q / self._A_zigzag
        print(f"\n  VELOCITY CHANGE (continuity):")
        print(f"    Start: v = Q/A = {v_duct_start:.2f} m/s")
        print(f"    End:   v = Q/A = {v_duct_end:.2f} m/s")
        
        print(f"\n  PARTICLE ZONE: Zone 10")
        print(f"    Transition to Zone 20 at Y = {self.zigzag_inlet_y*1000:.1f} mm")
        
        # =====================================================================
        # SEGMENT 3: ZIGZAG CLASSIFIER
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: ZIGZAG CLASSIFIER (Primary Separation)")
        print(f"{'-' * 80}")
        
        print(f"\n  GEOMETRY:")
        print(f"    Center position:     ({self.zigzag_center[0]*1000:.1f}, {self.zigzag_center[1]*1000:.1f}, {self.zigzag_center[2]*1000:.1f}) mm")
        print(f"    Channel width:       {self.zigzag_channel_width*1000:.0f} mm")
        print(f"    Channel depth:       {self.zigzag_channel_depth*1000:.0f} mm")
        print(f"    Number of stages:    {self.zigzag_num_stages}")
        print(f"    Stage height:        {self.zigzag_stage_height*1000:.1f} mm")
        print(f"    Total height:        {self.zigzag_total_height*1000:.0f} mm")

        print(f"\n  DEFLECTOR PLATE GEOMETRY:")
        print(f"    Plate angle:         {np.degrees(self.zigzag_plate_angle):.0f} deg from vertical")
        print(f"    Plate length:        {self.zigzag_plate_length*1000:.1f} mm")
        print(f"    Throat width:        {self.zigzag_throat_width*1000:.1f} mm ({100*self.zigzag_throat_width/self.zigzag_channel_width:.0f}% open)")
        print(f"    Blockage ratio:      {self.zigzag_blockage_ratio*100:.0f}%")
        print(f"    Throat velocity:     {self.zigzag_velocity_ratio_throat:.1f}x bulk")
        print(f"    Zone velocity:       {self.zigzag_velocity_ratio_zone:.0%} of bulk (separation)")
        print(f"    Turbulence (zone):   {self.zigzag_turbulence_intensity:.0%} intensity")
        
        print(f"\n  PORT POSITIONS (World Coordinates):")
        print(f"    Air inlet (bottom):  Y = {self.zigzag_inlet_y*1000:.1f} mm, dir = (0, -1, 0)")
        print(f"    Fines outlet (top):  Y = {self.zigzag_fines_outlet_y*1000:.1f} mm, dir = (0, +1, 0)")
        print(f"    Coarse outlet:       Y = {self.zigzag_coarse_outlet_y*1000:.1f} mm, dir = (0, -1, 0)")
        
        print(f"\n  FLOW CALCULATIONS:")
        print(f"    Cross-section A:     {self._A_zigzag*1e4:.2f} cm2 = {self._A_zigzag*1e6:.0f} mm2")
        print(f"    Bulk air velocity:   v_bulk = Q/A = {self.v_air_zigzag:.3f} m/s")
        print(f"    Zone air velocity:   v_zone = v_bulk x {self.zigzag_velocity_ratio_zone:.2f} = {self.v_air_zigzag_zone:.3f} m/s")
        print(f"    Throat air velocity: v_throat = v_bulk x {self.zigzag_velocity_ratio_throat:.2f} = {self.v_air_zigzag * self.zigzag_velocity_ratio_throat:.3f} m/s")

        Re_zigzag = compute_Re_rectangular(self.v_air_zigzag, self.zigzag_channel_width, self.zigzag_channel_depth)
        print(f"    Re (hydraulic):      {Re_zigzag:.0f} ({flow_regime(Re_zigzag)})")

        print(f"\n  SEPARATION PHYSICS (Counter-current classification):")
        print(f"    Air flows UP through zigzag with deflector plates")
        print(f"    Bulk velocity: v_bulk = {self.v_air_zigzag:.3f} m/s")
        print(f"    BUT separation occurs in RECIRCULATION ZONES behind plates!")
        print(f"    Zone velocity: v_zone = {self.v_air_zigzag_zone:.3f} m/s ({self.zigzag_velocity_ratio_zone:.0%} of bulk)")
        print(f"    Gravity pulls DOWN at g = {g} m/s2")
        print(f"    Particle terminal velocity: v_t = d^2*(rho_p-rho_f)*g / 18*mu")
        print(f"    ")
        print(f"    CUT SIZE CALCULATION (d50) - using ZONE velocity:")
        print(f"      v_zone = v_bulk x velocity_ratio = {self.v_air_zigzag:.3f} x {self.zigzag_velocity_ratio_zone:.2f} = {self.v_air_zigzag_zone:.3f} m/s")
        print(f"      d50 = sqrt(18*mu*v_zone / (g*(rho_p-rho_f)))")
        print(f"          = sqrt(18 x {mu:.2e} x {self.v_air_zigzag_zone:.3f} / ({g} x ({rho_p}-{rho_f})))")
        print(f"          = sqrt({18*mu*self.v_air_zigzag_zone:.6e} / {g*(rho_p-rho_f):.2f})")
        print(f"          = {self.zigzag_d50*1e6:.1f} um")
        print(f"      (if using bulk velocity: d50_bulk = {self.zigzag_d50_bulk*1e6:.1f} um - WRONG, ignores zone effect)")

        n_stages = self.zigzag_num_stages
        print(f"\n    MULTI-STAGE SHARPENING ({n_stages} stages):")
        print(f"      Each deflector plate creates a separation opportunity")
        print(f"      Senden (1979): T_total = T1^n / (T1^n + (1-T1)^n)")
        print(f"      With n={n_stages} stages, the grade efficiency curve is MUCH sharper")
        print(f"      d50 stays ~{self.zigzag_d50*1e6:.0f}um but transition is steeper")

        print(f"\n    PARTICLE FATE BY SIZE (v_t vs v_zone = {self.v_air_zigzag_zone*1000:.2f} mm/s):")
        for d_um in [10, 20, 30, 40, 50, 60, 80, 100]:
            d_m = d_um * 1e-6
            v_t = terminal_velocity(d_m)
            # Single-stage probability
            if self.v_air_zigzag_zone > 0:
                ratio = v_t / self.v_air_zigzag_zone
                # Probability of going to fines (single stage)
                # Simple model: T1 = 1 / (1 + (v_t/v_zone)^2)
                T1 = 1.0 / (1.0 + ratio ** 2)
                # Multi-stage: T_total = T1^n / (T1^n + (1-T1)^n)
                T1_n = T1 ** n_stages
                T1_n_comp = (1.0 - T1) ** n_stages
                T_total = T1_n / (T1_n + T1_n_comp + 1e-30)
            else:
                T_total = 1.0
            fate = "FINES (protein)" if v_t < self.v_air_zigzag_zone else "COARSE (starch)"
            print(f"      d = {d_um:3d} um: v_t = {v_t*1000:.2f} mm/s {'<' if v_t < self.v_air_zigzag_zone else '>'} v_zone = {self.v_air_zigzag_zone*1000:.2f} mm/s -> {fate} (P_fines={T_total:.1%})")
        
        print(f"\n  PARTICLE ZONES:")
        print(f"    Zone 20: Entering zigzag from below")
        print(f"    Zone 21: In zigzag stages (separation in progress)")
        print(f"    Zone 22: Rising particles -> Fines outlet -> Zone 40")
        print(f"    Zone 30: Falling particles -> Coarse outlet (COLLECTED)")
        
        # =====================================================================
        # SEGMENT 4: ZIGZAG -> CYCLONE TRANSITION + ELBOW
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: ZIGZAG -> CYCLONE PATH (Transition + 90 deg Elbow + Duct)")
        print(f"{'-' * 80}")
        
        print(f"\n  PATH OVERVIEW:")
        print(f"    Start: Zigzag fines outlet ({self.duct_zigzag_cyclone_start[0]*1000:.1f}, {self.duct_zigzag_cyclone_start[1]*1000:.1f}, {self.duct_zigzag_cyclone_start[2]*1000:.1f}) mm")
        print(f"    Elbow: Turn point          ({self.elbow_zigzag_cyclone_pos[0]*1000:.1f}, {self.elbow_zigzag_cyclone_pos[1]*1000:.1f}, {self.elbow_zigzag_cyclone_pos[2]*1000:.1f}) mm")
        print(f"    End:   Cyclone inlet       ({self.duct_zigzag_cyclone_end[0]*1000:.1f}, {self.duct_zigzag_cyclone_end[1]*1000:.1f}, {self.duct_zigzag_cyclone_end[2]*1000:.1f}) mm")
        
        vert_length = self.elbow_zigzag_cyclone_pos[1] - self.duct_zigzag_cyclone_start[1]
        horiz_length = self.duct_zigzag_cyclone_end[0] - self.elbow_zigzag_cyclone_pos[0]
        print(f"\n  SEGMENT LENGTHS:")
        print(f"    Vertical (rect->round trans): {vert_length*1000:.1f} mm")
        print(f"    90 deg Elbow (bend radius):     {self.elbow_zigzag_cyclone_bend_radius*1000:.1f} mm")
        print(f"    Horizontal to cyclone:       {horiz_length*1000:.1f} mm")
        
        print(f"\n  CROSS-SECTION:")
        print(f"    Duct diameter:   {self.duct_zigzag_cyclone_radius*2*1000:.1f} mm")
        A_duct = np.pi * self.duct_zigzag_cyclone_radius**2
        v_duct = Q / A_duct
        print(f"    Area:            {A_duct*1e4:.2f} cm2")
        print(f"    Velocity:        v = Q/A = {v_duct:.2f} m/s")
        
        print(f"\n  90 deg ELBOW PHYSICS:")
        print(f"    Inlet direction:  +Y (upward)")
        print(f"    Outlet direction: +X (horizontal)")
        print(f"    Bend radius:      {self.elbow_zigzag_cyclone_bend_radius*1000:.1f} mm")
        print(f"    Centripetal acc:  a = v^2/R = {v_duct**2/self.elbow_zigzag_cyclone_bend_radius:.1f} m/s2")
        
        print(f"\n  PARTICLE ZONES:")
        print(f"    Zone 22: Vertical transition after zigzag fines outlet")
        print(f"    Zone 40: In 90 deg elbow (turning from +Y to +X)")
        print(f"    Zone 41: Horizontal duct to cyclone inlet")
        print(f"    Transition to Zone 50 at X = {self.duct_zigzag_cyclone_end[0]*1000:.1f} mm")
        
        # =====================================================================
        # SEGMENT 5: MULTI-CYCLONE SYSTEM
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: MULTI-CYCLONE SYSTEM (Staged Centrifugal Separation)")
        print(f"{'-' * 80}")
        
        print(f"\n  INLET:")
        print(f"    Position:    ({self.duct_zigzag_cyclone_end[0]*1000:.1f}, {self.duct_zigzag_cyclone_end[1]*1000:.1f}, {self.duct_zigzag_cyclone_end[2]*1000:.1f}) mm")
        print(f"    Direction:   (-1, 0, 0) - tangential entry")
        print(f"    Inlet vel:   {self.v_air_cyclone_inlet:.1f} m/s")

        if not self.cyclone_vortex_ok:
            print(f"\n  *** VORTEX FORMATION CHECK: FAILED ***")
            print(f"    v_inlet = {self.v_air_cyclone_inlet:.1f} m/s < {self.cyclone_min_vortex_velocity:.0f} m/s minimum")
            print(f"    At this velocity, NO stable vortex forms in the cyclone body.")
            print(f"    The Lapple d50 = {self.cyclone_d50*1e6:.1f} um shown below is PHYSICALLY MEANINGLESS.")
            print(f"    Cyclone acts as a GRAVITY SETTLER - effective d50 >> Lapple d50.")
            print(f"    Need inlet velocity > {self.cyclone_good_vortex_velocity:.0f} m/s for reliable centrifugal separation.")
        elif not self.cyclone_d50_reliable:
            print(f"\n  *** VORTEX FORMATION CHECK: MARGINAL ***")
            print(f"    v_inlet = {self.v_air_cyclone_inlet:.1f} m/s - vortex exists but is weak")
            print(f"    Lapple d50 = {self.cyclone_d50*1e6:.1f} um is UNRELIABLE (actual d50 will be coarser)")
            print(f"    Need inlet velocity > {self.cyclone_good_vortex_velocity:.0f} m/s for accurate Lapple prediction.")
        
        cyclones = [
            ("PRIMARY", self.cyclone_primary_center, self.cyclone_primary_radius*2,
             self.cyclone_primary_cylinder_height, self.cyclone_primary_cone_height,
             self.cyclone_primary_vf_radius*2, self.cyclone_primary_dust_y,
             self.v_air_cyclone_inlet, self.cyclone_primary_inlet_width, self.cyclone_primary_inlet_height,
             self.cyclone_d50, self.cyclone_primary_cone_tip_ratio, 50, 55, "SECONDARY"),
            ("SECONDARY", self.cyclone_secondary_center, self.cyclone_secondary_radius*2,
             self.cyclone_secondary_cylinder_height, self.cyclone_secondary_cone_height,
             self.cyclone_secondary_vf_radius*2, self.cyclone_secondary_dust_y,
             self.v_air_cyclone_secondary_inlet, self.cyclone_secondary_inlet_width, self.cyclone_secondary_inlet_height,
             self.cyclone_secondary_d50, self.cyclone_secondary_cone_tip_ratio, 51, 56, "TERTIARY"),
            ("TERTIARY", self.cyclone_tertiary_center, self.cyclone_tertiary_radius*2,
             self.cyclone_tertiary_cylinder_height, self.cyclone_tertiary_cone_height,
             self.cyclone_tertiary_vf_radius*2, self.cyclone_tertiary_dust_y,
             self.v_air_cyclone_tertiary_inlet, self.cyclone_tertiary_inlet_width, self.cyclone_tertiary_inlet_height,
             self.cyclone_tertiary_d50, self.cyclone_tertiary_cone_tip_ratio, 52, 57, "BAG FILTER"),
        ]

        for name, center, D, H_cyl, H_cone, D_vf, dust_y, v_in, W_in, H_in, d50, tip_ratio, zone_in, zone_dust, next_stage in cyclones:
            print(f"\n  {name} CYCLONE:")
            print(f"    Center:           ({center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}) mm")
            print(f"    Body diameter:    {D*1000:.0f} mm")
            print(f"    Cylinder height:  {H_cyl*1000:.0f} mm")
            print(f"    Cone height:      {H_cone*1000:.0f} mm (tip ratio: {tip_ratio:.3f})")
            print(f"    Vortex finder D:  {D_vf*1000:.0f} mm")
            print(f"    Dust outlet Y:    {dust_y*1000:.1f} mm")
            print(f"    Inlet:            {W_in*1000:.0f}x{H_in*1000:.0f}mm  v={v_in:.1f} m/s")

            # Cyclone separation physics
            R = D / 2
            omega = v_in / R
            print(f"\n    SEPARATION PHYSICS:")
            print(f"      Inlet velocity:      {v_in:.1f} m/s (tangential)")
            print(f"      Angular velocity:    w = v/R = {omega:.1f} rad/s")
            print(f"      Centrifugal accel:   a_c = w^2*R = {omega**2*R:.0f} m/s2 ({omega**2*R/g:.0f}g)")
            print(f"      Lapple d50:          {d50*1e6:.1f} um")
            
            print(f"\n    PARTICLE ZONES:")
            print(f"      Zone {zone_in}: In cyclone body (swirling flow)")
            print(f"      Zone {zone_dust}: Collected in dust outlet (heavy particles)")
            print(f"      Light particles -> vortex finder -> {next_stage}")
        
        # =====================================================================
        # SEGMENT 6: CYCLONE -> BAG FILTER PATH
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: CYCLONE -> BAG FILTER PATH (90 deg Elbow + Expansion)")
        print(f"{'-' * 80}")
        
        print(f"\n  PATH OVERVIEW:")
        print(f"    Start: Cyclone overflow  ({self.duct_cyclone_bag_start[0]*1000:.1f}, {self.duct_cyclone_bag_start[1]*1000:.1f}, {self.duct_cyclone_bag_start[2]*1000:.1f}) mm")
        print(f"    Elbow: Turn point        ({self.elbow_cyclone_bag_pos[0]*1000:.1f}, {self.elbow_cyclone_bag_pos[1]*1000:.1f}, {self.elbow_cyclone_bag_pos[2]*1000:.1f}) mm")
        print(f"    End:   Bag filter inlet  ({self.duct_cyclone_bag_end[0]*1000:.1f}, {self.duct_cyclone_bag_end[1]*1000:.1f}, {self.duct_cyclone_bag_end[2]*1000:.1f}) mm")
        
        print(f"\n  DUCT DIMENSIONS:")
        print(f"    Initial diameter:   {self.duct_cyclone_bag_radius*2*1000:.0f} mm (from overflow)")
        print(f"    Elbow bend radius:  {self.elbow_cyclone_bag_bend_radius*1000:.0f} mm")
        print(f"    Final diameter:     300 mm (bag filter inlet)")
        
        A_small = np.pi * self.duct_cyclone_bag_radius**2
        A_large = np.pi * 0.15**2  # 300mm diameter
        v_small = Q / A_small
        v_large = Q / A_large
        print(f"\n  EXPANSION TRANSITION:")
        print(f"    Small duct: A = {A_small*1e4:.2f} cm2, v = {v_small:.2f} m/s")
        print(f"    Large duct: A = {A_large*1e4:.2f} cm2, v = {v_large:.2f} m/s")
        print(f"    Velocity ratio: {v_small/v_large:.1f}:1 (deceleration)")
        
        print(f"\n  PARTICLE ZONES:")
        print(f"    Zone 60: In 90 deg elbow (turning from +Y to +X)")
        print(f"    Zone 61: Horizontal duct with expansion")
        print(f"    Transition to Zone 70 at X = {self.duct_cyclone_bag_end[0]*1000:.1f} mm")
        
        # =====================================================================
        # SEGMENT 7: BAG FILTER
        # =====================================================================
        segment_num += 1
        print(f"\n{'-' * 80}")
        print(f"SEGMENT {segment_num}: BAG FILTER (Final Particle Capture)")
        print(f"{'-' * 80}")
        
        print(f"\n  GEOMETRY:")
        print(f"    Center position:     ({self.bagfilter_center[0]*1000:.1f}, {self.bagfilter_center[1]*1000:.1f}, {self.bagfilter_center[2]*1000:.1f}) mm")
        print(f"    Housing half-width:  {self.bagfilter_half_width*1000:.0f} mm")
        print(f"    Housing half-depth:  {self.bagfilter_half_depth*1000:.0f} mm")
        print(f"    Housing height:      {self.bagfilter_height*1000:.0f} mm")
        
        print(f"\n  PORT POSITIONS:")
        print(f"    Dirty air inlet:     Y = {self.bagfilter_inlet_y*1000:.1f} mm, dir = (-1, 0, 0)")
        print(f"    Clean air outlet:    Y = {self.bagfilter_outlet_y*1000:.1f} mm, dir = (0, +1, 0)")
        print(f"    Dust hopper:         Y = {self.bagfilter_dust_y*1000:.1f} mm, dir = (0, -1, 0)")
        
        print(f"\n  FILTRATION PHYSICS:")
        A_bags = 2 * self.bagfilter_half_width * self.bagfilter_height  # Approximate filter area
        v_filter = Q / A_bags
        print(f"    Filter area (approx): {A_bags:.2f} m2")
        print(f"    Face velocity:        {v_filter*100:.2f} cm/s = {v_filter*60:.1f} m/min")
        print(f"    Air-to-cloth ratio:   {Q*60/A_bags:.2f} m3/min/m2")
        
        print(f"\n  PARTICLE CAPTURE MECHANISMS:")
        print(f"    - Inertial impaction (large particles)")
        print(f"    - Interception (medium particles)")
        print(f"    - Diffusion (fine particles < 1um)")
        print(f"    Expected efficiency: > 99.9%")
        
        print(f"\n  PARTICLE ZONES:")
        print(f"    Zone 70: In bag filter (capture in progress)")
        print(f"    Zone 75: Collected in dust hopper (COLLECTED)")
        print(f"    Zone 80: Escaped with clean air (should be rare)")
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        print(f"\n{'=' * 80}")
        print("FLOW PATH SUMMARY")
        print(f"{'=' * 80}")
        
        print(f"\n  COMPLETE MATERIAL PATH:")
        print(f"    Feed -> Venturi Solids Inlet (0, {self.venturi_solids_inlet_pos[1]*1000:.0f}, {self.venturi_solids_inlet_pos[2]*1000:.0f}) mm")
        print(f"      | [Zone 0-2] Entrainment in venturi throat")
        print(f"      v")
        print(f"    Venturi Outlet (0, {self.venturi_total_length*1000:.0f}, 0) mm")
        print(f"      | [Zone 10] Round->Rect transition duct")
        print(f"      v")
        print(f"    Zigzag Inlet (0, {self.zigzag_inlet_y*1000:.0f}, 0) mm")
        print(f"      ^ [Zone 20-21] Counter-current separation")
        print(f"      |---> Coarse Outlet [Zone 30] -> STARCH COLLECTION")
        print(f"      v")
        print(f"    Zigzag Fines ({self.duct_zigzag_cyclone_start[0]*1000:.0f}, {self.zigzag_fines_outlet_y*1000:.0f}, 0) mm")
        print(f"      | [Zone 22] Vertical transition")
        print(f"      +--\\ [Zone 40] 90 deg Elbow")
        print(f"         --> [Zone 41] Horizontal to cyclone")
        print(f"    Cyclone Inlet ({self.duct_zigzag_cyclone_end[0]*1000:.0f}, {self.duct_zigzag_cyclone_end[1]*1000:.0f}, 0) mm")
        print(f"      @ [Zone 50-52] Staged centrifugal separation")
        print(f"      |---> Dust outlets [Zone 55-57] -> FINES COLLECTION")
        print(f"      v")
        print(f"    Cyclone Overflow ({self.duct_cyclone_bag_start[0]*1000:.0f}, {self.duct_cyclone_bag_start[1]*1000:.0f}, 0) mm")
        print(f"      +--\\ [Zone 60] 90 deg Elbow")
        print(f"         --> [Zone 61] Expansion duct")
        print(f"    Bag Filter ({self.duct_cyclone_bag_end[0]*1000:.0f}, {self.duct_cyclone_bag_end[1]*1000:.0f}, 0) mm")
        print(f"      | [Zone 70] Final capture")
        print(f"      |---> Dust hopper [Zone 75] -> ULTRA-FINES COLLECTION")
        print(f"      +---> Clean air [Zone 80] -> EXHAUST")
        
        print(f"\n  KEY PARAMETERS:")
        if self._bypass_ratio > 0:
            print(f"    Total air flow:      {self._Q_total*3600:.0f} m3/h (blower)")
            print(f"    Classification flow: {Q*3600:.1f} m3/h ({(1-self._bypass_ratio)*100:.1f}% through zigzag)")
            print(f"    Cyclone flow:        {self._Q_total*3600:.0f} m3/h (after merge)")
        else:
            print(f"    Total air flow:      {Q*3600:.0f} m3/h = {Q*1000:.1f} L/s")
        print(f"    Zigzag cut size d50: {self.zigzag_d50*1e6:.1f} um (using zone velocity)")
        print(f"    Zigzag zone velocity:{self.v_air_zigzag_zone:.2f} m/s ({self.zigzag_velocity_ratio_zone:.0%} of {self.v_air_zigzag:.2f} m/s bulk)")
        if self.cyclone_vortex_ok:
            reliability = " (unreliable)" if not self.cyclone_d50_reliable else ""
            print(f"    Cyclone cut size:    {self.cyclone_d50*1e6:.2f} um{reliability}")
        else:
            print(f"    Cyclone cut size:    {self.cyclone_d50*1e6:.2f} um (INVALID - no vortex at {self.v_air_cyclone_inlet:.1f} m/s)")

        print(f"\n  EXPECTED SEPARATION (at current conditions):")
        # Determine actual separation based on d50 values
        if self.zigzag_d50 > 200e-6:  # d50 > 200um means all flour passes
            print(f"    ALL FLOUR (<200um): -> Zigzag fines -> Cyclones")
            print(f"    Only large fiber (>{self.zigzag_d50*1e6:.0f}um) -> Coarse")
            if self.cyclone_d50 < 5e-6:  # Very fine cyclone cut
                print(f"    ALL fines collected in PRIMARY cyclone (d50={self.cyclone_d50*1e6:.1f}um)")
        elif self.zigzag_d50 > 60e-6:
            print(f"    Coarse starch (>{self.zigzag_d50*1e6:.0f}um): -> Zigzag coarse [Zone 30]")
            print(f"    Protein + fine starch (<{self.zigzag_d50*1e6:.0f}um): -> Cyclones")
        else:
            print(f"    Starch (>{self.zigzag_d50*1e6:.0f}um): -> Zigzag coarse [Zone 30]")
            print(f"    Protein (<{self.zigzag_d50*1e6:.0f}um):  -> Cyclones for collection")
        
        # =====================================================================
        # INDUSTRIAL OPERATING ANALYSIS
        # =====================================================================
        print(f"\n{'-' * 80}")
        print(f"INDUSTRIAL OPERATING ANALYSIS")
        print(f"{'-' * 80}")
        
        Q = self._Q_air
        mu = self._mu
        rho_p = self._rho_p
        rho_f = self._rho_f
        g = self._g
        zz_area = self._A_zigzag
        cyclone_inlet_area = self._A_cyclone_inlet
        
        print(f"\n  CURRENT OPERATING POINT:")
        if self._bypass_ratio > 0:
            print(f"    Blower output:       {self._Q_total*3600:.0f} m3/h")
            print(f"    Bypass ratio:        {self._bypass_ratio*100:.1f}%")
            print(f"    Classification flow: {Q*3600:.1f} m3/h (through venturi+zigzag)")
            print(f"    Cyclone flow:        {self._Q_total*3600:.0f} m3/h (after bypass merge)")
        else:
            print(f"    Air flow: {Q*3600:.0f} m3/h ({Q*1000:.1f} L/s)")
        print(f"    Zigzag bulk velocity: {self.v_air_zigzag:.2f} m/s")
        print(f"    Zigzag zone velocity: {self.v_air_zigzag_zone:.2f} m/s ({self.zigzag_velocity_ratio_zone:.0%} of bulk)")
        print(f"    Zigzag d50: {self.zigzag_d50*1e6:.1f} um (based on zone velocity)")
        print(f"    Cyclone inlet velocity: {self.v_air_cyclone_inlet:.1f} m/s", end="")
        if not self.cyclone_vortex_ok:
            print(f"  *** NO VORTEX ***")
        elif not self.cyclone_d50_reliable:
            print(f"  *** WEAK VORTEX ***")
        else:
            print()
        if self.cyclone_vortex_ok:
            print(f"    Cyclone d50: {self.cyclone_d50*1e6:.2f} um", end="")
            if not self.cyclone_d50_reliable:
                print(f" (unreliable)")
            else:
                print()
        else:
            print(f"    Cyclone d50: {self.cyclone_d50*1e6:.2f} um (INVALID - no vortex, acts as gravity settler)")

        # Operating mode determination
        if self.zigzag_d50 > 200e-6:
            print(f"\n  OPERATING MODE: BYPASS ZIGZAG")
            print(f"    At this flow, zigzag passes all material to cyclones.")
            print(f"    Zigzag acts as transport duct, not separator.")
        elif self.zigzag_d50 > 60e-6:
            print(f"\n  OPERATING MODE: COARSE SEPARATION")
            print(f"    Zigzag removes large particles (>{self.zigzag_d50*1e6:.0f}um) to coarse.")
            print(f"    Protein + fine starch pass to cyclones for further separation.")
        elif self.zigzag_d50 > 25e-6:
            print(f"\n  OPERATING MODE: PROTEIN SEPARATION")
            print(f"    Zigzag separates at d50={self.zigzag_d50*1e6:.1f}um.")
            print(f"    Good for protein/starch separation (target: 25-35um).")
        else:
            print(f"\n  OPERATING MODE: FINE SEPARATION")
            print(f"    Zigzag separates at d50={self.zigzag_d50*1e6:.1f}um.")
            print(f"    Very fine cut - most material goes to coarse.")

        if self.cyclone_d50 < 5e-6:
            print(f"    WARNING: Cyclone d50={self.cyclone_d50*1e6:.2f}um - all material collected in Cy1!")

        # Calculate recommended operating ranges
        # NOTE: d50 is based on zone velocity, so:
        # v_zone = v_bulk * ratio -> v_bulk = v_zone / ratio
        # d50 = sqrt(18*mu*v_zone / (g*delta_rho))
        # v_zone = d50^2 * g * delta_rho / (18*mu)
        # Q = v_bulk * A = (v_zone / ratio) * A
        vzr = self.zigzag_velocity_ratio_zone

        print(f"\n  RECOMMENDED OPERATING RANGES:")
        print(f"    (Zone velocity = {vzr:.0%} of bulk, accounts for deflector plate effect)")

        # For d50 = 35 um (protein/starch boundary)
        d50_target_ps = 35e-6
        v_zone_ps = (d50_target_ps**2 * g * (rho_p - rho_f)) / (18 * mu)
        v_bulk_ps = v_zone_ps / vzr
        Q_ps = v_bulk_ps * zz_area
        v_cyc_ps = Q_ps / cyclone_inlet_area

        # For d50 = 50 um (moderate separation)
        d50_target_mod = 50e-6
        v_zone_mod = (d50_target_mod**2 * g * (rho_p - rho_f)) / (18 * mu)
        v_bulk_mod = v_zone_mod / vzr
        Q_mod = v_bulk_mod * zz_area
        v_cyc_mod = Q_mod / cyclone_inlet_area

        # For d50 = 100 um (fiber rejection)
        d50_target_fiber = 100e-6
        v_zone_fiber = (d50_target_fiber**2 * g * (rho_p - rho_f)) / (18 * mu)
        v_bulk_fiber = v_zone_fiber / vzr
        Q_fiber = v_bulk_fiber * zz_area
        v_cyc_fiber = Q_fiber / cyclone_inlet_area

        # For cyclone at 20 m/s (typical industrial)
        Q_cyc_20 = 20.0 * cyclone_inlet_area
        v_zz_bulk_at_20 = Q_cyc_20 / zz_area
        v_zz_zone_at_20 = v_zz_bulk_at_20 * vzr
        d50_at_20 = np.sqrt(18 * mu * v_zz_zone_at_20 / (g * (rho_p - rho_f))) * 1e6

        print(f"\n    For protein/starch separation (d50=35um):")
        print(f"      v_zone = {v_zone_ps:.3f} m/s, v_bulk = {v_bulk_ps:.3f} m/s")
        print(f"      Q = {Q_ps*3600:.1f} m3/h, v_cyclone = {v_cyc_ps:.2f} m/s")
        if v_cyc_ps < 10:
            print(f"      NOTE: Cyclone velocity low - consider smaller cyclone or staged approach")

        print(f"\n    For moderate separation (d50=50um):")
        print(f"      v_zone = {v_zone_mod:.3f} m/s, v_bulk = {v_bulk_mod:.3f} m/s")
        print(f"      Q = {Q_mod*3600:.1f} m3/h, v_cyclone = {v_cyc_mod:.2f} m/s")

        print(f"\n    For fiber rejection (d50=100um):")
        print(f"      v_zone = {v_zone_fiber:.3f} m/s, v_bulk = {v_bulk_fiber:.3f} m/s")
        print(f"      Q = {Q_fiber*3600:.1f} m3/h, v_cyclone = {v_cyc_fiber:.2f} m/s")
        if v_cyc_fiber < 10:
            print(f"      NOTE: Cyclone velocity low for this flow rate")

        print(f"\n    For optimal cyclone operation (v_cyc=20 m/s):")
        print(f"      Q = {Q_cyc_20*3600:.0f} m3/h, zigzag d50 = {d50_at_20:.0f} um")
        if d50_at_20 > 200:
            print(f"      MODE: All flour to fines, cyclones do staged separation")
        elif d50_at_20 > 60:
            print(f"      MODE: Coarse separation, cyclones handle fine fractions")
        else:
            print(f"      MODE: Effective protein/starch separation")

        # Practical recommendation
        print(f"\n  PRACTICAL RECOMMENDATION:")
        if self.zigzag_d50 > 200e-6:
            print(f"    Current conditions ({Q*3600:.0f} m3/h) give d50={self.zigzag_d50*1e6:.0f}um")
            print(f"    Zigzag acts as transport - all flour to cyclones")
            print(f"    Options:")
            print(f"    1. Reduce air flow to ~{Q_ps*3600:.0f} m3/h for protein separation")
            print(f"    2. Increase channel size to reduce velocity at this flow")
            print(f"    3. Use cyclone-only separation (current bypass mode)")
        elif self.zigzag_d50 > 60e-6:
            print(f"    Current d50={self.zigzag_d50*1e6:.0f}um - coarse separation")
            print(f"    Reduce air flow or channel size for protein range (25-35um)")
        else:
            print(f"    Current d50={self.zigzag_d50*1e6:.0f}um - in protein separation range")
            print(f"    Adjust --air-flow to fine-tune separation point")
        
        print(f"\n{'=' * 80}")
    
    def initialize_particles(
        self,
        num_particles: int = None,
        mean_diameter: float = 30e-6,   # 30 um (flour average)
        diameter_std: float = 15e-6,    # 15 um std dev
        initial_velocity: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    ):
        """
        Initialize particles at the venturi solids inlet.
        
        Args:
            num_particles: Number of particles (default: config value)
            mean_diameter: Mean particle diameter [m] (default 30um)
            diameter_std: Standard deviation [m] (default 15um)
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
        print(f"    Diameter range: {diameters.min()*1e6:.1f} - {diameters.max()*1e6:.1f} um")
        print(f"    Mean diameter:  {diameters.mean()*1e6:.1f} um")
        print(f"    Inlet position: ({self.venturi_solids_inlet_pos[0]*1000:.0f}, {self.venturi_solids_inlet_pos[1]*1000:.0f}, {self.venturi_solids_inlet_pos[2]*1000:.0f}) mm")
    
    # =========================================================================
    # FEED SYSTEM INTEGRATION (particles from feed_flow_physics)
    # =========================================================================
    
    def get_solids_inlet_position(self) -> np.ndarray:
        """Return world position of venturi solids inlet (for feed->classification connection)."""
        return np.array(self.venturi_solids_inlet_pos, dtype=np.float64)
    
    def get_solids_inlet_direction(self) -> np.ndarray:
        """Return world direction of venturi solids inlet (flow into venturi)."""
        return np.array(self.venturi_solids_inlet_dir, dtype=np.float64)
    
    def inject_particles_from_feed(
        self,
        transfer_data: Dict[str, Any],
        offset_feed_outlet_to_solids_inlet: Optional[np.ndarray] = None,
    ) -> int:
        """
        Inject particles transferred from the feed system into the venturi solids inlet.
        
        Uses output from feed_flow_physics.get_particle_data_for_transfer().
        Positions are translated from feed outlet to classification solids inlet
        (optionally via offset_feed_outlet_to_solids_inlet = solids_inlet_pos - feed_outlet_pos).
        
        Args:
            transfer_data: Dict with keys positions, velocities, diameters, masses,
                types (optional), count; and optionally outlet_position, outlet_direction.
            offset_feed_outlet_to_solids_inlet: If provided, add this to positions.
                If None, computed as solids_inlet_pos - transfer_data['outlet_position'].
        
        Returns:
            Number of particles injected.
        """
        positions = np.asarray(transfer_data['positions'], dtype=np.float64)
        velocities = np.asarray(transfer_data['velocities'], dtype=np.float64)
        diameters = np.asarray(transfer_data['diameters'], dtype=np.float64)
        masses = np.asarray(transfer_data['masses'], dtype=np.float64)
        count = int(transfer_data['count'])
        
        if count == 0:
            return 0
        
        # Position offset: from feed outlet to classification solids inlet
        if offset_feed_outlet_to_solids_inlet is not None:
            offset = np.asarray(offset_feed_outlet_to_solids_inlet, dtype=np.float64)
        else:
            outlet_pos = transfer_data.get('outlet_position')
            if outlet_pos is not None:
                outlet_pos = np.asarray(outlet_pos, dtype=np.float64)
                offset = self.get_solids_inlet_position() - outlet_pos
            else:
                offset = np.zeros(3, dtype=np.float64)
        
        positions = positions[:count] + offset
        
        n_max = self.config.num_particles
        n_inject = min(count, n_max - self.state.particles_active)
        if n_inject <= 0:
            print("  Classification: no slot for injected particles; increase num_particles or clear some.")
            return 0
        
        start = self.state.particles_active
        end = start + n_inject
        
        # Update device arrays via numpy slice (read full, update slice, copy back)
        pos_full = self.state.positions.numpy().copy()
        pos_full[start:end] = positions[:n_inject]
        self.state.positions = wp.array(pos_full, dtype=wp.vec3, device=self.device)
        
        vel_full = self.state.velocities.numpy().copy()
        vel_full[start:end] = velocities[:n_inject]
        self.state.velocities = wp.array(vel_full, dtype=wp.vec3, device=self.device)
        
        dia_full = self.state.diameters.numpy().copy()
        dia_full[start:end] = diameters[:n_inject]
        self.state.diameters = wp.array(dia_full, dtype=float, device=self.device)
        
        mass_full = self.state.masses.numpy().copy()
        mass_full[start:end] = masses[:n_inject]
        self.state.masses = wp.array(mass_full, dtype=float, device=self.device)
        
        zone_full = self.state.zones.numpy().copy()
        zone_full[start:end] = 0  # VENTURI_INLET
        self.state.zones = wp.array(zone_full, dtype=wp.int32, device=self.device)
        
        active_full = self.state.is_active.numpy().copy()
        active_full[start:end] = 1
        self.state.is_active = wp.array(active_full, dtype=wp.int32, device=self.device)
        
        dens_full = self.state.densities.numpy().copy()
        if transfer_data.get('densities') is not None:
            dens_full[start:end] = np.asarray(transfer_data['densities'][:n_inject], dtype=np.float64)
        else:
            dens_full[start:end] = self.config.particle_density
        self.state.densities = wp.array(dens_full, dtype=float, device=self.device)
        
        type_full = self.state.particle_types.numpy().copy()
        if transfer_data.get('types') is not None:
            type_full[start:end] = np.asarray(transfer_data['types'][:n_inject], dtype=np.int32)
        else:
            type_full[start:end] = 0
        self.state.particle_types = wp.array(type_full, dtype=wp.int32, device=self.device)
        
        self.state.particles_active = end
        print(f"  Injected {n_inject} particles from feed system at venturi solids inlet (total active: {end})")
        return n_inject
    
    def initialize_particles_from_material(
        self,
        material: ParticleMaterial,
        num_particles: Optional[int] = None,
        initial_velocity: Tuple[float, float, float] = (0.0, 0.5, 0.0),
        visual_scale_diameter: Optional[float] = None,
    ) -> None:
        """
        Initialize particles at venturi solids inlet using a ParticleMaterial.
        
        Uses create_particle_population from the particles module for consistent
        size/density/sphericity with feed and other systems.
        """
        n = num_particles or self.config.num_particles
        n = min(n, self.config.num_particles)
        
        diameters_np, densities_np, sphericities_np = create_particle_population(material, n)
        diameters_np = np.asarray(diameters_np, dtype=np.float64)
        densities_np = np.asarray(densities_np, dtype=np.float64)
        
        # Masses from volume and density
        vols = (np.pi / 6.0) * (diameters_np ** 3)
        masses_np = densities_np * vols
        
        # Clamp diameters to classification range (e.g. 5-100 um) if needed
        scale = visual_scale_diameter
        if scale is not None and np.median(diameters_np) < scale * 0.1:
            diameters_np = np.clip(diameters_np, 5e-6, 100e-6)
        
        # Positions: at solids inlet, random in circle
        rng = np.random.default_rng(42)
        r = np.sqrt(rng.uniform(0, 1, n)) * self.venturi_solids_inlet_radius * 0.8
        theta = rng.uniform(0, 2 * np.pi, n)
        cx, cy, cz = self.venturi_solids_inlet_pos[0], self.venturi_solids_inlet_pos[1], self.venturi_solids_inlet_pos[2]
        positions_np = np.column_stack([
            cx + r * np.cos(theta),
            np.full(n, cy),
            cz + r * np.sin(theta),
        ]).astype(np.float64)
        
        # Velocities: initial_velocity + small random
        velocities_np = np.zeros((n, 3), dtype=np.float64)
        velocities_np[:] = initial_velocity
        velocities_np += rng.uniform(-0.05, 0.05, (n, 3))
        
        # Zones: all VENTURI_INLET (0)
        zones_np = np.zeros(n, dtype=np.int32)
        # Continuous feeding: start inactive
        if self.config.continuous_feeding:
            is_active_np = np.zeros(n, dtype=np.int32)
        else:
            is_active_np = np.ones(n, dtype=np.int32)
        types_np = np.zeros(n, dtype=np.int32)  # single material

        # Copy to device (full arrays: fill 0:n, rest unchanged)
        n_max = self.config.num_particles
        pos_full = self.state.positions.numpy().copy()
        pos_full[0:n] = positions_np
        self.state.positions = wp.array(pos_full, dtype=wp.vec3, device=self.device)
        vel_full = self.state.velocities.numpy().copy()
        vel_full[0:n] = velocities_np
        self.state.velocities = wp.array(vel_full, dtype=wp.vec3, device=self.device)
        dia_full = self.state.diameters.numpy().copy()
        dia_full[0:n] = diameters_np
        self.state.diameters = wp.array(dia_full, dtype=float, device=self.device)
        mass_full = self.state.masses.numpy().copy()
        mass_full[0:n] = masses_np
        self.state.masses = wp.array(mass_full, dtype=float, device=self.device)
        zone_full = self.state.zones.numpy().copy()
        zone_full[0:n] = zones_np
        self.state.zones = wp.array(zone_full, dtype=wp.int32, device=self.device)
        active_full = self.state.is_active.numpy().copy()
        active_full[0:n] = is_active_np
        self.state.is_active = wp.array(active_full, dtype=wp.int32, device=self.device)
        dens_full = self.state.densities.numpy().copy()
        dens_full[0:n] = densities_np
        self.state.densities = wp.array(dens_full, dtype=float, device=self.device)
        type_full = self.state.particle_types.numpy().copy()
        type_full[0:n] = types_np
        self.state.particle_types = wp.array(type_full, dtype=wp.int32, device=self.device)

        if self.config.continuous_feeding:
            self.state.particles_active = 0
            self.state.total_particles_to_feed = n
            self.state.particles_fed = 0
            self._feed_accumulator = 0.0
            print(f"\n  Pre-allocated {n} particles from material (continuous feeding)")
            print(f"    Feed rate: {self.config.particle_feed_rate:.0f} particles/s")
        else:
            self.state.particles_active = n
            print(f"\n  Initialized {n} particles from material at venturi inlet (batch)")
        print(f"    Diameter range: {diameters_np.min()*1e6:.1f} - {diameters_np.max()*1e6:.1f} um")
        print(f"    Mean diameter:  {diameters_np.mean()*1e6:.1f} um")
    
    def initialize_whole_flour_population(
        self,
        source: str = "yellow_pea",
        num_particles: Optional[int] = None,
        initial_velocity: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    ) -> None:
        """
        Initialize a whole-flour particle population (protein + starch + fiber) at venturi solids inlet.
        
        Uses create_whole_flour_population from the particles module for consistency
        with the feed system; particles enter classification as they would from the deagglomerator.
        """
        n = num_particles or self.config.num_particles
        n = min(n, self.config.num_particles)
        
        _material, diameters_np, densities_np, _sphericities, types_np = create_whole_flour_population(source, n, 43)
        diameters_np = np.asarray(diameters_np, dtype=np.float64)
        densities_np = np.asarray(densities_np, dtype=np.float64)
        types_np = np.asarray(types_np, dtype=np.int32)
        
        # Clamp to classification size range (microns)
        diameters_np = np.clip(diameters_np, 5e-6, 100e-6)
        
        vols = (np.pi / 6.0) * (diameters_np ** 3)
        masses_np = densities_np * vols
        
        rng = np.random.default_rng(43)
        r = np.sqrt(rng.uniform(0, 1, n)) * self.venturi_solids_inlet_radius * 0.8
        theta = rng.uniform(0, 2 * np.pi, n)
        cx, cy, cz = self.venturi_solids_inlet_pos[0], self.venturi_solids_inlet_pos[1], self.venturi_solids_inlet_pos[2]
        positions_np = np.column_stack([
            cx + r * np.cos(theta),
            np.full(n, cy),
            cz + r * np.sin(theta),
        ]).astype(np.float64)
        
        velocities_np = np.zeros((n, 3), dtype=np.float64)
        velocities_np[:] = initial_velocity
        velocities_np += rng.uniform(-0.05, 0.05, (n, 3))
        
        zones_np = np.zeros(n, dtype=np.int32)
        # Continuous feeding: all particles start inactive; step() activates them gradually
        if self.config.continuous_feeding:
            is_active_np = np.zeros(n, dtype=np.int32)
        else:
            is_active_np = np.ones(n, dtype=np.int32)

        n_max = self.config.num_particles
        pos_full = self.state.positions.numpy().copy()
        pos_full[0:n] = positions_np
        self.state.positions = wp.array(pos_full, dtype=wp.vec3, device=self.device)
        vel_full = self.state.velocities.numpy().copy()
        vel_full[0:n] = velocities_np
        self.state.velocities = wp.array(vel_full, dtype=wp.vec3, device=self.device)
        dia_full = self.state.diameters.numpy().copy()
        dia_full[0:n] = diameters_np
        self.state.diameters = wp.array(dia_full, dtype=float, device=self.device)
        mass_full = self.state.masses.numpy().copy()
        mass_full[0:n] = masses_np
        self.state.masses = wp.array(mass_full, dtype=float, device=self.device)
        zone_full = self.state.zones.numpy().copy()
        zone_full[0:n] = zones_np
        self.state.zones = wp.array(zone_full, dtype=wp.int32, device=self.device)
        active_full = self.state.is_active.numpy().copy()
        active_full[0:n] = is_active_np
        self.state.is_active = wp.array(active_full, dtype=wp.int32, device=self.device)
        dens_full = self.state.densities.numpy().copy()
        dens_full[0:n] = densities_np
        self.state.densities = wp.array(dens_full, dtype=float, device=self.device)
        type_full = self.state.particle_types.numpy().copy()
        type_full[0:n] = types_np
        self.state.particle_types = wp.array(type_full, dtype=wp.int32, device=self.device)
        
        n_protein = int(np.sum(types_np == 0))
        n_starch = int(np.sum(types_np == 1))
        n_fiber = int(np.sum(types_np == 2))
        total_mass = float(np.sum(masses_np))

        # Continuous feeding: pre-allocate all particles but start with 0 active
        if self.config.continuous_feeding:
            self.state.particles_active = 0
            self.state.total_particles_to_feed = n
            self.state.particles_fed = 0
            self._feed_accumulator = 0.0
            print(f"\n  Pre-allocated {n} particles as {source} whole flour (continuous feeding)")
            print(f"    Feed rate: {self.config.particle_feed_rate:.0f} particles/s")
            m_per_particle = float(np.mean(masses_np))
            m_dot = self.config.particle_feed_rate * m_per_particle
            print(f"    Mass flow: {m_dot*3600:.1f} kg/h  ({m_dot:.4f} kg/s)")
            fill_time = n / self.config.particle_feed_rate if self.config.particle_feed_rate > 0 else float('inf')
            print(f"    Time to feed all {n} particles: {fill_time:.1f} s")
        else:
            self.state.particles_active = n
            print(f"\n  Initialized {n} particles as {source} whole flour at venturi inlet (batch)")

        print(f"    Protein: {n_protein} ({100*n_protein/n:.0f}%)  Starch: {n_starch} ({100*n_starch/n:.0f}%)  Fiber: {n_fiber} ({100*n_fiber/n:.0f}%)")
        print(f"    Diameter range: {diameters_np.min()*1e6:.1f} - {diameters_np.max()*1e6:.1f} um  Total mass: {total_mass*1000:.2f} g")
    
    def step(self):
        """Advance simulation by one time step."""
        dt = self.config.dt
        cfg = self.config

        # Continuous feeding: activate pre-allocated particles at feed_rate
        if cfg.continuous_feeding and self.state.particles_active < self.state.total_particles_to_feed:
            feed_rate = cfg.particle_feed_rate
            if feed_rate > 0:
                if not hasattr(self, '_feed_accumulator'):
                    self._feed_accumulator = 0.0
                self._feed_accumulator += feed_rate * dt
                new_count = int(self._feed_accumulator)
                if new_count > 0:
                    self._feed_accumulator -= new_count
                    old_active = self.state.particles_active
                    new_active = min(self.state.total_particles_to_feed, old_active + new_count)
                    actually_added = new_active - old_active
                    if actually_added > 0:
                        # Mark newly activated particles as active on device
                        active_np = self.state.is_active.numpy().copy()
                        active_np[old_active:new_active] = 1
                        self.state.is_active = wp.array(active_np, dtype=wp.int32, device=self.device)
                        self.state.particles_active = new_active
                        self.state.particles_fed += actually_added

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
                # Deflector plate parameters (NEW)
                float(self.zigzag_plate_angle),
                float(self.zigzag_plate_length),
                float(self.zigzag_throat_width),
                float(self.zigzag_velocity_ratio_zone),
                float(self.zigzag_recirculation_length_ratio),

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
                float(self.bagfilter_inlet_radius),
                
                # Ducts
                wp.vec3(*self.duct_venturi_zigzag_start),
                wp.vec3(*self.duct_venturi_zigzag_end),
                float(self.duct_venturi_zigzag_radius),
                
                # Zigzag to cyclone path with elbow
                wp.vec3(*self.duct_zigzag_cyclone_start),
                wp.vec3(*self.duct_zigzag_cyclone_end),
                float(self.duct_zigzag_cyclone_radius),
                wp.vec3(*self.elbow_zigzag_cyclone_pos),
                float(self.elbow_zigzag_cyclone_bend_radius),
                
                # Cyclone to bag filter path with elbow
                wp.vec3(*self.duct_cyclone_bag_start),
                wp.vec3(*self.duct_cyclone_bag_end),
                float(self.duct_cyclone_bag_radius),
                wp.vec3(*self.elbow_cyclone_bag_pos),
                float(self.elbow_cyclone_bag_bend_radius),
                
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
                float(self.v_air_cyclone_secondary_inlet),
                float(self.v_air_cyclone_tertiary_inlet),

                # Cyclone cone tip ratios
                float(self.cyclone_primary_cone_tip_ratio),
                float(self.cyclone_secondary_cone_tip_ratio),
                float(self.cyclone_tertiary_cone_tip_ratio),
                
                # Turbulence
                float(cfg.turbulent_intensity),

                # Wheel classifier geometry
                int(1 if self.wheel_enabled else 0),
                wp.vec3(*self.wheel_center),
                float(self.wheel_radius),
                float(self.wheel_hub_radius),
                float(self.wheel_width),
                int(self.wheel_num_blades),
                float(self.wheel_blade_thickness),
                float(self.wheel_omega),
                float(self.wheel_housing_radius),
                float(self.wheel_center[1] - self.wheel_width / 2.0 - self.wheel_hopper_height),  # hopper bottom Y
                float(self.wheel_hopper_half_angle),
                float(self.wheel_coarse_outlet_diameter / 2.0),  # wheel_coarse_outlet_radius [m]
                float(self.wheel_fines_outlet_pos[1]),  # fines outlet Y
                float(self.wheel_coarse_outlet_pos[1]),  # coarse outlet Y
                float(self.wheel_inlet_pos[1]),  # inlet Y
                float(self.wheel_volumetric_flow),

                # Random seed
                random_seed,

                # Simulation time (for rotating blade collision)
                float(self.state.time),
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
        self._count_wheel_coarse.zero_()
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
                self._count_wheel_coarse,
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
            'wheel_coarse': int(self._count_wheel_coarse.numpy()[0]),
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
            'zigzag': int(np.sum((active_zones == 20) | (active_zones == 21))),
            'fines_path': int(np.sum(active_zones == 22)),
            'coarse_outlet': int(np.sum(active_zones == 30)),
            # Wheel classifier zones
            'wheel_housing': int(np.sum(active_zones == 34)),
            'wheel_fines': int(np.sum(active_zones == 35)),
            'wheel_coarse_hopper': int(np.sum(active_zones == 36)),
            'wheel_coarse_collected': int(np.sum(active_zones == 37)),
            # Continue to cyclones
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

    def get_cyclone_particle_size_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get particle size statistics for each cyclone dust outlet (zones 55, 56, 57).
        Returns count, mean diameter [µm], and design d50 [µm] per stage.
        """
        zones = self.state.zones.numpy()[:self.state.particles_active]
        diameters = self.state.diameters.numpy()[:self.state.particles_active]  # [m]
        zone_to_stage = {55: 'primary', 56: 'secondary', 57: 'tertiary'}
        stage_to_key = {'primary': 'cyclone_1', 'secondary': 'cyclone_2', 'tertiary': 'cyclone_3_protein'}
        out = {}
        for zone_id, stage_name in zone_to_stage.items():
            key = stage_to_key[stage_name]
            mask = (zones == zone_id)
            count = int(np.sum(mask))
            design_d50_um = None
            if self.geometry.get('cyclone_stages') and stage_name in self.geometry['cyclone_stages']:
                d50_m = self.geometry['cyclone_stages'][stage_name].get('design_d50')
                if d50_m is not None:
                    design_d50_um = float(d50_m) * 1e6
            if count > 0:
                mean_d_um = float(np.mean(diameters[mask]) * 1e6)
                median_d_um = float(np.median(diameters[mask]) * 1e6)
            else:
                mean_d_um = median_d_um = None
            out[key] = {
                'count': count,
                'mean_d_um': mean_d_um,
                'median_d_um': median_d_um,
                'design_d50_um': design_d50_um,
            }
        return out

    def print_separation_summary(self):
        """Print a summary of separation results with full particle balance along the path."""
        counts = self.get_separation_counts()
        total = sum(counts.values())
        n_slots = getattr(self.state, 'particles_active', total)

        print(f"\n  Separation Results (t = {self.state.time:.3f}s):")
        print(f"  {'='*50}")
        # Path order: zigzag coarse -> wheel coarse -> cyclones -> bag -> escaped -> still active
        labels = [
            ('coarse', 'Zigzag coarse (starch):  '),
            ('wheel_coarse', 'Wheel coarse (starch):   '),
            ('cyclone_1', 'Cyclone 1 (fines 1):     '),
            ('cyclone_2', 'Cyclone 2 (fines 2):     '),
            ('cyclone_3_protein', 'Cyclone 3 (PROTEIN):     '),
            ('bagfilter', 'Bag filter:               '),
            ('escaped', 'Escaped (loss):           '),
            ('active', 'Still active:              '),
        ]
        for key, label in labels:
            c = counts.get(key, 0)
            pct = 100 * c / max(1, total)
            print(f"    {label} {c:5d} ({pct:5.1f}%)")
        print(f"  {'='*50}")
        # Cyclone particle sizes (design d50 and actual mean in each stage)
        try:
            cy_stats = self.get_cyclone_particle_size_stats()
            if any(cy_stats[k]['count'] > 0 for k in cy_stats):
                print(f"\n  Cyclone particle sizes (design d50 vs actual mean):")
                for key in ('cyclone_1', 'cyclone_2', 'cyclone_3_protein'):
                    s = cy_stats.get(key, {})
                    design = s.get('design_d50_um')
                    mean_d = s.get('mean_d_um')
                    cnt = s.get('count', 0)
                    if design is not None or cnt > 0:
                        design_str = f"design d50={design:.0f} µm" if design is not None else "design d50=N/A"
                        mean_str = f", mean={mean_d:.1f} µm" if mean_d is not None else ""
                        print(f"    {key:20s} N={cnt:5d}  ({design_str}{mean_str})")
        except Exception:
            pass
        print(f"  {'='*50}")
        print(f"    Total (balance):       {total:5d}  (slots used: {n_slots})")
        if total != n_slots:
            print(f"    [Note: total by destination = {total}, particle slots = {n_slots}]")
        print(f"  {'='*50}")

        # Separation quality indicators
        protein_collected = counts['cyclone_3_protein'] + counts['bagfilter']
        starch_collected = counts['coarse'] + counts['wheel_coarse']

        print(f"\n    Protein recovery (cy3 + bag): {protein_collected}")
        print(f"    Starch recovery (zigzag + wheel coarse): {starch_collected}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_classification_simulator(
    air_flow_rate_m3s: float = 1768.0 / 3600.0,  # Air system at 2500 RPM (~1768 m3/h)
    particle_density: float = 1450.0,
    num_particles: int = 10000,
    device: str = "cuda",
) -> Tuple['ClassificationSystemAssembly', ClassificationFlowPhysicsSimulator]:
    """
    Create a classification system and flow simulator.
    
    Args:
        air_flow_rate_m3s: Air volumetric flow rate [m3/s] (default: 1768 m3/h from air system at 2500 RPM)
        particle_density: Particle density [kg/m3] (flour ~1450)
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
