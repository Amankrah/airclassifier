"""
Material definitions for particles in the cyclone air classifier.

Defines material properties including density, size distribution,
and shape factors for various particle types.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
import warp as wp

from ..utils.constants import (
    PI, 
    MaterialDensities, 
    FoodPowderSizeRanges,
    FoodPowderComposition,
)


class SizeDistributionType(Enum):
    """Types of particle size distributions."""
    MONODISPERSE = "monodisperse"       # Single size
    UNIFORM = "uniform"                  # Uniform distribution
    NORMAL = "normal"                    # Normal (Gaussian)
    LOGNORMAL = "lognormal"             # Log-normal
    ROSIN_RAMMLER = "rosin_rammler"     # Rosin-Rammler (Weibull)
    GATES_GAUDIN = "gates_gaudin"       # Gates-Gaudin-Schumann


@dataclass
class SizeDistributionParams:
    """Parameters for particle size distribution."""

    type: SizeDistributionType = SizeDistributionType.ROSIN_RAMMLER

    # Common parameters
    d_min: float = 1.0e-6    # [m] Minimum diameter
    d_max: float = 200.0e-6  # [m] Maximum diameter

    # Distribution-specific parameters
    d50: float = 50.0e-6     # [m] Median diameter (d50)
    d_mean: float = 50.0e-6  # [m] Mean diameter
    d_std: float = 20.0e-6   # [m] Standard deviation

    # Rosin-Rammler parameters
    spread: float = 2.0      # [-] Spread parameter (n)

    # Gates-Gaudin-Schumann parameter
    m: float = 0.5           # [-] Distribution modulus

    def get_params_dict(self) -> Dict[str, float]:
        """Get parameters as dictionary."""
        return {
            "d_min": self.d_min,
            "d_max": self.d_max,
            "d50": self.d50,
            "d_mean": self.d_mean,
            "d_std": self.d_std,
            "spread": self.spread,
            "m": self.m,
        }


@dataclass
class MaterialProperties:
    """
    Physical properties of a particle material.

    Defines all properties needed for particle dynamics calculations.
    """

    name: str                           # Material identifier
    density: float                      # [kg/m³] Particle density

    # Shape properties
    sphericity: float = 1.0             # [-] Sphericity (1.0 = perfect sphere)
    shape_factor: float = 1.0           # [-] Shape factor for drag

    # Surface properties
    surface_roughness: float = 0.0      # [-] Relative surface roughness
    restitution_coefficient: float = 0.8  # [-] Coefficient of restitution
    friction_coefficient: float = 0.3   # [-] Friction coefficient

    # Optional thermal/chemical properties
    specific_heat: Optional[float] = None      # [J/(kg·K)]
    thermal_conductivity: Optional[float] = None  # [W/(m·K)]

    @classmethod
    def from_preset(cls, name: str) -> "MaterialProperties":
        """
        Create material from preset values.

        Args:
            name: Material name (e.g., "quartz", "coal", "calcium_carbonate")

        Returns:
            MaterialProperties instance
        """
        presets = {
            "ite": cls(
                name="ite",
                density=MaterialDensities.ITE,
                sphericity=0.85,
                restitution_coefficient=0.7,
            ),
            "quartz": cls(
                name="quartz",
                density=MaterialDensities.QUARTZ,
                sphericity=0.85,
                restitution_coefficient=0.7,
            ),
            "coal": cls(
                name="coal",
                density=MaterialDensities.COAL,
                sphericity=0.75,
                restitution_coefficient=0.5,
            ),
            "calcium_carbonate": cls(
                name="calcium_carbonate",
                density=MaterialDensities.CALCIUM_CARBONATE,
                sphericity=0.9,
                restitution_coefficient=0.75,
            ),
            "cement": cls(
                name="cement",
                density=MaterialDensities.CEMENT,
                sphericity=0.8,
                restitution_coefficient=0.6,
            ),
            "fly_ash": cls(
                name="fly_ash",
                density=MaterialDensities.FLY_ASH,
                sphericity=0.95,  # Often nearly spherical
                restitution_coefficient=0.65,
            ),
            "limestone": cls(
                name="limestone",
                density=MaterialDensities.LIMESTONE,
                sphericity=0.82,
                restitution_coefficient=0.7,
            ),
            "iron_ore": cls(
                name="iron_ore",
                density=MaterialDensities.MAGNETITE,
                sphericity=0.78,
                restitution_coefficient=0.6,
            ),
            # =========================================================
            # Food Powders - Plant-Based Protein Sources
            # =========================================================
            # Yellow Pea
            "yellow_pea": cls(
                name="yellow_pea",
                density=MaterialDensities.YELLOW_PEA_WHOLE,
                sphericity=0.70,  # Irregular flour particles
                shape_factor=1.2,
                surface_roughness=0.15,
                restitution_coefficient=0.3,  # Soft organic material
                friction_coefficient=0.5,
            ),
            "yellow_pea_protein": cls(
                name="yellow_pea_protein",
                density=MaterialDensities.YELLOW_PEA_PROTEIN,
                sphericity=0.65,  # Fine, irregular protein bodies
                shape_factor=1.3,
                surface_roughness=0.20,
                restitution_coefficient=0.25,
                friction_coefficient=0.55,
            ),
            "yellow_pea_starch": cls(
                name="yellow_pea_starch",
                density=MaterialDensities.YELLOW_PEA_STARCH,
                sphericity=0.85,  # Starch granules are more rounded
                shape_factor=1.1,
                surface_roughness=0.08,
                restitution_coefficient=0.35,
                friction_coefficient=0.4,
            ),
            "yellow_pea_fiber": cls(
                name="yellow_pea_fiber",
                density=MaterialDensities.YELLOW_PEA_FIBER,
                sphericity=0.55,  # Very irregular fiber particles
                shape_factor=1.5,
                surface_roughness=0.25,
                restitution_coefficient=0.20,
                friction_coefficient=0.60,
            ),
            # Faba Bean
            "faba_bean": cls(
                name="faba_bean",
                density=MaterialDensities.FABA_BEAN_WHOLE,
                sphericity=0.72,
                shape_factor=1.2,
                surface_roughness=0.12,
                restitution_coefficient=0.3,
                friction_coefficient=0.5,
            ),
            "faba_bean_protein": cls(
                name="faba_bean_protein",
                density=MaterialDensities.FABA_BEAN_PROTEIN,
                sphericity=0.68,
                shape_factor=1.25,
                surface_roughness=0.18,
                restitution_coefficient=0.28,
                friction_coefficient=0.52,
            ),
            "faba_bean_starch": cls(
                name="faba_bean_starch",
                density=MaterialDensities.FABA_BEAN_STARCH,
                sphericity=0.82,  # Starch granules
                shape_factor=1.1,
                surface_roughness=0.10,
                restitution_coefficient=0.32,
                friction_coefficient=0.42,
            ),
            "faba_bean_fiber": cls(
                name="faba_bean_fiber",
                density=MaterialDensities.FABA_BEAN_FIBER,
                sphericity=0.52,  # Very irregular fiber particles
                shape_factor=1.55,
                surface_roughness=0.28,
                restitution_coefficient=0.18,
                friction_coefficient=0.62,
            ),
            # Oat
            "oat": cls(
                name="oat",
                density=MaterialDensities.OAT_WHOLE,
                sphericity=0.68,  # Oat particles tend to be more irregular
                shape_factor=1.3,
                surface_roughness=0.18,
                restitution_coefficient=0.28,
                friction_coefficient=0.55,
            ),
            "oat_protein": cls(
                name="oat_protein",
                density=MaterialDensities.OAT_PROTEIN,
                sphericity=0.62,
                shape_factor=1.35,
                surface_roughness=0.22,
                restitution_coefficient=0.25,
                friction_coefficient=0.58,
            ),
            "oat_starch": cls(
                name="oat_starch",
                density=MaterialDensities.OAT_STARCH,
                sphericity=0.80,
                shape_factor=1.15,
                surface_roughness=0.12,
                restitution_coefficient=0.30,
                friction_coefficient=0.45,
            ),
            "oat_bran": cls(
                name="oat_bran",
                density=MaterialDensities.OAT_BRAN,
                sphericity=0.55,  # Very irregular fiber particles
                shape_factor=1.5,
                surface_roughness=0.25,
                restitution_coefficient=0.20,
                friction_coefficient=0.60,
            ),
        }

        if name.lower() not in presets:
            raise ValueError(f"Unknown material preset: {name}. "
                           f"Available: {list(presets.keys())}")

        return presets[name.lower()]


@dataclass
class ParticleMaterial:
    """
    Complete particle material definition including size distribution.

    Combines material properties with size distribution for a complete
    description of the particle population.
    """

    properties: MaterialProperties
    size_distribution: SizeDistributionParams = field(
        default_factory=SizeDistributionParams
    )

    @property
    def name(self) -> str:
        """Material name."""
        return self.properties.name

    @property
    def density(self) -> float:
        """Particle density [kg/m³]."""
        return self.properties.density

    @property
    def sphericity(self) -> float:
        """Particle sphericity."""
        return self.properties.sphericity

    def sample_diameters(self, n: int, seed: int = 42) -> np.ndarray:
        """
        Sample particle diameters from the size distribution.

        Args:
            n: Number of samples
            seed: Random seed

        Returns:
            Array of diameters [m]
        """
        rng = np.random.default_rng(seed)
        sd = self.size_distribution

        if sd.type == SizeDistributionType.MONODISPERSE:
            return np.full(n, sd.d_mean)

        elif sd.type == SizeDistributionType.UNIFORM:
            return rng.uniform(sd.d_min, sd.d_max, n)

        elif sd.type == SizeDistributionType.NORMAL:
            samples = rng.normal(sd.d_mean, sd.d_std, n)
            return np.clip(samples, sd.d_min, sd.d_max)

        elif sd.type == SizeDistributionType.LOGNORMAL:
            # Convert to log-normal parameters
            mu = np.log(sd.d_mean ** 2 / np.sqrt(sd.d_std ** 2 + sd.d_mean ** 2))
            sigma = np.sqrt(np.log(1 + sd.d_std ** 2 / sd.d_mean ** 2))
            samples = rng.lognormal(mu, sigma, n)
            return np.clip(samples, sd.d_min, sd.d_max)

        elif sd.type == SizeDistributionType.ROSIN_RAMMLER:
            # Rosin-Rammler: F(d) = 1 - exp(-(d/d63.2)^n)
            # where d63.2 = d50 / (ln(2))^(1/n)
            d_char = sd.d50 / (np.log(2) ** (1.0 / sd.spread))
            # Inverse sampling: d = d_char * (-ln(1-F))^(1/n)
            u = rng.uniform(0.001, 0.999, n)  # Avoid edge cases
            samples = d_char * (-np.log(1 - u)) ** (1.0 / sd.spread)
            return np.clip(samples, sd.d_min, sd.d_max)

        elif sd.type == SizeDistributionType.GATES_GAUDIN:
            # Gates-Gaudin-Schumann: F(d) = (d/d_max)^m
            # Inverse: d = d_max * F^(1/m)
            u = rng.uniform(0.0, 1.0, n)
            samples = sd.d_max * (u ** (1.0 / sd.m))
            return np.clip(samples, sd.d_min, sd.d_max)

        else:
            raise ValueError(f"Unknown distribution type: {sd.type}")

    def get_mass(self, diameter: float) -> float:
        """
        Calculate mass of a single particle.

        Args:
            diameter: Particle diameter [m]

        Returns:
            Particle mass [kg]
        """
        volume = (PI / 6.0) * diameter ** 3
        return self.density * volume

    def get_masses(self, diameters: np.ndarray) -> np.ndarray:
        """
        Calculate masses for array of diameters.

        Args:
            diameters: Array of particle diameters [m]

        Returns:
            Array of particle masses [kg]
        """
        volumes = (PI / 6.0) * diameters ** 3
        return self.density * volumes

    def get_terminal_velocity_stokes(self, diameter: float,
                                      fluid_density: float,
                                      fluid_viscosity: float,
                                      g: float = 9.81) -> float:
        """
        Calculate Stokes terminal settling velocity.

        Valid for Rep < 0.1.

        Args:
            diameter: Particle diameter [m]
            fluid_density: Fluid density [kg/m³]
            fluid_viscosity: Fluid dynamic viscosity [Pa·s]
            g: Gravitational acceleration [m/s²]

        Returns:
            Terminal velocity [m/s]
        """
        return (self.density - fluid_density) * g * diameter ** 2 / (18.0 * fluid_viscosity)

    @classmethod
    def create(cls, material_name: str,
               distribution_type: str = "rosin_rammler",
               d50: float = 50.0e-6,
               spread: float = 2.0,
               d_min: float = 1.0e-6,
               d_max: float = 200.0e-6) -> "ParticleMaterial":
        """
        Convenience method to create a particle material.

        Args:
            material_name: Preset material name
            distribution_type: Size distribution type
            d50: Median diameter [m]
            spread: Rosin-Rammler spread parameter
            d_min: Minimum diameter [m]
            d_max: Maximum diameter [m]

        Returns:
            ParticleMaterial instance
        """
        props = MaterialProperties.from_preset(material_name)

        dist_type = SizeDistributionType(distribution_type)
        size_dist = SizeDistributionParams(
            type=dist_type,
            d50=d50,
            d_mean=d50,
            spread=spread,
            d_min=d_min,
            d_max=d_max,
        )

        return cls(properties=props, size_distribution=size_dist)
    
    @classmethod
    def create_food_powder(
        cls,
        source: str,
        fraction: str = "whole",
    ) -> "ParticleMaterial":
        """
        Create a food powder material with appropriate size distribution.
        
        Designed for protein separation from legumes and cereals.
        
        Args:
            source: "yellow_pea", "faba_bean", or "oat"
            fraction: "whole", "protein", "starch", or "fiber"/"bran"
            
        Returns:
            ParticleMaterial configured for the food powder type
            
        Example:
            >>> pea_protein = ParticleMaterial.create_food_powder("yellow_pea", "protein")
            >>> oat_flour = ParticleMaterial.create_food_powder("oat", "whole")
        """
        source = source.lower().replace(" ", "_")
        fraction = fraction.lower()
        
        # Build material name
        if fraction == "whole":
            material_name = source
        elif source == "oat" and fraction in ["fiber", "bran"]:
            material_name = f"{source}_bran"
        else:
            material_name = f"{source}_{fraction}"
        
        # Get size distribution based on fraction type
        if fraction == "protein":
            size_params = SizeDistributionParams(
                type=SizeDistributionType.LOGNORMAL,
                d_min=FoodPowderSizeRanges.PROTEIN_D_MIN,
                d_max=FoodPowderSizeRanges.PROTEIN_D_MAX,
                d50=FoodPowderSizeRanges.PROTEIN_D50,
                d_mean=FoodPowderSizeRanges.PROTEIN_D50,
                d_std=FoodPowderSizeRanges.PROTEIN_D50 * 0.6,
                spread=2.5,  # Narrow distribution for protein
            )
        elif fraction == "starch":
            size_params = SizeDistributionParams(
                type=SizeDistributionType.ROSIN_RAMMLER,
                d_min=FoodPowderSizeRanges.STARCH_D_MIN,
                d_max=FoodPowderSizeRanges.STARCH_D_MAX,
                d50=FoodPowderSizeRanges.STARCH_D50,
                d_mean=FoodPowderSizeRanges.STARCH_D50,
                d_std=FoodPowderSizeRanges.STARCH_D50 * 0.5,
                spread=2.0,
            )
        elif fraction in ["fiber", "bran"]:
            size_params = SizeDistributionParams(
                type=SizeDistributionType.ROSIN_RAMMLER,
                d_min=FoodPowderSizeRanges.FIBER_D_MIN,
                d_max=FoodPowderSizeRanges.FIBER_D_MAX,
                d50=FoodPowderSizeRanges.FIBER_D50,
                d_mean=FoodPowderSizeRanges.FIBER_D50,
                d_std=FoodPowderSizeRanges.FIBER_D50 * 0.7,
                spread=1.5,  # Wide distribution for fiber
            )
        else:  # whole
            size_params = SizeDistributionParams(
                type=SizeDistributionType.ROSIN_RAMMLER,
                d_min=FoodPowderSizeRanges.WHOLE_D_MIN,
                d_max=FoodPowderSizeRanges.WHOLE_D_MAX,
                d50=FoodPowderSizeRanges.WHOLE_D50,
                d_mean=FoodPowderSizeRanges.WHOLE_D50,
                d_std=FoodPowderSizeRanges.WHOLE_D50 * 0.8,
                spread=1.8,  # Broad distribution for whole flour
            )
        
        props = MaterialProperties.from_preset(material_name)
        return cls(properties=props, size_distribution=size_params)


# =============================================================================
# WARP STRUCTURES FOR GPU COMPUTATION
# =============================================================================

@wp.struct
class WarpMaterialProps:
    """Warp-compatible material properties structure."""
    density: float
    sphericity: float
    restitution: float
    friction: float


def material_to_warp(material: ParticleMaterial) -> WarpMaterialProps:
    """
    Convert ParticleMaterial to Warp-compatible structure.

    Args:
        material: ParticleMaterial instance

    Returns:
        WarpMaterialProps instance
    """
    props = WarpMaterialProps()
    props.density = material.density
    props.sphericity = material.sphericity
    props.restitution = material.properties.restitution_coefficient
    props.friction = material.properties.friction_coefficient
    return props


@wp.func
def particle_volume(diameter: float) -> float:
    """Calculate particle volume assuming sphere."""
    return (3.141592653589793 / 6.0) * diameter * diameter * diameter


@wp.func
def particle_mass(diameter: float, density: float) -> float:
    """Calculate particle mass."""
    return density * particle_volume(diameter)


@wp.func
def particle_projected_area(diameter: float) -> float:
    """Calculate particle projected area (sphere)."""
    return (3.141592653589793 / 4.0) * diameter * diameter


# =============================================================================
# SHARED FLUID CONFIGURATION
# =============================================================================

@dataclass
class FluidConfig:
    """
    Fluid properties configuration for particle physics simulations.
    
    Reusable across feed, air, and classification systems.
    Default values are for air at 20°C, 1 atm.
    """
    density: float = 1.204           # [kg/m³] Fluid density
    dynamic_viscosity: float = 1.825e-5  # [Pa·s] Dynamic viscosity
    temperature_c: Optional[float] = 20.0  # [°C] Temperature (for reference)
    pressure_Pa: Optional[float] = 101325.0  # [Pa] Pressure (for reference)
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity [m²/s]."""
        return self.dynamic_viscosity / self.density
    
    @classmethod
    def air_at_stp(cls) -> "FluidConfig":
        """Air at standard temperature and pressure (20°C, 1 atm)."""
        return cls(
            density=1.204,
            dynamic_viscosity=1.825e-5,
            temperature_c=20.0,
            pressure_Pa=101325.0,
        )
    
    @classmethod
    def air_at_temperature(cls, T_celsius: float, P_Pa: float = 101325.0) -> "FluidConfig":
        """
        Air properties at specified temperature and pressure.
        
        Args:
            T_celsius: Temperature in Celsius
            P_Pa: Pressure in Pascals (default: 1 atm)
        """
        T_K = T_celsius + 273.15
        R = 287.05  # Specific gas constant for air [J/(kg·K)]
        
        # Density from ideal gas law
        density = P_Pa / (R * T_K)
        
        # Viscosity from Sutherland's formula
        T_ref = 291.15  # Reference temperature [K]
        mu_ref = 1.827e-5  # Reference viscosity [Pa·s]
        S = 120.0  # Sutherland constant [K]
        viscosity = mu_ref * (T_K / T_ref) ** 1.5 * (T_ref + S) / (T_K + S)
        
        return cls(
            density=density,
            dynamic_viscosity=viscosity,
            temperature_c=T_celsius,
            pressure_Pa=P_Pa,
        )


# =============================================================================
# SHARED SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class ParticlePhysicsConfig:
    """
    Shared particle physics configuration.
    
    Used by feed_flow_physics, air_flow_physics, and classification_flow_physics.
    """
    # Particle properties
    particle_density: float = 1450.0     # [kg/m³] Default flour-like material
    sphericity: float = 0.75             # [-] Shape factor (0.5-1.0)
    
    # Collision parameters
    restitution: float = 0.3             # [-] Coefficient of restitution (0-1)
    friction: float = 0.4                # [-] Friction coefficient
    
    # Fluid properties
    fluid: FluidConfig = None
    
    # Physics toggles
    include_gravity: bool = True
    include_drag: bool = True
    include_buoyancy: bool = True
    include_turbulent_dispersion: bool = False
    turbulent_intensity: float = 0.15    # [-] Turbulence intensity (0-0.3)
    
    # Numerical parameters
    dt: float = 0.001                    # [s] Time step
    max_velocity: float = 100.0          # [m/s] Velocity clamp
    
    def __post_init__(self):
        if self.fluid is None:
            self.fluid = FluidConfig.air_at_stp()
    
    @classmethod
    def for_feed_system(cls) -> "ParticlePhysicsConfig":
        """Configuration optimized for feed system (gravity flow)."""
        return cls(
            particle_density=1450.0,
            sphericity=0.70,
            restitution=0.3,
            friction=0.5,
            include_gravity=True,
            include_drag=True,
            include_turbulent_dispersion=False,
            dt=0.005,  # Larger dt for slower flows
        )
    
    @classmethod
    def for_classification_system(cls) -> "ParticlePhysicsConfig":
        """Configuration optimized for classification system (air transport)."""
        return cls(
            particle_density=1450.0,
            sphericity=0.75,
            restitution=0.3,
            friction=0.4,
            include_gravity=True,
            include_drag=True,
            include_turbulent_dispersion=True,
            turbulent_intensity=0.15,
            dt=0.001,  # Smaller dt for high velocity flows
        )
    
    @classmethod
    def for_cyclone_separation(cls) -> "ParticlePhysicsConfig":
        """Configuration optimized for cyclone separation (swirling flow)."""
        return cls(
            particle_density=1450.0,
            sphericity=0.75,
            restitution=0.3,
            friction=0.4,
            include_gravity=True,
            include_drag=True,
            include_turbulent_dispersion=True,
            turbulent_intensity=0.10,
            dt=0.0005,  # Very small dt for high centrifugal accelerations
        )


# =============================================================================
# PARTICLE POPULATION FACTORY FUNCTIONS
# =============================================================================

def create_particle_population(
    material: ParticleMaterial,
    num_particles: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a particle population with realistic size distribution.
    
    Args:
        material: ParticleMaterial defining density and size distribution
        num_particles: Number of particles to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (diameters, densities, sphericities) arrays
    """
    diameters = material.sample_diameters(num_particles, seed=seed)
    densities = np.full(num_particles, material.density, dtype=np.float32)
    sphericities = np.full(num_particles, material.sphericity, dtype=np.float32)
    
    return diameters, densities, sphericities


def create_bimodal_population(
    material_fine: ParticleMaterial,
    material_coarse: ParticleMaterial,
    num_particles: int,
    fine_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a bimodal particle population (e.g., protein + starch).
    
    Args:
        material_fine: Material for fine particles (e.g., protein)
        material_coarse: Material for coarse particles (e.g., starch)
        num_particles: Total number of particles
        fine_fraction: Fraction of fine particles (0-1)
        seed: Random seed
        
    Returns:
        Tuple of (diameters, densities, sphericities, types) arrays
        types: 0 = fine (protein), 1 = coarse (starch)
    """
    rng = np.random.default_rng(seed)
    
    n_fine = int(num_particles * fine_fraction)
    n_coarse = num_particles - n_fine
    
    # Sample diameters
    d_fine = material_fine.sample_diameters(n_fine, seed=seed)
    d_coarse = material_coarse.sample_diameters(n_coarse, seed=seed + 1)
    
    diameters = np.concatenate([d_fine, d_coarse]).astype(np.float32)
    
    # Densities
    densities = np.concatenate([
        np.full(n_fine, material_fine.density),
        np.full(n_coarse, material_coarse.density),
    ]).astype(np.float32)
    
    # Sphericities
    sphericities = np.concatenate([
        np.full(n_fine, material_fine.sphericity),
        np.full(n_coarse, material_coarse.sphericity),
    ]).astype(np.float32)
    
    # Types
    types = np.concatenate([
        np.zeros(n_fine, dtype=np.int32),
        np.ones(n_coarse, dtype=np.int32),
    ])
    
    # Shuffle
    indices = rng.permutation(num_particles)
    return (
        diameters[indices],
        densities[indices], 
        sphericities[indices],
        types[indices],
    )


def create_whole_flour_population(
    source: str = "yellow_pea",
    num_particles: int = 1000,
    seed: int = 42,
) -> Tuple[ParticleMaterial, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a realistic whole flour population with protein, starch, and fiber.
    
    Args:
        source: Source material ("yellow_pea", "faba_bean", "oat")
        num_particles: Total number of particles
        seed: Random seed
        
    Returns:
        Tuple of (material, diameters, densities, sphericities, types)
        types: 0 = protein, 1 = starch, 2 = fiber
    """
    rng = np.random.default_rng(seed)
    
    # Create material fractions
    protein = ParticleMaterial.create_food_powder(source, "protein")
    starch = ParticleMaterial.create_food_powder(source, "starch")
    fiber = ParticleMaterial.create_food_powder(source, "fiber" if source != "oat" else "bran")
    whole = ParticleMaterial.create_food_powder(source, "whole")
    
    # Typical composition fractions
    f_protein = 0.25  # 25% protein
    f_starch = 0.55   # 55% starch
    f_fiber = 0.20    # 20% fiber
    
    n_protein = int(num_particles * f_protein)
    n_starch = int(num_particles * f_starch)
    n_fiber = num_particles - n_protein - n_starch
    
    # Sample diameters for each fraction
    d_protein = protein.sample_diameters(n_protein, seed=seed)
    d_starch = starch.sample_diameters(n_starch, seed=seed + 1)
    d_fiber = fiber.sample_diameters(n_fiber, seed=seed + 2)
    
    diameters = np.concatenate([d_protein, d_starch, d_fiber]).astype(np.float32)
    
    densities = np.concatenate([
        np.full(n_protein, protein.density),
        np.full(n_starch, starch.density),
        np.full(n_fiber, fiber.density),
    ]).astype(np.float32)
    
    sphericities = np.concatenate([
        np.full(n_protein, protein.sphericity),
        np.full(n_starch, starch.sphericity),
        np.full(n_fiber, fiber.sphericity),
    ]).astype(np.float32)
    
    types = np.concatenate([
        np.zeros(n_protein, dtype=np.int32),
        np.ones(n_starch, dtype=np.int32),
        np.full(n_fiber, 2, dtype=np.int32),
    ])
    
    # Shuffle
    indices = rng.permutation(num_particles)
    return (
        whole,
        diameters[indices],
        densities[indices],
        sphericities[indices],
        types[indices],
    )
