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
