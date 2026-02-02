"""
Physical constants and standard properties for cyclone air classifier simulation.

All values are in SI units unless otherwise noted.
"""

import warp as wp

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966

# Gravitational acceleration [m/s²]
GRAVITY = 9.80665
GRAVITY_VEC = wp.vec3(0.0, -9.80665, 0.0)  # Pointing downward (y-axis)

# =============================================================================
# AIR PROPERTIES AT STANDARD CONDITIONS (20°C, 1 atm)
# =============================================================================

class AirProperties:
    """Standard air properties at 20°C and 1 atm."""

    DENSITY = 1.204              # [kg/m³]
    DYNAMIC_VISCOSITY = 1.825e-5 # [Pa·s]
    KINEMATIC_VISCOSITY = 1.516e-5  # [m²/s]
    SPECIFIC_HEAT_CP = 1006.0    # [J/(kg·K)]
    THERMAL_CONDUCTIVITY = 0.0257  # [W/(m·K)]
    PRANDTL_NUMBER = 0.713       # [-]
    SPEED_OF_SOUND = 343.0       # [m/s]
    MOLAR_MASS = 0.02897         # [kg/mol]
    GAS_CONSTANT = 287.05        # [J/(kg·K)]


class AirPropertiesAtTemp:
    """
    Air properties as functions of temperature.
    Valid range: 200K - 400K (approximate).
    """

    @staticmethod
    def density(T: float, P: float = 101325.0) -> float:
        """
        Air density using ideal gas law.

        Args:
            T: Temperature [K]
            P: Pressure [Pa]

        Returns:
            Density [kg/m³]
        """
        R = 287.05  # Specific gas constant for air
        return P / (R * T)

    @staticmethod
    def dynamic_viscosity(T: float) -> float:
        """
        Dynamic viscosity using Sutherland's formula.

        Args:
            T: Temperature [K]

        Returns:
            Dynamic viscosity [Pa·s]
        """
        T_ref = 291.15  # Reference temperature [K]
        mu_ref = 1.827e-5  # Reference viscosity [Pa·s]
        S = 120.0  # Sutherland constant [K]

        return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

    @staticmethod
    def kinematic_viscosity(T: float, P: float = 101325.0) -> float:
        """
        Kinematic viscosity.

        Args:
            T: Temperature [K]
            P: Pressure [Pa]

        Returns:
            Kinematic viscosity [m²/s]
        """
        mu = AirPropertiesAtTemp.dynamic_viscosity(T)
        rho = AirPropertiesAtTemp.density(T, P)
        return mu / rho


# =============================================================================
# COMMON PARTICLE MATERIAL DENSITIES [kg/m³]
# =============================================================================

class MaterialDensities:
    """Common material densities for particles."""

    # Minerals
    ITE = 2650.0         # Genericite mineral
    QUARTZ = 2650.0
    FELDSPAR = 2560.0
    CALCITE = 2710.0
    LIMESTONE = 2700.0
    DOLOMITE = 2850.0
    GRANITE = 2750.0
    BARITE = 4500.0
    MAGNETITE = 5150.0
    HEMATITE = 5260.0

    # Coal and organics
    COAL = 1400.0
    ANTHRACITE = 1550.0
    LIGNITE = 1250.0
    WOOD = 600.0

    # Industrial materials
    CEMENT = 3150.0
    FLY_ASH = 2200.0
    CALCIUM_CARBONATE = 2710.0
    CALCIUM_OXIDE = 3340.0
    ALUMINA = 3950.0
    SILICA = 2200.0

    # Metals
    IRON = 7874.0
    COPPER = 8960.0
    ALUMINUM = 2700.0
    LEAD = 11340.0

    # Glass and ceramics
    GLASS = 2500.0
    CERAMIC = 2400.0

    # =========================================================================
    # FOOD POWDERS - Plant-Based Protein Sources
    # =========================================================================
    
    # Yellow Pea (Pisum sativum) - True densities
    YELLOW_PEA_WHOLE = 1420.0        # Whole flour particle density
    YELLOW_PEA_PROTEIN = 1350.0     # Protein-rich fraction (lighter, finer)
    YELLOW_PEA_STARCH = 1500.0      # Starch-rich fraction (denser, coarser)
    YELLOW_PEA_FIBER = 1250.0       # Fiber fraction (lightest)
    
    # Faba Bean (Vicia faba) - True densities
    FABA_BEAN_WHOLE = 1450.0        # Whole flour particle density
    FABA_BEAN_PROTEIN = 1380.0      # Protein-rich fraction
    FABA_BEAN_STARCH = 1520.0       # Starch-rich fraction
    FABA_BEAN_FIBER = 1280.0        # Fiber fraction
    
    # Oat (Avena sativa) - True densities
    OAT_WHOLE = 1350.0              # Whole oat flour
    OAT_PROTEIN = 1320.0            # Protein-rich fraction
    OAT_STARCH = 1450.0             # Starch-rich fraction
    OAT_BRAN = 1280.0               # Bran/fiber fraction
    OAT_BETA_GLUCAN = 1380.0        # Beta-glucan rich fraction
    
    # Generic food powder ranges
    FLOUR_TYPICAL = 1400.0          # Typical flour density
    PROTEIN_ISOLATE = 1300.0        # Pure protein isolate
    STARCH_GRANULE = 1500.0         # Pure starch granule


class FoodPowderBulkDensities:
    """
    Bulk densities for food powders (includes air between particles).
    Used for hopper/feeder calculations.
    """
    
    # Yellow Pea
    YELLOW_PEA_LOOSE = 450.0        # [kg/m³] Loose/poured
    YELLOW_PEA_TAPPED = 580.0       # [kg/m³] Tapped/settled
    
    # Faba Bean
    FABA_BEAN_LOOSE = 480.0
    FABA_BEAN_TAPPED = 620.0
    
    # Oat
    OAT_LOOSE = 400.0
    OAT_TAPPED = 520.0


class FoodPowderComposition:
    """
    Typical composition of plant protein sources (mass fractions).
    Used for determining particle type ratios in simulation.
    """
    
    # Yellow Pea flour composition
    YELLOW_PEA_PROTEIN_CONTENT = 0.23      # 23% protein
    YELLOW_PEA_STARCH_CONTENT = 0.52       # 52% starch
    YELLOW_PEA_FIBER_CONTENT = 0.15        # 15% fiber
    YELLOW_PEA_OTHER_CONTENT = 0.10        # 10% lipids, minerals, etc.
    
    # Faba Bean flour composition
    FABA_BEAN_PROTEIN_CONTENT = 0.28       # 28% protein (higher than pea)
    FABA_BEAN_STARCH_CONTENT = 0.48        # 48% starch
    FABA_BEAN_FIBER_CONTENT = 0.14         # 14% fiber
    FABA_BEAN_OTHER_CONTENT = 0.10         # 10% other
    
    # Oat flour composition
    OAT_PROTEIN_CONTENT = 0.13             # 13% protein
    OAT_STARCH_CONTENT = 0.60              # 60% starch
    OAT_FIBER_CONTENT = 0.10               # 10% fiber (incl. beta-glucan)
    OAT_LIPID_CONTENT = 0.07               # 7% lipids (higher than legumes)
    OAT_OTHER_CONTENT = 0.10               # 10% other


class FoodPowderSizeRanges:
    """
    Typical particle size ranges for food powder fractions [m].
    Based on air classification separation characteristics.
    """
    
    # Protein-rich fraction (finer particles)
    PROTEIN_D_MIN = 2.0e-6         # 2 μm
    PROTEIN_D50 = 12.0e-6          # 12 μm median
    PROTEIN_D_MAX = 35.0e-6        # 35 μm
    
    # Starch-rich fraction (coarser particles)  
    STARCH_D_MIN = 15.0e-6         # 15 μm
    STARCH_D50 = 45.0e-6           # 45 μm median
    STARCH_D_MAX = 120.0e-6        # 120 μm
    
    # Fiber fraction (large irregular particles)
    FIBER_D_MIN = 50.0e-6          # 50 μm
    FIBER_D50 = 150.0e-6           # 150 μm median
    FIBER_D_MAX = 500.0e-6         # 500 μm
    
    # Whole flour (before classification)
    WHOLE_D_MIN = 2.0e-6           # 2 μm
    WHOLE_D50 = 50.0e-6            # 50 μm median
    WHOLE_D_MAX = 500.0e-6         # 500 μm


# =============================================================================
# DIMENSIONLESS NUMBERS THRESHOLDS
# =============================================================================

class FlowRegimes:
    """Reynolds number thresholds for flow regimes."""

    # Particle Reynolds number
    STOKES_LIMIT = 0.1          # Stokes drag valid below this
    INTERMEDIATE_LOWER = 0.1    # Intermediate regime start
    INTERMEDIATE_UPPER = 1000.0 # Intermediate regime end
    NEWTON_LOWER = 1000.0       # Newton's law regime start

    # Pipe/duct flow
    LAMINAR_LIMIT = 2300.0      # Below: laminar flow
    TRANSITION_UPPER = 4000.0   # Above: fully turbulent


# =============================================================================
# NUMERICAL CONSTANTS
# =============================================================================

class NumericalConstants:
    """Constants for numerical stability and convergence."""

    EPSILON = 1.0e-10           # Small number to avoid division by zero
    SQRT_EPSILON = 1.0e-5       # Square root of epsilon
    MAX_VELOCITY = 1000.0       # Maximum allowable velocity [m/s]
    MIN_PARTICLE_DIAMETER = 1.0e-7  # Minimum particle size [m]
    MAX_PARTICLE_DIAMETER = 1.0e-1  # Maximum particle size [m]


# =============================================================================
# WARP-COMPATIBLE CONSTANTS (for use in kernels)
# =============================================================================

# These can be used directly in Warp kernels
WP_PI = wp.constant(PI)
WP_TWO_PI = wp.constant(TWO_PI)
WP_GRAVITY = wp.constant(GRAVITY)
WP_AIR_DENSITY = wp.constant(AirProperties.DENSITY)
WP_AIR_VISCOSITY = wp.constant(AirProperties.DYNAMIC_VISCOSITY)
WP_EPSILON = wp.constant(NumericalConstants.EPSILON)
