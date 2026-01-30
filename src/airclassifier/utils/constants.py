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
