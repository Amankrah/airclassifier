"""
Milling Warp Kernels
====================

GPU-accelerated kernels for hammer mill physics simulation.
All kernels have NumPy fallbacks for CPU execution.

Kernels:
    - transport: Particle advection in mill chamber
    - impact: Hammer-particle collision detection
    - breakage: Particle size reduction model
    - screen: Screen passage classification
"""

from .transport import (
    transport_step_np,
    GRAVITY,
    AIR_DENSITY,
    AIR_VISCOSITY,
)
from .impact import impact_detection_np
from .breakage import (
    breakage_step_np,
    breakage_psd_np,
)
from .screen import (
    screen_passage_np,
    apply_screen_discharge,
)

# Warp imports (optional)
try:
    import warp as wp
    WARP_AVAILABLE = True

    from .transport import transport_step_warp
    from .impact import impact_detection_warp
    from .breakage import breakage_step_warp
    from .screen import screen_passage_warp

except ImportError:
    WARP_AVAILABLE = False
    transport_step_warp = None
    impact_detection_warp = None
    breakage_step_warp = None
    screen_passage_warp = None


__all__ = [
    # Availability flag
    "WARP_AVAILABLE",
    # Transport
    "transport_step_np",
    "transport_step_warp",
    "GRAVITY",
    "AIR_DENSITY",
    "AIR_VISCOSITY",
    # Impact
    "impact_detection_np",
    "impact_detection_warp",
    # Breakage
    "breakage_step_np",
    "breakage_step_warp",
    "breakage_psd_np",
    # Screen
    "screen_passage_np",
    "screen_passage_warp",
    "apply_screen_discharge",
]
