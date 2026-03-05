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
    generate_fragments_np,
)
from .screen import (
    screen_passage_np,
    apply_screen_discharge,
)
from .reagglomeration import reagglomeration_step_np

# Warp imports (optional)
# Catch all exceptions because Warp JIT can fail with RuntimeError in PyInstaller bundles
try:
    import warp as wp
    # Try to initialize warp - this triggers JIT compilation
    wp.init()
    WARP_AVAILABLE = True

    from .transport import transport_step_warp
    from .impact import impact_detection_warp
    from .breakage import breakage_step_warp
    from .screen import screen_passage_warp

except Exception as e:
    # ImportError: warp not installed
    # RuntimeError: warp JIT compilation failed (e.g., in PyInstaller bundle)
    WARP_AVAILABLE = False
    wp = None
    transport_step_warp = None
    impact_detection_warp = None
    breakage_step_warp = None
    screen_passage_warp = None
    # Store error for debugging
    _WARP_INIT_ERROR = str(e)


# GPU device info cache
_GPU_INFO = None


def get_gpu_info() -> dict:
    """Get GPU device information.

    Returns:
        dict with keys:
            - available: bool, whether CUDA is available
            - device_name: str, GPU name or 'N/A'
            - device_count: int, number of CUDA devices
            - warp_version: str, Warp version or 'N/A'
    """
    global _GPU_INFO
    if _GPU_INFO is not None:
        return _GPU_INFO

    info = {
        "available": False,
        "device_name": "N/A",
        "device_count": 0,
        "warp_version": "N/A",
    }

    if not WARP_AVAILABLE or wp is None:
        _GPU_INFO = info
        return info

    try:
        # Initialize Warp
        wp.init()
        info["warp_version"] = getattr(wp, "__version__", "unknown")

        # Check CUDA devices
        devices = wp.get_cuda_devices()
        info["device_count"] = len(devices)

        if len(devices) > 0:
            info["available"] = True
            # Get device name from first CUDA device
            try:
                info["device_name"] = wp.get_cuda_device_name(0)
            except Exception:
                info["device_name"] = f"CUDA Device 0"
    except Exception as e:
        info["error"] = str(e)

    _GPU_INFO = info
    return info


def init_cuda_device() -> bool:
    """Initialize CUDA device for Warp kernels.

    Returns:
        True if CUDA is available and initialized, False otherwise.
    """
    if not WARP_AVAILABLE or wp is None:
        return False

    try:
        wp.init()
        devices = wp.get_cuda_devices()
        if len(devices) > 0:
            # Set default device to first CUDA device
            wp.set_device("cuda:0")
            return True
    except Exception:
        pass

    return False


__all__ = [
    # Availability flag
    "WARP_AVAILABLE",
    # GPU utilities
    "get_gpu_info",
    "init_cuda_device",
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
    "generate_fragments_np",
    # Screen
    "screen_passage_np",
    "screen_passage_warp",
    "apply_screen_discharge",
    # Reagglomeration
    "reagglomeration_step_np",
]
