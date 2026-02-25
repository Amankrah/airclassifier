"""
GUI Widgets for ProteinProcessIO
========================================

Custom widgets for the main window including modern KPI cards,
sparklines, gauges, and specialized milling components.
"""

from .viewport_3d import Viewport3D
from .assembly_canvas import AssemblyCanvas

# Import common widgets
try:
    from .common import (
        AnimatedKPICard,
        GlassCard,
        SparklineWidget,
        RadialGaugeWidget,
    )
    _HAS_COMMON = True
except ImportError:
    _HAS_COMMON = False

# Import milling widgets
try:
    from .milling import (
        TimelineWidget,
        ResultsOverlay,
        InteractivePSDChart,
        MillingControlPanel,
        MillingKPIDashboard,
    )
    _HAS_MILLING = True
except ImportError:
    _HAS_MILLING = False

# Import pretreatment widgets
try:
    from .pretreatment import (
        PretreatmentKPIDashboard,
        PretreatmentControlPanel,
        PretreatmentTimelineWidget,
        PretreatmentResultsOverlay,
        DesirabilityPanel,
    )
    _HAS_PRETREATMENT = True
except ImportError:
    _HAS_PRETREATMENT = False

__all__ = [
    "Viewport3D",
    "AssemblyCanvas",
]

if _HAS_COMMON:
    __all__.extend([
        "AnimatedKPICard",
        "GlassCard",
        "SparklineWidget",
        "RadialGaugeWidget",
    ])

if _HAS_MILLING:
    __all__.extend([
        "TimelineWidget",
        "ResultsOverlay",
        "InteractivePSDChart",
        "MillingControlPanel",
        "MillingKPIDashboard",
    ])

if _HAS_PRETREATMENT:
    __all__.extend([
        "PretreatmentKPIDashboard",
        "PretreatmentControlPanel",
        "PretreatmentTimelineWidget",
        "PretreatmentResultsOverlay",
        "DesirabilityPanel",
    ])
