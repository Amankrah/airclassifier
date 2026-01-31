"""
Assembly modules for air classification systems.

This package provides complete system assemblies:

Cyclone Assembly:
- CycloneAssembly: Single cyclone separator with all components
- CycloneGeometryParams: Cyclone geometry parameters

Classification System (Phase 1):
- ClassificationSystemAssembly: Zigzag + Cyclones + Bag Filter
- ClassificationSystemParams: Classification system parameters

Feed System (Phase 2):
- FeedSystemAssembly: Hopper + Airlock + Screw Feeder + De-agglomerator
- FeedSystemParams: Feed system parameters

Air System (Phase 3):
- AirSystemAssembly: Blower + Filter + Dampers
- AirSystemParams: Air system parameters

Ductwork System (Phase 4):
- DuctworkSystemAssembly: Ducts + Transitions + Elbows + Diverters
- DuctworkSystemParams: Ductwork system parameters

Safety & Instrumentation System (Phase 5):
- SafetyInstrumentationAssembly: Explosion vents + Grounding + Instrumentation
- SafetyInstrumentationParams: Safety/instrumentation system parameters
"""

from .cyclone import (
    CycloneAssembly,
    CycloneGeometryParams,
    create_standard_cyclone,
)

from .classification import (
    ClassificationSystemAssembly,
    ClassificationSystemParams,
    create_standard_classification_system,
    create_protein_separation_system,
)

from .feed_system import (
    FeedSystemAssembly,
    FeedSystemParams,
    create_standard_feed_system,
    create_feed_system_for_throughput,
)

from .air_system import (
    AirSystemAssembly,
    AirSystemParams,
    create_standard_air_system,
    create_air_system_for_classifier,
)

from .ductwork import (
    DuctworkSystemAssembly,
    DuctworkSystemParams,
    create_standard_ductwork,
    create_ductwork_for_classifier,
    create_simple_duct_run,
)

from .safety_instrumentation import (
    SafetyInstrumentationAssembly,
    SafetyInstrumentationParams,
    create_standard_safety_instrumentation,
    create_minimal_instrumentation,
    create_full_instrumentation,
)

__all__ = [
    # Cyclone Assembly
    "CycloneAssembly",
    "CycloneGeometryParams",
    "create_standard_cyclone",
    # Classification System (Phase 1)
    "ClassificationSystemAssembly",
    "ClassificationSystemParams",
    "create_standard_classification_system",
    "create_protein_separation_system",
    # Feed System (Phase 2)
    "FeedSystemAssembly",
    "FeedSystemParams",
    "create_standard_feed_system",
    "create_feed_system_for_throughput",
    # Air System (Phase 3)
    "AirSystemAssembly",
    "AirSystemParams",
    "create_standard_air_system",
    "create_air_system_for_classifier",
    # Ductwork System (Phase 4)
    "DuctworkSystemAssembly",
    "DuctworkSystemParams",
    "create_standard_ductwork",
    "create_ductwork_for_classifier",
    "create_simple_duct_run",
    # Safety & Instrumentation System (Phase 5)
    "SafetyInstrumentationAssembly",
    "SafetyInstrumentationParams",
    "create_standard_safety_instrumentation",
    "create_minimal_instrumentation",
    "create_full_instrumentation",
]
