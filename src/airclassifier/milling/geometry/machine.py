"""
Hammer Mill Machine Factory
===========================

High-level factory function for creating hammer mill machines.
Provides a simple interface for building complete mill assemblies.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .assembly import (
    HammerMillMachineAssembly,
    build_hammer_mill_meshes,
    create_hammer_mill_assembly,
    COMPONENT_COLORS,
)
from ..config import MillConfig


def create_hammer_mill_machine(
    config: Optional[MillConfig] = None,
    build_meshes: bool = True,
    resolution: int = 24,
) -> HammerMillMachineAssembly:
    """Create a complete hammer mill machine.

    This is the main entry point for creating hammer mill geometry.
    It creates the assembly with all components and optionally builds
    the meshes.

    Args:
        config: Mill configuration. Uses defaults if None.
        build_meshes: If True, build meshes immediately.
        resolution: Angular resolution for cylindrical surfaces.

    Returns:
        Configured HammerMillMachineAssembly with meshes built.

    Example:
        >>> config = MillConfig(rotor_rpm=3500, screen_aperture_mm=2.0)
        >>> assembly = create_hammer_mill_machine(config)
        >>> meshes = assembly.get_component_meshes()
        >>> for name, (verts, tris, meta) in meshes.items():
        ...     print(f"{name}: {len(verts)} vertices")
    """
    assembly = create_hammer_mill_assembly(config)

    if build_meshes:
        assembly.build_meshes(resolution)

    return assembly


def get_default_hammer_mill_meshes() -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    """Get meshes for a default hammer mill configuration.

    Convenience function for quick visualization.

    Returns:
        Dictionary mapping component name to (vertices, triangles, metadata).
    """
    return build_hammer_mill_meshes()


__all__ = [
    "create_hammer_mill_machine",
    "get_default_hammer_mill_meshes",
    "HammerMillMachineAssembly",
    "COMPONENT_COLORS",
    "MillConfig",
]
