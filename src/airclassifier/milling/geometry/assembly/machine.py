"""
Hammer Mill Machine Assembly
============================

Composes all geometry components into a complete hammer mill machine.
Provides a unified interface for building meshes and managing the
parameter chain.

The assembly:
    - Creates component params from MillConfig (parameter chain)
    - Builds all component geometries
    - Provides combined mesh output for rendering
    - Defines connection ports (infeed, outfeed)
    - Supports animation metadata for rotor/hammers

Coordinate system:
    X = along rotor axis
    Y = vertical (up)
    Z = lateral
    Origin at rotor centerline, X=0 at drive end
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..components import (
    RotorParams, RotorGeometry,
    HammerParams, HammerGeometry,
    ScreenParams, ScreenGeometry,
    HousingParams, HousingGeometry,
    FeedChuteParams, FeedChuteGeometry,
    DriveParams, DriveGeometry,
)
from ..mesh_utils import concat_meshes
from ...config import MillConfig


# Component colors for visualization (RGBA or named)
COMPONENT_COLORS = {
    "rotor": (0.6, 0.6, 0.65, 1.0),        # Steel gray
    "hammers": (0.8, 0.75, 0.3, 1.0),      # Gold/brass
    "hammer_pins": (0.55, 0.55, 0.58, 1.0), # Steel pins
    "screen": (0.5, 0.5, 0.55, 0.8),       # Dark gray, semi-transparent
    "housing": (0.45, 0.5, 0.55, 0.6),     # Blue-gray, transparent
    "housing_discharge": (0.45, 0.48, 0.52, 0.85),  # Discharge funnel
    "feed_chute": (0.5, 0.55, 0.6, 0.8),   # Light gray
    "drive": (0.3, 0.35, 0.4, 1.0),        # Dark gray
}


@dataclass
class HammerMillMachineAssembly:
    """Complete hammer mill machine assembly.

    Holds all component geometries and their parameters, built from
    a single MillConfig through the parameter chain. Dimensions are
    proportionally consistent: housing/rotor/screen from config; feed
    chute and drive from housing + config (e.g. drive pulleys and motor
    scale with rotor and power).
    """

    config: MillConfig = field(default_factory=MillConfig)

    # Component params (derived from config)
    rotor_params: RotorParams = field(default=None)
    hammer_params: HammerParams = field(default=None)
    screen_params: ScreenParams = field(default=None)
    housing_params: HousingParams = field(default=None)
    feed_chute_params: FeedChuteParams = field(default=None)
    drive_params: DriveParams = field(default=None)

    # Component geometries
    rotor_geometry: RotorGeometry = field(default=None)
    hammer_geometry: HammerGeometry = field(default=None)
    screen_geometry: ScreenGeometry = field(default=None)
    housing_geometry: HousingGeometry = field(default=None)
    feed_chute_geometry: FeedChuteGeometry = field(default=None)
    drive_geometry: DriveGeometry = field(default=None)

    # Cached meshes
    _component_meshes: Dict[str, Tuple[np.ndarray, np.ndarray, dict]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Initialize component params and geometries from config."""
        self._build_parameter_chain()
        self._create_geometries()

    def _build_parameter_chain(self):
        """Build component params from MillConfig (single source of truth)."""
        c = self.config

        # Housing first (other components position relative to it)
        if self.housing_params is None:
            self.housing_params = HousingParams.from_mill_config(c)

        # Rotor (inside housing)
        if self.rotor_params is None:
            self.rotor_params = RotorParams.from_mill_config(c)

        # Hammers (attached to rotor)
        if self.hammer_params is None:
            self.hammer_params = HammerParams.from_rotor(self.rotor_params, c)

        # Screen (inside housing, below rotor)
        if self.screen_params is None:
            self.screen_params = ScreenParams.from_mill_config(c)

        # Feed chute (on top of housing)
        if self.feed_chute_params is None:
            self.feed_chute_params = FeedChuteParams.from_housing(
                self.housing_params, c
            )

        # Drive (beside housing; dimensions proportional to config and housing)
        if self.drive_params is None:
            self.drive_params = DriveParams.from_housing(self.housing_params, c)

    def validate_proportional_dimensions(self) -> Dict[str, list]:
        """Run proportion checks on components that support it. Returns {component: [(name, ok, msg), ...]}."""
        out = {}
        if hasattr(self.drive_params, "validate_proportions"):
            out["drive"] = self.drive_params.validate_proportions()
        return out

    def _create_geometries(self):
        """Create geometry objects from params."""
        if self.rotor_geometry is None:
            self.rotor_geometry = RotorGeometry(self.rotor_params)
        if self.hammer_geometry is None:
            self.hammer_geometry = HammerGeometry(self.hammer_params)
        if self.screen_geometry is None:
            self.screen_geometry = ScreenGeometry(self.screen_params)
        if self.housing_geometry is None:
            self.housing_geometry = HousingGeometry(self.housing_params)
        if self.feed_chute_geometry is None:
            self.feed_chute_geometry = FeedChuteGeometry(self.feed_chute_params)
        if self.drive_geometry is None:
            self.drive_geometry = DriveGeometry(self.drive_params)

    def build_meshes(self, resolution: int = 24) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
        """Build all component meshes.

        Args:
            resolution: Angular resolution for cylindrical surfaces.

        Returns:
            Dictionary mapping component name to (vertices, triangles, metadata).
        """
        self._component_meshes = {}

        # Rotor
        verts, tris, meta = self.rotor_geometry.generate_mesh(resolution)
        self._component_meshes["rotor"] = (verts, tris, meta)

        # Hammers (color-coded parts: bodies + pins)
        hammer_meta = {
            "type": "hammers",
            "animation_type": "rotate",
            "animation_axis": "x",
            "pivot": (0.0, 0.0, 0.0),
        }
        for part_name, (v, t) in self.hammer_geometry.generate_mesh_parts(resolution).items():
            self._component_meshes[part_name] = (v, t, hammer_meta)

        # Screen
        verts, tris, meta = self.screen_geometry.generate_mesh(
            radial_resolution=resolution, axial_resolution=8
        )
        self._component_meshes["screen"] = (verts, tris, meta)

        # Housing (color-coded parts: shell + discharge funnel)
        housing_meta = {"type": "housing", "animation_type": None}
        for part_name, (v, t) in self.housing_geometry.generate_mesh_parts(resolution).items():
            self._component_meshes[part_name] = (v, t, housing_meta)

        # Feed chute
        verts, tris, meta = self.feed_chute_geometry.generate_mesh()
        self._component_meshes["feed_chute"] = (verts, tris, meta)

        # Drive (color-coded parts)
        hp = self.housing_params
        dp = self.drive_params
        drive_meta = {"type": "drive", "animation_type": None}
        for part_name, (v, t) in self.drive_geometry.generate_mesh_parts(resolution).items():
            self._component_meshes[part_name] = (v, t, drive_meta)

        return self._component_meshes

    def get_component_meshes(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
        """Get cached component meshes (builds if needed).

        Returns:
            Dictionary mapping component name to (vertices, triangles, metadata).
        """
        if not self._component_meshes:
            self.build_meshes()
        return self._component_meshes

    def get_combined_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all components as a single combined mesh.

        Returns:
            (combined_vertices, combined_triangles)
        """
        meshes = self.get_component_meshes()
        parts = [(m[0], m[1]) for m in meshes.values()]
        return concat_meshes(parts)

    def get_animated_components(self) -> Dict[str, dict]:
        """Get components that need animation.

        Returns:
            Dictionary mapping component name to animation metadata.
        """
        meshes = self.get_component_meshes()
        animated = {}
        for name, (_, _, meta) in meshes.items():
            if meta.get("animation_type") is not None:
                animated[name] = {
                    "animation_type": meta["animation_type"],
                    "animation_axis": meta.get("animation_axis", "x"),
                    "pivot": meta.get("pivot", (0.0, 0.0, 0.0)),
                }
        return animated

    @property
    def ports(self) -> Dict[str, Tuple[float, float, float]]:
        """Connection ports for pipeline integration.

        Returns:
            Dictionary with 'infeed_port' and 'outfeed_port' positions.
        """
        feed_ports = self.feed_chute_geometry.ports
        housing_ports = self.housing_geometry.ports

        return {
            "infeed_port": feed_ports.get("inlet", (0.0, 0.5, 0.0)),
            "outfeed_port": housing_ports.get("discharge_outlet", (0.0, -0.3, 0.0)),
        }


def build_hammer_mill_meshes(
    config: Optional[MillConfig] = None,
    resolution: int = 24,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    """Factory function to build hammer mill meshes.

    Args:
        config: Mill configuration (uses defaults if None).
        resolution: Angular resolution for cylindrical surfaces.

    Returns:
        Dictionary mapping component name to (vertices, triangles, metadata).
        Metadata includes animation info for rotor/hammers.
    """
    config = config or MillConfig()
    assembly = HammerMillMachineAssembly(config)
    return assembly.build_meshes(resolution)


def create_hammer_mill_assembly(
    config: Optional[MillConfig] = None,
) -> HammerMillMachineAssembly:
    """Factory function to create a hammer mill assembly.

    Args:
        config: Mill configuration (uses defaults if None).

    Returns:
        Configured HammerMillMachineAssembly instance.
    """
    config = config or MillConfig()
    return HammerMillMachineAssembly(config)
