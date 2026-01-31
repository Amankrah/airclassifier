"""
Multi-cyclone system for staged particle collection.

In protein separation from legumes, multiple cyclones are used:
- Primary cyclone: Collects coarse fraction (starch-rich)
- Secondary cyclone: Collects medium fraction
- Tertiary cyclone: Collects fine fraction (protein-rich)

Different cyclone sizes provide different cut points (d50).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np
import warp as wp

from .cyclone_body import CycloneBody, CycloneBodyParams
from .inlet import TangentialInlet, InletParams
from .vortex_finder import VortexFinder, VortexFinderParams
from .dust_outlet import DustOutlet, DustOutletParams
from ...utils.constants import PI


@dataclass
class CycloneStageParams:
    """Parameters for a single cyclone stage."""

    name: str                    # Stage name (e.g., "primary", "secondary")
    diameter: float              # [m] Cyclone diameter
    design_d50: float           # [m] Target d50 cut size
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Position offset

    # Optional geometry overrides (if None, use standard proportions)
    cylinder_height_ratio: float = 1.5     # Height/D ratio
    cone_height_ratio: float = 2.5         # Cone height/D ratio
    inlet_width_ratio: float = 0.25        # Inlet width/D
    inlet_height_ratio: float = 0.5        # Inlet height/D
    vortex_finder_ratio: float = 0.5       # VF diameter/D
    dust_outlet_ratio: float = 0.375       # Dust outlet/D


@dataclass
class SimpleCycloneParams:
    """Simplified cyclone parameters for multi-cyclone system."""

    cylinder_diameter: float
    cylinder_height: float
    cone_height: float
    cone_tip_diameter: float
    inlet_width: float
    inlet_height: float
    vortex_finder_diameter: float
    vortex_finder_length: float
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    resolution: int = 48

    @property
    def total_height(self) -> float:
        """Total height from top to cone tip."""
        return self.cylinder_height + self.cone_height


class SimpleCyclone:
    """
    Simplified cyclone for use in multi-cyclone systems.

    Avoids circular import with assembly module by using components directly.
    """

    def __init__(self, params: SimpleCycloneParams, device: str = "cpu"):
        self.params = params
        self.device = device
        self._vertices = None
        self._indices = None
        self._create_components()

    def _create_components(self):
        """Create cyclone components."""
        p = self.params

        # Cyclone body
        self.body = CycloneBody(CycloneBodyParams(
            cylinder_diameter=p.cylinder_diameter,
            cylinder_height=p.cylinder_height,
            cone_height=p.cone_height,
            cone_tip_diameter=p.cone_tip_diameter,
            center=p.center,
            resolution_radial=p.resolution,
            resolution_axial_cylinder=p.resolution // 3,
            resolution_axial_cone=p.resolution // 2
        ))

        # Vortex finder
        self.vortex_finder = VortexFinder(VortexFinderParams(
            diameter=p.vortex_finder_diameter,
            length=p.vortex_finder_length,
            cyclone_center=p.center,
            protrusion_above=0.05,
            resolution_radial=p.resolution // 2,
            resolution_axial=p.resolution // 4
        ))

        # Dust outlet
        cone_bottom_y = p.center[1] - p.cylinder_height - p.cone_height
        self.dust_outlet = DustOutlet(DustOutletParams(
            diameter=p.cone_tip_diameter,
            length=p.cylinder_diameter * 0.2,
            cone_bottom_center=(p.center[0], cone_bottom_y, p.center[2]),
            resolution_radial=p.resolution // 2,
            resolution_axial=p.resolution // 6
        ))

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build combined mesh."""
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Body mesh
        body_verts, body_idx, _ = self.body.generate_mesh()
        all_vertices.append(body_verts)
        all_indices.append(body_idx + vertex_offset)
        vertex_offset += len(body_verts)

        # Vortex finder mesh
        vf_verts, vf_idx, _ = self.vortex_finder.generate_mesh()
        all_vertices.append(vf_verts)
        all_indices.append(vf_idx + vertex_offset)
        vertex_offset += len(vf_verts)

        # Dust outlet mesh
        do_verts, do_idx, _ = self.dust_outlet.generate_mesh()
        all_vertices.append(do_verts)
        all_indices.append(do_idx + vertex_offset)

        self._vertices = np.vstack(all_vertices).astype(np.float32)
        self._indices = np.concatenate(all_indices).astype(np.int32)

        return self._vertices, self._indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        if self._vertices is None:
            self.build_mesh()

        return self._vertices.min(axis=0), self._vertices.max(axis=0)


@dataclass
class MultiCycloneParams:
    """Parameters for multi-cyclone system."""

    # Cyclone stages
    stages: List[CycloneStageParams] = field(default_factory=list)

    # Connection configuration
    arrangement: str = "series"  # "series" or "parallel"

    # Spacing between cyclones
    spacing: float = 0.5         # [m] Minimum spacing between cyclone centers

    # Base position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 48

    def __post_init__(self):
        if not self.stages:
            raise ValueError("At least one cyclone stage must be defined")

    @property
    def num_stages(self) -> int:
        """Number of cyclone stages."""
        return len(self.stages)


class MultiCycloneSystem:
    """
    Multi-cyclone collection system with multiple stages.

    Typical configuration for protein separation:
    - Primary (large): d50 ~ 30-50 um -> Starch fraction
    - Secondary (medium): d50 ~ 15-25 um -> Mixed fraction
    - Tertiary (small): d50 ~ 5-15 um -> Protein fraction

    Coordinate system:
    - Origin at system center
    - Cyclones arranged along X-axis (series) or in grid (parallel)
    - Y-axis vertical (cyclone axis)
    """

    def __init__(self, params: MultiCycloneParams):
        """
        Initialize multi-cyclone system.

        Args:
            params: MultiCycloneParams defining the system
        """
        self.params = params
        self._cyclones: Dict[str, SimpleCyclone] = {}
        self._vertices = None
        self._indices = None
        self._normals = None

        # Create individual cyclones
        self._create_cyclones()

    def _create_cyclones(self):
        """Create individual cyclone assemblies."""
        p = self.params

        # Calculate positions based on arrangement
        positions = self._calculate_positions()

        for i, stage in enumerate(p.stages):
            # Create cyclone geometry params
            D = stage.diameter
            geo_params = SimpleCycloneParams(
                cylinder_diameter=D,
                cylinder_height=D * stage.cylinder_height_ratio,
                cone_height=D * stage.cone_height_ratio,
                cone_tip_diameter=D * stage.dust_outlet_ratio,
                inlet_width=D * stage.inlet_width_ratio,
                inlet_height=D * stage.inlet_height_ratio,
                vortex_finder_diameter=D * stage.vortex_finder_ratio,
                vortex_finder_length=D * stage.vortex_finder_ratio,
                center=positions[i],
                resolution=p.resolution
            )

            # Create cyclone
            cyclone = SimpleCyclone(geo_params, device="cpu")
            self._cyclones[stage.name] = cyclone

    def _calculate_positions(self) -> List[Tuple[float, float, float]]:
        """Calculate positions for each cyclone stage."""
        p = self.params
        positions = []

        if p.arrangement == "series":
            # Arrange along X-axis
            total_width = sum(s.diameter for s in p.stages) + p.spacing * (len(p.stages) - 1)
            x_start = p.center[0] - total_width / 2

            x_current = x_start
            for stage in p.stages:
                x_pos = x_current + stage.diameter / 2
                positions.append((x_pos, p.center[1], p.center[2]))
                x_current += stage.diameter + p.spacing

        elif p.arrangement == "parallel":
            # Arrange in a row along Z-axis
            total_depth = sum(s.diameter for s in p.stages) + p.spacing * (len(p.stages) - 1)
            z_start = p.center[2] - total_depth / 2

            z_current = z_start
            for stage in p.stages:
                z_pos = z_current + stage.diameter / 2
                positions.append((p.center[0], p.center[1], z_pos))
                z_current += stage.diameter + p.spacing

        else:
            # Custom positions from stage params
            for stage in p.stages:
                pos = (
                    p.center[0] + stage.position[0],
                    p.center[1] + stage.position[1],
                    p.center[2] + stage.position[2]
                )
                positions.append(pos)

        return positions

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate combined mesh for all cyclones.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        for name, cyclone in self._cyclones.items():
            verts, idx = cyclone.build_mesh()

            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        self._vertices = np.vstack(all_vertices).astype(np.float32)
        self._indices = np.concatenate(all_indices).astype(np.int32)

        # Generate simple normals (pointing outward radially)
        self._normals = np.zeros_like(self._vertices)
        for i, v in enumerate(self._vertices):
            r = np.sqrt(v[0] ** 2 + v[2] ** 2)
            if r > 1e-6:
                self._normals[i] = [v[0] / r, 0, v[2] / r]
            else:
                self._normals[i] = [0, 1, 0]

        return self._vertices, self._indices, self._normals

    def get_cyclone(self, name: str) -> SimpleCyclone:
        """
        Get a specific cyclone by name.

        Args:
            name: Cyclone stage name

        Returns:
            SimpleCyclone for the specified stage
        """
        if name not in self._cyclones:
            raise KeyError(f"Cyclone '{name}' not found. Available: {list(self._cyclones.keys())}")
        return self._cyclones[name]

    def get_all_cyclones(self) -> Dict[str, SimpleCyclone]:
        """Get dictionary of all cyclones."""
        return self._cyclones

    def get_system_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of entire system.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        all_min = []
        all_max = []

        for cyclone in self._cyclones.values():
            min_c, max_c = cyclone.get_bounds()
            all_min.append(min_c)
            all_max.append(max_c)

        return np.min(all_min, axis=0), np.max(all_max, axis=0)

    def get_stage_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each stage.

        Returns:
            List of dictionaries with stage information
        """
        info = []
        for stage in self.params.stages:
            cyclone = self._cyclones[stage.name]
            info.append({
                'name': stage.name,
                'diameter': stage.diameter * 1000,  # mm
                'design_d50': stage.design_d50 * 1e6,  # μm
                'total_height': cyclone.params.total_height * 1000,  # mm
                'inlet_area': (cyclone.params.inlet_width *
                              cyclone.params.inlet_height) * 1e6,  # mm²
            })
        return info

    def print_summary(self):
        """Print summary of multi-cyclone system."""
        print("=" * 60)
        print("Multi-Cyclone System Summary")
        print("=" * 60)
        print(f"Number of stages: {self.params.num_stages}")
        print(f"Arrangement: {self.params.arrangement}")
        print("-" * 60)

        for stage_info in self.get_stage_info():
            print(f"\n{stage_info['name'].upper()} CYCLONE:")
            print(f"  Diameter:      {stage_info['diameter']:.1f} mm")
            print(f"  Design d50:    {stage_info['design_d50']:.1f} um")
            print(f"  Total height:  {stage_info['total_height']:.1f} mm")
            print(f"  Inlet area:    {stage_info['inlet_area']:.1f} mm²")

        min_b, max_b = self.get_system_bounds()
        extent = max_b - min_b
        print("-" * 60)
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print("=" * 60)

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the multi-cyclone geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices

    @property
    def indices(self) -> np.ndarray:
        """Get mesh triangle indices."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices


def create_protein_separation_cyclones(
    primary_diameter: float = 0.4,
    secondary_diameter: float = 0.25,
    tertiary_diameter: float = 0.15
) -> MultiCycloneSystem:
    """
    Create a multi-cyclone system optimized for legume protein separation.

    Args:
        primary_diameter: Primary cyclone diameter [m]
        secondary_diameter: Secondary cyclone diameter [m]
        tertiary_diameter: Tertiary cyclone diameter [m]

    Returns:
        MultiCycloneSystem configured for protein separation
    """
    stages = [
        CycloneStageParams(
            name="primary",
            diameter=primary_diameter,
            design_d50=40e-6,  # 40 μm - coarse starch
        ),
        CycloneStageParams(
            name="secondary",
            diameter=secondary_diameter,
            design_d50=20e-6,  # 20 μm - medium
        ),
        CycloneStageParams(
            name="tertiary",
            diameter=tertiary_diameter,
            design_d50=10e-6,  # 10 μm - fine protein
        ),
    ]

    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=0.3,
    )

    return MultiCycloneSystem(params)


def create_two_stage_cyclones(
    coarse_diameter: float = 0.3,
    fine_diameter: float = 0.2
) -> MultiCycloneSystem:
    """
    Create a two-stage cyclone system (coarse + fine).

    Args:
        coarse_diameter: Coarse cyclone diameter [m]
        fine_diameter: Fine cyclone diameter [m]

    Returns:
        MultiCycloneSystem with two stages
    """
    stages = [
        CycloneStageParams(
            name="coarse",
            diameter=coarse_diameter,
            design_d50=30e-6,  # 30 μm
        ),
        CycloneStageParams(
            name="fine",
            diameter=fine_diameter,
            design_d50=15e-6,  # 15 μm
        ),
    ]

    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=0.2,
    )

    return MultiCycloneSystem(params)
