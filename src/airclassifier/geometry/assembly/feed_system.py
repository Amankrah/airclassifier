"""
Feed system assembly module.

Provides complete feed system assembly combining Phase 2 components:
- Feed Hopper
- Rotary Airlock
- Screw Feeder
- De-agglomerator
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import warp as wp


@dataclass
class FeedSystemParams:
    """
    Parameters for complete feed system.

    Combines all Phase 2 components into a feed preparation system.
    """

    # Feed hopper parameters
    hopper_capacity_kg: float = 500       # [kg]
    hopper_discharge_diameter: float = 0.15  # [m]

    # Rotary airlock parameters
    airlock_rotor_diameter: float = 0.20  # [m]

    # Screw feeder parameters
    feeder_screw_diameter: float = 0.10   # [m]
    feeder_target_rate_kg_h: float = 500  # [kg/h]

    # De-agglomerator parameters
    deagg_rotor_diameter: float = 0.20    # [m]
    deagg_screen_aperture: float = 0.002  # [m] (2mm)

    # Layout parameters
    component_spacing: float = 0.1        # [m] Spacing between components
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Material properties for sizing
    bulk_density: float = 500.0           # [kg/m3]


class FeedSystemAssembly:
    """
    Complete feed system assembly.

    Combines all Phase 2 components:
    - Feed Hopper: Powder storage
    - Rotary Airlock: Pressure seal
    - Screw Feeder: Controlled dosing
    - De-agglomerator: Lump breaking

    Process flow:
    Hopper -> Airlock -> Feeder -> De-agglomerator -> Classifier

    Coordinate system:
    - Origin at center of system
    - Y-axis: Vertical (up, gravity direction)
    - Flow direction: primarily gravity-fed (downward)
    """

    def __init__(self, params: FeedSystemParams = None, device: str = "cpu"):
        """
        Initialize feed system assembly.

        Args:
            params: FeedSystemParams (uses defaults if None)
            device: Warp device for mesh operations
        """
        self.params = params or FeedSystemParams()
        self.device = device

        # Create components
        self._create_components()

        # Mesh data
        self._combined_vertices = None
        self._combined_indices = None
        self._mesh_built = False

    def _create_components(self):
        """Create all system components with proper positioning."""
        # Lazy imports to avoid circular dependency
        from ..components import (
            create_standard_feed_hopper,
            create_standard_rotary_airlock,
            create_standard_screw_feeder,
            create_standard_deagglomerator,
        )

        p = self.params
        spacing = p.component_spacing

        # Track Y position (vertical, top to bottom)
        y_pos = p.center[1]

        # 1. Feed Hopper (top of system)
        self.hopper = create_standard_feed_hopper(
            capacity_kg=p.hopper_capacity_kg,
            bulk_density=p.bulk_density,
            discharge_diameter=p.hopper_discharge_diameter
        )
        # Position hopper with discharge at current y_pos
        hopper_height = self.hopper.params.total_height
        self._hopper_position = (p.center[0], y_pos + hopper_height, p.center[2])
        y_pos -= spacing

        # 2. Rotary Airlock (under hopper)
        self.airlock = create_standard_rotary_airlock(
            rotor_diameter=p.airlock_rotor_diameter
        )
        airlock_height = self.airlock.params.housing_outer_radius * 2
        self._airlock_position = (p.center[0], y_pos, p.center[2])
        y_pos -= airlock_height + spacing

        # 3. Screw Feeder (horizontal, after airlock)
        self.feeder = create_standard_screw_feeder(
            screw_diameter=p.feeder_screw_diameter,
            feed_rate_kg_h=p.feeder_target_rate_kg_h,
            bulk_density=p.bulk_density
        )
        self._feeder_position = (p.center[0], y_pos, p.center[2])
        # Feeder extends along X axis
        feeder_length = self.feeder.params.trough_length
        x_end = p.center[0] + feeder_length

        # 4. De-agglomerator (at end of feeder)
        self.deagglomerator = create_standard_deagglomerator(
            rotor_diameter=p.deagg_rotor_diameter,
            screen_aperture=p.deagg_screen_aperture
        )
        deagg_height = self.deagglomerator.params.housing_radius * 2
        self._deagglomerator_position = (x_end + spacing, y_pos - deagg_height / 2, p.center[2])

    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh for all components.

        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Helper to add component mesh with position offset
        def add_component_mesh(component, position):
            nonlocal vertex_offset
            verts, idx, _ = component.generate_mesh()

            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset

            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        # Add each component
        add_component_mesh(self.hopper, self._hopper_position)
        add_component_mesh(self.airlock, self._airlock_position)
        add_component_mesh(self.feeder, self._feeder_position)
        add_component_mesh(self.deagglomerator, self._deagglomerator_position)

        self._combined_vertices = np.vstack(all_vertices).astype(np.float32)
        self._combined_indices = np.concatenate(all_indices).astype(np.int32)
        self._mesh_built = True

        return self._combined_vertices, self._combined_indices

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the entire system.

        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        if not self._mesh_built:
            self.build_mesh()

        min_corner = self._combined_vertices.min(axis=0)
        max_corner = self._combined_vertices.max(axis=0)

        return min_corner, max_corner

    def get_system_extent(self) -> np.ndarray:
        """
        Get system extent (dimensions) in each axis.

        Returns:
            Array of [width, height, depth]
        """
        min_c, max_c = self.get_bounds()
        return max_c - min_c

    def get_component(self, name: str) -> Any:
        """
        Get a specific component by name.

        Args:
            name: Component name ('hopper', 'airlock', 'feeder', 'deagglomerator')

        Returns:
            Component instance
        """
        components = {
            'hopper': self.hopper,
            'airlock': self.airlock,
            'feeder': self.feeder,
            'deagglomerator': self.deagglomerator,
        }
        if name not in components:
            raise KeyError(f"Unknown component: {name}. Available: {list(components.keys())}")
        return components[name]

    def get_component_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get positions of all components.

        Returns:
            Dictionary of component names to positions
        """
        return {
            'hopper': self._hopper_position,
            'airlock': self._airlock_position,
            'feeder': self._feeder_position,
            'deagglomerator': self._deagglomerator_position,
        }

    def get_feed_rate(self, rpm: float = None) -> float:
        """
        Get feed rate based on feeder settings.

        Args:
            rpm: Feeder RPM (uses default if None)

        Returns:
            Feed rate [kg/h]
        """
        return self.feeder.get_feed_rate(rpm=rpm, bulk_density=self.params.bulk_density)

    def to_warp_mesh(self) -> wp.Mesh:
        """
        Create a Warp mesh from the system geometry.

        Returns:
            wp.Mesh object
        """
        if not self._mesh_built:
            self.build_mesh()

        points = wp.array(self._combined_vertices, dtype=wp.vec3, device=self.device)
        indices = wp.array(self._combined_indices, dtype=wp.int32, device=self.device)

        return wp.Mesh(points=points, indices=indices)

    def print_summary(self):
        """Print summary of the feed system."""
        p = self.params

        print("=" * 60)
        print("Feed System Assembly Summary")
        print("=" * 60)

        print("\n1. FEED HOPPER")
        print(f"   Capacity:        {p.hopper_capacity_kg:.0f} kg")
        print(f"   Top diameter:    {self.hopper.params.top_diameter * 1000:.0f} mm")
        print(f"   Discharge dia:   {p.hopper_discharge_diameter * 1000:.0f} mm")
        print(f"   Total height:    {self.hopper.params.total_height * 1000:.0f} mm")

        print("\n2. ROTARY AIRLOCK")
        print(f"   Rotor diameter:  {p.airlock_rotor_diameter * 1000:.0f} mm")
        print(f"   Vanes:           {self.airlock.params.num_vanes}")
        print(f"   Capacity:        {self.airlock.params.capacity_kg_h(p.bulk_density):.0f} kg/h")

        print("\n3. SCREW FEEDER")
        print(f"   Screw diameter:  {p.feeder_screw_diameter * 1000:.0f} mm")
        print(f"   Trough length:   {self.feeder.params.trough_length * 1000:.0f} mm")
        print(f"   Design rate:     {p.feeder_target_rate_kg_h:.0f} kg/h")

        print("\n4. DE-AGGLOMERATOR")
        print(f"   Rotor diameter:  {p.deagg_rotor_diameter * 1000:.0f} mm")
        print(f"   Screen aperture: {p.deagg_screen_aperture * 1000:.1f} mm")
        print(f"   Tip speed:       {self.deagglomerator.get_tip_speed():.1f} m/s")

        print("-" * 60)
        extent = self.get_system_extent()
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")

        if self._mesh_built:
            n_verts = len(self._combined_vertices)
            n_tris = len(self._combined_indices) // 3
            print(f"Total mesh:    {n_verts} vertices, {n_tris} triangles")
        print("=" * 60)

    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_vertices

    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if not self._mesh_built:
            self.build_mesh()
        return self._combined_indices


def create_standard_feed_system(device: str = "cpu") -> FeedSystemAssembly:
    """
    Create a standard feed system with default parameters.

    Args:
        device: Warp device

    Returns:
        FeedSystemAssembly instance
    """
    return FeedSystemAssembly(device=device)


def create_feed_system_for_throughput(
    throughput_kg_h: float = 500,
    device: str = "cpu"
) -> FeedSystemAssembly:
    """
    Create a feed system sized for given throughput.

    Args:
        throughput_kg_h: Design throughput [kg/h]
        device: Warp device

    Returns:
        FeedSystemAssembly configured for given throughput
    """
    # Scale parameters based on throughput
    scale = (throughput_kg_h / 500) ** 0.5  # Square root scaling

    params = FeedSystemParams(
        hopper_capacity_kg=throughput_kg_h * 1.0,  # 1 hour buffer
        hopper_discharge_diameter=0.15 * scale,
        airlock_rotor_diameter=0.20 * scale,
        feeder_screw_diameter=0.10 * scale,
        feeder_target_rate_kg_h=throughput_kg_h,
        deagg_rotor_diameter=0.20 * scale,
        deagg_screen_aperture=0.002,  # Fixed screen size
    )

    return FeedSystemAssembly(params, device=device)
