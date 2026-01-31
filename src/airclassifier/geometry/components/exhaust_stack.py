"""
Exhaust stack components for air classification systems.

This module provides exhaust stack/chimney geometries for
safe discharge of process air to atmosphere.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class ExhaustStackParams:
    """
    Parameters for exhaust stack/chimney.
    
    Attributes:
        diameter: Stack diameter [m]
        height: Stack height [m]
        wall_thickness: Wall thickness [m]
        rain_cap: Whether to include rain cap
        cap_type: Cap type ("conical", "chinese_hat", "H_cap", "none")
        discharge_velocity: Target exit velocity [m/s]
        base_flange: Whether to include base mounting flange
        guy_wire_lugs: Whether to include guy wire attachment points
        access_door: Whether to include access/cleanout door
        center: Base center position (x, y, z) [m]
    """
    diameter: float = 0.3
    height: float = 3.0
    wall_thickness: float = 0.003
    rain_cap: bool = True
    cap_type: str = "conical"
    discharge_velocity: float = 15.0
    base_flange: bool = True
    flange_diameter: float = None
    flange_thickness: float = 0.015
    guy_wire_lugs: bool = False
    num_guy_wires: int = 3
    access_door: bool = False
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def __post_init__(self):
        """Set flange diameter if not specified."""
        if self.flange_diameter is None:
            self.flange_diameter = self.diameter + 0.15
    
    @property
    def radius(self) -> float:
        """Stack radius [m]."""
        return self.diameter / 2
    
    @property
    def inner_radius(self) -> float:
        """Stack inner radius [m]."""
        return self.radius - self.wall_thickness
    
    @property
    def cross_sectional_area(self) -> float:
        """Stack cross-sectional area [m²]."""
        return np.pi * self.inner_radius ** 2
    
    def get_flow_rate_for_velocity(self, velocity: float = None) -> float:
        """
        Get volumetric flow rate for design velocity [m³/s].
        
        Args:
            velocity: Exit velocity [m/s], defaults to discharge_velocity
            
        Returns:
            Flow rate [m³/s]
        """
        if velocity is None:
            velocity = self.discharge_velocity
        return self.cross_sectional_area * velocity


class ExhaustStack:
    """
    Exhaust stack/chimney geometry.
    
    Generates mesh for vertical exhaust stacks with various
    cap types and accessories.
    """
    
    def __init__(self, params: ExhaustStackParams):
        """
        Initialize exhaust stack.
        
        Args:
            params: Stack parameters
        """
        self.params = params
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get mesh indices."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices
    
    @property
    def normals(self) -> np.ndarray:
        """Get mesh normals."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals
    
    def generate_mesh(self, num_segments: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the exhaust stack.
        
        Args:
            num_segments: Circumferential segments
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        base = np.array(p.center)
        
        # Main stack cylinder
        self._add_stack_cylinder(all_vertices, all_indices, all_normals,
                                base, num_segments)
        
        # Base flange if specified
        if p.base_flange:
            self._add_base_flange(all_vertices, all_indices, all_normals,
                                 base, num_segments)
        
        # Rain cap if specified
        if p.rain_cap:
            self._add_rain_cap(all_vertices, all_indices, all_normals,
                              base, num_segments)
        
        # Guy wire lugs if specified
        if p.guy_wire_lugs:
            self._add_guy_wire_lugs(all_vertices, all_indices, all_normals,
                                   base, num_segments)
        
        # Access door if specified
        if p.access_door:
            self._add_access_door(all_vertices, all_indices, all_normals,
                                 base, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_stack_cylinder(self, all_vertices, all_indices, all_normals,
                            base, num_segments):
        """Add main stack cylinder."""
        p = self.params
        base_idx = len(all_vertices)
        
        for t in [0, 1]:
            z = base[2] + t * p.height
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = base[0] + p.radius * np.cos(theta)
                y = base[1] + p.radius * np.sin(theta)
                all_vertices.append([x, y, z])
                n = [np.cos(theta), np.sin(theta), 0]
                all_normals.append(n)
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_base_flange(self, all_vertices, all_indices, all_normals,
                         base, num_segments):
        """Add base mounting flange."""
        p = self.params
        flange_r = p.flange_diameter / 2
        
        # Flange outer ring
        base_idx = len(all_vertices)
        for t in [0, p.flange_thickness]:
            z = base[2] + t
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = base[0] + flange_r * np.cos(theta)
                y = base[1] + flange_r * np.sin(theta)
                all_vertices.append([x, y, z])
                n = [np.cos(theta), np.sin(theta), 0]
                all_normals.append(n)
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
        
        # Flange top face (annular)
        top_base = len(all_vertices)
        z_top = base[2] + p.flange_thickness
        
        for radius in [p.radius, flange_r]:
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = base[0] + radius * np.cos(theta)
                y = base[1] + radius * np.sin(theta)
                all_vertices.append([x, y, z_top])
                all_normals.append([0, 0, 1])
        
        for i in range(num_segments):
            i0 = top_base + i
            i1 = top_base + (i + 1) % num_segments
            i2 = top_base + num_segments + i
            i3 = top_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i1, i2])
            all_indices.extend([i1, i3, i2])
    
    def _add_rain_cap(self, all_vertices, all_indices, all_normals,
                      base, num_segments):
        """Add rain cap at top of stack."""
        p = self.params
        cap_z = base[2] + p.height
        
        if p.cap_type == "conical":
            self._add_conical_cap(all_vertices, all_indices, all_normals,
                                 base, cap_z, num_segments)
        elif p.cap_type == "chinese_hat":
            self._add_chinese_hat_cap(all_vertices, all_indices, all_normals,
                                     base, cap_z, num_segments)
        elif p.cap_type == "H_cap":
            self._add_h_cap(all_vertices, all_indices, all_normals,
                          base, cap_z, num_segments)
    
    def _add_conical_cap(self, all_vertices, all_indices, all_normals,
                         base, cap_z, num_segments):
        """Add conical rain cap."""
        p = self.params
        cap_radius = p.radius * 1.3
        cap_height = p.radius * 0.8
        standoff = p.radius * 0.3  # Gap for air exit
        
        base_idx = len(all_vertices)
        
        # Cone apex
        apex = [base[0], base[1], cap_z + standoff + cap_height]
        all_vertices.append(apex)
        all_normals.append([0, 0, 1])
        
        # Cone base ring
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            x = base[0] + cap_radius * np.cos(theta)
            y = base[1] + cap_radius * np.sin(theta)
            all_vertices.append([x, y, cap_z + standoff])
            # Normal for cone surface
            n_horiz = cap_height / np.sqrt(cap_height**2 + cap_radius**2)
            n_vert = cap_radius / np.sqrt(cap_height**2 + cap_radius**2)
            all_normals.append([n_horiz * np.cos(theta), n_horiz * np.sin(theta), n_vert])
        
        # Cone triangles
        for i in range(num_segments):
            all_indices.extend([base_idx, base_idx + 1 + i, base_idx + 1 + (i + 1) % num_segments])
        
        # Support brackets (3 simple struts)
        for bracket_idx in range(3):
            angle = 2 * np.pi * bracket_idx / 3
            bracket_base = len(all_vertices)
            
            # Vertical strut
            x = base[0] + p.radius * 0.9 * np.cos(angle)
            y = base[1] + p.radius * 0.9 * np.sin(angle)
            
            all_vertices.append([x, y, cap_z])
            all_vertices.append([x, y, cap_z + standoff])
            all_normals.append([np.cos(angle), np.sin(angle), 0])
            all_normals.append([np.cos(angle), np.sin(angle), 0])
    
    def _add_chinese_hat_cap(self, all_vertices, all_indices, all_normals,
                             base, cap_z, num_segments):
        """Add Chinese hat style cap."""
        p = self.params
        cap_radius = p.radius * 1.5
        cap_drop = p.radius * 0.4
        standoff = p.radius * 0.4
        
        base_idx = len(all_vertices)
        
        # Top disc
        top_z = cap_z + standoff
        all_vertices.append([base[0], base[1], top_z])
        all_normals.append([0, 0, 1])
        
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            x = base[0] + cap_radius * np.cos(theta)
            y = base[1] + cap_radius * np.sin(theta)
            all_vertices.append([x, y, top_z - cap_drop])
            all_normals.append([0, 0, 1])
        
        for i in range(num_segments):
            all_indices.extend([base_idx, base_idx + 1 + i, base_idx + 1 + (i + 1) % num_segments])
    
    def _add_h_cap(self, all_vertices, all_indices, all_normals,
                   base, cap_z, num_segments):
        """Add H-cap style cap."""
        p = self.params
        h_width = p.radius * 3
        h_height = p.radius * 2
        standoff = p.radius * 0.3
        
        base_idx = len(all_vertices)
        
        # Simplified H-cap as horizontal cylinder
        for t in [0, 1]:
            cx = base[0] + (t - 0.5) * h_width
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                y = base[1] + p.radius * np.cos(theta)
                z = cap_z + standoff + p.radius + p.radius * np.sin(theta)
                all_vertices.append([cx, y, z])
                n = [0, np.cos(theta), np.sin(theta)]
                all_normals.append(n)
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_guy_wire_lugs(self, all_vertices, all_indices, all_normals,
                           base, num_segments):
        """Add guy wire attachment lugs."""
        p = self.params
        lug_z = base[2] + p.height * 0.8
        lug_size = 0.05
        
        for i in range(p.num_guy_wires):
            angle = 2 * np.pi * i / p.num_guy_wires
            lug_base = len(all_vertices)
            
            cx = base[0] + (p.radius + 0.01) * np.cos(angle)
            cy = base[1] + (p.radius + 0.01) * np.sin(angle)
            
            # Simple triangular lug
            v0 = [cx, cy, lug_z]
            v1 = [cx + lug_size * np.cos(angle), cy + lug_size * np.sin(angle), lug_z - lug_size/2]
            v2 = [cx + lug_size * np.cos(angle), cy + lug_size * np.sin(angle), lug_z + lug_size/2]
            
            for v in [v0, v1, v2]:
                all_vertices.append(v)
                all_normals.append([np.cos(angle), np.sin(angle), 0])
            
            all_indices.extend([lug_base, lug_base + 1, lug_base + 2])
    
    def _add_access_door(self, all_vertices, all_indices, all_normals,
                         base, num_segments):
        """Add access/cleanout door."""
        p = self.params
        door_z = base[2] + 0.3
        door_height = 0.4
        door_width = 0.3
        door_angle = 0  # Front
        
        base_idx = len(all_vertices)
        
        # Door frame (simplified as rectangle)
        half_w = door_width / 2
        for z_off in [0, door_height]:
            for w_off in [-half_w, half_w]:
                theta = door_angle + w_off / p.radius
                x = base[0] + (p.radius + 0.005) * np.cos(theta)
                y = base[1] + (p.radius + 0.005) * np.sin(theta)
                z = door_z + z_off
                all_vertices.append([x, y, z])
                all_normals.append([np.cos(door_angle), np.sin(door_angle), 0])
        
        all_indices.extend([base_idx, base_idx + 1, base_idx + 3])
        all_indices.extend([base_idx, base_idx + 3, base_idx + 2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_exit_velocity(self, flow_rate: float) -> float:
        """
        Calculate exit velocity for given flow rate.
        
        Args:
            flow_rate: Volumetric flow rate [m³/s]
            
        Returns:
            Exit velocity [m/s]
        """
        return flow_rate / self.params.cross_sectional_area
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_standard_exhaust_stack(diameter: float = 0.3,
                                  height: float = 3.0,
                                  cap_type: str = "conical",
                                  **kwargs) -> ExhaustStack:
    """
    Create a standard exhaust stack.
    
    Args:
        diameter: Stack diameter [m]
        height: Stack height [m]
        cap_type: Cap type
        **kwargs: Additional parameters
        
    Returns:
        ExhaustStack instance
    """
    params = ExhaustStackParams(
        diameter=diameter,
        height=height,
        cap_type=cap_type,
        rain_cap=True,
        **kwargs
    )
    return ExhaustStack(params)


def create_tall_stack(diameter: float = 0.4,
                      height: float = 10.0,
                      **kwargs) -> ExhaustStack:
    """
    Create a tall exhaust stack with guy wires.
    
    Args:
        diameter: Stack diameter [m]
        height: Stack height [m]
        **kwargs: Additional parameters
        
    Returns:
        ExhaustStack instance
    """
    params = ExhaustStackParams(
        diameter=diameter,
        height=height,
        cap_type="conical",
        rain_cap=True,
        guy_wire_lugs=True,
        num_guy_wires=4,
        wall_thickness=0.005,
        **kwargs
    )
    return ExhaustStack(params)


def create_short_vent_stack(diameter: float = 0.25,
                            height: float = 1.5,
                            **kwargs) -> ExhaustStack:
    """
    Create a short vent stack.
    
    Args:
        diameter: Stack diameter [m]
        height: Stack height [m]
        **kwargs: Additional parameters
        
    Returns:
        ExhaustStack instance
    """
    params = ExhaustStackParams(
        diameter=diameter,
        height=height,
        cap_type="chinese_hat",
        rain_cap=True,
        base_flange=True,
        guy_wire_lugs=False,
        **kwargs
    )
    return ExhaustStack(params)
