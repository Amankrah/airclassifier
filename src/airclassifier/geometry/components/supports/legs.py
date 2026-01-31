"""
Equipment leg components for air classification systems.

This module provides support leg geometries for equipment mounting
including tubular legs, channel legs, and adjustable legs.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class EquipmentLegParams:
    """
    Parameters for equipment support legs.
    
    Attributes:
        leg_type: Type ("tubular", "channel", "adjustable")
        num_legs: Number of legs (typically 3 or 4)
        leg_height: Leg height [m]
        leg_diameter: Leg diameter/size [m]
        wall_thickness: Wall thickness for tubular legs [m]
        foot_type: Foot type ("flat", "leveling", "seismic")
        foot_diameter: Foot plate diameter [m]
        load_capacity: Per-leg load capacity [kg]
        mounting_diameter: Equipment mounting circle diameter [m]
        gusset_plates: Whether to include gusset plates
        center: Center position of leg assembly (x, y, z) [m]
    """
    leg_type: str = "tubular"
    num_legs: int = 4
    leg_height: float = 0.5
    leg_diameter: float = 0.076  # 3" pipe
    wall_thickness: float = 0.005
    foot_type: str = "leveling"
    foot_diameter: float = 0.150
    foot_thickness: float = 0.012
    load_capacity: float = 500.0
    mounting_diameter: float = 0.8
    gusset_plates: bool = True
    gusset_height: float = 0.15
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @property
    def leg_radius(self) -> float:
        """Leg outer radius [m]."""
        return self.leg_diameter / 2
    
    @property
    def inner_radius(self) -> float:
        """Leg inner radius for tubular [m]."""
        return self.leg_radius - self.wall_thickness
    
    @property
    def mounting_radius(self) -> float:
        """Mounting circle radius [m]."""
        return self.mounting_diameter / 2
    
    def get_leg_positions(self) -> List[Tuple[float, float]]:
        """Get (x, y) positions of each leg."""
        positions = []
        for i in range(self.num_legs):
            angle = 2 * np.pi * i / self.num_legs + np.pi / self.num_legs
            x = self.center[0] + self.mounting_radius * np.cos(angle)
            y = self.center[1] + self.mounting_radius * np.sin(angle)
            positions.append((x, y))
        return positions


class EquipmentLegs:
    """
    Equipment support legs geometry.
    
    Generates mesh for support legs with various configurations.
    """
    
    def __init__(self, params: EquipmentLegParams):
        """
        Initialize equipment legs.
        
        Args:
            params: Leg parameters
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
    
    def generate_mesh(self, num_segments: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for all legs.
        
        Args:
            num_segments: Circumferential segments
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        leg_positions = p.get_leg_positions()
        
        for leg_x, leg_y in leg_positions:
            base_idx = len(all_vertices)
            
            if p.leg_type == "tubular":
                self._add_tubular_leg(all_vertices, all_indices, all_normals,
                                     leg_x, leg_y, num_segments, base_idx)
            elif p.leg_type == "channel":
                self._add_channel_leg(all_vertices, all_indices, all_normals,
                                     leg_x, leg_y, num_segments, base_idx)
            else:  # adjustable
                self._add_adjustable_leg(all_vertices, all_indices, all_normals,
                                        leg_x, leg_y, num_segments, base_idx)
            
            # Add foot
            foot_base_idx = len(all_vertices)
            self._add_foot(all_vertices, all_indices, all_normals,
                          leg_x, leg_y, num_segments, foot_base_idx)
            
            # Add gusset if specified
            if p.gusset_plates:
                gusset_base_idx = len(all_vertices)
                self._add_gusset(all_vertices, all_indices, all_normals,
                               leg_x, leg_y, gusset_base_idx)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_tubular_leg(self, all_vertices, all_indices, all_normals,
                         leg_x, leg_y, num_segments, base_idx):
        """Add tubular leg cylinder."""
        p = self.params
        z_base = p.center[2]
        
        # Outer surface
        for t in [0, 1]:
            z = z_base + t * p.leg_height
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = leg_x + p.leg_radius * np.cos(theta)
                y = leg_y + p.leg_radius * np.sin(theta)
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
    
    def _add_channel_leg(self, all_vertices, all_indices, all_normals,
                         leg_x, leg_y, num_segments, base_idx):
        """Add channel/C-section leg."""
        p = self.params
        z_base = p.center[2]
        
        # Channel dimensions
        web = p.leg_diameter
        flange = web * 0.4
        thick = p.wall_thickness
        
        # Profile points (C-shape cross section)
        profile = [
            (flange, 0),
            (0, 0),
            (0, web),
            (flange, web),
            (flange, web - thick),
            (thick, web - thick),
            (thick, thick),
            (flange, thick),
        ]
        
        # Extrude profile
        for t in [0, 1]:
            z = z_base + t * p.leg_height
            for px, py in profile:
                x = leg_x + px - flange/2
                y = leg_y + py - web/2
                all_vertices.append([x, y, z])
                all_normals.append([0, 0, 1 if t > 0 else -1])
        
        # Create faces between profile rings
        n = len(profile)
        for i in range(n):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % n
            i2 = base_idx + n + i
            i3 = base_idx + n + (i + 1) % n
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_adjustable_leg(self, all_vertices, all_indices, all_normals,
                            leg_x, leg_y, num_segments, base_idx):
        """Add adjustable leg (threaded rod + tube)."""
        p = self.params
        z_base = p.center[2]
        
        # Upper tube section (2/3 of height)
        tube_height = p.leg_height * 0.65
        rod_height = p.leg_height * 0.35
        
        # Tube section
        for t in [0, 1]:
            z = z_base + rod_height + t * tube_height
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = leg_x + p.leg_radius * np.cos(theta)
                y = leg_y + p.leg_radius * np.sin(theta)
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
        
        # Threaded rod section (smaller diameter)
        rod_radius = p.leg_radius * 0.4
        rod_base = len(all_vertices)
        
        for t in [0, 1]:
            z = z_base + t * rod_height
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = leg_x + rod_radius * np.cos(theta)
                y = leg_y + rod_radius * np.sin(theta)
                all_vertices.append([x, y, z])
                n = [np.cos(theta), np.sin(theta), 0]
                all_normals.append(n)
        
        for i in range(num_segments):
            i0 = rod_base + i
            i1 = rod_base + (i + 1) % num_segments
            i2 = rod_base + num_segments + i
            i3 = rod_base + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_foot(self, all_vertices, all_indices, all_normals,
                  leg_x, leg_y, num_segments, base_idx):
        """Add foot plate."""
        p = self.params
        z_base = p.center[2]
        foot_r = p.foot_diameter / 2
        
        if p.foot_type == "leveling":
            # Leveling pad with central hole
            for t in [0, p.foot_thickness]:
                z = z_base - p.foot_thickness + t
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    x = leg_x + foot_r * np.cos(theta)
                    y = leg_y + foot_r * np.sin(theta)
                    all_vertices.append([x, y, z])
                    all_normals.append([0, 0, 1 if t > 0 else -1])
            
            # Outer surface
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
        else:
            # Simple flat plate
            face_base = len(all_vertices)
            all_vertices.append([leg_x, leg_y, z_base])
            all_normals.append([0, 0, -1])
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = leg_x + foot_r * np.cos(theta)
                y = leg_y + foot_r * np.sin(theta)
                all_vertices.append([x, y, z_base])
                all_normals.append([0, 0, -1])
            
            for i in range(num_segments):
                all_indices.extend([face_base, face_base + 1 + (i + 1) % num_segments, face_base + 1 + i])
    
    def _add_gusset(self, all_vertices, all_indices, all_normals,
                    leg_x, leg_y, base_idx):
        """Add triangular gusset plates."""
        p = self.params
        z_top = p.center[2] + p.leg_height
        z_gusset_bottom = z_top - p.gusset_height
        gusset_width = p.leg_radius * 2
        gusset_thick = 0.006
        
        # Add 4 gusset plates around the leg (at 45 degrees to give rigidity)
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Gusset triangle vertices
            v0 = [leg_x + p.leg_radius * dx, leg_y + p.leg_radius * dy, z_gusset_bottom]
            v1 = [leg_x + p.leg_radius * dx, leg_y + p.leg_radius * dy, z_top]
            v2 = [leg_x + (p.leg_radius + gusset_width) * dx, 
                  leg_y + (p.leg_radius + gusset_width) * dy, z_top]
            
            gusset_base = len(all_vertices)
            for v in [v0, v1, v2]:
                all_vertices.append(v)
                # Normal perpendicular to gusset
                n = [-dy, dx, 0]
                all_normals.append(n)
            
            all_indices.extend([gusset_base, gusset_base + 1, gusset_base + 2])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_total_load_capacity(self) -> float:
        """Get total load capacity of all legs [kg]."""
        return self.params.load_capacity * self.params.num_legs
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_tubular_legs(num_legs: int = 4,
                        height: float = 0.5,
                        diameter: float = 0.076,
                        mounting_diameter: float = 0.8,
                        **kwargs) -> EquipmentLegs:
    """
    Create tubular support legs.
    
    Args:
        num_legs: Number of legs
        height: Leg height [m]
        diameter: Leg pipe diameter [m]
        mounting_diameter: Equipment mounting circle diameter [m]
        **kwargs: Additional parameters
        
    Returns:
        EquipmentLegs instance
    """
    params = EquipmentLegParams(
        leg_type="tubular",
        num_legs=num_legs,
        leg_height=height,
        leg_diameter=diameter,
        mounting_diameter=mounting_diameter,
        **kwargs
    )
    return EquipmentLegs(params)


def create_adjustable_legs(num_legs: int = 4,
                           height: float = 0.5,
                           mounting_diameter: float = 0.8,
                           **kwargs) -> EquipmentLegs:
    """
    Create adjustable leveling legs.
    
    Args:
        num_legs: Number of legs
        height: Leg height [m]
        mounting_diameter: Equipment mounting circle diameter [m]
        **kwargs: Additional parameters
        
    Returns:
        EquipmentLegs instance
    """
    params = EquipmentLegParams(
        leg_type="adjustable",
        num_legs=num_legs,
        leg_height=height,
        foot_type="leveling",
        mounting_diameter=mounting_diameter,
        **kwargs
    )
    return EquipmentLegs(params)


def create_channel_legs(num_legs: int = 4,
                        height: float = 0.5,
                        size: float = 0.1,
                        mounting_diameter: float = 0.8,
                        **kwargs) -> EquipmentLegs:
    """
    Create channel section legs.
    
    Args:
        num_legs: Number of legs
        height: Leg height [m]
        size: Channel web size [m]
        mounting_diameter: Equipment mounting circle diameter [m]
        **kwargs: Additional parameters
        
    Returns:
        EquipmentLegs instance
    """
    params = EquipmentLegParams(
        leg_type="channel",
        num_legs=num_legs,
        leg_height=height,
        leg_diameter=size,
        mounting_diameter=mounting_diameter,
        **kwargs
    )
    return EquipmentLegs(params)
