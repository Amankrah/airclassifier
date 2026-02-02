"""
Structural frame components for air classification systems.

This module provides structural support frame geometries for
equipment mounting, platforms, and access structures.

Coordinate System (Y-up):
    - X: horizontal (width)
    - Y: vertical (height) - UP
    - Z: horizontal (depth)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class StructuralFrameParams:
    """
    Parameters for structural support frame.
    
    Attributes:
        frame_type: Frame type ("bolted", "welded")
        material: Material ("carbon_steel", "304SS", "aluminum")
        width: Frame width [m] (X direction)
        depth: Frame depth [m] (Z direction)
        height: Frame height [m] (Y direction - vertical)
        column_size: Column section size [m]
        beam_size: Beam section size [m]
        platform_levels: List of platform elevations [m] (Y values)
        has_bracing: Whether to include diagonal bracing
        has_grating: Whether to include platform grating
        center: Center position (x, y, z) [m]
    """
    frame_type: str = "bolted"
    material: str = "carbon_steel"
    width: float = 2.0
    depth: float = 2.0
    height: float = 3.0
    column_size: float = 0.1  # 100x100mm HSS
    beam_size: float = 0.08  # 80x80mm HSS
    platform_levels: List[float] = field(default_factory=lambda: [1.5, 3.0])
    has_bracing: bool = True
    has_grating: bool = True
    grating_thickness: float = 0.030
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    @property
    def column_radius(self) -> float:
        """Column equivalent radius for mesh [m]."""
        return self.column_size / 2
    
    @property
    def beam_radius(self) -> float:
        """Beam equivalent radius for mesh [m]."""
        return self.beam_size / 2
    
    def get_column_positions(self) -> List[Tuple[float, float]]:
        """Get (x, z) positions of columns in the horizontal plane."""
        cx, cy, cz = self.center
        hw = self.width / 2
        hd = self.depth / 2
        return [
            (cx - hw, cz - hd),
            (cx + hw, cz - hd),
            (cx + hw, cz + hd),
            (cx - hw, cz + hd),
        ]


class StructuralFrame:
    """
    Structural support frame geometry.
    
    Generates mesh for structural frames with columns, beams,
    bracing, and optional platforms.
    Uses Y-up coordinate system (Y is vertical).
    """
    
    def __init__(self, params: StructuralFrameParams):
        """
        Initialize structural frame.
        
        Args:
            params: Frame parameters
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
    
    def generate_mesh(self, num_segments: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the structural frame.
        
        Args:
            num_segments: Segments for round members
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        y_base = p.center[1]  # Y is vertical
        
        # Add columns (vertical along Y)
        for col_x, col_z in p.get_column_positions():
            self._add_column(all_vertices, all_indices, all_normals,
                           col_x, col_z, y_base, num_segments)
        
        # Add beams at each level (horizontal in XZ plane)
        for level_y in [0] + list(p.platform_levels):
            self._add_beams_at_level(all_vertices, all_indices, all_normals,
                                     y_base + level_y, num_segments)
        
        # Add diagonal bracing if specified
        if p.has_bracing:
            self._add_bracing(all_vertices, all_indices, all_normals,
                            y_base, num_segments)
        
        # Add platform grating if specified
        if p.has_grating:
            for level_y in p.platform_levels:
                self._add_platform(all_vertices, all_indices, all_normals,
                                  y_base + level_y)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_column(self, all_vertices, all_indices, all_normals,
                    col_x, col_z, y_base, num_segments):
        """Add vertical column (extends along Y axis)."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Square HSS column (simplified as cylinder along Y)
        for t in [0, 1]:
            y = y_base + t * p.height
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                x = col_x + p.column_radius * np.cos(theta)
                z = col_z + p.column_radius * np.sin(theta)
                all_vertices.append([x, y, z])
                n = [np.cos(theta), 0, np.sin(theta)]
                all_normals.append(n)
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_beams_at_level(self, all_vertices, all_indices, all_normals,
                            y_level, num_segments):
        """Add horizontal beams at a level (in XZ plane at given Y)."""
        p = self.params
        positions = p.get_column_positions()
        
        # Connect adjacent columns with beams
        for i in range(4):
            start = positions[i]
            end = positions[(i + 1) % 4]
            self._add_beam(all_vertices, all_indices, all_normals,
                          start[0], start[1], end[0], end[1], y_level, num_segments)
    
    def _add_beam(self, all_vertices, all_indices, all_normals,
                  x1, z1, x2, z2, y_level, num_segments):
        """Add single beam between two points (horizontal at given Y)."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Direction vector in XZ plane
        dx = x2 - x1
        dz = z2 - z1
        length = np.sqrt(dx*dx + dz*dz)
        if length < 1e-6:
            return
        
        dir_x = dx / length
        dir_z = dz / length
        
        # Perpendicular in XZ plane
        perp_x = -dir_z
        perp_z = dir_x
        
        # Create beam as extruded circle along XZ
        for t in [0, 1]:
            cx = x1 + t * dx
            cz = z1 + t * dz
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                # Rotate around beam axis
                r_perp = p.beam_radius * np.cos(theta)
                r_y = p.beam_radius * np.sin(theta)
                
                x = cx + r_perp * perp_x
                z = cz + r_perp * perp_z
                y = y_level + r_y
                
                all_vertices.append([x, y, z])
                # Normal points outward from beam axis
                n_perp = np.cos(theta)
                n_y = np.sin(theta)
                all_normals.append([n_perp * perp_x, n_y, n_perp * perp_z])
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_bracing(self, all_vertices, all_indices, all_normals,
                     y_base, num_segments):
        """Add diagonal bracing (in vertical planes)."""
        p = self.params
        positions = p.get_column_positions()
        
        # Add X-bracing on each face
        for i in range(4):
            start = positions[i]
            end = positions[(i + 1) % 4]
            
            # Diagonal from bottom to top
            mid_y = y_base + p.height / 2
            
            # Lower diagonal
            self._add_diagonal(all_vertices, all_indices, all_normals,
                             start[0], y_base, start[1],
                             (start[0] + end[0])/2, mid_y, (start[1] + end[1])/2,
                             num_segments)
            
            # Upper diagonal
            self._add_diagonal(all_vertices, all_indices, all_normals,
                             (start[0] + end[0])/2, mid_y, (start[1] + end[1])/2,
                             end[0], y_base + p.height, end[1],
                             num_segments)
    
    def _add_diagonal(self, all_vertices, all_indices, all_normals,
                      x1, y1, z1, x2, y2, z2, num_segments):
        """Add diagonal brace member."""
        p = self.params
        base_idx = len(all_vertices)
        brace_radius = p.beam_radius * 0.6
        
        # Direction vector
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-6:
            return
        
        dir_vec = np.array([dx, dy, dz]) / length
        
        # Find perpendicular vectors
        if abs(dir_vec[1]) < 0.9:  # Not nearly vertical
            perp1 = np.cross(dir_vec, [0, 1, 0])
        else:
            perp1 = np.cross(dir_vec, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(dir_vec, perp1)
        
        for t in [0, 1]:
            cx = x1 + t * dx
            cy = y1 + t * dy
            cz = z1 + t * dz
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                offset = brace_radius * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                
                all_vertices.append([cx + offset[0], cy + offset[1], cz + offset[2]])
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_platform(self, all_vertices, all_indices, all_normals,
                      y_level):
        """Add platform grating at level (horizontal in XZ plane)."""
        p = self.params
        base_idx = len(all_vertices)
        
        cx, cy, cz = p.center
        hw = (p.width - p.column_size) / 2
        hd = (p.depth - p.column_size) / 2
        thick = p.grating_thickness
        
        # Simple box for platform (horizontal)
        corners_top = [
            [cx - hw, y_level, cz - hd],
            [cx + hw, y_level, cz - hd],
            [cx + hw, y_level, cz + hd],
            [cx - hw, y_level, cz + hd],
        ]
        corners_bottom = [
            [cx - hw, y_level - thick, cz - hd],
            [cx + hw, y_level - thick, cz - hd],
            [cx + hw, y_level - thick, cz + hd],
            [cx - hw, y_level - thick, cz + hd],
        ]
        
        # Add vertices
        for corner in corners_top:
            all_vertices.append(corner)
            all_normals.append([0, 1, 0])  # Normal up
        for corner in corners_bottom:
            all_vertices.append(corner)
            all_normals.append([0, -1, 0])  # Normal down
        
        # Top face
        all_indices.extend([base_idx, base_idx + 1, base_idx + 2])
        all_indices.extend([base_idx, base_idx + 2, base_idx + 3])
        
        # Bottom face
        all_indices.extend([base_idx + 4, base_idx + 6, base_idx + 5])
        all_indices.extend([base_idx + 4, base_idx + 7, base_idx + 6])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_platform_area(self) -> float:
        """Get total platform area [m²]."""
        p = self.params
        platform_area = (p.width - p.column_size) * (p.depth - p.column_size)
        return platform_area * len(p.platform_levels)
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def create_standard_frame(width: float = 2.0,
                          depth: float = 2.0,
                          height: float = 3.0,
                          platform_levels: List[float] = None,
                          **kwargs) -> StructuralFrame:
    """
    Create a standard structural frame.
    
    Args:
        width: Frame width [m] (X direction)
        depth: Frame depth [m] (Z direction)
        height: Frame height [m] (Y direction - vertical)
        platform_levels: List of platform elevations [m] (Y values)
        **kwargs: Additional parameters
        
    Returns:
        StructuralFrame instance
    """
    if platform_levels is None:
        platform_levels = [height]
    
    params = StructuralFrameParams(
        width=width,
        depth=depth,
        height=height,
        platform_levels=platform_levels,
        **kwargs
    )
    return StructuralFrame(params)


def create_equipment_skid(width: float = 1.5,
                          depth: float = 1.0,
                          height: float = 0.3,
                          **kwargs) -> StructuralFrame:
    """
    Create an equipment skid (low frame for equipment mounting).
    
    Args:
        width: Skid width [m] (X direction)
        depth: Skid depth [m] (Z direction)
        height: Skid height [m] (Y direction - vertical)
        **kwargs: Additional parameters
        
    Returns:
        StructuralFrame instance
    """
    params = StructuralFrameParams(
        width=width,
        depth=depth,
        height=height,
        platform_levels=[height],
        has_bracing=False,
        has_grating=True,
        column_size=0.05,
        beam_size=0.05,
        **kwargs
    )
    return StructuralFrame(params)


def create_mezzanine_frame(width: float = 4.0,
                           depth: float = 3.0,
                           height: float = 3.0,
                           **kwargs) -> StructuralFrame:
    """
    Create a mezzanine-style frame with single platform.
    
    Args:
        width: Frame width [m] (X direction)
        depth: Frame depth [m] (Z direction)
        height: Platform height [m] (Y direction - vertical)
        **kwargs: Additional parameters
        
    Returns:
        StructuralFrame instance
    """
    params = StructuralFrameParams(
        width=width,
        depth=depth,
        height=height,
        platform_levels=[height],
        has_bracing=True,
        has_grating=True,
        column_size=0.15,
        beam_size=0.1,
        **kwargs
    )
    return StructuralFrame(params)
