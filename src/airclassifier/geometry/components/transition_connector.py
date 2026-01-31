"""
Transition connector for dust-tight connections between components.

In real industrial powder handling systems, connections between components like
screw feeders and deagglomerators use sealed transition pieces to prevent
particle escape. These typically include:

1. Cylindrical or conical pipe section
2. Flanged ends for bolting to adjacent equipment
3. Gaskets for sealing
4. Optional flexible bellows section for vibration isolation

This module provides geometry for such transition connectors.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False

PI = np.pi
TWO_PI = 2 * np.pi


@dataclass
class TransitionConnectorParams:
    """Parameters for transition connector geometry."""
    
    # Dimensions
    inlet_diameter: float = 0.08     # [m] Top inlet diameter (connects to upstream)
    outlet_diameter: float = 0.08    # [m] Bottom outlet diameter (connects to downstream)
    length: float = 0.05             # [m] Total length of connector
    wall_thickness: float = 0.003    # [m] Wall thickness
    
    # Flange dimensions
    flange_width: float = 0.015      # [m] Width of flange beyond pipe diameter
    flange_thickness: float = 0.008  # [m] Thickness of each flange
    
    # Optional bellows section (for flexibility)
    has_bellows: bool = False        # Include flexible bellows section
    bellows_length: float = 0.02     # [m] Length of bellows section
    num_bellows_folds: int = 3       # Number of folds in bellows
    
    # Position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Mesh resolution
    resolution_radial: int = 24
    resolution_axial: int = 8


class TransitionConnector:
    """
    Transition connector for dust-tight connections between equipment.
    
    This represents the sealed pipe section that connects two pieces of
    equipment (e.g., feeder outlet to deagglomerator inlet) to prevent
    particle escape during material transfer.
    
    Features:
    - Cylindrical or conical transition
    - Flanged ends for bolting
    - Optional flexible bellows section
    """
    
    def __init__(self, params: TransitionConnectorParams = None):
        """
        Initialize transition connector.
        
        Args:
            params: Connector parameters (uses defaults if None)
        """
        self.params = params or TransitionConnectorParams()
        self._vertices = None
        self._indices = None
        self._normals = None
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate mesh for the transition connector.
        
        Returns:
            Tuple of (vertices, indices, normals)
        """
        p = self.params
        vertices = []
        indices = []
        normals = []
        
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial
        
        # Calculate radii
        r_inlet = p.inlet_diameter / 2
        r_outlet = p.outlet_diameter / 2
        r_inlet_outer = r_inlet + p.wall_thickness
        r_outlet_outer = r_outlet + p.wall_thickness
        
        # Flange radii
        flange_r_inlet = r_inlet_outer + p.flange_width
        flange_r_outlet = r_outlet_outer + p.flange_width
        
        # Y positions (connector goes from top to bottom)
        y_top = p.center[1] + p.length / 2
        y_bottom = p.center[1] - p.length / 2
        
        # Flange Y positions
        y_top_flange = y_top + p.flange_thickness
        y_bottom_flange = y_bottom - p.flange_thickness
        
        # 1. Generate top flange (inlet side)
        self._add_flange(vertices, indices, normals, 
                        y_pos=y_top, 
                        y_face=y_top_flange,
                        inner_radius=r_inlet,
                        outer_radius=flange_r_inlet,
                        facing_up=True)
        
        # 2. Generate main pipe body (conical if diameters differ)
        if p.has_bellows:
            # Pipe with bellows section
            self._add_pipe_with_bellows(vertices, indices, normals,
                                        y_top, y_bottom,
                                        r_inlet_outer, r_outlet_outer)
        else:
            # Simple conical/cylindrical pipe
            self._add_pipe_section(vertices, indices, normals,
                                   y_top, y_bottom,
                                   r_inlet_outer, r_outlet_outer)
        
        # 3. Generate bottom flange (outlet side)
        self._add_flange(vertices, indices, normals,
                        y_pos=y_bottom,
                        y_face=y_bottom_flange,
                        inner_radius=r_outlet,
                        outer_radius=flange_r_outlet,
                        facing_up=False)
        
        self._vertices = np.array(vertices, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.int32)
        self._normals = np.array(normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_flange(self, vertices: List, indices: List, normals: List,
                   y_pos: float, y_face: float,
                   inner_radius: float, outer_radius: float,
                   facing_up: bool):
        """Add a flange ring at the specified position."""
        p = self.params
        n_radial = p.resolution_radial
        start_idx = len(vertices)
        
        sign = 1 if facing_up else -1
        
        # Inner ring (at pipe opening)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + inner_radius * np.cos(theta)
            z = p.center[2] + inner_radius * np.sin(theta)
            vertices.append([x, y_face, z])
            normals.append([0.0, sign, 0.0])
        
        # Outer ring (flange edge)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + outer_radius * np.cos(theta)
            z = p.center[2] + outer_radius * np.sin(theta)
            vertices.append([x, y_face, z])
            normals.append([0.0, sign, 0.0])
        
        # Triangles for flange face (annular ring)
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v_inner = start_idx + j
            v_inner_next = start_idx + j_next
            v_outer = start_idx + n_radial + j
            v_outer_next = start_idx + n_radial + j_next
            
            if facing_up:
                indices.extend([v_inner, v_inner_next, v_outer_next])
                indices.extend([v_inner, v_outer_next, v_outer])
            else:
                indices.extend([v_inner, v_outer_next, v_inner_next])
                indices.extend([v_inner, v_outer, v_outer_next])
        
        # Flange outer edge (vertical surface)
        edge_start = len(vertices)
        for j in range(n_radial):
            theta = (j / n_radial) * TWO_PI
            x = p.center[0] + outer_radius * np.cos(theta)
            z = p.center[2] + outer_radius * np.sin(theta)
            vertices.append([x, y_pos, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
            vertices.append([x, y_face, z])
            normals.append([np.cos(theta), 0.0, np.sin(theta)])
        
        # Triangles for flange edge
        for j in range(n_radial):
            j_next = (j + 1) % n_radial
            v0 = edge_start + j * 2
            v1 = edge_start + j * 2 + 1
            v2 = edge_start + j_next * 2 + 1
            v3 = edge_start + j_next * 2
            
            if facing_up:
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
            else:
                indices.extend([v0, v2, v1])
                indices.extend([v0, v3, v2])
    
    def _add_pipe_section(self, vertices: List, indices: List, normals: List,
                         y_top: float, y_bottom: float,
                         r_top: float, r_bottom: float):
        """Add a conical or cylindrical pipe section."""
        p = self.params
        n_radial = p.resolution_radial
        n_axial = p.resolution_axial
        
        start_idx = len(vertices)
        
        # Generate pipe surface
        for i in range(n_axial + 1):
            t = i / n_axial
            y = y_top - t * (y_top - y_bottom)
            r = r_top + t * (r_bottom - r_top)
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)
                vertices.append([x, y, z])
                
                # Normal for conical surface
                dr = r_bottom - r_top
                dy = y_top - y_bottom
                if abs(dr) < 0.0001:
                    # Cylindrical - normal is radial
                    normals.append([np.cos(theta), 0.0, np.sin(theta)])
                else:
                    # Conical - normal is angled
                    slant = np.sqrt(dy**2 + dr**2)
                    n_radial_comp = dy / slant
                    n_axial_comp = -dr / slant  # Negative because we want outward
                    normals.append([n_radial_comp * np.cos(theta), 
                                   n_axial_comp, 
                                   n_radial_comp * np.sin(theta)])
        
        # Generate triangles
        for i in range(n_axial):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j
                
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
    
    def _add_pipe_with_bellows(self, vertices: List, indices: List, normals: List,
                               y_top: float, y_bottom: float,
                               r_top: float, r_bottom: float):
        """Add a pipe section with flexible bellows in the middle."""
        p = self.params
        
        # Split the pipe into three sections: top, bellows, bottom
        bellows_center_y = (y_top + y_bottom) / 2
        bellows_half = p.bellows_length / 2
        
        # Top section (from top flange to bellows)
        y_bellows_top = bellows_center_y + bellows_half
        r_at_bellows = r_top + (r_bottom - r_top) * (y_top - y_bellows_top) / (y_top - y_bottom)
        
        self._add_pipe_section(vertices, indices, normals,
                              y_top, y_bellows_top, r_top, r_at_bellows)
        
        # Bellows section
        self._add_bellows_section(vertices, indices, normals,
                                 y_bellows_top, bellows_center_y - bellows_half,
                                 r_at_bellows)
        
        # Bottom section (from bellows to bottom flange)
        y_bellows_bottom = bellows_center_y - bellows_half
        r_at_bellows_bottom = r_top + (r_bottom - r_top) * (y_top - y_bellows_bottom) / (y_top - y_bottom)
        
        self._add_pipe_section(vertices, indices, normals,
                              y_bellows_bottom, y_bottom, r_at_bellows_bottom, r_bottom)
    
    def _add_bellows_section(self, vertices: List, indices: List, normals: List,
                            y_top: float, y_bottom: float, radius: float):
        """Add a corrugated bellows section for flexibility."""
        p = self.params
        n_radial = p.resolution_radial
        n_folds = p.num_bellows_folds
        
        start_idx = len(vertices)
        
        # Each fold has a peak and valley
        fold_height = (y_top - y_bottom) / (n_folds * 2)
        amplitude = radius * 0.15  # Bellows corrugation depth
        
        n_segments = n_folds * 4 + 1  # 4 segments per fold
        
        for i in range(n_segments):
            t = i / (n_segments - 1)
            y = y_top - t * (y_top - y_bottom)
            
            # Sinusoidal radius variation for bellows shape
            phase = t * n_folds * 2 * PI
            r = radius + amplitude * np.sin(phase)
            
            for j in range(n_radial):
                theta = (j / n_radial) * TWO_PI
                x = p.center[0] + r * np.cos(theta)
                z = p.center[2] + r * np.sin(theta)
                vertices.append([x, y, z])
                
                # Normal includes the bellows curvature
                dr_dt = amplitude * np.cos(phase) * n_folds * 2 * PI / (y_top - y_bottom)
                n_len = np.sqrt(1 + dr_dt**2)
                normals.append([np.cos(theta) / n_len, 
                               dr_dt / n_len, 
                               np.sin(theta) / n_len])
        
        # Generate triangles
        for i in range(n_segments - 1):
            for j in range(n_radial):
                j_next = (j + 1) % n_radial
                v0 = start_idx + i * n_radial + j
                v1 = start_idx + i * n_radial + j_next
                v2 = start_idx + (i + 1) * n_radial + j_next
                v3 = start_idx + (i + 1) * n_radial + j
                
                indices.extend([v0, v1, v2])
                indices.extend([v0, v2, v3])
    
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
    
    @property
    def normals(self) -> np.ndarray:
        """Get vertex normals."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals
    
    def get_total_height(self) -> float:
        """Get total height including flanges."""
        p = self.params
        return p.length + 2 * p.flange_thickness


def create_transition_connector(
    inlet_diameter: float,
    outlet_diameter: float,
    length: float = None,
    with_bellows: bool = False
) -> TransitionConnector:
    """
    Create a transition connector between two components.
    
    Args:
        inlet_diameter: Diameter at inlet (top) side [m]
        outlet_diameter: Diameter at outlet (bottom) side [m]
        length: Total length (auto-calculated if None) [m]
        with_bellows: Include flexible bellows section
    
    Returns:
        TransitionConnector instance
    """
    # Auto-calculate length based on diameter (typical L/D ratio of 0.5-1.0)
    if length is None:
        avg_diameter = (inlet_diameter + outlet_diameter) / 2
        length = max(0.03, avg_diameter * 0.5)  # Minimum 30mm
    
    params = TransitionConnectorParams(
        inlet_diameter=inlet_diameter,
        outlet_diameter=outlet_diameter,
        length=length,
        has_bellows=with_bellows,
    )
    
    return TransitionConnector(params)
