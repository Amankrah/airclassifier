"""
Explosion vent components for air classification systems.

This module provides explosion relief geometries including
rupture panels, hinged doors, and recoil vents for dust
explosion protection per NFPA 68 / EN 14491.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class ExplosionVentParams:
    """
    Parameters for explosion vent panel.
    
    Attributes:
        vent_area: Vent relief area [m²]
        vent_type: Type ("rupture_panel", "hinged_door", "recoil")
        static_burst_pressure: Burst pressure Pstat [bar]
        duct_diameter: Vent duct diameter [m]
        duct_length: Vent duct length to safe area [m]
        flame_arrestor: Whether to include flame arrestor
        shape: Vent shape ("circular", "rectangular")
        width: Width for rectangular vents [m]
        height: Height for rectangular vents [m]
        flange_thickness: Mounting flange thickness [m]
        panel_thickness: Vent panel thickness [m]
        center: Center position (x, y, z) [m]
        normal: Normal direction (outward) (nx, ny, nz)
    """
    vent_area: float
    vent_type: str = "rupture_panel"
    static_burst_pressure: float = 0.1  # 0.1 bar = 100 mbar typical
    duct_diameter: float = 0.0  # 0 = no duct
    duct_length: float = 0.0
    flame_arrestor: bool = False
    shape: str = "circular"
    width: float = 0.0  # For rectangular
    height: float = 0.0  # For rectangular
    flange_thickness: float = 0.015
    panel_thickness: float = 0.003
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    def __post_init__(self):
        """Calculate dimensions from area if not specified."""
        if self.shape == "circular":
            if self.width == 0.0:
                self.width = 2 * np.sqrt(self.vent_area / np.pi)
                self.height = self.width
        else:
            if self.width == 0.0 or self.height == 0.0:
                # Default to square
                side = np.sqrt(self.vent_area)
                self.width = side
                self.height = side
    
    @property
    def diameter(self) -> float:
        """Vent diameter for circular vents [m]."""
        if self.shape == "circular":
            return self.width
        return np.sqrt(self.width * self.height)  # Equivalent diameter
    
    @property
    def normal_normalized(self) -> Tuple[float, float, float]:
        """Normalized normal direction."""
        n = np.array(self.normal)
        return tuple(n / np.linalg.norm(n))
    
    @property
    def actual_area(self) -> float:
        """Calculate actual vent area based on dimensions [m²]."""
        if self.shape == "circular":
            return np.pi * (self.width / 2) ** 2
        return self.width * self.height
    
    def get_reduced_pressure(self, Kst: float = 150, 
                             volume: float = 1.0) -> float:
        """
        Estimate reduced explosion pressure Pred using EN 14491.
        
        Args:
            Kst: Dust explosion constant [bar·m/s] (100-200 for legumes)
            volume: Protected volume [m³]
            
        Returns:
            Reduced pressure Pred [bar]
        """
        # Simplified EN 14491 calculation
        Av = self.actual_area
        Pstat = self.static_burst_pressure
        
        # This is a simplified approximation
        # Full calculation requires iteration with Pred
        C = 0.1  # Empirical constant
        Pred = Pstat + C * (Kst * volume ** 0.333 / Av) ** 2
        
        return min(Pred, 2.0)  # Cap at typical vessel design pressure


class ExplosionVent:
    """
    Explosion vent geometry for air classification systems.
    
    Generates mesh for explosion relief devices including
    rupture panels, hinged doors, and recoil vents.
    """
    
    def __init__(self, params: ExplosionVentParams):
        """
        Initialize explosion vent.
        
        Args:
            params: Vent parameters
        """
        self.params = params
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
    
    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices, generating if needed."""
        if self._vertices is None:
            self.generate_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get mesh indices, generating if needed."""
        if self._indices is None:
            self.generate_mesh()
        return self._indices
    
    @property
    def normals(self) -> np.ndarray:
        """Get mesh normals, generating if needed."""
        if self._normals is None:
            self.generate_mesh()
        return self._normals
    
    def _get_coordinate_system(self):
        """Get local coordinate system."""
        normal = np.array(self.params.normal_normalized)
        
        if abs(normal[2]) < 0.9:
            perp1 = np.cross(normal, [0, 0, 1])
        else:
            perp1 = np.cross(normal, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(normal, perp1)
        
        return normal, perp1, perp2
    
    def generate_mesh(self, num_segments: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the explosion vent.
        
        Args:
            num_segments: Circumferential segments for round elements
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        normal, perp1, perp2 = self._get_coordinate_system()
        
        # Generate flange
        self._add_flange(all_vertices, all_indices, all_normals,
                        center, normal, perp1, perp2, num_segments)
        
        # Generate vent panel based on type
        if p.vent_type == "rupture_panel":
            self._add_rupture_panel(all_vertices, all_indices, all_normals,
                                    center, normal, perp1, perp2, num_segments)
        elif p.vent_type == "hinged_door":
            self._add_hinged_door(all_vertices, all_indices, all_normals,
                                  center, normal, perp1, perp2, num_segments)
        else:  # recoil
            self._add_recoil_vent(all_vertices, all_indices, all_normals,
                                  center, normal, perp1, perp2, num_segments)
        
        # Add duct if specified
        if p.duct_diameter > 0 and p.duct_length > 0:
            self._add_vent_duct(all_vertices, all_indices, all_normals,
                               center, normal, perp1, perp2, num_segments)
        
        # Add flame arrestor if specified
        if p.flame_arrestor:
            self._add_flame_arrestor(all_vertices, all_indices, all_normals,
                                     center, normal, perp1, perp2, num_segments)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_flange(self, all_vertices, all_indices, all_normals,
                    center, normal, perp1, perp2, num_segments):
        """Add mounting flange."""
        p = self.params
        base_idx = len(all_vertices)
        
        if p.shape == "circular":
            inner_r = p.width / 2
            outer_r = inner_r + 0.05  # 50mm flange width
            
            # Front face of flange
            for radius in [inner_r, outer_r]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    local_x = radius * np.cos(theta)
                    local_y = radius * np.sin(theta)
                    pt = center + local_x * perp1 + local_y * perp2
                    all_vertices.append(list(pt))
                    all_normals.append(list(-normal))  # Facing inward (vessel side)
            
            # Triangles for front face
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
            
            # Back face and outer surface
            back_offset = p.flange_thickness * normal
            base_idx2 = len(all_vertices)
            
            for radius in [inner_r, outer_r]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    local_x = radius * np.cos(theta)
                    local_y = radius * np.sin(theta)
                    pt = center + local_x * perp1 + local_y * perp2 + back_offset
                    all_vertices.append(list(pt))
                    all_normals.append(list(normal))
            
            # Triangles for back face
            for i in range(num_segments):
                i0 = base_idx2 + i
                i1 = base_idx2 + (i + 1) % num_segments
                i2 = base_idx2 + num_segments + i
                i3 = base_idx2 + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i1, i2])
                all_indices.extend([i1, i3, i2])
        
        else:  # Rectangular flange
            hw = p.width / 2
            hh = p.height / 2
            fw = 0.04  # Flange width
            
            # Front face corners (inner and outer)
            inner_corners = [
                center + hw * perp1 + hh * perp2,
                center - hw * perp1 + hh * perp2,
                center - hw * perp1 - hh * perp2,
                center + hw * perp1 - hh * perp2,
            ]
            outer_corners = [
                center + (hw + fw) * perp1 + (hh + fw) * perp2,
                center - (hw + fw) * perp1 + (hh + fw) * perp2,
                center - (hw + fw) * perp1 - (hh + fw) * perp2,
                center + (hw + fw) * perp1 - (hh + fw) * perp2,
            ]
            
            # Add rectangular flange faces
            for corner in inner_corners + outer_corners:
                all_vertices.append(list(corner))
                all_normals.append(list(-normal))
            
            # Create rectangular ring triangles
            for i in range(4):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % 4
                i2 = base_idx + 4 + i
                i3 = base_idx + 4 + (i + 1) % 4
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
    
    def _add_rupture_panel(self, all_vertices, all_indices, all_normals,
                           center, normal, perp1, perp2, num_segments):
        """Add rupture panel (simple thin disc/plate)."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Panel sits on outer face of flange
        panel_center = center + p.flange_thickness * normal
        
        if p.shape == "circular":
            radius = p.width / 2 - 0.005  # Slightly smaller than opening
            
            # Front and back faces of thin panel
            for face_offset, face_normal in [(0, normal), (p.panel_thickness, normal)]:
                face_center = panel_center + face_offset * normal
                face_base = len(all_vertices)
                
                # Center vertex
                all_vertices.append(list(face_center))
                all_normals.append(list(face_normal if face_offset > 0 else -normal))
                
                # Edge vertices
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    local_x = radius * np.cos(theta)
                    local_y = radius * np.sin(theta)
                    pt = face_center + local_x * perp1 + local_y * perp2
                    all_vertices.append(list(pt))
                    all_normals.append(list(face_normal if face_offset > 0 else -normal))
                
                # Fan triangles
                for i in range(num_segments):
                    if face_offset > 0:
                        all_indices.extend([face_base, face_base + 1 + i, 
                                           face_base + 1 + (i + 1) % num_segments])
                    else:
                        all_indices.extend([face_base, face_base + 1 + (i + 1) % num_segments,
                                           face_base + 1 + i])
    
    def _add_hinged_door(self, all_vertices, all_indices, all_normals,
                         center, normal, perp1, perp2, num_segments):
        """Add hinged explosion door."""
        p = self.params
        
        # Door frame
        door_center = center + p.flange_thickness * normal
        door_thickness = 0.02
        
        if p.shape == "circular":
            radius = p.width / 2
            base_idx = len(all_vertices)
            
            # Door disc
            for face, n_dir in [(0, -1), (door_thickness, 1)]:
                fc = door_center + face * normal
                face_base = len(all_vertices)
                
                all_vertices.append(list(fc))
                all_normals.append(list(n_dir * normal))
                
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    pt = fc + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                    all_vertices.append(list(pt))
                    all_normals.append(list(n_dir * normal))
                
                for i in range(num_segments):
                    if n_dir > 0:
                        all_indices.extend([face_base, face_base + 1 + i,
                                           face_base + 1 + (i + 1) % num_segments])
                    else:
                        all_indices.extend([face_base, face_base + 1 + (i + 1) % num_segments,
                                           face_base + 1 + i])
            
            # Add hinge (simplified as cylinder on one side)
            hinge_center = door_center + radius * perp1
            hinge_radius = 0.02
            hinge_length = p.height * 0.8 if p.shape != "circular" else 0.2
            
            h_base = len(all_vertices)
            for t in [0, 1]:
                hc = hinge_center + (t * hinge_length - hinge_length/2) * perp2
                for i in range(8):
                    theta = 2 * np.pi * i / 8
                    pt = hc + hinge_radius * np.cos(theta) * normal + hinge_radius * np.sin(theta) * perp1
                    all_vertices.append(list(pt))
                    n = np.cos(theta) * normal + np.sin(theta) * perp1
                    all_normals.append(list(n))
            
            for i in range(8):
                i0 = h_base + i
                i1 = h_base + (i + 1) % 8
                i2 = h_base + 8 + i
                i3 = h_base + 8 + (i + 1) % 8
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
    
    def _add_recoil_vent(self, all_vertices, all_indices, all_normals,
                         center, normal, perp1, perp2, num_segments):
        """Add recoil-type explosion vent."""
        p = self.params
        
        # Recoil vents have a spring-loaded cover
        vent_center = center + p.flange_thickness * normal
        cover_thickness = 0.015
        
        if p.shape == "circular":
            radius = p.width / 2
            
            # Outer housing/guide
            housing_r = radius + 0.03
            housing_length = 0.1
            
            base_idx = len(all_vertices)
            for t in [0, 1]:
                hc = vent_center + t * housing_length * normal
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    pt = hc + housing_r * np.cos(theta) * perp1 + housing_r * np.sin(theta) * perp2
                    all_vertices.append(list(pt))
                    n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                    all_normals.append(list(n))
            
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                all_indices.extend([i0, i2, i1])
                all_indices.extend([i1, i2, i3])
            
            # Cover disc (inside housing)
            cover_center = vent_center + 0.02 * normal
            cover_base = len(all_vertices)
            all_vertices.append(list(cover_center))
            all_normals.append(list(normal))
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = cover_center + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(normal))
            
            for i in range(num_segments):
                all_indices.extend([cover_base, cover_base + 1 + i,
                                   cover_base + 1 + (i + 1) % num_segments])
    
    def _add_vent_duct(self, all_vertices, all_indices, all_normals,
                       center, normal, perp1, perp2, num_segments):
        """Add vent duct leading to safe area."""
        p = self.params
        
        duct_start = center + (p.flange_thickness + 0.05) * normal
        duct_r = p.duct_diameter / 2
        
        base_idx = len(all_vertices)
        for t in [0, 1]:
            dc = duct_start + t * p.duct_length * normal
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = dc + duct_r * np.cos(theta) * perp1 + duct_r * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                n = np.cos(theta) * perp1 + np.sin(theta) * perp2
                all_normals.append(list(n))
        
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_flame_arrestor(self, all_vertices, all_indices, all_normals,
                            center, normal, perp1, perp2, num_segments):
        """Add flame arrestor (simplified as perforated disc)."""
        p = self.params
        
        # Position at end of duct or just past vent
        if p.duct_length > 0:
            arrestor_center = center + (p.flange_thickness + p.duct_length + 0.02) * normal
        else:
            arrestor_center = center + (p.flange_thickness + 0.05) * normal
        
        radius = (p.duct_diameter / 2 if p.duct_diameter > 0 else p.width / 2) * 0.95
        thickness = 0.025
        
        # Simple disc representation
        base_idx = len(all_vertices)
        for face in [0, thickness]:
            fc = arrestor_center + face * normal
            all_vertices.append(list(fc))
            all_normals.append(list(normal if face > 0 else -normal))
            
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                pt = fc + radius * np.cos(theta) * perp1 + radius * np.sin(theta) * perp2
                all_vertices.append(list(pt))
                all_normals.append(list(normal if face > 0 else -normal))
        
        # Front face triangles
        for i in range(num_segments):
            all_indices.extend([base_idx, base_idx + 1 + (i + 1) % num_segments, base_idx + 1 + i])
        
        # Back face triangles
        back_base = base_idx + num_segments + 1
        for i in range(num_segments):
            all_indices.extend([back_base, back_base + 1 + i, back_base + 1 + (i + 1) % num_segments])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box (min, max) corners."""
        verts = self.vertices
        return verts.min(axis=0), verts.max(axis=0)
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object from the geometry."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required for mesh creation")
        
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )


# Factory functions

def calculate_vent_area(volume: float, Kst: float = 150, 
                        Pstat: float = 0.1, Pred_max: float = 0.5) -> float:
    """
    Calculate required vent area per EN 14491 (simplified).
    
    Args:
        volume: Protected volume [m³]
        Kst: Dust explosion constant [bar·m/s]
        Pstat: Static burst pressure [bar]
        Pred_max: Maximum allowable reduced pressure [bar]
        
    Returns:
        Required vent area [m²]
    """
    # Simplified Bartknecht formula
    # Av = C * V^(2/3) * Kst / sqrt(Pred - Pstat)
    C = 0.1
    delta_P = max(Pred_max - Pstat, 0.1)
    Av = C * volume ** (2/3) * Kst / (100 * np.sqrt(delta_P))
    return max(Av, 0.01)  # Minimum 0.01 m²


def create_rupture_panel(vent_area: float,
                         shape: str = "circular",
                         **kwargs) -> ExplosionVent:
    """
    Create a rupture panel explosion vent.
    
    Args:
        vent_area: Vent area [m²]
        shape: "circular" or "rectangular"
        **kwargs: Additional parameters
        
    Returns:
        ExplosionVent instance
    """
    params = ExplosionVentParams(
        vent_area=vent_area,
        vent_type="rupture_panel",
        shape=shape,
        **kwargs
    )
    return ExplosionVent(params)


def create_hinged_explosion_door(vent_area: float,
                                 **kwargs) -> ExplosionVent:
    """
    Create a hinged explosion door.
    
    Args:
        vent_area: Vent area [m²]
        **kwargs: Additional parameters
        
    Returns:
        ExplosionVent instance
    """
    params = ExplosionVentParams(
        vent_area=vent_area,
        vent_type="hinged_door",
        shape="circular",
        **kwargs
    )
    return ExplosionVent(params)


def create_recoil_vent(vent_area: float, **kwargs) -> ExplosionVent:
    """
    Create a recoil-type explosion vent.
    
    Args:
        vent_area: Vent area [m²]
        **kwargs: Additional parameters
        
    Returns:
        ExplosionVent instance
    """
    params = ExplosionVentParams(
        vent_area=vent_area,
        vent_type="recoil",
        shape="circular",
        **kwargs
    )
    return ExplosionVent(params)
