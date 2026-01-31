"""
Diverter valve components for air classification systems.

This module provides diverter valve geometries for directing
flow between two outlets in pneumatic conveying and air
classification applications.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class DiverterValveParams:
    """
    Parameters for two-way diverter valve.
    
    Attributes:
        inlet_diameter: Inlet diameter [m]
        outlet1_diameter: Outlet 1 diameter [m]
        outlet2_diameter: Outlet 2 diameter [m]
        outlet_angle: Angle between outlets [rad]
        blade_type: Type of diverter blade ("flap", "rotating", "plug")
        actuator_type: Actuator type ("pneumatic", "electric", "manual")
        seal_type: Seal type ("flexible", "inflatable", "metal")
        housing_length: Total housing length [m]
        wall_thickness: Wall thickness [m]
        blade_thickness: Diverter blade thickness [m]
        position: Current position (0.0 = outlet1, 1.0 = outlet2)
        center: Center position of inlet (x, y, z) [m]
        inlet_direction: Direction of inlet flow (dx, dy, dz)
    """
    inlet_diameter: float
    outlet1_diameter: float = None  # Defaults to inlet_diameter
    outlet2_diameter: float = None  # Defaults to inlet_diameter
    outlet_angle: float = np.pi / 3  # 60 degrees
    blade_type: str = "flap"
    actuator_type: str = "pneumatic"
    seal_type: str = "flexible"
    housing_length: float = None  # Auto-calculated if None
    wall_thickness: float = 0.003
    blade_thickness: float = 0.005
    position: float = 0.0  # 0 = outlet1, 1 = outlet2
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    inlet_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    def __post_init__(self):
        """Set defaults for optional parameters."""
        if self.outlet1_diameter is None:
            self.outlet1_diameter = self.inlet_diameter
        if self.outlet2_diameter is None:
            self.outlet2_diameter = self.inlet_diameter
        if self.housing_length is None:
            # Calculate minimum length based on outlet angle
            self.housing_length = max(
                self.inlet_diameter * 2,
                (self.outlet1_diameter + self.outlet2_diameter) / (2 * np.sin(self.outlet_angle / 2))
            )
    
    @property
    def inlet_radius(self) -> float:
        """Inlet radius [m]."""
        return self.inlet_diameter / 2
    
    @property
    def outlet1_radius(self) -> float:
        """Outlet 1 radius [m]."""
        return self.outlet1_diameter / 2
    
    @property
    def outlet2_radius(self) -> float:
        """Outlet 2 radius [m]."""
        return self.outlet2_diameter / 2
    
    @property
    def inlet_area(self) -> float:
        """Inlet cross-sectional area [m²]."""
        return np.pi * self.inlet_radius ** 2
    
    @property
    def outlet1_area(self) -> float:
        """Outlet 1 cross-sectional area [m²]."""
        return np.pi * self.outlet1_radius ** 2
    
    @property
    def outlet2_area(self) -> float:
        """Outlet 2 cross-sectional area [m²]."""
        return np.pi * self.outlet2_radius ** 2
    
    @property
    def inlet_direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized inlet direction."""
        d = np.array(self.inlet_direction)
        return tuple(d / np.linalg.norm(d))
    
    @property
    def outlet1_direction(self) -> Tuple[float, float, float]:
        """Direction of outlet 1 (half angle from centerline)."""
        inlet = np.array(self.inlet_direction_normalized)
        # Rotate around perpendicular axis
        if abs(inlet[2]) < 0.9:
            perp = np.cross(inlet, [0, 0, 1])
        else:
            perp = np.cross(inlet, [1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        
        # Rotate inlet direction by -outlet_angle/2
        angle = -self.outlet_angle / 2
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([
            [c + perp[0]**2*(1-c), perp[0]*perp[1]*(1-c) - perp[2]*s, perp[0]*perp[2]*(1-c) + perp[1]*s],
            [perp[1]*perp[0]*(1-c) + perp[2]*s, c + perp[1]**2*(1-c), perp[1]*perp[2]*(1-c) - perp[0]*s],
            [perp[2]*perp[0]*(1-c) - perp[1]*s, perp[2]*perp[1]*(1-c) + perp[0]*s, c + perp[2]**2*(1-c)]
        ])
        outlet_dir = rot @ inlet
        return tuple(outlet_dir)
    
    @property
    def outlet2_direction(self) -> Tuple[float, float, float]:
        """Direction of outlet 2 (half angle from centerline, opposite side)."""
        inlet = np.array(self.inlet_direction_normalized)
        if abs(inlet[2]) < 0.9:
            perp = np.cross(inlet, [0, 0, 1])
        else:
            perp = np.cross(inlet, [1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        
        # Rotate inlet direction by +outlet_angle/2
        angle = self.outlet_angle / 2
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([
            [c + perp[0]**2*(1-c), perp[0]*perp[1]*(1-c) - perp[2]*s, perp[0]*perp[2]*(1-c) + perp[1]*s],
            [perp[1]*perp[0]*(1-c) + perp[2]*s, c + perp[1]**2*(1-c), perp[1]*perp[2]*(1-c) - perp[0]*s],
            [perp[2]*perp[0]*(1-c) - perp[1]*s, perp[2]*perp[1]*(1-c) + perp[0]*s, c + perp[2]**2*(1-c)]
        ])
        outlet_dir = rot @ inlet
        return tuple(outlet_dir)
    
    def get_flow_split(self) -> Tuple[float, float]:
        """
        Get flow split between outlets based on position.
        
        Returns:
            Tuple of (fraction_to_outlet1, fraction_to_outlet2)
        """
        # Linear interpolation based on position
        return (1.0 - self.position, self.position)
    
    def get_pressure_loss_coefficient(self) -> float:
        """
        Estimate pressure loss coefficient K.
        
        Returns:
            Loss coefficient K (ΔP = K * 0.5 * ρ * V²)
        """
        # Base coefficient depends on blade type
        if self.blade_type == "flap":
            K_base = 0.3
        elif self.blade_type == "rotating":
            K_base = 0.4
        else:  # plug
            K_base = 0.5
        
        # Additional loss from angle
        angle_factor = 1 + 0.5 * (self.outlet_angle / (np.pi / 2))
        
        # Intermediate positions have higher loss
        position_factor = 1 + 0.5 * np.sin(np.pi * self.position)
        
        return K_base * angle_factor * position_factor


class DiverterValve:
    """
    Diverter valve geometry for air classification systems.
    
    Generates mesh for Y-type diverter valves with configurable
    blade types and outlet angles.
    """
    
    def __init__(self, params: DiverterValveParams):
        """
        Initialize diverter valve.
        
        Args:
            params: Diverter valve parameters
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
    
    def set_position(self, position: float):
        """
        Set the diverter valve position.
        
        Args:
            position: Position from 0.0 (outlet1) to 1.0 (outlet2)
        """
        self.params.position = max(0.0, min(1.0, position))
        # Invalidate mesh to regenerate with new position
        self._vertices = None
        self._indices = None
        self._normals = None
    
    def generate_mesh(self, num_segments: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the diverter valve.
        
        Args:
            num_segments: Circumferential segments for round sections
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        outlet1_dir = np.array(p.outlet1_direction)
        outlet2_dir = np.array(p.outlet2_direction)
        
        # Get perpendicular axis
        if abs(inlet_dir[2]) < 0.9:
            perp = np.cross(inlet_dir, [0, 0, 1])
        else:
            perp = np.cross(inlet_dir, [1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        local_y = np.cross(inlet_dir, perp)
        
        # Housing geometry
        housing_length = p.housing_length
        branch_start = housing_length * 0.4  # Where Y-branch starts
        
        def add_cylinder(start_pos: np.ndarray, direction: np.ndarray,
                        length: float, radius: float, outer: bool = True):
            """Add a cylindrical section."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            # Create local coordinate system for this cylinder
            if abs(direction[2]) < 0.9:
                local_perp = np.cross(direction, [0, 0, 1])
            else:
                local_perp = np.cross(direction, [1, 0, 0])
            local_perp = local_perp / np.linalg.norm(local_perp)
            local_y = np.cross(direction, local_perp)
            
            # Two rings
            for t in [0.0, 1.0]:
                pos = start_pos + t * length * direction
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    offset = radius * (np.cos(theta) * local_perp + np.sin(theta) * local_y)
                    pt = pos + offset
                    all_vertices.append(list(pt))
                    
                    norm = np.cos(theta) * local_perp + np.sin(theta) * local_y
                    if not outer:
                        norm = -norm
                    all_normals.append(list(norm))
            
            # Triangles
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                
                if outer:
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
                else:
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
        
        # Inlet section (straight)
        inlet_r_outer = p.inlet_radius + p.wall_thickness
        add_cylinder(center, inlet_dir, branch_start, inlet_r_outer, outer=True)
        add_cylinder(center, inlet_dir, branch_start, p.inlet_radius, outer=False)
        
        # Branch point
        branch_center = center + branch_start * inlet_dir
        
        # Outlet 1 branch
        outlet1_length = housing_length - branch_start
        outlet1_start = branch_center
        outlet1_r_outer = p.outlet1_radius + p.wall_thickness
        add_cylinder(outlet1_start, outlet1_dir, outlet1_length, outlet1_r_outer, outer=True)
        add_cylinder(outlet1_start, outlet1_dir, outlet1_length, p.outlet1_radius, outer=False)
        
        # Outlet 2 branch
        outlet2_length = housing_length - branch_start
        outlet2_start = branch_center
        outlet2_r_outer = p.outlet2_radius + p.wall_thickness
        add_cylinder(outlet2_start, outlet2_dir, outlet2_length, outlet2_r_outer, outer=True)
        add_cylinder(outlet2_start, outlet2_dir, outlet2_length, p.outlet2_radius, outer=False)
        
        # Add diverter blade
        self._add_diverter_blade(all_vertices, all_indices, all_normals,
                                 branch_center, inlet_dir, outlet1_dir, outlet2_dir,
                                 perp, num_segments)
        
        # End caps
        def add_end_cap(pos: np.ndarray, direction: np.ndarray,
                       r_inner: float, r_outer: float, normal_positive: bool):
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            if abs(direction[2]) < 0.9:
                local_perp = np.cross(direction, [0, 0, 1])
            else:
                local_perp = np.cross(direction, [1, 0, 0])
            local_perp = local_perp / np.linalg.norm(local_perp)
            local_y = np.cross(direction, local_perp)
            
            norm = list(direction) if normal_positive else list(-direction)
            
            for radius in [r_inner, r_outer]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    offset = radius * (np.cos(theta) * local_perp + np.sin(theta) * local_y)
                    pt = pos + offset
                    all_vertices.append(list(pt))
                    all_normals.append(norm)
            
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                
                if normal_positive:
                    all_indices.extend([i0, i1, i2])
                    all_indices.extend([i1, i3, i2])
                else:
                    all_indices.extend([i0, i2, i1])
                    all_indices.extend([i1, i2, i3])
        
        # Inlet end cap
        add_end_cap(center, inlet_dir, p.inlet_radius, inlet_r_outer, False)
        
        # Outlet end caps
        outlet1_end = outlet1_start + outlet1_length * outlet1_dir
        add_end_cap(outlet1_end, outlet1_dir, p.outlet1_radius, outlet1_r_outer, True)
        
        outlet2_end = outlet2_start + outlet2_length * outlet2_dir
        add_end_cap(outlet2_end, outlet2_dir, p.outlet2_radius, outlet2_r_outer, True)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def _add_diverter_blade(self, all_vertices: list, all_indices: list,
                            all_normals: list, branch_center: np.ndarray,
                            inlet_dir: np.ndarray, outlet1_dir: np.ndarray,
                            outlet2_dir: np.ndarray, perp: np.ndarray,
                            num_segments: int):
        """Add the diverter blade geometry."""
        p = self.params
        
        if p.blade_type == "flap":
            self._add_flap_blade(all_vertices, all_indices, all_normals,
                                 branch_center, inlet_dir, outlet1_dir, outlet2_dir,
                                 perp)
        elif p.blade_type == "rotating":
            self._add_rotating_blade(all_vertices, all_indices, all_normals,
                                     branch_center, inlet_dir, perp)
        else:  # plug
            self._add_plug_blade(all_vertices, all_indices, all_normals,
                                 branch_center, inlet_dir, outlet1_dir, outlet2_dir,
                                 perp, num_segments)
    
    def _add_flap_blade(self, all_vertices: list, all_indices: list,
                        all_normals: list, branch_center: np.ndarray,
                        inlet_dir: np.ndarray, outlet1_dir: np.ndarray,
                        outlet2_dir: np.ndarray, perp: np.ndarray):
        """Add a flap-type diverter blade."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Blade pivots at branch point
        # Position determines which outlet is blocked
        blade_length = p.inlet_radius * 1.5
        blade_width = p.inlet_radius * 2
        
        # Calculate blade angle based on position
        # Position 0 = blocks outlet2, Position 1 = blocks outlet1
        total_angle = p.outlet_angle
        blade_angle = -total_angle/2 + p.position * total_angle
        
        # Rotate blade around pivot perpendicular axis
        c, s = np.cos(blade_angle), np.sin(blade_angle)
        rot = np.array([
            [c + perp[0]**2*(1-c), perp[0]*perp[1]*(1-c) - perp[2]*s, perp[0]*perp[2]*(1-c) + perp[1]*s],
            [perp[1]*perp[0]*(1-c) + perp[2]*s, c + perp[1]**2*(1-c), perp[1]*perp[2]*(1-c) - perp[0]*s],
            [perp[2]*perp[0]*(1-c) - perp[1]*s, perp[2]*perp[1]*(1-c) + perp[0]*s, c + perp[2]**2*(1-c)]
        ])
        
        blade_dir = rot @ inlet_dir
        local_y = np.cross(blade_dir, perp)
        
        # Blade corners (thin rectangular plate)
        corners = [
            branch_center + blade_length * blade_dir - blade_width/2 * local_y - p.blade_thickness/2 * perp,
            branch_center + blade_length * blade_dir + blade_width/2 * local_y - p.blade_thickness/2 * perp,
            branch_center + blade_length * blade_dir + blade_width/2 * local_y + p.blade_thickness/2 * perp,
            branch_center + blade_length * blade_dir - blade_width/2 * local_y + p.blade_thickness/2 * perp,
            branch_center - blade_width/2 * local_y - p.blade_thickness/2 * perp,
            branch_center + blade_width/2 * local_y - p.blade_thickness/2 * perp,
            branch_center + blade_width/2 * local_y + p.blade_thickness/2 * perp,
            branch_center - blade_width/2 * local_y + p.blade_thickness/2 * perp,
        ]
        
        for corner in corners:
            all_vertices.append(list(corner))
            all_normals.append([0, 0, 1])  # Simplified normal
        
        # 6 faces of the blade box
        faces = [
            ([0, 1, 2, 3], blade_dir),    # Front
            ([4, 7, 6, 5], -blade_dir),   # Back
            ([0, 4, 5, 1], -perp),         # Bottom
            ([2, 6, 7, 3], perp),          # Top
            ([0, 3, 7, 4], -local_y),      # Left
            ([1, 5, 6, 2], local_y),       # Right
        ]
        
        for face_indices, face_normal in faces:
            fi = [base_idx + i for i in face_indices]
            # Update normals for this face
            for idx in fi:
                all_normals[idx] = list(face_normal)
            
            all_indices.extend([fi[0], fi[1], fi[2]])
            all_indices.extend([fi[0], fi[2], fi[3]])
    
    def _add_rotating_blade(self, all_vertices: list, all_indices: list,
                            all_normals: list, branch_center: np.ndarray,
                            inlet_dir: np.ndarray, perp: np.ndarray):
        """Add a rotating drum-type diverter."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Rotating drum with cutout
        drum_radius = p.inlet_radius * 0.8
        drum_length = p.inlet_radius * 1.5
        
        # Drum rotation based on position
        rotation_angle = p.position * np.pi
        
        local_y = np.cross(inlet_dir, perp)
        
        # Create drum cylinder (simplified)
        num_seg = 16
        for t in [0.0, 1.0]:
            pos = branch_center + (t - 0.5) * drum_length * local_y
            for i in range(num_seg):
                theta = 2 * np.pi * i / num_seg + rotation_angle
                offset = drum_radius * (np.cos(theta) * inlet_dir + np.sin(theta) * perp)
                pt = pos + offset
                all_vertices.append(list(pt))
                
                norm = np.cos(theta) * inlet_dir + np.sin(theta) * perp
                all_normals.append(list(norm))
        
        # Triangles
        for i in range(num_seg):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_seg
            i2 = base_idx + num_seg + i
            i3 = base_idx + num_seg + (i + 1) % num_seg
            
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def _add_plug_blade(self, all_vertices: list, all_indices: list,
                        all_normals: list, branch_center: np.ndarray,
                        inlet_dir: np.ndarray, outlet1_dir: np.ndarray,
                        outlet2_dir: np.ndarray, perp: np.ndarray,
                        num_segments: int):
        """Add a plug-type diverter."""
        p = self.params
        base_idx = len(all_vertices)
        
        # Plug blocks one outlet
        plug_radius = min(p.outlet1_radius, p.outlet2_radius) * 0.9
        plug_length = p.inlet_radius * 0.5
        
        # Position determines which outlet is blocked
        if p.position < 0.5:
            # Block outlet2
            plug_center = branch_center + plug_radius * np.array(outlet2_dir)
            plug_dir = np.array(outlet2_dir)
        else:
            # Block outlet1
            plug_center = branch_center + plug_radius * np.array(outlet1_dir)
            plug_dir = np.array(outlet1_dir)
        
        # Simple cylinder for plug
        if abs(plug_dir[2]) < 0.9:
            local_perp = np.cross(plug_dir, [0, 0, 1])
        else:
            local_perp = np.cross(plug_dir, [1, 0, 0])
        local_perp = local_perp / np.linalg.norm(local_perp)
        local_y = np.cross(plug_dir, local_perp)
        
        for t in [0.0, 1.0]:
            pos = plug_center + t * plug_length * plug_dir
            for i in range(num_segments):
                theta = 2 * np.pi * i / num_segments
                offset = plug_radius * (np.cos(theta) * local_perp + np.sin(theta) * local_y)
                pt = pos + offset
                all_vertices.append(list(pt))
                
                norm = np.cos(theta) * local_perp + np.sin(theta) * local_y
                all_normals.append(list(norm))
        
        # Triangles
        for i in range(num_segments):
            i0 = base_idx + i
            i1 = base_idx + (i + 1) % num_segments
            i2 = base_idx + num_segments + i
            i3 = base_idx + num_segments + (i + 1) % num_segments
            
            all_indices.extend([i0, i2, i1])
            all_indices.extend([i1, i2, i3])
    
    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet."""
        return self.params.center
    
    def get_outlet1_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet 1."""
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        outlet1_dir = np.array(p.outlet1_direction)
        
        branch_start = p.housing_length * 0.4
        branch_center = center + branch_start * inlet_dir
        outlet_length = p.housing_length - branch_start
        
        outlet1_end = branch_center + outlet_length * outlet1_dir
        return tuple(outlet1_end)
    
    def get_outlet2_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet 2."""
        p = self.params
        center = np.array(p.center)
        inlet_dir = np.array(p.inlet_direction_normalized)
        outlet2_dir = np.array(p.outlet2_direction)
        
        branch_start = p.housing_length * 0.4
        branch_center = center + branch_start * inlet_dir
        outlet_length = p.housing_length - branch_start
        
        outlet2_end = branch_center + outlet_length * outlet2_dir
        return tuple(outlet2_end)
    
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

def create_flap_diverter(inlet_diameter: float = 0.2,
                         outlet_angle: float = np.pi / 3,
                         **kwargs) -> DiverterValve:
    """
    Create a flap-type diverter valve.
    
    Args:
        inlet_diameter: Inlet diameter [m]
        outlet_angle: Angle between outlets [rad]
        **kwargs: Additional parameters
        
    Returns:
        DiverterValve instance
    """
    params = DiverterValveParams(
        inlet_diameter=inlet_diameter,
        outlet_angle=outlet_angle,
        blade_type="flap",
        **kwargs
    )
    return DiverterValve(params)


def create_rotating_diverter(inlet_diameter: float = 0.2,
                             outlet_angle: float = np.pi / 2,
                             **kwargs) -> DiverterValve:
    """
    Create a rotating drum-type diverter valve.
    
    Args:
        inlet_diameter: Inlet diameter [m]
        outlet_angle: Angle between outlets [rad]
        **kwargs: Additional parameters
        
    Returns:
        DiverterValve instance
    """
    params = DiverterValveParams(
        inlet_diameter=inlet_diameter,
        outlet_angle=outlet_angle,
        blade_type="rotating",
        **kwargs
    )
    return DiverterValve(params)


def create_plug_diverter(inlet_diameter: float = 0.2,
                         outlet_angle: float = np.pi / 3,
                         **kwargs) -> DiverterValve:
    """
    Create a plug-type diverter valve.
    
    Args:
        inlet_diameter: Inlet diameter [m]
        outlet_angle: Angle between outlets [rad]
        **kwargs: Additional parameters
        
    Returns:
        DiverterValve instance
    """
    params = DiverterValveParams(
        inlet_diameter=inlet_diameter,
        outlet_angle=outlet_angle,
        blade_type="plug",
        **kwargs
    )
    return DiverterValve(params)


def create_diverter_for_classifier(inlet_diameter: float,
                                   protein_outlet_diameter: float = None,
                                   starch_outlet_diameter: float = None,
                                   **kwargs) -> DiverterValve:
    """
    Create a diverter valve sized for classifier product streams.
    
    Args:
        inlet_diameter: Inlet diameter from classifier [m]
        protein_outlet_diameter: Diameter for protein fraction outlet [m]
        starch_outlet_diameter: Diameter for starch fraction outlet [m]
        **kwargs: Additional parameters
        
    Returns:
        DiverterValve instance
    """
    # Default outlet diameters based on typical flow splits
    if protein_outlet_diameter is None:
        protein_outlet_diameter = inlet_diameter * 0.7  # Smaller for fines
    if starch_outlet_diameter is None:
        starch_outlet_diameter = inlet_diameter * 0.9   # Larger for coarse
    
    params = DiverterValveParams(
        inlet_diameter=inlet_diameter,
        outlet1_diameter=protein_outlet_diameter,
        outlet2_diameter=starch_outlet_diameter,
        outlet_angle=np.pi / 4,  # 45 degrees - moderate angle
        blade_type="flap",
        **kwargs
    )
    return DiverterValve(params)
