"""
Ductwork components for air classification systems.

This module provides round and rectangular duct geometries for
connecting system components in pneumatic conveying and air
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
class RoundDuctParams:
    """
    Parameters for circular duct section.
    
    Attributes:
        diameter: Inner diameter [m]
        length: Duct length [m]
        wall_thickness: Wall thickness [m]
        material: Material type ("galvanized", "304SS", "316SS")
        flanged: Has flanged connections
        insulated: Has insulation
        insulation_thickness: Thickness of insulation [m]
        center: Center position of duct start (x, y, z) [m]
        direction: Direction vector (dx, dy, dz), normalized internally
    """
    diameter: float
    length: float
    wall_thickness: float = 0.002
    material: str = "galvanized"
    flanged: bool = True
    insulated: bool = False
    insulation_thickness: float = 0.025
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    @property
    def radius(self) -> float:
        """Inner radius [m]."""
        return self.diameter / 2
    
    @property
    def outer_diameter(self) -> float:
        """Outer diameter including wall [m]."""
        return self.diameter + 2 * self.wall_thickness
    
    @property
    def outer_radius(self) -> float:
        """Outer radius including wall [m]."""
        return self.outer_diameter / 2
    
    @property
    def cross_section_area(self) -> float:
        """Internal cross-sectional area [m²]."""
        return np.pi * self.radius ** 2
    
    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter (same as diameter for round duct) [m]."""
        return self.diameter
    
    @property
    def surface_area_internal(self) -> float:
        """Internal surface area [m²]."""
        return np.pi * self.diameter * self.length
    
    @property
    def direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized direction vector."""
        d = np.array(self.direction)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return tuple(d)
    
    def get_velocity(self, flow_rate_m3_s: float) -> float:
        """
        Calculate flow velocity for a given volumetric flow rate.
        
        Args:
            flow_rate_m3_s: Volumetric flow rate [m³/s]
            
        Returns:
            Flow velocity [m/s]
        """
        return flow_rate_m3_s / self.cross_section_area
    
    def get_pressure_drop(self, flow_rate_m3_s: float, 
                          air_density: float = 1.2,
                          roughness: float = 0.00015) -> float:
        """
        Calculate pressure drop using Darcy-Weisbach equation.
        
        Args:
            flow_rate_m3_s: Volumetric flow rate [m³/s]
            air_density: Air density [kg/m³]
            roughness: Surface roughness [m]
            
        Returns:
            Pressure drop [Pa]
        """
        velocity = self.get_velocity(flow_rate_m3_s)
        Re = air_density * velocity * self.diameter / 1.81e-5  # Dynamic viscosity
        
        # Friction factor (Swamee-Jain approximation)
        if Re < 2300:
            f = 64 / max(Re, 1)
        else:
            term = roughness / (3.7 * self.diameter) + 5.74 / (Re ** 0.9)
            f = 0.25 / (np.log10(term) ** 2)
        
        # Darcy-Weisbach
        dp = f * (self.length / self.diameter) * 0.5 * air_density * velocity ** 2
        return dp


@dataclass
class RectangularDuctParams:
    """
    Parameters for rectangular duct section.
    
    Attributes:
        width: Internal width [m]
        height: Internal height [m]
        length: Duct length [m]
        corner_radius: Internal corner radius [m]
        wall_thickness: Wall thickness [m]
        material: Material type
        flanged: Has flanged connections
        center: Center position of duct start (x, y, z) [m]
        direction: Direction vector (dx, dy, dz)
    """
    width: float
    height: float
    length: float
    corner_radius: float = 0.0
    wall_thickness: float = 0.002
    material: str = "galvanized"
    flanged: bool = True
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    
    @property
    def cross_section_area(self) -> float:
        """Internal cross-sectional area [m²]."""
        # Subtract corner cutouts if radiused
        corner_cutout = 4 * (self.corner_radius ** 2 - 
                            np.pi * self.corner_radius ** 2 / 4)
        return self.width * self.height - corner_cutout
    
    @property
    def perimeter(self) -> float:
        """Internal perimeter [m]."""
        if self.corner_radius > 0:
            # Replace sharp corners with arcs
            straight = 2 * (self.width - 2 * self.corner_radius) + \
                      2 * (self.height - 2 * self.corner_radius)
            arc = 2 * np.pi * self.corner_radius
            return straight + arc
        return 2 * (self.width + self.height)
    
    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter [m]."""
        return 4 * self.cross_section_area / self.perimeter
    
    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio."""
        return self.width / self.height
    
    @property
    def direction_normalized(self) -> Tuple[float, float, float]:
        """Normalized direction vector."""
        d = np.array(self.direction)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        return tuple(d)
    
    def get_equivalent_round_diameter(self) -> float:
        """
        Get equivalent round duct diameter for same friction loss.
        Uses ASHRAE method.
        
        Returns:
            Equivalent diameter [m]
        """
        a, b = self.width, self.height
        # ASHRAE equivalent diameter formula
        De = 1.30 * ((a * b) ** 0.625) / ((a + b) ** 0.25)
        return De
    
    def get_velocity(self, flow_rate_m3_s: float) -> float:
        """Calculate flow velocity for given flow rate."""
        return flow_rate_m3_s / self.cross_section_area
    
    def get_pressure_drop(self, flow_rate_m3_s: float,
                          air_density: float = 1.2,
                          roughness: float = 0.00015) -> float:
        """Calculate pressure drop using equivalent diameter method."""
        velocity = self.get_velocity(flow_rate_m3_s)
        De = self.get_equivalent_round_diameter()
        Re = air_density * velocity * De / 1.81e-5
        
        if Re < 2300:
            f = 64 / max(Re, 1)
        else:
            term = roughness / (3.7 * De) + 5.74 / (Re ** 0.9)
            f = 0.25 / (np.log10(term) ** 2)
        
        dp = f * (self.length / De) * 0.5 * air_density * velocity ** 2
        return dp


class RoundDuct:
    """
    Round duct geometry for air classification systems.
    
    Generates mesh for circular ducts with optional flanges.
    """
    
    def __init__(self, params: RoundDuctParams):
        """
        Initialize round duct.
        
        Args:
            params: Duct parameters
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
    
    def generate_mesh(self, num_segments: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the duct.
        
        Args:
            num_segments: Number of circumferential segments
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        cx, cy, cz = p.center
        dx, dy, dz = p.direction_normalized
        
        # Create local coordinate system
        # Find perpendicular vectors
        if abs(dz) < 0.9:
            perp1 = np.cross([dx, dy, dz], [0, 0, 1])
        else:
            perp1 = np.cross([dx, dy, dz], [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross([dx, dy, dz], perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        def add_cylinder(radius: float, length: float, offset: float, 
                        outward_normal: bool = True):
            """Add a cylindrical surface."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            verts = []
            norms = []
            
            # Two rings at each end
            for t in [0.0, 1.0]:
                z_pos = offset + t * length
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    
                    # Position on circle
                    local_x = radius * np.cos(theta)
                    local_y = radius * np.sin(theta)
                    
                    # Transform to world coordinates
                    x = cx + local_x * perp1[0] + local_y * perp2[0] + z_pos * dx
                    y = cy + local_x * perp1[1] + local_y * perp2[1] + z_pos * dy
                    z = cz + local_x * perp1[2] + local_y * perp2[2] + z_pos * dz
                    verts.append([x, y, z])
                    
                    # Normal
                    nx = np.cos(theta) * perp1[0] + np.sin(theta) * perp2[0]
                    ny = np.cos(theta) * perp1[1] + np.sin(theta) * perp2[1]
                    nz = np.cos(theta) * perp1[2] + np.sin(theta) * perp2[2]
                    if not outward_normal:
                        nx, ny, nz = -nx, -ny, -nz
                    norms.append([nx, ny, nz])
            
            all_vertices.extend(verts)
            all_normals.extend(norms)
            
            # Triangles connecting the rings
            inds = []
            for i in range(num_segments):
                i0 = base_idx + i
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i
                i3 = base_idx + num_segments + (i + 1) % num_segments
                
                if outward_normal:
                    inds.extend([i0, i2, i1])
                    inds.extend([i1, i2, i3])
                else:
                    inds.extend([i0, i1, i2])
                    inds.extend([i1, i3, i2])
            
            all_indices.extend(inds)
        
        def add_annular_cap(inner_r: float, outer_r: float, z_offset: float,
                           normal_positive: bool = True):
            """Add an annular end cap."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            verts = []
            norms = []
            
            # Inner and outer rings
            for radius in [inner_r, outer_r]:
                for i in range(num_segments):
                    theta = 2 * np.pi * i / num_segments
                    local_x = radius * np.cos(theta)
                    local_y = radius * np.sin(theta)
                    
                    x = cx + local_x * perp1[0] + local_y * perp2[0] + z_offset * dx
                    y = cy + local_x * perp1[1] + local_y * perp2[1] + z_offset * dy
                    z = cz + local_x * perp1[2] + local_y * perp2[2] + z_offset * dz
                    verts.append([x, y, z])
                    
                    if normal_positive:
                        norms.append([dx, dy, dz])
                    else:
                        norms.append([-dx, -dy, -dz])
            
            all_vertices.extend(verts)
            all_normals.extend(norms)
            
            # Triangles
            inds = []
            for i in range(num_segments):
                i0 = base_idx + i  # inner ring
                i1 = base_idx + (i + 1) % num_segments
                i2 = base_idx + num_segments + i  # outer ring
                i3 = base_idx + num_segments + (i + 1) % num_segments
                
                if normal_positive:
                    inds.extend([i0, i1, i2])
                    inds.extend([i1, i3, i2])
                else:
                    inds.extend([i0, i2, i1])
                    inds.extend([i1, i2, i3])
            
            all_indices.extend(inds)
        
        # Main duct - outer surface
        add_cylinder(p.outer_radius, p.length, 0.0, outward_normal=True)
        
        # Main duct - inner surface
        add_cylinder(p.radius, p.length, 0.0, outward_normal=False)
        
        # End caps
        add_annular_cap(p.radius, p.outer_radius, 0.0, normal_positive=False)
        add_annular_cap(p.radius, p.outer_radius, p.length, normal_positive=True)
        
        # Add flanges if specified
        if p.flanged:
            flange_width = 0.03  # 30mm flange
            flange_thickness = 0.01  # 10mm thick
            flange_outer_r = p.outer_radius + flange_width
            
            # Inlet flange
            add_cylinder(flange_outer_r, flange_thickness, -flange_thickness, 
                        outward_normal=True)
            add_annular_cap(p.outer_radius, flange_outer_r, -flange_thickness, 
                           normal_positive=False)
            add_annular_cap(p.outer_radius, flange_outer_r, 0.0, 
                           normal_positive=True)
            
            # Outlet flange
            add_cylinder(flange_outer_r, flange_thickness, p.length, 
                        outward_normal=True)
            add_annular_cap(p.outer_radius, flange_outer_r, p.length, 
                           normal_positive=False)
            add_annular_cap(p.outer_radius, flange_outer_r, p.length + flange_thickness, 
                           normal_positive=True)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet."""
        return self.params.center
    
    def get_outlet_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet."""
        cx, cy, cz = self.params.center
        dx, dy, dz = self.params.direction_normalized
        L = self.params.length
        return (cx + L * dx, cy + L * dy, cz + L * dz)
    
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


class RectangularDuct:
    """
    Rectangular duct geometry for air classification systems.
    
    Generates mesh for rectangular ducts with optional corner radii.
    """
    
    def __init__(self, params: RectangularDuctParams):
        """
        Initialize rectangular duct.
        
        Args:
            params: Duct parameters
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
    
    def generate_mesh(self, corner_segments: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for rectangular duct.
        
        Args:
            corner_segments: Segments per corner radius
            
        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        all_normals = []
        
        p = self.params
        cx, cy, cz = p.center
        dx, dy, dz = p.direction_normalized
        
        # Create local coordinate system
        if abs(dz) < 0.9:
            perp1 = np.cross([dx, dy, dz], [0, 0, 1])
        else:
            perp1 = np.cross([dx, dy, dz], [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross([dx, dy, dz], perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        hw = p.width / 2
        hh = p.height / 2
        t = p.wall_thickness
        
        def add_quad_face(corners: list, normal: list):
            """Add a quadrilateral face as two triangles."""
            nonlocal all_vertices, all_indices, all_normals
            base_idx = len(all_vertices)
            
            for corner in corners:
                all_vertices.append(corner)
                all_normals.append(normal)
            
            all_indices.extend([base_idx, base_idx + 1, base_idx + 2])
            all_indices.extend([base_idx, base_idx + 2, base_idx + 3])
        
        def transform_point(local_x, local_y, z_offset):
            """Transform local (x, y) to world coordinates."""
            x = cx + local_x * perp1[0] + local_y * perp2[0] + z_offset * dx
            y = cy + local_x * perp1[1] + local_y * perp2[1] + z_offset * dy
            z = cz + local_x * perp1[2] + local_y * perp2[2] + z_offset * dz
            return [x, y, z]
        
        # Outer walls (4 faces)
        # Front wall (-Y outer)
        corners = [
            transform_point(-hw - t, -hh - t, 0),
            transform_point(hw + t, -hh - t, 0),
            transform_point(hw + t, -hh - t, p.length),
            transform_point(-hw - t, -hh - t, p.length),
        ]
        normal = [-perp2[0], -perp2[1], -perp2[2]]
        add_quad_face(corners, normal)
        
        # Back wall (+Y outer)
        corners = [
            transform_point(hw + t, hh + t, 0),
            transform_point(-hw - t, hh + t, 0),
            transform_point(-hw - t, hh + t, p.length),
            transform_point(hw + t, hh + t, p.length),
        ]
        normal = [perp2[0], perp2[1], perp2[2]]
        add_quad_face(corners, normal)
        
        # Left wall (-X outer)
        corners = [
            transform_point(-hw - t, hh + t, 0),
            transform_point(-hw - t, -hh - t, 0),
            transform_point(-hw - t, -hh - t, p.length),
            transform_point(-hw - t, hh + t, p.length),
        ]
        normal = [-perp1[0], -perp1[1], -perp1[2]]
        add_quad_face(corners, normal)
        
        # Right wall (+X outer)
        corners = [
            transform_point(hw + t, -hh - t, 0),
            transform_point(hw + t, hh + t, 0),
            transform_point(hw + t, hh + t, p.length),
            transform_point(hw + t, -hh - t, p.length),
        ]
        normal = [perp1[0], perp1[1], perp1[2]]
        add_quad_face(corners, normal)
        
        # Inner walls (4 faces with reversed normals)
        # Front inner (+Y direction)
        corners = [
            transform_point(hw, -hh, 0),
            transform_point(-hw, -hh, 0),
            transform_point(-hw, -hh, p.length),
            transform_point(hw, -hh, p.length),
        ]
        normal = [perp2[0], perp2[1], perp2[2]]
        add_quad_face(corners, normal)
        
        # Back inner (-Y direction)
        corners = [
            transform_point(-hw, hh, 0),
            transform_point(hw, hh, 0),
            transform_point(hw, hh, p.length),
            transform_point(-hw, hh, p.length),
        ]
        normal = [-perp2[0], -perp2[1], -perp2[2]]
        add_quad_face(corners, normal)
        
        # Left inner (+X direction)
        corners = [
            transform_point(-hw, -hh, 0),
            transform_point(-hw, hh, 0),
            transform_point(-hw, hh, p.length),
            transform_point(-hw, -hh, p.length),
        ]
        normal = [perp1[0], perp1[1], perp1[2]]
        add_quad_face(corners, normal)
        
        # Right inner (-X direction)
        corners = [
            transform_point(hw, hh, 0),
            transform_point(hw, -hh, 0),
            transform_point(hw, -hh, p.length),
            transform_point(hw, hh, p.length),
        ]
        normal = [-perp1[0], -perp1[1], -perp1[2]]
        add_quad_face(corners, normal)
        
        # End caps (inlet and outlet)
        # Inlet cap - ring between inner and outer rectangles
        for sign, z_off in [(-1.0, 0.0), (1.0, p.length)]:
            norm = [sign * dx, sign * dy, sign * dz]
            
            # Four edges of the rectangular ring
            # Bottom edge
            corners = [
                transform_point(-hw - t, -hh - t, z_off),
                transform_point(hw + t, -hh - t, z_off),
                transform_point(hw, -hh, z_off),
                transform_point(-hw, -hh, z_off),
            ]
            add_quad_face(corners if sign > 0 else corners[::-1], norm)
            
            # Top edge
            corners = [
                transform_point(hw + t, hh + t, z_off),
                transform_point(-hw - t, hh + t, z_off),
                transform_point(-hw, hh, z_off),
                transform_point(hw, hh, z_off),
            ]
            add_quad_face(corners if sign > 0 else corners[::-1], norm)
            
            # Left edge
            corners = [
                transform_point(-hw - t, hh + t, z_off),
                transform_point(-hw - t, -hh - t, z_off),
                transform_point(-hw, -hh, z_off),
                transform_point(-hw, hh, z_off),
            ]
            add_quad_face(corners if sign > 0 else corners[::-1], norm)
            
            # Right edge
            corners = [
                transform_point(hw + t, -hh - t, z_off),
                transform_point(hw + t, hh + t, z_off),
                transform_point(hw, hh, z_off),
                transform_point(hw, -hh, z_off),
            ]
            add_quad_face(corners if sign > 0 else corners[::-1], norm)
        
        self._vertices = np.array(all_vertices, dtype=np.float32)
        self._indices = np.array(all_indices, dtype=np.int32)
        self._normals = np.array(all_normals, dtype=np.float32)
        
        return self._vertices, self._indices, self._normals
    
    def get_inlet_position(self) -> Tuple[float, float, float]:
        """Get center position of inlet."""
        return self.params.center
    
    def get_outlet_position(self) -> Tuple[float, float, float]:
        """Get center position of outlet."""
        cx, cy, cz = self.params.center
        dx, dy, dz = self.params.direction_normalized
        L = self.params.length
        return (cx + L * dx, cy + L * dy, cz + L * dz)
    
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

def create_standard_round_duct(diameter: float = 0.2, 
                               length: float = 1.0,
                               **kwargs) -> RoundDuct:
    """
    Create a standard round duct.
    
    Args:
        diameter: Inner diameter [m]
        length: Duct length [m]
        **kwargs: Additional parameters
        
    Returns:
        RoundDuct instance
    """
    params = RoundDuctParams(
        diameter=diameter,
        length=length,
        **kwargs
    )
    return RoundDuct(params)


def create_standard_rectangular_duct(width: float = 0.3,
                                     height: float = 0.2,
                                     length: float = 1.0,
                                     **kwargs) -> RectangularDuct:
    """
    Create a standard rectangular duct.
    
    Args:
        width: Internal width [m]
        height: Internal height [m]
        length: Duct length [m]
        **kwargs: Additional parameters
        
    Returns:
        RectangularDuct instance
    """
    params = RectangularDuctParams(
        width=width,
        height=height,
        length=length,
        **kwargs
    )
    return RectangularDuct(params)


def create_duct_for_flow(flow_rate_m3_h: float,
                         velocity_target: float = 15.0,
                         length: float = 1.0,
                         duct_type: str = "round") -> Any:
    """
    Create a duct sized for a specific flow rate and target velocity.
    
    Args:
        flow_rate_m3_h: Volumetric flow rate [m³/h]
        velocity_target: Target flow velocity [m/s]
        length: Duct length [m]
        duct_type: "round" or "rectangular"
        
    Returns:
        RoundDuct or RectangularDuct instance
    """
    flow_rate_m3_s = flow_rate_m3_h / 3600
    area_required = flow_rate_m3_s / velocity_target
    
    if duct_type == "round":
        diameter = np.sqrt(4 * area_required / np.pi)
        # Round to standard sizes (50mm increments)
        diameter = np.ceil(diameter * 20) / 20
        return create_standard_round_duct(diameter=diameter, length=length)
    else:
        # Assume square cross-section
        side = np.sqrt(area_required)
        side = np.ceil(side * 20) / 20
        return create_standard_rectangular_duct(width=side, height=side, length=length)
