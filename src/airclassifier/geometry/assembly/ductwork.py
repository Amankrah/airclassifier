"""
Ductwork system assembly for air classification systems.

This module provides assembly of ductwork components including
ducts, transitions, elbows, and diverters into a complete
interconnection system.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class DuctworkSystemParams:
    """
    Parameters for a complete ductwork system.
    
    Attributes:
        main_duct_diameter: Primary duct diameter [m]
        total_length: Total approximate duct run length [m]
        num_elbows: Number of 90-degree elbows
        num_45_elbows: Number of 45-degree elbows
        has_diverter: Whether to include a flow diverter
        has_transition: Whether to include inlet/outlet transitions
        inlet_diameter: Inlet connection diameter [m]
        outlet_diameter: Outlet connection diameter [m]
        wall_thickness: Duct wall thickness [m]
        material: Duct material type
        elbow_r_d_ratio: Elbow radius-to-diameter ratio
    """
    main_duct_diameter: float = 0.2
    total_length: float = 5.0
    num_elbows: int = 2
    num_45_elbows: int = 0
    has_diverter: bool = False
    has_transition: bool = True
    inlet_diameter: float = None  # Defaults to main_duct_diameter
    outlet_diameter: float = None  # Defaults to main_duct_diameter
    wall_thickness: float = 0.002
    material: str = "galvanized"
    elbow_r_d_ratio: float = 1.5
    
    def __post_init__(self):
        """Set defaults for optional parameters."""
        if self.inlet_diameter is None:
            self.inlet_diameter = self.main_duct_diameter
        if self.outlet_diameter is None:
            self.outlet_diameter = self.main_duct_diameter
    
    @property
    def total_equivalent_length(self) -> float:
        """
        Calculate total equivalent length including fittings.
        
        Uses equivalent length method for pressure drop estimation.
        
        Returns:
            Equivalent length [m]
        """
        L_eq = self.total_length
        
        # Add equivalent lengths for elbows
        # 90° elbow: ~30 diameters equivalent length
        L_eq += self.num_elbows * 30 * self.main_duct_diameter
        
        # 45° elbow: ~16 diameters equivalent length
        L_eq += self.num_45_elbows * 16 * self.main_duct_diameter
        
        # Transitions: ~10 diameters each
        if self.has_transition:
            L_eq += 2 * 10 * self.main_duct_diameter
        
        # Diverter: ~20 diameters
        if self.has_diverter:
            L_eq += 20 * self.main_duct_diameter
        
        return L_eq


class DuctworkSystemAssembly:
    """
    Complete ductwork system assembly.
    
    Assembles ducts, transitions, elbows, and diverters into
    a complete interconnected ductwork system.
    """
    
    def __init__(self, params: DuctworkSystemParams):
        """
        Initialize ductwork system assembly.
        
        Args:
            params: System parameters
        """
        self.params = params
        self._components: Dict[str, Any] = {}
        self._component_positions: Dict[str, Tuple[float, float, float]] = {}
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._create_components()
    
    def _create_components(self):
        """Create all ductwork components based on parameters."""
        # Lazy imports to avoid circular dependencies
        from ..components.ductwork import (
            RoundDuct, RoundDuctParams,
            create_standard_round_duct
        )
        from ..components.transitions import (
            Transition, TransitionParams,
            create_round_reducer
        )
        from ..components.elbows import (
            Elbow, ElbowParams,
            create_90_degree_elbow, create_45_degree_elbow
        )
        from ..components.diverter import (
            DiverterValve, DiverterValveParams,
            create_flap_diverter
        )
        
        p = self.params
        current_pos = np.array([0.0, 0.0, 0.0])
        current_dir = np.array([0.0, 0.0, 1.0])
        
        # Calculate duct segment lengths
        num_fittings = p.num_elbows + p.num_45_elbows + (1 if p.has_diverter else 0)
        num_segments = num_fittings + 1
        segment_length = p.total_length / num_segments
        
        # Inlet transition (if different diameter)
        if p.has_transition and abs(p.inlet_diameter - p.main_duct_diameter) > 0.001:
            trans_length = abs(p.main_duct_diameter - p.inlet_diameter) / (2 * np.tan(np.radians(15)))
            trans_length = max(trans_length, 0.1)
            
            inlet_trans = create_round_reducer(
                inlet_diameter=p.inlet_diameter,
                outlet_diameter=p.main_duct_diameter,
                length=trans_length,
                center=tuple(current_pos),
                direction=tuple(current_dir)
            )
            self._components['inlet_transition'] = inlet_trans
            self._component_positions['inlet_transition'] = tuple(current_pos)
            current_pos = current_pos + trans_length * current_dir
        
        # Create duct segments and fittings
        segment_idx = 0
        elbow_idx = 0
        elbow_45_idx = 0
        
        while segment_idx < num_segments:
            # Add straight duct segment
            duct_name = f'duct_{segment_idx}'
            duct = create_standard_round_duct(
                diameter=p.main_duct_diameter,
                length=segment_length,
                wall_thickness=p.wall_thickness,
                center=tuple(current_pos),
                direction=tuple(current_dir)
            )
            self._components[duct_name] = duct
            self._component_positions[duct_name] = tuple(current_pos)
            current_pos = current_pos + segment_length * current_dir
            
            segment_idx += 1
            
            # Add fitting after segment (except last)
            if segment_idx < num_segments:
                # Determine which type of fitting
                if elbow_idx < p.num_elbows:
                    # Add 90° elbow
                    bend_radius = p.elbow_r_d_ratio * p.main_duct_diameter
                    
                    # Alternate direction for realistic layout
                    if elbow_idx % 2 == 0:
                        bend_axis = np.array([1.0, 0.0, 0.0])
                    else:
                        bend_axis = np.array([0.0, 1.0, 0.0])
                    
                    elbow_params = ElbowParams(
                        elbow_type="round",
                        diameter=p.main_duct_diameter,
                        bend_radius=bend_radius,
                        bend_angle=np.pi / 2,
                        wall_thickness=p.wall_thickness,
                        center=tuple(current_pos),
                        inlet_direction=tuple(current_dir),
                        bend_axis=tuple(bend_axis)
                    )
                    elbow = Elbow(elbow_params)
                    elbow_name = f'elbow_90_{elbow_idx}'
                    self._components[elbow_name] = elbow
                    self._component_positions[elbow_name] = tuple(current_pos)
                    
                    # Update position and direction after elbow
                    current_pos = np.array(elbow.get_outlet_position())
                    current_dir = np.array(elbow.params.outlet_direction)
                    current_dir = current_dir / np.linalg.norm(current_dir)
                    
                    elbow_idx += 1
                    
                elif elbow_45_idx < p.num_45_elbows:
                    # Add 45° elbow
                    bend_radius = p.elbow_r_d_ratio * p.main_duct_diameter
                    bend_axis = np.array([1.0, 0.0, 0.0])
                    
                    elbow_params = ElbowParams(
                        elbow_type="round",
                        diameter=p.main_duct_diameter,
                        bend_radius=bend_radius,
                        bend_angle=np.pi / 4,
                        wall_thickness=p.wall_thickness,
                        center=tuple(current_pos),
                        inlet_direction=tuple(current_dir),
                        bend_axis=tuple(bend_axis)
                    )
                    elbow = Elbow(elbow_params)
                    elbow_name = f'elbow_45_{elbow_45_idx}'
                    self._components[elbow_name] = elbow
                    self._component_positions[elbow_name] = tuple(current_pos)
                    
                    current_pos = np.array(elbow.get_outlet_position())
                    current_dir = np.array(elbow.params.outlet_direction)
                    current_dir = current_dir / np.linalg.norm(current_dir)
                    
                    elbow_45_idx += 1
                    
                elif p.has_diverter:
                    # Add diverter
                    diverter = create_flap_diverter(
                        inlet_diameter=p.main_duct_diameter,
                        center=tuple(current_pos),
                        inlet_direction=tuple(current_dir)
                    )
                    self._components['diverter'] = diverter
                    self._component_positions['diverter'] = tuple(current_pos)
                    
                    # For now, follow outlet1 direction
                    current_pos = np.array(diverter.get_outlet1_position())
                    current_dir = np.array(diverter.params.outlet1_direction)
                    current_dir = current_dir / np.linalg.norm(current_dir)
        
        # Outlet transition (if different diameter)
        if p.has_transition and abs(p.outlet_diameter - p.main_duct_diameter) > 0.001:
            trans_length = abs(p.main_duct_diameter - p.outlet_diameter) / (2 * np.tan(np.radians(15)))
            trans_length = max(trans_length, 0.1)
            
            outlet_trans = create_round_reducer(
                inlet_diameter=p.main_duct_diameter,
                outlet_diameter=p.outlet_diameter,
                length=trans_length,
                center=tuple(current_pos),
                direction=tuple(current_dir)
            )
            self._components['outlet_transition'] = outlet_trans
            self._component_positions['outlet_transition'] = tuple(current_pos)
    
    def get_component(self, name: str) -> Any:
        """
        Get a specific component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
        """
        return self._components.get(name)
    
    def get_component_names(self) -> List[str]:
        """Get list of all component names."""
        return list(self._components.keys())
    
    def get_component_position(self, name: str) -> Tuple[float, float, float]:
        """
        Get position of a component.
        
        Args:
            name: Component name
            
        Returns:
            Position tuple (x, y, z)
        """
        return self._component_positions.get(name)
    
    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh from all components.
        
        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0
        
        for name, component in self._components.items():
            verts = component.vertices
            inds = component.indices
            
            all_vertices.append(verts)
            all_indices.append(inds + vertex_offset)
            vertex_offset += len(verts)
        
        if all_vertices:
            self._vertices = np.vstack(all_vertices).astype(np.float32)
            self._indices = np.concatenate(all_indices).astype(np.int32)
        else:
            self._vertices = np.array([], dtype=np.float32).reshape(0, 3)
            self._indices = np.array([], dtype=np.int32)
        
        return self._vertices, self._indices
    
    @property
    def vertices(self) -> np.ndarray:
        """Get combined mesh vertices."""
        if self._vertices is None:
            self.build_mesh()
        return self._vertices
    
    @property
    def indices(self) -> np.ndarray:
        """Get combined mesh indices."""
        if self._indices is None:
            self.build_mesh()
        return self._indices
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of entire system.
        
        Returns:
            Tuple of (min_corner, max_corner) as numpy arrays
        """
        verts = self.vertices
        if len(verts) == 0:
            return np.zeros(3), np.zeros(3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_total_pressure_drop(self, flow_rate_m3_s: float,
                                 air_density: float = 1.2) -> float:
        """
        Estimate total pressure drop through ductwork.
        
        Args:
            flow_rate_m3_s: Volumetric flow rate [m³/s]
            air_density: Air density [kg/m³]
            
        Returns:
            Pressure drop [Pa]
        """
        p = self.params
        
        # Friction loss using equivalent length method
        D = p.main_duct_diameter
        A = np.pi * (D / 2) ** 2
        V = flow_rate_m3_s / A
        
        # Reynolds number
        mu = 1.81e-5  # Dynamic viscosity of air
        Re = air_density * V * D / mu
        
        # Friction factor (Swamee-Jain)
        roughness = 0.00015  # Galvanized steel
        if Re < 2300:
            f = 64 / max(Re, 1)
        else:
            term = roughness / (3.7 * D) + 5.74 / (Re ** 0.9)
            f = 0.25 / (np.log10(term) ** 2)
        
        # Darcy-Weisbach with equivalent length
        L_eq = p.total_equivalent_length
        dp = f * (L_eq / D) * 0.5 * air_density * V ** 2
        
        return dp
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get summary of ductwork system.
        
        Returns:
            Dictionary with system information
        """
        bounds_min, bounds_max = self.get_bounds()
        
        return {
            'main_diameter_m': self.params.main_duct_diameter,
            'total_length_m': self.params.total_length,
            'equivalent_length_m': self.params.total_equivalent_length,
            'num_90_elbows': self.params.num_elbows,
            'num_45_elbows': self.params.num_45_elbows,
            'has_diverter': self.params.has_diverter,
            'has_transitions': self.params.has_transition,
            'num_components': len(self._components),
            'bounds_min': bounds_min,
            'bounds_max': bounds_max,
            'total_vertices': len(self.vertices),
            'total_triangles': len(self.indices) // 3,
        }
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object from the geometry."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required for mesh creation")
        
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )
    
    def print_summary(self):
        """Print a summary of the ductwork system."""
        summary = self.get_system_summary()
        
        print("=" * 60)
        print("DUCTWORK SYSTEM SUMMARY")
        print("=" * 60)
        print(f"Main Duct Diameter: {summary['main_diameter_m']*1000:.0f} mm")
        print(f"Total Duct Length: {summary['total_length_m']:.2f} m")
        print(f"Equivalent Length: {summary['equivalent_length_m']:.2f} m")
        print("-" * 60)
        print("FITTINGS:")
        print(f"  90° Elbows: {summary['num_90_elbows']}")
        print(f"  45° Elbows: {summary['num_45_elbows']}")
        print(f"  Diverter: {'Yes' if summary['has_diverter'] else 'No'}")
        print(f"  Transitions: {'Yes' if summary['has_transitions'] else 'No'}")
        print("-" * 60)
        print("GEOMETRY:")
        print(f"  Total Components: {summary['num_components']}")
        print(f"  Total Vertices: {summary['total_vertices']}")
        print(f"  Total Triangles: {summary['total_triangles']}")
        print(f"  Bounds: [{summary['bounds_min']}] to [{summary['bounds_max']}]")
        print("=" * 60)


# Factory functions

def create_standard_ductwork(main_diameter: float = 0.2,
                             total_length: float = 5.0,
                             **kwargs) -> DuctworkSystemAssembly:
    """
    Create a standard ductwork system.
    
    Args:
        main_diameter: Main duct diameter [m]
        total_length: Total duct run length [m]
        **kwargs: Additional parameters
        
    Returns:
        DuctworkSystemAssembly instance
    """
    params = DuctworkSystemParams(
        main_duct_diameter=main_diameter,
        total_length=total_length,
        **kwargs
    )
    return DuctworkSystemAssembly(params)


def create_ductwork_for_classifier(flow_rate_m3_h: float,
                                   target_velocity: float = 15.0,
                                   run_length: float = 5.0,
                                   num_turns: int = 2,
                                   include_diverter: bool = True) -> DuctworkSystemAssembly:
    """
    Create ductwork sized for an air classifier.
    
    Args:
        flow_rate_m3_h: Volumetric flow rate [m³/h]
        target_velocity: Target flow velocity [m/s]
        run_length: Total duct run length [m]
        num_turns: Number of 90° turns
        include_diverter: Whether to include a flow diverter
        
    Returns:
        DuctworkSystemAssembly instance
    """
    # Calculate required diameter
    flow_rate_m3_s = flow_rate_m3_h / 3600
    area_required = flow_rate_m3_s / target_velocity
    diameter = np.sqrt(4 * area_required / np.pi)
    
    # Round to standard sizes (25mm increments)
    diameter = np.ceil(diameter * 40) / 40
    diameter = max(diameter, 0.1)  # Minimum 100mm
    
    params = DuctworkSystemParams(
        main_duct_diameter=diameter,
        total_length=run_length,
        num_elbows=num_turns,
        has_diverter=include_diverter,
        has_transition=True,
        elbow_r_d_ratio=2.0,  # Larger radius for particle-laden flow
    )
    return DuctworkSystemAssembly(params)


def create_simple_duct_run(diameter: float,
                           length: float,
                           start_position: Tuple[float, float, float] = (0, 0, 0),
                           direction: Tuple[float, float, float] = (0, 0, 1)) -> DuctworkSystemAssembly:
    """
    Create a simple straight duct run without fittings.
    
    Args:
        diameter: Duct diameter [m]
        length: Duct length [m]
        start_position: Starting position
        direction: Flow direction
        
    Returns:
        DuctworkSystemAssembly instance
    """
    params = DuctworkSystemParams(
        main_duct_diameter=diameter,
        total_length=length,
        num_elbows=0,
        num_45_elbows=0,
        has_diverter=False,
        has_transition=False,
    )
    return DuctworkSystemAssembly(params)
