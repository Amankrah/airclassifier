"""
Safety and instrumentation system assembly for air classification systems.

This module provides assembly of safety equipment (explosion vents,
grounding points) and process instrumentation (pressure ports,
thermowells, sample ports, sight glasses) for complete system outfitting.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class SafetyInstrumentationParams:
    """
    Parameters for safety and instrumentation system.
    
    Attributes:
        vessel_volume: Protected vessel volume for vent sizing [m³]
        Kst: Dust explosion constant [bar·m/s]
        num_explosion_vents: Number of explosion vents
        vent_type: Explosion vent type
        num_grounding_points: Number of grounding points
        num_pressure_ports: Number of pressure measurement points
        num_temp_ports: Number of temperature measurement points
        num_sample_ports: Number of sample extraction ports
        num_sight_glasses: Number of inspection windows
        has_level_sensor: Whether to include level sensor ports
    """
    vessel_volume: float = 1.0
    Kst: float = 150  # Legume dust typical
    num_explosion_vents: int = 1
    vent_type: str = "rupture_panel"
    num_grounding_points: int = 4
    num_pressure_ports: int = 2
    num_temp_ports: int = 2
    num_sample_ports: int = 1
    num_sight_glasses: int = 1
    has_level_sensor: bool = False
    
    @property
    def required_vent_area(self) -> float:
        """Calculate required vent area per EN 14491."""
        # Simplified calculation
        C = 0.1
        Pstat = 0.1  # bar
        Pred_max = 0.5  # bar
        delta_P = max(Pred_max - Pstat, 0.1)
        Av = C * self.vessel_volume ** (2/3) * self.Kst / (100 * np.sqrt(delta_P))
        return max(Av, 0.01)


class SafetyInstrumentationAssembly:
    """
    Complete safety and instrumentation system assembly.
    
    Assembles explosion vents, grounding points, pressure ports,
    temperature ports, sample ports, and sight glasses.
    """
    
    def __init__(self, params: SafetyInstrumentationParams):
        """
        Initialize safety and instrumentation assembly.
        
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
        """Create all safety and instrumentation components."""
        # Lazy imports
        from ..components.safety import (
            create_rupture_panel, create_hinged_explosion_door, create_recoil_vent,
            create_weld_stud_ground, calculate_vent_area
        )
        from ..components.instrumentation import (
            create_flush_pressure_port, create_threaded_thermowell,
            create_ball_valve_sample_port, create_standard_sight_glass
        )
        
        p = self.params
        
        # Calculate vent area and create explosion vents
        vent_area_each = p.required_vent_area / max(p.num_explosion_vents, 1)
        
        for i in range(p.num_explosion_vents):
            # Position vents around vessel (simplified - top placement)
            angle = 2 * np.pi * i / max(p.num_explosion_vents, 1)
            x = 0.5 * np.cos(angle)
            y = 0.5 * np.sin(angle)
            z = 1.0  # Top of vessel
            
            if p.vent_type == "rupture_panel":
                vent = create_rupture_panel(
                    vent_area=vent_area_each,
                    center=(x, y, z),
                    normal=(0, 0, 1)
                )
            elif p.vent_type == "hinged_door":
                vent = create_hinged_explosion_door(
                    vent_area=vent_area_each,
                    center=(x, y, z),
                    normal=(0, 0, 1)
                )
            else:
                vent = create_recoil_vent(
                    vent_area=vent_area_each,
                    center=(x, y, z),
                    normal=(0, 0, 1)
                )
            
            name = f'explosion_vent_{i}'
            self._components[name] = vent
            self._component_positions[name] = (x, y, z)
        
        # Create grounding points
        for i in range(p.num_grounding_points):
            angle = 2 * np.pi * i / max(p.num_grounding_points, 1)
            x = 0.8 * np.cos(angle)
            y = 0.8 * np.sin(angle)
            z = 0.3  # Lower portion of vessel
            
            ground = create_weld_stud_ground(
                location=(x, y, z),
                surface_normal=(np.cos(angle), np.sin(angle), 0)
            )
            name = f'grounding_point_{i}'
            self._components[name] = ground
            self._component_positions[name] = (x, y, z)
        
        # Create pressure ports
        for i in range(p.num_pressure_ports):
            z = 0.2 + 0.6 * i / max(p.num_pressure_ports - 1, 1)
            angle = np.pi / 4  # Side of vessel
            x = 0.6 * np.cos(angle)
            y = 0.6 * np.sin(angle)
            
            port = create_flush_pressure_port(
                location=(x, y, z),
                surface_normal=(np.cos(angle), np.sin(angle), 0)
            )
            name = f'pressure_port_{i}'
            self._components[name] = port
            self._component_positions[name] = (x, y, z)
        
        # Create temperature ports
        for i in range(p.num_temp_ports):
            z = 0.3 + 0.4 * i / max(p.num_temp_ports - 1, 1)
            angle = -np.pi / 4  # Opposite side
            x = 0.6 * np.cos(angle)
            y = 0.6 * np.sin(angle)
            
            port = create_threaded_thermowell(
                location=(x, y, z),
                surface_normal=(np.cos(angle), np.sin(angle), 0),
                immersion_length=0.1
            )
            name = f'temp_port_{i}'
            self._components[name] = port
            self._component_positions[name] = (x, y, z)
        
        # Create sample ports
        for i in range(p.num_sample_ports):
            z = 0.5
            angle = np.pi  # Back of vessel
            x = 0.6 * np.cos(angle)
            y = 0.6 * np.sin(angle)
            
            port = create_ball_valve_sample_port(
                location=(x, y, z),
                surface_normal=(np.cos(angle), np.sin(angle), 0)
            )
            name = f'sample_port_{i}'
            self._components[name] = port
            self._component_positions[name] = (x, y, z)
        
        # Create sight glasses
        for i in range(p.num_sight_glasses):
            z = 0.6
            angle = 0  # Front of vessel
            x = 0.6 * np.cos(angle)
            y = 0.6 * np.sin(angle)
            
            glass = create_standard_sight_glass(
                location=(x, y, z),
                surface_normal=(np.cos(angle), np.sin(angle), 0),
                diameter=0.1
            )
            name = f'sight_glass_{i}'
            self._components[name] = glass
            self._component_positions[name] = (x, y, z)
    
    def get_component(self, name: str) -> Any:
        """Get a specific component by name."""
        return self._components.get(name)
    
    def get_component_names(self) -> List[str]:
        """Get list of all component names."""
        return list(self._components.keys())
    
    def get_component_position(self, name: str) -> Tuple[float, float, float]:
        """Get position of a component."""
        return self._component_positions.get(name)
    
    def get_components_by_type(self, prefix: str) -> Dict[str, Any]:
        """
        Get all components matching a type prefix.
        
        Args:
            prefix: Component name prefix (e.g., "explosion_vent", "grounding_point")
            
        Returns:
            Dictionary of matching components
        """
        return {k: v for k, v in self._components.items() if k.startswith(prefix)}
    
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
        """Get bounding box of entire system."""
        verts = self.vertices
        if len(verts) == 0:
            return np.zeros(3), np.zeros(3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of safety and instrumentation system."""
        bounds_min, bounds_max = self.get_bounds()
        
        return {
            'vessel_volume_m3': self.params.vessel_volume,
            'required_vent_area_m2': self.params.required_vent_area,
            'num_explosion_vents': self.params.num_explosion_vents,
            'vent_type': self.params.vent_type,
            'num_grounding_points': self.params.num_grounding_points,
            'num_pressure_ports': self.params.num_pressure_ports,
            'num_temp_ports': self.params.num_temp_ports,
            'num_sample_ports': self.params.num_sample_ports,
            'num_sight_glasses': self.params.num_sight_glasses,
            'total_components': len(self._components),
            'total_vertices': len(self.vertices),
            'total_triangles': len(self.indices) // 3,
        }
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )
    
    def print_summary(self):
        """Print a summary of the safety and instrumentation system."""
        summary = self.get_system_summary()
        
        print("=" * 60)
        print("SAFETY & INSTRUMENTATION SYSTEM SUMMARY")
        print("=" * 60)
        print(f"Protected Volume: {summary['vessel_volume_m3']:.2f} m³")
        print(f"Required Vent Area: {summary['required_vent_area_m2']*10000:.1f} cm²")
        print("-" * 60)
        print("SAFETY COMPONENTS:")
        print(f"  Explosion Vents: {summary['num_explosion_vents']} ({summary['vent_type']})")
        print(f"  Grounding Points: {summary['num_grounding_points']}")
        print("-" * 60)
        print("INSTRUMENTATION:")
        print(f"  Pressure Ports: {summary['num_pressure_ports']}")
        print(f"  Temperature Ports: {summary['num_temp_ports']}")
        print(f"  Sample Ports: {summary['num_sample_ports']}")
        print(f"  Sight Glasses: {summary['num_sight_glasses']}")
        print("-" * 60)
        print("GEOMETRY:")
        print(f"  Total Components: {summary['total_components']}")
        print(f"  Total Vertices: {summary['total_vertices']}")
        print(f"  Total Triangles: {summary['total_triangles']}")
        print("=" * 60)


# Factory functions

def create_standard_safety_instrumentation(vessel_volume: float = 1.0,
                                           **kwargs) -> SafetyInstrumentationAssembly:
    """
    Create a standard safety and instrumentation package.
    
    Args:
        vessel_volume: Protected vessel volume [m³]
        **kwargs: Additional parameters
        
    Returns:
        SafetyInstrumentationAssembly instance
    """
    params = SafetyInstrumentationParams(
        vessel_volume=vessel_volume,
        **kwargs
    )
    return SafetyInstrumentationAssembly(params)


def create_minimal_instrumentation(vessel_volume: float = 1.0) -> SafetyInstrumentationAssembly:
    """
    Create minimal safety/instrumentation for small systems.
    
    Args:
        vessel_volume: Protected vessel volume [m³]
        
    Returns:
        SafetyInstrumentationAssembly instance
    """
    params = SafetyInstrumentationParams(
        vessel_volume=vessel_volume,
        num_explosion_vents=1,
        num_grounding_points=2,
        num_pressure_ports=1,
        num_temp_ports=1,
        num_sample_ports=0,
        num_sight_glasses=1,
    )
    return SafetyInstrumentationAssembly(params)


def create_full_instrumentation(vessel_volume: float,
                                Kst: float = 150) -> SafetyInstrumentationAssembly:
    """
    Create comprehensive safety/instrumentation for production systems.
    
    Args:
        vessel_volume: Protected vessel volume [m³]
        Kst: Dust explosion constant [bar·m/s]
        
    Returns:
        SafetyInstrumentationAssembly instance
    """
    # Scale instrumentation with vessel size
    num_vents = max(1, int(vessel_volume / 2))
    num_grounds = max(4, int(vessel_volume * 4))
    
    params = SafetyInstrumentationParams(
        vessel_volume=vessel_volume,
        Kst=Kst,
        num_explosion_vents=num_vents,
        num_grounding_points=num_grounds,
        num_pressure_ports=4,
        num_temp_ports=4,
        num_sample_ports=2,
        num_sight_glasses=2,
        has_level_sensor=True,
    )
    return SafetyInstrumentationAssembly(params)
