"""
Connection port definitions for component alignment.

This module provides a standardized way to define inlet/outlet connection
points on components, enabling proper port-to-port alignment when assembling
systems. Each port has a position, direction, diameter, and type.

Usage:
    # Define ports on a component
    component.ports = {
        'inlet': ConnectionPort(position=(0, 0, 0), direction=(0, 0, -1), diameter=0.1),
        'outlet': ConnectionPort(position=(0, 0, 0.5), direction=(0, 0, 1), diameter=0.1),
    }
    
    # Connect two components
    alignment = calculate_alignment(comp_a.ports['outlet'], comp_b.ports['inlet'])
    comp_b_position = alignment.position
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
from enum import Enum
import numpy as np


class PortType(Enum):
    """Type of connection port."""
    CIRCULAR = "circular"           # Round pipe/duct
    RECTANGULAR = "rectangular"     # Rectangular duct
    FLANGED = "flanged"            # Flanged connection
    SLIP = "slip"                  # Slip-fit connection
    THREADED = "threaded"          # Threaded connection
    GRAVITY = "gravity"            # Gravity discharge (open)


@dataclass
class ConnectionPort:
    """
    Defines a connection point on a component.
    
    Attributes:
        position: Local position of port center relative to component origin (x, y, z) [m]
        direction: Outward normal direction (dx, dy, dz) - points away from component
        diameter: Port diameter [m] for circular ports
        width: Port width [m] for rectangular ports
        height: Port height [m] for rectangular ports
        port_type: Type of port connection
        name: Optional name for identification
        flange_diameter: Outer diameter of flange if flanged [m]
        compatible_types: List of compatible port types for connection
    """
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    diameter: float = 0.0
    width: float = 0.0
    height: float = 0.0
    port_type: PortType = PortType.CIRCULAR
    name: str = ""
    flange_diameter: float = 0.0
    compatible_types: List[PortType] = field(default_factory=lambda: [PortType.CIRCULAR])
    
    def __post_init__(self):
        # Normalize direction vector
        d = np.array(self.direction)
        norm = np.linalg.norm(d)
        if norm > 0:
            self.direction = tuple(d / norm)
    
    @property
    def position_array(self) -> np.ndarray:
        """Get position as numpy array."""
        return np.array(self.position, dtype=np.float32)
    
    @property
    def direction_array(self) -> np.ndarray:
        """Get direction as numpy array."""
        return np.array(self.direction, dtype=np.float32)
    
    @property
    def area(self) -> float:
        """Connection area [m²]."""
        if self.port_type == PortType.RECTANGULAR:
            return self.width * self.height
        else:
            return np.pi * (self.diameter / 2) ** 2
    
    def get_world_position(self, component_position: Tuple[float, float, float]) -> np.ndarray:
        """Get port position in world coordinates."""
        return self.position_array + np.array(component_position)
    
    def get_world_direction(self, rotation_matrix: np.ndarray = None) -> np.ndarray:
        """Get port direction in world coordinates."""
        if rotation_matrix is not None:
            return rotation_matrix @ self.direction_array
        return self.direction_array
    
    def is_compatible(self, other: 'ConnectionPort', tolerance: float = 0.01) -> Tuple[bool, str]:
        """
        Check if this port can connect to another port.
        
        Args:
            other: Port to connect to
            tolerance: Diameter matching tolerance (fraction)
            
        Returns:
            Tuple of (is_compatible, reason_if_not)
        """
        # Check type compatibility
        if other.port_type not in self.compatible_types and self.port_type not in other.compatible_types:
            return False, f"Incompatible types: {self.port_type} vs {other.port_type}"
        
        # Check size compatibility
        if self.port_type == PortType.CIRCULAR and other.port_type == PortType.CIRCULAR:
            if abs(self.diameter - other.diameter) > tolerance * max(self.diameter, other.diameter):
                return False, f"Diameter mismatch: {self.diameter:.3f} vs {other.diameter:.3f}"
        
        if self.port_type == PortType.RECTANGULAR and other.port_type == PortType.RECTANGULAR:
            if abs(self.width - other.width) > tolerance * max(self.width, other.width):
                return False, f"Width mismatch: {self.width:.3f} vs {other.width:.3f}"
            if abs(self.height - other.height) > tolerance * max(self.height, other.height):
                return False, f"Height mismatch: {self.height:.3f} vs {other.height:.3f}"
        
        return True, "Compatible"


@dataclass
class PortAlignment:
    """
    Result of calculating alignment between two ports.
    
    Attributes:
        position_offset: Position offset to apply to second component
        rotation_matrix: Rotation to apply to second component (if needed)
        gap: Gap between ports (positive = separation, negative = overlap)
        is_aligned: Whether ports are properly aligned
        message: Description of alignment status
    """
    position_offset: np.ndarray
    rotation_matrix: Optional[np.ndarray] = None
    gap: float = 0.0
    is_aligned: bool = True
    message: str = ""


def calculate_alignment(
    source_port: ConnectionPort,
    target_port: ConnectionPort,
    source_position: Tuple[float, float, float] = (0, 0, 0),
    gap: float = 0.0,
    align_directions: bool = True
) -> PortAlignment:
    """
    Calculate position offset to align target component's port with source component's port.
    
    This calculates where to position a target component so its inlet port
    aligns with the source component's outlet port.
    
    Args:
        source_port: Output port on source component (e.g., hopper discharge)
        target_port: Input port on target component (e.g., airlock inlet)
        source_position: World position of source component origin
        gap: Desired gap between ports [m] (0 for direct contact)
        align_directions: Whether to ensure port directions are opposing
        
    Returns:
        PortAlignment with position offset for target component
    """
    source_pos = np.array(source_position)
    
    # World position of source port
    source_port_world = source_pos + source_port.position_array
    
    # For proper connection, target port should be at source port location
    # plus a gap in the source port direction
    gap_offset = source_port.direction_array * gap
    
    # Target component position = source_port_world - target_port.position + gap
    # This places target's port at source's port location
    target_component_position = source_port_world - target_port.position_array + gap_offset
    
    # Check direction alignment
    # For connection, port directions should be opposing (dot product ≈ -1)
    dot = np.dot(source_port.direction_array, target_port.direction_array)
    
    is_aligned = True
    message = "Aligned"
    
    if align_directions and dot > -0.9:  # Not opposing
        if dot > 0.9:  # Same direction
            message = "Ports face same direction - may need rotation"
            is_aligned = False
        else:
            message = f"Ports not opposing (dot={dot:.2f}) - may need rotation"
            is_aligned = False
    
    # Check compatibility
    compatible, reason = source_port.is_compatible(target_port)
    if not compatible:
        is_aligned = False
        message = reason
    
    return PortAlignment(
        position_offset=target_component_position,
        gap=gap,
        is_aligned=is_aligned,
        message=message
    )


def connect_in_series(
    components: List[Any],
    port_pairs: List[Tuple[str, str]],
    start_position: Tuple[float, float, float] = (0, 0, 0),
    gaps: List[float] = None
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate positions for a series of components connected port-to-port.
    
    Args:
        components: List of components with .ports dictionary
        port_pairs: List of (outlet_name, inlet_name) tuples for connections
                   e.g., [('discharge', 'inlet'), ('outlet', 'inlet')]
        start_position: Position of first component
        gaps: List of gaps between each connection (defaults to 0)
        
    Returns:
        Dictionary mapping component index to world position
    """
    if gaps is None:
        gaps = [0.0] * len(port_pairs)
    
    positions = {0: np.array(start_position)}
    
    for i, (outlet_name, inlet_name) in enumerate(port_pairs):
        if i + 1 >= len(components):
            break
            
        source = components[i]
        target = components[i + 1]
        gap = gaps[i] if i < len(gaps) else 0.0
        
        # Get ports
        if not hasattr(source, 'ports') or outlet_name not in source.ports:
            raise ValueError(f"Component {i} has no port '{outlet_name}'")
        if not hasattr(target, 'ports') or inlet_name not in target.ports:
            raise ValueError(f"Component {i+1} has no port '{inlet_name}'")
        
        source_port = source.ports[outlet_name]
        target_port = target.ports[inlet_name]
        
        alignment = calculate_alignment(
            source_port, target_port,
            source_position=tuple(positions[i]),
            gap=gap
        )
        
        positions[i + 1] = alignment.position_offset
    
    # Convert to tuples
    return {i: tuple(pos) for i, pos in positions.items()}


def get_connection_point(
    component: Any,
    port_name: str,
    component_position: Tuple[float, float, float] = (0, 0, 0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the world position and direction of a component's port.
    
    Args:
        component: Component with .ports dictionary
        port_name: Name of the port
        component_position: World position of component origin
        
    Returns:
        Tuple of (position, direction) as numpy arrays
    """
    if not hasattr(component, 'ports') or port_name not in component.ports:
        raise ValueError(f"Component has no port '{port_name}'")
    
    port = component.ports[port_name]
    position = port.get_world_position(component_position)
    direction = port.direction_array
    
    return position, direction


def validate_assembly_connections(
    components: List[Any],
    positions: Dict[int, Tuple[float, float, float]],
    connections: List[Tuple[int, str, int, str]],
    tolerance: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Validate that assembly connections are properly aligned.
    
    Args:
        components: List of components with .ports dictionary
        positions: Dictionary of component index to world position
        connections: List of (comp_a_idx, port_a_name, comp_b_idx, port_b_name)
        tolerance: Position tolerance [m]
        
    Returns:
        List of validation results for each connection
    """
    results = []
    
    for comp_a_idx, port_a_name, comp_b_idx, port_b_name in connections:
        comp_a = components[comp_a_idx]
        comp_b = components[comp_b_idx]
        
        pos_a = positions.get(comp_a_idx, (0, 0, 0))
        pos_b = positions.get(comp_b_idx, (0, 0, 0))
        
        try:
            port_a = comp_a.ports[port_a_name]
            port_b = comp_b.ports[port_b_name]
            
            # Get world positions
            world_a = port_a.get_world_position(pos_a)
            world_b = port_b.get_world_position(pos_b)
            
            # Calculate distance
            distance = np.linalg.norm(world_a - world_b)
            
            # Check direction alignment
            dot = np.dot(port_a.direction_array, port_b.direction_array)
            
            # Check compatibility
            compatible, reason = port_a.is_compatible(port_b)
            
            is_valid = (
                distance < tolerance and 
                dot < -0.9 and  # Opposing directions
                compatible
            )
            
            results.append({
                'connection': f"{comp_a_idx}.{port_a_name} -> {comp_b_idx}.{port_b_name}",
                'distance': distance,
                'direction_dot': dot,
                'compatible': compatible,
                'compatibility_reason': reason,
                'is_valid': is_valid,
                'position_a': tuple(world_a),
                'position_b': tuple(world_b),
            })
            
        except (AttributeError, KeyError) as e:
            results.append({
                'connection': f"{comp_a_idx}.{port_a_name} -> {comp_b_idx}.{port_b_name}",
                'error': str(e),
                'is_valid': False,
            })
    
    return results


def print_connection_report(validation_results: List[Dict[str, Any]]):
    """Print a formatted report of connection validation results."""
    print("=" * 70)
    print("CONNECTION VALIDATION REPORT")
    print("=" * 70)
    
    valid_count = sum(1 for r in validation_results if r.get('is_valid', False))
    total = len(validation_results)
    
    print(f"\nSummary: {valid_count}/{total} connections valid\n")
    
    for i, result in enumerate(validation_results, 1):
        status = "[OK]" if result.get('is_valid', False) else "[X]"
        print(f"{status} Connection {i}: {result['connection']}")
        
        if 'error' in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    Distance: {result['distance']*1000:.2f} mm")
            print(f"    Direction alignment: {result['direction_dot']:.3f} (ideal: -1.0)")
            print(f"    Size compatible: {result['compatible']} - {result['compatibility_reason']}")
            if not result['is_valid']:
                if result['distance'] > 0.01:
                    print(f"    ISSUE: Gap/misalignment of {result['distance']*1000:.1f} mm")
                if result['direction_dot'] > -0.9:
                    print(f"    ISSUE: Port directions not properly opposing")
    
    print("=" * 70)
