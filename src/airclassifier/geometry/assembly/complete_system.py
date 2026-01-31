"""
Complete air classifier system assembly.

This module provides the master assembly that integrates all system phases:
- Phase 1: Classification (Zigzag + Cyclones + Bag Filter)
- Phase 2: Feed System (Hopper + Airlock + Screw Feeder + Deagglomerator)
- Phase 3: Air System (Blower + Filter + Damper)
- Phase 4: Ductwork (Ducts + Transitions + Elbows + Diverters)
- Phase 5: Safety & Instrumentation (Vents + Grounding + Ports)
- Phase 6: Support & Exhaust (Frame + Legs + Silencer + Stack)

The complete system positions all components in 3D space with proper
connections and mounting of instrumentation.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class CompleteSystemParams:
    """
    Parameters for complete air classifier system.
    
    Attributes:
        throughput_kg_h: Design throughput [kg/h]
        cut_size_um: Target cut size [μm]
        air_flow_m3_h: Design air flow rate [m³/h]
        
        # Layout parameters
        feed_position: Feed system position offset (x, y, z) [m]
        classifier_position: Classification system position (x, y, z) [m]
        air_system_position: Air system position (x, y, z) [m]
        
        # Include flags
        include_feed_system: Whether to include feed system
        include_air_system: Whether to include air system
        include_ductwork: Whether to include connecting ductwork
        include_safety: Whether to include safety equipment
        include_instrumentation: Whether to include instrumentation
        include_support_structure: Whether to include support frame
        include_exhaust: Whether to include silencer and stack
        
        # Sizing
        classifier_width: Zigzag classifier width [m]
        cyclone_diameter: Primary cyclone diameter [m]
        frame_width: Support frame width [m]
        frame_depth: Support frame depth [m]
        frame_height: Support frame height [m]
    """
    throughput_kg_h: float = 500.0
    cut_size_um: float = 20.0
    air_flow_m3_h: float = 3000.0
    
    # Layout positions
    feed_position: Tuple[float, float, float] = (-2.0, 0.0, 0.0)
    classifier_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    air_system_position: Tuple[float, float, float] = (-3.0, -2.0, 0.0)
    
    # Include flags
    include_feed_system: bool = True
    include_air_system: bool = True
    include_ductwork: bool = True
    include_safety: bool = True
    include_instrumentation: bool = True
    include_support_structure: bool = True
    include_exhaust: bool = True
    
    # Sizing parameters
    classifier_width: float = 0.15
    cyclone_diameter: float = 0.3
    hopper_diameter: float = 0.6
    main_duct_diameter: float = 0.2
    frame_width: float = 4.0
    frame_depth: float = 3.0
    frame_height: float = 3.5
    stack_height: float = 4.0


class CompleteClassifierAssembly:
    """
    Complete air classifier system assembly.
    
    Integrates all system phases into a single unified assembly
    with proper positioning and connections.
    """
    
    def __init__(self, params: CompleteSystemParams):
        """
        Initialize complete classifier assembly.
        
        Args:
            params: System parameters
        """
        self.params = params
        self._subsystems: Dict[str, Any] = {}
        self._components: Dict[str, Any] = {}
        self._instrumentation: Dict[str, Any] = {}
        self._vertices: Optional[np.ndarray] = None
        self._indices: Optional[np.ndarray] = None
        self._build_system()
    
    def _build_system(self):
        """Build the complete system assembly."""
        p = self.params
        
        # Build support structure first (defines the frame of reference)
        if p.include_support_structure:
            self._build_support_structure()
        
        # Build classification system (core)
        self._build_classification_system()
        
        # Build feed system
        if p.include_feed_system:
            self._build_feed_system()
        
        # Build air system
        if p.include_air_system:
            self._build_air_system()
        
        # Build connecting ductwork
        if p.include_ductwork:
            self._build_ductwork()
        
        # Build exhaust system
        if p.include_exhaust:
            self._build_exhaust_system()
        
        # Add safety equipment
        if p.include_safety:
            self._add_safety_equipment()
        
        # Add instrumentation
        if p.include_instrumentation:
            self._add_instrumentation()
    
    def _build_support_structure(self):
        """Build support frame and legs."""
        from .support_exhaust import create_standard_support_exhaust
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        support = create_standard_support_exhaust(
            frame_width=p.frame_width,
            frame_depth=p.frame_depth,
            frame_height=p.frame_height,
            has_exhaust_stack=False,  # We'll add this separately
            has_silencer=False,
            center=(cx, cy, cz)
        )
        self._subsystems['support_structure'] = support
    
    def _build_classification_system(self):
        """Build the classification system (zigzag + cyclones + bag filter)."""
        from .classification import create_standard_classification_system
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        # Position classification on top of support frame
        if p.include_support_structure:
            class_z = cz + p.frame_height + 0.5  # 0.5m above frame
        else:
            class_z = cz + 0.5
        
        classification = create_standard_classification_system(device="cpu")
        self._subsystems['classification'] = classification
        
        # Store position offset for mesh transformation
        self._subsystems['classification_offset'] = (cx, cy, class_z)
    
    def _build_feed_system(self):
        """Build the feed system (hopper + airlock + screw + deagglomerator)."""
        from .feed_system import create_standard_feed_system
        
        p = self.params
        fx, fy, fz = p.feed_position
        
        # Position feed system elevated
        if p.include_support_structure:
            feed_z = fz + p.frame_height + 1.0
        else:
            feed_z = fz + 1.0
        
        feed = create_standard_feed_system(device="cpu")
        self._subsystems['feed_system'] = feed
        self._subsystems['feed_system_offset'] = (fx, fy, feed_z)
    
    def _build_air_system(self):
        """Build the air system (blower + filter + damper)."""
        from .air_system import create_standard_air_system
        
        p = self.params
        ax, ay, az = p.air_system_position
        
        air = create_standard_air_system(device="cpu")
        self._subsystems['air_system'] = air
        self._subsystems['air_system_offset'] = (ax, ay, az + 0.5)
    
    def _build_ductwork(self):
        """Build connecting ductwork between systems."""
        from .ductwork import create_standard_ductwork
        
        p = self.params
        
        # Main supply duct from air system to classifier
        supply_duct = create_standard_ductwork(
            main_diameter=p.main_duct_diameter,
            total_length=3.0,
            num_elbows=2,
            has_diverter=False
        )
        self._subsystems['supply_ductwork'] = supply_duct
        
        # Return duct from bag filter to atmosphere (with silencer/stack)
        return_duct = create_standard_ductwork(
            main_diameter=p.main_duct_diameter * 1.2,
            total_length=2.0,
            num_elbows=1,
            has_diverter=False
        )
        self._subsystems['return_ductwork'] = return_duct
    
    def _build_exhaust_system(self):
        """Build exhaust silencer and stack."""
        from ..components.silencer import create_absorptive_silencer
        from ..components.exhaust_stack import create_standard_exhaust_stack
        
        p = self.params
        cx, cy, cz = p.classifier_position
        
        # Silencer position (above frame, offset to side)
        silencer_x = cx + p.frame_width / 2 + 0.5
        silencer_z = cz + p.frame_height
        
        silencer = create_absorptive_silencer(
            diameter=p.main_duct_diameter * 1.2,
            length=1.0,
            center=(silencer_x, cy, silencer_z),
            direction=(0, 0, 1)
        )
        self._components['silencer'] = silencer
        
        # Stack on top of silencer
        stack_z = silencer_z + 1.0
        stack = create_standard_exhaust_stack(
            diameter=p.main_duct_diameter * 1.2,
            height=p.stack_height,
            center=(silencer_x, cy, stack_z)
        )
        self._components['exhaust_stack'] = stack
    
    def _add_safety_equipment(self):
        """Add explosion vents and grounding points."""
        from ..components.safety import (
            create_rupture_panel, create_weld_stud_ground
        )
        
        p = self.params
        cx, cy, cz = p.classifier_position
        base_z = cz + p.frame_height if p.include_support_structure else cz
        
        # Explosion vent on cyclone body
        cyclone_vent = create_rupture_panel(
            vent_area=0.05,
            center=(cx + 0.5, cy, base_z + 1.5),
            normal=(1, 0, 0)
        )
        self._instrumentation['cyclone_explosion_vent'] = cyclone_vent
        
        # Explosion vent on bag filter
        filter_vent = create_rupture_panel(
            vent_area=0.1,
            center=(cx + 1.5, cy, base_z + 2.0),
            normal=(0, 0, 1)
        )
        self._instrumentation['filter_explosion_vent'] = filter_vent
        
        # Grounding points on major equipment
        grounding_locations = [
            (cx, cy + 0.5, base_z + 0.3),      # Classifier
            (cx + 0.5, cy, base_z + 0.3),       # Cyclone
            (cx + 1.5, cy, base_z + 0.3),       # Bag filter
        ]
        
        if p.include_feed_system:
            fx, fy, fz = p.feed_position
            feed_base_z = fz + p.frame_height if p.include_support_structure else fz
            grounding_locations.append((fx, fy + 0.4, feed_base_z + 0.3))
        
        for i, loc in enumerate(grounding_locations):
            ground = create_weld_stud_ground(
                location=loc,
                surface_normal=(0, 1, 0)
            )
            self._instrumentation[f'grounding_point_{i}'] = ground
    
    def _add_instrumentation(self):
        """Add pressure, temperature, and sample ports."""
        from ..components.instrumentation import (
            create_flush_pressure_port, create_threaded_thermowell,
            create_ball_valve_sample_port, create_standard_sight_glass
        )
        
        p = self.params
        cx, cy, cz = p.classifier_position
        base_z = cz + p.frame_height if p.include_support_structure else cz
        
        # Pressure ports - measure pressure drop across system
        pressure_locations = [
            ('classifier_inlet', (cx - 0.3, cy, base_z + 1.0), (-1, 0, 0)),
            ('classifier_outlet', (cx + 0.3, cy, base_z + 1.0), (1, 0, 0)),
            ('cyclone_inlet', (cx + 0.3, cy, base_z + 1.5), (1, 0, 0)),
            ('filter_inlet', (cx + 1.2, cy, base_z + 1.5), (-1, 0, 0)),
            ('filter_outlet', (cx + 1.8, cy, base_z + 1.5), (1, 0, 0)),
        ]
        
        for name, loc, normal in pressure_locations:
            port = create_flush_pressure_port(
                location=loc,
                surface_normal=normal
            )
            self._instrumentation[f'pressure_{name}'] = port
        
        # Temperature ports
        temp_locations = [
            ('inlet_air', (cx - 0.5, cy, base_z + 0.8), (-1, 0, 0)),
            ('exhaust_air', (cx + 2.0, cy, base_z + 1.2), (1, 0, 0)),
        ]
        
        for name, loc, normal in temp_locations:
            port = create_threaded_thermowell(
                location=loc,
                surface_normal=normal,
                immersion_length=0.08
            )
            self._instrumentation[f'temp_{name}'] = port
        
        # Sample ports
        sample_locations = [
            ('feed', (cx - 0.2, cy + 0.2, base_z + 0.5), (0, 1, 0)),
            ('coarse_product', (cx + 0.5, cy + 0.3, base_z + 0.3), (0, 1, 0)),
            ('fine_product', (cx + 1.5, cy + 0.3, base_z + 0.3), (0, 1, 0)),
        ]
        
        for name, loc, normal in sample_locations:
            port = create_ball_valve_sample_port(
                location=loc,
                surface_normal=normal
            )
            self._instrumentation[f'sample_{name}'] = port
        
        # Sight glasses
        sight_glass_locations = [
            ('classifier', (cx, cy + 0.3, base_z + 1.0), (0, 1, 0)),
            ('cyclone', (cx + 0.5, cy + 0.3, base_z + 1.5), (0, 1, 0)),
        ]
        
        for name, loc, normal in sight_glass_locations:
            glass = create_standard_sight_glass(
                location=loc,
                surface_normal=normal,
                diameter=0.1
            )
            self._instrumentation[f'sight_glass_{name}'] = glass
    
    def build_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build combined mesh from all subsystems and components.
        
        Returns:
            Tuple of (vertices, indices)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0
        
        # Add subsystem meshes with offsets
        for name, subsystem in self._subsystems.items():
            if name.endswith('_offset'):
                continue
            
            verts, idx = subsystem.build_mesh()
            
            # Apply position offset if available
            offset_key = f'{name}_offset'
            if offset_key in self._subsystems:
                offset = np.array(self._subsystems[offset_key])
                verts = verts + offset
            
            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)
        
        # Add individual components
        for name, component in self._components.items():
            verts = component.vertices
            idx = component.indices
            
            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)
        
        # Add instrumentation
        for name, instrument in self._instrumentation.items():
            verts = instrument.vertices
            idx = instrument.indices
            
            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
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
    
    def get_subsystem(self, name: str) -> Any:
        """Get a subsystem by name."""
        return self._subsystems.get(name)
    
    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        return self._components.get(name)
    
    def get_instrument(self, name: str) -> Any:
        """Get an instrument by name."""
        return self._instrumentation.get(name)
    
    def get_all_subsystem_names(self) -> List[str]:
        """Get names of all subsystems."""
        return [k for k in self._subsystems.keys() if not k.endswith('_offset')]
    
    def get_all_component_names(self) -> List[str]:
        """Get names of all components."""
        return list(self._components.keys())
    
    def get_all_instrument_names(self) -> List[str]:
        """Get names of all instruments."""
        return list(self._instrumentation.keys())
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of entire system."""
        verts = self.vertices
        if len(verts) == 0:
            return np.zeros(3), np.zeros(3)
        return verts.min(axis=0), verts.max(axis=0)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the complete system."""
        bounds_min, bounds_max = self.get_bounds()
        dimensions = bounds_max - bounds_min
        
        # Count instrumentation by type
        instrument_counts = {}
        for name in self._instrumentation.keys():
            prefix = name.split('_')[0]
            instrument_counts[prefix] = instrument_counts.get(prefix, 0) + 1
        
        return {
            'design_throughput_kg_h': self.params.throughput_kg_h,
            'design_cut_size_um': self.params.cut_size_um,
            'design_air_flow_m3_h': self.params.air_flow_m3_h,
            'num_subsystems': len(self.get_all_subsystem_names()),
            'num_components': len(self._components),
            'num_instruments': len(self._instrumentation),
            'instrument_breakdown': instrument_counts,
            'total_vertices': len(self.vertices),
            'total_triangles': len(self.indices) // 3,
            'dimensions_m': list(dimensions),
            'bounds_min': list(bounds_min),
            'bounds_max': list(bounds_max),
            'includes': {
                'feed_system': self.params.include_feed_system,
                'air_system': self.params.include_air_system,
                'ductwork': self.params.include_ductwork,
                'safety': self.params.include_safety,
                'instrumentation': self.params.include_instrumentation,
                'support_structure': self.params.include_support_structure,
                'exhaust': self.params.include_exhaust,
            }
        }
    
    def get_bill_of_materials(self) -> List[Dict[str, Any]]:
        """
        Generate a bill of materials for the system.
        
        Returns:
            List of component entries with name, type, and quantity
        """
        bom = []
        
        # Subsystems
        subsystem_types = {
            'classification': 'Classification System',
            'feed_system': 'Feed System',
            'air_system': 'Air System',
            'support_structure': 'Support Structure',
            'supply_ductwork': 'Supply Ductwork',
            'return_ductwork': 'Return Ductwork',
        }
        
        for name, desc in subsystem_types.items():
            if name in self._subsystems:
                bom.append({
                    'item': desc,
                    'type': 'Subsystem',
                    'quantity': 1
                })
        
        # Individual components
        for name in self._components.keys():
            bom.append({
                'item': name.replace('_', ' ').title(),
                'type': 'Component',
                'quantity': 1
            })
        
        # Instrumentation
        instrument_counts = {}
        for name in self._instrumentation.keys():
            # Group by type
            if 'pressure' in name:
                key = 'Pressure Port'
            elif 'temp' in name:
                key = 'Thermowell'
            elif 'sample' in name:
                key = 'Sample Port'
            elif 'sight_glass' in name:
                key = 'Sight Glass'
            elif 'explosion_vent' in name:
                key = 'Explosion Vent'
            elif 'grounding' in name:
                key = 'Grounding Point'
            else:
                key = name.replace('_', ' ').title()
            
            instrument_counts[key] = instrument_counts.get(key, 0) + 1
        
        for item, qty in instrument_counts.items():
            bom.append({
                'item': item,
                'type': 'Instrumentation',
                'quantity': qty
            })
        
        return bom
    
    def to_warp_mesh(self) -> Any:
        """Create a Warp mesh object."""
        if wp is None:
            raise ImportError("NVIDIA Warp is required")
        return wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3),
            indices=wp.array(self.indices.flatten(), dtype=wp.int32)
        )
    
    def print_summary(self):
        """Print a comprehensive summary of the complete system."""
        summary = self.get_system_summary()
        
        print("=" * 70)
        print("COMPLETE AIR CLASSIFIER SYSTEM")
        print("=" * 70)
        print(f"Design Throughput: {summary['design_throughput_kg_h']:.0f} kg/h")
        print(f"Target Cut Size: {summary['design_cut_size_um']:.0f} μm")
        print(f"Air Flow Rate: {summary['design_air_flow_m3_h']:.0f} m³/h")
        print("-" * 70)
        print("SYSTEM COMPOSITION:")
        print(f"  Subsystems: {summary['num_subsystems']}")
        print(f"  Components: {summary['num_components']}")
        print(f"  Instruments: {summary['num_instruments']}")
        print("-" * 70)
        print("INSTRUMENTATION BREAKDOWN:")
        for inst_type, count in summary['instrument_breakdown'].items():
            print(f"  {inst_type.title()}: {count}")
        print("-" * 70)
        print("INCLUDED SYSTEMS:")
        for system, included in summary['includes'].items():
            status = "✓" if included else "✗"
            print(f"  [{status}] {system.replace('_', ' ').title()}")
        print("-" * 70)
        print("GEOMETRY:")
        print(f"  Dimensions: {summary['dimensions_m'][0]:.1f} x {summary['dimensions_m'][1]:.1f} x {summary['dimensions_m'][2]:.1f} m")
        print(f"  Total Vertices: {summary['total_vertices']:,}")
        print(f"  Total Triangles: {summary['total_triangles']:,}")
        print("=" * 70)
    
    def print_bill_of_materials(self):
        """Print the bill of materials."""
        bom = self.get_bill_of_materials()
        
        print("=" * 50)
        print("BILL OF MATERIALS")
        print("=" * 50)
        print(f"{'Item':<30} {'Type':<15} {'Qty':>5}")
        print("-" * 50)
        
        for entry in bom:
            print(f"{entry['item']:<30} {entry['type']:<15} {entry['quantity']:>5}")
        
        print("-" * 50)
        print(f"{'TOTAL ITEMS':<30} {'':<15} {len(bom):>5}")
        print("=" * 50)


# Factory functions

def create_complete_classifier_system(throughput_kg_h: float = 500,
                                      cut_size_um: float = 20,
                                      **kwargs) -> CompleteClassifierAssembly:
    """
    Create a complete air classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        cut_size_um: Target cut size [μm]
        **kwargs: Additional parameters
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        cut_size_um=cut_size_um,
        **kwargs
    )
    return CompleteClassifierAssembly(params)


def create_pilot_scale_system(throughput_kg_h: float = 100) -> CompleteClassifierAssembly:
    """
    Create a pilot-scale classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        air_flow_m3_h=1000,
        classifier_width=0.1,
        cyclone_diameter=0.2,
        hopper_diameter=0.4,
        main_duct_diameter=0.15,
        frame_width=2.5,
        frame_depth=2.0,
        frame_height=2.5,
        stack_height=3.0,
    )
    return CompleteClassifierAssembly(params)


def create_production_scale_system(throughput_kg_h: float = 2000) -> CompleteClassifierAssembly:
    """
    Create a production-scale classifier system.
    
    Args:
        throughput_kg_h: Design throughput [kg/h]
        
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=throughput_kg_h,
        air_flow_m3_h=8000,
        classifier_width=0.25,
        cyclone_diameter=0.5,
        hopper_diameter=1.0,
        main_duct_diameter=0.35,
        frame_width=6.0,
        frame_depth=4.0,
        frame_height=5.0,
        stack_height=8.0,
    )
    return CompleteClassifierAssembly(params)


def create_minimal_classifier_system() -> CompleteClassifierAssembly:
    """
    Create a minimal classifier with only core components.
    
    Returns:
        CompleteClassifierAssembly instance
    """
    params = CompleteSystemParams(
        throughput_kg_h=200,
        include_feed_system=False,
        include_air_system=False,
        include_ductwork=False,
        include_safety=False,
        include_instrumentation=False,
        include_support_structure=False,
        include_exhaust=False,
    )
    return CompleteClassifierAssembly(params)
