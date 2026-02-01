"""
Test script for multi-cyclone assembly verification.

This script verifies:
1. Individual cyclone positions and dimensions
2. Internal duct connections (inlet/outlet alignment)
3. Coordinate flow through the series arrangement
4. Transition dimensions matching
5. Gap/overlap detection between components
"""

import sys
import numpy as np
from pathlib import Path

# Mock warp BEFORE any other imports
import types
_wp = types.ModuleType("warp")

class _MockVec3(np.ndarray):
    def __new__(cls, *args):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            arr = np.asarray(args[0], dtype=np.float32).view(cls)
        elif len(args) == 3:
            arr = np.array(args, dtype=np.float32).view(cls)
        else:
            arr = np.zeros(3, dtype=np.float32).view(cls)
        return arr

class _MockArray:
    """Mock warp array that can be used as type annotation."""
    def __init__(self, data=None, dtype=None, device=None, **kwargs):
        self.data = np.array(data) if data is not None else None
        self.dtype = dtype
        self.device = device
    @classmethod
    def __class_getitem__(cls, item):
        return cls

_wp.vec3 = _MockVec3
_wp.int32 = np.int32
_wp.int64 = np.int64
_wp.uint32 = np.uint32
_wp.uint64 = np.uint64
_wp.float32 = np.float32
_wp.float64 = np.float64
_wp.uint8 = np.uint8
_wp.bool_ = bool
_wp.Mesh = lambda **kwargs: None
_wp.array = _MockArray
_wp.constant = lambda x: x
_wp.func = lambda f: f
_wp.kernel = lambda f: f
_wp.struct = lambda cls: cls
_wp.static = lambda x: x
_wp.tid = lambda: 0
_wp.length = np.linalg.norm
_wp.normalize = lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
_wp.dot = np.dot
_wp.cross = np.cross
_wp.sqrt = np.sqrt
_wp.abs = np.abs
_wp.min = np.minimum
_wp.max = np.maximum
_wp.clamp = lambda x, a, b: np.clip(x, a, b)
_wp.atomic_add = lambda a, i, v: None
_wp.mesh_query_point = lambda *args: (0.0, 0.0, 0, 0)
_wp.mesh_query_aabb = lambda *args: None
_wp.LAUNCH_ASYNC = False
_wp.ScopedTimer = lambda name: type('ScopedTimer', (), {'__enter__': lambda s: s, '__exit__': lambda s, *a: None})()
_wp.array3d = _MockArray
_wp.launch = lambda *args, **kwargs: None
_wp.synchronize = lambda: None
_wp.copy = lambda *args, **kwargs: None
_wp.zeros = lambda *args, **kwargs: _MockArray()
_wp.empty = lambda *args, **kwargs: _MockArray()

# Install mock before importing anything from airclassifier
sys.modules['warp'] = _wp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title):
    """Print formatted section."""
    print(f"\n--- {title} ---")


def verify_connection(name, outlet_pos, outlet_dir, inlet_pos, inlet_dir, 
                      outlet_dim, inlet_dim, gap_tolerance=0.02):
    """
    Verify that an outlet connects properly to an inlet.
    
    Args:
        name: Connection name for reporting
        outlet_pos: (x, y, z) position of outlet
        inlet_pos: (x, y, z) position of inlet
        outlet_dir: Direction vector of outlet flow
        inlet_dir: Direction vector of inlet (facing direction)
        outlet_dim: Outlet dimension (diameter or (width, height))
        inlet_dim: Inlet dimension (diameter or (width, height))
        gap_tolerance: Maximum allowed gap [m]
    
    Returns:
        dict with verification results
    """
    outlet_pos = np.array(outlet_pos)
    inlet_pos = np.array(inlet_pos)
    outlet_dir = np.array(outlet_dir)
    inlet_dir = np.array(inlet_dir)
    
    # Normalize directions
    outlet_dir = outlet_dir / np.linalg.norm(outlet_dir)
    inlet_dir = inlet_dir / np.linalg.norm(inlet_dir)
    
    # Calculate distance between ports
    distance = np.linalg.norm(inlet_pos - outlet_pos)
    
    # Calculate direction alignment (outlet should point toward inlet)
    to_inlet = inlet_pos - outlet_pos
    if np.linalg.norm(to_inlet) > 1e-6:
        to_inlet = to_inlet / np.linalg.norm(to_inlet)
        direction_alignment = np.dot(outlet_dir, to_inlet)
    else:
        direction_alignment = 1.0  # Coincident points
    
    # Check if directions are opposing (inlet faces opposite to outlet flow)
    # For proper connection: outlet_dir should be opposite to inlet_dir
    direction_opposition = np.dot(outlet_dir, -inlet_dir)
    
    # Dimension matching
    if isinstance(outlet_dim, (int, float)):
        outlet_area = np.pi * (outlet_dim / 2) ** 2
    else:
        outlet_area = outlet_dim[0] * outlet_dim[1]
    
    if isinstance(inlet_dim, (int, float)):
        inlet_area = np.pi * (inlet_dim / 2) ** 2
    else:
        inlet_area = inlet_dim[0] * inlet_dim[1]
    
    area_ratio = outlet_area / inlet_area if inlet_area > 0 else 0
    
    # Status determination
    issues = []
    if distance > gap_tolerance:
        issues.append(f"GAP: {distance*1000:.1f}mm > {gap_tolerance*1000:.1f}mm tolerance")
    if direction_alignment < 0.9:  # Should be ~1.0 if outlet points toward inlet
        issues.append(f"MISALIGNED: outlet not pointing to inlet (alignment={direction_alignment:.2f})")
    if direction_opposition < 0.9:  # Should be ~1.0 if properly opposing
        issues.append(f"DIRECTION: ports not properly opposing (opposition={direction_opposition:.2f})")
    if area_ratio < 0.5 or area_ratio > 2.0:
        issues.append(f"SIZE MISMATCH: area ratio={area_ratio:.2f}")
    
    status = "OK" if not issues else "ISSUE"
    
    return {
        'name': name,
        'status': status,
        'distance_mm': distance * 1000,
        'direction_alignment': direction_alignment,
        'direction_opposition': direction_opposition,
        'outlet_area_mm2': outlet_area * 1e6,
        'inlet_area_mm2': inlet_area * 1e6,
        'area_ratio': area_ratio,
        'issues': issues,
        'outlet_pos': outlet_pos,
        'inlet_pos': inlet_pos,
        'outlet_dir': outlet_dir,
        'inlet_dir': inlet_dir
    }


def test_multi_cyclone_internal():
    """Test multi-cyclone internal connections."""
    from airclassifier.geometry.components.multi_cyclone import (
        MultiCycloneSystem, MultiCycloneParams, CycloneStageParams
    )
    
    print_header("MULTI-CYCLONE INTERNAL CONNECTION TEST")
    
    # Create test multi-cyclone system (same as classification.py)
    stages = [
        CycloneStageParams(
            name="primary",
            diameter=0.3,
            design_d50=40e-6,
        ),
        CycloneStageParams(
            name="secondary", 
            diameter=0.2,
            design_d50=20e-6,
        ),
        CycloneStageParams(
            name="tertiary",
            diameter=0.12,
            design_d50=10e-6,
        ),
    ]
    
    cyclone_spacing = 0.3 * 0.3  # primary diameter * 0.3
    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=cyclone_spacing,
        center=(0.0, 0.0, 0.0),
        resolution=48,
    )
    
    mc = MultiCycloneSystem(params)
    
    # Get positions
    positions = mc._calculate_positions()
    
    print_section("Cyclone Positions")
    for i, (stage, pos) in enumerate(zip(stages, positions)):
        D = stage.diameter
        cyl_h = D * stage.cylinder_height_ratio
        cone_h = D * stage.cone_height_ratio
        total_h = cyl_h + cone_h
        
        print(f"\n{stage.name.upper()} CYCLONE:")
        print(f"  Center (top of cylinder): ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
        print(f"  Diameter: {D*1000:.0f} mm")
        print(f"  Cylinder height: {cyl_h*1000:.0f} mm")
        print(f"  Cone height: {cone_h*1000:.0f} mm")
        print(f"  Total height: {total_h*1000:.0f} mm")
        
        # VF position
        vf_d = D * stage.vortex_finder_ratio
        vf_top_y = pos[1] + 0.05  # protrusion_above
        print(f"  Vortex finder top: Y={vf_top_y*1000:.1f} mm, D={vf_d*1000:.0f} mm")
        
        # Inlet position (only for primary in series)
        inlet_w = D * stage.inlet_width_ratio
        inlet_h = D * stage.inlet_height_ratio
        if i == 0:  # Primary has TangentialInlet
            inlet_x = pos[0] - D/2 - inlet_w
            inlet_y = pos[1] - inlet_h/2
            print(f"  Inlet (outer end): ({inlet_x*1000:.1f}, {inlet_y*1000:.1f}, {pos[2]*1000:.1f}) mm")
            print(f"  Inlet dimensions: {inlet_w*1000:.0f} x {inlet_h*1000:.0f} mm")
        else:
            # Secondary/Tertiary receive from internal ducts
            cyclone_surface_x = pos[0] - D/2
            inlet_center_y = pos[1] - inlet_h/2
            print(f"  Inlet surface point: ({cyclone_surface_x*1000:.1f}, {inlet_center_y*1000:.1f}, {pos[2]*1000:.1f}) mm")
            print(f"  Inlet dimensions: {inlet_w*1000:.0f} x {inlet_h*1000:.0f} mm (internal duct)")
    
    # Analyze internal duct connections
    print_section("Internal Duct Connections")
    
    if mc._connecting_ducts:
        print(f"\nTotal internal duct components: {len(mc._connecting_ducts)}")
        
        for j, (duct, duct_pos) in enumerate(mc._connecting_ducts):
            duct_type = type(duct).__name__
            print(f"\n  [{j+1}] {duct_type} at ({duct_pos[0]*1000:.1f}, {duct_pos[1]*1000:.1f}, {duct_pos[2]*1000:.1f}) mm")
            
            # Get duct-specific info
            if hasattr(duct, 'params'):
                p = duct.params
                if hasattr(p, 'diameter'):
                    print(f"      Diameter: {p.diameter*1000:.0f} mm")
                if hasattr(p, 'length'):
                    print(f"      Length: {p.length*1000:.0f} mm")
                if hasattr(p, 'width') and hasattr(p, 'height'):
                    print(f"      Size: {p.width*1000:.0f} x {p.height*1000:.0f} mm")
                if hasattr(p, 'direction'):
                    print(f"      Direction: {p.direction}")
                if hasattr(p, 'bend_radius'):
                    print(f"      Bend radius: {p.bend_radius*1000:.0f} mm")
                if hasattr(p, 'angle'):
                    print(f"      Angle: {p.angle}°")
                if hasattr(p, 'inlet_direction'):
                    print(f"      Inlet direction: {p.inlet_direction}")
                    
            # For elbows, show outlet position
            if hasattr(duct, 'get_outlet_position'):
                outlet_local = duct.get_outlet_position()
                outlet_world = (
                    duct_pos[0] + outlet_local[0],
                    duct_pos[1] + outlet_local[1],
                    duct_pos[2] + outlet_local[2]
                )
                print(f"      Outlet position: ({outlet_world[0]*1000:.1f}, {outlet_world[1]*1000:.1f}, {outlet_world[2]*1000:.1f}) mm")
            
            if hasattr(duct, 'get_outlet_direction'):
                outlet_dir = duct.get_outlet_direction()
                print(f"      Outlet direction: ({outlet_dir[0]:.2f}, {outlet_dir[1]:.2f}, {outlet_dir[2]:.2f})")
    else:
        print("  No internal duct connections found!")
    
    # Verify connection chain for series arrangement
    print_section("Connection Chain Verification")
    
    # For series: Primary VF -> (ducts) -> Secondary inlet -> (ducts) -> Tertiary inlet
    gap = 0.005  # 5mm gap
    
    for i in range(len(stages) - 1):
        current_stage = stages[i]
        next_stage = stages[i + 1]
        current_pos = positions[i]
        next_pos = positions[i + 1]
        
        # Source: current cyclone vortex finder
        D_curr = current_stage.diameter
        vf_d = D_curr * current_stage.vortex_finder_ratio
        vf_top_y = current_pos[1] + 0.05
        source_pos = (current_pos[0], vf_top_y, current_pos[2])
        source_dir = (0, 1, 0)  # Up
        
        # Target: next cyclone inlet (at cyclone body surface)
        D_next = next_stage.diameter
        inlet_w = D_next * next_stage.inlet_width_ratio
        inlet_h = D_next * next_stage.inlet_height_ratio
        target_x = next_pos[0] - D_next / 2  # Cyclone surface
        target_y = next_pos[1] - inlet_h / 2
        target_pos = (target_x, target_y, next_pos[2])
        target_dir = (-1, 0, 0)  # Faces -X (receives from +X direction)
        
        print(f"\n{current_stage.name.upper()} -> {next_stage.name.upper()}:")
        print(f"  Source (VF top): ({source_pos[0]*1000:.1f}, {source_pos[1]*1000:.1f}, {source_pos[2]*1000:.1f}) mm, D={vf_d*1000:.0f}mm")
        print(f"  Target (inlet):  ({target_pos[0]*1000:.1f}, {target_pos[1]*1000:.1f}, {target_pos[2]*1000:.1f}) mm, {inlet_w*1000:.0f}x{inlet_h*1000:.0f}mm")
        
        # Calculate required path
        delta_x = target_pos[0] - source_pos[0]
        delta_y = target_pos[1] - source_pos[1]
        print(f"  Required travel: dX={delta_x*1000:.1f}mm, dY={delta_y*1000:.1f}mm")
        
        # The path should be: up -> elbow -> horizontal -> elbow -> down -> elbow -> horizontal
        # Verify if the internal ducts achieve this
        
    return mc


def test_classification_assembly():
    """Test the full classification assembly connections."""
    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly, ClassificationSystemParams
    )
    
    print_header("CLASSIFICATION ASSEMBLY CONNECTION TEST")
    
    # Create classification system with default parameters
    params = ClassificationSystemParams()
    assembly = ClassificationSystemAssembly(params)
    
    print_section("Component Positions")
    for name, pos in assembly._component_positions.items():
        print(f"  {name}: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
    
    print_section("Multi-Cyclone Port World Positions")
    mc_pos = assembly._component_positions.get('multi_cyclone', np.zeros(3))
    mc_ports = assembly.multi_cyclone.ports
    
    for port_name, port in mc_ports.items():
        world_pos = mc_pos + np.array(port.position)
        print(f"  {port_name}:")
        print(f"    Local:  ({port.position[0]*1000:.1f}, {port.position[1]*1000:.1f}, {port.position[2]*1000:.1f}) mm")
        print(f"    World:  ({world_pos[0]*1000:.1f}, {world_pos[1]*1000:.1f}, {world_pos[2]*1000:.1f}) mm")
        print(f"    Dir:    ({port.direction[0]:.2f}, {port.direction[1]:.2f}, {port.direction[2]:.2f})")
        if port.diameter:
            print(f"    Size:   D={port.diameter*1000:.0f} mm")
        elif port.width and port.height:
            print(f"    Size:   {port.width*1000:.0f} x {port.height*1000:.0f} mm")
    
    print_section("Ductwork Sections")
    for i, (duct, duct_pos) in enumerate(assembly._duct_sections):
        duct_type = type(duct).__name__
        print(f"\n  [{i+1}] {duct_type}")
        print(f"      Position: ({duct_pos[0]*1000:.1f}, {duct_pos[1]*1000:.1f}, {duct_pos[2]*1000:.1f}) mm")
        
        if hasattr(duct, 'params'):
            p = duct.params
            if hasattr(p, 'diameter'):
                print(f"      Diameter: {p.diameter*1000:.0f} mm")
            if hasattr(p, 'length'):
                print(f"      Length: {p.length*1000:.0f} mm")
            if hasattr(p, 'direction'):
                print(f"      Direction: {p.direction}")
            if hasattr(p, 'inlet_dimensions') and hasattr(p, 'outlet_dimensions'):
                print(f"      Inlet dims: {p.inlet_dimensions}")
                print(f"      Outlet dims: {p.outlet_dimensions}")
        
        # Calculate end position for ducts
        if hasattr(duct, 'params') and hasattr(duct.params, 'length') and hasattr(duct.params, 'direction'):
            p = duct.params
            end_pos = (
                duct_pos[0] + p.direction[0] * p.length,
                duct_pos[1] + p.direction[1] * p.length,
                duct_pos[2] + p.direction[2] * p.length
            )
            print(f"      End pos: ({end_pos[0]*1000:.1f}, {end_pos[1]*1000:.1f}, {end_pos[2]*1000:.1f}) mm")
        
        if hasattr(duct, 'get_outlet_position'):
            outlet_local = duct.get_outlet_position()
            outlet_world = (
                duct_pos[0] + outlet_local[0],
                duct_pos[1] + outlet_local[1],
                duct_pos[2] + outlet_local[2]
            )
            print(f"      Outlet: ({outlet_world[0]*1000:.1f}, {outlet_world[1]*1000:.1f}, {outlet_world[2]*1000:.1f}) mm")
    
    # Verify zigzag -> cyclone connection
    print_section("Zigzag -> Multi-Cyclone Connection Verification")
    
    zigzag_pos = assembly._component_positions.get('zigzag', np.zeros(3))
    zigzag_fines = assembly.zigzag.ports['fines_outlet']
    zigzag_fines_world = zigzag_pos + np.array(zigzag_fines.position)
    
    cyclone_inlet = assembly.multi_cyclone.ports['inlet']
    cyclone_inlet_world = mc_pos + np.array(cyclone_inlet.position)
    
    print(f"  Zigzag fines outlet:")
    print(f"    Position: ({zigzag_fines_world[0]*1000:.1f}, {zigzag_fines_world[1]*1000:.1f}, {zigzag_fines_world[2]*1000:.1f}) mm")
    print(f"    Direction: {zigzag_fines.direction}")
    print(f"    Size: {zigzag_fines.width*1000:.0f} x {zigzag_fines.height*1000:.0f} mm")
    
    print(f"\n  Multi-cyclone inlet:")
    print(f"    Position: ({cyclone_inlet_world[0]*1000:.1f}, {cyclone_inlet_world[1]*1000:.1f}, {cyclone_inlet_world[2]*1000:.1f}) mm")
    print(f"    Direction: {cyclone_inlet.direction}")
    print(f"    Size: {cyclone_inlet.width*1000:.0f} x {cyclone_inlet.height*1000:.0f} mm")
    
    # Path analysis
    delta = cyclone_inlet_world - zigzag_fines_world
    print(f"\n  Path delta: dX={delta[0]*1000:.1f}mm, dY={delta[1]*1000:.1f}mm, dZ={delta[2]*1000:.1f}mm")
    
    return assembly


def trace_internal_duct_path():
    """Trace the exact path of internal ducts in multi-cyclone."""
    from airclassifier.geometry.components.multi_cyclone import (
        MultiCycloneSystem, MultiCycloneParams, CycloneStageParams
    )
    
    print_header("INTERNAL DUCT PATH TRACE")
    
    # Create test multi-cyclone
    stages = [
        CycloneStageParams(name="primary", diameter=0.3, design_d50=40e-6),
        CycloneStageParams(name="secondary", diameter=0.2, design_d50=20e-6),
        CycloneStageParams(name="tertiary", diameter=0.12, design_d50=10e-6),
    ]
    
    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=0.09,
        center=(0.0, 0.0, 0.0),
        resolution=48,
    )
    
    mc = MultiCycloneSystem(params)
    positions = mc._calculate_positions()
    
    # Group ducts by connection (7 ducts per connection for 3-elbow path)
    ducts_per_connection = 7  # elbow1, horiz_duct, elbow2, vert_duct, elbow3, transition, rect_inlet
    
    print("\nExpected internal duct path per connection:")
    print("  1. Elbow: VF (up) -> horizontal (+X)")
    print("  2. Horizontal duct (+X)")
    print("  3. Elbow: horizontal -> down (-Y)")
    print("  4. Vertical duct (-Y)")
    print("  5. Elbow: down -> horizontal (+X)")
    print("  6. Round-to-rect transition")
    print("  7. Rectangular inlet duct")
    
    for conn_idx in range(len(stages) - 1):
        print_section(f"Connection {conn_idx + 1}: {stages[conn_idx].name} -> {stages[conn_idx + 1].name}")
        
        start_idx = conn_idx * ducts_per_connection
        end_idx = start_idx + ducts_per_connection
        
        if end_idx > len(mc._connecting_ducts):
            print(f"  WARNING: Expected {ducts_per_connection} ducts, only have {len(mc._connecting_ducts) - start_idx}")
            end_idx = len(mc._connecting_ducts)
        
        current_pos = None
        
        for i in range(start_idx, end_idx):
            if i >= len(mc._connecting_ducts):
                break
                
            duct, duct_pos = mc._connecting_ducts[i]
            duct_type = type(duct).__name__
            local_idx = i - start_idx + 1
            
            print(f"\n  [{local_idx}] {duct_type}")
            print(f"      Start: ({duct_pos[0]*1000:.1f}, {duct_pos[1]*1000:.1f}, {duct_pos[2]*1000:.1f}) mm")
            
            # Check continuity from previous component
            if current_pos is not None:
                gap = np.linalg.norm(np.array(duct_pos) - np.array(current_pos))
                status = "OK" if gap < 0.01 else f"GAP: {gap*1000:.1f}mm"
                print(f"      Continuity: {status}")
            
            # Calculate outlet
            if hasattr(duct, 'get_outlet_position'):
                outlet_local = duct.get_outlet_position()
                outlet_world = (
                    duct_pos[0] + outlet_local[0],
                    duct_pos[1] + outlet_local[1],
                    duct_pos[2] + outlet_local[2]
                )
                current_pos = outlet_world
                print(f"      End: ({outlet_world[0]*1000:.1f}, {outlet_world[1]*1000:.1f}, {outlet_world[2]*1000:.1f}) mm")
                
                if hasattr(duct, 'get_outlet_direction'):
                    out_dir = duct.get_outlet_direction()
                    print(f"      Out dir: ({out_dir[0]:.2f}, {out_dir[1]:.2f}, {out_dir[2]:.2f})")
            elif hasattr(duct, 'params'):
                p = duct.params
                if hasattr(p, 'length') and hasattr(p, 'direction'):
                    end_pos = (
                        duct_pos[0] + p.direction[0] * p.length,
                        duct_pos[1] + p.direction[1] * p.length,
                        duct_pos[2] + p.direction[2] * p.length
                    )
                    current_pos = end_pos
                    print(f"      End: ({end_pos[0]*1000:.1f}, {end_pos[1]*1000:.1f}, {end_pos[2]*1000:.1f}) mm")
                    print(f"      Length: {p.length*1000:.0f} mm")
        
        # Check if final position reaches the target cyclone inlet
        if current_pos is not None:
            next_stage = stages[conn_idx + 1]
            next_pos = positions[conn_idx + 1]
            D_next = next_stage.diameter
            inlet_h = D_next * next_stage.inlet_height_ratio
            
            target_x = next_pos[0] - D_next / 2
            target_y = next_pos[1] - inlet_h / 2
            target_z = next_pos[2]
            
            final_gap = np.sqrt(
                (current_pos[0] - target_x)**2 +
                (current_pos[1] - target_y)**2 +
                (current_pos[2] - target_z)**2
            )
            
            print(f"\n  Target inlet surface: ({target_x*1000:.1f}, {target_y*1000:.1f}, {target_z*1000:.1f}) mm")
            print(f"  Final position: ({current_pos[0]*1000:.1f}, {current_pos[1]*1000:.1f}, {current_pos[2]*1000:.1f}) mm")
            print(f"  Gap to target: {final_gap*1000:.1f} mm")
            if final_gap > 0.01:
                print(f"  *** ALIGNMENT ISSUE: Gap > 10mm ***")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" MULTI-CYCLONE ASSEMBLY VERIFICATION TEST")
    print("=" * 70)
    
    try:
        # Test 1: Multi-cyclone internal connections
        mc = test_multi_cyclone_internal()
        
        # Test 2: Trace internal duct paths
        trace_internal_duct_path()
        
        # Test 3: Full classification assembly
        assembly = test_classification_assembly()
        
        print("\n" + "=" * 70)
        print(" TEST COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
