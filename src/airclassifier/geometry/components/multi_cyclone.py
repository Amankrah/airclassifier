"""
Multi-cyclone system for staged particle collection.

In protein separation from legumes, multiple cyclones are used:
- Primary cyclone: Collects coarse fraction (starch-rich)
- Secondary cyclone: Collects medium fraction
- Tertiary cyclone: Collects fine fraction (protein-rich)

Different cyclone sizes provide different cut points (d50).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np
import warp as wp

# Use the properly assembled CycloneAssembly from the assembly module
from ..assembly.cyclone import CycloneAssembly, CycloneGeometryParams
from ...utils.constants import PI
from ..connection_ports import ConnectionPort, PortType


@dataclass
class CycloneStageParams:
    """Parameters for a single cyclone stage."""

    name: str                    # Stage name (e.g., "primary", "secondary")
    diameter: float              # [m] Cyclone diameter
    design_d50: float           # [m] Target d50 cut size
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Position offset

    # Optional geometry overrides (if None, use standard proportions)
    cylinder_height_ratio: float = 1.5     # Height/D ratio
    cone_height_ratio: float = 2.5         # Cone height/D ratio
    inlet_width_ratio: float = 0.25        # Inlet width/D
    inlet_height_ratio: float = 0.5        # Inlet height/D
    vortex_finder_ratio: float = 0.5       # VF diameter/D
    dust_outlet_ratio: float = 0.375       # Dust outlet/D


@dataclass
class MultiCycloneParams:
    """Parameters for multi-cyclone system."""

    # Cyclone stages
    stages: List[CycloneStageParams] = field(default_factory=list)

    # Connection configuration
    arrangement: str = "series"  # "series" or "parallel"

    # Spacing between cyclones
    spacing: float = 0.5         # [m] Minimum spacing between cyclone centers

    # Base position
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Mesh resolution
    resolution: int = 48

    def __post_init__(self):
        if not self.stages:
            raise ValueError("At least one cyclone stage must be defined")

    @property
    def num_stages(self) -> int:
        """Number of cyclone stages."""
        return len(self.stages)


class MultiCycloneSystem:
    """
    Multi-cyclone collection system with multiple stages.

    Uses CycloneAssembly from the assembly module for properly fitted cyclones
    with correctly connected cylinder+cone body, inlet, vortex finder, and dust outlet.

    Typical configuration for protein separation:
    - Primary (large): d50 ~ 30-50 um -> Starch fraction
    - Secondary (medium): d50 ~ 15-25 um -> Mixed fraction
    - Tertiary (small): d50 ~ 5-15 um -> Protein fraction

    Coordinate system:
    - Origin at system center
    - Cyclones arranged along X-axis (series) or in grid (parallel)
    - Y-axis vertical (cyclone axis)
    
    Series Flow Path:
    - Feed → Primary cyclone inlet (tangential)
    - Primary vortex finder → elbow → duct → Secondary cyclone inlet
    - Secondary vortex finder → elbow → duct → Tertiary cyclone inlet
    - Tertiary vortex finder → System overflow outlet
    """

    def __init__(self, params: MultiCycloneParams):
        """
        Initialize multi-cyclone system.

        Args:
            params: MultiCycloneParams defining the system
        """
        self.params = params
        self._cyclones: Dict[str, CycloneAssembly] = {}
        self._cyclone_positions: Dict[str, Tuple[float, float, float]] = {}
        self._connecting_ducts: List[Any] = []  # Internal connecting ductwork
        self._vertices = None
        self._indices = None
        self._normals = None

        # Create individual cyclones
        self._create_cyclones()
        
        # Create connecting ductwork for series arrangement
        if params.arrangement == "series" and len(params.stages) > 1:
            self._create_series_connections()

    def _create_cyclones(self):
        """Create individual cyclone assemblies using CycloneAssembly."""
        p = self.params

        # Calculate positions based on arrangement
        positions = self._calculate_positions()

        for i, stage in enumerate(p.stages):
            # Create cyclone geometry params using CycloneGeometryParams
            D = stage.diameter
            
            # Use the CycloneGeometryParams which properly constructs the cyclone
            geo_params = CycloneGeometryParams(
                cylinder_diameter=D,
                cylinder_height=D * stage.cylinder_height_ratio,
                cone_height=D * stage.cone_height_ratio,
                cone_tip_diameter=D * stage.dust_outlet_ratio,
                inlet_width=D * stage.inlet_width_ratio,
                inlet_height=D * stage.inlet_height_ratio,
                inlet_length=D * 0.3,  # Standard inlet length
                vortex_finder_diameter=D * stage.vortex_finder_ratio,
                vortex_finder_length=D * stage.vortex_finder_ratio,
                dust_outlet_diameter=D * stage.dust_outlet_ratio,
                dust_outlet_length=D * 0.2,
                center=positions[i],
                resolution=p.resolution
            )

            # Create properly assembled cyclone
            cyclone = CycloneAssembly(geo_params, device="cpu")
            self._cyclones[stage.name] = cyclone
            self._cyclone_positions[stage.name] = positions[i]

    def _calculate_positions(self) -> List[Tuple[float, float, float]]:
        """Calculate positions for each cyclone stage."""
        p = self.params
        positions = []

        if p.arrangement == "series":
            # For series arrangement, we need enough spacing for internal ductwork
            # The duct path requires: elbow radius + horizontal run + elbow radius
            # Minimum spacing = max(user_spacing, 2*D_current + duct_clearance)
            
            # Calculate spacing for each pair based on duct geometry needs
            spacings = []
            for i, stage in enumerate(p.stages[:-1]):
                next_stage = p.stages[i + 1]
                # Duct diameter is vortex finder of current stage
                duct_d = stage.diameter * stage.vortex_finder_ratio
                elbow_r = duct_d * 1.0  # R/D = 1.0
                # Minimum spacing: 3*elbow_radius + some horizontal run + clearance
                # The path adds ~3*R in X direction from elbows alone
                min_spacing = 3 * elbow_r + 0.1  # 100mm extra clearance
                spacings.append(max(p.spacing, min_spacing))
            
            # Arrange along X-axis with calculated spacing
            total_width = sum(s.diameter for s in p.stages)
            for s in spacings:
                total_width += s
            
            x_start = p.center[0] - total_width / 2
            x_current = x_start
            
            for i, stage in enumerate(p.stages):
                x_pos = x_current + stage.diameter / 2
                positions.append((x_pos, p.center[1], p.center[2]))
                x_current += stage.diameter
                if i < len(spacings):
                    x_current += spacings[i]

        elif p.arrangement == "parallel":
            # Arrange in a row along Z-axis
            total_depth = sum(s.diameter for s in p.stages) + p.spacing * (len(p.stages) - 1)
            z_start = p.center[2] - total_depth / 2

            z_current = z_start
            for stage in p.stages:
                z_pos = z_current + stage.diameter / 2
                positions.append((p.center[0], p.center[1], z_pos))
                z_current += stage.diameter + p.spacing

        else:
            # Custom positions from stage params
            for stage in p.stages:
                pos = (
                    p.center[0] + stage.position[0],
                    p.center[1] + stage.position[1],
                    p.center[2] + stage.position[2]
                )
                positions.append(pos)

        return positions

    def _create_series_connections(self):
        """
        Create connecting ductwork between cyclones in series arrangement.
        
        For protein separation, flow goes:
        - Primary VF (overflow/fines) -> Secondary tangential inlet
        - Secondary VF -> Tertiary tangential inlet
        
        CycloneAssembly creates tangential inlets on the +X side (angular_position=0).
        The inlet outer end is at: center_x + radius + inlet_length
        Flow direction into inlet is -X (toward cyclone center).
        
        Since cyclones are arranged along +X axis, and inlets are on +X side,
        we need ductwork that goes from VF (at current_pos) to PAST the next
        cyclone's center to reach the inlet on its far side.
        
        Path: VF(up) -> elbow(+X) -> horiz -> down to inlet level -> into inlet (-X)
        """
        from .ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams, RectangularDuct, RectangularDuctParams
        from .transitions import Transition, TransitionParams
        
        p = self.params
        positions = self._calculate_positions()
        gap = 0.005  # 5mm
        
        for i in range(len(p.stages) - 1):
            current_stage = p.stages[i]
            next_stage = p.stages[i + 1]
            current_pos = positions[i]
            next_pos = positions[i + 1]
            
            # Get current cyclone's vortex finder parameters
            current_cyclone = self._cyclones[current_stage.name]
            D_curr = current_stage.diameter
            vf_d = current_cyclone.params.vortex_finder_diameter
            vf_top_y = current_pos[1] + 0.05  # protrusion_above from CycloneAssembly
            
            # Get next cyclone's inlet parameters from CycloneAssembly
            next_cyclone = self._cyclones[next_stage.name]
            D_next = next_stage.diameter
            inlet_w = next_cyclone.params.inlet_width
            inlet_h = next_cyclone.params.inlet_height
            inlet_len = next_cyclone.params.inlet_length
            
            # CycloneAssembly creates inlet on +X side (angular_position=0):
            # - Surface point X = center_x + radius
            # - Inlet direction = -X (toward center)
            # - Outer end X = surface_x + inlet_length = center_x + radius + inlet_length
            # The inlet Y center is at: cyclone_center_y - inlet_top_offset - height/2
            # With inlet_top_offset=0.05: inlet_y = center_y - 0.05 - height/2
            inlet_outer_x = next_pos[0] + D_next / 2 + inlet_len
            inlet_y = next_pos[1] - 0.05 - inlet_h / 2
            
            # Duct diameter - use vortex finder diameter
            duct_d = vf_d
            R = duct_d * 1.0  # bend radius R/D = 1.0
            
            # Since inlet is on +X side and flow is -X, we approach from +X side
            # Path: VF(up) -> elbow(+X) -> horiz(+X) -> elbow(down) -> vert -> elbow(-X) -> transition -> rect
            
            # Final components entering cyclone in -X direction
            rect_len = inlet_w
            trans_len = 0.04
            
            # Calculate path:
            # Start at VF: X = current_pos[0]
            # Elbow1 (up->+X): outlet X = current_pos[0] + R
            # Horiz duct (+X): outlet X = current_pos[0] + R + L
            # Elbow2 (+X->down): outlet X = current_pos[0] + R + L + R
            # Vert duct: no X change
            # Elbow3 (down->-X): outlet X = current_pos[0] + R + L + R - R = current_pos[0] + R + L
            # Wait, this doesn't work - elbow down to -X adds no X, but outlet is displaced
            
            # Let me reconsider: approach inlet from the +X side (beyond inlet_outer_x)
            # Final path enters going -X direction
            # 
            # Working backwards from inlet:
            # - Rect duct ends at inlet_outer_x, travels -X, starts at inlet_outer_x + rect_len
            # - Transition ends where rect starts, starts at inlet_outer_x + rect_len + trans_len
            # - Elbow3 (down->-X): inlet at higher X, outlet direction is -X
            #   For elbow with inlet -Y and turning to -X:
            #   outlet position relative to inlet: X decreases by R, Y decreases by R
            #   So elbow3 inlet at X = inlet_outer_x + rect_len + trans_len + R
            # - Vert duct starts higher
            # - Elbow2 (+X->down): outlet direction -Y
            #   outlet at elbow3 inlet X, inlet at X - R
            # - Horiz duct ends at elbow2 inlet
            # - Elbow1 ends at horiz duct start
            
            # Target X for end of elbow3 (where transition starts):
            target_x_after_elbow3 = inlet_outer_x + rect_len + trans_len + 2*gap
            
            # Elbow3: down (-Y) -> left (-X)
            # Outlet direction -X means: if inlet direction is -Y and we turn left,
            # rotation axis should be -Z (looking from +Z, inlet -Y turns CCW to -X)
            # Outlet position: X - R, Y - R relative to inlet
            elbow3_inlet_x = target_x_after_elbow3 + R
            
            # Vert duct ends at elbow3 inlet
            # Calculate required vertical travel
            horiz_y = vf_top_y + R + gap  # Y level of horizontal duct (after first elbow)
            
            # Y calculations:
            # After elbow1: Y = vf_top_y + gap + R
            # Horiz duct: Y unchanged
            # After elbow2: Y = horiz_y - R
            # Vert duct: Y decreases by vert_len, ends at horiz_y - R - vert_len - gap
            # After elbow3: Y -= R
            # Transition + rect: Y unchanged
            # Final Y should be inlet_y
            # inlet_y = horiz_y - R - gap - vert_len - gap - R
            # vert_len = horiz_y - 2*R - 2*gap - inlet_y
            
            vert_len = horiz_y - 2*R - 2*gap - inlet_y
            if vert_len < 0.02:
                vert_len = 0.02
            
            # Elbow2: +X -> down (-Y)
            # Inlet at X = elbow3_inlet_x - vert_len change? No, vert duct doesn't change X
            # Elbow2 inlet X: elbow3_inlet_x
            # Elbow2 (+X -> -Y): outlet X += R, outlet Y -= R relative to inlet
            # So elbow2 inlet X = elbow3_inlet_x - R... wait, vert duct doesn't change X
            # Vert duct starts at elbow2 outlet X which equals elbow3 inlet X
            # So elbow2 outlet X = elbow3_inlet_x
            # For elbow from +X to -Y: outlet X = inlet X + R
            # So elbow2 inlet X = elbow3_inlet_x - R
            elbow2_inlet_x = elbow3_inlet_x - R
            
            # Horiz duct ends at elbow2 inlet, starts at elbow1 outlet
            # Elbow1: up (+Y) -> +X
            # outlet X = inlet X + R
            elbow1_outlet_x = current_pos[0] + R
            
            L = elbow2_inlet_x - elbow1_outlet_x - 2*gap
            if L < 0.03:
                L = 0.03
            
            # ============================================================
            # 1. Elbow1: UP (+Y) -> horizontal (+X)
            # ============================================================
            e1_pos = (current_pos[0], vf_top_y + gap, current_pos[2])
            elbow1 = DuctElbow(DuctElbowParams(
                diameter=duct_d,
                bend_radius=R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(0.0, 1.0, 0.0),
                rotation_axis=(0.0, 0.0, 1.0),
            ))
            self._connecting_ducts.append((elbow1, e1_pos))
            e1_out_x = e1_pos[0] + R
            e1_out_y = e1_pos[1] + R
            
            # ============================================================
            # 2. Horizontal duct (+X)
            # ============================================================
            h_pos = (e1_out_x + gap, e1_out_y, current_pos[2])
            horiz = RoundDuct(RoundDuctParams(
                diameter=duct_d,
                length=L,
                wall_thickness=0.002,
                direction=(1.0, 0.0, 0.0),
                center=(0, 0, 0),
                flanged=True
            ))
            self._connecting_ducts.append((horiz, h_pos))
            h_end_x = h_pos[0] + L
            
            # ============================================================
            # 3. Elbow2: horizontal (+X) -> down (-Y)
            # ============================================================
            e2_pos = (h_end_x + gap, h_pos[1], current_pos[2])
            elbow2 = DuctElbow(DuctElbowParams(
                diameter=duct_d,
                bend_radius=R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(1.0, 0.0, 0.0),
                rotation_axis=(0.0, 0.0, 1.0),
            ))
            self._connecting_ducts.append((elbow2, e2_pos))
            e2_out_x = e2_pos[0] + R
            e2_out_y = e2_pos[1] - R
            
            # ============================================================
            # 4. Vertical duct down (-Y)
            # ============================================================
            v_pos = (e2_out_x, e2_out_y - gap, current_pos[2])
            vert = RoundDuct(RoundDuctParams(
                diameter=duct_d,
                length=vert_len,
                wall_thickness=0.002,
                direction=(0.0, -1.0, 0.0),
                center=(0, 0, 0),
                flanged=True
            ))
            self._connecting_ducts.append((vert, v_pos))
            v_end_y = v_pos[1] - vert_len
            
            # ============================================================
            # 5. Elbow3: down (-Y) -> horizontal (-X) into inlet
            # ============================================================
            e3_pos = (v_pos[0], v_end_y - gap, current_pos[2])
            elbow3 = DuctElbow(DuctElbowParams(
                diameter=duct_d,
                bend_radius=R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(0.0, -1.0, 0.0),
                rotation_axis=(0.0, 0.0, -1.0),  # Turn to -X
            ))
            self._connecting_ducts.append((elbow3, e3_pos))
            # For -Y inlet, -Z rotation axis: turns to -X
            # Outlet: X -= R, Y -= R
            e3_out_x = e3_pos[0] - R
            e3_out_y = e3_pos[1] - R
            
            # ============================================================
            # 6. Transition: round to rect (-X direction into cyclone)
            # ============================================================
            t_pos = (e3_out_x - gap, e3_out_y, current_pos[2])
            transition = Transition(TransitionParams(
                transition_type="round_to_rect",
                inlet_dimensions=(duct_d,),
                outlet_dimensions=(inlet_w, inlet_h),
                length=trans_len,
                concentric=True,
                wall_thickness=0.002,
                direction=(-1.0, 0.0, 0.0),  # -X into cyclone
                center=(0, 0, 0)
            ))
            self._connecting_ducts.append((transition, t_pos))
            t_end_x = t_pos[0] - trans_len
            
            # ============================================================
            # 7. Rectangular inlet duct (-X into cyclone)
            # ============================================================
            r_pos = (t_end_x - gap, t_pos[1], current_pos[2])
            rect_inlet = RectangularDuct(RectangularDuctParams(
                width=inlet_w,
                height=inlet_h,
                length=rect_len,
                wall_thickness=0.002,
                direction=(-1.0, 0.0, 0.0),  # -X into cyclone
                center=(0, 0, 0),
                flanged=False
            ))
            self._connecting_ducts.append((rect_inlet, r_pos))

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate combined mesh for all cyclones and connecting ductwork.

        Returns:
            Tuple of (vertices, indices, normals)
        """
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        # Add cyclone meshes (CycloneAssembly.build_mesh returns vertices, indices)
        for name, cyclone in self._cyclones.items():
            verts, idx = cyclone.build_mesh()

            all_vertices.append(verts)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)
        
        # Add connecting ductwork meshes (for series arrangement)
        for duct_component, position in self._connecting_ducts:
            verts, idx, _ = duct_component.generate_mesh()
            
            # Apply position offset
            offset = np.array(position)
            verts_offset = verts + offset
            
            all_vertices.append(verts_offset)
            all_indices.append(idx + vertex_offset)
            vertex_offset += len(verts)

        self._vertices = np.vstack(all_vertices).astype(np.float32)
        self._indices = np.concatenate(all_indices).astype(np.int32)

        # Generate simple normals (pointing outward radially)
        self._normals = np.zeros_like(self._vertices)
        for i, v in enumerate(self._vertices):
            r = np.sqrt(v[0] ** 2 + v[2] ** 2)
            if r > 1e-6:
                self._normals[i] = [v[0] / r, 0, v[2] / r]
            else:
                self._normals[i] = [0, 1, 0]

        return self._vertices, self._indices, self._normals

    def get_cyclone(self, name: str) -> CycloneAssembly:
        """
        Get a specific cyclone by name.

        Args:
            name: Cyclone stage name

        Returns:
            CycloneAssembly for the specified stage
        """
        if name not in self._cyclones:
            raise KeyError(f"Cyclone '{name}' not found. Available: {list(self._cyclones.keys())}")
        return self._cyclones[name]

    def get_all_cyclones(self) -> Dict[str, CycloneAssembly]:
        """Get dictionary of all cyclones."""
        return self._cyclones

    def get_system_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of entire system.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        all_min = []
        all_max = []

        for cyclone in self._cyclones.values():
            min_c, max_c = cyclone.get_bounds()
            all_min.append(min_c)
            all_max.append(max_c)

        return np.min(all_min, axis=0), np.max(all_max, axis=0)

    def get_stage_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each stage.

        Returns:
            List of dictionaries with stage information
        """
        info = []
        for stage in self.params.stages:
            cyclone = self._cyclones[stage.name]
            # CycloneAssembly uses CycloneGeometryParams which has same properties
            info.append({
                'name': stage.name,
                'diameter': stage.diameter * 1000,  # mm
                'design_d50': stage.design_d50 * 1e6,  # μm
                'total_height': cyclone.params.total_height * 1000,  # mm
                'inlet_area': (cyclone.params.inlet_width *
                              cyclone.params.inlet_height) * 1e6,  # mm²
            })
        return info

    def print_summary(self):
        """Print summary of multi-cyclone system."""
        print("=" * 60)
        print("Multi-Cyclone System Summary")
        print("=" * 60)
        print(f"Number of stages: {self.params.num_stages}")
        print(f"Arrangement: {self.params.arrangement}")
        print("-" * 60)

        for stage_info in self.get_stage_info():
            print(f"\n{stage_info['name'].upper()} CYCLONE:")
            print(f"  Diameter:      {stage_info['diameter']:.1f} mm")
            print(f"  Design d50:    {stage_info['design_d50']:.1f} um")
            print(f"  Total height:  {stage_info['total_height']:.1f} mm")
            print(f"  Inlet area:    {stage_info['inlet_area']:.1f} mm²")

        min_b, max_b = self.get_system_bounds()
        extent = max_b - min_b
        print("-" * 60)
        print(f"System extent: {extent[0]*1000:.0f} x {extent[1]*1000:.0f} x {extent[2]*1000:.0f} mm")
        print("=" * 60)

    def to_warp_mesh(self, device: str = "cuda") -> wp.Mesh:
        """Create a Warp mesh from the multi-cyclone geometry."""
        if self._vertices is None:
            self.generate_mesh()

        points = wp.array(self._vertices, dtype=wp.vec3, device=device)
        indices = wp.array(self._indices, dtype=wp.int32, device=device)

        return wp.Mesh(points=points, indices=indices)

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
    def ports(self) -> Dict[str, ConnectionPort]:
        """
        Get connection ports for the multi-cyclone system.
        
        Ports:
        - inlet: Tangential inlet to first (primary) cyclone (rectangular)
        - overflow: Clean air outlet from last cyclone (vortex finder top)
        - dust_outlet_<name>: Dust collection from each cyclone stage
        
        Coordinate System:
        - Cyclone center is at the TOP of the cylindrical section
        - Y-axis is vertical (cyclone axis)
        - Tangential inlet is positioned at the top of the cylinder
        - Inlet duct extends radially outward from the cyclone surface
        """
        p = self.params
        ports = {}
        
        # Get first and last cyclone assemblies
        first_stage = p.stages[0]
        first_cyclone = self._cyclones[first_stage.name]
        first_pos = self._cyclone_positions[first_stage.name]
        
        # Get inlet information from the first cyclone's TangentialInlet component
        # CycloneAssembly creates the inlet with angular_position=0 (+X side)
        inlet_params = first_cyclone.params
        D = inlet_params.cylinder_diameter
        r = D / 2
        inlet_width = inlet_params.inlet_width
        inlet_height = inlet_params.inlet_height
        inlet_length = inlet_params.inlet_length
        
        # TangentialInlet geometry (from inlet.py):
        # - angular_position = 0 means inlet on +X side
        # - surface_point.x = center_x + r*cos(0) = center_x + r
        # - surface_point.y = center_y - inlet_top_offset - height/2
        #                   = center_y - 0.05 - height/2 (inlet_top_offset=0.05 in CycloneAssembly)
        # - inlet_direction = -radial = (-1, 0, 0) pointing inward (-X toward cyclone center)
        # - inlet_start (outer end) = surface_point - inlet_direction * length
        #                           = (center_x + r + length, center_y - 0.05 - height/2, center_z)
        
        # Inlet port at the OUTER END of the inlet duct
        inlet_x = first_pos[0] + r + inlet_length
        inlet_y = first_pos[1] - 0.05 - inlet_height / 2  # Centered on inlet opening
        inlet_z = first_pos[2]

        # Port direction faces +X (the opening faces toward +X, receives flow from -X direction)
        # Flow enters traveling -X (toward cyclone center)
        ports['inlet'] = ConnectionPort(
            position=(inlet_x, inlet_y, inlet_z),
            direction=(1.0, 0.0, 0.0),  # Port opening faces +X
            width=inlet_width,
            height=inlet_height,
            port_type=PortType.RECTANGULAR,
            name="inlet"
        )
        
        # Last cyclone overflow (clean air from vortex finder)
        last_stage = p.stages[-1]
        last_cyclone = self._cyclones[last_stage.name]
        last_pos = self._cyclone_positions[last_stage.name]
        vf_diameter = last_cyclone.params.vortex_finder_diameter
        
        # Vortex finder top is above the cyclone
        overflow_y = last_pos[1] + 0.05  # protrusion_above from CycloneAssembly
        
        ports['overflow'] = ConnectionPort(
            position=(last_pos[0], overflow_y, last_pos[2]),
            direction=(0.0, 1.0, 0.0),  # Faces up
            diameter=vf_diameter,
            port_type=PortType.CIRCULAR,
            name="overflow"
        )
        
        # Dust outlets for each cyclone
        for stage in p.stages:
            cyclone = self._cyclones[stage.name]
            pos = self._cyclone_positions[stage.name]
            cyc_params = cyclone.params
            
            # Get dust outlet position from CycloneAssembly
            dust_outlet_d = cyc_params.dust_outlet_diameter
            dust_outlet_len = cyc_params.dust_outlet_length
            cone_height = cyc_params.cone_height
            cyl_height = cyc_params.cylinder_height
            
            # Dust outlet at bottom of cone + outlet length
            dust_y = pos[1] - cyl_height - cone_height - dust_outlet_len
            
            ports[f'dust_outlet_{stage.name}'] = ConnectionPort(
                position=(pos[0], dust_y, pos[2]),
                direction=(0.0, -1.0, 0.0),  # Faces down
                diameter=dust_outlet_d,
                port_type=PortType.CIRCULAR,
                name=f"dust_outlet_{stage.name}"
            )
        
        return ports


def create_protein_separation_cyclones(
    primary_diameter: float = 0.4,
    secondary_diameter: float = 0.25,
    tertiary_diameter: float = 0.15
) -> MultiCycloneSystem:
    """
    Create a multi-cyclone system optimized for legume protein separation.

    Args:
        primary_diameter: Primary cyclone diameter [m]
        secondary_diameter: Secondary cyclone diameter [m]
        tertiary_diameter: Tertiary cyclone diameter [m]

    Returns:
        MultiCycloneSystem configured for protein separation
    """
    stages = [
        CycloneStageParams(
            name="primary",
            diameter=primary_diameter,
            design_d50=40e-6,  # 40 μm - coarse starch
        ),
        CycloneStageParams(
            name="secondary",
            diameter=secondary_diameter,
            design_d50=20e-6,  # 20 μm - medium
        ),
        CycloneStageParams(
            name="tertiary",
            diameter=tertiary_diameter,
            design_d50=10e-6,  # 10 μm - fine protein
        ),
    ]

    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=0.3,
    )

    return MultiCycloneSystem(params)


def create_two_stage_cyclones(
    coarse_diameter: float = 0.3,
    fine_diameter: float = 0.2
) -> MultiCycloneSystem:
    """
    Create a two-stage cyclone system (coarse + fine).

    Args:
        coarse_diameter: Coarse cyclone diameter [m]
        fine_diameter: Fine cyclone diameter [m]

    Returns:
        MultiCycloneSystem with two stages
    """
    stages = [
        CycloneStageParams(
            name="coarse",
            diameter=coarse_diameter,
            design_d50=30e-6,  # 30 μm
        ),
        CycloneStageParams(
            name="fine",
            diameter=fine_diameter,
            design_d50=15e-6,  # 15 μm
        ),
    ]

    params = MultiCycloneParams(
        stages=stages,
        arrangement="series",
        spacing=0.2,
    )

    return MultiCycloneSystem(params)
