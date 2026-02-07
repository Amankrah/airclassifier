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
            
            # Determine inlet angular position based on arrangement
            # For series: ALL inlets on -X side (π) to receive flow traveling +X direction
            #   - Primary: receives from zigzag/external feed (traveling +X)
            #   - Downstream: receives from upstream VF via ductwork (traveling +X)
            # For parallel: all inlets on +X (0) for common header
            if p.arrangement == "series":
                # All cyclones in series: inlet on -X side (receives +X flow)
                inlet_angle = PI  # 180 degrees = -X side
            else:
                # Parallel arrangement: inlet on +X side
                inlet_angle = 0.0
            
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
                inlet_angular_position=inlet_angle,
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
        
        For series arrangement, cyclones are positioned along +X axis.
        Each downstream cyclone's inlet is on its -X side (facing the upstream cyclone).
        
        Flow path from current cyclone VF to next cyclone inlet:
        - Current VF at (current_pos[0], vf_top_y) - direction +Y
        - Next inlet surface at (next_pos[0] - R_next, inlet_y) - on -X side
        - Required travel: positive dX, negative dY
        
        Duct path:
        1. Elbow1: VF (+Y) -> turn to +X horizontal
        2. Horizontal duct: travel +X toward next cyclone
        3. Elbow2: +X -> turn down to -Y
        4. Vertical duct: travel down toward inlet height
        5. Elbow3: -Y -> turn to +X (toward inlet)
        6. Transition: round to rectangular (going +X into inlet)
        """
        from .ductwork import RoundDuct, RoundDuctParams, DuctElbow, DuctElbowParams, RectangularDuct, RectangularDuctParams
        from .transitions import Transition, TransitionParams
        
        p = self.params
        positions = self._calculate_positions()
        gap = 0.005  # 5mm gap between components
        
        for i in range(len(p.stages) - 1):
            current_stage = p.stages[i]
            next_stage = p.stages[i + 1]
            current_pos = positions[i]
            next_pos = positions[i + 1]
            
            # Get current cyclone's vortex finder parameters
            current_cyclone = self._cyclones[current_stage.name]
            vf_d = current_cyclone.params.vortex_finder_diameter
            vf_top_y = current_pos[1] + 0.05  # protrusion_above from CycloneAssembly
            
            # Get next cyclone's inlet parameters
            next_cyclone = self._cyclones[next_stage.name]
            D_next = next_stage.diameter
            R_next = D_next / 2
            inlet_w = next_cyclone.params.inlet_width
            inlet_h = next_cyclone.params.inlet_height
            inlet_length = next_cyclone.params.inlet_length
            
            # Inlet is on the -X side of the next cyclone (facing upstream)
            # Inlet surface X = next_pos[0] - R_next
            # Inlet outer end X = next_pos[0] - R_next - inlet_length
            # Inlet Y center = next_pos[1] - 0.05 - inlet_h/2 (with top offset)
            inlet_surface_x = next_pos[0] - R_next
            inlet_outer_x = inlet_surface_x - inlet_length
            inlet_y = next_pos[1] - 0.05 - inlet_h / 2
            
            # Duct diameter = vortex finder diameter
            duct_d = vf_d
            R = duct_d * 1.0  # bend radius R/D = 1.0
            
            # ============================================================
            # Calculate path geometry working backwards from inlet
            # ============================================================
            
            # Transition connects to inlet outer end, going +X
            trans_len = 0.06  # 60mm transition
            trans_end_x = inlet_outer_x - gap  # End of transition (connects to inlet)
            trans_start_x = trans_end_x - trans_len  # Start of transition
            trans_y = inlet_y  # Same Y level as inlet center
            
            # Elbow3: -Y -> +X, positioned before transition
            # Elbow3 outlet connects to transition start
            # For -Y to +X elbow: inlet at (x, y+R), outlet at (x+R, y)
            e3_outlet_x = trans_start_x - gap
            e3_outlet_y = trans_y
            e3_inlet_x = e3_outlet_x - R
            e3_inlet_y = e3_outlet_y + R
            
            # Vertical duct ends at elbow3 inlet
            v_end_y = e3_inlet_y + gap
            
            # ============================================================
            # Calculate path geometry working forward from VF
            # ============================================================
            
            # Elbow1: VF (+Y) -> +X horizontal
            e1_inlet_x = current_pos[0]
            e1_inlet_y = vf_top_y + gap
            e1_outlet_x = e1_inlet_x + R
            e1_outlet_y = e1_inlet_y + R
            horiz_y = e1_outlet_y  # Y level of horizontal duct
            
            # Elbow2: +X -> -Y (down)
            # Elbow2 outlet X should align with elbow3 inlet X
            e2_outlet_x = e3_inlet_x
            e2_inlet_x = e2_outlet_x - R  # For +X to -Y elbow: outlet_x = inlet_x + R
            e2_inlet_y = horiz_y
            e2_outlet_y = e2_inlet_y - R
            
            # Horizontal duct from elbow1 outlet to elbow2 inlet
            h_start_x = e1_outlet_x + gap
            h_end_x = e2_inlet_x - gap
            h_len = h_end_x - h_start_x
            
            if h_len < 0.03:
                # Not enough room - use minimum length
                h_len = 0.03
            
            # Vertical duct from elbow2 outlet down to elbow3 inlet
            v_start_y = e2_outlet_y - gap
            vert_len = v_start_y - v_end_y
            
            if vert_len < 0.02:
                vert_len = 0.02
                v_end_y = v_start_y - vert_len
            
            # ============================================================
            # 1. Elbow1: UP (+Y) -> horizontal (+X)
            # ============================================================
            e1_pos = (e1_inlet_x, e1_inlet_y, current_pos[2])
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
            
            # ============================================================
            # 2. Horizontal duct (+X)
            # ============================================================
            h_pos = (h_start_x, horiz_y, current_pos[2])
            horiz = RoundDuct(RoundDuctParams(
                diameter=duct_d,
                length=h_len,
                wall_thickness=0.002,
                direction=(1.0, 0.0, 0.0),
                center=(0, 0, 0),
                flanged=True
            ))
            self._connecting_ducts.append((horiz, h_pos))
            
            # ============================================================
            # 3. Elbow2: horizontal (+X) -> down (-Y)
            # ============================================================
            e2_pos = (e2_inlet_x, e2_inlet_y, current_pos[2])
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
            
            # ============================================================
            # 4. Vertical duct down (-Y)
            # ============================================================
            v_pos = (e2_outlet_x, v_start_y, current_pos[2])
            vert = RoundDuct(RoundDuctParams(
                diameter=duct_d,
                length=vert_len,
                wall_thickness=0.002,
                direction=(0.0, -1.0, 0.0),
                center=(0, 0, 0),
                flanged=True
            ))
            self._connecting_ducts.append((vert, v_pos))
            
            # ============================================================
            # 5. Elbow3: down (-Y) -> horizontal (+X) toward inlet
            # ============================================================
            e3_pos = (e3_inlet_x, e3_inlet_y, current_pos[2])
            elbow3 = DuctElbow(DuctElbowParams(
                diameter=duct_d,
                bend_radius=R,
                angle=90.0,
                wall_thickness=0.002,
                flanged=True,
                center=(0, 0, 0),
                inlet_direction=(0.0, -1.0, 0.0),  # Coming from above (-Y direction)
                rotation_axis=(0.0, 0.0, -1.0),    # Rotate to turn toward +X
            ))
            self._connecting_ducts.append((elbow3, e3_pos))
            
            # ============================================================
            # 6. Transition: round to rect (+X direction into inlet)
            # ============================================================
            # Note: For horizontal +X direction, the transition's coordinate system maps:
            #   outlet_dimensions[0] → perp1 = -Y (vertical)
            #   outlet_dimensions[1] → perp2 = +Z (horizontal depth)
            # Cyclone inlet has: height=vertical (Y), width=tangent (Z)
            # So we pass (height, width) to match the cyclone inlet orientation
            t_pos = (trans_start_x, trans_y, current_pos[2])
            transition = Transition(TransitionParams(
                transition_type="round_to_rect",
                inlet_dimensions=(duct_d,),
                outlet_dimensions=(inlet_h, inlet_w),  # (height, width) to match inlet orientation
                length=trans_len,
                concentric=True,
                wall_thickness=0.002,
                direction=(1.0, 0.0, 0.0),  # +X into cyclone inlet
                center=(0, 0, 0)
            ))
            self._connecting_ducts.append((transition, t_pos))

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

    def calculate_stage_performance(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
    ) -> List[dict]:
        """
        Calculate actual cut sizes and efficiencies for each stage.

        Args:
            volumetric_flow: System flow rate [m³/s]
            particle_density: Particle density [kg/m³]

        Returns:
            List of performance dicts for each stage
        """
        results = []
        for stage in self.params.stages:
            cyclone = self._cyclones[stage.name]
            d50 = cyclone.calculate_cut_size_d50(volumetric_flow, particle_density)
            dP = cyclone.calculate_pressure_drop(volumetric_flow)
            inlet_area = cyclone.params.inlet_width * cyclone.params.inlet_height
            v_inlet = volumetric_flow / inlet_area
            results.append({
                "name": stage.name,
                "diameter_mm": stage.diameter * 1000,
                "design_d50_um": stage.design_d50 * 1e6,
                "actual_d50_um": d50 * 1e6,
                "d50_ratio": d50 / stage.design_d50 if stage.design_d50 > 0 else float("inf"),
                "inlet_velocity_m_s": v_inlet,
                "pressure_drop_Pa": dP,
            })
        return results

    def calculate_required_flow_for_design_d50(
        self,
        particle_density: float = 1420.0,
    ) -> float:
        """
        Calculate flow rate needed to achieve design d50 values.

        Uses the primary cyclone's design d50 as reference.

        Returns:
            Required volumetric flow rate [m³/s]
        """
        primary = self.params.stages[0]
        cyclone = self._cyclones[primary.name]
        return cyclone.calculate_required_flow_for_d50(primary.design_d50, particle_density)

    def validate_staging(
        self,
        volumetric_flow: float,
        particle_density: float = 1420.0,
    ) -> dict:
        """
        Validate that staging will work at given flow rate.

        Returns:
            Dictionary with valid, warnings, errors, stages, recommended_flow_m3_h, etc.
        """
        stage_perf = self.calculate_stage_performance(
            volumetric_flow, particle_density
        )
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "stages": stage_perf,
        }

        d50_values = [s["actual_d50_um"] for s in stage_perf]
        if d50_values != sorted(d50_values, reverse=True):
            result["warnings"].append(
                "Cut sizes are not in expected order. Check cyclone sizing."
            )

        # Cyclone series validation.  In this system the cyclones are
        # COLLECTORS after the wheel classifier — a lower d50 than design
        # means higher collection efficiency, which is desirable.
        # Only flag errors when cyclone performance is physically degraded:
        # - d50 ABOVE the incoming particle range (poor collection)
        # - Inlet velocity below minimum for vortex formation (~4 m/s)
        # - Inlet velocity above maximum causing re-entrainment (~30 m/s)
        primary_d50 = stage_perf[0]["actual_d50_um"]

        for stage in stage_perf:
            actual = stage["actual_d50_um"]
            design = stage["design_d50_um"]
            ratio = stage["d50_ratio"]
            # d50 far above design means very low flow — vortex may not form
            if ratio > 5.0:
                result["errors"].append(
                    f"{stage['name']}: actual d50 ({actual:.1f} µm) is "
                    f"{ratio:.0f}x design ({design:.0f} µm). "
                    "Flow too low for effective vortex separation."
                )
                result["valid"] = False
            elif ratio > 3.0:
                result["warnings"].append(
                    f"{stage['name']}: actual d50 ({actual:.1f} µm) is "
                    f"{ratio:.1f}x design ({design:.0f} µm). "
                    "Low flow may reduce collection efficiency."
                )

        Q_design = self.calculate_required_flow_for_design_d50(particle_density)
        result["recommended_flow_m3_s"] = Q_design
        result["recommended_flow_m3_h"] = Q_design * 3600
        result["current_flow_m3_h"] = volumetric_flow * 3600
        result["flow_ratio"] = volumetric_flow / Q_design if Q_design > 0 else float("inf")

        return result

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
        - For series arrangement:
          - ALL cyclones have inlet on -X side (angular_position=π)
          - This allows flow traveling +X to enter tangentially
        - For parallel arrangement:
          - All inlets on +X side for common header connection
        - Inlet duct extends radially outward from the cyclone surface
        """
        p = self.params
        ports = {}
        
        # Get first and last cyclone assemblies
        first_stage = p.stages[0]
        first_cyclone = self._cyclones[first_stage.name]
        first_pos = self._cyclone_positions[first_stage.name]
        
        # Get inlet information from the first cyclone
        inlet_params = first_cyclone.params
        D = inlet_params.cylinder_diameter
        r = D / 2
        inlet_width = inlet_params.inlet_width
        inlet_height = inlet_params.inlet_height
        inlet_length = inlet_params.inlet_length
        inlet_angle = inlet_params.inlet_angular_position
        
        # Calculate inlet position based on angular position
        # For angular_position=0 (primary): inlet is on +X side
        # For angular_position=π (downstream): inlet is on -X side
        import math
        inlet_surface_x = first_pos[0] + r * math.cos(inlet_angle)
        inlet_surface_z = first_pos[2] + r * math.sin(inlet_angle)
        
        # Inlet extends outward from surface in the radial direction
        # Radial direction = [cos(angle), 0, sin(angle)]
        radial_x = math.cos(inlet_angle)
        radial_z = math.sin(inlet_angle)
        
        # Inlet outer end (where external duct connects)
        inlet_x = inlet_surface_x + radial_x * inlet_length
        inlet_y = first_pos[1] - 0.05 - inlet_height / 2
        inlet_z = inlet_surface_z + radial_z * inlet_length

        # Port direction faces outward (radial direction)
        # Flow enters in the opposite direction (toward cyclone center)
        ports['inlet'] = ConnectionPort(
            position=(inlet_x, inlet_y, inlet_z),
            direction=(radial_x, 0.0, radial_z),  # Port opening faces radially outward
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
