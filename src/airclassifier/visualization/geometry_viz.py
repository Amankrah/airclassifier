"""
Geometry visualization utilities for cyclone air classifier.

Provides functions for visualizing cyclone geometry meshes,
cross-sections, and assembly renderings.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

from ..utils.constants import PI


def plot_cyclone_mesh_3d(
    vertices: np.ndarray,
    indices: np.ndarray,
    ax=None,
    title: str = "Cyclone Air Classifier",
    show_wireframe: bool = True,
    alpha: float = 0.6,
    color: str = "steelblue",
    max_faces: int = 3000
):
    """
    Render cyclone mesh using matplotlib 3D.
    
    Args:
        vertices: Mesh vertices (N, 3) in meters
        indices: Triangle indices (flat array, multiple of 3)
        ax: Matplotlib 3D axes (creates new if None)
        title: Plot title
        show_wireframe: Show wireframe edges
        alpha: Surface transparency
        color: Face color
        max_faces: Maximum faces to render (for performance)
        
    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Reshape indices to triangles
    triangles = indices.reshape(-1, 3)
    
    # Convert to mm for display
    verts_mm = vertices * 1000
    
    # Create triangular mesh faces
    mesh_faces = []
    for tri in triangles:
        face = [verts_mm[tri[0]], verts_mm[tri[1]], verts_mm[tri[2]]]
        mesh_faces.append(face)
    
    # Subsample faces for performance
    if len(mesh_faces) > max_faces:
        step = len(mesh_faces) // max_faces
        mesh_faces_subset = mesh_faces[::step]
    else:
        mesh_faces_subset = mesh_faces
    
    # Create Poly3DCollection
    mesh = Poly3DCollection(mesh_faces_subset, alpha=alpha)
    mesh.set_facecolor(color)
    if show_wireframe:
        mesh.set_edgecolor('darkblue')
        mesh.set_linewidth(0.1)
    else:
        mesh.set_edgecolor(color)
    
    ax.add_collection3d(mesh)
    
    # Set axis limits with equal aspect
    x_min, x_max = verts_mm[:, 0].min(), verts_mm[:, 0].max()
    y_min, y_max = verts_mm[:, 1].min(), verts_mm[:, 1].max()
    z_min, z_max = verts_mm[:, 2].min(), verts_mm[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Labels
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_cyclone_cross_section(
    assembly,
    ax=None,
    title: str = "Cyclone Cross-Section (Side View)",
    show_annotations: bool = True,
    show_dimensions: bool = True
):
    """
    Render a 2D cross-section view of the cyclone.
    
    Args:
        assembly: CycloneAssembly instance
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        show_annotations: Show flow direction annotations
        show_dimensions: Show dimension labels
        
    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    else:
        fig = ax.figure
    
    p = assembly.params
    
    # Convert to mm
    D = p.cylinder_diameter * 1000
    R = D / 2
    H_cyl = p.cylinder_height * 1000
    H_cone = p.cone_height * 1000
    R_tip = p.cone_tip_diameter * 1000 / 2
    R_vf = p.vortex_finder_diameter * 1000 / 2
    L_vf = p.vortex_finder_length * 1000
    W_inlet = p.inlet_width * 1000
    H_inlet = p.inlet_height * 1000
    L_inlet = p.inlet_length * 1000
    L_outlet = p.dust_outlet_length * 1000
    
    # Colors
    body_color = '#4A90D9'
    vf_color = '#5CB85C'
    inlet_color = '#F0AD4E'
    outlet_color = '#D9534F'
    
    # Cylinder outline
    ax.plot([-R, -R], [0, -H_cyl], color=body_color, linewidth=2.5)
    ax.plot([R, R], [0, -H_cyl], color=body_color, linewidth=2.5)
    ax.plot([-R, R], [0, 0], color=body_color, linewidth=2.5)
    
    # Cone outline
    ax.plot([-R, -R_tip], [-H_cyl, -H_cyl - H_cone], color=body_color, linewidth=2.5)
    ax.plot([R, R_tip], [-H_cyl, -H_cyl - H_cone], color=body_color, linewidth=2.5)
    
    # Dust outlet
    ax.plot([-R_tip, -R_tip], [-H_cyl - H_cone, -H_cyl - H_cone - L_outlet], 
            color=outlet_color, linewidth=2.5)
    ax.plot([R_tip, R_tip], [-H_cyl - H_cone, -H_cyl - H_cone - L_outlet], 
            color=outlet_color, linewidth=2.5)
    ax.plot([-R_tip, R_tip], [-H_cyl - H_cone - L_outlet, -H_cyl - H_cone - L_outlet], 
            color=outlet_color, linewidth=2.5)
    
    # Vortex finder (tube representation)
    vf_top = 20  # protrusion above
    ax.plot([-R_vf, -R_vf], [vf_top, -L_vf], color=vf_color, linewidth=2.5)
    ax.plot([R_vf, R_vf], [vf_top, -L_vf], color=vf_color, linewidth=2.5)
    ax.plot([-R_vf, R_vf], [vf_top, vf_top], color=vf_color, linewidth=2.5)
    ax.plot([-R_vf, R_vf], [-L_vf, -L_vf], color=vf_color, linewidth=2, linestyle='--')
    
    # Inlet (rectangle on side)
    inlet_y_top = -50  # 50mm below top
    inlet_rect = patches.Rectangle(
        (R, inlet_y_top - H_inlet), L_inlet, H_inlet,
        fill=True, facecolor=inlet_color, edgecolor='darkorange', 
        alpha=0.6, linewidth=2
    )
    ax.add_patch(inlet_rect)
    
    # Fill regions with light colors
    # Cylinder fill
    cyl_verts = [(-R, 0), (R, 0), (R, -H_cyl), (-R, -H_cyl)]
    cyl_patch = patches.Polygon(cyl_verts, closed=True, facecolor=body_color, 
                                 alpha=0.15, edgecolor='none')
    ax.add_patch(cyl_patch)
    
    # Cone fill
    cone_verts = [(-R, -H_cyl), (R, -H_cyl), (R_tip, -H_cyl - H_cone), (-R_tip, -H_cyl - H_cone)]
    cone_patch = patches.Polygon(cone_verts, closed=True, facecolor=body_color,
                                  alpha=0.15, edgecolor='none')
    ax.add_patch(cone_patch)
    
    if show_annotations:
        # Gas out arrow
        ax.annotate('', xy=(0, vf_top + 5), xytext=(0, vf_top + 40),
                    arrowprops=dict(arrowstyle='->', color=vf_color, lw=2))
        ax.text(5, vf_top + 50, 'Clean Gas Out\n(Overflow)', fontsize=10, 
                color=vf_color, fontweight='bold')
        
        # Inlet arrow
        ax.annotate('', xy=(R + 5, inlet_y_top - H_inlet/2), 
                    xytext=(R + L_inlet + 30, inlet_y_top - H_inlet/2),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))
        ax.text(R + L_inlet + 35, inlet_y_top - H_inlet/2, 'Feed\n(Gas + Particles)', 
                fontsize=10, color='darkorange', va='center', fontweight='bold')
        
        # Particles out arrow
        ax.annotate('', xy=(0, -H_cyl - H_cone - L_outlet - 5), 
                    xytext=(0, -H_cyl - H_cone - L_outlet - 40),
                    arrowprops=dict(arrowstyle='->', color=outlet_color, lw=2))
        ax.text(5, -H_cyl - H_cone - L_outlet - 50, 'Particles Out\n(Underflow)', 
                fontsize=10, color=outlet_color, fontweight='bold')
        
        # Swirl indication
        ax.annotate('', xy=(R * 0.7, -H_cyl * 0.3), xytext=(R * 0.3, -H_cyl * 0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                                   connectionstyle='arc3,rad=0.3'))
        ax.text(R * 0.5, -H_cyl * 0.25, 'Swirl', fontsize=9, color='gray', style='italic')
    
    if show_dimensions:
        # Diameter dimension
        dim_y = -H_cyl/2
        ax.annotate('', xy=(-R, dim_y), xytext=(R, dim_y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(0, dim_y + 20, f'D = {D:.0f} mm', ha='center', fontsize=11, 
                color='red', fontweight='bold')
        
        # Total height dimension
        dim_x = -R - 40
        total_h = H_cyl + H_cone
        ax.annotate('', xy=(dim_x, 0), xytext=(dim_x, -total_h),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
        ax.text(dim_x - 10, -total_h/2, f'H = {total_h:.0f} mm', 
                ha='right', va='center', fontsize=11, color='purple', 
                fontweight='bold', rotation=90)
    
    # Set limits
    ax.set_xlim(-R - 80, R + L_inlet + 100)
    ax.set_ylim(-H_cyl - H_cone - L_outlet - 80, vf_top + 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_cyclone_top_view(
    assembly,
    vertices: Optional[np.ndarray] = None,
    ax=None,
    title: str = "Cyclone Top View (XZ Plane)",
    show_mesh_points: bool = True
):
    """
    Plot top-down view of cyclone showing XZ plane.
    
    Args:
        assembly: CycloneAssembly instance
        vertices: Optional mesh vertices to plot as scatter
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        show_mesh_points: Show mesh vertices as scatter
        
    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    p = assembly.params
    R = p.cylinder_diameter * 1000 / 2
    R_vf = p.vortex_finder_diameter * 1000 / 2
    R_tip = p.cone_tip_diameter * 1000 / 2
    W_inlet = p.inlet_width * 1000
    L_inlet = p.inlet_length * 1000
    
    # Plot mesh vertices if provided
    if vertices is not None and show_mesh_points:
        verts_mm = vertices * 1000
        ax.scatter(verts_mm[:, 0], verts_mm[:, 2], s=0.5, c='steelblue', alpha=0.3)
    
    # Draw circles
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Cylinder outline
    ax.plot(R * np.cos(theta), R * np.sin(theta), 'b-', linewidth=2.5, 
            label=f'Cylinder (D={p.cylinder_diameter*1000:.0f}mm)')
    
    # Vortex finder
    ax.plot(R_vf * np.cos(theta), R_vf * np.sin(theta), 'g--', linewidth=2, 
            label=f'Vortex Finder (D={p.vortex_finder_diameter*1000:.0f}mm)')
    
    # Cone tip (dust outlet)
    ax.plot(R_tip * np.cos(theta), R_tip * np.sin(theta), 'r:', linewidth=2, 
            label=f'Dust Outlet (D={p.cone_tip_diameter*1000:.0f}mm)')
    
    # Inlet representation (rectangle)
    inlet_rect = plt.Rectangle(
        (R, -W_inlet/2), L_inlet, W_inlet,
        fill=True, facecolor='orange', edgecolor='darkorange', 
        alpha=0.5, linewidth=2, label='Inlet'
    )
    ax.add_patch(inlet_rect)
    
    # Tangential flow indicator
    ax.annotate('', xy=(R * 0.8, R * 0.6), xytext=(R * 0.6, R * 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(R * 0.5, R * 0.9, 'Swirl', fontsize=10, color='blue', style='italic')
    
    ax.set_xlim(-R - 30, R + L_inlet + 30)
    ax.set_ylim(-R - 30, R + 30)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def render_cyclone_assembly(
    assembly,
    save_prefix: str = "cyclone",
    show: bool = False,
    export_stl: bool = True,
    figsize_3d: Tuple[int, int] = (14, 10),
    figsize_2d: Tuple[int, int] = (10, 12)
) -> Dict[str, Any]:
    """
    Complete cyclone assembly visualization.
    
    Creates 3D mesh view, cross-section, and top view.
    Optionally exports STL file for external viewing.
    
    Args:
        assembly: CycloneAssembly instance
        save_prefix: Prefix for saved files
        show: Whether to display interactive plots
        export_stl: Whether to export STL file
        figsize_3d: Figure size for 3D view
        figsize_2d: Figure size for 2D views
        
    Returns:
        Dictionary with created figures and file paths
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')  # Non-interactive backend for saving only
    import matplotlib.pyplot as plt
    from ..geometry.mesh_generator import export_mesh_stl
    
    results = {
        'figures': {},
        'files': []
    }
    
    # Build mesh
    vertices, indices = assembly.build_mesh()
    triangles = indices.reshape(-1, 3)
    
    print(f"Mesh: {len(vertices)} vertices, {len(triangles)} triangles")
    
    # Export STL
    if export_stl:
        stl_file = f"{save_prefix}_assembly.stl"
        export_mesh_stl(vertices, triangles, stl_file, binary=True)
        results['files'].append(stl_file)
        print(f"Exported: {stl_file}")
    
    # 3D mesh view
    fig1 = plt.figure(figsize=figsize_3d)
    ax1 = fig1.add_subplot(111, projection='3d')
    D_mm = assembly.params.cylinder_diameter * 1000
    plot_cyclone_mesh_3d(
        vertices, indices, ax=ax1,
        title=f"Cyclone Air Classifier (D = {D_mm:.0f} mm)"
    )
    ax1.view_init(elev=20, azim=45)
    
    fig1_file = f"{save_prefix}_3d_view.png"
    fig1.savefig(fig1_file, dpi=150, bbox_inches='tight')
    results['figures']['3d'] = fig1
    results['files'].append(fig1_file)
    print(f"Saved: {fig1_file}")
    
    # Cross-section view
    fig2, ax2 = plt.subplots(figsize=figsize_2d)
    plot_cyclone_cross_section(assembly, ax=ax2)
    
    fig2_file = f"{save_prefix}_cross_section.png"
    fig2.savefig(fig2_file, dpi=150, bbox_inches='tight')
    results['figures']['cross_section'] = fig2
    results['files'].append(fig2_file)
    print(f"Saved: {fig2_file}")
    
    # Top view
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    plot_cyclone_top_view(assembly, vertices=vertices, ax=ax3)
    
    fig3_file = f"{save_prefix}_top_view.png"
    fig3.savefig(fig3_file, dpi=150, bbox_inches='tight')
    results['figures']['top_view'] = fig3
    results['files'].append(fig3_file)
    print(f"Saved: {fig3_file}")
    
    if show:
        plt.show()
    
    return results
