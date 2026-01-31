"""
Visualization script for Phase 1 components.

Generates 3D renders and saves images of:
1. Zigzag Classifier
2. Venturi Eductor
3. Multi-Cyclone System (3-stage protein separation)
4. Bag Filter

Run with: python visualize_phase1_components.py [--show]
"""

import sys
import argparse
import numpy as np

# Set non-interactive backend by default
show_plots = '--show' in sys.argv
if not show_plots:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from airclassifier.geometry.components import (
    create_standard_zigzag_classifier,
    create_standard_venturi_eductor,
    create_protein_separation_cyclones,
    create_standard_bag_filter,
)
from airclassifier.geometry.assembly import (
    create_standard_classification_system,
    create_protein_separation_system,
)


def plot_mesh_3d(vertices, indices, ax, color='steelblue', alpha=0.7, title=''):
    """Plot a mesh in 3D."""
    triangles = indices.reshape(-1, 3)
    
    # Create triangles for plotting
    tri_verts = []
    for tri in triangles[:500]:  # Limit triangles for performance
        v0, v1, v2 = tri
        tri_verts.append([vertices[v0], vertices[v1], vertices[v2]])
    
    collection = Poly3DCollection(tri_verts, alpha=alpha, 
                                   facecolor=color, edgecolor='darkblue',
                                   linewidth=0.3)
    ax.add_collection3d(collection)
    
    # Set axis limits
    max_range = np.ptp(vertices, axis=0).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)


def render_zigzag_classifier():
    """Render zigzag classifier."""
    print("Rendering Zigzag Classifier...")
    
    classifier = create_standard_zigzag_classifier(channel_width=0.15, num_stages=5)
    verts, idx, _ = classifier.generate_mesh()
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(verts, idx, ax1, color='forestgreen', title='Zigzag Classifier - 3D View')
    ax1.view_init(elev=20, azim=45)
    
    # Side view (X-Y plane)
    ax2 = fig.add_subplot(122)
    ax2.fill(verts[:, 0], verts[:, 1], alpha=0.5, color='forestgreen')
    ax2.scatter(verts[:, 0], verts[:, 1], s=5, c='darkgreen')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Zigzag Classifier - Side View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    p = classifier.params
    ax2.annotate(f'Stages: {p.num_stages}', xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, va='top')
    ax2.annotate(f'Height: {p.total_height*1000:.0f} mm', xy=(0.02, 0.93), 
                 xycoords='axes fraction', fontsize=10, va='top')
    ax2.annotate(f'Channel width: {p.channel_width*1000:.0f} mm', xy=(0.02, 0.88),
                 xycoords='axes fraction', fontsize=10, va='top')
    
    plt.tight_layout()
    plt.savefig('zigzag_classifier.png', dpi=150)
    print("  Saved: zigzag_classifier.png")
    return fig


def render_venturi_eductor():
    """Render venturi eductor."""
    print("Rendering Venturi Eductor...")
    
    eductor = create_standard_venturi_eductor(inlet_diameter=0.1, throat_ratio=0.5)
    verts, idx, _ = eductor.generate_mesh()
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(verts, idx, ax1, color='coral', title='Venturi Eductor - 3D View')
    ax1.view_init(elev=15, azim=30)
    
    # Cross-section view
    ax2 = fig.add_subplot(122)
    
    # Plot the venturi profile
    p = eductor.params
    x_vals = []
    r_vals_top = []
    r_vals_bot = []
    
    # Convergent section
    for i in range(20):
        t = i / 19
        x = t * p.convergent_length
        r = p.inlet_diameter/2 + (p.throat_diameter/2 - p.inlet_diameter/2) * t
        x_vals.append(x)
        r_vals_top.append(r)
        r_vals_bot.append(-r)
    
    # Throat
    for i in range(5):
        t = i / 4
        x = p.convergent_length + t * p.throat_length
        r = p.throat_diameter / 2
        x_vals.append(x)
        r_vals_top.append(r)
        r_vals_bot.append(-r)
    
    # Divergent section
    for i in range(20):
        t = i / 19
        x = p.convergent_length + p.throat_length + t * p.divergent_length
        r = p.throat_diameter/2 + (p.outlet_diameter/2 - p.throat_diameter/2) * t
        x_vals.append(x)
        r_vals_top.append(r)
        r_vals_bot.append(-r)
    
    ax2.fill_between(x_vals, r_vals_bot, r_vals_top, alpha=0.5, color='coral')
    ax2.plot(x_vals, r_vals_top, 'r-', linewidth=2)
    ax2.plot(x_vals, r_vals_bot, 'r-', linewidth=2)
    
    # Add solids inlet
    throat_x = p.throat_start_position + p.solids_inlet_position
    ax2.annotate('Solids\nInlet', xy=(throat_x, p.throat_diameter/2), 
                 xytext=(throat_x, p.inlet_diameter*0.8),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 ha='center', fontsize=10)
    
    ax2.axvline(x=p.convergent_length, color='gray', linestyle='--', alpha=0.5, label='Throat start')
    ax2.axvline(x=p.throat_end_position, color='gray', linestyle='--', alpha=0.5, label='Throat end')
    
    ax2.set_xlabel('Axial Position (m)')
    ax2.set_ylabel('Radius (m)')
    ax2.set_title('Venturi Eductor - Cross Section')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(loc='lower right')
    
    # Annotations
    ax2.annotate(f'Inlet D: {p.inlet_diameter*1000:.0f} mm', xy=(0.02, 0.98),
                 xycoords='axes fraction', fontsize=10, va='top')
    ax2.annotate(f'Throat D: {p.throat_diameter*1000:.0f} mm', xy=(0.02, 0.93),
                 xycoords='axes fraction', fontsize=10, va='top')
    ax2.annotate(f'Area ratio: {p.area_ratio:.1f}', xy=(0.02, 0.88),
                 xycoords='axes fraction', fontsize=10, va='top')
    
    plt.tight_layout()
    plt.savefig('venturi_eductor.png', dpi=150)
    print("  Saved: venturi_eductor.png")
    return fig


def render_multi_cyclone():
    """Render multi-cyclone system."""
    print("Rendering Multi-Cyclone System...")
    
    system = create_protein_separation_cyclones()
    verts, idx, _ = system.generate_mesh()
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(verts, idx, ax1, color='royalblue', title='Multi-Cyclone System - 3D View')
    ax1.view_init(elev=25, azim=45)
    
    # Top view with labels
    ax2 = fig.add_subplot(122)
    
    # Plot cyclone positions
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Primary\n(Coarse)', 'Secondary\n(Medium)', 'Tertiary\n(Fine)']
    
    for i, (stage, color, label) in enumerate(zip(system.params.stages, colors, labels)):
        cyclone = system.get_cyclone(stage.name)
        center = cyclone.params.center
        r = stage.diameter / 2
        
        circle = plt.Circle((center[0], center[2]), r, 
                            color=color, alpha=0.6, label=f'{stage.name.title()}')
        ax2.add_patch(circle)
        
        ax2.annotate(label, xy=(center[0], center[2]), ha='center', va='center',
                    fontsize=10, fontweight='bold')
        ax2.annotate(f'D={stage.diameter*1000:.0f}mm\nd50={stage.design_d50*1e6:.0f}um',
                    xy=(center[0], center[2] - r - 0.05), ha='center', fontsize=9)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Multi-Cyclone System - Top View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Set limits
    min_b, max_b = system.get_system_bounds()
    margin = 0.2
    ax2.set_xlim(min_b[0] - margin, max_b[0] + margin)
    ax2.set_ylim(min_b[2] - margin, max_b[2] + margin)
    
    # Add flow arrows
    ax2.annotate('', xy=(min_b[0] - 0.1, 0), xytext=(min_b[0] - 0.3, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(min_b[0] - 0.4, 0, 'Feed', ha='right', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('multi_cyclone_system.png', dpi=150)
    print("  Saved: multi_cyclone_system.png")
    return fig


def render_bag_filter():
    """Render bag filter."""
    print("Rendering Bag Filter...")
    
    bag_filter = create_standard_bag_filter(flow_rate_m3s=1.0, air_to_cloth=2.0)
    verts, idx, _ = bag_filter.generate_mesh()
    
    fig = plt.figure(figsize=(14, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(verts, idx, ax1, color='mediumpurple', 
                 title='Bag Filter - 3D View', alpha=0.5)
    ax1.view_init(elev=20, azim=30)
    
    # Schematic side view
    ax2 = fig.add_subplot(122)
    p = bag_filter.params
    
    # Draw housing outline
    hw = p.housing_width / 2
    housing_x = [p.center[0] - hw, p.center[0] + hw, p.center[0] + hw, 
                 p.center[0] - hw, p.center[0] - hw]
    housing_y = [p.center[1] + p.hopper_height, p.center[1] + p.hopper_height,
                 p.center[1] + p.housing_height, p.center[1] + p.housing_height,
                 p.center[1] + p.hopper_height]
    ax2.plot(housing_x, housing_y, 'b-', linewidth=2)
    
    # Draw hopper
    hopper_x = [p.center[0] - hw, p.center[0] - p.hopper_outlet_width/2,
                p.center[0] + p.hopper_outlet_width/2, p.center[0] + hw]
    hopper_y = [p.center[1] + p.hopper_height, p.center[1],
                p.center[1], p.center[1] + p.hopper_height]
    ax2.plot(hopper_x, hopper_y, 'b-', linewidth=2)
    
    # Draw bags (simplified)
    bag_top = p.center[1] + p.tube_sheet_height + p.tube_sheet_thickness
    for i in range(min(p.num_bags_x, 6)):
        x = p.center[0] - (p.num_bags_x - 1) * p.bag_spacing_x / 2 + i * p.bag_spacing_x
        ax2.add_patch(plt.Rectangle(
            (x - p.bag_diameter/2, bag_top - p.bag_length),
            p.bag_diameter, p.bag_length,
            facecolor='lightgray', edgecolor='gray', alpha=0.7
        ))
    
    # Draw tube sheet
    ax2.axhline(y=p.tube_sheet_height + p.center[1], color='brown', 
                linewidth=3, label='Tube Sheet')
    
    # Labels
    ax2.annotate('Clean Air\nPlenum', xy=(0.85, 0.85), xycoords='axes fraction',
                ha='center', fontsize=10)
    ax2.annotate('Bag Zone', xy=(0.85, 0.55), xycoords='axes fraction',
                ha='center', fontsize=10)
    ax2.annotate('Dirty Air\nSection', xy=(0.85, 0.35), xycoords='axes fraction',
                ha='center', fontsize=10)
    ax2.annotate('Hopper', xy=(0.85, 0.15), xycoords='axes fraction',
                ha='center', fontsize=10)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Bag Filter - Schematic Side View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Info box
    info_text = (f"Bags: {p.num_bags_x} x {p.num_bags_z} = {p.num_bags}\n"
                 f"Bag size: {p.bag_diameter*1000:.0f}mm x {p.bag_length*1000:.0f}mm\n"
                 f"Filter area: {p.total_filter_area:.1f} m2\n"
                 f"A/C ratio: {p.get_air_to_cloth(1.0):.2f} m3/min/m2")
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('bag_filter.png', dpi=150)
    print("  Saved: bag_filter.png")
    return fig


def render_complete_system():
    """Render complete classification system layout (schematic)."""
    print("Rendering Complete System Layout (Schematic)...")
    
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)
    
    # System layout (top view schematic)
    # Feed -> Zigzag -> Cyclone(s) -> Bag Filter
    
    y_center = 0.5
    
    # Feed
    ax.add_patch(plt.Rectangle((0.05, 0.4), 0.08, 0.2, facecolor='wheat', edgecolor='black'))
    ax.text(0.09, 0.5, 'Feed\nHopper', ha='center', va='center', fontsize=9)
    
    # Venturi Eductor
    ax.add_patch(FancyBboxPatch((0.15, 0.45), 0.08, 0.1, boxstyle='round,pad=0.01',
                                 facecolor='coral', edgecolor='black'))
    ax.text(0.19, 0.5, 'Venturi\nEductor', ha='center', va='center', fontsize=9)
    
    # Zigzag Classifier
    ax.add_patch(plt.Rectangle((0.26, 0.3), 0.1, 0.4, facecolor='forestgreen', 
                                edgecolor='black', alpha=0.7))
    ax.text(0.31, 0.5, 'Zigzag\nClassifier', ha='center', va='center', fontsize=9, color='white')
    
    # Primary Cyclone
    circle1 = plt.Circle((0.45, 0.5), 0.06, facecolor='#FF6B6B', edgecolor='black', alpha=0.7)
    ax.add_patch(circle1)
    ax.text(0.45, 0.5, 'Primary\nCyclone', ha='center', va='center', fontsize=8)
    
    # Secondary Cyclone
    circle2 = plt.Circle((0.58, 0.5), 0.045, facecolor='#4ECDC4', edgecolor='black', alpha=0.7)
    ax.add_patch(circle2)
    ax.text(0.58, 0.5, 'Secondary\nCyclone', ha='center', va='center', fontsize=8)
    
    # Tertiary Cyclone
    circle3 = plt.Circle((0.69, 0.5), 0.035, facecolor='#45B7D1', edgecolor='black', alpha=0.7)
    ax.add_patch(circle3)
    ax.text(0.69, 0.5, 'Tertiary', ha='center', va='center', fontsize=8)
    
    # Bag Filter
    ax.add_patch(plt.Rectangle((0.78, 0.35), 0.12, 0.3, facecolor='mediumpurple',
                                edgecolor='black', alpha=0.7))
    ax.text(0.84, 0.5, 'Bag\nFilter', ha='center', va='center', fontsize=9, color='white')
    
    # Fan/Blower
    ax.add_patch(plt.Circle((0.95, 0.5), 0.03, facecolor='gray', edgecolor='black'))
    ax.text(0.95, 0.5, 'Fan', ha='center', va='center', fontsize=8)
    
    # Collection bins
    for x, label in [(0.45, 'Starch\n(Coarse)'), (0.58, 'Mixed'), (0.69, 'Protein\n(Fine)')]:
        ax.add_patch(plt.Rectangle((x-0.02, 0.1), 0.04, 0.08, facecolor='tan', edgecolor='black'))
        ax.text(x, 0.06, label, ha='center', va='top', fontsize=8)
    
    ax.add_patch(plt.Rectangle((0.82, 0.1), 0.04, 0.08, facecolor='tan', edgecolor='black'))
    ax.text(0.84, 0.06, 'Ultra-fine', ha='center', va='top', fontsize=8)
    
    # Flow arrows
    arrows = [
        ((0.13, 0.5), (0.15, 0.5)),
        ((0.23, 0.5), (0.26, 0.5)),
        ((0.36, 0.65), (0.39, 0.5)),   # Fines from zigzag
        ((0.36, 0.35), (0.36, 0.2)),   # Coarse from zigzag
        ((0.51, 0.5), (0.54, 0.5)),
        ((0.62, 0.5), (0.66, 0.5)),
        ((0.72, 0.5), (0.78, 0.5)),
        ((0.90, 0.5), (0.92, 0.5)),
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Down arrows to collection
    for x in [0.45, 0.58, 0.69, 0.84]:
        ax.annotate('', xy=(x, 0.18), xytext=(x, 0.28),
                   arrowprops=dict(arrowstyle='->', color='darkgray', lw=1))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Protein Separation System - Process Flow Diagram', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.text(0.5, 0.02, 'Air Flow Direction', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('complete_system_layout.png', dpi=150)
    print("  Saved: complete_system_layout.png")
    return fig


def render_assembled_system_3d():
    """Render the complete assembled system in 3D using ClassificationSystemAssembly."""
    print("Rendering Assembled System (3D)...")
    
    # Create the complete system
    system = create_standard_classification_system()
    vertices, indices = system.build_mesh()
    
    fig = plt.figure(figsize=(18, 10))
    
    # 3D view - isometric
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(vertices, indices, ax1, color='steelblue', alpha=0.6,
                 title='Complete Classification System - 3D View')
    ax1.view_init(elev=25, azim=45)
    
    # 3D view - top down
    ax2 = fig.add_subplot(122, projection='3d')
    plot_mesh_3d(vertices, indices, ax2, color='steelblue', alpha=0.6,
                 title='Complete Classification System - Top View')
    ax2.view_init(elev=80, azim=0)
    
    # Add info text
    extent = system.get_system_extent()
    info_text = (
        f"System Extent:\n"
        f"  Length: {extent[0]*1000:.0f} mm\n"
        f"  Height: {extent[1]*1000:.0f} mm\n"
        f"  Depth: {extent[2]*1000:.0f} mm\n\n"
        f"Components:\n"
        f"  - Venturi Eductor\n"
        f"  - Zigzag Classifier ({system.zigzag.params.num_stages} stages)\n"
        f"  - Multi-Cyclone (3 stages)\n"
        f"  - Bag Filter ({system.bag_filter.params.num_bags} bags)"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('assembled_system_3d.png', dpi=150)
    print("  Saved: assembled_system_3d.png")
    
    # Print summary
    print("\n  System Summary:")
    system.print_summary()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase 1 components')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--assembly-only', action='store_true', 
                       help='Only render the assembled system')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Phase 1 Component Visualization")
    print("=" * 60)
    
    figs = []
    
    if args.assembly_only:
        # Only render the assembled system
        figs.append(render_assembled_system_3d())
    else:
        # Render all individual components
        figs.append(render_zigzag_classifier())
        figs.append(render_venturi_eductor())
        figs.append(render_multi_cyclone())
        figs.append(render_bag_filter())
        figs.append(render_complete_system())
        figs.append(render_assembled_system_3d())
    
    print("=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
    
    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
