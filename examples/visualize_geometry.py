#!/usr/bin/env python
"""
Geometry Visualization Example Script

This script demonstrates how to visualize air classifier geometries:
- Individual components with --color (color-coded) or --mesh (wireframe) modes
- Assembled systems (feed, air, classification)
- Complete core system with duct connections

Coordinate System (Y-up):
- X: Horizontal (width)
- Y: Vertical (height) - UP
- Z: Horizontal (depth)

Individual Component Modes:
    python examples/visualize_geometry.py --cyclone --color    # Cyclone with colors
    python examples/visualize_geometry.py --cyclone --mesh     # Cyclone wireframe
    python examples/visualize_geometry.py --multicyclone --color
    python examples/visualize_geometry.py --blower --mesh
    python examples/visualize_geometry.py --deagglomerator --color
    python examples/visualize_geometry.py --hopper --mesh
    python examples/visualize_geometry.py --airlock --color
    python examples/visualize_geometry.py --zigzag --mesh

Assembly Modes:
    python examples/visualize_geometry.py --feed       # Feed system assembly
    python examples/visualize_geometry.py --air        # Air system assembly
    python examples/visualize_geometry.py --classification  # Classification system

System Modes:
    python examples/visualize_geometry.py --core       # Core system (3 duct connections)
    python examples/visualize_geometry.py --pilot      # Pilot-scale system
    python examples/visualize_geometry.py --production # Production-scale system
    python examples/visualize_geometry.py --export     # Export all to STL files
    python examples/visualize_geometry.py --all        # All visualizations

Requirements:
    pip install pyvista  # For high-quality 3D (recommended)
    # Falls back to matplotlib if PyVista not available
"""

import argparse
import sys
import os

# Add src to path if running from examples folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from airclassifier.visualization import (
    GeometryVisualizer,
    VisualizationRequest,
    visualize_geometry,
    quick_render,
    PYVISTA_AVAILABLE,
)

# Component colors
COMPONENT_COLORS = {
    'cyclone': '#4A90D9',       # Blue
    'multicyclone': '#E74C3C',  # Red
    'blower': '#27AE60',        # Green
    'deagglomerator': '#9B59B6', # Purple
    'hopper': '#F0AD4E',        # Orange
    'airlock': '#3498DB',       # Light Blue
    'zigzag': '#2ECC71',        # Emerald
    'venturi': '#1ABC9C',       # Teal
    'bagfilter': '#F39C12',     # Yellow-Orange
}


def check_dependencies():
    """Check and report available visualization backends."""
    print("=" * 60)
    print("VISUALIZATION BACKEND STATUS")
    print("=" * 60)

    if PYVISTA_AVAILABLE:
        print("[OK] PyVista available - High-quality 3D rendering enabled")
    else:
        print("[!] PyVista not installed - Using matplotlib fallback")
        print("    Install with: pip install pyvista")

    try:
        import warp as wp
        print("[OK] NVIDIA Warp available - GPU acceleration enabled")
    except ImportError:
        print("[!] NVIDIA Warp not installed - CPU mode only")
        print("    Install with: pip install warp-lang")

    print("=" * 60)
    print()


def render_component(component, name: str, color: str, use_mesh: bool = False, use_color: bool = True):
    """
    Render a single component with color or mesh mode.

    Args:
        component: The geometry component to render
        name: Display name for the component
        color: Color to use in color mode
        use_mesh: If True, render as wireframe mesh
        use_color: If True, render with solid colors (default)
    """
    import numpy as np

    vertices, indices, normals = component.generate_mesh()

    print(f"\nComponent: {name}")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Triangles: {len(indices)//3:,}")

    if PYVISTA_AVAILABLE:
        import pyvista as pv

        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

        faces = np.hstack([[3] + list(face) for face in indices.reshape(-1, 3)])
        mesh = pv.PolyData(vertices, faces)

        if use_mesh:
            # Wireframe mesh mode
            plotter.add_mesh(mesh, style='wireframe', color='black',
                           line_width=1, label=name)
            plotter.add_title(f'{name} - Mesh View')
        else:
            # Color mode (solid with edges)
            plotter.add_mesh(mesh, color=color, opacity=0.85,
                           show_edges=True, edge_color='gray',
                           label=name)
            plotter.add_title(f'{name} - Color View')

        plotter.add_axes()
        plotter.add_legend(bcolor='white', face='circle')
        
        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        return {'success': True, 'message': f'{name} visualized with PyVista'}
    else:
        # Fallback to basic visualization
        viz = GeometryVisualizer()
        result = viz.visualize_component(
            component,
            name=name,
            show=True,
            opacity=0.8 if use_color else 0.3,
            color=color,
            title=f"{name} - {'Mesh' if use_mesh else 'Color'} View"
        )
        return result


def visualize_cyclone(use_mesh: bool = False):
    """Visualize a single cyclone body."""
    print("\n" + "=" * 60)
    print("CYCLONE BODY VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components import (
        CycloneBody,
        CycloneBodyParams,
    )

    params = CycloneBodyParams(
        cylinder_diameter=0.3,
        cylinder_height=0.3,
        cone_height=0.5,
        cone_tip_diameter=0.05,
    )
    cyclone = CycloneBody(params)

    print(f"  Cylinder: D={params.cylinder_diameter*1000:.0f}mm, H={params.cylinder_height*1000:.0f}mm")
    print(f"  Cone: H={params.cone_height*1000:.0f}mm, tip D={params.cone_tip_diameter*1000:.0f}mm")

    result = render_component(cyclone, "Cyclone Body",
                             COMPONENT_COLORS['cyclone'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_multicyclone(use_mesh: bool = False):
    """Visualize a multi-cyclone system."""
    print("\n" + "=" * 60)
    print("MULTI-CYCLONE SYSTEM VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.multi_cyclone import (
        create_protein_separation_cyclones,
    )

    system = create_protein_separation_cyclones(
        primary_diameter=0.4,
        secondary_diameter=0.25,
        tertiary_diameter=0.15
    )

    print("Multi-Cyclone System (3-stage protein separation):")
    print("  - Primary cyclone: D=400mm (d50 ~ 40 um)")
    print("  - Secondary cyclone: D=250mm (d50 ~ 20 um)")
    print("  - Tertiary cyclone: D=150mm (d50 ~ 10 um)")

    result = render_component(system, "Multi-Cyclone System",
                             COMPONENT_COLORS['multicyclone'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_blower(use_mesh: bool = False):
    """Visualize a centrifugal blower."""
    print("\n" + "=" * 60)
    print("CENTRIFUGAL BLOWER VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.centrifugal_blower import (
        create_standard_centrifugal_blower,
    )

    blower = create_standard_centrifugal_blower(flow_rate=3000, pressure_rise=5000)

    print("Centrifugal Blower:")
    print(f"  Flow rate: 3000 m³/h")
    print(f"  Pressure rise: 5000 Pa")
    print(f"  Impeller: {blower.params.num_blades} backward-curved blades")

    result = render_component(blower, "Centrifugal Blower",
                             COMPONENT_COLORS['blower'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_deagglomerator(use_mesh: bool = False):
    """Visualize a deagglomerator."""
    print("\n" + "=" * 60)
    print("DEAGGLOMERATOR VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.deagglomerator import (
        create_standard_deagglomerator,
    )

    deagg = create_standard_deagglomerator(rotor_diameter=0.2, screen_aperture=0.002)

    print("Deagglomerator (Pin Rotor + Screen):")
    print(f"  Rotor diameter: 200mm")
    print(f"  Pin rows: {deagg.params.num_pin_rows}")
    print(f"  Pins per row: {deagg.params.pins_per_row}")
    print(f"  Screen aperture: 2mm")

    result = render_component(deagg, "Deagglomerator",
                             COMPONENT_COLORS['deagglomerator'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_hopper(use_mesh: bool = False):
    """Visualize a feed hopper."""
    print("\n" + "=" * 60)
    print("FEED HOPPER VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components import (
        FeedHopper,
        FeedHopperParams,
    )

    params = FeedHopperParams(
        top_diameter=0.6,
        bottom_diameter=0.15,
        cylindrical_height=0.3,
        conical_height=0.5,
        has_lid=True,
    )
    hopper = FeedHopper(params)

    print("Feed Hopper (Conical Mass-Flow Design):")
    print(f"  Top diameter: {params.top_diameter*1000:.0f}mm")
    print(f"  Bottom diameter: {params.bottom_diameter*1000:.0f}mm")
    print(f"  Total height: {params.total_height*1000:.0f}mm")
    print(f"  Has lid: {params.has_lid}")

    result = render_component(hopper, "Feed Hopper",
                             COMPONENT_COLORS['hopper'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_airlock(use_mesh: bool = False):
    """Visualize a rotary airlock."""
    print("\n" + "=" * 60)
    print("ROTARY AIRLOCK VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.rotary_airlock import (
        create_standard_rotary_airlock,
    )

    airlock = create_standard_rotary_airlock(rotor_diameter=0.2, capacity_m3_h=5.0)

    print("Rotary Airlock (Pressure Seal):")
    print(f"  Rotor diameter: 200mm")
    print(f"  Number of vanes: {airlock.params.num_vanes}")
    print(f"  RPM: {airlock.params.rpm}")
    print(f"  Capacity: 5.0 m³/h")

    result = render_component(airlock, "Rotary Airlock",
                             COMPONENT_COLORS['airlock'], use_mesh=use_mesh)
    print(f"\nResult: {result['message']}")
    return result


def visualize_zigzag(use_mesh: bool = False):
    """
    Visualize a zigzag classifier using the actual geometry mesh.

    The zigzag classifier is a venturi-fed counter-current air classifier:
    - Particles + air enter from BOTTOM via venturi eductor
    - Air flows UP through the channel
    - Deflector plates create separation zones with recirculation
    - Light particles (protein) rise with air -> exit fines outlet (top)
    - Heavy particles (starch) fall against air -> exit coarse outlet (bottom)

    No side feed inlet is shown because particles enter via the venturi below.
    """
    print("\n" + "=" * 60)
    print("ZIGZAG CLASSIFIER VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.zigzag_classifier import (
        create_standard_zigzag_classifier
    )
    import numpy as np

    # Create classifier using factory function (venturi-fed, no side feed inlet)
    classifier = create_standard_zigzag_classifier(
        channel_width=0.15,
        num_stages=5,
        channel_depth=0.30,
        plate_angle_deg=45.0,
        plate_length_ratio=0.5,
    )

    print("Zigzag Classifier (Venturi-Fed Counter-Current Design):")
    print(f"  Channel width: {classifier.params.channel_width * 1000:.0f}mm")
    print(f"  Channel depth: {classifier.params.channel_depth * 1000:.0f}mm")
    print(f"  Number of stages: {classifier.params.num_stages}")
    print(f"\nDeflector Plate Geometry:")
    print(f"  Plate angle: {np.degrees(classifier.params.plate_angle):.0f} deg from vertical")
    print(f"  Plate length: {classifier.params.plate_length * 1000:.1f}mm ({classifier.params.plate_length_ratio*100:.0f}% of width)")
    print(f"  Throat width: {classifier.params.throat_width * 1000:.1f}mm")
    print(f"  Blockage ratio: {classifier.params.blockage_ratio * 100:.0f}%")
    print(f"\nSeparation Physics:")
    print(f"  Velocity ratio in separation zones: {classifier.params.velocity_ratio_in_zone:.1%} of bulk")
    print(f"  Turbulence intensity: {classifier.params.turbulence_intensity:.0%}")
    print(f"\nFlow Path (Venturi-Fed):")
    print("  - Air + particles enter from BOTTOM via venturi eductor")
    print("  - Light particles (v_t < v_air) rise -> fines outlet (TOP)")
    print("  - Heavy particles (v_t > v_air) fall -> coarse outlet (BOTTOM)")

    # Get actual mesh from component
    vertices, indices, normals = classifier.generate_mesh()

    print(f"\nMesh Statistics:")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Triangles: {len(indices)//3:,}")

    if PYVISTA_AVAILABLE:
        import pyvista as pv

        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

        # Create PyVista mesh from actual geometry
        faces = np.hstack([[3] + list(face) for face in indices.reshape(-1, 3)])
        mesh = pv.PolyData(vertices, faces)

        if use_mesh:
            # Wireframe mesh mode
            plotter.add_mesh(mesh, style='wireframe', color='black',
                           line_width=1, label='Zigzag Classifier')
            plotter.add_title('Zigzag Classifier - Mesh View (Actual Geometry)')
        else:
            # Solid color mode with edges visible
            plotter.add_mesh(mesh, color=COMPONENT_COLORS['zigzag'], opacity=0.85,
                           show_edges=True, edge_color='gray',
                           label='Zigzag Classifier')
            plotter.add_title('Zigzag Classifier - Actual Geometry (Venturi-Fed)')

        # Add flow direction arrows
        p = classifier.params
        inlet_y_bottom = p.center[1] - p.air_inlet_height
        fines_y_top = p.center[1] + p.total_height + p.fines_outlet_height
        coarse_y_bottom = inlet_y_bottom - p.coarse_outlet_height

        # Air + particles flow up from venturi (blue arrow)
        arrow_start = np.array([[p.center[0], inlet_y_bottom - 0.05, p.center[2]]])
        arrow_dir = np.array([[0, 0.15, 0]])
        plotter.add_arrows(arrow_start, arrow_dir, color='blue', mag=1.0)

        # Fines flow up (green arrow)
        arrow_start2 = np.array([[p.center[0], fines_y_top + 0.02, p.center[2]]])
        arrow_dir2 = np.array([[0, 0.08, 0]])
        plotter.add_arrows(arrow_start2, arrow_dir2, color='green', mag=1.0)

        # Coarse flow down (orange arrow)
        arrow_start3 = np.array([[p.center[0], coarse_y_bottom - 0.02, p.center[2]]])
        arrow_dir3 = np.array([[0, -0.08, 0]])
        plotter.add_arrows(arrow_start3, arrow_dir3, color='orange', mag=1.0)

        # Add text labels for flow
        plotter.add_point_labels(
            [[p.center[0] + 0.1, inlet_y_bottom - 0.08, p.center[2]]],
            ['Air+Particles\n(from Venturi)'],
            font_size=10, text_color='blue', shape_opacity=0
        )
        plotter.add_point_labels(
            [[p.center[0] + 0.1, fines_y_top + 0.05, p.center[2]]],
            ['Fines (Protein)\n(light particles)'],
            font_size=10, text_color='green', shape_opacity=0
        )
        plotter.add_point_labels(
            [[p.center[0] + 0.1, coarse_y_bottom - 0.05, p.center[2]]],
            ['Coarse (Starch)\n(heavy particles)'],
            font_size=10, text_color='orange', shape_opacity=0
        )

        plotter.add_axes()
        plotter.add_legend(bcolor='white', face='circle')

        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        result = {'success': True, 'message': 'Zigzag classifier visualized with actual geometry'}
    else:
        # Fallback to basic visualization
        result = render_component(classifier, "Zigzag Classifier",
                                 COMPONENT_COLORS['zigzag'], use_mesh=use_mesh)

    print(f"\nResult: {result['message']}")
    return result


def visualize_feed_system_assembly():
    """Visualize the feed system assembly with color-coded components."""
    print("\n" + "=" * 60)
    print("FEED SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_standard_feed_system
    import numpy as np

    feed = create_standard_feed_system()
    vertices, indices = feed.build_mesh()

    print("Feed System Assembly includes:")
    print("  - Feed Hopper (conical mass-flow design)")
    print("  - Rotary Airlock (8-vane pressure seal)")
    print("  - Screw Feeder (ENCLOSED TUBE - dust-tight)")
    print("  - De-agglomerator (pin rotor + screen)")
    print("  - Flanged Transitions (sealed connections)")
    print(f"\nTotal mesh: {len(vertices):,} vertices, {len(indices)//3:,} triangles")

    # Print summary and transition report
    feed.print_summary()
    feed.print_transition_report()

    if PYVISTA_AVAILABLE:
        import pyvista as pv

        print("\nInitializing PyVista plotter...")
        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

        # Component colors
        colors = {
            'hopper': '#F0AD4E',        # Orange
            'airlock': '#3498DB',       # Light Blue
            'screw_feeder': '#27AE60',  # Green
            'deagglomerator': '#9B59B6', # Purple
            'transition': '#95A5A6',    # Gray
        }

        # Add hopper
        if feed.hopper is not None:
            print("  Adding hopper mesh...")
            try:
                v, i, _ = feed.hopper.generate_mesh()
                v = v + np.array(feed._hopper_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['hopper'], label='Feed Hopper', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add hopper: {e}")

        # Add airlock
        if feed.airlock is not None:
            print("  Adding airlock mesh...")
            try:
                v, i, _ = feed.airlock.generate_mesh()
                v = v + np.array(feed._airlock_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['airlock'], label='Rotary Airlock', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add airlock: {e}")

        # Add screw feeder
        if feed.feeder is not None:
            print("  Adding screw feeder mesh...")
            try:
                v, i, _ = feed.feeder.generate_mesh()
                v = v + np.array(feed._feeder_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['screw_feeder'], label='Screw Feeder', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add screw feeder: {e}")

        # Add deagglomerator
        if feed.deagglomerator is not None:
            print("  Adding deagglomerator mesh...")
            try:
                v, i, _ = feed.deagglomerator.generate_mesh()
                v = v + np.array(feed._deagglomerator_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['deagglomerator'], label='Deagglomerator', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add deagglomerator: {e}")

        # Add transition connectors (format: transition, position, name)
        if hasattr(feed, '_transition_connectors') and feed._transition_connectors:
            print(f"  Adding {len(feed._transition_connectors)} transition connectors...")
            for idx, connector_data in enumerate(feed._transition_connectors):
                try:
                    trans = connector_data[0]
                    # Position is baked into transition center, use (0,0,0) offset
                    v, i, _ = trans.generate_mesh()
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    plotter.add_mesh(mesh, color=colors['transition'],
                                    label="Transitions" if idx == 0 else None, opacity=0.7)
                except Exception as e:
                    print(f"    Warning: Failed to add transition {idx}: {e}")

        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_title('Feed System Assembly')
        plotter.add_axes()
        
        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        result = {'success': True, 'message': 'Feed system visualized with PyVista'}
    else:
        # Fallback to basic visualization
        viz = GeometryVisualizer()
        request = VisualizationRequest(
            target_type="assembly",
            assembly=feed,
            show=True,
            opacity=0.8,
            show_edges=True,
            title="Feed System Assembly (Enclosed Design)",
            show_labels=True,
        )
        result = viz.render(request)

    print(f"\nResult: {result['message']}")
    return result


def visualize_air_system_assembly():
    """Visualize the air system assembly with color-coded components."""
    print("\n" + "=" * 60)
    print("AIR SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_standard_air_system
    import numpy as np

    air = create_standard_air_system()
    vertices, indices = air.build_mesh()

    print("Air System Assembly includes:")
    print("  - Inlet Air Filter")
    print("  - Centrifugal Blower")
    print("  - Flow Damper")
    print(f"\nTotal mesh: {len(vertices):,} vertices, {len(indices)//3:,} triangles")

    air.print_summary()

    if PYVISTA_AVAILABLE:
        import pyvista as pv

        print("\nInitializing PyVista plotter...")
        plotter = pv.Plotter()
        plotter.set_background('white')
        plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

        # Component colors
        colors = {
            'air_filter': '#3498DB',    # Blue
            'blower': '#27AE60',        # Green
            'damper': '#F39C12',        # Orange
            'duct': '#95A5A6',          # Gray
        }

        # Add inlet filter
        if air.inlet_filter is not None:
            print("  Adding inlet filter mesh...")
            try:
                v, i, _ = air.inlet_filter.generate_mesh()
                v = v + np.array(air._filter_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['air_filter'], label='Inlet Filter', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add inlet filter: {e}")

        # Add blower
        if air.blower is not None:
            print("  Adding blower mesh...")
            try:
                v, i, _ = air.blower.generate_mesh()
                v = v + np.array(air._blower_position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['blower'], label='Centrifugal Blower', opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add blower: {e}")

        # Add dampers
        if hasattr(air, 'dampers') and air.dampers:
            print(f"  Adding {len(air.dampers)} damper(s)...")
            for idx, (damper, position) in enumerate(zip(air.dampers, air._damper_positions)):
                try:
                    v, i, _ = damper.generate_mesh()
                    v = v + np.array(position)
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    plotter.add_mesh(mesh, color=colors['damper'],
                                    label='Flow Damper' if idx == 0 else None, opacity=0.85)
                except Exception as e:
                    print(f"    Warning: Failed to add damper {idx}: {e}")

        # Add duct sections
        if hasattr(air, '_duct_sections') and air._duct_sections:
            print(f"  Adding {len(air._duct_sections)} duct sections...")
            for idx, (duct, position) in enumerate(air._duct_sections):
                try:
                    v, i, _ = duct.generate_mesh()
                    v = v + np.array(position)
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    plotter.add_mesh(mesh, color=colors['duct'],
                                    label='Ductwork' if idx == 0 else None, opacity=0.7)
                except Exception as e:
                    print(f"    Warning: Failed to add duct section {idx}: {e}")

        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_title('Air System Assembly')
        plotter.add_axes()
        
        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        result = {'success': True, 'message': 'Air system visualized with PyVista'}
    else:
        # Fallback to basic visualization
        viz = GeometryVisualizer()
        result = viz.visualize_assembly(
            air,
            name="Air System",
            show=True,
            opacity=0.8,
            color="#5CB85C",
            title="Air System Assembly"
        )

    print(f"\nResult: {result['message']}")
    return result


def visualize_classification_system():
    """Visualize the classification system assembly with color-coded components."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly.classification import create_standard_classification_system
    import numpy as np

    try:
        print("Creating classification system...")
        cls = create_standard_classification_system()
        print("Building mesh...")
        vertices, indices = cls.build_mesh()

        print("Classification System Assembly includes:")
        print("  - Venturi Eductor (particle entrainment)")
        print("  - Zigzag Classifier (primary separation)")
        print("  - Multi-Cyclone System (staged collection)")
        print("  - Bag Filter (fine particle capture)")
        print("  - Connecting Ductwork")
        print(f"\nTotal mesh: {len(vertices):,} vertices, {len(indices)//3:,} triangles")

        cls.print_summary()

        # Use PyVista for color-coded visualization if available
        if PYVISTA_AVAILABLE:
            import pyvista as pv

            print("\nInitializing PyVista plotter...")
            plotter = pv.Plotter()
            plotter.set_background('white')
            plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

            # Component colors
            colors = {
                'venturi': '#3498DB',      # Blue
                'zigzag': '#2ECC71',       # Green
                'multi_cyclone': '#E74C3C', # Red
                'bag_filter': '#F39C12',   # Orange
                'duct': '#95A5A6'          # Gray
            }

            # Add Venturi
            print("  Adding Venturi mesh...")
            v, i, _ = cls.venturi.generate_mesh()
            v = v + cls._component_positions['venturi']
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=colors['venturi'],
                            label='Venturi Eductor', opacity=0.85)

            # Add Zigzag
            print("  Adding Zigzag mesh...")
            v, i, _ = cls.zigzag.generate_mesh()
            v = v + cls._component_positions['zigzag']
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=colors['zigzag'],
                            label='Zigzag Classifier', opacity=0.85)

            # Add Multi-Cyclone
            print("  Adding Multi-Cyclone mesh...")
            v, i, _ = cls.multi_cyclone.generate_mesh()
            v = v + cls._component_positions['multi_cyclone']
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=colors['multi_cyclone'],
                            label='Multi-Cyclone System', opacity=0.85)

            # Add Bag Filter
            print("  Adding Bag Filter mesh...")
            v, i, _ = cls.bag_filter.generate_mesh()
            v = v + cls._component_positions['bag_filter']
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=colors['bag_filter'],
                            label='Bag Filter', opacity=0.85)

            # Add Ducts (new format: list of (duct_component, position) tuples)
            print(f"  Adding {len(cls._duct_sections)} duct sections...")
            for idx, (duct, position) in enumerate(cls._duct_sections):
                v, i, _ = duct.generate_mesh()
                v = v + np.array(position)  # Apply position offset
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['duct'],
                                label="Ductwork" if idx == 0 else None, opacity=0.7)

            plotter.add_legend(bcolor='white', face='circle')
            plotter.add_title('Classification System - Port-Based Assembly')
            plotter.add_axes()
            
            # Reset camera to fit entire scene and set isometric view
            plotter.reset_camera()
            plotter.camera.azimuth = -170
            plotter.camera.elevation = -20

            print("\nOpening visualization window...")
            print("(Close the window to continue)")
            plotter.show(interactive=True)

            result = {'success': True, 'message': 'Classification system visualized with PyVista'}
        else:
            # Fallback to basic visualization
            viz = GeometryVisualizer()
            result = viz.visualize_assembly(
                cls,
                name="Classification System",
                show=True,
                opacity=0.8,
                color="#9B59B6",
                title="Classification System Assembly"
            )

        print(f"\nResult: {result['message']}")
        return result

    except Exception as e:
        import traceback
        print(f"\nERROR: Failed to visualize classification system!")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return {'success': False, 'message': f'Error: {e}'}


def visualize_core_system():
    """
    Visualize the core system with all three duct connections.

    Shows the main flow path with ductwork connections:
    1. Air System -> Venturi air_inlet
    2. Feed System -> Venturi solids_inlet
    3. Bag Filter -> Exhaust (Silencer)
    """
    print("\n" + "=" * 60)
    print("CORE SYSTEM VISUALIZATION")
    print("=" * 60)
    print("Focus: 3 Main Ductwork Connections")
    print("  1. Air System -> Venturi (air_inlet)")
    print("  2. Feed System -> Venturi (solids_inlet)")
    print("  3. Bag Filter -> Exhaust (Silencer)")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_core_connections_system
    import numpy as np

    system = create_core_connections_system()

    # Print summary
    system.print_summary()
    print()
    system.print_bill_of_materials()

    # Use PyVista directly for better control over rendering
    try:
        import pyvista as pv

        # Colors for different subsystems
        colors = {
            'classification': '#3498DB',  # Blue
            'feed_system': '#27AE60',      # Green
            'air_system': '#F39C12',       # Orange
            'silencer': '#E74C3C',         # Red
            'exhaust_stack': '#9B59B6',    # Purple
            'ductwork': '#7F8C8D',         # Gray
        }

        print("\nInitializing PyVista plotter...")
        plotter = pv.Plotter()
        plotter.camera.up = (0, 1, 0)  # Y-up coordinate system

        # Add each subsystem with its offset
        for sub_name in system.get_all_subsystem_names():
            offset_key = f'{sub_name}_offset'
            offset = np.array(system._subsystems.get(offset_key, (0, 0, 0)))
            subsystem = system.get_subsystem(sub_name)

            if subsystem is not None:
                print(f"  Adding {sub_name} mesh at offset {offset}...")
                try:
                    v, i = subsystem.build_mesh()
                    v = v + offset  # Apply position offset
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    color = colors.get(sub_name, '#808080')
                    plotter.add_mesh(mesh, color=color, label=sub_name, opacity=0.85)
                except Exception as e:
                    print(f"    Warning: Failed to add {sub_name}: {e}")

        # Add individual components (silencer, exhaust_stack)
        for comp_name in system.get_all_component_names():
            comp = system.get_component(comp_name)
            if comp is not None:
                print(f"  Adding {comp_name} mesh...")
                try:
                    v, i, _ = comp.generate_mesh()
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    color = colors.get(comp_name, '#808080')
                    plotter.add_mesh(mesh, color=color, label=comp_name, opacity=0.85)
                except Exception as e:
                    print(f"    Warning: Failed to add {comp_name}: {e}")

        # Add duct connections (the key part - render each duct with position offset!)
        if hasattr(system, '_duct_connections') and system._duct_connections:
            print(f"  Adding {len(system._duct_connections)} duct connection sections...")
            for idx, (duct, position) in enumerate(system._duct_connections):
                try:
                    v, i, _ = duct.generate_mesh()
                    v = v + np.array(position)  # Apply position offset
                    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    plotter.add_mesh(mesh, color=colors['ductwork'],
                                    label="Ductwork" if idx == 0 else None, opacity=0.7)
                except Exception as e:
                    print(f"    Warning: Failed to add duct section {idx}: {e}")

        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_title('Core System - 3 Duct Connections')
        plotter.add_axes()
        plotter.add_bounding_box(color='lightgray', opacity=0.1)
        
        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        result = {'success': True, 'message': 'Core system visualized with PyVista'}

    except ImportError:
        # Fallback to basic visualization
        print("PyVista not available, using fallback visualization...")
        viz = GeometryVisualizer()
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=system,
            show=True,
            opacity=0.8,
            show_edges=True,
            show_labels=True,
            show_bounds=True,
            show_axes=True,
            title="Core System - 3 Duct Connections",
            camera_position="iso",
        )
        result = viz.render(request)

    print(f"\nResult: {result['message']}")
    return result


def visualize_pilot_scale():
    """Visualize a pilot-scale system."""
    print("\n" + "=" * 60)
    print("PILOT-SCALE SYSTEM VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_pilot_scale_system

    pilot = create_pilot_scale_system(throughput_kg_h=100)
    pilot.print_summary()

    result = quick_render(pilot, show=True)

    print(f"\nResult: {result['message']}")
    return result


def visualize_production_scale():
    """Visualize a production-scale system."""
    print("\n" + "=" * 60)
    print("PRODUCTION-SCALE SYSTEM VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_production_scale_system

    production = create_production_scale_system(throughput_kg_h=2000)
    production.print_summary()

    result = quick_render(production, show=True)

    print(f"\nResult: {result['message']}")
    return result


def export_all_geometries(output_dir: str = "geometry_exports"):
    """Export all geometries to files."""
    print("\n" + "=" * 60)
    print(f"EXPORTING ALL GEOMETRIES TO: {output_dir}/")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    from airclassifier.geometry.assembly import (
        create_standard_feed_system,
        create_standard_air_system,
        create_pilot_scale_system,
        create_production_scale_system,
        create_core_connections_system,
    )
    from airclassifier.geometry.assembly.classification import create_standard_classification_system
    from airclassifier.geometry.components import (
        CycloneBody, CycloneBodyParams,
        FeedHopper, FeedHopperParams,
    )
    from airclassifier.geometry.components.multi_cyclone import create_protein_separation_cyclones
    from airclassifier.geometry.components.centrifugal_blower import create_standard_centrifugal_blower
    from airclassifier.geometry.components.deagglomerator import create_standard_deagglomerator
    from airclassifier.geometry.components.rotary_airlock import create_standard_rotary_airlock
    from airclassifier.geometry.components.zigzag_classifier import create_standard_zigzag_classifier

    viz = GeometryVisualizer()
    exported_files = []

    # Export individual components
    print("\n[1/10] Exporting Cyclone Body...")
    cyclone = CycloneBody(CycloneBodyParams(
        cylinder_diameter=0.3, cylinder_height=0.3,
        cone_height=0.5, cone_tip_diameter=0.05
    ))
    path = os.path.join(output_dir, "cyclone_body.stl")
    viz.export_to_stl(cyclone, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[2/10] Exporting Multi-Cyclone System...")
    multicyclone = create_protein_separation_cyclones()
    path = os.path.join(output_dir, "multi_cyclone_system.stl")
    viz.export_to_stl(multicyclone, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[3/10] Exporting Centrifugal Blower...")
    blower = create_standard_centrifugal_blower()
    path = os.path.join(output_dir, "centrifugal_blower.stl")
    viz.export_to_stl(blower, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[4/10] Exporting Deagglomerator...")
    deagg = create_standard_deagglomerator()
    path = os.path.join(output_dir, "deagglomerator.stl")
    viz.export_to_stl(deagg, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[5/10] Exporting Feed Hopper...")
    hopper = FeedHopper(FeedHopperParams(
        top_diameter=0.6, bottom_diameter=0.15,
        cylindrical_height=0.3, conical_height=0.5
    ))
    path = os.path.join(output_dir, "feed_hopper.stl")
    viz.export_to_stl(hopper, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[6/10] Exporting Rotary Airlock...")
    airlock = create_standard_rotary_airlock()
    path = os.path.join(output_dir, "rotary_airlock.stl")
    viz.export_to_stl(airlock, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[7/10] Exporting Zigzag Classifier...")
    zigzag = create_standard_zigzag_classifier()
    path = os.path.join(output_dir, "zigzag_classifier.stl")
    viz.export_to_stl(zigzag, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    # Export assemblies
    print("\n[8/10] Exporting Feed System Assembly...")
    feed = create_standard_feed_system()
    path = os.path.join(output_dir, "feed_system_assembly.stl")
    viz.export_to_stl(feed, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[9/10] Exporting Classification System Assembly...")
    classification = create_standard_classification_system()
    path = os.path.join(output_dir, "classification_system_assembly.stl")
    viz.export_to_stl(classification, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    print("\n[10/10] Exporting Core System...")
    core = create_core_connections_system()
    path = os.path.join(output_dir, "core_system.stl")
    viz.export_to_stl(core, path)
    exported_files.append(path)
    print(f"       Saved: {path}")

    # Also save screenshot if PyVista available
    if PYVISTA_AVAILABLE:
        print("\n[Bonus] Saving Core System Screenshot...")
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=core,
            show=False,
            save_path=os.path.join(output_dir, "core_system.png"),
            window_size=(1920, 1080),
        )
        viz.render(request)
        exported_files.append(os.path.join(output_dir, "core_system.png"))
        print(f"       Saved: {os.path.join(output_dir, 'core_system.png')}")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Total files exported: {len(exported_files)}")
    for f in exported_files:
        print(f"  - {f}")

    return exported_files


def interactive_menu():
    """Run interactive menu for visualization selection."""
    check_dependencies()

    while True:
        print("\n" + "=" * 60)
        print("AIR CLASSIFIER GEOMETRY VISUALIZER")
        print("=" * 60)
        print("\nIndividual Components (C=Color, M=Mesh):")
        print("  1C/1M. Cyclone Body")
        print("  2C/2M. Multi-Cyclone System")
        print("  3C/3M. Centrifugal Blower")
        print("  4C/4M. Deagglomerator")
        print("  5C/5M. Feed Hopper")
        print("  6C/6M. Rotary Airlock")
        print("  7C/7M. Zigzag Classifier")
        print("\nAssemblies:")
        print("  F. Feed System Assembly")
        print("  A. Air System Assembly")
        print("  S. Classification System Assembly")
        print("\nComplete Systems:")
        print("  C. Core System (3 Duct Connections)")
        print("  P. Pilot-Scale System (100 kg/h)")
        print("  R. Production-Scale System (2000 kg/h)")
        print("\nOther:")
        print("  E. Export All to Files")
        print("  X. Run All Visualizations")
        print("  0. Exit")
        print()

        try:
            choice = input("Enter choice: ").strip().upper()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if choice == "0":
            print("Goodbye!")
            break
        # Individual components - Color mode
        elif choice == "1C":
            visualize_cyclone(use_mesh=False)
        elif choice == "2C":
            visualize_multicyclone(use_mesh=False)
        elif choice == "3C":
            visualize_blower(use_mesh=False)
        elif choice == "4C":
            visualize_deagglomerator(use_mesh=False)
        elif choice == "5C":
            visualize_hopper(use_mesh=False)
        elif choice == "6C":
            visualize_airlock(use_mesh=False)
        elif choice == "7C":
            visualize_zigzag(use_mesh=False)
        # Individual components - Mesh mode
        elif choice == "1M":
            visualize_cyclone(use_mesh=True)
        elif choice == "2M":
            visualize_multicyclone(use_mesh=True)
        elif choice == "3M":
            visualize_blower(use_mesh=True)
        elif choice == "4M":
            visualize_deagglomerator(use_mesh=True)
        elif choice == "5M":
            visualize_hopper(use_mesh=True)
        elif choice == "6M":
            visualize_airlock(use_mesh=True)
        elif choice == "7M":
            visualize_zigzag(use_mesh=True)
        # Assemblies
        elif choice == "F":
            visualize_feed_system_assembly()
        elif choice == "A":
            visualize_air_system_assembly()
        elif choice == "S":
            visualize_classification_system()
        # Complete systems
        elif choice == "C":
            visualize_core_system()
        elif choice == "P":
            visualize_pilot_scale()
        elif choice == "R":
            visualize_production_scale()
        # Other
        elif choice == "E":
            export_all_geometries()
        elif choice == "X":
            run_all_visualizations()
        else:
            print("Invalid choice. Please try again.")


def run_all_visualizations():
    """Run the 3 system assemblies and the complete core system."""
    print("\nRunning system visualizations...")
    print("  1. Feed System Assembly")
    print("  2. Air System Assembly")
    print("  3. Classification System Assembly")
    print("  4. Complete Core System")
    print()

    visualize_feed_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_air_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_classification_system()
    input("\nPress Enter to continue to next visualization...")

    visualize_core_system()

    print("\n" + "=" * 60)
    print("ALL SYSTEM VISUALIZATIONS COMPLETE")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize air classifier geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu
  python visualize_geometry.py

  # Individual components with rendering mode
  python visualize_geometry.py --cyclone --color
  python visualize_geometry.py --cyclone --mesh
  python visualize_geometry.py --multicyclone --color
  python visualize_geometry.py --blower --mesh
  python visualize_geometry.py --deagglomerator --color
  python visualize_geometry.py --hopper --mesh
  python visualize_geometry.py --airlock --color
  python visualize_geometry.py --zigzag --mesh

  # Assemblies
  python visualize_geometry.py --feed
  python visualize_geometry.py --air
  python visualize_geometry.py --classification

  # Complete systems
  python visualize_geometry.py --core
  python visualize_geometry.py --pilot
  python visualize_geometry.py --production

  # Export and run all
  python visualize_geometry.py --export
  python visualize_geometry.py --all
        """
    )

    # Individual component options
    component_group = parser.add_argument_group('Individual Components')
    component_group.add_argument(
        "--cyclone",
        action="store_true",
        help="Visualize single cyclone body"
    )
    component_group.add_argument(
        "--multicyclone",
        action="store_true",
        help="Visualize multi-cyclone system (3-stage)"
    )
    component_group.add_argument(
        "--blower",
        action="store_true",
        help="Visualize centrifugal blower"
    )
    component_group.add_argument(
        "--deagglomerator",
        action="store_true",
        help="Visualize deagglomerator (pin rotor + screen)"
    )
    component_group.add_argument(
        "--hopper",
        action="store_true",
        help="Visualize feed hopper"
    )
    component_group.add_argument(
        "--airlock",
        action="store_true",
        help="Visualize rotary airlock"
    )
    component_group.add_argument(
        "--zigzag",
        action="store_true",
        help="Visualize zigzag classifier"
    )

    # Rendering mode options
    render_group = parser.add_argument_group('Rendering Mode (for individual components)')
    render_mode = render_group.add_mutually_exclusive_group()
    render_mode.add_argument(
        "--color",
        action="store_true",
        help="Render with solid colors (default)"
    )
    render_mode.add_argument(
        "--mesh",
        action="store_true",
        help="Render as wireframe mesh"
    )

    # Assembly options
    assembly_group = parser.add_argument_group('Assemblies')
    assembly_group.add_argument(
        "--feed", "-f",
        action="store_true",
        help="Visualize feed system assembly"
    )
    assembly_group.add_argument(
        "--air",
        action="store_true",
        help="Visualize air system assembly"
    )
    assembly_group.add_argument(
        "--classification", "-cls",
        action="store_true",
        help="Visualize classification system assembly"
    )

    # System options
    system_group = parser.add_argument_group('Complete Systems')
    system_group.add_argument(
        "--core",
        action="store_true",
        help="Visualize core system with 3 duct connections"
    )
    system_group.add_argument(
        "--pilot",
        action="store_true",
        help="Visualize pilot-scale system (100 kg/h)"
    )
    system_group.add_argument(
        "--production",
        action="store_true",
        help="Visualize production-scale system (2000 kg/h)"
    )

    # Other options
    other_group = parser.add_argument_group('Other')
    other_group.add_argument(
        "--all",
        action="store_true",
        help="Run all visualizations"
    )
    other_group.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export all geometries to STL files"
    )
    other_group.add_argument(
        "--output", "-o",
        default="geometry_exports",
        help="Output directory for exports (default: geometry_exports)"
    )

    args = parser.parse_args()

    # Determine if any visualization was requested
    has_component = any([args.cyclone, args.multicyclone, args.blower,
                        args.deagglomerator, args.hopper, args.airlock, args.zigzag])
    has_assembly = any([args.feed, args.air, args.classification])
    has_system = any([args.core, args.pilot, args.production])
    has_other = any([args.all, args.export])

    # If nothing specified, run interactive menu
    if not any([has_component, has_assembly, has_system, has_other]):
        interactive_menu()
        return

    check_dependencies()

    # Determine mesh mode (default is color)
    use_mesh = args.mesh

    # Individual components
    if args.cyclone:
        visualize_cyclone(use_mesh=use_mesh)

    if args.multicyclone:
        visualize_multicyclone(use_mesh=use_mesh)

    if args.blower:
        visualize_blower(use_mesh=use_mesh)

    if args.deagglomerator:
        visualize_deagglomerator(use_mesh=use_mesh)

    if args.hopper:
        visualize_hopper(use_mesh=use_mesh)

    if args.airlock:
        visualize_airlock(use_mesh=use_mesh)

    if args.zigzag:
        visualize_zigzag(use_mesh=use_mesh)

    # Assemblies
    if args.feed:
        visualize_feed_system_assembly()

    if args.air:
        visualize_air_system_assembly()

    if args.classification:
        visualize_classification_system()

    # Systems
    if args.core:
        visualize_core_system()

    if args.pilot:
        visualize_pilot_scale()

    if args.production:
        visualize_production_scale()

    # Other
    if args.all:
        run_all_visualizations()

    if args.export:
        export_all_geometries(args.output)


if __name__ == "__main__":
    main()
