#!/usr/bin/env python
"""
Geometry Visualization Example Script

This script demonstrates how to visualize air classifier geometries:
- Individual components with --color (color-coded) or --mesh (wireframe) modes
- Assembled systems (feed, air, classification with or without preclassification)

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
    python examples/visualize_geometry.py --wheel --color      # Wheel classifier

Assembly Modes:
    python examples/visualize_geometry.py --feed       # Feed system assembly
    python examples/visualize_geometry.py --air        # Air system assembly
    python examples/visualize_geometry.py --with-preclassification   # Classification with venturi, zigzag
    python examples/visualize_geometry.py --without-preclassification # Wheel-only (no venturi/zigzag)
    python examples/visualize_geometry.py --without-preclassification --animate-wheel  # Wheel-only with rotating wheel
    python examples/visualize_geometry.py --wheel --animate  # Standalone wheel rotating at params.omega (physics-coupled)
    python examples/visualize_geometry.py --wheel --animate --color  # Standalone wheel with color

Complete core system:
    python examples/visualize_geometry.py --core-with-preclassification   # Full core (venturi, zigzag, ductwork)
    python examples/visualize_geometry.py --core-without-preclassification  # Core wheel-only (ductwork)

Other:
    python examples/visualize_geometry.py --export     # Export all to STL files
    python examples/visualize_geometry.py --all       # Feed + air + classification (with preclassification)

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
    'wheel': '#FF6B6B',         # Coral Red (wheel classifier)
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


def visualize_wheel_classifier(use_mesh: bool = False, animate: bool = False):
    """
    Visualize a centrifugal wheel classifier using the actual geometry mesh.

    The wheel classifier uses centrifugal force (1000-7000g) for fine separation:
    - Feed enters tangentially or axially
    - Spinning wheel creates high centrifugal acceleration
    - Fine particles (low inertia) pass through blade gaps -> fines outlet
    - Coarse particles (high inertia) rejected -> coarse outlet
    - Achieves d50 = 20-25 um (vs zigzag's 30-150 um limit)

    When animate=True, the wheel and motor rotate at params.omega (rad/s), i.e.
    the same angular velocity used by the physics kernel (omega = 2*pi*rpm/60).
    """
    print("\n" + "=" * 60)
    print("WHEEL CLASSIFIER VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.components.wheel_classifier import (
        create_standard_wheel_classifier
    )
    import numpy as np
    import time

    # Create classifier using factory function
    classifier = create_standard_wheel_classifier()

    print("Centrifugal Wheel Classifier:")
    print(f"  Wheel diameter: {classifier.params.wheel_diameter * 1000:.0f}mm")
    print(f"  Wheel width: {classifier.params.wheel_width * 1000:.0f}mm")
    print(f"  Number of blades: {classifier.params.num_blades}")
    print(f"  RPM: {classifier.params.rpm:.0f}")
    print(f"\nCentrifugal Force:")
    print(f"  Tip speed: {classifier.params.tip_speed:.1f} m/s")
    print(f"  G-force at rim: {classifier.params.g_force:.0f} g")
    print(f"\nSeparation Physics:")
    print(f"  Target d50: {classifier.params.target_d50 * 1e6:.1f} um")
    print(f"  Blade gap: {classifier.params.blade_gap * 1000:.2f}mm")
    print(f"  Volute outer radius: {classifier.params.volute_outer_radius * 1000:.1f}mm")
    print(f"\nPorts:")
    print(f"  Inlet: {classifier.params.feed_inlet_width*1000:.0f}x{classifier.params.feed_inlet_height*1000:.0f}mm (rectangular)")
    print(f"  Fines outlet: D={classifier.params.fines_outlet_diameter*1000:.0f}mm (+Y)")
    print(f"  Coarse outlet: D={classifier.params.coarse_outlet_diameter*1000:.0f}mm (-Y)")

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
        # Vertices from wheel classifier are flat (x,y,z,x,y,z,...); reshape to (N, 3)
        pts = np.asarray(vertices).reshape(-1, 3)
        faces = np.hstack([[3] + list(face) for face in indices.reshape(-1, 3)])
        mesh = pv.PolyData(pts, faces)

        if use_mesh:
            # Wireframe mesh mode
            plotter.add_mesh(mesh, style='wireframe', color='black',
                           line_width=1, label='Wheel Classifier')
            plotter.add_title('Wheel Classifier - Mesh View (Actual Geometry)')
        elif not animate:
            # Solid color mode (no animation)
            plotter.add_mesh(mesh, color=COMPONENT_COLORS['wheel'], opacity=0.85,
                           show_edges=True, edge_color='gray',
                           label='Wheel Classifier')
            plotter.add_title('Wheel Classifier - Actual Geometry (Centrifugal)')

        # Add flow direction arrows (use feed_angular_position so label is on feed, not motor)
        p = classifier.params
        housing_r = p.volute_outer_radius
        theta = p.feed_angular_position  # PI = -X side (feed); motor is on +Y (top)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Feed opening (flange) is at volute_outer_radius + feed_inlet_length, at angle theta
        feed_r = housing_r + p.feed_inlet_length
        feed_x = cos_t * feed_r
        feed_z = sin_t * feed_r
        # Arrow at feed opening, pointing INTO classifier (toward center)
        arrow_start = np.array([[feed_x - cos_t * 0.03, 0, feed_z - sin_t * 0.03]])
        arrow_dir = np.array([[cos_t * 0.1, 0, sin_t * 0.1]])  # toward center
        plotter.add_arrows(arrow_start, arrow_dir, color='blue', mag=1.0)

        # Fines flow up (green arrow) - through hub
        fines_y = p.wheel_width / 2 + p.fines_outlet_length + 0.02
        arrow_start2 = np.array([[0, fines_y, 0]])
        arrow_dir2 = np.array([[0, 0.08, 0]])
        plotter.add_arrows(arrow_start2, arrow_dir2, color='green', mag=1.0)

        # Coarse flow down (orange arrow) - through hopper
        coarse_y = -p.wheel_width / 2 - p.coarse_hopper_height - 0.02
        arrow_start3 = np.array([[0, coarse_y, 0]])
        arrow_dir3 = np.array([[0, -0.08, 0]])
        plotter.add_arrows(arrow_start3, arrow_dir3, color='orange', mag=1.0)

        # Feed label at feed inlet (same side as feed, not on motor)
        label_offset = 0.06
        plotter.add_point_labels(
            [[feed_x - cos_t * label_offset, 0, feed_z - sin_t * label_offset]],
            ['Feed\n(tangential)'],
            font_size=10, text_color='blue', shape_opacity=0
        )
        plotter.add_point_labels(
            [[0.08, fines_y + 0.05, 0]],
            ['Fines (Protein)\n(low inertia)'],
            font_size=10, text_color='green', shape_opacity=0
        )
        plotter.add_point_labels(
            [[0.08, coarse_y - 0.05, 0]],
            ['Coarse (Starch)\n(high inertia)'],
            font_size=10, text_color='orange', shape_opacity=0
        )

        plotter.add_axes()
        plotter.add_legend(bcolor='white', face='circle')

        # Reset camera to fit entire scene and set isometric view
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20

        if animate:
            # Physics-coupled animation: omega from component (same as kernel uses)
            omega = classifier.params.omega  # rad/s = 2*pi*rpm/60 (WheelClassifierParams.omega)
            wheel_center = np.array([0.0, 0.0, 0.0])  # local center (component origin)
            base_mesh = mesh.copy(deep=True)
            start_time = time.time()
            wheel_actor = [None]  # use list so callback can rebind

            def update_rotation(step):
                t = time.time() - start_time
                angle_deg = np.degrees(omega * t)
                mesh_rot = base_mesh.copy(deep=True).rotate_y(angle_deg, point=wheel_center, inplace=False)
                if wheel_actor[0] is not None:
                    plotter.remove_actor(wheel_actor[0])
                wheel_actor[0] = plotter.add_mesh(mesh_rot, color=COMPONENT_COLORS['wheel'], opacity=0.85,
                                                  show_edges=True, edge_color='gray', label='Wheel Classifier')

            wheel_actor[0] = plotter.add_mesh(base_mesh, color=COMPONENT_COLORS['wheel'], opacity=0.85,
                                             show_edges=True, edge_color='gray', label='Wheel Classifier')
            # Timer (not render callback) to avoid re-entrant loop when remove_actor triggers render
            plotter.add_timer_event(max_steps=10**9, duration=33, callback=update_rotation)
            plotter.add_title(f'Wheel Classifier - {classifier.params.rpm:.0f} RPM (omega={omega:.1f} rad/s, physics-coupled)')
            print("\nOpening visualization window (wheel + motor rotating at kernel omega)...")
        else:
            print("\nOpening visualization window...")
        print("(Close the window to continue)")
        plotter.show(interactive=True)

        result = {'success': True, 'message': 'Wheel classifier visualized with actual geometry' + (' (animated)' if animate else '')}
    else:
        # Fallback to basic visualization
        result = render_component(classifier, "Wheel Classifier",
                                 COMPONENT_COLORS['wheel'], use_mesh=use_mesh)

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


def visualize_classification_system(wheel_only: bool = False, animate_wheel: bool = False):
    """Visualize the classification system assembly with color-coded components.

    When wheel_only=True, uses assembly without preclassification (no venturi, zigzag, dropout):
    air inlet + 15° solids chute -> wheel -> cyclones -> bag filter.
    When animate_wheel=True, the wheel and motor rotate at the assembly wheel's params.omega
    (same angular velocity as the physics kernel).
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)

    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly,
        ClassificationSystemParams,
        create_standard_classification_system,
    )
    import numpy as np
    import time

    try:
        if wheel_only:
            print("Creating classification system (wheel-only, no preclassification)...")
            params = ClassificationSystemParams(use_preclassification=False)
            cls = ClassificationSystemAssembly(params=params)
        else:
            print("Creating classification system...")
            cls = create_standard_classification_system()
        print("Building mesh...")
        vertices, indices = cls.build_mesh()

        print("Classification System Assembly includes:")
        if cls.venturi is not None:
            print("  - Venturi Eductor (particle entrainment)")
        if cls.zigzag is not None:
            print("  - Zigzag Classifier (primary separation)")
        if wheel_only:
            print("  - Air inlet + 15° solids chute -> wheel inlet")
        print("  - Wheel Classifier (centrifugal fine cut)")
        print("  - Multi-Cyclone System (staged collection)")
        print("  - Bag Filter (fine particle capture)")
        print("  - Connecting Ductwork")
        if getattr(cls, '_bypass_split_tee', None):
            print("  - Bypass Duct (flow split/merge around classifier)")
        # Check for dropout hopper in duct sections
        for duct, _ in cls._duct_sections:
            if type(duct).__name__ == 'ExpandingTransitionWithDropout':
                print("  - Coarse Dropout Hopper (pre-separation)")
                break
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
                'duct': '#95A5A6',         # Gray
                'dropout': '#8B4513',      # Saddle Brown (dropout hopper)
                'tee': '#1ABC9C',          # Teal (tee junctions)
                'airlock': '#CD853F',      # Peru/tan (rotary airlocks)
                'wheel': '#FF6B6B',        # Coral Red (wheel classifier)
            }

            # Add Venturi (only when present)
            if cls.venturi is not None:
                print("  Adding Venturi mesh...")
                v, i, _ = cls.venturi.generate_mesh()
                v = v + np.array(cls._component_positions['venturi'])
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=colors['venturi'],
                                label='Venturi Eductor', opacity=0.85)

            # Add Zigzag (only when present)
            if cls.zigzag is not None:
                print("  Adding Zigzag mesh...")
                v, i, _ = cls.zigzag.generate_mesh()
                v = v + np.array(cls._component_positions['zigzag'])
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

            # Add Wheel Classifier (always present) with optional physics-coupled animation
            wheel_classifier = cls.get_component('wheel_classifier')
            if wheel_classifier is not None:
                print("  Adding Wheel Classifier mesh" + (" (animated)" if animate_wheel else "") + "...")
                v, i, _ = wheel_classifier.generate_mesh()
                wheel_pos = np.array(cls._component_positions['wheel_classifier'])
                v_wheel = v + wheel_pos
                faces_wheel = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                wheel_mesh_base = pv.PolyData(v_wheel, faces_wheel)
                if animate_wheel:
                    omega = wheel_classifier.params.omega  # rad/s, same as kernel
                    wheel_center_world = wheel_pos.copy()
                    start_time = time.time()
                    wheel_actor_ref = [None]

                    def update_wheel_rotation(step):
                        t = time.time() - start_time
                        angle_deg = np.degrees(omega * t)
                        mesh_rot = wheel_mesh_base.copy(deep=True).rotate_y(angle_deg, point=wheel_center_world, inplace=False)
                        if wheel_actor_ref[0] is not None:
                            plotter.remove_actor(wheel_actor_ref[0])
                        wheel_actor_ref[0] = plotter.add_mesh(mesh_rot, color=colors['wheel'],
                                                             label='Wheel Classifier', opacity=0.85)

                    wheel_actor_ref[0] = plotter.add_mesh(wheel_mesh_base, color=colors['wheel'],
                                                         label='Wheel Classifier', opacity=0.85)
                    # Timer (not render callback) to avoid re-entrant loop when remove_actor triggers render
                    plotter.add_timer_event(max_steps=10**9, duration=33, callback=update_wheel_rotation)
                else:
                    plotter.add_mesh(wheel_mesh_base, color=colors['wheel'],
                                    label='Wheel Classifier', opacity=0.85)

            # Add Ducts (new format: list of (duct_component, position) tuples)
            # Color-code by component type: dropout hopper, tee junctions, regular duct
            all_duct_sections = list(cls._duct_sections)
            if hasattr(cls, '_collection_duct_sections'):
                all_duct_sections.extend(cls._collection_duct_sections)
            print(f"  Adding {len(all_duct_sections)} duct sections ({len(cls._duct_sections)} main + {len(getattr(cls, '_collection_duct_sections', []))} collection)...")
            label_added = {'duct': False, 'dropout': False, 'tee': False, 'three_pt': False, 'airlock': False}
            for idx, (duct, position) in enumerate(all_duct_sections):
                v, i, _ = duct.generate_mesh()
                v = v + np.array(position)  # Apply position offset
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)

                type_name = type(duct).__name__
                if type_name == 'ExpandingTransitionWithDropout':
                    color = colors['dropout']
                    label = "Dropout Hopper" if not label_added['dropout'] else None
                    label_added['dropout'] = True
                    opacity = 0.85
                elif type_name == 'TeeJunction':
                    color = colors['tee']
                    label = "Tee Junction" if not label_added['tee'] else None
                    label_added['tee'] = True
                    opacity = 0.85
                elif type_name == 'ThreePointJunction':
                    color = colors['tee']
                    label = "Three-Point Connector (Y, 15° Z, X)" if not label_added['three_pt'] else None
                    label_added['three_pt'] = True
                    opacity = 0.85
                elif type_name == 'RotaryAirlock':
                    color = colors['airlock']
                    label = "Rotary Airlock" if not label_added['airlock'] else None
                    label_added['airlock'] = True
                    opacity = 0.85
                else:
                    color = colors['duct']
                    label = "Ductwork" if not label_added['duct'] else None
                    label_added['duct'] = True
                    opacity = 0.7

                plotter.add_mesh(mesh, color=color, label=label, opacity=opacity)

            plotter.add_legend(bcolor='white', face='circle')
            title = 'Classification System - Port-Based Assembly'
            if wheel_only:
                title += ' (Wheel-Only)'
            if animate_wheel:
                title += ' - Wheel Animated (physics-coupled)'
            plotter.add_title(title)
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


def visualize_core_system(use_preclassification: bool = True):
    """
    Visualize the complete core system with all three duct connections.

    Args:
        use_preclassification: If True, classification has venturi + zigzag;
            if False, wheel-only (air + 15° solids -> three-point junction -> wheel).

    Shows the main flow path with ductwork connections:
    1. Air System -> Venturi (or wheel-only air inlet)
    2. Feed System -> Venturi solids_inlet (or wheel-only solids chute)
    3. Bag Filter -> Exhaust (Silencer)
    """
    mode = "with preclassification (venturi, zigzag)" if use_preclassification else "without preclassification (wheel-only)"
    print("\n" + "=" * 60)
    print("CORE SYSTEM VISUALIZATION")
    print("=" * 60)
    print(f"Mode: {mode}")
    print("Focus: 3 Main Ductwork Connections")
    if use_preclassification:
        print("  1. Air System -> Venturi (air_inlet)")
        print("  2. Feed System -> Venturi (solids_inlet)")
    else:
        print("  1. Air System -> Wheel-only junction (Y leg)")
        print("  2. Feed System -> Wheel-only junction (15° solids chute)")
    print("  3. Bag Filter -> Exhaust (Silencer)")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_core_connections_system
    import numpy as np

    system = create_core_connections_system(use_preclassification=use_preclassification)

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
            first_elbow_labeled = False
            for idx, (duct, position) in enumerate(system._duct_connections):
                try:
                    v, i, _ = duct.generate_mesh()
                    v = np.asarray(v).reshape(-1, 3) + np.array(position)
                    faces = np.hstack([[3] + list(face) for face in np.asarray(i).reshape(-1, 3)])
                    mesh = pv.PolyData(v, faces)
                    comp_type = type(duct).__name__
                    label = None
                    if idx == 0:
                        label = "Ductwork"
                    if comp_type == 'DuctElbow' and not first_elbow_labeled:
                        label = "Elbow (damper 2 → vertical duct)"
                        first_elbow_labeled = True
                    plotter.add_mesh(mesh, color=colors['ductwork'], label=label, opacity=0.7)
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
        print("  8C/8M. Wheel Classifier (centrifugal)")
        print("\nAssemblies:")
        print("  F. Feed System Assembly")
        print("  A. Air System Assembly")
        print("  S. Classification (with preclassification: venturi, zigzag)")
        print("  W. Classification (without preclassification: wheel-only)")
        print("\nComplete Core System:")
        print("  C. Core (with preclassification)")
        print("  K. Core (without preclassification, wheel-only)")
        print("\nOther:")
        print("  E. Export All to Files")
        print("  X. Run All Visualizations (feed + air + classification with preclassification)")
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
        elif choice == "8C":
            visualize_wheel_classifier(use_mesh=False)
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
        elif choice == "8M":
            visualize_wheel_classifier(use_mesh=True)
        # Assemblies
        elif choice == "F":
            visualize_feed_system_assembly()
        elif choice == "A":
            visualize_air_system_assembly()
        elif choice == "S":
            visualize_classification_system(wheel_only=False)
        elif choice == "W":
            visualize_classification_system(wheel_only=True)
        elif choice == "C":
            visualize_core_system(use_preclassification=True)
        elif choice == "K":
            visualize_core_system(use_preclassification=False)
        # Other
        elif choice == "E":
            export_all_geometries()
        elif choice == "X":
            run_all_visualizations()
        else:
            print("Invalid choice. Please try again.")


def run_all_visualizations():
    """Run the three system assemblies (feed, air, classification with preclassification)."""
    print("\nRunning system visualizations...")
    print("  1. Feed System Assembly")
    print("  2. Air System Assembly")
    print("  3. Classification (with preclassification)")
    print()

    visualize_feed_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_air_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_classification_system(wheel_only=False)

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
  python visualize_geometry.py --with-preclassification
  python visualize_geometry.py --without-preclassification
  python visualize_geometry.py --without-preclassification --animate-wheel

  # Complete core system (feed + air + classification + ductwork + exhaust)
  python visualize_geometry.py --core-with-preclassification
  python visualize_geometry.py --core-without-preclassification

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
    component_group.add_argument(
        "--wheel",
        action="store_true",
        help="Visualize centrifugal wheel classifier"
    )
    component_group.add_argument(
        "--animate",
        action="store_true",
        help="With --wheel: animate wheel and motor at params.omega (same as physics kernel)"
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
    classification_mode = assembly_group.add_mutually_exclusive_group()
    classification_mode.add_argument(
        "--with-preclassification",
        action="store_true",
        help="Visualize classification system with preclassification (venturi, zigzag, dropout)"
    )
    classification_mode.add_argument(
        "--without-preclassification",
        action="store_true",
        help="Visualize classification system without preclassification (wheel-only: 15° solids chute + air inlet)"
    )
    assembly_group.add_argument(
        "--animate-wheel",
        action="store_true",
        help="With classification: animate wheel and motor at assembly wheel RPM (physics-coupled)"
    )

    # Complete core system (feed + air + classification + ductwork + exhaust)
    core_group = parser.add_argument_group('Complete Core System')
    core_mode = core_group.add_mutually_exclusive_group()
    core_mode.add_argument(
        "--core-with-preclassification",
        action="store_true",
        help="Visualize complete core system with preclassification (venturi, zigzag, ductwork)"
    )
    core_mode.add_argument(
        "--core-without-preclassification",
        action="store_true",
        help="Visualize complete core system without preclassification (wheel-only, ductwork)"
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
                        args.deagglomerator, args.hopper, args.airlock, args.zigzag,
                        args.wheel])
    has_assembly = any([args.feed, args.air, args.with_preclassification, args.without_preclassification])
    has_core = any([args.core_with_preclassification, args.core_without_preclassification])
    has_other = any([args.all, args.export])

    # If nothing specified, run interactive menu
    if not any([has_component, has_assembly, has_core, has_other]):
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

    if args.wheel:
        visualize_wheel_classifier(use_mesh=use_mesh, animate=getattr(args, 'animate', False))

    # Assemblies
    if args.feed:
        visualize_feed_system_assembly()

    if args.air:
        visualize_air_system_assembly()

    if args.with_preclassification:
        visualize_classification_system(wheel_only=False, animate_wheel=args.animate_wheel)

    if args.without_preclassification:
        visualize_classification_system(wheel_only=True, animate_wheel=args.animate_wheel)

    # Complete core system
    if args.core_with_preclassification:
        visualize_core_system(use_preclassification=True)
    if args.core_without_preclassification:
        visualize_core_system(use_preclassification=False)

    # Other
    if args.all:
        run_all_visualizations()

    if args.export:
        export_all_geometries(args.output)


if __name__ == "__main__":
    main()
