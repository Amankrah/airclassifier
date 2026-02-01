#!/usr/bin/env python
"""
Geometry Visualization Example Script

This script demonstrates how to visualize air classifier geometries:
- Individual components (cyclone, hopper, etc.)
- Assembled systems (feed, air, classification)
- Complete classifier system with duct connections

Run modes:
    python examples/visualize_geometry.py              # Interactive menu
    python examples/visualize_geometry.py --component  # Single component
    python examples/visualize_geometry.py --feed       # Feed system assembly
    python examples/visualize_geometry.py --air        # Air system assembly
    python examples/visualize_geometry.py --complete   # Complete system
    python examples/visualize_geometry.py --core       # Core with 3 duct connections
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


def visualize_single_component():
    """Visualize a single geometry component."""
    print("\n" + "=" * 60)
    print("SINGLE COMPONENT VISUALIZATION")
    print("=" * 60)
    
    from airclassifier.geometry.components import (
        CycloneBody,
        CycloneBodyParams,
    )
    
    # Create a cyclone body component
    params = CycloneBodyParams(
        cylinder_diameter=0.3,
        cylinder_height=0.3,
        cone_height=0.5,
        cone_tip_diameter=0.05,
    )
    cyclone = CycloneBody(params)
    
    print(f"Component: CycloneBody")
    print(f"  Cylinder: D={params.cylinder_diameter*1000:.0f}mm, H={params.cylinder_height*1000:.0f}mm")
    print(f"  Cone: H={params.cone_height*1000:.0f}mm, tip D={params.cone_tip_diameter*1000:.0f}mm")
    print(f"  Vertices: {len(cyclone.vertices):,}")
    print(f"  Triangles: {len(cyclone.indices)//3:,}")
    
    # Visualize
    viz = GeometryVisualizer()
    result = viz.visualize_component(
        cyclone,
        name="Cyclone Body",
        show=True,
        opacity=0.8,
        color="#4A90D9",
        title="Cyclone Body Component"
    )
    
    print(f"\nResult: {result['message']}")
    return result


def visualize_feed_hopper():
    """Visualize a feed hopper component."""
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
    
    print(f"Component: FeedHopper")
    print(f"  Top D={params.top_diameter*1000:.0f}mm")
    print(f"  Bottom D={params.bottom_diameter*1000:.0f}mm")
    print(f"  Total height={params.total_height*1000:.0f}mm")
    print(f"  Vertices: {len(hopper.vertices):,}")
    
    viz = GeometryVisualizer()
    result = viz.visualize_component(
        hopper,
        name="Feed Hopper",
        show=True,
        opacity=0.7,
        color="#F0AD4E",
        title="Feed Hopper Component"
    )
    
    print(f"\nResult: {result['message']}")
    return result


def visualize_feed_system_assembly():
    """Visualize the feed system assembly."""
    print("\n" + "=" * 60)
    print("FEED SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)
    
    from airclassifier.geometry.assembly import create_standard_feed_system
    
    feed = create_standard_feed_system()
    vertices, indices = feed.build_mesh()
    
    print("Feed System Assembly includes:")
    print("  - Feed Hopper")
    print("  - Rotary Airlock")
    print("  - Screw Feeder")
    print("  - De-agglomerator")
    print(f"\nTotal mesh: {len(vertices):,} vertices, {len(indices)//3:,} triangles")
    
    # Print summary
    feed.print_summary()
    
    viz = GeometryVisualizer()
    request = VisualizationRequest(
        target_type="assembly",
        assembly=feed,
        show=True,
        opacity=0.8,
        show_edges=True,
        title="Feed System Assembly",
        show_labels=True,
    )
    result = viz.render(request)
    
    print(f"\nResult: {result['message']}")
    return result


def visualize_air_system_assembly():
    """Visualize the air system assembly."""
    print("\n" + "=" * 60)
    print("AIR SYSTEM ASSEMBLY VISUALIZATION")
    print("=" * 60)
    
    from airclassifier.geometry.assembly import create_standard_air_system
    
    air = create_standard_air_system()
    vertices, indices = air.build_mesh()
    
    print("Air System Assembly includes:")
    print("  - Inlet Air Filter")
    print("  - Centrifugal Blower")
    print("  - Flow Damper")
    print(f"\nTotal mesh: {len(vertices):,} vertices, {len(indices)//3:,} triangles")
    
    air.print_summary()
    
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
    from airclassifier.geometry.components.ductwork import RoundDuct
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


def visualize_complete_system():
    """Visualize the complete classifier system."""
    print("\n" + "=" * 60)
    print("COMPLETE CLASSIFIER SYSTEM VISUALIZATION")
    print("=" * 60)
    
    from airclassifier.geometry.assembly import create_complete_classifier_system
    
    system = create_complete_classifier_system(
        throughput_kg_h=500,
        cut_size_um=20
    )
    
    # Print comprehensive summary
    system.print_summary()
    print()
    system.print_bill_of_materials()
    
    # Visualize
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
        title="Complete Air Classifier System (500 kg/h)",
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


def visualize_core_connections():
    """
    Visualize the core system with all three duct connections.

    Shows the main flow path with ductwork connections:
    1. Air System -> Venturi air_inlet
    2. Feed System -> Venturi solids_inlet
    3. Bag Filter -> Exhaust (Silencer)
    """
    print("\n" + "=" * 60)
    print("CORE CONNECTIONS VISUALIZATION")
    print("=" * 60)
    print("Focus: 3 Main Ductwork Connections")
    print("  1. Air System -> Venturi (air_inlet)")
    print("  2. Feed System -> Venturi (solids_inlet)")
    print("  3. Bag Filter -> Exhaust (Silencer)")
    print("=" * 60)

    from airclassifier.geometry.assembly import create_core_connections_system

    system = create_core_connections_system()

    # Print summary
    system.print_summary()
    print()
    system.print_bill_of_materials()

    # Use PyVista directly for better control over rendering (like classification viz)
    try:
        import pyvista as pv
        import numpy as np

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


def export_all_geometries(output_dir: str = "geometry_exports"):
    """Export all geometries to files."""
    print("\n" + "=" * 60)
    print(f"EXPORTING ALL GEOMETRIES TO: {output_dir}/")
    print("=" * 60)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    from airclassifier.geometry.assembly import (
        create_standard_feed_system,
        create_standard_air_system,
        create_complete_classifier_system,
        create_pilot_scale_system,
        create_production_scale_system,
    )
    from airclassifier.geometry.assembly.classification import create_standard_classification_system
    from airclassifier.geometry.components import (
        CycloneBody, CycloneBodyParams,
        FeedHopper, FeedHopperParams,
    )
    
    viz = GeometryVisualizer()
    exported_files = []
    
    # Export cyclone body
    print("\n[1/7] Exporting Cyclone Body...")
    cyclone = CycloneBody(CycloneBodyParams(
        cylinder_diameter=0.3, cylinder_height=0.3,
        cone_height=0.5, cone_tip_diameter=0.05
    ))
    path = os.path.join(output_dir, "cyclone_body.stl")
    viz.export_to_stl(cyclone, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export feed hopper
    print("\n[2/7] Exporting Feed Hopper...")
    hopper = FeedHopper(FeedHopperParams(
        top_diameter=0.6, bottom_diameter=0.15,
        cylindrical_height=0.3, conical_height=0.5
    ))
    path = os.path.join(output_dir, "feed_hopper.stl")
    viz.export_to_stl(hopper, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export feed system
    print("\n[3/7] Exporting Feed System Assembly...")
    feed = create_standard_feed_system()
    path = os.path.join(output_dir, "feed_system_assembly.stl")
    viz.export_to_stl(feed, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export air system
    print("\n[4/7] Exporting Air System Assembly...")
    air = create_standard_air_system()
    path = os.path.join(output_dir, "air_system_assembly.stl")
    viz.export_to_stl(air, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export classification system
    print("\n[5/7] Exporting Classification System Assembly...")
    classification = create_standard_classification_system()
    path = os.path.join(output_dir, "classification_system_assembly.stl")
    viz.export_to_stl(classification, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export pilot scale
    print("\n[6/7] Exporting Pilot-Scale System...")
    pilot = create_pilot_scale_system()
    path = os.path.join(output_dir, "pilot_scale_system.stl")
    viz.export_to_stl(pilot, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export complete system
    print("\n[7/7] Exporting Complete Production System...")
    complete = create_complete_classifier_system()
    path = os.path.join(output_dir, "complete_system.stl")
    viz.export_to_stl(complete, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Also save screenshot if PyVista available
    if PYVISTA_AVAILABLE:
        print("\n[Bonus] Saving System Screenshot...")
        request = VisualizationRequest(
            target_type="complete_system",
            complete_system=complete,
            show=False,
            save_path=os.path.join(output_dir, "complete_system.png"),
            window_size=(1920, 1080),
        )
        viz.render(request)
        exported_files.append(os.path.join(output_dir, "complete_system.png"))
        print(f"       Saved: {os.path.join(output_dir, 'complete_system.png')}")
    
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
        print("\nSelect visualization:")
        print("  1. Single Component - Cyclone Body")
        print("  2. Single Component - Feed Hopper")
        print("  3. Assembly - Feed System")
        print("  4. Assembly - Air System")
        print("  5. Assembly - Classification System")
        print("  6. Complete System (Standard 500 kg/h)")
        print("  7. Core Connections (3 Duct Connections)")
        print("  8. Pilot-Scale System (100 kg/h)")
        print("  9. Production-Scale System (2000 kg/h)")
        print("  E. Export All to Files")
        print("  A. Run All Visualizations")
        print("  0. Exit")
        print()

        try:
            choice = input("Enter choice (0-9, E, A): ").strip().upper()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            visualize_single_component()
        elif choice == "2":
            visualize_feed_hopper()
        elif choice == "3":
            visualize_feed_system_assembly()
        elif choice == "4":
            visualize_air_system_assembly()
        elif choice == "5":
            visualize_classification_system()
        elif choice == "6":
            visualize_complete_system()
        elif choice == "7":
            visualize_core_connections()
        elif choice == "8":
            visualize_pilot_scale()
        elif choice == "9":
            visualize_production_scale()
        elif choice == "E":
            export_all_geometries()
        elif choice == "A":
            run_all_visualizations()
        else:
            print("Invalid choice. Please enter 0-9, E, or A.")


def run_all_visualizations():
    """Run all visualizations in sequence."""
    print("\nRunning all visualizations...")

    visualize_single_component()
    input("\nPress Enter to continue to next visualization...")

    visualize_feed_hopper()
    input("\nPress Enter to continue to next visualization...")

    visualize_feed_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_air_system_assembly()
    input("\nPress Enter to continue to next visualization...")

    visualize_classification_system()
    input("\nPress Enter to continue to next visualization...")

    visualize_complete_system()
    input("\nPress Enter to continue to next visualization...")

    visualize_core_connections()

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize air classifier geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_geometry.py              # Interactive menu
  python visualize_geometry.py --component  # Single component
  python visualize_geometry.py --feed       # Feed system assembly
  python visualize_geometry.py --complete   # Complete system
  python visualize_geometry.py --core       # Core with 3 duct connections
  python visualize_geometry.py --export     # Export all to STL files
        """
    )
    
    parser.add_argument(
        "--component", "-c",
        action="store_true",
        help="Visualize single component (cyclone body)"
    )
    parser.add_argument(
        "--feed", "-f",
        action="store_true",
        help="Visualize feed system assembly"
    )
    parser.add_argument(
        "--air",
        action="store_true",
        help="Visualize air system assembly"
    )
    parser.add_argument(
        "--classification", "-cls",
        action="store_true",
        help="Visualize classification system assembly"
    )
    parser.add_argument(
        "--complete", "-s",
        action="store_true",
        help="Visualize complete classifier system"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all visualizations"
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export all geometries to STL files"
    )
    parser.add_argument(
        "--output", "-o",
        default="geometry_exports",
        help="Output directory for exports (default: geometry_exports)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Visualize pilot-scale system"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Visualize production-scale system"
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Visualize core system with 3 duct connections"
    )
    
    args = parser.parse_args()
    
    # If no specific option, run interactive menu
    if not any([args.component, args.feed, args.air, args.classification,
                args.complete, args.all, args.export, args.pilot, args.production, args.core]):
        interactive_menu()
        return
    
    check_dependencies()
    
    if args.component:
        visualize_single_component()
    
    if args.feed:
        visualize_feed_system_assembly()
    
    if args.air:
        visualize_air_system_assembly()
    
    if args.classification:
        visualize_classification_system()
    
    if args.complete:
        visualize_complete_system()
    
    if args.pilot:
        visualize_pilot_scale()
    
    if args.production:
        visualize_production_scale()
    
    if args.core:
        visualize_core_connections()
    
    if args.all:
        run_all_visualizations()
    
    if args.export:
        export_all_geometries(args.output)


if __name__ == "__main__":
    main()
