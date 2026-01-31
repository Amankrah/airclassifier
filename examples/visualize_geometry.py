#!/usr/bin/env python
"""
Geometry Visualization Example Script

This script demonstrates how to visualize and inspect air classifier geometries:
- Individual components
- Assembled systems
- Complete classifier system

Run modes:
    python examples/visualize_geometry.py                    # Interactive menu
    python examples/visualize_geometry.py --component        # Single component
    python examples/visualize_geometry.py --assembly         # Assembled system
    python examples/visualize_geometry.py --complete         # Complete system
    python examples/visualize_geometry.py --all              # All visualizations
    python examples/visualize_geometry.py --export           # Export all to files

    # Single component
    python examples/visualize_geometry.py --component

    # Feed system assembly
    python examples/visualize_geometry.py --assembly

    # Complete classifier system
    python examples/visualize_geometry.py --complete

    # Pilot-scale system  
    python examples/visualize_geometry.py --pilot

    # Production-scale system
    python examples/visualize_geometry.py --production

    # Export all geometries to STL files
    python examples/visualize_geometry.py --export

    # Export to custom directory
    python examples/visualize_geometry.py --export -o ./my_exports

    # Run all visualizations sequentially
    python examples/visualize_geometry.py --all

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
    from airclassifier.geometry.components import (
        CycloneBody, CycloneBodyParams,
        FeedHopper, FeedHopperParams,
    )
    
    viz = GeometryVisualizer()
    exported_files = []
    
    # Export cyclone body
    print("\n[1/6] Exporting Cyclone Body...")
    cyclone = CycloneBody(CycloneBodyParams(
        cylinder_diameter=0.3, cylinder_height=0.3,
        cone_height=0.5, cone_tip_diameter=0.05
    ))
    path = os.path.join(output_dir, "cyclone_body.stl")
    viz.export_to_stl(cyclone, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export feed hopper
    print("\n[2/6] Exporting Feed Hopper...")
    hopper = FeedHopper(FeedHopperParams(
        top_diameter=0.6, bottom_diameter=0.15,
        cylindrical_height=0.3, conical_height=0.5
    ))
    path = os.path.join(output_dir, "feed_hopper.stl")
    viz.export_to_stl(hopper, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export feed system
    print("\n[3/6] Exporting Feed System Assembly...")
    feed = create_standard_feed_system()
    path = os.path.join(output_dir, "feed_system_assembly.stl")
    viz.export_to_stl(feed, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export air system
    print("\n[4/6] Exporting Air System Assembly...")
    air = create_standard_air_system()
    path = os.path.join(output_dir, "air_system_assembly.stl")
    viz.export_to_stl(air, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export pilot scale
    print("\n[5/6] Exporting Pilot-Scale System...")
    pilot = create_pilot_scale_system()
    path = os.path.join(output_dir, "pilot_scale_system.stl")
    viz.export_to_stl(pilot, path)
    exported_files.append(path)
    print(f"       Saved: {path}")
    
    # Export complete system
    print("\n[6/6] Exporting Complete Production System...")
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
        print("  5. Complete System (Standard 500 kg/h)")
        print("  6. Pilot-Scale System (100 kg/h)")
        print("  7. Production-Scale System (2000 kg/h)")
        print("  8. Export All to Files")
        print("  9. Run All Visualizations")
        print("  0. Exit")
        print()
        
        try:
            choice = input("Enter choice (0-9): ").strip()
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
            visualize_complete_system()
        elif choice == "6":
            visualize_pilot_scale()
        elif choice == "7":
            visualize_production_scale()
        elif choice == "8":
            export_all_geometries()
        elif choice == "9":
            run_all_visualizations()
        else:
            print("Invalid choice. Please enter 0-9.")


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
    
    visualize_complete_system()
    
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
  python visualize_geometry.py                 # Interactive menu
  python visualize_geometry.py --component     # Single component
  python visualize_geometry.py --assembly      # Feed system assembly
  python visualize_geometry.py --complete      # Complete system
  python visualize_geometry.py --all           # All visualizations
  python visualize_geometry.py --export        # Export all to STL files
  python visualize_geometry.py --export -o ./my_exports  # Custom output dir
        """
    )
    
    parser.add_argument(
        "--component", "-c",
        action="store_true",
        help="Visualize single component (cyclone body)"
    )
    parser.add_argument(
        "--assembly", "-a",
        action="store_true",
        help="Visualize feed system assembly"
    )
    parser.add_argument(
        "--air",
        action="store_true",
        help="Visualize air system assembly"
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
    
    args = parser.parse_args()
    
    # If no specific option, run interactive menu
    if not any([args.component, args.assembly, args.air, args.complete, 
                args.all, args.export, args.pilot, args.production]):
        interactive_menu()
        return
    
    check_dependencies()
    
    if args.component:
        visualize_single_component()
    
    if args.assembly:
        visualize_feed_system_assembly()
    
    if args.air:
        visualize_air_system_assembly()
    
    if args.complete:
        visualize_complete_system()
    
    if args.pilot:
        visualize_pilot_scale()
    
    if args.production:
        visualize_production_scale()
    
    if args.all:
        run_all_visualizations()
    
    if args.export:
        export_all_geometries(args.output)


if __name__ == "__main__":
    main()
