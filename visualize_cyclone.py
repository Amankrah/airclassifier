"""
Visualize the complete cyclone air classifier geometry.

This script uses the built-in visualization module to render
the cyclone assembly and export files for viewing.
"""

import sys
import warp as wp

from airclassifier.geometry.assembly import CycloneAssembly, CycloneGeometryParams
from airclassifier.visualization import render_cyclone_assembly


def main():
    """Main visualization function."""
    print("=" * 60)
    print("Cyclone Air Classifier Geometry Visualization")
    print("=" * 60)
    
    # Initialize Warp
    wp.init()
    print(f"Warp device: {wp.get_device()}")
    
    # Create cyclone assembly with 300mm diameter
    print("\nCreating cyclone geometry...")
    diameter = 0.3  # 300mm
    params = CycloneGeometryParams.from_diameter(diameter)
    assembly = CycloneAssembly(params, device="cpu")
    
    # Print summary
    assembly.print_summary()
    
    # Check if interactive display is requested
    show_interactive = "--show" in sys.argv
    
    # Render assembly using built-in visualization module
    print("\nRendering cyclone assembly...")
    results = render_cyclone_assembly(
        assembly,
        save_prefix="cyclone",
        show=show_interactive,
        export_stl=True
    )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nOutput files:")
    for f in results['files']:
        print(f"  - {f}")
    
    if not show_interactive:
        print("\nRun with --show to display interactive plots")
        print("Or open the STL file in Windows 3D Viewer / MeshLab")


if __name__ == "__main__":
    main()
