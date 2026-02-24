"""
Visualize Hammer Mill Geometry
==============================

Simple script to visualize the hammer mill geometry using PyVista.
Run this to see the 3D model of the mill with color-coded components.

Usage:
    python examples/visualize_hammer_mill.py
    python examples/visualize_hammer_mill.py --static
    python examples/visualize_hammer_mill.py --dark
"""

import sys
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("PyVista not installed. Install with: pip install pyvista")
    exit(1)

from airclassifier.milling import (
    create_hammer_mill_machine,
    MillConfig,
    COMPONENT_COLORS,
)


# Enhanced color scheme with labels for legend
COMPONENT_STYLE = {
    "rotor": {
        "color": (0.6, 0.6, 0.65),
        "opacity": 0.95,
        "label": "Rotor (steel)",
    },
    "hammers": {
        "color": (0.85, 0.70, 0.15),  # Bright gold
        "opacity": 1.0,
        "label": "Hammers (brass)",
    },
    "hammer_pins": {
        "color": (0.55, 0.55, 0.58),
        "opacity": 1.0,
        "label": "Hammer pins (steel)",
    },
    "screen": {
        "color": (0.4, 0.45, 0.5),
        "opacity": 0.6,
        "label": "Screen (perforated)",
    },
    "housing": {
        "color": (0.35, 0.45, 0.55),
        "opacity": 0.3,  # Very transparent to see inside
        "label": "Housing (casing)",
    },
    "feed_chute": {
        "color": (0.5, 0.6, 0.65),
        "opacity": 0.75,
        "label": "Feed Chute",
    },
    "drive": {
        "color": (0.25, 0.3, 0.35),
        "opacity": 0.9,
        "label": "Drive Assembly",
    },
    # Drive assembly parts (industrial look like centrifugal_blower: gray motor, visible belt)
    "drive_motor": {
        "color": (0.32, 0.34, 0.38),
        "opacity": 0.95,
        "label": "Motor (body)",
    },
    "drive_base": {
        "color": (0.5, 0.52, 0.55),
        "opacity": 0.95,
        "label": "Base plate",
    },
    "drive_feet": {
        "color": (0.45, 0.47, 0.5),
        "opacity": 0.95,
        "label": "Motor feet",
    },
    "drive_pulley_motor": {
        "color": (0.28, 0.30, 0.32),
        "opacity": 1.0,
        "label": "Motor pulley",
    },
    "drive_pulley_mill": {
        "color": (0.62, 0.63, 0.66),
        "opacity": 1.0,
        "label": "Mill pulley",
    },
    "drive_shaft": {
        "color": (0.35, 0.37, 0.4),
        "opacity": 1.0,
        "label": "Motor shaft",
    },
    "drive_belt": {
        "color": (0.22, 0.22, 0.24),
        "opacity": 1.0,
        "label": "Belt",
    },
}


def visualize_hammer_mill(config: MillConfig = None, animate: bool = True, dark: bool = False):
    """Visualize the hammer mill in 3D with color-coded components.

    Args:
        config: Mill configuration (uses defaults if None)
        animate: If True, animate the rotor rotation
        dark: If True, use dark background theme
    """
    # Create the mill assembly
    print("Building hammer mill geometry...")
    assembly = create_hammer_mill_machine(config=config, resolution=32)
    meshes = assembly.get_component_meshes()

    # Create PyVista plotter with larger window
    plotter = pv.Plotter(window_size=(1400, 900))
    bg_color = "#1a1a2e" if dark else "white"
    text_color = "white" if dark else "black"
    plotter.set_background(bg_color)

    # Add title
    plotter.add_text(
        "Hammer Mill - ProteinProcessIO",
        position="upper_left",
        font_size=12,
        color=text_color,
    )

    # Store actors for animation
    actors = {}
    mesh_data = {}

    # Add each component mesh with proper styling
    print("\nComponents:")
    for name, (verts, tris, meta) in meshes.items():
        # Create PyVista mesh
        faces = np.hstack([
            np.full((len(tris), 1), 3, dtype=np.int32),
            tris
        ]).ravel()
        mesh = pv.PolyData(verts, faces)

        # Get style from enhanced scheme or fallback to COMPONENT_COLORS
        style = COMPONENT_STYLE.get(name, {})
        if style:
            rgb = style["color"]
            opacity = style["opacity"]
            label = style["label"]
        else:
            color = COMPONENT_COLORS.get(name, (0.5, 0.5, 0.5, 1.0))
            rgb = color[:3]
            opacity = color[3] if len(color) > 3 else 0.8
            label = name.replace("_", " ").title()

        # Add to plotter
        actor = plotter.add_mesh(
            mesh,
            color=rgb,
            opacity=opacity,
            smooth_shading=True,
            name=name,
            label=label,
        )

        actors[name] = actor
        mesh_data[name] = {
            "mesh": mesh,
            "verts": verts.copy(),
            "meta": meta,
        }

        print(f"  {name:15} {len(verts):5} verts, {len(tris):5} tris, "
              f"color={rgb}, opacity={opacity:.2f}")

    # Add legend
    legend_bg = (0.1, 0.1, 0.15, 0.8) if dark else "white"
    plotter.add_legend(loc="upper right", bcolor=legend_bg)

    # Add axes
    plotter.add_axes(
        xlabel="X (rotor axis)",
        ylabel="Y (vertical)",
        zlabel="Z (lateral)",
        line_width=2,
    )

    # Frame the entire system (hammer mill + drive), not just the mill center
    plotter.reset_camera()
    # Y is vertical: set camera up vector so Y points up on screen (not Z)
    plotter.camera.up = (0, 1, 0)
    # Zoom out so the whole assembly fits with comfortable margin
    plotter.camera.zoom(1.4)

    print("\nControls: Left-click drag to rotate, scroll to zoom, close window to exit")

    if animate:
        # Animation parameters
        omega = config.rotor_angular_velocity if config else 314.0  # ~3000 rpm
        dt = 0.033  # ~30 fps
        theta = [0.0]

        def rotate_callback(*args):
            theta[0] += omega * dt * 0.05  # Slow down for visualization

            # Rotate rotor and hammers around X axis
            cos_t = np.cos(theta[0])
            sin_t = np.sin(theta[0])

            for name in ["rotor", "hammers", "hammer_pins"]:
                if name in mesh_data:
                    info = mesh_data[name]
                    verts = info["verts"]

                    new_verts = verts.copy()
                    new_verts[:, 1] = cos_t * verts[:, 1] - sin_t * verts[:, 2]
                    new_verts[:, 2] = sin_t * verts[:, 1] + cos_t * verts[:, 2]

                    info["mesh"].points = new_verts

        # Try different animation methods based on PyVista version
        try:
            plotter.add_on_render_callback(rotate_callback)
            print("Animation: Rotor spinning (move mouse to trigger updates)\n")
        except AttributeError:
            try:
                plotter.iren.add_observer("TimerEvent", rotate_callback)
                plotter.iren.create_repeating_timer(33)
                print("Animation: Rotor spinning continuously\n")
            except Exception:
                print("Animation not supported in this PyVista version. Showing static view.\n")
                animate = False

    # Show with interactive window
    plotter.show()


if __name__ == "__main__":
    # Parse command line flags
    animate = "--static" not in sys.argv
    dark = "--dark" in sys.argv

    print("=" * 60)
    print("  Hammer Mill Geometry Visualization")
    print("=" * 60)

    if not animate:
        print("(Static mode - animation disabled)")
    if dark:
        print("(Dark theme)")

    # Use corrected geometry defaults
    config = MillConfig(
        rotor_rpm=3000,
        rotor_diameter_m=0.20,      # Hub diameter (radius = 0.10m)
        rotor_length_m=0.30,
        hammer_rows=4,
        hammers_per_row=4,
        hammer_length_m=0.08,       # Tip radius = 0.10 + 0.08 = 0.18m
        screen_inner_radius_m=0.188,
        housing_inner_radius_m=0.20,
        screen_aperture_mm=1.5,
    )

    print(f"\nConfiguration:")
    print(f"  Rotor RPM:         {config.rotor_rpm}")
    print(f"  Rotor diameter:    {config.rotor_diameter_m * 100:.0f} cm")
    print(f"  Hammer tip radius: {config.rotor_diameter_m/2 + config.hammer_length_m:.3f} m")
    print(f"  Hammer tip speed:  {config.hammer_tip_speed:.1f} m/s")
    print(f"  Screen aperture:   {config.screen_aperture_mm} mm")
    print(f"  Total hammers:     {config.total_hammers}")

    visualize_hammer_mill(config, animate=animate, dark=dark)
