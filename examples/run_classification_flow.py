#!/usr/bin/env python3
"""
Physics-Based Classification Flow Simulation Example
=====================================================

This example demonstrates the physics-based particle separation simulation
through the classification system:

- Venturi entrainment (Bernoulli acceleration)
- Zigzag classification (counter-current air separation)
- Multi-stage cyclone separation (centrifugal forces)
- Bag filter collection (inertial impaction)

The simulation tracks particles through the system and shows:
- Real-time particle positions colored by velocity
- Zone transitions as particles flow through components
- Separation statistics (coarse vs. fines vs. protein)

Usage:
    python examples/run_classification_flow.py [--particles N] [--time T]
"""

import sys
import argparse
import time
import numpy as np

# Optional visualization
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Note: Install pyvista for 3D visualization")


def main():
    parser = argparse.ArgumentParser(
        description="Physics-based classification flow simulation"
    )
    parser.add_argument(
        "--particles", "-n", type=int, default=1000,
        help="Number of particles (default: 1000)"
    )
    parser.add_argument(
        "--time", "-t", type=float, default=5.0,
        help="Simulation time in seconds (default: 5)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001,
        help="Time step in seconds (default: 0.001)"
    )
    parser.add_argument(
        "--air-flow", type=float, default=0.3,
        help="Air flow rate in m³/s (default: 0.3)"
    )
    parser.add_argument(
        "--particle-dia", type=float, default=30.0,
        help="Mean particle diameter in microns (default: 30µm)"
    )
    parser.add_argument(
        "--particle-std", type=float, default=15.0,
        help="Particle diameter std dev in microns (default: 15µm)"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Enable 3D visualization (requires pyvista)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)"
    )
    parser.add_argument(
        "--turbulence", type=float, default=0.15,
        help="Turbulent intensity (default: 0.15)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHYSICS-BASED CLASSIFICATION FLOW SIMULATION")
    print("  Protein/Starch Separation via Air Classification")
    print("=" * 70)
    
    # Import modules
    from airclassifier.simulation.classification_flow_physics import (
        ClassificationFlowPhysicsSimulator,
        ClassificationFlowConfig,
    )
    from airclassifier.geometry.assembly.classification import (
        ClassificationSystemAssembly,
    )
    
    # Create assembly
    print("\nCreating classification system assembly...")
    assembly = ClassificationSystemAssembly()
    
    # Print assembly info
    print(f"\n  Components:")
    print(f"    - Venturi eductor (particle entrainment)")
    print(f"    - Zigzag classifier (primary separation)")
    print(f"    - Multi-cyclone system (staged separation)")
    print(f"    - Bag filter (final collection)")
    
    # Create physics config
    print("\nConfiguring physics simulation...")
    config = ClassificationFlowConfig(
        dt=args.dt,
        air_flow_rate_m3s=args.air_flow,
        num_particles=args.particles,
        device=args.device,
        turbulent_intensity=args.turbulence,
    )
    
    # Create simulator
    simulator = ClassificationFlowPhysicsSimulator(assembly, config)
    
    # Initialize particles
    print("\nInitializing particles at venturi inlet...")
    mean_dia_m = args.particle_dia * 1e-6
    std_dia_m = args.particle_std * 1e-6
    simulator.initialize_particles(
        num_particles=args.particles,
        mean_diameter=mean_dia_m,
        diameter_std=std_dia_m,
    )
    
    # Visualization setup
    plotter = None
    particle_actor = None
    
    if args.visualize and HAS_PYVISTA:
        print("\nSetting up 3D visualization...")
        pv.set_plot_theme("document")
        plotter = pv.Plotter(title="Classification Flow - Protein Separation")
        plotter.set_background("white")
        plotter.camera.up = (0, 1, 0)
        
        # ============================================
        # CREATE SIMPLIFIED GEOMETRY VISUALIZATION
        # ============================================
        
        # Get component positions from simulator
        venturi_center = simulator.venturi_center
        zigzag_center = simulator.zigzag_center
        cyclone_primary_center = simulator.cyclone_primary_center
        cyclone_secondary_center = simulator.cyclone_secondary_center
        cyclone_tertiary_center = simulator.cyclone_tertiary_center
        bagfilter_center = simulator.bagfilter_center
        
        # ============================================
        # VENTURI (simplified as cylinder + cone)
        # ============================================
        print("  Adding venturi eductor...")
        venturi = pv.Cylinder(
            center=venturi_center,
            direction=(0, 1, 0),
            radius=simulator.venturi_inlet_diameter / 2,
            height=simulator.venturi_total_length,
        )
        plotter.add_mesh(venturi, color='#3498DB', opacity=0.3, label='Venturi')
        
        # ============================================
        # ZIGZAG (simplified as box)
        # ============================================
        print("  Adding zigzag classifier...")
        zigzag_box = pv.Box(bounds=[
            zigzag_center[0] - simulator.zigzag_channel_width / 2,
            zigzag_center[0] + simulator.zigzag_channel_width / 2,
            zigzag_center[1] - simulator.zigzag_total_height / 2,
            zigzag_center[1] + simulator.zigzag_total_height / 2,
            zigzag_center[2] - simulator.zigzag_channel_depth / 2,
            zigzag_center[2] + simulator.zigzag_channel_depth / 2,
        ])
        plotter.add_mesh(zigzag_box, color='#2ECC71', opacity=0.3, label='Zigzag')
        
        # Add zigzag stage lines
        for i in range(int(simulator.zigzag_num_stages) + 1):
            y = simulator.zigzag_inlet_y + i * simulator.zigzag_stage_height
            line = pv.Line(
                [zigzag_center[0] - simulator.zigzag_channel_width / 2, y, zigzag_center[2]],
                [zigzag_center[0] + simulator.zigzag_channel_width / 2, y, zigzag_center[2]],
            )
            plotter.add_mesh(line, color='#27AE60', line_width=1)
        
        # ============================================
        # CYCLONES (simplified as cylinders + cones)
        # ============================================
        print("  Adding cyclones...")
        
        # Primary cyclone
        cy1_cyl = pv.Cylinder(
            center=cyclone_primary_center + np.array([0, -simulator.cyclone_primary_cylinder_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_primary_radius,
            height=simulator.cyclone_primary_cylinder_height,
        )
        cy1_cone = pv.Cone(
            center=cyclone_primary_center + np.array([0, -simulator.cyclone_primary_cylinder_height - simulator.cyclone_primary_cone_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_primary_radius,
            height=simulator.cyclone_primary_cone_height,
        )
        plotter.add_mesh(cy1_cyl, color='#E74C3C', opacity=0.3, label='Cyclone 1')
        plotter.add_mesh(cy1_cone, color='#E74C3C', opacity=0.3)
        
        # Secondary cyclone
        cy2_cyl = pv.Cylinder(
            center=cyclone_secondary_center + np.array([0, -simulator.cyclone_secondary_cylinder_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_secondary_radius,
            height=simulator.cyclone_secondary_cylinder_height,
        )
        cy2_cone = pv.Cone(
            center=cyclone_secondary_center + np.array([0, -simulator.cyclone_secondary_cylinder_height - simulator.cyclone_secondary_cone_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_secondary_radius,
            height=simulator.cyclone_secondary_cone_height,
        )
        plotter.add_mesh(cy2_cyl, color='#E67E22', opacity=0.3, label='Cyclone 2')
        plotter.add_mesh(cy2_cone, color='#E67E22', opacity=0.3)
        
        # Tertiary cyclone
        cy3_cyl = pv.Cylinder(
            center=cyclone_tertiary_center + np.array([0, -simulator.cyclone_tertiary_cylinder_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_tertiary_radius,
            height=simulator.cyclone_tertiary_cylinder_height,
        )
        cy3_cone = pv.Cone(
            center=cyclone_tertiary_center + np.array([0, -simulator.cyclone_tertiary_cylinder_height - simulator.cyclone_tertiary_cone_height / 2, 0]),
            direction=(0, 1, 0),
            radius=simulator.cyclone_tertiary_radius,
            height=simulator.cyclone_tertiary_cone_height,
        )
        plotter.add_mesh(cy3_cyl, color='#9B59B6', opacity=0.3, label='Cyclone 3')
        plotter.add_mesh(cy3_cone, color='#9B59B6', opacity=0.3)
        
        # ============================================
        # BAG FILTER (simplified as box)
        # ============================================
        print("  Adding bag filter...")
        bagfilter_box = pv.Box(bounds=[
            bagfilter_center[0] - simulator.bagfilter_half_width,
            bagfilter_center[0] + simulator.bagfilter_half_width,
            bagfilter_center[1] - simulator.bagfilter_height / 2,
            bagfilter_center[1] + simulator.bagfilter_height / 2,
            bagfilter_center[2] - simulator.bagfilter_half_depth,
            bagfilter_center[2] + simulator.bagfilter_half_depth,
        ])
        plotter.add_mesh(bagfilter_box, color='#95A5A6', opacity=0.3, label='Bag Filter')
        
        # ============================================
        # CONNECTING DUCTS (simplified as lines)
        # ============================================
        print("  Adding ducts...")
        
        # Venturi to zigzag
        duct1 = pv.Line(simulator.duct_venturi_zigzag_start, simulator.duct_venturi_zigzag_end)
        plotter.add_mesh(duct1, color='#7F8C8D', line_width=3)
        
        # Zigzag to cyclone
        duct2 = pv.Line(simulator.duct_zigzag_cyclone_start, simulator.duct_zigzag_cyclone_end)
        plotter.add_mesh(duct2, color='#7F8C8D', line_width=3)
        
        # Cyclone to bag filter
        duct3 = pv.Line(simulator.duct_cyclone_bag_start, simulator.duct_cyclone_bag_end)
        plotter.add_mesh(duct3, color='#7F8C8D', line_width=3)
        
        # ============================================
        # LABELS FOR OUTLETS
        # ============================================
        # Coarse outlet
        coarse_pos = np.array([zigzag_center[0], simulator.zigzag_coarse_outlet_y - 0.1, zigzag_center[2]])
        plotter.add_point_labels([coarse_pos], ["COARSE\n(Starch)"], font_size=12, 
                                 text_color='#8B4513', point_size=0)
        
        # Cyclone dust outlets
        dust1_pos = np.array([cyclone_primary_center[0], simulator.cyclone_primary_dust_y - 0.1, cyclone_primary_center[2]])
        plotter.add_point_labels([dust1_pos], ["Cy1 Dust"], font_size=10,
                                 text_color='#E74C3C', point_size=0)
        
        dust3_pos = np.array([cyclone_tertiary_center[0], simulator.cyclone_tertiary_dust_y - 0.1, cyclone_tertiary_center[2]])
        plotter.add_point_labels([dust3_pos], ["PROTEIN"], font_size=12,
                                 text_color='#9B59B6', point_size=0)
        
        # ============================================
        # INFO TEXT
        # ============================================
        plotter.add_text(
            "CLASSIFICATION FLOW\nInitializing...",
            position='upper_left', font_size=10, color='black', name='sim_info'
        )
        
        # ============================================
        # SEPARATION STATS TEXT
        # ============================================
        plotter.add_text(
            "Separation:\n---",
            position='upper_right', font_size=10, color='black', name='sep_info'
        )
        
        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_axes()
        plotter.reset_camera()
        plotter.camera.azimuth = -30
        plotter.camera.elevation = 20
        plotter.camera.zoom(0.8)
        plotter.show(interactive_update=True, auto_close=False)
    
    # Run simulation
    print("\n" + "-" * 70)
    print("RUNNING SIMULATION")
    print("-" * 70)
    print(f"  Time: {args.time:.1f} s")
    print(f"  dt:   {args.dt*1000:.2f} ms")
    print(f"  Steps: {int(args.time / args.dt):,}")
    print(f"  Air flow: {args.air_flow * 3600:.0f} m³/h")
    print(f"  Zigzag d50: {simulator.zigzag_d50 * 1e6:.1f} µm")
    print("-" * 70)
    
    total_steps = int(args.time / args.dt)
    print_interval = max(1, total_steps // 20)  # ~20 console updates
    
    # Animation timing
    target_fps = 30
    frame_interval = 1.0 / target_fps
    steps_per_frame = max(1, int(frame_interval / args.dt))
    
    # Start simulation
    start_time = time.time()
    last_wall_time = time.time()
    last_print_step = -print_interval
    
    step = 0
    while step < total_steps:
        # Run multiple simulation steps per visual frame
        frame_steps = min(steps_per_frame, total_steps - step)
        for _ in range(frame_steps):
            simulator.step()
            step += 1
        
        # Get current state
        zone_counts = simulator.get_zone_counts()
        sep_counts = simulator.get_separation_counts()
        
        # Console output at intervals
        if step - last_print_step >= print_interval or step >= total_steps:
            last_print_step = step
            progress = 100.0 * step / total_steps
            
            active = sep_counts['active']
            coarse = sep_counts['coarse']
            cy1 = sep_counts['cyclone_1']
            cy2 = sep_counts['cyclone_2']
            cy3 = sep_counts['cyclone_3_protein']
            bag = sep_counts['bagfilter']
            
            status = f"  [{progress:5.1f}%] t={simulator.state.time:5.2f}s"
            status += f" | Active:{active:4d} Coarse:{coarse:3d}"
            status += f" Cy1:{cy1:3d} Cy2:{cy2:3d} Cy3:{cy3:3d} Bag:{bag:3d}"
            print(status)
        
        # Update visualization every frame
        if plotter is not None:
            # Calculate wall-clock dt
            current_wall_time = time.time()
            wall_dt = current_wall_time - last_wall_time
            last_wall_time = current_wall_time
            
            # ============================================
            # UPDATE PARTICLES
            # ============================================
            positions = simulator.get_positions()
            velocities = simulator.get_velocities()
            diameters = simulator.get_diameters()
            zones = simulator.get_zones()
            
            if len(positions) > 0:
                # Filter finite positions
                finite = np.isfinite(positions).all(axis=1)
                if np.any(finite):
                    positions = np.asarray(positions[finite], dtype=np.float64)
                    velocities = np.asarray(velocities[finite], dtype=np.float64)
                    diameters = np.asarray(diameters[finite], dtype=np.float64)
                    zones = zones[finite]
                    n_show = len(positions)
                else:
                    n_show = 0
                
                if n_show > 0:
                    # Create particle point cloud
                    speeds = np.linalg.norm(velocities, axis=1)
                    
                    particle_mesh = pv.PolyData(positions)
                    particle_mesh['velocity'] = speeds
                    
                    # Point size based on particle diameter (scale up for visibility)
                    particle_dia_um = float(np.mean(diameters)) * 1e6
                    point_size = max(6, min(15, int(particle_dia_um / 3)))
                    
                    try:
                        new_actor = plotter.add_mesh(
                            particle_mesh,
                            scalars='velocity',
                            cmap='plasma',  # Velocity coloring
                            point_size=point_size,
                            render_points_as_spheres=True,
                            opacity=0.9,
                            clim=[0, 20.0],
                            show_scalar_bar=True,
                            scalar_bar_args={'title': 'Velocity (m/s)', 'n_labels': 3},
                        )
                        if particle_actor is not None:
                            try:
                                plotter.remove_actor(particle_actor)
                            except Exception:
                                pass
                        particle_actor = new_actor
                    except Exception as e:
                        if "plane" not in str(e).lower():
                            raise
                        pass  # Keep previous frame
            
            # ============================================
            # UPDATE INFO TEXT
            # ============================================
            info_text = (
                f"CLASSIFICATION FLOW\n"
                f"Protein Separation\n"
                f"\n"
                f"Venturi:   {zone_counts.get('venturi', 0):4d}\n"
                f"Zigzag:    {zone_counts.get('zigzag', 0):4d}\n"
                f"Cyclone 1: {zone_counts.get('cyclone_1', 0):4d}\n"
                f"Cyclone 2: {zone_counts.get('cyclone_2', 0):4d}\n"
                f"Cyclone 3: {zone_counts.get('cyclone_3', 0):4d}\n"
                f"Bag Filt:  {zone_counts.get('bagfilter', 0):4d}\n"
                f"\n"
                f"t = {simulator.state.time:.2f}s"
            )
            plotter.add_text(info_text, position='upper_left', font_size=10,
                            color='black', name='sim_info')
            
            # ============================================
            # UPDATE SEPARATION STATS
            # ============================================
            total_collected = (sep_counts['coarse'] + sep_counts['cyclone_1'] + 
                             sep_counts['cyclone_2'] + sep_counts['cyclone_3_protein'] +
                             sep_counts['bagfilter'])
            
            if total_collected > 0:
                pct_coarse = 100 * sep_counts['coarse'] / total_collected
                pct_cy1 = 100 * sep_counts['cyclone_1'] / total_collected
                pct_cy2 = 100 * sep_counts['cyclone_2'] / total_collected
                pct_cy3 = 100 * sep_counts['cyclone_3_protein'] / total_collected
                pct_bag = 100 * sep_counts['bagfilter'] / total_collected
            else:
                pct_coarse = pct_cy1 = pct_cy2 = pct_cy3 = pct_bag = 0
            
            sep_text = (
                f"SEPARATION\n"
                f"----------\n"
                f"Coarse: {sep_counts['coarse']:4d} ({pct_coarse:4.1f}%)\n"
                f"Cy1:    {sep_counts['cyclone_1']:4d} ({pct_cy1:4.1f}%)\n"
                f"Cy2:    {sep_counts['cyclone_2']:4d} ({pct_cy2:4.1f}%)\n"
                f"Cy3:    {sep_counts['cyclone_3_protein']:4d} ({pct_cy3:4.1f}%)\n"
                f"Bag:    {sep_counts['bagfilter']:4d} ({pct_bag:4.1f}%)\n"
                f"----------\n"
                f"Protein: Cy3+Bag\n"
                f"Starch:  Coarse"
            )
            plotter.add_text(sep_text, position='upper_right', font_size=10,
                            color='black', name='sep_info')
            
            plotter.update()
            time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print("SIMULATION COMPLETE")
    print("-" * 70)
    print(f"  Wall time: {elapsed:.1f} s")
    print(f"  Sim time:  {simulator.state.time:.2f} s")
    print(f"  Steps:     {simulator.state.step:,}")
    print(f"  Rate:      {simulator.state.step / elapsed:.0f} steps/s")
    
    # Print final separation summary
    simulator.print_separation_summary()
    
    print("=" * 70)
    
    if plotter is not None:
        print("\nVisualization window open. Close to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
