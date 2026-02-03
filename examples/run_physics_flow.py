#!/usr/bin/env python3
"""
Physics-Based Feed Flow Simulation Example
===========================================

This example demonstrates the new physics-based material flow simulation
that uses actual geometry from the FeedSystemAssembly with proper physics:

- Gravity computed from particle mass and buoyancy
- Drag using Schiller-Naumann correlation
- Inelastic wall collisions with restitution and friction
- Rotational effects from actual RPM values
- Screw conveying at computed axial velocity (pitch × RPM / 60)
- Particle-particle collisions with impulse response

NO magic numbers - everything derived from geometry and physics principles.

Usage:
    python examples/run_physics_flow.py [--particles N] [--time T]
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
        description="Physics-based feed flow simulation"
    )
    parser.add_argument(
        "--particles", "-n", type=int, default=2000,
        help="Number of particles (default: 2000)"
    )
    parser.add_argument(
        "--time", "-t", type=float, default=20.0,
        help="Simulation time in seconds (default: 20)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.005,
        help="Time step in seconds (default: 0.005)"
    )
    parser.add_argument(
        "--airlock-rpm", type=float, default=20,
        help="Airlock RPM (default: 20)"
    )
    parser.add_argument(
        "--feeder-rpm", type=float, default=60,
        help="Screw feeder RPM (default: 60)"
    )
    parser.add_argument(
        "--deagg-rpm", type=float, default=1500,
        help="Deagglomerator RPM (default: 1500)"
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
        "--pouring", "-p", action="store_true",
        help="Enable pouring simulation (lid opens, particles pour in, then flow)"
    )
    parser.add_argument(
        "--fill-percent", type=float, default=50,
        help="Hopper fill percentage (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHYSICS-BASED FEED FLOW SIMULATION")
    print("  Using actual geometry - no magic numbers")
    print("=" * 70)
    
    # Import modules
    from airclassifier.simulation.feed_flow_physics import (
        FeedFlowPhysicsSimulator,
        FlowPhysicsConfig,
        create_physics_flow_simulator,
    )
    from airclassifier.geometry.assembly.feed_system import (
        FeedSystemAssembly,
        FeedSystemParams,
    )
    
    # Create assembly and simulator
    print("\nCreating feed system assembly...")
    params = FeedSystemParams(
        hopper_capacity_kg=500,
        feeder_target_rate_kg_h=500,
    )
    assembly = FeedSystemAssembly(params, device="cpu")
    assembly.build_mesh()
    assembly.print_summary()
    
    # Create physics config
    print("\nConfiguring physics simulation...")
    config = FlowPhysicsConfig(
        dt=args.dt,
        total_time=args.time,
        airlock_rpm=args.airlock_rpm,
        feeder_rpm=args.feeder_rpm,
        deagg_rpm=args.deagg_rpm,
        num_particles=args.particles,
        device=args.device,
        enable_pouring=args.pouring,
        hopper_fill_percentage=args.fill_percent,
    )
    
    # Create simulator
    simulator = FeedFlowPhysicsSimulator(assembly, config)
    
    # Initialize based on mode
    if args.pouring:
        print("\nPouring mode enabled - particles will be poured into hopper")
    else:
        # Initialize particles directly in hopper
        print("\nInitializing particles in hopper...")
        simulator.initialize_particles(
            num_particles=args.particles,
            mean_diameter=0.04,  # 40mm visual particles
            std_diameter=0.005,
        )
    
    # Visualization setup
    plotter = None
    particle_actor = None
    animated_actors = {}
    
    # Helper function to rotate points around an axis
    def rotate_points_around_axis(points, axis_point, axis_dir, angle_rad):
        """Rotate points around an arbitrary axis using Rodrigues' formula."""
        p = points - axis_point
        k = axis_dir / np.linalg.norm(axis_dir)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        k_cross_p = np.cross(k, p)
        k_dot_p = np.dot(p, k)[:, np.newaxis]
        rotated = p * cos_a + k_cross_p * sin_a + k * k_dot_p * (1 - cos_a)
        return rotated + axis_point
    
    if args.visualize and HAS_PYVISTA:
        print("\nSetting up 3D visualization with animated components...")
        pv.set_plot_theme("document")
        plotter = pv.Plotter(title="Physics-Based Feed Flow Simulation")
        plotter.set_background("white")
        plotter.camera.up = (0, 1, 0)
        
        # Component positions from assembly
        hopper_offset = np.array(assembly._hopper_position)
        airlock_offset = np.array(assembly._airlock_position)
        feeder_offset = np.array(assembly._feeder_position)
        deagg_offset = np.array(assembly._deagglomerator_position)
        
        # ============================================
        # HOPPER BODY (STATIC - without lid)
        # ============================================
        print("  Adding hopper body (static)...")
        v_body, i_body, _ = assembly.hopper.get_body_mesh()
        v_body = v_body + hopper_offset
        faces_body = np.hstack([[3] + list(face) for face in i_body.reshape(-1, 3)])
        hopper_mesh = pv.PolyData(v_body, faces_body)
        plotter.add_mesh(hopper_mesh, color='#B8860B', label='Hopper', opacity=0.6)
        
        # ============================================
        # HOPPER LID (ANIMATED)
        # ============================================
        print("  Adding animated hopper lid...")
        v_lid, i_lid, _ = assembly.hopper.get_lid_mesh()
        if len(v_lid) > 0:
            v_lid = v_lid + hopper_offset
            faces_lid = np.hstack([[3] + list(face) for face in i_lid.reshape(-1, 3)])
            lid_mesh = pv.PolyData(v_lid, faces_lid)
            
            hinge_pos = assembly.hopper.get_lid_hinge_position()
            hinge_world = np.array([
                hinge_pos[0] + hopper_offset[0],
                hinge_pos[1] + hopper_offset[1],
                hinge_pos[2] + hopper_offset[2]
            ])
            
            lid_original_points = lid_mesh.points.copy()
            lid_actor = plotter.add_mesh(lid_mesh, color='#D4A574', label='Hopper Lid', opacity=0.95)
            
            animated_actors['lid'] = {
                'mesh': lid_mesh,
                'actor': lid_actor,
                'original_points': lid_original_points,
                'hinge_position': hinge_world,
                'current_angle': 0.0,
            }
        
        # ============================================
        # AIRLOCK (STATIC HOUSING + ANIMATED ROTOR)
        # ============================================
        print("  Adding airlock (static housing + animated rotor)...")
        v_static, i_static, _ = assembly.airlock.get_static_mesh()
        v_static = v_static + airlock_offset
        faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
        airlock_mesh = pv.PolyData(v_static, faces_static)
        plotter.add_mesh(airlock_mesh, color='#5F9EA0', label='Airlock Housing', opacity=0.7)
        
        v_rotor, i_rotor, _ = assembly.airlock.get_rotor_mesh(0)
        v_rotor = v_rotor + airlock_offset
        faces_rotor = np.hstack([[3] + list(face) for face in i_rotor.reshape(-1, 3)])
        airlock_rotor_mesh = pv.PolyData(v_rotor, faces_rotor)
        airlock_rotor_actor = plotter.add_mesh(airlock_rotor_mesh, color='#4A90D9', 
                                               label='Airlock Rotor', opacity=0.9)
        
        animated_actors['airlock_rotor'] = {
            'mesh': airlock_rotor_mesh,
            'actor': airlock_rotor_actor,
            'component': assembly.airlock,
            'position': airlock_offset,
        }
        
        # ============================================
        # SCREW FEEDER (STATIC TROUGH + ANIMATED SCREW)
        # ============================================
        print("  Adding screw feeder (static trough + animated screw)...")
        v_static, i_static, _ = assembly.feeder.get_static_mesh()
        v_static = v_static + feeder_offset
        faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
        feeder_mesh = pv.PolyData(v_static, faces_static)
        plotter.add_mesh(feeder_mesh, color='#8B4513', label='Feeder Trough', opacity=0.7)
        
        v_screw, i_screw, _ = assembly.feeder.get_screw_mesh(0)
        v_screw = v_screw + feeder_offset
        faces_screw = np.hstack([[3] + list(face) for face in i_screw.reshape(-1, 3)])
        feeder_screw_mesh = pv.PolyData(v_screw, faces_screw)
        feeder_screw_actor = plotter.add_mesh(feeder_screw_mesh, color='#2ECC71',
                                              label='Screw', opacity=0.9)
        
        animated_actors['feeder_screw'] = {
            'mesh': feeder_screw_mesh,
            'actor': feeder_screw_actor,
            'component': assembly.feeder,
            'position': feeder_offset,
        }
        
        # ============================================
        # DEAGGLOMERATOR (STATIC HOUSING + ANIMATED ROTOR)
        # ============================================
        print("  Adding deagglomerator (static housing + animated rotor)...")
        v_static, i_static, _ = assembly.deagglomerator.get_static_mesh()
        v_static = v_static + deagg_offset
        faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
        deagg_mesh = pv.PolyData(v_static, faces_static)
        plotter.add_mesh(deagg_mesh, color='#9B59B6', label='Deagg Housing', opacity=0.7)
        
        v_rotor, i_rotor, _ = assembly.deagglomerator.get_rotor_mesh(0)
        v_rotor = v_rotor + deagg_offset
        faces_rotor = np.hstack([[3] + list(face) for face in i_rotor.reshape(-1, 3)])
        deagg_rotor_mesh = pv.PolyData(v_rotor, faces_rotor)
        deagg_rotor_actor = plotter.add_mesh(deagg_rotor_mesh, color='#E74C3C',
                                             label='Deagg Rotor', opacity=0.9)
        
        animated_actors['deagg_rotor'] = {
            'mesh': deagg_rotor_mesh,
            'actor': deagg_rotor_actor,
            'component': assembly.deagglomerator,
            'position': deagg_offset,
        }
        
        # ============================================
        # ADD TRANSITIONS (STATIC)
        # ============================================
        print("  Adding transition connectors...")
        if hasattr(assembly, '_transition_connectors') and assembly._transition_connectors:
            for idx, connector_data in enumerate(assembly._transition_connectors):
                trans = connector_data[0]
                v, i, _ = trans.generate_mesh()
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color='#708090', 
                                label="Transitions" if idx == 0 else None, opacity=0.5)
        
        # ============================================
        # INFO TEXT
        # ============================================
        plotter.add_text(
            "PHYSICS-BASED FEED FLOW\nInitializing...",
            position='upper_left', font_size=10, color='black', name='sim_info'
        )
        
        plotter.add_legend(bcolor='white', face='circle')
        plotter.add_axes()
        plotter.reset_camera()
        plotter.camera.azimuth = -170
        plotter.camera.elevation = -20
        plotter.camera.zoom(0.8)
        plotter.show(interactive_update=True, auto_close=False)
    
    # Run simulation
    print("\n" + "-" * 70)
    print("RUNNING SIMULATION")
    print("-" * 70)
    print(f"  Time: {args.time:.1f} s")
    print(f"  dt:   {args.dt*1000:.1f} ms")
    print(f"  Steps: {int(args.time / args.dt):,}")
    if args.pouring:
        print(f"  Mode: Pouring simulation (lid opens -> pour -> lid closes -> flow)")
    else:
        print(f"  Mode: Direct discharge (particles pre-loaded)")
    print("-" * 70)
    
    total_steps = int(args.time / args.dt)
    print_interval = max(1, total_steps // 20)  # ~20 console updates
    
    # Animation timing - use wall clock for smooth visuals
    target_fps = 30
    frame_interval = 1.0 / target_fps
    steps_per_frame = max(1, int(frame_interval / args.dt))
    
    # Start simulation
    start_time = time.time()
    last_wall_time = time.time()
    last_print_step = -print_interval
    discharge_started = False
    
    if args.pouring:
        # Full workflow: lid opens -> pour -> settle -> discharge
        simulator.start_simulation()
        print("  [Starting pouring sequence]")
    
    step = 0
    while step < total_steps:
        # Run multiple simulation steps per visual frame
        frame_steps = min(steps_per_frame, total_steps - step)
        for _ in range(frame_steps):
            # For non-pouring mode, start discharge after 0.5s
            if not args.pouring and simulator.state.time >= 0.5:
                if not discharge_started:
                    simulator.start_discharge()
                    discharge_started = True
                    print("  [Discharge opened]")
            
            simulator.step()
            step += 1
        
        # Get current state
        counts = simulator.get_zone_counts()
        
        # Console output at intervals
        if step - last_print_step >= print_interval or step >= total_steps:
            last_print_step = step
            progress = 100.0 * step / total_steps
            
            status = f"  [{progress:5.1f}%] t={simulator.state.time:5.2f}s"
            
            if args.pouring:
                phase = simulator.state.phase.value
                lid_angle = simulator.state.lid_angle
                poured = simulator.state.particles_poured
                total = simulator.state.total_particles_to_pour
                status += f" | {phase:9s} lid:{lid_angle:3.0f} poured:{poured:4d}/{total}"
            
            status += f" | H:{counts['hopper']:4d} A:{counts['airlock']:3d} "
            status += f"F:{counts['feeder']:3d} D:{counts['deagg']:4d} "
            status += f"E:{counts['exited']:4d} I:{counts['inactive']:4d}"
            print(status)
        
        # Update visualization every frame
        if plotter is not None:
            # Calculate wall-clock dt for smooth animation
            current_wall_time = time.time()
            wall_dt = current_wall_time - last_wall_time
            last_wall_time = current_wall_time
            wall_dt = min(wall_dt, 0.1)  # Clamp to avoid jumps
            
            # ============================================
            # ANIMATE LID (using simulation state)
            # ============================================
            if 'lid' in animated_actors:
                lid_data = animated_actors['lid']
                current_angle = simulator.state.lid_angle
                
                if abs(current_angle - lid_data['current_angle']) > 0.1:
                    angle_rad = np.radians(current_angle)
                    hinge_pos = lid_data['hinge_position']
                    axis = np.array([0.0, 0.0, 1.0])
                    
                    rotated = rotate_points_around_axis(
                        lid_data['original_points'],
                        hinge_pos,
                        axis,
                        angle_rad
                    )
                    
                    lid_data['mesh'].points[:] = rotated
                    lid_data['mesh'].Modified()
                    lid_data['current_angle'] = current_angle
            
            # ============================================
            # ANIMATE ROTATING COMPONENTS
            # Uses wall_dt for smooth animation
            # Only animate when discharge is open (flowing phase)
            # ============================================
            if simulator._discharge_open == 1:
                # Airlock rotor
                if 'airlock_rotor' in animated_actors:
                    data = animated_actors['airlock_rotor']
                    component = data['component']
                    component.update_rotation(wall_dt, config.airlock_rpm)
                    angle = component.get_rotor_angle()
                    
                    v_rot, _, _ = component.get_rotor_mesh(angle)
                    v_rot = v_rot + data['position']
                    data['mesh'].points[:] = v_rot
                    data['mesh'].Modified()
                
                # Screw feeder
                if 'feeder_screw' in animated_actors:
                    data = animated_actors['feeder_screw']
                    component = data['component']
                    component.update_rotation(wall_dt, config.feeder_rpm)
                    angle = component.get_screw_angle()
                    
                    v_rot, _, _ = component.get_screw_mesh(angle)
                    v_rot = v_rot + data['position']
                    data['mesh'].points[:] = v_rot
                    data['mesh'].Modified()
                
                # Deagglomerator rotor (high speed!)
                if 'deagg_rotor' in animated_actors:
                    data = animated_actors['deagg_rotor']
                    component = data['component']
                    component.update_rotation(wall_dt, config.deagg_rpm)
                    angle = component.get_rotor_angle()
                    
                    v_rot, _, _ = component.get_rotor_mesh(angle)
                    v_rot = v_rot + data['position']
                    data['mesh'].points[:] = v_rot
                    data['mesh'].Modified()
            
            # ============================================
            # UPDATE PARTICLES
            # ============================================
            positions = simulator.get_positions()
            diameters = simulator.get_diameters()
            
            if len(positions) > 0:
                # Remove old actor
                if particle_actor is not None:
                    try:
                        plotter.remove_actor(particle_actor)
                    except:
                        pass
                
                # Create particle point cloud with velocity coloring
                velocities = simulator.state.velocities.numpy()[:len(positions)]
                speeds = np.linalg.norm(velocities, axis=1)
                
                particle_mesh = pv.PolyData(positions)
                particle_mesh['velocity'] = speeds
                
                # Point size based on particle diameter
                particle_dia_mm = diameters.mean() * 1000
                point_size = max(8, min(20, int(particle_dia_mm / 2.5)))
                
                particle_actor = plotter.add_mesh(
                    particle_mesh,
                    scalars='velocity',
                    cmap='YlOrBr',  # Brown/tan for flour-like appearance
                    point_size=point_size,
                    render_points_as_spheres=True,
                    opacity=0.85,
                    clim=[0, 2.0],
                    show_scalar_bar=False,
                )
            
            # ============================================
            # UPDATE INFO TEXT
            # ============================================
            # Get transition zone counts (will be 0 if not in kernel yet)
            t_ha = counts.get('trans_hopper_airlock', 0)
            t_af = counts.get('trans_airlock_feeder', 0)
            t_fd = counts.get('trans_feeder_deagg', 0)
            
            if args.pouring:
                phase_name = simulator.state.phase.value.upper()
                lid_angle = simulator.state.lid_angle
                poured = simulator.state.particles_poured
                total = simulator.state.total_particles_to_pour
                info_text = (
                    f"PHYSICS-BASED FEED FLOW\n"
                    f"Phase: {phase_name}\n"
                    f"Lid: {lid_angle:.0f}\n"
                    f"Poured: {poured:,}/{total:,}\n"
                    f"\n"
                    f"Hopper:  {counts['hopper']:4d}\n"
                    f"  T1:    {t_ha:4d}\n"
                    f"Airlock: {counts['airlock']:4d}\n"
                    f"  T2:    {t_af:4d}\n"
                    f"Feeder:  {counts['feeder']:4d}\n"
                    f"  T3:    {t_fd:4d}\n"
                    f"Deagg:   {counts['deagg']:4d}\n"
                    f"Exited:  {counts['exited']:4d}\n"
                    f"\n"
                    f"t = {simulator.state.time:.2f}s"
                )
            else:
                info_text = (
                    f"PHYSICS-BASED FEED FLOW\n"
                    f"\n"
                    f"Hopper:  {counts['hopper']:4d}\n"
                    f"  T1:    {t_ha:4d}\n"
                    f"Airlock: {counts['airlock']:4d}\n"
                    f"  T2:    {t_af:4d}\n"
                    f"Feeder:  {counts['feeder']:4d}\n"
                    f"  T3:    {t_fd:4d}\n"
                    f"Deagg:   {counts['deagg']:4d}\n"
                    f"Exited:  {counts['exited']:4d}\n"
                    f"\n"
                    f"t = {simulator.state.time:.2f}s"
                )
            
            plotter.add_text(info_text, position='upper_left', font_size=10, 
                            color='black', name='sim_info')
            plotter.update()
            
            # Small delay to control frame rate
            time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print("SIMULATION COMPLETE")
    print("-" * 70)
    print(f"  Wall time: {elapsed:.1f} s")
    print(f"  Sim time:  {simulator.state.time:.2f} s")
    print(f"  Steps:     {simulator.state.step:,}")
    print(f"  Rate:      {simulator.state.step / elapsed:.0f} steps/s")
    
    counts = simulator.get_zone_counts()
    print(f"\nFinal particle distribution:")
    print(f"  Hopper:               {counts['hopper']:5d}")
    print(f"  Trans Hopper->Airlock:{counts.get('trans_hopper_airlock', 0):5d}")
    print(f"  Airlock:              {counts['airlock']:5d}")
    print(f"  Trans Airlock->Feeder:{counts.get('trans_airlock_feeder', 0):5d}")
    print(f"  Feeder:               {counts['feeder']:5d}")
    print(f"  Trans Feeder->Deagg:  {counts.get('trans_feeder_deagg', 0):5d}")
    print(f"  Deagg:                {counts['deagg']:5d}")
    print(f"  Exited:               {counts['exited']:5d}")
    print(f"  Inactive:             {counts['inactive']:5d}")
    print("=" * 70)
    
    if plotter is not None:
        print("\nVisualization window open. Close to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
