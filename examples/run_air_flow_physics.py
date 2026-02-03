#!/usr/bin/env python3
"""
SPH Air Flow Simulation Example
================================

Demonstrates SPH (Smoothed Particle Hydrodynamics) air flow simulation:
- Air represented as SPH particles with density and pressure
- Pressure-driven flow from SPH equations of motion
- Centrifugal acceleration in blower impeller region
- Blower startup/shutdown with fan affinity laws
- Boundary containment within actual duct geometry

Usage:
    python examples/run_air_flow_physics.py --time 10
    python examples/run_air_flow_physics.py --time 15 --rpm 2500 --particles 2000 --visualize
"""

import argparse
import time
import numpy as np

# Check for visualization support
HAS_PYVISTA = False
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physics-based air flow simulation"
    )
    parser.add_argument(
        "--time", "-t", type=float, default=10.0,
        help="Simulation time in seconds (default: 10)"
    )
    parser.add_argument(
        "--dt", type=float, default=0.001,
        help="Time step in seconds (default: 0.001)"
    )
    parser.add_argument(
        "--rpm", type=float, default=3000.0,
        help="Target blower RPM (default: 3000)"
    )
    parser.add_argument(
        "--flow-rate", type=float, default=3000.0,
        help="Design flow rate in m³/h (default: 3000)"
    )
    parser.add_argument(
        "--pressure", type=float, default=5000.0,
        help="Design pressure rise in Pa (default: 5000)"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Enable 3D visualization with tracer particles"
    )
    parser.add_argument(
        "--particles", "-p", type=int, default=1000,
        help="Number of SPH air particles (default: 1000)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("SPH AIR FLOW SIMULATION")
    print("  Smoothed Particle Hydrodynamics for realistic air flow")
    print("=" * 70)
    
    # Import modules
    from airclassifier.simulation.air_flow_physics import (
        AirFlowPhysicsSimulator,
        AirFlowPhysicsConfig,
        create_air_flow_simulator,
    )
    from airclassifier.geometry.assembly.air_system import (
        AirSystemAssembly,
        AirSystemParams,
    )
    
    # Create assembly
    print("\nCreating air system assembly...")
    params = AirSystemParams(
        flow_rate_m3_h=args.flow_rate,
        pressure_rise_Pa=args.pressure,
        num_control_dampers=2,
    )
    assembly = AirSystemAssembly(params, device="cpu")
    assembly.build_mesh()
    assembly.print_summary()
    
    # Create physics config with SPH air particles
    print("\nConfiguring SPH physics simulation...")
    config = AirFlowPhysicsConfig(
        dt=args.dt,
        total_time=args.time,
        target_rpm=args.rpm,
        ramp_time=2.0,
        damper_ramp_time=1.0,
        enable_sph=args.visualize,
        num_particles=args.particles,
        smoothing_length=0.04,      # 40mm SPH kernel radius
        speed_of_sound=50.0,        # Artificial for stability
        sph_viscosity=0.01,         # SPH viscosity coefficient
        xsph_factor=0.1,            # Velocity smoothing
        device=args.device,
    )
    
    # Create simulator
    simulator = AirFlowPhysicsSimulator(assembly, config)
    
    # Initialize SPH air particles if enabled
    if args.visualize:
        print("\nInitializing SPH air particles...")
        simulator.initialize_particles()
    
    # Visualization setup
    plotter = None
    tracer_actor = None
    animated_actors = {}
    
    if args.visualize and HAS_PYVISTA:
        print("\nSetting up 3D visualization...")
        pv.set_plot_theme("document")
        plotter = pv.Plotter(title="SPH Air Flow Simulation")
        plotter.set_background("white")
        plotter.camera.up = (0, 1, 0)
        
        # Component positions
        filter_offset = np.array(assembly._filter_position)
        blower_offset = np.array(assembly._blower_position)
        
        # ============================================
        # ADD FILTER (STATIC)
        # ============================================
        print("  Adding inlet filter...")
        v, i, _ = assembly.inlet_filter.generate_mesh()
        v = v + filter_offset
        faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
        filter_mesh = pv.PolyData(v, faces)
        plotter.add_mesh(filter_mesh, color='#87CEEB', label='Inlet Filter', opacity=0.7)
        
        # ============================================
        # ADD BLOWER (STATIC + ANIMATED PARTS)
        # ============================================
        print("  Adding blower with belt drive system...")
        
        # Static parts (housing, motor body, belt)
        v_static, i_static, _ = assembly.blower.get_static_mesh()
        v_static = v_static + blower_offset
        faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
        blower_mesh = pv.PolyData(v_static, faces_static)
        plotter.add_mesh(blower_mesh, color='#4682B4', label='Blower Housing', opacity=0.8)
        
        # Animated impeller (rotates with driven pulley)
        v_imp, i_imp, _ = assembly.blower.get_impeller_mesh(0)
        v_imp = v_imp + blower_offset
        faces_imp = np.hstack([[3] + list(face) for face in i_imp.reshape(-1, 3)])
        impeller_mesh = pv.PolyData(v_imp, faces_imp)
        impeller_actor = plotter.add_mesh(impeller_mesh, color='#FFD700', 
                                          label='Impeller', opacity=0.95)
        
        animated_actors['impeller'] = {
            'mesh': impeller_mesh,
            'actor': impeller_actor,
            'component': assembly.blower,
            'position': blower_offset,
        }
        
        # Animated driven pulley (same rotation as impeller)
        v_driven, i_driven, _ = assembly.blower.get_driven_pulley_mesh(0)
        v_driven = v_driven + blower_offset
        faces_driven = np.hstack([[3] + list(face) for face in i_driven.reshape(-1, 3)])
        driven_mesh = pv.PolyData(v_driven, faces_driven)
        driven_actor = plotter.add_mesh(driven_mesh, color='#A0522D', opacity=0.95)
        
        animated_actors['driven_pulley'] = {
            'mesh': driven_mesh,
            'actor': driven_actor,
            'component': assembly.blower,
            'position': blower_offset,
        }
        
        # Animated motor pulley (rotates faster than impeller)
        v_motor, i_motor, _ = assembly.blower.get_motor_pulley_mesh(0)
        v_motor = v_motor + blower_offset
        faces_motor = np.hstack([[3] + list(face) for face in i_motor.reshape(-1, 3)])
        motor_mesh = pv.PolyData(v_motor, faces_motor)
        motor_actor = plotter.add_mesh(motor_mesh, color='#2F4F4F', opacity=0.95)
        
        animated_actors['motor_pulley'] = {
            'mesh': motor_mesh,
            'actor': motor_actor,
            'component': assembly.blower,
            'position': blower_offset,
        }
        
        # ============================================
        # ADD DAMPERS (STATIC + ANIMATED BLADES)
        # ============================================
        print("  Adding dampers...")
        for idx, (damper, pos) in enumerate(zip(assembly.dampers, assembly._damper_positions)):
            damper_offset = np.array(pos)
            
            # Static housing
            v_static, i_static, _ = damper.get_static_mesh()
            v_static = v_static + damper_offset
            faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
            damper_mesh = pv.PolyData(v_static, faces_static)
            plotter.add_mesh(damper_mesh, color='#CD853F', 
                            label=f'Damper {idx+1}' if idx == 0 else None, opacity=0.8)
            
            # Animated blade
            v_blade, i_blade, _ = damper.get_blade_mesh(0.0)  # Start closed
            v_blade = v_blade + damper_offset
            faces_blade = np.hstack([[3] + list(face) for face in i_blade.reshape(-1, 3)])
            blade_mesh = pv.PolyData(v_blade, faces_blade)
            blade_actor = plotter.add_mesh(blade_mesh, color='#FF6B35', opacity=0.95)
            
            animated_actors[f'damper_{idx}'] = {
                'mesh': blade_mesh,
                'actor': blade_actor,
                'component': damper,
                'position': damper_offset,
            }
        
        # ============================================
        # ADD DUCTWORK
        # ============================================
        print("  Adding ductwork...")
        for duct, pos in assembly._duct_sections:
            v, i, _ = duct.generate_mesh()
            v = v + np.array(pos)
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            duct_mesh = pv.PolyData(v, faces)
            plotter.add_mesh(duct_mesh, color='#708090', opacity=0.5)
        
        # ============================================
        # INFO TEXT
        # ============================================
        plotter.add_text(
            "AIR FLOW SIMULATION\nStarting...",
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
    print(f"  Time:       {args.time:.1f} s")
    print(f"  dt:         {args.dt*1000:.1f} ms")
    print(f"  Steps:      {int(args.time / args.dt):,}")
    print(f"  Target RPM: {args.rpm:.0f}")
    print("-" * 70)
    
    total_steps = int(args.time / args.dt)
    print_interval = max(1, total_steps // 20)  # ~20 console updates
    
    # Animation timing - use wall clock for smooth visuals
    target_fps = 30
    frame_interval = 1.0 / target_fps
    steps_per_frame = max(1, int(frame_interval / args.dt))
    
    # Start the system
    simulator.start_system()
    print("  [System starting]")
    
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
        results = simulator.get_results()
        phase = results['phase']
        rpm = results['blower_rpm']
        flow = results['volume_flow_rate_m3_h']
        pressure = results['pressure_rise_Pa']
        power = results['shaft_power_kW']
        efficiency = results['efficiency'] * 100
        dampers = results['damper_positions']
        
        # Console output at intervals
        if step - last_print_step >= print_interval or step >= total_steps:
            last_print_step = step
            progress = 100.0 * step / total_steps
            print(f"  [{progress:5.1f}%] t={results['time']:5.2f}s | "
                  f"{phase:8s} RPM:{rpm:4.0f} Q:{flow:5.0f}m3/h "
                  f"P:{pressure:5.0f}Pa W:{power:4.1f}kW n:{efficiency:4.1f}% "
                  f"D1:{dampers[0]*100:3.0f}% D2:{dampers[1]*100:3.0f}%")
        
        # Update visualization every frame
        if plotter is not None:
            # Calculate wall-clock dt for smooth animation
            current_wall_time = time.time()
            wall_dt = current_wall_time - last_wall_time
            last_wall_time = current_wall_time
            wall_dt = min(wall_dt, 0.1)  # Clamp to avoid jumps
            
            # ============================================
            # ANIMATE BLOWER COMPONENTS (BELT DRIVE SYSTEM)
            # Uses wall_dt for smooth animation independent of sim speed
            # ============================================
            if 'impeller' in animated_actors:
                data = animated_actors['impeller']
                blower = data['component']
                
                # Update rotation angles based on current RPM and wall time
                if rpm > 0:
                    blower.update_animation(wall_dt, rpm)
                
                # Update impeller mesh
                v_imp, _, _ = blower.get_impeller_mesh()
                v_imp = v_imp + data['position']
                data['mesh'].points[:] = v_imp
                data['mesh'].Modified()
            
            # Update driven pulley (same speed as impeller)
            if 'driven_pulley' in animated_actors:
                data = animated_actors['driven_pulley']
                blower = data['component']
                
                v_driven, _, _ = blower.get_driven_pulley_mesh()
                v_driven = v_driven + data['position']
                data['mesh'].points[:] = v_driven
                data['mesh'].Modified()
            
            # Update motor pulley (faster by pulley ratio)
            if 'motor_pulley' in animated_actors:
                data = animated_actors['motor_pulley']
                blower = data['component']
                
                v_motor, _, _ = blower.get_motor_pulley_mesh()
                v_motor = v_motor + data['position']
                data['mesh'].points[:] = v_motor
                data['mesh'].Modified()
            
            # ============================================
            # ANIMATE DAMPER BLADES
            # Dampers open/close based on simulation state
            # ============================================
            for idx, target_pos in enumerate(dampers):
                key = f'damper_{idx}'
                if key in animated_actors:
                    data = animated_actors[key]
                    damper = data['component']
                    
                    # Use damper's animation method if available
                    if hasattr(damper, 'update_animation'):
                        damper.update_animation(wall_dt, target_pos, transition_time=1.0)
                        current_pos = damper.get_blade_position() if hasattr(damper, 'get_blade_position') else target_pos
                    else:
                        current_pos = target_pos
                    
                    # Update blade mesh
                    v_blade, _, _ = damper.get_blade_mesh(current_pos)
                    v_blade = v_blade + data['position']
                    data['mesh'].points[:] = v_blade
                    data['mesh'].Modified()
                
                # ============================================
                # UPDATE SPH AIR PARTICLES
                # ============================================
                if args.visualize and simulator.state.positions is not None:
                    positions = simulator.get_particle_positions()
                    velocities = simulator.get_particle_velocities()
                    densities = simulator.get_particle_densities()
                    
                    if len(positions) > 0:
                        # Remove old actor
                        if tracer_actor is not None:
                            try:
                                plotter.remove_actor(tracer_actor)
                            except:
                                pass
                        
                        # Create point cloud with velocity coloring
                        speeds = np.linalg.norm(velocities, axis=1)
                        particle_mesh = pv.PolyData(positions)
                        particle_mesh['velocity'] = speeds
                        particle_mesh['density'] = densities
                        
                        tracer_actor = plotter.add_mesh(
                            particle_mesh,
                            scalars='velocity',
                            cmap='coolwarm',
                            point_size=8,
                            render_points_as_spheres=True,
                            opacity=0.9,
                            clim=[0, max(speeds.max(), 1.0)],
                            show_scalar_bar=True,
                            scalar_bar_args={'title': 'Velocity (m/s)', 'vertical': True},
                        )
                
                # ============================================
                # UPDATE INFO TEXT
                # ============================================
                tip_speed = results.get('tip_speed', 0)
                blower_state = results.get('blower_state', 'OFF')
                max_vel = results.get('max_velocity', 0)
                avg_density = results.get('avg_density', 1.225)
                # Motor RPM is impeller RPM * pulley ratio (2.0)
                motor_rpm = rpm * 2.0  # Belt drive ratio
                info_text = (
                    f"SPH AIR FLOW SIMULATION\n"
                    f"Phase: {phase.upper()}\n"
                    f"Blower: {blower_state}\n"
                    f"\n"
                    f"Impeller RPM: {rpm:.0f}\n"
                    f"Motor RPM: {motor_rpm:.0f}\n"
                    f"Tip Speed: {tip_speed:.1f} m/s\n"
                    f"\n"
                    f"Flow: {flow:.0f} m3/h\n"
                    f"Pressure: {pressure:.0f} Pa\n"
                    f"Power: {power:.1f} kW\n"
                    f"Efficiency: {efficiency:.1f}%\n"
                    f"\n"
                    f"SPH Max Vel: {max_vel:.1f} m/s\n"
                    f"SPH Density: {avg_density:.3f} kg/m3\n"
                    f"\n"
                    f"Damper 1: {dampers[0]*100:.0f}%\n"
                    f"Damper 2: {dampers[1]*100:.0f}%\n"
                    f"\n"
                    f"t = {results['time']:.2f}s"
                )
                
                plotter.add_text(info_text, position='upper_left', font_size=10,
                                color='black', name='sim_info')
                plotter.update()
                
                # Small delay to control frame rate
                time.sleep(0.001)
    
    elapsed = time.time() - start_time
    
    print("-" * 70)
    print("SIMULATION COMPLETE - SHUTTING DOWN")
    print("-" * 70)
    print(f"  Wall time:       {elapsed:.1f} s")
    print(f"  Sim time:        {simulator.state.time:.2f} s")
    print(f"  Steps:           {simulator.state.step:,}")
    print(f"  Rate:            {simulator.state.step / elapsed:.0f} steps/s")
    
    results = simulator.get_results()
    print(f"\nFinal operating point:")
    print(f"  Blower RPM:      {results['blower_rpm']:.0f}")
    print(f"  Flow rate:       {results['volume_flow_rate_m3_h']:.0f} m³/h")
    print(f"  Pressure rise:   {results['pressure_rise_Pa']:.0f} Pa")
    print(f"  Shaft power:     {results['shaft_power_kW']:.1f} kW")
    print(f"  Efficiency:      {results['efficiency']*100:.1f}%")
    print(f"  Total energy:    {results['total_energy_kWh']:.3f} kWh")
    
    # Print duct segment data
    print(f"\nDuct segment pressure drops:")
    for segment in simulator.get_duct_segment_data():
        print(f"  {segment['name']:20s}: {segment['pressure_drop']:.1f} Pa "
              f"(v={segment['velocity']:.1f} m/s, Re={segment['reynolds']:.0f})")
    
    # ============================================
    # SHUTDOWN SEQUENCE - Close dampers, slow motor
    # ============================================
    if plotter is not None and animated_actors:
        print("\n  Shutting down system...")
        shutdown_time = 3.0  # 3 second shutdown
        shutdown_start = time.time()
        
        final_rpm = results['blower_rpm']
        
        while time.time() - shutdown_start < shutdown_time:
            shutdown_elapsed = time.time() - shutdown_start
            shutdown_progress = shutdown_elapsed / shutdown_time
            
            # Close dampers gradually (1.0 -> 0.0)
            damper_position = 1.0 - shutdown_progress
            
            # Slow down motor/impeller
            current_rpm = final_rpm * (1.0 - shutdown_progress)
            
            # Update blower animation
            if 'impeller' in animated_actors:
                data = animated_actors['impeller']
                blower = data['component']
                
                if current_rpm > 0:
                    blower.update_animation(0.03, current_rpm)
                
                v_imp, _, _ = blower.get_impeller_mesh()
                v_imp = v_imp + data['position']
                data['mesh'].points[:] = v_imp
                data['mesh'].Modified()
            
            if 'driven_pulley' in animated_actors:
                data = animated_actors['driven_pulley']
                blower = data['component']
                v_driven, _, _ = blower.get_driven_pulley_mesh()
                v_driven = v_driven + data['position']
                data['mesh'].points[:] = v_driven
                data['mesh'].Modified()
            
            if 'motor_pulley' in animated_actors:
                data = animated_actors['motor_pulley']
                blower = data['component']
                v_motor, _, _ = blower.get_motor_pulley_mesh()
                v_motor = v_motor + data['position']
                data['mesh'].points[:] = v_motor
                data['mesh'].Modified()
            
            # Close dampers
            for idx in range(2):
                key = f'damper_{idx}'
                if key in animated_actors:
                    data = animated_actors[key]
                    damper = data['component']
                    
                    # Use animation method if available
                    if hasattr(damper, 'update_animation'):
                        damper.update_animation(0.03, damper_position, transition_time=0.5)
                        current_pos = damper.get_blade_position() if hasattr(damper, 'get_blade_position') else damper_position
                    else:
                        current_pos = damper_position
                    
                    v_blade, _, _ = damper.get_blade_mesh(current_pos)
                    v_blade = v_blade + data['position']
                    data['mesh'].points[:] = v_blade
                    data['mesh'].Modified()
            
            # Update info text
            shutdown_text = (
                f"AIR FLOW - SHUTTING DOWN\n"
                f"\n"
                f"Impeller: {current_rpm:.0f} RPM\n"
                f"Motor: {current_rpm * 2.0:.0f} RPM\n"
                f"\n"
                f"Damper 1: {damper_position*100:.0f}%\n"
                f"Damper 2: {damper_position*100:.0f}%\n"
                f"\n"
                f"Shutdown: {shutdown_progress*100:.0f}%"
            )
            plotter.add_text(shutdown_text, position='upper_left', font_size=10,
                           color='black', name='sim_info')
            plotter.update()
            time.sleep(0.03)
        
        # Final state - dampers fully closed
        for idx in range(2):
            key = f'damper_{idx}'
            if key in animated_actors:
                data = animated_actors[key]
                damper = data['component']
                
                if hasattr(damper, 'set_blade_position'):
                    damper.set_blade_position(0.0)
                
                v_blade, _, _ = damper.get_blade_mesh(0.0)
                v_blade = v_blade + data['position']
                data['mesh'].points[:] = v_blade
                data['mesh'].Modified()
        
        print("  System shutdown complete. Dampers closed.")
        
        # Final display
        final_text = (
            f"AIR FLOW - COMPLETE\n"
            f"\n"
            f"System: OFF\n"
            f"Dampers: CLOSED\n"
            f"\n"
            f"Run Summary:\n"
            f"  Energy: {results['total_energy_kWh']:.3f} kWh\n"
            f"  Flow: {results['volume_flow_rate_m3_h']:.0f} m3/h\n"
            f"\n"
            f"(Close window to exit)"
        )
        plotter.add_text(final_text, position='upper_left', font_size=10,
                        color='black', name='sim_info')
        plotter.update()
    
    print("=" * 70)
    
    if plotter is not None:
        print("\nVisualization window open. Close to exit.")
        plotter.show()


if __name__ == "__main__":
    main()
