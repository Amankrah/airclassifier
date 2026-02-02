#!/usr/bin/env python
"""
Live Visualization Simulation Script

Runs simulations with real-time 3D visualization of the system geometry
and live updates of simulation results.

Usage:
    python examples/run_viz_simulation.py --air           # Air system simulation
    python examples/run_viz_simulation.py --feed          # Feed system simulation
    python examples/run_viz_simulation.py --classification # Classification system
    python examples/run_viz_simulation.py --complete      # Complete system simulation
    python examples/run_viz_simulation.py --all           # Run all systems sequentially

Requirements:
    pip install pyvista numpy warp-lang
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path

# Add src to path if running from examples folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Check for PyVista
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista not available. Install with: pip install pyvista")

# Create results directory
RESULTS_DIR = Path("./results/live_simulation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Component colors for visualization
COLORS = {
    # Air system
    'air_filter': '#3498DB',     # Blue
    'blower': '#27AE60',         # Green
    'damper': '#F39C12',         # Orange
    'duct': '#95A5A6',           # Gray
    
    # Feed system
    'hopper': '#F0AD4E',         # Orange
    'airlock': '#3498DB',        # Light Blue
    'screw_feeder': '#27AE60',   # Green
    'deagglomerator': '#9B59B6', # Purple
    'transition': '#95A5A6',     # Gray
    
    # Classification system
    'venturi': '#3498DB',        # Blue
    'zigzag': '#2ECC71',         # Green
    'multi_cyclone': '#E74C3C',  # Red
    'bag_filter': '#F39C12',     # Orange
    
    # Complete system
    'classification': '#3498DB', # Blue
    'feed_system': '#27AE60',    # Green
    'air_system': '#F39C12',     # Orange
    'silencer': '#E74C3C',       # Red
    'exhaust_stack': '#9B59B6',  # Purple
    'ductwork': '#7F8C8D',       # Gray
}


class LiveSimulationVisualizer:
    """
    Manages live 3D visualization with simulation updates.
    """
    
    def __init__(self, title: str = "Live Simulation"):
        self.title = title
        self.plotter = None
        self.info_actor = None
        self.simulation_running = False
        self.simulation_results = {}
        
    def create_plotter(self):
        """Initialize PyVista plotter."""
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for live visualization")
        
        self.plotter = pv.Plotter(title=self.title)
        self.plotter.set_background('white')
        self.plotter.camera.up = (0, 1, 0)  # Y-up coordinate system
        self.plotter.add_axes()
        
    def add_info_panel(self, text: str):
        """Add or update info panel with simulation status."""
        if self.info_actor is not None:
            self.plotter.remove_actor(self.info_actor)
        
        self.info_actor = self.plotter.add_text(
            text,
            position='upper_left',
            font_size=10,
            color='black',
            name='info_panel'
        )
    
    def update_info(self, results: dict, system_type: str):
        """Update the info panel with current results."""
        if system_type == 'air':
            text = (
                f"AIR SYSTEM SIMULATION\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: {results.get('system_state', 'N/A')}\n"
                f"Blower: {results.get('blower_rpm', 0):.0f} RPM\n"
                f"Flow: {results.get('flow_rate_m3_h', 0):.0f} m³/h\n"
                f"Pressure: {results.get('pressure_Pa', 0):.0f} Pa\n"
                f"Power: {results.get('power_consumption_kW', 0):.2f} kW\n"
                f"Time: {results.get('time', 0):.2f} s"
            )
        elif system_type == 'feed':
            text = (
                f"FEED SYSTEM SIMULATION\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: {results.get('system_state', 'N/A')}\n"
                f"Airlock: {results.get('airlock_rpm', 0):.0f} RPM\n"
                f"Feeder: {results.get('feeder_rpm', 0):.0f} RPM\n"
                f"Deagg: {results.get('deagg_rpm', 0):.0f} RPM\n"
                f"Flow: {results.get('mass_flow_rate_kg_h', 0):.0f} kg/h\n"
                f"Hopper: {results.get('hopper_mass_kg', 0):.0f} kg\n"
                f"Time: {results.get('time', 0):.2f} s"
            )
        elif system_type == 'classification':
            text = (
                f"CLASSIFICATION SIMULATION\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Particles: {results.get('particles_injected', 0)}\n"
                f"Active: {results.get('particles_active', 0)}\n"
                f"Coarse: {results.get('particles_coarse', 0)}\n"
                f"Fines: {results.get('particles_fines', 0)}\n"
                f"Efficiency: {results.get('separation_efficiency', 0)*100:.1f}%\n"
                f"Time: {results.get('time', 0):.3f} s"
            )
        elif system_type == 'complete':
            text = (
                f"COMPLETE SYSTEM SIMULATION\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: {results.get('system_state', 'N/A')}\n"
                f"Flow: {results.get('total_flow_rate_m3_h', 0):.0f} m³/h\n"
                f"Pressure: {results.get('system_pressure_Pa', 0):.0f} Pa\n"
                f"Power: {results.get('total_power_kW', 0):.2f} kW\n"
                f"Feed: {results.get('feed_rate_kg_h', 0):.0f} kg/h\n"
                f"Time: {results.get('time', 0):.2f} s"
            )
        else:
            text = f"Simulation running...\nTime: {results.get('time', 0):.2f} s"
        
        self.add_info_panel(text)


def run_air_system_live():
    """
    Run Air System simulation with live 3D visualization.
    """
    print("\n" + "="*70)
    print("AIR SYSTEM - LIVE VISUALIZATION")
    print("="*70)
    
    from airclassifier.geometry.assembly import create_standard_air_system
    from airclassifier.simulation.simulator import AirSystemSimulator, AirSystemConfig
    
    # Create assembly
    print("\nCreating air system assembly...")
    assembly = create_standard_air_system(device="cpu")
    assembly.print_summary()
    
    if not PYVISTA_AVAILABLE:
        print("\nPyVista not available - running simulation without 3D visualization")
        config = AirSystemConfig(dt=1.0e-4, duration=2.0, device="cpu")
        simulator = AirSystemSimulator(assembly, config)
        simulator.run()
        results = simulator.get_results()
        print(f"\nFinal Results: {results}")
        return simulator
    
    # Create plotter
    print("\nInitializing 3D visualization with ANIMATION support...")
    plotter = pv.Plotter(title="Air System - Live Simulation")
    plotter.set_background('white')
    plotter.camera.up = (0, 1, 0)
    
    # Track animated meshes for updates
    animated_actors = {}
    
    # Add geometry meshes - separate STATIC and ANIMATED parts
    print("  Adding inlet filter (static)...")
    if assembly.inlet_filter is not None:
        v, i, _ = assembly.inlet_filter.generate_mesh()
        v = v + np.array(assembly._filter_position)
        faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
        mesh = pv.PolyData(v, faces)
        plotter.add_mesh(mesh, color=COLORS['air_filter'], label='Inlet Filter', opacity=0.85)
    
    print("  Adding blower (static + animated belt drive)...")
    blower_offset = np.array(assembly._blower_position)
    if assembly.blower is not None:
        # Static parts (scroll, inlet, outlet, motor body, belt, supports)
        v_static, i_static, _ = assembly.blower.get_static_mesh()
        v_static = v_static + blower_offset
        faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
        mesh_static = pv.PolyData(v_static, faces_static)
        plotter.add_mesh(mesh_static, color=COLORS['blower'], label='Blower Housing', opacity=0.85)
        
        # Animated part 1: IMPELLER (rotates with driven pulley)
        v_imp, i_imp, _ = assembly.blower.get_impeller_mesh(0)
        v_imp = v_imp + blower_offset
        faces_imp = np.hstack([[3] + list(face) for face in i_imp.reshape(-1, 3)])
        impeller_mesh = pv.PolyData(v_imp, faces_imp)
        impeller_actor = plotter.add_mesh(impeller_mesh, color='#FFD700', 
                                          label='Impeller', opacity=0.95)
        animated_actors['impeller'] = {
            'actor': impeller_actor,
            'mesh': impeller_mesh,
            'component': assembly.blower,
            'offset': blower_offset,
            'indices': i_imp,
        }
        
        # Animated part 2: DRIVEN PULLEY (large, same speed as impeller)
        v_dp, i_dp, _ = assembly.blower.get_driven_pulley_mesh(0)
        v_dp = v_dp + blower_offset
        faces_dp = np.hstack([[3] + list(face) for face in i_dp.reshape(-1, 3)])
        driven_pulley_mesh = pv.PolyData(v_dp, faces_dp)
        driven_pulley_actor = plotter.add_mesh(driven_pulley_mesh, color='#A0522D',
                                               label='Driven Pulley', opacity=0.95)
        animated_actors['driven_pulley'] = {
            'actor': driven_pulley_actor,
            'mesh': driven_pulley_mesh,
            'component': assembly.blower,
            'offset': blower_offset,
            'indices': i_dp,
        }
        
        # Animated part 3: MOTOR PULLEY (small, spins faster)
        v_mp, i_mp, _ = assembly.blower.get_motor_pulley_mesh(0)
        v_mp = v_mp + blower_offset
        faces_mp = np.hstack([[3] + list(face) for face in i_mp.reshape(-1, 3)])
        motor_pulley_mesh = pv.PolyData(v_mp, faces_mp)
        motor_pulley_actor = plotter.add_mesh(motor_pulley_mesh, color='#CD853F',
                                              label='Motor Pulley', opacity=0.95)
        animated_actors['motor_pulley'] = {
            'actor': motor_pulley_actor,
            'mesh': motor_pulley_mesh,
            'component': assembly.blower,
            'offset': blower_offset,
            'indices': i_mp,
        }
    
    print("  Adding dampers (static + animated)...")
    if hasattr(assembly, 'dampers') and assembly.dampers:
        for idx, (damper, position) in enumerate(zip(assembly.dampers, assembly._damper_positions)):
            damper_offset = np.array(position)
            
            # Static parts (housing, actuator, flanges)
            v_static, i_static, _ = damper.get_static_mesh()
            v_static = v_static + damper_offset
            faces_static = np.hstack([[3] + list(face) for face in i_static.reshape(-1, 3)])
            mesh_static = pv.PolyData(v_static, faces_static)
            plotter.add_mesh(mesh_static, color=COLORS['damper'],
                            label='Damper Housing' if idx == 0 else None, opacity=0.85)
            
            # Animated parts (blade)
            v_blade, i_blade, _ = damper.get_blade_mesh(1.0)  # Start open
            v_blade = v_blade + damper_offset
            faces_blade = np.hstack([[3] + list(face) for face in i_blade.reshape(-1, 3)])
            blade_mesh = pv.PolyData(v_blade, faces_blade)
            blade_actor = plotter.add_mesh(blade_mesh, color='#FF6B35',
                                          label='Damper Blade' if idx == 0 else None, opacity=0.95)
            animated_actors[f'damper_{idx}'] = {
                'actor': blade_actor,
                'mesh': blade_mesh,
                'component': damper,
                'offset': damper_offset,
            }
    
    print("  Adding ductwork (static)...")
    if hasattr(assembly, '_duct_sections') and assembly._duct_sections:
        for idx, (duct, position) in enumerate(assembly._duct_sections):
            v, i, _ = duct.generate_mesh()
            v = v + np.array(position)
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=COLORS['duct'],
                            label='Ductwork' if idx == 0 else None, opacity=0.7)
    
    plotter.add_legend(bcolor='white', face='circle')
    plotter.add_axes()
    plotter.reset_camera()
    plotter.camera.azimuth = -170
    plotter.camera.elevation = -20
    
    # Create simulator
    config = AirSystemConfig(
        dt=1.0e-3,      # 1ms timestep for visualization
        duration=60.0, # 1 minute simulation
        output_interval=0.1,
        blower_rpm=3000.0,
        blower_ramp_time=2.0,  # 2 second ramp for visible startup
        device="cpu",
    )
    simulator = AirSystemSimulator(assembly, config)
    
    # Start the system!
    simulator.start()
    
    print("\n" + "-"*70)
    print("LIVE SIMULATION STARTING")
    print("-"*70)
    print("3D window will show live simulation updates.")
    print("Close the window (press 'q') to stop.")
    print("-"*70)
    
    # Add initial info text (using named actor for smooth updates)
    plotter.add_text(
        "AIR SYSTEM SIMULATION\nStarting...",
        position='upper_left', font_size=12, color='black', name='sim_info'
    )
    
    # Show plotter in interactive mode for live updates
    plotter.show(interactive_update=True, auto_close=False)
    
    # Animation timing - use wall clock time for smooth visuals
    import time as time_module
    last_wall_time = time_module.time()
    target_fps = 30  # Target frame rate
    frame_interval = 1.0 / target_fps
    
    # Damper animation state
    # Butterfly dampers start CLOSED (0.0) when system is OFF
    # They open progressively as the blower ramps up
    damper_target_position = 0.0  # Start closed
    
    # Run simulation with live visualization updates
    total_steps = config.num_steps
    steps_per_frame = max(1, int(frame_interval / config.dt))  # Steps per visual frame
    
    # Progress tracking - print every 10%
    last_print_pct = -10  # Will trigger first print at 0%
    print_interval_pct = 10  # Print every 10%
    
    print(f"  Animation: {target_fps} FPS, {steps_per_frame} sim steps per frame")
    
    try:
        step = 0
        while step < total_steps:
            # Run multiple simulation steps per frame for efficiency
            frame_steps = min(steps_per_frame, total_steps - step)
            for _ in range(frame_steps):
                simulator.step()
                step += 1
            
            results = simulator.get_results()
            current_rpm = results['blower_rpm']
            
            # Calculate actual wall-clock dt for smooth animation
            current_wall_time = time_module.time()
            wall_dt = current_wall_time - last_wall_time
            last_wall_time = current_wall_time
            
            # Clamp wall_dt to avoid jumps
            wall_dt = min(wall_dt, 0.1)
            
            # ============================================
            # ANIMATE BELT DRIVE SYSTEM
            # Motor pulley spins FAST, driven pulley & impeller spin SLOW
            # ============================================
            if 'impeller' in animated_actors:
                anim_data = animated_actors['impeller']
                blower = anim_data['component']
                
                # Update all rotating components based on current RPM
                # update_animation updates: motor_angle, impeller_angle
                if current_rpm > 0:
                    blower.update_animation(wall_dt, current_rpm)
                
                # 1. Update IMPELLER mesh
                v_imp, i_imp, _ = blower.get_impeller_mesh()
                v_imp = v_imp + anim_data['offset']
                anim_data['mesh'].points[:] = v_imp
                anim_data['mesh'].Modified()
            
            # 2. Update DRIVEN PULLEY (same speed as impeller)
            if 'driven_pulley' in animated_actors:
                dp_data = animated_actors['driven_pulley']
                blower = dp_data['component']
                v_dp, _, _ = blower.get_driven_pulley_mesh()  # Uses current impeller_angle
                v_dp = v_dp + dp_data['offset']
                dp_data['mesh'].points[:] = v_dp
                dp_data['mesh'].Modified()
            
            # 3. Update MOTOR PULLEY (spins FASTER by pulley ratio)
            if 'motor_pulley' in animated_actors:
                mp_data = animated_actors['motor_pulley']
                blower = mp_data['component']
                v_mp, _, _ = blower.get_motor_pulley_mesh()  # Uses current motor_angle
                v_mp = v_mp + mp_data['offset']
                mp_data['mesh'].points[:] = v_mp
                mp_data['mesh'].Modified()
            
            # ============================================
            # ANIMATE DAMPERS - open during startup
            # ============================================
            # Damper control logic:
            # - OFF/STOPPING: CLOSED (prevents backflow)
            # - STARTING: Opens progressively with blower RPM
            # - RUNNING: FULLY OPEN
            if results['system_state'] == 'off':
                damper_target_position = 0.0  # Closed
            elif results['system_state'] == 'starting':
                # Open proportionally to blower RPM (0% RPM = closed, 100% RPM = open)
                damper_target_position = min(1.0, current_rpm / config.blower_rpm)
            elif results['system_state'] == 'running':
                damper_target_position = 1.0  # Fully open
            elif results['system_state'] == 'stopping':
                # Close proportionally as blower slows
                damper_target_position = min(1.0, current_rpm / config.blower_rpm)
            
            for key, anim_data in animated_actors.items():
                if key.startswith('damper_'):
                    damper = anim_data['component']
                    
                    # Animate toward target position
                    damper.update_animation(wall_dt, damper_target_position, transition_time=1.0)
                    
                    # Get blade mesh at current position and update
                    current_pos = damper.get_blade_position()
                    v_blade, i_blade, _ = damper.get_blade_mesh(current_pos)
                    v_blade = v_blade + anim_data['offset']
                    
                    # Update mesh points and mark as modified for rendering
                    try:
                        anim_data['mesh'].points[:] = v_blade
                        anim_data['mesh'].Modified()
                    except Exception as e:
                        # If point count differs, need different approach
                        print(f"Damper update issue: {e}")
            
            # Update display
            plotter.update()
            
            # Calculate current progress percentage
            pct = (step / total_steps) * 100
            
            # Get damper position for display
            damper_pos_pct = 100.0
            for key, anim_data in animated_actors.items():
                if key.startswith('damper_'):
                    damper_pos_pct = anim_data['component'].get_blade_position() * 100
                    break
            
            # Update info text every frame
            text = (
                f"AIR SYSTEM SIMULATION\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: {results['system_state']}\n"
                f"Blower: {results['blower_rpm']:.0f} RPM\n"
                f"Dampers: {damper_pos_pct:.0f}% open\n"
                f"Flow: {results['flow_rate_m3_h']:.0f} m³/h\n"
                f"Pressure: {results['pressure_Pa']:.0f} Pa\n"
                f"Power: {results['power_consumption_kW']:.2f} kW\n"
                f"Progress: {pct:.0f}%\n"
                f"Time: {results['time']:.2f} s"
            )
            
            plotter.add_text(text, position='upper_left', font_size=12, 
                           color='black', name='sim_info')
            
            # Print progress to console when crossing percentage thresholds
            if pct >= last_print_pct + print_interval_pct:
                last_print_pct = int(pct / print_interval_pct) * print_interval_pct
                print(f"  [{pct:5.1f}%] RPM: {results['blower_rpm']:4.0f} | "
                      f"Flow: {results['flow_rate_m3_h']:4.0f} m³/h | "
                      f"Power: {results['power_consumption_kW']:.2f} kW | "
                      f"Time: {results['time']:.1f}s")
            
            # Small delay to not overwhelm the display
            time_module.sleep(0.001)
        
        # Get final results
        results = simulator.get_results()
        
        print("\n" + "-"*70)
        print("SIMULATION COMPLETE - SHUTTING DOWN")
        print("-"*70)
        print(f"  Final State:     {results['system_state']}")
        print(f"  Blower RPM:      {results['blower_rpm']:.0f}")
        print(f"  Flow Rate:       {results['flow_rate_m3_h']:.0f} m³/h")
        print(f"  Pressure:        {results['pressure_Pa']:.0f} Pa")
        print(f"  Power:           {results['power_consumption_kW']:.2f} kW")
        print(f"  Total Energy:    {results['total_energy_kWh']:.4f} kWh")
        print("-"*70)
        
        # ============================================
        # SHUTDOWN SEQUENCE - Close dampers, stop motor
        # ============================================
        print("  Closing dampers...")
        shutdown_time = 2.0  # 2 second shutdown
        shutdown_start = time_module.time()
        
        while time_module.time() - shutdown_start < shutdown_time:
            shutdown_elapsed = time_module.time() - shutdown_start
            shutdown_progress = shutdown_elapsed / shutdown_time
            
            # Close dampers gradually (1.0 -> 0.0)
            damper_target = 1.0 - shutdown_progress
            
            for key, anim_data in animated_actors.items():
                if key.startswith('damper_'):
                    damper = anim_data['component']
                    damper.update_animation(0.03, damper_target, transition_time=0.5)
                    current_pos = damper.get_blade_position()
                    v_blade, i_blade, _ = damper.get_blade_mesh(current_pos)
                    v_blade = v_blade + anim_data['offset']
                    anim_data['mesh'].points[:] = v_blade
                    anim_data['mesh'].Modified()
            
            # Slow down motor/impeller
            slowdown_rpm = config.blower_rpm * (1.0 - shutdown_progress)
            if 'impeller' in animated_actors:
                blower = animated_actors['impeller']['component']
                blower.update_animation(0.03, slowdown_rpm)
                
                v_imp, _, _ = blower.get_impeller_mesh()
                v_imp = v_imp + animated_actors['impeller']['offset']
                animated_actors['impeller']['mesh'].points[:] = v_imp
                animated_actors['impeller']['mesh'].Modified()
                
            if 'driven_pulley' in animated_actors:
                v_dp, _, _ = blower.get_driven_pulley_mesh()
                v_dp = v_dp + animated_actors['driven_pulley']['offset']
                animated_actors['driven_pulley']['mesh'].points[:] = v_dp
                animated_actors['driven_pulley']['mesh'].Modified()
                
            if 'motor_pulley' in animated_actors:
                v_mp, _, _ = blower.get_motor_pulley_mesh()
                v_mp = v_mp + animated_actors['motor_pulley']['offset']
                animated_actors['motor_pulley']['mesh'].points[:] = v_mp
                animated_actors['motor_pulley']['mesh'].Modified()
            
            # Update status display
            shutdown_text = (
                f"AIR SYSTEM - SHUTTING DOWN\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"State: stopping\n"
                f"Blower: {slowdown_rpm:.0f} RPM\n"
                f"Dampers: {damper_target*100:.0f}% open\n"
                f"Progress: {shutdown_progress*100:.0f}%\n"
            )
            plotter.add_text(shutdown_text, position='upper_left', font_size=12, 
                           color='black', name='sim_info')
            plotter.update()
            time_module.sleep(0.03)
        
        # Final state - everything stopped, dampers closed
        for key, anim_data in animated_actors.items():
            if key.startswith('damper_'):
                damper = anim_data['component']
                damper.set_blade_position(0.0)  # Fully closed
                v_blade, i_blade, _ = damper.get_blade_mesh(0.0)
                v_blade = v_blade + anim_data['offset']
                anim_data['mesh'].points[:] = v_blade
                anim_data['mesh'].Modified()
        
        print("  System shutdown complete. Dampers closed.")
        
        # Update to final results display
        final_text = (
            f"AIR SYSTEM - COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"State: OFF\n"
            f"Dampers: CLOSED\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Run Summary:\n"
            f"  Total Energy: {results['total_energy_kWh']:.4f} kWh\n"
            f"\n(DONE - Press 'q' to close)"
        )
        plotter.add_text(final_text, position='upper_left', font_size=12, 
                        color='black', name='sim_info')
        plotter.update()
        
        # Keep window responsive until user presses Enter in console
        print("\nSimulation finished!")
        print(">>> Press ENTER in this terminal to close the 3D window <<<")
        
        done_flag = threading.Event()
        
        def wait_for_input():
            input()
            done_flag.set()
        
        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()
        
        while not done_flag.is_set():
            plotter.update()
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        try:
            plotter.close()
        except:
            pass
    
    return simulator


def run_feed_system_live():
    """
    Run Feed System simulation with live 3D visualization.
    """
    print("\n" + "="*70)
    print("FEED SYSTEM - LIVE VISUALIZATION")
    print("="*70)
    
    from airclassifier.geometry.assembly import create_standard_feed_system
    from airclassifier.simulation.simulator import FeedSystemSimulator, FeedSystemConfig
    
    # Create assembly
    print("\nCreating feed system assembly...")
    assembly = create_standard_feed_system(device="cpu")
    assembly.print_summary()
    
    if not PYVISTA_AVAILABLE:
        print("\nPyVista not available - running simulation without 3D visualization")
        config = FeedSystemConfig(dt=1.0e-4, duration=2.0, device="cpu")
        simulator = FeedSystemSimulator(assembly, config)
        simulator.run()
        results = simulator.get_results()
        print(f"\nFinal Results: {results}")
        return simulator
    
    # Create plotter
    print("\nInitializing 3D visualization...")
    plotter = pv.Plotter(title="Feed System - Live Simulation")
    plotter.set_background('white')
    plotter.camera.up = (0, 1, 0)
    
    # Add geometry meshes
    print("  Adding hopper...")
    v, i, _ = assembly.hopper.generate_mesh()
    v = v + np.array(assembly._hopper_position)
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['hopper'], label='Feed Hopper', opacity=0.85)
    
    print("  Adding airlock...")
    v, i, _ = assembly.airlock.generate_mesh()
    v = v + np.array(assembly._airlock_position)
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['airlock'], label='Rotary Airlock', opacity=0.85)
    
    print("  Adding screw feeder...")
    v, i, _ = assembly.feeder.generate_mesh()
    v = v + np.array(assembly._feeder_position)
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['screw_feeder'], label='Screw Feeder', opacity=0.85)
    
    print("  Adding deagglomerator...")
    v, i, _ = assembly.deagglomerator.generate_mesh()
    v = v + np.array(assembly._deagglomerator_position)
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['deagglomerator'], label='Deagglomerator', opacity=0.85)
    
    print("  Adding transitions...")
    if hasattr(assembly, '_transition_connectors') and assembly._transition_connectors:
        for idx, connector_data in enumerate(assembly._transition_connectors):
            trans = connector_data[0]
            v, i, _ = trans.generate_mesh()
            faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
            mesh = pv.PolyData(v, faces)
            plotter.add_mesh(mesh, color=COLORS['transition'],
                            label="Transitions" if idx == 0 else None, opacity=0.7)
    
    plotter.add_legend(bcolor='white', face='circle')
    plotter.add_axes()
    plotter.reset_camera()
    plotter.camera.azimuth = -170
    plotter.camera.elevation = -20
    
    # Create simulator
    config = FeedSystemConfig(
        dt=1.0e-3,
        duration=5.0,
        output_interval=0.1,
        feed_rate_kg_h=500.0,
        airlock_rpm=20.0,
        feeder_rpm=60.0,
        deagg_rpm=1500.0,
        ramp_time=1.0,
        num_particles=500,
        device="cpu",
    )
    simulator = FeedSystemSimulator(assembly, config)
    
    # Start the system!
    simulator.start()
    
    print("\n" + "-"*70)
    print("LIVE SIMULATION STARTING")
    print("-"*70)
    print("3D window will show live simulation updates.")
    print("Close the window (press 'q') to stop.")
    print("-"*70)
    
    # Add initial info text (using named actor for smooth updates)
    plotter.add_text(
        "FEED SYSTEM SIMULATION\nStarting...",
        position='upper_left', font_size=12, color='black', name='sim_info'
    )
    
    # Show plotter in interactive mode for live updates
    plotter.show(interactive_update=True, auto_close=False)
    
    # Run simulation with live visualization updates
    total_steps = config.num_steps
    update_interval = max(1, total_steps // 50)
    
    try:
        for step in range(total_steps):
            simulator.step()
            
            if step % update_interval == 0:
                results = simulator.get_results()
                pct = (step / total_steps) * 100
                
                # Update info text using same name to replace without blinking
                text = (
                    f"FEED SYSTEM SIMULATION\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"State: {results['system_state']}\n"
                    f"Airlock: {results['airlock_rpm']:.0f} RPM\n"
                    f"Feeder: {results['feeder_rpm']:.0f} RPM\n"
                    f"Deagg: {results['deagg_rpm']:.0f} RPM\n"
                    f"Flow: {results['mass_flow_rate_kg_h']:.0f} kg/h\n"
                    f"Hopper: {results['hopper_mass_kg']:.0f} kg\n"
                    f"Particles: {results['particles_injected']}\n"
                    f"Progress: {pct:.0f}%\n"
                    f"Time: {results['time']:.2f} s"
                )
                
                plotter.add_text(text, position='upper_left', font_size=12, 
                               color='black', name='sim_info')
                plotter.update()
                
                if step % (update_interval * 5) == 0:
                    print(f"  [{pct:5.1f}%] Airlock: {results['airlock_rpm']:4.0f} | "
                          f"Feeder: {results['feeder_rpm']:4.0f} | "
                          f"Flow: {results['mass_flow_rate_kg_h']:4.0f} kg/h")
        
        # Get final results
        results = simulator.get_results()
        
        print("\n" + "-"*70)
        print("SIMULATION COMPLETE")
        print("-"*70)
        print(f"  Final State:     {results['system_state']}")
        print(f"  Airlock RPM:     {results['airlock_rpm']:.0f}")
        print(f"  Feeder RPM:      {results['feeder_rpm']:.0f}")
        print(f"  Deagg RPM:       {results['deagg_rpm']:.0f}")
        print(f"  Mass Flow:       {results['mass_flow_rate_kg_h']:.0f} kg/h")
        print(f"  Hopper Mass:     {results['hopper_mass_kg']:.0f} kg")
        print(f"  Particles:       {results['particles_injected']}")
        print("-"*70)
        
        # Update to final results display
        final_text = (
            f"FEED SYSTEM - COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"State: {results['system_state']}\n"
            f"Airlock: {results['airlock_rpm']:.0f} RPM\n"
            f"Feeder: {results['feeder_rpm']:.0f} RPM\n"
            f"Deagg: {results['deagg_rpm']:.0f} RPM\n"
            f"Flow: {results['mass_flow_rate_kg_h']:.0f} kg/h\n"
            f"Hopper: {results['hopper_mass_kg']:.0f} kg\n"
            f"Particles: {results['particles_injected']}\n"
            f"\n[DONE - Press 'q' to close]"
        )
        plotter.add_text(final_text, position='upper_left', font_size=12, 
                        color='black', name='sim_info')
        plotter.update()
        
        # Keep window responsive until user presses Enter in console
        print("\nSimulation finished!")
        print(">>> Press ENTER in this terminal to close the 3D window <<<")
        
        done_flag = threading.Event()
        
        def wait_for_input():
            input()
            done_flag.set()
        
        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()
        
        while not done_flag.is_set():
            plotter.update()
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        try:
            plotter.close()
        except:
            pass
    
    return simulator


def run_classification_live():
    """
    Run Classification System simulation with live 3D visualization.
    """
    print("\n" + "="*70)
    print("CLASSIFICATION SYSTEM - LIVE VISUALIZATION")
    print("="*70)
    
    from airclassifier.geometry.assembly.classification import create_standard_classification_system
    from airclassifier.simulation.simulator import ClassificationSystemSimulator, ClassificationConfig
    
    # Create assembly
    print("\nCreating classification system assembly...")
    assembly = create_standard_classification_system(device="cpu")
    assembly.print_summary()
    
    if not PYVISTA_AVAILABLE:
        print("\nPyVista not available - running simulation without 3D visualization")
        config = ClassificationConfig(dt=1.0e-5, duration=0.5, device="cpu")
        simulator = ClassificationSystemSimulator(assembly, config)
        simulator.run()
        results = simulator.get_results()
        print(f"\nFinal Results: {results}")
        return simulator
    
    # Create plotter
    print("\nInitializing 3D visualization...")
    plotter = pv.Plotter(title="Classification System - Live Simulation")
    plotter.set_background('white')
    plotter.camera.up = (0, 1, 0)
    
    # Add geometry meshes
    print("  Adding Venturi...")
    v, i, _ = assembly.venturi.generate_mesh()
    v = v + assembly._component_positions['venturi']
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['venturi'], label='Venturi Eductor', opacity=0.85)
    
    print("  Adding Zigzag...")
    v, i, _ = assembly.zigzag.generate_mesh()
    v = v + assembly._component_positions['zigzag']
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['zigzag'], label='Zigzag Classifier', opacity=0.85)
    
    print("  Adding Multi-Cyclone...")
    v, i, _ = assembly.multi_cyclone.generate_mesh()
    v = v + assembly._component_positions['multi_cyclone']
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['multi_cyclone'], label='Multi-Cyclone', opacity=0.85)
    
    print("  Adding Bag Filter...")
    v, i, _ = assembly.bag_filter.generate_mesh()
    v = v + assembly._component_positions['bag_filter']
    faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
    mesh = pv.PolyData(v, faces)
    plotter.add_mesh(mesh, color=COLORS['bag_filter'], label='Bag Filter', opacity=0.85)
    
    print("  Adding Ductwork...")
    for idx, (duct, position) in enumerate(assembly._duct_sections):
        v, i, _ = duct.generate_mesh()
        v = v + np.array(position)
        faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
        mesh = pv.PolyData(v, faces)
        plotter.add_mesh(mesh, color=COLORS['duct'],
                        label="Ductwork" if idx == 0 else None, opacity=0.7)
    
    plotter.add_legend(bcolor='white', face='circle')
    plotter.add_axes()
    plotter.reset_camera()
    plotter.camera.azimuth = -170
    plotter.camera.elevation = -20
    
    # Create simulator
    config = ClassificationConfig(
        dt=1.0e-4,
        duration=1.0,
        output_interval=0.05,
        num_particles=2000,
        injection_duration=0.2,
        inlet_velocity=15.0,
        device="cpu",
    )
    simulator = ClassificationSystemSimulator(assembly, config)
    
    # Start the system!
    simulator.start()
    
    print("\n" + "-"*70)
    print("LIVE SIMULATION STARTING")
    print("-"*70)
    print("3D window will show live simulation updates.")
    print("Close the window (press 'q') to stop.")
    print("-"*70)
    
    # Add initial info text (using named actor for smooth updates)
    plotter.add_text(
        "CLASSIFICATION SIMULATION\nStarting...",
        position='upper_left', font_size=11, color='black', name='sim_info'
    )
    
    # Show plotter in interactive mode for live updates
    plotter.show(interactive_update=True, auto_close=False)
    
    # Run simulation with live visualization updates
    total_steps = config.num_steps
    update_interval = max(1, total_steps // 50)
    
    try:
        for step in range(total_steps):
            simulator.step()
            
            if step % update_interval == 0:
                results = simulator.get_results()
                pct = (step / total_steps) * 100
                eff = results['separation_efficiency'] * 100
                
                # Update info text using same name to replace without blinking
                text = (
                    f"CLASSIFICATION SIMULATION\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Particles: {results['particles_injected']}\n"
                    f"Active: {results['particles_active']}\n"
                    f"Coarse: {results['particles_coarse']}\n"
                    f"Fines: {results['particles_fines']}\n"
                    f"Cyclone 1: {results['particles_cyclone_1']}\n"
                    f"Cyclone 2: {results['particles_cyclone_2']}\n"
                    f"Cyclone 3: {results['particles_cyclone_3']}\n"
                    f"Bag Filter: {results['particles_bag_filter']}\n"
                    f"Efficiency: {eff:.1f}%\n"
                    f"Progress: {pct:.0f}%"
                )
                
                plotter.add_text(text, position='upper_left', font_size=11, 
                               color='black', name='sim_info')
                plotter.update()
                
                if step % (update_interval * 5) == 0:
                    print(f"  [{pct:5.1f}%] Particles: {results['particles_injected']:4d} | "
                          f"Coarse: {results['particles_coarse']:4d} | "
                          f"Fines: {results['particles_fines']:4d} | "
                          f"Eff: {eff:.1f}%")
        
        # Get final results
        results = simulator.get_results()
        eff = results['separation_efficiency'] * 100
        
        print("\n" + "-"*70)
        print("SIMULATION COMPLETE")
        print("-"*70)
        print(f"  Particles Injected: {results['particles_injected']}")
        print(f"  Particles Active:   {results['particles_active']}")
        print(f"  Coarse Fraction:    {results['particles_coarse']}")
        print(f"  Fines Fraction:     {results['particles_fines']}")
        print(f"  Cyclone 1:          {results['particles_cyclone_1']}")
        print(f"  Cyclone 2:          {results['particles_cyclone_2']}")
        print(f"  Cyclone 3:          {results['particles_cyclone_3']}")
        print(f"  Bag Filter:         {results['particles_bag_filter']}")
        print(f"  Separation Eff:     {eff:.1f}%")
        print("-"*70)
        
        # Update to final results display
        final_text = (
            f"CLASSIFICATION - COMPLETE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Particles: {results['particles_injected']}\n"
            f"Active: {results['particles_active']}\n"
            f"Coarse: {results['particles_coarse']}\n"
            f"Fines: {results['particles_fines']}\n"
            f"Cyclone 1: {results['particles_cyclone_1']}\n"
            f"Cyclone 2: {results['particles_cyclone_2']}\n"
            f"Cyclone 3: {results['particles_cyclone_3']}\n"
            f"Bag Filter: {results['particles_bag_filter']}\n"
            f"Efficiency: {eff:.1f}%\n"
            f"\n[DONE - Press 'q' to close]"
        )
        plotter.add_text(final_text, position='upper_left', font_size=11, 
                        color='black', name='sim_info')
        plotter.update()
        
        # Keep window responsive until user presses Enter in console
        print("\nSimulation finished!")
        print(">>> Press ENTER in this terminal to close the 3D window <<<")
        
        done_flag = threading.Event()
        
        def wait_for_input():
            input()
            done_flag.set()
        
        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()
        
        while not done_flag.is_set():
            plotter.update()
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        try:
            plotter.close()
        except:
            pass
    
    return simulator


def run_complete_system_live():
    """
    Run Complete System simulation with live 3D visualization.
    """
    print("\n" + "="*70)
    print("COMPLETE SYSTEM - LIVE VISUALIZATION")
    print("="*70)
    
    from airclassifier.geometry.assembly.complete_system import create_core_connections_system
    from airclassifier.simulation.simulator import CompleteSystemSimulator, CompleteSystemConfig
    
    # Create assembly
    print("\nCreating complete system assembly...")
    assembly = create_core_connections_system()
    assembly.print_summary()
    
    if not PYVISTA_AVAILABLE:
        print("\nPyVista not available - running simulation without 3D visualization")
        config = CompleteSystemConfig(dt=1.0e-4, duration=3.0, device="cpu")
        simulator = CompleteSystemSimulator(assembly, config)
        simulator.run()
        results = simulator.get_results()
        print(f"\nFinal Results: {results}")
        return simulator
    
    # Create plotter
    print("\nInitializing 3D visualization...")
    plotter = pv.Plotter(title="Complete System - Live Simulation")
    plotter.set_background('white')
    plotter.camera.up = (0, 1, 0)
    
    # Add subsystem meshes
    for sub_name in assembly.get_all_subsystem_names():
        offset_key = f'{sub_name}_offset'
        offset = np.array(assembly._subsystems.get(offset_key, (0, 0, 0)))
        subsystem = assembly.get_subsystem(sub_name)
        
        if subsystem is not None:
            print(f"  Adding {sub_name}...")
            try:
                v, i = subsystem.build_mesh()
                v = v + offset
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                color = COLORS.get(sub_name, '#808080')
                plotter.add_mesh(mesh, color=color, label=sub_name.replace('_', ' ').title(), opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add {sub_name}: {e}")
    
    # Add components
    for comp_name in assembly.get_all_component_names():
        comp = assembly.get_component(comp_name)
        if comp is not None:
            print(f"  Adding {comp_name}...")
            try:
                v, i, _ = comp.generate_mesh()
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                color = COLORS.get(comp_name, '#808080')
                plotter.add_mesh(mesh, color=color, label=comp_name.replace('_', ' ').title(), opacity=0.85)
            except Exception as e:
                print(f"    Warning: Failed to add {comp_name}: {e}")
    
    # Add duct connections
    if hasattr(assembly, '_duct_connections') and assembly._duct_connections:
        print(f"  Adding {len(assembly._duct_connections)} duct connections...")
        for idx, (duct, position) in enumerate(assembly._duct_connections):
            try:
                v, i, _ = duct.generate_mesh()
                v = v + np.array(position)
                faces = np.hstack([[3] + list(face) for face in i.reshape(-1, 3)])
                mesh = pv.PolyData(v, faces)
                plotter.add_mesh(mesh, color=COLORS['ductwork'],
                                label="Ductwork" if idx == 0 else None, opacity=0.7)
            except Exception as e:
                print(f"    Warning: Failed to add duct {idx}: {e}")
    
    plotter.add_legend(bcolor='white', face='circle')
    plotter.add_axes()
    plotter.add_bounding_box(color='lightgray', opacity=0.1)
    plotter.reset_camera()
    plotter.camera.azimuth = -170
    plotter.camera.elevation = -20
    
    # Create simulator
    config = CompleteSystemConfig(
        dt=1.0e-3,
        duration=5.0,
        output_interval=0.1,
        blower_rpm=3000.0,
        feed_rate_kg_h=500.0,
        num_particles=1000,
        startup_sequence=["air", "classification", "feed"],
        startup_delay=0.5,
        device="cpu",
    )
    simulator = CompleteSystemSimulator(assembly, config)
    
    # Start the system!
    simulator.start()
    
    print("\n" + "-"*70)
    print("LIVE SIMULATION STARTING")
    print("-"*70)
    print("Startup sequence: Air -> Classification -> Feed")
    print("3D window will show live simulation updates.")
    print("Close the window (press 'q') to stop.")
    print("-"*70)
    
    # Add initial info text (using named actor for smooth updates)
    plotter.add_text(
        "COMPLETE SYSTEM SIMULATION\nStarting...",
        position='upper_left', font_size=11, color='black', name='sim_info'
    )
    
    # Show plotter in interactive mode for live updates
    plotter.show(interactive_update=True, auto_close=False)
    
    # Run simulation with live visualization updates
    total_steps = config.num_steps
    update_interval = max(1, total_steps // 50)
    
    try:
        for step in range(total_steps):
            simulator.step()
            
            if step % update_interval == 0:
                results = simulator.get_results()
                pct = (step / total_steps) * 100
                
                # Update info text using same name to replace without blinking
                text = (
                    f"COMPLETE SYSTEM\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"State: {results['system_state']}\n"
                    f"────────────────────────\n"
                    f"AIR SYSTEM\n"
                    f"  Flow: {results['total_flow_rate_m3_h']:.0f} m³/h\n"
                    f"  Power: {results['total_power_kW']:.2f} kW\n"
                    f"────────────────────────\n"
                    f"FEED SYSTEM\n"
                    f"  Feed: {results['feed_rate_kg_h']:.0f} kg/h\n"
                    f"────────────────────────\n"
                    f"Progress: {pct:.0f}%\n"
                    f"Time: {results['time']:.2f} s"
                )
                
                plotter.add_text(text, position='upper_left', font_size=10, 
                               color='black', name='sim_info')
                plotter.update()
                
                if step % (update_interval * 5) == 0:
                    print(f"  [{pct:5.1f}%] State: {results['system_state']:<10} | "
                          f"Flow: {results['total_flow_rate_m3_h']:4.0f} m³/h | "
                          f"Power: {results['total_power_kW']:.2f} kW | "
                          f"Feed: {results['feed_rate_kg_h']:.0f} kg/h")
        
        # Get final results
        results = simulator.get_results()
        
        print("\n" + "-"*70)
        print("SIMULATION COMPLETE")
        print("-"*70)
        print(f"  System State:       {results['system_state']}")
        print(f"  Total Flow Rate:    {results['total_flow_rate_m3_h']:.0f} m³/h")
        print(f"  System Pressure:    {results['system_pressure_Pa']:.0f} Pa")
        print(f"  Total Power:        {results['total_power_kW']:.2f} kW")
        print(f"  Feed Rate:          {results['feed_rate_kg_h']:.0f} kg/h")
        print("-"*70)
        
        # Update to final results display
        final_text = (
            f"COMPLETE SYSTEM - DONE\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"State: {results['system_state']}\n"
            f"\n"
            f"Flow: {results['total_flow_rate_m3_h']:.0f} m³/h\n"
            f"Pressure: {results['system_pressure_Pa']:.0f} Pa\n"
            f"Power: {results['total_power_kW']:.2f} kW\n"
            f"Feed: {results['feed_rate_kg_h']:.0f} kg/h\n"
            f"\n[DONE - Press 'q' to close]"
        )
        plotter.add_text(final_text, position='upper_left', font_size=11, 
                        color='black', name='sim_info')
        plotter.update()
        
        # Keep window responsive until user presses Enter in console
        print("\nSimulation finished!")
        print(">>> Press ENTER in this terminal to close the 3D window <<<")
        
        done_flag = threading.Event()
        
        def wait_for_input():
            input()
            done_flag.set()
        
        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()
        
        while not done_flag.is_set():
            plotter.update()
            time.sleep(0.05)
            
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        try:
            plotter.close()
        except:
            pass
    
    return simulator


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run air classifier simulations with live 3D visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/run_viz_simulation.py --air           # Air system
  python examples/run_viz_simulation.py --feed          # Feed system
  python examples/run_viz_simulation.py --classification # Classification
  python examples/run_viz_simulation.py --complete      # Complete system
  python examples/run_viz_simulation.py --all           # All systems
        """
    )
    
    parser.add_argument("--air", action="store_true",
                        help="Run air system simulation with live visualization")
    parser.add_argument("--feed", action="store_true",
                        help="Run feed system simulation with live visualization")
    parser.add_argument("--classification", "-cls", action="store_true",
                        help="Run classification system simulation with live visualization")
    parser.add_argument("--complete", action="store_true",
                        help="Run complete system simulation with live visualization")
    parser.add_argument("--all", action="store_true",
                        help="Run all system simulations sequentially")
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any([args.air, args.feed, args.classification, args.complete, args.all]):
        parser.print_help()
        print("\n" + "="*70)
        print("No simulation selected. Use one of the options above.")
        print("="*70)
        return
    
    # Check PyVista
    if not PYVISTA_AVAILABLE:
        print("\n" + "="*70)
        print("WARNING: PyVista not installed!")
        print("Install with: pip install pyvista")
        print("Simulations will run without 3D visualization.")
        print("="*70)
    
    results = {}
    
    if args.air or args.all:
        results['air'] = run_air_system_live()
        if args.all:
            input("\nPress Enter to continue to next simulation...")
    
    if args.feed or args.all:
        results['feed'] = run_feed_system_live()
        if args.all:
            input("\nPress Enter to continue to next simulation...")
    
    if args.classification or args.all:
        results['classification'] = run_classification_live()
        if args.all:
            input("\nPress Enter to continue to next simulation...")
    
    if args.complete or args.all:
        results['complete'] = run_complete_system_live()
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()
