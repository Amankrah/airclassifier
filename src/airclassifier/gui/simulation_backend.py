"""
Simulation Backend Integration
==============================

Bridges the GUI with the existing Warp-based simulation code.

This module integrates:
- CompleteClassifierAssembly from complete_system.py (full system mesh)
- ClassificationSystemAssembly from classification.py (core classifier)
- ClassificationFlowSimulation from classification_flow_physics.py (Warp physics)
- Particle materials and populations from particles module

Supports both:
- With preclassification: venturi + zigzag + wheel classifier + cyclones
- Without preclassification (wheel-only): 3-point junction + wheel classifier + cyclones

Material sources:
- yellow_pea: Yellow peas (Pisum sativum)
- faba_bean: Faba beans (Vicia faba)
- oat: Oats (Avena sativa)

Particle fractions:
- whole: Complete flour with all components
- protein: Protein-rich fraction (target for separation)
- starch: Starch granules
- fiber: Fiber/hull fraction
"""

from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np

from PySide6.QtCore import QObject, Signal, QThread


# =============================================================================
# RESOURCE IMPORTS - Lazy-loaded to avoid import errors when deps missing
# =============================================================================

def get_particle_resources():
    """
    Get particle module resources for material definitions and populations.

    Returns dict with:
        - ParticleMaterial: Material class for food powders
        - FluidConfig: Air/fluid configuration
        - create_whole_flour_population: Factory for mixed particle population
        - create_particle_population: Factory for specific fraction
        - MATERIAL_SOURCES: List of available material sources
        - PARTICLE_FRACTIONS: List of available fractions
    """
    try:
        from ..particles import (
            ParticleMaterial,
            FluidConfig,
            ParticlePhysicsConfig,
            create_whole_flour_population,
            create_particle_population,
            create_bimodal_population,
        )
        return {
            "ParticleMaterial": ParticleMaterial,
            "FluidConfig": FluidConfig,
            "ParticlePhysicsConfig": ParticlePhysicsConfig,
            "create_whole_flour_population": create_whole_flour_population,
            "create_particle_population": create_particle_population,
            "create_bimodal_population": create_bimodal_population,
            "MATERIAL_SOURCES": ["yellow_pea", "faba_bean", "oat"],
            "PARTICLE_FRACTIONS": ["whole", "protein", "starch", "fiber"],
        }
    except ImportError as e:
        return {"error": str(e)}


def get_assembly_resources():
    """
    Get assembly module resources for classifier geometry.

    Returns dict with:
        - ClassificationSystemAssembly: Core classifier assembly
        - ClassificationSystemParams: Parameters for classifier
        - CompleteClassifierAssembly: Full system with feed/air
        - CompleteSystemParams: Parameters for complete system
        - create_standard_classification_system: Factory function
    """
    try:
        from ..geometry.assembly.classification import (
            ClassificationSystemAssembly,
            ClassificationSystemParams,
            create_standard_classification_system,
        )
        from ..geometry.assembly.complete_system import (
            CompleteClassifierAssembly,
            CompleteSystemParams,
        )
        return {
            "ClassificationSystemAssembly": ClassificationSystemAssembly,
            "ClassificationSystemParams": ClassificationSystemParams,
            "create_standard_classification_system": create_standard_classification_system,
            "CompleteClassifierAssembly": CompleteClassifierAssembly,
            "CompleteSystemParams": CompleteSystemParams,
        }
    except ImportError as e:
        return {"error": str(e)}


def get_simulation_resources():
    """
    Get simulation module resources for Warp physics.

    Returns dict with:
        - ClassificationFlowSimulation: Main simulation class
        - ClassificationFlowConfig: Simulation configuration
        - ClassificationFlowState: Simulation state
    """
    try:
        from ..simulation.classification_flow_physics import (
            ClassificationFlowSimulation,
            ClassificationFlowConfig,
        )
        return {
            "ClassificationFlowSimulation": ClassificationFlowSimulation,
            "ClassificationFlowConfig": ClassificationFlowConfig,
        }
    except ImportError as e:
        return {"error": str(e)}


# Available material sources and fractions for GUI dropdowns
MATERIAL_SOURCES = ["yellow_pea", "faba_bean", "oat"]
PARTICLE_FRACTIONS = ["whole", "protein", "starch", "fiber"]


@dataclass
class SimulationConfig:
    """Configuration passed from GUI to simulation backend."""
    # Assembly configuration
    assembly_data: Dict[str, Any] = None

    # Time settings
    total_time: float = 10.0
    dt: float = 0.001
    output_interval: float = 0.1

    # Particle settings
    num_particles: int = 5000
    particle_feed_rate: float = 1000.0
    continuous_feeding: bool = True

    # Material
    material_source: str = "yellow_pea"
    material_fraction: str = "whole"

    # Physics (turbulence base scales zone-specific: zigzag=0.25, cyclone=0.12 at base=0.15)
    turbulence_base: float = 0.15
    restitution: float = 0.3
    friction: float = 0.4

    # Compute
    device: str = "cuda"

    # Assembly mode
    use_preclassification: bool = True  # False = wheel-only mode

    # Complete system options
    include_feed_system: bool = True
    include_air_system: bool = True
    include_exhaust: bool = True

    # Wheel classifier parameters (optimized: 975 RPM, d50≈36 µm)
    wheel_diameter: float = 0.20
    wheel_rpm: float = 975.0
    wheel_target_d50: float = 25e-6  # 25 µm

    # Zigzag parameters (when use_preclassification=True)
    zigzag_channel_width: float = 0.15
    zigzag_num_stages: int = 5
    zigzag_channel_depth: float = 0.25

    # Venturi parameters (when use_preclassification=True)
    venturi_inlet_diameter: float = 0.08
    venturi_throat_ratio: float = 0.5

    # Cyclone parameters
    primary_cyclone_diameter: float = 0.30
    secondary_cyclone_diameter: float = 0.20
    tertiary_cyclone_diameter: float = 0.12


class SimulationBackend(QObject):
    """
    Backend that runs the air classifier simulation using NVIDIA Warp.

    This class bridges the GUI components with the existing simulation
    code in airclassifier.simulation.classification_flow_physics.

    Supports two assembly modes:
    - With preclassification (use_preclassification=True):
      venturi eductor + zigzag classifier + wheel classifier + cyclones + bag filter

    - Without preclassification / wheel-only (use_preclassification=False):
      3-point junction (air inlet + solids chute) -> wheel classifier -> cyclones + bag filter
    """

    # Signals for communication with GUI
    progress_updated = Signal(int, float, dict)  # (percent, time, stats)
    particles_updated = Signal(object, object)   # (positions, velocities)
    component_state_updated = Signal(dict)       # physics-driven component states per frame
    simulation_completed = Signal(dict)          # (results)
    simulation_error = Signal(str)               # (error_message)
    log_message = Signal(str)                    # (message)
    mesh_updated = Signal(object, object)        # (vertices, indices) for 3D viewport

    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config

        self._is_running = False
        self._is_paused = False
        self._sim = None
        self._assembly = None  # ClassificationSystemAssembly (core classifier)
        self._complete_assembly = None  # CompleteClassifierAssembly (full system)

    def setup(self) -> bool:
        """
        Setup simulation from assembly configuration.

        Creates assemblies and simulation objects based on GUI configuration.
        Supports both with-preclassification and wheel-only modes.

        Returns:
            True if setup succeeded, False otherwise
        """
        self.log_message.emit("Setting up simulation...")

        try:
            # Build assembly from GUI configuration
            self._build_assembly_from_gui()

            # Import simulation module
            from ..simulation.classification_flow_physics import (
                ClassificationFlowSimulation,
                ClassificationFlowConfig,
            )

            # Create simulation config
            _ts = self.config.turbulence_base / 0.15  # scale from base
            sim_config = ClassificationFlowConfig(
                num_particles=self.config.num_particles,
                dt=self.config.dt,
                continuous_feeding=self.config.continuous_feeding,
                particle_feed_rate=self.config.particle_feed_rate,
                turbulence_zigzag=0.25 * _ts,
                turbulence_cyclone=0.12 * _ts,
                restitution=self.config.restitution,
                friction=self.config.friction,
                device=self.config.device,
            )

            # Create simulation using the classification assembly
            self._sim = ClassificationFlowSimulation(
                assembly=self._assembly,
                config=sim_config,
            )

            # Emit mesh for 3D visualization
            self._emit_mesh()

            mode = "with preclassification" if self.config.use_preclassification else "wheel-only"
            self.log_message.emit(f"Simulation setup complete ({mode} mode)")
            return True

        except ImportError as e:
            self.simulation_error.emit(f"Import error: {e}\nMake sure all dependencies are installed.")
            return False
        except Exception as e:
            import traceback
            self.simulation_error.emit(f"Setup error: {e}\n{traceback.format_exc()}")
            return False

    def _build_assembly_from_gui(self):
        """
        Build assemblies from GUI node graph configuration.

        Creates both:
        - ClassificationSystemAssembly (for simulation physics)
        - CompleteClassifierAssembly (for full 3D visualization with feed/air systems)
        """
        from ..geometry.assembly.classification import (
            ClassificationSystemAssembly,
            ClassificationSystemParams,
            create_standard_classification_system,
        )
        from ..geometry.assembly.complete_system import (
            CompleteClassifierAssembly,
            CompleteSystemParams,
        )

        assembly_data = self.config.assembly_data or {}
        components = assembly_data.get("components", {})

        # Build ClassificationSystemParams from GUI or config
        class_params = ClassificationSystemParams(
            use_preclassification=self.config.use_preclassification,
            wheel_diameter=self.config.wheel_diameter,
            wheel_rpm=self.config.wheel_rpm,
            wheel_target_d50=self.config.wheel_target_d50,
            zigzag_channel_width=self.config.zigzag_channel_width,
            zigzag_num_stages=self.config.zigzag_num_stages,
            zigzag_channel_depth=self.config.zigzag_channel_depth,
            venturi_inlet_diameter=self.config.venturi_inlet_diameter,
            venturi_throat_ratio=self.config.venturi_throat_ratio,
            primary_cyclone_diameter=self.config.primary_cyclone_diameter,
            secondary_cyclone_diameter=self.config.secondary_cyclone_diameter,
            tertiary_cyclone_diameter=self.config.tertiary_cyclone_diameter,
        )

        # Override with GUI component parameters if present
        for comp_id, comp_data in components.items():
            comp_type = comp_data.get("type", "")
            comp_params = comp_data.get("params", {})

            if comp_type == "Venturi Eductor":
                class_params.venturi_inlet_diameter = comp_params.get("inlet_diameter", class_params.venturi_inlet_diameter)
                class_params.venturi_throat_ratio = comp_params.get("throat_ratio", class_params.venturi_throat_ratio)

            elif comp_type == "Zigzag Classifier":
                class_params.zigzag_channel_width = comp_params.get("channel_width", class_params.zigzag_channel_width)
                class_params.zigzag_num_stages = comp_params.get("num_stages", class_params.zigzag_num_stages)
                class_params.zigzag_channel_depth = comp_params.get("channel_depth", class_params.zigzag_channel_depth)

            elif comp_type == "Wheel Classifier":
                class_params.wheel_diameter = comp_params.get("wheel_diameter", class_params.wheel_diameter)
                class_params.wheel_rpm = comp_params.get("wheel_rpm", class_params.wheel_rpm)
                class_params.wheel_num_blades = comp_params.get("num_blades", class_params.wheel_num_blades)

            elif "Cyclone" in comp_type:
                if "Primary" in comp_type:
                    class_params.primary_cyclone_diameter = comp_params.get("diameter", class_params.primary_cyclone_diameter)
                elif "Secondary" in comp_type:
                    class_params.secondary_cyclone_diameter = comp_params.get("diameter", class_params.secondary_cyclone_diameter)
                elif "Tertiary" in comp_type:
                    class_params.tertiary_cyclone_diameter = comp_params.get("diameter", class_params.tertiary_cyclone_diameter)

            elif comp_type == "Bag Filter":
                class_params.bag_filter_flow_rate = comp_params.get("flow_rate", class_params.bag_filter_flow_rate)

        # Create core classification assembly (for physics simulation)
        self._assembly = create_standard_classification_system(
            device=self.config.device,
            params=class_params,
        )

        # Build complete system assembly (for 3D visualization)
        complete_params = CompleteSystemParams(
            classification_params=class_params,
            wheel_diameter=self.config.wheel_diameter,
            wheel_rpm=self.config.wheel_rpm,
            wheel_target_d50=self.config.wheel_target_d50,
            include_feed_system=self.config.include_feed_system,
            include_air_system=self.config.include_air_system,
            include_exhaust=self.config.include_exhaust,
            include_ductwork=True,
        )

        try:
            self._complete_assembly = CompleteClassifierAssembly(complete_params)
            summary = self._complete_assembly.get_system_summary()
            self.log_message.emit(f"Complete assembly built: {summary.get('num_subsystems', '?')} subsystems")
        except Exception as e:
            import traceback
            self.log_message.emit(f"WARNING: CompleteClassifierAssembly failed: {e}")
            self.log_message.emit(f"  {traceback.format_exc()}")
            self.log_message.emit("  Falling back to classification-only mesh.")
            self._complete_assembly = None

        mode = "with preclassification" if self.config.use_preclassification else "wheel-only"
        self.log_message.emit(f"Classification assembly created ({mode}, {len(components)} GUI components)")

    def create_subsidiary_simulators(self) -> Dict[str, Any]:
        """
        Create lightweight air/feed physics simulators from the complete assembly.

        These provide real physics state (blower VFD ramp, damper positions,
        lid servo angle) for driving the full-system animation -- both during
        simulation runs AND during build-time preview animation.

        Returns:
            Dict with keys 'air_sim' and 'feed_sim' (either may be None).
        """
        result = {"air_sim": None, "feed_sim": None}
        if self._complete_assembly is None:
            return result

        # --- Air Flow Physics Simulator (lightweight: no SPH) ---
        if self.config.include_air_system:
            try:
                from ..simulation.air_flow_physics import (
                    AirFlowPhysicsSimulator,
                    AirFlowPhysicsConfig,
                )
                air_assembly = self._complete_assembly.get_subsystem("air_system")
                if air_assembly is not None:
                    blower_rpm = getattr(self.config, 'blower_rpm', 700.0)
                    # Use airclass operating point if blower_rpm is set
                    if blower_rpm > 0:
                        try:
                            from ..simulation.airclass_flow_physics import compute_blower_operating_point
                            op = compute_blower_operating_point(blower_rpm)
                            blower_rpm = op.get("rpm", blower_rpm)
                        except Exception:
                            pass
                    air_config = AirFlowPhysicsConfig(
                        target_rpm=blower_rpm,
                        dt=0.01,
                        total_time=60.0,
                        ramp_time=2.0,
                        damper_ramp_time=2.0,
                        enable_sph=False,
                        device="cpu",
                    )
                    air_sim = AirFlowPhysicsSimulator(air_assembly, air_config)
                    air_sim.start_system()
                    result["air_sim"] = air_sim
            except Exception as e:
                print(f"  Subsidiary air sim failed: {e}")

        # --- Feed Flow Physics Simulator (lightweight: lid only) ---
        if self.config.include_feed_system:
            try:
                from ..simulation.feed_flow_physics import (
                    FeedFlowPhysicsSimulator,
                    FlowPhysicsConfig,
                )
                feed_assembly = self._complete_assembly.get_subsystem("feed_system")
                if feed_assembly is not None:
                    feed_config = FlowPhysicsConfig(
                        dt=0.01,
                        total_time=60.0,
                        animate_lid=True,
                        lid_open_angle=90.0,
                        lid_animation_time=2.0,
                        enable_pouring=False,
                        num_particles=0,
                        device="cpu",
                    )
                    feed_sim = FeedFlowPhysicsSimulator(feed_assembly, feed_config)
                    result["feed_sim"] = feed_sim
            except Exception as e:
                print(f"  Subsidiary feed sim failed: {e}")

        return result

    def _create_particle_population(self) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create particle population based on material settings.

        Returns:
            Tuple of (material, diameters, densities, sphericities, types)
        """
        resources = get_particle_resources()
        if "error" in resources:
            self.log_message.emit(f"Warning: Particle resources not available: {resources['error']}")
            return None, None, None, None, None

        source = self.config.material_source
        fraction = self.config.material_fraction
        num_particles = self.config.num_particles

        try:
            if fraction == "whole":
                # Use mixed population factory
                create_pop = resources["create_whole_flour_population"]
                material, diameters, densities, sphericities, types = create_pop(
                    source=source,
                    num_particles=num_particles,
                )
            else:
                # Use specific fraction factory
                create_pop = resources["create_particle_population"]
                material, diameters, densities, sphericities, types = create_pop(
                    source=source,
                    fraction=fraction,
                    num_particles=num_particles,
                )

            self.log_message.emit(
                f"Particle population: {num_particles} particles, "
                f"source={source}, fraction={fraction}, "
                f"d_mean={np.mean(diameters)*1e6:.1f}µm"
            )
            return material, diameters, densities, sphericities, types

        except Exception as e:
            self.log_message.emit(f"Warning: Could not create particle population: {e}")
            return None, None, None, None, None

    def get_fluid_config(self):
        """
        Get fluid (air) configuration for simulation.

        Returns:
            FluidConfig for air at STP, or None if not available
        """
        resources = get_particle_resources()
        if "error" in resources:
            return None

        FluidConfig = resources["FluidConfig"]
        return FluidConfig.air_at_stp()

    def get_material_info(self, source: str = None) -> Dict[str, Any]:
        """
        Get information about a material source.

        Args:
            source: Material source name (default: config.material_source)

        Returns:
            Dictionary with material properties
        """
        resources = get_particle_resources()
        if "error" in resources:
            return {"error": resources["error"]}

        source = source or self.config.material_source
        ParticleMaterial = resources["ParticleMaterial"]

        try:
            material = ParticleMaterial.create_food_powder(source, "whole")
            return {
                "source": source,
                "density_range": (material.protein.density, material.starch.density),
                "size_range_um": (
                    material.protein.size_distribution.d10 * 1e6,
                    material.starch.size_distribution.d90 * 1e6,
                ),
                "fractions": ["whole", "protein", "starch", "fiber"],
            }
        except Exception as e:
            return {"error": str(e)}

    def _emit_mesh(self):
        """Emit mesh data for 3D visualization."""
        try:
            # Prefer complete assembly for full system visualization
            if self._complete_assembly is not None:
                vertices, indices = self._complete_assembly.build_mesh()
                self.mesh_updated.emit(vertices, indices)
                self.log_message.emit(f"Mesh: {len(vertices)} vertices, {len(indices)//3} triangles")
            elif self._assembly is not None:
                # Fallback to classification-only mesh
                vertices, indices = self._assembly.build_mesh()
                self.mesh_updated.emit(vertices, indices)
                self.log_message.emit(f"Mesh (classifier only): {len(vertices)} vertices")
        except Exception as e:
            self.log_message.emit(f"Warning: Could not emit mesh: {e}")

    def get_mesh(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get mesh data for 3D visualization.

        Returns:
            Tuple of (vertices, indices) or (None, None) if not available
        """
        try:
            if self._complete_assembly is not None:
                return self._complete_assembly.build_mesh()
            elif self._assembly is not None:
                return self._assembly.build_mesh()
        except Exception:
            pass
        return None, None

    def get_assembly_mode(self) -> str:
        """Get current assembly mode description."""
        if self.config.use_preclassification:
            return "Full system (venturi + zigzag + wheel classifier)"
        else:
            return "Wheel-only (3-point junction + wheel classifier)"

    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the assembled system."""
        summary = {
            "mode": self.get_assembly_mode(),
            "use_preclassification": self.config.use_preclassification,
            "wheel_rpm": self.config.wheel_rpm,
            "wheel_diameter": self.config.wheel_diameter,
        }

        if self._complete_assembly is not None:
            summary.update(self._complete_assembly.get_system_summary())
        elif self._assembly is not None:
            try:
                bounds_min, bounds_max = self._assembly.get_bounds()
                summary["bounds_min"] = list(bounds_min)
                summary["bounds_max"] = list(bounds_max)
            except Exception:
                pass

        return summary

    def run(self):
        """Run the simulation loop."""
        if not self._sim:
            self.simulation_error.emit("Simulation not setup. Call setup() first.")
            return

        self._is_running = True
        self.log_message.emit("Starting simulation...")

        try:
            total_steps = int(self.config.total_time / self.config.dt)
            output_steps = max(1, int(self.config.output_interval / self.config.dt))

            for step in range(total_steps):
                # Check stop/pause
                if not self._is_running:
                    break

                while self._is_paused and self._is_running:
                    QThread.msleep(100)

                # Step simulation
                self._sim.step()

                # Emit progress at intervals
                if step % output_steps == 0:
                    progress = int(100 * step / total_steps)
                    current_time = step * self.config.dt

                    # Get statistics
                    stats = self._get_stats()
                    self.progress_updated.emit(progress, current_time, stats)

                    # Get particle data for visualization
                    positions, velocities = self._get_particle_data()
                    if positions is not None:
                        self.particles_updated.emit(positions, velocities)

                    # Emit physics-driven component states for animation
                    comp_state = self._get_component_states(current_time)
                    if comp_state:
                        self.component_state_updated.emit(comp_state)

            # Simulation complete
            results = self._finalize()
            self.simulation_completed.emit(results)
            self.log_message.emit("Simulation completed successfully")

        except Exception as e:
            self.simulation_error.emit(f"Simulation error: {e}")

    def _get_stats(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        if not self._sim:
            return {}

        state = getattr(self._sim, 'state', None)
        if state is None:
            return {}

        return {
            "active_particles": getattr(state, 'particles_active', 0),
            "collected_fines": getattr(state, 'collected_fines', 0),
            "collected_coarse": getattr(state, 'collected_coarse', 0),
            "collected_cyclone": getattr(state, 'collected_cyclone', {}),
            "collected_bagfilter": getattr(state, 'collected_bagfilter', 0),
            "simulation_time": getattr(state, 'time', 0),
            "simulation_step": getattr(state, 'step', 0),
        }

    def _get_particle_data(self):
        """Get particle positions and velocities for visualization."""
        if not self._sim:
            return None, None

        state = getattr(self._sim, 'state', None)
        if state is None:
            return None, None

        positions = getattr(state, 'positions', None)
        velocities = getattr(state, 'velocities', None)

        if positions is not None:
            # Convert from Warp arrays to numpy
            try:
                positions = positions.numpy()
                if velocities is not None:
                    velocities = velocities.numpy()
                return positions, velocities
            except:
                return None, None

        return None, None

    def _get_component_states(self, sim_time: float) -> Dict[str, Any]:
        """
        Extract physics-driven component states from the running simulation.

        Uses the ClassificationFlowPhysicsSimulator wheel_omega and the
        blower operating point from airclass_flow_physics. Phase timing
        follows the orchestration in air_flow_physics / feed_flow_physics.

        Returns a dict that AnimationController.update_from_physics() uses
        to drive the rendered full-system assembly directly.
        """
        import math
        TWO_PI = 2.0 * math.pi
        component_state = {"sim_time": float(sim_time)}

        # Wheel -- from ClassificationFlowPhysicsSimulator
        sim = self._sim
        if sim is not None:
            wheel_omega = getattr(sim, 'wheel_omega', 0.0)
            component_state["wheel_omega"] = float(wheel_omega)
            component_state["wheel_angle_rad"] = float(wheel_omega * sim_time) % TWO_PI
            state = getattr(sim, 'state', None)
            if state is not None:
                phase = getattr(state, 'phase', None)
                component_state["phase"] = phase.value if phase else "running"

        # During classification, preamble has already completed.
        # All systems at steady state -- open, running, full speed.
        # They only close/stop during the shutdown sequence after sim ends.
        blower_rpm = getattr(self.config, 'blower_rpm', 3000.0)
        component_state["blower_rpm"] = float(blower_rpm)
        component_state["blower_ramp_frac"] = 1.0
        component_state["damper_positions"] = [1.0, 1.0]  # fully open
        component_state["lid_angle_deg"] = 90.0             # fully open
        component_state["feed_ramp_frac"] = 1.0
        component_state["classification_ramp_frac"] = 1.0

        return component_state

    def _finalize(self) -> Dict[str, Any]:
        """Finalize simulation and compute results."""
        results = {
            "total_particles": self.config.num_particles,
            "simulation_time": self.config.total_time,
            "separation_efficiency": 0.0,
            "protein_recovery": 0.0,
            "protein_purity": 0.0,
            "cut_sizes": {},
            "mass_balance": {},
            "grade_efficiency": {},
            "collection_summary": {},
        }

        if self._sim:
            # Get final statistics
            stats = self._get_stats()
            results.update(stats)

            # Compute separation metrics
            total_collected = (
                stats.get("collected_fines", 0) +
                stats.get("collected_coarse", 0) +
                stats.get("collected_bagfilter", 0) +
                sum(stats.get("collected_cyclone", {}).values())
            )

            if total_collected > 0:
                results["separation_efficiency"] = 100.0 * stats.get("collected_fines", 0) / total_collected

        return results

    def pause(self):
        """Pause the simulation."""
        self._is_paused = True
        self.log_message.emit("Simulation paused")

    def resume(self):
        """Resume the simulation."""
        self._is_paused = False
        self.log_message.emit("Simulation resumed")

    def stop(self):
        """Stop the simulation."""
        self._is_running = False
        self._is_paused = False
        self.log_message.emit("Simulation stopped")

    def cleanup(self):
        """Clean up resources."""
        self._sim = None
        self._assembly = None


def create_mesh_from_assembly(assembly, complete_assembly=None) -> Dict[str, Any]:
    """
    Extract mesh data from assemblies for visualization.

    Supports both:
    - ClassificationSystemAssembly (classification components only)
    - CompleteClassifierAssembly (full system with feed/air/exhaust)

    Args:
        assembly: ClassificationSystemAssembly instance
        complete_assembly: Optional CompleteClassifierAssembly for full visualization

    Returns:
        Dictionary mapping component names to mesh data (vertices, faces, color)
    """
    meshes = {}

    # Prefer complete assembly for full visualization
    target_assembly = complete_assembly if complete_assembly is not None else assembly

    if target_assembly is None:
        return meshes

    try:
        # Get combined mesh using build_mesh()
        vertices, indices = target_assembly.build_mesh()

        if vertices is not None and len(vertices) > 0 and indices is not None and len(indices) > 0:
            meshes["combined"] = {
                "vertices": np.array(vertices),
                "faces": np.array(indices).reshape(-1, 3),
                "color": "#4ec9b0",
            }

        # Try to get individual component meshes for highlighting
        if hasattr(target_assembly, 'get_component_positions'):
            component_positions = target_assembly.get_component_positions()

            for name, pos in component_positions.items():
                # Handle different attribute naming conventions
                attr_name = name.replace('-', '_')
                component = getattr(target_assembly, attr_name, None)

                if component is None:
                    continue

                # Try generate_mesh() first, then get_mesh()
                try:
                    if hasattr(component, 'generate_mesh'):
                        verts, inds, _ = component.generate_mesh()
                    elif hasattr(component, 'get_mesh'):
                        verts, inds = component.get_mesh()
                    else:
                        continue

                    if verts is not None and inds is not None and len(verts) > 0:
                        # Offset vertices by component position
                        verts = np.array(verts) + np.array(pos)
                        meshes[name] = {
                            "vertices": verts,
                            "faces": np.array(inds).reshape(-1, 3),
                            "color": _get_component_color(name),
                        }
                except Exception:
                    pass

        # For complete assembly, also get subsystem meshes
        if complete_assembly is not None:
            try:
                for subsystem_name in complete_assembly.get_all_subsystem_names():
                    subsystem = complete_assembly.get_subsystem(subsystem_name)
                    if subsystem is not None and hasattr(subsystem, 'build_mesh'):
                        try:
                            verts, inds = subsystem.build_mesh()
                            offset_key = f'{subsystem_name}_offset'
                            if hasattr(complete_assembly, '_subsystems'):
                                offset = complete_assembly._subsystems.get(offset_key, (0, 0, 0))
                                verts = np.array(verts) + np.array(offset)

                            meshes[subsystem_name] = {
                                "vertices": verts,
                                "faces": np.array(inds).reshape(-1, 3),
                                "color": _get_subsystem_color(subsystem_name),
                            }
                        except Exception:
                            pass
            except Exception:
                pass

    except Exception as e:
        print(f"Error extracting meshes: {e}")

    return meshes


def _get_component_color(name: str) -> str:
    """Get color for component based on name."""
    name_lower = name.lower()
    if "venturi" in name_lower:
        return "#6495ED"  # Cornflower blue
    elif "zigzag" in name_lower:
        return "#90EE90"  # Light green
    elif "wheel" in name_lower:
        return "#FFB6C1"  # Light pink
    elif "cyclone" in name_lower:
        return "#98FB98"  # Pale green
    elif "bag" in name_lower or "filter" in name_lower:
        return "#DDA0DD"  # Plum
    elif "duct" in name_lower or "elbow" in name_lower:
        return "#C0C0C0"  # Silver
    elif "junction" in name_lower:
        return "#FFA500"  # Orange
    elif "transition" in name_lower:
        return "#87CEEB"  # Sky blue
    else:
        return "#4ec9b0"  # Default teal


def _get_subsystem_color(name: str) -> str:
    """Get color for subsystem based on name."""
    name_lower = name.lower()
    if "feed" in name_lower:
        return "#DAA520"  # Goldenrod
    elif "air" in name_lower:
        return "#87CEEB"  # Sky blue
    elif "classification" in name_lower:
        return "#4ec9b0"  # Teal
    elif "exhaust" in name_lower:
        return "#A9A9A9"  # Dark gray
    else:
        return "#B0C4DE"  # Light steel blue


def check_resources() -> Dict[str, Any]:
    """
    Check availability of all required resources for simulation.

    Returns dict with:
        - particles: bool - Particle module available
        - assemblies: bool - Assembly module available
        - simulation: bool - Simulation/Warp module available
        - warp: bool - NVIDIA Warp available
        - cuda: bool - CUDA device available
        - errors: list - List of error messages
        - warnings: list - List of warning messages
    """
    result = {
        "particles": False,
        "assemblies": False,
        "simulation": False,
        "warp": False,
        "cuda": False,
        "errors": [],
        "warnings": [],
    }

    # Check particle resources
    particle_res = get_particle_resources()
    if "error" not in particle_res:
        result["particles"] = True
    else:
        result["errors"].append(f"Particles: {particle_res['error']}")

    # Check assembly resources
    assembly_res = get_assembly_resources()
    if "error" not in assembly_res:
        result["assemblies"] = True
    else:
        result["errors"].append(f"Assemblies: {assembly_res['error']}")

    # Check simulation resources
    sim_res = get_simulation_resources()
    if "error" not in sim_res:
        result["simulation"] = True
    else:
        result["errors"].append(f"Simulation: {sim_res['error']}")

    # Check Warp and CUDA
    try:
        import warp as wp
        result["warp"] = True
        wp.init()
        devices = wp.get_devices()
        cuda_devices = [d for d in devices if "cuda" in str(d).lower()]
        if cuda_devices:
            result["cuda"] = True
        else:
            result["warnings"].append("No CUDA devices found - will use CPU")
    except ImportError:
        result["errors"].append("Warp: pip install warp-lang")
    except Exception as e:
        result["warnings"].append(f"Warp init warning: {e}")

    return result


def get_available_materials() -> List[str]:
    """Get list of available material sources."""
    return MATERIAL_SOURCES.copy()


def get_available_fractions() -> List[str]:
    """Get list of available particle fractions."""
    return PARTICLE_FRACTIONS.copy()


def create_config_from_preset(preset_data: Dict[str, Any]) -> SimulationConfig:
    """
    Create SimulationConfig from a preset configuration.

    Args:
        preset_data: Preset dictionary with assembly and parameter info

    Returns:
        SimulationConfig configured from the preset
    """
    config = SimulationConfig()
    assembly_data = preset_data.get("assembly", {})
    config.assembly_data = assembly_data

    # Check for explicit use_preclassification flag in assembly data
    if "use_preclassification" in assembly_data:
        config.use_preclassification = assembly_data["use_preclassification"]
    else:
        # Fallback: detect mode from components
        components = assembly_data.get("components", {})
        has_venturi = any("Venturi" in c.get("type", "") for c in components.values())
        has_zigzag = any("Zigzag" in c.get("type", "") for c in components.values())
        config.use_preclassification = has_venturi or has_zigzag

    # Extract component parameters
    components = assembly_data.get("components", {})
    for comp_id, comp_data in components.items():
        comp_type = comp_data.get("type", "")
        comp_params = comp_data.get("params", {})

        if comp_type == "Wheel Classifier":
            config.wheel_diameter = comp_params.get("wheel_diameter", config.wheel_diameter)
            config.wheel_rpm = comp_params.get("wheel_rpm", config.wheel_rpm)

        elif comp_type == "Venturi Eductor":
            config.venturi_inlet_diameter = comp_params.get("inlet_diameter", config.venturi_inlet_diameter)
            config.venturi_throat_ratio = comp_params.get("throat_ratio", config.venturi_throat_ratio)

        elif comp_type == "Zigzag Classifier":
            config.zigzag_channel_width = comp_params.get("channel_width", config.zigzag_channel_width)
            config.zigzag_num_stages = comp_params.get("num_stages", config.zigzag_num_stages)

    return config
