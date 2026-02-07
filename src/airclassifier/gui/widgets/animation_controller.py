"""
Animation Controller
====================

Drives synchronized mechanical animations using the actual component
animation APIs from the geometry codebase.

Each component has its own animation method:
  - CentrifugalBlower:  update_animation(dt, rpm) -> get_impeller_mesh(), get_driven_pulley_mesh(), get_motor_pulley_mesh()
  - FlowDamper:         update_animation(dt, target, time) -> get_blade_mesh(position)
  - RotaryAirlock:      update_rotation(dt, rpm) -> get_rotor_mesh(angle)
  - ScrewFeeder:        update_rotation(dt, rpm) -> get_screw_mesh(angle)
  - Deagglomerator:     update_rotation(dt, rpm) -> get_rotor_mesh(angle)
  - WheelClassifier:    omega property -> rotate full mesh by omega * t

Animation Sequence (matching physics orchestration):
1. AIR_STARTUP   t=0s   Blower spins up, dampers open
2. FEED_STARTUP  t=3s   Airlock, screw, deagglomerator start
3. CLASSIFICATION t=5s  Wheel classifier spinning
4. STEADY_STATE  t=8s   Everything at design speed
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
import time
import math
import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

TWO_PI = 2.0 * math.pi


class AnimationPhase(Enum):
    IDLE = "idle"
    AIR_STARTUP = "air_startup"
    FEED_STARTUP = "feed_startup"
    CLASSIFICATION = "classification"
    STEADY_STATE = "steady_state"
    SHUTDOWN = "shutdown"


@dataclass
class AnimationTimeline:
    """Phase timing in seconds of animation time."""
    air_start_time: float = 0.0
    air_ramp_duration: float = 2.0
    feed_start_time: float = 3.0
    feed_ramp_duration: float = 2.0
    classification_start_time: float = 5.0
    classification_ramp_duration: float = 2.0
    steady_time: float = 8.0


@dataclass
class AnimatedPart:
    """One animated sub-mesh in the viewport."""
    name: str
    actor: Any = None               # Current PyVista actor
    get_mesh: Callable = None        # () -> pv.PolyData  (called each frame)
    update: Callable = None          # (dt, frac) -> None  (called each frame)
    color: str = "#4A90D9"
    phase: str = "air"               # "air", "feed", "classification"
    active: bool = False


class AnimationController(QObject):
    """
    Drives component animations using their native APIs.

    Supports two modes:
      1. Timeline mode (default): fixed AnimationTimeline drives ramp fractions.
      2. Physics mode: simulation backend emits component_state dicts that
         override timeline fractions with actual physics values (wheel angle,
         damper position, lid angle, etc.).

    Usage:
        ctrl = AnimationController(plotter)
        ctrl.register_blower(blower, position, rpm)
        ctrl.register_damper(damper, position)
        ctrl.register_airlock(airlock, position, rpm)
        ...
        ctrl.start()

        # During simulation, call from the backend signal:
        ctrl.update_from_physics(component_state_dict)
    """

    phase_changed = Signal(str)
    FRAME_MS = 33  # ~30 fps

    def __init__(self, plotter=None, parent=None):
        super().__init__(parent)
        self._plotter = plotter
        self._parts: Dict[str, AnimatedPart] = {}
        self._timeline = AnimationTimeline()
        self._phase = AnimationPhase.IDLE
        self._anim_time = 0.0
        self._running = False

        # Physics-driven state (when available, overrides timeline)
        self._physics_state: Optional[Dict[str, Any]] = None
        self._physics_mode = False  # True when physics state is being received

        # Subsidiary simulators for autonomous physics-driven animation
        # (used during build-time preview when no simulation is running)
        self._air_sim = None   # AirFlowPhysicsSimulator
        self._feed_sim = None  # FeedFlowPhysicsSimulator
        self._feed_start_time = 3.0  # when feed system starts [s]

        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._tick)

    @property
    def phase(self) -> AnimationPhase:
        return self._phase

    def set_plotter(self, plotter):
        self._plotter = plotter

    def set_subsidiary_simulators(self, air_sim=None, feed_sim=None):
        """
        Attach lightweight air/feed physics simulators for autonomous animation.

        When set, the controller steps these simulators in its own _tick() loop
        and reads their live state (blower ramp, damper positions, lid angle)
        instead of using the fixed timeline. This works both during build-time
        preview AND during simulation (though during simulation the
        SimulationWorker's component_state_updated signal takes priority).

        Args:
            air_sim: AirFlowPhysicsSimulator (enable_sph=False) or None
            feed_sim: FeedFlowPhysicsSimulator (num_particles=0) or None
        """
        self._air_sim = air_sim
        self._feed_sim = feed_sim

    # ================================================================
    # Registration -- uses actual component animation APIs
    # ================================================================

    def register_blower(self, blower, world_offset: np.ndarray, target_rpm: float, color: str = "#27AE60"):
        """Register a CentrifugalBlower for impeller + pulley animation."""
        if not HAS_PYVISTA or blower is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)

        def update_fn(dt, frac):
            rpm = target_rpm * frac
            blower.update_animation(dt, rpm)

        def mesh_fn():
            meshes = []
            for getter in [blower.get_impeller_mesh, blower.get_driven_pulley_mesh, blower.get_motor_pulley_mesh]:
                try:
                    v, i, _ = getter()
                    if v is not None and len(v) > 0:
                        verts = v.astype(np.float64) + offset
                        meshes.append(self._make_pv_mesh(verts, i))
                except Exception:
                    pass
            if meshes:
                combined = meshes[0]
                for m in meshes[1:]:
                    combined = combined.merge(m)
                return combined
            return None

        self._parts["blower_impeller"] = AnimatedPart(
            name="blower_impeller", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="air",
        )

    def register_damper(self, damper, world_offset: np.ndarray, index: int = 0, color: str = "#B0BEC5"):
        """Register a FlowDamper for blade open/close animation."""
        if not HAS_PYVISTA or damper is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)

        def update_fn(dt, frac):
            damper.update_animation(dt, target_position=frac, transition_time=1.0)

        def mesh_fn():
            try:
                v, i, _ = damper.get_blade_mesh()
                if v is not None and len(v) > 0:
                    verts = v.astype(np.float64) + offset
                    return self._make_pv_mesh(verts, i)
            except Exception:
                pass
            return None

        self._parts[f"damper_{index}"] = AnimatedPart(
            name=f"damper_{index}", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="air",
        )

    def register_airlock(self, airlock, world_offset: np.ndarray, target_rpm: float,
                         name: str = "airlock", phase: str = "feed", color: str = "#3498DB"):
        """Register a RotaryAirlock for rotor animation."""
        if not HAS_PYVISTA or airlock is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)

        def update_fn(dt, frac):
            airlock.update_rotation(dt, target_rpm * frac)

        def mesh_fn():
            try:
                v, i, _ = airlock.get_rotor_mesh()
                if v is not None and len(v) > 0:
                    verts = v.astype(np.float64) + offset
                    return self._make_pv_mesh(verts, i)
            except Exception:
                pass
            return None

        self._parts[name] = AnimatedPart(
            name=name, get_mesh=mesh_fn, update=update_fn,
            color=color, phase=phase,
        )

    def register_screw(self, screw, world_offset: np.ndarray, target_rpm: float, color: str = "#27AE60"):
        """Register a ScrewFeeder for auger rotation animation."""
        if not HAS_PYVISTA or screw is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)

        def update_fn(dt, frac):
            screw.update_rotation(dt, target_rpm * frac)

        def mesh_fn():
            try:
                v, i, _ = screw.get_screw_mesh()
                if v is not None and len(v) > 0:
                    verts = v.astype(np.float64) + offset
                    return self._make_pv_mesh(verts, i)
            except Exception:
                pass
            return None

        self._parts["screw_feeder"] = AnimatedPart(
            name="screw_feeder", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="feed",
        )

    def register_deagglomerator(self, deagg, world_offset: np.ndarray, target_rpm: float, color: str = "#9B59B6"):
        """Register a Deagglomerator for rotor animation."""
        if not HAS_PYVISTA or deagg is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)

        def update_fn(dt, frac):
            deagg.update_rotation(dt, target_rpm * frac)

        def mesh_fn():
            try:
                v, i, _ = deagg.get_rotor_mesh()
                if v is not None and len(v) > 0:
                    verts = v.astype(np.float64) + offset
                    return self._make_pv_mesh(verts, i)
            except Exception:
                pass
            return None

        self._parts["deagglomerator"] = AnimatedPart(
            name="deagglomerator", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="feed",
        )

    def register_hopper_lid(self, hopper, world_offset: np.ndarray, color: str = "#F0AD4E"):
        """
        Register a FeedHopper lid for hinge-open animation.

        Uses hopper.get_lid_mesh() and hopper.get_lid_hinge_position().
        Matches feed_flow_physics.py get_lid_transform():
          Combined transform: T(hinge) * Rz(angle) * T(-hinge)
          Positive angle = lid opens upward (Y+).
        """
        if not HAS_PYVISTA or hopper is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)
        state = {"angle_deg": 0.0}

        # Get hinge position in component-local coords
        try:
            hinge_local = np.array(hopper.get_lid_hinge_position(), dtype=np.float64)
        except Exception:
            hinge_local = np.array([0, 0, 0], dtype=np.float64)

        hinge_world = tuple(hinge_local + offset)

        def update_fn(dt, frac):
            # Open from 0 to +90 degrees (Rz positive = lid swings upward Y+)
            state["angle_deg"] = 90.0 * frac

        def mesh_fn():
            try:
                v, i, _ = hopper.get_lid_mesh()
                if v is None or len(v) == 0:
                    return None
                verts = v.astype(np.float64) + offset
                mesh = self._make_pv_mesh(verts, i)
                # Rz at hinge point -- same as feed_flow_physics.get_lid_transform()
                return mesh.rotate_z(state["angle_deg"], point=hinge_world, inplace=False)
            except Exception:
                return None

        self._parts["hopper_lid"] = AnimatedPart(
            name="hopper_lid", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="feed",
        )

    def register_wheel(self, wheel_mesh_pv, world_center: np.ndarray, target_rpm: float, color: str = "#FF6B6B"):
        """
        Register the wheel classifier for Y-axis rotation.

        The wheel doesn't have a separated animated mesh API --
        we rotate the entire wheel mesh like visualize_geometry.py does.
        """
        if not HAS_PYVISTA or wheel_mesh_pv is None:
            return

        center = tuple(np.asarray(world_center, dtype=np.float64))
        base_mesh = wheel_mesh_pv.copy(deep=True)
        state = {"angle_rad": 0.0}

        def update_fn(dt, frac):
            omega = TWO_PI * target_rpm * frac / 60.0
            state["angle_rad"] += omega * dt

        def mesh_fn():
            angle_deg = math.degrees(state["angle_rad"])
            return base_mesh.copy(deep=True).rotate_y(angle_deg, point=center, inplace=False)

        self._parts["wheel_classifier"] = AnimatedPart(
            name="wheel_classifier", get_mesh=mesh_fn, update=update_fn,
            color=color, phase="classification",
        )

    # ================================================================
    # Lifecycle
    # ================================================================

    def start(self, timeline: Optional[AnimationTimeline] = None):
        if timeline:
            self._timeline = timeline
        self._anim_time = 0.0
        self._phase = AnimationPhase.AIR_STARTUP
        self._running = True
        self._physics_mode = False
        self._physics_state = None
        # Reset subsidiary sims' internal time so they start fresh
        # (they may have advanced during a previous preview cycle)
        if self._air_sim is not None:
            try:
                self._air_sim.state.time = 0.0
                self._air_sim.state.step = 0
                self._air_sim.start_system()
            except Exception:
                pass
        if self._feed_sim is not None:
            try:
                self._feed_sim.state.time = 0.0
                self._feed_sim.state.step = 0
                # Reset lid to closed state (use the enum from the state object)
                lid_state_cls = type(self._feed_sim.state.lid_state)
                self._feed_sim.state.lid_state = lid_state_cls("closed")
                self._feed_sim.state.lid_angle = 0.0
                self._feed_sim.state.lid_target_angle = 0.0
            except Exception:
                pass
        for p in self._parts.values():
            p.active = False
        self._timer.start()
        self.phase_changed.emit(self._phase.value)

    def sync_to_sim_time(self, sim_time: float):
        """Sync animation time to simulation progress.

        Call this from the progress callback so the animation phases
        match what the simulation is actually doing.
        """
        # Only sync forward (don't go backward)
        if sim_time > self._anim_time:
            self._anim_time = sim_time

    def update_from_physics(self, component_state: Dict[str, Any]):
        """
        Update animation from physics simulation state.

        When called, the controller switches to physics mode: the simulation's
        actual component states (wheel angle, damper positions, lid angle, etc.)
        drive the animation directly instead of using the fixed timeline ramps.

        This takes priority over subsidiary sims (build-time preview), which
        are cleared on first call.

        Args:
            component_state: Dict from SimulationWorker._get_component_states()
                Keys: sim_time, wheel_omega, wheel_angle_rad, blower_rpm,
                      damper_positions, lid_angle_deg, feed_ramp_frac,
                      blower_ramp_frac, classification_ramp_frac, phase
        """
        self._physics_state = component_state
        if not self._physics_mode:
            self._physics_mode = True
            # Simulation worker now owns physics state; clear build-time sims
            self._air_sim = None
            self._feed_sim = None

        # Also sync animation time
        sim_time = component_state.get("sim_time", 0.0)
        if sim_time > self._anim_time:
            self._anim_time = sim_time

    def begin_shutdown(self, duration: float = 3.0):
        """Begin the shutdown phase: dampers close, equipment ramps down."""
        self._phase = AnimationPhase.SHUTDOWN
        self._shutdown_start = self._anim_time
        self._shutdown_duration = duration
        self.phase_changed.emit(self._phase.value)

    def stop(self):
        self._timer.stop()
        self._running = False
        self._phase = AnimationPhase.IDLE
        self._physics_mode = False
        self._physics_state = None
        self._air_sim = None
        self._feed_sim = None

    # ================================================================
    # Frame tick
    # ================================================================

    def _tick(self):
        if not self._running or self._plotter is None:
            return

        dt = self.FRAME_MS / 1000.0

        # Use physics-driven tick when physics state is available
        # (from SimulationWorker.component_state_updated signal)
        if self._physics_mode and self._physics_state is not None:
            self._tick_physics(dt)
            return

        # If subsidiary simulators are attached (build-time preview),
        # step them and generate physics state locally
        if self._air_sim is not None or self._feed_sim is not None:
            self._tick_with_subsidiary_sims(dt)
            return

        # --- Timeline mode (fallback when no subsidiary sims and no physics) ---
        # During startup (before sim reports time), advance on wall clock;
        # once sim is reporting, sync_to_sim_time drives _anim_time
        if self._anim_time < self._timeline.steady_time:
            self._anim_time += dt
        t = self._anim_time
        tl = self._timeline

        # Phase transitions
        old = self._phase
        if t < tl.feed_start_time:
            self._phase = AnimationPhase.AIR_STARTUP
        elif t < tl.classification_start_time:
            self._phase = AnimationPhase.FEED_STARTUP
        elif t < tl.steady_time:
            self._phase = AnimationPhase.CLASSIFICATION
        else:
            self._phase = AnimationPhase.STEADY_STATE
        if self._phase != old:
            self.phase_changed.emit(self._phase.value)

        # Update each part
        for part in self._parts.values():
            frac = self._compute_frac(part.phase, t, tl)
            part.active = frac > 0.001

            if part.active and part.update:
                part.update(dt, frac)

            if part.active and part.get_mesh:
                self._update_actor(part)

        try:
            self._plotter.render()
        except Exception:
            pass

    def _tick_physics(self, dt: float):
        """
        Physics-driven tick: use actual simulation state to drive animations.

        Instead of computing ramp fractions from a fixed timeline, we read
        the component states emitted by the simulation backend and apply
        them directly. This makes animations match the actual physics.
        """
        ps = self._physics_state
        t = ps.get("sim_time", self._anim_time)
        tl = self._timeline

        # Phase transitions (based on physics time)
        old = self._phase
        if t < tl.feed_start_time:
            self._phase = AnimationPhase.AIR_STARTUP
        elif t < tl.classification_start_time:
            self._phase = AnimationPhase.FEED_STARTUP
        elif t < tl.steady_time:
            self._phase = AnimationPhase.CLASSIFICATION
        else:
            self._phase = AnimationPhase.STEADY_STATE
        if self._phase != old:
            self.phase_changed.emit(self._phase.value)

        # Map physics state to per-part ramp fractions
        # Physics state provides explicit fractions/values for each system
        physics_fracs = {
            "air": ps.get("blower_ramp_frac", self._compute_frac("air", t, tl)),
            "feed": ps.get("feed_ramp_frac", self._compute_frac("feed", t, tl)),
            "classification": ps.get("classification_ramp_frac", self._compute_frac("classification", t, tl)),
        }

        # Special handling: wheel gets physics-computed angle directly
        wheel_angle_rad = ps.get("wheel_angle_rad", None)
        lid_angle_deg = ps.get("lid_angle_deg", None)
        damper_positions = ps.get("damper_positions", None)

        for part in self._parts.values():
            frac = physics_fracs.get(part.phase, 0.0)
            part.active = frac > 0.001

            if not part.active:
                continue

            # --- Physics overrides for specific parts ---
            if part.name == "wheel_classifier" and wheel_angle_rad is not None:
                # Drive wheel directly from physics angle
                self._update_wheel_from_physics(part, wheel_angle_rad)
                continue

            if part.name == "hopper_lid" and lid_angle_deg is not None:
                # Drive lid directly from physics angle
                self._update_lid_from_physics(part, lid_angle_deg)
                continue

            if part.name.startswith("damper_") and damper_positions is not None:
                # Drive damper from physics position
                idx = int(part.name.split("_")[-1]) if "_" in part.name else 0
                if idx < len(damper_positions):
                    self._update_damper_from_physics(part, damper_positions[idx])
                    continue

            # --- Default: use physics-derived frac ---
            if part.update:
                part.update(dt, frac)
            if part.get_mesh:
                self._update_actor(part)

        try:
            self._plotter.render()
        except Exception:
            pass

    def _tick_with_subsidiary_sims(self, dt: float):
        """
        Autonomous physics-driven tick using attached subsidiary simulators.

        Used during build-time preview (no simulation running). Steps the
        air and feed physics simulators locally and builds a component_state
        dict, then delegates to _tick_physics() for the actual rendering.

        This gives the same realistic animation (VFD blower ramp, lid servo,
        damper dynamics) during preview as during a live simulation.
        """
        # Advance animation time
        self._anim_time += dt
        t = self._anim_time
        tl = self._timeline

        # Step air simulator to current time
        if self._air_sim is not None:
            try:
                air_dt = self._air_sim.config.dt
                while self._air_sim.state.time < t - air_dt * 0.5:
                    self._air_sim.step()
            except Exception:
                pass

        # Step feed simulator (starts at feed_start_time)
        if self._feed_sim is not None:
            try:
                if t >= self._feed_start_time:
                    # Trigger lid open once
                    if self._feed_sim.state.lid_state.value == "closed":
                        self._feed_sim.open_lid()
                    feed_dt = self._feed_sim.config.dt
                    target_feed_time = t - self._feed_start_time
                    while self._feed_sim.state.time < target_feed_time - feed_dt * 0.5:
                        self._feed_sim.step()
            except Exception:
                pass

        # Build component state dict from live simulator state
        ps = {"sim_time": t}

        # Air state
        if self._air_sim is not None and hasattr(self._air_sim, 'state'):
            ast = self._air_sim.state
            target_rpm = self._air_sim.config.target_rpm
            ps["blower_ramp_frac"] = float(min(1.0, ast.blower_rpm / target_rpm)) if target_rpm > 0 else 1.0
            ps["damper_positions"] = [float(p) for p in ast.damper_positions]
        else:
            ps["blower_ramp_frac"] = self._compute_frac("air", t, tl)

        # Feed state
        if self._feed_sim is not None and hasattr(self._feed_sim, 'state'):
            fst = self._feed_sim.state
            ps["lid_angle_deg"] = float(fst.lid_angle)
            lid_max = self._feed_sim.config.lid_open_angle if hasattr(self._feed_sim, 'config') else 90.0
            ps["feed_ramp_frac"] = float(min(1.0, fst.lid_angle / lid_max)) if lid_max > 0 else self._compute_frac("feed", t, tl)
        else:
            ps["feed_ramp_frac"] = self._compute_frac("feed", t, tl)

        # Classification (wheel: no subsidiary sim, use timeline frac)
        ps["classification_ramp_frac"] = self._compute_frac("classification", t, tl)

        # Use the physics tick path with the locally-built state
        self._physics_state = ps
        self._tick_physics(dt)

    def _update_wheel_from_physics(self, part: AnimatedPart, angle_rad: float):
        """Update wheel classifier directly from physics-computed angle.

        Reaches into the closure captured by register_wheel()'s mesh_fn
        to set state["angle_rad"] = physics angle, so the next mesh_fn()
        call rotates the wheel to the exact angle from the Warp kernel.
        """
        try:
            # Access the closure variables through the mesh function
            mesh_fn = part.get_mesh
            if mesh_fn is not None:
                # For wheel: mesh_fn creates a rotated copy using state["angle_rad"]
                # We override by directly computing the rotated mesh
                closure_vars = mesh_fn.__code__.co_freevars
                if 'state' in closure_vars:
                    idx = closure_vars.index('state')
                    state_cell = mesh_fn.__closure__[idx]
                    state_cell.cell_contents["angle_rad"] = angle_rad
                # Now call the normal mesh update
                self._update_actor(part)
        except Exception:
            # Fallback: just update normally
            if part.update:
                part.update(self.FRAME_MS / 1000.0, 1.0)
            self._update_actor(part)

    def _update_lid_from_physics(self, part: AnimatedPart, angle_deg: float):
        """Update hopper lid directly from physics-computed angle."""
        try:
            mesh_fn = part.get_mesh
            if mesh_fn is not None:
                closure_vars = mesh_fn.__code__.co_freevars
                if 'state' in closure_vars:
                    idx = closure_vars.index('state')
                    state_cell = mesh_fn.__closure__[idx]
                    state_cell.cell_contents["angle_deg"] = angle_deg
                self._update_actor(part)
        except Exception:
            if part.update:
                part.update(self.FRAME_MS / 1000.0, 1.0)
            self._update_actor(part)

    def _update_damper_from_physics(self, part: AnimatedPart, position: float):
        """Update damper blade directly from physics position (0=closed, 1=open)."""
        try:
            # For dampers, the update_fn calls damper.update_animation(dt, target, time)
            # and mesh_fn calls damper.get_blade_mesh(). We set the position directly.
            if part.update:
                # update_fn(dt, frac) -- frac is used as target_position
                part.update(self.FRAME_MS / 1000.0, position)
            self._update_actor(part)
        except Exception:
            pass

    def _compute_frac(self, phase: str, t: float, tl: AnimationTimeline) -> float:
        """Compute ramp fraction (0..1) for a given phase.

        During SHUTDOWN, all fractions ramp from 1 back to 0.
        """
        # Shutdown: everything ramps down
        if self._phase == AnimationPhase.SHUTDOWN:
            sd_start = getattr(self, '_shutdown_start', t)
            sd_dur = getattr(self, '_shutdown_duration', 3.0)
            progress = min(1.0, (t - sd_start) / max(sd_dur, 0.01))
            return max(0.0, 1.0 - progress)

        if phase == "air":
            if t < tl.air_start_time:
                return 0.0
            elapsed = t - tl.air_start_time
            return min(1.0, elapsed / max(tl.air_ramp_duration, 0.01))
        elif phase == "feed":
            if t < tl.feed_start_time:
                return 0.0
            elapsed = t - tl.feed_start_time
            return min(1.0, elapsed / max(tl.feed_ramp_duration, 0.01))
        elif phase == "classification":
            if t < tl.classification_start_time:
                return 0.0
            elapsed = t - tl.classification_start_time
            return min(1.0, elapsed / max(tl.classification_ramp_duration, 0.01))
        return 1.0 if t >= tl.air_start_time else 0.0

    def _update_actor(self, part: AnimatedPart):
        """Replace the PyVista actor for this part."""
        mesh = part.get_mesh()
        if mesh is None:
            return

        if part.actor is not None:
            try:
                self._plotter.remove_actor(part.actor)
            except Exception:
                pass

        try:
            part.actor = self._plotter.add_mesh(
                mesh, color=part.color, opacity=0.85,
                show_edges=True, edge_color="gray",
                name=f"anim_{part.name}",
            )
        except Exception:
            pass

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _make_pv_mesh(vertices: np.ndarray, indices: np.ndarray):
        """Build a PyVista PolyData from vertices and triangle indices."""
        faces = indices.reshape(-1, 3)
        n = len(faces)
        pv_faces = np.zeros((n, 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = faces
        return pv.PolyData(vertices, pv_faces.flatten())

    def get_status(self) -> Dict[str, Any]:
        active = [p.name for p in self._parts.values() if p.active]
        return {"phase": self._phase.value, "time": self._anim_time, "active": active}
