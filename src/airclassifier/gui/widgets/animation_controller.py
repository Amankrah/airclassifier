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

    Usage:
        ctrl = AnimationController(plotter)
        ctrl.register_blower(blower, position, rpm)
        ctrl.register_damper(damper, position)
        ctrl.register_airlock(airlock, position, rpm)
        ...
        ctrl.start()
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

        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._tick)

    def set_plotter(self, plotter):
        self._plotter = plotter

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
        Lid rotates around the hinge X-axis from 0 (closed) to 90 deg (open).
        """
        if not HAS_PYVISTA or hopper is None:
            return

        offset = np.asarray(world_offset, dtype=np.float64)
        state = {"angle_deg": 0.0}

        # Get hinge position in local coords
        try:
            hinge_local = np.array(hopper.get_lid_hinge_position(), dtype=np.float64)
        except Exception:
            hinge_local = np.array([0, 0, 0], dtype=np.float64)

        hinge_world = tuple(hinge_local + offset)

        def update_fn(dt, frac):
            # Open from 0 to 90 degrees during ramp
            state["angle_deg"] = 90.0 * frac

        def mesh_fn():
            try:
                v, i, _ = hopper.get_lid_mesh()
                if v is None or len(v) == 0:
                    return None
                verts = v.astype(np.float64) + offset
                mesh = self._make_pv_mesh(verts, i)
                # Rotate around hinge axis (X-axis rotation for lid)
                return mesh.rotate_x(state["angle_deg"], point=hinge_world, inplace=False)
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
        for p in self._parts.values():
            p.active = False
        self._timer.start()
        self.phase_changed.emit(self._phase.value)

    def stop(self):
        self._timer.stop()
        self._running = False
        self._phase = AnimationPhase.IDLE

    # ================================================================
    # Frame tick
    # ================================================================

    def _tick(self):
        if not self._running or self._plotter is None:
            return

        dt = self.FRAME_MS / 1000.0
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

    def _compute_frac(self, phase: str, t: float, tl: AnimationTimeline) -> float:
        """Compute ramp fraction (0..1) for a given phase."""
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
