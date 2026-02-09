"""
Cinematic Camera Controller
============================

Drives the 3D viewport camera through smooth, game-like camera movements
while a simulation is running.  Gives the user a hands-free visual tour
of the assembly from multiple angles.

Modes:
  - Orbit:     Smooth continuous orbit around the assembly centre.
  - Showcase:  Cycles through key viewpoints (overview, feed, wheel,
               cyclones, blower) with ease-in/out interpolation.
  - Flythrough: A scripted path that sweeps through the assembly.

The controller is optional -- toggled by a toolbar button.  When the user
touches the mouse (rotate/pan/zoom), the cinematic camera pauses for a
few seconds then resumes.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal


class CameraMode(Enum):
    ORBIT = "Orbit"
    SHOWCASE = "Showcase"
    FLYTHROUGH = "Flythrough"


@dataclass
class CameraKeyframe:
    """A single camera position/target used in Showcase and Flythrough."""
    position: Tuple[float, float, float]
    focal_point: Tuple[float, float, float]
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    hold_time: float = 2.0   # seconds to hold at this keyframe
    label: str = ""


def _smooth_step(t: float) -> float:
    """Smooth-step ease in/out  (0→1 → 0→1)."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp3(a, b, t: float):
    """Linear interpolation of 3-tuples."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


class CinematicCameraController(QObject):
    """
    Drives the PyVista camera through cinematic movements.

    Usage:
        cam = CinematicCameraController(plotter)
        cam.set_mode(CameraMode.ORBIT)
        cam.start()        # begins camera movement
        cam.stop()         # stops
        cam.pause(5.0)     # pause for 5 s (mouse interaction)

    The controller reads the current camera bounds from the plotter
    to auto-generate sensible orbits and viewpoints.
    """

    mode_changed = Signal(str)    # emitted when mode switches
    FRAME_MS = 33                 # ~30 fps

    def __init__(self, plotter=None, parent=None):
        super().__init__(parent)
        self._plotter = plotter
        self._mode = CameraMode.ORBIT
        self._running = False

        # Assembly bounding info (computed on start)
        self._center = np.array([0.0, 0.0, 0.0])
        self._radius = 1.0

        # Orbit state
        self._orbit_angle = 0.0         # current azimuth (degrees)
        self._orbit_speed = 12.0        # degrees per second
        self._orbit_elevation = 20.0    # elevation oscillates gently
        self._orbit_elev_speed = 3.0    # degrees per second for oscillation

        # Showcase state
        self._keyframes: List[CameraKeyframe] = []
        self._kf_index = 0
        self._kf_transition_time = 2.5  # seconds to interpolate between keyframes
        self._kf_timer = 0.0            # seconds into current segment
        self._kf_phase = "hold"         # "hold" or "transition"

        # Flythrough state (pre-built path of dense points)
        self._fly_points: List[CameraKeyframe] = []
        self._fly_index = 0
        self._fly_speed = 0.3  # fraction of segment per second

        # Pause-on-interaction
        self._paused_until = 0.0  # wall-clock time until which we're paused
        self._pause_duration = 4.0  # seconds to pause after mouse interaction

        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # ================================================================
    #  Public API
    # ================================================================

    def set_plotter(self, plotter):
        self._plotter = plotter

    def set_mode(self, mode: CameraMode):
        self._mode = mode
        self.mode_changed.emit(mode.value)
        if self._running:
            self._init_mode()

    @property
    def mode(self) -> CameraMode:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start cinematic camera movement."""
        if self._plotter is None:
            return
        self._compute_bounds()
        self._running = True
        self._init_mode()
        self._timer.start()

    def stop(self):
        """Stop cinematic camera and leave camera where it is."""
        self._timer.stop()
        self._running = False

    def pause_for_interaction(self):
        """Pause the cinematic camera briefly (user touched the mouse)."""
        self._paused_until = time.time() + self._pause_duration

    def set_orbit_speed(self, degrees_per_second: float):
        self._orbit_speed = degrees_per_second

    # ================================================================
    #  Bounds / scene analysis
    # ================================================================

    def _compute_bounds(self):
        """Read the assembly bounding box from the plotter."""
        try:
            bounds = self._plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            if bounds is not None and len(bounds) == 6:
                self._center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2,
                ])
                dx = bounds[1] - bounds[0]
                dy = bounds[3] - bounds[2]
                dz = bounds[5] - bounds[4]
                self._radius = max(0.5, math.sqrt(dx*dx + dy*dy + dz*dz) / 2)
            else:
                self._center = np.array([0.0, 0.0, 0.0])
                self._radius = 1.0
        except Exception:
            self._center = np.array([0.0, 0.0, 0.0])
            self._radius = 1.0

    # ================================================================
    #  Mode initialisation
    # ================================================================

    def _init_mode(self):
        """Re-initialise state for the current mode."""
        if self._mode == CameraMode.ORBIT:
            self._init_orbit()
        elif self._mode == CameraMode.SHOWCASE:
            self._init_showcase()
        elif self._mode == CameraMode.FLYTHROUGH:
            self._init_flythrough()

    def _init_orbit(self):
        """Start orbit from the current camera azimuth."""
        try:
            cam = self._plotter.camera
            # Compute current azimuth from camera position relative to center
            dx = cam.position[0] - self._center[0]
            dz = cam.position[2] - self._center[2]
            self._orbit_angle = math.degrees(math.atan2(dx, dz))
        except Exception:
            self._orbit_angle = -170.0

    def _init_showcase(self):
        """Build keyframes for a showcase tour of the assembly."""
        c = self._center
        r = self._radius
        up = (0.0, 1.0, 0.0)

        self._keyframes = [
            # 1. Wide overview (isometric-like)
            CameraKeyframe(
                position=(c[0] - r * 1.5, c[1] + r * 0.8, c[2] + r * 1.5),
                focal_point=tuple(c),
                up=up, hold_time=3.0, label="Overview",
            ),
            # 2. Feed system close-up (top-left area, looking down)
            CameraKeyframe(
                position=(c[0] - r * 0.4, c[1] + r * 1.2, c[2] + r * 0.6),
                focal_point=(c[0] - r * 0.3, c[1] + r * 0.4, c[2]),
                up=up, hold_time=3.0, label="Feed System",
            ),
            # 3. Wheel classifier close-up (centre, eye level)
            CameraKeyframe(
                position=(c[0] + r * 0.6, c[1] + r * 0.1, c[2] + r * 0.8),
                focal_point=(c[0], c[1] + r * 0.05, c[2]),
                up=up, hold_time=3.5, label="Wheel Classifier",
            ),
            # 4. Cyclones (right side, slightly above)
            CameraKeyframe(
                position=(c[0] + r * 1.2, c[1] + r * 0.5, c[2] - r * 0.3),
                focal_point=(c[0] + r * 0.3, c[1] + r * 0.2, c[2] - r * 0.2),
                up=up, hold_time=3.0, label="Cyclones",
            ),
            # 5. Blower (bottom, looking up)
            CameraKeyframe(
                position=(c[0] - r * 0.8, c[1] - r * 0.3, c[2] - r * 1.0),
                focal_point=(c[0] - r * 0.3, c[1] - r * 0.1, c[2] - r * 0.3),
                up=up, hold_time=3.0, label="Air System",
            ),
            # 6. Back to wide overview (different angle)
            CameraKeyframe(
                position=(c[0] + r * 1.3, c[1] + r * 1.0, c[2] + r * 1.3),
                focal_point=tuple(c),
                up=up, hold_time=3.0, label="Overview (rear)",
            ),
        ]
        self._kf_index = 0
        self._kf_timer = 0.0
        self._kf_phase = "hold"
        # Jump to first keyframe immediately
        self._apply_keyframe(self._keyframes[0])

    def _init_flythrough(self):
        """Build a dense-point path that sweeps around/through the assembly."""
        c = self._center
        r = self._radius
        up = (0.0, 1.0, 0.0)

        # Build a spiral-like path: orbit at decreasing radius, rising then falling
        self._fly_points = []
        n_points = 120   # total keyframes in the path
        for i in range(n_points):
            t = i / (n_points - 1)  # 0..1
            angle = t * 720  # 2 full orbits (degrees)
            rad = math.radians(angle)

            # Radius: start far, come close, go far again
            dist = r * (1.5 - 0.6 * math.sin(t * math.pi))
            # Elevation: gentle sine wave
            elev = r * 0.4 * math.sin(t * math.pi * 2)

            px = c[0] + dist * math.sin(rad)
            py = c[1] + elev + r * 0.3
            pz = c[2] + dist * math.cos(rad)

            # Look at center with a slight lead (look ahead along the path)
            lead = 0.05
            t2 = min(1.0, t + lead)
            angle2 = t2 * 720
            rad2 = math.radians(angle2)
            fx = c[0] + r * 0.3 * math.sin(rad2)
            fy = c[1] + r * 0.1 * math.sin(t2 * math.pi)
            fz = c[2] + r * 0.3 * math.cos(rad2)

            self._fly_points.append(CameraKeyframe(
                position=(px, py, pz),
                focal_point=(fx, fy, fz),
                up=up, hold_time=0, label=f"fly_{i}",
            ))

        self._fly_index = 0
        self._fly_speed = 0.5  # segments per second
        if self._fly_points:
            self._apply_keyframe(self._fly_points[0])

    # ================================================================
    #  Tick
    # ================================================================

    def _tick(self):
        if not self._running or self._plotter is None:
            return

        # Check pause-on-interaction
        if time.time() < self._paused_until:
            return

        dt = self.FRAME_MS / 1000.0

        if self._mode == CameraMode.ORBIT:
            self._tick_orbit(dt)
        elif self._mode == CameraMode.SHOWCASE:
            self._tick_showcase(dt)
        elif self._mode == CameraMode.FLYTHROUGH:
            self._tick_flythrough(dt)

    # ----------------------------------------------------------------
    #  Orbit
    # ----------------------------------------------------------------

    def _tick_orbit(self, dt: float):
        """Smooth orbit around the assembly centre."""
        self._orbit_angle += self._orbit_speed * dt
        if self._orbit_angle > 360:
            self._orbit_angle -= 360

        # Gentle elevation oscillation
        t = time.time()
        elev = self._orbit_elevation + 8.0 * math.sin(t * 0.15)

        rad = math.radians(self._orbit_angle)
        elev_rad = math.radians(elev)
        dist = self._radius * 2.0

        # Spherical → Cartesian (Y-up)
        cos_e = math.cos(elev_rad)
        px = self._center[0] + dist * math.sin(rad) * cos_e
        py = self._center[1] + dist * math.sin(elev_rad)
        pz = self._center[2] + dist * math.cos(rad) * cos_e

        try:
            cam = self._plotter.camera
            cam.position = (px, py, pz)
            cam.focal_point = tuple(self._center)
            cam.up = (0, 1, 0)
            self._plotter.render()
        except Exception:
            pass

    # ----------------------------------------------------------------
    #  Showcase
    # ----------------------------------------------------------------

    def _tick_showcase(self, dt: float):
        """Cycle through keyframes with smooth transitions."""
        if not self._keyframes:
            return

        self._kf_timer += dt
        kf_count = len(self._keyframes)

        if self._kf_phase == "hold":
            current_kf = self._keyframes[self._kf_index % kf_count]
            if self._kf_timer >= current_kf.hold_time:
                # Start transitioning to next keyframe
                self._kf_phase = "transition"
                self._kf_timer = 0.0
        elif self._kf_phase == "transition":
            if self._kf_timer >= self._kf_transition_time:
                # Arrived at next keyframe
                self._kf_index = (self._kf_index + 1) % kf_count
                self._kf_phase = "hold"
                self._kf_timer = 0.0
                self._apply_keyframe(self._keyframes[self._kf_index])
            else:
                # Interpolate
                t = _smooth_step(self._kf_timer / self._kf_transition_time)
                src = self._keyframes[self._kf_index % kf_count]
                dst = self._keyframes[(self._kf_index + 1) % kf_count]
                pos = _lerp3(src.position, dst.position, t)
                fp = _lerp3(src.focal_point, dst.focal_point, t)
                up = _lerp3(src.up, dst.up, t)
                try:
                    cam = self._plotter.camera
                    cam.position = pos
                    cam.focal_point = fp
                    cam.up = up
                    self._plotter.render()
                except Exception:
                    pass

    # ----------------------------------------------------------------
    #  Flythrough
    # ----------------------------------------------------------------

    def _tick_flythrough(self, dt: float):
        """Advance along the pre-built flythrough path."""
        if not self._fly_points:
            return

        n = len(self._fly_points)
        self._kf_timer += dt * self._fly_speed

        # Current segment index
        idx = int(self._kf_timer) % n
        frac = self._kf_timer - int(self._kf_timer)
        frac = _smooth_step(frac)

        src = self._fly_points[idx % n]
        dst = self._fly_points[(idx + 1) % n]

        pos = _lerp3(src.position, dst.position, frac)
        fp = _lerp3(src.focal_point, dst.focal_point, frac)

        try:
            cam = self._plotter.camera
            cam.position = pos
            cam.focal_point = fp
            cam.up = (0, 1, 0)
            self._plotter.render()
        except Exception:
            pass

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------

    def _apply_keyframe(self, kf: CameraKeyframe):
        """Jump camera to a keyframe instantly."""
        try:
            cam = self._plotter.camera
            cam.position = kf.position
            cam.focal_point = kf.focal_point
            cam.up = kf.up
            self._plotter.render()
        except Exception:
            pass
