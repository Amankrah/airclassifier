"""
Material Transport Kernels & Conveyor Drive Controller
======================================================

Conveyor belt advection of scalar fields (T, M) along the positive
X-axis, plus the kinematic drive controller that manages motor
start/stop, VFD speed ramping, and real-time angular positions
of all rotating components for animation.

**Advection kernels** (existing):
    Phase 1: First-order upwind (NumPy).
    Phase 2: Van Leer TVD with flux limiter (NumPy) — §4.4.1.
    Phase 3: @wp.kernel on GPU.

**Drive controller** (new):
    Manages the complete kinematic chain from the GP-15 manual
    Conveyor Detail C (pp. 97-99):

        Motor (0.75 kW, VFD) → Gearbox → Drive sprocket (r=38 mm)
        → Roller chain 16B-1 → Driven sprocket (r=75 mm)
        → Head roller shaft → Belt → All other rollers

    Provides:
        - ``belt_speed_m_per_s`` for the advection kernels
        - Angular positions/velocities for 3D animation
        - Start / stop / set_speed control interface
        - Encoder pulse count (500 PPR) for PLC feedback
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


# ── Phase 1: first-order upwind ──────────────────────────────────────

def advect_material_np(
    field: np.ndarray,
    v_belt_m_per_s: float,
    dx: float,
    dt: float,
    inlet_value: float | np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Advect a 3-D scalar field along +X (conveyor direction).

    First-order upwind scheme::

        phi_new[i] = phi[i] - C * (phi[i] - phi[i-1])

    where ``C = v_belt * dt / dx`` is the Courant number (must be < 1).

    Args:
        field: 3-D scalar field to advect (e.g. T or M).
        v_belt_m_per_s: Belt velocity [m/s].
        dx: Cell size in X direction [m].
        dt: Timestep [s].
        inlet_value: Value at infeed boundary (scalar or 2-D array
            of shape ``(ny, nz)``).
        out: Optional pre-allocated output of same shape.

    Returns:
        Advected field.
    """
    C = v_belt_m_per_s * dt / dx  # Courant number
    if C > 1.0:
        raise ValueError(
            f"Courant number {C:.3f} > 1 — reduce dt or increase dx."
        )

    if out is None:
        out = np.empty_like(field)

    # Infeed (i=0): inject fresh material
    out[0, :, :] = inlet_value

    # Interior + outfeed: upwind
    out[1:, :, :] = field[1:, :, :] - C * (field[1:, :, :] - field[:-1, :, :])

    return out


# ── Phase 2: Van Leer TVD with flux limiter ──────────────────────────

def _van_leer_limiter(r: np.ndarray) -> np.ndarray:
    """Van Leer flux limiter: psi(r) = (r + |r|) / (1 + |r|)."""
    return (r + np.abs(r)) / (1.0 + np.abs(r))


def advect_material_tvd_np(
    field: np.ndarray,
    v_belt_m_per_s: float,
    dx: float,
    dt: float,
    inlet_value: float | np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Advect a 3-D scalar field along +X using Van Leer TVD.

    Second-order flux-limited scheme from the engineering guide §4.4.1::

        r = (phi[i] - phi[i-1]) / (phi[i+1] - phi[i])
        psi(r) = (r + |r|) / (1 + |r|)         # Van Leer limiter
        phi_new[i] = phi[i] - C * (phi[i] - phi[i-1])
                     - 0.5 * C * (1 - C) * psi(r) * (phi[i+1] - phi[i])
                     + 0.5 * C * (1 - C) * psi(1/r) * (phi[i] - phi[i-1])

    Simplifies to the MUSCL form with Van Leer limiter.

    Courant number C = v_belt * dt / dx must be in (0, 1).

    Args:
        field: 3-D scalar field to advect.
        v_belt_m_per_s: Belt velocity [m/s].
        dx: Cell size in X direction [m].
        dt: Timestep [s].
        inlet_value: Value at infeed boundary.
        out: Optional pre-allocated output.

    Returns:
        Advected field.
    """
    C = v_belt_m_per_s * dt / dx
    if C > 1.0:
        raise ValueError(
            f"Courant number {C:.3f} > 1 — reduce dt or increase dx."
        )
    if C <= 0.0:
        if out is None:
            return field.copy()
        out[:] = field
        return out

    if out is None:
        out = np.empty_like(field)

    nx = field.shape[0]

    # Infeed (i=0): inject fresh material
    out[0, :, :] = inlet_value

    # i=1: only upwind available (no i-2 for the slope ratio)
    out[1, :, :] = field[1, :, :] - C * (field[1, :, :] - field[0, :, :])

    if nx > 3:
        # Interior i = 2 .. nx-2 : full TVD stencil  [i-1, i, i+1]
        phi_m1 = field[1:-2, :, :]    # phi[i-1]   for i in [2..nx-2]
        phi_0 = field[2:-1, :, :]     # phi[i]
        phi_p1 = field[3:, :, :]      # phi[i+1]   (only exists up to nx-2)

        # But we need phi[i-1], phi[i], phi[i+1] for i=2..nx-2
        # Let me redo indices carefully.
        # For i in range(2, nx-1):
        #   phi_im1 = field[i-1]
        #   phi_i   = field[i]
        #   phi_ip1 = field[i+1]  (exists for i up to nx-2)
        phi_im1 = field[1:-2, :, :]   # i-1 for i in [2, nx-1)
        phi_i = field[2:-1, :, :]     # i
        phi_ip1 = field[3:, :, :]     # i+1  — shape is (nx-3, ...)

        # Need same-size slices: use i in [2, nx-2) so i+1 exists
        # That means phi_im1 = field[1:-2], phi_i = field[2:-1], phi_ip1 = field[3:]
        # all have shape (nx-3, ny, nz)

        delta_fwd = phi_ip1 - phi_i   # phi[i+1] - phi[i]
        delta_bwd = phi_i - phi_im1   # phi[i] - phi[i-1]

        # Slope ratio r = delta_bwd / delta_fwd (with zero-division guard)
        r = np.where(
            np.abs(delta_fwd) > 1e-30,
            delta_bwd / delta_fwd,
            np.where(delta_bwd > 0, 1e10, np.where(delta_bwd < 0, -1e10, 0.0)),
        )

        psi = _van_leer_limiter(r)

        # TVD flux: upwind + anti-diffusive correction
        out[2:-1, :, :] = (
            phi_i
            - C * delta_bwd
            - 0.5 * C * (1.0 - C) * (psi * delta_fwd - psi * delta_bwd)
        )

    # Outfeed (i=nx-1): fall back to upwind (no phi[i+1])
    if nx > 1:
        out[-1, :, :] = field[-1, :, :] - C * (field[-1, :, :] - field[-2, :, :])

    return out


# ── Warp GPU kernels ─────────────────────────────────────────────────

try:
    import warp as wp

    @wp.kernel
    def advect_material_wp_kernel(
        field: wp.array3d(dtype=float),
        field_new: wp.array3d(dtype=float),
        v_belt_dx_dt: float,
        inlet_value: float,
        nx: int, ny: int, nz: int,
    ):
        """Advect a scalar field along +X (upwind scheme, Warp GPU)."""
        i, j, k = wp.tid()
        if i >= nx or j >= ny or k >= nz:
            return

        if i == 0:
            field_new[i, j, k] = inlet_value
        else:
            field_new[i, j, k] = field[i, j, k] - v_belt_dx_dt * (
                field[i, j, k] - field[i - 1, j, k]
            )

    _HAS_WARP = True

except ImportError:
    _HAS_WARP = False


# ════════════════════════════════════════════════════════════════════════
#  Conveyor Drive Controller
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ConveyorDriveState:
    """Instantaneous kinematic state of the GP-15 conveyor drive.

    Updated every timestep by :class:`ConveyorDriveController`.
    Read by the physics coupling loop (belt speed) and the 3-D
    renderer (angular positions for roller / sprocket animation).

    All angles are in radians (cumulative, not wrapped).
    """

    # ── Motor / drive ─────────────────────────────────────────────
    running: bool = False
    speed_setpoint_m_per_min: float = 0.0     # target belt speed
    belt_speed_m_per_min: float = 0.0         # actual (ramped)
    belt_speed_m_per_s: float = 0.0           # actual in SI

    # ── VFD ramp ──────────────────────────────────────────────────
    accel_rate: float = 0.5      # [m/min per s]  ramp-up rate
    decel_rate: float = 0.8      # [m/min per s]  ramp-down (faster)

    # ── Angular positions (cumulative radians) ────────────────────
    head_roller_angle: float = 0.0
    tail_roller_angle: float = 0.0
    drive_sprocket_angle: float = 0.0
    driven_sprocket_angle: float = 0.0

    # ── Angular velocities (rad/s) — current frame ───────────────
    head_roller_omega: float = 0.0
    tail_roller_omega: float = 0.0
    drive_sprocket_omega: float = 0.0

    # ── Linear positions (cumulative metres) ──────────────────────
    belt_position_m: float = 0.0
    chain_position_m: float = 0.0

    # ── Time / feedback ───────────────────────────────────────────
    elapsed_time_s: float = 0.0
    encoder_pulses: int = 0


class ConveyorDriveController:
    """Controls the GP-15 conveyor belt drive system.

    Manages motor start/stop, VFD speed ramping, and per-timestep
    kinematic computation of all rotating and translating parts.

    Kinematic chain (manual Conveyor Detail C, pp. 97-99)::

        Motor (0.75 kW, PF525 VFD)
          → Gearbox (11-0043-0121)
            → Drive sprocket  r = 38 mm  (41-0487-0029)
              → Roller chain 16B-1, 10 ft  (11-0147-0015)
                → Driven sprocket  r = 75 mm  (10.09.167)  ×2
                  → Head roller shaft
                    → Belt loop (~16 m PTFE)
                      → All other rollers (tail, tension, idlers)

    Usage::

        ctrl = ConveyorDriveController.from_params(conveyor_params)
        ctrl.start(speed_m_per_min=0.5)

        for _ in range(steps):
            state = ctrl.step(dt)

            # Physics kernel:
            advect_material_np(T, state.belt_speed_m_per_s, dx, dt, T_in)

            # Animation:
            anim = ctrl.animation_state   # dict of angles / offsets

    Parameters
    ----------
    head_roller_radius_m : float
        Head (drive) roller radius [m].
    tail_roller_radius_m : float
        Tail (nose) roller radius [m].
    tension_roller_radius_m : float
        Gravity tension roller radius [m].
    belt_speed_min_m_per_min : float
        Minimum belt speed [m/min].
    belt_speed_max_m_per_min : float
        Maximum belt speed [m/min].
    accel_rate : float
        VFD acceleration ramp [m/min per second].
    decel_rate : float
        VFD deceleration ramp [m/min per second].
    """

    def __init__(
        self,
        head_roller_radius_m: float = 0.075,
        tail_roller_radius_m: float = 0.060,
        tension_roller_radius_m: float = 0.075,
        driven_sprocket_radius_m: float = 0.075,
        drive_sprocket_radius_m: float = 0.038,
        tension_sprocket_radius_m: float = 0.022,
        encoder_ppr: int = 500,
        belt_speed_min_m_per_min: float = 0.1,
        belt_speed_max_m_per_min: float = 2.0,
        accel_rate: float = 0.5,
        decel_rate: float = 0.8,
    ):
        self.head_r = head_roller_radius_m
        self.tail_r = tail_roller_radius_m
        self.tension_r = tension_roller_radius_m
        self.driven_sprocket_r = driven_sprocket_radius_m
        self.drive_sprocket_r = drive_sprocket_radius_m
        self.tension_sprocket_r = tension_sprocket_radius_m
        self.encoder_ppr = encoder_ppr
        self.speed_min = belt_speed_min_m_per_min
        self.speed_max = belt_speed_max_m_per_min

        self.state = ConveyorDriveState(
            accel_rate=accel_rate,
            decel_rate=decel_rate,
        )

    @classmethod
    def from_params(
        cls,
        params: Any,
        config: Any = None,
    ) -> "ConveyorDriveController":
        """Create from ``ConveyorBeltParams`` and optional ``MachineConfig``.

        Pulls all roller radii, sprocket radii, and encoder PPR from
        the geometry params — single source of truth.

        Args:
            params: A ``ConveyorBeltParams`` instance (or duck-typed object
                    with ``head_roller_radius_m``, etc.).
            config: A ``MachineConfig`` instance for speed limits.

        Returns:
            Configured controller.
        """
        kw: Dict[str, Any] = {
            # Roller radii
            "head_roller_radius_m": getattr(params, "head_roller_radius_m", 0.075),
            "tail_roller_radius_m": getattr(params, "tail_roller_radius_m", 0.060),
            "tension_roller_radius_m": getattr(params, "tension_roller_radius_m", 0.075),
            # Drive system (from ConveyorBeltParams drive section)
            "driven_sprocket_radius_m": getattr(params, "driven_sprocket_radius_m", 0.075),
            "drive_sprocket_radius_m": getattr(params, "drive_sprocket_radius_m", 0.038),
            "tension_sprocket_radius_m": getattr(params, "tension_sprocket_radius_m", 0.022),
            "encoder_ppr": getattr(params, "encoder_ppr", 500),
        }
        if config is not None:
            kw["belt_speed_min_m_per_min"] = getattr(
                config, "belt_speed_min_m_per_min", 0.1,
            )
            kw["belt_speed_max_m_per_min"] = getattr(
                config, "belt_speed_max_m_per_min", 2.0,
            )
        return cls(**kw)

    # ── Control interface ─────────────────────────────────────────

    def start(self, speed_m_per_min: float | None = None) -> None:
        """Start the conveyor motor.  Ramps up to setpoint."""
        if speed_m_per_min is not None:
            self.set_speed(speed_m_per_min)
        self.state.running = True

    def stop(self) -> None:
        """Stop the conveyor motor.  Ramps down to zero."""
        self.state.running = False

    def set_speed(self, speed_m_per_min: float) -> None:
        """Set belt speed setpoint [m/min], clamped to limits."""
        self.state.speed_setpoint_m_per_min = max(
            0.0, min(speed_m_per_min, self.speed_max),
        )

    def emergency_stop(self) -> None:
        """Immediate stop — no ramp, instant zero speed."""
        self.state.running = False
        self.state.speed_setpoint_m_per_min = 0.0
        self.state.belt_speed_m_per_min = 0.0
        self.state.belt_speed_m_per_s = 0.0
        self.state.head_roller_omega = 0.0
        self.state.tail_roller_omega = 0.0
        self.state.drive_sprocket_omega = 0.0

    # ── Step (main kinematic update) ──────────────────────────────

    def step(self, dt: float) -> ConveyorDriveState:
        """Advance the drive system by *dt* seconds.

        Handles:
            1. VFD speed ramp (acceleration / deceleration)
            2. Belt speed in m/s (for advection kernels)
            3. Angular positions and velocities of all rotating parts
            4. Belt and chain cumulative linear position
            5. Encoder pulse count (500 PPR on head roller)

        Args:
            dt: Timestep in seconds.

        Returns:
            Updated :class:`ConveyorDriveState`.
        """
        s = self.state

        # ── 1. VFD speed ramp ─────────────────────────────────────
        target = s.speed_setpoint_m_per_min if s.running else 0.0
        diff = target - s.belt_speed_m_per_min

        if diff > 0:
            change = min(diff, s.accel_rate * dt)
            s.belt_speed_m_per_min += change
        elif diff < 0:
            change = min(-diff, s.decel_rate * dt)
            s.belt_speed_m_per_min -= change

        s.belt_speed_m_per_s = s.belt_speed_m_per_min / 60.0

        # ── 2. Kinematic chain ────────────────────────────────────
        v_belt = s.belt_speed_m_per_s

        # Head roller:  v = ω × r  →  ω = v / r
        if self.head_r > 0:
            omega_head = v_belt / self.head_r
        else:
            omega_head = 0.0
        s.head_roller_angle += omega_head * dt
        s.head_roller_omega = omega_head

        # Tail roller
        if self.tail_r > 0:
            omega_tail = v_belt / self.tail_r
        else:
            omega_tail = 0.0
        s.tail_roller_angle += omega_tail * dt
        s.tail_roller_omega = omega_tail

        # Driven sprocket (same shaft as head roller)
        s.driven_sprocket_angle = s.head_roller_angle

        # Chain linear speed  =  driven-sprocket ω × R_driven
        v_chain = omega_head * self.driven_sprocket_r
        s.chain_position_m += v_chain * dt

        # Drive sprocket:  v_chain = ω_drive × R_drive
        if self.drive_sprocket_r > 0:
            omega_drive = v_chain / self.drive_sprocket_r
        else:
            omega_drive = 0.0
        s.drive_sprocket_angle += omega_drive * dt
        s.drive_sprocket_omega = omega_drive

        # Belt cumulative linear position
        s.belt_position_m += v_belt * dt

        # ── 3. Encoder (PPR on head roller shaft) ──────────────────
        total_revs = s.head_roller_angle / (2.0 * math.pi)
        s.encoder_pulses = int(total_revs * self.encoder_ppr)

        # ── 4. Time ───────────────────────────────────────────────
        s.elapsed_time_s += dt

        return s

    # ── Roller angle helper ───────────────────────────────────────

    def roller_angle(self, roller_radius_m: float) -> float:
        """Cumulative rotation angle for any belt-driven roller.

        Since ``belt_position_m`` is the integral of belt speed
        over time, this correctly accounts for ramp-up/down.

        Args:
            roller_radius_m: Roller radius [m].

        Returns:
            Cumulative angle [rad].
        """
        if roller_radius_m <= 0:
            return 0.0
        return self.state.belt_position_m / roller_radius_m

    # ── Output interfaces ─────────────────────────────────────────

    @property
    def transport_params(self) -> Dict[str, Any]:
        """Parameters needed by the advection kernels.

        Returns dict with keys:
            - ``v_belt_m_per_s``: current belt speed [m/s]
            - ``belt_speed_m_per_min``: same in [m/min]
            - ``running``: motor running flag
            - ``belt_position_m``: cumulative belt travel [m]
        """
        s = self.state
        return {
            "v_belt_m_per_s": s.belt_speed_m_per_s,
            "belt_speed_m_per_min": s.belt_speed_m_per_min,
            "running": s.running,
            "belt_position_m": s.belt_position_m,
        }

    @property
    def animation_state(self) -> Dict[str, float]:
        """Kinematic state for 3-D renderer animation.

        Returns dict with keys:
            - Roller / sprocket cumulative angles [rad]
            - Angular velocities [rad/s]
            - Chain and belt cumulative offsets [m]
            - Current belt speed [m/s]
        """
        s = self.state
        return {
            "head_roller_angle_rad": s.head_roller_angle,
            "tail_roller_angle_rad": s.tail_roller_angle,
            "drive_sprocket_angle_rad": s.drive_sprocket_angle,
            "driven_sprocket_angle_rad": s.driven_sprocket_angle,
            "head_roller_omega_rad_s": s.head_roller_omega,
            "tail_roller_omega_rad_s": s.tail_roller_omega,
            "drive_sprocket_omega_rad_s": s.drive_sprocket_omega,
            "chain_offset_m": s.chain_position_m,
            "belt_offset_m": s.belt_position_m,
            "belt_speed_m_per_s": s.belt_speed_m_per_s,
        }

    @property
    def physics_state(self) -> Dict[str, Any]:
        """Complete state for the coupled physics orchestrator.

        Includes everything the ``CoupledSimulator`` needs: belt
        speed for advection, encoder feedback for PLC, and timing.
        """
        s = self.state
        omega = s.head_roller_omega
        return {
            "running": s.running,
            "belt_speed_m_per_s": s.belt_speed_m_per_s,
            "belt_speed_m_per_min": s.belt_speed_m_per_min,
            "belt_position_m": s.belt_position_m,
            "elapsed_time_s": s.elapsed_time_s,
            "encoder_pulses": s.encoder_pulses,
            "encoder_speed_rpm": omega * 60.0 / (2.0 * math.pi) if omega > 0 else 0.0,
            "courant_safe_dt": self.courant_safe_dt(),
        }

    def courant_safe_dt(self, dx: float = 0.025) -> float:
        """Maximum timestep for Courant stability at current speed.

        Args:
            dx: Cell size in conveyor direction [m].
                Default 0.025 m corresponds to a 60-cell grid
                over a 1.5 m oven.

        Returns:
            Safe dt [s] (0.9 × dx / v_belt), or 1.0 if belt is
            stationary.
        """
        v = self.state.belt_speed_m_per_s
        if v <= 0:
            return 1.0
        return 0.9 * dx / v


# ── Animation helper ─────────────────────────────────────────────────

def rotate_mesh_around_z_axis(
    vertices: np.ndarray,
    center_x: float,
    center_y: float,
    angle_rad: float,
) -> np.ndarray:
    """Rotate mesh vertices around a Z-parallel axis.

    For animating rollers and sprockets whose axis is along Z:
    rotates the vertices in the X-Y plane around the roller/sprocket
    centre ``(center_x, center_y)``.

    Args:
        vertices: (N, 3) vertex array.
        center_x: Rotation centre X.
        center_y: Rotation centre Y.
        angle_rad: Rotation angle [rad] (positive = CCW in X-Y).

    Returns:
        Rotated vertex array (copy — original is not modified).
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    out = vertices.copy()
    dx = vertices[:, 0] - center_x
    dy = vertices[:, 1] - center_y
    out[:, 0] = center_x + dx * cos_a - dy * sin_a
    out[:, 1] = center_y + dx * sin_a + dy * cos_a
    return out
