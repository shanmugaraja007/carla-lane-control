"""Lateral controllers and the kinematic vehicle model used to test them.

Two controllers are implemented. The PID is the one the CARLA ROS 2 reference
stack uses and the one the portfolio project describes. Stanley is here as a
comparison, because on a curve the PID has no way to anticipate and Stanley
does, and having both in the repository makes that difference measurable rather
than assertable.

The bicycle model is what makes the controllers testable without a simulator.
It is the standard kinematic model, exact for low lateral acceleration and
wrong above roughly 0.4 g, which is well outside anything a lane keeping
controller should be producing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import ControllerConfig, VehicleConfig

log = logging.getLogger(__name__)


@dataclass
class PIDLateralController:
    """PID on the cross track error.

    The output is a normalised steering command in [-1, 1], which is what
    ``CarlaEgoVehicleControl.steer`` expects.
    """

    cfg: ControllerConfig
    _integral: float = field(default=0.0, init=False)
    _previous_error: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def step(self, cross_track_error: float, dt: float | None = None) -> float:
        """Compute one steering command.

        Args:
            cross_track_error: metres from the lane centre, positive when the
                vehicle is right of centre.
            dt: control period. Defaults to the configured period.

        Returns:
            Normalised steering command, negative to steer left.
        """
        dt = self.cfg.dt if dt is None else dt
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = -cross_track_error  # steer left when we are right of centre

        self._integral += error * dt
        # Anti windup. Without this a long straight with a small bias saturates
        # the integrator and the first curve is entered with a large stale term.
        self._integral = float(
            np.clip(self._integral, -self.cfg.integral_limit, self.cfg.integral_limit)
        )

        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        self._previous_error = error

        command = self.cfg.kp * error + self.cfg.ki * self._integral + self.cfg.kd * derivative
        return float(np.clip(command, -self.cfg.steer_limit, self.cfg.steer_limit))


@dataclass
class StanleyLateralController:
    """Stanley controller: heading error plus a speed scaled cross track term.

    delta = heading_error + arctan(k * e / (v + softening))

    The softening constant keeps the cross track term finite at a standstill.
    """

    cfg: ControllerConfig

    def reset(self) -> None:  # kept for interface parity with the PID
        return None

    def step(
        self, cross_track_error: float, heading_error: float, speed_ms: float
    ) -> float:
        """Compute one steering command.

        Args:
            cross_track_error: metres from the lane centre, positive right.
            heading_error: radians between vehicle heading and lane direction.
            speed_ms: forward speed.

        Returns:
            Normalised steering command.
        """
        cross_term = np.arctan(
            self.cfg.stanley_k * (-cross_track_error) / (abs(speed_ms) + self.cfg.stanley_softening)
        )
        angle = heading_error + cross_term
        command = angle / self.cfg.max_steer_angle
        return float(np.clip(command, -self.cfg.steer_limit, self.cfg.steer_limit))


@dataclass
class BicycleState:
    """Pose and speed of the kinematic bicycle."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0


class KinematicBicycle:
    """Rear axle kinematic bicycle model, integrated with explicit Euler.

    Used to close the loop around a controller without a simulator. At a 20 Hz
    control period and highway speeds the integration error is far below the
    lane keeping tolerances being measured.
    """

    def __init__(self, cfg: VehicleConfig, state: BicycleState | None = None):
        self.cfg = cfg
        self.state = state or BicycleState()

    def step(self, steer_command: float, acceleration: float, dt: float,
             max_steer_angle: float) -> BicycleState:
        """Advance the model one control period.

        Args:
            steer_command: normalised steering in [-1, 1].
            acceleration: forward acceleration in m/s^2.
            dt: timestep in seconds.
            max_steer_angle: physical steering angle at a command of 1.0.

        Returns:
            The updated state.
        """
        delta = float(np.clip(steer_command, -1.0, 1.0)) * max_steer_angle
        s = self.state

        s.x += s.speed * np.cos(s.yaw) * dt
        s.y += s.speed * np.sin(s.yaw) * dt
        s.yaw += (s.speed / self.cfg.wheelbase_m) * np.tan(delta) * dt
        s.speed = float(np.clip(s.speed + acceleration * dt, 0.0, self.cfg.max_speed_ms))

        return s


def tracking_metrics(error: np.ndarray, dt: float) -> dict[str, float]:
    """Summarise how well a controller held a path.

    Valid for any run, including one that starts on the path, which is the case
    a step response cannot describe.

    Args:
        error: cross track error over time.
        dt: sample period, used only to report when the peak occurred.

    Returns:
        rmse_m, max_abs_error_m, time_of_max_s, mean_abs_error_m,
        steady_state_error_m.
    """
    error = np.asarray(error, dtype=float)
    if error.size == 0:
        raise ValueError("empty error signal")

    tail = error[-max(1, len(error) // 10) :]
    peak_index = int(np.argmax(np.abs(error)))
    return {
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "time_of_max_s": float(peak_index * dt),
        "mean_abs_error_m": float(np.mean(np.abs(error))),
        "steady_state_error_m": float(np.mean(np.abs(tail))),
    }


def step_response_metrics(
    error: np.ndarray, dt: float, settle_band: float = 0.05
) -> dict[str, float]:
    """Characterise a step response.

    Args:
        error: cross track error over time, starting from the step.
        dt: sample period.
        settle_band: fraction of the initial error that counts as settled.

    Returns:
        rise_time_s, overshoot_pct, settling_time_s, steady_state_error_m, rmse_m.

    Raises:
        ValueError: if the run does not start with a step. Use
            ``tracking_metrics`` for a run that begins on the path.
    """
    error = np.asarray(error, dtype=float)
    if error.size == 0:
        raise ValueError("empty error signal")

    initial = float(error[0])
    if abs(initial) < 1e-9:
        raise ValueError(
            "initial error is zero, so there is no step to characterise; "
            "use tracking_metrics for a path following run"
        )

    # Rise time: first crossing of 10 percent of the initial error.
    crossed = np.where(np.abs(error) <= 0.1 * abs(initial))[0]
    rise_time = float(crossed[0] * dt) if crossed.size else float("nan")

    # Overshoot: the largest excursion past zero, as a percentage of the step.
    opposite = -np.sign(initial) * error
    peak = float(np.max(opposite)) if opposite.size else 0.0
    overshoot = 100.0 * max(peak, 0.0) / abs(initial)

    # Settling time: last moment the error leaves the band.
    band = settle_band * abs(initial)
    outside = np.where(np.abs(error) > band)[0]
    settling_time = float((outside[-1] + 1) * dt) if outside.size else 0.0

    tail = error[-max(1, len(error) // 10) :]
    return {
        "rise_time_s": rise_time,
        "overshoot_pct": overshoot,
        "settling_time_s": settling_time,
        "steady_state_error_m": float(np.mean(np.abs(tail))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
    }
