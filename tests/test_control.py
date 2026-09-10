"""Tests for the controllers, the vehicle model and the response metrics."""

from __future__ import annotations

import numpy as np
import pytest

from lanectl.config import ControllerConfig, VehicleConfig
from lanectl.control import (
    BicycleState,
    KinematicBicycle,
    PIDLateralController,
    StanleyLateralController,
    step_response_metrics,
    tracking_metrics,
)


@pytest.fixture
def cfg() -> ControllerConfig:
    return ControllerConfig()


class TestPID:
    def test_zero_error_gives_zero_command(self, cfg):
        assert PIDLateralController(cfg).step(0.0) == pytest.approx(0.0)

    def test_steers_left_when_the_vehicle_is_right_of_centre(self, cfg):
        # Positive cross track error means right of centre, so the command
        # must be negative to bring the vehicle back.
        assert PIDLateralController(cfg).step(0.5) < 0

    def test_steers_right_when_the_vehicle_is_left_of_centre(self, cfg):
        assert PIDLateralController(cfg).step(-0.5) > 0

    def test_command_is_clipped_to_the_steer_limit(self, cfg):
        controller = PIDLateralController(cfg)
        assert controller.step(1000.0) == pytest.approx(-cfg.steer_limit)

    def test_integral_is_clamped(self, cfg):
        controller = PIDLateralController(cfg)
        for _ in range(10_000):
            controller.step(1.0)
        assert abs(controller._integral) <= cfg.integral_limit + 1e-9

    def test_reset_clears_state(self, cfg):
        controller = PIDLateralController(cfg)
        for _ in range(50):
            controller.step(1.0)
        controller.reset()
        assert controller._integral == 0.0
        assert controller._previous_error is None
        assert controller.step(0.0) == pytest.approx(0.0)

    def test_rejects_a_non_positive_timestep(self, cfg):
        with pytest.raises(ValueError, match="dt must be positive"):
            PIDLateralController(cfg).step(0.1, dt=0.0)

    def test_derivative_opposes_a_growing_error(self, cfg):
        """With kd > 0 the command should be damped as the error grows."""
        without_history = PIDLateralController(cfg).step(0.5)

        with_history = PIDLateralController(cfg)
        with_history.step(0.1)
        second = with_history.step(0.5)

        # The error grew, so the derivative term pushes back harder.
        assert abs(second) > abs(without_history)


class TestStanley:
    def test_zero_error_gives_zero_command(self, cfg):
        assert StanleyLateralController(cfg).step(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_cross_track_term_shrinks_with_speed(self, cfg):
        controller = StanleyLateralController(cfg)
        slow = abs(controller.step(1.0, 0.0, 2.0))
        fast = abs(controller.step(1.0, 0.0, 30.0))
        assert fast < slow

    def test_finite_at_a_standstill(self, cfg):
        assert np.isfinite(StanleyLateralController(cfg).step(1.0, 0.0, 0.0))

    def test_heading_error_alone_produces_a_command(self, cfg):
        assert StanleyLateralController(cfg).step(0.0, 0.2, 10.0) > 0

    def test_command_is_clipped(self, cfg):
        assert abs(StanleyLateralController(cfg).step(1e6, 0.0, 1.0)) <= cfg.steer_limit


class TestKinematicBicycle:
    def test_straight_driving_advances_x_only(self):
        model = KinematicBicycle(VehicleConfig(), BicycleState(speed=10.0))
        for _ in range(20):
            model.step(0.0, 0.0, 0.05, 0.5236)
        assert model.state.x == pytest.approx(10.0, rel=1e-6)
        assert model.state.y == pytest.approx(0.0, abs=1e-9)
        assert model.state.yaw == pytest.approx(0.0, abs=1e-9)

    def test_positive_steer_turns_left(self):
        """Positive steer is a positive yaw rate, which is a left turn."""
        model = KinematicBicycle(VehicleConfig(), BicycleState(speed=10.0))
        for _ in range(20):
            model.step(0.3, 0.0, 0.05, 0.5236)
        assert model.state.yaw > 0
        assert model.state.y > 0

    def test_turn_radius_matches_the_bicycle_geometry(self):
        """Yaw rate should be v/L * tan(delta)."""
        cfg = VehicleConfig()
        model = KinematicBicycle(cfg, BicycleState(speed=10.0))
        delta_norm, max_angle, dt = 0.4, 0.5236, 0.01
        model.step(delta_norm, 0.0, dt, max_angle)
        expected = (10.0 / cfg.wheelbase_m) * np.tan(delta_norm * max_angle) * dt
        assert model.state.yaw == pytest.approx(expected, rel=1e-9)

    def test_speed_is_clipped_to_the_configured_range(self):
        model = KinematicBicycle(VehicleConfig(max_speed_ms=20.0), BicycleState(speed=19.0))
        model.step(0.0, 100.0, 1.0, 0.5236)
        assert model.state.speed == 20.0

        model = KinematicBicycle(VehicleConfig(), BicycleState(speed=1.0))
        model.step(0.0, -100.0, 1.0, 0.5236)
        assert model.state.speed == 0.0

    def test_steer_command_is_clipped_before_use(self):
        a = KinematicBicycle(VehicleConfig(), BicycleState(speed=10.0))
        b = KinematicBicycle(VehicleConfig(), BicycleState(speed=10.0))
        a.step(1.0, 0.0, 0.1, 0.5236)
        b.step(5.0, 0.0, 0.1, 0.5236)
        assert a.state.yaw == pytest.approx(b.state.yaw)


class TestMetrics:
    def test_step_response_of_a_clean_exponential_decay(self):
        dt = 0.05
        t = np.arange(0, 10, dt)
        error = np.exp(-t)  # no overshoot, settles smoothly

        m = step_response_metrics(error, dt)
        assert m["overshoot_pct"] == pytest.approx(0.0)
        assert m["rise_time_s"] == pytest.approx(2.3, abs=0.1)  # ln(10)
        assert m["steady_state_error_m"] < 1e-3

    def test_overshoot_is_detected(self):
        dt = 0.05
        t = np.arange(0, 10, dt)
        error = np.exp(-t) * np.cos(4 * t)  # rings through zero
        assert step_response_metrics(error, dt)["overshoot_pct"] > 5

    def test_rejects_an_empty_signal(self):
        with pytest.raises(ValueError, match="empty"):
            step_response_metrics(np.array([]), 0.05)

    def test_rejects_a_run_that_starts_on_the_path(self):
        with pytest.raises(ValueError, match="no step"):
            step_response_metrics(np.zeros(100), 0.05)

    def test_tracking_metrics_handle_a_run_starting_on_the_path(self):
        error = np.concatenate([np.zeros(10), np.full(90, 0.1)])
        m = tracking_metrics(error, 0.05)
        assert m["rmse_m"] == pytest.approx(np.sqrt(np.mean(error**2)))
        assert m["max_abs_error_m"] == pytest.approx(0.1)
        assert m["steady_state_error_m"] == pytest.approx(0.1)

    def test_tracking_metrics_reject_an_empty_signal(self):
        with pytest.raises(ValueError, match="empty"):
            tracking_metrics(np.array([]), 0.05)


class TestClosedLoop:
    """The property that actually matters: the loop converges."""

    def test_pid_drives_a_lateral_offset_to_near_zero(self):
        cfg, vehicle_cfg = ControllerConfig(), VehicleConfig()
        controller = PIDLateralController(cfg)
        model = KinematicBicycle(vehicle_cfg, BicycleState(y=1.0, speed=15.0))

        errors = []
        for _ in range(int(12 / cfg.dt)):
            errors.append(model.state.y)
            model.step(controller.step(model.state.y), 0.0, cfg.dt, cfg.max_steer_angle)

        assert abs(errors[-1]) < 0.05, "PID failed to converge"
        assert abs(errors[-1]) < abs(errors[0]) / 10

    def test_stanley_drives_a_lateral_offset_to_near_zero(self):
        cfg, vehicle_cfg = ControllerConfig(), VehicleConfig()
        controller = StanleyLateralController(cfg)
        model = KinematicBicycle(vehicle_cfg, BicycleState(y=1.0, speed=15.0))

        errors = []
        for _ in range(int(20 / cfg.dt)):
            s = model.state
            errors.append(s.y)
            model.step(controller.step(s.y, -s.yaw, s.speed), 0.0, cfg.dt, cfg.max_steer_angle)

        assert abs(errors[-1]) < 0.05, "Stanley failed to converge"

    @staticmethod
    def _run(kp: float, kd: float = 0.0, duration: float = 20.0) -> np.ndarray:
        cfg = ControllerConfig(kp=kp, ki=0.0, kd=kd)
        controller = PIDLateralController(cfg)
        model = KinematicBicycle(VehicleConfig(), BicycleState(y=1.0, speed=15.0))
        errors = []
        for _ in range(int(duration / cfg.dt)):
            errors.append(model.state.y)
            model.step(controller.step(model.state.y), 0.0, cfg.dt, cfg.max_steer_angle)
        return np.array(errors)

    @staticmethod
    def _first_zero_crossing(errors: np.ndarray, dt: float = 0.05) -> float:
        crossings = np.where(np.sign(errors[:-1]) != np.sign(errors[1:]))[0]
        return float(crossings[0] * dt) if crossings.size else float("inf")

    def test_a_higher_proportional_gain_reaches_the_lane_centre_sooner(self):
        """Higher gain gets there faster. It does not get there more calmly."""
        assert self._first_zero_crossing(self._run(0.8)) < self._first_zero_crossing(
            self._run(0.2)
        )

    def test_pure_proportional_gain_is_underdamped_at_speed(self):
        """A documented property of the plant, not a bug.

        With no derivative term the only damping is the vehicle's own geometry,
        which is not enough at 15 m/s: the loop rings. This is exactly why the
        shipped config carries a non zero kd, and why the step response in
        docs/RESULTS.md still shows roughly 30 percent overshoot.
        """
        high_gain = self._run(0.8, kd=0.0)
        overshoot = np.max(-high_gain) / abs(high_gain[0])
        assert overshoot > 0.3, "expected a strongly underdamped response"

    def test_the_derivative_term_reduces_overshoot(self):
        def overshoot(kd: float) -> float:
            errors = self._run(0.8, kd=kd)
            return float(np.max(-errors) / abs(errors[0]))

        assert overshoot(0.4) < overshoot(0.0)


class TestLongitudinalControl:
    """compute_control is the only decision logic in the ROS node.

    The node itself cannot run here, so this is what keeps it honest.
    """

    def test_below_target_speed_opens_the_throttle(self):
        from lanectl.ros2_node import compute_control

        cmd = compute_control(0.0, 0.0, speed_ms=5.0, target_speed_ms=12.0,
                              curvature_radius_m=1e6)
        assert cmd.throttle > 0 and cmd.brake == 0

    def test_above_target_speed_applies_the_brake(self):
        from lanectl.ros2_node import compute_control

        cmd = compute_control(0.0, 0.0, speed_ms=20.0, target_speed_ms=12.0,
                              curvature_radius_m=1e6)
        assert cmd.brake > 0 and cmd.throttle == 0

    def test_a_tight_curve_caps_the_target_speed(self):
        """v_max = sqrt(a_lat * R), so a 20 m radius caps at 6.3 m/s."""
        from lanectl.ros2_node import compute_control

        straight = compute_control(0.0, 0.0, 6.0, 12.0, curvature_radius_m=1e6)
        curve = compute_control(0.0, 0.0, 6.0, 12.0, curvature_radius_m=20.0)
        assert curve.throttle < straight.throttle

    def test_curve_cap_matches_the_lateral_acceleration_formula(self):
        import numpy as np

        from lanectl.ros2_node import compute_control

        radius, a_lat = 50.0, 2.0
        v_max = np.sqrt(a_lat * radius)  # 10 m/s
        # Sitting exactly at the cap means no throttle and no brake.
        cmd = compute_control(0.0, 0.0, speed_ms=v_max, target_speed_ms=30.0,
                              curvature_radius_m=radius, max_lateral_accel=a_lat)
        assert cmd.throttle == pytest.approx(0.0, abs=1e-9)
        assert cmd.brake == pytest.approx(0.0, abs=1e-9)

    def test_outputs_are_always_in_range(self):
        from lanectl.ros2_node import compute_control

        for speed, target in [(0, 100), (100, 0), (0, 0), (50, 50)]:
            cmd = compute_control(0.0, 5.0, speed, target, 1e6)
            assert 0.0 <= cmd.throttle <= 1.0
            assert 0.0 <= cmd.brake <= 1.0
            assert -1.0 <= cmd.steer <= 1.0

    def test_throttle_and_brake_are_never_both_on(self):
        from lanectl.ros2_node import compute_control

        for speed in range(0, 40, 3):
            cmd = compute_control(0.0, 0.0, float(speed), 15.0, 1e6)
            assert cmd.throttle == 0 or cmd.brake == 0

    def test_negative_radius_does_not_produce_a_nan(self):
        from lanectl.ros2_node import compute_control

        cmd = compute_control(0.0, 0.0, 10.0, 12.0, curvature_radius_m=-5.0)
        assert np.isfinite(cmd.throttle) and np.isfinite(cmd.brake)
