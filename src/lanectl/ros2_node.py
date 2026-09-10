"""ROS 2 node wrapping the lane pipeline.

NOT EXECUTED IN THIS REPOSITORY'S RESULTS. There is no ROS 2 installation in
the environment the results in docs/RESULTS.md were produced on, so this module
is published as integration code with its logic factored out and unit tested,
not as a measured result. docs/RESULTS.md says exactly what was and was not run.

The ROS specific surface is kept deliberately thin. Everything that decides
anything lives in ``LanePipeline`` and in ``compute_control``, both of which
are importable and tested without ROS. What is left here is message conversion
and topic wiring.

Topics follow the CARLA ROS bridge naming so this node drops into a standard
bridge setup:

    subscribes  /carla/<role>/rgb_front/image          sensor_msgs/Image
                /carla/<role>/odometry                 nav_msgs/Odometry
    publishes   /carla/<role>/vehicle_control_cmd      carla_msgs/CarlaEgoVehicleControl
                /lanectl/lane_state                    std_msgs/Float32MultiArray
                /lanectl/overlay                       sensor_msgs/Image

Run it with:

    ros2 run lanectl lane_control_node --ros-args \
        -p config:=configs/highway.yaml -p role_name:=ego_vehicle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .pipeline import LanePipeline

log = logging.getLogger(__name__)


@dataclass
class ControlCommand:
    """What the node publishes, independent of any ROS message type."""

    throttle: float
    steer: float
    brake: float


def compute_control(
    lateral_offset_m: float,
    steer_command: float,
    speed_ms: float,
    target_speed_ms: float,
    curvature_radius_m: float,
    kp_speed: float = 0.2,
    max_lateral_accel: float = 2.0,
) -> ControlCommand:
    """Turn a lane measurement into a throttle, steer and brake triple.

    Longitudinal control is a proportional term on the speed error, with the
    target speed capped by the lateral acceleration the upcoming curve implies:
    v_max = sqrt(a_lat_max * R). Without that cap the vehicle enters curves at
    the straight line target and the lateral controller cannot save it.

    Args:
        lateral_offset_m: distance from lane centre, positive right.
        steer_command: normalised steering from the lateral controller.
        speed_ms: current forward speed.
        target_speed_ms: desired cruising speed.
        curvature_radius_m: radius of the upcoming curve.
        kp_speed: proportional gain on speed error.
        max_lateral_accel: comfort limit in m/s^2.

    Returns:
        A ControlCommand with each field in its valid range.
    """
    curve_limited = float(np.sqrt(max(max_lateral_accel * curvature_radius_m, 0.0)))
    target = min(target_speed_ms, curve_limited)

    speed_error = target - speed_ms
    raw = kp_speed * speed_error

    throttle = float(np.clip(raw, 0.0, 1.0))
    brake = float(np.clip(-raw, 0.0, 1.0))

    return ControlCommand(
        throttle=throttle,
        steer=float(np.clip(steer_command, -1.0, 1.0)),
        brake=brake,
    )


def _require_ros():
    """Import ROS 2, with an error that says what to do if it is missing."""
    try:
        import rclpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "ROS 2 (rclpy) is not installed. This node needs a ROS 2 Humble or "
            "newer environment plus the CARLA ROS bridge; see docs/PIPELINE.md. "
            "The perception and control code itself runs without ROS: use "
            "`lanectl detect` and `lanectl simulate`."
        ) from exc


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs ROS 2
    """Entry point for ``ros2 run``."""
    _require_ros()

    import rclpy
    from carla_msgs.msg import CarlaEgoVehicleControl
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray

    class LaneControlNode(Node):
        def __init__(self) -> None:
            super().__init__("lane_control_node")

            self.declare_parameter("config", "configs/highway.yaml")
            self.declare_parameter("role_name", "ego_vehicle")
            self.declare_parameter("target_speed_ms", 12.0)
            self.declare_parameter("publish_overlay", True)

            config_path = self.get_parameter("config").value
            role = self.get_parameter("role_name").value
            self.target_speed = float(self.get_parameter("target_speed_ms").value)
            self.publish_overlay = bool(self.get_parameter("publish_overlay").value)

            cfg = PipelineConfig.load(config_path)
            calibration = None
            try:
                from .calibration import CameraCalibration

                calibration = CameraCalibration.load(cfg.calibration_path)
            except (FileNotFoundError, OSError):
                self.get_logger().warn(
                    f"no calibration at {cfg.calibration_path}; "
                    "running on raw frames, metric output will be biased"
                )

            self.pipeline = LanePipeline(cfg, calibration)
            self.bridge = CvBridge()
            self.speed = 0.0

            # Sensor data is best effort: a dropped frame is better than a
            # queue that grows until the control loop is acting on stale images.
            sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.create_subscription(
                Image, f"/carla/{role}/rgb_front/image", self.on_image, sensor_qos
            )
            self.create_subscription(
                Odometry, f"/carla/{role}/odometry", self.on_odometry, sensor_qos
            )

            self.control_pub = self.create_publisher(
                CarlaEgoVehicleControl, f"/carla/{role}/vehicle_control_cmd", 10
            )
            self.state_pub = self.create_publisher(
                Float32MultiArray, "/lanectl/lane_state", 10
            )
            self.overlay_pub = self.create_publisher(Image, "/lanectl/overlay", 1)

            self.get_logger().info(f"lane_control_node up, role={role}, cfg={config_path}")

        def on_odometry(self, msg) -> None:
            v = msg.twist.twist.linear
            self.speed = float(np.hypot(v.x, v.y))

        def on_image(self, msg) -> None:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            result = self.pipeline.process(frame, draw=self.publish_overlay)
            lane = result.lane

            command = compute_control(
                lateral_offset_m=lane.lateral_offset_m,
                steer_command=result.steer_command,
                speed_ms=self.speed,
                target_speed_ms=self.target_speed,
                curvature_radius_m=lane.curvature_radius_m,
            )

            control = CarlaEgoVehicleControl()
            control.header.stamp = msg.header.stamp
            control.throttle = command.throttle
            control.steer = command.steer
            control.brake = command.brake
            control.hand_brake = False
            control.reverse = False
            control.manual_gear_shift = False
            self.control_pub.publish(control)

            state = Float32MultiArray()
            state.data = [
                float(lane.lateral_offset_m),
                float(lane.heading_error_rad),
                float(lane.curvature_radius_m),
                float(lane.lane_width_m),
                float(result.steer_command),
                1.0 if lane.valid else 0.0,
                float(result.latency_ms),
            ]
            self.state_pub.publish(state)

            if self.publish_overlay:
                overlay = self.bridge.cv2_to_imgmsg(result.overlay, encoding="bgr8")
                overlay.header = msg.header
                self.overlay_pub.publish(overlay)

    rclpy.init(args=argv)
    node = LaneControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
