"""Drive a CARLA ego vehicle with the lane pipeline, without ROS in the loop.

NOT EXECUTED IN THIS REPOSITORY'S RESULTS. The environment the results were
produced on has no CARLA server. This script is published so the loop can be
reproduced by anyone who does have one; docs/RESULTS.md is explicit about which
numbers came from a local run and which did not.

    # terminal 1
    ./CarlaUE4.sh -quality-level=Epic -carla-server

    # terminal 2
    python scripts/carla_run.py --config configs/highway.yaml \
        --town Town04 --duration 120 --output outputs/carla

Town04 is the right default: it has a highway loop with clean lane markings,
which is what the classical perception in this repository is tuned for. Town10
and the urban maps have junctions and worn paint that the thresholds were never
meant to survive, and running there without retuning will produce a stream of
rejected frames rather than a fair test.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lanectl.config import PipelineConfig  # noqa: E402
from lanectl.pipeline import LanePipeline  # noqa: E402
from lanectl.ros2_node import compute_control  # noqa: E402

log = logging.getLogger("carla_run")


def _require_carla():
    try:
        import carla  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the CARLA Python API is not installed. Install the client that "
            "matches your server build:\n"
            "    pip install carla==0.9.15\n"
            "and start the simulator before running this script. See "
            "docs/PIPELINE.md for the full setup."
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/highway.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--target-speed", type=float, default=12.0, help="m/s")
    parser.add_argument("--fps", type=int, default=20, help="fixed simulation rate")
    parser.add_argument("--output", type=Path, default=Path("outputs/carla"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    _require_carla()

    import carla
    import cv2

    args.output.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.town)

    # Synchronous mode with a fixed step. Without it the perception and the
    # controller see a variable dt and the run is not reproducible.
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    blueprints = world.get_blueprint_library()
    vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
    spawn = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn)

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", str(1.0 / args.fps))
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        attach_to=vehicle,
    )

    frames: list[np.ndarray] = []
    camera.listen(
        lambda image: frames.append(
            np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[
                :, :, :3
            ].copy()
        )
    )

    cfg = PipelineConfig.load(args.config)
    pipeline = LanePipeline(cfg)

    writer = cv2.VideoWriter(
        str(args.output / "carla_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1280, 720),
    )

    records: list[dict] = []
    steps = int(args.duration * args.fps)
    started = time.perf_counter()

    try:
        for step in range(steps):
            world.tick()
            if not frames:
                continue
            frame = frames.pop()
            frames.clear()  # only ever act on the newest frame

            result = pipeline.process(frame)
            velocity = vehicle.get_velocity()
            speed = float(np.hypot(velocity.x, velocity.y))

            command = compute_control(
                lateral_offset_m=result.lane.lateral_offset_m,
                steer_command=result.steer_command,
                speed_ms=speed,
                target_speed_ms=args.target_speed,
                curvature_radius_m=result.lane.curvature_radius_m,
            )
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=command.throttle, steer=command.steer, brake=command.brake
                )
            )

            writer.write(result.overlay)

            # CARLA's own map query gives a ground truth lane centre, which is
            # what makes a CARLA run worth more than an offline one: the
            # measured offset can be scored against it.
            waypoint = world.get_map().get_waypoint(vehicle.get_location())
            truth_offset = float(
                vehicle.get_location().distance(waypoint.transform.location)
            )

            records.append(
                {
                    "step": step,
                    "t": step / args.fps,
                    "speed_ms": round(speed, 3),
                    "measured_offset_m": round(result.lane.lateral_offset_m, 4),
                    "ground_truth_offset_m": round(truth_offset, 4),
                    "steer": round(command.steer, 4),
                    "throttle": round(command.throttle, 4),
                    "brake": round(command.brake, 4),
                    "valid": bool(result.lane.valid),
                    "latency_ms": round(result.latency_ms, 2),
                }
            )
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        writer.release()
        world.apply_settings(original_settings)

    (args.output / "carla_run.json").write_text(json.dumps(records, indent=2))

    if records:
        valid = sum(r["valid"] for r in records)
        errors = np.array(
            [abs(r["measured_offset_m"]) - r["ground_truth_offset_m"] for r in records]
        )
        print(f"{len(records)} steps in {time.perf_counter() - started:.1f} s wall clock")
        print(f"frames accepted   {valid}/{len(records)} ({100*valid/len(records):.1f}%)")
        print(f"offset error MAE  {np.mean(np.abs(errors)):.4f} m vs CARLA ground truth")
        print(f"outputs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
