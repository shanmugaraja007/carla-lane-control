"""Command line entry point.

Four subcommands, each of which is a stage of the pipeline that can be run and
inspected on its own:

    lanectl calibrate   fit camera intrinsics from chessboard views
    lanectl detect      run lane perception over images or a video
    lanectl simulate    close the loop on the bicycle model and score it
    lanectl bench       measure throughput
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraCalibration, calibrate_from_chessboards
from .config import PipelineConfig
from .control import (
    BicycleState,
    KinematicBicycle,
    PIDLateralController,
    StanleyLateralController,
    step_response_metrics,
    tracking_metrics,
)
from .pipeline import LanePipeline

log = logging.getLogger("lanectl")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def _images_in(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _load_calibration(cfg: PipelineConfig, override: Path | None) -> CameraCalibration | None:
    candidate = override or Path(cfg.calibration_path)
    if candidate.exists():
        log.info("using calibration %s", candidate)
        return CameraCalibration.load(candidate)
    log.warning("no calibration at %s, running on raw frames", candidate)
    return None


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def cmd_calibrate(args: argparse.Namespace) -> int:
    images = _images_in(args.images)
    if not images:
        log.error("no images under %s", args.images)
        return 1

    calib = calibrate_from_chessboards(images, tuple(args.pattern), args.square_size)
    calib.save(args.output)

    print(f"views used            {calib.n_views_used}/{len(images)}")
    print(f"RMS reprojection err  {calib.rms_reprojection_error:.4f} px")
    print(f"focal length fx, fy   {calib.fx:.2f}, {calib.fy:.2f}")
    print(f"principal point       {calib.cx:.2f}, {calib.cy:.2f}")
    print(f"distortion k1..k3,p1,p2  {np.round(calib.dist_coeffs.ravel(), 5).tolist()}")
    print(f"written to            {args.output}")

    if args.undistort_sample:
        sample = cv2.imread(str(images[0]))
        out = Path(args.output).parent / "undistort_example.jpg"
        cv2.imwrite(str(out), np.hstack([sample, calib.undistort(sample)]))
        print(f"before/after sample   {out}")
    return 0


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #
def cmd_detect(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    calib = _load_calibration(cfg, args.calibration)
    pipeline = LanePipeline(cfg, calib)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input.suffix.lower() in VIDEO_SUFFIXES:
        return _detect_video(args, pipeline, out_dir)

    images = _images_in(args.input)
    if not images:
        log.error("no images under %s", args.input)
        return 1

    records = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            log.warning("unreadable: %s", path)
            continue
        pipeline.reset()  # independent stills, so no prior carries over
        result = pipeline.process(frame)

        cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), result.overlay)
        if args.save_stages:
            cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), result.mask * 255)
            cv2.imwrite(str(out_dir / f"{path.stem}_warped.png"), result.warped_mask * 255)

        records.append(
            {
                "image": path.name,
                "valid": bool(result.lane.valid),
                "lateral_offset_m": round(result.lane.lateral_offset_m, 4),
                "heading_error_deg": round(float(np.degrees(result.lane.heading_error_rad)), 4),
                "curvature_radius_m": round(result.lane.curvature_radius_m, 1),
                "lane_width_m": round(result.lane.lane_width_m, 3),
                "steer_command": round(result.steer_command, 4),
                "left_px": result.lane.n_left_pixels,
                "right_px": result.lane.n_right_pixels,
                "latency_ms": round(result.latency_ms, 2),
            }
        )
        print(
            f"{path.name:24} offset {result.lane.lateral_offset_m:+.3f} m  "
            f"width {result.lane.lane_width_m:.2f} m  "
            f"R {result.lane.curvature_radius_m:8.0f} m  "
            f"{'ok' if result.lane.valid else 'REJECTED'}  "
            f"{result.latency_ms:.1f} ms"
        )

    summary = out_dir / "detections.json"
    summary.write_text(json.dumps(records, indent=2))
    valid = sum(r["valid"] for r in records)
    print(f"\n{valid}/{len(records)} frames accepted -> {summary}")
    return 0


def _detect_video(args: argparse.Namespace, pipeline: LanePipeline, out_dir: Path) -> int:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        log.error("cannot open video %s", args.input)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    records, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and index >= args.max_frames):
            break
        result = pipeline.process(frame)
        writer.write(result.overlay)
        records.append(
            {
                "frame": index,
                "valid": bool(result.lane.valid),
                "lateral_offset_m": round(result.lane.lateral_offset_m, 4),
                "heading_error_deg": round(float(np.degrees(result.lane.heading_error_rad)), 4),
                "curvature_radius_m": round(result.lane.curvature_radius_m, 1),
                "steer_command": round(result.steer_command, 4),
                "latency_ms": round(result.latency_ms, 2),
            }
        )
        index += 1
        if index % 100 == 0:
            log.info("processed %d frames", index)

    capture.release()
    writer.release()
    (out_dir / "detections.json").write_text(json.dumps(records, indent=2))

    valid = sum(r["valid"] for r in records)
    mean_latency = float(np.mean([r["latency_ms"] for r in records])) if records else 0.0
    print(f"{index} frames, {valid} accepted ({100*valid/max(index,1):.1f}%)")
    print(f"mean latency {mean_latency:.2f} ms  ->  {1000/max(mean_latency,1e-6):.1f} FPS")
    print(f"video -> {out_dir/'overlay.mp4'}")
    return 0


# --------------------------------------------------------------------------- #
# simulate
# --------------------------------------------------------------------------- #
def cmd_simulate(args: argparse.Namespace) -> int:
    """Close the loop on the kinematic bicycle and score the controller.

    The reference path is analytic, so the cross track error is exact and the
    result isolates the controller from the perception. Perception noise can be
    injected with --noise to see how the controller degrades.
    """
    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    ctrl_cfg = cfg.controller
    dt = ctrl_cfg.dt
    steps = int(args.duration / dt)
    rng = np.random.default_rng(args.seed)

    pid = PIDLateralController(ctrl_cfg)
    stanley = StanleyLateralController(ctrl_cfg)

    results: dict[str, dict] = {}

    for name in args.controllers:
        vehicle = KinematicBicycle(
            cfg.vehicle, BicycleState(x=0.0, y=args.initial_offset, yaw=0.0, speed=args.speed)
        )
        pid.reset()
        stanley.reset()

        errors, steers, xs, ys = [], [], [], []

        for _step in range(steps):
            s = vehicle.state
            # Reference path: a straight line for the step test, or a constant
            # radius arc when --curve-radius is given.
            if args.curve_radius:
                ref_y = args.curve_radius - np.sqrt(
                    max(args.curve_radius**2 - s.x**2, 1e-9)
                ) * np.sign(args.curve_radius)
                ref_heading = np.arctan(s.x / max(np.sqrt(max(args.curve_radius**2 - s.x**2, 1e-9)), 1e-9))
            else:
                ref_y, ref_heading = 0.0, 0.0

            cross_track = s.y - ref_y  # positive = right of the path
            heading_error = ref_heading - s.yaw

            measured = cross_track + (rng.normal(0, args.noise) if args.noise else 0.0)

            if name == "pid":
                steer = pid.step(measured, dt)
            elif name == "stanley":
                steer = stanley.step(measured, heading_error, s.speed)
            else:
                raise ValueError(f"unknown controller {name}")

            vehicle.step(steer, 0.0, dt, ctrl_cfg.max_steer_angle)

            errors.append(cross_track)
            steers.append(steer)
            xs.append(s.x)
            ys.append(s.y)

        err = np.array(errors)
        # A run that starts on the path has no step to characterise, so it is
        # scored as path tracking instead.
        if abs(args.initial_offset) < 1e-9:
            metrics = tracking_metrics(err, dt)
        else:
            metrics = step_response_metrics(err, dt)
        metrics["max_abs_steer"] = float(np.max(np.abs(steers)))
        results[name] = {
            "metrics": metrics,
            "error": [round(float(e), 5) for e in errors],
            "steer": [round(float(v), 5) for v in steers],
            "x": [round(float(v), 4) for v in xs],
            "y": [round(float(v), 4) for v in ys],
        }

        print(f"\n[{name}]")
        for key, value in metrics.items():
            print(f"  {key:24} {value:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": {
            "initial_offset_m": args.initial_offset,
            "speed_ms": args.speed,
            "duration_s": args.duration,
            "dt_s": dt,
            "curve_radius_m": args.curve_radius,
            "measurement_noise_std_m": args.noise,
            "gains": {"kp": ctrl_cfg.kp, "ki": ctrl_cfg.ki, "kd": ctrl_cfg.kd},
            "wheelbase_m": cfg.vehicle.wheelbase_m,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {args.output}")
    return 0


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def cmd_bench(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig()
    calib = _load_calibration(cfg, args.calibration)
    pipeline = LanePipeline(cfg, calib)

    images = _images_in(args.input)
    if not images:
        log.error("no images under %s", args.input)
        return 1
    frames = [cv2.imread(str(p)) for p in images]
    frames = [f for f in frames if f is not None]

    for f in frames[:2]:  # warm up
        pipeline.process(f, draw=False)

    for label, draw in (("perception only", False), ("with overlay", True)):
        timings = []
        for _ in range(args.repeats):
            for frame in frames:
                started = time.perf_counter()
                pipeline.process(frame, draw=draw)
                timings.append((time.perf_counter() - started) * 1000)
        arr = np.array(timings)
        print(
            f"{label:16}  n={arr.size:4d}  "
            f"mean {arr.mean():6.2f} ms  p50 {np.percentile(arr,50):6.2f}  "
            f"p95 {np.percentile(arr,95):6.2f}  -> {1000/arr.mean():.1f} FPS"
        )
    print(f"resolution {frames[0].shape[1]}x{frames[0].shape[0]}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lanectl",
        description="Vision based lane perception and lateral control.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="fit camera intrinsics from chessboard views")
    p.add_argument("images", type=Path, help="directory of chessboard images")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/calibration/camera.json"))
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6], metavar=("COLS", "ROWS"))
    p.add_argument("--square-size", type=float, default=1.0)
    p.add_argument("--undistort-sample", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("detect", help="run lane perception over images or a video")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/detect"))
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--save-stages", action="store_true", help="also write mask and warp")
    p.add_argument("--max-frames", type=int, default=0, help="0 means the whole video")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("simulate", help="closed loop controller test on the bicycle model")
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=Path("outputs/control/step_response.json"))
    p.add_argument("--controllers", nargs="+", default=["pid", "stanley"])
    p.add_argument("--initial-offset", type=float, default=1.0, help="metres")
    p.add_argument("--speed", type=float, default=15.0, help="m/s")
    p.add_argument("--duration", type=float, default=12.0, help="seconds")
    p.add_argument("--curve-radius", type=float, default=0.0, help="0 means straight")
    p.add_argument("--noise", type=float, default=0.0, help="measurement noise std, metres")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("bench", help="measure perception throughput")
    p.add_argument("input", type=Path)
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
