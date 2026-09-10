"""Measure the pixel to metre scale of the bird's eye view.

The warped image has no inherent scale. Two known real world lengths fix it:

  * lane width      3.70 m, the US Interstate standard (12 ft)
  * painted dash    3.05 m, the 10 ft line of the 10 ft line / 30 ft gap
                    pattern in the MUTCD. The 3:1 gap to line ratio means the
                    full period is 12.2 m, but a column profile can only
                    measure the painted segment, so that is what is used.

This script measures both directly out of a straight road frame instead of
assuming them, and prints the values to paste into the config. Run it once per
camera mounting; the numbers are a property of the mounting geometry and the
source quad, not of the footage.

    python scripts/measure_scale.py data/raw/test_images/straight_lines1.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lanectl.calibration import CameraCalibration  # noqa: E402
from lanectl.config import PipelineConfig  # noqa: E402
from lanectl.lane import histogram_peaks  # noqa: E402
from lanectl.perception import (  # noqa: E402
    lane_pixel_mask,
    perspective_matrices,
    region_of_interest,
    warp_to_birdseye,
)


def measure_lane_width_px(warped: np.ndarray) -> float:
    """Mean horizontal separation of the two lane lines, in warped pixels.

    Measured row by row over the lower half of the image, where the warp is
    least extrapolated, then averaged. A single histogram peak pair would be
    noisier and would not reveal a warp that is not quite rectifying.
    """
    height = warped.shape[0]
    left_base, right_base = histogram_peaks(warped)
    separations = []

    for row in range(height // 2, height):
        cols = np.nonzero(warped[row])[0]
        if cols.size < 2:
            continue
        left = cols[cols < (left_base + right_base) / 2]
        right = cols[cols >= (left_base + right_base) / 2]
        if left.size and right.size:
            separations.append(float(right.mean() - left.mean()))

    if not separations:
        raise ValueError("no rows contained both lane lines")
    return float(np.median(separations))


def measure_dash_length_px(warped: np.ndarray, side: str = "right") -> float:
    """Length of one painted dash along the warped image, in pixels.

    The dashed line is on the right in the sample footage. The row profile of
    that half of the image is a square wave; the median run length of the on
    state is the painted segment.
    """
    height, width = warped.shape[:2]
    half = warped[:, width // 2 :] if side == "right" else warped[:, : width // 2]
    profile = (half.sum(axis=1) > 0).astype(np.int8)

    runs, current = [], 0
    for value in profile:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)

    # Discard the shortest runs, which are speckle rather than paint, and the
    # longest, which mean the line on this side is solid rather than dashed.
    # A solid line carries no length information, so measuring it would give a
    # scale equal to the whole image height, silently and wrongly.
    upper = 0.25 * height
    runs = [r for r in runs if 5 <= r <= upper]
    if not runs:
        raise ValueError(
            f"no dash segments on the {side}; the line there is probably solid. "
            f"Try --side {'left' if side == 'right' else 'right'}, or use a frame "
            "where the dashed line is visible."
        )
    return float(np.median(runs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="a straight road frame")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=Path("outputs/calibration/camera.json"))
    parser.add_argument("--lane-width-m", type=float, default=3.7)
    parser.add_argument("--dash-length-m", type=float, default=3.05)
    parser.add_argument("--side", choices=["left", "right"], default="right",
                        help="which lane line is dashed in this frame")
    parser.add_argument("--save-warp", type=Path, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig()

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 1

    if args.calibration.exists():
        frame = CameraCalibration.load(args.calibration).undistort(frame)

    mask = region_of_interest(lane_pixel_mask(frame, cfg.thresholds))
    forward, _ = perspective_matrices(frame.shape[:2], cfg.perspective)
    warped = warp_to_birdseye(mask, forward)

    try:
        lane_px = measure_lane_width_px(warped)
        dash_px = measure_dash_length_px(warped, args.side)
    except ValueError as exc:
        print(f"measurement failed: {exc}", file=sys.stderr)
        return 1

    print(f"source image        {args.image}")
    print(f"lane width          {lane_px:.1f} px  ->  {args.lane_width_m} m")
    print(f"dash segment        {dash_px:.1f} px  ->  {args.dash_length_m} m")
    print(f"xm_per_pix          {args.lane_width_m / lane_px:.6f} m/px")
    print(f"ym_per_pix          {args.dash_length_m / dash_px:.6f} m/px")
    print()
    print("paste into the perspective section of your config:")
    print(f"  lane_width_px: {lane_px:.1f}")
    print(f"  dash_length_px: {dash_px:.1f}")

    if args.save_warp:
        args.save_warp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_warp), warped * 255)
        print(f"\nwarped mask -> {args.save_warp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
