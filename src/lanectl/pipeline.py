"""End to end perception pipeline: frame in, lane measurement and overlay out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraCalibration
from .config import PipelineConfig
from .control import PIDLateralController
from .lane import LaneFit, fit_lane
from .perception import (
    lane_pixel_mask,
    perspective_matrices,
    region_of_interest,
    warp_to_birdseye,
)

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Everything one frame produced."""

    lane: LaneFit
    steer_command: float
    overlay: np.ndarray
    mask: np.ndarray
    warped_mask: np.ndarray
    latency_ms: float


class LanePipeline:
    """Stateful pipeline. Keeps the previous fit so the search can use a prior."""

    def __init__(
        self,
        cfg: PipelineConfig,
        calibration: CameraCalibration | None = None,
        controller: PIDLateralController | None = None,
    ):
        self.cfg = cfg
        self.calibration = calibration
        self.controller = controller or PIDLateralController(cfg.controller)
        self._prior: LaneFit | None = None
        self._dropped = 0
        self._matrices: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        self._prior = None
        self._dropped = 0
        self.controller.reset()

    def _get_matrices(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._matrices is None:
            self._matrices = perspective_matrices(shape, self.cfg.perspective)
        return self._matrices

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> FrameResult:
        """Run one frame through the whole pipeline.

        Args:
            frame_bgr: raw BGR frame from the camera.
            draw: build the annotated overlay. Turn it off when benchmarking
                the perception itself, since drawing is a large share of the
                per frame cost.

        Returns:
            The frame's lane measurement, steering command and visualisations.
        """
        started = time.perf_counter()

        if self.calibration is not None:
            frame_bgr = self.calibration.undistort(frame_bgr)

        mask = lane_pixel_mask(frame_bgr, self.cfg.thresholds)
        mask = region_of_interest(mask)

        forward, inverse = self._get_matrices(frame_bgr.shape[:2])
        warped = warp_to_birdseye(mask, forward)

        prior = self._prior if self._dropped < self.cfg.lane_fit.max_dropped_frames else None
        lane = fit_lane(warped, self.cfg.lane_fit, self.cfg.perspective, prior)

        if lane.valid:
            self._prior = lane
            self._dropped = 0
        else:
            self._dropped += 1

        steer = self.controller.step(lane.lateral_offset_m)

        overlay = (
            draw_lane_overlay(frame_bgr, lane, inverse, self.cfg, steer)
            if draw
            else frame_bgr
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(
            lane=lane,
            steer_command=steer,
            overlay=overlay,
            mask=mask,
            warped_mask=warped,
            latency_ms=latency_ms,
        )


def draw_lane_overlay(
    frame_bgr: np.ndarray,
    lane: LaneFit,
    inverse_matrix: np.ndarray,
    cfg: PipelineConfig,
    steer_command: float = 0.0,
) -> np.ndarray:
    """Project the fitted lane back onto the camera image and annotate it."""
    out = frame_bgr.copy()
    height, width = frame_bgr.shape[:2]

    if lane.left_fit is not None and lane.right_fit is not None:
        plot_y = np.linspace(0, height - 1, height)
        left_x = np.polyval(lane.left_fit, plot_y)
        right_x = np.polyval(lane.right_fit, plot_y)

        canvas = np.zeros_like(frame_bgr)
        left_pts = np.array([np.transpose(np.vstack([left_x, plot_y]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])
        points = np.hstack((left_pts, right_pts))

        colour = (0, 200, 90) if lane.valid else (0, 120, 200)
        cv2.fillPoly(canvas, np.int32([points]), colour)
        cv2.polylines(canvas, np.int32([left_pts]), False, (255, 220, 60), 18)
        cv2.polylines(canvas, np.int32([right_pts]), False, (255, 220, 60), 18)

        unwarped = cv2.warpPerspective(canvas, inverse_matrix, (width, height))
        out = cv2.addWeighted(out, 1.0, unwarped, 0.35, 0)

    lines = [
        f"lateral offset  {lane.lateral_offset_m:+.3f} m",
        f"heading error   {np.degrees(lane.heading_error_rad):+.2f} deg",
        f"curve radius    {lane.curvature_radius_m:8.0f} m",
        f"lane width      {lane.lane_width_m:.2f} m",
        f"steer command   {steer_command:+.3f}",
        f"fit             {'ok' if lane.valid else 'REJECTED'}",
    ]
    _draw_panel(out, lines)
    return out


def _draw_panel(image: np.ndarray, lines: list[str]) -> None:
    """Draw a readable text panel in the top left corner, in place."""
    pad, line_height = 12, 26
    box_w = 340
    box_h = pad * 2 + line_height * len(lines)

    panel = image[pad : pad + box_h, pad : pad + box_w]
    if panel.size:
        image[pad : pad + box_h, pad : pad + box_w] = cv2.addWeighted(
            panel, 0.35, np.zeros_like(panel), 0.65, 0
        )

    for i, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (pad + 12, pad + line_height * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 245, 240),
            1,
            cv2.LINE_AA,
        )
