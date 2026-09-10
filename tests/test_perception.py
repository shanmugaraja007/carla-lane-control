"""Tests for lane pixel extraction and the bird's eye view transform."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from lanectl.config import PerspectiveConfig, ThresholdConfig
from lanectl.perception import (
    lane_pixel_mask,
    perspective_matrices,
    region_of_interest,
    warp_to_birdseye,
)


def synthetic_road(width: int = 1280, height: int = 720) -> np.ndarray:
    """A grey road with one white and one yellow lane line, drawn in perspective.

    Good enough to exercise the geometry without pulling a dataset into the
    test suite, and it has a known ground truth the fitting tests can check.
    """
    image = np.full((height, width, 3), 60, dtype=np.uint8)
    horizon = int(height * 0.6)
    image[:horizon] = (170, 150, 130)  # sky

    # Two lines converging on a vanishing point at the image centre.
    apex = (width // 2, horizon)
    cv2.line(image, (int(width * 0.20), height), apex, (240, 240, 240), 14)
    cv2.line(image, (int(width * 0.80), height), apex, (60, 200, 235), 14)
    return image


class TestLanePixelMask:
    def test_finds_painted_lines_and_not_tarmac(self):
        image = synthetic_road()
        mask = lane_pixel_mask(image, ThresholdConfig())

        assert mask.dtype == np.uint8
        assert mask.shape == image.shape[:2]
        assert mask.max() == 1, "mask should be binary 0/1"

        # The painted lines are a small fraction of the frame. If the mask is
        # firing on most of the image the thresholds have collapsed.
        coverage = mask.mean()
        assert 0.001 < coverage < 0.20, f"implausible mask coverage {coverage:.4f}"

    def test_rejects_a_single_channel_image(self):
        with pytest.raises(ValueError, match="3 channel"):
            lane_pixel_mask(np.zeros((10, 10), dtype=np.uint8), ThresholdConfig())

    def test_uniform_image_produces_almost_nothing(self):
        flat = np.full((200, 200, 3), 90, dtype=np.uint8)
        mask = lane_pixel_mask(flat, ThresholdConfig())
        assert mask.sum() == 0


class TestPerspective:
    def test_forward_and_inverse_are_mutually_inverse(self):
        forward, inverse = perspective_matrices((720, 1280), PerspectiveConfig())
        identity = forward @ inverse
        identity = identity / identity[2, 2]
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-6)

    def test_round_trip_preserves_a_point_in_the_road_plane(self):
        cfg = PerspectiveConfig()
        forward, inverse = perspective_matrices((720, 1280), cfg)

        point = np.array([[[640.0, 700.0]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(point, forward)
        back = cv2.perspectiveTransform(warped, inverse)
        np.testing.assert_allclose(back, point, atol=1e-3)

    def test_warp_preserves_shape(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[600:700, 300:320] = 1
        forward, _ = perspective_matrices(mask.shape, PerspectiveConfig())
        warped = warp_to_birdseye(mask, forward)
        assert warped.shape == mask.shape

    def test_converging_lines_become_more_parallel(self):
        """The whole point of the warp: perspective convergence should go away."""
        image = synthetic_road()
        mask = region_of_interest(lane_pixel_mask(image, ThresholdConfig()))
        forward, _ = perspective_matrices(mask.shape, PerspectiveConfig())
        warped = warp_to_birdseye(mask, forward)

        def separation(m: np.ndarray, row: int) -> float:
            cols = np.nonzero(m[row])[0]
            return float(cols.max() - cols.min()) if cols.size > 1 else float("nan")

        near_before, far_before = separation(mask, 700), separation(mask, 460)
        near_after, far_after = separation(warped, 700), separation(warped, 200)

        # Before the warp the lines converge sharply with distance.
        assert far_before < near_before * 0.6
        # After it they stay within a modest band of each other.
        assert abs(far_after - near_after) < 0.45 * near_after


class TestRegionOfInterest:
    def test_zeroes_everything_above_the_horizon(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        out = region_of_interest(mask, top_fraction=0.5)
        assert out[:50].sum() == 0
        assert out[50:].sum() == 50 * 100

    def test_does_not_mutate_the_input(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        region_of_interest(mask, 0.5)
        assert mask.sum() == 100
