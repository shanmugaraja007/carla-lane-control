"""Tests for lane fitting, curvature and the derived control quantities."""

from __future__ import annotations

import numpy as np
import pytest

from lanectl.config import LaneFitConfig, PerspectiveConfig
from lanectl.lane import (
    LaneFit,
    curvature_radius,
    fit_lane,
    histogram_peaks,
    sliding_window_search,
)


def warped_lane(
    width: int = 1280,
    height: int = 720,
    left_x: int = 367,
    right_x: int = 913,
    curvature: float = 0.0,
    thickness: int = 12,
) -> np.ndarray:
    """A synthetic bird's eye view mask with two lane lines at known positions.

    The defaults put the lines 546 px apart, which is the separation measured
    on the real footage, so a fit of this mask should report a 3.7 m lane.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        offset = curvature * (height - y) ** 2
        for base in (left_x, right_x):
            x = int(base + offset)
            mask[y, max(0, x - thickness // 2) : min(width, x + thickness // 2)] = 1
    return mask


class TestHistogramPeaks:
    def test_finds_both_lane_bases(self):
        mask = warped_lane()
        left, right = histogram_peaks(mask)
        assert abs(left - 367) < 15
        assert abs(right - 913) < 15

    def test_left_peak_is_left_of_right_peak(self):
        left, right = histogram_peaks(warped_lane())
        assert left < right


class TestSlidingWindow:
    def test_separates_left_from_right_pixels(self):
        mask = warped_lane()
        left_idx, right_idx = sliding_window_search(mask, LaneFitConfig())
        assert len(left_idx) > 500
        assert len(right_idx) > 500

        nonzero_x = np.array(mask.nonzero()[1])
        assert nonzero_x[left_idx].mean() < nonzero_x[right_idx].mean()

    def test_returns_empty_on_a_blank_mask(self):
        left_idx, right_idx = sliding_window_search(
            np.zeros((720, 1280), dtype=np.uint8), LaneFitConfig()
        )
        assert len(left_idx) == 0 and len(right_idx) == 0


class TestCurvatureRadius:
    def test_straight_line_has_a_very_large_radius(self):
        assert curvature_radius(np.array([0.0, 0.0, 5.0]), 10.0) >= 1e6

    def test_matches_the_analytic_radius_of_a_parabola(self):
        # x = a y^2. At the vertex y = 0 the radius of curvature is 1/(2a).
        a = 0.002
        assert curvature_radius(np.array([a, 0.0, 0.0]), 0.0) == pytest.approx(
            1 / (2 * a), rel=1e-6
        )

    def test_tighter_curves_give_smaller_radii(self):
        gentle = curvature_radius(np.array([1e-4, 0.0, 0.0]), 0.0)
        sharp = curvature_radius(np.array([1e-2, 0.0, 0.0]), 0.0)
        assert sharp < gentle


class TestFitLane:
    def test_recovers_a_known_lane_width(self):
        persp = PerspectiveConfig()  # 546 px == 3.7 m
        fit = fit_lane(warped_lane(), LaneFitConfig(), persp)

        assert fit.valid
        assert fit.lane_width_m == pytest.approx(3.7, abs=0.15)
        assert fit.left_fit is not None and fit.right_fit is not None

    def test_centred_vehicle_reports_near_zero_offset(self):
        # Lines placed symmetrically about the image centre column.
        mask = warped_lane(left_x=640 - 273, right_x=640 + 273)
        fit = fit_lane(mask, LaneFitConfig(), PerspectiveConfig())
        assert abs(fit.lateral_offset_m) < 0.05

    def test_offset_sign_is_positive_when_the_lane_sits_left_of_the_vehicle(self):
        # Shifting both lines left means the lane centre is left of the image
        # centre, so the vehicle is right of the lane centre: positive offset.
        mask = warped_lane(left_x=367 - 60, right_x=913 - 60)
        fit = fit_lane(mask, LaneFitConfig(), PerspectiveConfig())
        assert fit.lateral_offset_m > 0.2

    def test_rejects_an_implausible_lane_width(self):
        # 120 px apart is well under the 2.5 m floor.
        mask = warped_lane(left_x=600, right_x=720)
        fit = fit_lane(mask, LaneFitConfig(), PerspectiveConfig())
        assert not fit.valid

    def test_blank_mask_degrades_instead_of_raising(self):
        fit = fit_lane(np.zeros((720, 1280), np.uint8), LaneFitConfig(), PerspectiveConfig())
        assert not fit.valid
        assert fit.left_fit is None

    def test_a_bad_frame_keeps_the_prior_geometry(self):
        good = fit_lane(warped_lane(), LaneFitConfig(), PerspectiveConfig())
        blank = np.zeros((720, 1280), np.uint8)
        degraded = fit_lane(blank, LaneFitConfig(), PerspectiveConfig(), prior=good)

        assert not degraded.valid
        assert degraded.left_fit is not None
        np.testing.assert_allclose(degraded.left_fit, good.left_fit)
        assert degraded.lateral_offset_m == good.lateral_offset_m

    def test_curved_lane_reports_a_finite_radius(self):
        fit = fit_lane(warped_lane(curvature=2e-4), LaneFitConfig(), PerspectiveConfig())
        assert fit.valid
        assert 10 < fit.curvature_radius_m < 5000

    def test_straight_lane_reports_a_large_radius(self):
        fit = fit_lane(warped_lane(curvature=0.0), LaneFitConfig(), PerspectiveConfig())
        assert fit.curvature_radius_m > 5000

    def test_centre_fit_is_between_the_two_edges(self):
        fit = fit_lane(warped_lane(), LaneFitConfig(), PerspectiveConfig())
        centre = fit.centre_fit
        assert centre is not None
        y = 700.0
        assert np.polyval(fit.left_fit, y) < np.polyval(centre, y) < np.polyval(fit.right_fit, y)

    def test_centre_fit_is_none_without_a_fit(self):
        empty = LaneFit(None, None, None, None, 0, 0, 1e6, 0, False, 0, 0)
        assert empty.centre_fit is None
