"""Lane line fitting, curvature and lateral offset.

The warped mask goes in, two second order polynomials come out, and from those
the three numbers the controller actually needs: how far the vehicle is from
the lane centre, how much its heading differs from the lane direction, and how
tight the upcoming curve is.

Fitting x as a function of y rather than the other way round is deliberate. A
lane line in bird's eye view is close to vertical, and a function of x would be
near singular exactly where the lane matters most.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import LaneFitConfig, PerspectiveConfig

log = logging.getLogger(__name__)


@dataclass
class LaneFit:
    """The lane as measured from one frame."""

    left_fit: np.ndarray | None  # x = a y^2 + b y + c, in warped pixels
    right_fit: np.ndarray | None
    left_fit_m: np.ndarray | None  # the same fit in metres
    right_fit_m: np.ndarray | None
    lateral_offset_m: float  # positive when the vehicle is right of centre
    heading_error_rad: float  # positive when the lane bends left of the heading
    curvature_radius_m: float
    lane_width_m: float
    valid: bool
    n_left_pixels: int
    n_right_pixels: int

    @property
    def centre_fit(self) -> np.ndarray | None:
        """The lane centre line, averaged from the two edges."""
        if self.left_fit is None or self.right_fit is None:
            return None
        return (self.left_fit + self.right_fit) / 2.0


def histogram_peaks(warped_mask: np.ndarray) -> tuple[int, int]:
    """Find the two lane line bases from a column histogram of the lower half.

    Returns:
        (left_base_x, right_base_x) in warped pixel columns.
    """
    bottom = warped_mask[warped_mask.shape[0] // 2 :, :]
    histogram = bottom.sum(axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)
    return left_base, right_base


def sliding_window_search(
    warped_mask: np.ndarray, cfg: LaneFitConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Locate lane pixels with a bottom up sliding window search.

    Args:
        warped_mask: binary bird's eye view mask.
        cfg: window count, margin and recentring threshold.

    Returns:
        (left_indices, right_indices) into the flattened nonzero arrays,
        expressed as boolean index arrays over the nonzero pixels.
    """
    height, width = warped_mask.shape[:2]
    left_current, right_current = histogram_peaks(warped_mask)

    nonzero = warped_mask.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    window_height = height // cfg.n_windows
    left_hits: list[np.ndarray] = []
    right_hits: list[np.ndarray] = []

    for window in range(cfg.n_windows):
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height

        in_band = (nonzero_y >= y_low) & (nonzero_y < y_high)

        left_in = in_band & (nonzero_x >= left_current - cfg.window_margin) & (
            nonzero_x < left_current + cfg.window_margin
        )
        right_in = in_band & (nonzero_x >= right_current - cfg.window_margin) & (
            nonzero_x < right_current + cfg.window_margin
        )

        left_hits.append(left_in.nonzero()[0])
        right_hits.append(right_in.nonzero()[0])

        # Recentre the next window on this one's centre of mass, but only when
        # there is enough evidence. Recentring on three stray pixels is how a
        # search walks off a dashed line and onto the kerb.
        if left_in.sum() > cfg.min_pixels_to_recentre:
            left_current = int(np.mean(nonzero_x[left_in]))
        if right_in.sum() > cfg.min_pixels_to_recentre:
            right_current = int(np.mean(nonzero_x[right_in]))

    return np.concatenate(left_hits), np.concatenate(right_hits)


def search_around_prior(
    warped_mask: np.ndarray,
    left_fit: np.ndarray,
    right_fit: np.ndarray,
    cfg: LaneFitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a band around the previous frame's fit.

    Far cheaper than a fresh window search and much less likely to jump lanes,
    but it inherits whatever error the prior had, so the caller has to be
    willing to fall back when the result stops making sense.
    """
    nonzero = warped_mask.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_centre = np.polyval(left_fit, nonzero_y)
    right_centre = np.polyval(right_fit, nonzero_y)

    left_in = np.abs(nonzero_x - left_centre) < cfg.search_margin
    right_in = np.abs(nonzero_x - right_centre) < cfg.search_margin

    return left_in.nonzero()[0], right_in.nonzero()[0]


def _fit_or_none(x: np.ndarray, y: np.ndarray, min_points: int = 50) -> np.ndarray | None:
    if len(x) < min_points:
        return None
    return np.polyfit(y, x, 2)


def curvature_radius(fit_m: np.ndarray, y_eval_m: float) -> float:
    """Radius of curvature of x = a y^2 + b y + c at a given y, in metres.

    A straight line has an infinite radius; this returns a large finite number
    instead so downstream arithmetic stays well behaved.
    """
    a, b, _c = fit_m
    if abs(a) < 1e-9:
        return 1e6
    return float(((1 + (2 * a * y_eval_m + b) ** 2) ** 1.5) / abs(2 * a))


def fit_lane(
    warped_mask: np.ndarray,
    cfg: LaneFitConfig,
    persp: PerspectiveConfig,
    prior: LaneFit | None = None,
) -> LaneFit:
    """Fit both lane lines and derive the control quantities.

    Args:
        warped_mask: binary bird's eye view mask.
        cfg: fitting settings.
        persp: supplies the pixel to metre scale.
        prior: the previous frame's fit, used to seed the search and to
            smooth the result. Pass None for the first frame.

    Returns:
        A LaneFit. Check ``valid`` before trusting the numbers: an invalid fit
        still carries the prior's values so a controller can coast through a
        few bad frames rather than lurching.
    """
    height, width = warped_mask.shape[:2]
    nonzero = warped_mask.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    use_prior = (
        prior is not None
        and prior.left_fit is not None
        and prior.right_fit is not None
        and prior.valid
    )

    if use_prior:
        left_idx, right_idx = search_around_prior(
            warped_mask, prior.left_fit, prior.right_fit, cfg
        )
        if len(left_idx) < 50 or len(right_idx) < 50:
            # The prior has drifted off the paint. Start over.
            left_idx, right_idx = sliding_window_search(warped_mask, cfg)
    else:
        left_idx, right_idx = sliding_window_search(warped_mask, cfg)

    left_x, left_y = nonzero_x[left_idx], nonzero_y[left_idx]
    right_x, right_y = nonzero_x[right_idx], nonzero_y[right_idx]

    left_fit = _fit_or_none(left_x, left_y)
    right_fit = _fit_or_none(right_x, right_y)

    if left_fit is None or right_fit is None:
        log.debug("lane fit failed: %d left px, %d right px", len(left_x), len(right_x))
        return _degraded(prior, len(left_x), len(right_x))

    # Smooth against the prior before anything is derived from the fit, so the
    # offset and curvature inherit the smoothing for free.
    if use_prior and cfg.smoothing > 0:
        alpha = cfg.smoothing
        left_fit = alpha * prior.left_fit + (1 - alpha) * left_fit
        right_fit = alpha * prior.right_fit + (1 - alpha) * right_fit

    xm, ym = persp.xm_per_pix, persp.ym_per_pix
    left_fit_m = np.polyfit(left_y * ym, left_x * xm, 2)
    right_fit_m = np.polyfit(right_y * ym, right_x * xm, 2)

    # Everything below is evaluated at the bottom of the warped image, which is
    # the closest point to the vehicle.
    y_bottom = float(height - 1)
    left_x_bottom = float(np.polyval(left_fit, y_bottom))
    right_x_bottom = float(np.polyval(right_fit, y_bottom))

    lane_width_m = (right_x_bottom - left_x_bottom) * xm

    # The camera sits on the vehicle centreline, so the image centre column is
    # where the vehicle is. Positive offset means the vehicle is right of the
    # lane centre, which is the sign convention the controller expects.
    lane_centre_px = (left_x_bottom + right_x_bottom) / 2.0
    lateral_offset_m = (width / 2.0 - lane_centre_px) * xm

    centre_fit_m = (left_fit_m + right_fit_m) / 2.0
    y_bottom_m = y_bottom * ym
    # dx/dy of the centre line is the tangent of the angle between the lane and
    # the vehicle's forward axis.
    dx_dy = 2 * centre_fit_m[0] * y_bottom_m + centre_fit_m[1]
    heading_error_rad = float(np.arctan(dx_dy))

    radius = 0.5 * (
        curvature_radius(left_fit_m, y_bottom_m)
        + curvature_radius(right_fit_m, y_bottom_m)
    )

    valid = cfg.min_lane_width_m <= lane_width_m <= cfg.max_lane_width_m
    if not valid:
        log.debug("rejecting fit: lane width %.2f m out of range", lane_width_m)

    return LaneFit(
        left_fit=left_fit,
        right_fit=right_fit,
        left_fit_m=left_fit_m,
        right_fit_m=right_fit_m,
        lateral_offset_m=lateral_offset_m,
        heading_error_rad=heading_error_rad,
        curvature_radius_m=radius,
        lane_width_m=lane_width_m,
        valid=valid,
        n_left_pixels=len(left_x),
        n_right_pixels=len(right_x),
    )


def _degraded(prior: LaneFit | None, n_left: int, n_right: int) -> LaneFit:
    """Return the prior's geometry marked invalid, or an empty fit."""
    if prior is not None:
        return LaneFit(
            left_fit=prior.left_fit,
            right_fit=prior.right_fit,
            left_fit_m=prior.left_fit_m,
            right_fit_m=prior.right_fit_m,
            lateral_offset_m=prior.lateral_offset_m,
            heading_error_rad=prior.heading_error_rad,
            curvature_radius_m=prior.curvature_radius_m,
            lane_width_m=prior.lane_width_m,
            valid=False,
            n_left_pixels=n_left,
            n_right_pixels=n_right,
        )
    return LaneFit(
        left_fit=None,
        right_fit=None,
        left_fit_m=None,
        right_fit_m=None,
        lateral_offset_m=0.0,
        heading_error_rad=0.0,
        curvature_radius_m=1e6,
        lane_width_m=0.0,
        valid=False,
        n_left_pixels=n_left,
        n_right_pixels=n_right,
    )
