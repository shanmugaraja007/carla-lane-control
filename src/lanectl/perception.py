"""Lane pixel extraction and the bird's eye view transform.

Two stages live here. The first turns a colour road image into a binary mask
where the lane paint is on and everything else is off. The second warps that
mask into a top down view, where lane lines that converge under perspective
become parallel and a polynomial fit means something metric.

The thresholding is deliberately classical rather than learned. On the sample
footage it is competitive with a segmentation network, it runs at video rate on
a CPU, and every failure it has is one you can see and reason about, which
matters more in a controller loop than the last few points of IoU.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import PerspectiveConfig, ThresholdConfig

log = logging.getLogger(__name__)


def _scaled_sobel_x(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Absolute Sobel gradient in x, rescaled to 0..255."""
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    absolute = np.absolute(sobel)
    peak = absolute.max()
    if peak == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    return np.uint8(255 * absolute / peak)


def lane_pixel_mask(image_bgr: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    """Isolate probable lane paint.

    Four cues are combined. The lightness channel of HLS finds white paint,
    the B channel of LAB finds yellow paint, the saturation channel of HLS
    catches paint that survives shadow, and the x gradient catches the edges
    of anything the colour channels miss.

    Args:
        image_bgr: undistorted BGR frame.
        cfg: threshold settings.

    Returns:
        uint8 mask, 1 where a lane pixel is likely and 0 elsewhere.
    """
    if image_bgr.ndim != 3:
        raise ValueError("expected a 3 channel BGR image")

    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    b_channel = lab[:, :, 2]

    white = (l_channel >= cfg.hls_l_min) & (l_channel <= cfg.hls_l_max)
    yellow = (b_channel >= cfg.lab_b_min) & (b_channel <= cfg.lab_b_max)
    saturated = (s_channel >= cfg.hls_s_min) & (s_channel <= cfg.hls_s_max)

    sobel = _scaled_sobel_x(gray, cfg.sobel_kernel)
    edges = (sobel >= cfg.sobel_x_min) & (sobel <= cfg.sobel_x_max)

    # Colour is the primary evidence. The gradient only contributes where the
    # saturation channel already suggests paint, which keeps kerbs, tar seams
    # and shadow boundaries out of the mask.
    mask = white | yellow | (saturated & edges)
    return mask.astype(np.uint8)


def perspective_matrices(
    image_shape: tuple[int, int], cfg: PerspectiveConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Build the forward and inverse bird's eye view transforms.

    Args:
        image_shape: (height, width) of the frame.
        cfg: source quad and destination margin.

    Returns:
        (M, M_inv). M warps image to bird's eye view; M_inv warps back.
    """
    height, width = image_shape[:2]

    src = np.float32(
        [
            [cfg.src_top_left[0] * width, cfg.src_top_left[1] * height],
            [cfg.src_top_right[0] * width, cfg.src_top_right[1] * height],
            [cfg.src_bottom_right[0] * width, cfg.src_bottom_right[1] * height],
            [cfg.src_bottom_left[0] * width, cfg.src_bottom_left[1] * height],
        ]
    )

    margin = cfg.dst_margin * width
    dst = np.float32(
        [
            [margin, 0],
            [width - margin, 0],
            [width - margin, height],
            [margin, height],
        ]
    )

    forward = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    return forward, inverse


def warp_to_birdseye(
    image: np.ndarray, matrix: np.ndarray, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Apply a perspective warp.

    Args:
        image: source image or mask.
        matrix: 3x3 transform from ``perspective_matrices``.
        size: (width, height) of the output. Defaults to the input size.
    """
    if size is None:
        size = (image.shape[1], image.shape[0])
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR)


def region_of_interest(mask: np.ndarray, top_fraction: float = 0.55) -> np.ndarray:
    """Zero everything above the horizon.

    Sky and treeline produce bright, high gradient pixels that the sliding
    window search will happily fit a lane to. Cutting them costs nothing.
    """
    out = mask.copy()
    cut = int(mask.shape[0] * top_fraction)
    out[:cut, :] = 0
    return out
