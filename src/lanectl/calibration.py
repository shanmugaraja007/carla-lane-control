"""Camera intrinsic calibration and distortion correction.

Everything downstream of this module works in metres, so it is worth being
strict here: an uncalibrated camera produces a lane width that changes with
distance, and the lateral error the controller consumes is then wrong in a way
that looks like a controller problem rather than a calibration problem.

The calibration itself is the standard planar chessboard method: detect the
inner corners of a known grid in a set of views, then solve for the intrinsic
matrix and the distortion coefficients that best explain the observed corner
positions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """Intrinsics and distortion for one camera."""

    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # 1x5
    image_size: tuple[int, int]  # (width, height)
    rms_reprojection_error: float
    n_views_used: int

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image taken with this camera."""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["camera_matrix"] = self.camera_matrix.tolist()
        payload["dist_coeffs"] = self.dist_coeffs.ravel().tolist()
        payload["image_size"] = list(self.image_size)
        path.write_text(json.dumps(payload, indent=2))
        log.info("wrote calibration to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibration:
        payload = json.loads(Path(path).read_text())
        return cls(
            camera_matrix=np.array(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(payload["dist_coeffs"], dtype=np.float64).reshape(1, -1),
            image_size=tuple(payload["image_size"]),
            rms_reprojection_error=float(payload["rms_reprojection_error"]),
            n_views_used=int(payload["n_views_used"]),
        )


def calibrate_from_chessboards(
    image_paths: list[str | Path],
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> CameraCalibration:
    """Calibrate a camera from chessboard views.

    Args:
        image_paths: views of the same physical chessboard from different angles.
        pattern_size: number of *inner* corners, (columns, rows).
        square_size: physical edge length of one square. Leaving this at 1.0
            gives intrinsics in pixels, which is all the rest of the pipeline
            needs; set it if you also want the extrinsics in real units.

    Returns:
        The fitted calibration.

    Raises:
        ValueError: if fewer than three views yield a usable corner detection.
            Three is the mathematical floor for a well posed planar calibration.
    """
    # One row per inner corner, laid out on the z = 0 plane.
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            log.warning("could not read %s, skipping", path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            # Not a failure worth stopping for. Some views in any calibration
            # set have the board clipped by the frame edge.
            log.debug("no %sx%s chessboard in %s", *pattern_size, path)
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(refined)

    if len(object_points) < 3:
        raise ValueError(
            f"only {len(object_points)} usable chessboard views out of "
            f"{len(image_paths)}; need at least 3 for a stable calibration"
        )

    assert image_size is not None
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    log.info(
        "calibrated on %d/%d views, RMS reprojection error %.4f px",
        len(object_points),
        len(image_paths),
        rms,
    )

    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_reprojection_error=float(rms),
        n_views_used=len(object_points),
    )


def find_chessboard_corners(
    image: np.ndarray, pattern_size: tuple[int, int] = (9, 6)
) -> np.ndarray | None:
    """Return refined inner corner positions, or None if the board is not found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
