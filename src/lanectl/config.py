"""Configuration objects, loaded from YAML.

Every tunable number in the pipeline lives here rather than in the algorithm
modules, so a run can be reproduced from a config file plus a commit hash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ThresholdConfig:
    """Colour and gradient thresholds used to isolate lane pixels."""

    #: Saturation channel of HLS. White and yellow paint stay saturated where
    #: worn tarmac does not.
    hls_s_min: int = 170
    hls_s_max: int = 255
    #: Lightness channel of HLS, which is what actually catches white paint.
    hls_l_min: int = 200
    hls_l_max: int = 255
    #: B channel of LAB, the most reliable single channel for yellow paint.
    lab_b_min: int = 155
    lab_b_max: int = 200
    #: Sobel gradient in x, scaled to 0..255. Lane edges are near vertical in
    #: the image, so the x gradient is the informative one.
    sobel_x_min: int = 20
    sobel_x_max: int = 100
    sobel_kernel: int = 3


@dataclass
class PerspectiveConfig:
    """Bird's eye view transform and the scale that converts it to metres.

    The source quad is expressed as fractions of image width and height so the
    same config survives a change of camera resolution.
    """

    #: (x, y) as fractions of (width, height): top left, top right,
    #: bottom right, bottom left of a road patch that is rectangular in reality.
    src_top_left: tuple[float, float] = (0.445, 0.640)
    src_top_right: tuple[float, float] = (0.558, 0.640)
    src_bottom_right: tuple[float, float] = (0.885, 0.960)
    src_bottom_left: tuple[float, float] = (0.164, 0.960)

    #: Horizontal inset of the warped lane in the output image, as a fraction.
    dst_margin: float = 0.25

    #: Scale of the warped image. Defaults follow the US highway standard the
    #: sample footage was shot on: a 3.7 m lane and a 3.05 m painted dash. Note
    #: this is the length of the painted segment, not the dash-plus-gap period,
    #: because the painted segment is what a column profile can measure.
    lane_width_m: float = 3.7
    dash_length_m: float = 3.05
    #: Warped pixels spanned by one lane width and one painted dash. These are
    #: a property of the source quad and the camera mounting, so they must be
    #: measured rather than assumed; see scripts/measure_scale.py.
    lane_width_px: float = 546.0
    dash_length_px: float = 59.5

    @property
    def xm_per_pix(self) -> float:
        """Metres per pixel across the warped image."""
        return self.lane_width_m / self.lane_width_px

    @property
    def ym_per_pix(self) -> float:
        """Metres per pixel along the warped image."""
        return self.dash_length_m / self.dash_length_px


@dataclass
class LaneFitConfig:
    """Sliding window search and polynomial fitting."""

    n_windows: int = 9
    window_margin: int = 100
    min_pixels_to_recentre: int = 50
    #: Half width of the band searched around the previous frame's fit.
    search_margin: int = 80
    #: Exponential smoothing factor on the fitted coefficients. 0 disables
    #: smoothing, 1 freezes the fit.
    smoothing: float = 0.75
    #: A fit is rejected if the two lane lines are further apart than this
    #: many metres, or closer than the lower bound.
    min_lane_width_m: float = 2.5
    max_lane_width_m: float = 5.0
    #: Consecutive rejected frames before the search falls back to sliding
    #: windows rather than searching around a stale prior.
    max_dropped_frames: int = 5


@dataclass
class ControllerConfig:
    """Lateral controller gains and vehicle limits."""

    #: PID on the cross track error. The defaults follow the CARLA ROS 2
    #: reference controller documented in docs/RESEARCH.md.
    kp: float = 0.5
    ki: float = 0.01
    kd: float = 0.1
    #: Stanley cross track gain, used by the alternative controller.
    stanley_k: float = 0.5
    stanley_softening: float = 1.0
    #: Control period in seconds. 20 Hz matches the reference bridge.
    dt: float = 0.05
    #: Steering command limits, normalised as CARLA expects.
    steer_limit: float = 1.0
    #: Physical steering angle at a normalised command of 1.0, in radians.
    max_steer_angle: float = 0.5236  # 30 degrees
    #: Anti windup clamp on the integral term.
    integral_limit: float = 1.0


@dataclass
class VehicleConfig:
    """Kinematic bicycle model parameters for the closed loop simulation."""

    wheelbase_m: float = 2.875  # CARLA Tesla Model 3 default
    max_speed_ms: float = 30.0


@dataclass
class PipelineConfig:
    """Everything needed to reproduce a run."""

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    lane_fit: LaneFitConfig = field(default_factory=LaneFitConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    calibration_path: str = "outputs/calibration/camera.json"

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Read a config from YAML, filling anything absent with the default."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            thresholds=ThresholdConfig(**raw.get("thresholds", {})),
            perspective=PerspectiveConfig(**_tuplify(raw.get("perspective", {}))),
            lane_fit=LaneFitConfig(**raw.get("lane_fit", {})),
            controller=ControllerConfig(**raw.get("controller", {})),
            vehicle=VehicleConfig(**raw.get("vehicle", {})),
            calibration_path=raw.get("calibration_path", cls.calibration_path),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        return path


def _tuplify(section: dict) -> dict:
    """YAML gives lists where the dataclass wants tuples of floats."""
    out = dict(section)
    for key in ("src_top_left", "src_top_right", "src_bottom_right", "src_bottom_left"):
        if key in out and isinstance(out[key], list):
            out[key] = tuple(float(v) for v in out[key])
    return out
