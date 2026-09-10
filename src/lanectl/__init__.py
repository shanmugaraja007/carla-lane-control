"""Vision based lane perception and lateral control for CARLA and ROS 2."""

from .calibration import CameraCalibration, calibrate_from_chessboards
from .config import (
    ControllerConfig,
    LaneFitConfig,
    PerspectiveConfig,
    PipelineConfig,
    ThresholdConfig,
    VehicleConfig,
)
from .control import (
    BicycleState,
    KinematicBicycle,
    PIDLateralController,
    StanleyLateralController,
    step_response_metrics,
    tracking_metrics,
)
from .lane import LaneFit, fit_lane
from .perception import lane_pixel_mask, perspective_matrices, warp_to_birdseye
from .pipeline import FrameResult, LanePipeline

__version__ = "0.1.0"

__all__ = [
    "CameraCalibration",
    "calibrate_from_chessboards",
    "PipelineConfig",
    "ThresholdConfig",
    "PerspectiveConfig",
    "LaneFitConfig",
    "ControllerConfig",
    "VehicleConfig",
    "PIDLateralController",
    "StanleyLateralController",
    "KinematicBicycle",
    "BicycleState",
    "step_response_metrics",
    "tracking_metrics",
    "LaneFit",
    "fit_lane",
    "lane_pixel_mask",
    "perspective_matrices",
    "warp_to_birdseye",
    "LanePipeline",
    "FrameResult",
]
