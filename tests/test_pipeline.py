"""End to end pipeline tests, plus config round tripping."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from test_perception import synthetic_road

from lanectl.calibration import CameraCalibration, calibrate_from_chessboards
from lanectl.config import PerspectiveConfig, PipelineConfig
from lanectl.pipeline import LanePipeline


@pytest.fixture
def road() -> np.ndarray:
    return synthetic_road()


class TestConfig:
    def test_defaults_construct(self):
        cfg = PipelineConfig()
        assert cfg.controller.kp > 0
        assert cfg.perspective.lane_width_m == 3.7

    def test_metric_scale_is_derived_consistently(self):
        persp = PerspectiveConfig(lane_width_m=3.7, lane_width_px=546.0)
        assert persp.xm_per_pix == pytest.approx(3.7 / 546.0)
        # 546 px across at that scale must give back 3.7 m.
        assert 546.0 * persp.xm_per_pix == pytest.approx(3.7)

    def test_yaml_round_trip(self, tmp_path):
        original = PipelineConfig()
        original.controller.kp = 0.77
        original.perspective.lane_width_px = 512.0
        path = original.save(tmp_path / "cfg.yaml")

        loaded = PipelineConfig.load(path)
        assert loaded.controller.kp == pytest.approx(0.77)
        assert loaded.perspective.lane_width_px == pytest.approx(512.0)
        assert isinstance(loaded.perspective.src_top_left, tuple)

    def test_partial_yaml_fills_in_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text("controller:\n  kp: 1.25\n")
        cfg = PipelineConfig.load(path)
        assert cfg.controller.kp == pytest.approx(1.25)
        assert cfg.controller.kd == pytest.approx(0.1)  # untouched default
        assert cfg.perspective.lane_width_m == pytest.approx(3.7)

    def test_empty_yaml_gives_all_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert PipelineConfig.load(path).controller.kp == PipelineConfig().controller.kp


class TestCalibration:
    def test_round_trips_through_json(self, tmp_path):
        calib = CameraCalibration(
            camera_matrix=np.array([[1156.0, 0, 671.0], [0, 1151.0, 389.0], [0, 0, 1.0]]),
            dist_coeffs=np.array([[-0.24, -0.02, -0.0007, 0.0001, 0.01]]),
            image_size=(1280, 720),
            rms_reprojection_error=1.0,
            n_views_used=17,
        )
        path = calib.save(tmp_path / "cam.json")
        loaded = CameraCalibration.load(path)

        np.testing.assert_allclose(loaded.camera_matrix, calib.camera_matrix)
        np.testing.assert_allclose(loaded.dist_coeffs, calib.dist_coeffs)
        assert loaded.image_size == (1280, 720)
        assert loaded.fx == pytest.approx(1156.0)
        assert loaded.cx == pytest.approx(671.0)

    def test_undistort_preserves_shape(self, tmp_path, road):
        calib = CameraCalibration(
            camera_matrix=np.array([[1156.0, 0, 640.0], [0, 1151.0, 360.0], [0, 0, 1.0]]),
            dist_coeffs=np.array([[-0.24, -0.02, 0.0, 0.0, 0.01]]),
            image_size=(1280, 720),
            rms_reprojection_error=1.0,
            n_views_used=17,
        )
        assert calib.undistort(road).shape == road.shape

    def test_refuses_to_calibrate_from_too_few_views(self, tmp_path):
        blank = tmp_path / "blank.png"
        cv2.imwrite(str(blank), np.zeros((100, 100, 3), np.uint8))
        with pytest.raises(ValueError, match="at least 3"):
            calibrate_from_chessboards([blank, blank])

    def test_unreadable_paths_are_skipped_not_fatal(self, tmp_path):
        with pytest.raises(ValueError, match="at least 3"):
            calibrate_from_chessboards([tmp_path / "does_not_exist.jpg"])


class TestPipeline:
    def test_processes_a_frame_end_to_end(self, road):
        result = LanePipeline(PipelineConfig()).process(road)

        assert result.overlay.shape == road.shape
        assert result.mask.shape == road.shape[:2]
        assert result.warped_mask.shape == road.shape[:2]
        assert -1.0 <= result.steer_command <= 1.0
        assert result.latency_ms > 0

    def test_draw_false_skips_the_overlay(self, road):
        result = LanePipeline(PipelineConfig()).process(road, draw=False)
        np.testing.assert_array_equal(result.overlay, road)

    def test_survives_an_all_black_frame(self):
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = LanePipeline(PipelineConfig()).process(black)
        assert not result.lane.valid
        assert np.isfinite(result.steer_command)

    def test_repeated_frames_converge_to_a_stable_command(self, road):
        pipeline = LanePipeline(PipelineConfig())
        commands = [pipeline.process(road, draw=False).steer_command for _ in range(30)]
        # A static scene should not produce a wandering command.
        assert np.std(commands[-10:]) < 0.05

    def test_reset_clears_the_prior(self, road):
        pipeline = LanePipeline(PipelineConfig())
        pipeline.process(road, draw=False)
        assert pipeline._prior is not None
        pipeline.reset()
        assert pipeline._prior is None
        assert pipeline._dropped == 0

    def test_falls_back_to_a_window_search_after_too_many_bad_frames(self, road):
        cfg = PipelineConfig()
        pipeline = LanePipeline(cfg)
        pipeline.process(road, draw=False)

        black = np.zeros_like(road)
        for _ in range(cfg.lane_fit.max_dropped_frames + 2):
            pipeline.process(black, draw=False)

        assert pipeline._dropped > cfg.lane_fit.max_dropped_frames


class TestCLIContract:
    """The detection JSON is a published interface, so its shape is a test."""

    def test_detection_record_keys(self, tmp_path, road):
        from lanectl.cli import main

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        cv2.imwrite(str(image_dir / "frame.jpg"), road)
        out_dir = tmp_path / "out"

        assert main(["detect", str(image_dir), "-o", str(out_dir)]) == 0

        records = json.loads((out_dir / "detections.json").read_text())
        assert len(records) == 1
        assert set(records[0]) == {
            "image",
            "valid",
            "lateral_offset_m",
            "heading_error_deg",
            "curvature_radius_m",
            "lane_width_m",
            "steer_command",
            "left_px",
            "right_px",
            "latency_ms",
        }
        assert (out_dir / "frame_overlay.jpg").exists()

    def test_simulate_writes_metrics(self, tmp_path):
        from lanectl.cli import main

        out = tmp_path / "sim.json"
        assert main(["simulate", "-o", str(out), "--duration", "3"]) == 0

        payload = json.loads(out.read_text())
        assert set(payload["results"]) == {"pid", "stanley"}
        assert "rmse_m" in payload["results"]["pid"]["metrics"]
        assert payload["setup"]["gains"]["kp"] > 0

    def test_missing_input_returns_nonzero(self, tmp_path):
        from lanectl.cli import main

        assert main(["detect", str(tmp_path / "nope"), "-o", str(tmp_path / "o")]) == 1
