# Pipeline

What was built, stage by stage, and how data moves through it.

## Overview

```
                    ┌──────────────────── setup, run once ────────────────────┐
                    │                                                          │
   camera_cal/*.jpg │  calibrate_from_chessboards()                            │
   (20 views)  ─────┼─►  findChessboardCorners → cornerSubPix → calibrateCamera│
                    │         │                                                │
                    │         ▼                                                │
                    │  outputs/calibration/camera.json                         │
                    │  K, dist, RMS 1.0029 px                                  │
                    │                                                          │
   straight_lines1  │  scripts/measure_scale.py                                │
              ──────┼─►  lane separation 546.0 px ≡ 3.70 m                     │
                    │    painted dash    59.5 px ≡ 3.05 m                      │
                    │         │                                                │
                    │         ▼                                                │
                    │  configs/highway.yaml                                    │
                    └──────────┼───────────────────────────────────────────────┘
                               │
   ┌───────────────────────────┼─── per frame, 20 Hz ──────────────────────────┐
   │                           ▼                                               │
   │  frame ──► undistort ──► lane_pixel_mask ──► region_of_interest           │
   │  1280×720                    │                       │                    │
   │                              │                       ▼                    │
   │              HLS L ─┐        │              warp_to_birdseye              │
   │              LAB B ─┼─► OR ──┘              (homography H)                │
   │       HLS S & Sobel ┘                               │                     │
   │                                                     ▼                     │
   │                               prior valid?  ──yes──► search_around_prior  │
   │                                     │                       │             │
   │                                     no                      │             │
   │                                     ▼                       │             │
   │                            sliding_window_search ◄──too few─┘             │
   │                                     │                                     │
   │                                     ▼                                     │
   │                             np.polyfit ×2  →  smooth vs prior (α 0.75)    │
   │                                     │                                     │
   │                                     ▼                                     │
   │                        plausibility gate: 2.5 m ≤ width ≤ 5.0 m           │
   │                             │                        │                    │
   │                           pass                     reject                 │
   │                             │                        │                    │
   │                             ▼                        ▼                    │
   │                 offset, heading, radius     carry prior, drops += 1       │
   │                             │                        │                    │
   │                             │              drops > 5 → discard prior      │
   │                             ▼                                             │
   │        ┌────────────────────┴────────────────────┐                        │
   │        ▼                                         ▼                        │
   │  PID / Stanley                          compute_control()                 │
   │  → steer                                → throttle, brake                 │
   │        │                                         │                        │
   │        └──────────────────┬──────────────────────┘                        │
   │                           ▼                                               │
   │              CarlaEgoVehicleControl  /  overlay  /  detections.json        │
   └───────────────────────────────────────────────────────────────────────────┘
```

## Stage by stage

### 1. Data acquisition

`scripts/download_data.sh` fetches the Udacity CarND set into `data/raw/`.
Nothing is committed; the script also writes `data/README.md` with the
attribution and the MIT licence notice.

**Input:** none. **Output:** 20 chessboard views, 8 road stills, optionally a
1260 frame clip.

### 2. Data validation

Validation is built into the stages that consume the data rather than done as a
separate pass, because a separate pass would only duplicate the checks.

- `calibrate_from_chessboards` skips views where the board is not found and
  **raises if fewer than three survive**, three being the mathematical floor
  for a planar calibration. On the sample set 17 of 20 are usable; the other 3
  have the board clipped by the frame edge.
- `lane_pixel_mask` rejects anything that is not a 3 channel image.
- `measure_dash_length_px` rejects runs longer than a quarter of the image,
  which is how it detects that the line on that side is solid rather than
  dashed. Without that guard it silently returns the full image height as the
  dash length, and every along track metre is then wrong by a factor of twelve.
  This is not hypothetical: it is what `straight_lines2.jpg` does.

### 3. Preprocessing

`CameraCalibration.undistort` then `lane_pixel_mask` then
`region_of_interest`. Details and the reasoning are in `docs/RESEARCH.md` §8.

**Input:** BGR frame. **Output:** binary uint8 mask, same height and width.

### 4. Algorithm

No model, so this stage is geometry.

`perspective_matrices` builds the forward and inverse homographies from a
source quad expressed as *fractions* of image width and height, so a change of
resolution does not silently invalidate the config. `warp_to_birdseye` applies
it.

`fit_lane` then does the search, the fit, the smoothing, the gate and the
metric conversion, returning a `LaneFit`.

**Input:** binary mask. **Output:** `LaneFit` with `lateral_offset_m`,
`heading_error_rad`, `curvature_radius_m`, `lane_width_m` and a `valid` flag.

### 5. Training or setup

There is no training. Setup is two commands, each run once per camera:

```bash
lanectl calibrate data/raw/camera_cal -o outputs/calibration/camera.json
python scripts/measure_scale.py data/raw/test_images/straight_lines1.jpg
```

The second prints the two pixel scales to paste into the config. **Both of
these are properties of the camera mounting and the source quad**, so they must
be redone after any change to either. Reusing another camera's numbers is the
single easiest way to get metric output that looks reasonable and is wrong.

### 6. Inference

`LanePipeline.process` runs one frame and holds the previous fit as state. The
prior tracking logic is what turns a per frame detector into something usable
in a loop, and it is described in `docs/RESEARCH.md` §10.

**Input:** BGR frame. **Output:** `FrameResult` — the lane, the steer command,
the overlay, both intermediate masks and the measured latency.

### 7. Post-processing

Three things happen after the fit and before the numbers leave the pipeline:

- **Exponential smoothing** of the polynomial coefficients against the prior,
  applied *before* anything is derived, so the offset and curvature inherit it.
- **The plausibility gate**, which is the pipeline's only self assessment. A
  lane cannot be 1.2 m or 9 m wide; when the measurement says it is, the fit is
  wrong in a way no downstream stage can detect.
- **Graceful degradation**: a rejected frame returns the prior's geometry
  marked `valid=False`, so a controller can coast rather than lurch, and the
  caller can still see that something went wrong.

### 8. Evaluation

- `lanectl detect` — per frame metric measurements to JSON, plus overlays
- `lanectl simulate` — closed loop on the bicycle model, metrics to JSON
- `lanectl bench` — latency distribution at a stated resolution

### 9. Visualisation

`draw_lane_overlay` projects the fitted lane back through the inverse
homography onto the camera image and draws a panel with the live numbers.
`scripts/plot_results.py` turns the JSON into the figures in `docs/RESULTS.md`.

### 10. Results

`docs/RESULTS.md`, with every number labelled `LOCAL RESULT` or
`REFERENCE RESULT`.

## The simulator paths

Two integration surfaces are published but **were not executed** when producing
`docs/RESULTS.md`, because the environment has neither ROS 2 nor a CARLA
server. They are marked as such in their own module docstrings as well as here.

### ROS 2 node

`src/lanectl/ros2_node.py`. The ROS specific surface is deliberately thin:
message conversion and topic wiring only. The one piece of decision logic,
`compute_control`, is a plain function with no ROS import, and it is unit
tested — seven tests in `tests/test_control.py::TestLongitudinalControl` cover
the throttle/brake split, the curvature speed cap against
`v_max = sqrt(a_lat R)`, and range safety.

```
subscribes  /carla/<role>/rgb_front/image        sensor_msgs/Image
            /carla/<role>/odometry               nav_msgs/Odometry
publishes   /carla/<role>/vehicle_control_cmd    carla_msgs/CarlaEgoVehicleControl
            /lanectl/lane_state                  std_msgs/Float32MultiArray
            /lanectl/overlay                     sensor_msgs/Image
```

Image topics use `BEST_EFFORT` QoS with depth 1. A reliable queue would grow
under load and the controller would end up acting on old frames, which is worse
than dropping them.

To run it, on a machine with ROS 2 Humble or newer and the CARLA ROS bridge:

```bash
source /opt/ros/humble/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
ros2 run lanectl lane_control_node --ros-args \
    -p config:=configs/highway.yaml -p role_name:=ego_vehicle
```

### Direct CARLA client

`scripts/carla_run.py` drives the ego vehicle without ROS in the loop. It runs
the simulator in synchronous mode with a fixed step, because a variable `dt`
makes the run irreproducible and the controller's derivative term meaningless.

Its real value is the ground truth: it logs CARLA's own map derived lane centre
alongside the measured offset, which is the one comparison the offline
evaluation here cannot make.

```bash
./CarlaUE4.sh -quality-level=Epic -carla-server     # terminal 1
python scripts/carla_run.py --config configs/highway.yaml \
    --town Town04 --duration 120 --output outputs/carla
```

Town04 is the default deliberately. It has a highway loop with clean markings,
which is what these thresholds are tuned for. Running the urban maps without
retuning produces a stream of rejected frames, which is a fair result for the
thresholds and an unfair one for the pipeline.

## Configuration

Everything tunable lives in `configs/highway.yaml`, in five blocks:
`thresholds`, `perspective`, `lane_fit`, `controller`, `vehicle`. A run is
reproducible from that file plus a commit hash. Partial configs are legal;
anything absent falls back to the dataclass default.
