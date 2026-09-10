# Research

Background reading and design decisions behind this repository. Everything
here was settled before the code was written; `docs/PIPELINE.md` describes what
was built and `docs/RESULTS.md` records what it actually did.

## 1. Problem definition

Given a single forward facing camera on a road vehicle, recover enough lane
geometry to keep the vehicle centred in its lane:

- **lateral offset** `e`, the signed distance in metres from the vehicle to the
  lane centre
- **heading error** `psi`, the angle between the vehicle's forward axis and the
  lane direction
- **curvature** `1/R` of the upcoming lane

and turn those into a steering command at a fixed control rate, closing the
loop.

Three constraints shape every choice below:

1. **The control loop runs at 20 Hz.** Perception that takes longer than 50 ms
   per frame is not a slower system, it is a broken one, because the controller
   then acts on stale measurements.
2. **The output must be metric.** A controller consuming pixels cannot have
   gains that mean anything, and cannot be transferred between cameras.
3. **Failure must be detectable.** A lane detector that returns a confident
   wrong answer is more dangerous than one that returns nothing, because the
   controller has no way to tell the difference.

## 2. Background

### The geometry

A camera on a vehicle sees the road plane in perspective: parallel lane lines
converge, and equal distances on the road map to unequal distances in the
image. Inverse perspective mapping (IPM) undoes this by assuming the road is
locally planar and applying the homography that maps the road plane to a top
down view. In that view lane lines are parallel again, and a second order
polynomial in the along track direction is a good model of a road lane, because
highway geometry is specified in clothoid segments whose curvature varies
slowly.

The planarity assumption is the whole game. It holds well on a highway and
poorly on a crest, a dip or a banked curve, and every metric number this
repository produces inherits that limitation.

### The control problem

Lane keeping is path following for a nonholonomic vehicle. The standard model
is the kinematic bicycle: two wheels, a wheelbase `L`, a steering angle
`delta`, and the yaw rate `v/L * tan(delta)`. It is exact when tyre slip is
negligible, which is true below roughly 0.4 g of lateral acceleration and is
the regime a lane keeping controller should never leave.

## 3. Existing approaches

### Lane perception

| Family | Representative | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Colour and gradient thresholding + IPM + polynomial fit | the classical CarND pipeline | no training, no GPU, fully interpretable, trivially real time on a CPU | thresholds are tuned per domain; fails on worn paint, heavy shadow, snow |
| Semantic segmentation | UNet, DeepLab on a lane class | robust to appearance change, learns paint that thresholds miss | needs labelled data and a GPU; per pixel output still needs the same IPM and fitting stages afterwards |
| Row anchor classification | Ultra Fast Lane Detection | very fast, formulates lanes as per row selection rather than per pixel | fixed anchor grid limits geometric precision |
| Line anchor refinement | CLRNet, CLRerNet | current benchmark leaders | heaviest of the four, and the output is still image space lines |

Reported CULane F1 for the line anchor family, from the CLRerNet repository
(DLA34 backbone, 18.4 GFLOPs), is **80.47 for CLRNet** and **81.12 ± 0.04 for
CLRerNet**, rising to **81.43 ± 0.14** with EMA. These are `REFERENCE RESULT`
figures from the authors, quoted here to size the gap, and are not comparable
to anything measured in this repository: they are image space F1 on CULane,
whereas what is measured here is metric lane geometry on highway footage.

The important observation is that **all four families feed the same downstream
stages**. Whatever produces the lane pixels, something still has to warp them,
fit them, convert to metres and hand a number to a controller. Those stages are
where the metric correctness lives, and they are what this repository is about.

### Lateral control

| Controller | Idea | Behaviour |
| --- | --- | --- |
| PID on cross track error | `delta = Kp e + Ki integral(e) + Kd de/dt` | simple, no model needed, but no anticipation: it can only react to error that has already appeared |
| Stanley | `delta = psi + arctan(k e / (v + ks))` | uses heading error directly, so it is much better damped; has a steady state offset on a constant radius curve without a feedforward term |
| Pure pursuit | steer to a lookahead point on the path | naturally smooth, needs a path rather than a lane measurement |
| MPC | optimise over a horizon subject to constraints | anticipates curves and respects limits; needs a model, a solver and far more compute |

The CARLA ROS 2 reference stack uses a PID for both axes, with lateral gains
`Kp 0.5, Ki 0.01, Kd 0.1` and longitudinal `Kp 0.2, Ki 0.01, Kd 0.3` at
`dt = 0.05 s`, publishing `CarlaEgoVehicleControl` with `throttle`, `steer` and
`brake`. Those gains are adopted here as the starting point precisely so the
results are comparable to a known reference rather than to a private tuning.

## 4. Selected approach

**Perception:** classical thresholding into IPM into a sliding window search
into a second order polynomial fit, with the metric scale measured rather than
assumed.

**Control:** PID on the cross track error as the primary controller, with
Stanley implemented alongside it for comparison, and a longitudinal term whose
target speed is capped by the curvature ahead.

**Validation:** offline on real road footage for the perception, and closed
loop on a kinematic bicycle model for the controller.

## 5. Why this approach was selected

1. **It runs at rate on a CPU.** Measured at 20.3 FPS on 720p on two vCPUs,
   which clears the 20 Hz loop with nothing to spare and no GPU at all. A
   segmentation network would need a GPU to make the same claim.
2. **Every failure is inspectable.** When a frame is rejected you can look at
   the mask, the warp and the histogram and see which stage gave up. That is
   worth more in a control loop than several points of F1.
3. **The interesting engineering is downstream anyway.** The pixel to metre
   scale, the plausibility gate, the fallback when a fit is lost: none of that
   changes if the mask comes from a network instead, and all of it is where
   metric correctness actually gets decided.
4. **It is honest about what could be measured here.** With no GPU and no
   simulator, a learned detector could have been written but not trained, and a
   CARLA loop could have been written but not run. The classical pipeline could
   be built *and executed end to end*, which the brief asks for.

The clear cost: this will not survive snow, heavy rain, worn paint or a
construction zone. Section 13 says what to do about that.

## 6. Architecture

```
frame
  -> undistort                 (camera intrinsics, k1..k3, p1, p2)
  -> lane pixel mask           (HLS L, LAB B, HLS S & Sobel x)
  -> region of interest        (drop everything above the horizon)
  -> perspective warp          (homography to bird's eye view)
  -> sliding window search     (or a band around the previous fit)
  -> polynomial fit            x = a y^2 + b y + c, twice
  -> plausibility gate         (2.5 m <= lane width <= 5.0 m)
  -> metric conversion         (xm_per_pix, ym_per_pix)
  -> offset, heading, radius
  -> PID / Stanley             -> normalised steer
  -> longitudinal P + curve cap -> throttle, brake
```

## 7. Dataset

**Udacity CarND Advanced Lane Finding**, MIT licensed:
20 chessboard views and 8 road stills at 1280x720, plus a 1260 frame highway
clip at 25 fps, all from the same camera.

That last point is why this set was chosen over CULane or TuSimple, both of
which are far larger and more varied. Neither ships chessboard views of the
capture camera, so neither can be undistorted, so no metric claim made on them
would be trustworthy. A small correctly calibrated set beats a large
uncalibrated one when the output is in metres.

## 8. Preprocessing

1. **Distortion correction.** Fitted from the chessboard views by the standard
   planar method. Skipping it bends straight lane lines near the frame edge,
   which the polynomial fit then faithfully reports as curvature.
2. **Colour space conversion.** BGR to HLS and to LAB. The L channel isolates
   white paint, the LAB B channel isolates yellow far better than any RGB
   combination, and HLS S survives shadow better than L.
3. **Region of interest.** Everything above 55 percent of the frame height is
   zeroed. Sky and treeline are bright and high gradient, and the window search
   will happily fit a lane to them.

## 9. Training

**There is none.** No component of this pipeline is learned; the only fitting
that happens is the camera calibration, which is a least squares solve over the
chessboard corner detections, and the per frame polynomial fit.

This is a deliberate property, not a gap. It means there is no train/test split
to leak across, no checkpoint to distribute, and the only thing that has to be
reproduced to reproduce a result is a config file.

If the mask stage were swapped for a segmentation network, the training
procedure would be: a UNet or DeepLabv3+ on the CULane lane class, cross entropy
plus Dice, and the pipeline downstream of `lane_pixel_mask` would not change at
all. That is documented as future work rather than claimed as done.

## 10. Inference

Per frame, holding the previous fit as state:

1. undistort with the cached calibration
2. threshold to a binary mask, cut the region of interest
3. warp with the cached homography
4. if the previous fit was valid, search a ±80 px band around it; otherwise run
   the 9 window search from the histogram peaks
5. fit both lines, exponentially smooth the coefficients against the prior
   (alpha 0.75)
6. gate on lane width; on rejection, carry the prior forward and count a drop
7. after 5 consecutive drops, discard the prior and go back to the full search
8. convert to metres, derive offset, heading and radius
9. step the controller

Step 7 is the part that matters. Without it a bad prior is self reinforcing:
the band search keeps finding pixels near the wrong place and the fit never
recovers.

## 11. Evaluation

Three things get measured, because they fail independently.

**Perception, on real footage.** Acceptance rate over the clip; measured lane
width against the 3.7 m standard, which is the only ground truth available
without instrumented data; and frame to frame offset change, which detects
jitter that an aggregate would hide.

**Controller, in closed loop.** Rise time, overshoot, settling time, steady
state error and RMSE for a lateral step; RMSE and steady state error for
constant radius curve following. Both against an analytic reference path, so
the controller is scored without perception noise, then re run with injected
noise to see the degradation.

**Throughput.** Mean, p50 and p95 latency at 720p, with and without the
overlay, on stated hardware.

Not measured: image space F1 against lane annotations, since the sample set has
none; and lane offset against instrumented ground truth, which needs either RTK
or a simulator. `scripts/carla_run.py` logs CARLA's own map derived lane centre
for exactly that comparison, for anyone who has a server.

## 12. Expected outputs

- `outputs/calibration/camera.json` — intrinsics and distortion
- `outputs/detect/*_overlay.jpg` — lane projected back onto the road, annotated
- `outputs/detect/detections.json` — per frame metric measurements
- `outputs/video/overlay.mp4` — the same over the full clip
- `outputs/control/*.json` — closed loop traces and metrics
- `outputs/figures/*.png` — the figures in `docs/RESULTS.md`

## 13. Alternatives considered

**A segmentation network for the mask stage.** Rejected for this iteration: no
GPU in the environment, so it could have been written but never trained, and an
untrained network is not a result. It is the single highest value upgrade and
the interface (`lane_pixel_mask`) is deliberately a drop in point.

**Model predictive control.** Rejected as premature. MPC's advantage is
anticipation and constraint handling, and on the highway footage here the PID
already holds 0.049 m RMSE on a 200 m radius curve. MPC becomes worth its
complexity at tighter radii and higher speeds.

**Pure pursuit.** Rejected because it wants a path, and what the perception
produces is a lane measurement at the vehicle. Converting one to the other adds
a stage whose errors are hard to attribute.

**Hough transform line detection.** Rejected: it fits straight lines, and
fitting straight lines to a curved lane produces a curvature estimate of
infinity exactly when curvature matters most.

**Assuming the standard 30/720 m per pixel scale** that circulates with this
dataset. Rejected, and this one is worth stating plainly: that constant is only
correct for one specific source quad. With a different quad it is simply wrong,
and it is wrong silently, because every downstream number stays plausible.
`scripts/measure_scale.py` measures it instead, and the two straight line
frames cross check to within 2 percent (546.0 px and 558.5 px per 3.7 m).

## 14. References

**Papers**

- Thrun, S. et al. (2006). *Stanley: The Robot that Won the DARPA Grand
  Challenge.* Journal of Field Robotics 23(9). The Stanley controller.
- Dosovitskiy, A. et al. (2017). *CARLA: An Open Urban Driving Simulator.*
  CoRL. arXiv:1711.03938.
- Zheng, T. et al. (2022). *CLRNet: Cross Layer Refinement Network for Lane
  Detection.* CVPR. arXiv:2203.10350.
- Honda, H. and Uchida, Y. (2023). *CLRerNet: Improving Confidence of Lane
  Detection with LaneIoU.* arXiv:2305.08366.
- Qin, Z. et al. (2020). *Ultra Fast Structure aware Deep Lane Detection.*
  ECCV. arXiv:2004.11757.
- Bertozzi, M. and Broggi, A. (1998). *GOLD: A Parallel Real Time Stereo Vision
  System for Generic Obstacle and Lane Detection.* IEEE TIP 7(1). The origin of
  inverse perspective mapping for lane finding.

**Documentation and code**

- CARLA ROS bridge, `carla_ackermann_control`:
  https://github.com/carla-simulator/ros-bridge
- Building an Autonomous Vehicle in CARLA: PID Controller and ROS 2, LearnOpenCV.
  Source of the reference gains quoted in section 3:
  https://learnopencv.com/pid-controller-ros-2-carla/
- CLRerNet reference implementation and CULane numbers:
  https://github.com/hirotomusiker/CLRerNet
- OpenCV camera calibration:
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Udacity CarND Advanced Lane Finding, the data source (MIT):
  https://github.com/udacity/CarND-Advanced-Lane-Lines

**Standards**

- FHWA MUTCD, Part 3: 10 ft painted line with a 30 ft gap, a 3:1 ratio. The
  3.05 m painted segment used as the along track scale reference.
- AASHTO Green Book: 12 ft (3.7 m) Interstate lane width, used as the across
  track scale reference.
