# Data

Nothing in this directory is committed. Run `bash scripts/download_data.sh` to
recreate it.

## Source

`camera_cal/` and `test_images/` come from Udacity's CarND Advanced Lane Finding
project:

    https://github.com/udacity/CarND-Advanced-Lane-Lines

That repository is MIT licensed, Copyright (c) 2016-2018 Udacity, Inc. The
images are used here unmodified as pipeline input; no Udacity code is included
in this repository.

## Contents

| Path | What it is | Used for |
| --- | --- | --- |
| `camera_cal/calibration*.jpg` | 20 views of a 9x6 chessboard, 1280x720 | camera intrinsics and distortion |
| `test_images/straight_lines*.jpg` | straight highway, 1280x720 | measuring the pixel to metre scale |
| `test_images/test*.jpg` | curved highway, shadow, changing tarmac | perception evaluation |
| `project_video.mp4` | 50 s highway clip, 1260 frames, 25 fps | temporal evaluation, `--video` only |

## Why this dataset

It is the smallest public set that carries both the chessboard views and road
frames from the *same* camera. Without that pairing the calibration cannot be
applied to the road frames, and every metric number downstream would be wrong
by an unknown factor.
