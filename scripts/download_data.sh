#!/usr/bin/env bash
# Fetch the sample data used by docs/RESULTS.md.
#
# Nothing here is committed to the repository. The images come from Udacity's
# CarND Advanced Lane Finding project, which is MIT licensed; the licence and
# attribution are reproduced in data/README.md, which this script writes.
#
#   bash scripts/download_data.sh          # calibration + test images, ~4 MB
#   bash scripts/download_data.sh --video  # also the 25 MB project video

set -euo pipefail

BASE="https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw"

mkdir -p "$DEST/camera_cal" "$DEST/test_images"

echo "fetching chessboard calibration views..."
for i in $(seq 1 20); do
  curl -fsSL -o "$DEST/camera_cal/calibration$i.jpg" \
    "$BASE/camera_cal/calibration$i.jpg" || echo "  calibration$i.jpg unavailable, skipping"
done

echo "fetching road test images..."
for name in straight_lines1 straight_lines2 test1 test2 test3 test4 test5 test6; do
  curl -fsSL -o "$DEST/test_images/$name.jpg" \
    "$BASE/test_images/$name.jpg" || echo "  $name.jpg unavailable, skipping"
done

if [[ "${1:-}" == "--video" ]]; then
  echo "fetching project_video.mp4 (25 MB)..."
  curl -fL --progress-bar -o "$DEST/project_video.mp4" "$BASE/project_video.mp4"
fi

cat > "$ROOT/data/README.md" <<'EOF'
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
EOF

echo
echo "done. calibration views: $(ls -1 "$DEST/camera_cal" 2>/dev/null | wc -l)"
echo "      test images:       $(ls -1 "$DEST/test_images" 2>/dev/null | wc -l)"
echo "      attribution written to data/README.md"
