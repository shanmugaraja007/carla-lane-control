"""Turn the JSON that ``lanectl`` writes into the figures used in docs/RESULTS.md.

    python scripts/plot_results.py --control outputs/control/step_response.json \
                                   --detections outputs/detect/detections.json \
                                   --output outputs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#16202a"
GRID = "#d7dee5"
SERIES = ["#1f6feb", "#d1741f", "#2a9d5c", "#8b5cf6"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def plot_control(path: Path, out_dir: Path) -> list[Path]:
    payload = json.loads(path.read_text())
    setup = payload["setup"]
    dt = setup["dt_s"]
    written = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    for i, (name, block) in enumerate(payload["results"].items()):
        error = np.array(block["error"])
        steer = np.array(block["steer"])
        t = np.arange(len(error)) * dt
        colour = SERIES[i % len(SERIES)]
        m = block["metrics"]
        label = f"{name}  (rise {m['rise_time_s']:.2f}s, OS {m['overshoot_pct']:.0f}%)"
        axes[0].plot(t, error, color=colour, linewidth=1.8, label=label)
        axes[1].plot(t, steer, color=colour, linewidth=1.8, label=name)

    axes[0].axhline(0, color=INK, linewidth=0.8, linestyle="--", alpha=0.5)
    _style(
        axes[0],
        f"Cross track error, {setup['initial_offset_m']} m step at {setup['speed_ms']} m/s",
        "time (s)",
        "error (m)",
    )
    axes[0].legend(fontsize=8, frameon=False)

    _style(axes[1], "Steering command", "time (s)", "normalised steer")
    axes[1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    dest = out_dir / "step_response.png"
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    written.append(dest)

    # Path traces, which show the same thing geometrically.
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150)
    for i, (name, block) in enumerate(payload["results"].items()):
        ax.plot(block["x"], block["y"], color=SERIES[i % len(SERIES)], linewidth=1.8, label=name)
    ax.axhline(0, color=INK, linewidth=0.9, linestyle="--", alpha=0.6, label="lane centre")
    _style(ax, "Vehicle path, kinematic bicycle model", "x (m)", "y (m)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    dest = out_dir / "vehicle_path.png"
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    written.append(dest)
    return written


def plot_detections(path: Path, out_dir: Path) -> list[Path]:
    records = json.loads(path.read_text())
    if not records:
        return []

    key = "image" if "image" in records[0] else "frame"
    labels = [str(r[key]) for r in records]
    width = np.array([r["lane_width_m"] for r in records]) if "lane_width_m" in records[0] else None
    offset = np.array([r["lateral_offset_m"] for r in records])
    written = []

    n_panels = 2 if width is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.0 * n_panels), dpi=150, squeeze=False)
    axes = axes.ravel()

    x = np.arange(len(records))
    axes[0].bar(x, offset, color=SERIES[0], width=0.6)
    axes[0].axhline(0, color=INK, linewidth=0.8)
    _style(axes[0], "Lateral offset from lane centre", "", "offset (m)")
    if len(labels) <= 20:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    if width is not None:
        axes[1].bar(x, width, color=SERIES[2], width=0.6)
        axes[1].axhline(3.7, color=SERIES[1], linewidth=1.4, linestyle="--",
                        label="3.7 m nominal")
        _style(axes[1], "Measured lane width, a check on the metric scale", "", "width (m)")
        axes[1].set_ylim(0, max(5.0, width.max() * 1.15))
        axes[1].legend(fontsize=8, frameon=False)
        if len(labels) <= 20:
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)

    fig.tight_layout()
    dest = out_dir / "detection_summary.png"
    fig.savefig(dest, facecolor="white")
    plt.close(fig)
    written.append(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if args.control and args.control.exists():
        written += plot_control(args.control, args.output)
    if args.detections and args.detections.exists():
        written += plot_detections(args.detections, args.output)

    if not written:
        print("nothing to plot; pass --control and/or --detections")
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
