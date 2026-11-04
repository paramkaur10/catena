"""Told against blind in the pilot: the scores per validation camera, and how they grew.

    python scripts/pilot_report.py [RUNS]          (default /root/catena_data/runs/pilot)

Reads RUNS/seed*/metrics.jsonl, written by catena.train, and prints:
  1. for every validation camera and for their mean, at the last validation: told and blind
     (each averaged over the seeds), told minus blind, and told minus blind seed by seed;
     "all +" or "all -" marks a difference with the same sign in every seed;
  2. at every validation: the mean F1 and the cloud balanced accuracy of both arms, for the
     mean of the cameras, averaged over the seeds.
It also draws RUNS/pilot_curves.png: the mean F1 of each camera against the training step,
told and blind, each seed thin and their mean thick, and told minus blind for the mean.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARMS = ("told", "blind")
METRICS = ("f1_clear", "f1_thin", "f1_thick", "f1_shadow", "f1_mean", "cloud_ba")
LABELS = ("F1 clear", "F1 thin", "F1 thick", "F1 shadow", "F1 mean", "cloud BA")
COLOURS = {"told": "#2a78d6", "blind": "#eb6834"}  # blue and orange
STYLES = {"told": "-", "blind": "--"}  # the line style tells the arms apart without colour
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def read_runs(runs: Path) -> dict[str, dict]:
    """{seed folder: {(step, arm, camera): scores}}. A later line for the same key wins, so a
    run that was restarted into the same folder counts once."""
    out = {}
    for path in sorted(runs.glob("seed*/metrics.jsonl")):
        found = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "val":
                found[(row["step"], row["arm"], row["camera"])] = row
        if found:
            out[path.parent.name] = found
    return out


def describe_cameras() -> dict[str, str]:
    """One line per validation camera, from catena.train."""
    try:
        from catena.train import VAL_CAMERAS
    except ImportError:
        return {}
    out = {}
    for name, q, bands, chain in VAL_CAMERAS:
        text = f"{10 * q} m, {len(bands)} bands, "
        if chain is None:
            text += "Sentinel-2 as delivered"
        else:
            text += f"MTF {chain['mtf']}, SNR {chain['snr']}, {chain['bits']} bits, "
            text += f"top {chain['top']}"
        out[name] = text
    return out


def values(data: dict, seeds: list[str], step: int, arm: str, camera: str, metric: str):
    """One score of one arm and camera at one step, for every seed."""
    return np.array([data[s][(step, arm, camera)][metric] for s in seeds])


def draw(path: Path, data: dict, seeds: list[str], steps: list[int], cameras: list[str]):
    x = np.array(steps) / 1000
    about = describe_cameras()
    columns = 4
    rows = math.ceil((len(cameras) + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.8 * rows), squeeze=False)
    fig.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax, camera in zip(axes.flat, cameras, strict=False):
        for arm in ARMS:
            y = np.array([[data[s][(t, arm, camera)]["f1_mean"] for t in steps] for s in seeds])
            for row in y:
                ax.plot(x, row, STYLES[arm], color=COLOURS[arm], linewidth=1, alpha=0.35)
            ax.plot(x, y.mean(axis=0), STYLES[arm], color=COLOURS[arm], linewidth=2)
        title = "mean of the cameras" if camera == "mean" else camera
        if camera in about:
            title += "\n" + about[camera]
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylabel("F1, mean of 4 classes", fontsize=8, color=MUTED)
    ax = axes.flat[len(cameras)]
    diff = np.array(
        [
            [
                data[s][(t, "told", "mean")]["f1_mean"] - data[s][(t, "blind", "mean")]["f1_mean"]
                for t in steps
            ]
            for s in seeds
        ]
    )
    ax.axhline(0, color=MUTED, linewidth=1)
    for row in diff:
        ax.plot(x, row, color=INK, linewidth=1, alpha=0.35)
    ax.plot(x, diff.mean(axis=0), color=INK, linewidth=2)
    ax.set_title(
        "told minus blind, mean of the cameras\nthin: each seed, thick: mean", fontsize=9, color=INK
    )
    ax.set_ylabel("difference in F1", fontsize=8, color=MUTED)
    for ax in axes.flat[len(cameras) + 1 :]:
        ax.set_visible(False)
    for ax in axes.flat:
        ax.set_xlabel("training step (thousands)", fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=COLOURS[a], linestyle=STYLES[a], linewidth=2) for a in ARMS]
    labels = ["told: full descriptor (solid)", "blind: wavelengths only (dashed)"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="?", default=Path("/root/catena_data/runs/pilot"))
    args = parser.parse_args()
    data = read_runs(args.runs)
    if not data:
        raise SystemExit(f"no validation lines in {args.runs}/seed*/metrics.jsonl")
    seeds = list(data)
    steps = sorted(set.intersection(*({key[0] for key in d} for d in data.values())))
    cameras = list(dict.fromkeys(key[2] for key in data[seeds[0]]))
    about = describe_cameras()

    last = steps[-1]
    print(f"{len(seeds)} seeds ({', '.join(seeds)}), last validation at step {last:,}")
    print("told, blind: mean over the seeds; per seed: told minus blind in each seed\n")
    for camera in cameras:
        name = "mean of the cameras" if camera == "mean" else camera
        print(name + (f": {about[camera]}" if camera in about else ""))
        print(f"  {'':<10}{'told':>7}{'blind':>8}{'told-blind':>12}   per seed")
        for metric, label in zip(METRICS, LABELS, strict=True):
            told = values(data, seeds, last, "told", camera, metric)
            blind = values(data, seeds, last, "blind", camera, metric)
            diff = told - blind
            mark = "  all +" if (diff > 0).all() else "  all -" if (diff < 0).all() else ""
            each = " ".join(f"{d:+.3f}" for d in diff)
            print(
                f"  {label:<10}{told.mean():7.3f}{blind.mean():8.3f}{diff.mean():+12.3f}   "
                f"{each}{mark}"
            )
        print()

    print("At every validation: mean of the cameras, averaged over the seeds")
    print(f"{'step':>7}   F1 told  blind  told-blind   cloud BA told  blind  told-blind")
    for step in steps:
        f1 = {arm: values(data, seeds, step, arm, "mean", "f1_mean").mean() for arm in ARMS}
        ba = {arm: values(data, seeds, step, arm, "mean", "cloud_ba").mean() for arm in ARMS}
        print(
            f"{step:>7,}   {f1['told']:7.3f} {f1['blind']:6.3f} {f1['told'] - f1['blind']:+11.3f}"
            f"   {ba['told']:13.3f} {ba['blind']:6.3f} {ba['told'] - ba['blind']:+11.3f}"
        )

    figure = args.runs / "pilot_curves.png"
    draw(figure, data, seeds, steps, cameras)
    print(f"\nwrote {figure}")


if __name__ == "__main__":
    main()
