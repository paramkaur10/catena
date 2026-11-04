"""Test A of the pilot: both arms of every seed on the CloudSEN12+ test split, seen through
fixed virtual cameras at three window sizes, the told arm also with its descriptor hidden or
wrong.

    python -m catena.evaluate RUNS CHIPS [--windows 800] [--sizes 240 120 48]

RUNS holds one folder per seed (seed0, seed1, ...) with config.json and final.pt, as
scripts/pilot.sh made them; CHIPS is the CloudSEN12+ folder made by D2.

The cameras (TEST_CAMERAS):
  sentinel2                   Sentinel-2 as delivered, 13 bands
  rgbn_<10|30>m_mtf<m>_snr<s>  18 cameras with 4 bands (blue, green, red, NIR) at 10 or 30 m,
                              MTF 0.08, 0.15 or 0.3, SNR 15, 40 or 150 at reflectance 0.3,
                              12 bits, saturating at 1.2: all inside what the networks met
                              in training
  beyond_<property><value>    4 cameras at 10 m, MTF 0.3, SNR 150, 12 bits, saturating at
                              1.2, except for one property beyond the training range:
                              MTF 0.002 (blur sigma 1.12 pixels; training reached 0.81),
                              SNR 4 (training: 7 or more), 6 bits (8 or more), saturation
                              at 0.45 (0.6 or more)
Window sizes are in 10 m pixels. 240 is what the networks trained on; 48 is a small tile, as
on board, with little image to read the blur and the noise from.

What the told arm is shown (VARIANTS); the blind arm always gets the wavelengths and the
pixel size only:
  honest     the descriptor of what the camera did
  hidden     blur, noise, step and saturation unknown, as descriptor dropout taught it
  noise x3   the noise said 3 times larger than it is; noise /3: 3 times smaller
  blur x2    the blur said twice as wide
If the told arm reads its descriptor, the wrong ones move its scores.

Every arm, seed and variant sees the same windows with the same noise. Writes
RUNS/test_a.csv, one row per seed, arm, variant, camera and size, with the scores and the
4 x 4 confusion matrix (rows: label, columns: prediction; clear, thin, thick, shadow), and
prints:
  1. the mean F1 of the told arm under each variant, and of the blind arm;
  2. told minus blind per camera and size, for the mean F1 and for thin cloud, with one sign
     per seed: + told better, - told worse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from catena import descriptor as desc
from catena import sim
from catena import train as T
from catena.data import loader
from catena.model import CatenaNet

TEST_SEED = 20261104
RGBN = ("B02", "B03", "B04", "B08")
BASE = {"mtf": 0.3, "snr": 150, "bits": 12, "top": 1.2}
BEYOND = {"mtf": 0.002, "snr": 4, "bits": 6, "top": 0.45}
VARIANTS = ("honest", "hidden", "noise x3", "noise /3", "blur x2")
TABLES = (("f1_mean", "F1, mean of the 4 classes"), ("f1_thin", "F1 of thin cloud"))


def make_cameras() -> tuple:
    """(name, q, bands, chain) of every test camera; chain None is Sentinel-2 as delivered."""
    cameras = [("sentinel2", 1, loader.BANDS, None)]
    for q in (1, 3):
        for mtf in (0.08, 0.15, 0.3):
            for snr in (15, 40, 150):
                chain = {"mtf": mtf, "snr": snr, "bits": 12, "top": 1.2}
                cameras.append((f"rgbn_{10 * q}m_mtf{mtf}_snr{snr}", q, RGBN, chain))
    for key, value in BEYOND.items():
        cameras.append((f"beyond_{key}{value}", 1, RGBN, BASE | {key: value}))
    return tuple(cameras)


TEST_CAMERAS = make_cameras()


def told_variant(d: torch.Tensor, name: str) -> torch.Tensor:
    """The told arm's descriptors (..., bands, 15) under one variant. Values nobody knows, and
    the empty band slots, stay 0."""
    d = d.clone()
    if name == "hidden":
        for cols, flag in desc.CHAIN.values():
            d[..., cols] = 0.0
            d[..., flag] = 0.0
    elif name in ("noise x3", "noise /3"):
        cols, flag = desc.CHAIN["noise"]  # log10(sigma) + 3: x3 adds log10(3)
        sign = 1.0 if name == "noise x3" else -1.0
        d[..., cols] += sign * math.log10(3) * d[..., [flag]]
    elif name == "blur x2":
        d[..., desc.CHAIN["psf"][0][0]] *= 2.0
    elif name != "honest":
        raise ValueError(f"unknown variant: {name}")
    return d


class CameraWindows(Dataset):
    """One camera on n windows picked evenly through a split. Sample j is always the same
    image with the same noise, whichever networks look at it."""

    def __init__(self, windows: loader.GridWindows, n: int, camera: tuple, index: int) -> None:
        self.windows, self.camera, self.index = windows, camera, index
        count = min(n, len(windows))
        self.picks = np.linspace(0, len(windows) - 1, count).round().astype(int)
        self.keep = np.array([loader.BANDS.index(b) for b in camera[2]])

    def __len__(self) -> int:
        return len(self.picks)

    def __getitem__(self, j: int) -> dict:
        _, q, _, chain = self.camera
        s = self.windows[int(self.picks[j])]
        rng = np.random.default_rng([TEST_SEED, self.index, j])
        camera = None if chain is None else sim.fixed_camera(len(self.keep), **chain)
        x, d, real, label = T.render(
            s["image"], s["label"], s["sun_zenith"], q, self.keep, camera, rng
        )
        return {"image": x, "real": real, "d_told": d, "d_blind": desc.blind(d), "label": label}


def load_arms(run: Path, device: torch.device) -> dict:
    """Both arms of one finished run, ready to predict."""
    args = json.loads((run / "config.json").read_text())["args"]
    state = torch.load(run / "final.pt", map_location="cpu")
    fields = state["models"]["told"]["bands.embed.0.weight"].shape[1]  # 14 in the pilot
    arms = {}
    for arm in T.ARMS:
        net = CatenaNet(widths=tuple(args["widths"]), stem=args["stem"], fields=fields)
        net.load_state_dict(state["models"][arm])
        arms[arm] = net.to(device).eval()
    return arms


@torch.no_grad()
def evaluate(runs: dict, data: DataLoader, device: torch.device, amp: bool) -> dict:
    """Confusion matrices {(seed, arm, variant): 4 x 4} of every run on one camera."""
    conf = {}
    for batch in data:
        x = batch["image"].to(device, non_blocking=True)
        real = batch["real"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        told = batch["d_told"].to(device, non_blocking=True)
        shown = {("told", v): told_variant(told, v) for v in VARIANTS}
        shown[("blind", "honest")] = batch["d_blind"].to(device, non_blocking=True)
        ok = label < 4
        for seed, arms in runs.items():
            for (arm, variant), d in shown.items():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = arms[arm](x, d, real)
                pairs = label[ok] * 4 + logits.argmax(dim=1)[ok]
                key = (seed, arm, variant)
                if key not in conf:
                    conf[key] = torch.zeros(4, 4, dtype=torch.long, device=device)
                conf[key] += torch.bincount(pairs, minlength=16).view(4, 4)
    return {key: m.cpu().numpy() for key, m in conf.items()}


def summary(results: dict, seeds: list[str], cameras: list[str], sizes: list[int]) -> None:
    """The two tables: what the told arm reads, and told minus blind per camera."""

    def mean(arm: str, variant: str, size: int) -> float:
        scores = [results[(s, arm, variant, c, size)]["f1_mean"] for s in seeds for c in cameras]
        return float(np.mean(scores))

    heads = "".join(f"{f'{size} px':>9}" for size in sizes)
    print("\n1. Does the told arm read its descriptor? F1 (mean of the 4 classes), averaged")
    print("   over the cameras and the seeds. Wrong descriptors should move it.")
    print(f"  {'':<17}{heads}")
    for variant in VARIANTS:
        print(f"  told {variant:<12}" + "".join(f"{mean('told', variant, z):9.3f}" for z in sizes))
    print(f"  {'blind':<17}" + "".join(f"{mean('blind', 'honest', z):9.3f}" for z in sizes))

    inside = [c for c in cameras if c.startswith("rgbn_")]
    beyond = [c for c in cameras if c.startswith("beyond_")]
    groups = [(c, [c]) for c in cameras]
    groups += [("mean, inside the range", inside)] if inside else []
    groups += [("mean, beyond the range", beyond)] if beyond else []
    for metric, title in TABLES:
        print(f"\n2. Told minus blind, {title}: mean over the seeds, then one sign per seed")
        print(f"  {'camera':<26}" + "".join(f"{f'{size} px':>15}" for size in sizes))
        for name, members in groups:
            cells = []
            for size in sizes:
                diffs = [
                    np.mean(
                        [
                            results[(s, "told", "honest", c, size)][metric]
                            - results[(s, "blind", "honest", c, size)][metric]
                            for c in members
                        ]
                    )
                    for s in seeds
                ]
                signs = "".join("+" if d > 0 else "-" if d < 0 else "0" for d in diffs)
                cells.append(f"{np.mean(diffs):+.3f} {signs}")
            print(f"  {name:<26}" + "".join(f"{cell:>15}" for cell in cells))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, help="one folder per seed with config.json and final.pt")
    p.add_argument("chips", type=Path, help="the CloudSEN12+ folder made by D2")
    p.add_argument("--windows", type=int, default=800, help="windows per camera and size")
    p.add_argument("--sizes", type=int, nargs="+", default=[240, 120, 48], help="10 m pixels")
    p.add_argument("--cameras", nargs="+", help="only these cameras (default: all)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=6, help="data-loading processes")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, help="the CSV to write (default RUNS/test_a.csv)")
    args = p.parse_args(argv)
    unknown = set(args.cameras or []) - {c[0] for c in TEST_CAMERAS}
    if unknown:
        p.error(f"unknown cameras: {', '.join(sorted(unknown))}")
    for size in args.sizes:
        for name, q, _, _ in TEST_CAMERAS:
            if (not args.cameras or name in args.cameras) and (size % q or (size // q) % 8):
                p.error(f"size {size} does not suit {name}: size / {q} must be a multiple of 8")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = True
    folders = sorted(path.parent for path in args.runs.glob("*/final.pt"))
    if not folders:
        raise SystemExit(f"no finished runs (*/final.pt) in {args.runs}")
    runs = {folder.name: load_arms(folder, device) for folder in folders}
    chosen = [c for c in TEST_CAMERAS if not args.cameras or c[0] in args.cameras]
    print(
        f"{len(runs)} runs ({', '.join(runs)}) on {device}; {len(chosen)} cameras; sizes "
        f"{', '.join(map(str, args.sizes))}; up to {args.windows} windows each",
        flush=True,
    )

    started, results, rows = time.perf_counter(), {}, []
    for size in args.sizes:
        windows = loader.GridWindows(args.chips, "test", size=size)
        for index, camera in enumerate(TEST_CAMERAS):
            if camera not in chosen:
                continue
            began = time.perf_counter()
            data = CameraWindows(windows, args.windows, camera, index)
            batches = DataLoader(
                data,
                batch_size=args.batch,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=T.quiet,
            )
            for (seed, arm, variant), conf in evaluate(runs, batches, device, amp).items():
                scores = T.scores(conf)
                results[(seed, arm, variant, camera[0], size)] = scores
                row = {"seed": seed, "arm": arm, "variant": variant, "camera": camera[0]}
                row |= {"size": size} | scores
                rows.append(row | {"confusion": " ".join(str(int(v)) for v in conf.ravel())})
            print(
                f"  {camera[0]:<26}{size:>4} px {len(data):>5} windows "
                f"{time.perf_counter() - began:6.0f} s",
                flush=True,
            )

    out = args.out or args.runs / "test_a.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary(results, list(runs), [c[0] for c in chosen], args.sizes)
    minutes = (time.perf_counter() - started) / 60
    print(f"\nwrote {out} ({len(rows)} rows) in {minutes:.0f} min", flush=True)


if __name__ == "__main__":
    main()
