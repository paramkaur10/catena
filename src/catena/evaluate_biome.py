"""Test B of the pilot: both arms of every seed on real Landsat 8 scenes (Biome), a camera
neither arm ever saw, the told arm also with part of its descriptor hidden.

    python -m catena.evaluate_biome RUNS BIOME [--sizes 80 16] [--table pilot]

RUNS holds one folder per seed with config.json and final.pt (scripts/pilot.sh); BIOME is
the folder scripts/biome_convert.py made (pilot.csv, index.csv, one folder per scene).

Each scene is cut into square tiles of each size, in 30 m pixels: 80 (2.4 km) is what the
networks trained on, 16 (480 m) is a small tile. Rows and columns that do not fill a tile are
left out, and so are tiles without a labelled pixel. Both arms of every seed predict every
tile, the told arm under each of VARIANTS:
  honest     Landsat 8's chain (catena.data.biome): band edges, 30 m, noise, step and
             saturation; the blur is not published, so it stays unknown
  no noise   the same with the noise unknown
  hidden     band edges, pixel size and top of atmosphere only
The blind arm gets the band edges only.

Scores pool the pixels of the scenes. Cloud (thin or thick) against not cloud, and the F1 of
thin and of thick cloud, use every scene; the mean F1 of the four classes uses only the
scenes whose shadows were labelled, since elsewhere clear may hide shadow. Writes
RUNS/test_b.csv (one row per seed, arm, variant, scene and size, with the 4 x 4 confusion
matrix: rows label, columns prediction; clear, thin, thick, shadow) and prints the scores
per variant, then told minus blind per biome, for cloud BA and for thin cloud, with one sign
per seed: + told better, - told worse.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from catena import descriptor as desc
from catena import train as T
from catena.data import biome
from catena.evaluate import load_arms

VARIANTS = ("honest", "no noise", "hidden")
HIDE = {"honest": (), "no noise": ("noise",), "hidden": tuple(desc.CHAIN)}
PIXELS_PER_BATCH = 409_600  # 64 tiles of 80 x 80


def told_variant(d: torch.Tensor, name: str) -> torch.Tensor:
    """The told arm's descriptors (..., bands, 14) with some chain groups made unknown."""
    if name not in HIDE:
        raise ValueError(f"unknown variant: {name}")
    d = d.clone()
    for group in HIDE[name]:
        cols, flag = desc.CHAIN[group]
        d[..., cols] = 0.0
        d[..., flag] = 0.0
    return d


def tiles(dn: np.ndarray, label: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """The size x size tiles of a scene that hold a labelled pixel: DN (n, bands, size, size)
    and labels (n, size, size). Rows and columns that do not fill a tile are left out."""
    bands, height, width = dn.shape
    rows, cols = height // size, width // size
    xs, ys = [], []
    for r in range(rows):
        top = r * size
        y = label[top : top + size, : cols * size].reshape(size, cols, size).transpose(1, 0, 2)
        keep = (y != T.NO_LABEL).any(axis=(1, 2))
        if keep.any():
            x = dn[:, top : top + size, : cols * size].reshape(bands, size, cols, size)
            xs.append(x.transpose(2, 0, 1, 3)[keep])
            ys.append(y[keep])
    if not xs:
        return np.zeros((0, bands, size, size), dn.dtype), np.zeros((0, size, size), label.dtype)
    return np.concatenate(xs), np.concatenate(ys)


@torch.no_grad()
def score_scene(runs, scene, dn, label, d, size, device, amp) -> dict:
    """Confusion matrices {(seed, arm, variant): 4 x 4} of one scene's tiles of one size."""
    x_all, y_all = tiles(dn, label, size)
    told = torch.from_numpy(d).to(device)
    shown = {("told", v): told_variant(told, v) for v in VARIANTS}
    shown[("blind", "honest")] = torch.from_numpy(desc.blind(d)).to(device)
    conf = {
        (seed, arm, variant): torch.zeros(4, 4, dtype=torch.long, device=device)
        for seed in runs
        for arm, variant in shown
    }
    batch = max(1, PIXELS_PER_BATCH // (size * size))
    for start in range(0, len(y_all), batch):
        x = torch.from_numpy(biome.reflectance(scene, x_all[start : start + batch])).to(device)
        y = torch.from_numpy(y_all[start : start + batch].astype(np.int64)).to(device)
        n = x.shape[0]
        real = torch.ones(n, x.shape[1], dtype=torch.bool, device=device)
        ok = y < 4
        for seed, arms in runs.items():
            for (arm, variant), shown_d in shown.items():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = arms[arm](x, shown_d.expand(n, -1, -1), real)
                pairs = y[ok] * 4 + logits.argmax(dim=1)[ok]
                conf[(seed, arm, variant)] += torch.bincount(pairs, minlength=16).view(4, 4)
    return {key: m.cpu().numpy() for key, m in conf.items()}


def pooled(conf: dict, seeds: list[str], arm: str, variant: str, names: list[str], size: int):
    """The scores of one arm, per seed, on the pixels of some scenes pooled."""
    return [T.scores(sum(conf[(s, arm, variant, n, size)] for n in names)) for s in seeds]


def summary(conf: dict, seeds: list[str], scenes: list[biome.Scene], sizes: list[int]) -> None:
    names = [s.name for s in scenes]
    shadowed = [s.name for s in scenes if s.has_shadow]
    print(f"\n1. Scores pooled over the {len(names)} scenes, mean of the seeds (F1 mean: only")
    print(f"   the {len(shadowed)} scenes with shadow labels)")
    shown = [("told", v) for v in VARIANTS] + [("blind", "honest")]
    for size in sizes:
        head = f"   {size} px tiles ({size * 30 / 1000:g} km)"
        print(f"{head:<26}cloud BA   F1 thin  F1 thick   F1 mean")
        for arm, variant in shown:
            every = pooled(conf, seeds, arm, variant, names, size)
            cells = [np.mean([s[k] for s in every]) for k in ("cloud_ba", "f1_thin", "f1_thick")]
            if shadowed:
                labelled = pooled(conf, seeds, arm, variant, shadowed, size)
                cells.append(np.mean([s["f1_mean"] for s in labelled]))
            name = f"told {variant}" if arm == "told" else "blind"
            print(f"  {name:<24}" + "".join(f"{c:9.3f}" for c in cells))

    biomes = dict.fromkeys(s.biome for s in scenes)  # in the order of the table
    groups = [(b, [s.name for s in scenes if s.biome == b]) for b in biomes]
    groups.append((f"all {len(names)} scenes", names))
    for metric, title in (("cloud_ba", "cloud BA"), ("f1_thin", "F1 of thin cloud")):
        print(f"\n2. Told minus blind, {title}: per biome, mean over the seeds, one sign per seed")
        print(f"  {'biome':<22}" + "".join(f"{f'{size} px':>15}" for size in sizes))
        for name, members in groups:
            cells = []
            for size in sizes:
                told = pooled(conf, seeds, "told", "honest", members, size)
                blind = pooled(conf, seeds, "blind", "honest", members, size)
                diffs = [t[metric] - b[metric] for t, b in zip(told, blind, strict=True)]
                signs = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in diffs)
                cells.append(f"{np.mean(diffs):+.3f} {signs}")
            print(f"  {name:<22}" + "".join(f"{cell:>15}" for cell in cells))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, help="one folder per seed with config.json and final.pt")
    p.add_argument("biome", type=Path, help="the folder made by scripts/biome_convert.py")
    p.add_argument("--table", default="pilot", help="pilot (24 scenes) or index (all 96)")
    p.add_argument("--sizes", type=int, nargs="+", default=[80, 16], help="30 m pixels")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, help="the CSV to write (default RUNS/test_b.csv)")
    args = p.parse_args(argv)
    wrong = [size for size in args.sizes if size <= 0 or size % 8]
    if wrong:
        p.error(f"tile sizes must be multiples of 8: {wrong}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    folders = sorted(path.parent for path in args.runs.glob("*/final.pt"))
    if not folders:
        raise SystemExit(f"no finished runs (*/final.pt) in {args.runs}")
    runs = {folder.name: load_arms(folder, device) for folder in folders}
    chosen = biome.scenes(args.biome, args.table)
    print(
        f"{len(runs)} runs ({', '.join(runs)}) on {device}; {len(chosen)} Landsat 8 scenes; "
        f"tiles of {', '.join(map(str, args.sizes))} px",
        flush=True,
    )

    started, conf, rows = time.perf_counter(), {}, []
    for scene in chosen:
        began = time.perf_counter()
        dn, label = biome.read(scene)
        d = biome.descriptors(scene)
        for size in args.sizes:
            found = score_scene(runs, scene, dn, label, d, size, device, amp)
            for (seed, arm, variant), m in found.items():
                conf[(seed, arm, variant, scene.name, size)] = m
                row = {"seed": seed, "arm": arm, "variant": variant, "scene": scene.name}
                row |= {"biome": scene.biome, "size": size} | T.scores(m)
                rows.append(row | {"confusion": " ".join(str(int(v)) for v in m.ravel())})
        print(
            f"  {scene.name}  {scene.biome:<12} sun {scene.sun_elevation:4.1f} deg  "
            f"{time.perf_counter() - began:5.0f} s",
            flush=True,
        )

    out = args.out or args.runs / "test_b.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary(conf, list(runs), chosen, args.sizes)
    minutes = (time.perf_counter() - started) / 60
    print(f"\nwrote {out} ({len(rows)} rows) in {minutes:.0f} min", flush=True)


if __name__ == "__main__":
    main()
