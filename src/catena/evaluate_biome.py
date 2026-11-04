"""Tests B and C: both arms of every seed on real scenes, the told arm also with part of its
descriptor hidden or changed. Test B: Landsat 8 (Biome). Test C: Landsat 7 (L7 Irish), a
satellite no network trains on.

    python -m catena.evaluate_biome RUNS DATA [--source biome] [--sizes 80 16]
        [--table pilot] [--variants honest "no noise" hidden] [--out CSV]

RUNS holds one folder per seed with config.json and final.pt (scripts/pilot.sh). DATA is the
folder of the scenes, with pilot.csv and index.csv: for --source biome the one
scripts/biome_convert.py made (catena.data.biome), for --source irish the one
scripts/irish_fetch.sh unpacked into, indexed by python -m catena.data.irish
(catena.data.irish). Both have 30 m pixels.

Each scene is cut into square tiles of each size, in 30 m pixels: 80 (2.4 km) is what the
networks trained on, 16 (480 m) is a small tile. Rows and columns that do not fill a tile are
left out, and so are tiles without a labelled pixel. Both arms of every seed predict every
tile, the told arm under each variant asked for (VARIANTS unless --variants; honest always):
  honest     the satellite's chain as its data module tells it: for Landsat 8 band edges,
             30 m, noise, step and saturation (the blur is not published); for Landsat 7
             band edges, 30 m, step and saturation from each scene's gains (no noise, no
             blur published)
  no noise   the same with the noise unknown
  hidden     band edges, pixel size and top of atmosphere only
  no step    honest with the step unknown
  no top     honest with the saturation unknown
  fine step  honest with the step told as 0.0001, as every real Sentinel-2 image was in
             training. Landsat 8's honest step, one level of the 12 bits OLI sends down
             (0.0003 to 0.0009 in the pilot scenes), came in training only with
             simulated cameras.
The blind arm gets the band edges only.

Scores pool the pixels of the scenes. Cloud (thin or thick) against not cloud, and the F1 of
thin and of thick cloud, use every scene; the mean F1 of the four classes uses only the
scenes whose shadows were labelled, since elsewhere clear may hide shadow. Writes
RUNS/test_b.csv (Biome) or RUNS/test_c.csv (L7 Irish), or --out (one row per seed, arm,
variant, scene and size, with the 4 x 4 confusion matrix: rows label, columns prediction;
clear, thin, thick, shadow) and prints the scores per variant, then told (honest) minus
blind per biome (per zone for L7 Irish), for cloud BA and for thin cloud, with one sign per
seed: + told better, - told worse.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from catena import descriptor as desc
from catena import train as T
from catena.data import biome, irish
from catena.evaluate import load_arms

# --source: the data module, the satellite, the test, and what scenes are grouped by
SOURCES = {
    "biome": (biome, "Landsat 8", "test_b", "biome"),
    "irish": (irish, "Landsat 7", "test_c", "zone"),
}

VARIANTS = ("honest", "no noise", "hidden")  # Test B as it was first run
HIDE = {
    "honest": (),
    "no noise": ("noise",),
    "hidden": tuple(desc.CHAIN),
    "no step": ("step",),
    "no top": ("top",),
    "fine step": (),
}
FINE_STEP = math.log10(desc.S2_STEP) + 4  # the step code descriptor.describe gives 0.0001
PIXELS_PER_BATCH = 409_600  # 64 tiles of 80 x 80


def told_variant(d: torch.Tensor, name: str) -> torch.Tensor:
    """The told arm's descriptors (..., bands, 14) under a variant: some chain groups made
    unknown, or the step told as Sentinel-2's."""
    if name not in HIDE:
        raise ValueError(f"unknown variant: {name}")
    d = d.clone()
    for group in HIDE[name]:
        cols, flag = desc.CHAIN[group]
        d[..., cols] = 0.0
        d[..., flag] = 0.0
    if name == "fine step":
        cols, flag = desc.CHAIN["step"]
        d[..., cols] = FINE_STEP
        d[..., flag] = 1.0
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
def score_scene(
    runs, scene, dn, label, d, size, device, amp, variants=VARIANTS, reflectance=None
) -> dict:
    """Confusion matrices {(seed, arm, variant): 4 x 4} of one scene's tiles of one size;
    reflectance turns the scene's DN into reflectance (default: Biome's)."""
    reflectance = reflectance or biome.reflectance
    x_all, y_all = tiles(dn, label, size)
    told = torch.from_numpy(d).to(device)
    shown = {("told", v): told_variant(told, v) for v in variants}
    shown[("blind", "honest")] = torch.from_numpy(desc.blind(d)).to(device)
    conf = {
        (seed, arm, variant): torch.zeros(4, 4, dtype=torch.long, device=device)
        for seed in runs
        for arm, variant in shown
    }
    batch = max(1, PIXELS_PER_BATCH // (size * size))
    for start in range(0, len(y_all), batch):
        x = torch.from_numpy(reflectance(scene, x_all[start : start + batch])).to(device)
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


def summary(
    conf: dict,
    seeds: list[str],
    scenes: list,
    sizes: list[int],
    variants=VARIANTS,
    group: str = "biome",
) -> None:
    names = [s.name for s in scenes]
    shadowed = [s.name for s in scenes if s.has_shadow]
    print(f"\n1. Scores pooled over the {len(names)} scenes, mean of the seeds (F1 mean: only")
    print(f"   the {len(shadowed)} scenes with shadow labels)")
    shown = [("told", v) for v in variants] + [("blind", "honest")]
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
        print(f"\n2. Told minus blind, {title}: per {group}, mean of the seeds, one sign per seed")
        print(f"  {group:<22}" + "".join(f"{f'{size} px':>15}" for size in sizes))
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
    p.add_argument("data", type=Path, help="the scenes' folder, with pilot.csv and index.csv")
    p.add_argument("--source", choices=list(SOURCES), default="biome", help="whose scenes")
    p.add_argument("--table", default="pilot", help="pilot (a few per group) or index (all)")
    p.add_argument("--sizes", type=int, nargs="+", default=[80, 16], help="30 m pixels")
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(HIDE),
        help="what the told arm is shown (honest is always added)",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, help="the CSV (default RUNS/test_b.csv or test_c.csv)")
    args = p.parse_args(argv)
    wrong = [size for size in args.sizes if size <= 0 or size % 8]
    if wrong:
        p.error(f"tile sizes must be multiples of 8: {wrong}")
    args.variants = list(dict.fromkeys(["honest", *args.variants]))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    folders = sorted(path.parent for path in args.runs.glob("*/final.pt"))
    if not folders:
        raise SystemExit(f"no finished runs (*/final.pt) in {args.runs}")
    runs = {folder.name: load_arms(folder, device) for folder in folders}
    data, satellite, test, group = SOURCES[args.source]
    chosen = data.scenes(args.data, args.table)
    print(
        f"{len(runs)} runs ({', '.join(runs)}) on {device}; {len(chosen)} {satellite} scenes; "
        f"tiles of {', '.join(map(str, args.sizes))} px; told: {', '.join(args.variants)}",
        flush=True,
    )

    started, conf, rows = time.perf_counter(), {}, []
    for scene in chosen:
        began = time.perf_counter()
        dn, label = data.read(scene)
        d = data.descriptors(scene)
        for size in args.sizes:
            found = score_scene(
                runs, scene, dn, label, d, size, device, amp, args.variants, data.reflectance
            )
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

    out = args.out or args.runs / f"{test}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary(conf, list(runs), chosen, args.sizes, args.variants, group)
    minutes = (time.perf_counter() - started) / 60
    print(f"\nwrote {out} ({len(rows)} rows) in {minutes:.0f} min", flush=True)


if __name__ == "__main__":
    main()
