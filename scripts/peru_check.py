"""CloudPeru2 straight from its zip: copies, value range and masks. Nothing is unpacked.

    python peru_check.py CloudPeru2.zip OUT.csv

The 22,400 patches should be 2,800 originals x 8 rotations/flips. Every image gets a
fingerprint that the 8 transforms cannot change: the band sums and sums of squares, plus
the 4 corner pixels and the 4 centre pixels taken as unordered sets. Images with the same
fingerprint are treated as copies of one patch, and a sample of groups is then checked pixel
by pixel. One row per image goes to OUT.csv.
"""

from __future__ import annotations

import csv
import random
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio


def number(member: str) -> int:
    return int(Path(member).stem.split("_")[0])  # "PatchesM/12_Mask.tif" -> 12


def read(zip_path: Path, member: str) -> np.ndarray:
    with rasterio.open(f"/vsizip/{zip_path}/{member}") as src:
        return src.read()


def fingerprint(img: np.ndarray) -> tuple:
    a = img.astype(np.int64)
    h, w = a.shape[1:]
    corners = sorted(tuple(a[:, r, c].tolist()) for r in (0, h - 1) for c in (0, w - 1))
    centre = [(r, c) for r in (h // 2 - 1, h // 2) for c in (w // 2 - 1, w // 2)]
    middle = sorted(tuple(a[:, r, c].tolist()) for r, c in centre)
    sums = a.sum(axis=(1, 2)).tolist()
    squares = (a * a).sum(axis=(1, 2)).tolist()
    return (tuple(sums), tuple(squares), tuple(corners), tuple(middle))


def transforms(x: np.ndarray) -> list[np.ndarray]:
    """The 8 rotations and flips of (..., H, W)."""
    out = []
    for flipped in (x, x[..., ::-1]):
        out += [np.rot90(flipped, k, axes=(-2, -1)) for k in range(4)]
    return out


def main() -> None:
    warnings.simplefilter("ignore")  # the patches carry no georeferencing
    zip_path, out_csv = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    names = zipfile.ZipFile(zip_path).namelist()
    images = sorted(
        (n for n in names if n.startswith("PatchesI/") and n.endswith(".tif")), key=number
    )
    masks = {number(n): n for n in names if n.startswith("PatchesM/") and n.endswith(".tif")}
    first, last = number(images[0]), number(images[-1])
    print(f"{len(images)} images, {len(masks)} masks, numbered {first} to {last}")

    hist = np.zeros((4, 65536), np.int64)  # value histogram per band, valid pixels only
    class_sum = {"cloud": np.zeros(4), "clear": np.zeros(4)}
    class_px = Counter()
    mask_values, mask_on_nodata = Counter(), Counter()
    groups, rows = defaultdict(list), []
    for i, member in enumerate(images):
        n = number(member)
        img = read(zip_path, member)
        mask = read(zip_path, masks[n])[0]
        valid = img.any(axis=0)  # a pixel with all four bands 0 lies outside the scene
        for b in range(min(4, img.shape[0])):
            hist[b] += np.bincount(img[b][valid], minlength=65536)[:65536]
        values, counts = np.unique(mask, return_counts=True)
        mask_values.update(dict(zip(values.tolist(), counts.tolist(), strict=True)))
        values, counts = np.unique(mask[~valid], return_counts=True)
        mask_on_nodata.update(dict(zip(values.tolist(), counts.tolist(), strict=True)))
        cloud = (mask == 255) & valid
        clear = (mask == 0) & valid
        class_sum["cloud"] += img[:, cloud].sum(axis=1)
        class_sum["clear"] += img[:, clear].sum(axis=1)
        class_px.update(cloud=int(cloud.sum()), clear=int(clear.sum()))
        groups[fingerprint(img)].append(n)
        rows.append(
            {
                "n": n,
                "bands": img.shape[0],
                "nodata_frac": round(1 - valid.mean(), 4),
                "cloud_frac": round(cloud.sum() / max(valid.sum(), 1), 4),
                **{f"max_b{b + 1}": int(img[b].max()) for b in range(img.shape[0])},
            }
        )
        if (i + 1) % 2000 == 0:
            print(f"   read {i + 1}")

    group_of = {}
    for g, members in enumerate(sorted(groups.values(), key=min)):
        for n in members:
            group_of[n] = g
    for row in rows:
        row["group"] = group_of[row["n"]]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\nvalues of valid pixels, per band:")
    for b in range(4):
        h = hist[b]
        total = h.sum()
        cdf = np.cumsum(h) / total
        p = [int(np.searchsorted(cdf, q)) for q in (0.01, 0.5, 0.99)]
        top = int(np.nonzero(h)[0].max())
        print(
            f"   band {b + 1}: 1% {p[0]}, median {p[1]}, 99% {p[2]}, max {top}, "
            f"share at 255: {h[255] / total:.3%}, share above 255: {h[256:].sum() / total:.3%}"
        )
    print("\nmask values (all pixels):", dict(mask_values))
    print("mask values where the image has no data:", dict(mask_on_nodata))
    for name in ("cloud", "clear"):
        mean = class_sum[name] / max(class_px[name], 1)
        print(
            f"mean value of {name} pixels (mask {255 if name == 'cloud' else 0}), bands 1-4:",
            np.round(mean, 1).tolist(),
        )

    sizes = Counter(len(m) for m in groups.values())
    print(f"\n{len(groups)} distinct patches; group sizes {dict(sorted(sizes.items()))}")
    blocks = sum(sorted(m) == list(range(min(m), min(m) + len(m))) for m in groups.values())
    print(f"{blocks} of {len(groups)} groups are runs of consecutive numbers")
    starts = Counter((min(m) - 1) % 8 for m in groups.values() if len(m) == 8)
    print("where groups of 8 start, as (first number - 1) mod 8:", dict(starts))

    random.seed(0)
    copies = [m for m in groups.values() if len(m) > 1]
    sample = random.sample(copies, k=min(30, len(copies)))
    exact = 0
    for members in sample:
        base = read(zip_path, f"PatchesI/{members[0]}.tif")
        base_mask = read(zip_path, masks[members[0]])
        ok = True
        for n in members[1:]:
            img, mask = read(zip_path, f"PatchesI/{n}.tif"), read(zip_path, masks[n])
            hits = [k for k, t in enumerate(transforms(base)) if np.array_equal(t, img)]
            ok &= bool(hits) and np.array_equal(transforms(base_mask)[hits[0]], mask)
        exact += ok
    print(f"pixel-by-pixel check: {exact} of {len(sample)} sampled groups are exact rotations or")
    print("flips of their first image, and their masks follow the same transform")
    print(f"one row per image written to {out_csv}")


if __name__ == "__main__":
    main()
