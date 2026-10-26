"""CloudSEN12+ windows for training and testing (D3).

D2 (catena.data.cloudsen12) copied the hand-labelled patches to CHIPS/<split>/<id>.s2l1c.tif
(13 bands, uint16) and <id>.target.tif (CloudSEN12's label codes), listed in
CHIPS/index.csv. Here they are cut into windows of 240 x 240 pixels at 10 m: 240 divides by
1, 3 and 5, the simulator's pixel sizes, and stays a multiple of 8 for the U-Net at each of
them (240, 80 and 48 pixels).

Three traps of the dataset are fixed on the way:
  1. empty pixels: a pixel that is 0 in every band holds no image, yet the label calls it
     clear. It gets label 255 (ignored by the loss) and reflectance 0. So does a pixel
     with 65535 (no data) in any band.
  2. frames: 509-pixel images sit in 512-pixel frames and 2000-pixel images in 2048-pixel
     ones, and the 847 big patches hold more pixels than the 10,000 small ones. Training
     windows are drawn by area, not by patch, and a window less than half labelled is
     drawn again.
  3. the offset: from processing baseline 04.00 (January 2022) L1C values carry +1000, so
     reflectance = (DN - 1000) / 10000; before it, DN / 10000.

    python -m catena.data.loader CHIPS [--split train] [--n 2000] [--figure FILE]

draws n training windows, prints what they hold and how long they took, and with --figure
saves the first eight as a picture.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# the order of the 13 bands in every image
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
SIZE = 240
NO_LABEL = 255
NODATA = 65535  # the images' no-data value, masked like the empty pixels
OFFSET_FROM = "04.00"  # processing baselines from this one on carry the +1000 offset

# CloudSEN12's codes (0 clear, 1 thick, 2 thin, 3 shadow) to ours (0 clear, 1 thin,
# 2 thick, 3 shadow); any other value becomes 255
CODES = np.full(256, NO_LABEL, np.uint8)
CODES[[0, 1, 2, 3]] = [0, 2, 1, 3]


@dataclass(frozen=True)
class Patch:
    id: str
    image: Path
    label: Path
    height: int
    width: int
    offset: int  # 1000 from processing baseline 04.00, else 0
    sun_zenith: float  # the patch's mean, in degrees


def offset_of(baseline: str) -> int:
    """The DN offset of a processing baseline such as "02.07" or "05.09"."""
    return 1000 if baseline >= OFFSET_FROM else 0


def patches(chips: Path, split: str) -> list[Patch]:
    """The patches of one split ("train", "validation" or "test"), from CHIPS/index.csv."""
    with open(chips / "index.csv", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == split]
    if not rows:
        raise ValueError(f"no '{split}' patches in {chips / 'index.csv'}")
    return [
        Patch(
            id=row["id"],
            image=chips / row["s2l1c_file"],
            label=chips / row["target_file"],
            height=int(row["height"]),
            width=int(row["width"]),
            offset=offset_of(row["baseline"]),
            sun_zenith=float(row["sun_zenith"] or "nan"),
        )
        for row in rows
    ]


def read(patch: Patch, row: int, col: int, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """One window of a patch, its top-left corner at (row, col): reflectance (13, size, size)
    as float32 and the label (size, size) in Catena's codes, with traps 1 and 3 fixed."""
    window = Window(col, row, size, size)
    with rasterio.open(patch.image) as src:
        dn = src.read(window=window)
    with rasterio.open(patch.label) as src:
        raw = src.read(1, window=window)
    empty = ~dn.any(axis=0) | (dn == NODATA).any(axis=0)
    image = (dn.astype(np.float32) - patch.offset) / 10000
    image[:, empty] = 0.0
    label = CODES[raw]
    label[empty] = NO_LABEL
    return image, label


def dihedral(x: np.ndarray, turn: int) -> np.ndarray:
    """One of the 8 rotations and flips of (..., rows, cols): turn % 4 quarter turns, then a
    left-right flip when turn >= 4."""
    x = np.rot90(x, turn % 4, axes=(-2, -1))
    if turn >= 4:
        x = x[..., ::-1]
    return np.ascontiguousarray(x)


class TrainingWindows:
    """Random training windows, drawn by area (trap 2), then turned and flipped at random.

    Sample i depends only on the seed and i, so a run repeats exactly whatever the number of
    data-loading workers; `length` is the number of samples in one pass. A sample is a dict:
    image, label, the patch id, the window's row and column in the patch, its turn (0-7, see
    dihedral) and the patch's sun zenith angle."""

    def __init__(
        self,
        chips: Path | str,
        split: str = "train",
        size: int = SIZE,
        length: int = 100_000,
        seed: int = 0,
        min_labelled: float = 0.5,
        tries: int = 20,
    ) -> None:
        self.patches = patches(Path(chips), split)
        self.size, self.length, self.seed = size, length, seed
        self.min_labelled, self.tries = min_labelled, tries
        small = [p.id for p in self.patches if min(p.height, p.width) < size]
        if small:
            raise ValueError(f"{len(small)} patches are smaller than {size} px, e.g. {small[0]}")
        area = np.array([p.height * p.width for p in self.patches], dtype=np.float64)
        self.cdf = np.cumsum(area) / area.sum()

    def __len__(self) -> int:
        return self.length

    def draw(self, rng: np.random.Generator) -> tuple[Patch, int, int]:
        """A patch, with a chance in proportion to its area, then a place in it."""
        k = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        patch = self.patches[min(k, len(self.patches) - 1)]
        row = int(rng.integers(0, patch.height - self.size + 1))
        col = int(rng.integers(0, patch.width - self.size + 1))
        return patch, row, col

    def __getitem__(self, i: int) -> dict:
        if not 0 <= i < self.length:
            raise IndexError(i)
        rng = np.random.default_rng([self.seed, int(i)])
        best = None
        for _ in range(self.tries):  # a window less than half labelled is drawn again
            patch, row, col = self.draw(rng)
            image, label = read(patch, row, col, self.size)
            share = float(np.mean(label != NO_LABEL))
            if best is None or share > best[0]:
                best = (share, patch, row, col, image, label)
            if share >= self.min_labelled:
                break
        _, patch, row, col, image, label = best
        turn = int(rng.integers(8))
        return {
            "image": dihedral(image, turn),
            "label": dihedral(label, turn),
            "id": patch.id,
            "row": row,
            "col": col,
            "turn": turn,
            "sun_zenith": patch.sun_zenith,
        }


def grid(height: int, width: int, size: int = SIZE) -> list[tuple[int, int]]:
    """Top-left corners of windows that tile a patch without overlap. The last rows and
    columns can be left out (29 of 509, 80 of 2000), so that every pixel is scored once and
    all windows of a patch share one pixel grid at q = 3 and 5 too."""
    return [
        (row, col)
        for row in range(0, height - size + 1, size)
        for col in range(0, width - size + 1, size)
    ]


class GridWindows:
    """Fixed windows for validation and testing: every patch of the split tiled by grid(),
    never turned. Window i is always the same, so scores from different runs compare."""

    def __init__(self, chips: Path | str, split: str = "validation", size: int = SIZE) -> None:
        self.size = size
        self.items = [
            (patch, row, col)
            for patch in patches(Path(chips), split)
            for row, col in grid(patch.height, patch.width, size)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        patch, row, col = self.items[i]
        image, label = read(patch, row, col, self.size)
        return {
            "image": image,
            "label": label,
            "id": patch.id,
            "row": row,
            "col": col,
            "turn": 0,
            "sun_zenith": patch.sun_zenith,
        }


# --- a look at the windows ----------------------------------------------------------------

CLASS_NAMES = {0: "clear", 1: "thin", 2: "thick", 3: "shadow", NO_LABEL: "no label"}
COLOURS = ["#e1e0d9", "#86b6ef", "#1c5cab", "#52514e", "#ffffff"]  # clear ... shadow, none


def show(samples: list[dict], frames: dict[str, int], path: Path) -> None:
    """Up to eight windows: true colour (B04, B03, B02) above, the label below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch as Swatch

    fig, axes = plt.subplots(4, 4, figsize=(6.7, 7.6), dpi=150)
    for ax in axes.flat:
        ax.set_axis_off()
    for k, s in enumerate(samples[:8]):
        top, ax = 2 * (k // 4), k % 4
        rgb = np.sqrt(np.clip(s["image"][[3, 2, 1]].transpose(1, 2, 0) / 0.8, 0, 1))
        classes = np.where(s["label"] == NO_LABEL, 4, s["label"])
        axes[top, ax].imshow(rgb, interpolation="nearest")
        axes[top + 1, ax].imshow(
            classes, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=4.5, interpolation="nearest"
        )
        axes[top, ax].set_title(
            f"{frames[s['id']]} px patch\nrow {s['row']}, col {s['col']}", fontsize=7, loc="left"
        )
    swatches = [
        Swatch(facecolor=c, edgecolor="#898781", label=n)
        for c, n in zip(COLOURS, CLASS_NAMES.values(), strict=True)
    ]
    fig.legend(handles=swatches, loc="lower center", ncol=5, fontsize=7, frameon=False)
    size = samples[0]["label"].shape[-1]
    fig.suptitle(
        f"Training windows, {size} x {size} px at 10 m: true colour above, label below",
        fontsize=9,
        x=0.02,
        ha="left",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chips", type=Path, help="the CloudSEN12+ folder made by D2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=2000, help="training windows to draw")
    parser.add_argument("--size", type=int, default=SIZE, help="window size in pixels")
    parser.add_argument("--figure", type=Path, help="save the first eight windows here")
    args = parser.parse_args(argv)

    windows = TrainingWindows(args.chips, args.split, size=args.size, length=args.n)
    frames = {p.id: max(p.height, p.width) for p in windows.patches}
    offsets = {p.id: p.offset for p in windows.patches}
    area = Counter()
    for p in windows.patches:
        area[frames[p.id]] += p.height * p.width
    total, biggest = sum(area.values()), max(area)
    counted = Counter(frames.values())
    shown = ", ".join(
        f"{counted[f]} of {f} px ({area[f] / total:.0%} of the area)" for f in sorted(counted)
    )
    with_offset = sum(v > 0 for v in offsets.values())
    print(f"{len(frames)} {args.split} patches: {shown}; {with_offset} with the +1000 offset")

    counts = np.zeros(256, np.int64)
    shares, dark, samples, from_biggest = [], {0: [], 1000: []}, [], 0
    start = time.perf_counter()
    for i in range(args.n):
        s = windows[i]
        labelled = s["label"] != NO_LABEL
        counts += np.bincount(s["label"].ravel(), minlength=256)
        shares.append(float(labelled.mean()))
        from_biggest += frames[s["id"]] == biggest
        if labelled.any():  # B10 (cirrus) is nearly black where there is no high cloud
            dark[offsets[s["id"]]].append(float(np.percentile(s["image"][10][labelled], 1)))
        if i < 8:
            samples.append(s)
    seconds = time.perf_counter() - start

    pixels = ", ".join(
        f"{name} {counts[code] / counts.sum():.1%}" for code, name in CLASS_NAMES.items()
    )
    print(
        f"{args.n} windows in {seconds:.0f} s: {1000 * seconds / args.n:.0f} ms each, one process"
    )
    print(f"   from the {biggest} px patches: {from_biggest / args.n:.0%}")
    print(f"   labelled share of a window: mean {np.mean(shares):.1%}, lowest {min(shares):.1%}")
    print(f"   pixels: {pixels}")
    for value, darkest in dark.items():
        if darkest:
            where = "with the offset removed" if value else "without an offset"
            print(
                f"   darkest B10 reflectance, {where}: {np.median(darkest):.4f} "
                f"(median over {len(darkest)} windows)"
            )
    if args.figure:
        show(samples, frames, args.figure)
        print(f"{len(samples)} windows drawn in {args.figure}")


if __name__ == "__main__":
    main()
