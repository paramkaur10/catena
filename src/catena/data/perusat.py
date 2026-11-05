"""PeruSat-1 patches of CloudPeru2 for training and testing: surface reflectance in 4 bands at
2.8 m, labelled cloud or not cloud, and what the model is told about them.

    from catena.data import perusat
    windows = perusat.Windows(ROOT, "train", size=240)
    n, dn, label = windows.draw(rng)      # (4, 240, 240) uint16 and (240, 240) uint8
    rho = perusat.reflectance(dn)         # float32, DN / 255
    d = perusat.descriptors()             # (4, 15)

ROOT (DATA/chips/cloudperu2) holds images/<n>.tif (blue, green, red, near-infrared; 512 x 512
pixels, values 0-252, not georeferenced), masks/<n>_Mask.tif (255 cloud, 0 not cloud) and
index.csv (n, group, split, nodata_frac, cloud_frac): one patch of each group of 8 rotations
and flips, 5% of the groups for validation.

Labels in Catena's codes: cloud 4 (thickness unknown), not cloud 5 (clear or shadow), and 255
where the image is empty (0 in every band). CloudPeru2 is atmospherically corrected and does
not document its scale; DN / 255 gives believable ground (red about 0.10, near-infrared
about 0.30), so it is used, and training gives each sample a random gain on top so that the
model cannot lean on the exact scale. What the model is told: the band edges
(perusat1.yaml), 2.8 m pixels, surface reflectance; blur, noise, step and saturation unknown.
"""

from __future__ import annotations

import csv
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from catena import descriptor as desc
from catena.sensors import spec

BANDS = ("blue", "green", "red", "nir")  # the order of the images' bands
GRID_M = 2.8
SCALE = 255.0  # DN / 255: the scale CloudPeru2 does not document
CLOUD, NOT_CLOUD, NO_LABEL = 4, 5, 255


@contextmanager
def quiet():
    """The patches carry no georeferencing, which rasterio warns about on every file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        yield


def patches(root: Path, split: str) -> list[str]:
    """The patch numbers of one split ("train" or "validation"), from ROOT/index.csv."""
    with open(Path(root) / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"no '{split}' patches in {Path(root) / 'index.csv'}")
    return [r["n"] for r in rows]


def read(root: Path, n: str, window: Window | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A patch, or a window of it: DN (4, h, w) uint16 and the label (h, w) in Catena's codes."""
    root = Path(root)
    with quiet():
        with rasterio.open(root / "images" / f"{n}.tif") as src:
            dn = src.read(window=window)
        with rasterio.open(root / "masks" / f"{n}_Mask.tif") as src:
            mask = src.read(1, window=window)
    if dn.shape[0] != len(BANDS) or dn.shape[1:] != mask.shape:
        raise ValueError(f"patch {n}: image {dn.shape} and mask {mask.shape} do not fit")
    label = np.where(mask == 255, CLOUD, NOT_CLOUD).astype(np.uint8)
    label[~dn.any(axis=0)] = NO_LABEL
    return dn, label


def reflectance(dn: np.ndarray) -> np.ndarray:
    """DN as float32 surface reflectance, DN / 255."""
    return dn.astype(np.float32) / SCALE


def descriptors() -> np.ndarray:
    """PeruSat-1's chain as CloudPeru2 delivers it, one descriptor per band (4, 15)."""
    sensor = spec.load("perusat1")
    rows = []
    for name in BANDS:
        band = sensor.band(name)
        rows.append(desc.describe(band.lower_nm, band.upper_nm, GRID_M, GRID_M, surface=True))
    return np.stack(rows)


class Windows:
    """Training windows of size x size pixels from the patches of one split.

    draw(rng) gives (patch number, DN (4, size, size) uint16, label (size, size) uint8): a
    patch, then a window anywhere in it; a window less than half labelled is drawn again, up
    to `tries` times, and the best one is kept. card holds PeruSat-1's descriptors."""

    def __init__(self, root: Path, split: str = "train", size: int = 240, tries: int = 20):
        self.root, self.size, self.tries = Path(root), size, tries
        self.names = patches(self.root, split)
        with quiet(), rasterio.open(self.root / "images" / f"{self.names[0]}.tif") as src:
            self.height, self.width = src.height, src.width
        if min(self.height, self.width) < size:
            raise ValueError(f"CloudPeru2 patches of {self.height} px hold no {size} px window")
        self.card = descriptors()

    def __len__(self) -> int:
        return len(self.names)

    def draw(self, rng: np.random.Generator) -> tuple[str, np.ndarray, np.ndarray]:
        best = None
        for _ in range(self.tries):
            n = self.names[int(rng.integers(len(self.names)))]
            row = int(rng.integers(0, self.height - self.size + 1))
            col = int(rng.integers(0, self.width - self.size + 1))
            dn, label = read(self.root, n, Window(col, row, self.size, self.size))
            share = float(np.mean(label != NO_LABEL))
            if best is None or share > best[0]:
                best = (share, n, dn, label)
            if share >= 0.5:
                break
        _, n, dn, label = best
        return n, dn, label
