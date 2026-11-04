"""Landsat 8 Biome scenes, as scripts/biome_convert.py left them: reflectance, labels, what
the model is told about each band, and training windows.

    from catena.data import biome
    for scene in biome.scenes(Path("/root/catena_data/chips/l8_biome"), "pilot"):
        dn, label = biome.read(scene)         # (8, H, W) uint16 and (H, W) uint8
        rho = biome.reflectance(scene, dn)     # float32, 0 where the product has no data
        d = biome.descriptors(scene)           # (8, 15): Landsat 8's chain at this sun angle

Bands B1-B7 and B9 (no pan, no thermal). Labels in Catena's codes: 0 clear, 1 thin, 2 thick,
3 shadow, 255 no data; in a scene with has_shadow False, clear may hide shadow.

Reflectance = (DN x REFLECTANCE_MULT + REFLECTANCE_ADD) / sin(sun elevation), from the MTL.
The descriptor takes the band edges, 30 m, the on-orbit SNR and the saturation radiance from
landsat8_oli.yaml and brings them to this scene with the MTL:
  k = REFLECTANCE_MULT / (RADIANCE_MULT x sin(sun elevation)) is the reflectance of one
      radiance unit here, so noise (a, b) in radiance becomes (k a, k^2 b) and the
      saturation radiance L_sat becomes the top, k L_sat;
  the step is one level of the 12 bits OLI digitises up to saturation, top / 4095, by the
      rule every camera follows (catena.descriptor); the product's 16-bit numbers only
      rescale those 12 bits.
Landsat 8's MTF is not published, so the blur stays unknown (0, flag 0).

Windows(scenes) draws training windows: a scene by its area, then a window at least half
labelled. In a scene without shadow labels, its clear becomes 5 (not cloud), which the loss
reads as clear or shadow.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from catena import descriptor as desc
from catena.sensors import spec

BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9")  # the order of <scene>.bands.tif
GRID_M = 30.0
LEVELS = 4095  # OLI digitises 12 bits
SHOT_FRACTION = 0.7  # the share of shot noise at the SNR's radiance, as for Sentinel-2
NO_LABEL = 255
NOT_CLOUD = 5  # clear or shadow
KEYS = (
    "REFLECTANCE_MULT",
    "REFLECTANCE_ADD",
    "RADIANCE_MULT",
    "REFLECTANCE_MAXIMUM",
    "REFLECTANCE_MINIMUM",
)


def mtl_values(text: str) -> dict[str, str]:
    """The KEY = VALUE pairs of an MTL file."""
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


@dataclass(frozen=True)
class Scene:
    name: str
    biome: str
    folder: Path
    has_shadow: bool
    sun_elevation: float  # degrees, from the MTL
    numbers: dict  # {(KEY, band): float} for every KEYS entry and band

    @property
    def bands(self) -> Path:
        return self.folder / f"{self.name}.bands.tif"

    @property
    def mask(self) -> Path:
        return self.folder / f"{self.name}.mask.tif"

    def per_band(self, key: str) -> np.ndarray:
        return np.array([self.numbers[(key, b)] for b in BANDS])


def open_scene(folder: Path, biome: str, has_shadow: bool) -> Scene:
    """A converted scene, after checking that its MTL has every number needed."""
    name = folder.name
    mtl = mtl_values((folder / f"{name}_MTL.txt").read_text())
    wanted = [(key, b) for key in KEYS for b in BANDS]
    missing = [f"{key}_BAND_{b[1:]}" for key, b in wanted if f"{key}_BAND_{b[1:]}" not in mtl]
    missing += [] if "SUN_ELEVATION" in mtl else ["SUN_ELEVATION"]
    if missing:
        raise ValueError(f"{name}: the MTL lacks {', '.join(missing)}")
    numbers = {(key, b): float(mtl[f"{key}_BAND_{b[1:]}"]) for key, b in wanted}
    return Scene(name, biome, folder, has_shadow, float(mtl["SUN_ELEVATION"]), numbers)


def scenes(root: Path, table: str = "pilot") -> list[Scene]:
    """The scenes listed in ROOT/<table>.csv ("pilot": 24 scenes, "index": all 96)."""
    with open(root / f"{table}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no scenes in {root / f'{table}.csv'}")
    return [open_scene(root / r["scene"], r["biome"], r["has_shadow"] == "True") for r in rows]


def read(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The scene's DN (8, H, W) uint16 and its label (H, W) uint8."""
    with rasterio.open(scene.bands) as src:
        dn = src.read()
    with rasterio.open(scene.mask) as src:
        label = src.read(1)
    if dn.shape[0] != len(BANDS) or dn.shape[1:] != label.shape:
        raise ValueError(f"{scene.name}: bands {dn.shape} do not match the mask {label.shape}")
    return dn, label


def reflectance(scene: Scene, dn: np.ndarray) -> np.ndarray:
    """DN of one scene (8, H, W) or of tiles (n, 8, h, w) as float32 top-of-atmosphere
    reflectance; 0 where the product has no data (DN 0)."""
    shape = [1] * dn.ndim
    shape[dn.ndim - 3] = len(BANDS)  # the band axis: 0 for a scene, 1 for tiles
    sin_elevation = math.sin(math.radians(scene.sun_elevation))
    mult = (scene.per_band("REFLECTANCE_MULT") / sin_elevation).astype(np.float32)
    add = (scene.per_band("REFLECTANCE_ADD") / sin_elevation).astype(np.float32)
    rho = dn.astype(np.float32) * mult.reshape(shape) + add.reshape(shape)
    rho[dn == 0] = 0.0
    return rho


def descriptors(scene: Scene) -> np.ndarray:
    """Landsat 8's chain for this scene, one 15-number descriptor per band (8, 15)."""
    sensor = spec.load("landsat8_oli")
    sin_elevation = math.sin(math.radians(scene.sun_elevation))
    rows = []
    for b in BANDS:
        band = sensor.band(b)
        k = scene.numbers[("REFLECTANCE_MULT", b)]
        k /= scene.numbers[("RADIANCE_MULT", b)] * sin_elevation
        noise = band.noise_model(SHOT_FRACTION)
        saturation = band.saturation_radiance
        top = k * saturation.value if saturation.known else None
        rows.append(
            desc.describe(
                band.lower_nm,
                band.upper_nm,
                band.gsd_m.value,
                GRID_M,
                mtf_nyquist=None,
                noise_ab=None if noise is None else (k * noise[0], k * k * noise[1]),
                step=None if top is None else top / LEVELS,
                top=top,
            )
        )
    return np.stack(rows)


class Windows:
    """Training windows of size x size pixels from some scenes, each at least half labelled.

    draw(rng) gives (scene, DN (8, size, size) uint16, label (size, size) uint8). A scene is
    drawn by its area, then a window anywhere in it; a window less than half labelled is
    drawn again. In a scene without shadow labels, clear becomes NOT_CLOUD. cards holds each
    scene's descriptors (descriptors()), worked out once."""

    def __init__(self, chosen: list[Scene], size: int = 80, tries: int = 50) -> None:
        if not chosen:
            raise ValueError("no scenes to draw windows from")
        self.scenes, self.size, self.tries = list(chosen), size, tries
        self.shapes = []
        for scene in self.scenes:
            with rasterio.open(scene.mask) as src:
                if src.height < size or src.width < size:
                    raise ValueError(f"{scene.name}: smaller than a window of {size} px")
                self.shapes.append((src.height, src.width))
        area = np.array([h * w for h, w in self.shapes], np.float64)
        self.p = area / area.sum()
        self.cards = {scene.name: descriptors(scene) for scene in self.scenes}

    def __len__(self) -> int:
        return len(self.scenes)

    def draw(self, rng: np.random.Generator) -> tuple[Scene, np.ndarray, np.ndarray]:
        for _ in range(self.tries):
            k = int(rng.choice(len(self.scenes), p=self.p))
            scene, (height, width) = self.scenes[k], self.shapes[k]
            row = int(rng.integers(0, height - self.size + 1))
            col = int(rng.integers(0, width - self.size + 1))
            window = Window(col, row, self.size, self.size)
            with rasterio.open(scene.mask) as src:
                label = src.read(1, window=window)
            if (label != NO_LABEL).mean() < 0.5:
                continue
            with rasterio.open(scene.bands) as src:
                dn = src.read(window=window)
            if not scene.has_shadow:
                label = np.where(label == 0, NOT_CLOUD, label).astype(np.uint8)
            return scene, dn, label
        raise RuntimeError(f"no window of {self.size} px half labelled in {self.tries} tries")
