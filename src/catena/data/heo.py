"""IRIDE HEO scenes for training and testing: each product's 8 bands on the 5 m grid of the
scene's label, as top-of-atmosphere reflectance, and what the model is told about them.

    python -m catena.data.heo DATA [--force]

reads DATA/labels/iride_heo/scenes.csv and, for every scene with a label.tif (from the
review notebook) that is not dropped, writes DATA/chips/iride_heo/<scene>.bands.tif: the
product's bands B00-B07 (pan, blue, green, red, red edges 1-3, near-infrared) on the grid of
the scene's label.tif, 5 m as the drafts were made, each pixel the mean of the valid product
pixels in it (B00-B03 and B07 are 2.5 m in the product, the red edges 5 m); uint16, 0 no
data. A scene already converted is skipped unless --force. The labels stay where the
notebook writes them, so a corrected label.tif counts at once, without converting again.

    from catena.data import heo
    for scene in heo.scenes(DATA, roles=("train",)):    # train, test or stress
        dn, label = heo.read(scene)                     # (8, H, W) uint16, (H, W) uint8
        rho = heo.reflectance(dn)                       # float32, DN / 4094, at most 1
    d = heo.descriptors()                               # (8, 15)

The product stores reflectance x 4094 (QUANTIFICATION_VALUE) up to 1; DN 4095 is detector
saturation or reflectance above 1, and is read as 1. What the model is told: the band edges
(iride_heo.yaml), 5 m pixels, top of atmosphere, one level of the 12 bits (1 / 4094) and the
top of what the product can store (1). The blur and the noise are only bounded and the
saturation radiance is not known, so they stay unknown; the pan band clips lower than 1 in
bright cloud (0.39-0.57), which no number here says.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from catena import descriptor as desc
from catena import viz
from catena.sensors import spec

BANDS = ("B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07")  # the order of bands.tif
GRID_M = 5.0  # the labels' grid
LEVELS = 4094  # reflectance x 4094 up to 1 (QUANTIFICATION_VALUE)
SATURATED = 4095  # detector saturation or reflectance above 1
NO_LABEL = 255


@dataclass(frozen=True)
class Scene:
    name: str
    role: str  # train, test or stress
    product: Path  # the product zip
    bands: Path  # DATA/chips/iride_heo/<scene>.bands.tif
    label: Path  # DATA/labels/iride_heo/<scene>/label.tif

    @property
    def biome(self) -> str:
        """The group a scene's scores are pooled in (Biome's biome; here its role)."""
        return self.role


def folders(data: Path) -> tuple[Path, Path, Path]:
    """HEO's labels, products and converted bands under DATA."""
    data = Path(data)
    return (
        data / "labels" / "iride_heo",
        data / "raw" / "iride_heo" / "products",
        data / "chips" / "iride_heo",
    )


def listed(data: Path) -> list[Scene]:
    """Every scene of scenes.csv that has a label.tif and is not dropped, in its order."""
    labels, products, chips = folders(data)
    with open(labels / "scenes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Scene(
            name=r["scene"],
            role=r.get("role", ""),
            product=products / r["product"],
            bands=chips / f"{r['scene']}.bands.tif",
            label=labels / r["scene"] / "label.tif",
        )
        for r in rows
        if r.get("verdict") != "drop" and (labels / r["scene"] / "label.tif").exists()
    ]


def scenes(data: Path, roles: tuple[str, ...] = ("train",)) -> list[Scene]:
    """The labelled scenes of some roles, after checking that each one has been converted."""
    chosen = [s for s in listed(data) if s.role in roles]
    if not chosen:
        raise ValueError(f"no labelled HEO scenes with role {' or '.join(roles)} in {data}")
    missing = [s.name for s in chosen if not s.bands.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} HEO scenes are not converted yet, e.g. {missing[0]}: "
            f"run python -m catena.data.heo {data}"
        )
    return chosen


def band_paths(product: Path) -> list[str]:
    """GDAL paths of the 8 band files in a product zip, in the order of BANDS."""
    with zipfile.ZipFile(product) as z:
        names = z.namelist()
    found = {b: n for n in names for b in BANDS if n.endswith(f".{b}.tif")}
    missing = [b for b in BANDS if b not in found]
    if missing:
        raise ValueError(f"{product.name}: no {', '.join(missing)} file in the zip")
    return [f"/vsizip/{product.resolve()}/{found[b]}" for b in BANDS]


def convert(scene: Scene, force: bool = False) -> bool:
    """Write the scene's bands.tif on the grid of its label.tif; False if it was there."""
    if scene.bands.exists() and not force:
        return False
    with rasterio.open(scene.label) as src:
        crs, transform, height, width = src.crs, src.transform, src.height, src.width
    stack = np.zeros((len(BANDS), height, width), np.uint16)
    for k, path in enumerate(band_paths(scene.product)):
        mean = np.zeros((height, width), np.float32)
        with rasterio.open(path) as src:
            reproject(
                source=src.read(1),
                destination=mean,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=0,
                resampling=Resampling.average,
            )
        stack[k] = np.round(np.clip(mean, 0, SATURATED)).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(BANDS),
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
    }
    if min(height, width) >= 256:  # tiles: a training window reads only the tiles it covers
        profile |= {"tiled": True, "blockxsize": 256, "blockysize": 256}
    scene.bands.parent.mkdir(parents=True, exist_ok=True)
    part = scene.bands.with_name(scene.bands.name + ".part")
    with rasterio.open(part, "w", **profile) as dst:
        dst.write(stack)
        dst.update_tags(bands=" ".join(BANDS), product=scene.product.name, grid="label.tif")
    part.replace(scene.bands)
    return True


def read(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The scene's DN (8, H, W) uint16 on the label's grid and its label (H, W) uint8."""
    with rasterio.open(scene.bands) as src:
        dn = src.read()
    with rasterio.open(scene.label) as src:
        label = src.read(1)
    if dn.shape != (len(BANDS), *label.shape):
        raise ValueError(f"{scene.name}: bands {dn.shape} do not fit the label {label.shape}")
    return dn, label


def reflectance(dn: np.ndarray) -> np.ndarray:
    """DN as float32 top-of-atmosphere reflectance: DN / 4094, saturation (4095) read as 1;
    0 (no data) stays 0."""
    return np.minimum(dn.astype(np.float32) / LEVELS, 1.0)


def descriptors() -> np.ndarray:
    """HEO's chain, one 15-number descriptor per band (8, 15), on the 5 m grid."""
    sensor = spec.load("iride_heo")
    rows = []
    for name in BANDS:
        band = sensor.band(name)
        rows.append(
            desc.describe(band.lower_nm, band.upper_nm, GRID_M, GRID_M, step=1.0 / LEVELS, top=1.0)
        )
    return np.stack(rows)


class Windows:
    """Training windows of size x size pixels from some scenes, each at least half labelled.

    draw(rng) gives (scene, DN (8, size, size) uint16, label (size, size) uint8): a scene by
    its area, then a window anywhere in it; a window less than half labelled is drawn again.
    A pixel without image data gets no label. card holds HEO's descriptors."""

    def __init__(self, chosen: list[Scene], size: int = 240, tries: int = 50) -> None:
        if not chosen:
            raise ValueError("no scenes to draw windows from")
        self.scenes, self.size, self.tries = list(chosen), size, tries
        self.shapes = []
        for scene in self.scenes:
            with rasterio.open(scene.label) as src:
                if src.height < size or src.width < size:
                    raise ValueError(f"{scene.name}: smaller than a window of {size} px")
                self.shapes.append((src.height, src.width))
        area = np.array([h * w for h, w in self.shapes], np.float64)
        self.p = area / area.sum()
        self.card = descriptors()

    def __len__(self) -> int:
        return len(self.scenes)

    def draw(self, rng: np.random.Generator) -> tuple[Scene, np.ndarray, np.ndarray]:
        for _ in range(self.tries):
            k = int(rng.choice(len(self.scenes), p=self.p))
            scene, (height, width) = self.scenes[k], self.shapes[k]
            row = int(rng.integers(0, height - self.size + 1))
            col = int(rng.integers(0, width - self.size + 1))
            window = Window(col, row, self.size, self.size)
            with rasterio.open(scene.label) as src:
                label = src.read(1, window=window)
            if (label != NO_LABEL).mean() < 0.5:
                continue
            with rasterio.open(scene.bands) as src:
                dn = src.read(window=window)
            label = np.where(dn.any(axis=0), label, NO_LABEL).astype(np.uint8)
            return scene, dn, label
        raise RuntimeError(f"no window of {self.size} px half labelled in {self.tries} tries")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HEO products onto their labels' 5 m grid")
    p.add_argument("data", type=Path, help="the data folder, e.g. /root/catena_data")
    p.add_argument("--force", action="store_true", help="convert scenes done before again")
    args = p.parse_args(argv)
    found = listed(args.data)
    if not found:
        raise SystemExit(f"no HEO scene with a label.tif in {folders(args.data)[0]}")
    for scene in found:
        made = convert(scene, args.force)
        dn, label = read(scene)
        valid = dn.any(axis=0)
        saturated = float((dn[:, valid] == SATURATED).any(axis=0).mean()) if valid.any() else 0.0
        print(
            f"  {scene.name}  {scene.role or '-':<6} {'converted' if made else 'was there':<9}  "
            f"{label.shape[0]} x {label.shape[1]} px  {viz.describe(label)}; "
            f"saturated {saturated:.2%}",
            flush=True,
        )
    roles = {}
    for scene in found:
        roles[scene.role or "-"] = roles.get(scene.role or "-", 0) + 1
    shown = ", ".join(f"{n} {role}" for role, n in roles.items())
    print(f"\n{len(found)} scenes in {folders(args.data)[2]}: {shown}")


if __name__ == "__main__":
    main()
