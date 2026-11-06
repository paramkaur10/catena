"""IRIDE Eaglet II scenes for testing: each product's three bands on the 5 m grid of the
scene's label, as top-of-atmosphere reflectance, and what the model is told about them.

    python -m catena.data.eaglet2 DATA [--read]

prints, for every product in DATA/raw/iride_eaglet2/products, the gain and the offset of each
band as its STAC item and its XML give them, and whether the two agree; with --read, also
each labelled scene's median and top reflectance per band (slow: it reads every image).

    from catena.data import eaglet2
    for scene in eaglet2.scenes(DATA):          # the labelled scenes (all test scenes)
        rho, label = eaglet2.read(scene)        # (3, H, W) float32, (H, W) uint8
    d = eaglet2.descriptors()                   # (3, 15)

The product zip holds one image of three bands, uint16: 1 red (about 665-835 nm, reaching
into the near infrared), 2 green, 3 blue. Reflectance = DN x gain + offset, with a gain and an
offset per band and per product, not clipped at 1; DN 0 is no data. The XML gives them in full
(GAIN and OFFSET, one per band in the band order), the STAC item rounded (assets, raster:bands:
scale and offset; the gains to two significant figures, 1.7e-05 for the XML's 1.7463e-05).
gains() reads both and stops unless every STAC value is the XML's rounded to the digits the
STAC item writes, so a swapped band or other units cannot pass unseen, and uses the XML's.
read() brings the bands onto the label's grid exactly as the drafts and the labels were made
(catena.drafts.on_grid: the mean of the valid pixels in each 5 m cell), then applies the gains;
a pixel where a band has no data gets no label. What the model is told: the band edges
(iride_eaglet2.yaml), 5 m pixels, top of atmosphere.
"""

from __future__ import annotations

import argparse
import csv
import decimal
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from catena import descriptor as desc
from catena import drafts
from catena.sensors import spec

BANDS = ("red", "green", "blue")  # bands 1-3 of the image
SPEC_BANDS = ("Band 1", "Band 2", "Band 3")  # their names in iride_eaglet2.yaml
GRID_M = 5.0  # the labels' grid
NO_LABEL = 255
DRAFT = "ocm4_nirblue"  # OmniCloudMask's draft the labels start from (blue as near infrared)
TAG = r"<(?:\w+:)?{0}\b[^>]*>\s*([-+0-9.eE]+)\s*</(?:\w+:)?{0}>"
GAIN, OFFSET = re.compile(TAG.format("GAIN")), re.compile(TAG.format("OFFSET"))


@dataclass(frozen=True)
class Scene:
    name: str
    role: str
    product: Path  # the product zip
    label: Path  # DATA/labels/iride_eaglet2/<scene>/label.tif

    def draft(self, name: str = DRAFT) -> Path:
        """One of OmniCloudMask's drafts of the scene, in its own codes."""
        return self.label.parent / f"{name}.tif"


def folders(data: Path) -> tuple[Path, Path]:
    """Eaglet II's labels and products under DATA."""
    data = Path(data)
    return data / "labels" / "iride_eaglet2", data / "raw" / "iride_eaglet2" / "products"


def scenes(data: Path, roles: tuple[str, ...] = ("test",)) -> list[Scene]:
    """The scenes of scenes.csv with a label.tif and a role among roles, in its order."""
    labels, products = folders(data)
    with open(labels / "scenes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = [
        Scene(
            r["scene"],
            r.get("role", ""),
            products / r["product"],
            labels / r["scene"] / "label.tif",
        )
        for r in rows
        if r.get("verdict") != "drop" and r.get("role", "") in roles
    ]
    chosen = [s for s in chosen if s.label.exists()]
    if not chosen:
        raise ValueError(f"no labelled Eaglet II scenes with role {' or '.join(roles)} in {data}")
    return chosen


def from_stac(item: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Scale and offset of the three bands from a STAC item, or None if it has no three."""
    found = []
    for asset in item.get("assets", {}).values():
        entries = asset.get("raster:bands") or asset.get("bands") or []
        scaled = [b for b in entries if "scale" in b or "raster:scale" in b]
        if len(scaled) == len(BANDS):
            found.append(scaled)
    if len(found) != 1:  # none, or several assets with three: not clear which is the image
        return None
    scale = [float(b.get("scale", b.get("raster:scale"))) for b in found[0]]
    offset = [float(b.get("offset", b.get("raster:offset", 0.0))) for b in found[0]]
    return np.array(scale), np.array(offset)


def from_xml(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """GAIN and OFFSET of the three bands from the product's XML, or None without three."""
    gain, offset = GAIN.findall(text), OFFSET.findall(text)
    if len(gain) != len(BANDS) or len(offset) != len(BANDS):
        return None
    return np.array(gain, np.float64), np.array(offset, np.float64)


def metadata(product: Path) -> dict:
    """What the product's STAC item and XML say about the gains: {"stac": (scale, offset) or
    None, "xml": (gain, offset) or None}."""
    with zipfile.ZipFile(product) as z:
        names = z.namelist()
        jsons = [n for n in names if n.lower().endswith(".json")]
        items = [n for n in jsons if Path(n).stem == product.stem] or jsons
        stac = None
        for name in items:
            try:
                stac = from_stac(json.loads(z.read(name)))
            except ValueError:
                continue
            if stac is not None:
                break
        text = "".join(z.read(n).decode("utf-8", "ignore") for n in names if n.endswith(".xml"))
    return {"stac": stac, "xml": from_xml(text)}


def rounded(short: float, full: float) -> bool:
    """short is full rounded to the digits short is written with: 1.7e-05 is 1.7463e-05 to
    two significant figures, 1.8e-05 is not."""
    if not (math.isfinite(short) and math.isfinite(full)):
        return False
    last = decimal.Decimal(repr(float(short))).as_tuple().exponent  # the last digit's place
    return abs(full - short) <= 0.5 * 10.0 ** int(last) * (1 + 1e-9) + 1e-15


def agree(stac, xml) -> bool:
    """Every gain and offset of the STAC item is the XML's, rounded to the STAC item's
    digits."""
    pairs = zip(np.concatenate(stac), np.concatenate(xml), strict=True)
    return all(rounded(float(s), float(x)) for s, x in pairs)


def gains(product: Path) -> tuple[np.ndarray, np.ndarray]:
    """Gain and offset of each band (3,), (3,): reflectance = DN x gain + offset. The XML's,
    checked against the STAC item's when both have them; the STAC item's without an XML."""
    found = metadata(product)
    stac, xml = found["stac"], found["xml"]
    if stac is None and xml is None:
        raise ValueError(f"{product.name}: neither the STAC item nor the XML gives three gains")
    if stac is not None and xml is not None and not agree(stac, xml):
        raise ValueError(
            f"{product.name}: the STAC item (scale {stac[0]}, offset {stac[1]}) and the XML "
            f"(GAIN {xml[0]}, OFFSET {xml[1]}) disagree"
        )
    return xml if xml is not None else stac


def read(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The scene's reflectance (3, H, W) float32 on its label's grid, 0 where a band has no
    data, and its label (H, W) uint8, with no label there."""
    grid = drafts.grid_of(scene.label)
    paths = drafts.band_paths(scene.product)
    mean = np.stack([drafts.on_grid(*paths[band], grid) for band in BANDS])
    gain, offset = gains(scene.product)
    valid = (mean > 0).all(axis=0)
    rho = mean * gain[:, None, None].astype(np.float32) + offset[:, None, None].astype(np.float32)
    rho = np.where(valid, rho, 0.0).astype(np.float32)
    with rasterio.open(scene.label) as src:
        label = src.read(1)
    if label.shape != rho.shape[1:]:
        raise ValueError(f"{scene.name}: bands {rho.shape} do not fit the label {label.shape}")
    return rho, np.where(valid, label, NO_LABEL).astype(np.uint8)


def descriptors() -> np.ndarray:
    """Eaglet II's bands as the model is told them (3, 15): band edges, 5 m, top of
    atmosphere; the chain unknown."""
    sensor = spec.load("iride_eaglet2")
    rows = []
    for name in SPEC_BANDS:
        band = sensor.band(name)
        rows.append(desc.describe(band.lower_nm, band.upper_nm, GRID_M, GRID_M))
    return np.stack(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path, help="the data folder (/root/catena_data)")
    p.add_argument("--read", action="store_true", help="also read the labelled scenes")
    args = p.parse_args(argv)
    _, products = folders(args.data)
    zips = sorted(products.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no product zips in {products}")

    def numbers(pair) -> str:
        if pair is None:
            return "-"
        return "  ".join(f"{g:.4e} {o:+.5f}" for g, o in zip(*pair, strict=True))

    print(f"{len(zips)} products; per band (red, green, blue): gain offset")
    agreed = 0
    for product in zips:
        found = metadata(product)
        try:
            gains(product)
            state = "ok" if None not in found.values() else "one source only"
        except ValueError as e:
            state = f"PROBLEM: {e}"
        if state == "ok":
            agreed += 1
            stac, xml = found["stac"][0], found["xml"][0]
            worst = float(np.max(np.abs(stac - xml) / np.abs(xml)))
            if worst > 1e-6:
                state += f" (the STAC item's gains rounded, by up to {worst:.1%})"
        print(f"\n{product.stem[:32]}  {state}")
        print(f"   STAC {numbers(found['stac'])}")
        print(f"   XML  {numbers(found['xml'])}")
    print(f"\n{agreed} of {len(zips)} products: the STAC item and the XML agree")
    if args.read:
        print("\nreflectance per band (red, green, blue): median, top")
        for scene in scenes(args.data):
            rho, label = read(scene)
            valid = label != NO_LABEL
            cells = [
                f"{np.median(rho[k][valid]):.3f} {rho[k][valid].max():.3f}"
                for k in range(len(BANDS))
            ]
            print(f"   {scene.name}  " + "   ".join(cells))


if __name__ == "__main__":
    main()
