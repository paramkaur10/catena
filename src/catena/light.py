"""How much light each real test scene had: the sun's angle and, where a camera's noise is
known, the SNR its card states at that angle. Low sun is where a real camera is noisiest, so
the real scenes sorted by light are the real-data test of what the SNR is worth.

    python -m catena.light DATA [--out light.csv]

writes one row per scene of the real test sets of catena.evaluate_clarity, named as it names
them (test, set, scene): Landsat 8 (B), Landsat 7 (C), HEO (D, test and stress), Eaglet II
and Phi-sat-2 (E). PeruSat-1's patches carry no time or place and are left out; Sentinel-2's
test windows carry their sun angle themselves (evaluate_clarity --per-window). Columns:
  sun_zenith   degrees from the vertical: 90 minus the sun's height above the horizon
  source       where it comes from: mtl (Landsat's metadata), stac (the product's STAC item,
               view:sun_elevation) or computed (catena.data.phisat2.sun_position, from the
               time and the centre: the time in the product's STAC item or, without one, in
               the scene's name, UTC; the centre of the scene's label)
  computed     for HEO and Eaglet II, the zenith computed from time and place beside a stated
               one: a check of both
  lat, lon     the scene's centre, degrees
  snr_dark     the SNR the scene's card states at reflectance 0.05 and 0.5 (the median over
  snr_bright   the bands whose noise is known), empty where no band's is
A set whose data is missing is said and left out; the rest go on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform

from catena import descriptor as desc
from catena.data import biome, eaglet2, heo, irish, phisat2

COLUMNS = ("test", "set", "scene", "sun_zenith", "source", "computed", "lat", "lon")
COLUMNS += ("snr_dark", "snr_bright")
NOISE = [desc.FIELDS.index("noise_dark"), desc.FIELDS.index("noise_bright")]
KNOWN_NOISE = desc.FIELDS.index("known_noise")
STAMP = re.compile(r"(\d{8}T\d{6})")


def centre(path: Path) -> tuple[float, float]:
    """The centre of a georeferenced raster: latitude and longitude, degrees."""
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    if crs is None:
        raise ValueError(f"{path}: not georeferenced")
    x, y = (left + right) / 2, (bottom + top) / 2
    if crs.is_geographic:
        return y, x
    lon, lat = transform(crs, "EPSG:4326", [x], [y])
    return float(lat[0]), float(lon[0])


def stamp(name: str) -> datetime | None:
    """The time in a scene's name (20260930T092923, UTC), or None."""
    found = STAMP.search(name)
    if found is None:
        return None
    return datetime.strptime(found.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def stac_item(product: Path) -> dict:
    """The properties of the product's STAC item ({} without a product or an item)."""
    if not product.exists():
        return {}
    with zipfile.ZipFile(product) as z:
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            try:
                item = json.loads(z.read(name))
            except ValueError:
                continue
            if isinstance(item, dict) and isinstance(item.get("properties"), dict):
                return item["properties"]
    return {}


def when_of(properties: dict, name: str) -> datetime | None:
    """The time of an acquisition: the STAC item's, else the one in the scene's name."""
    text = properties.get("datetime") or properties.get("start_datetime")
    if text:
        when = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return when if when.tzinfo else when.replace(tzinfo=UTC)
    return stamp(name)


def snr(d: np.ndarray) -> tuple[float | None, float | None]:
    """The SNR a card (bands, 15) states at reflectance 0.05 and 0.5: the median over the
    bands whose noise is known, or None."""
    known = d[:, KNOWN_NOISE] > 0
    if not known.any():
        return None, None
    sigma = 10.0 ** (d[known][:, NOISE].astype(np.float64) - 3)
    return float(np.median(desc.DARK / sigma[:, 0])), float(np.median(desc.BRIGHT / sigma[:, 1]))


def row(test, name, scene, zenith, source, lat, lon, computed=None, d=None) -> dict:
    """One row of the CSV, numbers to 4 decimals, an unknown one empty."""
    dark, bright = snr(d) if d is not None else (None, None)
    values = {"test": test, "set": name, "scene": scene, "sun_zenith": zenith, "source": source}
    values |= {"computed": computed, "lat": lat, "lon": lon}
    values |= {"snr_dark": dark, "snr_bright": bright}
    return {
        k: "" if v is None else (f"{float(v):.4f}" if isinstance(v, int | float) else v)
        for k, v in values.items()
    }


def landsat(test: str, name: str, module, root: Path, image) -> list[dict]:
    """A Landsat set's pilot scenes: the zenith from the MTL, the SNR from the card."""
    rows = []
    for scene in module.scenes(root, "pilot"):
        lat, lon = centre(image(scene))
        zenith = 90.0 - scene.sun_elevation
        d = module.descriptors(scene)
        rows.append(row(test, name, scene.name, zenith, "mtl", lat, lon, d=d))
    return rows


def landsat8(data: Path) -> list[dict]:
    return landsat("B", "landsat8", biome, data / "chips" / "l8_biome", lambda s: s.bands)


def landsat7(data: Path) -> list[dict]:
    return landsat("C", "landsat7", irish, data / "chips" / "l7_irish", lambda s: s.image)


def heo_scenes(data: Path) -> list[dict]:
    chosen = [("D", f"heo_{s.role}", s) for s in heo.listed(data) if s.role in ("test", "stress")]
    return iride(chosen)


def eaglet2_scenes(data: Path) -> list[dict]:
    return iride([("E", "eaglet2", s) for s in eaglet2.scenes(data)])


def iride(chosen: list) -> list[dict]:
    """HEO's or Eaglet II's scenes (test, set, scene): the zenith the product's STAC item
    states, or the one computed from time and place, with the computed one beside."""
    rows = []
    for test, name, scene in chosen:
        lat, lon = centre(scene.label)
        properties = stac_item(scene.product)
        when = when_of(properties, scene.name)
        computed = None if when is None else phisat2.sun_position(when, lat, lon)[0]
        elevation = properties.get("view:sun_elevation")
        if elevation is not None:
            zenith, source = 90.0 - float(elevation), "stac"
        else:
            zenith, source = computed, "computed"
        rows.append(row(test, name, scene.name, zenith, source, lat, lon, computed))
    return rows


def phisat(data: Path) -> list[dict]:
    """Phi-sat-2's scenes as evaluate_clarity reads them; the zenith as its loader works it
    out from the product's time and centre."""
    labels, products = data / "labels" / "phisat2", data / "raw" / "phisat2" / "products"
    with open(labels / "scenes.csv", newline="") as f:
        listed = [r for r in csv.DictReader(f) if r.get("verdict") != "drop"]
    rows = []
    for r in listed:
        if not (labels / r["scene"] / "label.tif").exists():
            continue
        info = phisat2.describe(phisat2.open_product(products / r["product"]))
        zenith = float(info["sun_zenith"])
        rows.append(row("E", "phisat2", r["scene"], zenith, "computed", info["lat"], info["lon"]))
    return rows


def scenes(data: Path) -> list[dict]:
    """Every real test scene's row; a set that cannot be read is said and left out."""
    rows = []
    for part in (landsat8, landsat7, heo_scenes, eaglet2_scenes, phisat):
        try:
            rows += part(Path(data))
        except Exception as e:  # one missing set must not cost the others
            print(f"  {part.__name__}: left out: {type(e).__name__}: {e}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("--out", type=Path, default=Path("light.csv"))
    args = p.parse_args(argv)
    rows = scenes(args.data)
    if not rows:
        raise SystemExit("no real test scene could be read")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'set':<12}{'scenes':>7}  sun zenith, degrees (lowest, median, highest)  SNR at 0.05")
    for name in dict.fromkeys(r["set"] for r in rows):
        some = [r for r in rows if r["set"] == name]
        z = [float(r["sun_zenith"]) for r in some if r["sun_zenith"]]
        dark = [float(r["snr_dark"]) for r in some if r["snr_dark"]]
        spread = f"{min(z):5.1f} {np.median(z):6.1f} {max(z):6.1f}" if z else "    -"
        told = f"{np.median(dark):.0f}" if dark else "-"
        print(f"{name:<12}{len(some):>7}  {spread:<46}{told}")
    if any(r["computed"] and r["source"] == "stac" for r in rows):
        gaps = [
            abs(float(r["sun_zenith"]) - float(r["computed"]))
            for r in rows
            if r["computed"] and r["source"] == "stac"
        ]
        print(f"stated against computed: up to {max(gaps):.1f} degrees apart")
    print(f"wrote {args.out} ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
