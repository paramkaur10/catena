"""OmniCloudMask drafts for IRIDE and Phi-sat-2 products, written into one folder per scene.

    python scripts/ocm_prelabel.py PRODUCTS_FOLDER LABELS_FOLDER [--nir-from zeros|blue|red]
                                   [--res 5] [--device cuda:0] [--redo]

For HEO the drafts become training labels after review; for Eaglet II and Phi-sat-2 they are
first drafts of the test labels, which a person then corrects. For every product zip in
PRODUCTS_FOLDER:
  1. read red, green and NIR. HEO: B03, B02, B07. Eaglet II has no NIR band, so that input
     is filled as --nir-from says: zeros (as the OmniCloudMask documentation suggests) or a
     copy of the blue or red band (ECLIPSE used blue). Phi-sat-2: Band_3, Band_2 and
     Band_7 of the BC multiband file, as top-of-atmosphere reflectance (catena.data.phisat2);
  2. resample to --res metres (5 m is the finest OmniCloudMask supports), averaging only
     valid pixels, so swath edges do not darken. Phi-sat-2 stays on its own grid (about
     4.75 m, in longitude and latitude), so its draft fits its pixels one to one;
  3. predict with OmniCloudMask, model version 4.0 (pinned, for reproducibility). It
     rescales every tile band by band to zero mean and unit spread, so the units of the
     inputs (DN, radiance or reflectance) do not matter, only that 0 means no data;
  4. write into LABELS_FOLDER/<scene>/, where <scene> is the satellite and the start time
     of the acquisition (e.g. IMH02_20261009T133222, PHISAT2_20261025T171555):
       <draft>.tif       classes, OmniCloudMask's codes: 0 clear, 1 thick cloud,
                         2 thin cloud, 3 cloud shadow; 255 = no data
       <draft>.conf.tif  the 4 class probabilities x 250 as uint8; 255 = no data
       <draft>.png       true colour beside the same picture with the classes drawn on
     <draft> is "ocm4" for a product with a NIR band and "ocm4_nir<fill>" otherwise.
A scene that already has this draft is skipped unless --redo. LABELS_FOLDER/drafts.csv holds
one row per scene and draft (class shares, inputs, date). LABELS_FOLDER/scenes.csv gets a row
for every new scene; its review columns (role, verdict, use, labeller, date, notes) are for
people to fill in and are never overwritten. Every <draft>.tif carries its provenance in
GDAL tags. OmniCloudMask and PyTorch are installed separately (pip install omnicloudmask).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from catena.data import phisat2

MODEL_VERSION = 4.0
NAMES = ("clear", "thick", "thin", "shadow")
COLOURS = {1: (1.0, 0.1, 0.1), 2: (1.0, 0.85, 0.0), 3: (0.1, 0.4, 1.0)}  # thick, thin, shadow
STAMP = re.compile(r"^\d{8}T\d{6}$")
SCENE_COLUMNS = ["scene", "product", "satellite", "start", "lat", "lon", "sun_elevation"]
REVIEW_COLUMNS = ["role", "verdict", "use", "labeller", "date", "notes"]
DRAFT_COLUMNS = ["scene", "draft", "valid", *NAMES, "inputs", "made_utc"]


def scene_id(product: Path) -> str:
    """Satellite and start time: IMH02_1CST__OPT8_20261009T133222_... -> IMH02_20261009T133222."""
    if phisat2.is_product(product):
        return phisat2.open_product(product).scene
    tokens = [t for t in product.stem.split("_") if t]
    stamps = [t for t in tokens if STAMP.match(t)]
    return f"{tokens[0]}_{stamps[0]}" if stamps else product.stem


def band_paths(product: Path) -> tuple[dict[str, tuple[str, int]], str]:
    """GDAL paths (file, band index) of red, green, NIR and blue, and a note on the inputs."""
    names = zipfile.ZipFile(product).namelist()
    root = f"/vsizip/{product.resolve()}/"
    heo = {b: n for n in names for b in ("B01", "B02", "B03", "B07") if n.endswith(f".{b}.tif")}
    if len(heo) == 4:
        paths = {"red": "B03", "green": "B02", "nir": "B07", "blue": "B01"}
        return {k: (root + heo[b], 1) for k, b in paths.items()}, "HEO: red B03, green B02, NIR B07"
    tif = next(n for n in sorted(names) if n.lower().endswith((".tif", ".tiff")))
    paths = {"red": (root + tif, 1), "green": (root + tif, 2), "blue": (root + tif, 3)}
    return paths, "Eaglet II: red Band 1, green Band 2 (no NIR band)"


def target_grid(path: str, res: float) -> dict:
    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"{path}: grid is rotated; this script expects north-up images")
        width = math.ceil(src.width * abs(t.a) / res)
        height = math.ceil(src.height * abs(t.e) / res)
        transform = Affine(res, 0.0, t.c, 0.0, -res, t.f)
        return {"crs": src.crs, "transform": transform, "width": width, "height": height}


def resampled(path: str, band: int, grid: dict) -> np.ndarray:
    """One band on the target grid: the mean of the valid (non-zero) pixels in each cell."""
    with rasterio.open(path) as src:
        data = src.read(band)
        out = np.zeros((grid["height"], grid["width"]), np.float32)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=0,
            resampling=Resampling.average,
        )
    return out


def write(path: Path, data: np.ndarray, grid: dict, tags: dict | None = None) -> None:
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": data.shape[0],
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)


def quicklook(path: Path, rgb: np.ndarray, classes: np.ndarray, valid: np.ndarray) -> None:
    step = max(1, max(classes.shape) // 1000)
    rgb, classes, valid = rgb[:, ::step, ::step], classes[::step, ::step], valid[::step, ::step]
    image = np.moveaxis(rgb, 0, -1)
    lo, hi = np.percentile(image[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    image = np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)
    image[~valid] = 1.0
    drawn = image.copy()
    for code, colour in COLOURS.items():
        hit = classes == code
        drawn[hit] = 0.45 * drawn[hit] + 0.55 * np.array(colour)
    plt.imsave(path, np.concatenate([image, np.ones((image.shape[0], 8, 3)), drawn], axis=1))


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draft_one(product: Path, folder: Path, draft: str, args, predict) -> dict:
    """Run OmniCloudMask on one product and write its draft; returns its drafts.csv row."""
    if phisat2.is_product(product):
        item = phisat2.open_product(product)
        info = phisat2.describe(item)
        rho = phisat2.reflectance(item, (4, 3, 8, 2), info)  # red, green, NIR, blue
        bands = dict(zip(("red", "green", "nir", "blue"), rho, strict=True))
        grid = phisat2.grid(item)
        inputs = "Phi-sat-2: red Band_3, green Band_2, NIR Band_7, TOA reflectance"
        east, north = info["pixel_m"]
        resampling = f"none: the product's own grid, {east:.2f} x {north:.2f} m"
    else:
        paths, inputs = band_paths(product)
        grid = target_grid(paths["red"][0], args.res)
        bands = {k: resampled(p, i, grid) for k, (p, i) in paths.items()}
        resampling = f"mean of valid pixels to {args.res:g} m"
    nir = bands.get("nir")
    if nir is None:
        fill = args.nir_from
        nir = np.zeros_like(bands["red"]) if fill == "zeros" else bands[fill]
        inputs += f", NIR input: {fill}"
    stack = np.stack([bands["red"], bands["green"], nir])
    valid = (bands["red"] > 0) | (bands["green"] > 0)
    settings = {"no_data_value": 0, "inference_device": args.device, "batch_size": 4}
    settings["model_version"] = MODEL_VERSION
    classes = predict(stack, **settings)[0].astype(np.uint8)
    conf = predict(stack, export_confidence=True, **settings)
    classes[~valid] = 255
    conf8 = np.round(np.clip(conf, 0, 1) * 250).astype(np.uint8)
    conf8[:, ~valid] = 255
    made = datetime.now(UTC).isoformat(timespec="seconds")
    tags = {
        "labeller": f"omnicloudmask {version('omnicloudmask')}, model {MODEL_VERSION}",
        "torch": version("torch"),
        "inputs": inputs,
        "resampling": resampling,
        "classes": "0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow, 255 no data",
        "made_utc": made,
        "product": product.stem,
    }
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / f"{draft}.tif", classes, grid, tags)
    write(folder / f"{draft}.conf.tif", conf8, grid)
    rgb = np.stack([bands["red"], bands["green"], bands["blue"]])
    quicklook(folder / f"{draft}.png", rgb, classes, valid)
    row = {"scene": folder.name, "draft": draft, "valid": round(float(valid.mean()), 3)}
    row |= {n: round(float(np.mean(classes[valid] == c)), 3) for c, n in enumerate(NAMES)}
    return row | {"inputs": inputs, "made_utc": made}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("products", type=Path, help="folder of product zips")
    parser.add_argument("labels", type=Path, help="labels folder of this dataset")
    parser.add_argument("--res", type=float, default=5.0, help="working resolution, metres")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nir-from",
        choices=("zeros", "blue", "red"),
        default="zeros",
        help="what fills the NIR input when a product has no NIR band",
    )
    parser.add_argument("--redo", action="store_true", help="redo drafts that already exist")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    products = sorted(args.products.glob("*.zip"))
    scenes: dict[str, Path] = {}
    for product in products:
        scene = scene_id(product)
        if scene in scenes:
            raise SystemExit(
                f"two products for scene {scene}: {scenes[scene].name}, {product.name}"
            )
        scenes[scene] = product
    catalog = {r["product"]: r for r in read_rows(args.products / "catalog.csv")}
    args.labels.mkdir(parents=True, exist_ok=True)
    drafts = read_rows(args.labels / "drafts.csv")
    register = read_rows(args.labels / "scenes.csv")
    known = {r["scene"] for r in register}

    predict = None
    made, skipped = [], 0
    for scene, product in scenes.items():
        has_nir = phisat2.is_product(product) or "nir" in band_paths(product)[0]
        draft = f"ocm{int(MODEL_VERSION)}" + ("" if has_nir else f"_nir{args.nir_from}")
        folder = args.labels / scene
        if (folder / f"{draft}.tif").exists() and not args.redo:
            skipped += 1
        else:
            if predict is None:
                from omnicloudmask import predict_from_array as predict  # loads PyTorch
            row = draft_one(product, folder, draft, args, predict)
            made.append(row)
            print(
                f"{scene}  {draft:<14} valid {row['valid']:.0%}  "
                + "  ".join(f"{n} {row[n]:.0%}" for n in NAMES),
                flush=True,
            )
        if scene not in known:
            info = catalog.get(product.stem, {})
            if phisat2.is_product(product):
                where = phisat2.describe(phisat2.open_product(product))
                info = {"lat": round(where["lat"], 5), "lon": round(where["lon"], 5)}
                info["view:sun_elevation"] = round(90 - where["sun_zenith"], 2)
            row = {"scene": scene, "product": product.name, "satellite": scene.split("_")[0]}
            row["start"] = scene.split("_")[-1]
            row |= {"lat": info.get("lat", ""), "lon": info.get("lon", "")}
            row["sun_elevation"] = info.get("view:sun_elevation", "")
            register.append(row | dict.fromkeys(REVIEW_COLUMNS, ""))
            known.add(scene)

    replaced = {(r["scene"], r["draft"]) for r in made}
    drafts = [r for r in drafts if (r["scene"], r["draft"]) not in replaced] + made
    drafts.sort(key=lambda r: (r["scene"], r["draft"]))
    register.sort(key=lambda r: r["scene"])
    write_rows(args.labels / "drafts.csv", drafts, DRAFT_COLUMNS)
    write_rows(args.labels / "scenes.csv", register, SCENE_COLUMNS + REVIEW_COLUMNS)
    print(f"{len(made)} drafts made, {skipped} already there; {len(register)} scenes listed")
    print(f"in {args.labels}: scenes.csv, drafts.csv and one folder per scene")


if __name__ == "__main__":
    main()
