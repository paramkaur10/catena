"""Hand-labelled CloudSEN12+ patches, copied out of the TACO files byte for byte.

    python -m catena.data.cloudsen12 RAW_FOLDER OUT_FOLDER [--label-type high] [--check 40]

CloudSEN12+ (Aybar et al. 2024) comes as five TACO files holding 50,247 Sentinel-2 L1C
patches. Each patch is two GeoTIFFs: the image ("s2l1c", 13 bands, uint16) and its label
("target", uint8). For every patch of the chosen label type ("high" = drawn by hand), both
are copied out unchanged to OUT_FOLDER/<split>/<id>.s2l1c.tif and <id>.target.tif, and
each copy is read back and its CRC-32 compared with the source bytes. A copy already there
that matches is kept, so a stopped run simply starts again. OUT_FOLDER/index.csv gets one
row per patch: the dataset's own metadata, the two files with their CRCs, and the label
values counted.

The labels keep CloudSEN12's codes (0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow);
the training loader maps them to ours. A last check reads some images in full, for two
facts the loader needs:
  - border: pixels that are 0 in every band (the 509-pixel patches sit in 512-pixel
    frames), where they are and what the label calls them;
  - offset: from processing baseline 04.00 (25 January 2022) L1C values carry +1000,
    reflectance = (DN - 1000) / 10000, unless the data were harmonised. B10, the cirrus
    band, is nearly black wherever there is no high cloud, so its darkest values for each
    baseline show which is the case.

TACO v1, as tacoreader 0.6.5 reads it (tacoreader 2 no longer does): 2 bytes of magic
(b"WX" or b"#y"), then the offset and length of the footer as little-endian 8-byte
integers. The footer is a Parquet table, one row per sample: where its bytes are
("tortilla:offset", "tortilla:length") and its metadata. A sample is itself a container
like this, with offsets counted from its own first byte.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import time
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio

MAGIC = (b"WX", b"#y")
CHUNK = 1 << 24  # bytes per read while copying: 16 MB
ITEMS = ("s2l1c", "target")
CLASSES = {0: "clear", 1: "thick", 2: "thin", 3: "shadow"}  # CloudSEN12's label codes
B10 = 10  # position of B10 among B01-B08, B8A, B09, B10, B11, B12
NODATA = 65535  # the images' no-data value


# --- TACO v1 ------------------------------------------------------------------------------


def footer(path: Path, start: int = 0) -> list[dict]:
    """The rows of the table of the container that begins at byte `start` of `path`."""
    with open(path, "rb") as f:
        f.seek(start)
        head = f.read(18)
        if len(head) < 18 or head[:2] not in MAGIC:
            raise ValueError(f"{path} at byte {start}: not a TACO container")
        offset = int.from_bytes(head[2:10], "little") + start
        length = int.from_bytes(head[10:18], "little")
        f.seek(offset)
        return pq.read_table(pa.BufferReader(f.read(length))).to_pylist()


def items(path: Path, sample: dict) -> dict[str, tuple[int, int]]:
    """{item: (offset, length)} of one sample of a TACO file, offsets from the file start."""
    start = sample["tortilla:offset"]
    return {
        row["tortilla:id"]: (start + row["tortilla:offset"], row["tortilla:length"])
        for row in footer(path, start)
    }


# --- copying ------------------------------------------------------------------------------


def crc32(path: Path, offset: int = 0, length: int | None = None) -> int:
    """CRC-32 of `length` bytes of `path` from `offset`; to the end if `length` is None."""
    value, left = 0, length
    with open(path, "rb") as f:
        f.seek(offset)
        while left is None or left > 0:
            chunk = f.read(CHUNK if left is None else min(CHUNK, left))
            if not chunk:
                break
            value = zlib.crc32(chunk, value)
            if left is not None:
                left -= len(chunk)
    if left:
        raise OSError(f"{path} ends before byte {offset + length}")
    return value


def copy_bytes(src: Path, offset: int, length: int, target: Path) -> int:
    """Copy `length` bytes of `src` from `offset` to `target`; returns their CRC-32."""
    part = target.with_name(target.name + ".part")  # renamed only once complete
    value, left = 0, length
    with open(src, "rb") as f, open(part, "wb") as out:
        f.seek(offset)
        while left > 0:
            chunk = f.read(min(CHUNK, left))
            if not chunk:
                raise OSError(f"{src} ends before byte {offset + length}")
            value = zlib.crc32(chunk, value)
            out.write(chunk)
            left -= len(chunk)
    part.replace(target)
    return value


def copy_out(src: Path, offset: int, length: int, target: Path) -> tuple[int, str]:
    """Copy one file out of `src`, or keep an identical copy that is already there.
    Returns the CRC-32 of the bytes and "kept" or "copied"."""
    if target.exists() and target.stat().st_size == length:
        value = crc32(src, offset, length)
        if crc32(target) == value:
            return value, "kept"
    value = copy_bytes(src, offset, length, target)
    if crc32(target) != value:
        raise OSError(f"{target}: the copy, read back, differs from the source")
    return value, "copied"


# --- the index ----------------------------------------------------------------------------

POINT = re.compile(r"POINT \(\s*(\S+)\s+(\S+)\s*\)")
BASELINE = re.compile(r"_N(\d{2})(\d{2})_")


def baseline(product: str | None) -> str:
    """The processing baseline in a product name: "..._N0207_..." gives "02.07"."""
    found = BASELINE.search(product or "")
    return f"{found[1]}.{found[2]}" if found else ""


def lon_lat(point: str | None) -> tuple[float, float] | tuple[str, str]:
    """ "POINT (lon lat)" gives (lon, lat), rounded to 5 decimals (about 1 m)."""
    found = POINT.match(point or "")
    if not found:
        return "", ""
    return round(float(found[1]), 5), round(float(found[2]), 5)


def describe(sample: dict) -> dict:
    """The index columns that come from the dataset's own table."""
    lon, lat = lon_lat(sample.get("stac:centroid"))
    start = sample.get("stac:time_start")
    when = datetime.fromtimestamp(start, UTC).isoformat(timespec="seconds") if start else ""
    transform = sample.get("stac:geotransform") or [None, None]
    product = sample.get("s2_id") or ""
    return {
        "id": sample["tortilla:id"],
        "split": sample["tortilla:data_split"],
        "label_type": sample.get("label_type"),
        "real_proj_shape": sample.get("real_proj_shape"),
        "time_utc": when,
        "lon": lon,
        "lat": lat,
        "crs": sample.get("stac:crs"),
        "pixel_m": transform[1],
        "s2_id": product,
        "baseline": baseline(product),
        "sun_zenith": sample.get("s2_mean_solar_zenith_angle"),
        "sun_azimuth": sample.get("s2_mean_solar_azimuth_angle"),
        "elevation_m": sample.get("rai:ele"),
        "country": sample.get("rai:admin0"),
        "thick_pct": sample.get("thick_percentage"),
        "thin_pct": sample.get("thin_percentage"),
        "shadow_pct": sample.get("cloud_shadow_percentage"),
        "clear_pct": sample.get("clear_percentage"),
    }


def size_checked(path: Path, count: int, dtype: str) -> tuple[int, int]:
    """Width and height of a GeoTIFF, after checking its number of bands and data type."""
    with rasterio.open(path) as src:
        if src.count != count or src.dtypes[0] != dtype:
            found = f"{src.count} bands of {src.dtypes[0]}"
            raise ValueError(f"{path}: {found}, expected {count} of {dtype}")
        return src.width, src.height


def label_values(path: Path) -> Counter:
    """How many pixels of the label carry each value."""
    with rasterio.open(path) as src:
        values, counts = np.unique(src.read(1), return_counts=True)
    return Counter(dict(zip(values.tolist(), counts.tolist(), strict=True)))


def extract(raw: Path, out: Path, label_types: set[str]) -> tuple[list[dict], Counter]:
    """Copy out every patch of `label_types` and write OUT/index.csv.
    Returns the index rows and how many files were copied and kept."""
    tacos = sorted(raw.glob("*.taco"))
    if not tacos:
        raise SystemExit(f"no .taco files in {raw}")
    chosen = {t: [s for s in footer(t) if s.get("label_type") in label_types] for t in tacos}
    total = sum(len(samples) for samples in chosen.values())
    if not total:
        raise SystemExit(f"no patches of label type {sorted(label_types)}")
    need = sum(s["tortilla:length"] for samples in chosen.values() for s in samples)
    out.mkdir(parents=True, exist_ok=True)
    have = sum(p.stat().st_size for p in out.rglob("*.tif"))
    free = shutil.disk_usage(out).free
    if need - have > free - 2e9:  # keep 2 GB free for everything else
        raise SystemExit(f"need {(need - have) / 1e9:.1f} GB more, {free / 1e9:.1f} GB free")
    print(f"{total} patches, {need / 1e9:.1f} GB, to copy into {out}", flush=True)
    rows, done, began = [], Counter(), time.monotonic()
    for taco, samples in chosen.items():
        for sample in samples:
            row = describe(sample)
            found = items(taco, sample)
            if missing := [k for k in ITEMS if k not in found]:
                raise ValueError(f"{taco.name}, {row['id']}: no {', '.join(missing)}")
            folder = out / row["split"]
            folder.mkdir(exist_ok=True)
            for key in ITEMS:
                offset, length = found[key]
                target = folder / f"{row['id']}.{key}.tif"
                value, what = copy_out(taco, offset, length, target)
                done[what] += 1
                row[f"{key}_file"] = target.relative_to(out).as_posix()
                row[f"{key}_bytes"] = length
                row[f"{key}_crc32"] = f"{value:08x}"
            size = size_checked(out / row["s2l1c_file"], 13, "uint16")
            if size_checked(out / row["target_file"], 1, "uint8") != size:
                raise ValueError(f"{row['id']}: the image and its label differ in size")
            row["width"], row["height"] = size
            counts = label_values(out / row["target_file"])
            for code, name in CLASSES.items():
                row[f"px_{name}"] = counts.pop(code, 0)
            row["px_other"] = sum(counts.values())
            row["other_values"] = " ".join(f"{v}:{n}" for v, n in sorted(counts.items()))
            row["source"] = taco.name
            row["extra_items"] = " ".join(sorted(set(found) - set(ITEMS)))
            rows.append(row)
            if len(rows) % 500 == 0 or len(rows) == total:
                minutes = (time.monotonic() - began) / 60
                print(f"   {len(rows)} of {total} patches, {minutes:.1f} min", flush=True)
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, done


# --- the check ----------------------------------------------------------------------------


def edges(zero: np.ndarray) -> tuple[int, int, int, int]:
    """How many whole rows (top, bottom) and whole columns (left, right) of `zero` are True."""

    def leading(flags: np.ndarray) -> int:
        return len(flags) if flags.all() else int(np.argmin(flags))

    rows, cols = zero.all(axis=1), zero.all(axis=0)
    return leading(rows), leading(rows[::-1]), leading(cols), leading(cols[::-1])


def check(out: Path, rows: list[dict], per_baseline: int = 40, seed: int = 0) -> dict:
    """Read up to `per_baseline` images per processing baseline in full: their all-zero
    pixels, the label values there, any no-data values, and the darkest B10 values."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[row["baseline"]].append(row)
    report = {"patches": 0, "with_border": 0, "pixels": 0, "zero": 0, "nodata": 0}
    report |= {"edges": Counter(), "border_labels": Counter(), "dark_b10": {}}
    for name, members in sorted(groups.items()):
        darkest = []
        for row in rng.sample(members, min(per_baseline, len(members))):
            with rasterio.open(out / row["s2l1c_file"]) as src:
                image = src.read()
            with rasterio.open(out / row["target_file"]) as src:
                label = src.read(1)
            zero = ~image.any(axis=0)
            nodata = (image == NODATA).any(axis=0)
            report["patches"] += 1
            report["pixels"] += zero.size
            report["zero"] += int(zero.sum())
            report["nodata"] += int(nodata.sum())
            if zero.any():
                report["with_border"] += 1
                report["edges"][edges(zero)] += 1
                values, counts = np.unique(label[zero], return_counts=True)
                found = dict(zip(values.tolist(), counts.tolist(), strict=True))
                report["border_labels"].update(found)
            valid = ~zero & ~nodata
            if valid.any():
                darkest.append(float(np.percentile(image[B10][valid], 1)))
        report["dark_b10"][name] = (len(darkest), float(np.median(darkest)) if darkest else None)
    return report


def band_names(path: Path) -> str:
    with rasterio.open(path) as src:
        names = [d or "" for d in src.descriptions]
    return " ".join(names) if any(names) else "none stored"


def summary(rows: list[dict], done: Counter, report: dict | None, out: Path) -> None:
    print(f"\n{done['copied']} files copied, {done['kept']} already there and identical")
    print("\npatches per split, by size, GB:")
    for split in ("train", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        if part:
            sizes = Counter(f"{r['width']}x{r['height']}" for r in part)
            gb = sum(r["s2l1c_bytes"] + r["target_bytes"] for r in part) / 1e9
            print(f"   {split:<10} {len(part):>6}  {dict(sizes)}  {gb:.1f} GB")
    pixels, others = Counter(), Counter()
    for r in rows:
        for name in (*CLASSES.values(), "other"):
            pixels[name] += r[f"px_{name}"]
        for pair in r["other_values"].split():
            value, n = pair.split(":")
            others[int(value)] += int(n)
    total = sum(pixels.values())
    print("\nlabel pixels: " + ", ".join(f"{k} {v / total:.1%}" for k, v in pixels.items()))
    if others:
        print(f"   other label values: {dict(sorted(others.items()))}")
    print(f"\nband names stored in the images: {band_names(out / rows[0]['s2l1c_file'])}")
    baselines = Counter(r["baseline"] or "?" for r in rows)
    print(f"processing baselines: {dict(sorted(baselines.items()))}")
    if report:
        n, k = report["patches"], report["with_border"]
        share = report["zero"] / max(report["pixels"], 1)
        print(f"\n{n} images read in full; {k} have pixels that are 0 in every band")
        print(f"   ({share:.2%} of all pixels read)")
        if k:
            common = dict(report["edges"].most_common(5))
            print(f"   whole zero rows and columns (top, bottom, left, right): {common}")
            labels = dict(sorted(report["border_labels"].items()))
            print(f"   label values on those pixels: {labels}")
        print(f"   pixels with {NODATA} (no data) in some band: {report['nodata']}")
        print("   darkest B10 values (1st percentile per image, median over images):")
        for name, (count, value) in report["dark_b10"].items():
            shown = f"{value:.0f}" if value is not None else "-"
            print(f"      baseline {name or '?'}: {count} images, {shown}")
    print(f"\nindex written to {out / 'index.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="folder with the CloudSEN12+ .taco files")
    parser.add_argument("out", type=Path, help="folder the patches go to")
    parser.add_argument(
        "--label-type", nargs="+", default=["high"], help="high (default), scribble, nolabel"
    )
    parser.add_argument(
        "--check", type=int, default=40, help="images read in full per baseline (0: none)"
    )
    args = parser.parse_args(argv)
    rows, done = extract(args.raw, args.out, set(args.label_type))
    report = check(args.out, rows, args.check) if args.check else None
    summary(rows, done, report, args.out)


if __name__ == "__main__":
    main()
