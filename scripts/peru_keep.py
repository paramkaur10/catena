"""Keep one patch from each CloudPeru2 group of 8; the other 7 are its rotations and flips.

    python scripts/peru_keep.py CloudPeru2.zip peru_check.csv OUT

peru_check.csv comes from peru_check.py. For every group the lowest-numbered patch is kept:
its image and mask are copied byte for byte out of the zip into OUT/images and OUT/masks.
OUT/index.csv gets one row per kept patch (number, group, no-data and cloud shares, split:
5% of groups go to validation, chosen with a fixed seed) and OUT/PROVENANCE.txt records the
zip's size and md5 and what was kept. The zip is not touched: delete it yourself once the
counts check out.
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    zip_path, check_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    with open(check_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["group"])].append(row)
    keep = [min(members, key=lambda r: int(r["n"])) for _, members in sorted(groups.items())]
    random.seed(0)
    validation = set(random.sample(sorted(groups), k=round(0.05 * len(groups))))

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for i, row in enumerate(keep, start=1):
            n = row["n"]
            (out / "images" / f"{n}.tif").write_bytes(archive.read(f"PatchesI/{n}.tif"))
            (out / "masks" / f"{n}_Mask.tif").write_bytes(archive.read(f"PatchesM/{n}_Mask.tif"))
            if i % 500 == 0:
                print(f"   copied {i}")

    fields = ["n", "group", "split", "nodata_frac", "cloud_frac"]
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in keep:
            split = "validation" if int(row["group"]) in validation else "train"
            writer.writerow(row | {"split": split})

    print("md5 of the zip (a minute for 14 GB) ...")
    total = sum(p.stat().st_size for p in out.rglob("*.tif"))
    lines = [
        f"source: {zip_path.resolve()}",
        f"size: {zip_path.stat().st_size} bytes",
        f"md5: {md5_of(zip_path)}",
        f"kept: {len(keep)} of {len(rows)} patches, the lowest number of each group of 8",
        f"groups: {len(groups)}; validation groups: {len(validation)}",
        f"copied: {2 * len(keep)} files, {total / 1e9:.2f} GB",
        "labels as delivered: 255 cloud, 0 not cloud; a pixel with all bands 0 has no data",
    ]
    (out / "PROVENANCE.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    images = len(list((out / "images").glob("*.tif")))
    masks = len(list((out / "masks").glob("*.tif")))
    print(f"check: {images} images and {masks} masks in {out}; expected {len(keep)} of each")


if __name__ == "__main__":
    main()
