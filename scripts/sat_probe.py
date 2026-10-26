"""Each band's top value in the data, and what the product metadata says about scaling.

    python scripts/sat_probe.py PRODUCTS [--first N]

For each product zip in the PRODUCTS folder: the raster:bands entries of its STAC JSON
(scale, offset, nodata, bits), then for every image band the highest value and the share of
valid pixels sitting exactly at it. A band that saturates shows a pile-up: a clear share of
pixels at one top value, in bright cloud. 4095 is the top of 12 bits, 16383 of 14 bits.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio

KEYS = ("scale", "offset", "nodata", "bits_per_sample", "data_type")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("products", type=Path, help="folder of product zips")
    parser.add_argument("--first", type=int, default=0, help="only the first N products")
    args = parser.parse_args()
    zips = sorted(args.products.resolve().glob("*.zip"))
    for z in zips[: args.first or None]:
        print(f"\n{z.stem}")
        with zipfile.ZipFile(z) as archive:
            names = archive.namelist()
            for member in (n for n in names if n.endswith(".json")):
                for key, asset in json.loads(archive.read(member)).get("assets", {}).items():
                    for b in asset.get("raster:bands", []):
                        words = ", ".join(f"{k} {b[k]}" for k in KEYS if k in b)
                        print(f"   metadata {key}: {words}")
        for member in (n for n in names if n.endswith(".tif") and "AUX_DATA" not in n):
            with rasterio.open(f"/vsizip/{z}/{member}") as src:
                data = src.read()
            for i, band in enumerate(data, start=1):
                valid = band[band > 0]
                if valid.size == 0:
                    continue
                top = int(valid.max())
                at_top = int((valid == top).sum())
                share = at_top / valid.size
                p999 = int(np.percentile(valid, 99.9))
                flag = "   <- pile-up: saturation?" if at_top >= 100 and share > 1e-4 else ""
                print(f"   {Path(member).stem[-3:]} band {i}: max {top}, "
                      f"at max {share:.3%}, 99.9% {p999}{flag}")


if __name__ == "__main__":
    main()
