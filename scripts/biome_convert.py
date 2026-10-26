"""Landsat 8 Biome: each scene's tarball into two compact files, then (optionally) delete it.

    python scripts/biome_convert.py RAW OUT [--limit N] [--delete]

RAW holds the 96 tarballs; OUT gets one folder per scene with
    <scene>.bands.tif   the 8 reflective bands B1-B7 and B9 as delivered (DN, uint16), 30 m,
                        deflate-compressed; B8 (pan), B10, B11 (thermal) and BQA are left out
    <scene>.mask.tif    the hand-made mask in Catena's codes: 0 clear, 1 thin, 2 thick,
                        3 shadow, 255 no data
    <scene>_MTL.txt     the metadata: reflectance = (DN x MULT + ADD) / sin(sun elevation)
    <scene>.json        what the index is built from
plus OUT/index.csv (one row per scene) and OUT/pilot.csv (3 scenes per biome, the ones
closest to 10, 40 and 70% cloud). Shadows were drawn only where they were easy to see (32
of the 96 scenes); has_shadow in the index says which. In the other scenes clear may hide
shadow, so a loader reads their clear as code 5 (clear or shadow). A scene whose .json
exists is skipped, so the script can be stopped and re-run. With --delete a tarball is
removed only after its outputs were written and re-read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"]
CODES = {128: 0, 192: 1, 255: 2, 64: 3, 0: 255}  # Biome: clear, thin, thick, shadow, fill
BIOMES = {  # from the USGS Biome page, 12 scenes each
    "Barren": "LC80420082013220LGN00 LC80500092014231LGN00 LC80530022014156LGN00 "
    "LC81330312013202LGN00 LC81360302014162LGN00 LC81390292014135LGN00 LC81550082014263LGN00 "
    "LC81570452014213LGN00 LC81640502013179LGN01 LC81750432013144LGN00 LC81930452013126LGN01 "
    "LC81990402014267LGN00",
    "Forest": "LC80070662014234LGN00 LC80160502014041LGN00 LC80200462014005LGN00 "
    "LC80500172014247LGN00 LC81170272014189LGN00 LC81310182013108LGN01 LC81330182013186LGN00 "
    "LC81720192013331LGN00 LC81750622013304LGN00 LC81800662014230LGN00 LC82290572014141LGN00 "
    "LC82310592014139LGN00",
    "Grass/Crops": "LC80290292014132LGN00 LC80290372013257LGN00 LC80980712014024LGN00 "
    "LC81220312014208LGN00 LC81220422014096LGN00 LC81320352013243LGN00 LC81440462014250LGN00 "
    "LC81490432014141LGN00 LC81510262014139LGN00 LC81750512013208LGN00 LC81820302014180LGN00 "
    "LC82020522013141LGN01",
    "Shrubland": "LC80010732013109LGN00 LC80320382013278LGN00 LC80350192014190LGN00 "
    "LC80630152013207LGN00 LC80670172014206LGN00 LC80750172013163LGN00 LC80760182013170LGN00 "
    "LC80980762014216LGN00 LC81020802014100LGN00 LC81490122013218LGN00 LC81590362014051LGN00 "
    "LC81600462013215LGN00",
    "Snow/Ice": "LC80010112014080LGN00 LC80060102014147LGN00 LC80211222013361LGN00 "
    "LC80250022014232LGN00 LC80441162013330LGN00 LC80841202014309LGN00 LC81001082014022LGN00 "
    "LC81321192014054LGN00 LC82001192013335LGN00 LC82171112014297LGN00 LC82271192014287LGN00 "
    "LC82320072014226LGN00",
    "Urban": "LC80150312014226LGN00 LC80170312013157LGN00 LC80410372013357LGN00 "
    "LC80460282014171LGN00 LC80640452014041LGN00 LC81180382014244LGN00 LC81620432014072LGN00 "
    "LC81660432014020LGN00 LC81770262013254LGN00 LC81920192013103LGN01 LC81940222013245LGN00 "
    "LC81970242013218LGN00",
    "Water": "LC80120552013202LGN00 LC80180082014215LGN00 LC80210072014236LGN00 "
    "LC80430122014214LGN00 LC80650182013237LGN00 LC81040622014066LGN00 LC81130632014241LGN00 "
    "LC81240462014238LGN00 LC81620582014104LGN00 LC81660032014196LGN00 LC81910182013240LGN00 "
    "LC82150712013152LGN00",
    "Wetlands": "LC80310202013223LGN00 LC80340192014167LGN00 LC81010142014189LGN00 "
    "LC81020152014036LGN00 LC81030162014107LGN00 LC81070152013260LGN00 LC81080162013171LGN00 "
    "LC81080182014238LGN00 LC81460162014168LGN00 LC81500152013225LGN00 LC81580172013201LGN00 "
    "LC81750732014035LGN00",
}
PILOT_CLOUD = (0.1, 0.4, 0.7)  # pilot.csv takes, per biome, the scenes closest to these
BIOME_OF = {scene: biome for biome, scenes in BIOMES.items() for scene in scenes.split()}
INDEX = ["scene", "biome", "date", "path", "row", "sun_elevation", "cloud_cover_mtl", "height",
         "width", "valid", "clear", "thin", "thick", "shadow", "has_shadow", "mb"]


def read_envi(img: Path, hdr: Path) -> np.ndarray:
    """A single-band uint8 ENVI image, read with numpy (the mask's own format)."""
    header = hdr.read_text()

    def value(key: str, default: str | None = None) -> str:
        found = re.search(rf"^\s*{re.escape(key)}\s*=\s*(\S+)", header, re.M | re.I)
        if found is None and default is None:
            raise SystemExit(f"{hdr.name}: no '{key}' in the header")
        return found.group(1) if found else default

    if value("data type") != "1" or value("bands", "1") != "1":
        raise SystemExit(f"{hdr.name}: expected one band of uint8 (data type 1)")
    samples, lines = int(value("samples")), int(value("lines"))
    data = np.fromfile(img, dtype=np.uint8, offset=int(value("header offset", "0")))
    if data.size != samples * lines:
        raise SystemExit(f"{img.name}: {data.size} bytes for {lines} x {samples} pixels")
    return data.reshape(lines, samples)


def mtl_values(text: str) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def convert(tar_path: Path, out: Path, delete: bool) -> dict | None:
    scene = tar_path.name.split(".")[0]
    if scene not in BIOME_OF:
        raise SystemExit(f"{scene} is not in the Biome list; stopping")
    folder = out / scene
    if (folder / f"{scene}.json").exists():  # converted before: only the deletion may be left
        if delete and (folder / f"{scene}.mask.tif").exists():
            tar_path.unlink()
            print(f"{scene}  converted before; tarball deleted")
        return None
    start = time.time()
    names = {f"{scene}_{b}.TIF" for b in BANDS}
    names |= {f"{scene}_fixedmask.img", f"{scene}_fixedmask.hdr", f"{scene}_MTL.txt"}
    tmp = Path(tempfile.mkdtemp(dir=out, prefix=".tmp_"))
    try:
        with tarfile.open(tar_path, "r|gz") as tar:  # one pass through the stream
            for member in tar:
                name = Path(member.name).name
                if member.isfile() and name in names:
                    with tar.extractfile(member) as src, open(tmp / name, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 24)
        missing = sorted(n for n in names if not (tmp / n).exists())
        if missing:
            raise SystemExit(f"{scene}: missing in the tarball: {missing}")

        arrays, profile = [], None
        for b in BANDS:
            with rasterio.open(tmp / f"{scene}_{b}.TIF") as src:
                arrays.append(src.read(1))
                profile = profile or dict(src.profile)
        bands = np.stack(arrays)
        raw = read_envi(tmp / f"{scene}_fixedmask.img", tmp / f"{scene}_fixedmask.hdr")
        if raw.shape != bands.shape[1:]:
            raise SystemExit(f"{scene}: mask {raw.shape} but bands {bands.shape[1:]}")
        unknown = sorted(set(np.unique(raw).tolist()) - set(CODES))
        if unknown:
            raise SystemExit(f"{scene}: unexpected mask values {unknown}")
        lut = np.full(256, 255, np.uint8)
        for k, v in CODES.items():
            lut[k] = v
        mask = lut[raw]

        folder.mkdir(parents=True, exist_ok=True)
        common = {"driver": "GTiff", "crs": profile["crs"], "transform": profile["transform"],
                  "height": bands.shape[1], "width": bands.shape[2], "tiled": True,
                  "blockxsize": 512, "blockysize": 512, "compress": "deflate",
                  "BIGTIFF": "IF_SAFER", "NUM_THREADS": "ALL_CPUS"}
        band_file, mask_file = folder / f"{scene}.bands.tif", folder / f"{scene}.mask.tif"
        with rasterio.open(band_file, "w", **common, count=len(BANDS), dtype="uint16",
                           nodata=0, predictor=2) as dst:
            dst.write(bands)
            for i, b in enumerate(BANDS, start=1):
                dst.set_band_description(i, b)
        with rasterio.open(mask_file, "w", **common, count=1, dtype="uint8", nodata=255) as dst:
            dst.write(mask[None])
        shutil.copy(tmp / f"{scene}_MTL.txt", folder / f"{scene}_MTL.txt")

        with rasterio.open(band_file) as check:  # read back before anything is deleted
            if check.count != len(BANDS) or check.read(4).shape != bands.shape[1:]:
                raise SystemExit(f"{scene}: the written bands do not read back")
        with rasterio.open(mask_file) as check:
            if not np.array_equal(check.read(1), mask):
                raise SystemExit(f"{scene}: the written mask does not read back")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    mtl = mtl_values((folder / f"{scene}_MTL.txt").read_text())
    counts = np.bincount(mask.ravel(), minlength=256)
    valid = max(int(mask.size - counts[255]), 1)
    mb = sum(p.stat().st_size for p in folder.iterdir()) / 1e6
    info = {
        "scene": scene, "biome": BIOME_OF[scene], "date": mtl.get("DATE_ACQUIRED", ""),
        "path": mtl.get("WRS_PATH", ""), "row": mtl.get("WRS_ROW", ""),
        "sun_elevation": float(mtl.get("SUN_ELEVATION", "nan")),
        "cloud_cover_mtl": float(mtl.get("CLOUD_COVER", "nan")),
        "height": bands.shape[1], "width": bands.shape[2],
        "valid": round(valid / mask.size, 4),
        **{k: round(int(counts[c]) / valid, 4)
           for k, c in (("clear", 0), ("thin", 1), ("thick", 2), ("shadow", 3))},
        "has_shadow": bool(counts[3] > 0), "mb": round(mb, 1),
        "reflectance_mult": {b: mtl.get(f"REFLECTANCE_MULT_BAND_{b[1:]}") for b in BANDS},
        "reflectance_add": {b: mtl.get(f"REFLECTANCE_ADD_BAND_{b[1:]}") for b in BANDS},
    }
    (folder / f"{scene}.json").write_text(json.dumps(info, indent=1))
    if delete:
        tar_path.unlink()
    print(f"{scene}  {info['biome']:11s} cloud {info['thin'] + info['thick']:6.1%}  "
          f"shadow {info['shadow']:5.1%}  {mb:6.0f} MB  {time.time() - start:4.0f} s"
          f"{'  tarball deleted' if delete else ''}")
    return info


def write_tables(out: Path) -> None:
    infos = [json.loads(p.read_text()) for p in sorted(out.glob("*/*.json"))]
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(infos)
    pilot = []
    for biome in BIOMES:
        scenes = [i for i in infos if i["biome"] == biome]
        if len(scenes) == 12:  # little, some and much cloud; never all clear or all cloud
            for target in PILOT_CLOUD:
                left = [i for i in scenes if i not in pilot]
                pilot.append(min(left, key=lambda i, t=target: abs(i["thin"] + i["thick"] - t)))
    with open(out / "pilot.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pilot)
    shadows = sum(i["has_shadow"] for i in pilot)
    print(f"\n{len(infos)} scenes in {out / 'index.csv'}; {len(pilot)} in pilot.csv "
          f"({shadows} with shadow labels); "
          f"{sum(not i['has_shadow'] for i in infos)} scenes have no shadow label at all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="convert at most N scenes")
    parser.add_argument("--delete", action="store_true", help="delete each tarball once done")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tarballs = sorted(args.raw.glob("*.tar.gz"))
    unknown = [t.name for t in tarballs if t.name.split(".")[0] not in BIOME_OF]
    if unknown:
        raise SystemExit(f"not in the Biome list, nothing done: {unknown}")
    done = 0
    for tar_path in tarballs:
        if convert(tar_path, args.out, args.delete) is not None:
            done += 1
            if args.limit and done >= args.limit:
                break
    write_tables(args.out)


if __name__ == "__main__":
    main()
