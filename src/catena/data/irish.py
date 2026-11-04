"""Landsat 7 ETM+ scenes of L7 Irish, as scripts/irish_fetch.sh unpacked TorchGeo's copy:
reflectance, labels, and what the model is told about each band. No network trains on it.

    from catena.data import irish
    for scene in irish.scenes(Path("/root/catena_data/chips/l7_irish"), "pilot"):
        dn, label = irish.read(scene)          # (6, H, W) uint8 and (H, W) uint8
        rho = irish.reflectance(scene, dn)     # float32, 0 where the product has no data
        d = irish.descriptors(scene)           # (6, 15): this scene's chain

    python -m catena.data.irish ROOT           writes ROOT/index.csv (every scene) and
                                               ROOT/pilot.csv (3 scenes per zone)

Layout: ROOT/<zone>/l7irish/<zone>/p<path>_r<row>/ holds L71<path><row>_<row><date>.TIF (9
bands, uint8, in the order B1-B5, B61, B62, B7, B8), its _MTL.txt and
L7_p<path>_r<row>_newmask2015.TIF. The 6 reflective 30 m bands are used: B1-B5 and B7.

Masks: 0 fill, 64 shadow, 128 clear, 192 thin, 255 thick, as in Biome, read into Catena's
codes (0 clear, 1 thin, 2 thick, 3 shadow, 255 no label). A mask without any shadow is taken
as one whose shadows were not labelled (has_shadow False): there clear may hide shadow.

Reflectance, from the pre-collection MTL (Landsat 7 handbook):
  L = (LMAX - LMIN) / (QCALMAX - QCALMIN) x (DN - QCALMIN) + LMIN       radiance
  rho = pi L d^2 / (ESUN sin(sun elevation))                            ESUN: Chander 2009
with d the Earth-Sun distance on the acquisition date. What the model is told: the band
edges and 30 m (landsat7_etm.yaml); the top, LMAX of the band's gain as reflectance,
k LMAX with k = pi d^2 / (ESUN sin(sun elevation)); the step, one of the levels between LMIN
and LMAX, k (LMAX - LMIN) / (QCALMAX - QCALMIN), by the rule every camera follows
(catena.descriptor). Noise and blur are in none of the sources read, so they stay unknown.
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

from catena import descriptor as desc
from catena.sensors import spec

BANDS = ("B1", "B2", "B3", "B4", "B5", "B7")
STACK = (1, 2, 3, 4, 5, 8)  # where they sit in the 9-band image, counted from 1
GRID_M = 30.0
NO_LABEL = 255
CODES = np.full(256, NO_LABEL, np.uint8)  # the masks' codes to Catena's
CODES[[128, 192, 255, 64]] = [0, 1, 2, 3]
PILOT_CLOUD = (0.1, 0.4, 0.7)  # pilot.csv: per zone, the scenes nearest these cloud shares
COLUMNS = (
    "scene",
    "zone",
    "folder",
    "has_shadow",
    "cloud",
    "labelled",
    "gains",
    "gain_change",
    "matched",
)


def mtl_values(text: str) -> dict[str, str]:
    """The KEY = VALUE pairs of an MTL file."""
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def first(mtl: dict, *keys: str) -> str:
    """The value of the first key the MTL has: older and newer MTL files name things apart."""
    for key in keys:
        if key in mtl:
            return mtl[key]
    raise KeyError(keys[0])


def earth_sun_distance(day: date) -> float:
    """In astronomical units, from the day of the year; within 0.0002 of the ephemeris."""
    doy = day.timetuple().tm_yday
    return 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))


@dataclass(frozen=True)
class Scene:
    name: str  # p<path>_r<row>
    zone: str
    folder: Path
    image: Path
    mask: Path
    has_shadow: bool
    sun_elevation: float  # degrees
    distance_au: float
    lmax: tuple  # per band of BANDS, W m-2 sr-1 um-1
    lmin: tuple
    qcalmax: tuple
    qcalmin: tuple
    gains: str  # the gain of each band of BANDS, "H" or "L"
    gain_change: bool  # a band switched gain within the scene

    @property
    def biome(self) -> str:
        """The group a scene's scores are pooled in (Biome's biome; here the zone)."""
        return self.zone

    def per_band(self, name: str) -> np.ndarray:
        return np.array(getattr(self, name), np.float64)


def open_scene(folder: Path, zone: str, has_shadow: bool = False) -> Scene:
    """A scene folder: its image, mask and MTL, and the numbers the reflectance needs."""
    images = [p for p in folder.glob("L71*.TIF") if "_B" not in p.name]
    masks = list(folder.glob("*mask*.TIF"))
    mtls = list(folder.glob("*_MTL.txt"))
    if len(images) != 1 or len(masks) != 1 or len(mtls) != 1:
        raise ValueError(f"{folder}: want one image, one mask and one MTL file")
    mtl = mtl_values(mtls[0].read_text())
    numbers = {"lmax": [], "lmin": [], "qcalmax": [], "qcalmin": []}
    gains, change = "", False
    for b in BANDS:
        n = b[1:]
        numbers["lmax"].append(float(first(mtl, f"LMAX_BAND{n}", f"RADIANCE_MAXIMUM_BAND_{n}")))
        numbers["lmin"].append(float(first(mtl, f"LMIN_BAND{n}", f"RADIANCE_MINIMUM_BAND_{n}")))
        numbers["qcalmax"].append(
            float(first(mtl, f"QCALMAX_BAND{n}", f"QUANTIZE_CAL_MAX_BAND_{n}"))
        )
        numbers["qcalmin"].append(
            float(first(mtl, f"QCALMIN_BAND{n}", f"QUANTIZE_CAL_MIN_BAND_{n}"))
        )
        gains += first(mtl, f"BAND{n}_GAIN", f"GAIN_BAND_{n}")
        change |= mtl.get(f"BAND{n}_GAIN_CHANGE", mtl.get(f"GAIN_CHANGE_BAND_{n}", "0")) != "0"
    day = date.fromisoformat(first(mtl, "ACQUISITION_DATE", "DATE_ACQUIRED"))
    return Scene(
        name=folder.name,
        zone=zone,
        folder=folder,
        image=images[0],
        mask=masks[0],
        has_shadow=has_shadow,
        sun_elevation=float(mtl["SUN_ELEVATION"]),
        distance_au=earth_sun_distance(day),
        gains=gains,
        gain_change=change,
        **{k: tuple(v) for k, v in numbers.items()},
    )


def folders(root: Path) -> list[tuple[str, Path]]:
    """(zone, scene folder) of every scene under ROOT, in a fixed order."""
    found = []
    for zone in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for folder in sorted((root / zone).glob("**/p*_r*")):
            if folder.is_dir() and any(folder.glob("L71*.TIF")):
                found.append((zone, folder))
    return found


def scenes(root: Path, table: str = "pilot") -> list[Scene]:
    """The scenes listed in ROOT/<table>.csv ("pilot": 3 per zone, "index": all)."""
    with open(root / f"{table}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no scenes in {root / f'{table}.csv'}")
    return [open_scene(root / r["folder"], r["zone"], r["has_shadow"] == "True") for r in rows]


def read(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """The scene's DN (6, H, W) uint8 and its label (H, W) in Catena's codes."""
    with rasterio.open(scene.image) as src:
        dn = src.read(list(STACK))
    with rasterio.open(scene.mask) as src:
        label = CODES[src.read(1)]
    if dn.shape[1:] != label.shape:
        raise ValueError(f"{scene.name}: bands {dn.shape} do not match the mask {label.shape}")
    return dn, label


@functools.cache
def esun() -> tuple:
    """ESUN of each band of BANDS, W m-2 um-1 (landsat7_etm.yaml), read once."""
    sensor = spec.load("landsat7_etm")
    return tuple(sensor.band(b).e0.value for b in BANDS)


def per_radiance(scene: Scene) -> np.ndarray:
    """k per band: the reflectance of one radiance unit in this scene."""
    sin_elevation = math.sin(math.radians(scene.sun_elevation))
    return math.pi * scene.distance_au**2 / (np.array(esun()) * sin_elevation)


def reflectance(scene: Scene, dn: np.ndarray) -> np.ndarray:
    """DN of one scene (6, H, W) or of tiles (n, 6, h, w) as float32 top-of-atmosphere
    reflectance; 0 where the product has no data (DN 0)."""
    shape = [1] * dn.ndim
    shape[dn.ndim - 3] = len(BANDS)  # the band axis: 0 for a scene, 1 for tiles
    k = per_radiance(scene)
    lmax, lmin = scene.per_band("lmax"), scene.per_band("lmin")
    qmax, qmin = scene.per_band("qcalmax"), scene.per_band("qcalmin")
    gain = (lmax - lmin) / (qmax - qmin)
    mult = (k * gain).astype(np.float32).reshape(shape)
    add = (k * (lmin - gain * qmin)).astype(np.float32).reshape(shape)
    rho = dn.astype(np.float32) * mult + add
    rho[dn == 0] = 0.0
    return rho


def descriptors(scene: Scene) -> np.ndarray:
    """Landsat 7's chain for this scene, one 15-number descriptor per band (6, 15)."""
    sensor = spec.load("landsat7_etm")
    k = per_radiance(scene)
    lmax, lmin = scene.per_band("lmax"), scene.per_band("lmin")
    levels = scene.per_band("qcalmax") - scene.per_band("qcalmin")
    rows = []
    for i, b in enumerate(BANDS):
        band = sensor.band(b)
        rows.append(
            desc.describe(
                band.lower_nm,
                band.upper_nm,
                band.gsd_m.value,
                GRID_M,
                mtf_nyquist=None,
                noise_ab=None,
                step=k[i] * (lmax[i] - lmin[i]) / levels[i],
                top=k[i] * lmax[i],
            )
        )
    return np.stack(rows)


def survey(scene: Scene) -> dict:
    """One row of index.csv: the scene's mask, summed up, and whether it fits the image."""
    with rasterio.open(scene.image) as src:
        size = (src.height, src.width)
    with rasterio.open(scene.mask) as src:
        label = CODES[src.read(1)]
    labelled = int((label != NO_LABEL).sum())
    cloud = int(((label == 1) | (label == 2)).sum())
    return {
        "scene": scene.name,
        "zone": scene.zone,
        "has_shadow": bool((label == 3).any()),
        "cloud": round(cloud / max(labelled, 1), 4),
        "labelled": round(labelled / label.size, 4),
        "gains": scene.gains,
        "gain_change": scene.gain_change,
        "matched": size == label.shape,
    }


def pilot(rows: list[dict]) -> list[dict]:
    """Per zone, the scenes nearest 10%, 40% and 70% cloud, each once; scenes whose gain
    changed within the scene, or whose mask does not fit the image, are left out."""
    chosen = []
    for zone in dict.fromkeys(r["zone"] for r in rows):
        left = [r for r in rows if r["zone"] == zone and r["matched"] and not r["gain_change"]]
        for target in PILOT_CLOUD:
            if left:
                best = min(left, key=lambda r, t=target: abs(r["cloud"] - t))
                chosen.append(best)
                left.remove(best)
    return chosen


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="index.csv and pilot.csv of the L7 Irish scenes")
    p.add_argument("root", type=Path, help="the folder scripts/irish_fetch.sh unpacked into")
    args = p.parse_args(argv)
    rows = []
    for zone, folder in folders(args.root):
        scene = open_scene(folder, zone)
        row = survey(scene) | {"folder": str(folder.relative_to(args.root))}
        rows.append(row)
        print(
            f"  {zone:<20} {scene.name:<10} cloud {row['cloud']:5.1%}  labelled "
            f"{row['labelled']:5.1%}  shadow {'yes' if row['has_shadow'] else 'no ':<3}  "
            f"gains {scene.gains}{'  gain change' if scene.gain_change else ''}"
            f"{'' if row['matched'] else '  MASK DOES NOT FIT'}",
            flush=True,
        )
    chosen = pilot(rows)
    for table, listed in (("index", rows), ("pilot", chosen)):
        with open(args.root / f"{table}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows({k: r[k] for k in COLUMNS} for r in listed)
    zones = len({r["zone"] for r in rows})
    print(
        f"\n{len(rows)} scenes in {zones} zones; {sum(r['has_shadow'] for r in rows)} with "
        f"shadow labels; {sum(r['gain_change'] for r in rows)} with a gain change; "
        f"{sum(not r['matched'] for r in rows)} whose mask does not fit; "
        f"{len(chosen)} in pilot.csv ({sum(r['has_shadow'] for r in chosen)} with shadow labels)"
    )


if __name__ == "__main__":
    main()
