"""Phi-sat-2 L1 products: what each one holds, and its bands as top-of-atmosphere reflectance.

    python -m catena.data.phisat2 PRODUCTS

A product is a zip named PHISAT-2_L1_<session>_<start>_<stop>_<id>.zip, its start and stop
times in UTC (YYYYMMDDhhmmss). Inside, bands/scene_0_BC_multiband.tiff holds 8 bands as
uint16 on a 4096 x 4096 grid in longitude and latitude (EPSG:4326):

    index  1    2     3      4    5         6         7         8
    band   pan  blue  green  red  rededge1  rededge2  rededge3  nir   (Band_0 ... Band_7)

The values look like top-of-atmosphere radiance L in W m-2 sr-1 um-1, in steps of 1 (see
the phisat2 spec). Reflectance is rho = pi * L * d^2 / (E0 * cos(sun zenith)), with E0 from
the spec, d the Earth-Sun distance in AU, and the sun's position computed here for the
scene's centre and start time, because the products do not carry it.

The command prints, for every product, its scene name, centre, time, sun elevation and pixel
size on the ground (east-west x north-south), then per band the median reflectance, what one
DN is worth in reflectance (the digitising step), the top (the largest value, as reflectance)
and the share of pixels sitting at that top. A share marked * is a pile-up: the band
saturated, as it does over bright snow and cloud.
"""

from __future__ import annotations

import argparse
import math
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio

from catena.sensors import spec

NAME = re.compile(r"^PHISAT-2_L1_(\d+)_(\d{14})_(\d{14})_([0-9A-Fa-f]+)$")
MULTIBAND = re.compile(r"(^|/)bands/scene_0_BC_multiband\.tiff?$")
BANDS = ("pan", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir")
PILE_UP = 5e-4  # more than this share of pixels at the top value: the band saturated


@dataclass(frozen=True)
class Product:
    path: Path
    session: int
    start: datetime  # UTC
    member: str  # the multiband file inside the zip

    @property
    def scene(self) -> str:
        """Satellite and start time, as for the IRIDE scenes: PHISAT2_20261025T171555."""
        return f"PHISAT2_{self.start:%Y%m%dT%H%M%S}"

    @property
    def gdal_path(self) -> str:
        return f"/vsizip/{self.path.resolve()}/{self.member}"


def is_product(path: Path) -> bool:
    return path.suffix.lower() == ".zip" and NAME.match(path.stem) is not None


def open_product(path: Path) -> Product:
    """The product in a zip, after checking its name and its one multiband file."""
    found = NAME.match(path.stem)
    if found is None:
        raise ValueError(f"{path.name} is not named like a Phi-sat-2 L1 product")
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if MULTIBAND.search(n)]
    if len(members) != 1:
        raise ValueError(f"{path.name}: {len(members)} BC multiband files, expected 1")
    start = datetime.strptime(found[2], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return Product(path, int(found[1]), start, members[0])


def sun_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """The sun's zenith angle in degrees and the Earth-Sun distance in AU, at a time (UTC)
    and a place (degrees, north and east positive). NOAA's solar calculator, after Meeus:
    good to about 0.01 degree between 1800 and 2100, without atmospheric refraction."""
    t = (when.timestamp() / 86400 + 2440587.5 - 2451545.0) / 36525  # centuries from J2000
    mean_long = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
    anomaly = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    centre = (
        math.sin(anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * anomaly) * 0.000289
    )
    distance = 1.000001018 * (1 - ecc**2) / (1 + ecc * math.cos(anomaly + math.radians(centre)))
    omega = math.radians(125.04 - 1934.136 * t)
    apparent = math.radians(mean_long + centre - 0.00569 - 0.00478 * math.sin(omega))
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = math.radians(23 + (26 + seconds / 60) / 60 + 0.00256 * math.cos(omega))
    declination = math.asin(math.sin(obliquity) * math.sin(apparent))
    y, l0 = math.tan(obliquity / 2) ** 2, math.radians(mean_long)
    equation_of_time = 4 * math.degrees(  # minutes
        y * math.sin(2 * l0)
        - 2 * ecc * math.sin(anomaly)
        + 4 * ecc * y * math.sin(anomaly) * math.cos(2 * l0)
        - 0.5 * y**2 * math.sin(4 * l0)
        - 1.25 * ecc**2 * math.sin(2 * anomaly)
    )
    minutes = when.hour * 60 + when.minute + when.second / 60 + when.microsecond / 6e7
    hour_angle = math.radians(((minutes + equation_of_time + 4 * lon) % 1440) / 4 - 180)
    phi = math.radians(lat)
    cos_zenith = math.sin(phi) * math.sin(declination) + math.cos(phi) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith)))), distance


def pixel_metres(xres: float, yres: float, lat: float) -> tuple[float, float]:
    """A pixel's size on the ground in metres, east-west and north-south, from its size in
    degrees at a latitude: the length of a degree on the WGS84 ellipsoid, as a series."""
    phi = math.radians(lat)
    per_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)
    per_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    return xres * per_lon, yres * per_lat


def describe(item: Product) -> dict:
    """Centre, sun, pixel size and, per band, the reflectance of one radiance unit."""
    with rasterio.open(item.gdal_path) as src:
        if src.count != len(BANDS) or src.crs is None or not src.crs.is_geographic:
            raise ValueError(f"{item.path.name}: expected 8 bands in longitude and latitude")
        left, bottom, right, top = src.bounds
        xres, yres = src.res  # degrees
    lat, lon = (bottom + top) / 2, (left + right) / 2
    zenith, distance = sun_position(item.start, lat, lon)
    if zenith >= 90:
        raise ValueError(f"{item.scene}: the sun is below the horizon at {lat:.2f}, {lon:.2f}")
    sensor = spec.load("phisat2")
    e0 = np.array([sensor.band(f"Band_{k}").e0.value for k in range(len(BANDS))])
    factor = math.pi * distance**2 / (e0 * math.cos(math.radians(zenith)))
    return {
        "scene": item.scene,
        "lat": lat,
        "lon": lon,
        "sun_zenith": zenith,
        "distance": distance,
        "pixel_m": pixel_metres(xres, yres, lat),  # east-west, north-south
        "scale": float(sensor.product["scale"]),
        "offset": float(sensor.product["offset"]),
        "factor": factor,  # reflectance per unit of radiance, one per band
    }


def grid(item: Product) -> dict:
    """The product's own pixel grid."""
    with rasterio.open(item.gdal_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def reflectance(item: Product, indexes: Sequence[int] = range(1, 9), info: dict | None = None):
    """Bands (1-based indexes, as above) as float32 top-of-atmosphere reflectance; no-data
    pixels (0) stay 0."""
    info = info or describe(item)
    indexes = list(indexes)
    with rasterio.open(item.gdal_path) as src:
        dn = src.read(indexes)
    factor = info["factor"][np.array(indexes) - 1][:, None, None]
    rho = (dn.astype(np.float64) * info["scale"] + info["offset"]) * factor
    rho[dn == 0] = 0.0
    return rho.astype(np.float32)


def summary(item: Product) -> list[str]:
    """The lines the command prints for one product."""
    info = describe(item)
    with rasterio.open(item.gdal_path) as src:
        dn = src.read()
    lat, lon = info["lat"], info["lon"]
    east, north = info["pixel_m"]
    head = (
        f"{item.scene}  {abs(lat):.2f}{'N' if lat >= 0 else 'S'} {abs(lon):.2f}"
        f"{'E' if lon >= 0 else 'W'}  {item.start:%Y-%m-%d %H:%M} UTC  "
        f"sun {90 - info['sun_zenith']:.1f} deg above the horizon  "
        f"pixel {east:.2f} x {north:.2f} m"
    )
    rows = {"median": [], "1 DN": [], "top": [], "at top %": []}
    saturated = []
    for k, band in enumerate(BANDS):
        valid = dn[k][dn[k] > 0]
        if valid.size == 0:
            for row in rows.values():
                row.append("-")
            continue

        def rho(value: float, k: int = k) -> float:
            return (value * info["scale"] + info["offset"]) * info["factor"][k]

        top = int(valid.max())
        share = float(np.mean(valid == top))
        pile = share > PILE_UP and int(np.sum(valid == top)) >= 50
        saturated += [band] if pile else []
        rows["median"].append(f"{rho(float(np.median(valid))):.3f}")
        rows["1 DN"].append(f"{info['scale'] * info['factor'][k]:.4f}")
        rows["top"].append(f"{rho(top):.3f}")
        rows["at top %"].append(f"{100 * share:.2f}" + ("*" if pile else ""))
    lines = [head, " " * 12 + "".join(f"{b:>9}" for b in BANDS)]
    lines += [f"   {name:<9}" + "".join(f"{v:>9}" for v in values) for name, values in rows.items()]
    lines.append(f"   saturated: {', '.join(saturated) if saturated else 'none'}")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("products", type=Path, help="folder of Phi-sat-2 L1 product zips")
    args = parser.parse_args(argv)
    items = [open_product(p) for p in sorted(args.products.glob("*.zip")) if is_product(p)]
    if not items:
        raise SystemExit(f"no Phi-sat-2 L1 products in {args.products}")
    print(f"{len(items)} products in {args.products}\n")
    for item in items:
        print("\n".join(summary(item)) + "\n", flush=True)


if __name__ == "__main__":
    main()
