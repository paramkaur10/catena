"""Phi-sat-2 L1 products: names, the sun's position, pixel sizes, and radiance turned into
reflectance.

The products here are made by the tests in the layout of the real ones: a zip named
PHISAT-2_L1_<session>_<start>_<stop>_<id>.zip holding bands/scene_0_BC_multiband.tiff,
8 bands of uint16 on a grid in longitude and latitude.
"""

from __future__ import annotations

import math
import zipfile
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena.data import phisat2 as P  # noqa: E402

NAME = "PHISAT-2_L1_000008357_20261025110655_20261025110658_284FD4D8"
SIZE = 64  # pixels a side
XRES, YRES = 4.5e-5, 3.0e-5  # degrees per pixel, east-west and north-south
WEST, NORTH = 9.0, 46.5  # the top-left corner


def product(folder, name=NAME, member="bands/scene_0_BC_multiband.tiff"):
    """A product zip: band k is 100 + 10 k plus a gentle slope, except a corner with no
    data and a square of blue at its top value, 253."""
    rows, cols = np.mgrid[:SIZE, :SIZE]
    data = np.zeros((8, SIZE, SIZE), np.uint16)
    for k in range(8):
        data[k] = 100 + 10 * k + (rows + cols) // 4
    data[:, :4, :4] = 0
    data[1, 20:40, 20:40] = 253
    tif = folder / "multiband.tif"
    profile = {
        "driver": "GTiff",
        "count": 8,
        "height": SIZE,
        "width": SIZE,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(WEST, NORTH, XRES, YRES),
        "nodata": 0,
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data)
    path = folder / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.write(tif, member)
        z.writestr("session_8357_metadata.json", "{}")
    return path


def noon(day: datetime, lat: float, lon: float = 0.0) -> tuple[float, datetime]:
    """The smallest zenith angle of a day, and when, in steps of 15 s."""
    times = [day + timedelta(seconds=15 * k) for k in range(24 * 60 * 4)]
    return min((P.sun_position(t, lat, lon)[0], t) for t in times)


def test_names_and_scenes(tmp_path):
    path = product(tmp_path)
    item = P.open_product(path)
    assert P.is_product(path) and not P.is_product(tmp_path / "IMH02_1CST__OPT8.zip")
    assert item.session == 8357 and item.start == datetime(2026, 10, 25, 11, 6, 55, tzinfo=UTC)
    assert item.scene == "PHISAT2_20261025T110655"
    assert item.gdal_path.endswith(".zip/bands/scene_0_BC_multiband.tiff")
    with pytest.raises(ValueError, match="not named like"):
        P.open_product(tmp_path / "PHISAT-2_L2_1_2_3_4.zip")
    other = product(tmp_path, name=NAME.replace("8357", "8358"), member="bands/other.tiff")
    with pytest.raises(ValueError, match="0 BC multiband"):
        P.open_product(other)


def test_the_sun_at_the_solstices_and_equinox():
    """At noon the zenith angle is the latitude minus the sun's declination: +23.44 degrees
    at the June solstice, -23.44 in December, 0 at the March equinox."""
    june, december = datetime(2026, 6, 21, tzinfo=UTC), datetime(2026, 12, 21, tzinfo=UTC)
    assert noon(june, 45)[0] == pytest.approx(45 - 23.44, abs=0.05)
    assert noon(june, 0)[0] == pytest.approx(23.44, abs=0.05)
    assert noon(december, 45)[0] == pytest.approx(45 + 23.44, abs=0.05)
    assert noon(datetime(2026, 3, 20, tzinfo=UTC), 0)[0] < 0.5


def test_noon_moves_with_the_equation_of_time_and_the_longitude():
    """The sun runs 16 minutes fast in early November and 14 minutes slow in mid-February;
    each degree of longitude east brings noon 4 minutes earlier."""
    november = noon(datetime(2026, 11, 3, tzinfo=UTC), 45)[1]
    february = noon(datetime(2026, 2, 11, tzinfo=UTC), 45)[1]
    assert abs(november - datetime(2026, 11, 3, 11, 43, 36, tzinfo=UTC)) < timedelta(minutes=1)
    assert abs(february - datetime(2026, 2, 11, 12, 14, 12, tzinfo=UTC)) < timedelta(minutes=1)
    east = noon(datetime(2026, 11, 3, tzinfo=UTC), 45, lon=15)[1]
    assert abs(november - east - timedelta(minutes=60)) < timedelta(minutes=1)


def test_earth_sun_distance_at_perihelion_and_aphelion():
    assert P.sun_position(datetime(2026, 1, 3, 17, tzinfo=UTC), 0, 0)[1] == pytest.approx(
        0.98330, abs=3e-4
    )
    assert P.sun_position(datetime(2026, 7, 6, 17, tzinfo=UTC), 0, 0)[1] == pytest.approx(
        1.01664, abs=3e-4
    )


def test_a_degree_on_the_ground():
    """The length of one degree on the WGS84 ellipsoid, east-west and north-south, from the
    exact radii of curvature: 111319.5 and 110574.3 m at the equator, 78846.8 and 111131.8 m
    at 45 degrees, 55800.0 and 111412.3 m at 60 degrees."""
    table = [(0, 111319.5, 110574.3), (45, 78846.8, 111131.8), (60, 55800.0, 111412.3)]
    for lat, east, north in table:
        assert P.pixel_metres(1.0, 1.0, lat) == pytest.approx((east, north), abs=1.0)
        assert P.pixel_metres(1.0, 1.0, -lat) == pytest.approx((east, north), abs=1.0)
    assert P.pixel_metres(2e-5, 1e-5, 0) == pytest.approx((2.2264, 1.10574), abs=1e-4)


def test_reflectance_is_pi_l_d2_over_e0_cos(tmp_path):
    item = P.open_product(product(tmp_path))
    info = P.describe(item)
    lat, lon = NORTH - SIZE * YRES / 2, WEST + SIZE * XRES / 2
    assert (info["lat"], info["lon"]) == pytest.approx((lat, lon))
    zenith, d = P.sun_position(item.start, lat, lon)
    assert info["sun_zenith"] == pytest.approx(zenith)
    assert info["pixel_m"] == pytest.approx(P.pixel_metres(XRES, YRES, lat))
    rho = P.reflectance(item, (4, 3, 8), info)  # red, green, NIR
    assert rho.dtype == np.float32 and rho.shape == (3, SIZE, SIZE)
    for row, (index, e0) in enumerate([(4, 1512.06), (3, 1823.24), (8, 1041.63)]):
        radiance = 100 + 10 * (index - 1) + (50 + 50) // 4
        expected = math.pi * radiance * d**2 / (e0 * math.cos(math.radians(zenith)))
        assert rho[row, 50, 50] == pytest.approx(expected, rel=1e-5)
    assert (rho[:, :4, :4] == 0).all()  # no data stays no data
    shifted = P.reflectance(item, (4,), {**info, "offset": 5.0})  # also with an offset
    assert (shifted[:, :4, :4] == 0).all() and shifted[0, 50, 50] > rho[0, 50, 50]
    assert P.grid(item)["width"] == SIZE


def test_summary_marks_the_saturated_band(tmp_path):
    lines = P.summary(P.open_product(product(tmp_path)))
    assert lines[0].startswith("PHISAT2_20261025T110655  46.50N 9.00E  2026-10-25 11:06 UTC")
    assert lines[0].endswith("pixel 3.45 x 3.33 m")  # 4.5e-5 x 3e-5 degrees at 46.5 N
    at_top = next(line for line in lines if line.strip().startswith("at top %")).split()[3:]
    assert at_top[1].endswith("*") and not any(v.endswith("*") for v in at_top[:1] + at_top[2:])
    assert lines[-1].strip() == "saturated: blue"


def test_the_command_lists_every_product(tmp_path, capsys):
    product(tmp_path)
    (tmp_path / "notes.zip").write_bytes(b"")  # not a product: left alone
    P.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert out.startswith("1 products in") and "PHISAT2_20261025T110655" in out
