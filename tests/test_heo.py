"""IRIDE HEO: a made-up product brought onto its label's 5 m grid, the scenes of each role, the
reflectance, what the model is told, and training windows."""

from __future__ import annotations

import csv
import math
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena.data import heo  # noqa: E402

H, W = 32, 40  # the label's grid, 5 m
X0, Y0 = 500000.0, 4500000.0
CRS = "EPSG:32633"
TRAIN, TEST = "IMH12_20261008T090908", "IMH14_20261008T090717"


def raster(path, data, res, dtype, nodata=None):
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": dtype,
        "crs": CRS,
        "transform": from_origin(X0, Y0, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def product(path, tmp, scene):
    """A product zip: band k holds 1000 + 100 k, the last 5 m column has no data, and the
    pan band saturates in the first 5 m pixel and in half of the second."""
    with zipfile.ZipFile(path, "w") as z:
        for k, band in enumerate(heo.BANDS):
            res = 5.0 if band in ("B04", "B05", "B06") else 2.5
            f = int(5.0 / res)  # product pixels per 5 m pixel, across
            image = np.full((H * f, W * f), 1000 + 100 * k, np.uint16)
            image[:, -f:] = 0
            if band == "B00":
                image[:2, :2] = heo.SATURATED
                image[:2, 2] = heo.SATURATED  # one column of the second pixel's two
            raster(tmp / f"x.{band}.tif", image, res, "uint16")
            z.write(tmp / f"x.{band}.tif", f"{scene}/IMG_DATA/{scene}.{band}.tif")


@pytest.fixture
def data(tmp_path):
    """DATA with three HEO scenes: one to train on, one to test on, one dropped."""
    labels, products = tmp_path / "labels/iride_heo", tmp_path / "raw/iride_heo/products"
    products.mkdir(parents=True)
    rows = []
    for name, role, verdict in ((TRAIN, "train", "correct"), (TEST, "test", "correct")):
        (labels / name).mkdir(parents=True)
        label = np.zeros((H, W), np.uint8)
        label[:8], label[8:16] = 2, 1
        label[:, -1] = 255
        raster(labels / name / "label.tif", label, 5.0, "uint8", 255)
        zipped = products / f"{name[:5]}_1CST__OPT8_{name[6:]}_O_Z99.zip"
        product(zipped, tmp_path, name)
        rows.append({"scene": name, "product": zipped.name, "role": role, "verdict": verdict})
    rows.append({"scene": "IMH03_20261007T121948", "product": "x.zip", "role": "train"})
    rows[-1]["verdict"] = "drop"  # dropped, and without a label.tif
    with open(labels / "scenes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "product", "role", "verdict"])
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_converting_and_reading(data, capsys):
    with pytest.raises(FileNotFoundError, match=r"python -m catena\.data\.heo"):
        heo.scenes(data, ("train",))  # not converted yet
    heo.main([str(data)])
    assert "2 scenes in" in capsys.readouterr().out
    (scene,) = heo.scenes(data, ("train",))
    assert scene.name == TRAIN and scene.role == "train" and scene.biome == "train"
    assert [s.name for s in heo.scenes(data, ("test", "stress"))] == [TEST]
    dn, label = heo.read(scene)
    assert dn.shape == (8, H, W) and dn.dtype == np.uint16 and label.shape == (H, W)
    for k in range(8):  # each 5 m pixel the mean of the product pixels in it
        assert (dn[k, 4:, 1:-1] == 1000 + 100 * k).all()
    assert (dn[:, :, -1] == 0).all()  # no data stays 0
    assert dn[0, 0, 0] == heo.SATURATED
    assert abs(int(dn[0, 0, 1]) - (heo.SATURATED + 1000) / 2) <= 1  # half saturated
    rho = heo.reflectance(dn)
    assert rho.dtype == np.float32 and rho[0, 0, 0] == 1.0 and (rho[:, :, -1] == 0).all()
    assert rho[3, 10, 10] == pytest.approx(1300 / 4094)
    before = scene.bands.stat().st_mtime_ns
    heo.main([str(data)])
    assert scene.bands.stat().st_mtime_ns == before  # done before: skipped
    assert "was there" in capsys.readouterr().out


def test_what_heo_is_told():
    d = heo.descriptors()
    assert d.shape == (8, desc.N_FIELDS)
    assert d[0, 0] == pytest.approx(desc.wavelength_code(499.3))  # pan from 499.3 nm
    assert d[7, 2] == pytest.approx(desc.wavelength_code(891.5))  # NIR to 891.5 nm
    card = desc.card(d)
    assert (card[:, 3] == 0).all()  # top of atmosphere
    assert (card[:, 4] == 0).all()  # every band on the 5 m grid
    assert card[:, 5] == pytest.approx(math.log2(0.5))  # 5 m pixels
    assert (d[:, 10] == 0).all() and (d[:, 11] == 0).all()  # blur and noise unknown
    assert d[0, 8] == pytest.approx(math.log10(1 / 4094) + 4) and d[0, 9] == 1.0


def test_training_windows(data):
    heo.main([str(data)])
    windows = heo.Windows(heo.scenes(data, ("train",)), size=16)
    assert len(windows) == 1 and windows.card.shape == (8, desc.N_FIELDS)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, dn, label = windows.draw(rng)
        assert dn.shape == (8, 16, 16) and label.shape == (16, 16)
        assert (label != 255).mean() >= 0.5
        assert (label[~dn.any(axis=0)] == 255).all()  # no image, no label
    with pytest.raises(ValueError, match="smaller"):
        heo.Windows(heo.scenes(data, ("train",)), size=48)
    with pytest.raises(ValueError, match="no labelled HEO scenes"):
        heo.scenes(data, ("stress",))
