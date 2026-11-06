"""IRIDE Eaglet II: the gains of a product from its STAC item and its XML, a scene's bands on
its label's grid as reflectance, and what the model is told; on made-up products laid out as
the real ones (one image of three bands at 2.5 m in a zip, with a STAC item and an XML)."""

from __future__ import annotations

import csv
import json
import math
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena.data import eaglet2  # noqa: E402

H, W = 8, 10  # the label's grid, 5 m
X0, Y0 = 500000.0, 4500000.0
GAIN = (1.92e-5, 2.5e-5, 3.0e-5)
OFFSET = (-0.0008, -0.0006, 0.0)
SCENE = "IME01_20261018T052237"
STEM = f"IME01_1CSP__RGB3_{SCENE.split('_')[1]}_20261018T052239_20261019T101010_00001_R001_O_A01"


def raster(path, data, res, dtype) -> None:
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": dtype,
        "crs": "EPSG:32633",
        "transform": from_origin(X0, Y0, res, res),
        "nodata": 0 if dtype == "uint16" else 255,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))


def image() -> np.ndarray:
    """(3, 16, 20) DN at 2.5 m: band k is 10000 (k + 1) plus the column, no data in the top-left
    2 x 2 (one 5 m cell) and in band 2's last column (the 5 m cells of the last column)."""
    cols = np.tile(np.arange(2 * W), (2 * H, 1))
    dn = np.stack([10000 * (k + 1) + cols for k in range(3)]).astype(np.uint16)
    dn[:, :2, :2] = 0
    dn[2, :, -2:] = 0
    return dn


def stac(scale=GAIN, offset=OFFSET) -> dict:
    bands = [{"scale": s, "offset": o, "nodata": 0} for s, o in zip(scale, offset, strict=True)]
    return {
        "type": "Feature",
        "assets": {"image": {"href": f"{STEM}.tif", "raster:bands": bands}, "thumb": {}},
    }


def xml(gain=GAIN, offset=OFFSET) -> str:
    pairs = zip(gain, offset, strict=True)
    parts = [f"<BAND><GAIN>{g}</GAIN><OFFSET>{o}</OFFSET></BAND>" for g, o in pairs]
    return "<PRODUCT><RADIOMETRY>" + "".join(parts) + "</RADIOMETRY></PRODUCT>"


def product(folder, item: dict | None, text: str | None, stem: str = STEM):
    folder.mkdir(parents=True, exist_ok=True)
    tif = folder / "image.tif"
    raster(tif, image(), 2.5, "uint16")
    path = folder / f"{stem}.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.write(tif, f"{stem}/{stem}.tif")
        if item is not None:
            z.writestr(f"{stem}/{stem}.json", json.dumps(item))
        if text is not None:
            z.writestr(f"{stem}/{stem}.xml", text)
    tif.unlink()
    return path


@pytest.fixture
def data(tmp_path):
    """DATA with one labelled Eaglet II test scene and one dropped scene."""
    labels, products = eaglet2.folders(tmp_path)
    product(products, stac(), xml())
    (labels / SCENE).mkdir(parents=True)
    label = np.zeros((H, W), np.uint8)
    label[:4] = 2
    label[-1] = 3
    raster(labels / SCENE / "label.tif", label, 5.0, "uint8")
    rows = [
        {"scene": SCENE, "product": f"{STEM}.zip", "role": "test", "verdict": "correct"},
        {"scene": "IME02_20261017T065625", "product": "y.zip", "role": "test", "verdict": "drop"},
    ]
    with open(labels / "scenes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_gains(tmp_path):
    both = product(tmp_path / "a", stac(), xml())
    gain, offset = eaglet2.gains(both)
    assert gain == pytest.approx(GAIN) and offset == pytest.approx(OFFSET)
    stac_only = product(tmp_path / "b", stac(), None)
    assert eaglet2.gains(stac_only)[0] == pytest.approx(GAIN)
    xml_only = product(tmp_path / "c", None, xml())
    assert eaglet2.gains(xml_only)[1] == pytest.approx(OFFSET)
    full = ([1.74632311e-05, 1.48511197e-05, 2.30016455e-05], [-0.00080327, 0.00244834, 0.00288029])
    short = ([1.7e-05, 1.5e-05, 2.3e-05], [-0.000803, 0.002448, 0.00288])  # IME08's STAC item
    real = product(tmp_path / "d", stac(*short), xml(*full))
    assert eaglet2.gains(real)[0] == pytest.approx(full[0])  # the XML's, in full
    one_off = product(tmp_path / "g", stac([1.8e-05, 1.5e-05, 2.3e-05], short[1]), xml(*full))
    with pytest.raises(ValueError, match="disagree"):
        eaglet2.gains(one_off)
    swapped = product(tmp_path / "e", stac(scale=GAIN[::-1]), xml())
    with pytest.raises(ValueError, match="disagree"):
        eaglet2.gains(swapped)
    with pytest.raises(ValueError, match="neither"):
        eaglet2.gains(product(tmp_path / "f", {"assets": {}}, "<PRODUCT/>"))
    tagged = '<a:GAIN band="1"> 1.0e-5 </a:GAIN><ANALOG_GAIN>4</ANALOG_GAIN>'
    assert eaglet2.from_xml(tagged * 3 + "<OFFSET>0</OFFSET>" * 3)[0] == pytest.approx([1e-5] * 3)


def test_rounded():
    assert eaglet2.rounded(1.7e-05, 1.74632311e-05)  # to two significant figures
    assert eaglet2.rounded(1.5e-05, 1.48511197e-05) and eaglet2.rounded(0.00288, 0.00288029)
    assert eaglet2.rounded(1.7463e-05, 1.74632311e-05) and eaglet2.rounded(2.0, 2.0)
    assert not eaglet2.rounded(1.8e-05, 1.74632311e-05)  # a wrong digit
    assert not eaglet2.rounded(1.7464e-05, 1.74632311e-05)  # more digits, and off
    assert not eaglet2.rounded(2.3e-05, 1.74632311e-05)  # another band
    assert not eaglet2.rounded(float("nan"), 1.0)


def test_scenes_and_reading_one(data):
    chosen = eaglet2.scenes(data)
    assert [s.name for s in chosen] == [SCENE] and chosen[0].role == "test"
    assert chosen[0].draft().name == "ocm4_nirblue.tif"
    rho, label = eaglet2.read(chosen[0])
    assert rho.shape == (3, H, W) and rho.dtype == np.float32 and label.shape == (H, W)
    dn = image().astype(np.float64)
    for k in range(3):  # each 5 m cell: the mean of its 2 x 2 pixels, then the gain
        mean = dn[k].reshape(H, 2, W, 2).mean(axis=(1, 3))
        expect = mean * GAIN[k] + OFFSET[k]
        assert rho[k, 1:, :-1] == pytest.approx(expect[1:, :-1], rel=1e-5)
    assert (rho[:, 0, 0] == 0).all() and label[0, 0] == 255  # no data at all
    assert (rho[:, :, -1] == 0).all() and (label[:, -1] == 255).all()  # no blue
    assert label[1, 0] == 2 and label[-1, 0] == 3
    with pytest.raises(ValueError, match="no labelled"):
        eaglet2.scenes(data, roles=("train",))


def test_what_the_model_is_told():
    d = eaglet2.descriptors()
    assert d.shape == (3, desc.N_FIELDS)
    card = desc.card(d)
    assert card[0, 0] == pytest.approx(math.log(665 - 300) - 2)  # red from 665 nm
    assert card[2, 2] == pytest.approx(math.log(515 - 300) - 2)  # blue up to 515 nm
    assert (card[:, 3] == 0).all() and (card[:, 4] == 0).all()  # top of atmosphere, 5 m bands
    assert card[:, 5] == pytest.approx(-1.0)  # a 5 m grid
    assert (d[:, 5:14] == 0).all()  # the chain unknown


def test_the_probe(data, capsys):
    eaglet2.main([str(data), "--read"])
    out = capsys.readouterr().out
    assert "1 of 1 products: the STAC item and the XML agree" in out
    assert "1.9200e-05 -0.00080" in out and SCENE in out and "rounded" not in out
