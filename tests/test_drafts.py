"""Reviewing drafts: two made-up HEO scenes and one Eaglet II scene, laid out as
ocm_prelabel.py leaves them; their table, true colour on the draft's grid, edits, labels and
verdicts."""

from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import drafts, viz  # noqa: E402

H, W = 32, 40  # the drafts' grid: 5 m
X0, Y0 = 500000.0, 4500000.0
CRS = "EPSG:32633"
REVIEW = ["scene", "product", "satellite", "start", "lat", "lon", "sun_elevation"]
REVIEW += ["role", "verdict", "use", "labeller", "date", "notes"]
HEO = {"IMH02_20261009T133222": "keep", "IMH03_20261007T121948": "correct"}


def raster(path, data, res, dtype="uint8", nodata=None):
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


def draft_files(folder, draft="ocm4"):
    """A draft: thick cloud (OmniCloudMask's 1) at the top, thin (2) in the middle, no data in
    the last column; its certainty low (0.4) in the top-left 16 x 16 window."""
    folder.mkdir(parents=True)
    codes = np.zeros((H, W), np.uint8)
    codes[:8], codes[8:16] = 1, 2
    codes[:, -1] = 255
    raster(folder / f"{draft}.tif", codes, 5.0, nodata=255)
    conf = np.full((4, H, W), 225, np.uint8)
    conf[:, :16, :16] = 100
    conf[:, :, -1] = 255
    raster(folder / f"{draft}.conf.tif", conf, 5.0, nodata=255)


def scenes_csv(labels, rows):
    with open(labels / "scenes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW)
        writer.writeheader()
        writer.writerows(rows)
    with open(labels / "drafts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "draft", "valid", "clear", "thick", "thin", "shadow"])
        for r in rows:
            writer.writerow([r["scene"], "ocm4", 0.975, 0.5, 0.25, 0.25, 0.0])


@pytest.fixture
def data(tmp_path):
    """DATA with two HEO scenes (their bands at 2.5 m in zips) and one Eaglet II scene."""
    labels, products = tmp_path / "labels/iride_heo", tmp_path / "raw/iride_heo/products"
    products.mkdir(parents=True)
    rows = []
    for name, verdict in HEO.items():
        draft_files(labels / name)
        product = products / f"{name.split('_')[0]}_1CST__OPT8_{name.split('_')[1]}_O_Z99.zip"
        with zipfile.ZipFile(product, "w") as z:
            for band, value in (("B01", 100), ("B02", 200), ("B03", 300), ("B07", 400)):
                image = np.full((2 * H, 2 * W), value, np.uint16)
                image[:, -2:] = 0  # the last draft column has no data
                raster(tmp_path / f"x.{band}.tif", image, 2.5, "uint16")
                z.write(tmp_path / f"x.{band}.tif", f"{name}/{name}.{band}.tif")
        rows.append({"scene": name, "product": product.name, "role": "train", "verdict": verdict})
    scenes_csv(labels, rows)
    labels = tmp_path / "labels/iride_eaglet2"
    products = tmp_path / "raw/iride_eaglet2/products"
    products.mkdir(parents=True)
    draft_files(labels / "IME01_20261018T052237", "ocm4")
    image = np.stack([np.full((2 * H, 2 * W), v, np.uint16) for v in (30, 20, 10)])
    raster(tmp_path / "e.tif", image, 2.5, "uint16")
    product = products / "IME01_1CSP__RGB3_20261018T052237_A01.zip"
    with zipfile.ZipFile(product, "w") as z:
        z.write(tmp_path / "e.tif", "IME01/IME01_20261018T052237.tif")
    scenes_csv(labels, [{"scene": "IME01_20261018T052237", "product": product.name}])
    return tmp_path


def test_table(data):
    d = drafts.Drafts(data, "iride_heo")
    assert d.draft == "ocm4" and d.names == list(HEO)
    rows = d.table()
    assert [r["verdict"] for r in rows] == ["keep", "correct"]
    assert rows[0]["thin"] == 0.25 and rows[0]["label.tif"] == ""
    assert "2 scenes, draft ocm4; 1 correct, 1 keep; 0 with label.tif" in repr(d)
    assert "<table>" in d.html() and "IMH03_20261007T121948" in d.html()
    with pytest.raises(ValueError, match="no draft"):
        drafts.Drafts(data, "iride_heo", draft="ocm9")


def test_a_scene_on_the_draft_grid(data):
    d = drafts.Drafts(data, "iride_heo")
    s = d.scene("IMH02_20261009T133222")
    assert s.rgb.shape == (3, H, W) and s.label is None
    assert np.allclose(s.rgb[:, :, :-1], np.array([300, 200, 100])[:, None, None])  # R, G, B
    assert (s.rgb[:, :, -1] == 0).all()  # no data stays 0
    assert (s.draft[:8, :-1] == 2).all() and (s.draft[8:16, :-1] == 1).all()  # Catena's codes
    assert (s.draft[:, -1] == viz.NO_LABEL).all() and s.picture.shape == (H, W, 3)
    assert s.sure[0, 0] == pytest.approx(0.4) and np.isnan(s.sure[0, -1])
    assert d.unsure(s, size=16, n=1)[0][:2] == (0, 0)  # where it was least sure
    assert "verdict keep" in s.summary()
    e = drafts.Drafts(data, "iride_eaglet2").scene("IME01_20261018T052237")
    assert np.allclose(e.rgb[:, :, :-1], np.array([30, 20, 10])[:, None, None])


def test_edit_and_save(data):
    d = drafts.Drafts(data, "iride_heo")
    s = d.scene("IMH03_20261007T121948")
    assert s.edit(0, where=1) == 8 * (W - 1)  # its thin cloud is clear
    assert s.edit(3, box=(16, 18, 0, 4)) == 8  # shadow in a box
    assert s.edit(0, below=0.5, where=2) == 8 * 16  # thick cloud it was unsure of
    assert s.edit(2, box=(0, H, W - 1, W)) == 0  # no data stays no data
    with pytest.raises(ValueError):
        s.edit(9)
    out = d.save_label(s, who="tester", note="made up")
    with rasterio.open(out) as src:
        assert np.array_equal(src.read(1), s.label) and src.tags()["labeller"] == "tester"
        assert "shadow (box (16, 18, 0, 4)): 8 pixels" in src.tags()["edits"]
    assert s.edits == [] and d.has_label(s.name)
    again = d.scene(s.name)
    assert np.array_equal(again.work, s.label)  # the saved label is now the working one


def test_decide(data):
    d = drafts.Drafts(data, "iride_heo")
    path = d.decide([{"scene": "IMH02_20261009T133222", "verdict": "all_clear"}], day="2026-11-05")
    path = d.decide(
        [{"scene": "IMH03_20261007T121948", "verdict": "drop", "notes": "haze"}], day="2026-11-05"
    )
    path = d.decide(
        [{"scene": "IMH02_20261009T133222", "verdict": "keep", "role": "test"}], day="2026-11-05"
    )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(drafts.DECISIONS) and len(rows) == 2
    assert (rows[0]["verdict"], rows[0]["role"]) == ("keep", "test")  # the later decision
    assert (rows[1]["verdict"], rows[1]["notes"], rows[1]["use"]) == ("drop", "haze", "label")
    for wrong in (
        {"scene": "nope", "verdict": "keep"},
        {"scene": rows[0]["scene"], "verdict": "ok"},
    ):
        with pytest.raises(ValueError):
            d.decide(
                [{"scene": "IMH03_20261007T121948", "verdict": "keep"}, wrong], day="2026-11-05"
            )
    with open(path, newline="") as f:
        assert list(csv.DictReader(f)) == rows  # a wrong decision writes nothing


def test_keep_labels_and_export(data):
    d = drafts.Drafts(data, "iride_heo")
    assert d.keep_labels() == ["IMH02_20261009T133222"]  # the one marked keep
    assert d.keep_labels() == []  # an existing label.tif is never touched
    s = d.scene("IMH02_20261009T133222")
    assert np.array_equal(s.label, s.draft)
    rgb, start = d.export_for_editing(s)
    with rasterio.open(rgb) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8"
        assert (src.read()[:, :, -1] == 255).all()  # no data
    with rasterio.open(start) as src:
        assert np.array_equal(src.read(1), s.work)


def test_figures(data):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    d = drafts.Drafts(data, "iride_heo")
    s = d.scene("IMH02_20261009T133222")
    assert len(d.show(s, show=False).axes) == 5  # four pictures and a colour bar
    fig = d.zoom(s, 8, 10, size=12, show=False)
    assert fig.axes[0].get_ylim()[1] == pytest.approx(7.5)  # rows counted from 8
    assert len(d.compare(s, "ocm4", show=False).axes) == 3
