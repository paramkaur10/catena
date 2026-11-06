"""The tests of the big run: tiles that cover a scene once, each pixel taken from the tile it
lies deepest in; the scores; every test set read from made-up data laid out as the real ones;
and the command with two tiny untrained runs (torch, CPU)."""

from __future__ import annotations

import copy
import csv
import json
import sys
import types
import warnings
import zipfile
from itertools import pairwise

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena import evaluate_clarity as E  # noqa: E402
from catena.data import eaglet2, heo  # noqa: E402

UTM = "EPSG:32633"
HEO_SCENES = {  # name: role; a train scene the tests leave alone
    "IMH12_20260930T092923": "test",
    "IMH14_20261008T090717": "test",
    "IMH02_20260308T122749": "stress",
    "IMH03_20261007T121948": "train",
}
EAGLET = "IME04_20261017T093952"
EAGLET_STEM = "IME04_1CSP__RGB3_20261017T093952_20261017T093954_20261018T000000_00002_R002_O_A01"
PHISAT = "PHISAT-2_L1_000008357_20261025110655_20261025110658_284FD4D8"


def geotiff(path, data: np.ndarray, res: float, crs=UTM, west=300000.0, north=4650000.0):
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": from_origin(west, north, res, res),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labels_and_draft(rng, h: int, w: int, shadow: bool = True):
    """A label (cloud on top, shadow below if any), and OmniCloudMask's draft of it in its
    own codes: right but for a few pixels, and no data in the last column."""
    label = np.zeros((h, w), np.uint8)
    label[: h // 3] = 2
    label[h // 3 : h // 2] = 1
    if shadow:
        label[-3:, : w // 2] = 3
    draft = np.array([0, 2, 1, 3], np.uint8)[label]  # Catena's codes to OmniCloudMask's
    flip = rng.random((h, w)) < 0.1
    draft[flip] = 0
    draft[:, -1] = 255
    return label, draft


def sentinel2(root) -> None:
    """Two test patches of 128 px in D2's layout."""
    rows = []
    for k in range(2):
        rng = np.random.default_rng(k)
        geotiff(
            root / "test" / f"t{k}.s2l1c.tif",
            rng.integers(500, 6000, (13, 128, 128)).astype(np.uint16),
            10,
        )
        geotiff(
            root / "test" / f"t{k}.target.tif", rng.integers(0, 4, (128, 128)).astype(np.uint8), 10
        )
        rows.append(
            {
                "id": f"t{k}",
                "split": "test",
                "s2l1c_file": f"test/t{k}.s2l1c.tif",
                "target_file": f"test/t{k}.target.tif",
                "height": 128,
                "width": 128,
                "baseline": "02.07",
                "sun_zenith": "35.5",
            }
        )
    write_csv(root / "index.csv", rows)


def landsat8(root) -> None:
    """Two Biome scenes of 90 x 100 px in pilot.csv, the first with shadow labels; no data in
    the first 8 columns."""
    rows = []
    for k, name in enumerate(("LC80000002014001LGN00", "LC80000012014002LGN00")):
        rng = np.random.default_rng(10 + k)
        dn = rng.integers(10000, 40000, (8, 90, 100)).astype(np.uint16)
        dn[:, :, :8] = 0
        label, _ = labels_and_draft(rng, 90, 100, shadow=k == 0)
        label[:, :8] = 255
        geotiff(root / name / f"{name}.bands.tif", dn, 30)
        geotiff(root / name / f"{name}.mask.tif", label, 30)
        lines = ["GROUP = L1_METADATA_FILE", f"    SUN_ELEVATION = {40.0 + 10 * k}"]
        for n in (1, 2, 3, 4, 5, 6, 7, 9):
            lines += [
                f"    RADIANCE_MULT_BAND_{n} = 1.2E-02",
                f"    REFLECTANCE_MULT_BAND_{n} = 2.0000E-05",
                f"    REFLECTANCE_ADD_BAND_{n} = -0.100000",
                f"    REFLECTANCE_MAXIMUM_BAND_{n} = 1.210700",
                f"    REFLECTANCE_MINIMUM_BAND_{n} = -0.099980",
            ]
        (root / name / f"{name}_MTL.txt").write_text("\n".join([*lines, "END"]) + "\n")
        rows.append({"scene": name, "biome": "Water", "has_shadow": str(k == 0)})
    write_csv(root / "pilot.csv", rows)
    write_csv(root / "index.csv", rows)


def landsat7(root) -> None:
    """One L7 Irish scene of 90 x 100 px in TorchGeo's layout, listed in pilot.csv."""
    folder = root / "tropical" / "l7irish" / "tropical" / "p5_r60"
    rng = np.random.default_rng(20)
    image = rng.integers(1, 256, (9, 90, 100)).astype(np.uint8)
    mask = rng.choice([128, 192, 255], (90, 100)).astype(np.uint8)  # clear, thin, thick
    geotiff(folder / "L71p5_r60_p5_r6020010104.TIF", image, 30, "EPSG:32605")
    geotiff(folder / "L7_p5_r60_newmask2015.TIF", mask, 30, "EPSG:32605")
    lines = ["GROUP = L1_METADATA_FILE", "    ACQUISITION_DATE = 2001-01-04"]
    for n in ("1", "2", "3", "4", "5", "7"):
        lines += [
            f"    LMAX_BAND{n} = 191.6",
            f"    LMIN_BAND{n} = -6.2",
            f"    QCALMAX_BAND{n} = 255.0",
            f"    QCALMIN_BAND{n} = 1.0",
            f'    BAND{n}_GAIN = "H"',
            f'    BAND{n}_GAIN_CHANGE = "0"',
        ]
    lines.append("    SUN_ELEVATION = 30.0")
    (folder / "L71p5_r60_p5_r6020010104_MTL.txt").write_text("\n".join([*lines, "END"]) + "\n")
    folder_name = "tropical/l7irish/tropical/p5_r60"
    write_csv(
        root / "pilot.csv", [{"zone": "tropical", "folder": folder_name, "has_shadow": "False"}]
    )


def iride_heo(data) -> None:
    """Four HEO scenes of 60 x 70 px at 5 m, converted, with labels and drafts."""
    labels, chips = data / "labels/iride_heo", data / "chips/iride_heo"
    rows = []
    for k, (name, role) in enumerate(HEO_SCENES.items()):
        rng = np.random.default_rng(30 + k)
        label, draft = labels_and_draft(rng, 60, 70, shadow=k != 1)
        dn = rng.integers(300, 3000, (8, 60, 70)).astype(np.uint16)
        dn[3, :2] = 0  # no red in the first two rows
        geotiff(labels / name / "label.tif", label, 5)
        geotiff(labels / name / "ocm4.tif", draft, 5)
        geotiff(chips / f"{name}.bands.tif", dn, 5)
        rows.append({"scene": name, "product": f"{name}.zip", "role": role, "verdict": "correct"})
    write_csv(labels / "scenes.csv", rows)


def iride_eaglet2(data) -> None:
    """One Eaglet II scene: its image at 2.5 m in a zip with its STAC item and XML, and its
    label and both drafts on a 5 m grid of 50 x 60 px."""
    labels, products = eaglet2.folders(data)
    rng = np.random.default_rng(40)
    label, draft = labels_and_draft(rng, 50, 60)
    geotiff(labels / EAGLET / "label.tif", label, 5)
    geotiff(labels / EAGLET / "ocm4_nirblue.tif", draft, 5)
    other = draft.copy()
    other[:2] = 255  # this one has no data in the first two rows
    geotiff(labels / EAGLET / "ocm4_nirzeros.tif", other, 5)
    dn = rng.integers(5000, 40000, (3, 100, 120)).astype(np.uint16)
    geotiff(data / "e.tif", dn, 2.5)
    bands = [{"scale": 2e-5, "offset": -0.001} for _ in range(3)]
    item = {"assets": {"image": {"raster:bands": bands}}}
    text = "<P>" + "<GAIN>2e-05</GAIN><OFFSET>-0.001</OFFSET>" * 3 + "</P>"
    products.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(products / f"{EAGLET_STEM}.zip", "w") as z:
        z.write(data / "e.tif", f"{EAGLET_STEM}/{EAGLET_STEM}.tif")
        z.writestr(f"{EAGLET_STEM}/{EAGLET_STEM}.json", json.dumps(item))
        z.writestr(f"{EAGLET_STEM}/{EAGLET_STEM}.xml", text)
    (data / "e.tif").unlink()
    row = {"scene": EAGLET, "product": f"{EAGLET_STEM}.zip", "role": "test", "verdict": "correct"}
    write_csv(labels / "scenes.csv", [row])


def phisat2(data) -> None:
    """One Phi-sat-2 product of 64 x 64 px on a longitude-latitude grid, its label and draft
    on the same grid."""
    labels, products = data / "labels/phisat2", data / "raw/phisat2/products"
    rng = np.random.default_rng(50)
    grid = {"res": 4.5e-5, "crs": "EPSG:4326", "west": 9.0, "north": 46.5}
    dn = rng.integers(100, 900, (8, 64, 64)).astype(np.uint16)
    dn[:, :4, :4] = 0
    geotiff(data / "p.tif", dn, **grid)
    products.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(products / f"{PHISAT}.zip", "w") as z:
        z.write(data / "p.tif", "bands/scene_0_BC_multiband.tiff")
    (data / "p.tif").unlink()
    label, draft = labels_and_draft(rng, 64, 64)
    geotiff(labels / "PHISAT2_20261025T110655" / "label.tif", label, **grid)
    geotiff(labels / "PHISAT2_20261025T110655" / "ocm4.tif", draft, **grid)
    row = {"scene": "PHISAT2_20261025T110655", "product": f"{PHISAT}.zip", "role": "test"}
    write_csv(labels / "scenes.csv", [row | {"verdict": "correct"}])


def cloudperu2(root) -> None:
    """One train and two validation patches of 32 px, cloud in the top half."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rows = []
    for n, split in ((1, "train"), (2, "validation"), (3, "validation")):
        image = np.random.default_rng(60 + n).integers(10, 250, (4, 32, 32)).astype(np.uint16)
        mask = np.zeros((1, 32, 32), np.uint8)
        mask[:, :16] = 255
        with warnings.catch_warnings():  # no georeferencing, as in CloudPeru2
            warnings.simplefilter("ignore")
            for path, x in ((f"images/{n}.tif", image), (f"masks/{n}_Mask.tif", mask)):
                profile = {"driver": "GTiff", "height": 32, "width": 32, "count": len(x)}
                with rasterio.open(root / path, "w", dtype=x.dtype.name, **profile) as f:
                    f.write(x)
        rows.append({"n": n, "group": n, "split": split, "nodata_frac": 0, "cloud_frac": 0.5})
    write_csv(root / "index.csv", rows)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A DATA folder with every test set, in the layouts the tests read."""
    root = tmp_path_factory.mktemp("catena_data")
    sentinel2(root / "chips/cloudsen12plus")
    landsat8(root / "chips/l8_biome")
    landsat7(root / "chips/l7_irish")
    iride_heo(root)
    iride_eaglet2(root)
    phisat2(root)
    cloudperu2(root / "chips/cloudperu2")
    return root


# --- tiles ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, tile, margin",
    [(240, 240, 30), (250, 240, 30), (512, 240, 30), (1000, 240, 30), (787, 80, 10)],
)
def test_tiles_cover_every_pixel_once(n, tile, margin):
    first = E.starts(n, tile, margin)
    parts = E.owned(first, tile, n)
    assert first[0] == 0 and first[-1] == n - tile and first == sorted(set(first))
    assert parts[0][0] == 0 and parts[-1][1] == n
    assert all(a[1] == b[0] for a, b in pairwise(parts))  # no gap, no overlap
    for start, (a, b) in zip(first, parts, strict=True):
        assert start <= a < b <= start + tile  # inside its tile
        assert a == 0 or a - start >= margin  # deep inside, but at the scene's edges
        assert b == n or start + tile - b >= margin


def fake(margin: int):
    """A classifier whose answer shows where in a tile a pixel lay: clear, thin or thick by
    its first band, but shadow within margin of the tile's edge."""

    def classify(x: np.ndarray, told: np.ndarray, rgbn=None) -> dict:
        assert x.dtype == np.float32 and told.shape == (x.shape[1], desc.N_CHAIN_CARD)
        assert rgbn is None or (len(rgbn) == 4 and max(rgbn) < x.shape[1])
        out = np.clip(np.floor(x[:, 0] * 3), 0, 2).astype(np.uint8)
        edge = np.ones(x.shape[-2:], bool)
        edge[margin:-margin, margin:-margin] = False
        out[:, edge] = 3
        return {"m": out, "m2": out}

    return classify


def test_each_pixel_comes_from_the_tile_it_lies_deepest_in():
    rng = np.random.default_rng(0)
    image = rng.random((2, 300, 520)).astype(np.float32)
    label = np.zeros((300, 520), np.uint8)
    label[:, 400:] = 255  # no label on the right: those tiles are not predicted
    told = np.zeros((2, desc.N_CHAIN_CARD), np.float32)
    item = E.Item("x", image, told, label, tile=80, margin=10)
    pred = E.predict(fake(10), item)["m"]
    expect = np.clip(np.floor(image[0] * 3), 0, 2)
    inner = (slice(10, -10), slice(10, 400))  # away from the scene's edges
    assert (pred[inner] == expect[inner]).all()  # never within 10 px of a tile's edge
    assert (pred[:, :400][label[:, :400] != 255] < 255).all()
    assert (pred[:, 480:] == 255).all()  # tiles over no label at all: left out
    small = E.Item("s", image[:, :50, :60], told, label[:50, :60])  # smaller than a tile
    out = E.predict(fake(30), small)["m2"]
    assert out.shape == (50, 60) and (out[30:, 30:] == expect[30:50, 30:60]).all()


def test_scores():
    conf = np.zeros((6, 4), np.int64)
    conf[0, 0], conf[0, 1] = 80, 20  # clear: 20 called thin
    conf[2, 2], conf[2, 1] = 50, 50  # thick: half called thin
    conf[1, 1] = 30
    s = E.scores(conf)
    assert s["f1_clear"] == pytest.approx(2 * 80 / (100 + 80))
    assert s["f1_thin"] == pytest.approx(2 * 30 / (30 + 100))
    assert np.isnan(s["f1_shadow"])  # neither labelled nor predicted
    assert s["f1_mean"] == pytest.approx(np.mean([s["f1_clear"], s["f1_thin"], s["f1_thick"]]))
    assert s["accuracy"] == pytest.approx(160 / 230)
    assert s["cloud_recall"] == 1.0 and s["cloud_precision"] == pytest.approx(130 / 150)
    assert s["cloud_ba"] == pytest.approx((1.0 + 0.8) / 2)
    peru = np.zeros((6, 4), np.int64)
    peru[4, 2], peru[4, 0], peru[5, 3], peru[5, 1] = 90, 10, 95, 5  # cloud, not cloud only
    s = E.scores(peru)
    assert np.isnan(s["f1_mean"]) and np.isnan(s["accuracy"])
    assert s["cloud_ba"] == pytest.approx((0.9 + 0.95) / 2)
    assert s["cloud_f1"] == pytest.approx(2 * 90 / (2 * 90 + 5 + 10))
    label = np.array([[0, 1, 255], [4, 5, 3]], np.uint8)
    pred = np.array([[0, 2, 1], [1, 255, 3]], np.uint8)
    conf = E.confusion(label, pred)
    assert conf.sum() == 4 and conf[0, 0] == conf[1, 2] == conf[4, 1] == conf[3, 3] == 1
    both = E.pooled(conf, np.zeros((6, 4), np.int64))  # no scene with shadow labels
    assert both["cloud_f1"] == pytest.approx(1.0) and np.isnan(both["f1_mean"])


def test_card_checks():
    told = np.arange(3 * desc.N_CHAIN_CARD, dtype=np.float32).reshape(3, desc.N_CHAIN_CARD)
    told[:, 9:] = [[1, 1], [0, 0], [1, 0]]  # blur and noise known, neither, the blur only
    x = np.ones((2, 3, 4, 4), np.float32)
    rgbn = (2, 1, 0, 2)  # as Eaglet II: blue stands in for the near infrared
    assert E.vary("honest", x, told, rgbn)[1] is told
    shuffled = E.vary("shuffled", x, told, rgbn)[1]
    assert (shuffled[1] == told[0]).all() and (shuffled[0] == told[2]).all()
    for check, step in (("pixel x3", np.log2(3)), ("pixel /3", -np.log2(3))):
        moved = E.vary(check, x, told, rgbn)[1]
        assert moved[:, 5] == pytest.approx(told[:, 5] + step)
        assert (np.delete(moved, 5, axis=1) == np.delete(told, 5, axis=1)).all()
    only = E.vary("wavelengths only", x, told, rgbn)[1]
    assert (only[:, :3] == told[:, :3]).all() and (only[:, 3:] == 0).all()
    noisy = E.vary("noise x3", x, told, rgbn)[1]
    assert noisy[0, 7:9] == pytest.approx(told[0, 7:9] + np.log10(3))
    assert (noisy[1:, 7:9] == told[1:, 7:9]).all()  # unknown noise stays as it was
    assert (E.vary("blur x2", x, told, rgbn)[1][:, 6] == 2 * told[:, 6]).all()
    assert (told[:, 9:] == [[1, 1], [0, 0], [1, 0]]).all()  # the checks work on copies
    vx, vtold, four = E.vary("rgbn only", x, told, rgbn)
    assert vx.shape == (2, 3, 4, 4) and four == (0, 1, 2, 0) and (vtold == told[[2, 1, 0]]).all()
    vx, vtold, four = E.vary("rgb only", np.ones((2, 5, 4, 4)), np.ones((5, 11)), (1, 2, 3, 4))
    assert vx.shape == (2, 3, 4, 4) and vtold.shape == (3, 11) and four is None
    assert E.vary("rgb only", x, told, None) is None  # where the bands are not known
    assert E.named("seed0/sim", "honest") == "seed0/sim"
    assert E.group_of(E.named("seed0/sim", "pixel x3")) == "Clarity sim [pixel x3]"
    with pytest.raises(ValueError):
        E.vary("lies", x, told, rgbn)


# --- the sets -------------------------------------------------------------------------------


def test_the_sets(data, capsys):
    sets = {s.name: s for s in E.scene_sets(data, ["B", "C", "D", "E", "F"])}
    names = ["landsat8", "landsat7", "heo_test", "heo_stress", "eaglet2", "phisat2", "perusat1"]
    assert list(sets) == names
    assert [s.scenes for s in sets.values()] == [2, 1, 2, 1, 1, 1, 2]
    items = {name: list(s.items()) for name, s in sets.items()}
    l8 = items["landsat8"]
    assert [i.shadow_ok for i in l8] == [True, False] and (l8[0].tile, l8[0].margin) == (80, 10)
    assert l8[0].told.shape == (8, desc.N_CHAIN_CARD) and (l8[0].label[:, :8] == 255).all()
    assert (l8[0].told[:, 10] == 1).all() and (l8[0].told[:, 9] == 0).all()  # SNR, no MTF
    rho = l8[0].to_rho(l8[0].image[None, :, :8, 8:16])
    assert rho.shape == (1, 8, 8, 8) and rho.dtype == np.float32 and (rho > 0).all()
    assert items["landsat7"][0].told.shape == (6, desc.N_CHAIN_CARD)
    rgbn = {name: found[0].rgbn for name, found in items.items()}
    assert rgbn == {
        "landsat8": (1, 2, 3, 4),
        "landsat7": (0, 1, 2, 3),
        "heo_test": (1, 2, 3, 7),
        "heo_stress": (1, 2, 3, 7),
        "eaglet2": (2, 1, 0, 2),  # no near infrared: blue stands in
        "phisat2": (1, 2, 3, 7),
        "perusat1": (0, 1, 2, 3),
    }
    test = items["heo_test"]
    assert [i.name for i in test] == ["IMH12_20260930T092923", "IMH14_20261008T090717"]
    assert [i.shadow_ok for i in test] == [True, False]  # the second: cloud, no shadow label
    first = test[0]
    assert set(first.others) == {"ocm4"} and (first.label[:2] == 255).all()  # no red there
    assert (first.label[:, -1] == 255).all()  # the draft has no data there
    assert first.told[:, 5] == pytest.approx(-1.0) and first.to_rho is heo.reflectance
    ocm = first.others["ocm4"]
    assert ocm[5, 5] in (0, 2) and ocm[-2, 0] in (0, 3)  # in Catena's codes
    eag = items["eaglet2"][0]
    assert set(eag.others) == {"ocm4_nirblue", "ocm4_nirzeros"} and eag.image.shape == (3, 50, 60)
    assert (eag.label[:2] == 255).all()  # where one of the drafts has no data
    phi = items["phisat2"][0]
    assert phi.image.shape == (8, 64, 64) and phi.told.shape == (8, desc.N_CHAIN_CARD)
    assert (phi.label[:4, :4] == 255).all() and -2.2 < phi.told[0, 5] < -0.5  # about 4 m
    peru = items["perusat1"]
    assert [p.name for p in peru] == ["patch 2", "patch 3"] and not peru[0].shadow_ok
    assert set(np.unique(peru[0].label).tolist()) <= {4, 5}
    empty = E.scene_sets(data.parent / "nothing", ["D", "F"])
    assert empty == [] and "left out" in capsys.readouterr().out


def test_scoring_a_set(data, tmp_path):
    sets = {s.name: s for s in E.scene_sets(data, ["D", "E", "F"])}
    rows, lines = E.score_set(sets["eaglet2"], fake(30), maps=tmp_path / "maps")
    models = {r["model"] for r in rows}
    assert models == {"m", "m2", "ocm4_nirblue", "ocm4_nirzeros"}
    assert {r["scene"] for r in rows} == {EAGLET, "all"}
    assert len({r["pixels"] for r in rows}) == 1  # every model on the same pixels
    item = next(sets["eaglet2"].items())
    mine = next(r for r in rows if r["model"] == "ocm4_nirblue" and r["scene"] == "all")
    conf = E.confusion(item.label, item.others["ocm4_nirblue"])
    assert mine["confusion"] == " ".join(map(str, conf.ravel()))
    assert mine["cloud_f1"] == pytest.approx(E.scores(conf)["cloud_f1"])
    assert any("OmniCloudMask, NIR from blue" in line for line in lines)
    assert any("Test E: IRIDE Eaglet II" in line for line in lines)
    kept = np.load(tmp_path / "maps" / f"eaglet2__{EAGLET}.npz")
    assert set(kept.files) == {"label", "m", "m2", "ocm4_nirblue", "ocm4_nirzeros", "rgb", "step"}
    assert kept["rgb"].shape == (50, 60, 3) and kept["rgb"].dtype == np.uint8 and kept["step"] == 1
    rows, lines = E.score_set(sets["perusat1"], fake(30))
    pooled = next(r for r in rows if r["scene"] == "all" and r["model"] == "m")
    assert np.isnan(pooled["f1_mean"]) and 0 <= pooled["cloud_ba"] <= 1
    assert any("cloud or not" in line for line in lines)
    checks = ["honest", "shuffled", "rgb only"]
    rows, lines = E.score_set(sets["eaglet2"], fake(30), tmp_path / "checked", checks)
    models = {r["model"] for r in rows}
    assert {"m", "m@shuffled", "m@rgb only", "m2@rgb only", "ocm4_nirblue"} <= models
    assert any("[shuffled]" in line for line in lines)
    kept = np.load(tmp_path / "checked" / f"eaglet2__{EAGLET}.npz")
    assert not any("@" in k for k in kept.files)  # the maps: honest only


# --- the command (torch) ----------------------------------------------------------------


def test_the_command(data, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    from catena import train_clarity as C
    from catena.model import Clarity

    runs = tmp_path / "runs"
    config = {"args": {"widths": [8, 8, 8, 8], "stem": 8}}
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = Clarity(widths=(8, 8, 8, 8), stem=8, fields=desc.N_CARD)
        (runs / f"seed{seed}").mkdir(parents=True)
        models = {"real": net, "sim": copy.deepcopy(net)}
        C.save(runs / f"seed{seed}" / "final.pt", 1, models, config)
    arms = tmp_path / "arms"  # a run of two arms, one folder deeper
    (arms / "chain/seed0").mkdir(parents=True)
    models = {a: C.build(a, (8, 8, 8, 8), 8) for a in ("chain", "fixed")}
    C.save(arms / "chain/seed0/final.pt", 1, models, config)
    nets = E.load_runs([arms], "cpu")
    assert set(nets) == {"seed0/chain", "seed0/fixed"} and nets["seed0/fixed"].fields == 0
    with pytest.raises(SystemExit, match="evaluate them apart"):
        E.load_runs([runs, runs], "cpu")
    args = [str(runs), str(arms), "--data", str(data), "--device", "cpu", "--workers", "0"]
    E.main([*args, "--windows", "2", "--a-size", "48", "--batch", "4"])
    assert not (runs / "clarity_tests.csv").exists()
    (arms / "clarity_tests.csv").rename(runs / "clarity_tests.csv")  # the last RUNS folder's
    with open(runs / "clarity_tests.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["test"] for r in rows} == set(E.TESTS)
    clarity = {"seed0/real", "seed0/sim", "seed1/real", "seed1/sim", "seed0/chain", "seed0/fixed"}
    assert clarity <= {r["model"] for r in rows}
    assert {r["model"] for r in rows if r["set"] == "heo_test"} == clarity | {"ocm4"}
    eaglet = {r["model"] for r in rows if r["set"] == "eaglet2"}
    assert "seed0/fixed" in eaglet  # with blue for its near infrared
    cameras = {r["scene"] for r in rows if r["test"] == "A"}
    assert len(cameras) == 23 and "sentinel2" in cameras
    for r in rows:
        counts = [int(v) for v in r["confusion"].split()]
        assert len(counts) == 24 and sum(counts) == int(r["pixels"]) > 0
    out = capsys.readouterr().out
    assert "Clarity real (2 seeds)" in out and "Clarity sim (2 seeds)" in out
    assert "Clarity + MTF, SNR (1 seed)" in out and "U-Net fixed bands (1 seed)" in out
    assert "4 cameras beyond the range" in out and "Test D: IRIDE HEO test scenes" in out
    with pytest.raises(SystemExit):
        E.parse_args([str(runs), "--a-size", "40"])


def test_card_checks_on_the_command(data, tmp_path):
    torch = pytest.importorskip("torch")
    from catena import train_clarity as C

    runs = tmp_path / "arms"
    (runs / "seed0").mkdir(parents=True)
    torch.manual_seed(0)
    models = {a: C.build(a, (8, 8, 8, 8), 8) for a in ("sim", "fixed")}
    C.save(runs / "seed0/final.pt", 1, models, {"args": {"widths": [8, 8, 8, 8], "stem": 8}})
    args = [str(runs), "--data", str(data), "--device", "cpu", "--workers", "0", "--tests", "A"]
    args += ["F", "--windows", "2", "--a-size", "48", "--batch", "4"]
    E.main([*args, "--cards", "honest", "shuffled", "rgb only", "rgbn only"])
    with open(runs / "clarity_tests_AF_cards.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for test in ("A", "F"):
        models = {r["model"] for r in rows if r["test"] == test}
        assert {"seed0/sim", "seed0/sim@shuffled", "seed0/sim@rgb only"} <= models
        assert "seed0/fixed@rgbn only" in models and "seed0/fixed@rgb only" not in models
    a = {r["scene"] for r in rows if r["model"] == "seed0/sim@rgb only"}
    assert len(a & {"sentinel2", "rgbn_10m_mtf0.3_snr150"}) == 2  # every camera, too


def test_omnicloudmask_beside_on_landsat(data, monkeypatch):
    seen = []

    def ocm(red, green, nir):
        seen.append((red, green, nir))
        out = np.where(red > np.median(red[red > 0]), 2, 0).astype(np.uint8)  # Catena's codes
        out[:, :3] = 255
        return out

    sets = {s.name: s for s in E.scene_sets(data, ["B", "C"], ocm)}
    first = next(sets["landsat8"].items())
    assert set(first.others) == {"ocm4"} and len(seen) == 1
    red, green, nir = seen[0]
    rho = first.to_rho(first.image[None])[0]
    assert np.allclose(red, rho[3]) and np.allclose(green, rho[2]) and np.allclose(nir, rho[4])
    assert (first.label[:, :8] == 255).all()  # no data there, for both
    assert sets["landsat8"].rgb == (3, 2, 1) and sets["landsat7"].rgb == (2, 1, 0)
    rows, lines = E.score_set(sets["landsat7"], fake(10))
    assert {r["model"] for r in rows} == {"m", "m2", "ocm4"}
    assert any("the labels do not come from it" in line for line in lines)
    assert not E.scene_sets(data, ["B"])[0].note  # without OmniCloudMask, no note

    calls = []

    def predict_from_array(stack, **settings):
        calls.append(settings)
        return np.ones((1, *stack.shape[1:]), np.uint8)  # its code 1: thick cloud

    module = types.ModuleType("omnicloudmask")
    module.predict_from_array = predict_from_array
    monkeypatch.setitem(sys.modules, "omnicloudmask", module)
    red = np.array([[0.2, 0.0], [0.3, 0.1]], np.float32)
    green = np.array([[0.2, 0.0], [0.0, 0.1]], np.float32)
    out = E.omnicloudmask(red, green, red, "cpu")
    assert out.tolist() == [[2, 255], [2, 2]]  # thick in Catena's codes; no data where both 0
    assert calls == [
        {"model_version": 4.0, "no_data_value": 0, "inference_device": "cpu", "batch_size": 4}
    ]


def test_default_out(tmp_path):
    assert E.default_out(tmp_path, list(E.TESTS), False) == tmp_path / "clarity_tests.csv"
    checks = ["honest", "shuffled"]
    assert E.default_out(tmp_path, ["A", "C"], False, checks).name == "clarity_tests_AC_cards.csv"
    two = [tmp_path / "big", tmp_path / "arms"]
    assert E.default_out(two, list(E.TESTS), False) == tmp_path / "arms" / "clarity_tests.csv"
    assert E.default_out(tmp_path, ["E", "D"], False).name == "clarity_tests_DE.csv"
    assert E.default_out(tmp_path, ["B", "C", "D", "E"], True).name == "clarity_tests_BCDE_ocm.csv"
