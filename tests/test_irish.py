"""Landsat 7 (L7 Irish): finding and indexing the scenes, reading one, its reflectance, what the
model is told about it from its gains, and Test C on made-up scenes with untrained networks
(CPU)."""

from __future__ import annotations

import csv
import json
import math
from datetime import date

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena import evaluate_biome as B  # noqa: E402
from catena.data import irish  # noqa: E402
from catena.model import Clarity  # noqa: E402

H, W = 40, 48
LMAX = {  # the handbook's LMAX of B1-B5 and B7, high and low gain, from July 2000
    "H": (191.6, 196.5, 152.9, 157.4, 31.06, 10.80),
    "L": (293.7, 300.9, 234.4, 241.1, 47.57, 16.54),
}
LMIN = (-6.2, -6.4, -5.0, -5.1, -1.0, -0.35)
# zone, folder, gain, a band changing gain, sun elevation, date, shadow labelled
SCENES = (
    ("polar_south", "p10_r120", "H", False, 30.0, "2001-01-04", True),
    ("tropical", "p5_r60", "L", False, 30.0, "2001-01-04", False),
    ("tropical", "p7_r61", "H", True, 60.0, "2001-07-04", False),
)


def mtl(gain: str, change: bool, elevation: float, day: str) -> str:
    """The pre-collection MTL lines irish.py reads; band 3 may switch gain mid-scene."""
    lines = ["GROUP = L1_METADATA_FILE", f"    ACQUISITION_DATE = {day}"]
    for n, hi, lo in zip(("1", "2", "3", "4", "5", "7"), LMAX[gain], LMIN, strict=True):
        lines += [
            f"    LMAX_BAND{n} = {hi}",
            f"    LMIN_BAND{n} = {lo}",
            f"    QCALMAX_BAND{n} = 255.0",
            f"    QCALMIN_BAND{n} = 1.0",
            f'    BAND{n}_GAIN = "{gain}"',
            f'    BAND{n}_GAIN_CHANGE = "{"HL" if change and n == "3" else "0"}"',
        ]
    lines.append(f"    SUN_ELEVATION = {elevation}")
    return "\n".join([*lines, "END_GROUP = L1_METADATA_FILE", "END"]) + "\n"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Three made-up scenes in TorchGeo's layout, with a corner of no data in each."""
    root = tmp_path_factory.mktemp("l7_irish")
    for k, (zone, name, gain, change, elevation, day, shadow) in enumerate(SCENES):
        folder = root / zone / "l7irish" / zone / name
        folder.mkdir(parents=True)
        rng = np.random.default_rng(k)
        image = rng.integers(1, 256, (9, H, W)).astype(np.uint8)
        mask = rng.choice([128, 192, 255] + ([64] if shadow else []), (1, H, W)).astype(np.uint8)
        image[:, :16, :16] = 0
        mask[:, :16, :16] = 0
        profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "dtype": "uint8",
            "crs": "EPSG:32605",
            "transform": from_origin(300000, 4650000, 30, 30),
        }
        stem = f"L71{name}_{name}{day.replace('-', '')}"
        with rasterio.open(folder / f"{stem}.TIF", "w", count=9, **profile) as f:
            f.write(image)
        with rasterio.open(folder / f"L7_{name}_newmask2015.TIF", "w", count=1, **profile) as f:
            f.write(mask)
        (folder / f"{stem}_MTL.txt").write_text(mtl(gain, change, elevation, day))
    (root / "polar_south" / ".done").touch()
    irish.main([str(root)])
    return root


def test_index_and_pilot(root):
    with open(root / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["zone"], r["scene"]) for r in rows] == [(z, n) for z, n, *_ in SCENES]
    assert [r["has_shadow"] for r in rows] == ["True", "False", "False"]
    assert [r["gains"] for r in rows] == ["HHHHHH", "LLLLLL", "HHHHHH"]
    assert [r["gain_change"] for r in rows] == ["False", "False", "True"]
    assert all(r["matched"] == "True" for r in rows)
    labelled = (H * W - 16 * 16) / (H * W)
    assert all(float(r["labelled"]) == pytest.approx(labelled, abs=1e-4) for r in rows)
    picked = [s.name for s in irish.scenes(root, "pilot")]
    assert picked == ["p10_r120", "p5_r60"]  # the scene whose gain changed is left out


def test_reading_a_scene(root):
    scene = irish.scenes(root, "index")[0]
    assert scene.zone == scene.biome == "polar_south" and scene.has_shadow
    assert scene.distance_au == pytest.approx(irish.earth_sun_distance(date(2001, 1, 4)))
    assert scene.distance_au == pytest.approx(0.98328, abs=1e-5)  # perihelion
    dn, label = irish.read(scene)
    assert dn.shape == (6, H, W) and dn.dtype == np.uint8 and label.shape == (H, W)
    assert set(np.unique(label).tolist()) == {0, 1, 2, 3, 255}
    assert (label[:16, :16] == 255).all()
    with rasterio.open(scene.image) as src:
        assert np.array_equal(dn[5], src.read(8))  # B7 is the 8th band of the image
    rho = irish.reflectance(scene, dn)
    assert rho.dtype == np.float32 and (rho[:, :16, :16] == 0).all()
    q = int(dn[0, 20, 30])
    radiance = (191.6 + 6.2) / 254 * (q - 1) - 6.2
    expected = math.pi * radiance * scene.distance_au**2 / (1997 * math.sin(math.radians(30)))
    assert rho[0, 20, 30] == pytest.approx(expected, rel=1e-5)
    assert np.array_equal(irish.reflectance(scene, dn[None])[0], rho)  # tiles: the same


def test_what_landsat_7_is_told(root):
    high, low, _ = irish.scenes(root, "index")  # the same sun and date, high and low gain
    d = irish.descriptors(high)
    assert d.shape == (len(irish.BANDS), desc.N_FIELDS)
    assert d[0, 0] == pytest.approx(desc.wavelength_code(450))  # B1 starts at 450 nm
    assert d[5, 2] == pytest.approx(desc.wavelength_code(2350))  # B7 ends at 2350 nm
    assert d[:, desc.GSD] == pytest.approx(math.log2(3)) and (d[:, 4] == 0).all()  # 30 m
    assert (d[:, 10] == 0).all() and (d[:, 11] == 0).all()  # blur and noise: unknown
    assert (d[:, 12:14] == 1).all()  # step and saturation: known, from the gains
    k = math.pi * high.distance_au**2 / (230.8 * math.sin(math.radians(30)))  # B5
    assert d[4, 9] == pytest.approx(k * 31.06, rel=1e-5)  # high gain: LMAX 31.06
    assert d[4, 8] == pytest.approx(math.log10(k * (31.06 + 1.0) / 254) + 4, rel=1e-5)
    low_d = irish.descriptors(low)
    assert (low_d[:, 9] >= d[:, 9]).all() and low_d[4, 9] > d[4, 9]  # low gain saturates later
    assert (low_d[:, 8] > d[:, 8]).all()  # and its 254 levels are coarser


def tiny_run(folder, seed: int) -> None:
    """A run folder as catena.train leaves one, with untrained networks."""
    torch.manual_seed(seed)
    folder.mkdir(parents=True)
    nets = {arm: Clarity(widths=(8, 8, 8, 8), stem=8) for arm in ("told", "blind")}
    (folder / "config.json").write_text(json.dumps({"args": {"widths": [8] * 4, "stem": 8}}))
    state = {"step": 0, "models": {arm: net.state_dict() for arm, net in nets.items()}}
    torch.save(state, folder / "final.pt")


def test_c_on_the_scenes(root, tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        tiny_run(runs / f"seed{seed}", seed)
    args = [str(runs), str(root), "--source", "irish", "--table", "index", "--sizes", "16"]
    B.main([*args, "--device", "cpu", "--variants", "no top"])
    with open(runs / "test_c.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 3 * len(SCENES)  # seeds, (told honest, told no top, blind), scenes
    assert {r["biome"] for r in rows} == {"polar_south", "tropical"}
    for name in {r["scene"] for r in rows}:  # every arm, seed and variant on the same pixels
        totals = {sum(map(int, r["confusion"].split())) for r in rows if r["scene"] == name}
        assert len(totals) == 1 and totals.pop() > 0
    out = capsys.readouterr().out
    assert "Landsat 7 scenes" in out and "per zone" in out and "all 3 scenes" in out
