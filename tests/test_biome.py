"""Landsat 8 Biome: reading a converted scene, its reflectance, what the model is told about
it, its tiles, and Test B on two made-up scenes with untrained networks (CPU)."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena import evaluate_biome as B  # noqa: E402
from catena.data import biome  # noqa: E402
from catena.model import CatenaNet  # noqa: E402

H, W = 40, 48
RADIANCE_MULT = 1.2e-2
NAMES = ("LC80000002014001LGN00", "LC80000012014002LGN00")


def mtl(sun_elevation: float) -> str:
    """The MTL lines biome.py reads, the same for every band."""
    lines = ["GROUP = L1_METADATA_FILE", f"    SUN_ELEVATION = {sun_elevation}"]
    for n in (1, 2, 3, 4, 5, 6, 7, 9):
        lines += [
            f"    RADIANCE_MULT_BAND_{n} = {RADIANCE_MULT}",
            f"    REFLECTANCE_MULT_BAND_{n} = 2.0000E-05",
            f"    REFLECTANCE_ADD_BAND_{n} = -0.100000",
            f"    REFLECTANCE_MAXIMUM_BAND_{n} = 1.210700",
            f"    REFLECTANCE_MINIMUM_BAND_{n} = -0.099980",
        ]
    return "\n".join([*lines, "END_GROUP = L1_METADATA_FILE", "END"]) + "\n"


def write_scene(root, name, biome_name, has_shadow, sun_elevation, seed) -> dict:
    """A scene in biome_convert.py's layout: random DN and labels, and a corner tile of
    16 x 16 pixels with no data, as at the edge of a Landsat scene."""
    folder = root / name
    folder.mkdir()
    rng = np.random.default_rng(seed)
    dn = rng.integers(10000, 40000, (8, H, W)).astype(np.uint16)
    label = rng.integers(0, 4 if has_shadow else 3, (H, W)).astype(np.uint8)
    dn[:, :16, :16] = 0
    label[:16, :16] = 255
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "crs": "EPSG:32633",
        "transform": from_origin(300000, 4650000, 30, 30),
    }
    with rasterio.open(folder / f"{name}.bands.tif", "w", count=8, dtype="uint16", **profile) as f:
        f.write(dn)
    with rasterio.open(folder / f"{name}.mask.tif", "w", count=1, dtype="uint8", **profile) as f:
        f.write(label[None])
    (folder / f"{name}_MTL.txt").write_text(mtl(sun_elevation))
    return {
        "scene": name,
        "biome": biome_name,
        "sun_elevation": sun_elevation,
        "has_shadow": has_shadow,
    }


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Two scenes: snow with shadow labels and the sun 45 degrees up, water without shadow
    labels and the sun 30 degrees up."""
    root = tmp_path_factory.mktemp("l8_biome")
    rows = [
        write_scene(root, NAMES[0], "Snow/Ice", True, 45.0, 0),
        write_scene(root, NAMES[1], "Water", False, 30.0, 1),
    ]
    with open(root / "pilot.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_reading_a_scene(root):
    found = biome.scenes(root, "pilot")
    assert [s.name for s in found] == list(NAMES)
    assert [s.biome for s in found] == ["Snow/Ice", "Water"]
    assert [s.has_shadow for s in found] == [True, False]
    scene = found[0]
    dn, label = biome.read(scene)
    assert dn.shape == (8, H, W) and dn.dtype == np.uint16 and label.shape == (H, W)
    rho = biome.reflectance(scene, dn)
    assert rho.dtype == np.float32 and (rho[:, :16, :16] == 0).all()  # no data stays 0
    expected = (int(dn[3, 20, 30]) * 2e-5 - 0.1) / math.sin(math.radians(45))
    assert rho[3, 20, 30] == pytest.approx(expected, rel=1e-5)
    assert np.array_equal(biome.reflectance(scene, dn[None])[0], rho)  # tiles: the same


def test_a_scene_needs_its_mtl_numbers(tmp_path):
    folder = tmp_path / "LC8broken"
    folder.mkdir()
    (folder / "LC8broken_MTL.txt").write_text("SUN_ELEVATION = 40.0\n")
    with pytest.raises(ValueError, match="REFLECTANCE_MULT_BAND_1"):
        biome.open_scene(folder, "Water", False)


def test_what_landsat_8_is_told(root):
    high, low = biome.scenes(root, "pilot")  # the sun 45 and 30 degrees above the horizon
    d = biome.descriptors(high)
    assert d.shape == (8, desc.N_FIELDS)
    assert d[0, 0] == pytest.approx(desc.wavelength_code(435))  # B1 starts at 435 nm
    assert d[7, 2] == pytest.approx(desc.wavelength_code(1384))  # B9 ends at 1384 nm
    assert (d[:, 4] == 0).all()  # 30 m bands on a 30 m grid
    assert (d[:, 5] == 0).all() and (d[:, 10] == 0).all()  # blur not published: unknown
    assert (d[:, 11:14] == 1).all()  # noise, step and saturation known
    step = (1.2107 + 0.09998) / 4095 / math.sin(math.radians(45))
    assert d[0, 8] == pytest.approx(math.log10(step) + 4, rel=1e-5)
    k = 2e-5 / (RADIANCE_MULT * math.sin(math.radians(45)))  # reflectance of 1 W m-2 sr-1 um-1
    assert d[4, 9] == pytest.approx(k * 500, rel=1e-5)  # B5 saturates at 500 W m-2 sr-1 um-1
    assert d[0, 9] == 2.0  # B1's saturation, above 2 here, is capped
    # with the sun lower, the same radiance noise is more reflectance noise
    assert (biome.descriptors(low)[:, 6:8] > d[:, 6:8]).all()


def test_tiles_keep_every_labelled_pixel_once(root):
    scene = biome.scenes(root, "pilot")[0]
    dn, label = biome.read(scene)
    x, y = B.tiles(dn, label, 16)
    # 40 x 48 holds 2 x 3 tiles of 16; the corner one has no label and is left out
    assert x.shape == (5, 8, 16, 16) and y.shape == (5, 16, 16)
    assert (y != 255).sum() == (label[:32, :48] != 255).sum()
    assert np.array_equal(x[0], dn[:, 0:16, 16:32]) and np.array_equal(y[0], label[0:16, 16:32])


def test_what_the_told_arm_is_shown():
    one = desc.describe(450, 520, 30, 30, None, (1e-4, 1e-6), 1e-3, 1.2)
    d = torch.from_numpy(np.stack([one, one]))
    assert torch.equal(B.told_variant(d, "honest"), d)
    quiet = B.told_variant(d, "no noise")
    assert (quiet[:, 6:8] == 0).all() and (quiet[:, 11] == 0).all()
    assert torch.equal(quiet[:, 8:10], d[:, 8:10]) and (quiet[:, 12:14] == 1).all()
    hidden = B.told_variant(d, "hidden")
    assert torch.equal(hidden[:, :5], d[:, :5]) and (hidden[:, 5:] == 0).all()
    no_step = B.told_variant(d, "no step")  # columns 8 and 12: the step and its flag
    assert (no_step[:, [8, 12]] == 0).all()
    assert torch.equal(no_step[:, [6, 7, 9, 11, 13]], d[:, [6, 7, 9, 11, 13]])
    no_top = B.told_variant(d, "no top")  # columns 9 and 13: the saturation and its flag
    assert (no_top[:, [9, 13]] == 0).all()
    assert torch.equal(no_top[:, [6, 7, 8, 11, 12]], d[:, [6, 7, 8, 11, 12]])
    fine = B.told_variant(d, "fine step")
    assert float(d[0, 8]) == pytest.approx(1.0)  # told 0.001
    assert fine[:, 8].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)  # told 0.0001
    assert (fine[:, 12] == 1).all()
    others = [c for c in range(desc.N_FIELDS) if c != 8]
    assert torch.equal(fine[:, others], d[:, others])
    with pytest.raises(ValueError, match="unknown variant"):
        B.told_variant(d, "blur x2")


def tiny_run(folder, seed: int) -> None:
    """A run folder as catena.train leaves one, with untrained networks."""
    torch.manual_seed(seed)
    folder.mkdir(parents=True)
    nets = {arm: CatenaNet(widths=(8, 8, 8, 8), stem=8) for arm in ("told", "blind")}
    (folder / "config.json").write_text(json.dumps({"args": {"widths": [8] * 4, "stem": 8}}))
    state = {"step": 0, "models": {arm: net.state_dict() for arm, net in nets.items()}}
    torch.save(state, folder / "final.pt")


def test_b_on_two_scenes(root, tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        tiny_run(runs / f"seed{seed}", seed)
    B.main([str(runs), str(root), "--sizes", "16", "8", "--device", "cpu"])
    with open(runs / "test_b.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * (len(B.VARIANTS) + 1) * len(NAMES) * 2
    for name in NAMES:
        for size in ("16", "8"):  # every arm, seed and variant scored on the same pixels
            same = [r for r in rows if r["scene"] == name and r["size"] == size]
            totals = {sum(map(int, r["confusion"].split())) for r in same}
            assert len(same) == 8 and len(totals) == 1 and totals.pop() > 0
    out = capsys.readouterr().out
    assert "Told minus blind, cloud BA" in out and "Snow/Ice" in out and "all 2 scenes" in out

    # other variants, into another CSV: honest is added, and scores as in the first run
    other = tmp_path / "fields.csv"
    args = [str(runs), str(root), "--sizes", "16", "--device", "cpu", "--out", str(other)]
    B.main([*args, "--variants", "no step", "fine step"])
    with open(other, newline="") as f:
        again = list(csv.DictReader(f))
    shown = {(r["arm"], r["variant"]) for r in again}
    told = {("told", v) for v in ("honest", "no step", "fine step")}
    assert shown == told | {("blind", "honest")} and len(again) == 2 * 4 * len(NAMES)
    assert honest_counts(again, "16") == honest_counts(rows, "16")
    assert "told fine step" in capsys.readouterr().out


def honest_counts(rows: list[dict], size: str) -> dict:
    """The told arm's honest confusion matrix per seed and scene, for one tile size."""
    keep = [r for r in rows if (r["arm"], r["variant"], r["size"]) == ("told", "honest", size)]
    return {(r["seed"], r["scene"]): r["confusion"] for r in keep}
