"""Test A: the cameras, what the told arm is shown, and a short evaluation of two tiny runs
on a few made-up patches (CPU)."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import descriptor as desc  # noqa: E402
from catena import evaluate as E  # noqa: E402
from catena import train as T  # noqa: E402

FRAME = 128


def geotiff(path, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype.name,
        "crs": "EPSG:32633",
        "transform": from_origin(300000, 4650000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.fixture(scope="module")
def chips(tmp_path_factory):
    """Two patches of 128 px per split, in D2's layout: random reflectance between 0.05 and
    0.6, random labels."""
    root = tmp_path_factory.mktemp("chips")
    rows = []
    for k, split in enumerate(["train", "train", "validation", "validation", "test", "test"]):
        rng = np.random.default_rng(k)
        name = f"{split}{k}"
        (root / split).mkdir(exist_ok=True)
        image = rng.integers(500, 6000, (13, FRAME, FRAME)).astype(np.uint16)
        label = rng.integers(0, 4, (1, FRAME, FRAME)).astype(np.uint8)
        geotiff(root / split / f"{name}.s2l1c.tif", image)
        geotiff(root / split / f"{name}.target.tif", label)
        rows.append(
            {
                "id": name,
                "split": split,
                "s2l1c_file": f"{split}/{name}.s2l1c.tif",
                "target_file": f"{split}/{name}.target.tif",
                "height": FRAME,
                "width": FRAME,
                "baseline": "02.07",
                "sun_zenith": "35.5",
            }
        )
    with open(root / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_the_cameras():
    names = [c[0] for c in E.TEST_CAMERAS]
    assert len(names) == len(set(names)) == 23 and names[0] == "sentinel2"
    inside = [c for c in E.TEST_CAMERAS if c[0].startswith("rgbn_")]
    assert len(inside) == 18
    for _, q, bands, chain in inside:  # all inside what imagine_camera draws
        assert q in (1, 3) and bands == E.RGBN
        assert 0.04 <= chain["mtf"] <= 0.48 and 7 <= chain["snr"] <= 390
        assert chain["bits"] in (8, 10, 12, 14) and 0.6 <= chain["top"] <= 1.6
    beyond = {c[0]: c[3] for c in E.TEST_CAMERAS if c[0].startswith("beyond_")}
    assert set(beyond) == {"beyond_mtf0.002", "beyond_snr4", "beyond_bits6", "beyond_top0.45"}
    for chain in beyond.values():  # exactly one property beyond the training range
        outside = [chain["mtf"] < 0.04, chain["snr"] < 7, chain["bits"] < 8, chain["top"] < 0.6]
        assert sum(outside) == 1


def test_what_the_told_arm_is_shown():
    one = desc.describe(450, 520, 10, 10, 0.3, (1e-4, 1e-6), 1e-3, 1.2)
    d = torch.zeros(2, 3, desc.N_FIELDS)
    d[0, :2] = torch.from_numpy(np.stack([one, one]))  # sample 0: 2 bands and an empty slot
    honest = E.told_variant(d, "honest")
    assert torch.equal(honest, d) and honest is not d
    hidden = E.told_variant(d, "hidden")  # wavelengths, surface and pixel size stay
    assert torch.equal(hidden[..., :5], d[..., :5]) and (hidden[..., 5:] == 0).all()
    louder = E.told_variant(d, "noise x3")
    assert torch.allclose(louder[0, :2, 6:8], d[0, :2, 6:8] + math.log10(3))
    assert torch.equal(louder[..., :6], d[..., :6]) and torch.equal(louder[..., 8:], d[..., 8:])
    assert (louder[0, 2] == 0).all() and (louder[1] == 0).all()  # empty slots stay empty
    quieter = E.told_variant(d, "noise /3")
    assert torch.allclose(quieter[0, :2, 6:8], d[0, :2, 6:8] - math.log10(3))
    wider = E.told_variant(d, "blur x2")
    assert torch.allclose(wider[0, :2, 5], 2 * d[0, :2, 5]) and d[0, 0, 5] > 0
    assert torch.equal(wider[..., 6:], d[..., 6:])
    unknown = d.clone()
    unknown[..., 6:8] = 0.0
    unknown[..., 11] = 0.0  # noise not known: saying it louder changes nothing
    assert torch.equal(E.told_variant(unknown, "noise x3"), unknown)
    with pytest.raises(ValueError, match="unknown variant"):
        E.told_variant(d, "noise x2")


def test_sizes_must_suit_every_camera():
    assert E.parse_args(["runs", "chips"]).sizes == [240, 120, 48]
    with pytest.raises(SystemExit):
        E.parse_args(["runs", "chips", "--sizes", "60"])  # 60 / 3 = 20: not a multiple of 8
    with pytest.raises(SystemExit):
        E.parse_args(["runs", "chips", "--cameras", "no_such_camera"])
    args = E.parse_args(["runs", "chips", "--sizes", "40", "--cameras", "sentinel2"])
    assert args.sizes == [40]  # fine for a 10 m camera alone


def test_two_tiny_runs(chips, tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        args = [str(chips), str(runs / f"seed{seed}"), "--seed", str(seed), "--steps", "2"]
        args += ["--batch", "2", "--size", "120", "--workers", "0", "--stem", "8"]
        args += ["--widths", "8", "8", "8", "8", "--warmup", "1", "--log-every", "2"]
        T.main([*args, "--val-every", "2", "--val-windows", "1", "--devices", "cpu", "cpu"])
    cameras = ["sentinel2", "rgbn_30m_mtf0.15_snr40", "beyond_snr4"]
    args = [str(runs), str(chips), "--windows", "3", "--sizes", "48", "--device", "cpu"]
    E.main([*args, "--workers", "0", "--cameras", *cameras])
    with open(runs / "test_a.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * (len(E.VARIANTS) + 1) * len(cameras)
    assert {r["camera"] for r in rows} == set(cameras)
    assert {r["seed"] for r in rows} == {"seed0", "seed1"}
    for r in rows:
        counts = [int(v) for v in r["confusion"].split()]
        assert len(counts) == 16 and sum(counts) > 0
        assert 0.0 <= float(r["cloud_ba"]) <= 1.0 and r["size"] == "48"
    for camera in cameras:  # every arm, seed and variant was scored on the same pixels
        totals = {sum(map(int, r["confusion"].split())) for r in rows if r["camera"] == camera}
        assert len(totals) == 1
    out = capsys.readouterr().out
    assert "Does the told arm read its descriptor?" in out
    assert "mean, inside the range" in out and "mean, beyond the range" in out
