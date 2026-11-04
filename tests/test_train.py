"""Training: the partial-label loss, the scores, the samples both arms share, Landsat 8
batches, and short runs on a few small made-up patches and scenes (CPU, a tiny network)."""

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
from catena import train as T  # noqa: E402
from catena.data import biome, loader  # noqa: E402

FRAME, WINDOW = 128, 120  # 120 divides by 1, 3 and 5, then by 8
L8 = ("LC80000002014001LGN00", "LC80000012014002LGN00")
L8_H, L8_W = 40, 48


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
    """Three training and two validation patches of 128 px, in D2's layout: random
    reflectance between 0.05 and 0.6, random labels."""
    root = tmp_path_factory.mktemp("chips")
    rows = []
    for k, split in enumerate(["train"] * 3 + ["validation"] * 2):
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


def l8_mtl(sun_elevation: float) -> str:
    """The MTL lines biome.py reads, the same for every band."""
    lines = ["GROUP = L1_METADATA_FILE", f"    SUN_ELEVATION = {sun_elevation}"]
    for n in (1, 2, 3, 4, 5, 6, 7, 9):
        lines += [
            f"    RADIANCE_MULT_BAND_{n} = 1.2E-02",
            f"    REFLECTANCE_MULT_BAND_{n} = 2.0000E-05",
            f"    REFLECTANCE_ADD_BAND_{n} = -0.100000",
            f"    REFLECTANCE_MAXIMUM_BAND_{n} = 1.210700",
            f"    REFLECTANCE_MINIMUM_BAND_{n} = -0.099980",
        ]
    return "\n".join([*lines, "END_GROUP = L1_METADATA_FILE", "END"]) + "\n"


@pytest.fixture(scope="module")
def l8(tmp_path_factory):
    """Two made-up Biome scenes of 40 x 48 px without shadow labels, in biome_convert.py's
    layout; pilot.csv holds the second out of training."""
    root = tmp_path_factory.mktemp("l8_biome")
    rows = []
    for k, name in enumerate(L8):
        folder = root / name
        folder.mkdir()
        rng = np.random.default_rng(10 + k)
        dn = rng.integers(10000, 40000, (8, L8_H, L8_W)).astype(np.uint16)
        label = rng.integers(0, 3, (1, L8_H, L8_W)).astype(np.uint8)  # clear, thin, thick
        label[:, :, :8] = 255  # an unlabelled strip
        profile = {
            "driver": "GTiff",
            "height": L8_H,
            "width": L8_W,
            "crs": "EPSG:32633",
            "transform": from_origin(300000, 4650000, 30, 30),
        }
        bands, mask = folder / f"{name}.bands.tif", folder / f"{name}.mask.tif"
        with rasterio.open(bands, "w", count=8, dtype="uint16", **profile) as f:
            f.write(dn)
        with rasterio.open(mask, "w", count=1, dtype="uint8", **profile) as f:
            f.write(label)
        (folder / f"{name}_MTL.txt").write_text(l8_mtl(40.0 + 10 * k))
        rows.append({"scene": name, "biome": "Water", "has_shadow": "False"})
    for table, listed in (("index", rows), ("pilot", rows[1:])):
        with open(root / f"{table}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(listed)
    return root


def test_the_loss_scores_the_allowed_set():
    logits = torch.tensor([0.0, 1.0, 2.0, -1.0]).view(1, 4, 1, 1)
    p = torch.softmax(logits.view(4), dim=0).tolist()

    def loss(code, weights=None):
        return T.set_loss(logits, torch.tensor([[[code]]]), weights).item()

    for code in range(4):  # one allowed class: the usual cross-entropy
        assert loss(code) == pytest.approx(-math.log(p[code]), rel=1e-5)
    assert loss(4) == pytest.approx(-math.log(p[1] + p[2]), rel=1e-5)  # cloud: thin or thick
    assert loss(5) == pytest.approx(-math.log(p[0] + p[3]), rel=1e-5)  # not cloud
    assert loss(3, T.CODE_WEIGHTS) == pytest.approx(2.80 * loss(3), rel=1e-5)
    two = torch.cat([logits, logits], dim=3)  # a second pixel, unlabelled: ignored
    assert T.set_loss(two, torch.tensor([[[2, 255]]])).item() == pytest.approx(loss(2), rel=1e-5)
    assert T.set_loss(logits, torch.tensor([[[255]]])).item() == 0.0


def test_scores_from_a_confusion_matrix():
    conf = np.array([[50, 0, 0, 0], [0, 10, 10, 0], [0, 0, 20, 0], [0, 0, 0, 10]])
    s = T.scores(conf)
    assert s["f1_clear"] == 1.0 and s["f1_shadow"] == 1.0
    assert s["f1_thin"] == pytest.approx(2 * 10 / (10 + 20), abs=1e-4)
    assert s["f1_thick"] == pytest.approx(2 * 20 / (30 + 20), abs=1e-4)
    assert s["cloud_ba"] == 1.0  # thin taken for thick is still cloud
    assert s["accuracy"] == pytest.approx(0.9)


def test_samples_for_both_arms(chips):
    windows = loader.TrainingWindows(chips, "train", size=WINDOW, length=12, seed=0)
    data = T.TrainSamples(windows, batch=4, seed=0)
    sides = []
    for i in range(12):
        s = data[i]
        n, side = int(s["real"].sum()), s["label"].shape[0]
        assert side in (120, 40, 24) and s["image"].shape == (13, side, side)
        assert 3 <= n <= 13 and s["real"][:n].all() and not s["real"][n:].any()
        assert (s["image"][n:] == 0).all() and (s["d_told"][n:] == 0).all()
        assert (s["d_blind"][:, 3 : desc.GSD] == 0).all()  # no chain for blind
        assert (s["d_blind"][:n, :3] == s["d_told"][:n, :3]).all()  # wavelengths never hidden
        assert (s["d_blind"][:, desc.GSD] == s["d_told"][:, desc.GSD]).all()  # nor pixel size
        assert s["label"].dtype == np.int64 and set(np.unique(s["label"])) <= {0, 1, 2, 3, 255}
        sides.append(side)
    assert all(len(set(sides[b : b + 4])) == 1 for b in range(0, 12, 4))  # one size per batch
    assert (data[5]["image"] == data[5]["image"]).all()  # sample i is always the same


def test_validation_cameras(chips):
    data = T.ValSamples(loader.GridWindows(chips, "validation", size=WINDOW), 2)
    assert len(data) == 2 * len(T.VAL_CAMERAS)
    for batch in data.batches(4):
        assert len({data[i]["camera"] for i in batch}) == 1  # a batch never mixes cameras
    first = data[0]  # Sentinel-2 as delivered: 13 bands; noise, step and saturation known
    assert first["real"].all() and (first["d_told"][:, 11:14] == 1).all()
    ten = [loader.BANDS.index(b) for b in ("B02", "B03", "B04", "B08")]
    assert first["d_told"][ten, 10].sum() == 4 == first["d_told"][:, 10].sum()  # blur: 10 m
    assert (data[3]["image"] == data[3]["image"]).all()  # the same noise every time


def test_landsat8_batches(chips, l8):
    windows = loader.TrainingWindows(chips, "train", size=WINDOW, length=160, seed=0)
    held = {s.name for s in biome.scenes(l8, "pilot")}
    chosen = [s for s in biome.scenes(l8, "index") if s.name not in held]
    assert [s.name for s in chosen] == [L8[0]]
    landsat = biome.Windows(chosen, size=16)
    data = T.TrainSamples(windows, batch=4, seed=0, landsat=landsat, simulate=False, jitter=1.0)
    sources = [data.source(i) for i in range(160)]
    assert all(len(set(sources[b : b + 4])) == 1 for b in range(0, 160, 4))  # one per batch
    assert 0.25 < sources.count("landsat8") / 160 < 0.75
    s = data[sources.index("landsat8")]
    n = int(s["real"].sum())
    assert s["image"].shape == (13, 16, 16) and 3 <= n <= 8 and (s["image"][:n] > 0).any()
    assert set(np.unique(s["label"])) <= {1, 2, 5, 255}  # no shadow labels: clear is 5
    assert s["d_blind"][:n, desc.GSD] == pytest.approx(math.log2(3))  # 30 m
    assert (s["d_blind"][:, 3 : desc.GSD] == 0).all()
    s = data[sources.index("sentinel2")]
    assert s["image"].shape[-1] == WINDOW  # no simulation: Sentinel-2 at 10 m as delivered
    assert (s["d_blind"][:, desc.GSD] == 0).all()


def test_a_short_run_with_landsat8(chips, l8, tmp_path):
    out = tmp_path / "run"
    args = [str(chips), str(out), "--steps", "4", "--batch", "2", "--size", str(WINDOW)]
    args += ["--workers", "0", "--widths", "8", "8", "8", "8", "--stem", "8", "--warmup", "2"]
    args += ["--log-every", "2", "--val-every", "4", "--val-windows", "2"]
    args += ["--biome", str(l8), "--landsat-size", "16", "--no-sim", "--jitter", "1"]
    T.main([*args, "--devices", "cpu", "cpu"])
    config = json.loads((out / "config.json").read_text())
    assert config["landsat_scenes"] == [L8[0]] and config["args"]["no_sim"]
    assert len(config["fields"]) == desc.N_FIELDS
    state = torch.load(out / "final.pt", map_location="cpu")
    assert state["step"] == 4
    assert state["models"]["told"]["bands.embed.0.weight"].shape[1] == desc.N_FIELDS


def test_a_short_run(chips, tmp_path):
    out = tmp_path / "run"
    args = [str(chips), str(out), "--steps", "4", "--batch", "2", "--size", str(WINDOW)]
    args += ["--workers", "0", "--widths", "8", "8", "8", "8", "--stem", "8", "--warmup", "2"]
    args += ["--log-every", "2", "--val-every", "4", "--val-windows", "2"]
    T.main([*args, "--devices", "cpu", "cpu"])
    entries = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in entries if e["kind"] == "train"] == [2, 4]
    cameras = {e["camera"] for e in entries if e["kind"] == "val"}
    assert cameras == {c[0] for c in T.VAL_CAMERAS} | {"mean"}
    config = json.loads((out / "config.json").read_text())
    assert config["args"]["steps"] == 4 and config["parameters"] > 0
    state = torch.load(out / "final.pt", map_location="cpu")
    assert state["step"] == 4 and set(state["models"]) == {"told", "blind"}
    told, blind = state["models"]["told"], state["models"]["blind"]
    assert any(not torch.equal(told[k], blind[k]) for k in told)  # same start, other lessons
    with pytest.raises(SystemExit):
        T.main([str(chips), str(out)])  # a finished run is never overwritten
