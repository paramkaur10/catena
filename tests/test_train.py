"""Training: the partial-label loss, the scores, the samples both arms share, and a short
run on a few small made-up patches (CPU, a tiny network)."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import train as T  # noqa: E402
from catena.data import loader  # noqa: E402

FRAME, WINDOW = 128, 120  # 120 divides by 1, 3 and 5, then by 8


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
        assert (s["d_blind"][:, 3:] == 0).all()
        assert (s["d_blind"][:n, :3] == s["d_told"][:n, :3]).all()  # wavelengths never hidden
        assert s["label"].dtype == np.int64 and set(np.unique(s["label"])) <= {0, 1, 2, 3, 255}
        sides.append(side)
    assert all(len(set(sides[b : b + 4])) == 1 for b in range(0, 12, 4))  # one size per batch
    assert (data[5]["image"] == data[5]["image"]).all()  # sample i is always the same


def test_validation_cameras(chips):
    data = T.ValSamples(loader.GridWindows(chips, "validation", size=WINDOW), 2)
    assert len(data) == 2 * len(T.VAL_CAMERAS)
    for batch in data.batches(4):
        assert len({data[i]["camera"] for i in batch}) == 1  # a batch never mixes cameras
    first = data[0]  # Sentinel-2 as delivered: 13 bands, blur, noise and step known
    assert first["real"].all() and (first["d_told"][:, 10:13] == 1).all()
    assert (data[3]["image"] == data[3]["image"]).all()  # the same noise every time


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
