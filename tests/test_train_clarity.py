"""The big run: samples from the four satellites for both versions, the real version
untouched by the made-up cameras, what the arms add drawn apart, and short runs on made-up
data (CPU, a tiny network), stopped and gone on with, which end where an unbroken run ends."""

from __future__ import annotations

import csv
import json
import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena import train_clarity as C  # noqa: E402
from catena.data import biome, heo, loader, perusat  # noqa: E402

FRAME, WINDOW, SMALL = 128, 120, 16  # Sentinel-2 patches and windows; the others' windows
L8 = ("LC80000002014001LGN00", "LC80000012014002LGN00")
HEO_SCENE = "IMH12_20261008T090908"
QUARTERS = (0.25, 0.25, 0.25, 0.25)


def geotiff(path, data: np.ndarray, res: float = 10.0) -> None:
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype.name,
        "crs": "EPSG:32633",
        "transform": from_origin(300000, 4650000, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sentinel2(root) -> None:
    """Three training and two validation patches of 128 px in D2's layout."""
    rows = []
    for k, split in enumerate(["train"] * 3 + ["validation"] * 2):
        rng = np.random.default_rng(k)
        name = f"{split}{k}"
        (root / split).mkdir(parents=True, exist_ok=True)
        image = rng.integers(500, 6000, (13, FRAME, FRAME)).astype(np.uint16)
        geotiff(root / split / f"{name}.s2l1c.tif", image)
        geotiff(
            root / split / f"{name}.target.tif", rng.integers(0, 4, (FRAME, FRAME)).astype(np.uint8)
        )
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
    write_csv(root / "index.csv", rows)


def landsat8(root) -> None:
    """Two Biome scenes of 40 x 48 px without shadow labels; pilot.csv holds the second."""
    rows = []
    for k, name in enumerate(L8):
        folder = root / name
        folder.mkdir(parents=True)
        rng = np.random.default_rng(10 + k)
        dn = rng.integers(10000, 40000, (8, 40, 48)).astype(np.uint16)
        geotiff(folder / f"{name}.bands.tif", dn, 30)
        label = rng.integers(0, 3, (40, 48)).astype(np.uint8)
        label[:, :8] = 255
        geotiff(folder / f"{name}.mask.tif", label, 30)
        lines = ["GROUP = L1_METADATA_FILE", f"    SUN_ELEVATION = {40.0 + 10 * k}"]
        for n in (1, 2, 3, 4, 5, 6, 7, 9):
            lines += [
                f"    RADIANCE_MULT_BAND_{n} = 1.2E-02",
                f"    REFLECTANCE_MULT_BAND_{n} = 2.0000E-05",
                f"    REFLECTANCE_ADD_BAND_{n} = -0.100000",
                f"    REFLECTANCE_MAXIMUM_BAND_{n} = 1.210700",
                f"    REFLECTANCE_MINIMUM_BAND_{n} = -0.099980",
            ]
        (folder / f"{name}_MTL.txt").write_text("\n".join([*lines, "END"]) + "\n")
        rows.append({"scene": name, "biome": "Water", "has_shadow": "False"})
    write_csv(root / "index.csv", rows)
    write_csv(root / "pilot.csv", rows[1:])


def iride_heo(data) -> None:
    """One HEO train scene of 40 x 40 px at 5 m, already converted."""
    labels, chips = data / "labels/iride_heo", data / "chips/iride_heo"
    (labels / HEO_SCENE).mkdir(parents=True)
    chips.mkdir(parents=True)
    rng = np.random.default_rng(20)
    label = rng.integers(0, 4, (40, 40)).astype(np.uint8)
    label[:, -4:] = 255
    geotiff(labels / HEO_SCENE / "label.tif", label, 5)
    dn = rng.integers(300, 3000, (8, 40, 40)).astype(np.uint16)
    geotiff(chips / f"{HEO_SCENE}.bands.tif", dn, 5)
    row = {"scene": HEO_SCENE, "product": "x.zip", "role": "train", "verdict": "correct"}
    write_csv(labels / "scenes.csv", [row])


def cloudperu2(root) -> None:
    """Two train patches and one validation patch of 32 px, cloud in the top half."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rows = []
    for n, split in ((1, "train"), (2, "train"), (3, "validation")):
        image = np.random.default_rng(30 + n).integers(10, 250, (4, 32, 32)).astype(np.uint16)
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
    """A DATA folder in the layout the big run reads."""
    root = tmp_path_factory.mktemp("catena_data")
    sentinel2(root / "chips/cloudsen12plus")
    landsat8(root / "chips/l8_biome")
    iride_heo(root)
    cloudperu2(root / "chips/cloudperu2")
    return root


def samples(data, length=240, **kwargs) -> C.Samples:
    windows = loader.TrainingWindows(
        data / "chips/cloudsen12plus", "train", size=WINDOW, length=length, seed=0
    )
    l8 = data / "chips/l8_biome"
    held = {s.name for s in biome.scenes(l8, "pilot")}
    chosen = [s for s in biome.scenes(l8, "index") if s.name not in held]
    others = {
        "landsat": biome.Windows(chosen, size=SMALL),
        "heo_windows": heo.Windows(heo.scenes(data, ("train",)), size=SMALL),
        "peru": perusat.Windows(data / "chips/cloudperu2", "train", size=SMALL),
    }
    return C.Samples(windows, 4, 0, **(others | kwargs))


def test_samples_of_every_satellite(data):
    s = samples(data, shares=QUARTERS)
    sources = [s.source(i) for i in range(240)]
    assert all(len(set(sources[b : b + 4])) == 1 for b in range(0, 240, 4))  # one per batch
    assert set(sources) == set(C.SOURCES)
    for source in C.SOURCES:
        i = sources.index(source)
        x = s[i]
        n = int(x["real"].sum())
        assert x["told"].shape == (13, 11) and x["label"].dtype == np.int64
        assert x["sim_told"].shape == (13, 11) and "fixed_image" not in x
        assert (x["image"][n:] == 0).all() and (x["told"][n:] == 0).all()
        assert not x["real"][n:].any() and C.SOURCES[x["source"]] == source
        labels = set(np.unique(x["label"]).tolist())
        if source == "sentinel2":
            q = s.pixel_size(i)
            assert 3 <= n <= 13 and x["image"].shape == (13, WINDOW, WINDOW) and not x["same"]
            assert x["sim_image"].shape == (13, WINDOW // q, WINDOW // q)
            assert x["sim_label"].shape == (WINDOW // q, WINDOW // q)
            assert (x["told"][:n, 5] == 0).all()  # 10 m
            assert x["sim_told"][:n, 5] == pytest.approx(math.log2(q))
            assert labels <= {0, 1, 2, 3, 255}
            continue
        assert x["same"] and x["image"].shape == (13, SMALL, SMALL)
        if source == "landsat8":
            assert 3 <= n <= 8 and x["told"][:n, 5] == pytest.approx(math.log2(3))
            assert labels <= {1, 2, 5, 255}  # no shadow labels: clear is 5
        elif source == "heo":
            assert 3 <= n <= 8 and x["told"][:n, 5] == pytest.approx(-1.0)
            assert (x["told"][:n, 3] == 0).all() and labels <= {0, 1, 2, 3, 255}
        else:
            assert n in (3, 4) and (x["told"][:n, 3] == 1).all()  # surface reflectance
            assert x["told"][:n, 5] == pytest.approx(math.log2(0.28))
            assert labels <= {4, 5, 255}
            top = 1.5 * 1.05 * 250 / 255
            assert (x["image"][:n] > 0).all() and (x["image"][:n] <= top + 1e-6).all()
    again = s[sources.index("sentinel2")]
    first = s[sources.index("sentinel2")]
    for key in ("image", "sim_image", "told", "sim_told", "label", "sim_label"):
        assert np.array_equal(again[key], first[key])  # sample i is always the same
    with pytest.raises(ValueError, match="landsat8 has a share"):
        C.Samples(s.windows, 4, 0, shares=(0.5, 0.5, 0, 0))


def test_the_real_version_never_sees_a_camera(data):
    keep_all = samples(data, shares=(1, 0, 0, 0), p_s2=1.0)  # sim: every 10 m sample kept
    keep_none = samples(data, shares=(1, 0, 0, 0), p_s2=0.0)  # sim: a camera for every one
    for i in range(0, 240, 4):
        a, b = keep_all[i], keep_none[i]
        for key in ("image", "told", "label", "real"):
            assert np.array_equal(a[key], b[key])  # the real version: the same either way
        if keep_all.pixel_size(i) == 1:
            assert np.array_equal(a["sim_image"], a["image"])  # kept as delivered
            assert not np.array_equal(b["sim_image"], b["image"])  # through a camera


def test_the_sim_options_leave_the_real_version_alone(data):
    plain = samples(data, shares=QUARTERS)
    other = samples(data, shares=QUARTERS, q_weights=(1, 0, 0), faults=1.0)
    for i in range(0, 240, 4):
        a, b = plain[i], other[i]
        for key in ("image", "told", "label", "real"):
            assert np.array_equal(a[key], b[key])  # the real version: the same either way
        assert other.pixel_size(i) == 1 and not b["same"]  # 10 m; its own copy everywhere
        assert b["sim_image"].shape == b["image"].shape
        assert np.array_equal(b["sim_label"], b["label"])
        assert np.array_equal(b["sim_told"][:, :6], b["told"][:, :6])  # a fault is not on the card
        n = int(b["real"].sum())
        assert not np.array_equal(b["sim_image"][:n], b["image"][:n])  # every sample faulty
        assert (b["sim_image"][n:] == 0).all()
    equal = samples(data, shares=QUARTERS, q_weights=(2, 2, 2))
    sizes = [plain.pixel_size(i) for i in range(0, 400, 4)]
    assert [equal.pixel_size(i) for i in range(0, 400, 4)] == sizes  # equal: the first run's
    tens = samples(data, shares=QUARTERS, q_weights=(0.5, 0.3, 0.2))
    many = [tens.pixel_size(i) for i in range(0, 4000, 4)]
    assert 0.42 < many.count(1) / len(many) < 0.58 and set(many) == {1, 3, 5}
    with pytest.raises(ValueError, match="q_weights"):
        samples(data, q_weights=(1, 1))
    some = samples(data, shares=(1, 0, 0, 0), q_weights=(1, 0, 0), p_s2=1.0, faults=0.5)
    looks = [np.array_equal(some[i]["sim_image"], some[i]["image"]) for i in range(0, 160, 4)]
    assert 0 < sum(looks) < len(looks)  # half the samples faulty, the others as delivered


def test_going_on_draws_the_same_samples(data):
    whole, later = samples(data, shares=QUARTERS), samples(data, shares=QUARTERS, start=8)
    assert len(later) == len(whole) - 8
    for j in (0, 5):
        for key, value in later[j].items():
            assert np.array_equal(value, whole[8 + j][key])


def test_what_the_arms_add_is_drawn_apart(data):
    plain = samples(data, shares=QUARTERS, chain_jitter=0.0)
    arms = samples(data, shares=QUARTERS, fixed=True, chain_jitter=0.0)
    hidden = kept = 0
    for i in range(240):
        a, b = plain[i], arms[i]
        for key in ("image", "told", "label", "real", "sim_image", "sim_label", "same"):
            assert np.array_equal(a[key], b[key])  # the fixed arm's bands: drawn apart
        assert np.array_equal(a["sim_told"], b["sim_told"])  # dropout and errors: drawn apart
        n = int(b["real"].sum())
        assert np.array_equal(b["sim_told"][:, :6], b["told"][:, :6]) or not b["same"]
        x, same = b["fixed_image"], bool(b["same"])
        label = b["label"] if same else b["sim_label"]
        assert x.shape == (4, *label.shape) and x.dtype == np.float32 and np.isfinite(x).all()
        source = C.SOURCES[b["source"]]
        if source != "sentinel2":  # its bands as delivered, turned as the others are
            assert (x > 0).all()
        if source == "landsat8":  # Landsat 8's noise is known: dropout hides it now and then
            told, sim_told = b["told"][:n], b["sim_told"][:n]
            assert (told[:, 10] == 1).all() and (told[:, 9] == 0).all()  # SNR yes, MTF no
            hidden += int((sim_told[:, 10] == 0).all())
            kept += int((sim_told[:, 7:9] == told[:, 7:9]).all() and (sim_told[:, 10] == 1).all())
    assert hidden > 0 and kept > 0  # hidden in some samples, as it was in the others (no errors)
    moved = samples(data, shares=QUARTERS, chain_jitter=1.0)
    changed = 0
    for i in range(240):
        a, b = arms[i], moved[i]
        assert np.array_equal(a["sim_told"][:, :6], b["sim_told"][:, :6])  # never the card
        changed += int(not np.array_equal(a["sim_told"], b["sim_told"]))
    assert changed > 0  # errors move the known MTF and SNR
    with pytest.raises(ValueError, match="no faults"):
        samples(data, fixed=True, faults=0.5)


def test_the_arms():
    assert set(C.ARMS) == {"real", "sim", "sensei", "chain", "blind", "fixed", "nas", "nas2"}
    assert C.NAS == ("nas", "nas2") and C.codes_of(["nas", "sim"], ["L"]) == {"nas": "L"}
    assert C.ARMS["sensei"].fields == 4 and C.ARMS["chain"].fields == 11  # their first numbers
    assert C.ARMS["fixed"].fields == 0 and C.ARMS["blind"].blind
    assert all(a.data == "sim" for k, a in C.ARMS.items() if k != "real")
    assert C.rgbn_slots(loader.BANDS) == [1, 2, 3, 7]
    assert C.rgbn_slots(("B01", "B02", "B03", "B04", "B8A", "B10", "B11", "B12")) == [1, 2, 3, 4]
    for bad in (["--arms", "sim", "sim"], ["--arms", "fixed", "sim", "--faults", "0.2"]):
        with pytest.raises(SystemExit):
            C.parse_args(["x", *bad])
    with pytest.raises(SystemExit):
        C.parse_args(["x", "--arms", "chain", "other"])


def run_args(data, out, *more) -> list[str]:
    args = [str(out), "--data", str(data), "--steps", "4", "--batch", "2"]
    args += ["--size", str(WINDOW), "--workers", "0", "--widths", "8", "8", "8", "8"]
    args += ["--stem", "8", "--warmup", "2", "--log-every", "2", "--val-every", "2"]
    args += ["--val-windows", "2", "--landsat-size", "16", "--heo-size", "16"]
    args += ["--perusat-size", "16", "--devices", "cpu", "cpu"]
    return args + list(more)


def test_a_short_run_stopped_and_gone_on_with(data, tmp_path):
    C.main(run_args(data, tmp_path / "whole"))
    config = json.loads((tmp_path / "whole/config.json").read_text())
    assert config["card"] == ["lo", "mid", "hi", "surface", "sampling", "gsd"]
    assert [r["source"] for r in config["sources"]] == list(C.SOURCES)
    assert config["heo_scenes"] == [HEO_SCENE] and config["landsat_scenes"] == [L8[0]]
    assert config["made_up_cameras"]["share_of_sentinel2"] == 0.8
    entries = [json.loads(x) for x in (tmp_path / "whole/metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in entries if e["kind"] == "train"] == [2, 4]
    versions = {e["version"] for e in entries if e["kind"] == "val"}
    assert versions == set(C.VERSIONS)
    whole = torch.load(tmp_path / "whole/final.pt", map_location="cpu")
    assert whole["step"] == 4 and set(whole["models"]) == set(C.VERSIONS)
    assert whole["models"]["real"]["bands.embed.0.weight"].shape[1] == 6  # the card
    with pytest.raises(SystemExit):
        C.main(run_args(data, tmp_path / "whole"))  # a finished run is never touched

    parts = tmp_path / "parts"
    C.main(run_args(data, parts, "--stop-after", "2"))
    assert (parts / "last.pt").exists() and not (parts / "final.pt").exists()
    with pytest.raises(SystemExit, match="other settings"):
        C.main(run_args(data, parts, "--seed", "1"))
    C.main(run_args(data, parts))  # goes on from step 2
    ended = torch.load(parts / "final.pt", map_location="cpu")
    assert ended["step"] == 4 and ended["config"]["resumed"][0]["step"] == 2
    for v in C.VERSIONS:  # where the unbroken run ended
        for key, value in whole["models"][v].items():
            assert torch.allclose(ended["models"][v][key], value, atol=1e-5), (v, key)
    entries = [json.loads(x) for x in (parts / "metrics.jsonl").read_text().splitlines()]
    assert [e["step"] for e in entries if e["kind"] == "train"] == [2, 4]


def test_a_short_run_with_the_sim_options(data, tmp_path):
    options = ("--q-weights", "0.5", "0.3", "0.2", "--faults", "0.5", "--fault-nir", "0.3")
    C.main(run_args(data, tmp_path / "run", *options))
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["made_up_cameras"]["share_of_sentinel2"] == 0.7
    assert config["made_up_cameras"]["pixel_shares"] == [0.5, 0.3, 0.2]
    faults = config["calibration_faults"]
    assert faults["share_of_samples"] == 0.5 and faults["nir_share_up_to"] == 0.3
    assert faults["bands"]["heo"] == {"visible": [1, 2, 3], "nir": 7}
    state = torch.load(tmp_path / "run/final.pt", map_location="cpu")
    assert state["step"] == 4 and set(state["models"]) == set(C.VERSIONS)
    for bad in (["--q-weights", "1", "-1", "0"], ["--faults", "1.5"], ["--fault-nir", "1"]):
        with pytest.raises(SystemExit):
            C.parse_args([str(tmp_path / "x"), *bad])


def test_short_runs_of_the_arms(data, tmp_path):
    C.main(run_args(data, tmp_path / "chain", "--arms", "chain", "fixed"))
    state = torch.load(tmp_path / "chain/final.pt", map_location="cpu")
    assert set(state["models"]) == {"chain", "fixed"} and state["step"] == 4
    assert state["arms"]["chain"] == {"data": "sim", "fields": 11, "blind": False}
    assert state["arms"]["fixed"]["fields"] == 0
    assert state["models"]["chain"]["bands.embed.0.weight"].shape[1] == 11
    assert not any(k.startswith("bands.") for k in state["models"]["fixed"])
    config = json.loads((tmp_path / "chain/config.json").read_text())
    assert set(config["parameters"]) == {"chain", "fixed"} and config["chain_card"][6] == "psf"
    entries = [json.loads(x) for x in (tmp_path / "chain/metrics.jsonl").read_text().splitlines()]
    assert {e["version"] for e in entries if e["kind"] == "val"} == {"chain", "fixed"}
    C.main(run_args(data, tmp_path / "sensei", "--arms", "sensei", "blind"))
    state = torch.load(tmp_path / "sensei/final.pt", map_location="cpu")
    assert state["models"]["sensei"]["bands.embed.0.weight"].shape[1] == 4
    assert state["models"]["blind"]["bands.embed.0.weight"].shape[1] == 6
    assert state["arms"]["blind"]["blind"]


def test_the_versions_part_on_sentinel2(data, tmp_path):
    C.main(run_args(data, tmp_path / "run", "--shares", "1", "0", "0", "0", "--p-s2", "0"))
    state = torch.load(tmp_path / "run/final.pt", map_location="cpu")
    real, made_up = state["models"]["real"], state["models"]["sim"]
    assert any(not torch.equal(real[k], made_up[k]) for k in real)  # same start, other lessons
    assert [r["source"] for r in state["config"]["sources"]] == ["sentinel2"]
