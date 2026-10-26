"""D2: CloudSEN12+ patches copied out of TACO files.

The TACO files here are built by the tests in the layout of the real ones: a container of
samples, each sample a container of two GeoTIFFs, each container ending in a Parquet table.
"""

from __future__ import annotations

import csv
import zlib

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena.data import cloudsen12 as cs  # noqa: E402

SIZE, REAL = 16, 13  # the 509-pixel patches in 512-pixel frames, scaled down
SAMPLES = [  # (file, id, split, label type, product)
    (0, "ROI_1__A", "train", "high", "S2A_MSIL1C_20190116T142751_N0207_R053_T19GDN_20190116"),
    (0, "ROI_2__B", "train", "scribble", "S2B_MSIL1C_20190201T142749_N0207_R053_T19GDN_2019"),
    (0, "ROI_3__C", "test", "high", "S2B_MSIL1C_20230301T101029_N0509_R022_T32TMR_20230301"),
    (1, "ROI_4__D", "validation", "high", "S2A_MSIL1C_20200510T101031_N0209_R022_T32TMR_2020"),
    (1, "ROI_5__E", "train", "nolabel", "S2A_MSIL1C_20200610T101031_N0209_R022_T32TMR_2020"),
]
HIGH = ["ROI_1__A", "ROI_3__C", "ROI_4__D"]


def patch(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A 13-band image and its label; on a 3-pixel border, bottom and right, the image is 0
    in every band and the label says 0 (clear), as in the real 509-pixel patches."""
    rng = np.random.default_rng(seed)
    image = rng.integers(1, 10000, (13, SIZE, SIZE), dtype=np.uint16)
    label = rng.integers(0, 4, (1, SIZE, SIZE), dtype=np.uint8)
    for array in (image, label):
        array[:, REAL:, :] = 0
        array[:, :, REAL:] = 0
    return image, label


def geotiff(path, data: np.ndarray) -> bytes:
    """The bytes of a small GeoTIFF holding `data` (bands, rows, columns)."""
    profile = {
        "driver": "GTiff",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype.name,
        "crs": "EPSG:32719",
        "transform": from_origin(419020, 5251220, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path.read_bytes()


def container(parts: list[tuple[bytes, dict]], magic: bytes) -> bytes:
    """A TACO container: a 200-byte header, the parts, then a Parquet table of them."""
    body, rows, offset = bytearray(), [], 200
    for data, meta in parts:
        rows.append({**meta, "tortilla:offset": offset, "tortilla:length": len(data)})
        body += data
        offset += len(data)
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    table = sink.getvalue().to_pybytes()
    head = magic + offset.to_bytes(8, "little") + len(table).to_bytes(8, "little")
    return head + bytes(200 - len(head)) + bytes(body) + table


@pytest.fixture
def raw(tmp_path):
    """Two TACO files holding the five SAMPLES; returns their folder and the GeoTIFF bytes."""
    folder, made, files = tmp_path / "raw", {}, {0: [], 1: []}
    folder.mkdir()
    for i, (n, name, split, kind, product) in enumerate(SAMPLES):
        image, label = patch(i)
        tiffs = {
            "s2l1c": geotiff(tmp_path / f"{name}_image.tif", image),
            "target": geotiff(tmp_path / f"{name}_label.tif", label),
        }
        made[name] = tiffs
        inner = [(tiffs[k], {"tortilla:id": k, "tortilla:file_format": "GTiff"}) for k in tiffs]
        meta = {
            "tortilla:id": name,
            "tortilla:file_format": "TORTILLA",
            "tortilla:data_split": split,
            "label_type": kind,
            "real_proj_shape": 509,
            "s2_id": product,
            "stac:crs": "EPSG:32719",
            "stac:geotransform": [419020.0, 10.0, 0.0, 5251220.0, 0.0, -10.0],
            "stac:time_start": 1547649880.0 + i,
            "stac:centroid": "POINT (-69.960716 -42.910647)",
            "s2_mean_solar_zenith_angle": 56.1,
            "rai:admin0": "Argentina",
            "thick_percentage": 86,
        }
        files[n].append((container(inner, b"#y"), meta))
    for n, parts in files.items():
        (folder / f"cloudsen12-l1c.000{n}.part.taco").write_bytes(container(parts, b"WX"))
    return folder, made


def test_the_footers_find_every_patch_and_its_two_files(raw):
    folder, made = raw
    taco = folder / "cloudsen12-l1c.0000.part.taco"
    samples = cs.footer(taco)
    assert [s["tortilla:id"] for s in samples] == ["ROI_1__A", "ROI_2__B", "ROI_3__C"]
    data = taco.read_bytes()
    for sample in samples:
        for key, (offset, length) in cs.items(taco, sample).items():
            assert data[offset : offset + length] == made[sample["tortilla:id"]][key]


def test_a_file_that_is_not_taco_is_refused(tmp_path):
    path = tmp_path / "other.taco"
    path.write_bytes(b"PK" + bytes(40))
    with pytest.raises(ValueError, match="not a TACO"):
        cs.footer(path)


def test_only_hand_labelled_patches_are_copied_and_unchanged(raw, tmp_path):
    folder, made = raw
    out = tmp_path / "chips"
    rows, done = cs.extract(folder, out, {"high"})
    assert [r["id"] for r in rows] == HIGH
    assert done == {"copied": 6}
    for row in rows:
        for key in cs.ITEMS:
            copy = out / row[f"{key}_file"]
            assert copy.parent.name == row["split"]
            assert copy.read_bytes() == made[row["id"]][key]
            assert row[f"{key}_crc32"] == f"{zlib.crc32(made[row['id']][key]):08x}"
    assert sorted(p.name for p in out.iterdir()) == ["index.csv", "test", "train", "validation"]


def test_the_index_holds_metadata_and_label_counts(raw, tmp_path):
    folder, _ = raw
    out = tmp_path / "chips"
    cs.extract(folder, out, {"high"})
    with open(out / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["baseline"] for r in rows] == ["02.07", "05.09", "02.09"]
    first = rows[0]
    assert (first["lon"], first["lat"]) == ("-69.96072", "-42.91065")
    assert first["time_utc"] == "2019-01-16T14:44:40+00:00"
    assert (first["width"], first["height"], first["pixel_m"]) == ("16", "16", "10.0")
    _, label = patch(0)
    for code, name in cs.CLASSES.items():
        assert int(first[f"px_{name}"]) == int((label == code).sum())
    assert (first["px_other"], first["other_values"], first["extra_items"]) == ("0", "", "")


def test_a_rerun_keeps_good_copies_and_replaces_a_damaged_one(raw, tmp_path):
    folder, made = raw
    out = tmp_path / "chips"
    rows, _ = cs.extract(folder, out, {"high"})
    damaged = out / rows[0]["s2l1c_file"]
    data = bytearray(damaged.read_bytes())
    data[-1] ^= 0xFF  # same size, one byte different
    damaged.write_bytes(bytes(data))
    _, done = cs.extract(folder, out, {"high"})
    assert done == {"kept": 5, "copied": 1}
    assert damaged.read_bytes() == made[rows[0]["id"]]["s2l1c"]


def test_the_check_finds_the_zero_border_and_what_the_label_calls_it(raw, tmp_path):
    folder, _ = raw
    rows, _ = cs.extract(folder, tmp_path / "chips", {"high"})
    report = cs.check(tmp_path / "chips", rows)
    border = SIZE * SIZE - REAL * REAL
    assert report["patches"] == report["with_border"] == 3
    assert report["zero"] == 3 * border
    assert report["edges"] == {(0, SIZE - REAL, 0, SIZE - REAL): 3}
    assert report["border_labels"] == {0: 3 * border}
    assert report["nodata"] == 0
    assert sorted(report["dark_b10"]) == ["02.07", "02.09", "05.09"]


def test_baselines_and_centroids_are_read_from_their_text():
    assert cs.baseline("S2A_MSIL1C_20190116T142751_N0207_R053_T19GDN") == "02.07"
    assert cs.baseline("S2B_MSIL1C_20230301T101029_N0509_R022_T32TMR") == "05.09"
    assert cs.baseline("") == cs.baseline(None) == ""
    assert cs.lon_lat("POINT (-69.960716 -42.910647)") == (-69.96072, -42.91065)
    assert cs.lon_lat(None) == ("", "")


def test_not_enough_disk_stops_before_anything_is_copied(raw, tmp_path, monkeypatch):
    folder, _ = raw
    out = tmp_path / "chips"
    monkeypatch.setattr(cs.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 0}))
    with pytest.raises(SystemExit, match="GB free"):
        cs.extract(folder, out, {"high"})
    assert not list(out.rglob("*.tif"))


def test_the_command_runs_end_to_end(raw, tmp_path, capsys):
    folder, _ = raw
    out = tmp_path / "chips"
    cs.main([str(folder), str(out), "--label-type", "high", "scribble"])
    printed = capsys.readouterr().out
    assert "4 patches" in printed and "label values on those pixels: {0: 348}" in printed
    with open(out / "index.csv") as f:
        assert [r["id"] for r in csv.DictReader(f)] == ["ROI_1__A", "ROI_2__B", *HIGH[1:]]
