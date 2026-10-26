"""D3: CloudSEN12+ windows, with the dataset's three traps fixed.

The chips here are made by the tests in the layout D2 writes: <split>/<id>.s2l1c.tif (13
bands, uint16) and <split>/<id>.target.tif (CloudSEN12's codes), listed in index.csv. Sizes
are the real ones divided by 10: 24-pixel windows, 52-pixel frames holding 50 pixels of
image, 200-pixel frames holding 196, with the empty rows and columns at the bottom and right
labelled clear, as in CloudSEN12+.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from catena.data import loader as ld  # noqa: E402

SIZE = 24
SMALL, SMALL_REAL = 52, 50
BIG, BIG_REAL = 200, 196


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


def patch(chips, split, name, frame, real, seed, baseline="02.07", image=None, label=None):
    """Write one patch; returns its index row. Outside the real area the image is 0 in every
    band and the label says clear."""
    rng = np.random.default_rng(seed)
    offset = ld.offset_of(baseline)
    if image is None:
        image = rng.integers(1, 4000, (13, real, real)) + offset
    if label is None:
        label = rng.integers(0, 4, (real, real))
    full = np.zeros((13, frame, frame), np.uint16)
    full[:, :real, :real] = image
    codes = np.zeros((1, frame, frame), np.uint8)
    codes[0, :real, :real] = label
    (chips / split).mkdir(parents=True, exist_ok=True)
    geotiff(chips / split / f"{name}.s2l1c.tif", full)
    geotiff(chips / split / f"{name}.target.tif", codes)
    return {
        "id": name,
        "split": split,
        "s2l1c_file": f"{split}/{name}.s2l1c.tif",
        "target_file": f"{split}/{name}.target.tif",
        "height": frame,
        "width": frame,
        "baseline": baseline,
        "sun_zenith": "35.5",
    }


def index(chips, rows) -> None:
    with open(chips / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def chips(tmp_path_factory):
    """train: ten small patches and one big; validation: one of each; test: one small
    patch from processing baseline 05.09, with the +1000 offset."""
    chips = tmp_path_factory.mktemp("cloudsen12plus")
    rows = [patch(chips, "train", f"small_{k}", SMALL, SMALL_REAL, k) for k in range(10)]
    rows.append(patch(chips, "train", "big_0", BIG, BIG_REAL, 10))
    rows.append(patch(chips, "validation", "small_v", SMALL, SMALL_REAL, 11))
    rows.append(patch(chips, "validation", "big_v", BIG, BIG_REAL, 12))
    rows.append(patch(chips, "test", "small_t", SMALL, SMALL_REAL, 13, baseline="05.09"))
    index(chips, rows)
    return chips


def test_codes_become_catenas():
    assert ld.CODES[[0, 1, 2, 3]].tolist() == [0, 2, 1, 3]  # clear, thick, thin, shadow
    assert ld.CODES[4] == ld.CODES[99] == ld.CODES[255] == ld.NO_LABEL


def test_offset_starts_with_baseline_04_00():
    baselines = ("02.07", "03.01", "04.00", "05.09", "05.10")
    assert [ld.offset_of(b) for b in baselines] == [0, 0, 1000, 1000, 1000]


def test_patches_of_one_split(chips):
    train = ld.patches(chips, "train")
    assert len(train) == 11 and all(p.offset == 0 for p in train)
    (test,) = ld.patches(chips, "test")
    assert (test.offset, test.sun_zenith, test.height) == (1000, 35.5, SMALL)
    assert test.image == chips / "test" / "small_t.s2l1c.tif"
    with pytest.raises(ValueError, match="no 'nosuch' patches"):
        ld.patches(chips, "nosuch")


def test_read_fixes_empty_pixels_no_data_and_the_offset(tmp_path):
    """Trap 1 (empty pixels, labelled clear) and trap 3 (the offset), in one window that
    reaches the empty border."""
    image = np.full((13, SMALL_REAL, SMALL_REAL), 1000 + 500)  # reflectance 0.05
    image[5, 30, 30] = ld.NODATA
    label = np.ones((SMALL_REAL, SMALL_REAL), np.uint8)  # CloudSEN12's thick
    label[:, 40:] = 2  # thin
    index(tmp_path, [patch(tmp_path, "test", "p", SMALL, SMALL_REAL, 0, "04.00", image, label)])
    (p,) = ld.patches(tmp_path, "test")
    corner = SMALL - SIZE  # 28: the window's last 2 rows and columns are the empty border
    img, lab = ld.read(p, corner, corner, SIZE)
    assert img.dtype == np.float32 and img.shape == (13, SIZE, SIZE)
    assert lab.dtype == np.uint8 and lab.shape == (SIZE, SIZE)
    border = np.zeros((SIZE, SIZE), bool)
    border[-2:, :] = border[:, -2:] = True
    border[30 - corner, 30 - corner] = True  # the no-data pixel
    assert (lab[border] == ld.NO_LABEL).all() and (img[:, border] == 0).all()
    assert np.allclose(img[:, ~border], 0.05)
    assert set(lab[~border].tolist()) == {1, 2}  # thick 1 became 2 and thin 2 became 1
    assert (lab[:, : 40 - corner][~border[:, : 40 - corner]] == 2).all()


def test_grid_tiles_a_patch_without_overlap():
    assert ld.grid(SMALL, SMALL, SIZE) == [(0, 0), (0, 24), (24, 0), (24, 24)]
    corners = ld.grid(BIG, BIG, SIZE)
    cover = np.zeros((BIG, BIG), int)
    for row, col in corners:
        cover[row : row + SIZE, col : col + SIZE] += 1
    assert len(corners) == 64 and cover.max() == 1 and cover[:192, :192].min() == 1
    # the real sizes: 4 and 64 windows, leaving out 29 of 509 and 80 of 2000 pixels
    assert len(ld.grid(512, 512)) == 4 and len(ld.grid(2048, 2048)) == 64
    assert max(row for row, _ in ld.grid(2048, 2048)) + ld.SIZE == 2000 - 80


def test_dihedral_gives_eight_views_and_keeps_bands_aligned():
    x = np.arange(12, dtype=np.uint8).reshape(3, 4)
    views = [ld.dihedral(x, t) for t in range(8)]
    assert len({(v.shape, v.tobytes()) for v in views}) == 8
    both = np.stack([x, 2 * x])  # band 1 is twice band 0 everywhere
    for t in range(8):
        turned = ld.dihedral(both, t)
        assert np.array_equal(turned[0], views[t]) and np.array_equal(turned[1], 2 * views[t])
        assert turned.flags.c_contiguous


def test_training_windows_repeat_exactly(chips):
    a = ld.TrainingWindows(chips, size=SIZE, length=50, seed=3)
    b = ld.TrainingWindows(chips, size=SIZE, length=50, seed=3)
    keys = ("id", "row", "col", "turn")
    for i in (7, 0, 49, 7):  # any order, any copy (each data-loading worker has its own)
        sa, sb = a[i], b[i]
        assert [sa[k] for k in keys] == [sb[k] for k in keys]
        assert np.array_equal(sa["image"], sb["image"]) and np.array_equal(sa["label"], sb["label"])
        assert sa["image"].shape == (13, SIZE, SIZE) and sa["label"].shape == (SIZE, SIZE)
    assert len({tuple(a[i][k] for k in keys) for i in range(10)}) == 10  # each i its own
    assert len({a[i]["turn"] for i in range(50)}) == 8  # all eight turns are used
    s = a[7]  # the sample is its window, turned
    p = next(p for p in a.patches if p.id == s["id"])
    image, label = ld.read(p, s["row"], s["col"], SIZE)
    assert np.array_equal(ld.dihedral(image, s["turn"]), s["image"])
    assert np.array_equal(ld.dihedral(label, s["turn"]), s["label"])
    c = ld.TrainingWindows(chips, size=SIZE, length=50, seed=4)
    first = [[a[i][k] for k in keys] for i in range(10)]
    assert first != [[c[i][k] for k in keys] for i in range(10)]  # another seed, other windows
    with pytest.raises(IndexError):
        a[50]


def test_training_windows_are_drawn_by_area(chips):
    """Trap 2: the one big patch has 40,000 pixels against 27,040 in the ten small ones, so
    it should give 60% of the windows, not 1 in 11; and every place in a patch can be drawn."""
    windows = ld.TrainingWindows(chips, size=SIZE)
    rng = np.random.default_rng(0)
    draws = [windows.draw(rng) for _ in range(4000)]
    big = np.mean([p.height == BIG for p, _, _ in draws])
    assert abs(big - BIG**2 / (BIG**2 + 10 * SMALL**2)) < 0.03
    rows = [row for p, row, _ in draws if p.height == SMALL]
    cols = [col for p, _, col in draws if p.height == BIG]
    assert (min(rows), max(rows)) == (0, SMALL - SIZE) and (min(cols), max(cols)) == (0, BIG - SIZE)


def test_windows_less_than_half_labelled_are_drawn_again(tmp_path):
    """One patch whose image stops at column 30: a third of its windows are less than half
    labelled. They are drawn again; a patch with no image at all still gives a window."""
    image = np.full((13, SMALL_REAL, SMALL_REAL), 700)
    image[:, :, 30:] = 0
    index(tmp_path, [patch(tmp_path, "train", "half", SMALL, SMALL_REAL, 0, image=image)])
    redrawn = ld.TrainingWindows(tmp_path, size=SIZE, length=40, seed=1)
    once = ld.TrainingWindows(tmp_path, size=SIZE, length=40, seed=1, tries=1)
    share = [np.mean(redrawn[i]["label"] != ld.NO_LABEL) for i in range(40)]
    share_once = [np.mean(once[i]["label"] != ld.NO_LABEL) for i in range(40)]
    assert min(share) >= 0.5 and min(share_once) < 0.5

    image[:] = 0
    index(tmp_path, [patch(tmp_path, "train", "empty", SMALL, SMALL_REAL, 0, image=image)])
    sample = ld.TrainingWindows(tmp_path, size=SIZE, length=1)[0]
    assert (sample["label"] == ld.NO_LABEL).all() and (sample["image"] == 0).all()


def test_patches_smaller_than_a_window_are_refused(chips):
    with pytest.raises(ValueError, match="smaller than 53 px"):
        ld.TrainingWindows(chips, size=SMALL + 1)


def test_grid_windows_are_fixed_and_cover_the_split(chips):
    val = ld.GridWindows(chips, "validation", size=SIZE)
    assert len(val) == 4 + 64
    first, again = val[0], val[0]
    assert (first["id"], first["row"], first["col"], first["turn"]) == ("small_v", 0, 0, 0)
    assert np.array_equal(first["image"], again["image"])
    assert [val[i]["id"] for i in (3, 4, 67)] == ["small_v", "big_v", "big_v"]
    assert (val[67]["row"], val[67]["col"]) == (168, 168)


def test_main_reports_the_windows(chips, capsys, tmp_path):
    ld.main([str(chips), "--n", "12", "--size", str(SIZE)])
    out = capsys.readouterr().out
    assert "11 train patches: 10 of 52 px (40% of the area), 1 of 200 px (60% of the area)" in out
    assert "12 windows in" in out and "from the 200 px patches" in out
    pytest.importorskip("matplotlib")
    picture = tmp_path / "windows.png"
    ld.main([str(chips), "--n", "8", "--size", str(SIZE), "--figure", str(picture)])
    assert picture.stat().st_size > 10_000
