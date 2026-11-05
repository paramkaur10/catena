"""CloudPeru2 (PeruSat-1): the patches of a split, a patch's labels and reflectance, what the
model is told, and training windows, on four made-up patches."""

from __future__ import annotations

import csv
import math
import warnings

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from catena import descriptor as desc  # noqa: E402
from catena.data import perusat  # noqa: E402

SIZE = 32


def tif(path, data):
    data = data if data.ndim == 3 else data[None]
    profile = {"driver": "GTiff", "height": SIZE, "width": SIZE, "count": data.shape[0]}
    with warnings.catch_warnings():  # no georeferencing, as in CloudPeru2
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", dtype=data.dtype.name, **profile) as dst:
            dst.write(data)


@pytest.fixture
def root(tmp_path):
    """Three train patches and one validation patch: cloud in the top rows, the first 4
    columns empty."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rows = []
    for n, split in ((1, "train"), (2, "train"), (3, "train"), (4, "validation")):
        rng = np.random.default_rng(n)
        image = rng.integers(10, 250, (4, SIZE, SIZE)).astype(np.uint16)
        image[:, :, :4] = 0
        mask = np.zeros((SIZE, SIZE), np.uint8)
        mask[: 4 * n] = 255
        tif(tmp_path / "images" / f"{n}.tif", image)
        tif(tmp_path / "masks" / f"{n}_Mask.tif", mask)
        rows.append({"n": n, "group": n, "split": split, "nodata_frac": 0.125, "cloud_frac": 0})
    with open(tmp_path / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_a_patch(root):
    assert perusat.patches(root, "train") == ["1", "2", "3"]
    assert perusat.patches(root, "validation") == ["4"]
    dn, label = perusat.read(root, "2")
    assert dn.shape == (4, SIZE, SIZE) and label.shape == (SIZE, SIZE)
    assert (label[:8, 4:] == perusat.CLOUD).all() and (label[8:, 4:] == perusat.NOT_CLOUD).all()
    assert (label[:, :4] == perusat.NO_LABEL).all()  # no image, no label
    rho = perusat.reflectance(dn)
    assert rho.dtype == np.float32 and rho[0, 5, 5] == pytest.approx(dn[0, 5, 5] / 255)
    with pytest.raises(ValueError, match="no 'test' patches"):
        perusat.patches(root, "test")


def test_what_perusat_is_told():
    d = perusat.descriptors()
    assert d.shape == (4, desc.N_FIELDS)
    assert d[0, 0] == pytest.approx(desc.wavelength_code(450))  # blue from 450 nm
    assert d[3, 2] == pytest.approx(desc.wavelength_code(885))  # NIR to 885 nm
    card = desc.card(d)
    assert (card[:, 3] == 1).all()  # surface reflectance
    assert card[:, 5] == pytest.approx(math.log2(0.28))  # 2.8 m pixels
    assert (d[:, 10:14] == 0).all()  # blur, noise, step and saturation unknown


def test_training_windows(root):
    windows = perusat.Windows(root, "train", size=16)
    assert len(windows) == 3 and windows.card.shape == (4, desc.N_FIELDS)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(30):
        n, dn, label = windows.draw(rng)
        assert dn.shape == (4, 16, 16) and label.shape == (16, 16)
        assert set(np.unique(label)) <= {perusat.CLOUD, perusat.NOT_CLOUD, perusat.NO_LABEL}
        assert (label != perusat.NO_LABEL).mean() >= 0.5
        seen.add(n)
    assert seen == {"1", "2", "3"}  # never the validation patch
    with pytest.raises(ValueError, match="hold no 64 px window"):
        perusat.Windows(root, "train", size=64)
