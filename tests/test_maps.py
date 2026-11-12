"""The kept maps: an arm's map is the vote of its seeds; each scene's numbers and its
picture, from made-up .npz files laid out as catena.evaluate_clarity --maps writes them."""

from __future__ import annotations

import numpy as np
import pytest

from catena import maps


def kept_scene(folder, name: str = "eaglet2", scene: str = "IME01_20261019T160210"):
    """A 4 x 6 scene: the label's top half thick cloud; seeds that mostly agree."""
    label = np.zeros((4, 6), np.uint8)
    label[:2] = 2
    label[:, -1] = 255
    real = [label.copy() for _ in range(3)]
    real[0][3, 0] = 1  # one seed alone: outvoted
    real[1][2, :2] = 2  # two seeds: a vote
    real[2][2, :2] = 2
    sim = [np.full((4, 6), 2, np.uint8) for _ in range(3)]  # all cloud
    arrays = {f"seed{k}_real": m for k, m in enumerate(real)}
    arrays |= {f"seed{k}_sim": m for k, m in enumerate(sim)}
    arrays["ocm4_nirblue"] = np.zeros((4, 6), np.uint8)  # no cloud at all
    rgb = np.full((2, 3, 3), 128, np.uint8)  # every second pixel
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}__{scene}.npz", label=label, rgb=rgb, step=2, **arrays)
    return label


def test_vote():
    a = np.array([[0, 1], [2, 255]], np.uint8)
    b = np.array([[0, 2], [2, 255]], np.uint8)
    c = np.array([[3, 2], [1, 255]], np.uint8)
    assert maps.vote([a, b, c]).tolist() == [[0, 2], [2, 255]]
    assert maps.vote([a, c]).tolist() == [[0, 1], [1, 255]]  # a tie: the lowest code


def test_numbers_and_picture(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    label = kept_scene(tmp_path)
    kept = maps.Kept(tmp_path, "eaglet2")
    assert kept.scenes == ["IME01_20261019T160210"] and "1 scenes" in repr(kept)
    models = kept.models(kept.load(kept.scenes[0]))
    assert list(models) == ["Clarity real", "Clarity sim", "OmniCloudMask, NIR from blue"]
    voted = models["Clarity real"]
    assert voted[3, 0] == 0 and (voted[2, :2] == 2).all()
    row = kept.row(kept.scenes[0])
    labelled = int((label != 255).sum())
    assert row["labelled"] == labelled == 20 and row["cloud"] == pytest.approx(10 / 20)
    assert row["Clarity sim cloud"] == 1.0 and row["Clarity sim BA"] == pytest.approx(0.5)
    assert row["OmniCloudMask, NIR from blue cloud"] == 0.0
    assert row["Clarity real cloud"] == pytest.approx(12 / 20)
    table = kept.table()
    assert table[0]["Clarity sim BA"] == "0.500" and table[0]["cloud"] == 0.5
    columns = kept.columns()
    assert columns[:3] == maps.COLUMNS and "Clarity real BA" in columns
    fig = kept.show(kept.scenes[0], show=False)
    assert len(fig.axes) == 5 and fig.axes[1].get_title() == "label: cloud 50%"
    assert fig.axes[3].get_title().startswith("Clarity sim: cloud 100%")
    assert maps.Kept(tmp_path / "none", "eaglet2").scenes == []
    assert maps.Kept(tmp_path / "none", "eaglet2").columns() == maps.COLUMNS


def test_every_arm_has_its_title(tmp_path):
    kept_scene(tmp_path)
    kept = maps.Kept(tmp_path, "eaglet2")
    arrays = kept.load(kept.scenes[0])
    arrays |= {"seed0_fixed": arrays["seed0_sim"], "seed0_chain": arrays["seed0_real"]}
    names = list(kept.models(arrays))
    assert names[:4] == ["Clarity real", "Clarity sim", "Clarity + MTF, SNR", "U-Net fixed bands"]
    real = arrays["seed0_real"]
    other = (real + 1) % 4  # another class everywhere
    apart = {"seed0_real": real, "seed0_chain_real": other, "seed1_chain_real": other}
    found = kept.models(apart)  # an arm on real's samples is not one of real's seeds
    assert list(found) == ["Clarity real", "Clarity real + MTF, SNR"]
    assert np.array_equal(found["Clarity real"], real)
    assert np.array_equal(found["Clarity real + MTF, SNR"], other)
