"""Pictures of labels: the stretch, the colours, the shares, and figures that open."""

from __future__ import annotations

import numpy as np
import pytest

from catena import viz


def test_stretch_and_overlay():
    rgb = np.zeros((3, 4, 5), np.float32)
    rgb[:, :, 1:] = np.linspace(1, 100, 16, dtype=np.float32).reshape(4, 4)
    picture = viz.stretch(rgb)
    assert picture.shape == (4, 5, 3)
    assert (picture[:, 0] == 1.0).all()  # no data: white
    valid = picture[:, 1:]
    assert valid.min() == 0.0 and valid.max() == 1.0
    label = np.zeros((4, 5), np.uint8)
    label[0, 1], label[1, 1], label[2, 1], label[3, 1] = 1, 2, 3, viz.NO_LABEL
    drawn = viz.overlay(picture, label)
    expected = (1 - viz.STRENGTH) * picture[0, 1] + viz.STRENGTH * np.array(viz.COLOURS[1])
    assert np.allclose(drawn[0, 1], expected)  # thin cloud: yellow
    assert np.allclose(drawn[3, 1], 0.5 * picture[3, 1] + 0.4)  # no label: washed grey
    assert np.array_equal(drawn[:, 2:], picture[:, 2:])  # clear: as it was
    assert drawn.max() <= 1.0
    lifted = viz.stretch(rgb, gamma=0.5)
    assert np.allclose(lifted[:, 1:], np.sqrt(picture[:, 1:])) and (lifted[:, 0] == 1).all()


def test_codes_and_shares():
    assert viz.FROM_OCM[[0, 1, 2, 3, 255, 7]].tolist() == [0, 2, 1, 3, 255, 255]
    label = np.array([[0, 0, 1, 2], [3, 255, 255, 0]], np.uint8)
    s = viz.shares(label)
    assert s == {"clear": 0.5, "thin": 1 / 6, "thick": 1 / 6, "shadow": 1 / 6, "labelled": 0.75}
    assert "cloud" not in s  # codes 4 and 5 only when present
    assert viz.shares(np.array([[4, 5]], np.uint8))["cloud"] == 0.5
    assert viz.describe(label) == "clear 50%, thin 17%, thick 17%, shadow 17% (75% labelled)"


def test_certainty_step_and_table():
    conf = np.full((4, 2, 2), 50, np.uint8)
    conf[1, 0, 0] = 200
    conf[:, 1, 1] = 255
    sure = viz.certainty(conf)
    assert sure[0, 0] == pytest.approx(0.8) and sure[0, 1] == pytest.approx(0.2)
    assert np.isnan(sure[1, 1])
    assert viz.step((2816, 2810), 1400) == 3 and viz.step((500, 400)) == 1
    rows = [{"scene": "A<1>", "thin": 0.25, "sun": 40.5}]
    table = viz.html_table(rows, ["scene", "thin", "sun", "notes"])
    assert "A&lt;1&gt;" in table and "25%" in table and "40.50" in table
    assert table.count("<td>") == 4


def test_figures_open():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    picture = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    label = np.zeros((20, 30), np.uint8)
    sure = np.full((20, 30), 0.7, np.float32)
    fig = viz.panels([picture, viz.overlay(picture, label), sure], ["a", "b", "c"], show=False)
    assert len(fig.axes) == 4  # three pictures and the certainty's colour bar
    fig = viz.panels([picture, picture], ["a", "b"], origin=(100, 200), show=False)
    assert fig.axes[0].get_xlim()[0] == pytest.approx(199.5)  # axes count columns from 200
    items = [(picture, picture, f"scene {k}") for k in range(3)]
    fig = viz.sheet(items, per_row=2, show=False)
    assert len(fig.axes) == 8  # two rows of two pairs, the last pair empty
    fig = viz.sheet(items[:1], per_row=1, legend=(4,), show=False)
    texts = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert texts == ["cloud"]  # CloudPeru2's one cloud class
