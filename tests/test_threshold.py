"""The thresholds floor: the bands it finds by their wavelengths, its features, a tree that
finds thresholds that are there, and the rule as it is saved and read."""

from __future__ import annotations

import json

import numpy as np

from catena import descriptor as desc
from catena import threshold as T

S2_BLUE, S2_GREEN, S2_RED, S2_NIR = (458, 523), (543, 578), (650, 680), (785, 900)


def card(*edges) -> np.ndarray:
    """Chain cards (bands, 11) of bands with these edges in nm, at 10 m."""
    return desc.chain_card(np.stack([desc.describe(lo, hi, 10, 10) for lo, hi in edges]))


def test_the_bands_by_their_wavelengths():
    told = card(S2_NIR, S2_RED, S2_GREEN, S2_BLUE)  # in any order
    assert np.allclose(T.wavelengths(told), [842.5, 665, 560.5, 490.5], atol=0.01)
    r = T.roles(told)
    assert list(r["visible"]) == [1, 2, 3] and list(r["nir"]) == [0]
    assert (r["blue"], r["red"]) == (3, 1)
    assert list(T.roles(told, real=[False, True, True, True])["nir"]) == []
    assert T.roles(card(S2_NIR)) is None  # no visible band
    padded = np.concatenate([told, np.zeros((2, desc.N_CHAIN_CARD), np.float32)])
    assert len(T.roles(padded)["visible"]) == 3  # an empty slot is no band


def test_the_features():
    told = card(S2_BLUE, S2_GREEN, S2_RED, S2_NIR)
    rho = np.zeros((4, 2, 3), np.float32)
    rho[:, 0, 0] = [0.3, 0.3, 0.3, 0.3]  # white and bright: a cloud
    rho[:, 0, 1] = [0.04, 0.06, 0.04, 0.4]  # vegetation
    f = T.features(rho, T.roles(told))
    assert f.shape == (5, 2, 3) and f.dtype == np.float32
    assert np.allclose(f[:, 0, 0], [0.3, 0, 0.3 - 0.15 - 0.08, 0.3, 0], atol=1e-6)
    assert f[4, 0, 1] > 0.8 and f[1, 0, 1] > 0.1  # green leaves: high ndvi, not white
    assert np.isfinite(f).all()  # no data (zeros) gives numbers too
    assert T.features(rho[:3], T.roles(told[:3])).shape == (3, 2, 3)


def test_a_tree_finds_the_thresholds_that_are_there():
    rng = np.random.default_rng(0)
    x = rng.random((20_000, 3)).astype(np.float32)
    y = np.where(x[:, 0] > 0.6, 2, np.where(x[:, 0] > 0.3, 1, 0))  # thresholds on feature 0
    y[(x[:, 0] <= 0.3) & (x[:, 2] < 0.1)] = 3  # and a rare class on feature 2
    tree = T.grow(x, y, depth=3, least=50)
    assert np.mean(T.decide(tree, x.T) == y) > 0.98
    found = sorted(n["threshold"] for n in tree if n.get("feature") == 0)
    assert any(abs(t - 0.3) < 0.04 for t in found) and any(abs(t - 0.6) < 0.04 for t in found)
    assert any(n.get("class") == 3 for n in tree)  # rare, but weighted: it gets a leaf
    assert sum("feature" in n for n in tree) <= 4  # no threshold that changes nothing
    assert T.lines(tree)[0].startswith("  brightness")
    assert T.grow(x, np.zeros(len(x), np.int64), depth=3) == [{"class": 0}]


def test_the_rule_as_saved(tmp_path):
    always = {
        "vnir": [{"class": 2}],
        "vis": [
            {"feature": 0, "threshold": 0.2, "left": 1, "right": 2},
            {"class": 0},
            {"class": 1},
        ],
    }
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"trees": always}))
    rule = T.load(path)
    rho = np.full((4, 2, 5, 5), 0.3, np.float32)  # (bands, n, h, w): a batch
    with_nir = card(S2_BLUE, S2_GREEN, S2_RED, S2_NIR)
    assert (rule.classes(rho, with_nir) == 2).all() and rule.classes(rho, with_nir).shape == (
        2,
        5,
        5,
    )
    rgb = rho[:3].copy()
    rgb[:, 0] = 0.1
    found = rule.classes(rgb, with_nir[:3])  # no near infrared: the other tree
    assert (found[0] == 0).all() and (found[1] == 1).all()
    assert (rule.classes(rho[:1], card(S2_NIR)) == 0).all()  # nothing visible: clear


def test_the_command_picks_the_weighting_with_the_best_mean_f1(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    x = rng.random((30_000, 5)).astype(np.float32)
    y = np.zeros(len(x), np.int64)  # clear, mostly: thin, thick and shadow rarer
    y[x[:, 0] > 0.75] = 2
    y[(x[:, 0] > 0.6) & (x[:, 0] <= 0.75)] = 1
    y[(x[:, 0] < 0.08) & (x[:, 3] < 0.5)] = 3
    noise = rng.random(len(x)) < 0.1
    y[noise] = rng.integers(0, 4, noise.sum())
    monkeypatch.setattr(T, "samples", lambda *args: (x, y))
    T.main([str(tmp_path), "--depth", "3"])
    found = json.loads((tmp_path / "threshold.json").read_text())
    assert set(found["trees"]) == {"vnir", "vis"} and set(found["power"]) == {"vnir", "vis"}
    for name in ("vnir", "vis"):
        tried = {p: found["mean_f1_tried"][f"{name} {p:g}"] for p in T.POWERS}
        assert found["power"][name] in T.POWERS
        assert tried[found["power"][name]] == max(tried.values())
    rule = T.load(tmp_path / "threshold.json")
    assert rule.trees == found["trees"]
