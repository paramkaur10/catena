"""The card's newest parts: the pixel size on the ground, told to both arms, roughly right
cards (jitter), the six numbers Clarity reads, and the chain card: the card, then the MTF
and the SNR."""

from __future__ import annotations

import math

import numpy as np
import pytest

from catena import descriptor as desc

RED = (645.5, 684.5)
AB = (1e-4, 1e-6)


def test_the_pixel_size_is_told_to_both_arms():
    for grid, code in ((10, 0.0), (30, math.log2(3)), (2.5, -2.0)):
        d = desc.describe(*RED, grid, grid, 0.25, AB, 1e-3, 1.2)
        assert d[desc.GSD] == pytest.approx(code)
        b = desc.blind(d)
        assert b[desc.GSD] == d[desc.GSD] and (b[:3] == d[:3]).all()
        assert (b[3 : desc.GSD] == 0).all()  # the chain and its flags: unknown to blind
    rng = np.random.default_rng(0)
    d = np.stack([desc.describe(*RED, 30, 30, 0.25, AB, 1e-3, 1.2)] * 2)
    for _ in range(50):  # dropout never hides it
        assert (desc.hide(d, rng)[:, desc.GSD] == d[:, desc.GSD]).all()


def test_roughly_right_cards():
    known = desc.describe(*RED, 30, 30, 0.25, AB, 1e-3, 1.2)
    unknown = desc.describe(*RED, 30, 30)  # only the band and the pixel size are known
    d = np.stack([known, known, unknown])
    assert np.array_equal(desc.jitter(d, np.random.default_rng(0), 0.0), d)
    rng = np.random.default_rng(1)
    moved = [desc.jitter(d, rng) for _ in range(200)]
    same = [0, 1, 2, 3, 4, 10, 11, 12, 13, desc.GSD]  # bands, sampling, flags, pixel size
    for m in moved:
        assert np.array_equal(m[:, same], d[:, same])
        assert (m[2] == unknown).all()  # unknown values stay unknown
        psf = m[:2, 5] / d[:2, 5]
        assert ((psf >= 1 / 1.25 - 1e-6) & (psf <= 1.25 + 1e-6)).all()
        shift = m[:2, 6] - d[:2, 6]
        assert np.allclose(m[:2, 7] - d[:2, 7], shift, atol=1e-6)  # dark and bright together
        assert (np.abs(shift) <= 0.3 + 1e-6).all()
        assert (np.abs(m[:2, 8] - d[:2, 8]) <= 0.3 + 1e-6).all()
        top = m[:2, 9] / d[:2, 9]
        assert ((top >= 0.85 - 1e-6) & (top <= 1.15 + 1e-6)).all()
    first = np.stack([m[0] for m in moved])
    assert first[:, 5].std() > 0 and first[:, 8].std() > 0  # it does move them
    assert not np.array_equal(moved[0][0], moved[0][1])  # each band its own error


def test_what_clarity_reads():
    d = desc.describe(*RED, 20, 10, 0.25, AB, 1e-3, 1.2, surface=True)
    c = desc.card(d)
    assert c.shape == (desc.N_CARD,) == (6,)
    assert c[:3].tolist() == d[:3].tolist()  # the wavelengths
    assert c[3] == 1.0 and c[4] == 1.0  # surface reflectance; a 20 m band on a 10 m grid
    assert c[5] == d[desc.GSD] == 0.0  # 10 m grid
    stacked = desc.card(np.stack([d, d, d]))
    assert stacked.shape == (3, 6) and (stacked[1] == c).all()


def test_the_chain_card():
    known = desc.describe(*RED, 20, 10, 0.25, AB, 1e-3, 1.2)
    unknown = desc.describe(*RED, 30, 30)  # no MTF, no SNR
    told = desc.chain_card(np.stack([known, unknown]))
    assert told.shape == (2, desc.N_CHAIN_CARD) == (2, 11)
    assert np.array_equal(told[:, : desc.N_CARD], desc.card(np.stack([known, unknown])))
    assert told[0, 6] == known[5] > 0  # the blur, from the MTF
    assert told[0, 7:9].tolist() == known[6:8].tolist()  # the noise at 0.05 and 0.5, from the SNR
    assert told[0, 9:].tolist() == [1.0, 1.0]  # both known
    assert (told[1, 6:] == 0).all()  # unknown: 0, flags 0
    assert told[0, 5] == known[desc.GSD]  # the grid's pixel size stays where the card has it
    rng = np.random.default_rng(0)
    for _ in range(50):  # dropout and errors never touch the card's six numbers
        moved = desc.chain_card(desc.jitter(desc.hide(np.stack([known] * 3), rng), rng))
        assert np.array_equal(moved[:, : desc.N_CARD], np.stack([told[0, : desc.N_CARD]] * 3))
