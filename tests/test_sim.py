"""Descriptors and virtual cameras (A3, A4): unknown values look unknown, a camera is never
better than Sentinel-2, the descriptor tells what the camera got, labels move by majority,
dropout hides whole groups, and Sentinel-2's own chain comes from its spec."""

from __future__ import annotations

import math

import numpy as np
import pytest

from catena.descriptor import FIELDS, bands_from_spec, blind, describe, hide
from catena.sensors.physics import mtf_from_sigma, noise_from_snr, sigma_from_mtf
from catena.sim import downsample_label, fixed_camera, imagine_camera, simulate

S2_SIGMA = sigma_from_mtf(0.25)
S2_AB = noise_from_snr(142, 0.224, 0.7)
RED = (645.5, 684.5)


def field(d, name):
    return float(d[FIELDS.index(name)])


def chip(n=3, size=90, value=0.2):
    return np.full((n, size, size), value)


def test_unknown_fields_stay_zero_with_flag_zero():
    d = describe(*RED, 30, 30)
    for name in FIELDS[4:]:
        if name != "gsd":
            assert field(d, name) == 0.0
    assert field(d, "gsd") == pytest.approx(math.log2(3))  # the pixel size is always known
    assert field(d, "lo") < field(d, "mid") < field(d, "hi")


def test_known_fields():
    d = describe(*RED, 20, 10, mtf_nyquist=0.25, noise_ab=S2_AB, step=1e-4, top=1.2, surface=True)
    assert field(d, "sampling") == 1.0
    assert field(d, "psf") == pytest.approx(2 * S2_SIGMA)
    assert field(d, "step") == pytest.approx(0.0)
    assert field(d, "surface") == 1.0
    assert all(field(d, k) == 1.0 for k in ("known_psf", "known_noise", "known_step", "known_top"))


def test_never_sharper_than_sentinel2():
    rng = np.random.default_rng(0)
    cam = fixed_camera(1, mtf=0.4, snr=50, bits=12, top=1.2)
    img, desc, _ = simulate(chip(1), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng)
    assert img.shape == (1, 90, 90) and img.dtype == np.float32
    assert field(desc[0], "psf") == pytest.approx(S2_SIGMA)  # asked 0.4, got S2's 0.25


def test_coarser_grid_and_blur_it_asked_for():
    rng = np.random.default_rng(0)
    cam = fixed_camera(2, mtf=0.1, snr=50, bits=12, top=1.2)
    img, desc, _ = simulate(chip(2), [S2_SIGMA] * 2, [S2_AB] * 2, [(500, 550)] * 2, 3, cam, rng)
    assert img.shape == (2, 30, 30)
    assert mtf_from_sigma(field(desc[0], "psf")) == pytest.approx(0.1)


def test_noise_matches_its_descriptor():
    rng = np.random.default_rng(1)
    cam = fixed_camera(1, mtf=0.3, snr=20, bits=14, top=1.6)
    flat = np.full((1, 300, 300), 0.5)  # S2's own noise is not in this chip, so set it to 0
    img, desc, _ = simulate(flat, [S2_SIGMA], [(0.0, 0.0)], [RED], 1, cam, rng)
    told = 10 ** (field(desc[0], "noise_bright") - 3)
    assert img[0].std() == pytest.approx(told, rel=0.05)


def test_noise_never_below_what_the_chip_had():
    rng = np.random.default_rng(2)
    cam = fixed_camera(1, mtf=0.25, snr=1000, bits=14, top=1.6)
    _, desc, _ = simulate(chip(1), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng)
    told = 10 ** (field(desc[0], "noise_bright") - 3)
    assert told == pytest.approx(math.sqrt(S2_AB[0] * 0.5 + S2_AB[1]), rel=1e-5)


def test_digitizing_steps_and_top():
    rng = np.random.default_rng(3)
    cam = fixed_camera(1, mtf=0.25, snr=300, bits=8, top=0.8)
    img, desc, _ = simulate(chip(1, value=0.95), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng)
    assert img.max() == pytest.approx(0.8)
    assert 10 ** (field(desc[0], "step") - 4) == pytest.approx(0.8 / 255)


def test_never_finer_than_sentinel2s_own_bits():
    """A 14-bit camera made from a Sentinel-2 band that saturates at 1.0 has only the
    information of S2's 12 bits: its step is 1.0 / 4095, not 1.2 / 16383."""
    rng = np.random.default_rng(8)
    cam = fixed_camera(1, mtf=0.25, snr=300, bits=14, top=1.2)
    _, desc, _ = simulate(chip(1), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng, s2_top=[1.0])
    assert 10 ** (field(desc[0], "step") - 4) == pytest.approx(1.0 / 4095, rel=1e-5)


def test_the_top_is_where_the_data_stop():
    """A band Sentinel-2 itself saturates at 0.5 cannot reach a camera's 1.2."""
    rng = np.random.default_rng(4)
    cam = fixed_camera(1, mtf=0.25, snr=300, bits=12, top=1.2)
    _, desc, got = simulate(chip(1), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng, s2_top=[0.5])
    assert field(desc[0], "top") == pytest.approx(0.5) and got[0]["top"] == pytest.approx(0.5)
    _, desc, _ = simulate(chip(1), [S2_SIGMA], [S2_AB], [RED], 1, cam, rng, s2_top=[None])
    assert field(desc[0], "top") == pytest.approx(1.2)


def test_label_mode_and_ignore():
    lab = np.zeros((6, 6), np.uint8)
    lab[:3, :3] = 2  # thick in the top-left block
    lab[:3, 3:] = 255  # top-right block unlabelled
    lab[3:, :3] = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]  # 6 thin against 3 clear
    assert downsample_label(lab, 3).tolist() == [[2, 255], [1, 0]]


def test_imagined_cameras_stay_in_range():
    rng = np.random.default_rng(5)
    for _ in range(200):
        q, bands = imagine_camera(rng, 5)
        assert q in (1, 3, 5) and len(bands) == 5
        for b in bands:
            assert 0.03 <= b["mtf"] <= 0.6 and 7 <= b["snr"] <= 390
            assert b["bits"] in (8, 10, 12, 14) and 0.6 <= b["top"] <= 1.6
    assert imagine_camera(rng, 3, q=5)[0] == 5


def test_a_20m_band_keeps_its_own_pixel_size():
    rng = np.random.default_rng(6)
    cam = fixed_camera(1, mtf=0.4, snr=50, bits=12, top=1.2)
    s20 = 2 * S2_SIGMA  # a 20 m band's blur, in 10 m pixels
    _, desc, _ = simulate(chip(1), [s20], [S2_AB], [(1541, 1683)], 1, cam, rng, s2_gsd=[20])
    assert field(desc[0], "sampling") == 1.0
    assert field(desc[0], "psf") == pytest.approx(s20)


def test_hide_and_blind():
    d = np.stack([describe(*RED, 10, 10, 0.25, S2_AB, 1e-4, 1.2)] * 3)
    rng = np.random.default_rng(7)
    seen = dict.fromkeys(("known_psf", "known_noise", "known_step", "known_top"), 0)
    for _ in range(2000):
        h = hide(d, rng)
        for name in seen:
            col = FIELDS.index(name)
            assert len(set(h[:, col])) == 1  # the same for every band
            seen[name] += int(h[0, col] == 0)
        assert (h[:, :5] == d[:, :5]).all()  # wavelengths, surface, sampling never hidden
    for n in seen.values():  # hidden 0.1 + 0.9 * 0.2 = 28% of the time
        assert 0.24 < n / 2000 < 0.32
    b = blind(d)
    assert (b[:, :3] == d[:, :3]).all() and (b[:, 3:] == 0).all()


def test_sentinel2_from_its_spec():
    """B04: MTF 0.225, the middle of ESA's 0.15-0.30 window; noise about 0.001 over dark
    ground with the sun overhead (SNR 142 at 108 W m-2 sr-1 um-1); saturation at
    pi * 484 / 1512.06 = 1.006. With the sun 60 degrees from overhead, k = pi / (E0 cos)
    doubles: a doubles, b quadruples, and the top doubles."""
    bands = {b.name: b for b in bands_from_spec("sentinel2_msi")}
    assert list(bands)[:4] == ["B01", "B02", "B03", "B04"] and len(bands) == 13
    red = bands["B04"]
    assert red.mtf == pytest.approx(0.225) and red.mtf_known
    assert red.sigma_px(10) == pytest.approx(sigma_from_mtf(0.225))
    a, b = red.noise(0.0)
    assert math.sqrt(a * 0.05 + b) == pytest.approx(0.00107, abs=2e-5)
    assert red.top(0.0) == pytest.approx(math.pi * 484 / 1512.06)
    assert red.noise(60.0) == pytest.approx((2 * a, 4 * b))
    assert red.top(60.0) == pytest.approx(2 * red.top(0.0))
    swir = bands["B11"]
    assert swir.gsd_m == 20 and not swir.mtf_known
    assert swir.sigma_px(10) == pytest.approx(2 * sigma_from_mtf(0.25))
