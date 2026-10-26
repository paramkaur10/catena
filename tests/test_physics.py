"""A1 tests: measure what the imaging-chain code does and compare it with what it claims."""

import numpy as np
import pytest

from catena.sensors import physics as P

# ---------------------------------------------------------------------------- helpers


def impulse(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size))
    img[size // 2, size // 2] = 1.0
    return img


def measured_transfer(output_of_impulse: np.ndarray) -> np.ndarray:
    """|FFT| of an impulse response: the transfer function at every frequency bin."""
    return np.abs(np.fft.fft2(output_of_impulse))


def sinusoid_amplitude(signal: np.ndarray, f: float) -> float:
    """Least-squares amplitude of a sinusoid of known frequency ``f`` (cycles/sample)."""
    n = np.arange(len(signal))
    design = np.stack(
        [np.cos(2 * np.pi * f * n), np.sin(2 * np.pi * f * n), np.ones_like(n, dtype=float)], 1
    )
    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


# ------------------------------------------------------------------------------ optics


@pytest.mark.parametrize("mtf", [0.05, 0.15, 0.30, 0.45])
def test_mtf_and_sigma_are_two_views_of_one_number(mtf):
    assert P.mtf_from_sigma(P.sigma_from_mtf(mtf)) == pytest.approx(mtf, rel=1e-12)


def test_typical_sensors_differ_by_fractions_of_a_pixel():
    # MTF at Nyquist from 0.45 down to 0.05 is a PSF width of only 0.40 -> 0.78 px
    assert P.sigma_from_mtf(0.45) == pytest.approx(0.402, abs=1e-3)
    assert P.sigma_from_mtf(0.05) == pytest.approx(0.779, abs=1e-3)


@pytest.mark.parametrize("mtf_rows, mtf_cols", [(0.30, 0.30), (0.10, 0.25), (0.45, 0.05)])
def test_apply_mtf_has_exactly_the_requested_transfer_function(mtf_rows, mtf_cols):
    s_r, s_c = P.sigma_from_mtf(mtf_rows), P.sigma_from_mtf(mtf_cols)
    out = P.apply_mtf(impulse(), s_r, s_c)
    measured = measured_transfer(out)
    fy = np.fft.fftfreq(128)[:, None]
    fx = np.fft.fftfreq(128)[None, :]
    expected = np.exp(-2 * np.pi**2 * (s_r**2 * fy**2 + s_c**2 * fx**2))
    assert np.abs(measured - expected).max() < 1e-3
    # at Nyquist along each axis the image keeps exactly the spec-sheet value
    assert measured[64, 0] == pytest.approx(mtf_rows, abs=1e-3)
    assert measured[0, 64] == pytest.approx(mtf_cols, abs=1e-3)


@pytest.mark.parametrize("mtf", [0.05, 0.15, 0.30])
def test_sampled_gaussian_kernels_roughly_double_the_mtf_at_nyquist(mtf):
    """The pitfall this module avoids: a Gaussian kernel sampled on the pixel grid."""
    sigma = P.sigma_from_mtf(mtf)
    t = np.arange(-6, 7)
    sampled_kernel = np.exp(-0.5 * (t / sigma) ** 2)
    actual = P.discrete_kernel_mtf(sampled_kernel)
    assert actual == pytest.approx(2 * mtf, rel=0.02)


def test_discrete_kernel_mtf_of_known_kernels():
    assert P.discrete_kernel_mtf(np.array([1.0])) == pytest.approx(1.0)
    # [1, 2, 1] / 4 cancels the Nyquist pattern exactly
    assert P.discrete_kernel_mtf(np.array([1.0, 2.0, 1.0])) == pytest.approx(0.0, abs=1e-12)
    # [1, 1, 1] / 3 keeps one third (sign flipped) at Nyquist
    assert P.discrete_kernel_mtf(np.ones(3)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("f", [0.10, 0.25, 0.40, 0.45])
@pytest.mark.parametrize("mtf", [0.08, 0.30])
def test_blur_and_sample_gives_the_requested_mtf(f, mtf):
    """Image a fine sinusoid with a sensor and measure the contrast it keeps."""
    q, n = 5, 96
    sigma = P.sigma_from_mtf(mtf)
    # scene sample i sits at sensor coordinate (i - (q - 1) / 2) / q, so pixel j's centre is at j
    x = (np.arange(n * q) - (q - 1) / 2) / q
    scene = np.tile(0.3 + 0.1 * np.cos(2 * np.pi * f * x), (n * q, 1))
    image = P.blur_and_sample(scene, q, sigma)
    row = image[n // 2, 10:-10]
    assert sinusoid_amplitude(row, f) == pytest.approx(0.1 * P.mtf_from_sigma(sigma, f), rel=1e-3)


def test_blur_and_sample_samples_at_pixel_centres():
    q, n = 5, 21
    scene = np.zeros((n * q, n * q))
    j = 10
    scene[q * j + 2, q * j + 2] = 1.0  # a point exactly at the centre of sensor pixel j
    image = P.blur_and_sample(scene, q, 0.6)
    assert image[j, j - 1] == pytest.approx(image[j, j + 1], rel=1e-9)
    assert image[j - 1, j] == pytest.approx(image[j + 1, j], rel=1e-9)
    assert image[j, j] == image.max()


def test_blur_and_sample_rejects_even_oversampling():
    with pytest.raises(ValueError):
        P.blur_and_sample(np.zeros((8, 8)), 2, 0.5)


def test_flat_images_stay_flat():
    flat = np.full((3, 40, 50), 0.37)
    assert np.allclose(P.apply_mtf(flat, 0.7, 0.4), 0.37)


# ------------------------------------------------------------------------------- noise


def test_noise_from_snr_reproduces_the_spec():
    a, b = P.noise_from_snr(snr=142.0, reference=0.224, shot_fraction=0.6)
    assert P.snr_at(0.224, a, b) == pytest.approx(142.0)
    assert a * 0.224 / P.noise_variance(0.224, a, b) == pytest.approx(0.6)


def test_dark_scenes_have_lower_snr_than_the_spec_level():
    a, b = P.noise_from_snr(snr=100.0, reference=0.2)
    assert P.snr_at(0.05, a, b) < 100.0 / 2  # thin clouds over water live here


def test_two_point_noise_fit_is_exact():
    a, b = P.noise_from_two_points(0.05, 80.0, 0.5, 400.0)
    assert P.snr_at(0.05, a, b) == pytest.approx(80.0)
    assert P.snr_at(0.5, a, b) == pytest.approx(400.0)


def test_two_point_noise_fit_rejects_impossible_specs():
    # variance falling with signal is not a detector
    with pytest.raises(ValueError):
        P.noise_from_two_points(0.1, 50.0, 0.5, 1000.0)


@pytest.mark.parametrize("level", [0.03, 0.3])
def test_added_noise_matches_the_model(level):
    a, b = P.noise_from_snr(snr=60.0, reference=0.2, shot_fraction=0.8)
    rng = np.random.default_rng(0)
    noisy = P.add_noise(np.full(400_000, level), a, b, rng)
    assert noisy.std() == pytest.approx(float(P.noise_std(level, a, b)), rel=0.01)
    assert noisy.mean() == pytest.approx(level, abs=1e-4)


# ------------------------------------------------------------------------ digitisation


def test_digitize_rounds_to_the_adc_grid_and_saturates():
    step = P.quantization_step(0.0, 1.0, 8)
    x = np.array([-0.2, 0.0, 0.1234, 0.5, 0.99, 1.7])
    out = P.digitize(x, 0.0, 1.0, 8)
    assert out[0] == 0.0 and out[-1] == 1.0
    assert np.allclose(np.round(out / step) * step, out)
    assert np.abs(out[1:-1] - x[1:-1]).max() <= step / 2 + 1e-12


def test_quantization_error_variance_is_step_squared_over_twelve():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 0.9, 500_000)
    step = P.quantization_step(0.0, 1.0, 8)
    err = P.digitize(x, 0.0, 1.0, 8) - x
    assert err.var() == pytest.approx(P.quantization_variance(step), rel=0.01)


# -------------------------------------------------------------------------- radiometry


def test_radiance_reflectance_round_trip_and_sun_angle():
    rho = P.radiance_to_reflectance(108.0, esun=1512.0)
    assert rho == pytest.approx(np.pi * 108.0 / 1512.0)
    assert P.reflectance_to_radiance(rho, esun=1512.0) == pytest.approx(108.0)
    # the same radiance is a brighter reflectance under a low sun
    assert P.radiance_to_reflectance(108.0, 1512.0, sun_zenith_deg=60.0) == pytest.approx(2 * rho)
