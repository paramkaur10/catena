"""The imaging chain: how a sensor turns a scene into numbers.

A spaceborne optical sensor produces an image in a few physical steps, and each step
hides a different kind of cloud:

    scene ─► optics + detector ─► pixel grid ─► noise ─► ADC
              (MTF)               (sampling)    (SNR)    (bits, saturation)

* MTF smears small and broken clouds and widens cloud edges.
* Noise decides whether the faint brightening of a thin cloud is visible at all.
* The ADC rounds away differences smaller than one digital number and flattens
  bright clouds that exceed the saturation level.

This module implements those four links on a single pixel grid. Spectral bands,
resampling between grids and band-to-band misregistration come in later steps.

Conventions used throughout Catena
----------------------------------
* Images are arrays shaped ``(..., rows, cols)``. For pushbroom sensors the rows run
  along track and the columns across track.
* Spatial frequency ``f`` is measured in cycles per pixel. The finest pattern a grid
  can hold (one bright, one dark pixel) has ``f = 0.5``, the Nyquist frequency.
* Band values are physical units, TOA reflectance for reflective bands.
* The system MTF (optics, detector footprint, motion and electronics together) is
  modelled as a Gaussian, ``MTF(f) = exp(-2 pi^2 sigma^2 f^2)``, and parameterised by its
  value at Nyquist, the number every spec sheet gives. ``sigma`` is the standard
  deviation, in pixels, of the matching Gaussian point spread function (PSF).
* Noise is signal dependent, ``var(x) = a * max(x, 0) + b``: photon shot noise grows
  with the signal (``a x``), read-out, dark-current and electronic noise do not (``b``).

Why blur is applied in the frequency domain
-------------------------------------------
A Gaussian kernel sampled on the pixel grid does not have the MTF of the Gaussian it
was sampled from. Sampling the kernel folds its spectrum, and at Nyquist the two folded
copies add up, so the response there is roughly doubled: ``scipy.ndimage.gaussian_filter``
with the sigma of an MTF-0.30 sensor leaves 0.59 at Nyquist. Catena multiplies the
image spectrum by the transfer function itself, so the MTF it claims for an image is
the MTF the image has. :func:`discrete_kernel_mtf` measures the true MTF of any kernel,
for example the PSF kernel inside another simulator.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

NYQUIST = 0.5  # cycles per pixel

# --------------------------------------------------------------------------------------
# 1. Optics and detector: MTF and the point spread function
# --------------------------------------------------------------------------------------


def sigma_from_mtf(mtf_nyquist: float) -> float:
    """PSF standard deviation (pixels) of the Gaussian system with this MTF at Nyquist.

    Solves ``exp(-2 pi^2 sigma^2 0.5^2) = mtf_nyquist`` for ``sigma``.
    """
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError(f"MTF at Nyquist must be in (0, 1), got {mtf_nyquist}")
    return math.sqrt(-2.0 * math.log(mtf_nyquist)) / math.pi


def gaussian_mtf(f: np.ndarray | float, sigma: float) -> np.ndarray:
    """MTF at frequency ``f`` (cycles/pixel) of a Gaussian PSF with std ``sigma`` (pixels)."""
    f = np.asarray(f, dtype=np.float64)
    return np.exp(-2.0 * math.pi**2 * sigma**2 * f**2)


def mtf_from_sigma(sigma: float, f: float = NYQUIST) -> float:
    """MTF of a Gaussian PSF with std ``sigma`` (pixels) at one frequency (default Nyquist)."""
    return float(gaussian_mtf(f, sigma))


def discrete_kernel_mtf(
    kernel: np.ndarray, f: np.ndarray | float = NYQUIST, axis: int = -1
) -> np.ndarray:
    """True MTF of a discrete, centred blur kernel at frequency ``f`` (cycles/pixel).

    For a 2-D kernel the response along ``axis`` is that of its marginal (the kernel
    summed over the other axis). The result is normalised so a flat image is unchanged.
    """
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim == 2:
        k = k.sum(axis=0 if axis in (-1, 1) else 1)
    if k.ndim != 1 or len(k) % 2 == 0:
        raise ValueError("kernel must be 1-D or 2-D with odd, centred support")
    n = np.arange(len(k)) - (len(k) - 1) // 2
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    response = np.abs(np.exp(-2j * math.pi * f[:, None] * n[None, :]) @ k) / k.sum()
    return response if response.size > 1 else response[0]


def _padding(sigma_px: float) -> int:
    """Reflection margin for spectral filtering: covers the PSF tail (5 sigma) plus a few
    pixels for the small ripples from cutting the transfer function at Nyquist."""
    return math.ceil(5.0 * sigma_px) + 8


def _spectral_filter(
    image: np.ndarray, transfer: Callable[[np.ndarray, np.ndarray], np.ndarray], pad: int
) -> np.ndarray:
    """Multiply the spectrum of ``image`` (..., H, W) by ``transfer(fy, fx)``.

    The image is mirrored by ``pad`` pixels on every side first, so the circular FFT
    does not wrap the right edge onto the left one.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("image must have at least 2 dimensions (..., rows, cols)")
    h, w = x.shape[-2:]
    widths = [(0, 0)] * (x.ndim - 2) + [(pad, pad), (pad, pad)]
    xp = np.pad(x, widths, mode="symmetric")
    fy = np.fft.fftfreq(xp.shape[-2])[:, None]
    fx = np.fft.rfftfreq(xp.shape[-1])[None, :]
    spectrum = np.fft.rfft2(xp) * transfer(fy, fx)
    out = np.fft.irfft2(spectrum, s=xp.shape[-2:])
    return out[..., pad : pad + h, pad : pad + w]


def apply_mtf(image: np.ndarray, sigma_rows: float, sigma_cols: float | None = None) -> np.ndarray:
    """Blur ``image`` with a Gaussian MTF on its own pixel grid.

    ``sigma_rows`` acts along the rows axis (along track), ``sigma_cols`` across track
    (defaults to ``sigma_rows``). Both are PSF standard deviations in pixels of this
    grid, e.g. ``sigma_from_mtf(0.3)``. The transfer function is exact for every
    frequency up to Nyquist.
    """
    sigma_cols = sigma_rows if sigma_cols is None else sigma_cols
    if sigma_rows < 0 or sigma_cols < 0:
        raise ValueError("sigma must be >= 0")
    if sigma_rows == 0 and sigma_cols == 0:
        return np.array(image, dtype=np.float64, copy=True)
    c = -2.0 * math.pi**2

    def transfer(fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
        return np.exp(c * (sigma_rows**2 * fy**2 + sigma_cols**2 * fx**2))

    return _spectral_filter(image, transfer, _padding(max(sigma_rows, sigma_cols)))


def blur_and_sample(
    scene: np.ndarray, oversample: int, sigma_rows: float, sigma_cols: float | None = None
) -> np.ndarray:
    """Image a finely sampled scene: blur with the system PSF, sample at pixel centres.

    ``scene`` is the "true" scene on a grid ``oversample`` times finer than the
    sensor's, shaped ``(..., oversample * H, oversample * W)``. ``sigma_rows`` and
    ``sigma_cols`` are PSF standard deviations in *sensor* pixels. Returns
    ``(..., H, W)``.

    The PSF already contains the detector footprint, so each sensor pixel is a point
    sample of the blurred scene at its centre. ``oversample`` must be odd so every
    pixel centre lands exactly on a scene sample. Detail finer than the sensor's
    Nyquist frequency aliases into the image as it would in a real instrument.
    """
    q = int(oversample)
    if q < 1 or q % 2 == 0:
        raise ValueError(f"oversample must be a positive odd integer, got {oversample}")
    rows, cols = np.shape(scene)[-2:]
    if rows % q or cols % q:
        raise ValueError(f"scene size {(rows, cols)} is not a multiple of oversample={q}")
    sigma_cols = sigma_rows if sigma_cols is None else sigma_cols
    blurred = apply_mtf(scene, sigma_rows * q, sigma_cols * q)  # sensor px -> scene px
    centre = (q - 1) // 2
    return blurred[..., centre::q, centre::q]


# --------------------------------------------------------------------------------------
# 2. Detector noise: var(x) = a x + b
# --------------------------------------------------------------------------------------


def noise_from_snr(snr: float, reference: float, shot_fraction: float = 0.7) -> tuple[float, float]:
    """Noise coefficients ``(a, b)`` from one spec-sheet SNR value.

    A spec sheet gives the SNR at one reference level ``x_ref``. That fixes the total
    noise variance there, ``(x_ref / SNR)^2``, but not how it splits between shot noise
    (proportional to the signal) and the signal-independent floor. ``shot_fraction`` is
    the shot-noise share of the variance at ``x_ref``; well-exposed CCD/CMOS imagers sit
    around 0.5-0.9.
    """
    if snr <= 0 or reference <= 0:
        raise ValueError("snr and reference must be positive")
    if not 0.0 <= shot_fraction <= 1.0:
        raise ValueError("shot_fraction must be in [0, 1]")
    var_ref = (reference / snr) ** 2
    return shot_fraction * var_ref / reference, (1.0 - shot_fraction) * var_ref


def noise_from_two_points(x1: float, snr1: float, x2: float, snr2: float) -> tuple[float, float]:
    """Noise coefficients ``(a, b)`` from SNR at two signal levels.

    Some spec sheets give SNR at two levels (Landsat 8 quotes Ltyp and Lhigh). Two
    points determine both coefficients, so no shot-fraction assumption is needed.
    """
    if x1 == x2:
        raise ValueError("the two levels must differ")
    if min(x1, x2, snr1, snr2) <= 0:
        raise ValueError("levels and SNRs must be positive")
    v1, v2 = (x1 / snr1) ** 2, (x2 / snr2) ** 2
    a = (v2 - v1) / (x2 - x1)
    b = v1 - a * x1
    if a < 0 or b < 0:
        raise ValueError(
            f"SNR {snr1} at {x1} and {snr2} at {x2} do not fit var = a x + b with a, b >= 0 "
            f"(got a={a:.3g}, b={b:.3g}); check the levels and units"
        )
    return a, b


def noise_variance(x: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Noise variance at signal level ``x``."""
    return a * np.maximum(np.asarray(x, dtype=np.float64), 0.0) + b


def noise_std(x: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Noise standard deviation at signal level ``x``."""
    return np.sqrt(noise_variance(x, a, b))


def snr_at(x: np.ndarray | float, a: float, b: float) -> np.ndarray:
    """Signal-to-noise ratio at signal level ``x``."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(noise_std(x, a, b), 1e-30)


def add_noise(image: np.ndarray, a: float, b: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with ``var = a x + b`` at every pixel.

    The Gaussian form approximates Poisson photon statistics, which is accurate at the
    signal levels of daytime Earth observation.
    """
    x = np.asarray(image, dtype=np.float64)
    return x + rng.standard_normal(x.shape) * noise_std(x, a, b)


# --------------------------------------------------------------------------------------
# 3. Digitisation: bit depth and saturation
# --------------------------------------------------------------------------------------


def quantization_step(range_min: float, saturation: float, bits: int) -> float:
    """Size of one digital number in band units for an ADC spanning [range_min, saturation]."""
    if saturation <= range_min:
        raise ValueError("saturation must exceed range_min")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    return (saturation - range_min) / (2**bits - 1)


def quantization_variance(step: float) -> float:
    """Variance of rounding to a grid of size ``step`` (uniform error, ``step^2 / 12``).

    Valid when the signal varies by more than a step within the data, which noise
    usually guarantees.
    """
    return step**2 / 12.0


def digitize(image: np.ndarray, range_min: float, saturation: float, bits: int) -> np.ndarray:
    """Clip to [range_min, saturation] and round to the ADC grid; returns band units."""
    step = quantization_step(range_min, saturation, bits)
    x = np.clip(np.asarray(image, dtype=np.float64), range_min, saturation)
    return range_min + np.round((x - range_min) / step) * step


# --------------------------------------------------------------------------------------
# 4. Radiometry: reading spec sheets quoted in radiance
# --------------------------------------------------------------------------------------


def radiance_to_reflectance(
    radiance: np.ndarray | float,
    esun: float,
    sun_zenith_deg: float = 0.0,
    earth_sun_au: float = 1.0,
) -> np.ndarray:
    """TOA radiance (W m-2 sr-1 um-1) to TOA reflectance.

    ``rho = pi L d^2 / (E_sun cos(theta_s))``. Spec-sheet reference radiances are
    converted with the sun at zenith and ``d = 1`` AU unless stated otherwise.
    """
    cos_sz = math.cos(math.radians(sun_zenith_deg))
    if cos_sz <= 0:
        raise ValueError("sun zenith must be below 90 degrees")
    return math.pi * np.asarray(radiance, dtype=np.float64) * earth_sun_au**2 / (esun * cos_sz)


def reflectance_to_radiance(
    reflectance: np.ndarray | float,
    esun: float,
    sun_zenith_deg: float = 0.0,
    earth_sun_au: float = 1.0,
) -> np.ndarray:
    """Inverse of :func:`radiance_to_reflectance`."""
    cos_sz = math.cos(math.radians(sun_zenith_deg))
    if cos_sz <= 0:
        raise ValueError("sun zenith must be below 90 degrees")
    return np.asarray(reflectance, dtype=np.float64) * esun * cos_sz / (math.pi * earth_sun_au**2)
