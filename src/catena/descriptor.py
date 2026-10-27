"""What the model is told about each band: its imaging chain as 14 numbers (A3).

    lo, mid, hi    shortest, central and longest wavelength, as log(nm - 300) - 2
    surface        1 for surface reflectance, 0 for top of atmosphere
    sampling       log2(band pixel size / grid pixel size): 1 for a 20 m band on a 10 m grid
    psf            blur width: the Gaussian PSF's sigma, in pixels of the grid
    noise_dark     log10(noise sigma) + 3 at reflectance 0.05
    noise_bright   the same at reflectance 0.5
    step           log10(one digital number, in reflectance) + 4
    top            the saturation level in reflectance, capped at 2
    known_psf, known_noise, known_step, known_top    1 when that value is known

A value nobody knows is 0 with its flag at 0. In training, descriptor dropout (hide) hides
known values the same way, so the model learns to work with gaps; the blind arm (blind)
keeps only the three wavelength fields, which is what SEnSeI is told.

bands_from_spec() reads a sensor spec (A2) into the numbers the simulator and the
descriptor need. Noise and saturation are specified in radiance; they move to reflectance
with the sun angle of each scene, because the same radiance is a higher reflectance when the
sun is low.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from catena.sensors import physics, spec

FIELDS = (
    "lo",  # spectral: the only fields the blind arm keeps
    "mid",
    "hi",
    "surface",  # processing: 1 surface reflectance, 0 top of atmosphere
    "sampling",  # spatial: band pixel size against the grid
    "psf",  # spatial: blur width in grid pixels
    "noise_dark",  # radiometric: noise at reflectance 0.05
    "noise_bright",  # radiometric: noise at reflectance 0.5
    "step",  # digital: one DN in reflectance
    "top",  # digital: saturation
    "known_psf",
    "known_noise",
    "known_step",
    "known_top",
)
N_FIELDS = len(FIELDS)
N_SPECTRAL = 3
S2_STEP = 1e-4  # Sentinel-2 L1C stores reflectance x 10000
DARK, BRIGHT = 0.05, 0.5  # the two reflectance levels the noise is told at

# what dropout hides together: the value columns of each group and their flag column
CHAIN = {"psf": ([5], 10), "noise": ([6, 7], 11), "step": ([8], 12), "top": ([9], 13)}


def wavelength_code(nm: float) -> float:
    """SEnSeIv1's scaling of a wavelength: log(nm - 300) - 2, about 2.8 to 5.6. Smooth: two
    bands with close wavelengths always get close codes."""
    return math.log(nm - 300.0) - 2.0


def describe(
    lo_nm: float,
    hi_nm: float,
    gsd_m: float,
    grid_m: float,
    mtf_nyquist: float | None = None,
    noise_ab: tuple[float, float] | None = None,
    step: float | None = None,
    top: float | None = None,
    surface: bool = False,
) -> np.ndarray:
    """One band's descriptor, 14 numbers. A chain value nobody knows stays 0, flag 0.

    mtf_nyquist is the MTF at the band's own Nyquist frequency (0.5 cycles per band pixel);
    noise_ab is (a, b) of var = a * rho + b in reflectance units."""
    d = np.zeros(N_FIELDS, np.float32)
    d[0:3] = [wavelength_code(x) for x in (lo_nm, (lo_nm + hi_nm) / 2, hi_nm)]
    d[3] = 1.0 if surface else 0.0
    d[4] = math.log2(gsd_m / grid_m)  # 0 native grid; 1 a 20 m band resampled to 10 m
    if mtf_nyquist is not None:
        d[5] = physics.sigma_from_mtf(mtf_nyquist) * gsd_m / grid_m  # blur in grid pixels
        d[10] = 1.0
    if noise_ab is not None:
        a, b = noise_ab
        d[6] = math.log10(math.sqrt(a * DARK + b)) + 3  # 0 means noise 0.001
        d[7] = math.log10(math.sqrt(a * BRIGHT + b)) + 3
        d[11] = 1.0
    if step is not None:
        d[8] = math.log10(step) + 4  # 0 means one DN = 0.0001
        d[12] = 1.0
    if top is not None:
        d[9] = min(top, 2.0)
        d[13] = 1.0
    return d


def hide(
    desc: np.ndarray, rng: np.random.Generator, p_one: float = 0.2, p_all: float = 0.1
) -> np.ndarray:
    """Descriptor dropout on one sample's (bands, 14) descriptors. A hidden value becomes 0
    with its flag 0, exactly how an unknown value looks at test time. A group is hidden for
    every band at once, as real gaps are (nobody tells us HEO's SNR for one band only)."""
    d = desc.copy()
    hide_all = rng.random() < p_all
    for cols, flag in CHAIN.values():
        if hide_all or rng.random() < p_one:
            d[:, cols] = 0.0
            d[:, flag] = 0.0
    return d


def blind(desc: np.ndarray) -> np.ndarray:
    """The blind arm's descriptors: the wavelengths, and every other field unknown."""
    d = desc.copy()
    d[..., N_SPECTRAL:] = 0.0
    return d


@dataclass(frozen=True)
class SpecBand:
    """One band of a sensor spec, in the numbers the simulator and the descriptor use."""

    name: str
    edges_nm: tuple[float, float]
    gsd_m: float
    mtf: float  # MTF at the band's own Nyquist; assumed when the spec gives none
    mtf_known: bool
    noise_radiance: tuple[float, float] | None  # var(L) = a L + b, from the spec's SNR
    e0: float | None  # W m-2 um-1
    saturation_radiance: float | None  # W m-2 sr-1 um-1

    def sigma_px(self, grid_m: float = 10.0) -> float:
        """PSF sigma in pixels of a grid: a 20 m band on the 10 m grid blurs twice as wide."""
        return physics.sigma_from_mtf(self.mtf) * self.gsd_m / grid_m

    def per_radiance(self, sun_zenith: float) -> float | None:
        """Reflectance of one radiance unit, pi / (E0 cos(sun zenith)), at 1 AU."""
        if self.e0 is None:
            return None
        return math.pi / (self.e0 * math.cos(math.radians(sun_zenith)))

    def noise(self, sun_zenith: float) -> tuple[float, float] | None:
        """Noise (a, b) of var = a rho + b in reflectance. rho = k L, so a becomes k a and b
        becomes k^2 b: the lower the sun, the noisier the same band in reflectance."""
        k = self.per_radiance(sun_zenith)
        if k is None or self.noise_radiance is None:
            return None
        a, b = self.noise_radiance
        return k * a, k * k * b

    def top(self, sun_zenith: float) -> float | None:
        """Saturation in reflectance, k L_sat."""
        k = self.per_radiance(sun_zenith)
        if k is None or self.saturation_radiance is None:
            return None
        return k * self.saturation_radiance


def bands_from_spec(
    name: str, assumed_mtf: float = 0.25, shot_fraction: float = 0.7
) -> list[SpecBand]:
    """The bands of a sensor spec (A2), in its order. A band without an MTF gets assumed_mtf
    (flagged mtf_known False); one SNR point is split with shot_fraction (A1)."""
    out = []
    for band in spec.load(name).bands:
        if band.lower_nm is None or band.upper_nm is None or not band.gsd_m.known:
            raise ValueError(f"{name} {band.name}: band edges and GSD are needed")
        mtf = band.mtf_along
        out.append(
            SpecBand(
                name=band.name,
                edges_nm=(band.lower_nm, band.upper_nm),
                gsd_m=band.gsd_m.value,
                mtf=mtf.value if mtf.known else assumed_mtf,
                mtf_known=mtf.known,
                noise_radiance=band.noise_model(shot_fraction),
                e0=band.e0.value if band.e0.known else None,
                saturation_radiance=(
                    band.saturation_radiance.value if band.saturation_radiance.known else None
                ),
            )
        )
    return out
