"""Virtual cameras (A4): a real 10 m Sentinel-2 chip shown as another camera would have
made it, with an honest descriptor of what that camera got.

    imagine_camera()    a random camera: pixel size, blur, noise, bits and saturation
    fixed_camera()      one chain for every band, for validation and tests
    simulate()          that camera's image of a chip, band by band, and its descriptors
    downsample_label()  the label moved onto the camera's grid

Three rules keep it honest. Only worse: blur and noise are added, never removed, so a
camera asked to be sharper or quieter than Sentinel-2 gets Sentinel-2's own. The
descriptor tells what the camera got, not what was asked. The label follows the pixels:
each new pixel takes the most common class of its q x q block, or 255 when less than half
of the block is labelled.
"""

from __future__ import annotations

import math

import numpy as np

from catena.descriptor import S2_LEVELS, S2_STEP, describe
from catena.sensors.physics import (
    blur_and_sample,
    digitize,
    mtf_from_sigma,
    noise_from_snr,
    sigma_from_mtf,
)

Q_CHOICES = (1, 3, 5)  # 10, 30 or 50 m pixels: odd, so every new pixel centre is an old one
GRID_M = 10.0  # Sentinel-2's grid in CloudSEN12+
SNR_LEVEL = 0.3  # the reflectance at which a camera's SNR is drawn


def imagine_camera(
    rng: np.random.Generator,
    n_bands: int,
    q: int | None = None,
    tops: tuple[float, float] = (0.6, 1.6),
) -> tuple[int, list[dict]]:
    """A random camera: one pixel size (q, drawn unless given), a base chain, and small
    differences between its bands, as real bands have. Its saturation is drawn from `tops`,
    in reflectance: 0.6-1.6 in the pilot; the big run reaches down to 0.25, where Eaglet II,
    HEO's pan band and Landsat 7 clip bright cloud."""
    q = int(rng.choice(Q_CHOICES)) if q is None else int(q)
    mtf = rng.uniform(0.05, 0.4)
    snr = math.exp(rng.uniform(math.log(10), math.log(300)))  # at reflectance 0.3
    shot = rng.uniform(0.3, 0.9)
    bits = int(rng.choice([8, 10, 12, 14]))
    top = rng.uniform(*tops)
    bands = [
        {
            "mtf": float(np.clip(mtf * rng.uniform(0.8, 1.2), 0.03, 0.6)),
            "snr": float(snr * rng.uniform(0.7, 1.3)),
            "shot": float(shot),
            "bits": bits,
            "top": float(top),
        }
        for _ in range(n_bands)
    ]
    return q, bands


def fixed_camera(
    n_bands: int, mtf: float, snr: float, bits: int, top: float, shot: float = 0.7
) -> list[dict]:
    """The same chain for every band."""
    chain = {"mtf": mtf, "snr": snr, "shot": shot, "bits": bits, "top": top}
    return [dict(chain) for _ in range(n_bands)]


def simulate(chip, s2_sigma, s2_ab, edges_nm, q, bands, rng, s2_gsd=None, s2_top=None):
    """Show a 10 m Sentinel-2 chip as a virtual camera with pixels q times bigger.

    chip      (n, H, W) TOA reflectance on the 10 m grid, H and W multiples of q
    s2_sigma  PSF sigma of each band, in 10 m pixels (blur the chip already has)
    s2_ab     noise (a, b) of each band in reflectance (noise the chip already has)
    edges_nm  (lo, hi) of each band
    bands     the camera, one chain per band (imagine_camera, fixed_camera)
    s2_gsd    each band's own pixel size in m (20 and 60 m bands sit on the 10 m grid)
    s2_top    each band's saturation in reflectance, where the chip's data already stop
    Returns the image (n, H/q, W/q) as float32, one honest descriptor per band, and per band
    what the camera really got (for logs and figures).
    """
    out, descs, got = [], [], []
    grid = GRID_M * q
    for k, cam in enumerate(bands):
        want = sigma_from_mtf(cam["mtf"])  # PSF sigma asked for, in new pixels
        have = s2_sigma[k] / q  # what S2 already did, in new pixels
        sigma = max(want, have)  # we can add blur, never remove it
        extra = math.sqrt(sigma**2 - have**2)
        x = blur_and_sample(chip[k], q, extra)

        # noise the chip already has, after the extra blur (in 10 m pixels: extra * q)
        keep = min(1.0, 1.0 / (4 * math.pi * (extra * q) ** 2)) if extra > 0 else 1.0
        a0, b0 = keep * s2_ab[k][0], keep * s2_ab[k][1]
        at, bt = noise_from_snr(cam["snr"], SNR_LEVEL, cam["shot"])
        da, db = max(at - a0, 0.0), max(bt - b0, 0.0)  # add only what is missing
        x = x + rng.standard_normal(x.shape) * np.sqrt(da * np.clip(x, 0, None) + db)

        x = digitize(x, 0.0, cam["top"], cam["bits"])
        finest = S2_STEP  # nothing made from the chip is finer than the chip
        top = cam["top"]
        if s2_top is not None and s2_top[k] is not None:
            top = min(top, s2_top[k])  # the data cannot go above where S2 itself saturated
            finest = max(finest, s2_top[k] / S2_LEVELS)  # nor finer than S2's own 12 bits
        step = max(cam["top"] / (2 ** cam["bits"] - 1), finest)
        gsd = max(grid, s2_gsd[k] if s2_gsd is not None else GRID_M)  # a 60 m band stays 60 m
        out.append(x)
        descs.append(
            describe(
                *edges_nm[k],
                gsd_m=gsd,
                grid_m=grid,
                mtf_nyquist=mtf_from_sigma(sigma * grid / gsd),  # at the band's own Nyquist
                noise_ab=(a0 + da, b0 + db),
                step=step,
                top=top,
            )
        )
        snr = SNR_LEVEL / math.sqrt((a0 + da) * SNR_LEVEL + b0 + db)
        got.append({"mtf": mtf_from_sigma(sigma), "snr": snr, "step": step, "top": top})
    return np.stack(out).astype(np.float32), np.stack(descs), got


def downsample_label(label: np.ndarray, q: int, ignore: int = 255) -> np.ndarray:
    """Move a label to a grid q times coarser: the most common class in each q x q block,
    or `ignore` when fewer than half of the block's pixels are labelled."""
    if q == 1:
        return label.copy()
    h, w = label.shape
    blocks = label.reshape(h // q, q, w // q, q).transpose(0, 2, 1, 3).reshape(h // q, w // q, -1)
    counts = np.stack([(blocks == c).sum(-1) for c in range(4)], -1)  # classes 0..3
    out = counts.argmax(-1).astype(np.uint8)
    out[counts.sum(-1) < q * q / 2] = ignore
    return out
