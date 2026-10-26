"""Sensor specifications: every number the model is conditioned on, and where it came from (A2).

A spec is a YAML file in ``catena/sensors/specs/``. It describes one sensor in four parts:

* ``instrument``  what the camera is: imaging principle, detector, optics, ADC bits;
* ``product``     what we receive: processing level, units, integer scaling, special codes;
* ``processing``  what the ground or on-board chain did to the pixels, in order
                  (binning, sharpening, denoising, co-registration, resampling);
* ``bands``       per band: spectral edges, GSD, MTF, SNR, saturation, E0, registration, timing.

Every physical number can carry its provenance::

    mtf_along: {value: 0.25, kind: measured, source: gascon2017}

``kind`` says how far to trust it:

=============  ==========================================================================
measured       characterised on this instrument: lab, in flight, or read from real products
nominal        the operator's published design value
requirement    a specification limit; a range is allowed, e.g. ``{range: [0.15, 0.30]}``
derived        computed by us from other values
assumed        our placeholder, to be replaced
estimated      estimated from the images themselves (step A5)
=============  ==========================================================================

``source`` must be a key of the file's ``sources`` table, so every number can be cited.
A missing number is ``null`` (or ``{value: null, note: "asked Argotec"}``): the model then
falls back to an estimate, never to a silent default.

Values shared by many bands go in ``defaults``. A mapping in ``defaults`` is merged into each
band's own mapping, so ``defaults: {snr: {kind: requirement, source: esa_msi}}`` gives every SNR
point of the file that provenance unless a band says otherwise.

Why this layer exists: specifications mix radiance and reflectance, requirements and
measurements, instrument and product. Making all of that explicit here means nothing
downstream guesses. Turning a spec into what the model sees in one scene (noise in reflectance
units for that sun angle, the MTF left after binning, sharpening and resampling) is step A3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from catena.sensors import physics

SPEC_DIR = Path(__file__).with_name("specs")

KINDS = ("measured", "nominal", "requirement", "derived", "assumed", "estimated")
UNITS = ("toa_reflectance", "surface_reflectance", "radiance", "dn")
IMAGING = ("pushbroom", "whiskbroom", "frame", "unknown")
RESAMPLE_KERNELS = ("nearest", "bilinear", "cubic", "lanczos", "unknown")
PROCESSING_OPS = (
    "demosaic",
    "bin",
    "sharpen",
    "denoise",
    "coregister",
    "resample",
    "compress",
    "pansharpen",
)

_TOP_KEYS = {
    "id",
    "name",
    "platform",
    "note",
    "sources",
    "instrument",
    "product",
    "processing",
    "defaults",
    "bands",
}
_INSTRUMENT_TEXT = {"imaging", "detector", "note"}
_INSTRUMENT_VALUES = {
    "adc_bits",
    "tdi_stages_max",
    "pixel_pitch_um",
    "focal_length_m",
    "aperture_m",
    "altitude_km",
}
_PRODUCT_KEYS = {
    "level",
    "units",
    "dtype",
    "bits",
    "scale",
    "offset",
    "nodata",
    "saturation_code",
    "max_reflectance",
    "note",
}
_VALUE_KEYS = {"value", "kind", "source", "range", "note"}
_BAND_VALUES = {
    "spectral",
    "gsd_m",
    "pixel_m",
    "mtf_along",
    "mtf_across",
    "saturation_radiance",
    "e0",
    "registration_sigma_px",
    "time_offset_s",
}
_BAND_KEYS = _BAND_VALUES | {"name", "common_name", "snr", "srf", "note"}
_STEP_KEYS = {"op", "stage", "bands", "kind", "source", "note"}


class SpecError(ValueError):
    """A spec file is malformed or contradicts itself."""


# ---------------------------------------------------------------------------------- values


@dataclass(frozen=True)
class Value:
    """A number with provenance. ``low``/``high`` hold a range (requirement windows, band edges).

    A two-sided range without an explicit value takes its midpoint as the value: the best
    guess when all we know is "somewhere in this window". A one-sided range (``[null, 0.45]``)
    leaves the value unknown and keeps only the bound.
    """

    value: float | None = None
    kind: str | None = None
    source: str | None = None
    low: float | None = None
    high: float | None = None
    note: str | None = None

    @property
    def known(self) -> bool:
        return self.value is not None

    @property
    def bounded(self) -> bool:
        return self.low is not None or self.high is not None


@dataclass(frozen=True)
class SNRPoint:
    """SNR measured or required at one radiance level (W m-2 sr-1 um-1)."""

    snr: float
    radiance: float
    kind: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class ProcessingStep:
    """One operation of the processing chain. ``bands`` is None when it applies to all bands."""

    op: str
    params: Mapping[str, Any]
    bands: tuple[str, ...] | None = None
    stage: str | None = None
    kind: str | None = None
    source: str | None = None
    note: str | None = None

    def applies_to(self, band: str) -> bool:
        return self.bands is None or band in self.bands


@dataclass(frozen=True)
class Band:
    name: str
    common_name: str
    spectral: Value  # low/high = FWHM edges in nm
    gsd_m: Value  # native ground sampling distance at nadir
    pixel_m: Value  # pixel spacing of the delivered product
    mtf_along: Value  # system MTF at Nyquist (0.5 cycles per native pixel)
    mtf_across: Value
    snr: tuple[SNRPoint, ...]
    saturation_radiance: Value  # W m-2 sr-1 um-1
    e0: Value  # band-averaged solar irradiance, W m-2 um-1
    registration_sigma_px: Value  # 1-sigma residual misalignment to the other bands
    time_offset_s: Value  # acquisition time after the first band of the scene
    srf: str | None = None  # "file.csv:column" under data/srf/
    note: str | None = None

    @property
    def lower_nm(self) -> float | None:
        return self.spectral.low

    @property
    def upper_nm(self) -> float | None:
        return self.spectral.high

    @property
    def centre_nm(self) -> float | None:
        return self.spectral.value

    @property
    def fwhm_nm(self) -> float | None:
        if self.spectral.low is None or self.spectral.high is None:
            return None
        return self.spectral.high - self.spectral.low

    def psf_sigma_px(self) -> tuple[float | None, float | None]:
        """Gaussian PSF width (native pixels) along and across track, from the MTF (A1)."""
        return _sigma(self.mtf_along), _sigma(self.mtf_across)

    def noise_model(self, shot_fraction: float = 0.7) -> tuple[float, float] | None:
        """Noise variance ``a * L + b`` in radiance units, or None if no SNR is known.

        Two SNR points at different radiances fix ``a`` (shot noise) and ``b`` (read noise).
        One point fixes only their sum, so ``shot_fraction`` says how much of the noise at that
        radiance is shot noise (A1). Converting to reflectance units needs the sun angle of a
        scene, so it happens in A3.
        """
        if len(self.snr) >= 2:
            p, q = sorted(self.snr[:2], key=lambda s: s.radiance)
            return physics.noise_from_two_points(p.radiance, p.snr, q.radiance, q.snr)
        if len(self.snr) == 1:
            p = self.snr[0]
            return physics.noise_from_snr(p.snr, p.radiance, shot_fraction)
        return None


def _sigma(mtf: Value) -> float | None:
    return physics.sigma_from_mtf(mtf.value) if mtf.known else None


# ---------------------------------------------------------------------------------- sensors


@dataclass(frozen=True)
class Sensor:
    id: str
    name: str
    platform: str
    sources: Mapping[str, str]
    instrument: Mapping[str, Any]
    product: Mapping[str, Any]
    processing: tuple[ProcessingStep, ...] | None  # None = not documented
    bands: tuple[Band, ...]
    note: str | None = None

    @property
    def band_names(self) -> list[str]:
        return [b.name for b in self.bands]

    def band(self, name: str) -> Band:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"{self.id} has no band {name!r}; bands are {self.band_names}")

    def steps_for(self, band: str) -> list[ProcessingStep]:
        """Processing steps that touched this band, in order."""
        self.band(band)
        return [s for s in self.processing or () if s.applies_to(band)]

    def completeness(self) -> dict[str, str]:
        """How well each descriptor is known: the weakest status across bands.

        ``measured`` > ``nominal`` > ``requirement`` > ``derived`` > ``assumed`` > ``estimated``,
        then ``unspecified`` (a number without provenance), ``bound`` (only a limit),
        ``partial`` (some bands missing) and ``missing``.
        """
        report = {
            field: _summarise([_status(b, field) for b in self.bands]) for field in BAND_FIELDS
        }
        if self.processing is None:
            report["processing"] = "missing"
        elif not self.processing:
            report["processing"] = "unspecified"  # documented as "nothing applied"
        else:
            report["processing"] = _summarise([s.kind or "unspecified" for s in self.processing])
        return report


BAND_FIELDS = ("spectral", "gsd", "mtf", "snr", "saturation", "e0", "registration", "timing")
_RANK = (*KINDS, "unspecified", "bound", "partial", "missing")


def _value_status(v: Value) -> str:
    if v.known:
        return v.kind or "unspecified"
    return "bound" if v.bounded else "missing"


def _status(band: Band, field: str) -> str:
    if field == "spectral":
        return (band.spectral.kind or "unspecified") if band.fwhm_nm is not None else "missing"
    if field == "snr":
        return _summarise([p.kind or "unspecified" for p in band.snr]) if band.snr else "missing"
    if field == "mtf":
        return _summarise([_value_status(band.mtf_along), _value_status(band.mtf_across)])
    value = {
        "gsd": band.gsd_m,
        "saturation": band.saturation_radiance,
        "e0": band.e0,
        "registration": band.registration_sigma_px,
        "timing": band.time_offset_s,
    }[field]
    return _value_status(value)


def _summarise(statuses: Sequence[str]) -> str:
    if all(s == "missing" for s in statuses):
        return "missing"
    if any(s == "missing" for s in statuses):
        return "partial"
    return max(statuses, key=_RANK.index)


# ---------------------------------------------------------------------------------- loading


def available() -> list[str]:
    """Ids of the specs shipped with the package (files starting with ``_`` are templates)."""
    return sorted(p.stem for p in SPEC_DIR.glob("*.yaml") if not p.stem.startswith("_"))


def load(name: str | Path) -> Sensor:
    """Load a shipped spec by id (``"iride_heo"``) or any spec file by path."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = SPEC_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no spec {name!r}; available: {available()}")
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")), origin=str(path))


def parse(raw: Mapping[str, Any], origin: str = "<spec>") -> Sensor:
    """Validate a spec mapping (as read from YAML) and build a :class:`Sensor`."""
    if not isinstance(raw, Mapping):
        raise SpecError(f"{origin}: a spec must be a mapping")
    # top-level keys starting with "_" only hold YAML anchors (&name) reused in the bands
    _no_unknown({k: v for k, v in raw.items() if not str(k).startswith("_")}, _TOP_KEYS, origin)
    for key in ("id", "name", "bands", "sources"):
        if key not in raw:
            raise SpecError(f"{origin}: missing '{key}'")
    sources = raw["sources"]
    if not isinstance(sources, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in sources.items()
    ):
        raise SpecError(f"{origin}: 'sources' must map short keys to citations")
    ctx = _Context(origin, sources)

    defaults = raw.get("defaults") or {}
    _no_unknown(defaults, _BAND_KEYS - {"name"}, f"{origin}: defaults")
    raw_bands = raw["bands"]
    if not isinstance(raw_bands, list) or not raw_bands:
        raise SpecError(f"{origin}: 'bands' must be a non-empty list")
    bands = tuple(_band(_with_defaults(defaults, b), ctx) for b in raw_bands)
    names = [b.name for b in bands]
    if len(set(names)) != len(names):
        raise SpecError(f"{origin}: duplicate band names in {names}")

    processing = raw.get("processing")
    steps = None if processing is None else tuple(_step(s, ctx, names) for s in processing)

    return Sensor(
        id=str(raw["id"]),
        name=str(raw["name"]),
        platform=str(raw.get("platform", "")),
        sources=dict(sources),
        instrument=_instrument(raw.get("instrument") or {}, ctx),
        product=_product(raw.get("product") or {}, ctx),
        processing=steps,
        bands=bands,
        note=raw.get("note"),
    )


@dataclass(frozen=True)
class _Context:
    origin: str
    sources: Mapping[str, str]

    def where(self, *parts: str) -> str:
        return ": ".join((self.origin, *parts))


def _no_unknown(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")


def _with_defaults(defaults: Mapping[str, Any], band: Any) -> dict[str, Any]:
    """Merge ``defaults`` into one band: band values win, mappings are merged key by key."""
    if not isinstance(band, Mapping):
        raise SpecError(f"each band must be a mapping, got {band!r}")
    merged = dict(band)
    for key, default in defaults.items():
        if key not in band:
            if key != "snr":  # an SNR default is a provenance template, not a point
                merged[key] = default
            continue
        given = band[key]
        if given is None or not isinstance(default, Mapping):
            continue  # explicit null, or nothing to merge: the band's own entry stands
        if isinstance(given, Mapping):
            merged[key] = {**default, **given}
        elif isinstance(given, list):
            merged[key] = [{**default, **g} if isinstance(g, Mapping) else g for g in given]
        else:  # a bare number inherits the default's provenance
            merged[key] = {**{k: v for k, v in default.items() if k != "range"}, "value": given}
    return merged


def _check_provenance(kind: Any, source: Any, where: str, ctx: _Context) -> None:
    if kind is not None and kind not in KINDS:
        raise SpecError(f"{where}: kind {kind!r} is not one of {KINDS}")
    if source is not None and source not in ctx.sources:
        raise SpecError(f"{where}: source {source!r} is not in this file's sources")


def _number(x: Any, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SpecError(f"{where}: expected a finite number, got {x!r}")
    return float(x)


def _value(raw: Any, where: str, ctx: _Context) -> Value:
    if raw is None:
        return Value()
    if not isinstance(raw, Mapping):
        return Value(value=_number(raw, where))
    _no_unknown(raw, _VALUE_KEYS, where)
    kind, source = raw.get("kind"), raw.get("source")
    _check_provenance(kind, source, where, ctx)
    low = high = None
    if raw.get("range") is not None:
        rng = raw["range"]
        if not isinstance(rng, list) or len(rng) != 2:
            raise SpecError(f"{where}: range must be [low, high] (use null for an open side)")
        low = None if rng[0] is None else _number(rng[0], where)
        high = None if rng[1] is None else _number(rng[1], where)
        if low is not None and high is not None and low >= high:
            raise SpecError(f"{where}: range low {low} must be below high {high}")
    value = raw.get("value")
    if value is not None:
        value = _number(value, where)
    elif low is not None and high is not None:
        value = (low + high) / 2
    if value is None and low is None and high is None:
        kind = source = None  # nothing known: provenance would describe nothing
    return Value(value, kind, source, low, high, raw.get("note"))


def _check_range(v: Value, lo: float, hi: float, where: str, inclusive_low: bool = False) -> None:
    for x in (v.value, v.low, v.high):
        if x is None:
            continue
        below = x < lo if inclusive_low else x <= lo
        if below or x > hi:
            raise SpecError(f"{where}: {x} is outside the plausible range ({lo}, {hi}]")


def _band(raw: Mapping[str, Any], ctx: _Context) -> Band:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise SpecError(ctx.where("bands", f"band without a name: {dict(raw)!r}"))
    where = ctx.where(f"band {name}")
    _no_unknown(raw, _BAND_KEYS, where)
    v = {key: _value(raw.get(key), f"{where}: {key}", ctx) for key in _BAND_VALUES}

    spectral = v["spectral"]
    if spectral.low is None or spectral.high is None:
        raise SpecError(f"{where}: spectral needs both edges, e.g. {{range: [649.1, 680.1]}}")
    _check_range(spectral, 300.0, 15000.0, f"{where}: spectral (nm)")
    for key in ("gsd_m", "pixel_m", "saturation_radiance", "e0"):
        _check_range(v[key], 0.0, 1e6, f"{where}: {key}")
    for key in ("mtf_along", "mtf_across"):
        _check_range(v[key], 0.0, 1.0, f"{where}: {key}")
    _check_range(
        v["registration_sigma_px"],
        0.0,
        100.0,
        f"{where}: registration_sigma_px",
        inclusive_low=True,
    )
    _check_range(v["time_offset_s"], 0.0, 600.0, f"{where}: time_offset_s", inclusive_low=True)

    return Band(
        name=name,
        common_name=str(raw.get("common_name", name)),
        snr=_snr(raw.get("snr"), f"{where}: snr", ctx),
        srf=raw.get("srf"),
        note=raw.get("note"),
        **v,
    )


def _snr(raw: Any, where: str, ctx: _Context) -> tuple[SNRPoint, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SpecError(f"{where}: expected a list of {{snr, radiance}} points")
    points = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise SpecError(f"{where}: each point is a mapping, got {item!r}")
        _no_unknown(item, {"snr", "radiance", "kind", "source", "note"}, where)
        _check_provenance(item.get("kind"), item.get("source"), where, ctx)
        if item.get("snr") is None or item.get("radiance") is None:
            raise SpecError(f"{where}: an SNR without its radiance cannot be used: {dict(item)}")
        snr, radiance = _number(item["snr"], where), _number(item["radiance"], where)
        if snr <= 0 or radiance <= 0:
            raise SpecError(f"{where}: snr and radiance must be positive")
        points.append(SNRPoint(snr, radiance, item.get("kind"), item.get("source")))
    if len({p.radiance for p in points}) != len(points):
        raise SpecError(f"{where}: two SNR points at the same radiance")
    return tuple(points)


def _instrument(raw: Mapping[str, Any], ctx: _Context) -> dict[str, Any]:
    where = ctx.where("instrument")
    _no_unknown(raw, _INSTRUMENT_TEXT | _INSTRUMENT_VALUES, where)
    out: dict[str, Any] = {k: raw[k] for k in _INSTRUMENT_TEXT if k in raw}
    if out.get("imaging", "unknown") not in IMAGING:
        raise SpecError(f"{where}: imaging must be one of {IMAGING}")
    for key in _INSTRUMENT_VALUES:
        out[key] = _value(raw.get(key), f"{where}: {key}", ctx)
        _check_range(out[key], 0.0, 1e6, f"{where}: {key}")
    return out


def _product(raw: Mapping[str, Any], ctx: _Context) -> dict[str, Any]:
    where = ctx.where("product")
    _no_unknown(raw, _PRODUCT_KEYS, where)
    if raw.get("units") is not None and raw["units"] not in UNITS:
        raise SpecError(f"{where}: units must be one of {UNITS}")
    for key in ("scale", "offset"):
        x = raw.get(key)
        if x is not None and x != "per_scene":
            _number(x, f"{where}: {key}")
    return dict(raw)


def _step(raw: Any, ctx: _Context, band_names: list[str]) -> ProcessingStep:
    if not isinstance(raw, Mapping) or "op" not in raw:
        raise SpecError(ctx.where("processing", f"each step needs an 'op': {raw!r}"))
    op = raw["op"]
    where = ctx.where("processing", str(op))
    if op not in PROCESSING_OPS:
        raise SpecError(f"{where}: unknown op; known ops are {PROCESSING_OPS}")
    _check_provenance(raw.get("kind"), raw.get("source"), where, ctx)
    bands = raw.get("bands")
    if bands is not None:
        missing = [b for b in bands if b not in band_names]
        if missing:
            raise SpecError(f"{where}: refers to unknown bands {missing}")
        bands = tuple(bands)
    params = {k: v for k, v in raw.items() if k not in _STEP_KEYS}
    if op == "resample" and params.get("kernel") not in RESAMPLE_KERNELS:
        raise SpecError(f"{where}: kernel must be one of {RESAMPLE_KERNELS}")
    if op == "bin":
        factor = params.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
            raise SpecError(f"{where}: factor must be an integer >= 2")
    if op == "sharpen":
        for key in ("radius_px", "amount"):
            if key in params:
                _number(params[key], f"{where}: {key}")
    return ProcessingStep(
        op, params, bands, raw.get("stage"), raw.get("kind"), raw.get("source"), raw.get("note")
    )


# ------------------------------------------------------------------------- spectral helper


def edges_from_response(wavelength_nm: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    """FWHM edges of a spectral response: outermost wavelengths at half the peak.

    Used to turn a measured response curve into the two numbers a spec stores. The crossing
    is interpolated linearly between samples.
    """
    x = np.asarray(wavelength_nm, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or np.any(np.diff(x) <= 0):
        raise ValueError("wavelengths must be 1-D and increasing, with one response per value")
    half = y.max() / 2.0
    above = np.flatnonzero(y >= half)
    i0, i1 = above[0], above[-1]
    if i0 == 0 or i1 == len(y) - 1:
        raise ValueError("the response does not fall below half its peak on both sides")
    lo = np.interp(half, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])
    hi = np.interp(half, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])
    return float(lo), float(hi)
