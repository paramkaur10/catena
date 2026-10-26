"""A2 tests: the spec files load, refuse bad input, and reproduce the numbers they cite."""

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from catena.sensors import physics
from catena.sensors import spec as S

REPO = Path(__file__).resolve().parents[1]

MINIMAL = """
id: toy
name: Toy sensor
sources: {sheet: "a datasheet"}
bands:
  - {name: B1, spectral: {range: [450, 520], kind: nominal, source: sheet}}
"""


def toy(**changes):
    """The minimal spec above with some top-level keys replaced."""
    raw = yaml.safe_load(MINIMAL)
    raw.update(changes)
    return raw


# ------------------------------------------------------------------- shipped specs load


@pytest.mark.parametrize("sensor_id", S.available())
def test_every_shipped_spec_loads(sensor_id):
    sensor = S.load(sensor_id)
    assert sensor.id == sensor_id  # the file name and the id must agree
    assert sensor.bands
    for band in sensor.bands:
        assert band.lower_nm < band.centre_nm < band.upper_nm


def test_the_template_is_a_valid_spec():
    assert S.load(S.SPEC_DIR / "_template.yaml").id == "new_sensor"


def test_shipped_sensors():
    assert set(S.available()) == {
        "sentinel2_msi",
        "landsat8_oli",
        "perusat1",
        "iride_heo",
        "iride_eaglet2",
        "phisat2",
    }


# ------------------------------------------------------------------------ provenance


def test_bare_numbers_and_provenance():
    raw = toy(
        bands=[
            {
                "name": "B1",
                "spectral": {"range": [450, 520]},
                "gsd_m": 10,
                "e0": {"value": 1900, "kind": "measured", "source": "sheet"},
            }
        ]
    )
    band = S.parse(raw).band("B1")
    assert band.gsd_m.value == 10 and band.gsd_m.kind is None
    assert band.e0.kind == "measured" and band.e0.source == "sheet"


def test_every_source_must_be_cited_in_the_file():
    raw = toy(bands=[{"name": "B1", "spectral": {"range": [450, 520], "source": "nowhere"}}])
    with pytest.raises(S.SpecError, match="nowhere"):
        S.parse(raw)


def test_unknown_kind_is_rejected():
    raw = toy(bands=[{"name": "B1", "spectral": {"range": [450, 520], "kind": "guessed"}}])
    with pytest.raises(S.SpecError, match="guessed"):
        S.parse(raw)


def test_two_sided_range_gives_its_midpoint_one_sided_gives_only_a_bound():
    s2 = S.load("sentinel2_msi")
    b04, b05 = s2.band("B04"), s2.band("B05")
    assert b04.mtf_along.value == pytest.approx(0.225)  # requirement window 0.15-0.30
    assert b04.mtf_along.kind == "requirement"
    assert b05.mtf_along.value is None and b05.mtf_along.high == 0.45  # "below 0.45"


def test_missing_values_carry_a_note_but_no_provenance():
    heo = S.load("iride_heo")
    saturation = heo.band("B03").saturation_radiance
    assert not saturation.known and saturation.kind is None and "Argotec" in saturation.note


# -------------------------------------------------------------------------- defaults


def test_defaults_merge_into_bands_and_bands_win():
    l8 = S.load("landsat8_oli")
    assert l8.band("B4").gsd_m.value == 30 and l8.band("B4").gsd_m.source == "usgs_l8"
    pan = l8.band("B8")
    assert pan.gsd_m.value == 15 and pan.gsd_m.kind == "nominal"  # bare 15 keeps provenance
    assert pan.snr[0].kind == "measured" and pan.snr[0].source == "morfitt2015"


def test_a_band_can_override_a_default_with_null():
    raw = toy(
        defaults={"gsd_m": 10},
        bands=[{"name": "B1", "spectral": {"range": [450, 520]}, "gsd_m": None}],
    )
    assert not S.parse(raw).band("B1").gsd_m.known


# ------------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "band, message",
    [
        ({"name": "B1", "spectral": {"range": [520, 450]}}, "below high"),
        ({"name": "B1", "spectral": {"range": [450, None]}}, "both edges"),
        ({"name": "B1", "spectral": {"range": [100, 200]}}, "plausible range"),
        ({"name": "B1", "spectral": {"range": [450, 520]}, "mtf_along": 1.3}, "plausible"),
        ({"name": "B1", "spectral": {"range": [450, 520]}, "gsd_m": 0}, "plausible"),
        (
            {"name": "B1", "spectral": {"range": [450, 520]}, "snr": [{"snr": 100}]},
            "without its radiance",
        ),
        ({"name": "B1", "spectral": {"range": [450, 520]}, "colour": "blue"}, "unknown keys"),
    ],
)
def test_bad_bands_are_rejected(band, message):
    with pytest.raises(S.SpecError, match=message):
        S.parse(toy(bands=[band]))


def test_duplicate_band_names_are_rejected():
    band = {"name": "B1", "spectral": {"range": [450, 520]}}
    with pytest.raises(S.SpecError, match="duplicate"):
        S.parse(toy(bands=[band, band]))


@pytest.mark.parametrize(
    "step, message",
    [
        ({"op": "blur"}, "unknown op"),
        ({"op": "resample", "kernel": "magic"}, "kernel"),
        ({"op": "bin", "factor": 1}, "factor"),
        ({"op": "bin", "factor": 2, "bands": ["B9"]}, "unknown bands"),
    ],
)
def test_bad_processing_steps_are_rejected(step, message):
    with pytest.raises(S.SpecError, match=message):
        S.parse(toy(processing=[step]))


def test_unknown_top_level_keys_are_rejected_but_anchor_holders_are_not():
    with pytest.raises(S.SpecError, match="unknown keys"):
        S.parse(toy(colour="blue"))
    S.parse(toy(_anchors={"anything": 1}))


# -------------------------------------------------------------- derived physical numbers


def test_centre_and_width_come_from_the_edges():
    b04 = S.load("sentinel2_msi").band("B04")  # S2A: centre 664.6 nm, width 31 nm
    assert b04.centre_nm == pytest.approx(664.6)
    assert b04.fwhm_nm == pytest.approx(31.0)


def test_psf_width_follows_the_mtf_through_a1():
    b02 = S.load("sentinel2_msi").band("B02")
    along, across = b02.psf_sigma_px()
    assert along == pytest.approx(physics.sigma_from_mtf(0.225))
    assert across == along
    assert S.load("iride_heo").band("B01").psf_sigma_px() == (None, None)


def test_noise_model_reproduces_the_cited_snr():
    for sensor_id, band, snr, radiance in [
        ("sentinel2_msi", "B04", 142, 108),
        ("landsat8_oli", "B4", 227, 22),
    ]:
        a, b = S.load(sensor_id).band(band).noise_model()
        assert physics.snr_at(radiance, a, b) == pytest.approx(snr)


def test_two_snr_points_fix_shot_and_read_noise():
    raw = toy(
        bands=[
            {
                "name": "B1",
                "spectral": {"range": [450, 520]},
                "snr": [{"snr": 300, "radiance": 200}, {"snr": 80, "radiance": 20}],
            }
        ]
    )
    a, b = S.parse(raw).band("B1").noise_model()
    assert physics.snr_at(20, a, b) == pytest.approx(80)
    assert physics.snr_at(200, a, b) == pytest.approx(300)


# ------------------------------------------------------------------ facts the specs cite


def test_heo_edges_match_argotec_curves():
    """The YAML edges must be what the payload sheet says, to the 0.1 nm they are rounded to."""
    path = REPO / "data" / "srf" / "iride_heo_argotec.csv"
    if not path.exists():
        pytest.skip("Argotec curves are private and not in this checkout")
    rows = [r for r in csv.reader(path.open()) if r and not r[0].startswith("#")]
    header, data = rows[0], np.array(rows[1:], dtype=float)
    wavelength = data[:, header.index("wavelength_nm")]
    for band in S.load("iride_heo").bands:
        column = band.srf.split(":")[1]
        lo, hi = S.edges_from_response(wavelength, data[:, header.index(column)])
        assert lo == pytest.approx(band.lower_nm, abs=0.051)
        assert hi == pytest.approx(band.upper_nm, abs=0.051)


def test_heo_band_timing_gives_the_stripe_order():
    heo = S.load("iride_heo")
    order = [b.common_name for b in sorted(heo.bands, key=lambda b: b.time_offset_s.value)]
    assert order == ["rededge3", "rededge2", "rededge1", "nir", "pan", "red", "green", "blue"]


def test_heo_red_edges_are_resampled_on_ground_not_binned():
    """Argotec, 22 Oct 2026: the 5 m red edges come from ground resampling, with no binning,
    sharpening or denoising anywhere in the chain."""
    heo = S.load("iride_heo")
    ops = {s.op for s in heo.processing}
    assert "bin" not in ops and "sharpen" not in ops and "denoise" not in ops
    coarse = {b.name for b in heo.bands if b.pixel_m.value == 5.0}
    assert coarse == {"B04", "B05", "B06"}
    assert all(any(s.op == "resample" and s.bands for s in heo.steps_for(n)) for n in coarse)
    assert heo.product["max_reflectance"] == 1.0
    assert heo.band("B03").mtf_along.low == 0.10  # "> 10 %, measured in flight"


def test_eaglet_is_a_bayer_sensor_and_is_sharpened():
    eaglet = S.load("iride_eaglet2")
    red = eaglet.band("Band 1")
    assert red.upper_nm > 800  # vegetation is bright here
    ops = [s.op for s in eaglet.processing]
    assert ops.index("demosaic") < ops.index("sharpen") < ops.index("resample")
    sharpen = eaglet.processing[ops.index("sharpen")]
    assert sharpen.params["amount"] == 2.5
    assert eaglet.instrument["tdi_stages_max"].value == 15


def test_band_lookup_names_the_bands_it_knows():
    with pytest.raises(KeyError, match="B04"):
        S.load("sentinel2_msi").band("red")


# ---------------------------------------------------------------------- completeness


def test_completeness_reports_the_weakest_status():
    assert S.load("landsat8_oli").completeness()["snr"] == "measured"
    assert S.load("sentinel2_msi").completeness()["snr"] == "requirement"
    assert S.load("sentinel2_msi").completeness()["mtf"] == "bound"  # 20 m bands: only "< 0.45"
    assert S.load("iride_heo").completeness()["mtf"] == "bound"  # Argotec: "> 10 %"
    assert S.load("iride_heo").completeness()["snr"] == "missing"
    raw = toy(
        bands=[
            {"name": "B1", "spectral": {"range": [450, 520]}, "gsd_m": 10},
            {"name": "B2", "spectral": {"range": [520, 600]}},
        ]
    )
    assert S.parse(raw).completeness()["gsd"] == "partial"


# ----------------------------------------------------------------------- spectral helper


def test_edges_from_a_gaussian_response():
    x = np.arange(400.0, 701.0)
    sigma = 20.0
    y = np.exp(-0.5 * ((x - 550.0) / sigma) ** 2)
    lo, hi = S.edges_from_response(x, y)
    half_width = sigma * np.sqrt(2 * np.log(2))
    assert lo == pytest.approx(550 - half_width, abs=0.05)
    assert hi == pytest.approx(550 + half_width, abs=0.05)


def test_edges_need_a_band_that_falls_off_on_both_sides():
    x = np.arange(400.0, 500.0)
    with pytest.raises(ValueError):
        S.edges_from_response(x, np.ones_like(x))
