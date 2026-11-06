"""The tests of the big run: every arm of every seed (Clarity real and sim, and the arms that
test the method) on six test sets, beside OmniCloudMask wherever its predictions exist.

    python -m catena.evaluate_clarity RUNS [RUNS ...] [--data /root/catena_data]
        [--tests A B C D E F] [--device cuda:0] [--windows 800] [--ocm] [--sensei] [--maps DIR]
        [--out CSV] [--cards honest shuffled "pixel x3" ...]

Each RUNS folder holds finished runs (final.pt) at any depth, one folder per seed, as
scripts/big.sh makes them (runs/big/seed0, or runs/arms/chain/seed0 for OUT=runs/arms/chain);
each run brings its two arms (catena.train_clarity), named <seed folder>/<arm>; a nas arm's
trunk is rebuilt from its architecture code (catena.nas, PyNAS). Two runs that bring the same
name stop the command: evaluate them apart. The tests, by what the networks met in training:
  A  Sentinel-2, seen: the CloudSEN12+ test split through the 23 cameras of catena.evaluate
     (Sentinel-2 as delivered, 18 four-band cameras at 10 and 30 m, 4 beyond the training
     range), --windows windows of --a-size (240) 10 m pixels per camera
  B  Landsat 8, seen: the 24 scenes of L8 Biome's pilot.csv, kept out of training
  C  Landsat 7, never seen: the scenes of L7 Irish's pilot.csv
  D  IRIDE HEO, seen: its 4 test scenes, and its 3 stress scenes (processing version A02),
     none of them seen
  E  IRIDE Eaglet II (22 scenes) and Phi-sat-2 (7 scenes), never seen
  F  PeruSat-1, seen: CloudPeru2's validation split, cloud or not cloud only
The data under DATA: chips/cloudsen12plus (A), chips/l8_biome (B), chips/l7_irish (C),
chips/iride_heo and labels/iride_heo (D), labels/iride_eaglet2 with raw/iride_eaglet2 and
labels/phisat2 with raw/phisat2 (E), chips/cloudperu2 (F). A set whose data is missing is
left out, with a note.

Every scene of B-F is predicted whole, in tiles of the training windows (80 px at 30 m, 240 px
otherwise) that overlap by twice a margin (10 px at 30 m, 30 px otherwise); each pixel takes
the tile it lies deepest in. A pixel is scored where it has a label and every band has data
(CloudPeru2, surface reflectance that can be 0: any band). Each network reads what its arm
reads of each band's chain card (catena.descriptor.chain_card): the card (its band edges,
surface or top of atmosphere, its pixel against the grid's and the grid's pixel on the
ground), then what the MTF and the SNR say where they are known (Sentinel-2 and the cameras
of Test A, Landsat 8's noise); the blind arm reads zeros. The fixed arm reads blue, green, red
and near infrared in that order: Landsat 8's B2-B5, Landsat 7's B1-B4, HEO's B01-B03 and B07,
Phi-sat-2's blue, green, red and nir, PeruSat-1's four; Eaglet II has no near infrared band,
so its blue stands in, as in OmniCloudMask's draft the labels start from (ocm4_nirblue).

OmniCloudMask (D, E): its drafts (scripts/ocm_prelabel.py) are its predictions, scored on the
same pixels; for Eaglet II both drafts, near infrared filled with blue (ocm4_nirblue, which
the labels start from) and with zeros (ocm4_nirzeros). The labels were made by correcting
these drafts, so this comparison leans towards OmniCloudMask: wherever the review kept a draft
as it was, OmniCloudMask scores perfectly by construction. A Clarity win is the conservative
result. With --ocm, OmniCloudMask also runs on Landsat 8 and Landsat 7 (B, C), as it ran for
the drafts (model 4.0 on red, green and near infrared): labels that do not come from it, on a
satellite (Landsat 7) neither model trained on, so an even comparison.

SEnSeIv2 (--sensei; Francis et al., IEEE TGRS 2024; senseiv2, installed from its repository):
its sensor-independent model (SEnSeIv2-SegFormerB2-alldata-ambiguous, from Hugging Face) on
every scene of B-F, as its own CloudMask runs it (patches of its size, overlapping by half),
told each band's edges as our cards state them and its type, top-of-atmosphere reflectance
(the only optical type it knows: PeruSat-1's surface reflectance is told so too). Its four
classes (clear, thick, thin, shadow) are put in Catena's codes. Not on Test A's windows.

Scores pool the pixels of a set: cloud (thin or thick) against not cloud, as F1 and balanced
accuracy, and the F1 of thin and of thick cloud, over every scene; the F1 of clear and of
shadow, the mean F1 of the four classes and the accuracy over the scenes whose shadows were
labelled (L8 Biome and L7 Irish: as pilot.csv says; our labels: a scene with a shadow, or
without cloud), since elsewhere clear may hide shadow. A class neither labelled nor predicted
has no F1, and the mean is over the classes that have one. CloudPeru2 gives cloud against not
cloud only. Test A scores each camera alone and averages them in groups.

Writes --out (default RUNS/clarity_tests.csv), again after every set: one row per test, set,
scene (and per set the pooled scores, scene "all"), model and size, with the confusion matrix
(6 x 4: rows the label codes 0 clear, 1 thin, 2 thick, 3 shadow, 4 cloud, 5 not cloud; columns
the predicted class clear, thin, thick, shadow) and the scores. Prints per set the mean and the
standard deviation over the seeds of each version, and OmniCloudMask. A set that fails is
reported and skipped. Without --out, the CSV goes to the last RUNS folder: clarity_tests.csv
for all six tests and clarity_tests_<tests>[_ocm].csv otherwise, so a partial run never
overwrites the full one.
--maps DIR keeps every scene's predictions, label, OmniCloudMask's classes and true colour,
one .npz per scene, for pictures (catena.maps, notebooks/results_<dataset>.ipynb).

Card checks (--cards): every network also sees each scene (and each Test A window) as told
otherwise, no training needed; a network that reads its card must do worse when the card is
wrong, and should hold up when bands go missing. Its rows are named <model>@<check>:
  honest            every card as the product states it (the plain test)
  shuffled          each band given the next band's card: the wavelengths all wrong
  pixel x3          the pixels told 3 times larger than they are (the grid's pixel size)
  pixel /3          the pixels told 3 times smaller than they are
  wavelengths only  the card's other numbers 0: top of atmosphere, a 10 m grid, no chain
  rgbn only         only blue, green, red and near infrared kept (the bands themselves)
  rgb only          only blue, green and red kept; the fixed arm has nothing to read
  noise x3          the noise told 3 times larger where it is known (the chain arm reads it)
  blur x2           the blur told twice as wide where it is known
Without --out, a run with checks other than honest writes clarity_tests_<tests>[_ocm]_cards.csv
(and with SEnSeIv2, _sensei before .csv); the maps (--maps) keep the honest predictions only.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio

from catena import descriptor as desc
from catena import viz
from catena.data import biome, eaglet2, heo, irish, perusat, phisat2
from catena.sensors import spec

VERSIONS = ("real", "sim")  # the first big run's arms
ARMS = ("real", "sim", "sensei", "chain", "blind", "fixed", "nas", "nas2")  # the tables' order
ARM_TITLES = {
    "real": "Clarity real",
    "sim": "Clarity sim",
    "sensei": "Clarity no pixel size",
    "chain": "Clarity + MTF, SNR",
    "blind": "Clarity told nothing",
    "fixed": "U-Net fixed bands",
    "nas": "PyNAS trunk 1",
    "nas2": "PyNAS trunk 2",
}
TESTS = ("A", "B", "C", "D", "E", "F")
NO_LABEL = 255
NAMES = ("clear", "thin", "thick", "shadow")
CLOUD_CODES, CLEAR_CODES = [1, 2, 4], [0, 3, 5]  # label codes: cloud, not cloud
CLOUD_CLASSES, CLEAR_CLASSES = [1, 2], [0, 3]  # predicted classes
TILE, MARGIN = 240, 30  # the training window, and how deep a pixel must lie to count
LANDSAT_TILE, LANDSAT_MARGIN = 80, 10  # at 30 m
PIXELS_PER_BATCH = 2_000_000
EVERY = ("cloud_f1", "cloud_ba", "f1_thin", "f1_thick")  # over every scene
SHADOWED = ("f1_clear", "f1_shadow", "f1_mean", "accuracy")  # over scenes with shadow labels
SHOWN = ("cloud_f1", "cloud_ba", "f1_thin", "f1_thick", "f1_shadow", "f1_mean", "accuracy")
HEADS = ("cloud F1", "cloud BA", "F1 thin", "F1 thick", "F1 shadow", "F1 mean", "accuracy")
OCM_VERSION = 4.0  # OmniCloudMask's model, as for the drafts
GAMMA = 0.5  # the true colour's gamma, as the review's pictures (catena.drafts)
OCM = {
    "ocm4": "OmniCloudMask",
    "ocm4_nirblue": "OmniCloudMask, NIR from blue",
    "ocm4_nirzeros": "OmniCloudMask, NIR zeros",
}
BASELINES = OCM | {"sensei2": "SEnSeIv2"}  # other models, by the name of their predictions
SENSEI_MODEL = "SEnSeIv2-SegFormerB2-alldata-ambiguous"
SENSEI_CODES = np.array([0, 2, 1, 3], np.uint8)  # its clear, thick, thin, shadow in our codes


def same(x: np.ndarray) -> np.ndarray:
    return x


@dataclass
class Item:
    """One scene to score: its image (bands, H, W) in any units, to_rho turning tiles of it
    (n, bands, h, w) into reflectance, its chain cards (bands, 11), its label in Catena's codes,
    other predictions of it (OmniCloudMask's drafts, in Catena's codes, by name), whether its
    shadows were labelled, the tiles to predict it in, and where blue, green, red and near
    infrared sit among its bands (rgbn: for the fixed arm; None, no fixed-band prediction)."""

    name: str
    image: np.ndarray
    told: np.ndarray
    label: np.ndarray
    to_rho: Callable[[np.ndarray], np.ndarray] = same
    others: dict = field(default_factory=dict)
    shadow_ok: bool = True
    tile: int = TILE
    margin: int = MARGIN
    rgbn: tuple | None = None


@dataclass
class SceneSet:
    """A set of scenes, read one at a time by items(); rgb, where red, green and blue sit
    among an item's bands; note, what OmniCloudMask's classes are here."""

    test: str  # A-F
    name: str
    title: str
    scenes: int
    items: Callable[[], Iterator[Item]]
    rgb: tuple[int, int, int] = (0, 1, 2)
    note: str = ""


DRAFTED = "OmniCloudMask: its drafts, which the labels were corrected from"
RAN = "OmniCloudMask: run here, model 4.0 on red, green and NIR; the labels do not come from it"


# --- predicting a scene in tiles ------------------------------------------------------------


def starts(n: int, tile: int, margin: int) -> list[int]:
    """Where the tiles start along an axis of n pixels (n >= tile): every tile - 2 margin
    pixels, and a last one flush with the end."""
    if n <= tile:
        return [0]
    return [*range(0, n - tile, tile - 2 * margin), n - tile]


def owned(first: list[int], tile: int, n: int) -> list[tuple[int, int]]:
    """The stretch of the axis each tile owns: from the middle of its overlap with the one
    before to the middle of its overlap with the one after."""
    cuts = [0, *((a + tile + b) // 2 for a, b in pairwise(first)), n]
    return list(pairwise(cuts))


def predict(classify, item: Item) -> dict[str, np.ndarray]:
    """Every model's classes (H, W) uint8 over the scene: classify(x, told, rgbn), x the
    reflectance of a batch of tiles (n, bands, tile, tile), gives {model: classes (n, tile,
    tile)}. Tiles that own no labelled pixel are not predicted (255 there)."""
    image, size = item.image, item.tile
    _, h, w = image.shape
    height, width = max(h, size), max(w, size)
    if (height, width) != (h, w):  # smaller than a tile: padded with no data
        image = np.pad(image, ((0, 0), (0, height - h), (0, width - w)))
    labelled = np.zeros((height, width), bool)
    labelled[:h, :w] = item.label != NO_LABEL
    rows, cols = starts(height, size, item.margin), starts(width, size, item.margin)
    jobs = [
        (r, c, (r0, r1), (c0, c1))
        for r, (r0, r1) in zip(rows, owned(rows, size, height), strict=True)
        for c, (c0, c1) in zip(cols, owned(cols, size, width), strict=True)
        if labelled[r0:r1, c0:c1].any()
    ]
    out: dict[str, np.ndarray] = {}
    per_batch = max(1, PIXELS_PER_BATCH // (size * size))
    for k in range(0, len(jobs), per_batch):
        part = jobs[k : k + per_batch]
        x = np.stack([image[:, r : r + size, c : c + size] for r, c, _, _ in part])
        found = classify(np.ascontiguousarray(item.to_rho(x), np.float32), item.told, item.rgbn)
        for model, classes in found.items():
            pred = out.setdefault(model, np.full((height, width), NO_LABEL, np.uint8))
            for (r, c, (r0, r1), (c0, c1)), p in zip(part, classes, strict=True):
                pred[r0:r1, c0:c1] = p[r0 - r : r1 - r, c0 - c : c1 - c]
    return {model: pred[:h, :w] for model, pred in out.items()}


# --- card checks ----------------------------------------------------------------------------

CARDS = (
    "honest",
    "shuffled",
    "pixel x3",
    "pixel /3",
    "wavelengths only",
    "rgbn only",
    "rgb only",
    "noise x3",
    "blur x2",
)
GSD_AT, PSF_AT, NOISE_AT, KNOWN_PSF_AT, KNOWN_NOISE_AT = 5, 6, (7, 8), 9, 10  # in a chain card


def _copied(a):
    return a.copy() if isinstance(a, np.ndarray) else a.clone()


def vary(check: str, x, told, rgbn):
    """A batch as a card check tells it (CARDS): x (n, bands, h, w), told (..., bands, 11),
    rgbn where blue, green, red and near infrared sit among the bands (or None). Numpy arrays
    or torch tensors; the result (x, told, rgbn), or None where the check cannot apply (a
    band check without rgbn)."""
    if check not in CARDS:
        raise ValueError(f"unknown card check: {check}")
    if check == "honest":
        return x, told, rgbn
    if check in ("rgbn only", "rgb only"):
        if rgbn is None:
            return None
        keep = list(dict.fromkeys(rgbn if check == "rgbn only" else rgbn[:3]))
        four = tuple(keep.index(b) for b in rgbn) if check == "rgbn only" else None
        return x[:, keep], told[..., keep, :], four
    told = _copied(told)
    if check == "shuffled":
        told = np.roll(told, 1, axis=-2) if isinstance(told, np.ndarray) else told.roll(1, -2)
    elif check in ("pixel x3", "pixel /3"):
        told[..., GSD_AT] += math.log2(3) if check == "pixel x3" else -math.log2(3)
    elif check == "wavelengths only":
        told[..., 3:] = 0
    elif check == "noise x3":
        told[..., NOISE_AT[0] : NOISE_AT[1] + 1] += (
            math.log10(3) * told[..., KNOWN_NOISE_AT : KNOWN_NOISE_AT + 1]
        )
    else:  # blur x2: an unknown blur (0) stays 0
        told[..., PSF_AT] *= 2
    return x, told, rgbn


def checked(classify, check: str):
    """classify(x, told, rgbn) as a card check tells it; nothing where it cannot apply."""

    def run(x, told, rgbn=None):
        varied = vary(check, x, told, rgbn)
        return {} if varied is None else classify(*varied)

    return run


def named(model: str, check: str) -> str:
    """seed0/sim under a check: seed0/sim@shuffled; honest keeps the plain name."""
    return model if check == "honest" else f"{model}@{check}"


# --- scores ---------------------------------------------------------------------------------


def confusion(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """6 x 4: rows the label codes 0-5, columns the predicted classes 0-3."""
    ok = (label < 6) & (pred < 4)
    pairs = label[ok].astype(np.int64) * 4 + pred[ok].astype(np.int64)
    return np.bincount(pairs, minlength=24).reshape(6, 4)


def scores(conf: np.ndarray) -> dict:
    """From a 6 x 4 confusion matrix: the F1 of each class and their mean, the accuracy (the
    label codes 0-3), and cloud against not cloud (codes 1, 2, 4 against 0, 3, 5): F1,
    precision, recall and balanced accuracy. A score without pixels to say it is nan."""
    conf = np.asarray(conf, np.float64)
    four = conf[:4]
    tp = np.diag(four)
    both = four.sum(0) + four.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(both > 0, 2 * tp / both, np.nan)
    out = {f"f1_{n}": float(v) for n, v in zip(NAMES, f1, strict=True)}
    out["f1_mean"] = float(np.nanmean(f1)) if np.isfinite(f1).any() else math.nan
    out["accuracy"] = float(tp.sum() / four.sum()) if four.sum() else math.nan
    hit = conf[np.ix_(CLOUD_CODES, CLOUD_CLASSES)].sum()
    miss = conf[np.ix_(CLOUD_CODES, CLEAR_CLASSES)].sum()
    false = conf[np.ix_(CLEAR_CODES, CLOUD_CLASSES)].sum()
    calm = conf[np.ix_(CLEAR_CODES, CLEAR_CLASSES)].sum()
    out["cloud_f1"] = float(2 * hit / (2 * hit + false + miss)) if hit + false + miss else math.nan
    out["cloud_precision"] = float(hit / (hit + false)) if hit + false else math.nan
    out["cloud_recall"] = float(hit / (hit + miss)) if hit + miss else math.nan
    rates = [hit / (hit + miss) if hit + miss else math.nan]
    rates.append(calm / (calm + false) if calm + false else math.nan)
    out["cloud_ba"] = float(np.mean(rates)) if all(np.isfinite(rates)) else math.nan
    return out


def pooled(every: np.ndarray, shadowed: np.ndarray) -> dict:
    """A set's scores: the cloud scores and thin and thick over every scene (every: their
    confusion matrices summed), the rest over the scenes with shadow labels (shadowed)."""
    a, s = scores(every), scores(shadowed)
    return {k: a[k] for k in (*EVERY, "cloud_precision", "cloud_recall")} | {
        k: s[k] for k in SHADOWED
    }


def shadow_labelled(label: np.ndarray) -> bool:
    """Our labels mark shadow wherever it was seen: a scene with a shadow, or without cloud,
    has its shadows labelled."""
    return bool((label == 3).any() or not np.isin(label, (1, 2)).any())


# --- the test sets --------------------------------------------------------------------------


def codes(path: Path) -> np.ndarray:
    """One of OmniCloudMask's drafts in Catena's codes."""
    with rasterio.open(path) as src:
        return viz.FROM_OCM[src.read(1)]


def found_drafts(folder: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The drafts of a scene that exist, by name, in Catena's codes."""
    return {n: codes(folder / f"{n}.tif") for n in names if (folder / f"{n}.tif").exists()}


def with_drafts(label: np.ndarray, drafts: dict) -> np.ndarray:
    """The label without the pixels a draft has no prediction for, so every model is scored
    on the same pixels."""
    label = label.copy()
    for draft in drafts.values():
        label[draft == NO_LABEL] = NO_LABEL
    return label


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def omnicloudmask(red: np.ndarray, green: np.ndarray, nir: np.ndarray, device) -> np.ndarray:
    """OmniCloudMask's classes (H, W) in Catena's codes, run as for the drafts
    (scripts/ocm_prelabel.py): model 4.0 on red, green and near infrared, 0 = no data, and
    no prediction where red and green both have none."""
    from omnicloudmask import predict_from_array  # loads PyTorch

    stack = np.stack([red, green, nir]).astype(np.float32)
    settings = {"no_data_value": 0, "inference_device": str(device), "batch_size": 4}
    classes = predict_from_array(stack, model_version=OCM_VERSION, **settings)[0]
    classes = classes.astype(np.uint8)
    classes[(red <= 0) & (green <= 0)] = NO_LABEL
    return viz.FROM_OCM[classes]


def sensei_descriptors(told: np.ndarray) -> list[dict]:
    """SEnSeIv2's band descriptors from chain cards (bands, 11): each band's edges in nm
    (wavelength codes are log(nm - 300) - 2) and its type, top-of-atmosphere reflectance."""
    nm = np.exp(np.asarray(told, np.float64)[:, [0, 2]] + 2.0) + 300.0
    kind = "TOA Reflectance"
    return [
        {
            "band_type": kind,
            "min_wavelength": round(float(lo), 1),
            "max_wavelength": round(float(hi), 1),
        }
        for lo, hi in nm
    ]


def sensei(device, batch_size: int = 4):
    """SEnSeIv2's sensor-independent model as its CloudMask runs it: run(rho (bands, H, W),
    told (bands, 11)) -> classes (H, W) in Catena's codes. Downloads the model the first
    time (Hugging Face, aliFrancis/SEnSeIv2)."""
    from senseiv2.inference import CloudMask
    from senseiv2.utils import get_model_files

    config, weights = get_model_files(SENSEI_MODEL)
    settings = {"output_style": "4-class", "categorise": True, "batch_size": batch_size}
    mask = CloudMask(config, weights, device=str(device), **settings)
    stride = int(mask.model_config["PATCH_SIZE"]) // 2

    def run(rho: np.ndarray, told: np.ndarray) -> np.ndarray:
        rho = np.ascontiguousarray(rho, np.float32)
        classes = mask(rho, descriptors=sensei_descriptors(told), stride=stride)
        return SENSEI_CODES[np.asarray(classes).astype(np.int64)]

    return run


def whole_scene(item: Item) -> np.ndarray:
    """An item's reflectance, all of it (bands, H, W) float32."""
    return item.to_rho(np.ascontiguousarray(item.image[None]))[0].astype(np.float32)


def some_bands(module, scene, dn: np.ndarray, bands, rows: int = 1024) -> np.ndarray:
    """The reflectance of a few bands of a Landsat scene (len(bands), H, W), a strip of rows
    at a time, so the whole scene is never in float32 at once."""
    out = np.zeros((len(bands), *dn.shape[1:]), np.float32)
    for r0 in range(0, dn.shape[1], rows):
        out[:, r0 : r0 + rows] = module.reflectance(scene, dn[:, r0 : r0 + rows])[list(bands)]
    return out


def landsat_set(test: str, module, root: Path, title: str, ocm=None) -> SceneSet:
    """A Landsat test set; with ocm(red, green, nir) -> classes, OmniCloudMask beside."""
    chosen = module.scenes(root, "pilot")
    landsat8 = module is biome
    rgb, rgn = ((3, 2, 1), (3, 2, 4)) if landsat8 else ((2, 1, 0), (2, 1, 3))
    four = (1, 2, 3, 4) if landsat8 else (0, 1, 2, 3)  # blue, green, red, near infrared

    def items() -> Iterator[Item]:
        for scene in chosen:
            dn, label = module.read(scene)
            label = np.where(dn.all(axis=0), label, NO_LABEL).astype(np.uint8)
            others = {}
            if ocm is not None:
                others["ocm4"] = ocm(*some_bands(module, scene, dn, rgn))
                label = with_drafts(label, others)
            yield Item(
                scene.name,
                dn,
                desc.chain_card(module.descriptors(scene)),
                label,
                lambda x, scene=scene: module.reflectance(scene, x),
                others,
                shadow_ok=scene.has_shadow,
                tile=LANDSAT_TILE,
                margin=LANDSAT_MARGIN,
                rgbn=four,
            )

    name = "landsat8" if landsat8 else "landsat7"
    title = f"{title}, {len(chosen)} scenes"
    return SceneSet(test, name, title, len(chosen), items, rgb, RAN if ocm else "")


def heo_set(data: Path, role: str, title: str) -> SceneSet:
    chosen = heo.scenes(data, (role,))

    def items() -> Iterator[Item]:
        told = desc.chain_card(heo.descriptors())
        for scene in chosen:
            dn, label = heo.read(scene)
            drafts = found_drafts(scene.label.parent, ("ocm4",))
            label = with_drafts(np.where(dn.all(axis=0), label, NO_LABEL), drafts)
            ok = shadow_labelled(label)
            yield Item(scene.name, dn, told, label, heo.reflectance, drafts, ok, rgbn=(1, 2, 3, 7))

    title = f"{title}, {len(chosen)} scenes"
    return SceneSet("D", f"heo_{role}", title, len(chosen), items, (3, 2, 1), DRAFTED)


def eaglet2_set(data: Path) -> SceneSet:
    chosen = eaglet2.scenes(data)

    def items() -> Iterator[Item]:
        told = desc.chain_card(eaglet2.descriptors())
        for scene in chosen:
            rho, label = eaglet2.read(scene)
            drafts = found_drafts(scene.label.parent, ("ocm4_nirblue", "ocm4_nirzeros"))
            label = with_drafts(label, drafts)
            ok = shadow_labelled(label)  # blue stands in for the near infrared it has not
            yield Item(scene.name, rho, told, label, same, drafts, ok, rgbn=(2, 1, 0, 2))

    title = f"IRIDE Eaglet II, never seen, {len(chosen)} scenes"
    return SceneSet("E", "eaglet2", title, len(chosen), items, (0, 1, 2), DRAFTED)


def phisat2_descriptors(info: dict) -> np.ndarray:
    """Phi-sat-2's 8 bands as the model is told them (8, 15): band edges, the product's
    pixel on the ground (the square of the same area), top of atmosphere."""
    sensor = spec.load("phisat2")
    pixel = math.sqrt(info["pixel_m"][0] * info["pixel_m"][1])
    rows = []
    for k in range(len(phisat2.BANDS)):
        band = sensor.band(f"Band_{k}")
        rows.append(desc.describe(band.lower_nm, band.upper_nm, pixel, pixel))
    return np.stack(rows)


def phisat2_set(data: Path) -> SceneSet:
    labels, products = data / "labels" / "phisat2", data / "raw" / "phisat2" / "products"
    rows = [r for r in read_rows(labels / "scenes.csv") if r.get("verdict") != "drop"]
    chosen = [r for r in rows if (labels / r["scene"] / "label.tif").exists()]
    if not chosen:
        raise ValueError(f"no labelled Phi-sat-2 scenes in {labels}")

    def items() -> Iterator[Item]:
        for r in chosen:
            product = phisat2.open_product(products / r["product"])
            info = phisat2.describe(product)
            rho = phisat2.reflectance(product, range(1, len(phisat2.BANDS) + 1), info)
            with rasterio.open(labels / r["scene"] / "label.tif") as src:
                label = src.read(1)
            if label.shape != rho.shape[1:]:
                raise ValueError(f"{r['scene']}: bands {rho.shape} do not fit {label.shape}")
            drafts = found_drafts(labels / r["scene"], ("ocm4",))
            label = with_drafts(np.where((rho > 0).all(axis=0), label, NO_LABEL), drafts)
            told = desc.chain_card(phisat2_descriptors(info))
            ok = shadow_labelled(label)
            yield Item(r["scene"], rho, told, label, same, drafts, ok, rgbn=(1, 2, 3, 7))

    title = f"Phi-sat-2, never seen, {len(chosen)} scenes"
    return SceneSet("E", "phisat2", title, len(chosen), items, (3, 2, 1), DRAFTED)


def perusat_set(root: Path) -> SceneSet:
    names = perusat.patches(root, "validation")

    def items() -> Iterator[Item]:
        told = desc.chain_card(perusat.descriptors())
        for n in names:
            dn, label = perusat.read(root, n)
            yield Item(
                f"patch {n}",
                dn,
                told,
                label,
                perusat.reflectance,
                shadow_ok=False,
                rgbn=(0, 1, 2, 3),
            )

    title = f"PeruSat-1, CloudPeru2 validation split, {len(names)} patches (cloud or not)"
    return SceneSet("F", "perusat1", title, len(names), items, (2, 1, 0))


def scene_sets(data: Path, tests, ocm=None) -> list[SceneSet]:
    """The scene sets of tests B-F that have their data; a note for each one that does not.
    ocm(red, green, nir) -> classes runs OmniCloudMask on the Landsat sets."""
    l8, l7 = data / "chips/l8_biome", data / "chips/l7_irish"
    makers = {
        "B": [lambda: landsat_set("B", biome, l8, "Landsat 8 (L8 Biome)", ocm)],
        "C": [lambda: landsat_set("C", irish, l7, "Landsat 7 (L7 Irish)", ocm)],
        "D": [
            lambda: heo_set(data, "test", "IRIDE HEO test scenes"),
            lambda: heo_set(data, "stress", "IRIDE HEO stress scenes (version A02)"),
        ],
        "E": [lambda: eaglet2_set(data), lambda: phisat2_set(data)],
        "F": [lambda: perusat_set(data / "chips/cloudperu2")],
    }
    found = []
    for test in tests:
        for make in makers.get(test, []):
            try:
                found.append(make())
            except (OSError, ValueError, KeyError) as e:
                print(f"test {test}: a set left out, its data is missing or wrong: {e}", flush=True)
    return found


# --- the networks (torch from here on) -----------------------------------------------------


@dataclass
class Net:
    """A trained arm: its network; how many numbers of the chain card it reads (0: the fixed
    arm, blue, green, red and near infrared instead); whether it is told nothing (zeros)."""

    net: object
    fields: int
    blind: bool = False


def load_runs(runs, device) -> dict[str, Net]:
    """Every arm of every finished run under the RUNS folders: {"seed0/real": Net, ...}.
    A run without a list of its arms is the first big run's: real and sim, on the card."""
    import torch

    from catena.model import BandEncoder, Clarity, FixedBands

    roots = [Path(r) for r in ([runs] if isinstance(runs, str | Path) else runs)]
    nets: dict[str, Net] = {}
    where: dict[str, Path] = {}
    for root in roots:
        folders = sorted(path.parent for path in root.glob("**/final.pt"))
        if not folders:
            raise SystemExit(f"no finished runs (final.pt) in {root}")
        for folder in folders:
            state = torch.load(folder / "final.pt", map_location="cpu")
            card = state.get("card")
            if card != [desc.FIELDS[k] for k in desc.CARD]:
                raise SystemExit(f"{folder}: not a run of Clarity on the card ({card})")
            args = state["config"]["args"]
            first = {v: {"fields": desc.N_CARD, "blind": False} for v in VERSIONS}
            for arm, design in (state.get("arms") or first).items():
                name = f"{folder.name}/{arm}"
                if name in nets:
                    raise SystemExit(f"{name} in {where[name]} and {folder}: evaluate them apart")
                fields, widths = int(design["fields"]), tuple(args["widths"])
                if design.get("code"):  # a trunk the architecture search found
                    from catena import nas

                    trunk = nas.build_trunk(design["code"], 2 * args["stem"])
                    net = nas.Candidate(BandEncoder(args["stem"], fields), trunk)
                elif fields == 0:
                    net = FixedBands(widths=widths, stem=args["stem"])
                else:
                    net = Clarity(widths=widths, stem=args["stem"], fields=fields)
                net.load_state_dict(state["models"][arm])
                blind = bool(design.get("blind", False))
                nets[name] = Net(net.to(device).eval(), fields, blind)
                where[name] = folder
    return nets


def run_nets(nets: dict[str, Net], image, told, real, four, amp: bool) -> dict:
    """Every network's logits on a batch (torch): image (n, bands, h, w), told (n, bands, 11)
    and real (n, bands) on its device; four, where blue, green, red and near infrared sit
    among the bands, or None (then the fixed arm says nothing)."""
    import torch

    zeros = torch.zeros_like(told)
    out = {}
    for model, entry in nets.items():
        if entry.fields == 0:
            if four is None:
                continue
            inputs = (image[:, list(four)], None, None)
        else:
            inputs = (image, zeros if entry.blind else told, real)
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16, enabled=amp):
            out[model] = entry.net(*inputs)
    return out


def classifier(nets: dict[str, Net], device, amp: bool):
    """classify(x, told, rgbn) for predict(): every network on a batch of tiles."""
    import torch

    @torch.no_grad()
    def classify(x: np.ndarray, told: np.ndarray, rgbn=None) -> dict[str, np.ndarray]:
        n, bands = x.shape[:2]
        image = torch.from_numpy(x).to(device)
        cards = torch.from_numpy(np.ascontiguousarray(told, np.float32)).to(device)
        cards = cards.expand(n, -1, -1)
        real = torch.ones(n, bands, dtype=torch.bool, device=device)
        logits = run_nets(nets, image, cards, real, rgbn, amp)
        return {m: v.argmax(dim=1).to(torch.uint8).cpu().numpy() for m, v in logits.items()}

    return classify


def cameras(
    nets: dict,
    chips: Path,
    n: int,
    size: int,
    device,
    amp: bool,
    workers: int,
    batch: int,
    cards=("honest",),
):
    """Test A: {camera name: {model: 6 x 4 confusion matrix}}, one camera at a time, each
    model under every card check (named <model>@<check>)."""
    import torch
    from torch.utils.data import DataLoader

    from catena import train as T
    from catena.data import loader
    from catena.evaluate import TEST_CAMERAS, CameraWindows
    from catena.train_clarity import rgbn_slots

    windows = loader.GridWindows(chips, "test", size=size)
    found = {}
    for index, camera in enumerate(TEST_CAMERAS):
        began = time.perf_counter()
        data = CameraWindows(windows, n, camera, index)
        batches = DataLoader(
            data,
            batch_size=batch,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=T.quiet,
        )
        conf: dict = {}
        four = rgbn_slots(camera[2])
        bands = len(camera[2])  # the camera's bands fill the first slots of every window
        with torch.no_grad():
            for b in batches:
                x = b["image"][:, :bands].to(device, non_blocking=True)
                told = desc.chain_card(b["d_told"][:, :bands]).to(device, non_blocking=True)
                y = b["label"].to(device, non_blocking=True)
                ok = y < 4
                for check in cards:
                    varied = vary(check, x, told, four)
                    if varied is None:
                        continue
                    vx, vtold, vfour = varied
                    real = torch.ones(vx.shape[:2], dtype=torch.bool, device=device)
                    for model, logits in run_nets(nets, vx, vtold, real, vfour, amp).items():
                        pairs = y[ok] * 4 + logits.argmax(dim=1)[ok]
                        key = named(model, check)
                        if key not in conf:
                            conf[key] = torch.zeros(4, 4, dtype=torch.long, device=device)
                        conf[key] += torch.bincount(pairs, minlength=16).view(4, 4)
        found[camera[0]] = {}
        for model, m in conf.items():
            six = np.zeros((6, 4), np.int64)
            six[:4] = m.cpu().numpy()
            found[camera[0]][model] = six
        seconds = time.perf_counter() - began
        print(f"  A {camera[0]:<26}{len(data):>5} windows {seconds:5.0f} s", flush=True)
    return found


# --- what is printed ------------------------------------------------------------------------


def group_of(model: str) -> str:
    """seed0/real -> Clarity real; seed1/fixed -> U-Net fixed bands; ocm4 -> OmniCloudMask;
    seed0/sim@shuffled -> Clarity sim [shuffled]."""
    model, _, check = model.partition("@")
    if "/" in model:
        arm = model.rsplit("/", 1)[1]
        title = ARM_TITLES.get(arm, f"Clarity {arm}")
    else:
        title = BASELINES.get(model, model)
    return f"{title} [{check}]" if check else title


def cell(values: list[float]) -> str:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f}±{np.std(values, ddof=1):.3f}"


def table(per_model: dict[str, dict]) -> list[str]:
    """Lines: one per group of models (a version over its seeds, or OmniCloudMask), the mean
    and the spread of each score."""
    groups: dict[str, list[dict]] = {}
    for model, s in per_model.items():
        groups.setdefault(group_of(model), []).append(s)
    lines = [f"  {'':<44}" + "".join(f"{h:>14}" for h in HEADS)]
    for name, members in groups.items():
        cells = [cell([m[k] for m in members]) for k in SHOWN]
        n = len(members)
        seeds = f" ({n} seed{'s' * (n != 1)})" if name not in BASELINES.values() else ""
        lines.append(f"  {name + seeds:<44}" + "".join(f"{c:>14}" for c in cells))
    return lines


def write(path: Path, rows: list[dict]) -> None:
    columns = ["test", "set", "scene", "model", "size", "shadow_ok", "pixels"]
    columns += ["f1_clear", "f1_thin", "f1_thick", "f1_shadow", "f1_mean", "accuracy"]
    columns += ["cloud_f1", "cloud_precision", "cloud_recall", "cloud_ba", "confusion"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})


def row(test: str, name: str, scene: str, model: str, size: int, conf, s: dict, ok) -> dict:
    flat = " ".join(str(int(v)) for v in np.asarray(conf).ravel())
    pixels = int(np.asarray(conf).sum())
    out = {"test": test, "set": name, "scene": scene, "model": model, "size": size}
    return out | {"shadow_ok": ok, "pixels": pixels, "confusion": flat} | s


def score_set(
    ts: SceneSet, classify, maps: Path | None = None, cards=("honest",), extra=None
) -> tuple[list[dict], list[str]]:
    """One set's rows for the CSV and its printed lines; every model under every card check
    (named <model>@<check>), each scene read once; extra(item) -> {name: classes}, other
    models' predictions of each scene (SEnSeIv2)."""
    every: dict[str, np.ndarray] = {}
    shadowed: dict[str, np.ndarray] = {}
    rows, counted, shadow_scenes, began = [], 0, 0, time.perf_counter()
    for item in ts.items():
        if not (item.label != NO_LABEL).any():
            print(f"  {ts.test} {item.name}: no pixel to score, left out", flush=True)
            continue
        started = time.perf_counter()
        preds = {}
        for check in cards:
            found = predict(checked(classify, check), item)
            preds |= {named(model, check): p for model, p in found.items()}
        preds |= item.others
        if extra is not None:
            preds |= extra(item)
        counted += 1
        shadow_scenes += item.shadow_ok
        for model, pred in preds.items():
            conf = confusion(item.label, pred)
            every[model] = every.get(model, 0) + conf
            if item.shadow_ok:
                shadowed[model] = shadowed.get(model, 0) + conf
            rows.append(
                row(
                    ts.test,
                    ts.name,
                    item.name,
                    model,
                    item.tile,
                    conf,
                    scores(conf),
                    item.shadow_ok,
                )
            )
        if maps is not None:
            maps.mkdir(parents=True, exist_ok=True)
            arrays = {m.replace("/", "_"): p for m, p in preds.items() if "@" not in m}
            picture, k = true_colour(item, ts.rgb)
            path = maps / f"{ts.name}__{item.name}.npz"
            np.savez_compressed(path, label=item.label, rgb=picture, step=k, **arrays)
        pixels = int((item.label != NO_LABEL).sum())
        print(
            f"  {ts.test} {ts.name} {item.name}: {pixels:,} pixels "
            f"{time.perf_counter() - started:5.0f} s",
            flush=True,
        )
    per_model = {}
    for model, conf in every.items():
        s = pooled(conf, shadowed.get(model, np.zeros((6, 4), np.int64)))
        per_model[model] = s
        rows.append(row(ts.test, ts.name, "all", model, 0, conf, s, ""))
    pixels = int(next(iter(every.values())).sum()) if every else 0
    lines = [
        f"\nTest {ts.test}: {ts.title}; {counted} scored, {shadow_scenes} with shadow labels, "
        f"{pixels:,} pixels ({(time.perf_counter() - began) / 60:.1f} min)"
    ]
    if per_model:
        lines += table(per_model)
    if any(m in OCM for m in per_model) and ts.note:
        lines.append(f"  ({ts.note})")
    return rows, lines


def true_colour(item: Item, rgb, longest: int = 1000) -> tuple[np.ndarray, int]:
    """The scene's true colour, every k-th pixel (k the smallest that keeps it within
    `longest` across), as (h, w, 3) uint8: the review's 2-98% stretch and gamma; and k."""
    k = viz.step(item.image.shape[1:], longest)
    rho = item.to_rho(np.ascontiguousarray(item.image[None, :, ::k, ::k]))[0]
    picture = viz.stretch(rho[list(rgb)], gamma=GAMMA)
    return np.round(picture * 255).astype(np.uint8), k


def camera_lines(found: dict, size: int = TILE) -> tuple[list[dict], list[str]]:
    """Test A's rows and lines: Sentinel-2 as delivered, then the mean of the cameras inside
    the training range and of those beyond it."""
    rows, per_camera = [], {}
    for camera, models in found.items():
        for model, conf in models.items():
            s = scores(conf)
            per_camera.setdefault(camera, {})[model] = s
            rows.append(row("A", "sentinel2", camera, model, size, conf, s, True))
    groups = [
        ("Sentinel-2 as delivered", [c for c in per_camera if c == "sentinel2"]),
        ("18 cameras inside the range", [c for c in per_camera if c.startswith("rgbn_")]),
        ("4 cameras beyond the range", [c for c in per_camera if c.startswith("beyond_")]),
    ]
    lines = ["\nTest A: CloudSEN12+ test split (Sentinel-2, seen), per camera, then averaged"]
    for title, cameras in groups:
        if not cameras:
            continue
        models = dict.fromkeys(m for c in cameras for m in per_camera[c])
        mean = {
            model: {
                k: float(
                    np.mean([per_camera[c][model][k] for c in cameras if model in per_camera[c]])
                )
                for k in SHOWN
            }
            for model in models
        }
        lines.append(f" {title}:")
        lines += table(mean)
    return rows, lines


# --- the command ----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, nargs="+", help="folders of finished runs (runs/big)")
    p.add_argument("--data", type=Path, default=Path("/root/catena_data"))
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=list(TESTS))
    p.add_argument("--windows", type=int, default=800, help="Test A: windows per camera")
    p.add_argument("--a-size", type=int, default=TILE, help="Test A: window size, 10 m pixels")
    p.add_argument("--batch", type=int, default=16, help="Test A: windows per batch")
    p.add_argument("--workers", type=int, default=6, help="Test A: data-loading processes")
    p.add_argument("--device", help="default cuda:0 if there is a GPU, else cpu")
    p.add_argument("--ocm", action="store_true", help="OmniCloudMask on Landsat 8 and 7 too")
    p.add_argument("--sensei", action="store_true", help="SEnSeIv2 on the scenes of B-F")
    p.add_argument("--sensei-batch", type=int, default=4, help="SEnSeIv2's patches per batch")
    p.add_argument("--out", type=Path, help="the CSV (default: see above)")
    p.add_argument("--maps", type=Path, help="keep every scene's predictions here (.npz)")
    p.add_argument(
        "--cards", nargs="+", default=["honest"], choices=list(CARDS), help="card checks (above)"
    )
    args = p.parse_args(argv)
    if args.a_size <= 0 or args.a_size % 24:
        p.error("--a-size must be a multiple of 24: of 8 at 10 m and at 30 m")
    return args


def default_out(runs, tests, ocm: bool, cards=("honest",), sensei: bool = False) -> Path:
    """In the (last) RUNS folder: clarity_tests.csv for the six tests, plainly told,
    clarity_tests_<tests>[_ocm][_cards][_sensei].csv otherwise."""
    folder = Path(runs if isinstance(runs, str | Path) else runs[-1])
    checks = list(cards) != ["honest"]
    if set(tests) == set(TESTS) and not ocm and not checks and not sensei:
        return folder / "clarity_tests.csv"
    more = ("_ocm" if ocm else "") + ("_cards" if checks else "") + ("_sensei" if sensei else "")
    return folder / f"clarity_tests_{''.join(sorted(tests))}{more}.csv"


def main(argv: list[str] | None = None) -> None:
    import torch

    args = parse_args(argv)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = True
    nets = load_runs(args.runs, device)
    tests = [t for t in TESTS if t in args.tests]
    out = args.out or default_out(args.runs, tests, args.ocm, args.cards, args.sensei)
    print(f"{len(nets)} networks ({', '.join(nets)}) on {device}; tests {' '.join(tests)}")
    if args.cards != ["honest"]:
        print(f"card checks: {', '.join(args.cards)}", flush=True)
    started, rows, report = time.perf_counter(), [], []

    def keep(more: list[dict], lines: list[str]) -> None:
        """A set's results, printed and added to the CSV at once, so none is lost later."""
        rows.extend(more)
        report.extend(lines)
        print("\n".join(lines), flush=True)
        write(out, rows)

    if "A" in tests:
        chips = args.data / "chips" / "cloudsen12plus"
        if (chips / "index.csv").exists():
            found = cameras(
                nets,
                chips,
                args.windows,
                args.a_size,
                device,
                amp,
                args.workers,
                args.batch,
                args.cards,
            )
            keep(*camera_lines(found, args.a_size))
        else:
            print(f"test A left out: no {chips / 'index.csv'}", flush=True)
    classify = classifier(nets, device, amp)
    ocm = (lambda r, g, n: omnicloudmask(r, g, n, device)) if args.ocm else None
    extra = None
    if args.sensei:
        run_sensei = sensei(device, args.sensei_batch)

        def extra(item: Item) -> dict:
            return {"sensei2": run_sensei(whole_scene(item), item.told)}

    for ts in scene_sets(args.data, [t for t in tests if t != "A"], ocm):
        try:
            keep(*score_set(ts, classify, args.maps, args.cards, extra))
        except Exception:  # one broken set must not cost the others
            traceback.print_exc()
            print(f"test {ts.test}: {ts.name} stopped by the error above; the rest go on")
    if not rows:
        raise SystemExit("nothing was scored")
    print("\n\nAll the tests:" + "\n".join(report))
    minutes = (time.perf_counter() - started) / 60
    print(f"\nwrote {out} ({len(rows)} rows) in {minutes:.0f} min", flush=True)


if __name__ == "__main__":
    main()
