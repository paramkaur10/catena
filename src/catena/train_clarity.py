"""The big run: Clarity trained on four satellites, two arms side by side.

    python -m catena.train_clarity OUT [--data /root/catena_data] [--seed 0] [--steps 60000]
        [--arms real sim] [--shares 0.4 0.2 0.2 0.2] [--devices cuda:0 cuda:1]
        [--q-weights 1 1 1] [--faults 0] [--fault-gain 0.3] [--fault-nir 0.4] [--coarser 0]
        [--hailo] [--freeze-norms 0.9]
        [--chain-jitter 1] [--trunk CODE [CODE]]

Clarity reads, for each band, only what every product states (catena.descriptor.card): its
wavelengths, surface or top-of-atmosphere reflectance, the band's pixel size against the
grid's, and the grid's pixel size on the ground. Nothing about the imaging chain.

Two arms (--arms), one per device (--devices, in the same order), see the same windows in the
same order, with the same bands and gains; arms of one design start from the same weights.
The first big run's two, the default:
  real    every sample as its satellite delivered it; the card;
  sim     the same, except that 80% of the Sentinel-2 samples are re-imaged by a made-up
          camera (catena.sim): 30 or 50 m pixels, or 10 m, with more blur and noise, fewer
          bits and a saturation from 0.25 to 1.6. These are the same Sentinel-2 windows and
          labels, processed: no new images and no new labels.
The arms that test the method learn from sim's samples and differ from it only in what they
read:
  sensei  the card without its pixel sizes: the wavelengths, and surface or top of
          atmosphere; what SEnSeIv2 is told (a band's type and edges);
  chain   the chain card (catena.descriptor.chain_card): the card, then what the MTF and the
          SNR say where they are known (Sentinel-2's spec, the made-up cameras, Landsat 8's
          noise), with descriptor dropout (catena.descriptor.hide) and errors the size real
          specs carry (--chain-jitter, catena.descriptor.jitter), so it copes where they
          are unknown or roughly right;
  blind   nothing: every band's card is zeros, so it cannot tell one band from another;
  fixed   no band encoder: catena.model.FixedBands, the U-Net on blue, green, red and near
          infrared in that order, as a network made for one band set; the same windows,
          pixel sizes and made-up cameras or not as sim, but always these four bands, with
          gains and a camera of their own.
  nas     a trunk the architecture search found (catena.nas): the band encoder in front of
  nas2    PyNAS's U-Net for an architecture code (--trunk, one code per nas arm, in --arms
          order), trained end to end like the others; it reads the card.
With the same seed every arm sees what the first big run's sim version saw: what an arm adds
(its four bands, dropout and errors) is drawn from random streams of its own.
Three arms learn from real's samples instead, with no made-up camera at all:
  chain_real   the chain card of each sample as delivered: the MTF and the SNR only where a
               product's documents state them (Sentinel-2's spec, Landsat 8's noise), never a
               made-up value, with chain's dropout and errors. Beside real, on the same
               samples: what telling the measured MTF and SNR alone is worth;
  sensei_real  sensei's four numbers;
  blind_real   nothing.
A run with no batch from HEO (--shares 0.5 0.25 0 0.25) leaves HEO as a satellite never
seen, beside IRIDE Eaglet II, Phi-sat-2 and Landsat 7.

So real and sim differ only in the made-up cameras. Two options change sim's samples only:
  --q-weights  how often its Sentinel-2 samples get 10, 30 and 50 m pixels (default 1 1 1:
               each a third; 0.5 0.3 0.2 gives 10 m half the time, so 70% are made up);
  --faults     the share of its samples, from every satellite, given a calibration fault
               that no product states (catena.sim.calibration_fault): up to --fault-nir of
               near infrared leaking into each visible band, and every band's gain off by up
               to --fault-gain, as IRIDE Eaglet II delivers its bands. Not on the card.

One option changes what every arm learns from, made-up cameras or not:
  --coarser    the share of the batches from Landsat 8, HEO and PeruSat-1 shown at a pixel 2
               or 3 times larger (Landsat 8: 2 only), as a coarser camera would deliver them:
               the image averaged over 2 x 2 or 3 x 3 pixels, the label the most common code
               of each block, and the card told the new pixel (gsd, sampling, a known blur
               narrower, a known noise lower). Without it each real satellite has one pixel
               size, which the networks learn as a name; with it the pixel size says what it
               means. Drawn from a random stream of its own: at 0 every sample is as before.

Made for the Hailo-8 (--hailo): every network of the run uses only layers the chip's compiler
takes as they are (catena.model, catena.nas): BatchNorm for GroupNorm, ReLU for GELU, nearest
neighbours for bilinear upsampling. A batch comes from one satellite, so BatchNorm's statistics
of one batch are one satellite's: NORM_BATCHES batches before --freeze-norms of the steps, the
statistics restart as a plain average of the batches (every satellite in its share), and from
then on they are frozen, the rest of the training learning with the constants that fly.

Each batch comes from one satellite, drawn with --shares (Sentinel-2, Landsat 8, HEO,
PeruSat-1); the data under DATA:
  sentinel2  chips/cloudsen12plus, the train split: 240 px windows at 10 m, 3-13 of 13 bands
  landsat8   chips/l8_biome, the scenes outside the --hold-out table (pilot: the 24 of Test
             B): 80 px windows at 30 m, 3-8 of 8 bands
  heo        chips/iride_heo (python -m catena.data.heo) with the labels of the train scenes
             in labels/iride_heo: 240 px windows at 5 m, 3-8 of 8 bands
  perusat1   chips/cloudperu2, the train split: 240 px windows at 2.8 m, 3-4 of 4 bands,
             surface reflectance, labelled only cloud or not cloud
Every sample: a window at least half labelled, a random quarter turn or flip, a random subset
of its bands, each band's gain moved by up to 5%; a PeruSat-1 sample also gets one gain for
all its bands, between 0.7 and 1.5, as CloudPeru2's scale is not documented.

Loss, optimiser and validation as in catena.train: the partial-label cross-entropy (a cloud
of unknown thickness counts as thin or thick, not cloud as clear or shadow), AdamW with a
warm-up and a cosine decay, float32 weights with bf16 sums on the GPU (--no-amp: float32
throughout), and validation on the 6 cameras of the CloudSEN12+ validation split, with the
honest chain cards (the fixed arm: each camera's blue, green, red and B08, or B8A without it).

OUT receives config.json, metrics.jsonl, last.pt (both arms with their optimisers and
schedules, at every validation) and final.pt, which lists each arm's design (arms). After a
stop, the same command goes on from last.pt, with the same samples it would have drawn; a
finished run is never touched.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from catena import descriptor as desc
from catena import sim
from catena import train as T
from catena.data import biome, heo, loader, perusat
from catena.model import BandEncoder, Clarity, FixedBands, count_parameters

VERSIONS = ("real", "sim")  # the first big run's arms, the default
SOURCES = ("sentinel2", "landsat8", "heo", "perusat1")
SHARES = (0.4, 0.2, 0.2, 0.2)
TOPS = (0.25, 1.6)  # the made-up cameras' saturation, in reflectance
PERUSAT_GAIN = (0.7, 1.5)  # one gain per PeruSat-1 sample, drawn on a log scale
N_BANDS = T.N_BANDS  # band slots: 13, Sentinel-2's count
PLACE = 8  # the sim version's placeholder for a sample it sees as the real one does
NIR_LEAK = {  # per satellite: the visible bands a fault mixes near infrared into, and the NIR
    "sentinel2": ((0, 1, 2, 3), 7),  # B01-B04; B08
    "landsat8": ((0, 1, 2, 3), 4),  # B1-B4; B5
    "heo": ((1, 2, 3), 7),  # B01-B03; B07
    "perusat1": ((0, 1, 2), 3),  # blue, green, red; nir
}
COARSER = {"landsat8": (2,), "heo": (2, 3), "perusat1": (2, 3)}  # --coarser: pixel factors
SAMPLING, PSF = desc.FIELDS.index("sampling"), desc.FIELDS.index("psf")
NOISE = [desc.FIELDS.index("noise_dark"), desc.FIELDS.index("noise_bright")]
KNOWN_NOISE = desc.FIELDS.index("known_noise")
RGBN = {  # where blue, green, red and near infrared sit among each satellite's bands
    "sentinel2": (1, 2, 3, 7),  # B02, B03, B04, B08
    "landsat8": (1, 2, 3, 4),  # B2, B3, B4, B5
    "heo": (1, 2, 3, 7),  # B01, B02, B03, B07
    "perusat1": (0, 1, 2, 3),
}
# what a resumed run must share with the run it goes on from; a setting older runs did not
# write counts as its default (OLDER)
SAME_ARGS = ("seed", "steps", "batch", "size", "landsat_size", "heo_size", "perusat_size")
SAME_ARGS += ("shares", "p_s2", "widths", "stem", "lr", "weight_decay", "warmup", "hold_out")
SAME_ARGS += ("q_weights", "faults", "fault_gain", "fault_nir", "arms", "chain_jitter", "trunk")
SAME_ARGS += ("coarser", "hailo", "freeze_norms")
OLDER = {"q_weights": [1.0, 1.0, 1.0], "faults": 0.0, "fault_gain": 0.3, "fault_nir": 0.4}
OLDER |= {"coarser": 0.0, "hailo": False, "freeze_norms": 0.9}
NORM_BATCHES = 500  # --hailo: the batches BatchNorm's statistics are averaged over, then frozen
OLDER |= {"arms": list(VERSIONS), "chain_jitter": 1.0, "trunk": None}


@dataclass(frozen=True)
class Arm:
    """What an arm learns from and reads: data, the samples ("real" or "sim"); fields, how
    many numbers of each band's chain card its Clarity reads (0: FixedBands, no card at all);
    blind, its cards are zeros."""

    data: str
    fields: int
    blind: bool = False


ARMS = {
    "real": Arm("real", desc.N_CARD),
    "sim": Arm("sim", desc.N_CARD),
    "sensei": Arm("sim", 4),  # lo, mid, hi, surface
    "chain": Arm("sim", desc.N_CHAIN_CARD),
    "blind": Arm("sim", desc.N_CARD, blind=True),
    "fixed": Arm("sim", 0),
    "nas": Arm("sim", desc.N_CARD),
    "nas2": Arm("sim", desc.N_CARD),
    "sensei_real": Arm("real", 4),  # the three that never see a made-up camera
    "chain_real": Arm("real", desc.N_CHAIN_CARD),
    "blind_real": Arm("real", desc.N_CARD, blind=True),
}
NAS = ("nas", "nas2")  # the arms whose trunk is an architecture code


def build(
    name: str, widths, stem: int, code: str | None = None, hailo: bool = False
) -> torch.nn.Module:
    """An arm's network, its weights drawn from torch's generator as it stands; a nas arm's
    trunk is PyNAS's U-Net for the architecture code. hailo: made for the Hailo-8."""
    if name in NAS:
        from catena import nas  # PyNAS is needed only here

        if not code:
            raise ValueError(f"the {name} arm needs an architecture code")
        head = BandEncoder(stem, ARMS[name].fields, hailo)
        return nas.Candidate(head, nas.build_trunk(code, 2 * stem, hailo=hailo), hailo=hailo)
    if ARMS[name].fields == 0:
        return FixedBands(widths=tuple(widths), stem=stem, hailo=hailo)
    return Clarity(widths=tuple(widths), stem=stem, fields=ARMS[name].fields, hailo=hailo)


def batch_norms(net: torch.nn.Module) -> list:
    """The BatchNorm layers of a network (none but in one made for the Hailo-8)."""
    return [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def norm_steps(args) -> tuple[int, int] | None:
    """--hailo: the step from which BatchNorm's statistics are a plain average of the batches,
    and the step from which they are frozen (never, at --freeze-norms 1); None without."""
    if not args.hailo:
        return None
    frozen = round(args.freeze_norms * args.steps)
    return max(frozen - NORM_BATCHES, 0), frozen


def average_norms(net: torch.nn.Module, restart: bool = True) -> None:
    """From now on BatchNorm's statistics are the plain average of the batches seen (restart:
    from none; else going on from those a stopped run saved, with their count)."""
    for m in batch_norms(net):
        if restart:
            m.reset_running_stats()
        m.momentum = None


def freeze_norms(net: torch.nn.Module) -> None:
    """BatchNorm's statistics frozen: used as they fly, never updated again."""
    net.frozen_norms = True
    for m in batch_norms(net):
        m.eval()


def train_mode(net: torch.nn.Module) -> torch.nn.Module:
    """net.train(), with frozen BatchNorm statistics kept frozen."""
    net.train()
    if getattr(net, "frozen_norms", False):
        for m in batch_norms(net):
            m.eval()
    return net


def codes_of(arms, trunk) -> dict[str, str]:
    """{nas arm: its architecture code}, in --arms order."""
    found = [a for a in arms if a in NAS]
    return dict(zip(found, trunk or [], strict=True))


def rgbn_slots(names) -> list[int]:
    """Where blue, green, red and near infrared sit among a Sentinel-2 camera's bands: B02,
    B03, B04, and B08, or B8A where there is no B08."""
    names = list(names)
    nir = "B08" if "B08" in names else "B8A"
    return [names.index(b) for b in ("B02", "B03", "B04", nir)]


def pack(x: np.ndarray, d: np.ndarray, real: np.ndarray, label: np.ndarray) -> dict:
    return {"image": x, "desc": d, "real": real, "label": label}


class Samples(Dataset):
    """Training samples for both arms. Sample i depends only on the seed and i, so a run
    repeats exactly whatever the number of data-loading processes, and goes on exactly after
    a stop: `start` is the first sample, step x batch.

    Each item holds the real version's image (13 slots, the kept bands first), chain card
    (told), band slots and label; the sim version's image, chain card (sim_told, with
    descriptor dropout and errors, which touch only its MTF and SNR numbers) and label; and,
    with fixed, the fixed arm's blue, green, red and near infrared (fixed_image) on the sim
    version's grid. hid_told is the real version's chain card with the same dropout and
    errors, for the arms that read the MTF and the SNR of samples as delivered (chain_real):
    sim_told itself wherever the sim version sees what the real one sees. same is True when
    the sim version sees what the real one sees (every batch but Sentinel-2's, unless the sim
    version gets faults); its own image and label are then small placeholders. The real
    version's samples do not depend on q_weights, faults, fixed or chain_jitter, and the sim
    version's images not on fixed or chain_jitter: they are drawn first, and the rest from
    random streams of its own. With coarser, a share of the batches from Landsat 8, HEO and
    PeruSat-1 come at a pixel 2 or 3 times larger (coarsened), for every version."""

    def __init__(
        self,
        windows: loader.TrainingWindows,
        batch: int,
        seed: int,
        landsat: biome.Windows | None = None,
        heo_windows: heo.Windows | None = None,
        peru: perusat.Windows | None = None,
        shares=SHARES,
        p_s2: float = 0.6,
        gain: float = 0.05,
        min_bands: int = 3,
        start: int = 0,
        q_weights=None,
        faults: float = 0.0,
        fault_gain: float = 0.3,
        fault_nir: float = 0.4,
        fixed: bool = False,
        chain_jitter: float = 1.0,
        coarser: float = 0.0,
    ) -> None:
        self.windows, self.batch, self.seed = windows, batch, seed
        self.coarser = coarser
        self.data = {"landsat8": landsat, "heo": heo_windows, "perusat1": peru}
        self.shares = np.asarray(shares, np.float64) / float(np.sum(shares))
        for name, share in zip(SOURCES, self.shares, strict=True):
            if share > 0 and name != "sentinel2" and self.data[name] is None:
                raise ValueError(f"{name} has a share of the batches but no data")
        self.p_s2, self.gain, self.min_bands, self.start = p_s2, gain, min_bands, start
        self.q_weights = None
        if q_weights is not None:
            w = np.asarray(q_weights, np.float64)
            if len(w) != len(sim.Q_CHOICES) or (w < 0).any() or w.sum() <= 0:
                raise ValueError(f"q_weights: one weight per pixel size {sim.Q_CHOICES}, >= 0")
            if not np.allclose(w, w[0]):  # equal weights: the draw of the first big run
                self.q_weights = w / w.sum()
        self.faults, self.fault_gain, self.fault_nir = faults, fault_gain, fault_nir
        if fixed and faults > 0:
            raise ValueError("the fixed arm takes no faults")
        self.fixed, self.chain_jitter = fixed, chain_jitter

    def faulty(self, x: np.ndarray, source: str, rng: np.random.Generator) -> np.ndarray | None:
        """Every band of a sample with a calibration fault, for the sim version, or None when
        the sample draws none (faults is the chance of one)."""
        if self.faults <= 0 or rng.random() >= self.faults:
            return None
        visible, nir = NIR_LEAK[source]
        return sim.calibration_fault(x, visible, nir, rng, self.fault_gain, self.fault_nir)

    def __len__(self) -> int:
        return len(self.windows) - self.start

    def source(self, i: int) -> str:
        """Where sample i comes from: the same for a whole batch (i // batch)."""
        rng = np.random.default_rng([self.seed, 3, i // self.batch])
        return SOURCES[int(rng.choice(len(SOURCES), p=self.shares))]

    def coarser_factor(self, i: int, source: str, size: int) -> int:
        """How many times larger the pixels of sample i are shown (--coarser): 1, or a factor
        of COARSER that divides its windows of `size` pixels into a multiple of 8, the same
        for its whole batch (i // batch), from a random stream of its own."""
        if self.coarser <= 0 or source not in COARSER:
            return 1
        rng = np.random.default_rng([self.seed, 6, i // self.batch])
        if rng.random() >= self.coarser:
            return 1
        fits = [f for f in COARSER[source] if size % f == 0 and (size // f) % 8 == 0]
        return int(rng.choice(fits)) if fits else 1

    def pixel_size(self, i: int) -> int:
        """The q of the sim version's Sentinel-2 sample i, shared by its batch."""
        rng = np.random.default_rng([self.seed, 2, i // self.batch])
        return int(rng.choice(sim.Q_CHOICES, p=self.q_weights))

    def bands(self, total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Which bands a sample keeps, and their gains."""
        n = int(rng.integers(min(self.min_bands, total), total + 1))
        keep = np.sort(rng.choice(total, n, replace=False))
        return keep, rng.uniform(1 - self.gain, 1 + self.gain, n)

    def four_bands(self, rho: np.ndarray, source: str, frng, gain: float = 1.0):
        """The fixed arm's blue, green, red and near infrared of a sample as delivered, with
        gains of its own (frng), or None without the fixed arm."""
        if frng is None:
            return None
        four = rho[list(RGBN[source])]
        gains = gain * frng.uniform(1 - self.gain, 1 + self.gain, len(four))
        return (four * gains[:, None, None]).astype(np.float32)

    def from_sentinel2(self, i: int, rng: np.random.Generator, frng=None) -> dict:
        s = self.windows[i]  # drawn, turned and flipped by the loader
        keep, gains = self.bands(N_BANDS, rng)
        scene = (s["image"], s["label"], s["sun_zenith"])
        x, d, real, label = T.render(*scene, 1, keep, None, rng, gains)
        out = pack(x, d, real, label)
        q = self.pixel_size(i)  # everything the real version sees is drawn by now
        made_up = q > 1 or rng.random() >= self.p_s2
        image = self.faulty(s["image"], "sentinel2", rng)
        if made_up or image is not None:
            image = s["image"] if image is None else image
            camera = sim.imagine_camera(rng, len(keep), q, tops=TOPS)[1] if made_up else None
            x, d, _, label = T.render(
                image, s["label"], s["sun_zenith"], q, keep, camera, rng, gains
            )
        out |= {"sim_image": x, "sim_desc": d, "sim_label": label, "same": False}
        if frng is not None:  # the fixed arm's four bands, made up or not as sim's
            four = np.array(RGBN["sentinel2"])
            gains = frng.uniform(1 - self.gain, 1 + self.gain, len(four))
            camera = sim.imagine_camera(frng, len(four), q, tops=TOPS)[1] if made_up else None
            x, *_ = T.render(*scene, q, four, camera, frng, gains)
            out["fixed_image"] = x[: len(four)]
        return out

    def as_delivered(
        self, x: np.ndarray, d: np.ndarray, label, turn: int, sim_x=None, fixed=None
    ) -> dict:
        """A sample as its satellite delivered it: given a quarter turn or flip (turn, 0-7),
        its bands in the first slots. Both versions see it alike, unless sim_x, the same bands
        with a fault, is given (or the sim version gets faults at all: then it sees x as its
        own copy). fixed: the fixed arm's four bands, turned alike."""
        n, size = len(x), x.shape[-1]
        image = np.zeros((N_BANDS, size, size), np.float32)
        image[:n] = loader.dihedral(x, turn)
        descs = np.zeros((N_BANDS, desc.N_FIELDS), np.float32)
        descs[:n] = d
        real = np.zeros(N_BANDS, bool)
        real[:n] = True
        out = pack(image, descs, real, loader.dihedral(label, turn).astype(np.int64))
        if fixed is not None:
            out["fixed_image"] = loader.dihedral(fixed, turn)
        if sim_x is None and self.faults <= 0:
            return out | {
                "sim_image": np.zeros((N_BANDS, PLACE, PLACE), np.float32),
                "sim_desc": descs,
                "sim_label": np.zeros((PLACE, PLACE), np.int64),
                "same": True,
            }
        own = image.copy()
        if sim_x is not None:
            own[:n] = loader.dihedral(sim_x, turn)
        return (
            out | {"sim_image": own, "sim_desc": descs, "sim_label": out["label"]} | {"same": False}
        )

    def with_fault(self, rho: np.ndarray, source: str, keep, gains, rng) -> np.ndarray | None:
        """The kept bands of a sample with a fault for the sim version, or None."""
        bad = self.faulty(rho, source, rng)
        return None if bad is None else bad[keep] * gains[:, None, None]

    def from_landsat8(self, rng: np.random.Generator, frng=None) -> dict:
        data = self.data["landsat8"]
        scene, dn, label = data.draw(rng)  # clear is 5 in a scene without shadow labels
        keep, gains = self.bands(len(biome.BANDS), rng)
        rho = biome.reflectance(scene, dn)
        x, turn = rho[keep] * gains[:, None, None], int(rng.integers(8))
        sim_x = self.with_fault(rho, "landsat8", keep, gains, rng)  # drawn after the real's
        fixed = self.four_bands(rho, "landsat8", frng)
        return self.as_delivered(x, data.cards[scene.name][keep], label, turn, sim_x, fixed)

    def from_heo(self, rng: np.random.Generator, frng=None) -> dict:
        data = self.data["heo"]
        _, dn, label = data.draw(rng)
        keep, gains = self.bands(len(heo.BANDS), rng)
        rho = heo.reflectance(dn)
        x, turn = rho[keep] * gains[:, None, None], int(rng.integers(8))
        sim_x = self.with_fault(rho, "heo", keep, gains, rng)
        fixed = self.four_bands(rho, "heo", frng)
        return self.as_delivered(x, data.card[keep], label, turn, sim_x, fixed)

    def from_perusat1(self, rng: np.random.Generator, frng=None) -> dict:
        data = self.data["perusat1"]
        _, dn, label = data.draw(rng)
        keep, gains = self.bands(len(perusat.BANDS), rng)
        low, high = (math.log(g) for g in PERUSAT_GAIN)
        gains = gains * math.exp(rng.uniform(low, high))  # one more gain for the whole sample
        rho = perusat.reflectance(dn)
        x, turn = rho[keep] * gains[:, None, None], int(rng.integers(8))
        sim_x = self.with_fault(rho, "perusat1", keep, gains, rng)
        whole = math.exp(frng.uniform(low, high)) if frng is not None else 1.0
        fixed = self.four_bands(rho, "perusat1", frng, whole)
        return self.as_delivered(x, data.card[keep], label, turn, sim_x, fixed)

    def __getitem__(self, j: int) -> dict:
        i = self.start + j
        rng = np.random.default_rng([self.seed, 1, i])
        frng = np.random.default_rng([self.seed, 5, i]) if self.fixed else None
        source = self.source(i)
        if source == "sentinel2":
            out = self.from_sentinel2(i, rng, frng)
        elif source == "landsat8":
            out = self.from_landsat8(rng, frng)
        elif source == "heo":
            out = self.from_heo(rng, frng)
        else:
            out = self.from_perusat1(rng, frng)
        f = self.coarser_factor(i, source, out["image"].shape[-1])
        if f > 1:
            out = coarsened(out, f)
        n = int(out["real"].sum())
        drop = np.random.default_rng([self.seed, 4, i])  # the chain arm's dropout and errors
        hidden = out.pop("sim_desc").copy()
        hidden[:n] = desc.jitter(desc.hide(hidden[:n], drop), drop, self.chain_jitter)
        again = np.random.default_rng([self.seed, 4, i])  # the same draws, for the real version
        shown = out["desc"].copy()
        shown[:n] = desc.jitter(desc.hide(shown[:n], again), again, self.chain_jitter)
        out["told"] = desc.chain_card(out.pop("desc"))
        out["sim_told"] = desc.chain_card(hidden)
        out["hid_told"] = desc.chain_card(shown)
        return out | {"source": SOURCES.index(source)}


def block_mean(x: np.ndarray, f: int) -> np.ndarray:
    """x (..., h, w) averaged over f x f pixels: (..., h / f, w / f), its own dtype."""
    h, w = x.shape[-2:]
    return x.reshape(*x.shape[:-2], h // f, f, w // f, f).mean((-3, -1)).astype(x.dtype)


def coarse_label(label: np.ndarray, f: int) -> np.ndarray:
    """A label (h, w) of codes 0-5 on a grid f times coarser: the most common code of each
    f x f block, or no label where fewer than half of the block's pixels have one."""
    h, w = label.shape
    blocks = label.reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, -1)
    counts = np.stack([(blocks == c).sum(-1) for c in range(len(T.SETS))], -1)
    out = counts.argmax(-1).astype(label.dtype)
    out[counts.sum(-1) < f * f / 2] = T.NO_LABEL
    return out


def coarse_card(d: np.ndarray, f: int) -> np.ndarray:
    """Descriptors (bands, 15) of bands averaged into pixels f times larger: the grid's pixel
    f times larger, a band's own pixel against it smaller (never below the grid's), a known
    blur f times narrower in the new pixels, a known noise f times lower. A copy; an empty
    slot (all 0) gets the new pixel size only, which no network reads there."""
    d = d.copy()
    d[:, desc.GSD] += math.log2(f)
    d[:, SAMPLING] = np.maximum(d[:, SAMPLING] - math.log2(f), 0)
    d[:, PSF] /= f  # an unknown blur (0) stays 0
    d[:, NOISE] -= math.log10(f) * d[:, KNOWN_NOISE : KNOWN_NOISE + 1]
    return d


def coarsened(out: dict, f: int) -> dict:
    """A sample as delivered (Samples.as_delivered) at a pixel f times larger, every version
    alike; the sim version's placeholders stay placeholders."""
    out = dict(out)
    out["image"] = block_mean(out["image"], f)
    out["label"] = coarse_label(out["label"], f)
    out["desc"] = coarse_card(out["desc"], f)
    out["sim_desc"] = coarse_card(out["sim_desc"], f)
    if not out["same"]:
        out["sim_image"] = block_mean(out["sim_image"], f)
        out["sim_label"] = coarse_label(out["sim_label"], f)
    if "fixed_image" in out:
        out["fixed_image"] = block_mean(out["fixed_image"], f)
    return out


def arm_inputs(batch: dict, name: str) -> tuple:
    """What an arm sees in a batch: image, chain cards (None for the fixed arm), band slots
    (None for the fixed arm) and label."""
    arm = ARMS[name]
    own = arm.data == "sim" and not bool(batch["same"][0])
    label = batch["sim_label"] if own else batch["label"]
    if arm.fields == 0:
        return batch["fixed_image"], None, None, label
    image = batch["sim_image"] if own else batch["image"]
    if arm.data == "sim":
        told = batch["sim_told"]
    elif arm.fields > desc.N_CARD:  # the MTF and the SNR of samples as delivered: with dropout
        told = batch["hid_told"]
    else:
        told = batch["told"]
    if arm.blind:
        told = torch.zeros_like(told)
    return image, told, batch["real"], label


def forward(net, x, told, real, device: torch.device, amp: bool) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    if told is not None:
        told = told.to(device, non_blocking=True)
        real = real.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return net(x, told, real)


@torch.no_grad()
def validate(models: dict, devices: dict, data: DataLoader, amp: dict) -> dict:
    """Confusion matrices (camera, label, prediction) of every arm on the validation cameras:
    the same images and the honest chain cards (blind: zeros; fixed: each camera's blue,
    green, red and near infrared)."""
    conf = {
        a: torch.zeros(len(T.VAL_CAMERAS), 4, 4, dtype=torch.long, device=devices[a])
        for a in models
    }
    for net in models.values():
        net.eval()
    for batch in data:
        c = int(batch["camera"][0])
        told = desc.chain_card(batch["d_told"])
        four = rgbn_slots(T.VAL_CAMERAS[c][2])
        for a, net in models.items():
            if ARMS[a].fields == 0:
                x, d, real = batch["image"][:, four], None, None
            else:
                d = torch.zeros_like(told) if ARMS[a].blind else told
                x, real = batch["image"], batch["real"]
            logits = forward(net, x, d, real, devices[a], amp[a])
            y = batch["label"].to(devices[a], non_blocking=True)
            ok = y < 4
            pairs = y[ok] * 4 + logits.argmax(dim=1)[ok]
            conf[a][c] += torch.bincount(pairs, minlength=16).view(4, 4)
    for net in models.values():
        train_mode(net)
    return {a: m.cpu().numpy() for a, m in conf.items()}


def designs(arms, config: dict) -> dict:
    """Each arm's design, with its architecture code for a nas arm, and hailo when it is made
    for the Hailo-8."""
    trunks = config.get("trunks", {})
    hailo = {"hailo": True} if config.get("args", {}).get("hailo") else {}
    return {a: asdict(ARMS[a]) | ({"code": trunks[a]} if a in trunks else {}) | hailo for a in arms}


def save(path: Path, step: int, models: dict, config: dict, opts=None, scheds=None) -> None:
    """Both arms (with their optimisers and schedules, to go on from) and their designs,
    written atomically."""
    state = {
        "step": step,
        "config": config,
        "card": [desc.FIELDS[k] for k in desc.CARD],
        "chain_card": [desc.FIELDS[k] for k in desc.CHAIN_CARD],
        "arms": designs(models, config),
        "models": {a: net.state_dict() for a, net in models.items()},
    }
    if opts is not None:
        state["optimizers"] = {a: opt.state_dict() for a, opt in opts.items()}
        state["schedulers"] = {a: s.state_dict() for a, s in scheds.items()}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def training_data(args) -> tuple[loader.TrainingWindows, dict]:
    """The Sentinel-2 training windows (steps x batch of them) and the windows of every other
    satellite with a share of the batches (None for the others), as args says."""
    s2 = args.s2 or args.data / "chips" / "cloudsen12plus"
    shares = dict(zip(SOURCES, args.shares, strict=True))
    data = {"landsat8": None, "heo": None, "perusat1": None}
    if shares["landsat8"] > 0:
        l8 = args.biome or args.data / "chips" / "l8_biome"
        held = {s.name for s in biome.scenes(l8, args.hold_out)}
        chosen = [s for s in biome.scenes(l8, "index") if s.name not in held]
        data["landsat8"] = biome.Windows(chosen, size=args.landsat_size)
    if shares["heo"] > 0:
        data["heo"] = heo.Windows(heo.scenes(args.data, ("train",)), size=args.heo_size)
    if shares["perusat1"] > 0:
        root = args.perusat or args.data / "chips" / "cloudperu2"
        data["perusat1"] = perusat.Windows(root, "train", size=args.perusat_size)
    windows = loader.TrainingWindows(
        s2, "train", size=args.size, length=args.steps * args.batch, seed=args.seed
    )
    return windows, data


def breakdown(args, windows, data: dict) -> list[dict]:
    """What each satellite brings: the rows of the table printed at the start."""
    shares = dict(zip(SOURCES, np.asarray(args.shares) / float(np.sum(args.shares)), strict=True))
    rows = [
        {
            "source": "sentinel2",
            "data": f"CloudSEN12+ train split, {len(windows.patches):,} patches",
            "pixel_m": 10.0,
            "bands": len(loader.BANDS),
            "labels": "clear, thin, thick, shadow",
        }
    ]
    if data["landsat8"] is not None:
        n = len(data["landsat8"].scenes)
        rows.append(
            {
                "source": "landsat8",
                "data": f"L8 Biome, {n} scenes (table {args.hold_out} kept out)",
                "pixel_m": biome.GRID_M,
                "bands": len(biome.BANDS),
                "labels": "clear, thin, thick, shadow (some scenes: no shadow)",
            }
        )
    if data["heo"] is not None:
        n = len(data["heo"].scenes)
        rows.append(
            {
                "source": "heo",
                "data": f"IRIDE HEO, {n} train scenes, our labels",
                "pixel_m": heo.GRID_M,
                "bands": len(heo.BANDS),
                "labels": "clear, thin, thick, shadow",
            }
        )
    if data["perusat1"] is not None:
        n = len(data["perusat1"])
        rows.append(
            {
                "source": "perusat1",
                "data": f"CloudPeru2 train split, {n:,} patches",
                "pixel_m": perusat.GRID_M,
                "bands": len(perusat.BANDS),
                "labels": "cloud, not cloud",
            }
        )
    for row in rows:
        row["share"] = round(float(shares[row["source"]]), 4)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="a folder for this run (one seed)")
    p.add_argument("--data", type=Path, default=Path("/root/catena_data"))
    p.add_argument("--s2", type=Path, help="CloudSEN12+ (default DATA/chips/cloudsen12plus)")
    p.add_argument("--biome", type=Path, help="L8 Biome (default DATA/chips/l8_biome)")
    p.add_argument("--perusat", type=Path, help="CloudPeru2 (default DATA/chips/cloudperu2)")
    p.add_argument("--hold-out", default="pilot", help="the Biome table kept out of training")
    p.add_argument(
        "--arms",
        nargs=2,
        default=list(VERSIONS),
        choices=list(ARMS),
        help="the two arms, one per device (default: real sim)",
    )
    p.add_argument(
        "--shares",
        type=float,
        nargs=4,
        default=list(SHARES),
        metavar=("S2", "L8", "HEO", "PERUSAT"),
        help="batches from Sentinel-2, Landsat 8, HEO and PeruSat-1",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=loader.SIZE, help="Sentinel-2 windows, 10 m px")
    p.add_argument("--landsat-size", type=int, default=80, help="Landsat 8 windows, 30 m px")
    p.add_argument("--heo-size", type=int, default=240, help="HEO windows, 5 m px")
    p.add_argument("--perusat-size", type=int, default=240, help="PeruSat-1 windows, 2.8 m px")
    p.add_argument("--workers", type=int, default=6, help="data-loading processes")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--p-s2", type=float, default=0.6, help="sim: 10 m samples kept as delivered")
    p.add_argument(
        "--q-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("W10", "W30", "W50"),
        help="sim: how often its Sentinel-2 samples get 10, 30 and 50 m pixels",
    )
    p.add_argument("--faults", type=float, default=0.0, help="sim: share of samples with a fault")
    p.add_argument("--fault-gain", type=float, default=0.3, help="a fault's gains: within 1+-x")
    p.add_argument("--fault-nir", type=float, default=0.4, help="a fault's near infrared: up to")
    p.add_argument(
        "--chain-jitter", type=float, default=1.0, help="chain: errors of the MTF and SNR, 0-1"
    )
    p.add_argument(
        "--coarser", type=float, default=0.0, help="share of real satellites' batches coarser"
    )
    p.add_argument("--hailo", action="store_true", help="networks made for the Hailo-8")
    p.add_argument(
        "--freeze-norms", type=float, default=0.9, help="--hailo: BatchNorm frozen at this share"
    )
    p.add_argument("--trunk", nargs="+", help="nas, nas2: their architecture codes (catena.nas)")
    p.add_argument("--widths", type=int, nargs=4, default=[32, 64, 128, 256])
    p.add_argument("--stem", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-windows", type=int, default=240, help="windows per validation camera")
    p.add_argument("--devices", nargs=2, help="the two arms' devices, in --arms order")
    p.add_argument("--no-amp", action="store_true", help="float32 sums instead of bf16")
    p.add_argument("--stop-after", type=int, help="stop at this step and keep last.pt")
    args = p.parse_args(argv)
    if args.size % 120:
        p.error("--size must be a multiple of 120: divisible by 1, 3 and 5, then by 8")
    for name in ("landsat_size", "heo_size", "perusat_size"):
        if getattr(args, name) <= 0 or getattr(args, name) % 8:
            p.error(f"--{name.replace('_', '-')} must be a multiple of 8")
    if min(args.shares) < 0 or sum(args.shares) <= 0 or args.shares[0] <= 0:
        p.error("--shares: none below 0, and Sentinel-2 above 0 (it carries the validation)")
    if min(args.q_weights) < 0 or sum(args.q_weights) <= 0:
        p.error("--q-weights: none below 0, not all 0")
    if not 0 <= args.faults <= 1 or args.fault_gain < 0 or not 0 <= args.fault_nir < 1:
        p.error("--faults within 0 and 1, --fault-gain from 0, --fault-nir within 0 and 1")
    if args.arms[0] == args.arms[1]:
        p.error("--arms: two different arms")
    if "fixed" in args.arms and args.faults > 0:
        p.error("--faults: the fixed arm takes none")
    if args.chain_jitter < 0:
        p.error("--chain-jitter: 0 or more")
    if not 0 <= args.coarser <= 1:
        p.error("--coarser within 0 and 1")
    if not 0 < args.freeze_norms <= 1:
        p.error("--freeze-norms above 0, at most 1")
    if args.coarser > 0 and any(
        getattr(args, k) % 16 for k in ("landsat_size", "heo_size", "perusat_size")
    ):
        p.error("--coarser: --landsat-size, --heo-size and --perusat-size multiples of 16")
    if len(args.trunk or []) != sum(a in NAS for a in args.arms):
        p.error("--trunk: one architecture code for each nas arm, none without")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.out / "final.pt").exists():
        raise SystemExit(f"{args.out} already holds a finished run: choose a new folder")
    args.out.mkdir(parents=True, exist_ok=True)
    s2 = args.s2 or args.data / "chips" / "cloudsen12plus"
    arms = tuple(args.arms)
    devices = dict(zip(arms, T.pick_devices(args.devices), strict=True))
    amp = {
        v: dev.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
        for v, dev in devices.items()
    }
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    trunks = codes_of(arms, args.trunk)
    models = {}
    for a in arms:
        torch.manual_seed(args.seed)  # arms of one design start from the same weights
        net = build(a, args.widths, args.stem, trunks.get(a), args.hailo)
        models[a] = net.to(devices[a]).train()
    opts = {
        v: torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for v, net in models.items()
    }
    scheds = {
        v: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: T.lr_factor(s, args.warmup, args.steps))
        for v, opt in opts.items()
    }
    weights = {a: T.CODE_WEIGHTS.to(dev) for a, dev in devices.items()}
    windows, data = training_data(args)
    rows = breakdown(args, windows, data)

    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    weights_q = np.asarray(args.q_weights, np.float64) / float(np.sum(args.q_weights))
    at_10m = float(weights_q[sim.Q_CHOICES.index(1)])
    step, last = 0, args.out / "last.pt"
    if last.exists():  # going on after a stop
        state = torch.load(last, map_location="cpu")
        before = state["config"]["args"]
        changed = [k for k in SAME_ARGS if before.get(k, OLDER.get(k)) != arguments.get(k)]
        if changed:
            raise SystemExit(f"{last} was made with other settings: {', '.join(changed)}")
        for a in arms:
            models[a].load_state_dict(state["models"][a])
            opts[a].load_state_dict(state["optimizers"][a])
            scheds[a].load_state_dict(state["schedulers"][a])
        step = int(state["step"])
        config = state["config"]
        config.setdefault("resumed", []).append(
            {"step": step, "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        )
    else:
        config = {
            "args": arguments,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": T.git_state(),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "host": platform.node(),
            "devices": {
                v: torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
                for v, dev in devices.items()
            },
            "bf16": amp,
            "parameters": {a: count_parameters(net) for a, net in models.items()},
            "trunks": trunks,
            "arms": designs(arms, {"trunks": trunks}),
            "card": [desc.FIELDS[k] for k in desc.CARD],
            "chain_card": [desc.FIELDS[k] for k in desc.CHAIN_CARD],
            "sources": rows,
            "made_up_cameras": {
                "version": "sim",
                "share_of_sentinel2": round(1 - args.p_s2 * at_10m, 4),
                "pixel_m": [sim.GRID_M * q for q in sim.Q_CHOICES],
                "pixel_shares": [round(float(w), 4) for w in weights_q],
                "saturation": list(TOPS),
            },
            "calibration_faults": {
                "version": "sim",
                "share_of_samples": args.faults,
                "gain_within": args.fault_gain,
                "nir_share_up_to": args.fault_nir,
                "bands": {k: {"visible": list(v), "nir": n} for k, (v, n) in NIR_LEAK.items()},
            },
            "coarser_pixels": {
                "share_of_batches": args.coarser,
                "factors": {k: list(v) for k, v in COARSER.items()},
            },
            "hailo": {
                "made_for_hailo8": args.hailo,
                "layers": "BatchNorm, ReLU, nearest neighbours" if args.hailo else None,
                "norms_averaged_from_step": norm_steps(args)[0] if args.hailo else None,
                "norms_frozen_from_step": norm_steps(args)[1] if args.hailo else None,
            },
            "landsat_scenes": [s.name for s in data["landsat8"].scenes] if data["landsat8"] else [],
            "heo_scenes": [s.name for s in data["heo"].scenes] if data["heo"] else [],
            "validation_cameras": [c[0] for c in T.VAL_CAMERAS],
            "code_weights": T.CODE_WEIGHTS.tolist(),
        }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    train_data = Samples(
        windows,
        args.batch,
        args.seed,
        landsat=data["landsat8"],
        heo_windows=data["heo"],
        peru=data["perusat1"],
        shares=args.shares,
        p_s2=args.p_s2,
        start=step * args.batch,
        q_weights=args.q_weights,
        faults=args.faults,
        fault_gain=args.fault_gain,
        fault_nir=args.fault_nir,
        fixed="fixed" in arms,
        chain_jitter=args.chain_jitter,
        coarser=args.coarser,
    )
    val_data = T.ValSamples(loader.GridWindows(s2, "validation", size=args.size), args.val_windows)
    pin = any(dev.type == "cuda" for dev in devices.values())
    more = {"persistent_workers": True, "prefetch_factor": 4} if args.workers else {}
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=pin,
        worker_init_fn=T.quiet,
        **more,
    )
    val_loader = DataLoader(
        val_data,
        batch_sampler=val_data.batches(args.batch),
        num_workers=args.workers,
        pin_memory=pin,
        worker_init_fn=T.quiet,
    )

    print(f"two arms, git {config['git']}", flush=True)
    for a in arms:
        n = config["parameters"][a]
        net = "FixedBands" if ARMS[a].fields == 0 else f"Clarity reading {ARMS[a].fields} numbers"
        net = f"band encoder and PyNAS trunk {trunks[a]}" if a in trunks else net
        net += ", told nothing" if ARMS[a].blind else ""
        print(
            f"  {a:<12}{net}; {ARMS[a].data}'s samples; {n:,} parameters, "
            f"{2 * n / 1e6:.1f} MB at float16; on {devices[a]}",
            flush=True,
        )
    print(f"  {'source':<10}{'data':<52}{'pixel':>7}{'bands':>7}{'share':>7}  labels")
    for r in rows:
        print(
            f"  {r['source']:<10}{r['data']:<52}{r['pixel_m']:>5g} m{r['bands']:>7}"
            f"{r['share']:>7.0%}  {r['labels']}"
        )
    pixel = "/".join(f"{w:.0%}" for w in weights_q)
    if any(ARMS[a].data == "sim" for a in arms):
        share = config["made_up_cameras"]["share_of_sentinel2"]
        print(
            f"  made-up cameras: sim only, {share:.0%} of the Sentinel-2 samples "
            f"(pixels 10/30/50 m: {pixel}), saturation {TOPS[0]}-{TOPS[1]}",
            flush=True,
        )
    else:
        print("  made-up cameras: none, both arms learn from the samples as delivered", flush=True)
    if args.faults > 0:
        print(
            f"  calibration faults: sim only, {args.faults:.0%} of every satellite's samples: "
            f"up to {args.fault_nir:.0%} near infrared in the visible bands, gains within "
            f"+-{args.fault_gain:.0%}; not on the card",
            flush=True,
        )
    if args.coarser > 0:
        print(
            f"  coarser pixels: every arm, {args.coarser:.0%} of the batches from Landsat 8 "
            "(x2), HEO and PeruSat-1 (x2 or x3), the card told the new pixel",
            flush=True,
        )
    norms = norm_steps(args)
    if norms:
        average, frozen = norms
        when = f"frozen from step {frozen:,}" if frozen < args.steps else "never frozen"
        print(
            f"  made for the Hailo-8: BatchNorm, ReLU, nearest-neighbour upsampling; BatchNorm's "
            f"statistics a plain average from step {average:,}, {when}",
            flush=True,
        )
        if step >= frozen:  # going on after the statistics were frozen
            for net in models.values():
                freeze_norms(net)
        elif step > average:  # going on while they were being averaged
            for net in models.values():
                average_norms(net, restart=False)
    if step:
        print(f"going on from step {step:,} (last.pt)", flush=True)

    log = open(args.out / "metrics.jsonl", "a", buffering=1)  # one line at a time
    running = {v: torch.zeros((), device=dev) for v, dev in devices.items()}
    counts = Counter()
    began, first_step, logged = time.perf_counter(), step, step
    for batch in train_loader:
        if norms and step == norms[0]:  # BatchNorm's statistics: a plain average from here
            for net in models.values():
                average_norms(net)
        if norms and step == norms[1]:  # and frozen from here, as they fly
            for net in models.values():
                freeze_norms(net)
        counts[SOURCES[int(batch["source"][0])]] += 1
        for a in arms:
            x, told, real, label = arm_inputs(batch, a)
            logits = forward(models[a], x, told, real, devices[a], amp[a])
            loss = T.set_loss(logits, label.to(devices[a], non_blocking=True), weights[a])
            opts[a].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[a].parameters(), args.clip)
            opts[a].step()
            scheds[a].step()
            running[a] += loss.detach()
        step += 1

        if step % args.log_every == 0 or step == args.steps:
            losses = {a: running[a].item() / (step - logged) for a in arms}
            for value in running.values():
                value.zero_()
            logged = step
            if not all(math.isfinite(x) for x in losses.values()):
                raise SystemExit(f"step {step}: the loss is not finite ({losses})")
            per_step = (time.perf_counter() - began) / (step - first_step)
            memory = {
                v: round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
                for v, dev in devices.items()
                if dev.type == "cuda"
            }
            lr = scheds[arms[0]].get_last_lr()[0]
            entry = {"kind": "train", "step": step, "lr": lr, "s_per_step": round(per_step, 4)}
            entry |= {f"loss_{a}": round(x, 5) for a, x in losses.items()}
            entry |= {"batches": dict(counts), "gpu_gb": memory}
            log.write(json.dumps(entry) + "\n")
            said = " ".join(f"{a} {x:.4f}" for a, x in losses.items())
            print(
                f"step {step:,}/{args.steps:,}  loss {said}  lr {lr:.2e}  {per_step:.2f} s/step  "
                f"{(args.steps - step) * per_step / 3600:.1f} h to go  GPU GB {memory}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            started = time.perf_counter()
            conf = validate(models, devices, val_loader, amp)
            for a in arms:
                per_camera = [T.scores(conf[a][c]) for c in range(len(T.VAL_CAMERAS))]
                for (name, *_), s in zip(T.VAL_CAMERAS, per_camera, strict=True):
                    entry = {"kind": "val", "step": step, "version": a, "camera": name}
                    log.write(json.dumps(entry | s) + "\n")
                mean = {
                    k: round(float(np.mean([s[k] for s in per_camera])), 4) for k in per_camera[0]
                }
                entry = {"kind": "val", "step": step, "version": a, "camera": "mean"}
                log.write(json.dumps(entry | mean) + "\n")
                print(
                    f"   validation {a:<11}  F1 clear {mean['f1_clear']:.3f} thin "
                    f"{mean['f1_thin']:.3f} thick {mean['f1_thick']:.3f} shadow "
                    f"{mean['f1_shadow']:.3f}  cloud BA {mean['cloud_ba']:.3f}",
                    flush=True,
                )
            save(last, step, models, config, opts, scheds)
            print(
                f"   validation and checkpoint: {time.perf_counter() - started:.0f} s", flush=True
            )

        if step >= args.steps:
            break
        if args.stop_after is not None and step >= args.stop_after:
            save(last, step, models, config, opts, scheds)
            log.close()
            print(f"stopped at step {step:,}; the same command goes on from here", flush=True)
            return

    save(args.out / "final.pt", step, models, config)
    log.close()
    hours = (time.perf_counter() - began) / 3600
    print(f"done: {step:,} steps ({step - first_step:,} now, {hours:.2f} h), in {args.out}")


if __name__ == "__main__":
    main()
