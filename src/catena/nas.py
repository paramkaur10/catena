"""The architecture search behind the descriptor head: PyNAS (py-q-nas) draws U-Net trunks,
crosses and mutates them; every candidate is a band encoder in front of a PyNAS trunk, trained
on the big run's samples and scored on the validation cameras only.

    python -m catena.nas OUT --head RUN/seed0/final.pt [--arm sim] [--data /root/catena_data]
        [--population 16] [--generations 8] [--steps 3000] [--budget-mb 6] [--max-layers 4]
        [--device cuda:0] [--seed 0]

The head: the band encoder of a finished run's arm (--head, --arm; it reads the card), kept as
it is during the search, so that every trunk is trained on the same features and quickly. The
trunk: PyNAS's GenericUNetNetwork on the head's 2 x stem channels (32), four classes out, with
two changes for this data:
  - GroupNorm for its BatchNorm: a batch comes from one satellite, so batch statistics would
    differ by satellite between training and use (Clarity has GroupNorm for the same reason);
  - any window size: PyNAS sizes its decoder for one input; here every candidate is built,
    and so checked, at the smallest window the training draws (48 px: Sentinel-2 at 50 m),
    each decoder stage is brought to its skip's size, and the output to the window's.
A trunk made for the Hailo-8 (build_trunk hailo=True, train_clarity --hailo) keeps PyNAS's
BatchNorm, takes ReLU wherever its code says GELU, and brings each decoder stage to its skip's
size by nearest neighbours, not bilinear: only layers the chip's compiler takes as they are.
PyNAS reads Catena's copy of its config (catena/pynas.ini, through PYNAS_CONFIG): a layer
keeps or doubles the channels of the one before, as a U-Net's levels do.

PyNAS's Population runs the search (random first population, single-point crossover, gene
mutation, the best kept, a few random newcomers, checkpoints); this module gives it its
candidates (build_model) and its fitness (run_precision_suite), and two repairs: crossover on
copies of the parents, both children crossed (PyNAS's changes the first parent in place, and
its second child comes out as the second parent), and all newcomers when the parents to mate
from are all alike (PyNAS would draw pairs forever). A candidate's size counts the
head and the trunk, and one at or above --budget-mb at float16 (2 bytes a parameter) is never
built twice: PyNAS draws another. Each candidate trains for --steps steps of --batch windows
on the sim version's samples of seed --seed (catena.train_clarity), the same samples for every
candidate, from the same start (torch seed --seed), with AdamW, a warm-up and a cosine decay,
and the partial-label loss. Fitness: the mean F1 of the four classes on the 6 validation
cameras (catena.train.VAL_CAMERAS: Sentinel-2 as delivered and 5 made-up cameras, on windows
of the CloudSEN12+ validation split), averaged over the cameras. No test set is ever read.

OUT receives config.json; search.jsonl, one line per trained candidate (generation, index,
architecture code, parameters and megabytes, fitness, the scores per camera, milliseconds per
240 px tile, seconds); PyNAS's checkpoints (src/population_<g>.pkl and df_population_<g>.pkl,
backups/) and population.log. Run again with the same OUT, it goes on from the last
population saved, without training again the candidates already scored. At the end it prints
the best trunks and the command that trains the best two end to end (train_clarity's nas arms).

A candidate that fails (out of memory, an error) scores 0 and the search goes on. A fault on
the GPU (CUDA) is different: it breaks the process's hold on the GPU for good, or takes the
process down at once, so the process stops (exit 3, or the fault's own). trying.json, written
before each candidate trains, names the one it was on, and how many times it was started.
Started again (scripts/search.sh does so), the search trains that candidate again, one level
of caution up (LEVELS), with every GPU kernel launched one at a time (CUDA_LAUNCH_BLOCKING=1),
so a fault that comes back is raised at the very operation that made it and its traceback
goes to the log:
  0  as fast as it goes: bfloat16, pinned data, fused attention kernels;
  1  float32, data not pinned, attention by its formula (no fused kernel, no fast path);
  2  as 1, and convolutions without cuDNN.
A candidate trained so is scored as any other (its line says how: rescued, caution); the
process then stops (exit 4) and is started again without the slow launches, and from then on
the search trains every candidate at the level that worked (caution.json). A candidate that
stopped the process at the last level too, kernels one at a time, scores 0, is left out and
is never trained again; each failed candidate gets a line in search.jsonl with its reason
(failed). If GIVE_UP candidates are left out so and none has been scored, the fault is in what
they all share, not in their trunks: the search gives up (exit 5), so a GPU fault costs
minutes, not the days of a search. A search stopped by hand (tmux closed, kill, Ctrl-C) trains
the candidate it was on afresh when started again, at the level it had.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import logging
import math
import os
import platform
import random
import signal
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from catena import descriptor as desc
from catena import sim
from catena import train as T
from catena import train_clarity as C
from catena.data import loader
from catena.model import BandEncoder, count_parameters

CONFIG = Path(__file__).with_name("pynas.ini")
TRYING = "trying.json"  # the candidate being trained, while it is
CAUTION = "caution.json"  # the level of caution the search has needed, once it needed one
LEVELS = (
    "bfloat16, pinned data, fused attention",
    "float32, data not pinned, attention by its formula",
    "float32, data not pinned, attention by its formula, convolutions without cuDNN",
)
LEFT_OUT = "it took the process down (a fault on the GPU)"
GIVE_UP = 3  # this many left out so, and none scored: the fault is not the trunks'
CLASSES = 4
BYTES = 2  # a parameter at float16
TILE = 240  # the speed is measured on tiles of this size


_PYNAS: list[SimpleNamespace] = []


def pynas() -> SimpleNamespace:
    """PyNAS's modules, set to read Catena's search space (catena/pynas.ini)."""
    if not _PYNAS:
        os.environ["PYNAS_CONFIG"] = str(CONFIG)
        from pynas.core import architecture_builder, config, generic_unet, population

        architecture_builder._load_config.cache_clear()  # in case PyNAS read its own already
        generic_unet._load_config.cache_clear()
        population.single_point_crossover = crossover
        _PYNAS.append(
            SimpleNamespace(
                builder=architecture_builder,
                unet=generic_unet,
                population=population,
                config=config,
            )
        )
    return _PYNAS[0]


def crossover(parents, verbose: bool = False) -> list:
    """Single-point crossover, as PyNAS means it: two children, each the head of one parent's
    genes and the tail of the other's, made from copies, so the parents stay as they were."""
    first, second = (copy.deepcopy(p) for p in parents)
    a, b = list(first.chromosome), list(second.chromosome)
    cut = random.randint(1, min(len(a), len(b)) - 2)
    first.chromosome, second.chromosome = a[:cut] + b[cut:], b[:cut] + a[cut:]
    for child in (first, second):
        child._reparse_layers()
    if verbose:
        print(f"cut at {cut}: {first.chromosome} / {second.chromosome}")
    return [first, second]


# --- the candidates -------------------------------------------------------------------------


def any_size(net: nn.Module) -> nn.Module:
    """PyNAS's U-Net without the sizes it was built for: each decoder stage leaves its map as
    it is and the decoder brings it to its skip's size (as it does when they differ), and the
    output stays at the first skip's size (Candidate brings it to the window's)."""
    decoder = net.decoder
    for up in decoder.up_convs:
        if isinstance(up[0], nn.Upsample):
            up[0] = nn.Identity()
    if isinstance(decoder.out_conv[-1], nn.Upsample):
        decoder.out_conv[-1] = nn.Identity()
    decoder._skip_shape_guard_for_fx = False
    return net


def group_norms(module: nn.Module) -> nn.Module:
    """Every BatchNorm2d of a module replaced by a GroupNorm of its channels: 8 groups, or the
    most that divide them."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            setattr(module, name, nn.GroupNorm(math.gcd(channels, 8), channels))
        else:
            group_norms(child)
    return module


def nearest_forward(self, encoder_features, verbose=False):
    """PyNAS's UNetDecoder.forward with each stage brought to its skip's size by nearest
    neighbours (a decoder made for the Hailo-8, set by nearest())."""
    if len(encoder_features) != self.num_stages + 1:
        raise ValueError(f"expected {self.num_stages + 1} encoder features")
    x = encoder_features[-1]  # the bottleneck
    for i in range(self.num_stages):
        skip = encoder_features[self.num_stages - 1 - i]
        x = self.up_convs[i](x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.conv_blocks[i](torch.cat([x, skip], dim=1))
    return self.out_conv(x)


def nearest(net: nn.Module) -> nn.Module:
    """A trunk made for any size (any_size) whose decoder upsamples by nearest neighbours."""
    net.decoder.forward = MethodType(nearest_forward, net.decoder)
    return net


def build_trunk(code_or_layers, channels: int, size: int = 48, hailo: bool = False) -> nn.Module:
    """PyNAS's U-Net for an architecture code (or its parsed layers) on `channels` channels,
    four classes out: built, and so checked, at size x size pixels, then made for any size,
    with GroupNorm; or, made for the Hailo-8 (hailo), with PyNAS's BatchNorm, ReLU for GELU
    and nearest-neighbour upsampling."""
    p = pynas()
    layers = code_or_layers
    if isinstance(code_or_layers, str):
        layers = p.builder.parse_architecture_code(code_or_layers)
    if hailo:
        layers = [
            {**layer, "activation": "ReLU"} if layer.get("activation") == "GELU" else layer
            for layer in layers
        ]
    net = p.unet.GenericUNetNetwork(
        layers, input_channels=channels, input_height=size, input_width=size, num_classes=CLASSES
    )
    if hailo:
        return nearest(any_size(net))
    return group_norms(any_size(net))


class Candidate(nn.Module):
    """A band encoder in front of a trunk: any bands with their chain cards in, the four
    classes out, at the window's size. fixed: the band encoder's weights stay as they are.
    hailo: the output brought to the window's size by nearest neighbours."""

    def __init__(
        self, head: BandEncoder, trunk: nn.Module, fixed: bool = False, hailo: bool = False
    ):
        super().__init__()
        self.head, self.trunk, self.fixed, self.hailo = head, trunk, fixed, hailo
        if fixed:
            self.head.requires_grad_(False)

    def train(self, mode: bool = True):
        """A fixed band encoder stays in eval mode while the trunk trains, as a frozen
        encoder should: its features as it gives them in use (no dropout)."""
        super().train(mode)
        if self.fixed:
            self.head.eval()
        return self

    def forward(self, x, told, real):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.fixed):
            features, _ = self.head(x, told, real)
        out = self.trunk(features)
        if out.shape[-2:] != x.shape[-2:] and self.hailo:
            out = F.interpolate(out, size=x.shape[-2:], mode="nearest")
        elif out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def load_head(path: Path, arm: str = "sim") -> BandEncoder:
    """The band encoder of a finished run's arm, with its weights."""
    state = torch.load(path, map_location="cpu")
    design = (state.get("arms") or {}).get(arm) or {"fields": desc.N_CARD}
    if arm not in state["models"] or int(design["fields"]) == 0 or design.get("blind"):
        raise SystemExit(f"{path}: no {arm} arm with a band encoder that reads cards")
    weights = {
        k.removeprefix("bands."): v
        for k, v in state["models"][arm].items()
        if k.startswith("bands.")
    }
    head = BandEncoder(state["config"]["args"]["stem"], int(design["fields"]))
    head.load_state_dict(weights)
    return head


def smallest(args) -> int:
    """The smallest window the training draws, in pixels: Sentinel-2 at its largest pixel
    size, or another satellite's window."""
    sizes = [args.size // max(sim.Q_CHOICES)]
    for name, size in zip(
        C.SOURCES[1:], (args.landsat_size, args.heo_size, args.perusat_size), strict=True
    ):
        if args.shares[C.SOURCES.index(name)] > 0:
            sizes.append(size)
    return min(sizes)


# --- the fitness ----------------------------------------------------------------------------


class Trainer:
    """Trains a candidate on the sim version's samples and scores it on the validation
    cameras; every candidate gets the same samples. Writes a line of search.jsonl for each.
    level: of caution (LEVELS); from 1, float32 and data not pinned (careful() sets the rest
    for the whole process)."""

    def __init__(self, args, device: torch.device, record: Path, level: int = 0):
        self.args, self.device, self.record, self.level = args, device, record, level
        cuda = device.type == "cuda" and level == 0
        self.amp = cuda and not args.no_amp and torch.cuda.is_bf16_supported()
        self.note = None  # said in the line of a candidate trained again after a fault
        self.weights = T.CODE_WEIGHTS.to(device)
        windows, data = C.training_data(args)
        samples = C.Samples(
            windows,
            args.batch,
            args.seed,
            landsat=data["landsat8"],
            heo_windows=data["heo"],
            peru=data["perusat1"],
            shares=args.shares,
            p_s2=args.p_s2,
        )
        s2 = args.s2 or args.data / "chips" / "cloudsen12plus"
        checks = T.ValSamples(
            loader.GridWindows(s2, "validation", size=args.size), args.val_windows
        )
        pin = cuda
        more = {"persistent_workers": True, "prefetch_factor": 4} if args.workers else {}
        self.train = DataLoader(
            samples,
            batch_size=args.batch,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=pin,
            worker_init_fn=T.quiet,
            **more,
        )
        self.checks = DataLoader(
            checks,
            batch_sampler=checks.batches(args.batch),
            num_workers=args.workers,
            pin_memory=pin,
            worker_init_fn=T.quiet,
        )

    def failed(self, generation: int, index: int, code: str, net: nn.Module, why: str) -> dict:
        """A candidate that could not be trained: fitness 0, a line in search.jsonl."""
        parameters = count_parameters(net)
        result = {"generation": generation, "index": index, "code": code, "steps": 0}
        result |= {"parameters": parameters, "mb": round(BYTES * parameters / 1e6, 3)}
        result |= {"fitness": 0.0, "ms_per_tile": None, "per_camera": {}, "failed": why}
        with open(self.record, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"  gen {generation} #{index}: {why}; scored 0", flush=True)
        return result

    @torch.no_grad()
    def speed(self, net: nn.Module, bands: int = 4, batch: int = 8) -> float:
        """Milliseconds per 240 px tile of `bands` bands, on the device, after a warm-up."""
        net.eval()
        x = torch.rand(batch, bands, TILE, TILE, device=self.device)
        told = torch.zeros(batch, bands, desc.N_CHAIN_CARD, device=self.device)
        real = torch.ones(batch, bands, dtype=torch.bool, device=self.device)
        timed = []
        for k in range(6):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            began = time.perf_counter()
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                net(x, told, real)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if k >= 2:
                timed.append(time.perf_counter() - began)
        net.train()
        return 1000 * float(np.median(timed)) / batch

    def __call__(self, net: nn.Module, generation: int, index: int, code: str) -> dict:
        args, device = self.args, self.device
        began = time.perf_counter()
        net = net.to(device).train()
        trained = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trained, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: T.lr_factor(s, args.warmup, args.steps)
        )
        losses, step = [], 0
        for batch in self.train:
            x, told, real, label = C.arm_inputs(batch, "sim")
            logits = C.forward(net, x, told, real, device, self.amp)
            loss = T.set_loss(logits, label.to(device, non_blocking=True), self.weights)
            if not torch.isfinite(loss):
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, args.clip)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if step >= args.steps:
                break
        result = {"generation": generation, "index": index, "code": code, "steps": step}
        parameters = count_parameters(net)
        result |= {"parameters": parameters, "mb": round(BYTES * parameters / 1e6, 3)}
        if step < args.steps:  # the loss went beyond numbers: no fitness
            per_camera, fitness = {}, 0.0
        else:
            conf = C.validate({"sim": net}, {"sim": device}, self.checks, {"sim": self.amp})["sim"]
            per_camera = {name: T.scores(conf[c]) for c, (name, *_) in enumerate(T.VAL_CAMERAS)}
            fitness = float(np.nanmean([s["f1_mean"] for s in per_camera.values()]))
            fitness = fitness if math.isfinite(fitness) else 0.0
        mean = {
            k: round(float(np.nanmean([s[k] for s in per_camera.values()])), 4)
            for k in ("f1_mean", "cloud_ba", "f1_thin", "f1_thick", "f1_shadow")
            if per_camera
        }
        result |= {"fitness": round(fitness, 4)} | mean
        result["loss"] = round(float(np.mean(losses[-100:])), 4) if losses else None
        result["ms_per_tile"] = round(self.speed(net, batch=8 if device.type == "cuda" else 1), 3)
        result["seconds"] = round(time.perf_counter() - began, 1)
        if self.level:
            result["caution"] = LEVELS[self.level]
        if self.note:
            result["rescued"] = self.note
        result["per_camera"] = {
            name: {k: round(v, 4) for k, v in s.items() if math.isfinite(v)}
            for name, s in per_camera.items()
        }
        with open(self.record, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(
            f"  gen {generation} #{index}: fitness {fitness:.4f}, cloud BA "
            f"{mean.get('cloud_ba', float('nan')):.4f}, {parameters:,} parameters "
            f"({result['mb']} MB), {result['ms_per_tile']} ms/tile, {result['seconds']:.0f} s",
            flush=True,
        )
        return result


# --- the search -----------------------------------------------------------------------------


def search_class():
    """PyNAS's Population, given Catena's candidates and fitness."""
    found = pynas()
    population = found.population

    class Search(population.Population):
        """The population of trunks behind one head."""

        def __init__(self, out: Path, head: BandEncoder, trainer, args, size: int):
            self.head, self.trainer, self.args, self.size = head, trainer, args, size
            channels = 2 * head.stem1.out_channels
            dm = SimpleNamespace(input_shape=(channels, size, size), num_classes=CLASSES)
            budget = int(args.budget_mb * 1e6 / BYTES)
            super().__init__(
                n_individuals=args.population,
                max_layers=args.max_layers,
                dm=dm,
                max_parameters=budget,
                save_directory=str(out),
            )
            self.cfg = found.config.load_config(CONFIG)  # what it prints: fp32, as trained here
            self.retry = False  # main(): this process trains again the one the last stopped on

        def setup_logger(self, *_args, **_kwargs):
            """PyNAS's log, in OUT (PyNAS writes ../logs/population.log)."""
            logger = logging.getLogger("catena.nas")
            logger.setLevel(logging.INFO)
            path = Path(self.save_directory) / "population.log"
            if not any(getattr(h, "baseFilename", None) == str(path) for h in logger.handlers):
                handler = logging.FileHandler(path)
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                logger.addHandler(handler)
            return logger

        def evolve(self, mating_pool_cutoff=0.5, mutation_probability=0.2, k_best=1, n_random=1):
            """PyNAS's step to the next generation; all newcomers when the parents to mate
            from are all alike."""
            ranked = sorted(self.population, key=lambda i: i.fitness, reverse=True)
            pool = ranked[: int(np.floor(mating_pool_cutoff * self.n_individuals))]
            if len({json.dumps(i.parsed_layers, default=str) for i in pool}) < 2:
                n_random = self.n_individuals - k_best
            return super().evolve(mating_pool_cutoff, mutation_probability, k_best, n_random)

        def build_model(self, parsed_layers, task: str = "segmentation"):
            trunk = build_trunk(parsed_layers, self.dm.input_shape[0], self.size)
            return Candidate(self.head, trunk, fixed=True), True

        def run_precision_suite(self, idx, task="segmentation", epochs=0, lr=0.0, batch_size=None):
            individual = self.population[idx]
            code = individual.chromosome2architecture(individual.chromosome)
            individual.architecture = code  # PyNAS leaves it as first drawn
            torch.manual_seed(self.args.seed)  # every trunk from the same start
            net, _ = self.build_model(individual.parsed_layers)
            out = Path(self.save_directory)
            marker = out / TRYING
            before = json.loads(marker.read_text()) if marker.exists() else {}
            again = (before.get("generation"), before.get("index")) == (self.generation, idx)
            tries = 1 + int(before.get("tries", 1)) if again else 1  # the last run stopped on it
            trying = {"generation": self.generation, "index": idx, "code": code, "tries": tries}
            marker.write_text(json.dumps(trying))
            retry = again and self.retry  # this process was started to train it again
            if retry:
                faults = "a fault" if tries == 2 else f"{tries - 1} faults"
                self.trainer.note = f"trained again after {faults}, kernels one at a time"
            why = None
            try:
                result = self.trainer(net, self.generation, idx, code)
            except torch.OutOfMemoryError:
                why = "out of memory"
            except Exception as error:
                if "CUDA" in str(error) or type(error).__name__ == "AcceleratorError":
                    print(
                        f"  gen {self.generation} #{idx}: a fault on the GPU: {error}", flush=True
                    )
                    traceback.print_exc()  # where it surfaced, for the log
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(3)  # the process cannot use the GPU again; trying.json stays
                why = f"{type(error).__name__}: {error}"[:300]
            if why:  # out of the except clause, the error has let go of its tensors
                net.to("cpu")
                torch.cuda.empty_cache()
                result = self.trainer.failed(self.generation, idx, code, net, why)
            elif retry and self.trainer.level > caution(out):  # the level that worked
                raise_caution(out, self.trainer.level, self.generation, idx, code)
            self.trainer.note = None
            marker.unlink(missing_ok=True)
            individual.fitness = float(result["fitness"])
            individual.metric = result.get("cloud_ba")
            individual.fps = 1000.0 / result["ms_per_tile"] if result["ms_per_tile"] else None
            individual.model_size = result["parameters"]
            if self.df is None or len(self.df) != len(self.population):
                self._update_df()
            for column, value in (
                ("Fitness", individual.fitness or 1e-9),  # 0 tells PyNAS "not trained yet"
                ("Metric", individual.metric),
                ("FPS", individual.fps),
                ("MB", result["mb"]),
                ("Code", code),
            ):
                self.df.loc[idx, column] = value
            self.save_dataframe()
            self.save_population()
            if retry:  # done; the rest in a process without the slow launches
                level = caution(out)
                print(
                    f"  gen {self.generation} #{idx}: trained again; starting again, caution "
                    f"{level} ({LEVELS[level]})",
                    flush=True,
                )
                sys.stdout.flush()
                os._exit(4)
            return result

    return Search


def last_generation(out: Path) -> int | None:
    """The last generation whose population PyNAS saved in OUT, or None."""
    found = [int(p.stem.rsplit("_", 1)[1]) for p in (out / "src").glob("population_*.pkl")]
    return max(found) if found else None


def caution(out: Path) -> int:
    """The level of caution (LEVELS) the search trains at: 0, or the level that trained a
    candidate the process had stopped on (caution.json)."""
    path = out / CAUTION
    return int(json.loads(path.read_text())["level"]) if path.exists() else 0


def raise_caution(out: Path, level: int, generation: int, index: int, code: str) -> None:
    """The search's level of caution from now on: the one that trained a candidate the process
    had stopped on."""
    found = {"level": level, "settings": LEVELS[level]}
    found |= {"since_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    found |= {"found_with": {"generation": generation, "index": index, "code": code}}
    (out / CAUTION).write_text(json.dumps(found) + "\n")


def allowed(level: int) -> int:
    """How many times a candidate may stop the process, at the search's level of caution,
    before it is left out: until it has stopped it at the last level, kernels one at a time."""
    return max(2, len(LEVELS) - level)


def plan(out: Path) -> tuple[int, bool]:
    """This process's level of caution, and whether it launches the GPU's kernels one at a
    time: the search's level; or, to train again the candidate the last run stopped on
    (trying.json) and not to leave out yet, one level up per time it stopped, one at a time."""
    level = caution(out)
    marker = out / TRYING
    if marker.exists():
        tries = int(json.loads(marker.read_text()).get("tries", 1))
        if tries < allowed(level):
            return min(level + tries, len(LEVELS) - 1), True
    return level, False


def careful(level: int) -> None:
    """The settings of a level of caution that are the whole process's: from 1, attention by
    its formula (no fused kernel; no fast path of the Transformer layer); from 2, no cuDNN.
    The Trainer sets float32 and unpinned data."""
    if level >= 1:
        for switch in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            if hasattr(torch.backends.cuda, switch):
                getattr(torch.backends.cuda, switch)(False)
        mha = getattr(torch.backends, "mha", None)
        if mha is not None and hasattr(mha, "set_fastpath_enabled"):
            mha.set_fastpath_enabled(False)
    if level >= 2:
        torch.backends.cudnn.enabled = False


def leave_out(pop, out: Path, record: Path) -> bool:
    """The candidate the last run stopped on (trying.json, of this generation): trained again,
    one level of caution up, until it has stopped the process at the last level too
    (allowed()); then it is scored 0, marked scored and left out. True if it was left out."""
    marker = out / TRYING
    if not marker.exists():
        return False
    found = json.loads(marker.read_text())
    idx = found.get("index", -1)
    if found.get("generation") != pop.generation or not 0 <= idx < len(pop.population):
        marker.unlink()
        return False
    if int(found.get("tries", 1)) < allowed(caution(out)):  # run_precision_suite counts them
        level, _ = plan(out)
        print(
            f"  gen {pop.generation} #{idx}: the last run stopped on it; again, caution {level} "
            f"({LEVELS[level]}), kernels one at a time",
            flush=True,
        )
        return False
    marker.unlink()
    individual = pop.population[idx]
    individual.fitness = 0.0
    if pop.df is None or len(pop.df) != len(pop.population):
        pop._update_df()
    pop.df.loc[idx, "Fitness"] = 1e-9  # 0 tells PyNAS "not trained yet"
    pop.df.loc[idx, "Code"] = found.get("code", "")
    pop.save_dataframe()
    pop.save_population()
    line = found | {"steps": 0, "fitness": 0.0, "ms_per_tile": None, "per_camera": {}}
    if getattr(individual, "model_size", None):
        line["parameters"] = int(individual.model_size)
        line["mb"] = round(BYTES * line["parameters"] / 1e6, 3)
    line["failed"] = LEFT_OUT
    with open(record, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"  gen {pop.generation} #{idx}: {line['failed']}; scored 0, left out", flush=True)
    return True


def stopping(out: Path) -> None:
    """A process stopped by hand (tmux closed, kill, Ctrl-C) forgets the candidate it was on,
    which is trained afresh when the search is started again: only a fault on the GPU (which
    leaves trying.json) moves a candidate a level of caution up."""

    def stop(signum, _frame):
        (out / TRYING).unlink(missing_ok=True)
        os._exit(128 + signum)

    for name in ("SIGHUP", "SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)


def given_up(record: Path) -> bool:
    """GIVE_UP candidates or more left out for stopping the process at every level of caution,
    and none scored: the fault is in what every candidate shares (the band encoder, the data,
    the loss, the GPU), not in their trunks."""
    if not record.exists():
        return False
    rows = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
    downs = sum(r.get("failed") == LEFT_OUT for r in rows)
    return downs >= GIVE_UP and not any(not r.get("failed") for r in rows)


def best(record: Path, n: int = 10) -> list[dict]:
    """The best trunks of search.jsonl: each code once, with its mean fitness over the times it
    was trained (the kept best are trained again each generation)."""
    rows = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if not r.get("failed")]
    by_code: dict[str, list[dict]] = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append(r)
    out = []
    for code, found in by_code.items():
        fitness = [r["fitness"] for r in found]
        first = found[0]
        out.append(
            {
                "code": code,
                "fitness": float(np.mean(fitness)),
                "trained": len(found),
                "spread": float(np.max(fitness) - np.min(fitness)),
                "cloud_ba": first.get("cloud_ba"),
                "parameters": first["parameters"],
                "mb": first["mb"],
                "ms_per_tile": first["ms_per_tile"],
            }
        )
    return sorted(out, key=lambda r: r["fitness"], reverse=True)[:n]


def top_codes(records, n: int = 2) -> list[str]:
    """The n best trunks of several searches (their search.jsonl; a missing one is skipped),
    each code once, by its mean fitness (best). Searches trained alike only: a fitness after
    2,000 steps is not one after 3,000."""
    found = [row for record in map(Path, records) if record.exists() for row in best(record, 10**9)]
    found.sort(key=lambda row: row["fitness"], reverse=True)
    return list(dict.fromkeys(row["code"] for row in found))[:n]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="a folder for the search")
    p.add_argument("--head", type=Path, required=True, help="a finished run's final.pt")
    p.add_argument("--arm", default="sim", help="whose band encoder: an arm that reads the card")
    p.add_argument("--data", type=Path, default=Path("/root/catena_data"))
    p.add_argument("--s2", type=Path, help="CloudSEN12+ (default DATA/chips/cloudsen12plus)")
    p.add_argument("--biome", type=Path, help="L8 Biome (default DATA/chips/l8_biome)")
    p.add_argument("--perusat", type=Path, help="CloudPeru2 (default DATA/chips/cloudperu2)")
    p.add_argument("--hold-out", default="pilot", help="the Biome table kept out of training")
    p.add_argument("--shares", type=float, nargs=4, default=list(C.SHARES))
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--max-layers", type=int, default=4, help="encoder layers: 3 to this")
    p.add_argument("--budget-mb", type=float, default=6.0, help="head and trunk, at float16")
    p.add_argument("--mating", type=float, default=0.5, help="the best share that mates")
    p.add_argument("--mutation", type=float, default=0.2, help="a gene's chance to mutate")
    p.add_argument("--k-best", type=int, default=1, help="the best kept as they are")
    p.add_argument("--n-random", type=int, default=1, help="new random ones per generation")
    p.add_argument("--steps", type=int, default=3000, help="training steps per candidate")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=loader.SIZE, help="Sentinel-2 windows, 10 m px")
    p.add_argument("--landsat-size", type=int, default=80)
    p.add_argument("--heo-size", type=int, default=240)
    p.add_argument("--perusat-size", type=int, default=240)
    p.add_argument("--p-s2", type=float, default=0.6, help="10 m samples kept as delivered")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--val-windows", type=int, default=120, help="windows per validation camera")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", help="default cuda:0 if there is a GPU, else cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.population < 2 or args.generations < 1 or not 3 <= args.max_layers:
        p.error("--population 2 or more, --generations 1 or more, --max-layers 3 or more")
    if args.size % 120:
        p.error("--size must be a multiple of 120: divisible by 1, 3 and 5, then by 8")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not faulthandler.is_enabled():  # a fault that kills the process at once: where it was
        try:
            faulthandler.enable()
        except (AttributeError, OSError, RuntimeError, ValueError):  # stderr is not a file
            pass
    if argv is None:  # the command: a stop by hand is not a fault to step up from
        stopping(args.out)
    level, one_at_a_time = plan(args.out)
    if one_at_a_time:  # read by CUDA when it starts, so before anything touches the GPU
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    careful(level)
    random.seed(args.seed)  # PyNAS draws with random and numpy
    np.random.seed(args.seed)  # noqa: NPY002  (PyNAS draws from numpy's global generator)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False  # every trunk has new shapes: no tuning for each
    head = load_head(args.head, args.arm)
    size = smallest(args)
    record = args.out / "search.jsonl"
    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": T.git_state(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "host": platform.node(),
        "head_parameters": count_parameters(head),
        "checked_at_px": size,
        "pynas_config": str(CONFIG),
        "fitness": "mean F1 of the four classes, mean over the validation cameras",
        "validation_cameras": [c[0] for c in T.VAL_CAMERAS],
    }
    previous = args.out / "config.json"
    if previous.exists():
        config = json.loads(previous.read_text()) | {"resumed_utc": config["started_utc"]}
    previous.write_text(json.dumps(config, indent=2) + "\n")

    trainer = Trainer(args, device, record, level)
    pop = search_class()(args.out, head, trainer, args, size)
    pop.retry = one_at_a_time
    print(
        f"search: {args.population} trunks x {args.generations} generations behind the "
        f"{args.arm} head of {args.head} ({config['head_parameters']:,} parameters); "
        f"budget {args.budget_mb} MB; checked at {size} px; {args.steps} steps each on "
        f"{device}; caution {level} ({LEVELS[level]})"
        + ("; kernels one at a time" if one_at_a_time else ""),
        flush=True,
    )
    g = last_generation(args.out)
    if g is None:  # a new search: no candidate to train again
        (args.out / TRYING).unlink(missing_ok=True)
    else:  # going on: PyNAS skips the candidates its dataframe has a fitness for
        pop.population, pop.df, pop.generation = pop.load_population(g), pop.load_dataframe(g), g
        print(f"going on from generation {g}", flush=True)
        if leave_out(pop, args.out, record) and given_up(record):
            print(
                f"the search gives up: {GIVE_UP} or more candidates stopped the process at every "
                "level of caution and none was scored, so the fault is in what they all share "
                "(the band encoder, the data, the loss, the GPU), not in their trunks; the "
                "tracebacks above, kernels one at a time, name the operation",
                flush=True,
            )
            raise SystemExit(5)
    if one_at_a_time and not (args.out / TRYING).exists():
        raise SystemExit(4)  # nothing to train again after all: started again as usual
    if g is None:
        pop.initial_poll()
    while True:
        pop.train_generation(task="segmentation", lr=args.lr, epochs=0, batch_size=args.batch)
        if pop.generation + 1 >= args.generations:
            break
        pop.evolve(
            mating_pool_cutoff=args.mating,
            mutation_probability=args.mutation,
            k_best=args.k_best,
            n_random=args.n_random,
        )

    top = best(record)
    print("\nthe best trunks (mean fitness when trained more than once):")
    for r in top:
        print(
            f"  {r['fitness']:.4f} (x{r['trained']})  cloud BA {r['cloud_ba']}  "
            f"{r['parameters']:,} parameters, {r['mb']} MB, {r['ms_per_tile']} ms/tile  {r['code']}"
        )
    if len(top) >= 2:
        print(
            "\nto train the best two end to end (3 seeds):\n"
            f"  OUT=/root/catena_data/runs/nas/best bash scripts/big.sh --arms nas nas2 "
            f'--trunk "{top[0]["code"]}" "{top[1]["code"]}"'
        )


if __name__ == "__main__":
    main()
