"""The big run: Clarity trained on four satellites, in two versions side by side.

    python -m catena.train_clarity OUT [--data /root/catena_data] [--seed 0] [--steps 60000]
        [--shares 0.4 0.2 0.2 0.2] [--devices cuda:0 cuda:1]

Clarity reads, for each band, only what every product states (catena.descriptor.card): its
wavelengths, surface or top-of-atmosphere reflectance, the band's pixel size against the
grid's, and the grid's pixel size on the ground. Nothing about the imaging chain.

The two versions, one per GPU, start from the same weights and see the same windows in the
same order, with the same bands and gains:
  real   every sample as its satellite delivered it;
  sim    the same, except that 80% of the Sentinel-2 samples are re-imaged by a made-up
         camera (catena.sim): 30 or 50 m pixels, or 10 m, with more blur and noise, fewer
         bits and a saturation from 0.25 to 1.6. These are the same Sentinel-2 windows and
         labels, processed: no new images and no new labels.
So the two differ only in the made-up cameras.

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
throughout), and validation on the 6 cameras of the CloudSEN12+ validation split.

OUT receives config.json, metrics.jsonl, last.pt (both versions with their optimisers and
schedules, at every validation) and final.pt. After a stop, the same command goes on from
last.pt, with the same samples it would have drawn; a finished run is never touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from catena import descriptor as desc
from catena import sim
from catena import train as T
from catena.data import biome, heo, loader, perusat
from catena.model import Clarity, count_parameters

VERSIONS = ("real", "sim")
SOURCES = ("sentinel2", "landsat8", "heo", "perusat1")
SHARES = (0.4, 0.2, 0.2, 0.2)
TOPS = (0.25, 1.6)  # the made-up cameras' saturation, in reflectance
PERUSAT_GAIN = (0.7, 1.5)  # one gain per PeruSat-1 sample, drawn on a log scale
N_BANDS = T.N_BANDS  # band slots: 13, Sentinel-2's count
PLACE = 8  # the sim version's placeholder for a sample it sees as the real one does
# what a resumed run must share with the run it goes on from
SAME_ARGS = ("seed", "steps", "batch", "size", "landsat_size", "heo_size", "perusat_size")
SAME_ARGS += ("shares", "p_s2", "widths", "stem", "lr", "weight_decay", "warmup", "hold_out")


def pack(x: np.ndarray, d: np.ndarray, real: np.ndarray, label: np.ndarray) -> dict:
    return {"image": x, "card": desc.card(d), "real": real, "label": label}


class Samples(Dataset):
    """Training samples for both versions. Sample i depends only on the seed and i, so a run
    repeats exactly whatever the number of data-loading processes, and goes on exactly after
    a stop: `start` is the first sample, step x batch.

    Each item holds the real version's image (13 slots, the kept bands first), card, band
    slots and label, and the sim version's image, card and label. same is True when the sim
    version sees what the real one sees (every batch but Sentinel-2's); its own entries are
    then small placeholders."""

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
    ) -> None:
        self.windows, self.batch, self.seed = windows, batch, seed
        self.data = {"landsat8": landsat, "heo": heo_windows, "perusat1": peru}
        self.shares = np.asarray(shares, np.float64) / float(np.sum(shares))
        for name, share in zip(SOURCES, self.shares, strict=True):
            if share > 0 and name != "sentinel2" and self.data[name] is None:
                raise ValueError(f"{name} has a share of the batches but no data")
        self.p_s2, self.gain, self.min_bands, self.start = p_s2, gain, min_bands, start

    def __len__(self) -> int:
        return len(self.windows) - self.start

    def source(self, i: int) -> str:
        """Where sample i comes from: the same for a whole batch (i // batch)."""
        rng = np.random.default_rng([self.seed, 3, i // self.batch])
        return SOURCES[int(rng.choice(len(SOURCES), p=self.shares))]

    def pixel_size(self, i: int) -> int:
        """The q of the sim version's Sentinel-2 sample i, shared by its batch."""
        rng = np.random.default_rng([self.seed, 2, i // self.batch])
        return int(rng.choice(sim.Q_CHOICES))

    def bands(self, total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Which bands a sample keeps, and their gains."""
        n = int(rng.integers(min(self.min_bands, total), total + 1))
        keep = np.sort(rng.choice(total, n, replace=False))
        return keep, rng.uniform(1 - self.gain, 1 + self.gain, n)

    def from_sentinel2(self, i: int, rng: np.random.Generator) -> dict:
        s = self.windows[i]  # drawn, turned and flipped by the loader
        keep, gains = self.bands(N_BANDS, rng)
        scene = (s["image"], s["label"], s["sun_zenith"])
        x, d, real, label = T.render(*scene, 1, keep, None, rng, gains)
        out = pack(x, d, real, label)
        q = self.pixel_size(i)  # everything the real version sees is drawn by now
        if q > 1 or rng.random() >= self.p_s2:
            _, camera = sim.imagine_camera(rng, len(keep), q, tops=TOPS)
            x, d, _, label = T.render(*scene, q, keep, camera, rng, gains)
        return out | {"sim_image": x, "sim_card": desc.card(d), "sim_label": label, "same": False}

    def as_delivered(self, x: np.ndarray, d: np.ndarray, label, rng) -> dict:
        """A sample both versions see alike: turned at random, its bands in the first slots."""
        turn = int(rng.integers(8))
        n, size = len(x), x.shape[-1]
        image = np.zeros((N_BANDS, size, size), np.float32)
        image[:n] = loader.dihedral(x, turn)
        descs = np.zeros((N_BANDS, desc.N_FIELDS), np.float32)
        descs[:n] = d
        real = np.zeros(N_BANDS, bool)
        real[:n] = True
        out = pack(image, descs, real, loader.dihedral(label, turn).astype(np.int64))
        return out | {
            "sim_image": np.zeros((N_BANDS, PLACE, PLACE), np.float32),
            "sim_card": out["card"],
            "sim_label": np.zeros((PLACE, PLACE), np.int64),
            "same": True,
        }

    def from_landsat8(self, rng: np.random.Generator) -> dict:
        data = self.data["landsat8"]
        scene, dn, label = data.draw(rng)  # clear is 5 in a scene without shadow labels
        keep, gains = self.bands(len(biome.BANDS), rng)
        x = biome.reflectance(scene, dn)[keep] * gains[:, None, None]
        return self.as_delivered(x, data.cards[scene.name][keep], label, rng)

    def from_heo(self, rng: np.random.Generator) -> dict:
        data = self.data["heo"]
        _, dn, label = data.draw(rng)
        keep, gains = self.bands(len(heo.BANDS), rng)
        x = heo.reflectance(dn)[keep] * gains[:, None, None]
        return self.as_delivered(x, data.card[keep], label, rng)

    def from_perusat1(self, rng: np.random.Generator) -> dict:
        data = self.data["perusat1"]
        _, dn, label = data.draw(rng)
        keep, gains = self.bands(len(perusat.BANDS), rng)
        low, high = (math.log(g) for g in PERUSAT_GAIN)
        gains = gains * math.exp(rng.uniform(low, high))  # one more gain for the whole sample
        x = perusat.reflectance(dn)[keep] * gains[:, None, None]
        return self.as_delivered(x, data.card[keep], label, rng)

    def __getitem__(self, j: int) -> dict:
        i = self.start + j
        rng = np.random.default_rng([self.seed, 1, i])
        source = self.source(i)
        if source == "sentinel2":
            out = self.from_sentinel2(i, rng)
        elif source == "landsat8":
            out = self.from_landsat8(rng)
        elif source == "heo":
            out = self.from_heo(rng)
        else:
            out = self.from_perusat1(rng)
        return out | {"source": SOURCES.index(source)}


def version_inputs(batch: dict, version: str) -> tuple:
    """The image, card and label a version sees in a batch."""
    if version == "sim" and not bool(batch["same"][0]):
        return batch["sim_image"], batch["sim_card"], batch["sim_label"]
    return batch["image"], batch["card"], batch["label"]


def forward(net, x, card, real, device: torch.device, amp: bool) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    card = card.to(device, non_blocking=True)
    real = real.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return net(x, card, real)


@torch.no_grad()
def validate(models: dict, devices: dict, data: DataLoader, amp: dict) -> dict:
    """Confusion matrices (camera, label, prediction) of both versions on the validation
    cameras; both see the same images and cards."""
    conf = {
        v: torch.zeros(len(T.VAL_CAMERAS), 4, 4, dtype=torch.long, device=devices[v])
        for v in VERSIONS
    }
    for net in models.values():
        net.eval()
    for batch in data:
        c = int(batch["camera"][0])
        card = desc.card(batch["d_told"])
        for v in VERSIONS:
            logits = forward(models[v], batch["image"], card, batch["real"], devices[v], amp[v])
            y = batch["label"].to(devices[v], non_blocking=True)
            ok = y < 4
            pairs = y[ok] * 4 + logits.argmax(dim=1)[ok]
            conf[v][c] += torch.bincount(pairs, minlength=16).view(4, 4)
    for net in models.values():
        net.train()
    return {v: m.cpu().numpy() for v, m in conf.items()}


def save(path: Path, step: int, models: dict, config: dict, opts=None, scheds=None) -> None:
    """Both versions (with their optimisers and schedules, to go on from), written atomically."""
    state = {
        "step": step,
        "config": config,
        "card": [desc.FIELDS[k] for k in desc.CARD],
        "models": {v: net.state_dict() for v, net in models.items()},
    }
    if opts is not None:
        state["optimizers"] = {v: opt.state_dict() for v, opt in opts.items()}
        state["schedulers"] = {v: s.state_dict() for v, s in scheds.items()}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


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
    p.add_argument("--widths", type=int, nargs=4, default=[32, 64, 128, 256])
    p.add_argument("--stem", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--val-windows", type=int, default=240, help="windows per validation camera")
    p.add_argument("--devices", nargs=2, help="the real and the sim version's devices")
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
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.out / "final.pt").exists():
        raise SystemExit(f"{args.out} already holds a finished run: choose a new folder")
    args.out.mkdir(parents=True, exist_ok=True)
    s2 = args.s2 or args.data / "chips" / "cloudsen12plus"
    devices = dict(zip(VERSIONS, T.pick_devices(args.devices), strict=True))
    amp = {
        v: dev.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
        for v, dev in devices.items()
    }
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    first = Clarity(widths=tuple(args.widths), stem=args.stem, fields=desc.N_CARD)
    models = {"real": first, "sim": copy.deepcopy(first)}  # the same starting weights
    for v, net in models.items():
        net.to(devices[v]).train()
    opts = {
        v: torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for v, net in models.items()
    }
    scheds = {
        v: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: T.lr_factor(s, args.warmup, args.steps))
        for v, opt in opts.items()
    }
    weights = {v: T.CODE_WEIGHTS.to(dev) for v, dev in devices.items()}

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
    rows = breakdown(args, windows, data)

    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    step, last = 0, args.out / "last.pt"
    if last.exists():  # going on after a stop
        state = torch.load(last, map_location="cpu")
        before = state["config"]["args"]
        changed = [k for k in SAME_ARGS if before.get(k) != arguments.get(k)]
        if changed:
            raise SystemExit(f"{last} was made with other settings: {', '.join(changed)}")
        for v in VERSIONS:
            models[v].load_state_dict(state["models"][v])
            opts[v].load_state_dict(state["optimizers"][v])
            scheds[v].load_state_dict(state["schedulers"][v])
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
            "parameters": count_parameters(first),
            "card": [desc.FIELDS[k] for k in desc.CARD],
            "sources": rows,
            "made_up_cameras": {
                "version": "sim",
                "share_of_sentinel2": round(1 - args.p_s2 / len(sim.Q_CHOICES), 4),
                "pixel_m": [sim.GRID_M * q for q in sim.Q_CHOICES],
                "saturation": list(TOPS),
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

    parameters = config["parameters"]
    print(
        f"Clarity: {parameters:,} parameters, {2 * parameters / 1e6:.1f} MB at float16; real on "
        f"{devices['real']}, sim on {devices['sim']}; git {config['git']}",
        flush=True,
    )
    print(f"  {'source':<10}{'data':<52}{'pixel':>7}{'bands':>7}{'share':>7}  labels")
    for r in rows:
        print(
            f"  {r['source']:<10}{r['data']:<52}{r['pixel_m']:>5g} m{r['bands']:>7}"
            f"{r['share']:>7.0%}  {r['labels']}"
        )
    print(
        f"  made-up cameras: sim only, {config['made_up_cameras']['share_of_sentinel2']:.0%} of "
        f"the Sentinel-2 samples, saturation {TOPS[0]}-{TOPS[1]}",
        flush=True,
    )
    if step:
        print(f"going on from step {step:,} (last.pt)", flush=True)

    log = open(args.out / "metrics.jsonl", "a", buffering=1)  # one line at a time
    running = {v: torch.zeros((), device=dev) for v, dev in devices.items()}
    counts = Counter()
    began, first_step, logged = time.perf_counter(), step, step
    for batch in train_loader:
        counts[SOURCES[int(batch["source"][0])]] += 1
        for v in VERSIONS:
            x, card, label = version_inputs(batch, v)
            logits = forward(models[v], x, card, batch["real"], devices[v], amp[v])
            loss = T.set_loss(logits, label.to(devices[v], non_blocking=True), weights[v])
            opts[v].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[v].parameters(), args.clip)
            opts[v].step()
            scheds[v].step()
            running[v] += loss.detach()
        step += 1

        if step % args.log_every == 0 or step == args.steps:
            losses = {v: running[v].item() / (step - logged) for v in VERSIONS}
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
            lr = scheds["real"].get_last_lr()[0]
            entry = {"kind": "train", "step": step, "lr": lr, "s_per_step": round(per_step, 4)}
            entry |= {f"loss_{v}": round(x, 5) for v, x in losses.items()}
            entry |= {"batches": dict(counts), "gpu_gb": memory}
            log.write(json.dumps(entry) + "\n")
            print(
                f"step {step:,}/{args.steps:,}  loss real {losses['real']:.4f} sim "
                f"{losses['sim']:.4f}  lr {lr:.2e}  {per_step:.2f} s/step  "
                f"{(args.steps - step) * per_step / 3600:.1f} h to go  GPU GB {memory}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            started = time.perf_counter()
            conf = validate(models, devices, val_loader, amp)
            for v in VERSIONS:
                per_camera = [T.scores(conf[v][c]) for c in range(len(T.VAL_CAMERAS))]
                for (name, *_), s in zip(T.VAL_CAMERAS, per_camera, strict=True):
                    entry = {"kind": "val", "step": step, "version": v, "camera": name}
                    log.write(json.dumps(entry | s) + "\n")
                mean = {
                    k: round(float(np.mean([s[k] for s in per_camera])), 4) for k in per_camera[0]
                }
                entry = {"kind": "val", "step": step, "version": v, "camera": "mean"}
                log.write(json.dumps(entry | mean) + "\n")
                print(
                    f"   validation {v:<4}  F1 clear {mean['f1_clear']:.3f} thin "
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
