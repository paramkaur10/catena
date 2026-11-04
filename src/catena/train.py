"""Training: the told and the blind arm on the same batches.

    python -m catena.train CHIPS OUT [--seed 0] [--steps 30000] [--devices cuda:0 cuda:1]
        [--biome L8_BIOME] [--no-sim] [--jitter 1]

CHIPS is the CloudSEN12+ folder made by D2 (index.csv and the patches), OUT a new folder
for this run. Every batch is made once and given to both arms, each on its own device: the
told arm reads the full descriptor, with descriptor dropout; the blind arm reads only the
wavelengths, as SEnSeI does, and the pixel size on the ground. The arms start from the same
weights and see the same windows, cameras, bands and noise, so they differ only in what they
are told about the imaging chain.

One training sample from Sentinel-2:
  1. a 240 x 240 window at 10 m from the training split, drawn by area (D3 loader);
  2. a pixel size shared by the whole batch: 10, 30 or 50 m (always 10 m with --no-sim);
  3. a random subset of 3 to 13 bands, each band's gain jittered by up to 5%;
  4. at 10 m, 60% of the samples stay as Sentinel-2 made them, with Sentinel-2's own
     descriptor; all others go through a random camera, so 80% are simulated overall
     (none with --no-sim);
  5. the label moved onto the camera's grid.
With --biome, a share of the batches (--landsat-share, half by default) comes from Landsat 8
instead: an 80 x 80 window at 30 m from a Biome scene outside the --hold-out table, 3 to 8
of its 8 bands, the same gain jitter, and Landsat 8's own descriptor. Every card follows one
rule (catena.descriptor): a blur nobody published is unknown, and the step is one level of
the bits the camera digitises. With --jitter, told's cards are only roughly right.
Loss: the partial-label cross-entropy with mild class weights. AdamW, a linear warm-up,
then cosine decay to zero; gradients clipped at 1; bf16 on the GPU.

Validation: 6 fixed cameras on fixed windows of the validation split, with the same noise
every time, so runs and arms compare. OUT receives:
  config.json     settings, versions, devices and the git commit
  metrics.jsonl   training loss every --log-every steps, validation every --val-every
  last.pt         both arms and their optimisers, at every validation
  final.pt        both arms at the end
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from catena import descriptor as desc
from catena import sim
from catena.data import biome, loader
from catena.model import CatenaNet, count_parameters

ARMS = ("told", "blind")
NAMES = ("clear", "thin", "thick", "shadow")
N_BANDS = len(loader.BANDS)
NO_LABEL = loader.NO_LABEL
DEFAULT_ZENITH = 45.0  # for a patch without a sun angle

# the classes each label code allows (columns: clear, thin, thick, shadow); 255 is ignored
SETS = torch.tensor(
    [
        [1, 0, 0, 0],  # 0 clear
        [0, 1, 0, 0],  # 1 thin cloud
        [0, 0, 1, 0],  # 2 thick cloud
        [0, 0, 0, 1],  # 3 cloud shadow
        [0, 1, 1, 0],  # 4 cloud, thickness unknown
        [1, 0, 0, 1],  # 5 not cloud
    ],
    dtype=torch.float32,
)
# mild weights, sqrt(clear share / class share), from 2,000 training windows (D3, 26 Oct):
# clear 58.6%, thin 10.1%, thick 23.1%, shadow 7.5%; codes 4 and 5 take thick's and clear's
CODE_WEIGHTS = torch.tensor([1.0, 2.41, 1.59, 2.80, 1.59, 1.0])

# validation cameras: name, pixel size factor q, bands kept, chain (None: Sentinel-2 as it
# was delivered). Their noise comes from VAL_SEED, so it is the same at every validation.
VAL_SEED = 20261027
VAL_CAMERAS = (
    ("sentinel2", 1, loader.BANDS, None),
    (
        "vnir_10m",
        1,
        ("B02", "B03", "B04", "B05", "B06", "B07", "B08"),
        {"mtf": 0.3, "snr": 100, "bits": 12, "top": 1.2},
    ),
    (
        "rgbn_10m_soft",
        1,
        ("B02", "B03", "B04", "B08"),
        {"mtf": 0.12, "snr": 30, "bits": 10, "top": 1.0},
    ),
    (
        "landsat_like_30m",
        3,
        ("B01", "B02", "B03", "B04", "B8A", "B10", "B11", "B12"),
        {"mtf": 0.3, "snr": 150, "bits": 12, "top": 1.2},
    ),
    (
        "rgbn_30m_noisy",
        3,
        ("B02", "B03", "B04", "B08"),
        {"mtf": 0.15, "snr": 15, "bits": 12, "top": 1.2},
    ),
    (
        "six_50m_8bit",
        5,
        ("B02", "B03", "B04", "B08", "B11", "B12"),
        {"mtf": 0.2, "snr": 40, "bits": 8, "top": 0.8},
    ),
)

# Sentinel-2's own chain, from its spec (A2), in the loader's band order
S2 = desc.bands_from_spec("sentinel2_msi")
if [b.name for b in S2] != list(loader.BANDS):
    raise RuntimeError("the Sentinel-2 spec and the loader list the bands differently")


# --- samples ------------------------------------------------------------------------------


def render(
    image: np.ndarray,
    label: np.ndarray,
    sun_zenith: float,
    q: int,
    keep: np.ndarray,
    camera: list[dict] | None,
    rng: np.random.Generator,
    gains: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One window as a camera sees it.

    image (13, H, W) is Sentinel-2 reflectance at 10 m and label (H, W) its label; keep are
    the indexes of the bands kept; camera has one chain per kept band, or is None for
    Sentinel-2 as delivered (then q must be 1). Returns the image (13, H/q, W/q) as float32
    with the kept bands first and zeros after, their honest descriptors (13, 15), which
    slots hold a band (13,), and the label on the camera's grid as int64.
    """
    zenith = sun_zenith if np.isfinite(sun_zenith) and sun_zenith < 85 else DEFAULT_ZENITH
    bands = [S2[k] for k in keep]
    chip = image[keep].astype(np.float64)
    if gains is not None:
        chip *= gains[:, None, None]
    noise = [b.noise(zenith) or (0.0, 0.0) for b in bands]
    tops = [b.top(zenith) for b in bands]
    if camera is None:
        if q != 1:
            raise ValueError("Sentinel-2 as delivered has 10 m pixels: q must be 1")
        x = chip.astype(np.float32)
        d = np.stack(
            [
                desc.describe(
                    *b.edges_nm,
                    b.gsd_m,
                    sim.GRID_M,
                    b.mtf if b.mtf_known else None,  # an assumed blur is no fact: unknown
                    ab,
                    None if top is None else top / desc.S2_LEVELS,  # one level of 12 bits
                    top,
                )
                for b, ab, top in zip(bands, noise, tops, strict=True)
            ]
        )
    else:
        x, d, _ = sim.simulate(
            chip,
            [b.sigma_px(sim.GRID_M) for b in bands],
            noise,
            [b.edges_nm for b in bands],
            q,
            camera,
            rng,
            s2_gsd=[b.gsd_m for b in bands],
            s2_top=tops,
        )
    n, (h, w) = len(keep), x.shape[-2:]
    out = np.zeros((N_BANDS, h, w), np.float32)
    out[:n] = x
    descs = np.zeros((N_BANDS, desc.N_FIELDS), np.float32)
    descs[:n] = d
    real = np.zeros(N_BANDS, bool)
    real[:n] = True
    return out, descs, real, sim.downsample_label(label, q).astype(np.int64)


class TrainSamples(Dataset):
    """Training samples for both arms. Sample i depends only on the seed and i, so a run
    repeats exactly whatever the number of data-loading processes.

    All samples of one batch (i // batch) come from one source: Sentinel-2, or Landsat 8 for
    a share of the batches when landsat windows are given; a Sentinel-2 batch also shares one
    pixel size. told gets the card with dropout (desc.hide), roughly right when jitter > 0
    (desc.jitter); blind gets the wavelengths and the pixel size only."""

    def __init__(
        self,
        windows: loader.TrainingWindows,
        batch: int,
        seed: int,
        p_s2: float = 0.6,
        gain: float = 0.05,
        min_bands: int = 3,
        landsat: biome.Windows | None = None,
        share: float = 0.5,
        simulate: bool = True,
        jitter: float = 0.0,
    ) -> None:
        self.windows, self.batch, self.seed = windows, batch, seed
        self.p_s2, self.gain, self.min_bands = p_s2, gain, min_bands
        self.landsat, self.share = landsat, share
        self.simulate, self.jitter = simulate, jitter

    def __len__(self) -> int:
        return len(self.windows)

    def source(self, i: int) -> str:
        """Where sample i comes from, "sentinel2" or "landsat8": the same for a whole batch."""
        if self.landsat is None:
            return "sentinel2"
        rng = np.random.default_rng([self.seed, 3, i // self.batch])
        return "landsat8" if rng.random() < self.share else "sentinel2"

    def pixel_size(self, i: int) -> int:
        """The q of sample i: every sample of one batch (i // batch) shares it."""
        if not self.simulate:
            return 1
        rng = np.random.default_rng([self.seed, 2, i // self.batch])
        return int(rng.choice(sim.Q_CHOICES))

    def sentinel2(self, i: int, rng: np.random.Generator) -> tuple:
        s = self.windows[i]
        q = self.pixel_size(i)
        n = int(rng.integers(self.min_bands, N_BANDS + 1))
        keep = np.sort(rng.choice(N_BANDS, n, replace=False))
        gains = rng.uniform(1 - self.gain, 1 + self.gain, n)
        camera = None
        if self.simulate and (q > 1 or rng.random() >= self.p_s2):
            _, camera = sim.imagine_camera(rng, n, q)
        return render(s["image"], s["label"], s["sun_zenith"], q, keep, camera, rng, gains)

    def landsat8(self, rng: np.random.Generator) -> tuple:
        scene, dn, label = self.landsat.draw(rng)
        bands = len(biome.BANDS)
        n = int(rng.integers(min(self.min_bands, bands), bands + 1))
        keep = np.sort(rng.choice(bands, n, replace=False))
        gains = rng.uniform(1 - self.gain, 1 + self.gain, n)
        size = dn.shape[-1]
        x = np.zeros((N_BANDS, size, size), np.float32)
        x[:n] = biome.reflectance(scene, dn)[keep] * gains[:, None, None]
        d = np.zeros((N_BANDS, desc.N_FIELDS), np.float32)
        d[:n] = self.landsat.cards[scene.name][keep]
        real = np.zeros(N_BANDS, bool)
        real[:n] = True
        return x, d, real, label.astype(np.int64)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, 1, i])
        if self.source(i) == "landsat8":
            x, d, real, label = self.landsat8(rng)
        else:
            x, d, real, label = self.sentinel2(i, rng)
        n = int(real.sum())
        told = d.copy()
        told[:n] = desc.jitter(desc.hide(d[:n], rng), rng, self.jitter)
        return {
            "image": x,
            "real": real,
            "d_told": told,
            "d_blind": desc.blind(d),
            "label": label,
            "camera": -1,
        }


class ValSamples(Dataset):
    """The validation cameras on n fixed windows: sample c * n + j is camera c on window j,
    the same image every time. The told arm gets the honest descriptors, without dropout."""

    def __init__(self, windows: loader.GridWindows, n: int) -> None:
        self.windows = windows
        self.picks = np.linspace(0, len(windows) - 1, n).round().astype(int)

    def __len__(self) -> int:
        return len(VAL_CAMERAS) * len(self.picks)

    def batches(self, size: int) -> list[list[int]]:
        """Batches that never mix two cameras, whose images differ in size."""
        n = len(self.picks)
        return [
            list(range(c * n + s, c * n + min(s + size, n)))
            for c in range(len(VAL_CAMERAS))
            for s in range(0, n, size)
        ]

    def __getitem__(self, i: int) -> dict:
        c, j = divmod(i, len(self.picks))
        _, q, names, chain = VAL_CAMERAS[c]
        s = self.windows[int(self.picks[j])]
        rng = np.random.default_rng([VAL_SEED, c, j])
        keep = np.array([loader.BANDS.index(b) for b in names])
        camera = None if chain is None else sim.fixed_camera(len(keep), **chain)
        x, d, real, label = render(s["image"], s["label"], s["sun_zenith"], q, keep, camera, rng)
        return {
            "image": x,
            "real": real,
            "d_told": d,
            "d_blind": desc.blind(d),
            "label": label,
            "camera": c,
        }


# --- loss and scores ----------------------------------------------------------------------


def set_loss(
    logits: torch.Tensor, label: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Partial-label cross-entropy: -log of the probability given to the classes a label
    code allows, averaged over the labelled pixels (255 is ignored). With one allowed class
    it is the usual cross-entropy. logits (B, 4, H, W); label (B, H, W), codes 0-5 or 255."""
    valid = label != NO_LABEL
    code = torch.where(valid, label, torch.zeros_like(label)).long()
    allowed = SETS.to(logits.device)[code]  # (B, H, W, 4)
    logp = F.log_softmax(logits.float(), dim=1).permute(0, 2, 3, 1)
    loss = -torch.logsumexp(logp + torch.log(allowed), dim=-1)  # log(0) drops a class
    if weights is not None:
        loss = loss * weights.to(logits.device)[code]
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return loss.sum() / valid.sum().clamp(min=1)


def scores(conf: np.ndarray) -> dict:
    """From a 4 x 4 confusion matrix (rows: label, columns: prediction; clear, thin, thick,
    shadow): F1 per class and their mean, accuracy, and the balanced accuracy of cloud (thin
    or thick) against not cloud."""
    conf = np.asarray(conf, np.float64)
    tp = np.diag(conf)
    f1 = 2 * tp / np.maximum(conf.sum(0) + conf.sum(1), 1)
    cloud, other = [1, 2], [0, 3]
    hit, miss = conf[np.ix_(cloud, cloud)].sum(), conf[np.ix_(cloud, other)].sum()
    calm, false = conf[np.ix_(other, other)].sum(), conf[np.ix_(other, cloud)].sum()
    ba = (hit / max(hit + miss, 1) + calm / max(calm + false, 1)) / 2
    out = {f"f1_{n}": round(float(v), 4) for n, v in zip(NAMES, f1, strict=True)}
    out["f1_mean"] = round(float(f1.mean()), 4)
    out["accuracy"] = round(float(tp.sum() / max(conf.sum(), 1)), 4)
    out["cloud_ba"] = round(float(ba), 4)
    return out


# --- the run ------------------------------------------------------------------------------


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to zero at the last step."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def quiet(_worker: int) -> None:
    """One thread per data-loading process: together the processes fill every core."""
    torch.set_num_threads(1)


def git_state() -> str:
    """The commit this code comes from, with '+changes' when tracked files differ from it."""
    here = Path(__file__).resolve().parent
    try:
        run = {"cwd": here, "capture_output": True, "text": True, "check": True}
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], **run).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+changes" if dirty else "")


def pick_devices(names: list[str] | None) -> list[torch.device]:
    """Told and blind devices: as given, else the first two GPUs, one GPU twice, or the CPU."""
    if names:
        return [torch.device(n) for n in names]
    count = torch.cuda.device_count()
    if count >= 2:
        return [torch.device("cuda:0"), torch.device("cuda:1")]
    return [torch.device("cuda:0" if count else "cpu")] * 2


def forward(net, batch: dict, arm: str, device: torch.device, amp: bool) -> torch.Tensor:
    x = batch["image"].to(device, non_blocking=True)
    d = batch[f"d_{arm}"].to(device, non_blocking=True)
    real = batch["real"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return net(x, d, real)


@torch.no_grad()
def validate(models: dict, devices: dict, data: DataLoader, amp: dict) -> dict:
    """Confusion matrices (camera, label, prediction) of both arms on the validation set."""
    conf = {
        arm: torch.zeros(len(VAL_CAMERAS), 4, 4, dtype=torch.long, device=devices[arm])
        for arm in ARMS
    }
    for net in models.values():
        net.eval()
    for batch in data:
        c = int(batch["camera"][0])
        for arm in ARMS:
            logits = forward(models[arm], batch, arm, devices[arm], amp[arm])
            y = batch["label"].to(devices[arm], non_blocking=True)
            ok = y < 4
            pairs = y[ok] * 4 + logits.argmax(dim=1)[ok]
            conf[arm][c] += torch.bincount(pairs, minlength=16).view(4, 4)
    for net in models.values():
        net.train()
    return {arm: m.cpu().numpy() for arm, m in conf.items()}


def save(path: Path, step: int, models: dict, opts: dict | None, config: dict) -> None:
    """Both arms (and their optimisers, to go on from here), written atomically."""
    state = {
        "step": step,
        "config": config,
        "fields": list(desc.FIELDS),
        "models": {arm: net.state_dict() for arm, net in models.items()},
    }
    if opts is not None:
        state["optimizers"] = {arm: opt.state_dict() for arm, opt in opts.items()}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("chips", type=Path, help="the CloudSEN12+ folder made by D2")
    p.add_argument("out", type=Path, help="a new folder for this run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=loader.SIZE, help="window size at 10 m")
    p.add_argument("--workers", type=int, default=6, help="data-loading processes")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--p-s2", type=float, default=0.6, help="10 m samples kept as delivered")
    p.add_argument("--biome", type=Path, help="the Landsat 8 Biome folder: its scenes join in")
    p.add_argument("--hold-out", default="pilot", help="the Biome table kept out of training")
    p.add_argument("--landsat-share", type=float, default=0.5, help="batches from Landsat 8")
    p.add_argument("--landsat-size", type=int, default=80, help="Landsat 8 windows, 30 m px")
    p.add_argument("--no-sim", action="store_true", help="no simulated cameras at all")
    p.add_argument("--jitter", type=float, default=0.0, help="told's cards roughly right: 1")
    p.add_argument("--widths", type=int, nargs=4, default=[32, 64, 128, 256])
    p.add_argument("--stem", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2500)
    p.add_argument("--val-windows", type=int, default=240, help="windows per validation camera")
    p.add_argument("--devices", nargs=2, help="told and blind devices (default cuda:0 cuda:1)")
    p.add_argument("--no-amp", action="store_true", help="float32 instead of bf16")
    args = p.parse_args(argv)
    if args.size % 120:
        p.error("--size must be a multiple of 120: divisible by 1, 3 and 5, then by 8")
    if args.landsat_size <= 0 or args.landsat_size % 8:
        p.error("--landsat-size must be a multiple of 8")
    if not 0.0 <= args.landsat_share <= 1.0:
        p.error("--landsat-share must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.out / "final.pt").exists():
        raise SystemExit(f"{args.out} already holds a finished run: choose a new folder")
    args.out.mkdir(parents=True, exist_ok=True)
    devices = dict(zip(ARMS, pick_devices(args.devices), strict=True))
    amp = {
        arm: dev.type == "cuda" and not args.no_amp and torch.cuda.is_bf16_supported()
        for arm, dev in devices.items()
    }
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    first = CatenaNet(widths=tuple(args.widths), stem=args.stem)
    models = {"told": first, "blind": copy.deepcopy(first)}  # the same starting weights
    for arm, net in models.items():
        net.to(devices[arm]).train()
    opts = {
        arm: torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for arm, net in models.items()
    }
    scheds = {
        arm: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, args.warmup, args.steps))
        for arm, opt in opts.items()
    }
    weights = {arm: CODE_WEIGHTS.to(dev) for arm, dev in devices.items()}

    windows = loader.TrainingWindows(
        args.chips, "train", size=args.size, length=args.steps * args.batch, seed=args.seed
    )
    landsat = None
    if args.biome is not None:
        held = {s.name for s in biome.scenes(args.biome, args.hold_out)}
        chosen = [s for s in biome.scenes(args.biome, "index") if s.name not in held]
        landsat = biome.Windows(chosen, size=args.landsat_size)
    train_data = TrainSamples(
        windows,
        args.batch,
        args.seed,
        p_s2=args.p_s2,
        landsat=landsat,
        share=args.landsat_share,
        simulate=not args.no_sim,
        jitter=args.jitter,
    )
    val_data = ValSamples(
        loader.GridWindows(args.chips, "validation", size=args.size), args.val_windows
    )
    pin = any(dev.type == "cuda" for dev in devices.values())
    more = {"persistent_workers": True, "prefetch_factor": 4} if args.workers else {}
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=pin,
        worker_init_fn=quiet,
        **more,
    )
    val_loader = DataLoader(
        val_data,
        batch_sampler=val_data.batches(args.batch),
        num_workers=args.workers,
        pin_memory=pin,
        worker_init_fn=quiet,
    )

    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "host": platform.node(),
        "devices": {
            arm: torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
            for arm, dev in devices.items()
        },
        "bf16": amp,
        "parameters": count_parameters(first),
        "train_patches": len(windows.patches),
        "landsat_scenes": [s.name for s in landsat.scenes] if landsat else [],
        "validation_windows": len(val_data.picks),
        "validation_cameras": [c[0] for c in VAL_CAMERAS],
        "fields": list(desc.FIELDS),
        "code_weights": CODE_WEIGHTS.tolist(),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = open(args.out / "metrics.jsonl", "a", buffering=1)  # one line at a time
    print(
        f"{config['parameters']:,} parameters per arm; told on {devices['told']}, blind on "
        f"{devices['blind']}; {len(windows.patches):,} training patches; git {config['git']}",
        flush=True,
    )
    print(
        f"Landsat 8: {len(landsat) if landsat else 0} scenes"
        + (f", {args.landsat_share:.0%} of the batches" if landsat else "")
        + f"; simulated cameras {'off' if args.no_sim else 'on'}; jitter {args.jitter}",
        flush=True,
    )

    running = {arm: torch.zeros((), device=dev) for arm, dev in devices.items()}
    start, step, logged = time.perf_counter(), 0, 0
    for batch in train_loader:
        for arm in ARMS:
            logits = forward(models[arm], batch, arm, devices[arm], amp[arm])
            loss = set_loss(
                logits, batch["label"].to(devices[arm], non_blocking=True), weights[arm]
            )
            opts[arm].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[arm].parameters(), args.clip)
            opts[arm].step()
            scheds[arm].step()
            running[arm] += loss.detach()
        step += 1

        if step % args.log_every == 0 or step == args.steps:
            losses = {arm: running[arm].item() / (step - logged) for arm in ARMS}
            for value in running.values():
                value.zero_()
            logged = step
            if not all(math.isfinite(v) for v in losses.values()):
                raise SystemExit(f"step {step}: the loss is not finite ({losses})")
            seconds = time.perf_counter() - start
            per_step = seconds / step
            memory = {
                arm: round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
                for arm, dev in devices.items()
                if dev.type == "cuda"
            }
            lr = scheds["told"].get_last_lr()[0]
            entry = {"kind": "train", "step": step, "lr": lr, "s_per_step": round(per_step, 4)}
            entry |= {f"loss_{arm}": round(v, 5) for arm, v in losses.items()}
            log.write(json.dumps(entry | {"gpu_gb": memory}) + "\n")
            print(
                f"step {step:,}/{args.steps:,}  loss told {losses['told']:.4f} blind "
                f"{losses['blind']:.4f}  lr {lr:.2e}  {per_step:.2f} s/step  "
                f"{(args.steps - step) * per_step / 3600:.1f} h to go  GPU GB {memory}",
                flush=True,
            )

        if step % args.val_every == 0 or step == args.steps:
            began = time.perf_counter()
            conf = validate(models, devices, val_loader, amp)
            for arm in ARMS:
                per_camera = [scores(conf[arm][c]) for c in range(len(VAL_CAMERAS))]
                for (name, *_), s in zip(VAL_CAMERAS, per_camera, strict=True):
                    log.write(
                        json.dumps({"kind": "val", "step": step, "arm": arm, "camera": name} | s)
                        + "\n"
                    )
                mean = {
                    k: round(float(np.mean([s[k] for s in per_camera])), 4) for k in per_camera[0]
                }
                log.write(
                    json.dumps({"kind": "val", "step": step, "arm": arm, "camera": "mean"} | mean)
                    + "\n"
                )
                print(
                    f"   validation {arm:<5}  F1 clear {mean['f1_clear']:.3f} thin "
                    f"{mean['f1_thin']:.3f} thick {mean['f1_thick']:.3f} shadow "
                    f"{mean['f1_shadow']:.3f}  cloud BA {mean['cloud_ba']:.3f}",
                    flush=True,
                )
            save(args.out / "last.pt", step, models, opts, config)
            print(f"   validation and checkpoint: {time.perf_counter() - began:.0f} s", flush=True)

        if step >= args.steps:
            break

    save(args.out / "final.pt", step, models, None, config)
    log.close()
    hours = (time.perf_counter() - start) / 3600
    print(f"done: {step:,} steps in {hours:.2f} h, in {args.out}", flush=True)


if __name__ == "__main__":
    main()
