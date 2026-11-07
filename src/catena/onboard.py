"""What flies: the part of a trained network that runs on board for one fixed sensor, its size
and its work per tile, its speed here, and its 8-bit copy.

For one sensor the cards are constants, so the card's path (a band encoder's MLP, its
Transformer layer and its pooling scores, and every FiLM maker) runs once on the ground, and
its outputs fold into the layers they scale (catena.model.fold_block). What runs on board is
the image's path: every convolution, the norms, and a searched trunk's own linear layers.
Its size at 8 bits is a byte per weight and four per other parameter (biases and norms stay
in 32 bits on integer chips); its work, the multiply-adds of one 240 px tile.

8 bits: PyNAS's simulation of the Hailo-8 contract (pynas.core.quant), the chip PyNAS
quantizes for: weights signed and symmetric per output channel, activations unsigned and
asymmetric per tensor, where each convolution or linear layer reads them. quantized() puts it
on the image's path of a trained network, calibrate() sets each activation's range with the
float network: the mean, over batches, of its smallest and its largest value, on the
validation cameras (never test data). Post-training quantization: nothing is retrained. An
8-bit twin runs in float32, never bfloat16 (catena.evaluate_clarity --int8 scores them).

    python -m catena.onboard RUNS [RUNS ...] [--device cuda:0] [--out CSV]

prints and writes (default: the last RUNS folder's onboard.csv) one row per arm (its seed0,
since the seeds share the design): the parameters, all and on board; MB at float16 and at 8
bits; multiply-adds per 240 px tile of 4 and of 8 bands; milliseconds per tile on the GPU
(bfloat16, batches of 8) and on the CPU (float32, one tile; one thread and all of them); and
the tiles per second Hailo-8 could give at its peak (26 TOPS at 8 bits), an upper bound, not
a measure. Time it on an idle machine: other jobs slow the clocks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from catena import descriptor as desc
from catena.model import BandEncoder, FiLM

TILE = 240
CARD_PATH = ("embed", "mix", "score")  # a band encoder's parts that read only the cards
HAILO8_TOPS = 26.0  # Hailo-8's published peak at 8 bits
CALIBRATION_WINDOWS = 64  # per validation camera


def walk(net: nn.Module, visit) -> None:
    """visit(parent, name, child, on_board) for every module under net, depth first; on_board
    is False inside the card's path and the FiLM makers."""

    def down(module: nn.Module, ground: bool) -> None:
        for name, child in list(module.named_children()):
            inside = (
                ground
                or isinstance(child, FiLM)
                or (isinstance(module, BandEncoder) and name in CARD_PATH)
            )
            visit(module, name, child, not inside)
            down(child, inside)

    down(net, False)


def board_parameters(net: nn.Module) -> tuple[int, int]:
    """(weights, others) of what runs on board: the weights of convolutions and linear layers,
    and every other parameter there (biases, norms)."""
    weights, others = [0], [0]

    def visit(parent, name, child, on_board):
        if not on_board:
            return
        for pname, p in child.named_parameters(recurse=False):
            is_weight = pname == "weight" and isinstance(child, nn.Conv2d | nn.Linear)
            (weights if is_weight else others)[0] += p.numel()

    walk(net, visit)
    return weights[0], others[0]


def inputs(fields: int, bands: int, batch: int, device) -> tuple:
    """A batch of random tiles as a network reads them (fields 0: the fixed arm, 4 bands)."""
    if fields == 0:
        return (torch.rand(batch, 4, TILE, TILE, device=device),)
    x = torch.rand(batch, bands, TILE, TILE, device=device)
    told = torch.zeros(batch, bands, desc.N_CHAIN_CARD, device=device)
    return x, told, torch.ones(batch, bands, dtype=torch.bool, device=device)


@torch.no_grad()
def multiply_adds(net: nn.Module, fields: int, bands: int) -> int:
    """The multiply-adds of one 240 px tile on the image's path (convolutions and linear
    layers; the rest is a small share)."""
    total, hooks = [0], []

    def conv(m, args, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += out.numel() * (m.in_channels // m.groups) * k

    def linear(m, args, out):
        total[0] += out.numel() * m.in_features

    def visit(parent, name, child, on_board):
        if on_board and isinstance(child, nn.Conv2d):
            hooks.append(child.register_forward_hook(conv))
        elif on_board and isinstance(child, nn.Linear):
            hooks.append(child.register_forward_hook(linear))

    walk(net, visit)
    device = next(net.parameters()).device
    was = net.training
    net.eval()
    try:
        net(*inputs(fields, bands, 1, device))
    finally:
        for h in hooks:
            h.remove()
        net.train(was)
    return total[0]


@torch.no_grad()
def ms_per_tile(net, fields: int, device, batch: int, amp: bool, repeats: int = 6) -> float:
    """Median milliseconds per tile of 8 bands (4 for the fixed arm), after two warm-ups."""
    net.eval()
    x = inputs(fields, 8, batch, device)
    timed = []
    for k in range(repeats + 2):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            net(*x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if k >= 2:
            timed.append(time.perf_counter() - began)
    return 1000 * float(np.median(timed)) / batch


# --- 8 bits ---------------------------------------------------------------------------------


def quant():
    """PyNAS's quantization simulation (pynas.core.quant), with PyNAS set to Catena's config
    before any of it is imported."""
    from catena import nas

    nas.pynas()
    from pynas.core import quant as q

    return q


def quantized(net: nn.Module) -> nn.Module:
    """A copy of a trained network with 8 bits on the image's path: every convolution, and
    every linear layer outside the card's path and the FiLM makers. Its activations pass
    through unchanged until calibrate() gives them their ranges."""
    q = quant()
    opts = q.QuantOpts()
    twin = copy.deepcopy(net)

    def visit(parent, name, child, on_board):
        if isinstance(child, nn.Conv2d):
            setattr(parent, name, q.QuantConv2d(child, None, opts))
        elif on_board and isinstance(child, nn.Linear):
            setattr(parent, name, q.QuantLinear(child, opts))

    walk(twin, visit)
    return twin.eval()


@torch.no_grad()
def calibrate(twins: dict, data, device) -> int:
    """Each twin's activation ranges from batches of the validation cameras (data: a
    DataLoader of catena.train.ValSamples), read with the float network: the mean of each
    quantized input's smallest and largest value. twins: {name: evaluate_clarity.Net}.
    Returns the number of batches read."""
    from catena import train as T
    from catena.train_clarity import rgbn_slots

    q = quant()
    seen: dict = {}

    def observe(module, args):
        x = args[0].detach()
        seen[module][0].append(x.amin().float())
        seen[module][1].append(x.amax().float())

    hooks = []
    for entry in twins.values():
        entry.net.eval()
        for m in entry.net.modules():
            if isinstance(m, q.ActFakeQuant):
                seen[m] = ([], [])
                hooks.append(m.register_forward_pre_hook(observe))
    batches = 0
    try:
        for batch in data:
            c = int(batch["camera"][0])
            told = desc.chain_card(batch["d_told"]).to(device)
            image, real = batch["image"].to(device), batch["real"].to(device)
            four = rgbn_slots(T.VAL_CAMERAS[c][2])
            for entry in twins.values():
                if entry.fields == 0:
                    entry.net(image[:, four])
                else:
                    entry.net(image, torch.zeros_like(told) if entry.blind else told, real)
            batches += 1
    finally:
        for h in hooks:
            h.remove()
    for m, (lows, highs) in seen.items():
        if lows:
            m.running_min.fill_(float(torch.stack(lows).mean()))
            m.running_max.fill_(float(torch.stack(highs).mean()))
            m.initialized.fill_(True)
    return batches


def int8_twins(
    nets: dict, data: Path, device, workers: int = 6, batch: int = 16, size: int = TILE
) -> dict:
    """Every network's 8-bit twin, named <model>@int8, calibrated on the CloudSEN12+
    validation split through the validation cameras (CALIBRATION_WINDOWS windows of size
    10 m pixels each)."""
    from torch.utils.data import DataLoader

    from catena import evaluate_clarity as E
    from catena import train as T
    from catena.data import loader

    twins = {
        f"{name}@int8": E.Net(quantized(entry.net), entry.fields, entry.blind, amp=False)
        for name, entry in nets.items()
    }
    checks = T.ValSamples(
        loader.GridWindows(data / "chips" / "cloudsen12plus", "validation", size=size),
        CALIBRATION_WINDOWS,
    )
    batches = DataLoader(
        checks,
        batch_sampler=checks.batches(batch),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=T.quiet,
    )
    n = calibrate(twins, batches, device)
    print(f"8 bits: {len(twins)} twins calibrated on {n} batches of the validation cameras")
    return twins


# --- the command ----------------------------------------------------------------------------


def measure(name: str, entry, device) -> dict:
    """One arm's row: sizes, multiply-adds and speeds."""
    net, fields = entry.net, entry.fields
    weights, others = board_parameters(net)
    row = {"model": name, "parameters": sum(p.numel() for p in net.parameters())}
    row |= {"on_board": weights + others}
    row |= {"mb_float16": round(2 * (weights + others) / 1e6, 3)}
    row |= {"mb_int8": round((weights + 4 * others) / 1e6, 3)}
    for bands in (4, 8):
        row[f"gmacs_{bands}_bands"] = round(multiply_adds(net, fields, bands) / 1e9, 3)
    if device.type == "cuda":
        row["ms_gpu_bf16"] = round(ms_per_tile(net, fields, device, 8, True), 3)
    cpu = copy.deepcopy(net).to("cpu").float()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        row["ms_cpu_1_thread"] = round(ms_per_tile(cpu, fields, torch.device("cpu"), 1, False), 1)
        torch.set_num_threads(threads)
        row["ms_cpu_all_threads"] = round(
            ms_per_tile(cpu, fields, torch.device("cpu"), 1, False), 1
        )
    finally:
        torch.set_num_threads(threads)
    macs = multiply_adds(net, fields, 8 if fields else 4)
    row["hailo8_peak_tiles_per_s"] = round(HAILO8_TOPS * 1e12 / (2 * macs))
    return row


def main(argv: list[str] | None = None) -> None:
    from catena import evaluate_clarity as E

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, nargs="+", help="folders of finished runs")
    p.add_argument("--device", help="default cuda:0 if there is a GPU, else cpu")
    p.add_argument("--out", type=Path, help="the CSV (default: LAST RUNS/onboard.csv)")
    args = p.parse_args(argv)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    nets = E.load_runs(args.runs, device)
    arms: dict = {}
    for name, entry in nets.items():  # one seed per arm: the seeds share the design
        arms.setdefault(name.rsplit("/", 1)[-1], (name, entry))
    rows = []
    for arm, (name, entry) in arms.items():
        row = {"arm": arm} | measure(name, entry, device)
        rows.append(row)
        print("  ".join(f"{k} {v}" for k, v in row.items()), flush=True)
    out = args.out or Path(args.runs[-1]) / "onboard.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
