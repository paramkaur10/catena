"""What flies: the image's path of a network, its size and its work per tile, its 8-bit twin
(PyNAS's quantization simulation) close to it, and the command's table. Needs PyNAS: skipped
without it."""

from __future__ import annotations

import csv

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pynas")

from catena import descriptor as desc  # noqa: E402
from catena import evaluate_clarity as E  # noqa: E402
from catena import onboard  # noqa: E402
from catena import train_clarity as C  # noqa: E402
from catena.model import Clarity, FiLM, FixedBands  # noqa: E402


def small(fields: int = desc.N_CARD) -> Clarity:
    torch.manual_seed(0)
    return Clarity(widths=(8, 8, 8, 8), stem=4, fields=fields).eval()


def test_what_runs_on_board():
    net = small()
    weights, others = onboard.board_parameters(net)
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert weights == sum(m.weight.numel() for m in convs)  # Clarity has no linear layer there
    films = sum(p.numel() for m in net.modules() if isinstance(m, FiLM) for p in m.parameters())
    card = sum(p.numel() for n in onboard.CARD_PATH for p in getattr(net.bands, n).parameters())
    assert weights + others == sum(p.numel() for p in net.parameters()) - films - card
    fixed = FixedBands(widths=(8, 8, 8, 8), stem=4)
    assert sum(onboard.board_parameters(fixed)) == sum(p.numel() for p in fixed.parameters())


def test_the_work_of_a_tile():
    net = small()
    four, eight = (onboard.multiply_adds(net, desc.N_CARD, b) for b in (4, 8))
    per_band = (1 * 4 * 9 + 4 * 4 * 9) * 240 * 240  # the stem's two 3x3 convolutions
    assert eight - four == 4 * per_band and four > 4 * per_band
    assert onboard.multiply_adds(FixedBands(widths=(8, 8, 8, 8), stem=4), 0, 4) > 0


def batches(n: int = 3):
    g = torch.Generator().manual_seed(1)
    return [
        {
            "camera": torch.zeros(2, dtype=torch.long),
            "d_told": torch.randn(2, 5, desc.N_FIELDS, generator=g) * 0.5,
            "image": torch.rand(2, 5, 48, 48, generator=g),
            "real": torch.ones(2, 5, dtype=torch.bool),
        }
        for _ in range(n)
    ]


def test_an_8_bit_twin_stays_close():
    net = small()
    twin = onboard.quantized(net)
    q = onboard.quant()
    convs = sum(isinstance(m, torch.nn.Conv2d) for m in net.modules())
    assert sum(isinstance(m, q.QuantConv2d) for m in twin.modules()) == convs
    assert not any(isinstance(m, q.QuantLinear) for m in twin.modules())  # card's path: float
    assert not any(isinstance(m, q.QuantConv2d) for m in net.modules())  # a copy
    twins = {"m@int8": E.Net(twin, desc.N_CARD, amp=False)}
    assert onboard.calibrate(twins, batches(), torch.device("cpu")) == 3
    observers = [m for m in twin.modules() if isinstance(m, q.ActFakeQuant)]
    assert observers and all(bool(m.initialized) for m in observers)
    assert all(float(m.running_max) > float(m.running_min) for m in observers)
    b = batches(1)[0]
    told = desc.chain_card(b["d_told"])
    with torch.no_grad():
        a, z = net(b["image"], told, b["real"]), twin(b["image"], told, b["real"])
    assert not torch.equal(a, z)  # 8 bits change something
    assert float((a - z).norm() / a.norm()) < 0.25  # and not much


def test_the_command(tmp_path):
    runs = tmp_path / "runs"
    (runs / "seed0").mkdir(parents=True)
    torch.manual_seed(0)
    models = {a: C.build(a, (8, 8, 8, 8), 8) for a in ("sim", "fixed")}
    C.save(runs / "seed0/final.pt", 1, models, {"args": {"widths": [8, 8, 8, 8], "stem": 8}})
    out = tmp_path / "onboard.csv"
    onboard.main([str(runs), "--device", "cpu", "--out", str(out)])
    with open(out, newline="") as f:
        rows = {r["arm"]: r for r in csv.DictReader(f)}
    assert set(rows) == {"sim", "fixed"}
    sim = rows["sim"]
    assert int(sim["on_board"]) < int(sim["parameters"])  # the card's path stays on the ground
    assert float(sim["mb_int8"]) < float(sim["mb_float16"])
    assert float(sim["gmacs_8_bands"]) > float(sim["gmacs_4_bands"]) > 0
    assert float(sim["ms_cpu_1_thread"]) > 0 and int(sim["hailo8_peak_tiles_per_s"]) > 0
    assert rows["fixed"]["on_board"] == rows["fixed"]["parameters"]
