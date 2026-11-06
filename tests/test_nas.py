"""The architecture search behind the descriptor head: PyNAS's trunks behind a band encoder,
for any window size and with GroupNorm; a run's head, kept as it is; crossover that leaves the
parents alone; a tiny search on made-up data (CPU), gone on with, and its candidates that fail
(out of memory; a fault on the GPU, tried again, then left out); and a trunk it finds, trained
as an arm and tested. Needs PyNAS (py-q-nas) installed: skipped without it."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("pynas")
pd = pytest.importorskip("pandas")

from test_train_clarity import data, run_args  # noqa: E402, F401  (the made-up DATA folder)

from catena import descriptor as desc  # noqa: E402
from catena import evaluate_clarity as E  # noqa: E402
from catena import nas  # noqa: E402
from catena import train as T  # noqa: E402
from catena import train_clarity as C  # noqa: E402
from catena.model import BandEncoder, Clarity  # noqa: E402

# three layers: ConvBnAct keeping 16 channels, ConvBnAct doubling them, a ResNet block; pools
CODE = "Lco01k3s1p1arn1EPM2ELco02k3s1p1agn1EPM2ELRr3arn1EPa2EE"


def batch(bands: int, size: int, n: int = 2):
    g = torch.Generator().manual_seed(size)
    x = torch.rand(n, bands, size, size, generator=g)
    told = torch.randn(n, bands, desc.N_CHAIN_CARD, generator=g)
    return x, told, torch.ones(n, bands, dtype=torch.bool)


def test_a_trunk_for_any_window():
    torch.manual_seed(0)
    trunk = nas.build_trunk(CODE, 16, size=24)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in trunk.modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in trunk.modules())
    net = nas.Candidate(BandEncoder(8, desc.N_CARD), trunk).eval()
    for size in (24, 40, 120, 240):
        with torch.no_grad():
            assert net(*batch(5, size)).shape == (2, 4, size, size)
    with torch.no_grad():  # any bands, in any number
        assert net(*batch(13, 48)).shape == (2, 4, 48, 48)


def test_the_head_of_a_run_stays_as_it_is(tmp_path):
    torch.manual_seed(0)
    models = {v: Clarity(widths=(8, 8, 8, 8), stem=8, fields=desc.N_CARD) for v in C.VERSIONS}
    C.save(tmp_path / "final.pt", 1, models, {"args": {"widths": [8, 8, 8, 8], "stem": 8}})
    head = nas.load_head(tmp_path / "final.pt", "sim")
    for k, v in models["sim"].bands.state_dict().items():
        assert torch.equal(head.state_dict()[k], v)
    net = nas.Candidate(head, nas.build_trunk(CODE, 16, size=24), fixed=True)
    net(*batch(4, 24)).square().mean().backward()
    assert all(p.grad is None for p in net.head.parameters())
    assert all(p.grad is not None for p in net.trunk.parameters())
    with pytest.raises(SystemExit):
        nas.load_head(tmp_path / "final.pt", "chain")  # no such arm in this run


def test_crossover_leaves_the_parents_alone():
    pynas = nas.pynas()
    from pynas.core.individual import Individual

    parents = [Individual(max_layers=4), Individual(max_layers=4)]
    p, q = [list(x.chromosome) for x in parents]
    for _ in range(20):
        a, b = nas.crossover(parents)
        assert [list(x.chromosome) for x in parents] == [p, q]  # the parents as they were
        cuts = range(1, min(len(p), len(q)) - 1)
        assert any(a.chromosome == p[:k] + q[k:] and b.chromosome == q[:k] + p[k:] for k in cuts)
        assert a.parsed_layers == pynas.builder.parse_architecture_code(
            a.chromosome2architecture(a.chromosome)
        )
    assert pynas.population.single_point_crossover is nas.crossover


def test_the_best_trunks_of_several_searches(tmp_path):
    def row(code, fitness, **more):
        known = {"parameters": 1, "mb": 0.1, "ms_per_tile": 1}
        return {"code": code, "fitness": fitness} | known | more

    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return tmp_path / name

    a = write("a.jsonl", [row("A", 0.5), row("B", 0.7), row("B", 0.5), row("C", 0, failed="x")])
    b = write("b.jsonl", [row("D", 0.65), row("A", 0.9)])
    # by mean fitness: A 0.9 (b), D 0.65, B 0.6, A 0.5 (a); each code once; C failed
    assert nas.top_codes([a, b, tmp_path / "missing.jsonl"]) == ["A", "D"]
    assert nas.top_codes([a], 5) == ["B", "A"]
    assert nas.top_codes([tmp_path / "missing.jsonl"]) == []


def test_a_tiny_search(data, tmp_path, monkeypatch):  # noqa: F811
    run = tmp_path / "run"
    C.main(run_args(data, run))  # a finished run, real and sim, to take the head from
    out = tmp_path / "search"
    args = [str(out), "--head", str(run / "final.pt"), "--data", str(data), "--population", "4"]
    args += ["--generations", "2", "--steps", "2", "--batch", "2", "--size", "120"]
    args += ["--landsat-size", "32", "--heo-size", "32", "--perusat-size", "32"]
    args += ["--workers", "0", "--val-windows", "2", "--device", "cpu", "--max-layers", "3"]
    args += ["--warmup", "1"]
    nas.main(args)
    rows = [json.loads(x) for x in (out / "search.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and [r["generation"] for r in rows] == [0] * 4 + [1] * 4
    for r in rows:
        assert 0 <= r["fitness"] <= 1 and r["parameters"] > 0 and r["ms_per_tile"] > 0
        assert r["mb"] == pytest.approx(2 * r["parameters"] / 1e6, abs=1e-3)
        assert r["parameters"] < 6e6 / 2  # within the budget, head included
    scored = [r for r in rows if r["steps"] == 2]
    assert scored and set(scored[0]["per_camera"]) == {c[0] for c in T.VAL_CAMERAS}
    assert (out / "src/population_1.pkl").exists() and (out / "population.log").exists()
    config = json.loads((out / "config.json").read_text())
    assert config["checked_at_px"] == 24 and config["head_parameters"] > 0
    nas.main(args)  # again: everything is scored already
    assert len((out / "search.jsonl").read_text().splitlines()) == 8
    top = nas.best(out / "search.jsonl", 3)
    assert len(top) <= 3 and top == sorted(top, key=lambda r: r["fitness"], reverse=True)
    assert all(np.isfinite(r["fitness"]) for r in top)

    # candidates that fail, in the last generation: each made "not trained yet" in turn
    frame, marker = out / "src/df_population_1.pkl", out / nas.TRYING

    def lines():
        return [json.loads(x) for x in (out / "search.jsonl").read_text().splitlines()]

    def not_trained(idx):
        df = pd.read_pickle(frame)
        df.loc[idx, "Fitness"] = np.nan
        df.to_pickle(frame)

    def failing(error):
        def train(self, net, generation, index, code):
            raise error

        return train

    def stop(code):
        raise SystemExit(code)

    fault = RuntimeError("CUDA error: unspecified launch failure")
    not_trained(0)  # out of memory: scored 0, and the search goes on
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(torch.OutOfMemoryError("out of memory")))
        nas.main(args)
    assert lines()[-1]["index"] == 0 and lines()[-1]["failed"] == "out of memory"
    assert len(lines()) == 9 and not marker.exists()
    not_trained(1)  # a fault on the GPU: the process stops, then trains it once more
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(fault))
        m.setattr(nas.os, "_exit", stop)
        with pytest.raises(SystemExit):
            nas.main(args)
    assert json.loads(marker.read_text()) | {"code": ""} == {
        "generation": 1,
        "index": 1,
        "code": "",
        "tries": 1,
    }
    nas.main(args)
    assert len(lines()) == 10 and lines()[-1]["index"] == 1 and "failed" not in lines()[-1]
    assert not marker.exists()
    not_trained(2)  # stopped on twice: left out, not trained again
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(fault))
        m.setattr(nas.os, "_exit", stop)
        for _ in range(2):
            with pytest.raises(SystemExit):
                nas.main(args)
    assert json.loads(marker.read_text())["tries"] == 2
    nas.main(args)
    assert len(lines()) == 11 and lines()[-1]["index"] == 2 and lines()[-1]["steps"] == 0
    assert lines()[-1]["failed"].startswith("it took the process down") and not marker.exists()
    fitness = pd.read_pickle(frame)["Fitness"]
    assert fitness[0] == fitness[2] == pytest.approx(1e-9) and fitness.notna().all()
    nas.main(args)  # nothing left to train
    assert len(lines()) == 11
    assert all(r["code"] for r in nas.best(out / "search.jsonl"))


def test_a_found_trunk_trained_and_tested(data, tmp_path):  # noqa: F811
    out = tmp_path / "picks"
    C.main(run_args(data, out, "--arms", "nas", "sim", "--trunk", CODE))
    state = torch.load(out / "final.pt", map_location="cpu")
    assert state["arms"]["nas"]["code"] == CODE and "code" not in state["arms"]["sim"]
    assert any(k.startswith("trunk.") for k in state["models"]["nas"])
    assert any(k.startswith("head.") for k in state["models"]["nas"])
    nets = E.load_runs([out], "cpu")
    assert set(nets) == {"picks/nas", "picks/sim"}
    x = np.random.default_rng(0).random((2, 4, 80, 80)).astype(np.float32)
    told = np.zeros((4, desc.N_CHAIN_CARD), np.float32)
    found = E.classifier(nets, torch.device("cpu"), False)(x, told, (0, 1, 2, 3))
    assert {k: v.shape for k, v in found.items()} == {k: (2, 80, 80) for k in nets}
    for bad in (["--arms", "nas", "sim"], ["--arms", "real", "sim", "--trunk", CODE]):
        with pytest.raises(SystemExit):
            C.parse_args(["x", *bad])
