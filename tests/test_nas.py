"""The architecture search behind the descriptor head: PyNAS's trunks behind a band encoder,
for any window size and with GroupNorm; a run's head, kept as it is; crossover that leaves the
parents alone; a tiny search on made-up data (CPU), gone on with, and its candidates that fail
(out of memory; a fault on the GPU, trained again a level of caution up, then left out; the
search giving up); and a trunk it finds, trained as an arm and tested. Needs PyNAS
(py-q-nas) installed: skipped without it."""

from __future__ import annotations

import copy
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


def test_a_trunk_made_for_the_hailo_8(monkeypatch):
    torch.manual_seed(0)
    trunk = nas.build_trunk(CODE, 16, size=24, hailo=True)
    kinds = {type(m) for m in trunk.modules()}
    assert torch.nn.BatchNorm2d in kinds and torch.nn.GroupNorm not in kinds
    assert torch.nn.GELU not in kinds  # CODE's second layer asks for GELU: ReLU instead
    plain = nas.build_trunk(CODE, 16, size=24)
    assert torch.nn.GELU in {type(m) for m in plain.modules()}
    modes = []
    interpolate = torch.nn.functional.interpolate

    def spy(*args, **kwargs):
        modes.append(kwargs.get("mode", "nearest"))
        return interpolate(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy)
    net = nas.Candidate(BandEncoder(8, desc.N_CARD, hailo=True), trunk, hailo=True).eval()
    for size in (24, 40, 120, 240):
        with torch.no_grad():
            assert net(*batch(5, size)).shape == (2, 4, size, size)
    assert modes and set(modes) == {"nearest"}  # every decoder stage
    copied = copy.deepcopy(net)  # as the 8-bit twins are made
    with torch.no_grad():
        x, told, real = batch(4, 40)
        assert torch.equal(copied(x, told, real), net(x, told, real))


def test_the_head_of_a_run_stays_as_it_is(tmp_path):
    torch.manual_seed(0)
    models = {v: Clarity(widths=(8, 8, 8, 8), stem=8, fields=desc.N_CARD) for v in C.VERSIONS}
    C.save(tmp_path / "final.pt", 1, models, {"args": {"widths": [8, 8, 8, 8], "stem": 8}})
    head = nas.load_head(tmp_path / "final.pt", "sim")
    for k, v in models["sim"].bands.state_dict().items():
        assert torch.equal(head.state_dict()[k], v)
    net = nas.Candidate(head, nas.build_trunk(CODE, 16, size=24), fixed=True).train()
    assert net.trunk.training and not net.head.training  # a fixed head: no dropout
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


def test_the_levels_of_caution():
    sdp, mha = torch.backends.cuda, getattr(torch.backends, "mha", None)
    names = [
        n for n in ("flash_sdp", "mem_efficient_sdp", "cudnn_sdp") if hasattr(sdp, f"{n}_enabled")
    ]
    was = {n: getattr(sdp, f"{n}_enabled")() for n in names}
    fast = mha.get_fastpath_enabled() if mha is not None else None
    cudnn = torch.backends.cudnn.enabled
    try:
        nas.careful(0)  # as fast as it goes: nothing changed
        assert all(getattr(sdp, f"{n}_enabled")() == was[n] for n in names)
        assert torch.backends.cudnn.enabled == cudnn
        nas.careful(1)  # attention by its formula
        assert names and not any(getattr(sdp, f"{n}_enabled")() for n in names)
        assert sdp.math_sdp_enabled() and torch.backends.cudnn.enabled == cudnn
        assert mha is None or not mha.get_fastpath_enabled()
        nas.careful(2)  # and no cuDNN
        assert not torch.backends.cudnn.enabled
    finally:  # the whole process's settings: as they were, for the other tests
        for n, v in was.items():
            getattr(sdp, f"enable_{n}")(v)
        if mha is not None:
            mha.set_fastpath_enabled(fast)
        torch.backends.cudnn.enabled = cudnn
    assert [nas.allowed(level) for level in range(len(nas.LEVELS))] == [3, 2, 2]


def test_a_stop_by_hand_is_not_a_fault(tmp_path, monkeypatch):
    handlers, exits = {}, []
    monkeypatch.setattr(nas.signal, "signal", lambda s, f: handlers.setdefault(s, f))
    monkeypatch.setattr(nas.os, "_exit", exits.append)
    nas.stopping(tmp_path)
    assert {nas.signal.SIGHUP, nas.signal.SIGINT, nas.signal.SIGTERM} <= set(handlers)
    (tmp_path / nas.TRYING).write_text(json.dumps({"generation": 0, "index": 3, "tries": 1}))
    handlers[nas.signal.SIGHUP](nas.signal.SIGHUP, None)  # tmux closed
    assert exits == [128 + nas.signal.SIGHUP] and not (tmp_path / nas.TRYING).exists()
    assert nas.plan(tmp_path) == (0, False)  # trained afresh, at the search's level


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
    levels: list[int] = []  # careful()'s settings are the whole process's: noted, not made
    monkeypatch.setattr(nas, "careful", levels.append)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")  # a retry sets it; put back after the test
    not_trained(0)  # out of memory: scored 0, and the search goes on
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(torch.OutOfMemoryError("out of memory")))
        nas.main(args)
    assert lines()[-1]["index"] == 0 and lines()[-1]["failed"] == "out of memory"
    assert len(lines()) == 9 and not marker.exists() and levels[-1] == 0
    not_trained(1)  # a fault on the GPU: the process stops, then trains it again
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(fault))
        m.setattr(nas.os, "_exit", stop)
        with pytest.raises(SystemExit) as stopped:
            nas.main(args)
    assert stopped.value.code == 3 and nas.os.environ["CUDA_LAUNCH_BLOCKING"] == "0"
    assert json.loads(marker.read_text()) | {"code": ""} == {
        "generation": 1,
        "index": 1,
        "code": "",
        "tries": 1,
    }
    with monkeypatch.context() as m:  # again, a level of caution up, kernels one at a time
        m.setattr(nas.os, "_exit", stop)
        with pytest.raises(SystemExit) as stopped:
            nas.main(args)
    assert stopped.value.code == 4 and nas.os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
    assert levels[-1] == 1 and nas.caution(out) == 1  # the search's level from now on
    assert len(lines()) == 10 and lines()[-1]["index"] == 1 and "failed" not in lines()[-1]
    assert lines()[-1]["caution"] == nas.LEVELS[1] and not marker.exists()
    assert lines()[-1]["rescued"] == "trained again after a fault, kernels one at a time"
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    not_trained(2)  # at caution 1: stops it, then again at 2 one at a time: left out
    with monkeypatch.context() as m:
        m.setattr(nas.Trainer, "__call__", failing(fault))
        m.setattr(nas.os, "_exit", stop)
        for level, blocking in ((1, "0"), (2, "1")):
            with pytest.raises(SystemExit) as stopped:
                nas.main(args)
            assert stopped.value.code == 3 and levels[-1] == level
            assert nas.os.environ["CUDA_LAUNCH_BLOCKING"] == blocking
    assert json.loads(marker.read_text())["tries"] == 2
    nas.main(args)
    assert len(lines()) == 11 and lines()[-1]["index"] == 2 and lines()[-1]["steps"] == 0
    assert lines()[-1]["failed"] == nas.LEFT_OUT and not marker.exists() and levels[-1] == 1
    fitness = pd.read_pickle(frame)["Fitness"]
    assert fitness[0] == fitness[2] == pytest.approx(1e-9) and fitness.notna().all()
    nas.main(args)  # nothing left to train
    assert len(lines()) == 11
    assert all(r["code"] for r in nas.best(out / "search.jsonl"))
    # a third left out so, with none scored (as made here): the search gives up
    down = {"generation": 0, "index": 9, "code": "x", "steps": 0, "failed": nas.LEFT_OUT}
    (out / "search.jsonl").write_text(2 * (json.dumps(down) + "\n"))
    not_trained(3)
    marker.write_text(json.dumps({"generation": 1, "index": 3, "code": "y", "tries": 2}))
    with pytest.raises(SystemExit) as stopped:
        nas.main(args)
    assert stopped.value.code == 5 and len(lines()) == 3 and nas.given_up(out / "search.jsonl")


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


def test_a_found_trunk_trained_for_the_hailo_8(data, tmp_path):  # noqa: F811
    out = tmp_path / "flight"
    more = ("--arms", "nas", "sim", "--trunk", CODE, "--hailo", "--freeze-norms", "0.5")
    C.main(run_args(data, out, *more, "--stop-after", "1"))  # stopped while averaging
    C.main(run_args(data, out, *more))
    state = torch.load(out / "final.pt", map_location="cpu")
    nas_arm = {"data": "sim", "fields": desc.N_CARD, "blind": False, "code": CODE, "hailo": True}
    assert state["arms"]["nas"] == nas_arm and state["arms"]["sim"]["hailo"] is True
    for arm in ("nas", "sim"):  # 4 steps: averaged over steps 0 and 1, frozen from step 2
        counts = [int(v) for k, v in state["models"][arm].items() if k.endswith("batches_tracked")]
        assert 2 in counts and set(counts) <= {0, 2}  # 0: a BatchNorm the forward never uses
    config = json.loads((out / "config.json").read_text())
    assert config["hailo"]["norms_averaged_from_step"] == 0
    assert config["hailo"]["norms_frozen_from_step"] == 2
    nets = E.load_runs([out], "cpu")
    assert set(nets) == {"flight/nas", "flight/sim"}
    kinds = {type(m) for m in nets["flight/nas"].net.modules()}
    assert torch.nn.BatchNorm2d in kinds and torch.nn.GroupNorm not in kinds
    x = np.random.default_rng(0).random((2, 4, 80, 80)).astype(np.float32)
    told = np.zeros((4, desc.N_CHAIN_CARD), np.float32)
    found = E.classifier(nets, torch.device("cpu"), False)(x, told, (0, 1, 2, 3))
    assert {k: v.shape for k, v in found.items()} == {k: (2, 80, 80) for k in nets}
    with pytest.raises(SystemExit):
        C.parse_args(["x", "--hailo", "--freeze-norms", "0"])
