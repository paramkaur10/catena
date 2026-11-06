"""The notebooks: scripts/make_notebooks.py writes valid notebooks whose code compiles, the
labelling records apart from the pictures and from the tools, and leaves notebooks that exist
alone unless --force. Run All writes nothing: neither the records nor the tools."""

from __future__ import annotations

import importlib.util
import json
import struct
import zlib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_notebooks.py"
NAMES = {
    "labelling_iride_heo.ipynb",
    "labelling_iride_eaglet2.ipynb",
    "labelling_phisat2.ipynb",
    "labels_iride_heo.ipynb",
    "labels_iride_eaglet2.ipynb",
    "labels_phisat2.ipynb",
    "labels_cloudsen12plus.ipynb",
    "labels_l8_biome.ipynb",
    "labels_l7_irish.ipynb",
    "labels_cloudperu2.ipynb",
    "label_tools.ipynb",
    "results_iride_heo.ipynb",
    "results_iride_eaglet2.ipynb",
    "results_phisat2.ipynb",
    "results_l7_irish.ipynb",
    "results_l8_biome.ipynb",
}
WRITES = ("fix", "save_label", "decide", "keep_labels", "export_for_editing")


def load_script():
    spec = importlib.util.spec_from_file_location("make_notebooks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(path: Path) -> str:
    """All the cells of a notebook, one after the other."""
    return "\n".join("".join(c["source"]) for c in json.loads(path.read_text())["cells"])


def test_notebooks(tmp_path, capsys):
    make = load_script()
    make.main(["--out", str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} == NAMES
    try:
        import nbformat
    except ImportError:
        nbformat = None
    for path in tmp_path.iterdir():
        book = json.loads(path.read_text())
        assert book["nbformat"] == 4 and book["metadata"]["kernelspec"]["name"] == "python3"
        for cell in book["cells"]:
            source = "".join(cell["source"])
            assert source and "__" not in source  # every gap of the templates filled
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None
                compile(source, path.name, "exec")
        if nbformat is not None:
            nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
    heo = text(tmp_path / "labelling_iride_heo.ipynb")
    assert 'drafts.Drafts(DATA, "iride_heo", draft="ocm4")' in heo
    assert 'label_fixes.FIXES["iride_heo"]' in heo
    assert "d.check(FIXES, show=False, pictures=figures)" in heo
    eaglet = text(tmp_path / "labelling_iride_eaglet2.ipynb")
    assert 'draft="ocm4_nirblue"' in eaglet and "labels_iride_eaglet2.ipynb" in eaglet
    for record in (heo, eaglet):  # the records only read, but for the one switch
        assert "WRITE = False" in record and record.count("d.fix(") == 1
        assert all(f"d.{w}(" not in record for w in WRITES if w != "fix")
        assert "d.apply(" not in record and "d.pages()" not in record
    tools = text(tmp_path / "label_tools.ipynb")
    for switch in ('NAME = ""', "PAGES = False", "WRITE = False", "KEEP = False"):
        assert switch in tools
    assert "d.compare(s, OTHER)" in tools and '"iride_eaglet2": "ocm4_nirzeros"' in tools
    heo_results = text(tmp_path / "results_iride_heo.ipynb")
    assert "('heo_test', 'heo_stress')" in heo_results and "corrected from" in heo_results
    l7_results = text(tmp_path / "results_l7_irish.ipynb")
    assert "('landsat7',)" in l7_results and "the comparison is even" in l7_results
    assert 'MAPS = Path("/root/catena_data/runs/big/maps")' in l7_results
    pictures = text(tmp_path / "labels_phisat2.ipynb")
    assert '"phisat2", draft="ocm4"' in pictures and "viz.sheet(items" in pictures
    assert "labelling_phisat2.ipynb" in pictures and "d.fix" not in pictures
    (tmp_path / "labelling_iride_heo.ipynb").write_text("my work")
    capsys.readouterr()
    make.main(["--out", str(tmp_path)])
    assert (tmp_path / "labelling_iride_heo.ipynb").read_text() == "my work"  # left alone
    assert "kept" in capsys.readouterr().out
    make.main(["--out", str(tmp_path), "--force"])
    assert (tmp_path / "labelling_iride_heo.ipynb").read_text() != "my work"


def tiny_png() -> bytes:
    """A valid PNG of one grey pixel."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    head = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    pixels = chunk(b"IDAT", zlib.compress(b"\x00\x80"))
    return b"\x89PNG\r\n\x1a\n" + head + pixels + chunk(b"IEND", b"")


class Reader:
    """Stands in for drafts.Drafts when a notebook runs here: two made-up scenes, A labelled
    and B not, and any method that writes fails the test."""

    def __init__(self, data, dataset, draft=None):
        self.dataset, self.draft = dataset, draft
        self.labels = Path(data) / "labels" / dataset
        self.reviews = {
            "A": {"scene": "A", "role": "test", "verdict": "correct"},
            "B": {"scene": "B", "role": "", "verdict": ""},
        }
        self.names = list(self.reviews)

    def __getattr__(self, name):
        if name in WRITES:
            raise AssertionError(f"Run All called d.{name}()")
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"{self.dataset}: two made-up scenes"

    def has_label(self, name: str) -> bool:
        return name == "A"

    def html(self) -> str:
        return "<table></table>"

    def reload(self) -> None:
        pass

    def pages(self) -> list:
        return []

    def check(self, fixes, show=True, pictures=None):
        assert not show  # the pictures come after the table
        if pictures is not None:
            pictures["A"] = tiny_png()
        row = {"scene": "A", "role": "test", "verdict": "correct", "differ": 0}
        return [row | {"pixels": 10, "changed": 0.5, "unlabelled": 0.0}]


def run_all(path: Path) -> None:
    """A notebook's code cells, one after the other, as Run All runs them."""
    namespace: dict = {}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), path.name, "exec"), namespace)


def test_run_all_writes_nothing(tmp_path, monkeypatch, capsys):
    pytest.importorskip("IPython")
    from catena import drafts

    monkeypatch.setattr(drafts, "Drafts", Reader)
    make = load_script()
    make.main(["--out", str(tmp_path)])
    capsys.readouterr()
    for name in sorted(NAMES):
        if name.startswith("labelling_") or name == "label_tools.ipynb":
            run_all(tmp_path / name)
    out = capsys.readouterr().out
    assert out.count("1 of 1 scenes changed by the review") == 3
    assert "1 scenes without label.tif: B" in out and "Put a scene's name in NAME" in out
