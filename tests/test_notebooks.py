"""The notebooks: scripts/make_notebooks.py writes valid notebooks whose code compiles, the
labelling records apart from the pictures, and leaves notebooks that exist alone unless
--force."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
}


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
    assert 'drafts.Drafts(DATA, "iride_heo", draft="ocm4")' in heo and "d.compare" not in heo
    assert 'label_fixes.FIXES["iride_heo"]' in heo and "d.check(FIXES)" in heo
    eaglet = text(tmp_path / "labelling_iride_eaglet2.ipynb")
    assert 'draft="ocm4_nirblue"' in eaglet and 'd.compare(s, "ocm4_nirzeros")' in eaglet
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
