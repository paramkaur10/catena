"""Write the notebooks into notebooks/, without outputs (run them in Jupyter):

  labelling_<dataset>.ipynb   how the labels of IRIDE HEO, IRIDE Eaglet II and Phi-sat-2 were
                              made: the review, the rules of every corrected scene
                              (catena.label_fixes), a check that makes every label again
                              from its OmniCloudMask draft, and the tools to label new scenes
  labels_<dataset>.ipynb      the labels themselves, as pictures: the three above and the
                              hand-labelled datasets CloudSEN12+, L8 Biome, L7 Irish and
                              CloudPeru2

    python scripts/make_notebooks.py [--out notebooks] [--force]

A notebook that exists is left alone (it may hold your work) unless --force. They read the
data from /root/catena_data (DATA in their first cell). Edit this script, not the
notebooks, to change them for good.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}

LABELLED = {
    "iride_heo": {
        "title": "IRIDE HEO",
        "draft": "ocm4",
        "other": None,
        "sensor": (
            "IRIDE HEO (Argotec HAWK), Level-1C top-of-atmosphere reflectance, 8 bands at "
            "2.5 m with the red edges at 5 m. The drafts and the labels are on a 5 m grid; the "
            "drafts were made from red B03, green B02 and near-infrared B07. HEO trains the "
            "networks (16 scenes) and tests them on scenes they never saw (4), and the three "
            "March 2026 products of processing version A02, whose red edges 1 and 3 read about "
            "0.3 of red edge 2, are a stress test."
        ),
    },
    "iride_eaglet2": {
        "title": "IRIDE Eaglet II",
        "draft": "ocm4_nirblue",
        "other": "ocm4_nirzeros",
        "sensor": (
            "IRIDE Eaglet II (OHB), Level-1C, three broad bands (red about 665-835 nm, green, "
            "blue) at about 1.9-2.6 m, labelled on a 5 m grid. It has no near-infrared band, "
            "so OmniCloudMask was given blue (ocm4_nirblue) or zeros (ocm4_nirzeros) in its "
            "place. Eaglet II is a test satellite only: no network trains on it."
        ),
    },
    "phisat2": {
        "title": "Phi-sat-2",
        "draft": "ocm4",
        "other": None,
        "sensor": (
            "Phi-sat-2 (ESA, Open Cosmos): 8 bands at about 4.75 m on a longitude-latitude "
            "grid, delivered as radiance from before the processor's reflectance step and "
            "turned into top-of-atmosphere reflectance by catena.data.phisat2; labelled on the "
            "product's own grid. A test satellite only: no network trains on it."
        ),
    },
}

COLOURS = (
    "**Colours:** thin cloud yellow, thick cloud red, shadow blue; grey has no label (no data, "
    "or left out on purpose, so that no test scores it)."
)

LABELLING_INTRO = (
    """\
# __TITLE__: how the labels were made

__SENSOR__

**How.** OmniCloudMask 1.8.0 (model 4.0) drafted every scene (`scripts/ocm_prelabel.py`).
Each draft was reviewed and given a role (`train`, `test` or `stress`) and a verdict:
`keep` (the draft is right), `all_clear` (no cloud and no shadow), `correct` (it needs
correcting) or `drop`. A scene to correct got rules: steps applied to its draft in order,
each acting on the pixels of some classes, brighter or darker than a level of the scene's
own brightness, and inside a box when one is given. The rules of every corrected scene are
recorded in `catena/label_fixes.py`, and the check below makes every label again from its
draft. Rules rather than painting: they are written down, they act the same way on every
pixel they name, and they leave the labels open to checking. The labels start from
OmniCloudMask's drafts, which a comparison with OmniCloudMask must bear in mind: the check
reports how much of each draft the review changed.

"""
    + COLOURS
)

LABELLING_SETUP = """\
from pathlib import Path

from IPython.display import HTML, Image, display

from catena import drafts, label_fixes, viz

DATA = Path("/root/catena_data")
REPO = Path.home() / "projects" / "catena"
d = drafts.Drafts(DATA, "__DATASET__", draft="__DRAFT__")
FIXES = label_fixes.FIXES["__DATASET__"]
print(d)
HTML(d.html())"""

RULES_TEXT = """\
## The rules, scene by scene

First the steps a rule can use, then every corrected scene with its steps and a note.
Brightness is the scene's own, 0 its darkest pixel and 1 its brightest, as in its true
colour; a box is (top, bottom, left, right) as fractions of the scene's height and width."""

RULES = """\
for step, meaning in drafts.STEPS.items():
    print(f"{step:<6} {meaning}")
print()
for name, (steps, note) in FIXES.items():
    role = d.reviews.get(name, {}).get("role", "-")
    print(f"{name} ({role}): {note}")
    for step in steps or ["the draft as it is"]:
        print("    ", step)"""

CHECK_TEXT = """\
## Check: every label made again from its draft

Each reviewed scene's label made again from its draft, by its verdict and its rules, and
compared with its `label.tif`. `differ` counts the pixels that are not the same: 0 means the
file is exactly what the record makes. `changed` is the share of the labelled pixels the
review changed from the draft, `unlabelled` the share it left without a label. The pictures
show the true colour, the draft and the label, with the boxes of the rules in cyan."""

CHECK = """\
rows = d.check(FIXES)
columns = ["scene", "role", "verdict", "differ", "pixels", "changed", "unlabelled"]
display(HTML(viz.html_table(rows, columns)))
for role in drafts.ROLES:
    mine = [r for r in rows if r["role"] == role]
    if mine:
        pixels = sum(r["pixels"] for r in mine)
        changed = sum(r["pixels"] * r["changed"] for r in mine) / max(pixels, 1)
        same = sum(r["differ"] == 0 for r in mine)
        print(
            f"{role}: {len(mine)} scenes, {same} the same as their label.tif; the review "
            f"changed {changed:.1%} of their {pixels:,} labelled pixels"
        )"""

REMAKE_TEXT = """\
## Writing the labels again

After a rule changes in `catena/label_fixes.py`, set `WRITE = True` and run this cell: every
corrected scene's `label.tif` is written again from its draft (the `all_clear` and `keep`
scenes are made by `set_review.py` and `keep_labels()`, below). Then run the check again."""

REMAKE = """\
WRITE = False
if WRITE:
    for name, (steps, note) in FIXES.items():
        d.fix(name, steps, note, who="Param")"""

TOOLS_TEXT = """\
# Tools for new scenes

The review pages show every scene's true colour beside its draft."""

PAGES = """\
for page in d.pages():
    display(Image(filename=str(page), width=1200))"""

SCENE_TEXT = """\
## One scene

`todo` lists the scenes still to do (not reviewed, or to correct and without `label.tif`);
`name` is the first of them, or put any scene's name from the table. Four pictures: the true
colour, the draft, the working label (`label.tif` if the scene has one, else the draft) and
how sure OmniCloudMask was (1 = sure)."""

SCENE = """\
todo = [n for n in d.names if d.reviews[n].get("verdict") in ("", "correct")]
todo = [n for n in todo if not d.has_label(n)]
print(len(todo), "scenes to do:", ", ".join(todo))
name = todo[0] if todo else d.names[0]
s = d.scene(name)
print(s.summary())
d.show(s)"""

ZOOM_TEXT = """\
## Closer: where OmniCloudMask was least sure

Windows of 400 x 400 pixels at full resolution. Their axes give rows and columns.
`d.zoom(s, row, col, size)` shows any other window."""

ZOOM = """\
for row, col, sure in d.unsure(s, size=400, n=3):
    print(f"rows {row}-{row + 400}, columns {col}-{col + 400}: mean certainty {sure:.2f}")
    d.zoom(s, row, col, size=400)"""

COMPARE_TEXT = """\
## The other draft

The same scene drafted with the near-infrared input filled the other way."""

COMPARE = 'd.compare(s, "__OTHER__")'

FIX_TEXT = """\
## Correcting a scene

Try rules on the scene with `d.apply(s, steps)` and `d.picture(s, boxes)`; when they are
right, `d.fix(name, steps, note, who=...)` writes `label.tif`. Then add the scene, its steps
and the note to `catena/label_fixes.py`, so that the record stays complete and the check
passes."""

FIX = """\
steps = [("cores", 0.85), ("trim", 0.3)]  # for example; the steps are in drafts.STEPS
boxes = d.apply(s, steps)
print("; ".join(s.edits))
d.picture(s, boxes)
# d.fix(s.name, steps, "what was wrong and what the rules do", who="Param")"""

DECIDE_TEXT = """\
## Verdicts

One line per scene you decide on. This cell writes them to `review_<today>.csv`; the next one
runs `scripts/set_review.py`, which puts them into `scenes.csv` (and makes `label.tif` for
`all_clear` scenes)."""

DECIDE = """\
decisions = [
    # dict(scene=name, role="test", verdict="correct", notes=""),
]
review = d.decide(decisions) if decisions else None
print(review)"""

SET_REVIEW = """\
import subprocess
import sys

if review:
    script = str(REPO / "scripts" / "set_review.py")
    command = [sys.executable, script, str(d.labels), str(review), "--who", "Param"]
    run = subprocess.run(command, capture_output=True, text=True)
    print(run.stdout, run.stderr)
    d.reload()
print(d)"""

KEEP_TEXT = """\
## Labels for the kept scenes

`label.tif` from the draft for every scene marked `keep` that has none yet (an existing
`label.tif` is never touched)."""

KEEP = 'print("label.tif made for:", d.keep_labels())'

PICTURES_INTRO = (
    """\
# __TITLE__: labels

__SENSOR__

The labels: OmniCloudMask drafts, reviewed and corrected by rules scene by scene; how, and
the rules of every scene, in `labelling___DATASET__.ipynb`. Below, every labelled scene by
role: `train` scenes are used in training, `test` scenes only in testing, `stress` scenes
test products with known faults. Each scene is shown every k-th pixel, about 900 across.

"""
    + COLOURS
)

PICTURES = """\
from pathlib import Path

import numpy as np
from IPython.display import HTML, display

from catena import drafts, viz

d = drafts.Drafts(Path("/root/catena_data"), "__DATASET__", draft="__DRAFT__")
rows = [r for r in d.table() if r["label.tif"]]
display(HTML(viz.html_table(rows, ["scene", "role", "verdict", "notes"])))
for role in drafts.ROLES:
    names = [r["scene"] for r in rows if r["role"] == role]
    if not names:
        continue
    counts, items = np.zeros(256, np.int64), []
    for name in names:
        s = d.scene(name)
        counts += np.bincount(s.label.ravel(), minlength=256)
        k = viz.step(s.label.shape, 900)
        picture = s.picture[::k, ::k]
        drawn = viz.overlay(picture, s.label[::k, ::k])
        items.append((picture, drawn, f"{name}\\n{viz.describe(s.label)}"))
    labelled = max(int(counts[:4].sum()), 1)
    shares = ", ".join(f"{viz.NAMES[c]} {counts[c] / labelled:.1%}" for c in range(4))
    print(f"{role}: {len(names)} scenes, {labelled:,} labelled pixels: {shares}")
    viz.sheet(items, per_row=2)"""

EXAMPLES = {
    "cloudsen12plus": (
        """\
# CloudSEN12+: Sentinel-2 labels

CloudSEN12+ (Aybar et al., 2024): Sentinel-2 L1C, 13 bands on a 10 m grid, expert-labelled
patches (10,000 of 509 x 509 pixels and 847 of 2,000 x 2,000) with four classes: clear, thick
cloud, thin cloud and cloud shadow. Catena reads them as 0 clear, 1 thin, 2 thick, 3 shadow
(`catena.data.loader`), with two traps fixed: pixels with no image get no label instead of
clear, and the +1000 offset of processing baseline 04.00 and later is removed. Below, eight
patches of the test split, drawn with a fixed seed. Colours: thin cloud yellow, thick cloud
red, shadow blue.""",
        """\
from pathlib import Path

import numpy as np

from catena import viz
from catena.data import loader

CHIPS = Path("/root/catena_data/chips/cloudsen12plus")
test = loader.patches(CHIPS, "test")
print(len(test), "test patches")
rng = np.random.default_rng(0)
items = []
for k in rng.choice(len(test), 8, replace=False):
    patch = test[k]
    image, label = loader.read(patch, 0, 0, size=min(patch.height, patch.width))
    picture = viz.stretch(image[[3, 2, 1]], label != viz.NO_LABEL)
    items.append((picture, viz.overlay(picture, label), f"{patch.id}\\n{viz.describe(label)}"))
viz.sheet(items, per_row=2)""",
    ),
    "l8_biome": (
        """\
# L8 Biome: Landsat 8 labels

L8 Biome (Foga et al., 2017): 96 Landsat 8 OLI scenes at 30 m in eight biomes, labelled by
hand: clear, thin cloud, thick cloud, and cloud shadow, which was drawn in only 32 of the 96
scenes (where shadows were easy to see). Catena's codes: 0 clear, 1 thin, 2 thick, 3 shadow.
Below, one scene per biome from the 24 test scenes (`pilot.csv`), about 900 pixels across.
Colours: thin cloud yellow, thick cloud red, shadow blue.""",
        """\
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from catena import viz
from catena.data import biome

ROOT = Path("/root/catena_data/chips/l8_biome")
chosen = biome.scenes(ROOT, "pilot")


def look(scene, longest=900):
    \"\"\"True colour (red B4, green B3, blue B2) and the label, every k-th pixel.\"\"\"
    with rasterio.open(scene.bands) as src:
        k = max(1, -(-max(src.height, src.width) // longest))
        shape = (src.height // k, src.width // k)
        dn = src.read([4, 3, 2], out_shape=(3, *shape), resampling=Resampling.nearest)
    with rasterio.open(scene.mask) as src:
        label = src.read(1, out_shape=shape, resampling=Resampling.nearest)
    return viz.stretch(dn.astype(np.float32), dn.all(axis=0)), label


items = []
for scene in chosen[::3]:
    picture, label = look(scene)
    shadow = "shadow labelled" if scene.has_shadow else "no shadow labels"
    title = f"{scene.name} ({scene.biome}, {shadow})\\n{viz.describe(label)}"
    items.append((picture, viz.overlay(picture, label), title))
viz.sheet(items, per_row=2)""",
    ),
    "l7_irish": (
        """\
# L7 Irish: Landsat 7 labels

L7 Irish (Irish et al., 2006; masks revised in 2015): 206 Landsat 7 ETM+ scenes at 30 m from
2000-2001 in nine latitude zones, labelled by hand: clear, thin cloud, thick cloud, and cloud
shadow (drawn in 43 scenes). `catena.data.irish` turns the DN into top-of-atmosphere
reflectance with each scene's gains. No network here trains on Landsat 7: it tests a camera
never seen. Below, one scene per zone from the 27 test scenes (`pilot.csv`). Colours: thin
cloud yellow, thick cloud red, shadow blue.""",
        """\
from pathlib import Path

import rasterio
from rasterio.enums import Resampling

from catena import viz
from catena.data import irish

ROOT = Path("/root/catena_data/chips/l7_irish")
chosen = irish.scenes(ROOT, "pilot")


def look(scene, longest=900):
    \"\"\"True colour (red B3, green B2, blue B1) as reflectance, and the label, every k-th
    pixel.\"\"\"
    with rasterio.open(scene.image) as src:
        k = max(1, -(-max(src.height, src.width) // longest))
        shape = (src.height // k, src.width // k)
        bands = list(irish.STACK)
        out = (len(bands), *shape)
        dn = src.read(bands, out_shape=out, resampling=Resampling.nearest)
    with rasterio.open(scene.mask) as src:
        label = irish.CODES[src.read(1, out_shape=shape, resampling=Resampling.nearest)]
    rho = irish.reflectance(scene, dn)
    return viz.stretch(rho[[2, 1, 0]], dn.all(axis=0)), label


items = []
for scene in chosen[::3]:
    picture, label = look(scene)
    title = f"{scene.zone} {scene.name}, gains {scene.gains}\\n{viz.describe(label)}"
    items.append((picture, viz.overlay(picture, label), title))
viz.sheet(items, per_row=2)""",
    ),
    "cloudperu2": (
        """\
# CloudPeru2: PeruSat-1 labels

CloudPeru2 (Morales et al., 2019): PeruSat-1 images at 2.8 m, four bands (blue, green, red,
near-infrared) as surface reflectance in 8 bits, cut into 512 x 512 patches, each with a cloud
mask: cloud or not cloud, with no thickness and no shadow. Every image came 8 times (rotations
and flips); one of each group of 8 is kept (2,800 of 22,400 patches), and 5% of the groups go
to validation. Catena reads cloud as 4 (cloud of unknown thickness) and the rest as 5 (not
cloud: clear or shadow), which the loss takes as either. Below, eight patches with 10-90%
cloud, drawn with a fixed seed. Colours: cloud orange.""",
        """\
import csv
import warnings
from pathlib import Path

import numpy as np
import rasterio

from catena import viz

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
ROOT = Path("/root/catena_data/chips/cloudperu2")
with open(ROOT / "index.csv", newline="") as f:
    rows = list(csv.DictReader(f))
cloudy = [r for r in rows if 0.1 <= float(r["cloud_frac"]) <= 0.9]
cloudy = [r for r in cloudy if float(r["nodata_frac"]) < 0.2]
print(len(rows), "patches;", len(cloudy), "with 10-90% cloud and under 20% without image")
rng = np.random.default_rng(0)
items = []
for k in rng.choice(len(cloudy), 8, replace=False):
    n = cloudy[k]["n"]
    with rasterio.open(ROOT / "images" / f"{n}.tif") as src:
        image = src.read().astype(np.float32)  # blue, green, red, near-infrared
    with rasterio.open(ROOT / "masks" / f"{n}_Mask.tif") as src:
        mask = src.read(1)
    label = np.where(mask == 255, 4, 5).astype(np.uint8)  # cloud, not cloud
    label[~image.any(axis=0)] = viz.NO_LABEL  # no image: no label
    picture = viz.stretch(image[[2, 1, 0]], label != viz.NO_LABEL)
    title = f"patch {n} ({cloudy[k]['split']})\\n{viz.describe(label)}"
    items.append((picture, viz.overlay(picture, label), title))
viz.sheet(items, per_row=2, legend=(4,))""",
    ),
}


def cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def notebook(cells: list[dict]) -> dict:
    metadata = {"kernelspec": KERNEL, "language_info": {"name": "python"}}
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}


def filler(dataset: str, spec: dict):
    def fill(text: str) -> str:
        for key, value in (
            ("__TITLE__", spec["title"]),
            ("__SENSOR__", spec["sensor"]),
            ("__DATASET__", dataset),
            ("__DRAFT__", spec["draft"]),
            ("__OTHER__", spec["other"] or ""),
        ):
            text = text.replace(key, value)
        return text

    return fill


def labelling(dataset: str, spec: dict) -> dict:
    """The record of how a dataset's labels were made, and the tools to label new scenes."""
    fill = filler(dataset, spec)
    cells = [
        cell("markdown", fill(LABELLING_INTRO)),
        cell("code", fill(LABELLING_SETUP)),
        cell("markdown", RULES_TEXT),
        cell("code", RULES),
        cell("markdown", CHECK_TEXT),
        cell("code", CHECK),
        cell("markdown", REMAKE_TEXT),
        cell("code", REMAKE),
        cell("markdown", TOOLS_TEXT),
        cell("code", PAGES),
        cell("markdown", SCENE_TEXT),
        cell("code", SCENE),
        cell("markdown", ZOOM_TEXT),
        cell("code", ZOOM),
    ]
    if spec["other"]:
        cells += [cell("markdown", COMPARE_TEXT), cell("code", fill(COMPARE))]
    cells += [
        cell("markdown", FIX_TEXT),
        cell("code", FIX),
        cell("markdown", DECIDE_TEXT),
        cell("code", DECIDE),
        cell("code", SET_REVIEW),
        cell("markdown", KEEP_TEXT),
        cell("code", KEEP),
    ]
    return notebook(cells)


def pictures(dataset: str, spec: dict) -> dict:
    """Every labelled scene of a dataset, as pictures."""
    fill = filler(dataset, spec)
    return notebook([cell("markdown", fill(PICTURES_INTRO)), cell("code", fill(PICTURES))])


def example(text: str, code: str) -> dict:
    return notebook([cell("markdown", text), cell("code", code)])


def build() -> dict[str, dict]:
    """Every notebook, by file name."""
    books = {f"labelling_{k}.ipynb": labelling(k, spec) for k, spec in LABELLED.items()}
    books |= {f"labels_{k}.ipynb": pictures(k, spec) for k, spec in LABELLED.items()}
    books |= {f"labels_{k}.ipynb": example(*parts) for k, parts in EXAMPLES.items()}
    return books


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("notebooks"))
    p.add_argument("--force", action="store_true", help="replace notebooks that exist")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, book in build().items():
        path = args.out / name
        if path.exists() and not args.force:
            print(f"kept    {path} (it exists; --force replaces it)")
            continue
        path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote   {path}")


if __name__ == "__main__":
    main()
