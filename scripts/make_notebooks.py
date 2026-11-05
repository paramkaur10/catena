"""Write the label notebooks into notebooks/: one review notebook per set of OmniCloudMask
drafts (IRIDE HEO, IRIDE Eaglet II, Phi-sat-2), to check and correct them, and one example
notebook per hand-labelled dataset (CloudSEN12+, L8 Biome, L7 Irish, CloudPeru2), to show
their labels.

    python scripts/make_notebooks.py [--out notebooks] [--force]

The notebooks are written without outputs: run them in Jupyter. A notebook that exists is
left alone (it may hold your work) unless --force. They read the data from /root/catena_data
(DATA in their first cell). Edit this script, not the notebooks, to change them for good.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}

REVIEWS = {
    "iride_heo": {
        "title": "IRIDE HEO",
        "draft": "ocm4",
        "other": None,
        "about": (
            "IRIDE HEO (Argotec HAWK), Level-1C, 8 bands at 2.5 m (red edges at 5 m). The "
            "drafts were made at 5 m from red B03, green B02 and near-infrared B07. HEO is a "
            "training sensor: 16 scenes for training, 4 for testing, and the three A02 "
            "products of March as a stress test (their red edges B04 and B06 read about 0.3 "
            "of B05). Watch for bright desert called thin cloud (IMH02 over Algeria, 9 Oct)."
        ),
    },
    "iride_eaglet2": {
        "title": "IRIDE Eaglet II",
        "draft": "ocm4_nirblue",
        "other": "ocm4_nirzeros",
        "about": (
            "IRIDE Eaglet II (OHB), Level-1C, three broad bands (red about 665-835 nm, green, "
            "blue) at about 1.9-2.6 m, drafted at 5 m. It has no near-infrared band, so "
            "OmniCloudMask was given blue (ocm4_nirblue) or zeros (ocm4_nirzeros) in its "
            "place; with zeros it misses large cloud fields. Eaglet II is a test sensor, the "
            "camera no network trains on: its labels must be corrected, not only approved, "
            "or a comparison with OmniCloudMask would favour OmniCloudMask."
        ),
    },
    "phisat2": {
        "title": "Phi-sat-2",
        "draft": "ocm4",
        "other": None,
        "about": (
            "Phi-sat-2 (ESA, Open Cosmos): 8 bands at about 4.75 m on a longitude-latitude "
            "grid, delivered as radiance from before the processor's reflectance step and "
            "turned into top-of-atmosphere reflectance by catena.data.phisat2. The drafts sit "
            "on the product's own grid. A test sensor: check snow, which OmniCloudMask may "
            "call cloud, and correct the labels rather than only approve them."
        ),
    },
}

REVIEW_INTRO = """\
# __TITLE__: reviewing the OmniCloudMask drafts

__ABOUT__

**Colours:** thin cloud yellow, thick cloud red, shadow blue; grey has no data.

**Verdicts:** `keep` (the draft is right), `all_clear` (no cloud and no shadow), `correct`
(it needs correcting by hand), `drop` (not usable). **Roles:** `train`, `test`, `stress`.

**Steps:** look at the table and the review pages; open a scene and zoom where OmniCloudMask was
least sure; correct it here (`s.edit`, then `d.save_label`) or record a verdict; run
`set_review.py` to put the verdicts into `scenes.csv`; make the labels of the kept scenes."""

REVIEW_SETUP = """\
from pathlib import Path

from IPython.display import HTML, Image, display

from catena import drafts

DATA = Path("/root/catena_data")
REPO = Path.home() / "projects" / "catena"
d = drafts.Drafts(DATA, "__DATASET__", draft="__DRAFT__")
print(d)
HTML(d.html())"""

PAGES = """\
for page in d.pages():
    display(Image(filename=str(page), width=1200))"""

SCENE_TEXT = """\
## One scene

`todo` lists the scenes still to do (not reviewed, or to correct and without `label.tif`);
`name` is the first of them, or put any scene's name from the table. Four pictures: the true
colour, the draft, the working label (`label.tif` if the scene has one, else the draft, with
your edits) and how sure OmniCloudMask was (1 = sure)."""

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

Windows of 400 x 400 pixels at full resolution. Their axes give rows and columns, for edits.
`d.zoom(s, row, col, size)` shows any other window."""

ZOOM = """\
for row, col, sure in d.unsure(s, size=400, n=3):
    print(f"rows {row}-{row + 400}, columns {col}-{col + 400}: mean certainty {sure:.2f}")
    d.zoom(s, row, col, size=400)"""

COMPARE_TEXT = """\
## The other draft

The same scene drafted with the near-infrared input filled the other way."""

COMPARE = 'd.compare(s, "__OTHER__")'

EDIT_TEXT = """\
## Correcting a scene here

`s.edit(to, ...)` sets pixels of the working label to class `to` (0 clear, 1 thin, 2 thick,
3 shadow, 255 no label). Each condition you add narrows where: `where=` the classes to change
(one or a list), `box=(row0, row1, col0, col1)` read off the zoom axes, `below=` OmniCloudMask's
certainty under this, `brighter=` or `darker=` the pixel's brightness in the true colour (0 the
scene's darkest, 1 its brightest) above or under this. Look again with `d.show(s)` or
`d.zoom(...)`; nothing is saved until `d.save_label(s)`, which writes `label.tif` with the edits
in its tags. To start again, run the scene cell once more. To correct by hand instead,
`d.export_for_editing(s)` writes `rgb.tif` and `label_start.tif` for QGIS; save the corrected
file as `label.tif` in the scene's folder."""

EDIT = """\
# s.edit(1, where=2, darker=0.6)                      # dim "thick" you can see through: thin
# s.edit(2, where=3, brighter=0.5)                    # "shadow" inside a bright cloud: thick
# s.edit(0, where=3, brighter=0.3)                    # "shadow" that is not dark: clear
# s.edit(0, where=1)                                  # all its thin cloud is clear ground
# s.edit(0, box=(0, 800, 1200, 2000), where=[1, 2])   # no cloud in this box
# d.show(s)
# d.save_label(s, who="Param", note="")"""

DECIDE_TEXT = """\
## Verdicts

One line per scene you decide on. This cell writes them to `review_<today>.csv`; the next one
runs `scripts/set_review.py`, which puts them into `scenes.csv` (and makes `label.tif` for
`all_clear` scenes)."""

DECIDE = """\
decisions = [
    # dict(scene=name, role="train", verdict="keep", notes=""),
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


def review(dataset: str, spec: dict) -> dict:
    def fill(text: str) -> str:
        for key, value in (
            ("__TITLE__", spec["title"]),
            ("__ABOUT__", spec["about"]),
            ("__DATASET__", dataset),
            ("__DRAFT__", spec["draft"]),
            ("__OTHER__", spec["other"] or ""),
        ):
            text = text.replace(key, value)
        return text

    cells = [
        cell("markdown", fill(REVIEW_INTRO)),
        cell("code", fill(REVIEW_SETUP)),
        cell(
            "markdown",
            "## All scenes at a glance\n\nThe review pages: true colour beside the draft.",
        ),
        cell("code", PAGES),
        cell("markdown", SCENE_TEXT),
        cell("code", SCENE),
        cell("markdown", ZOOM_TEXT),
        cell("code", ZOOM),
    ]
    if spec["other"]:
        cells += [cell("markdown", COMPARE_TEXT), cell("code", fill(COMPARE))]
    cells += [
        cell("markdown", EDIT_TEXT),
        cell("code", EDIT),
        cell("markdown", DECIDE_TEXT),
        cell("code", DECIDE),
        cell("code", SET_REVIEW),
        cell("markdown", KEEP_TEXT),
        cell("code", KEEP),
    ]
    return notebook(cells)


def example(text: str, code: str) -> dict:
    return notebook([cell("markdown", text), cell("code", code)])


def build() -> dict[str, dict]:
    """Every notebook, by file name."""
    books = {f"review_{k}.ipynb": review(k, spec) for k, spec in REVIEWS.items()}
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
