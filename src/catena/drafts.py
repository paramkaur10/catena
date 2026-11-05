"""The OmniCloudMask drafts of one dataset (scripts/ocm_prelabel.py), to review and correct
in a notebook (notebooks/review_<dataset>.ipynb).

    from catena import drafts
    d = drafts.Drafts(Path("/root/catena_data"), "iride_heo")   # or draft="ocm4_nirblue"
    HTML(d.html())                        # one row per scene: review state, the draft's shares
    s = d.scene("IMH07_20261007T084546")  # true colour, draft, certainty, label: one grid
    d.show(s)                             # true colour | draft | working label | certainty
    for row, col, sure in d.unsure(s):    # where OmniCloudMask was least sure
        d.zoom(s, row, col)
    s.edit(0, where=1)                    # e.g. its thin cloud is bright desert: clear
    d.save_label(s, who="Param")          # label.tif
    d.decide([dict(scene=s.name, role="train", verdict="keep")])   # review_<day>.csv

Layout, as ocm_prelabel.py and set_review.py leave it: DATA/labels/<dataset>/ holds
scenes.csv (one row per scene, with the review columns role, verdict, use, labeller, date,
notes), drafts.csv (the class shares of each scene's drafts), review_<draft>_<n>.png (the
review pages) and one folder per scene with <draft>.tif (OmniCloudMask's codes: 0 clear,
1 thick, 2 thin, 3 shadow, 255 no data), <draft>.conf.tif (its four probabilities x 250) and,
once reviewed, label.tif (Catena's codes: 0 clear, 1 thin, 2 thick, 3 shadow, 255 no label).
The products are DATA/raw/<dataset>/products/<product>.

The true colour is read as the drafts were made: HEO's B03, B02 and B01 files, or bands 1-3 of
Eaglet II's one image, brought onto the draft's grid as the mean of the valid pixels in each
cell; Phi-sat-2 on its own grid as top-of-atmosphere reflectance (catena.data.phisat2). So the
picture and the draft match pixel for pixel.

Verdicts: keep (the draft is right), all_clear (no cloud and no shadow), correct (it needs
correcting by hand), drop (not usable). Roles: train, test, stress. decide() writes decisions
to review_<day>.csv, which scripts/set_review.py puts into scenes.csv (and it makes label.tif
for all_clear scenes); keep_labels() then makes label.tif from the draft for kept scenes. A
scene to correct is edited here (Scene.edit, then save_label) or by hand in QGIS from the
files export_for_editing() writes; either way the result is its label.tif.
"""

from __future__ import annotations

import csv
import datetime
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from catena import viz
from catena.data import phisat2

ROLES = ("train", "test", "stress")
VERDICTS = ("keep", "all_clear", "correct", "drop")
DECISIONS = ("scene", "role", "verdict", "use", "notes")  # set_review.py's input
SHARES = ("valid", "clear", "thick", "thin", "shadow")  # drafts.csv's numbers
CLASSES = "0 clear, 1 thin cloud, 2 thick cloud, 3 cloud shadow, 255 no label"
CODES = (0, 1, 2, 3, viz.NO_LABEL)  # what a label may hold
GAMMA = 0.5  # lifts the dark parts of the true colour: sea, shadow, thin cloud over sea


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grid_of(path: Path) -> dict:
    """The pixel grid of a raster: crs, transform, width and height."""
    with rasterio.open(path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def band_paths(product: Path) -> dict[str, tuple[str, int]]:
    """GDAL paths (file, band) of red, green and blue in a HEO or Eaglet II product zip."""
    with zipfile.ZipFile(product) as z:
        names = z.namelist()
    root = f"/vsizip/{product.resolve()}/"
    heo = {b: n for n in names for b in ("B01", "B02", "B03") if n.endswith(f".{b}.tif")}
    if len(heo) == 3:
        return {
            k: (root + heo[b], 1) for k, b in (("red", "B03"), ("green", "B02"), ("blue", "B01"))
        }
    tif = next(n for n in sorted(names) if n.lower().endswith((".tif", ".tiff")))
    return {k: (root + tif, i) for k, i in (("red", 1), ("green", 2), ("blue", 3))}


def on_grid(path: str, band: int, grid: dict) -> np.ndarray:
    """One band on a draft's grid: the mean of the valid (non-zero) pixels in each cell."""
    with rasterio.open(path) as src:
        data = src.read(band)
        out = np.zeros((grid["height"], grid["width"]), np.float32)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=0,
            resampling=Resampling.average,
        )
    return out


def write(path: Path, data: np.ndarray, grid: dict, tags: dict | None = None) -> None:
    """uint8 bands (n, H, W), or one band (H, W), as a GeoTIFF on a grid; 255 = no data."""
    data = data if data.ndim == 3 else data[None]
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": data.shape[0],
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": viz.NO_LABEL,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8))
        if tags:
            dst.update_tags(**tags)


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass
class Scene:
    """One scene on its draft's grid. `work` is the label being made: label.tif if the scene
    has one, else the draft, with every edit applied; `edits` says what was changed."""

    name: str
    review: dict = field(repr=False)  # its row of scenes.csv
    rgb: np.ndarray = field(repr=False)  # (3, H, W) red, green, blue in the product's units
    draft: np.ndarray = field(repr=False)  # (H, W) the draft, in Catena's codes
    sure: np.ndarray = field(repr=False)  # (H, W) OmniCloudMask's certainty; nan: no data
    label: np.ndarray | None = field(default=None, repr=False)  # label.tif, if there is one
    work: np.ndarray = field(init=False, repr=False)
    edits: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.work = (self.draft if self.label is None else self.label).copy()

    @property
    def valid(self) -> np.ndarray:
        return self.draft != viz.NO_LABEL

    @cached_property
    def picture(self) -> np.ndarray:
        """True colour, (H, W, 3) in 0..1: a 2-98% stretch over the valid pixels, then a gamma
        of 0.5 that lifts the dark parts."""
        return viz.stretch(self.rgb, self.valid, gamma=GAMMA)

    @cached_property
    def brightness(self) -> np.ndarray:
        """(H, W) the mean of the picture's red, green and blue: 0 the scene's darkest, 1 its
        brightest, as seen in the true colour."""
        return self.picture.mean(axis=-1)

    def edit(
        self,
        to: int,
        box=None,
        where=None,
        below: float | None = None,
        brighter: float | None = None,
        darker: float | None = None,
    ) -> int:
        """Set pixels of the working label to class `to` (0 clear, 1 thin, 2 thick, 3 shadow,
        255 no label). Each condition given narrows where: inside box = (row0, row1, col0,
        col1); where the working label is `where` (a code or a list of codes); where
        OmniCloudMask's certainty is below `below`; where the brightness (0..1, as in the
        true colour) is above `brighter` or below `darker`. Never outside the image. Returns
        how many pixels changed; nothing is saved until Drafts.save_label()."""
        if to not in CODES:
            raise ValueError(f"class {to}: use one of {CODES}")
        hit = self.valid.copy()
        if box is not None:
            row0, row1, col0, col1 = box
            inside = np.zeros_like(hit)
            inside[row0:row1, col0:col1] = True
            hit &= inside
        if where is not None:
            hit &= np.isin(self.work, np.atleast_1d(where))
        if below is not None:
            hit &= np.nan_to_num(self.sure, nan=1.0) < below
        if brighter is not None:
            hit &= self.brightness > brighter
        if darker is not None:
            hit &= self.brightness < darker
        changed = int((self.work[hit] != to).sum())
        self.work[hit] = to
        rules = [
            f"box {box}" if box is not None else "",
            f"where {where}" if where is not None else "",
            f"certainty below {below}" if below is not None else "",
            f"brightness above {brighter}" if brighter is not None else "",
            f"brightness below {darker}" if darker is not None else "",
        ]
        rule = ", ".join(r for r in rules if r) or "everywhere"
        self.edits.append(f"{viz.NAMES.get(to, 'no label')} ({rule}): {changed} pixels")
        return changed

    def summary(self) -> str:
        r = self.review
        state = f"role {r.get('role') or '-'}, verdict {r.get('verdict') or '-'}"
        if r.get("labeller"):
            state += f", by {r['labeller']} on {r.get('date') or '?'}"
        lines = [f"{self.name}: {state}", f"  notes: {r.get('notes') or '-'}"]
        sure = self.sure[np.isfinite(self.sure)]
        mean = f"{sure.mean():.2f}" if sure.size else "-"
        lines.append(f"  draft: {viz.describe(self.draft)}; mean certainty {mean}")
        if self.label is not None:
            lines.append(f"  label.tif: {viz.describe(self.label)}")
        if self.edits:
            lines.append(f"  edits not yet saved: {'; '.join(self.edits)}")
        return "\n".join(lines)


class Drafts:
    """The drafts of one dataset, with their review state (see the module's docstring)."""

    def __init__(self, data: Path, dataset: str, draft: str | None = None):
        self.dataset = dataset
        self.labels = Path(data) / "labels" / dataset
        self.products = Path(data) / "raw" / dataset / "products"
        rows = read_rows(self.labels / "drafts.csv")
        self.kinds = sorted({r["draft"] for r in rows})
        self.draft = draft or (self.kinds[0] if self.kinds else "ocm4")
        if self.kinds and self.draft not in self.kinds:
            raise ValueError(f"no draft {self.draft!r} in drafts.csv: {', '.join(self.kinds)}")
        self.shares = {r["scene"]: r for r in rows if r["draft"] == self.draft}
        self.reload()

    def reload(self) -> None:
        """Read scenes.csv again, after set_review.py changed it."""
        self.reviews = {r["scene"]: r for r in read_rows(self.labels / "scenes.csv")}
        if not self.reviews:
            raise ValueError(f"no scenes in {self.labels / 'scenes.csv'}")
        self.names = list(self.reviews)

    def has_label(self, name: str) -> bool:
        return (self.labels / name / "label.tif").exists()

    def __repr__(self) -> str:
        verdicts = Counter(r.get("verdict") or "not reviewed" for r in self.reviews.values())
        state = ", ".join(f"{v} {k}" for k, v in sorted(verdicts.items()))
        made = sum(self.has_label(n) for n in self.names)
        return (
            f"{self.dataset}: {len(self.names)} scenes, draft {self.draft}; {state}; "
            f"{made} with label.tif"
        )

    def table(self) -> list[dict]:
        """One row per scene: its review columns, whether it has label.tif, the draft's shares."""
        rows = []
        for name in self.names:
            r = self.reviews[name]
            row = {k: r.get(k, "") for k in ("scene", "role", "verdict", "labeller", "notes")}
            row["label.tif"] = "yes" if self.has_label(name) else ""
            share = self.shares.get(name, {})
            row |= {k: float(share[k]) for k in SHARES if share.get(k, "") != ""}
            rows.append(row)
        return rows

    def html(self) -> str:
        columns = ["scene", "role", "verdict", "label.tif", *SHARES, "labeller", "notes"]
        return viz.html_table(self.table(), columns)

    def pages(self) -> list[Path]:
        """This draft's review pages (scripts/review_sheet.py), in order."""

        def number(path: Path) -> int:
            tail = path.stem.rsplit("_", 1)[-1]
            return int(tail) if tail.isdigit() else 0

        return sorted(self.labels.glob(f"review_{self.draft}_*.png"), key=number)

    def codes(self, name: str, draft: str | None = None) -> np.ndarray:
        """A scene's draft (this one or another) in Catena's codes."""
        with rasterio.open(self.labels / name / f"{draft or self.draft}.tif") as src:
            return viz.FROM_OCM[src.read(1)]

    def scene(self, name: str) -> Scene:
        """A scene, read: true colour on the draft's grid, the draft, OmniCloudMask's
        certainty and label.tif if it exists."""
        if name not in self.reviews:
            raise KeyError(f"{name} is not in {self.labels / 'scenes.csv'}")
        folder = self.labels / name
        path = folder / f"{self.draft}.tif"
        if not path.exists():
            raise FileNotFoundError(f"{path}: this scene has no {self.draft} draft")
        with rasterio.open(folder / f"{self.draft}.conf.tif") as src:
            sure = viz.certainty(src.read())
        label = None
        if self.has_label(name):
            with rasterio.open(folder / "label.tif") as src:
                label = src.read(1)
        rgb = self.true_colour(self.reviews[name]["product"], grid_of(path))
        return Scene(name, self.reviews[name], rgb, self.codes(name), sure, label)

    def true_colour(self, product_name: str, grid: dict) -> np.ndarray:
        """Red, green and blue of a product (3, H, W), on a draft's grid."""
        product = self.products / product_name
        if phisat2.is_product(product):
            rgb = phisat2.reflectance(phisat2.open_product(product), (4, 3, 2))
            if rgb.shape[1:] != (grid["height"], grid["width"]):
                raise ValueError(f"{product.name}: {rgb.shape[1:]} pixels, not the draft's grid")
            return rgb
        paths = band_paths(product)
        return np.stack([on_grid(*paths[k], grid) for k in ("red", "green", "blue")])

    def _panels(self, scene: Scene, cut, title: str, origin=None, show: bool = True):
        picture = scene.picture[cut]
        work = "label.tif" if scene.label is not None else "working label"
        if scene.edits:
            work += f", {len(scene.edits)} edits"
        pictures = [
            picture,
            viz.overlay(picture, scene.draft[cut]),
            viz.overlay(picture, scene.work[cut]),
            scene.sure[cut],
        ]
        titles = [
            "true colour",
            f"draft {self.draft}\n{viz.describe(scene.draft[cut])}",
            f"{work}\n{viz.describe(scene.work[cut])}",
            "OmniCloudMask's certainty",
        ]
        return viz.panels(pictures, titles, title=title, origin=origin, show=show)

    def show(self, scene: Scene, longest: int = 1400, show: bool = True):
        """The whole scene, every k-th pixel: true colour, the draft, the working label and
        OmniCloudMask's certainty."""
        k = viz.step(scene.draft.shape, longest)
        cut = (slice(None, None, k), slice(None, None, k))
        note = f" (every {k}th pixel)" if k > 1 else ""
        return self._panels(scene, cut, scene.name + note, show=show)

    def zoom(self, scene: Scene, row: int, col: int, size: int = 400, show: bool = True):
        """A window at full resolution; its axes give rows and columns, for edits."""
        cut = (slice(row, row + size), slice(col, col + size))
        title = f"{scene.name}: rows {row}-{row + size}, columns {col}-{col + size}"
        return self._panels(scene, cut, title, origin=(row, col), show=show)

    def compare(self, scene: Scene, other: str, longest: int = 1400, show: bool = True):
        """This draft beside another of the same scene (Eaglet II: NIR from blue or zeros)."""
        k = viz.step(scene.draft.shape, longest)
        picture = scene.picture[::k, ::k]
        drafts = [scene.draft[::k, ::k], self.codes(scene.name, other)[::k, ::k]]
        names = [self.draft, other]
        pictures = [picture, *(viz.overlay(picture, c) for c in drafts)]
        titles = [
            "true colour",
            *(f"{n}\n{viz.describe(c)}" for n, c in zip(names, drafts, strict=True)),
        ]
        return viz.panels(pictures, titles, title=scene.name, show=show)

    def unsure(self, scene: Scene, size: int = 400, n: int = 3, min_valid: float = 0.5):
        """The n windows of size x size pixels (side by side, not overlapping) where
        OmniCloudMask was least sure on average, among those at least min_valid with data:
        (row, col, mean certainty), least sure first."""
        found = []
        h, w = scene.sure.shape
        for row in range(0, max(h - size, 0) + 1, size):
            for col in range(0, max(w - size, 0) + 1, size):
                window = scene.sure[row : row + size, col : col + size]
                ok = np.isfinite(window)
                if ok.mean() >= min_valid:
                    found.append((row, col, float(window[ok].mean())))
        return sorted(found, key=lambda t: t[2])[:n]

    def save_label(self, scene: Scene, who: str = "", note: str = "") -> Path:
        """Write the scene's working label as its label.tif (replacing one that exists), with
        its edits, who and when in the file's tags."""
        folder = self.labels / scene.name
        source = "label.tif" if scene.label is not None else f"{self.draft}.tif"
        tags = {
            "classes": CLASSES,
            "source": source,
            "edits": " | ".join(scene.edits) or "none",
            "labeller": who,
            "note": note,
            "made_utc": now(),
        }
        out = folder / "label.tif"
        write(out, scene.work, grid_of(folder / f"{self.draft}.tif"), tags)
        scene.label, scene.edits = scene.work.copy(), []
        return out

    def decide(self, decisions: list[dict], day: str | None = None) -> Path:
        """Write decisions (dicts with scene and verdict; role, use and notes when wanted) to
        review_<day>.csv, today's by default, for scripts/set_review.py. The file's earlier
        rows are kept, and a later decision on a scene replaces an earlier one; if any
        decision is wrong, nothing is written."""
        day = day or datetime.date.today().isoformat()
        path = self.labels / f"review_{day}.csv"
        rows = {r["scene"]: r for r in read_rows(path)}
        for d in decisions:
            name = d.get("scene", "")
            if name not in self.reviews:
                raise ValueError(f"{name!r} is not in scenes.csv")
            role = d.get("role") or self.reviews[name].get("role", "")
            if role not in ROLES:
                raise ValueError(f"{name}: role {role!r} is not one of {', '.join(ROLES)}")
            verdict = d.get("verdict", "")
            if verdict not in VERDICTS:
                raise ValueError(f"{name}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
            use, notes = d.get("use", "label"), d.get("notes", "")
            rows[name] = {
                "scene": name,
                "role": role,
                "verdict": verdict,
                "use": use,
                "notes": notes,
            }
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(DECISIONS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows[k] for k in sorted(rows))
        return path

    def keep_labels(self) -> list[str]:
        """label.tif from the draft for every scene scenes.csv marks keep that has none yet;
        an existing label.tif is never touched. Returns the scenes done."""
        self.reload()
        done = []
        for name, review in self.reviews.items():
            if review.get("verdict") != "keep" or self.has_label(name):
                continue
            source = self.labels / name / f"{self.draft}.tif"
            with rasterio.open(source) as src:
                codes, made_by = src.read(1), src.tags().get("labeller", "OmniCloudMask")
            tags = {
                "classes": CLASSES,
                "source": f"{source.name}, kept on review ({made_by})",
                "made_utc": now(),
            }
            write(self.labels / name / "label.tif", viz.FROM_OCM[codes], grid_of(source), tags)
            done.append(name)
        return done

    def export_for_editing(self, scene: Scene) -> tuple[Path, Path]:
        """rgb.tif (the true colour in 8 bits, 255 = no data) and label_start.tif (the working
        label) on the draft's grid, to correct the scene by hand in QGIS; save the result as
        label.tif. Files of the same names are replaced."""
        folder = self.labels / scene.name
        grid = grid_of(folder / f"{self.draft}.tif")
        rgb = np.round(np.moveaxis(scene.picture, -1, 0) * 254).astype(np.uint8)
        rgb[:, ~scene.valid] = viz.NO_LABEL
        tags = {"bands": f"red, green, blue; 2-98% stretch, gamma {GAMMA}"}
        write(folder / "rgb.tif", rgb, grid, tags)
        write(folder / "label_start.tif", scene.work, grid, {"classes": CLASSES})
        return folder / "rgb.tif", folder / "label_start.tif"
