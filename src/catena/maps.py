"""What the networks predicted, scene by scene, beside the true colour and the label: the files
catena.evaluate_clarity --maps keeps, one per scene (<set>__<scene>.npz) with the label,
every network's classes (seed0_real ... seed2_sim, and seed0_chain ... for the arms that test
the method), OmniCloudMask's where it ran (ocm4, ocm4_nirblue, ocm4_nirzeros), the true
colour (rgb, every step-th pixel) and step.

    from catena import maps
    kept = maps.Kept(Path("/root/catena_data/runs/big/maps"), "eaglet2")
    HTML(viz.html_table(kept.table(), maps.COLUMNS))  # per scene: cloud labelled, cloud called
    for scene in kept.scenes:
        kept.show(scene)  # true colour | label | OmniCloudMask | Clarity real | Clarity sim

An arm's map is the class most of its seeds chose at each pixel; when the seeds all differ,
the lowest code wins (clear, then thin, thick, shadow). Cloud is thin or thick (label codes
1, 2 and, for CloudPeru2, 4); balanced accuracy is that of cloud against not cloud over the
labelled pixels, as in the tests (catena.evaluate_clarity.scores).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from catena import evaluate_clarity as E
from catena import viz

NO_LABEL = viz.NO_LABEL
CLOUD_CODES = (1, 2, 4)  # label codes that are cloud
COLUMNS = ["scene", "labelled", "cloud"]  # then "<model> cloud" and "<model> BA" per model


def vote(maps: list[np.ndarray]) -> np.ndarray:
    """The class most maps give each pixel (ties to the lowest code); 255 where none has one."""
    stack = np.stack(maps)
    counts = np.stack([(stack == c).sum(axis=0) for c in range(4)])
    out = counts.argmax(axis=0).astype(np.uint8)
    out[counts.sum(axis=0) == 0] = NO_LABEL
    return out


def name_of(model: str) -> str:
    """ocm4_nirblue -> OmniCloudMask, NIR from blue."""
    return E.OCM.get(model, model)


class Kept:
    """The kept predictions of one test set (heo_test, eaglet2, landsat7, ...)."""

    def __init__(self, folder: Path, name: str) -> None:
        self.folder, self.name = Path(folder), name
        self.files = sorted(self.folder.glob(f"{name}__*.npz"))
        self.scenes = [f.stem.split("__", 1)[1] for f in self.files]

    def __repr__(self) -> str:
        return f"{self.name}: {len(self.scenes)} scenes kept in {self.folder}"

    def load(self, scene: str) -> dict[str, np.ndarray]:
        with np.load(self.folder / f"{self.name}__{scene}.npz") as z:
            return {k: z[k] for k in z.files}

    def models(self, arrays: dict) -> dict[str, np.ndarray]:
        """Every arm kept (Clarity real, sim, ..., each the vote of its seeds), then
        OmniCloudMask's maps."""
        out = {}
        for arm in E.ARMS:
            longer = [a for a in E.ARMS if a.endswith(f"_{arm}")]  # chain_real is not real
            seeds = [
                arrays[k]
                for k in sorted(arrays)
                if k.endswith(f"_{arm}") and not any(k.endswith(f"_{a}") for a in longer)
            ]
            if seeds:
                out[E.ARM_TITLES[arm]] = vote(seeds)
        out |= {name_of(k): arrays[k] for k in E.OCM if k in arrays}
        return out

    def row(self, scene: str) -> dict:
        """How much of the scene is labelled cloud, and how much each model calls cloud, with
        its balanced accuracy, over the labelled pixels."""
        arrays = self.load(scene)
        label = arrays["label"]
        labelled = label != NO_LABEL
        n = max(int(labelled.sum()), 1)
        out = {"scene": scene, "labelled": int(labelled.sum())}
        out["cloud"] = float(np.isin(label[labelled], CLOUD_CODES).sum()) / n
        for model, pred in self.models(arrays).items():
            called = np.isin(pred[labelled], (1, 2)).sum() / n
            out[f"{model} cloud"] = float(called)
            out[f"{model} BA"] = E.scores(E.confusion(label, pred))["cloud_ba"]
        return out

    def table(self) -> list[dict]:
        """Every scene's row, balanced accuracies written 0.xxx (shares stay shares)."""
        rows = [self.row(scene) for scene in self.scenes]
        for r in rows:
            r |= {k: f"{v:.3f}" for k, v in r.items() if k.endswith(" BA")}
        return rows

    def columns(self) -> list[str]:
        """The table's columns: those of COLUMNS, then each model's cloud and BA."""
        if not self.scenes:
            return COLUMNS
        names = list(self.models(self.load(self.scenes[0])))
        return COLUMNS + [f"{m} cloud" for m in names] + [f"{m} BA" for m in names]

    def show(self, scene: str, size: float = 3.6, show: bool = True):
        """The true colour, then the label and every model's classes drawn on it."""
        arrays = self.load(scene)
        k = int(arrays["step"])
        picture = arrays["rgb"].astype(np.float32) / 255
        label = arrays["label"][::k, ::k]
        labelled = label != NO_LABEL
        n = max(int(labelled.sum()), 1)
        cloud = np.isin(label[labelled], CLOUD_CODES).sum() / n
        pictures = [picture, viz.overlay(picture, label)]
        titles = ["true colour", f"label: cloud {cloud:.0%}"]
        for model, pred in self.models(arrays).items():
            small = pred[::k, ::k]
            called = np.isin(small[labelled], (1, 2)).sum() / n
            ba = E.scores(E.confusion(arrays["label"], pred))["cloud_ba"]
            pictures.append(viz.overlay(picture, np.where(labelled, small, NO_LABEL)))
            titles.append(f"{model}: cloud {called:.0%}, BA {ba:.2f}")
        return viz.panels(pictures, titles, size, title=scene, legend=(1, 2, 3), show=show)
