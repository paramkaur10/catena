"""A floor for the tables: what thresholds alone do. Classic cloud masks test each pixel's
brightness, whiteness and haze against thresholds; this is the best such rule a few thresholds
make, fitted on the validation cameras and scored beside the networks (catena.evaluate_clarity
--threshold). A network that beats it by little has learned little more than thresholds; its
margin on thin cloud, on shadow and on bright ground (L8 Biome's snow, ice and barren scenes)
says how much more.

A pixel's features, from its bands' wavelengths alone (the chain card), so every camera with
visible bands has them, whatever its bands:
  brightness  the mean reflectance of the visible bands (400-700 nm)
  whiteness   the mean of |band - brightness| / brightness over them (Fmask's whiteness)
  haze        blue - 0.5 red - 0.08, blue and red the visible bands nearest 490 and 665 nm
              (Fmask's haze optimised transform)
  nir         the mean reflectance of the near infrared bands (760-1000 nm)
  ndvi        (nir - red) / (nir + red)
The rule is a decision tree of depth 4 (at most 15 thresholds), grown by Gini impurity with
each class weighted by the inverse of its share of the pixels (so clear, the most common, does
not win every leaf), on PER_WINDOW pixels of each window of the CloudSEN12+ validation split
as the six validation cameras see it (catena.train.ValSamples: never test data). A camera
without near infrared gets a second tree, grown on the first three features of the same
pixels. Per pixel, so moving the pixels about changes nothing (evaluate_clarity's "pixels
scrambled" check).

    python -m catena.threshold OUT [--data /root/catena_data] [--windows 120] [--depth 4]

writes OUT/threshold.json (both trees, each node a feature and a threshold or a class) and
prints the trees and their scores on the pixels they were grown on.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from catena import descriptor as desc

FEATURES = ("brightness", "whiteness", "haze", "nir", "ndvi")
VISIBLE_ONLY = 3  # the features of a camera without near infrared
VISIBLE_NM, NIR_NM = (400.0, 700.0), (760.0, 1000.0)
BLUE_NM, RED_NM = 490.0, 665.0
CLASSES = ("clear", "thin", "thick", "shadow")  # Catena's codes 0-3
PER_WINDOW = 4000
BINS = 64  # candidate thresholds per feature: its quantiles
MIN_LEAF = 200  # pixels at least on each side of a threshold
SIZE = 240  # the validation windows, in 10 m pixels, as the search and training validate


def wavelengths(told: np.ndarray) -> np.ndarray:
    """Each band's middle wavelength in nm, from its chain card (bands, 11): the card's mid
    field is log(nm - 300) - 2 (catena.descriptor.wavelength_code)."""
    return 300.0 + np.exp(np.asarray(told, np.float64)[..., 1] + 2.0)


def roles(told: np.ndarray, real=None) -> dict | None:
    """Which bands are visible, near infrared, blue and red, by their wavelengths (real: the
    slots that hold a band); None without a visible band."""
    nm = wavelengths(told)
    ok = np.ones(len(nm), bool) if real is None else np.asarray(real, bool)
    visible = np.flatnonzero(ok & (nm >= VISIBLE_NM[0]) & (nm < VISIBLE_NM[1]))
    if len(visible) == 0:
        return None
    nir = np.flatnonzero(ok & (nm >= NIR_NM[0]) & (nm <= NIR_NM[1]))
    blue = int(visible[np.argmin(np.abs(nm[visible] - BLUE_NM))])
    red = int(visible[np.argmin(np.abs(nm[visible] - RED_NM))])
    return {"visible": visible, "nir": nir, "blue": blue, "red": red}


def features(rho: np.ndarray, r: dict) -> np.ndarray:
    """The features (5, or 3 without near infrared) of every pixel of rho (bands, ...), the
    bands' reflectance; float32, (features, ...)."""
    vis = np.asarray(rho[r["visible"]], np.float32)
    bright = vis.mean(axis=0)
    white = np.abs(vis - bright).mean(axis=0) / np.maximum(bright, 1e-3)
    red = np.asarray(rho[r["red"]], np.float32)
    haze = np.asarray(rho[r["blue"]], np.float32) - 0.5 * red - 0.08
    out = [bright, white, haze]
    if len(r["nir"]):
        nir = np.asarray(rho[r["nir"]], np.float32).mean(axis=0)
        out += [nir, (nir - red) / np.maximum(nir + red, 1e-3)]
    return np.stack(out).astype(np.float32)


# --- the tree -------------------------------------------------------------------------------


def grow(x: np.ndarray, y: np.ndarray, depth: int, bins: int = BINS, least: int = MIN_LEAF):
    """A decision tree on x (n, features) for the classes y (n,) 0-3: Gini impurity, each class
    weighted by the inverse of its share; thresholds among each feature's quantiles. A list of
    nodes, the root first: {"feature", "threshold", "left", "right"} or {"class"}."""
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    weight = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)[y]
    edges = [
        np.unique(np.quantile(x[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        for j in range(x.shape[1])
    ]
    # code k: the number of edges below the value, so value <= edges[k] exactly when code <= k
    code = np.stack([np.searchsorted(e, x[:, j], side="left") for j, e in enumerate(edges)], 1)
    nodes: list[dict] = []

    def leaf(where) -> int:
        share = np.bincount(y[where], weights=weight[where], minlength=len(CLASSES))
        nodes.append({"class": int(np.argmax(share))})
        return len(nodes) - 1

    def split(where, level: int) -> int:
        if level == depth or len(np.unique(y[where])) < 2:
            return leaf(where)
        best = None
        for j, e in enumerate(edges):
            if len(e) == 0:
                continue
            pairs = code[where, j] * len(CLASSES) + y[where]
            size = (len(e) + 1) * len(CLASSES)
            hist = np.bincount(pairs, weights=weight[where], minlength=size).reshape(
                -1, len(CLASSES)
            )
            many = np.bincount(code[where, j], minlength=len(e) + 1)
            left, n_left = np.cumsum(hist, 0)[:-1], np.cumsum(many)[:-1]
            right, n_right = hist.sum(0) - left, len(where) - n_left
            wl, wr = left.sum(1), right.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                gini = wl * (1 - ((left / wl[:, None]) ** 2).sum(1))
                gini += wr * (1 - ((right / wr[:, None]) ** 2).sum(1))
            gini[(n_left < least) | (n_right < least) | ~np.isfinite(gini)] = np.inf
            k = int(np.argmin(gini))
            if np.isfinite(gini[k]) and (best is None or gini[k] < best[0]):
                best = (gini[k], j, k)
        if best is None:
            return leaf(where)
        _, j, k = best
        here = len(nodes)
        nodes.append({"feature": j, "threshold": float(edges[j][k])})
        go = code[where, j] <= k
        nodes[here]["left"] = split(where[go], level + 1)
        nodes[here]["right"] = split(where[~go], level + 1)
        return here

    split(np.arange(len(y)), 0)
    return pruned(nodes)


def pruned(tree: list[dict]) -> list[dict]:
    """The same rule without a threshold whose two sides give the same class."""
    out: list[dict] = []

    def copy(node: int) -> dict:
        n = tree[node]
        if "class" in n:
            return dict(n)
        left, right = copy(n["left"]), copy(n["right"])
        if "class" in left and "class" in right and left["class"] == right["class"]:
            return dict(left)
        return {"feature": n["feature"], "threshold": n["threshold"], "left": left, "right": right}

    def flat(n: dict) -> int:
        here = len(out)
        out.append({k: v for k, v in n.items() if k not in ("left", "right")})
        if "left" in n:
            out[here]["left"] = flat(n["left"])
            out[here]["right"] = flat(n["right"])
        return here

    flat(copy(0))
    return out


def decide(tree: list[dict], x: np.ndarray) -> np.ndarray:
    """The tree's classes (uint8) for features x (features, ...)."""
    out = np.zeros(x.shape[1:], np.uint8)

    def walk(node: int, mask: np.ndarray) -> None:
        n = tree[node]
        if "class" in n:
            out[mask] = n["class"]
            return
        go = x[n["feature"]] <= n["threshold"]
        walk(n["left"], mask & go)
        walk(n["right"], mask & ~go)

    walk(0, np.ones(x.shape[1:], bool))
    return out


def lines(tree: list[dict], names=FEATURES) -> list[str]:
    """The tree as indented rules."""
    out: list[str] = []

    def walk(node: int, indent: str) -> None:
        n = tree[node]
        if "class" in n:
            out.append(f"{indent}-> {CLASSES[n['class']]}")
            return
        out.append(f"{indent}{names[n['feature']]} <= {n['threshold']:.4g}:")
        walk(n["left"], indent + "    ")
        out.append(f"{indent}else:")
        walk(n["right"], indent + "    ")

    walk(0, "  ")
    return out


@dataclass
class Rule:
    """The fitted thresholds: a tree for cameras with near infrared, one for those without."""

    trees: dict

    def classes(self, rho: np.ndarray, told: np.ndarray, real=None) -> np.ndarray:
        """Classes (uint8, Catena's codes) of every pixel of rho (bands, ...), the reflectance
        of the bands the chain cards told (bands, 11) describe; all clear without a visible
        band."""
        r = roles(told, real)
        if r is None:
            return np.zeros(rho.shape[1:], np.uint8)
        tree = self.trees["vnir"] if len(r["nir"]) else self.trees["vis"]
        return decide(tree, features(rho, r))


def load(path: Path) -> Rule:
    return Rule(json.loads(Path(path).read_text())["trees"])


# --- fitting --------------------------------------------------------------------------------


def samples(data: Path, windows: int, size: int, workers: int, seed: int = 0):
    """Features (n, 5) and classes (n,) of PER_WINDOW labelled pixels drawn from each window
    of the validation cameras."""
    from torch.utils.data import DataLoader

    from catena import train as T
    from catena.data import loader

    checks = T.ValSamples(
        loader.GridWindows(data / "chips" / "cloudsen12plus", "validation", size=size), windows
    )
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for s in DataLoader(checks, batch_size=None, num_workers=workers, worker_init_fn=T.quiet):
        y = s["label"].numpy().ravel()
        r = roles(desc.chain_card(s["d_told"].numpy()), s["real"].numpy())
        if r is None or not len(r["nir"]):
            continue
        f = features(s["image"].numpy(), r).reshape(len(FEATURES), -1)
        ok = np.flatnonzero(y < len(CLASSES))
        pick = rng.choice(ok, size=min(PER_WINDOW, len(ok)), replace=False)
        xs.append(f[:, pick].T)
        ys.append(y[pick].astype(np.int64))
    if not xs:
        raise SystemExit("no labelled pixel in the validation windows")
    return np.concatenate(xs), np.concatenate(ys)


def scored(tree: list[dict], x: np.ndarray, y: np.ndarray) -> str:
    """The F1 of each class and their mean, on the pixels grown on."""
    pred = decide(tree, x.T)
    f1 = []
    for c in range(len(CLASSES)):
        tp, both = np.sum((pred == c) & (y == c)), np.sum(pred == c) + np.sum(y == c)
        f1.append(2 * tp / both if both else np.nan)
    parts = [f"F1 {n} {v:.3f}" for n, v in zip(CLASSES, f1, strict=True)]
    return ", ".join(parts) + f"; mean {np.nanmean(f1):.3f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="the folder of threshold.json")
    p.add_argument("--data", type=Path, default=Path("/root/catena_data"))
    p.add_argument("--windows", type=int, default=120, help="per validation camera")
    p.add_argument("--size", type=int, default=SIZE, help="window size, 10 m pixels")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    began = time.perf_counter()
    x, y = samples(args.data, args.windows, args.size, args.workers)
    counts = np.bincount(y, minlength=len(CLASSES))
    print(
        f"{len(y):,} pixels: "
        + ", ".join(f"{n} {c:,}" for n, c in zip(CLASSES, counts, strict=True))
    )
    trees = {"vnir": grow(x, y, args.depth), "vis": grow(x[:, :VISIBLE_ONLY], y, args.depth)}
    for name, tree in trees.items():
        cols = x if name == "vnir" else x[:, :VISIBLE_ONLY]
        print(f"\n{name} ({scored(tree, cols, y)}):\n" + "\n".join(lines(tree)))
    args.out.mkdir(parents=True, exist_ok=True)
    found = {"features": FEATURES, "classes": CLASSES, "depth": args.depth, "trees": trees}
    found |= {"fitted_on": {"windows_per_camera": args.windows, "size": args.size}}
    found |= {"pixels": {n: int(c) for n, c in zip(CLASSES, counts, strict=True)}}
    (args.out / "threshold.json").write_text(json.dumps(found, indent=1) + "\n")
    print(f"\nwrote {args.out / 'threshold.json'} in {(time.perf_counter() - began) / 60:.1f} min")


if __name__ == "__main__":
    main()
