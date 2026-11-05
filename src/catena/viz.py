"""Pictures of images and their labels, the same for every dataset: true colour, the classes
drawn on it, and their shares.

    from catena import viz
    picture = viz.stretch(rgb, valid)          # (H, W, 3) in 0..1: a 2-98% stretch
    drawn = viz.overlay(picture, label)        # the classes drawn on it (Catena's codes)
    viz.describe(label)                        # "clear 71%, thin 4%, thick 20%, ..."
    viz.panels([picture, drawn], ["true colour", "label"])
    viz.sheet([(picture, drawn, "title"), ...])   # several scenes, two per row
    k = viz.step(label.shape); quick = picture[::k, ::k]    # a large scene, thinned out

Catena's codes: 0 clear, 1 thin cloud, 2 thick cloud, 3 cloud shadow, 4 cloud of unknown
thickness, 5 not cloud (clear or shadow), 255 no label. OmniCloudMask's drafts use 0 clear,
1 thick, 2 thin, 3 shadow; FROM_OCM turns them into Catena's. The colours are those of the
review pages: thin cloud yellow, thick cloud red, shadow blue, cloud of unknown thickness
orange; clear and not cloud are left as they are, and pixels without a label are washed grey.
"""

from __future__ import annotations

import html

import numpy as np

NAMES = {0: "clear", 1: "thin", 2: "thick", 3: "shadow", 4: "cloud", 5: "not cloud"}
NO_LABEL = 255
COLOURS = {1: (1.0, 0.85, 0.0), 2: (1.0, 0.1, 0.1), 3: (0.1, 0.4, 1.0), 4: (1.0, 0.5, 0.0)}
STRENGTH = 0.55  # how much of a class colour covers the picture
FROM_OCM = np.full(256, NO_LABEL, np.uint8)  # OmniCloudMask's codes to Catena's
FROM_OCM[[0, 1, 2, 3]] = [0, 2, 1, 3]


def step(shape: tuple, longest: int = 1200) -> int:
    """The smallest k for which every k-th row and column of an (H, W) array is at most
    `longest` across: a quick view of a large scene is a[::k, ::k]."""
    return max(1, -(-max(shape[:2]) // longest))


def stretch(
    rgb: np.ndarray,
    valid: np.ndarray | None = None,
    low: float = 2.0,
    high: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Red, green and blue (3, H, W), in any units, as an (H, W, 3) picture in 0..1: the low
    and high percentiles of the valid pixels, all three bands together so the colours keep
    their balance, become 0 and 1; a gamma below 1 then lifts the dark parts (dark sea, thin
    cloud over it). Pixels outside `valid` (default: a band above 0) are white."""
    picture = np.moveaxis(np.asarray(rgb, np.float32), 0, -1)
    if valid is None:
        valid = np.isfinite(picture).all(axis=-1) & (picture > 0).any(axis=-1)
    if valid.any():
        lo, hi = np.percentile(picture[valid], [low, high])
    else:
        lo, hi = 0.0, 1.0
    out = np.clip((picture - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    if gamma != 1.0:
        out **= gamma
    out[~valid] = 1.0
    return out


def overlay(picture: np.ndarray, label: np.ndarray, strength: float = STRENGTH) -> np.ndarray:
    """The classes of `label` (Catena's codes) drawn on an (H, W, 3) picture."""
    out = picture.copy()
    for code, colour in COLOURS.items():
        hit = label == code
        out[hit] = (1 - strength) * out[hit] + strength * np.array(colour, np.float32)
    gone = label == NO_LABEL
    out[gone] = 0.5 * out[gone] + 0.4
    return np.clip(out, 0.0, 1.0, out=out)  # rounding can step a hair past 1


def shares(label: np.ndarray) -> dict[str, float]:
    """Each class's share of the labelled pixels (the four main classes always, codes 4 and 5
    when present), and the labelled share of all pixels."""
    labelled = label != NO_LABEL
    n = max(int(labelled.sum()), 1)
    out = {}
    for code, name in NAMES.items():
        share = float((label == code).sum()) / n
        if code < 4 or share > 0:
            out[name] = share
    out["labelled"] = float(labelled.mean()) if label.size else 0.0
    return out


def describe(label: np.ndarray) -> str:
    """The shares on one line: clear 71%, thin 4%, thick 20%, shadow 5% (90% labelled)."""
    s = shares(label)
    labelled = s.pop("labelled")
    return ", ".join(f"{k} {v:.0%}" for k, v in s.items()) + f" ({labelled:.0%} labelled)"


def certainty(conf: np.ndarray) -> np.ndarray:
    """OmniCloudMask's probability of the class it chose, from a <draft>.conf.tif read as
    (4, H, W): probabilities x 250, 255 where there is no data (nan there)."""
    sure = conf.max(axis=0).astype(np.float32) / 250
    sure[(conf == NO_LABEL).any(axis=0)] = np.nan
    return sure


def _legend(ax, codes=(1, 2, 3)) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(color=COLOURS[c], label=NAMES[c]) for c in codes]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)


def _draw(ax, picture: np.ndarray, title: str, origin: tuple | None = None) -> None:
    """One picture; with origin (row, col) its axes count rows and columns from there."""
    extent = None
    if origin is not None:
        h, w = picture.shape[:2]
        extent = (origin[1] - 0.5, origin[1] + w - 0.5, origin[0] + h - 0.5, origin[0] - 0.5)
    if picture.ndim == 2:  # a map of numbers in 0..1, such as certainty
        shown = ax.imshow(
            picture, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", extent=extent
        )
        ax.figure.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
    else:
        ax.imshow(picture, interpolation="nearest", extent=extent)
    ax.set_title(title, fontsize=9)
    if origin is None:
        ax.set_axis_off()
    else:
        ax.tick_params(labelsize=7)


def panels(
    pictures,
    titles,
    size: float = 4.5,
    title: str | None = None,
    origin: tuple | None = None,
    show: bool = True,
):
    """Pictures side by side, with the class legend on the second; with origin (row, col)
    the axes give rows and columns, for edits. Returns the figure when show is False."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(pictures), figsize=(size * len(pictures), size))
    axes = np.atleast_1d(axes)
    for ax, picture, name in zip(axes, pictures, titles, strict=True):
        _draw(ax, picture, name, origin)
    _legend(axes[min(1, len(axes) - 1)])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _finish(fig, show)


def sheet(items, per_row: int = 2, size: float = 3.6, show: bool = True):
    """Several scenes on one figure, `per_row` to a row: each item is (picture, drawn,
    title), shown as the picture beside the same picture with its classes."""
    import matplotlib.pyplot as plt

    rows = max(1, -(-len(items) // per_row))
    fig, axes = plt.subplots(rows, 2 * per_row, figsize=(2 * per_row * size, rows * size))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_axis_off()
    for k, (picture, drawn, title) in enumerate(items):
        row, col = divmod(k, per_row)
        _draw(axes[row, 2 * col], picture, title)
        _draw(axes[row, 2 * col + 1], drawn, "")
    _legend(axes[0, 1])
    fig.tight_layout()
    return _finish(fig, show)


def _finish(fig, show: bool):
    if not show:
        return fig
    import matplotlib.pyplot as plt

    plt.show()
    return None


def html_table(rows: list[dict], columns: list[str]) -> str:
    """Rows as an HTML table (for IPython.display.HTML): shares as percentages."""

    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.0%}" if 0 <= value <= 1 else f"{value:.2f}"
        return html.escape(str(value))

    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell(r.get(c, ''))}</td>" for c in columns) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
