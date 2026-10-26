"""Review pages: every scene's draft on a few pictures, to judge them all at a glance.

    python review_sheet.py LABELS_FOLDER DRAFT [--per-page 8]

Reads LABELS_FOLDER/<scene>/<DRAFT>.png, made by ocm_prelabel.py, and the class shares in
LABELS_FOLDER/drafts.csv. Writes LABELS_FOLDER/review_<DRAFT>_<page>.png: two scenes per
row, each its true-colour picture beside the same picture with the classes drawn on
(red thick cloud, yellow thin cloud, blue shadow), titled with the scene and its shares.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ("clear", "thick", "thin", "shadow")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, help="labels folder of one dataset")
    parser.add_argument("draft", help="draft name, e.g. ocm4 or ocm4_nirblue")
    parser.add_argument("--per-page", type=int, default=8, help="scenes per picture")
    args = parser.parse_args()
    shares = {}
    with open(args.labels / "drafts.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["draft"] == args.draft:
                shares[row["scene"]] = row
    pictures = sorted(args.labels.glob(f"*/{args.draft}.png"))
    if not pictures:
        raise SystemExit(f"no {args.draft}.png in the scene folders of {args.labels}")
    per = args.per_page
    for page in range(math.ceil(len(pictures) / per)):
        chunk = pictures[page * per : (page + 1) * per]
        rows = math.ceil(len(chunk) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, path in zip(axes.flat, chunk, strict=False):
            scene = path.parent.name
            ax.imshow(plt.imread(path))
            row = shares.get(scene, {})
            text = "  ".join(f"{n} {float(row[n]):.0%}" for n in NAMES if row.get(n))
            ax.set_title(f"{scene}\n{text}", fontsize=10)
        fig.tight_layout()
        out = args.labels / f"review_{args.draft}_{page + 1}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
