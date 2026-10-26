"""Write review decisions into a dataset's scenes.csv; make label.tif for all-clear scenes.

    python scripts/set_review.py LABELS DECISIONS.csv [--who NAME]

DECISIONS.csv has the columns scene, role, verdict, use, notes. Only those five columns of
scenes.csv change (labeller and date too, for scenes whose label is made here). A scene that
is not in scenes.csv, or a role or verdict outside the lists below, stops the script before
anything is written.

For verdict all_clear the label is made at once, on the grid of the scene's first draft:
0 (clear) wherever the draft has data, 255 (no label) elsewhere. Re-running is safe.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

ROLES = {"train", "test", "stress"}
VERDICTS = {"keep", "all_clear", "drop", "correct"}
ALLOWED = {(r, v) for r in ROLES for v in VERDICTS}
CODES = "0 clear, 1 thin cloud, 2 thick cloud, 3 cloud shadow, 255 no label"


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def all_clear_label(folder: Path) -> str:
    drafts = sorted(p for p in folder.glob("ocm4*.tif") if not p.name.endswith(".conf.tif"))
    if not drafts:
        raise SystemExit(f"no draft in {folder} to take the grid from")
    with rasterio.open(drafts[0]) as src:
        draft = src.read(1)
        profile = src.profile
    label = np.where(draft == 255, 255, 0).astype(np.uint8)
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(folder / "label.tif", "w", **profile) as dst:
        dst.write(label[None])
        dst.update_tags(source=f"all clear, footprint of {drafts[0].name}", codes=CODES)
    return drafts[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path, help="e.g. /root/catena_data/labels/iride_heo")
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--who", default="", help="name for the labeller column")
    args = parser.parse_args()

    register = read_rows(args.labels / "scenes.csv")
    columns = list(register[0])
    by_scene = {row["scene"]: row for row in register}
    decisions = read_rows(args.decisions)
    unknown = [d["scene"] for d in decisions if d["scene"] not in by_scene]
    wrong = [d["scene"] for d in decisions if (d["role"], d["verdict"]) not in ALLOWED]
    if unknown or wrong:
        raise SystemExit(
            f"nothing written. Not in scenes.csv: {unknown}; wrong role or verdict: {wrong}"
        )

    today = datetime.date.today().isoformat()
    for d in decisions:
        row = by_scene[d["scene"]]
        row.update(role=d["role"], verdict=d["verdict"], use=d["use"], notes=d["notes"])
        if d["verdict"] == "all_clear":
            source = all_clear_label(args.labels / d["scene"])
            row.update(labeller=args.who, date=today)
            print(f"{d['scene']}  label.tif made: all clear inside the footprint of {source}")

    with open(args.labels / "scenes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(register)
    counts = Counter(f"{d['role']} {d['verdict']}" for d in decisions)
    print(f"\n{len(decisions)} scenes updated in {args.labels / 'scenes.csv'}:")
    for key, n in sorted(counts.items()):
        print(f"   {key:16s} {n}")
    left = [row["scene"] for row in register if not row.get("verdict")]
    print(f"scenes still without a verdict: {left or 'none'}")


if __name__ == "__main__":
    main()
