"""The record of the label corrections (catena.label_fixes): every step is one catena.drafts
knows, with values it can use, every box lies inside its scene, and no scene is listed twice."""

from __future__ import annotations

import re

from catena import label_fixes
from catena.drafts import CODES, STEPS

NAME = re.compile(r"^(IMH\d\d|IME\d\d|PHISAT2)_\d{8}T\d{6}$")
PREFIX = {"iride_heo": "IMH", "iride_eaglet2": "IME", "phisat2": "PHISAT2"}
LEVEL = {"cores", "pale", "trim", "grey", "lift", "thin", "thick", "dark"}


def test_the_record_is_well_formed():
    assert set(label_fixes.FIXES) == set(PREFIX)
    seen = set()
    for dataset, scenes in label_fixes.FIXES.items():
        for scene, (steps, note) in scenes.items():
            assert NAME.match(scene) and scene.startswith(PREFIX[dataset]), scene
            assert scene not in seen and note.strip(), scene
            seen.add(scene)
            for step in steps:
                name, *v = step
                assert name in STEPS, (scene, step)
                if v and isinstance(v[-1], tuple):
                    top, bottom, left, right = v.pop()
                    assert 0 <= top < bottom <= 1 and 0 <= left < right <= 1, (scene, step)
                if name in LEVEL:
                    assert len(v) == 1 and 0 < v[0] < 1, (scene, step)
                elif name == "haze":
                    assert len(v) == 2 and 0 < v[0] < v[1] <= 1, (scene, step)
                elif name == "set":
                    sources = v[0] if isinstance(v[0], list) else [v[0]]
                    assert len(v) == 2 and set(sources) <= set(CODES) and v[1] in CODES
                elif name == "start":
                    assert len(v) == 1 and v[0].startswith("ocm4"), (scene, step)
                else:  # frame and skip take no value
                    assert v == [], (scene, step)
    assert [len(s) for s in label_fixes.FIXES.values()] == [15, 17, 6]
