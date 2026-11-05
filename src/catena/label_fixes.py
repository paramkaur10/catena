"""How the labels of IRIDE HEO, IRIDE Eaglet II and Phi-sat-2 were made from the OmniCloudMask
drafts (scripts/ocm_prelabel.py): the rules each corrected scene got, as steps applied to its
draft in order (catena.drafts.STEPS).

    FIXES[dataset][scene] = (steps, note)

A scene reviewed as correct is its draft with its steps; one reviewed as all_clear is clear
wherever its draft has data (scripts/set_review.py); one kept is its draft. Drafts.check()
makes every label again this way and compares it with its label.tif, so the files and this
record cannot drift apart (notebooks/labelling_<dataset>.ipynb, section Check).

Brightness is the scene's own: the mean of red, green and blue after a 2-98% stretch over
the scene and a gamma of 0.5 (catena.drafts.Scene.brightness), 0 its darkest pixel and 1 its
brightest. A box is (top, bottom, left, right) as fractions of the scene's height and width.
Labelled by Param (Parampuneet Kaur Thind), late October to 5 November 2026: HEO's training
scenes first, then the test and stress scenes of the three satellites, in two rounds; this
record holds the last rules of each scene.
"""

from __future__ import annotations

THIN_EDGES = "dim thick at the edges made thin"
CORES = "bright cores the draft left clear made thick"

FIXES = {
    "iride_heo": {
        # training scenes
        "IMH03_20261007T121948": ([("thin", 0.6)], "cloud over the sea: dim thick made thin"),
        "IMH03_20261007T121953": ([("thin", 0.6)], "overcast over the sea: dim thick made thin"),
        "IMH03_20261007T121955": ([("thin", 0.6)], "cloud over the sea: dim thick made thin"),
        "IMH03_20261007T121956": ([("thin", 0.6)], "overcast over the sea: dim thick made thin"),
        "IMH14_20260927T090347": (
            [("lift", 0.5), ("cores", 0.85)],
            "big cumulus: the shadow the draft put inside the cloud made thick; " + CORES,
        ),
        "IMH14_20260929T092121": (
            [("trim", 0.3), ("cores", 0.85)],
            "cumulus: shadows that are not dark made clear; " + CORES,
        ),
        "IMH14_20260929T092124": (
            [("trim", 0.3), ("cores", 0.85)],
            "cumulus: shadows that are not dark made clear; " + CORES,
        ),
        "IMH14_20260929T092126": (
            [("trim", 0.3), ("cores", 0.85)],
            "cumulus: shadows that are not dark made clear; " + CORES,
        ),
        # test scenes
        "IMH12_20260930T092923": (
            [("thin", 0.7), ("set", 3, 1)],
            "the veil thin, the contrail thick; the dark sea under the veil is no shadow",
        ),
        "IMH12_20260930T092931": (
            [("thin", 0.6), ("set", 2, 1, (0.0, 1.0, 0.0, 0.45))],
            "the whole veil thin, the cumulus thick",
        ),
        "IMH12_20261008T090908": ([], "the draft as it is: scattered cumulus and shadows"),
        "IMH14_20261008T090717": (
            [("thin", 0.5), ("cores", 0.85)],
            "overcast: " + THIN_EDGES + "; the saturated core at the top thick",
        ),
        # stress scenes: processing version A02, red edges 1 and 3 at 0.3 of red edge 2
        "IMH02_20260308T122749": (
            [("frame",), ("thin", 0.5)],
            "the no-data frame the draft called cloud removed; " + THIN_EDGES,
        ),
        "IMH04_20260303T123520": ([], "the draft as it is: mostly clear, one small cloud"),
        "IMH08_20260301T130916": (
            [("thin", 0.6)],
            "cellular cloud field: the dimmer gaps made thin",
        ),
    },
    "iride_eaglet2": {
        "IME01_20261019T160210": (
            [("cores", 0.85), ("trim", 0.3)],
            "cumulus field: " + CORES + "; shadows that are not dark made clear",
        ),
        "IME01_20261023T095758": ([("cores", 0.85)], "scattered cumulus: " + CORES),
        "IME02_20261017T065625": ([], "the draft as it is: cloud over the sea"),
        "IME02_20261017T100237": (
            [("start", "ocm4_nirzeros"), ("set", 2, 1)],
            "a veil over the town and the harbour: thin, as the draft without blue found it",
        ),
        "IME02_20261021T101816": ([("thin", 0.5)], "large cloud mass: " + THIN_EDGES),
        "IME02_20261022T095827": (
            [("haze", 0.45, 0.8)],
            "hazy deck the draft missed: labelled by brightness",
        ),
        "IME03_20261019T160144": ([("cores", 0.85)], "cumulus field: " + CORES),
        "IME04_20261017T093952": (
            [("start", "ocm4_nirzeros")],
            "the draft without blue: it keeps the thin cloud the other one dropped",
        ),
        "IME04_20261017T093955": ([], "the draft as it is: big cloud"),
        "IME04_20261022T093916": (
            [("haze", 0.5, 0.85)],
            "very bright scene with false shadow: labelled by brightness",
        ),
        "IME04_20261022T094005": (
            [("cores", 0.85), ("trim", 0.3)],
            "cumulus field: " + CORES + "; shadows that are not dark made clear",
        ),
        "IME06_20261021T093605": (
            [("haze", 0.8, 0.96)],
            "a saturated deck, cumulus and haze the draft missed: labelled by brightness",
        ),
        "IME07_20261020T100437": ([], "the draft as it is: cloud over the mountains"),
        "IME07_20261020T100518": (
            [("haze", 0.45, 0.8)],
            "hazy deck the draft missed: labelled by brightness",
        ),
        "IME08_20261017T094335": (
            [("cores", 0.85), ("pale", 0.6, (0.0, 0.6, 0.0, 0.16))],
            "mixed cumulus: " + CORES + "; the pale left side of the big cloud thin",
        ),
        "IME08_20261022T094032": (
            [("thick", 0.7), ("cores", 0.85), ("pale", 0.55)],
            "big bright cloud: bright thin made thick, " + CORES + "; the pale haze thin",
        ),
        "IME08_20261022T094057": ([("cores", 0.85)], "scattered cumulus: " + CORES),
    },
    "phisat2": {
        "PHISAT2_20260819T104833": (
            [("dark", 0.5), ("cores", 0.85), ("skip", (0.0, 0.33, 0.08, 0.32))],
            "cloud on dark ground cleared; the grey plume, cloud or smoke, left without a label",
        ),
        "PHISAT2_20260906T051725": (
            [("dark", 0.3), ("cores", 0.6), ("grey", 0.45)],
            "saturated cloud and its stripes thick; grey shadow inside the cloud thin",
        ),
        "PHISAT2_20261003T104909": (
            [("set", [1, 2, 3], 0), ("cores", 0.85, (0.38, 0.68, 0.74, 1.0))],
            "false cloud on towns, roofs and fields cleared; cloud only in the puffs on the right",
        ),
        "PHISAT2_20261019T233624": (
            [
                ("cores", 0.85),
                ("set", [1, 2, 3], 0, (0.06, 0.40, 0.70, 0.87)),
                ("set", [1, 2, 3], 0, (0.06, 0.31, 0.87, 0.95)),
                ("set", [1, 2, 3], 0, (0.55, 0.76, 0.17, 0.36)),
            ],
            "false cloud on the town and on the reef cleared",
        ),
        "PHISAT2_20261020T023017": ([("cores", 0.85)], "cumulus over the coast: " + CORES),
        "PHISAT2_20261025T110655": (
            [("dark", 0.4), ("cores", 0.85), ("thin", 0.8)],
            "the sea cleared; the veil over the sea thin, the bright cloud thick",
        ),
    },
}
