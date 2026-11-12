#!/usr/bin/env python3
"""Scan lists for the region-occlusion analysis (local; they carry labels).

  results/interp/pilot_scans.csv  all test scans of the five pilot syndromes
  results/interp/ext_scans.csv    all test scans of nine further syndromes: the
                                  remaining best-recognised syndromes and those
                                  involved in the confusion pairs

Rows are in manifest order; test_idx is the row's position in the test split of
the 49-class task (= the row of the cached test features).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from facebench.paths import MANIFESTS, RESULTS  # noqa: E402

PILOT = ["Achondroplasia", "Down Syndrome", "Mucopolysaccharidosis",
         "Treacher Collins Syndrome", "Williams-Beuren Syndrome"]
EXT = ["22q 11.2 Deletion Syndrome", "Cardiofaciocutaneous Syndrome", "Cockayne Syndrome",
       "Ehlers Danlos Syndrome", "Loeys-Dietz Syndrome", "Marfan Syndrome",
       "Pallister-Killian Syndrome", "Phelan McDermid Syndrome",
       "Rhizomelic Chondrodysplasia Punctata"]


def main():
    m = pd.read_csv(MANIFESTS / "syndrome_k20_manifest.csv")
    test = m[m.fixed_split == "test"].reset_index(drop=True)
    test["test_idx"] = test.index
    out = RESULTS / "interp"
    out.mkdir(parents=True, exist_ok=True)
    for name, syns in [("pilot_scans.csv", PILOT), ("ext_scans.csv", EXT)]:
        d = test[test.syndrome_name.isin(syns)]
        assert d.syndrome_name.nunique() == len(syns)
        d.to_csv(out / name, index=False)
        print(f"{name}: {len(d)} scans, {d.fbid.nunique()} subjects")


if __name__ == "__main__":
    main()
