"""
The final Syndrome-label pool, up to (not including) the class-size filter:
build_syndrome_final_manifest.py's rules (n50 rules from build_syndrome_name_manifest.py,
then the Moebius / listed-conflict / review-conflict exclusions recorded in
data/labels/syndrome_final_excluded.csv). Used by build_class_filter_manifests.py.
"""
import importlib.util
import os
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "fin", Path(__file__).parent / "build_syndrome_final_manifest.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
N50 = F.N50


def build_pool():
    """Replicates the final builder up to the class-size filter; returns steps too."""
    steps = []
    df = pd.read_csv(N50.SRC)
    df = df[df["eligible"]].reset_index(drop=True)   # scans in the benchmark (data/splits/scans.csv)
    meta = pd.read_csv(N50.META, usecols=["FBID", "Syndrome"])
    df = df.merge(meta, left_on="fbid", right_on="FBID", how="left")
    df["Syndrome"] = df["Syndrome"].str.strip()
    key = df["Syndrome"].str.lower()
    canon = (df.assign(k=key).groupby(["k", "Syndrome"]).fbid.nunique()
               .reset_index().sort_values("fbid")
               .drop_duplicates("k", keep="last").set_index("k")["Syndrome"])
    df["syndrome_name"] = key.map(canon)
    steps.append(("All processed scans with a Syndrome value", df))
    df = df[~df["syndrome_name"].isin(N50.EXCLUDE)]
    steps.append(("Drop non-diagnostic values, Cleft Lip/Palate, Rett subtypes", df))
    excl = pd.read_csv(F.OUT_EXCL)
    for reason, txt in [("A_moebius_class", "Drop Moebius Syndrome class"),
                        ("B_listed_conflict", "Drop listed CD/Syndrome conflicts"),
                        ("C_review_conflict", "Drop conflicts from the manual review")]:
        df = df[~df.fbid.isin(set(excl.loc[excl.reason == reason, "fbid"]))]
        steps.append((txt, df))
    return df, steps
