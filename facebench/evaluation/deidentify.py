"""Write per-patient result tables without FaceBase IDs or diagnoses.

Some figures and macros need per-patient rows, but only as grouping keys
(how many patients, how many classes, which rows belong to one patient).
deidentify() replaces the subject ID and the class label with codes drawn from
fresh OS entropy on every call — deliberately NOT seeded, so the codes cannot
be regenerated and mapped back — and shuffles the rows. Every count, mean and
group-by on the result is identical to the original table.
"""
import secrets

import numpy as np
import pandas as pd


class Deidentifier:
    """One mapping shared by all tables written in a run, so a patient keeps the
    same code across them."""

    def __init__(self):
        self.rng = np.random.default_rng(secrets.randbits(128))
        self.maps = {}

    def _code(self, col, values, prefix):
        m = self.maps.setdefault(col, {})
        new = sorted(set(values) - set(m), key=str)
        codes = self.rng.permutation(len(m) + len(new))[len(m):] if new else []
        width = len(str(len(m) + len(new)))
        for v, c in zip(self.rng.permutation(np.array(new, dtype=object)), codes):
            m[v] = f"{prefix}{int(c):0{width}d}"
        return values.map(m)

    def __call__(self, df, subject="subject", label="label"):
        df = df.copy()
        if subject in df:
            df[subject] = self._code(subject, df[subject], "P")
        if label in df:
            df[label] = self._code(label, df[label], "C")
        return df.sample(frac=1.0, random_state=self.rng.integers(2**31)).reset_index(drop=True)
