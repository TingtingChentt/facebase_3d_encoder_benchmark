"""Class-level summaries derived from per-subject predictions, cached for release.

Per-subject prediction files carry FaceBase subject IDs together with true
labels, so they are not distributed. Several figures and macros need only
class-level summaries of them (per-class F1 and test size, a confusion matrix
averaged over models, error-concentration statistics). Those summaries are
written to results/derived/ whenever the predictions are present, and read back
from there when they are not — so the shipped repository regenerates every
figure and macro, and a full rebuild refreshes and checks the cache.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from facebench.paths import PREDICTIONS, RESULTS  # noqa: E402

DERIVED = RESULTS / "derived"


def have_predictions(pattern: str) -> bool:
    return PREDICTIONS.exists() and any(PREDICTIONS.glob(pattern))


def cached_json(name: str, pattern: str, compute):
    """compute() -> JSON-serialisable; recomputed if predictions match `pattern`."""
    f = DERIVED / f"{name}.json"
    if have_predictions(pattern):
        val = compute()
        DERIVED.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(val, indent=1))   # key order kept (Counter tie order)
        return json.loads(f.read_text())          # same types on both paths
    if not f.exists():
        raise SystemExit(f"no predictions matching {pattern} and no cache {f}")
    return json.loads(f.read_text())
