#!/usr/bin/env python3
"""
Batch convert PLY scans to OBJ format for Exp C (combined dataset).
Supports FB-TJ0 (from syndrome_manifest.csv) and FB-TK0 (directory scan).

FB-TJ0 output: data/raw/FB-TJ0_obj/<scan_id>.obj
FB-TK0 output: data/raw/FB-TK0_obj/fbtk0_<stem>.obj
"""

import argparse
import os
import gc
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from plyfile import PlyData

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TJ0_MANIFEST  = PROJECT_ROOT / "data/processed/syndrome_manifest.csv"
TJ0_OUT_DIR   = PROJECT_ROOT / "data/raw/FB-TJ0_obj"
TK0_PLY_DIR   = Path(os.environ.get("FACEBASE_ROOT", "/path/to/facebase")) / "FB-TK0_files"
TK0_OUT_DIR   = PROJECT_ROOT / "data/raw/FB-TK0_obj"
LOG_PATH      = PROJECT_ROOT / "data/processed/ply_to_obj_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def build_task_list(dataset: str):
    """Return list of (scan_id, src_path, out_path) tuples."""
    tasks = []
    if dataset in ("tj0", "both"):
        df = pd.read_csv(TJ0_MANIFEST)
        for row in df.itertuples(index=False):
            out = str(TJ0_OUT_DIR / f"{row.scan_id}.obj")
            tasks.append((row.scan_id, row.src_path, out))

    if dataset in ("tk0", "both"):
        for ply in sorted(TK0_PLY_DIR.glob("*.ply")):
            scan_id = f"fbtk0_{ply.stem}"
            out = str(TK0_OUT_DIR / f"{scan_id}.obj")
            tasks.append((scan_id, str(ply), out))

    return tasks


def load_ply(src_path: str) -> trimesh.Trimesh:
    """Load PLY using plyfile to handle 3dMD's vertex_index (singular) format."""
    data = PlyData.read(src_path)
    verts = data["vertex"]
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    if "face" in data:
        faces_raw = data["face"]["vertex_index"]  # 3dMD uses singular; plyfile handles both
        faces = np.array([f.tolist() for f in faces_raw], dtype=np.int64)
        return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    return trimesh.PointCloud(xyz)


def convert_one(args):
    scan_id, src_path, out_path = args
    out = Path(out_path)
    if out.exists():
        return {"scan_id": scan_id, "status": "skip", "error": ""}
    try:
        t0 = time.time()
        mesh = load_ply(src_path)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            del mesh
            return {"scan_id": scan_id, "status": "fail", "error": "empty mesh after load"}
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out))
        elapsed = time.time() - t0
        del mesh
        gc.collect()
        return {"scan_id": scan_id, "status": "ok", "error": "", "elapsed": elapsed}
    except Exception as e:
        return {"scan_id": scan_id, "status": "fail", "error": str(e)[:120]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["tj0", "tk0", "both"], default="both")
    parser.add_argument("--worker-id",  type=int, default=0)
    parser.add_argument("--n-workers",  type=int, default=1)
    parser.add_argument("--n-procs",    type=int, default=8,
                        help="multiprocessing pool size per SLURM task")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    all_tasks = build_task_list(args.dataset)
    shard = all_tasks[args.worker_id::args.n_workers]
    log.info(f"Worker {args.worker_id}/{args.n_workers}: {len(shard)}/{len(all_tasks)} scans "
             f"(dataset={args.dataset})")

    if args.dry_run:
        log.info(f"Dry-run: would process {len(shard)} scans")
        return

    results = []
    if args.n_procs > 1:
        with Pool(args.n_procs) as pool:
            for i, res in enumerate(pool.imap_unordered(convert_one, shard, chunksize=4)):
                results.append(res)
                if (i + 1) % 200 == 0:
                    ok   = sum(1 for r in results if r["status"] == "ok")
                    skip = sum(1 for r in results if r["status"] == "skip")
                    fail = sum(1 for r in results if r["status"] == "fail")
                    log.info(f"  {i+1}/{len(shard)} | ok={ok} skip={skip} fail={fail}")
    else:
        for i, task in enumerate(shard):
            results.append(convert_one(task))
            if (i + 1) % 200 == 0:
                log.info(f"  {i+1}/{len(shard)}")

    ok   = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip")
    fail = sum(1 for r in results if r["status"] == "fail")
    log.info(f"Done: ok={ok} skip={skip} fail={fail}")

    # Merge into shared log
    log_df = pd.DataFrame(results)
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        log_df = pd.concat(
            [existing[~existing["scan_id"].isin(log_df["scan_id"])], log_df],
            ignore_index=True
        )
    log_df.to_csv(LOG_PATH, index=False)
    log.info(f"Log written: {LOG_PATH}")


if __name__ == "__main__":
    main()
