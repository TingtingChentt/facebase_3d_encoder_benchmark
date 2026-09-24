#!/usr/bin/env python3
"""
ICP Registration — manifest gap-fill pass.

The original preprocess_icp_v2.py ran against syndrome_processing_log.csv
(3326 scans). The B1 manifest has 12990 scans; 9743 were missing ICP-registered
counterparts and were silently dropped by the dataset loader.

This script:
  1. Reads one or more manifest CSVs to find all desired out_paths.
  2. Skips scans whose icp_v2 output file already exists.
  3. Runs point-to-plane ICP (max_correspondence_distance=0.15) on the rest.
  4. Appends results to the combined icp_processing_log_v2.csv.

Usage:
  python preprocess_icp_v2_manifest.py [--n-workers 14] [--dry-run]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
TEMPLATE_PATH  = PROCESSED_ROOT / "mean_face_template_v2.npy"
ICP_OUT_DIR    = PROCESSED_ROOT / "syndrome_icp_v2"
LOG_CSV        = PROCESSED_ROOT / "icp_processing_log_syndrome_v2_manifest.csv"

ICP_MAX_DIST = 0.15
ICP_MAX_ITER = 100
NORMAL_RADIUS = 0.1
NORMAL_MAX_NN = 30

MANIFESTS = [
    PROCESSED_ROOT / "syndrome_manifest_b1.csv",
]


def build_task_list() -> list[tuple[str, str]]:
    seen = set()
    tasks = []
    for mpath in MANIFESTS:
        df = pd.read_csv(mpath)
        for _, row in df.iterrows():
            src = row["out_path"]
            fname = Path(src).name
            if fname in seen:
                continue
            seen.add(fname)
            dst = str(ICP_OUT_DIR / fname)
            if os.path.exists(dst):
                continue  # already registered
            if not os.path.exists(src):
                continue  # source missing — skip
            tasks.append((src, dst))
    return tasks


def _add_normals(pcd) -> None:
    import open3d as o3d
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )


def register_one(args_tuple):
    import open3d as o3d
    src_path, dst_path, template = args_tuple
    t0 = time.time()
    scan_id = Path(src_path).stem
    result = {
        "scan_id": scan_id,
        "out_path": src_path,
        "icp_out_path": dst_path,
        "status": "ok",
        "icp_fitness": float("nan"),
        "icp_inlier_rmse": float("nan"),
        "elapsed": float("nan"),
        "error": "",
    }
    try:
        pts = np.load(src_path).astype(np.float64)
        if pts.shape[0] != 4096:
            result["status"] = "shape_error"
            result["error"] = f"expected 4096 pts, got {pts.shape[0]}"
            return result

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pts)
        _add_normals(source)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(template.astype(np.float64))
        _add_normals(target)

        reg = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=ICP_MAX_DIST,
            init=np.eye(4),
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            ),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=ICP_MAX_ITER
            ),
        )

        result["icp_fitness"] = float(reg.fitness)
        result["icp_inlier_rmse"] = float(reg.inlier_rmse)

        T = reg.transformation
        R, t = T[:3, :3], T[:3, 3]
        pts_aligned = (R @ pts.T + t[:, None]).T.astype(np.float32)

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_path, pts_aligned)
        result["elapsed"] = round(time.time() - t0, 2)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    return result


def run_batch(tasks, template, n_workers):
    import multiprocessing as mp

    args = [(src, dst, template) for src, dst in tasks]
    results = []
    t_start = time.time()

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(register_one, args, chunksize=4)):
                results.append(r)
                if (i + 1) % 200 == 0 or (i + 1) == len(args):
                    elapsed = time.time() - t_start
                    ok_n = sum(1 for x in results if x["status"] == "ok")
                    print(f"  {i+1}/{len(args)} done | {elapsed:.0f}s | ok={ok_n}")
    else:
        for i, arg in enumerate(args):
            r = register_one(arg)
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(args)} done")

    return results


def write_log(results, log_csv):
    if not results:
        return
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[ICP] Log written: {log_csv}")


def print_summary(results):
    ok = [r for r in results if r["status"] == "ok"]
    fit = [r["icp_fitness"] for r in ok if not np.isnan(r["icp_fitness"])]
    low = [f for f in fit if f < 0.3]
    print(f"\n[Summary] total={len(results)} ok={len(ok)} "
          f"mean_fit={np.mean(fit):.4f} pct_pass={100*(1-len(low)/len(ok)):.1f}%"
          if fit else f"\n[Summary] total={len(results)} ok={len(ok)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=14)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not TEMPLATE_PATH.exists():
        print(f"ERROR: template not found at {TEMPLATE_PATH}. Run preprocess_icp_v2.py --template-only first.")
        sys.exit(1)

    template = np.load(str(TEMPLATE_PATH)).astype(np.float64)
    print(f"[Template] Loaded from {TEMPLATE_PATH}")

    tasks = build_task_list()
    print(f"[Tasks] {len(tasks)} scans need ICP registration (already-done skipped)")

    if args.dry_run:
        print("[Dry-run] Exiting without processing.")
        return

    if len(tasks) == 0:
        print("Nothing to do — all manifest scans already have ICP v2 files.")
        return

    results = run_batch(tasks, template, args.n_workers)
    print_summary(results)
    write_log(results, LOG_CSV)

    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
