#!/usr/bin/env python3
"""
ICP Registration Preprocessing v2 — FaceBase 3D Encoder

Fixes the failed v1 run (mean-of-unregistered-clouds template).
Uses:
  - Single representative Control scan as template (selected by PCA proximity to centroid)
  - Point-to-plane ICP (requires normal estimation)
  - max_iteration=100 (up from 50)
  - Outputs to syndrome_icp_v2/ and ofc_icp_v2/

Usage:
  python preprocess_icp_v2.py [--template-only] [--n-workers 14] [--batch syndrome|ofc|both]
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
TEMPLATE_PATH   = PROCESSED_ROOT / "mean_face_template_v2.npy"
TEMPLATE_INFO   = PROCESSED_ROOT / "icp_template_info.json"

ICP_MAX_DIST = 0.15
ICP_MAX_ITER = 100
NORMAL_RADIUS = 0.1
NORMAL_MAX_NN = 30


# ── Template selection via PCA ─────────────────────────────────────────────

def select_pca_template(syndrome_log_csv: Path) -> tuple[str, np.ndarray]:
    """
    Load all ok Control scans from FB-TJ0.
    Flatten each (4096,3) cloud → (12288,) vector.
    PCA → find scan closest to centroid in PC1/PC2 → use as template.
    Returns (scan_id, point_cloud_4096x3).
    """
    log = pd.read_csv(syndrome_log_csv)
    controls = log[
        (log["syndrome_category"] == "Control") &
        (log["status"] == "ok")
    ].reset_index(drop=True)
    print(f"[Template] Loading {len(controls)} ok Control scans for PCA...")

    rows = []
    paths = []
    for _, r in controls.iterrows():
        p = r["out_path"]
        if not os.path.exists(p):
            continue
        pts = np.load(p).astype(np.float32)
        if pts.shape != (4096, 3):
            continue
        rows.append(pts.flatten())
        paths.append((r["scan_id"], p))

    print(f"[Template] Loaded {len(rows)} scans successfully.")
    X = np.stack(rows, axis=0)  # (N, 12288)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)          # (N, 2)
    centroid = coords.mean(axis=0)         # (2,)
    dists = np.linalg.norm(coords - centroid, axis=1)
    best_idx = int(np.argmin(dists))

    scan_id, scan_path = paths[best_idx]
    template = rows[best_idx].reshape(4096, 3).astype(np.float32)

    info = {
        "scan_id": scan_id,
        "scan_path": scan_path,
        "n_candidates": len(rows),
        "pc1": float(coords[best_idx, 0]),
        "pc2": float(coords[best_idx, 1]),
        "centroid_pc1": float(centroid[0]),
        "centroid_pc2": float(centroid[1]),
        "dist_to_centroid": float(dists[best_idx]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    TEMPLATE_INFO.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATE_INFO, "w") as f:
        json.dump(info, f, indent=2)
    np.save(str(TEMPLATE_PATH), template)

    print(f"[Template] Selected: {scan_id}")
    print(f"  PC1={info['pc1']:.4f}  PC2={info['pc2']:.4f}")
    print(f"  Centroid PC1={info['centroid_pc1']:.4f}  PC2={info['centroid_pc2']:.4f}")
    print(f"  Distance to centroid: {info['dist_to_centroid']:.4f}")
    print(f"  Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Template saved to {TEMPLATE_PATH}")
    print(f"  Info saved to {TEMPLATE_INFO}")

    return scan_id, template


# ── ICP registration (point-to-plane) ────────────────────────────────────

def _add_normals(pcd: o3d.geometry.PointCloud) -> None:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )


def register_one_v2(args_tuple):
    """
    Worker function for multiprocessing (point-to-plane ICP).
    args_tuple: (out_path, icp_out_path, template_np)
    """
    out_path, icp_out_path, template = args_tuple
    t0 = time.time()
    scan_id = Path(out_path).stem
    result = {
        "scan_id": scan_id,
        "out_path": out_path,
        "icp_out_path": icp_out_path,
        "status": "ok",
        "icp_fitness": float("nan"),
        "icp_inlier_rmse": float("nan"),
        "elapsed": float("nan"),
        "error": "",
    }

    try:
        if not os.path.exists(out_path):
            result["status"] = "missing"
            result["error"] = "source .npy not found"
            return result

        pts = np.load(out_path).astype(np.float64)
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

        Path(icp_out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(icp_out_path, pts_aligned)
        result["elapsed"] = round(time.time() - t0, 2)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]

    return result


def run_icp_batch_v2(df: pd.DataFrame, icp_out_dir: Path,
                     template: np.ndarray, n_workers: int) -> list[dict]:
    valid = df[df["status"].isin(["ok", "skip"])].reset_index(drop=True)
    print(f"[ICP] Processing {len(valid)} scans with {n_workers} workers...")

    tasks = [
        (row["out_path"],
         str(icp_out_dir / Path(row["out_path"]).name),
         template)
        for _, row in valid.iterrows()
    ]

    results = []
    t_start = time.time()
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(register_one_v2, tasks, chunksize=4)):
                results.append(r)
                if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t_start
                    ok_n = sum(1 for x in results if x["status"] == "ok")
                    print(f"  {i+1}/{len(tasks)} done | {elapsed:.0f}s | ok={ok_n}")
    else:
        for i, task in enumerate(tasks):
            r = register_one_v2(task)
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} done")

    return results


def write_log(results: list[dict], log_csv: Path) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[ICP] Log written to {log_csv}")


def print_summary(results: list[dict], label: str = "") -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    fitness_vals = [r["icp_fitness"] for r in ok if not np.isnan(r["icp_fitness"])]
    low_fit = [r for r in ok if not np.isnan(r["icp_fitness"]) and r["icp_fitness"] < 0.3]
    n_dropped = len(results) - len(ok)

    mean_fit = float(np.mean(fitness_vals)) if fitness_vals else float("nan")
    pct_pass = 100.0 * (1 - len(low_fit) / len(ok)) if ok else 0.0

    tag = f" [{label}]" if label else ""
    print(f"\n[ICP Summary{tag}]")
    print(f"  Total processed: {len(results)}")
    print(f"  ok={len(ok)}  failed/missing={n_dropped}")
    if fitness_vals:
        print(f"  fitness: mean={mean_fit:.4f}  "
              f"min={min(fitness_vals):.4f}  max={max(fitness_vals):.4f}")
    print(f"  fitness >= 0.3: {len(ok)-len(low_fit)}/{len(ok)} ({pct_pass:.1f}%)")
    print(f"  fitness <  0.3: {len(low_fit)} (will be flagged in log)")

    return {"mean_fitness": mean_fit, "pct_pass": pct_pass,
            "n_ok": len(ok), "n_dropped": n_dropped,
            "n_low_fit": len(low_fit)}


# ── Sanity visualisation ───────────────────────────────────────────────────

def sanity_visualise_v2(results: list[dict], n_samples: int = 5,
                        out_dir: Path = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Sanity] matplotlib not available — skipping")
        return

    ok_results = [r for r in results if r["status"] == "ok"]
    rng = np.random.default_rng(42)
    sample = rng.choice(ok_results, size=min(n_samples, len(ok_results)), replace=False)
    out_dir = out_dir or PROCESSED_ROOT / "icp_sanity_plots_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in sample:
        sid = r["scan_id"]
        if not os.path.exists(r["out_path"]) or not os.path.exists(r["icp_out_path"]):
            continue
        orig = np.load(r["out_path"]).astype(np.float32)
        aligned = np.load(r["icp_out_path"]).astype(np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, pts, title in [
            (axes[0], orig, "Unregistered"),
            (axes[1], aligned, f"ICP v2 aligned (fit={r['icp_fitness']:.3f})"),
        ]:
            ax.scatter(pts[::8, 0], pts[::8, 1], s=1, alpha=0.5)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        fig.suptitle(sid)
        plt.tight_layout()
        plt.savefig(out_dir / f"{sid}.png", dpi=80)
        plt.close(fig)

    print(f"[Sanity] Plots saved to {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=["syndrome", "ofc", "both"], default="both")
    parser.add_argument("--template-only", action="store_true")
    parser.add_argument("--n-workers", type=int, default=14)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    syndrome_log = PROCESSED_ROOT / "syndrome_processing_log.csv"
    ofc_log      = PROCESSED_ROOT / "ofc_processing_log.csv"

    # Step 1: Template
    if TEMPLATE_PATH.exists() and TEMPLATE_INFO.exists():
        print(f"[Template] Loading existing v2 template from {TEMPLATE_PATH}")
        template = np.load(str(TEMPLATE_PATH)).astype(np.float64)
        with open(TEMPLATE_INFO) as f:
            info = json.load(f)
        print(f"  scan_id={info['scan_id']}  "
              f"PC1={info['pc1']:.4f}  PC2={info['pc2']:.4f}")
    else:
        _, template_f32 = select_pca_template(syndrome_log)
        template = template_f32.astype(np.float64)

    if args.template_only:
        print("Template built. Exiting (--template-only).")
        return

    all_results = []

    # Step 2: Register
    if args.batch in ("syndrome", "both"):
        syn_df  = pd.read_csv(syndrome_log)
        icp_dir = PROCESSED_ROOT / "syndrome_icp_v2"
        res = run_icp_batch_v2(syn_df, icp_dir, template, args.n_workers)
        all_results.extend(res)
        stats = print_summary(res, label="syndrome")
        write_log(res, PROCESSED_ROOT / "icp_processing_log_syndrome_v2.csv")
        if args.sanity:
            sanity_visualise_v2(res, out_dir=PROCESSED_ROOT / "icp_sanity_plots_v2" / "syndrome")

        # Sanity gate
        if stats["mean_fitness"] < 0.3:
            print(f"\n[GATE] FAILED: mean_fitness={stats['mean_fitness']:.4f} < 0.3. "
                  "Stopping before OFC and training. "
                  "Please review template or increase ICP distance.")
            sys.exit(1)
        if (100 - stats["pct_pass"]) > 20:
            print(f"\n[GATE] FAILED: {100-stats['pct_pass']:.1f}% scans have fitness<0.3 (>20% threshold). "
                  "Stopping before OFC and training.")
            sys.exit(1)
        print(f"\n[GATE] PASSED: mean_fitness={stats['mean_fitness']:.4f}, "
              f"{stats['pct_pass']:.1f}% fitness>=0.3")

    if args.batch in ("ofc", "both"):
        ofc_df  = pd.read_csv(ofc_log)
        icp_dir = PROCESSED_ROOT / "ofc_icp_v2"
        res = run_icp_batch_v2(ofc_df, icp_dir, template, args.n_workers)
        all_results.extend(res)
        print_summary(res, label="ofc")
        write_log(res, PROCESSED_ROOT / "icp_processing_log_ofc_v2.csv")
        if args.sanity:
            sanity_visualise_v2(res, out_dir=PROCESSED_ROOT / "icp_sanity_plots_v2" / "ofc")

    # Combined log
    if all_results:
        write_log(all_results, PROCESSED_ROOT / "icp_processing_log_v2.csv")

    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
