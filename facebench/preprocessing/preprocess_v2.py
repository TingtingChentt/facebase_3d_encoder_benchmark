#!/usr/bin/env python3
"""
M1 Preprocessing Pipeline — FaceBase 3D Encoder
Converts OBJ/PLY meshes to normalized 4096-point numpy arrays.
Handles both Exp A (OFC: FB-5A + FB-56) and Exp B (Syndrome: FB-TJ0).
"""

import gc
import os
import re
import sys
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from plyfile import PlyData

FACEBASE_ROOT = Path(os.environ.get("FACEBASE_ROOT", "/path/to/facebase"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
AUDIT_DIR = DATA_ROOT / "audit"

N_POINTS = 4096


# ── Point sampling ─────────────────────────────────────────────────────────

def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Farthest point sampling. Returns indices of n selected points."""
    N = len(points)
    if N <= n:
        idx = np.concatenate([np.arange(N), np.random.choice(N, n - N)])
        return idx
    selected = np.zeros(n, dtype=np.int64)
    dists = np.full(N, np.inf)
    selected[0] = np.random.randint(N)
    for i in range(1, n):
        last = points[selected[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return selected


def sample_points(mesh, n: int) -> np.ndarray:
    """Sample n points from mesh surface (or vertices if PointCloud)."""
    # PointCloud: FPS directly on vertices
    if isinstance(mesh, trimesh.PointCloud) or not hasattr(mesh, "faces"):
        pts = np.array(mesh.vertices, dtype=np.float64)
        idx = farthest_point_sample(pts, n)
        return pts[idx]
    # Mesh: surface sampling
    try:
        pts, _ = trimesh.sample.sample_surface_even(mesh, n * 3)
        if len(pts) >= n:
            idx = farthest_point_sample(pts, n)
            return pts[idx]
    except Exception:
        pass
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n * 3)
        if len(pts) >= n:
            idx = farthest_point_sample(pts, n)
            return pts[idx]
    except Exception:
        pts = np.array(mesh.vertices, dtype=np.float64)
    idx = np.random.choice(len(pts), n, replace=True)
    return pts[idx]


# ── Mesh cleaning ───────────────────────────────────────────────────────────

def load_mesh(src_path: Path) -> trimesh.Trimesh:
    """Load OBJ or PLY, returning a Trimesh. Uses plyfile for binary PLY."""
    suffix = src_path.suffix.lower()
    if suffix == ".ply":
        try:
            data = PlyData.read(str(src_path))
            verts = data["vertex"]
            xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
            if "face" in data:
                # Try both field name variants (3dMD uses 'vertex_index'; others use 'vertex_indices')
                face_data = None
                for field in ("vertex_index", "vertex_indices"):
                    try:
                        face_data = data["face"][field]
                        break
                    except (ValueError, KeyError):
                        continue
                if face_data is not None:
                    faces = np.array([f.tolist() for f in face_data], dtype=np.int64)
                    # process=False: skip expensive merge_vertices/fix_winding — saves 10-50x peak memory
                    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
                else:
                    return trimesh.PointCloud(xyz)
            else:
                return trimesh.PointCloud(xyz)
        except Exception:
            # Fallback to trimesh native loader — skip_materials avoids loading .jpg textures
            return trimesh.load(str(src_path), process=False, force="mesh",
                                skip_materials=True)
    else:
        mesh = trimesh.load(str(src_path), process=False, force="mesh",
                            skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
        return mesh


def clean_mesh(mesh) -> trimesh.Trimesh:
    """Keep largest connected component. Skips split() for large meshes (memory)."""
    if isinstance(mesh, trimesh.PointCloud):
        return mesh
    if not mesh.is_empty and len(mesh.vertices) > 0:
        # mesh.split() builds a full adjacency graph — skip for large meshes (>100K verts)
        # to avoid multi-GB temporary allocations
        if len(mesh.vertices) < 100_000:
            try:
                comps = mesh.split(only_watertight=False)
                if len(comps) > 1:
                    mesh = max(comps, key=lambda m: len(m.vertices))
            except Exception:
                pass
    return mesh


def normalize(pts: np.ndarray) -> np.ndarray:
    """Center to centroid, scale to unit sphere."""
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


# ── Per-scan processing ─────────────────────────────────────────────────────

def process_scan(src_path: Path, out_path: Path) -> dict:
    """Load, clean, sample, normalize a single scan. Returns status dict."""
    t0 = time.time()
    result = {"src": str(src_path), "out": str(out_path), "status": "ok",
              "n_verts_raw": 0, "n_faces_raw": 0, "n_verts_clean": 0, "error": ""}
    try:
        mesh = load_mesh(src_path)
        if (isinstance(mesh, trimesh.PointCloud) and len(mesh.vertices) < 10) or \
           (isinstance(mesh, trimesh.Trimesh) and (mesh.is_empty or len(mesh.vertices) < 10)):
            result["status"] = "empty"
            result["error"] = "mesh empty or too small after load"
            return result

        result["n_verts_raw"] = len(mesh.vertices)
        result["n_faces_raw"] = len(getattr(mesh, "faces", np.array([])))

        mesh = clean_mesh(mesh)
        if len(mesh.vertices) < 10:
            result["status"] = "failed_clean"
            result["error"] = "mesh empty after cleaning"
            return result

        result["n_verts_clean"] = len(mesh.vertices)
        pts = sample_points(mesh, N_POINTS)
        del mesh  # trimesh holds circular refs via cache; del triggers GC eligibility
        pts = normalize(pts).astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), pts)
        del pts
        result["elapsed"] = round(time.time() - t0, 2)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


# ── Exp A: OFC (FB-5A + FB-56) ─────────────────────────────────────────────

def build_ofc_manifest() -> pd.DataFrame:
    """Join scan filenames with Cleft_Type labels for FB-5A and FB-56."""
    rows = []

    # FB-5A
    fb5a_dir = FACEBASE_ROOT / "FB-5A_files"
    allids5a = pd.read_csv(fb5a_dir / "Data/OFC1_ALLIDs.csv",
                           encoding="latin-1", low_memory=False,
                           usecols=["StudyID", "Cleft_Type"])
    allids5a = allids5a.dropna(subset=["Cleft_Type"])
    allids5a_map = dict(zip(allids5a["StudyID"], allids5a["Cleft_Type"]))

    # Pre-index all OBJ files once (avoids O(N²) rglob)
    print("  Indexing FB-5A OBJ files...")
    fb5a_index = {p.name: p for p in (fb5a_dir / "Images and Videos/3DImages").rglob("*.obj")}
    print(f"  Indexed {len(fb5a_index)} FB-5A files")

    mapping5a = pd.read_csv(
        fb5a_dir / "Images and Videos/OFC1_StudyIDs_to_Image_Files_Mappings.csv"
    )
    for _, row in mapping5a.iterrows():
        fp = str(row["File Path"])
        sid = str(row["StudyID"])
        if fp == "No Files" or not fp.endswith(".obj"):
            continue
        label = allids5a_map.get(sid)
        if label is None:
            continue
        fname = Path(fp.replace("\\", "/")).name
        src = fb5a_index.get(fname)
        if src is None:
            continue
        scan_id = f"fb5a_{sid}_{Path(fname).stem}"
        site = re.match(r"([A-Z]{2})", sid)
        site = site.group(1) if site else "UNK"
        rows.append({
            "scan_id": scan_id, "src_path": str(src), "dataset": "FB-5A",
            "study_id": sid, "cleft_type": label, "site": site,
            "out_path": str(PROCESSED_ROOT / "ofc" / f"{scan_id}.npy"),
        })

    # FB-56
    fb56_dir = FACEBASE_ROOT / "FB-56_files"
    allids56 = pd.read_csv(fb56_dir / "Data/OFC2_ALLIDs.csv",
                           encoding="latin-1", low_memory=False,
                           usecols=["StudyID", "Cleft_Type"])
    allids56 = allids56.dropna(subset=["Cleft_Type"])
    allids56_map = dict(zip(allids56["StudyID"], allids56["Cleft_Type"]))

    print("  Indexing FB-56 OBJ files...")
    fb56_index = {p.name: p for p in (fb56_dir / "Images and Videos/3DImages").rglob("*.obj")}
    print(f"  Indexed {len(fb56_index)} FB-56 files")

    mapping56 = pd.read_csv(
        fb56_dir / "Images and Videos/OFC2_StudyIDs_to_Image_Files_Mappings.csv"
    )
    for _, row in mapping56.iterrows():
        fp = str(row["File Path"])
        sid = str(row["StudyID"])
        if "3DImages" not in fp or not fp.endswith(".obj"):
            continue
        label = allids56_map.get(sid)
        if label is None:
            continue
        fname = Path(fp.replace("\\", "/")).name
        src = fb56_index.get(fname)
        if src is None:
            continue
        scan_id = f"fb56_{sid}_{Path(fname).stem}"
        site = re.match(r"([A-Z]{2})", sid)
        site = site.group(1) if site else "UNK"
        rows.append({
            "scan_id": scan_id, "src_path": str(src), "dataset": "FB-56",
            "study_id": sid, "cleft_type": label, "site": site,
            "out_path": str(PROCESSED_ROOT / "ofc" / f"{scan_id}.npy"),
        })

    return pd.DataFrame(rows)


# ── Exp B: Syndrome (FB-TJ0) ────────────────────────────────────────────────

def build_syndrome_manifest() -> pd.DataFrame:
    """Join key_file + metadata for FB-TJ0 to get scan→Syndrome Category."""
    fb_dir = FACEBASE_ROOT / "FB-TJ0_files"
    key = pd.read_csv(fb_dir / "FB00000861_key_file_2020-09-14.csv")
    meta = pd.read_csv(fb_dir / "FB00000861_metadata_2020-09-14.csv",
                       usecols=["FBID", "Syndrome Category"])
    meta = meta.dropna(subset=["Syndrome Category"])
    joined = key.merge(meta, on="FBID", how="inner")

    rows = []
    scans_dir = fb_dir / "scans"
    for _, row in joined.iterrows():
        scan_name = row["Scan_name"]
        fbid = row["FBID"]
        cat = row["Syndrome Category"]
        src = scans_dir / scan_name
        if not src.exists():
            continue
        stem = Path(scan_name).stem
        scan_id = f"fbtj0_{fbid}_{stem}"
        rows.append({
            "scan_id": scan_id, "src_path": str(src), "fbid": fbid,
            "syndrome_category": cat,
            "out_path": str(PROCESSED_ROOT / "syndrome" / f"{scan_id}.npy"),
        })

    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────────

def run_experiment(exp: str, manifest: pd.DataFrame, out_csv: Path,
                   dry_run: bool = False, max_scans: int = None,
                   worker_id: int = 0, n_workers: int = 1):
    """Process all scans in manifest. Supports simple sharding for SLURM arrays."""
    # Shard
    manifest = manifest.reset_index(drop=True)
    manifest = manifest.iloc[worker_id::n_workers]
    if max_scans:
        manifest = manifest.head(max_scans)

    print(f"[{exp}] Processing {len(manifest)} scans "
          f"(worker {worker_id}/{n_workers}, dry_run={dry_run})")

    log_rows = []
    n_ok = n_skip = n_fail = 0

    for i, row in manifest.iterrows():
        src = Path(row["src_path"])
        out = Path(row["out_path"])

        if out.exists():
            n_skip += 1
            log_rows.append({**row.to_dict(), "status": "skip", "error": ""})
            continue

        if dry_run:
            print(f"  DRY {src.name}")
            continue

        res = process_scan(src, out)
        status = res["status"]
        if status == "ok":
            n_ok += 1
        else:
            n_fail += 1
            print(f"  FAIL [{status}] {src.name}: {res['error'][:80]}")

        log_rows.append({**row.to_dict(), **res})

        if (i + 1) % 5 == 0:
            gc.collect()

        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(manifest)} done, "
                  f"ok={n_ok} skip={n_skip} fail={n_fail}")

    print(f"[{exp}] Done: ok={n_ok} skip={n_skip} fail={n_fail}")

    if log_rows and not dry_run:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(out_csv, index=False)
        print(f"  Log written to {out_csv}")

    return n_ok, n_skip, n_fail


def write_qc_report(ofc_manifest, syn_manifest, ofc_log_csv, syn_log_csv):
    """Write QC report after processing."""
    lines = ["# M1 Preprocessing QC Report", ""]

    for exp, manifest, log_csv, label_col in [
        ("Exp A — OFC", ofc_manifest, ofc_log_csv, "cleft_type"),
        ("Exp B — Syndrome", syn_manifest, syn_log_csv, "syndrome_category"),
    ]:
        lines.append(f"## {exp}")
        lines.append(f"**Total in manifest:** {len(manifest)}")

        if manifest.empty:
            lines.append("_No data_")
            lines.append("")
            continue

        lines.append(f"**Class distribution (input):**")
        lines.append("| Class | Count |")
        lines.append("|-------|-------|")
        for cls, cnt in manifest[label_col].value_counts().items():
            lines.append(f"| {cls} | {cnt} |")
        lines.append("")

        if log_csv.exists():
            log = pd.read_csv(log_csv)
            ok = (log["status"] == "ok").sum()
            skip = (log["status"] == "skip").sum()
            fail = log["status"].isin(["error", "failed_clean", "empty"]).sum()
            lines.append(f"**Processing results:** ok={ok}, skip={skip}, fail={fail}")
            lines.append(f"**Success rate:** {100*(ok+skip)/max(len(log),1):.1f}%")
            if fail > 0:
                fails = log[log["status"].isin(["error","failed_clean","empty"])]
                lines.append(f"**Failed scans (sample):**")
                for _, r in fails.head(10).iterrows():
                    lines.append(f"- `{Path(r['src_path']).name}`: {r.get('error','?')[:80]}")
            lines.append("")

            if "cleft_type" in log.columns or "syndrome_category" in log.columns:
                ok_log = log[log["status"].isin(["ok","skip"])]
                col = "cleft_type" if "cleft_type" in log.columns else "syndrome_category"
                lines.append(f"**Class distribution (processed):**")
                lines.append("| Class | Count |")
                lines.append("|-------|-------|")
                for cls, cnt in ok_log[col].value_counts().items():
                    lines.append(f"| {cls} | {cnt} |")
                lines.append("")

        if label_col == "cleft_type":
            unaffected = (manifest[label_col] == "Unaffected").sum()
            affected = (manifest[label_col] != "Unaffected").sum()
            ratio = unaffected / max(affected, 1)
            lines.append(f"**Class imbalance (Unaffected:Affected):** {ratio:.1f}:1")
            lines.append("_Note: Do NOT downsample — leave for M3 weighted loss._")
            lines.append("")

        lines.append("---")
        lines.append("")

    qc_path = PROJECT_ROOT / "facebench/preprocessing/qc_report.md"
    with open(qc_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"QC report written to {qc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["ofc", "syndrome", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Limit for testing")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--qc-only", action="store_true",
                        help="Skip processing, just write QC report from logs")
    args = parser.parse_args()

    ofc_log_csv = PROCESSED_ROOT / "ofc_processing_log.csv"
    syn_log_csv = PROCESSED_ROOT / "syndrome_processing_log.csv"

    ofc_manifest = pd.DataFrame()
    syn_manifest = pd.DataFrame()

    if args.exp in ("ofc", "both"):
        print("Building OFC manifest...")
        ofc_manifest = build_ofc_manifest()
        ofc_manifest.to_csv(PROCESSED_ROOT / "ofc_manifest.csv", index=False)
        print(f"  OFC manifest: {len(ofc_manifest)} scans")
        if not args.qc_only:
            run_experiment("ofc", ofc_manifest, ofc_log_csv,
                           dry_run=args.dry_run, max_scans=args.max_scans,
                           worker_id=args.worker_id, n_workers=args.n_workers)

    if args.exp in ("syndrome", "both"):
        print("Building syndrome manifest...")
        syn_manifest = build_syndrome_manifest()
        syn_manifest.to_csv(PROCESSED_ROOT / "syndrome_manifest.csv", index=False)
        print(f"  Syndrome manifest: {len(syn_manifest)} scans")
        if not args.qc_only:
            run_experiment("syndrome", syn_manifest, syn_log_csv,
                           dry_run=args.dry_run, max_scans=args.max_scans,
                           worker_id=args.worker_id, n_workers=args.n_workers)

    write_qc_report(ofc_manifest, syn_manifest, ofc_log_csv, syn_log_csv)
