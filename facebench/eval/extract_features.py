#!/usr/bin/env python3
"""
TASK 3, stage 1 — feature extraction for one (task, encoder, inference seed).

The whole 3 x 2 decision grid (softmax / CE+prototype / cosine-kNN, at scan and
subject level) is a function of exactly ONE forward pass per scan, because all
three heads read off the same trunk:

    feat   = model.encode(x)                 # trunk output
    embed  = fc1->bn4->relu->fc2->bn5->relu  # 256-d pre-logit  (proto head)
    logits = fc3(embed)                      # softmax head
    feat   L2-normalised                     # cosine-kNN head

So we run the encoder ONCE, cache feat/embed/logits, and derive every head,
every level, and every bootstrap replicate downstream in numpy. Re-running the
point-cloud trunk per head would cost 3x for identical numbers.

Both the TRAIN split (gallery: prototypes + kNN neighbours) and the TEST split
are extracted.

INFERENCE ONLY — checkpoints are opened read-only and never written.

Usage:
  python3 extract_features.py --task B2_corrected --encoder pointnet2 --seed 0

DEVICE (added 2026-08-09, thread facebase3d-paper-2026-08-08 PART B).
--device defaults to "cpu", which is byte-for-byte the behaviour this file had
when the cached features under results/features/ were produced.
--device cuda is used ONLY by the GPU-vs-CPU sampler diagnostic and MUST be
paired with --feature-dir so the CPU cache is never overwritten: those two
feature sets have to coexist to be differenced. The FPS start index is drawn on
the generator's own (CPU) device in farthest_point_sample and only then moved,
so the seeded draw itself is device-independent by construction; anything that
differs between devices comes from the float arithmetic downstream of it.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                  # noqa: E402

sys.path.insert(0, str(C.PROJECT / "facebench/models"))
from dataset import FaceBaseDataset                       # noqa: E402
from pointnet import PointNet                             # noqa: E402
from pointnet2 import PointNet2                           # noqa: E402
from dgcnn import DGCNN                                   # noqa: E402
from geom_mlp import GeomMLP                              # noqa: E402

MODEL_CLASSES = {"geommlp": GeomMLP, "pointnet": PointNet,
                 "dgcnn": DGCNN, "pointnet2": PointNet2}


def build_model(task, encoder):
    cfg = C.TASKS[task]
    cls = MODEL_CLASSES[encoder]
    model = cls(num_classes=cfg["num_classes"], aux_dim=cfg["aux_dim"])
    ckpt = C.checkpoint_path(task, encoder)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()
    left_training = [n for n, m in model.named_modules() if m.training]
    assert not left_training, f"modules still in train mode: {left_training}"
    return model, ckpt


def final_linear(model):
    """The classifier's last Linear. The four encoders name their heads
    differently (PN++ bn4/bn5, PointNet bn6/bn7, DGCNN bn1/bn2 + leaky-ReLU,
    GeomMLP a bare Sequential), so we locate the final layer rather than
    re-implementing each head by hand."""
    if hasattr(model, "fc3"):
        return model.fc3
    return list(model.net.children())[-1]


def head_chain(model, feat, aux, use_aux):
    """Run the model's OWN classify() on a cached trunk feature, capturing the
    pre-logit embedding via a hook on the final Linear.

    Using the model's code (rather than a hand-written copy of it) is what
    keeps this correct across four encoders with four different head layouts.
    Equivalence to model()/model.embed() is asserted in verify_equivalence().
    """
    captured = {}

    def hook(_m, inp, _out):
        captured["embed"] = inp[0].detach()

    h = final_linear(model).register_forward_hook(hook)
    try:
        logits = model.classify(feat, aux if use_aux else None)
    finally:
        h.remove()
    return captured["embed"], logits


def verify_equivalence(model, pts, aux, use_aux, seed, encoder):
    """Guard against the cached-trunk shortcut silently diverging from the real
    forward path. Checks BOTH logits and the pre-logit embedding. Must be exact
    — a nonzero delta here means every downstream head is scoring the wrong
    tensor, so we fail loudly rather than warn."""
    def kw():
        return {"generator": make_gen(seed)} if encoder == "pointnet2" else {}

    with torch.no_grad():
        feat = model.encode(pts, **kw())
        emb_cached, logits_cached = head_chain(model, feat, aux, use_aux)
        logits_live = model(pts, aux if use_aux else None, **kw())
        emb_live = model.embed(pts, aux if use_aux else None, **kw()) \
            if encoder == "pointnet2" else model.embed(pts, aux if use_aux else None)

    d_log = (logits_cached - logits_live).abs().max().item()
    d_emb = (emb_cached - emb_live).abs().max().item()
    assert d_log == 0.0, f"cached logits diverge from forward(): {d_log:.3e}"
    assert d_emb == 0.0, f"cached embed diverges from embed(): {d_emb:.3e}"
    return d_log, d_emb


def make_gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def extract(task, encoder, seed, split, model, device, feature_dir=None,
            batch_size=None):
    cfg = C.TASKS[task]
    use_aux = cfg["aux_dim"] > 0
    ds = FaceBaseDataset(
        str(cfg["manifest"]), exp=cfg["exp_type"], split=split,
        val_frac=0.15, test_frac=0.15, seed=C.DATA_SPLIT_SEED,
        augment=False, binary_mode=cfg["binary_mode"],
        # None for every pre-existing task, so their splits are untouched.
        holdout_site=cfg.get("holdout_site"),
    )
    # The FPS start index is drawn once per SAMPLE per batch, from a generator
    # advanced batch to batch, so the batch size is part of the estimator for
    # PointNet++: rebatching the same scans changes which points get sampled.
    bs = batch_size or C.BATCH_SIZE[encoder]
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    id_col = cfg["id_col"]
    if id_col and id_col in ds.df.columns:
        ids = ds.df[id_col].astype(str).tolist()
    else:
        # No subject grouping for this task: each scan is its own unit.
        ids = [f"scan{i}" for i in range(len(ds))]

    feats, embeds, logits_l, labels_l, auxes = [], [], [], [], []
    t0 = time.time()
    # ONE generator for the whole split, advanced batch to batch. Re-seeding
    # per batch would make every batch draw the same start index, which is a
    # different (and worse) estimator than the original.
    g = make_gen(seed) if encoder == "pointnet2" else None
    checked = False
    with torch.no_grad():
        for bi, (pts, aux, lbl) in enumerate(loader):
            pts, aux = pts.to(device), aux.to(device)
            if not checked:
                verify_equivalence(model, pts, aux, use_aux, seed, encoder)
                checked = True
            kw = {"generator": g} if encoder == "pointnet2" else {}
            feat = model.encode(pts, **kw)
            emb, log = head_chain(model, feat, aux, use_aux)
            feats.append(feat.cpu().numpy().astype(np.float32))
            embeds.append(emb.cpu().numpy().astype(np.float32))
            logits_l.append(log.cpu().numpy().astype(np.float32))
            labels_l.append(lbl.numpy())
            auxes.append(aux.cpu().numpy().astype(np.float32))
            if bi % 25 == 0:
                done = (bi + 1) * bs
                el = time.time() - t0
                print(f"    batch {bi+1}/{len(loader)}  ~{done}/{len(ds)}  "
                      f"{el:.0f}s  ({el/max(done,1):.3f}s/scan)", flush=True)

    out = C.feature_path(task, encoder, seed, split)
    if feature_dir is not None:
        out = Path(feature_dir) / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feat=np.concatenate(feats), embed=np.concatenate(embeds),
        logits=np.concatenate(logits_l), labels=np.concatenate(labels_l),
        aux=np.concatenate(auxes), ids=np.array(ids, dtype=object),
        label_names=np.array(
            [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])],
            dtype=object),
    )
    dt = time.time() - t0
    print(f"  [{split}] {len(ds)} scans in {dt:.0f}s "
          f"({dt/max(len(ds),1):.3f}s/scan) -> {out.name}", flush=True)
    return len(ds), dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(C.TASKS))
    p.add_argument("--encoder", required=True, choices=C.ENCODERS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--splits", default="train,test")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Default cpu — reproduces the cached feature set exactly.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override the per-encoder eval batch size. Changes the "
                        "PointNet++ FPS draw pattern, so it is part of the "
                        "estimator, not just a throughput knob. Use only with "
                        "--feature-dir.")
    p.add_argument("--feature-dir", default=None,
                   help="Override the .npz output directory. REQUIRED with "
                        "--device cuda so the CPU cache is not overwritten.")
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device visible")
        if a.feature_dir is None:
            raise SystemExit(
                "--device cuda requires --feature-dir: writing GPU features over "
                "the CPU cache would destroy the very comparison being made")
    device = torch.device(a.device)

    print(f"[extract] task={a.task} encoder={a.encoder} inference_seed={a.seed} "
          f"device={a.device} threads={torch.get_num_threads()}")
    if a.device == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}  torch={torch.__version__} "
              f"cuda={torch.version.cuda}")
    model, ckpt = build_model(a.task, a.encoder)
    model.to(device)
    print(f"  checkpoint: {ckpt.relative_to(C.PROJECT)}")

    for split in a.splits.split(","):
        extract(a.task, a.encoder, a.seed, split, model, device,
                feature_dir=a.feature_dir, batch_size=a.batch_size)
    print("[extract] done")


if __name__ == "__main__":
    main()
