#!/usr/bin/env python3
"""
M3 Training entry point — FaceBase 3D encoder experiments.
Usage: python train.py --model pointnet --exp-id A1 [options]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).parent.parent.parent
MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(MODELS_DIR))

from dataset import FaceBaseDataset
from seeding import seed_everything, loader_kwargs
from trainer import train
from evaluate import evaluate, evaluate_subject_level, metrics_summary
from pointnet import PointNet
from pointnet2 import PointNet2
from dgcnn import DGCNN
from geom_mlp import GeomMLP

# ── Default hyperparameters per model ────────────────────────────────────────
MODEL_DEFAULTS = {
    "pointnet":  {"batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 200},
    "pointnet2": {"batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 200},
    "dgcnn":     {"batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 200},
    "geommlp":   {"batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 200},
}

MODEL_CLASSES = {
    "pointnet":  PointNet,
    "pointnet2": PointNet2,
    "dgcnn":     DGCNN,
    "geommlp":   GeomMLP,
}

# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CLASSES))
    p.add_argument("--exp-id", required=True,
                   help="Experiment identifier, e.g. A1, A2, B1, B2, C")
    p.add_argument("--manifest", required=True,
                   help="Path to manifest CSV")
    p.add_argument("--exp-type", required=True,
                   choices=["ofc", "ofc_a3", "ofc_a4", "syndrome", "clinical", "combined"],
                   help="Experiment type (determines label column)")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--binary-mode", action="store_true")
    # Hyperparameter overrides
    p.add_argument("--batch-size",    type=int, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight-decay",  type=float, default=None)
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--patience",      type=int, default=20)
    p.add_argument("--seed",          type=int, default=42,
                   help="DATA SPLIT seed. Frozen at 42 for every published "
                        "result: changing it changes test-set membership, so "
                        "runs at different values are NOT comparable.")
    p.add_argument("--train-seed",    type=int, default=None,
                   help="TRAINING seed: weight init, batch order, augmentation. "
                        "The data split is unaffected, so seeds are scored on the "
                        "identical test subjects and the spread is pure retraining "
                        "variance. Omit for the legacy unseeded behaviour.")
    # Imbalance strategies
    p.add_argument("--sampler",       choices=["none", "weighted"], default="none")
    p.add_argument("--loss",          choices=["ce", "focal"], default="ce")
    p.add_argument("--focal-gamma",   type=float, default=2.0)
    p.add_argument("--aux-dim",       type=int, default=0,
                   help="Auxiliary covariate dim (2 = age+sex). 0 = disabled.")
    # Prototypical head
    p.add_argument("--head",          choices=["softmax", "proto"], default="softmax",
                   help="Classifier head: softmax (default) or proto (prototypical network)")
    p.add_argument("--proto-distance", choices=["cosine", "euclidean"], default="cosine",
                   help="Distance metric for prototypical head")
    p.add_argument("--proto-train-loss", choices=["proto", "ce"], default="proto",
                   help="Loss used during training with --head proto. "
                        "'proto': proto loss end-to-end. 'ce': CE loss during training, "
                        "proto head only at eval.")
    p.add_argument("--subject-level-eval", action="store_true",
                   help="After training, also evaluate with subject-level embedding aggregation")
    # ICP registered data
    p.add_argument("--icp", action="store_true",
                   help="Use ICP v1 registered point clouds (reads from *_icp/ directories)")
    p.add_argument("--icp-v2", action="store_true",
                   help="Use ICP v2 registered point clouds (single-scan template + point-to-plane)")
    # Cross-site (leave-one-site-out)
    p.add_argument("--holdout-site", default=None,
                   help="Leave-one-site-out: hold this site out as the TEST set "
                        "and train on all remaining sites. Requires a 'site' "
                        "column in the manifest. Omit for the normal random split.")
    return p.parse_args()


def main():
    args = parse_args()

    # Before anything constructs a tensor or touches an RNG.
    if args.train_seed is not None:
        seed_everything(args.train_seed)
        print(f"Train seed: {args.train_seed} (data split seed: {args.seed}, frozen)")
    else:
        print("Train seed: UNSEEDED (legacy path) — this run is not reproducible")

    defaults = MODEL_DEFAULTS[args.model]

    batch_size   = args.batch_size   or defaults["batch_size"]
    lr           = args.lr           or defaults["lr"]
    weight_decay = args.weight_decay or defaults["weight_decay"]
    epochs       = args.epochs       or defaults["epochs"]

    use_proto = (args.head == "proto")

    # Directories
    run_dir  = PROJECT / "experiments" / "runs" / args.exp_id / args.model
    ckpt_dir = run_dir / "checkpoints"
    log_csv  = run_dir / "train_log.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    dataset_kwargs = dict(
        manifest_csv=args.manifest,
        exp=args.exp_type,
        seed=args.seed,
        binary_mode=args.binary_mode,
        icp=args.icp,
        icp_v2=args.icp_v2,
        holdout_site=args.holdout_site,
    )
    train_ds = FaceBaseDataset(**dataset_kwargs, split="train", augment=True)
    val_ds   = FaceBaseDataset(**dataset_kwargs, split="val",   augment=False)
    test_ds  = FaceBaseDataset(**dataset_kwargs, split="test",  augment=False)

    print(f"Dataset splits — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")
    print(f"Classes ({train_ds.num_classes}): {train_ds.label_map}")
    if args.holdout_site:
        # Printed for every cross-site run, not just the smoke run: a fold that
        # looks viable in the manifest and is degenerate after preprocessing
        # attrition is the failure mode this experiment is exposed to.
        print(f"Cross-site fold — held-out TEST site: {args.holdout_site}")
        for nm, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            print(f"  {nm:5s} n={len(ds):5d}  {ds.class_counts()}")
        thin = {c: n for c, n in test_ds.class_counts().items() if n < 5}
        if thin:
            print(f"  WARNING: held-out site {args.holdout_site} has classes with "
                  f"<5 test scans: {thin} — per-class F1 is not reportable here")
    if args.icp_v2:
        print("Using ICP v2-registered point clouds (single-scan template, point-to-plane)")
    elif args.icp:
        print("Using ICP v1-registered point clouds")

    # Model
    num_classes = train_ds.num_classes
    model_kwargs = dict(num_classes=num_classes)
    if args.aux_dim > 0:
        model_kwargs["aux_dim"] = args.aux_dim
    model = MODEL_CLASSES[args.model](**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__} | params={n_params:,}")

    # Prototypical head
    proto_head = None
    if use_proto:
        from proto_head import ProtoHead
        proto_head = ProtoHead(
            embed_dim=model.embed_dim,
            n_classes=num_classes,
            distance=args.proto_distance,
        )
        print(f"ProtoHead: embed_dim={model.embed_dim} distance={args.proto_distance} "
              f"train_loss={args.proto_train_loss}")

    # Save run config
    cfg = dict(
        exp=args.exp_id, exp_type=args.exp_type, model=args.model,
        num_classes=num_classes, binary_mode=args.binary_mode,
        aux_dim=args.aux_dim,
        split_seed=args.seed, train_seed=args.train_seed,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        epochs=epochs, patience=args.patience,
        sampler=args.sampler, loss=args.loss, focal_gamma=args.focal_gamma,
        head=args.head, proto_distance=args.proto_distance,
        proto_train_loss=args.proto_train_loss if use_proto else None,
        subject_level_eval=args.subject_level_eval,
        icp=args.icp,
        icp_v2=args.icp_v2,
        holdout_site=args.holdout_site,
        checkpoint_dir=str(ckpt_dir), log_csv=str(log_csv),
        n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
        test_class_counts=test_ds.class_counts(),
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Added AFTER the json dump: torch.Generator is not serialisable, and
    # config.json already records train_seed, which is the provenance that
    # matters. Empty dict when --train-seed is omitted (legacy path).
    cfg["loader_kwargs"] = loader_kwargs(args.train_seed)

    # Train
    best_f1 = train(model, train_ds, val_ds, cfg, proto_head=proto_head)

    # ── Test evaluation ──────────────────────────────────────────────────────
    ckpt_path = ckpt_dir / f"{type(model).__name__}_{args.exp_id}_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if proto_head is not None:
        proto_head = proto_head.to(device)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=4, pin_memory=True)
    label_names = list(train_ds.label_map.keys())

    # Compute training-set prototypes for proto head evaluation
    train_protos = None
    if use_proto:
        from proto_head import compute_prototypes
        train_loader_eval = DataLoader(train_ds, batch_size=batch_size * 2,
                                       shuffle=False, num_workers=4, pin_memory=True)
        print("Computing training-set prototypes for test evaluation...")
        train_protos = compute_prototypes(
            train_loader_eval, model, device, num_classes,
            use_aux=args.aux_dim > 0
        )
        print(f"Prototypes computed: shape={train_protos.shape}")

    # Scan-level test evaluation
    test_metrics = evaluate(model, test_loader, device, label_names,
                            use_aux=args.aux_dim > 0,
                            proto_head=proto_head,
                            prototypes=train_protos)

    np.savez(
        run_dir / "predictions.npz",
        scan_ids=np.array(test_ds.df["scan_id"].tolist()),
        y_true=test_metrics["y_true"],
        y_pred=test_metrics["y_pred"],
        y_prob=test_metrics["y_prob"],
        label_names=np.array(label_names),
    )
    print(f"Predictions saved to {run_dir / 'predictions.npz'}")

    print(f"\nTest results ({args.model} / {args.exp_id}) [scan-level]:")
    print(metrics_summary(test_metrics))
    print(test_metrics["classification_report"])

    # Subject-level test evaluation (proto head only)
    subj_metrics = None
    if use_proto and args.subject_level_eval:
        try:
            subj_metrics = evaluate_subject_level(
                model, test_ds, device, label_names,
                proto_head, train_protos,
                use_aux=args.aux_dim > 0,
                batch_size=batch_size * 2,
            )
            print(f"\nTest results ({args.model} / {args.exp_id}) [subject-level]:")
            print(metrics_summary(subj_metrics))
            print(subj_metrics["classification_report"])
        except Exception as e:
            print(f"Subject-level eval skipped: {e}")

    # Save metrics to shared CSV
    metrics_csv = results_dir / f"metrics_{args.exp_id}.csv"
    write_header = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "exp_id", "head", "level", "accuracy",
                             "f1_macro", "f1_weighted", "auc_macro_ovr",
                             "n_test"] + [f"f1_{c}" for c in label_names])

        def _write_row(metrics, level):
            writer.writerow([
                args.model, args.exp_id, args.head, level,
                round(metrics["accuracy"], 4),
                round(metrics["f1_macro"], 4),
                round(metrics["f1_weighted"], 4),
                round(metrics["auc_macro_ovr"], 4),
                metrics["n_samples"],
            ] + [round(v, 4) for v in metrics["f1_per_class"]])

        _write_row(test_metrics, "scan")
        if subj_metrics is not None:
            _write_row(subj_metrics, "subject")

    print(f"Metrics appended to {metrics_csv}")

    # Save classification report
    with open(run_dir / "test_report.txt", "w") as f:
        f.write(f"Model: {args.model}\nExp: {args.exp_id}\nHead: {args.head}\n\n")
        f.write("=== Scan-level ===\n")
        f.write(metrics_summary(test_metrics) + "\n\n")
        f.write(test_metrics["classification_report"])
        if subj_metrics is not None:
            f.write("\n=== Subject-level ===\n")
            f.write(metrics_summary(subj_metrics) + "\n\n")
            f.write(subj_metrics["classification_report"])

    # Save prototypes if using proto head
    if train_protos is not None:
        proto_path = run_dir / "train_prototypes.pt"
        torch.save(train_protos, proto_path)
        print(f"Prototypes saved to {proto_path}")


if __name__ == "__main__":
    main()
