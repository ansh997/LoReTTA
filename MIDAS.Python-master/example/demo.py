#!/usr/bin/env python3
"""
Run MIDAS on any dataset that follows

    data/<DATASET>/<DATASET>_processed.csv      # src,dst,ts
    data/<DATASET>/<DATASET>_ground_truth.csv   # label

and report AUC, confusion matrix, precision, recall, accuracy, F1,
using Youden's J statistic to pick the threshold (instead of CLT 3‑σ).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from tqdm import tqdm, trange
import csv

# --------------------------------------------------------------------------- #
# Make repo‑local `src/` discoverable                                         #
# --------------------------------------------------------------------------- #
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src"))

from MIDAS import FilteringCore, NormalCore, RelationalCore  # noqa: E402

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIDAS demo with full metrics.")
    p.add_argument("--data", default="DARPA",
                   help="Dataset name (sub‑dir of data/, default: DARPA)")
    return p.parse_args()

# --------------------------------------------------------------------------- #
# Helper: Youden‑J threshold                                                  #
# --------------------------------------------------------------------------- #

def youden_threshold(labels: np.ndarray, scores: np.ndarray):
    """Return threshold that maximises Youden's J = TPR − FPR.

    Returns (thr*, fpr*, tpr*). We use `sklearn.metrics.roc_curve` to obtain
    every operating point and pick the first threshold that attains the
    maximal J value.
    """
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    J = tpr - fpr
    idx = np.argmax(J)
    return thr[idx], fpr[idx], tpr[idx]

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()

    # ------------------------- paths ------------------------------------ #
    data_dir   = repo_root / "data" / args.data
    path_data  = data_dir / f"{args.data}_processed.csv"
    path_label = data_dir / f"{args.data}_ground_truth.csv"

    out_dir    = repo_root / "out"
    out_dir.mkdir(exist_ok=True)
    path_score = out_dir / f"{args.data}_Score.txt"
    path_auc   = out_dir / f"{args.data}_AUC.txt"

    # ------------------------- load dataset ----------------------------- #
    print(f"Loading edge stream  : {path_data}")
    print(f"Loading labels       : {path_label}")

    data = [
        [int(x) for x in line.split(b",")]
        for line in tqdm(path_data.read_bytes().splitlines(),
                         "Load Dataset", unit_scale=True)
    ]
    labels = np.fromiter(map(int, path_label.read_bytes().splitlines()),
                         dtype=int)

    # ------------------------- choose MIDAS core ------------------------ #
    midas = NormalCore(2, 1024)
    # midas = RelationalCore(2, 1024)
    # midas = FilteringCore(2, 1024, 1e3)

    # ------------------------- scoring ---------------------------------- #
    scores_raw = np.empty_like(labels, dtype=float)
    for i in trange(len(labels), desc=midas.nameAlg, unit_scale=True):
        scores_raw[i] = midas.Call(*data[i])

    # Save raw scores (optional)
    np.savetxt(path_score, scores_raw.reshape(-1, 1))
    print(f"# Raw scores saved to {path_score}")

    # ------------------------- evaluation ------------------------------- #
    # 1. AUC (uses raw scores)
    auc_val = roc_auc_score(labels, scores_raw)
    print(f"\nROC‑AUC            : {auc_val:.4f}")

    # 2. Convert to binary via Youden‑J threshold
    scores = np.log1p(np.nan_to_num(scores_raw, nan=0.0, posinf=1e10))
    thr, fpr_y, tpr_y = youden_threshold(labels, scores)
    preds = (scores >= thr).astype(int)
    print(f"Youden‑J threshold  : {thr:.4f}  (TPR={tpr_y:.3f}, FPR={fpr_y:.3f})")

    # 3. Confusion matrix & derived metrics
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n=== Confusion matrix (row = actual, col = pred) ===")
    print(f"               Pred-0   Pred-1")
    print(f"Actual-0 (norm) {tn:7d} {fp:7d}")
    print(f"Actual-1 (anom) {fn:7d} {tp:7d}")

    # ---------- precision / recall per class --------------------------- #
    report = classification_report(
        labels, preds, target_names=["normal", "anomaly"], output_dict=True
    )
    print("\n" + classification_report(
        labels, preds, target_names=["normal", "anomaly"]))

    prec_norm   = report["normal"]["precision"]
    rec_norm    = report["normal"]["recall"]
    prec_anom   = report["anomaly"]["precision"]
    rec_anom    = report["anomaly"]["recall"]

    # ---------- append to CSV ------------------------------------------ #
    pr_file     = out_dir / "precision_recall_scores.csv"
    columns     = [
        "dataset",
        "precision_normal", "recall_normal",
        "precision_anomaly", "recall_anomaly",
    ]
    write_header = not pr_file.exists()

    with pr_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "dataset": args.data,
            "precision_normal":  f"{prec_norm:.6f}",
            "recall_normal":    f"{rec_norm:.6f}",
            "precision_anomaly":f"{prec_anom:.6f}",
            "recall_anomaly":   f"{rec_anom:.6f}",
        })
    print(f"Precision/recall appended to {pr_file}")

    # 4. Save AUC
    with open(path_auc, "w") as fh:
        fh.write(f"{auc_val:.6f}\n")
    print(f"AUC saved to        {path_auc}")
