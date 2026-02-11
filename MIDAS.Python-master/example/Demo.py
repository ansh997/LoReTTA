#!/usr/bin/env python3
"""
Run MIDAS on any dataset that follows

    data/<DATASET>/<DATASET>_processed.csv      # src,dst,ts
    data/<DATASET>/<DATASET>_ground_truth.csv   # label

and report AUC, confusion matrix, precision, recall, accuracy, F1,
using the same CLT-bootstrap 3-σ thresholding you use for F-FADE.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report)
from tqdm import tqdm, trange

import sys

# make …/MIDAS.Python-master/src discoverable
sys.path.append(str(Path(__file__).resolve().parents[1] / "../src"))

from MIDAS import FilteringCore, NormalCore, RelationalCore


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIDAS demo with full metrics.")
    p.add_argument("--data", default="DARPA",
                   help="Dataset name (sub-dir of data/, default: DARPA)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helper: CLT-bootstrap threshold                                             #
# --------------------------------------------------------------------------- #
def clt_threshold(scores, n=1_000, m=5_000, K=3.0, seed=42):
    """
    Draw *m* bootstrap samples of size *n*, compute their means, then return
        threshold = grand_mean + K·std(sample_means)
    """
    rng = np.random.default_rng(seed)
    sample_means = np.empty(m, dtype=float)
    for i in range(m):
        sample_means[i] = rng.choice(scores, size=n, replace=True).mean()
    mu = sample_means.mean()
    sigma = sample_means.std(ddof=1)
    return mu + K * sigma


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()
    prefix = Path(__file__).resolve().parents[1]

    # ------------------------- paths -------------------------------------- #
    data_dir   = prefix / "data" / args.data
    path_data  = data_dir / f"{args.data}_processed.csv"
    path_label = data_dir / f"{args.data}_ground_truth.csv"

    out_dir    = prefix / "out"
    out_dir.mkdir(exist_ok=True)
    path_score = out_dir / f"{args.data}_Score.txt"
    path_auc   = out_dir / f"{args.data}_AUC.txt"

    # ------------------------- load dataset ------------------------------- #
    print(f"Loading edge stream  : {path_data}")
    print(f"Loading labels       : {path_label}")

    data = [
        [int(x) for x in line.split(b",")]
        for line in tqdm(path_data.read_bytes().splitlines(),
                         "Load Dataset", unit_scale=True)
    ]
    labels = np.fromiter(map(int, path_label.read_bytes().splitlines()),
                         dtype=int)

    # ------------------------- choose MIDAS core -------------------------- #
    midas = NormalCore(2, 1024)
    # midas = RelationalCore(2, 1024)
    # midas = FilteringCore(2, 1024, 1e3)

    # ------------------------- scoring ------------------------------------ #
    scores_raw = np.empty_like(labels, dtype=float)
    for i in trange(len(labels), desc=midas.nameAlg, unit_scale=True):
        scores_raw[i] = midas.Call(*data[i])

    # Save raw scores (optional)
    np.savetxt(path_score, scores_raw.reshape(-1, 1))
    print(f"# Raw scores saved to {path_score}")

    # ------------------------- evaluation --------------------------------- #
    # 1. AUC (uses raw scores)
    auc_val = roc_auc_score(labels, scores_raw)
    print(f"\nROC-AUC            : {auc_val:.4f}")

    # 2. Convert to binary via CLT-bootstrap threshold
    scores = np.log1p(np.nan_to_num(scores_raw, nan=0.0, posinf=1e10))
    thr    = clt_threshold(scores)
    preds  = (scores >= thr).astype(int)
    print(f"CLT 3-σ threshold  : {thr:.4f}")

    # 3. Confusion matrix & derived metrics
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\n=== Confusion matrix (row = actual, col = pred) ===")
    print(f"               Pred-0   Pred-1")
    print(f"Actual-0 (norm) {tn:7d} {fp:7d}")
    print(f"Actual-1 (anom) {fn:7d} {tp:7d}")

    print("\n" + classification_report(labels, preds,
                                     target_names=["normal", "anomaly"]))

    # 4. Save AUC
    with open(path_auc, "w") as fh:
        fh.write(f"{auc_val:.6f}\n")
    print(f"AUC saved to        {path_auc}")
