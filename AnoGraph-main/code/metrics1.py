#!/usr/bin/env python3
"""
Run AnoEdge/AnoGraph evaluation, print detailed metrics,
and append the key ones to results.csv.
Also appends AUPRC (AP) to results_new.csv with dataset and algorithm.
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn import metrics

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="DARPA")
parser.add_argument("--time_window", type=int, default=30)
parser.add_argument("--edge_threshold", type=int, default=50)
args = parser.parse_args()

# --------------------------------------------------------------------------- #
# CLT-bootstrap 3 σ threshold                                                 #
# --------------------------------------------------------------------------- #
def clt_threshold(scores, n=1_000, m=5_000, K=3.0, seed=42) -> float:
    rng   = np.random.default_rng(seed)
    n     = min(n, len(scores))
    means = rng.choice(scores, size=(m, n), replace=True).mean(axis=1)
    return means.mean() + K * means.std(ddof=1)

# --------------------------------------------------------------------------- #
# Confusion-matrix-derived stats                                              #
# --------------------------------------------------------------------------- #
def _confusion_stats(y_true, y_pred) -> tuple[int, int, int, int, float, float, float, float]:
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tn, fp, fn, tp, acc, prec, rec, f1

# --------------------------------------------------------------------------- #
# Helpers to append results                                                   #
# --------------------------------------------------------------------------- #
def _append_to_csv(
    dataset: str,
    algorithm: str,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    csv_path: str = "results.csv",
) -> None:
    """Log class-wise precision/recall for this run (Youden threshold numbers)."""

    recall_normal      = tn / (tn + fp) if (tn + fp) else 0.0          # specificity
    precision_normal   = tn / (tn + fn) if (tn + fn) else 0.0          # NPV

    recall_anomaly     = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity
    precision_anomaly  = tp / (tp + fp) if (tp + fp) else 0.0          # PPV

    row = pd.DataFrame(
        [[dataset, algorithm,
          precision_normal, recall_normal,
          precision_anomaly, recall_anomaly]],
        columns=[
            "dataset",
            "algorithm",
            "precision_normal",
            "recall_normal",
            "precision_anomaly",
            "recall_anomaly",
        ],
    )

    row.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),  # write header exactly once
        index=False,
        float_format="%.6f",
    )

def _append_auprc_to_new_csv(
    dataset: str,
    algorithm: str,
    auprc_ap: float,
    csv_path: str = "results_new.csv",
) -> None:
    """Append dataset, algorithm, and AUPRC (AP) to results_new.csv."""
    row = pd.DataFrame([[dataset, algorithm, auprc_ap]],
                       columns=["dataset", "algorithm", "auprc_ap"])
    row.to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
        float_format="%.6f",
    )

# --------------------------------------------------------------------------- #
# Main evaluation for AnoEdge                                                 #
# --------------------------------------------------------------------------- #
def print_anoedge_auc_time(base_path: str, dataset_name: str, algorithm: str) -> None:
    """Evaluate AnoEdge scores with CLT-bootstrap and Youden thresholds."""

    # ---------- load files -------------------------------------------------- #
    df_scores = pd.read_csv(
        f"{base_path}{algorithm}_{dataset_name}_score.csv",
        names=["score", "label"],
        sep=" ",
    )
    df_time = pd.read_csv(
        f"{base_path}{algorithm}_{dataset_name}_time.csv",
        names=["avg", "total"],
        sep=" ",
    )

    y_true     = df_scores["label"].astype(int).values
    raw_scores = df_scores["score"].astype(float).values

    # ---------- AUC (on raw scores, unchanged) ----------------------------- #
    auc_val = metrics.roc_auc_score(y_true, raw_scores)

    # ---------- stabilise scores (same as MIDAS script) --------------------- #
    scores = np.log1p(np.nan_to_num(raw_scores, nan=0.0, posinf=1e10))

    # ---------- AUPRC (AP) on stabilised scores ----------------------------- #
    if y_true.min() != y_true.max():
        auprc_ap = metrics.average_precision_score(y_true, scores)
    else:
        auprc_ap = np.nan

    print(f"{algorithm},{dataset_name}")
    print(f"AUC                           : {auc_val:.4f}")
    print(f"AUPRC (AP)                    : {auprc_ap:.4f}" if not np.isnan(auprc_ap) else "AUPRC (AP)                    : n/a")
    print()

    # ---------- CLT threshold ---------------------------------------------- #
    thr_clt      = clt_threshold(scores)
    y_pred_clt   = (scores >= thr_clt).astype(int)
    tn_c, fp_c, fn_c, tp_c, acc_c, prec_c, rec_c, f1_c = _confusion_stats(
        y_true, y_pred_clt
    )

    # ---------- Youden threshold ------------------------------------------- #
    fpr, tpr, thr_list          = metrics.roc_curve(y_true, scores)
    youden_idx                  = np.argmax(tpr - fpr)  # J = TPR − FPR
    thr_youden                  = thr_list[youden_idx]
    y_pred_youden               = (scores >= thr_youden).astype(int)
    tn_y, fp_y, fn_y, tp_y, acc_y, prec_y, rec_y, f1_y = _confusion_stats(
        y_true, y_pred_youden
    )

    # ---------- pretty print ------------------------------------------------ #
    print(f"CLT-bootstrap 3σ threshold    : {thr_clt:.4f}")
    print("Confusion matrix (row=actual, col=pred)")
    print("               Pred-0  Pred-1")
    print(f"Actual-0 (norm) {tn_c:7d} {fp_c:7d}")
    print(f"Actual-1 (anom) {fn_c:7d} {tp_c:7d}")
    print(
        f"Accuracy  {acc_c:.4f}  Precision {prec_c:.4f}  Recall {rec_c:.4f}  F1 {f1_c:.4f}\n"
    )

    print(f"Youden’s J threshold          : {thr_youden:.4f}")
    print("Confusion matrix (row=actual, col=pred)")
    print("               Pred-0  Pred-1")
    print(f"Actual-0 (norm) {tn_y:7d} {fp_y:7d}")
    print(f"Actual-1 (anom) {fn_y:7d} {tp_y:7d}")
    print(
        f"Accuracy  {acc_y:.4f}  Precision {prec_y:.4f}  Recall {rec_y:.4f}  F1 {f1_y:.4f}"
    )

    print(f"\nTime (total)                 : {df_time['total'].iloc[1]}\n")

    # --- append class-wise results to results.csv (Youden numbers) ---------- #
    _append_to_csv(dataset_name, algorithm, tn_y, fp_y, fn_y, tp_y)

    # --- append AUPRC(AP) to results_new.csv --------------------------------#
    _append_auprc_to_new_csv(
        dataset_name,
        algorithm,
        float(auprc_ap) if not np.isnan(auprc_ap) else np.nan
    )

# --------------------------------------------------------------------------- #
# Simple AnoGraph evaluation (unchanged + AUPRC)                              #
# --------------------------------------------------------------------------- #
def print_anograph_auc_time(
    base_path: str, dataset_name: str, time_window: int, edge_threshold: int, algorithm: str
) -> None:
    data = pd.read_csv(
        f"{base_path}{algorithm}_{dataset_name}_{time_window}_{edge_threshold}_score.csv",
        names=["score", "label"],
        sep=" ",
    )
    time_values = pd.read_csv(
        f"{base_path}{algorithm}_{dataset_name}_{time_window}_{edge_threshold}_time.csv",
        names=["avg", "total"],
        sep=" ",
    )

    # AUC on raw scores (unchanged)
    auc = metrics.roc_auc_score(data.label, data.score)

    # AUPRC (AP) on stabilised scores (consistent with AnoEdge path)
    scores_stab = np.log1p(np.nan_to_num(data.score.values, nan=0.0, posinf=1e10))
    if data.label.min() != data.label.max():
        auprc_ap = metrics.average_precision_score(data.label.values.astype(int), scores_stab)
    else:
        auprc_ap = np.nan

    print(f"{algorithm},{dataset_name}")
    print(f"AUC: {auc:.3f}")
    print(f"AUPRC (AP): {auprc_ap:.4f}" if not np.isnan(auprc_ap) else "AUPRC (AP): n/a")
    print(f"Time: {time_values['total'].iloc[1]}\n")

    # Append AUPRC(AP) to results_new.csv
    _append_auprc_to_new_csv(
        dataset_name,
        algorithm,
        float(auprc_ap) if not np.isnan(auprc_ap) else np.nan
    )

# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if args.dataset == "DARPA":
        print_anoedge_auc_time("../results/", "DARPA", "anoedge_g")
        print_anoedge_auc_time("../results/", "DARPA", "anoedge_l")

    elif args.dataset == "wikipedia":
        print_anoedge_auc_time("../results/", "wikipedia", "anoedge_g")
        print_anoedge_auc_time("../results/", "wikipedia", "anoedge_l")

    elif args.dataset == "ISCX":
        print_anograph_auc_time("../results/", "ISCX", args.time_window, args.edge_threshold, "anograph")
        print_anograph_auc_time("../results/", "ISCX", args.time_window, args.edge_threshold, "anograph_k")

        print_anoedge_auc_time("../results/", "ISCX", "anoedge_g")
        print_anoedge_auc_time("../results/", "ISCX", "anoedge_l")

    elif args.dataset == "IDS2018":
        print_anograph_auc_time("../results/", "IDS2018", args.time_window, args.edge_threshold, "anograph")
        print_anograph_auc_time("../results/", "IDS2018", args.time_window, args.edge_threshold, "anograph_k")

        print_anoedge_auc_time("../results/", "IDS2018", "anoedge_g")
        print_anoedge_auc_time("../results/", "IDS2018", "anoedge_l")

    elif args.dataset == "DDOS2019":
        print_anograph_auc_time("../results/", "DDOS2019", args.time_window, args.edge_threshold, "anograph")
        print_anograph_auc_time("../results/", "DDOS2019", args.time_window, args.edge_threshold, "anograph_k")

        print_anoedge_auc_time("../results/", "DDOS2019", "anoedge_g")
        print_anoedge_auc_time("../results/", "DDOS2019", "anoedge_l")

    else:
        print_anoedge_auc_time("../results/", args.dataset, "anoedge_g")
        print_anoedge_auc_time("../results/", args.dataset, "anoedge_l")
