from utils import *
from model import *
from sklearn import metrics
from arguments import parse_arguments

import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Normalize model_dir to have a trailing separator
    model_dir = args.model_dir
    if not model_dir.endswith(os.sep):
        model_dir += os.sep

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    with open(os.path.join(model_dir, 'arg_list.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    dataset = Dataset(args.dataset)
    print(dataset.num_nodes)
    print(dataset.num_edges)

    model = Model(num_nodes=dataset.num_nodes, embedding_size=args.embedding_size).to(device=device)
    print(model.h_static)

    # ------------------------------------------------------------------
    # Run F-FADE detector to get per-edge anomaly scores
    # ------------------------------------------------------------------
    F_FADE = F_FADE(model, dataset, args.t_setup, args.W_upd, args.alpha,
                    args.M, args.T_th, args.epochs, args.online_train_steps,
                    args.batch_size, device)

    np.savetxt(os.path.join(model_dir, 'score.txt'), np.array(F_FADE).reshape((-1, 1)))
    F_FADE = np.array(F_FADE)
    print(F_FADE)

    # Replace NaNs with 0 to be safe
    for i in range(len(F_FADE)):
        if np.isnan(F_FADE[i]):
            F_FADE[i] = 0

    # ----- inputs -----
    labels = np.asarray(dataset.label[-len(F_FADE):])  # 1 = positive (adversarial), 0 = negative (normal)
    scores = np.asarray(F_FADE, dtype=np.float64)      # higher = “more anomalous”

    # Safe monotonic transform (optional but robust)
    scores = np.log1p(scores)
    scores = np.nan_to_num(scores, posinf=1e10)

    # ------------------------------------------------------------------
    # ROC AUC + Youden threshold (TPR - FPR)
    # ------------------------------------------------------------------
    AUC = np.nan
    best_thr = None
    if labels.min() != labels.max():
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        AUC = metrics.auc(fpr, tpr)

        J_scores = tpr - fpr
        best_idx = np.argmax(J_scores)
        best_thr = thresholds[best_idx]
    else:
        # Degenerate case: only one class present
        fpr = tpr = thresholds = np.array([])
        best_thr = 0.0

    # Turn scores into 0/1 predictions at Youden threshold
    pred = (scores >= best_thr).astype(int)

    # Confusion matrix
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\nThreshold chosen : {best_thr:.4f}")
    total_anom = int(labels.sum())
    detected_anom = int(((labels == 1) & (pred == 1)).sum())
    print(f"Total anomalies  : {total_anom}")
    print(f"Detected (TP)    : {detected_anom}")
    print(f"Recall           : {(detected_anom / total_anom):.4%}" if total_anom > 0 else "Recall           : n/a")
    print(f"AUC              : {AUC:.4f}" if not np.isnan(AUC) else "AUC              : n/a")

    print(f"\n=== Confusion matrix @ threshold {best_thr:.4f} ===")
    print(f"                Pred-0    Pred-1")
    print(f"Actual-0 (normal)  {tn:6d}    {fp:6d}")
    print(f"Actual-1 (anomaly) {fn:6d}    {tp:6d}")

    print("\n" + classification_report(labels, pred, target_names=["normal", "anomaly"], zero_division=0))

    with open(os.path.join(model_dir, 'AUC.txt'), 'w') as file:
        file.write(json.dumps(float(AUC) if not np.isnan(AUC) else None))

    # ------------------------------------------------------------------
    # AUPRC (threshold-free summary) — AP variant only
    # ------------------------------------------------------------------
    auprc_ap = np.nan
    if labels.min() != labels.max():
        # PR curve (positive class = adversarial edges)
        prec, rec, thr_pr = precision_recall_curve(labels, scores)

        # Average Precision (AP variant of AUPRC)
        auprc_ap = float(average_precision_score(labels, scores))

        # Save PR curve CSV (recall, precision, threshold)
        pr_csv = os.path.join(model_dir, "pr_curve.csv")
        pr_mat = np.c_[rec, prec, np.r_[thr_pr, np.nan]]  # pad threshold to align lengths
        np.savetxt(pr_csv, pr_mat, delimiter=",",
                   header="recall,precision,threshold", comments="")

        # Plot PR curve (step to match AP definition)
        plt.figure()
        plt.step(rec, prec, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f"PR Curve (AP={auprc_ap:.3f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "pr_curve.png"), dpi=200)
        plt.close()

    print(f"\nAUPRC (AP)       : {auprc_ap:.4f}" if not np.isnan(auprc_ap) else "\nAUPRC (AP)       : n/a")

    # ------------------------------------------------------------------
    # Append per-class precision/recall + ROC AUC + AUPRC(AP) to results.csv
    # ------------------------------------------------------------------
    report_dict = classification_report(
        labels,
        pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0,
    )

    norm_prec = report_dict["normal"]["precision"]
    norm_rec  = report_dict["normal"]["recall"]
    anom_prec = report_dict["anomaly"]["precision"]
    anom_rec  = report_dict["anomaly"]["recall"]

    # Original results file (trimmed columns to exclude removed metrics)
    csv_path = os.path.join(model_dir, "results.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["dataset",
                 "normal_precision", "normal_recall",
                 "anomaly_precision", "anomaly_recall",
                 "roc_auc", "auprc_ap"]
            )
        writer.writerow(
            [args.dataset,
             f"{norm_prec:.4f}", f"{norm_rec:.4f}",
             f"{anom_prec:.4f}", f"{anom_rec:.4f}",
             "" if np.isnan(AUC) else f"{AUC:.4f}",
             "" if np.isnan(auprc_ap) else f"{auprc_ap:.4f}"]
        )

    # New results file with just dataset and AUPRC(AP)
    csv_path_new = os.path.join(model_dir, "results_new.csv")
    write_header_new = not os.path.exists(csv_path_new)

    with open(csv_path_new, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header_new:
            writer.writerow(["dataset", "auprc_ap"])
        writer.writerow([
            args.dataset,
            "" if np.isnan(auprc_ap) else f"{auprc_ap:.4f}"
        ])

    print(f"\nSaved PR curve to {os.path.join(model_dir, 'pr_curve.csv')}")
    print(f"Saved PR plot to  {os.path.join(model_dir, 'pr_curve.png')}")
    print(f"Appended results to {csv_path}")
    print(f"Appended AUPRC(AP) to {csv_path_new}")
