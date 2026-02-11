#!/usr/bin/env python3
# main.py — F-FADE baseline + Youden-J threshold selection
# ------------------------------------------------------------------

from utils     import Dataset
from model     import Model, F_FADE
from arguments import parse_arguments

import os, json, torch, numpy as np
from pathlib import Path
from sklearn.metrics import (roc_curve, auc,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------
# Helper: Youden-J threshold (maximises TPR − FPR)
# ------------------------------------------------------------------

def youden_threshold(labels, scores):
    """Return the threshold that maximises Youden's J statistic.

    J = TPR − FPR.  We compute it over all unique thresholds produced by
    `sklearn.metrics.roc_curve` and take the first threshold that attains
    the maximal J (i.e. the most balanced point on the ROC curve).
    Returns (thr*, fpr*, tpr*) so you can inspect the operating point.
    """
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    J = tpr - fpr
    idx = np.argmax(J)
    return thr[idx], fpr[idx], tpr[idx]

# ------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------
if __name__ == "__main__":
    args   = parse_arguments()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.model_dir) / "arg_list.txt", "w") as fh:
        json.dump(args.__dict__, fh, indent=2)

    # -------------------- load data & model ------------------------
    dataset = Dataset(args.dataset)
    print("#Nodes:", dataset.num_nodes)
    print("#Edges:", dataset.num_edges)

    model = Model(num_nodes=dataset.num_nodes,
                  embedding_size=args.embedding_size).to(device)
    print("Initial static embeddings:", model.h_static.shape)

    # -------------------- run F-FADE -------------------------------
    raw_scores = F_FADE(model, dataset,
                        args.t_setup, args.W_upd, args.alpha, args.M,
                        args.T_th, args.epochs, args.online_train_steps,
                        args.batch_size, device)

    np.savetxt(Path(args.model_dir) / "score.txt",
               np.array(raw_scores).reshape(-1, 1))

    # -------------------- prepare scores --------------------------
    scores = np.log1p(np.nan_to_num(raw_scores, nan=0.0, posinf=1e10))
    scores = scores.astype(np.float64)
    labels = np.asarray(dataset.label[-len(scores):])

    # -------------------- Youden-J threshold ----------------------
    thr, fpr_y, tpr_y = youden_threshold(labels, scores)
    print(f"Youden-J threshold : {thr:.4f}  (TPR={tpr_y:.3f}, FPR={fpr_y:.3f})")

    pred = (scores >= thr).astype(int)

    # -------------------- metrics ---------------------------------
    total_anom    = labels.sum()
    detected_anom = ((labels == 1) & (pred == 1)).sum()

    fpr, tpr, _   = roc_curve(labels, scores, pos_label=1)
    AUC           = auc(fpr, tpr)

    print("\n=== Evaluation (Youden-J) ===================================")
    print(f"AUC               : {AUC:.4f}")
    print(f"Total anomalies   : {total_anom}")
    print(f"Detected (TP)     : {detected_anom}")
    print(f"Recall            : {detected_anom / total_anom:.2%}")

    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (row = actual, col = pred)")
    print(f"                Pred-0   Pred-1")
    print(f"Actual-0 (normal) {tn:7d} {fp:7d}")
    print(f"Actual-1 (anomaly){fn:7d} {tp:7d}")

    print("\n" + classification_report(labels, pred,
                                     target_names=["normal", "anomaly"]))

    # -------------------- plot & save -----------------------------
    # (intentionally left blank; extend as needed) 
    
    # -------------------- save AUC --------------------------------
    with open(Path(args.model_dir) / "AUC.txt", "w") as fh:
        fh.write(json.dumps({"AUC": AUC}))
