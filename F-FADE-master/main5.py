#!/usr/bin/env python3
# main.py — F-FADE baseline + CLT-bootstrap 3-σ threshold
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
# Helper: CLT bootstrap threshold
# ------------------------------------------------------------------
def clt_threshold(scores, n=1_000, m=5_000, K=3.0, seed=42):
    """
    Draw m bootstrap samples of size n, compute their means, then return
    threshold = grand_mean + K * std(sample_means).
    """
    rng = np.random.default_rng(seed)
    sample_means = np.empty(m, dtype=float)
    for i in range(m):
        sample = rng.choice(scores, size=n, replace=True)
        sample_means[i] = sample.mean()
    k = sample_means.mean()
    p = sample_means.std(ddof=1)
    thr = k + K * p
    return thr, sample_means

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

    # -------------------- CLT-bootstrap threshold -----------------
    thr, _ = clt_threshold(scores,
                           n   = 1_000,   # sample size
                           m   = 5_000,   # #bootstrap replicates
                           K   = 3.0)     # three-sigma

    print(f"CLT threshold     : {thr:.4f}")

    pred = (scores >= thr).astype(int)

    # -------------------- metrics ---------------------------------
    total_anom    = labels.sum()
    detected_anom = ((labels == 1) & (pred == 1)).sum()

    fpr, tpr, _   = roc_curve(labels, scores, pos_label=1)
    AUC           = auc(fpr, tpr)

    print("\n=== Evaluation (CLT-bootstrap) ============================")
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
    # 
    
    # -------------------- save AUC --------------------------------
    with open(Path(args.model_dir) / "AUC.txt", "w") as fh:
        fh.write(json.dumps({"AUC": AUC}))
