#!/usr/bin/env python3
# main.py – F-FADE + Rolling-Z (3σ) threshold evaluation
# ------------------------------------------------------

from utils import Dataset
from model import Model, F_FADE
from arguments import parse_arguments

import os, json, torch, numpy as np
from pathlib import Path
from sklearn.metrics import (roc_curve, auc,
                             confusion_matrix, classification_report)
from tqdm import tqdm

# ----------------------------------------------------------------------
# Rolling Z-score (3 σ) helper
# ----------------------------------------------------------------------
def rolling_z_predict(scores, window=50_000, K=3.0):
    """
    Parameters
    ----------
    scores : 1-D numpy array (already log-compressed & finite)
    window : length of sliding window used to estimate mean & std
    K      : #std devs above the mean to trigger an anomaly

    Returns
    -------
    pred : np.ndarray of 0/1 flags (1 = anomaly)
    """
    scores = np.asarray(scores, dtype=float)
    n      = len(scores)
    pred   = np.zeros(n, dtype=int)

    # initialise with first 'window' points
    mu  = scores[:window].mean()
    var = scores[:window].var()

    for i in tqdm(range(window, n),"Finding threshold"):
        sigma = np.sqrt(var)
        if sigma > 0 and scores[i] >= mu + K * sigma:
            pred[i] = 1

        # ---- O(1) update of mean & var (Welford sliding) ------------
        x_out = scores[i - window]
        x_in  = scores[i]
        mu += (x_in - x_out) / window
        var += (x_in - x_out) * (x_in - mu + x_out - mu) / window
        # var can go negative due to FP error → clamp
        if var < 0:
            var = 0.0

    return pred


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":

    args   = parse_arguments()
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.model_dir) / "arg_list.txt", "w") as fh:
        json.dump(args.__dict__, fh, indent=2)

    # -------------------- data & model -------------------------------
    dataset = Dataset(args.dataset)
    print("#Nodes:", dataset.num_nodes)
    print("#Edges:", dataset.num_edges)

    model = Model(num_nodes=dataset.num_nodes,
                  embedding_size=args.embedding_size).to(device)
    print("Initial static embeddings:", model.h_static.shape)

    # -------------------- run F-FADE ---------------------------------
    raw_scores = F_FADE(model, dataset,
                        args.t_setup, args.W_upd, args.alpha, args.M,
                        args.T_th, args.epochs, args.online_train_steps,
                        args.batch_size, device)

    np.savetxt(Path(args.model_dir) / "score.txt",
               np.array(raw_scores).reshape(-1, 1))

    # -------------------- preprocess scores --------------------------
    scores = np.asarray(raw_scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10)
    scores = np.log1p(scores)           # compress heavy-tail

    labels = np.asarray(dataset.label[-len(scores):])   # aligns w/ F-FADE

    # -------------------- Rolling Z-score threshold ------------------
    pred = rolling_z_predict(scores,
                             window=len(scores),   # <- adjust if needed
                             K=3.0)           # 3-σ rule

    # -------------------- metrics ------------------------------------
    total_anom    = labels.sum()
    detected_anom = ((labels == 1) & (pred == 1)).sum()

    fpr, tpr, _   = roc_curve(labels, scores, pos_label=1)
    AUC           = auc(fpr, tpr)

    print("\n=== Evaluation =============================================")
    print(f"AUC               : {AUC:.4f}")
    print(f"Total anomalies   : {total_anom}")
    print(f"Detected (TP)     : {detected_anom}")
    print(f"Recall            : {(detected_anom/total_anom):.2%}")

    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion matrix (row = actual, col = pred)")
    print(f"                Pred-0   Pred-1")
    print(f"Actual-0 (normal) {tn:7d} {fp:7d}")
    print(f"Actual-1 (anomaly){fn:7d} {tp:7d}")

    print("\n" + classification_report(labels, pred,
                                     target_names=["normal", "anomaly"]))

    # -------------------- save AUC -----------------------------------
    with open(Path(args.model_dir) / "AUC.txt", "w") as fh:
        fh.write(json.dumps({"AUC": AUC}))
