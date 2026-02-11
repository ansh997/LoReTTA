#!/usr/bin/env python3
# main.py – F-FADE + fixed top-q % quantile threshold
# ---------------------------------------------------

from utils     import Dataset
from model     import Model, F_FADE
from arguments import parse_arguments

import os, json, torch, numpy as np
from pathlib  import Path
from sklearn.metrics import (roc_curve, auc,
                             confusion_matrix, classification_report)

# ----------------------------------------------------------------------
# Helper: fixed-q% quantile threshold
# ----------------------------------------------------------------------
def quantile_predict(scores, q=0.005):
    """
    Parameters
    ----------
    scores : 1-D numpy array of anomaly scores (positive, finite)
    q      : proportion of edges to flag (e.g. 0.005 = top 0.5 %)

    Returns
    -------
    pred : 0/1 numpy vector where 1 = anomaly (top-q %)
    thr  : the numeric threshold chosen
    """
    thr  = np.quantile(scores, 1 - q)
    pred = (scores >= thr).astype(int)
    return pred, thr


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
    scores = np.log1p(scores)           # tame heavy tail

    labels = np.asarray(dataset.label[-len(scores):])   # aligns w/ F-FADE

    # -------------------- fixed quantile threshold -------------------
    q_alert = 0.05                     # ← ALERT BUDGET (top 0.5 %)
    pred, thr = quantile_predict(scores, q=q_alert)

    # -------------------- metrics ------------------------------------
    total_anom    = labels.sum()
    detected_anom = ((labels == 1) & (pred == 1)).sum()

    fpr, tpr, _   = roc_curve(labels, scores, pos_label=1)
    AUC           = auc(fpr, tpr)

    print("\n=== Evaluation =============================================")
    print(f"AUC               : {AUC:.4f}")
    print(f"Quantile q        : {q_alert:.3%}")
    print(f"Threshold chosen  : {thr:.4f}")
    print(f"Total anomalies   : {total_anom}")
    print(f"Detected (TP)     : {detected_anom}")
    rec = detected_anom / total_anom if total_anom else 0.0
    print(f"Recall            : {rec:.2%}")

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
