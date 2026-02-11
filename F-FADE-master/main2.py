#!/usr/bin/env python3
# main.py — F-FADE + numerically-stable, fast Peaks-Over-Threshold (POT)
# --------------------------------------------------------------------

from utils     import Dataset
from model     import Model, F_FADE
from arguments import parse_arguments

import os, json, math, torch, numpy as np
from pathlib import Path
from collections import deque
from tdigest import TDigest                    # fast quantile sketch
from sklearn.metrics import (roc_curve, auc,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------
# Helper: TDigest API wrapper (works for any package version)
# --------------------------------------------------------------------
def _tdigest_q(digest: TDigest, q: float) -> float:
    """Return the q-quantile (q in [0,1]) regardless of tdigest API."""
    if hasattr(digest, "quantile"):            # newest API
        return digest.quantile(q)
    if hasattr(digest, "percentile"):
        try:                                   # some forks expect 0-1
            return digest.percentile(q)
        except ValueError:                     # classic 0-100
            return digest.percentile(100 * q)
    raise AttributeError("TDigest missing quantile/percentile method")

# --------------------------------------------------------------------
# Log-space GP survival  (avoids overflow when ξ ≈ 0)
# --------------------------------------------------------------------
def _gp_log_survival(excess: float, xi: float, beta: float) -> float:
    """
    log S(x)  for Generalised Pareto.
    Handles xi→0 (Gumbel) with the exp-limit form.
    """
    if xi == 0.0:
        return -excess / beta
    return -(1.0 / xi) * math.log1p(xi * excess / beta)

# --------------------------------------------------------------------
# Fast, streaming POT  (DSPOT-style refit + t-digest + MoM)
# --------------------------------------------------------------------
def fast_pot(scores,
             window=30_000,        # sliding window size
             tail_perc=0.05,       # use top 5 % as tail
             alpha=0.05,           # alert if tail P ≤ 5 %
             refresh_gap=1_000,    # fit GP every 1 k edges
             min_tail=50):
    """
    Parameters
    ----------
    scores : 1-D array of log-compressed scores
    Returns
    -------
    thr_vec : per-edge threshold
    pred    : 0/1 flags, 1 = anomaly
    """
    n        = len(scores)
    thr_vec  = np.empty(n, dtype=np.float64)
    pred     = np.zeros(n, dtype=np.int8)

    buf = deque(maxlen=window)     # raw scores window
    qsk = TDigest()                # quantile sketch

    # initialise GP parameters
    xi_hat   = 0.3
    beta_hat = 1.0
    u        = 0.0
    log_alpha= math.log(alpha)

    for i, s in enumerate(tqdm(scores, desc="Fast-POT thresholding")):
        # maintain window & sketch
        if len(buf) == window:
            old = buf.popleft()
            qsk.update(-old)       # removal trick
        buf.append(s)
        qsk.update(s)

        if i < window:
            thr_vec[i] = np.nan
            continue

        # ----- refit GP every refresh_gap edges --------------------
        if (i - window) % refresh_gap == 0:
            u = _tdigest_q(qsk, 1 - tail_perc)
            tail = np.asarray([x for x in buf if x >= u]) - u
            if tail.size >= max(min_tail, 2):
                m  = tail.mean()
                v  = tail.var(ddof=1)
                xi_hat  = max(0.5 * (m * m / v - 1.0), 1e-8)  # shape κ ≥ 0
                beta_hat= m * (1.0 - xi_hat)                 # scale β

        # ----- score current edge ---------------------------------
        excess = s - u
        if excess > 0:
            log_sf = _gp_log_survival(excess, xi_hat, beta_hat)
            if log_sf <= log_alpha:
                pred[i] = 1

        # ----- write threshold safely (avoid exp overflow) ----------
        # exp_arg = -xi_hat * log_alpha  (log_alpha is negative)
        exp_arg = -xi_hat * log_alpha
        if exp_arg > 700:                     # 709 is float64 limit
            factor = math.exp(700) - 1.0      # ≈ 8.2e304
        else:
            factor = math.exp(exp_arg) - 1.0
        thr_vec[i] = u + (beta_hat / xi_hat) * factor


    # back-fill first window
    thr_vec[:window] = thr_vec[window]
    return thr_vec, pred

# --------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------
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

    # -------------------- fast POT thresholding -------------------
    thr_vec, pred = fast_pot(scores)

    # -------------------- metrics ---------------------------------
    total_anom    = labels.sum()
    detected_anom = ((labels == 1) & (pred == 1)).sum()

    fpr, tpr, _   = roc_curve(labels, scores, pos_label=1)
    AUC           = auc(fpr, tpr)

    print("\n=== Evaluation (fast POT) =================================")
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
    # idx = np.arange(len(scores))
    # plt.figure(figsize=(14, 5))
    # plt.plot(idx, scores,  lw=0.4, color="tab:blue", alpha=0.4)
    # plt.plot(idx, thr_vec, lw=1.0, color="black", label="POT threshold")
    # plt.scatter(idx[labels == 1], scores[labels == 1],
    #             s=12, color="tab:red", label="True anomaly")
    # plt.xlabel("Edge index")
    # plt.ylabel("log score")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(Path(args.model_dir) / "score_plot.png", dpi=300)
    # plt.close()

    # -------------------- save AUC --------------------------------
    with open(Path(args.model_dir) / "AUC.txt", "w") as fh:
        fh.write(json.dumps({"AUC": AUC}))
