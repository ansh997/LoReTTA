from utils import *
from model import *
from sklearn import metrics
from arguments import parse_arguments
import os
import json
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

if __name__ == '__main__':
  args = parse_arguments()
  device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

  if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

  with open(args.model_dir + 'arg_list.txt', 'w') as file:
    json.dump(args.__dict__, file, indent=2)

  dataset = Dataset(args.dataset)
  print(dataset.num_nodes)
  print(dataset.num_edges)

  model = Model(num_nodes = dataset.num_nodes, embedding_size = args.embedding_size).to(device = device)
  print(model.h_static)

  F_FADE = F_FADE(model, dataset, args.t_setup, args.W_upd, args.alpha, args.M, args.T_th, args.epochs, args.online_train_steps, args.batch_size, device)
  np.savetxt(args.model_dir + 'score.txt', np.array(F_FADE).reshape((-1, 1)))

  F_FADE = np.array(F_FADE)
  print(F_FADE)

  for i in range(len(F_FADE)):
    if np.isnan(F_FADE[i]):
      F_FADE[i] = 0

  # ----- inputs -----
  labels  = np.asarray(dataset.label[-len(F_FADE):])   # 1 = positive, 0 = negative
  scores  = np.asarray(F_FADE)                         # higher = “more positive”

  scores = np.asarray(F_FADE, dtype=np.float64)
  scores = np.log1p(scores)              # safe monotonic transform
  scores = np.nan_to_num(scores, posinf=1e10)

  # ----- AUC (what you already have) -----
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
  AUC = metrics.auc(fpr, tpr)

  # Threshold that maximises TPR-FPR
  J_scores      = tpr - fpr
  best_idx      = np.argmax(J_scores)
  best_thr      = thresholds[best_idx]

  pred_anom     = (scores >= best_thr).astype(int)

  total_anom    = labels.sum()
  detected_anom = ((labels == 1) & (pred_anom == 1)).sum()

  print(f"Threshold chosen : {best_thr:.4f}")
  print(f"Total anomalies  : {total_anom}")
  print(f"Detected (TP)    : {detected_anom}")
  print(f"Recall           : {detected_anom / total_anom:.4%}")


  print(f"AUC  = {AUC:.4f}")


  # ------------------------------------
  # 2.  Turn scores into 0/1 predictions
  # ------------------------------------------------------------------
  pred = (scores >= best_thr).astype(int)

  # ------------------------------------------------------------------
  # 3.  Confusion matrix
  # ------------------------------------------------------------------
  cm = confusion_matrix(labels, pred, labels=[0, 1])
  tn, fp, fn, tp = cm.ravel()

  print(f"\n=== Confusion matrix @ threshold {best_thr:.4f} ===")
  print(f"                Pred-0    Pred-1")
  print(f"Actual-0 (normal)  {tn:6d}    {fp:6d}")
  print(f"Actual-1 (anomaly) {fn:6d}    {tp:6d}")

  # optional: precision, recall, F1
  print("\n" + classification_report(labels, pred, target_names=["normal", "anomaly"]))

  with open(args.model_dir + 'AUC.txt', 'w') as file:
    file.write(json.dumps(AUC))
  
  # ------------------------------------------------------------------
  # 4.  Save per-class precision / recall to results.csv
  # ------------------------------------------------------------------
  from sklearn.metrics import classification_report
  import csv

  # Get the per-class scores as a dict
  report_dict = classification_report(
      labels,
      pred,
      target_names=["normal", "anomaly"],
      output_dict=True,
      zero_division=0,        # avoid warnings if a class is absent
  )

  norm_prec = report_dict["normal"]["precision"]
  norm_rec  = report_dict["normal"]["recall"]
  anom_prec = report_dict["anomaly"]["precision"]
  anom_rec  = report_dict["anomaly"]["recall"]

  csv_path = os.path.join(args.model_dir, "results.csv")
  write_header = not os.path.exists(csv_path)

  with open(csv_path, "a", newline="") as f:
      writer = csv.writer(f)
      # write a header only once (first time the file is created)
      if write_header:
          writer.writerow(
              ["dataset",
              "normal_precision", "normal_recall",
              "anomaly_precision", "anomaly_recall"]
          )
      writer.writerow(
          [args.dataset,
          f"{norm_prec:.4f}", f"{norm_rec:.4f}",
          f"{anom_prec:.4f}", f"{anom_rec:.4f}"]
      )

  print(f"Appended results to {csv_path}")
