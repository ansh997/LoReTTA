#!/usr/bin/env python3
"""
Aggregate Test_AUC scores from AAAI attack CSVs into ONE big table:

    rows    = (ss_method, model)  ← MultiIndex
    columns = datasets            ← ['enron','mooc','uci','wiki']
    value   = Test_AUC

Author : <your name>
Date   : 2025-06-15
"""

import os, glob
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATASETS = ["enron", "mooc", "uci", "wiki"]
MODELS   = ["dysat", "jodie", "tgat", "tgn"]
CSV_DIR  = "."                       # <— edit me

# ------------------------------------------------------------
# 1. Collect numbers → scores[ss][model][dataset] = auc
# ------------------------------------------------------------
scores = defaultdict(lambda: defaultdict(dict))

for ds in DATASETS:
    for mdl in MODELS:
        pattern = (f"Our_Attacks_results_neg_samples100_AAAI_{ds}_sprs_allmetrics.csv"
                   if mdl == "tgn"
                   else f"Our_Attacks_results_neg_samples100_AAAI_{ds}_sprs_allmetrics_{mdl}.csv")

        paths = glob.glob(os.path.join(CSV_DIR, pattern))
        if not paths:
            print(f"[WARN] missing file for {ds=} {mdl=}")
            continue

        df = pd.read_csv(paths[0], usecols=["ss", "Test_AUC"])
        for _, r in df.iterrows():
            scores[r.ss][mdl][ds] = r.Test_AUC

# ------------------------------------------------------------
# 2. Build ONE big MultiIndex table
# ------------------------------------------------------------
all_ss   = sorted(scores.keys())
index    = pd.MultiIndex.from_product([all_ss, MODELS],
                                      names=["ss", "model"])
big_tbl  = pd.DataFrame(index=index, columns=DATASETS, dtype=float)

# fill
for s, mdl_dict in scores.items():
    for mdl, ds_dict in mdl_dict.items():
        for ds, auc in ds_dict.items():
            big_tbl.loc[(s, mdl), ds] = auc

# pretty view
print("\n=== MASTER TABLE (rows = ss×model, cols = datasets) ===")
print(big_tbl.to_markdown(floatfmt=".4f"))

# ------------------------------------------------------------
# 3. Save to disk
# ------------------------------------------------------------
out_path = os.path.join(CSV_DIR, "ALL_ss_models_datasets_sprs_AUC_table.csv")
big_tbl.to_csv(out_path)
print("\nSaved table →", out_path)
