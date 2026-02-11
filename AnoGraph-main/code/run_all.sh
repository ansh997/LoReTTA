#!/usr/bin/env bash
set -euo pipefail           # safest bash defaults

# --------------------------------------------------------------------------- #
# 1. Build everything once                                                    #
# --------------------------------------------------------------------------- #
echo "==> make clean"
make clean
echo "==> make"
make

# --------------------------------------------------------------------------- #
# 2. Datasets to evaluate                                                     #
# --------------------------------------------------------------------------- #
datasets=(
  wikipedia_ts_tpr_remove_TER
  wikipedia_ts_tpr_remove_cosine
  wikipedia_ts_tpr_remove_jaccard
  wikipedia_degree
  wikipedia_pagerank
  uci_ts_tpr_remove_TER
  uci_ts_tpr_remove_cosine
  uci_ts_tpr_remove_jaccard
  uci_degree
  uci_pagerank
)

# --------------------------------------------------------------------------- #
# 3. Main evaluation loop                                                     #
# --------------------------------------------------------------------------- #
for ds in "${datasets[@]}"; do
    echo "============================"
    echo "Dataset: $ds"
    echo "============================"

    echo "Running AnoEdge-G"
    ./main anoedge_g "$ds" 2 32 0.9

    echo "Running AnoEdge-L"
    ./main anoedge_l "$ds" 2 32 0.9

    # Append precision/recall numbers for both runs to results.csv
    python3 metrics1.py --dataset "$ds"
done

echo "All datasets processed. Results accumulated in results.csv."
