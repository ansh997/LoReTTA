#!/usr/bin/env python3
"""
Convert a sparsified edge list with anomaly labels into the two-file
format expected by MIDAS:

    data/<DATASET>/<DATASET>_processed.csv        # src,dst,ts
    data/<DATASET>/<DATASET>_ground_truth.csv     # label

Changes from original script:
1. **Timestamp normalisation** – ensures the earliest timestamp is **1**
   (never 0 or negative) by shifting the whole column when necessary.
2. Minor type-safety cleanup and richer CLI help.
"""
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert sparsified CSV to MIDAS two-file format.")
    p.add_argument("--data", required=True, help="Dataset name (e.g. DARPA)")
    p.add_argument("--ss", required=True, help="Sparsification strategy tag")
    p.add_argument("--upto", required=True, help="'upto' percentage string")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1.  Input ---------------------------------------------------------
    # ------------------------------------------------------------------
    in_dir = Path("/raid/t2/TGN_adv/poisoned_data") / args.data / args.ss
    in_file = in_dir / f"{args.data}_{args.ss}_sparsified_{args.upto}.csv"
    if not in_file.exists():
        raise FileNotFoundError(in_file)
    print(f"Loading {in_file}")
    df = pd.read_csv(in_file)

    # ------------------------------------------------------------------
    # 2.  Basic sanity --------------------------------------------------
    # ------------------------------------------------------------------
    rename_map = {"src": "u", "dst": "i", "time": "ts"}
    df = df.rename(columns=rename_map)

    if "adv" not in df.columns:
        raise ValueError("Column 'adv' (anomaly label) missing.")

    # Enforce column order & dtypes early
    df = df[["u", "i", "ts", "adv"]]
    df = df.astype({"ts": "int64"})
    df = df.rename(columns={"adv": "label"})

    # ------------------------------------------------------------------
    # 3.  Timestamp normalisation --------------------------------------
    # ------------------------------------------------------------------
    min_ts = df["ts"].min()
    if min_ts < 1:
        shift = 1 - min_ts  # makes earliest timestamp exactly 1
        print(f"Shifting all timestamps by +{shift} so that min(ts) == 1")
        df["ts"] = df["ts"] + shift
    # Now min(ts) >= 1 is guaranteed

    # ------------------------------------------------------------------
    # 4.  Re-index node IDs to be 1-based -------------------------------
    # ------------------------------------------------------------------
    unique_nodes = pd.concat([df["u"], df["i"]]).unique()
    node_map = {old: new for new, old in enumerate(sorted(unique_nodes), start=1)}
    df["u"] = df["u"].map(node_map)
    df["i"] = df["i"].map(node_map)

    # ------------------------------------------------------------------
    # 5.  Split into edge-stream and label files ------------------------
    # ------------------------------------------------------------------
    data_df = df[["u", "i", "ts"]]
    label_df = df[["label"]]

    # ------------------------------------------------------------------
    # 6.  Output --------------------------------------------------------
    # ------------------------------------------------------------------
    out_name = f"{args.data}_{args.ss}"
    out_root = Path("data") / out_name
    out_root.mkdir(parents=True, exist_ok=True)

    data_file = out_root / f"{args.data}_{args.ss}_processed.csv"
    label_file = out_root / f"{args.data}_{args.ss}_ground_truth.csv"

    print(f"Saving edge stream to  {data_file}")
    data_df.to_csv(data_file, header=False, index=False)

    print(f"Saving labels to       {label_file}")
    label_df.to_csv(label_file, header=False, index=False)

    print("DONE ✓")


if __name__ == "__main__":
    main()
