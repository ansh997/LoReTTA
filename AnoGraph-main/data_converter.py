#!/usr/bin/env python3
# convert_to_midas_format.py
"""
Convert a sparsified edge list with anomaly labels into two files:

    /data/<DATASET>/Data.csv    # u,i,ts   (1-based node IDs, no header)
    /data/<DATASET>/Label.csv   # label    (0/1, no header)

Input file format stays the same.
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert sparsified CSV to /data/<dataset>/Data.csv + Label.csv"
    )
    p.add_argument("--data", required=True, help="Dataset name")
    p.add_argument("--ss", required=True, help="Sparsification strategy")
    p.add_argument("--upto", required=True, help="'upto' percentage string")
    return p.parse_args()


def main():
    args = parse_args()

    # -------- input path --------------------------------------------------- #
    in_dir = Path("/raid/t2/TGN_adv/poisoned_data") / args.data / args.ss
    in_file = in_dir / f"{args.data}_{args.ss}_sparsified_{args.upto}.csv"
    if not in_file.exists():
        raise FileNotFoundError(in_file)
    print(f"[+] Loading {in_file}")
    df = pd.read_csv(in_file)

    # -------- column prep -------------------------------------------------- #
    df = df.rename(columns={"src": "u", "dst": "i", "time": "ts"})
    if "adv" not in df.columns:
        raise ValueError("Column 'adv' (anomaly label) missing.")

    df = df[["u", "i", "ts", "adv"]].astype({"ts": "int64"})
    df = df.rename(columns={"adv": "label"})

    # -------- 1-based re-indexing ----------------------------------------- #
    unique_nodes = pd.concat([df["u"], df["i"]]).unique()
    node_map = {old: new for new, old in enumerate(sorted(unique_nodes), start=1)}
    df["u"] = df["u"].map(node_map)
    df["i"] = df["i"].map(node_map)

    # -------- split -------------------------------------------------------- #
    data_df  = df[["u", "i", "ts"]]
    label_df = df[["label"]]

    # -------- output paths ------------------------------------------------- #
    folder_name = f"{args.data}_{args.ss}"
    out_root = Path("./data") / folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    data_file  = out_root / "Data.csv"
    label_file = out_root / "Label.csv"

    # -------- write -------------------------------------------------------- #
    print(f"[+] Saving edges  → {data_file}")
    data_df.to_csv(data_file, header=False, index=False)

    print(f"[+] Saving labels → {label_file}")
    label_df.to_csv(label_file, header=False, index=False)

    print("[✓] Conversion finished")


if __name__ == "__main__":
    main()
