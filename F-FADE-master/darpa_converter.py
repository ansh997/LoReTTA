#!/usr/bin/env python3
# convert_to_darpa.py
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Convert sparsified CSV to DARPA edge stream.")
    p.add_argument("--data", required=True, help="Dataset name")
    p.add_argument("--ss",   required=True, help="Sparsification strategy")
    p.add_argument("--upto", required=True, help="'upto' percentage string")
    return p.parse_args()


def main():
    args = parse_args()

    in_dir  = Path("/raid/t2/TGN_adv/poisoned_data") / args.data / args.ss
    in_file = in_dir / f"{args.data}_{args.ss}_sparsified_{args.upto}.csv"

    out_dir  = Path("./darpa")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.data}_{args.ss}_sparsified_{args.upto}.txt"

    if not in_file.exists():
        raise FileNotFoundError(in_file)

    print(f"Loading {in_file}")
    df = pd.read_csv(in_file)

    # --- keep only the needed columns ---------------------------------------
    rename_map = {"src": "u", "dst": "i", "time": "ts"}
    df = df.rename(columns=rename_map)

    if "adv" not in df.columns:
        raise ValueError("Column 'adv' (anomaly label) missing.")
    df = df[["ts", "u", "i", "adv"]]          # desired order
    df = df.astype({"ts": "int64"})           # timestamps must be ints
    df = df.rename(columns={"adv": "label"})  # final field name

    # --- re-index node IDs (1-indexed) --------------------------------------
    unique_nodes = pd.concat([df["u"], df["i"]]).unique()
    new_ids = {old: new for new, old in enumerate(sorted(unique_nodes), start=1)}
    df["u"] = df["u"].map(new_ids)
    df["i"] = df["i"].map(new_ids)

    # --- write out (space-separated, no header) -----------------------------
    print(f"Saving DARPA format to {out_file}")
    df.to_csv(out_file, sep=" ", header=False, index=False)
    print("DONE ✓")


if __name__ == "__main__":
    main()
