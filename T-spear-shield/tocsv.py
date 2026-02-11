#!/usr/bin/env python3
"""extract_columns.py — print a nicely formatted table instead of raw CSV"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate          # pip install tabulate

COLUMNS = ["data", "robust", "ss", "Final Avg"]

def main(path: str):
    csv_path = Path(path)
    if not csv_path.exists():
        sys.exit(f"[error] File not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=COLUMNS)

    # Print as a GitHub/Markdown table (no index column)
    df_filtered = df[df["robust"].isin(["proposed", "cosine"])]
    print(tabulate(df_filtered, headers="keys", tablefmt="github", showindex=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python extract_columns.py <file.csv>")
    main(sys.argv[1])
