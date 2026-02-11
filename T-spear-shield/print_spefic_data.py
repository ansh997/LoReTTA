#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Read a CSV and display only the 'data', 'model', and 'ss' columns, "
                    "optionally filtering by the 'data' column."
    )
    parser.add_argument(
        "-d", "--data",
        dest="data_filter",
        help="If provided, only rows where the 'data' column equals this value are printed"
    )
    args = parser.parse_args()
    csv_file = "Our_Attacks_results_neg_samples100_Neurips.csv"
    try:
        # Read only the specified columns
        df = pd.read_csv(csv_file, usecols=["data", "model", "ss","Test_AUC"])
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except FileNotFoundError:
        sys.stderr.write(f"Error: file not found: {csv_file}\n")
        sys.exit(1)

    # Apply filter if requested
    if args.data_filter is not None:
        df = df[df["data"] == args.data_filter]
        if df.empty:
            sys.stderr.write(f"No rows found with data = '{args.data_filter}'\n")
            sys.exit(0)

    # Print to terminal without the index column
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
