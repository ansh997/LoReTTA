#!/usr/bin/env python3
"""
Generate a LaTeX table from ALL_ss_models_datasets_AUC_table.csv
Author : <you>   •   Date : 2025-06-15
"""

import pandas as pd
from typing import Union
import textwrap

# ----------------------------------------------------------------------
# 1) LOAD
# ----------------------------------------------------------------------
CSV = "ALL_ss_models_datasets_sprs_AUC_table.csv"
df  = pd.read_csv(CSV)

# ----------------------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------------------
MODEL_ORDER = ["tgn", "jodie", "dysat", "tgat"]

def nice_attack(raw: str) -> str:
    return raw.replace("_", " ").title()

def nice_val(x: Union[str, float]) -> str:
    if pd.isna(x) or x == "":
        return "--"
    if isinstance(x, float):
        x = f"{x:.2f}\\%"
    x = str(x).replace("%", "\\%").replace("±", "$\\pm$")
    return x

# ----------------------------------------------------------------------
# 3) Build LaTeX
# ----------------------------------------------------------------------
lines = []
lines.append("\\begin{table}[]")
lines.append("\\begin{tabular}{rlllll}")
lines.append("\\multicolumn{1}{l}{Attack} & Model & Wikipedia & UCI & MOOC & Enron \\\\")

for attack in df["ss"].unique():
    sub = df[df["ss"] == attack].set_index("model")
    for i, mdl in enumerate(MODEL_ORDER):
        row_vals = ["--", "--", "--", "--"]
        if mdl in sub.index:
            row_vals = [
                nice_val(sub.loc[mdl, "wiki"]),
                nice_val(sub.loc[mdl, "uci"]),
                nice_val(sub.loc[mdl, "mooc"]),
                nice_val(sub.loc[mdl, "enron"]),
            ]
        attack_cell = f"\\multirow{{4}}{{*}}{{{nice_attack(attack)}}}" if i == 0 else " " * 5
        latex_row   = f"{attack_cell} & {mdl.upper():5} & {' & '.join(row_vals)} \\\\"
        lines.append(latex_row)
    lines.append("")  # blank line between attacks (optional)

lines.append("\\end{tabular}")
lines.append("\\end{table}")

print(textwrap.dedent("\n".join(lines)))
