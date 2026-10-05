#!/usr/bin/env python3
"""Merge Imed's pareto_2_full_train.csv + eval_morl_final_2.csv into ONE file
with exactly the same columns / names / order as results_RL_budget.csv.

Usage: python merge_imed.py
       python merge_imed.py -a pareto_2_full_train.csv -b eval_morl_final_2.csv \
                            -t results_RL_budget.csv -o imed_merged.csv
"""
import argparse
import pandas as pd

KEYS = ["benchmark", "w_ops", "w_keys"]
INT_COLS = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply",
            "Depth", "Multiplicative Depth", "Remaining_noise_budget"]
TIME_COLS = ["compile_time (s)", "circuit_execution_time (s)",
             "galois_keys_generation_time (s)", "total_execution_time (s)"]
DEFAULT_HEADER = ["benchmark", "w_ops", "w_keys", "noise_budget", "add", "sub",
                  "multiply_plain", "rotate_rows", "negate", "multiply", "Depth",
                  "Multiplicative Depth", "compile_time (s)",
                  "circuit_execution_time (s)", "galois_keys_generation_time (s)",
                  "total_execution_time (s)", "Remaining_noise_budget",
                  "rotation_keys_size (MB)", 
                  "final_ops_cost", "final_keys_cost"]


def load(path):
    df = pd.read_csv(path, na_values=["N/A", "NA", "", "nan"])
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Multplicative Depth": "Multiplicative Depth",
                            "rotation_key_size (MB)": "rotation_keys_size (MB)"})
    for c in ("w_ops", "w_keys"):
        df[c] = df[c].round(6)
    return df.drop_duplicates(KEYS, keep="last")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-a", "--pareto", default="pareto_2_full_train.csv")
    p.add_argument("-b", "--eval", default="eval_morl_final_2.csv")
    p.add_argument("-t", "--template", default="results_RL_budget.csv",
                   help="your CSV: its header is copied (falls back to the built-in one)")
    p.add_argument("-o", "--out", default="imed_merged.csv")
    p.add_argument("--how", default="outer", choices=["outer", "inner", "left", "right"])
    a = p.parse_args()

    try:
        header = list(pd.read_csv(a.template, nrows=0).columns)
    except FileNotFoundError:
        header = DEFAULT_HEADER

    par, ev = load(a.pareto), load(a.eval)
    m = par.merge(ev, on=KEYS, how=a.how, suffixes=("", "_ev"))

    # columns present in both files: pareto value, else eval value
    for c in ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply",
              "Depth", "Multiplicative Depth", "compile_time (s)"]:
        m[c] = m[c].combine_first(m[f"{c}_ev"])

    # eval columns -> names used in your file
    m["final_ops_cost"] = m["final_exec_cost"]          # same quantity
    # final_keys_cost already has the same name (cost-model key count)

    # columns Imed's files do not contain -> left empty (N/A)
    for c in header:
        if c not in m.columns:
            m[c] = pd.NA

    out = m[header].sort_values(KEYS).reset_index(drop=True)
    for c in INT_COLS:
        if c in out:
            out[c] = out[c].astype("Float64").round().astype("Int64")
    for c in TIME_COLS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(
                lambda v: f"{v:.3f}" if pd.notna(v) else pd.NA)

    out.to_csv(a.out, index=False, na_rep="N/A")
    print(f"[ok] {a.out}: {len(out)} rows, {len(header)} columns")
    print("     empty (not in Imed's files):",
          [c for c in header if out[c].isna().all()])


if __name__ == "__main__":
    main()