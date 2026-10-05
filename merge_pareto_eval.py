#!/usr/bin/env python3
"""Fusionne les deux CSV produits par CHEHAB (branche chehab-morl).

  A) pareto_2_full_train.csv : métriques compilateur + exécution FHE réelle
     (ops HE, depth, temps, rotation_key_size (MB), Final_Cost)
  B) eval_morl_final_2.csv   : sortie de l'agent RL (final_exec_cost,
     final_keys_cost = nombre de clés de rotation prédit par le cost model)

Faits vérifiés sur les données :
  * clé de jointure = (benchmark, w_ops, w_keys)   (benchmark = nom_slotcount)
  * Final_Cost (A) == final_exec_cost + final_keys_cost (B)  -> 200/200 lignes
  * rotation_key_size (MB) (A) ~= 18.2 MB * final_keys_cost (B)
  * colonnes communes (add, sub, ..., Depth) identiques quand présentes des 2 côtés
  * compile_time (s) diffère (runs différents) -> les deux sont gardées

Usage:
    python merge_pareto_eval.py
    python merge_pareto_eval.py -a pareto_2_full_train.csv -b eval_morl_final_2.csv \
        -o merged.csv --how outer
"""
import argparse
import numpy as np
import pandas as pd

KEYS = ["benchmark", "w_ops", "w_keys"]
SHARED = ["add", "sub", "multiply_plain", "rotate_rows", "negate",
          "multiply", "Depth", "Multiplicative Depth"]

# Noms utilisés par les différentes versions de run_benchmarks.py /
# run_morl_benchmarks.py / benchmark_runner.py du repo
ALIASES = {
    "Multplicative Depth": "Multiplicative Depth",   # typo dans le pareto
    "rotation_keys_size (MB)": "rotation_key_size (MB)",
    "final_ops_cost": "final_exec_cost",
    "final_cost": "Final_Cost",
}


def load(path):
    # run_morl_benchmarks.py écrit "N/A" quand une métrique manque
    df = pd.read_csv(path, na_values=["N/A", "NA", "", "nan"])
    df.columns = df.columns.str.strip()
    df = df.rename(columns=ALIASES)
    for c in ("w_ops", "w_keys"):
        df[c] = df[c].round(6)      # évite 0.1 vs 0.10000000001
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-a", "--pareto", default="pareto_2_full_train.csv")
    p.add_argument("-b", "--eval", default="eval_morl_final_2.csv")
    p.add_argument("-o", "--out", default="merged_pareto_eval.csv")
    p.add_argument("--how", default="outer",
                   choices=["outer", "inner", "left", "right"],
                   help="outer = toutes les lignes des 2 fichiers (défaut)")
    args = p.parse_args()

    a, b = load(args.pareto), load(args.eval)

    for name, df in (("pareto", a), ("eval", b)):
        d = df.duplicated(KEYS).sum()
        if d:
            print(f"[warn] {d} clés dupliquées dans {name} (dernière gardée)")
            df.drop_duplicates(KEYS, keep="last", inplace=True)

    m = a.merge(b, on=KEYS, how=args.how, suffixes=("_pareto", "_eval"),
                indicator="source")
    m["source"] = m["source"].map({"left_only": "pareto_only",
                                   "right_only": "eval_only",
                                   "both": "both"})

    # colonnes communes : valeur du pareto, sinon celle de l'eval
    for c in SHARED:
        m[c] = m[f"{c}_pareto"].combine_first(m[f"{c}_eval"])
        m = m.drop(columns=[f"{c}_pareto", f"{c}_eval"])

    m = m.rename(columns={"compile_time (s)_pareto": "compile_time_pareto (s)",
                          "compile_time (s)_eval": "compile_time_eval (s)",
                          "final_keys_cost": "rotation_keys_count"})

    # Final_Cost complet : exec + keys (vérifié == Final_Cost du pareto)
    total = m["final_exec_cost"] + m["rotation_keys_count"]
    both = m["Final_Cost"].notna() & total.notna()
    mismatch = (m.loc[both, "Final_Cost"] != total[both]).sum()
    print(f"Contrôle Final_Cost == exec+keys : {both.sum() - mismatch}/{both.sum()} OK")
    m["Final_Cost_full"] = m["Final_Cost"].combine_first(total)

    first = KEYS + SHARED
    rest = [c for c in m.columns if c not in first + ["source"]]
    m = m[first + rest + ["source"]]
    m = m.sort_values(["benchmark", "w_ops"], ascending=[True, False])
    m.to_csv(args.out, index=False)

    print(f"pareto: {len(a)} | eval: {len(b)} -> fusionné ({args.how}): "
          f"{len(m)} lignes, {m.shape[1]} colonnes -> {args.out}")
    print(m["source"].value_counts().to_string())


if __name__ == "__main__":
    main()