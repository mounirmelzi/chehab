#!/usr/bin/env python3
"""Match Excel violating expressions to local file indices by (cost, noise)."""
import sys, os
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(RL_DIR / "pytrs"))
os.chdir(str(RL_DIR))

import pandas as pd
from fhe_rl.utils import load_expressions
from pytrs import NoiseEstimator, parse_sexpr, calculate_cost

exprs = load_expressions("fhe_rl/datasets/benchmarks_69_augmented.txt")
ne = NoiseEstimator()
print(f"Loaded {len(exprs)} expressions locally")

local_data = []
for i, e in enumerate(exprs):
    parsed = parse_sexpr(e)
    noise = float(ne.estimate(parsed))
    c = float(calculate_cost(parsed))
    local_data.append((i, c, noise))

unc_df = pd.read_excel("../test_results/test_aug_unconstrained_ppo.xlsx", sheet_name="all_results")
unc_b369 = unc_df[unc_df["Budget"] == 369]
violations = unc_b369[unc_b369["Final Noise"] > 369]

print(f"\nMatching {len(violations)} violating expressions from Excel to local indices...\n")
print(f"{'Excel#':>7} {'ExlNoise':>9} {'ExlCost':>9} {'FinalN':>8} {'CR%':>7} => {'LocalIdx':>9} {'LocNoise':>9} {'LocCost':>9}")
print("-" * 90)

matched = []
for _, row in violations.iterrows():
    excel_num = int(row["Expression #"])
    excel_noise = float(row["Initial Noise"])
    excel_cost = float(row["Initial Cost"])
    excel_final_noise = float(row["Final Noise"])
    excel_cr = float(row["Cost Reduction (%)"])

    best_match = None
    best_diff = 999999
    for li, lc, ln in local_data:
        cost_diff = abs(lc - excel_cost)
        noise_diff = abs(ln - excel_noise)
        total_diff = cost_diff + noise_diff
        if total_diff < best_diff:
            best_diff = total_diff
            best_match = (li, lc, ln)

    if best_match and best_diff < 5.0:
        li, lc, ln = best_match
        print(f"{excel_num:>7} {excel_noise:>9.1f} {excel_cost:>9.0f} {excel_final_noise:>8.1f} {excel_cr:>6.1f}% => idx={li:>4} noise={ln:>9.1f} cost={lc:>9.0f}  (diff={best_diff:.2f})")
        matched.append((excel_num, li, excel_final_noise, excel_cr))
    else:
        print(f"{excel_num:>7} {excel_noise:>9.1f} {excel_cost:>9.0f} {excel_final_noise:>8.1f} {excel_cr:>6.1f}% => NO MATCH (diff={best_diff:.1f})")

print(f"\nMatched {len(matched)}/{len(violations)} expressions")
print("\nBest candidates for demo (high violation + good CR):")
for en, li, fn, cr in sorted(matched, key=lambda x: -x[2]):
    if cr > 20:
        print(f"  Excel#{en} => local idx {li}: UNC_noise={fn:.1f}, CR={cr:.1f}%")
