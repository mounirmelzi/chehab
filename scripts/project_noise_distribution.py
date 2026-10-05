#!/usr/bin/env python3
"""
Project noise distribution of current dataset vs. augmented dataset.

Reads the noise_estimator_stats_dataset.csv, computes the current noise
distribution by multiplicative depth, then simulates an augmented dataset
with ~6000 high-MultDepth expressions and outputs an Excel workbook with
three sheets: Current Distribution, Augmented Distribution, Per-Budget Analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

NOISE_COEFF_PER_MULTDEPTH = 33.383  # from NoiseEstimator model

EXACT_BUDGETS = [172, 230, 236, 369, 9_000_000]
BUDGET_LABELS = {
    172: "172 (n=16384, 256-bit, t=65537)",
    230: "230 (n=16384, 192-bit, t=786433)",
    236: "236 (n=16384, 192-bit, t=12289)",
    369: "369 (n=16384, 128-bit, t=786433)",
    9_000_000: "9M  (unconstrained)",
}

AUGMENTATION_PLAN = {
    5: 1500,
    6: 1500,
    7: 1200,
    8: 800,
    9: 600,
    10: 400,
}


def estimate_noise_for_multdepth(md):
    """Approximate consumed noise for a given multiplicative depth."""
    return md * NOISE_COEFF_PER_MULTDEPTH


def build_distribution_table(df, label):
    """Build a distribution table grouped by Multiplicative Depth."""
    total = len(df)
    groups = df.groupby("Multiplicative Depth").agg(
        Count=("used_noise", "size"),
        Min_Noise=("used_noise", "min"),
        Max_Noise=("used_noise", "max"),
        Mean_Noise=("used_noise", "mean"),
        Median_Noise=("used_noise", "median"),
    ).reset_index()
    groups["Percentage (%)"] = (groups["Count"] / total * 100).round(2)
    groups["Min_Noise"] = groups["Min_Noise"].round(1)
    groups["Max_Noise"] = groups["Max_Noise"].round(1)
    groups["Mean_Noise"] = groups["Mean_Noise"].round(1)
    groups["Median_Noise"] = groups["Median_Noise"].round(1)

    for b in EXACT_BUDGETS:
        col = f"Constrained at {b}"
        groups[col] = groups["Mean_Noise"].apply(
            lambda n: "YES" if n > b else "no"
        )

    groups = groups.sort_values("Multiplicative Depth").reset_index(drop=True)
    groups.rename(columns={"Multiplicative Depth": "MultDepth"}, inplace=True)

    cols = ["MultDepth", "Count", "Percentage (%)", "Min_Noise", "Max_Noise",
            "Mean_Noise", "Median_Noise"] + [
        f"Constrained at {b}" for b in EXACT_BUDGETS
    ]
    return groups[cols]


def build_per_budget_analysis(df_current, df_augmented):
    """For each budget, show how many expressions are constrained."""
    rows = []
    for b in EXACT_BUDGETS:
        cur_total = len(df_current)
        cur_above = int((df_current["used_noise"] > b).sum())
        aug_total = len(df_augmented)
        aug_above = int((df_augmented["used_noise"] > b).sum())
        rows.append({
            "Budget": b,
            "Budget Config": BUDGET_LABELS[b],
            "Current Total": cur_total,
            "Current Constrained": cur_above,
            "Current Constrained (%)": round(cur_above / cur_total * 100, 2),
            "Augmented Total": aug_total,
            "Augmented Constrained": aug_above,
            "Augmented Constrained (%)": round(aug_above / aug_total * 100, 2),
            "New Constrained Expressions": aug_above - cur_above,
        })
    return pd.DataFrame(rows)


def build_percentile_comparison(df_current, df_augmented):
    """Compare percentile statistics between current and augmented datasets."""
    pcts = [10, 25, 50, 75, 90, 95, 99]
    rows = []
    for p in pcts:
        rows.append({
            "Percentile": f"P{p}",
            "Current Noise": round(np.percentile(df_current["used_noise"], p), 1),
            "Augmented Noise": round(np.percentile(df_augmented["used_noise"], p), 1),
        })
    rows.append({
        "Percentile": "Max",
        "Current Noise": round(df_current["used_noise"].max(), 1),
        "Augmented Noise": round(df_augmented["used_noise"].max(), 1),
    })
    rows.append({
        "Percentile": "Count",
        "Current Noise": len(df_current),
        "Augmented Noise": len(df_augmented),
    })
    return pd.DataFrame(rows)


def main():
    base = Path(__file__).resolve().parent.parent
    stats_path = base / "RL" / "fhe_rl" / "datasets" / "noise_estimator_stats_dataset.csv"
    output_path = base / "generate_statistics_graphs" / "projected_augmented_noise_distribution.xlsx"

    initial_budget = 369
    df = pd.read_csv(stats_path)
    df["used_noise"] = initial_budget - df["Remaining_noise_budget"]

    # --- Simulate augmented expressions ---
    np.random.seed(42)
    aug_rows = []
    for md, count in AUGMENTATION_PLAN.items():
        mean_noise = estimate_noise_for_multdepth(md)
        noises = np.random.normal(loc=mean_noise, scale=5.0, size=count)
        for n in noises:
            aug_rows.append({
                "Multiplicative Depth": md,
                "used_noise": max(0, n),
                "source": "generated",
            })

    df["source"] = "original"
    df_aug_new = pd.DataFrame(aug_rows)
    df_combined = pd.concat([
        df[["Multiplicative Depth", "used_noise", "source"]],
        df_aug_new,
    ], ignore_index=True)

    # --- Build tables ---
    current_dist = build_distribution_table(df, "Current")
    augmented_dist = build_distribution_table(df_combined, "Augmented")
    per_budget = build_per_budget_analysis(df, df_combined)
    percentiles = build_percentile_comparison(df, df_combined)

    # --- Write Excel ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        current_dist.to_excel(writer, sheet_name="Current Distribution", index=False)
        augmented_dist.to_excel(writer, sheet_name="Augmented Distribution", index=False)
        per_budget.to_excel(writer, sheet_name="Per-Budget Analysis", index=False)
        percentiles.to_excel(writer, sheet_name="Percentile Comparison", index=False)

    print(f"Written to: {output_path}")
    print(f"  Current dataset:   {len(df):,} expressions")
    print(f"  Augmented dataset: {len(df_combined):,} expressions (+{len(df_aug_new):,} generated)")
    print()
    print("Per-Budget Analysis:")
    print(per_budget.to_string(index=False))
    print()
    print("Percentile Comparison:")
    print(percentiles.to_string(index=False))


if __name__ == "__main__":
    main()
