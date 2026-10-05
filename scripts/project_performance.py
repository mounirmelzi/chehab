#!/usr/bin/env python3
"""
Project estimated performance of 3 best RL methods on 5 exact SEAL-derived
noise budgets.

Methodology: linear interpolation/extrapolation from observed V3 per-budget
test results.  No magical improvement factors -- the methods don't change,
only budgets and dataset composition change slightly.
"""

import pandas as pd
import numpy as np
from pathlib import Path

EXACT_BUDGETS = [172, 230, 236, 369, 9_000_000]

METHODS = {
    "V1 Margin Barrier 10b": {
        "file": "test_margin_barrier_10b_8envs.xlsx",
        "trained_budgets": "240..1000,9M",
    },
    "V3 MB + FiLM 3b": {
        "file": "test_v3_mb_film_3b.xlsx",
        "trained_budgets": "200,300,9M",
    },
    "V3 Noise Masking 5b": {
        "file": "test_v3_nmask_5b.xlsx",
        "trained_budgets": "180..240,9M",
    },
}


def interp_extrapolate(x_new, xs, ys):
    """Linear interpolation with linear extrapolation at the edges."""
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    if x_new <= xs[0]:
        if len(xs) >= 2:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0
            return ys[0] + slope * (x_new - xs[0])
        return ys[0]
    if x_new >= xs[-1]:
        if len(xs) >= 2:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0
            return ys[-1] + slope * (x_new - xs[-1])
        return ys[-1]
    return float(np.interp(x_new, xs, ys))


def load_observed_summary(test_results_dir, filename):
    """Load the summary_per_budget sheet from a test result file."""
    path = test_results_dir / filename
    df = pd.read_excel(path, sheet_name="summary_per_budget")
    return df


def load_baseline_per_expression(test_results_dir):
    """Load unconstrained PPO baseline per-expression results at 9M budget."""
    path = test_results_dir / "test_unconstrained_ppo_baseline.xlsx"
    df = pd.read_excel(path, sheet_name="all_results")
    return df[df["Budget"] == 9_000_000].copy()


def project_baseline(baseline_expr_df):
    """
    Project unconstrained PPO on exact budgets.
    The baseline has NO constraint awareness -- its final noise is fixed
    regardless of budget.  Violations are counted directly.
    """
    n_expr = len(baseline_expr_df)
    base_cost_red = baseline_expr_df["Cost Reduction (%)"].mean()
    base_noise = baseline_expr_df["Final Noise"].mean()
    base_steps = baseline_expr_df["Steps"].mean()

    rows = []
    for b in EXACT_BUDGETS:
        violations = int((baseline_expr_df["Final Noise"] > b).sum())
        rows.append({
            "Budget": b,
            "Num Expressions": n_expr,
            "Avg Cost Reduction (%)": round(base_cost_red, 2),
            "Avg Final Noise": round(base_noise, 2),
            "Violations": violations,
            "Violation Rate (%)": round(violations / n_expr * 100, 1),
            "Avg Noise Margin": round(b - base_noise, 2),
            "Avg Steps": round(base_steps, 1),
        })
    return pd.DataFrame(rows)


def project_constrained_method(obs_df, method_name):
    """
    Project a constrained method on exact budgets by interpolating from
    its observed per-budget performance.

    For budgets WITHIN the observed constrained range: linear interpolation.
    For budgets BELOW the min observed: linear extrapolation from the two
      lowest observed points.
    For budgets ABOVE the max observed constrained budget but < 9M:
      interpolate between the highest constrained observation and the 9M
      unconstrained result (converging to unconstrained behaviour).
    """
    constrained = obs_df[obs_df["Budget"] < 1_000_000].copy()
    unconstrained = obs_df[obs_df["Budget"] >= 1_000_000].iloc[0]

    obs_budgets = constrained["Budget"].values
    obs_cost_red = constrained["Avg Cost Reduction (%)"].values
    obs_viol_rate = constrained["Violation Rate (%)"].values
    obs_noise = constrained["Avg Final Noise"].values
    obs_steps = constrained["Avg Steps"].values

    max_obs_budget = obs_budgets.max()

    n_expr = int(unconstrained["Num Expressions"])
    unc_cost_red = unconstrained["Avg Cost Reduction (%)"]
    unc_noise = unconstrained["Avg Final Noise"]
    unc_steps = unconstrained["Avg Steps"]

    rows = []
    for b in EXACT_BUDGETS:
        if b >= 1_000_000:
            rows.append({
                "Budget": b,
                "Num Expressions": n_expr,
                "Avg Cost Reduction (%)": round(unc_cost_red, 2),
                "Avg Final Noise": round(unc_noise, 2),
                "Violations": 0,
                "Violation Rate (%)": 0.0,
                "Avg Noise Margin": round(b - unc_noise, 2),
                "Avg Steps": round(unc_steps, 1),
            })
            continue

        if b > max_obs_budget:
            # Beyond observed constrained range: blend toward unconstrained.
            # Use the highest constrained observation as anchor.
            idx_max = np.argmax(obs_budgets)
            top_cost = obs_cost_red[idx_max]
            top_noise = obs_noise[idx_max]
            top_steps = obs_steps[idx_max]
            # Fraction of the way from max_obs_budget to 9M
            frac = min(1.0, (b - max_obs_budget) / (9_000_000 - max_obs_budget))
            cost_red = top_cost + frac * (unc_cost_red - top_cost)
            noise = top_noise + frac * (unc_noise - top_noise)
            steps = top_steps + frac * (unc_steps - top_steps)
            viol_rate = 0.0  # well above training range → no violations
        else:
            # Within or below observed range: interpolate/extrapolate
            cost_red = interp_extrapolate(b, obs_budgets, obs_cost_red)
            viol_rate = interp_extrapolate(b, obs_budgets, obs_viol_rate)
            noise = interp_extrapolate(b, obs_budgets, obs_noise)
            steps = interp_extrapolate(b, obs_budgets, obs_steps)

        # Clamp to reasonable ranges
        viol_rate = max(0.0, min(100.0, viol_rate))
        cost_red = max(0.0, min(100.0, cost_red))
        noise = max(0.0, noise)
        steps = max(1.0, steps)

        # Round violations to discrete count (43 test expressions)
        violations = int(round(viol_rate * n_expr / 100))
        violations = max(0, min(n_expr, violations))
        actual_viol_rate = round(violations / n_expr * 100, 1)

        rows.append({
            "Budget": b,
            "Num Expressions": n_expr,
            "Avg Cost Reduction (%)": round(cost_red, 2),
            "Avg Final Noise": round(noise, 2),
            "Violations": violations,
            "Violation Rate (%)": actual_viol_rate,
            "Avg Noise Margin": round(b - noise, 2),
            "Avg Steps": round(steps, 1),
        })

    return pd.DataFrame(rows)


def build_quick_comparison(all_summaries):
    """Build a quick comparison table sorted by violation rate then cost red."""
    rows = []
    for method_name, df in all_summaries.items():
        constrained = df[df["Budget"] < 1_000_000]
        unconstrained = df[df["Budget"] >= 1_000_000]
        rows.append({
            "Agent": method_name,
            "Training Budgets": ", ".join(str(b) for b in EXACT_BUDGETS),
            "Overall Avg Cost Red (%)": round(df["Avg Cost Reduction (%)"].mean(), 2),
            "Overall Avg Violation (%)": round(df["Violation Rate (%)"].mean(), 2),
            "Constrained Avg Cost Red (%)": round(constrained["Avg Cost Reduction (%)"].mean(), 2) if len(constrained) else 0,
            "Constrained Avg Violation (%)": round(constrained["Violation Rate (%)"].mean(), 2) if len(constrained) else 0,
            "Unconstrained Cost Red (%)": round(unconstrained["Avg Cost Reduction (%)"].mean(), 2) if len(unconstrained) else 0,
        })
    out = pd.DataFrame(rows)
    out.sort_values(
        ["Overall Avg Violation (%)", "Overall Avg Cost Red (%)"],
        ascending=[True, False],
        inplace=True,
    )
    return out.reset_index(drop=True)


def main():
    base = Path(__file__).resolve().parent.parent
    test_dir = base / "test_results"
    out_path = base / "generate_statistics_graphs" / "projected_v4_performance.xlsx"

    # --- Baseline ---
    baseline_expr = load_baseline_per_expression(test_dir)
    baseline_proj = project_baseline(baseline_expr)

    all_summaries = {"Unconstrained PPO (baseline)": baseline_proj}

    # --- Constrained methods ---
    for name, info in METHODS.items():
        obs = load_observed_summary(test_dir, info["file"])
        proj = project_constrained_method(obs, name)
        all_summaries[name] = proj

    # --- Build stacked "All Methods" sheet ---
    cols = list(baseline_proj.columns)
    frames = []
    for method_name, proj_df in all_summaries.items():
        hdr = {c: "" for c in cols}
        hdr[cols[0]] = method_name
        frames.append(pd.DataFrame([hdr]))
        frames.append(proj_df.astype({cols[0]: object}))
        frames.append(pd.DataFrame([{c: "" for c in cols}]))

    all_methods_df = pd.concat(frames, ignore_index=True)
    quick_df = build_quick_comparison(all_summaries)

    # --- Write ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        all_methods_df.to_excel(w, sheet_name="All Methods", index=False)
        quick_df.to_excel(w, sheet_name="Quick Comparison", index=False)

    print(f"Written to: {out_path}")
    print()

    # Print per-method detail
    for name, df in all_summaries.items():
        print(f"--- {name} ---")
        print(df.to_string(index=False))
        print()

    print("Quick Comparison:")
    print(quick_df.to_string(index=False))
    print()
    print("NOTE: Projections based on linear interpolation/extrapolation")
    print("from observed V1/V3 per-budget test results on the existing")
    print("43-expression benchmark set.  The methods themselves are unchanged;")
    print("only budgets differ.  Actual results after retraining may vary.")


if __name__ == "__main__":
    main()
