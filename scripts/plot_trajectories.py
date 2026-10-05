"""Plot cost & noise trajectories for selected expressions.

Generates paper-ready figures showing:
- Cost and noise evolution over agent steps
- Budget threshold line
- Best valid checkpoint (safety rollback point)
- Agent's final (potentially bad) state

Usage:
    python scripts/plot_trajectories.py test_results/test_v2_v4_mb_B_safety.xlsx
    python scripts/plot_trajectories.py test_results/test_v2_v5_ppo_natosc_A.xlsx

Outputs PNGs to scripts/trajectory_plots/
"""

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plot_expression_trajectory(row, output_dir):
    budget = row["Budget"]
    expr_num = int(row["Expression #"])
    initial_cost = row["Initial Cost"]

    costs = [int(c) for c in str(row["Trajectory Costs"]).split("|")]
    noises = [float(n) for n in str(row["Trajectory Noises"]).split("|")]
    steps = list(range(len(costs)))

    best_step = int(row["Best Valid Step"])
    total_steps = int(row["Total Steps"])
    safe_cost = row["Safe Final Cost"]
    safe_noise = row["Safe Final Noise"]

    cost_reductions = [(initial_cost - c) / initial_cost * 100 for c in costs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Expression #{expr_num}  —  Budget = {budget} bits",
                 fontsize=14, fontweight="bold")

    # --- Cost plot ---
    ax1.plot(steps, cost_reductions, "o-", color="#2196F3", markersize=4,
             linewidth=1.5, label="Cost Reduction (%)")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    if best_step < len(cost_reductions):
        ax1.plot(best_step, cost_reductions[best_step], "s", color="#4CAF50",
                 markersize=12, zorder=5, label=f"Safety checkpoint (step {best_step})")
    ax1.plot(steps[-1], cost_reductions[-1], "X", color="#F44336",
             markersize=12, zorder=5, label=f"Agent final (step {total_steps})")

    ax1.set_ylabel("Cost Reduction (%)", fontsize=12)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Noise plot ---
    ax2.plot(steps, noises, "o-", color="#FF9800", markersize=4,
             linewidth=1.5, label="Noise (SEAL bits)")
    ax2.axhline(y=budget, color="#F44336", linestyle="--", linewidth=2,
                alpha=0.8, label=f"Budget = {budget}")

    valid_mask = [n <= budget for n in noises]
    valid_steps = [s for s, v in zip(steps, valid_mask) if v]
    valid_noises = [n for n, v in zip(noises, valid_mask) if v]
    invalid_steps = [s for s, v in zip(steps, valid_mask) if not v]
    invalid_noises = [n for n, v in zip(noises, valid_mask) if not v]

    if valid_steps:
        ax2.fill_between(steps, 0, budget, alpha=0.08, color="#4CAF50")

    if best_step < len(noises):
        ax2.plot(best_step, noises[best_step], "s", color="#4CAF50",
                 markersize=12, zorder=5, label=f"Safety checkpoint (noise={noises[best_step]:.1f})")
    ax2.plot(steps[-1], noises[-1], "X", color="#F44336",
             markersize=12, zorder=5, label=f"Agent final (noise={noises[-1]:.1f})")

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Noise Budget Used (bits)", fontsize=12)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"trajectory_expr{expr_num}_budget{budget}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_trajectories.py <results.xlsx>")
        sys.exit(1)

    xlsx_path = sys.argv[1]
    df = pd.read_excel(xlsx_path, sheet_name="all_results")

    if "Trajectory Costs" not in df.columns:
        print("ERROR: No trajectory data in this file. Re-run test with updated test.py.")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "trajectory_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Auto-select interesting expressions: safety activated + good safe CR
    interesting = df[
        (df["Safety Activated"] == True) &
        (df["Safe Cost Reduction (%)"] > 30) &
        (df["Total Steps"] > 3)
    ].sort_values("Safe Cost Reduction (%)", ascending=False)

    # Pick up to 6 diverse examples (different budgets and expression numbers)
    seen_exprs = set()
    selected = []
    for _, row in interesting.iterrows():
        if row["Expression #"] not in seen_exprs and len(selected) < 6:
            seen_exprs.add(row["Expression #"])
            selected.append(row)

    if not selected:
        print("No interesting trajectories found. Plotting first 4 with safety activated.")
        selected = [row for _, row in df[df["Safety Activated"] == True].head(4).iterrows()]

    print(f"Plotting {len(selected)} trajectories from {xlsx_path}")
    for row in selected:
        plot_expression_trajectory(row, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
