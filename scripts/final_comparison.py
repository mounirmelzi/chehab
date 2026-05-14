"""Final comparison: Constrained Agent (with Safety Rollback) vs Unconstrained PPO.

Generates a comprehensive Excel workbook with multiple analysis sheets
suitable for inclusion in a research paper.

Usage:
    python scripts/final_comparison.py <constrained_results.xlsx>

Example:
    python scripts/final_comparison.py test_results/test_v2_v4_mb_B_safety.xlsx
    python scripts/final_comparison.py test_results/test_v2_v5_ppo_natosc_A.xlsx
"""

import sys, os
import pandas as pd
import numpy as np

UNC_PATH = os.path.join(os.path.dirname(__file__), "..", "test_results", "test_aug_unconstrained_ppo.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_results")


def load_unconstrained():
    df = pd.read_excel(UNC_PATH, sheet_name="all_results")
    return df


def load_constrained(path):
    df = pd.read_excel(path, sheet_name="all_results")
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/final_comparison.py <constrained_results.xlsx>")
        sys.exit(1)

    constr_path = sys.argv[1]
    agent_name = os.path.basename(constr_path).replace("test_v2_", "").replace(".xlsx", "")

    unc = load_unconstrained()
    constr = load_constrained(constr_path)

    has_safety = "Safe Cost Reduction (%)" in constr.columns

    common_budgets = sorted(set(unc["Budget"].unique()) & set(constr["Budget"].unique()))
    if not common_budgets:
        print("ERROR: No common budgets between unconstrained and constrained results.")
        print(f"  Unconstrained budgets: {sorted(unc['Budget'].unique())}")
        print(f"  Constrained budgets: {sorted(constr['Budget'].unique())}")
        sys.exit(1)

    print(f"Comparing: {agent_name} vs Unconstrained PPO")
    print(f"Common budgets: {common_budgets}")
    print(f"Has safety rollback: {has_safety}")

    output_path = os.path.join(OUTPUT_DIR, f"FINAL_COMPARISON_{agent_name}.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ============================================================
        # SHEET 1: Per-Budget Summary Comparison
        # ============================================================
        summary_rows = []
        for budget in common_budgets:
            unc_b = unc[unc["Budget"] == budget]
            con_b = constr[constr["Budget"] == budget]

            unc_cr = unc_b["Cost Reduction (%)"].mean()
            unc_viol = (unc_b["Budget Violated"] == "YES").sum()
            unc_viol_pct = unc_viol / len(unc_b) * 100

            if has_safety:
                con_cr = con_b["Safe Cost Reduction (%)"].mean()
                con_viol = con_b["Safe Violated"].sum() if con_b["Safe Violated"].dtype == bool else (con_b["Safe Violated"] == "YES").sum()
                agent_cr = con_b["Agent Cost Reduction (%)"].mean()
            else:
                con_cr = con_b["Cost Reduction (%)"].mean()
                con_viol = (con_b["Budget Violated"] == "YES").sum()
                agent_cr = con_cr

            con_viol_pct = con_viol / len(con_b) * 100

            feasible_count = int(con_b["Is Feasible"].sum()) if "Is Feasible" in con_b.columns else "N/A"

            row = {
                "Budget (bits)": budget,
                "N Expressions": len(unc_b),
                "Feasible Count": feasible_count,
                "Unconstrained CR (%)": round(unc_cr, 2),
                "Unconstrained Violations": int(unc_viol),
                "Unconstrained Viol Rate (%)": round(unc_viol_pct, 1),
                f"{agent_name} Agent CR (%)": round(agent_cr, 2),
                f"{agent_name} Safe CR (%)": round(con_cr, 2),
                f"{agent_name} Violations": int(con_viol),
                f"{agent_name} Viol Rate (%)": round(con_viol_pct, 1),
                "CR Advantage (%)": round(con_cr - unc_cr, 2),
                "Violations Avoided": int(unc_viol - con_viol),
            }
            summary_rows.append(row)

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="budget_comparison", index=False)

        # ============================================================
        # SHEET 2: Feasible-Only Comparison (primary metric)
        # ============================================================
        if "Is Feasible" in constr.columns:
            feasible_rows = []
            for budget in common_budgets:
                unc_b = unc[unc["Budget"] == budget]
                con_b = constr[(constr["Budget"] == budget) & (constr["Is Feasible"] == True)]

                if len(con_b) == 0:
                    continue

                feasible_exprs = set(con_b["Expression #"].values)
                unc_feasible = unc_b[unc_b["Expression #"].isin(feasible_exprs)]

                unc_cr = unc_feasible["Cost Reduction (%)"].mean()
                unc_viol = (unc_feasible["Budget Violated"] == "YES").sum()

                if has_safety:
                    con_cr = con_b["Safe Cost Reduction (%)"].mean()
                    con_viol = con_b["Safe Violated"].sum() if con_b["Safe Violated"].dtype == bool else 0
                else:
                    con_cr = con_b["Cost Reduction (%)"].mean()
                    con_viol = (con_b["Budget Violated"] == "YES").sum()

                feasible_rows.append({
                    "Budget (bits)": budget,
                    "Feasible Expressions": len(con_b),
                    "Unconstrained CR (%)": round(unc_cr, 2),
                    "Unconstrained Violations": int(unc_viol),
                    f"{agent_name} Safe CR (%)": round(con_cr, 2),
                    f"{agent_name} Violations": int(con_viol),
                    "CR Advantage (%)": round(con_cr - unc_cr, 2),
                    "Guarantee": "0 violations" if con_viol == 0 else f"{con_viol} violations",
                })

            if feasible_rows:
                pd.DataFrame(feasible_rows).to_excel(writer, sheet_name="feasible_comparison", index=False)

        # ============================================================
        # SHEET 3: Per-Expression Detail (at each common budget)
        # ============================================================
        detail_rows = []
        for budget in common_budgets:
            unc_b = unc[unc["Budget"] == budget].set_index("Expression #")
            con_b = constr[constr["Budget"] == budget].set_index("Expression #")

            for expr_num in sorted(set(unc_b.index) & set(con_b.index)):
                u = unc_b.loc[expr_num]
                c = con_b.loc[expr_num]

                unc_cr = u["Cost Reduction (%)"]
                unc_violated = u["Budget Violated"] == "YES"
                unc_noise = u["Final Noise"]

                if has_safety:
                    con_cr = c["Safe Cost Reduction (%)"]
                    con_violated = bool(c["Safe Violated"])
                    con_noise = c["Safe Final Noise"]
                    agent_cr_raw = c["Agent Cost Reduction (%)"]
                else:
                    con_cr = c["Cost Reduction (%)"]
                    con_violated = c["Budget Violated"] == "YES"
                    con_noise = c.get("Final Noise", 0)
                    agent_cr_raw = con_cr

                is_showcase = (unc_violated and not con_violated and con_cr > 15)

                detail_rows.append({
                    "Budget": budget,
                    "Expression #": expr_num,
                    "Initial Cost": u["Initial Cost"],
                    "Initial Noise": u["Initial Noise"],
                    "Unc Final Cost": u["Final Cost"],
                    "Unc CR (%)": round(unc_cr, 2),
                    "Unc Final Noise": round(unc_noise, 2),
                    "Unc Violated": "YES" if unc_violated else "NO",
                    "Ours Final Cost": c.get("Safe Final Cost", c.get("Final Cost", "")),
                    "Ours CR (%)": round(con_cr, 2),
                    "Ours Final Noise": round(con_noise, 2),
                    "Ours Violated": "YES" if con_violated else "NO",
                    "CR Advantage (%)": round(con_cr - unc_cr, 2),
                    "Showcase": "YES" if is_showcase else "",
                })

        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="per_expression", index=False)

        # ============================================================
        # SHEET 4: Showcase Expressions (best cases for paper)
        # ============================================================
        showcase_df = pd.DataFrame([r for r in detail_rows if r["Showcase"] == "YES"])
        if len(showcase_df) > 0:
            showcase_df = showcase_df.sort_values("CR Advantage (%)", ascending=False)
            showcase_df.to_excel(writer, sheet_name="showcase_for_paper", index=False)

        # ============================================================
        # SHEET 5: Expression Difficulty Buckets
        # ============================================================
        bucket_rows = []
        for budget in common_budgets:
            unc_b = unc[unc["Budget"] == budget]
            con_b = constr[constr["Budget"] == budget]

            for label, noise_lo, noise_hi in [
                ("Low noise (<100)", 0, 100),
                ("Medium noise (100-230)", 100, 230),
                ("High noise (230-370)", 230, 370),
                ("Very high noise (>370)", 370, 99999),
            ]:
                unc_bucket = unc_b[(unc_b["Initial Noise"] >= noise_lo) & (unc_b["Initial Noise"] < noise_hi)]
                con_bucket = con_b[(con_b["Initial Noise"] >= noise_lo) & (con_b["Initial Noise"] < noise_hi)]

                if len(unc_bucket) == 0:
                    continue

                unc_cr = unc_bucket["Cost Reduction (%)"].mean()
                unc_viol = (unc_bucket["Budget Violated"] == "YES").sum()

                if has_safety and len(con_bucket) > 0:
                    con_cr = con_bucket["Safe Cost Reduction (%)"].mean()
                    con_viol = con_bucket["Safe Violated"].sum() if con_bucket["Safe Violated"].dtype == bool else 0
                elif len(con_bucket) > 0:
                    con_cr = con_bucket["Cost Reduction (%)"].mean()
                    con_viol = (con_bucket["Budget Violated"] == "YES").sum()
                else:
                    con_cr = 0
                    con_viol = 0

                bucket_rows.append({
                    "Budget": budget,
                    "Noise Bucket": label,
                    "N Expressions": len(unc_bucket),
                    "Unc CR (%)": round(unc_cr, 2),
                    "Unc Violations": int(unc_viol),
                    "Ours CR (%)": round(con_cr, 2),
                    "Ours Violations": int(con_viol),
                    "CR Advantage (%)": round(con_cr - unc_cr, 2),
                })

        pd.DataFrame(bucket_rows).to_excel(writer, sheet_name="difficulty_buckets", index=False)

        # ============================================================
        # SHEET 6: Key Findings Summary (text)
        # ============================================================
        all_detail = pd.DataFrame(detail_rows)
        total_unc_viol = (all_detail["Unc Violated"] == "YES").sum()
        total_ours_viol = (all_detail["Ours Violated"] == "YES").sum()
        total_showcase = (all_detail["Showcase"] == "YES").sum()

        ours_better_cr = (all_detail["CR Advantage (%)"] > 0).sum()
        unc_better_cr = (all_detail["CR Advantage (%)"] < 0).sum()

        b9m = all_detail[all_detail["Budget"] == 9000000]
        cr_9m_unc = b9m["Unc CR (%)"].mean() if len(b9m) > 0 else 0
        cr_9m_ours = b9m["Ours CR (%)"].mean() if len(b9m) > 0 else 0

        findings = [
            {"Finding": "Total expression-budget pairs compared", "Value": len(all_detail)},
            {"Finding": "Unconstrained total violations", "Value": total_unc_viol},
            {"Finding": "Our method total violations", "Value": total_ours_viol},
            {"Finding": "Violations eliminated by our method", "Value": total_unc_viol - total_ours_viol},
            {"Finding": "Showcase cases (unc violates, ours doesn't + good CR)", "Value": total_showcase},
            {"Finding": "Cases where our CR > unconstrained CR", "Value": ours_better_cr},
            {"Finding": "Cases where unconstrained CR > our CR", "Value": unc_better_cr},
            {"Finding": "Unconstrained avg CR at 9M budget", "Value": f"{cr_9m_unc:.2f}%"},
            {"Finding": "Our method avg CR at 9M budget", "Value": f"{cr_9m_ours:.2f}%"},
            {"Finding": "CR advantage at 9M (ours - unc)", "Value": f"{cr_9m_ours - cr_9m_unc:+.2f}%"},
        ]
        pd.DataFrame(findings).to_excel(writer, sheet_name="key_findings", index=False)

    print(f"\nFinal comparison written to: {output_path}")
    print(f"\nSheets:")
    print(f"  1. budget_comparison     — Per-budget head-to-head")
    print(f"  2. feasible_comparison   — Feasible-only (primary metric)")
    print(f"  3. per_expression        — Every expression detail")
    print(f"  4. showcase_for_paper    — Best cases for paper figures")
    print(f"  5. difficulty_buckets    — By expression noise difficulty")
    print(f"  6. key_findings          — Summary statistics")

    print(f"\n{'='*60}")
    print(f"  KEY RESULTS")
    print(f"{'='*60}")
    for f in findings:
        print(f"  {f['Finding']:50s} {f['Value']}")


if __name__ == "__main__":
    main()
