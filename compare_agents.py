#!/usr/bin/env python3
"""
compare_agents.py
Compare your RL agent (results_RL_budget.csv) with Imed's agent (imed_merged.csv)
and write a compact, supervisor-friendly Excel workbook.

Usage:
    python compare_agents.py --mine results_RL_budget.csv --imed imed_merged.csv \
                             --out comparison_chehab_vs_imed.xlsx [--noise-budget 1000|all]

How the comparison is made (see the "Notes" sheet in the output too):
  * Only (benchmark, w_ops) pairs present in BOTH files with a valid result are compared.
  * Your file has 3 noise budgets (300/1000/9000), Imed's has none. Imed's
    Remaining_noise_budget (~325 on l2_distance_32) matches your noise_budget=1000 rows,
    so 1000 is the default reference. Use --noise-budget all to average your 3 budgets.
  * Imed's dense w_ops sweep is reduced to the w_ops values you also ran.
  * Tree benchmark names are aligned:  tree_5_1_50-50  <->  tree_50-50_5_1.
  * final_ops_cost / final_keys_cost: lower is better.
"""
import argparse
import re

import numpy as np
import pandas as pd
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
OPS_TOL = 0.01          # ops cost within 1 % -> tie
HEADER_FILL = PatternFill("solid", start_color="1F3864")
SUB_FILL = PatternFill("solid", start_color="D9E2F3")
GOOD_FILL = PatternFill("solid", start_color="C6EFCE")
BAD_FILL = PatternFill("solid", start_color="FFC7CE")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# --------------------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    df.columns = [c.strip() for c in df.columns]
    return df


def align_name(name):
    """tree_5_1_50-50 -> tree_50-50_5_1 (Imed's naming). Others unchanged."""
    m = re.fullmatch(r"tree_(\d+)_(\d+)_(\d+-\d+)", name)
    return f"tree_{m.group(3)}_{m.group(1)}_{m.group(2)}" if m else name


def family(name):
    if name.startswith("tree_"):
        return "tree"
    return re.sub(r"_\d+$", "", name)


def build_detail(mine, imed, noise_budget):
    mine = mine.copy()
    mine["bench"] = mine["benchmark"].map(align_name)
    mine["w_ops"] = mine["w_ops"].round(4)

    if noise_budget == "all":
        num = mine.select_dtypes("number").columns.difference(["w_ops", "w_keys", "noise_budget"])
        mine = mine.groupby(["bench", "w_ops"], as_index=False)[list(num)].mean()
    else:
        mine = mine[mine["noise_budget"] == int(noise_budget)]

    imed = imed.copy()
    imed["bench"] = imed["benchmark"]
    imed["w_ops"] = imed["w_ops"].round(4)
    common_w = sorted(set(mine["w_ops"]) & set(imed["w_ops"]))
    imed = imed[imed["w_ops"].isin(common_w)]

    cols = {
        "final_ops_cost": "ops",
        "final_keys_cost": "keys",
        "Multiplicative Depth": "mdepth",
        "compile_time (s)": "compile",
        "circuit_execution_time (s)": "exec",
        "Remaining_noise_budget": "noise",
    }
    a = mine[["bench", "w_ops"] + list(cols)].rename(columns=cols).add_suffix("_m")
    b = imed[["bench", "w_ops"] + list(cols)].rename(columns=cols).add_suffix("_i")
    a = a.rename(columns={"bench_m": "bench", "w_ops_m": "w_ops"})
    b = b.rename(columns={"bench_i": "bench", "w_ops_i": "w_ops"})
    d = a.merge(b, on=["bench", "w_ops"], how="outer")
    d["family"] = d["bench"].map(family)

    d["status"] = np.select(
        [d.ops_m.notna() & d.ops_i.notna(), d.ops_m.notna(), d.ops_i.notna()],
        ["compared", "Imed missing", "Mine missing"],
        default="both missing",
    )
    ok = d.status == "compared"
    d["ops_ratio"] = np.where(ok, d.ops_m / d.ops_i, np.nan)          # <1 = you are better
    d["ops_result"] = np.where(
        ~ok, "", np.where(d.ops_ratio < 1 - OPS_TOL, "Win",
                          np.where(d.ops_ratio > 1 + OPS_TOL, "Loss", "Tie")))
    d["keys_result"] = np.where(
        ~ok | d.keys_m.isna() | d.keys_i.isna(), "",
        np.where(d.keys_m < d.keys_i, "Win", np.where(d.keys_m > d.keys_i, "Loss", "Tie")))
    d["dominates"] = np.where(
        (d.ops_result != "") & (d.keys_result != ""),
        np.where(d.ops_result.isin(["Win", "Tie"]) & d.keys_result.isin(["Win", "Tie"])
                 & ((d.ops_result == "Win") | (d.keys_result == "Win")), "Yes",
                 np.where(d.ops_result.isin(["Loss", "Tie"]) & d.keys_result.isin(["Loss", "Tie"])
                          & ((d.ops_result == "Loss") | (d.keys_result == "Loss")), "Dominated", "Trade-off")),
        "")
    return d.sort_values(["family", "bench", "w_ops"]).reset_index(drop=True), common_w


# --------------------------------------------------------------------- aggregation
def geomean(x):
    x = x.dropna()
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def agg(g):
    c = g[g.status == "compared"]
    n = len(c)
    out = {
        "Compared points": n,
        "Ops cost - Mine (avg)": c.ops_m.mean() if n else np.nan,
        "Ops cost - Imed (avg)": c.ops_i.mean() if n else np.nan,
        "Ops ratio Mine/Imed (geo-mean)": geomean(c.ops_ratio),
        "Ops improvement vs Imed": (1 - geomean(c.ops_ratio)) if n else np.nan,
        "Ops W": int((c.ops_result == "Win").sum()),
        "Ops T": int((c.ops_result == "Tie").sum()),
        "Ops L": int((c.ops_result == "Loss").sum()),
        "Keys - Mine (avg)": c.keys_m.mean() if n else np.nan,
        "Keys - Imed (avg)": c.keys_i.mean() if n else np.nan,
        "Keys W": int((c.keys_result == "Win").sum()),
        "Keys T": int((c.keys_result == "Tie").sum()),
        "Keys L": int((c.keys_result == "Loss").sum()),
        "Pareto: Mine better": int((c.dominates == "Yes").sum()),
        "Pareto: trade-off": int((c.dominates == "Trade-off").sum()),
        "Pareto: Imed better": int((c.dominates == "Dominated").sum()),
        "Mult. depth - Mine": c.mdepth_m.mean() if n else np.nan,
        "Mult. depth - Imed": c.mdepth_i.mean() if n else np.nan,
        "Compile s - Mine": c.compile_m.mean() if n else np.nan,
        "Compile s - Imed": c.compile_i.mean() if n else np.nan,
    }
    return pd.Series(out)


# ------------------------------------------------------------------------- styling
PCT_COLS = {"Ops improvement vs Imed"}
RATIO_COLS = {"Ops ratio Mine/Imed (geo-mean)", "ops_ratio"}
INT_COLS = {"Compared points", "Pareto: Mine better", "Pareto: trade-off", "Pareto: Imed better", "Ops W", "Ops T", "Ops L", "Keys W", "Keys T", "Keys L"}


def write_table(ws, df, r0=1, c0=1, widths=True):
    for j, col in enumerate(df.columns):
        cell = ws.cell(row=r0, column=c0 + j, value=col)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            if isinstance(v, float) and np.isnan(v):
                v = None
            if isinstance(v, (np.integer,)):
                v = int(v)
            if isinstance(v, (np.floating,)):
                v = float(v)
            cell = ws.cell(row=r0 + i, column=c0 + j, value=v)
            col = df.columns[j]
            cell.font = Font(name=FONT, size=10)
            cell.border = BORDER
            if col in PCT_COLS:
                cell.number_format = "0.0%"
            elif col in RATIO_COLS:
                cell.number_format = "0.000"
            elif col in INT_COLS:
                cell.number_format = "0"
            elif isinstance(v, float):
                cell.number_format = "#,##0.0"
            if col in ("ops_result", "keys_result"):
                cell.alignment = Alignment(horizontal="center")
                if v == "Win":
                    cell.fill = GOOD_FILL
                elif v == "Loss":
                    cell.fill = BAD_FILL
            if col == "Ops improvement vs Imed" and isinstance(v, float):
                cell.fill = GOOD_FILL if v > 0.01 else (BAD_FILL if v < -0.01 else PatternFill())
    ws.row_dimensions[r0].height = 42
    if widths:
        for j, col in enumerate(df.columns, start=c0):
            mx = max([len(str(col)) * 0.6] + [len(str(x)) for x in df.iloc[:, j - c0].head(200)])
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, mx + 2), 26)


def title(ws, text, sub=None):
    ws["A1"] = text
    ws["A1"].font = Font(name=FONT, bold=True, size=14, color="1F3864")
    if sub:
        ws["A2"] = sub
        ws["A2"].font = Font(name=FONT, italic=True, size=9, color="595959")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", default="results_RL_budget.csv")
    ap.add_argument("--imed", default="imed_merged.csv")
    ap.add_argument("--out", default="comparison_chehab_vs_imed.xlsx")
    ap.add_argument("--noise-budget", default="1000", help="300 | 1000 | 9000 | all")
    args = ap.parse_args()

    detail, common_w = build_detail(load(args.mine), load(args.imed), args.noise_budget)
    comp = detail[detail.status == "compared"]

    by_family = detail.groupby("family").apply(agg, include_groups=False).reset_index()
    by_bench = detail.groupby(["family", "bench"]).apply(agg, include_groups=False).reset_index()
    by_bench = by_bench[by_bench["Compared points"] > 0]
    by_weight = detail.groupby("w_ops").apply(agg, include_groups=False).reset_index()
    by_family = by_family[by_family["Compared points"] > 0]
    overall = agg(detail)

    # ---- coverage: which benchmarks have results on one side only
    cov = (detail.assign(**{
        "Both": detail.status == "compared",
        "Only Mine": detail.status == "Imed missing",
        "Only Imed": detail.status == "Mine missing",
        "Neither": detail.status == "both missing"})
        .groupby(["family", "bench"])[["Both", "Only Mine", "Only Imed", "Neither"]].sum().reset_index())
    cov = cov[(cov["Only Mine"] > 0) | (cov["Only Imed"] > 0) | (cov["Neither"] > 0)]

    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        wb = xw.book
        ws = wb.create_sheet("Summary")
        nb = "all 3 budgets averaged" if args.noise_budget == "all" else f"noise_budget = {args.noise_budget}"
        title(ws, "Chehab agent vs Imed agent - summary",
              f"Mine = {nb}; w_ops in {common_w}; only points where BOTH agents produced a result; lower cost = better")

        # headline KPIs
        kpis = [
            ("Compared points (benchmark x w_ops)", int(overall["Compared points"]), "0"),
            ("Benchmarks compared", int(comp.bench.nunique()), "0"),
            ("Ops cost: geo-mean ratio Mine/Imed (<1 = mine better)", overall["Ops ratio Mine/Imed (geo-mean)"], "0.000"),
            ("Ops cost: average improvement vs Imed", overall["Ops improvement vs Imed"], "0.0%"),
            ("Ops cost: Win / Tie / Loss", f'{int(overall["Ops W"])} / {int(overall["Ops T"])} / {int(overall["Ops L"])}', "@"),
            ("Keys cost: Win / Tie / Loss", f'{int(overall["Keys W"])} / {int(overall["Keys T"])} / {int(overall["Keys L"])}', "@"),
            ("Pareto (ops+keys): Mine better / trade-off / Imed better",
             f'{int(overall["Pareto: Mine better"])} / {int(overall["Pareto: trade-off"])} / {int(overall["Pareto: Imed better"])}', "@"),
            ("Avg keys - Mine / Imed", f'{overall["Keys - Mine (avg)"]:.2f} / {overall["Keys - Imed (avg)"]:.2f}', "@"),
            ("Points where Imed has no result but Mine does", int((detail.status == "Imed missing").sum()), "0"),
            ("Points where Mine has no result but Imed does", int((detail.status == "Mine missing").sum()), "0"),
        ]
        for k, (lab, val, fmt) in enumerate(kpis, start=4):
            a, b = ws.cell(row=k, column=1, value=lab), ws.cell(row=k, column=2, value=val)
            a.font = Font(name=FONT, bold=True, size=10)
            b.font = Font(name=FONT, size=10)
            b.number_format = fmt
            b.alignment = Alignment(horizontal="right")
            a.fill = SUB_FILL
            a.border = b.border = BORDER
        ws.column_dimensions["A"].width = 52
        ws.column_dimensions["B"].width = 18

        r0 = len(kpis) + 6
        ws.cell(row=r0 - 1, column=1, value="By benchmark family (averaged over sizes and w_ops)").font = \
            Font(name=FONT, bold=True, size=11, color="1F3864")
        fam_tbl = by_family.rename(columns={"family": "Family"})
        write_table(ws, fam_tbl, r0=r0, widths=False)
        ws.column_dimensions["A"].width = 52
        for j in range(2, len(fam_tbl.columns) + 1):
            ws.column_dimensions[get_column_letter(j)].width = 14

        # chart: ops improvement per family
        imp_col = list(fam_tbl.columns).index("Ops improvement vs Imed") + 1
        ch = BarChart()
        ch.type = "col"
        ch.title = "Ops-cost improvement vs Imed (positive = mine better)"
        ch.y_axis.number_format = "0%"
        ch.add_data(Reference(ws, min_col=imp_col, min_row=r0, max_row=r0 + len(fam_tbl)), titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=r0 + len(fam_tbl)))
        ch.legend = None
        ch.height, ch.width = 8, 18
        ws.add_chart(ch, f"A{r0 + len(fam_tbl) + 3}")

        ws2 = wb.create_sheet("By_Benchmark")
        title(ws2, "Per benchmark (averaged over w_ops)")
        write_table(ws2, by_bench.rename(columns={"family": "Family", "bench": "Benchmark"}), r0=3)
        ws2.freeze_panes = "C4"

        ws3 = wb.create_sheet("By_Weight")
        title(ws3, "Per w_ops (averaged over all benchmarks) - shows behaviour across the ops/keys trade-off")
        write_table(ws3, by_weight.rename(columns={"w_ops": "w_ops"}), r0=3)

        ws4 = wb.create_sheet("Detail")
        dcols = ["family", "bench", "w_ops", "status", "ops_m", "ops_i", "ops_ratio", "ops_result",
                 "keys_m", "keys_i", "keys_result", "dominates", "mdepth_m", "mdepth_i",
                 "compile_m", "compile_i", "exec_m", "exec_i", "noise_m", "noise_i"]
        det = detail[dcols].rename(columns={
            "ops_m": "ops_mine", "ops_i": "ops_imed", "keys_m": "keys_mine", "keys_i": "keys_imed",
            "mdepth_m": "mdepth_mine", "mdepth_i": "mdepth_imed", "compile_m": "compile_s_mine",
            "compile_i": "compile_s_imed", "exec_m": "exec_s_mine", "exec_i": "exec_s_imed",
            "noise_m": "noise_left_mine", "noise_i": "noise_left_imed"})
        title(ws4, "All (benchmark, w_ops) points - raw comparison")
        write_table(ws4, det, r0=3)
        ws4.freeze_panes = "D4"
        ws4.auto_filter.ref = f"A3:{get_column_letter(len(det.columns))}{3 + len(det)}"

        ws5 = wb.create_sheet("Coverage")
        title(ws5, "Benchmarks where at least one agent has no result (excluded from averages)")
        write_table(ws5, cov.rename(columns={"family": "Family", "bench": "Benchmark"}), r0=3)

        ws6 = wb.create_sheet("Notes")
        notes = [
            "Method / assumptions",
            f"- Reference for Mine: {nb}. Imed's file has no noise_budget column; its Remaining_noise_budget matches the 1000 setting.",
            f"- w_ops values compared: {common_w} (Imed's finer sweep is reduced to these).",
            "- Tree names aligned: tree_<d>_<k>_<a>-<b> (Mine) = tree_<a>-<b>_<d>_<k> (Imed). Mine-only trees (k=2) have no counterpart.",
            "- Only points where both agents have a valid final_ops_cost are averaged; all others are listed in 'Coverage'.",
            f"- Ops W/T/L: Win if Mine/Imed < {1 - OPS_TOL:.2f}, Loss if > {1 + OPS_TOL:.2f}, else Tie. Keys W/T/L: exact comparison of final_keys_cost.",
            "- Geo-mean ratio is used for ops cost because costs span 3+ orders of magnitude (plain averages are dominated by big circuits).",
            "- Improvement = 1 - geo-mean ratio (e.g. 20% = Mine's ops cost is 20% lower than Imed's on average).",
            "- 'Dominates' (Detail): Yes = no worse on both ops and keys and strictly better on one; Dominated = opposite; Trade-off = better on one, worse on the other.",
            "- Timing columns depend on the machine each agent was run on; use with care.",
        ]
        for k, t in enumerate(notes, start=1):
            c = ws6.cell(row=k, column=1, value=t)
            c.font = Font(name=FONT, size=10, bold=(k == 1))
        ws6.column_dimensions["A"].width = 150

        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]
        wb.move_sheet("Summary", offset=-wb.index(wb["Summary"]))

    print(f"Wrote {args.out}")
    print(overall.to_string())


if __name__ == "__main__":
    main()