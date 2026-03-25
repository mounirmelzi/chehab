#!/usr/bin/env python3
"""
Generate post-augmentation training data noise analysis.

Combines:
  - Pre-augmentation stats from training_data_noise_analysis.xlsx (38,245 expr)
  - Known augmentation: 6,000 high-noise expressions at MultDepth 7-12

Output: generate_statistics_graphs/training_data_noise_analysis_POST_AUGMENTATION.xlsx
"""

import os
import numpy as np
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PRE_XLSX = os.path.join(BASE_DIR, "generate_statistics_graphs", "training_data_noise_analysis.xlsx")
OUTPUT   = os.path.join(BASE_DIR, "generate_statistics_graphs", "training_data_noise_analysis_POST_AUGMENTATION.xlsx")

HEADER_FONT = Font(bold=True, size=11)
SECTION_FONT = Font(bold=True, size=11, color="4472C4")
GOOD_FILL = PatternFill("solid", fgColor="C6EFCE")
WARN_FILL = PatternFill("solid", fgColor="FFEB9C")
BAD_FILL  = PatternFill("solid", fgColor="FFC7CE")
NEW_FILL  = PatternFill("solid", fgColor="BDD7EE")
THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)

# ── Original dataset MultDepth distribution (from pre-augmentation sheet) ──
ORIGINAL_MD = {
    0: 69, 1: 20069, 2: 14513, 3: 2363, 4: 590, 5: 280, 6: 137,
    7: 64, 8: 27, 9: 25, 10: 10, 11: 9, 12: 10,
    13: 3, 14: 4, 15: 4, 16: 3, 17: 3, 18: 3, 19: 1, 20: 3,
    22: 1, 24: 3, 25: 2, 26: 5, 27: 1, 28: 2,
    31: 2, 32: 3, 33: 8, 34: 4, 35: 1, 36: 3,
    41: 2, 42: 1, 49: 1, 51: 1, 52: 1, 67: 1,
    72: 1, 73: 1, 76: 1, 77: 1, 89: 1, 104: 1,
    105: 1, 110: 1, 126: 1, 127: 1, 173: 1, 251: 2,
}
ORIGINAL_TOTAL = sum(ORIGINAL_MD.values())  # 38,245

# ── Augmented expressions ──
# Actual NoiseEstimator values are higher than 33.383*MD due to other ops
# MD 7 actual ≈ 248, not 234; using calibrated estimates
AUGMENTED_MD = {7: 2000, 8: 1200, 9: 600, 10: 500, 11: 1000, 12: 700}
AUGMENTED_TOTAL = sum(AUGMENTED_MD.values())  # 6,000

# Approximate noise per MultDepth from NoiseEstimator (includes add/sub/neg ops)
# The linear model: noise ≈ 33.383*MD + 1.198*Depth + 1.028*negate + ...
# Generated expressions have extra ops, so actual noise > 33.383*MD
NOISE_PER_MD = {
    0: 0, 1: 33.4, 2: 66.8, 3: 100.1, 4: 133.5, 5: 166.9, 6: 200.3,
    7: 248, 8: 281, 9: 315, 10: 348, 11: 381, 12: 415,
}
# For MD > 12, use simple linear extrapolation
for md in list(ORIGINAL_MD.keys()):
    if md not in NOISE_PER_MD:
        NOISE_PER_MD[md] = md * 33.383

# Original noise distribution (from the pre-augmentation sheet)
ORIGINAL_NOISE_DIST = {
    "0-40": 19891, "41-60": 211, "61-80": 14506, "81-100": 9,
    "101-120": 2356, "121-150": 579, "151-200": 297,
    "201-250": 168, "251-300": 62, "301-500": 61, "500+": 72,
}

# Map augmented expressions to noise buckets
def noise_bucket(noise):
    if noise <= 40: return "0-40"
    if noise <= 60: return "41-60"
    if noise <= 80: return "61-80"
    if noise <= 100: return "81-100"
    if noise <= 120: return "101-120"
    if noise <= 150: return "121-150"
    if noise <= 200: return "151-200"
    if noise <= 250: return "201-250"
    if noise <= 300: return "251-300"
    if noise <= 500: return "301-500"
    return "500+"


def apply_style(ws, row, max_col, font=None, fill=None, align_center=True):
    for c in range(1, max_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.border = THIN_BORDER
        if font:
            cell.font = font
        if fill:
            cell.fill = fill
        if align_center:
            cell.alignment = Alignment(horizontal="center", wrap_text=True)


def main():
    wb = openpyxl.Workbook()

    # ═══════════════════════════════════════════════════════════
    # Sheet 1: Summary
    # ═══════════════════════════════════════════════════════════
    ws = wb.active
    ws.title = "Summary"
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 20

    r = 1
    for c, h in enumerate(["Statistic", "Pre-Augmentation", "Post-Augmentation", "Change"], 1):
        ws.cell(row=r, column=c, value=h)
    apply_style(ws, r, 4, font=HEADER_FONT)

    total_post = ORIGINAL_TOTAL + AUGMENTED_TOTAL

    # Compute post-augmentation stats
    # Build approximate full noise array for post-augmentation
    post_noises = []
    for md, cnt in ORIGINAL_MD.items():
        n = NOISE_PER_MD.get(md, md * 33.383)
        post_noises.extend([n] * cnt)
    for md, cnt in AUGMENTED_MD.items():
        n = NOISE_PER_MD[md]
        post_noises.extend([n] * cnt)
    post_noises = np.array(post_noises)

    pre_noises = []
    for md, cnt in ORIGINAL_MD.items():
        n = NOISE_PER_MD.get(md, md * 33.383)
        pre_noises.extend([n] * cnt)
    pre_noises = np.array(pre_noises)

    rows_data = [
        ("Total expressions", ORIGINAL_TOTAL, total_post, f"+{AUGMENTED_TOTAL}"),
        ("Mean noise", round(np.mean(pre_noises), 1), round(np.mean(post_noises), 1),
         f"+{round(np.mean(post_noises) - np.mean(pre_noises), 1)}"),
        ("Median noise", round(np.median(pre_noises), 1), round(np.median(post_noises), 1), ""),
        ("Std noise", round(np.std(pre_noises), 1), round(np.std(post_noises), 1),
         f"+{round(np.std(post_noises) - np.std(pre_noises), 1)}"),
        ("P90", round(np.percentile(pre_noises, 90), 1), round(np.percentile(post_noises, 90), 1), ""),
        ("P95", round(np.percentile(pre_noises, 95), 1), round(np.percentile(post_noises, 95), 1), ""),
        ("P99", round(np.percentile(pre_noises, 99), 1), round(np.percentile(post_noises, 99), 1), ""),
        ("Max noise", round(max(pre_noises)), round(max(post_noises)), ""),
    ]
    for i, (stat, pre, post, change) in enumerate(rows_data, 2):
        ws.cell(row=i, column=1, value=stat)
        ws.cell(row=i, column=2, value=pre)
        ws.cell(row=i, column=3, value=post)
        ws.cell(row=i, column=4, value=change)
        apply_style(ws, i, 4)
    r = len(rows_data) + 2

    # Budget analysis section
    r += 1
    ws.cell(row=r, column=1, value="V4 TRAINING BUDGETS — Constraint Activation Rate")
    ws.cell(row=r, column=1).font = SECTION_FONT
    r += 1
    for c, h in enumerate(["Budget", "Pre: Constrained %", "Post: Constrained %", "Improvement"], 1):
        ws.cell(row=r, column=c, value=h)
    apply_style(ws, r, 4, font=HEADER_FONT)
    r += 1

    for budget in [172, 230, 233, 236, 369, 372, 9000000]:
        pre_above = sum(1 for n in pre_noises if n > budget)
        post_above = sum(1 for n in post_noises if n > budget)
        pre_pct = round(pre_above / len(pre_noises) * 100, 2)
        post_pct = round(post_above / len(post_noises) * 100, 2)
        ws.cell(row=r, column=1, value=budget)
        ws.cell(row=r, column=2, value=f"{pre_pct}%")
        ws.cell(row=r, column=3, value=f"{post_pct}%")
        ws.cell(row=r, column=4, value=f"+{round(post_pct - pre_pct, 2)}pp")
        apply_style(ws, r, 4)
        if post_pct >= 10:
            ws.cell(row=r, column=3).fill = GOOD_FILL
        elif post_pct >= 1:
            ws.cell(row=r, column=3).fill = WARN_FILL
        else:
            ws.cell(row=r, column=3).fill = BAD_FILL
        r += 1

    # ═══════════════════════════════════════════════════════════
    # Sheet 2: Noise Distribution (before vs after)
    # ═══════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Noise Distribution")
    buckets_order = ["0-40", "41-60", "61-80", "81-100", "101-120",
                     "121-150", "151-200", "201-250", "251-300", "301-500", "500+"]

    # Compute augmented additions per bucket
    aug_bucket_counts = {b: 0 for b in buckets_order}
    for md, cnt in AUGMENTED_MD.items():
        n = NOISE_PER_MD[md]
        b = noise_bucket(n)
        aug_bucket_counts[b] += cnt

    headers = ["Noise Range", "Pre Count", "Pre %", "Added", "Post Count", "Post %", "Change"]
    for c, h in enumerate(headers, 1):
        ws2.cell(row=1, column=c, value=h)
    apply_style(ws2, 1, len(headers), font=HEADER_FONT)

    for i, bucket in enumerate(buckets_order, 2):
        pre_cnt = ORIGINAL_NOISE_DIST[bucket]
        aug_cnt = aug_bucket_counts[bucket]
        post_cnt = pre_cnt + aug_cnt
        pre_pct = round(pre_cnt / ORIGINAL_TOTAL * 100, 1)
        post_pct = round(post_cnt / total_post * 100, 1)
        ws2.cell(row=i, column=1, value=bucket)
        ws2.cell(row=i, column=2, value=pre_cnt)
        ws2.cell(row=i, column=3, value=f"{pre_pct}%")
        ws2.cell(row=i, column=4, value=f"+{aug_cnt}" if aug_cnt > 0 else "—")
        ws2.cell(row=i, column=5, value=post_cnt)
        ws2.cell(row=i, column=6, value=f"{post_pct}%")
        ws2.cell(row=i, column=7, value=f"+{round(post_pct - pre_pct, 1)}pp" if aug_cnt > 0 else "—")
        apply_style(ws2, i, len(headers))
        if aug_cnt > 0:
            ws2.cell(row=i, column=4).fill = NEW_FILL
            ws2.cell(row=i, column=5).fill = NEW_FILL

    # Totals
    r2 = len(buckets_order) + 2
    ws2.cell(row=r2, column=1, value="TOTAL")
    ws2.cell(row=r2, column=2, value=ORIGINAL_TOTAL)
    ws2.cell(row=r2, column=4, value=f"+{AUGMENTED_TOTAL}")
    ws2.cell(row=r2, column=5, value=total_post)
    apply_style(ws2, r2, len(headers), font=HEADER_FONT)

    for c in range(1, len(headers) + 1):
        ws2.column_dimensions[get_column_letter(c)].width = 16

    # ═══════════════════════════════════════════════════════════
    # Sheet 3: MultDepth Distribution (before vs after)
    # ═══════════════════════════════════════════════════════════
    ws3 = wb.create_sheet("MultDepth Distribution")
    md_headers = ["MultDepth", "Pre Count", "Pre %", "Added", "Post Count", "Post %",
                  "Approx Noise", "Note"]
    for c, h in enumerate(md_headers, 1):
        ws3.cell(row=1, column=c, value=h)
    apply_style(ws3, 1, len(md_headers), font=HEADER_FONT)

    all_mds = sorted(set(list(ORIGINAL_MD.keys()) + list(AUGMENTED_MD.keys())))
    for i, md in enumerate(all_mds, 2):
        pre_cnt = ORIGINAL_MD.get(md, 0)
        aug_cnt = AUGMENTED_MD.get(md, 0)
        post_cnt = pre_cnt + aug_cnt
        pre_pct = round(pre_cnt / ORIGINAL_TOTAL * 100, 2)
        post_pct = round(post_cnt / total_post * 100, 2)
        noise = NOISE_PER_MD.get(md, md * 33.383)

        note = ""
        if md == 7: note = "critical boundary for budget 230"
        elif md == 8: note = "above 230/236 threshold"
        elif md == 9: note = "bridge zone toward 369"
        elif md == 10: note = "approaching budget 369"
        elif md == 11: note = "critical boundary for budget 369"
        elif md == 12: note = "above budget 369"

        ws3.cell(row=i, column=1, value=md)
        ws3.cell(row=i, column=2, value=pre_cnt)
        ws3.cell(row=i, column=3, value=f"{pre_pct}%")
        ws3.cell(row=i, column=4, value=f"+{aug_cnt}" if aug_cnt > 0 else "—")
        ws3.cell(row=i, column=5, value=post_cnt)
        ws3.cell(row=i, column=6, value=f"{post_pct}%")
        ws3.cell(row=i, column=7, value=round(noise, 1))
        ws3.cell(row=i, column=8, value=note)
        apply_style(ws3, i, len(md_headers))
        if aug_cnt > 0:
            ws3.cell(row=i, column=4).fill = NEW_FILL
            ws3.cell(row=i, column=5).fill = NEW_FILL
            ws3.cell(row=i, column=8).fill = NEW_FILL
            ws3.cell(row=i, column=8).alignment = Alignment(horizontal="left")

    for c in range(1, len(md_headers) + 1):
        ws3.column_dimensions[get_column_letter(c)].width = 18
    ws3.column_dimensions["H"].width = 35

    # ═══════════════════════════════════════════════════════════
    # Sheet 4: Augmentation Details
    # ═══════════════════════════════════════════════════════════
    ws4 = wb.create_sheet("Augmentation Details")
    det_headers = ["MultDepth", "Count Added", "Est. Noise (bits)",
                   "Target Budget Boundary", "Purpose"]
    for c, h in enumerate(det_headers, 1):
        ws4.cell(row=1, column=c, value=h)
    apply_style(ws4, 1, len(det_headers), font=HEADER_FONT)

    details = [
        (7, 2000, 248, "230/233/236", "Right at the 3 tight 192-bit SEAL budgets"),
        (8, 1200, 281, "above 236", "Clearly above all 192-bit thresholds"),
        (9, 600, 315, "bridge", "Between 236 and 369 regions"),
        (10, 500, 348, "approaching 369", "High noise approaching 128-bit SEAL budget"),
        (11, 1000, 381, "369/372", "Right at the 2 loose 128-bit SEAL budgets"),
        (12, 700, 415, "above 372", "Clearly above all constrained budgets"),
    ]
    for i, (md, cnt, noise, boundary, purpose) in enumerate(details, 2):
        ws4.cell(row=i, column=1, value=md)
        ws4.cell(row=i, column=2, value=cnt)
        ws4.cell(row=i, column=3, value=noise)
        ws4.cell(row=i, column=4, value=boundary)
        ws4.cell(row=i, column=5, value=purpose)
        apply_style(ws4, i, len(det_headers))
        ws4.cell(row=i, column=5).alignment = Alignment(horizontal="left")

    r4 = len(details) + 3
    ws4.cell(row=r4, column=1, value="Total Added")
    ws4.cell(row=r4, column=2, value=AUGMENTED_TOTAL)
    apply_style(ws4, r4, 2, font=HEADER_FONT)

    r4 += 2
    ws4.cell(row=r4, column=1, value="Augmentation Strategy")
    ws4.cell(row=r4, column=1).font = SECTION_FONT
    r4 += 1
    strategy_notes = [
        "53% of augmented data at MD 7-8 (noise 248-281): targets the tight 230-236 budget region",
        "18% at MD 9-10 (noise 315-348): bridges the gap between budget clusters",
        "28% at MD 11-12 (noise 381-415): targets the 369-372 budget region",
        "Vec sizes: randomly chosen from [1, 4, 8, 9, 16, 25, 32] matching original distribution",
        "Deduplication: all generated expressions checked against existing dataset + each other",
        "Noise estimation: NoiseEstimator (LinearRegression on SEAL features) verified each expression",
    ]
    for note in strategy_notes:
        ws4.cell(row=r4, column=1, value=note)
        ws4.merge_cells(start_row=r4, start_column=1, end_row=r4, end_column=5)
        r4 += 1

    for c in range(1, len(det_headers) + 1):
        ws4.column_dimensions[get_column_letter(c)].width = 22
    ws4.column_dimensions["E"].width = 55

    # ═══════════════════════════════════════════════════════════
    # Sheet 5: Budget Threshold Analysis (pre vs post)
    # ═══════════════════════════════════════════════════════════
    ws5 = wb.create_sheet("Budget Threshold Analysis")
    bt_headers = ["Budget", "Pre: Constrained", "Pre: %", "Post: Constrained",
                  "Post: %", "Change (pp)", "Assessment"]
    for c, h in enumerate(bt_headers, 1):
        ws5.cell(row=1, column=c, value=h)
    apply_style(ws5, 1, len(bt_headers), font=HEADER_FONT)

    budgets_full = [40, 60, 80, 100, 120, 150, 172, 200, 230, 233, 236, 240,
                    300, 369, 372, 500, 700, 1000, 9000000]
    for i, budget in enumerate(budgets_full, 2):
        pre_above = sum(1 for n in pre_noises if n > budget)
        post_above = sum(1 for n in post_noises if n > budget)
        pre_pct = round(pre_above / len(pre_noises) * 100, 2)
        post_pct = round(post_above / len(post_noises) * 100, 2)
        change = round(post_pct - pre_pct, 2)

        if post_pct >= 10:
            assessment = "STRONG constraint signal"
            fill = GOOD_FILL
        elif post_pct >= 3:
            assessment = "GOOD constraint signal"
            fill = GOOD_FILL
        elif post_pct >= 1:
            assessment = "MODERATE signal"
            fill = WARN_FILL
        else:
            assessment = "WEAK signal"
            fill = BAD_FILL

        ws5.cell(row=i, column=1, value=budget)
        ws5.cell(row=i, column=2, value=pre_above)
        ws5.cell(row=i, column=3, value=f"{pre_pct}%")
        ws5.cell(row=i, column=4, value=post_above)
        ws5.cell(row=i, column=5, value=f"{post_pct}%")
        ws5.cell(row=i, column=6, value=f"+{change}pp" if change >= 0 else f"{change}pp")
        ws5.cell(row=i, column=7, value=assessment)
        apply_style(ws5, i, len(bt_headers))
        ws5.cell(row=i, column=7).fill = fill
        ws5.cell(row=i, column=7).alignment = Alignment(horizontal="left")

    for c in range(1, len(bt_headers) + 1):
        ws5.column_dimensions[get_column_letter(c)].width = 20
    ws5.column_dimensions["G"].width = 28

    wb.save(OUTPUT)
    print(f"Saved: {OUTPUT}")

    # ── Print summary to console ──
    print(f"\n{'='*70}")
    print(f"  POST-AUGMENTATION TRAINING DATA ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total expressions: {ORIGINAL_TOTAL:,} -> {total_post:,} (+{AUGMENTED_TOTAL:,})")
    print(f"  Mean noise:        {np.mean(pre_noises):.1f} -> {np.mean(post_noises):.1f}")
    print(f"  P95 noise:         {np.percentile(pre_noises, 95):.1f} -> {np.percentile(post_noises, 95):.1f}")
    print(f"  P99 noise:         {np.percentile(pre_noises, 99):.1f} -> {np.percentile(post_noises, 99):.1f}")

    print(f"\n  Budget constraint activation (% of dataset with noise > budget):")
    print(f"  {'Budget':>10s}  {'Pre':>8s}  {'Post':>8s}  {'Change':>8s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for budget in [172, 230, 233, 236, 369, 372, 9000000]:
        pre_above = sum(1 for n in pre_noises if n > budget)
        post_above = sum(1 for n in post_noises if n > budget)
        pre_pct = pre_above / len(pre_noises) * 100
        post_pct = post_above / len(post_noises) * 100
        print(f"  {budget:>10,d}  {pre_pct:>7.2f}%  {post_pct:>7.2f}%  {post_pct-pre_pct:>+7.2f}pp")


if __name__ == "__main__":
    main()
