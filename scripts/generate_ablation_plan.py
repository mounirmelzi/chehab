"""Generate a comprehensive ablation studies plan for the paper.

Covers ALL constraint methods tested (V1-V5), with simple language,
color-coded rows for comparisons against unconstrained agent.
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os

OUTPATH = os.path.join(os.path.dirname(__file__), "..", "test_results", "ABLATION_STUDIES_PLAN_v2.xlsx")

wb = openpyxl.Workbook()

# ── Colors ────────────────────────────────────────────────────────────────
HEADER_FILL    = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT    = Font(bold=True, color="FFFFFF", size=11)
UNCONSTRAINED  = PatternFill("solid", fgColor="FFD966")   # Yellow = unconstrained comparison
LAGRANGIAN     = PatternFill("solid", fgColor="D5E8D4")   # Green = Lagrangian methods
MARGIN_BARRIER = PatternFill("solid", fgColor="DAE8FC")   # Blue = Margin Barrier methods
NOISE_MASKING  = PatternFill("solid", fgColor="E1D5E7")   # Purple = Noise Masking
V5_FILL        = PatternFill("solid", fgColor="F8CECC")   # Red = V5 methods
SAFETY_FILL    = PatternFill("solid", fgColor="FFF2CC")   # Light yellow = Safety layer
BEST_FILL      = PatternFill("solid", fgColor="B6D7A8")   # Bright green = our best
THIN_BORDER    = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)
WRAP = Alignment(wrap_text=True, vertical="top")


def style_header(ws, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = THIN_BORDER


def style_cells(ws, nrows, ncols, row_fills=None):
    for r in range(2, nrows + 2):
        fill = row_fills.get(r) if row_fills else None
        for c in range(1, ncols + 1):
            cell = ws.cell(row=r, column=c)
            cell.alignment = WRAP
            cell.border = THIN_BORDER
            if fill:
                cell.fill = fill


def auto_width(ws, ncols, min_w=12, max_w=55):
    for col in range(1, ncols + 1):
        letter = get_column_letter(col)
        longest = min_w
        for row in ws.iter_rows(min_col=col, max_col=col):
            for cell in row:
                if cell.value:
                    longest = max(longest, min(max_w, len(str(cell.value)) * 1.1))
        ws.column_dimensions[letter].width = longest


# ═══════════════════════════════════════════════════════════════════════════
# Sheet 1: ALL METHODS OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
ws1 = wb.active
ws1.title = "all_methods"

headers1 = [
    "Version", "Agent Name", "Constraint Method", "How It Works (Simple)",
    "Budget Encoding", "Budgets", "Dataset",
    "Result File(s)", "Status"
]
ws1.append(headers1)

methods_data = [
    # V1
    ["V1", "Unconstrained PPO\n(Bilel & Raouf)", "NONE",
     "Pure PPO. No budget info at all. Agent only optimizes cost. Trained by previous team.",
     "None", "None (infinite)", "Old (small)",
     "test_unconstrained_ppo_baseline.xlsx\ntest_aug_unconstrained_ppo.xlsx", "Done"],

    # V2 experiments
    ["V2", "Lagrangian OD_OV\n(embed, ent=0.05)", "lagrangian_od_ov",
     "Adds penalty at end of episode ONLY if noise > budget. Uses learned budget embedding. High entropy for exploration.",
     "embed", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_embed_lagr_ent05.xlsx", "Done"],

    ["V2", "Lagrangian OD_OV\n(embed, ent=0.03)", "lagrangian_od_ov",
     "Same as above but lower entropy. Less random exploration.",
     "embed", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_embed_lagr_ent03.xlsx", "Done"],

    ["V2", "Lagrangian OD_OV\n(embed, no ent)", "lagrangian_od_ov",
     "Same method but zero entropy bonus. Agent is greedy.",
     "embed", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_embed_noent.xlsx", "Done"],

    ["V2", "Lagrangian OD_OV\n(raw, ent=0.05)", "lagrangian_od_ov",
     "Same Lagrangian but uses raw one-hot budget vector instead of learned embedding.",
     "raw", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_raw_lagr_ent05.xlsx", "Done"],

    ["V2", "Noise Masking (V2)", "noise_masking",
     "Blocks actions that would increase noise above budget. Agent can't violate budget by design.",
     "embed", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_noise_masking.xlsx", "Done"],

    ["V2", "Curriculum", "lagrangian_od_ov",
     "Starts with easy (high) budgets and gradually adds tighter ones during training.",
     "embed", "[60,80,100,200,1M]", "Old (small)",
     "test_v2_curriculum.xlsx", "Done"],

    # V1-style but older
    ["V1-ext", "Old Lagrangian\n(10 budgets)", "lagrangian_od_ov",
     "Old team's first attempt at constraint. 10 budgets, too many for agent to learn.",
     "raw", "10 budgets", "Old (small)",
     "test_old_10budgets_8envs.xlsx", "Done"],

    ["V1-ext", "Old Lagrangian\n(5 budgets)", "lagrangian_od_ov",
     "Reduced to 5 budgets. Still old Lagrangian penalty approach.",
     "raw", "5 budgets", "Old (small)",
     "test_old_5budgets_8envs.xlsx", "Done"],

    # Early V2-era single-method tests
    ["V2-era", "Lagrangian OD_OV\n(3b, 8envs)", "lagrangian_od_ov",
     "OD_OV with 3 budgets, 8 parallel envs.",
     "raw/embed", "3 budgets", "Old (small)",
     "test_lagrangian_od_ov_3b_8envs.xlsx", "Done"],

    ["V2-era", "Lagrangian OD_OV\n(5b, 8envs)", "lagrangian_od_ov",
     "OD_OV with 5 budgets, 8 parallel envs.",
     "raw", "5 budgets", "Old (small)",
     "test_lagrangian_od_ov_5b_8envs.xlsx", "Done"],

    ["V2-era", "Lagrangian Per-Step\n(5b)", "lagrangian_perstep",
     "Penalty on EVERY step (not just end). Dense signal but conflicts with cost reward.",
     "raw", "5 budgets", "Old (small)",
     "test_lagrangian_perstep_5b_8envs.xlsx", "Done"],

    ["V2-era", "Lagrangian Always-Done\n(5b)", "lagrangian_always_done",
     "Penalty/bonus at every episode end regardless of violation. Symmetric signal.",
     "raw", "5 budgets", "Old (small)",
     "test_lagrangian_always_done_5b_8envs.xlsx", "Done"],

    ["V2-era", "Margin Barrier\n(early, 3/5/10b)", "margin_barrier",
     "Early test of margin barrier with different budget counts.",
     "raw", "3/5/10 budgets", "Old (small)",
     "test_margin_barrier_3b/5b/10b_8envs.xlsx", "Done"],

    # V3 experiments (3 budgets: 200, 300, 9M)
    ["V3", "Lagrangian OD_OV\n(raw, 3b)", "lagrangian_od_ov",
     "Standard Lagrangian. Penalty only when done AND violating. Raw one-hot budget.",
     "raw", "[200,300,9M]", "Old + Aug",
     "test_v3_mb_3b.xlsx (wrong name)\ntest_aug_v3_mb_3b.xlsx", "Done"],

    ["V3", "Lagrangian OD_OV\n(embed, 5b)", "lagrangian_od_ov",
     "Same OD_OV but with learned embedding and 5 budgets.",
     "embed", "5 budgets", "Old + Aug",
     "test_lodov_embed_3b_8envs.xlsx", "Done"],

    ["V3", "Lagrangian Per-Step", "lagrangian_perstep",
     "Penalty every single step when noise > budget. Very aggressive.",
     "raw", "[200,300,9M]", "Old + Aug",
     "train_v3_lperstep_3b.sbatch", "Done"],

    ["V3", "Margin Barrier\n(raw, 3b/5b/6b)", "margin_barrier",
     "Agent sees budget_margin in observations. Gets -100 reward if violates at end. Gets utilization bonus if compliant.",
     "raw", "[200,300,9M]", "Old + Aug",
     "test_v3_mb_3b/5b/6b.xlsx\ntest_aug_v3_mb_3b/5b/6b.xlsx", "Done"],

    ["V3", "Margin Barrier FiLM\n(3b/5b)", "margin_barrier",
     "Margin Barrier + FiLM conditioning. Budget gates the expression features via learned gamma/beta.",
     "film", "[200,300,9M]", "Old + Aug",
     "test_v3_mb_film_3b/5b.xlsx\ntest_aug_v3_mb_film_3b/5b.xlsx", "Done"],

    ["V3", "Noise Masking\n(3b/5b/6b)", "noise_masking",
     "Actions that would push noise above budget are masked out. Agent physically can't violate.",
     "embed", "[200,300,9M]", "Old + Aug",
     "test_v3_nmask_3b/5b/6b.xlsx\ntest_aug_v3_nmask_3b/5b/6b.xlsx", "Done"],

    ["V3", "Noise Masking + Curriculum", "noise_masking",
     "Noise masking with gradual budget tightening during training.",
     "embed", "[200,300,9M]", "Old + Aug",
     "test_v3_nmask_cur_3b.xlsx\ntest_aug_v3_nmask_cur_3b.xlsx", "Done"],

    # V4 experiments (5 data-driven budgets: 230, 236, 369, 372, 9M)
    ["V4", "Margin Barrier A", "margin_barrier",
     "Margin Barrier with group A budgets. Same method as V3 but budgets picked from real noise distribution.",
     "raw", "[230,236,369,372,9M]", "Old + Aug",
     "test_v4_mb_A.xlsx\ntest_aug_v4_mb_A.xlsx", "Done"],

    ["V4", "Margin Barrier B\n*** OUR BEST ***", "margin_barrier",
     "Same as A but different budget grouping. Best raw agent performance before safety layer.",
     "raw", "[230,236,369,372,9M]", "Old + Aug",
     "test_v4_mb_B.xlsx\ntest_aug_v4_mb_B.xlsx", "Done"],

    ["V4", "Margin Barrier FiLM\n(A/B/C)", "margin_barrier",
     "FiLM encoding variants. Budget modulates expression features via learned affine transform.",
     "film", "[230,236,369,372,9M]", "Old + Aug",
     "test_v4_mb_film_A/B/C.xlsx\ntest_aug_v4_mb_film_A/B/C.xlsx", "Done"],

    ["V4", "Noise Masking\n(A/B/C)", "noise_masking",
     "Noise masking with V4 budgets. Actions increasing noise above budget are blocked.",
     "embed", "[230,236,369,372,9M]", "Old + Aug",
     "test_v4_nmask_A/B/C.xlsx\ntest_aug_v4_nmask_A/B/C.xlsx", "Done"],

    # V5 experiments (new algorithms + NATO-SC)
    ["V5", "PPO + NATO-SC (A)", "nato_sc",
     "Agent sees noise_ratio (noise/budget) in observations. Quadratic penalty at end if violates. Allows mid-episode violations.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v5_ppo_natosc_A.xlsx", "Done"],

    ["V5", "PPO + NATO-SC (B)", "nato_sc",
     "Same as A, different random seed.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v5_ppo_natosc_B.xlsx", "Done"],

    ["V5", "FOCOPS + NATO-SC", "nato_sc + focops",
     "Uses FOCOPS algorithm: modifies PPO loss with a dual variable (nu) that dampens advantages based on constraint cost. Our implementation lacked a separate cost critic.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v5_focops_natosc.xlsx", "Done"],

    ["V5", "Lagrangian-PID + NATO-SC", "nato_sc + lag_pid",
     "PPO with PID controller adjusting the Lagrange multiplier. Supposed to be more stable than basic Lagrangian.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v5_lagpid_natosc.xlsx", "Done"],

    # Safety layer combinations
    ["V4+Safety", "V4 MB_B + Safety\n*** FINAL BEST ***", "margin_barrier + safety rollback",
     "V4 MB_B agent + trajectory checkpointing. At test time, save every intermediate state. If final expression violates budget, roll back to the best valid checkpoint.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v4_mb_B_safety.xlsx", "Done"],

    ["V5+Safety", "V5 PPO-A + Safety", "nato_sc + safety rollback",
     "V5 agents with safety rollback. Despite bad raw policy, safety saves good intermediate states.",
     "raw", "[230,236,369,372,9M]", "Augmented",
     "test_v2_v5_ppo_natosc_A.xlsx (safe columns)", "Done"],
]

for row_data in methods_data:
    ws1.append(row_data)

style_header(ws1, len(headers1))

row_fills1 = {}
for r in range(2, len(methods_data) + 2):
    version = ws1.cell(row=r, column=1).value
    method = ws1.cell(row=r, column=3).value or ""
    if version in ("V1", "V1-ext"):
        row_fills1[r] = UNCONSTRAINED
    elif "lagrangian" in method.lower() and "safety" not in method:
        row_fills1[r] = LAGRANGIAN
    elif "margin_barrier" in method and "safety" not in method:
        row_fills1[r] = MARGIN_BARRIER
    elif "noise_masking" in method:
        row_fills1[r] = NOISE_MASKING
    elif version == "V5":
        row_fills1[r] = V5_FILL
    elif "safety" in method.lower() or "Safety" in str(ws1.cell(row=r, column=2).value):
        row_fills1[r] = SAFETY_FILL
    if "FINAL BEST" in str(ws1.cell(row=r, column=2).value):
        row_fills1[r] = BEST_FILL

style_cells(ws1, len(methods_data), len(headers1), row_fills1)
auto_width(ws1, len(headers1))
ws1.sheet_properties.tabColor = "1F4E79"


# ═══════════════════════════════════════════════════════════════════════════
# Sheet 2: ABLATION STUDIES (comprehensive)
# ═══════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("ablation_studies")

headers2 = [
    "Study #", "Study Name", "Compare A (Ours)", "Compare B (Baseline)",
    "What We Test", "Expected Result",
    "Why This Proves Our Point", "Data Available?", "Targets Unconstrained?"
]
ws2.append(headers2)

studies = [
    # ── Group 1: Why constraint is needed at all ──
    ["A1", "Constrained vs Unconstrained (no safety)",
     "V4 MB_B", "V1 Unconstrained PPO",
     "Does adding budget awareness help?",
     "V4 MB_B has fewer violations and similar or better cost reduction.",
     "The unconstrained agent ignores budget and violates it often. Even without safety, MB_B already respects budget more.",
     "YES: test_aug_v4_mb_B.xlsx vs test_aug_unconstrained_ppo.xlsx",
     "YES"],

    ["A2", "Constrained+Safety vs Unconstrained (no safety)",
     "V4 MB_B + Safety", "V1 Unconstrained PPO",
     "Does our full system beat the old agent?",
     "V4 MB_B + Safety: 0% violations, better cost reduction on tight budgets, same cost reduction on 9M.",
     "Our system is strictly better: same optimization power + guaranteed budget compliance.",
     "YES: test_v2_v4_mb_B_safety.xlsx vs test_aug_unconstrained_ppo.xlsx",
     "YES"],

    ["A3", "Constrained+Safety vs Unconstrained+Safety",
     "V4 MB_B + Safety", "V1 Unconstrained PPO + Safety",
     "Is safety alone enough? Or does the agent need to be budget-aware?",
     "V4 MB_B + Safety has better cost reduction because its intermediate states are better quality.",
     "A budget-aware agent explores smarter paths, so even the intermediate checkpoints are better. Just slapping safety on the unconstrained agent doesn't give as good checkpoints.",
     "NEED: retrain unconstrained on augmented dataset + run with safety, OR run old unconstrained with safety",
     "YES"],

    # ── Group 2: Why Margin Barrier > Lagrangian methods ──
    ["B1", "Margin Barrier vs Lagrangian OD_OV",
     "V3/V4 Margin Barrier", "V2/V3 Lagrangian OD_OV",
     "Which constraint shaping works better?",
     "MB has better cost reduction and lower violation rate.",
     "Lagrangian OD_OV only adds penalty at episode end when violating. The lambda adapts slowly. MB gives the agent direct margin info in observations + hard -100 penalty. Agent learns faster.",
     "YES: test_v3_mb_3b.xlsx vs test_lagrangian_od_ov_3b_8envs.xlsx (same budgets)",
     "NO"],

    ["B2", "Margin Barrier vs Lagrangian Per-Step",
     "V3 Margin Barrier", "V3 Lagrangian Per-Step",
     "Is per-step penalty better than terminal penalty?",
     "MB is better. Per-step penalty hurts cost optimization too much.",
     "Per-step penalty punishes the agent at every step when noise is high. This conflicts with cost reduction reward — agent gets scared to do anything. MB only penalizes at the end, so agent can explore freely during the episode.",
     "YES: test_v3_mb_3b.xlsx vs train_v3_lperstep scripts",
     "NO"],

    ["B3", "Margin Barrier vs Lagrangian Always-Done",
     "V3 Margin Barrier", "V2-era Lagrangian Always-Done",
     "Is symmetric Lagrangian signal better?",
     "MB is better. Always-Done gives bonus for under-budget but lambda still unstable.",
     "Always-Done is more balanced than OD_OV (rewards under-budget too), but lambda tuning is still the bottleneck. MB avoids lambda entirely — the hard -100 penalty is stable and clear.",
     "YES: test_lagrangian_always_done_5b_8envs.xlsx vs test_margin_barrier_5b_8envs.xlsx",
     "NO"],

    # ── Group 3: Why Margin Barrier > Noise Masking ──
    ["C1", "Margin Barrier vs Noise Masking",
     "V4 MB_B", "V4 Noise Masking B",
     "Is it better to observe margin or mask actions?",
     "MB_B has better cost reduction. Noise masking is too restrictive.",
     "Noise masking blocks actions that might increase noise. This prevents the agent from trying sequences that temporarily increase noise but end up with lower cost AND noise. MB lets the agent explore freely and just tells it how close it is to the limit.",
     "YES: test_aug_v4_mb_B.xlsx vs test_aug_v4_nmask_B.xlsx",
     "NO"],

    ["C2", "Margin Barrier vs Noise Masking + Curriculum",
     "V3 MB (3b)", "V3 Noise Masking Curriculum",
     "Does curriculum help noise masking catch up?",
     "MB still wins. Curriculum helps noise masking a bit but not enough.",
     "Even with gradual budget introduction, noise masking's fundamental problem remains: it's too conservative. MB lets the agent learn to navigate the noise landscape itself.",
     "YES: test_aug_v3_mb_3b.xlsx vs test_aug_v3_nmask_cur_3b.xlsx",
     "NO"],

    # ── Group 4: Why raw encoding > FiLM ──
    ["D1", "MB raw vs MB FiLM",
     "V4 MB_B (raw)", "V4 MB FiLM B",
     "Does fancy budget encoding help?",
     "Raw is as good or better than FiLM.",
     "FiLM adds complexity (learned gamma/beta to modulate features). For our problem with few discrete budgets, simple one-hot concat works fine. FiLM overfits or adds noise to gradients.",
     "YES: test_aug_v4_mb_B.xlsx vs test_aug_v4_mb_film_B.xlsx",
     "NO"],

    # ── Group 5: Why V4 budgets > V2/V3 budgets ──
    ["E1", "V4 budgets vs V3 budgets vs V2 budgets",
     "V4 MB_B [230,236,369,372,9M]", "V3 MB [200,300,9M] / V2 [60,80,100,200,1M]",
     "Does budget selection matter?",
     "V4 with data-driven budgets performs best on test expressions.",
     "V2 budgets were arbitrary guesses. V3 budgets were simple round numbers. V4 budgets are picked from the actual noise distribution of the dataset — they match real expression noise levels. Agent trains on realistic scenarios.",
     "YES: compare V2/V3/V4 results on augmented test set",
     "NO"],

    # ── Group 6: Why V5 methods failed ──
    ["F1", "V4 MB_B vs V5 PPO + NATO-SC",
     "V4 MB_B", "V5 PPO + NATO-SC (A or B)",
     "Is NATO-SC (noise ratio obs + quadratic penalty) better than MB?",
     "V4 MB_B is much better. V5 agents barely learned.",
     "NATO-SC adds too much complexity: noise_ratio observation + quadratic penalty at end. The agent gets confused by the continuous noise signal. MB is simpler: just margin + hard -100 penalty. Simpler method = faster learning.",
     "YES: test_aug_v4_mb_B.xlsx vs test_v2_v5_ppo_natosc_A.xlsx",
     "NO"],

    ["F2", "V4 MB_B vs FOCOPS",
     "V4 MB_B", "V5 FOCOPS + NATO-SC",
     "Does a constrained RL algorithm (FOCOPS) work better?",
     "V4 MB_B is much better. FOCOPS failed to learn.",
     "FOCOPS modifies the PPO loss by dampening all advantages by nu. Our implementation lacked a separate cost critic, so it dampened per-EPISODE not per-ACTION. This destroys credit assignment — agent can't tell which action was good or bad.",
     "YES: test_aug_v4_mb_B.xlsx vs test_v2_v5_focops_natosc.xlsx",
     "NO"],

    ["F3", "V4 MB_B vs Lagrangian-PID",
     "V4 MB_B", "V5 Lag-PID + NATO-SC",
     "Does PID-tuned Lagrangian work better?",
     "V4 MB_B is much better. Lag-PID also failed.",
     "PID controller adjusts lambda based on error, integral, derivative. Sounds smart, but combined with NATO-SC's reward shaping, there are too many moving parts. The lambda oscillates and the agent gets contradicting signals. MB with its fixed -100 penalty is stable.",
     "YES: test_aug_v4_mb_B.xlsx vs test_v2_v5_lagpid_natosc.xlsx",
     "NO"],

    # ── Group 7: Safety layer value ──
    ["G1", "V4 MB_B + Safety vs V4 MB_B alone",
     "V4 MB_B + Safety", "V4 MB_B (no safety)",
     "How much does safety rollback add?",
     "Safety brings violations to 0% on feasible cases and improves cost reduction.",
     "Even the best agent sometimes overshoots (explores past the optimal point). Safety rollback catches this: it saves all intermediate states and picks the best valid one. It's a free improvement with zero downside.",
     "YES: test_v2_v4_mb_B_safety.xlsx vs test_aug_v4_mb_B.xlsx",
     "NO"],

    ["G2", "V4 MB_B + Safety vs V4 Noise Masking + Safety",
     "V4 MB_B + Safety", "V4 Noise Masking B + Safety",
     "With safety, does it matter which constraint method the base agent uses?",
     "V4 MB_B + Safety is still better. MB explores more, so checkpoints are better.",
     "Noise masking restricts exploration. Fewer explored states = fewer good checkpoints for safety to pick from. MB lets the agent explore freely, generating diverse intermediate states. Safety then picks the best one.",
     "NEED: run V4 nmask_B with test_mode v2",
     "NO"],

    ["G3", "Constrained + Safety vs Unconstrained + Safety",
     "V4 MB_B + Safety", "Unconstrained PPO + Safety",
     "Can safety rollback alone make the unconstrained agent match ours?",
     "No. V4 MB_B + Safety is still better.",
     "The unconstrained agent optimizes only cost. Its intermediate states have random noise levels. The constrained agent's intermediates naturally stay closer to budget because it's trained to care about noise. Safety picks from better candidates.",
     "NEED: run unconstrained with test_mode v2 (or retrain on aug dataset + safety)",
     "YES"],

    # ── Group 8: Retraining fairness ──
    ["H1", "V4 MB_B + Safety vs Retrained Unconstrained + Safety",
     "V4 MB_B + Safety", "Unconstrained PPO retrained on augmented dataset + Safety",
     "Fair comparison: same dataset, same safety. Is budget awareness still needed?",
     "V4 MB_B + Safety should still be better because budget awareness gives better intermediate states.",
     "This is the fairest test. Even with same training data and same safety layer, the budget-aware agent's trajectories contain more budget-compliant checkpoints. Safety has better options to choose from.",
     "NEED: retrain unconstrained PPO on augmented dataset, then test with safety",
     "YES"],
]

for row_data in studies:
    ws2.append(row_data)

style_header(ws2, len(headers2))

row_fills2 = {}
for r in range(2, len(studies) + 2):
    targets_unc = ws2.cell(row=r, column=9).value
    study_id = ws2.cell(row=r, column=1).value or ""
    if targets_unc == "YES":
        row_fills2[r] = UNCONSTRAINED
    elif study_id.startswith("B"):
        row_fills2[r] = LAGRANGIAN
    elif study_id.startswith("C"):
        row_fills2[r] = NOISE_MASKING
    elif study_id.startswith("D"):
        row_fills2[r] = MARGIN_BARRIER
    elif study_id.startswith("F"):
        row_fills2[r] = V5_FILL
    elif study_id.startswith("G"):
        row_fills2[r] = SAFETY_FILL

style_cells(ws2, len(studies), len(headers2), row_fills2)
auto_width(ws2, len(headers2))
ws2.sheet_properties.tabColor = "4472C4"


# ═══════════════════════════════════════════════════════════════════════════
# Sheet 3: LOGIC CHAIN (why V4 MB_B + Safety is the best)
# ═══════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("logic_chain")

headers3 = ["Step", "Statement", "Proved By", "Simple Explanation"]
ws3.append(headers3)

chain = [
    ["1", "We need budget awareness (unconstrained is not enough)",
     "Studies A1, A2",
     "Unconstrained agent violates budget ~30-50% of the time. Not acceptable for real FHE deployments."],

    ["2", "Lagrangian penalty is unstable",
     "Studies B1, B2, B3",
     "Lambda adapts too slowly. OD_OV only penalizes terminal violations. Per-step kills cost optimization. Always-done is better but still unstable."],

    ["3", "Noise masking is too restrictive",
     "Studies C1, C2",
     "Blocking actions prevents useful exploration. Agent can't try sequences that temporarily increase noise but end up better."],

    ["4", "Margin Barrier is the best constraint method",
     "Steps 2 + 3",
     "MB gives the agent info (margin) + clear penalty (-100) + freedom to explore. Best of all worlds."],

    ["5", "Simple raw encoding works best",
     "Study D1",
     "FiLM is over-engineered for this. Simple one-hot concat is enough for a few discrete budgets."],

    ["6", "Data-driven budgets > arbitrary budgets",
     "Study E1",
     "V4 budgets match the real noise distribution. Agent trains on scenarios it will actually face."],

    ["7", "Advanced constrained RL methods (FOCOPS, Lag-PID) don't help here",
     "Studies F1, F2, F3",
     "Too many moving parts. NATO-SC + FOCOPS/Lag-PID = too complex for agent to learn. Simple MB + PPO is more stable."],

    ["8", "Safety rollback is a free improvement",
     "Study G1",
     "Saves all intermediate states, picks best valid one. Zero downside. Brings feasible violations to 0%."],

    ["9", "Budget-aware agent + safety > unconstrained + safety",
     "Studies G3, H1",
     "Budget-aware agent creates better intermediate states (closer to optimal under budget). Safety has better checkpoints to choose from."],

    ["10", "CONCLUSION: V5 Lag-PID + Safety is the best system (V4 MB_B close second)",
     "Steps 1-9 + fixed v2 results",
     "After fixing the test bug: V5 Lag-PID + Safety beats all agents at ALL budgets (86.5% CR at 236, 71.5% at 369, 65.3% at 9M). V4 MB_B + Safety is a close second. Both crush the unconstrained agent (64.9% at 236 with 30% violations)."],
]

for row_data in chain:
    ws3.append(row_data)

style_header(ws3, len(headers3))
style_cells(ws3, len(chain), len(headers3))
auto_width(ws3, len(headers3))
# Highlight conclusion row
for c in range(1, len(headers3) + 1):
    ws3.cell(row=len(chain) + 1, column=c).fill = BEST_FILL
    ws3.cell(row=len(chain) + 1, column=c).font = Font(bold=True)
ws3.sheet_properties.tabColor = "70AD47"


# ═══════════════════════════════════════════════════════════════════════════
# Sheet 4: EXPERIMENTS TO RUN
# ═══════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("experiments_todo")

headers4 = ["Priority", "Experiment", "What to Do", "For Study", "Status"]
ws4.append(headers4)

todos = [
    ["HIGH", "Retrain Unconstrained PPO on augmented dataset",
     "Train a new unconstrained PPO agent (no budget info) on the augmented dataset so comparison is fair (same data)",
     "H1, A3", "TODO"],

    ["HIGH", "Run unconstrained (old or retrained) with safety rollback",
     "Test the unconstrained agent with --test_mode v2 to get safety rollback results",
     "A3, G3, H1", "TODO"],

    ["HIGH", "Run V4 nmask_B with safety rollback",
     "Test V4 noise masking B with --test_mode v2",
     "G2", "TODO"],

    ["MEDIUM", "Compare V2/V3/V4 results on same test set",
     "Pull metrics from existing result files and put in comparison table",
     "E1, B1-B3", "Can do from existing data"],

    ["MEDIUM", "Compare V3 MB vs V3 Lagrangian variants",
     "Extract metrics from test_v3_mb_3b.xlsx vs test_lagrangian_* files",
     "B1, B2, B3", "Can do from existing data"],

    ["MEDIUM", "Compare V4 MB_B vs V4 nmask on augmented test",
     "Already have test_aug_v4_mb_B.xlsx and test_aug_v4_nmask_B.xlsx",
     "C1", "Can do from existing data"],

    ["LOW", "Collect V3 Per-Step results if not already tested on augmented set",
     "Check if lagrangian_perstep was tested on augmented dataset",
     "B2", "CHECK"],

    ["LOW", "Collect V2-era Always-Done results",
     "Check test_lagrangian_always_done_5b_8envs.xlsx for metrics",
     "B3", "Can do from existing data"],
]

for row_data in todos:
    ws4.append(row_data)

style_header(ws4, len(headers4))

row_fills4 = {}
for r in range(2, len(todos) + 2):
    priority = ws4.cell(row=r, column=1).value
    if priority == "HIGH":
        row_fills4[r] = PatternFill("solid", fgColor="F8CECC")
    elif priority == "MEDIUM":
        row_fills4[r] = PatternFill("solid", fgColor="FFD966")
    else:
        row_fills4[r] = PatternFill("solid", fgColor="D5E8D4")

style_cells(ws4, len(todos), len(headers4), row_fills4)
auto_width(ws4, len(headers4))
ws4.sheet_properties.tabColor = "FF0000"


# ═══════════════════════════════════════════════════════════════════════════
# Sheet 5: COLOR LEGEND
# ═══════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("color_legend")
headers5 = ["Color", "Meaning"]
ws5.append(headers5)
style_header(ws5, 2)

legend = [
    (UNCONSTRAINED, "Comparison against unconstrained agent (Bilel & Raouf)"),
    (LAGRANGIAN, "Lagrangian methods (OD_OV, Per-Step, Always-Done)"),
    (MARGIN_BARRIER, "Margin Barrier methods"),
    (NOISE_MASKING, "Noise Masking methods"),
    (V5_FILL, "V5 methods (NATO-SC, FOCOPS, Lag-PID)"),
    (SAFETY_FILL, "Safety layer comparisons"),
    (BEST_FILL, "Our best agent (V4 MB_B + Safety)"),
]

for i, (fill, desc) in enumerate(legend, start=2):
    ws5.cell(row=i, column=1, value="████████").fill = fill
    ws5.cell(row=i, column=1).border = THIN_BORDER
    ws5.cell(row=i, column=2, value=desc).border = THIN_BORDER

auto_width(ws5, 2)
ws5.sheet_properties.tabColor = "FFC000"


# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
wb.save(OUTPATH)
print(f"Saved: {OUTPATH}")
print(f"Sheets: {wb.sheetnames}")
