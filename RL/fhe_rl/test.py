from stable_baselines3 import PPO
import sys, importlib, time
from .utils import load_expressions, load_expressions_named, load_embeddings, create_rules
from pytrs import parse_sexpr, NoiseEstimator
from .env import fheEnv
from .policy import HierarchicalMaskablePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd


def test_agent(
    expressions_file: str,
    embeddings_model,
    model_filepath: str,
    noise_budget: int = 300,
    budget_options: list = None,
    test_budgets: list = None,
    constraint_method: str = "lagrangian_od_ov",
    output_file: str = None,
):
    """Test a trained agent on benchmark expressions across multiple budgets.

    Parameters
    ----------
    expressions_file : str
        Path to the benchmark expressions file.
    embeddings_model : object
        The loaded embeddings model.
    model_filepath : str
        Path to the saved model (.zip).
    noise_budget : int
        Single budget for backward-compatible calls.
    budget_options : list
        Budget options the model was TRAINED on (needed for correct obs space).
    test_budgets : list
        Budgets to TEST on. If None, uses [noise_budget].
    constraint_method : str
        Constraint method the model was trained with (affects obs space).
    output_file : str
        Path for the output Excel file. If None, auto-generated.
    """

    expressions = load_expressions(expressions_file)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    max_positions = 16

    # Determine budgets to test
    if test_budgets is None:
        test_budgets = [noise_budget]

    # Budget options must match what the model was trained on (for obs space)
    if budget_options is None:
        budget_options = test_budgets  # Assume test budgets = train budgets

    # Create env with correct budget_options and constraint_method
    env = DummyVecEnv([
        lambda: Monitor(
            fheEnv(
                rules_list,
                expressions,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )
    ])

    # Load model
    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    noise_estimator = NoiseEstimator()

    all_results = []

    for budget in test_budgets:
        print(f"\n{'='*60}")
        print(f"  Testing with budget = {budget}")
        print(f"{'='*60}")

        # Set the budget for all episodes
        env.set_options({"budget": budget})

        for expr_idx in range(len(expressions)):
            obs = env.reset()

            wrapper = env.envs[0]
            fhe_env = wrapper.env

            test_expr = fhe_env.initial_expression
            initial_cost = fhe_env.initial_cost
            initial_noise = noise_estimator.estimate(parse_sexpr(test_expr))

            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                steps += 1

            # Use info dict for terminal state (fhe_env may have been
            # auto-reset by DummyVecEnv when done=True).
            terminal = infos[0]
            last_expr = terminal["expression"]
            last_cost = terminal["cost"]
            final_noise = float(terminal["noise"])
            cost_reduction = ((initial_cost - last_cost) / initial_cost * 100) if initial_cost > 0 else 0
            budget_violated = final_noise > budget

            all_results.append({
                "Budget": budget,
                "Expression #": expr_idx + 1,
                "Initial Expression": test_expr,
                "Final Expression": last_expr,
                "Initial Cost": initial_cost,
                "Final Cost": last_cost,
                "Cost Reduction (%)": round(cost_reduction, 2),
                "Initial Noise": round(initial_noise, 2),
                "Final Noise": round(final_noise, 2),
                "Budget Violated": budget_violated,
                "Noise Margin": round(budget - final_noise, 2),
                "Steps": steps,
            })

            status = "VIOLATED" if budget_violated else "OK"
            print(f"  [{status}] Expr {expr_idx+1}: cost {initial_cost}->{last_cost} "
                  f"({cost_reduction:+.1f}%), noise {initial_noise:.0f}->{final_noise:.0f}, "
                  f"margin={budget - final_noise:.0f}")

    # ── Write results to Excel ──────────────────────────────────────────
    df = pd.DataFrame(all_results)

    if output_file is None:
        job_id = os.environ.get("SLURM_JOB_ID", "jobid")
        model_name = os.path.basename(model_filepath).replace(".zip", "")
        output_file = f"test_results_{model_name}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Full results sheet
        df.to_excel(writer, sheet_name="all_results", index=False)

        # Summary per budget
        summary_rows = []
        for budget in test_budgets:
            bdf = df[df["Budget"] == budget]
            summary_rows.append({
                "Budget": budget,
                "Num Expressions": len(bdf),
                "Avg Cost Reduction (%)": round(bdf["Cost Reduction (%)"].mean(), 2),
                "Avg Final Noise": round(bdf["Final Noise"].mean(), 2),
                "Violations": int(bdf["Budget Violated"].sum()),
                "Violation Rate (%)": round(bdf["Budget Violated"].mean() * 100, 1),
                "Avg Noise Margin": round(bdf["Noise Margin"].mean(), 2),
                "Avg Steps": round(bdf["Steps"].mean(), 1),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_per_budget", index=False)

    print(f"\n{'='*60}")
    print(f"Results written to: {output_file}")
    print(f"  - Sheet 'all_results': {len(all_results)} rows (expressions x budgets)")
    print(f"  - Sheet 'summary_per_budget': {len(test_budgets)} rows")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n  {'Budget':>10}  {'Violations':>10}  {'Avg Cost Red%':>14}  {'Avg Noise':>10}")
    print(f"  {'-'*50}")
    for s in summary_rows:
        print(f"  {s['Budget']:>10}  {s['Violations']:>10}  {s['Avg Cost Reduction (%)']:>14.1f}  {s['Avg Final Noise']:>10.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  test_agent_v2 — Trajectory Checkpointing + Safety Rollback + Feasible Split
# ═══════════════════════════════════════════════════════════════════════════════

def test_agent_v2(
    expressions_file: str,
    embeddings_model,
    model_filepath: str,
    noise_budget: int = 300,
    budget_options: list = None,
    test_budgets: list = None,
    constraint_method: str = "nato_sc",
    output_file: str = None,
    save_optimized: str = None,
):
    """Test with trajectory checkpointing, safety rollback, and feasible/infeasible split.

    For each (expression, budget) pair the agent runs a full episode.
    At every step the current (expression, cost, noise) is saved.  After the
    episode, the best intermediate state that satisfies the budget is selected
    (safety rollback).  Results are split into feasible (initial_noise <= budget)
    and infeasible categories for honest evaluation.
    """

    named_pairs = load_expressions_named(expressions_file)
    expr_names = [name for _, name in named_pairs]
    expressions = load_expressions(expressions_file)
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    max_positions = 16

    if test_budgets is None:
        test_budgets = [noise_budget]
    if budget_options is None:
        budget_options = test_budgets

    env = DummyVecEnv([
        lambda: Monitor(
            fheEnv(
                rules_list,
                expressions,
                max_positions=max_positions,
                embeddings_model=embeddings_model,
                budget_options=budget_options,
                constraint_method=constraint_method,
            )
        )
    ])

    model = PPO(policy=HierarchicalMaskablePolicy, env=env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_filepath)
    noise_estimator = NoiseEstimator()

    all_results = []

    for budget in test_budgets:
        print(f"\n{'='*60}")
        print(f"  Testing (v2) with budget = {budget}")
        print(f"{'='*60}")

        env.set_options({"budget": budget})

        for expr_idx in range(len(expressions)):
            t0 = time.perf_counter()

            obs = env.reset()

            wrapper = env.envs[0]
            fhe_env = wrapper.env

            test_expr = fhe_env.initial_expression
            initial_cost = fhe_env.initial_cost
            initial_noise = float(noise_estimator.estimate(parse_sexpr(test_expr)))

            is_feasible = initial_noise <= budget

            # Trajectory checkpoints: step 0 is the initial state
            checkpoints = [{
                "step": 0,
                "expression": test_expr,
                "cost": initial_cost,
                "noise": initial_noise,
            }]

            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])
                steps += 1

                if done:
                    # DummyVecEnv auto-resets on done, so fhe_env already
                    # holds the NEXT episode's state. Read the terminal
                    # state from the info dict captured before the reset.
                    terminal = infos[0]
                    cp_expr = terminal["expression"]
                    cp_cost = terminal["cost"]
                    cp_noise = float(terminal["noise"])
                else:
                    cp_expr = fhe_env.expression
                    cp_cost = fhe_env.current_cost
                    cp_noise = float(noise_estimator.estimate(parse_sexpr(cp_expr)))

                checkpoints.append({
                    "step": steps,
                    "expression": cp_expr,
                    "cost": cp_cost,
                    "noise": cp_noise,
                })

            rl_time_ms = (time.perf_counter() - t0) * 1000

            # Agent's raw result (last checkpoint)
            agent_final = checkpoints[-1]

            # Safety rollback: pick best valid checkpoint (lowest cost among valid)
            valid_checkpoints = [cp for cp in checkpoints if cp["noise"] <= budget]
            if valid_checkpoints:
                best = min(valid_checkpoints, key=lambda cp: cp["cost"])
                safe_expr = best["expression"]
                safe_cost = best["cost"]
                safe_noise = best["noise"]
                best_valid_step = best["step"]
                safety_activated = (best["step"] != agent_final["step"])
            else:
                safe_expr = test_expr
                safe_cost = initial_cost
                safe_noise = initial_noise
                best_valid_step = 0
                safety_activated = True

            agent_cr = ((initial_cost - agent_final["cost"]) / initial_cost * 100) if initial_cost > 0 else 0
            safe_cr = ((initial_cost - safe_cost) / initial_cost * 100) if initial_cost > 0 else 0

            expr_name = expr_names[expr_idx] if expr_idx < len(expr_names) else f"expr_{expr_idx}"

            all_results.append({
                "Budget": budget,
                "Expression #": expr_idx + 1,
                "Expression Name": expr_name,
                "Is Feasible": is_feasible,
                "Initial Cost": initial_cost,
                "Initial Noise": round(initial_noise, 2),
                "Agent Final Cost": agent_final["cost"],
                "Agent Final Noise": round(agent_final["noise"], 2),
                "Agent Cost Reduction (%)": round(agent_cr, 2),
                "Agent Violated": agent_final["noise"] > budget,
                "Safe Final Cost": safe_cost,
                "Safe Final Noise": round(safe_noise, 2),
                "Safe Cost Reduction (%)": round(safe_cr, 2),
                "Safe Violated": safe_noise > budget,
                "Safety Activated": safety_activated,
                "Best Valid Step": best_valid_step,
                "Total Steps": steps,
                "RL Time (ms)": round(rl_time_ms, 1),
                "Noise Margin": round(budget - safe_noise, 2),
                "Trajectory Costs": "|".join(str(cp["cost"]) for cp in checkpoints),
                "Trajectory Noises": "|".join(f'{cp["noise"]:.2f}' for cp in checkpoints),
                "_safe_expr": safe_expr,
            })

            tag = "FEASIBLE" if is_feasible else "INFEAS"
            safe_tag = "SAFE" if safe_noise <= budget else "VIOL"
            print(f"  [{tag}][{safe_tag}] Expr {expr_idx+1}: "
                  f"cost {initial_cost}->{safe_cost} ({safe_cr:+.1f}%), "
                  f"noise {initial_noise:.0f}->{safe_noise:.0f}, "
                  f"rl_time={rl_time_ms:.0f}ms, "
                  f"safety={'ON' if safety_activated else 'off'}")

    # ── Write results to Excel ──────────────────────────────────────────
    excel_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    df = pd.DataFrame(excel_results)

    if output_file is None:
        job_id = os.environ.get("SLURM_JOB_ID", "jobid")
        model_name = os.path.basename(model_filepath).replace(".zip", "")
        output_file = f"test_results_v2_{model_name}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)

        # Summary per budget (all expressions)
        summary_rows = []
        for b in test_budgets:
            bdf = df[df["Budget"] == b]
            summary_rows.append({
                "Budget": b,
                "Num Expressions": len(bdf),
                "Feasible Count": int(bdf["Is Feasible"].sum()),
                "Infeasible Count": int((~bdf["Is Feasible"]).sum()),
                "Agent Avg Cost Red (%)": round(bdf["Agent Cost Reduction (%)"].mean(), 2),
                "Agent Violations": int(bdf["Agent Violated"].sum()),
                "Agent Violation Rate (%)": round(bdf["Agent Violated"].mean() * 100, 1),
                "Safe Avg Cost Red (%)": round(bdf["Safe Cost Reduction (%)"].mean(), 2),
                "Safe Violations": int(bdf["Safe Violated"].sum()),
                "Safe Violation Rate (%)": round(bdf["Safe Violated"].mean() * 100, 1),
                "Safety Activations": int(bdf["Safety Activated"].sum()),
                "Avg Noise Margin": round(bdf["Noise Margin"].mean(), 2),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary_per_budget", index=False)

        # Feasible-only summary (primary metric for paper)
        feasible_rows = []
        for b in test_budgets:
            fdf = df[(df["Budget"] == b) & (df["Is Feasible"])]
            if len(fdf) == 0:
                continue
            feasible_rows.append({
                "Budget": b,
                "Feasible Expressions": len(fdf),
                "Agent Avg Cost Red (%)": round(fdf["Agent Cost Reduction (%)"].mean(), 2),
                "Agent Violations": int(fdf["Agent Violated"].sum()),
                "Agent Violation Rate (%)": round(fdf["Agent Violated"].mean() * 100, 1),
                "Safe Avg Cost Red (%)": round(fdf["Safe Cost Reduction (%)"].mean(), 2),
                "Safe Violations": int(fdf["Safe Violated"].sum()),
                "Safe Violation Rate (%)": round(fdf["Safe Violated"].mean() * 100, 1),
                "Safety Activations": int(fdf["Safety Activated"].sum()),
                "Avg Safe Noise Margin": round(fdf["Noise Margin"].mean(), 2),
            })
        if feasible_rows:
            pd.DataFrame(feasible_rows).to_excel(writer, sheet_name="feasible_summary", index=False)

        # Infeasible-only summary
        infeasible_rows = []
        for b in test_budgets:
            idf = df[(df["Budget"] == b) & (~df["Is Feasible"])]
            if len(idf) == 0:
                continue
            infeasible_rows.append({
                "Budget": b,
                "Infeasible Expressions": len(idf),
                "Agent Avg Cost Red (%)": round(idf["Agent Cost Reduction (%)"].mean(), 2),
                "Agent Avg Final Noise": round(idf["Agent Final Noise"].mean(), 2),
                "Safe Avg Cost Red (%)": round(idf["Safe Cost Reduction (%)"].mean(), 2),
                "Avg Noise Increase": round((idf["Agent Final Noise"] - idf["Initial Noise"]).mean(), 2),
            })
        if infeasible_rows:
            pd.DataFrame(infeasible_rows).to_excel(writer, sheet_name="infeasible_summary", index=False)

    print(f"\n{'='*60}")
    print(f"Results (v2) written to: {output_file}")
    print(f"  - Sheet 'all_results': {len(all_results)} rows")
    print(f"  - Sheet 'summary_per_budget': {len(test_budgets)} budgets")
    if feasible_rows:
        print(f"  - Sheet 'feasible_summary': {len(feasible_rows)} budgets (primary metric)")
    if infeasible_rows:
        print(f"  - Sheet 'infeasible_summary': {len(infeasible_rows)} budgets")
    print(f"{'='*60}")

    # Print feasible summary table
    if feasible_rows:
        print(f"\n  FEASIBLE pairs (primary metric):")
        print(f"  {'Budget':>10}  {'N':>4}  {'AgentViol%':>10}  {'SafeViol%':>10}  {'SafeCR%':>8}  {'SafetyON':>8}")
        print(f"  {'-'*60}")
        for r in feasible_rows:
            print(f"  {r['Budget']:>10}  {r['Feasible Expressions']:>4}  "
                  f"{r['Agent Violation Rate (%)']:>10.1f}  {r['Safe Violation Rate (%)']:>10.1f}  "
                  f"{r['Safe Avg Cost Red (%)']:>8.1f}  {r['Safety Activations']:>8}")

    # Save RL-optimized expressions for downstream compilation
    if save_optimized:
        best_per_expr = {}
        for r in all_results:
            name = r["Expression Name"]
            if name not in best_per_expr or r["Safe Cost Reduction (%)"] > best_per_expr[name]["Safe Cost Reduction (%)"]:
                best_per_expr[name] = r
        with open(save_optimized, "w") as f:
            for name, r in best_per_expr.items():
                idx = r["Expression #"] - 1
                safe_expr = r.get("_safe_expr", expressions[idx])
                f.write(f"{safe_expr}:{name}\n")
        print(f"\nRL-optimized expressions saved to: {save_optimized}")
        print(f"  {len(best_per_expr)} expressions (best result per expression across budgets)")
