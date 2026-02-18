from stable_baselines3 import PPO
import sys, importlib
from .utils import load_expressions, load_embeddings, create_rules
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
            last_expr = None
            last_cost = initial_cost

            while not done:
                last_expr = fhe_env.expression
                last_cost = fhe_env.current_cost

                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                done = bool(dones[0])
                steps += 1

            final_noise = noise_estimator.estimate(parse_sexpr(last_expr))
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
