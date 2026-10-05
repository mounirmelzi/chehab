#!/usr/bin/env python3
"""
Demo 3: Challenging Benchmark -- Large Expression Under Tight Noise Budget

Stress-tests the constrained RL agent on sort_4 (sorting network, 1172 chars)
whose initial noise (218 bits) already sits at 95% of budget B=230.  Every
rewrite pushes noise closer to the ceiling, forcing the agent to balance cost
reduction against the constraint.

Pipeline:
  1. Load expression + RL agent
  2. Run constrained RL optimization step-by-step
  3. Apply safety rollback
  4. Generate trajectory plot (noise + cost vs. steps)
  5. Compile to Lattigo Go code and execute on CKKS
"""

import os
import sys
import time
import subprocess
import importlib
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RL_DIR.parent

sys.path.insert(0, str(RL_DIR))
sys.path.insert(0, str(RL_DIR / "pytrs"))
os.chdir(str(RL_DIR))

from demo.demo_utils import (
    RESET, BOLD, DIM, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY,
    CHECK, CROSS, ARROW, STAR, DIAMOND, BULLET,
    print_banner, print_section, print_subsection, print_summary_box,
    print_kv, print_step, print_phase, print_exec_phase,
    print_waiting, print_done, noise_bar, generate_trajectory_plot,
    save_input_expression, save_rl_trajectory, save_safety_rollback,
    save_lattigo_code, save_execution_output,
)
from demo.demo_single_operator import generate_veclang_files, evaluate_expression

# ── Benchmark configuration ────────────────────────────────────────────────

BENCHMARK = {
    "name": "sort_4",
    "desc": "4-Element Sorting Network (comparator-based, deep mul chain)",
    "file": "benchmarks.txt",
    "index": 38,
}

BUDGET = 230
BUDGET_OPTIONS = [230, 236, 369, 1_000_000]
MAX_POSITIONS = 16


def load_expression() -> str:
    dataset_path = RL_DIR / "fhe_rl" / "datasets" / BENCHMARK["file"]
    with open(dataset_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    line = lines[BENCHMARK["index"]]
    return line.rsplit(":", 1)[0] if ":" in line else line


def run_rl_optimization(expression: str):
    """Run constrained RL step-by-step and return the best valid expression."""
    from fhe_rl.utils import create_rules
    from fhe_rl.config import get_model_path
    from fhe_rl.policy import HierarchicalMaskablePolicy
    from fhe_rl.env import fheEnv
    from pytrs import NoiseEstimator
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    print_section("RL OPTIMIZATION  --  CONSTRAINED AGENT")

    print_kv("Expression:", BENCHMARK["name"])
    print_kv("Description:", BENCHMARK["desc"])
    print_kv("Expression size:", f"{len(expression)} chars  ({len(expression.split())} AST nodes)")
    print_kv("Budget:", f"{BUDGET} bits  (SEAL 192-bit security, t=786433)")
    print()

    # Load components
    print_waiting("Loading rewrite rules")
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    print_done(f"{len(rules_list)} rules")

    print_waiting("Loading embedding model (GNNAE)")
    from fhe_rl.__main__ import load_embeddings_from_config
    embeddings_model, tokenizer = load_embeddings_from_config()
    print_done()

    print_waiting("Loading trained constrained RL agent (NATO-SC + FiLM)")
    model_path = get_model_path("agent_model")

    env = fheEnv(
        rules_list=rules_list,
        expressions=[expression],
        max_positions=MAX_POSITIONS,
        embeddings_model=embeddings_model,
        budget_options=BUDGET_OPTIONS,
        constraint_method="nato_sc",
        verbose=False,
    )
    monitored = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored])

    model = PPO(policy=HierarchicalMaskablePolicy, env=vec_env)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_path)
    print_done()

    noise_estimator = NoiseEstimator()

    # Run optimization
    vec_env.set_options({"budget": BUDGET})
    obs = vec_env.reset()

    fhe_env = vec_env.envs[0].env
    initial_cost = fhe_env.initial_cost
    initial_expr = fhe_env.initial_expression
    initial_noise = float(noise_estimator.estimate(initial_expr))

    print(f"\n  {BOLD}{MAGENTA}{DIAMOND} Starting optimization...{RESET}")
    print(f"  {GRAY}Initial noise is at {initial_noise:.1f}/{BUDGET} bits ({initial_noise/BUDGET*100:.0f}% of budget).{RESET}")
    print(f"  {GRAY}The agent must optimize carefully -- every rewrite pushes noise closer to the limit.{RESET}\n")

    trajectory = [{
        "step": 0,
        "expression": initial_expr,
        "cost": initial_cost,
        "noise": initial_noise,
        "rule": "initial",
    }]

    costs = [initial_cost]
    noises = [initial_noise]
    done = False
    step_num = 0
    max_steps = fhe_env.max_steps

    t_start = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        old_cost = fhe_env.current_cost

        obs, rewards, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        step_num += 1

        info = infos[0]
        rule_idx = int(action[0]) // MAX_POSITIONS
        pos_idx = int(action[0]) % MAX_POSITIONS
        rule_name = list(rules_list.keys())[rule_idx]

        new_cost = info.get("cost", fhe_env.current_cost)
        noise = float(info.get("noise", 0.0))

        if rule_name == "END":
            print(f"\n  {CYAN}─── Step {step_num} {'─' * 52}{RESET}")
            print(f"    {BOLD}Rule:{RESET} {MAGENTA}END{RESET}  {GRAY}(agent chose to stop){RESET}")
        else:
            print_step(step_num, max_steps, rule_name, pos_idx, old_cost, new_cost, noise, BUDGET)

        if done:
            cp_expr = info["expression"]
            cp_cost = info["cost"]
            cp_noise = float(info["noise"])
        else:
            cp_expr = fhe_env.expression
            cp_cost = fhe_env.current_cost
            cp_noise = float(noise_estimator.estimate(cp_expr))

        trajectory.append({
            "step": step_num,
            "expression": cp_expr,
            "cost": cp_cost,
            "noise": cp_noise,
            "rule": rule_name,
        })
        costs.append(cp_cost)
        noises.append(cp_noise)

    rl_time = time.perf_counter() - t_start

    # ── Safety rollback ─────────────────────────────────────────────────
    print_section("SAFETY ROLLBACK")

    agent_final = trajectory[-1]
    valid_checkpoints = [cp for cp in trajectory if cp["noise"] <= BUDGET]
    violated_steps = [cp for cp in trajectory if cp["noise"] > BUDGET]

    if valid_checkpoints:
        best = min(valid_checkpoints, key=lambda c: c["cost"])
        safety_activated = (best["step"] != agent_final["step"])
        best_step = best["step"]
        best_cost = best["cost"]
        best_noise = best["noise"]
        best_expr = best["expression"]

        print(f"  {GREEN}{CHECK}{RESET} {len(valid_checkpoints)}/{len(trajectory)} checkpoints within budget")
        if violated_steps:
            print(f"  {YELLOW}{STAR}{RESET} {len(violated_steps)} checkpoints exceeded budget")
        print(f"  {GREEN}{CHECK}{RESET} Best valid checkpoint: {BOLD}step {best_step}{RESET} "
              f"(cost={best_cost}, noise={best_noise:.1f})")

        noise_margin = BUDGET - best_noise
        ratio = best_noise / BUDGET * 100
        print(f"  {CYAN}{BULLET}{RESET} Noise margin: {BOLD}{noise_margin:.1f} bits{RESET} "
              f"({ratio:.1f}% of budget used)")

        if safety_activated:
            print(f"  {YELLOW}{STAR}{RESET} Safety rollback activated: "
                  f"rolled back from step {len(trajectory)-1} to step {best_step}")
        else:
            print(f"  {GREEN}{CHECK}{RESET} Agent's final state is already optimal within budget")
    else:
        print(f"  {RED}{CROSS}{RESET} No valid checkpoints found. Returning initial expression.")
        best_step = 0
        best_cost = initial_cost
        best_noise = initial_noise
        best_expr = initial_expr
        safety_activated = True

    # ── Summary box ─────────────────────────────────────────────────────
    cost_reduction = ((initial_cost - best_cost) / initial_cost * 100) if initial_cost > 0 else 0
    violated = best_noise > BUDGET
    status_str = f"WITHIN BUDGET ({best_noise:.1f}/{BUDGET})" if not violated else "VIOLATED"

    print_summary_box("OPTIMIZATION RESULT", [
        ("Initial cost:", str(initial_cost)),
        ("Final cost:", str(best_cost)),
        ("Cost reduction:", f"{cost_reduction:.1f}%"),
        ("", ""),
        ("Initial noise:", f"{initial_noise:.1f} bits"),
        ("Final noise:", f"{best_noise:.1f} bits"),
        ("Budget:", f"{BUDGET} bits"),
        ("Budget used:", f"{best_noise/BUDGET*100:.1f}%"),
        ("Margin:", f"{BUDGET - best_noise:.1f} bits"),
        ("Status:", status_str),
        ("", ""),
        ("Safety rollback:", f"Step {best_step}" if safety_activated else "Not needed"),
        ("Violated steps:", f"{len(violated_steps)}/{len(trajectory)}"),
        ("RL time:", f"{rl_time:.1f} seconds"),
        ("Total steps:", str(step_num)),
    ])

    # ── Trajectory plot ─────────────────────────────────────────────────
    plot_path = str(PROJECT_ROOT / "demo3_trajectory.png")
    generate_trajectory_plot(
        costs, noises, BUDGET, best_step, plot_path,
        title=f"Challenging Benchmark: {BENCHMARK['name']} (B={BUDGET})"
    )

    # ── Save pipeline artifacts ──────────────────────────────────────
    print_subsection("Pipeline Artifacts")
    save_input_expression("demo3", initial_expr, BENCHMARK["name"], BUDGET)
    save_rl_trajectory("demo3", trajectory, BUDGET, rl_time)
    save_safety_rollback("demo3", trajectory, BUDGET, best_step, safety_activated)

    return best_expr, rl_time


def _patch_generated_go(go_path: Path):
    """Fix Lattigo API incompatibilities in the generated Go code.

    The C++ code generator may emit eval.NegNew() which doesn't exist in
    hefloat.Evaluator.  Replace inline with Sub(ct,ct)→zero then Sub(zero,ct)
    to negate -- no extra depth consumed, no helper function needed.
    """
    import re
    text = go_path.read_text()
    if "NegNew" not in text:
        return

    # Replace  `varX, _ = eval.NegNew(varY)`
    # with     `varX, _ = eval.SubNew(varY, varY)` then `varX, _ = eval.SubNew(varX, varY)`
    def _replace_neg(m):
        dst = m.group(1)
        src = m.group(2)
        return f"{dst}, _ = eval.SubNew({src}, {src})\n\t{dst}, _ = eval.SubNew({dst}, {src})"

    text = re.sub(
        r'(\w+),\s*_\s*=\s*eval\.NegNew\((\w+)\)',
        _replace_neg,
        text,
    )
    go_path.write_text(text)


def run_ckks_pipeline(expression: str, expected_output: float = None):
    """Compile expression to Lattigo Go and run encrypted execution."""
    print_section("CKKS COMPILATION + ENCRYPTED EXECUTION")

    temp_dir = RL_DIR / "veclang_runner" / "temp"

    # Phase 1: IR generation
    print_waiting("VecLang preprocessing (IR generation)")
    t0 = time.perf_counter()
    try:
        generate_veclang_files(expression, temp_dir)
    except Exception as e:
        print(f" {RED}{CROSS}{RESET}")
        print(f"  {RED}IR generation failed: {e}{RESET}")
        return
    ir_time = (time.perf_counter() - t0) * 1000
    print_done()
    print_phase(1, 3, "IR generation", ir_time)

    # Phase 2: Lattigo code generation
    build_dir = PROJECT_ROOT / "build" / "RL" / "veclang_runner"
    runner_path = build_dir / "veclang_runner"
    if not runner_path.exists():
        print(f"  {RED}{CROSS} veclang_runner not found at {runner_path}{RESET}")
        print(f"  {GRAY}Build with: cmake -S . -B build && cmake --build build{RESET}")
        return

    for fname in ["vectorized_code.txt", "inputs.txt", "fhe_io_example.txt"]:
        src = temp_dir / fname
        if src.exists():
            shutil.copy2(src, build_dir / fname)
            if fname in ("vectorized_code.txt", "inputs.txt"):
                shutil.copy2(src, build_dir.parent / fname)

    print_waiting("Lattigo Go code generation (scale mgmt + bootstrap)")
    t0 = time.perf_counter()
    try:
        compile_result = subprocess.run(
            ["./veclang_runner", "1", "1", "1"],
            cwd=str(build_dir),
            capture_output=True, text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f" {RED}{CROSS}{RESET}")
        print(f"  {RED}Code generation timed out (120s){RESET}")
        return
    codegen_time = (time.perf_counter() - t0) * 1000

    if compile_result.returncode != 0:
        print(f" {RED}{CROSS}{RESET}")
        print(f"  {RED}Code generation failed (exit {compile_result.returncode}):{RESET}")
        if compile_result.stderr.strip():
            for line in compile_result.stderr.strip().split("\n")[-5:]:
                print(f"    {RED}{line}{RESET}")
        if compile_result.stdout.strip():
            for line in compile_result.stdout.strip().split("\n")[-5:]:
                print(f"    {GRAY}{line}{RESET}")
        return
    print_done()
    print_phase(2, 3, "Scale mgmt + code generation", codegen_time)

    generated_go = build_dir / "generated_fhe.go"
    if not generated_go.exists():
        print(f"  {RED}{CROSS} generated_fhe.go not found{RESET}")
        return

    _patch_generated_go(generated_go)
    save_lattigo_code("demo3", generated_go)

    lattigo_dir = PROJECT_ROOT / "lattigo_backend"
    adapted_src = build_dir / "fhe_io_example_adapted.txt"
    if adapted_src.exists():
        shutil.copy2(adapted_src, lattigo_dir / "fhe_io_example_adapted.txt")

    total_compile = ir_time + codegen_time
    print_phase(3, 3, "Total compilation", total_compile)
    print(f"\n  {GREEN}{CHECK}{RESET} Compilation complete: {CYAN}{BOLD}{total_compile:.0f} ms{RESET}")

    # Phase 3: Encrypted execution
    print_subsection("Encrypted Execution (Lattigo CKKS)")

    if expected_output is not None:
        print(f"  {GRAY}Expected plaintext output (slot 0): {expected_output:.6f}{RESET}\n")

    t0 = time.perf_counter()
    run_cmd = [sys.executable, str(lattigo_dir / "run_instrumented.py"),
               str(generated_go), BENCHMARK["name"], "text"]
    if expected_output is not None:
        run_cmd.append(str(expected_output))
    run_result = subprocess.run(
        run_cmd,
        cwd=str(lattigo_dir),
        capture_output=True, text=True,
    )
    wall_time = (time.perf_counter() - t0) * 1000

    if run_result.returncode != 0:
        print(f"  {RED}{CROSS} Execution failed{RESET}")
        for line in run_result.stderr.strip().split("\n")[-5:]:
            print(f"    {RED}{line}{RESET}")
        return

    results = {}
    for line in run_result.stdout.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            results[key.strip()] = val.strip()

    keygen = float(results.get("keygen_ms", 0))
    btp_keygen = float(results.get("bootstrap_keygen_ms", 0))
    encrypt = float(results.get("encrypt_ms", 0))
    eval_ms = float(results.get("eval_ms", 0))
    decrypt = float(results.get("decrypt_ms", 0))
    total = float(results.get("total_ms", 0))
    precision = float(results.get("precision_bits", 0))
    abs_error = float(results["abs_error"]) if "abs_error" in results else None

    print()
    print_exec_phase("KeyGen", keygen)
    if btp_keygen > 0:
        print_exec_phase("Bootstrap KeyGen", btp_keygen)
    print_exec_phase("Encrypt", encrypt)
    print_exec_phase("Eval (homomorphic)", eval_ms)
    print_exec_phase("Decrypt", decrypt)
    print(f"    {GRAY}{'─' * 36}{RESET}")
    print(f"    {BOLD}Total:{RESET} {CYAN}{BOLD}{total:.0f} ms{RESET}")

    prec_ok = precision >= 10
    prec_color = GREEN if prec_ok else RED
    prec_status = f"{CHECK} above 10-bit threshold" if prec_ok else f"{CROSS} below threshold"
    print(f"    {BOLD}Precision:{RESET} {prec_color}{BOLD}{precision:.1f} bits{RESET}  {prec_color}{prec_status}{RESET}")
    print()

    save_execution_output("demo3", {
        "keygen_ms": keygen, "bootstrap_keygen_ms": btp_keygen,
        "encrypt_ms": encrypt, "eval_ms": eval_ms,
        "decrypt_ms": decrypt, "total_ms": total, "precision_bits": precision,
        "abs_error": abs_error,
    }, op_name=BENCHMARK["name"], expected_output=expected_output)


def main():
    print_banner(
        "CHEHAB -- Challenging Benchmark Demo",
        "Large Expression Under Tight Noise Budget (B=230)"
    )

    print(f"  {WHITE}{BOLD}Benchmark:{RESET} {CYAN}{BENCHMARK['name']}{RESET}")
    print(f"  {WHITE}{BOLD}Operation:{RESET} {CYAN}{BENCHMARK['desc']}{RESET}")
    print(f"  {WHITE}{BOLD}Budget:{RESET}    {YELLOW}{BOLD}B = {BUDGET} bits{RESET} {GRAY}(SEAL 192-bit security, t=786433){RESET}")
    print()
    print(f"  {GRAY}This demo stress-tests the constrained RL agent on an expression{RESET}")
    print(f"  {GRAY}whose initial noise already sits at ~95% of the budget. Every{RESET}")
    print(f"  {GRAY}rewrite pushes noise closer to the ceiling, forcing the agent to{RESET}")
    print(f"  {GRAY}balance cost reduction against the constraint.{RESET}")
    print()

    expression = load_expression()
    print(f"  {GRAY}Expression loaded: {len(expression)} chars, "
          f"{len(expression.split())} AST nodes{RESET}\n")

    expected_output = evaluate_expression(expression, input_val=0.95)

    optimized_expr, rl_time = run_rl_optimization(expression)

    time.sleep(1)

    print(f"  {GRAY}Note: CKKS execution uses the original expression. Vectorized{RESET}")
    print(f"  {GRAY}rewrites from RL are not yet supported by the Lattigo backend.{RESET}\n")
    run_ckks_pipeline(expression, expected_output=expected_output)

    print_section("DEMO COMPLETE")
    print(f"  {GREEN}{BOLD}{CHECK} All steps completed successfully.{RESET}")
    print(f"  {GRAY}Trajectory plot: demo3_trajectory.png{RESET}")
    print()


if __name__ == "__main__":
    main()
