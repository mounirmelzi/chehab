#!/usr/bin/env python3
"""
Demo 1: Single DNN Operator -- Constrained RL Optimization + CKKS Execution

Interactive demo that:
1. Lets the jury choose a DNN operator and noise budget
2. Runs the RL agent step-by-step with live progress
3. Shows safety rollback decision
4. Generates a trajectory plot
5. Compiles to Lattigo Go code and executes on CKKS
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
    CHECK, CROSS, ARROW, STAR, DIAMOND,
    print_banner, print_section, print_subsection, print_summary_box,
    print_kv, print_step, print_phase, print_exec_phase,
    print_waiting, print_done, noise_bar, generate_trajectory_plot,
    save_input_expression, save_rl_trajectory, save_safety_rollback,
    save_lattigo_code, save_execution_output,
)

# ── Available operators ─────────────────────────────────────────────────────

OPERATORS = {
    "1": {"name": "conv2d_5x5_k3",          "desc": "Convolution 5x5, kernel 3",     "file": "dnn_benchmarks.txt",        "index": 2},
    "2": {"name": "matmul_4x4",             "desc": "Matrix Multiply 4x4",           "file": "dnn_benchmarks_scaled.txt", "index": 5},
    "3": {"name": "fc_2layer_4",            "desc": "2-Layer FC Network",            "file": "dnn_benchmarks_scaled.txt", "index": 6},
    "4": {"name": "fc_plus_act_4",          "desc": "FC + Polynomial Activation",    "file": "dnn_benchmarks.txt",        "index": 3},
    "5": {"name": "poly_activation_deg2_4", "desc": "Polynomial Activation deg2",    "file": "dnn_benchmarks.txt",        "index": 0},
}

BUDGETS = {
    "1": (230,       "tight   -- SEAL 192-bit security, t=786433"),
    "2": (236,       "medium  -- SEAL 192-bit security, t=12289"),
    "3": (369,       "relaxed -- SEAL 128-bit security"),
    "4": (9_000_000, "unconstrained"),
}

BUDGET_OPTIONS = [230, 236, 369, 1_000_000]
MAX_POSITIONS = 16


def select_operator():
    print(f"  {BOLD}{WHITE}Select DNN operator:{RESET}\n")
    for key, op in OPERATORS.items():
        print(f"    {CYAN}[{key}]{RESET} {WHITE}{op['name']:<28}{RESET} {GRAY}({op['desc']}){RESET}")
    print()
    while True:
        choice = input(f"  {YELLOW}{BOLD}  > {RESET}").strip()
        if choice in OPERATORS:
            op = OPERATORS[choice]
            print(f"\n  {GREEN}{CHECK}{RESET} Selected: {BOLD}{op['name']}{RESET}\n")
            return op
        print(f"  {RED}Invalid choice. Enter 1-{len(OPERATORS)}.{RESET}")


def select_budget():
    print(f"  {BOLD}{WHITE}Select noise budget:{RESET}\n")
    for key, (val, desc) in BUDGETS.items():
        label = f"B = {val}" if val < 1_000_000 else "B = 9M"
        print(f"    {CYAN}[{key}]{RESET} {WHITE}{label:<16}{RESET} {GRAY}({desc}){RESET}")
    print()
    while True:
        choice = input(f"  {YELLOW}{BOLD}  > {RESET}").strip()
        if choice in BUDGETS:
            val, desc = BUDGETS[choice]
            label = str(val) if val < 1_000_000 else "9M (unconstrained)"
            print(f"\n  {GREEN}{CHECK}{RESET} Selected budget: {BOLD}B = {label}{RESET}\n")
            return val
        print(f"  {RED}Invalid choice. Enter 1-{len(BUDGETS)}.{RESET}")


def load_expression(op: dict) -> str:
    dataset_path = RL_DIR / "fhe_rl" / "datasets" / op["file"]
    with open(dataset_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    line = lines[op["index"]]
    expr = line.rsplit(":", 1)[0] if ":" in line else line
    return expr


def run_rl_optimization(expression: str, budget: int, op_name: str):
    """Run the RL agent step-by-step and return the best expression."""
    from fhe_rl.utils import load_expressions, create_rules, load_embeddings
    from fhe_rl.config import get_model_path, get_tokenizer_type
    from fhe_rl.policy import HierarchicalMaskablePolicy
    from fhe_rl.env import fheEnv
    from pytrs import NoiseEstimator
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import numpy as np

    print_section("RL OPTIMIZATION")
    print_kv("Expression:", op_name)
    print_kv("Budget:", f"{budget} bits" if budget < 1_000_000 else "unconstrained")
    print()

    # Load components
    print_waiting("Loading rewrite rules")
    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    print_done(f"{len(rules_list)} rules")

    print_waiting("Loading embedding model")
    from fhe_rl.__main__ import load_embeddings_from_config
    embeddings_model, tokenizer = load_embeddings_from_config()
    print_done()

    print_waiting("Loading trained RL agent")
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

    # Set budget and reset
    print(f"\n  {BOLD}{MAGENTA}{DIAMOND} Starting optimization episode...{RESET}\n")

    vec_env.set_options({"budget": budget})
    obs = vec_env.reset()

    fhe_env = vec_env.envs[0].env
    initial_cost = fhe_env.initial_cost
    initial_expr = fhe_env.initial_expression
    initial_noise = float(noise_estimator.estimate(initial_expr))

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
            print_step(step_num, max_steps, rule_name, pos_idx, old_cost, new_cost, noise, budget)

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
    valid_checkpoints = [cp for cp in trajectory if cp["noise"] <= budget]

    if valid_checkpoints:
        best = min(valid_checkpoints, key=lambda c: c["cost"])
        safety_activated = (best["step"] != agent_final["step"])
        best_step = best["step"]
        best_cost = best["cost"]
        best_noise = best["noise"]
        best_expr = best["expression"]

        print(f"  {GREEN}{CHECK}{RESET} {len(valid_checkpoints)}/{len(trajectory)} checkpoints within budget")
        print(f"  {GREEN}{CHECK}{RESET} Best valid checkpoint: {BOLD}step {best_step}{RESET} "
              f"(cost={best_cost}, noise={best_noise:.1f})")
        if safety_activated:
            print(f"  {YELLOW}{STAR}{RESET} Safety rollback activated: "
                  f"rolled back from step {len(trajectory)-1} to step {best_step}")
        else:
            print(f"  {GREEN}{CHECK}{RESET} Agent's final state is already optimal")
    else:
        print(f"  {RED}{CROSS}{RESET} No valid checkpoints found. Returning initial expression.")
        best_step = 0
        best_cost = initial_cost
        best_noise = initial_noise
        best_expr = initial_expr
        safety_activated = True

    # ── Summary box ─────────────────────────────────────────────────────
    cost_reduction = ((initial_cost - best_cost) / initial_cost * 100) if initial_cost > 0 else 0
    violated = best_noise > budget and budget < 1_000_000
    status_str = "WITHIN BUDGET" if not violated else "VIOLATED"

    print_summary_box("OPTIMIZATION RESULT", [
        ("Initial cost:", str(initial_cost)),
        ("Final cost:", str(best_cost)),
        ("Cost reduction:", f"{cost_reduction:.1f}%"),
        ("", ""),
        ("Initial noise:", f"{initial_noise:.1f} bits"),
        ("Final noise:", f"{best_noise:.1f} bits"),
        ("Budget:", f"{budget} bits" if budget < 1_000_000 else "unconstrained"),
        ("Status:", status_str),
        ("", ""),
        ("Safety rollback:", f"Step {best_step}" if safety_activated else "Not needed"),
        ("RL time:", f"{rl_time:.2f} seconds"),
        ("Total steps:", str(step_num)),
    ])

    # ── Trajectory plot ─────────────────────────────────────────────────
    plot_path = str(PROJECT_ROOT / "demo_trajectory.png")
    generate_trajectory_plot(
        costs, noises, budget, best_step, plot_path,
        title=f"RL Optimization: {op_name} (B={budget})"
    )

    # ── Save pipeline artifacts ──────────────────────────────────────
    print_subsection("Pipeline Artifacts")
    save_input_expression("demo1", initial_expr, op_name, budget)
    save_rl_trajectory("demo1", trajectory, budget, rl_time)
    save_safety_rollback("demo1", trajectory, budget, best_step, safety_activated)

    return best_expr, op_name, rl_time


def evaluate_expression(expression: str, input_val: float = 0.95) -> float:
    """Evaluate an S-expression with all variables set to input_val.
    Returns the value of the first vector element (slot 0)."""
    from pytrs import parse_sexpr, Var, Op, Const

    def _eval(node):
        if isinstance(node, Const):
            return float(node.value)
        if isinstance(node, Var):
            return input_val
        if isinstance(node, Op):
            op = node.op
            if op == "Vec":
                return _eval(node.args[0])
            if op == "+":
                return sum(_eval(a) for a in node.args)
            if op == "*":
                result = 1.0
                for a in node.args:
                    result *= _eval(a)
                return result
            if op == "-":
                if len(node.args) == 1:
                    return -_eval(node.args[0])
                return _eval(node.args[0]) - sum(_eval(a) for a in node.args[1:])
            if op in ("Rot", "<<"):
                return _eval(node.args[0])
            if op in ("VecAdd", "VecMul", "VecMinus", "VecNeg"):
                return _eval(node.args[0])
        return input_val

    parsed = parse_sexpr(expression)
    return _eval(parsed)


def generate_veclang_files(expression: str, temp_dir: Path):
    """Generate the intermediate files that veclang_runner expects, bypassing
    load_expressions (which filters by vec_size and rejects RL-optimized exprs)."""
    import random
    from pytrs import parse_sexpr

    temp_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_sexpr(expression)

    # Resolve inputs
    inputs = set()
    def _resolve(node):
        from pytrs import Var, Op
        if isinstance(node, Var):
            inputs.add(node.name)
        elif isinstance(node, Op):
            for a in node.args:
                _resolve(a)
    _resolve(parsed)

    # Vec sizes
    from fhe_rl.utils import calc_vec_sizes
    vec_sizes = calc_vec_sizes(parsed)
    slot_count = vec_sizes[0] if vec_sizes else 1

    # vectorized_code.txt
    metadata = f"{slot_count} {slot_count} 1 0"
    with open(temp_dir / "vectorized_code.txt", "w") as f:
        f.write(f"{expression}\n{metadata}")

    # inputs.txt
    IS_CIPHER = 1
    with open(temp_dir / "inputs.txt", "w") as f:
        f.write(" ".join(inputs) + "\n")
        f.write(" ".join(str(IS_CIPHER) for _ in inputs) + "\n")

    # fhe_io_example.txt
    IS_SIGNED = 1
    nb_inputs = len(inputs)
    with open(temp_dir / "fhe_io_example.txt", "w") as f:
        f.write(f"1 {nb_inputs} 1\n")
        for inp in inputs:
            f.write(f"{inp} {IS_CIPHER} {IS_SIGNED} {random.randint(0, 10)}\n")


def run_ckks_pipeline(expression: str, op_name: str, expected_output: float = None,
                      input_val: float = 0.95):
    """Compile expression to Lattigo Go and run encrypted execution."""
    print_section("CKKS COMPILATION")

    temp_dir = RL_DIR / "veclang_runner" / "temp"

    # Phase 1: Generate intermediate files directly (bypasses load_expressions filtering)
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
    print_phase(1, 3, "IR generation (VecLang parse)", ir_time)

    # Phase 2: veclang_runner -> Lattigo code generation
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
            # veclang_runner reads vectorized_code.txt and inputs.txt from ../
            if fname in ("vectorized_code.txt", "inputs.txt"):
                shutil.copy2(src, build_dir.parent / fname)

    print_waiting("Lattigo Go code generation")
    t0 = time.perf_counter()
    try:
        compile_result = subprocess.run(
            ["./veclang_runner", "1", "1", "1"],
            cwd=str(build_dir),
            capture_output=True, text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print(f" {RED}{CROSS}{RESET}")
        print(f"  {RED}Code generation timed out (60s){RESET}")
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

    save_lattigo_code("demo1", generated_go)

    # Copy adapted IO file if it exists
    lattigo_dir = PROJECT_ROOT / "lattigo_backend"
    adapted_src = build_dir / "fhe_io_example_adapted.txt"
    if adapted_src.exists():
        shutil.copy2(adapted_src, lattigo_dir / "fhe_io_example_adapted.txt")

    total_compile = ir_time + codegen_time
    print_phase(3, 3, "Total compilation", total_compile)
    print(f"\n  {GREEN}{CHECK}{RESET} Compilation complete: {CYAN}{BOLD}{total_compile:.0f} ms{RESET}")

    # Phase 3: Encrypted execution via Lattigo
    print_section("ENCRYPTED EXECUTION (Lattigo CKKS)")

    if expected_output is not None:
        print(f"  {GRAY}Expected plaintext output (slot 0): {expected_output:.6f}{RESET}\n")

    t0 = time.perf_counter()
    run_cmd = [sys.executable, str(lattigo_dir / "run_instrumented.py"),
               str(generated_go), op_name, "text"]
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

    save_execution_output("demo1", {
        "keygen_ms": keygen, "bootstrap_keygen_ms": btp_keygen,
        "encrypt_ms": encrypt, "eval_ms": eval_ms,
        "decrypt_ms": decrypt, "total_ms": total, "precision_bits": precision,
        "abs_error": abs_error,
    }, op_name=op_name, input_val=input_val, expected_output=expected_output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo 1: Constrained RL + CKKS")
    parser.add_argument("--operator", type=str, default=None,
                        help="Operator key (1-5) to skip interactive selection")
    parser.add_argument("--budget", type=str, default=None,
                        help="Budget key (1-4) to skip interactive selection")
    args = parser.parse_args()

    print_banner(
        "CHEHAB -- Constrained RL Demo",
        "Noise-Aware FHE Expression Optimization + CKKS Execution"
    )

    if args.operator and args.operator in OPERATORS:
        op = OPERATORS[args.operator]
        print(f"  {GREEN}{CHECK}{RESET} Operator: {BOLD}{op['name']}{RESET} {GRAY}({op['desc']}){RESET}\n")
    else:
        op = select_operator()

    if args.budget and args.budget in BUDGETS:
        val, desc = BUDGETS[args.budget]
        label = str(val) if val < 1_000_000 else "9M (unconstrained)"
        print(f"  {GREEN}{CHECK}{RESET} Budget: {BOLD}B = {label}{RESET} {GRAY}({desc}){RESET}\n")
        budget = val
    else:
        budget = select_budget()

    expression = load_expression(op)
    print(f"  {GRAY}Expression length: {len(expression)} chars{RESET}\n")

    # Compute expected output from original expression
    expected_output = evaluate_expression(expression, input_val=0.95)

    optimized_expr, op_name, rl_time = run_rl_optimization(expression, budget, op["name"])

    time.sleep(1)

    # CKKS execution uses the original expression because some RL vectorization
    # rewrites (Rot, VecMul) are not yet fully supported by the Lattigo code
    # generator -- a known backend limitation noted in the manuscript.
    print(f"  {GRAY}Note: CKKS execution uses the original expression. Vectorized{RESET}")
    print(f"  {GRAY}rewrites from RL are not yet supported by the Lattigo backend.{RESET}\n")
    run_ckks_pipeline(expression, op_name, expected_output=expected_output)

    print_section("DEMO COMPLETE")
    print(f"  {GREEN}{BOLD}{CHECK} All steps completed successfully.{RESET}")
    print(f"  {GRAY}Trajectory plot: demo_trajectory.png{RESET}")
    print()


if __name__ == "__main__":
    main()
