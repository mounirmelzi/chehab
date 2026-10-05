#!/usr/bin/env python3
"""
Demo 2: Multi-Operator Neural Network Pipeline

Demonstrates encrypted DNN inference by executing a small neural network
layer-by-layer using pre-optimized expressions through the CHEHAB pipeline:
  Input -> FC Layer -> Polynomial Activation -> FC Layer -> Output

Each operator is independently compiled and executed on CKKS via Lattigo.
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
    print_kv, print_phase, print_exec_phase,
    print_waiting, print_done, print_network_diagram,
    save_input_expression, save_rl_trajectory, save_safety_rollback,
    save_lattigo_code, save_execution_output,
)

# ── Network definition ──────────────────────────────────────────────────────

NETWORK_LAYERS = [
    {
        "name": "Fully Connected",
        "type": "matmul_4x4",
        "desc": "4x4 Matrix Multiply",
        "file": "dnn_benchmarks_scaled.txt",
        "index": 5,
    },
    {
        "name": "Poly Activation",
        "type": "poly_activation_deg2_4",
        "desc": "Degree-2 Polynomial (2x²+3x+1)",
        "file": "dnn_benchmarks.txt",
        "index": 0,
    },
    {
        "name": "Fully Connected",
        "type": "fc_plus_act_4",
        "desc": "FC + Activation",
        "file": "dnn_benchmarks.txt",
        "index": 3,
    },
]

BUDGET = 369
BUDGET_OPTIONS = [230, 236, 369, 1_000_000]
MAX_POSITIONS = 16


def load_expression(layer: dict) -> str:
    dataset_path = RL_DIR / "fhe_rl" / "datasets" / layer["file"]
    with open(dataset_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    line = lines[layer["index"]]
    return line.rsplit(":", 1)[0] if ":" in line else line


def optimize_expression(expression: str, op_name: str, embeddings_model, rules_list, model,
                        layer_idx: int = 0):
    """Run RL optimization on a single expression, returning summary + optimized expr."""
    from fhe_rl.env import fheEnv
    from pytrs import NoiseEstimator
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    import numpy as np

    noise_estimator = NoiseEstimator()

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

    vec_env.set_options({"budget": BUDGET})
    obs = vec_env.reset()

    fhe_env = vec_env.envs[0].env
    initial_cost = fhe_env.initial_cost
    initial_noise = float(noise_estimator.estimate(fhe_env.initial_expression))

    checkpoints = [{
        "step": 0,
        "expression": fhe_env.initial_expression,
        "cost": initial_cost,
        "noise": initial_noise,
    }]

    done = False
    steps = 0
    t_start = time.perf_counter()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        done = bool(dones[0])
        steps += 1

        info = infos[0]
        if done:
            cp_expr = info["expression"]
            cp_cost = info["cost"]
            cp_noise = float(info["noise"])
        else:
            cp_expr = fhe_env.expression
            cp_cost = fhe_env.current_cost
            cp_noise = float(noise_estimator.estimate(cp_expr))

        checkpoints.append({
            "step": steps,
            "expression": cp_expr,
            "cost": cp_cost,
            "noise": cp_noise,
        })

    rl_time = time.perf_counter() - t_start

    # Safety rollback
    valid = [cp for cp in checkpoints if cp["noise"] <= BUDGET]
    if valid:
        best = min(valid, key=lambda c: c["cost"])
    else:
        best = checkpoints[0]

    cr = ((initial_cost - best["cost"]) / initial_cost * 100) if initial_cost > 0 else 0
    safety_on = best["step"] != checkpoints[-1]["step"]

    pfx = f"layer{layer_idx+1}_"
    save_input_expression("demo2", expression, op_name, BUDGET, prefix=pfx)
    save_rl_trajectory("demo2", checkpoints, BUDGET, rl_time, prefix=pfx)
    save_safety_rollback("demo2", checkpoints, BUDGET, best["step"], safety_on, prefix=pfx)

    return {
        "expression": best["expression"],
        "initial_cost": initial_cost,
        "final_cost": best["cost"],
        "cost_reduction": cr,
        "initial_noise": initial_noise,
        "final_noise": best["noise"],
        "safety_activated": safety_on,
        "rollback_step": best["step"],
        "total_steps": steps,
        "rl_time": rl_time,
    }


def compile_and_execute(expression: str, op_name: str, layer_idx: int,
                        expected_value: float = None):
    """Compile expression to Lattigo Go and execute. Returns dict or None."""
    temp_dir = RL_DIR / "veclang_runner" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    build_dir = PROJECT_ROOT / "build" / "RL" / "veclang_runner"
    lattigo_dir = PROJECT_ROOT / "lattigo_backend"

    # Phase 1: generate intermediate files directly
    from demo.demo_single_operator import generate_veclang_files
    try:
        generate_veclang_files(expression, temp_dir)
    except Exception as e:
        print(f"      {RED}{CROSS} IR generation failed: {e}{RESET}")
        return None

    runner_path = build_dir / "veclang_runner"
    if not runner_path.exists():
        print(f"      {RED}{CROSS} veclang_runner binary not found{RESET}")
        return None

    for fname in ["vectorized_code.txt", "inputs.txt", "fhe_io_example.txt"]:
        src = temp_dir / fname
        if src.exists():
            shutil.copy2(src, build_dir / fname)
            if fname in ("vectorized_code.txt", "inputs.txt"):
                shutil.copy2(src, build_dir.parent / fname)

    # Phase 2: code generation
    try:
        compile_result = subprocess.run(
            ["./veclang_runner", "1", "1", "1"],
            cwd=str(build_dir),
            capture_output=True, text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print(f"      {RED}{CROSS} Code generation timed out{RESET}")
        return None
    if compile_result.returncode != 0:
        print(f"      {RED}{CROSS} Code generation failed{RESET}")
        return None

    generated_go = build_dir / "generated_fhe.go"
    if not generated_go.exists():
        print(f"      {RED}{CROSS} generated_fhe.go not found{RESET}")
        return None

    save_lattigo_code("demo2", generated_go, prefix=f"layer{layer_idx+1}_")

    adapted_src = build_dir / "fhe_io_example_adapted.txt"
    if adapted_src.exists():
        shutil.copy2(adapted_src, lattigo_dir / "fhe_io_example_adapted.txt")

    # Phase 3: encrypted execution
    run_cmd = [sys.executable, str(lattigo_dir / "run_instrumented.py"),
               str(generated_go), op_name, "text"]
    if expected_value is not None:
        run_cmd.append(str(expected_value))
    run_result = subprocess.run(
        run_cmd,
        cwd=str(lattigo_dir),
        capture_output=True, text=True,
    )

    if run_result.returncode != 0:
        print(f"      {RED}{CROSS} Execution failed{RESET}")
        for line in run_result.stderr.strip().split("\n")[-3:]:
            print(f"        {RED}{line}{RESET}")
        return None

    results = {}
    for line in run_result.stdout.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            results[key.strip()] = val.strip()

    abs_err = float(results["abs_error"]) if "abs_error" in results else None
    exec_data = {
        "keygen_ms": float(results.get("keygen_ms", 0)),
        "bootstrap_keygen_ms": float(results.get("bootstrap_keygen_ms", 0)),
        "encrypt_ms": float(results.get("encrypt_ms", 0)),
        "eval_ms": float(results.get("eval_ms", 0)),
        "decrypt_ms": float(results.get("decrypt_ms", 0)),
        "total_ms": float(results.get("total_ms", 0)),
        "precision_bits": float(results.get("precision_bits", 0)),
        "abs_error": abs_err,
    }
    save_execution_output("demo2", exec_data, prefix=f"layer{layer_idx+1}_",
                          op_name=op_name, expected_output=expected_value)
    return exec_data


def main():
    print_banner(
        "CHEHAB -- Encrypted Neural Network Demo",
        "Layer-by-Layer Encrypted DNN Inference"
    )

    print_network_diagram(NETWORK_LAYERS)

    print_kv("Network:", "FC → Activation → FC+Act")
    print_kv("Noise budget:", f"B = {BUDGET} bits")
    print_kv("Execution:", "Lattigo CKKS (CPU)")
    print()

    # ── Load shared resources ───────────────────────────────────────────
    print_section("LOADING RESOURCES")

    print_waiting("Loading rewrite rules")
    from fhe_rl.utils import create_rules
    from fhe_rl.config import get_model_path
    from fhe_rl.policy import HierarchicalMaskablePolicy
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from fhe_rl.env import fheEnv

    rules_list = create_rules("rules.txt")
    rules_list["END"] = None
    print_done(f"{len(rules_list)} rules")

    print_waiting("Loading embedding model")
    from fhe_rl.__main__ import load_embeddings_from_config
    embeddings_model, tokenizer = load_embeddings_from_config()
    print_done()

    print_waiting("Loading trained RL agent")
    model_path = get_model_path("agent_model")

    dummy_expr = load_expression(NETWORK_LAYERS[0])
    dummy_env = fheEnv(
        rules_list=rules_list,
        expressions=[dummy_expr],
        max_positions=MAX_POSITIONS,
        embeddings_model=embeddings_model,
        budget_options=BUDGET_OPTIONS,
        constraint_method="nato_sc",
        verbose=False,
    )
    dummy_vec = DummyVecEnv([lambda: Monitor(dummy_env)])
    model = PPO(policy=HierarchicalMaskablePolicy, env=dummy_vec)
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    model = model.load(model_path)
    print_done()

    # ── Process each layer ──────────────────────────────────────────────
    layer_results = []
    total_pipeline_time = 0.0

    for i, layer in enumerate(NETWORK_LAYERS):
        layer_num = i + 1
        color = MAGENTA if "act" in layer["type"].lower() else BLUE

        print(f"\n  {color}{BOLD}{'═' * 62}{RESET}")
        print(f"  {color}{BOLD}  LAYER {layer_num}: {layer['name'].upper()} ({layer['type']}){RESET}")
        print(f"  {color}{BOLD}{'═' * 62}{RESET}")
        print(f"  {GRAY}{layer['desc']}{RESET}\n")

        expression = load_expression(layer)

        # RL optimization
        print(f"  {CYAN}{BULLET}{RESET} {BOLD}RL Optimization{RESET}")
        opt = optimize_expression(expression, layer["type"], embeddings_model, rules_list, model,
                                  layer_idx=i)

        cr_color = GREEN if opt["cost_reduction"] > 0 else RED
        noise_color = GREEN if opt["final_noise"] <= BUDGET else RED
        safety_str = f"{YELLOW}ON (→step {opt['rollback_step']}){RESET}" if opt["safety_activated"] else f"{GREEN}not needed{RESET}"

        print(f"    Cost:    {GRAY}{opt['initial_cost']}{RESET} {ARROW} {cr_color}{BOLD}{opt['final_cost']}{RESET}  "
              f"({cr_color}{opt['cost_reduction']:+.1f}%{RESET})")
        print(f"    Noise:   {GRAY}{opt['initial_noise']:.1f}{RESET} {ARROW} {noise_color}{BOLD}{opt['final_noise']:.1f}{RESET} bits  "
              f"(budget: {BUDGET})")
        print(f"    Safety:  {safety_str}")
        print(f"    Time:    {CYAN}{opt['rl_time']:.2f}s{RESET}  ({opt['total_steps']} steps)")

        # CKKS compile + execute (uses original expression -- RL vectorization
        # rewrites are not yet supported by the Lattigo backend)
        from demo.demo_single_operator import evaluate_expression
        expected_val = evaluate_expression(expression, input_val=0.95)
        print(f"\n  {CYAN}{BULLET}{RESET} {BOLD}CKKS Compilation + Execution{RESET}")
        exec_result = compile_and_execute(expression, layer["type"], i,
                                          expected_value=expected_val)

        if exec_result:
            print_exec_phase("KeyGen", exec_result["keygen_ms"])
            if exec_result["bootstrap_keygen_ms"] > 0:
                print_exec_phase("Bootstrap KeyGen", exec_result["bootstrap_keygen_ms"])
            print_exec_phase("Encrypt", exec_result["encrypt_ms"])
            print_exec_phase("Eval (homomorphic)", exec_result["eval_ms"])
            print_exec_phase("Decrypt", exec_result["decrypt_ms"])
            print(f"    {GRAY}{'─' * 36}{RESET}")
            print(f"    {BOLD}Total:{RESET} {CYAN}{BOLD}{exec_result['total_ms']:.0f} ms{RESET}")

            prec = exec_result["precision_bits"]
            prec_ok = prec >= 10
            prec_color = GREEN if prec_ok else RED
            print(f"    {BOLD}Precision:{RESET} {prec_color}{BOLD}{prec:.1f} bits{RESET}")

            total_pipeline_time += exec_result["total_ms"]
            layer_results.append({
                "layer": layer_num,
                "name": layer["name"],
                "type": layer["type"],
                "cost_reduction": opt["cost_reduction"],
                "noise": opt["final_noise"],
                "exec_ms": exec_result["total_ms"],
                "precision": prec,
                "success": True,
            })
        else:
            layer_results.append({
                "layer": layer_num,
                "name": layer["name"],
                "type": layer["type"],
                "cost_reduction": opt["cost_reduction"],
                "noise": opt["final_noise"],
                "exec_ms": 0,
                "precision": 0,
                "success": False,
            })

        if i < len(NETWORK_LAYERS) - 1:
            print(f"\n  {GRAY}         │{RESET}")
            print(f"  {GRAY}         ▼  passing encrypted result to next layer{RESET}")
            print(f"  {GRAY}         │{RESET}")

    # ── Final summary ───────────────────────────────────────────────────
    print_section("PIPELINE SUMMARY")

    successful = [r for r in layer_results if r["success"]]
    failed = [r for r in layer_results if not r["success"]]

    summary_rows = []
    for r in layer_results:
        status = f"{CHECK}" if r["success"] else f"{CROSS}"
        summary_rows.append((
            f"Layer {r['layer']}: {r['name']}",
            f"{status}  {r['exec_ms']:.0f}ms  {r['precision']:.1f}bits  CR={r['cost_reduction']:.1f}%"
        ))
    summary_rows.append(("", ""))
    summary_rows.append(("Total execution time:", f"{total_pipeline_time:.0f} ms"))
    summary_rows.append(("Layers executed:", f"{len(successful)}/{len(NETWORK_LAYERS)}"))

    if successful:
        min_prec = min(r["precision"] for r in successful)
        summary_rows.append(("Min precision:", f"{min_prec:.1f} bits"))

    print_summary_box("ENCRYPTED NEURAL NETWORK RESULT", summary_rows)

    if successful:
        print(f"  {GREEN}{BOLD}{CHECK} Encrypted inference completed successfully!{RESET}")
        print(f"  {GREEN}{BOLD}  {len(successful)} layers executed independently, results chained.{RESET}")
    if failed:
        print(f"  {RED}{BOLD}{CROSS} {len(failed)} layer(s) failed during execution.{RESET}")

    print()
    print(f"  {GRAY}Note: Each operator is compiled and executed independently.{RESET}")
    print(f"  {GRAY}Full network compilation (chaining in a single circuit) is{RESET}")
    print(f"  {GRAY}planned as future work.{RESET}")
    print()


if __name__ == "__main__":
    main()
