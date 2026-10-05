#!/usr/bin/env python3
"""
Cross-validate the safety rollback noise estimator against actual CKKS execution.

For each expression:
  1. Estimate BFV noise using the NNLS model (same as safety rollback uses)
  2. Compile to CKKS/Lattigo via veclang_runner
  3. Execute generated Go code with instrumented timing
  4. Report: expression | estimated_noise | budget | within_budget | ckks_precision_bits

This empirically validates that expressions the noise estimator deems "safe"
(within budget) consistently produce correct CKKS results.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "RL"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "RL" / "pytrs"))

from parser import parse_sexpr
from noise_estimator import NoiseEstimator
from cost import calculate_cost, get_multiplicative_depth
from fhe_rl.utils import load_expressions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_expected import eval_veclang


def estimate_noise(expr_str: str, estimator: NoiseEstimator) -> float:
    expr = parse_sexpr(expr_str)
    return float(estimator.estimate(expr))


def compile_to_lattigo(expr_line: str, name: str, proj_root: Path) -> Path | None:
    """Compile a VecLang expression to Lattigo Go via the same pipeline as E2E sbatch.

    Uses: generator.py -> copy files -> veclang_runner binary (backend=1)
    """
    import shutil

    rl_dir = proj_root / "RL"
    build_dir = proj_root / "build" / "RL" / "veclang_runner"
    results_dir = proj_root / "validation_results"

    tmp_expr = results_dir / f"_tmp_validate_{name}.txt"
    tmp_expr.write_text(expr_line + "\n")

    rm_target = build_dir / "generated_fhe.go"
    rm_target.unlink(missing_ok=True)

    gen_result = subprocess.run(
        [sys.executable, "-m", "veclang_runner.generator",
         "--veclang_expression_file", str(tmp_expr)],
        cwd=str(rl_dir),
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": f"{rl_dir}:{rl_dir / 'pytrs'}:{os.environ.get('PYTHONPATH', '')}"}
    )
    if gen_result.returncode != 0:
        print(f"  ERROR: generator failed: {gen_result.stderr[:200]}", file=sys.stderr)
        return None

    temp_dir = rl_dir / "veclang_runner" / "temp"
    vec_file = temp_dir / "vectorized_code.txt"
    if not vec_file.exists() or vec_file.stat().st_size == 0:
        print("  ERROR: generator produced empty vectorized_code.txt", file=sys.stderr)
        return None

    shutil.copy2(str(temp_dir / "fhe_io_example.txt"),
                 str(build_dir / "fhe_io_example.txt"))
    shutil.copy2(str(temp_dir / "vectorized_code.txt"),
                 str(build_dir.parent / "vectorized_code.txt"))
    shutil.copy2(str(temp_dir / "inputs.txt"),
                 str(build_dir.parent / "inputs.txt"))

    runner = build_dir / "veclang_runner"
    if not runner.exists():
        print(f"  ERROR: veclang_runner not found at {runner}", file=sys.stderr)
        return None

    result = subprocess.run(
        [str(runner), "1", "1", "1"],
        cwd=str(build_dir),
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  ERROR: veclang_runner failed: {result.stderr[:200]}", file=sys.stderr)
        return None

    go_file = build_dir / "generated_fhe.go"
    if not go_file.exists() or go_file.stat().st_size == 0:
        print("  ERROR: generated_fhe.go not found or empty", file=sys.stderr)
        return None

    return go_file


def run_ckks(go_file: Path, bench_name: str, lattigo_dir: Path,
             expected_value: float | None = None) -> dict | None:
    """Run instrumented CKKS execution and parse results."""
    import shutil
    dest = lattigo_dir / f"validate_{bench_name}.go"
    shutil.copy2(str(go_file), str(dest))

    io_adapted = go_file.parent / "fhe_io_example_adapted.txt"
    dest_io = lattigo_dir / "fhe_io_example_adapted.txt"
    if io_adapted.exists():
        shutil.copy2(str(io_adapted), str(dest_io))

    env = os.environ.copy()
    env["CGO_ENABLED"] = "0"

    cmd = ["python3", "run_instrumented.py", str(dest), bench_name, "csv"]
    if expected_value is not None:
        cmd.append(f"{expected_value:.15e}")

    result = subprocess.run(
        cmd, cwd=str(lattigo_dir),
        capture_output=True, text=True, env=env, timeout=600
    )
    dest.unlink(missing_ok=True)
    dest_io.unlink(missing_ok=True)

    if result.returncode != 0:
        err_lines = [l for l in result.stderr.splitlines() if not l.startswith("[info]")]
        err_summary = "\n".join(err_lines[:15])
        print(f"  ERROR: CKKS execution failed:\n{err_summary}", file=sys.stderr)
        return None

    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        return None

    header = lines[0].split(',')
    values = lines[1].split(',')
    return dict(zip(header, values))


def main():
    proj_root = Path(os.environ.get("PROJ_ROOT", Path(__file__).resolve().parent.parent))
    lattigo_dir = proj_root / "lattigo_backend"

    budgets = [230, 236, 369]

    bench_files = [
        proj_root / "RL" / "fhe_rl" / "datasets" / "dnn_benchmarks.txt",
        proj_root / "RL" / "fhe_rl" / "datasets" / "dnn_benchmarks_scaled.txt",
    ]

    expressions = []
    for bf in bench_files:
        if bf.exists():
            with open(bf) as f:
                raw_lines = [l.strip() for l in f if l.strip()]
            for line in raw_lines:
                if ":" in line:
                    name = line.split(":")[-1]
                    expr_str = ":".join(line.split(":")[:-1])
                else:
                    name = f"expr_{len(expressions)}"
                    expr_str = line
                expressions.append((name, expr_str, line))
            print(f"Loaded {len(raw_lines)} expressions from {bf.name}", file=sys.stderr)

    print(f"Total expressions: {len(expressions)}", file=sys.stderr)

    estimator = NoiseEstimator()
    print("Noise estimator loaded", file=sys.stderr)

    results_dir = proj_root / "validation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "rollback_validation.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "expression", "budget", "estimated_noise", "within_budget",
            "mult_depth", "cost",
            "ckks_precision_bits", "ckks_abs_error",
            "keygen_ms", "encrypt_ms", "eval_ms", "decrypt_ms", "total_ms",
            "estimator_correct"
        ])

        for name, expr_str, full_line in expressions:
            try:
                noise = estimate_noise(expr_str, estimator)
                parsed = parse_sexpr(expr_str)
                cost = calculate_cost(parsed)
                depth = get_multiplicative_depth(parsed)
            except Exception as e:
                print(f"  SKIP {name}: noise estimation failed: {e}", file=sys.stderr)
                continue

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Expression: {name}", file=sys.stderr)
            print(f"  Cost: {cost}, Mult depth: {depth}, Est noise: {noise:.2f}", file=sys.stderr)

            try:
                expected_val = eval_veclang(expr_str)
                print(f"  Expected plaintext output: {expected_val:.6e}", file=sys.stderr)
            except Exception:
                expected_val = None

            go_file = compile_to_lattigo(full_line, name, proj_root)
            if go_file is None:
                print(f"  SKIP: compilation failed", file=sys.stderr)
                for budget in budgets:
                    within = noise <= budget
                    writer.writerow([name, budget, f"{noise:.2f}", within, depth, cost,
                                     "", "", "", "", "", "", "", ""])
                continue

            ckks_result = run_ckks(go_file, name, lattigo_dir, expected_val)
            if ckks_result is None:
                print(f"  SKIP: CKKS execution failed", file=sys.stderr)
                for budget in budgets:
                    within = noise <= budget
                    writer.writerow([name, budget, f"{noise:.2f}", within, depth, cost,
                                     "", "", "", "", "", "", "", ""])
                continue

            prec_bits = float(ckks_result.get("precision_bits", 0))
            abs_error = ckks_result.get("abs_error", "")
            keygen_ms = ckks_result.get("keygen_ms", "")
            encrypt_ms = ckks_result.get("encrypt_ms", "")
            eval_ms = ckks_result.get("eval_ms", "")
            decrypt_ms = ckks_result.get("decrypt_ms", "")
            total_ms = ckks_result.get("total_ms", "")

            print(f"  CKKS precision: {prec_bits:.2f} bits, abs_error: {abs_error}", file=sys.stderr)

            for budget in budgets:
                within = noise <= budget
                correct = (within and prec_bits > 5) or (not within)
                writer.writerow([
                    name, budget, f"{noise:.2f}", within, depth, cost,
                    f"{prec_bits:.2f}", abs_error,
                    keygen_ms, encrypt_ms, eval_ms, decrypt_ms, total_ms,
                    correct
                ])

    print(f"\nResults written to {csv_path}", file=sys.stderr)
    print("\nCSV contents:", file=sys.stderr)
    with open(csv_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
