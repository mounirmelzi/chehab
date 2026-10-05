"""Compute noise (SEAL bits) for every expression in the training dataset.

Usage (from RL/ directory):
    python ../scripts/compute_noise_local.py

Output: scripts/dataset_noise.txt  (one line per expression: noise ||| expression)
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RL", "pytrs"))

from pytrs import parse_sexpr, NoiseEstimator

RL_DIR = os.path.join(os.path.dirname(__file__), "..", "RL")
DATASET = os.path.join(RL_DIR, "fhe_rl", "datasets", "final_llm_dataset.txt")
OUTPUT = os.path.join(os.path.dirname(__file__), "dataset_noise.txt")


def main():
    os.chdir(RL_DIR)

    with open(DATASET) as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded {len(lines)} expressions from {DATASET}")

    estimator = NoiseEstimator()
    results = []
    errors = 0

    for i, expr in enumerate(lines):
        try:
            parsed = parse_sexpr(expr)
            noise = float(estimator.estimate(parsed))
            results.append((noise, expr))
        except Exception as e:
            errors += 1
            results.append((None, expr))
            if errors <= 5:
                print(f"  [WARN] expr #{i+1}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(lines)} done")

    with open(OUTPUT, "w") as f:
        for noise, expr in results:
            noise_str = f"{noise:.2f}" if noise is not None else "ERROR"
            f.write(f"{noise_str} ||| {expr}\n")

    valid = [n for n, _ in results if n is not None]
    print(f"\nDone. {len(valid)} valid, {errors} errors.")
    print(f"Noise stats: min={min(valid):.1f}, max={max(valid):.1f}, "
          f"mean={sum(valid)/len(valid):.1f}, median={sorted(valid)[len(valid)//2]:.1f}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
