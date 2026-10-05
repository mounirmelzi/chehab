"""
Quick, cheap sanity check — just runs the same two load_expressions() calls
that train_agent() runs at startup, and prints the final counts. No env,
no PPO, no GPU work. Takes seconds, not hours.

Usage (from RL/, inside chehabEnv):
    python check_dataset_size.py fhe_rl/datasets/final_llm_dataset.txt.cleaned
"""
import sys
import io
import contextlib

from fhe_rl.utils import load_expressions


def main(dataset_path: str):
    # Suppress the per-line exception prints (the '/' , 'v2_0',
    # 'Too many variables...' noise) so only the summary counts show.
    # Nothing else changes — same function, same logic, just quieter.
    buf = io.StringIO()

    print(f"Loading validation set: ./fhe_rl/datasets/benchmarks.txt")
    with contextlib.redirect_stdout(buf):
        benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt")
    print(f"  -> {len(benchmarks)} unique valid benchmark expressions")

    print(f"\nLoading training dataset: {dataset_path}")
    with contextlib.redirect_stdout(buf):
        expressions = load_expressions(dataset_path, benchmarks)
    print(f"  -> {len(expressions)} unique valid training expressions")

    print(f"\nTotal rejected during this second pass "
          f"(malformed OR too-many-variables OR unsupported-op OR "
          f"overlaps validation set): "
          f"{'unknown from here — rerun without redirect to see reasons'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_dataset_size.py <path_to_dataset.txt>")
        sys.exit(1)
    main(sys.argv[1])