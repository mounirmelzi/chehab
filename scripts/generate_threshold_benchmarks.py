#!/usr/bin/env python3
"""
Generate a benchmark dataset focused on expressions with noise CLOSE to
each budget threshold.  These are the "hard" cases where constrained
agents should outperform unconstrained ones.

For each target budget, we generate expressions whose estimated noise
falls within [budget * low_pct, budget * high_pct].  Default range is
80-120 %, which captures the critical zone on both sides of the threshold.

Run from the RL/ directory with pytrs on PYTHONPATH:
    cd $PROJ_ROOT/RL
    python ../scripts/generate_threshold_benchmarks.py
"""

import os
import sys
import random
import time
import argparse

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_rl = os.path.join(_here, os.pardir, "RL")
sys.path.insert(0, os.path.abspath(_rl))
sys.path.insert(0, os.path.join(os.path.abspath(_rl), "pytrs"))

from pytrs import (
    parse_sexpr, Op, VARIABLE_RANGE, CONST_OFFSET,
    PAREN_CLOSE, PAREN_OPEN, node_to_id,
    get_multiplicative_depth,
)
from noise_estimator import NoiseEstimator

BUDGETS = [230, 236, 369]

EXPRESSIONS_PER_BUDGET = 20
LOW_PCT = 0.80
HIGH_PCT = 1.00

VEC_SIZES = [1, 4, 8, 9, 16, 25, 32]
OUTPUT_PATH = "./fhe_rl/datasets/threshold_benchmarks.txt"
MAX_ATTEMPTS_PER_EXPR = 5000

_vc = 0

def _reset():
    global _vc
    _vc = 0

def _v():
    global _vc
    _vc += 1
    return f"v{_vc}"

def _add_tree(depth, extra=2):
    if depth <= 0 or random.random() < 0.4:
        return _v()
    if random.random() < 0.15:
        return f"(- {_add_tree(depth - 1)})"
    op = random.choice(["+", "-"])
    return f"({op} {_add_tree(depth - 1)} {_add_tree(depth - 1)})"

def _scalar_md(target_md, extra=2):
    if target_md <= 0:
        return _add_tree(random.randint(0, extra))
    main_md = target_md - 1
    other_md = random.randint(0, min(2, target_md - 1))
    mc = _scalar_md(main_md, extra)
    oc = _scalar_md(other_md, extra)
    lhs, rhs = (mc, oc) if random.random() < 0.5 else (oc, mc)
    result = f"(* {lhs} {rhs})"
    if random.random() < 0.25:
        ap = _add_tree(random.randint(0, extra))
        op = random.choice(["+", "-"])
        result = (f"({op} {result} {ap})"
                  if random.random() < 0.5
                  else f"({op} {ap} {result})")
    return result

def gen_vec(target_md, vec_size):
    _reset()
    target_idx = random.randint(0, vec_size - 1)
    elems = []
    for i in range(vec_size):
        md = target_md if i == target_idx else random.randint(0, min(target_md, 4))
        elems.append(_scalar_md(md))
    return f"(Vec {' '.join(elems)})"


def _dfs(expr, d=0, nl=None):
    if nl is None:
        nl = []
    if isinstance(expr, Op):
        nl.append((PAREN_OPEN, d))
        nl.append((expr, d))
        for c in expr.args:
            _dfs(c, d + 1, nl)
        nl.append((PAREN_CLOSE, d))
    else:
        nl.append((expr, d))
    return nl

def _tok_seq(expr_str):
    expr = parse_sexpr(expr_str)
    nl = _dfs(expr)
    vm, im = {}, {}
    nvi, nii = VARIABLE_RANGE[0], CONST_OFFSET
    ids = []
    for node, _ in nl:
        if node in (PAREN_OPEN, PAREN_CLOSE):
            ids.append(node)
        else:
            nid, nvi, nii, _ = node_to_id(node, vm, im, nvi, nii)
            ids.append(nid)
    return tuple(ids)


def main():
    ap = argparse.ArgumentParser(
        description="Generate threshold-focused benchmark expressions."
    )
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--output", type=str, default=OUTPUT_PATH)
    ap.add_argument("--per_budget", type=int, default=EXPRESSIONS_PER_BUDGET)
    ap.add_argument("--low_pct", type=float, default=LOW_PCT)
    ap.add_argument("--high_pct", type=float, default=HIGH_PCT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    estimator = NoiseEstimator()

    # noise ~ mult_depth * 33.383, so target MD range per budget:
    md_ranges = {}
    for b in BUDGETS:
        low_noise = b * args.low_pct
        high_noise = b * args.high_pct
        md_low = max(1, int(low_noise / 33.383) - 1)
        md_high = int(high_noise / 33.383) + 2
        md_ranges[b] = (md_low, md_high, low_noise, high_noise)

    print(f"Threshold benchmark generation")
    print(f"  Noise range: [{args.low_pct*100:.0f}%, {args.high_pct*100:.0f}%] of each budget")
    print(f"  Expressions per budget: {args.per_budget}")
    print()
    for b in BUDGETS:
        md_lo, md_hi, n_lo, n_hi = md_ranges[b]
        print(f"  Budget {b:>5}: noise [{n_lo:.0f}, {n_hi:.0f}], MD search [{md_lo}, {md_hi}]")

    all_generated = []
    all_tokens = set()
    t0 = time.time()

    for b in BUDGETS:
        md_lo, md_hi, noise_lo, noise_hi = md_ranges[b]
        generated = []
        attempts = 0

        while len(generated) < args.per_budget and attempts < args.per_budget * MAX_ATTEMPTS_PER_EXPR:
            attempts += 1
            md = random.randint(md_lo, md_hi)
            vs = random.choice(VEC_SIZES)
            expr_str = gen_vec(md, vs)

            try:
                expr = parse_sexpr(expr_str)
            except Exception:
                continue

            actual_md = get_multiplicative_depth(expr)
            if len(expr.args) not in {1, 4, 8, 9, 16, 25, 32}:
                continue

            noise = float(estimator.estimate(expr))
            if noise < noise_lo or noise > noise_hi:
                continue

            try:
                ts = _tok_seq(expr_str)
            except Exception:
                continue
            if ts in all_tokens:
                continue

            all_tokens.add(ts)
            name = f"thresh_b{b}_md{actual_md}_vs{vs}_n{int(noise)}"
            generated.append((expr_str, name, b, actual_md, noise, vs))

            if len(generated) % 5 == 0:
                el = time.time() - t0
                print(f"    Budget {b}: {len(generated)}/{args.per_budget} "
                      f"({attempts} attempts, {el:.1f}s)")

        el = time.time() - t0
        noises = [g[4] for g in generated]
        print(f"  Budget {b}: {len(generated)}/{args.per_budget} generated, "
              f"noise mean={np.mean(noises):.1f}, "
              f"range=[{min(noises):.1f}, {max(noises):.1f}] "
              f"({attempts} attempts, {el:.1f}s)")
        all_generated.extend(generated)

    with open(args.output, "w") as fh:
        for expr_str, name, *_ in all_generated:
            fh.write(f"{expr_str}:{name}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Generated {len(all_generated)} threshold benchmark expressions")
    print(f"  Saved to: {args.output}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*65}")

    print(f"\nPer-budget breakdown:")
    print(f"  {'Budget':>8}  {'Count':>6}  {'Mean Noise':>10}  {'Min':>8}  {'Max':>8}  {'Noise/Budget':>12}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*12}")
    for b in BUDGETS:
        entries = [g for g in all_generated if g[2] == b]
        if not entries:
            continue
        noises = [g[4] for g in entries]
        ratios = [n / b * 100 for n in noises]
        print(f"  {b:8d}  {len(entries):6d}  {np.mean(noises):10.1f}  "
              f"{min(noises):8.1f}  {max(noises):8.1f}  "
              f"{np.mean(ratios):8.1f}%")

    print(f"\nNoise/budget ratio distribution:")
    for b in BUDGETS:
        entries = [g for g in all_generated if g[2] == b]
        if not entries:
            continue
        ratios = sorted([g[4] / b * 100 for g in entries])
        below = sum(1 for r in ratios if r < 100)
        above = len(ratios) - below
        print(f"  Budget {b}: {below} below threshold, {above} above threshold")


if __name__ == "__main__":
    main()
