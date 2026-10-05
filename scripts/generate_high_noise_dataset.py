#!/usr/bin/env python3
"""
Generate high-noise VecLang expressions and append to the training dataset.

Optimized for constrained RL agents on noise budgets above 220 bits.
Uses a targeted random generator that guarantees specific multiplicative
depths, then verifies with NoiseEstimator before appending.

Run from the RL/ directory with pytrs on PYTHONPATH:
    cd $PROJ_ROOT/RL
    python ../scripts/generate_high_noise_dataset.py
"""

import os
import sys
import random
import time
import argparse

# ── path setup (must come before pytrs imports) ──────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_rl   = os.path.join(_here, os.pardir, "RL")
sys.path.insert(0, os.path.abspath(_rl))
sys.path.insert(0, os.path.join(os.path.abspath(_rl), "pytrs"))

import numpy as np
from pytrs import (
    parse_sexpr,
    Expr, Const, Var, Op,
    VARIABLE_RANGE, CONST_OFFSET,
    PAREN_CLOSE, PAREN_OPEN, node_to_id,
    get_multiplicative_depth,
)
from noise_estimator import NoiseEstimator

# =====================================================================
#  AUGMENTATION PLAN  — optimised for budgets [230, 236, 369, 9M]
# =====================================================================
# Noise ≈ MultDepth × 33.38  (NoiseEstimator's dominant coefficient).
#
# Budget 230 : constrained when noise > 230  →  MultDepth ≥ 7  (234)
# Budget 236 : constrained when noise > 236  →  MultDepth ≥ 8  (267)
# Budget 369 : constrained when noise > 369  →  MultDepth ≥ 12 (401)
#
# Distribution rationale
# ──────────────────────
# • 53 % at MD 7–8   → critical boundary for budgets 230 / 236
# • 18 % at MD 9–10  → bridge zone between 236 and 369
# • 28 % at MD 11–12 → critical boundary for budget 369
#
# This gives the agent plenty of "near-threshold" examples for each
# constrained budget, which is where the learning signal is strongest.
# =====================================================================

AUGMENTATION_PLAN = {
    7:  2000,   # noise ~234  — right at 230 threshold
    8:  1200,   # noise ~267  — above 230/236
    9:   600,   # noise ~301  — bridge
    10:  500,   # noise ~334  — approaching 369
    11: 1000,   # noise ~367  — right at 369 threshold
    12:  700,   # noise ~401  — above 369
}

VEC_SIZES    = [1, 4, 8, 9, 16, 25, 32]
DATASET_PATH = "./fhe_rl/datasets/final_llm_dataset.txt"
BACKUP_SUFFIX = ".bak_before_augmentation"


# =====================================================================
#  Variable counter (unique names per expression)
# =====================================================================
_vc = 0

def _reset():
    global _vc
    _vc = 0

def _v():
    global _vc
    _vc += 1
    return f"v{_vc}"


# =====================================================================
#  Expression generators
# =====================================================================

def _add_tree(depth: int) -> str:
    """Additive-only sub-tree (no multiplications)."""
    if depth <= 0 or random.random() < 0.4:
        return _v()
    if random.random() < 0.15:
        return f"(- {_add_tree(depth - 1)})"
    op = random.choice(["+", "-"])
    return f"({op} {_add_tree(depth - 1)} {_add_tree(depth - 1)})"


def _scalar_md(target_md: int, extra: int = 2) -> str:
    """
    Scalar expression with *exactly* ``target_md`` multiplicative depth.

    Uses a linear chain strategy (one child carries the chain, the other
    is a short branch with MD ≤ 2) so expression size grows O(target_md)
    instead of O(2^target_md).
    """
    if target_md <= 0:
        return _add_tree(random.randint(0, extra))

    main_md  = target_md - 1
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


def gen_vec(target_md: int, vec_size: int) -> str:
    """
    ``(Vec ...)`` with exactly ``target_md`` multiplicative depth.

    One randomly chosen lane carries the full mult-depth chain.
    Other lanes get lower depths (≤ 4) to keep expression size moderate
    for large vec sizes (16, 25, 32).
    """
    _reset()
    target_idx = random.randint(0, vec_size - 1)
    elems = []
    for i in range(vec_size):
        md = target_md if i == target_idx else random.randint(0, min(target_md, 4))
        elems.append(_scalar_md(md))
    return f"(Vec {' '.join(elems)})"


# =====================================================================
#  Deduplication via canonical token-sequence  (no PyTorch needed)
# =====================================================================

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


def _tok_seq(expr_str: str) -> tuple:
    """Canonical token-ID sequence — same logic as fhe_rl.utils.get_token_sequence."""
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


def _load_existing(path: str) -> set:
    """Load token-sequences of every expression already in the dataset."""
    existing = set()
    if not os.path.exists(path):
        return existing
    with open(path) as fh:
        for line in fh:
            s = line.split(":")[0].strip()
            if not s:
                continue
            try:
                existing.add(_tok_seq(s))
            except Exception:
                continue
    return existing


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate high-noise VecLang expressions for constrained RL training."
    )
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--dataset", type=str, default=DATASET_PATH)
    ap.add_argument("--backup",  action="store_true", default=True,
                    help="Back up the dataset before appending")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── noise estimator ──────────────────────────────────────────────
    print("Loading NoiseEstimator …")
    estimator = NoiseEstimator()

    # ── existing dataset ─────────────────────────────────────────────
    print(f"Loading existing dataset: {args.dataset}")
    existing_tokens = _load_existing(args.dataset)
    n_existing = len(existing_tokens)
    print(f"  {n_existing:,} unique expressions in current dataset")

    # ── backup ───────────────────────────────────────────────────────
    if args.backup and os.path.exists(args.dataset):
        bak = args.dataset + BACKUP_SUFFIX
        if not os.path.exists(bak):
            import shutil
            shutil.copy2(args.dataset, bak)
            print(f"  Backup saved → {bak}")

    # ── plan summary ─────────────────────────────────────────────────
    total_plan = sum(AUGMENTATION_PLAN.values())
    print(f"\nAugmentation plan  ({total_plan:,} expressions):")
    print(f"  {'MD':>4s}  {'Count':>6s}  {'Est. Noise':>10s}  {'Purpose'}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*10}  {'─'*35}")
    purpose = {
        7:  "critical boundary for budget 230",
        8:  "above 230/236 threshold",
        9:  "bridge zone toward 369",
        10: "approaching budget 369",
        11: "critical boundary for budget 369",
        12: "above budget 369",
    }
    for md, cnt in sorted(AUGMENTATION_PLAN.items()):
        est = md * 33.383
        print(f"  {md:4d}  {cnt:6d}  {est:8.0f} b   {purpose.get(md, '')}")

    # ── generation loop ──────────────────────────────────────────────
    generated = []
    new_tokens = set()
    t0 = time.time()

    for target_md, target_count in sorted(AUGMENTATION_PLAN.items()):
        done = 0
        attempts = 0

        while done < target_count:
            attempts += 1
            vs = random.choice(VEC_SIZES)
            expr_str = gen_vec(target_md, vs)

            # 1. must parse
            try:
                expr = parse_sexpr(expr_str)
            except Exception:
                continue

            # 2. verify mult-depth (safety check — should always match)
            if get_multiplicative_depth(expr) != target_md:
                continue

            # 3. vec-size must be in valid set (guaranteed by construction)
            if len(expr.args) not in {1, 4, 8, 9, 16, 25, 32}:
                continue

            # 4. dedup against existing + already-generated
            try:
                ts = _tok_seq(expr_str)
            except Exception:
                continue
            if ts in existing_tokens or ts in new_tokens:
                continue

            new_tokens.add(ts)
            noise = float(estimator.estimate(expr))
            generated.append((expr_str, target_md, noise, vs))
            done += 1

            if done % 500 == 0:
                el = time.time() - t0
                print(f"    MD {target_md:2d}: {done:5d}/{target_count}  ({el:.1f}s)")

        el = time.time() - t0
        print(f"  ✓ MD {target_md:2d}: {done}/{target_count}  "
              f"({attempts} attempts, {el:.1f}s)")

    # ── append to dataset ────────────────────────────────────────────
    with open(args.dataset, "a") as fh:
        for expr_str, _, _, _ in generated:
            fh.write(expr_str + "\n")

    # ── also save generated-only file for reference ──────────────────
    gen_only_path = os.path.join(
        os.path.dirname(args.dataset),
        "generated_high_noise_expressions.txt",
    )
    with open(gen_only_path, "w") as fh:
        for expr_str, md, noise, vs in generated:
            fh.write(f"{expr_str}|||md={md}|noise={noise:.1f}|vs={vs}\n")

    # ── summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*65}")
    print(f"  Generation complete in {elapsed:.1f}s")
    print(f"  New expressions:   {len(generated):,}")
    print(f"  Appended to:       {args.dataset}")
    print(f"  Reference copy:    {gen_only_path}")
    print(f"  Dataset size:      {n_existing:,} → {n_existing + len(generated):,}")
    print(f"{'═'*65}")

    print(f"\nNoise distribution of generated expressions:")
    print(f"  {'MD':>4s}  {'Count':>6s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}  {'Std':>7s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for md in sorted(AUGMENTATION_PLAN.keys()):
        noises = [n for _, m, n, _ in generated if m == md]
        if noises:
            print(f"  {md:4d}  {len(noises):6d}  {np.mean(noises):8.1f}  "
                  f"{min(noises):8.1f}  {max(noises):8.1f}  {np.std(noises):7.1f}")

    budgets = [172, 230, 236, 369, 9_000_000]
    print(f"\nPer-budget constraint analysis (generated expressions only):")
    print(f"  {'Budget':>12s}  {'Constrained':>12s}  {'%':>6s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*6}")
    for b in budgets:
        above = sum(1 for _, _, n, _ in generated if n > b)
        pct = above / len(generated) * 100 if generated else 0
        print(f"  {b:12,d}  {above:8d}/{len(generated):<4d}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
