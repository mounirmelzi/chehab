#!/usr/bin/env python3
"""
Augment the test benchmark set (benchmarks.txt) with high-noise expressions.

The original 43 benchmarks have max noise ~217 bits, which means V4 agents
(trained on budgets 230-372) are never truly challenged.  This script adds
27 synthetic expressions at MultDepth 7-12 to stress those budget boundaries.

Runs from the RL/ directory with pytrs on PYTHONPATH:
    cd $PROJ_ROOT/RL
    python ../scripts/augment_benchmarks.py
"""

import os
import sys
import random
import time
import shutil

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
#  BENCHMARK AUGMENTATION PLAN
# =====================================================================
# Target: stress budgets 230, 233, 236, 369, 372
#
#   MD 7  →  noise ~248  →  stresses budgets 230-236
#   MD 8  →  noise ~281  →  above 236
#   MD 9  →  noise ~315  →  bridge zone
#   MD 10 →  noise ~348  →  approaching 369
#   MD 11 →  noise ~381  →  stresses budget 369-372
#   MD 12 →  noise ~415  →  above 372
# =====================================================================

AUGMENTATION_PLAN = {
    7:  5,
    8:  5,
    9:  4,
    10: 3,
    11: 5,
    12: 5,
}

VEC_SIZES = [1, 4, 8, 9, 16, 25, 32]
BENCHMARKS_PATH = "./fhe_rl/datasets/benchmarks.txt"
BACKUP_SUFFIX = ".bak_before_augmentation"


# =====================================================================
#  Expression generators (same as generate_high_noise_dataset.py)
# =====================================================================
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

def gen_vec(target_md, vec_size):
    _reset()
    target_idx = random.randint(0, vec_size - 1)
    elems = []
    for i in range(vec_size):
        md = target_md if i == target_idx else random.randint(0, min(target_md, 4))
        elems.append(_scalar_md(md))
    return f"(Vec {' '.join(elems)})"


# =====================================================================
#  Dedup via canonical token-sequence
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


def _load_existing(path):
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
#  VERIFICATION — validates EVERY expression in the final file
# =====================================================================
def verify_benchmarks(path, estimator):
    """Full validation pass on the benchmark file. Returns True if OK."""
    print(f"\n{'='*65}")
    print(f"  VERIFICATION — validating all expressions in {os.path.basename(path)}")
    print(f"{'='*65}")

    errors = []
    stats = []
    valid_vs = {1, 4, 8, 9, 16, 25, 32}

    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    print(f"  Total lines: {len(lines)}")

    for i, line in enumerate(lines, 1):
        expr_str = line.split(":")[0].strip()
        name = line.split(":")[-1].strip() if ":" in line else f"expr_{i}"

        # 1. Parse
        try:
            expr = parse_sexpr(expr_str)
        except Exception as e:
            errors.append(f"  Line {i} [{name}]: PARSE ERROR — {e}")
            continue

        # 2. Vec size
        vs = len(expr.args)
        if vs not in valid_vs:
            errors.append(f"  Line {i} [{name}]: INVALID vec_size={vs}")
            continue

        # 3. Mult depth + noise
        md = get_multiplicative_depth(expr)
        noise = float(estimator.estimate(expr))
        stats.append((i, name, vs, md, noise))

    # Report errors
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(e)
    else:
        print(f"  All {len(lines)} expressions parsed and validated OK")

    # Noise distribution
    noises = [s[4] for s in stats]
    mds    = [s[3] for s in stats]

    print(f"\n  Noise distribution:")
    print(f"    Min:    {min(noises):.1f}")
    print(f"    Max:    {max(noises):.1f}")
    print(f"    Mean:   {np.mean(noises):.1f}")
    print(f"    Median: {np.median(noises):.1f}")
    print(f"    Std:    {np.std(noises):.1f}")

    # Per MultDepth
    print(f"\n  Per multiplicative depth:")
    print(f"    {'MD':>4s}  {'Count':>6s}  {'Mean Noise':>10s}  {'Min':>8s}  {'Max':>8s}")
    print(f"    {'─'*4}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")
    for md_val in sorted(set(mds)):
        md_noises = [s[4] for s in stats if s[3] == md_val]
        print(f"    {md_val:4d}  {len(md_noises):6d}  {np.mean(md_noises):10.1f}  "
              f"{min(md_noises):8.1f}  {max(md_noises):8.1f}")

    # Budget constraint activation
    budgets = [172, 230, 233, 236, 369, 372, 9_000_000]
    print(f"\n  Budget constraint activation (expressions with noise > budget):")
    print(f"    {'Budget':>12s}  {'Count':>8s}  {'%':>6s}")
    print(f"    {'─'*12}  {'─'*8}  {'─'*6}")
    for b in budgets:
        above = sum(1 for n in noises if n > b)
        pct = above / len(noises) * 100
        print(f"    {b:12,d}  {above:6d}/{len(noises):<4d}  {pct:5.1f}%")

    # High-noise coverage check
    high_noise_count = sum(1 for n in noises if n > 230)
    print(f"\n  High-noise expressions (noise > 230): {high_noise_count}/{len(noises)}")

    ok = len(errors) == 0 and high_noise_count >= 20
    if ok:
        print(f"\n  VERIFICATION PASSED — dataset is ready for constrained agent testing")
    else:
        if high_noise_count < 20:
            print(f"\n  WARNING — only {high_noise_count} high-noise expressions "
                  f"(need >= 20 for meaningful constraint testing)")
        if errors:
            print(f"\n  VERIFICATION FAILED — {len(errors)} errors found")

    print(f"{'='*65}")
    return ok


# =====================================================================
#  Main
# =====================================================================
def main():
    random.seed(2025)
    np.random.seed(2025)

    print("Loading NoiseEstimator ...")
    estimator = NoiseEstimator()

    # ── Check current benchmarks ─────────────────────────────────────
    if not os.path.exists(BENCHMARKS_PATH):
        print(f"ERROR: benchmarks.txt not found at {BENCHMARKS_PATH}")
        sys.exit(1)

    with open(BENCHMARKS_PATH) as fh:
        original_lines = [l.strip() for l in fh if l.strip()]
    n_original = len(original_lines)
    print(f"Original benchmarks: {n_original} expressions")

    # ── Check if already augmented ───────────────────────────────────
    has_synth = any(":synth_md" in l for l in original_lines)
    if has_synth:
        n_synth = sum(1 for l in original_lines if ":synth_md" in l)
        print(f"Benchmarks already contain {n_synth} synthetic expressions.")
        print("Running verification on current file ...")
        ok = verify_benchmarks(BENCHMARKS_PATH, estimator)
        sys.exit(0 if ok else 1)

    # ── Backup ───────────────────────────────────────────────────────
    bak = BENCHMARKS_PATH + BACKUP_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(BENCHMARKS_PATH, bak)
        print(f"Backup saved -> {bak}")
    else:
        print(f"Backup already exists: {bak}")

    # ── Load existing tokens for dedup ───────────────────────────────
    existing_tokens = _load_existing(BENCHMARKS_PATH)
    print(f"Loaded {len(existing_tokens)} existing token sequences for dedup")

    # ── Plan summary ─────────────────────────────────────────────────
    total_plan = sum(AUGMENTATION_PLAN.values())
    print(f"\nAugmentation plan ({total_plan} expressions):")
    print(f"  {'MD':>4s}  {'Count':>6s}  {'Est. Noise':>10s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*10}")
    for md, cnt in sorted(AUGMENTATION_PLAN.items()):
        est = md * 33.383
        print(f"  {md:4d}  {cnt:6d}  {est:8.0f} b")

    # ── Generate ─────────────────────────────────────────────────────
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

            try:
                expr = parse_sexpr(expr_str)
            except Exception:
                continue

            if get_multiplicative_depth(expr) != target_md:
                continue
            if len(expr.args) not in {1, 4, 8, 9, 16, 25, 32}:
                continue

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

        el = time.time() - t0
        noises_md = [n for _, m, n, _ in generated if m == target_md]
        print(f"  MD {target_md:2d}: {done}/{target_count} generated "
              f"(noise mean={np.mean(noises_md):.0f}, "
              f"range={min(noises_md):.0f}-{max(noises_md):.0f}) "
              f"[{attempts} attempts, {el:.1f}s]")

    # ── Append to benchmarks ─────────────────────────────────────────
    with open(BENCHMARKS_PATH, "a") as fh:
        for expr_str, md, noise, vs in generated:
            name = f"synth_md{md}_vs{vs}_n{int(noise)}"
            fh.write(f"{expr_str}:{name}\n")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Augmentation complete in {elapsed:.1f}s")
    print(f"  New expressions:   {len(generated)}")
    print(f"  Benchmarks size:   {n_original} -> {n_original + len(generated)}")
    print(f"{'='*65}")

    # ── Full verification pass ───────────────────────────────────────
    ok = verify_benchmarks(BENCHMARKS_PATH, estimator)

    if ok:
        print("\nAugmented benchmark file is READY for testing.")
    else:
        print("\nWARNING: Verification had issues — check output above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
