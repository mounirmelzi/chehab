#!/usr/bin/env python3
"""Quick verification of the augmented dataset — no expression printing."""

import os
import sys
import io

DATASET  = "./fhe_rl/datasets/final_llm_dataset.txt"
GEN_FILE = "./fhe_rl/datasets/generated_high_noise_expressions.txt"
BACKUP   = DATASET + ".bak_before_augmentation"

def count_lines(path):
    if not os.path.exists(path):
        return -1
    with open(path) as f:
        return sum(1 for line in f if line.strip())

def main():
    print("=" * 60)
    print("  AUGMENTED DATASET VERIFICATION")
    print("=" * 60)

    # 1. Line counts
    n_dataset = count_lines(DATASET)
    n_backup  = count_lines(BACKUP)
    n_gen     = count_lines(GEN_FILE)

    print(f"\n1. File sizes:")
    print(f"   Dataset:    {n_dataset:,} lines  ({DATASET})")
    print(f"   Backup:     {n_backup:,} lines  ({BACKUP})")
    print(f"   Generated:  {n_gen:,} lines  ({GEN_FILE})")
    if n_backup > 0 and n_dataset > 0:
        print(f"   Added:      {n_dataset - n_backup:,} expressions")

    # 2. Parse generated file metadata (no expression parsing needed)
    if n_gen <= 0:
        print("\n   ERROR: generated file not found or empty!")
        return

    print(f"\n2. Generated expressions metadata (from ||| tags):")
    from collections import Counter
    md_counts = Counter()
    md_noises = {}
    vs_counts = Counter()

    with open(GEN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) < 2:
                continue
            meta = parts[1]
            md_str = [p for p in meta.split("|") if p.startswith("md=")]
            noise_str = [p for p in meta.split("|") if p.startswith("noise=")]
            vs_str = [p for p in meta.split("|") if p.startswith("vs=")]
            if md_str and noise_str:
                md = int(md_str[0].split("=")[1])
                noise = float(noise_str[0].split("=")[1])
                md_counts[md] += 1
                md_noises.setdefault(md, []).append(noise)
            if vs_str:
                vs_counts[int(vs_str[0].split("=")[1])] += 1

    print(f"   {'MD':>4s}  {'Count':>6s}  {'MeanNoise':>10s}  {'Min':>8s}  {'Max':>8s}")
    print(f"   {'─'*4}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")
    for md in sorted(md_counts.keys()):
        noises = md_noises[md]
        mn = sum(noises) / len(noises)
        print(f"   {md:4d}  {md_counts[md]:6d}  {mn:10.1f}  {min(noises):8.1f}  {max(noises):8.1f}")
    print(f"   Total: {sum(md_counts.values())}")

    # 3. Vec size distribution
    print(f"\n3. Vec sizes of generated expressions:")
    for vs in sorted(vs_counts.keys()):
        print(f"   vs={vs:2d}: {vs_counts[vs]}")

    # 4. Per-budget constraint (from metadata, no parsing)
    budgets = [230, 233, 236, 369, 9_000_000]
    all_noises = [n for ns in md_noises.values() for n in ns]
    total = len(all_noises)

    print(f"\n4. Per-budget constraint (generated only):")
    print(f"   {'Budget':>12s}  {'Constrained':>15s}  {'%':>6s}")
    print(f"   {'─'*12}  {'─'*15}  {'─'*6}")
    for b in budgets:
        above = sum(1 for n in all_noises if n > b)
        print(f"   {b:12,d}  {above:7d}/{total:<5d}  {above/total*100:5.1f}%")

    # 5. Spot-check: last 3 lines of dataset match generated
    print(f"\n5. Spot-check: last lines of dataset are generated expressions:")
    with open(DATASET) as f:
        lines = f.readlines()
    with open(GEN_FILE) as f:
        gen_lines = f.readlines()

    last_dataset = [l.strip() for l in lines[-3:]]
    last_gen = [l.split("|||")[0].strip() for l in gen_lines[-3:]]
    match = last_dataset == last_gen
    print(f"   Last 3 lines match: {match}")

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
