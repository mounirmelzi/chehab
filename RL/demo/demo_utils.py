"""Shared formatting, box-drawing, and plotting utilities for PFE demo scripts."""

import time
import sys
import os

# ── ANSI escape codes ───────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
ITALIC  = "\033[3m"
ULINE   = "\033[4m"

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"

BG_GREEN  = "\033[42m"
BG_RED    = "\033[41m"
BG_BLUE   = "\033[44m"
BG_YELLOW = "\033[43m"

CHECK = "✓"
CROSS = "✗"
ARROW = "→"
STAR  = "★"
DIAMOND = "◆"
BULLET  = "●"
TRIANGLE_DOWN = "▼"
TRIANGLE_UP   = "▲"
BAR_FULL  = "█"
BAR_EMPTY = "░"


# ── Box-drawing helpers ─────────────────────────────────────────────────────

def print_banner(title: str, subtitle: str = ""):
    w = 64
    print()
    print(f"  {CYAN}{BOLD}╔{'═' * w}╗{RESET}")
    print(f"  {CYAN}{BOLD}║{RESET}{WHITE}{BOLD}{title:^{w}}{RESET}{CYAN}{BOLD}║{RESET}")
    if subtitle:
        print(f"  {CYAN}{BOLD}║{RESET}{GRAY}{subtitle:^{w}}{RESET}{CYAN}{BOLD}║{RESET}")
    print(f"  {CYAN}{BOLD}╚{'═' * w}╝{RESET}")
    print()


def print_section(title: str):
    w = 62
    pad = w - len(title) - 1
    print(f"\n  {YELLOW}{BOLD}═══ {title} {'═' * max(pad, 0)}{RESET}\n")


def print_subsection(title: str):
    w = 62
    pad = w - len(title) - 1
    print(f"  {GRAY}─── {title} {'─' * max(pad, 0)}{RESET}")


def print_summary_box(title: str, rows: list[tuple[str, str]]):
    """Print a bordered box with key-value rows.

    rows: list of (label, value) pairs
    """
    w = 62
    print(f"\n  {GREEN}{BOLD}┌{'─' * w}┐{RESET}")
    print(f"  {GREEN}{BOLD}│{RESET} {WHITE}{BOLD}{title:<{w - 2}}{RESET} {GREEN}{BOLD}│{RESET}")
    print(f"  {GREEN}{BOLD}├{'─' * w}┤{RESET}")
    for label, value in rows:
        line = f"  {label:<28}{value}"
        print(f"  {GREEN}{BOLD}│{RESET} {line:<{w - 2}} {GREEN}{BOLD}│{RESET}")
    print(f"  {GREEN}{BOLD}└{'─' * w}┘{RESET}")
    print()


def print_kv(label: str, value: str, color: str = WHITE):
    print(f"  {GRAY}{label:<26}{RESET}{color}{BOLD}{value}{RESET}")


# ── Step printer ────────────────────────────────────────────────────────────

def noise_bar(noise: float, budget: float, width: int = 20) -> str:
    if budget <= 0:
        return ""
    ratio = min(noise / budget, 1.5)
    filled = int(ratio * width)
    filled = min(filled, width)
    empty = width - filled
    if ratio > 1.0:
        color = RED
    elif ratio > 0.8:
        color = YELLOW
    else:
        color = GREEN
    pct = ratio * 100
    return f"{color}{BAR_FULL * filled}{GRAY}{BAR_EMPTY * empty}{RESET} {color}{pct:.1f}%{RESET}"


def print_step(step_num: int, max_steps: int, rule_name: str, pos_idx: int,
               old_cost: float, new_cost: float, noise: float, budget: float):
    cost_delta = new_cost - old_cost
    if old_cost > 0:
        cost_pct = (cost_delta / old_cost) * 100
    else:
        cost_pct = 0.0

    if cost_delta < 0:
        cost_color = GREEN
        cost_arrow = TRIANGLE_DOWN
    elif cost_delta > 0:
        cost_color = RED
        cost_arrow = TRIANGLE_UP
    else:
        cost_color = GRAY
        cost_arrow = "="

    violated = noise > budget and budget < 1_000_000
    status_str = f"{RED}{CROSS} VIOLATED{RESET}" if violated else f"{GREEN}{CHECK} Within budget{RESET}"

    bar = noise_bar(noise, budget) if budget < 1_000_000 else f"{GREEN}unconstrained{RESET}"

    print(f"\n  {CYAN}─── Step {step_num}/{max_steps} {'─' * 48}{RESET}")
    print(f"    {BOLD}Rule:{RESET} {MAGENTA}{rule_name:<24}{RESET} {BOLD}Position:{RESET} {BLUE}{pos_idx}{RESET}")
    print(f"    {BOLD}Cost:{RESET} {GRAY}{old_cost}{RESET} {ARROW} {cost_color}{BOLD}{new_cost}{RESET}  ({cost_color}{cost_pct:+.1f}%{RESET})  {cost_color}{cost_arrow}{RESET}")
    print(f"    {BOLD}Noise:{RESET} {YELLOW}{noise:.1f}{RESET} bits   {BOLD}Budget:{RESET} {YELLOW}{budget}{RESET} bits  {bar}")
    print(f"    {BOLD}Status:{RESET} {status_str}")


# ── Phase printer (compilation/execution) ───────────────────────────────────

def print_phase(index: int, total: int, name: str, time_ms: float, success: bool = True):
    status = f"{GREEN}{CHECK}{RESET}" if success else f"{RED}{CROSS}{RESET}"
    dots = "." * max(1, 30 - len(name))
    print(f"    [{index}/{total}] {name} {GRAY}{dots}{RESET} {CYAN}{time_ms:.1f} ms{RESET}  {status}")


def print_exec_phase(name: str, time_ms: float, success: bool = True):
    status = f"{GREEN}{CHECK}{RESET}" if success else f"{RED}{CROSS}{RESET}"
    dots = "." * max(1, 24 - len(name))
    print(f"    {name} {GRAY}{dots}{RESET} {CYAN}{time_ms:.0f} ms{RESET}  {status}")


# ── Progress / waiting ──────────────────────────────────────────────────────

def print_waiting(msg: str):
    print(f"  {GRAY}{BULLET} {msg}...{RESET}", end="", flush=True)


def print_done(extra: str = ""):
    print(f" {GREEN}{CHECK}{RESET} {extra}")


# ── Trajectory plot ─────────────────────────────────────────────────────────

def generate_trajectory_plot(costs: list, noises: list, budget: float,
                             rollback_step: int, output_path: str,
                             title: str = "RL Optimization Trajectory"):
    """Generate a dual-axis trajectory plot: cost + noise over steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  {YELLOW}matplotlib not available, skipping plot{RESET}")
        return

    steps = list(range(len(costs)))
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # Cost axis (left)
    ax1.set_facecolor("#16213e")
    color_cost = "#66BB6A"
    ax1.set_xlabel("Step", fontsize=12, color="white", fontweight="bold")
    ax1.set_ylabel("Cost", fontsize=12, color=color_cost, fontweight="bold")
    ax1.plot(steps, costs, color=color_cost, linewidth=2, marker="o", markersize=3, label="Cost")
    ax1.tick_params(axis="y", labelcolor=color_cost, colors="white")
    ax1.tick_params(axis="x", colors="white")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax1.spines[s].set_color("white")

    # Noise axis (right)
    ax2 = ax1.twinx()
    color_noise = "#EF5350"
    ax2.set_ylabel("Noise (bits)", fontsize=12, color=color_noise, fontweight="bold")
    ax2.plot(steps, noises, color=color_noise, linewidth=2, marker="s", markersize=3, label="Noise")
    ax2.tick_params(axis="y", labelcolor=color_noise, colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(color_noise)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_color("white")

    # Budget line
    if budget < 1_000_000:
        ax2.axhline(y=budget, color="#FFD600", linestyle="--", linewidth=2, alpha=0.8, label=f"Budget = {budget}")

    # Safety rollback marker
    if 0 <= rollback_step < len(costs):
        ax1.axvline(x=rollback_step, color="#42A5F5", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.plot(rollback_step, costs[rollback_step], marker="*", color="#FFD600",
                 markersize=18, zorder=10, markeredgecolor="white", markeredgewidth=1)
        ax2.plot(rollback_step, noises[rollback_step], marker="*", color="#FFD600",
                 markersize=18, zorder=10, markeredgecolor="white", markeredgewidth=1)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               facecolor="#2a2a4a", edgecolor="white", labelcolor="white", fontsize=10)

    ax1.set_title(title, fontsize=14, color="white", fontweight="bold", pad=15)
    ax1.grid(axis="both", alpha=0.15, color="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {GREEN}{CHECK}{RESET} Trajectory plot saved to: {CYAN}{output_path}{RESET}")


# ── Network diagram (text-based) ───────────────────────────────────────────

def print_network_diagram(layers: list[dict]):
    """Print a text-based neural network diagram.

    layers: list of dicts with keys 'name', 'type'
    """
    print(f"\n  {BOLD}{WHITE}Network Architecture:{RESET}\n")
    print(f"    {CYAN}{BOLD}┌──────────┐{RESET}")
    print(f"    {CYAN}{BOLD}│  Input   │{RESET}")
    print(f"    {CYAN}{BOLD}└────┬─────┘{RESET}")
    for i, layer in enumerate(layers):
        print(f"         {GRAY}│{RESET}")
        print(f"         {GRAY}▼{RESET}")
        color = MAGENTA if "act" in layer["type"].lower() else BLUE
        print(f"    {color}{BOLD}┌──────────────────────┐{RESET}")
        print(f"    {color}{BOLD}│{RESET} {WHITE}Layer {i + 1}: {layer['name']:<13}{RESET}{color}{BOLD}│{RESET}")
        print(f"    {color}{BOLD}└──────────┬───────────┘{RESET}")
    print(f"         {GRAY}│{RESET}")
    print(f"         {GRAY}▼{RESET}")
    print(f"    {GREEN}{BOLD}┌──────────┐{RESET}")
    print(f"    {GREEN}{BOLD}│  Output  │{RESET}")
    print(f"    {GREEN}{BOLD}└──────────┘{RESET}")
    print()


# ── Misc ────────────────────────────────────────────────────────────────────

def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def sleep_visual(seconds: float, msg: str = ""):
    """Sleep with a visible countdown (for pacing the demo)."""
    if msg:
        print(f"  {GRAY}{msg}{RESET}", end="", flush=True)
    time.sleep(seconds)
    if msg:
        print()


# ── Artifact saving ─────────────────────────────────────────────────────────

from pathlib import Path
import re
import shutil

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def _out_dir(demo_name: str) -> Path:
    d = _PROJECT_ROOT / "demo_outputs" / demo_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_input_expression(demo: str, expression: str, name: str = "",
                          budget: int | None = None, prefix: str = ""):
    p = _out_dir(demo) / f"{prefix}step1_input_expression.txt"
    with open(p, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  STEP 1 — INPUT VECLANG EXPRESSION\n")
        f.write(f"{'='*70}\n\n")
        if name:
            f.write(f"Operator:  {name}\n")
        if budget is not None:
            bstr = f"{budget} bits" if budget < 1_000_000 else "9M (unconstrained)"
            f.write(f"Budget:    {bstr}\n")
        f.write(f"Length:    {len(expression)} chars, {len(expression.split())} AST nodes\n\n")
        f.write(f"Expression:\n{expression}\n")
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{p.relative_to(_PROJECT_ROOT)}{RESET}")


def save_rl_trajectory(demo: str, trajectory: list[dict], budget: int,
                       rl_time: float, prefix: str = ""):
    p = _out_dir(demo) / f"{prefix}step2_rl_trajectory.txt"
    with open(p, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  STEP 2 — RL OPTIMIZATION TRAJECTORY\n")
        f.write(f"{'='*70}\n\n")
        bstr = f"{budget} bits" if budget < 1_000_000 else "9M (unconstrained)"
        f.write(f"Budget:       {bstr}\n")
        f.write(f"Total steps:  {len(trajectory)-1}\n")
        f.write(f"RL time:      {rl_time:.2f} s\n\n")
        f.write(f"{'Step':>5}  {'Rule':<26} {'Cost':>10} {'Noise':>10}  Status\n")
        f.write(f"{'-'*5}  {'-'*26} {'-'*10} {'-'*10}  {'-'*15}\n")
        for cp in trajectory:
            rule = cp.get("rule", "—")
            cost = cp["cost"]
            noise = cp["noise"]
            status = "VIOLATED" if noise > budget and budget < 1_000_000 else "ok"
            f.write(f"{cp['step']:>5}  {rule:<26} {cost:>10} {noise:>10.1f}  {status}\n")
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{p.relative_to(_PROJECT_ROOT)}{RESET}")


def save_safety_rollback(demo: str, trajectory: list[dict], budget: int,
                         best_step: int, safety_activated: bool, prefix: str = ""):
    p = _out_dir(demo) / f"{prefix}step3_safety_rollback.txt"
    best = next((cp for cp in trajectory if cp["step"] == best_step), trajectory[0])
    agent_final = trajectory[-1]
    with open(p, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  STEP 3 — SAFETY ROLLBACK DECISION\n")
        f.write(f"{'='*70}\n\n")
        bstr = f"{budget} bits" if budget < 1_000_000 else "9M"
        f.write(f"Budget:                {bstr}\n")
        valid = [cp for cp in trajectory if cp["noise"] <= budget or budget >= 1_000_000]
        f.write(f"Valid checkpoints:     {len(valid)} / {len(trajectory)}\n")
        f.write(f"Safety activated:      {'YES' if safety_activated else 'NO'}\n\n")
        f.write(f"Agent's final state (step {agent_final['step']}):\n")
        f.write(f"  Cost:  {agent_final['cost']}\n")
        f.write(f"  Noise: {agent_final['noise']:.1f} bits\n")
        agent_viol = "VIOLATED" if agent_final["noise"] > budget and budget < 1_000_000 else "WITHIN BUDGET"
        f.write(f"  Status: {agent_viol}\n\n")
        f.write(f"Selected state (step {best['step']}):\n")
        f.write(f"  Cost:  {best['cost']}\n")
        f.write(f"  Noise: {best['noise']:.1f} bits\n")
        sel_viol = "VIOLATED" if best["noise"] > budget and budget < 1_000_000 else "WITHIN BUDGET"
        f.write(f"  Status: {sel_viol}\n\n")
        if safety_activated:
            f.write(f"Rollback: stepped back from step {agent_final['step']} to step {best['step']}\n")
        else:
            f.write(f"No rollback needed — agent's final state is already the best valid checkpoint.\n")
        f.write(f"\nOptimized expression:\n{best.get('expression', best.get('expr', 'N/A'))}\n")
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{p.relative_to(_PROJECT_ROOT)}{RESET}")


def save_lattigo_code(demo: str, go_path, prefix: str = ""):
    """Copy generated Go code and extract CKKS parameters from it."""
    go_path = Path(go_path)
    if not go_path.exists():
        return
    out = _out_dir(demo)
    dst = out / f"{prefix}step4_lattigo_code.go"
    shutil.copy2(go_path, dst)
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{dst.relative_to(_PROJECT_ROOT)}{RESET}")

    code = go_path.read_text()
    params_file = out / f"{prefix}step4_ckks_params.txt"
    with open(params_file, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  STEP 4 — CKKS PARAMETERS & COMPILATION\n")
        f.write(f"{'='*70}\n\n")

        m = re.search(r'LogN:\s*(\d+)', code)
        if m:
            logn = int(m.group(1))
            f.write(f"Ring dimension:    N = 2^{logn} = {2**logn}\n")

        m = re.search(r'LogQ:\s*\[([^\]]+)\]', code)
        if m:
            logq = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
            f.write(f"LogQ (modulus):    {logq}\n")
            f.write(f"  Levels:          {len(logq)}\n")
            f.write(f"  Total bits:      {sum(logq)}\n")

        m = re.search(r'LogP:\s*\[([^\]]+)\]', code)
        if m:
            logp = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
            f.write(f"LogP (special):    {logp}\n")
            f.write(f"  Total P bits:    {sum(logp)}\n")

        m = re.search(r'LogDefaultScale:\s*(\d+)', code)
        if m:
            f.write(f"Default scale:     2^{m.group(1)}\n")

        boot_count = len(re.findall(r'bootstrapper\.Bootstrap', code))
        f.write(f"\nBootstrap ops:     {boot_count}\n")
        if boot_count > 0:
            f.write(f"  (bootstrapper.Bootstrap calls found in generated code)\n")

        rescale_count = len(re.findall(r'\.Rescale\(', code))
        modswitch_count = len(re.findall(r'\.ModSwitch\(|ModSwitchDown', code, re.IGNORECASE))
        f.write(f"Rescale ops:       {rescale_count}\n")
        f.write(f"ModSwitch ops:     {modswitch_count}\n")

        mul_count = len(re.findall(r'MulRelinNew|MulNew', code))
        add_count = len(re.findall(r'AddNew', code))
        sub_count = len(re.findall(r'SubNew', code))
        rot_count = len(re.findall(r'RotateNew', code))
        f.write(f"\nHE operations in generated code:\n")
        f.write(f"  Multiplications: {mul_count}\n")
        f.write(f"  Additions:       {add_count}\n")
        f.write(f"  Subtractions:    {sub_count}\n")
        f.write(f"  Rotations:       {rot_count}\n")

    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{params_file.relative_to(_PROJECT_ROOT)}{RESET}")


def save_execution_output(demo: str, results: dict, prefix: str = "",
                          op_name: str = "", input_val: float = 0.95,
                          expected_output: float | None = None):
    p = _out_dir(demo) / f"{prefix}step5_execution_output.txt"
    with open(p, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  STEP 5 — FHE EXECUTION OUTPUT (Lattigo CKKS)\n")
        f.write(f"{'='*70}\n\n")
        if op_name:
            f.write(f"Operator:       {op_name}\n")
        f.write(f"Input value:    x = {input_val}  (all cipher slots set to this value)\n")
        if expected_output is not None:
            f.write(f"Expected output (plaintext): {expected_output:.6f}\n")
        abs_err = results.get('abs_error', None)
        if expected_output is not None and abs_err is not None:
            decrypted = expected_output + abs_err if abs_err < 1e10 else None
            if decrypted is not None:
                f.write(f"Decrypted output (CKKS):     {decrypted:.6f}\n")
                f.write(f"Absolute error:              {abs_err:.2e}\n")
        elif abs_err is not None:
            f.write(f"Absolute error:              {abs_err:.2e}\n")
        f.write(f"\n{'Phase':<28} {'Time':>10}\n")
        f.write(f"{'-'*28} {'-'*10}\n")
        f.write(f"{'KeyGen':<28} {results.get('keygen_ms', 0):>8.0f} ms\n")
        btp = results.get('bootstrap_keygen_ms', 0)
        if btp > 0:
            f.write(f"{'Bootstrap KeyGen':<28} {btp:>8.0f} ms\n")
        f.write(f"{'Encrypt':<28} {results.get('encrypt_ms', 0):>8.0f} ms\n")
        f.write(f"{'Eval (homomorphic)':<28} {results.get('eval_ms', 0):>8.0f} ms\n")
        f.write(f"{'Decrypt':<28} {results.get('decrypt_ms', 0):>8.0f} ms\n")
        f.write(f"{'-'*28} {'-'*10}\n")
        f.write(f"{'TOTAL':<28} {results.get('total_ms', 0):>8.0f} ms\n\n")
        prec = results.get('precision_bits', 0)
        f.write(f"Precision:  {prec:.1f} bits\n")
        f.write(f"Status:     {'PASS (>= 10 bits)' if prec >= 10 else 'FAIL (< 10 bits)'}\n")
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{p.relative_to(_PROJECT_ROOT)}{RESET}")


def save_comparison_summary(demo: str, unc_result: dict, con_result: dict,
                            budget: int, unc_label: str = "Unconstrained",
                            con_label: str = "Constrained"):
    p = _out_dir(demo) / "comparison_summary.txt"
    with open(p, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  AGENT COMPARISON SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Budget: {budget} bits\n\n")
        f.write(f"{'Metric':<28} {unc_label:>18}   {con_label:>18}\n")
        f.write(f"{'-'*28} {'-'*18}   {'-'*18}\n")
        f.write(f"{'Initial cost':<28} {str(unc_result['initial_cost']):>18}   {str(con_result['initial_cost']):>18}\n")
        f.write(f"{'Final cost':<28} {str(unc_result['final_cost']):>18}   {str(con_result['final_cost']):>18}\n")
        f.write(f"{'Cost reduction':<28} {unc_result['cost_reduction']:>17.1f}%   {con_result['cost_reduction']:>17.1f}%\n")
        f.write(f"{'-'*28} {'-'*18}   {'-'*18}\n")
        f.write(f"{'Initial noise':<28} {unc_result['initial_noise']:>15.1f} b   {con_result['initial_noise']:>15.1f} b\n")
        f.write(f"{'Final noise':<28} {unc_result['final_noise']:>15.1f} b   {con_result['final_noise']:>15.1f} b\n")
        f.write(f"{'Budget':<28} {budget:>15} b   {budget:>15} b\n")
        unc_st = "VIOLATED" if unc_result['violated'] else "WITHIN"
        con_st = "VIOLATED" if con_result['violated'] else "WITHIN"
        f.write(f"{'Status':<28} {unc_st:>18}   {con_st:>18}\n")
        f.write(f"{'-'*28} {'-'*18}   {'-'*18}\n")
        f.write(f"{'Steps':<28} {unc_result['steps']:>18}   {con_result['steps']:>18}\n")
        f.write(f"{'Time':<28} {unc_result['time']:>16.1f} s   {con_result['time']:>16.1f} s\n")
    print(f"  {GREEN}{CHECK}{RESET} Artifact saved: {CYAN}{p.relative_to(_PROJECT_ROOT)}{RESET}")
