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
