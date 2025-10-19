# RL/pytrs/noise_estimator.py
from typing import Union
from .expr import Expr, Op, Const, Var
from .parser import parse_sexpr
from pathlib import Path

NOISE_CONSUMPTION = {
    "encrypt": 8,
    "mul": 40,
    "mul_plain": 31,
    "add": 1,
    "add_plain": 0,
    "rotate": 5,     
    "relin": 0,
    "VecMul": 40,
    "VecAdd": 1,
    "VecMinus": 1,
    "VecNeg": 1,
}

def _op_to_noise_key(op: str) -> str:
    if op in ("+", "Add"): return "add"
    if op in ("-", "Minus", "Neg"): return "add"  
    if op in ("*", "Mul"): return "mul"
    if op == "VecAdd": return "VecAdd"
    if op == "VecMinus": return "VecMinus"
    if op == "VecNeg": return "VecNeg"
    if op == "VecMul": return "VecMul"
    return ""  

def estimate_expression_noise(expr: Union[str, Expr],
                              initial_budget_bits: int = 369,
                              noise_table: dict = NOISE_CONSUMPTION) -> dict:
    """
    Longest-path noise estimate (professor’s model):
      - noise(node) = base_noise(op) + max(noise(children))
      - rotations '<< k' add k * rotate_noise and then take max(child)
    Returns: { "noise_used": int, "remaining_budget": int, "violated": bool }
    """
    if isinstance(expr, str):
        expr = parse_sexpr(expr)

    rotate_noise = noise_table.get("rotate", 5)

    def dfs(node) -> int:
        """Duck-typed traversal to avoid class identity issues across packages."""
        if hasattr(node, "value") and not hasattr(node, "op"):
            return 0
        if hasattr(node, "name") and not hasattr(node, "op"):
            return 0

        if hasattr(node, "op"):
            op = getattr(node, "op")
            args = list(getattr(node, "args", []) or [])

            if op == "<<":
                step = 0
                if len(args) == 2 and hasattr(args[1], "value"):
                    try:
                        step = int(getattr(args[1], "value"))
                    except Exception:
                        step = 0
                child_noise = dfs(args[0]) if args else 0
                return step * rotate_noise + child_noise

            base_key = _op_to_noise_key(op)
            base_noise = noise_table.get(base_key, 0)

            if not args:
                return base_noise
            child_noises = [dfs(arg) for arg in args]
            return base_noise + (max(child_noises) if child_noises else 0)

        if isinstance(node, (list, tuple)):
            child_noises = [dfs(x) for x in node]
            return max(child_noises) if child_noises else 0

        return 0

    noise_used = dfs(expr)
    remaining = max(0, initial_budget_bits - noise_used)
    return {
        "noise_used": noise_used,
        "remaining_budget": remaining,
        "violated": remaining == 0 and noise_used > initial_budget_bits,
    }


def test_noise_on_file(file_path: str, initial_budget_bits: int = 369) -> None:
    """
    Quick test runner: reads one expression per line from file_path,
    estimates noise using the longest-path model, and prints per-line stats.
    """
    p = Path(file_path)
    if not p.exists():
        print(f"ERROR: file not found: {file_path}")
        return

    with p.open("r") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                stats = estimate_expression_noise(s, initial_budget_bits=initial_budget_bits)
                noise_used = int(stats.get("noise_used", 0))
                remaining = int(stats.get("remaining_budget", max(0, initial_budget_bits - noise_used)))
                print(f"{p.name}:{idx} noise_used={noise_used} remaining_budget={remaining}", flush=True)
            except Exception as e:
                print(f"{p.name}:{idx} ERROR: {e}", flush=True)


if __name__ == "__main__":
    import sys
    rl_dir = Path(__file__).resolve().parents[1]
    default_path = rl_dir / "fhe_rl" / "datasets" / "data50.txt"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    test_noise_on_file(str(path))