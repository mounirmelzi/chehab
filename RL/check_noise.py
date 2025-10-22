import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RL.pytrs.noise_estimator import estimate_expression_noise
BUDGET = 100
tests = [
    ("(+ x y)", 1),
    ("(- x y)", 1),
    ("(<< x 1)", 5),
    ("(<< (<< x 3) 2)", 25),
    ("(* x y)", 40),
    ("(* (+ x y) z)", 41),
    ("(* (<< x 4) y)", 60),
    ("(* (+ (<< x 2) y) z)", 51),
]

print(f"Budget={BUDGET}\n")
rows = []
for expr, expect in tests:
    out = estimate_expression_noise(expr, initial_budget_bits=BUDGET)
    used = int(out["noise_used"])
    rem = int(out["remaining_budget"])
    violated = bool(out["violated"])
    ok = (used == expect)
    rows.append((expr, used, expect, rem, violated, ok))
    print(f"{expr:30s} used={used:3d} expect={expect:3d} rem={rem:3d} violated={violated}  [{ 'OK' if ok else 'MISMATCH' }]")

# Simple ordering checks (assertions)
def noise(expr): 
    return next(u for e,u,_,_,_,_ in rows if e==expr)
assert noise("(+ x y)") < noise("(* x y)")
assert noise("(<< x 1)") < noise("(* x y)")
assert noise("(* (+ x y) z)") > noise("(* x y)")
assert noise("(* (<< x 4) y)") > noise("(* (+ (<< x 2) y) z)")

print("\nAll ordering assertions passed.")