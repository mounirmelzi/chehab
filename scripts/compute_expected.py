#!/usr/bin/env python3
"""
Compute the expected plaintext output of a VecLang expression or deep_network circuit.

Used by the instrumented CKKS runner to calculate correct precision metrics.

Usage:
    python3 compute_expected.py veclang "<sexpr>:name"    # from VecLang expression
    python3 compute_expected.py polynomial <depth>         # x^(2^depth)
    python3 compute_expected.py conv <img> <kern> <layers>
    python3 compute_expected.py linear <size> <layers>
"""
from __future__ import annotations
import math
import re
import sys


INPUT_VAL = 0.95


def eval_veclang(expr_str: str, val: float = INPUT_VAL) -> float:
    """Evaluate a VecLang S-expression with all variables set to val.
    Returns the value of the FIRST output slot."""
    tokens = tokenize(expr_str)
    tree, _ = parse_tokens(tokens, 0)
    if isinstance(tree, list) and len(tree) > 0:
        op = tree[0]
        if op == "Vec":
            return eval_node(tree[1], val)
        return eval_node(tree, val)
    return eval_node(tree, val)


def tokenize(s: str) -> list[str]:
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]


def parse_tokens(tokens: list[str], pos: int):
    if pos >= len(tokens):
        return None, pos
    token = tokens[pos]
    if token == "(":
        lst = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            elem, pos = parse_tokens(tokens, pos)
            if elem is not None:
                lst.append(elem)
        pos += 1  # skip ')'
        return lst, pos
    else:
        return token, pos + 1


def eval_node(node, val: float) -> float:
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return val

    if isinstance(node, list):
        if len(node) == 0:
            return 0.0
        op = node[0]
        if op == "+":
            return eval_node(node[1], val) + eval_node(node[2], val)
        elif op == "*":
            return eval_node(node[1], val) * eval_node(node[2], val)
        elif op == "-":
            return eval_node(node[1], val) - eval_node(node[2], val)
        elif op == "Vec":
            return eval_node(node[1], val)
        else:
            return val
    return float(node) if node is not None else 0.0


def expected_polynomial(depth: int, val: float = INPUT_VAL) -> float:
    """x^(2^depth) with x=val."""
    return val ** (2 ** depth)


def expected_conv(img_size: int, kernel_size: int, num_layers: int,
                  val: float = INPUT_VAL) -> float:
    """Simulate the deep_network conv circuit.
    Each layer: conv(all_vals, all_vals) then square.
    encrypt(0) accumulator is assumed folded away by constant folding."""
    pixel_val = val
    for _ in range(num_layers):
        dot = kernel_size * kernel_size * (pixel_val * val)
        pixel_val = dot * dot
    return pixel_val


def expected_linear(size: int, num_layers: int,
                    val: float = INPUT_VAL) -> float:
    """Simulate the deep_network linear circuit.
    Each layer: dot(weights * inputs) then square.
    encrypt(0) accumulator is assumed folded away by constant folding."""
    elem_val = val
    for _ in range(num_layers):
        dot = size * (elem_val * val)
        dot_sq = dot * dot
        elem_val = dot_sq
    return elem_val


def main():
    if len(sys.argv) < 2:
        print("Usage: compute_expected.py <mode> [args...]", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "veclang":
        expr_line = sys.argv[2]
        if ":" in expr_line:
            expr_str = ":".join(expr_line.split(":")[:-1])
        else:
            expr_str = expr_line
        result = eval_veclang(expr_str)
        print(f"{result:.15e}")

    elif mode == "polynomial":
        depth = int(sys.argv[2])
        result = expected_polynomial(depth)
        print(f"{result:.15e}")

    elif mode == "conv":
        img = int(sys.argv[2])
        kern = int(sys.argv[3])
        layers = int(sys.argv[4])
        result = expected_conv(img, kern, layers)
        print(f"{result:.15e}")

    elif mode == "linear":
        size = int(sys.argv[2])
        layers = int(sys.argv[3])
        result = expected_linear(size, layers)
        print(f"{result:.15e}")

    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
