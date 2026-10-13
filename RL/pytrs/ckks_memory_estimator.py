"""
Closed-form peak-memory estimator for CKKS programs compiled by CHEHAB.

    peak = (1 + G) * SWK                                   keys
         + sum over live ciphertexts of 2 * N * (level + 1) * 8   ciphertexts
         + (3 + a + c * L + r * [G > 0]) * N * (L + K) * 8        pk+sk + library overhead

    SWK  = dnum * 2 * N * (L + K) * 8,  dnum = ceil(L / K)   one key-switching key
    N    ring dimension, L number of Q primes, K number of P primes
    G    rotation keys: distinct rotation steps (+1 on HEonGPU, which stores an extra key)

a, c, r were fitted on HEonGPU (NVIDIA A100) and Lattigo v5.0.7 measurements with
N in {8192, 16384}; held-out error is ~1-2% on synthetic workloads up to N = 65536
and 64 rotation keys, and 2.3% (GPU) / 1.6% (CPU) on real CHEHAB DNN programs.
Bootstrapping keys and matrices are not modelled.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from expr import Expr, Const, Var, Op
from parser import parse_sexpr
from cost import get_multiplicative_depth

WORD_BYTES = 8
MB = 1024 * 1024

# (a, c, r) in units of N * (L + K) * 8 bytes
OVERHEAD_COEFFS = {
    "gpu": (8.01788211, 1.13505002, 5.97810219),
    "cpu": (26.50443709, 0.72034038, 1.49652300),
}

# Mirrors fheco::dsl::Compiler + CKKSParamSelector::default_params.
MAX_DEPTH_WITHOUT_BOOTSTRAP = 12
FIRST_Q_BITS, Q_BITS, P_BITS = 55, 40, (45, 45)
LOG_N_BY_TOTAL_BITS = ((109, 12), (218, 13), (438, 14), (881, 15))


@dataclass
class CKKSParams:
    N: int
    L: int
    K: int
    bootstrap_required: bool = False


@dataclass
class MemoryEstimate:
    params: CKKSParams
    rotation_steps: int
    galois_keys: int
    peak_live_ciphertexts: int
    terms: Dict[str, float] = field(default_factory=dict)

    @property
    def total_bytes(self) -> float:
        return sum(self.terms.values())

    @property
    def total_mb(self) -> float:
        return self.total_bytes / MB


def chehab_params(mult_depth: int) -> CKKSParams:
    """CKKS parameters CHEHAB selects for a circuit of the given multiplicative depth."""
    bootstrap = mult_depth > MAX_DEPTH_WITHOUT_BOOTSTRAP
    levels = MAX_DEPTH_WITHOUT_BOOTSTRAP if bootstrap else max(mult_depth + 1, 3)
    total_bits = FIRST_Q_BITS + Q_BITS * levels + sum(P_BITS)
    log_n = next((n for bits, n in LOG_N_BY_TOTAL_BITS if total_bits <= bits), 16)
    if bootstrap:
        log_n = 16
    return CKKSParams(N=1 << log_n, L=levels + 1, K=len(P_BITS), bootstrap_required=bootstrap)


def switching_key_bytes(N: int, L: int, K: int) -> int:
    return math.ceil(L / K) * 2 * N * (L + K) * WORD_BYTES


def ciphertext_bytes(N: int, level: int) -> int:
    return 2 * N * (level + 1) * WORD_BYTES


def estimate_peak_memory(
    N: int,
    L: int,
    K: int,
    rotation_steps: int,
    live_ciphertext_bytes: float,
    backend: str = "gpu",
) -> Dict[str, float]:
    """Formula terms in bytes, given parameters and program statistics."""
    a, c, r = OVERHEAD_COEFFS[backend]
    unit = N * (L + K) * WORD_BYTES
    swk = switching_key_bytes(N, L, K)
    galois = rotation_steps + 1 if (backend == "gpu" and rotation_steps > 0) else rotation_steps
    return {
        "relinearization_key": swk,
        "galois_keys": galois * swk,
        "ciphertexts": live_ciphertext_bytes,
        "public_secret_key": 3 * unit,
        "context_evaluator": a * unit,
        "key_switching_scratch": (c * L + r * (rotation_steps > 0)) * unit,
    }


class _Liveness:
    """Static liveness over the expression DAG in CHEHAB's code-gen order:
    all inputs are encrypted first, then operations run in post-order.
    Identical subtrees are computed once. Constants are plaintexts and not counted."""

    def __init__(self, expr: Expr, top_level: int):
        self.top = top_level
        self.nodes = {}       # key -> (kind, child keys, level)
        self.inputs = []
        self.ops = []
        self.rotations = set()
        self.root = self._visit(expr)

    def _visit(self, node: Expr):
        if isinstance(node, Const):
            return None
        key = str(node)
        if key in self.nodes:
            return key
        if isinstance(node, Var):
            self.nodes[key] = ([], self.top)
            self.inputs.append(key)
            return key
        if node.op == "Vec" and all(not isinstance(a, Op) for a in node.args):
            if all(isinstance(a, Const) for a in node.args):
                return None
            self.nodes[key] = ([], self.top)
            self.inputs.append(key)
            return key
        children = [self._visit(a) for a in node.args]
        cts = [ch for ch in children if ch is not None]
        if not cts:
            return None
        level = min(self.nodes[ch][1] for ch in cts)
        if node.op in ("*", "VecMul"):
            level -= 1
        if node.op == "<<" and isinstance(node.args[1], Const) and node.args[1].value != 0:
            self.rotations.add(node.args[1].value)
        self.nodes[key] = (cts, level)
        self.ops.append(key)
        return key

    def peak(self, N: int, level_aware: bool):
        order = self.inputs + self.ops
        last_use = {k: i for i, k in enumerate(order)}
        for i, k in enumerate(order):
            for ch in self.nodes[k][0]:
                last_use[ch] = i
        size = lambda k: ciphertext_bytes(N, max(self.nodes[k][1], 0) if level_aware else self.top)
        live, live_bytes, peak_cts, peak_bytes = set(), 0, 0, 0
        for i, k in enumerate(order):
            live.add(k)
            live_bytes += size(k)
            peak_cts, peak_bytes = max(peak_cts, len(live)), max(peak_bytes, live_bytes)
            for dead in [v for v in [k, *self.nodes[k][0]] if last_use[v] == i and v != self.root]:
                if dead in live:
                    live.discard(dead)
                    live_bytes -= size(dead)
        return peak_cts, peak_bytes


def estimate_expression_memory(
    expr: Union[str, Expr],
    backend: str = "gpu",
    params: Optional[CKKSParams] = None,
) -> MemoryEstimate:
    """Peak memory of a VecLang expression run with CKKS on HEonGPU ("gpu") or Lattigo ("cpu")."""
    if isinstance(expr, str):
        expr = parse_sexpr(expr)
    if params is None:
        params = chehab_params(get_multiplicative_depth(expr))
    live = _Liveness(expr, top_level=params.L - 1)
    # HEonGPU keeps a ciphertext's top-level buffer after rescaling; Lattigo shrinks it.
    peak_cts, peak_bytes = live.peak(params.N, level_aware=(backend == "cpu"))
    terms = estimate_peak_memory(params.N, params.L, params.K, len(live.rotations), peak_bytes, backend)
    galois = len(live.rotations) + (1 if backend == "gpu" and live.rotations else 0)
    return MemoryEstimate(params, len(live.rotations), galois, peak_cts, terms)


class CKKSMemoryEstimator:
    def __init__(self, backend: str = "gpu"):
        if backend not in OVERHEAD_COEFFS:
            raise ValueError(f"backend must be one of {list(OVERHEAD_COEFFS)}")
        self.backend = backend

    def estimate(self, expr: Union[str, Expr]) -> float:
        """Estimated peak memory in MB."""
        return estimate_expression_memory(expr, self.backend).total_mb
