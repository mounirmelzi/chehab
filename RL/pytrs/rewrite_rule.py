from __future__ import annotations
"""Rewrite rule functionality."""

from typing import List, Dict, Union, Tuple, Optional
from collections import deque

from .expr import Expr, Const, Var, Op
from .pattern import Pattern
from .vectorization_analyzer import VectorizationAnalyzer
from .util import generate_random_assignments, evaluate_expr

MAX_VECTOR_SIZE = 32

# Binary SIMD ops whose operands must all share one lane count.
BINARY_VEC_OPS = {"VecMul", "VecAdd", "VecMinus"}


def _simd_lane_count(e: Expr) -> Optional[int]:
    """Lane count of the SIMD value *e* evaluates to (None if not a vector)[cite: 15]."""
    if isinstance(e, Op):
        if e.op == "Vec":
            return len(e.args)
        if e.op in ("VecAdd", "VecMinus", "VecMul", "<<") and len(e.args) == 2:
            return _simd_lane_count(e.args[0])
    return None


class RewriteRule:
    """
    A single rewrite rule: when LHS pattern matches, replace by RHS template.
    Supports flexible vectorization, rotation vectorization, and de-rotation[cite: 15, 16].
    """

    def __init__(self, name: str, lhs: Expr, rhs: Expr, rule_type: str = "normal"):
        self.name = name
        self.rule_type = rule_type
        
        if rule_type == "vectorize":
            self.scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-flexible":
            scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.target_ops = [scalar_op]
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.min_count = 2
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-rotation":
            self.scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.max_vector_size = MAX_VECTOR_SIZE
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-rotation-flexible":
            scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.target_ops = [scalar_op]
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.min_count = 2
            self.max_vector_size = MAX_VECTOR_SIZE
            self.lhs = None
            self.rhs = None
        else:
            self.lhs = Pattern(lhs)
            self.rhs = rhs
            
        self.rotation_rules = []
        if name not in ["rotation-mul", "rotation-add", "rotation-sub", "rotation-neg"]:
            from .rule_parser import parse_rules_from_text
            self.rotation_rules = parse_rules_from_text("""
                Rewrite { name: "rotation-mul", searcher: (VecMul x (<< x a)), applier: (VecMul x (<< x a)) }
                Rewrite { name: "rotation-add", searcher: (VecAdd x (<< x a)), applier: (VecAdd x (<< x a)) }
                Rewrite { name: "rotation-sub", searcher: (VecMinus x (<< x a)), applier: (VecMinus x (<< x a)) }
            """)

    def coverage_progress(self, expr_tree: Expr):
        if self.rule_type == "vectorize":
            return VectorizationAnalyzer.vectorize_coverage(expr_tree, self.scalar_op)
        elif self.rule_type == "vectorize-flexible":
            return VectorizationAnalyzer.flexible_vectorize_coverage(expr_tree, self.target_ops, self.min_count)
        elif self.rule_type == "vectorize-rotation":
            return VectorizationAnalyzer.rotation_vectorize_coverage(expr_tree, self.scalar_op, self.max_vector_size)
        elif self.rule_type == "vectorize-rotation-flexible":
            return VectorizationAnalyzer.flexible_rotation_vectorize_coverage(expr_tree, self.target_ops, self.min_count, self.max_vector_size)
        else:
            return VectorizationAnalyzer.analyze_lane_coverages(self.lhs.expr, expr_tree)

    def apply(self, expr: Expr) -> Optional[Expr]:
        if self.rule_type == "vectorize":
            return self._apply_vectorize(expr)
        elif self.rule_type == "vectorize-flexible":
            return self._apply_flexible_vectorize(expr)
        elif self.rule_type == "vectorize-rotation":
            return self._apply_rotation_vectorize(expr)
        elif self.rule_type == "vectorize-rotation-flexible":
            return self._apply_flexible_rotation_vectorize(expr)
        else:
            subst = self.lhs.match(expr)
            if subst is None:
                return None
            result = self._build_rhs(self.rhs, subst)
            return result if result.validate_expression() else None

    def _apply_guarded(
        self,
        target: Expr,
        parent: Optional[Expr] = None,
        parent_idx: Optional[int] = None,
    ) -> Optional[Expr]:
        """Guard against invalid lane configurations during rotation vectorization[cite: 15]."""
        rewritten = self.apply(target)
        if rewritten is None:
            return None
        lanes_before = _simd_lane_count(target)
        lanes_after  = _simd_lane_count(rewritten)
        lane_change  = (lanes_before is not None
                        and lanes_after is not None
                        and lanes_before != lanes_after)
        if (lane_change and parent is not None and parent_idx is not None
                and parent.op in BINARY_VEC_OPS and len(parent.args) == 2):
            sibling = parent.args[1 - parent_idx]
            if not (isinstance(sibling, Op) and sibling.op == "<<"):
                return None
        return rewritten

    def _apply_vectorize(self, expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
        if len(expr.args) < 2:
            return None
        first_lane = expr.args[0]
        if not isinstance(first_lane, Op) or first_lane.op != self.scalar_op:
            return None
        is_unary = len(first_lane.args) == 1
        is_binary = len(first_lane.args) == 2
        if not (is_unary or is_binary):
            return None
        if is_binary:
            left_ops, right_ops = [], []
            for lane in expr.args:
                if not (isinstance(lane, Op) and lane.op == self.scalar_op and len(lane.args) == 2):
                    return None
                left_ops.append(lane.args[0])
                right_ops.append(lane.args[1])
            result = Op(self.vector_op, [Op("Vec", left_ops), Op("Vec", right_ops)])
        else:
            operands = []
            for lane in expr.args:
                if not (isinstance(lane, Op) and lane.op == self.scalar_op and len(lane.args) == 1):
                    return None
                operands.append(lane.args[0])
            result = Op(self.vector_op, [Op("Vec", operands)])
        return result if result.validate_expression() else None

    def _apply_flexible_vectorize(self, expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, Op) and expr.op == "Vec") or len(expr.args) < 2:
            return None
        binary_lanes, unary_lanes = [], []
        bin_left, bin_right, un_ops = [], [], []
        for i, lane in enumerate(expr.args):
            if isinstance(lane, Op) and lane.op in self.target_ops:
                if len(lane.args) == 2:
                    binary_lanes.append(i)
                    bin_left.append(lane.args[0])
                    bin_right.append(lane.args[1])
                elif len(lane.args) == 1:
                    unary_lanes.append(i)
                    un_ops.append(lane.args[0])
        total = len(binary_lanes) + len(unary_lanes)
        if total < self.min_count or total == len(expr.args):
            return None
        target_op = self.target_ops[0]
        identity_value = 1 if target_op == "*" else 0
        if len(binary_lanes) >= len(unary_lanes):
            left_els, right_els, b_idx, u_idx = [], [], 0, 0
            for i in range(len(expr.args)):
                if i in binary_lanes:
                    left_els.append(bin_left[b_idx])
                    right_els.append(bin_right[b_idx])
                    b_idx += 1
                elif i in unary_lanes:
                    left_els.append(un_ops[u_idx])
                    right_els.append(Const(identity_value))
                    u_idx += 1
                else:
                    left_els.append(expr.args[i])
                    right_els.append(Const(identity_value))
            result = Op(self.vector_op, [Op("Vec", left_els), Op("Vec", right_els)])
        else:
            op_els, b_idx, u_idx = [], 0, 0
            for i in range(len(expr.args)):
                if i in unary_lanes:
                    op_els.append(un_ops[u_idx])
                    u_idx += 1
                elif i in binary_lanes:
                    op_els.append(bin_left[b_idx])
                    b_idx += 1
                else:
                    op_els.append(expr.args[i])
            result = Op(self.vector_op, [Op("Vec", op_els)])
        return result if result.validate_expression() else None

    def _apply_rotation_vectorize(self, expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
        original_size = len(expr.args)
        if original_size < 1 or original_size * 2 > self.max_vector_size:
            return None
        first_lane = expr.args[0]
        if not isinstance(first_lane, Op) or first_lane.op != self.scalar_op:
            return None
        is_unary = len(first_lane.args) == 1
        is_binary = len(first_lane.args) == 2
        if not (is_unary or is_binary):
            return None
        if is_binary:
            first_ops, second_ops = [], []
            for lane in expr.args:
                if not (isinstance(lane, Op) and lane.op == self.scalar_op and len(lane.args) == 2):
                    return None
                first_ops.append(lane.args[0])
                second_ops.append(lane.args[1])
            doubled = first_ops + second_ops
        else:
            operands = []
            for lane in expr.args:
                if not (isinstance(lane, Op) and lane.op == self.scalar_op and len(lane.args) == 1):
                    return None
                operands.append(lane.args[0])
            doubled = operands + operands
        doubled_vector = Op("Vec", doubled)
        rotated_vector = Op("<<", [doubled_vector, Const(original_size)])
        result = Op(self.vector_op, [doubled_vector, rotated_vector])
        return result if result.validate_expression() else None

    def _apply_flexible_rotation_vectorize(self, expr: Expr) -> Optional[Expr]:
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
        original_size = len(expr.args)
        if original_size < 2 or original_size * 2 > self.max_vector_size:
            return None
        binary_lanes, unary_lanes = [], []
        bin_first, bin_second, un_ops = [], [], []
        for i, lane in enumerate(expr.args):
            if isinstance(lane, Op) and lane.op in self.target_ops:
                if len(lane.args) == 2:
                    binary_lanes.append(i)
                    bin_first.append(lane.args[0])
                    bin_second.append(lane.args[1])
                elif len(lane.args) == 1:
                    unary_lanes.append(i)
                    un_ops.append(lane.args[0])
        total = len(binary_lanes) + len(unary_lanes)
        
        def _is_identity_padding_lane(lane: Expr) -> bool:
            return isinstance(lane, Const) and lane.value == (1 if self.target_ops[0] == "*" else 0)

        vectorizable_idx = set(binary_lanes) | set(unary_lanes)
        non_vectorizable_non_padding = sum(
            1 for i, lane in enumerate(expr.args)
            if i not in vectorizable_idx and not _is_identity_padding_lane(lane)
        )
        min_required = 1 if (original_size == 2 or non_vectorizable_non_padding == 0) else self.min_count
        if total < min_required or total == original_size:
            return None

        target_op = self.target_ops[0]
        identity_value = 1 if target_op == "*" else 0
        first_half, second_half, b_idx, u_idx = [], [], 0, 0
        for i in range(original_size):
            if i in binary_lanes:
                first_half.append(bin_first[b_idx])
                second_half.append(bin_second[b_idx])
                b_idx += 1
            elif i in unary_lanes:
                first_half.append(un_ops[u_idx])
                second_half.append(un_ops[u_idx])
                u_idx += 1
            else:
                first_half.append(expr.args[i])
                second_half.append(Const(identity_value))
        doubled_vector = Op("Vec", first_half + second_half)
        rotated_vector = Op("<<", [doubled_vector, Const(original_size)])
        result = Op(self.vector_op, [doubled_vector, rotated_vector])
        return result if result.validate_expression() else None

    def _build_rhs(self, template: Expr, subst: Dict[str, Expr]) -> Expr:
        if isinstance(template, Var):
            return subst[template.name]
        if isinstance(template, Const):
            return Const(template.value)
        if isinstance(template, Op):
            return Op(template.op, [self._build_rhs(arg, subst) for arg in template.args])
        raise TypeError("unknown Expr in RHS build")

    def find_matching_subexpressions(self, expr: Expr) -> List[Tuple[List[int], Expr]]:
        if self.rule_type in ["vectorize", "vectorize-flexible", "vectorize-rotation", "vectorize-rotation-flexible"]:
            return self._find_vectorize_matches(expr)
        elif self.rule_type == "de-rotate":
            matches = []
            self._find_matches_recursive(expr, [], matches)
            valid = [(p, m) for p, m in matches if self._apply_via_path(expr, p) is not None]
            return [([], expr)] if valid else []
        else:
            matches: List[Tuple[List[int], Expr]] = []
            self._find_matches_recursive(expr, [], matches)
            return [(p, m) for p, m in matches if self._apply_via_path(expr, p) is not None]

    def _find_vectorize_matches(self, expr: Expr) -> List[Tuple[List[int], Expr]]:
        matches = []
        def _find_recursive(current: Expr, path: List[int], parent: Optional[Expr] = None, parent_idx: Optional[int] = None):
            if isinstance(current, Op) and current.op == "Vec":
                if self._apply_guarded(current, parent, parent_idx) is not None:
                    matches.append((path.copy(), current))
            if isinstance(current, Op):
                rotation = any(rule.lhs.match(current) is not None for rule in self.rotation_rules)
                if not rotation or self.name.startswith("rotate_"):
                    for i, child in enumerate(current.args):
                        _find_recursive(child, path + [i], current, i)
                else:
                    _find_recursive(current.args[0], path + [0], current, 0)
        _find_recursive(expr, [])
        return matches

    def _find_matches_recursive(self, current: Expr, path: List[int], matches: List[Tuple[List[int], Expr]]):
        queue = deque([(path, current)])
        while queue:
            cur_path, node = queue.popleft()
            if self.lhs.match(node) is not None:
                matches.append((cur_path.copy(), node))
            if isinstance(node, Op):
                if self.rule_type == "de-rotate":
                    for i, child in enumerate(node.args):
                        queue.append((cur_path + [i], child))
                else:
                    rotation = any(rule.lhs.match(node) is not None for rule in self.rotation_rules)
                    if not rotation:
                        for i, child in enumerate(node.args):
                            queue.append((cur_path + [i], child))
                    else:
                        queue.append((cur_path + [0], node.args[0]))

    def _apply_via_path(self, expr: Expr, path: List[int]) -> Optional[Expr]:
        WRAPPER_OPS = {"VecMul", "VecAdd", "VecMinus"}

        def rec(node: Expr, subpath: List[int], parent: Optional[Expr] = None, parent_idx: Optional[int] = None) -> Expr:
            if not subpath:
                return self._apply_guarded(node, parent, parent_idx) or node
            if not isinstance(node, Op):
                return node
            idx = subpath[0]
            if node.op in WRAPPER_OPS and idx == 0 and self.rule_type in {"vectorize", "vectorize-flexible", "vectorize-rotation", "vectorize-rotation-flexible"}:
                new_vec = rec(node.args[0], subpath[1:], node, 0)
                other = node.args[1]
                new_other = other
                if isinstance(other, Op) and other.op == "<<" and len(other.args) >= 1:
                    new_other = Op("<<", [new_vec, *other.args[1:]])
                return Op(node.op, [new_vec, new_other])

            new_args = [
                rec(a, subpath[1:], node, i) if i == idx else a
                for i, a in enumerate(node.args)
            ]
            return Op(node.op, new_args)

        return rec(expr, path)

    @staticmethod
    def _replace_expr(node: Expr, old: Expr, new: Expr) -> Expr:
        if node is old or node == old:
            return new
        if isinstance(node, Op):
            return Op(node.op, [RewriteRule._replace_expr(a, old, new) for a in node.args])
        return node

    def apply_rule(self, expr: Expr, match: Optional[Expr] = None, path: Optional[List[int]] = None) -> Expr:
        if self.rule_type == "de-rotate":
            return self._apply_rule_everywhere(expr)
        if path is not None:
            return self._apply_via_path(expr, path)
        if match is not None:
            return self._apply_rule_match(expr, match)
        return self.apply(expr) or expr

    def _apply_rule_everywhere(self, expr: Expr) -> Expr:
        while True:
            matches = []
            self._find_matches_recursive(expr, [], matches)
            valid = [(p, m) for p, m in matches if self._apply_via_path(expr, p) is not None]
            if not valid:
                break
            expr = self._apply_via_path(expr, valid[0][0])
        return expr

    def _apply_rule_match(self, expr: Expr, match: Expr) -> Expr:
        if expr is match:
            return self.apply(expr) or expr
        if isinstance(expr, Op):
            new_args = [self._apply_rule_match(a, match) for a in expr.args]
            return Op(expr.op, new_args)
        return expr

    def apply_rule_check(self, expr: Expr, match: Expr | None = None) -> Tuple[Expr, bool]:
        subst_expr = self.apply_rule(expr, match)
        val_map = generate_random_assignments(expr)
        return (
            subst_expr,
            evaluate_expr(expr, val_map) == evaluate_expr(subst_expr, val_map)
        )

    def __repr__(self) -> str:
        if self.rule_type == "vectorize":
            return f"name:{self.name},type:vectorize,scalar_op:{self.scalar_op},vector_op:{self.vector_op}"
        elif self.rule_type == "vectorize-flexible":
            return f"name:{self.name},type:vectorize-flexible,target_ops:{self.target_ops},vector_op:{self.vector_op}"
        elif self.rule_type == "vectorize-rotation":
            return f"name:{self.name},type:vectorize-rotation,scalar_op:{self.scalar_op},vector_op:{self.vector_op},max_size:{self.max_vector_size}"
        elif self.rule_type == "vectorize-rotation-flexible":
            return f"name:{self.name},type:vectorize-rotation-flexible,target_ops:{self.target_ops},vector_op:{self.vector_op},max_size:{self.max_vector_size}"
        else:
            return f"name:{self.name},searcher:{self.lhs.expr},applier:{self.rhs}"