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
    """Lane count of the SIMD value *e* evaluates to (None if not a vector)."""
    if isinstance(e, Op):
        if e.op == "Vec":
            return len(e.args)
        if e.op in ("VecAdd", "VecMinus", "VecMul", "<<") and len(e.args) == 2:
            return _simd_lane_count(e.args[0])
    return None


class RewriteRule:
    """
    A single rewrite rule: when LHS pattern matches, replace by RHS template.
    Now supports flexible vectorization and rotation vectorization with binary/unary operations.
    """

    def __init__(self, name: str, lhs: Expr, rhs: Expr, rule_type: str = "normal"):
        self.name = name
        self.rule_type = rule_type
        
        if rule_type == "vectorize":
            # Simple uniform vectorization - extract operator names from Var names
            self.scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-flexible":
            # Flexible vectorization - extract operator names from Var names
            scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.target_ops = [scalar_op]
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.min_count = 2  # minimum matching operations to vectorize
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-rotation":
            # Uniform rotation vectorization
            self.scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.max_vector_size = MAX_VECTOR_SIZE  # size limit for resulting vector
            self.lhs = None
            self.rhs = None
        elif rule_type == "vectorize-rotation-flexible":
            # Flexible rotation vectorization
            scalar_op = lhs.name if isinstance(lhs, Var) else str(lhs)
            self.target_ops = [scalar_op]
            self.vector_op = rhs.name if isinstance(rhs, Var) else str(rhs)
            self.min_count = 2
            self.max_vector_size = MAX_VECTOR_SIZE
            self.lhs = None
            self.rhs = None
        else:
            # Regular rule behavior
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
        """
        Returns coverage statistics for *this* rule's search pattern.
        """
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
        """
        If rule matches *expr* at the top level, return transformed Expr;
        else None.
        """
        if self.rule_type == "vectorize":
            return self._apply_vectorize(expr)
        elif self.rule_type == "vectorize-flexible":
            return self._apply_flexible_vectorize(expr)
        elif self.rule_type == "vectorize-rotation":
            return self._apply_rotation_vectorize(expr)
        elif self.rule_type == "vectorize-rotation-flexible":
            return self._apply_flexible_rotation_vectorize(expr)
        else:
            # Regular rule logic
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
        """self.apply() plus the sibling-lane-count precondition.

        Rotation-vectorization rewrites double the lane count of the Vec
        they fire on. If that Vec is an operand of a binary SIMD op
        (VecAdd/VecMinus/VecMul) whose sibling keeps the old lane count,
        the rewrite breaks the equal-lanes invariant of the parent (the
        "VecMinus operands must be equal-length vectors" eval crash).
        Such rewrites are refused (return None). Siblings that are `<<`
        rotation nodes are exempt: _apply_via_path rebuilds them in
        lockstep from the rewritten vector.
        """
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
        """Apply uniform vectorization - handles both binary and unary operations."""
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
            left_operands = []
            right_operands = []
            
            for lane in expr.args:
                if not (isinstance(lane, Op) and 
                       lane.op == self.scalar_op and 
                       len(lane.args) == 2):
                    return None
                    
                left_operands.append(lane.args[0])
                right_operands.append(lane.args[1])
            
            left_vec = Op("Vec", left_operands)
            right_vec = Op("Vec", right_operands)
            result = Op(self.vector_op, [left_vec, right_vec])
            
        else:  
            operands = []
            
            for lane in expr.args:
                if not (isinstance(lane, Op) and 
                       lane.op == self.scalar_op and 
                       len(lane.args) == 1):
                    return None
                    
                operands.append(lane.args[0])
            
            operand_vec = Op("Vec", operands)
            result = Op(self.vector_op, [operand_vec])
        
        return result if result.validate_expression() else None

    def _apply_flexible_vectorize(self, expr: Expr) -> Optional[Expr]:
        """Apply flexible vectorization - handles both binary and unary operations with correct ordering."""
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
            
        if len(expr.args) < 2:
            return None
            
        binary_lanes = []
        unary_lanes = []
        binary_left_operands = []
        binary_right_operands = []
        unary_operands = []
        
        for i, lane in enumerate(expr.args):
            if isinstance(lane, Op) and lane.op in self.target_ops:
                if len(lane.args) == 2:
                    binary_lanes.append(i)
                    binary_left_operands.append(lane.args[0])
                    binary_right_operands.append(lane.args[1])
                elif len(lane.args) == 1:
                    unary_lanes.append(i)
                    unary_operands.append(lane.args[0])
        
        total_vectorizable = len(binary_lanes) + len(unary_lanes)
        
        if total_vectorizable < self.min_count:
            return None
        
        if total_vectorizable == len(expr.args):
            return None
        
        target_op = self.target_ops[0]
        is_non_commutative = target_op in {"-", "/"}
        
        if target_op == "*":
            identity_value = 1
        elif target_op in {"+", "-"}:
            identity_value = 0
        elif target_op == "/":
            identity_value = 1
        else:
            identity_value = 0
        
        if len(binary_lanes) >= len(unary_lanes):
            left_vec_elements = []
            right_vec_elements = []
            
            binary_idx = 0
            unary_idx = 0
            
            for i in range(len(expr.args)):
                if i in binary_lanes:
                    left_vec_elements.append(binary_left_operands[binary_idx])
                    right_vec_elements.append(binary_right_operands[binary_idx])
                    binary_idx += 1
                elif i in unary_lanes:
                    left_vec_elements.append(unary_operands[unary_idx])
                    right_vec_elements.append(Const(identity_value))
                    unary_idx += 1
                else:
                    if is_non_commutative:
                        left_vec_elements.append(expr.args[i])
                        right_vec_elements.append(Const(identity_value))
                    else:
                        left_vec_elements.append(Const(identity_value))
                        right_vec_elements.append(expr.args[i])
            
            left_vec = Op("Vec", left_vec_elements)
            right_vec = Op("Vec", right_vec_elements)
            result = Op(self.vector_op, [left_vec, right_vec])
            
        else:
            operand_vec_elements = []
            binary_idx = 0
            unary_idx = 0
            
            for i in range(len(expr.args)):
                if i in unary_lanes:
                    operand_vec_elements.append(unary_operands[unary_idx])
                    unary_idx += 1
                elif i in binary_lanes:
                    operand_vec_elements.append(binary_left_operands[binary_idx])
                    binary_idx += 1
                else:
                    operand_vec_elements.append(expr.args[i])
            
            operand_vec = Op("Vec", operand_vec_elements)
            result = Op(self.vector_op, [operand_vec])
        
        return result if result.validate_expression() else None

    def _apply_rotation_vectorize(self, expr: Expr) -> Optional[Expr]:
        """Apply uniform rotation vectorization - handles both binary and unary operations."""
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
            
        original_size = len(expr.args)
        if original_size < 1:
            return None
            
        new_size = original_size * 2
        if new_size > self.max_vector_size:
            return None
            
        first_lane = expr.args[0]
        if not isinstance(first_lane, Op) or first_lane.op != self.scalar_op:
            return None
        
        is_unary = len(first_lane.args) == 1
        is_binary = len(first_lane.args) == 2
        
        if not (is_unary or is_binary):
            return None
        
        if is_binary:
            first_operands = []
            second_operands = []
            
            for lane in expr.args:
                if not (isinstance(lane, Op) and 
                       lane.op == self.scalar_op and 
                       len(lane.args) == 2):
                    return None
                first_operands.append(lane.args[0])
                second_operands.append(lane.args[1])
            
            doubled_vector_elements = first_operands + second_operands
            
        else:  
            operands = []
            
            for lane in expr.args:
                if not (isinstance(lane, Op) and 
                       lane.op == self.scalar_op and 
                       len(lane.args) == 1):
                    return None
                operands.append(lane.args[0])
            
            doubled_vector_elements = operands + operands
        
        doubled_vector = Op("Vec", doubled_vector_elements)
        shift_amount = Const(original_size)
        rotated_vector = Op("<<", [doubled_vector, shift_amount])
        result = Op(self.vector_op, [doubled_vector, rotated_vector])
        
        return result if result.validate_expression() else None

    def _apply_flexible_rotation_vectorize(self, expr: Expr) -> Optional[Expr]:
        """Apply flexible rotation vectorization - handles both binary and unary operations with correct ordering."""
        if not (isinstance(expr, Op) and expr.op == "Vec"):
            return None
            
        original_size = len(expr.args)
        if original_size < 2:
            return None
            
        new_size = original_size * 2
        if new_size > self.max_vector_size:
            return None
            
        binary_lanes = []
        unary_lanes = []
        binary_first_operands = []
        binary_second_operands = []
        unary_operands = []
        
        for i, lane in enumerate(expr.args):
            if isinstance(lane, Op) and lane.op in self.target_ops:
                if len(lane.args) == 2:
                    binary_lanes.append(i)
                    binary_first_operands.append(lane.args[0])
                    binary_second_operands.append(lane.args[1])
                elif len(lane.args) == 1:
                    unary_lanes.append(i)
                    unary_operands.append(lane.args[0])
        
        total_vectorizable = len(binary_lanes) + len(unary_lanes)

        target_op = self.target_ops[0]
        is_non_commutative = target_op in {"-", "/"}
        
        if target_op == "*":
            identity_value = 1
        elif target_op in {"+", "-"}:
            identity_value = 0
        elif target_op == "/":
            identity_value = 1
        else:
            identity_value = 0

        def _is_identity_padding_lane(lane: Expr) -> bool:
            return isinstance(lane, Const) and lane.value == identity_value

        vectorizable_lane_idx = set(binary_lanes) | set(unary_lanes)
        non_vectorizable_non_padding = sum(
            1
            for i, lane in enumerate(expr.args)
            if i not in vectorizable_lane_idx and not _is_identity_padding_lane(lane)
        )

        if original_size == 2 or non_vectorizable_non_padding == 0:
            min_required = 1
        else:
            min_required = self.min_count
            
        if total_vectorizable < min_required:
            return None
        
        if total_vectorizable == original_size:
            return None
        
        first_half = []
        second_half = []
        
        binary_idx = 0
        unary_idx = 0
        
        for i in range(original_size):
            if i in binary_lanes:
                first_half.append(binary_first_operands[binary_idx])
                second_half.append(binary_second_operands[binary_idx])
                binary_idx += 1
            elif i in unary_lanes:
                first_half.append(unary_operands[unary_idx])
                second_half.append(unary_operands[unary_idx])
                unary_idx += 1
            else:
                if is_non_commutative:
                    first_half.append(expr.args[i])
                    second_half.append(Const(identity_value))
                else:
                    first_half.append(expr.args[i])
                    second_half.append(Const(identity_value))
        
        doubled_vector_elements = first_half + second_half
        doubled_vector = Op("Vec", doubled_vector_elements)
        shift_amount = Const(original_size)
        rotated_vector = Op("<<", [doubled_vector, shift_amount])
        result = Op(self.vector_op, [doubled_vector, rotated_vector])
        
        return result if result.validate_expression() else None

    def _build_rhs(self, template: Expr, subst: Dict[str, Expr]) -> Expr:
        """Instantiate RHS template using the substitution dict."""
        if isinstance(template, Var):
            return subst[template.name]
        if isinstance(template, Const):
            return Const(template.value)
        if isinstance(template, Op):
            return Op(
                template.op,
                [self._build_rhs(arg, subst) for arg in template.args]
            )
        raise TypeError("unknown Expr in RHS build")

    def find_matching_subexpressions(
        self, expr: Expr
    ) -> List[Tuple[List[int], Expr]]:
        if self.rule_type in ["vectorize", "vectorize-flexible", "vectorize-rotation", "vectorize-rotation-flexible"]:
            return self._find_vectorize_matches(expr)
        elif self.rule_type == "de-rotate":
            matches: List[Tuple[List[int], Expr]] = []
            self._find_matches_recursive(expr, [], matches)
            valid = [
                (p, m) for p, m in matches
                if self._apply_via_path(expr, p) is not None
            ]
            return [([], expr)] if valid else []
        else:
            # Regular rules
            matches: List[Tuple[List[int], Expr]] = []
            self._find_matches_recursive(expr, [], matches)
            return [
                (p, m) for p, m in matches
                if self._apply_via_path(expr, p) is not None
            ]

    def _find_vectorize_matches(self, expr: Expr) -> List[Tuple[List[int], Expr]]:
        """Find vectorization matches for all vectorization rule types."""
        matches = []
        
        def _find_recursive(current: Expr, path: List[int],
                            parent: Optional[Expr] = None,
                            parent_idx: Optional[int] = None):
            if isinstance(current, Op) and current.op == "Vec":
                if self._apply_guarded(current, parent, parent_idx) is not None:
                    matches.append((path.copy(), current))
            
            if isinstance(current, Op):
                rotation = False
                for rule in self.rotation_rules:
                    if rule.lhs.match(current) is not None:
                        rotation = True
                        break
                if not rotation or self.rule_type == "de-rotate":
                    for i, child in enumerate(current.args):
                        _find_recursive(child, path + [i], current, i)
                else:
                    _find_recursive(current.args[0], path + [0], current, 0)
        _find_recursive(expr, [])
        return matches

    def _find_matches_recursive(
        self, current: Expr, path: List[int],
        matches: List[Tuple[List[int], Expr]]
    ):
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
                    rotation = False
                    for rule in self.rotation_rules:
                        if rule.lhs.match(node) is not None:
                            rotation = True
                    if not rotation:
                        for i, child in enumerate(node.args):
                            queue.append((cur_path + [i], child))
                    else:
                        queue.append((cur_path + [0], node.args[0]))

    def _apply_via_path(self, expr: Expr, path: List[int]) -> Optional[Expr]:
        WRAPPER_OPS = {"VecMul", "VecAdd", "VecMinus"}

        def rec(node: Expr, subpath: List[int],
                parent: Optional[Expr] = None,
                parent_idx: Optional[int] = None) -> Expr:
            if not subpath:
                return self._apply_guarded(node, parent, parent_idx) or node

            if not isinstance(node, Op):
                return node

            idx = subpath[0]

            if (node.op in WRAPPER_OPS and idx == 0
                    and self.rule_type in {"vectorize", "vectorize-flexible",
                                           "vectorize-rotation", "vectorize-rotation-flexible"}):
                new_vec  = rec(node.args[0], subpath[1:], node, 0)   
                other    = node.args[1]
                new_other = self._replace_expr(other, node.args[0], new_vec)

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
            return Op(node.op, [RewriteRule._replace_expr(a, old, new)
                                for a in node.args])
        return node

    def apply_rule(
        self,
        expr:  Expr,
        match: Optional[Expr] = None,
        path:  Optional[List[int]] = None
    ) -> Expr:
        if self.rule_type == "de-rotate":
            return self._apply_rule_everywhere(expr)
        if path is not None:
            return self._apply_via_path(expr, path)
        if match is not None:
            return self._apply_rule_match(expr, match)
        return self.apply(expr) or expr

    def _apply_rule_everywhere(self, expr: Expr) -> Expr:
        """de-rotate: rewrite every occurrence of the offset in one call."""
        while True:
            matches: List[Tuple[List[int], Expr]] = []
            self._find_matches_recursive(expr, [], matches)
            valid = [
                (p, m) for p, m in matches
                if self._apply_via_path(expr, p) is not None
            ]
            if not valid:
                break
            rewritten = self._apply_via_path(expr, valid[0][0])
            if rewritten is None:
                break
            expr = rewritten
        return expr

    def _apply_rule_match(self, expr: Expr, match: Expr) -> Expr:
        if expr is match:
            return self.apply(expr) or expr
        if isinstance(expr, Op):
            new_args = [self._apply_rule_match(a, match) for a in expr.args]
            return Op(expr.op, new_args)
        return expr

    def apply_rule_check(self, expr: Expr, match: Expr | None = None
                         ) -> Tuple[Expr, bool]:
        subst_expr = self.apply_rule(expr, match)
        val_map    = generate_random_assignments(expr)
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