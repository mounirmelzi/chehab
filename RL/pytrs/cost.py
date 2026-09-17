from expr import Expr, Var, Const, Op
from serializer import expr_to_str
import subprocess
from rules import create_rules
try:
    from . import config as pytrs_config
except ImportError:
    import config as pytrs_config

LITERAL = 0
STRUCTURE = 2000
VEC_OP = 1
OP = 1

def _operations_cost_constrained(expr: Expr) -> int:
    if isinstance(expr, (Const, Var)):
        node_cost = LITERAL
    elif isinstance(expr, Op):
        op = expr.op
        visit_all_children = True
        if op in ("+", "Add", "-", "Minus", "*", "Mul"):
            node_cost = OP * 250
        elif op == "Neg":
            node_cost = OP * 250
        elif op == "<<":
            node_cost = VEC_OP * 50
            visit_all_children = False
        elif op == "Vec":
            node_cost = 0
        elif op == "VecAdd":
            node_cost = VEC_OP
        elif op == "VecMinus":
            node_cost = VEC_OP
        elif op == "VecMul":
            node_cost = VEC_OP * 100
        elif op == "VecNeg":
            node_cost = VEC_OP
        else:
            raise ValueError(f"Unknown operator: {op}")

        if visit_all_children:
            for child in expr.args:
                node_cost += _operations_cost_constrained(child)
    else:
        node_cost = 0
    return node_cost

def _operations_cost_mo(expr: Expr) -> int:
    if isinstance(expr, (Const, Var)):
        node_cost = LITERAL
    elif isinstance(expr, Op):
        op = expr.op
        visit_all_children = True
        if op in ("+", "Add", "-", "Minus", "*", "Mul"):
            node_cost = OP * 250
        elif op == "Neg":
            node_cost = OP * 250
        elif op == "<<":
            node_cost = VEC_OP * 50
            if not isinstance(expr.args[0], Op) or expr.args[0].op != "<<":
                visit_all_children = False
        elif op == "Vec":
            node_cost = 0
        elif op == "VecAdd":
            node_cost = VEC_OP
        elif op == "VecMinus":
            node_cost = VEC_OP
        elif op == "VecMul":
            node_cost = VEC_OP * 100
        elif op == "VecNeg":
            node_cost = VEC_OP
        else:
            raise ValueError(f"Unknown operator: {op}")

        if visit_all_children:
            if op == "<<":
                node_cost += _operations_cost_mo(expr.args[0])
            else:    
                for child in expr.args:
                    node_cost += _operations_cost_mo(child)
    else:
        node_cost = 0
    return node_cost

def operations_cost(expr: Expr) -> int:
    if getattr(pytrs_config, "framework", "constrained") == "morl":
        return _operations_cost_mo(expr)
    return _operations_cost_constrained(expr)

def get_multiplicative_depth(expr: Expr) -> int:
    if isinstance(expr, Const) or isinstance(expr, Var):
        return 0
    elif isinstance(expr, Op):
        op = expr.op
        if not expr.args:
            return 0
        args = expr.args
        if op in {"*", "VecMul"}:
            return 1 + max(get_multiplicative_depth(arg) for arg in args)
        else:
            return max(get_multiplicative_depth(arg) for arg in args)
    else:
        raise ValueError(f"Unknown expression type: {expr}")

def get_normal_depth(expr: Expr) -> int:
    if isinstance(expr, Const) or isinstance(expr, Var):
        return 1
    elif isinstance(expr, Op):
        args = expr.args
        if not args:
            return 1
        else:
            return 1 + max(get_normal_depth(arg) for arg in args)
    else:
        raise ValueError(f"Unknown expression type: {expr}")

def count_operations(expr: Expr) -> int:
    if isinstance(expr, Const) or isinstance(expr, Var):
        return 0
    elif isinstance(expr, Op):
        return 1 + sum(count_operations(arg) for arg in expr.args)
    else:
        raise ValueError(f"Unknown expression type: {expr}")

def count_nodes(expr: Expr) -> int:
    if isinstance(expr, Const) or isinstance(expr, Var):
        return 1
    elif isinstance(expr, Op):
        return 1 + sum(count_nodes(arg) for arg in expr.args)
    else:
        raise ValueError(f"Unknown expression type: {expr}")
    
def rotations_cost(expr: Expr, parent: Expr = None) -> float:
    if isinstance(expr, Op):
        if expr.op == "<<" and isinstance(parent, Op) and parent.op in ("VecAdd", "VecMinus", "VecMul"):
            if len(expr.args) == 2 and isinstance(expr.args[1], Const):
                num_rotations = expr.args[1].value
                base_cost = 1.0 * num_rotations
                children_cost = sum(rotations_cost(child, expr) for child in expr.args)
                return base_cost + children_cost
        return sum(rotations_cost(child, expr) for child in expr.args)
    return 0.0

def evaluate_const_expr(expr: Expr) -> int:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return None
    if isinstance(expr, Op):
        operands = []
        for arg in expr.args:
            result = evaluate_const_expr(arg)
            if result is None:
                return None
            operands.append(result)
        if expr.op == '+':
            return sum(operands)
        elif expr.op == '-':
            if len(operands) == 1:
                return -operands[0]
            return operands[0] - operands[1]
        elif expr.op == '*':
            result = 1
            for op in operands:
                result *= op
            return result
        else:
            return None
    return None

def get_unique_rotations(expr: Expr, rotations=None) -> int:
    if rotations is None:
        rotations = set()
    if isinstance(expr, Op):
        if expr.op == "<<":
            offset = expr.args[1]
            rotation_value = None
            if isinstance(offset, Const):
                rotation_value = offset.value
            elif isinstance(offset, Op):
                try:
                    rotation_value = evaluate_const_expr(offset)
                except Exception as e:
                    pass
            if rotation_value is not None:
                rotations.add(rotation_value)
        for child in expr.args:
            get_unique_rotations(child, rotations)
    return len(rotations)

def get_total_rotations(expr: Expr) -> int:
    count = 0
    if isinstance(expr, Op):
        if expr.op == "<<":
            count += 1
        for child in expr.args:
            count += get_total_rotations(child)
    return count

def _calculate_cost_constrained(expr: Expr,
               w_ops=1.0,
               w_rot=1.0,
               w_depth=1.0,
               w_muldepth=1.0,
               w_vec=-1.0,
               ) -> float:
    return (
        w_ops * operations_cost(expr) +
        w_rot * rotations_cost(expr) +
        w_depth * get_normal_depth(expr) +
        w_muldepth * get_multiplicative_depth(expr) 
    )

def _calculate_cost_mo(expr: Expr,
               w_ops=1.0,
               w_rot=0.0,
               w_depth=1.0,
               w_muldepth=1.0,
               w_keys=1.0
               ) -> float:
    return (
        w_ops * operations_cost(expr) +
        w_rot * rotations_cost(expr) +
        w_depth * get_normal_depth(expr) +
        w_muldepth * get_multiplicative_depth(expr) +
        w_keys * get_unique_rotations(expr)
    )

def calculate_cost(expr: Expr,
               w_ops=1.0,
               w_rot=None,
               w_depth=1.0,
               w_muldepth=1.0,
               w_keys=None,
               w_vec=-1.0
               ) -> float:
    if getattr(pytrs_config, "framework", "constrained") == "morl":
        return _calculate_cost_mo(
            expr, 
            w_ops=w_ops, 
            w_rot=0.0 if w_rot is None else w_rot, 
            w_depth=w_depth, 
            w_muldepth=w_muldepth, 
            w_keys=1.0 if w_keys is None else w_keys
        )
        
    return _calculate_cost_constrained(
        expr, 
        w_ops=w_ops, 
        w_rot=1.0 if w_rot is None else w_rot, 
        w_depth=w_depth, 
        w_muldepth=w_muldepth, 
        w_vec=w_vec
    )
