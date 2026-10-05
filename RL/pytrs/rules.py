import re
import subprocess
from typing import List

try:
    from .expr import Expr, Const, Var, Op
    from .rewrite_rule import RewriteRule
    from .rule_parser import parse_rules_from_text
    from .parser import parse_sexpr
    from . import config as pytrs_config
except ImportError:
    from expr import Expr, Const, Var, Op
    from rewrite_rule import RewriteRule
    from rule_parser import parse_rules_from_text
    from parser import parse_sexpr
    import config as pytrs_config


def _create_rules_constrained(path: str = None, rotations_rules_path: str = None) -> dict:
    rules = []
    if path is not None:
        rules_text = open(path, "r").read().replace("?", "")
        rules.extend(parse_rules_from_text(rules_text))

    if rotations_rules_path is not None:
        rotations_rules_text = open(rotations_rules_path, "r").read().replace("?", "")
        rules.extend(parse_rules_from_text(rotations_rules_text))

    rules_dict = {rule.name: rule for rule in rules}
    return rules_dict


def _create_rules_mo(rules_path: str = None, rotations_rules_path: str = None) -> dict:
    rules = []
    if rules_path is not None:
        rules_text = open(rules_path, "r").read().replace("?", "")
        rules.extend(parse_rules_from_text(rules_text))

    if rotations_rules_path is not None:
        rotations_rules_text = open(rotations_rules_path, "r").read().replace("?", "")
        rules.extend(parse_rules_from_text(rotations_rules_text))

    rules_dict = {rule.name: rule for rule in rules}
    return rules_dict


def create_rules(path: str = None, rotations_rules_path: str = None, rules_path: str = None) -> dict:
    """Charge les règles en supportant les frameworks 'constrained' et 'morl'."""
    target_path = rules_path if rules_path is not None else path
    if getattr(pytrs_config, "framework", "constrained") == "morl":
        return _create_rules_mo(target_path, rotations_rules_path)
    return _create_rules_constrained(target_path, rotations_rules_path)


def group_rules_from_dict(rules_dict):
    grouped = {}
    keys = list(rules_dict.keys()) + ["END"]
    for rule_name in keys:
        if rule_name == "END":
            grouped.setdefault("END", []).append(rule_name)
        else:
            m = re.match(r"^(.*?)-(\d+)$", rule_name)
            if m:
                base = m.group(1)
                grouped.setdefault(base, []).append(rule_name)
            else:
                grouped.setdefault(rule_name, []).append(rule_name)
    return grouped