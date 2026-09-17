from expr import Const, Var, Op
from rewrite_rule import RewriteRule
import re
from typing import List
from rule_parser import parse_rules_from_text
from expr import Expr, Const, Var
from parser import parse_sexpr
import subprocess


try:
    from . import config as pytrs_config
except ImportError:
    import config as pytrs_config


def _create_rules_constrained(path: str) -> dict:

    rules_text = open(path, "r").read().replace("?", "")
    rules = parse_rules_from_text(rules_text)

    rules_dict = {rule.name: rule for rule in rules}
    return rules_dict


def _create_rules_mo(rules_path: str, rotations_rules_path: str = None) -> dict:
    rules_text = open(rules_path, "r").read().replace("?", "")
    rules = parse_rules_from_text(rules_text)

    if rotations_rules_path:
        rotations_rules_text = open(rotations_rules_path, "r").read().replace("?", "")
        rotations_rules = parse_rules_from_text(rotations_rules_text)
        rules.extend(rotations_rules)

    rules_dict = {rule.name: rule for rule in rules}
    return rules_dict


def create_rules(path: str, rotations_rules_path: str = None) -> dict:
    if getattr(pytrs_config, "framework", "constrained") == "morl":
        return _create_rules_mo(path, rotations_rules_path)
    return _create_rules_constrained(path)


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
