import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytrs'))

from parser import parse_sexpr
from rule_parser import parse_rules_from_text

rules_text = open("rules.txt").read().replace("?", "")
rules = parse_rules_from_text(rules_text)
rot_text = open("rotations_rules.txt").read().replace("?", "")
rules.extend(parse_rules_from_text(rot_text))
rules_dict = {r.name: r for r in rules}

expr_str = "(VecAdd (Vec (+ (+ (- (+ v1_0 v2_0) (* (* v1_0 v2_0) 2)) (- (+ v1_1 v2_1) (* 2 (* v1_1 v2_1)))) (+ (- (+ v1_2 v2_2) (* 2 (* v1_2 v2_2))) (- (+ v1_3 v2_3) (* 2 (* v1_3 v2_3))))) (+ (+ (- (+ v1_4 v2_4) (* 2 (* v1_4 v2_4))) (- (+ v1_5 v2_5) (* 2 (* v1_5 v2_5)))) (+ (- (+ v1_6 v2_6) (* 2 (* v1_6 v2_6))) (- (+ v1_7 v2_7) (* 2 (* v1_7 v2_7)))))) (<< (Vec (+ (+ (- (+ v1_0 v2_0) (* (* v1_0 v2_0) 2)) (- (+ v1_1 v2_1) (* 2 (* v1_1 v2_1)))) (+ (- (+ v1_2 v2_2) (* 2 (* v1_2 v2_2))) (- (+ v1_3 v2_3) (* 2 (* v1_3 v2_3))))) (+ (+ (- (+ v1_4 v2_4) (* 2 (* v1_4 v2_4))) (- (+ v1_5 v2_5) (* 2 (* v1_5 v2_5)))) (+ (- (+ v1_6 v2_6) (* 2 (* v1_6 v2_6))) (- (+ v1_7 v2_7) (* 2 (* v1_7 v2_7)))))) 1))"

expr = parse_sexpr(expr_str)

for name, rule in rules_dict.items():
    matches = rule.find_matching_subexpressions(expr)
    if matches:
        print(name + ": " + str(len(matches)) + " match(es), paths=" + str([p for p, m in matches]))
