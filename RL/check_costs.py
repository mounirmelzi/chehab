import glob
from pytrs import parse_sexpr, calculate_cost

for f in sorted(glob.glob("expression_*.txt")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        t = parse_sexpr(line)
        old = calculate_cost(t, w_rot=1.0, w_keys=0.0)   # previous default
        new = calculate_cost(t, w_keys=0.0)               # new default (w_rot=0.0)
        keys = calculate_cost(t, w_keys=1.0) - new
        print(f"{f:38s} ops old={old:9.1f}  new={new:9.1f}  diff={old-new:6.1f}  keys={keys}")
        break   # first expression per file