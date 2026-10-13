"""
Convert a CHEHAB-generated Lattigo program (generated_*.go) into an op trace
that both memory probes (HEonGPU and Lattigo) can execute.

    python make_trace.py <program.go> <out.trace> <out.json>

Trace format (one op per line):
    PARAMS <logN> <logQ comma list> <logP comma list>
    IN <id>            encrypt a fresh input at the top level
    MUL <dst> <a> <b>  ciphertext x ciphertext, relinearized
    RS <id>            rescale in place
    ADD <dst> <a> <b>
    FREE <id>          value is dead after this point
    OUT <id>

Values are renamed to SSA ids, and FREE is emitted right after the last use
of each value, so the trace has the same live set as the compiled program.
The JSON file holds static statistics used by the memory model.
"""
import json
import re
import sys

RE_PARAMS = {
    "LogN": re.compile(r"LogN:\s*(\d+)"),
    "LogQ": re.compile(r"LogQ:\s*\[\]int\{([^}]*)\}"),
    "LogP": re.compile(r"LogP:\s*\[\]int\{([^}]*)\}"),
}
RE_IN = re.compile(r"^\s*(\w+)\s*:?=\s*encryptedInputs\[\"[^\"]+\"\]\s*$")
RE_MUL = re.compile(r"^\s*(\w+),\s*_\s*:?=\s*eval\.MulRelinNew\((\w+),\s*(\w+)\)\s*$")
RE_ADD = re.compile(r"^\s*(\w+),\s*_\s*:?=\s*eval\.AddNew\((\w+),\s*(\w+)\)\s*$")
RE_RS = re.compile(r"^\s*_\s*=\s*eval\.Rescale\((\w+),\s*(\w+)\)\s*$")
RE_OUT = re.compile(r"^\s*encryptedOutputs\[\"[^\"]+\"\]\s*=\s*(\w+)\s*$")


def main():
    src, trace_path, json_path = sys.argv[1:4]
    text = open(src).read()
    main_idx = text.find("func main()")
    params = {k: r.search(text[main_idx:]) for k, r in RE_PARAMS.items()}
    logN = int(params["LogN"].group(1))
    logQ = [int(x) for x in params["LogQ"].group(1).split(",") if x.strip()]
    logP = [int(x) for x in params["LogP"].group(1).split(",") if x.strip()]
    if "bootstrapper.Bootstrap" in text:
        raise SystemExit("bootstrapping programs are not supported")

    body = text[:main_idx]
    ops, cur, nxt = [], {}, [0]

    def new(var):
        cur[var] = nxt[0]
        nxt[0] += 1
        return cur[var]

    for line in body.splitlines():
        if "eval." not in line and "encryptedInputs[" not in line and "encryptedOutputs[" not in line:
            continue
        if m := RE_IN.match(line):
            ops.append(["IN", new(m.group(1))])
        elif m := RE_MUL.match(line):
            a, b = cur[m.group(2)], cur[m.group(3)]
            ops.append(["MUL", new(m.group(1)), a, b])
        elif m := RE_ADD.match(line):
            a, b = cur[m.group(2)], cur[m.group(3)]
            ops.append(["ADD", new(m.group(1)), a, b])
        elif m := RE_RS.match(line):
            assert m.group(1) == m.group(2), line
            ops.append(["RS", cur[m.group(1)]])
        elif m := RE_OUT.match(line):
            ops.append(["OUT", cur[m.group(1)]])
        elif "func " in line:
            continue
        else:
            raise SystemExit(f"unsupported line: {line.strip()}")

    outputs = {op[1] for op in ops if op[0] == "OUT"}
    last_use = {}
    for i, op in enumerate(ops):
        for v in op[1:]:
            last_use[v] = i

    top = len(logQ) - 1
    level, live, peak_cts, peak_bytes_lvl = {}, set(), 0, 0
    N = 1 << logN
    lines = [f"PARAMS {logN} {','.join(map(str, logQ))} {','.join(map(str, logP))}"]
    counts = {"IN": 0, "MUL": 0, "ADD": 0, "RS": 0}

    def live_bytes():
        return sum(2 * N * (level[v] + 1) * 8 for v in live)

    for i, op in enumerate(ops):
        kind = op[0]
        if kind in counts:
            counts[kind] += 1
        if kind == "IN":
            level[op[1]] = top
            live.add(op[1])
        elif kind in ("MUL", "ADD"):
            level[op[1]] = min(level[op[2]], level[op[3]])
            live.add(op[1])
        elif kind == "RS":
            level[op[1]] -= 1
        peak_cts = max(peak_cts, len(live))
        peak_bytes_lvl = max(peak_bytes_lvl, live_bytes())
        lines.append(" ".join(map(str, op)))
        for v in sorted(set(op[1:])):
            if last_use[v] == i and v not in outputs and v in live:
                lines.append(f"FREE {v}")
                live.discard(v)

    open(trace_path, "w").write("\n".join(lines) + "\n")
    stats = {
        "logN": logN, "N": N, "L": len(logQ), "K": len(logP), "logQ": logQ, "logP": logP,
        "inputs": counts["IN"], "mul": counts["MUL"], "add": counts["ADD"], "rescale": counts["RS"],
        "outputs": len(outputs), "mult_depth": top - min(level[v] for v in outputs),
        "peak_live_ciphertexts": peak_cts,
        "peak_live_ct_bytes_level_aware": peak_bytes_lvl,
        "peak_live_ct_bytes_top_level": peak_cts * 2 * N * len(logQ) * 8,
    }
    json.dump(stats, open(json_path, "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
