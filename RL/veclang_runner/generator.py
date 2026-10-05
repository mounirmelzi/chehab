from __future__ import annotations

import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from pytrs import parse_sexpr, Expr, Var, Op
from fhe_rl.utils import load_expressions, calc_vec_sizes


GENERATION_FOLDER = Path("veclang_runner", "temp")
IS_CIPHER = 1
IS_SIGNED = 1


VECTORIZED_CODE = "vectorized_code.txt"
INPUTS = "inputs.txt"
FHE_IO_EXAMPLE = "fhe_io_example.txt"


def resolve_inputs(node: Expr, inputs: set[str]):
    if isinstance(node, Var):
        inputs.add(node.name)
        return

    if isinstance(node, Op):
        for arg in node.args:
            resolve_inputs(arg, inputs)


if __name__ == "__main__":
    parser = ArgumentParser(description="veclang expression runner")
    parser.add_argument(
        "--veclang_expression_file",
        help="Input expression file",
        required=True,
    )

    args = parser.parse_args()
    expression_file = Path(args.veclang_expression_file)
    expressions = load_expressions(expression_file)

    print(f"[generator] num expressions: {len(expressions)}", flush=True)
    if not expressions:
        print("[generator] ERROR: no expressions loaded", flush=True)
        sys.exit(1)

    expression_str = expressions[0]
    print(f"[generator] expression_str length: {len(expression_str)}", flush=True)
    print(f"[generator] expression_str[:80]: {expression_str[:80]}", flush=True)
    expression_parsed = parse_sexpr(expression_str)
    print(f"[generator] parse_sexpr succeeded, type: {type(expression_parsed).__name__}", flush=True)

    inputs: set[str] = set()
    resolve_inputs(expression_parsed, inputs)
    print(f"[generator] inputs: {inputs}", flush=True)

    # Generate vectorized_code.txt
    vec_sizes_list = calc_vec_sizes(expression_parsed)
    print(f"[generator] vec_sizes_list: {vec_sizes_list}", flush=True)
    vec_sizes = " ".join(str(x) for x in vec_sizes_list)

    GENERATION_FOLDER.mkdir(parents=True, exist_ok=True)

    out_path = GENERATION_FOLDER / VECTORIZED_CODE
    print(f"[generator] writing to: {out_path.resolve()}", flush=True)
    slot_count = vec_sizes_list[0] if vec_sizes_list else 1
    metadata = f"{slot_count} {slot_count} 1 0"
    with open(out_path, "w") as file:
        content = f"{expression_str}\n{metadata}"
        file.write(content)
        file.flush()
    print(f"[generator] wrote {len(content)} bytes to vectorized_code.txt", flush=True)
    print(f"[generator] metadata line: {metadata}", flush=True)

    import os
    actual_size = os.path.getsize(out_path)
    print(f"[generator] verify: file size on disk = {actual_size} bytes", flush=True)

    # Generate inputs.txt
    with open(GENERATION_FOLDER / INPUTS, "w") as file:
        file.writelines(
            [
                " ".join(inputs),
                "\n",
                " ".join(str(IS_CIPHER) * len(inputs)),
                "\n",
            ]
        )

    # Generate fhe_io_example.txt
    with open(GENERATION_FOLDER / FHE_IO_EXAMPLE, "w") as file:
        slot_count, nb_inputs, nb_outputs = 1, len(inputs), 1
        file.write(f"{slot_count} {nb_inputs} {nb_outputs}\n")
        file.writelines(
            f"{i} {IS_CIPHER} {IS_SIGNED} {random.randint(0, 10)}\n" for i in inputs
        )
