import random
from argparse import ArgumentParser
from pathlib import Path
from enum import StrEnum
from pytrs import parse_sexpr, Expr, Var, Op
from fhe_rl.utils import load_expressions, calc_vec_sizes


GENERATION_FOLDER = Path("veclang_runner", "temp")
IS_CIPHER = 1
IS_SIGNED = 1


class Files(StrEnum):
    VECTORIZED_CODE = "vectorized_code.txt"
    INPUTS = "inputs.txt"
    FHE_IO_EXAMPLE = "fhe_io_example.txt"


def resolve_inputs(node: Expr, inputs: list[str]):
    if isinstance(node, Var):
        name = node.name
        inputs.append(name)
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

    expression_str = expressions[0]
    expression_parsed = parse_sexpr(expression_str)

    inputs: list[str] = []
    resolve_inputs(expression_parsed, inputs)

    # Generate vectorized_code.txt
    vec_sizes = " ".join(str(x) for x in calc_vec_sizes(expression_parsed))
    with open(GENERATION_FOLDER / Files.VECTORIZED_CODE, "w") as file:
        file.write(f"{expression_str}\n{vec_sizes}")

    # Generate inputs.txt
    with open(GENERATION_FOLDER / Files.INPUTS, "w") as file:
        file.writelines(
            [
                " ".join(inputs),
                "\n",
                " ".join(str(IS_CIPHER) * len(inputs)),
                "\n",
            ]
        )

    # Generate fhe_io_example.txt
    with open(GENERATION_FOLDER / Files.FHE_IO_EXAMPLE, "w") as file:
        slot_count, nb_inputs, nb_outputs = 1, len(inputs), 1
        file.write(f"{slot_count} {nb_inputs} {nb_outputs}\n")
        file.writelines(
            f"{i} {IS_CIPHER} {IS_SIGNED} {random.randint(0, 10)}\n" for i in inputs
        )
