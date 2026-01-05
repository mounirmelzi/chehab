from argparse import ArgumentParser
from pathlib import Path
from pytrs import parse_sexpr, Expr, Const, Var, Op


GENERATION_FOLDER = Path("veclang_runner", "temp")


def calc_vec_sizes(expr: Expr):
    vec_sizes = []

    def rec(node: Expr):
        if isinstance(node, (Const, Var)):
            return
        if isinstance(node, Op):
            if node.op == "Vec":
                vec_sizes.append(len(node.args))
            else:
                for arg in node.args:
                    rec(arg)

    rec(expr)
    return vec_sizes


if __name__ == "__main__":
    parser = ArgumentParser(description="veclang expression runner")
    parser.add_argument(
        "--veclang_expression_file",
        help="Input expression file",
        required=True,
    )

    args = parser.parse_args()
    expression_file = Path(args.veclang_expression_file)
    print(f"'{expression_file.resolve()}': {expression_file.exists()}")
