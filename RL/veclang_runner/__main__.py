from veclang_runner.runner import VeclangRunner
from fhe_rl.utils import load_expressions
from enum import StrEnum
import csv


class Files(StrEnum):
    EXPRESSIONS_FILE = "fhe_rl/datasets/final_llm_dataset.txt"
    EXPRESSION_FILE = "veclang_runner/temp/expression.txt"
    STATS_FILE = "veclang_runner/temp/stats.csv"


COLUMNS = [
    "add",
    "sub",
    "multiply_plain",
    "rotate_rows",
    "negate",
    "multiply",
    "Depth",
    "Multiplicative Depth",
    "execution_time (s)",
    "Remaining_noise_budget",
]


if __name__ == "__main__":
    expressions = load_expressions(Files.EXPRESSIONS_FILE)
    with open(Files.STATS_FILE, mode="w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(COLUMNS)
        for expression in expressions:
            with open(Files.EXPRESSION_FILE, "w") as expression_file:
                expression_file.write(expression)
            runner = VeclangRunner(Files.EXPRESSION_FILE)
            try:
                runner.run()
            except Exception:
                continue
            writer.writerow(runner.stats.get(column) for column in COLUMNS)
            stats_file.flush()
