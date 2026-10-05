import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Union
from .expr import Expr, Op
from .parser import parse_sexpr
from .cost import get_multiplicative_depth, get_normal_depth


class NoiseEstimator:
    def __init__(
        self,
        stats_dataset_path: Path = Path(
            "./fhe_rl/datasets/noise_estimator_stats_dataset.csv"
        ),
        noise_budget: int = 369,
    ):
        df = pd.read_csv(stats_dataset_path)
        df["used_noise_budget"] = noise_budget - df["Remaining_noise_budget"]
        y = df["used_noise_budget"]
        x = df[
            [
                "add",
                "sub",
                "multiply_plain",
                "rotate_rows",
                "negate",
                "multiply",
                "Depth",
                "Multiplicative Depth",
            ]
        ]
        self.model = LinearRegression(fit_intercept=False, positive=True)
        self.model.fit(x, y)

    def estimate(
        self,
        expr: Union[str, Expr],
    ):
        if isinstance(expr, str):
            expr = parse_sexpr(expr)
        return self._estimate(
            **self._count_operations(expr),
            Depth=get_normal_depth(expr),
            DepthMultiplicative_Depth=get_multiplicative_depth(expr),
        )

    def _estimate(
        self,
        add=0.0,
        sub=0.0,
        multiply_plain=0.0,
        rotate_rows=0.0,
        negate=0.0,
        multiply=0.0,
        Depth=0.0,
        DepthMultiplicative_Depth=0.0,
    ):
        return self.model.predict(
            pd.DataFrame(
                {
                    "add": [add],
                    "sub": [sub],
                    "multiply_plain": [multiply_plain],
                    "rotate_rows": [rotate_rows],
                    "negate": [negate],
                    "multiply": [multiply],
                    "Depth": [Depth],
                    "Multiplicative Depth": [DepthMultiplicative_Depth],
                }
            )
        )[0]

    @staticmethod
    def _count_operations(expr: Expr):
        ops_count = {
            "add": 0,
            "sub": 0,
            "multiply_plain": 0,
            "rotate_rows": 0,
            "negate": 0,
            "multiply": 0,
        }

        ops_matches = {
            "+": "add",
            "Add": "add",
            "-": "sub",
            "Minus": "sub",
            "*": "multiply",
            "Mul": "multiply",
            "Neg": "negate",
            "<<": "rotate_rows",
            "Vec": None,
            "VecAdd": "add",
            "VecMinus": "sub",
            "VecMul": "multiply",
            "VecNeg": "negate",
        }

        def dfs(node: Expr):
            if isinstance(node, Op):
                if op_name := ops_matches.get(node.op):
                    ops_count[op_name] += 1
                for child in node.args:
                    dfs(child)

        dfs(expr)
        return ops_count
