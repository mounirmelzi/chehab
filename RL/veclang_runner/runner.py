import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union


def run_command(
    command: str,
    *,
    path: Optional[Union[Path, str]] = None,
) -> subprocess.CompletedProcess:
    cwd = os.getcwd()
    if path:
        os.chdir(path)
    result = subprocess.run(command.split(), capture_output=True, check=True, text=True)
    os.chdir(cwd)
    return result


class VeclangRunner:
    GENERATED_CPP_FILE = "_gen_he_fhe.cpp"
    OPERATIONS = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply"]

    def __init__(self, expression_file_path: str):
        self._expression: str = expression_file_path
        self._stats: dict = {}

    @property
    def stats(self) -> dict:
        if self._stats:
            return self._stats
        self.run()
        return self._stats

    def run(self, *, cse=1, const_folding=1) -> None:
        cwd = os.getcwd()

        run_command(
            f"python -m veclang_runner.generator --veclang_expression_file {self._expression}"
        )

        os.chdir("..")

        run_command("cmake -S . -B build")

        os.chdir("build")

        run_command("make")

        os.chdir("RL/veclang_runner")

        result = run_command(f"./veclang_runner {cse} {const_folding}")
        self._parse_compiler_outputs(stdout=result.stdout)

        os.chdir("he")

        with open(self.GENERATED_CPP_FILE) as file:
            self._parse_generated_cpp_file(file_content=file.read())

        run_command("cmake -S . -B build")

        os.chdir("build")

        run_command("make")

        result = run_command("./main")
        self._parse_execution_outputs(result.stdout)

        os.chdir(cwd)

    def _parse_compiler_outputs(self, stdout: str) -> None:
        depth_match = re.search(r"max:\s*\((\d+),\s*(\d+)\)", stdout)
        self._stats["Depth"] = int(depth_match.group(1)) if depth_match else None
        self._stats["Multiplicative Depth"] = int(depth_match.group(2)) if depth_match else None

    def _parse_execution_outputs(self, stdout: str) -> None:
        for line in stdout.splitlines():
            if "execution_time_(ms):" in line:
                self._stats["execution_time (s)"] = format(float(line.split()[1]) / 1000, ".3f")
            elif "Remaining_noise_budget:" in line:
                self._stats["Remaining_noise_budget"] = int(line.split()[1])

    def _parse_generated_cpp_file(self, file_content: str) -> None:
        for op in VeclangRunner.OPERATIONS:
            self._stats[op] = int(len(re.findall(rf"\b{op}", file_content)))
