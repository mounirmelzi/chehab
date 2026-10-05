import os
import re
import shutil
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
    LATTIGO_OPERATIONS = ["AddNew", "SubNew", "MulRelinNew", "MulNew", "RotateNew", "NegNew", "Rescale", "Bootstrap"]

    def __init__(self, expression_file_path: str, backend: str = "seal"):
        self._expression: str = expression_file_path
        self._backend: str = backend
        self._stats: dict = {}

    @property
    def stats(self) -> dict:
        return self._stats

    def run(self, *, cse=1, const_folding=1) -> None:
        if self._backend == "lattigo":
            self._run_lattigo(cse=cse, const_folding=const_folding)
        else:
            self._run_seal(cse=cse, const_folding=const_folding)

    def _run_seal(self, *, cse=1, const_folding=1) -> None:
        cwd = os.getcwd()
        try:
            run_command(
                f"python -m veclang_runner.generator --veclang_expression_file {self._expression}"
            )

            os.chdir("..")

            run_command("cmake -S . -B build")

            os.chdir("build")

            run_command("make")

            os.chdir("RL/veclang_runner")

            result = run_command(f"./veclang_runner {cse} {const_folding} 0")
            self._parse_compiler_outputs(stdout=result.stdout)

            os.chdir("he")

            with open(self.GENERATED_CPP_FILE) as file:
                self._parse_generated_cpp_file(file_content=file.read())

            run_command("cmake -S . -B build")

            os.chdir("build")

            run_command("make")

            result = run_command("./main")
            self._parse_execution_outputs(result.stdout)
        finally:
            os.chdir(cwd)

    def _run_lattigo(self, *, cse=1, const_folding=1) -> None:
        cwd = os.getcwd()
        try:
            run_command(
                f"python -m veclang_runner.generator --veclang_expression_file {self._expression}"
            )

            os.chdir("..")
            run_command("cmake -S . -B build")
            os.chdir("build")
            run_command("make")

            os.chdir("RL/veclang_runner")

            result = run_command(f"./veclang_runner {cse} {const_folding} 1")
            self._parse_compiler_outputs(stdout=result.stdout)

            generated_go = Path("generated_fhe.go")
            if not generated_go.exists():
                raise FileNotFoundError("veclang_runner did not produce generated_fhe.go")

            lattigo_dir = Path(cwd) / ".." / "lattigo_backend"
            lattigo_dir = lattigo_dir.resolve()
            dest = lattigo_dir / "generated_fhe.go"
            shutil.copy2(generated_go, dest)

            with open(generated_go) as file:
                self._parse_generated_go_file(file_content=file.read())

            result = run_command("go run generated_fhe.go", path=lattigo_dir)
            self._parse_lattigo_outputs(result.stdout)
        finally:
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

    def _parse_lattigo_outputs(self, stdout: str) -> None:
        for line in stdout.splitlines():
            for key in ["keygen_ms", "encrypt_ms", "eval_ms", "bootstrap_ms",
                        "decrypt_ms", "total_ms", "precision_bits", "abs_error"]:
                if line.strip().startswith(key + ":"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        self._stats[key] = float(val)
                    except ValueError:
                        self._stats[key] = val

    def _parse_generated_cpp_file(self, file_content: str) -> None:
        for op in VeclangRunner.OPERATIONS:
            self._stats[op] = int(len(re.findall(rf"\b{op}", file_content)))

    def _parse_generated_go_file(self, file_content: str) -> None:
        for op in VeclangRunner.LATTIGO_OPERATIONS:
            self._stats[f"lattigo_{op}"] = len(re.findall(rf"\b{op}", file_content))
