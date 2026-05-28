#!/usr/bin/env python3
"""
Prepare a generated Lattigo Go file for instrumented benchmarking.

Replaces the generated main() function with an instrumented version that
reports separated timing for keygen, encrypt, eval, and decrypt.

Usage:
    python3 run_instrumented.py <generated_fhe.go> [benchmark_name] [text|csv|json]
"""
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_input_names(go_source: str) -> list:
    """Extract all input variable names from encryptedInputs[\"...\"] accesses."""
    return list(set(re.findall(r'encryptedInputs\["([^"]+)"\]', go_source)))


def extract_fhe_func_name(go_source: str) -> str:
    """Extract the FHE computation function name (not main, not getRotationSteps)."""
    funcs = re.findall(r'^func\s+(\w+)\s*\(', go_source, re.MULTILINE)
    for f in funcs:
        if f not in ('main', 'getRotationSteps'):
            return f
    return 'fhe'


def strip_main(go_source: str) -> str:
    """Remove the main() function body from Go source."""
    lines = go_source.split('\n')
    result = []
    in_main = False
    brace_depth = 0

    for line in lines:
        if not in_main:
            if re.match(r'^func\s+main\s*\(\s*\)', line):
                in_main = True
                brace_depth = 0
                for ch in line:
                    if ch == '{':
                        brace_depth += 1
                continue
            result.append(line)
        else:
            for ch in line:
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
            if brace_depth <= 0:
                in_main = False

    return '\n'.join(result)


def build_instrumented_main(input_names: list, func_name: str = "fhe") -> str:
    """Build an instrumented main() that creates only the needed inputs."""
    encrypt_lines = []
    for name in sorted(input_names):
        encrypt_lines.append(f'\tencryptedInputs["{name}"] = encryptSingle(inputVal)')

    return '''
func main() {{
	benchName := "fhe_benchmark"
	if len(os.Args) > 1 {{
		benchName = os.Args[1]
	}}
	outputFormat := "text"
	if len(os.Args) > 2 {{
		outputFormat = os.Args[2]
	}}

	logN := 14
	maxLevel := 5
	logQ := make([]int, maxLevel+1)
	logQ[0] = 55
	for i := 1; i <= maxLevel; i++ {{
		logQ[i] = 40
	}}

	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            []int{{45, 45}},
		LogDefaultScale: 40,
	}})
	if err != nil {{
		panic(err)
	}}

	// --- Key generation ---
	keygenStart := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {{
		galoisElements[i] = params.GaloisElement(r)
	}}
	var gks []*rlwe.GaloisKey
	if len(galoisElements) > 0 {{
		gks = kgen.GenGaloisKeysNew(galoisElements, sk)
	}}
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	keygenMs := float64(time.Since(keygenStart).Microseconds()) / 1000.0

	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// --- Encryption ---
	encryptStart := time.Now()
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	inputVal := 0.95
	encryptSingle := func(value float64) *rlwe.Ciphertext {{
		values := make([]float64, params.MaxSlots())
		for i := range values {{
			values[i] = value
		}}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		_ = encoder.Encode(values, pt)
		ct, _ := enc.EncryptNew(pt)
		return ct
	}}

{encrypt_block}

	encryptMs := float64(time.Since(encryptStart).Microseconds()) / 1000.0

	// --- Eval ---
	evalStart := time.Now()
	{func_name}(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)
	evalMs := float64(time.Since(evalStart).Microseconds()) / 1000.0

	// --- Decrypt ---
	decryptStart := time.Now()
	var decVal float64
	for _, ct := range encryptedOutputs {{
		pt := dec.DecryptNew(ct)
		vals := make([]float64, params.MaxSlots())
		_ = encoder.Decode(pt, vals)
		decVal = vals[0]
	}}
	decryptMs := float64(time.Since(decryptStart).Microseconds()) / 1000.0

	totalMs := keygenMs + encryptMs + evalMs + decryptMs
	absErr := math.Abs(decVal - inputVal)
	precBits := 53.0
	if absErr > 0 {{
		precBits = -math.Log2(absErr)
	}}

	switch outputFormat {{
	case "json":
		fmt.Printf("{{\\"name\\":\\"%s\\",\\"keygen_ms\\":%.1f,\\"encrypt_ms\\":%.1f,\\"eval_ms\\":%.1f,\\"bootstrap_ms\\":0,\\"decrypt_ms\\":%.1f,\\"total_ms\\":%.1f,\\"precision_bits\\":%.2f,\\"abs_error\\":%.2e}}\\n",
			benchName, keygenMs, encryptMs, evalMs, decryptMs, totalMs, precBits, absErr)
	case "csv":
		fmt.Println("name,keygen_ms,encrypt_ms,eval_ms,bootstrap_ms,decrypt_ms,total_ms,precision_bits,abs_error")
		fmt.Printf("%s,%.1f,%.1f,%.1f,0,%.1f,%.1f,%.2f,%.2e\\n",
			benchName, keygenMs, encryptMs, evalMs, decryptMs, totalMs, precBits, absErr)
	default:
		fmt.Printf("keygen_ms: %.1f\\n", keygenMs)
		fmt.Printf("encrypt_ms: %.1f\\n", encryptMs)
		fmt.Printf("eval_ms: %.1f\\n", evalMs)
		fmt.Printf("bootstrap_ms: 0\\n")
		fmt.Printf("decrypt_ms: %.1f\\n", decryptMs)
		fmt.Printf("total_ms: %.1f\\n", totalMs)
		fmt.Printf("precision_bits: %.2f\\n", precBits)
		fmt.Printf("abs_error: %.2e\\n", absErr)
	}}

	_ = encodedOutputs
	_ = encodedInputs
}}
'''.format(encrypt_block='\n'.join(encrypt_lines), func_name=func_name)


REQUIRED_IMPORTS = '''import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)
'''


def main():
    if len(sys.argv) < 2:
        print("Usage: run_instrumented.py <generated_fhe.go> [name] [text|csv|json]")
        sys.exit(1)

    generated_file = Path(sys.argv[1]).resolve()
    bench_name = sys.argv[2] if len(sys.argv) > 2 else generated_file.stem
    output_fmt = sys.argv[3] if len(sys.argv) > 3 else "text"

    script_dir = Path(__file__).resolve().parent

    if not generated_file.exists():
        print(f"Error: {generated_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(generated_file) as f:
        source = f.read()

    input_names = extract_input_names(source)
    func_name = extract_fhe_func_name(source)
    print(f"[info] Found {len(input_names)} input variables: {sorted(input_names)}", file=sys.stderr)
    print(f"[info] FHE function: {func_name}", file=sys.stderr)

    stripped = strip_main(source)
    # Replace import block
    stripped = re.sub(r'import\s*\([^)]*\)', REQUIRED_IMPORTS.strip(), stripped, count=1)
    stripped = re.sub(r'import\s+"[^"]+"\s*\n', '', stripped)

    instrumented = stripped + build_instrumented_main(input_names, func_name)

    out_file = script_dir / "instrumented_run.go"
    try:
        with open(out_file, "w") as f:
            f.write(instrumented)

        env = os.environ.copy()
        env["GOMODCACHE"] = str(Path.home() / "go" / "pkg" / "mod")
        cmd = ["go", "run", "instrumented_run.go", bench_name, output_fmt]
        result = subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"Build/run failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        print(result.stdout, end="")
    finally:
        out_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
