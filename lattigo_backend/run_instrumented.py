#!/usr/bin/env python3
"""
Prepare a generated Lattigo Go file for instrumented benchmarking.

Replaces the generated main() function with an instrumented version that
reports separated timing for keygen, encrypt, eval, bootstrap-keygen, and decrypt.

Automatically parses CKKS parameters and bootstrap configuration from the
generated Go source so it works for both shallow and deep (bootstrap) circuits.

Usage:
    python3 run_instrumented.py <generated_fhe.go> [name] [text|csv|json] [expected_value]

    expected_value: float64 expected plaintext output for precision measurement.
                    If omitted, defaults to inputVal (0.95) which is only
                    correct for identity circuits.
"""
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_input_names(go_source: str) -> list:
    """Extract all input variable names from encryptedInputs[\"...\"] accesses."""
    return list(set(re.findall(r'encryptedInputs\["([^"]+)"\]', go_source)))


def extract_plain_input_names(go_source: str) -> list:
    """Extract all plaintext variable names from encodedInputs[\"...\"] accesses."""
    return list(set(re.findall(r'encodedInputs\["([^"]+)"\]', go_source)))


def parse_io_adapted_file(io_path: Path) -> dict:
    """Parse fhe_io_example_adapted.txt to get actual values for each plain input.

    Returns a dict mapping label -> list of float slot values for plaintext entries.
    """
    if not io_path.exists():
        return {}
    plain_values = {}
    with open(io_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return {}
    header = lines[0].split()
    if len(header) < 3:
        return {}
    nb_inputs = int(header[1])
    for i in range(1, min(nb_inputs + 1, len(lines))):
        tokens = lines[i].split()
        if len(tokens) < 4:
            continue
        label = tokens[0]
        input_type = int(tokens[1])
        if input_type != 0:
            continue
        vals = []
        for t in tokens[3:]:
            try:
                vals.append(float(t))
            except ValueError:
                vals = []
                break
        if vals:
            plain_values[label] = vals
    return plain_values


def extract_fhe_func_name(go_source: str) -> str:
    """Extract the FHE computation function name (not main, not getRotationSteps)."""
    funcs = re.findall(r'^func\s+(\w+)\s*\(', go_source, re.MULTILINE)
    for f in funcs:
        if f not in ('main', 'getRotationSteps'):
            return f
    return 'fhe'


def detect_bootstrap(go_source: str) -> bool:
    """Check if the generated Go file uses bootstrapping."""
    return 'bootstrapping.NewParametersFromLiteral' in go_source or \
           'bootstrapper.Bootstrap' in go_source


def extract_params_block(go_source: str) -> str:
    """Extract the hefloat.ParametersLiteral{...} block from main()."""
    m = re.search(
        r'(params\s*,\s*err\s*:=\s*hefloat\.NewParametersFromLiteral\(hefloat\.ParametersLiteral\{.*?\}\))',
        go_source, re.DOTALL
    )
    if m:
        return m.group(1)
    return None


def extract_bootstrap_block(go_source: str) -> str:
    """Extract the full bootstrap setup block from the generated main()."""
    lines = go_source.split('\n')
    result = []
    capturing = False
    for line in lines:
        if 'btpParamsLit' in line and ':=' in line:
            capturing = True
        if capturing:
            result.append(line)
            if 'Bootstrap keys generated successfully' in line or \
               (result and 'bootstrapper, err' in line and line.strip().startswith('bootstrapper')):
                pass
            if 'fmt.Println("Bootstrap keys generated' in line:
                capturing = False
    if not result:
        return None
    return '\n'.join(result)


def extract_bootstrap_setup(go_source: str) -> str:
    """Extract bootstrap setup from btpParamsLit declaration through bootstrapper creation.
    
    Handles both generated code (with fmt.Println) and hand-written code.
    """
    lines = go_source.split('\n')
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and 'btpParamsLit' in line and ':=' in line:
            start_idx = i
        if start_idx is not None and 'Bootstrap keys generated' in line:
            end_idx = i
            break
    if start_idx is not None and end_idx is None:
        for i in range(start_idx, len(lines)):
            if 'bootstrapper, err' in lines[i] and 'NewEvaluator' in lines[i]:
                brace_depth = 0
                for j in range(i, min(i + 6, len(lines))):
                    for ch in lines[j]:
                        if ch == '{':
                            brace_depth += 1
                        elif ch == '}':
                            brace_depth -= 1
                    if brace_depth <= 0 and j > i:
                        end_idx = j
                        break
                if end_idx is None:
                    end_idx = i
                break
    if start_idx is not None and end_idx is not None:
        return '\n'.join(lines[start_idx:end_idx + 1])
    return None


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


def build_instrumented_main(input_names: list, plain_names: list = None,
                            func_name: str = "fhe",
                            params_block: str = None,
                            has_bootstrap: bool = False,
                            bootstrap_setup: str = None,
                            plain_values: dict = None) -> str:
    """Build an instrumented main() with params parsed from the generated file."""
    encrypt_lines = []
    for name in sorted(input_names):
        encrypt_lines.append(f'\tencryptedInputs["{name}"] = encryptSingle(inputVal)')

    plain_values = plain_values or {}
    plain_lines = []
    for name in sorted(plain_names or []):
        if name in plain_values:
            vals = plain_values[name]
            if len(set(vals)) == 1:
                plain_lines.append(f'\tencodedInputs["{name}"] = encodeSingle({vals[0]})')
            else:
                vec_str = ", ".join(str(v) for v in vals)
                plain_lines.append(f'\tencodedInputs["{name}"] = encodeVector([]float64{{{vec_str}}})')
        else:
            plain_lines.append(f'\tencodedInputs["{name}"] = encodeSingle(inputVal)')

    if params_block is None:
        params_block = '''params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})'''

    bootstrap_keygen_code = ""
    if has_bootstrap and bootstrap_setup:
        bootstrap_keygen_code = f'''
	// --- Bootstrap key generation ---
	btpKeygenStart := time.Now()
{bootstrap_setup}
	btpKeygenMs = float64(time.Since(btpKeygenStart).Microseconds()) / 1000.0
'''
    elif has_bootstrap:
        bootstrap_keygen_code = "\tbtpKeygenMs = 0.0 // bootstrap detected but setup not parsed\n"

    btp_var_decl = "\tvar btpKeygenMs float64" if has_bootstrap else "\tbtpKeygenMs := 0.0"

    return f'''
func main() {{
	benchName := "fhe_benchmark"
	if len(os.Args) > 1 {{
		benchName = os.Args[1]
	}}
	outputFormat := "text"
	if len(os.Args) > 2 {{
		outputFormat = os.Args[2]
	}}
	expectedValOverride := math.NaN()
	if len(os.Args) > 3 {{
		if v, err := strconv.ParseFloat(os.Args[3], 64); err == nil {{
			expectedValOverride = v
		}}
	}}

	{params_block}
	if err != nil {{
		panic(err)
	}}

{btp_var_decl}

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
{bootstrap_keygen_code}
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
	encodeSingle := func(value float64) *rlwe.Plaintext {{
		values := make([]float64, params.MaxSlots())
		for i := range values {{
			values[i] = value
		}}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		_ = encoder.Encode(values, pt)
		return pt
	}}
	encodeVector := func(vec []float64) *rlwe.Plaintext {{
		values := make([]float64, params.MaxSlots())
		for i := range values {{
			values[i] = vec[i % len(vec)]
		}}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		_ = encoder.Encode(values, pt)
		return pt
	}}

{chr(10).join(encrypt_lines)}
{chr(10).join(plain_lines)}
	_ = encodeSingle
	_ = encodeVector

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

	totalMs := keygenMs + btpKeygenMs + encryptMs + evalMs + decryptMs

	expectedVal := inputVal
	if !math.IsNaN(expectedValOverride) {{
		expectedVal = expectedValOverride
	}}
	absErr := math.Abs(decVal - expectedVal)
	precBits := 53.0
	if absErr > 0 {{
		precBits = -math.Log2(absErr)
	}}

	switch outputFormat {{
	case "json":
		fmt.Printf("{{\\"name\\":\\"%s\\",\\"keygen_ms\\":%.1f,\\"bootstrap_keygen_ms\\":%.1f,\\"encrypt_ms\\":%.1f,\\"eval_ms\\":%.1f,\\"decrypt_ms\\":%.1f,\\"total_ms\\":%.1f,\\"precision_bits\\":%.2f,\\"abs_error\\":%.2e}}\\n",
			benchName, keygenMs, btpKeygenMs, encryptMs, evalMs, decryptMs, totalMs, precBits, absErr)
	case "csv":
		fmt.Println("name,keygen_ms,bootstrap_keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,precision_bits,abs_error")
		fmt.Printf("%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2e\\n",
			benchName, keygenMs, btpKeygenMs, encryptMs, evalMs, decryptMs, totalMs, precBits, absErr)
	default:
		fmt.Printf("keygen_ms: %.1f\\n", keygenMs)
		fmt.Printf("bootstrap_keygen_ms: %.1f\\n", btpKeygenMs)
		fmt.Printf("encrypt_ms: %.1f\\n", encryptMs)
		fmt.Printf("eval_ms: %.1f\\n", evalMs)
		fmt.Printf("decrypt_ms: %.1f\\n", decryptMs)
		fmt.Printf("total_ms: %.1f\\n", totalMs)
		fmt.Printf("precision_bits: %.2f\\n", precBits)
		fmt.Printf("abs_error: %.2e\\n", absErr)
	}}

	_ = encodedOutputs
	_ = encodedInputs
}}
'''


IMPORTS_NO_BOOTSTRAP = '''import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)
'''

IMPORTS_BOOTSTRAP = '''import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)
'''


def fix_unused_vars(source: str, unused_vars: set) -> str:
    """Add _ = varName after each unused variable declaration inside functions."""
    lines = source.split('\n')
    result = []
    for line in lines:
        result.append(line)
        for var in unused_vars:
            if re.match(rf'^\s+var\s+{re.escape(var)}\s+', line) or \
               re.match(rf'^\s+{re.escape(var)}\s*:=', line) or \
               re.match(rf'^\s+{re.escape(var)}\s*,', line):
                indent = line[:len(line) - len(line.lstrip())]
                result.append(f'{indent}_ = {var}')
                break
    return '\n'.join(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: run_instrumented.py <generated_fhe.go> [name] [text|csv|json]")
        sys.exit(1)

    generated_file = Path(sys.argv[1]).resolve()
    bench_name = sys.argv[2] if len(sys.argv) > 2 else generated_file.stem
    output_fmt = sys.argv[3] if len(sys.argv) > 3 else "text"
    expected_value = sys.argv[4] if len(sys.argv) > 4 else None

    script_dir = Path(__file__).resolve().parent

    if not generated_file.exists():
        print(f"Error: {generated_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(generated_file) as f:
        source = f.read()

    input_names = extract_input_names(source)
    plain_names = extract_plain_input_names(source)
    func_name = extract_fhe_func_name(source)
    has_bootstrap = detect_bootstrap(source)
    params_block = extract_params_block(source)
    bootstrap_setup = extract_bootstrap_setup(source) if has_bootstrap else None

    io_adapted = generated_file.parent / "fhe_io_example_adapted.txt"
    plain_values = parse_io_adapted_file(io_adapted)
    if plain_values:
        print(f"[info] Loaded {len(plain_values)} plaintext values from {io_adapted.name}", file=sys.stderr)
        for k, v in sorted(plain_values.items()):
            sample = v[:4]
            print(f"[info]   {k} = {sample}{'...' if len(v) > 4 else ''}", file=sys.stderr)

    print(f"[info] Found {len(input_names)} cipher inputs: {sorted(input_names)}", file=sys.stderr)
    print(f"[info] Found {len(plain_names)} plain inputs: {sorted(plain_names)}", file=sys.stderr)
    print(f"[info] FHE function: {func_name}", file=sys.stderr)
    print(f"[info] Bootstrap: {has_bootstrap}", file=sys.stderr)
    if params_block:
        first_line = params_block.split('\n')[0][:80]
        print(f"[info] Params: {first_line}...", file=sys.stderr)

    stripped = strip_main(source)

    imports = IMPORTS_BOOTSTRAP if has_bootstrap else IMPORTS_NO_BOOTSTRAP
    stripped = re.sub(r'import\s*\([^)]*\)', imports.strip(), stripped, count=1)
    stripped = re.sub(r'import\s+"[^"]+"\s*\n', '', stripped)

    instrumented = stripped + build_instrumented_main(
        input_names, plain_names, func_name,
        params_block, has_bootstrap, bootstrap_setup,
        plain_values=plain_values
    )

    out_file = script_dir / "instrumented_run.go"
    try:
        env = os.environ.copy()
        env["GOMODCACHE"] = str(Path.home() / "go" / "pkg" / "mod")
        env["CGO_ENABLED"] = "0"
        cmd = ["go", "run", "instrumented_run.go", bench_name, output_fmt]
        if expected_value is not None:
            cmd.append(expected_value)

        def try_build(source_code):
            with open(out_file, "w") as f:
                f.write(source_code)
            return subprocess.run(cmd, cwd=str(script_dir), capture_output=True, text=True, env=env)

        current_source = instrumented
        for attempt in range(5):
            result = try_build(current_source)
            if result.returncode == 0:
                break
            if "declared and not used" not in result.stderr:
                break
            unused_vars = set(re.findall(r'declared and not used: (\w+)', result.stderr))
            if not unused_vars:
                break
            print(f"[info] Pass {attempt+1}: fixing {len(unused_vars)} unused variables: {sorted(unused_vars)}", file=sys.stderr)
            current_source = fix_unused_vars(current_source, unused_vars)

        if result.returncode != 0:
            print(f"Build/run failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        print(result.stdout, end="")
    finally:
        out_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
