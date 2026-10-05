package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

// ============================================================
// BENCHMARK: FHE Execution Time Comparison
// Tests deep circuits WITH and WITHOUT bootstrapping
// ============================================================

type BenchmarkResult struct {
	Name       string
	KeyGen     time.Duration
	Encrypt    time.Duration
	FHEExec    time.Duration
	Decrypt    time.Duration
	Total      time.Duration
	NumMults   int
	NumBoots   int
}

func runPolyBenchmark(depth int, withBootstrap bool) BenchmarkResult {
	result := BenchmarkResult{
		Name:     fmt.Sprintf("poly_depth%d", depth),
		NumMults: depth,
	}

	// Parameters - enough levels for the depth
	levels := depth + 2
	if withBootstrap {
		levels = 12 // Fixed for bootstrap comparison
	}

	logQ := make([]int, levels+1)
	logQ[0] = 55
	for i := 1; i <= levels; i++ {
		logQ[i] = 40
	}

	paramsLit := hefloat.ParametersLiteral{
		LogN:            16, // Required for bootstrap
		LogQ:            logQ,
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(paramsLit)
	if err != nil {
		panic(err)
	}

	// Key generation
	keyGenStart := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	result.KeyGen = time.Since(keyGenStart)

	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Bootstrap setup if needed
	var bootstrapper *bootstrapping.Evaluator
	if withBootstrap && depth > levels {
		btpParamsLit := bootstrapping.ParametersLiteral{
			LogN:     utils.Pointy(params.LogN()),
			LogP:     []int{61, 61, 61, 61},
			Xs:       ring.Ternary{H: 192},
			LogSlots: utils.Pointy(15),
		}
		btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParamsLit)
		if err != nil {
			panic(err)
		}
		btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
		if err != nil {
			panic(err)
		}
		bootstrapper, err = bootstrapping.NewEvaluator(btpParams, btpKeys)
		if err != nil {
			panic(err)
		}
	}

	// Encryption
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = 0.9
	}

	encStart := time.Now()
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	_ = encoder.Encode(values, pt)
	ct, _ := enc.EncryptNew(pt)
	result.Encrypt = time.Since(encStart)

	// FHE Execution
	fheStart := time.Now()
	currentLevel := params.MaxLevel()
	bootsUsed := 0

	for i := 0; i < depth; i++ {
		ct, _ = eval.MulRelinNew(ct, ct)
		_ = eval.Rescale(ct, ct)
		currentLevel--

		// Bootstrap if level too low
		if withBootstrap && bootstrapper != nil && currentLevel <= 1 && i < depth-1 {
			ct, _ = bootstrapper.Bootstrap(ct)
			currentLevel = params.MaxLevel()
			bootsUsed++
		}
	}
	result.FHEExec = time.Since(fheStart)
	result.NumBoots = bootsUsed

	// Decryption
	decStart := time.Now()
	resultPt := dec.DecryptNew(ct)
	resultValues := make([]float64, params.MaxSlots())
	encoder.Decode(resultPt, resultValues)
	result.Decrypt = time.Since(decStart)

	result.Total = result.KeyGen + result.Encrypt + result.FHEExec + result.Decrypt

	return result
}

func printResult(r BenchmarkResult) {
	fmt.Printf("\n%-20s | Mults: %2d | Boots: %d\n", r.Name, r.NumMults, r.NumBoots)
	fmt.Printf("  KeyGen:   %10v\n", r.KeyGen.Round(time.Millisecond))
	fmt.Printf("  Encrypt:  %10v\n", r.Encrypt.Round(time.Millisecond))
	fmt.Printf("  FHE Exec: %10v  <-- Main comparison metric\n", r.FHEExec.Round(time.Millisecond))
	fmt.Printf("  Decrypt:  %10v\n", r.Decrypt.Round(time.Millisecond))
	fmt.Printf("  TOTAL:    %10v\n", r.Total.Round(time.Millisecond))
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║     CHEHAB Generated Code - FHE Execution Time Benchmark         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	results := []BenchmarkResult{}

	// Without bootstrap (enough levels)
	fmt.Println("\n=== WITHOUT BOOTSTRAP (sufficient levels) ===")
	for _, depth := range []int{4, 8, 10} {
		r := runPolyBenchmark(depth, false)
		printResult(r)
		results = append(results, r)
	}

	// With bootstrap (limited levels)
	fmt.Println("\n=== WITH BOOTSTRAP (limited to 12 levels) ===")
	for _, depth := range []int{15, 20, 25} {
		r := runPolyBenchmark(depth, true)
		printResult(r)
		results = append(results, r)
	}

	// Summary table
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        SUMMARY TABLE                             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Printf("\n%-15s | %6s | %5s | %12s | %12s\n", "Benchmark", "Mults", "Boots", "FHE Time", "Total")
	fmt.Println(string(make([]byte, 65)))
	for _, r := range results {
		fmt.Printf("%-15s | %6d | %5d | %12v | %12v\n",
			r.Name, r.NumMults, r.NumBoots,
			r.FHEExec.Round(time.Millisecond),
			r.Total.Round(time.Millisecond))
	}
}

