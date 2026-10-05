package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func benchmark_deep_poly(depth int) {
	fmt.Printf("\n=== Benchmark: x^%d (depth %d) ===\n", 1<<depth, depth)
	
	// CKKS Parameters
	numLevels := depth + 2
	logQ := make([]int, numLevels)
	logQ[0] = 55
	for i := 1; i < numLevels; i++ {
		logQ[i] = 40
	}
	
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            logQ,
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	// Key Generation
	startKeyGen := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	keyGenTime := time.Since(startKeyGen)
	
	// Create encoder, encryptor, etc.
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Encode and encrypt input
	x_val := 1.1
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = x_val
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(values, pt)
	
	startEnc := time.Now()
	ct, _ := enc.EncryptNew(pt)
	encTime := time.Since(startEnc)

	// Compute x^(2^depth)
	startComp := time.Now()
	result := ct
	for i := 0; i < depth; i++ {
		result, _ = eval.MulRelinNew(result, result)
		eval.Rescale(result, result)
	}
	compTime := time.Since(startComp)

	// Decrypt
	startDec := time.Now()
	ptResult := dec.DecryptNew(result)
	decValues := make([]float64, params.MaxSlots())
	encoder.Decode(ptResult, decValues)
	decTime := time.Since(startDec)

	// Expected result
	expected := 1.0
	for i := 0; i < (1 << depth); i++ {
		expected *= x_val
	}

	fmt.Printf("Parameters: LogN=%d, Levels=%d\n", params.LogN(), numLevels)
	fmt.Printf("Key generation: %v\n", keyGenTime)
	fmt.Printf("Encryption: %v\n", encTime)
	fmt.Printf("Computation: %v\n", compTime)
	fmt.Printf("Decryption: %v\n", decTime)
	fmt.Printf("Result: %.4f (expected %.4f)\n", decValues[0], expected)
	fmt.Printf("Error: %.2e\n", decValues[0]-expected)
}

func main() {
	// Test different depths
	for _, depth := range []int{5, 7, 10, 12} {
		benchmark_deep_poly(depth)
	}
}
