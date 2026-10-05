package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func deep_poly_depth8(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	eval *hefloat.Evaluator,
) {
	c1 := encryptedInputs["x"]

	// 8 successive squarings: x -> x^256
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)

	encryptedOutputs["result"] = c1
}

func main() {
	fmt.Println("==============================================")
	fmt.Println("CHEHAB Generated Code - Depth 8 Benchmark")
	fmt.Println("==============================================")

	// CKKS Parameters (same as Orion)
	paramsLit := hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(paramsLit)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Parameters: LogN=%d, Levels=%d\n", params.LogN(), params.MaxLevel())

	// Key generation timing
	keyGenStart := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	keyGenTime := time.Since(keyGenStart)
	fmt.Printf("Key generation: %v\n", keyGenTime)

	// Create objects
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Encrypt input
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = 0.5
	}

	encStart := time.Now()
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(values, pt); err != nil {
		panic(err)
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	encTime := time.Since(encStart)
	fmt.Printf("Encryption: %v\n", encTime)

	// Run FHE computation
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encryptedInputs["x"] = ct

	fmt.Println("\nRunning FHE computation (x^256)...")
	fheStart := time.Now()
	deep_poly_depth8(encryptedInputs, encryptedOutputs, eval)
	fheTime := time.Since(fheStart)
	fmt.Printf("FHE computation time: %v\n", fheTime)

	// Decrypt and verify
	decStart := time.Now()
	resultPt := dec.DecryptNew(encryptedOutputs["result"])
	resultValues := make([]float64, params.MaxSlots())
	encoder.Decode(resultPt, resultValues)
	decTime := time.Since(decStart)
	fmt.Printf("Decryption: %v\n", decTime)

	// Results
	fmt.Printf("\n=== Results ===\n")
	fmt.Printf("FHE result: %.10f\n", resultValues[0])
	fmt.Printf("Expected:   %.10f (0.5^256)\n", 0.0) // 0.5^256 ≈ 0
	fmt.Printf("\nTotal FHE time: %v\n", fheTime)
	fmt.Printf("Total E2E time: %v\n", keyGenTime+encTime+fheTime+decTime)
}

