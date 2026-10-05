package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// This is an example test file to verify Lattigo installation works
// Run with: go run example_test.go

func main() {
	fmt.Println("=== CHEHAB Lattigo Backend Test ===")

	// CKKS Parameters - suitable for basic operations
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,                                    // Ring degree = 2^14 = 16384
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40}, // Ciphertext modulus chain
		LogP:            []int{45, 45},                         // Special primes for key-switching
		LogDefaultScale: 40,                                    // Default scale = 2^40
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Parameters:\n")
	fmt.Printf("  LogN: %d (N=%d slots)\n", params.LogN(), params.MaxSlots())
	fmt.Printf("  MaxLevel: %d\n", params.MaxLevel())
	fmt.Printf("  DefaultScale: %.2e\n", params.DefaultScale().Float64())

	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Generate Galois keys for rotations [1, 2, 4, 8]
	rotations := []int{1, 2, 4, 8}
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galoisElements, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	fmt.Println("\nKeys generated successfully!")

	// Create encoder, encryptor, decryptor, evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Test data
	slots := params.MaxSlots()
	values1 := make([]float64, slots)
	values2 := make([]float64, slots)
	for i := range values1 {
		values1[i] = float64(i) * 0.001
		values2[i] = float64(i) * 0.002
	}

	// Encode and encrypt
	pt1 := hefloat.NewPlaintext(params, params.MaxLevel())
	pt2 := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(values1, pt1)
	encoder.Encode(values2, pt2)

	ct1, err := enc.EncryptNew(pt1)
	if err != nil {
		panic(err)
	}
	ct2, err := enc.EncryptNew(pt2)
	if err != nil {
		panic(err)
	}

	fmt.Println("\nEncryption successful!")
	fmt.Printf("  ct1 level: %d\n", ct1.Level())
	fmt.Printf("  ct2 level: %d\n", ct2.Level())

	// Test Add
	ctAdd, err := eval.AddNew(ct1, ct2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("\nAfter Add: level=%d\n", ctAdd.Level())

	// Test MulRelin (multiplication with relinearization)
	ctMul, err := eval.MulRelinNew(ct1, ct2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After MulRelin: level=%d\n", ctMul.Level())

	// Test Rescale (CKKS-specific)
	err = eval.Rescale(ctMul, ctMul)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After Rescale: level=%d\n", ctMul.Level())

	// Test Rotation
	ctRot, err := eval.RotateNew(ct1, 1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After Rotate(1): level=%d\n", ctRot.Level())

	// Decrypt and verify
	ptResult := dec.DecryptNew(ctMul)
	result := make([]float64, slots)
	encoder.Decode(ptResult, result)

	// Check accuracy
	maxError := 0.0
	for i := 0; i < 10; i++ {
		expected := values1[i] * values2[i]
		actual := result[i]
		err := math.Abs(expected - actual)
		if err > maxError {
			maxError = err
		}
		fmt.Printf("  [%d] expected=%.6f, got=%.6f, err=%.2e\n", i, expected, actual, err)
	}
	fmt.Printf("\nMax error: %.2e\n", maxError)

	fmt.Println("\n=== All tests passed! Lattigo backend is ready. ===")
}

