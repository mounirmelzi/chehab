package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// ============================================================================
// Test harness for CHEHAB-generated Lattigo code
// ============================================================================

func main() {
	fmt.Println("=== CHEHAB Lattigo Test Runner ===")
	fmt.Println()

	// Test dot product
	testDotProduct()
}

func testDotProduct() {
	fmt.Println("--- Testing Dot Product (4 elements) ---")

	// CKKS Parameters (same as generated)
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Parameters: LogN=%d, MaxLevel=%d, Slots=%d\n",
		params.LogN(), params.MaxLevel(), params.MaxSlots())

	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// No rotations needed for this simple dot product
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Create encoder, encryptor, decryptor, evaluator
	encoder := hefloat.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	evaluator := hefloat.NewEvaluator(params, evk)

	// ========================================
	// Prepare test inputs
	// ========================================
	// dot_product: sum(v1[i] * v2[i]) for i in 0..3
	// v1 = [1.0, 2.0, 3.0, 4.0]
	// v2 = [5.0, 6.0, 7.0, 8.0]
	// Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

	v1 := []float64{1.0, 2.0, 3.0, 4.0}
	v2 := []float64{5.0, 6.0, 7.0, 8.0}

	// Expected result (cleartext computation)
	expected := 0.0
	for i := 0; i < 4; i++ {
		expected += v1[i] * v2[i]
	}
	fmt.Printf("\nInput v1: %v\n", v1)
	fmt.Printf("Input v2: %v\n", v2)
	fmt.Printf("Expected result: %.4f\n", expected)

	// Encrypt each element separately (as per generated code structure)
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// Helper function to encrypt a single value (fills all slots with same value)
	encryptValue := func(value float64) *rlwe.Ciphertext {
		values := make([]float64, params.MaxSlots())
		for i := range values {
			values[i] = value
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(values, pt)
		ct, _ := encryptor.EncryptNew(pt)
		return ct
	}

	// Encrypt v1 elements
	for i := 0; i < 4; i++ {
		name := fmt.Sprintf("v1_%d", i)
		encryptedInputs[name] = encryptValue(v1[i])
	}

	// Encrypt v2 elements
	for i := 0; i < 4; i++ {
		name := fmt.Sprintf("v2_%d", i)
		encryptedInputs[name] = encryptValue(v2[i])
	}

	fmt.Println("\n--- Running FHE computation ---")
	start := time.Now()

	// ========================================
	// Call the generated function (inline here for testing)
	// This is equivalent to the generated fhe() function
	// ========================================
	c9 := encryptedInputs["v2_3"]
	c8 := encryptedInputs["v1_3"]
	c7 := encryptedInputs["v2_2"]
	c6 := encryptedInputs["v1_2"]
	c5 := encryptedInputs["v2_1"]
	c4 := encryptedInputs["v1_1"]
	c3 := encryptedInputs["v2_0"]
	c2 := encryptedInputs["v1_0"]

	// FHE Operations (from generated code)
	c2, _ = evaluator.MulRelinNew(c2, c3)
	_ = evaluator.Rescale(c2, c2)
	_ = evaluator.Relinearize(c2, c2)

	c4, _ = evaluator.MulRelinNew(c4, c5)
	_ = evaluator.Rescale(c4, c4)
	_ = evaluator.Relinearize(c4, c4)

	c2, _ = evaluator.AddNew(c2, c4)

	c6, _ = evaluator.MulRelinNew(c6, c7)
	_ = evaluator.Rescale(c6, c6)
	_ = evaluator.Relinearize(c6, c6)

	c2, _ = evaluator.AddNew(c2, c6)

	c8, _ = evaluator.MulRelinNew(c8, c9)
	_ = evaluator.Rescale(c8, c8)
	_ = evaluator.Relinearize(c8, c8)

	c2, _ = evaluator.AddNew(c2, c8)

	// Store output
	encryptedOutputs["output"] = c2

	elapsed := time.Since(start)
	fmt.Printf("FHE computation time: %v\n", elapsed)

	// ========================================
	// Decrypt and verify result
	// ========================================
	fmt.Println("\n--- Decrypting results ---")

	for name, ct := range encryptedOutputs {
		pt := decryptor.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		encoder.Decode(pt, values)

		// All slots should have the same value
		result := values[0]
		error_pct := math.Abs(result-expected) / math.Abs(expected) * 100

		fmt.Printf("\nOutput '%s':\n", name)
		fmt.Printf("  Decrypted value: %.6f\n", result)
		fmt.Printf("  Expected value:  %.6f\n", expected)
		fmt.Printf("  Absolute error:  %.6f\n", math.Abs(result-expected))
		fmt.Printf("  Relative error:  %.4f%%\n", error_pct)

		if error_pct < 1.0 {
			fmt.Println("  ✅ TEST PASSED!")
		} else {
			fmt.Println("  ❌ TEST FAILED (error > 1%)")
		}
	}

	_ = encodedInputs
	_ = encodedOutputs

	fmt.Println("\n=== Test Complete ===")
}

