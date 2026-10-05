package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{}
}

func fhe(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c9 := encryptedInputs["v2_3"]
	c8 := encryptedInputs["v1_3"]
	c7 := encryptedInputs["v2_2"]
	c6 := encryptedInputs["v1_2"]
	c5 := encryptedInputs["v2_1"]
	c4 := encryptedInputs["v1_1"]
	c3 := encryptedInputs["v2_0"]
	c2 := encryptedInputs["v1_0"]

	// FHE Operations
	c2, _ = eval.MulRelinNew(c2, c3)
	_ = eval.Rescale(c2, c2)
	_ = eval.Relinearize(c2, c2)
	c4, _ = eval.MulRelinNew(c4, c5)
	_ = eval.Rescale(c4, c4)
	_ = eval.Relinearize(c4, c4)
	c2, _ = eval.AddNew(c2, c4)
	c6, _ = eval.MulRelinNew(c6, c7)
	_ = eval.Rescale(c6, c6)
	_ = eval.Relinearize(c6, c6)
	c2, _ = eval.AddNew(c2, c6)
	c8, _ = eval.MulRelinNew(c8, c9)
	_ = eval.Rescale(c8, c8)
	_ = eval.Relinearize(c8, c8)
	c2, _ = eval.AddNew(c2, c8)

	// Store outputs
	encryptedOutputs["output"] = c2
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=14 (n=16384, slots=8192)
	// MaxLevel=3, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois keys for rotations
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galoisElements, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Encoder, Encryptor, Decryptor, Evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Input/Output maps
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// ============================================
	// Test: dot_product(v1, v2) = sum(v1[i] * v2[i])
	// v1 = [1.0, 2.0, 3.0, 4.0]
	// v2 = [5.0, 6.0, 7.0, 8.0]
	// Expected: 1*5 + 2*6 + 3*7 + 4*8 = 70
	// ============================================
	
	v1 := []float64{1.0, 2.0, 3.0, 4.0}
	v2 := []float64{5.0, 6.0, 7.0, 8.0}
	
	// Helper to encrypt a single value
	encryptValue := func(value float64) *rlwe.Ciphertext {
		values := make([]float64, params.MaxSlots())
		for i := range values {
			values[i] = value
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(values, pt)
		ct, _ := enc.EncryptNew(pt)
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
	
	fmt.Println("=== CHEHAB Dot Product Test ===")
	fmt.Printf("v1 = %v\n", v1)
	fmt.Printf("v2 = %v\n", v2)
	fmt.Printf("Expected result: %.1f\n\n", 1.0*5.0+2.0*6.0+3.0*7.0+4.0*8.0)

	// Run computation
	fhe(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

	// Decrypt and print results
	for name, ct := range encryptedOutputs {
		pt := dec.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		encoder.Decode(pt, values)
		fmt.Printf("%s: [%.4f, %.4f, %.4f, ...]\n", name, values[0], values[1], values[2])
	}

	_ = encodedOutputs
	fmt.Println("CKKS computation completed!")
}
