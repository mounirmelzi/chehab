package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func pow(base float64, exp int) float64 {
	return math.Pow(base, float64(exp))
}

func getRotationSteps() []int {
	return []int{}
}

func deep_poly(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c1 := encryptedInputs["x"]

	// FHE Operations
	c1, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c1, c1)
	var c12 *rlwe.Ciphertext
	c12 = c1.CopyNew()
	_ = eval.Relinearize(c12, c12)
	c12, _ = eval.MulRelinNew(c12, c12)
	_ = eval.Rescale(c12, c12)
	var c13 *rlwe.Ciphertext
	c13 = c12.CopyNew()
	_ = eval.Relinearize(c13, c13)
	c13, _ = eval.MulRelinNew(c13, c13)
	_ = eval.Rescale(c13, c13)
	var c14 *rlwe.Ciphertext
	c14 = c13.CopyNew()
	_ = eval.Relinearize(c14, c14)
	c14, _ = eval.MulRelinNew(c14, c14)
	_ = eval.Rescale(c14, c14)
	var c15 *rlwe.Ciphertext
	c15 = c14.CopyNew()
	_ = eval.Relinearize(c15, c15)
	c15, _ = eval.MulRelinNew(c15, c15)
	_ = eval.Rescale(c15, c15)
	var c16 *rlwe.Ciphertext
	c16 = c15.CopyNew()
	_ = eval.Relinearize(c16, c16)
	c16, _ = eval.MulRelinNew(c16, c16)
	_ = eval.Rescale(c16, c16)
	var c17 *rlwe.Ciphertext
	c17 = c16.CopyNew()
	_ = eval.Relinearize(c17, c17)
	c17, _ = eval.MulRelinNew(c17, c17)
	_ = eval.Rescale(c17, c17)
	var c18 *rlwe.Ciphertext
	c18 = c17.CopyNew()
	_ = eval.Relinearize(c18, c18)
	c18, _ = eval.MulRelinNew(c18, c18)
	_ = eval.Rescale(c18, c18)
	var c19 *rlwe.Ciphertext
	c19 = c18.CopyNew()
	_ = eval.Relinearize(c19, c19)
	c19, _ = eval.MulRelinNew(c19, c19)
	_ = eval.Rescale(c19, c19)
	var c20 *rlwe.Ciphertext
	c20 = c19.CopyNew()
	_ = eval.Relinearize(c20, c20)
	c20, _ = eval.MulRelinNew(c20, c20)
	_ = eval.Rescale(c20, c20)
	var c21 *rlwe.Ciphertext
	c21 = c20.CopyNew()
	_ = eval.Relinearize(c21, c21)

	// Store outputs
	encryptedOutputs["result"] = c21
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=15 (n=32768, slots=16384)
	// MaxLevel=11, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
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

	// Prepare test input: x = 1.0001
	// Computing x^(2^10) = x^1024
	values := make([]float64, params.MaxSlots())
	testValue := 1.0001
	for i := range values {
		values[i] = testValue
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(values, pt)
	ct, _ := enc.EncryptNew(pt)
	encryptedInputs["x"] = ct

	expected := pow(testValue, 1024)
	fmt.Printf("Input: x = %.6f\n", testValue)
	fmt.Printf("Expected: x^1024 = %.6f\n", expected)
	fmt.Println("Running FHE computation (no bootstrap needed)...")

	// Run computation
	deep_poly(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

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
