package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils"
)

func pow(base float64, exp int) float64 {
	return math.Pow(base, float64(exp))
}

var bootstrapper *bootstrapping.Evaluator

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
	var c15 *rlwe.Ciphertext
	c15 = c1.CopyNew()
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
	c21, _ = eval.MulRelinNew(c21, c21)
	_ = eval.Rescale(c21, c21)
	var c22 *rlwe.Ciphertext
	c22 = c21.CopyNew()
	_ = eval.Relinearize(c22, c22)
	c22, _ = eval.MulRelinNew(c22, c22)
	_ = eval.Rescale(c22, c22)
	var c23 *rlwe.Ciphertext
	c23 = c22.CopyNew()
	_ = eval.Relinearize(c23, c23)
	c23, _ = eval.MulRelinNew(c23, c23)
	_ = eval.Rescale(c23, c23)
	var c24 *rlwe.Ciphertext
	c24 = c23.CopyNew()
	_ = eval.Relinearize(c24, c24)
	c24, _ = eval.MulRelinNew(c24, c24)
	_ = eval.Rescale(c24, c24)
	var c25 *rlwe.Ciphertext
	c25 = c24.CopyNew()
	_ = eval.Relinearize(c25, c25)
	// Bootstrap: refresh to max level
	c25, _ = bootstrapper.Bootstrap(c25)
	c25, _ = eval.MulRelinNew(c25, c25)
	_ = eval.Rescale(c25, c25)
	var c26 *rlwe.Ciphertext
	c26 = c25.CopyNew()
	_ = eval.Relinearize(c26, c26)
	c26, _ = eval.MulRelinNew(c26, c26)
	_ = eval.Rescale(c26, c26)
	var c27 *rlwe.Ciphertext
	c27 = c26.CopyNew()
	_ = eval.Relinearize(c27, c27)

	// Store outputs
	encryptedOutputs["result"] = c27
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=15 (n=32768, slots=16384)
	// MaxLevel=12, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
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

	// Bootstrapper setup
	btpParamsLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(params.LogN()),
		LogP:     []int{61, 61, 61, 61},
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

	// Input/Output maps
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// Prepare test input: x = 1.0001
	// Computing x^(2^13) = x^8192
	// Note: x=1.0001^8192 ≈ 2.26 (stays in reasonable range)
	values := make([]float64, params.MaxSlots())
	testValue := 1.0001
	for i := range values {
		values[i] = testValue
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(values, pt)
	ct, _ := enc.EncryptNew(pt)
	encryptedInputs["x"] = ct

	expected := pow(testValue, 8192)
	fmt.Printf("Input: x = %.6f\n", testValue)
	fmt.Printf("Expected: x^8192 = %.6f\n", expected)
	fmt.Println("Running FHE computation with bootstrapping (depth 13)...")

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
