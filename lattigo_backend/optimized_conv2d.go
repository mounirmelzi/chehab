package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{1, 34, 16, 2, 17, 32, 18, 33}
}

func optimized_conv2d(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c1 := encryptedInputs["packed_image"]

	// Encode constants
	slotCount := params.MaxSlots()
	p3 := hefloat.NewPlaintext(params, params.MaxLevel())
	{
		values := make([]float64, slotCount)
		for i := range values {
			values[i] = float64(111111)
		}
		encoder.Encode(values, p3)
	}
	p2 := hefloat.NewPlaintext(params, params.MaxLevel())
	{
		values := make([]float64, slotCount)
		for i := range values {
			values[i] = float64(0)
		}
		encoder.Encode(values, p2)
	}

	// FHE Operations
	var c4 *rlwe.Ciphertext
	c4, _ = eval.MulNew(c1, p3)
	var c5 *rlwe.Ciphertext
	c5, _ = eval.RotateNew(c1, 1)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 2)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 16)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 17)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 18)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 32)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 33)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c1, _ = eval.RotateNew(c1, 34)
	c1, _ = eval.MulNew(c1, p3)
	c4, _ = eval.AddNew(c4, c1)

	// Store outputs
	encryptedOutputs["optimized_output"] = c4
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

	// TODO: Prepare your inputs here
	// Example:
	// values := make([]float64, params.MaxSlots())
	// for i := range values { values[i] = float64(i) }
	// pt := hefloat.NewPlaintext(params, params.MaxLevel())
	// encoder.Encode(values, pt)
	// ct, _ := enc.EncryptNew(pt)
	// encryptedInputs["c0"] = ct

	// Run computation
	optimized_conv2d(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

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
