package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{1, 24, 34, 11, 16, 2, 25, 17, 28, 18, 33, 10, 14, 29, 12, 13, 15, 26, 32, 9, 20, 21, 22, 30, 8}
}

func deep_conv2d_5layers(
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
	c1, _ = eval.MulNew(c4, p3)
	c5, _ = eval.RotateNew(c4, 1)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 2)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 14)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 15)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 16)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 28)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 29)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c4, _ = eval.RotateNew(c4, 30)
	c4, _ = eval.MulNew(c4, p3)
	c1, _ = eval.AddNew(c1, c4)
	c4, _ = eval.MulNew(c1, p3)
	c5, _ = eval.RotateNew(c1, 1)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 2)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 12)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 13)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 14)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 24)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 25)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c1, _ = eval.RotateNew(c1, 26)
	c1, _ = eval.MulNew(c1, p3)
	c4, _ = eval.AddNew(c4, c1)
	c1, _ = eval.MulNew(c4, p3)
	c5, _ = eval.RotateNew(c4, 1)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 2)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 10)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 11)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 12)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 20)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c5, _ = eval.RotateNew(c4, 21)
	c5, _ = eval.MulNew(c5, p3)
	c1, _ = eval.AddNew(c1, c5)
	c4, _ = eval.RotateNew(c4, 22)
	c4, _ = eval.MulNew(c4, p3)
	c1, _ = eval.AddNew(c1, c4)
	c4, _ = eval.MulNew(c1, p3)
	c5, _ = eval.RotateNew(c1, 1)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 2)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 8)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 9)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 10)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 16)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c5, _ = eval.RotateNew(c1, 17)
	c5, _ = eval.MulNew(c5, p3)
	c4, _ = eval.AddNew(c4, c5)
	c1, _ = eval.RotateNew(c1, 18)
	c1, _ = eval.MulNew(c1, p3)
	c4, _ = eval.AddNew(c4, c1)

	// Store outputs
	encryptedOutputs["deep_conv_output"] = c4
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
	deep_conv2d_5layers(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

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
