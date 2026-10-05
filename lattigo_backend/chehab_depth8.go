package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

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
	var c3 *rlwe.Ciphertext
	c3, _ = eval.MulRelinNew(c1, c1)
	_ = eval.Rescale(c3, c3)
	var c4 *rlwe.Ciphertext
	c4, _ = eval.MulRelinNew(c3, c3)
	_ = eval.Rescale(c4, c4)
	var c5 *rlwe.Ciphertext
	c5, _ = eval.MulRelinNew(c4, c4)
	_ = eval.Rescale(c5, c5)
	var c6 *rlwe.Ciphertext
	c6, _ = eval.MulRelinNew(c5, c5)
	_ = eval.Rescale(c6, c6)
	var c7 *rlwe.Ciphertext
	c7, _ = eval.MulRelinNew(c6, c6)
	_ = eval.Rescale(c7, c7)
	var c8 *rlwe.Ciphertext
	c8, _ = eval.MulRelinNew(c7, c7)
	_ = eval.Rescale(c8, c8)
	var c9 *rlwe.Ciphertext
	c9, _ = eval.MulRelinNew(c8, c8)
	_ = eval.Rescale(c9, c9)

	// Store outputs
	encryptedOutputs["result"] = c9
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=15 (n=32768, slots=16384)
	// MaxLevel=9, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40},
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
