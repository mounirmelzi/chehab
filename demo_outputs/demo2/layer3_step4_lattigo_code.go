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
	p22 := encodedInputs["p0"]
	p21 := encodedInputs["p0"]
	c20 := encryptedInputs["c2"]
	c19 := encryptedInputs["c2"]
	p18 := encodedInputs["p9"]
	p17 := encodedInputs["p9"]
	c16 := encryptedInputs["c3"]
	c15 := encryptedInputs["c3"]
	c14 := encryptedInputs["c1"]
	c13 := encryptedInputs["c1"]
	p12 := encodedInputs["p10"]
	p11 := encodedInputs["p10"]
	c10 := encryptedInputs["c4"]
	c9 := encryptedInputs["c4"]
	c8 := encryptedInputs["c6"]
	c7 := encryptedInputs["c6"]
	c6 := encryptedInputs["c7"]
	c5 := encryptedInputs["c7"]
	c4 := encryptedInputs["c5"]
	c3 := encryptedInputs["c5"]
	c2 := encryptedInputs["c8"]
	c1 := encryptedInputs["c8"]

	// FHE Operations
	c13, _ = eval.MulRelinNew(c13, c19)
	_ = eval.Rescale(c13, c13)
	c9, _ = eval.MulRelinNew(c9, c15)
	_ = eval.Rescale(c9, c9)
	c2, _ = eval.AddNew(c13, c9)
	c3, _ = eval.MulRelinNew(c3, c7)
	_ = eval.Rescale(c3, c3)
	c1, _ = eval.MulRelinNew(c1, c5)
	_ = eval.Rescale(c1, c1)
	c4, _ = eval.AddNew(c3, c1)
	c2, _ = eval.AddNew(c2, c4)
	c5, _ = eval.MulRelinNew(c2, c2)
	_ = eval.Rescale(c5, c5)
	c4, _ = eval.MulNew(c5, p21)
	c2, _ = eval.MulNew(c2, p17)
	c4, _ = eval.AddNew(c4, c2)
	c4, _ = eval.AddNew(c4, p11)

	// Store outputs
	encryptedOutputs["c11"] = c4
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
