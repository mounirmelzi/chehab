package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

// Global bootstrapper (will be initialized in main)
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
	var c10 *rlwe.Ciphertext
	c10, _ = eval.MulRelinNew(c9, c9)
	_ = eval.Rescale(c10, c10)
	var c11 *rlwe.Ciphertext
	c11, _ = eval.MulRelinNew(c10, c10)
	_ = eval.Rescale(c11, c11)
	var c12 *rlwe.Ciphertext
	c12, _ = eval.MulRelinNew(c11, c11)
	_ = eval.Rescale(c12, c12)
	// Bootstrap: refresh to max level
	c12, _ = bootstrapper.Bootstrap(c12)
	var c13 *rlwe.Ciphertext
	c13, _ = eval.MulRelinNew(c12, c12)
	_ = eval.Rescale(c13, c13)
	var c14 *rlwe.Ciphertext
	c14, _ = eval.MulRelinNew(c13, c13)
	_ = eval.Rescale(c14, c14)
	var c15 *rlwe.Ciphertext
	c15, _ = eval.MulRelinNew(c14, c14)
	_ = eval.Rescale(c15, c15)
	var c16 *rlwe.Ciphertext
	c16, _ = eval.MulRelinNew(c15, c15)
	_ = eval.Rescale(c16, c16)
	var c17 *rlwe.Ciphertext
	c17, _ = eval.MulRelinNew(c16, c16)
	_ = eval.Rescale(c17, c17)
	var c18 *rlwe.Ciphertext
	c18, _ = eval.MulRelinNew(c17, c17)
	_ = eval.Rescale(c18, c18)
	var c19 *rlwe.Ciphertext
	c19, _ = eval.MulRelinNew(c18, c18)
	_ = eval.Rescale(c19, c19)
	var c20 *rlwe.Ciphertext
	c20, _ = eval.MulRelinNew(c19, c19)
	_ = eval.Rescale(c20, c20)
	var c21 *rlwe.Ciphertext
	c21, _ = eval.MulRelinNew(c20, c20)
	_ = eval.Rescale(c21, c21)
	var c22 *rlwe.Ciphertext
	c22, _ = eval.MulRelinNew(c21, c21)
	_ = eval.Rescale(c22, c22)
	var c23 *rlwe.Ciphertext
	c23, _ = eval.MulRelinNew(c22, c22)
	_ = eval.Rescale(c23, c23)
	// Bootstrap: refresh to max level
	c23, _ = bootstrapper.Bootstrap(c23)
	var c24 *rlwe.Ciphertext
	c24, _ = eval.MulRelinNew(c23, c23)
	_ = eval.Rescale(c24, c24)
	var c25 *rlwe.Ciphertext
	c25, _ = eval.MulRelinNew(c24, c24)
	_ = eval.Rescale(c25, c25)
	var c26 *rlwe.Ciphertext
	c26, _ = eval.MulRelinNew(c25, c25)
	_ = eval.Rescale(c26, c26)

	// Store outputs
	encryptedOutputs["result"] = c26
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=16 (n=65536, slots=32768)
	// MaxLevel=12, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            16,
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

	// Bootstrapper setup (Orion-compatible full configuration)
	// This configuration matches Orion's bootstrapping parameters
	btpParamsLit := bootstrapping.ParametersLiteral{
		LogN: utils.Pointy(params.LogN()),
		LogP: []int{61, 61, 61, 61},
		Xs: ring.Ternary{H: 192},
		LogSlots: utils.Pointy(15),
	}
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParamsLit)
	if err != nil {
		panic(fmt.Errorf("bootstrap params error: %v", err))
	}

	// Generate bootstrap evaluation keys
	fmt.Println("Generating bootstrap keys (this may take a moment)...")
	btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(fmt.Errorf("bootstrap keygen error: %v", err))
	}

	// Create bootstrapper evaluator (global variable)
	bootstrapper, err = bootstrapping.NewEvaluator(btpParams, btpKeys)
	if err != nil {
		panic(fmt.Errorf("bootstrap evaluator error: %v", err))
	}
	fmt.Println("Bootstrap keys generated successfully!")

	// Input/Output maps
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// Prepare test inputs
	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = 0.9 // x = 0.9, x^(2^25) ≈ 0 (underflows)
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(values, pt); err != nil {
		panic(err)
	}
	ct, err := enc.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	encryptedInputs["x"] = ct
	fmt.Println("Input x = 0.9 encrypted successfully!")

	// Run computation (depth=25 with 2 bootstraps)
	fmt.Println("Running FHE computation (depth=25, requires 2 bootstraps)...")
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
