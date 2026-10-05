package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{}
}

func conv2d(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c18 := encryptedInputs["kernel_2_2"]
	c17 := encryptedInputs["kernel_2_1"]
	c16 := encryptedInputs["kernel_2_0"]
	c15 := encryptedInputs["kernel_1_2"]
	c14 := encryptedInputs["kernel_1_1"]
	c13 := encryptedInputs["kernel_1_0"]
	c12 := encryptedInputs["kernel_0_2"]
	c11 := encryptedInputs["kernel_0_1"]
	c10 := encryptedInputs["kernel_0_0"]
	c9 := encryptedInputs["img_2_2"]
	c8 := encryptedInputs["img_2_1"]
	c7 := encryptedInputs["img_2_0"]
	c6 := encryptedInputs["img_1_2"]
	c5 := encryptedInputs["img_1_1"]
	c4 := encryptedInputs["img_1_0"]
	c3 := encryptedInputs["img_0_2"]
	c2 := encryptedInputs["img_0_1"]
	c1 := encryptedInputs["img_0_0"]

	// FHE Operations
	c1, _ = eval.MulRelinNew(c1, c10)
	_ = eval.Rescale(c1, c1)
	_ = eval.Relinearize(c1, c1)
	c2, _ = eval.MulRelinNew(c2, c11)
	_ = eval.Rescale(c2, c2)
	_ = eval.Relinearize(c2, c2)
	c1, _ = eval.AddNew(c1, c2)
	c3, _ = eval.MulRelinNew(c3, c12)
	_ = eval.Rescale(c3, c3)
	_ = eval.Relinearize(c3, c3)
	c1, _ = eval.AddNew(c1, c3)
	c4, _ = eval.MulRelinNew(c4, c13)
	_ = eval.Rescale(c4, c4)
	_ = eval.Relinearize(c4, c4)
	c1, _ = eval.AddNew(c1, c4)
	c5, _ = eval.MulRelinNew(c5, c14)
	_ = eval.Rescale(c5, c5)
	_ = eval.Relinearize(c5, c5)
	c1, _ = eval.AddNew(c1, c5)
	c6, _ = eval.MulRelinNew(c6, c15)
	_ = eval.Rescale(c6, c6)
	_ = eval.Relinearize(c6, c6)
	c1, _ = eval.AddNew(c1, c6)
	c7, _ = eval.MulRelinNew(c7, c16)
	_ = eval.Rescale(c7, c7)
	_ = eval.Relinearize(c7, c7)
	c1, _ = eval.AddNew(c1, c7)
	c8, _ = eval.MulRelinNew(c8, c17)
	_ = eval.Rescale(c8, c8)
	_ = eval.Relinearize(c8, c8)
	c1, _ = eval.AddNew(c1, c8)
	c9, _ = eval.MulRelinNew(c9, c18)
	_ = eval.Rescale(c9, c9)
	_ = eval.Relinearize(c9, c9)
	c1, _ = eval.AddNew(c1, c9)

	// Store outputs
	encryptedOutputs["out_0_0"] = c1
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
	// Test: Conv2D 3x3 image with 3x3 kernel
	// ============================================
	// Image (3x3):
	// [1 2 3]
	// [4 5 6]
	// [7 8 9]
	//
	// Kernel (3x3):
	// [1 0 -1]
	// [2 0 -2]
	// [1 0 -1]
	//
	// Output (1x1 valid convolution):
	// out[0][0] = sum(img * kernel)
	//           = 1*1 + 2*0 + 3*(-1) + 4*2 + 5*0 + 6*(-2) + 7*1 + 8*0 + 9*(-1)
	//           = 1 + 0 - 3 + 8 + 0 - 12 + 7 + 0 - 9 = -8
	// ============================================
	
	img := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	
	kernel := [][]float64{
		{1, 0, -1},
		{2, 0, -2},
		{1, 0, -1},
	}
	
	// Calculate expected result
	expected := 0.0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected += img[i][j] * kernel[i][j]
		}
	}
	
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
	
	// Encrypt image
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			name := fmt.Sprintf("img_%d_%d", i, j)
			encryptedInputs[name] = encryptValue(img[i][j])
		}
	}
	
	// Encrypt kernel
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			name := fmt.Sprintf("kernel_%d_%d", i, j)
			encryptedInputs[name] = encryptValue(kernel[i][j])
		}
	}
	
	fmt.Println("=== CHEHAB Conv2D Test ===")
	fmt.Println("Image (3x3):")
	for i := 0; i < 3; i++ {
		fmt.Printf("  %v\n", img[i])
	}
	fmt.Println("Kernel (3x3):")
	for i := 0; i < 3; i++ {
		fmt.Printf("  %v\n", kernel[i])
	}
	fmt.Printf("\nExpected result: %.1f\n\n", expected)

	// Run computation
	conv2d(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

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
