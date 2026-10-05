package main

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func runSingleLayerConv(imgSize, kernelSize int) (float64, float64, float64) {
	fmt.Printf("\n%dx%d image, %dx%d kernel:\n", imgSize, imgSize, kernelSize, kernelSize)

	logN := 14
	paramLit := ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{60, 45, 45, 45, 45, 60},
		LogP:            []int{60, 60},
		LogDefaultScale: 45,
	}

	params, _ := ckks.NewParametersFromLiteral(paramLit)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Generate rotation keys
	rotSet := make(map[int]bool)
	for ki := 0; ki < kernelSize; ki++ {
		for kj := 0; kj < kernelSize; kj++ {
			rot := ki*imgSize + kj
			if rot != 0 {
				rotSet[rot] = true
			}
		}
	}

	galEls := make([]uint64, 0)
	for r := range rotSet {
		galEls = append(galEls, params.GaloisElement(r))
	}
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	eval := ckks.NewEvaluator(params, evk)

	slots := params.MaxSlots()
	inputData := make([]float64, slots)
	for i := 0; i < imgSize*imgSize && i < slots; i++ {
		inputData[i] = float64(i%10) / 10.0
	}

	// Encryption
	start := time.Now()
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(inputData, pt)
	ct, _ := encryptor.EncryptNew(pt)
	encryptTime := time.Since(start)

	// FHE Execution (single conv layer)
	start = time.Now()
	var result *rlwe.Ciphertext
	first := true
	weight := 1.0 / float64(kernelSize*kernelSize)

	for ki := 0; ki < kernelSize; ki++ {
		for kj := 0; kj < kernelSize; kj++ {
			rotation := ki*imgSize + kj
			var rotated *rlwe.Ciphertext
			if rotation == 0 {
				rotated = ct.CopyNew()
			} else {
				rotated, _ = eval.RotateNew(ct, rotation)
			}
			term, _ := eval.MulNew(rotated, weight)
			if first {
				result = term
				first = false
			} else {
				eval.Add(result, term, result)
			}
		}
	}
	fheTime := time.Since(start)

	// Decryption
	start = time.Now()
	ptResult := decryptor.DecryptNew(result)
	outputData := make([]float64, slots)
	encoder.Decode(ptResult, outputData)
	decryptTime := time.Since(start)

	totalFHE := encryptTime + fheTime + decryptTime

	encMs := float64(encryptTime.Microseconds()) / 1000
	fheMs := float64(fheTime.Microseconds()) / 1000
	totalMs := float64(totalFHE.Microseconds()) / 1000

	fmt.Printf("  Encrypt:     %.2f ms\n", encMs)
	fmt.Printf("  FHE Compute: %.2f ms\n", fheMs)
	fmt.Printf("  Decrypt:     %.2f ms\n", float64(decryptTime.Microseconds())/1000)
	fmt.Printf("  Total FHE:   %.2f ms\n", totalMs)

	return encMs, fheMs, totalMs
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("  CHEHAB Single-Layer Conv2D Benchmark")
	fmt.Println("============================================================")

	results := make([]struct {
		config string
		enc    float64
		fhe    float64
		total  float64
	}, 0)

	for _, imgSize := range []int{8, 16, 32} {
		enc, fhe, total := runSingleLayerConv(imgSize, 3)
		results = append(results, struct {
			config string
			enc    float64
			fhe    float64
			total  float64
		}{
			config: fmt.Sprintf("%dx%d", imgSize, imgSize),
			enc:    enc,
			fhe:    fhe,
			total:  total,
		})
	}

	fmt.Println("\n============================================================")
	fmt.Println("  Summary")
	fmt.Println("============================================================")
	fmt.Printf("%-12s %-15s %-18s %-15s\n", "Config", "Encrypt(ms)", "FHE Compute(ms)", "Total FHE(ms)")
	fmt.Println("------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-12s %-15.2f %-18.2f %-15.2f\n", r.config, r.enc, r.fhe, r.total)
	}
	fmt.Println("============================================================")
}
