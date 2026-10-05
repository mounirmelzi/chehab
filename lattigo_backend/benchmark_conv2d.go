package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

/*
 * Benchmark: CHEHAB Optimized Conv2D Execution Time
 *
 * This measures the FHE execution time for CHEHAB's rotation-based
 * convolution, which is similar to Orion's Toeplitz-diagonal method.
 *
 * Key metrics:
 * - KeyGen time
 * - Encrypt time
 * - FHE execution time (rotations + multiplications + additions)
 * - Decrypt time
 */

// Conv2D parameters
type Conv2dConfig struct {
	ImageSize  int
	KernelSize int
	NumLayers  int
}

// Execute optimized conv2d (SIMD packing + rotations)
func executeOptimizedConv2d(
	eval *ckks.Evaluator,
	encoder *ckks.Encoder,
	params ckks.Parameters,
	ct *rlwe.Ciphertext,
	config Conv2dConfig,
) *rlwe.Ciphertext {
	// Kernel weights (averaging filter)
	kernelWeights := make([][]float64, config.KernelSize)
	for i := 0; i < config.KernelSize; i++ {
		kernelWeights[i] = make([]float64, config.KernelSize)
		for j := 0; j < config.KernelSize; j++ {
			kernelWeights[i][j] = 1.0 / float64(config.KernelSize*config.KernelSize)
		}
	}

	result := ct.CopyNew()
	currentWidth := config.ImageSize

	// Process each layer
	for layer := 0; layer < config.NumLayers; layer++ {
		var layerResult *rlwe.Ciphertext
		first := true

		// For each kernel position
		for ki := 0; ki < config.KernelSize; ki++ {
			for kj := 0; kj < config.KernelSize; kj++ {
				// Rotation offset
				rotation := ki*currentWidth + kj

				// Rotate input
				var rotated *rlwe.Ciphertext
				if rotation == 0 {
					rotated = result.CopyNew()
				} else {
					rotated, _ = eval.RotateNew(result, rotation)
				}

				// Multiply by weight (plaintext scalar)
				weight := kernelWeights[ki][kj]
				term, _ := eval.MulNew(rotated, weight)

				// Accumulate
				if first {
					layerResult = term
					first = false
				} else {
					eval.Add(layerResult, term, layerResult)
				}
			}
		}

		result = layerResult
		currentWidth = currentWidth - config.KernelSize + 1

		// Rescale after each layer if needed
		if result.Level() > 1 {
			eval.Rescale(result, result)
		}
	}

	return result
}

func runConv2dBenchmark(config Conv2dConfig) {
	fmt.Printf("\n%s\n", "============================================================")
	fmt.Printf("CHEHAB Optimized Conv2D Benchmark\n")
	fmt.Printf("Image: %dx%d, Kernel: %dx%d, Layers: %d\n",
		config.ImageSize, config.ImageSize,
		config.KernelSize, config.KernelSize,
		config.NumLayers)
	fmt.Printf("%s\n\n", "============================================================")

	// Calculate required levels
	levelsNeeded := config.NumLayers + 3 // Some headroom

	// CKKS Parameters
	logN := 15
	logQ := make([]int, levelsNeeded+2)
	logQ[0] = 60
	for i := 1; i <= levelsNeeded; i++ {
		logQ[i] = 45
	}
	logQ[levelsNeeded+1] = 60

	logP := []int{60, 60}

	paramLit := ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: 45,
	}

	params, err := ckks.NewParametersFromLiteral(paramLit)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Parameters:\n")
	fmt.Printf("  LogN: %d (N=%d)\n", params.LogN(), params.N())
	fmt.Printf("  Slots: %d\n", params.MaxSlots())
	fmt.Printf("  Levels: %d\n", params.MaxLevel())

	// Key Generation
	fmt.Printf("\n--- Key Generation ---\n")
	start := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Generate rotation keys for all needed rotations
	rotations := []int{}
	currentWidth := config.ImageSize
	for layer := 0; layer < config.NumLayers; layer++ {
		for ki := 0; ki < config.KernelSize; ki++ {
			for kj := 0; kj < config.KernelSize; kj++ {
				rot := ki*currentWidth + kj
				if rot != 0 {
					rotations = append(rotations, rot)
				}
			}
		}
		currentWidth = currentWidth - config.KernelSize + 1
	}

	// Remove duplicates
	rotSet := make(map[int]bool)
	for _, r := range rotations {
		rotSet[r] = true
	}
	uniqueRots := []int{}
	for r := range rotSet {
		uniqueRots = append(uniqueRots, r)
	}

	galEls := make([]uint64, len(uniqueRots))
	for i, r := range uniqueRots {
		galEls[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galEls, sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	keygenTime := time.Since(start)
	fmt.Printf("KeyGen time: %.2f s\n", keygenTime.Seconds())
	fmt.Printf("Rotation keys: %d\n", len(uniqueRots))

	// Encoder and evaluator
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	eval := ckks.NewEvaluator(params, evk)

	// Create input (random image)
	slots := params.MaxSlots()
	inputData := make([]float64, slots)
	for i := 0; i < config.ImageSize*config.ImageSize && i < slots; i++ {
		inputData[i] = float64(i%10) / 10.0
	}

	// Encryption
	fmt.Printf("\n--- Encryption ---\n")
	start = time.Now()
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(inputData, pt)
	ct, _ := encryptor.EncryptNew(pt)
	encryptTime := time.Since(start)
	fmt.Printf("Encrypt time: %.2f ms\n", float64(encryptTime.Microseconds())/1000)

	// FHE Execution
	fmt.Printf("\n--- FHE Execution ---\n")
	start = time.Now()
	result := executeOptimizedConv2d(eval, encoder, params, ct, config)
	fheTime := time.Since(start)
	fmt.Printf("FHE time: %.2f ms\n", float64(fheTime.Microseconds())/1000)

	// Decryption
	fmt.Printf("\n--- Decryption ---\n")
	start = time.Now()
	ptResult := decryptor.DecryptNew(result)
	outputData := make([]float64, slots)
	encoder.Decode(ptResult, outputData)
	decryptTime := time.Since(start)
	fmt.Printf("Decrypt time: %.2f ms\n", float64(decryptTime.Microseconds())/1000)

	// Summary
	totalFHE := encryptTime + fheTime + decryptTime
	fmt.Printf("\n--- Summary ---\n")
	fmt.Printf("Total FHE time: %.2f ms\n", float64(totalFHE.Microseconds())/1000)
	fmt.Printf("  Encrypt: %.2f ms\n", float64(encryptTime.Microseconds())/1000)
	fmt.Printf("  FHE Compute: %.2f ms\n", float64(fheTime.Microseconds())/1000)
	fmt.Printf("  Decrypt: %.2f ms\n", float64(decryptTime.Microseconds())/1000)

	// Verify output (first few values)
	fmt.Printf("\nOutput sample (first 5 values): ")
	for i := 0; i < 5 && i < len(outputData); i++ {
		fmt.Printf("%.4f ", outputData[i])
	}
	fmt.Printf("\n")
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("  CHEHAB Optimized Conv2D - FHE Execution Benchmark")
	fmt.Println("============================================================")

	// Benchmarks matching Orion comparison
	benchmarks := []Conv2dConfig{
		{ImageSize: 8, KernelSize: 3, NumLayers: 3},
		{ImageSize: 16, KernelSize: 3, NumLayers: 5},
		{ImageSize: 16, KernelSize: 3, NumLayers: 8},
		{ImageSize: 32, KernelSize: 3, NumLayers: 5},
	}

	results := make([]struct {
		Config      string
		KeygenTime  float64
		EncryptTime float64
		FHETime     float64
		DecryptTime float64
		TotalTime   float64
	}, 0)

	for _, config := range benchmarks {
		fmt.Printf("\n")

		// Calculate required levels
		levelsNeeded := config.NumLayers + 3

		// CKKS Parameters
		logN := 15
		logQ := make([]int, levelsNeeded+2)
		logQ[0] = 60
		for i := 1; i <= levelsNeeded; i++ {
			logQ[i] = 45
		}
		logQ[levelsNeeded+1] = 60
		logP := []int{60, 60}

		paramLit := ckks.ParametersLiteral{
			LogN:            logN,
			LogQ:            logQ,
			LogP:            logP,
			LogDefaultScale: 45,
		}

		params, err := ckks.NewParametersFromLiteral(paramLit)
		if err != nil {
			fmt.Printf("Error creating params for %dx%d: %v\n", config.ImageSize, config.ImageSize, err)
			continue
		}

		// Quick timing run
		kgen := rlwe.NewKeyGenerator(params)
		sk := kgen.GenSecretKeyNew()
		pk := kgen.GenPublicKeyNew(sk)
		rlk := kgen.GenRelinearizationKeyNew(sk)

		// Generate rotation keys
		rotSet := make(map[int]bool)
		currentWidth := config.ImageSize
		for layer := 0; layer < config.NumLayers; layer++ {
			for ki := 0; ki < config.KernelSize; ki++ {
				for kj := 0; kj < config.KernelSize; kj++ {
					rot := ki*currentWidth + kj
					if rot != 0 {
						rotSet[rot] = true
					}
				}
			}
			currentWidth = currentWidth - config.KernelSize + 1
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

		// Input
		slots := params.MaxSlots()
		inputData := make([]float64, slots)
		for i := 0; i < config.ImageSize*config.ImageSize && i < slots; i++ {
			inputData[i] = float64(i%10) / 10.0
		}

		// Timing
		start := time.Now()
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(inputData, pt)
		ct, _ := encryptor.EncryptNew(pt)
		encryptTime := time.Since(start)

		start = time.Now()
		result := executeOptimizedConv2d(eval, encoder, params, ct, config)
		fheTime := time.Since(start)

		start = time.Now()
		ptResult := decryptor.DecryptNew(result)
		outputData := make([]float64, slots)
		encoder.Decode(ptResult, outputData)
		decryptTime := time.Since(start)

		totalTime := encryptTime + fheTime + decryptTime

		configStr := fmt.Sprintf("%dx%dx%dL", config.ImageSize, config.ImageSize, config.NumLayers)
		results = append(results, struct {
			Config      string
			KeygenTime  float64
			EncryptTime float64
			FHETime     float64
			DecryptTime float64
			TotalTime   float64
		}{
			Config:      configStr,
			EncryptTime: float64(encryptTime.Microseconds()) / 1000,
			FHETime:     float64(fheTime.Microseconds()) / 1000,
			DecryptTime: float64(decryptTime.Microseconds()) / 1000,
			TotalTime:   float64(totalTime.Microseconds()) / 1000,
		})

		// Suppress unused variable warning
		_ = outputData
		_ = math.Abs(0)
	}

	// Summary table
	fmt.Printf("\n%s\n", "============================================================")
	fmt.Printf("  Summary Table (times in ms)\n")
	fmt.Printf("%s\n", "============================================================")
	fmt.Printf("%-15s %-12s %-12s %-12s %-12s\n", "Config", "Encrypt", "FHE", "Decrypt", "Total")
	fmt.Printf("%s\n", "------------------------------------------------------------")
	for _, r := range results {
		fmt.Printf("%-15s %-12.2f %-12.2f %-12.2f %-12.2f\n",
			r.Config, r.EncryptTime, r.FHETime, r.DecryptTime, r.TotalTime)
	}
	fmt.Printf("%s\n", "============================================================")
	fmt.Printf("\nCompare with: python benchmark_conv2d_comparison.py\n")
}
