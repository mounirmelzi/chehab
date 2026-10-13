// CPU memory probe (Lattigo v5 CKKS), same workload as gpu_mem_probe.cu.
//
//	go run . <N> <num_q_primes> <num_rot_keys> <live_cts> <rep> [num_p_primes=1]
//
// Live heap is read after a forced GC at each phase (Go's HeapAlloc without
// GC also counts unreclaimed garbage). Exact key/ciphertext sizes are taken
// from BinarySize(), and process peak RSS from getrusage.
package main

import (
	"fmt"
	"math/bits"
	"os"
	"runtime"
	"strconv"
	"syscall"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func live() int64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.HeapAlloc)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	N, _ := strconv.Atoi(os.Args[1])
	L, _ := strconv.Atoi(os.Args[2])
	k, _ := strconv.Atoi(os.Args[3])
	w, _ := strconv.Atoi(os.Args[4])
	rep, _ := strconv.Atoi(os.Args[5])
	K := 1
	if len(os.Args) > 6 {
		K, _ = strconv.Atoi(os.Args[6])
	}

	logQ := []int{60}
	for i := 1; i < L; i++ {
		logQ = append(logQ, 40)
	}
	logP := make([]int, K)
	for i := range logP {
		logP[i] = 60
	}

	m0 := live()
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN: bits.Len(uint(N)) - 1, LogQ: logQ, LogP: logP, LogDefaultScale: 40,
	})
	must(err)
	mCtx := live()

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	mSkPk := live()

	rlk := kgen.GenRelinearizationKeyNew(sk)
	mRlk := live()

	galEls := make([]uint64, k)
	for i := 1; i <= k; i++ {
		galEls[i-1] = params.GaloisElement(i)
	}
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	mGk := live()

	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)
	ecd := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	eval := hefloat.NewEvaluator(params, evk)
	mTools := live()

	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = 0.5
	}
	cts := make([]*rlwe.Ciphertext, w)
	for j := 0; j < w; j++ {
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		must(ecd.Encode(values, pt))
		cts[j], err = enc.EncryptNew(pt)
		must(err)
	}
	mCts := live()

	acc := cts[0].CopyNew()
	for i := 1; i <= k; i++ {
		tmp, err := eval.RotateNew(cts[i%w], i)
		must(err)
		must(eval.Add(acc, tmp, acc))
	}
	acc, err = eval.MulRelinNew(acc, cts[0])
	must(err)
	must(eval.Rescale(acc, acc))
	mEnd := live()

	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	peakRSS := ru.Maxrss * 1024 // Linux reports KB

	gkBytes := 0
	for _, g := range gks {
		gkBytes += g.BinarySize()
	}

	// backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak_rss,
	// exact_rlk,exact_gk,exact_ct,exact_pk,exact_sk
	fmt.Printf("cpu,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
		N, L, len(logP), k, w, rep,
		mCtx-m0, mSkPk-mCtx, mRlk-mSkPk, mGk-mRlk, mTools-mGk, mCts-mTools, mEnd-m0, peakRSS,
		rlk.BinarySize(), gkBytes, cts[0].BinarySize(), pk.BinarySize(), sk.BinarySize())

	runtime.KeepAlive(acc)
	runtime.KeepAlive(cts)
	runtime.KeepAlive(eval)
	runtime.KeepAlive(enc)
	runtime.KeepAlive(ecd)
	runtime.KeepAlive(sk)
}
