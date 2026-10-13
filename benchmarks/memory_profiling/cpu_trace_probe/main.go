// Executes a CHEHAB program trace (see make_trace.py) on Lattigo v5 and
// reports the peak live heap, measured after a forced GC following every op.
//
//	cpu_trace_probe <program.trace> <name>
package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
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

func ints(s string) []int {
	var out []int
	for _, t := range strings.Split(s, ",") {
		v, err := strconv.Atoi(t)
		must(err)
		out = append(out, v)
	}
	return out
}

func main() {
	f, err := os.Open(os.Args[1])
	must(err)
	name := os.Args[2]
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)

	sc.Scan()
	hdr := strings.Fields(sc.Text())
	logN, _ := strconv.Atoi(hdr[1])
	logQ, logP := ints(hdr[2]), ints(hdr[3])

	m0 := live()
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN: logN, LogQ: logQ, LogP: logP, LogDefaultScale: 40,
	})
	must(err)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	ecd := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	eval := hefloat.NewEvaluator(params, evk)
	mSetup := live()

	values := make([]float64, params.MaxSlots())
	for i := range values {
		values[i] = 0.5
	}
	vals := map[int]*rlwe.Ciphertext{}
	var mInputs int64 = -1
	peak := mSetup
	nOps := 0

	for sc.Scan() {
		t := strings.Fields(sc.Text())
		id, _ := strconv.Atoi(t[1])
		switch t[0] {
		case "IN":
			pt := hefloat.NewPlaintext(params, params.MaxLevel())
			must(ecd.Encode(values, pt))
			vals[id], err = enc.EncryptNew(pt)
			must(err)
			continue
		case "MUL", "ADD":
			a, _ := strconv.Atoi(t[2])
			b, _ := strconv.Atoi(t[3])
			if t[0] == "MUL" {
				vals[id], err = eval.MulRelinNew(vals[a], vals[b])
			} else {
				vals[id], err = eval.AddNew(vals[a], vals[b])
			}
			must(err)
			nOps++
		case "RS":
			must(eval.Rescale(vals[id], vals[id]))
			nOps++
		case "FREE":
			delete(vals, id)
			continue
		case "OUT":
			continue
		}
		if mInputs < 0 {
			mInputs = live()
		}
		if l := live(); l > peak {
			peak = l
		}
	}
	mEnd := live()

	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)

	// name,N,L,K,ops,setup,after_inputs,end,peak_live,peak_rss
	fmt.Printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", name, 1<<logN, len(logQ), len(logP), nOps,
		mSetup-m0, mInputs-m0, mEnd-m0, peak-m0, ru.Maxrss*1024)
	runtime.KeepAlive(eval)
	runtime.KeepAlive(enc)
	runtime.KeepAlive(sk)
}
