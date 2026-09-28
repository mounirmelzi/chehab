#!/bin/bash
# À lancer depuis la racine du repo (~/chehab-eval), PAS depuis RL/
# Capture CHAQUE version de expression.txt pendant l'exécution de chaque
# benchmark, avant qu'elle soit écrasée par l'appel suivant à call_rl_vectorizer.
# Utile pour les benchmarks qui appellent le compilateur RL plusieurs fois
# (une fois par pixel/ligne/sous-calcul) : box_blur, gx_kernel, gy_kernel,
# matrix_mul, roberts_cross.
set -u

REPO_ROOT="$(pwd)"
declare -A SLOT_COUNTS=( ["max"]=4 ["sort"]=3 )

BENCHMARKS=(lin_reg box_blur matrix_mul max sort dot_product gx_kernel gy_kernel hamming_dist l2_distance poly_reg roberts_cross)

mkdir -p "$REPO_ROOT/RL/expr_snapshots"

for bench in "${BENCHMARKS[@]}"; do
    slot="${SLOT_COUNTS[$bench]:-8}"
    bin_path="$REPO_ROOT/build/benchmarks/$bench/$bench"

    if [ ! -x "$bin_path" ]; then
        echo "!! Binaire introuvable pour $bench — on saute."
        continue
    fi

    echo "=== $bench (slot_count=$slot) ==="
    OUT_DIR="$REPO_ROOT/RL/expr_snapshots/$bench"
    mkdir -p "$OUT_DIR"
    rm -f "$OUT_DIR"/*.txt

    (
        cd "$REPO_ROOT/build/benchmarks/$bench"
        ./"$bench" 1 "$slot" 1 0 1 1 1 0 1.0 0.0 &
        PID=$!

        i=0
        LAST_HASH=""
        while kill -0 "$PID" 2>/dev/null; do
            if [ -f "$REPO_ROOT/build/benchmarks/expression.txt" ]; then
                CUR_HASH=$(md5sum "$REPO_ROOT/build/benchmarks/expression.txt" 2>/dev/null | awk '{print $1}')
                if [ "$CUR_HASH" != "$LAST_HASH" ]; then
                    cp "$REPO_ROOT/build/benchmarks/expression.txt" \
                       "$OUT_DIR/snap_$(printf '%03d' $i).txt" 2>/dev/null
                    LAST_HASH="$CUR_HASH"
                    i=$((i+1))
                fi
            fi
            sleep 0.1
        done
        wait "$PID" 2>/dev/null
    )

    n=$(ls "$OUT_DIR"/*.txt 2>/dev/null | wc -l)
    echo "   -> $n version(s) d'expression.txt capturée(s) dans RL/expr_snapshots/$bench/"
done

echo ""
echo "Terminé. Résumé (taille en octets par snapshot) :"
for bench in "${BENCHMARKS[@]}"; do
    d="$REPO_ROOT/RL/expr_snapshots/$bench"
    if [ -d "$d" ] && [ "$(ls -A "$d" 2>/dev/null)" ]; then
        echo "--- $bench ---"
        wc -c "$d"/*.txt
    fi
done