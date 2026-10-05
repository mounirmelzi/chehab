#!/bin/bash
set -e


# --------- #
# benchmark=box_blur
# benchmark=lin_reg
# --------- #
benchmark=dot_product
# benchmark=hamming_dist
# benchmark=l2_distance
# --------- #


slot_count=3
window=0

optimization_method=1 # egraph(0) RL(1)

vectorize_code=1
call_quantifier=1
cse=1
const_folding=1

cd ..
cmake -S . -B build
cd build
make

cd benchmarks/$benchmark
python generate_$benchmark.py --slot_count $slot_count
./$benchmark $vectorize_code $slot_count $optimization_method $window $call_quantifier $cse $const_folding

cd he
cmake -S . -B build
cd build
make
./main
