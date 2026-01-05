#!/bin/bash
set -e

cse=1
const_folding=1

cd ../RL
python -m veclang_runner.generator --veclang_expression_file "path/to/expression.txt"

cd ..
cmake -S . -B build
cd build
make

cd RL/veclang_runner
./veclang_runner $cse $const_folding

cd he
cmake -S . -B build
cd build
make

./main
