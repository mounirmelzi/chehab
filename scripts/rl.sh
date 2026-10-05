#!/bin/bash
set -e

cd ../RL
python -m fhe_rl --tokenizer_type dynamic train
