#!/bin/sh
# Run one arm of the experiment on this Mac.
# Usage: ./run_arm.sh <model-stem> <gpu|cpu> <fp16|fp32> <cases> <label> [maxNumTokens]
#   MTP=1 ./run_arm.sh ... enables speculative decoding (the model's MTP drafter).
#   e.g. ./run_arm.sh int-minus2048 gpu fp16 long int-minus2048
# Reads models/<model-stem>.litertlm and cases/<cases>.json; writes
# results/mac/<cases>__<label>__<backend>-<precision>.jsonl
set -eu
cd "$(dirname "$0")"
out="results/mac/${4}__${5}__${2}-${3}.jsonl"
runner/.build/release/runner "models/$1.litertlm" "$2" "$3" "cases/$4.json" ${6:-} 2>>runner.log > "$out"
echo "$5 $2 $3 $4: $(grep -c '"exact":true' "$out")/$(wc -l < "$out" | tr -d ' ') exact"
