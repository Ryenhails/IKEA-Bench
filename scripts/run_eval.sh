#!/bin/bash
# Example: Evaluate a model on IKEA-Bench
#
# Usage:
#   bash scripts/run_eval.sh <model_name> <setting>
#
# Examples:
#   bash scripts/run_eval.sh qwen3-vl-8b baseline
#   bash scripts/run_eval.sh qwen3.5-9b text_grounding
#   bash scripts/run_eval.sh internvl3.5-8b text_only

MODEL=${1:-"qwen3-vl-8b"}
SETTING=${2:-"baseline"}

python -m ikea_bench.eval \
    --model "$MODEL" \
    --setting "$SETTING" \
    --input data/qa_benchmark.json \
    --data-dir data \
    --output "results/${MODEL}_${SETTING}.json" \
    --max_new_tokens 128
