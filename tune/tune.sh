#!/usr/bin/env bash
# Bayesian hyperparameter search on the train+val folds, then runs
# train/train.sh with the winning parameters.
#
# Input:  judgements/data/graded/train/train.txt
#         judgements/data/graded/val/val.txt
# Output: tune/tuning_report.json
#         train/trained_models/lambdamart_<ts>.txt   (via train/train.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

GRADED_DIR="judgements/data/graded"
REPORT="tune/tuning_report.json"

SAMPLER="gp"
TREES=500  # madeup value
# N_TRIALS x FOLDS jar runs, so 150 by default. Override for a quick look:
#   N_TRIALS=5 FOLDS=3 ./tune/tune.sh
N_TRIALS="${N_TRIALS:-30}"  # madeup value
FOLDS="${FOLDS:-5}"  # madeup value

python3 tune/tune.py \
    --train "$GRADED_DIR/train/train.txt" \
    --val "$GRADED_DIR/val/val.txt" \
    --sampler "$SAMPLER" --n-trials "$N_TRIALS" --folds "$FOLDS" --trees "$TREES" \
    --report-out "$REPORT"

TREES=$(jq -r '.best.trees' "$REPORT") \
SHRINKAGE=$(jq -r '.best.shrinkage' "$REPORT") \
LEAF=$(jq -r '.best.leaf' "$REPORT") \
MLS=$(jq -r '.best.mls' "$REPORT") \
ESTOP=$(jq -r '.best.estop' "$REPORT") \
METRIC=$(jq -r '.best.metric' "$REPORT") \
    ./train/train.sh
