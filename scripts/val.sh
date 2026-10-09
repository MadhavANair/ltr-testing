#!/usr/bin/env bash
# Hyperparameter search on the train+val folds. Prints a train_lambdamart.py
# command to paste into train.sh; ships no model.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK_DIR="data"

SAMPLER="gp"
TREES=500  # madeup value
# N_TRIALS x FOLDS jar runs, so 150 by default. Override for a quick look:
#   N_TRIALS=5 FOLDS=3 ./scripts/val.sh
N_TRIALS="${N_TRIALS:-30}"  # madeup value
FOLDS="${FOLDS:-5}"  # madeup value

python3 scripts/val/tune.py \
    --train "$WORK_DIR/splits/train.txt" \
    --val "$WORK_DIR/splits/val.txt" \
    --sampler "$SAMPLER" --n-trials "$N_TRIALS" --folds "$FOLDS" --trees "$TREES" \
    --report-out Models/tuning/tuning_report.json
