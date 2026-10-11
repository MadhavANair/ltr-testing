#!/usr/bin/env bash
# Trains one LambdaMART model. Run judgements/judgements.sh first.
# Hyperparameters can be overridden by env var (tune/tune.sh does this).
#
# Input:  judgements/data/graded/train/train.txt
#         judgements/data/graded/val/val.txt
# Output: train/trained_models/lambdamart_<ts>.txt
set -euo pipefail
cd "$(dirname "$0")/.."

GRADED_DIR="judgements/data/graded"
OUT_DIR="train/trained_models"

TREES="${TREES:-500}"  # madeup value
LEAF="${LEAF:-10}"  # madeup value
SHRINKAGE="${SHRINKAGE:-0.05}"  # madeup value
MLS="${MLS:-5}"  # madeup value
ESTOP="${ESTOP:-100}"  # madeup value
METRIC="${METRIC:-NDCG@10}"  # madeup value

python3 train/train_lambdamart.py \
    --train "$GRADED_DIR/train/train.txt" \
    --val "$GRADED_DIR/val/val.txt" \
    --out-dir "$OUT_DIR" \
    --trees "$TREES" --leaf "$LEAF" --shrinkage "$SHRINKAGE" --mls "$MLS" \
    --estop "$ESTOP" --metric "$METRIC"
