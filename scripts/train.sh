#!/usr/bin/env bash
# log -> split -> train. Stops at the model; no evaluation, no promotion.
set -euo pipefail
cd "$(dirname "$0")/.."

GRADED_CSV="data/raw/mock-judgments-601-final.csv"
WORK_DIR="data"
OUT_DIR="Models"

TREES=500
LEAF=10
SHRINKAGE=0.05
MLS=5
ESTOP=100
METRIC="NDCG@10"

python3 scripts/train/log_judgments_with_features.py \
    --graded-csv "$GRADED_CSV" \
    --out "$WORK_DIR/logged_with_features.txt" \
    --fmap "$WORK_DIR/fmap.txt"

python3 scripts/train/split_judgments.py \
    --judgments "$WORK_DIR/logged_with_features.txt" \
    --out-dir "$WORK_DIR/splits" \
    --manifest-out "$WORK_DIR/splits/manifest.json"

python3 scripts/train/train_lambdamart.py \
    --train "$WORK_DIR/splits/train.txt" \
    --val "$WORK_DIR/splits/val.txt" \
    --out-dir "$OUT_DIR" \
    --trees "$TREES" --leaf "$LEAF" --shrinkage "$SHRINKAGE" --mls "$MLS" \
    --estop "$ESTOP" --metric "$METRIC"
