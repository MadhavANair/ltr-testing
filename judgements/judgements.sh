#!/usr/bin/env bash
# Converts the content team's raw CSVs into RankLib files. Run once per new
# drop, before train/train.sh or tune/tune.sh. QIDs must be globally unique,
# and val/test may share no query (QID or keywords) with the sets before them.
#
# Input:  judgements/data/raw/train/*.csv   (all combined)
#         judgements/data/raw/val/*.csv     (all combined)
#         judgements/data/raw/test/*.csv    (one output each)
# Output: judgements/data/graded/train/train.txt
#         judgements/data/graded/val/val.txt
#         judgements/data/graded/test/<csv name>.txt
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR="judgements/data/raw"
GRADED_DIR="judgements/data/graded"

shopt -s nullglob
train=("$RAW_DIR"/train/*.csv)
val=("$RAW_DIR"/val/*.csv)
test=("$RAW_DIR"/test/*.csv)
for set in train val test; do
  declare -n files=$set
  [ ${#files[@]} -gt 0 ] || { echo "No CSVs in $RAW_DIR/$set/"; exit 1; }
done

rm -rf "$GRADED_DIR"/{train,val,test}

python3 judgements/log.py \
    --graded-csv "${train[@]}" \
    --out "$GRADED_DIR/train/train.txt"

python3 judgements/log.py \
    --graded-csv "${val[@]}" \
    --disjoint-from "${train[@]}" \
    --out "$GRADED_DIR/val/val.txt"

for csv in "${test[@]}"; do
  python3 judgements/log.py \
      --graded-csv "$csv" \
      --disjoint-from "${train[@]}" "${val[@]}" \
      --out "$GRADED_DIR/test/$(basename "${csv%.csv}").txt"
done
