#!/usr/bin/env bash
# Scores every model in train/trained_models/ on every test set. A model that
# reaches the NDCG threshold on ALL test sets is MOVED to deploy/models/;
# a model failing any set stays in train/trained_models/.
#
# Input:  train/trained_models/lambdamart_*.txt
#         judgements/data/graded/test/*.txt   (one per test CSV)
# Output: deploy/models/lambdamart_*.txt      (passing models only)
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="train/trained_models"
TEST_DIR="judgements/data/graded/test"
DEPLOY_DIR="deploy/models"

MIN_NDCG=0.7  # from you
K=10  # madeup value

mkdir -p "$DEPLOY_DIR"
shopt -s nullglob
models=("$MODELS_DIR"/lambdamart_*.txt)
tests=("$TEST_DIR"/*.txt)
[ ${#models[@]} -gt 0 ] || { echo "No models in $MODELS_DIR"; exit 1; }
[ ${#tests[@]} -gt 0 ] || { echo "No test sets in $TEST_DIR (run judgements/judgements.sh)"; exit 1; }

for model in "${models[@]}"; do
  echo "== $model"
  failed=()
  for test in "${tests[@]}"; do
    rc=0
    out=$(python3 test/evaluate.py --model "$model" --test "$test" --k "$K" --min-ndcg "$MIN_NDCG" 2>&1) || rc=$?
    grep -E "^(mean|PASS|FAIL|ERROR)" <<<"$out" | sed "s|^|  $(basename "$test"): |" || true
    case $rc in
      0) ;;
      2) failed+=("$(basename "$test")") ;;
      *) echo "ERROR evaluating $model on $test"; exit $rc ;;
    esac
  done
  if [ ${#failed[@]} -eq 0 ]; then
    mv "$model" "$DEPLOY_DIR/"; echo "PASS all ${#tests[@]} test sets -> $DEPLOY_DIR/$(basename "$model")"
  else
    echo "FAIL on ${failed[*]}, kept in $MODELS_DIR"
  fi
done
