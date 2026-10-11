# Train

Run everything from `opensearch-ltr/`.

## Setup (once per machine)

```bash
source venv/bin/activate
pip install -r requirements.txt
java -version
```

## What each script does

| Script | Does | Writes |
|---|---|---|
| `judgements/judgements.sh` | logs `raw/train`, `raw/val`, `raw/test` CSVs (no split step) | `judgements/data/graded/{train,val,test}/` |
| `train/train.sh` | trains one model from graded train + val | `train/trained_models/lambdamart_<ts>.txt` |
| `test/test.sh` | NDCG@10 >= 0.7 on every `graded/test/*.txt`; moves passing models to `deploy/models/` | `deploy/models/` |
| `tune/tune.sh` | hyperparameter search, then retrains the winner and reports absolute NDCG@k on `val.txt` | `tune/tuning_report.json` |
| `judgements/log.py` | one or more graded CSVs → one RankLib file; checks QID uniqueness and `--disjoint-from` overlap | `--out` path |
| `train/train_lambdamart.py` | trains LambdaMART via the RankLib jar | `train/trained_models/lambdamart_<ts>.txt` |
| `tune/tune.py` | Bayesian search over shrinkage/leaf/mls/estop, then one retrain of the winner for an absolute NDCG check | `tune/tuning_report.json` |
| `test/evaluate.py` | actual-vs-predicted NDCG@k on any model + any test file; exit 2 below `--min-ndcg`. Called by `test/test.sh` | stderr; optional CSV via `--out` |
| `deploy/featureset.json` | the `alamy_phase1` feature definitions, in `ltr/features.py` order | — |
| `./deploy/upload-model` | uploads the featureset and newest `deploy/models/` model to the local bot OpenSearch; skips if `alamy_phase1` already has a model (run after `local-dev/scripts/create-index`) | model in `.ltrstore` |
| `./deploy/delete-model` | deletes a model (latest by default, or a path/name) from the local bot OpenSearch; leaves the featureset | — |

Everything runs offline against the data drop — no live search API involved.

## Produce the model

```bash
./train/train.sh
```

Needs `./judgements/judgements.sh` first, which logs the content team's
`raw/train/`, `raw/val/` and `raw/test/` CSVs. Hyperparameters are variables
at the top of `train.sh`.

Or by hand:

```bash
python3 judgements/log.py --graded-csv judgements/data/raw/train/*.csv \
    --out judgements/data/graded/train/train.txt
python3 judgements/log.py --graded-csv judgements/data/raw/val/*.csv \
    --disjoint-from judgements/data/raw/train/*.csv \
    --out judgements/data/graded/val/val.txt

python3 train/train_lambdamart.py \
    --train judgements/data/graded/train/train.txt --val judgements/data/graded/val/val.txt \
    --trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100
```

Writes one file: `train/trained_models/lambdamart_<timestamp>.txt`. `--trees` is a cap, not a
count — with `--estop` the jar keeps the best validation iteration, so a 500-cap
run typically saves ~64 trees. Read the `validation metric` line it prints, not
`training metric`, which hits 1.0 within a few rounds on this data.

## Tune the hyperparameters (this is also the NDCG check)

```bash
./tune/tune.sh
```

Bayesian search over shrinkage/leaf/mls/estop, scored by k-fold CV on the
pooled train+val queries — that CV score is for comparing configurations
against each other, not a score against `val.txt` itself. Once a winner is
picked, `tune.py` retrains it *once* more on the plain, unpooled train/val
sets and prints RankLib's own **absolute** NDCG@k for that exact config —
the same number `train_lambdamart.py` would print, written to the report
JSON as `final_val_ndcg`. `tune.sh` then runs `train/train.sh` with the
winning parameters, which writes the model to `train/trained_models/`. Costs
`N_TRIALS x FOLDS` training runs plus one more for the final check (151 by
default) — for a quick look: `N_TRIALS=5 FOLDS=3 ./tune/tune.sh`.

There's no baseline comparison anywhere in this — it's a quality sanity
check (does the model rank held-out graded documents sensibly), not a
launch decision.

## Ad hoc: check a specific model against a specific test set

```bash
python3 test/evaluate.py --model train/trained_models/lambdamart_X.txt --test judgements/data/graded/test/<name>.txt
```

`test/test.sh` calls this for every model in `train/trained_models/` against
every `graded/test/*.txt` with `--min-ndcg 0.7`; by hand it works against any file. Scores the model, then prints per-query and mean NDCG@k to stderr —
actual (graded) value vs predicted (model score) value, freely re-runnable
against any RankLib-format file. Pass `--out <path.csv>` for a per-row
`qid, doc_id, rank, actual_grade, predicted_score` table.

## Upload the model to OpenSearch

```bash
./deploy/upload-model                      # newest deploy/models/ model
./deploy/upload-model deploy/models/X.txt  # or a specific one
```

Checks `deploy/featureset.json` order against `ltr/features.py` (RankLib scores
by position), PUTs the `alamy_phase1` featureset from that file if missing,
then wraps the model `.txt` with `jq --rawfile` and POSTs it to `_createmodel`.
Nothing is written to disk. If `alamy_phase1` already has a model it skips;
`_createmodel` will not overwrite, so delete first:

```bash
./deploy/delete-model <model name>
```

Then query it:

```bash
curl -s -X GET "http://localhost:9201/images/_search" \
  -H 'Content-Type: application/json' -d '{
  "query": {
    "sltr": {
      "params": { "keywords": "london bus" },
      "model": "lambdamart_20260911_092634"
    }
  }
}' | jq
```

Note: the featureset's field names are snake_case while the bot index mapping
is PascalCase (`Caption`, `EssentialWords`, …), and the seven non-text features
are templated as keyword matches. The mapping is `dynamic: false`, so those
features resolve to 0 rather than erroring — scores from this featureset are
not yet comparable to training-time values.
