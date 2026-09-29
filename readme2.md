# Train

Run everything from `training-ltr/`.

## Setup (once per machine)

```bash
source venv/bin/activate
pip install -r requirements.txt
java -version
```

## What each script does

| Script | Does | Writes |
|---|---|---|
| `scripts/train.sh` | log → split → train, stops at the model | `Models/lambdamart_<ts>.txt` |
| `scripts/val.sh` | hyperparameter search, then retrains the winner and reports absolute NDCG@k on `val.txt` | `Models/tuning/tuning_report.json` |
| `scripts/train/log_judgments_with_features.py` | graded CSV drop → RankLib vectors | `data/logged_with_features.txt`, `data/fmap.txt` |
| `scripts/train/split_judgments.py` | splits by query (stable qid hash) | `data/splits/{train,val,test}.txt`, `manifest.json` |
| `scripts/train/train_lambdamart.py` | trains LambdaMART via the RankLib jar | `Models/lambdamart_<ts>.txt` |
| `scripts/val/tune.py` | Bayesian search over shrinkage/leaf/mls/estop, then one retrain of the winner for an absolute NDCG check | `Models/tuning/tuning_report.json` |
| `scripts/test/evaluate.py` | manual, standalone: actual-vs-predicted NDCG@k on any model + any split. Nothing calls this automatically | stderr; optional CSV via `--out` |
| `scripts/export_model/export_model.py` | builds the `_createmodel` request body | `export/<model>_model_body.json` |
| `./upload-model` | uploads the latest model to local OpenSearch and checks it loads and scores (run after `local-dev/scripts/create-index`) | model in `.ltrstore` |

Everything runs offline against the data drop — no live search API involved.

## Produce the model

```bash
./scripts/train.sh
```

Runs the three steps below and stops at the model. Input file and
hyperparameters are variables at the top of the script.

Or step by step:

```bash
python3 scripts/train/log_judgments_with_features.py \
    --graded-csv data/raw/mock-judgments-601-final.csv \
    --out data/logged_with_features.txt --fmap data/fmap.txt

python3 scripts/train/split_judgments.py \
    --judgments data/logged_with_features.txt \
    --out-dir data/splits --manifest-out data/splits/manifest.json

python3 scripts/train/train_lambdamart.py \
    --train data/splits/train.txt --val data/splits/val.txt \
    --trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100
```

Writes one file: `Models/lambdamart_<timestamp>.txt`. `--trees` is a cap, not a
count — with `--estop` the jar keeps the best validation iteration, so a 500-cap
run typically saves ~64 trees. Read the `validation metric` line it prints, not
`training metric`, which hits 1.0 within a few rounds on this data.

## Tune the hyperparameters (this is also the NDCG check)

```bash
./scripts/val.sh
```

Bayesian search over shrinkage/leaf/mls/estop, scored by k-fold CV on the
pooled train+val queries — that CV score is for comparing configurations
against each other, not a score against `val.txt` itself. Once a winner is
picked, `tune.py` retrains it *once* more on the plain, unpooled train/val
split and prints RankLib's own **absolute** NDCG@k for that exact config —
the same number `train_lambdamart.py` would print, written to the report
JSON as `final_val_ndcg`. It still writes no shippable model — it prints a
`train_lambdamart.py` command to paste into `train.sh`. Costs
`N_TRIALS x FOLDS` training runs plus one more for the final check (151 by
default) — for a quick look: `N_TRIALS=5 FOLDS=3 ./scripts/val.sh`.

There's no baseline comparison anywhere in this — it's a quality sanity
check (does the model rank held-out graded documents sensibly), not a
launch decision.

## Ad hoc: check a specific model against a specific split

```bash
python3 scripts/test/evaluate.py --model Models/lambdamart_X.txt --test data/splits/val.txt
```

Manual, standalone — nothing in the pipeline calls `evaluate.py`
automatically (`val.sh` above is what reports NDCG as part of the normal
flow). Scores the model, then prints per-query and mean NDCG@k to stderr —
actual (graded) value vs predicted (model score) value, freely re-runnable
against any RankLib-format split. Pass `--out <path.csv>` for a per-row
`qid, doc_id, rank, actual_grade, predicted_score` table.

## Upload the model to OpenSearch

```bash
MODEL=$(ls -t Models/lambdamart_*.txt | head -1)
python3 scripts/export_model/export_model.py --model "$MODEL"
```

Writes `export/<model>_model_body.json` — the `_createmodel` request body and
nothing else, so it POSTs as-is with no `jq` step:

```bash
curl -X POST "http://localhost:9201/_ltr/_featureset/alamy_phase1/_createmodel" \
    -H 'Content-Type: application/json' \
    --data @export/$(basename "${MODEL%.txt}")_model_body.json
```

If that reports the model already exists, delete it first — `_createmodel` will
not overwrite:

```bash
curl -X DELETE "http://localhost:9201/_ltr/_model/$(basename "${MODEL%.txt}")"
```

The `alamy_phase1` featureset is hardcoded in the script but is **not** emitted
— it is only the reference the model's feature order is checked against, since
RankLib scores by position. It only needs uploading once per cluster, and
already exists on the local bot node. To re-create it elsewhere, dump it from
the script and PUT that:

```bash
python3 -c "import sys, json; sys.path.insert(0, 'scripts/export_model'); \
import export_model as e; print(json.dumps(e.FEATURESET, indent=2))" \
> export/featureset.json

curl -X PUT "http://localhost:9201/_ltr/_featureset/alamy_phase1" \
    -H 'Content-Type: application/json' --data @export/featureset.json
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
