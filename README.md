# LTR Pipeline

Builds a LambdaMART reranking model from graded judgements, gates it on NDCG,
and uploads it to the `alamy_phase1` featureset. Run everything from
`opensearch-ltr/`. Everything up to `deploy/` runs offline.

## Setup (once per machine)

```bash
source venv/bin/activate          # or: python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
java -version                     # the trainer shells out to the RankyMcRankFace jar
```

`tune/tune.sh` and `deploy/upload-model` also need `jq`.

## Flow

```
judgements/judgements.sh  ->  tune/tune.sh (runs train/train.sh)  ->  test/test.sh  ->  deploy/upload-model
                              or train/train.sh on its own
```

| Stage | Script | Input | Output |
|---|---|---|---|
| judgements | `judgements/judgements.sh` (`log.py` per set) | `judgements/data/raw/{train,val,test}/*.csv` | `judgements/data/graded/train/train.txt`, `judgements/data/graded/val/val.txt` (each set's CSVs combined), `judgements/data/graded/test/<csv name>.txt` (one per test CSV) |
| train | `train/train.sh` (`train_lambdamart.py`) | `judgements/data/graded/{train/train,val/val}.txt` | `train/trained_models/lambdamart_<ts>.txt` |
| tune | `tune/tune.sh` (`tune.py`, then `train/train.sh` with the best params) | `judgements/data/graded/{train/train,val/val}.txt` | `tune/tuning_report.json`, `train/trained_models/lambdamart_<ts>.txt` |
| test | `test/test.sh` (`evaluate.py` per model) | `train/trained_models/lambdamart_*.txt`, `judgements/data/graded/test/*.txt` | models passing **every** test set **moved** to `deploy/models/`; others stay put |
| deploy | `deploy/upload-model [model]` | newest `deploy/models/lambdamart_*.txt`, `deploy/featureset.json` | featureset + model on `localhost:9201` (nothing on disk) |
| deploy | `deploy/delete-model [model or name]` | model name | model removed from `localhost:9201` |

Settings (hyperparameters, threshold) are variables at the top of
each `.sh`. Every `.py` is an argparse CLI; `--help` lists its flags.

## Layout

```
opensearch-ltr/
├── judgements/
│   ├── data/
│   │   ├── raw/                      # CSVs from the content team, as received
│   │   │   ├── train/                # 10-20 datasets, combined for training
│   │   │   ├── val/                  # combined; early stopping + tuning
│   │   │   └── test/                 # each dataset gated separately
│   │   └── graded/                   # RankLib format, written by judgements.sh
│   │       ├── train/train.txt
│   │       ├── val/val.txt
│   │       └── test/<csv name>.txt
│   ├── log.py                        # CSVs -> one RankLib file; QID + overlap checks
│   └── judgements.sh
├── train/
│   ├── train_lambdamart.py
│   ├── train.sh
│   └── trained_models/               # every trained model; test.sh moves passing ones out
├── tune/
│   ├── tune.py                       # Bayesian (GP) hyperparameter search
│   ├── tune.sh
│   └── tuning_report.json
├── test/
│   ├── evaluate.py                   # NDCG@k on one test set; exit 0 pass / 2 below --min-ndcg
│   └── test.sh
├── deploy/
│   ├── models/                       # models that passed test.sh
│   ├── featureset.json               # alamy_phase1 feature definitions (query templates)
│   ├── upload-model
│   └── delete-model
└── ltr/                              # shared library: features, metrics, ranklib, judgments
```

`judgements/data/`, `train/trained_models/`, `tune/tuning_report.json`,
and `deploy/models/` are gitignored.

## Stage notes

**judgements.** The content team delivers train, val and test CSVs
separately; there is no split step. Each row carries Grade, QID, Keywords,
Image ID, all 13 feature columns and a caption. `log.py` fails (writes
nothing) if a QID appears with two different keywords anywhere in its inputs
(QIDs must be globally unique), or if a val query shares a QID or keywords
with train, or a test query with train or val. Each run clears and rewrites
`graded/`.

**train.** Writes only the model `.txt`, which is already a valid
`model/ranklib` definition. `--trees` is a cap: with `--estop` the jar keeps
the best validation iteration (~64 trees on the current drop). Read RankLib's
validation NDCG, not the training figure, which hits 1.0 within a few rounds.
Hyperparameters can be overridden by env var (`SHRINKAGE=0.1 ./train/train.sh`).

**tune.** Scores each trial by k-fold CV over pooled train+val queries , never
opens the test sets, then retrains the winner on plain train/val and records
`final_val_ndcg`. `tune.sh` then passes `.best` from the report to
`train/train.sh`. Cost is `N_TRIALS x FOLDS` jar runs (150 by default);
quick look: `N_TRIALS=5 FOLDS=3 ./tune/tune.sh`. `--sampler tpe` swaps in
Optuna's TPE.

**test.** Mean NDCG@10 must be >= 0.7 on **every** file in `graded/test/`;
failing any one keeps the model out of `deploy/models/`. That is the only check
implemented; `evaluate.py --out <csv>` adds a per-row actual-vs-predicted table.

**deploy.** `upload-model` waits for `:9201`, checks `featureset.json`'s
feature order against `ltr/features.py`, creates the `alamy_phase1` featureset
from it if missing (refuses if the cluster's copy differs), then wraps the model
`.txt` with `jq --rawfile` and POSTs it to `_createmodel`. If `alamy_phase1` already has a model, it skips; remove
that one first with `delete-model <name>`.

## After test.sh: deploying a model

Run from `opensearch-ltr/`. Everything targets the bot node on `localhost:9201`.

**0. Start the cluster** (skip if it's already up). From
`opensearch-ingest/opensearch-ingest/local-dev/`:

```bash
docker compose up -d       # OpenSearch user (:9200) + bot (:9201), SQS, DynamoDB
./scripts/create-index     # first time only; "already exists" errors are harmless
```

**1. See which models passed the gate.** Only these can be uploaded by default.
Newest first; the newest is what `upload-model` picks with no argument.

```bash
ls -t deploy/models/
```

**2. See what's on the cluster.** `alamy_phase1` holds one model at a time.

```bash
curl -s "localhost:9201/_ltr/_model?size=100" | jq -r '.hits.hits[]._source.name'
```

**3. Remove the current model** if step 2 listed one. `upload-model` skips
rather than replace, and `_createmodel` can't overwrite. Always pass the name:
with no argument `delete-model` deletes the newest name in `deploy/models/`,
which may not be the one on the cluster. The featureset is left in place.

```bash
./deploy/delete-model <name from step 2>
```

**4. Upload.** Checks `deploy/featureset.json` order against `ltr/features.py`,
creates the `alamy_phase1` featureset if missing (stops if the cluster's copy
differs), then sends the model `.txt` as-is. Nothing is written to disk.

```bash
./deploy/upload-model                                  # newest in deploy/models/
./deploy/upload-model deploy/models/lambdamart_<ts>.txt   # or a specific one
```

**5. Confirm it's live.** Should print the name you just uploaded.

```bash
curl -s "localhost:9201/_ltr/_model?size=100" | jq -r '.hits.hits[]._source.name'
```

**6. Query with it.** `sltr` scores each document with the model's 13
features. Until the featureset templates match the index fields (see
*Featureset vs index* below), every feature reads 0 and all hits get the same
score.

```bash
curl -s "localhost:9201/images/_search" -H 'Content-Type: application/json' -d '{
  "size": 5,
  "query": {"sltr": {"params": {"keywords": "london bus"}, "model": "lambdamart_<ts>"}}
}' | jq '[.hits.hits[] | {id: ._id, score: ._score}]'
```

## Model information

- **Algorithm:** LambdaMART (RankLib ranker 6), trained with `-validate`, never `-kcv` (the jar saves no model when cross-validating).
- **13 features**, order fixed in `ltr/features.py`: `caption_bm25`, `essential_words_bm25`, `main_words_bm25`, `extra_words_bm25`, `description_bm25`, `cross_fields_score`, `pseudonym_rank`, `image_boost`, `is_archive`, `is_black_and_white`, `date_recency`, `is_premium_collection`, `asset_type_photo`. RankLib addresses features by position and nothing on disk records which order a model was trained on. `deploy/upload-model` refuses to run if `deploy/featureset.json`'s order diverges from `ltr/features.py`. Retrain after any change to that file.
- **Feature scaling is off.** `standardize_features()` exists but is only called from a commented-out line in `judgements/log.py`. LambdaMART is scale-invariant. Enabling it would need matching query-time scaling.
- **Featureset vs index:** the featureset uses snake_case fields while the bot index mapping is PascalCase, and `dynamic: false` makes unmatched features resolve to 0. Live scores are not yet comparable to training-time values.
