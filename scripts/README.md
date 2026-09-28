# LTR Pipeline

Builds a custom LambdaMART reranking model from graded judgments to `Models/`.

## Steps

| # | Script | Does | Output |
|---|--------|------|--------|
| 1 | `scripts/train/log_judgments_with_features.py` | Reads the combined grade+feature CSV drop and writes it straight to RankLib vectors | `data/logged_with_features.txt` (RankLib format) |
| 2 | `scripts/train/split_judgments.py` | Splits by query (stable qid hash) into train/val/test | `data/splits/{train,val,test}.txt` |
| 3 | `scripts/train/train_lambdamart.py` | Trains LambdaMART with early stopping | `Models/lambdamart_<ts>.txt` (only — no sidecar) |
| 4 | `scripts/export_model/export_model.py` | Builds the `_createmodel` request body | `export/<model>_model_body.json` |
| — | `scripts/val/tune.py` | Optional Bayesian hyperparameter search over shrinkage/leaf/mls/estop, scored by k-fold CV on pooled train+val; ends by retraining the winner on the plain train/val split and reporting its absolute NDCG@k on `val.txt` | `Models/tuning/tuning_report.json` |
| — | `scripts/test/evaluate.py` | Manual, standalone check: scores any model against any RankLib-format split and reports actual-vs-predicted NDCG@k. Nothing in the pipeline calls this automatically — `tune.py` above is what reports NDCG now | printed to stderr; optional per-row CSV via `--out` |

All steps run fully offline against the data drop — nothing here needs a live search API.

**The real data shape:** every drop arrives as one CSV carrying grade, qid, keywords,
doc_id, all 13 feature columns, and a caption/rationale together — see
`data/raw/mock-judgments-601-final.csv`. There's exactly one input shape and one
logging script (step 1) — no separate judgment-list-building stage, since the
feature columns are already on the row.

**Layout:** all pipeline data lives under `data/` — raw drops in `data/raw/`,
the logged RankLib file directly under `data/`, and the train/val/test split
together in `data/splits/`. `Models/` is a separate top-level folder, including
`Models/tuning/` for hyperparameter-search trial models — nothing model-related
lives under `data/`.

## Where the NDCG check lives

There is no dedicated test stage or wrapper anymore. `tune.py` (`val.sh`) is
what reports NDCG: after its hyperparameter search it retrains the winning
config once more on the plain, unpooled train/val split and prints RankLib's
own **absolute** validation NDCG@k for that exact config — the same number
`train_lambdamart.py` would print, not a CV average. It's written to
`Models/tuning/tuning_report.json` as `final_val_ndcg`.

`scripts/test/evaluate.py` still exists as a manual, standalone tool — point
it at any model and any RankLib-format split (`test.txt`, `val.txt`, whatever)
and it reports **absolute** NDCG@k (`ndcg_at_k` in `ltr/metrics.py`, the same
gain/discount formula RankLib itself uses) between the model's predicted
order and the actual graded relevance. It does **not** compare against a
baseline ranking, compute Recall@D, or produce a go/no-go launch
recommendation — there is no live search API or candidate pool in this
pipeline to compare against. Nothing in the pipeline calls it automatically;
run it by hand when you want to check a specific model against a specific file.

## Quickstart

One wrapper per stage directory — settings are variables at the top of each:

```bash
./scripts/train.sh    # log -> split -> train; stops at Models/lambdamart_<ts>.txt
./scripts/val.sh      # hyperparameter search, then reports absolute NDCG@k on val.txt
```

### Running the stages by hand

Every stage is a plain argparse CLI — `--help` on any of them lists its flags.
Run them in this order, from the repo root; each one's output is the next one's
input.

**Setup (once per machine):**

```bash
cd /home/madhav/Projects/training-ltr
source venv/bin/activate          # or: python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
java -version                     # Java is required — the trainer shells out to a jar
```

**Step 1 — log the judgments with features** (reads the raw data drop directly
and writes it straight to RankLib format — this IS the logged judgment list,
grades plus the 13 feature columns, no intermediate file):

```bash
python3 scripts/train/log_judgments_with_features.py \
    --graded-csv data/raw/mock-judgments-601-final.csv \
    --out data/logged_with_features.txt --fmap data/fmap.txt
```
→ `data/logged_with_features.txt` (RankLib format), `data/fmap.txt`
Expect: `queries logged : 100`, `judgments written : 601 (13 features each)`.

**Step 2 — split by query:**

```bash
python3 scripts/train/split_judgments.py \
    --judgments data/logged_with_features.txt --out-dir data/splits \
    --manifest-out data/splits/manifest.json
```
→ `data/splits/train.txt`, `val.txt`, `test.txt`, `manifest.json`
Expect: `76 train / 13 val / 11 test` queries = `452 / 71 / 78` judgments (601 total).

**Step 3 — train (the model comes out here):**

```bash
python3 scripts/train/train_lambdamart.py \
    --train data/splits/train.txt --val data/splits/val.txt \
    --trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100
```
→ `Models/lambdamart_<timestamp>.txt` and nothing else. This stage writes
exactly one file, the RankLib model. Provenance (git SHA, hyperparameters,
feature-list hash, metrics, feature impacts) is printed to stderr instead of
written to a sidecar, so capture the output if you need to attribute the model
later.
Expect: both a `training metric` and a `validation metric` line, plus a
per-feature impact breakdown. Read the validation one — the training figure hits
1.0 within a few rounds on this data and means nothing. Also expect `trees
saved` to be far below `--trees`: with `-validate` the jar keeps the best
iteration, so a 500-cap run legitimately saves ~64 trees.

**Step 4 — build the upload JSON.** Needs the timestamp from step 3 — find it
with `ls Models/`, or capture it:

```bash
MODEL=$(ls -t Models/lambdamart_*.txt | head -1)
python3 scripts/export_model/export_model.py --model "$MODEL"
```
→ `export/<model>_model_body.json`, which is the `_createmodel` request body and
nothing else, so it POSTs as-is:

```bash
curl -X POST "http://localhost:9201/_ltr/_featureset/alamy_phase1/_createmodel" \
    -H 'Content-Type: application/json' \
    --data @export/$(basename "${MODEL%.txt}")_model_body.json
```

The `alamy_phase1` featureset (with the real per-field query templates) is
hardcoded in that script but never emitted — it is the reference the model's
feature order is checked against, since RankLib scores by position. Upload the
featureset separately, once, with `PUT /_ltr/_featureset/alamy_phase1`.

To search hyperparameters instead of guessing them — Gaussian Process regression
+ Probability-of-Improvement acquisition by default, the same technique as
`references folder/bayesian-optimization.ipynb` (there: OpenSearch query boosts;
here: LambdaMART hyperparameters). Each trial's real training result narrows
where the next trial samples, rather than enumerating a grid:

```bash
python3 scripts/val/tune.py \
    --train data/splits/train.txt --val data/splits/val.txt \
    --n-trials 30 --folds 5 \
    --shrinkage-min 0.01 --shrinkage-max 0.3 \
    --leaf-min 2 --leaf-max 50 --mls-min 1 --mls-max 20 \
    --estop-min 10 --estop-max 150
```

Each trial is scored by **k-fold cross-validation over the pooled train+val
queries**, on the held-out fold, not on the 13-query val split — with a split
that small the gap between two configurations is mostly noise. One trial
therefore costs `--folds` jar runs (`--n-trials 30 --folds 5` = 150 training
runs). Once a winner is picked, `tune.py` retrains it *once* more — this time
on the plain, unpooled train/val split — and reports RankLib's own absolute
validation NDCG@k for that run (`final_val_ndcg` in the report JSON, also
printed to stderr). It still never writes a model you'd ship: that retrain,
like the fold models, is throwaway and deleted unless `--out-dir` is given —
`tune.py` prints a ready-to-run `train_lambdamart.py` command for the actual
shippable model.

`--sampler tpe` switches to Optuna's Tree-structured Parzen Estimator instead (a
different Bayesian method; Optuna's own GP sampler needs PyTorch, which this repo has
no other use for, so the default path uses scikit-learn's GP directly). Either way
the training engine is unchanged — still the RankLib jar — so the model format and
OpenSearch export are unaffected by which sampler you pick. `--seed` makes a search
reproducible.

## Repo layout

```
training-ltr/
├── Models/                       # top-level, separate from data/ — nothing here is training data
│   ├── lambdamart_<ts>.txt       # the ONLY thing training writes
│   └── tuning/                   # tune.py's report (fold models are temp-dir'd and deleted)
├── data/
│   ├── raw/                      # data drops as received (grade + features + caption)
│   │   └── mock-judgments-601-final.csv
│   ├── logged_with_features.txt  # step 1 output, RankLib format
│   ├── fmap.txt
│   └── splits/                   # step 2 output
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── manifest.json
├── ltr/                          # shared library — features, metrics, ranklib, judgments
└── scripts/
    ├── train/                    # log_judgments_with_features, split_judgments, train_lambdamart
    ├── val/                      # tune (hyperparameter search + final absolute NDCG@k on val.txt)
    ├── test/                     # evaluate — manual, standalone check; not called by any wrapper
    ├── train.sh                  # log -> split -> train
    ├── val.sh                    # tune
    ├── export_model/             # export_model (featureset + model upload JSON)
    └── README.md
```

## Model information

- **Algorithm:** LambdaMART, RankLib ranker 6, via the RankyMcRankFace jar (auto-downloaded).
- **13 features**, canonical order in `ltr/features.py`: `caption_bm25`, `essential_words_bm25`, `main_words_bm25`, `extra_words_bm25`, `description_bm25`, `cross_fields_score`, `pseudonym_rank`, `image_boost`, `is_archive`, `is_black_and_white`, `date_recency`, `is_premium_collection`, `asset_type_photo`.
- **Default hyperparameters:** `--trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100 --metric NDCG@10`, always trained with `-validate` (never `-kcv` — the jar saves no model when cross-validating). `--trees` is a deliberately high *cap*: early stopping ends training and the jar saves the best validation iteration, so the shipped ensemble is typically much smaller (~64 trees on the current data drop).
- **Training writes one file.** `train_lambdamart.py` emits only `Models/lambdamart_<ts>.txt`. No metadata sidecar, no run-log line. That `.txt` is already a valid `model/ranklib` definition, so it uploads with `jq --rawfile` + `_createmodel` and needs no wrapper. The trade-off: the feature-list hash is no longer recorded on disk, so nothing can later prove a model was trained against today's `ltr/features.py`. RankLib addresses features by position, so a reorder of that file would score garbage silently — keep it under version control and re-train after any change to it.
- **Training vs validation metric.** RankLib prints both. `metric()` is the training-data figure and LambdaMART drives it to 1.0 within a handful of rounds on a set this size; `validation_metric()` is the held-out one. Only the latter is a quality signal, and it is the only thing `tune.py` selects on.
- **Hyperparameter search:** `scripts/val/tune.py` defaults to Gaussian Process regression + Probability-of-Improvement (matching `references folder/bayesian-optimization.ipynb`'s technique), with Optuna's TPE as an alternative (`--sampler tpe`) — over shrinkage/leaf/mls/estop, scored by k-fold CV on the pooled train+val queries (the 13-query val split alone is too small to separate configurations). The test split is never opened. The training engine is unchanged — still the RankLib jar — so this doesn't affect the model format or the upload path.
- **Feature scaling exists but is disabled.** `ltr/features.py`'s `standardize_features()` z-score standardizes continuous features; it's only ever invoked from a commented-out line in `log_judgments_with_features.py`. LambdaMART is scale-invariant (a tree split thresholds one feature at a time, so raw magnitude doesn't change what it learns), so this changes nothing while off. Enabling it would need matching query-time feature scaling in OpenSearch, or training-time and query-time values diverge silently — the same failure mode `feature_list_hash()` guards against for feature *order*, with nothing guarding scale.
- **Stages are plain argparse CLIs.** The `.sh` wrappers just call them in order.
