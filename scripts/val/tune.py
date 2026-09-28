#!/usr/bin/env python3
"""Bayesian hyperparameter search for LambdaMART over shrinkage/leaf/mls/estop.

Each trial is scored by k-fold CV on the pooled train+val queries, using the
held-out fold's validation metric. Two reasons: RankLib's training metric hits
1.0 within a few rounds here, so maximising it selects for memorisation; and
the plain val split is 13 queries, too few to separate configurations. The test
split is never opened. RankLib's own -kcv can't do this — it saves no model.

One trial costs --folds jar runs. Recommend-only: this never writes a model you
could ship, it prints a train_lambdamart.py command.

Samplers: `gp` (default) is Gaussian Process regression + Probability of
Improvement, the technique from `references folder/bayesian-optimization.ipynb`
adapted from query boosts to training hyperparameters. `tpe` is Optuna's
Tree-structured Parzen Estimator. Optuna's own GPSampler needs PyTorch, which
this repo has no other use for.

Usage:
    python3 scripts/val/tune.py \\
        --train data/splits/train.txt --val data/splits/val.txt \\
        --n-trials 30 --folds 5
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ltr.judgments import judgments_open
from ltr.ranklib import trainModel
from ltr.features import FEATURES


def assign_fold(qid, fold_salt, folds):
    # Stable hash, mirroring split_judgments.assign_split, so a search is
    # reproducible across runs.
    return int(hashlib.md5((str(qid) + fold_salt).encode('utf-8')).hexdigest(), 16) % folds


def build_folds(judgments, folds, fold_salt):
    # By query, never by row: a query split across folds would be scored on
    # documents whose siblings the model trained on.
    by_qid = OrderedDict()
    for j in judgments:
        by_qid.setdefault(j.qid, []).append(j)

    judgments_by_fold = [[] for _ in range(folds)]
    qids_by_fold = [[] for _ in range(folds)]
    for qid, rows in by_qid.items():
        fold = assign_fold(qid, fold_salt, folds)
        judgments_by_fold[fold].extend(rows)
        qids_by_fold[fold].append(qid)
    return qids_by_fold, judgments_by_fold


def check_feature_width(judgments, path):
    width = len(FEATURES)
    for j in judgments:
        if len(j.features) != width:
            raise ValueError("%s: qid %s doc %s has %d features, expected %d"
                             % (path, j.qid, j.docId, len(j.features), width))


def run_fold(train_judgments, val_judgments, trees, shrinkage, leaf, mls, estop,
             metric, model_path):
    kwargs = dict(training_set=train_judgments, out=model_path, ranker=6,
                  leafs=leaf, trees=trees, shrinkage=shrinkage,
                  metric2t=metric, metric2T=metric,
                  validation_set=val_judgments, mls=mls)
    if estop is not None:
        kwargs['estop'] = estop
    result = trainModel(**kwargs)
    if not result.trainingLogs:
        raise RuntimeError("ranklib produced no training log; check the jar output above")
    value = result.trainingLogs[-1].validation_metric()
    if value is None:
        raise RuntimeError(
            "ranklib reported no validation metric — expected a '%s on validation data:' "
            "line. Scoring on the training metric instead would select for memorisation."
            % metric)
    return value


def run_cv_trial(qids_by_fold, judgments_by_fold, trees, shrinkage, leaf, mls, estop,
                 metric, model_path_for_fold):
    fold_scores = []
    for i in range(len(judgments_by_fold)):
        held_out = judgments_by_fold[i]
        rest = [j for k, fold in enumerate(judgments_by_fold) if k != i for j in fold]
        fold_scores.append(run_fold(rest, held_out, trees, shrinkage, leaf, mls,
                                    estop, metric, model_path_for_fold(i)))
    return {
        'fold_scores': fold_scores,
        'mean': float(np.mean(fold_scores)),
        'std': float(np.std(fold_scores)),
    }


def build_bounds(args):
    """name -> (low, high, kind), kind being 'log' or 'int'."""
    bounds = {
        'shrinkage': (args.shrinkage_min, args.shrinkage_max, 'log'),
        'leaf': (args.leaf_min, args.leaf_max, 'int'),
        'mls': (args.mls_min, args.mls_max, 'int'),
    }
    if args.estop_min is not None:
        bounds['estop'] = (args.estop_min, args.estop_max, 'int')
    if args.trees_min is not None:
        bounds['trees'] = (args.trees_min, args.trees_max, 'int')
    return bounds


def sample_random_params(bounds, rng):
    params = {}
    for name, (low, high, kind) in bounds.items():
        if kind == 'log':
            params[name] = float(math.exp(rng.uniform(math.log(low), math.log(high))))
        elif kind == 'int':
            params[name] = int(rng.integers(low, high + 1))
        else:
            params[name] = float(rng.uniform(low, high))
    return params


def to_unit_cube(params, bounds):
    # Per-dimension [0,1] scaling so one GP length-scale isn't misled by
    # shrinkage (0.01-0.3) and leaf (2-50) living on different scales.
    x = []
    for name, (low, high, kind) in bounds.items():
        v = params[name]
        if kind == 'log':
            v, low, high = math.log(v), math.log(low), math.log(high)
        x.append((v - low) / (high - low) if high > low else 0.0)
    return x


def optimize_gp(objective_fn, bounds, n_trials, n_startup_trials, theta, n_candidates, seed):
    rng = np.random.default_rng(seed)
    observations = []

    for _ in range(min(n_startup_trials, n_trials)):
        params = sample_random_params(bounds, rng)
        observations.append({'params': params, 'value': objective_fn(params)})

    while len(observations) < n_trials:
        X = np.array([to_unit_cube(o['params'], bounds) for o in observations])
        y = np.array([o['value'] for o in observations])

        gpr = GaussianProcessRegressor(normalize_y=True, random_state=seed)
        gpr.fit(X, y)

        # Candidates are drawn at random rather than off a grid, which does not
        # scale past 2-D.
        candidates = [sample_random_params(bounds, rng) for _ in range(n_candidates)]
        Xc = np.array([to_unit_cube(c, bounds) for c in candidates])
        mu, sigma = gpr.predict(Xc, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        prob_of_improvement = norm.cdf((mu - y.max() - theta) / sigma)
        next_params = candidates[int(np.argmax(prob_of_improvement))]
        observations.append({'params': next_params, 'value': objective_fn(next_params)})

    return observations


def optimize_tpe(objective_fn, bounds, n_trials, n_startup_trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {}
        for name, (low, high, kind) in bounds.items():
            if kind == 'log':
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == 'int':
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)
        return objective_fn(params)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return [{'params': t.params, 'value': t.value} for t in study.trials]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True,
                        help='pooled with --train and re-folded, not used as a fixed split')
    parser.add_argument('--folds', type=int, default=5,
                        help='CV folds over the pooled train+val queries; one trial costs '
                             'this many jar runs (default: %(default)s)')
    parser.add_argument('--fold-salt', default='',
                        help='mixed into the fold hash; change for a different but still '
                             'reproducible fold layout')
    parser.add_argument('--sampler', choices=['gp', 'tpe'], default='gp')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='hyperparameter combinations tried, including startup '
                             '(default: %(default)s)')
    parser.add_argument('--n-startup-trials', type=int, default=8,
                        help='random trials before the surrogate takes over (default: %(default)s)')
    parser.add_argument('--n-candidates', type=int, default=2000,
                        help='[gp] candidate points scored per iteration (default: %(default)s)')
    parser.add_argument('--theta', type=float, default=0.0,
                        help='[gp] exploration margin; higher favours uncertain regions '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trees', type=int, default=500,
                        help='fixed tree cap unless --trees-min/--trees-max are given. Left '
                             'high on purpose: early stopping should end training, not this '
                             'cap (default: %(default)s)')
    parser.add_argument('--trees-min', type=int, default=None,
                        help='adds tree count to the search; largely redundant while estop '
                             'is searched, since patience ends training first')
    parser.add_argument('--trees-max', type=int, default=None)
    parser.add_argument('--shrinkage-min', type=float, default=0.01)
    parser.add_argument('--shrinkage-max', type=float, default=0.3)
    parser.add_argument('--leaf-min', type=int, default=2)
    parser.add_argument('--leaf-max', type=int, default=50)
    parser.add_argument('--mls-min', type=int, default=1)
    parser.add_argument('--mls-max', type=int, default=20)
    parser.add_argument('--estop-min', type=int, default=10)
    parser.add_argument('--estop-max', type=int, default=150)
    parser.add_argument('--no-estop', action='store_true',
                        help='drop patience from the search and never pass -estop')
    parser.add_argument('--metric', default='NDCG@10')
    parser.add_argument('--out-dir', default=None,
                        help='keep fold models here; by default they go to a temp directory '
                             'that is deleted when the search ends')
    parser.add_argument('--report-out', default='Models/tuning/tuning_report.json')
    parser.add_argument('--top-n', type=int, default=3)
    args = parser.parse_args(argv)

    if bool(args.trees_min) != bool(args.trees_max):
        print("ERROR --trees-min and --trees-max must be given together or not at all",
              file=sys.stderr)
        return 1
    if args.n_trials <= 0:
        print("ERROR --n-trials must be positive", file=sys.stderr)
        return 1
    if args.folds < 2:
        print("ERROR --folds must be at least 2", file=sys.stderr)
        return 1
    if args.no_estop:
        args.estop_min = None

    with judgments_open(args.train) as r:
        train_judgments = list(r)
    with judgments_open(args.val) as r:
        val_judgments = list(r)
    if not train_judgments or not val_judgments:
        print("ERROR empty train or val split", file=sys.stderr)
        return 1

    try:
        check_feature_width(train_judgments, args.train)
        check_feature_width(val_judgments, args.val)
    except ValueError as e:
        print("ERROR %s" % e, file=sys.stderr)
        return 1

    pooled = train_judgments + val_judgments
    qids_by_fold, judgments_by_fold = build_folds(pooled, args.folds, args.fold_salt)

    empty = [i for i, q in enumerate(qids_by_fold) if not q]
    if empty:
        print("ERROR fold(s) %s came out empty — reduce --folds or change --fold-salt"
              % empty, file=sys.stderr)
        return 1
    thin = [i for i, q in enumerate(qids_by_fold) if len(q) < 2]
    if thin:
        print("WARNING fold(s) %s hold fewer than 2 queries; their scores will be very "
              "noisy. Consider reducing --folds." % thin, file=sys.stderr)

    total_queries = sum(len(q) for q in qids_by_fold)
    print("pooled queries   : %d (%d train + %d val judgments)"
          % (total_queries, len(train_judgments), len(val_judgments)), file=sys.stderr)
    print("folds            : %d, queries per fold %s"
          % (args.folds, [len(q) for q in qids_by_fold]), file=sys.stderr)
    print("jar runs planned : %d trials x %d folds = %d\n"
          % (args.n_trials, args.folds, args.n_trials * args.folds), file=sys.stderr)

    keep_models = args.out_dir is not None
    work_dir = args.out_dir if keep_models else tempfile.mkdtemp(prefix='ltr_tune_')
    os.makedirs(work_dir, exist_ok=True)

    bounds = build_bounds(args)
    trial_counter = [0]
    outcomes_by_number = {}

    def scalar_objective(params):
        n = trial_counter[0]
        trial_counter[0] += 1
        trees = int(round(params['trees'])) if 'trees' in params else args.trees
        estop = int(round(params['estop'])) if 'estop' in params else None

        def model_path_for_fold(fold):
            return os.path.join(work_dir, 'trial_%03d_fold%d.txt' % (n, fold))

        cv = run_cv_trial(qids_by_fold, judgments_by_fold, trees, params['shrinkage'],
                          params['leaf'], params['mls'], estop, args.metric,
                          model_path_for_fold)
        outcomes_by_number[n] = {
            'number': n, 'trees': trees, 'shrinkage': params['shrinkage'],
            'leaf': params['leaf'], 'mls': params['mls'], 'estop': estop,
            'metric': args.metric, 'metric_value': cv['mean'], 'metric_std': cv['std'],
            'fold_scores': cv['fold_scores'],
        }
        print("[trial %d/%d] shrinkage=%.4f leaf=%d mls=%d estop=%s trees=%d -> %s %.4f "
              "(+/-%.4f across folds)"
              % (n + 1, args.n_trials, params['shrinkage'], params['leaf'], params['mls'],
                 estop, trees, args.metric, cv['mean'], cv['std']), file=sys.stderr)
        return cv['mean']

    try:
        if args.sampler == 'gp':
            optimize_gp(scalar_objective, bounds, args.n_trials, args.n_startup_trials,
                        args.theta, args.n_candidates, args.seed)
        else:
            optimize_tpe(scalar_objective, bounds, args.n_trials, args.n_startup_trials,
                         args.seed)
    finally:
        if not keep_models:
            shutil.rmtree(work_dir, ignore_errors=True)

    trials_sorted = sorted(outcomes_by_number.values(), key=lambda o: o['metric_value'],
                           reverse=True)
    best = trials_sorted[0]

    # Every trial above is CV over the pooled train+val queries — useful for
    # comparing configurations against each other, but not a score against
    # val.txt itself. Retrain once with the winning config on the ordinary
    # (unpooled) train/val split and read RankLib's own validation NDCG for
    # it — absolute, the same number train_lambdamart.py would print for
    # this exact config, not CV-averaged across folds.
    final_model_path = os.path.join(work_dir, 'final_check.txt')
    final_kwargs = dict(training_set=train_judgments, out=final_model_path, ranker=6,
                        leafs=best['leaf'], trees=best['trees'], shrinkage=best['shrinkage'],
                        metric2t=args.metric, metric2T=args.metric,
                        validation_set=val_judgments, mls=best['mls'])
    if best['estop'] is not None:
        final_kwargs['estop'] = best['estop']
    final_result = trainModel(**final_kwargs)
    final_val_ndcg = None
    if not final_result.trainingLogs:
        print("WARNING final check retrain produced no training log; skipping", file=sys.stderr)
    else:
        final_val_ndcg = final_result.trainingLogs[-1].validation_metric()

    report = {
        'train': args.train,
        'val': args.val,
        'selection': 'mean %s on held-out fold, %d-fold CV over pooled train+val queries'
                     % (args.metric, args.folds),
        'folds': args.folds,
        'fold_salt': args.fold_salt,
        'fold_query_counts': [len(q) for q in qids_by_fold],
        'pooled_queries': total_queries,
        'metric': args.metric,
        'sampler': args.sampler,
        'seed': args.seed,
        'n_trials': args.n_trials,
        'n_startup_trials': args.n_startup_trials,
        'theta': args.theta if args.sampler == 'gp' else None,
        'search_space': {
            'shrinkage': [args.shrinkage_min, args.shrinkage_max],
            'leaf': [args.leaf_min, args.leaf_max],
            'mls': [args.mls_min, args.mls_max],
            'estop': ([args.estop_min, args.estop_max] if args.estop_min is not None
                      else None),
            'trees': ([args.trees_min, args.trees_max] if args.trees_min is not None
                      else args.trees),
        },
        'trials': trials_sorted,
        'best': best,
        'final_val_ndcg': final_val_ndcg,
        'tuned_at': datetime.now(timezone.utc).isoformat(),
    }
    report_dir = os.path.dirname(os.path.abspath(args.report_out))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_out, 'w') as f:
        json.dump(report, f, indent=2)

    print("\ntop %d by mean held-out %s:" % (min(args.top_n, len(trials_sorted)), args.metric),
          file=sys.stderr)
    for t in trials_sorted[:args.top_n]:
        print("  shrinkage=%-8.4f leaf=%-3d mls=%-3d estop=%-5s trees=%-4d -> %.4f +/- %.4f"
              % (t['shrinkage'], t['leaf'], t['mls'], t['estop'], t['trees'],
                 t['metric_value'], t['metric_std']), file=sys.stderr)

    spread = trials_sorted[0]['metric_value'] - trials_sorted[-1]['metric_value']
    if spread < best['metric_std']:
        print("\nNOTE best-to-worst spread across trials (%.4f) is smaller than the winner's "
              "own fold-to-fold spread (%.4f) — these configurations are not meaningfully "
              "separated on this data, so treat the winner as arbitrary among the top few."
              % (spread, best['metric_std']), file=sys.stderr)

    if final_val_ndcg is not None:
        print("\nfinal check — winning config retrained on the ordinary train/val split:",
              file=sys.stderr)
        print("  %s on val.txt (absolute): %.4f" % (args.metric, final_val_ndcg), file=sys.stderr)

    print("\nbest config found — train the model you ship with it, on the ordinary "
          "train/val split:", file=sys.stderr)
    estop_arg = ' --estop %d' % best['estop'] if best['estop'] is not None else ' --estop 0'
    print("  python3 scripts/train/train_lambdamart.py --train %s --val %s \\\n"
          "      --trees %d --leaf %d --shrinkage %.4f --mls %d%s --metric %s"
          % (args.train, args.val, best['trees'], best['leaf'],
             best['shrinkage'], best['mls'], estop_arg, args.metric), file=sys.stderr)
    print("\nwrote %s" % args.report_out, file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
