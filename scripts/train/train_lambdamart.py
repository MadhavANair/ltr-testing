#!/usr/bin/env python3
"""Train LambdaMART. Writes one file: Models/<name>_<timestamp>.txt, the
RankLib model, which is also a valid OpenSearch `model/ranklib` definition.
Provenance is printed to stderr, not written to disk.

A validation set is mandatory, and never use -kcv for a model you intend to
ship: the jar saves no model when cross-validating.

Usage:
    python3 scripts/train/train_lambdamart.py \\
        --train data/splits/train.txt --val data/splits/val.txt \\
        --trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from ltr.judgments import judgments_open
from ltr.ranklib import trainModel
from ltr.features import FEATURES, feature_list_hash


def git_sha(repo_root):
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, cwd=repo_root)
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def verify_feature_widths(judgments, features, source_label):
    expected = len(features)
    errors = []
    for j in judgments:
        if len(j.features) != expected:
            errors.append("%s qid=%s doc=%s: %d features, expected %d"
                          % (source_label, j.qid, j.docId, len(j.features), expected))
    return errors


def build_impacts_by_name(log, features):
    # RankLib keys impacts by 1-based index as a string; this repo's fmap is
    # 0-based, so without the offset the report blames the neighbouring feature.
    by_name = {}
    for key, impact in log.impacts.items():
        idx = int(key) - 1
        if idx < 0 or idx >= len(features):
            raise ValueError("feature impact key %r out of range for %d features"
                              % (key, len(features)))
        by_name[features[idx].name] = impact
    return dict(sorted(by_name.items(), key=lambda kv: kv[1], reverse=True))


def count_trees(model_path):
    try:
        with open(model_path) as f:
            return f.read().count('<tree id=')
    except OSError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True, help='training set, RankLib format')
    parser.add_argument('--val', required=True,
                        help='validation set, RankLib format (required — training without '
                             'one trains until the model has memorised the training data)')
    parser.add_argument('--out-dir', default='Models')
    parser.add_argument('--name', default='lambdamart', help='filename prefix')
    parser.add_argument('--trees', type=int, default=500,
                        help='upper bound on boosting rounds; with --estop the jar saves the '
                             'best validation iteration, usually far fewer (default: %(default)s)')
    parser.add_argument('--leaf', type=int, default=10)
    parser.add_argument('--shrinkage', type=float, default=0.05)
    parser.add_argument('--mls', type=int, default=5)
    parser.add_argument('--metric', default='NDCG@10')
    parser.add_argument('--tc', type=int, help='threshold candidates for tree splitting')
    parser.add_argument('--estop', type=int, default=100,
                        help='stop after this many rounds with no validation improvement '
                             '(default: %(default)s); 0 disables')
    args = parser.parse_args(argv)

    if not os.path.exists(args.train):
        print("ERROR training file not found: %s" % args.train, file=sys.stderr)
        return 1
    if not os.path.exists(args.val):
        print("ERROR validation file not found: %s" % args.val, file=sys.stderr)
        return 1

    with judgments_open(args.train, 'r') as reader:
        train_judgments = list(reader)
    with judgments_open(args.val, 'r') as reader:
        val_judgments = list(reader)

    if not train_judgments:
        print("ERROR training file has no judgments: %s" % args.train, file=sys.stderr)
        return 1
    if not val_judgments:
        print("ERROR validation file has no judgments: %s" % args.val, file=sys.stderr)
        return 1

    errors = verify_feature_widths(train_judgments, FEATURES, 'train')
    errors += verify_feature_widths(val_judgments, FEATURES, 'val')
    if errors:
        for err in errors[:20]:
            print("ERROR %s" % err, file=sys.stderr)
        print("ERROR %d feature-width mismatch(es) against %d-feature contract; "
              "nothing trained." % (len(errors), len(FEATURES)), file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(args.out_dir, '%s_%s.txt' % (args.name, timestamp))

    estop = args.estop or None
    result = trainModel(training_set=train_judgments, validation_set=val_judgments,
                        out=model_path, ranker=6, trees=args.trees, leafs=args.leaf,
                        shrinkage=args.shrinkage, mls=args.mls, tc=args.tc, estop=estop,
                        metric2t=args.metric, metric2T=args.metric)

    if not result.trainingLogs:
        print("ERROR ranklib produced no training log; check the jar output above",
              file=sys.stderr)
        return 1
    log = result.trainingLogs[-1]

    if not os.path.exists(model_path):
        print("ERROR ranklib did not write a model file at %s" % model_path,
              file=sys.stderr)
        return 1

    impacts_by_name = build_impacts_by_name(log, FEATURES)

    print("\n=== provenance ===", file=sys.stderr)
    print("trained at       : %s" % datetime.now(timezone.utc).isoformat(), file=sys.stderr)
    print("git sha          : %s" % git_sha(REPO_ROOT), file=sys.stderr)
    print("train judgments  : %d (%s)" % (len(train_judgments), args.train), file=sys.stderr)
    print("val judgments    : %d (%s)" % (len(val_judgments), args.val), file=sys.stderr)
    print("features         : %d (hash %s)" % (len(FEATURES), feature_list_hash(FEATURES)),
          file=sys.stderr)
    print("hyperparameters  : trees=%s leaf=%s shrinkage=%s mls=%s tc=%s estop=%s metric=%s ranker=6"
          % (args.trees, args.leaf, args.shrinkage, args.mls, args.tc, estop, args.metric),
          file=sys.stderr)
    print("trees saved      : %s (of --trees %d; -validate keeps the best iteration)"
          % (count_trees(model_path), args.trees), file=sys.stderr)
    print("training metric  : %s = %s  <- on data the model has seen; not a quality signal"
          % (log.trainMetricName, log.metric()), file=sys.stderr)
    print("validation metric: %s = %s  <- the honest score"
          % (log.validationMetricName, log.validation_metric()), file=sys.stderr)
    print("feature impacts  : %s" % impacts_by_name, file=sys.stderr)
    print("\nmodel            : %s" % model_path, file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
