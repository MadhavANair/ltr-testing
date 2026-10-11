#!/usr/bin/env python3
"""Train LambdaMART. Writes one file: train/trained_models/<name>_<timestamp>.txt, the
RankLib model, which is also a valid OpenSearch `model/ranklib` definition.
The model path and its sha256 are printed to stderr.

Usage:
    python3 train/train_lambdamart.py \\
        --train judgements/data/graded/train/train.txt --val judgements/data/graded/val/val.txt \\
        --trees 500 --leaf 10 --shrinkage 0.05 --mls 5 --estop 100
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ltr.judgments import judgments_open
from ltr.ranklib import trainModel
from ltr.features import FEATURES


def model_sha(model_path):
    with open(model_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_feature_widths(judgments, features, source_label):
    expected = len(features)
    errors = []
    for j in judgments:
        if len(j.features) != expected:
            errors.append("%s qid=%s doc=%s: %d features, expected %d"
                          % (source_label, j.qid, j.docId, len(j.features), expected))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', required=True, help='training set, RankLib format')
    parser.add_argument('--val', required=True,
                        help='validation set, RankLib format (required — training without '
                             'one trains until the model has memorised the training data)')
    parser.add_argument('--out-dir', default='train/trained_models')
    parser.add_argument('--name', default='lambdamart', help='filename prefix')
    parser.add_argument('--trees', type=int, default=500,  # madeup value
                        help='upper bound on boosting rounds; with --estop the jar saves the '
                             'best validation iteration, usually far fewer (default: %(default)s)')
    parser.add_argument('--leaf', type=int, default=10)  # madeup value
    parser.add_argument('--shrinkage', type=float, default=0.05)  # madeup value
    parser.add_argument('--mls', type=int, default=5)  # madeup value
    parser.add_argument('--metric', default='NDCG@10')  # madeup value
    parser.add_argument('--tc', type=int, help='threshold candidates for tree splitting')
    parser.add_argument('--estop', type=int, default=100,  # madeup value
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
    # actual training: runs the RankLib LambdaMART jar and writes the model file
    result = trainModel(training_set=train_judgments, validation_set=val_judgments,
                        out=model_path, ranker=6, trees=args.trees, leafs=args.leaf,
                        shrinkage=args.shrinkage, mls=args.mls, tc=args.tc, estop=estop,
                        metric2t=args.metric, metric2T=args.metric)

    if not result.trainingLogs:
        print("ERROR ranklib produced no training log; check the jar output above",
              file=sys.stderr)
        return 1

    if not os.path.exists(model_path):
        print("ERROR ranklib did not write a model file at %s" % model_path,
              file=sys.stderr)
        return 1

    print("model            : %s" % model_path, file=sys.stderr)

    # --- model sha: identity of this model; upload-model uses it to skip duplicates ---
    print("model sha256     : %s" % model_sha(model_path), file=sys.stderr)
    # ---------------------------------------------------------------------------------

    return 0


if __name__ == '__main__':
    sys.exit(main())
