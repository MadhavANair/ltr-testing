#!/usr/bin/env python3
"""Split judgments into train / val / test (step 5 of the pipeline).

Splits a logged (RankLib-format) judgment file by *query*, never by row — a
query present in both train and test lets the model be graded on queries it
has effectively already seen, which inflates the score.

The split is a stable hash of the qid, so the test set survives every future
re-logging of the same queries without having to be pinned by hand; a
deliberate full re-split is forced only by changing --seed-salt.

It also re-applies the uniform-grade and single-doc guardrails that
log_judgments_with_features.py already enforces upstream — defence in depth, in case this
file was produced some other way (e.g. sourced live) and skipped that step.

Usage:
    python scripts/train/split_judgments.py \\
        --judgments data/logged_with_features.txt \\
        --out-dir data/splits \\
        --train-pct 70 --val-pct 15 --test-pct 15 \\
        --manifest-out data/splits/manifest.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ltr.judgments import judgments_open

SPLITS = ('train', 'val', 'test')


def assign_split(qid, seed_salt, train_pct, val_pct):
    """Bucket a qid via md5, not hash(), so it is stable across processes."""
    bucket = int(hashlib.md5((str(qid) + seed_salt).encode('utf-8')).hexdigest(), 16) % 100
    if bucket < train_pct:
        return 'train'
    elif bucket < train_pct + val_pct:
        return 'val'
    else:
        return 'test'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--judgments', '--judgements', dest='judgments', required=True,
                        help='logged judgment list, RankLib format (from log_judgments_with_features.py)')
    parser.add_argument('--out-dir', default='data/splits',
                        help='directory to write train.txt / val.txt / test.txt into')
    parser.add_argument('--train-pct', type=int, default=70)  # madeup value
    parser.add_argument('--val-pct', type=int, default=15)  # madeup value
    parser.add_argument('--test-pct', type=int, default=15)  # madeup value
    parser.add_argument('--seed-salt', default='',
                        help='mixed into the qid hash; change to force a deliberate full re-split without touching code')
    parser.add_argument('--min-distinct-grades', type=int, default=2,
                        help='drop any qid with fewer distinct grade values than this')
    parser.add_argument('--min-docs', type=int, default=2,
                        help='drop any qid with fewer documents than this')
    parser.add_argument('--manifest-out', default='data/splits/manifest.json')
    args = parser.parse_args(argv)

    if args.train_pct + args.val_pct + args.test_pct != 100:
        print("ERROR --train-pct/--val-pct/--test-pct must sum to 100 (got %d)"
              % (args.train_pct + args.val_pct + args.test_pct), file=sys.stderr)
        return 1

    by_qid = OrderedDict()
    with judgments_open(args.judgments, mode='r') as reader:
        for judgment in reader:
            by_qid.setdefault(judgment.qid, []).append(judgment)

    if not by_qid:
        print("ERROR no judgments read from %s" % args.judgments, file=sys.stderr)
        return 1

    dropped_single_doc = []
    dropped_uniform_grade = []
    kept_by_qid = OrderedDict()
    for qid, judgments in by_qid.items():
        if len(judgments) < args.min_docs:
            dropped_single_doc.append(qid)
            continue
        distinct_grades = {j.grade for j in judgments}
        if len(distinct_grades) < args.min_distinct_grades:
            dropped_uniform_grade.append(qid)
            continue
        kept_by_qid[qid] = judgments

    print("qids read           : %d" % len(by_qid), file=sys.stderr)
    print("dropped single_doc  : %d %s" % (len(dropped_single_doc), dropped_single_doc or ''), file=sys.stderr)
    print("dropped uniform_grade: %d %s" % (len(dropped_uniform_grade), dropped_uniform_grade or ''), file=sys.stderr)
    print("qids kept           : %d" % len(kept_by_qid), file=sys.stderr)

    if not kept_by_qid:
        print("ERROR every qid was dropped; nothing to split", file=sys.stderr)
        return 1

    split_judgments = {name: [] for name in SPLITS}
    split_qids = {name: [] for name in SPLITS}
    for qid, judgments in kept_by_qid.items():
        split = assign_split(qid, args.seed_salt, args.train_pct, args.val_pct)
        split_judgments[split].extend(judgments)
        split_qids[split].append(qid)

    empty_splits = [name for name, qids in split_qids.items() if not qids]
    if empty_splits:
        print("ERROR split(s) ended up empty: %s" % empty_splits, file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_paths = {name: os.path.join(args.out_dir, '%s.txt' % name) for name in SPLITS}
    hashes = {}
    for name, path in out_paths.items():
        with judgments_open(path, mode='w') as writer:
            writer.write(judgments=split_judgments[name])
        with open(path, 'rb') as f:
            hashes[name] = hashlib.sha256(f.read()).hexdigest()[:16]

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
    manifest = {
        'train_pct': args.train_pct,
        'val_pct': args.val_pct,
        'test_pct': args.test_pct,
        'seed_salt': args.seed_salt,
        'train_qids': sorted(split_qids['train']),
        'val_qids': sorted(split_qids['val']),
        'test_qids': sorted(split_qids['test']),
        'train_hash': hashes['train'],
        'val_hash': hashes['val'],
        'test_hash': hashes['test'],
        'dropped_uniform_grade': sorted(dropped_uniform_grade),
        'dropped_single_doc': sorted(dropped_single_doc),
        'min_distinct_grades': args.min_distinct_grades,
        'min_docs': args.min_docs,
        'split_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(args.manifest_out, 'w') as f:
        json.dump(manifest, f, indent=2)

    for name in SPLITS:
        print("%-20s: %d (%d judgments)" % (name + ' qids', len(split_qids[name]),
                                             len(split_judgments[name])), file=sys.stderr)
    for path in out_paths.values():
        print("wrote %s" % path, file=sys.stderr)
    print("wrote %s" % args.manifest_out, file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
