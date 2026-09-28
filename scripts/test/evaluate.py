#!/usr/bin/env python3
"""Score a test split with a trained model and report actual-vs-predicted NDCG@k."""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ltr.judgments import judgments_open as jdgment_open
from ltr.ranklib import check_for_rankymcrankface
from ltr.metrics import ndcg_at_k


def scr_test_set(jar_path, model_path, test_path, score_path):
    """Score the test rows and return aligned judgments and model scores."""
    cmd = ['java', '-jar', jar_path, '-load', model_path, '-rank', test_path,
           '-score', score_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("scoring jar failed (exit %d): %s" %
                            (proc.returncode, (proc.stderr or proc.stdout)[-2000:]))

    with jdgment_open(test_path) as reader:
        judgments = list(reader)

    with open(score_path) as f:
        lines = [line for line in f if line.strip()]

    if len(lines) != len(judgments):
        raise ValueError(
            "score file has %d row(s) but test split has %d judgment(s) — "
            "the -score output must have exactly one line per input row, in "
            "the same order, or scores cannot be safely zipped onto "
            "judgments" % (len(lines), len(judgments)))

    scores = []
    for i, line in enumerate(lines):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3:
            raise ValueError("malformed score line %d: %r" % (i, line))
        file_qid = int(parts[0])
        if file_qid != judgments[i].qid:
            raise ValueError(
                "score/judgment qid mismatch at row %d: score file says "
                "qid=%s, judgments[%d].qid=%s — scores would be attributed "
                "to the wrong query if this were ignored"
                % (i, file_qid, i, judgments[i].qid))
        scores.append(float(parts[2]))

    return judgments, scores


def ltr_ord(doc_ids, scores_by_doc):
    """Order by model score, breaking ties by document ID."""
    return sorted(doc_ids, key=lambda d: (-scores_by_doc[d], d))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test', default='data/splits/test.txt')
    parser.add_argument('--model', required=True)
    parser.add_argument('--k', type=int, default=10, help='the @k for NDCG (default: %(default)s)')
    parser.add_argument('--out', help='optional CSV path for the per-row actual-vs-predicted table')
    args = parser.parse_args(argv)

    if not os.path.exists(args.test):
        print("ERROR test split not found: %s" % args.test, file=sys.stderr)
        return 1
    if not os.path.exists(args.model):
        print("ERROR model file not found: %s" % args.model, file=sys.stderr)
        return 1

    jar_path = check_for_rankymcrankface()
    score_fd, score_path = tempfile.mkstemp(suffix='.scores.txt')
    os.close(score_fd)
    try:
        judgments, scores = scr_test_set(jar_path, args.model, args.test, score_path)
    except (RuntimeError, ValueError) as e:
        print("ERROR %s" % e, file=sys.stderr)
        return 1
    finally:
        if os.path.exists(score_path):
            os.remove(score_path)

    if not judgments:
        print("ERROR test split %s is empty" % args.test, file=sys.stderr)
        return 1

    by_qid = defaultdict(list)
    for j, s in zip(judgments, scores):
        by_qid[j.qid].append((j, s))

    rows = []
    ndcgs = []
    excluded = 0
    for qid in sorted(by_qid.keys()):
        qid_rows = by_qid[qid]
        doc_ids = [j.docId for j, _ in qid_rows]
        grade_by_doc = {j.docId: j.grade for j, _ in qid_rows}
        score_by_doc = {j.docId: s for j, s in qid_rows}
        ranked_ids = ltr_ord(doc_ids, score_by_doc)

        for rank, doc_id in enumerate(ranked_ids, start=1):
            rows.append({
                'qid': qid,
                'doc_id': doc_id,
                'rank': rank,
                'actual_grade': grade_by_doc[doc_id],
                'predicted_score': score_by_doc[doc_id],
            })

        ndcg = ndcg_at_k([grade_by_doc[d] for d in ranked_ids], args.k)
        if ndcg is None:
            excluded += 1
            continue
        ndcgs.append(ndcg)
        print("qid %s: NDCG@%d = %.4f" % (qid, args.k, ndcg), file=sys.stderr)

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['qid', 'doc_id', 'rank', 'actual_grade', 'predicted_score'])
            writer.writeheader()
            writer.writerows(rows)
        print("wrote %s" % args.out, file=sys.stderr)

    mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else None

    print(file=sys.stderr)
    print("queries scored    : %d" % len(by_qid), file=sys.stderr)
    print("queries excluded  : %d (no relevant doc in list; NDCG undefined)" % excluded, file=sys.stderr)
    print("mean NDCG@%-3d     : %s" % (args.k, mean_ndcg), file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
