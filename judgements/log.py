#!/usr/bin/env python3
"""Converts one or more graded CSVs into one RankLib-format file.

Each CSV row carries grade, QID, keywords, Image ID and the 13 feature
columns (see ltr/features.py for the contract). All inputs are combined
into --out. QIDs must be globally unique: the same QID with different
keywords is an error.

--disjoint-from lists the CSVs of the other sets (train/val when logging
test, etc.); any query whose QID or keywords appears there is an error, so
the set being logged never shares a query with them.

Usage:
    python3 judgements/log.py \\
        --graded-csv judgements/data/raw/test/<name>.csv \\
        --disjoint-from judgements/data/raw/train/*.csv judgements/data/raw/val/*.csv \\
        --out judgements/data/graded/test/<name>.txt
"""

import argparse
import csv
import os
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltr.judgments import Judgment, judgments_to_file
from ltr.features import FEATURES, format_value

FEATURE_SOURCE_COLUMNS = [spec.source for spec in FEATURES]


def read_graded_csv(path):
    """grade, qid, keywords, doc_id and the 13 feature columns together in one
    file, under their original headers (Grade, QID, Keywords, Image ID)."""
    required = {'Grade', 'QID', 'Keywords', 'Image ID'} | set(FEATURE_SOURCE_COLUMNS)
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError("%s missing required column(s): %s"
                              % (path, sorted(missing_cols)))
        for line_no, row in enumerate(reader, start=2):
            try:
                grade = int(float(row['Grade']))
                qid = int(row['QID'])
            except (TypeError, ValueError) as e:
                raise ValueError("%s:%d bad grade/qid: %s" % (path, line_no, e))
            doc_id = (row['Image ID'] or '').strip()
            if not doc_id:
                raise ValueError("%s:%d missing Image ID" % (path, line_no))
            rows.append({
                'qid': qid,
                'doc_id': doc_id,
                'grade': grade,
                'keywords': (row.get('Keywords') or '').strip(),
                'path': path,
                'raw': row,
            })
    return rows


def norm_keywords(keywords):
    return ' '.join(keywords.lower().split())


def check_qids_unique(rows):
    """Same QID with different keywords means two queries would merge into one group."""
    keywords_by_qid = {}
    errors = []
    for row in rows:
        kw = norm_keywords(row['keywords'])
        prev = keywords_by_qid.setdefault(row['qid'], (kw, row['path']))
        if prev[0] != kw:
            errors.append("QID %s is '%s' in %s but '%s' in %s"
                          % (row['qid'], prev[0], prev[1], kw, row['path']))
            keywords_by_qid[row['qid']] = (kw, row['path'])
    return errors


def check_disjoint(rows, other_paths):
    """No QID or keywords may appear in both this set and other_paths."""
    other_qids, other_keywords = {}, {}
    for path in other_paths:
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                other_qids.setdefault(int(row['QID']), path)
                other_keywords.setdefault(norm_keywords(row.get('Keywords') or ''), path)
    errors = set()
    for row in rows:
        kw = norm_keywords(row['keywords'])
        if row['qid'] in other_qids:
            errors.add("QID %s (%s) also in %s" % (row['qid'], row['path'], other_qids[row['qid']]))
        if kw in other_keywords:
            errors.add("query '%s' (%s) also in %s" % (kw, row['path'], other_keywords[kw]))
    return sorted(errors)


def dedupe(rows, report):
    """First row wins per (qid, doc_id) — a duplicate would silently double its weight."""
    seen = set()
    kept = []
    for row in rows:
        key = (row['qid'], row['doc_id'])
        if key in seen:
            report['duplicates'] += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept


def log_judgments(rows, report, drop_unlabelled=True):
    by_qid = OrderedDict()
    for row in rows:
        by_qid.setdefault(row['qid'], []).append(row)

    judgments = []
    for qid, qid_rows in by_qid.items():
        keywords = qid_rows[0]['keywords']
        grades = {r['grade'] for r in qid_rows}

        if drop_unlabelled and len(grades) < 2:
            # No pairwise signal to learn from; LambdaMART's lambdas are zero here.
            report['flat_queries'] += 1
            report['flat_query_ids'].append(qid)
            continue

        for row in qid_rows:
            values = [spec.normalise(row['raw'].get(spec.source)) for spec in FEATURES]
            judgments.append(Judgment(
                grade=row['grade'],
                qid=qid,
                keywords=keywords,
                docId=row['doc_id'],
                features=[format_value(spec, value) for spec, value in zip(FEATURES, values)],
            ))
        report['queries'] += 1

    return judgments


def validate(judgments, features):
    errors = []
    for j in judgments:
        for spec, value in zip(features, j.features):
            if value != value:  # NaN
                errors.append("qid %s doc %s: %s is NaN" % (j.qid, j.docId, spec.name))
            elif value < 0:
                errors.append("qid %s doc %s: %s is negative (%s)"
                              % (j.qid, j.docId, spec.name, value))
        if j.grade < 0:
            errors.append("qid %s doc %s: negative grade %s" % (j.qid, j.docId, j.grade))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--graded-csv', nargs='+', required=True,
                        help='one or more graded CSVs, combined into --out')
    parser.add_argument('--out', required=True,
                        help='RankLib output path (grades + feature vectors)')
    parser.add_argument('--disjoint-from', nargs='*', default=[],
                        help='CSVs of other sets that must share no QID or keywords with these')
    parser.add_argument('--keep-flat-queries', action='store_true',
                        help='keep queries whose docs all share one grade')
    args = parser.parse_args(argv)

    try:
        rows = [row for path in args.graded_csv for row in read_graded_csv(path)]
        errors = check_qids_unique(rows) + check_disjoint(rows, args.disjoint_from)
    except (OSError, ValueError, KeyError) as e:
        print("ERROR %s" % e, file=sys.stderr)
        return 1

    if errors:
        for err in errors[:20]:
            print("ERROR %s" % err, file=sys.stderr)
        print("%d error(s); nothing written." % len(errors), file=sys.stderr)
        return 1

    if not rows:
        print("ERROR %s produced zero rows" % ' '.join(args.graded_csv), file=sys.stderr)
        return 1

    report = Counter()
    report['flat_query_ids'] = []
    report['rows_in'] = len(rows)
    rows = dedupe(rows, report)

    judgments = log_judgments(rows, report, drop_unlabelled=not args.keep_flat_queries)

    # Feature scaling — disabled by default. LambdaMART is scale-invariant
    # (see ltr.features.standardize_features's docstring for why), so this
    # has no effect on today's model. Uncomment only alongside a matching
    # change at query time, or training-time and query-time feature scales
    # diverge silently.
    # from ltr.features import standardize_features
    # judgments = standardize_features(judgments, FEATURES)

    errors = validate(judgments, FEATURES)
    if errors:
        for err in errors[:20]:
            print("ERROR %s" % err, file=sys.stderr)
        print("%d validation error(s); nothing written." % len(errors), file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        judgments_to_file(f, judgments)

    grades = Counter(j.grade for j in judgments)
    print("input             : %s" % ' '.join(args.graded_csv), file=sys.stderr)
    print("rows read         : %d" % report['rows_in'], file=sys.stderr)
    print("duplicates dropped: %d" % report['duplicates'], file=sys.stderr)
    print("flat queries drop : %d %s" % (report['flat_queries'],
                                         report['flat_query_ids'] or ''), file=sys.stderr)
    print("queries logged    : %d" % report['queries'], file=sys.stderr)
    print("judgments written : %d (%d features each)" % (len(judgments), len(FEATURES)),
          file=sys.stderr)
    print("grade histogram   : %s" % dict(sorted(grades.items())), file=sys.stderr)
    print("wrote %s" % args.out, file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
