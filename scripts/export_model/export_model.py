#!/usr/bin/env python3
"""Builds the _createmodel request body for a trained model, ready to POST as-is.

The featureset below is not emitted; it is the reference this checks the
model's feature order against.

Usage:
    python3 scripts/export_model/export_model.py --model Models/lambdamart_<ts>.txt
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from ltr.features import FEATURES

FEATURESET = {
    "featureset": {
        "name": "alamy_phase1",
        "features": [
            {
                "name": "caption_bm25",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"caption": "{{keywords}}"}},
            },
            {
                "name": "essential_words_bm25",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"essential_words": "{{keywords}}"}},
            },
            {
                "name": "main_words_bm25",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"main_words": "{{keywords}}"}},
            },
            {
                "name": "extra_words_bm25",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"extra_words": "{{keywords}}"}},
            },
            {
                "name": "description_bm25",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"description": "{{keywords}}"}},
            },
            {
                "name": "cross_fields_score",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "multi_match": {
                        "query": "{{keywords}}",
                        "fields": [
                            "caption",
                            "essential_words",
                            "main_words",
                            "extra_words",
                            "description",
                        ],
                        "type": "cross_fields",
                    }
                },
            },
            {
                "name": "pseudonym_rank",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"pseudonym": "{{keywords}}"}},
            },
            {
                "name": "image_boost",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"image_boost": "{{keywords}}"}},
            },
            {
                "name": "is_archive",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"is_archive": "{{keywords}}"}},
            },
            {
                "name": "is_black_and_white",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"is_black_and_white": "{{keywords}}"}},
            },
            {
                "name": "date_recency",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"date_recency": "{{keywords}}"}},
            },
            {
                "name": "is_premium_collection",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"is_premium_collection": "{{keywords}}"}},
            },
            {
                "name": "asset_type_photo",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {"match": {"asset_type": "{{keywords}}"}},
            },
        ],
    }
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='trained RankLib model .txt')
    parser.add_argument('--model-name', help='defaults to the model filename without .txt')
    parser.add_argument('--out-dir', default='export', help='default: %(default)s')
    parser.add_argument('--out', help='full output path, overriding --out-dir')
    args = parser.parse_args(argv)

    try:
        with open(args.model) as f:
            definition = f.read()
    except OSError as e:
        print("ERROR --model not readable: %s" % e, file=sys.stderr)
        return 1

    # RankLib scores by feature POSITION, so a featureset in a different order
    # than training silently scores garbage.
    names = [f['name'] for f in FEATURESET['featureset']['features']]
    expected = [f.name for f in FEATURES]
    if names != expected:
        print("ERROR featureset order does not match ltr/features.py:\n  featureset: %s\n  "
              "features.py: %s" % (names, expected), file=sys.stderr)
        return 1

    stem = os.path.splitext(os.path.basename(args.model))[0]
    model_name = args.model_name or stem
    featureset_name = FEATURESET['featureset']['name']
    out_path = args.out or os.path.join(args.out_dir, stem + '_model_body.json')

    payload = {
        'model': {
            'name': model_name,
            'model': {'type': 'model/ranklib', 'definition': definition},
        }
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print("model       : %s" % model_name, file=sys.stderr)
    print("featureset  : %s (%d features, order checked)" % (featureset_name, len(names)),
          file=sys.stderr)
    print("wrote       : %s" % out_path, file=sys.stderr)
    print("\nupload with:", file=sys.stderr)
    print("  curl -X POST \"http://localhost:9201/_ltr/_featureset/%s/_createmodel\" \\\n"
          "    -H 'Content-Type: application/json' --data @%s"
          % (featureset_name, out_path), file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
