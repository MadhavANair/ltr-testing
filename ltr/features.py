"""Canonical feature list for the Phase 1 LTR model.

Single source of truth for feature names, order and normalisation. The
logger, the trainer, the evaluator and the OpenSearch exporter all import
FEATURES from here rather than declaring their own copy — a divergence in
order between training and query-time scoring is silent and fatal, since
RankLib features are addressed by position, not name.
"""
import hashlib


class FeatureSpec:
    """One logged feature: where it comes from and how it is normalised."""

    def __init__(self, name, source, transform=None, missing=0.0, kind='float'):
        self.name = name            # canonical name: fmap entry, sltr feature name
        self.source = source        # CSV column / raw field carrying the value
        self.transform = transform  # raw value -> logged value
        self.missing = missing      # substituted when nothing was logged
        self.kind = kind            # 'float' or 'binary', drives formatting

    def normalise(self, raw):
        if raw is None or raw == '':
            return float(self.missing)
        value = float(raw)
        if self.transform is not None:
            value = float(self.transform(value))
        return value


FEATURES = [
    FeatureSpec('caption_bm25',          'Caption BM25'),
    FeatureSpec('essential_words_bm25',  'Essential Words BM25'),
    FeatureSpec('main_words_bm25',       'Main Words BM25'),
    FeatureSpec('extra_words_bm25',      'Extra Words BM25'),
    FeatureSpec('description_bm25',      'Description BM25'),
    FeatureSpec('cross_fields_score',    'Cross-fields score'),
    # Logged as a rank in 1..100; scaled to (0, 1] so tree splits stay
    # comparable with the rest of the feature set.
    FeatureSpec('pseudonym_rank',        'Pseudonym Rank', transform=lambda v: v / 100.0),
    FeatureSpec('image_boost',           'Image Boost'),
    FeatureSpec('is_archive',            'Is Archive', kind='binary'),
    FeatureSpec('is_black_and_white',    'Is Black and White', kind='binary'),
    FeatureSpec('date_recency',          'Date Recency'),
    FeatureSpec('is_premium_collection', 'Is Premium Collection', kind='binary'),
    FeatureSpec('asset_type_photo',      'Asset Type: Photo', kind='binary'),
]


def feature_names(features=FEATURES):
    return [f.name for f in features]


def feature_list_hash(features=FEATURES):
    """Stable hash of feature names in order. Changes iff the contract
    changes — compare this between training time and export/query time
    to catch a silent feature-order divergence before it ships."""
    h = hashlib.sha256()
    for f in features:
        h.update(f.name.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()[:16]


def format_value(spec, value):
    """Binary features stay integral, scores keep 4dp."""
    if spec.kind == 'binary':
        return int(round(value))
    return round(value, 4)


def standardize_features(judgments, features=FEATURES):
    """Z-score standardize continuous features across all judgments, in place.
    Binary features are left untouched (scaling a 0/1 flag is meaningless).

    Disabled by default — see the commented-out call site in
    log_judgments_with_features.py. Two reasons it's off, not just unneeded:

    1. LambdaMART is a tree-based ranker: a split picks a threshold on one
       feature at a time, so a feature's raw scale doesn't change what the
       model can learn. Standardizing wouldn't change the shipped model.
    2. Mean/std here are computed from THIS logged set only. If ever enabled,
       query-time feature values (from OpenSearch's sltr query templates)
       would need the same mean/std applied before scoring, or training-time
       and query-time features diverge silently — the same class of bug
       `feature_list_hash()` guards against for feature *order*, but nothing
       guards against for feature *scale*.
    """
    continuous_idx = [i for i, spec in enumerate(features) if spec.kind != 'binary']
    for idx in continuous_idx:
        values = [j.features[idx] for j in judgments]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        if std == 0:
            continue
        for j in judgments:
            j.features[idx] = (j.features[idx] - mean) / std
    return judgments


def write_fmap(path, features=FEATURES):
    """Feature index -> name map. 0-based, matching this repo's existing
    fmap files — NOTE RankLib's own feature numbering in its training
    output and model files is 1-based, so a feature-impact report read
    through this fmap must add 1 before looking a name up, or it blames
    the wrong feature."""
    with open(path, 'w') as f:
        for idx, spec in enumerate(features):
            f.write("%d %s q\n" % (idx, spec.name))
