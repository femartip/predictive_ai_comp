"""The general assessor's input row of a [subject, item] pair, after data_processing, subject_processing and
item_annotation: the features it was trained on (scripts/assessors_router, prior R1+B0+G1+I2 plus M1), in its column
order, and its start.

    from assessor_input import build
    row, labeled_rows = build(processed_input, processed_labeled)
    # row = {"start": float, "features": {column: value}, "keys": T2's levels}; labeled_rows: one per labeled pair
    # (same order), or None

Columns (assessor/meta.json lists them in order):
    I2   <code>_exp           the item's 18 ADeLe v1 levels (item_annotation)
    B0   bench_<code>_exp     its benchmark's mean levels, from the call's own evidence only (the platform forbids
                              relying on earlier calls): for the target row, the mean over its item and the distinct
                              labeled items of the same benchmark_id; for a labeled row, the mean over those labeled
                              items. At budget 0 (no labels) that is the item's own levels. Levels without annotation
                              are skipped, as in training (components.bench_profiles).
    G1   group_rel            how the model's group (provider x size tier x reasoning) beats or trails the training
                              benchmarks' base rates (assessor/groups.parquet); missing for a group never seen
    M1   strength, mean_score the model's values on the training data if known (assessor/models.parquet, by model_key),
                              else predicted from its fields (assessor/router_<measure>.txt, by lgb_numpy); model_known;
                              completeness (how many of provider, parameters and release date are known, 0-3)
    start                     R1: the logit of the mean base rate of the training benchmarks
    keys                      T2's levels: benchmark_id; the subject (all its competition fields) on that benchmark;
                              the item (its content and benchmark_id)
The subject features (S1) are computed as in training (assessors_v3/components.system_table, on the fields predict()
gets: model_name and thinking are not used there either), from the subject's own fields and, where empty, the model
attributes subject_processing filled in (model_provider, model_release_date, model_parameters), as bank.py's model
table does. assessor/ is written by submission/fit_tables.py.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lgb_numpy import Model

ASSESSOR = Path(__file__).with_name('assessor')
CODES = ['AS', 'AT', 'CEc', 'CEe', 'CL', 'KNa', 'KNc', 'KNf', 'KNn', 'KNs', 'MCr', 'MCt', 'MCu', 'MS', 'QLl', 'QLq',
         'SNs', 'VO']
LEVELS = [f'{c}_exp' for c in CODES]
MEASURES = ('strength', 'mean_score')

# components.system_table's constants (assessors_v3/components.py)
ORIGIN = pd.Timestamp('2023-01-01')
EFFORT = {'none': 0, 'minimal': 0.5, 'low': 1, 'medium': 2, 'high': 3, 'xhigh': 4, 'max': 5}
REASONING = re.compile(r'think|reason|\br1\b|-r1|\bo[134]\b|o[134]-|qwq|deepseek-r|magistral', re.I)
SCAFFOLD = re.compile(r'agent|scaffold|hands|aider|codex|claude.code|cursor|swe-|mini-swe', re.I)
SMALL = re.compile(r'\b(?:mini|flash|haiku|nano|small|lite|tiny|micro)\b', re.I)
LARGE = re.compile(r'\b(?:pro|opus|ultra|large|max|plus|turbo)\b', re.I)
PARAMS = re.compile(r'(\d+(?:\.\d+)?)\s*[bB]\b')


def text(v):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def years(value):
    t = pd.to_datetime(pd.Series([value or None], dtype=object), errors='coerce', format='mixed',
                       utc=True).dt.tz_localize(None)
    return float(((t - ORIGIN).dt.days / 365.25).iloc[0])


def system_row(subject):
    """S1 of one subject (components.system_table), from the fields predict() gets plus the model attributes."""
    name = text(subject.get('model_key') or subject.get('normalized_name')).strip()
    effort = EFFORT.get(text(subject.get('reasoning_effort')).lower(), 0.0)
    harness = text(subject.get('harness'))
    provider = text(subject.get('provider')) or text(subject.get('model_provider')) or '(unknown)'
    release = text(subject.get('release_date')) or text(subject.get('model_release_date'))
    params = subject.get('model_parameters')
    params = float('nan') if params is None else float(params)
    with np.errstate(divide='ignore', invalid='ignore'):     # as numpy's log10 in system_table: 0 -> -inf
        log_params = float(np.log10(params))
        if math.isnan(log_params):
            m = PARAMS.search(name)
            log_params = float(np.log10(float(m[1]) * 1e9)) if m else float('nan')
    return {'provider': provider.lower(), 'release_years': years(release),
            'access_years': years(text(subject.get('access_date'))), 'effort': float(effort),
            'reasoning': float(bool(REASONING.search(name)) or effort > 0),
            'has_harness': float(harness != ''), 'scaffold': float(bool(SCAFFOLD.search(harness))),
            'log_params': log_params, 'small_tier': float(bool(SMALL.search(name))),
            'large_tier': float(bool(LARGE.search(name))), 'name_missing': float(name == '')}


@lru_cache(maxsize=1)
def tables():
    meta = json.loads((ASSESSOR / 'meta.json').read_text(encoding='utf-8'))
    models = pd.read_parquet(ASSESSOR / 'models.parquet').set_index('model_key')
    groups = pd.read_parquet(ASSESSOR / 'groups.parquet').set_index('group')['group_rel']
    routers = {m: Model(ASSESSOR / f'router_{m}.txt') for m in MEASURES}
    return meta, models, groups, routers


def model_columns(subject):
    """G1 and M1 of a subject."""
    meta, models, groups, routers = tables()
    s1 = system_row(subject)
    tier = 'small' if s1['small_tier'] > 0 else 'large' if s1['large_tier'] > 0 else 'mid'
    group_rel = groups.get(f"{s1['provider']}|{tier}|{int(s1['reasoning'])}", float('nan'))
    key = subject.get('model_key')
    known = key is not None and key in models.index
    if known:
        values = {m: float(models.at[key, m]) for m in MEASURES}
    else:
        values = {m: float(routers[m].raw_score([s1])[0]) for m in MEASURES}
    completeness = ((s1['provider'] != '(unknown)') + math.isfinite(s1['log_params'])
                    + (not math.isnan(s1['release_years'])))
    return {'group_rel': float(group_rel), **values, 'model_known': float(known), 'completeness': float(completeness)}


def item_key(item):
    fields = {k: item.get(k) for k in ('benchmark_id', 'item_content', 'item_features', 'interactors')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


SUBJECT_FIELDS = ('normalized_name', 'provider', 'release_date', 'access_date', 'harness', 'harness_version',
                  'reasoning_effort', 'subject_features_extra')


def subject_key(subject):
    """A subject's identity: all its competition fields (the platform gives no subject id)."""
    fields = {k: text(subject.get(k)) for k in SUBJECT_FIELDS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def levels_of(item):
    return [None if item.get(c) is None else float(item[c]) for c in LEVELS]


def mean_levels(items):
    """B0 columns: the mean of each level over the given items' levels, skipping missing ones."""
    V = np.array([[np.nan if v is None else v for v in lv] for lv in items], float).reshape(-1, len(LEVELS))
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(V, 0) if np.isfinite(V).any() else np.full(len(LEVELS), np.nan)
    return {f'bench_{c}': float(v) for c, v in zip(LEVELS, mean)}


def row(value, evidence):
    """The input row of a [subject, item] pair; evidence: the call's distinct labeled items of each benchmark_id
    (item key -> levels), the pair's own item added for its B0 when it is the target."""
    subject, item = value
    bench, key = text(item.get('benchmark_id')), item_key(item)
    levels = levels_of(item)
    i2 = {c: (float('nan') if v is None else v) for c, v in zip(LEVELS, levels)}
    features = {**i2, **mean_levels(evidence.get(bench, {}).values()), **model_columns(subject)}
    meta = tables()[0]
    assert list(features) == meta['columns'], 'feature columns differ from the trained ones'
    # T2's levels (testtime.keys): the benchmark, the subject on that benchmark, the item
    keys = {'bench': bench, 'subject': f'{bench}|{subject_key(subject)}', 'item': key}
    return {'start': meta['r1'], 'features': features, 'keys': keys}


def build(input: list, labeled=None):
    """Input rows of the target pair and of the labeled pairs, from this call's input and labeled pairs only."""
    labeled_items = {}
    for (_, item), _label in labeled or []:
        labeled_items.setdefault(text(item.get('benchmark_id')), {})[item_key(item)] = levels_of(item)
    target_item = input[1]
    bench = text(target_item.get('benchmark_id'))
    own = {item_key(target_item): levels_of(target_item)}
    with_target = {**labeled_items, bench: {**labeled_items.get(bench, {}), **own}}
    return row(input, with_target), (None if labeled is None else [row(value, labeled_items) for value, _ in labeled])
