"""The model attributes of a subject, found by its normalized_name.

1. The database: the model table of the assessor data build (model_info/models.parquet), i.e. every model the
   assessor trains on, with the values epoch_models_csv.py found for it. Looked up by normalized_name, then by its
   key (lowercase letters and digits) when that key names one model only.
2. A model not in the database: epoch_models_csv.model_table on that model (model_info/epoch_models_csv.py, a copy
   of the code that built the database), offline: Epoch AI, models.dev and ReleasedModels from the local copies in
   model_info/data, the Hugging Face Hub only through its cache. The competition allows no web access at prediction
   time.
The subject gets the training rows' model_* columns (assessor_data.MODEL_COLUMNS, with their types), plus model_key
(the database name, else the normalized_name) and model_info_source ("database", "catalogues" or "none"). Its own
fields are kept as given, as in training.

model_info/ comes from the repository: models.parquet from data/datasets/assessor_data/<build>/, data/ from data/
(ai_models.zip, renamed ai_models_zip.bin: the platform refuses a zip inside the submission; models_dev.json, releasedmodels.json, hf_hub.json). Copy them again after a new build.
"""

import contextlib
from functools import lru_cache
import io
import logging
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)
MODEL_INFO = Path(__file__).with_name('model_info')
REPO_ID = re.compile(r'^[\w.-]+/[\w.-]+$')      # a Hugging Face repo id: organisation/name (as assessor_data.py)


def key(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def missing(v):
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v) or v == ''


def convert(v, kind):
    """A model table value as the training rows' type (assessor_data.to_table); None when missing."""
    if missing(v):
        return None
    if kind == 'bool':
        return v if isinstance(v, bool) else {'true': True, 'false': False}[str(v).lower()]
    if kind == 'float':
        return float(v)
    if kind == 'int':
        return int(float(v))
    return v if isinstance(v, str) else str(v)


# model table column -> training row column (assessor_data.MODEL_COLUMNS), and the non-text types
# (assessor_data.MODEL_FLOAT, MODEL_INT, MODEL_BOOL)
MODEL_COLUMNS = {'kind': 'model_kind', 'provider': 'model_provider', 'release_date': 'model_release_date',
                 'open_weights': 'model_open_weights', 'publication_date': 'model_publication_date',
                 'organization': 'model_organization', 'country': 'model_country',
                 'model_accessibility': 'model_accessibility', 'epoch_domain': 'model_domain',
                 'base_model': 'model_base_model', 'parameters': 'model_parameters',
                 'active_parameters': 'model_active_parameters', 'training_compute_flop': 'model_training_compute_flop',
                 'training_dataset_size': 'model_training_dataset_size',
                 'finetune_compute_flop': 'model_finetune_compute_flop',
                 'post_training_compute_flop': 'model_post_training_compute_flop',
                 'training_hardware': 'model_training_hardware', 'frontier_model': 'model_frontier',
                 'architecture': 'model_architecture', 'moe': 'model_moe', 'n_experts': 'model_n_experts',
                 'multimodal_input': 'model_multimodal_input', 'context_window': 'model_context_window',
                 'knowledge_cutoff': 'model_knowledge_cutoff', 'thinking_mode': 'model_thinking_mode'}
TYPES = {**dict.fromkeys(['model_parameters', 'model_active_parameters', 'model_training_compute_flop',
                          'model_training_dataset_size', 'model_finetune_compute_flop',
                          'model_post_training_compute_flop'], 'float'),
         **dict.fromkeys(['model_n_experts', 'model_context_window'], 'int'),
         **dict.fromkeys(['model_open_weights', 'model_moe', 'model_multimodal_input', 'model_thinking_mode'], 'bool')}


def columns():
    """model table column -> (row column, type)."""
    return {c: (row, TYPES.get(row, 'str')) for c, row in MODEL_COLUMNS.items()}


@lru_cache(maxsize=1)
def database():
    """The model table by normalized_name, and key -> normalized_name for keys naming one model."""
    db = pd.read_parquet(MODEL_INFO / 'models.parquet').set_index('normalized_name')
    names = pd.Series(db.index, index=db.index.map(key))
    return db, {k: v.iloc[0] for k, v in names.groupby(level=0) if len(v) == 1}


@lru_cache(maxsize=1)
def epoch_models():
    """epoch_models_csv, reading the bundled catalogues."""
    sys.path.insert(0, str(MODEL_INFO))
    import epoch_models_csv as E
    data = MODEL_INFO / 'data'
    E.ZIP, E.MODELS_DEV_CACHE = data / 'ai_models_zip.bin', data / 'models_dev.json'
    E.RELEASED_CACHE, E.HF_CACHE = data / 'releasedmodels.json', data / 'hf_hub.json'
    # clean() writes RAW and prints it relative to ROOT; the submission folder may be read-only
    E.ROOT = Path(tempfile.gettempdir())
    E.RAW = E.ROOT / 'models_raw.csv'
    return E


def catalogue_row(name, provider, release_date):
    """model_table's row for one model, built as assessor_data.models_of builds the database's input."""
    E = epoch_models()
    if not provider and REPO_ID.match(name):
        provider = name.split('/')[0]
    row = {'normalized_name': name, 'provider': provider or None, 'release_date': release_date or None,
           'benchmarks': '', 'display_names': '', 'n_subjects': 1, 'runs_thinking': None,
           'kind': 'agent system' if name.startswith('Agent: ') else E.NOT_A_MODEL.get(name, 'model')}
    # model_table strips a provider prefix from names and fills an empty provider from the providers in its input,
    # which for the database were all of them: they come along as rows that are not models (never matched).
    providers = sorted(set(database()[0]['provider'].dropna()) - {provider})
    others = pd.DataFrame({'normalized_name': [f'(provider) {p}' for p in providers], 'provider': providers,
                           'release_date': None, 'benchmarks': '', 'display_names': '', 'n_subjects': 0,
                           'runs_thinking': None, 'kind': 'provider'})
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        table = E.model_table(pd.concat([pd.DataFrame([row]), others], ignore_index=True), offline=True)
    logger.debug('model_table for %s:\n%s', name, log.getvalue())
    return table.iloc[0]


@lru_cache(maxsize=None)
def model_info(name, provider='', release_date=''):
    out = {'model_key': name or None, 'model_info_source': 'none', **{row: None for row, _ in columns().values()}}
    if not name:
        return out
    db, keys = database()
    hit = name if name in db.index else keys.get(key(name))
    try:
        if hit is not None:
            values, out['model_key'], out['model_info_source'] = db.loc[hit], hit, 'database'
        else:
            values, out['model_info_source'] = catalogue_row(name, provider, release_date), 'catalogues'
    except Exception as error:
        logger.warning('No model attributes for %s: %s: %s', name, type(error).__name__, error)
        return out
    out.update({row: convert(values[c], kind) for c, (row, kind) in columns().items()})
    return out


def process_subject(subject: dict) -> dict:
    name = (subject.get('normalized_name') or '').strip()
    return {**subject, **model_info(name, subject.get('provider') or '', subject.get('release_date') or '')}


def process(input: list, labeled=None):
    """Copies of the target [subject, item] and the revealed examples, every subject with its model attributes."""
    def pair(value):
        subject, item = value
        return [process_subject(subject), item]
    return pair(input), (None if labeled is None else [[pair(value), label] for value, label in labeled])
