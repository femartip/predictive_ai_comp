"""ADeLe v1 demand levels of an item at prediction time: the data bank's annotator, run inside the submission.

    from item_annotation import annotate
    processed_input, processed_labeled = annotate(*process(*preprocess(input, labeled)))

The item gets the 18 levels its benchmark's items got in the data bank (annotate_hf.py), by the same means:
- model: Qwen/Qwen2.5-7B-Instruct with the LoRA of ceferra/qwen2.5-7b-adele-annotator merged into its weights
  (W += lora_alpha / r * B @ A, what peft's merge_and_unload does in merge_adapter.py), bf16; the tokenizer of the
  adapter repository. Both repositories are in models.txt (the hub is offline at prediction time);
- prompts: adele_prompt.py and rubrics/adele_v1/, the bank's own files;
- score: one forward pass per (item, rubric), no generation; the probabilities of the tokens "0"-"5" at the next
  position, renormalised; <code>_exp = sum of k * p_k;
- text: item_content after data_processing.preprocess, i.e. with the asset descriptions and inlined files appended:
  the converter's `question` field, which is what the bank annotated;
- window: 32,767 tokens, the bank's, on a GPU of 40 GiB or more. On 20-40 GiB, 16,384 (as on the cluster's A30s): a
  longer prompt keeps its start and end, and the item is marked truncated_any. Below 20 GiB, or without a GPU, there is
  no annotation (a 7B model on CPU takes minutes per item) unless ADELE_DEVICE=cpu.

Each item gets the bank table's columns: <code>_exp for the 18 rubrics, n_tokens_max, truncated_any, digit_mass_min;
all None when the item has no annotation (empty text, no annotator, or a failure, which is logged and not cached).

Cache: an SQLite file in the submission folder (ADELE_CACHE), shared by the prediction workers and kept across
checkpoints (workers restart at each one), keyed by the annotator's revisions, the window and the text's SHA-256.

GPU: in a prediction worker (PAIEC_PREDICTION_WORKER_ID set) the annotator loads at import, so its loading (about a
minute) is not counted in a predict() call's timeout; elsewhere (the parent process, local checks) at first use. It
takes about 15 GB next to the one asset model data_processing may hold (at most about 17 GB). If the GPU runs out of
memory while loading or scoring, the asset model is released (it reloads at its next asset) and the step retried once.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys

import adele_prompt as P

logger = logging.getLogger(__name__)

BASE = 'Qwen/Qwen2.5-7B-Instruct'
ADAPTER = 'ceferra/qwen2.5-7b-adele-annotator'
RUBRICS = P.load_rubrics(P.RUBRICS_V1)
CODES = list(RUBRICS)
COLUMNS = [f'{c}_exp' for c in CODES] + ['n_tokens_max', 'truncated_any', 'digit_mass_min']
# window by GPU size (GiB): the bank's 32,767 needs a ~48 GB card; 16,384 fits a 24 GB one (annotate_hf.py)
WINDOWS = ((40, 32767, 65536), (20, 16384, 16384))       # (minimum GiB, window, padded tokens per forward pass)
CACHE = Path(os.getenv('ADELE_CACHE', Path(__file__).with_name('.adele_cache') / 'adele_v1.sqlite3'))
NO_LEVELS = dict.fromkeys(COLUMNS)


def is_oom(error):
    message = str(error)
    return type(error).__name__ == 'OutOfMemoryError' or (
        isinstance(error, RuntimeError) and ('out of memory' in message or 'CUDA driver error' in message))


def free_asset_models():
    """Drop the multimodal model data_processing holds (if any) to make room on the GPU; it reloads when needed."""
    dp = sys.modules.get('data_processing')
    context = getattr(dp._processor(), '_context', None) if dp is not None else None
    if context is not None and context._models:
        # private, but the context's own way to unload: keeping 'text' (never a model) releases every model family
        context._release_other_family('text')
        logger.warning('ADeLe annotator: GPU out of memory, released the asset description models')
        return True
    return False


def batches(rows, max_tokens, max_batch=64):
    """Length-sorted batches (longest first) under a padded-token budget, as annotate_hf.py; lists of row indices."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]['ids']))
    batch, longest = [], 0
    for i in order:
        n = len(rows[i]['ids'])
        if batch and (max(longest, n) * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


class Annotator:
    def __init__(self, device, max_len, max_batch_tokens):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.max_len, self.max_batch_tokens = torch, device, max_len, max_batch_tokens
        # the pre-downloaded snapshot folders, loaded as local paths: offline, a repository id makes AutoTokenizer look
        # for the adapter repository's config.json, which it does not have, and fail
        base = Path(snapshot_download(BASE, local_files_only=True))
        adapter = Path(snapshot_download(ADAPTER, local_files_only=True))
        self.tok = AutoTokenizer.from_pretrained(str(adapter))
        # torch_dtype (older transformers ignore dtype=), as annotate_hf.py; straight onto the GPU, as merge_adapter.py
        self.model = AutoModelForCausalLM.from_pretrained(
            str(base), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            **({'device_map': device} if device != 'cpu' else {})).eval()
        self.merge_lora(json.loads((adapter / 'adapter_config.json').read_text()),
                        load_file(adapter / 'adapter_model.safetensors'))
        # the snapshot folders are named by commit: the cache key changes if either repository does
        self.revision = f'{adapter.name}+{base.name}'
        self.skeletons = P.skeletons(self.tok, RUBRICS)
        self.digit_ids = [self.tok.convert_tokens_to_ids(str(k)) for k in range(6)]
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def merge_lora(self, config, weights):
        """W += lora_alpha / r * B @ A on every adapted linear layer (delta in float32, added to the bf16 weight)."""
        if config.get('peft_type') != 'LORA' or config.get('use_dora') or config.get('use_rslora') \
                or config.get('modules_to_save') or config.get('rank_pattern') or config.get('alpha_pattern'):
            raise ValueError('the adapter is not a plain LoRA; merge it with peft instead')
        scale = config['lora_alpha'] / config['r']
        modules = dict(self.model.named_modules())
        prefix, suffix = 'base_model.model.', '.lora_A.weight'
        merged = 0
        for key, a in weights.items():
            if not key.endswith(suffix):
                continue
            weight = modules[key[len(prefix):-len(suffix)]].weight
            b = weights[key[:-len(suffix)] + '.lora_B.weight']
            weight.data += (b.to(weight.device, self.torch.float32) @ a.to(weight.device, self.torch.float32)) * scale
            merged += 1
        if merged == 0 or 2 * merged != len(weights):
            raise ValueError(f'merged {merged} LoRA pairs out of {len(weights)} adapter tensors')

    def score(self, seqs):
        """(probabilities of the six level tokens renormalised, their raw total) at the next position, left padding."""
        torch = self.torch
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.tensor(s)
            mask[i, width - len(s):] = 1
        with torch.inference_mode():
            # no mask for a single sequence, so SDPA can use its memory-efficient causal kernel
            hidden = self.model.model(input_ids=ids.to(self.device),
                                      attention_mask=None if len(seqs) == 1 else mask.to(self.device),
                                      use_cache=False).last_hidden_state
            probs = torch.softmax(self.model.lm_head(hidden[:, -1, :]).float(), dim=-1)[:, self.digit_ids]
        mass = probs.sum(dim=-1, keepdim=True)
        return list(zip((probs / mass).cpu().tolist(), mass.squeeze(-1).cpu().tolist()))

    def score_safe(self, seqs, retried=False):
        try:
            return self.score(seqs)
        except RuntimeError as error:
            if not is_oom(error):
                raise
            if self.device != 'cpu':
                self.torch.cuda.empty_cache()
            if len(seqs) > 1:
                mid = len(seqs) // 2
                return self.score_safe(seqs[:mid]) + self.score_safe(seqs[mid:])
            if not retried and free_asset_models():
                return self.score_safe(seqs, retried=True)
            raise

    def levels(self, text):
        rows = P.prompt_ids(self.tok, [{'text_id': 0, 'content': text}], self.skeletons, self.max_len)
        out = [None] * len(rows)
        for idx in batches(rows, self.max_batch_tokens):
            for i, result in zip(idx, self.score_safe([rows[i]['ids'] for i in idx])):
                out[i] = result
        levels = {f"{r['rubric']}_exp": sum(k * p for k, p in enumerate(probs)) for r, (probs, _) in zip(rows, out)}
        return {**levels, 'n_tokens_max': max(len(r['ids']) for r in rows),
                'truncated_any': any(r['truncated'] for r in rows), 'digit_mass_min': min(m for _, m in out)}


def device_and_window():
    """(device, window, padded tokens per pass), or None when this machine cannot run the annotator in useful time."""
    import torch
    if os.getenv('ADELE_DEVICE') == 'cpu':
        return 'cpu', WINDOWS[0][1], WINDOWS[0][2]
    if not torch.cuda.is_available():
        logger.warning('ADeLe annotator: no GPU, items get no levels (ADELE_DEVICE=cpu forces the CPU)')
        return None
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    for minimum, window, batch_tokens in WINDOWS:
        if gib >= minimum:
            return 'cuda', window, batch_tokens
    logger.warning('ADeLe annotator: the GPU has %.1f GiB, at least %d needed; items get no levels', gib, WINDOWS[-1][0])
    return None


_annotator = None
_loaded = False


def annotator():
    """The annotator, loaded once per process; None if it cannot be (the reason is logged once)."""
    global _annotator, _loaded
    if not _loaded:
        _loaded = True
        try:
            setting = device_and_window()
            if setting is not None:
                try:
                    _annotator = Annotator(*setting)
                except Exception as error:
                    if not (is_oom(error) and free_asset_models()):
                        raise
                    import torch
                    torch.cuda.empty_cache()
                    _annotator = Annotator(*setting)
                logger.info('ADeLe annotator loaded on %s, window %d tokens, revision %s',
                            setting[0], setting[1], _annotator.revision)
        except Exception as error:
            logger.warning('ADeLe annotator could not be loaded; items get no levels: %s: %s',
                           type(error).__name__, error)
            _annotator = None
    return _annotator


class Cache:
    """Levels by key in an SQLite table; in memory only if the file cannot be opened."""

    def __init__(self, path):
        self.memory, self.db = {}, None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=60)
            self.db.execute('CREATE TABLE IF NOT EXISTS levels (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.commit()
        except Exception as error:
            logger.warning('ADeLe cache %s unavailable, keeping levels in memory: %s', path, error)
            self.db = None

    def get(self, key):
        if key not in self.memory and self.db is not None:
            row = self.db.execute('SELECT value FROM levels WHERE key = ?', (key,)).fetchone()
            if row is not None:
                self.memory[key] = json.loads(row[0])
        return self.memory.get(key)

    def put(self, key, value):
        self.memory[key] = value
        if self.db is not None:
            try:
                with self.db:
                    self.db.execute('INSERT OR REPLACE INTO levels VALUES (?, ?)', (key, json.dumps(value)))
            except sqlite3.Error as error:
                logger.warning('ADeLe cache write failed: %s', error)


_cache = None


def item_levels(text):
    """The item's columns (COLUMNS), all None when it has no annotation."""
    global _cache
    if not text:
        return dict(NO_LEVELS)
    model = annotator()
    if model is None:
        return dict(NO_LEVELS)
    if _cache is None:
        _cache = Cache(CACHE)
    key = f"{model.revision}:{model.max_len}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    levels = _cache.get(key)
    if levels is None:
        try:
            levels = model.levels(text)
        except Exception as error:
            logger.warning('ADeLe annotation failed; item gets no levels: %s: %s', type(error).__name__, error)
            return dict(NO_LEVELS)
        _cache.put(key, levels)
    return {**NO_LEVELS, **levels}


def annotate_item(item: dict) -> dict:
    return {**item, **item_levels(item.get('item_content') or '')}


def annotate(input: list, labeled=None):
    """Copies of the target [subject, item] and the revealed examples, every item with its ADeLe levels."""
    def pair(value):
        subject, item = value
        return [subject, annotate_item(item)]
    return pair(input), (None if labeled is None else [[pair(value), label] for value, label in labeled])


# a prediction worker loads the annotator before its first call (no per-call timeout during import)
if os.getenv('PAIEC_PREDICTION_WORKER_ID') is not None:
    annotator()
