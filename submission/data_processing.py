"""Competition adapter for the same asset processors used by measurement_db.py.

ASSET FORMAT ADAPTER: change asset_references() when organizers specify how assets
are listed, and read_asset() when they specify how their bytes can be accessed.
Everything below these two functions can stay unchanged.
"""

from functools import lru_cache
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)


def asset_references(item: dict) -> list[dict]:
    """FORMAT ASSUMPTION: JSON path objects/lists, or Markdown file links.

    Preferred item_features value:
      {"assets": [{"path": "/data/input.csv", "media_type": "text/csv", "ordinal": 1}]}
    Also accepts asset_manifest, string paths in those lists, and nested path
    objects in JSON item_content. This is the place to change the input format.
    """
    references = []

    def visit(value, path_list=False):
        if isinstance(value, str) and path_list:
            references.append({'path': value, 'media_type': ''})
        elif isinstance(value, list):
            def order(pair):
                index, entry = pair
                ordinal = entry.get('ordinal', index) if isinstance(entry, dict) else index
                return ordinal if isinstance(ordinal, (int, float)) else index
            for _, entry in sorted(enumerate(value), key=order):
                visit(entry, path_list)
        elif isinstance(value, dict):
            path = value.get('path', value.get('file_path'))
            if isinstance(path, str) and path:
                references.append({'path': path, 'media_type': value.get('media_type', value.get('mime_type', ''))})
            for key, entry in value.items():
                if key in {'assets', 'asset_manifest'} and isinstance(entry, str):
                    try:
                        entry = json.loads(entry)
                    except ValueError:
                        continue
                if isinstance(entry, (dict, list)):
                    visit(entry, key in {'assets', 'asset_manifest'})

    for field in ('item_features', 'item_content'):
        try:
            visit(json.loads(item.get(field) or ''))
        except (ValueError, TypeError):
            pass
    for match in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)', item.get('item_content', '')):
        references.append({'path': match[1].strip().strip('<>'), 'media_type': ''})
    unique = {}
    for reference in references:
        unique.setdefault((reference['path'], str(reference['media_type'])), reference)
    return list(unique.values())


def read_asset(reference: dict, asset_root: Path) -> tuple[str, bytes, str]:
    """TRANSPORT ASSUMPTION: accessible local paths (including file:// URLs).

    Relative paths are resolved under DATA_PROCESSING_ASSET_ROOT (default cwd).
    Replace this function if assets later arrive as URLs, base64, or handles.
    Return (filename, raw bytes, MIME type); the processors need no other changes.
    """
    from assets import SUPPORTED_MEDIA_TYPES
    from assets.application import EXTENSION_TYPES

    name = reference['path']
    if name.startswith('file:'):
        parsed = urlsplit(name)
        if parsed.netloc not in {'', 'localhost'}:
            raise ValueError('Only local file:// paths are supported')
        name = unquote(parsed.path)
    elif '://' in name or name.startswith('data:'):
        raise ValueError('This adapter currently supports local asset paths only')
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = asset_root / path
    media_type = (reference.get('media_type') or EXTENSION_TYPES.get(path.suffix.lower())
                  or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
    media_type = media_type.split(';', 1)[0].strip().lower()
    aliases = {'audio/x-wav': 'audio/wav', 'application/x-sqlite3': 'application/vnd.sqlite3',
               'application/ndjson': 'application/x-ndjson'}
    media_type = aliases.get(media_type, media_type)
    if media_type not in SUPPORTED_MEDIA_TYPES:
        media_type = 'application/octet-stream'  # existing verified content detection
    return path.name, path.read_bytes(), media_type


class DataProcessor:
    def __init__(self, asset_root=None, cache_dir=None):
        self.asset_root = Path(asset_root or os.getenv('DATA_PROCESSING_ASSET_ROOT', Path.cwd()))
        self.cache_dir = Path(cache_dir or os.getenv(
            'DATA_PROCESSING_CACHE', Path(__file__).with_name('.data_processing_cache')))
        self._context = None
        self._budget = None
        # Items and their files are assumed immutable during evaluation.
        self._cached_question = lru_cache(maxsize=256)(self._question)

    def _process_asset(self, reference):
        from assets import AssetProcessingContext, process_asset

        filename, data, media_type = read_asset(reference, self.asset_root)
        if self._context is None:
            self._context = AssetProcessingContext(
                self.cache_dir / 'descriptions.sqlite3', artifact_root=self.cache_dir / 'assets',
                local_files_only=True,
            )
        result = process_asset(hashlib.sha256(data).hexdigest(), media_type, data,
                               self._context, filename=filename)
        if result.relative_path is not None:
            raise ValueError('Asset could only be preserved as a binary artifact; omitting its description')
        return result.question_fragment, media_type.split('/', 1)[0] in {'text', 'application'}

    def _question(self, item_json):
        item = json.loads(item_json)
        original = item.get('item_content', '')
        fragments = []
        for reference in asset_references(item):
            try:
                fragments.append(self._process_asset(reference))
            except Exception as error:
                logger.warning('Skipping asset %s: %s: %s', reference.get('path'), type(error).__name__, error)
        if not fragments:
            return original
        if any(needs_budget for _, needs_budget in fragments):
            try:
                from assets.question_budget import QuestionBudget

                if self._budget is None:
                    self._budget = QuestionBudget(local_files_only=True)
                return self._budget.assemble(original, [text for text, _ in fragments])[0]
            except Exception as error:
                logger.warning('Skipping text/application assets: question budget failed: %s', error)
                fragments = [(text, needed) for text, needed in fragments if not needed]
        return original + ('\n\n' + '\n\n'.join(text for text, _ in fragments) if fragments else '')

    def process_item(self, item: dict) -> dict:
        processed = dict(item)
        try:
            processed['item_content'] = self._cached_question(json.dumps(item, sort_keys=True))
        except Exception as error:
            logger.warning('Item preprocessing failed; keeping original question: %s', error)
        return processed

    def preprocess(self, input: list, labeled=None):
        def pair(value):
            subject, item = value
            return [dict(subject), self.process_item(item)]
        return pair(input), (None if labeled is None else [[pair(value), label] for value, label in labeled])


@lru_cache(maxsize=1)
def _processor():
    return DataProcessor()


def preprocess(input: list, labeled=None):
    """Return processed copies of the target [subject, item] and revealed examples."""
    return _processor().preprocess(input, labeled)
