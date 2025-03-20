"""Read original JSONL conversation records without modifying the sources."""
from pathlib import Path
import json
import logging
from typing import Iterator
LOG = logging.getLogger(__name__)
SPLITS = ('train', 'validation', 'test')

class DatasetLoadError(ValueError):
    """A dataset record could not be read or decoded."""

def load_jsonl(path: Path, *, strict: bool = True) -> Iterator[dict]:
    """Yield JSON objects; strict=False logs and skips malformed JSON/shape rows."""
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                message = f'{path}:{line_no}: blank JSONL line'
                if strict: raise DatasetLoadError(message)
                LOG.warning(message); continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('expected a JSON object')
            except (json.JSONDecodeError, ValueError) as exc:
                message = f'{path}:{line_no}: invalid JSONL record: {exc}'
                if strict: raise DatasetLoadError(message) from exc
                LOG.warning(message); continue
            yield record

def load_split(data_dir: Path, split: str, *, strict: bool = True) -> list[dict]:
    if split not in SPLITS: raise ValueError(f'Unknown split: {split!r}')
    return list(load_jsonl(Path(data_dir) / f'{split}.jsonl', strict=strict))
