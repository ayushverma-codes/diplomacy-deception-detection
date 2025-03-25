"""Message-only tokenization and training-only vocabulary for Phase 3."""
from collections import Counter
import json
import re
from pathlib import Path
from torch.utils.data import Dataset

PAD = '<PAD>'
UNK = '<UNK>'
# A documented, lightweight approximation of the original AllenNLP tokenizer.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def tokenize(text: str, lowercase=True) -> list[str]:
    tokens = TOKEN_PATTERN.findall(str(text or ''))
    tokens = [t.lower() for t in tokens] if lowercase else tokens
    return tokens or [UNK]

class Vocabulary:
    def __init__(self, token_to_id: dict[str, int], lowercase=True):
        self.token_to_id = dict(token_to_id)
        assert self.token_to_id[PAD] == 0 and self.token_to_id[UNK] == 1
        self.id_to_token = [None] * len(token_to_id)
        for token, index in token_to_id.items():
            self.id_to_token[index] = token
        self.lowercase = lowercase

    @classmethod
    def fit(cls, texts, min_freq=1, max_size=None, lowercase=True):
        counts = Counter(token for text in texts for token in tokenize(text, lowercase))
        terms = sorted((t for t, n in counts.items() if n >= min_freq and t not in {PAD, UNK}),
                       key=lambda t: (-counts[t], t))
        if max_size is not None:
            terms = terms[:max(0, int(max_size) - 2)]
        return cls({t: i for i, t in enumerate([PAD, UNK] + terms)}, lowercase)

    def encode(self, text, max_tokens=None):
        tokens = tokenize(text, self.lowercase)
        if max_tokens is not None:
            tokens = tokens[:max(1, int(max_tokens))]
        return [self.token_to_id.get(t, 1) for t in tokens]

    def to_dict(self):
        return {'token_to_id': self.token_to_id, 'lowercase': self.lowercase}

    @classmethod
    def from_dict(cls, data):
        return cls(data['token_to_id'], data['lowercase'])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf8')))

    def __len__(self):
        return len(self.token_to_id)

def load_rows(path: Path, task: str):
    records = []
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('target') not in (0, 1) or type(row.get('target')) is not int:
                raise ValueError(f'Invalid binary target in {path}')
            if row.get('target_name') != task:
                raise ValueError(f'Task mismatch in {path}: {row.get("target_name")}')
            if not isinstance(row.get('message'), str):
                raise ValueError('Invalid message')
            records.append(row)
    ids = [r['message_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate message IDs in {path}')
    return records

class MessageDataset(Dataset):
    def __init__(self, records, vocab: Vocabulary, max_tokens=256):
        self.records = records
        self.vocab = vocab
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ids = self.vocab.encode(row['message'], self.max_tokens)
        return {'input_ids': ids, 'length': len(ids), 'target': int(row['target']),
                'message_id': row['message_id'], 'game_id': row['game_id']}
