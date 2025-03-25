"""Optional exact GloVe Twitter 27B 200d loading, reading only needed tokens."""
from pathlib import Path
import io
import urllib.request
import zipfile
import torch

GLOVE_URL = 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
GLOVE_MEMBER = 'glove.twitter.27B.200d.txt'

def ensure_glove_zip(path, download=False):
    path = Path(path)
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f'GloVe archive missing: {path}. Set download_glove=true to obtain ~1.4 GB archive.')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    try:
        urllib.request.urlretrieve(GLOVE_URL, tmp)
        with zipfile.ZipFile(tmp) as z:
            if GLOVE_MEMBER not in z.namelist():
                raise ValueError('Expected 200d Twitter GloVe member not found')
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path

def initialize_embeddings(vocab, dim=200, mode='glove', glove_zip=None, download_glove=False, seed=1994):
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(len(vocab), dim, generator=generator) * 0.05
    weights[0].zero_()
    covered = 0
    if mode == 'random':
        return weights, {'mode': 'random', 'coverage': 0, 'source': None}
    if mode != 'glove':
        raise ValueError(f'Unknown embedding mode: {mode}')
    if dim != 200:
        raise ValueError('Twitter GloVe 27B selected member has dimension 200')
    archive = ensure_glove_zip(glove_zip, download_glove)
    with zipfile.ZipFile(archive) as z:
        with z.open(GLOVE_MEMBER) as raw:
            for line in io.TextIOWrapper(raw, encoding='utf8', errors='replace'):
                pieces = line.rstrip().split(' ')
                if len(pieces) < dim + 1:
                    continue
                token = ' '.join(pieces[:-dim])
                index = vocab.token_to_id.get(token)
                if index is None or index == 0:
                    continue
                try:
                    weights[index] = torch.tensor([float(v) for v in pieces[-dim:]])
                    covered += 1
                except ValueError:
                    continue
    return weights, {'mode': 'glove', 'source': GLOVE_URL, 'member': GLOVE_MEMBER,
                     'coverage': covered, 'vocab_size': len(vocab),
                     'coverage_percent': 100 * covered / max(1, len(vocab) - 2)}
