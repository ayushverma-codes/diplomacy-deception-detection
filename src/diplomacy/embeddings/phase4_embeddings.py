"""Optional reuse of frozen pretrained GloVe vectors from a verified Phase 3 checkpoint."""
from pathlib import Path
import torch
from diplomacy.training.lstm_trainer import load_checkpoint


def initialize_from_phase3(vocab, root, dim=200, seed=1994):
    """Copy frozen Phase 3 *GloVe* vectors, without loading a large archive again.

    Vocabulary is always constructed from Phase 4 training data. New tokens receive
    seeded random initialization; this is an explicitly documented reproduction deviation.
    """
    path=Path(root)/'artifacts'/'models'/'neural'/'standard_lstm'/'actual_lie'/'best_model.pt'
    if not path.exists():
        raise FileNotFoundError(f'Phase 3 pretrained checkpoint not found: {path}')
    state=load_checkpoint(path,'cpu')
    if state['embedding_info'].get('mode')!='glove' or not state['model_config']['freeze_embeddings']:
        raise ValueError('Phase 3 checkpoint must use frozen pretrained GloVe')
    old_vocab=state['vocabulary']['token_to_id']
    old_weights=state['model_state']['embedding.weight']
    if old_weights.shape[1]!=dim:
        raise ValueError('Embedding dimension mismatch')
    gen=torch.Generator().manual_seed(seed)
    weights=torch.randn(len(vocab),dim,generator=gen)*0.05
    weights[0].zero_()
    copied=0
    for token,position in vocab.token_to_id.items():
        old=old_vocab.get(token)
        if old is not None and token!='<PAD>':
            weights[position]=old_weights[old]
            copied+=1
    return weights,{
        'mode':'phase3_glove_reuse',
        'source_checkpoint':str(path),
        'source_embedding_info':state['embedding_info'],
        'copied_tokens':copied,
        'vocab_size':len(vocab),
        'new_token_count':len(vocab)-1-copied,
        'exact_glove_reproduction':False,
    }
