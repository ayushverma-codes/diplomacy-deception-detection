"""Game-safe chronological context construction for Phase 4."""
from collections import defaultdict
import json
from pathlib import Path
from torch.utils.data import Dataset
from diplomacy.data.text_dataset import Vocabulary

SPLITS = ('train', 'validation', 'test')
TASKS = ('actual_lie', 'suspected_lie')

def chronological_games(root, split, task):
    """Restore every game message from original conversation JSONL; keep unannotated context."""
    if split not in SPLITS or task not in TASKS:
        raise ValueError('Unknown split/task')
    root = Path(root)
    raw = root/'data'/f'{split}.jsonl'
    processed = root/'artifacts'/'processed'/task/f'{split}.jsonl'
    with processed.open(encoding='utf8') as f:
        labeled = {r['message_id']: r for line in f if line.strip() for r in [json.loads(line)]}
    games = defaultdict(list)
    seen = set()
    with raw.open(encoding='utf8') as f:
        for ci, line in enumerate(f):
            if not line.strip():
                raise ValueError('Blank raw conversation would change conversation index')
            record = json.loads(line)
            n = len(record['messages'])
            if len(record['absolute_message_index']) != n:
                raise ValueError('Bad raw array alignment')
            for i, text in enumerate(record['messages']):
                mid = f"{record['game_id']}:{ci}:{i}"
                if mid in seen: raise ValueError('Duplicate message ID')
                seen.add(mid)
                label = labeled.get(mid)
                # Do not treat NOANNOTATION as negative; it is still usable as context.
                games[record['game_id']].append({
                    'message_id': mid, 'game_id': record['game_id'], 'message': text,
                    'conversation_index': ci, 'message_position': i,
                    'absolute_message_index': record['absolute_message_index'][i],
                    'target': int(label['target']) if label is not None else -100,
                })
    if any(mid not in seen for mid in labeled):
        raise ValueError('Processed label does not match original message')
    for game in games:
        games[game].sort(key=lambda r: (int(r['absolute_message_index']), r['conversation_index'], r['message_position']))
    if sum(r['target'] >= 0 for g in games.values() for r in g) != len(labeled):
        raise ValueError('Target/message alignment mismatch')
    return dict(games)

def target_preserving_text(previous, target, following=None, max_tokens=256, separator=' <CTX> ', lowercase=True):
    """Preserve target first; spend remaining token budget on nearest context.

    Prior text receives half the remaining budget when future text is present;
    any unused allocation can be spent by the other side. The target is kept
    intact when it fits. If it exceeds max_tokens, its FIRST max_tokens remain.
    Training and inference call this exact function.
    """
    from diplomacy.data.text_dataset import tokenize
    if max_tokens is None or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    target_tokens=tokenize(target,lowercase)
    if len(target_tokens)>=max_tokens:
        return ' '.join(target_tokens[:max_tokens])
    sep=tokenize(separator,lowercase)
    earlier=[tokenize(x,lowercase) for x in previous]
    later=[tokenize(x,lowercase) for x in (following or [])]
    prior_tokens=[]
    for words in earlier:
        prior_tokens.extend(words+sep)
    future_tokens=[]
    for words in later:
        future_tokens.extend(sep+words)
    remaining=max_tokens-len(target_tokens)
    if prior_tokens and future_tokens:
        half=remaining//2
        pre_budget=min(len(prior_tokens),half)
        post_budget=min(len(future_tokens),remaining-pre_budget)
        pre_budget=min(len(prior_tokens),remaining-post_budget)
    else:
        pre_budget=min(len(prior_tokens),remaining)
        post_budget=min(len(future_tokens),remaining-pre_budget)
    result=prior_tokens[-pre_budget:] if pre_budget else []
    result+=target_tokens
    if post_budget:result+=future_tokens[:post_budget]
    return ' '.join(result)


def concatenate_context(games, before=3, after=0, separator=' <CTX> ', max_tokens=256,
                        lowercase=True, include_target=True):
    """Create one target-first-budgeted input per annotated message."""
    if not include_target or before<0 or after<0 or not separator.strip():
        raise ValueError('Target, nonnegative context window and separator required')
    rows=[]
    for game, messages in games.items():
        for i,target in enumerate(messages):
            if target['target'] < 0:continue
            prior=messages[max(0,i-before):i]
            future=messages[i+1:i+1+after]
            ctx=prior+[target]+future
            combined=target_preserving_text([m['message'] for m in prior],target['message'],
                        [m['message'] for m in future],max_tokens,separator,lowercase)
            rows.append({**target,'message':combined,'source_message':target['message'],
                        'context_ids':[m['message_id'] for m in ctx]})
    return rows

class ConversationDataset(Dataset):
    """Chronologically ordered game chunks, with annotation-masked targets."""
    def __init__(self, games, vocab: Vocabulary, max_tokens=128, max_messages=16):
        if max_tokens < 1 or max_messages < 1: raise ValueError('Invalid limits')
        self.samples = []
        for game, messages in games.items():
            for start in range(0, len(messages), max_messages):
                chunk = messages[start:start + max_messages]
                self.samples.append({
                    'game_id': game,
                    'input_ids': [vocab.encode(r['message'], max_tokens) for r in chunk],
                    'lengths': [len(vocab.encode(r['message'], max_tokens)) for r in chunk],
                    'targets': [int(r['target']) for r in chunk],
                    'message_ids': [r['message_id'] for r in chunk],
                    'indices': [int(r['absolute_message_index']) for r in chunk]})
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]

def collate_conversations(batch):
    import torch
    B=len(batch); M=max(len(b['targets']) for b in batch)
    T=max(max(b['lengths']) for b in batch)
    ids=torch.zeros(B,M,T,dtype=torch.long)
    lengths=torch.ones(B,M,dtype=torch.long)
    targets=torch.full((B,M),-100,dtype=torch.long)
    mask=torch.zeros(B,M,dtype=torch.bool)
    for b,sample in enumerate(batch):
        for m,(tokens,label) in enumerate(zip(sample['input_ids'],sample['targets'])):
            ids[b,m,:len(tokens)]=torch.tensor(tokens)
            lengths[b,m]=len(tokens)
            targets[b,m]=label
            mask[b,m]=True
    return {'input_ids':ids,'lengths':lengths,'message_mask':mask,
            'targets':targets, 'message_ids':[x['message_ids'] for x in batch],
            'game_ids':[x['game_id'] for x in batch], 'indices':[x['indices'] for x in batch]}


from diplomacy.data.text_dataset import MessageDataset
from diplomacy.data.text_collator import collate_messages

class Phase4MessageDataset(MessageDataset):
    """Existing Phase 3 message inputs plus immutable original-index metadata."""
    def __getitem__(self, index):
        result=super().__getitem__(index)
        result['absolute_message_index']=self.records[index].get('absolute_message_index')
        return result


def collate_phase4_messages(batch):
    combined=collate_messages(batch)
    combined['indices']=[b['absolute_message_index'] for b in batch]
    return combined
