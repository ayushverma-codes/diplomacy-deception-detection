"""Game-safe Phase 5 data, original sender-minus-recipient power and masking."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import math
import statistics
import torch
from torch.utils.data import Dataset
from diplomacy.data.text_dataset import Vocabulary

TASK_LABELS = {'actual_lie':'sender_labels','suspected_lie':'receiver_labels'}

def binary_label(value):
    if value is True or str(value).lower() == 'true': return 0
    if value is False or str(value).lower() == 'false': return 1
    if str(value).upper() == 'NOANNOTATION': return -100
    raise ValueError(f'Unexpected original annotation: {value!r}')

def parse_score_delta(value):
    if value is None or str(value).strip() == '': return None
    try: number=float(value)
    except (TypeError, ValueError): return None
    if not math.isfinite(number) or not -18 <= number <= 18:
        raise ValueError(f'Invalid Diplomacy score differential: {value!r}')
    return number

def load_games(root, split, task):
    """Original JSONL supplies aligned game_score_delta = sender minus recipient.

    No game_score fallback: that is the sender score, NOT their differential.
    Messages without receiver annotation are still represented in context.
    """
    if split not in ('train','validation','test') or task not in TASK_LABELS:
        raise ValueError('Invalid split or task')
    groups=defaultdict(list)
    with (Path(root)/'data'/f'{split}.jsonl').open(encoding='utf8') as f:
        for conversation_index,line in enumerate(f):
            record=json.loads(line)
            count=len(record['messages'])
            fields=('sender_labels','receiver_labels','speakers','receivers',
                    'absolute_message_index','game_score_delta')
            if any(len(record[key])!=count for key in fields):
                raise ValueError('Original arrays have misaligned lengths')
            for i in range(count):
                # Validate original sender score; never replace the differential with it.
                if 'game_score' in record:
                    if len(record['game_score'])!=count: raise ValueError('Sender-score alignment error')
                    try: float(record['game_score'][i])
                    except (ValueError,TypeError): pass
                raw=record['game_score_delta'][i]
                delta=parse_score_delta(raw)
                entry={
                    'game_id':record['game_id'],
                    'message_id':f"{record['game_id']}:{conversation_index}:{i}",
                    'absolute_message_index':int(record['absolute_message_index'][i]),
                    'conversation_index':conversation_index,
                    'message_position':i,
                    'message':record['messages'][i],
                    'speaker':record['speakers'][i],
                    'receiver':record['receivers'][i],
                    'score_delta':delta,
                    'target':binary_label(record[TASK_LABELS[task]][i]),
                }
                groups[entry['game_id']].append(entry)
    for game,records in groups.items():
        records.sort(key=lambda x:(x['absolute_message_index'],x['conversation_index'],x['message_position']))
        if len(set(r['message_id'] for r in records))!=len(records):
            raise ValueError('Duplicate message ID')
    return dict(groups)

class PowerScaler:
    """Train-only median imputation and standardization. Missingness is explicit."""
    def __init__(self, median=0., mean=0., std=1., fitted=False):
        self.median=float(median);self.mean=float(mean);self.std=float(std);self.fitted=fitted

    @classmethod
    def fit(cls, training_games):
        values=[r['score_delta'] for g in training_games.values() for r in g if r['score_delta'] is not None]
        median=float(statistics.median(values)) if values else 0.
        completed=[median if r['score_delta'] is None else r['score_delta'] for g in training_games.values() for r in g]
        mean=float(statistics.mean(completed)) if completed else 0.
        std=float(statistics.pstdev(completed)) if len(completed)>1 else 1.
        return cls(median,mean,max(std,1e-6),True)

    def transform(self,value):
        if not self.fitted: raise ValueError('PowerScaler must be fitted on training games')
        return (float(self.median if value is None else value)-self.mean)/self.std

    def to_dict(self):
        return {'median':self.median,'mean':self.mean,'std':self.std,'fitted':self.fitted}

    @classmethod
    def from_dict(cls, obj): return cls(**obj)

class Phase5ConversationDataset(Dataset):
    """Chunks preserve game order; masked annotations stay in context.

    Chunk boundaries are an experimental adaptation of original full-game context.
    """
    def __init__(self, games, vocab=None, tokenizer=None, max_messages=12, max_tokens=96, power_scaler=None):
        if (vocab is None)==(tokenizer is None): raise ValueError('Exactly one tokenization method required')
        if max_messages<1 or max_tokens<2: raise ValueError('Invalid sequence limits')
        self.samples=[]
        for game, messages in games.items():
            for start in range(0,len(messages),max_messages):
                chunk=messages[start:start+max_messages]
                if tokenizer is not None:
                    enc=tokenizer([r['message'] for r in chunk],padding=False,truncation=True,max_length=max_tokens,
                                  add_special_tokens=True)
                    ids=enc['input_ids'];attn=enc['attention_mask']
                else:
                    ids=[vocab.encode(r['message'],max_tokens) for r in chunk]
                    attn=[[1]*len(x) for x in ids]
                self.samples.append({
                    'game_id':game, 'message_ids':[r['message_id'] for r in chunk],
                    'indices':[r['absolute_message_index'] for r in chunk],
                    'texts':[r['message'] for r in chunk],
                    'input_ids':ids,'attention_mask':attn,
                    'labels':[r['target'] for r in chunk],
                    'power':[power_scaler.transform(r['score_delta']) if power_scaler else 0. for r in chunk],
                    'raw_power':[r['score_delta'] for r in chunk],
                })
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]

def collate_phase5(batch):
    b=len(batch);m=max(len(r['labels']) for r in batch)
    t=max(len(tokens) for r in batch for tokens in r['input_ids'])
    ids=torch.zeros((b,m,t),dtype=torch.long)
    attention=torch.zeros((b,m,t),dtype=torch.long)
    mask=torch.zeros((b,m),dtype=torch.bool)
    labels=torch.full((b,m),-100,dtype=torch.long)
    power=torch.zeros((b,m,1),dtype=torch.float32)
    lengths=torch.ones((b,m),dtype=torch.long)
    for i,record in enumerate(batch):
        for j,(tok,att) in enumerate(zip(record['input_ids'],record['attention_mask'])):
            ids[i,j,:len(tok)]=torch.tensor(tok)
            attention[i,j,:len(att)]=torch.tensor(att)
            mask[i,j]=True;labels[i,j]=record['labels'][j]
            lengths[i,j]=len(tok);power[i,j,0]=record['power'][j]
    return {'input_ids':ids,'attention_mask':attention,'message_mask':mask,
            'lengths':lengths,'targets':labels,'power':power,
            'game_ids':[r['game_id'] for r in batch],
            'message_ids':[r['message_ids'] for r in batch],
            'indices':[r['indices'] for r in batch]}
