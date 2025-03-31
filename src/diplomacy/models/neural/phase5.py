"""Four configurable Phase 5 variants; Phase 4 baseline reused unchanged."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from diplomacy.models.neural.phase4 import HierarchicalContextLSTM

class HierarchicalPowerLSTM(HierarchicalContextLSTM):
    """Phase 4 text encoder/context LSTM; concatenate standardized power before class head.

    The fusion point is an explicit adaptation, not a claim of exact AllenNLP fidelity.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.classifier=nn.Linear(self.context_lstm.hidden_size+1,2)

    def forward(self,input_ids,lengths,message_mask,power):
        if power.shape != (*message_mask.shape,1): raise ValueError('Power alignment mismatch')
        # Reuse the parent encoding path, intercept the final classifier through a hook-free method.
        B,M,T=input_ids.shape
        flat=input_ids.reshape(B*M,T)
        flat_lengths=lengths.reshape(-1).clamp(min=1)
        embedded=self.embedding(flat)
        packed=pack_padded_sequence(embedded,flat_lengths.cpu(),batch_first=True,enforce_sorted=False)
        output,_=pad_packed_sequence(self.word_lstm(packed)[0],batch_first=True,total_length=T)
        valid=torch.arange(T,device=flat.device)[None,:]<flat_lengths[:,None]
        output=output.masked_fill(~valid.unsqueeze(-1),torch.finfo(output.dtype).min)
        vectors=output.max(dim=1).values.reshape(B,M,-1)*message_mask.unsqueeze(-1)
        counts=message_mask.long().sum(1).clamp(min=1)
        conv,_=pad_packed_sequence(self.context_lstm(pack_padded_sequence(vectors,counts.cpu(),batch_first=True,enforce_sorted=False))[0],batch_first=True,total_length=M)
        fused=torch.cat([self.dropout(conv),power.to(conv.dtype)],dim=-1)
        return self.classifier(fused)

class HierarchicalBERTLSTM(nn.Module):
    """BERT pooler per message -> forward context LSTM -> optional power -> binary head.

    For testing, pass a small BERT-compatible backbone; in production use pretrained
    bert-base-uncased from HuggingFace. Fine-tune when freeze_bert=False.
    """
    def __init__(self,bert_backbone,context_hidden_size=200,context_num_layers=1,dropout=0.4,
                 use_power=False,freeze_bert=False):
        super().__init__()
        self.bert=bert_backbone
        self.use_power=bool(use_power)
        self.freeze_bert=bool(freeze_bert)
        for p in self.bert.parameters():p.requires_grad=not freeze_bert
        hidden=int(self.bert.config.hidden_size)
        self.context_lstm=nn.LSTM(hidden,context_hidden_size,num_layers=context_num_layers,
                                  batch_first=True,bidirectional=False,
                                  dropout=dropout if context_num_layers>1 else 0.)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(context_hidden_size+int(use_power),2)

    def forward(self,input_ids,attention_mask,message_mask,power=None):
        B,M,T=input_ids.shape
        if attention_mask.shape != input_ids.shape or message_mask.shape != (B,M):
            raise ValueError('Input/attention/message masks misaligned')
        if self.use_power and (power is None or power.shape!=(B,M,1)):
            raise ValueError('Power shape must be [batch,messages,1]')
        flat_ids=input_ids.reshape(B*M,T)
        flat_mask=attention_mask.reshape(B*M,T)
        # Process only actual messages so padded BERT messages do not alter the result.
        valid=message_mask.reshape(-1)
        if not bool(valid.any()):raise ValueError('Empty conversation batch')
        outputs=self.bert(input_ids=flat_ids[valid],attention_mask=flat_mask[valid])
        pool=getattr(outputs,'pooler_output',None)
        if pool is None:
            # Clearly documented fallback for custom backbones; actual BERT has pooler.
            pool=outputs.last_hidden_state[:,0,:]
        reprs=pool.new_zeros(B*M,pool.shape[-1])
        reprs[valid]=pool
        messages=reprs.reshape(B,M,-1)
        lengths=message_mask.long().sum(1).clamp(min=1)
        packed=pack_padded_sequence(messages,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=pad_packed_sequence(self.context_lstm(packed)[0],batch_first=True,total_length=M)
        if self.use_power:encoded=torch.cat([encoded,power.to(encoded.dtype)],dim=-1)
        return self.classifier(self.dropout(encoded))
