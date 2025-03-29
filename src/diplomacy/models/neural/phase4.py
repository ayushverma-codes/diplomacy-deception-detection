"""Three Phase 4 architectures: message BiLSTM, concatenated BiLSTM, hierarchical forward Context LSTM."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from diplomacy.models.neural.lstm import StandardLSTM

class MessageBiLSTM(StandardLSTM):
    """Exactly Phase 3's bidirectional message-only encoder (control replication)."""
    pass

class ConcatenatedContextBiLSTM(StandardLSTM):
    """Same word encoder, input is target plus selected neighboring raw text."""
    pass

class HierarchicalContextLSTM(nn.Module):
    """Bidirectional word encoder, forward LSTM over chronologically ordered message vectors."""
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=100, context_hidden_size=100,
                 bidirectional=True, num_layers=1, context_num_layers=1, dropout=0.5,
                 freeze_embeddings=True, embedding_weights=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        if embedding_weights is not None:
            if tuple(embedding_weights.shape) != tuple(self.embedding.weight.shape):
                raise ValueError('Embedding shape mismatch')
            with torch.no_grad(): self.embedding.weight.copy_(embedding_weights)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.word_lstm=nn.LSTM(embedding_dim,hidden_size,num_layers=num_layers,
                              bidirectional=bidirectional,batch_first=True,
                              dropout=dropout if num_layers>1 else 0.)
        message_dim=hidden_size*(2 if bidirectional else 1)
        self.context_lstm=nn.LSTM(message_dim,context_hidden_size,num_layers=context_num_layers,
                                  bidirectional=False,batch_first=True,
                                  dropout=dropout if context_num_layers>1 else 0.)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(context_hidden_size,2)

    def forward(self, input_ids, lengths, message_mask):
        B,M,T=input_ids.shape
        if input_ids.shape[:2] != lengths.shape or lengths.shape != message_mask.shape:
            raise ValueError('Input dimensions mismatch')
        flat=input_ids.view(B*M,T)
        flat_lengths=lengths.reshape(-1).clamp(min=1)
        embedded=self.embedding(flat)
        packed=pack_padded_sequence(embedded,flat_lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_out,_=self.word_lstm(packed)
        output,_=pad_packed_sequence(packed_out,batch_first=True,total_length=T)
        word_mask=torch.arange(T,device=input_ids.device)[None,:]<flat_lengths[:,None]
        output=output.masked_fill(~word_mask.unsqueeze(-1),torch.finfo(output.dtype).min)
        message_vectors=output.max(dim=1).values.view(B,M,-1)
        message_vectors=message_vectors*message_mask.unsqueeze(-1)
        message_lengths=message_mask.long().sum(dim=1).clamp(min=1)
        context_packed=pack_padded_sequence(message_vectors,message_lengths.cpu(),batch_first=True,enforce_sorted=False)
        context_out,_=self.context_lstm(context_packed)
        context,_=pad_packed_sequence(context_out,batch_first=True,total_length=M)
        return self.classifier(self.dropout(context))
