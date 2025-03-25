"""Bidirectional max-pooled, message-only LSTM from the paper's config."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class StandardLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=100,
                 bidirectional=True, num_layers=1, dropout=0.5,
                 freeze_embeddings=True, embedding_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            if tuple(embedding_weights.shape) != tuple(self.embedding.weight.shape):
                raise ValueError('Embedding matrix shape mismatch')
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_weights)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * (2 if bidirectional else 1), 2)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.detach().cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed)
        sequence, _ = pad_packed_sequence(packed_outputs, batch_first=True,
                                         total_length=input_ids.shape[1])
        mask = torch.arange(sequence.shape[1], device=sequence.device)[None, :] < lengths[:, None]
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), torch.finfo(sequence.dtype).min)
        pooled = sequence.max(dim=1).values
        return self.classifier(self.dropout(pooled))
