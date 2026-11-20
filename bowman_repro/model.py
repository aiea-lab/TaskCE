"""Bowman inference layers from CoEx compexp-nli/code/models.py."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 300, padding_idx=1)
        self.rnn = nn.LSTM(300, 512)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.emb(tokens), lengths.cpu(), enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        return hidden[-1]


class BowmanEntailmentClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = TextEncoder(vocab_size)
        self.bn = nn.BatchNorm1d(4 * 512)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(4 * 512, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, 3)
        )

    def forward(self, premise, premise_lengths, hypothesis, hypothesis_lengths):
        p = self.encoder(premise, premise_lengths)
        h = self.encoder(hypothesis, hypothesis_lengths)
        pair = torch.cat([p, h, p - h, p * h], dim=1)
        return self.mlp(self.dropout(self.bn(pair)))
