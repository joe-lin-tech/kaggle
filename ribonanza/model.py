import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class RNALoss(nn.Module):
    def __init__(self):
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, preds, labels):
        mask = labels['mask']
        preds = preds[mask]
        labels = torch.clip(labels['reactivity'][mask], 0, 1)
        mae = self.mae_loss(preds, labels)
        return mae[~torch.isnan(mae)].mean()
    

class RNAPredictor(nn.Module):
    def __init__(self, d_model: int, n_heads: int): # n_heads should be d_model // head_size
        super(RNAPredictor).__init__()
        self.embedding = nn.Embedding(4, d_model) # 4 corresponds to the number of RNA base types
        self.pos_encoding = PositionalEncoding(d_model // 2)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, activation=F.gelu, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 12)
        self.out = nn.Linear(d_model, 2)
    
    def forward(self, inputs):
        mask = inputs['mask']
        batch_length = torch.sum(mask, -1).max()
        mask = mask[:, :batch_length]
        x = inputs['sequence'][:, :batch_length]
        embeddings = self.embedding(x) + self.pos_encoding
        return self.out(self.transformer(embeddings, src_key_padding_mask=~mask))
        