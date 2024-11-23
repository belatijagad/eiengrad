"""
  Reference:
  - https://github.com/hkproj/pytorch-transformer/blob/main/model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from eiengrad.models.transformer.multi_head_attention import MultiHeadAttention

class InputEmbedding(nn.Module):
  def __init__(self, vocab_size: int, d_model: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.corpus_size = vocab_size
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout_prob: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout_prob = nn.Dropout(p=dropout_prob)
    pe = torch.zeros(seq_len, d_model) # [seq_len, d_model]
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(dim=1) # [seq_len]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # Add batch dimension
    pe = pe.unsqueeze(0) # [1, seq_len, d_model]
    # register as a buffer
    self.register_buffer(pe, 'pe')
  def forward(self, x):
    x += (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return x

class LayerNorm(nn.Module):
  def __init__(self, features: int, eps: float=1e-6) -> None:
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features)) # Multiply
    self.bias = nn.Parameter(torch.zeros(features)) # Addition
  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
  
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_hidden: int, dropout_prob: float) -> None:
    self.input = nn.Linear(in_features=d_model, out_features=d_hidden)
    self.dropout = nn.Dropout(p=dropout_prob)
    self.output = nn.Linear(in_features=d_hidden, out_features=d_model)
  def forward(self, x):
    x = F.relu(self.input(x))
    x = self.dropout(x)
    return self.output(x)
  
class ResidualConnection(nn.Module):
  def __init__(self, features=int, dropout_prob=float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)
    self.norm = LayerNorm(features)
  def forward(self, x):
    return x + self.dropout(self.norm(x))
  
class EncoderBlock(nn.Module):
  def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout_prob: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = self.feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout_prob=dropout_prob) for _ in range(2)])
  def forward(self, x, src_mask):
    x = self.residual_connections(x)[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connections(x)[1](x, self.feed_forward_block)
    return x
  
class Encoder(nn.Module):
  def __init__(self, features: int, layers: nn.ModuleList) -> None:
    super().__init__()
    self.layers = layers
    self.layernorm = LayerNorm(features=features)
  def forward(self, x, mask):
    for layer in self.layers: x = layer(x, mask)
    return self.norm(x)
  
  