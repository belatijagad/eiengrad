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
  
class DecoderBlock(nn.Module):
  def __init__(self, features: int, masked_self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout_prob: float) -> None:
    self.masked_self_attention_block = masked_self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout_prob=dropout_prob) for _ in range(3)])
  def forward(self, x, y, src_mask, tgt_mask):
    x = self.residual_connections[0](x, lambda x: self.masked_self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, y, y, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x