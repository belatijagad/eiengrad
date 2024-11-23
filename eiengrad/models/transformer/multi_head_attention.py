import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, dropout_prob: float) -> None:
    super().__init__()
    assert d_model % num_heads == 0, "Embedding dimension must be divisible by num_heads"
    self.d_model = d_model
    self.H = num_heads
    self.d_k = d_model // num_heads
    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)
    self.w_o = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(p=dropout_prob)
  def forward(self, q, k, v, mask=None):
    B, Ti, d_model = q.shape
    _, Tj, _       = k.shape
    # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
    Q = self.w_q(q).view(B, Ti, self.H, self.d_k)
    K = self.w_k(k).view(B, Tj, self.H, self.d_k)
    V = self.w_v(v).view(B, Tj, self.H, self.d_k)
    attention_logits = torch.einsum('bihd,bjhd->bhij', [Q, K]) / math.sqrt(self.d_k)
    if mask is not None:
      attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
    attention_logits = attention_logits.softmax(dim=-1)
    self.attention = self.dropout(attention_logits)
    attention_scores = torch.einsum('bhij,bjhd->bihd', [self.attention, V])
    attention_scores = attention_scores.reshape(B, Ti, d_model)
    return self.w_o(attention_scores)
