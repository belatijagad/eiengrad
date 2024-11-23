import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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