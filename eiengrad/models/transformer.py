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
    self.num_heads = num_heads
    self.d_k = d_model // num_heads
    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)
    self.w_o = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(p=dropout_prob)
  def forward(self, q, k, v, mask):
    batch, seq_len, d_model = q.shape
    # [batch, seq_len, d_model] -> [batch, seq_len, d_k, num_heads]
    Q = self.w_q(q).view(batch, seq_len, self.d_k, self.num_heads)
    K = self.w_k(k).view(batch, seq_len, self.d_k, self.num_heads)
    V = self.w_v(v).view(batch, seq_len, self.d_k, self.num_heads)
    # [batch, seq_len, d_k, num_heads] @ [batch, d_k, seq_len, num_heads] -> [batch, seq_len, seq_len, num_heads]
    attention_logits = torch.einsum('bsdh,bdsh->bssh', [Q, K]) / math.sqrt(self.d_k)
    if mask is not None:
      attention_logits = attention_logits.masked_fill_(mask == 0, float('-inf'))
    attention_logits = attention_logits.softmax(dim=-1)
    self.attention = self.dropout(attention_scores)
    # [batch, seq_len, seq_len, num_head] @ [batch, seq_len, d_k, num_head] -> [batch, seq_len, d_k, num_head]
    attention_scores = torch.einsum('bssh,bsdh->bsdh', [self.attention, V])
    # [batch, seq_len, d_k, num_head] -> [batch, seq_len, d_model]
    attention_scores = attention_scores.view(batch, seq_len, d_model)
    return self.w_o(attention_scores)