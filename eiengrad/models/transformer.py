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
