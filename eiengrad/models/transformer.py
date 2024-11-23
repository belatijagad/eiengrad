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

