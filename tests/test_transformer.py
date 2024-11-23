import torch
import unittest
from eiengrad.models.transformer import *

class TestMultiHeadAttention(unittest.TestCase):
  def test_output_shape(self):
    d_model = 512
    num_heads = 8
    dropout_prob = 0.1
    seq_len = 10
    batch_size = 2
    
    mha = MultiHeadAttention(d_model, num_heads, dropout_prob)
    q, k, v = (torch.rand(batch_size, seq_len, d_model) for _ in range(3))
    mask = torch.ones(batch_size, 1, seq_len, d_model)
    output = mha(q, k, v, mask)
    self.assertEquals(output.shape, [batch_size, seq_len, d_model])