import torch
import unittest
from eiengrad.models.transformer import *

class TestMultiHeadAttention(unittest.TestCase):
  def setUp(self):
    self.d_model = 512
    self.H = 8
    self.dropout_prob = 0.1
    self.Ti = 48
    self.Tj = 32
    self.B = 2
  def test_self_attention(self):
    mha = MultiHeadAttention(self.d_model, self.H, self.dropout_prob)
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = mha(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    mha = MultiHeadAttention(self.d_model, self.H, self.dropout_prob)
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti)).view(1, 1, self.Ti, self.Ti)
    output = mha(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    mha = MultiHeadAttention(self.d_model, self.H, self.dropout_prob)
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj)).view(1, 1, self.Ti, self.Tj)
    output = mha(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))