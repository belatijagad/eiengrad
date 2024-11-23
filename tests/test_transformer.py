import torch
import unittest
from eiengrad.models.transformer.multi_head_attention import *
from eiengrad.models.transformer.attention_free_transformer import *

class TestMultiHeadAttention(unittest.TestCase):
  def setUp(self):
    self.d_model = 512
    self.H = 8
    self.dropout_prob = 0.1
    self.Ti = 48
    self.Tj = 32
    self.B = 2
    self.mha = MultiHeadAttention(self.d_model, self.H, self.dropout_prob)
  def test_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = self.mha(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti)).view(1, 1, self.Ti, self.Ti)
    output = self.mha(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj)).view(1, 1, self.Ti, self.Tj)
    output = self.mha(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
    
class TestAttentionFreeTransformer(unittest.TestCase):
  def setUp(self):
    self.d_model = 512
    self.H = 8
    self.dropout_prob = 0.1
    self.Ti = 48
    self.Tj = 32
    self.B = 2
    self.aft = AttentionFreeTransformer(self.d_model, self.dropout_prob)
  def test_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = self.aft(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti)).view(1, 1, self.Ti, self.Ti)
    output = self.aft(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj)).view(1, 1, self.Ti, self.Tj)
    output = self.aft(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))