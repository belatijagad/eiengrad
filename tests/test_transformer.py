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
    self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.H, dropout_prob=self.dropout_prob)
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
    
class TestAFTFull(unittest.TestCase):
  def setUp(self):
    self.max_seqlen = 64
    self.d_model = 512
    self.d_hidden = 128
    self.H = 8
    self.Ti = 48
    self.Tj = 32
    self.B = 2
    self.aft = AttentionFreeTransformer(d_model=self.d_model, max_seqlen=self.max_seqlen, d_hidden=self.d_hidden)
  def test_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = self.aft(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti, dtype=torch.bool)).view(1, self.Ti, self.Ti)
    output = self.aft(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj, dtype=torch.bool)).view(1, self.Ti, self.Tj)
    output = self.aft(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
    
class TestAFTLocal(unittest.TestCase):
  def setUp(self):
    self.max_seqlen = 64
    self.d_model = 512
    self.d_hidden = 128
    self.local_window_size = 16
    self.H = 8
    self.Ti = 48
    self.Tj = 32
    self.B = 2
    self.aft = AttentionFreeTransformer(d_model=self.d_model, max_seqlen=self.max_seqlen, d_hidden=self.d_hidden, local_window_size=self.local_window_size)
  def test_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = self.aft(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti, dtype=torch.bool)).view(1, self.Ti, self.Ti)
    output = self.aft(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj, dtype=torch.bool)).view(1, self.Ti, self.Tj)
    output = self.aft(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
    
class TestAFTSimple(unittest.TestCase):
  def setUp(self):
    self.max_seqlen = 64
    self.d_model = 512
    self.d_hidden = 128
    self.local_window_size = 0 # The only difference between AFTLocal and AFTSimple is that AFTSimple has s = 0
    self.H = 8
    self.Ti = 48
    self.Tj = 32
    self.B = 2
    self.aft = AttentionFreeTransformer(d_model=self.d_model, max_seqlen=self.max_seqlen, d_hidden=self.d_hidden, local_window_size=self.local_window_size)
  def test_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    output = self.aft(x, x, x)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_masked_self_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Ti, dtype=torch.bool)).view(1, self.Ti, self.Ti)
    output = self.aft(x, x, x, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))
  def test_cross_attention(self):
    x = torch.rand(self.B, self.Ti, self.d_model)
    y = torch.rand(self.B, self.Tj, self.d_model)
    mask = torch.tril(torch.ones(self.Ti, self.Tj, dtype=torch.bool)).view(1, self.Ti, self.Tj)
    output = self.aft(x, y, y, mask)
    self.assertEqual(output.shape, (self.B, self.Ti, self.d_model))