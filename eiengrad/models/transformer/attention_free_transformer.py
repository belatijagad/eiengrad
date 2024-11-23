"""
    References:
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/transformers/aft
    - https://github.com/rish-16/aft-pytorch/tree/main/aft_pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFreeTransformer(nn.Module):
  def __init__(self, d_model: int, max_seqlen: int, d_hidden: int, local_window_size: int=None, activation=F.sigmoid, bias: bool=False) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_hidden = d_hidden
    self.activation = activation
    self.local_window_size = local_window_size
    self.w_q = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_k = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_v = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_o = nn.Linear(d_hidden, d_model, bias=bias)
    self.W_bias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen), requires_grad=True)
    if local_window_size is not None: self.local_mask = nn.Parameter(self.create_local_mask(max_seqlen, local_window_size), requires_grad=False)
    nn.init.xavier_uniform_(self.W_bias)
  @staticmethod
  def create_local_mask(seq_len, local_window_size):
    local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    local_mask = torch.tril(local_mask, local_window_size - 1)
    local_mask = torch.triu(local_mask, -(local_window_size - 1))
    return local_mask
  def forward(self, q, k, v, mask=None):
    _, Ti, _ = q.shape
    _, Tj, _ = k.shape
    Q = self.w_q(q)
    K = self.w_k(k)
    V = self.w_v(v)
    pos_bias = self.W_bias[:Ti, :Tj]
    if self.local_window_size is not None: pos_bias = pos_bias * self.local_mask[:Ti, :Tj]
    pos_bias = pos_bias.unsqueeze(0)
    if mask is not None:
      assert mask.shape[1] == 1 or mask.shape[1] == q.shape[1]
      assert mask.shape[2] == k.shape[1]
      assert mask.shape[0] == 1 or mask.shape[0] == q.shape[2]
      pos_bias = pos_bias.masked_fill(~mask, float('-inf'))
    max_key = K.max(dim=0, keepdims=True)[0]
    max_pos_bias = pos_bias.max(dim=0, keepdims=True)[0]
    exp_key = torch.exp(K - max_key) # For numerical stability
    exp_pos_bias = torch.exp(pos_bias - max_pos_bias)
    # [B, Ti, Tj] @ [B, Tj, d_hidden] -> [B, Ti, d_hidden]
    num = torch.einsum('bij,bjd->bid', [exp_pos_bias, exp_key * V])
    den = torch.einsum('bij,bjd->bid', [exp_pos_bias, exp_key])
    Yt = self.activation(Q) * num / den
    return self.w_o(Yt)