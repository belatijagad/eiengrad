import torch
import torch.nn as nn
import torch.nn.functional as F

class AFTFull(nn.Module):
  def __init__(self, d_model: int, max_seqlen: int, d_hidden: int, activation=F.sigmoid, bias: bool=False) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_hidden = d_hidden
    self.activation = activation
    self.w_q = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_k = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_v = nn.Linear(d_model, d_hidden, bias=bias)
    self.w_o = nn.Linear(d_hidden, d_model, bias=bias)
    self.W_bias = nn.Parameter(torch.Tensor(max_seqlen, max_seqlen), requires_grad=True)
    nn.init.xavier_uniform_(self.W_bias)
  def forward(self, q, k, v, mask=None):
    _, Ti, _ = q.shape
    _, Tj, _ = k.shape
    Q = self.w_q(q)
    K = self.w_k(k)
    V = self.w_v(v)
    pos_bias = self.W_bias[:Ti, :Tj].unsqueeze(0)
    max_key = K.max(dim=0, keepdims=True)[0]
    max_pos_bias = pos_bias.max(dim=0, keepdims=True)[0]
    exp_key = torch.exp(K - max_key) # For numerical stability
    exp_pos_bias = torch.exp(pos_bias - max_pos_bias)
    # [B, Ti, Tj] @ [B, Tj, d_hidden] -> [B, Ti, d_hidden]
    num = torch.einsum('bij,bjd->bid', [exp_pos_bias, exp_key * V])
    den = torch.einsum('bij,bjd->bid', [exp_pos_bias, exp_key])
    Yt = self.activation(Q) * num / den
    return self.w_o(Yt)
