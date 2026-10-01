"""Conv1D + Transformer with ordered horizons (plan section 7.2)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal(length: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ConvTransformer(nn.Module):
    def __init__(self, tab_dim: int = 67, channels=(32, 64, 64), kernel: int = 7, strides=(2, 2, 5),
                 d_model: int = 64, heads: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1,
                 tab_mlp=(64, 64), head_hidden: int = 64, n_out: int = 5):
        super().__init__()
        convs, c_in = [], 2
        for c, s in zip(channels, strides):
            convs += [nn.Conv1d(c_in, c, kernel, stride=s, padding=kernel // 2), nn.BatchNorm1d(c), nn.GELU()]
            c_in = c
        self.conv = nn.Sequential(*convs)
        self.proj = nn.Identity() if c_in == d_model else nn.Linear(c_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.tab = nn.Sequential(nn.Linear(tab_dim, tab_mlp[0]), nn.GELU(), nn.Linear(tab_mlp[0], tab_mlp[1]))
        self.head = nn.Sequential(nn.Linear(d_model + tab_mlp[1], head_hidden), nn.GELU(), nn.Linear(head_hidden, n_out))
        self.d_model = d_model

    def forward(self, wave: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.conv(wave).transpose(1, 2))           # [B, L, d]
        z = z + sinusoidal(z.shape[1], self.d_model, z.device).to(z.dtype)
        z_wave = self.encoder(z).mean(1)                          # [B, d]
        a = self.head(torch.cat([z_wave, self.tab(tab)], 1))      # [B, 5]
        # logit_1 = a_1; logit_k = logit_{k-1} + softplus(a_k): probabilities increase with the horizon
        return torch.cumsum(torch.cat([a[:, :1], F.softplus(a[:, 1:])], 1), 1)

    @classmethod
    def from_config(cls, dl, tab_dim: int = 67) -> "ConvTransformer":
        c, t = dl.conv, dl.transformer
        return cls(tab_dim=tab_dim, channels=tuple(c["channels"]), kernel=c["kernel"], strides=tuple(c["strides"]),
                   d_model=t["d_model"], heads=t["heads"], layers=t["layers"], ff=t["ff"], dropout=t["dropout"],
                   tab_mlp=tuple(dl.tab_mlp), head_hidden=dl.head_hidden)


def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean BCE over label cells != -1."""
    known = y >= 0
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[known], y[known])
