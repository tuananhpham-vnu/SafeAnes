"""Independent compact adaptations; see docs/SOURCES.md#review-e03.

Only observed history is passed here. Symmetric convolutions stay within that
window; they do not claim causal per-timestep hidden states as the TCN does.
"""

import torch
from torch import nn
from torch.nn import functional as F


class InceptionModule(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.bottleneck = nn.Conv1d(inputs, width, 1, bias=False)
        self.branches = nn.ModuleList([nn.Conv1d(width, width, k, padding=k // 2, bias=False)
                                       for k in (39, 19, 9)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(inputs, width, 1, bias=False))
        self.norm = nn.BatchNorm1d(width * 4)

    def forward(self, x):
        reduced = self.bottleneck(x)
        return F.relu(self.norm(torch.cat([branch(reduced) for branch in self.branches] + [self.pool(x)], dim=1)))


class InceptionEncoder(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.groups = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        for _ in range(2):
            self.groups.append(nn.Sequential(InceptionModule(inputs, width),
                InceptionModule(width * 4, width), InceptionModule(width * 4, width)))
            self.shortcuts.append(nn.Sequential(nn.Conv1d(inputs, width * 4, 1, bias=False), nn.BatchNorm1d(width * 4)))
            inputs = width * 4

    def forward(self, x):
        for group, shortcut in zip(self.groups, self.shortcuts):
            x = F.relu(group(x) + shortcut(x))
        return x.mean(dim=-1)


class MultiScale2D(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(inputs, outputs, k, padding=k // 2) for k in (1, 3, 5)])

    def forward(self, x):
        return torch.stack([branch(x) for branch in self.branches]).mean(dim=0)


class PeriodBlock(nn.Module):
    """Sample-specific periods avoid inference dependence on other patients."""
    def __init__(self, width, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.mix = nn.Sequential(MultiScale2D(width, width * 2), nn.GELU(), MultiScale2D(width * 2, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        # B,C,T; FFT remains FP32 under AMP (T=300 is not a power of two).
        length = x.shape[-1]
        amplitude = torch.fft.rfft(x.float(), dim=-1).abs().mean(dim=1)
        # Exclude DC explicitly, including the all-zero input case.
        scores, frequency = amplitude[:, 1:].topk(min(self.top_k, amplitude.shape[1] - 1), dim=-1)
        periods = torch.div(length, frequency + 1, rounding_mode="floor")
        weights = scores.softmax(dim=-1)
        combined = torch.zeros_like(x)
        for period in periods.unique().tolist():
            selected = (periods == period)
            indices = selected.any(dim=1).nonzero(as_tuple=True)[0]
            subset = x.index_select(0, indices)
            padded = ((length + period - 1) // period) * period
            plane = F.pad(subset, (0, padded - length)).reshape(len(indices), x.shape[1], padded // period, period)
            transformed = self.mix(plane).reshape(len(indices), x.shape[1], padded)[..., :length]
            contribution = (weights * selected).sum(dim=-1).index_select(0, indices)
            # Index-copy into each selected sample; no reductions across patients.
            combined = combined.index_add(0, indices, (transformed * contribution[:, None, None]).to(x.dtype))
        return self.norm((x + combined).transpose(1, 2)).transpose(1, 2)


class TimesNetEncoder(nn.Module):
    def __init__(self, inputs, width, length, layers, dropout):
        super().__init__()
        self.embedding = nn.Conv1d(inputs, width, 3, padding=1)
        self.position = nn.Parameter(torch.zeros(1, width, length))
        nn.init.normal_(self.position, std=.02)
        self.blocks = nn.Sequential(*[PeriodBlock(width) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.embedding(x) + self.position)
        return F.gelu(self.blocks(x)).mean(dim=-1)
