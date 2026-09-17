"""Small numeric models; project adaptations, not reproduced paper weights."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, (self.dilation[0] * (self.kernel_size[0] - 1), 0)))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, channels, dilation, dropout):
        super().__init__()
        self.path = nn.Sequential(CausalConv(inputs, channels, 3, dilation=dilation), nn.ReLU(),
            nn.Dropout(dropout), CausalConv(channels, channels, 3, dilation=dilation),
            nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(inputs, channels, 1) if inputs != channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.path(x) + self.skip(x))


class NumericModel(nn.Module):
    def __init__(self, architecture, inputs, static_dim, length=300, width=32,
                 dropout=.1, patch=10, layers=2):
        super().__init__()
        self.architecture = architecture
        if architecture == "tcn":
            self.encoder = nn.Sequential(*[ResidualBlock(inputs if i == 0 else width, width, 2**i, dropout)
                                          for i in range(7)])
        elif architecture == "transformer":
            if length % patch or width % 4:
                raise ValueError("Patch must divide history; width must be divisible by 4")
            self.patch = nn.Conv1d(inputs, width, patch, stride=patch)
            self.position = nn.Parameter(torch.zeros(1, length // patch, width))
            nn.init.normal_(self.position, std=.02)
            block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout,
                                                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            # Independent initial draws rather than cloned layer parameters.
            for layer in self.encoder.layers:
                for p in layer.parameters():
                    if p.ndim > 1:
                        nn.init.xavier_uniform_(p)
        else:
            raise ValueError("Expected tcn or transformer")
        self.head = nn.Sequential(nn.Linear(width + static_dim, width), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, x, static):
        if self.architecture == "tcn":
            encoded = self.encoder(x)[:, :, -1]
        else:
            encoded = self.encoder(self.patch(x).transpose(1, 2) + self.position).mean(dim=1)
        logits = self.head(torch.cat([encoded, static], dim=1)).float()
        # Ordered logits enforce p10 >= p5 without clipping gradients through p.
        return torch.stack([logits[:, 0], logits[:, 0] + F.softplus(logits[:, 1])], dim=1)


def masked_bce(logits, target):
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])
