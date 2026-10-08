"""TabM on the UC04 tabular columns, one model for the 5 horizons (E06 TabM adapted to plan section 6).

Differences from E06 (`safeanes.tabular_sota.TabMRisk`): 5 ordered horizons with the DL head rule
(logit_k = logit_{k-1} + softplus(a_k)), masked BCE on known labels of eligible rows, GPU when
available. Kept from E06: TabM k=16, 2 blocks of width 64, dropout 0.1, piecewise-linear embeddings
(8 bins, d=4), median imputation + quantile-normal scaling fitted on the optimisation patients,
AdamW lr 2e-3 wd 3e-4, early stopping on a fixed 1/5 of train patients (hash of subjectid).
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

HORIZONS = (300, 600, 900, 1200, 1800)


def stop_mask(subjects: np.ndarray) -> np.ndarray:
    """Fixed patient split inside train for early stopping (as E06, different salt)."""
    return np.array([int(hashlib.sha256(f"uc04-tabm-stop:{s}".encode()).hexdigest()[:8], 16) % 5 == 0
                     for s in subjects])


def ordered(a: torch.Tensor) -> torch.Tensor:
    """[..., 5] raw outputs -> logits that increase with the horizon."""
    return torch.cumsum(torch.cat([a[..., :1], F.softplus(a[..., 1:])], -1), -1)


class TabMMulti:
    def __init__(self, columns, seed: int, epochs: int = 30, patience: int = 6, k: int = 16, width: int = 64,
                 batch_size: int = 512, device: str = "auto", log=print):
        self.columns, self.seed, self.epochs, self.patience = list(columns), seed, epochs, patience
        self.k, self.width, self.batch_size, self.log = k, width, batch_size, log
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def _x(self, X) -> torch.Tensor:
        a = self.imputer.transform(X[self.columns].to_numpy(float))[:, self.keep]
        return torch.as_tensor(self.scaler.transform(a), dtype=torch.float32)

    def fit(self, X, Y: np.ndarray, subjects: np.ndarray):
        """Y [n, 5] in {-1, 0, 1} (-1 ignored)."""
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import QuantileTransformer
        from tabm import TabM
        torch.manual_seed(self.seed)
        stop = stop_mask(subjects)
        raw = X[self.columns].to_numpy(float)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(raw[~stop])
        a = self.imputer.transform(raw[~stop])
        self.keep = np.ptp(a, axis=0) > 1e-10
        self.scaler = QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=200_000,
                                          random_state=self.seed % (2 ** 32)).fit(a[:, self.keep])
        dev = self.device
        xx = self._x(X).to(dev)
        yy = torch.as_tensor(np.asarray(Y, np.float32), device=dev)
        bins = compute_bins(xx[torch.from_numpy(~stop).to(dev)].cpu(), n_bins=8)
        self.model = TabM.make(n_num_features=xx.shape[1], d_out=len(HORIZONS), k=self.k, n_blocks=2,
                               d_block=self.width, dropout=0.1,
                               num_embeddings=PiecewiseLinearEmbeddings(bins, 4, activation=False, version="B")).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=3e-4)
        tr = torch.from_numpy(np.flatnonzero(~stop)).to(dev)
        st = torch.from_numpy(np.flatnonzero(stop)).to(dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, stale, state, self.history = float("inf"), 0, None, []
        for epoch in range(self.epochs):
            self.model.train()
            for ids in tr[torch.randperm(len(tr), generator=g).to(dev)].split(self.batch_size):
                logits = ordered(self.model(xx[ids]))                 # [B, k, 5]
                y = yy[ids, None, :].expand_as(logits)
                known = y >= 0
                loss = F.binary_cross_entropy_with_logits(logits[known], y[known])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            score = self._stop_bce(xx, yy, st)
            self.history.append({"epoch": epoch, "stop_bce": score})
            improved = score < best - 1e-5
            if improved:
                best, stale, self.best_epoch, state = score, 0, epoch, copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
            self.log(f"    TabM seed {self.seed} epoch {epoch}: stop BCE {score:.5f}{' *' if improved else ''}")
            if stale >= self.patience:
                break
        self.model.load_state_dict(state)
        self.model.eval()
        return self

    @torch.no_grad()
    def _stop_bce(self, xx, yy, idx) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for ids in idx.split(8192):
            p = ordered(self.model(xx[ids])).sigmoid().mean(1)       # mean probability over the k members
            y = yy[ids]
            known = y >= 0
            tot += F.binary_cross_entropy(p[known].clamp(1e-6, 1 - 1e-6), y[known], reduction="sum").item()
            n += int(known.sum())
        return tot / max(n, 1)

    @torch.no_grad()
    def logits(self, X) -> np.ndarray:
        """[n, 5] logit of the mean member probability."""
        self.model.eval()
        xx = self._x(X)
        out = [ordered(self.model(b.to(self.device))).sigmoid().mean(1).cpu() for b in xx.split(8192)]
        p = torch.cat(out).numpy().astype(np.float64) if out else np.zeros((0, len(HORIZONS)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
