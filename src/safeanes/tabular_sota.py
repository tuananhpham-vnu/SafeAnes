"""Official TabM backbone with a project-specific ordered two-horizon head."""
import copy
import hashlib

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from torch.nn import functional as F


def inner_stop_mask(subjects):
    """Fixed patient split inside FIT; never use calibration/validation/test here."""
    return np.array([int(hashlib.sha256(f"E06-stop:{s}".encode()).hexdigest()[:8], 16) % 5 == 0
                     for s in subjects])


def ordered_logits(logits):
    return torch.stack((logits[..., 0], logits[..., 0] + F.softplus(logits[..., 1])), dim=-1)


def ordered_probabilities(probability):
    """Euclidean projection onto p5 <= p10, after independent calibration."""
    p = np.asarray(probability, dtype=float).copy()
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite two-horizon probabilities")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    crossing = p[:, 0] > p[:, 1]
    p[crossing] = p[crossing].mean(axis=1, keepdims=True)
    return p


class TabMRisk:
    """CPU-sized TabM + PLE. Hyperparameters are E06 choices, not paper replication."""
    def __init__(self, seed=20260917, epochs=30, patience=6, k=16, width=64, batch_size=512):
        self.seed, self.epochs, self.patience = seed, epochs, patience
        self.k, self.width, self.batch_size = k, width, batch_size

    def transform(self, x):
        a = self.imputer.transform(np.asarray(x, dtype=float))[:, self.keep]
        return torch.as_tensor(self.scaler.transform(a), dtype=torch.float32)

    def fit(self, x, y, subjects):
        from tabm import TabM
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        x, y = np.asarray(x, float), np.asarray(y, np.float32)
        stop = inner_stop_mask(subjects)
        if not stop.any() or stop.all():
            raise ValueError("Need patients on both sides of the inner split")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x[~stop])
        a = self.imputer.transform(x[~stop])
        self.keep = np.ptp(a, axis=0) > 1e-10
        if not self.keep.any():
            raise ValueError("No nonconstant training features")
        self.scaler = QuantileTransformer(n_quantiles=min(256, len(a)), output_distribution="normal",
                                          random_state=self.seed).fit(a[:, self.keep])
        xx, yy = self.transform(x), torch.from_numpy(y)
        bins = compute_bins(xx[~stop], n_bins=8)
        self.model = TabM.make(n_num_features=xx.shape[1], d_out=2, k=self.k,
            n_blocks=2, d_block=self.width, dropout=.1,
            num_embeddings=PiecewiseLinearEmbeddings(bins, 4, activation=False, version="B"))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.002, weight_decay=.0003)
        train_indices = torch.from_numpy(np.flatnonzero(~stop))
        stop_indices = torch.from_numpy(np.flatnonzero(stop))
        generator = torch.Generator().manual_seed(self.seed)
        best, stale, self.history = float("inf"), 0, []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            order = train_indices[torch.randperm(len(train_indices), generator=generator)]
            for ids in order.split(self.batch_size):
                logits = ordered_logits(self.model(xx[ids]))
                target = yy[ids, None, :].expand_as(logits)
                known = target >= 0
                if not known.any():
                    continue
                loss = F.binary_cross_entropy_with_logits(logits[known], target[known])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                prob = torch.cat([ordered_logits(self.model(xx[ids])).sigmoid().mean(1)
                                  for ids in stop_indices.split(self.batch_size)])
                target = yy[stop_indices]
                known = target >= 0
                score = F.binary_cross_entropy(prob[known].clamp(1e-6, 1-1e-6), target[known]).item()
            self.history.append({"epoch": epoch, "inner_patient_bce": score})
            if score < best - 1e-5:
                best, stale, self.best_epoch = score, 0, epoch
                state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
            print(f"TabM seed={self.seed} epoch={epoch} stop_BCE={score:.5f}", flush=True)
            if stale >= self.patience:
                break
        self.model.load_state_dict(state)
        self.model.eval()
        self.inner_subjects = {"optimization": sorted(set(np.asarray(subjects)[~stop].tolist())),
                               "stopping": sorted(set(np.asarray(subjects)[stop].tolist()))}
        return self

    def predict_risk(self, x):
        xx = self.transform(x)
        self.model.eval()
        if len(xx) == 0:
            return np.empty((0, 2))
        with torch.no_grad():
            return torch.cat([ordered_logits(self.model(batch)).sigmoid().mean(1)
                              for batch in xx.split(self.batch_size)]).numpy()


def log_odds(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p))


def calibrated_risk(bundle, frame):
    x = frame[bundle["features"]]
    if bundle["kind"] == "tabm":
        raw = bundle["model"].predict_risk(x)
    else:
        raw = np.column_stack([m.predict_proba(x)[:, 1] for m in bundle["models"]])
    p = np.column_stack([cal.predict_proba(log_odds(raw[:, i]).reshape(-1, 1))[:, 1]
                         for i, cal in enumerate(bundle["calibrators"])])
    return ordered_probabilities(p)
