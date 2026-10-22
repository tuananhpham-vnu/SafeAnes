"""Redundancy among inputs: pairwise Spearman with co-observation counts, multiple R^2 of each input from all the
others, and near-duplicate groups (complete linkage: every pair |rho| >= T) with the member to keep."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def spearman(X: pd.DataFrame, cols: Sequence[str], min_periods: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rho, n_both): pairwise-complete Spearman (NaN below min_periods rows seen together) and the number of rows
    where both inputs are observed."""
    cols = list(cols)
    rho = X[cols].rank().corr(min_periods=min_periods)
    obs = X[cols].notna().to_numpy(np.float32)
    return rho, pd.DataFrame(obs.T @ obs, index=cols, columns=cols).round().astype(np.int64)


def multiple_r2(X: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """R^2 of each input regressed linearly on all the others, on percentile ranks with missing set to the middle
    (0.5): 1 - 1 / diag(C^-1). Catches inputs that are a combination of several others (pp = sbp - dbp, ...),
    which no single pair shows. NaN for constant inputs."""
    cols = list(cols)
    Z = X[cols].rank(pct=True).fillna(0.5)
    sd = Z.std().to_numpy()
    use = [c for c, s in zip(cols, sd) if s > 1e-9]
    C = np.corrcoef(Z[use].to_numpy(np.float64), rowvar=False)
    d = np.diag(np.linalg.pinv(C))
    r2 = pd.Series(np.clip(1 - 1 / np.maximum(d, 1.0), 0, 1), index=use)
    return r2.reindex(cols)


def _distance(rho: pd.DataFrame) -> np.ndarray:
    d = 1 - rho.abs().fillna(0).to_numpy()
    np.fill_diagonal(d, 0)
    return np.clip((d + d.T) / 2, 0, 1)


def order(rho: pd.DataFrame) -> list[str]:
    """Inputs in average-linkage dendrogram order of 1 - |rho| (for the heat map)."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    z = linkage(squareform(_distance(rho), checks=False), "average")
    return [rho.columns[i] for i in leaves_list(z)]


def near_dup_groups(rho: pd.DataFrame, X: pd.DataFrame, T: float, priority: Callable[[str], tuple],
                    extra_cover: float = 0.05) -> pd.DataFrame:
    """Groups where every pair has |rho| >= T (complete linkage on 1 - |rho|; a pair seen together on too few rows
    has rho NaN and never joins). The member with the largest priority(feature) is kept. Pairwise-complete rho
    ignores missingness, so another member is kept too when it is observed on >= extra_cover of the rows where the
    kept one is missing (it then covers rows the kept one cannot). One row per member of a group of >= 2."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    cols = list(rho.columns)
    lab = fcluster(linkage(squareform(_distance(rho), checks=False), "complete"), 1 - T, "distance")
    rows = []
    for gid, k in enumerate(sorted(np.unique(lab), key=lambda k: -int((lab == k).sum())), start=1):
        fs = [c for c, g in zip(cols, lab) if g == k]
        if len(fs) < 2:
            continue
        keep = max(fs, key=priority)
        miss = X[keep].isna().to_numpy()
        tri = rho.loc[fs, fs].abs().to_numpy()[np.triu_indices(len(fs), 1)]
        for f in fs:
            cover = float((miss & X[f].notna().to_numpy()).mean()) if f != keep else 0.0
            role = "giữ" if f == keep else ("phủ thêm dòng" if cover >= extra_cover else "trùng")
            rows.append({"group_id": gid, "n_group": len(fs), "feature": f, "keep": role != "trùng", "role": role,
                         "keep_feature": keep, "rho_with_kept": float(rho.loc[f, keep]) if f != keep else 1.0,
                         "min_abs_rho_in_group": float(np.nanmin(tri)), "cover_extra": cover,
                         "missing": float(X[f].isna().mean())})
    cols_out = ["group_id", "n_group", "feature", "keep", "role", "keep_feature", "rho_with_kept",
                "min_abs_rho_in_group", "cover_extra", "missing"]
    return pd.DataFrame(rows, columns=cols_out)


def pairs(rho: pd.DataFrame, n_both: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Every pair with |rho| >= threshold, strongest first."""
    a = rho.abs().to_numpy()
    i, j = np.triu_indices(len(a), 1)
    m = np.nan_to_num(a[i, j]) >= threshold
    i, j = i[m], j[m]
    c = rho.columns
    out = pd.DataFrame({"feature_a": c[i], "feature_b": c[j], "rho": rho.to_numpy()[i, j],
                        "n_both": n_both.to_numpy()[i, j]})
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def block_mean(rho: pd.DataFrame, family: Callable[[str], str], families: Sequence[str]) -> pd.DataFrame:
    """Mean |rho| between every two families (diagonal: between different members of one family)."""
    fam = np.array([family(c) for c in rho.columns])
    a = rho.abs().to_numpy().copy()
    np.fill_diagonal(a, np.nan)
    out = pd.DataFrame(index=list(families), columns=list(families), dtype=float)
    for f in families:
        for g in families:
            b = a[np.ix_(fam == f, fam == g)]
            out.loc[f, g] = float(np.nanmean(b)) if np.isfinite(b).any() else np.nan
    return out
