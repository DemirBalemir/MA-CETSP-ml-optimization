"""Metrics conditional on logged solutions, not the complete live population."""
import numpy as np
from scipy.stats import norm, rankdata, spearmanr


def cohort_mask(birth, death):
    birth, death = np.asarray(birth), np.asarray(death)
    mask = (birth[None, :] <= birth[:, None]) & (death[None, :] > birth[:, None])
    np.fill_diagonal(mask, False)
    return mask


def cohort_rank(values, mask):
    values = np.asarray(values)
    counts = mask.sum(axis=1)
    ranks = ((values[None, :] < values[:, None]) & mask).sum(axis=1)
    return np.divide(ranks, counts, out=np.full(len(values), np.nan), where=counts >= 3)


def pairwise_concordance(predictor, target):
    """Exclude target ties, award half credit for predictor ties. Not tau-b."""
    predictor, target = np.asarray(predictor), np.asarray(target)
    valid = np.isfinite(predictor) & np.isfinite(target)
    predictor, target = predictor[valid], target[valid]
    i, j = np.triu_indices(len(target), 1)
    dx, dy = predictor[i] - predictor[j], target[i] - target[j]
    comparable = dy != 0
    if not comparable.any():
        return np.nan
    product = dx[comparable] * dy[comparable]
    return float(((product > 0).sum() + .5 * (dx[comparable] == 0).sum()) / comparable.sum())


def rank_gain_kappa(input_cohort_rank, relative_gain):
    x, g = np.asarray(input_cohort_rank), np.asarray(relative_gain)
    valid = np.isfinite(x) & np.isfinite(g)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(g[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], g[valid]).statistic)


def couple_to_kappa(base_rank, values, target, rng):
    """Copula reassignment; finite samples/ties require measuring realised kappa."""
    u = (rankdata(base_rank) - .5) / len(values)
    z_base = norm.ppf(u)
    rho = 2 * np.sin(np.pi * np.clip(target, -1, 1) / 6)
    z = rho * z_base + np.sqrt(max(0, 1 - rho ** 2)) * rng.standard_normal(len(values))
    out = np.empty(len(values))
    out[np.argsort(z)] = np.sort(values)
    return out
