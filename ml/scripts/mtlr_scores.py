"""MTLR risk from the entire fitted survival curve, without horizon-tail saturation."""
import numpy as np
import torch
import torch.nn.functional as F

SCORE_VERSION = "negative_rmst_step_v1"


def negative_rmst(survival, cuts):
    """Negative restricted mean lifetime over [0, cuts[-1]], discrete step S(t)."""
    survival = np.asarray(survival, dtype=np.float64)
    cuts = np.asarray(cuts, dtype=np.float64)
    if survival.ndim != 2 or cuts.ndim != 1 or len(cuts) != survival.shape[0]:
        raise ValueError("MTLR survival/cut shape mismatch")
    if len(cuts) < 2 or cuts[0] != 0 or not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
        raise ValueError("MTLR cuts must increase strictly from zero")
    if not np.isfinite(survival).all() or (survival < -1e-5).any() or (survival > 1+1e-5).any():
        raise ValueError("Invalid MTLR survival probabilities")
    if (np.diff(survival, axis=0) > 1e-5).any():
        raise ValueError("MTLR survival must be nonincreasing")
    return -np.sum(np.clip(survival[:-1], 0, 1) * np.diff(cuts)[:, None], axis=0)


def risk_from_logits(phi, cuts):
    """Stable MTLR survival sums; shared by calibration and the resident server."""
    if phi.ndim != 2 or phi.shape[1] != len(cuts) or not torch.isfinite(phi).all():
        raise ValueError("Invalid MTLR logits")
    reversed_sum = phi.flip(1).cumsum(1).flip(1)
    probabilities = F.softmax(F.pad(reversed_sum, [0, 1]), dim=1)
    survival = probabilities[:, 1:].flip(1).cumsum(1).flip(1)
    return negative_rmst(survival.detach().cpu().numpy().T, cuts)
