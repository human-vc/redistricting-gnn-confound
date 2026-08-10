from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, t as tdist

def rank_correlations(x: np.ndarray, y: np.ndarray) -> dict:
    rho, p_rho = spearmanr(x, y)
    tau, p_tau = kendalltau(x, y)
    return {"spearman_rho": float(rho), "spearman_p": float(p_rho), "kendall_tau": float(tau), "kendall_p": float(p_tau)}

def percentile_rank(ensemble_scores: np.ndarray, test_score: float) -> float:

    return 100.0 * float(np.mean(np.asarray(ensemble_scores) <= test_score))

@dataclass
class CFPResult:
    eps: float
    p_value: float
    rank_from_extreme: float
    n_total: int
    percentile: float

def cfp_epsilon_test(ensemble_scores: np.ndarray, test_score: float) -> CFPResult:

    combined = np.concatenate([np.asarray(ensemble_scores), [test_score]])
    n = len(combined)
    ranks = rankdata(combined, method="average")
    r = ranks[-1]
    eps = min(r, n - r + 1) / n
    p_value = min(1.0, 2 * eps)
    return CFPResult(eps=float(eps), p_value=float(p_value), rank_from_extreme=float(min(r, n - r + 1)), n_total=n, percentile=percentile_rank(ensemble_scores, test_score))

def paired_permutation_test(
    a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator, alternative: str = "greater"
) -> dict:

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return {"observed_diff": float("nan"), "p_value": float("nan"), "null_diffs": np.array([]), "n": 0}
    observed = float(np.mean(a) - np.mean(b))

    null_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        aa = np.where(swap, b, a)
        bb = np.where(swap, a, b)
        null_diffs[i] = np.mean(aa) - np.mean(bb)

    if alternative == "greater":
        p_value = (1 + np.sum(null_diffs >= observed)) / (n_perm + 1)
    elif alternative == "less":
        p_value = (1 + np.sum(null_diffs <= observed)) / (n_perm + 1)
    else:
        p_value = (1 + np.sum(np.abs(null_diffs) >= abs(observed))) / (n_perm + 1)

    return {"observed_diff": observed, "p_value": float(p_value), "null_diffs": null_diffs, "n": n}

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 2:
        return {"dm_stat": float("nan"), "dm_stat_adj": float("nan"), "p_value": float("nan"), "n": n, "mean_diff": float("nan")}
    dbar = d.mean()
    d_centered = d - dbar
    gamma0 = float(np.mean(d_centered ** 2))
    var_d = gamma0
    for lag in range(1, h):
        gamma_k = float(np.sum(d_centered[:-lag] * d_centered[lag:]) / n)
        var_d += 2 * (1 - lag / h) * gamma_k
    var_d = max(var_d, 1e-12)
    se = np.sqrt(var_d / n)
    dm_stat = dbar / se
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_adj = dm_stat * hln
    p_value = 2 * (1 - tdist.cdf(abs(dm_adj), df=n - 1))
    return {"dm_stat": float(dm_stat), "dm_stat_adj": float(dm_adj), "p_value": float(p_value), "n": n, "mean_diff": float(dbar)}

def bootstrap_ci(arrays: tuple, statistic_fn, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> dict:

    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    point = float(statistic_fn(*arrays))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic_fn(*[a[idx] for a in arrays])
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"), "boots": boots}
    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "boots": boots}
